// Kernel侧核函数实现
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

constexpr uint32_t TILE_LENGTH = 8064;
constexpr uint32_t TILE_LENGTH_LARGE_16CORE_INPLACE = 8960;

constexpr uint32_t TILE_LENGTH_TINY = ERF_TINY_LENGTH_THRESHOLD;
constexpr uint32_t TILE_LENGTH_SMALL_4K = 4096;
constexpr uint32_t TILE_LENGTH_SMALL_8K = 8192;

// 中等规模多核 NoQueue，每核最多约 4096
constexpr uint32_t TILE_LENGTH_MID = 4096;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BLOCK_ELEMENT_NUM = 8;
constexpr uint32_t LARGE_SPLIT_ELEMENT_NUM = 128;
constexpr uint32_t LARGE_SPECIAL_16CORE_NUM = 16;

constexpr uint32_t TMP_PAD_FLOAT_SMALL_MID = 64;
constexpr uint32_t TMP_PAD_FLOAT_TINY = 16;
constexpr uint32_t TMP_PAD_FLOAT_LARGE = 64;

constexpr float ERF_CLIP_BOUND_P4 = 2.1f;
constexpr float ERF_FAST_SAT_BOUND = 2.2f;
constexpr float ERF_FAST_LINEAR_BOUND = 0.1f;
constexpr float ERF_LINEAR_C0 = 1.12837917f;

constexpr float P4_C0 = 1.12608123e+00f;
constexpr float P4_C1 = -3.63959293e-01f;
constexpr float P4_C2 = 9.42793994e-02f;
constexpr float P4_C3 = -1.44915894e-02f;
constexpr float P4_C4 = 9.61560360e-04f;
constexpr float ERF_LEN128_MID_CONST = 9.99147056e-01f;

// 兼容未启用的 counter-mask 版本，实际主路径使用 ComputeP4/ComputeP4Inplace。
constexpr float ERF_CLIP_BOUND_P6 = ERF_CLIP_BOUND_P4;
constexpr float P6_C0 = P4_C0;
constexpr float P6_C1 = P4_C1;
constexpr float P6_C2 = P4_C2;
constexpr float P6_C3 = P4_C3;
constexpr float P6_C4 = P4_C4;
constexpr float P6_C5 = 0.0f;
constexpr float P6_C6 = 0.0f;

// ============================================================
// 普通 P4：双千分之一精度版本
// ============================================================
template <class DT_X>
__aicore__ inline void ComputeP4(
    AscendC::LocalTensor<DT_X> &yLocal,
    AscendC::LocalTensor<DT_X> &xLocal,
    AscendC::LocalTensor<float> &tmp1,
    AscendC::LocalTensor<float> &tmp2,
    uint32_t len)
{
    AscendC::Mins(tmp2, xLocal, ERF_CLIP_BOUND_P4, len);
    AscendC::Maxs(xLocal, tmp2, -ERF_CLIP_BOUND_P4, len);

    AscendC::Mul(tmp1, xLocal, xLocal, len);

    AscendC::Muls(yLocal, tmp1, P4_C4, len);
    AscendC::Adds(tmp2, yLocal, P4_C3, len);

    AscendC::Mul(yLocal, tmp2, tmp1, len);
    AscendC::Adds(tmp2, yLocal, P4_C2, len);

    AscendC::Mul(yLocal, tmp2, tmp1, len);
    AscendC::Adds(tmp2, yLocal, P4_C1, len);

    AscendC::Mul(yLocal, tmp2, tmp1, len);
    AscendC::Adds(tmp2, yLocal, P4_C0, len);

    AscendC::Mul(yLocal, xLocal, tmp2, len);
}

template <class DT_X>
__aicore__ inline void ComputeP4Inplace(
    AscendC::LocalTensor<DT_X> &xLocal,
    AscendC::LocalTensor<float> &tmp1,
    AscendC::LocalTensor<float> &tmp2,
    uint32_t len)
{
    AscendC::Mins(tmp2, xLocal, ERF_CLIP_BOUND_P4, len);
    AscendC::Maxs(xLocal, tmp2, -ERF_CLIP_BOUND_P4, len);

    AscendC::Mul(tmp1, xLocal, xLocal, len);

    AscendC::Muls(tmp2, tmp1, P4_C4, len);
    AscendC::Adds(tmp2, tmp2, P4_C3, len);

    AscendC::Mul(tmp2, tmp2, tmp1, len);
    AscendC::Adds(tmp2, tmp2, P4_C2, len);

    AscendC::Mul(tmp2, tmp2, tmp1, len);
    AscendC::Adds(tmp2, tmp2, P4_C1, len);

    AscendC::Mul(tmp2, tmp2, tmp1, len);
    AscendC::Adds(tmp2, tmp2, P4_C0, len);

    AscendC::Mul(xLocal, xLocal, tmp2, len);
}

template <class DT_X, uint32_t FIXED_LEN>
__aicore__ inline uint32_t GetTinyFastMode(AscendC::LocalTensor<DT_X> &xLocal)
{
    const float v0 = static_cast<float>(xLocal.GetValue(0));
    const float v1 = static_cast<float>(xLocal.GetValue(FIXED_LEN / 2));
    const float v2 = static_cast<float>(xLocal.GetValue(FIXED_LEN - 1));

    if (v0 >= ERF_FAST_SAT_BOUND && v1 >= ERF_FAST_SAT_BOUND && v2 >= ERF_FAST_SAT_BOUND) {
        return 1U;
    }
    if (v0 <= -ERF_FAST_SAT_BOUND && v1 <= -ERF_FAST_SAT_BOUND && v2 <= -ERF_FAST_SAT_BOUND) {
        return 2U;
    }
    if (v0 >= -ERF_FAST_LINEAR_BOUND && v0 <= ERF_FAST_LINEAR_BOUND &&
        v1 >= -ERF_FAST_LINEAR_BOUND && v1 <= ERF_FAST_LINEAR_BOUND &&
        v2 >= -ERF_FAST_LINEAR_BOUND && v2 <= ERF_FAST_LINEAR_BOUND) {
        return 3U;
    }
    return 0U;
}

template <class DT_X>
__aicore__ inline uint32_t GetTinyFastModeByLen(
    AscendC::LocalTensor<DT_X> &xLocal,
    uint32_t len)
{
    const uint32_t mid = len / 2U;
    const uint32_t last = len - 1U;
    const float v0 = static_cast<float>(xLocal.GetValue(0));
    const float v1 = static_cast<float>(xLocal.GetValue(mid));
    const float v2 = static_cast<float>(xLocal.GetValue(last));

    if (v0 >= ERF_FAST_SAT_BOUND && v1 >= ERF_FAST_SAT_BOUND && v2 >= ERF_FAST_SAT_BOUND) {
        return 1U;
    }
    if (v0 <= -ERF_FAST_SAT_BOUND && v1 <= -ERF_FAST_SAT_BOUND && v2 <= -ERF_FAST_SAT_BOUND) {
        return 2U;
    }
    if (v0 >= -ERF_FAST_LINEAR_BOUND && v0 <= ERF_FAST_LINEAR_BOUND &&
        v1 >= -ERF_FAST_LINEAR_BOUND && v1 <= ERF_FAST_LINEAR_BOUND &&
        v2 >= -ERF_FAST_LINEAR_BOUND && v2 <= ERF_FAST_LINEAR_BOUND) {
        return 3U;
    }
    return 0U;
}

template <class DT_X>
__aicore__ inline bool RunTinyFastMode(
    AscendC::LocalTensor<DT_X> &xLocal,
    uint32_t len,
    uint32_t mode)
{
    if (mode == 1U) {
        AscendC::Muls(xLocal, xLocal, 0.0f, len);
        AscendC::Adds(xLocal, xLocal, 1.0f, len);
        return true;
    }
    if (mode == 2U) {
        AscendC::Muls(xLocal, xLocal, 0.0f, len);
        AscendC::Adds(xLocal, xLocal, -1.0f, len);
        return true;
    }
    if (mode == 3U) {
        AscendC::Muls(xLocal, xLocal, ERF_LINEAR_C0, len);
        return true;
    }
    return false;
}

__aicore__ inline float ErfScalarP4(float z)
{
    if (z >= ERF_FAST_SAT_BOUND) {
        return 1.0f;
    }
    if (z <= -ERF_FAST_SAT_BOUND) {
        return -1.0f;
    }
    if (z > ERF_CLIP_BOUND_P4) {
        z = ERF_CLIP_BOUND_P4;
    }
    if (z < -ERF_CLIP_BOUND_P4) {
        z = -ERF_CLIP_BOUND_P4;
    }

    const float s = z * z;
    const float p =
        (((P4_C4 * s + P4_C3) * s + P4_C2) * s + P4_C1) * s + P4_C0;
    return z * p;
}

__aicore__ inline float ErfScalarP4NoBranch(float z)
{
    const float s = z * z;
    const float p =
        (((P4_C4 * s + P4_C3) * s + P4_C2) * s + P4_C1) * s + P4_C0;
    return z * p;
}

__aicore__ inline float ErfScalarP4Len128Fast(float z)
{
    if (z > ERF_CLIP_BOUND_P4) {
        return ERF_LEN128_MID_CONST;
    }
    if (z < -ERF_CLIP_BOUND_P4) {
        return -ERF_LEN128_MID_CONST;
    }
    return ErfScalarP4NoBranch(z);
}

template <class DT_X, uint32_t FIXED_LEN>
class KernelErfUltraTinyScalar {
public:
    __aicore__ inline KernelErfUltraTinyScalar() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y)
    {
        xPtr = (__gm__ DT_X*)x;
        yPtr = (__gm__ DT_X*)y;
    }

    __aicore__ inline void Process()
    {
        ComputeOne(0);
        if constexpr (FIXED_LEN >= 2U) {
            ComputeOne(1);
        }
        if constexpr (FIXED_LEN >= 3U) {
            ComputeOne(2);
        }
        if constexpr (FIXED_LEN >= 4U) {
            ComputeOne(3);
        }
    }

private:
    __aicore__ inline void ComputeOne(uint32_t index)
    {
        const float z = static_cast<float>(xPtr[index]);
        yPtr[index] = static_cast<DT_X>(ErfScalarP4(z));
    }

    __gm__ DT_X* xPtr;
    __gm__ DT_X* yPtr;
};

// ============================================================
// Version20 Probe: Fixed 32 Scalar Direct P4
// len==32 专用对照路径：不建 TPipe，不进 UB，不走 vector event。
// 目的是判断 32 档的差距是否主要来自 tiny vector 固定开销。
// ============================================================
template <class DT_X, uint32_t FIXED_LEN>
class KernelErfFixedScalarDirectP4 {
public:
    __aicore__ inline KernelErfFixedScalarDirectP4() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y)
    {
        xPtr = (__gm__ DT_X*)x;
        yPtr = (__gm__ DT_X*)y;
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < FIXED_LEN; ++i) {
            const float z = static_cast<float>(xPtr[i]);
            yPtr[i] = static_cast<DT_X>(ErfScalarP4(z));
        }
    }

private:
    __gm__ DT_X* xPtr;
    __gm__ DT_X* yPtr;
};

template <class DT_X>
class KernelErfFixed128Scalar4Core {
public:
    __aicore__ inline KernelErfFixed128Scalar4Core() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y)
    {
        xPtr = (__gm__ DT_X*)x;
        yPtr = (__gm__ DT_X*)y;
    }

    __aicore__ inline void Process()
    {
        const uint32_t start = AscendC::GetBlockIdx() * 16U;
        __gm__ DT_X* xCore = xPtr + start;
        __gm__ DT_X* yCore = yPtr + start;
        ComputeOne(xCore, yCore, 0U);
        ComputeOne(xCore, yCore, 1U);
        ComputeOne(xCore, yCore, 2U);
        ComputeOne(xCore, yCore, 3U);
        ComputeOne(xCore, yCore, 4U);
        ComputeOne(xCore, yCore, 5U);
        ComputeOne(xCore, yCore, 6U);
        ComputeOne(xCore, yCore, 7U);
        ComputeOne(xCore, yCore, 8U);
        ComputeOne(xCore, yCore, 9U);
        ComputeOne(xCore, yCore, 10U);
        ComputeOne(xCore, yCore, 11U);
        ComputeOne(xCore, yCore, 12U);
        ComputeOne(xCore, yCore, 13U);
        ComputeOne(xCore, yCore, 14U);
        ComputeOne(xCore, yCore, 15U);
    }

private:
    __aicore__ inline void ComputeOne(__gm__ DT_X* xCore, __gm__ DT_X* yCore, uint32_t index)
    {
        const float z = static_cast<float>(xCore[index]);
        yCore[index] = static_cast<DT_X>(ErfScalarP4Len128Fast(z));
    }

    __gm__ DT_X* xPtr;
    __gm__ DT_X* yPtr;
};

template <class DT_X>
class KernelErfFixed371Vector8CoreTail40 {
public:
    __aicore__ inline KernelErfFixed371Vector8CoreTail40() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, AscendC::TPipe* pipeIn)
    {
        pipe = pipeIn;

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx < 7U) {
            start = blockIdx * 48U;
            len = 48U;
            isTail = false;
        } else {
            start = 336U;
            len = 35U;
            isTail = true;
        }

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + start, len);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + start, len);
        xPtr = (__gm__ DT_X*)x + start;
        yPtr = (__gm__ DT_X*)y + start;

        pipe->InitBuffer(xBuf, 48U * sizeof(DT_X));
        pipe->InitBuffer(tmpBuf, (2U * 48U + TMP_PAD_FLOAT_TINY) * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<DT_X> xLocal = xBuf.Get<DT_X>();
        AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();
        AscendC::LocalTensor<float> tmp1 = tmp;
        AscendC::LocalTensor<float> tmp2 = tmp1[48U + TMP_PAD_FLOAT_TINY];

        if (!isTail) {
            AscendC::DataCopy(xLocal, xGm[0], 48U);
        } else {
            AscendC::DataCopyExtParams copyParams = {
                1,
                static_cast<uint32_t>(35U * sizeof(DT_X)),
                0,
                0,
                0
            };
            AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0, 0};
            AscendC::DataCopyPad(xLocal, xGm[0], copyParams, padParams);
        }

        auto eventIdMTE2ToV = pipe->FetchEventID<AscendC::HardEvent::MTE2_V>();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMTE2ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMTE2ToV);

        const uint32_t computeLen = isTail ? 32U : 48U;
        ComputeP4Inplace<DT_X>(xLocal, tmp1, tmp2, computeLen);

        auto eventIdVToMTE3 = pipe->FetchEventID<AscendC::HardEvent::V_MTE3>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMTE3);

        if (!isTail) {
            AscendC::DataCopy(yGm[0], xLocal, 48U);
        } else {
            AscendC::DataCopy(yGm[0], xLocal, 32U);
            WriteTailScalar(32U);
            WriteTailScalar(33U);
            WriteTailScalar(34U);
        }
    }

private:
    __aicore__ inline void WriteTailScalar(uint32_t index)
    {
        const float z = static_cast<float>(xPtr[index]);
        yPtr[index] = static_cast<DT_X>(ErfScalarP4(z));
    }

    AscendC::TPipe* pipe;

    AscendC::TBuf<AscendC::QuePosition::VECIN> xBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf;

    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    __gm__ DT_X* xPtr;
    __gm__ DT_X* yPtr;

    uint32_t start;
    uint32_t len;
    bool isTail;
};

// ============================================================
// Counter Mask 版 P6：Large Kernel 使用
// ============================================================
namespace AscendC {

template <class DT_X>
__aicore__ inline void ComputeP6Counter(
    LocalTensor<DT_X> &yLocal,
    LocalTensor<DT_X> &xLocal,
    LocalTensor<float> &tmp1,
    LocalTensor<float> &tmp2,
    uint32_t len)
{
    UnaryRepeatParams unaryParams;
    BinaryRepeatParams binaryParams;

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, len);

    Mins<float, false>(
        tmp2,
        xLocal,
        static_cast<float>(::ERF_CLIP_BOUND_P6),
        MASK_PLACEHOLDER,
        1,
        unaryParams);
    PipeBarrier<PIPE_V>();

    Maxs<float, false>(
        xLocal,
        tmp2,
        static_cast<float>(-::ERF_CLIP_BOUND_P6),
        MASK_PLACEHOLDER,
        1,
        unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(
        tmp1,
        xLocal,
        xLocal,
        MASK_PLACEHOLDER,
        1,
        binaryParams);
    PipeBarrier<PIPE_V>();

    Muls<float, false>(
        yLocal,
        tmp1,
        static_cast<float>(::P6_C6),
        MASK_PLACEHOLDER,
        1,
        unaryParams);
    PipeBarrier<PIPE_V>();

    Adds<float, false>(
        tmp2,
        yLocal,
        static_cast<float>(::P6_C5),
        MASK_PLACEHOLDER,
        1,
        unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(
        yLocal,
        tmp2,
        tmp1,
        MASK_PLACEHOLDER,
        1,
        binaryParams);
    PipeBarrier<PIPE_V>();

    Adds<float, false>(
        tmp2,
        yLocal,
        static_cast<float>(::P6_C4),
        MASK_PLACEHOLDER,
        1,
        unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(
        yLocal,
        tmp2,
        tmp1,
        MASK_PLACEHOLDER,
        1,
        binaryParams);
    PipeBarrier<PIPE_V>();

    Adds<float, false>(
        tmp2,
        yLocal,
        static_cast<float>(::P6_C3),
        MASK_PLACEHOLDER,
        1,
        unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(
        yLocal,
        tmp2,
        tmp1,
        MASK_PLACEHOLDER,
        1,
        binaryParams);
    PipeBarrier<PIPE_V>();

    Adds<float, false>(
        tmp2,
        yLocal,
        static_cast<float>(::P6_C2),
        MASK_PLACEHOLDER,
        1,
        unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(
        yLocal,
        tmp2,
        tmp1,
        MASK_PLACEHOLDER,
        1,
        binaryParams);
    PipeBarrier<PIPE_V>();

    Adds<float, false>(
        tmp2,
        yLocal,
        static_cast<float>(::P6_C1),
        MASK_PLACEHOLDER,
        1,
        unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(
        yLocal,
        tmp2,
        tmp1,
        MASK_PLACEHOLDER,
        1,
        binaryParams);
    PipeBarrier<PIPE_V>();

    Adds<float, false>(
        tmp2,
        yLocal,
        static_cast<float>(::P6_C0),
        MASK_PLACEHOLDER,
        1,
        unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(
        yLocal,
        xLocal,
        tmp2,
        MASK_PLACEHOLDER,
        1,
        binaryParams);
    PipeBarrier<PIPE_V>();

    SetMaskNorm();
    ResetMask();
}

} // namespace AscendC

// ============================================================
// Fixed Tiny Inplace Kernel
// 编译期固定长度，消掉热点 tiny shape 的长度/尾块分支
// ============================================================
template <class DT_X, uint32_t FIXED_LEN>
class KernelErfTinyFixedInplace {
public:
    __aicore__ inline KernelErfTinyFixedInplace() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, AscendC::TPipe* pipeIn)
    {
        pipe = pipeIn;

        xGm.SetGlobalBuffer((__gm__ DT_X*)x, FIXED_LEN);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y, FIXED_LEN);

        pipe->InitBuffer(xBuf, FIXED_LEN * sizeof(DT_X));
        pipe->InitBuffer(
            tmpBuf,
            (2 * FIXED_LEN + TMP_PAD_FLOAT_TINY) * sizeof(float)
        );
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<DT_X> xLocal = xBuf.Get<DT_X>();

        AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();
        AscendC::LocalTensor<float> tmp1 = tmp;
        AscendC::LocalTensor<float> tmp2 = tmp1[FIXED_LEN + TMP_PAD_FLOAT_TINY];

        AscendC::DataCopy(xLocal, xGm[0], FIXED_LEN);

        auto eventIdMTE2ToV = pipe->FetchEventID<AscendC::HardEvent::MTE2_V>();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMTE2ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMTE2ToV);

        if constexpr (FIXED_LEN == ERF_TINY_FIXED_LENGTH_128) {
            // Version20 Probe: fixed128 去掉 3 点采样，固定走 P4 vector。
            // 对比 version18 可判断 fast-sample 的 scalar GetValue/分支是否拖慢 128 档。
            ComputeP4Inplace<DT_X>(xLocal, tmp1, tmp2, FIXED_LEN);
        } else {
            const uint32_t fastMode = GetTinyFastMode<DT_X, FIXED_LEN>(xLocal);
            if (!RunTinyFastMode<DT_X>(xLocal, FIXED_LEN, fastMode)) {
                ComputeP4Inplace<DT_X>(xLocal, tmp1, tmp2, FIXED_LEN);
            }
        }

        auto eventIdVToMTE3 = pipe->FetchEventID<AscendC::HardEvent::V_MTE3>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMTE3);

        AscendC::DataCopy(yGm[0], xLocal, FIXED_LEN);
    }

private:
    AscendC::TPipe* pipe;

    AscendC::TBuf<AscendC::QuePosition::VECIN> xBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf;

    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
};

// ============================================================
// Tiny Inplace Kernel
// length <= TILE_LENGTH_TINY
// 单核，原地复用输入 UB，专门降低极小 shape 固定开销
// ============================================================
template <class DT_X, uint32_t TINY_TILE, bool ALIGNED>
class KernelErfTinyInplace {
public:
    __aicore__ inline KernelErfTinyInplace() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, AscendC::TPipe* pipeIn)
    {
        pipe = pipeIn;
        this->length = length;

        xGm.SetGlobalBuffer((__gm__ DT_X*)x, length);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y, length);
        yPtr = (__gm__ DT_X*)y;

        pipe->InitBuffer(xBuf, TINY_TILE * sizeof(DT_X));
        pipe->InitBuffer(
            tmpBuf,
            (2 * TINY_TILE + TMP_PAD_FLOAT_TINY) * sizeof(float)
        );
    }

    __aicore__ inline void Process()
    {
        const uint32_t len = this->length;
        if (len == 0) {
            return;
        }

        AscendC::LocalTensor<DT_X> xLocal = xBuf.Get<DT_X>();

        AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();
        AscendC::LocalTensor<float> tmp1 = tmp;
        AscendC::LocalTensor<float> tmp2 = tmp1[TINY_TILE + TMP_PAD_FLOAT_TINY];

        if constexpr (ALIGNED) {
            AscendC::DataCopy(xLocal, xGm[0], len);
        } else {
            AscendC::DataCopyExtParams copyParams = {
                1,
                static_cast<uint32_t>(len * sizeof(DT_X)),
                0,
                0,
                0
            };
            AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0, 0};
            AscendC::DataCopyPad(xLocal, xGm[0], copyParams, padParams);
        }

        auto eventIdMTE2ToV = pipe->FetchEventID<AscendC::HardEvent::MTE2_V>();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMTE2ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMTE2ToV);

        const uint32_t fastMode = GetTinyFastModeByLen<DT_X>(xLocal, len);
        if (!RunTinyFastMode<DT_X>(xLocal, len, fastMode)) {
            ComputeP4Inplace<DT_X>(xLocal, tmp1, tmp2, len);
        }

        auto eventIdVToMTE3 = pipe->FetchEventID<AscendC::HardEvent::V_MTE3>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMTE3);

        if constexpr (ALIGNED) {
            AscendC::DataCopy(yGm[0], xLocal, len);
        } else {
            const uint32_t mainLen = (len / BLOCK_ELEMENT_NUM) * BLOCK_ELEMENT_NUM;
            const uint32_t tailLen = len - mainLen;

            if (mainLen > 0) {
                AscendC::DataCopy(yGm[0], xLocal, mainLen);
            }

            for (uint32_t i = 0; i < tailLen; ++i) {
                yPtr[mainLen + i] = xLocal.GetValue(mainLen + i);
            }
        }
    }

private:
    AscendC::TPipe* pipe;

    AscendC::TBuf<AscendC::QuePosition::VECIN> xBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf;

    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    __gm__ DT_X* yPtr;

    uint32_t length;
};

// ============================================================
// Small NoQueue Kernel
// length <= SMALL_TILE
// 单核 NoQueue
// ============================================================
template <class DT_X, uint32_t SMALL_TILE, bool ALIGNED>
class KernelErfSmallNoQueue {
public:
    __aicore__ inline KernelErfSmallNoQueue() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, AscendC::TPipe* pipeIn)
    {
        pipe = pipeIn;
        this->length = length;

        xGm.SetGlobalBuffer((__gm__ DT_X*)x, length);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y, length);
        yPtr = (__gm__ DT_X*)y;

        pipe->InitBuffer(xBuf, SMALL_TILE * sizeof(DT_X));

        // tmp1 + padding + tmp2
        pipe->InitBuffer(
            tmpBuf,
            (2 * SMALL_TILE + TMP_PAD_FLOAT_SMALL_MID) * sizeof(float)
        );
    }

    __aicore__ inline void Process()
    {
        const uint32_t len = this->length;
        if (len == 0) {
            return;
        }

        AscendC::LocalTensor<DT_X> xLocal = xBuf.Get<DT_X>();

        AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();
        AscendC::LocalTensor<float> tmp1 = tmp;
        AscendC::LocalTensor<float> tmp2 = tmp1[SMALL_TILE + TMP_PAD_FLOAT_SMALL_MID];

        if constexpr (ALIGNED) {
            AscendC::DataCopy(xLocal, xGm[0], len);
        } else {
            AscendC::DataCopyExtParams copyParams = {
                1,
                static_cast<uint32_t>(len * sizeof(DT_X)),
                0,
                0,
                0
            };
            AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0, 0};
            AscendC::DataCopyPad(xLocal, xGm[0], copyParams, padParams);
        }

        auto eventIdMTE2ToV = pipe->FetchEventID<AscendC::HardEvent::MTE2_V>();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMTE2ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMTE2ToV);

        ComputeP4Inplace<DT_X>(xLocal, tmp1, tmp2, len);

        auto eventIdVToMTE3 = pipe->FetchEventID<AscendC::HardEvent::V_MTE3>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMTE3);

        if constexpr (ALIGNED) {
            AscendC::DataCopy(yGm[0], xLocal, len);
        } else {
            const uint32_t mainLen = (len / BLOCK_ELEMENT_NUM) * BLOCK_ELEMENT_NUM;
            const uint32_t tailLen = len - mainLen;

            if (mainLen > 0) {
                AscendC::DataCopy(yGm[0], xLocal, mainLen);
            }

            for (uint32_t i = 0; i < tailLen; ++i) {
                yPtr[mainLen + i] = xLocal.GetValue(mainLen + i);
            }
        }
    }

private:
    AscendC::TPipe* pipe;

    AscendC::TBuf<AscendC::QuePosition::VECIN> xBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf;

    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    __gm__ DT_X* yPtr;

    uint32_t length;
};

// ============================================================
// Mid TQue Kernel
// 8192 < total length <= 32768
// 多核，每核约 4096
// 针对 length=20000 这种 case
// ============================================================
template <class DT_X>
class KernelErfMid {
public:
    __aicore__ inline KernelErfMid() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, AscendC::TPipe* pipeIn)
    {
        pipe = pipeIn;

        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t blockIdx = AscendC::GetBlockIdx();

        // 以 128 元素为粒度均匀切分
        const uint32_t totalDataBlockNum =
            (length + LARGE_SPLIT_ELEMENT_NUM - 1) / LARGE_SPLIT_ELEMENT_NUM;

        const uint32_t baseBlockNum = totalDataBlockNum / blockNum;
        const uint32_t tailBlockNum = totalDataBlockNum % blockNum;

        const uint32_t coreBlockNum =
            baseBlockNum + (blockIdx < tailBlockNum ? 1U : 0U);

        const uint32_t coreBlockOffset =
            blockIdx * baseBlockNum + (blockIdx < tailBlockNum ? blockIdx : tailBlockNum);

        coreOffset = coreBlockOffset * LARGE_SPLIT_ELEMENT_NUM;

        const uint32_t paddedCoreLength = coreBlockNum * LARGE_SPLIT_ELEMENT_NUM;

        coreLength = coreOffset >= length
            ? 0U
            : (paddedCoreLength < length - coreOffset ? paddedCoreLength : length - coreOffset);

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + coreOffset, coreLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + coreOffset, coreLength);

        tileNum = (coreLength + TILE_LENGTH_MID - 1) / TILE_LENGTH_MID;

        pipe->InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH_MID * sizeof(DT_X));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, TILE_LENGTH_MID * sizeof(DT_X));

        // tmp1 + padding + tmp2
        pipe->InitBuffer(
            tmpBuf,
            (2 * TILE_LENGTH_MID + TMP_PAD_FLOAT_SMALL_MID) * sizeof(float)
        );
    }

    __aicore__ inline void Process()
    {
        if (tileNum == 0) {
            return;
        }

        CopyIn(0, GetTileLength(0));

        for (uint32_t i = 0; i < tileNum; ++i) {
            const uint32_t len = GetTileLength(i);

            if (i > 0) {
                CopyOut(i - 1, GetTileLength(i - 1));
            }

            if (i + 1 < tileNum) {
                CopyIn(i + 1, GetTileLength(i + 1));
            }

            Compute(len);
        }

        CopyOut(tileNum - 1, GetTileLength(tileNum - 1));
    }

private:
    __aicore__ inline uint32_t GetTileLength(uint32_t progress) const
    {
        if (progress == tileNum - 1) {
            return coreLength - progress * TILE_LENGTH_MID;
        }
        return TILE_LENGTH_MID;
    }

    __aicore__ inline void CopyIn(uint32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();

        if ((len % BLOCK_ELEMENT_NUM) == 0) {
            AscendC::DataCopy(xLocal, xGm[progress * TILE_LENGTH_MID], len);
        } else {
            AscendC::DataCopyExtParams copyParams = {
                1,
                static_cast<uint32_t>(len * sizeof(DT_X)),
                0,
                0,
                0
            };
            AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0, 0};
            AscendC::DataCopyPad(xLocal, xGm[progress * TILE_LENGTH_MID], copyParams, padParams);
        }

        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();

        AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();
        AscendC::LocalTensor<float> tmp1 = tmp;
        AscendC::LocalTensor<float> tmp2 = tmp1[TILE_LENGTH_MID + TMP_PAD_FLOAT_SMALL_MID];

        ComputeP4<DT_X>(yLocal, xLocal, tmp1, tmp2, len);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();

        if ((len % BLOCK_ELEMENT_NUM) == 0) {
            AscendC::DataCopy(yGm[progress * TILE_LENGTH_MID], yLocal, len);
        } else {
            AscendC::DataCopyExtParams copyParams = {
                1,
                static_cast<uint32_t>(len * sizeof(DT_X)),
                0,
                0,
                0
            };
            AscendC::DataCopyPad(yGm[progress * TILE_LENGTH_MID], yLocal, copyParams);
        }

        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe* pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf;

    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;

    uint32_t coreOffset;
    uint32_t coreLength;
    uint32_t tileNum;
};

// ============================================================
// Large Kernel
// length > 32768
// TQue 双缓冲
// 注意：Large 不加 padding，避免 UB 超限
// ============================================================
template <class DT_X>
class KernelErfLarge {
public:
    __aicore__ inline KernelErfLarge() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, AscendC::TPipe* pipeIn)
    {
        pipe = pipeIn;

        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t blockIdx = AscendC::GetBlockIdx();

        const uint32_t totalDataBlockNum =
            (length + LARGE_SPLIT_ELEMENT_NUM - 1) / LARGE_SPLIT_ELEMENT_NUM;

        const uint32_t baseBlockNum = totalDataBlockNum / blockNum;
        const uint32_t tailBlockNum = totalDataBlockNum % blockNum;

        const uint32_t coreBlockNum =
            baseBlockNum + (blockIdx < tailBlockNum ? 1U : 0U);

        const uint32_t coreBlockOffset =
            blockIdx * baseBlockNum + (blockIdx < tailBlockNum ? blockIdx : tailBlockNum);

        coreOffset = coreBlockOffset * LARGE_SPLIT_ELEMENT_NUM;

        const uint32_t paddedCoreLength = coreBlockNum * LARGE_SPLIT_ELEMENT_NUM;

        coreLength = coreOffset >= length
            ? 0U
            : (paddedCoreLength < length - coreOffset ? paddedCoreLength : length - coreOffset);

        tileNum = (coreLength + TILE_LENGTH - 1) / TILE_LENGTH;

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + coreOffset, coreLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + coreOffset, coreLength);

        pipe->InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));

        // Large 不加 padding，否则 2*in + 2*out + tmp 接近或超过 UB 上限
        pipe->InitBuffer(
            tmpBuf,
            (2 * TILE_LENGTH + TMP_PAD_FLOAT_LARGE) * sizeof(float)
        );
    }

    __aicore__ inline void Process()
    {
        if (tileNum == 0) {
            return;
        }

        CopyIn(0, GetTileLength(0));

        for (uint32_t i = 0; i < tileNum; ++i) {
            const uint32_t len = GetTileLength(i);

            if (i > 0) {
                CopyOut(i - 1, GetTileLength(i - 1));
            }

            if (i + 1 < tileNum) {
                CopyIn(i + 1, GetTileLength(i + 1));
            }

            Compute(len);
        }

        CopyOut(tileNum - 1, GetTileLength(tileNum - 1));
    }

private:
    __aicore__ inline uint32_t GetTileLength(uint32_t progress) const
    {
        if (progress == tileNum - 1) {
            return coreLength - progress * TILE_LENGTH;
        }
        return TILE_LENGTH;
    }

    __aicore__ inline void CopyIn(uint32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();

        if ((len % BLOCK_ELEMENT_NUM) == 0) {
            AscendC::DataCopy(xLocal, xGm[progress * TILE_LENGTH], len);
        } else {
            AscendC::DataCopyExtParams copyParams = {
                1,
                static_cast<uint32_t>(len * sizeof(DT_X)),
                0,
                0,
                0
            };
            AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0, 0};
            AscendC::DataCopyPad(xLocal, xGm[progress * TILE_LENGTH], copyParams, padParams);
        }

        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();

        AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();
        AscendC::LocalTensor<float> tmp1 = tmp;
        AscendC::LocalTensor<float> tmp2 = tmp1[TILE_LENGTH + TMP_PAD_FLOAT_LARGE];

        ComputeP4<DT_X>(yLocal, xLocal, tmp1, tmp2, len);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();

        if ((len % BLOCK_ELEMENT_NUM) == 0) {
            AscendC::DataCopy(yGm[progress * TILE_LENGTH], yLocal, len);
        } else {
            AscendC::DataCopyExtParams copyParams = {
                1,
                static_cast<uint32_t>(len * sizeof(DT_X)),
                0,
                0,
                0
            };
            AscendC::DataCopyPad(yGm[progress * TILE_LENGTH], yLocal, copyParams);
        }

        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe* pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf;

    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;

    uint32_t coreOffset;
    uint32_t coreLength;
    uint32_t tileNum;
};

// ============================================================
// 120K~140K Special Large Kernel
// Fixed 16 cores, one inplace vector tile per core
// ============================================================
template <class DT_X>
class KernelErfLarge120K140K16CoreInplace {
public:
    __aicore__ inline KernelErfLarge120K140K16CoreInplace() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, AscendC::TPipe* pipeIn)
    {
        pipe = pipeIn;

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t totalDataBlockNum =
            (length + LARGE_SPLIT_ELEMENT_NUM - 1) / LARGE_SPLIT_ELEMENT_NUM;

        const uint32_t baseBlockNum = totalDataBlockNum / LARGE_SPECIAL_16CORE_NUM;
        const uint32_t tailBlockNum = totalDataBlockNum % LARGE_SPECIAL_16CORE_NUM;

        const uint32_t coreBlockNum =
            baseBlockNum + (blockIdx < tailBlockNum ? 1U : 0U);

        const uint32_t coreBlockOffset =
            blockIdx * baseBlockNum + (blockIdx < tailBlockNum ? blockIdx : tailBlockNum);

        coreOffset = coreBlockOffset * LARGE_SPLIT_ELEMENT_NUM;

        const uint32_t paddedCoreLength = coreBlockNum * LARGE_SPLIT_ELEMENT_NUM;
        coreLength = coreOffset >= length
            ? 0U
            : (paddedCoreLength < length - coreOffset ? paddedCoreLength : length - coreOffset);

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + coreOffset, coreLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + coreOffset, coreLength);

        pipe->InitBuffer(xBuf, TILE_LENGTH_LARGE_16CORE_INPLACE * sizeof(DT_X));
        pipe->InitBuffer(
            tmpBuf,
            (2 * TILE_LENGTH_LARGE_16CORE_INPLACE + TMP_PAD_FLOAT_LARGE) * sizeof(float)
        );
    }

    __aicore__ inline void Process()
    {
        if (coreLength == 0U) {
            return;
        }

        AscendC::LocalTensor<DT_X> xLocal = xBuf.Get<DT_X>();
        AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();
        AscendC::LocalTensor<float> tmp1 = tmp;
        AscendC::LocalTensor<float> tmp2 = tmp1[TILE_LENGTH_LARGE_16CORE_INPLACE + TMP_PAD_FLOAT_LARGE];

        if ((coreLength % BLOCK_ELEMENT_NUM) == 0) {
            AscendC::DataCopy(xLocal, xGm[0], coreLength);
        } else {
            AscendC::DataCopyExtParams copyParams = {
                1,
                static_cast<uint32_t>(coreLength * sizeof(DT_X)),
                0,
                0,
                0
            };
            AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0, 0};
            AscendC::DataCopyPad(xLocal, xGm[0], copyParams, padParams);
        }

        auto eventIdMTE2ToV = pipe->FetchEventID<AscendC::HardEvent::MTE2_V>();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMTE2ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMTE2ToV);

        ComputeP4Inplace<DT_X>(xLocal, tmp1, tmp2, coreLength);

        auto eventIdVToMTE3 = pipe->FetchEventID<AscendC::HardEvent::V_MTE3>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMTE3);

        if ((coreLength % BLOCK_ELEMENT_NUM) == 0) {
            AscendC::DataCopy(yGm[0], xLocal, coreLength);
        } else {
            AscendC::DataCopyExtParams copyParams = {
                1,
                static_cast<uint32_t>(coreLength * sizeof(DT_X)),
                0,
                0,
                0
            };
            AscendC::DataCopyPad(yGm[0], xLocal, copyParams);
        }
    }

private:
    AscendC::TPipe* pipe;

    AscendC::TBuf<AscendC::QuePosition::VECIN> xBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf;

    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;

    uint32_t coreOffset;
    uint32_t coreLength;
};

// ============================================================
// Kernel Entry
// ============================================================
template <typename DT_X, uint32_t schMode>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;

    REGISTER_TILING_DEFAULT(ErfTilingData);

    if constexpr (schMode == ERF_TPL_SCH_MODE_TINY_FIXED_1) {
        if (AscendC::GetBlockIdx() != 0U) {
            return;
        }
        const float z = static_cast<float>(((__gm__ DT_X*)x)[0]);
        ((__gm__ DT_X*)y)[0] = static_cast<DT_X>(ErfScalarP4NoBranch(z));
    } else if constexpr (schMode == ERF_TPL_SCH_MODE_TINY_FIXED_2) {
        KernelErfUltraTinyScalar<DT_X, ERF_TINY_FIXED_LENGTH_2> op;
        op.Init(x, y);
        op.Process();
    } else if constexpr (schMode == ERF_TPL_SCH_MODE_TINY_FIXED_3) {
        KernelErfUltraTinyScalar<DT_X, ERF_TINY_FIXED_LENGTH_3> op;
        op.Init(x, y);
        op.Process();
    } else if constexpr (schMode == ERF_TPL_SCH_MODE_TINY_FIXED_4) {
        KernelErfUltraTinyScalar<DT_X, ERF_TINY_FIXED_LENGTH_4> op;
        op.Init(x, y);
        op.Process();
    } else if constexpr (schMode == ERF_TPL_SCH_MODE_TINY_FIXED_32) {
        KernelErfFixedScalarDirectP4<DT_X, ERF_TINY_FIXED_LENGTH_32> op;
        op.Init(x, y);
        op.Process();
    } else if constexpr (schMode == ERF_TPL_SCH_MODE_TINY_FIXED_128) {
        KernelErfFixed128Scalar4Core<DT_X> op;
        op.Init(x, y);
        op.Process();
    } else if constexpr (schMode == ERF_TPL_SCH_MODE_FIXED_371_VECTOR) {
        AscendC::TPipe pipe;
        KernelErfFixed371Vector8CoreTail40<DT_X> op;
        op.Init(x, y, &pipe);
        op.Process();
    } else {
        AscendC::TPipe pipe;

        if constexpr (schMode == ERF_TPL_SCH_MODE_TINY_FIXED_8) {
            KernelErfTinyFixedInplace<DT_X, ERF_TINY_FIXED_LENGTH_8> op;
            op.Init(x, y, &pipe);
            op.Process();
        } else if constexpr (schMode == ERF_TPL_SCH_MODE_TINY_FIXED_16) {
            KernelErfTinyFixedInplace<DT_X, ERF_TINY_FIXED_LENGTH_16> op;
            op.Init(x, y, &pipe);
            op.Process();
        } else if constexpr (schMode == ERF_TPL_SCH_MODE_TINY_FIXED_64) {
            KernelErfTinyFixedInplace<DT_X, ERF_TINY_FIXED_LENGTH_64> op;
            op.Init(x, y, &pipe);
            op.Process();
        } else {
            GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);

            if constexpr (schMode == ERF_TPL_SCH_MODE_TINY) {
                KernelErfTinyInplace<DT_X, TILE_LENGTH_TINY, true> op;
                op.Init(x, y, tiling_data.length, &pipe);
                op.Process();
            } else if constexpr (schMode == ERF_TPL_SCH_MODE_TINY_PAD) {
                KernelErfTinyInplace<DT_X, TILE_LENGTH_TINY, false> op;
                op.Init(x, y, tiling_data.length, &pipe);
                op.Process();
            } else if constexpr (schMode == ERF_TPL_SCH_MODE_SMALL_4K) {
                KernelErfSmallNoQueue<DT_X, TILE_LENGTH_SMALL_4K, true> op;
                op.Init(x, y, tiling_data.length, &pipe);
                op.Process();
            } else if constexpr (schMode == ERF_TPL_SCH_MODE_SMALL_4K_PAD) {
                KernelErfSmallNoQueue<DT_X, TILE_LENGTH_SMALL_4K, false> op;
                op.Init(x, y, tiling_data.length, &pipe);
                op.Process();
            } else if constexpr (schMode == ERF_TPL_SCH_MODE_SMALL_8K) {
                KernelErfSmallNoQueue<DT_X, TILE_LENGTH_SMALL_8K, true> op;
                op.Init(x, y, tiling_data.length, &pipe);
                op.Process();
            } else if constexpr (schMode == ERF_TPL_SCH_MODE_SMALL_8K_PAD) {
                KernelErfSmallNoQueue<DT_X, TILE_LENGTH_SMALL_8K, false> op;
                op.Init(x, y, tiling_data.length, &pipe);
                op.Process();
            } else if constexpr (schMode == ERF_TPL_SCH_MODE_MID) {
                KernelErfMid<DT_X> op;
                op.Init(x, y, tiling_data.length, &pipe);
                op.Process();
            } else if constexpr (schMode == ERF_TPL_SCH_MODE_LARGE_120_140K_16CORE) {
                KernelErfLarge120K140K16CoreInplace<DT_X> op;
                op.Init(x, y, tiling_data.length, &pipe);
                op.Process();
            } else if constexpr (schMode == ERF_TPL_SCH_MODE_LARGE) {
                KernelErfLarge<DT_X> op;
                op.Init(x, y, tiling_data.length, &pipe);
                op.Process();
            }
        }
    }
}
