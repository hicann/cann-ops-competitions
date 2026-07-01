// Kernel implementation
// op_kernel/erf.cpp

#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

constexpr uint32_t COPY_ALIGN_BYTES = 32;
constexpr float ERF_LINEAR_R = 0.015f;
constexpr float ERF_LINEAR_A = 1.12837916709f;

#ifndef ERF_ENABLE_SCALAR_CUBIC
#define ERF_ENABLE_SCALAR_CUBIC 1
#endif

constexpr float ERF_CUBIC_R = 0.16f;
constexpr float ERF_CUBIC_A = 1.128379167095f;
constexpr float ERF_CUBIC_B = -0.37612638903f;

// P2/Q1 rational approximation with direct Div:
// erf(x) ~= x * P(x^2) / Q(x^2)
// P(t) = P0 + P1*t + P2*t^2
// Q(t) = 1 + Q1*t
// This variant removes Reciprocal + Newton and keeps one direct Div.
// It is intended as a faster accurate fallback after raw Reciprocal failed precision
// and Reciprocal+Newton did not improve runtime.
constexpr float ERF_CLAMP_R = 2.2603462f;

constexpr float P0 = 1.13202866f;
constexpr float P1 = -0.00408418f;
constexpr float P2 = 0.00416779f;

constexpr float Q1 = 0.34436976f;

#ifndef ERF_LARGE_DISABLE_L2_CACHE
#define ERF_LARGE_DISABLE_L2_CACHE 1
#endif


// Fine-grained switches for small/medium paths.
// Host default writes all tiny lengths 1..31 as exact no-read tiling-key paths.
// RangeC1 full-fixed stays disabled by default to keep the measured P2/Q1 baseline.
#ifndef ERF_RANGE_32_64_FORCE_FULL_FIXED
#define ERF_RANGE_32_64_FORCE_FULL_FIXED 0
#endif
#ifndef ERF_RANGE_65_128_FORCE_FULL_FIXED
#define ERF_RANGE_65_128_FORCE_FULL_FIXED 0
#endif
#ifndef ERF_RANGE_129_256_FORCE_FULL_FIXED
#define ERF_RANGE_129_256_FORCE_FULL_FIXED 0
#endif
#ifndef ERF_RANGE_257_512_FORCE_FULL_FIXED
#define ERF_RANGE_257_512_FORCE_FULL_FIXED 0
#endif

// 1: Tiny lengths selected by the host tiling key use exact no-tiling-read scalar paths.
#ifndef ERF_ENABLE_TINY_EXACT_NO_READ
#define ERF_ENABLE_TINY_EXACT_NO_READ 1
#endif

// 1: Tiny exact scalar path is statically unrolled by template recursion.
#ifndef ERF_TINY_EXACT_UNROLL
#define ERF_TINY_EXACT_UNROLL 1
#endif

// 220x: 16 个 bank group。
// 对三个连续 buffer: x, y, p，buffer 间距为 tensorBytes + padBytes。
// 为避免 x 与 p 的 bank group 间距变成 0，要求：
//     (tensorBlocks + padBlocks) % 8 != 0
// 若 32B pad 不满足，则改用 64B pad。
template <class DT_X>
__aicore__ inline uint32_t ChooseLocalBufferBankPadBytes(uint32_t elemCount)
{
    uint32_t tensorBytes = elemCount * static_cast<uint32_t>(sizeof(DT_X));
    uint32_t tensorBlocks = tensorBytes / COPY_ALIGN_BYTES;  // 当前代码中 elemCount 均按 32B 对齐

    uint32_t padBlocks = 1U;  // 1 * 32B

    if (((tensorBlocks + padBlocks) & 7U) == 0U) {
        padBlocks = 2U;      // 2 * 32B = 64B
    }

    return padBlocks * COPY_ALIGN_BYTES;
}

constexpr event_t EVENT_ID_MTE2_TO_V = static_cast<event_t>(0);
constexpr event_t EVENT_ID_V_TO_MTE3 = static_cast<event_t>(1);
constexpr event_t EVENT_ID_V_TO_MTE2 = static_cast<event_t>(2);
constexpr event_t EVENT_ID_MTE3_TO_V = static_cast<event_t>(3);

template <class DT_X>
__aicore__ inline constexpr uint32_t GetCopyAlignElems()
{
    constexpr uint32_t ALIGN_ELEMS = COPY_ALIGN_BYTES / sizeof(DT_X);
    static_assert(ALIGN_ELEMS > 0U, "ALIGN_ELEMS must be positive.");
    return ALIGN_ELEMS;
}

__aicore__ inline uint32_t AlignUpU32(uint32_t value, uint32_t align)
{
    if (align == 0U) {
        return value;
    }
    return ((value + align - 1U) / align) * align;
}

template <class DT_X>
__aicore__ inline void BindStaticCalcTensors(
    uint32_t elemCount,
    AscendC::LocalTensor<DT_X> &xLocal,
    AscendC::LocalTensor<DT_X> &yLocal,
    AscendC::LocalTensor<DT_X> &pLocal)
{
    uint32_t tensorBytes = elemCount * static_cast<uint32_t>(sizeof(DT_X));
    uint32_t padBytes = ChooseLocalBufferBankPadBytes<DT_X>(elemCount);

    uint32_t xAddr = 0U;
    uint32_t yAddr = xAddr + tensorBytes + padBytes;
    uint32_t pAddr = yAddr + tensorBytes + padBytes;

    xLocal = AscendC::LocalTensor<DT_X>(AscendC::TPosition::VECCALC, xAddr, elemCount);
    yLocal = AscendC::LocalTensor<DT_X>(AscendC::TPosition::VECCALC, yAddr, elemCount);
    pLocal = AscendC::LocalTensor<DT_X>(AscendC::TPosition::VECCALC, pAddr, elemCount);
}

template <class DT_X, uint32_t BUF_LEN>
__aicore__ inline void BindStaticCalcTensorsFixed(
    AscendC::LocalTensor<DT_X> &xLocal,
    AscendC::LocalTensor<DT_X> &yLocal,
    AscendC::LocalTensor<DT_X> &pLocal)
{
    constexpr uint32_t TENSOR_BYTES =
        BUF_LEN * static_cast<uint32_t>(sizeof(DT_X));

    static_assert((TENSOR_BYTES % COPY_ALIGN_BYTES) == 0U,
                  "fixed local tensor size must be 32B aligned.");

    constexpr uint32_t TENSOR_BLOCKS = TENSOR_BYTES / COPY_ALIGN_BYTES;

    constexpr uint32_t PAD_BYTES =
        (((TENSOR_BLOCKS + 1U) & 7U) == 0U)
            ? (2U * COPY_ALIGN_BYTES)
            : COPY_ALIGN_BYTES;

    constexpr uint32_t X_ADDR = 0U;
    constexpr uint32_t Y_ADDR = X_ADDR + TENSOR_BYTES + PAD_BYTES;
    constexpr uint32_t P_ADDR = Y_ADDR + TENSOR_BYTES + PAD_BYTES;

    xLocal = AscendC::LocalTensor<DT_X>(
        AscendC::TPosition::VECCALC, X_ADDR, BUF_LEN);
    yLocal = AscendC::LocalTensor<DT_X>(
        AscendC::TPosition::VECCALC, Y_ADDR, BUF_LEN);
    pLocal = AscendC::LocalTensor<DT_X>(
        AscendC::TPosition::VECCALC, P_ADDR, BUF_LEN);
}

template <class DT_X>
__aicore__ inline DT_X ScalarErfApproxValue(DT_X x)
{
    if (x <= static_cast<DT_X>(-ERF_CLAMP_R)) {
        return static_cast<DT_X>(-1.0f);
    }
    if (x >= static_cast<DT_X>(ERF_CLAMP_R)) {
        return static_cast<DT_X>(1.0f);
    }

    if (x >= static_cast<DT_X>(-ERF_LINEAR_R) &&
        x <= static_cast<DT_X>(ERF_LINEAR_R)) {
        return static_cast<DT_X>(ERF_LINEAR_A) * x;
    }

#if ERF_ENABLE_SCALAR_CUBIC
    if (x >= static_cast<DT_X>(-ERF_CUBIC_R) &&
        x <= static_cast<DT_X>(ERF_CUBIC_R)) {
        DT_X t = x * x;
        return x * (static_cast<DT_X>(ERF_CUBIC_A) +
                    static_cast<DT_X>(ERF_CUBIC_B) * t);
    }
#endif

    DT_X t = x * x;

    DT_X p = static_cast<DT_X>(P2) * t + static_cast<DT_X>(P1);
    p = p * t + static_cast<DT_X>(P0);
    p = p * x;

    DT_X q = static_cast<DT_X>(Q1) * t + static_cast<DT_X>(1.0f);

    return p / q;
}

template <typename DT_X>
__aicore__ inline DT_X ScalarErfP2Q1(DT_X x)
{
    DT_X xc = x;
    if (xc < static_cast<DT_X>(-ERF_CLAMP_R)) {
        xc = static_cast<DT_X>(-ERF_CLAMP_R);
    }
    if (xc > static_cast<DT_X>(ERF_CLAMP_R)) {
        xc = static_cast<DT_X>(ERF_CLAMP_R);
    }

    DT_X t = xc * xc;

    DT_X p = static_cast<DT_X>(P2) * t + static_cast<DT_X>(P1);
    p = p * t + static_cast<DT_X>(P0);
    p = p * xc;

    DT_X q = static_cast<DT_X>(Q1) * t + static_cast<DT_X>(1.0f);
    return p / q;
}

template <typename DT_X, uint32_t COUNT>
__aicore__ inline void RunLocalPadeFixed(GM_ADDR x, GM_ADDR y, uint32_t offset)
{
    __gm__ DT_X *xGm = reinterpret_cast<__gm__ DT_X *>(x);
    __gm__ DT_X *yGm = reinterpret_cast<__gm__ DT_X *>(y);

#pragma unroll
    for (uint32_t i = 0U; i < COUNT; ++i) {
        DT_X vx = xGm[offset + i];
        yGm[offset + i] = ScalarErfP2Q1<DT_X>(vx);
    }
}

template <typename DT_X, uint32_t COUNT>
__aicore__ inline void RunLocalPadeBounded(
    GM_ADDR x,
    GM_ADDR y,
    uint32_t offset,
    uint32_t length)
{
    __gm__ DT_X *xGm = reinterpret_cast<__gm__ DT_X *>(x);
    __gm__ DT_X *yGm = reinterpret_cast<__gm__ DT_X *>(y);

#pragma unroll
    for (uint32_t i = 0U; i < COUNT; ++i) {
        uint32_t idx = offset + i;
        if (idx < length) {
            DT_X vx = xGm[idx];
            yGm[idx] = ScalarErfP2Q1<DT_X>(vx);
        }
    }
}

template <class DT_X>
__aicore__ inline void ApplyErfApproxRuntimeMaskConfigured(
    AscendC::LocalTensor<DT_X> yLocal,
    AscendC::LocalTensor<DT_X> xLocal,
    AscendC::LocalTensor<DT_X> pLocal)
{
    AscendC::Maxs<DT_X, false>(yLocal, xLocal, static_cast<DT_X>(-ERF_CLAMP_R), 1);
    AscendC::Mins<DT_X, false>(yLocal, yLocal, static_cast<DT_X>(ERF_CLAMP_R), 1);

    AscendC::Mul<DT_X, false>(
        xLocal, yLocal, yLocal, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});

    // pLocal = P(t) = (P2*t + P1)*t + P0
    AscendC::Muls<DT_X, false>(pLocal, xLocal, static_cast<DT_X>(P2), 1);
    AscendC::Adds<DT_X, false>(pLocal, pLocal, static_cast<DT_X>(P1), 1);
    AscendC::Mul<DT_X, false>(
        pLocal, pLocal, xLocal, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
    AscendC::Adds<DT_X, false>(pLocal, pLocal, static_cast<DT_X>(P0), 1);

    // pLocal = numerator = xClamp * P(t)
    AscendC::Mul<DT_X, false>(
        pLocal, pLocal, yLocal, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});

    // xLocal = q = Q(t) = 1 + Q1*t
    AscendC::Muls<DT_X, false>(xLocal, xLocal, static_cast<DT_X>(Q1), 1);
    AscendC::Adds<DT_X, false>(xLocal, xLocal, static_cast<DT_X>(1.0f), 1);

    // pLocal = numerator / q
    AscendC::Div<DT_X, false>(
        pLocal, pLocal, xLocal, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
}

template <class DT_X>
__aicore__ inline void ApplyErfApproxRuntimeLen(
    AscendC::LocalTensor<DT_X> yLocal,
    AscendC::LocalTensor<DT_X> xLocal,
    AscendC::LocalTensor<DT_X> pLocal,
    uint32_t len)
{
    AscendC::SetMaskCount();
    AscendC::SetVectorMask<DT_X, AscendC::MaskMode::COUNTER>(static_cast<int32_t>(len));
    ApplyErfApproxRuntimeMaskConfigured<DT_X>(yLocal, xLocal, pLocal);
    AscendC::SetMaskNorm();
    AscendC::ResetMask();
}

template <class DT_X, uint32_t LEN>
__aicore__ inline void ApplyErfApproxFixedLen(
    AscendC::LocalTensor<DT_X> yLocal,
    AscendC::LocalTensor<DT_X> xLocal,
    AscendC::LocalTensor<DT_X> pLocal)
{
    AscendC::Maxs(yLocal, xLocal, static_cast<DT_X>(-ERF_CLAMP_R), LEN);
    AscendC::Mins(yLocal, yLocal, static_cast<DT_X>(ERF_CLAMP_R), LEN);

    // xLocal = t = xClamp^2
    AscendC::Mul(xLocal, yLocal, yLocal, LEN);

    // pLocal = P(t) = (P2*t + P1)*t + P0
    AscendC::Muls(pLocal, xLocal, static_cast<DT_X>(P2), LEN);
    AscendC::Adds(pLocal, pLocal, static_cast<DT_X>(P1), LEN);
    AscendC::Mul(pLocal, pLocal, xLocal, LEN);
    AscendC::Adds(pLocal, pLocal, static_cast<DT_X>(P0), LEN);

    // pLocal = numerator = xClamp * P(t)
    AscendC::Mul(pLocal, pLocal, yLocal, LEN);

    // xLocal = q = Q(t) = 1 + Q1*t
    AscendC::Muls(xLocal, xLocal, static_cast<DT_X>(Q1), LEN);
    AscendC::Adds(xLocal, xLocal, static_cast<DT_X>(1.0f), LEN);

    // pLocal = numerator / q
    AscendC::Div(pLocal, pLocal, xLocal, LEN);
}

template <class DT_X>
__aicore__ inline void ApplyErfApproxPreferFixed(
    AscendC::LocalTensor<DT_X> yLocal,
    AscendC::LocalTensor<DT_X> xLocal,
    AscendC::LocalTensor<DT_X> pLocal,
    uint32_t len)
{
    if (len == 32U) {
        ApplyErfApproxFixedLen<DT_X, 32U>(yLocal, xLocal, pLocal);
    } else if (len == 64U) {
        ApplyErfApproxFixedLen<DT_X, 64U>(yLocal, xLocal, pLocal);
    } else if (len == 128U) {
        ApplyErfApproxFixedLen<DT_X, 128U>(yLocal, xLocal, pLocal);
    } else if (len == 256U) {
        ApplyErfApproxFixedLen<DT_X, 256U>(yLocal, xLocal, pLocal);
    } else if (len == 512U) {
        ApplyErfApproxFixedLen<DT_X, 512U>(yLocal, xLocal, pLocal);
    } else if (len == 768U) {
        ApplyErfApproxFixedLen<DT_X, 768U>(yLocal, xLocal, pLocal);
    } else if (len == 1024U) {
        ApplyErfApproxFixedLen<DT_X, 1024U>(yLocal, xLocal, pLocal);
    } else if (len == 1280U) {
        ApplyErfApproxFixedLen<DT_X, 1280U>(yLocal, xLocal, pLocal);
    } else if (len == 1536U) {
        ApplyErfApproxFixedLen<DT_X, 1536U>(yLocal, xLocal, pLocal);
    } else if (len == 1792U) {
        ApplyErfApproxFixedLen<DT_X, 1792U>(yLocal, xLocal, pLocal);
    } else {
        ApplyErfApproxRuntimeLen<DT_X>(yLocal, xLocal, pLocal, len);
    }
}

template <class DT_X>
__aicore__ inline void ProcessTinyScalarRuntime(GM_ADDR x, GM_ADDR y, uint32_t length)
{
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;

    xGm.SetGlobalBuffer((__gm__ DT_X *)x, length);
    yGm.SetGlobalBuffer((__gm__ DT_X *)y, length);

    for (uint32_t i = 0U; i < length; ++i) {
        DT_X xVal = xGm.GetValue(i);
        DT_X yVal = ScalarErfApproxValue<DT_X>(xVal);
        yGm.SetValue(i, yVal);
    }
}


#if ERF_ENABLE_TINY_EXACT_NO_READ

template <class DT_X, uint32_t IDX, uint32_t LEN>
struct TinyScalarExactUnroll {
    __aicore__ static inline void Run(
        AscendC::GlobalTensor<DT_X> &xGm,
        AscendC::GlobalTensor<DT_X> &yGm)
    {
        DT_X xVal = xGm.GetValue(IDX);
        DT_X yVal = ScalarErfApproxValue<DT_X>(xVal);
        yGm.SetValue(IDX, yVal);
        TinyScalarExactUnroll<DT_X, IDX + 1U, LEN>::Run(xGm, yGm);
    }
};

template <class DT_X, uint32_t LEN>
struct TinyScalarExactUnroll<DT_X, LEN, LEN> {
    __aicore__ static inline void Run(
        AscendC::GlobalTensor<DT_X> &xGm,
        AscendC::GlobalTensor<DT_X> &yGm)
    {
        (void)xGm;
        (void)yGm;
    }
};

template <class DT_X, uint32_t LEN>
__aicore__ inline void ProcessTinyScalarExact(GM_ADDR x, GM_ADDR y)
{
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;

    xGm.SetGlobalBuffer((__gm__ DT_X *)x, LEN);
    yGm.SetGlobalBuffer((__gm__ DT_X *)y, LEN);

#if ERF_TINY_EXACT_UNROLL
    TinyScalarExactUnroll<DT_X, 0U, LEN>::Run(xGm, yGm);
#else
    for (uint32_t i = 0U; i < LEN; ++i) {
        DT_X xVal = xGm.GetValue(i);
        DT_X yVal = ScalarErfApproxValue<DT_X>(xVal);
        yGm.SetValue(i, yVal);
    }
#endif
}

#endif  // ERF_ENABLE_TINY_EXACT_NO_READ

template <class DT_X>
__aicore__ inline void ProcessTailScalarUpTo7(
    AscendC::GlobalTensor<DT_X> &xGm,
    AscendC::GlobalTensor<DT_X> &yGm,
    uint32_t offset,
    uint32_t validLen)
{
    if (validLen > 0U) {
        DT_X x0 = xGm.GetValue(offset + 0U);
        yGm.SetValue(offset + 0U, ScalarErfApproxValue<DT_X>(x0));
    }
    if (validLen > 1U) {
        DT_X x1 = xGm.GetValue(offset + 1U);
        yGm.SetValue(offset + 1U, ScalarErfApproxValue<DT_X>(x1));
    }
    if (validLen > 2U) {
        DT_X x2 = xGm.GetValue(offset + 2U);
        yGm.SetValue(offset + 2U, ScalarErfApproxValue<DT_X>(x2));
    }
    if (validLen > 3U) {
        DT_X x3 = xGm.GetValue(offset + 3U);
        yGm.SetValue(offset + 3U, ScalarErfApproxValue<DT_X>(x3));
    }
    if (validLen > 4U) {
        DT_X x4 = xGm.GetValue(offset + 4U);
        yGm.SetValue(offset + 4U, ScalarErfApproxValue<DT_X>(x4));
    }
    if (validLen > 5U) {
        DT_X x5 = xGm.GetValue(offset + 5U);
        yGm.SetValue(offset + 5U, ScalarErfApproxValue<DT_X>(x5));
    }
    if (validLen > 6U) {
        DT_X x6 = xGm.GetValue(offset + 6U);
        yGm.SetValue(offset + 6U, ScalarErfApproxValue<DT_X>(x6));
    }
}

template <class DT_X, uint32_t LEN>
class KernelErfExactFixedStatic {
public:
    __aicore__ inline KernelErfExactFixedStatic() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y)
    {
        xGm.SetGlobalBuffer((__gm__ DT_X *)x, LEN);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, LEN);
    }

    __aicore__ inline void Process()
{
    AscendC::LocalTensor<DT_X> xLocal;
    AscendC::LocalTensor<DT_X> yLocal;
    AscendC::LocalTensor<DT_X> pLocal;

    BindStaticCalcTensorsFixed<DT_X, LEN>(xLocal, yLocal, pLocal);

    AscendC::DataCopy(xLocal, xGm[0], LEN);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);

    ApplyErfApproxFixedLen<DT_X, LEN>(yLocal, xLocal, pLocal);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    AscendC::DataCopy(yGm[0], pLocal, LEN);
}

private:
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
};

template <class DT_X, uint32_t BUF_LEN, bool HAS_TAIL, bool FORCE_FULL_FIXED>
class KernelErfRangeC1Static {
public:
    __aicore__ inline KernelErfRangeC1Static() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t length,
        uint32_t alignedLengthIn)
    {
        totalLength = length;
        alignedLength = alignedLengthIn;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x, length);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, length);
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<DT_X> xLocal;
        AscendC::LocalTensor<DT_X> yLocal;
        AscendC::LocalTensor<DT_X> pLocal;
        BindStaticCalcTensorsFixed<DT_X, BUF_LEN>(xLocal, yLocal, pLocal);

        AscendC::DataCopy(xLocal, xGm[0], alignedLength);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);

        if constexpr (FORCE_FULL_FIXED) {
            // For non-exact lengths the copied prefix is 32B aligned, but may be shorter
            // than BUF_LEN. We intentionally compute the full static buffer and only copy
            // alignedLength elements back. The uninitialized padded lanes are never written to GM.
            ApplyErfApproxFixedLen<DT_X, BUF_LEN>(yLocal, xLocal, pLocal);
        } else {
            ApplyErfApproxPreferFixed<DT_X>(yLocal, xLocal, pLocal, alignedLength);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
        AscendC::DataCopy(yGm[0], pLocal, alignedLength);

        if constexpr (HAS_TAIL) {
            if (totalLength > alignedLength) {
                ProcessTailScalar(alignedLength, totalLength - alignedLength);
            }
        }
    }

private:
    __aicore__ inline void ProcessTailScalar(uint32_t offset, uint32_t validLen)
    {
        ProcessTailScalarUpTo7<DT_X>(xGm, yGm, offset, validLen);
    }

private:
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t totalLength = 0U;
    uint32_t alignedLength = 0U;
};

template <class DT_X, bool HAS_TAIL>
class KernelErfRangeSplitStatic {
public:
    __aicore__ inline KernelErfRangeSplitStatic() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t length,
        uint32_t alignedLengthIn,
        uint32_t tileLengthIn,
        uint32_t formerNumIn,
        uint32_t formerLengthIn,
        uint32_t tailLengthIn)
    {
        totalLength = length;
        globalAlignedLength = alignedLengthIn;
        localBufferLength = tileLengthIn;
        formerNum = formerNumIn;
        formerLength = formerLengthIn;
        tailLength = tailLengthIn;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x, length);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, length);

        ComputeCoreRange();

        if (alignedLength > 0U) {
            constexpr uint32_t ALIGN_ELEMS = GetCopyAlignElems<DT_X>();
            if (localBufferLength < alignedLength) {
                localBufferLength = alignedLength;
            }
            if (localBufferLength < ALIGN_ELEMS) {
                localBufferLength = ALIGN_ELEMS;
            }
            localBufferLength = AlignUpU32(localBufferLength, ALIGN_ELEMS);
        }
    }

    __aicore__ inline void Process()
    {
        if (alignedLength > 0U) {
            ProcessAlignedChunk(coreStart, alignedLength);
        }

        if constexpr (HAS_TAIL) {
            if (scalarLength > 0U) {
                ProcessTailScalar(scalarStart, scalarLength);
            }
        }
    }

private:
    __aicore__ inline void ComputeCoreRange()
    {
        uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());

        coreStart = 0U;
        alignedLength = 0U;
        scalarStart = 0U;
        scalarLength = 0U;

        if (blockNum == 0U || blockIdx >= blockNum) {
            return;
        }

        if (blockIdx < formerNum) {
            coreStart = blockIdx * formerLength;
            alignedLength = formerLength;
        } else {
            coreStart = formerNum * formerLength + (blockIdx - formerNum) * tailLength;
            alignedLength = tailLength;
        }

        if (coreStart >= globalAlignedLength) {
            alignedLength = 0U;
        } else if (coreStart + alignedLength > globalAlignedLength) {
            alignedLength = globalAlignedLength - coreStart;
        }

        if (blockIdx + 1U == blockNum && totalLength > globalAlignedLength) {
            scalarStart = globalAlignedLength;
            scalarLength = totalLength - globalAlignedLength;
        }
    }

    __aicore__ inline void ProcessTailScalar(uint32_t offset, uint32_t validLen)
    {
        ProcessTailScalarUpTo7<DT_X>(xGm, yGm, offset, validLen);
    }

    __aicore__ inline void ProcessAlignedChunk(uint32_t offset, uint32_t len)
    {
        AscendC::LocalTensor<DT_X> xLocal;
        AscendC::LocalTensor<DT_X> yLocal;
        AscendC::LocalTensor<DT_X> pLocal;
        BindStaticCalcTensors<DT_X>(localBufferLength, xLocal, yLocal, pLocal);

        AscendC::DataCopy(xLocal, xGm[offset], len);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);

        ApplyErfApproxPreferFixed<DT_X>(yLocal, xLocal, pLocal, len);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
        AscendC::DataCopy(yGm[offset], pLocal, len);
    }

private:
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t totalLength = 0U;
    uint32_t globalAlignedLength = 0U;
    uint32_t localBufferLength = 0U;
    uint32_t formerNum = 0U;
    uint32_t formerLength = 0U;
    uint32_t tailLength = 0U;
    uint32_t coreStart = 0U;
    uint32_t alignedLength = 0U;
    uint32_t scalarStart = 0U;
    uint32_t scalarLength = 0U;
};

template <class DT_X, uint32_t BUCKET_LEN, bool HAS_TAIL>
class KernelErfMediumBucketStatic {
public:
    __aicore__ inline KernelErfMediumBucketStatic() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t length,
        uint32_t alignedLengthIn,
        uint32_t formerNumIn,
        uint32_t formerLengthIn,
        uint32_t tailLengthIn)
    {
        totalLength = length;
        globalAlignedLength = alignedLengthIn;
        formerNum = formerNumIn;
        formerLength = formerLengthIn;
        tailLength = tailLengthIn;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x, length);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, length);

        ComputeCoreRange();
    }

    __aicore__ inline void Process()
    {
        if (coreLength > 0U) {
            AscendC::LocalTensor<DT_X> xLocal;
            AscendC::LocalTensor<DT_X> yLocal;
            AscendC::LocalTensor<DT_X> pLocal;
            BindStaticCalcTensorsFixed<DT_X, BUCKET_LEN>(xLocal, yLocal, pLocal);
            ProcessAlignedChunk(xLocal, yLocal, pLocal, coreStartOffset, coreLength);
        }

        if constexpr (HAS_TAIL) {
            if (scalarLength > 0U) {
                ProcessTailScalar(scalarStart, scalarLength);
            }
        }
    }

private:
    __aicore__ inline void ComputeCoreRange()
    {
        uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());

        coreStartOffset = 0U;
        coreLength = 0U;
        scalarStart = 0U;
        scalarLength = 0U;

        if (blockNum == 0U || blockIdx >= blockNum) {
            return;
        }

        if (blockIdx < formerNum) {
            coreStartOffset = blockIdx * formerLength;
            coreLength = formerLength;
        } else {
            coreStartOffset = formerNum * formerLength + (blockIdx - formerNum) * tailLength;
            coreLength = tailLength;
        }

        if (coreStartOffset >= globalAlignedLength) {
            coreLength = 0U;
        } else if (coreStartOffset + coreLength > globalAlignedLength) {
            coreLength = globalAlignedLength - coreStartOffset;
        }

        if (blockIdx + 1U == blockNum && totalLength > globalAlignedLength) {
            scalarStart = globalAlignedLength;
            scalarLength = totalLength - globalAlignedLength;
        }
    }

    __aicore__ inline void ProcessTailScalar(uint32_t offset, uint32_t validLen)
    {
        ProcessTailScalarUpTo7<DT_X>(xGm, yGm, offset, validLen);
    }

    __aicore__ inline void ProcessAlignedChunk(
        AscendC::LocalTensor<DT_X> xLocal,
        AscendC::LocalTensor<DT_X> yLocal,
        AscendC::LocalTensor<DT_X> pLocal,
        uint32_t offset,
        uint32_t len)
    {
        AscendC::DataCopy(xLocal, xGm[offset], len);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);

        if (len == BUCKET_LEN) {
            ApplyErfApproxFixedLen<DT_X, BUCKET_LEN>(yLocal, xLocal, pLocal);
        } else {
            ApplyErfApproxRuntimeLen<DT_X>(yLocal, xLocal, pLocal, len);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
        AscendC::DataCopy(yGm[offset], pLocal, len);
    }

private:
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t totalLength = 0U;
    uint32_t globalAlignedLength = 0U;
    uint32_t formerNum = 0U;
    uint32_t formerLength = 0U;
    uint32_t tailLength = 0U;
    uint32_t coreStartOffset = 0U;
    uint32_t coreLength = 0U;
    uint32_t scalarStart = 0U;
    uint32_t scalarLength = 0U;
};

template <class DT_X, bool HAS_TAIL>
class KernelErfLargeSingleTileStatic {
public:
    __aicore__ inline KernelErfLargeSingleTileStatic() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t length,
        uint32_t alignedLengthIn,
        uint32_t tileLengthIn,
        uint32_t formerNumIn,
        uint32_t formerLengthIn,
        uint32_t tailLengthIn)
    {
        totalLength = length;
        globalAlignedLength = alignedLengthIn;
        tileLength = tileLengthIn;
        formerNum = formerNumIn;
        formerLength = formerLengthIn;
        tailLength = tailLengthIn;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x, length);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, length);
#if ERF_LARGE_DISABLE_L2_CACHE
        //xGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
#endif

        ComputeCoreRange();

        if (coreLength > 0U) {
            constexpr uint32_t ALIGN_ELEMS = GetCopyAlignElems<DT_X>();
            if (tileLength < ALIGN_ELEMS) {
                tileLength = ALIGN_ELEMS;
            }
            tileLength = AlignUpU32(tileLength, ALIGN_ELEMS);
        }
    }

    __aicore__ inline void Process()
    {
        if (coreLength == 0U) {
            if constexpr (HAS_TAIL) {
                if (scalarLength > 0U) {
                    ProcessTailScalar(scalarStart, scalarLength);
                }
            }
            return;
        }

        AscendC::LocalTensor<DT_X> xLocal;
        AscendC::LocalTensor<DT_X> yLocal;
        AscendC::LocalTensor<DT_X> pLocal;
        BindStaticCalcTensors<DT_X>(tileLength, xLocal, yLocal, pLocal);
        ProcessSingleAlignedChunk(xLocal, yLocal, pLocal, coreStartOffset, coreLength);

        if constexpr (HAS_TAIL) {
            if (scalarLength > 0U) {
                ProcessTailScalar(scalarStart, scalarLength);
            }
        }
    }

private:
    __aicore__ inline void ComputeCoreRange()
    {
        uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());

        coreStartOffset = 0U;
        coreLength = 0U;
        scalarStart = 0U;
        scalarLength = 0U;

        if (blockNum == 0U || blockIdx >= blockNum) {
            return;
        }

        if (blockIdx < formerNum) {
            coreStartOffset = blockIdx * formerLength;
            coreLength = formerLength;
        } else {
            coreStartOffset = formerNum * formerLength + (blockIdx - formerNum) * tailLength;
            coreLength = tailLength;
        }

        if (coreStartOffset >= globalAlignedLength) {
            coreLength = 0U;
        } else if (coreStartOffset + coreLength > globalAlignedLength) {
            coreLength = globalAlignedLength - coreStartOffset;
        }

        if (blockIdx + 1U == blockNum && totalLength > globalAlignedLength) {
            scalarStart = globalAlignedLength;
            scalarLength = totalLength - globalAlignedLength;
        }
    }

    __aicore__ inline void ProcessTailScalar(uint32_t offset, uint32_t validLen)
    {
        ProcessTailScalarUpTo7<DT_X>(xGm, yGm, offset, validLen);
    }

    __aicore__ inline void ProcessSingleAlignedChunk(
        AscendC::LocalTensor<DT_X> xLocal,
        AscendC::LocalTensor<DT_X> yLocal,
        AscendC::LocalTensor<DT_X> pLocal,
        uint32_t offset,
        uint32_t len)
    {
        AscendC::DataCopy(xLocal, xGm[offset], len);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
        ApplyErfApproxRuntimeLen<DT_X>(yLocal, xLocal, pLocal, len);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
        AscendC::DataCopy(yGm[offset], pLocal, len);
    }

private:
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t totalLength = 0U;
    uint32_t globalAlignedLength = 0U;
    uint32_t tileLength = 0U;
    uint32_t formerNum = 0U;
    uint32_t formerLength = 0U;
    uint32_t tailLength = 0U;
    uint32_t coreStartOffset = 0U;
    uint32_t coreLength = 0U;
    uint32_t scalarStart = 0U;
    uint32_t scalarLength = 0U;
};

template <class DT_X, bool HAS_TAIL>
class KernelErfLargeMultiTileStatic {
public:
    __aicore__ inline KernelErfLargeMultiTileStatic() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t length,
        uint32_t alignedLengthIn,
        uint32_t tileLengthIn,
        uint32_t formerNumIn,
        uint32_t formerLengthIn,
        uint32_t tailLengthIn)
    {
        totalLength = length;
        globalAlignedLength = alignedLengthIn;
        tileLength = tileLengthIn;
        formerNum = formerNumIn;
        formerLength = formerLengthIn;
        tailLength = tailLengthIn;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x, length);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, length);
#if ERF_LARGE_DISABLE_L2_CACHE
        //xGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
#endif

        ComputeCoreRange();

        if (coreLength > 0U) {
            constexpr uint32_t ALIGN_ELEMS = GetCopyAlignElems<DT_X>();
            if (tileLength < ALIGN_ELEMS) {
                tileLength = ALIGN_ELEMS;
            }
            tileLength = AlignUpU32(tileLength, ALIGN_ELEMS);
        }
    }

    __aicore__ inline void Process()
    {
        if (coreLength == 0U) {
            if constexpr (HAS_TAIL) {
                if (scalarLength > 0U) {
                    ProcessTailScalar(scalarStart, scalarLength);
                }
            }
            return;
        }

        uint32_t fullTileCount = 0U;
        if (tileLength > 0U) {
            fullTileCount = coreLength / tileLength;
        }

        uint32_t processedLength = fullTileCount * tileLength;
        uint32_t remainLength = coreLength - processedLength;
        uint32_t vectorTileCount = fullTileCount + ((remainLength > 0U) ? 1U : 0U);

        AscendC::LocalTensor<DT_X> xLocal;
        AscendC::LocalTensor<DT_X> yLocal;
        AscendC::LocalTensor<DT_X> pLocal;
        BindStaticCalcTensors<DT_X>(tileLength, xLocal, yLocal, pLocal);

        uint32_t globalOffset = coreStartOffset;
        for (uint32_t i = 0U; i < fullTileCount; ++i) {
            bool hasPrevVectorTile = (i != 0U);
            bool hasNextVectorTile = (i + 1U) < vectorTileCount;
            ProcessAlignedChunk(
                xLocal,
                yLocal,
                pLocal,
                globalOffset,
                tileLength,
                hasPrevVectorTile,
                hasNextVectorTile);
            globalOffset += tileLength;
        }

        if (remainLength > 0U) {
            bool hasPrevVectorTile = (vectorTileCount > 1U);
            ProcessAlignedChunk(
                xLocal,
                yLocal,
                pLocal,
                globalOffset,
                remainLength,
                hasPrevVectorTile,
                false);
        }

        if constexpr (HAS_TAIL) {
            if (scalarLength > 0U) {
                ProcessTailScalar(scalarStart, scalarLength);
            }
        }
    }

private:
    __aicore__ inline void ComputeCoreRange()
    {
        uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());

        coreStartOffset = 0U;
        coreLength = 0U;
        scalarStart = 0U;
        scalarLength = 0U;

        if (blockNum == 0U || blockIdx >= blockNum) {
            return;
        }

        if (blockIdx < formerNum) {
            coreStartOffset = blockIdx * formerLength;
            coreLength = formerLength;
        } else {
            coreStartOffset = formerNum * formerLength + (blockIdx - formerNum) * tailLength;
            coreLength = tailLength;
        }

        if (coreStartOffset >= globalAlignedLength) {
            coreLength = 0U;
        } else if (coreStartOffset + coreLength > globalAlignedLength) {
            coreLength = globalAlignedLength - coreStartOffset;
        }

        if (blockIdx + 1U == blockNum && totalLength > globalAlignedLength) {
            scalarStart = globalAlignedLength;
            scalarLength = totalLength - globalAlignedLength;
        }
    }

    __aicore__ inline void ProcessTailScalar(uint32_t offset, uint32_t validLen)
    {
        ProcessTailScalarUpTo7<DT_X>(xGm, yGm, offset, validLen);
    }

    __aicore__ inline void ProcessAlignedChunk(
        AscendC::LocalTensor<DT_X> xLocal,
        AscendC::LocalTensor<DT_X> yLocal,
        AscendC::LocalTensor<DT_X> pLocal,
        uint32_t offset,
        uint32_t len,
        bool hasPrevVectorTile,
        bool hasNextVectorTile)
    {
        if (hasPrevVectorTile) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2);
        }

        AscendC::DataCopy(xLocal, xGm[offset], len);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);

        if (hasPrevVectorTile) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V);
        }

        ApplyErfApproxRuntimeLen<DT_X>(yLocal, xLocal, pLocal, len);

        if (hasNextVectorTile) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
        AscendC::DataCopy(yGm[offset], pLocal, len);

        if (hasNextVectorTile) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V);
        }
    }

private:
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t totalLength = 0U;
    uint32_t globalAlignedLength = 0U;
    uint32_t tileLength = 0U;
    uint32_t formerNum = 0U;
    uint32_t formerLength = 0U;
    uint32_t tailLength = 0U;
    uint32_t coreStartOffset = 0U;
    uint32_t coreLength = 0U;
    uint32_t scalarStart = 0U;
    uint32_t scalarLength = 0U;
};

template <typename DT_X, uint32_t ALGO_KIND, uint32_t MODE_KIND>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)
    AscendC::InitSocState();

    if constexpr (ALGO_KIND == ERF_ALGO_LOCAL_PADE &&
                  MODE_KIND == ERF_MODE_LOCAL_PADE_128_FIXED) {
        RunLocalPadeFixed<DT_X, 16U>(
            x,
            y,
            static_cast<uint32_t>(AscendC::GetBlockIdx()) * 16U);
        return;
    }

    if constexpr (ALGO_KIND == ERF_ALGO_RANGE_C1) {
        if constexpr (MODE_KIND == ERF_MODE_RANGE_32_FIXED) {
            KernelErfExactFixedStatic<DT_X, 32U> op;
            op.Init(x, y);
            op.Process();
            return;
        }

        if constexpr (MODE_KIND == ERF_MODE_RANGE_64_FIXED) {
            KernelErfExactFixedStatic<DT_X, 64U> op;
            op.Init(x, y);
            op.Process();
            return;
        }

        if constexpr (MODE_KIND == ERF_MODE_RANGE_128_FIXED) {
            KernelErfExactFixedStatic<DT_X, 128U> op;
            op.Init(x, y);
            op.Process();
            return;
        }

        if constexpr (MODE_KIND == ERF_MODE_RANGE_256_FIXED) {
            KernelErfExactFixedStatic<DT_X, 256U> op;
            op.Init(x, y);
            op.Process();
            return;
        }

        if constexpr (MODE_KIND == ERF_MODE_RANGE_512_FIXED) {
            KernelErfExactFixedStatic<DT_X, 512U> op;
            op.Init(x, y);
            op.Process();
            return;
        }
    }

#if ERF_ENABLE_TINY_EXACT_NO_READ
    if constexpr (ALGO_KIND == ERF_ALGO_TINY_SCALAR) {
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_1) {
            ProcessTinyScalarExact<DT_X, 1U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_2) {
            ProcessTinyScalarExact<DT_X, 2U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_3) {
            ProcessTinyScalarExact<DT_X, 3U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_4) {
            ProcessTinyScalarExact<DT_X, 4U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_5) {
            ProcessTinyScalarExact<DT_X, 5U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_6) {
            ProcessTinyScalarExact<DT_X, 6U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_7) {
            ProcessTinyScalarExact<DT_X, 7U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_8) {
            ProcessTinyScalarExact<DT_X, 8U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_9) {
            ProcessTinyScalarExact<DT_X, 9U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_10) {
            ProcessTinyScalarExact<DT_X, 10U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_11) {
            ProcessTinyScalarExact<DT_X, 11U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_12) {
            ProcessTinyScalarExact<DT_X, 12U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_13) {
            ProcessTinyScalarExact<DT_X, 13U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_14) {
            ProcessTinyScalarExact<DT_X, 14U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_15) {
            ProcessTinyScalarExact<DT_X, 15U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_16) {
            ProcessTinyScalarExact<DT_X, 16U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_17) {
            ProcessTinyScalarExact<DT_X, 17U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_18) {
            ProcessTinyScalarExact<DT_X, 18U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_19) {
            ProcessTinyScalarExact<DT_X, 19U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_20) {
            ProcessTinyScalarExact<DT_X, 20U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_21) {
            ProcessTinyScalarExact<DT_X, 21U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_22) {
            ProcessTinyScalarExact<DT_X, 22U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_23) {
            ProcessTinyScalarExact<DT_X, 23U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_24) {
            ProcessTinyScalarExact<DT_X, 24U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_25) {
            ProcessTinyScalarExact<DT_X, 25U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_26) {
            ProcessTinyScalarExact<DT_X, 26U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_27) {
            ProcessTinyScalarExact<DT_X, 27U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_28) {
            ProcessTinyScalarExact<DT_X, 28U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_29) {
            ProcessTinyScalarExact<DT_X, 29U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_30) {
            ProcessTinyScalarExact<DT_X, 30U>(x, y);
            return;
        }
        if constexpr (MODE_KIND == ERF_MODE_TINY_LEN_31) {
            ProcessTinyScalarExact<DT_X, 31U>(x, y);
            return;
        }
    }
#endif

    REGISTER_TILING_DEFAULT(ErfTilingData);

    if constexpr (ALGO_KIND == ERF_ALGO_LOCAL_PADE &&
                  MODE_KIND == ERF_MODE_LOCAL_PADE_16) {
        GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tilingData, tiling);

        uint32_t offset = static_cast<uint32_t>(AscendC::GetBlockIdx()) * 16U;
        RunLocalPadeBounded<DT_X, 16U>(
            x,
            y,
            offset,
            tilingData.length);
        return;
    }

    if constexpr (ALGO_KIND == ERF_ALGO_TINY_SCALAR) {
        GET_TILING_DATA_MEMBER(ErfTilingData, length, tinyLength, tiling);
        ProcessTinyScalarRuntime<DT_X>(x, y, tinyLength);
        return;
    }

    if constexpr (ALGO_KIND == ERF_ALGO_RANGE_C1) {
        GET_TILING_DATA_MEMBER(ErfTilingData, length, c1Length, tiling);
        GET_TILING_DATA_MEMBER(ErfTilingData, alignedLength, c1AlignedLength, tiling);

        if constexpr (MODE_KIND == ERF_MODE_RANGE_32_64_VECTOR) {
            if (c1Length > c1AlignedLength) {
                KernelErfRangeC1Static<DT_X, 64U, true, (ERF_RANGE_32_64_FORCE_FULL_FIXED != 0)> op;
                op.Init(x, y, c1Length, c1AlignedLength);
                op.Process();
            } else {
                KernelErfRangeC1Static<DT_X, 64U, false, (ERF_RANGE_32_64_FORCE_FULL_FIXED != 0)> op;
                op.Init(x, y, c1Length, c1AlignedLength);
                op.Process();
            }
            return;
        }

        if constexpr (MODE_KIND == ERF_MODE_RANGE_65_128) {
            if (c1Length > c1AlignedLength) {
                KernelErfRangeC1Static<DT_X, 128U, true, (ERF_RANGE_65_128_FORCE_FULL_FIXED != 0)> op;
                op.Init(x, y, c1Length, c1AlignedLength);
                op.Process();
            } else {
                KernelErfRangeC1Static<DT_X, 128U, false, (ERF_RANGE_65_128_FORCE_FULL_FIXED != 0)> op;
                op.Init(x, y, c1Length, c1AlignedLength);
                op.Process();
            }
            return;
        }

        if constexpr (MODE_KIND == ERF_MODE_RANGE_129_256) {
            if (c1Length > c1AlignedLength) {
                KernelErfRangeC1Static<DT_X, 256U, true, (ERF_RANGE_129_256_FORCE_FULL_FIXED != 0)> op;
                op.Init(x, y, c1Length, c1AlignedLength);
                op.Process();
            } else {
                KernelErfRangeC1Static<DT_X, 256U, false, (ERF_RANGE_129_256_FORCE_FULL_FIXED != 0)> op;
                op.Init(x, y, c1Length, c1AlignedLength);
                op.Process();
            }
            return;
        }

        if constexpr (MODE_KIND == ERF_MODE_RANGE_257_512_C1) {
            if (c1Length > c1AlignedLength) {
                KernelErfRangeC1Static<DT_X, 512U, true, (ERF_RANGE_257_512_FORCE_FULL_FIXED != 0)> op;
                op.Init(x, y, c1Length, c1AlignedLength);
                op.Process();
            } else {
                KernelErfRangeC1Static<DT_X, 512U, false, (ERF_RANGE_257_512_FORCE_FULL_FIXED != 0)> op;
                op.Init(x, y, c1Length, c1AlignedLength);
                op.Process();
            }
            return;
        }
    }

    if constexpr (ALGO_KIND == ERF_ALGO_MEDIUM_BUCKET) {
        GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tilingData, tiling);
        constexpr uint32_t BUCKET_LEN =
            (MODE_KIND == ERF_MODE_MEDIUM_BUCKET_256)  ? 256U  :
            (MODE_KIND == ERF_MODE_MEDIUM_BUCKET_512)  ? 512U  :
            (MODE_KIND == ERF_MODE_MEDIUM_BUCKET_768)  ? 768U  :
            (MODE_KIND == ERF_MODE_MEDIUM_BUCKET_1024) ? 1024U :
            (MODE_KIND == ERF_MODE_MEDIUM_BUCKET_1280) ? 1280U :
            (MODE_KIND == ERF_MODE_MEDIUM_BUCKET_1536) ? 1536U :
            (MODE_KIND == ERF_MODE_MEDIUM_BUCKET_1792) ? 1792U : 256U;

        if (tilingData.length > tilingData.alignedLength) {
            KernelErfMediumBucketStatic<DT_X, BUCKET_LEN, true> op;
            op.Init(
                x,
                y,
                tilingData.length,
                tilingData.alignedLength,
                tilingData.formerNum,
                tilingData.formerLength,
                tilingData.tailLength);
            op.Process();
        } else {
            KernelErfMediumBucketStatic<DT_X, BUCKET_LEN, false> op;
            op.Init(
                x,
                y,
                tilingData.length,
                tilingData.alignedLength,
                tilingData.formerNum,
                tilingData.formerLength,
                tilingData.tailLength);
            op.Process();
        }
        return;
    }

    if constexpr (ALGO_KIND == ERF_ALGO_RANGE_SPLIT) {
        GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tilingData, tiling);
        if (tilingData.length > tilingData.alignedLength) {
            KernelErfRangeSplitStatic<DT_X, true> op;
            op.Init(
                x,
                y,
                tilingData.length,
                tilingData.alignedLength,
                tilingData.tileLength,
                tilingData.formerNum,
                tilingData.formerLength,
                tilingData.tailLength);
            op.Process();
        } else {
            KernelErfRangeSplitStatic<DT_X, false> op;
            op.Init(
                x,
                y,
                tilingData.length,
                tilingData.alignedLength,
                tilingData.tileLength,
                tilingData.formerNum,
                tilingData.formerLength,
                tilingData.tailLength);
            op.Process();
        }
        return;
    }

    if constexpr (ALGO_KIND == ERF_ALGO_LARGE_TILED) {
        GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tilingData, tiling);
        uint32_t blockLength = tilingData.formerLength > tilingData.tailLength
                                   ? tilingData.formerLength
                                   : tilingData.tailLength;
        bool isSingleTile =
            (blockLength > 0U) && (blockLength <= tilingData.tileLength);
        bool hasTail = tilingData.length > tilingData.alignedLength;

        if (isSingleTile) {
            if (hasTail) {
                KernelErfLargeSingleTileStatic<DT_X, true> op;
                op.Init(
                    x,
                    y,
                    tilingData.length,
                    tilingData.alignedLength,
                    tilingData.tileLength,
                    tilingData.formerNum,
                    tilingData.formerLength,
                    tilingData.tailLength);
                op.Process();
            } else {
                KernelErfLargeSingleTileStatic<DT_X, false> op;
                op.Init(
                    x,
                    y,
                    tilingData.length,
                    tilingData.alignedLength,
                    tilingData.tileLength,
                    tilingData.formerNum,
                    tilingData.formerLength,
                    tilingData.tailLength);
                op.Process();
            }
        } else {
            if (hasTail) {
                KernelErfLargeMultiTileStatic<DT_X, true> op;
                op.Init(
                    x,
                    y,
                    tilingData.length,
                    tilingData.alignedLength,
                    tilingData.tileLength,
                    tilingData.formerNum,
                    tilingData.formerLength,
                    tilingData.tailLength);
                op.Process();
            } else {
                KernelErfLargeMultiTileStatic<DT_X, false> op;
                op.Init(
                    x,
                    y,
                    tilingData.length,
                    tilingData.alignedLength,
                    tilingData.tileLength,
                    tilingData.formerNum,
                    tilingData.formerLength,
                    tilingData.tailLength);
                op.Process();
            }
        }
        return;
    }
}
