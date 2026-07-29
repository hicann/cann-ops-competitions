#include <type_traits>

#include "kernel_operator.h"

#include "gelu_v2_tiling.h"

constexpr static uint32_t PIPELINE_DEPTH = 2;
constexpr static uint32_t GELU_V2_DT_FLOAT = 0;
constexpr static uint32_t GELU_V2_DT_FLOAT16 = 1;
constexpr static uint32_t GELU_V2_DT_BF16 = 27;
constexpr static float ONE_OVER_SQRT_TWO = 0.70710678118654752440f;
constexpr static float HALF = 0.5f;
constexpr static float ONE = 1.0f;
constexpr static float GELU_TANH_CUBE_COEFF = 0.044715f;
constexpr static float NEG_SQRT_EIGHT_OVER_PI = -1.5957691216057308f;

template <typename T>
class KernelGeluV2 {
public:
    __aicore__ inline KernelGeluV2() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const GeluV2TilingData *tiling)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockOffset = blockIdx * tiling->blockLength;
        uint32_t remain = (blockOffset < tiling->length) ? (tiling->length - blockOffset) : 0;
        blockLength_ = (remain < tiling->blockLength) ? remain : tiling->blockLength;
        length_ = tiling->length;
        tileSize_ = tiling->tileSize;
        approximate_ = tiling->approximate;

        xGm_.SetGlobalBuffer((__gm__ T *)x + blockOffset, blockLength_);
        yGm_.SetGlobalBuffer((__gm__ T *)y + blockOffset, blockLength_);

        pipe_.InitBuffer(inQueueX_, PIPELINE_DEPTH, tileSize_ * sizeof(T));
        pipe_.InitBuffer(outQueueY_, PIPELINE_DEPTH, tileSize_ * sizeof(T));
        if constexpr (!std::is_same<T, float>::value) {
            pipe_.InitBuffer(xFloatBuf_, tileSize_ * sizeof(float));
        }
        pipe_.InitBuffer(work1Buf_, tileSize_ * sizeof(float));
        pipe_.InitBuffer(work2Buf_, tileSize_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t loopCount = (blockLength_ + tileSize_ - 1) / tileSize_;
        for (uint32_t i = 0; i < loopCount; ++i) {
            uint32_t offset = i * tileSize_;
            uint32_t count = (i == loopCount - 1) ? (blockLength_ - offset) : tileSize_;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count)
    {
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        auto xLocal = inQueueX_.template AllocTensor<T>();
        AscendC::DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void ComputeExactFloat(const AscendC::LocalTensor<float> &yLocal,
                                             const AscendC::LocalTensor<float> &xLocal,
                                             uint32_t count)
    {
        auto work1 = work1Buf_.Get<float>();
        auto work2 = work2Buf_.Get<float>();
        AscendC::Muls(work1, xLocal, ONE_OVER_SQRT_TWO, count);
        AscendC::Erf(work1, work1, count);
        AscendC::Adds(work1, work1, ONE, count);
        AscendC::Muls(work2, xLocal, HALF, count);
        AscendC::Mul(yLocal, work1, work2, count);
    }

    __aicore__ inline void ComputeTanhFloat(const AscendC::LocalTensor<float> &yLocal,
                                            const AscendC::LocalTensor<float> &xLocal,
                                            uint32_t count)
    {
        auto work1 = work1Buf_.Get<float>();
        auto work2 = work2Buf_.Get<float>();
        AscendC::Mul(work1, xLocal, xLocal, count);
        AscendC::Mul(work2, work1, xLocal, count);
        AscendC::Muls(work2, work2, GELU_TANH_CUBE_COEFF, count);
        AscendC::Add(work2, work2, xLocal, count);
        AscendC::Muls(work2, work2, NEG_SQRT_EIGHT_OVER_PI, count);
        AscendC::Exp(work2, work2, count);
        AscendC::Adds(work2, work2, ONE, count);
        if constexpr (std::is_same<T, float>::value) {
            if (length_ >= GELU_V2_FP32_DIV_LENGTH) {
                AscendC::Div(yLocal, xLocal, work2, count);
            } else {
                AscendC::Reciprocal(work2, work2, count);
                AscendC::Mul(yLocal, xLocal, work2, count);
            }
        } else {
            AscendC::Reciprocal(work2, work2, count);
            AscendC::Mul(yLocal, xLocal, work2, count);
        }
    }

    __aicore__ inline void ComputeTanhHalf(const AscendC::LocalTensor<half> &yLocal,
                                           const AscendC::LocalTensor<half> &xLocal,
                                           uint32_t count)
    {
        auto work1 = work1Buf_.Get<half>();
        auto work2 = work2Buf_.Get<half>();
        AscendC::Mul(work1, xLocal, xLocal, count);
        AscendC::Mul(work2, work1, xLocal, count);
        AscendC::Muls(work2, work2, static_cast<half>(GELU_TANH_CUBE_COEFF), count);
        AscendC::Add(work2, work2, xLocal, count);
        AscendC::Muls(work2, work2, static_cast<half>(NEG_SQRT_EIGHT_OVER_PI), count);
        AscendC::Exp(work2, work2, count);
        AscendC::Adds(work2, work2, static_cast<half>(ONE), count);
        AscendC::Div(yLocal, xLocal, work2, count);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        auto xLocal = inQueueX_.template DeQue<T>();
        auto yLocal = outQueueY_.template AllocTensor<T>();

        if constexpr (std::is_same<T, float>::value) {
            if (approximate_ == 1) {
                ComputeTanhFloat(yLocal, xLocal, count);
            } else {
                ComputeExactFloat(yLocal, xLocal, count);
            }
        } else if constexpr (std::is_same<T, half>::value) {
            ComputeTanhHalf(yLocal, xLocal, count);
        } else {
            auto xF = xFloatBuf_.Get<float>();
            auto yF = work2Buf_.Get<float>();
            AscendC::Cast(xF, xLocal, AscendC::RoundMode::CAST_NONE, count);
            ComputeTanhFloat(yF, xF, count);
            constexpr auto roundMode = std::is_same<T, bfloat16_t>::value
                ? AscendC::RoundMode::CAST_RINT : AscendC::RoundMode::CAST_NONE;
            AscendC::Cast(yLocal, yF, roundMode, count);
        }

        outQueueY_.template EnQue<T>(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        auto yLocal = outQueueY_.template DeQue<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(yGm_[offset], yLocal, copyParams);
        outQueueY_.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<T> xGm_, yGm_;
    AscendC::TQue<AscendC::TPosition::VECIN, PIPELINE_DEPTH> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, PIPELINE_DEPTH> outQueueY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xFloatBuf_, work1Buf_, work2Buf_;
    uint32_t blockLength_ = 0;
    uint32_t length_ = 0;
    uint32_t tileSize_ = GELU_V2_DEFAULT_TILE;
    uint32_t approximate_ = 0;
};

extern "C" __global__ __aicore__ void gelu_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(GeluV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluV2TilingData, tilingData, tiling);
    if (tilingData.dtype == GELU_V2_DT_FLOAT) {
        KernelGeluV2<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (tilingData.dtype == GELU_V2_DT_FLOAT16) {
        KernelGeluV2<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (tilingData.dtype == GELU_V2_DT_BF16) {
        KernelGeluV2<bfloat16_t> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
