#include "kernel_operator.h"

#include <type_traits>

using namespace AscendC;

namespace {

constexpr uint64_t BLOCK_ELEMENTS = 256;
constexpr uint32_t VECTOR_ALIGN_ELEMENTS = 64;
constexpr float INV_SQRT_TWO = 0.70710678118654752440f;
// The tail form remains well conditioned below Acklam's original split when
// evaluated in fp32. Moving the split inward avoids cancellation in the
// central denominator around |x| ~= 0.95.
constexpr float TAIL_SPLIT = 0.8f;

__aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

__aicore__ inline uint64_t MinU64(uint64_t lhs, uint64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

template <typename T>
class KernelErfinv {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t totalLength,
                                uint64_t blocksPerCore, uint32_t extraCoreBlocks,
                                uint32_t tileLength, TPipe* pipe)
    {
        pipe_ = pipe;
        tileLength_ = tileLength;

        const uint64_t coreIndex = GetBlockIdx();
        const uint64_t coreBlocks = blocksPerCore + static_cast<uint64_t>(coreIndex < extraCoreBlocks);
        const uint64_t startBlock = coreIndex * blocksPerCore + MinU64(coreIndex, extraCoreBlocks);
        startOffset_ = startBlock * BLOCK_ELEMENTS;
        const uint64_t scheduledLength = coreBlocks * BLOCK_ELEMENTS;
        localLength_ = startOffset_ < totalLength
            ? MinU64(scheduledLength, totalLength - startOffset_)
            : 0;

        if (localLength_ == 0) {
            return;
        }

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + startOffset_, localLength_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + startOffset_, localLength_);

        const uint32_t ioBytes = tileLength_ * sizeof(T);
        const uint32_t fp32Bytes = tileLength_ * sizeof(float);
        const uint32_t maskBytes = AlignUp(tileLength_, 256) / 8;
        pipe_->InitBuffer(inputQueue_, 1, ioBytes);
        pipe_->InitBuffer(outputQueue_, 1, ioBytes);
        pipe_->InitBuffer(xFloatBuffer_, fp32Bytes);
        pipe_->InitBuffer(absBuffer_, fp32Bytes);
        pipe_->InitBuffer(workBuffer_, fp32Bytes);
        pipe_->InitBuffer(numeratorBuffer_, fp32Bytes);
        pipe_->InitBuffer(denominatorBuffer_, fp32Bytes);
        pipe_->InitBuffer(centerBuffer_, fp32Bytes);
        pipe_->InitBuffer(tailBuffer_, fp32Bytes);
        pipe_->InitBuffer(maskBuffer_, maskBytes);
    }

    __aicore__ inline void Process()
    {
        for (uint64_t offset = 0; offset < localLength_; offset += tileLength_) {
            const uint32_t validLength = static_cast<uint32_t>(MinU64(tileLength_, localLength_ - offset));
            CopyIn(offset, validLength);
            Compute(validLength);
            CopyOut(offset, validLength);
        }
    }

private:
    template <typename U>
    __aicore__ inline void CastInput(LocalTensor<float>& dst, LocalTensor<U>& src, uint32_t count)
    {
        if constexpr (std::is_same_v<U, float>) {
            Adds(dst, src, 0.0f, count);
        } else {
            Cast(dst, src, RoundMode::CAST_NONE, count);
        }
    }

    template <typename U>
    __aicore__ inline void CastOutput(LocalTensor<U>& dst, LocalTensor<float>& src, uint32_t count)
    {
        if constexpr (std::is_same_v<U, float>) {
            Adds(dst, src, 0.0f, count);
        } else {
            Cast(dst, src, RoundMode::CAST_RINT, count);
        }
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t validLength)
    {
        LocalTensor<T> input = inputQueue_.AllocTensor<T>();
        DataCopyExtParams copyParams{1, validLength * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(input, xGm_[offset], copyParams, padParams);
        inputQueue_.EnQue(input);
    }

    __aicore__ inline void EvaluateCentral(LocalTensor<float>& output,
                                           LocalTensor<float>& x,
                                           LocalTensor<float>& r,
                                           LocalTensor<float>& numerator,
                                           LocalTensor<float>& denominator,
                                           uint32_t count)
    {
        Muls(r, x, 0.5f, count);
        Mul(output, r, r, count);

        Duplicate(numerator, -3.96968303e+1f, count);
        Mul(numerator, numerator, output, count);
        Adds(numerator, numerator, 2.20946098e+2f, count);
        Mul(numerator, numerator, output, count);
        Adds(numerator, numerator, -2.75928503e+2f, count);
        Mul(numerator, numerator, output, count);
        Adds(numerator, numerator, 1.38357758e+2f, count);
        Mul(numerator, numerator, output, count);
        Adds(numerator, numerator, -3.06647987e+1f, count);
        Mul(numerator, numerator, output, count);
        Adds(numerator, numerator, 2.50662828f, count);
        Mul(numerator, numerator, r, count);

        Duplicate(denominator, -5.44760971e+1f, count);
        Mul(denominator, denominator, output, count);
        Adds(denominator, denominator, 1.61585831e+2f, count);
        Mul(denominator, denominator, output, count);
        Adds(denominator, denominator, -1.55698975e+2f, count);
        Mul(denominator, denominator, output, count);
        Adds(denominator, denominator, 6.68013153e+1f, count);
        Mul(denominator, denominator, output, count);
        Adds(denominator, denominator, -1.32806816e+1f, count);
        Mul(denominator, denominator, output, count);
        Adds(denominator, denominator, 1.0f, count);

        Div(output, numerator, denominator, count);
        Muls(output, output, INV_SQRT_TWO, count);
    }

    __aicore__ inline void EvaluateTail(LocalTensor<float>& output,
                                        LocalTensor<float>& x,
                                        LocalTensor<float>& absX,
                                        LocalTensor<float>& q,
                                        LocalTensor<float>& numerator,
                                        LocalTensor<float>& denominator,
                                        uint32_t count)
    {
        Muls(q, absX, -0.5f, count);
        Adds(q, q, 0.5f, count);
        Ln(q, q, count);
        Muls(q, q, -2.0f, count);
        Sqrt(q, q, count);

        Duplicate(numerator, -7.78489400e-3f, count);
        Mul(numerator, numerator, q, count);
        Adds(numerator, numerator, -3.22396457e-1f, count);
        Mul(numerator, numerator, q, count);
        Adds(numerator, numerator, -2.40075827f, count);
        Mul(numerator, numerator, q, count);
        Adds(numerator, numerator, -2.54973254f, count);
        Mul(numerator, numerator, q, count);
        Adds(numerator, numerator, 4.37466431f, count);
        Mul(numerator, numerator, q, count);
        Adds(numerator, numerator, 2.93816398f, count);

        Duplicate(denominator, 7.78469571e-3f, count);
        Mul(denominator, denominator, q, count);
        Adds(denominator, denominator, 3.22467119e-1f, count);
        Mul(denominator, denominator, q, count);
        Adds(denominator, denominator, 2.44513416f, count);
        Mul(denominator, denominator, q, count);
        Adds(denominator, denominator, 3.75440866f, count);
        Mul(denominator, denominator, q, count);
        Adds(denominator, denominator, 1.0f, count);

        Div(output, numerator, denominator, count);
        Muls(output, output, -INV_SQRT_TWO, count);
        Adds(q, absX, 1.0e-30f, count);
        Div(numerator, x, q, count);
        Mul(output, output, numerator, count);
    }

    __aicore__ inline void Compute(uint32_t validLength)
    {
        const uint32_t count = AlignUp(validLength, VECTOR_ALIGN_ELEMENTS);
        LocalTensor<T> input = inputQueue_.DeQue<T>();
        LocalTensor<T> output = outputQueue_.AllocTensor<T>();
        LocalTensor<float> x = xFloatBuffer_.Get<float>();
        LocalTensor<float> absX = absBuffer_.Get<float>();
        LocalTensor<float> work = workBuffer_.Get<float>();
        LocalTensor<float> numerator = numeratorBuffer_.Get<float>();
        LocalTensor<float> denominator = denominatorBuffer_.Get<float>();
        LocalTensor<float> center = centerBuffer_.Get<float>();
        LocalTensor<float> tail = tailBuffer_.Get<float>();
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>();

        CastInput(x, input, count);
        Abs(absX, x, count);
        EvaluateCentral(center, x, work, numerator, denominator, count);
        EvaluateTail(tail, x, absX, work, numerator, denominator, count);

        CompareScalar(mask, absX, TAIL_SPLIT, CMPMODE::GT, count);
        Select(center, mask, tail, center, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);

        // The rational tail evaluates inf/inf at exactly +/-1. Restore the
        // mathematical endpoint result while leaving domain errors as NaN.
        Muls(denominator, absX, -1.0f, count);
        Adds(denominator, denominator, 1.0f, count);
        Div(tail, x, denominator, count);
        CompareScalar(mask, absX, 1.0f, CMPMODE::EQ, count);
        Select(center, mask, tail, center, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);

        CastOutput(output, center, count);
        outputQueue_.EnQue(output);
        inputQueue_.FreeTensor(input);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t validLength)
    {
        LocalTensor<T> output = outputQueue_.DeQue<T>();
        DataCopyExtParams copyParams{1, validLength * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[offset], output, copyParams);
        outputQueue_.FreeTensor(output);
    }

private:
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<TPosition::VECCALC> xFloatBuffer_;
    TBuf<TPosition::VECCALC> absBuffer_;
    TBuf<TPosition::VECCALC> workBuffer_;
    TBuf<TPosition::VECCALC> numeratorBuffer_;
    TBuf<TPosition::VECCALC> denominatorBuffer_;
    TBuf<TPosition::VECCALC> centerBuffer_;
    TBuf<TPosition::VECCALC> tailBuffer_;
    TBuf<TPosition::VECCALC> maskBuffer_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    uint64_t startOffset_ = 0;
    uint64_t localLength_ = 0;
    uint32_t tileLength_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void erfinv(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KernelErfinv<DTYPE_X> op;
    op.Init(x, y, tilingData.totalLength, tilingData.blocksPerCore,
            tilingData.extraCoreBlocks, tilingData.tileLength, &pipe);
    op.Process();
}
