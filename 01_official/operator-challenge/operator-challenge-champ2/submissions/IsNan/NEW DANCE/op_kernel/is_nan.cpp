#include "kernel_operator.h"

#include <type_traits>

using namespace AscendC;

namespace {

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t CALC_ALIGN_NUM = 256;
constexpr uint32_t kAlignBytes = 32;

template <typename T, bool IsBf16Input>
class KernelIsNan {
public:
    __aicore__ inline KernelIsNan() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t blockOffset,
                                uint32_t currentBlockLength, uint32_t tileLength)
    {
        blockLength_ = currentBlockLength;
        tileLength_ = tileLength;
        inputGm_.SetGlobalBuffer((__gm__ T*)input + blockOffset, currentBlockLength);
        outputGm_.SetGlobalBuffer((__gm__ uint8_t*)output + blockOffset, currentBlockLength);

        pipe_.InitBuffer(inQueue_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(outQueue_, BUFFER_NUM, tileLength_ * sizeof(uint8_t));

        pipe_.InitBuffer(valueQueue_, BUFFER_NUM, tileLength_ * sizeof(half));
        pipe_.InitBuffer(maskQueue_, BUFFER_NUM, tileLength_ * sizeof(uint8_t));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < blockLength_; offset += tileLength_) {
            calcLength_ = blockLength_ - offset;
            if (calcLength_ > tileLength_) {
                calcLength_ = tileLength_;
            }
            CopyIn(offset);
            Compute();
            CopyOut(offset);
        }
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const
    {
        return ((value + align - 1U) / align) * align;
    }

    __aicore__ inline void CopyIn(uint32_t offset)
    {
        LocalTensor<T> inputLocal = inQueue_.AllocTensor<T>();
        if ((calcLength_ * sizeof(T)) % kAlignBytes == 0) {
            DataCopy(inputLocal, inputGm_[offset], calcLength_);
        } else {
            DataCopyExtParams copyParams = {1, static_cast<uint32_t>(calcLength_ * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams = {true, 0, 0, 0};
            DataCopyPad(inputLocal, inputGm_[offset], copyParams, padParams);
        }
        inQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset)
    {
        LocalTensor<uint8_t> outputLocal = outQueue_.DeQue<uint8_t>();
        if (offset % kAlignBytes == 0 && calcLength_ % kAlignBytes == 0) {
            DataCopy(outputGm_[offset], outputLocal, calcLength_);
        } else {
            DataCopyExtParams copyParams = {1, static_cast<uint32_t>(calcLength_ * sizeof(uint8_t)), 0, 0, 0};
            DataCopyPad(outputGm_[offset], outputLocal, copyParams);
        }
        outQueue_.FreeTensor(outputLocal);
    }

    template <typename ComputeT>
    __aicore__ inline uint32_t AlignForVector(uint32_t len) const
    {
        return AlignUp(len, CALC_ALIGN_NUM / sizeof(ComputeT));
    }

    __aicore__ inline void WriteMaskAsBool(LocalTensor<uint8_t>& maskLocal, LocalTensor<half>& valueLocal,
                                           LocalTensor<uint8_t>& outputLocal, uint32_t alignedLen)
    {
        Duplicate(valueLocal, static_cast<half>(0.0f), alignedLen);
        PipeBarrier<PIPE_V>();
        Select(valueLocal, maskLocal, valueLocal, static_cast<half>(1.0f),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedLen);
        PipeBarrier<PIPE_V>();
        Cast(outputLocal, valueLocal, RoundMode::CAST_TRUNC, alignedLen);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<T> inputLocal = inQueue_.DeQue<T>();
        LocalTensor<uint8_t> outputLocal = outQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> maskLocal = maskQueue_.AllocTensor<uint8_t>();

        if constexpr (std::is_same_v<T, float>) {
            uint32_t alignedLen = AlignForVector<float>(calcLength_);
            Compare(maskLocal, inputLocal, inputLocal, CMPMODE::EQ, alignedLen);
            PipeBarrier<PIPE_V>();
            LocalTensor<half> valueLocal = valueQueue_.AllocTensor<half>();
            WriteMaskAsBool(maskLocal, valueLocal, outputLocal, alignedLen);
            valueQueue_.FreeTensor(valueLocal);
        } else if constexpr (IsBf16Input) {
            uint32_t alignedLen = AlignForVector<half>(calcLength_);
            LocalTensor<half> inputHalf = inputLocal.template ReinterpretCast<half>();
            Compare(maskLocal, inputHalf, inputHalf, CMPMODE::EQ, alignedLen);
            PipeBarrier<PIPE_V>();
            LocalTensor<half> valueLocal = valueQueue_.AllocTensor<half>();
            WriteMaskAsBool(maskLocal, valueLocal, outputLocal, alignedLen);
            valueQueue_.FreeTensor(valueLocal);
        } else {
            uint32_t alignedLen = AlignForVector<half>(calcLength_);
            LocalTensor<half> inputHalf = inputLocal.template ReinterpretCast<half>();
            Compare(maskLocal, inputHalf, inputHalf, CMPMODE::EQ, alignedLen);
            PipeBarrier<PIPE_V>();
            LocalTensor<half> valueLocal = valueQueue_.AllocTensor<half>();
            WriteMaskAsBool(maskLocal, valueLocal, outputLocal, alignedLen);
            valueQueue_.FreeTensor(valueLocal);
        }

        maskQueue_.FreeTensor(maskLocal);
        outQueue_.EnQue(outputLocal);
        inQueue_.FreeTensor(inputLocal);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    TQue<QuePosition::VECCALC, BUFFER_NUM> maskQueue_;
    TQue<QuePosition::VECCALC, BUFFER_NUM> valueQueue_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<uint8_t> outputGm_;
    uint32_t blockLength_ = 0;
    uint32_t tileLength_ = 0;
    uint32_t calcLength_ = 0;
};

template <typename T, bool IsBf16Input>
__aicore__ inline void RunIsNan(GM_ADDR input, GM_ADDR output, uint32_t blockLength,
                                uint32_t lastBlockLength, uint32_t tileLength)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockOffset = blockLength * blockIdx;
    uint32_t currentBlockLength = (blockIdx + 1 == GetBlockNum()) ? lastBlockLength : blockLength;
    KernelIsNan<T, IsBf16Input> kernel;
    kernel.Init(input, output, blockOffset, currentBlockLength, tileLength);
    kernel.Process();
}

}  // namespace

extern "C" __global__ __aicore__ void is_nan(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        RunIsNan<half, false>(input, output, tilingData.blockLength, tilingData.lastBlockLength,
                              tilingData.tileLength);
    } else if (TILING_KEY_IS(1)) {
        RunIsNan<bfloat16_t, true>(input, output, tilingData.blockLength, tilingData.lastBlockLength,
                                   tilingData.tileLength);
    } else if (TILING_KEY_IS(2)) {
        RunIsNan<float, false>(input, output, tilingData.blockLength, tilingData.lastBlockLength,
                               tilingData.tileLength);
    }
}
