#define K_MAX_SHAPE_DIM 0

#include "kernel_operator.h"

#include <type_traits>

using namespace AscendC;

namespace {

constexpr int32_t BUFFER_NUM = 1;

template <typename T>
__aicore__ inline T MinValue(T x, T y) {
    return x < y ? x : y;
}

template <typename T>
__aicore__ inline T CeilDiv(T x, T y) {
    return (x + y - 1) / y;
}

template <typename T>
__aicore__ inline bool IsAligned32(uint32_t length) {
    return ((length * sizeof(T)) & 31U) == 0U;
}

template <typename T>
__aicore__ inline bool IsAligned32(uint32_t offset, uint32_t length) {
    return (((offset * sizeof(T)) & 31U) == 0U) && IsAligned32<T>(length);
}

template <typename T>
class KernelIsNan {
public:
    __aicore__ inline KernelIsNan() = default;

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t totalLength,
        uint32_t blockLength,
        uint32_t dataBlockSize,
        uint32_t tileLength,
        uint32_t alignedInputLength,
        uint32_t alignedOutputLength,
        uint32_t useBaselineSplit,
        TPipe* pipeIn) {
        pipe = pipeIn;
        this->totalLength = totalLength;
        this->blockLength = blockLength;
        this->dataBlockSize = dataBlockSize;
        this->tileLength = tileLength;
        this->alignedInputLength = alignedInputLength;
        this->alignedOutputLength = alignedOutputLength;
        this->useBaselineSplit = useBaselineSplit;
        blockIdx = GetBlockIdx();

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), totalLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y), totalLength);

        pipe->InitBuffer(inQueue, BUFFER_NUM, alignedInputLength * sizeof(T));
        pipe->InitBuffer(outQueue, BUFFER_NUM, alignedOutputLength * sizeof(uint8_t));
        if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, half>) {
            pipe->InitBuffer(compareBuf, alignedInputLength * sizeof(float));
        }
        pipe->InitBuffer(maskBuf, alignedInputLength * sizeof(uint8_t));
        pipe->InitBuffer(zeroBuf, alignedInputLength * sizeof(half));
        pipe->InitBuffer(outHalfBuf, alignedInputLength * sizeof(half));
    }

    __aicore__ inline void Process() {
        if (totalLength == 0U) {
            return;
        }
        const uint32_t blockDim = GetBlockNum();
        uint32_t blockStart = 0U;
        uint32_t blockEnd = 0U;
        if (useBaselineSplit != 0U) {
            blockStart = blockIdx * blockLength;
            blockEnd = MinValue(totalLength, blockStart + blockLength);
        } else {
            const uint32_t computeBlocks = CeilDiv(totalLength, dataBlockSize);
            blockStart = computeBlocks * blockIdx / blockDim * dataBlockSize;
            blockEnd = MinValue(
                totalLength,
                computeBlocks * (blockIdx + 1U) / blockDim * dataBlockSize);
        }
        if (blockStart >= blockEnd) {
            return;
        }
        LocalTensor<half> zeroLocal = zeroBuf.Get<half>();
        Duplicate(zeroLocal, static_cast<half>(0.0f), alignedInputLength);
        for (uint32_t offset = blockStart; offset < blockEnd; offset += tileLength) {
            const uint32_t validLength = MinValue(tileLength, blockEnd - offset);
            CopyIn(offset, validLength);
            Compute(validLength);
            CopyOut(offset, validLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t validLength) {
        LocalTensor<T> xLocal = inQueue.template AllocTensor<T>();
        if (IsAligned32<T>(offset, validLength)) {
            DataCopy(xLocal, xGm[offset], validLength);
        } else {
            DataCopyExtParams copyParams{
                static_cast<uint16_t>(1),
                static_cast<uint32_t>(validLength * sizeof(T)),
                0,
                0,
                0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        }
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t validLength) {
        LocalTensor<T> xLocal = inQueue.template DeQue<T>();
        LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
            const uint32_t cmpLength = CeilDiv(validLength * static_cast<uint32_t>(sizeof(T)), 256U) *
                (256U / static_cast<uint32_t>(sizeof(T)));
            Compare(maskLocal, xLocal, xLocal, CMPMODE::EQ, cmpLength);
        } else {
            const uint32_t cmpLength = CeilDiv(validLength, 64U) * 64U;
            LocalTensor<float> compareLocal = compareBuf.Get<float>();
            Cast(compareLocal, xLocal, RoundMode::CAST_NONE, cmpLength);
            Compare(maskLocal, compareLocal, compareLocal, CMPMODE::EQ, cmpLength);
        }
        inQueue.FreeTensor(xLocal);

        LocalTensor<half> zeroLocal = zeroBuf.Get<half>();
        LocalTensor<half> outHalf = outHalfBuf.Get<half>();
        Select(outHalf, maskLocal, zeroLocal, static_cast<half>(1.0f), SELMODE::VSEL_TENSOR_SCALAR_MODE, validLength);

        LocalTensor<uint8_t> yLocal = outQueue.template AllocTensor<uint8_t>();
        Cast(yLocal, outHalf, RoundMode::CAST_RINT, validLength);
        outQueue.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t validLength) {
        LocalTensor<uint8_t> yLocal = outQueue.template DeQue<uint8_t>();
        if (IsAligned32<uint8_t>(offset, validLength)) {
            DataCopy(yGm[offset], yLocal, validLength);
        } else {
            DataCopyExtParams copyParams{
                static_cast<uint16_t>(1),
                static_cast<uint32_t>(validLength * sizeof(uint8_t)),
                0,
                0,
                0};
            DataCopyPad(yGm[offset], yLocal, copyParams);
        }
        outQueue.FreeTensor(yLocal);
    }

private:
    TPipe* pipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<QuePosition::VECCALC> compareBuf;
    TBuf<QuePosition::VECCALC> maskBuf;
    TBuf<QuePosition::VECCALC> zeroBuf;
    TBuf<QuePosition::VECCALC> outHalfBuf;
    GlobalTensor<T> xGm;
    GlobalTensor<uint8_t> yGm;
    uint32_t totalLength = 0U;
    uint32_t blockLength = 0U;
    uint32_t dataBlockSize = 1U;
    uint32_t tileLength = 0U;
    uint32_t alignedInputLength = 0U;
    uint32_t alignedOutputLength = 0U;
    uint32_t useBaselineSplit = 0U;
    uint32_t blockIdx = 0U;
};

}  // namespace

extern "C" __global__ __aicore__ void is_nan(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    TPipe pipe;
    KernelIsNan<DTYPE_X> op;
    op.Init(
        x,
        y,
        tiling_data.totalLength,
        tiling_data.blockLength,
        tiling_data.dataBlockSize,
        tiling_data.tileLength,
        tiling_data.alignedInputLength,
        tiling_data.alignedOutputLength,
        tiling_data.useBaselineSplit,
        &pipe);
    op.Process();
}
