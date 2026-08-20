#include "kernel_operator.h"

#include <type_traits>

using namespace AscendC;

namespace {

constexpr uint32_t kAlignBytes = 512;

template <typename T>
__aicore__ inline T CastFillValue(float value)
{
    return static_cast<T>(value);
}

template <>
__aicore__ inline uint8_t CastFillValue<uint8_t>(float value)
{
    int32_t intValue = static_cast<int32_t>(value);
    if (intValue < 0) {
        intValue = 0;
    } else if (intValue > 255) {
        intValue = 255;
    }
    return static_cast<uint8_t>(intValue);
}

template <typename T>
class KernelFills {
public:
    __aicore__ inline KernelFills() {}

    __aicore__ inline void Init(GM_ADDR output, uint32_t blockOffset, uint32_t currentBlockLength,
                                uint32_t tileLength, float fillValue)
    {
        blockLength_ = currentBlockLength;
        tileLength_ = tileLength;
        fillValue_ = CastFillValue<T>(fillValue);
        outputGm_.SetGlobalBuffer((__gm__ T*)output + blockOffset, currentBlockLength);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < blockLength_; offset += tileLength_) {
            uint32_t currentTileLength = blockLength_ - offset;
            if (currentTileLength > tileLength_) {
                currentTileLength = tileLength_;
            }
            FillAndCopy(offset, currentTileLength);
        }
    }

private:
    __aicore__ inline void CopyToGm(uint32_t offset, LocalTensor<T>& outputLocal, uint32_t len)
    {
        if ((len * sizeof(T)) % kAlignBytes == 0) {
            DataCopy(outputGm_[offset], outputLocal, len);
        } else {
            DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
            DataCopyPad(outputGm_[offset], outputLocal, copyParams);
        }
    }

    __aicore__ inline void FillTensor(LocalTensor<T>& tensor, uint32_t len)
    {
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float> ||
                      std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t>) {
            constexpr uint32_t alignUnit = kAlignBytes / sizeof(T);
            uint32_t alignedLen = (len / alignUnit) * alignUnit;
            if (alignedLen > 0) {
                Duplicate(tensor, fillValue_, alignedLen);
            }
            for (uint32_t i = alignedLen; i < len; ++i) {
                tensor.SetValue(i, fillValue_);
            }
        } else {
            for (uint32_t i = 0; i < len; ++i) {
                tensor.SetValue(i, fillValue_);
            }
        }
    }

    __aicore__ inline void FillAndCopy(uint32_t offset, uint32_t len)
    {
        LocalTensor<T> outputLocal(TPosition::VECOUT, 0, tileLength_);
        FillTensor(outputLocal, len);
        CopyToGm(offset, outputLocal, len);
    }

private:
    GlobalTensor<T> outputGm_;
    uint32_t blockLength_ = 0;
    uint32_t tileLength_ = 0;
    T fillValue_ {};
};

class KernelFillsBf16 {
public:
    __aicore__ inline KernelFillsBf16() {}

    __aicore__ inline void Init(GM_ADDR output, uint32_t blockOffset, uint32_t currentBlockLength,
                                uint32_t tileLength, uint16_t fillValueBf16)
    {
        blockLength_ = currentBlockLength;
        fillValueBf16_ = fillValueBf16;
        tileLength_ = tileLength;
        outputBits_.SetGlobalBuffer((__gm__ int16_t*)output + blockOffset, currentBlockLength);
    }

    __aicore__ inline void Process()
    {
        const int16_t fillBits = static_cast<int16_t>(fillValueBf16_);
        for (uint32_t offset = 0; offset < blockLength_; offset += tileLength_) {
            uint32_t currentTileLength = blockLength_ - offset;
            if (currentTileLength > tileLength_) {
                currentTileLength = tileLength_;
            }
            LocalTensor<int16_t> outputLocal(TPosition::VECOUT, 0, tileLength_);
            constexpr uint32_t alignUnit = kAlignBytes / sizeof(int16_t);
            uint32_t alignedLen = (currentTileLength / alignUnit) * alignUnit;
            if (alignedLen > 0) {
                Duplicate(outputLocal, fillBits, alignedLen);
            }
            for (uint32_t i = alignedLen; i < currentTileLength; ++i) {
                outputLocal.SetValue(i, fillBits);
            }
            CopyToGm(offset, outputLocal, currentTileLength);
        }
    }

private:
    __aicore__ inline void CopyToGm(uint32_t offset, LocalTensor<int16_t>& outputLocal, uint32_t len)
    {
        if ((len * sizeof(int16_t)) % kAlignBytes == 0) {
            DataCopy(outputBits_[offset], outputLocal, len);
        } else {
            DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(int16_t)), 0, 0, 0};
            DataCopyPad(outputBits_[offset], outputLocal, copyParams);
        }
    }

    GlobalTensor<int16_t> outputBits_;
    uint32_t blockLength_ = 0;
    uint32_t tileLength_ = 0;
    uint16_t fillValueBf16_ = 0;
};

template <typename T>
class KernelFillsByte {
public:
    __aicore__ inline KernelFillsByte() {}

    __aicore__ inline void Init(GM_ADDR output, uint32_t blockOffset, uint32_t currentBlockLength,
                                uint32_t tileLength, T fillValue)
    {
        blockLength_ = currentBlockLength;
        fillValue_ = fillValue;
        const uint8_t fillByte = static_cast<uint8_t>(fillValue);
        repeatedHalf_ = static_cast<uint16_t>(fillByte) | (static_cast<uint16_t>(fillByte) << 8);
        halfTileLength_ = tileLength / 2U > 0 ? tileLength / 2U : 1U;
        outputHalfs_.SetGlobalBuffer((__gm__ int16_t*)output + blockOffset / 2U, currentBlockLength / 2U);
        outputBytes_.SetGlobalBuffer((__gm__ T*)output + blockOffset, currentBlockLength);
    }

    __aicore__ inline void Process()
    {
        const uint32_t halfCount = blockLength_ / 2U;
        const int16_t fillHalf = static_cast<int16_t>(repeatedHalf_);
        for (uint32_t offset = 0; offset < halfCount; offset += halfTileLength_) {
            uint32_t currentTileLength = halfCount - offset;
            if (currentTileLength > halfTileLength_) {
                currentTileLength = halfTileLength_;
            }
            LocalTensor<int16_t> halfLocal(TPosition::VECOUT, 0, halfTileLength_);
            constexpr uint32_t alignUnit = kAlignBytes / sizeof(uint16_t);
            uint32_t alignedLen = (currentTileLength / alignUnit) * alignUnit;
            if (alignedLen > 0) {
                Duplicate(halfLocal, fillHalf, alignedLen);
            }
            for (uint32_t i = alignedLen; i < currentTileLength; ++i) {
                halfLocal.SetValue(i, fillHalf);
            }
            CopyToGm(offset, halfLocal, currentTileLength);
        }

        const uint32_t tailStart = halfCount * 2U;
        for (uint32_t i = tailStart; i < blockLength_; ++i) {
            outputBytes_.SetValue(i, fillValue_);
        }
    }

private:
    __aicore__ inline void CopyToGm(uint32_t offset, LocalTensor<int16_t>& halfLocal, uint32_t len)
    {
        if ((len * sizeof(int16_t)) % kAlignBytes == 0) {
            DataCopy(outputHalfs_[offset], halfLocal, len);
        } else {
            DataCopyExtParams copyParams = {1, static_cast<uint32_t>(len * sizeof(int16_t)), 0, 0, 0};
            DataCopyPad(outputHalfs_[offset], halfLocal, copyParams);
        }
    }

    GlobalTensor<int16_t> outputHalfs_;
    GlobalTensor<T> outputBytes_;
    uint32_t blockLength_ = 0;
    uint32_t halfTileLength_ = 0;
    uint16_t repeatedHalf_ = 0;
    T fillValue_ = 0;
};

template <typename T>
__aicore__ inline void RunFills(GM_ADDR output, uint32_t blockLength, uint32_t lastBlockLength,
                                uint32_t tileLength, float fillValue)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockOffset = blockLength * blockIdx;
    uint32_t currentBlockLength = (blockIdx + 1 == GetBlockNum()) ? lastBlockLength : blockLength;
    KernelFills<T> kernel;
    kernel.Init(output, blockOffset, currentBlockLength, tileLength, fillValue);
    kernel.Process();
}

__aicore__ inline void RunFillsBf16(GM_ADDR output, uint32_t blockLength, uint32_t lastBlockLength,
                                    uint32_t tileLength, uint16_t fillValueBf16)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockOffset = blockLength * blockIdx;
    uint32_t currentBlockLength = (blockIdx + 1 == GetBlockNum()) ? lastBlockLength : blockLength;
    KernelFillsBf16 kernel;
    kernel.Init(output, blockOffset, currentBlockLength, tileLength, fillValueBf16);
    kernel.Process();
}

template <typename T>
__aicore__ inline void RunFillsByte(GM_ADDR output, uint32_t blockLength, uint32_t lastBlockLength,
                                    uint32_t tileLength, T fillValue)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockOffset = blockLength * blockIdx;
    uint32_t currentBlockLength = (blockIdx + 1 == GetBlockNum()) ? lastBlockLength : blockLength;
    KernelFillsByte<T> kernel;
    kernel.Init(output, blockOffset, currentBlockLength, tileLength, fillValue);
    kernel.Process();
}

}  // namespace

extern "C" __global__ __aicore__ void fills(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)input;
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    if constexpr (std::is_same_v<DTYPE_OUTPUT, bfloat16_t>) {
        RunFillsBf16(output, tilingData.blockLength, tilingData.lastBlockLength,
                     tilingData.tileLength, tilingData.fillValueBf16);
    } else if constexpr (std::is_same_v<DTYPE_OUTPUT, int8_t> || std::is_same_v<DTYPE_OUTPUT, uint8_t>) {
        RunFillsByte<DTYPE_OUTPUT>(output, tilingData.blockLength, tilingData.lastBlockLength,
                                   tilingData.tileLength, CastFillValue<DTYPE_OUTPUT>(tilingData.fillValue));
    } else {
        RunFills<DTYPE_OUTPUT>(output, tilingData.blockLength, tilingData.lastBlockLength,
                               tilingData.tileLength, tilingData.fillValue);
    }
}
