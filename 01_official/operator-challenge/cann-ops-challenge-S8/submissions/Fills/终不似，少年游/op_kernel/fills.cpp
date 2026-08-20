#include "kernel_operator.h"

#include <cstdint>
#include <type_traits>

using namespace AscendC;

namespace {

constexpr uint32_t kAlignBytes = 32;
constexpr uint32_t kPreferredAlignBytes = 512;

template <typename T>
struct GmStorePlan {
    uint32_t bytes = 0;
    uint32_t headElems = 0;
    uint32_t tailBytes = 0;
    bool usePreferredSplit = false;
    bool usePlainCopy = false;
};

template <typename T>
__aicore__ inline uint32_t AlignElems(uint32_t elems)
{
    const uint32_t alignElems = kAlignBytes / sizeof(T);
    return ((elems + alignElems - 1U) / alignElems) * alignElems;
}

template <typename T>
__aicore__ inline GmStorePlan<T> BuildGmStorePlan(uint32_t offsetElems, uint32_t lenElems, uint32_t preferredAlignBytes)
{
    GmStorePlan<T> plan;
    plan.bytes = lenElems * sizeof(T);
    const uint64_t offsetBytes = static_cast<uint64_t>(offsetElems) * sizeof(T);
    const bool offsetBlockAligned = (offsetBytes % kAlignBytes) == 0U;
    plan.usePlainCopy = offsetBlockAligned && ((plan.bytes % kAlignBytes) == 0U);

    const uint32_t alignBytes = preferredAlignBytes > kAlignBytes ? preferredAlignBytes : kAlignBytes;
    if (alignBytes > kAlignBytes &&
        offsetBlockAligned &&
        (offsetBytes % alignBytes) == 0U &&
        plan.bytes > alignBytes &&
        (plan.bytes % alignBytes) != 0U) {
        const uint32_t headBytes = (plan.bytes / alignBytes) * alignBytes;
        if (headBytes >= kAlignBytes) {
            plan.usePreferredSplit = true;
            plan.headElems = headBytes / sizeof(T);
            plan.tailBytes = plan.bytes - headBytes;
        }
    }
    return plan;
}

template <typename T>
__aicore__ inline void CopyOutPad(GlobalTensor<T>& outputGm, uint32_t offsetElems, LocalTensor<T>& outputLocal, uint32_t lenElems)
{
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(lenElems * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGm[offsetElems], outputLocal, copyParams);
}

template <typename T>
__aicore__ inline void CopyOutPlain(
    GlobalTensor<T>& outputGm,
    uint32_t offsetElems,
    LocalTensor<T>& outputLocal,
    uint32_t lenElems)
{
    DataCopyParams copyParams = {1, static_cast<uint16_t>(lenElems * sizeof(T) / kAlignBytes), 0, 0};
    DataCopy(outputGm[offsetElems], outputLocal, copyParams);
}

template <typename T>
__aicore__ inline void CopyOutAuto(
    GlobalTensor<T>& outputGm,
    uint32_t offsetElems,
    LocalTensor<T>& outputLocal,
    uint32_t lenElems,
    uint32_t preferredAlignBytes)
{
    const GmStorePlan<T> plan = BuildGmStorePlan<T>(offsetElems, lenElems, preferredAlignBytes);
    if (plan.usePreferredSplit) {
        CopyOutPlain(outputGm, offsetElems, outputLocal, plan.headElems);
        DataCopyExtParams tailParams = {1, plan.tailBytes, 0, 0, 0};
        DataCopyPad(outputGm[offsetElems + plan.headElems], outputLocal[plan.headElems], tailParams);
        return;
    }

    if (plan.usePlainCopy) {
        CopyOutPlain(outputGm, offsetElems, outputLocal, lenElems);
        return;
    }

    CopyOutPad(outputGm, offsetElems, outputLocal, lenElems);
}

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

template <typename StorageT>
class UniformFillWriter {
public:
    __aicore__ inline UniformFillWriter() {}

    __aicore__ inline void Init(
        GM_ADDR output,
        uint32_t blockOffset,
        uint32_t currentBlockLength,
        uint32_t tileLength,
        uint32_t preferredAlignBytes,
        StorageT fillPattern)
    {
        InitSocState();
        blockLength_ = currentBlockLength;
        if (blockLength_ == 0U) {
            return;
        }

        tileLength_ = tileLength == 0U ? 1U : tileLength;
        if (tileLength_ > blockLength_) {
            tileLength_ = blockLength_;
        }
        allocTileLength_ = AlignElems<StorageT>(tileLength_);
        preferredAlignBytes_ = preferredAlignBytes;
        fillPattern_ = fillPattern;
        fullTileCount_ = blockLength_ / tileLength_;
        tailLength_ = blockLength_ - fullTileCount_ * tileLength_;
        hasTail_ = tailLength_ != 0U;
        const uint64_t baseBytes = static_cast<uint64_t>(blockOffset) * sizeof(StorageT);
        fullTileUsePlainCopy_ = ((baseBytes % kAlignBytes) == 0U) &&
            (((static_cast<uint64_t>(tileLength_) * sizeof(StorageT)) % kAlignBytes) == 0U);
        outputGm_.SetGlobalBuffer((__gm__ StorageT*)output + blockOffset, currentBlockLength);
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ == 0U) {
            return;
        }

        LocalTensor<StorageT> fillTile(TPosition::VECOUT, 0, allocTileLength_);
        Duplicate(fillTile, fillPattern_, allocTileLength_);

        uint32_t offset = 0;
        if (fullTileUsePlainCopy_) {
            for (uint32_t idx = 0; idx < fullTileCount_; ++idx) {
                CopyOutPlain(outputGm_, offset, fillTile, tileLength_);
                offset += tileLength_;
            }
        } else {
            for (uint32_t idx = 0; idx < fullTileCount_; ++idx) {
                CopyOutAuto(outputGm_, offset, fillTile, tileLength_, preferredAlignBytes_);
                offset += tileLength_;
            }
        }
        if (hasTail_) {
            CopyOutAuto(outputGm_, offset, fillTile, tailLength_, preferredAlignBytes_);
        }
    }

private:
    GlobalTensor<StorageT> outputGm_;
    uint32_t blockLength_ = 0;
    uint32_t tileLength_ = 0;
    uint32_t allocTileLength_ = 0;
    uint32_t fullTileCount_ = 0;
    uint32_t tailLength_ = 0;
    uint32_t preferredAlignBytes_ = kPreferredAlignBytes;
    bool hasTail_ = false;
    bool fullTileUsePlainCopy_ = false;
    StorageT fillPattern_ {};
};

template <typename T>
class ByteFillWriter {
public:
    __aicore__ inline ByteFillWriter() {}

    __aicore__ inline void Init(
        GM_ADDR output,
        uint32_t blockOffset,
        uint32_t currentBlockLength,
        uint32_t tileLength,
        uint32_t preferredAlignBytes,
        T fillValue)
    {
        InitSocState();
        blockLength_ = currentBlockLength;
        if (blockLength_ == 0U) {
            return;
        }

        fillValue_ = fillValue;
        preferredAlignBytes_ = preferredAlignBytes;
        const uint8_t fillByte = static_cast<uint8_t>(fillValue);
        repeatedHalf_ = static_cast<int16_t>(
            static_cast<uint16_t>(fillByte) | (static_cast<uint16_t>(fillByte) << 8));
        mainHalfCount_ = blockLength_ / 2U;
        hasOddByte_ = (blockLength_ & 1U) != 0U;
        halfTileLength_ = tileLength / 2U;
        if (mainHalfCount_ > 0U) {
            if (halfTileLength_ == 0U) {
                halfTileLength_ = 1U;
            }
            if (halfTileLength_ > mainHalfCount_) {
                halfTileLength_ = mainHalfCount_;
            }
            allocHalfTileLength_ = AlignElems<int16_t>(halfTileLength_);
            fullHalfTileCount_ = mainHalfCount_ / halfTileLength_;
            tailHalfLength_ = mainHalfCount_ - fullHalfTileCount_ * halfTileLength_;
            hasHalfTail_ = tailHalfLength_ != 0U;
            fullHalfUsePlainCopy_ = ((blockOffset % kAlignBytes) == 0U) &&
                (((static_cast<uint64_t>(halfTileLength_) * sizeof(int16_t)) % kAlignBytes) == 0U);
            outputHalfs_.SetGlobalBuffer((__gm__ int16_t*)output + blockOffset / 2U, mainHalfCount_);
        }
        outputBytes_.SetGlobalBuffer((__gm__ T*)output + blockOffset, currentBlockLength);
    }

    __aicore__ inline void Process()
    {
        if (mainHalfCount_ > 0U) {
            LocalTensor<int16_t> fillTile(TPosition::VECOUT, 0, allocHalfTileLength_);
            Duplicate(fillTile, repeatedHalf_, allocHalfTileLength_);

            uint32_t offset = 0;
            if (fullHalfUsePlainCopy_) {
                for (uint32_t idx = 0; idx < fullHalfTileCount_; ++idx) {
                    CopyOutPlain(outputHalfs_, offset, fillTile, halfTileLength_);
                    offset += halfTileLength_;
                }
            } else {
                for (uint32_t idx = 0; idx < fullHalfTileCount_; ++idx) {
                    CopyOutAuto(outputHalfs_, offset, fillTile, halfTileLength_, preferredAlignBytes_);
                    offset += halfTileLength_;
                }
            }
            if (hasHalfTail_) {
                CopyOutAuto(outputHalfs_, offset, fillTile, tailHalfLength_, preferredAlignBytes_);
            }
        }

        if (hasOddByte_) {
            outputBytes_.SetValue(blockLength_ - 1U, fillValue_);
        }
    }

private:
    GlobalTensor<int16_t> outputHalfs_;
    GlobalTensor<T> outputBytes_;
    uint32_t blockLength_ = 0;
    uint32_t mainHalfCount_ = 0;
    uint32_t halfTileLength_ = 0;
    uint32_t allocHalfTileLength_ = 0;
    uint32_t fullHalfTileCount_ = 0;
    uint32_t tailHalfLength_ = 0;
    uint32_t preferredAlignBytes_ = kPreferredAlignBytes;
    bool hasHalfTail_ = false;
    bool hasOddByte_ = false;
    bool fullHalfUsePlainCopy_ = false;
    int16_t repeatedHalf_ = 0;
    T fillValue_ = 0;
};

template <typename StorageT>
__aicore__ inline void RunUniformFill(
    GM_ADDR output,
    uint32_t blockLength,
    uint32_t lastBlockLength,
    uint32_t tileLength,
    uint32_t preferredAlignBytes,
    StorageT fillPattern)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockOffset = blockLength * blockIdx;
    uint32_t currentBlockLength = (blockIdx + 1 == GetBlockNum()) ? lastBlockLength : blockLength;
    UniformFillWriter<StorageT> kernel;
    kernel.Init(output, blockOffset, currentBlockLength, tileLength, preferredAlignBytes, fillPattern);
    kernel.Process();
}

template <typename T>
__aicore__ inline void RunByteFill(
    GM_ADDR output,
    uint32_t blockLength,
    uint32_t lastBlockLength,
    uint32_t tileLength,
    uint32_t preferredAlignBytes,
    T fillValue)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockOffset = blockLength * blockIdx;
    uint32_t currentBlockLength = (blockIdx + 1 == GetBlockNum()) ? lastBlockLength : blockLength;
    ByteFillWriter<T> kernel;
    kernel.Init(output, blockOffset, currentBlockLength, tileLength, preferredAlignBytes, fillValue);
    kernel.Process();
}

}  // namespace

extern "C" __global__ __aicore__ void fills(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)input;
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    if constexpr (std::is_same_v<DTYPE_OUTPUT, bfloat16_t>) {
        RunUniformFill<int16_t>(output, tilingData.blockLength, tilingData.lastBlockLength,
                                tilingData.tileLength, tilingData.preferredAlignBytes,
                                static_cast<int16_t>(tilingData.fillValueBf16));
    } else if constexpr (std::is_same_v<DTYPE_OUTPUT, int8_t> || std::is_same_v<DTYPE_OUTPUT, uint8_t>) {
        RunByteFill<DTYPE_OUTPUT>(output, tilingData.blockLength, tilingData.lastBlockLength,
                                  tilingData.tileLength, tilingData.preferredAlignBytes,
                                  CastFillValue<DTYPE_OUTPUT>(tilingData.fillValue));
    } else {
        RunUniformFill<DTYPE_OUTPUT>(output, tilingData.blockLength, tilingData.lastBlockLength,
                                     tilingData.tileLength, tilingData.preferredAlignBytes,
                                     CastFillValue<DTYPE_OUTPUT>(tilingData.fillValue));
    }
}
