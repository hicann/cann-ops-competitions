#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t ABS_MASK_BUFFER_BYTES = 32;
constexpr uint32_t ABS_MASK_BUFFER_ELEMENTS = ABS_MASK_BUFFER_BYTES / sizeof(uint16_t);
constexpr uint32_t UINT16_ELEMENTS_PER_REPEAT = 128;

template <typename T>
struct IsNanBitTraits;

template <>
struct IsNanBitTraits<float> {
    using BitType = uint32_t;
    static constexpr uint32_t ABS_MASK = 0x7fffffffU;
    static constexpr uint32_t INF_BITS = 0x7f800000U;
};

template <>
struct IsNanBitTraits<half> {
    using BitType = uint16_t;
    static constexpr uint16_t ABS_MASK = 0x7fffU;
    static constexpr uint16_t INF_BITS = 0x7c00U;
};

template <>
struct IsNanBitTraits<bfloat16_t> {
    using BitType = uint16_t;
    static constexpr uint16_t ABS_MASK = 0x7fffU;
    static constexpr uint16_t INF_BITS = 0x7f80U;
};

template <typename T>
class KernelIsNan {
public:
    __aicore__ inline KernelIsNan() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t smallSize, uint32_t incSize,
                                uint32_t totalSize, uint16_t formerNum, TPipe* pipeIn) {
        pipe = pipeIn;

        uint32_t beginIndex = 0;
        if (GetBlockIdx() < formerNum) {
            size = smallSize + incSize;
            beginIndex = size * GetBlockIdx();
        } else {
            size = smallSize;
            beginIndex = size * GetBlockIdx() + formerNum * incSize;
        }
        uint32_t alignedTotal = smallSize * static_cast<uint32_t>(GetBlockNum()) + incSize * formerNum;
        if (GetBlockIdx() + 1 == GetBlockNum()) {
            size += totalSize - alignedTotal;
        }

        xGm.SetGlobalBuffer((__gm__ T*)x + beginIndex, size);
        yGm.SetGlobalBuffer((__gm__ uint8_t*)y + beginIndex, size);

        const uint32_t maxElementsPerIter = (std::is_same<T, float>::value ? 16 * 1024 : 32 * 1024) / BUFFER_NUM;
        nElementsPerIter = min(size, maxElementsPerIter);
        loopTimes = nElementsPerIter == 0 ? 0 : (size + nElementsPerIter - 1) / nElementsPerIter;
        nElementsPerIter = nElementsPerIter == 0 ? 1 : nElementsPerIter;

        uint32_t inputBytes = AlignUp(nElementsPerIter * sizeof(T), 32);
        uint32_t outputBytes = AlignUp(nElementsPerIter, 32);
        pipe->InitBuffer(inputBuf, BUFFER_NUM, inputBytes);
        pipe->InitBuffer(outputBuf, BUFFER_NUM, outputBytes);
        if constexpr (std::is_same<T, float>::value) {
            uint32_t tmpBytes = AlignUp(nElementsPerIter * sizeof(int16_t), 32);
            pipe->InitBuffer(tmpBuf, tmpBytes);
        }
        pipe->InitBuffer(absMaskBuf, ABS_MASK_BUFFER_BYTES);
        if constexpr (std::is_same<T, float>::value) {
            LocalTensor<uint16_t> absMask = absMaskBuf.Get<uint16_t>();
            Duplicate(absMask, static_cast<uint16_t>(0xffffU), ABS_MASK_BUFFER_ELEMENTS);
            uint64_t highHalfMask[2] = {0xAAAAULL, 0};
            Duplicate(absMask, static_cast<uint16_t>(0x7fffU), highHalfMask, 1, 1, 8);
        } else {
            Duplicate(absMaskBuf.Get<uint16_t>(), static_cast<uint16_t>(IsNanBitTraits<T>::ABS_MASK), ABS_MASK_BUFFER_ELEMENTS);
        }
    }

    __aicore__ inline void Process() {
        if (loopTimes == 0) {
            return;
        }

        uint32_t offset = 0;
        uint32_t iterSize = min(nElementsPerIter, size);
        CopyIn(offset, iterSize);

        for (uint32_t i = 0; i < loopTimes; ++i) {
            ComputeIter(iterSize);
            CopyOut(offset, iterSize);
            uint32_t nextOffset = offset + iterSize;
            uint32_t nextIterSize = 0;
            // Keep the second input buffer busy while the current tile is computed.
            if (i + 1 < loopTimes) {
                nextIterSize = min(nElementsPerIter, size - nextOffset);
                CopyIn(nextOffset, nextIterSize);
            }

            offset = nextOffset;
            iterSize = nextIterSize;
        }
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) {
        return (value + align - 1) / align * align;
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t iterSize) {
        uint16_t blockCount = 1;
        uint32_t inputBytes = iterSize * sizeof(T);
        DataCopyExtParams copyParams{blockCount, inputBytes, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        LocalTensor<T> xLocal = inputBuf.AllocTensor<T>();
        if ((inputBytes & 31U) == 0) {
            DataCopy(xLocal, xGm[offset], iterSize);
        } else {
            DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        }
        inputBuf.EnQue<T>(xLocal);
    }

    __aicore__ inline void ComputeIter(uint32_t iterSize) {
        LocalTensor<T> xLocal = inputBuf.DeQue<T>();
        LocalTensor<uint8_t> yLocal = outputBuf.AllocTensor<uint8_t>();
        if constexpr (std::is_same<T, float>::value) {
            LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();
            ComputeFloat(yLocal, xLocal, tmpLocal, iterSize);
        } else {
            Compute16(yLocal, xLocal, iterSize);
        }

        inputBuf.FreeTensor<T>(xLocal);
        outputBuf.EnQue<uint8_t>(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t iterSize) {
        LocalTensor<uint8_t> yLocal = outputBuf.DeQue<uint8_t>();
        uint16_t blockCount = 1;
        DataCopyExtParams storeParams{blockCount, iterSize, 0, 0, 0};
        if ((iterSize & 31U) == 0) {
            DataCopy(yGm[offset], yLocal, iterSize);
        } else {
            DataCopyPad(yGm[offset], yLocal, storeParams);
        }
        outputBuf.FreeTensor<uint8_t>(yLocal);
    }

    __aicore__ inline void Compute16(const LocalTensor<uint8_t>& yLocal, const LocalTensor<T>& xLocal,
                                     uint32_t iterSize) {
        using BitTraits = IsNanBitTraits<T>;
        LocalTensor<uint16_t> absBits = xLocal.template ReinterpretCast<uint16_t>();
        ApplyAbsMask(absBits, iterSize);

        // Positive diff means NaN; after reinterpret as half, CAST_CEIL maps any tiny positive to 1.
        LocalTensor<int16_t> diff = absBits.template ReinterpretCast<int16_t>();
        Adds(diff, diff, static_cast<int16_t>(-static_cast<int32_t>(BitTraits::INF_BITS)), iterSize);
        Maxs(diff, diff, static_cast<int16_t>(0), iterSize);
        Cast(yLocal, diff.template ReinterpretCast<half>(), RoundMode::CAST_CEIL, iterSize);
    }

    __aicore__ inline void ComputeFloat(const LocalTensor<uint8_t>& yLocal, const LocalTensor<T>& xLocal,
                                        const LocalTensor<uint8_t>& tmpLocal, uint32_t iterSize) {
        using BitTraits = IsNanBitTraits<float>;

        LocalTensor<uint16_t> absBits = xLocal.template ReinterpretCast<uint16_t>();
        ApplyAbsMask(absBits, iterSize * 2);

        // Positive diff is a float subnormal after reinterpret; CAST_CEIL maps it to int16 1.
        LocalTensor<int32_t> diff = xLocal.template ReinterpretCast<int32_t>();
        Adds(diff, diff, static_cast<int32_t>(-static_cast<int32_t>(BitTraits::INF_BITS)), iterSize);
        Maxs(diff, diff, static_cast<int32_t>(0), iterSize);

        LocalTensor<int16_t> resultInt16 = tmpLocal.template ReinterpretCast<int16_t>();
        Cast(resultInt16, diff.template ReinterpretCast<float>(), RoundMode::CAST_CEIL, iterSize);
        Cast(yLocal, resultInt16.template ReinterpretCast<half>(), RoundMode::CAST_CEIL, iterSize);
    }

    __aicore__ inline void ApplyAbsMask(const LocalTensor<uint16_t>& absBits, uint32_t elementCount) {
        LocalTensor<uint16_t> absMask = absMaskBuf.Get<uint16_t>();
        BinaryRepeatParams repeatParams;
        repeatParams.src1BlkStride = 0;
        repeatParams.src1RepStride = 0;

        uint32_t repeatTimes = elementCount / UINT16_ELEMENTS_PER_REPEAT;
        uint32_t tailSize = elementCount % UINT16_ELEMENTS_PER_REPEAT;
        if (repeatTimes > 0) {
            And(absBits, absBits, absMask, static_cast<uint64_t>(UINT16_ELEMENTS_PER_REPEAT),
                static_cast<uint8_t>(repeatTimes), repeatParams);
        }
        if (tailSize > 0) {
            uint32_t tailOffset = repeatTimes * UINT16_ELEMENTS_PER_REPEAT;
            And(absBits[tailOffset], absBits[tailOffset], absMask, static_cast<uint64_t>(tailSize),
                static_cast<uint8_t>(1), repeatParams);
        }
    }

private:
    TPipe* pipe;
    GlobalTensor<T> xGm;
    GlobalTensor<uint8_t> yGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputBuf;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> absMaskBuf;
    uint32_t size;
    uint32_t loopTimes;
    uint32_t nElementsPerIter;
};

template <typename T>
__aicore__ inline void RunIsNan(GM_ADDR x, GM_ADDR y, uint32_t smallSize, uint32_t incSize,
                                uint32_t totalSize, uint16_t formerNum) {
    TPipe pipe;
    KernelIsNan<T> op;
    op.Init(x, y, smallSize, incSize, totalSize, formerNum, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void is_nan(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    if (tilingData.totalSize == 0) {
        return;
    }

    if (TILING_KEY_IS(1)) {
        RunIsNan<half>(x, y, tilingData.smallSize, tilingData.incSize, tilingData.totalSize, tilingData.formerNum);
    } else if (TILING_KEY_IS(2)) {
        RunIsNan<bfloat16_t>(x, y, tilingData.smallSize, tilingData.incSize, tilingData.totalSize,
                             tilingData.formerNum);
    } else if (TILING_KEY_IS(3)) {
        RunIsNan<float>(x, y, tilingData.smallSize, tilingData.incSize, tilingData.totalSize, tilingData.formerNum);
    }
}