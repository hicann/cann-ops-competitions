// Kernel侧核函数实现
#include "kernel_operator.h"

#include "addcmul_tiling.h"
#include "tiling_key_addcmul.h"

using namespace AscendC;

constexpr int32_t ADDCMUL_BUFFER_NUM = 1;
constexpr int32_t ADDCMUL_VECTOR_BUFFER_NUM = 2;

__aicore__ inline uint64_t MinU64(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}

__aicore__ inline float AddcmulEval(float inputData, float x1, float x2, float value)
{
    return inputData + x1 * x2 * value;
}

__aicore__ inline half AddcmulEval(half inputData, half x1, half x2, half value)
{
    const float res = static_cast<float>(inputData) +
                      static_cast<float>(x1) * static_cast<float>(x2) * static_cast<float>(value);
    return static_cast<half>(res);
}

__aicore__ inline int32_t AddcmulEval(int32_t inputData, int32_t x1, int32_t x2, int32_t value)
{
    const int64_t res = static_cast<int64_t>(inputData) +
                        static_cast<int64_t>(x1) * static_cast<int64_t>(x2) * static_cast<int64_t>(value);
    return static_cast<int32_t>(res);
}

__aicore__ inline int8_t AddcmulEval(int8_t inputData, int8_t x1, int8_t x2, int8_t value)
{
    const int32_t res = static_cast<int32_t>(inputData) +
                        static_cast<int32_t>(x1) * static_cast<int32_t>(x2) * static_cast<int32_t>(value);
    return static_cast<int8_t>(res);
}

__aicore__ inline float AddcmulEvalValueOne(float inputData, float x1, float x2)
{
    return inputData + x1 * x2;
}

__aicore__ inline half AddcmulEvalValueOne(half inputData, half x1, half x2)
{
    const float res = static_cast<float>(inputData) + static_cast<float>(x1) * static_cast<float>(x2);
    return static_cast<half>(res);
}

__aicore__ inline int32_t AddcmulEvalValueOne(int32_t inputData, int32_t x1, int32_t x2)
{
    const int64_t res = static_cast<int64_t>(inputData) + static_cast<int64_t>(x1) * static_cast<int64_t>(x2);
    return static_cast<int32_t>(res);
}

__aicore__ inline int8_t AddcmulEvalValueOne(int8_t inputData, int8_t x1, int8_t x2)
{
    const int32_t res = static_cast<int32_t>(inputData) + static_cast<int32_t>(x1) * static_cast<int32_t>(x2);
    return static_cast<int8_t>(res);
}

__aicore__ inline bool IsValueOne(float value)
{
    return value == 1.0f;
}

__aicore__ inline bool IsValueOne(half value)
{
    return static_cast<float>(value) == 1.0f;
}

__aicore__ inline bool IsValueOne(int32_t value)
{
    return value == 1;
}

__aicore__ inline bool IsValueOne(int8_t value)
{
    return value == 1;
}


__aicore__ inline float ScaledMulScalar(float x1, float x2, float value)
{
    return x1 * x2 * value;
}

__aicore__ inline half ScaledMulScalar(half x1, half x2, half value)
{
    const float res = static_cast<float>(x1) * static_cast<float>(x2) * static_cast<float>(value);
    return static_cast<half>(res);
}

__aicore__ inline int32_t ScaledMulScalar(int32_t x1, int32_t x2, int32_t value)
{
    const int64_t res = static_cast<int64_t>(x1) * static_cast<int64_t>(x2) * static_cast<int64_t>(value);
    return static_cast<int32_t>(res);
}

__aicore__ inline int8_t ScaledMulScalar(int8_t x1, int8_t x2, int8_t value)
{
    const int32_t res = static_cast<int32_t>(x1) * static_cast<int32_t>(x2) * static_cast<int32_t>(value);
    return static_cast<int8_t>(res);
}

template <class T>
class KernelAddcmulScalarBase {
public:
    __aicore__ inline KernelAddcmulScalarBase() {}

    __aicore__ inline void InitBase(GM_ADDR inputData, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
                                    const AddcmulTilingData &tiling)
    {
        inputGm.SetGlobalBuffer((__gm__ T *)inputData);
        x1Gm.SetGlobalBuffer((__gm__ T *)x1);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2);
        valueGm.SetGlobalBuffer((__gm__ T *)value);
        yGm.SetGlobalBuffer((__gm__ T *)y);

        length = tiling.length;
        perCoreElements = tiling.perCoreElements;
        dimNum = tiling.dimNum;
        sameShape = tiling.sameShape;
        tileLength = tiling.tileLength;
        inputMode = tiling.inputMode;
        x1Mode = tiling.x1Mode;
        x2Mode = tiling.x2Mode;
        inputRepeat = tiling.inputRepeat;
        x1Repeat = tiling.x1Repeat;
        x2Repeat = tiling.x2Repeat;
        inputDataLength = tiling.inputDataLength;
        x1DataLength = tiling.x1DataLength;
        x2DataLength = tiling.x2DataLength;
        hasGenericMode = (inputMode == ADDCMUL_MODE_GENERIC ||
                          x1Mode == ADDCMUL_MODE_GENERIC ||
                          x2Mode == ADDCMUL_MODE_GENERIC);
        if (hasGenericMode) {
            for (uint32_t i = 0; i < dimNum && i < ADDCMUL_MAX_DIMS; ++i) {
                outShape[i] = tiling.outShape[i];
                inputStrides[i] = tiling.inputStrides[i];
                x1Strides[i] = tiling.x1Strides[i];
                x2Strides[i] = tiling.x2Strides[i];
                coords[i] = 0;
            }
        }
    }

protected:
    __aicore__ inline bool NoGenericMode() const
    {
        return !hasGenericMode;
    }

    __aicore__ inline uint64_t FastOffsetByMode(uint32_t mode, uint64_t linear,
                                                 uint64_t repeat, uint64_t dataLength) const
    {
        if (mode == ADDCMUL_MODE_LINEAR) {
            return linear;
        }
        if (mode == ADDCMUL_MODE_SCALAR) {
            return 0;
        }
        if (mode == ADDCMUL_MODE_SEGMENT) {
            const uint64_t safeRepeat = (repeat == 0) ? 1 : repeat;
            const uint64_t safeDataLength = (dataLength == 0) ? 1 : dataLength;
            return (linear / safeRepeat) % safeDataLength;
        }
        return 0;
    }

    __aicore__ inline uint64_t OffsetByMode(uint32_t mode, uint64_t linear, uint64_t genericOffset,
                                             uint64_t repeat, uint64_t dataLength) const
    {
        if (mode == ADDCMUL_MODE_GENERIC) {
            return genericOffset;
        }
        return FastOffsetByMode(mode, linear, repeat, dataLength);
    }

    __aicore__ inline void InitGenericOffsets(uint64_t linearIndex)
    {
        inputOffset = 0;
        x1Offset = 0;
        x2Offset = 0;
        uint64_t tmp = linearIndex;
        for (int32_t dim = static_cast<int32_t>(dimNum) - 1; dim >= 0; --dim) {
            const uint64_t extent = outShape[dim];
            const uint64_t coord = (extent == 0) ? 0 : (tmp % extent);
            coords[dim] = coord;
            tmp = (extent == 0) ? tmp : (tmp / extent);
            inputOffset += coord * inputStrides[dim];
            x1Offset += coord * x1Strides[dim];
            x2Offset += coord * x2Strides[dim];
        }
    }

    __aicore__ inline void AdvanceGenericOffsets()
    {
        for (int32_t dim = static_cast<int32_t>(dimNum) - 1; dim >= 0; --dim) {
            const uint64_t next = coords[dim] + 1;
            if (next < outShape[dim]) {
                coords[dim] = next;
                inputOffset += inputStrides[dim];
                x1Offset += x1Strides[dim];
                x2Offset += x2Strides[dim];
                break;
            }
            inputOffset -= coords[dim] * inputStrides[dim];
            x1Offset -= coords[dim] * x1Strides[dim];
            x2Offset -= coords[dim] * x2Strides[dim];
            coords[dim] = 0;
        }
    }

    __aicore__ inline uint64_t InitFastOffset(uint32_t mode, uint64_t linear,
                                             uint64_t repeat, uint64_t dataLength,
                                             uint64_t &repeatPos) const
    {
        repeatPos = 0;
        if (mode == ADDCMUL_MODE_LINEAR) {
            return linear;
        }
        if (mode == ADDCMUL_MODE_SCALAR) {
            return 0;
        }
        const uint64_t safeRepeat = repeat == 0 ? 1 : repeat;
        const uint64_t safeDataLength = dataLength == 0 ? 1 : dataLength;
        if (safeRepeat == 1) {
            return linear % safeDataLength;
        }
        repeatPos = linear % safeRepeat;
        return (linear / safeRepeat) % safeDataLength;
    }

    __aicore__ inline void AdvanceFastOffset(uint32_t mode, uint64_t repeat, uint64_t dataLength,
                                             uint64_t &offset, uint64_t &repeatPos) const
    {
        if (mode == ADDCMUL_MODE_LINEAR) {
            ++offset;
            return;
        }
        if (mode == ADDCMUL_MODE_SCALAR) {
            return;
        }
        const uint64_t safeRepeat = repeat == 0 ? 1 : repeat;
        const uint64_t safeDataLength = dataLength == 0 ? 1 : dataLength;
        if (safeRepeat == 1) {
            ++offset;
            if (offset >= safeDataLength) {
                offset = 0;
            }
            return;
        }
        ++repeatPos;
        if (repeatPos >= safeRepeat) {
            repeatPos = 0;
            ++offset;
            if (offset >= safeDataLength) {
                offset = 0;
            }
        }
    }

    __aicore__ inline void ScalarLoopAllLinear(uint64_t start, uint64_t end, T valueScalar)
    {
        if (IsValueOne(valueScalar)) {
            for (uint64_t idx = start; idx < end; ++idx) {
                yGm.SetValue(idx, AddcmulEvalValueOne(inputGm.GetValue(idx), x1Gm.GetValue(idx), x2Gm.GetValue(idx)));
            }
        } else {
            for (uint64_t idx = start; idx < end; ++idx) {
                yGm.SetValue(idx, AddcmulEval(inputGm.GetValue(idx), x1Gm.GetValue(idx),
                                              x2Gm.GetValue(idx), valueScalar));
            }
        }
    }

    __aicore__ inline void ScalarLoopX2Scalar(uint64_t start, uint64_t end, T valueScalar)
    {
        const T x2Scalar = x2Gm.GetValue(0);
        if (IsValueOne(valueScalar)) {
            for (uint64_t idx = start; idx < end; ++idx) {
                yGm.SetValue(idx, AddcmulEvalValueOne(inputGm.GetValue(idx), x1Gm.GetValue(idx), x2Scalar));
            }
        } else {
            for (uint64_t idx = start; idx < end; ++idx) {
                yGm.SetValue(idx, AddcmulEval(inputGm.GetValue(idx), x1Gm.GetValue(idx), x2Scalar, valueScalar));
            }
        }
    }

    __aicore__ inline void ScalarLoopX1Scalar(uint64_t start, uint64_t end, T valueScalar)
    {
        const T x1Scalar = x1Gm.GetValue(0);
        if (IsValueOne(valueScalar)) {
            for (uint64_t idx = start; idx < end; ++idx) {
                yGm.SetValue(idx, AddcmulEvalValueOne(inputGm.GetValue(idx), x1Scalar, x2Gm.GetValue(idx)));
            }
        } else {
            for (uint64_t idx = start; idx < end; ++idx) {
                yGm.SetValue(idx, AddcmulEval(inputGm.GetValue(idx), x1Scalar, x2Gm.GetValue(idx), valueScalar));
            }
        }
    }

    __aicore__ inline void ScalarLoopBothMulScalar(uint64_t start, uint64_t end, T valueScalar)
    {
        const T x1Scalar = x1Gm.GetValue(0);
        const T x2Scalar = x2Gm.GetValue(0);
        if (IsValueOne(valueScalar)) {
            for (uint64_t idx = start; idx < end; ++idx) {
                yGm.SetValue(idx, AddcmulEvalValueOne(inputGm.GetValue(idx), x1Scalar, x2Scalar));
            }
        } else {
            for (uint64_t idx = start; idx < end; ++idx) {
                yGm.SetValue(idx, AddcmulEval(inputGm.GetValue(idx), x1Scalar, x2Scalar, valueScalar));
            }
        }
    }

    __aicore__ inline void ScalarLoop(uint64_t start, uint64_t end, T valueScalar)
    {
        if (sameShape == 1U) {
            ScalarLoopAllLinear(start, end, valueScalar);
            return;
        }
        if (inputMode == ADDCMUL_MODE_LINEAR && x1Mode == ADDCMUL_MODE_LINEAR && x2Mode == ADDCMUL_MODE_SCALAR) {
            ScalarLoopX2Scalar(start, end, valueScalar);
            return;
        }
        if (inputMode == ADDCMUL_MODE_LINEAR && x1Mode == ADDCMUL_MODE_SCALAR && x2Mode == ADDCMUL_MODE_LINEAR) {
            ScalarLoopX1Scalar(start, end, valueScalar);
            return;
        }
        if (inputMode == ADDCMUL_MODE_LINEAR && x1Mode == ADDCMUL_MODE_SCALAR && x2Mode == ADDCMUL_MODE_SCALAR) {
            ScalarLoopBothMulScalar(start, end, valueScalar);
            return;
        }
        if (NoGenericMode()) {
            uint64_t inputRepeatPos = 0;
            uint64_t x1RepeatPos = 0;
            uint64_t x2RepeatPos = 0;
            uint64_t inputIdx = InitFastOffset(inputMode, start, inputRepeat, inputDataLength, inputRepeatPos);
            uint64_t x1Idx = InitFastOffset(x1Mode, start, x1Repeat, x1DataLength, x1RepeatPos);
            uint64_t x2Idx = InitFastOffset(x2Mode, start, x2Repeat, x2DataLength, x2RepeatPos);

            if (IsValueOne(valueScalar)) {
                for (uint64_t idx = start; idx < end; ++idx) {
                    yGm.SetValue(idx, AddcmulEvalValueOne(inputGm.GetValue(inputIdx), x1Gm.GetValue(x1Idx),
                                                          x2Gm.GetValue(x2Idx)));
                    AdvanceFastOffset(inputMode, inputRepeat, inputDataLength, inputIdx, inputRepeatPos);
                    AdvanceFastOffset(x1Mode, x1Repeat, x1DataLength, x1Idx, x1RepeatPos);
                    AdvanceFastOffset(x2Mode, x2Repeat, x2DataLength, x2Idx, x2RepeatPos);
                }
            } else {
                for (uint64_t idx = start; idx < end; ++idx) {
                    yGm.SetValue(idx, AddcmulEval(inputGm.GetValue(inputIdx), x1Gm.GetValue(x1Idx),
                                                  x2Gm.GetValue(x2Idx), valueScalar));
                    AdvanceFastOffset(inputMode, inputRepeat, inputDataLength, inputIdx, inputRepeatPos);
                    AdvanceFastOffset(x1Mode, x1Repeat, x1DataLength, x1Idx, x1RepeatPos);
                    AdvanceFastOffset(x2Mode, x2Repeat, x2DataLength, x2Idx, x2RepeatPos);
                }
            }
            return;
        }

        InitGenericOffsets(start);
        if (IsValueOne(valueScalar)) {
            for (uint64_t idx = start; idx < end; ++idx) {
                const uint64_t inputIdx = OffsetByMode(inputMode, idx, inputOffset, inputRepeat, inputDataLength);
                const uint64_t x1Idx = OffsetByMode(x1Mode, idx, x1Offset, x1Repeat, x1DataLength);
                const uint64_t x2Idx = OffsetByMode(x2Mode, idx, x2Offset, x2Repeat, x2DataLength);
                yGm.SetValue(idx, AddcmulEvalValueOne(inputGm.GetValue(inputIdx), x1Gm.GetValue(x1Idx),
                                                      x2Gm.GetValue(x2Idx)));
                AdvanceGenericOffsets();
            }
        } else {
            for (uint64_t idx = start; idx < end; ++idx) {
                const uint64_t inputIdx = OffsetByMode(inputMode, idx, inputOffset, inputRepeat, inputDataLength);
                const uint64_t x1Idx = OffsetByMode(x1Mode, idx, x1Offset, x1Repeat, x1DataLength);
                const uint64_t x2Idx = OffsetByMode(x2Mode, idx, x2Offset, x2Repeat, x2DataLength);
                yGm.SetValue(idx, AddcmulEval(inputGm.GetValue(inputIdx), x1Gm.GetValue(x1Idx),
                                              x2Gm.GetValue(x2Idx), valueScalar));
                AdvanceGenericOffsets();
            }
        }
    }

protected:
    GlobalTensor<T> inputGm;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> valueGm;
    GlobalTensor<T> yGm;

    uint64_t length;
    uint64_t perCoreElements;
    uint32_t dimNum;
    uint32_t sameShape;
    uint32_t tileLength;
    uint32_t inputMode;
    uint32_t x1Mode;
    uint32_t x2Mode;
    uint64_t inputRepeat;
    uint64_t x1Repeat;
    uint64_t x2Repeat;
    uint64_t inputDataLength;
    uint64_t x1DataLength;
    uint64_t x2DataLength;
    bool hasGenericMode;

    uint64_t outShape[ADDCMUL_MAX_DIMS];
    uint64_t inputStrides[ADDCMUL_MAX_DIMS];
    uint64_t x1Strides[ADDCMUL_MAX_DIMS];
    uint64_t x2Strides[ADDCMUL_MAX_DIMS];
    uint64_t coords[ADDCMUL_MAX_DIMS];
    uint64_t inputOffset;
    uint64_t x1Offset;
    uint64_t x2Offset;
};

template <class T>
class KernelAddcmul : public KernelAddcmulScalarBase<T> {
public:
    __aicore__ inline KernelAddcmul() {}

    __aicore__ inline void Init(GM_ADDR inputData, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
                                const AddcmulTilingData &tiling)
    {
        this->InitBase(inputData, x1, x2, value, y, tiling);
        const uint32_t blockElems = ADDCMUL_DATA_BLOCK_BYTES / sizeof(T);
        canSafeUbPath = CanUseSafeUbPath(blockElems);
        if (this->tileLength > 0 && canSafeUbPath) {
            pipe.InitBuffer(inputQueue, ADDCMUL_BUFFER_NUM, this->tileLength * sizeof(T));
            pipe.InitBuffer(x1Queue, ADDCMUL_BUFFER_NUM, this->tileLength * sizeof(T));
            pipe.InitBuffer(x2Queue, ADDCMUL_BUFFER_NUM, this->tileLength * sizeof(T));
            pipe.InitBuffer(outQueue, ADDCMUL_BUFFER_NUM, this->tileLength * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        if (this->length == 0 || this->perCoreElements == 0) {
            return;
        }
        const uint64_t start = static_cast<uint64_t>(GetBlockIdx()) * this->perCoreElements;
        if (start >= this->length) {
            return;
        }
        uint64_t end = start + this->perCoreElements;
        if (end > this->length) {
            end = this->length;
        }

        const T valueScalar = this->valueGm.GetValue(0);
        const uint32_t blockElems = ADDCMUL_DATA_BLOCK_BYTES / sizeof(T);
        if (this->tileLength > 0 && canSafeUbPath) {
            ProcessSafeUb(start, end, valueScalar, blockElems);
        } else {
            this->ScalarLoop(start, end, valueScalar);
            DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(this->yGm);
        }
    }

private:
    __aicore__ inline bool IsLinearOrScalar(uint32_t mode) const
    {
        return mode == ADDCMUL_MODE_LINEAR || mode == ADDCMUL_MODE_SCALAR;
    }

    __aicore__ inline bool IsSafeSegment(uint32_t mode, uint64_t repeat, uint64_t dataLength,
                                         uint32_t blockElems) const
    {
        if (mode != ADDCMUL_MODE_SEGMENT) {
            return false;
        }
        if (repeat > 1) {
            return true;
        }
        return dataLength >= blockElems && (dataLength % static_cast<uint64_t>(blockElems)) == 0;
    }

    __aicore__ inline bool IsSafeUbMode(uint32_t mode, uint64_t repeat, uint64_t dataLength,
                                        uint32_t blockElems) const
    {
        return IsLinearOrScalar(mode) || IsSafeSegment(mode, repeat, dataLength, blockElems);
    }

    __aicore__ inline bool CanUseSafeUbPath(uint32_t blockElems) const
    {
        return IsSafeUbMode(this->inputMode, this->inputRepeat, this->inputDataLength, blockElems) &&
               IsSafeUbMode(this->x1Mode, this->x1Repeat, this->x1DataLength, blockElems) &&
               IsSafeUbMode(this->x2Mode, this->x2Repeat, this->x2DataLength, blockElems);
    }

    __aicore__ inline void LimitBySegment(uint32_t mode, uint64_t repeat, uint64_t dataLength, uint64_t linear,
                                          uint64_t &limit) const
    {
        if (mode != ADDCMUL_MODE_SEGMENT || repeat > 1) {
            return;
        }
        const uint64_t pos = linear % dataLength;
        const uint64_t remain = dataLength - pos;
        if (remain < limit) {
            limit = remain;
        }
    }

    __aicore__ inline uint32_t CalcUbLen(uint64_t linear, uint64_t remaining,
                                         uint32_t blockElems) const
    {
        uint64_t limit = this->tileLength;
        if (limit > remaining) {
            limit = remaining;
        }
        LimitBySegment(this->inputMode, this->inputRepeat, this->inputDataLength, linear, limit);
        LimitBySegment(this->x1Mode, this->x1Repeat, this->x1DataLength, linear, limit);
        LimitBySegment(this->x2Mode, this->x2Repeat, this->x2DataLength, linear, limit);
        limit = (limit / static_cast<uint64_t>(blockElems)) * static_cast<uint64_t>(blockElems);
        return static_cast<uint32_t>(limit);
    }

    __aicore__ inline void FillLocalScalar(LocalTensor<T> local, T scalar, uint32_t count)
    {
        for (uint32_t i = 0; i < count; ++i) {
            local.SetValue(i, scalar);
        }
    }

    __aicore__ inline void FillSegmentRepeat(LocalTensor<T> local, const GlobalTensor<T> &gm,
                                             uint64_t linear, uint64_t repeat, uint64_t dataLength,
                                             uint32_t count)
    {
        const uint64_t safeRepeat = repeat == 0 ? 1 : repeat;
        const uint64_t safeDataLength = dataLength == 0 ? 1 : dataLength;
        uint32_t filled = 0;
        uint64_t repeatPos = linear % safeRepeat;
        uint64_t dataIdx = (linear / safeRepeat) % safeDataLength;
        while (filled < count) {
            uint64_t seg = safeRepeat - repeatPos;
            if (seg > static_cast<uint64_t>(count - filled)) {
                seg = static_cast<uint64_t>(count - filled);
            }
            const T scalar = gm.GetValue(dataIdx);
            for (uint32_t i = 0; i < static_cast<uint32_t>(seg); ++i) {
                local.SetValue(filled + i, scalar);
            }
            filled += static_cast<uint32_t>(seg);
            repeatPos = 0;
            ++dataIdx;
            if (dataIdx >= safeDataLength) {
                dataIdx = 0;
            }
        }
    }

    __aicore__ inline void CopyTensorByMode(LocalTensor<T> local, const GlobalTensor<T> &gm,
                                            uint32_t mode, uint64_t linear, uint64_t repeat,
                                            uint64_t dataLength, uint32_t count)
    {
        if (mode == ADDCMUL_MODE_SCALAR) {
            FillLocalScalar(local, gm.GetValue(0), count);
        } else if (mode == ADDCMUL_MODE_SEGMENT && repeat > 1) {
            FillSegmentRepeat(local, gm, linear, repeat, dataLength, count);
        } else {
            const uint64_t gmOffset = this->FastOffsetByMode(mode, linear, repeat, dataLength);
            DataCopy(local, gm[gmOffset], count);
        }
    }

    __aicore__ inline void CopyInByMode(uint64_t offset, uint32_t count)
    {
        LocalTensor<T> inputLocal = inputQueue.template AllocTensor<T>();
        LocalTensor<T> x1Local = x1Queue.template AllocTensor<T>();
        LocalTensor<T> x2Local = x2Queue.template AllocTensor<T>();

        CopyTensorByMode(inputLocal, this->inputGm, this->inputMode, offset,
                         this->inputRepeat, this->inputDataLength, count);
        CopyTensorByMode(x1Local, this->x1Gm, this->x1Mode, offset,
                         this->x1Repeat, this->x1DataLength, count);
        CopyTensorByMode(x2Local, this->x2Gm, this->x2Mode, offset,
                         this->x2Repeat, this->x2DataLength, count);

        inputQueue.EnQue(inputLocal);
        x1Queue.EnQue(x1Local);
        x2Queue.EnQue(x2Local);
    }

    __aicore__ inline void ComputeUb(uint32_t count, T valueScalar)
    {
        LocalTensor<T> inputLocal = inputQueue.template DeQue<T>();
        LocalTensor<T> x1Local = x1Queue.template DeQue<T>();
        LocalTensor<T> x2Local = x2Queue.template DeQue<T>();
        LocalTensor<T> outLocal = outQueue.template AllocTensor<T>();

        if (IsValueOne(valueScalar)) {
            for (uint32_t i = 0; i < count; ++i) {
                outLocal.SetValue(i, AddcmulEvalValueOne(inputLocal.GetValue(i), x1Local.GetValue(i),
                                                         x2Local.GetValue(i)));
            }
        } else {
            for (uint32_t i = 0; i < count; ++i) {
                outLocal.SetValue(i, AddcmulEval(inputLocal.GetValue(i), x1Local.GetValue(i),
                                                 x2Local.GetValue(i), valueScalar));
            }
        }
        outQueue.EnQue(outLocal);

        inputQueue.FreeTensor(inputLocal);
        x1Queue.FreeTensor(x1Local);
        x2Queue.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count)
    {
        LocalTensor<T> outLocal = outQueue.template DeQue<T>();
        DataCopy(this->yGm[offset], outLocal, count);
        outQueue.FreeTensor(outLocal);
    }

    __aicore__ inline void CopyInputOnlyByMode(uint64_t offset, uint32_t count)
    {
        LocalTensor<T> inputLocal = inputQueue.template AllocTensor<T>();
        CopyTensorByMode(inputLocal, this->inputGm, this->inputMode, offset,
                         this->inputRepeat, this->inputDataLength, count);
        inputQueue.EnQue(inputLocal);
    }

    __aicore__ inline void CopyInputAndX1ByMode(uint64_t offset, uint32_t count)
    {
        LocalTensor<T> inputLocal = inputQueue.template AllocTensor<T>();
        LocalTensor<T> x1Local = x1Queue.template AllocTensor<T>();
        CopyTensorByMode(inputLocal, this->inputGm, this->inputMode, offset,
                         this->inputRepeat, this->inputDataLength, count);
        CopyTensorByMode(x1Local, this->x1Gm, this->x1Mode, offset,
                         this->x1Repeat, this->x1DataLength, count);
        inputQueue.EnQue(inputLocal);
        x1Queue.EnQue(x1Local);
    }

    __aicore__ inline void CopyInputAndX2ByMode(uint64_t offset, uint32_t count)
    {
        LocalTensor<T> inputLocal = inputQueue.template AllocTensor<T>();
        LocalTensor<T> x2Local = x2Queue.template AllocTensor<T>();
        CopyTensorByMode(inputLocal, this->inputGm, this->inputMode, offset,
                         this->inputRepeat, this->inputDataLength, count);
        CopyTensorByMode(x2Local, this->x2Gm, this->x2Mode, offset,
                         this->x2Repeat, this->x2DataLength, count);
        inputQueue.EnQue(inputLocal);
        x2Queue.EnQue(x2Local);
    }

    __aicore__ inline void ComputeUbScalarMulX2(uint32_t count, T valueScalar, T x2Scalar)
    {
        LocalTensor<T> inputLocal = inputQueue.template DeQue<T>();
        LocalTensor<T> x1Local = x1Queue.template DeQue<T>();
        LocalTensor<T> outLocal = outQueue.template AllocTensor<T>();
        if (IsValueOne(valueScalar)) {
            for (uint32_t i = 0; i < count; ++i) {
                outLocal.SetValue(i, AddcmulEvalValueOne(inputLocal.GetValue(i), x1Local.GetValue(i), x2Scalar));
            }
        } else {
            for (uint32_t i = 0; i < count; ++i) {
                outLocal.SetValue(i, AddcmulEval(inputLocal.GetValue(i), x1Local.GetValue(i), x2Scalar, valueScalar));
            }
        }
        outQueue.EnQue(outLocal);
        inputQueue.FreeTensor(inputLocal);
        x1Queue.FreeTensor(x1Local);
    }

    __aicore__ inline void ComputeUbScalarMulX1(uint32_t count, T valueScalar, T x1Scalar)
    {
        LocalTensor<T> inputLocal = inputQueue.template DeQue<T>();
        LocalTensor<T> x2Local = x2Queue.template DeQue<T>();
        LocalTensor<T> outLocal = outQueue.template AllocTensor<T>();
        if (IsValueOne(valueScalar)) {
            for (uint32_t i = 0; i < count; ++i) {
                outLocal.SetValue(i, AddcmulEvalValueOne(inputLocal.GetValue(i), x1Scalar, x2Local.GetValue(i)));
            }
        } else {
            for (uint32_t i = 0; i < count; ++i) {
                outLocal.SetValue(i, AddcmulEval(inputLocal.GetValue(i), x1Scalar, x2Local.GetValue(i), valueScalar));
            }
        }
        outQueue.EnQue(outLocal);
        inputQueue.FreeTensor(inputLocal);
        x2Queue.FreeTensor(x2Local);
    }

    __aicore__ inline void ComputeUbScalarMulBoth(uint32_t count, T valueScalar, T x1Scalar, T x2Scalar)
    {
        LocalTensor<T> inputLocal = inputQueue.template DeQue<T>();
        LocalTensor<T> outLocal = outQueue.template AllocTensor<T>();
        if (IsValueOne(valueScalar)) {
            for (uint32_t i = 0; i < count; ++i) {
                outLocal.SetValue(i, AddcmulEvalValueOne(inputLocal.GetValue(i), x1Scalar, x2Scalar));
            }
        } else {
            for (uint32_t i = 0; i < count; ++i) {
                outLocal.SetValue(i, AddcmulEval(inputLocal.GetValue(i), x1Scalar, x2Scalar, valueScalar));
            }
        }
        outQueue.EnQue(outLocal);
        inputQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void ProcessSafeUb(uint64_t start, uint64_t end, T valueScalar, uint32_t blockElems)
    {
        const uint64_t total = end - start;
        const uint64_t mainLen = (total / blockElems) * blockElems;
        const bool x1Scalar = (this->x1Mode == ADDCMUL_MODE_SCALAR);
        const bool x2Scalar = (this->x2Mode == ADDCMUL_MODE_SCALAR);
        const T x1ScalarValue = x1Scalar ? this->x1Gm.GetValue(0) : static_cast<T>(0);
        const T x2ScalarValue = x2Scalar ? this->x2Gm.GetValue(0) : static_cast<T>(0);

        uint64_t done = 0;
        while (done < mainLen) {
            const uint32_t curLen = CalcUbLen(start + done, mainLen - done, blockElems);
            if (curLen < blockElems) {
                break;
            }
            if (x1Scalar && x2Scalar) {
                CopyInputOnlyByMode(start + done, curLen);
                ComputeUbScalarMulBoth(curLen, valueScalar, x1ScalarValue, x2ScalarValue);
            } else if (x2Scalar) {
                CopyInputAndX1ByMode(start + done, curLen);
                ComputeUbScalarMulX2(curLen, valueScalar, x2ScalarValue);
            } else if (x1Scalar) {
                CopyInputAndX2ByMode(start + done, curLen);
                ComputeUbScalarMulX1(curLen, valueScalar, x1ScalarValue);
            } else {
                CopyInByMode(start + done, curLen);
                ComputeUb(curLen, valueScalar);
            }
            CopyOut(start + done, curLen);
            done += curLen;
        }

        if (done < total) {
            this->ScalarLoop(start + done, end, valueScalar);
            DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(this->yGm);
        }
    }

private:
    TPipe pipe;
    bool canSafeUbPath = false;
    TQue<QuePosition::VECIN, ADDCMUL_BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECIN, ADDCMUL_BUFFER_NUM> x1Queue;
    TQue<QuePosition::VECIN, ADDCMUL_BUFFER_NUM> x2Queue;
    TQue<QuePosition::VECOUT, ADDCMUL_BUFFER_NUM> outQueue;
};

template <class T>
class KernelAddcmulVector : public KernelAddcmulScalarBase<T> {
public:
    __aicore__ inline KernelAddcmulVector() {}

    __aicore__ inline void Init(GM_ADDR inputData, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
                                const AddcmulTilingData &tiling)
    {
        this->InitBase(inputData, x1, x2, value, y, tiling);
        const uint32_t blockElems = ADDCMUL_DATA_BLOCK_BYTES / sizeof(T);
        canSafeVectorPath = CanUseSafeVectorPath(blockElems);
        activeInputBufferNum = (this->sameShape == 1U) ? ADDCMUL_VECTOR_BUFFER_NUM : ADDCMUL_BUFFER_NUM;
        if (this->tileLength > 0 && canSafeVectorPath) {
            pipe.InitBuffer(inputQueue, activeInputBufferNum, this->tileLength * sizeof(T));
            pipe.InitBuffer(x1Queue, activeInputBufferNum, this->tileLength * sizeof(T));
            pipe.InitBuffer(x2Queue, activeInputBufferNum, this->tileLength * sizeof(T));
            pipe.InitBuffer(outQueue, ADDCMUL_BUFFER_NUM, this->tileLength * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        if (this->length == 0 || this->perCoreElements == 0) {
            return;
        }
        const uint64_t start = static_cast<uint64_t>(GetBlockIdx()) * this->perCoreElements;
        if (start >= this->length) {
            return;
        }
        uint64_t end = start + this->perCoreElements;
        if (end > this->length) {
            end = this->length;
        }

        const T valueScalar = this->valueGm.GetValue(0);
        const uint32_t blockElems = ADDCMUL_DATA_BLOCK_BYTES / sizeof(T);
        if (this->tileLength > 0 && this->sameShape == 1) {
            ProcessLinearVector(start, end, valueScalar, blockElems);
        } else if (this->tileLength > 0 && canSafeVectorPath) {
            ProcessSafeVector(start, end, valueScalar, blockElems);
        } else {
            this->ScalarLoop(start, end, valueScalar);
            DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(this->yGm);
        }
    }

private:
    __aicore__ inline bool IsLinearOrScalar(uint32_t mode) const
    {
        return mode == ADDCMUL_MODE_LINEAR || mode == ADDCMUL_MODE_SCALAR;
    }

    __aicore__ inline bool IsSafeSegment(uint32_t mode, uint64_t repeat, uint64_t dataLength,
                                         uint32_t blockElems) const
    {
        if (mode != ADDCMUL_MODE_SEGMENT) {
            return false;
        }
        if (repeat > 1) {
            return true;
        }
        return dataLength >= blockElems && (dataLength % static_cast<uint64_t>(blockElems)) == 0;
    }

    __aicore__ inline bool IsSafeVectorMode(uint32_t mode, uint64_t repeat, uint64_t dataLength,
                                            uint32_t blockElems) const
    {
        return IsLinearOrScalar(mode) || IsSafeSegment(mode, repeat, dataLength, blockElems);
    }

    __aicore__ inline bool CanUseSafeVectorPath(uint32_t blockElems) const
    {
        return IsSafeVectorMode(this->inputMode, this->inputRepeat, this->inputDataLength, blockElems) &&
               IsSafeVectorMode(this->x1Mode, this->x1Repeat, this->x1DataLength, blockElems) &&
               IsSafeVectorMode(this->x2Mode, this->x2Repeat, this->x2DataLength, blockElems);
    }

    __aicore__ inline void LimitBySegment(uint32_t mode, uint64_t repeat, uint64_t dataLength, uint64_t linear,
                                          uint64_t &limit) const
    {
        if (mode != ADDCMUL_MODE_SEGMENT || repeat > 1) {
            return;
        }
        const uint64_t pos = linear % dataLength;
        const uint64_t remain = dataLength - pos;
        if (remain < limit) {
            limit = remain;
        }
    }

    __aicore__ inline uint32_t CalcVectorLen(uint64_t linear, uint64_t remaining,
                                             uint32_t blockElems) const
    {
        uint64_t limit = this->tileLength;
        if (limit > remaining) {
            limit = remaining;
        }
        LimitBySegment(this->inputMode, this->inputRepeat, this->inputDataLength, linear, limit);
        LimitBySegment(this->x1Mode, this->x1Repeat, this->x1DataLength, linear, limit);
        LimitBySegment(this->x2Mode, this->x2Repeat, this->x2DataLength, linear, limit);
        limit = (limit / static_cast<uint64_t>(blockElems)) * static_cast<uint64_t>(blockElems);
        return static_cast<uint32_t>(limit);
    }

    __aicore__ inline uint32_t CalcLinearVectorLen(uint64_t remaining, uint32_t blockElems) const
    {
        uint64_t limit = this->tileLength;
        if (limit > remaining) {
            limit = remaining;
        }
        limit = (limit / static_cast<uint64_t>(blockElems)) * static_cast<uint64_t>(blockElems);
        return static_cast<uint32_t>(limit);
    }

    __aicore__ inline void ProcessLinearVector(uint64_t start, uint64_t end, T valueScalar, uint32_t blockElems)
    {
        if (activeInputBufferNum > ADDCMUL_BUFFER_NUM) {
            ProcessLinearVectorDoubleBuffer(start, end, valueScalar, blockElems);
            return;
        }

        const uint64_t total = end - start;
        const uint64_t mainLen = (total / blockElems) * blockElems;

        uint64_t done = 0;
        while (done < mainLen) {
            const uint32_t curLen = CalcLinearVectorLen(mainLen - done, blockElems);
            if (curLen < blockElems) {
                break;
            }
            CopyInLinear(start + done, curLen);
            Compute(curLen, valueScalar);
            CopyOut(start + done, curLen);
            done += curLen;
        }

        if (done < total) {
            this->ScalarLoop(start + done, end, valueScalar);
            DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(this->yGm);
        }
    }

    __aicore__ inline void ProcessLinearVectorDoubleBuffer(uint64_t start, uint64_t end, T valueScalar,
                                                          uint32_t blockElems)
    {
        const uint64_t total = end - start;
        const uint64_t mainLen = (total / blockElems) * blockElems;
        if (mainLen < static_cast<uint64_t>(blockElems)) {
            if (total > 0) {
                this->ScalarLoop(start, end, valueScalar);
                DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(this->yGm);
            }
            return;
        }

        uint64_t done = 0;
        uint64_t curOffset = start;
        uint32_t curLen = CalcLinearVectorLen(mainLen, blockElems);
        if (curLen < blockElems) {
            this->ScalarLoop(start, end, valueScalar);
            DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(this->yGm);
            return;
        }

        CopyInLinear(curOffset, curLen);
        done += curLen;

        while (done < mainLen) {
            const uint64_t nextOffset = start + done;
            const uint32_t nextLen = CalcLinearVectorLen(mainLen - done, blockElems);
            if (nextLen < blockElems) {
                break;
            }
            CopyInLinear(nextOffset, nextLen);
            Compute(curLen, valueScalar);
            CopyOut(curOffset, curLen);
            curOffset = nextOffset;
            curLen = nextLen;
            done += nextLen;
        }

        Compute(curLen, valueScalar);
        CopyOut(curOffset, curLen);

        if (done < total) {
            this->ScalarLoop(start + done, end, valueScalar);
            DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(this->yGm);
        }
    }

    __aicore__ inline void ProcessSafeVector(uint64_t start, uint64_t end, T valueScalar, uint32_t blockElems)
    {
        const uint64_t total = end - start;
        const uint64_t mainLen = (total / blockElems) * blockElems;
        const bool x1Scalar = (this->x1Mode == ADDCMUL_MODE_SCALAR);
        const bool x2Scalar = (this->x2Mode == ADDCMUL_MODE_SCALAR);
        const T x1ScalarValue = x1Scalar ? this->x1Gm.GetValue(0) : static_cast<T>(0);
        const T x2ScalarValue = x2Scalar ? this->x2Gm.GetValue(0) : static_cast<T>(0);

        uint64_t done = 0;
        while (done < mainLen) {
            const uint32_t curLen = CalcVectorLen(start + done, mainLen - done, blockElems);
            if (curLen < blockElems) {
                break;
            }

            if (x1Scalar && x2Scalar) {
                CopyInputOnlyByMode(start + done, curLen);
                ComputeScalarMulBoth(curLen, valueScalar, x1ScalarValue, x2ScalarValue);
            } else if (x2Scalar) {
                CopyInputAndX1ByMode(start + done, curLen);
                ComputeScalarMulX2(curLen, valueScalar, x2ScalarValue);
            } else if (x1Scalar) {
                CopyInputAndX2ByMode(start + done, curLen);
                ComputeScalarMulX1(curLen, valueScalar, x1ScalarValue);
            } else {
                CopyInByMode(start + done, curLen);
                Compute(curLen, valueScalar);
            }
            CopyOut(start + done, curLen);
            done += curLen;
        }

        if (done < total) {
            this->ScalarLoop(start + done, end, valueScalar);
            DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(this->yGm);
        }
    }

    __aicore__ inline void FillSegmentRepeat(LocalTensor<T> local, const GlobalTensor<T> &gm,
                                             uint64_t linear, uint64_t repeat, uint64_t dataLength,
                                             uint32_t count)
    {
        const uint64_t safeRepeat = repeat == 0 ? 1 : repeat;
        const uint64_t safeDataLength = dataLength == 0 ? 1 : dataLength;
        uint32_t filled = 0;
        uint64_t repeatPos = linear % safeRepeat;
        uint64_t dataIdx = (linear / safeRepeat) % safeDataLength;
        while (filled < count) {
            uint64_t seg = safeRepeat - repeatPos;
            if (seg > static_cast<uint64_t>(count - filled)) {
                seg = static_cast<uint64_t>(count - filled);
            }
            Duplicate(local[filled], gm.GetValue(dataIdx), static_cast<uint32_t>(seg));
            filled += static_cast<uint32_t>(seg);
            repeatPos = 0;
            ++dataIdx;
            if (dataIdx >= safeDataLength) {
                dataIdx = 0;
            }
        }
    }

    __aicore__ inline void CopyTensorByMode(LocalTensor<T> local, const GlobalTensor<T> &gm,
                                            uint32_t mode, uint64_t linear, uint64_t repeat,
                                            uint64_t dataLength, uint32_t count)
    {
        if (mode == ADDCMUL_MODE_SCALAR) {
            Duplicate(local, gm.GetValue(0), count);
        } else if (mode == ADDCMUL_MODE_SEGMENT && repeat > 1) {
            FillSegmentRepeat(local, gm, linear, repeat, dataLength, count);
        } else {
            const uint64_t gmOffset = this->FastOffsetByMode(mode, linear, repeat, dataLength);
            DataCopy(local, gm[gmOffset], count);
        }
    }

    __aicore__ inline void CopyInLinear(uint64_t offset, uint32_t count)
    {
        LocalTensor<T> inputLocal = inputQueue.template AllocTensor<T>();
        LocalTensor<T> x1Local = x1Queue.template AllocTensor<T>();
        LocalTensor<T> x2Local = x2Queue.template AllocTensor<T>();

        DataCopy(inputLocal, this->inputGm[offset], count);
        DataCopy(x1Local, this->x1Gm[offset], count);
        DataCopy(x2Local, this->x2Gm[offset], count);

        inputQueue.EnQue(inputLocal);
        x1Queue.EnQue(x1Local);
        x2Queue.EnQue(x2Local);
    }

    __aicore__ inline void CopyInputOnlyByMode(uint64_t offset, uint32_t count)
    {
        LocalTensor<T> inputLocal = inputQueue.template AllocTensor<T>();
        CopyTensorByMode(inputLocal, this->inputGm, this->inputMode, offset,
                         this->inputRepeat, this->inputDataLength, count);
        inputQueue.EnQue(inputLocal);
    }

    __aicore__ inline void CopyInputAndX1ByMode(uint64_t offset, uint32_t count)
    {
        LocalTensor<T> inputLocal = inputQueue.template AllocTensor<T>();
        LocalTensor<T> x1Local = x1Queue.template AllocTensor<T>();

        CopyTensorByMode(inputLocal, this->inputGm, this->inputMode, offset,
                         this->inputRepeat, this->inputDataLength, count);
        CopyTensorByMode(x1Local, this->x1Gm, this->x1Mode, offset,
                         this->x1Repeat, this->x1DataLength, count);

        inputQueue.EnQue(inputLocal);
        x1Queue.EnQue(x1Local);
    }

    __aicore__ inline void CopyInputAndX2ByMode(uint64_t offset, uint32_t count)
    {
        LocalTensor<T> inputLocal = inputQueue.template AllocTensor<T>();
        LocalTensor<T> x2Local = x2Queue.template AllocTensor<T>();

        CopyTensorByMode(inputLocal, this->inputGm, this->inputMode, offset,
                         this->inputRepeat, this->inputDataLength, count);
        CopyTensorByMode(x2Local, this->x2Gm, this->x2Mode, offset,
                         this->x2Repeat, this->x2DataLength, count);

        inputQueue.EnQue(inputLocal);
        x2Queue.EnQue(x2Local);
    }

    __aicore__ inline void CopyInByMode(uint64_t offset, uint32_t count)
    {
        LocalTensor<T> inputLocal = inputQueue.template AllocTensor<T>();
        LocalTensor<T> x1Local = x1Queue.template AllocTensor<T>();
        LocalTensor<T> x2Local = x2Queue.template AllocTensor<T>();

        CopyTensorByMode(inputLocal, this->inputGm, this->inputMode, offset,
                         this->inputRepeat, this->inputDataLength, count);
        CopyTensorByMode(x1Local, this->x1Gm, this->x1Mode, offset,
                         this->x1Repeat, this->x1DataLength, count);
        CopyTensorByMode(x2Local, this->x2Gm, this->x2Mode, offset,
                         this->x2Repeat, this->x2DataLength, count);

        inputQueue.EnQue(inputLocal);
        x1Queue.EnQue(x1Local);
        x2Queue.EnQue(x2Local);
    }

    __aicore__ inline void Compute(uint32_t count, T valueScalar)
    {
        LocalTensor<T> inputLocal = inputQueue.template DeQue<T>();
        LocalTensor<T> x1Local = x1Queue.template DeQue<T>();
        LocalTensor<T> x2Local = x2Queue.template DeQue<T>();
        LocalTensor<T> outLocal = outQueue.template AllocTensor<T>();

        Mul(outLocal, x1Local, x2Local, count);
        PipeBarrier<PIPE_V>();
        if (!IsValueOne(valueScalar)) {
            Muls(outLocal, outLocal, valueScalar, count);
            PipeBarrier<PIPE_V>();
        }
        Add(outLocal, inputLocal, outLocal, count);
        outQueue.EnQue(outLocal);

        inputQueue.FreeTensor(inputLocal);
        x1Queue.FreeTensor(x1Local);
        x2Queue.FreeTensor(x2Local);
    }

    __aicore__ inline void ComputeScalarMulX2(uint32_t count, T valueScalar, T x2Scalar)
    {
        LocalTensor<T> inputLocal = inputQueue.template DeQue<T>();
        LocalTensor<T> x1Local = x1Queue.template DeQue<T>();
        LocalTensor<T> outLocal = outQueue.template AllocTensor<T>();

        Muls(outLocal, x1Local, x2Scalar, count);
        PipeBarrier<PIPE_V>();
        if (!IsValueOne(valueScalar)) {
            Muls(outLocal, outLocal, valueScalar, count);
            PipeBarrier<PIPE_V>();
        }
        Add(outLocal, inputLocal, outLocal, count);
        outQueue.EnQue(outLocal);

        inputQueue.FreeTensor(inputLocal);
        x1Queue.FreeTensor(x1Local);
    }

    __aicore__ inline void ComputeScalarMulX1(uint32_t count, T valueScalar, T x1Scalar)
    {
        LocalTensor<T> inputLocal = inputQueue.template DeQue<T>();
        LocalTensor<T> x2Local = x2Queue.template DeQue<T>();
        LocalTensor<T> outLocal = outQueue.template AllocTensor<T>();

        Muls(outLocal, x2Local, x1Scalar, count);
        PipeBarrier<PIPE_V>();
        if (!IsValueOne(valueScalar)) {
            Muls(outLocal, outLocal, valueScalar, count);
            PipeBarrier<PIPE_V>();
        }
        Add(outLocal, inputLocal, outLocal, count);
        outQueue.EnQue(outLocal);

        inputQueue.FreeTensor(inputLocal);
        x2Queue.FreeTensor(x2Local);
    }

    __aicore__ inline void ComputeScalarMulBoth(uint32_t count, T valueScalar, T x1Scalar, T x2Scalar)
    {
        LocalTensor<T> inputLocal = inputQueue.template DeQue<T>();
        LocalTensor<T> outLocal = outQueue.template AllocTensor<T>();

        const T scaled = ScaledMulScalar(x1Scalar, x2Scalar, valueScalar);
        Duplicate(outLocal, scaled, count);
        PipeBarrier<PIPE_V>();
        Add(outLocal, inputLocal, outLocal, count);
        outQueue.EnQue(outLocal);

        inputQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count)
    {
        LocalTensor<T> outLocal = outQueue.template DeQue<T>();
        DataCopy(this->yGm[offset], outLocal, count);
        outQueue.FreeTensor(outLocal);
    }

private:
    TPipe pipe;
    bool canSafeVectorPath = false;
    uint32_t activeInputBufferNum = ADDCMUL_BUFFER_NUM;
    TQue<QuePosition::VECIN, ADDCMUL_VECTOR_BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECIN, ADDCMUL_VECTOR_BUFFER_NUM> x1Queue;
    TQue<QuePosition::VECIN, ADDCMUL_VECTOR_BUFFER_NUM> x2Queue;
    TQue<QuePosition::VECOUT, ADDCMUL_BUFFER_NUM> outQueue;
};

template <>
class KernelAddcmul<float> : public KernelAddcmulVector<float> {
public:
    __aicore__ inline KernelAddcmul() {}
};

template <>
class KernelAddcmul<half> : public KernelAddcmulVector<half> {
public:
    __aicore__ inline KernelAddcmul() {}
};


template <>
class KernelAddcmul<int32_t> : public KernelAddcmulVector<int32_t> {
public:
    __aicore__ inline KernelAddcmul() {}
};

template <typename DT_INPUT_DATA>
__global__ __aicore__ void addcmul(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
                                   GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddcmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddcmulTilingData, tiling_data, tiling);
    KernelAddcmul<DT_INPUT_DATA> op;
    op.Init(input_data, x1, x2, value, y, tiling_data);
    op.Process();
}
