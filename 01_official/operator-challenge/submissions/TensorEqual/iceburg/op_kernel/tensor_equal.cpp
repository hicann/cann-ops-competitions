#include "kernel_operator.h"
#include <type_traits>

using namespace AscendC;

template <typename T>
class KernelTensorEqualBase {
public:
    __aicore__ inline void InitBase(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const TensorEqualTilingData& tilingData)
    {
        x1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1));
        x2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));

        totalSize = tilingData.totalSize;
        tileLength = tilingData.tileLength;
        for (uint32_t i = 0; i < 8; ++i) {
            outDims[i] = tilingData.outDims[i];
            x1Strides[i] = tilingData.x1Strides[i];
            x2Strides[i] = tilingData.x2Strides[i];
        }
    }

protected:
    __aicore__ inline uint32_t Offset(uint32_t linear, const uint32_t* strides) const
    {
        uint32_t offset = 0;
        for (int32_t dim = 7; dim >= 0; --dim) {
            const uint32_t coord = linear % outDims[dim];
            linear = linear / outDims[dim];
            offset += coord * strides[dim];
        }
        return offset;
    }

    __aicore__ inline uint32_t AlignInput(uint32_t length) const
    {
        const uint32_t alignElems = 32U / sizeof(T);
        return (length + alignElems - 1U) / alignElems * alignElems;
    }

    __aicore__ inline uint32_t AlignOutput(uint32_t length) const
    {
        return (length + 31U) / 32U * 32U;
    }

    template <typename U>
    __aicore__ inline LocalTensor<U> WorkAs(uint32_t byteOffset)
    {
        return workBuf.Get<U>()[byteOffset / sizeof(U)];
    }

    __aicore__ inline LocalTensor<uint8_t> Mask()
    {
        return workBuf.Get<uint8_t>();
    }

    __aicore__ inline uint32_t MaskBytes() const
    {
        return tileLength;
    }

    __aicore__ inline uint32_t CompareBytes() const
    {
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> ||
                      std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t>) {
            return tileLength * sizeof(float);
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            return tileLength * sizeof(half);
        } else {
            return 0;
        }
    }

    __aicore__ inline void CompareTwoTTo(LocalTensor<T> x1Local, LocalTensor<T> x2Local,
                                         LocalTensor<uint8_t> yLocal, uint32_t length,
                                         uint32_t castBaseOffset)
    {
        LocalTensor<uint8_t> mask = Mask();
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> ||
                      std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t>) {
            LocalTensor<float> x1Float = WorkAs<float>(castBaseOffset);
            LocalTensor<float> x2Float = WorkAs<float>(castBaseOffset + CompareBytes());
            Cast(x1Float, x1Local, RoundMode::CAST_NONE, length);
            Cast(x2Float, x2Local, RoundMode::CAST_NONE, length);
            PipeBarrier<PIPE_V>();
            Compare(mask, x1Float, x2Float, CMPMODE::EQ, length);
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            LocalTensor<half> x1Half = WorkAs<half>(castBaseOffset);
            LocalTensor<half> x2Half = WorkAs<half>(castBaseOffset + CompareBytes());
            Cast(x1Half, x1Local, RoundMode::CAST_NONE, length);
            Cast(x2Half, x2Local, RoundMode::CAST_NONE, length);
            PipeBarrier<PIPE_V>();
            Compare(mask, x1Half, x2Half, CMPMODE::EQ, length);
        } else {
            Compare(mask, x1Local, x2Local, CMPMODE::EQ, length);
        }
        PipeBarrier<PIPE_V>();
        MaterializeBoolTo(yLocal, length);
    }

    __aicore__ inline float ScalarToFloat(T value) const
    {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            return ToFloat(value);
        } else {
            return static_cast<float>(value);
        }
    }

    __aicore__ inline void CompareScalarToInput(T scalarValue, LocalTensor<T> inputLocal,
                                                LocalTensor<uint8_t> yLocal, uint32_t length)
    {
        LocalTensor<uint8_t> mask = Mask();
        const uint32_t scalarOffset = MaskBytes();
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> ||
                      std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t>) {
            LocalTensor<float> scalarLocal = WorkAs<float>(scalarOffset);
            LocalTensor<float> inputFloat = WorkAs<float>(scalarOffset + CompareBytes());
            Duplicate(scalarLocal, ScalarToFloat(scalarValue), length);
            Cast(inputFloat, inputLocal, RoundMode::CAST_NONE, length);
            PipeBarrier<PIPE_V>();
            Compare(mask, scalarLocal, inputFloat, CMPMODE::EQ, length);
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            LocalTensor<int16_t> scalarWide = WorkAs<int16_t>(scalarOffset);
            LocalTensor<half> scalarLocal = WorkAs<half>(scalarOffset + tileLength * sizeof(int16_t));
            LocalTensor<half> inputHalf = WorkAs<half>(scalarOffset + tileLength * sizeof(int16_t) + CompareBytes());
            Duplicate(scalarWide, static_cast<int16_t>(scalarValue), length);
            Cast(scalarLocal, scalarWide, RoundMode::CAST_NONE, length);
            Cast(inputHalf, inputLocal, RoundMode::CAST_NONE, length);
            PipeBarrier<PIPE_V>();
            Compare(mask, scalarLocal, inputHalf, CMPMODE::EQ, length);
        } else {
            LocalTensor<T> scalarLocal = WorkAs<T>(scalarOffset);
            Duplicate(scalarLocal, scalarValue, length);
            PipeBarrier<PIPE_V>();
            Compare(mask, scalarLocal, inputLocal, CMPMODE::EQ, length);
        }
        PipeBarrier<PIPE_V>();
        MaterializeBoolTo(yLocal, length);
    }

    __aicore__ inline void MaterializeBoolTo(LocalTensor<uint8_t> yLocal, uint32_t length)
    {
        LocalTensor<uint8_t> mask = Mask();
        LocalTensor<half> ones = WorkAs<half>(MaskBytes());
        LocalTensor<half> outHalf = WorkAs<half>(MaskBytes() + tileLength * sizeof(half));
        Duplicate(ones, static_cast<half>(1.0), length);
        Duplicate(outHalf, static_cast<half>(0.0), length);
        Select(outHalf, mask, ones, outHalf, SELMODE::VSEL_TENSOR_TENSOR_MODE, length);
        PipeBarrier<PIPE_V>();
        Cast(yLocal, outHalf, RoundMode::CAST_NONE, length);
        PipeBarrier<PIPE_V>();
    }

protected:
    static constexpr int32_t BUFFER_NUM = 1;
    TBuf<QuePosition::VECCALC> workBuf;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<uint8_t> yGm;
    uint32_t totalSize;
    uint32_t tileLength;
    uint32_t outDims[8];
    uint32_t x1Strides[8];
    uint32_t x2Strides[8];
};

template <typename T>
class KernelTensorEqualContiguous : public KernelTensorEqualBase<T> {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const TensorEqualTilingData& tilingData)
    {
        this->InitBase(x1, x2, y, tilingData);
        pipe.InitBuffer(x1Queue, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(x2Queue, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(uint8_t));
        pipe.InitBuffer(this->workBuf, this->tileLength * FastWorkBytesPerElem());
    }

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < this->totalSize; offset += this->tileLength) {
            uint32_t length = this->totalSize - offset;
            if (length > this->tileLength) {
                length = this->tileLength;
            }
            const uint32_t inputCopyLength = this->AlignInput(length);
            const uint32_t outputCopyLength = this->AlignOutput(length);

            LocalTensor<T> x1Local = x1Queue.AllocTensor<T>();
            LocalTensor<T> x2Local = x2Queue.AllocTensor<T>();
            DataCopy(x1Local, this->x1Gm[offset], inputCopyLength);
            DataCopy(x2Local, this->x2Gm[offset], inputCopyLength);
            x1Queue.EnQue(x1Local);
            x2Queue.EnQue(x2Local);

            x1Local = x1Queue.DeQue<T>();
            x2Local = x2Queue.DeQue<T>();
            LocalTensor<uint8_t> yLocal = outQueue.AllocTensor<uint8_t>();
            this->CompareTwoTTo(x1Local, x2Local, yLocal, length, this->MaskBytes());
            outQueue.EnQue(yLocal);
            x1Queue.FreeTensor(x1Local);
            x2Queue.FreeTensor(x2Local);

            yLocal = outQueue.DeQue<uint8_t>();
            DataCopy(this->yGm[offset], yLocal, outputCopyLength);
            outQueue.FreeTensor(yLocal);
        }
    }

private:
    __aicore__ inline uint32_t FastWorkBytesPerElem() const
    {
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> ||
                      std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t>) {
            return 9;
        } else {
            return 5;
        }
    }

private:
    static constexpr int32_t BUFFER_NUM = 1;
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> x2Queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
};

template <typename T>
class KernelTensorEqualScalarBroadcast : public KernelTensorEqualBase<T> {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const TensorEqualTilingData& tilingData,
                                bool x1Scalar)
    {
        this->InitBase(x1, x2, y, tilingData);
        scalarOnX1 = x1Scalar;
        pipe.InitBuffer(inputQueue, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(uint8_t));
        pipe.InitBuffer(this->workBuf, this->tileLength * FastWorkBytesPerElem());
    }

    __aicore__ inline void Process()
    {
        const T scalarValue = scalarOnX1 ? this->x1Gm.GetValue(0) : this->x2Gm.GetValue(0);
        for (uint32_t offset = 0; offset < this->totalSize; offset += this->tileLength) {
            uint32_t length = this->totalSize - offset;
            if (length > this->tileLength) {
                length = this->tileLength;
            }
            const uint32_t inputCopyLength = this->AlignInput(length);
            const uint32_t outputCopyLength = this->AlignOutput(length);

            LocalTensor<T> inputLocal = inputQueue.AllocTensor<T>();
            if (scalarOnX1) {
                DataCopy(inputLocal, this->x2Gm[offset], inputCopyLength);
            } else {
                DataCopy(inputLocal, this->x1Gm[offset], inputCopyLength);
            }
            inputQueue.EnQue(inputLocal);

            inputLocal = inputQueue.DeQue<T>();
            LocalTensor<uint8_t> yLocal = outQueue.AllocTensor<uint8_t>();
            this->CompareScalarToInput(scalarValue, inputLocal, yLocal, length);
            outQueue.EnQue(yLocal);
            inputQueue.FreeTensor(inputLocal);

            yLocal = outQueue.DeQue<uint8_t>();
            DataCopy(this->yGm[offset], yLocal, outputCopyLength);
            outQueue.FreeTensor(yLocal);
        }
    }

private:
    __aicore__ inline uint32_t FastWorkBytesPerElem() const
    {
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> ||
                      std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t>) {
            return 9;
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            return 6;
        } else {
            return 5;
        }
    }

private:
    static constexpr int32_t BUFFER_NUM = 1;
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    bool scalarOnX1;
};

template <typename T>
class KernelTensorEqualRowVectorBroadcast : public KernelTensorEqualBase<T> {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const TensorEqualTilingData& tilingData)
    {
        this->InitBase(x1, x2, y, tilingData);
        rowLength = this->outDims[7];
        pipe.InitBuffer(x1Queue, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(x2Queue, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(uint8_t));
        pipe.InitBuffer(this->workBuf, this->tileLength * FastWorkBytesPerElem());
    }

    __aicore__ inline void Process()
    {
        for (uint32_t rowBase = 0; rowBase < this->totalSize; rowBase += rowLength) {
            const uint32_t x1Base = this->Offset(rowBase, this->x1Strides);
            const uint32_t x2Base = this->Offset(rowBase, this->x2Strides);
            for (uint32_t innerOffset = 0; innerOffset < rowLength; innerOffset += this->tileLength) {
                uint32_t length = rowLength - innerOffset;
                if (length > this->tileLength) {
                    length = this->tileLength;
                }
                const uint32_t inputCopyLength = this->AlignInput(length);
                const uint32_t outputCopyLength = this->AlignOutput(length);

                LocalTensor<T> x1Local = x1Queue.AllocTensor<T>();
                LocalTensor<T> x2Local = x2Queue.AllocTensor<T>();
                DataCopy(x1Local, this->x1Gm[x1Base + innerOffset], inputCopyLength);
                DataCopy(x2Local, this->x2Gm[x2Base + innerOffset], inputCopyLength);
                x1Queue.EnQue(x1Local);
                x2Queue.EnQue(x2Local);

                x1Local = x1Queue.DeQue<T>();
                x2Local = x2Queue.DeQue<T>();
                LocalTensor<uint8_t> yLocal = outQueue.AllocTensor<uint8_t>();
                this->CompareTwoTTo(x1Local, x2Local, yLocal, length, this->MaskBytes());
                outQueue.EnQue(yLocal);
                x1Queue.FreeTensor(x1Local);
                x2Queue.FreeTensor(x2Local);

                yLocal = outQueue.DeQue<uint8_t>();
                DataCopy(this->yGm[rowBase + innerOffset], yLocal, outputCopyLength);
                outQueue.FreeTensor(yLocal);
            }
        }
    }

private:
    __aicore__ inline uint32_t FastWorkBytesPerElem() const
    {
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> ||
                      std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t>) {
            return 9;
        } else {
            return 5;
        }
    }

private:
    static constexpr int32_t BUFFER_NUM = 1;
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> x2Queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    uint32_t rowLength;
};

template <typename T>
class KernelTensorEqualRowScalarBroadcast : public KernelTensorEqualBase<T> {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const TensorEqualTilingData& tilingData,
                                bool x1Scalar)
    {
        this->InitBase(x1, x2, y, tilingData);
        rowLength = this->outDims[7];
        scalarOnX1 = x1Scalar;
        pipe.InitBuffer(inputQueue, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(uint8_t));
        pipe.InitBuffer(this->workBuf, this->tileLength * FastWorkBytesPerElem());
    }

    __aicore__ inline void Process()
    {
        for (uint32_t rowBase = 0; rowBase < this->totalSize; rowBase += rowLength) {
            const uint32_t scalarOffset = scalarOnX1 ? this->Offset(rowBase, this->x1Strides)
                                                     : this->Offset(rowBase, this->x2Strides);
            const uint32_t inputBase = scalarOnX1 ? this->Offset(rowBase, this->x2Strides)
                                                  : this->Offset(rowBase, this->x1Strides);
            const T scalarValue = scalarOnX1 ? this->x1Gm.GetValue(scalarOffset)
                                             : this->x2Gm.GetValue(scalarOffset);
            for (uint32_t innerOffset = 0; innerOffset < rowLength; innerOffset += this->tileLength) {
                uint32_t length = rowLength - innerOffset;
                if (length > this->tileLength) {
                    length = this->tileLength;
                }
                const uint32_t inputCopyLength = this->AlignInput(length);
                const uint32_t outputCopyLength = this->AlignOutput(length);

                LocalTensor<T> inputLocal = inputQueue.AllocTensor<T>();
                if (scalarOnX1) {
                    DataCopy(inputLocal, this->x2Gm[inputBase + innerOffset], inputCopyLength);
                } else {
                    DataCopy(inputLocal, this->x1Gm[inputBase + innerOffset], inputCopyLength);
                }
                inputQueue.EnQue(inputLocal);

                inputLocal = inputQueue.DeQue<T>();
                LocalTensor<uint8_t> yLocal = outQueue.AllocTensor<uint8_t>();
                this->CompareScalarToInput(scalarValue, inputLocal, yLocal, length);
                outQueue.EnQue(yLocal);
                inputQueue.FreeTensor(inputLocal);

                yLocal = outQueue.DeQue<uint8_t>();
                DataCopy(this->yGm[rowBase + innerOffset], yLocal, outputCopyLength);
                outQueue.FreeTensor(yLocal);
            }
        }
    }

private:
    __aicore__ inline uint32_t FastWorkBytesPerElem() const
    {
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> ||
                      std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t>) {
            return 9;
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            return 6;
        } else {
            return 5;
        }
    }

private:
    static constexpr int32_t BUFFER_NUM = 1;
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    uint32_t rowLength;
    bool scalarOnX1;
};

template <typename T>
class KernelTensorEqualGeneric : public KernelTensorEqualBase<T> {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const TensorEqualTilingData& tilingData)
    {
        this->InitBase(x1, x2, y, tilingData);
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(uint8_t));
        pipe.InitBuffer(this->workBuf, this->tileLength * GenericWorkBytesPerElem());
    }

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < this->totalSize; offset += this->tileLength) {
            uint32_t length = this->totalSize - offset;
            if (length > this->tileLength) {
                length = this->tileLength;
            }
            FillInputsBroadcast(offset, length);
            LocalTensor<uint8_t> yLocal = outQueue.AllocTensor<uint8_t>();
            this->CompareTwoTTo(X1Local(), X2Local(), yLocal, length, CastBaseOffset());
            outQueue.EnQue(yLocal);

            yLocal = outQueue.DeQue<uint8_t>();
            DataCopy(this->yGm[offset], yLocal, this->AlignOutput(length));
            outQueue.FreeTensor(yLocal);
        }
    }

private:
    __aicore__ inline uint32_t X1Offset() const
    {
        return this->MaskBytes();
    }

    __aicore__ inline uint32_t X2Offset() const
    {
        return this->MaskBytes() + this->tileLength * sizeof(T);
    }

    __aicore__ inline uint32_t CastBaseOffset() const
    {
        return this->MaskBytes() + 2U * this->tileLength * sizeof(T);
    }

    __aicore__ inline LocalTensor<T> X1Local()
    {
        return this->template WorkAs<T>(X1Offset());
    }

    __aicore__ inline LocalTensor<T> X2Local()
    {
        return this->template WorkAs<T>(X2Offset());
    }

    __aicore__ inline void FillInputsBroadcast(uint32_t offset, uint32_t length)
    {
        LocalTensor<T> x1Local = X1Local();
        LocalTensor<T> x2Local = X2Local();
        for (uint32_t i = 0; i < length; ++i) {
            const uint32_t linear = offset + i;
            x1Local.SetValue(i, this->x1Gm.GetValue(this->Offset(linear, this->x1Strides)));
            x2Local.SetValue(i, this->x2Gm.GetValue(this->Offset(linear, this->x2Strides)));
        }
    }

    __aicore__ inline uint32_t GenericWorkBytesPerElem() const
    {
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> ||
                      std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t>) {
            return 1 + 2 * sizeof(T) + 2 * sizeof(float);
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            return 7;
        } else {
            return 1 + 2 * sizeof(T);
        }
    }

private:
    static constexpr int32_t BUFFER_NUM = 1;
    TPipe pipe;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
};

extern "C" __global__ __aicore__ void tensor_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {
        KernelTensorEqualContiguous<DTYPE_X1> op;
        op.Init(x1, x2, y, tiling_data);
        op.Process();
    }
    if (TILING_KEY_IS(2)) {
        KernelTensorEqualScalarBroadcast<DTYPE_X1> op;
        op.Init(x1, x2, y, tiling_data, true);
        op.Process();
    }
    if (TILING_KEY_IS(3)) {
        KernelTensorEqualScalarBroadcast<DTYPE_X1> op;
        op.Init(x1, x2, y, tiling_data, false);
        op.Process();
    }
    if (TILING_KEY_IS(4)) {
        KernelTensorEqualRowVectorBroadcast<DTYPE_X1> op;
        op.Init(x1, x2, y, tiling_data);
        op.Process();
    }
    if (TILING_KEY_IS(5)) {
        KernelTensorEqualRowScalarBroadcast<DTYPE_X1> op;
        op.Init(x1, x2, y, tiling_data, true);
        op.Process();
    }
    if (TILING_KEY_IS(6)) {
        KernelTensorEqualRowScalarBroadcast<DTYPE_X1> op;
        op.Init(x1, x2, y, tiling_data, false);
        op.Process();
    }
    if (TILING_KEY_IS(0)) {
        KernelTensorEqualGeneric<DTYPE_X1> op;
        op.Init(x1, x2, y, tiling_data);
        op.Process();
    }
}
