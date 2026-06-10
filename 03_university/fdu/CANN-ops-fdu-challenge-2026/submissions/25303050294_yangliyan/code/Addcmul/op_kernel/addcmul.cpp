// Kernel侧核函数实现
#include "kernel_operator.h"

#include <type_traits>

#include "addcmul_tiling.h"
#include "tiling_key_addcmul.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <class DT>
__aicore__ inline int64_t CalcBroadcastOffset(int64_t linearIdx, uint32_t dimNum, const int64_t *shape,
                                              const int64_t *strides)
{
    int64_t offset = 0;
    int64_t tmp = linearIdx;
    for (int32_t d = static_cast<int32_t>(dimNum) - 1; d >= 0; --d) {
        int64_t coord = tmp % shape[d];
        tmp /= shape[d];
        offset += coord * strides[d];
    }
    return offset;
}

__aicore__ inline uint32_t CalcCoreStart(uint32_t coreIdx, uint32_t coreSize, uint32_t coreRemain)
{
    return coreIdx * coreSize + (coreIdx < coreRemain ? coreIdx : coreRemain);
}

__aicore__ inline uint32_t CalcCoreLength(uint32_t coreIdx, uint32_t coreSize, uint32_t coreRemain)
{
    return coreSize + (coreIdx < coreRemain ? 1U : 0U);
}

template <class DT>
class KernelAddcmul {
public:
    __aicore__ inline KernelAddcmul() {}
    __aicore__ inline void Init(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
                                uint32_t totalLength, uint32_t alignNum, uint32_t blockSize, uint32_t coreSize,
                                uint32_t coreRemain)
    {
        this->tileLength = blockSize;
        uint32_t coreIdx = GetBlockIdx();
        this->actualLength = CalcCoreLength(coreIdx, coreSize, coreRemain);
        uint32_t startPointer = CalcCoreStart(coreIdx, coreSize, coreRemain);

        inputDataGm.SetGlobalBuffer((__gm__ DT *)input_data + startPointer, this->actualLength);
        x1Gm.SetGlobalBuffer((__gm__ DT *)x1 + startPointer, this->actualLength);
        x2Gm.SetGlobalBuffer((__gm__ DT *)x2 + startPointer, this->actualLength);
        valueGm.SetGlobalBuffer((__gm__ DT *)value, 1);
        yGm.SetGlobalBuffer((__gm__ DT *)y + startPointer, this->actualLength);

        this->tileNum =
            this->actualLength / this->tileLength + (this->actualLength % this->tileLength > 0 ? 1U : 0U);

        pipe.InitBuffer(inQueueInputData, BUFFER_NUM, this->tileLength * sizeof(DT));
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileLength * sizeof(DT));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileLength * sizeof(DT));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DT));
        if constexpr (std::is_same_v<DT, half>) {
            this->hValue = valueGm.GetValue(0);
        } else if constexpr (std::is_same_v<DT, int8_t>) {
            pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(half));
            pipe.InitBuffer(tmpBuf2, this->tileLength * sizeof(half));
            this->iValue = static_cast<int32_t>(valueGm.GetValue(0));
        } else {
            this->mValue = valueGm.GetValue(0);
        }
    }

    __aicore__ inline void Process()
    {
        if (this->tileNum == 0 || this->actualLength == 0) {
            return;
        }
        int32_t loopCount = static_cast<int32_t>(this->tileNum);
        for (int32_t i = 0; i < loopCount - 1; ++i) {
            CopyIn(i, this->tileLength);
            Compute(this->tileLength);
            CopyOut(i, this->tileLength);
        }
        uint32_t tailLen = this->actualLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, tailLen);
        Compute(tailLen);
        CopyOut(loopCount - 1, tailLen);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
    {
        LocalTensor<DT> inputLocal = inQueueInputData.AllocTensor<DT>();
        LocalTensor<DT> x1Local = inQueueX1.AllocTensor<DT>();
        LocalTensor<DT> x2Local = inQueueX2.AllocTensor<DT>();
        CopyTensor(inputLocal, inputDataGm, progress * this->tileLength, length);
        CopyTensor(x1Local, x1Gm, progress * this->tileLength, length);
        CopyTensor(x2Local, x2Gm, progress * this->tileLength, length);
        inQueueInputData.EnQue(inputLocal);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void CopyTensor(LocalTensor<DT> local, GlobalTensor<DT> gm, uint32_t offset, uint32_t length)
    {
        uint32_t copyBytes = length * sizeof(DT);
        if (copyBytes % 32 == 0) {
            DataCopy(local, gm[offset], length);
        } else {
            DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            DataCopyPadExtParams<DT> padParams{true, 0, 0, static_cast<DT>(0)};
            DataCopyPad(local, gm[offset], copyParams, padParams);
        }
    }

    __aicore__ inline void Compute(uint32_t length)
    {
        LocalTensor<DT> inputLocal = inQueueInputData.DeQue<DT>();
        LocalTensor<DT> x1Local = inQueueX1.DeQue<DT>();
        LocalTensor<DT> x2Local = inQueueX2.DeQue<DT>();
        LocalTensor<DT> yLocal = outQueueY.AllocTensor<DT>();
        int32_t calCount = static_cast<int32_t>(length);

        if constexpr (std::is_same_v<DT, int8_t>) {
            LocalTensor<half> p1 = tmpBuf1.Get<half>();
            LocalTensor<half> p2 = tmpBuf2.Get<half>();
            half hVal = static_cast<half>(this->iValue);
            Cast(p1, x1Local, RoundMode::CAST_NONE, calCount);
            Cast(p2, x2Local, RoundMode::CAST_NONE, calCount);
            Mul(p1, p1, p2, calCount);
            Muls(p1, p1, hVal, calCount);
            Cast(p2, inputLocal, RoundMode::CAST_NONE, calCount);
            Add(p2, p1, p2, calCount);
            Cast(p1.ReinterpretCast<int16_t>(), p2, RoundMode::CAST_RINT, calCount);
            ShiftLeft(p1.ReinterpretCast<int16_t>(), p1.ReinterpretCast<int16_t>(), static_cast<int16_t>(8),
                      calCount);
            ShiftRight(p1.ReinterpretCast<int16_t>(), p1.ReinterpretCast<int16_t>(), static_cast<int16_t>(8),
                       calCount);
            Cast(p2, p1.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, calCount);
            Cast(yLocal, p2, RoundMode::CAST_NONE, calCount);
        } else if constexpr (std::is_same_v<DT, half>) {
            Mul(x1Local, x1Local, x2Local, calCount);
            Muls(x1Local, x1Local, this->hValue, calCount);
            Add(yLocal, x1Local, inputLocal, calCount);
        } else {
            Mul(x1Local, x1Local, x2Local, calCount);
            Muls(x1Local, x1Local, this->mValue, calCount);
            Add(yLocal, x1Local, inputLocal, calCount);
        }

        inQueueInputData.FreeTensor(inputLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
        outQueueY.EnQue<DT>(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        LocalTensor<DT> yLocal = outQueueY.DeQue<DT>();
        uint32_t copyBytes = length * sizeof(DT);
        if (copyBytes % 32 == 0) {
            DataCopy(yGm[progress * this->tileLength], yLocal, length);
        } else {
            DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            DataCopyPad(yGm[progress * this->tileLength], yLocal, copyParams);
        }
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueInputData, inQueueX1, inQueueX2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<TPosition::VECCALC> tmpBuf1, tmpBuf2;
    GlobalTensor<DT> inputDataGm;
    GlobalTensor<DT> x1Gm;
    GlobalTensor<DT> x2Gm;
    GlobalTensor<DT> valueGm;
    GlobalTensor<DT> yGm;
    uint32_t actualLength;
    uint32_t tileNum;
    uint32_t tileLength;
    half hValue;
    DT mValue;
    int32_t iValue;
};

template <class DT>
class KernelAddcmulBroadcast {
public:
    __aicore__ inline KernelAddcmulBroadcast() {}
    __aicore__ inline void Init(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
                                uint32_t inputDataLength, uint32_t x1Length, uint32_t x2Length, uint32_t totalLength,
                                uint32_t blockSize, uint32_t coreSize, uint32_t coreRemain, uint32_t dimNum,
                                const int64_t *shape, const int64_t *inDataStride, const int64_t *x1Stride,
                                const int64_t *x2Stride)
    {
        this->inputDataLength = inputDataLength;
        this->x1Length = x1Length;
        this->x2Length = x2Length;
        this->tileLength = blockSize;
        this->dimNum = dimNum;
        uint32_t coreIdx = GetBlockIdx();
        this->actualLength = CalcCoreLength(coreIdx, coreSize, coreRemain);
        this->startPointer = CalcCoreStart(coreIdx, coreSize, coreRemain);

        for (uint32_t i = 0; i < dimNum; ++i) {
            this->shape[i] = shape[i];
            this->inDataStride[i] = inDataStride[i];
            this->x1Stride[i] = x1Stride[i];
            this->x2Stride[i] = x2Stride[i];
        }

        inputDataGm.SetGlobalBuffer((__gm__ DT *)input_data, totalLength);
        x1Gm.SetGlobalBuffer((__gm__ DT *)x1, totalLength);
        x2Gm.SetGlobalBuffer((__gm__ DT *)x2, totalLength);
        valueGm.SetGlobalBuffer((__gm__ DT *)value, 1);
        yGm.SetGlobalBuffer((__gm__ DT *)y, totalLength);

        pipe.InitBuffer(inQueueInputData, BUFFER_NUM, this->tileLength * sizeof(DT));
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileLength * sizeof(DT));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileLength * sizeof(DT));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DT));
        if constexpr (std::is_same_v<DT, half>) {
            this->hValue = valueGm.GetValue(0);
        } else if constexpr (std::is_same_v<DT, int8_t>) {
            pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(half));
            pipe.InitBuffer(tmpBuf2, this->tileLength * sizeof(half));
            this->iValue = static_cast<int32_t>(valueGm.GetValue(0));
        } else {
            this->mValue = valueGm.GetValue(0);
        }
    }

    __aicore__ inline void Process()
    {
        if (this->actualLength == 0 || this->tileLength == 0) {
            return;
        }
        int32_t loopCount =
            static_cast<int32_t>(this->actualLength / this->tileLength + (this->actualLength % this->tileLength > 0));
        for (int32_t i = 0; i < loopCount - 1; ++i) {
            uint32_t position = this->startPointer + static_cast<uint32_t>(i) * this->tileLength;
            CopyIn(position, this->tileLength);
            Compute(this->tileLength);
            CopyOut(position, this->tileLength);
        }
        uint32_t position = this->startPointer + static_cast<uint32_t>(loopCount - 1) * this->tileLength;
        uint32_t tailLen = this->actualLength - this->tileLength * (loopCount - 1);
        CopyIn(position, tailLen);
        Compute(tailLen);
        CopyOut(position, tailLen);
    }

private:
    __aicore__ inline void GatherBroadcast(LocalTensor<DT> local, GlobalTensor<DT> gm, uint32_t tensorLength,
                                           const int64_t *strides, uint32_t position, uint32_t length)
    {
        if (tensorLength == 1) {
            DT scalar = gm.GetValue(0);
            if constexpr (std::is_same_v<DT, int8_t>) {
                LocalTensor<half> tmpHalf = tmpBuf1.Get<half>();
                Duplicate(tmpHalf, static_cast<half>(scalar), length);
                Cast(local, tmpHalf, RoundMode::CAST_NONE, static_cast<int32_t>(length));
            } else {
                Duplicate(local, scalar, length);
            }
            return;
        }
        for (uint32_t i = 0; i < length; ++i) {
            int64_t off = CalcBroadcastOffset<DT>(static_cast<int64_t>(position) + static_cast<int64_t>(i), this->dimNum,
                                                  this->shape, strides);
            local.SetValue(i, gm.GetValue(off));
        }
    }

    __aicore__ inline void CopyIn(uint32_t position, uint32_t length)
    {
        LocalTensor<DT> inputLocal = inQueueInputData.AllocTensor<DT>();
        LocalTensor<DT> x1Local = inQueueX1.AllocTensor<DT>();
        LocalTensor<DT> x2Local = inQueueX2.AllocTensor<DT>();
        GatherBroadcast(inputLocal, inputDataGm, this->inputDataLength, this->inDataStride, position, length);
        GatherBroadcast(x1Local, x1Gm, this->x1Length, this->x1Stride, position, length);
        GatherBroadcast(x2Local, x2Gm, this->x2Length, this->x2Stride, position, length);
        inQueueInputData.EnQue(inputLocal);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void Compute(uint32_t length)
    {
        LocalTensor<DT> inputLocal = inQueueInputData.DeQue<DT>();
        LocalTensor<DT> x1Local = inQueueX1.DeQue<DT>();
        LocalTensor<DT> x2Local = inQueueX2.DeQue<DT>();
        LocalTensor<DT> yLocal = outQueueY.AllocTensor<DT>();
        int32_t calCount = static_cast<int32_t>(length);

        if constexpr (std::is_same_v<DT, int8_t>) {
            LocalTensor<half> p1 = tmpBuf1.Get<half>();
            LocalTensor<half> p2 = tmpBuf2.Get<half>();
            half hVal = static_cast<half>(this->iValue);
            Cast(p1, x1Local, RoundMode::CAST_NONE, calCount);
            Cast(p2, x2Local, RoundMode::CAST_NONE, calCount);
            Mul(p1, p1, p2, calCount);
            Muls(p1, p1, hVal, calCount);
            Cast(p2, inputLocal, RoundMode::CAST_NONE, calCount);
            Add(p2, p1, p2, calCount);
            Cast(p1.ReinterpretCast<int16_t>(), p2, RoundMode::CAST_RINT, calCount);
            ShiftLeft(p1.ReinterpretCast<int16_t>(), p1.ReinterpretCast<int16_t>(), static_cast<int16_t>(8),
                      calCount);
            ShiftRight(p1.ReinterpretCast<int16_t>(), p1.ReinterpretCast<int16_t>(), static_cast<int16_t>(8),
                       calCount);
            Cast(p2, p1.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, calCount);
            Cast(yLocal, p2, RoundMode::CAST_NONE, calCount);
        } else if constexpr (std::is_same_v<DT, half>) {
            Mul(x1Local, x1Local, x2Local, calCount);
            Muls(x1Local, x1Local, this->hValue, calCount);
            Add(yLocal, x1Local, inputLocal, calCount);
        } else {
            Mul(x1Local, x1Local, x2Local, calCount);
            Muls(x1Local, x1Local, this->mValue, calCount);
            Add(yLocal, x1Local, inputLocal, calCount);
        }

        inQueueInputData.FreeTensor(inputLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
        outQueueY.EnQue<DT>(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t position, uint32_t length)
    {
        LocalTensor<DT> yLocal = outQueueY.DeQue<DT>();
        uint32_t copyBytes = length * sizeof(DT);
        if (copyBytes % 32 == 0) {
            DataCopy(yGm[position], yLocal, length);
        } else {
            DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            DataCopyPad(yGm[position], yLocal, copyParams);
        }
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueInputData, inQueueX1, inQueueX2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<TPosition::VECCALC> tmpBuf1, tmpBuf2;
    GlobalTensor<DT> inputDataGm;
    GlobalTensor<DT> x1Gm;
    GlobalTensor<DT> x2Gm;
    GlobalTensor<DT> valueGm;
    GlobalTensor<DT> yGm;
    int64_t shape[ADDCMUL_MAX_DIM];
    int64_t inDataStride[ADDCMUL_MAX_DIM];
    int64_t x1Stride[ADDCMUL_MAX_DIM];
    int64_t x2Stride[ADDCMUL_MAX_DIM];
    uint32_t actualLength;
    uint32_t tileLength;
    uint32_t startPointer;
    uint32_t inputDataLength;
    uint32_t x1Length;
    uint32_t x2Length;
    uint32_t dimNum;
    half hValue;
    DT mValue;
    int32_t iValue;
};

template <typename DT_INPUT_DATA>
__global__ __aicore__ void addcmul(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
                                   GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddcmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddcmulTilingData, tiling_data, tiling);

    if (tiling_data.totalLength == 0) {
        return;
    }

    if (tiling_data.isContiguous != 0) {
        KernelAddcmul<DT_INPUT_DATA> op;
        op.Init(input_data, x1, x2, value, y, tiling_data.totalLength, tiling_data.alignNum, tiling_data.blockSize,
                tiling_data.coreSize, tiling_data.coreRemain);
        op.Process();
    } else {
        KernelAddcmulBroadcast<DT_INPUT_DATA> op;
        op.Init(input_data, x1, x2, value, y, tiling_data.inputDataLength, tiling_data.x1Length, tiling_data.x2Length,
                tiling_data.totalLength, tiling_data.blockSize, tiling_data.coreSize, tiling_data.coreRemain,
                tiling_data.dimNum, tiling_data.shape, tiling_data.inDataStride, tiling_data.x1Stride,
                tiling_data.x2Stride);
        op.Process();
    }
}
