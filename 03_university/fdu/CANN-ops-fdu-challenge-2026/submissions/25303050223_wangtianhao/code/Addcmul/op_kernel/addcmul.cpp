#include "kernel_operator.h"

#include "addcmul_tiling.h"
#include "tiling_key_addcmul.h"

namespace {
constexpr uint32_t ADDCMUL_MAX_DIMS_KERNEL = 16;
constexpr uint32_t TILE_DATA_NUM = 4096;
constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t FAST_STATE_UNKNOWN = 0;
constexpr uint32_t FAST_STATE_BROADCAST = 1;
constexpr uint32_t FAST_STATE_CONTIGUOUS = 2;
}

template <typename T>
__aicore__ inline T AddcmulCompute(T inputVal, T x1Val, T x2Val, T valueVal)
{
    return static_cast<T>(inputVal + x1Val * x2Val * valueVal);
}

template <>
__aicore__ inline half AddcmulCompute<half>(half inputVal, half x1Val, half x2Val, half valueVal)
{
    half mul12 = static_cast<half>(static_cast<float>(x1Val) * static_cast<float>(x2Val));
    half scaled = static_cast<half>(static_cast<float>(mul12) * static_cast<float>(valueVal));
    return static_cast<half>(static_cast<float>(inputVal) + static_cast<float>(scaled));
}

template <>
__aicore__ inline int32_t AddcmulCompute<int32_t>(int32_t inputVal, int32_t x1Val, int32_t x2Val, int32_t valueVal)
{
    int64_t result = static_cast<int64_t>(inputVal) +
        static_cast<int64_t>(x1Val) * static_cast<int64_t>(x2Val) * static_cast<int64_t>(valueVal);
    return static_cast<int32_t>(result);
}

template <>
__aicore__ inline int8_t AddcmulCompute<int8_t>(int8_t inputVal, int8_t x1Val, int8_t x2Val, int8_t valueVal)
{
    uint32_t a = static_cast<uint32_t>(static_cast<int32_t>(inputVal));
    uint32_t b = static_cast<uint32_t>(static_cast<int32_t>(x1Val));
    uint32_t c = static_cast<uint32_t>(static_cast<int32_t>(x2Val));
    uint32_t v = static_cast<uint32_t>(static_cast<int32_t>(valueVal));
    uint8_t low8 = static_cast<uint8_t>(a + b * c * v);

    if (low8 <= 127) {
        return static_cast<int8_t>(low8);
    }
    return static_cast<int8_t>(static_cast<int16_t>(low8) - 256);
}

template <class DT_INPUT_DATA>
class KernelAddcmul {
public:
    __aicore__ inline KernelAddcmul() {}

    __aicore__ inline void Init(
        GM_ADDR input_data,
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR value,
        GM_ADDR y,
        uint32_t length,
        uint32_t blockLength,
        uint32_t rank,
        uint32_t isNoBroadcast,
        const uint32_t *outShape,
        const uint32_t *inputStride,
        const uint32_t *x1Stride,
        const uint32_t *x2Stride)
    {
        this->length = length;
        this->blockLength = blockLength;
        this->rank = rank;
        this->isNoBroadcast = isNoBroadcast;

        uint32_t coreId = AscendC::GetBlockIdx();
        this->start = coreId * blockLength;

        if (this->start >= length) {
            this->coreLength = 0;
        } else {
            uint32_t remain = length - this->start;
            this->coreLength = remain > blockLength ? blockLength : remain;
        }

        for (uint32_t i = 0; i < ADDCMUL_MAX_DIMS_KERNEL; ++i) {
            this->outShape[i] = outShape[i];
            this->inputStride[i] = inputStride[i];
            this->x1Stride[i] = x1Stride[i];
            this->x2Stride[i] = x2Stride[i];
        }

        inputGm.SetGlobalBuffer((__gm__ DT_INPUT_DATA *)input_data);
        x1Gm.SetGlobalBuffer((__gm__ DT_INPUT_DATA *)x1);
        x2Gm.SetGlobalBuffer((__gm__ DT_INPUT_DATA *)x2);
        valueGm.SetGlobalBuffer((__gm__ DT_INPUT_DATA *)value);
        yGm.SetGlobalBuffer((__gm__ DT_INPUT_DATA *)y);

        InitVectorBuffers();
    }

    __aicore__ inline void Process()
    {
        if (this->coreLength == 0) {
            return;
        }

        if (this->isNoBroadcast != 0) {
            ProcessNoBroadcast();
        } else {
            ProcessBroadcast();
        }
    }

private:
    __aicore__ inline void InitVectorBuffers();

    __aicore__ inline void ProcessNoBroadcast();

    __aicore__ inline void ProcessBroadcast()
    {
        ProcessBroadcastTailScalarFast();
    }

    __aicore__ inline void CalcInputOffsets(
        uint32_t linearIndex,
        uint32_t &inputOffset,
        uint32_t &x1Offset,
        uint32_t &x2Offset)
    {
        uint32_t tmp = linearIndex;
        inputOffset = 0;
        x1Offset = 0;
        x2Offset = 0;

        for (int32_t axis = static_cast<int32_t>(this->rank) - 1; axis >= 0; --axis) {
            uint32_t coord = tmp % this->outShape[axis];
            tmp = tmp / this->outShape[axis];

            inputOffset += coord * this->inputStride[axis];
            x1Offset += coord * this->x1Stride[axis];
            x2Offset += coord * this->x2Stride[axis];
        }
    }

    __aicore__ inline uint32_t GetAxisState(uint32_t stride, uint32_t expectedStride)
    {
        if (stride == 0) {
            return FAST_STATE_BROADCAST;
        }
        if (stride == expectedStride) {
            return FAST_STATE_CONTIGUOUS;
        }
        return FAST_STATE_UNKNOWN;
    }

    __aicore__ inline bool UpdateFastState(uint32_t dim, uint32_t axisState, uint32_t &state)
    {
        if (dim == 1) {
            return true;
        }

        if (state == FAST_STATE_UNKNOWN) {
            state = axisState;
            return true;
        }

        return state == axisState;
    }

    __aicore__ inline bool GetTailFastInfo(
        uint32_t &fastDim,
        uint32_t &inputStep,
        uint32_t &x1Step,
        uint32_t &x2Step)
    {
        fastDim = 1;

        uint32_t inputState = FAST_STATE_UNKNOWN;
        uint32_t x1State = FAST_STATE_UNKNOWN;
        uint32_t x2State = FAST_STATE_UNKNOWN;

        uint32_t inputExpected = 1;
        uint32_t x1Expected = 1;
        uint32_t x2Expected = 1;

        for (int32_t axis = static_cast<int32_t>(this->rank) - 1; axis >= 0; --axis) {
            uint32_t dim = this->outShape[axis];
            if (dim == 0) {
                return false;
            }

            uint32_t inputAxisState = GetAxisState(this->inputStride[axis], inputExpected);
            uint32_t x1AxisState = GetAxisState(this->x1Stride[axis], x1Expected);
            uint32_t x2AxisState = GetAxisState(this->x2Stride[axis], x2Expected);

            if (inputAxisState == FAST_STATE_UNKNOWN ||
                x1AxisState == FAST_STATE_UNKNOWN ||
                x2AxisState == FAST_STATE_UNKNOWN) {
                break;
            }

            if (!UpdateFastState(dim, inputAxisState, inputState) ||
                !UpdateFastState(dim, x1AxisState, x1State) ||
                !UpdateFastState(dim, x2AxisState, x2State)) {
                break;
            }

            fastDim *= dim;

            if (inputAxisState == FAST_STATE_CONTIGUOUS) {
                inputExpected *= dim;
            }
            if (x1AxisState == FAST_STATE_CONTIGUOUS) {
                x1Expected *= dim;
            }
            if (x2AxisState == FAST_STATE_CONTIGUOUS) {
                x2Expected *= dim;
            }
        }

        inputStep = (inputState == FAST_STATE_CONTIGUOUS) ? 1 : 0;
        x1Step = (x1State == FAST_STATE_CONTIGUOUS) ? 1 : 0;
        x2Step = (x2State == FAST_STATE_CONTIGUOUS) ? 1 : 0;

        return fastDim > 1;
    }

    __aicore__ inline void ProcessBroadcastScalar()
    {
        DT_INPUT_DATA valueScalar = valueGm.GetValue(0);

        for (uint32_t i = 0; i < this->coreLength; ++i) {
            uint32_t outOffset = this->start + i;

            uint32_t inputOffset = 0;
            uint32_t x1Offset = 0;
            uint32_t x2Offset = 0;
            CalcInputOffsets(outOffset, inputOffset, x1Offset, x2Offset);

            DT_INPUT_DATA yVal = AddcmulCompute<DT_INPUT_DATA>(
                inputGm.GetValue(inputOffset),
                x1Gm.GetValue(x1Offset),
                x2Gm.GetValue(x2Offset),
                valueScalar);

            yGm.SetValue(outOffset, yVal);
        }
    }

    __aicore__ inline void ProcessBroadcastTailScalarFast()
    {
        uint32_t fastDim = 1;
        uint32_t inputStep = 0;
        uint32_t x1Step = 0;
        uint32_t x2Step = 0;

        if (!GetTailFastInfo(fastDim, inputStep, x1Step, x2Step)) {
            ProcessBroadcastScalar();
            return;
        }

        DT_INPUT_DATA valueScalar = valueGm.GetValue(0);
        uint32_t end = this->start + this->coreLength;
        uint32_t outOffset = this->start;

        while (outOffset < end) {
            uint32_t remain = end - outOffset;
            uint32_t tailRemain = fastDim - (outOffset % fastDim);
            uint32_t processDataNum = remain < tailRemain ? remain : tailRemain;

            uint32_t inputOffset = 0;
            uint32_t x1Offset = 0;
            uint32_t x2Offset = 0;
            CalcInputOffsets(outOffset, inputOffset, x1Offset, x2Offset);

            for (uint32_t i = 0; i < processDataNum; ++i) {
                DT_INPUT_DATA yVal = AddcmulCompute<DT_INPUT_DATA>(
                    inputGm.GetValue(inputOffset),
                    x1Gm.GetValue(x1Offset),
                    x2Gm.GetValue(x2Offset),
                    valueScalar);

                yGm.SetValue(outOffset + i, yVal);

                inputOffset += inputStep;
                x1Offset += x1Step;
                x2Offset += x2Step;
            }

            outOffset += processDataNum;
        }
    }

    __aicore__ inline bool Is32BAligned(uint32_t dataNum)
    {
        return ((dataNum * sizeof(DT_INPUT_DATA)) % 32) == 0;
    }

    __aicore__ inline bool IsInputContiguous()
    {
        uint32_t running = 1;
        for (int32_t axis = static_cast<int32_t>(this->rank) - 1; axis >= 0; --axis) {
            if (this->inputStride[axis] != running) {
                return false;
            }
            running *= this->outShape[axis];
        }
        return true;
    }

    __aicore__ inline bool IsX1Contiguous()
    {
        uint32_t running = 1;
        for (int32_t axis = static_cast<int32_t>(this->rank) - 1; axis >= 0; --axis) {
            if (this->x1Stride[axis] != running) {
                return false;
            }
            running *= this->outShape[axis];
        }
        return true;
    }

    __aicore__ inline bool IsX2Contiguous()
    {
        uint32_t running = 1;
        for (int32_t axis = static_cast<int32_t>(this->rank) - 1; axis >= 0; --axis) {
            if (this->x2Stride[axis] != running) {
                return false;
            }
            running *= this->outShape[axis];
        }
        return true;
    }

    __aicore__ inline bool IsX1ScalarBroadcast()
    {
        for (uint32_t i = 0; i < this->rank; ++i) {
            if (this->x1Stride[i] != 0) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline bool IsX2ScalarBroadcast()
    {
        for (uint32_t i = 0; i < this->rank; ++i) {
            if (this->x2Stride[i] != 0) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void CopyGmToLocalPad(
        AscendC::LocalTensor<DT_INPUT_DATA> dst,
        AscendC::GlobalTensor<DT_INPUT_DATA> src,
        uint32_t dataNum)
    {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = dataNum * sizeof(DT_INPUT_DATA);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        AscendC::DataCopyPadExtParams<DT_INPUT_DATA> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;

        AscendC::DataCopyPad(dst, src, copyParams, padParams);
    }

    __aicore__ inline void CopyLocalToGmPad(
        AscendC::GlobalTensor<DT_INPUT_DATA> dst,
        AscendC::LocalTensor<DT_INPUT_DATA> src,
        uint32_t dataNum)
    {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = dataNum * sizeof(DT_INPUT_DATA);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        AscendC::DataCopyPad(dst, src, copyParams);
    }

    __aicore__ inline void CopyIn3(
        uint32_t gmOffset,
        uint32_t processDataNum,
        AscendC::LocalTensor<DT_INPUT_DATA> inputLocal,
        AscendC::LocalTensor<DT_INPUT_DATA> x1Local,
        AscendC::LocalTensor<DT_INPUT_DATA> x2Local)
    {
        if (Is32BAligned(processDataNum)) {
            AscendC::DataCopy(inputLocal, inputGm[gmOffset], processDataNum);
            AscendC::DataCopy(x1Local, x1Gm[gmOffset], processDataNum);
            AscendC::DataCopy(x2Local, x2Gm[gmOffset], processDataNum);
        } else {
            CopyGmToLocalPad(inputLocal, inputGm[gmOffset], processDataNum);
            CopyGmToLocalPad(x1Local, x1Gm[gmOffset], processDataNum);
            CopyGmToLocalPad(x2Local, x2Gm[gmOffset], processDataNum);
        }
    }

    __aicore__ inline void CopyIn2X1(
        uint32_t gmOffset,
        uint32_t processDataNum,
        AscendC::LocalTensor<DT_INPUT_DATA> inputLocal,
        AscendC::LocalTensor<DT_INPUT_DATA> x1Local)
    {
        if (Is32BAligned(processDataNum)) {
            AscendC::DataCopy(inputLocal, inputGm[gmOffset], processDataNum);
            AscendC::DataCopy(x1Local, x1Gm[gmOffset], processDataNum);
        } else {
            CopyGmToLocalPad(inputLocal, inputGm[gmOffset], processDataNum);
            CopyGmToLocalPad(x1Local, x1Gm[gmOffset], processDataNum);
        }
    }

    __aicore__ inline void CopyIn2X2(
        uint32_t gmOffset,
        uint32_t processDataNum,
        AscendC::LocalTensor<DT_INPUT_DATA> inputLocal,
        AscendC::LocalTensor<DT_INPUT_DATA> x2Local)
    {
        if (Is32BAligned(processDataNum)) {
            AscendC::DataCopy(inputLocal, inputGm[gmOffset], processDataNum);
            AscendC::DataCopy(x2Local, x2Gm[gmOffset], processDataNum);
        } else {
            CopyGmToLocalPad(inputLocal, inputGm[gmOffset], processDataNum);
            CopyGmToLocalPad(x2Local, x2Gm[gmOffset], processDataNum);
        }
    }

    __aicore__ inline void CopyOutY(
        uint32_t gmOffset,
        uint32_t processDataNum,
        AscendC::LocalTensor<DT_INPUT_DATA> yLocal)
    {
        if (Is32BAligned(processDataNum)) {
            AscendC::DataCopy(yGm[gmOffset], yLocal, processDataNum);
        } else {
            CopyLocalToGmPad(yGm[gmOffset], yLocal, processDataNum);
        }
    }

    __aicore__ inline void ProcessNoBroadcastVector()
    {
        DT_INPUT_DATA valueScalar = valueGm.GetValue(0);
        uint32_t loopCount = (this->coreLength + TILE_DATA_NUM - 1) / TILE_DATA_NUM;

        for (uint32_t loop = 0; loop < loopCount; ++loop) {
            uint32_t offset = loop * TILE_DATA_NUM;
            uint32_t remain = this->coreLength - offset;
            uint32_t processDataNum = remain > TILE_DATA_NUM ? TILE_DATA_NUM : remain;
            uint32_t gmOffset = this->start + offset;

            AscendC::LocalTensor<DT_INPUT_DATA> inputLocal = inQueueInput.AllocTensor<DT_INPUT_DATA>();
            AscendC::LocalTensor<DT_INPUT_DATA> x1Local = inQueueX1.AllocTensor<DT_INPUT_DATA>();
            AscendC::LocalTensor<DT_INPUT_DATA> x2Local = inQueueX2.AllocTensor<DT_INPUT_DATA>();

            CopyIn3(gmOffset, processDataNum, inputLocal, x1Local, x2Local);

            inQueueInput.EnQue(inputLocal);
            inQueueX1.EnQue(x1Local);
            inQueueX2.EnQue(x2Local);

            AscendC::LocalTensor<DT_INPUT_DATA> inputDeq = inQueueInput.DeQue<DT_INPUT_DATA>();
            AscendC::LocalTensor<DT_INPUT_DATA> x1Deq = inQueueX1.DeQue<DT_INPUT_DATA>();
            AscendC::LocalTensor<DT_INPUT_DATA> x2Deq = inQueueX2.DeQue<DT_INPUT_DATA>();
            AscendC::LocalTensor<DT_INPUT_DATA> yLocal = outQueueY.AllocTensor<DT_INPUT_DATA>();

            AscendC::LocalTensor<DT_INPUT_DATA> tmp1 = tmpBuf1.Get<DT_INPUT_DATA>();
            AscendC::LocalTensor<DT_INPUT_DATA> tmp2 = tmpBuf2.Get<DT_INPUT_DATA>();

            AscendC::Mul(tmp1, x1Deq, x2Deq, processDataNum);
            AscendC::Muls(tmp2, tmp1, valueScalar, processDataNum);
            AscendC::Add(yLocal, inputDeq, tmp2, processDataNum);

            outQueueY.EnQue<DT_INPUT_DATA>(yLocal);

            inQueueInput.FreeTensor(inputDeq);
            inQueueX1.FreeTensor(x1Deq);
            inQueueX2.FreeTensor(x2Deq);

            AscendC::LocalTensor<DT_INPUT_DATA> yDeq = outQueueY.DeQue<DT_INPUT_DATA>();
            CopyOutY(gmOffset, processDataNum, yDeq);
            outQueueY.FreeTensor(yDeq);
        }
    }

    __aicore__ inline void ProcessX2ScalarBroadcastVector()
    {
        DT_INPUT_DATA valueScalar = valueGm.GetValue(0);
        DT_INPUT_DATA x2Scalar = x2Gm.GetValue(0);
        uint32_t loopCount = (this->coreLength + TILE_DATA_NUM - 1) / TILE_DATA_NUM;

        for (uint32_t loop = 0; loop < loopCount; ++loop) {
            uint32_t offset = loop * TILE_DATA_NUM;
            uint32_t remain = this->coreLength - offset;
            uint32_t processDataNum = remain > TILE_DATA_NUM ? TILE_DATA_NUM : remain;
            uint32_t gmOffset = this->start + offset;

            AscendC::LocalTensor<DT_INPUT_DATA> inputLocal = inQueueInput.AllocTensor<DT_INPUT_DATA>();
            AscendC::LocalTensor<DT_INPUT_DATA> x1Local = inQueueX1.AllocTensor<DT_INPUT_DATA>();

            CopyIn2X1(gmOffset, processDataNum, inputLocal, x1Local);

            inQueueInput.EnQue(inputLocal);
            inQueueX1.EnQue(x1Local);

            AscendC::LocalTensor<DT_INPUT_DATA> inputDeq = inQueueInput.DeQue<DT_INPUT_DATA>();
            AscendC::LocalTensor<DT_INPUT_DATA> x1Deq = inQueueX1.DeQue<DT_INPUT_DATA>();
            AscendC::LocalTensor<DT_INPUT_DATA> yLocal = outQueueY.AllocTensor<DT_INPUT_DATA>();

            AscendC::LocalTensor<DT_INPUT_DATA> tmp1 = tmpBuf1.Get<DT_INPUT_DATA>();
            AscendC::LocalTensor<DT_INPUT_DATA> tmp2 = tmpBuf2.Get<DT_INPUT_DATA>();

            AscendC::Muls(tmp1, x1Deq, x2Scalar, processDataNum);
            AscendC::Muls(tmp2, tmp1, valueScalar, processDataNum);
            AscendC::Add(yLocal, inputDeq, tmp2, processDataNum);

            outQueueY.EnQue<DT_INPUT_DATA>(yLocal);

            inQueueInput.FreeTensor(inputDeq);
            inQueueX1.FreeTensor(x1Deq);

            AscendC::LocalTensor<DT_INPUT_DATA> yDeq = outQueueY.DeQue<DT_INPUT_DATA>();
            CopyOutY(gmOffset, processDataNum, yDeq);
            outQueueY.FreeTensor(yDeq);
        }
    }

    __aicore__ inline void ProcessX1ScalarBroadcastVector()
    {
        DT_INPUT_DATA valueScalar = valueGm.GetValue(0);
        DT_INPUT_DATA x1Scalar = x1Gm.GetValue(0);
        uint32_t loopCount = (this->coreLength + TILE_DATA_NUM - 1) / TILE_DATA_NUM;

        for (uint32_t loop = 0; loop < loopCount; ++loop) {
            uint32_t offset = loop * TILE_DATA_NUM;
            uint32_t remain = this->coreLength - offset;
            uint32_t processDataNum = remain > TILE_DATA_NUM ? TILE_DATA_NUM : remain;
            uint32_t gmOffset = this->start + offset;

            AscendC::LocalTensor<DT_INPUT_DATA> inputLocal = inQueueInput.AllocTensor<DT_INPUT_DATA>();
            AscendC::LocalTensor<DT_INPUT_DATA> x2Local = inQueueX2.AllocTensor<DT_INPUT_DATA>();

            CopyIn2X2(gmOffset, processDataNum, inputLocal, x2Local);

            inQueueInput.EnQue(inputLocal);
            inQueueX2.EnQue(x2Local);

            AscendC::LocalTensor<DT_INPUT_DATA> inputDeq = inQueueInput.DeQue<DT_INPUT_DATA>();
            AscendC::LocalTensor<DT_INPUT_DATA> x2Deq = inQueueX2.DeQue<DT_INPUT_DATA>();
            AscendC::LocalTensor<DT_INPUT_DATA> yLocal = outQueueY.AllocTensor<DT_INPUT_DATA>();

            AscendC::LocalTensor<DT_INPUT_DATA> tmp1 = tmpBuf1.Get<DT_INPUT_DATA>();
            AscendC::LocalTensor<DT_INPUT_DATA> tmp2 = tmpBuf2.Get<DT_INPUT_DATA>();

            AscendC::Muls(tmp1, x2Deq, x1Scalar, processDataNum);
            AscendC::Muls(tmp2, tmp1, valueScalar, processDataNum);
            AscendC::Add(yLocal, inputDeq, tmp2, processDataNum);

            outQueueY.EnQue<DT_INPUT_DATA>(yLocal);

            inQueueInput.FreeTensor(inputDeq);
            inQueueX2.FreeTensor(x2Deq);

            AscendC::LocalTensor<DT_INPUT_DATA> yDeq = outQueueY.DeQue<DT_INPUT_DATA>();
            CopyOutY(gmOffset, processDataNum, yDeq);
            outQueueY.FreeTensor(yDeq);
        }
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueInput;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX1;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf2;

    AscendC::GlobalTensor<DT_INPUT_DATA> inputGm;
    AscendC::GlobalTensor<DT_INPUT_DATA> x1Gm;
    AscendC::GlobalTensor<DT_INPUT_DATA> x2Gm;
    AscendC::GlobalTensor<DT_INPUT_DATA> valueGm;
    AscendC::GlobalTensor<DT_INPUT_DATA> yGm;

    uint32_t length;
    uint32_t blockLength;
    uint32_t rank;
    uint32_t isNoBroadcast;
    uint32_t start;
    uint32_t coreLength;

    uint32_t outShape[ADDCMUL_MAX_DIMS_KERNEL];
    uint32_t inputStride[ADDCMUL_MAX_DIMS_KERNEL];
    uint32_t x1Stride[ADDCMUL_MAX_DIMS_KERNEL];
    uint32_t x2Stride[ADDCMUL_MAX_DIMS_KERNEL];
};

template <class DT_INPUT_DATA>
__aicore__ inline void KernelAddcmul<DT_INPUT_DATA>::InitVectorBuffers()
{
}

template <>
__aicore__ inline void KernelAddcmul<half>::InitVectorBuffers()
{
    pipe.InitBuffer(inQueueInput, BUFFER_NUM, TILE_DATA_NUM * sizeof(half));
    pipe.InitBuffer(inQueueX1, BUFFER_NUM, TILE_DATA_NUM * sizeof(half));
    pipe.InitBuffer(inQueueX2, BUFFER_NUM, TILE_DATA_NUM * sizeof(half));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_DATA_NUM * sizeof(half));
    pipe.InitBuffer(tmpBuf1, TILE_DATA_NUM * sizeof(half));
    pipe.InitBuffer(tmpBuf2, TILE_DATA_NUM * sizeof(half));
}

template <>
__aicore__ inline void KernelAddcmul<float>::InitVectorBuffers()
{
    pipe.InitBuffer(inQueueInput, BUFFER_NUM, TILE_DATA_NUM * sizeof(float));
    pipe.InitBuffer(inQueueX1, BUFFER_NUM, TILE_DATA_NUM * sizeof(float));
    pipe.InitBuffer(inQueueX2, BUFFER_NUM, TILE_DATA_NUM * sizeof(float));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_DATA_NUM * sizeof(float));
    pipe.InitBuffer(tmpBuf1, TILE_DATA_NUM * sizeof(float));
    pipe.InitBuffer(tmpBuf2, TILE_DATA_NUM * sizeof(float));
}

template <>
__aicore__ inline void KernelAddcmul<int32_t>::InitVectorBuffers()
{
    pipe.InitBuffer(inQueueInput, BUFFER_NUM, TILE_DATA_NUM * sizeof(int32_t));
    pipe.InitBuffer(inQueueX1, BUFFER_NUM, TILE_DATA_NUM * sizeof(int32_t));
    pipe.InitBuffer(inQueueX2, BUFFER_NUM, TILE_DATA_NUM * sizeof(int32_t));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_DATA_NUM * sizeof(int32_t));
    pipe.InitBuffer(tmpBuf1, TILE_DATA_NUM * sizeof(int32_t));
    pipe.InitBuffer(tmpBuf2, TILE_DATA_NUM * sizeof(int32_t));
}

template <class DT_INPUT_DATA>
__aicore__ inline void KernelAddcmul<DT_INPUT_DATA>::ProcessNoBroadcast()
{
    DT_INPUT_DATA valueScalar = valueGm.GetValue(0);

    for (uint32_t i = 0; i < this->coreLength; ++i) {
        uint32_t offset = this->start + i;

        DT_INPUT_DATA yVal = AddcmulCompute<DT_INPUT_DATA>(
            inputGm.GetValue(offset),
            x1Gm.GetValue(offset),
            x2Gm.GetValue(offset),
            valueScalar);

        yGm.SetValue(offset, yVal);
    }
}

template <>
__aicore__ inline void KernelAddcmul<half>::ProcessNoBroadcast()
{
    ProcessNoBroadcastVector();
}

template <>
__aicore__ inline void KernelAddcmul<float>::ProcessNoBroadcast()
{
    ProcessNoBroadcastVector();
}

template <>
__aicore__ inline void KernelAddcmul<int32_t>::ProcessNoBroadcast()
{
    ProcessNoBroadcastVector();
}

template <>
__aicore__ inline void KernelAddcmul<half>::ProcessBroadcast()
{
    if (IsInputContiguous() && IsX1Contiguous() && IsX2ScalarBroadcast()) {
        ProcessX2ScalarBroadcastVector();
        return;
    }

    if (IsInputContiguous() && IsX2Contiguous() && IsX1ScalarBroadcast()) {
        ProcessX1ScalarBroadcastVector();
        return;
    }

    ProcessBroadcastTailScalarFast();
}

template <>
__aicore__ inline void KernelAddcmul<float>::ProcessBroadcast()
{
    if (IsInputContiguous() && IsX1Contiguous() && IsX2ScalarBroadcast()) {
        ProcessX2ScalarBroadcastVector();
        return;
    }

    if (IsInputContiguous() && IsX2Contiguous() && IsX1ScalarBroadcast()) {
        ProcessX1ScalarBroadcastVector();
        return;
    }

    ProcessBroadcastTailScalarFast();
}

template <>
__aicore__ inline void KernelAddcmul<int32_t>::ProcessBroadcast()
{
    if (IsInputContiguous() && IsX1Contiguous() && IsX2ScalarBroadcast()) {
        ProcessX2ScalarBroadcastVector();
        return;
    }

    if (IsInputContiguous() && IsX2Contiguous() && IsX1ScalarBroadcast()) {
        ProcessX1ScalarBroadcastVector();
        return;
    }

    ProcessBroadcastTailScalarFast();
}

template <>
__aicore__ inline void KernelAddcmul<int8_t>::ProcessBroadcast()
{
    ProcessBroadcastTailScalarFast();
}

template <typename DT_INPUT_DATA>
__global__ __aicore__ void addcmul(
    GM_ADDR input_data,
    GM_ADDR x1,
    GM_ADDR x2,
    GM_ADDR value,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddcmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddcmulTilingData, tiling_data, tiling);

    KernelAddcmul<DT_INPUT_DATA> op;
    op.Init(
        input_data,
        x1,
        x2,
        value,
        y,
        tiling_data.length,
        tiling_data.blockLength,
        tiling_data.rank,
        tiling_data.isNoBroadcast,
        tiling_data.outShape,
        tiling_data.inputStride,
        tiling_data.x1Stride,
        tiling_data.x2Stride);

    op.Process();
}