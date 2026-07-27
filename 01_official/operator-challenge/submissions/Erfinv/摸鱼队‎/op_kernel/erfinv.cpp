#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template <typename T>
class KernelErfinv {
public:
    __aicore__ inline KernelErfinv() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR out,
                                uint32_t bigNum, uint32_t bigLength, uint32_t smallLength, TPipe* pipeIn) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreDataNum;
        if (GetBlockIdx() < bigNum) {
            coreDataNum = bigLength;
            xGm.SetGlobalBuffer((__gm__ T*)input + bigLength * GetBlockIdx(), bigLength);
            yGm.SetGlobalBuffer((__gm__ T*)out + bigLength * GetBlockIdx(), bigLength);
        } else {
            coreDataNum = smallLength;
            xGm.SetGlobalBuffer((__gm__ T*)input + bigLength * bigNum + smallLength * (GetBlockIdx() - bigNum), smallLength);
            yGm.SetGlobalBuffer((__gm__ T*)out + bigLength * bigNum + smallLength * (GetBlockIdx() - bigNum), smallLength);
        }
        if constexpr (std::is_same_v<DTYPE_X, float>)
        {
            this->tileDataLenght = 7808;
        }
        else if constexpr (std::is_same_v<DTYPE_X, half>)
        {
            this->tileDataLenght = 9792;
        }
        else
        {
            this->tileDataLenght = 7808;
        }

        this->loopCount = coreDataNum/this->tileDataLenght;
        this->tailDataLenght = coreDataNum - this->tileDataLenght*this->loopCount;
        if(this->tailDataLenght == 0){this->tailDataLenght = this->tileDataLenght; this->loopCount--;}

        pipeIn->InitBuffer(inQueue, BUFFER_NUM, this->tileDataLenght * sizeof(T));
        pipeIn->InitBuffer(outQueue, BUFFER_NUM, this->tileDataLenght * sizeof(T));
        if constexpr (std::is_same_v<T, float>)
        {
            pipeIn->InitBuffer(tmpBuf, 2*this->tileDataLenght * sizeof(float));
            pipeIn->InitBuffer(maskBuf, this->tileDataLenght * sizeof(uint8_t));
        }
        else if constexpr (std::is_same_v<T, half>)
        {
            pipeIn->InitBuffer(tmpBuf, 3*this->tileDataLenght * sizeof(float));
        }
        else
        {
            pipeIn->InitBuffer(tmpBuf, 4*this->tileDataLenght * sizeof(float));
            pipeIn->InitBuffer(maskBuf, this->tileDataLenght * sizeof(uint8_t));
        }
        
    }
    __aicore__ inline void Process() {
        this->processDataLenght = this->tileDataLenght;
        for (int32_t i = 0; i < this->loopCount; i++) {
            CopyIn(i);
            Compute(this->processDataLenght);
            CopyOut(i);
        }
        this->processDataLenght = this->tailDataLenght;
        CopyIn(loopCount);
        Compute(this->processDataLenght);
        CopyOut(loopCount);
    }
    __aicore__ inline void Compute(uint32_t len)
    {
        LocalTensor<T> inputLocal = inQueue.DeQue<T>();
        LocalTensor<T> outLocal = outQueue.AllocTensor<T>();

        if constexpr (std::is_same_v<T, float>)
        {
            LocalTensor<float> tmp = tmpBuf.Get<float>();
            LocalTensor<float> z = tmp[this->tileDataLenght];
            LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();

            // <0.7
            Mul(z, inputLocal, inputLocal, len);

            Muls(tmp, z, (float)-0.140543331, len);
            Adds(tmp, tmp, (float)0.914624893, len);
            Mul(tmp, tmp, z, len);
            Adds(tmp, tmp, (float)-1.645349621, len);
            Mul(tmp, tmp, z, len);
            Adds(outLocal, tmp, (float)0.886226899, len);

            Muls(tmp, z, (float)0.012229801, len);
            Adds(tmp, tmp, (float)-0.329097515, len);
            Mul(tmp, tmp, z, len);
            Adds(tmp, tmp, (float)1.442710462, len);
            Mul(tmp, tmp, z, len);
            Adds(tmp, tmp, (float)-2.118377725, len);
            Mul(tmp, tmp, z, len);
            Adds(tmp, tmp, (float)1, len);

            Mul(outLocal, inputLocal, outLocal, len);
            Div(outLocal, outLocal, tmp, len);
            Abs(z, inputLocal, len);
            CompareScalar(mask, z, (float)0.7, CMPMODE::LT, len);

            ///>0.7
            ShiftRight(inputLocal.template ReinterpretCast<int32_t>(), inputLocal.template ReinterpretCast<int32_t>(), 31, len);
            ShiftLeft(inputLocal.template ReinterpretCast<int32_t>(), inputLocal.template ReinterpretCast<int32_t>(), 31, len);
            Muls(z, z, (float)(-1), len);
            Adds(z, z, (float)(1), len);
            Muls(z, z, (float)(0.5), len);
            Ln(z, z, len);
            Muls(z, z, (float)(-1), len);
            Sqrt(z, z, len);

            Muls(tmp, z, (float)1.641345311, len);
            Adds(tmp, tmp, (float)3.429567803, len);
            Mul(tmp, tmp, z, len);
            Adds(tmp, tmp, (float)-1.624906493, len);
            Mul(tmp, tmp, z, len);
            Adds(tmp, tmp, (float)-1.970840454, len);
            Or(inputLocal.template ReinterpretCast<uint16_t>(), inputLocal.template ReinterpretCast<uint16_t>(), tmp.template ReinterpretCast<uint16_t>(), len*2);

            Muls(tmp, z, (float)1.637067800, len);
            Adds(tmp, tmp, (float)3.543889200, len);
            Mul(tmp, tmp, z, len);
            Adds(tmp, tmp, (float)1, len);

            Div(inputLocal, inputLocal, tmp, len);
            Select(outLocal, mask, outLocal, inputLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, len);
        }
        else if constexpr (std::is_same_v<T, half>)
        { 
            LocalTensor<float> tmp = tmpBuf.Get<float>();
            LocalTensor<float> y = tmp[this->tileDataLenght*1];
            LocalTensor<float> x = tmp[this->tileDataLenght*2];

            Cast(y, inputLocal, RoundMode::CAST_NONE, len);
            ShiftRight(inputLocal.template ReinterpretCast<int16_t>(), inputLocal.template ReinterpretCast<int16_t>(), (int16_t)15, len);
            ShiftLeft(inputLocal.template ReinterpretCast<int16_t>(), inputLocal.template ReinterpretCast<int16_t>(), (int16_t)15, len);
            Mul(tmp, y, y, len);
            Muls(tmp, tmp, (float)(-1), len);
            Adds(tmp, tmp, (float)1, len);
            Ln(tmp, tmp, len);
            Muls(y, tmp, (float)0.4977180748, len);
            Adds(y, y, (float)3.8499244974, len);
            Muls(tmp, tmp, (float)6.0304073099, len);
            Mul(x, y, y, len);
            Sub(x, x, tmp, len);
            Sqrt(x, x, len);
            Sub(x, x, y, len);
            Sqrt(x,x,len);
            Cast(outLocal, x, RoundMode::CAST_NONE, len);
            Or(outLocal.template ReinterpretCast<uint16_t>(), outLocal.template ReinterpretCast<uint16_t>(), inputLocal.template ReinterpretCast<uint16_t>(), len);
        }
        else if constexpr (std::is_same_v<T, bfloat16_t>)
        {
            LocalTensor<float> tmp = tmpBuf.Get<float>();
            LocalTensor<float> z = tmp[this->tileDataLenght];
            LocalTensor<float> y = tmp[this->tileDataLenght*2];
            LocalTensor<float> x = tmp[this->tileDataLenght*3];
            LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();

            Cast(y, inputLocal, RoundMode::CAST_NONE, len);

            // <0.7
            Mul(z, y, y, len);

            Muls(tmp, z, (float)-0.140543331, len);
            Adds(tmp, tmp, (float)0.914624893, len);
            Mul(tmp, tmp, z, len);
            Adds(tmp, tmp, (float)-1.645349621, len);
            Mul(tmp, tmp, z, len);
            Adds(x, tmp, (float)0.886226899, len);

            Muls(tmp, z, (float)0.012229801, len);
            Adds(tmp, tmp, (float)-0.329097515, len);
            Mul(tmp, tmp, z, len);
            Adds(tmp, tmp, (float)1.442710462, len);
            Mul(tmp, tmp, z, len);
            Adds(tmp, tmp, (float)-2.118377725, len);
            Mul(tmp, tmp, z, len);
            Adds(tmp, tmp, (float)1, len);

            Mul(x, y, x, len);
            Div(x, x, tmp, len);

            Abs(z, y, len);
            CompareScalar(mask, z, (float)1, CMPMODE::LT, len);
            uint32_t nan_bits = 0x7FC00000;
            Select(x, mask, x, *reinterpret_cast<float*>(&nan_bits), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            Adds(tmp.template ReinterpretCast<int32_t>(), mask.template ReinterpretCast<int32_t>(), 0, len/4); //别忘了改

            CompareScalar(mask, y, (float)1, CMPMODE::NE, len);
            uint32_t inf1_bits = 0x7F800000;
            Select(x, mask, x, *reinterpret_cast<float*>(&inf1_bits), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            And(tmp.template ReinterpretCast<int16_t>(), mask.template ReinterpretCast<int16_t>(), tmp.template ReinterpretCast<int16_t>(), len/2); //别忘了改

            CompareScalar(mask, y, (float)-1, CMPMODE::NE, len);
            uint32_t inf2_bits = 0xFF800000;
            Select(x, mask, x, *reinterpret_cast<float*>(&inf2_bits), SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
            And(tmp.template ReinterpretCast<int16_t>(), mask.template ReinterpretCast<int16_t>(), tmp.template ReinterpretCast<int16_t>(), len/2); //别忘了改

            CompareScalar(mask, z, (float)0.7, CMPMODE::GT, len);
            And(tmp.template ReinterpretCast<int16_t>(), mask.template ReinterpretCast<int16_t>(), tmp.template ReinterpretCast<int16_t>(), len/2); //别忘了改

            Not(mask.template ReinterpretCast<int16_t>(), tmp.template ReinterpretCast<int16_t>(), len/2);


            ///>0.7
            ShiftRight(y.template ReinterpretCast<int32_t>(), y.template ReinterpretCast<int32_t>(), 31, len);
            ShiftLeft(y.template ReinterpretCast<int32_t>(), y.template ReinterpretCast<int32_t>(), 31, len);
            Muls(z, z, (float)(-1), len);
            Adds(z, z, (float)(1), len);
            Muls(z, z, (float)(0.5), len);
            Ln(z, z, len);
            Muls(z, z, (float)(-1), len);
            Sqrt(z, z, len);

            Muls(tmp, z, (float)1.641345311, len);
            Adds(tmp, tmp, (float)3.429567803, len);
            Mul(tmp, tmp, z, len);
            Adds(tmp, tmp, (float)-1.624906493, len);
            Mul(tmp, tmp, z, len);
            Adds(tmp, tmp, (float)-1.970840454, len);
            Or(y.template ReinterpretCast<uint16_t>(), y.template ReinterpretCast<uint16_t>(), tmp.template ReinterpretCast<uint16_t>(), len*2);

            Muls(tmp, z, (float)1.637067800, len);
            Adds(tmp, tmp, (float)3.543889200, len);
            Mul(tmp, tmp, z, len);
            Adds(tmp, tmp, (float)1, len);

            Div(y, y, tmp, len);

            Select(x, mask, x, y, SELMODE::VSEL_TENSOR_TENSOR_MODE, len);
            

            Cast(outLocal, x, RoundMode::CAST_RINT, len);
        }

        inQueue.FreeTensor(inputLocal);
        outQueue.EnQue<T>(outLocal);
    }
private:
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();
        DataCopy(inputLocal, xGm[progress * this->tileDataLenght], this->processDataLenght);
        inQueue.EnQue(inputLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<T> outLocal = outQueue.DeQue<T>();
        DataCopy(yGm[progress * this->tileDataLenght], outLocal, this->processDataLenght);
        outQueue.FreeTensor(outLocal);
    }
private:
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> maskBuf;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    uint32_t loopCount;
    uint32_t tileDataLenght;
    uint32_t tailDataLenght;
    uint32_t processDataLenght;
};

extern "C" __global__ __aicore__ void erfinv(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    TPipe pipe;
    KernelErfinv<DTYPE_X> op;
        op.Init(x, y,
                tiling_data.bigNum, tiling_data.bigLength, tiling_data.smallLength, &pipe);
        op.Process();
}