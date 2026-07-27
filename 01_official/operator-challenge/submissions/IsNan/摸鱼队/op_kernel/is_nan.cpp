#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
using namespace AscendC;
class KernelIsNan {
public:
    __aicore__ inline KernelIsNan() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR out,
                                uint32_t bigNum, uint32_t bigLength, uint32_t smallLength) {//, TPipe* pipeIn
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreDataNum;
        if (GetBlockIdx() < bigNum) {
            coreDataNum = bigLength;
            xGm.SetGlobalBuffer((__gm__ DTYPE_X*)input + bigLength * GetBlockIdx(), bigLength);
            yGm.SetGlobalBuffer((__gm__ uint8_t*)out + bigLength * GetBlockIdx(), bigLength);
        } else {
            coreDataNum = smallLength;
            xGm.SetGlobalBuffer((__gm__ DTYPE_X*)input + bigLength * bigNum + smallLength * (GetBlockIdx() - bigNum), smallLength);
            yGm.SetGlobalBuffer((__gm__ uint8_t*)out + bigLength * bigNum + smallLength * (GetBlockIdx() - bigNum), smallLength);
        }
        if constexpr (std::is_same_v<DTYPE_X, float>)
        {
            this->tileDataLenght = 19456;
        }
        else
        {
            this->tileDataLenght = 32768;
        }

        this->loopCount = coreDataNum/this->tileDataLenght;
        this->tailDataLenght = coreDataNum - this->tileDataLenght*this->loopCount;
        if(this->tailDataLenght == 0){this->tailDataLenght = this->tileDataLenght; this->loopCount--;}
    }
    __aicore__ inline void Process() {
    LocalTensor<DTYPE_X> xLocalPing(TPosition::VECIN, 0,                                           this->tileDataLenght);
    LocalTensor<uint8_t> zLocalPing(TPosition::VECOUT, this->tileDataLenght*(sizeof(DTYPE_X)),     this->tileDataLenght);
    LocalTensor<DTYPE_X> xLocalPong(TPosition::VECIN, this->tileDataLenght*(sizeof(DTYPE_X)+1),    this->tileDataLenght);
    LocalTensor<uint8_t> zLocalPong(TPosition::VECOUT, this->tileDataLenght*(sizeof(DTYPE_X)*2+1), this->tileDataLenght);

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    this->processDataLenght = this->tileDataLenght;
    for (int i = 0; i < this->loopCount+1; i++) {
        if(i == this->loopCount)
        {
            this->processDataLenght = this->tailDataLenght;
        }
        int32_t eventID = (i % 2 == 0 ? EVENT_ID0 : EVENT_ID1);
        LocalTensor<DTYPE_X> &inputLocal = (i % 2 == 0 ? xLocalPing : xLocalPong);
        LocalTensor<uint8_t> &outLocal = (i % 2 == 0 ? zLocalPing : zLocalPong);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID);
        DataCopy(inputLocal, xGm[i * this->tileDataLenght], this->processDataLenght);

        SetFlag<HardEvent::MTE2_V>(eventID);
        WaitFlag<HardEvent::MTE2_V>(eventID);

        if constexpr (std::is_same_v<DTYPE_X, float16_t>)
        {
            Maxs(inputLocal.ReinterpretCast<int16_t>(), inputLocal.ReinterpretCast<int16_t>(), (int16_t)(0x7E00-1), this->processDataLenght);
            Adds(inputLocal.ReinterpretCast<int16_t>(), inputLocal.ReinterpretCast<int16_t>(), (int16_t)(-0x7E00+1), this->processDataLenght);
            Cast(inputLocal.ReinterpretCast<half>(), inputLocal.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, this->processDataLenght);
            Cast(outLocal, inputLocal.ReinterpretCast<half>(), RoundMode::CAST_NONE, this->processDataLenght);
        }
        else if constexpr (std::is_same_v<DTYPE_X, bfloat16_t>)
        {
            Maxs(inputLocal.ReinterpretCast<int16_t>(), inputLocal.ReinterpretCast<int16_t>(), (int16_t)(0x7FC0-1), this->processDataLenght);
            Adds(inputLocal.ReinterpretCast<int16_t>(), inputLocal.ReinterpretCast<int16_t>(), (int16_t)(-0x7FC0+1), this->processDataLenght);
            Cast(inputLocal.ReinterpretCast<half>(), inputLocal.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, this->processDataLenght);
            Cast(outLocal, inputLocal.ReinterpretCast<half>(), RoundMode::CAST_NONE, this->processDataLenght);
        }
        else
        {
            Maxs(inputLocal.ReinterpretCast<int32_t>(), inputLocal.ReinterpretCast<int32_t>(), (int32_t)(0x7FC00000-1), this->processDataLenght);
            Adds(inputLocal.ReinterpretCast<int32_t>(), inputLocal.ReinterpretCast<int32_t>(), (int32_t)(-0x7FC00000+1), this->processDataLenght);
            half scale = 1.0;
            SetDeqScale(scale);
            Cast(inputLocal.ReinterpretCast<half>(), inputLocal.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, this->processDataLenght);
            Cast(outLocal, inputLocal.ReinterpretCast<half>(), RoundMode::CAST_NONE, this->processDataLenght);
        }

        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);
        DataCopy(yGm[i * this->tileDataLenght], outLocal, this->processDataLenght);
        SetFlag<HardEvent::MTE3_MTE2>(eventID);
    }

    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

    }
private:
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<uint8_t> yGm;

    uint32_t loopCount;
    uint32_t tileDataLenght;
    uint32_t tailDataLenght;
    uint32_t processDataLenght;
};


extern "C" __global__ __aicore__ void is_nan(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelIsNan op;
        op.Init(x, y,
                tiling_data.bigNum, tiling_data.bigLength, tiling_data.smallLength);
        op.Process();
}