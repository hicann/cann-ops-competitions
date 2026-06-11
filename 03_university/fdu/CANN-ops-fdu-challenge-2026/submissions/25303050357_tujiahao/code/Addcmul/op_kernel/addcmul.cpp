//kernel侧实现
#include "kernel_operator.h"
#include "addcmul_tiling.h"
#include "tiling_key_addcmul.h"
#include <type_traits>

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE>
class KernelAddcmul {
public:
    __aicore__ inline KernelAddcmul() {}
    __aicore__ inline void Init(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
                                uint32_t smallCoreDataNum, uint32_t bigCoreDataNum,
                                uint32_t finalBigTileNum, uint32_t finalSmallTileNum,
                                uint32_t tileDataNum, uint32_t smallTailDataNum,
                                uint32_t bigTailDataNum, uint32_t tailBlockNum,
                                uint32_t needBroadcast_, uint32_t totalLength_,
                                uint32_t input_data_length, uint32_t x1_length, uint32_t x2_length) {
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * coreNum;
        this->tileDataNum = tileDataNum;
        this->needBroadcast = needBroadcast_;
        this->totalLength = totalLength_;

        if (coreNum < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreNum - tailBlockNum);
        }

        if (needBroadcast_) {
            this->outputStartIdx = globalBufferIndex;
            this->dataLength = input_data_length;
            this->x1Length = x1_length;
            this->x2Length = x2_length;

            dataGm.SetGlobalBuffer((__gm__ TYPE*)input_data, input_data_length);
            x1Gm.SetGlobalBuffer((__gm__ TYPE*)x1, x1_length);
            x2Gm.SetGlobalBuffer((__gm__ TYPE*)x2, x2_length);
            yGm.SetGlobalBuffer((__gm__ TYPE*)y, totalLength_);
        } else {
            dataGm.SetGlobalBuffer((__gm__ TYPE*)input_data + globalBufferIndex, this->coreDataNum);
            x1Gm.SetGlobalBuffer((__gm__ TYPE*)x1 + globalBufferIndex, this->coreDataNum);
            x2Gm.SetGlobalBuffer((__gm__ TYPE*)x2 + globalBufferIndex, this->coreDataNum);
            yGm.SetGlobalBuffer((__gm__ TYPE*)y + globalBufferIndex, this->coreDataNum);
        }

        AscendC::GlobalTensor<TYPE> valueGm;
        valueGm.SetGlobalBuffer((__gm__ TYPE*)value, 1);
        this->scalarValue = valueGm.GetValue(0);

        pipe.InitBuffer(inQueueData, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));

        if constexpr (std::is_same_v<TYPE, int8_t>) {
            pipe.InitBuffer(calcBuf1, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(calcBuf2, this->tileDataNum * sizeof(half));
        }
    }

    __aicore__ inline void Process() {
        if (this->coreDataNum == 0) return;
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;

        for (int32_t i = 0; i < loopCount; i++) {
            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
            }
            uint32_t absEnd = this->outputStartIdx + i * this->tileDataNum + this->processDataNum;
            if (absEnd > this->totalLength) {
                if (this->outputStartIdx + i * this->tileDataNum >= this->totalLength) {
                    this->processDataNum = 0;
                } else {
                    this->processDataNum = this->totalLength - (this->outputStartIdx + i * this->tileDataNum);
                }
            }
            if (this->processDataNum == 0) return;
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        AscendC::LocalTensor<TYPE> dataLocal = inQueueData.AllocTensor<TYPE>();
        AscendC::LocalTensor<TYPE> x1Local = inQueueX1.AllocTensor<TYPE>();
        AscendC::LocalTensor<TYPE> x2Local = inQueueX2.AllocTensor<TYPE>();

        if (this->needBroadcast) {
            uint32_t absPos = this->outputStartIdx + progress * this->tileDataNum;
            AscendC::DataCopy(dataLocal, dataGm[absPos % this->dataLength], this->processDataNum);
            AscendC::DataCopy(x1Local, x1Gm[absPos % this->x1Length], this->processDataNum);
            AscendC::DataCopy(x2Local, x2Gm[absPos % this->x2Length], this->processDataNum);
        } else {
            AscendC::DataCopy(dataLocal, dataGm[progress * this->tileDataNum], this->processDataNum);
            AscendC::DataCopy(x1Local, x1Gm[progress * this->tileDataNum], this->processDataNum);
            AscendC::DataCopy(x2Local, x2Gm[progress * this->tileDataNum], this->processDataNum);
        }

        inQueueData.EnQue(dataLocal);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void Compute(int32_t progress) {
        AscendC::LocalTensor<TYPE> dataLocal = inQueueData.DeQue<TYPE>();
        AscendC::LocalTensor<TYPE> x1Local = inQueueX1.DeQue<TYPE>();
        AscendC::LocalTensor<TYPE> x2Local = inQueueX2.DeQue<TYPE>();
        AscendC::LocalTensor<TYPE> yLocal = outQueueY.AllocTensor<TYPE>();

        if constexpr (std::is_same_v<TYPE, int8_t>) {
            AscendC::LocalTensor<half> cb1 = calcBuf1.Get<half>(this->processDataNum * sizeof(half));
            AscendC::LocalTensor<half> cb2 = calcBuf2.Get<half>(this->processDataNum * sizeof(half));
            half scalarHalf = static_cast<half>(this->scalarValue);

            AscendC::Cast(cb1, x1Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(cb2, x2Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Mul(cb1, cb1, cb2, this->processDataNum);
            AscendC::Muls(cb1, cb1, scalarHalf, this->processDataNum);
            AscendC::Cast(cb2, dataLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Add(cb1, cb1, cb2, this->processDataNum);

            AscendC::Cast(cb2.ReinterpretCast<int16_t>(), cb1, AscendC::RoundMode::CAST_RINT, this->processDataNum);
            AscendC::ShiftLeft(cb2.ReinterpretCast<int16_t>(), cb2.ReinterpretCast<int16_t>(), int16_t(8), this->processDataNum);
            AscendC::ShiftRight(cb2.ReinterpretCast<int16_t>(), cb2.ReinterpretCast<int16_t>(), int16_t(8), this->processDataNum);
            AscendC::Cast(cb1, cb2.ReinterpretCast<int16_t>(), AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(yLocal, cb1, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        } else {
            AscendC::Mul(x1Local, x1Local, this->scalarValue, this->processDataNum);
            AscendC::Muls(x1Local, x1Local, x2Local, this->processDataNum);
            AscendC::Add(yLocal, dataLocal, x1Local, this->processDataNum);
        }

        outQueueY.EnQue<TYPE>(yLocal);

        inQueueData.FreeTensor(dataLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        AscendC::LocalTensor<TYPE> yLocal = outQueueY.DeQue<TYPE>();
        if (this->needBroadcast) {
            AscendC::DataCopy(yGm[this->outputStartIdx + progress * this->tileDataNum], yLocal, this->processDataNum);
        } else {
            AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        }
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueData, inQueueX1, inQueueX2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    AscendC::GlobalTensor<TYPE> dataGm, x1Gm, x2Gm, yGm;
    TYPE scalarValue;

    AscendC::TBuf<> calcBuf1;
    AscendC::TBuf<> calcBuf2;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;

    uint32_t needBroadcast;
    uint32_t outputStartIdx;
    uint32_t totalLength;
    uint32_t dataLength;
    uint32_t x1Length;
    uint32_t x2Length;
};

template <typename DT_INPUT_DATA>
__global__ __aicore__ void addcmul(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(AddcmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddcmulTilingData, tilingData, tiling);
    KernelAddcmul<DT_INPUT_DATA> op;
    op.Init(input_data, x1, x2, value, y,
            tilingData.smallCoreDataNum, tilingData.bigCoreDataNum,
            tilingData.finalBigTileNum, tilingData.finalSmallTileNum,
            tilingData.tileDataNum, tilingData.smallTailDataNum,
            tilingData.bigTailDataNum, tilingData.tailBlockNum,
            tilingData.needBroadcast, tilingData.totalLength,
            tilingData.input_data_length, tilingData.x1_length, tilingData.x2_length);
    op.Process();
}
