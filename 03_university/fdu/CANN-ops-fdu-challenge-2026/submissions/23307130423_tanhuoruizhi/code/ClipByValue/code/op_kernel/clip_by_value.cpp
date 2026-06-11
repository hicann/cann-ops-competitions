// Kernel侧核函数实现
#include "kernel_operator.h"

#include "clip_by_value_tiling.h"
#include "tiling_key_clip_by_value.h"

constexpr uint32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelClipByValue {
public:
    __aicore__ inline KernelClipByValue() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum,
                                float minValue, float maxValue) {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * blockIdx;

        this->tileDataNum = tileDataNum;
        if (blockIdx < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (blockIdx - tailBlockNum);
        }
        this->minValue = static_cast<DT_X>(minValue);
        this->maxValue = static_cast<DT_X>(maxValue);
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        this->processDataNum = this->tileDataNum;
        for (uint32_t i = 0; i < this->tileNum; ++i) {
            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t progress) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        AscendC::Maxs(yLocal, xLocal, this->minValue, this->processDataNum);
        AscendC::Mins(yLocal, yLocal, this->maxValue, this->processDataNum);
        outQueueY.EnQue<DT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress) {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    DT_X minValue;
    DT_X maxValue;
};

template <typename DT_X>
 __global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ClipByValueTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tiling_data, tiling);
    KernelClipByValue<DT_X> op;
    op.Init(x, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum,
            tiling_data.finalBigTileNum, tiling_data.finalSmallTileNum,
            tiling_data.tileDataNum, tiling_data.smallTailDataNum,
            tiling_data.bigTailDataNum, tiling_data.tailBlockNum,
            tiling_data.minValue, tiling_data.maxValue);
    op.Process();
}