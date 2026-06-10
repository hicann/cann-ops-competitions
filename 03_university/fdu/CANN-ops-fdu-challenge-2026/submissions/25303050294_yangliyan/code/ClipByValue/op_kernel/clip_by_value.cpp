// Kernel侧核函数实现
#include "kernel_operator.h"

#include <type_traits>

#include "clip_by_value_tiling.h"
#include "tiling_key_clip_by_value.h"

constexpr int32_t BUFFER_NUM = 1;

template <class DT_X>
class KernelClipByValue {
public:
    __aicore__ inline KernelClipByValue() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum, uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum, float min, float max) {
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
        uint32_t totalDataNum = finalBigTileNum;
        this->tileDataNum = tileDataNum;
        if (coreNum < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
        }
        if (globalBufferIndex >= totalDataNum) {
            this->coreDataNum = 0;
            this->tileNum = 0;
            this->tailDataNum = 0;
            return;
        }
        if (globalBufferIndex + this->coreDataNum > totalDataNum) {
            this->coreDataNum = totalDataNum - globalBufferIndex;
        }
        this->tileNum = (this->coreDataNum + this->tileDataNum - 1) / this->tileDataNum;
        this->tailDataNum = this->coreDataNum - (this->tileNum - 1) * this->tileDataNum;
        if (this->tailDataNum == 0) {
            this->tailDataNum = this->tileDataNum;
        }
        xGm.SetGlobalBuffer((__gm__ DT_X*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + globalBufferIndex, this->coreDataNum);
        if constexpr (std::is_same<DT_X, half>::value) {
            this->minVal = half(min);
            this->maxVal = half(max);
        } else if constexpr (std::is_same<DT_X, float>::value) {
            this->minVal = min;
            this->maxVal = max;
        } else {
            this->minVal = static_cast<DT_X>(min);
            this->maxVal = static_cast<DT_X>(max);
        }
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        if (this->tileNum == 0) {
            return;
        }
        int32_t loopCount = static_cast<int32_t>(this->tileNum);
        for (int32_t i = 0; i < loopCount; ++i) {
            uint32_t dataNum = (i == loopCount - 1) ? this->tailDataNum : this->tileDataNum;
            CopyIn(i, dataNum);
            Compute(dataNum);
            CopyOut(i, dataNum);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t dataNum) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        uint32_t copyBytes = dataNum * sizeof(DT_X);
        if (copyBytes % 32 == 0) {
            AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], dataNum);
        } else {
            AscendC::DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> padParams{true, 0, 0, static_cast<DT_X>(0)};
            AscendC::DataCopyPad(xLocal, xGm[progress * this->tileDataNum], copyParams, padParams);
        }
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(uint32_t dataNum) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        int32_t calCount = static_cast<int32_t>(dataNum);
        AscendC::Maxs(yLocal, xLocal, this->minVal, calCount);
        AscendC::Mins(yLocal, yLocal, this->maxVal, calCount);
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t dataNum) {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        uint32_t copyBytes = dataNum * sizeof(DT_X);
        if (copyBytes % 32 == 0) {
            AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, dataNum);
        } else {
            AscendC::DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            AscendC::DataCopyPad(yGm[progress * this->tileDataNum], yLocal, copyParams);
        }
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    DT_X minVal;
    DT_X maxVal;
};


template <typename DT_X>
__global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ClipByValueTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tiling_data, tiling);
    KernelClipByValue<DT_X> op;
    op.Init(x, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum, tiling_data.min, tiling_data.max);
    op.Process();
}
