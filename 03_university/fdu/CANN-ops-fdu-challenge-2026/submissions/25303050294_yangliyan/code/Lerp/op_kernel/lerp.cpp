// Kernel侧核函数实现
#include "kernel_operator.h"

#include "lerp_tiling.h"
#include "tiling_key_lerp.h"

template <class DT_START>
class KernelLerp {
public:
    __aicore__ inline KernelLerp() {}
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum, float weight)
    {
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
        this->tileDataNum = tileDataNum;
        if (coreNum < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
        }
        sGm.SetGlobalBuffer((__gm__ DT_START*)start + globalBufferIndex, this->coreDataNum);
        eGm.SetGlobalBuffer((__gm__ DT_START*)end + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_START*)y + globalBufferIndex, this->coreDataNum);

        this->weightVal = static_cast<DT_START>(weight);
        uint32_t tileUbSize = tileDataNum * sizeof(DT_START);
        pipe.InitBuffer(inQueueStart, 1, tileUbSize);
        pipe.InitBuffer(inQueueEnd, 1, tileUbSize);
        pipe.InitBuffer(outQueueY, 1, tileUbSize);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->tileNum; i++) {
            this->processDataNum = (i == this->tileNum - 1) ? this->tailDataNum : this->tileDataNum;
            uint32_t offset = i * this->tileDataNum;

            AscendC::LocalTensor<DT_START> startLocal = inQueueStart.AllocTensor<DT_START>();
            AscendC::LocalTensor<DT_START> endLocal = inQueueEnd.AllocTensor<DT_START>();
            AscendC::DataCopy(startLocal, sGm[offset], this->processDataNum);
            AscendC::DataCopy(endLocal, eGm[offset], this->processDataNum);
            inQueueStart.EnQue(startLocal);
            inQueueEnd.EnQue(endLocal);

            startLocal = inQueueStart.DeQue<DT_START>();
            endLocal = inQueueEnd.DeQue<DT_START>();
            AscendC::LocalTensor<DT_START> yLocal = outQueueY.AllocTensor<DT_START>();
            AscendC::Sub(yLocal, endLocal, startLocal, this->processDataNum);
            AscendC::Muls(yLocal, yLocal, this->weightVal, this->processDataNum);
            AscendC::Add(yLocal, startLocal, yLocal, this->processDataNum);
            outQueueY.EnQue<DT_START>(yLocal);
            inQueueStart.FreeTensor(startLocal);
            inQueueEnd.FreeTensor(endLocal);

            yLocal = outQueueY.DeQue<DT_START>();
            AscendC::DataCopy(yGm[offset], yLocal, this->processDataNum);
            outQueueY.FreeTensor(yLocal);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueStart, inQueueEnd;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueY;
    AscendC::GlobalTensor<DT_START> sGm;
    AscendC::GlobalTensor<DT_START> eGm;
    AscendC::GlobalTensor<DT_START> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    DT_START weightVal;
};

template <typename DT_START>
__global__ __aicore__ void lerp(GM_ADDR start, GM_ADDR end, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LerpTilingData);
    GET_TILING_DATA_WITH_STRUCT(LerpTilingData, tiling_data, tiling);
    KernelLerp<DT_START> op;
    op.Init(start, end, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum, tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum, tiling_data.weight);
    op.Process();
}
