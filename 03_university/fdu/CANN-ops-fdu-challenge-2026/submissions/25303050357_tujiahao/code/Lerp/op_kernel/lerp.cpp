// Kernel侧核函数实现
#include "kernel_operator.h"

#include "lerp_tiling.h"
#include "tiling_key_lerp.h"
constexpr int32_t BUFFER_NUM = 2;

template <class DT_START>
class KernelLerp {
public:
    __aicore__ inline KernelLerp() {}
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR y, const LerpTilingData& tilingData) {
        uint32_t smallCoreDataNum = tilingData.smallCoreDataNum;
        uint32_t bigCoreDataNum = tilingData.bigCoreDataNum;
        uint32_t finalBigTileNum = tilingData.finalBigTileNum;
        uint32_t finalSmallTileNum = tilingData.finalSmallTileNum;
        uint32_t tileDataNum = tilingData.tileDataNum;
        uint32_t smallTailDataNum = tilingData.smallTailDataNum;
        uint32_t bigTailDataNum = tilingData.bigTailDataNum;
        uint32_t tailBlockNum = tilingData.tailBlockNum;

        this->weight = static_cast<DT_START>(tilingData.weight);

        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * coreNum;
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

        pipe.InitBuffer(inQueueS, BUFFER_NUM, this->tileDataNum * sizeof(DT_START));
        pipe.InitBuffer(inQueueE, BUFFER_NUM, this->tileDataNum * sizeof(DT_START));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DT_START));
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }
private:
    __aicore__ inline void CopyIn(int32_t progress){
        AscendC::LocalTensor<DT_START> sLocal = inQueueS.AllocTensor<DT_START>();
        AscendC::LocalTensor<DT_START> eLocal = inQueueE.AllocTensor<DT_START>();
        AscendC::DataCopy(sLocal, sGm[progress * this->tileDataNum], this->processDataNum);
        AscendC::DataCopy(eLocal, eGm[progress * this->tileDataNum], this->processDataNum);

        inQueueS.EnQue(sLocal);
        inQueueE.EnQue(eLocal);
    }

    __aicore__ inline void Compute(int32_t progress){
        AscendC::LocalTensor<DT_START> sLocal = inQueueS.DeQue<DT_START>();
        AscendC::LocalTensor<DT_START> eLocal = inQueueE.DeQue<DT_START>();
        AscendC::LocalTensor<DT_START> yLocal = outQueueY.AllocTensor<DT_START>();

        AscendC::Sub(eLocal, eLocal, sLocal, this->processDataNum);
        AscendC::Muls(eLocal, eLocal, this->weight, this->processDataNum);
        AscendC::Add(yLocal, sLocal, eLocal, this->processDataNum);

        outQueueY.EnQue(yLocal);
        inQueueS.FreeTensor(sLocal);
        inQueueE.FreeTensor(eLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress){
        AscendC::LocalTensor<DT_START> yLocal = outQueueY.DeQue<DT_START>();
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueS;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueE;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DT_START> sGm;
    AscendC::GlobalTensor<DT_START> eGm;
    AscendC::GlobalTensor<DT_START> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    DT_START weight;
};

template <typename DT_START>
 __global__ __aicore__ void lerp(GM_ADDR start, GM_ADDR end, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LerpTilingData);
    GET_TILING_DATA_WITH_STRUCT(LerpTilingData, tiling_data, tiling);
    KernelLerp<DT_START> op;
    op.Init(start, end, y, tiling_data);
    op.Process();
}

