// Kernel侧核函数实现
#include "kernel_operator.h"

#include "clip_by_value_tiling.h"
#include "tiling_key_clip_by_value.h"

constexpr int32_t BUFFER_NUM = 1;

template <class DT_X>
class KernelClipByValue {

public:
    __aicore__ inline KernelClipByValue() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, 
                                uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, 
                                uint32_t finalBigTileNum, 
                                uint32_t finalSmallTileNum, 
                                uint32_t tileDataNum, 
                                uint32_t smallTailDataNum, 
                                uint32_t bigTailDataNum, 
                                uint32_t tailBlockNum,
                                float min, float max) 
    {
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
        this->tileDataNum = tileDataNum;
        if (coreIdx < tailBlockNum) { 
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        }
        else { 
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
        }

        this->min = min;
        this->max = max;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DT_X));

        if constexpr (std::is_same_v<DT_X, int32_t) {
            pipe.InitBuffer(tmpBuf, this->tileDataNum * sizeof(float));
        }
    }
    
    __aicore__ inline void Process() 
    {
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

    __aicore__ inline void CopyIn(int32_t progress)
    {
      AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
      AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
      inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();

        if constexpr (std::is_same_v<DT_X, half>) {
            AscendC::Maxs(yLocal, xLocal, (half)this->min, this->processDataNum);
            AscendC::Mins(yLocal, yLocal, (half)this->max, this->processDataNum);
        } else if constexpr(std::is_same_v<DT_X, int32_t>) {
            auto tmp = tmpBuf.Get<float>();
            // int32 -> float
            AscendC::Cast(tmp, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            // float 上做 clamp
            AscendC::Maxs(tmp, tmp, this->min, this->processDataNum);
            AscendC::Mins(tmp, tmp, this->max, this->processDataNum);
            // float → int32
            AscendC::Cast(yLocal, tmp, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        } else {
            AscendC::Maxs(yLocal, xLocal, this->min, this->processDataNum);
            AscendC::Mins(yLocal, yLocal, this->max, this->processDataNum);
        }
        
        outQueueY.EnQue<DT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();  
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }


private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf; 
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    float min;
    float max;
};

template <typename DT_X>
 __global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ClipByValueTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tiling_data, tiling);

    KernelClipByValue<DT_X> op;

    op.Init(x, y, 
            tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, 
            tiling_data.finalBigTileNum, tiling_data.finalSmallTileNum, 
            tiling_data.tileDataNum, tiling_data.smallTailDataNum, 
            tiling_data.bigTailDataNum, tiling_data.tailBlockNum,
            tiling_data.min, tiling_data.max);
    
    op.Process();
}



