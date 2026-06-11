// Kernel侧核函数实现
#include "kernel_operator.h"

#include "lerp_tiling.h"
#include "tiling_key_lerp.h"



constexpr int32_t BUFFER_NUM = 2;


template <class DT_START>
class KernelLerp {
public:
    __aicore__ inline KernelLerp() {}

    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR y, 
                                uint32_t smallCoreDataNum, uint32_t bigCoreDataNum, 
                                uint32_t finalBigTileNum, uint32_t finalSmallTileNum, 
                                uint32_t tileDataNum, uint32_t smallTailDataNum, 
                                uint32_t bigTailDataNum, uint32_t tailBlockNum,
                                float weight, float oneMinusW) 
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

        this->weight = weight;
        this->oneMinusW = oneMinusW;

        startGm.SetGlobalBuffer((__gm__ DT_START *)start + globalBufferIndex, this->coreDataNum);
        endGm.SetGlobalBuffer((__gm__ DT_START *)end + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_START *)y + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueStart, BUFFER_NUM, this->tileDataNum * sizeof(DT_START));
        pipe.InitBuffer(inQueueEnd, BUFFER_NUM, this->tileDataNum * sizeof(DT_START));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DT_START));

        // 仅 float16 时分配TBuf
        if constexpr (std::is_same_v<DT_START, half>) {
            pipe.InitBuffer(startBuf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(endBuf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(yBuf, this->tileDataNum * sizeof(float));
        }
    }

    __aicore__ inline void ProcessFloat32() 
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            ComputeFloat32(i);
            CopyOut(i);
        }
    }

    __aicore__ inline void ProcessFloat16() 
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            ComputeFloat16(i);
            CopyOut(i);
        }
    }

private:

    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DT_START> startLocal = inQueueStart.AllocTensor<DT_START>();
        AscendC::DataCopy(startLocal, startGm[progress * this->tileDataNum], this->processDataNum);
        inQueueStart.EnQue(startLocal);
        AscendC::LocalTensor<DT_START> endLocal = inQueueEnd.AllocTensor<DT_START>();
        AscendC::DataCopy(endLocal, endGm[progress * this->tileDataNum], this->processDataNum);
        inQueueEnd.EnQue(endLocal);
    }

    __aicore__ inline void ComputeFloat32(int32_t progress)
    {
        AscendC::LocalTensor<DT_START> startLocal = inQueueStart.DeQue<DT_START>();
        AscendC::LocalTensor<DT_START> endLocal = inQueueEnd.DeQue<DT_START>();
        AscendC::LocalTensor<DT_START> yLocal = outQueueY.AllocTensor<DT_START>();

        AscendC::Muls(startLocal, startLocal, this->oneMinusW, this->processDataNum);
        AscendC::Muls(endLocal, endLocal, this->weight, this->processDataNum);
        AscendC::Add(yLocal, startLocal, endLocal, this->processDataNum);
        
        outQueueY.EnQue<DT_START>(yLocal);
                
        inQueueStart.FreeTensor(startLocal);
        inQueueEnd.FreeTensor(endLocal);
    }

    __aicore__ inline void ComputeFloat16(int32_t progress)
    {
        AscendC::LocalTensor<DT_START> startLocal = inQueueStart.DeQue<DT_START>();
        AscendC::LocalTensor<DT_START> endLocal = inQueueEnd.DeQue<DT_START>();
        AscendC::LocalTensor<DT_START> yLocal = outQueueY.AllocTensor<DT_START>();
        
        auto tmpStart = startBuf.Get<float>();
        auto tmpEnd = endBuf.Get<float>();
        auto tmpY = yBuf.Get<float>();

        AscendC::Cast(tmpStart, startLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(tmpEnd, endLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);

        AscendC::Muls(tmpStart, tmpStart, this->oneMinusW, this->processDataNum);
        AscendC::Muls(tmpEnd, tmpEnd, this->weight, this->processDataNum);
        AscendC::Add(tmpY, tmpStart, tmpEnd, this->processDataNum);

        AscendC::Cast(yLocal, tmpY, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        
        outQueueY.EnQue<DT_START>(yLocal);
                
        inQueueStart.FreeTensor(startLocal);
        inQueueEnd.FreeTensor(endLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DT_START> yLocal = outQueueY.DeQue<DT_START>();  
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }


private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueStart;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueEnd;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DT_START> startGm;
    AscendC::GlobalTensor<DT_START> endGm;
    AscendC::GlobalTensor<DT_START> yGm;
    // 新增三个TBuf成员变量
    AscendC::TBuf<AscendC::TPosition::VECCALC> startBuf, endBuf, yBuf;    
    
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    float weight;
    float oneMinusW;
};

template <typename DT_START>
 __global__ __aicore__ void lerp(GM_ADDR start, GM_ADDR end, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LerpTilingData);
    GET_TILING_DATA_WITH_STRUCT(LerpTilingData, tiling_data, tiling);
    
    KernelLerp<DT_START> op;
    
    op.Init(start, end, y, 
            tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, 
            tiling_data.finalBigTileNum, tiling_data.finalSmallTileNum, 
            tiling_data.tileDataNum, tiling_data.smallTailDataNum, 
            tiling_data.bigTailDataNum, tiling_data.tailBlockNum,
            tiling_data.weight, tiling_data.oneMinusW);
    
    if constexpr (std::is_same_v<DT_START, float>) {
        op.ProcessFloat32();    // 无Cast，无TBuf
    } else {
        op.ProcessFloat16();    // half → float → half
    }
}



// extern "C" __global__ __aicore__ void lerp(GM_ADDR start, GM_ADDR end, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
//     GET_TILING_DATA(tiling_data, tiling);
//     // TODO: user kernel impl
// }