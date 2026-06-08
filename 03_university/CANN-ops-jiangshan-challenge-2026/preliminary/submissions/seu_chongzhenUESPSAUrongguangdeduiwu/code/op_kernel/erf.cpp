#include "kernel_operator.h"
#include "erf_tiling.h"

constexpr int8_t BUFFER_NUM = 2;

constexpr float SMALL_CLAMP_MAX = 2.25f;
constexpr float SMALL_CLAMP_MIN = -2.25f;

constexpr float SP0 = 1.12419718735f;
constexpr float SP1 = -0.357146096964f;
constexpr float SP2 = 0.0879196786385f;
constexpr float SP3 = -0.0123575870297f;
constexpr float SP4 = 0.000727834008646f;

template <class DT_X>
class KernelErf {
public:
     __aicore__ inline KernelErf() {}
    __aicore__ inline void Init(AscendC::TPipe *pipePtr, GM_ADDR x, GM_ADDR y,
        uint32_t smallCoreDataNum, uint32_t bigCoreDataNum,
        uint32_t finalBigTileNum, uint32_t finalSmallTileNum,
        uint32_t tileDataNum,
        uint32_t smallTailDataNum, uint32_t bigTailDataNum,
        uint32_t tailBlockNum)
    {
        this->pipe = pipePtr;
        uint8_t coreNum = AscendC::GetBlockIdx();
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
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum)
                                * (AscendC::GetBlockIdx() - tailBlockNum);
        }

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + globalBufferIndex, this->coreDataNum);
       xGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
       yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);

       pipe->InitBuffer(inQueueX_max, BUFFER_NUM, this->tileDataNum * sizeof(DT_X));
       pipe->InitBuffer(outQueueY_max, BUFFER_NUM, this->tileDataNum * sizeof(DT_X));
        pipe->InitBuffer(tmpBuf, this->tileDataNum * sizeof(float));
        pipe->InitBuffer(tmpBuf_1, this->tileDataNum * sizeof(float));
        pipe->InitBuffer(tmpBuf_2, this->tileDataNum * sizeof(float));
        pipe->InitBuffer(tmpBuf_3, this->tileDataNum * sizeof(float));
        eventIdMte2ToV = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V);
        eventIdVToMte3 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3);

    }

    __aicore__ inline void Process1()
    {
        uint32_t loopCount = this->tileNum;
        if (loopCount == 1) {
            uint32_t n = this->tailDataNum;
            CopyIn1(0, n);
            Compute1(n);
            CopyOut1(0, n);
            return;
        }
        uint32_t curN = this->tileDataNum;
        CopyIn1(0, curN);
        for (uint32_t i = 0; i < loopCount - 1; ++i) {
            uint32_t nextN = (i + 1 == loopCount - 1) ? this->tailDataNum : this->tileDataNum;
            CopyIn1(i + 1, nextN);
            Compute1(curN);
            CopyOut1(i, curN);
            curN = nextN;
        }
        Compute1(curN);
        CopyOut1(loopCount - 1, curN);
    }


private:
    __aicore__ inline void CopyIn1(uint32_t progress, uint32_t n)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX_max.AllocTensor<float>();
        AscendC::DataCopy(xLocal,xGm[progress * this->tileDataNum],n);
        inQueueX_max.EnQue(xLocal);
      
    }

    __aicore__ inline void Compute1(uint32_t n)
    {
        auto xLocal = inQueueX_max.DeQue<float>();
        auto yLocal = outQueueY_max.AllocTensor<float>();

        auto pBuf = tmpBuf.Get<float>();  // P(t)


        Mins(xLocal, xLocal, SMALL_CLAMP_MAX, n);
        Maxs(xLocal, xLocal, SMALL_CLAMP_MIN, n);
        Mul(pBuf , xLocal, xLocal, n);
        
        Muls(yLocal, pBuf , SP4, n);
        Adds(yLocal, yLocal, SP3, n);
        Mul(yLocal, yLocal, pBuf , n);
        Adds(yLocal, yLocal, SP2, n);
        Mul(yLocal, yLocal, pBuf , n);
        Adds(yLocal, yLocal, SP1, n);
        Mul(yLocal, yLocal, pBuf , n);
        Adds(yLocal, yLocal, SP0, n);
        Mul(yLocal, xLocal, yLocal, n); 

        outQueueY_max.EnQue(yLocal);
        inQueueX_max.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut1(uint32_t progress, uint32_t n)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY_max.DeQue<float>();
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, n);
        outQueueY_max.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe *pipe;  // 指针
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQueueX_max;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outQueueY_max;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_1,tmpBuf_2,tmpBuf_3,tmpBuf;
    AscendC::TEventID eventIdMte2ToV;
    AscendC::TEventID eventIdVToMte3;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

//*****************
//以下是为小数据准备的部分
//******************
template <class DT_X>
class KernelErf_1 {
public:
     __aicore__ inline KernelErf_1() {}
    __aicore__ inline void Init(AscendC::TPipe *pipePtr, GM_ADDR x, GM_ADDR y,
        uint32_t smallCoreDataNum, uint32_t bigCoreDataNum,
        uint32_t finalBigTileNum, uint32_t finalSmallTileNum,
        uint32_t tileDataNum,
        uint32_t smallTailDataNum, uint32_t bigTailDataNum,
        uint32_t tailBlockNum)
    {
        this->pipe = pipePtr;
        uint8_t coreNum = AscendC::GetBlockIdx();
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
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum)
                                * (AscendC::GetBlockIdx() - tailBlockNum);
        }

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + globalBufferIndex, this->coreDataNum);
       xGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
       yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);

        pipe->InitBuffer(tmpBuf_1, this->tileDataNum * sizeof(float));
        pipe->InitBuffer(tmpBuf_2, this->tileDataNum * sizeof(float));
        pipe->InitBuffer(tmpBuf_3, this->tileDataNum * sizeof(float));
        eventIdMte2ToV = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V);
        eventIdVToMte3 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3);

    }

     __aicore__ inline void Process2()//省去入队出队操作
    {    
         uint32_t loopCount = this->tileNum;
         if(loopCount <= 1){
             uint32_t n = this->tailDataNum;
             auto pBuf_1 = tmpBuf_1.Get<float>();  // P(t)
             auto pBuf_2 = tmpBuf_2.Get<float>();
             auto pBuf_3 = tmpBuf_3.Get<float>();
             AscendC::DataCopy(pBuf_1 , xGm[0], n);
             AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
             AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            
             Mins(pBuf_1 , pBuf_1 , SMALL_CLAMP_MAX, n);
             Maxs(pBuf_1 , pBuf_1 , SMALL_CLAMP_MIN, n);
             Mul(pBuf_2 , pBuf_1 , pBuf_1 , n);
        
             Muls(pBuf_3 , pBuf_2 , SP4, n);
             Adds(pBuf_3 , pBuf_3 , SP3, n);
             Mul(pBuf_3 , pBuf_3 , pBuf_2 , n);
             Adds(pBuf_3 , pBuf_3 , SP2, n);
             Mul(pBuf_3 , pBuf_3 , pBuf_2 , n);
             Adds(pBuf_3 , pBuf_3 , SP1, n);
             Mul(pBuf_3 , pBuf_3 , pBuf_2 , n);
             Adds(pBuf_3 , pBuf_3 , SP0, n);
             Mul(pBuf_3 , pBuf_1 , pBuf_3 , n); 
             AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
             AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
             AscendC::DataCopy(yGm[0], pBuf_3 , n);
         }

    }
private:
    AscendC::TPipe *pipe;  // 指针
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_1,tmpBuf_2,tmpBuf_3;
    AscendC::TEventID eventIdMte2ToV;
    AscendC::TEventID eventIdVToMte3;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};
//*******************

extern "C" __global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    uint32_t mode = tiling_data.mode;
    KernelErf<float> op;
    KernelErf_1<float> op_1;
     if(mode == 2){
     op.Init(&pipe, x, y, tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum);
     op.Process1();
     }
     else{
     op_1.Init(&pipe, x, y, tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum);
     op_1.Process2();
     } 

}
