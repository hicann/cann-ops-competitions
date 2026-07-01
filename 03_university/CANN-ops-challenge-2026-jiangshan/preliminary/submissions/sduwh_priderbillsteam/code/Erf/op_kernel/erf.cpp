// Kernel侧核函数实现
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"
constexpr int32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelErfDoubleBuffer {
public:
    __aicore__ inline KernelErfDoubleBuffer() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, AscendC::TPipe* pipeIn,
                                uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum)
    {
        pipe=pipeIn;
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = 0;
        this->tileDataNum = tileDataNum;
        if (coreNum < tailBlockNum) {
          this->coreDataNum = bigCoreDataNum;
          this->tileNum = finalBigTileNum;
          this->tailDataNum = bigTailDataNum;
globalBufferIndex = bigCoreDataNum * coreNum;
        }
        else {
          this->coreDataNum = smallCoreDataNum;
          this->tileNum = finalSmallTileNum;
          this->tailDataNum = smallTailDataNum;
          globalBufferIndex = bigCoreDataNum * tailBlockNum + smallCoreDataNum * (coreNum - tailBlockNum);
        }
uint32_t byteslength = this->tileDataNum * sizeof(float);
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + globalBufferIndex, this->coreDataNum);

        pipe->InitBuffer(inQueueX, BUFFER_NUM, byteslength);
        pipe->InitBuffer(outQueueY, BUFFER_NUM, byteslength);
        pipe->InitBuffer(tmp1, byteslength);
        pipe->InitBuffer(tmp2, byteslength);

    }
    __aicore__ inline void Process()
    {
        uint32_t tileOffset = 0;
        for (int32_t i = 0; i < this->tileNum-1; i++) {
            CopyIn(tileOffset,this->tileDataNum);
            Compute(this->tileDataNum);
            CopyOut(tileOffset,this->tileDataNum);
                tileOffset += this->tileDataNum;
        }
            CopyIn( tileOffset,this->tailDataNum);
            Compute( this->tailDataNum);
            CopyOut( tileOffset,this->tailDataNum);
    }

private:
    __aicore__ inline void CopyIn(uint32_t & tileOffset, uint32_t & length)
    {
      AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
      AscendC::DataCopy(xLocal, xGm[tileOffset], length);
      inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(uint32_t & length)
    {
      AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
      AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
      AscendC::LocalTensor<DTYPE_X> tmpp1 = tmp1.Get<DTYPE_X>();
      AscendC::LocalTensor<DTYPE_X> tmpp2 = tmp2.Get<DTYPE_X>();

        AscendC::Maxs(yLocal, xLocal, static_cast<float>(-2.327), length);
        AscendC::Mins(xLocal, yLocal, static_cast<float>(2.327), length);
AscendC::Mul(tmpp1,xLocal,xLocal,length);

AscendC::Mul(tmpp2,tmpp1,tmpp1,length);
AscendC::Adds(tmpp2,tmpp2,static_cast<float>(300.8241),length);
AscendC::Mul(tmpp2,tmpp2,xLocal,length);

AscendC::Muls(tmpp1,tmpp1,static_cast<float>(92.86162),length);
AscendC::Adds(tmpp1,tmpp1,static_cast<float>(265.6042),length);

AscendC::Div(yLocal,tmpp2,tmpp1,length);
      outQueueY.EnQue<DTYPE_Y>(yLocal);
      inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(uint32_t & tileOffset, uint32_t & length)
    {
      AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
      AscendC::DataCopy(yGm[tileOffset], yLocal, length);
      outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe *pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp1,tmp2;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
};


constexpr int32_t BUFFER_NUM1 = 1;

template <class DT_X>
class KernelErfSingleTile {
public:
    __aicore__ inline KernelErfSingleTile() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, AscendC::TPipe* pipeIn,
                                uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum)
    {
        pipe=pipeIn;
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = 0;
        this->tileDataNum = tileDataNum;
        if (coreNum < tailBlockNum) {
          this->coreDataNum = bigCoreDataNum;
          this->tileNum = finalBigTileNum;
          this->tailDataNum = bigTailDataNum;
globalBufferIndex = bigCoreDataNum * coreNum;
        }
        else {
          this->coreDataNum = smallCoreDataNum;
          this->tileNum = finalSmallTileNum;
          this->tailDataNum = smallTailDataNum;
          globalBufferIndex = bigCoreDataNum * tailBlockNum + smallCoreDataNum * (coreNum - tailBlockNum);
        }
uint32_t byteslength = this->tileDataNum * sizeof(float);
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + globalBufferIndex, this->coreDataNum);

        pipe->InitBuffer(inQueueX, BUFFER_NUM1, byteslength);
        pipe->InitBuffer(outQueueY, BUFFER_NUM1, byteslength);
        pipe->InitBuffer(tmp1, byteslength);
        pipe->InitBuffer(tmp2, byteslength);

    }
    __aicore__ inline void Process()
    {
        uint32_t tileOffset = 0;
            CopyIn( tileOffset,this->tailDataNum);
            Compute( this->tailDataNum);
            CopyOut( tileOffset,this->tailDataNum);
    }

private:
    __aicore__ inline void CopyIn(uint32_t & tileOffset, uint32_t & length)
    {
      AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
      AscendC::DataCopy(xLocal, xGm[tileOffset], length);
      inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(uint32_t & length)
    {
      AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
      AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
      AscendC::LocalTensor<DTYPE_X> tmpp1 = tmp1.Get<DTYPE_X>();
      AscendC::LocalTensor<DTYPE_X> tmpp2 = tmp2.Get<DTYPE_X>();

        AscendC::Maxs(yLocal, xLocal, static_cast<float>(-2.327), length);
        AscendC::Mins(xLocal, yLocal, static_cast<float>(2.327), length);
AscendC::Mul(tmpp1,xLocal,xLocal,length);

AscendC::Mul(tmpp2,tmpp1,tmpp1,length);
AscendC::Adds(tmpp2,tmpp2,static_cast<float>(300.8241),length);
AscendC::Mul(tmpp2,tmpp2,xLocal,length);

AscendC::Muls(tmpp1,tmpp1,static_cast<float>(92.86162),length);
AscendC::Adds(tmpp1,tmpp1,static_cast<float>(265.6042),length);

AscendC::Div(yLocal,tmpp2,tmpp1,length);
      outQueueY.EnQue<DTYPE_Y>(yLocal);
      inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(uint32_t & tileOffset, uint32_t & length)
    {
      AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
      AscendC::DataCopy(yGm[tileOffset], yLocal, length);
      outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe *pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp1,tmp2;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
};


template <typename DT_X, bool IS_SINGLE_TILE>
 __global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
AscendC::TPipe pipe;
 	     if constexpr (IS_SINGLE_TILE) {
 	         KernelErfSingleTile<DT_X> op;
    op.Init(x, y, &pipe, tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum);
    op.Process();
 	     } else {
 	         KernelErfDoubleBuffer<DT_X> op;
    op.Init(x, y, &pipe, tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum);
    op.Process();
 	     }

}