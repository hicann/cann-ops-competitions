#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
// tensor num for each queue
constexpr int32_t BUFFER_NUM = 2;

using namespace AscendC;

template<typename TYPE_X> class KernelAtanh {
    using T = TYPE_X;
public:
    __aicore__ inline KernelAtanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum, 
                                uint32_t finalSmallTileNum, uint32_t tileDataNum, 
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum, 
                                uint32_t tailBlockNum,TPipe * pipeIn) 
    {
        // ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * coreNum;
        this->tileDataNum = tileDataNum;
        if (coreNum < tailBlockNum) { 
          this->coreDataNum = bigCoreDataNum;
          this->tileNum = finalBigTileNum;
          this->tailDataNum = bigTailDataNum;
        }
        else { 
          this->coreDataNum = smallCoreDataNum;
          this->tileNum = finalSmallTileNum;
          this->tailDataNum = smallTailDataNum;
          globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreNum - tailBlockNum);
        }
    
        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_X*)y + globalBufferIndex, this->coreDataNum);
        pipeIn->InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipeIn->InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        
        
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
      AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
      AscendC::DataCopy(xLocal, yGm[progress * this->tileDataNum], this->processDataNum);
      inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
      AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
      AscendC::LocalTensor<TYPE_X> zLocal = outQueueZ.AllocTensor<TYPE_X>();
      //AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpBuf.Get<uint8_t>();
      if constexpr (std::is_same_v<TYPE_X, bool>) {

            DataCopy(zLocal,xLocal,this->processDataNum);

          
      }
      if constexpr (std::is_same_v<TYPE_X, int8_t> ||std::is_same_v<TYPE_X, uint8_t>) {

            DataCopy(zLocal,xLocal,this->processDataNum);
      }
      else if constexpr (std::is_same_v<TYPE_X, int16_t>){

            DataCopy(zLocal,xLocal,this->processDataNum);
      }
      else if constexpr (std::is_same_v<TYPE_X, bfloat16_t>){


        DataCopy(zLocal,xLocal,this->processDataNum);

      }

      else if constexpr (std::is_same_v<TYPE_X, int32_t>){

            DataCopy(zLocal,xLocal,this->processDataNum);
      }
      else if constexpr (std::is_same_v<TYPE_X, half>){

            DataCopy(zLocal,xLocal,this->processDataNum);


      }
      else if constexpr (std::is_same_v<TYPE_X, float> ) {

        DataCopy(zLocal,xLocal,this->processDataNum);

      }
      outQueueZ.EnQue<TYPE_X>(zLocal);
      inQueueX.FreeTensor(xLocal);

    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
      AscendC::LocalTensor<TYPE_X> zLocal = outQueueZ.DeQue<TYPE_X>();  
      AscendC::DataCopy(xGm[progress * this->tileDataNum], zLocal, this->processDataNum);
      outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe * pipe;;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1, tmp2;
    //AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    GlobalTensor<TYPE_X> xGm, yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;

};


template<typename TYPE_X1> class KernelAssign_Broadcast {
    using T = TYPE_X1;
public:
    __aicore__ inline KernelAssign_Broadcast() {}
    __aicore__ inline void Init( GM_ADDR x1, GM_ADDR y,int32_t y_dimensional,int32_t* y_ndarray, int32_t* x1_ndarray, int32_t* x2_ndarray, int32_t* y_sumndarray, int32_t* x1_sumndarray, int32_t* x2_sumndarray) {
      
        this->y_dimensional = y_dimensional;

        this->y_ndarray = y_ndarray;

        this->x1_ndarray = x1_ndarray;
        this->x2_ndarray = x2_ndarray;

        this->y_sumndarray = y_sumndarray;
        this->x1_sumndarray = x1_sumndarray;
        this->x2_sumndarray = x2_sumndarray;


        yGm.SetGlobalBuffer((__gm__ TYPE_X1*)x1, 1);
        x1Gm.SetGlobalBuffer((__gm__ TYPE_X1*)y, 1);


    }
    __aicore__ inline void Process() {

      

        int dim = this->y_dimensional;
        
        for(int j=0; j<this->y_sumndarray[dim]; j++)
        {
            int x1_start = 0, x2_start = 0;
            for(int k=0; k<dim; k++)
            {
                // if(this->x1_ndarray[k] != 1){
                //     x1_start += this->x1_sumndarray[k] * (j / this->y_sumndarray[k] % this->y_ndarray[k]);
                // }
                if(this->x2_ndarray[k] != 1){
                    x2_start += this->x2_sumndarray[k] * (j / this->y_sumndarray[k] % this->y_ndarray[k]);
                }
            }
            
            yGm.SetValue(j, x1Gm.GetValue(x2_start));
        }
    }
    
  //  __aicore__ inline void Process() {
  //   int dim = this->y_dimensional;
  //   int total = this->y_sumndarray[dim];   // y 的总元素数

  //   // 缓存成员指针，减少 this 解引用
  //   const int* y_ndarray     = this->y_ndarray;
  //   const int* x2_ndarray    = this->x2_ndarray;
  //   const int* x2_sumndarray = this->x2_sumndarray;

  //   // 多维坐标，全 0 起始
  //   int coords[8] = {0};   // 假定 dim <= 8，实际可按最大维数调整
  //   int x2_start = 0;

  //   for (int j = 0; j < total; ++j) {
  //       yGm.SetValue(j, x1Gm.GetValue(x2_start));

  //       // ---- 增量更新坐标（最低维 k=0 最快） ----
  //       int k = 0;
  //       while (k < dim) {
  //           // 先累加一个 stride（如果该维参与映射）
  //           if (x2_ndarray[k] != 1) {
  //               x2_start += x2_sumndarray[k];
  //           }

  //           // 坐标 +1，若未满则停止进位
  //           if (++coords[k] < y_ndarray[k]) {
  //               break;
  //           }

  //           // 溢出：该维归零，并撤回本维贡献的所有 stride
  //           if (x2_ndarray[k] != 1) {
  //               x2_start -= x2_sumndarray[k] * y_ndarray[k];
  //           }
  //           coords[k] = 0;
  //           ++k;                     // 向高位（k+1）进位
  //       }
  //   }
  // }
    // __aicore__ inline void Process() {
    //     int dim = this->y_dimensional;
    //     int total = this->y_sumndarray[dim];
    
    //     for (int j = 0; j < total; ++j) {
    //         int x1_start = 0, x2_start = 0;
    
    //         int k = 0;
    //         // 两路展开的主循环：每次处理 k 和 k+1
    //         for (; k + 1 < dim; k += 2) {
    //             if (this->x2_ndarray[k] != 1) {
    //                 x2_start += this->x2_sumndarray[k] *
    //                     (j / this->y_sumndarray[k] % this->y_ndarray[k]);
    //             }
    //             if (this->x2_ndarray[k+1] != 1) {
    //                 x2_start += this->x2_sumndarray[k+1] *
    //                     (j / this->y_sumndarray[k+1] % this->y_ndarray[k+1]);
    //             }
    //         }
    //         // 处理剩余的维度（dim 为奇数时执行一次）
    //         for (; k < dim; ++k) {
    //             if (this->x2_ndarray[k] != 1) {
    //                 x2_start += this->x2_sumndarray[k] *
    //                     (j / this->y_sumndarray[k] % this->y_ndarray[k]);
    //             }
    //         }
    
    //         yGm.SetValue(j, x1Gm.GetValue(x2_start));
    //     }
    // }
private:



    GlobalTensor<DTYPE_INPUT> x1Gm;
    GlobalTensor<DTYPE_INPUT> yGm;
    int32_t y_dimensional;
    int32_t *y_ndarray;
    int32_t *x1_ndarray;
    int32_t *x2_ndarray;

    int32_t *y_sumndarray;
    int32_t *x1_sumndarray;
    int32_t *x2_sumndarray;

};

template<typename TYPE_X>
class KernelAssignHalf {
    using T = TYPE_X;

public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR y,
                               uint32_t coreDataNum,
                               uint32_t tileDataNum,
                               uint32_t tileNum,
                               uint32_t lastTileDataNum,TPipe * pipeIn)
    {
        uint32_t coreIdx = GetBlockIdx();

        this->coreDataNum = coreDataNum;
        this->tileDataNum = tileDataNum;
        this->tileNum = tileNum;
        this->lastTileDataNum = lastTileDataNum;


        uint32_t globalOffset = coreIdx * coreDataNum;
        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalOffset, coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_X*)y + globalOffset, coreDataNum);
        pipeIn->InitBuffer(queBind, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));

       
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < tileNum; i++) {

            uint32_t len =
                (i == tileNum - 1) ? lastTileDataNum : tileDataNum;

            LocalTensor<T> zLocal =queBind.AllocTensor<T>();
            

            //Compute(zLocal, len);
            AscendC::DataCopy(zLocal, yGm[i * this->tileDataNum], len);
            queBind.EnQue(zLocal);
            zLocal = queBind.DeQue<TYPE_X>(); 
            AscendC::DataCopy(xGm[i * this->tileDataNum],zLocal, len);

            queBind.FreeTensor(zLocal);
        }
    }

private:
    // __aicore__ inline void Compute(LocalTensor<T>& zLocal, uint32_t len)
    // {
    //     if constexpr (std::is_same_v<T, float>) {
    //         Duplicate(zLocal, value, len);
    //     }
    //     else if constexpr (std::is_same_v<T, half>) {
    //         Duplicate(zLocal, (half)value, len);
    //     }
    //     else if constexpr (std::is_same_v<T, int8_t> ||
    //                        std::is_same_v<T, uint8_t>) {
    //         auto tmp = tmp1.Get<half>();
    //         Duplicate(tmp, value2, len);
    //         Cast(zLocal, tmp, AscendC::RoundMode::CAST_NONE, len);
    //     }
    //     else if constexpr (std::is_same_v<T, bfloat16_t>) {
    //         auto tmp = tmp2.Get<float>();
    //         Duplicate(tmp, value, len);
    //         Cast(zLocal, tmp, AscendC::RoundMode::CAST_RINT, len);
    //     }
    //     else if constexpr (std::is_same_v<T, int16_t>) {
    //         Duplicate(zLocal, value16, len);
    //     }
    //     else if constexpr (std::is_same_v<T, int32_t>) {
    //         Duplicate(zLocal, value32, len);
    //     }
    // }

private:
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> queBind;
    //AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1, tmp2;
    //AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    GlobalTensor<TYPE_X> xGm, yGm;
    uint32_t coreDataNum;
    uint32_t tileDataNum;
    uint32_t tileNum;
    uint32_t lastTileDataNum;
    float value;
    // AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ;

    // TBuf<AscendC::QuePosition::VECCALC> tmp1, tmp2;
};

extern "C" __global__ __aicore__ void assign(GM_ADDR input, GM_ADDR other, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
     TPipe pipe;
    if(TILING_KEY_IS(0)) {
     GET_TILING_DATA_WITH_STRUCT(AssignTilingDataHalf, tiling_data, tiling);
         KernelAssignHalf<DTYPE_INPUT> op;
        op.Init(input,other,
                tiling_data.coreDataNum,
                tiling_data.tileDataNum,
                tiling_data.tileNum,
                tiling_data.lastTileDataNum,&pipe);
        
        op.Process();
    }
    else if(TILING_KEY_IS(1)) {
    KernelAtanh<DTYPE_INPUT> op;
   
    op.Init(input,other,tiling_data.smallCoreDataNum, 
            tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum, 
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum, 
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum, 
            tiling_data.tailBlockNum,&pipe);  
    op.Process();
    }
    else if(TILING_KEY_IS(2)) {
        KernelAssign_Broadcast<DTYPE_INPUT> op;
        op.Init(input, other, tiling_data.y_dimensional,tiling_data.y_ndarray, tiling_data.x1_ndarray, tiling_data.x2_ndarray,
                tiling_data.y_sumndarray, tiling_data.x1_sumndarray, tiling_data.x2_sumndarray);
        op.Process();
    }
}