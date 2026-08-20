#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
// tensor num for each queue
constexpr int32_t BUFFER_NUM = 1;

using namespace AscendC;

template<typename TYPE_X>
class KernelFills {
    using T = TYPE_X;

public:
    __aicore__ inline void Init(GM_ADDR y,
                               float value,
                               uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum, 
                                uint32_t finalSmallTileNum, uint32_t tileDataNum, 
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum, 
                                uint32_t tailBlockNum,TPipe * pipeIn)
    {
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
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
          globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
        }

        this->value = value;

        yGm.SetGlobalBuffer((__gm__ TYPE_X*)y + globalBufferIndex, this->coreDataNum);
        pipeIn->InitBuffer(outQueueZ, 1, this->tileDataNum * sizeof(TYPE_X));

        // dtype 预处理
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            value2 = (half)value;
            pipeIn->InitBuffer(tmp1, this->tileDataNum * sizeof(half));

        } else if constexpr (std::is_same_v<T, int16_t>) {
            value16 = (int16_t)value;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            value32 = (int32_t)value;
        } 

        //__bf16 value2 =value;
        else if constexpr (std::is_same_v<TYPE_X, bfloat16_t> )
         {

             //this->valuebf16 = AscendC::ToBfloat16(value);

           pipeIn->InitBuffer(tmp1, this->tileDataNum * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {   
        for (uint32_t i = 0; i < tileNum; i++) {

            uint32_t len =
                (i == tileNum - 1) ? this->tailDataNum : this->tileDataNum;

            LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

            if constexpr (std::is_same_v<T, float>) {
                Duplicate(zLocal, value, len);
            }
            else if constexpr (std::is_same_v<T, half>) {
                Duplicate(zLocal, (half)value, len);
            }
            else if constexpr (std::is_same_v<T, int8_t> ||
                               std::is_same_v<T, uint8_t>) {
                auto tmp = tmp1.Get<half>();
                Duplicate(tmp, value2, len);
                Cast(zLocal, tmp, AscendC::RoundMode::CAST_NONE, len);
            }
            else if constexpr (std::is_same_v<T, bfloat16_t>) {

                 auto tmp = tmp1.Get<float>();
                //__bf16 value2 =value;
                //Duplicate(zLocal, this->valuebf16 , len);
                //Cast(zLocal, tmp, AscendC::RoundMode::CAST_RINT, len);
                Duplicate(tmp, value , len);
               
                Cast(zLocal, tmp, AscendC::RoundMode::CAST_RINT, len);
            }
            else if constexpr (std::is_same_v<T, int16_t>) {
                Duplicate(zLocal, value16, len);
            }
            else if constexpr (std::is_same_v<T, int32_t>) {
                Duplicate(zLocal, value32, len);
            }

            AscendC::DataCopy(
                yGm[i * tileDataNum],
                zLocal,
                len);

            outQueueZ.FreeTensor(zLocal);
        }
    }

private:
    // __aicore__ inline void Compute(LocalTensor<T>& zLocal, uint32_t len)
    // {   
        
        
    // }

private:
    GlobalTensor<T> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    float value;
    half value2;
    int16_t value16;
    int32_t value32;
    bfloat16_t valuebf16;
    //__bf16 valuebf16;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ;

    TBuf<AscendC::QuePosition::VECCALC> tmp1;
};


template<typename TYPE_X> class KernelFillsHalf {
    using T = TYPE_X;
public:
    __aicore__ inline KernelFillsHalf() {}
    __aicore__ inline void Init(GM_ADDR y,float value, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum, 
                                uint32_t finalSmallTileNum, uint32_t tileDataNum, 
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum, 
                                uint32_t tailBlockNum,TPipe * pipeIn) 
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
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
          globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
        }


        this->value =value;
       

        //xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ half*)y + globalBufferIndex, this->coreDataNum);
        //pipeIn->InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipeIn->InitBuffer(outQueueZ, 1, this->tileDataNum * sizeof(half));
        
        
    }
    __aicore__ inline void Process()
    {
        //int32_t loopCount = this->tileNum;
        //this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < this->tileNum; i++) {
            // if (i == this->tileNum - 1) {
            //   this->processDataNum = this->tailDataNum;
            // }
            uint32_t len =
                (i == tileNum - 1) ? this->tailDataNum : this->tileDataNum;

            //CopyIn(i);
            AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
             Duplicate(zLocal,(half)this->value,len);
            AscendC::DataCopy(yGm[i * this->tileDataNum], zLocal, len);
           outQueueZ.FreeTensor(zLocal);
            // Compute(i);
            // CopyOut(i);
        }
    }
private:
    AscendC::TPipe * pipe;;
    //AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1, tmp2;
    GlobalTensor<half> xGm, yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    float value;
};

template<typename TYPE_X>
class KernelFillsBF16 {
    using T = TYPE_X;

public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR y,
                               float value,
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
        this->value = value;
        uint32_t globalOffset = coreIdx * coreDataNum;
        yGm.SetGlobalBuffer((__gm__ bfloat16_t*)y + globalOffset, coreDataNum);
        pipeIn->InitBuffer(outQueueZ, 1, this->tileDataNum * sizeof(bfloat16_t));
        pipeIn->InitBuffer(tmp2, this->tileDataNum * sizeof(float));

    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < tileNum; i++) {

            uint32_t len =
                (i == tileNum - 1) ? lastTileDataNum : tileDataNum;

            LocalTensor<bfloat16_t> zLocal = outQueueZ.AllocTensor<bfloat16_t>();

            auto tmp = tmp2.Get<float>(); 
            Duplicate(tmp, value, len);
            Cast(zLocal, tmp, AscendC::RoundMode::CAST_RINT, len);

            AscendC::DataCopy(
                yGm[i * tileDataNum],
                zLocal,
                len);
            outQueueZ.FreeTensor(zLocal);
        }
    }

private:
    GlobalTensor<bfloat16_t> yGm;
    uint32_t coreDataNum;
    uint32_t tileDataNum;
    uint32_t tileNum;
    uint32_t lastTileDataNum;
    float value;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ;

    TBuf<AscendC::QuePosition::VECCALC> tmp1, tmp2;
};



extern "C" __global__ __aicore__ void fills(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    // GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
     TPipe pipe;
     
    if (TILING_KEY_IS(0)) { 
         GET_TILING_DATA(tiling_data, tiling);
         KernelFills<DTYPE_INPUT> op;
        op.Init(output,
                tiling_data.value,
                tiling_data.smallCoreDataNum, 
                tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum, 
                tiling_data.finalSmallTileNum, tiling_data.tileDataNum, 
                tiling_data.smallTailDataNum, tiling_data.bigTailDataNum, 
                tiling_data.tailBlockNum,&pipe);
        
        op.Process();}
     else if (TILING_KEY_IS(1)) {
         GET_TILING_DATA_WITH_STRUCT(FillsTilingDataBF16, tiling_data, tiling);
         KernelFillsBF16<DTYPE_INPUT> op;

        op.Init(input,output,
                tiling_data.value,
                tiling_data.coreDataNum,
                tiling_data.tileDataNum,
                tiling_data.tileNum,
                tiling_data.lastTileDataNum,&pipe);
        op.Process();
         
         
     }
    else if (TILING_KEY_IS(2)) {

        
        GET_TILING_DATA_WITH_STRUCT(FillsTilingDataHalf, tiling_data, tiling);
        KernelFillsHalf<DTYPE_INPUT> op;
        op.Init(output,tiling_data.value, tiling_data.smallCoreDataNum, 
                tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum, 
                tiling_data.finalSmallTileNum, tiling_data.tileDataNum, 
                tiling_data.smallTailDataNum, tiling_data.bigTailDataNum, 
                tiling_data.tailBlockNum,&pipe);  
        op.Process();
    
    }
    
}