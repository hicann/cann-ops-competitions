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
        //ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
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
        //this->maxLiveNodeCount =0;

        //this->extraBuffer = 0;



        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_X*)y + globalBufferIndex, this->coreDataNum);
        pipeIn->InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipeIn->InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));

        if constexpr (std::is_same_v<TYPE_X, int8_t> ||std::is_same_v<TYPE_X, uint8_t> ){

          pipeIn->InitBuffer(tmp1, this->tileDataNum * sizeof(half));
          pipeIn->InitBuffer(tmp2, this->tileDataNum * sizeof(half));
        }
        else if constexpr (std::is_same_v<TYPE_X, half>) {

          // pipeIn->InitBuffer(tmp1, this->tileDataNum * sizeof(float));
          // pipeIn->InitBuffer(tmp2, this->tileDataNum * sizeof(float));

            
                }
        else if constexpr ( std::is_same_v<TYPE_X, int16_t> ||std::is_same_v<TYPE_X, bfloat16_t> ||std::is_same_v<TYPE_X, int32_t> )
        {
          pipeIn->InitBuffer(tmp1, this->tileDataNum * sizeof(float));
          pipeIn->InitBuffer(tmp2, this->tileDataNum * sizeof(float));
         
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
      AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
      AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
      inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
      AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
      AscendC::LocalTensor<TYPE_X> zLocal = outQueueZ.AllocTensor<TYPE_X>();
      if constexpr (std::is_same_v<TYPE_X, int8_t> ||std::is_same_v<TYPE_X, uint8_t>) {
        auto p4 = tmp1.Get<half>();
        auto p5 = tmp2.Get<half>();
        //auto p6 = tmp6.Get<float>();
        //Duplicate(p4,half(0), this->processDataNum);
        AscendC::Cast(p4, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        //Duplicate(p5,(float)(1),this->processDataNum);

        Adds(p5,p4,(half)(1.0),this->processDataNum);
        //Sub(xLocal,this->two,zLocal,this->processDataNum);
        Muls(p4,p4,(half)(-1.0),this->processDataNum);
        Adds(p4,p4,(half)(1.0),this->processDataNum);

          

        Div(p4,p5,p4,this->processDataNum);
        Ln(p4,p4,this->processDataNum);
        Muls(p4,p4,(half)(0.50),this->processDataNum);
        AscendC::Cast(zLocal, p4, AscendC::RoundMode::CAST_NONE, this->processDataNum);
      }
      else if constexpr (std::is_same_v<TYPE_X, int16_t>){
        auto p4 = tmp1.Get<float>();
        auto p5 = tmp2.Get<float>();
        //auto p6 = tmp6.Get<float>();
        AscendC::Cast(p4, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        //Duplicate(p5,(float)(1),this->processDataNum);

        Adds(p5,p4,(float)(1.0),this->processDataNum);
        //Sub(xLocal,this->two,zLocal,this->processDataNum);
        Muls(p4,p4,(float)(-1.0),this->processDataNum);
        Adds(p4,p4,(float)(1.0),this->processDataNum);

          

        Div(p4,p5,p4,this->processDataNum);
        Ln(p4,p4,this->processDataNum);
        Muls(p4,p4,(float)(0.50),this->processDataNum);
        AscendC::Cast(zLocal, p4, AscendC::RoundMode::CAST_NONE, this->processDataNum);
      }
      else if constexpr (std::is_same_v<TYPE_X, bfloat16_t>){
        auto p4 = tmp1.Get<float>();
        auto p5 = tmp2.Get<float>();
        //auto p6 = tmp6.Get<float>();
        AscendC::Cast(p4, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        //Duplicate(p5,(float)(1),this->processDataNum);

        Adds(p5,p4,(float)(1.0),this->processDataNum);
        //Sub(xLocal,this->two,zLocal,this->processDataNum);
        Muls(p4,p4,(float)(-1.0),this->processDataNum);
        Adds(p4,p4,(float)(1.0),this->processDataNum);

          

        Div(p4,p5,p4,this->processDataNum);
        Ln(p4,p4,this->processDataNum);
        Muls(p4,p4,(float)(0.50),this->processDataNum);
        AscendC::Cast(zLocal, p4, AscendC::RoundMode::CAST_RINT, this->processDataNum);
      }

      else if constexpr (std::is_same_v<TYPE_X, int32_t>){
         auto p4 = tmp1.Get<float>();
        auto p5 = tmp2.Get<float>();
        //auto p6 = tmp6.Get<float>();
        AscendC::Cast(p4, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        //Duplicate(p5,(float)(1),this->processDataNum);

        Adds(p5,p4,(float)(1.0),this->processDataNum);
        //Sub(xLocal,this->two,zLocal,this->processDataNum);
        Muls(p4,p4,(float)(-1.0),this->processDataNum);
        Adds(p4,p4,(float)(1.0),this->processDataNum);

          

        Div(p4,p5,p4,this->processDataNum);
        Ln(p4,p4,this->processDataNum);
        Muls(p4,p4,(float)(0.50),this->processDataNum);
        AscendC::Cast(zLocal, p4, AscendC::RoundMode::CAST_FLOOR, this->processDataNum);
      }
      else if constexpr (std::is_same_v<TYPE_X, half>){

        // auto p4 = tmp1.Get<float>();
        // auto p5 = tmp2.Get<float>();
        // //auto p6 = tmp6.Get<float>();
        // AscendC::Cast(p4, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        // //Duplicate(p5,(float)(1),this->processDataNum);

        // Adds(p5,p4,(float)(1.0),this->processDataNum);
        // //Sub(xLocal,this->two,zLocal,this->processDataNum);
        // Muls(p4,p4,(float)(-1.0),this->processDataNum);
        // Adds(p4,p4,(float)(1.0),this->processDataNum);

          

        // Div(p4,p5,p4,this->processDataNum);
        // Ln(p4,p4,this->processDataNum);
        // Muls(p4,p4,(float)(0.50),this->processDataNum);
        // AscendC::Cast(zLocal, p4, AscendC::RoundMode::CAST_NONE, this->processDataNum);

            Adds(zLocal,xLocal,(half)(1.0),this->processDataNum);
            //Sub(xLocal,this->two,zLocal,this->processDataNum);
            Muls(xLocal,xLocal,(half)(-1.0),this->processDataNum);
            Adds(xLocal,xLocal,(half)(1.0),this->processDataNum);
            //Muls(xLocal,xLocal,(float)(-1.0),this->processDataNum);
            //Adds(xLocal,xLocal,(float)(1.0),this->processDataNum);
            //Sub(xLocal,p7,xLocal,this->processDataNum);
            Div(zLocal,zLocal,xLocal,this->processDataNum);
            Ln(zLocal,zLocal,this->processDataNum);
            Muls(zLocal,zLocal,(half)(0.50),this->processDataNum);


          


      }
      else if constexpr (std::is_same_v<TYPE_X, float> ) {

        //auto p7 = tmp7.Get<float>();
        //Duplicate(p7,(float)(1),this->processDataNum);
        Adds(zLocal,xLocal,(float)(1.0),this->processDataNum);
        //Sub(xLocal,this->two,zLocal,this->processDataNum);
        Muls(xLocal,xLocal,(float)(-1.0),this->processDataNum);
        Adds(xLocal,xLocal,(float)(1.0),this->processDataNum);
        //Muls(xLocal,xLocal,(float)(-1.0),this->processDataNum);
        //Adds(xLocal,xLocal,(float)(1.0),this->processDataNum);
        //Sub(xLocal,p7,xLocal,this->processDataNum);
        Div(zLocal,zLocal,xLocal,this->processDataNum);
        Ln(zLocal,zLocal,this->processDataNum);
        Muls(zLocal,zLocal,(float)(0.50),this->processDataNum);
        
      }
      outQueueZ.EnQue<TYPE_X>(zLocal);
      inQueueX.FreeTensor(xLocal);

    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
      AscendC::LocalTensor<TYPE_X> zLocal = outQueueZ.DeQue<TYPE_X>();  
      AscendC::DataCopy(yGm[progress * this->tileDataNum], zLocal, this->processDataNum);
      outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe * pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1, tmp2;
    GlobalTensor<TYPE_X> xGm, yGm;
    //LocalTensor<float> two;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template<typename TYPE_X> class KernelAtanhHalf {
    using T = TYPE_X;
public:
    __aicore__ inline KernelAtanhHalf() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum,
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
        //this->maxLiveNodeCount =0;

        //this->extraBuffer = 0;



        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_X*)y + globalBufferIndex, this->coreDataNum);
        pipeIn->InitBuffer(inQueueX, 1, this->tileDataNum * sizeof(TYPE_X));
        pipeIn->InitBuffer(outQueueZ, 1, this->tileDataNum * sizeof(TYPE_X));

        if constexpr (std::is_same_v<TYPE_X, int8_t> ||std::is_same_v<TYPE_X, uint8_t> ){

          pipeIn->InitBuffer(tmp1, this->tileDataNum * sizeof(half));
          pipeIn->InitBuffer(tmp2, this->tileDataNum * sizeof(half));
        }
        else if constexpr ( std::is_same_v<TYPE_X, int16_t> ||std::is_same_v<TYPE_X, bfloat16_t> ||std::is_same_v<TYPE_X, int32_t> )
        {
          pipeIn->InitBuffer(tmp1, this->tileDataNum * sizeof(float));
          pipeIn->InitBuffer(tmp2, this->tileDataNum * sizeof(float));
         
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
      AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
      AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
      inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
      AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
      AscendC::LocalTensor<TYPE_X> zLocal = outQueueZ.AllocTensor<TYPE_X>();
      if constexpr (std::is_same_v<TYPE_X, int8_t> ||std::is_same_v<TYPE_X, uint8_t>) {
        auto p4 = tmp1.Get<half>();
        auto p5 = tmp2.Get<half>();
        //auto p6 = tmp6.Get<float>();
        AscendC::Cast(p4, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        //Duplicate(p5,(float)(1),this->processDataNum);

        Adds(p5,p4,(half)(1.0),this->processDataNum);
        //Sub(xLocal,this->two,zLocal,this->processDataNum);
        Muls(p4,p4,(half)(-1.0),this->processDataNum);
        Adds(p4,p4,(half)(1.0),this->processDataNum);

          

        Div(p4,p5,p4,this->processDataNum);
        Ln(p4,p4,this->processDataNum);
        Muls(p4,p4,(half)(0.50),this->processDataNum);
        AscendC::Cast(zLocal, p4, AscendC::RoundMode::CAST_NONE, this->processDataNum);
      }
      else if constexpr (std::is_same_v<TYPE_X, int16_t>){
        auto p4 = tmp1.Get<float>();
        auto p5 = tmp2.Get<float>();
        //auto p6 = tmp6.Get<float>();
        AscendC::Cast(p4, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        //Duplicate(p5,(float)(1),this->processDataNum);

        Adds(p5,p4,(float)(1.0),this->processDataNum);
        //Sub(xLocal,this->two,zLocal,this->processDataNum);
        Muls(p4,p4,(float)(-1.0),this->processDataNum);
        Adds(p4,p4,(float)(1.0),this->processDataNum);

          

        Div(p4,p5,p4,this->processDataNum);
        Ln(p4,p4,this->processDataNum);
        Muls(p4,p4,(float)(0.50),this->processDataNum);
        AscendC::Cast(zLocal, p4, AscendC::RoundMode::CAST_FLOOR, this->processDataNum);
      }
      else if constexpr (std::is_same_v<TYPE_X, bfloat16_t>){
        auto p4 = tmp1.Get<float>();
        auto p5 = tmp2.Get<float>();
        //auto p6 = tmp6.Get<float>();
        AscendC::Cast(p4, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        //Duplicate(p5,(float)(1),this->processDataNum);

        Adds(p5,p4,(float)(1.0),this->processDataNum);
        //Sub(xLocal,this->two,zLocal,this->processDataNum);
        Muls(p4,p4,(float)(-1.0),this->processDataNum);
        Adds(p4,p4,(float)(1.0),this->processDataNum);

          

        Div(p4,p5,p4,this->processDataNum);
        Ln(p4,p4,this->processDataNum);
        Muls(p4,p4,(float)(0.50),this->processDataNum);
        AscendC::Cast(zLocal, p4, AscendC::RoundMode::CAST_RINT, this->processDataNum);
      }

      else if constexpr (std::is_same_v<TYPE_X, int32_t>){
         auto p4 = tmp1.Get<float>();
        auto p5 = tmp2.Get<float>();
        //auto p6 = tmp6.Get<float>();
        AscendC::Cast(p4, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        //Duplicate(p5,(float)(1),this->processDataNum);

        Adds(p5,p4,(float)(1.0),this->processDataNum);
        //Sub(xLocal,this->two,zLocal,this->processDataNum);
        Muls(p4,p4,(float)(-1.0),this->processDataNum);
        Adds(p4,p4,(float)(1.0),this->processDataNum);

          

        Div(p4,p5,p4,this->processDataNum);
        Ln(p4,p4,this->processDataNum);
        Muls(p4,p4,(float)(0.50),this->processDataNum);
        AscendC::Cast(zLocal, p4, AscendC::RoundMode::CAST_FLOOR, this->processDataNum);
      }
      else if constexpr (std::is_same_v<TYPE_X, half>){

        // auto p4 = tmp1.Get<float>();
        // auto p5 = tmp2.Get<float>();
        // //auto p6 = tmp6.Get<float>();
        // AscendC::Cast(p4, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        // //Duplicate(p5,(float)(1),this->processDataNum);

        // Adds(p5,p4,(float)(1.0),this->processDataNum);
        // //Sub(xLocal,this->two,zLocal,this->processDataNum);
        // Muls(p4,p4,(float)(-1.0),this->processDataNum);
        // Adds(p4,p4,(float)(1.0),this->processDataNum);

          

        // Div(p4,p5,p4,this->processDataNum);
        // Ln(p4,p4,this->processDataNum);
        // Muls(p4,p4,(float)(0.50),this->processDataNum);
        // AscendC::Cast(zLocal, p4, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        Adds(zLocal,xLocal,(half)(1.0),this->processDataNum);
        //Sub(xLocal,this->two,zLocal,this->processDataNum);
        Muls(xLocal,xLocal,(half)(-1.0),this->processDataNum);
        Adds(xLocal,xLocal,(half)(1.0),this->processDataNum);
        //Muls(xLocal,xLocal,(float)(-1.0),this->processDataNum);
        //Adds(xLocal,xLocal,(float)(1.0),this->processDataNum);
        //Sub(xLocal,p7,xLocal,this->processDataNum);
        Div(zLocal,zLocal,xLocal,this->processDataNum);
        Ln(zLocal,zLocal,this->processDataNum);
        Muls(zLocal,zLocal,(half)(0.50),this->processDataNum);
        //GetAtanhTmpBufferFactorSize(this->typeSize, this->maxLiveNodeCount, this->extraBuffer);

      }
      else if constexpr (std::is_same_v<TYPE_X, float> ) {

        //auto p7 = tmp7.Get<float>();
        //Duplicate(p7,(float)(1),this->processDataNum);
        Adds(zLocal,xLocal,(float)(1.0),this->processDataNum);
        //Sub(xLocal,this->two,zLocal,this->processDataNum);
        Muls(xLocal,xLocal,(float)(-1.0),this->processDataNum);
        Adds(xLocal,xLocal,(float)(1.0),this->processDataNum);
        //Muls(xLocal,xLocal,(float)(-1.0),this->processDataNum);
        //Adds(xLocal,xLocal,(float)(1.0),this->processDataNum);
        //Sub(xLocal,p7,xLocal,this->processDataNum);
        Div(zLocal,zLocal,xLocal,this->processDataNum);
        Ln(zLocal,zLocal,this->processDataNum);
        Muls(zLocal,zLocal,(float)(0.50),this->processDataNum);
        
      }
      outQueueZ.EnQue<TYPE_X>(zLocal);
      inQueueX.FreeTensor(xLocal);

    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
      AscendC::LocalTensor<TYPE_X> zLocal = outQueueZ.DeQue<TYPE_X>();  
      AscendC::DataCopy(yGm[progress * this->tileDataNum], zLocal, this->processDataNum);
      outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe * pipe;;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1, tmp2;
    GlobalTensor<TYPE_X> xGm, yGm;
    //LocalTensor<float> two;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

extern "C" __global__ __aicore__ void atanh(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{

    if (TILING_KEY_IS(0)) { 
        GET_TILING_DATA(tiling_data, tiling);
        KernelAtanh<DTYPE_INPUT> op;
        TPipe pipe;
        op.Init(input,output,tiling_data.smallCoreDataNum, 
                tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum, 
                tiling_data.finalSmallTileNum, tiling_data.tileDataNum, 
                tiling_data.smallTailDataNum, tiling_data.bigTailDataNum, 
                tiling_data.tailBlockNum,&pipe);  
        op.Process();
    }
    else if (TILING_KEY_IS(1)) {

         GET_TILING_DATA(tiling_data, tiling);
        KernelAtanhHalf<DTYPE_INPUT> op;
        TPipe pipe;
        op.Init(input,output,tiling_data.smallCoreDataNum, 
                tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum, 
                tiling_data.finalSmallTileNum, tiling_data.tileDataNum, 
                tiling_data.smallTailDataNum, tiling_data.bigTailDataNum, 
                tiling_data.tailBlockNum,&pipe);  
        op.Process();

        

    }
    
}