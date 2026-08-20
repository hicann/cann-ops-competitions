#include "kernel_operator.h"



#define K_MAX_SHAPE_DIM 0


using namespace AscendC;

// #define ASSIGN_DEBUG

#ifdef ASSIGN_DEBUG
#define DEBUG_PRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define DEBUG_PRINT(fmt, ...)
#endif

constexpr int32_t BUFFER_NUM = 2;
typedef long long ll;

//暴力
template <typename T>
class Assign_Bruteforce {
private:
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 0> queBind; 
    // 使用TQueBind替换
    TPipe * pipe;
    GlobalTensor<T> otherGm, outputGm;
    LocalTensor<T> tmpInput;


    ll totallength;
    ll inputlength;
    ll tililength;
    ll Align_length;

    // ll repeatNum;

    int  blockNum;
    int blockIdx;
    int dimNum;
    int dimOutput[4];
    int dimInput[4];

    bool lastDimBroadcast;

    // ll outputStride[4];
    ll inputStride[4];

public:
    __aicore__ inline Assign_Bruteforce() {}
    __aicore__ inline void init(TPipe * pipein,  GM_ADDR other, GM_ADDR output, ll totallength, int dimNum, uint16_t* dimOutput, uint16_t* dimInput) {
        this->pipe = pipein;
        this->totallength = totallength;
        this->dimNum = dimNum;
        for (int i = 0; i < 4; i++) {
            this->dimOutput[i] = dimOutput[i];
            this->dimInput[i] = dimInput[i];
        }
        // 以下变量原从TilingData传入，现改为init内计算
        // ll tililength = dimOutput[dimNum - 1];
        // ll inputlength = ...;
        // bool lastDimBroadcast = ...;
        this->tililength = dimOutput[dimNum - 1];
        this->inputlength = 1;
        for (int i = 0; i < dimNum; i++) {
            this->inputlength *= dimInput[i];
        }
        this->lastDimBroadcast = (dimOutput[dimNum - 1] != dimInput[dimNum - 1]);
        //tililength 向32位对齐

        blockIdx = GetBlockIdx();
        blockNum = GetBlockNum();

        Align_length = (( this->tililength + 31) / 32) * 32;
        // repeatNum = inputlength / tililength;


        // outputStride[dimNum - 1] = 1;
        inputStride[dimNum - 1] = 1;
        for (int k = dimNum - 2; k >= 0; k--) {
            // outputStride[k] = outputStride[k + 1] * dimOutput[k + 1];
            inputStride[k] = inputStride[k + 1] * dimInput[k + 1];
        }



        otherGm.SetGlobalBuffer((__gm__ T *)other,this->inputlength);
        outputGm.SetGlobalBuffer((__gm__ T *)output,totallength);


        pipe->InitBuffer(queBind,BUFFER_NUM, Align_length * sizeof(T));

        DEBUG_PRINT("[Assign] blockIdx=%d blockNum=%d totlength=%lld tililength=%lld inputlength=%lld dimNum=%d lastDimBroadcast=%d\n",
            blockIdx, blockNum, totallength, this->tililength, this->inputlength, dimNum, this->lastDimBroadcast);
        DEBUG_PRINT("[Assign] dimOutput=[%d,%d,%d,%d] dimInput=[%d,%d,%d,%d]\n",
            dimOutput[0], dimOutput[1], dimOutput[2], dimOutput[3],
            dimInput[0], dimInput[1], dimInput[2], dimInput[3]);
    }

    __aicore__ inline void process(){

       

        for(ll offset = blockIdx*tililength ; offset < totallength ; offset += blockNum*tililength){
            ll inOffset = 0;
            ll temp = offset / tililength ;
            //计算当前对应的input地址
            int tmpidx = 0;
            for(int k = dimNum - 2; k >= 0; k--){
                tmpidx = (temp ) % dimOutput[k];
                temp /= dimOutput[k];
                if(dimInput[k] != 1){
                    inOffset += tmpidx * inputStride[k];
                }
            }
            if(offset + tililength > totallength){
                compute(offset, inOffset, totallength - offset);
            }
            else{
                compute(offset, inOffset, tililength);
            }
            // compute(offset, inOffset, ));
        }
            

    }
    __aicore__ inline void compute(ll &offset ,ll & inOffset , ll calclength ){
        //计算当前块的计算长度     
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyParams = {
            (uint16_t)1, 
            (uint32_t)(calclength * sizeof(T)), 
            0,
            0,
            0
        };

        
        DEBUG_PRINT("[Assign] blockIdx=%d offset=%lld inOffset=%lld\n", blockIdx, offset, inOffset);
        queBind.AllocTensor<T>(tmpInput);
        if(lastDimBroadcast){
            DataCopyExtParams copyParamsLastDim = {
                (uint16_t)1, 
                (uint32_t)(dimInput[dimNum - 1] * sizeof(T)), 
                0,
                0,
                0
            };  
            DataCopyPad(tmpInput, otherGm[inOffset], copyParamsLastDim, padParams);
            queBind.EnQue(tmpInput);
            // T tmpscalar = tmpInput(0);
            // tmpInput(1) = tmpscalar;
            // if constexpr ( std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>|| std::is_same_v<T, bool>){
            //     tmpInput(2) = tmpscalar;
            //     // tmpInput(3) = tmpscalar;

            // }

            queBind.DeQue<T>(tmpInput);
            for(int i=0; i< calclength;i++){
                DataCopyPad(outputGm[offset+i], tmpInput, copyParamsLastDim);
            }
            queBind.FreeTensor(tmpInput);
        }
        else{
            DataCopyPad(tmpInput, otherGm[inOffset], copyParams, padParams);
            queBind.EnQue(tmpInput);

            queBind.DeQue<T>(tmpInput);
            DataCopyPad(outputGm[offset], tmpInput, copyParams);
            queBind.FreeTensor(tmpInput);
        }
    }
};


//无广播
template <typename T>
class Assign_NoBroadcast {
private:
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 0> queBind; 
    // 使用TQueBind替换
    TPipe * pipe;
    GlobalTensor<T> otherGm, outputGm;
    LocalTensor<T> tmpInput;
    
    ll tililength;

    ll totallength;
    // ll repeatNum;

    int  blockNum;
    int blockIdx;



public:
    __aicore__ inline Assign_NoBroadcast() {}
    __aicore__ inline void init(TPipe * pipein,  GM_ADDR other, GM_ADDR output, ll totallength, ll tilisize) {
        this->pipe = pipein;
        this->totallength = totallength;
        this->tililength = tilisize;

        // tililength 向32位对齐

        blockIdx = GetBlockIdx();
        blockNum = GetBlockNum();

        otherGm.SetGlobalBuffer((__gm__ T *)other, this->totallength);
        outputGm.SetGlobalBuffer((__gm__ T *)output, totallength);


        pipe->InitBuffer(queBind,1, tililength * sizeof(T));

       
    }

    __aicore__ inline void process(){

       

        for(ll offset = blockIdx*tililength ; offset < totallength ; offset += blockNum*tililength){

            if(offset + tililength > totallength){
                ll tmp = totallength - offset;
                tmp = ((tmp + 31) / 32) * 32; // 向32对齐

                compute(offset, tmp);
            }
            else{
                compute(offset, tililength);
            }
            // compute(offset, inOffset, ));
        }
            

    }
    __aicore__ inline void compute(ll &offset , ll calclength ){
        

        DEBUG_PRINT("[Assign] blockIdx=%d offset=%lld ", blockIdx, offset);
        queBind.AllocTensor<T>(tmpInput);

        DataCopy(tmpInput, otherGm[offset], calclength );
        queBind.EnQue(tmpInput);

        queBind.DeQue<T>(tmpInput);
        DataCopy(outputGm[offset], tmpInput, calclength);
        queBind.FreeTensor(tmpInput);
    }
};

extern "C" __global__ __aicore__ void assign(GM_ADDR input, GM_ADDR other, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (workspace == nullptr) {
        return;
    }
    TPipe pipe;

    if (TILING_KEY_IS(1)) {
        Assign_Bruteforce<DTYPE_INPUT> assign_bruteforce;
        // 旧init传参: totlength, tilisize, inputlength, dimNum, outputShape, inputShape, lastDimBroadcast
        assign_bruteforce.init(&pipe, other, input, tiling_data.totlength, tiling_data.dimNum, tiling_data.outputShape, tiling_data.inputShape);
        assign_bruteforce.process();
        // TODO: 需要广播，最后一维不需要广播
    } else if (TILING_KEY_IS(2)) {
        Assign_Bruteforce<DTYPE_INPUT> assign_bruteforce;
        // 旧init传参: totlength, tilisize, inputlength, dimNum, outputShape, inputShape, lastDimBroadcast
        assign_bruteforce.init(&pipe, other, input, tiling_data.totlength, tiling_data.dimNum, tiling_data.outputShape, tiling_data.inputShape);
        assign_bruteforce.process();
        // TODO: 需要广播，最后一维需要广播
    } else if (TILING_KEY_IS(3)) {
        Assign_NoBroadcast<DTYPE_INPUT> assign_no_broadcast;
        assign_no_broadcast.init(&pipe, other, input, tiling_data.totlength, tiling_data.tilisize);
        assign_no_broadcast.process();
        
        // TODO: 不需要广播
    }
}
