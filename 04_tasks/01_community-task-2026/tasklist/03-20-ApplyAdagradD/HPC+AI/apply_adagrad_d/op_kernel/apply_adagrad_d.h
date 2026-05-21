/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_adagrad_d.h
 * \brief
 */
#ifndef APPLYADAGRADD_H
#define APPLYADAGRADD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "apply_adagrad_d_tiling_data.h"
#include "apply_adagrad_d_tiling_key.h"

namespace NsApplyAdagradD {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr float  NUM_ZERO = 0.0f;

template <typename T>
class ApplyAdagradD {
public:
    __aicore__ inline ApplyAdagradD(){};

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR accum, GM_ADDR lr, GM_ADDR grad, GM_ADDR varOut, GM_ADDR accumOut, const ApplyAdagradDTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress, int32_t lastloopcount);
    __aicore__ inline void CopyOut(int32_t progress, int32_t lastloopcount);
    __aicore__ inline void Compute(int32_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueVAR;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueACCUM;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueLR;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueGRAD;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueVAROUT;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueACCUMOUT;

    TBuf<QuePosition::VECCALC>  tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;

    GlobalTensor<T> inputGMVAR;
    GlobalTensor<T> inputGMACCUM;
    GlobalTensor<T> inputGMLR;
    GlobalTensor<T> inputGMGRAD;
    GlobalTensor<T> outputGMVAROUT;
    GlobalTensor<T> outputGMACCUMOUT;

    int32_t coreDataNum = 0;
    int32_t tileNum = 0;
    int32_t tileDataNum = 0;
    int32_t tailDataNum = 0;
    int32_t processDataNum = 0;
    
    float beta1_lr =  0.0f;
    bool update_slot = true;
};

template <typename T>
__aicore__ inline void ApplyAdagradD<T>::Init(GM_ADDR var, GM_ADDR accum, GM_ADDR lr, GM_ADDR grad, GM_ADDR varOut, GM_ADDR accumOut, const ApplyAdagradDTilingData* tilingData)
{   
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero"); 
    uint32_t blockIdx = GetBlockIdx();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    this->update_slot = tilingData->updateSlots;

    if (blockIdx <  tilingData->tailBlockNum){
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    }
    else{
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (GetBlockIdx() - tilingData->tailBlockNum);
    }

    if (blockIdx == (uint32_t)(GetBlockNum() - 1)) {
            this->tailDataNum = this->tailDataNum  - tilingData->overlapDataNum;
    }

    inputGMVAR.SetGlobalBuffer((__gm__ T*)var + globalBufferIndex, this->coreDataNum);
    inputGMACCUM.SetGlobalBuffer((__gm__ T*)accum + globalBufferIndex, this->coreDataNum);
    inputGMLR.SetGlobalBuffer((__gm__ T*)lr, 1);
    inputGMGRAD.SetGlobalBuffer((__gm__ T*)grad + globalBufferIndex, this->coreDataNum);
    outputGMVAROUT.SetGlobalBuffer((__gm__ T*)varOut + globalBufferIndex, this->coreDataNum);
    outputGMACCUMOUT.SetGlobalBuffer((__gm__ T*)accumOut + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inputQueueVAR, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(inputQueueACCUM, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(inputQueueLR, BUFFER_NUM, 32);
    pipe.InitBuffer(inputQueueGRAD, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(outputQueueVAROUT, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(outputQueueACCUMOUT, BUFFER_NUM, this->tileDataNum * sizeof(T));

    pipe.InitBuffer(tmp5, this->tileDataNum * sizeof(float));  //res1 
    pipe.InitBuffer(tmp6, this->tileDataNum * sizeof(float));  //res2
    
    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>){
        pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(float));  //var_fp32 
        pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(float));  //accum_fp32
        pipe.InitBuffer(tmp3, 32);  //lr_fp32
        pipe.InitBuffer(tmp4, this->tileDataNum * sizeof(float));  //grad_fp32
    }
}

template <typename T>
__aicore__ inline void ApplyAdagradD<T>::CopyIn(int32_t progress, int32_t lastloopcount)
{
    LocalTensor<T> var = inputQueueVAR.AllocTensor<T>();
    LocalTensor<T> accum = inputQueueACCUM.AllocTensor<T>();
    LocalTensor<T> lr = inputQueueLR.AllocTensor<T>();
    LocalTensor<T> grad = inputQueueGRAD.AllocTensor<T>();

    DataCopyExtParams copyParams_lr{1, sizeof(T), 0, 0, 0};
    DataCopyPadExtParams<T> padParams_lr{true, 0, 0, 0};
    DataCopyPad(lr, inputGMLR[0], copyParams_lr, padParams_lr);

    if (progress == lastloopcount) {
        uint32_t dataBytes = this->processDataNum * sizeof(T);
        DataCopyExtParams copyParams{1, dataBytes, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        DataCopyPad(var,   inputGMVAR[progress * this->tileDataNum], copyParams, padParams);
        DataCopyPad(accum, inputGMACCUM[progress * this->tileDataNum], copyParams, padParams);
        DataCopyPad(grad,  inputGMGRAD[progress * this->tileDataNum], copyParams, padParams);
    } else {
        DataCopy(var, inputGMVAR[progress * this->tileDataNum], this->processDataNum);
        DataCopy(accum, inputGMACCUM[progress * this->tileDataNum], this->processDataNum);
        DataCopy(grad, inputGMGRAD[progress * this->tileDataNum], this->processDataNum);
    }

    inputQueueVAR.EnQue(var);
    inputQueueACCUM.EnQue(accum);
    inputQueueLR.EnQue(lr);
    inputQueueGRAD.EnQue(grad);
}

template <typename T>
__aicore__ inline void ApplyAdagradD<T>::CopyOut(int32_t progress, int32_t lastloopcount)
{
    LocalTensor<T> varOut = outputQueueVAROUT.DeQue<T>();
    LocalTensor<T> accumOut = outputQueueACCUMOUT.DeQue<T>();
    if (progress == lastloopcount) {
        uint32_t dataBytes = this->processDataNum * sizeof(T);
        DataCopyExtParams copyParams{1, dataBytes, 0, 0, 0};
        DataCopyPad(outputGMVAROUT[progress * this->tileDataNum], varOut, copyParams);
        DataCopyPad(outputGMACCUMOUT[progress * this->tileDataNum], accumOut, copyParams);
    } 
    else{
        DataCopy(outputGMVAROUT[progress * this->tileDataNum], varOut, this->processDataNum);
        DataCopy(outputGMACCUMOUT[progress * this->tileDataNum], accumOut, this->processDataNum);
    }
    outputQueueVAROUT.FreeTensor(varOut);
    outputQueueACCUMOUT.FreeTensor(accumOut);
}

template <typename T>
__aicore__ inline void ApplyAdagradD<T>::Compute(int32_t progress){
    LocalTensor<T> var = inputQueueVAR.DeQue<T>();
    LocalTensor<T> accum = inputQueueACCUM.DeQue<T>();
    LocalTensor<T> lr = inputQueueLR.DeQue<T>();
    LocalTensor<T> grad = inputQueueGRAD.DeQue<T>();
    LocalTensor<T> varOut = outputQueueVAROUT.AllocTensor<T>();
    LocalTensor<T> accumOut = outputQueueACCUMOUT.AllocTensor<T>();
    LocalTensor<float> var_fp32, accum_fp32, lr_fp32, grad_fp32;
    LocalTensor<float>  res1, res2;

    res1 = tmp5.Get<float>();
    res2 = tmp6.Get<float>();

    if constexpr (std::is_same_v<T, half>){ 
        var_fp32   = tmp1.Get<float>();
        accum_fp32 = tmp2.Get<float>();
        lr_fp32    = tmp3.Get<float>();
        grad_fp32  = tmp4.Get<float>();
        Cast(var_fp32, var, RoundMode::CAST_NONE, this->processDataNum);    
        Cast(accum_fp32, accum, RoundMode::CAST_NONE, this->processDataNum);   
        Cast(lr_fp32, lr, RoundMode::CAST_NONE, 32/sizeof(T));    
        Cast(grad_fp32, grad, RoundMode::CAST_NONE, this->processDataNum);  
        beta1_lr = lr_fp32.GetValue(0);

        if(update_slot){
            Mul(res1, grad_fp32, grad_fp32, this->processDataNum); 
            Add(accum_fp32, accum_fp32, res1, this->processDataNum);
        }
        Muls(res2, grad_fp32, beta1_lr, this->processDataNum); 
        Sqrt(res1, accum_fp32, this->processDataNum);
        Div(res2, res2, res1, this->processDataNum);
        Sub(var_fp32, var_fp32, res2, this->processDataNum);
        Adds(res1, var_fp32, NUM_ZERO, this->processDataNum);  
        Adds(res2, accum_fp32, NUM_ZERO, this->processDataNum); 
        Cast(varOut, res1, RoundMode::CAST_NONE, this->processDataNum);    
        Cast(accumOut, res2, RoundMode::CAST_NONE, this->processDataNum);  

    }
    else if constexpr (std::is_same_v<T, bfloat16_t>){ 
        var_fp32   = tmp1.Get<float>();
        accum_fp32 = tmp2.Get<float>();
        lr_fp32    = tmp3.Get<float>();
        grad_fp32  = tmp4.Get<float>();
        Cast(var_fp32, var, RoundMode::CAST_NONE, this->processDataNum);    
        Cast(accum_fp32, accum, RoundMode::CAST_NONE, this->processDataNum);   
        Cast(lr_fp32, lr, RoundMode::CAST_NONE, 32/sizeof(T));    
        Cast(grad_fp32, grad, RoundMode::CAST_NONE, this->processDataNum);  
        beta1_lr = lr_fp32.GetValue(0);

        if(update_slot == true){
            Mul(res1, grad_fp32, grad_fp32, this->processDataNum); 
            Add(accum_fp32, accum_fp32, res1, this->processDataNum);
        }
        Muls(res2, grad_fp32, beta1_lr, this->processDataNum); 
        Sqrt(res1, accum_fp32, this->processDataNum);
        Div(res2, res2, res1, this->processDataNum);
        Sub(var_fp32, var_fp32, res2, this->processDataNum);
        Adds(res1, var_fp32, NUM_ZERO, this->processDataNum);  
        Adds(res2, accum_fp32, NUM_ZERO, this->processDataNum); 
        Cast(varOut, res1, RoundMode::CAST_ROUND, this->processDataNum);    
        Cast(accumOut, res2, RoundMode::CAST_ROUND, this->processDataNum);  

    }
    else{
        beta1_lr = lr.GetValue(0);
        if(update_slot == true){
            Mul(res1, grad, grad, this->processDataNum); 
            Add(accum, accum, res1, this->processDataNum);
        }
        else{
        }
        Muls(res2, grad, beta1_lr, this->processDataNum); 
        Sqrt(res1, accum, this->processDataNum);
        Div(res2, res2, res1, this->processDataNum);
        Sub(var, var, res2, this->processDataNum);
        Adds(varOut, var, NUM_ZERO, this->processDataNum);  
        Adds(accumOut, accum, NUM_ZERO, this->processDataNum); 
    }

    outputQueueVAROUT.EnQue<T>(varOut);
    outputQueueACCUMOUT.EnQue<T>(accumOut);
    inputQueueVAR.FreeTensor(var);
    inputQueueACCUM.FreeTensor(accum);
    inputQueueLR.FreeTensor(lr);
    inputQueueGRAD.FreeTensor(grad);
}

template <typename T>
__aicore__ inline void ApplyAdagradD<T>::Process()
{
    int32_t loopCount = this->tileNum ;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount; i++) {
        if( i == loopCount - 1){
            this->processDataNum = this->tailDataNum;
        }
        CopyIn(i,loopCount - 1);
        Compute(i);
        CopyOut(i,loopCount - 1);
    }
}

} // namespace NsApplyAdagradD
#endif // APPLYADAGRADD_H
