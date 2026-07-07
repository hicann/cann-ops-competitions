/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gelu_grad_v2.h
 * \brief
 */
#ifndef GELU_GRAD_V2_H
#define GELU_GRAD_V2_H
#pragma once

using namespace AscendC;

template <typename T>
class GeluGradV2None {
public:
    __aicore__ inline GeluGradV2None(){};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR z,
                                uint64_t smallCoreDataNum, uint64_t bigCoreDataNum,
                                uint64_t finalBigTileNum, uint64_t finalSmallTileNum,
                                uint64_t tileDataNum, uint64_t smallTailDataNum,
                                uint64_t bigTailDataNum, uint64_t tailBlockNum,
                                uint64_t usedDb    
    );
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);
    __aicore__ inline void ComputeWithNone(LocalTensor<float> &z, 
                                                            LocalTensor<float> &dy, LocalTensor<float> &x,
                                                            uint32_t length);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueIN;
    TQue<QuePosition::VECOUT, 1> outQueueOUT;
    GlobalTensor<T> dyGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> zGm;
    TBuf<TPosition::VECCALC> resultTmpBuf;
    TBuf<TPosition::VECCALC> calcValueLocal;
    TBuf<TPosition::VECCALC> tempBuf;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
    int32_t bufferNum;
};

template <typename T>
__aicore__ inline void GeluGradV2None<T>::Init(GM_ADDR dy, GM_ADDR x, GM_ADDR z,
                                        uint64_t smallCoreDataNum, uint64_t bigCoreDataNum,
                                        uint64_t finalBigTileNum, uint64_t finalSmallTileNum,
                                        uint64_t tileDataNum, uint64_t smallTailDataNum,
                                        uint64_t bigTailDataNum, uint64_t tailBlockNum,
                                        uint64_t usedDb)
{
    // AscendC::printf("gelu_grad_v2 Init!\n");
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
    this->tileDataNum = tileDataNum;
    if (coreId < tailBlockNum) {
        this->coreDataNum = bigCoreDataNum;
        this->tileNum = finalBigTileNum;
        this->tailDataNum = bigTailDataNum;
    } else {
        this->coreDataNum = smallCoreDataNum;
        this->tileNum = finalSmallTileNum;
        this->tailDataNum = smallTailDataNum;
        globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
    }
    // AscendC::printf("this->coreDataNum = %d\n", this->coreDataNum);
    // AscendC::printf("this->tileNum = %d\n", this->tileNum);
    // AscendC::printf("this->tailDataNum = %d\n", this->tailDataNum);
    this->bufferNum = 1;
    if (static_cast<int32_t>(usedDb) == 1) {
        // 2 means using double buffer
        this->bufferNum = 2;
    }  
    // AscendC::printf("this->bufferNum = %d\n", this->bufferNum);
    dyGm.SetGlobalBuffer((__gm__ T*)dy + globalBufferIndex, this->coreDataNum);
    xGm.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
    zGm.SetGlobalBuffer((__gm__ T*)z + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inQueueIN, this->bufferNum, this->tileDataNum * 2 * sizeof(T));
    pipe.InitBuffer(outQueueOUT, this->bufferNum, this->tileDataNum * sizeof(T));

    if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        pipe.InitBuffer(resultTmpBuf, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(calcValueLocal, this->tileDataNum * 2 * sizeof(float));
    }    
    pipe.InitBuffer(tempBuf, this->tileDataNum * 5 * sizeof(float));
}

template <typename T>
__aicore__ inline void GeluGradV2None<T>::CopyIn(int32_t progress)
{
    LocalTensor<T> inLocal = inQueueIN.AllocTensor<T>();

    DataCopy(inLocal[0], dyGm[progress * this->tileDataNum], this->processDataNum);
    DataCopy(inLocal[this->tileDataNum], xGm[progress * this->tileDataNum], this->processDataNum);

    inQueueIN.EnQue(inLocal);
}

template <typename T>
__aicore__ inline void GeluGradV2None<T>::CopyOut(int32_t progress)
{
    LocalTensor<T> outLocal = outQueueOUT.DeQue<T>();

    DataCopy(zGm[progress * this->tileDataNum], outLocal, this->processDataNum);

    outQueueOUT.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void GeluGradV2None<T>::Compute(int32_t progress)
{
    LocalTensor<T> inLocal = inQueueIN.DeQue<T>();
    if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        // convert bf16 to fp32
        LocalTensor<float> calcValueLocalFP32 = calcValueLocal.Get<float>();
        // 3 means there are 3 inputs
        Cast(calcValueLocalFP32, inLocal, RoundMode::CAST_NONE, this->tileDataNum * 2);
        // DumpTensor(inLocal, 1001, 16);
        // DumpTensor(inLocal[this->tileDataNum], 1002, 16);
        // DumpTensor(inLocal[2 * (this->tileDataNum)], 1003, 16);
        // AscendC::printf("inLocal[0] = %f\n", inLocal.GetValue(0));
        // AscendC::printf("inLocal[this->tileDataNum] = %f\n", inLocal[this->tileDataNum].GetValue(0));
        // AscendC::printf("inLocal[2 * this->tileDataNum] = %f\n", inLocal[2 * this->tileDataNum].GetValue(0));
        LocalTensor<float> dyLocal = calcValueLocalFP32;
        LocalTensor<float> xLocal = calcValueLocalFP32[this->tileDataNum];
        // AscendC::printf("dyLocal[0] = %f\n", xLocal.GetValue(0));
        // AscendC::printf("xLocal[0] = %f\n", yLocal.GetValue(0));
        // DumpTensor(dyLocal, 2001, 16);
        // DumpTensor(xLocal, 2002, 16);
        LocalTensor<float> outLoclFp32 = resultTmpBuf.Get<float>();

        ComputeWithNone(outLoclFp32, dyLocal, xLocal, this->processDataNum);
        // AscendC::printf("outLoclFp32[0] = %f\n", outLoclFp32.GetValue(0));
        // DumpTensor(outLoclFp32, 3001, 16);
        // convert fp32 to bf16
        LocalTensor<T> outLocal = outQueueOUT.AllocTensor<T>();
        // 3 means there are 3 inputs
        Cast(outLocal, outLoclFp32, RoundMode::CAST_RINT, this->tileDataNum);
        // DumpTensor(outLocal, 4001, 16);
        // AscendC::printf("outLocal[0] = %f\n", outLocal.GetValue(0));
        outQueueOUT.EnQue(outLocal);
    } else {
        LocalTensor<T> dyLocal = inLocal;
        LocalTensor<T> xLocal = inLocal[this->tileDataNum];
        LocalTensor<T> outLocal = outQueueOUT.AllocTensor<T>();
 
        ComputeWithNone(outLocal, dyLocal, xLocal, this->processDataNum);

        outQueueOUT.EnQue(outLocal);
    }
    inQueueIN.FreeTensor(inLocal);
}

// template <typename T>
// __aicore__ inline void GeluGradV2None<T>::ComputeWithNone(LocalTensor<float> &z, 
//                                                             LocalTensor<float> &dy, LocalTensor<float> &x,
//                                                             uint32_t length)
// {
//     // 
//     //1. 计算erf()
//     // cdf_erf_input, cdf_erf_x2, cdf_erf_numer, cdf_erf_denom, pdf_res

//     LocalTensor<float> tempLocal = tempBuf.Get<float>();
//     LocalTensor<float> cdf_erf_input = tempLocal;
//     LocalTensor<float> cdf_erf_x2 = tempLocal[this->tileDataNum];
//     LocalTensor<float> cdf_erf_numer = tempLocal[2 * this->tileDataNum];
//     LocalTensor<float> cdf_erf_denom = tempLocal[3 * this->tileDataNum];
//     LocalTensor<float> pdf_res = tempLocal[4 * this->tileDataNum];

//     // 计算 erf(x/sqrt(2))
//     Muls(cdf_erf_input, x, static_cast<float>(0.707106781), length); // 707106781 = 1/sqrt(2)

//     // 截断 -3.92 ≤ x ≤ 3.92
//     Mins(cdf_erf_input, cdf_erf_input, static_cast<float>(3.92), length);
//     Maxs(cdf_erf_input, cdf_erf_input, static_cast<float>(-3.92), length);
    
//     Mul(cdf_erf_x2, cdf_erf_input, cdf_erf_input, length);

//     Muls(cdf_erf_numer, cdf_erf_x2, static_cast<float>(0.53443748819e-1), length);
//     Adds(cdf_erf_numer, cdf_erf_numer, static_cast<float>(0.75517016694e1), length);
//     Mul(cdf_erf_numer, cdf_erf_numer, cdf_erf_x2, length);
//     Adds(cdf_erf_numer, cdf_erf_numer, static_cast<float>(0.10162808918e3), length);
//     Mul(cdf_erf_numer, cdf_erf_numer, cdf_erf_x2, length);
//     Adds(cdf_erf_numer, cdf_erf_numer, static_cast<float>(0.13938061484e4), length);
//     Mul(cdf_erf_numer, cdf_erf_numer, cdf_erf_x2, length);
//     Adds(cdf_erf_numer, cdf_erf_numer, static_cast<float>(0.50637915060e4), length);
//     Mul(cdf_erf_numer, cdf_erf_numer, cdf_erf_x2, length);
//     Adds(cdf_erf_numer, cdf_erf_numer, static_cast<float>(0.29639384698e5), length);
//     Mul(cdf_erf_numer, cdf_erf_numer, cdf_erf_input, length);

//     Adds(cdf_erf_denom, cdf_erf_x2, static_cast<float>(0.31212858877e2), length); 
//     Mul(cdf_erf_denom, cdf_erf_denom, cdf_erf_x2, length);
//     Adds(cdf_erf_denom, cdf_erf_denom, static_cast<float>(0.39856963806e3), length); 
//     Mul(cdf_erf_denom, cdf_erf_denom, cdf_erf_x2, length);  
//     Adds(cdf_erf_denom, cdf_erf_denom, static_cast<float>(0.30231248150e4), length); 
//     Mul(cdf_erf_denom, cdf_erf_denom, cdf_erf_x2, length);        
//     Adds(cdf_erf_denom, cdf_erf_denom, static_cast<float>(0.13243365831e5), length); 
//     Mul(cdf_erf_denom, cdf_erf_denom, cdf_erf_x2, length);  
//     Adds(cdf_erf_denom, cdf_erf_denom, static_cast<float>(0.26267224157e5), length); 

//     Div(cdf_erf_numer, cdf_erf_numer, cdf_erf_denom, length); 

    
//     Muls(cdf_erf_numer, cdf_erf_numer, static_cast<float>(0.5), length);
//     Adds(cdf_erf_numer, cdf_erf_numer, static_cast<float>(0.5), length);

//     // pdf
//     Mul(pdf_res, x, x, length); 
//     Muls(pdf_res, pdf_res, static_cast<float>(-0.5), length);
//     Exp(pdf_res, pdf_res, length);
//     Muls(pdf_res, pdf_res, static_cast<float>(0.3989422804), length);

//     Mul(pdf_res, pdf_res, x, length); 
//     Add(cdf_erf_numer, cdf_erf_numer, pdf_res, length); 
//     Mul(z, cdf_erf_numer, dy, length); 
// }

template <typename T>
__aicore__ inline void GeluGradV2None<T>::ComputeWithNone(LocalTensor<float> &z, 
                                                            LocalTensor<float> &dy, LocalTensor<float> &x,
                                                            uint32_t length)
{
    LocalTensor<float> tempLocal = tempBuf.Get<float>();
    LocalTensor<float> tmp0 = tempLocal;
    LocalTensor<float> tmp1 = tempLocal[this->tileDataNum];

    AscendC::ShiftRight(tmp0.ReinterpretCast<uint32_t>(), x.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(31), length);
    AscendC::Cast(tmp0, tmp0.ReinterpretCast<int32_t>(), AscendC::RoundMode::CAST_NONE, length);
    AscendC::Adds(tmp0, tmp0, static_cast<float>(-0.5), length);
    
    AscendC::Abs(x, x, length);
    AscendC::Mins(x, x, static_cast<float>(30.0), length);
    AscendC::Muls(z, x, static_cast<float>(-0.7978849236182753), length);
    AscendC::Adds(z, z, static_cast<float>(-6.223065142195392), length);
    AscendC::Mul(z, z, x, length);
    AscendC::Adds(z, z, static_cast<float>(-20.451359645895863), length);
    AscendC::Mul(z, z, x, length);
    AscendC::Adds(z, z, static_cast<float>(-29.633095101286557), length);
    AscendC::Mul(z, z, x, length);
    AscendC::Adds(z, z, static_cast<float>(-4.474629239730099), length);
    AscendC::Mul(z, z, x, length);
    AscendC::Adds(z, z, static_cast<float>(30.9817411426), length);
    
    AscendC::Adds(tmp1, x, static_cast<float>(7.79950327625), length);
    AscendC::Mul(tmp1, tmp1, x, length);
    AscendC::Adds(tmp1, tmp1, static_cast<float>(26.6304065815), length);
    AscendC::Mul(tmp1, tmp1, x, length);
    AscendC::Adds(tmp1, tmp1, static_cast<float>(44.965039885), length);
    AscendC::Mul(tmp1, tmp1, x, length);
    AscendC::Adds(tmp1, tmp1, static_cast<float>(30.9817411527), length);
    
    AscendC::Div(z, z, tmp1, length);
    AscendC::Mul(x, x, x, length);
    AscendC::Muls(x, x, static_cast<float>(-0.5), length);
    AscendC::Exp(x, x, length);
    AscendC::Mul(z, z, x,  length);

    AscendC::Mul(z, tmp0, z, length);
    AscendC::Adds(tmp1, tmp0, static_cast<float>(-0.5), length);
    AscendC::Sub(z, z, tmp1, length);
    AscendC::Mul(z, z, dy, length);
}


template <typename T>
__aicore__ inline void GeluGradV2None<T>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount - 1; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    // AscendC::printf("this->processDataNum = %d \n", this->processDataNum);
    CopyIn(loopCount - 1);
    Compute(loopCount - 1);
    CopyOut(loopCount - 1);
}


template <typename T>
class GeluGradV2Tanh {
public:
    __aicore__ inline GeluGradV2Tanh(){};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR z, 
                                uint64_t smallCoreDataNum, uint64_t bigCoreDataNum,
                                uint64_t finalBigTileNum, uint64_t finalSmallTileNum,
                                uint64_t tileDataNum, uint64_t smallTailDataNum,
                                uint64_t bigTailDataNum, uint64_t tailBlockNum,
                                uint64_t usedDb   
                                );
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);
    __aicore__ inline void ComputeWithTanh(LocalTensor<float> &z, 
                                                            LocalTensor<float> &dy, LocalTensor<float> &x,
                                                            uint32_t length);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueIN;
    TQue<QuePosition::VECOUT, 1> outQueueOUT;
    GlobalTensor<T> dyGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> zGm;
    TBuf<TPosition::VECCALC> resultTmpBuf;
    TBuf<TPosition::VECCALC> calcValueLocal;
    TBuf<TPosition::VECCALC> tempBuf;

    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
    int32_t bufferNum;
};

template <typename T>
__aicore__ inline void GeluGradV2Tanh<T>::Init(GM_ADDR dy, GM_ADDR x, GM_ADDR z,
                                uint64_t smallCoreDataNum, uint64_t bigCoreDataNum,
                                uint64_t finalBigTileNum, uint64_t finalSmallTileNum,
                                uint64_t tileDataNum, uint64_t smallTailDataNum,
                                uint64_t bigTailDataNum, uint64_t tailBlockNum,
                                uint64_t usedDb
                                )
{
    // AscendC::printf("GeluGradV2Tanh Init!\n");
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
    this->tileDataNum = tileDataNum;
    if (coreId < tailBlockNum) {
        this->coreDataNum = bigCoreDataNum;
        this->tileNum = finalBigTileNum;
        this->tailDataNum = bigTailDataNum;
    } else {
        this->coreDataNum = smallCoreDataNum;
        this->tileNum = finalSmallTileNum;
        this->tailDataNum = smallTailDataNum;
        globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
    }
    // AscendC::printf("this->coreDataNum = %d\n", this->coreDataNum);
    // AscendC::printf("this->tileNum = %d\n", this->tileNum);
    // AscendC::printf("this->tailDataNum = %d\n", this->tailDataNum);
    this->bufferNum = 1;
    if (static_cast<int32_t>(usedDb) == 1) {
        // 2 means using double buffer
        this->bufferNum = 2;
    }  

    // AscendC::printf("this->bufferNum = %d\n", this->bufferNum);
    dyGm.SetGlobalBuffer((__gm__ T*)dy + globalBufferIndex, this->coreDataNum);
    xGm.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
    zGm.SetGlobalBuffer((__gm__ T*)z + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inQueueIN, this->bufferNum, this->tileDataNum * 2 * sizeof(T));
    pipe.InitBuffer(outQueueOUT, this->bufferNum, this->tileDataNum * sizeof(T));

    if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        pipe.InitBuffer(resultTmpBuf, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(calcValueLocal, this->tileDataNum * 2 * sizeof(float));
    }    
    pipe.InitBuffer(tempBuf, this->tileDataNum * 7 * sizeof(float));      
}

template <typename T>
__aicore__ inline void GeluGradV2Tanh<T>::CopyIn(int32_t progress)
{
    LocalTensor<T> inLocal = inQueueIN.AllocTensor<T>();

    DataCopy(inLocal[0], dyGm[progress * this->tileDataNum], this->processDataNum);
    DataCopy(inLocal[this->tileDataNum], xGm[progress * this->tileDataNum], this->processDataNum);

    inQueueIN.EnQue(inLocal);
}

template <typename T>
__aicore__ inline void GeluGradV2Tanh<T>::CopyOut(int32_t progress)
{
    LocalTensor<T> outLocal = outQueueOUT.DeQue<T>();

    DataCopy(zGm[progress * this->tileDataNum], outLocal, this->processDataNum);

    outQueueOUT.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void GeluGradV2Tanh<T>::ComputeWithTanh(LocalTensor<float> &z, 
                                                            LocalTensor<float> &dy, LocalTensor<float> &x,
                                                            uint32_t length)
{
    // const LocalTensor: one
    // tmp LocalTensor:px,x2, res0, t, resp, maskcmpMask
    // // 完整的计算流程
    LocalTensor<float> tempLocal = tempBuf.Get<float>();
    LocalTensor<float> one = tempLocal;
    LocalTensor<float> px = tempLocal[this->tileDataNum];
    LocalTensor<float> x2 = tempLocal[2 * this->tileDataNum];
    LocalTensor<float> res0 = tempLocal[3 * this->tileDataNum];
    LocalTensor<float> t = tempLocal[4 * this->tileDataNum];    
    LocalTensor<float> resp = tempLocal[5 * this->tileDataNum]; 
    LocalTensor<float> cmpMask = tempLocal[6 * this->tileDataNum]; 

    Duplicate(one, static_cast<float>(1.0), length);
    Duplicate(px, static_cast<float>(-1.595769121605730711759), length);
    Mul(x2, x, x, length);
    Axpy(px, x2, static_cast<float>(-0.0713548162726002527220), length);
    Mul(px, px, x, length);
    Exp(px, px, length);

    Duplicate(res0, static_cast<float>(1.595769121605730711759), length);
    Axpy(res0, x2, static_cast<float>(0.2140644488178007), length);
    Mul(res0, res0, x, length);

    Adds(t, px, static_cast<float>(1.0), length);
    Div(t, one, t, length);

    Mul(resp, px, t, length);
    Mul(resp, resp, res0, length);
    Mul(resp, resp, t, length);

    // length 需要对齐maskLength
    uint32_t maskLength = (length + 63) / 64 * 64;
    Compare(cmpMask.ReinterpretCast<uint8_t>(), resp, resp, AscendC::CMPMODE::EQ, maskLength);
    Select(resp, cmpMask.ReinterpretCast<uint8_t>(), resp, static_cast<float>(0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, maskLength);

    Add(resp, resp, t, length);
    Mul(z, dy, resp, length);
}


template <typename T>
__aicore__ inline void GeluGradV2Tanh<T>::Compute(int32_t progress)
{
    LocalTensor<T> inLocal = inQueueIN.DeQue<T>();
    if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        // convert to fp32
        LocalTensor<float> calcValueLocalFP32 = calcValueLocal.Get<float>();

        Cast(calcValueLocalFP32, inLocal, RoundMode::CAST_NONE, this->tileDataNum * 2);

        LocalTensor<float> dyLocal = calcValueLocalFP32;
        LocalTensor<float> xLocal = calcValueLocalFP32[this->tileDataNum];

        // AscendC::printf("xLocal[0] = %f\n", xLocal.GetValue(0));
        // AscendC::printf("xLocal[0] = %f\n", yLocal.GetValue(0));

        LocalTensor<float> outLoclFp32 = resultTmpBuf.Get<float>();

        ComputeWithTanh(outLoclFp32, dyLocal, xLocal, this->processDataNum);

        // convert fp32 to origin
        LocalTensor<T> outLocal = outQueueOUT.AllocTensor<T>();
        // 3 means there are 3 inputs
        Cast(outLocal, outLoclFp32, RoundMode::CAST_RINT, this->tileDataNum);
        // DumpTensor(outLocal, 4001, 16);
        // AscendC::printf("outLocal[0] = %f\n", outLocal.GetValue(0));
        outQueueOUT.EnQue(outLocal);
    } else {
        LocalTensor<T> dyLocal = inLocal;
        LocalTensor<T> xLocal = inLocal[this->tileDataNum];

        LocalTensor<T> outLocal = outQueueOUT.AllocTensor<T>();

        ComputeWithTanh(outLocal, dyLocal, xLocal, this->processDataNum);

        outQueueOUT.EnQue(outLocal);
    }
    inQueueIN.FreeTensor(inLocal);
}

template <typename T>
__aicore__ inline void GeluGradV2Tanh<T>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount - 1; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    // AscendC::printf("this->processDataNum = %d \n", this->processDataNum);
    CopyIn(loopCount - 1);
    Compute(loopCount - 1);
    CopyOut(loopCount - 1);
}

#endif // _GELU_GRAD_V2_H
