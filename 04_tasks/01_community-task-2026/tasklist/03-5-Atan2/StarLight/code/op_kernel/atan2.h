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
 * \file atan2.h
 * \brief
*/
#ifndef ATAN2_H
#define ATAN2_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 8
#endif

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "atan2_tiling_data.h"
#include "atan2_tiling_key.h"

namespace MyAtan2 {

using namespace AscendC;

#define TAN_PI_BY_EIGHT      0.4142135623730950f
#define NEG_TAN_PI_BY_EIGHT -0.4142135623730950f
#define CONST_PI_BY_EIGHT 0.39269908169872415480783042290994f

constexpr int32_t BUFFER_NUM = 2;
constexpr float CONST_ONE = 1.0f;
constexpr float NE_CONST_ONE = -1.0f;
constexpr float CONST_ZERO = 0.0f;
constexpr float NE_CONST_ZERO = -0.0f;
constexpr int32_t CONST_HIGH_ONE = 0x80000000;
constexpr float TAYLOR[] = {
    1.0f, -1.0f/3, 1.0f/5, -1.0f/7, 1.0f/9, -1.0f/11, 1.0f/13
};
constexpr float CONST_PI = 3.1415926535897932384626433832795f;
constexpr float NE_CONST_PI = -3.1415926535897932384626433832795f;
constexpr float CONST_PI_BY_TWO = 1.5707963267948966192313216916398f;
constexpr float NE_CONST_PI_BY_TWO = -1.5707963267948966192313216916398f;
constexpr float CONST_PI_BY_FOUR = 0.78539816339744830961566084581988f;
constexpr float NE_CONST_PI_BY_FOUR = -0.78539816339744830961566084581988f;
constexpr float CONST_PI_BY_THREE_QUARTERS = 2.356194490192345f;
constexpr float NE_CONST_PI_BY_THREE_QUARTERS = -2.356194490192345f;
constexpr uint32_t CONST_INF_HEX = 0x7F800000;
constexpr uint32_t NE_CONST_INF_HEX = 0xFF800000;

const float CONST_INF = reinterpret_cast<const float&>(CONST_INF_HEX);
const float NE_CONST_INF = reinterpret_cast<const float&>(NE_CONST_INF_HEX);

template <typename TYPE_X, typename TYPE_Y, uint64_t BUFFER_NUM>
class KernelAtan2 {
public:
    __aicore__ inline KernelAtan2(){};

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const Atan2TilingData* tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInAndCompute(int32_t offset);
    __aicore__ inline void CopyOut(int32_t offset);
    __aicore__ inline void CopyInAndCompute16B(LocalTensor<TYPE_X>& x1Local, LocalTensor<TYPE_X>& x2Local, LocalTensor<TYPE_Y>& yLocal, int eventIDMTE2ToV, int offset);
    __aicore__ inline void DoTaylor(LocalTensor<float>& p1, LocalTensor<float>& p2, LocalTensor<float>& p3, LocalTensor<float>& p4);
    __aicore__ inline void AtanCompute(LocalTensor<float>& p1, LocalTensor<float>& p2, LocalTensor<float>& p3, LocalTensor<float>& p4, LocalTensor<float>& sign, LocalTensor<float>& xLocal);
    __aicore__ inline void DoCompute(LocalTensor<float>& x1Local, LocalTensor<float>& x2Local, LocalTensor<float>& yLocal);
private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> castBuf, maskBuf, calcBuf;
    GlobalTensor<TYPE_X> x1Gm;
    GlobalTensor<TYPE_X> x2Gm;
    GlobalTensor<TYPE_Y> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <typename TYPE_X, typename TYPE_Y, uint64_t BUFFER_NUM>
__aicore__ inline void KernelAtan2<TYPE_X, TYPE_Y, BUFFER_NUM>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const Atan2TilingData* tilingData, TPipe* pipeIn)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreNum = GetBlockIdx();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    this->pipe = pipeIn;
    if (coreNum < tilingData->tailBlockNum)
    {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    }
    else
    {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (GetBlockIdx() - tilingData->tailBlockNum);
    }
    x1Gm.SetGlobalBuffer((__gm__ TYPE_X *)x1 + globalBufferIndex, this->coreDataNum);
    x2Gm.SetGlobalBuffer((__gm__ TYPE_X *)x2 + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ TYPE_Y *)y + globalBufferIndex, this->coreDataNum);
    pipe->InitBuffer(inQueue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X) * 2);
    pipe->InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
    pipe->InitBuffer(maskBuf, this->tileDataNum * sizeof(uint8_t) * 4);
    pipe->InitBuffer(calcBuf, this->tileDataNum * sizeof(float) * 6);
    if constexpr ( IsSameType<TYPE_X, bfloat16_t>::value || IsSameType<TYPE_X, half>::value)
    {
        pipe->InitBuffer(castBuf, this->tileDataNum * sizeof(float) * 2);
    }
}

template <typename TYPE_X, typename TYPE_Y, uint64_t BUFFER_NUM>
__aicore__ inline void KernelAtan2<TYPE_X, TYPE_Y, BUFFER_NUM>::CopyInAndCompute16B(LocalTensor<TYPE_X>& x1Local, LocalTensor<TYPE_X>& x2Local,
                                                                                      LocalTensor<TYPE_Y>& yLocal, int eventIDMTE2ToV, int offset)
{
    LocalTensor<float> x1LocalFp = castBuf.Get<float>();
    LocalTensor<float> yLocalFp = x1LocalFp[this->tileDataNum];
    LocalTensor<float> x2LocalFp = x1Local.template ReinterpretCast<float>();
    DataCopy(x1Local, x1Gm[offset], this->processDataNum);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Cast(x1LocalFp, x1Local, RoundMode::CAST_NONE, this->processDataNum);
    DataCopy(x2Local, x2Gm[offset], this->processDataNum);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Cast(x2LocalFp, x2Local, RoundMode::CAST_NONE, this->processDataNum);
    DoCompute(x1LocalFp, x2LocalFp, yLocalFp);
    Cast(yLocal, yLocalFp, RoundMode::CAST_RINT, this->processDataNum);
}

template <typename TYPE_X, typename TYPE_Y, uint64_t BUFFER_NUM>
__aicore__ inline void KernelAtan2<TYPE_X, TYPE_Y, BUFFER_NUM>::DoTaylor(LocalTensor<float>& p1, LocalTensor<float>& p2, LocalTensor<float>& p3, LocalTensor<float>& p4)
{
    Muls(p3, p1, TAN_PI_BY_EIGHT, this->processDataNum);
    Adds(p3, p3, CONST_ONE, this->processDataNum);
    Adds(p2, p1, NEG_TAN_PI_BY_EIGHT, this->processDataNum);
    Div(p3, p2, p3, this->processDataNum);
    Abs(p4, p3, this->processDataNum);
    
    Mul(p2, p4, p4, this->processDataNum);   
    Duplicate(p3, static_cast<float>(TAYLOR[6]), this->processDataNum);
    Mul(p3, p3, p2, this->processDataNum);
    Adds(p3, p3, TAYLOR[5], this->processDataNum);
    Mul(p3, p3, p2, this->processDataNum);
    Adds(p3, p3, TAYLOR[4], this->processDataNum);
    Mul(p3, p3, p2, this->processDataNum);
    Adds(p3, p3, TAYLOR[3], this->processDataNum);
    Mul(p3, p3, p2, this->processDataNum);
    Adds(p3, p3, TAYLOR[2], this->processDataNum);
    Mul(p3, p3, p2, this->processDataNum);
    Adds(p3, p3, TAYLOR[1], this->processDataNum);
    Mul(p3, p3, p2, this->processDataNum);
    Adds(p3, p3, TAYLOR[0], this->processDataNum);
    Mul(p3, p3, p4, this->processDataNum);
    Adds(p4, p3, CONST_PI_BY_EIGHT, this->processDataNum);  

    Mul(p2, p1, p1, this->processDataNum);  
    Duplicate(p3, static_cast<float>(TAYLOR[4]), this->processDataNum);
    Mul(p3, p3, p2, this->processDataNum);
    Adds(p3, p3, TAYLOR[3], this->processDataNum);
    Mul(p3, p3, p2, this->processDataNum);
    Adds(p3, p3, TAYLOR[2], this->processDataNum);
    Mul(p3, p3, p2, this->processDataNum);
    Adds(p3, p3, TAYLOR[1], this->processDataNum);
    Mul(p3, p3, p2, this->processDataNum);
    Adds(p3, p3, TAYLOR[0], this->processDataNum);
    Mul(p3, p3, p1, this->processDataNum);   

    Min(p1, p4, p3, this->processDataNum);   
}

template <typename TYPE_X, typename TYPE_Y, uint64_t BUFFER_NUM>
__aicore__ inline void KernelAtan2<TYPE_X, TYPE_Y, BUFFER_NUM>::AtanCompute(LocalTensor<float>& p1, LocalTensor<float>& p2, LocalTensor<float>& p3, LocalTensor<float>& p4, LocalTensor<float>& sign, LocalTensor<float>& xLocal)
{
    float  max_input_value = 10000.0f;
    float  min_input_value = -max_input_value;
    Mins(p1, xLocal, max_input_value, this->processDataNum);
    Maxs(xLocal, p1, min_input_value,  this->processDataNum);

    Abs(p1, xLocal, this->processDataNum);         //p1 = abs_data
    Div(sign, xLocal, p1,  this->processDataNum);  //sign = xLocal / abs_data

    Duplicate(p2, CONST_ONE, this->processDataNum);

    Sub(p3, p1, p2, this->processDataNum);  
    Add(p4, p1, p2, this->processDataNum);  
    Div(xLocal, p3, p4, this->processDataNum);        
    Abs(xLocal, xLocal, this->processDataNum); //xLocal = abs_data2

    DoTaylor(p1,p2,p3,p4); 
    DoTaylor(xLocal,p2,p3,p4);

    Adds(xLocal, xLocal, CONST_PI_BY_FOUR, this->processDataNum);
    Min(p1, p1, xLocal, this->processDataNum);
    Mul(xLocal, p1, sign, this->processDataNum);
}

template <typename TYPE_X, typename TYPE_Y, uint64_t BUFFER_NUM>
__aicore__ inline void KernelAtan2<TYPE_X, TYPE_Y, BUFFER_NUM>::DoCompute(LocalTensor<float>& x1Local, LocalTensor<float>& x2Local,
                                                                                      LocalTensor<float>& yLocal)
{
    LocalTensor<float> p1 = calcBuf.Get<float>();
    LocalTensor<float> p2 = p1[this->tileDataNum];
    LocalTensor<float> p3 = p2[this->tileDataNum];
    LocalTensor<float> p4 = p3[this->tileDataNum];
    LocalTensor<float> p5 = p4[this->tileDataNum];
    LocalTensor<float> p6 = p5[this->tileDataNum];
    LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
    LocalTensor<uint8_t> maskX2Inf = maskLocal[this->tileDataNum];
    LocalTensor<uint8_t> maskX2NeInf = maskX2Inf[this->tileDataNum];
    LocalTensor<uint8_t> maskX2LtInf = maskX2NeInf[this->tileDataNum];
    LocalTensor<float>& oneLocal = p1;

    Div(yLocal, x1Local, x2Local, this->processDataNum); // 3 块固定数据

    AtanCompute(p1, p2, p3, p4, p5, yLocal); // 额外5块+1块mask
    
    Duplicate(oneLocal, CONST_ONE, this->processDataNum);
    {
        LocalTensor<float>& x1CmpZero = p2;
        LocalTensor<float>& x2LtZero = p3;
        // maskLocal zeroLocal oneLocal x2CmpZero x1LtZero 复用之前的数据空间  4块+1块mask
        /*
        _init_atan2_mask
        [x1LtZero, x2CmpZero]
        */
        CompareScalar(maskLocal, x1Local, CONST_ZERO, CMPMODE::GE, this->processDataNum);
        Select(x1CmpZero, maskLocal, oneLocal, NE_CONST_ONE, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);

        CompareScalar(maskLocal, x2Local, CONST_ZERO, CMPMODE::LT, this->processDataNum);
        Select(x2LtZero, maskLocal, x1CmpZero, CONST_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);

        /*
        y_cmp_zero = tbe.vmuls(mask[const_one], tvm.const(const_pi_by_two, y.dtype))
        res_x_lt_zero = tbe.vmuls(mask[Constant.CONST_ZERO], tvm.const(const_pi, y.dtype))
        */
        Muls(x1CmpZero, x1CmpZero, CONST_PI_BY_TWO, this->processDataNum);
        Muls(x2LtZero, x2LtZero, CONST_PI, this->processDataNum);

        /*
        tensor_zero = tbe.broadcast(tvm.const(Constant.CONST_ZERO, x.dtype), shape_broadcast)
        x_equal_zero = tbe.vcmp(x, tensor_zero, 'eq')
        res = tbe.vsel(x_equal_zero, y_cmp_zero, res)
        */
        CompareScalar(maskLocal, x2Local, CONST_ZERO, CMPMODE::EQ, this->processDataNum);
        Select(yLocal, maskLocal, x1CmpZero, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);

        /*
        res = tbe.vadd(res, res_x_lt_zero)
        */
        Add(yLocal, yLocal, x2LtZero, this->processDataNum);
    }
    
    /*
    res = corner_case_post_process(y, x, res)
    */
    {   // 只剩下x1Local、x2Local、yLocal里有用到的数据，其余的均可释放复用
        // zeroLocal oneLocal maskLocal
        // 固定3+3个 + 最多
        // x2ZeroIndex = x1SignbitOne = negZeroLocal = x2InfX1Inf =
        // piByFourLocal = x2NeInfXInf = negPiByFourLocal = x2InfXNeInf = piByThreeQuartersLocal = x2NeInfXNeInf
        // negPiByThreeQuartersLocal = x1AbsLocal = x1InfX2LtInf = piByTwoLocal = x2NeInfX1LtInf = negPiByTwoLocal
        LocalTensor<float>& x1ZeroIndex = p2;
        LocalTensor<float>& x1PosZero = p2;
        LocalTensor<float>& negX1 = p2;
        LocalTensor<float>& negX1GtInf = p2;
        LocalTensor<float>& negX1GtInfX2Inf = p2;
        LocalTensor<float>& x1Inf = p2;
        LocalTensor<float>& x2InfX1LtInf = p2;
        LocalTensor<float>& x1NeInfX2LtInf = p2;

        LocalTensor<int32_t> signLocalInt32 = p3.template ReinterpretCast<int32_t>();
        LocalTensor<int16_t> signLocalInt16 = p3.template ReinterpretCast<int16_t>();
        LocalTensor<int16_t> x1LocalInt16 = x1Local.template ReinterpretCast<int16_t>();
        LocalTensor<int16_t> x2LocalInt16 = x2Local.template ReinterpretCast<int16_t>();
        LocalTensor<float>& x1PosZeroX2SignOne = p3;
        LocalTensor<float>& x1PosZeroX2SignZero = p3;
        LocalTensor<float>& x1NegZeroX2SignOne = p3;
        LocalTensor<float>& x1NegZeroX2SignZero = p3;

        LocalTensor<float>& x1Signbit = p4;
        LocalTensor<float>& x1NegZero = p4;
        LocalTensor<float>& x1NeInf = p4;

        LocalTensor<int32_t> tmpLocalInt32 = p5.template ReinterpretCast<int32_t>();
        LocalTensor<int16_t> tmpLocalInt16 = p5.template ReinterpretCast<int16_t>();
        LocalTensor<float>& x2SignbitOne = p5;
        LocalTensor<float>& x2AbsLocal = p5;

        LocalTensor<float>& x1InfX2Inf = p6;
        LocalTensor<float>& x1NeInfX2Inf = p6;
        LocalTensor<float>& x1InfX2NeInf = p6;
        LocalTensor<float>& x1NeInfX2NeInf = p6;

        LocalTensor<float>& x2SignbitZero = p6;

        /*
        y_zero_temp = tbe.vcmp(y, tensor_zero, "eq")
        y_zero_index = tbe.vsel(y_zero_temp, tensor_one, tensor_zero)
        */
        CompareScalar(maskLocal, x1Local, CONST_ZERO, CMPMODE::EQ, this->processDataNum);
        Select(x1ZeroIndex, maskLocal, oneLocal, CONST_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum); 
        /*
        y_signbit_one = tbe.vsignbit(y)
        */
        Duplicate(signLocalInt32, CONST_HIGH_ONE, this->processDataNum);

        And(tmpLocalInt16, x1LocalInt16, signLocalInt16, this->processDataNum + this->processDataNum);
        CompareScalar(maskLocal, tmpLocalInt32, CONST_HIGH_ONE, CMPMODE::EQ, this->processDataNum);
        Select(x1Signbit, maskLocal, oneLocal, CONST_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        // x2Signbit 可以复用tmpLocalInt32
        /*
        y_negative_zero = tbe.vmul(y_signbit_one, y_zero_index)
        y_positive_zero = tbe.vsub(y_zero_index, y_negative_zero)
        */
        Mul(x1NegZero, x1Signbit, x1ZeroIndex, this->processDataNum);
        Sub(x1PosZero, x1ZeroIndex, x1NegZero, this->processDataNum);

        /*
        x_signbit_one = tbe.vsignbit(x)
        */
        And(tmpLocalInt16, x2LocalInt16, signLocalInt16, this->processDataNum + this->processDataNum);
        CompareScalar(maskLocal, tmpLocalInt32, CONST_HIGH_ONE, CMPMODE::EQ, this->processDataNum);
        Select(x2SignbitOne, maskLocal, oneLocal, CONST_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        /*
        x_signbit_zero = tbe.vsub(tensor_one, x_signbit_one)
        */
        Sub(x2SignbitZero, oneLocal, x2SignbitOne, this->processDataNum);

        /*
        y_pos_zero_x_sign_one = tbe.vmul(y_positive_zero, x_signbit_one)
        res_temp = tbe.vcmp(y_pos_zero_x_sign_one, tensor_one, "eq")
        res = tbe.vsel(res_temp, const_pi, res)
        */
        Mul(x1PosZeroX2SignOne, x1PosZero, x2SignbitOne, this->processDataNum);
        CompareScalar(maskLocal, x1PosZeroX2SignOne, CONST_ONE, CMPMODE::NE, this->processDataNum);
        Select(yLocal, maskLocal, yLocal, CONST_PI, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);

        /*
        y_pos_zero_x_sign_zero = tbe.vmul(y_positive_zero, x_signbit_zero)
        res_temp = tbe.vcmp(y_pos_zero_x_sign_zero, tensor_one, "eq")
        res = tbe.vsel(res_temp, tensor_zero, res)
        */
        Mul(x1PosZeroX2SignZero, x1PosZero, x2SignbitZero, this->processDataNum);
        CompareScalar(maskLocal, x1PosZeroX2SignZero, CONST_ONE, CMPMODE::NE, this->processDataNum);
        Select(yLocal, maskLocal, yLocal, CONST_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);

        /*
        y_neg_zero_x_sign_one = tbe.vmul(y_negative_zero, x_signbit_one)
        res_temp = tbe.vcmp(y_neg_zero_x_sign_one, tensor_one, "eq")
        res = tbe.vsel(res_temp, ne_const_pi, res)
        */
        Mul(x1NegZeroX2SignOne, x1NegZero, x2SignbitOne, this->processDataNum);
        CompareScalar(maskLocal, x1NegZeroX2SignOne, CONST_ONE, CMPMODE::NE, this->processDataNum);
        Select(yLocal, maskLocal, yLocal, NE_CONST_PI, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);

        /*
        y_neg_zero_x_sign_zero = tbe.vmul(y_negative_zero, x_signbit_zero)
        res_temp = tbe.vcmp(y_neg_zero_x_sign_zero, tensor_one, "eq")
        res = tbe.vsel(res_temp, tvm.const(-0.0, xdtype), res)
        */
        Mul(x1NegZeroX2SignZero, x1NegZero, x2SignbitZero, this->processDataNum);
        CompareScalar(maskLocal, x1NegZeroX2SignZero, CONST_ONE, CMPMODE::NE, this->processDataNum);
        Select(yLocal, maskLocal, yLocal, NE_CONST_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);

        /*x1PosZeroX2SignOne, x1PosZeroX2SignZero, x1NegZeroX2SignOne, x1NegZeroX2SignZero 可以共用*/

        /*
        neg_y_mask = tbe.vcmp(y, tensor_zero, "lt")
        neg_y = tbe.vsel(neg_y_mask, tensor_one, tensor_zero)
        */
        CompareScalar(maskLocal, x1Local, CONST_ZERO, CMPMODE::LT, this->processDataNum);
        Select(negX1, maskLocal, oneLocal, CONST_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);

        /*
        neg_y_mask_2 = tbe.vcmp(y, tensor_ne_inf, "gt")
        neg_y_gt_inf = tbe.vsel(neg_y_mask_2, neg_y, tensor_zero)
        */
        CompareScalar(maskLocal, x1Local, NE_CONST_INF, CMPMODE::GT, this->processDataNum);
        Select(negX1GtInf, maskLocal, negX1, CONST_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);

        /*
        x_inf = tbe.vcmp(x, tensor_inf, "eq")
        neg_y_gt_inf_x_inf = tbe.vsel(x_inf, neg_y_gt_inf, tensor_zero)
        */
        CompareScalar(maskX2Inf, x2Local, CONST_INF, CMPMODE::EQ, this->processDataNum);
        Select(negX1GtInfX2Inf, maskX2Inf, negX1GtInf, CONST_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);

        /*
        res_temp = tbe.vcmp(neg_y_gt_inf_x_inf, tensor_one, "eq")
        res = tbe.vsel(res_temp, tvm.const(-0.0, xdtype), res)
        */
        CompareScalar(maskLocal, negX1GtInfX2Inf, CONST_ONE, CMPMODE::NE, this->processDataNum);
        Select(yLocal, maskLocal, yLocal, NE_CONST_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);


        /*
        y_inf_mask = tbe.vcmp(y, tensor_inf, "eq")
        y_inf = tbe.vsel(y_inf_mask, tensor_one, tensor_zero)
        */
        CompareScalar(maskLocal, x1Local, CONST_INF, CMPMODE::EQ, this->processDataNum);
        Select(x1Inf, maskLocal, oneLocal, CONST_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);

        /*
        y_ne_inf_mask = tbe.vcmp(y, tensor_ne_inf, "eq")
        y_ne_inf = tbe.vsel(y_ne_inf_mask, tensor_one, tensor_zero)
        */
        CompareScalar(maskLocal, x1Local, NE_CONST_INF, CMPMODE::EQ, this->processDataNum);
        Select(x1NeInf, maskLocal, oneLocal, CONST_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);


        /*
        y_inf_x_inf = tbe.vsel(x_inf, y_inf, tensor_zero)
        res_temp = tbe.vcmp(y_inf_x_inf, tensor_one, "eq")
        res = tbe.vsel(res_temp, const_pi_by_four, res)
        */
        Select(x1InfX2Inf, maskX2Inf, x1Inf, CONST_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        CompareScalar(maskLocal, x1InfX2Inf, CONST_ONE, CMPMODE::NE, this->processDataNum);
        Select(yLocal, maskLocal, yLocal, CONST_PI_BY_FOUR, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);

        
        /*
        y_ne_inf_x_inf = tbe.vsel(x_inf, y_ne_inf, tensor_zero)
        res_temp = tbe.vcmp(y_ne_inf_x_inf, tensor_one, "eq")
        res = tbe.vsel(res_temp, ne_const_pi_by_four, res)
        */
        Select(x1NeInfX2Inf, maskX2Inf, x1NeInf, CONST_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        CompareScalar(maskLocal, x1NeInfX2Inf, CONST_ONE, CMPMODE::NE, this->processDataNum);
        Select(yLocal, maskLocal, yLocal, NE_CONST_PI_BY_FOUR, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);


        /*
        x_ne_inf = tbe.vcmp(x, tensor_ne_inf, "eq")
        y_inf_x_ne_inf = tbe.vsel(x_ne_inf, y_inf, tensor_zero)
        */
        CompareScalar(maskX2NeInf, x2Local, NE_CONST_INF, CMPMODE::EQ, this->processDataNum);
        Select(x1InfX2NeInf, maskX2NeInf, x1Inf, CONST_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);

        /*
        res_temp = tbe.vcmp(y_inf_x_ne_inf, tensor_one, "eq")
        res = tbe.vsel(res_temp, const_pi_by_three_quarters, res)
        */
        CompareScalar(maskLocal, x1InfX2NeInf, CONST_ONE, CMPMODE::NE, this->processDataNum);
        Select(yLocal, maskLocal, yLocal, CONST_PI_BY_THREE_QUARTERS, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);


        /*
        y_ne_inf_x_ne_inf = tbe.vsel(x_ne_inf, y_ne_inf, tensor_zero)
        res_temp = tbe.vcmp(y_ne_inf_x_ne_inf, tensor_one, "eq")
        res = tbe.vsel(res_temp, ne_const_pi_by_three_quarters, res)
        */
        Select(x1NeInfX2NeInf, maskX2NeInf, x1NeInf, CONST_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        CompareScalar(maskLocal, x1NeInfX2NeInf, CONST_ONE, CMPMODE::NE, this->processDataNum);
        Select(yLocal, maskLocal, yLocal, NE_CONST_PI_BY_THREE_QUARTERS, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);

        
        /*
        x_abs = tbe.vabs(x)
        x_lt_inf = tbe.vcmp(x_abs, tensor_inf, "lt")
        */
        Abs(x2AbsLocal, x2Local, this->processDataNum);
        CompareScalar(maskX2LtInf, x2AbsLocal, CONST_INF, CMPMODE::LT, this->processDataNum);

        /*
        y_inf_x_lt_inf = tbe.vsel(x_lt_inf, y_inf, tensor_zero)
        res_temp = tbe.vcmp(y_inf_x_lt_inf, tensor_one, "eq")
        res = tbe.vsel(res_temp, const_pi_by_two, res)
        */
        Select(x2InfX1LtInf, maskX2LtInf, x1Inf, CONST_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        CompareScalar(maskLocal, x2InfX1LtInf, CONST_ONE, CMPMODE::NE, this->processDataNum);
        Select(yLocal, maskLocal, yLocal, CONST_PI_BY_TWO, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);

        
        /*
        y_ne_inf_x_lt_inf = tbe.vsel(x_lt_inf, y_ne_inf, tensor_zero)
        res_temp = tbe.vcmp(y_ne_inf_x_lt_inf, tensor_one, "eq")
        res = tbe.vsel(res_temp, ne_const_pi_by_two, res)
        */
        Select(x1NeInfX2LtInf, maskX2LtInf, x1NeInf, CONST_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        CompareScalar(maskLocal, x1NeInfX2LtInf, CONST_ONE, CMPMODE::NE, this->processDataNum);
        Select(yLocal, maskLocal, yLocal, NE_CONST_PI_BY_TWO, SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
    }

    Compare(maskLocal, x1Local, x1Local, CMPMODE::EQ, this->processDataNum);
    Select(yLocal, maskLocal, yLocal, x1Local, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
    Compare(maskLocal, x2Local, x2Local, CMPMODE::EQ, this->processDataNum);
    Select(yLocal, maskLocal, yLocal, x2Local, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
}

template <typename TYPE_X, typename TYPE_Y, uint64_t BUFFER_NUM>
__aicore__ inline void KernelAtan2<TYPE_X, TYPE_Y, BUFFER_NUM>::CopyInAndCompute(int32_t offset)
{    
    LocalTensor<TYPE_X> x1Local = inQueue.template AllocTensor<TYPE_X>();
    LocalTensor<TYPE_X> x2Local = x1Local[this->processDataNum];
    LocalTensor<TYPE_Y> yLocal = outQueueY.template AllocTensor<TYPE_Y>();
    int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    if constexpr ( IsSameType<TYPE_X, bfloat16_t>::value || IsSameType<TYPE_X, half>::value)
    {
        CopyInAndCompute16B(x1Local, x2Local, yLocal, eventIDMTE2ToV, offset);
    }
    else if constexpr ( IsSameType<TYPE_X, float>::value)
    {
        DataCopy(x1Local, x1Gm[offset], this->processDataNum);
        DataCopy(x2Local, x2Gm[offset], this->processDataNum);
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        DoCompute(x1Local, x2Local, yLocal);
    }
    outQueueY.template EnQue<TYPE_Y>(yLocal);
    inQueue.template FreeTensor(x1Local);
}

template <typename TYPE_X, typename TYPE_Y, uint64_t BUFFER_NUM>
__aicore__ inline void KernelAtan2<TYPE_X, TYPE_Y, BUFFER_NUM>::CopyOut(int32_t offset)
{
    LocalTensor<TYPE_Y> yLocal = outQueueY.template DeQue<TYPE_Y>();
    DataCopy(yGm[offset], yLocal, this->processDataNum);
    outQueueY.template FreeTensor(yLocal);
}

template <typename TYPE_X, typename TYPE_Y, uint64_t BUFFER_NUM>
__aicore__ inline void KernelAtan2<TYPE_X, TYPE_Y, BUFFER_NUM>::Process()
{
    int32_t loopCount = this->tileNum - 1;
    this->processDataNum = this->tileDataNum;
    int32_t offset = 0;
    for (int32_t i = 0; i < loopCount; i++, offset+=this->tileDataNum)
    {
        CopyInAndCompute(offset);
        CopyOut(offset);
    }
    this->processDataNum = this->tailDataNum;
    CopyInAndCompute(offset);
    CopyOut(offset);
}

} // namespace KernelAtan2
#endif // ATAN2_H