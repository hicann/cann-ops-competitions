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
 * \file nan_to_num.h
 * \brief Kernel implementation for nan_to_num operator, which replaces NaN, positive infinity,
 *        and negative infinity values with specified replacement values.
 */
#ifndef NAN_TO_NUM_H_
#define NAN_TO_NUM_H_

#include <cmath>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "nan_to_num_tiling_data.h"
#include "nan_to_num_tiling_key.h"

namespace NsNanToNum {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X>
class KernelNanToNum {
public:
    __aicore__ inline KernelNanToNum() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const NanToNumTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<TPosition::VECCALC> tmpQueue;
    TBuf<TPosition::VECCALC> tmpQueueMask;
    TBuf<TPosition::VECCALC> tmpQueue0;

    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<TYPE_X> yGm;
    uint64_t coreDataNum = 0;
    uint64_t tileNum = 0;
    uint64_t tileDataNum = 0;
    uint64_t tailDataNum = 0;
    uint64_t processDataNum = 0;
    float nanValue = 0.0f;
    float posinf = 0.0f;
    float neginf = 0.0f;
};

template <typename TYPE_X>
__aicore__ inline void KernelNanToNum<TYPE_X>::Init(GM_ADDR x, GM_ADDR y, const NanToNumTilingData* tilingData)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = GetBlockIdx();
    uint64_t globalBufferIndex = tilingData->bigCoreDataNum * coreId;
    this->tileDataNum = tilingData->tileDataNum;
    this->nanValue = tilingData->nanValue;
    this->posinf = tilingData->posinf;
    this->neginf = tilingData->neginf;
    if (coreId < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (coreId - tilingData->tailBlockNum);
    }
    xGm.SetGlobalBuffer((__gm__ TYPE_X *)x + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ TYPE_X *)y + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));

    if constexpr (std::is_same_v<TYPE_X, bfloat16_t>) {
        pipe.InitBuffer(tmpQueue, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpQueue0, this->tileDataNum * sizeof(float));
    }
    pipe.InitBuffer(tmpQueueMask, this->tileDataNum * sizeof(uint8_t));
}

template <typename TYPE_X>
__aicore__ inline void KernelNanToNum<TYPE_X>::CopyIn(int32_t progress)
{
    LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
    DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    inQueueX.EnQue(xLocal);
}

template <typename TYPE_X>
__aicore__ inline void KernelNanToNum<TYPE_X>::CopyOut(int32_t progress)
{
    LocalTensor<TYPE_X> yLocal = outQueueY.DeQue<TYPE_X>();
    DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
    outQueueY.FreeTensor(yLocal);
}

template <typename TYPE_X>
__aicore__ inline void KernelNanToNum<TYPE_X>::Compute(int32_t progress)
{
    LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
    if constexpr (!std::is_same_v<TYPE_X, bfloat16_t>) {
        LocalTensor<TYPE_X> yLocal = outQueueY.AllocTensor<TYPE_X>();
        LocalTensor<uint8_t> mask1Local = tmpQueueMask.AllocTensor<uint8_t>();
        Compare(mask1Local, xLocal, xLocal, CMPMODE::EQ, this->processDataNum);
        Select(yLocal, mask1Local, xLocal, static_cast<TYPE_X>(this->nanValue), SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        Duplicate(xLocal, static_cast<TYPE_X>(INFINITY), this->processDataNum);
        Compare(mask1Local, yLocal, xLocal, CMPMODE::EQ, this->processDataNum);
        Duplicate(xLocal, static_cast<TYPE_X>(this->posinf), this->processDataNum);
        Select(yLocal, mask1Local, xLocal, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        Duplicate(xLocal, static_cast<TYPE_X>(-INFINITY), this->processDataNum);
        Compare(mask1Local, yLocal, xLocal, CMPMODE::EQ, this->processDataNum);
        Duplicate(xLocal, static_cast<TYPE_X>(this->neginf), this->processDataNum);
        Select(yLocal, mask1Local, xLocal, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        outQueueY.EnQue<TYPE_X>(yLocal);
    } else {
        LocalTensor<TYPE_X> yLocal = outQueueY.AllocTensor<TYPE_X>();
        LocalTensor<uint8_t> mask1Local = tmpQueueMask.AllocTensor<uint8_t>();
        LocalTensor<float> tmpLocal = tmpQueue.AllocTensor<float>();
        LocalTensor<float> tmp0Local = tmpQueue0.AllocTensor<float>();
        Cast(tmpLocal, xLocal, RoundMode::CAST_NONE, this->processDataNum);
        Compare(mask1Local, tmpLocal, tmpLocal, CMPMODE::EQ, this->processDataNum);
        Select(tmpLocal, mask1Local, tmpLocal, static_cast<float>(this->nanValue), SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        Duplicate(tmp0Local, static_cast<float>(INFINITY), this->processDataNum);
        Compare(mask1Local, tmpLocal, tmp0Local, CMPMODE::EQ, this->processDataNum);
        Duplicate(tmp0Local, static_cast<float>(this->posinf), this->processDataNum);
        Select(tmpLocal, mask1Local, tmp0Local, tmpLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        Duplicate(tmp0Local, static_cast<float>(-INFINITY), this->processDataNum);
        Compare(mask1Local, tmpLocal, tmp0Local, CMPMODE::EQ, this->processDataNum);
        Duplicate(tmp0Local, static_cast<float>(this->neginf), this->processDataNum);
        Select(tmpLocal, mask1Local, tmp0Local, tmpLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        Cast(yLocal, tmpLocal, RoundMode::CAST_RINT, this->processDataNum);
        outQueueY.EnQue<TYPE_X>(yLocal);
    }

    inQueueX.FreeTensor(xLocal);
}

template <typename TYPE_X>
__aicore__ inline void KernelNanToNum<TYPE_X>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount - 1; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    CopyIn(loopCount - 1);
    Compute(loopCount - 1);
    CopyOut(loopCount - 1);
}

} // namespace NsNanToNum
#endif // NAN_TO_NUM_H