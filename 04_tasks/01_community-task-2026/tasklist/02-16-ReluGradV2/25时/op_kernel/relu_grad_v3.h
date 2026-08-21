/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file relu_grad_v3.h
 * \brief ReluGradV3 算子 AscendC 实现
 */
#ifndef __RELU_GRAD_V3_H__
#define __RELU_GRAD_V3_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_grad_v3_tiling_data.h"
#include "relu_grad_v3_tiling_key.h"

namespace NsReluGradV3 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t DATA_ALIGN_BYTES = 32;
constexpr int32_t CMP_ALIGN_BYTES = 256;                     // Compares/Select 要求 256B 对齐
constexpr uint32_t SELECT_INTERNAL_KB = 8U * 1024U;           // Select 模式1/2 额外占用 8KB UB
constexpr uint32_t CMP_ALIGN_ELEMS = 256U / sizeof(half);     // Compares 256B 对齐 = 128 个 half

#define ALIGN256(size) (((size) + CMP_ALIGN_BYTES - 1U) / CMP_ALIGN_BYTES * CMP_ALIGN_BYTES)

template <typename T>
class ReluGradV3 {
public:
    __aicore__ inline ReluGradV3(){};

    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR mask, GM_ADDR grad_x, const ReluGradV3TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueGrad;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueMask;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    // ---- half/float 大 tile 临时 buffer (Compares+Select 路径) ----
    TBuf<QuePosition::VECCALC> tmpBufMaskHalf;     // uint8 → half 转换
    TBuf<QuePosition::VECCALC> tmpBufCmpMask;      // Compares 输出的 packed-bit mask (N/8 字节)
    TBuf<QuePosition::VECCALC> tmpBufSelectInternal; // Select 内部 8KB 保留区

    // ---- 其他 dtype + half/float 小 tile 回退 buffer (Mul+Cast 路径) ----
    TBuf<QuePosition::VECCALC> tmpBufGradCast;     // grad → float/half 转换
    TBuf<QuePosition::VECCALC> tmpBufOutCast;      // maskFloat / resultHalf
    TBuf<QuePosition::VECCALC> tmpBufResultFloat;  // bf16/int32 Mul 独立输出（避免 dst/src 重叠）

    GlobalTensor<T> gradGM;
    GlobalTensor<uint8_t> maskGM;
    GlobalTensor<T> outputGMY;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <typename T>
__aicore__ inline void ReluGradV3<T>::Init(GM_ADDR grad, GM_ADDR mask, GM_ADDR grad_x,
                                           const ReluGradV3TilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalElementIndex = 0;

    uint64_t bigCoreDataNum = tilingData->bigCoreDataNum;
    uint64_t smallCoreDataNum = tilingData->smallCoreDataNum;
    uint64_t tailBlockNum = tilingData->tailBlockNum;

    this->tileDataNum = static_cast<uint32_t>(tilingData->tileDataNum);

    globalElementIndex = bigCoreDataNum * coreId;

    if (coreId < tailBlockNum) {
        this->coreDataNum = static_cast<uint32_t>(bigCoreDataNum);
        this->tileNum = static_cast<uint32_t>(tilingData->finalBigTileNum);
        this->tailDataNum = static_cast<uint32_t>(tilingData->bigTailDataNum);
    } else {
        this->coreDataNum = static_cast<uint32_t>(smallCoreDataNum);
        this->tileNum = static_cast<uint32_t>(tilingData->finalSmallTileNum);
        this->tailDataNum = static_cast<uint32_t>(tilingData->smallTailDataNum);

        globalElementIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreId - tailBlockNum);
    }

    gradGM.SetGlobalBuffer((__gm__ T*)grad + globalElementIndex, this->coreDataNum);
    maskGM.SetGlobalBuffer((__gm__ uint8_t*)mask + globalElementIndex, this->coreDataNum);
    outputGMY.SetGlobalBuffer((__gm__ T*)grad_x + globalElementIndex, this->coreDataNum);

    // half/float 需要 256B 对齐（Compares/Select 约束），其他 dtype 不需要
    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float>) {
        pipe.InitBuffer(inputQueueGrad, BUFFER_NUM, ALIGN256(this->tileDataNum * sizeof(T)));
        pipe.InitBuffer(inputQueueMask, BUFFER_NUM, ALIGN256(this->tileDataNum * sizeof(uint8_t) + DATA_ALIGN_BYTES));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, ALIGN256(this->tileDataNum * sizeof(T)));
    } else {
        pipe.InitBuffer(inputQueueGrad, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(inputQueueMask, BUFFER_NUM, this->tileDataNum * sizeof(uint8_t) + DATA_ALIGN_BYTES);
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileDataNum * sizeof(T));
    }

    // 临时 buffer 分配:
    //   half/float 大 tile: maskHalf + cmpMask + Select 8KB
    //   half/float 大 tile: maskHalf + cmpMask + Select; 小 tile: Mul 回退
    //   其他 dtype:         maskHalf + gradCast + outCast
    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>) {
        pipe.InitBuffer(tmpBufMaskHalf, ALIGN256(this->tileDataNum * sizeof(half) + DATA_ALIGN_BYTES));
        uint32_t maxCmpDataNum = (this->tileDataNum > this->tailDataNum) ? this->tileDataNum : this->tailDataNum;
        uint32_t maxCmpCount = ((maxCmpDataNum + CMP_ALIGN_ELEMS - 1U) / CMP_ALIGN_ELEMS) * CMP_ALIGN_ELEMS;
        uint32_t cmpSize = (maxCmpCount / 8U > 32U) ? ALIGN256(maxCmpCount / 8U + DATA_ALIGN_BYTES) : ALIGN256(32U + DATA_ALIGN_BYTES);

        pipe.InitBuffer(tmpBufCmpMask, cmpSize);
        pipe.InitBuffer(tmpBufSelectInternal, SELECT_INTERNAL_KB);  // Select 内部 8KB 保留区
        // bf16: Select/grad→float 需要 gradCast+outCast(float), 小 tile Mul 回退额外需要 resultFloat
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            pipe.InitBuffer(tmpBufGradCast, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmpBufOutCast,  this->tileDataNum * sizeof(float));
            if (this->tileDataNum < CMP_ALIGN_ELEMS) {
                pipe.InitBuffer(tmpBufResultFloat, this->tileDataNum * sizeof(float));
            }
        } else {
        if (this->tileDataNum < CMP_ALIGN_ELEMS) {
            pipe.InitBuffer(tmpBufOutCast, this->tileDataNum * sizeof(float));
        }
        }
    } else {
        pipe.InitBuffer(tmpBufMaskHalf, this->tileDataNum * sizeof(half) + DATA_ALIGN_BYTES);
        pipe.InitBuffer(tmpBufCmpMask, 32U + DATA_ALIGN_BYTES);    // 不使用
        // int8/uint8 只用 half 精度，无需 float buffer
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            pipe.InitBuffer(tmpBufGradCast, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(tmpBufOutCast,  this->tileDataNum * sizeof(half));
        } else {
            pipe.InitBuffer(tmpBufGradCast, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmpBufOutCast,  this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmpBufResultFloat, this->tileDataNum * sizeof(float));
        }
    }
}

template <typename T>
__aicore__ inline void ReluGradV3<T>::CopyIn(int32_t progress)
{
    this->processDataNum = (progress == this->tileNum - 1) ? this->tailDataNum : this->tileDataNum;
    uint32_t tileStartIdx = progress * this->tileDataNum;

    AscendC::LocalTensor<T> gradLocal = inputQueueGrad.AllocTensor<T>();
    {
        uint32_t gradAlignNum = DATA_ALIGN_BYTES / static_cast<uint32_t>(sizeof(T));
        uint32_t gradPaddedNum = ((this->processDataNum + gradAlignNum - 1) / gradAlignNum) * gradAlignNum;

        AscendC::DataCopyExtParams gradCopyParams{1, static_cast<uint32_t>(this->processDataNum * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> gradPadParams{true, 0,
            static_cast<uint8_t>(gradPaddedNum - this->processDataNum), static_cast<T>(0)};
        AscendC::DataCopyPad(gradLocal, gradGM[tileStartIdx], gradCopyParams, gradPadParams);
    }
    inputQueueGrad.EnQue(gradLocal);

    AscendC::LocalTensor<uint8_t> maskLocal = inputQueueMask.AllocTensor<uint8_t>();
    {
        uint32_t maskPaddedNum = ((this->processDataNum + DATA_ALIGN_BYTES - 1) / DATA_ALIGN_BYTES) * DATA_ALIGN_BYTES;

        AscendC::DataCopyExtParams maskCopyParams{1, static_cast<uint32_t>(this->processDataNum * sizeof(uint8_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint8_t> maskPadParams{true, 0,
            static_cast<uint8_t>(maskPaddedNum - this->processDataNum), static_cast<uint8_t>(0)};
        AscendC::DataCopyPad(maskLocal, maskGM[tileStartIdx], maskCopyParams, maskPadParams);
    }
    inputQueueMask.EnQue(maskLocal);
}

template <typename T>
__aicore__ inline void ReluGradV3<T>::CopyOut(int32_t progress)
{
    uint32_t tileStartIdx = progress * this->tileDataNum;
    AscendC::LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    {
        AscendC::DataCopyExtParams outCopyParams{1, static_cast<uint32_t>(this->processDataNum * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(outputGMY[tileStartIdx], yLocal, outCopyParams);
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void ReluGradV3<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> gradLocal = inputQueueGrad.DeQue<T>();
    AscendC::LocalTensor<uint8_t> maskLocal = inputQueueMask.DeQue<uint8_t>();
    AscendC::LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    // ================================================================
    // half/float: small tile Mul+Adds, large tile Compares+Select
    // ================================================================
    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float>) {
        if (this->tileDataNum < CMP_ALIGN_ELEMS) {
            LocalTensor<half> maskHalf = tmpBufMaskHalf.Get<half>();
            Cast(maskHalf, maskLocal, RoundMode::CAST_NONE, this->processDataNum);
            if constexpr (std::is_same_v<T, half>) {
                Mul(yLocal, gradLocal, maskHalf, this->processDataNum);
                Adds(yLocal, yLocal, static_cast<half>(0), this->processDataNum);
            } else {
                LocalTensor<float> maskFloat = tmpBufOutCast.Get<float>();
                Cast(maskFloat, maskHalf, RoundMode::CAST_NONE, this->processDataNum);
                Mul(yLocal, gradLocal, maskFloat, this->processDataNum);
                Adds(yLocal, yLocal, 0.0f, this->processDataNum);
            }
        } else {
            LocalTensor<half> maskHalf = tmpBufMaskHalf.Get<half>();
            Cast(maskHalf, maskLocal, RoundMode::CAST_NONE, this->processDataNum);
            uint32_t cmpCount = ((this->processDataNum + CMP_ALIGN_ELEMS - 1U) / CMP_ALIGN_ELEMS) * CMP_ALIGN_ELEMS;
            LocalTensor<uint8_t> cmpMask = tmpBufCmpMask.Get<uint8_t>();
            Compares(cmpMask, maskHalf, static_cast<half>(0), CMPMODE::NE, cmpCount);
            Select(yLocal, cmpMask, gradLocal, static_cast<T>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, cmpCount);
        }
    }
    // int8 路径: grad→half, mask→half, Mul(half), Cast→int8
    // ================================================================
    else if constexpr (std::is_same_v<T, int8_t>) {
        LocalTensor<half> gradHalf = tmpBufGradCast.Get<half>();
        Cast(gradHalf, gradLocal, RoundMode::CAST_NONE, this->processDataNum);

        LocalTensor<half> maskHalf = tmpBufMaskHalf.Get<half>();
        Cast(maskHalf, maskLocal, RoundMode::CAST_NONE, this->processDataNum);

        LocalTensor<half> resultHalf = tmpBufOutCast.Get<half>();
        Mul(resultHalf, gradHalf, maskHalf, this->processDataNum);

        Cast(yLocal, resultHalf, RoundMode::CAST_NONE, this->processDataNum);
    }
    // ================================================================
    // uint8 路径: grad→half, mask→half, Mul(half), Cast→uint8
    // ================================================================
    else if constexpr (std::is_same_v<T, uint8_t>) {
        LocalTensor<half> gradHalf = tmpBufGradCast.Get<half>();
        Cast(gradHalf, gradLocal, RoundMode::CAST_NONE, this->processDataNum);

        LocalTensor<half> maskHalf = tmpBufMaskHalf.Get<half>();
        Cast(maskHalf, maskLocal, RoundMode::CAST_NONE, this->processDataNum);

        LocalTensor<half> resultHalf = tmpBufOutCast.Get<half>();
        Mul(resultHalf, gradHalf, maskHalf, this->processDataNum);

        Cast(yLocal, resultHalf, RoundMode::CAST_NONE, this->processDataNum);
    }
    // ================================================================
    // bfloat16 路径：大 tile Compares+Select(float), 小 tile Mul+Cast 回退
    // ================================================================
    else if constexpr (std::is_same_v<T, bfloat16_t>) {
        LocalTensor<float> gradFloat = tmpBufGradCast.Get<float>();
        Cast(gradFloat, gradLocal, RoundMode::CAST_NONE, this->processDataNum);
        // mask uint8 → half
        LocalTensor<half> maskHalf = tmpBufMaskHalf.Get<half>();
        Cast(maskHalf, maskLocal, RoundMode::CAST_NONE, this->processDataNum);
        if (this->tileDataNum < CMP_ALIGN_ELEMS) {
            LocalTensor<float> maskFloat = tmpBufOutCast.Get<float>();
            Cast(maskFloat, maskHalf, RoundMode::CAST_NONE, this->processDataNum);
            LocalTensor<float> resultFloat = tmpBufResultFloat.Get<float>();
            Mul(resultFloat, gradFloat, maskFloat, this->processDataNum);
            Cast(yLocal, resultFloat, RoundMode::CAST_RINT, this->processDataNum);
        } else {
            uint32_t cmpCount = ((this->processDataNum + CMP_ALIGN_ELEMS - 1U) / CMP_ALIGN_ELEMS) * CMP_ALIGN_ELEMS;
            LocalTensor<uint8_t> cmpMask = tmpBufCmpMask.Get<uint8_t>();
            Compares(cmpMask, maskHalf, static_cast<half>(0), CMPMODE::NE, cmpCount);
            LocalTensor<float> selectDst = tmpBufOutCast.Get<float>();
            Select(selectDst, cmpMask, gradFloat, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, cmpCount);
            Cast(yLocal, selectDst, RoundMode::CAST_RINT, this->processDataNum);
        }
    }
    // ================================================================
    // int32 路径（保留原 Mul+Cast）
    // ================================================================
    else if constexpr (std::is_same_v<T, int32_t>) {
        LocalTensor<float> gradFloat = tmpBufGradCast.Get<float>();
        Cast(gradFloat, gradLocal, RoundMode::CAST_NONE, this->processDataNum);

        LocalTensor<half> maskHalf = tmpBufMaskHalf.Get<half>();
        Cast(maskHalf, maskLocal, RoundMode::CAST_NONE, this->processDataNum);

        LocalTensor<float> maskFloat = tmpBufOutCast.Get<float>();
        Cast(maskFloat, maskHalf, RoundMode::CAST_NONE, this->processDataNum);
        LocalTensor<float> resultFloat = tmpBufResultFloat.Get<float>();

        Mul(resultFloat, gradFloat, maskFloat, this->processDataNum);

        Cast(yLocal, resultFloat, RoundMode::CAST_RINT, this->processDataNum);
    }

    outputQueueY.EnQue<T>(yLocal);
    inputQueueGrad.FreeTensor(gradLocal);
    inputQueueMask.FreeTensor(maskLocal);
}

template <typename T>
__aicore__ inline void ReluGradV3<T>::Process()
{
    this->processDataNum = this->tileDataNum;
    CopyIn(0);

    for (int32_t i = 0; i < this->tileNum - 1; i++) {
        Compute(i);
        uint32_t curProcessDataNum = this->processDataNum;
        this->processDataNum = ((i + 1) == this->tileNum - 1) ? this->tailDataNum : this->tileDataNum;
        CopyIn(i + 1);
        this->processDataNum = curProcessDataNum;
        CopyOut(i);
    }

    if (this->tileNum > 0) {
        this->processDataNum = this->tailDataNum;
        Compute(this->tileNum - 1);
        CopyOut(this->tileNum - 1);
    }
}

} // namespace NsReluGradV3
#endif // __RELU_GRAD_V3_H__