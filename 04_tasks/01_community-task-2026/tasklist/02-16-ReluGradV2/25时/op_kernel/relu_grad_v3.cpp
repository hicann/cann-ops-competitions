/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file relu_grad_v3.cpp
 * \brief ReluGradV3 operator kernel implementation for AscendC
 */

#include "relu_grad_v3.h"

enum class ReluGradV3TilingKey : uint32_t
{
    TILING_KEY_FLOAT16 = 0,   
    TILING_KEY_FLOAT32 = 1, 
    TILING_KEY_INT32 = 2,      
    TILING_KEY_INT8 = 3,     
    TILING_KEY_UINT8 = 4,     
    TILING_KEY_BFLOAT16 = 5     
};

template <uint32_t schMode>
__global__ __aicore__ void relu_grad_v3(GM_ADDR gradients, GM_ADDR mask, GM_ADDR backprops, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluGradV3TilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluGradV3TilingData, tilingData, tiling);

    // 分支1：float16类型
    if constexpr (schMode == static_cast<uint32_t>(ReluGradV3TilingKey::TILING_KEY_FLOAT16)) {
        NsReluGradV3::ReluGradV3<half> op; 
        op.Init(gradients, mask, backprops, &tilingData);
        op.Process();
    }
    // 分支2：float32类型
    else if constexpr (schMode == static_cast<uint32_t>(ReluGradV3TilingKey::TILING_KEY_FLOAT32)) {
        NsReluGradV3::ReluGradV3<float> op; 
        op.Init(gradients, mask, backprops, &tilingData);
        op.Process();
    }
    // 分支3：int32类型
    else if constexpr (schMode == static_cast<uint32_t>(ReluGradV3TilingKey::TILING_KEY_INT32)) {
        NsReluGradV3::ReluGradV3<int32_t> op; 
        op.Init(gradients, mask, backprops, &tilingData);
        op.Process();
    }
    // 分支4：int8类型
    else if constexpr (schMode == static_cast<uint32_t>(ReluGradV3TilingKey::TILING_KEY_INT8)) {
        NsReluGradV3::ReluGradV3<int8_t> op; 
        op.Init(gradients, mask, backprops, &tilingData);
        op.Process();
    }
    // 分支5：uint8类型
    else if constexpr (schMode == static_cast<uint32_t>(ReluGradV3TilingKey::TILING_KEY_UINT8)) {
        NsReluGradV3::ReluGradV3<uint8_t> op; 
        op.Init(gradients, mask, backprops, &tilingData);
        op.Process();
    }
    // 分支6：bfloat16类型
    else if constexpr (schMode == static_cast<uint32_t>(ReluGradV3TilingKey::TILING_KEY_BFLOAT16)) {
        NsReluGradV3::ReluGradV3<bfloat16_t> op; 
        op.Init(gradients, mask, backprops, &tilingData);
        op.Process();
    }
}