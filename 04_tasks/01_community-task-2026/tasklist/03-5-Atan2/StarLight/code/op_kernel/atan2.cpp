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
 * \file atan2.cpp
 * \brief
*/

#include "atan2.h"

#define DOUBLE_BUFFER_NUM 2
#define SINGLE_BUFFER_NUM 1

enum class Atan2TilingKey : uint32_t
{
    TILING_KEY_DB = 0,
    TILING_KEY_NDB = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void atan2(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(Atan2TilingData);
    GET_TILING_DATA_WITH_STRUCT(Atan2TilingData, tilingData, tiling);
    AscendC::TPipe pipe;
    
    if constexpr (schMode == static_cast<uint32_t>(Atan2TilingKey::TILING_KEY_DB))
    {
        MyAtan2::KernelAtan2<DTYPE_X1, DTYPE_Y, DOUBLE_BUFFER_NUM> op;
        op.Init(x1, x2, y, &tilingData, &pipe);      // 算子kernel实例初始化
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(Atan2TilingKey::TILING_KEY_NDB))
    {
        MyAtan2::KernelAtan2<DTYPE_X1, DTYPE_Y, SINGLE_BUFFER_NUM> op;
        op.Init(x1, x2, y, &tilingData, &pipe);      // 算子kernel实例初始化
        op.Process();
    }
}