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
 * \file roll.cpp
 * \brief Roll kernel entry point
 */

#include "roll.h"

enum class RollTilingKey : uint32_t
{
    TILING_KEY_FLOAT = 0,
    TILING_KEY_FLOAT16 = 1,
    TILING_KEY_INT32 = 2,
    TILING_KEY_UINT32 = 3,
    TILING_KEY_INT8 = 4,
    TILING_KEY_UINT8 = 5,
    TILING_KEY_BFLOAT16 = 6,
};

template <uint32_t schMode>
__global__ __aicore__ void roll(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RollTilingData);
    GET_TILING_DATA_WITH_STRUCT(RollTilingData, tilingData, tiling);
    AscendC::TPipe pipe;

    if constexpr (schMode == static_cast<uint32_t>(RollTilingKey::TILING_KEY_FLOAT)) {
        NsRoll::Roll<float> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(RollTilingKey::TILING_KEY_FLOAT16)) {
        NsRoll::Roll<half> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(RollTilingKey::TILING_KEY_INT32)) {
        NsRoll::Roll<int32_t> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(RollTilingKey::TILING_KEY_UINT32)) {
        NsRoll::Roll<uint32_t> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(RollTilingKey::TILING_KEY_INT8)) {
        NsRoll::Roll<int8_t> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(RollTilingKey::TILING_KEY_UINT8)) {
        NsRoll::Roll<uint8_t> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(RollTilingKey::TILING_KEY_BFLOAT16)) {
        NsRoll::Roll<half> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    }
}
