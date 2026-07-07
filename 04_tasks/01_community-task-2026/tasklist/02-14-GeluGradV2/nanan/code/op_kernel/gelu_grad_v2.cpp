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
 * \file gelu_grad_v2.cpp
 * \brief
 */
#include <cstdint>
#include "kernel_operator.h"
#include "gelu_grad_v2.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void gelu_grad_v2(
    GM_ADDR dy, GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    
    GET_TILING_DATA(tiling_data, tiling);

    if (TILING_KEY_IS(1)) { 
        GeluGradV2None<DTYPE_DY> op;
        op.Init(                                                                                    
            dy, x, z,
            tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum,
            tiling_data.finalBigTileNum, tiling_data.finalSmallTileNum,
            tiling_data.tileDataNum, tiling_data.smallTailDataNum,
            tiling_data.bigTailDataNum, tiling_data.tailBlockNum,
            tiling_data.usedDb); 
       op.Process();
    } else if (TILING_KEY_IS(2)) { 
        GeluGradV2Tanh<DTYPE_DY> op1;
        op1.Init(                                                                                    
            dy, x, z,
            tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum,
            tiling_data.finalBigTileNum, tiling_data.finalSmallTileNum,
            tiling_data.tileDataNum, tiling_data.smallTailDataNum,
            tiling_data.bigTailDataNum, tiling_data.tailBlockNum,
            tiling_data.usedDb); 
        op1.Process();
    }
}
