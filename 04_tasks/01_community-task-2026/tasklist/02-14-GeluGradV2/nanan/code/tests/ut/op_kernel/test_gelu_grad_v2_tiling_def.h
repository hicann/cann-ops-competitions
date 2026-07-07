/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _GELU_GRAD_V2_TILING_H
#define _GELU_GRAD_V2_TILING_H

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

struct GeluGradV2TilingDataTest {
    uint64_t smallCoreDataNum = 1024;
    uint64_t bigCoreDataNum = 1152;
    uint64_t finalBigTileNum = 2;
    uint64_t finalSmallTileNum = 1;
    uint64_t tileDataNum = 1024;
    uint64_t smallTailDataNum = 1024;
    uint64_t bigTailDataNum = 128;
    uint64_t tailBlockNum = 0;
    uint64_t usedDb = 0;
};

inline void InitGeluGradV2TilingData(uint8_t* tiling, GeluGradV2TilingDataTest* const_data)
{
    memcpy(const_data, tiling, sizeof(GeluGradV2TilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    GeluGradV2TilingData tilingData;            \
    InitGeluGradV2TilingData(tilingPointer, &tilingData)
#endif


