/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file relu_grad_v3_tiling_data.h
 * \brief ReluGradV3 operator tiling data struct definition
 */

#ifndef RELU_GRAD_V3_TILING_DATA_H_
#define RELU_GRAD_V3_TILING_DATA_H_

struct ReluGradV3TilingData {
    uint64_t smallCoreDataNum;   // 普通Core每个处理的元素数
    uint64_t bigCoreDataNum;     // 尾Core每个处理的元素数（多1个Block）
    uint64_t tileDataNum;        // 每个Tile的元素数（UB分块基准）
    uint64_t tailBlockNum;       // 尾Core的数量

    uint64_t smallTailDataNum;   // Small Core尾Tile的元素数
    uint64_t bigTailDataNum;     // Big Core尾Tile的元素数
    uint64_t finalSmallTileNum;  // Small Core的总Tile数
    uint64_t finalBigTileNum;    // Big Core的总Tile数
};

#endif // RELU_GRAD_V3_TILING_DATA_H_