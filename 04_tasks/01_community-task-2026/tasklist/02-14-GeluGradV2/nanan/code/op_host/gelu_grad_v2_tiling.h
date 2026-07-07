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
 * \file mse_loss_grad_tiling.h
 * \brief
 */
#ifndef MSE_LOSS_GRAD_TILING_H
#define MSE_LOSS_GRAD_TILING_H

#pragma once
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GeluGradV2TilingData)
TILING_DATA_FIELD_DEF(uint64_t, smallCoreDataNum); // 总计算数据量
TILING_DATA_FIELD_DEF(uint64_t, bigCoreDataNum);     // 每个核上总计算数据分块个数
TILING_DATA_FIELD_DEF(uint64_t, finalBigTileNum);   // 尾块的个数
TILING_DATA_FIELD_DEF(uint64_t, finalSmallTileNum);
TILING_DATA_FIELD_DEF(uint64_t, tileDataNum);
TILING_DATA_FIELD_DEF(uint64_t, smallTailDataNum);
TILING_DATA_FIELD_DEF(uint64_t, bigTailDataNum);
TILING_DATA_FIELD_DEF(uint64_t, tailBlockNum);
TILING_DATA_FIELD_DEF(uint64_t, usedDb); //  
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GeluGradV2, GeluGradV2TilingData)

void GetTilingKey(const uint32_t dtypeKey, uint32_t& tilingKey);
} // namespace optiling
#endif
