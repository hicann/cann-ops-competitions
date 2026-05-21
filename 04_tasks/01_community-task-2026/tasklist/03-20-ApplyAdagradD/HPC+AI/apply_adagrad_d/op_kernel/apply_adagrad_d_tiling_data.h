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
 * \file apply_adagrad_d_tiling_data.h
 * \brief tiling data struct
 */

 #ifndef APPLY_ADAGRAD_D_TILING_DATA_H
 #define APPLY_ADAGRAD_D_TILING_DATA_H 
 
 struct ApplyAdagradDTilingData {
    uint32_t smallCoreDataNum; 
    uint32_t bigCoreDataNum;   
    uint32_t finalBigTileNum;   
    uint32_t finalSmallTileNum; 
    uint32_t smallTailDataNum;  
    uint32_t bigTailDataNum;    
    uint32_t tileDataNum;       
    uint32_t tailBlockNum;  
    bool updateSlots;   
    uint32_t overlapDataNum;
 };
 #endif
 