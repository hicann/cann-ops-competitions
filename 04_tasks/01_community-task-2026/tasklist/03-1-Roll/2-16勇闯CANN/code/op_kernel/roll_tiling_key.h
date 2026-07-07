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
 * \file roll_tiling_key.h
 * \brief roll tiling key declare
 */

#ifndef __ROLL_TILING_KEY_H__
#define __ROLL_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define ELEMENTWISE_TPL_SCH_MODE_0 0   // float
#define ELEMENTWISE_TPL_SCH_MODE_1 1   // float16
#define ELEMENTWISE_TPL_SCH_MODE_2 2   // int32
#define ELEMENTWISE_TPL_SCH_MODE_3 3   // uint32
#define ELEMENTWISE_TPL_SCH_MODE_4 4   // int8
#define ELEMENTWISE_TPL_SCH_MODE_5 5   // uint8
#define ELEMENTWISE_TPL_SCH_MODE_6 6   // bfloat16 (treated as float16 in kernel)

ASCENDC_TPL_ARGS_DECL(
    Roll,
    ASCENDC_TPL_UINT_DECL(schMode, 3, ASCENDC_TPL_UI_LIST,
        ELEMENTWISE_TPL_SCH_MODE_0,
        ELEMENTWISE_TPL_SCH_MODE_1,
        ELEMENTWISE_TPL_SCH_MODE_2,
        ELEMENTWISE_TPL_SCH_MODE_3,
        ELEMENTWISE_TPL_SCH_MODE_4,
        ELEMENTWISE_TPL_SCH_MODE_5,
        ELEMENTWISE_TPL_SCH_MODE_6));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
        ELEMENTWISE_TPL_SCH_MODE_0,
        ELEMENTWISE_TPL_SCH_MODE_1,
        ELEMENTWISE_TPL_SCH_MODE_2,
        ELEMENTWISE_TPL_SCH_MODE_3,
        ELEMENTWISE_TPL_SCH_MODE_4,
        ELEMENTWISE_TPL_SCH_MODE_5,
        ELEMENTWISE_TPL_SCH_MODE_6)));

#endif
