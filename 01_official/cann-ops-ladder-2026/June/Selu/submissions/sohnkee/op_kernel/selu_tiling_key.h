/*!
 * \file selu_tiling_key.h
 * \brief Tiling 模板参数定义
 */

#ifndef __SELU_TILING_KEY_H__
#define __SELU_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define SELU_TPL_SCH_MODE_0 0
#define SELU_TPL_SCH_MODE_1 1
#define SELU_TPL_SCH_MODE_2 2
#define SELU_TPL_SCH_MODE_3 3
#define SELU_TPL_SCH_MODE_4 4
#define SELU_TPL_SCH_MODE_5 5
#define SELU_TPL_SCH_MODE_6 6
#define SELU_TPL_SCH_MODE_7 7
#define SELU_TPL_SCH_MODE_8 8

ASCENDC_TPL_ARGS_DECL(
    Selu,
    ASCENDC_TPL_UINT_DECL(
        schMode,
        4,
        ASCENDC_TPL_UI_LIST,
        SELU_TPL_SCH_MODE_0,
        SELU_TPL_SCH_MODE_1,
        SELU_TPL_SCH_MODE_2,
        SELU_TPL_SCH_MODE_3,
        SELU_TPL_SCH_MODE_4,
        SELU_TPL_SCH_MODE_5,
        SELU_TPL_SCH_MODE_6,
        SELU_TPL_SCH_MODE_7,
        SELU_TPL_SCH_MODE_8));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(
        schMode,
        ASCENDC_TPL_UI_LIST,
        SELU_TPL_SCH_MODE_0,
        SELU_TPL_SCH_MODE_1,
        SELU_TPL_SCH_MODE_2,
        SELU_TPL_SCH_MODE_3,
        SELU_TPL_SCH_MODE_4,
        SELU_TPL_SCH_MODE_5,
        SELU_TPL_SCH_MODE_6,
        SELU_TPL_SCH_MODE_7,
        SELU_TPL_SCH_MODE_8)));

#endif
