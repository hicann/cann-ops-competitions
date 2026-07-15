/*!
 * \file celu_tiling_key.h
 * \brief Tiling 模板参数定义
 */

#ifndef __CELU_TILING_KEY_H__
#define __CELU_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define CELU_TPL_SCH_MODE_0 0
#define CELU_TPL_SCH_MODE_1 1
#define CELU_TPL_SCH_MODE_2 2
#define CELU_TPL_SCH_MODE_3 3
#define CELU_TPL_SCH_MODE_4 4
#define CELU_TPL_SCH_MODE_5 5
#define CELU_TPL_SCH_MODE_6 6
#define CELU_TPL_SCH_MODE_7 7
#define CELU_TPL_SCH_MODE_8 8
#define CELU_TPL_SCH_MODE_9 9
#define CELU_TPL_SCH_MODE_10 10
#define CELU_TPL_SCH_MODE_11 11
#define CELU_TPL_SCH_MODE_12 12

ASCENDC_TPL_ARGS_DECL(
    Celu,
    ASCENDC_TPL_UINT_DECL(
        schMode,
        4,
        ASCENDC_TPL_UI_LIST,
        CELU_TPL_SCH_MODE_0,
        CELU_TPL_SCH_MODE_1,
        CELU_TPL_SCH_MODE_2,
        CELU_TPL_SCH_MODE_3,
        CELU_TPL_SCH_MODE_4,
        CELU_TPL_SCH_MODE_5,
        CELU_TPL_SCH_MODE_6,
        CELU_TPL_SCH_MODE_7,
        CELU_TPL_SCH_MODE_8,
        CELU_TPL_SCH_MODE_9,
        CELU_TPL_SCH_MODE_10,
        CELU_TPL_SCH_MODE_11,
        CELU_TPL_SCH_MODE_12));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(
        schMode,
        ASCENDC_TPL_UI_LIST,
        CELU_TPL_SCH_MODE_0,
        CELU_TPL_SCH_MODE_1,
        CELU_TPL_SCH_MODE_2,
        CELU_TPL_SCH_MODE_3,
        CELU_TPL_SCH_MODE_4,
        CELU_TPL_SCH_MODE_5,
        CELU_TPL_SCH_MODE_6,
        CELU_TPL_SCH_MODE_7,
        CELU_TPL_SCH_MODE_8,
        CELU_TPL_SCH_MODE_9,
        CELU_TPL_SCH_MODE_10,
        CELU_TPL_SCH_MODE_11,
        CELU_TPL_SCH_MODE_12)));

#endif
