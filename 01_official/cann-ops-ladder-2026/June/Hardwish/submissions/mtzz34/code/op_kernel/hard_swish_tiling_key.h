/*!
 * \file hard_swish_tiling_key.h
 * \brief Tiling 模板参数定义
 */

#ifndef __HARDSWISH_TILING_KEY_H__
#define __HARDSWISH_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define HARDSWISH_TPL_SCH_MODE_0 0
#define HARDSWISH_TPL_SCH_MODE_1 1
#define HARDSWISH_TPL_SCH_MODE_2 2
#define HARDSWISH_TPL_SCH_MODE_3 3

ASCENDC_TPL_ARGS_DECL(
    HardSwish,
    ASCENDC_TPL_UINT_DECL(
        schMode,
        3,
        ASCENDC_TPL_UI_LIST,
        HARDSWISH_TPL_SCH_MODE_0,
        HARDSWISH_TPL_SCH_MODE_1,
        HARDSWISH_TPL_SCH_MODE_2,
        HARDSWISH_TPL_SCH_MODE_3));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(
        schMode,
        ASCENDC_TPL_UI_LIST,
        HARDSWISH_TPL_SCH_MODE_0,
        HARDSWISH_TPL_SCH_MODE_1,
        HARDSWISH_TPL_SCH_MODE_2,
        HARDSWISH_TPL_SCH_MODE_3)));

#endif
