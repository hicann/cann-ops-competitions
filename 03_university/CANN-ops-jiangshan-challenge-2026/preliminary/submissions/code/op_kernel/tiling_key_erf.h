#pragma once
#include "ascendc/host_api/tiling/template_argument.h"

#define ELEMENTWISE_TPL_SCH_MODE_0 0  // 小输入，无双缓冲
#define ELEMENTWISE_TPL_SCH_MODE_1 1  // 大输入，有双缓冲

ASCENDC_TPL_ARGS_DECL(
    Erf,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST,
        ELEMENTWISE_TPL_SCH_MODE_0, ELEMENTWISE_TPL_SCH_MODE_1));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
            ELEMENTWISE_TPL_SCH_MODE_0, ELEMENTWISE_TPL_SCH_MODE_1)
    ),
);
