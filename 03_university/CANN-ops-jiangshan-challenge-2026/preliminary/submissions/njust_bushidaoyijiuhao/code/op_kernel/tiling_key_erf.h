// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(Erf,
    ASCENDC_TPL_DATATYPE_DECL(DT_X, C_DT_FLOAT),
    ASCENDC_TPL_BOOL_DECL(USE_PIPE, 0, 1),
    ASCENDC_TPL_BOOL_DECL(ONE_CORE, 0, 1),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT),
        ASCENDC_TPL_BOOL_SEL(USE_PIPE, 0, 1),
        ASCENDC_TPL_BOOL_SEL(ONE_CORE, 0, 1),
    ),
);
