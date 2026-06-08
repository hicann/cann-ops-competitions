#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

// float32 + 是否单 tile，两条编译期路径保持不变。
ASCENDC_TPL_ARGS_DECL(Erf,
    ASCENDC_TPL_DATATYPE_DECL(DT_X, C_DT_FLOAT),
    ASCENDC_TPL_BOOL_DECL(IS_SINGLE_TILE, 0, 1),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT),
        ASCENDC_TPL_BOOL_SEL(IS_SINGLE_TILE, 0, 1),
    ),
);
