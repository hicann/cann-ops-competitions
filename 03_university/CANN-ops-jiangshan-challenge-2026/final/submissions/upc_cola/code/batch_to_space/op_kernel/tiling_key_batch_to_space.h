// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(BatchToSpace,
    ASCENDC_TPL_DATATYPE_DECL(DT_X, C_DT_FLOAT16, C_DT_FLOAT),
    ASCENDC_TPL_UINT_DECL(MODE, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6, 7, 8),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT16, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6, 7, 8),
    ),
);
