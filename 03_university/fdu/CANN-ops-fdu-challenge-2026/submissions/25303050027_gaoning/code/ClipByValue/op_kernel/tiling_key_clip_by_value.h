// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

// Kernel模板只按x/y的数据类型分发；min/max标量与切分信息由ClipByValueTilingData运行时传入。
ASCENDC_TPL_ARGS_DECL(ClipByValue,
    ASCENDC_TPL_DATATYPE_DECL(DT_X, C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT32),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT32),
    ),
);
