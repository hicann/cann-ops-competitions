// TilingKey模板定义的头文件
#pragma once

#include "addcmul_tiling.h"
#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(Addcmul,
    ASCENDC_TPL_DATATYPE_DECL(DT_INPUT_DATA, C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT8, C_DT_INT32)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_DATA, C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT8, C_DT_INT32)
    )
);
