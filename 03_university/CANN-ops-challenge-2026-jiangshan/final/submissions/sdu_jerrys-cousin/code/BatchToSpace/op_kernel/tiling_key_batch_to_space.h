#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

// MODE 取值见 batch_to_space_tiling.h，主机侧按数据点选择编译期特化核函数。
ASCENDC_TPL_ARGS_DECL(BatchToSpace,
    ASCENDC_TPL_DATATYPE_DECL(DT_X, C_DT_FLOAT16, C_DT_FLOAT),
    ASCENDC_TPL_UINT_DECL(MODE, 16, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                          10, 11, 12, 13, 14, 15),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT16, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             10, 11, 12, 13, 14, 15),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY)
    ),
);
