#ifndef TILING_KEY_ERF_H
#define TILING_KEY_ERF_H
#include "ascendc/host_api/tiling/template_argument.h"

// 声明 D_T 模板参数, 默认值为 C_DT_FLOAT
ASCENDC_TPL_ARGS_DECL(Erf,
    ASCENDC_TPL_DATATYPE_DECL(D_T, C_DT_FLOAT),
);

// 注册 D_T = C_DT_FLOAT 的编译实例
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT),
    ),
);
#endif  // TILING_KEY_ERF_H