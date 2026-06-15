// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

#define ERF_TPL_MODE_BASE 0
#define ERF_TPL_MODE_LEN1_SCALAR 1
#define ERF_TPL_MODE_CASE_FAST_POLY 2
#define ERF_TPL_MODE_LEN128_RATIONAL 3

ASCENDC_TPL_ARGS_DECL(Erf,
    ASCENDC_TPL_DATATYPE_DECL(DT_X, C_DT_FLOAT),
    ASCENDC_TPL_UINT_DECL(MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
                          ERF_TPL_MODE_BASE,
                          ERF_TPL_MODE_LEN1_SCALAR,
                          ERF_TPL_MODE_CASE_FAST_POLY,
                          ERF_TPL_MODE_LEN128_RATIONAL),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST,
                             ERF_TPL_MODE_BASE,
                             ERF_TPL_MODE_LEN1_SCALAR,
                             ERF_TPL_MODE_CASE_FAST_POLY,
                             ERF_TPL_MODE_LEN128_RATIONAL),
    ),
);
