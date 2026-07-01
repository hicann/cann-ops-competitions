// TilingKey模板定义的头文件
// Enhanced with IS_SINGLE_TILE compile-time dispatch for SingleTile vs DoubleBuffer paths
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(BatchToSpace,
                      ASCENDC_TPL_DATATYPE_DECL(DT_X, C_DT_FLOAT16, C_DT_FLOAT),
                      ASCENDC_TPL_BOOL_DECL(IS_SINGLE_TILE, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(IS_DEPTH_ALIGNED, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(IS_FAST_W_INCREMENT, 0, 1),
                      ASCENDC_TPL_UINT_DECL(BLOCK_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 2, 3, 4), );

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT16, C_DT_FLOAT),
        ASCENDC_TPL_BOOL_SEL(IS_SINGLE_TILE, 0, 1),
        ASCENDC_TPL_BOOL_SEL(IS_DEPTH_ALIGNED, 0, 1),
        ASCENDC_TPL_BOOL_SEL(IS_FAST_W_INCREMENT, 0, 1),
        ASCENDC_TPL_UINT_SEL(BLOCK_MODE, ASCENDC_TPL_UI_LIST, 0, 2, 3, 4), ), );
