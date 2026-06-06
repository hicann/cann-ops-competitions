// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

#define BTS_TPL_MODE_GENERIC 0
#define BTS_TPL_MODE_DIRECT_COPY 1
#define BTS_TPL_MODE_BLOCK_SIZE_ONE 2
#define BTS_TPL_MODE_BLOCK_SIZE_TWO_RESIDUE 3
#define BTS_TPL_MODE_BLOCK_SIZE_TWO_ROW_PACKED 4
#define BTS_TPL_MODE_BLOCK_SIZE_TWO_HEIGHT_PACKED 5
#define BTS_TPL_MODE_GENERIC_ROW_PACKED 6
#define BTS_TPL_MODE_BLOCK_SIZE_TWO_MICRO_TILE 7
#define BTS_TPL_MODE_BLOCK_SIZE_TWO_SMALL_NOCROP 8
#define BTS_TPL_MODE_BLOCK_SIZE_TWO_LARGE_DEPTH_INPUT_PACKED 9

ASCENDC_TPL_ARGS_DECL(BatchToSpace,
                      ASCENDC_TPL_DATATYPE_DECL(DT_X, C_DT_FLOAT16, C_DT_FLOAT),
                      ASCENDC_TPL_UINT_DECL(MODE, 10, ASCENDC_TPL_UI_LIST,
                                            BTS_TPL_MODE_GENERIC,
                                            BTS_TPL_MODE_DIRECT_COPY,
                                            BTS_TPL_MODE_BLOCK_SIZE_ONE,
                                            BTS_TPL_MODE_BLOCK_SIZE_TWO_RESIDUE,
                                            BTS_TPL_MODE_BLOCK_SIZE_TWO_ROW_PACKED,
                                            BTS_TPL_MODE_BLOCK_SIZE_TWO_HEIGHT_PACKED,
                                            BTS_TPL_MODE_GENERIC_ROW_PACKED,
                                            BTS_TPL_MODE_BLOCK_SIZE_TWO_MICRO_TILE,
                                            BTS_TPL_MODE_BLOCK_SIZE_TWO_SMALL_NOCROP,
                                            BTS_TPL_MODE_BLOCK_SIZE_TWO_LARGE_DEPTH_INPUT_PACKED));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT16, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST,
                             BTS_TPL_MODE_GENERIC,
                             BTS_TPL_MODE_DIRECT_COPY,
                             BTS_TPL_MODE_BLOCK_SIZE_ONE,
                             BTS_TPL_MODE_BLOCK_SIZE_TWO_RESIDUE,
                             BTS_TPL_MODE_BLOCK_SIZE_TWO_ROW_PACKED,
                             BTS_TPL_MODE_BLOCK_SIZE_TWO_HEIGHT_PACKED,
                             BTS_TPL_MODE_GENERIC_ROW_PACKED,
                             BTS_TPL_MODE_BLOCK_SIZE_TWO_MICRO_TILE,
                             BTS_TPL_MODE_BLOCK_SIZE_TWO_SMALL_NOCROP,
                             BTS_TPL_MODE_BLOCK_SIZE_TWO_LARGE_DEPTH_INPUT_PACKED)));
