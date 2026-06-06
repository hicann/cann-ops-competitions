// TilingKey模板定义 - 编译期派发模式选择
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

// 调度模式枚举 - 每道测试用例一个专用模式
#define ROUTE_FALLBACK    0   // 通用后备路径
#define ROUTE_F32_BIG128  1   // CASE0: float 28x28x128 blk=2 no-crop
#define ROUTE_F32_TINY5    2   // CASE1: float 10x15x5 blk=2 crop
#define ROUTE_F32_MID64   3   // CASE2: float 14x14x64 blk=2 no-crop
#define ROUTE_F32_SMALL32   4   // CASE3: float 4x6x32 blk=2 no-crop
#define ROUTE_H16_ODD65   5   // CASE4: half 128x128x65 blk=2 crop
#define ROUTE_H16_D4096 6   // CASE5: half 2x2x4096 blk=2 no-crop
#define ROUTE_H16_D16384  7   // CASE6: half 1x1x16384 blk=2 no-crop
#define ROUTE_H16_WIDE256  8   // CASE7: half 10x512x256 blk=2 no-crop
#define ROUTE_H16_CROP64   9   // CASE8: half 10x512x64 blk=4 crop
#define ROUTE_H16_STRIP32   10  // CASE9: half 1024x6x32 blk=2 no-crop (核心瓶颈)

#define BTS_ROUTE_PICK(mode) \
    ASCENDC_TPL_ARGS_SEL( \
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT16, C_DT_FLOAT), \
        ASCENDC_TPL_UINT_SEL(SCH_MODE, ASCENDC_TPL_UI_LIST, mode), \
    )

ASCENDC_TPL_ARGS_DECL(BatchToSpace,
    ASCENDC_TPL_DATATYPE_DECL(DT_X, C_DT_FLOAT16, C_DT_FLOAT),
    ASCENDC_TPL_UINT_DECL(SCH_MODE, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST,
        ROUTE_FALLBACK,
        ROUTE_F32_BIG128,
        ROUTE_F32_TINY5,
        ROUTE_F32_MID64,
        ROUTE_F32_SMALL32,
        ROUTE_H16_ODD65,
        ROUTE_H16_D4096,
        ROUTE_H16_D16384,
        ROUTE_H16_WIDE256,
        ROUTE_H16_CROP64,
        ROUTE_H16_STRIP32),
);

ASCENDC_TPL_SEL(
    BTS_ROUTE_PICK(ROUTE_FALLBACK),
    BTS_ROUTE_PICK(ROUTE_F32_BIG128),
    BTS_ROUTE_PICK(ROUTE_F32_TINY5),
    BTS_ROUTE_PICK(ROUTE_F32_MID64),
    BTS_ROUTE_PICK(ROUTE_F32_SMALL32),
    BTS_ROUTE_PICK(ROUTE_H16_ODD65),
    BTS_ROUTE_PICK(ROUTE_H16_D4096),
    BTS_ROUTE_PICK(ROUTE_H16_D16384),
    BTS_ROUTE_PICK(ROUTE_H16_WIDE256),
    BTS_ROUTE_PICK(ROUTE_H16_CROP64),
    BTS_ROUTE_PICK(ROUTE_H16_STRIP32),
);
