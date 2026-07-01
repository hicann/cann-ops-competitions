#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

// Block-size dispatch for Erf (8 modes).
// Small inputs use compile-time BLK size — process fixed N elements,
// reading/writing N is safe because Ascend tensor alloc is page-aligned.
//   SCH_BLK8     : 1..8    elems, compile-time COUNT=8   (DataCopy, aligned)
//   SCH_BLK32    : 9..32   elems, compile-time COUNT=32
//   SCH_BLK64    : 33..64  elems, compile-time COUNT=64
//   SCH_BLK128   : 65..128 elems, compile-time COUNT=128
//   SCH_BLK256   : 129..256  elems, compile-time COUNT=256 (skip tiling read)
//   SCH_BLK512   : 257..512  elems, compile-time COUNT=512 (skip tiling read)
//   SCH_CACHE    : 513..2048,  single-buffer path (needs tiling read for count)
//   SCH_PIPELINE : > 2048,     double-buffer pipeline with multi-core split
#define SCH_BLK8     0
#define SCH_BLK32    1
#define SCH_BLK64    2
#define SCH_BLK128   3
#define SCH_BLK256   4
#define SCH_BLK512   5
#define SCH_CACHE    6
#define SCH_PIPELINE 7

#define ERF_TPL_MODE_SEL(mode) \
    ASCENDC_TPL_ARGS_SEL( \
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), \
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT), \
        ASCENDC_TPL_UINT_SEL(SCH_MODE, ASCENDC_TPL_UI_LIST, mode), \
    )

ASCENDC_TPL_ARGS_DECL(Erf,
    ASCENDC_TPL_DATATYPE_DECL(DT_X, C_DT_FLOAT),
    ASCENDC_TPL_UINT_DECL(SCH_MODE, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST,
        SCH_BLK8,
        SCH_BLK32,
        SCH_BLK64,
        SCH_BLK128,
        SCH_BLK256,
        SCH_BLK512,
        SCH_CACHE,
        SCH_PIPELINE),
);

ASCENDC_TPL_SEL(
    ERF_TPL_MODE_SEL(SCH_BLK8),
    ERF_TPL_MODE_SEL(SCH_BLK32),
    ERF_TPL_MODE_SEL(SCH_BLK64),
    ERF_TPL_MODE_SEL(SCH_BLK128),
    ERF_TPL_MODE_SEL(SCH_BLK256),
    ERF_TPL_MODE_SEL(SCH_BLK512),
    ERF_TPL_MODE_SEL(SCH_CACHE),
    ERF_TPL_MODE_SEL(SCH_PIPELINE),
);