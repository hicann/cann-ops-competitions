// TilingKey: DT_X(datatype) + usePipeline(0=Direct, 1=Pipeline)
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(Erf,
    ASCENDC_TPL_UINT_DECL(usePipeline, 1, ASCENDC_TPL_UI_LIST, 0, 1),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(usePipeline, ASCENDC_TPL_UI_LIST, 0, 1),
    ),
);

