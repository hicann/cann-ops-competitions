/* IsNan TilingData —— elementwise，分核 + tile + dtypeCode（0=fp16,1=bf16,2=fp32）。 */
#ifndef IS_NAN_TILING_H
#define IS_NAN_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IsNanTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLen);    // dtype 走 TilingKey；分核/tailExtra 在 kernel 内重算
    TILING_DATA_FIELD_DEF(uint32_t, tileLen);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(IsNan, IsNanTilingData)
} // namespace optiling
#endif
