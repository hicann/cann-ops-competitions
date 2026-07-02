/* Erfinv TilingData —— elementwise，分核 + tile + dtypeCode（0=fp16,1=bf16,2=fp32）。 */
#ifndef ERFINV_TILING_H
#define ERFINV_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(ErfinvTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLen);
    TILING_DATA_FIELD_DEF(uint32_t, tileLen);
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreLen);
    TILING_DATA_FIELD_DEF(uint32_t, smallCoreLen);
    TILING_DATA_FIELD_DEF(uint32_t, coreNum);
    TILING_DATA_FIELD_DEF(uint32_t, dtypeCode);
    TILING_DATA_FIELD_DEF(uint32_t, tailExtra);  // 末核额外的不足 ALN 的零头（核长 32B 对齐后的余数）
    TILING_DATA_FIELD_DEF(uint32_t, useFlip);    // 1=启用 workspace 翻转复用 L2（仅 shape 够大时）；0=禁用，省 workGm 读写
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Erfinv, ErfinvTilingData)
} // namespace optiling
#endif
