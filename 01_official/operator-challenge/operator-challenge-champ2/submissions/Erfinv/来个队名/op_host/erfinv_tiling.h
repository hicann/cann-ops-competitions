#pragma once

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ErfinvTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, coreSize);
  TILING_DATA_FIELD_DEF(uint32_t, lastCoreSize);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Erfinv, ErfinvTilingData)
} // namespace optiling
