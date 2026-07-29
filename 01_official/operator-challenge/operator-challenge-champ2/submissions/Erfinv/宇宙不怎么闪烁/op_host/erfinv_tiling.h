#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ErfinvTilingData)
  TILING_DATA_FIELD_DEF(int64_t, totalLength);
  TILING_DATA_FIELD_DEF(int32_t, tileLength);
  TILING_DATA_FIELD_DEF(int32_t, perCoreBlock);
  TILING_DATA_FIELD_DEF(int32_t, lastCoreBlock);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Erfinv, ErfinvTilingData)
}
