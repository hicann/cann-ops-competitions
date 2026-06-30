#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GeluV2TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalSize);
  TILING_DATA_FIELD_DEF(uint32_t, approximate);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GeluV2, GeluV2TilingData)
}
