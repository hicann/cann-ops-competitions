
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GeluV2TilingData)
  //TILING_DATA_FIELD_DEF(uint32_t, size);
  TILING_DATA_FIELD_DEF(uint32_t, CoreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, finalTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, TailDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, Approximate);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GeluV2, GeluV2TilingData)
}
