#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GeluV2TilingData1)
  TILING_DATA_FIELD_DEF(int32_t, xTotalLength);
  TILING_DATA_FIELD_DEF(int32_t, tileNum);
  TILING_DATA_FIELD_DEF(int32_t, tileLength);
  TILING_DATA_FIELD_DEF(int32_t, lastTileLength);
  TILING_DATA_FIELD_DEF(int32_t, invalidLength);
  TILING_DATA_FIELD_DEF(int32_t, resTileNum);
  TILING_DATA_FIELD_DEF(int32_t, invalidLenPerResTile);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GeluV2, GeluV2TilingData1)
}



