
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FillsTilingData)
  TILING_DATA_FIELD_DEF(float, value);
  TILING_DATA_FIELD_DEF(uint32_t, smallSize);
  TILING_DATA_FIELD_DEF(uint16_t, incSize);
  TILING_DATA_FIELD_DEF(uint16_t, formerNum);
  TILING_DATA_FIELD_DEF(uint32_t, valuePack);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Fills, FillsTilingData)
}
