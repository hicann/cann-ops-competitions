
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AtanhTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, formerNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallSize);
  TILING_DATA_FIELD_DEF(uint32_t, incSize);
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Atanh, AtanhTilingData)
}
