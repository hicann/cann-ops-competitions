
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ErfinvTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, bigNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigLength);
  TILING_DATA_FIELD_DEF(uint32_t, smallLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Erfinv, ErfinvTilingData)
}
