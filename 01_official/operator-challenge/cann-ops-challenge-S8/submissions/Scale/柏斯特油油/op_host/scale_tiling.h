
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScaleTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, preLength);
  TILING_DATA_FIELD_DEF(uint64_t, midLength);
  TILING_DATA_FIELD_DEF(uint64_t, postLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Scale, ScaleTilingData)
}
