
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScaleTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, length);
  TILING_DATA_FIELD_DEF(uint32_t, batch_size);
  TILING_DATA_FIELD_DEF(uint32_t, batch_size_vec);
  TILING_DATA_FIELD_DEF(uint32_t, has_bias);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Scale, ScaleTilingData)
}