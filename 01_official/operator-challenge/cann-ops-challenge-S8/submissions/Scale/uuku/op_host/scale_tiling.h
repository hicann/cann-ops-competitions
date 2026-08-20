#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScaleTilingData)
  TILING_DATA_FIELD_DEF_ARR(int, 4, x_dims);
  TILING_DATA_FIELD_DEF_ARR(int, 4, scale_dims);
  TILING_DATA_FIELD_DEF(uint32_t, has_bias);
  TILING_DATA_FIELD_DEF(uint32_t, tile_length);
  TILING_DATA_FIELD_DEF(uint32_t, ub_bytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Scale, ScaleTilingData)
}
