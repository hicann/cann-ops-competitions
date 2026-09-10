
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScaleTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, outer_size);    // product of input dims [0, axis)
  TILING_DATA_FIELD_DEF(uint32_t, scale_size);    // product of scale shape (== input[axis:end_axis])
  TILING_DATA_FIELD_DEF(uint32_t, inner_size);    // product of input dims [end_axis, rank)
  TILING_DATA_FIELD_DEF(uint32_t, has_bias);      // 1 if bias is present, 0 otherwise
  TILING_DATA_FIELD_DEF(uint32_t, dtype);         // ge::DataType integer value
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Scale, ScaleTilingData)
}
