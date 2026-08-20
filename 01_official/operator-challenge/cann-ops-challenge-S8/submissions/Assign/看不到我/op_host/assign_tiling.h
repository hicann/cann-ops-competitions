
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AssignTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, size);
  TILING_DATA_FIELD_DEF(uint32_t, dims);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, input_shape);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, other_shape); 
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Assign, AssignTilingData)
}
