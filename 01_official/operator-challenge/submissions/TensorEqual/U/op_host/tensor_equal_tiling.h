#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TensorEqualTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, block_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);
  // Support up to 4D broadcast. Layout per input: [rank, dim0, dim1, dim2, dim3].
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 2*5, shapeInf);
  TILING_DATA_FIELD_DEF(uint8_t, ALIGN_NUM);
  TILING_DATA_FIELD_DEF(bool, boardCast);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TensorEqual, TensorEqualTilingData)
}
