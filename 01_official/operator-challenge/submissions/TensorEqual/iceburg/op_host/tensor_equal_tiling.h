
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TensorEqualTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalSize);
  TILING_DATA_FIELD_DEF(uint32_t, rank);
  TILING_DATA_FIELD_DEF(uint32_t, mode);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, outDims);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, x1Strides);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, x2Strides);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TensorEqual, TensorEqualTilingData)
}
