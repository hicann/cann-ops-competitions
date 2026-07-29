
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TensorEqualTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, outSize);
  TILING_DATA_FIELD_DEF(uint32_t, elemBytes);
  TILING_DATA_FIELD_DEF(uint32_t, rank);
  TILING_DATA_FIELD_DEF(uint32_t, outShape0);
  TILING_DATA_FIELD_DEF(uint32_t, outShape1);
  TILING_DATA_FIELD_DEF(uint32_t, outShape2);
  TILING_DATA_FIELD_DEF(uint32_t, outShape3);
  TILING_DATA_FIELD_DEF(uint32_t, x1Stride0);
  TILING_DATA_FIELD_DEF(uint32_t, x1Stride1);
  TILING_DATA_FIELD_DEF(uint32_t, x1Stride2);
  TILING_DATA_FIELD_DEF(uint32_t, x1Stride3);
  TILING_DATA_FIELD_DEF(uint32_t, x2Stride0);
  TILING_DATA_FIELD_DEF(uint32_t, x2Stride1);
  TILING_DATA_FIELD_DEF(uint32_t, x2Stride2);
  TILING_DATA_FIELD_DEF(uint32_t, x2Stride3);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TensorEqual, TensorEqualTilingData)
}
