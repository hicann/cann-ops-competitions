
#include "register/tilingdata_base.h"

namespace optiling
{
BEGIN_TILING_DATA_DEF(TensorEqualTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, formerLength);
  TILING_DATA_FIELD_DEF(uint32_t, formerNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailLength);
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, partitionLength);

  TILING_DATA_FIELD_DEF(int32_t, y_dimensional);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, y_ndarray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, x1_ndarray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, x2_ndarray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, y_sumndarray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, x1_sumndarray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, x2_sumndarray);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TensorEqual, TensorEqualTilingData)
}
