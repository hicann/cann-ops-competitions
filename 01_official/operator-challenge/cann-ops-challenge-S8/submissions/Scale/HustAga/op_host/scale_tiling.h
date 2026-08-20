
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScaleTilingData)
  TILING_DATA_FIELD_DEF(int32_t, tileDataNum);
  TILING_DATA_FIELD_DEF(int32_t, innerStride);
  TILING_DATA_FIELD_DEF(int32_t, outerStride);
  TILING_DATA_FIELD_DEF(int32_t, bigCoreNum);
  TILING_DATA_FIELD_DEF(int32_t, bigCoreProcessNum);
  TILING_DATA_FIELD_DEF(int32_t, smallCoreProcessNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Scale, ScaleTilingData)
}
