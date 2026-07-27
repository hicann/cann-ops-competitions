
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GeluV2TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);   // total element count
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);    // elements per tile (32B-aligned)
  TILING_DATA_FIELD_DEF(int32_t,  approximate);   // 0=none(erf), 1=tanh
  TILING_DATA_FIELD_DEF(int32_t,  dtype);         // 0=fp16, 1=bf16, 2=fp32
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GeluV2, GeluV2TilingData)
}
