
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FillsTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, size);
  TILING_DATA_FIELD_DEF(float, value);
  TILING_DATA_FIELD_DEF(uint32_t, dtype);
  TILING_DATA_FIELD_DEF(int32_t, int_value);
  TILING_DATA_FIELD_DEF(uint32_t, bf16_value);
  TILING_DATA_FIELD_DEF(uint32_t, byte_value);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Fills, FillsTilingData)
}
