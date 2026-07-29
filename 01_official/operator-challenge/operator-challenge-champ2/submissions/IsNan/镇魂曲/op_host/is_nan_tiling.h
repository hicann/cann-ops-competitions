#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IsNanTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, blockLength);
  TILING_DATA_FIELD_DEF(uint32_t, dataBlockSize);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, alignedInputLength);
  TILING_DATA_FIELD_DEF(uint32_t, alignedOutputLength);
  TILING_DATA_FIELD_DEF(uint32_t, useBaselineSplit);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IsNan, IsNanTilingData)
}  // namespace optiling
