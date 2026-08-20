
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FillsTilingData)
TILING_DATA_FIELD_DEF(int32_t, baseSize);
TILING_DATA_FIELD_DEF(int32_t, value);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Fills, FillsTilingData)
} // namespace optiling
