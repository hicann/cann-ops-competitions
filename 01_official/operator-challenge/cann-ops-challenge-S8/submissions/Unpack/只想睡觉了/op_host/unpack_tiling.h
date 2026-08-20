
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(UnpackTilingData)
TILING_DATA_FIELD_DEF(int32_t, D);
TILING_DATA_FIELD_DEF(int32_t, H);
TILING_DATA_FIELD_DEF(int32_t, W);
TILING_DATA_FIELD_DEF(int32_t, coreRows);
TILING_DATA_FIELD_DEF(int32_t, tailRows);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Unpack, UnpackTilingData)
} // namespace optiling
