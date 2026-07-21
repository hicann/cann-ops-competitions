#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IsNanTilingData)
TILING_DATA_FIELD_DEF(uint32_t, smallSize);
TILING_DATA_FIELD_DEF(uint32_t, incSize);
TILING_DATA_FIELD_DEF(uint32_t, totalSize);
TILING_DATA_FIELD_DEF(uint16_t, formerNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IsNan, IsNanTilingData)
} // namespace optiling¡¤