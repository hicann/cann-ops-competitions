#ifndef FILLS_TILING_H
#define FILLS_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, tileLength);
TILING_DATA_FIELD_DEF(uint32_t, smallCoreLength);
TILING_DATA_FIELD_DEF(uint32_t, incCoreLength);
TILING_DATA_FIELD_DEF(uint32_t, formerNum);
TILING_DATA_FIELD_DEF(float, value);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Fills, TilingData)
} // namespace optiling

#endif // FILLS_TILING_H
