
#ifndef UNPACK_TILING_H
#define UNPACK_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(UnpackTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, outerLength);
TILING_DATA_FIELD_DEF(uint32_t, axisLength);
TILING_DATA_FIELD_DEF(uint32_t, innerLength);
TILING_DATA_FIELD_DEF(uint32_t, blockLength);
TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Unpack, UnpackTilingData)
} // namespace optiling

#endif // UNPACK_TILING_H
