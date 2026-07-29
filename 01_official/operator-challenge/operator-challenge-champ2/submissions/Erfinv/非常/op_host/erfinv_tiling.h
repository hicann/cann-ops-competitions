#ifndef ERFINV_TILING_H
#define ERFINV_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ErfinvTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, totalLength);
    TILING_DATA_FIELD_DEF(uint64_t, blocksPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, extraCoreBlocks);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Erfinv, ErfinvTilingData)
}  // namespace optiling

#endif  // ERFINV_TILING_H
