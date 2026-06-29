#ifndef GELU_V2_TILING_H
#define GELU_V2_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GeluV2TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, coreDataNum);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GeluV2, GeluV2TilingData)
} // namespace optiling

#endif // GELU_V2_TILING_H
