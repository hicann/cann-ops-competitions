#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AtanhTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, size);
    TILING_DATA_FIELD_DEF(uint32_t, tile_length);
    TILING_DATA_FIELD_DEF(uint32_t, split_unit);
    TILING_DATA_FIELD_DEF(uint32_t, per_core_block);
    TILING_DATA_FIELD_DEF(uint32_t, last_core_block);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Atanh, AtanhTilingData)
}
