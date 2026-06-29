
#include "register/tilingdata_base.h"
#include <cstdint>

constexpr int32_t TE_MAX_SHAPE = 4;

namespace optiling {
BEGIN_TILING_DATA_DEF(TensorEqualTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, total_len);
    TILING_DATA_FIELD_DEF(uint32_t, tile_len);
    TILING_DATA_FIELD_DEF_ARR(int32_t, TE_MAX_SHAPE, x1_shape);
    TILING_DATA_FIELD_DEF_ARR(int32_t, TE_MAX_SHAPE, x2_shape);
    TILING_DATA_FIELD_DEF_ARR(int32_t, TE_MAX_SHAPE, out_shape);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TensorEqual, TensorEqualTilingData)
}
