#include "register/tilingdata_base.h"

namespace optiling
{
    BEGIN_TILING_DATA_DEF(ScaleTilingData)
        TILING_DATA_FIELD_DEF(int, low_size);
        TILING_DATA_FIELD_DEF(int, mid_size);
        TILING_DATA_FIELD_DEF(int, high_size);
        TILING_DATA_FIELD_DEF(int, broadcast_size);
        TILING_DATA_FIELD_DEF_ARR(int, 2, broadcast_stride);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(Scale, ScaleTilingData)
}
