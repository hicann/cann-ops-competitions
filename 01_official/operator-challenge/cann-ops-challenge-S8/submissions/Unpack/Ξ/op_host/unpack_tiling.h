#include "register/tilingdata_base.h"

namespace optiling
{
    BEGIN_TILING_DATA_DEF(UnpackTilingData)
        TILING_DATA_FIELD_DEF(int, low_size);
        TILING_DATA_FIELD_DEF(int, mid_size);
        TILING_DATA_FIELD_DEF(int, high_size);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(Unpack, UnpackTilingData)
}
