#include "register/tilingdata_base.h"

namespace optiling
{
    BEGIN_TILING_DATA_DEF(AtanhTilingData)
        TILING_DATA_FIELD_DEF(int, size);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(Atanh, AtanhTilingData)
}
