#include "register/tilingdata_base.h"

namespace optiling
{
    BEGIN_TILING_DATA_DEF(AssignTilingData)
        TILING_DATA_FIELD_DEF(int, size);
        TILING_DATA_FIELD_DEF_ARR(int, 4, input_n);
        TILING_DATA_FIELD_DEF_ARR(int, 4, other_n);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(Assign, AssignTilingData)
}
