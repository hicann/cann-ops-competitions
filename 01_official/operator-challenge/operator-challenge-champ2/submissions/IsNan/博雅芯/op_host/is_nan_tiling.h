
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IsNanTilingData)
    TILING_DATA_FIELD_DEF(int32_t, total);
    TILING_DATA_FIELD_DEF(int32_t, each);
    TILING_DATA_FIELD_DEF(int32_t, buffer);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IsNan, IsNanTilingData)
}
