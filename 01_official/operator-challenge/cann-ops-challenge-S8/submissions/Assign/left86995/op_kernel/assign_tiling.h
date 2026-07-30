
#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint32_t ASSIGN_MAX_DIMS = 8;

BEGIN_TILING_DATA_DEF(AssignTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, otherLength);
TILING_DATA_FIELD_DEF(uint32_t, blockLength);
TILING_DATA_FIELD_DEF(uint32_t, tileLength);
TILING_DATA_FIELD_DEF(uint32_t, alignNum);
TILING_DATA_FIELD_DEF(uint32_t, dimNum);
TILING_DATA_FIELD_DEF(uint32_t, isBroadcast);
TILING_DATA_FIELD_DEF_ARR(uint32_t, ASSIGN_MAX_DIMS, outShape);
TILING_DATA_FIELD_DEF_ARR(uint32_t, ASSIGN_MAX_DIMS, otherShape);
TILING_DATA_FIELD_DEF_ARR(uint32_t, ASSIGN_MAX_DIMS, otherStride);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Assign, AssignTilingData)
} // namespace optiling
