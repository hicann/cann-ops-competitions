#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AssignTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, totalSize);
  TILING_DATA_FIELD_DEF(uint64_t, perCoreSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Assign, AssignTilingData)
}
