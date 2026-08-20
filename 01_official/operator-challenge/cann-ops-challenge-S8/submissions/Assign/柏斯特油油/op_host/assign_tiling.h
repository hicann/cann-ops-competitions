
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AssignTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, tilisize);
  TILING_DATA_FIELD_DEF(uint64_t, totlength);
  // TILING_DATA_FIELD_DEF(uint64_t, inputlength);
  TILING_DATA_FIELD_DEF(uint16_t, dimNum);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, 4, outputShape);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, 4, inputShape);
  
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Assign, AssignTilingData)
}
