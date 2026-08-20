
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AssignTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, finalBigTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, finalSmallTileNum);
  TILING_DATA_FIELD_DEF(int32_t, tileDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallTailDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigTailDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailBlockNum);

  TILING_DATA_FIELD_DEF(int32_t, y_dimensional);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, y_ndarray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, x1_ndarray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, x2_ndarray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, y_sumndarray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, x1_sumndarray);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, x2_sumndarray);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Assign, AssignTilingData)

BEGIN_TILING_DATA_DEF(AssignTilingDataHalf)
  TILING_DATA_FIELD_DEF(uint32_t, coreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, lastTileDataNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Assign_1, AssignTilingDataHalf)
}
