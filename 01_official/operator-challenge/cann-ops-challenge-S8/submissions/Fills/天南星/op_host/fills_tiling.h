
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FillsTilingData)

  TILING_DATA_FIELD_DEF(uint32_t, smallCoreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, finalBigTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, finalSmallTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallTailDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigTailDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailBlockNum);
  TILING_DATA_FIELD_DEF(float,value);
  
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Fills, FillsTilingData)


BEGIN_TILING_DATA_DEF(FillsTilingDataBF16)
  
  TILING_DATA_FIELD_DEF(uint32_t, coreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, lastTileDataNum);
  TILING_DATA_FIELD_DEF(float, value);
  
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Fills_1, FillsTilingDataBF16)

BEGIN_TILING_DATA_DEF(FillsTilingDataHalf)

  TILING_DATA_FIELD_DEF(uint32_t, smallCoreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, finalBigTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, finalSmallTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallTailDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigTailDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailBlockNum);
  TILING_DATA_FIELD_DEF(float,value);
  
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Fills_2, FillsTilingDataHalf)

}
