
#include "register/tilingdata_base.h"

namespace optiling {

enum AtanhDType : uint32_t {
    DTYPE_FP32 = 1,
    DTYPE_FP16 = 2,
    DTYPE_BF16 = 3,
    DTYPE_INT32 = 4,
    DTYPE_INT16 = 5,
    DTYPE_INT8 = 6,
    DTYPE_UINT8 = 7
};

BEGIN_TILING_DATA_DEF(AtanhTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, size);
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreDataNum);   
  TILING_DATA_FIELD_DEF(uint32_t, finalSmallTileNum);  
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);        
  TILING_DATA_FIELD_DEF(uint32_t, smallTailDataNum);   
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreDataNum);     
  TILING_DATA_FIELD_DEF(uint32_t, finalBigTileNum);  
  TILING_DATA_FIELD_DEF(uint32_t, bigTailDataNum);     
  TILING_DATA_FIELD_DEF(uint32_t, tailBlockNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Atanh, AtanhTilingData)
}
