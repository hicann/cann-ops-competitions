
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GeluV2TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, inputNum);         // 输入元素数量
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);      // 输入每个tile的元素数（32B对齐单元）
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);          // tile总数
  TILING_DATA_FIELD_DEF(uint32_t, inputTypeLen);     // 输入是16 / 32
  TILING_DATA_FIELD_DEF(uint32_t, tailDataNum);      // 尾部元素数量
  TILING_DATA_FIELD_DEF(float, sqrt2);               // sqrt(2)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GeluV2, GeluV2TilingData)
}
