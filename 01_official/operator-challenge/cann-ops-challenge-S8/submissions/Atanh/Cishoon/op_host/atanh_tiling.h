
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AtanhTilingData)
  // smallCoreDataNum 低 7 bit 打包 tailBlockNum；真实 smallCoreDataNum 至少 128 对齐
  // bigCoreDataNum 不传：kernel 内 constexpr (512/sizeof(T)) 加上 smallCoreDataNum 即可
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Atanh, AtanhTilingData)
}
