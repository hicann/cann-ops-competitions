
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FillsTilingData)
  // smallCoreDataNum 低 7 bit 打包 tailBlockNum；真实 smallCoreDataNum 按 128 对齐。
  // bigCoreDataNum 在 kernel 里由 smallCoreDataNum + 128 现算（int32 路径下 blockDataNum 恒为 128）
  // tileLength 通过 TilingKey 分发：小尺寸走 fast path 不需要；大尺寸 kernel 内 constexpr 49152
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, valueUint32);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Fills, FillsTilingData)
}
