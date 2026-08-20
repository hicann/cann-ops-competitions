#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FillsTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, size);        // 以 32-bit 元素为单位
  TILING_DATA_FIELD_DEF(uint32_t, chunk);       // 同上，单核任务量
  TILING_DATA_FIELD_DEF(uint32_t, value_bits);  // 打包后的 4 字节常量
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Fills, FillsTilingData)
}