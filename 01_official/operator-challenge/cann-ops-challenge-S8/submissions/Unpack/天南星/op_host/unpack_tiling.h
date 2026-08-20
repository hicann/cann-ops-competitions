#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(UnpackTilingData)
  // 1. 核心 Shape 与 Axis 信息
  TILING_DATA_FIELD_DEF(int32_t,  axis);
  TILING_DATA_FIELD_DEF(uint32_t, num);        // shape[axis] 的大小
  TILING_DATA_FIELD_DEF(int32_t,  ndim);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 4, shape); // 严格限制 4 维

  // 2. 数据类型与 UB Tiling 参数
  TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum); // 【修正】改为 uint32_t，与 Kernel 侧对齐
  TILING_DATA_FIELD_DEF(uint32_t, transRows);   // Path 30(Gather转置): 每块处理的 outer 行数 R

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Unpack, UnpackTilingData)

} // namespace optiling
