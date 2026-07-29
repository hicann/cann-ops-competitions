#ifndef GELUV2_TILING_H
#define GELUV2_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GeluV2TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);              // 每个完整tile的元素数
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);                  // tile总数(含尾部)
    TILING_DATA_FIELD_DEF(uint32_t, lastTileDataNum);          // 最后一个tile的元素数(<=tileDataNum)
    TILING_DATA_FIELD_DEF(uint32_t, processDataNum_computes);  // 计算缓冲区对齐大小(float元素数)
    TILING_DATA_FIELD_DEF(uint32_t, typeLength);               // 数据类型字节数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GeluV2, GeluV2TilingData)
} // namespace optiling

#endif // GELUV2_TILING_H
