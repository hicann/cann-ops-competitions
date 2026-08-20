#ifndef FILLS_TILING_H
#define FILLS_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(FillsTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, blockLength);
  TILING_DATA_FIELD_DEF(uint32_t, lastBlockLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
  TILING_DATA_FIELD_DEF(float, fillValue);
  TILING_DATA_FIELD_DEF(uint16_t, fillValueBf16);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Fills, FillsTilingData)

}  // namespace optiling

#endif
