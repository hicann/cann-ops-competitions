#ifndef FILLS_TILING_H
#define FILLS_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FillsTilingData)
TILING_DATA_FIELD_DEF(uint32_t, fillBits32);    // unified uint32_t fill value bit pattern
TILING_DATA_FIELD_DEF(uint32_t, formerNum);     // cores with larger chunk (elem count)
TILING_DATA_FIELD_DEF(uint32_t, formerElems);   // uint32 elements per former core
TILING_DATA_FIELD_DEF(uint32_t, tailElems);     // uint32 elements per tail core
TILING_DATA_FIELD_DEF(uint32_t, lastCoreBytes); // actual valid bytes for the last core
TILING_DATA_FIELD_DEF(uint32_t, bufferLen);     // UB buffer size (bytes), 512B-aligned
TILING_DATA_FIELD_DEF(uint32_t, bufferElems);   // bufferLen / 4
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);   // blockDim
TILING_DATA_FIELD_DEF(uint32_t, totalBytes);    // total data bytes (for TILING_KEY 0)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Fills, FillsTilingData)
} // namespace optiling
#endif // FILLS_TILING_H
