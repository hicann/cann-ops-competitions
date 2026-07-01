#ifndef BATCH_TO_SPACE_TILING_H
#define BATCH_TO_SPACE_TILING_H

#include <cstdint>
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(BatchToSpaceTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, totalJobs);
    TILING_DATA_FIELD_DEF(uint32_t, mode);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, channels);
    TILING_DATA_FIELD_DEF(uint32_t, outputBatch);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, blockSize);
    TILING_DATA_FIELD_DEF(int32_t, cropTop);
    TILING_DATA_FIELD_DEF(int32_t, cropBottom);
    TILING_DATA_FIELD_DEF(int32_t, cropLeft);
    TILING_DATA_FIELD_DEF(int32_t, cropRight);
    TILING_DATA_FIELD_DEF(uint32_t, tilePixels);
    TILING_DATA_FIELD_DEF(uint32_t, tilesPerRow);
    TILING_DATA_FIELD_DEF(uint32_t, channelTileElems);
    TILING_DATA_FIELD_DEF(uint32_t, bufferBytes);
    TILING_DATA_FIELD_DEF(uint32_t, enableDoubleBuffer);
    TILING_DATA_FIELD_DEF(uint32_t, flagMergeGroup);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchToSpace, BatchToSpaceTilingData)

}  // namespace optiling

#endif  // BATCH_TO_SPACE_TILING_H
