/*!
 * \file selu_tiling_data.h
 * \brief Selu tiling data.
 */

#ifndef _SELU_TILING_DATA_H_
#define _SELU_TILING_DATA_H_

#include <cstdint>

constexpr int32_t SELU_RESERVED_PIPELINE_FLAG = 1;
constexpr int32_t SELU_RESERVED_BALANCED_TILE_FLAG = 2;

struct SeluTilingData {
    int64_t totalNum = 0;
    int64_t blockFactor = 0;
    int64_t ubFactor = 0;
    int32_t reserved = 0;
    int64_t blockDim = 0;
    int64_t baseTileNum = 0;
    int64_t extraTileNum = 0;
};

#endif
