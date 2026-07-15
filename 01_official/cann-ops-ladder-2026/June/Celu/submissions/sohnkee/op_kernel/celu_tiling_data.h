/*!
 * \file celu_tiling_data.h
 * \brief Celu tiling data.
 */

#ifndef _CELU_TILING_DATA_H_
#define _CELU_TILING_DATA_H_

#include <cstdint>

constexpr int32_t CELU_RESERVED_PIPELINE_FLAG = 1;
constexpr int32_t CELU_RESERVED_BALANCED_TILE_FLAG = 2;

struct CeluTilingData {
    int64_t totalNum = 0;
    int64_t blockFactor = 0;
    int64_t ubFactor = 0;
    float alpha = 1.0f;
    int32_t reserved = 0;
    int64_t blockDim = 0;
    int64_t baseTileNum = 0;
    int64_t extraTileNum = 0;
};

#endif
