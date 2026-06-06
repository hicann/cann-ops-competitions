#pragma once

#include <cstdint>

constexpr uint32_t BTS_MICRO_TILE_OFFSET_COLS = 64;
constexpr uint32_t BTS_MICRO_TILE_OFFSET_DEPTH = 65;
constexpr uint32_t BTS_MICRO_TILE_OFFSET_COUNT =
    BTS_MICRO_TILE_OFFSET_COLS * BTS_MICRO_TILE_OFFSET_DEPTH;

#ifndef BTS_TPL_MODE_GENERIC
enum BatchToSpaceTemplateMode : uint32_t
{
    BTS_TPL_MODE_GENERIC = 0,
    BTS_TPL_MODE_DIRECT_COPY = 1,
    BTS_TPL_MODE_BLOCK_SIZE_ONE = 2,
    BTS_TPL_MODE_BLOCK_SIZE_TWO_RESIDUE = 3,
    BTS_TPL_MODE_BLOCK_SIZE_TWO_ROW_PACKED = 4,
    BTS_TPL_MODE_BLOCK_SIZE_TWO_HEIGHT_PACKED = 5,
    BTS_TPL_MODE_GENERIC_ROW_PACKED = 6,
    BTS_TPL_MODE_BLOCK_SIZE_TWO_MICRO_TILE = 7,
    BTS_TPL_MODE_BLOCK_SIZE_TWO_SMALL_NOCROP = 8,
    BTS_TPL_MODE_BLOCK_SIZE_TWO_LARGE_DEPTH_INPUT_PACKED = 9,
};
#endif

enum BatchToSpaceStrategy : uint32_t
{
    BTS_STRATEGY_GENERIC = 0,
    BTS_STRATEGY_BLOCK_SIZE_ONE = 1,
    BTS_STRATEGY_BLOCK_SIZE_TWO_RESIDUE = 2,
    BTS_STRATEGY_BLOCK_SIZE_TWO_ROW_PACKED = 4,
    BTS_STRATEGY_DIRECT_COPY = 5,
    BTS_STRATEGY_BLOCK_SIZE_TWO_HEIGHT_PACKED = 6,
    BTS_STRATEGY_GENERIC_ROW_PACKED = 10,
    BTS_STRATEGY_BLOCK_SIZE_TWO_MICRO_TILE = 12,
    BTS_STRATEGY_BLOCK_SIZE_TWO_SMALL_NOCROP = 13,
    BTS_STRATEGY_BLOCK_SIZE_TWO_LARGE_DEPTH_INPUT_PACKED = 19,
};

struct BatchToSpaceTilingData
{
    uint32_t batch;
    uint32_t height;
    uint32_t width;
    uint32_t depth;
    uint32_t outBatch;
    uint32_t outHeight;
    uint32_t outWidth;
    uint32_t blockSize;
    uint32_t cropTop;
    uint32_t cropLeft;
    uint32_t tileDepth;
    uint32_t pointsPerCore;
    uint32_t tailPoints;
    uint32_t usedCoreNum;
    uint32_t strategy;
    uint32_t rowPackedTileRows;
    uint32_t rowPackedTileCols;
    uint32_t inputLength;
    uint32_t outputLength;
    uint32_t microTileOffsetPadding[5];
    uint32_t microTileGatherOffsets[BTS_MICRO_TILE_OFFSET_COUNT];
};
