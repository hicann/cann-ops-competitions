#pragma once

#include <cstdint>

#ifndef BTS_FAST_PATHS
#define BTS_FAST_PATHS 1
#endif

enum BatchToSpaceStaticVariant : uint8_t {
  BTS_STATIC_VARIANT_NONE = 0,
  BTS_STATIC_VARIANT_ROW_PLANE_STRIDED = 1,
  BTS_STATIC_VARIANT_ROW_GATHER_FP32 = 2,
  BTS_STATIC_VARIANT_ROW_PLANE_COMPACT = 3,
  BTS_STATIC_VARIANT_ROW_PLANE_SMALL = 4,
  BTS_STATIC_VARIANT_ROW_GATHER_WIDE = 5,
  BTS_STATIC_VARIANT_DEPTH_SPLIT_LARGE = 6,
  BTS_STATIC_VARIANT_DEPTH_SPLIT_TINY = 7,
  BTS_STATIC_VARIANT_HALF_ROW_BIG = 8,
  BTS_STATIC_VARIANT_MC_WIDE = 9,
  BTS_STATIC_VARIANT_H_FUSE = 10,
};

enum BatchToSpaceCopyPath : uint8_t {
  BTS_PATH_DEPTH_32B_ALIGNED = 0,
  BTS_PATH_DEPTH_32B_PADDED = 1,
  BTS_PATH_DEPTH_512B_ALIGNED = 2,
  BTS_PATH_DEPTH_TINY_SPATIAL = 3,
  BTS_PATH_ROW_GATHER_FP32 = 4,
  BTS_PATH_WIDE_512B = 5,
  BTS_PATH_MASK_COMPACT = 6,
  BTS_PATH_H_FUSE = 7,
  BTS_PATH_GATHER_INTERLEAVE = 8,
  BTS_PATH_DEPTH_SPLIT_TINY = 9,
  BTS_PATH_ROW_PLANE_FP32 = 10,
  // MASK_COMPACT geometry with one full-UB buffer (no double buffer):
  // maximal chunk width -> minimal MTE op count (latency-bound shapes).
  BTS_PATH_MC_BIG = 11,
};

enum BatchToSpaceTilingMode : uint32_t {
  BTS_TILING_MODE_FULL = 0,
  BTS_TILING_MODE_ROW_GATHER_FP32_STATIC = 1,
  BTS_TILING_MODE_DEPTH_SPLIT_TINY_STATIC = 2,
  BTS_TILING_MODE_ROW_PLANE_STRIDED_STATIC = 3,
  BTS_TILING_MODE_DEPTH_SPLIT_LARGE_STATIC = 4,
  BTS_TILING_MODE_ROW_GATHER_WIDE_STATIC = 5,
  BTS_TILING_MODE_DEPTH_SPLIT_TINY_TUNING = 6,
  BTS_TILING_MODE_DEPTH_SPLIT_LARGE_TUNING = 7,
  BTS_TILING_MODE_ROW_PLANE_SMALL_STATIC = 8,
  // Wide fp16 sub-burst depth: fixed chunk width, one output row per core.
  BTS_TILING_MODE_MC_WIDE_STATIC = 9,
  // Wide fp16 burst-depth rows: mode-only header; the kernel owns the chunk
  // geometry (two cores per row, gap-DMA width chunks).
  BTS_TILING_MODE_HALF_ROW_BIG_STATIC = 10,
  BTS_TILING_MODE_ROW_PLANE_COMPACT_STATIC = 11,
  BTS_TILING_MODE_H_FUSE_STATIC = 12,
  // Half-row big-tile mover: two single-buffer halves per row sized to the
  // full UB, minimizing MTE op count on latency-bound wide shapes.
  BTS_TILING_MODE_MC_BIGTILE = 13,
};

struct BatchToSpaceTilingHeader {
  uint32_t mode;
};

using BatchToSpaceRowPlaneStridedTilingData = BatchToSpaceTilingHeader;
using BatchToSpaceRowGatherFp32TilingData = BatchToSpaceTilingHeader;
using BatchToSpaceRowPlaneCompactTilingData = BatchToSpaceTilingHeader;
using BatchToSpaceRowPlaneSmallTilingData = BatchToSpaceTilingHeader;
using BatchToSpaceRowGatherWideTilingData = BatchToSpaceTilingHeader;
using BatchToSpaceDepthSplitLargeTilingData = BatchToSpaceTilingHeader;
using BatchToSpaceDepthSplitTinyTilingData = BatchToSpaceTilingHeader;
using BatchToSpaceHalfRowBigTilingData = BatchToSpaceTilingHeader;
using BatchToSpaceHFuseTilingData = BatchToSpaceTilingHeader;

struct alignas(8) BatchToSpaceMcWideTilingData {
  uint32_t mode;
  uint32_t tileOW;
};

static_assert(sizeof(BatchToSpaceMcWideTilingData) % 8 == 0,
              "MC_WIDE static tiling data must be 8-byte aligned");

struct alignas(8) BatchToSpaceTuningData {
  uint32_t mode;
  uint32_t tileOW;
  uint32_t tileOH;
  uint32_t tileDepth;
  uint32_t tileElemAligned;
  uint32_t totalTasks;
  uint32_t activeCores;
};

static_assert(sizeof(BatchToSpaceTuningData) % 8 == 0,
              "BatchToSpaceTuningData must be 8-byte aligned");

using BatchToSpaceDepthSplitLargeTuningData = BatchToSpaceTuningData;
using BatchToSpaceDepthSplitTinyTuningData = BatchToSpaceTuningData;

struct alignas(8) BatchToSpaceTilingData {
  uint32_t mode;
  uint32_t height;
  uint32_t width;
  uint32_t depth;
  uint32_t outBatch;
  uint32_t outHeight;
  uint32_t outWidth;
  uint32_t blockSize;
  uint32_t cropTop;
  uint32_t cropLeft;
  uint32_t tileOW;
  uint32_t tileOH;
  uint32_t tileElemAligned;
  uint32_t totalTasks;
  uint32_t tileDepth;
  uint16_t activeCores;
  uint8_t path;
  uint8_t disableL2;
};

static_assert(sizeof(BatchToSpaceTilingData) % 8 == 0,
              "BatchToSpaceTilingData must be 8-byte aligned");
static_assert(alignof(BatchToSpaceTilingData) == 8,
              "BatchToSpaceTilingData alignment must be 8 bytes");
