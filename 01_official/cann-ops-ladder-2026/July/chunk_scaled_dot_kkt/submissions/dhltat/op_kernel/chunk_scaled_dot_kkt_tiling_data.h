/*!
 * \file chunk_scaled_dot_kkt_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _CHUNKSCALEDDOTKKT_TILING_DATA_H_
#define _CHUNKSCALEDDOTKKT_TILING_DATA_H_

#include <cstdint>

#if __has_include("tiling/tiling_api.h")
#include "tiling/tiling_api.h"
#else
#include "tiling_api.h"
#endif

// Keep this as a byte-identical prefix of ChunkScaledDotKktTilingData. Manual
// Cube kernels do not consume the comparatively large TCubeTiling payload.
struct ChunkScaledDotKktManualTilingData {
    uint32_t seqLen = 0;
    uint32_t groupHeads = 0;
    uint32_t keyDim = 0;
    uint32_t headPerGroup = 1;
    uint32_t taskNum = 0;
};

struct ChunkScaledDotKktTilingData {
    uint32_t seqLen = 0;
    uint32_t groupHeads = 0;
    uint32_t keyDim = 0;
    uint32_t headPerGroup = 1;
    uint32_t taskNum = 0;
    uint32_t heads = 0;
    uint32_t chunkNum = 0;
    uint32_t chunkSize = 64;
    uint32_t taskCoreNum = 1;
    uint32_t blockFactor = 1;
    uint32_t workspacePerCore = 0;
    TCubeTiling matmulTiling;
};

#endif
