#ifndef TILING_KEY_BATCH_TO_SPACE_H
#define TILING_KEY_BATCH_TO_SPACE_H

// dtype tiling keys
#define BATCH_TO_SPACE_TILING_KEY_FP16 1
#define BATCH_TO_SPACE_TILING_KEY_FP32 2

// kernel execution modes
#define BATCH_TO_SPACE_MODE_ROW_SCATTER 0U
#define BATCH_TO_SPACE_MODE_CHANNEL_FALLBACK 1U
// V3: block-granular interleave + ONE contiguous MTE3 write, for pixelBytes
// that are a multiple of 32B (C=16/32/64 fp16, C=8/16/32 fp32, ...). Generalized
// to any block_size; crop variant uses an owExpStart%block start phase.
#define BATCH_TO_SPACE_MODE_BLOCK_INTERLEAVE_NO_CROP 4U
#define BATCH_TO_SPACE_MODE_BLOCK_INTERLEAVE_CROP 5U

#endif  // TILING_KEY_BATCH_TO_SPACE_H
