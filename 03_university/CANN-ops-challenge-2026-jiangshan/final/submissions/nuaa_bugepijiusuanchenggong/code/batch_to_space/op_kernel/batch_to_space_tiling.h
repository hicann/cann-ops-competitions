// Tiling data for BatchToSpace.
#pragma once

#include <cstdint>

#define BTS_MODE_F16_B1_LINEAR          1U
#define BTS_MODE_F16_B1_SLAB            2U
#define BTS_MODE_F16_B1_ROW2D           3U
#define BTS_MODE_F16_B2_STRIPE          4U
#define BTS_MODE_F16_B3_STRIPE          5U
#define BTS_MODE_F16_B4_STRIPE          6U
#define BTS_MODE_F16_B5_STRIPE          7U
#define BTS_MODE_F16_B6_STRIPE          8U
#define BTS_MODE_F16_B7_STRIPE          9U
#define BTS_MODE_F16_B8_STRIPE          10U
#define BTS_MODE_F16_GENERIC_STRIPE     11U

#define BTS_MODE_F32_B1_LINEAR          21U
#define BTS_MODE_F32_B1_SLAB            22U
#define BTS_MODE_F32_B1_ROW2D           23U
#define BTS_MODE_F32_B2_STRIPE          24U
#define BTS_MODE_F32_B3_STRIPE          25U
#define BTS_MODE_F32_B4_STRIPE          26U
#define BTS_MODE_F32_B5_STRIPE          27U
#define BTS_MODE_F32_B6_STRIPE          28U
#define BTS_MODE_F32_B7_STRIPE          29U
#define BTS_MODE_F32_B8_STRIPE          30U
#define BTS_MODE_F32_GENERIC_STRIPE     31U

#define BTS_MAX_COMPACT_BS              8U
#define BTS_MAX_COMPACT_PHASE           64U

// Host-selected minor strategy inside a major MODE_KIND.
// MODE_KIND keeps dtype/block-size specialization; strategy avoids over-expanding
// template modes while still moving path decisions out of hot loops.
#define BTS_STRATEGY_DEFAULT              0U
#define BTS_STRATEGY_B1_DIRECT            1U
#define BTS_STRATEGY_B1_ROW2D_PIPELINE    2U
#define BTS_STRATEGY_COMPACT_SMALL        3U
#define BTS_STRATEGY_COMPACT_STRIPE       4U
#define BTS_STRATEGY_COMPACT_ROWASM       5U
#define BTS_STRATEGY_GENERIC              6U
#define BTS_STRATEGY_COMPACT_TINY_PHASE   7U
#define BTS_STRATEGY_B2_NOCROP_FAST       8U

struct BatchToSpaceTilingData {
    // Input shape: [batch, height, width, depth], NHWC.
    uint32_t batch;
    uint32_t height;
    uint32_t width;
    uint32_t depth;

    // Output shape: [out_batch, out_height, out_width, depth].
    uint32_t out_batch;
    uint32_t out_height;
    uint32_t out_width;
    uint32_t total_elems;
    uint32_t total_groups;

    // Attributes and selected mode.
    uint32_t block_size;
    uint32_t crop_top;
    uint32_t crop_bottom;
    uint32_t crop_left;
    uint32_t crop_right;
    uint32_t mode;
    uint32_t strategy;

    // Copy parameters.
    uint32_t dtype_size;
    uint32_t depth_bytes;
    uint32_t depth_aligned_bytes;
    uint32_t tile_elems;     // B1 linear/slab copy: elements per tile.
    uint32_t tile_rows;      // B1 row2D: output rows per group.
    uint32_t tile_iw;        // Stripe modes: input width positions per group.
    uint32_t ub_bytes;       // Per-buffer UB size, 32B aligned.

    // Host-side work distribution, inspired by the compact Erf tiling style.
    // The work unit is selected by mode:
    //   B1_LINEAR      : output elements
    //   other modes    : logical copy groups
    // This removes per-core ceil-div from the hot kernel path.
    uint32_t used_core_num;
    uint32_t big_core_num;
    uint32_t big_core_work_num;
    uint32_t small_core_work_num;

    // Compact phase scheduling for block_size <= 8.
    uint32_t h_begin[BTS_MAX_COMPACT_BS];
    uint32_t h_count[BTS_MAX_COMPACT_BS];
    uint32_t w_begin[BTS_MAX_COMPACT_BS];
    uint32_t w_count[BTS_MAX_COMPACT_BS];
    uint32_t tiles_iw[BTS_MAX_COMPACT_BS];
    uint32_t phase_prefix[BTS_MAX_COMPACT_PHASE + 1U];
    uint32_t compact_phase_num;
    uint32_t compact_uniform;
    uint32_t compact_phase_group_size;

    // Compact row assembly strategy. When enabled, tile_iw stores output-width
    // positions per row tile instead of input-width positions per phase stripe.
    // This path is selected only for depth_bytes aligned to 32B, so UB-side
    // strided input assembly is safe and output rows can be written contiguously.
    uint32_t compact_row_assembly;

    // Host-precomputed row-assembly decode constants. These remove repeated
    // ceil-divs from MakeCompactRowJob() and give a cheap full-width fast path
    // when compact_row_tiles_per_width == 1.
    uint32_t compact_row_tiles_per_width;
    uint32_t compact_row_groups_per_batch;
};
