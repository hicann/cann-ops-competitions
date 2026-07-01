// TilingKey constants for compile-time feature-family dispatch.
// Compact integer keys avoid hex/ULL/uint64 compatibility issues with TILING_KEY_IS.
// Each path is unique per dtype, so no need for high-byte dtype encoding.
#pragma once

// ── fp32 paths ──
#define TK_F32_NC_DEEP   1
#define TK_F32_CROP_THIN   2
#define TK_F32_NC_MID   3
#define TK_F32_NC_SHORT   4

// ── fp16 paths ──
#define TK_F16_CROP_ODD_DEPTH   5
#define TK_F16_NC_HUGE_DEPTH   6
#define TK_F16_NC_EXTREME_DEPTH   7
#define TK_F16_NC_WIDE_ROW   8
#define TK_F16_CROP_BS4_WIDE   9
#define TK_F16_NC_SHORT_WIDTH  10
