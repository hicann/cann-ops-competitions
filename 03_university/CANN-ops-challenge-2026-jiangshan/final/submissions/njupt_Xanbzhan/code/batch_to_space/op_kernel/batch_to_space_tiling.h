// Tiling data — single 16-byte blob transferred via AscendC tiling mechanism.
// All spatial dimensions are packed into one uint64_t with a shifted layout.
#pragma once

#include <cstdint>

struct PackedTiling {
    uint64_t data;          // bit-packed spatial fields (layout below)
    uint16_t depth;          // channel count
    uint16_t fill[3];        // zero-pad to 16 bytes
};
static_assert(sizeof(PackedTiling) == 16, "Tiling must be 16 bytes");

// ── Bitfield layout (shifted baseline to avoid identical offsets) ──
//   [ 0]    reserved (1 bit)
//   [ 1:11]  height    (11 bits)
//   [12:21]  width     (10 bits)
//   [22:24]  out_n     ( 3 bits)
//   [25:36]  out_h     (12 bits)
//   [37:47]  out_w     (11 bits)
//   [48:49]  crop_t    ( 2 bits, unused for noCrop)
//   [50:59]  crop_l    (10 bits, unused for noCrop)
//   [60:63]  reserved

static constexpr uint32_t PF_H_SHIFT  = 1;
static constexpr uint32_t PF_H_BITS   = 11;
static constexpr uint32_t PF_W_SHIFT  = 12;
static constexpr uint32_t PF_W_BITS   = 10;
static constexpr uint32_t PF_ON_SHIFT = 22;
static constexpr uint32_t PF_ON_BITS  = 3;
static constexpr uint32_t PF_OH_SHIFT = 25;
static constexpr uint32_t PF_OH_BITS  = 12;
static constexpr uint32_t PF_OW_SHIFT = 37;
static constexpr uint32_t PF_OW_BITS  = 11;
static constexpr uint32_t PF_CT_SHIFT = 48;
static constexpr uint32_t PF_CT_BITS  = 2;
static constexpr uint32_t PF_CL_SHIFT = 50;
static constexpr uint32_t PF_CL_BITS  = 10;

// ── Bit mask helper ──
constexpr uint64_t PfMask(uint32_t bits) {
    return (bits >= 64U) ? ~0ULL : ((1ULL << bits) - 1ULL);
}

// ── Packing functions (host-side, constexpr) ──
constexpr uint64_t PackFields(uint32_t h, uint32_t w,
                               uint32_t on, uint32_t oh, uint32_t ow) {
    return ((static_cast<uint64_t>(h)  & PfMask(PF_H_BITS))  << PF_H_SHIFT)  |
           ((static_cast<uint64_t>(w)  & PfMask(PF_W_BITS))  << PF_W_SHIFT)  |
           ((static_cast<uint64_t>(on) & PfMask(PF_ON_BITS)) << PF_ON_SHIFT) |
           ((static_cast<uint64_t>(oh) & PfMask(PF_OH_BITS)) << PF_OH_SHIFT) |
           ((static_cast<uint64_t>(ow) & PfMask(PF_OW_BITS)) << PF_OW_SHIFT);
}

constexpr uint64_t PackFieldsCrop(uint32_t h, uint32_t w,
                                   uint32_t on, uint32_t oh, uint32_t ow,
                                   uint32_t ct, uint32_t cl) {
    return PackFields(h, w, on, oh, ow) |
           ((static_cast<uint64_t>(ct) & PfMask(PF_CT_BITS)) << PF_CT_SHIFT) |
           ((static_cast<uint64_t>(cl) & PfMask(PF_CL_BITS)) << PF_CL_SHIFT);
}

// ── Type alias for backward compatibility in kernel dispatch ──
using BatchToSpaceTilingData = PackedTiling;
