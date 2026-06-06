// Tiling鏁版嵁缁撴瀯瀹氫箟
#pragma once

#include <cstdint>

struct BtsRouteTile {
    uint32_t in_n;
    uint32_t in_h;
    uint32_t in_w;
    uint32_t in_c;
    uint32_t out_n;
    uint32_t out_h;
    uint32_t out_w;
    uint32_t out_c;
    uint32_t crop_y0;
    uint32_t crop_x0;
    uint32_t block_span;
    uint32_t element_count;
    uint32_t scratch_[4];
    uint32_t odd65_lut[128 * 65];  // 棰勮绠梘ather鍋忕Щ锛岀敤浜嶤ASE4
};
