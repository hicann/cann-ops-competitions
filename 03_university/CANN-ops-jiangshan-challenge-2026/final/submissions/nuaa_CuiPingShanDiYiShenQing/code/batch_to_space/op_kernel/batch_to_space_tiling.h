#ifndef BTS_REWORKED_TILING_PACKET_H_
#define BTS_REWORKED_TILING_PACKET_H_

#include <stdint.h>

struct BtsShapePacket {
    uint32_t src_n;
    uint32_t src_h;
    uint32_t src_w;
    uint32_t src_c;
    uint32_t dst_n;
    uint32_t dst_h;
    uint32_t dst_w;
    uint32_t dst_c;
    uint32_t crop_t;
    uint32_t crop_b;
    uint32_t crop_l;
    uint32_t crop_r;
    uint32_t block;
    uint32_t dst_elems;
};

#endif
