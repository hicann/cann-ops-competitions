/*!
 * \file hard_swish_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _HARDSWISH_TILING_DATA_H_
#define _HARDSWISH_TILING_DATA_H_

#include <cstdint>

struct HardSwishTilingData {
    int64_t totalNum = 0;
    int64_t blockFactor = 0;
    int64_t ubFactor = 0;
};

#endif
