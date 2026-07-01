#ifndef ERF_TILING_H
#define ERF_TILING_H

#include <cstdint>

struct ErfTilingData {
    uint32_t totalLength;
    uint32_t tileLength;
    uint32_t coreNum;
    uint32_t mode;
};

#endif // ERF_TILING_H