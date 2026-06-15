#pragma once
#include <cstdint>

struct ErfTilingDesc {
    uint64_t total;
    uint32_t nCore;
    uint32_t chunk;
    uint32_t nBlk;
    uint32_t rem;
};