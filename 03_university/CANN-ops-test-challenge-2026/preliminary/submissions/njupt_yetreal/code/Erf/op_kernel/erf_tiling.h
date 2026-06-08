#pragma once

#include <cstdint>

struct ErfTilingData {
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t blockTail;
    uint32_t usedCores;
};