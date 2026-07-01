// Tiling data shared by host tiling and AI Core kernel.
#pragma once

#include <cstdint>

struct ErfTilingData {
    uint32_t length;
    uint32_t blockNum;
};
