// Tiling data for erfinv.
#pragma once

#include <cstdint>

constexpr uint32_t ERFINV_BUFFER_NUM = 2;
constexpr uint32_t ERFINV_UB_ALIGN = 8;       // 8 float32 elements = 32 bytes.
constexpr uint32_t ERFINV_UB_LENGTH_LIMIT = 4096;

struct ErfinvTilingData {
    uint32_t length;
    uint32_t blockLength;
    uint32_t ubLength;
};
