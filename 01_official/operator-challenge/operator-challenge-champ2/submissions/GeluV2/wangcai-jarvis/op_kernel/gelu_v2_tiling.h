#pragma once

#include <cstdint>

constexpr uint32_t GELU_V2_TINY_TILE = 1024;
constexpr uint32_t GELU_V2_SMALL_TILE = 4096;
constexpr uint32_t GELU_V2_DEFAULT_TILE = 6144;
constexpr uint32_t GELU_V2_FP32_FAST_TILE = 7168;
constexpr uint32_t GELU_V2_FP16_LARGE_TILE = 8192;
constexpr uint32_t GELU_V2_BF16_LARGE_TILE = 7168;
constexpr uint32_t GELU_V2_FP16_MEDIUM_LENGTH = 4U * 1024U * 1024U;
constexpr uint32_t GELU_V2_BF16_MEDIUM_LENGTH = 1U * 1024U * 1024U;
constexpr uint32_t GELU_V2_FP32_FAST_TANH_LENGTH = 4U * 1024U * 1024U;
constexpr uint32_t GELU_V2_FP32_DIV_LENGTH = 4U * 1024U * 1024U;
constexpr uint32_t GELU_V2_FP32_CORE_CAP_LENGTH = 32U * 1024U * 1024U;
constexpr uint32_t GELU_V2_FP32_48M_LENGTH = 48U * 1024U * 1024U;
constexpr uint32_t GELU_V2_FP32_ULTRA_LENGTH = 64U * 1024U * 1024U;
constexpr uint32_t GELU_V2_SMALL_MIN_ELEMS_PER_CORE = 4096;
constexpr uint32_t GELU_V2_DENSE_MIN_ELEMS_PER_CORE = 2048;
constexpr uint32_t GELU_V2_FP32_32M_CORE_CAP = 32;
constexpr uint32_t GELU_V2_FP32_50M_CORE_CAP = 40;
constexpr uint32_t GELU_V2_FP32_64M_CORE_CAP = 32;

struct GeluV2TilingData {
    uint32_t length;
    uint32_t blockLength;
    uint32_t tileSize;
    uint32_t approximate;
    uint32_t dtype;
};
