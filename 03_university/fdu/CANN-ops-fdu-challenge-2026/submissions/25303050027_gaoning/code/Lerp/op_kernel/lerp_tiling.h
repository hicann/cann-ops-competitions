// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// Ascend C DataCopy常用的最小对齐粒度：32B。
constexpr uint32_t LERP_DATA_BLOCK_BYTES = 32;
// Host侧分核的基础对齐粒度，保证每个core起始位置尽量落在较友好的边界上。
constexpr uint32_t LERP_CACHE_LINE_BYTES = 64;
// 单core单次处理的默认元素数。比已通过版1024更大，可减少大shape下的循环和队列开销。
constexpr uint32_t LERP_DEFAULT_TILE_LENGTH = 8192;
// 保持单buffer顺序执行，兼容性优先，不引入double buffer风险。
constexpr uint32_t LERP_BUFFER_NUM = 1;
// 安全版kernel仍使用start、end、out三个LocalTensor，避免VECIN直接写GM的不稳定问题。
constexpr uint32_t LERP_LOCAL_BUFFER_NUM = 3;
// 预留部分UB空间给运行时和队列管理，避免把UB打满。
constexpr uint64_t LERP_UB_RESERVED_BYTES = 2048;
// 小shape减少启动core数量，降低调度开销；大shape仍会自动铺开更多core。
constexpr uint32_t LERP_MIN_BYTES_PER_CORE = 2048;

constexpr uint32_t LERP_MODE_NORMAL = 0;
constexpr uint32_t LERP_MODE_COPY_START = 1;
constexpr uint32_t LERP_MODE_COPY_END = 2;

struct LerpTilingData {
    uint64_t length;
    uint64_t perCoreElements;
    uint32_t tileLength;
    uint32_t mode;
    float weight;
};
