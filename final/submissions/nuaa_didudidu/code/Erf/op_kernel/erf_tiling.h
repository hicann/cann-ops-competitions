#pragma once

#include <cstdint>

// Host 侧写入，Kernel 侧按同一布局读取；字段顺序不要随意调整。
struct ErfTilingData {
    uint64_t length;
    uint32_t usedCoreNum;
    uint32_t tileLength;
    uint64_t smallCoreDataNum;
    uint64_t bigCoreDataNum;
    uint32_t tailBlockNum;
    uint32_t tailNum;
};
