#pragma once
#include <cstdint>

// ==========================================
// [功能块] Tiling 数据结构定义
// ==========================================
struct ErfTilingData {
    uint32_t totalLength;   // 张量数据的总元素个数
    uint32_t coreDataNum;   // 分配给单个AI Core需要处理的最大数据元素个数
    uint32_t tileDataNum;   // 单个AI Core单次处理(送入UB)的数据块大小(元素个数)
    uint32_t usedCoreNum;   // 实际调度使用的AI Core数量
};
