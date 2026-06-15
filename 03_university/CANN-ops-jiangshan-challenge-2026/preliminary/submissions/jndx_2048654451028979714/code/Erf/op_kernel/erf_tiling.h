// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct ErfTilingData {
    uint32_t totalLength;   // 输入张量总元素数
    uint32_t coreLength;    // 每个核处理的元素数（除最后一个核外，已 32B 对齐）
    uint32_t tileLen;       // 低31位: tile长度, 高1位: 小shape rational模式
};
