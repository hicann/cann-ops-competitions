// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

constexpr uint32_t ADDCMUL_MAX_DIM = 8;

// 精简后的 TilingData: 删去 kernel 未读的冗余字段(rows/coreNum, 核数走 SetBlockDim, 行数由 blockRows/remainder 推);
// 类型最小化(维度<=10000 用 uint16, 小计数 uint8, 步长/长度/块数保持 uint32);按对齐降序排布减少补齐 -> 减小 GET_TILING_DATA 拷贝开销
struct AddcmulTilingData {
    uint32_t totalLength;             // 输出总元素个数(可达 ~4e9)
    uint32_t tileLength;              // 单 tile 元素数
    uint32_t blockRows;              // 每核基础块数/行数(扁平路径可 >65535)
    uint32_t s0[ADDCMUL_MAX_DIM];     // input_data 各维步长(广播维=0, 可达 ~4e9)
    uint32_t s1[ADDCMUL_MAX_DIM];     // x1 各维步长
    uint32_t s2[ADDCMUL_MAX_DIM];     // x2 各维步长
    uint16_t yDim[ADDCMUL_MAX_DIM];   // 输出各维大小(左补1, <=10000)
    uint16_t lastDim;                 // 最后一维长度(<=10000)
    uint16_t last0;                   // input_data 最后一维(1=按行广播)
    uint16_t last1;                   // x1 最后一维
    uint16_t last2;                   // x2 最后一维
    uint8_t remainder;                // 前 remainder 个核各多 1 块/行(<coreNum<=~48)
    uint8_t bcast;                    // 0=三输入同形(纯逐元素), 1=需要广播
};
