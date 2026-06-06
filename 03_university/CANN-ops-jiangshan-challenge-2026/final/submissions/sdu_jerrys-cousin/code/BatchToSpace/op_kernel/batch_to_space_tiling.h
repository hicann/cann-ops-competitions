#pragma once

#include <cstdint>

constexpr uint32_t BTS_MODE_ROW_TILE = 0;
constexpr uint32_t BTS_MODE_FLAT_TILE = 1;
constexpr uint32_t BTS_MODE_LARGE_D = 2;
constexpr uint32_t BTS_MODE_GATHER_ROW = 3;
constexpr uint32_t BTS_MODE_P10_OH_GATHER_TILE = 4;
constexpr uint32_t BTS_MODE_P10_UB_PACK_TILE = 5;
constexpr uint32_t BTS_MODE_P1_ROW_FAST = 6;
constexpr uint32_t BTS_MODE_P2_GATHER_FAST = 7;
constexpr uint32_t BTS_MODE_P3_ROW_FAST = 8;
constexpr uint32_t BTS_MODE_B4_W_PACK_TILE = 9;
constexpr uint32_t BTS_MODE_P1_B2_DIRECT = 10;
constexpr uint32_t BTS_MODE_P3_B2_DIRECT = 11;
constexpr uint32_t BTS_MODE_P4_B2_DIRECT = 12;
constexpr uint32_t BTS_MODE_P6_B2_DIRECT = 13;
constexpr uint32_t BTS_MODE_P7_D_DIRECT = 14;
constexpr uint32_t BTS_MODE_P7_FLAT_COPY = 15;
constexpr uint32_t BTS_MAX_GATHER_IDX_ELEMS = 16510;  // 数据点 5: OW(254) * D(65)

// 兼容历史文件名中的 V71/V73 后缀，实际语义以 FAST 名称为准。
constexpr uint32_t BTS_MODE_P1_ROW_V71 = BTS_MODE_P1_ROW_FAST;
constexpr uint32_t BTS_MODE_P2_GATHER_V73 = BTS_MODE_P2_GATHER_FAST;
constexpr uint32_t BTS_MODE_P3_ROW_V73 = BTS_MODE_P3_ROW_FAST;

struct BatchToSpaceTilingData {
    uint32_t N;
    uint32_t H;
    uint32_t W;
    uint32_t D;
    uint32_t block;
    uint32_t ct;
    uint32_t cl;
    uint32_t OH;
    uint32_t OW;
    uint32_t totalUnits;   // 当前 mode 的任务总数
    uint32_t usedCoreNum;  // 实际启用核数
    uint32_t tileW;        // ROW 为源 W tile，B4 为输出 OW tile，FLAT 为 H*W tile，P10 为 tileOH，P7_COPY 为元素块
    uint32_t aligned;      // 非 0 表示 D * sizeof(T) 32B 对齐；2 表示 P4 exact 快路径
    uint32_t dTileLen;     // LARGE_D/P6 direct 的 D 方向切分长度，其余路径等于 D
    uint32_t gatherDepth;  // GATHER/P10/B4/P4 队列深度
};

struct BatchToSpaceTilingDataBuf {
    BatchToSpaceTilingData base;
    uint32_t reservedIdx[BTS_MAX_GATHER_IDX_ELEMS];
};
