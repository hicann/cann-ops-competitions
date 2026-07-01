// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// usedCoreNum removed: retrieved in kernel via GetBlockNum() per AscendC best practice.
// tileLength kept for struct padding: 3 × uint32_t = 12B → padded to 16B.
// Kernel ignores this field and computes tileLen locally; removing it causes
// mysterious regressions (T11 +0.82μs, T13 +0.94μs), possibly due to GM layout
// or GET_TILING_DATA alignment assumptions.
struct ErfTilingData {
    uint32_t totalLength;    // total elements across all cores
    uint32_t perCoreLength;  // elements per core
    uint32_t tileLength;     // UNUSED by kernel (computes locally), kept for 16B padding
};
