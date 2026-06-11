// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct AddcmulTilingData {
    // Common parameters
    int32_t dtypeSize;       // sizeof(dtype): 1(int8), 2(float16), 4(float32/int32)
    uint32_t totalLength;    // total elements in broadcast output

    // Multi-core split parameters
    uint32_t blockFormer;    // base elements per core (256B aligned)
    uint32_t blockTail;      // tail block elements

    // UB tile parameters
    uint32_t ubFormer;       // UB tile elements (256B aligned)
    uint32_t ubTail;         // last UB tile in main block
    uint32_t loopNum;        // UB loop count for main block
    uint32_t tailLoopNum;    // UB loop count for tail block
    uint32_t tailUbTail;     // last UB tile in tail block

    // Broadcast support
    // When needBroadcast=1, per-input total elements may differ from totalLength.
    // For each input, the GM offset wraps at its own total element count.
    int32_t needBroadcast;   // 0=linear, 1=broadcast needed
    uint32_t x1Total;        // x1 actual total elements (pre-broadcast)
    uint32_t x2Total;        // x2 actual total elements (pre-broadcast)
    uint32_t inTotal;        // input_data actual total elements (pre-broadcast)
};
