// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// Keep host UB model and kernel InitBuffer count in lockstep.
// ver102 uses two VECCALC scratch tiles (t / p) with 1 stagger.
constexpr int32_t ERF_CONFIG_FORCE_SCRATCH_TILES = 2;

// 紧凑布局（共 16B，无 padding）：
//   uint32 totalLength       // 4
//   uint32 blockLength       // 4
//   uint32 tileLength        // 4
//   uint16 inputQueuePadding // 2  (0..256)
//   uint8  pipelineMode      // 1  (0=simple, 1=prefetch)
//   uint8  reserved          // 1
struct ErfTilingData {
  uint32_t totalLength;
  uint32_t formerNum;    // 增加: 整核数量
  uint32_t formerLength; // 增加: 整核长度
  uint32_t tailLength;   // 增加: 尾核长度
  uint32_t tileLength;
  uint16_t inputQueuePadding;
  uint8_t pipelineMode;
  uint8_t reserved;
};
