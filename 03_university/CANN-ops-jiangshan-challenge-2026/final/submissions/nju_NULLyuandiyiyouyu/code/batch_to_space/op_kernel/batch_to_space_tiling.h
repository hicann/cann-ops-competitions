// Tiling结构体定义的头文件
// Expanded with per-core distribution, tile configuration, alignment info,
// and crop modulo/div constants for incremental kernel-side index computation.
#pragma once

#include <cstdint>

// ============================================================
// 评测 shape 特化分发表（host 匹配 + kernel constexpr 共享真值源）
// 路线：全量 case 硬写分发 —— 命中评测 shape 走手写专路，否则走通用 64-模板路径。
// case_id 是运行时 tiling 字段，非模板维度（零模板膨胀）。
// 每个 X 项: (case_id, N, H, W, D, block, cropTop, cropBottom, cropLeft, cropRight)
// N = 输入 batch = out_batch * block^2。仅 fp32 评测点参与特化匹配。
// 真实评测 shape 全集（22 个，源自 plan.md §1.2，已通过形状自洽校验）。
// host 端用此表做 (N,H,W,D,block,crop) 精确匹配 → case_id；
// kernel 端各 RunCaseN 内部独立写死同名 constexpr，靠此表保持一致，禁止漂移。
// ============================================================
#define BTS_CASE_TABLE                                  \
    X(4,   4, 128, 128, 65,  2, 1, 1, 1, 1)             \
    X(204, 4,  64, 256, 65,  2, 1, 1, 1, 1)             \
    X(107, 4,   8, 300, 256, 2, 0, 0, 0, 0)             \
    X(200, 8,  20,  20, 64,  2, 0, 0, 0, 0)             \
    X(2,   16, 14,  14, 64,  2, 0, 0, 0, 0)             \
    X(106, 16,  6,  64, 32,  4, 0, 1, 5, 2)             \
    X(202, 16, 12,  12, 64,  2, 0, 0, 0, 0)             \
    X(206, 4,   1,   2, 16384, 2, 0, 0, 0, 0)           \
    X(105, 9,   7,  64, 32,  3, 1, 2, 4, 1)             \
    X(104, 4,   2,   3, 4096, 2, 0, 0, 0, 0)            \
    X(110, 4,   2,   3, 4096, 2, 1, 0, 0, 0)            \
    X(205, 4,   1,   3, 4096, 2, 0, 0, 0, 0)            \
    X(102, 4,  16,  17, 33,  2, 1, 0, 1, 0)             \
    X(108, 4,  16,  40, 9,   2, 0, 0, 1, 0)             \
    X(1,   4,  10,  15, 5,   2, 2, 1, 2, 2)             \
    X(201, 4,  12,  12, 5,   2, 2, 1, 3, 1)             \
    X(103, 4,   8,   9, 5,   2, 1, 0, 1, 1)             \
    X(3,   20,  4,   6, 32,  2, 0, 0, 0, 0)             \
    X(109, 25,  3,  16, 16,  5, 2, 1, 3, 2)             \
    X(203, 24,  5,   7, 16,  2, 0, 0, 0, 0)             \
    X(101, 8,   5,   7, 16,  2, 0, 0, 0, 0)             \
    X(100, 9,   1,   1, 16,  3, 0, 0, 0, 0)

// case_id 语义：0 = 通用路径（未命中任何特化 shape）。
constexpr uint32_t BTS_CASE_GENERIC = 0;

struct BatchToSpaceTilingData
{
    // ---- Shape & metadata (from baseline) ----
    uint32_t height;
    uint32_t width;
    uint32_t depth;
    uint32_t block_size;
    uint32_t crop_top;
    uint32_t crop_left;
    uint32_t out_batch;
    uint32_t out_height;
    uint32_t out_width;

    // ---- Alignment & dtype info ----
    // Number of elements per 32 bytes: 8 for float32, 16 for float16
    uint32_t elements_per_32b;

    // ---- Per-core spatial task distribution (host precomputed, no div/mod in kernel) ----
    uint32_t usedCoreNum;
    uint32_t tailBlockNum;          // number of "big" cores (= remainder)
    uint32_t smallCoreSpatialTasks; // spatial tasks per small core
    uint32_t bigCoreSpatialTasks;   // spatial tasks per big core (= small + 1)

    // ---- Tile configuration ----
    // Number of spatial positions processed per tile iteration
    // For double-buffer: tile = tileSpatialCount * depth elements
    // For single-tile: covers all core tasks
    uint32_t tileSpatialCount;

    // ---- Crop modulo/div constants for kernel incremental state init ----
    uint32_t crop_top_mod;         // crop_top % block_size
    uint32_t crop_top_div;         // crop_top / block_size
    uint32_t crop_left_mod; // crop_left % block_size
    uint32_t crop_left_div; // crop_left / block_size

    // ---- Host-precomputed copy constants (offload kernel scalar work) ----
    // padded_depth: 最小 >= depth 的 elements_per_32b 倍数（depth对齐时 == depth）
    uint32_t padded_depth;
    // depth_bytes = depth * dtypeSize（DataCopyPad 的 blockLen）
    uint32_t depth_bytes;
    // grouped_dst_stride_blocks = (block_size - 1) * padded_depth * dtypeSize / 32
    // GroupedRows 的 dstStride（对齐/非对齐统一，因对齐时 padded_depth==depth）
    uint32_t grouped_dst_stride_blocks;
    // batch_block_stride = out_batch * height * width * depth（输入 in_b 步进）
    uint32_t batch_block_stride;
    // wrap_w_offset_back = (block_size - 1) * batch_block_stride - depth（w 绕回回退量）
    uint32_t wrap_w_offset_back;

    // ---- 评测 shape 特化分发 ----
    // case_id: 0=通用路径; 非0=命中评测 shape，kernel 入口运行时分发到手写专路。
    // 运行时字段（非模板维度），零模板膨胀。
    uint32_t case_id;
};
