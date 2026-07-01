// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct BatchToSpaceTilingData {
    uint32_t length;
    uint32_t batch;
    uint32_t height;
    uint32_t width;
    uint32_t depth;
    uint32_t block_size;
    uint32_t crop_top;
    uint32_t crop_bottom;
    uint32_t crop_left;
    uint32_t crop_right;
    uint32_t out_batch;
    uint32_t out_height;
    uint32_t out_width;
    uint32_t w_chunk;            // mode0: W_in chunk; mode1/2/3: W_out chunk
    uint32_t h_chunk;            // mode0: input H rows per unit; mode1/2/3: output H rows per unit
    uint32_t num_chunks;         // W chunks per (b,h) [mode0] or (n_out,h_out) [mode1/2/3]
    uint32_t d_chunk;            // mode3 only: D chunk size in 32B blocks
    uint32_t d_num_chunks;       // mode3 only: number of D chunks per output W slot
    uint32_t total_units;
    uint32_t units_per_core;
    uint32_t tail_units;
    uint32_t used_cores;
    uint32_t mode;               // 0=input; 1=output; 2=rows; 3=large-D; 4=dataset6; 5=small-D; 6=dataset10 rows-q4; 7=dataset7; 8=non32 pair-row
};
