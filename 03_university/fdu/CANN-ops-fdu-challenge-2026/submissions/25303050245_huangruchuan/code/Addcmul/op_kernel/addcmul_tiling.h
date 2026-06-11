#pragma once
#ifndef ADDCMUL_TILING_H
#define ADDCMUL_TILING_H
#include <cstdint>
struct AddcmulTilingData {
    uint32_t tileNum;
    uint32_t ALIGN_NUM;
    uint32_t block_size;
    uint32_t aivNum;
    uint32_t core_size;
    uint32_t core_remain;
    uint32_t total_length;
    uint32_t input_data_length;
    uint32_t x1_length;
    uint32_t x2_length;
};
#endif