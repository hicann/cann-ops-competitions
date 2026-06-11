// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct AddcmulTilingData {
    uint32_t length;
    uint32_t blockDim;
    uint32_t tileLength;
    uint32_t contiguous;
    uint32_t rank;
    uint32_t inputKind;       // 0=FULL, 1=SCALAR, 2=INNER_CONTIG, 3=INNER_BCAST, 4=GENERAL
    uint32_t x1Kind;
    uint32_t x2Kind;
    uint32_t inputContigLen;  // For INNER_CONTIG: inner-contig run length; for INNER_BCAST: inner-bcast block size
    uint32_t x1ContigLen;
    uint32_t x2ContigLen;
    uint32_t rowLen;          // binding broadcast run K (min constrained contigLen); 0 if none
    uint32_t coreStride;      // K-aligned per-core element count for broadcast paths; 0 = use ceil(length/blockDim)
    uint32_t fastBcast;       // 1 = all constrained operands are pure row-vectors [1,K]/[K] -> row-batched fast path
    uint32_t inputStride[16];
    uint32_t x1Stride[16];
    uint32_t x2Stride[16];
    uint32_t valueStride[16];
    uint32_t outputShape[16];
};
