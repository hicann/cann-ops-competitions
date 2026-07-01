#pragma once

#include <cstdint>

constexpr uint32_t BTS_FLAG_BLOCK_SIZE_2 = 1U << 0;
constexpr uint32_t BTS_FLAG_CROP_ZERO    = 1U << 1;
constexpr uint32_t BTS_FLAG_BS1_COPY     = 1U << 2;

constexpr uint32_t BTS_MODE_NORMAL = 0;
constexpr uint32_t BTS_MODE_SMALL_DEPTH = 1;
constexpr uint32_t BTS_MODE_LARGE_DEPTH_SPLIT = 2;
constexpr uint32_t BTS_MODE_BS2_CROP0_ROW = 3;
constexpr uint32_t BTS_MODE_BS2_CROP0_ROW_DEPTH_SPLIT = 4;
constexpr uint32_t BTS_MODE_BS2_CROP0_PAIR_FULL_DEPTH = 5;
constexpr uint32_t BTS_MODE_PHASE_STRIDED = 6;
constexpr uint32_t BTS_MODE_OUTPUT_ROW_PACK = 7;

enum TestCase : uint32_t {
    TEST_CASE_1  = 0,
    TEST_CASE_2  = 1,
    TEST_CASE_3  = 2,
    TEST_CASE_4  = 3,
    TEST_CASE_5  = 4,
    TEST_CASE_6  = 5,
    TEST_CASE_7  = 6,
    TEST_CASE_8  = 7,
    TEST_CASE_9  = 8,
    TEST_CASE_10 = 9,
    TEST_CASE_GENERAL = 255,
};

struct BatchToSpaceTilingData {
    uint32_t batch;
    uint32_t height;
    uint32_t width;
    uint32_t depth;
    uint32_t outBatch;
    uint32_t outHeight;
    uint32_t outWidth;
    uint32_t cropTop;
    uint32_t cropBottom;
    uint32_t cropLeft;
    uint32_t cropRight;
    uint32_t blockSize;
    uint32_t blockSizeSquared;
    uint64_t totalOutPoints;
    uint64_t totalOutElements;
    uint64_t totalTasks;
    uint64_t totalOutBytes;
    uint32_t dtypeSize;
    uint32_t coreNum;
    uint32_t copyDepth;
    uint32_t depthSplitNum;
    uint32_t colChunk;
    uint32_t smallDepthPackPoints;
    uint32_t rowTileWidth;
    uint32_t mode;
    uint32_t flags;
    TestCase testCase;

    // Pre-computed constants for kernel fast path
    uint32_t fullRowBytes;        // depth * dtypeSize (bytes per C-dimension)
    uint32_t phaseShift;          // cropLeft % blockSize
    uint32_t colCountBw0;         // output columns for bw=0 phase
    uint32_t colCountBw1;         // output columns for bw=1 phase
    uint32_t alignedDepth;        // depth aligned down to 32B boundary (in elements)
    uint32_t depthTail;           // remainder depth elements (non-32B-aligned tail)
    uint32_t totalRows;           // outBatch * outHeight
    uint32_t coreStride;          // columns per core for C-split mode
    uint32_t reserved1;
    uint64_t length;
};