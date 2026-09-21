#pragma once
#include <cstdint>

// Host normalizes N*C planes and attributes before launch. All offsets are int64.
struct PoolTiling {
    int64_t planes, height, width, outHeight, outWidth;
    int64_t kernelH, kernelW, strideH, strideW, padH, padW;
};
