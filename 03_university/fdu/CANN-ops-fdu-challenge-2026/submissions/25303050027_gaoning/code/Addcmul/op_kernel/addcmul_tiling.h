// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

constexpr uint32_t ADDCMUL_MAX_DIMS = 32;
constexpr uint32_t ADDCMUL_DATA_BLOCK_BYTES = 32;
constexpr uint32_t ADDCMUL_CACHE_LINE_BYTES = 64;
constexpr uint32_t ADDCMUL_DEFAULT_TILE_LENGTH = 32768;
constexpr uint32_t ADDCMUL_LOCAL_BUFFER_NUM = 4;      // input、x1、x2、output，普通单缓冲路径
constexpr uint32_t ADDCMUL_LINEAR_DOUBLE_BUFFER_NUM = 7; // 同形状连续浮点双缓冲路径：3输入*2 + 1输出
constexpr uint64_t ADDCMUL_UB_RESERVED_BYTES = 8 * 1024;

// kernel侧访问模式：host侧提前判断，减少kernel里每个元素重复做复杂广播判断。
constexpr uint32_t ADDCMUL_MODE_GENERIC = 0;      // 任意广播，需要按多维stride递推offset
constexpr uint32_t ADDCMUL_MODE_LINEAR = 1;       // 广播后仍可按输出线性下标连续访问
constexpr uint32_t ADDCMUL_MODE_SCALAR = 2;       // 输入只有1个元素，offset恒为0
constexpr uint32_t ADDCMUL_MODE_SEGMENT = 3;      // 一个连续片段在外层/内层重复，offset=(idx/repeat)%dataLength

struct AddcmulTilingData {
    uint64_t length;            // 广播后输出元素总数
    uint64_t perCoreElements;   // 每个AI Vector core处理的元素数，按cache line对齐
    uint32_t dimNum;            // 输出维度数，最大 ADDCMUL_MAX_DIMS
    uint32_t sameShape;         // 1: input/x1/x2均可线性连续访问
    uint32_t tileLength;        // 连续向量路径单次处理元素数，按32B对齐
    uint32_t inputMode;         // input_data访问模式
    uint32_t x1Mode;            // x1访问模式
    uint32_t x2Mode;            // x2访问模式
    uint32_t reserved;
    uint64_t inputRepeat;       // SEGMENT模式下，单个连续源片段在输出中每个元素重复的内层长度
    uint64_t x1Repeat;
    uint64_t x2Repeat;
    uint64_t inputDataLength;   // SEGMENT模式下，连续源片段元素数
    uint64_t x1DataLength;
    uint64_t x2DataLength;
    uint64_t outShape[ADDCMUL_MAX_DIMS];
    uint64_t inputStrides[ADDCMUL_MAX_DIMS];
    uint64_t x1Strides[ADDCMUL_MAX_DIMS];
    uint64_t x2Strides[ADDCMUL_MAX_DIMS];
};
