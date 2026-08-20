#include "kernel_operator.h"
#include "kernel.h"
#include "unpack_2_0.h"
#include "unpack_2_1.h"
#include "unpack_3.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void unpack(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    InitSocState();
    GET_TILING_DATA(tiling_data, tiling);
    int32_t D = tiling_data.D;
    int32_t H = tiling_data.H;
    int32_t W = tiling_data.W;
    if (TILING_KEY_IS(1)) {// 2_0 small loop
        if constexpr (is_one_of_v<DTYPE_INPUT, float, int32_t>) {
            unpack_2_0_small<float>(input, output, tiling_data.coreRows, tiling_data.tailRows, tiling_data.H, tiling_data.W);
        } else if constexpr (is_one_of_v<DTYPE_INPUT, half, int16_t, bfloat16_t>) {
            unpack_2_0_small<half>(input, output, tiling_data.coreRows, tiling_data.tailRows, tiling_data.H, tiling_data.W);
        } else if constexpr (is_one_of_v<DTYPE_INPUT, int8_t, uint8_t, bool>) {
            unpack_2_0_small<half>(input, output, tiling_data.coreRows, tiling_data.tailRows, tiling_data.H,
                                   (tiling_data.W + 1) / 2); // int8/uint8/bool类型的W需要除以2，因为一个half可以存储两个int8/uint8/bool
        }
    } else if (TILING_KEY_IS(2)) {// 2_0 big loop
        if constexpr (is_one_of_v<DTYPE_INPUT, bool>) {
            unpack_2_0_big<int8_t>(input, output, tiling_data.coreRows, tiling_data.tailRows, tiling_data.H, tiling_data.W);
        } else {
            unpack_2_0_big<DTYPE_INPUT>(input, output, tiling_data.coreRows, tiling_data.tailRows, tiling_data.H, tiling_data.W);
        }
    } else if (TILING_KEY_IS(3)) {// 2_1
        if constexpr (is_one_of_v<DTYPE_INPUT, float, int32_t>) {
            unpack_2_1<float>(input, output, tiling_data.coreRows, tiling_data.tailRows, tiling_data.H, tiling_data.W);
        } else if constexpr (is_one_of_v<DTYPE_INPUT, half, int16_t, bfloat16_t>) {
            unpack_2_1<half>(input, output, tiling_data.coreRows, tiling_data.tailRows, tiling_data.H, tiling_data.W);
        } else if constexpr (is_one_of_v<DTYPE_INPUT, int8_t, uint8_t, bool>) {
            unpack_2_1_naive<DTYPE_INPUT>(input, output, tiling_data.coreRows, tiling_data.tailRows, tiling_data.H, tiling_data.W);
        }

    } else if (TILING_KEY_IS(4)) {// 3维
        if (W * sizeof(DTYPE_INPUT) <= 32 * 1024) {
            bool reverse = needReverse(workspace);
            if (reverse) {
                if constexpr (is_one_of_v<DTYPE_INPUT, float, int32_t>) {
                    unpack_3_small<float, true>(input, output, tiling_data.coreRows, tiling_data.tailRows, tiling_data.H, tiling_data.W);
                } else if constexpr (is_one_of_v<DTYPE_INPUT, half, int16_t, bfloat16_t>) {
                    unpack_3_small<half, true>(input, output, tiling_data.coreRows, tiling_data.tailRows, tiling_data.H, tiling_data.W);
                } else if constexpr (is_one_of_v<DTYPE_INPUT, int8_t, uint8_t, bool>) {
                    unpack_3_small<half, true>(input, output, tiling_data.coreRows, tiling_data.tailRows, tiling_data.H,
                                               (tiling_data.W + 1) / 2); // int8/uint8/bool类型的W需要除以2，因为一个half可以存储两个int8/uint8/bool
                }
            } else {
                if constexpr (is_one_of_v<DTYPE_INPUT, float, int32_t>) {
                    unpack_3_small<float, false>(input, output, tiling_data.coreRows, tiling_data.tailRows, tiling_data.H, tiling_data.W);
                } else if constexpr (is_one_of_v<DTYPE_INPUT, half, int16_t, bfloat16_t>) {
                    unpack_3_small<half, false>(input, output, tiling_data.coreRows, tiling_data.tailRows, tiling_data.H, tiling_data.W);
                } else if constexpr (is_one_of_v<DTYPE_INPUT, int8_t, uint8_t, bool>) {
                    unpack_3_small<half, false>(input, output, tiling_data.coreRows, tiling_data.tailRows, tiling_data.H,
                                                (tiling_data.W + 1) / 2); // int8/uint8/bool类型的W需要除以2，因为一个half可以存储两个int8/uint8/bool
                }
            }
        } else {
            unpack_3_big<DTYPE_INPUT>(input, output, tiling_data.coreRows, tiling_data.tailRows, tiling_data.H, tiling_data.W);
        }
    }
}