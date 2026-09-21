#include "max_pool_backward.h"

extern "C" __global__ __aicore__ void max_pool2d_with_mask_backward(
    GM_ADDR gradOutput, GM_ADDR self, GM_ADDR indices, GM_ADDR gradInput,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
    REGISTER_TILING_DEFAULT(PoolTiling);
    GET_TILING_DATA(t, tiling);
    MaxPoolBackward::Compute<DTYPE_GRADOUTPUT, false>(gradOutput, indices, gradInput, t);
}
