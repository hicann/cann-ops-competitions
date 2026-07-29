#include "kernel_operator.h"
#include "erfinv_tiling.h"

extern "C" __global__ __aicore__ void erfinv(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfinvTilingData);
    GET_TILING_DATA(tilingData, tiling);
    // TODO: user kernel impl
}