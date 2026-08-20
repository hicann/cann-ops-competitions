#include "kernel_operator.h"
#include "kernel.h"
using namespace AscendC;

__aicore__ inline void Process(GM_ADDR output, int32_t value, int32_t baseSize) {
    GlobalTensor<int32_t> outGm;
    int32_t coreSize = baseSize;
    int32_t srcBeginIndex = coreSize * GetBlockIdx();
    outGm.SetGlobalBuffer((__gm__ int32_t *)output + srcBeginIndex, coreSize);
    int32_t spaceSize = MAX_UB_SIZE_KB * 1024;
    spaceSize = min((int32_t)(coreSize * sizeof(int32_t)), spaceSize);
    int32_t n_elements_per_iter = spaceSize / sizeof(int32_t);
    int32_t smallLoopTimes = (coreSize + n_elements_per_iter - 1) / n_elements_per_iter;

    LocalTensor<int32_t> outLocal(TPosition::VECOUT, 0, n_elements_per_iter);
    Duplicate(outLocal, value, n_elements_per_iter);
    int32_t blockBeginIndex = 0;
    __ubuf__ void *srcAddr = (__ubuf__ void *)outLocal.GetPhyAddr();
    smallLoopTimes -= 1;
    constexpr static uint16_t nBurst = 1;
    const uint16_t lenBurst = n_elements_per_iter * sizeof(int32_t) / ONE_BLOCK_BYTES;
    constexpr static uint16_t srcGap = 0;
    constexpr static uint16_t dstGap = 0;
    for (int32_t i = 0; i < smallLoopTimes; ++i) {
        __gm__ void *dstAddr = (__gm__ void *)outGm[blockBeginIndex].GetPhyAddr();
        copy_ubuf_to_gm(dstAddr, srcAddr, 8, nBurst, lenBurst, srcGap, dstGap, bm_t::BM_DISABLE);
        blockBeginIndex += n_elements_per_iter;
    }
    int32_t iterSize = min(n_elements_per_iter, coreSize - blockBeginIndex);
    DataCopy(outGm[blockBeginIndex], outLocal, iterSize);
}

__aicore__ inline void ProcessReverse(GM_ADDR output, int32_t value, int32_t baseSize) {
    GlobalTensor<int32_t> outGm;
    int32_t coreSize = baseSize;
    int32_t srcBeginIndex = coreSize * GetBlockIdx();
    outGm.SetGlobalBuffer((__gm__ int32_t *)output + srcBeginIndex, coreSize);
    int32_t spaceSize = MAX_UB_SIZE_KB * 1024;
    spaceSize = min((int32_t)(coreSize * sizeof(int32_t)), spaceSize);
    int32_t n_elements_per_iter = spaceSize / sizeof(int32_t);
    int32_t smallLoopTimes = (coreSize + n_elements_per_iter - 1) / n_elements_per_iter;

    LocalTensor<int32_t> outLocal(TPosition::VECOUT, 0, n_elements_per_iter);
    Duplicate(outLocal, value, n_elements_per_iter);
    int32_t blockBeginIndex = smallLoopTimes * n_elements_per_iter - n_elements_per_iter;
    __ubuf__ void *srcAddr = (__ubuf__ void *)outLocal.GetPhyAddr();
    int32_t iterSize = min(n_elements_per_iter, coreSize - blockBeginIndex);
    DataCopy(outGm[blockBeginIndex], outLocal, iterSize);
    blockBeginIndex -= n_elements_per_iter;
    constexpr static uint16_t nBurst = 1;
    const uint16_t lenBurst = n_elements_per_iter * sizeof(int32_t) / ONE_BLOCK_BYTES;
    constexpr static uint16_t srcGap = 0;
    constexpr static uint16_t dstGap = 0;
    smallLoopTimes -= 1;
    for (int32_t i = 0; i < smallLoopTimes; ++i) {
        __gm__ void *dstAddr = (__gm__ void *)outGm[blockBeginIndex].GetPhyAddr();
        copy_ubuf_to_gm(dstAddr, srcAddr, 2, nBurst, lenBurst, srcGap, dstGap, bm_t::BM_DISABLE);
        blockBeginIndex -= n_elements_per_iter;
    }
}
extern "C" __global__ __aicore__ void fills(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {
        Process(output, tiling_data.value, tiling_data.baseSize);
    } else if (TILING_KEY_IS(2)) {
        if (needReverse(workspace)) {
            ProcessReverse(output, tiling_data.value, tiling_data.baseSize);
        } else {
            Process(output, tiling_data.value, tiling_data.baseSize);
        }
    }
}