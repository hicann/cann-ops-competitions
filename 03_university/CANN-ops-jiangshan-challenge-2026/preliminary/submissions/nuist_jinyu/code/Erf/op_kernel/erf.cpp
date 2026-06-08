#include "kernel_operator.h"
using namespace AscendC;

constexpr int BUFFER_NUM = 2;

__aicore__ inline float ErfFastScalar(float x) {
    x = (x > 3.92f) ? 3.92f : x;
    x = (x < -3.92f) ? -3.92f : x;
    float z = x * x;
    float num = 1.1283254237e+00f
              + z * (1.7734201499e-01f
              + z * (4.1160905427e-02f
              + z * 6.0036044517e-04f));
    float den = 1.0f
              + z * (4.8990832263e-01f
              + z * (1.0095401939e-01f
              + z * 8.0962790643e-03f));
    return x * num / den;
}

extern "C" __global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    uint32_t blockIdx   = GetBlockIdx();
    uint32_t blockNum   = GetBlockNum();
    uint32_t totalSize  = tilingData.totalSize;
    uint32_t tileSize   = tilingData.tileSize;
    uint32_t perCoreSize = tilingData.perCoreSize;

    if (blockNum == 0 || totalSize == 0) return;

    if (totalSize < 8) {
        if (blockIdx == 0) {
            __gm__ float* xPtr = (__gm__ float*)x;
            __gm__ float* yPtr = (__gm__ float*)y;
            for (uint32_t i = 0; i < totalSize; i++) {
                yPtr[i] = ErfFastScalar(xPtr[i]);
            }
        }
        return;
    }

    uint32_t offset = blockIdx * perCoreSize;
    if (offset >= totalSize) return;

    uint32_t processSize = (offset + perCoreSize > totalSize)
        ? (totalSize - offset)
        : perCoreSize;

    uint32_t alignedSize = processSize / 8 * 8;
    uint32_t tailSize    = processSize - alignedSize;
    uint32_t tailOffset  = offset + alignedSize;

    // 单轮即可处理完：完全展开，零函数调用 overhead
    if (alignedSize > 0 && alignedSize <= tileSize) {
        TPipe pipe;
        TQue<TPosition::VECIN, 1> inQueueX;
        TQue<TPosition::VECOUT, 1> outQueueY;
        pipe.InitBuffer(inQueueX, 1, tileSize * sizeof(float));
        pipe.InitBuffer(outQueueY, 1, tileSize * sizeof(float));

        GlobalTensor<float> xGm, yGm;
        xGm.SetGlobalBuffer((__gm__ float*)x, totalSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalSize);

        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[offset], alignedSize);
        inQueueX.EnQue(xLocal);

        xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        Erf(yLocal, xLocal, alignedSize);
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);

        yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[offset], yLocal, alignedSize);
        outQueueY.FreeTensor(yLocal);
    }
    // 需要多轮：走标准双缓冲
    else if (alignedSize > tileSize) {
        GlobalTensor<float> xGm, yGm;
        xGm.SetGlobalBuffer((__gm__ float*)x, totalSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalSize);

        TPipe pipe;
        TQue<TPosition::VECIN, BUFFER_NUM> inQueueX;
        TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY;
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileSize * sizeof(float));

        uint32_t loopCount = (alignedSize + tileSize - 1) / tileSize;
        uint32_t firstSize = (loopCount == 1) ? alignedSize : tileSize;

        {
            LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            DataCopy(xLocal, xGm[offset], firstSize);
            inQueueX.EnQue(xLocal);
        }

        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t curOffset = offset + i * tileSize;
            uint32_t curSize = (i == loopCount - 1)
                ? (alignedSize - i * tileSize)
                : tileSize;

            {
                LocalTensor<float> xLocal = inQueueX.DeQue<float>();
                LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
                Erf(yLocal, xLocal, curSize);
                outQueueY.EnQue<float>(yLocal);
                inQueueX.FreeTensor(xLocal);
            }

            if (i < loopCount - 1) {
                uint32_t nextOffset = curOffset + tileSize;
                uint32_t nextSize = (i == loopCount - 2)
                    ? (alignedSize - (i + 1) * tileSize)
                    : tileSize;
                LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
                DataCopy(xLocal, xGm[nextOffset], nextSize);
                inQueueX.EnQue(xLocal);
            }

            {
                LocalTensor<float> yLocal = outQueueY.DeQue<float>();
                DataCopy(yGm[curOffset], yLocal, curSize);
                outQueueY.FreeTensor(yLocal);
            }
        }
    }

    if (tailSize > 0) {
        __gm__ float* xPtr = (__gm__ float*)x;
        __gm__ float* yPtr = (__gm__ float*)y;
        for (uint32_t i = 0; i < tailSize; i++) {
            yPtr[tailOffset + i] = ErfFastScalar(xPtr[tailOffset + i]);
        }
    }
}
