#include "kernel_operator.h"
#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

// 7闃?Horner 澶氶」寮忚繃绋嬬骇杞婚噺鍖栧睍寮€
__aicore__ inline void ComputePoly(LocalTensor<float> yLocal, LocalTensor<float> xLocal, LocalTensor<float> zLocal, LocalTensor<float> tmp, uint32_t calcLength) {
    const float c0 = 1.1283549039f;
    const float c1 = -0.375666690534f;
    const float c2 = 0.111394679188f;
    const float c3 = -0.0251047297656f;
    const float c4 = 0.00412842883061f;
    const float c5 = -0.000458782482048f;
    const float c6 = 0.0000301684849548f;
    const float c7 = -0.000000873146177582f;

    Muls(tmp, zLocal, c7, calcLength);
    Adds(tmp, tmp, c6, calcLength);
    Mul(tmp, tmp, zLocal, calcLength);
    Adds(tmp, tmp, c5, calcLength);
    Mul(tmp, tmp, zLocal, calcLength);
    Adds(tmp, tmp, c4, calcLength);
    Mul(tmp, tmp, zLocal, calcLength);
    Adds(tmp, tmp, c3, calcLength);
    Mul(tmp, tmp, zLocal, calcLength);
    Adds(tmp, tmp, c2, calcLength);
    Mul(tmp, tmp, zLocal, calcLength);
    Adds(tmp, tmp, c1, calcLength);
    Mul(tmp, tmp, zLocal, calcLength);
    Adds(tmp, tmp, c0, calcLength);
    Mul(yLocal, tmp, xLocal, calcLength);
}

extern "C" __global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    (void)workspace;
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA(tilingData, tiling);

    uint32_t totalLength = tilingData.totalLength;
    uint32_t tileLength = tilingData.tileLength;
    uint32_t coreNum = tilingData.coreNum;
    uint32_t mode = tilingData.mode;

    if (totalLength == 0U) return;

    uint32_t blockIdx = GetBlockIdx();

    if (mode == 0U) {
        // ==================== 妯″紡 0: 鐩撮€氬揩閫熻矾寰?(TBuf 瓒呬綆寤惰繜) ====================
        uint32_t blockStride = tileLength;
        uint32_t offset = blockIdx * blockStride;
        if (offset >= totalLength) return;
        
        uint32_t remainLength = totalLength - offset;
        uint32_t blockLength = remainLength < blockStride ? remainLength : blockStride;

        GlobalTensor<float> xGm;
        GlobalTensor<float> yGm;
        xGm.SetGlobalBuffer((__gm__ float *)x + offset, blockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + offset, blockLength);

        TPipe pipe;
        TBuf<QuePosition::VECIN> xBuf, yBuf, tmpBuf;
        pipe.InitBuffer(xBuf, tileLength * sizeof(float));
        pipe.InitBuffer(yBuf, tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf, tileLength * sizeof(float));

        LocalTensor<float> xLocal = xBuf.Get<float>();
        LocalTensor<float> yLocal = yBuf.Get<float>();
        LocalTensor<float> tmp = tmpBuf.Get<float>();

        // 鎼叆 X
        bool isAligned8 = ((offset & 7U) == 0U) && ((blockLength & 7U) == 0U);
        if (isAligned8) {
            DataCopy(xLocal, xGm[0], blockLength);
        } else {
            DataCopyExtParams copyParams{1U, static_cast<uint32_t>(blockLength * sizeof(float)), 0U, 0U, 0U};
            DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0f};
            DataCopyPad(xLocal, xGm[0], copyParams, padParams);
        }

        // 灞忛殰 1
        PipeBarrier<PIPE_ALL>();

        // 鐭㈤噺杩愮畻
        uint32_t calcLength = (blockLength + 63U) & (~63U);
        Mins(yLocal, xLocal, 2.7f, calcLength);
        Maxs(yLocal, yLocal, -2.7f, calcLength);
        Mul(xLocal, yLocal, yLocal, calcLength); 
        ComputePoly(yLocal, yLocal, xLocal, tmp, calcLength);

        // 灞忛殰 2
        PipeBarrier<PIPE_ALL>();

        // 鎼嚭 Y
        if (isAligned8) {
            DataCopy(yGm[0], yLocal, blockLength);
        } else {
            DataCopyExtParams copyParams{1U, static_cast<uint32_t>(blockLength * sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(yGm[0], yLocal, copyParams);
        }

    } else {
        // ==================== 2. 楂樺苟鍙戝紓姝ラ槦鍒楁祦姘寸嚎妯″紡 (瓒呭ぇ寮犻噺) ====================
        uint32_t blockStride = (totalLength + coreNum - 1U) / coreNum;
        blockStride = (blockStride + 63U) & (~63U);
        uint32_t offset = blockIdx * blockStride;
        if (offset >= totalLength) return;

        uint32_t remainLength = totalLength - offset;
        uint32_t blockLength = remainLength < blockStride ? remainLength : blockStride;

        GlobalTensor<float> xGm;
        GlobalTensor<float> yGm;
        xGm.SetGlobalBuffer((__gm__ float *)x + offset, blockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + offset, blockLength);

        TPipe pipe;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
        TBuf<QuePosition::VECIN> tmpBuf;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf, tileLength * sizeof(float));

        uint32_t loopCount = blockLength / tileLength;
        uint32_t tailLength = blockLength % tileLength;

        for (uint32_t loopIdx = 0U; loopIdx < loopCount; ++loopIdx) {
            uint32_t gmOffset = loopIdx * tileLength;
            
            LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            if ((((offset + gmOffset) & 7U) == 0U) && ((tileLength & 7U) == 0U)) {
                DataCopy(xLocal, xGm[gmOffset], tileLength);
            } else {
                DataCopyExtParams copyParams{1U, static_cast<uint32_t>(tileLength * sizeof(float)), 0U, 0U, 0U};
                DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0f};
                DataCopyPad(xLocal, xGm[gmOffset], copyParams, padParams);
            }
            inQueueX.EnQue(xLocal);

            xLocal = inQueueX.DeQue<float>();
            LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
            LocalTensor<float> tmp = tmpBuf.Get<float>();

            Mins(yLocal, xLocal, 2.7f, tileLength);
            Maxs(yLocal, yLocal, -2.7f, tileLength);
            Mul(xLocal, yLocal, yLocal, tileLength); 
            ComputePoly(yLocal, yLocal, xLocal, tmp, tileLength);

            outQueueY.EnQue(yLocal);
            inQueueX.FreeTensor(xLocal);

            yLocal = outQueueY.DeQue<float>();
            if ((((offset + gmOffset) & 7U) == 0U) && ((tileLength & 7U) == 0U)) {
                DataCopy(yGm[gmOffset], yLocal, tileLength);
            } else {
                DataCopyExtParams copyParams{1U, static_cast<uint32_t>(tileLength * sizeof(float)), 0U, 0U, 0U};
                DataCopyPad(yGm[gmOffset], yLocal, copyParams);
            }
            outQueueY.FreeTensor(yLocal);
        }

        if (tailLength > 0U) {
            uint32_t gmOffset = loopCount * tileLength;
            
            LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            if ((((offset + gmOffset) & 7U) == 0U) && ((tailLength & 7U) == 0U)) {
                DataCopy(xLocal, xGm[gmOffset], tailLength);
            } else {
                DataCopyExtParams copyParams{1U, static_cast<uint32_t>(tailLength * sizeof(float)), 0U, 0U, 0U};
                DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0f};
                DataCopyPad(xLocal, xGm[gmOffset], copyParams, padParams);
            }
            inQueueX.EnQue(xLocal);

            uint32_t calcLength = (tailLength + 63U) & (~63U);
            xLocal = inQueueX.DeQue<float>();
            LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
            LocalTensor<float> tmp = tmpBuf.Get<float>();

            Mins(yLocal, xLocal, 2.7f, calcLength);
            Maxs(yLocal, yLocal, -2.7f, calcLength);
            Mul(xLocal, yLocal, yLocal, calcLength); 
            ComputePoly(yLocal, yLocal, xLocal, tmp, calcLength);

            outQueueY.EnQue(yLocal);
            inQueueX.FreeTensor(xLocal);

            yLocal = outQueueY.DeQue<float>();
            if ((((offset + gmOffset) & 7U) == 0U) && ((tailLength & 7U) == 0U)) {
                DataCopy(yGm[gmOffset], yLocal, tailLength);
            } else {
                DataCopyExtParams copyParams{1U, static_cast<uint32_t>(tailLength * sizeof(float)), 0U, 0U, 0U};
                DataCopyPad(yGm[gmOffset], yLocal, copyParams);
            }
            outQueueY.FreeTensor(yLocal);
        }
    }
}
