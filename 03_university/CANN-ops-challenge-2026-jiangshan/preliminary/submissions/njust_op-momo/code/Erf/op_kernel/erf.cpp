#include "kernel_operator.h"
#include "erf_tiling.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM  = 2;      // 双缓冲，掩盖 DMA 延迟
constexpr uint32_t TILE_LENGTH = 4096;   // 双缓冲下每片减半，保证 UB 够用

constexpr float ERF_CLAMP_MIN = -2.95f;
constexpr float ERF_CLAMP_MAX =  2.95f;
constexpr float ERF_P0 = 1.1282972640e+00f;
constexpr float ERF_P1 = 1.9529287110e-01f;
constexpr float ERF_P2 = 3.0080328840e-02f;
constexpr float ERF_Q1 = 5.0535413165e-01f;
constexpr float ERF_Q2 = 9.7607653769e-02f;
constexpr float ERF_Q3 = 3.4495698431e-03f;

class KernelErf {
public:
    __aicore__ inline KernelErf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, uint32_t blockNum) {
        xGm.SetGlobalBuffer((__gm__ float *)x, length);
        yGm.SetGlobalBuffer((__gm__ float *)y, length);

        totalLength = length;
        coreNum     = blockNum;
        coreIndex   = GetBlockIdx();

        // 单核时 coreOffset=0, coreLength=length，无需对齐计算
        if (coreNum == 1) {
            coreOffset = 0;
            coreLength = totalLength;
        } else {
            // 多核：每核向上对齐到 8，最后一核取余数
            uint32_t elemsPerCore = (totalLength + coreNum - 1) / coreNum;
            elemsPerCore = (elemsPerCore + 7) / 8 * 8;
            coreOffset = elemsPerCore * coreIndex;
            if (coreOffset >= totalLength) {
                coreLength = 0;
            } else {
                uint32_t remain = totalLength - coreOffset;
                coreLength = remain < elemsPerCore ? remain : elemsPerCore;
            }
        }

        pipe.InitBuffer(inQueueX,  BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(numBuf, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(denBuf, TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process() {
        // 数据量 <= TILE_LENGTH：整块处理，省去循环判断开销
        if (coreLength <= TILE_LENGTH) {
            if (coreLength > 0) {
                CopyIn(0, coreLength);
                Compute(coreLength);
                CopyOut(0, coreLength);
            }
            return;
        }
        // 数据量 > TILE_LENGTH：分片循环
        uint32_t processed = 0;
        while (processed < coreLength) {
            uint32_t count = coreLength - processed;
            if (count > TILE_LENGTH) count = TILE_LENGTH;
            CopyIn(processed, count);
            Compute(count);
            CopyOut(processed, count);
            processed += count;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        if ((count & 7U) == 0U) {
            DataCopy(xLocal, xGm[coreOffset + offset], count);
        } else {
            DataCopyExtParams copyParams{1,
                static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, xGm[coreOffset + offset], copyParams, padParams);
        }
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<float> num    = numBuf.Get<float>();
        LocalTensor<float> den    = denBuf.Get<float>();

        Maxs(xLocal, xLocal, ERF_CLAMP_MIN, count);
        Mins(xLocal, xLocal, ERF_CLAMP_MAX, count);

        Mul(yLocal, xLocal, xLocal, count);  // t = x^2

        // 分子 P(t)
        Muls(num, yLocal, ERF_P2, count);
        Adds(num, num,    ERF_P1, count);
        Mul (num, num,    yLocal, count);
        Adds(num, num,    ERF_P0, count);

        // 分母 Q(t)
        Muls(den, yLocal, ERF_Q3, count);
        Adds(den, den,    ERF_Q2, count);
        Mul (den, den,    yLocal, count);
        Adds(den, den,    ERF_Q1, count);
        Mul (den, den,    yLocal, count);
        Adds(den, den,    1.0f,   count);

        Div(yLocal, num, den,    count);
        Mul(yLocal, yLocal, xLocal, count);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        if ((count & 7U) == 0U) {
            DataCopy(yGm[coreOffset + offset], yLocal, count);
        } else {
            DataCopyExtParams copyParams{1,
                static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
            DataCopyPad(yGm[coreOffset + offset], yLocal, copyParams);
        }
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN,  BUFFER_NUM> inQueueX;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> numBuf;
    TBuf<QuePosition::VECCALC> denBuf;

    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    uint32_t totalLength = 0;
    uint32_t coreNum     = 1;
    uint32_t coreIndex   = 0;
    uint32_t coreOffset  = 0;
    uint32_t coreLength  = 0;
};

__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tilingData, tiling);
    KernelErf op;
    op.Init(x, y, tilingData.length, tilingData.blockNum);
    op.Process();
}