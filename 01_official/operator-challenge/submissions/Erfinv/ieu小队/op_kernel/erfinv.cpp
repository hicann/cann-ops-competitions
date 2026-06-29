#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

__aicore__ inline uint32_t CeilAlign(uint32_t v, uint32_t a) {
    return ((v + a - 1) / a) * a;
}

// ---- D5 (half path) ----
__aicore__ inline void ErfinvD5w(const LocalTensor<float>& out, const LocalTensor<float>& x,
                                 const LocalTensor<float>& w,  const LocalTensor<float>& p,
                                 const LocalTensor<float>& ones,
                                 uint32_t n)
{
    Mul(w, x, x, n);
    Sub(w, ones, w, n);                            // w = 1 - x^2
    Ln(w, w, n);                                   // w = ln(1 - x^2)

    Duplicate(p,  8.3436689e-06f, n);
    Mul(p, p, w, n);  Adds(p, p,  2.8631649e-04f, n);
    Mul(p, p, w, n);  Adds(p, p,  3.5353551e-03f, n);
    Mul(p, p, w, n);  Adds(p, p,  1.3037874e-02f, n);
    Mul(p, p, w, n);  Adds(p, p, -2.3137343e-01f, n);
    Mul(p, p, w, n);  Adds(p, p,  8.8626832e-01f, n);

    Abs(p, p, n);
    Mul(out, p, x, n);
}

// ---- D7 (bf16 path, no Abs) ----
__aicore__ inline void ErfinvD7wb(const LocalTensor<float>& out, const LocalTensor<float>& x,
                                  const LocalTensor<float>& w,  const LocalTensor<float>& p,
                                  const LocalTensor<float>& ones,
                                  uint32_t n)
{
    Mul(w, x, x, n);
    Sub(w, ones, w, n);
    Ln(w, w, n);

    Duplicate(p, -1.1235100e-07f, n);
    Mul(p, p, w, n);  Adds(p, p, -3.7272373e-06f, n);
    Mul(p, p, w, n);  Adds(p, p, -3.8566790e-05f, n);
    Mul(p, p, w, n);  Adds(p, p,  5.6636771e-06f, n);
    Mul(p, p, w, n);  Adds(p, p,  2.7163153e-03f, n);
    Mul(p, p, w, n);  Adds(p, p,  1.1967503e-02f, n);
    Mul(p, p, w, n);  Adds(p, p, -2.3186068e-01f, n);
    Mul(p, p, w, n);  Adds(p, p,  8.8623518e-01f, n);

    Mul(out, p, x, n);
}

// ---- D8 (fp32 path, Abs needed) ----
__aicore__ inline void ErfinvD8wf(const LocalTensor<float>& out, const LocalTensor<float>& x,
                                  const LocalTensor<float>& w,  const LocalTensor<float>& p,
                                  const LocalTensor<float>& ones,
                                  uint32_t n)
{
    Mul(w, x, x, n);
    Sub(w, ones, w, n);
    Ln(w, w, n);

    Duplicate(p, -1.3558763e-08f, n);
    Mul(p, p, w, n);  Adds(p, p, -6.2482927e-07f, n);
    Mul(p, p, w, n);  Adds(p, p, -1.1371966e-05f, n);
    Mul(p, p, w, n);  Adds(p, p, -9.5967892e-05f, n);
    Mul(p, p, w, n);  Adds(p, p, -2.2379049e-04f, n);
    Mul(p, p, w, n);  Adds(p, p,  2.2409516e-03f, n);
    Mul(p, p, w, n);  Adds(p, p,  1.1509661e-02f, n);
    Mul(p, p, w, n);  Adds(p, p, -2.3201820e-01f, n);
    Mul(p, p, w, n);  Adds(p, p,  8.8622694e-01f, n);

    Abs(p, p, n);
    Mul(out, p, x, n);
}

// ---------------------------- float ----------------------------
class KernelErfinvFloat {
public:
    __aicore__ inline KernelErfinvFloat() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t totalLength, uint32_t blockLength, uint32_t tileLength,
                                TPipe* p)
    {
        this->pipe = p;
        this->tileLength = tileLength;
        uint32_t offset = blockLength * GetBlockIdx();
        uint32_t remaining = (offset < totalLength) ? (totalLength - offset) : 0;
        this->actualLength = (blockLength < remaining) ? blockLength : remaining;
        if (this->actualLength > 0) {
            xGm.SetGlobalBuffer((__gm__ float*)x + offset, this->actualLength);
            yGm.SetGlobalBuffer((__gm__ float*)y + offset, this->actualLength);
            pipe->InitBuffer(inQueueX,  BUFFER_NUM, tileLength * sizeof(float));
            pipe->InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(float));
            pipe->InitBuffer(wBuf, tileLength * sizeof(float));
            pipe->InitBuffer(pBuf, tileLength * sizeof(float));
            pipe->InitBuffer(onesBuf, tileLength * sizeof(float));
        }
    }
    __aicore__ inline void Process()
    {
        if (actualLength == 0) return;
        LocalTensor<float> ones = onesBuf.Get<float>();
        Duplicate(ones, 1.0f, tileLength);

        uint32_t loopCount = (actualLength + tileLength - 1) / tileLength;
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t curLen = (i == loopCount - 1) ? (actualLength - i * tileLength) : tileLength;
            CopyIn(i, curLen);
            Compute(curLen, ones);
            CopyOut(i, curLen);
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t curLen)
    {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        if (curLen == tileLength) {
            DataCopy(xLocal, xGm[progress * tileLength], tileLength);
        } else {
            DataCopyExtParams params{1, static_cast<uint32_t>(curLen * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(xLocal, xGm[progress * tileLength], params, pad);
        }
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(uint32_t curLen, LocalTensor<float>& ones)
    {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        uint32_t n = CeilAlign(curLen, 64);
        ErfinvD8wf(yLocal, xLocal, wBuf.Get<float>(), pBuf.Get<float>(), ones, n);
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t curLen)
    {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        if (curLen == tileLength) {
            DataCopy(yGm[progress * tileLength], yLocal, tileLength);
        } else {
            DataCopyExtParams params{1, static_cast<uint32_t>(curLen * sizeof(float)), 0, 0, 0};
            DataCopyPad(yGm[progress * tileLength], yLocal, params);
        }
        outQueueY.FreeTensor(yLocal);
    }
private:
    TPipe* pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> wBuf, pBuf, onesBuf;
    GlobalTensor<float> xGm, yGm;
    uint32_t tileLength;
    uint32_t actualLength;
};

// ---------------------------- half ----------------------------
class KernelErfinvHalf {
public:
    __aicore__ inline KernelErfinvHalf() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t totalLength, uint32_t blockLength, uint32_t tileLength,
                                TPipe* p)
    {
        this->pipe = p;
        this->tileLength = tileLength;
        uint32_t offset = blockLength * GetBlockIdx();
        uint32_t remaining = (offset < totalLength) ? (totalLength - offset) : 0;
        this->actualLength = (blockLength < remaining) ? blockLength : remaining;
        if (this->actualLength > 0) {
            xGm.SetGlobalBuffer((__gm__ half*)x + offset, this->actualLength);
            yGm.SetGlobalBuffer((__gm__ half*)y + offset, this->actualLength);
            pipe->InitBuffer(inQueueX,  BUFFER_NUM, tileLength * sizeof(half));
            pipe->InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(half));
            pipe->InitBuffer(xfBuf, tileLength * sizeof(float));
            pipe->InitBuffer(wBuf,  tileLength * sizeof(float));
            pipe->InitBuffer(pBuf,  tileLength * sizeof(float));
            pipe->InitBuffer(onesBuf, tileLength * sizeof(float));
        }
    }
    __aicore__ inline void Process()
    {
        if (actualLength == 0) return;
        LocalTensor<float> ones = onesBuf.Get<float>();
        Duplicate(ones, 1.0f, tileLength);

        uint32_t loopCount = (actualLength + tileLength - 1) / tileLength;
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t curLen = (i == loopCount - 1) ? (actualLength - i * tileLength) : tileLength;
            CopyIn(i, curLen);
            Compute(curLen, ones);
            CopyOut(i, curLen);
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t curLen)
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        if (curLen == tileLength) {
            DataCopy(xLocal, xGm[progress * tileLength], tileLength);
        } else {
            DataCopyExtParams params{1, static_cast<uint32_t>(curLen * sizeof(half)), 0, 0, 0};
            DataCopyPadExtParams<half> pad{false, 0, 0, 0};
            DataCopyPad(xLocal, xGm[progress * tileLength], params, pad);
        }
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(uint32_t curLen, LocalTensor<float>& ones)
    {
        LocalTensor<half>  xLocal = inQueueX.DeQue<half>();
        LocalTensor<half>  yLocal = outQueueY.AllocTensor<half>();
        LocalTensor<float> xf = xfBuf.Get<float>();
        uint32_t n = CeilAlign(curLen, 64);
        Cast(xf, xLocal, RoundMode::CAST_NONE, n);
        ErfinvD5w(xf, xf, wBuf.Get<float>(), pBuf.Get<float>(), ones, n);
        Cast(yLocal, xf, RoundMode::CAST_RINT, curLen);
        outQueueY.EnQue<half>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t curLen)
    {
        LocalTensor<half> yLocal = outQueueY.DeQue<half>();
        if (curLen == tileLength) {
            DataCopy(yGm[progress * tileLength], yLocal, tileLength);
        } else {
            DataCopyExtParams params{1, static_cast<uint32_t>(curLen * sizeof(half)), 0, 0, 0};
            DataCopyPad(yGm[progress * tileLength], yLocal, params);
        }
        outQueueY.FreeTensor(yLocal);
    }
private:
    TPipe* pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> xfBuf, wBuf, pBuf, onesBuf;
    GlobalTensor<half> xGm, yGm;
    uint32_t tileLength;
    uint32_t actualLength;
};

// ---------------------------- bf16 ----------------------------
class KernelErfinvBf16 {
public:
    __aicore__ inline KernelErfinvBf16() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t totalLength, uint32_t blockLength, uint32_t tileLength,
                                TPipe* p)
    {
        this->pipe = p;
        this->tileLength = tileLength;
        uint32_t offset = blockLength * GetBlockIdx();
        uint32_t remaining = (offset < totalLength) ? (totalLength - offset) : 0;
        this->actualLength = (blockLength < remaining) ? blockLength : remaining;
        if (this->actualLength > 0) {
            xGm.SetGlobalBuffer((__gm__ bfloat16_t*)x + offset, this->actualLength);
            yGm.SetGlobalBuffer((__gm__ bfloat16_t*)y + offset, this->actualLength);
            pipe->InitBuffer(inQueueX,  BUFFER_NUM, tileLength * sizeof(bfloat16_t));
            pipe->InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(bfloat16_t));
            pipe->InitBuffer(xfBuf, tileLength * sizeof(float));
            pipe->InitBuffer(wBuf,  tileLength * sizeof(float));
            pipe->InitBuffer(pBuf,  tileLength * sizeof(float));
            pipe->InitBuffer(onesBuf, tileLength * sizeof(float));
        }
    }
    __aicore__ inline void Process()
    {
        if (actualLength == 0) return;
        LocalTensor<float> ones = onesBuf.Get<float>();
        Duplicate(ones, 1.0f, tileLength);

        uint32_t loopCount = (actualLength + tileLength - 1) / tileLength;
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t curLen = (i == loopCount - 1) ? (actualLength - i * tileLength) : tileLength;
            CopyIn(i, curLen);
            Compute(curLen, ones);
            CopyOut(i, curLen);
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t curLen)
    {
        LocalTensor<bfloat16_t> xLocal = inQueueX.AllocTensor<bfloat16_t>();
        if (curLen == tileLength) {
            DataCopy(xLocal, xGm[progress * tileLength], tileLength);
        } else {
            DataCopyExtParams params{1, static_cast<uint32_t>(curLen * sizeof(bfloat16_t)), 0, 0, 0};
            DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, 0};
            DataCopyPad(xLocal, xGm[progress * tileLength], params, pad);
        }
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(uint32_t curLen, LocalTensor<float>& ones)
    {
        LocalTensor<bfloat16_t> xLocal = inQueueX.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> yLocal = outQueueY.AllocTensor<bfloat16_t>();
        LocalTensor<float> xf = xfBuf.Get<float>();
        uint32_t n = CeilAlign(curLen, 64);
        Cast(xf, xLocal, RoundMode::CAST_NONE, n);
        ErfinvD7wb(xf, xf, wBuf.Get<float>(), pBuf.Get<float>(), ones, n);
        Cast(yLocal, xf, RoundMode::CAST_RINT, curLen);
        outQueueY.EnQue<bfloat16_t>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t curLen)
    {
        LocalTensor<bfloat16_t> yLocal = outQueueY.DeQue<bfloat16_t>();
        if (curLen == tileLength) {
            DataCopy(yGm[progress * tileLength], yLocal, tileLength);
        } else {
            DataCopyExtParams params{1, static_cast<uint32_t>(curLen * sizeof(bfloat16_t)), 0, 0, 0};
            DataCopyPad(yGm[progress * tileLength], yLocal, params);
        }
        outQueueY.FreeTensor(yLocal);
    }
private:
    TPipe* pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> xfBuf, wBuf, pBuf, onesBuf;
    GlobalTensor<bfloat16_t> xGm, yGm;
    uint32_t tileLength;
    uint32_t actualLength;
};

extern "C" __global__ __aicore__ void erfinv(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(td, tiling);
    TPipe pipe;

    if (TILING_KEY_IS(0)) {
        KernelErfinvHalf op;
        op.Init(x, y, td.totalLength, td.blockLength, td.tileLength, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KernelErfinvFloat op;
        op.Init(x, y, td.totalLength, td.blockLength, td.tileLength, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelErfinvBf16 op;
        op.Init(x, y, td.totalLength, td.blockLength, td.tileLength, &pipe);
        op.Process();
    }
}
