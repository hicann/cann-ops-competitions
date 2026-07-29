#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

__aicore__ inline uint32_t CeilAlign(uint32_t v, uint32_t a) {
    return ((v + a - 1) / a) * a;
}

// ---------------------------- half (v9: 3 ops, no selBuf) ----------------------------
class KernelIsNanHalf {
public:
    __aicore__ inline KernelIsNanHalf() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t totalLength, uint32_t blockLength, uint32_t tileLength)
    {
        this->tileLength = tileLength;
        uint32_t offset = blockLength * GetBlockIdx();
        uint32_t remaining = (offset < totalLength) ? (totalLength - offset) : 0;
        this->actualLength = (blockLength < remaining) ? blockLength : remaining;
        if (this->actualLength > 0) {
            xGm.SetGlobalBuffer((__gm__ half*)x + offset, this->actualLength);
            yGm.SetGlobalBuffer((__gm__ int8_t*)y + offset, this->actualLength);
            pipe.InitBuffer(inQueueX,  BUFFER_NUM, tileLength * sizeof(half));
            pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(int8_t));
            pipe.InitBuffer(maskBuf, tileLength / 8);
            pipe.InitBuffer(zeroBuf, tileLength * sizeof(half));
        }
    }
    __aicore__ inline void Process()
    {
        if (actualLength == 0) return;
        LocalTensor<half> zeros = zeroBuf.Get<half>();
        Duplicate(zeros, (half)0.0, tileLength);
        uint32_t loopCount = (actualLength + tileLength - 1) / tileLength;
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t curLen = (i == loopCount - 1) ? (actualLength - i * tileLength) : tileLength;
            CopyIn(i, curLen);
            Compute(curLen, zeros);
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
    __aicore__ inline void Compute(uint32_t curLen, LocalTensor<half>& zeros)
    {
        LocalTensor<half>    xLocal = inQueueX.DeQue<half>();
        LocalTensor<int8_t>  yLocal = outQueueY.AllocTensor<int8_t>();
        LocalTensor<uint8_t> mask   = maskBuf.Get<uint8_t>();
        uint32_t n = CeilAlign(curLen, 128);
        Compare(mask, xLocal, xLocal, CMPMODE::EQ, n);
        // Reuse xLocal as the half sel buffer in place.
        Select(xLocal, mask, zeros, (half)1.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, n);
        Cast(yLocal, xLocal, RoundMode::CAST_ROUND, curLen);
        outQueueY.EnQue<int8_t>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t curLen)
    {
        LocalTensor<int8_t> yLocal = outQueueY.DeQue<int8_t>();
        if (curLen == tileLength) {
            DataCopy(yGm[progress * tileLength], yLocal, tileLength);
        } else {
            DataCopyExtParams params{1, static_cast<uint32_t>(curLen * sizeof(int8_t)), 0, 0, 0};
            DataCopyPad(yGm[progress * tileLength], yLocal, params);
        }
        outQueueY.FreeTensor(yLocal);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> maskBuf, zeroBuf;
    GlobalTensor<half>   xGm;
    GlobalTensor<int8_t> yGm;
    uint32_t tileLength;
    uint32_t actualLength;
};

// ---------------------------- float (v9: 3 ops, no selBuf) ----------------------------
class KernelIsNanFloat {
public:
    __aicore__ inline KernelIsNanFloat() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t totalLength, uint32_t blockLength, uint32_t tileLength)
    {
        this->tileLength = tileLength;
        uint32_t offset = blockLength * GetBlockIdx();
        uint32_t remaining = (offset < totalLength) ? (totalLength - offset) : 0;
        this->actualLength = (blockLength < remaining) ? blockLength : remaining;
        if (this->actualLength > 0) {
            xGm.SetGlobalBuffer((__gm__ float*)x + offset, this->actualLength);
            yGm.SetGlobalBuffer((__gm__ int8_t*)y + offset, this->actualLength);
            pipe.InitBuffer(inQueueX,  BUFFER_NUM, tileLength * sizeof(float));
            pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(int8_t));
            pipe.InitBuffer(maskBuf, tileLength / 8);
            pipe.InitBuffer(zeroBuf, tileLength * sizeof(half));
        }
    }
    __aicore__ inline void Process()
    {
        if (actualLength == 0) return;
        LocalTensor<half> zeros = zeroBuf.Get<half>();
        Duplicate(zeros, (half)0.0, tileLength);
        uint32_t loopCount = (actualLength + tileLength - 1) / tileLength;
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t curLen = (i == loopCount - 1) ? (actualLength - i * tileLength) : tileLength;
            CopyIn(i, curLen);
            Compute(curLen, zeros);
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
    __aicore__ inline void Compute(uint32_t curLen, LocalTensor<half>& zeros)
    {
        LocalTensor<float>   xLocal = inQueueX.DeQue<float>();
        LocalTensor<int8_t>  yLocal = outQueueY.AllocTensor<int8_t>();
        LocalTensor<uint8_t> mask   = maskBuf.Get<uint8_t>();
        uint32_t n = CeilAlign(curLen, 128);
        Compare(mask, xLocal, xLocal, CMPMODE::EQ, n);
        // xLocal is float (4 B/elem); reinterpret first half as a half-typed sel buffer.
        LocalTensor<half> sel = xLocal.ReinterpretCast<half>();
        Select(sel, mask, zeros, (half)1.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, n);
        Cast(yLocal, sel, RoundMode::CAST_ROUND, curLen);
        outQueueY.EnQue<int8_t>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t curLen)
    {
        LocalTensor<int8_t> yLocal = outQueueY.DeQue<int8_t>();
        if (curLen == tileLength) {
            DataCopy(yGm[progress * tileLength], yLocal, tileLength);
        } else {
            DataCopyExtParams params{1, static_cast<uint32_t>(curLen * sizeof(int8_t)), 0, 0, 0};
            DataCopyPad(yGm[progress * tileLength], yLocal, params);
        }
        outQueueY.FreeTensor(yLocal);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> maskBuf, zeroBuf;
    GlobalTensor<float>  xGm;
    GlobalTensor<int8_t> yGm;
    uint32_t tileLength;
    uint32_t actualLength;
};

// ---------------------------- bf16 (v9: 4 ops, no selBuf) ----------------------------
class KernelIsNanBf16 {
public:
    __aicore__ inline KernelIsNanBf16() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t totalLength, uint32_t blockLength, uint32_t tileLength)
    {
        this->tileLength = tileLength;
        uint32_t offset = blockLength * GetBlockIdx();
        uint32_t remaining = (offset < totalLength) ? (totalLength - offset) : 0;
        this->actualLength = (blockLength < remaining) ? blockLength : remaining;
        if (this->actualLength > 0) {
            xGm.SetGlobalBuffer((__gm__ bfloat16_t*)x + offset, this->actualLength);
            yGm.SetGlobalBuffer((__gm__ int8_t*)y + offset, this->actualLength);
            pipe.InitBuffer(inQueueX,  BUFFER_NUM, tileLength * sizeof(bfloat16_t));
            pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(int8_t));
            pipe.InitBuffer(maskBuf, tileLength / 8);
            pipe.InitBuffer(zeroBuf, tileLength * sizeof(half));
            pipe.InitBuffer(fxBuf,   tileLength * sizeof(float));
        }
    }
    __aicore__ inline void Process()
    {
        if (actualLength == 0) return;
        LocalTensor<half> zeros = zeroBuf.Get<half>();
        Duplicate(zeros, (half)0.0, tileLength);
        uint32_t loopCount = (actualLength + tileLength - 1) / tileLength;
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t curLen = (i == loopCount - 1) ? (actualLength - i * tileLength) : tileLength;
            CopyIn(i, curLen);
            Compute(curLen, zeros);
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
    __aicore__ inline void Compute(uint32_t curLen, LocalTensor<half>& zeros)
    {
        LocalTensor<bfloat16_t> xLocal = inQueueX.DeQue<bfloat16_t>();
        LocalTensor<int8_t>     yLocal = outQueueY.AllocTensor<int8_t>();
        LocalTensor<uint8_t>    mask   = maskBuf.Get<uint8_t>();
        LocalTensor<float>      fx     = fxBuf.Get<float>();
        uint32_t n = CeilAlign(curLen, 128);
        Cast(fx, xLocal, RoundMode::CAST_NONE, n);
        Compare(mask, fx, fx, CMPMODE::EQ, n);
        // Reuse xLocal (bf16, 2 B/elem) as half sel buffer in place.
        LocalTensor<half> sel = xLocal.ReinterpretCast<half>();
        Select(sel, mask, zeros, (half)1.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, n);
        Cast(yLocal, sel, RoundMode::CAST_ROUND, curLen);
        outQueueY.EnQue<int8_t>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t curLen)
    {
        LocalTensor<int8_t> yLocal = outQueueY.DeQue<int8_t>();
        if (curLen == tileLength) {
            DataCopy(yGm[progress * tileLength], yLocal, tileLength);
        } else {
            DataCopyExtParams params{1, static_cast<uint32_t>(curLen * sizeof(int8_t)), 0, 0, 0};
            DataCopyPad(yGm[progress * tileLength], yLocal, params);
        }
        outQueueY.FreeTensor(yLocal);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> maskBuf, zeroBuf, fxBuf;
    GlobalTensor<bfloat16_t> xGm;
    GlobalTensor<int8_t>     yGm;
    uint32_t tileLength;
    uint32_t actualLength;
};

extern "C" __global__ __aicore__ void is_nan(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(td, tiling);

    if (TILING_KEY_IS(0)) {
        KernelIsNanHalf op;
        op.Init(x, y, td.totalLength, td.blockLength, td.tileLength);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KernelIsNanFloat op;
        op.Init(x, y, td.totalLength, td.blockLength, td.tileLength);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelIsNanBf16 op;
        op.Init(x, y, td.totalLength, td.blockLength, td.tileLength);
        op.Process();
    }
}
