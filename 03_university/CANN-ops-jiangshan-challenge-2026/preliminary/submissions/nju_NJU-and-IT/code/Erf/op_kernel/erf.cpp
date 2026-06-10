#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

template <class DT_X>
class KernelErfTinyDirect {
public:
    __aicore__ inline KernelErfTinyDirect() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length)
    {
        xPtr = reinterpret_cast<__gm__ DT_X *>(x);
        yPtr = reinterpret_cast<__gm__ DT_X *>(y);
        totalLength = length;
    }

    __aicore__ inline void Process()
    {
        if (totalLength > 0U) Store(0U);
        if (totalLength > 1U) Store(1U);
        if (totalLength > 2U) Store(2U);
        if (totalLength > 3U) Store(3U);
        if (totalLength > 4U) Store(4U);
        if (totalLength > 5U) Store(5U);
        if (totalLength > 6U) Store(6U);
        if (totalLength > 7U) Store(7U);
    }

private:
    __aicore__ inline void Store(uint32_t idx) const
    {
        yPtr[idx] = Calc(xPtr[idx]);
    }

    __aicore__ inline DT_X Calc(DT_X input) const
    {
        float x = static_cast<float>(input);
        if (x >= 2.327f) {
            return static_cast<DT_X>(1.0f);
        }
        if (x <= -2.327f) {
            return static_cast<DT_X>(-1.0f);
        }
        if (x >= 3.0f) {
            return static_cast<DT_X>(1.0f);
        }
        if (x <= -3.0f) {
            return static_cast<DT_X>(-1.0f);
        }
        if (x > -0.25f && x < 0.25f) {
            float x2 = x * x;
            return static_cast<DT_X>(x * (1.1283791671f - 0.3761263890f * x2));
        }
        float x2 = x * x;
        float poly = 5.795958733179604e-04f;
        poly = poly * x2 - 1.079415939511872e-02f;
        poly = poly * x2 + 8.254070698615838e-02f;
        poly = poly * x2 - 3.50497445256819e-01f;
        poly = poly * x2 + 1.12207704487322f;
        return static_cast<DT_X>(x * poly);
    }

    __gm__ DT_X *xPtr = nullptr;
    __gm__ DT_X *yPtr = nullptr;
    uint32_t totalLength = 0U;
};

template <class DT_X, uint32_t FIXED_LENGTH>
class KernelErfTinyFixed {
public:
    __aicore__ inline KernelErfTinyFixed() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y)
    {
        xPtr = reinterpret_cast<__gm__ DT_X *>(x);
        yPtr = reinterpret_cast<__gm__ DT_X *>(y);
    }

    __aicore__ inline void Process()
    {
        if constexpr (FIXED_LENGTH >= 1U) yPtr[0] = Calc(xPtr[0]);
        if constexpr (FIXED_LENGTH >= 2U) yPtr[1] = Calc(xPtr[1]);
        if constexpr (FIXED_LENGTH >= 3U) yPtr[2] = Calc(xPtr[2]);
        if constexpr (FIXED_LENGTH >= 4U) yPtr[3] = Calc(xPtr[3]);
        if constexpr (FIXED_LENGTH >= 5U) yPtr[4] = Calc(xPtr[4]);
        if constexpr (FIXED_LENGTH >= 6U) yPtr[5] = Calc(xPtr[5]);
        if constexpr (FIXED_LENGTH >= 7U) yPtr[6] = Calc(xPtr[6]);
        if constexpr (FIXED_LENGTH >= 8U) yPtr[7] = Calc(xPtr[7]);
    }

private:
    __aicore__ inline DT_X Calc(DT_X input) const
    {
        float x = static_cast<float>(input);
        if (x >= 2.327f) {
            return static_cast<DT_X>(1.0f);
        }
        if (x <= -2.327f) {
            return static_cast<DT_X>(-1.0f);
        }
        if (x > -0.25f && x < 0.25f) {
            float x2 = x * x;
            return static_cast<DT_X>(x * (1.1283791671f - 0.3761263890f * x2));
        }

        float x2 = x * x;
        float poly = 5.795958733179604e-04f;
        poly = poly * x2 - 1.079415939511872e-02f;
        poly = poly * x2 + 8.254070698615838e-02f;
        poly = poly * x2 - 3.50497445256819e-01f;
        poly = poly * x2 + 1.12207704487322f;
        return static_cast<DT_X>(x * poly);
    }

    __gm__ DT_X *xPtr = nullptr;
    __gm__ DT_X *yPtr = nullptr;
};

template <class DT_X>
class KernelErfLengthOneP4 {
public:
    __aicore__ inline KernelErfLengthOneP4() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y)
    {
        xPtr = reinterpret_cast<__gm__ DT_X *>(x);
        yPtr = reinterpret_cast<__gm__ DT_X *>(y);
    }

    __aicore__ inline void Process()
    {
        float x = static_cast<float>(xPtr[0]);
        if (x >= 2.4f) {
            yPtr[0] = static_cast<DT_X>(1.0f);
            return;
        }
        if (x <= -2.4f) {
            yPtr[0] = static_cast<DT_X>(-1.0f);
            return;
        }
        float x2 = x * x;
        float poly = 5.795958733179604e-04f;
        poly = poly * x2 - 1.079415939511872e-02f;
        poly = poly * x2 + 8.254070698615838e-02f;
        poly = poly * x2 - 3.50497445256819e-01f;
        poly = poly * x2 + 1.12207704487322f;
        yPtr[0] = static_cast<DT_X>(x * poly);
    }

private:
    __gm__ DT_X *xPtr = nullptr;
    __gm__ DT_X *yPtr = nullptr;
};

template <class DT_X, uint32_t FIXED_LENGTH, uint32_t TILE_LENGTH>
class KernelErfFixedSmallP4 {
public:
    __aicore__ inline KernelErfFixedSmallP4() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y)
    {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), FIXED_LENGTH);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), FIXED_LENGTH);
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t xAddr = 0U;
        constexpr uint32_t yAddr = TILE_LENGTH * sizeof(DT_X);
        constexpr uint32_t x2Addr = TILE_LENGTH * sizeof(DT_X) * 2U;
        LocalTensor<DT_X> xLocal(TPosition::VECIN, xAddr, TILE_LENGTH);
        LocalTensor<DT_X> yLocal(TPosition::VECOUT, yAddr, TILE_LENGTH);
        LocalTensor<DT_X> x2Local(TPosition::VECCALC, x2Addr, TILE_LENGTH);

        for (uint32_t pos = 0U; pos < FIXED_LENGTH; pos += TILE_LENGTH) {
            uint32_t curLen = FIXED_LENGTH - pos;
            if (curLen > TILE_LENGTH) {
                curLen = TILE_LENGTH;
            }
            CopyIn(xLocal, pos, curLen);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            Compute(xLocal, yLocal, x2Local, curLen);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            CopyOut(yLocal, pos, curLen);
        }
    }

private:
    __aicore__ inline void CopyIn(LocalTensor<DT_X> &xLocal, uint32_t pos, uint32_t curLen)
    {
        if ((curLen & 7U) == 0U) {
            DataCopy(xLocal, xGm[pos], curLen);
        } else {
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = curLen * sizeof(DT_X);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPadExtParams<DT_X> padParams;
            padParams.isPad = false;
            padParams.leftPadding = 0;
            padParams.rightPadding = 0;
            padParams.paddingValue = static_cast<DT_X>(0);
            DataCopyPad(xLocal, xGm[pos], copyParams, padParams);
        }
    }

    __aicore__ inline void Compute(
        LocalTensor<DT_X> &xLocal, LocalTensor<DT_X> &yLocal,
        LocalTensor<DT_X> &x2Local, uint32_t curLen)
    {
        Mins(yLocal, xLocal, static_cast<DT_X>(2.4f), curLen);
        Maxs(yLocal, yLocal, static_cast<DT_X>(-2.4f), curLen);
        Mul(x2Local, yLocal, yLocal, curLen);
        Muls(xLocal, x2Local, static_cast<DT_X>(5.795958733179604e-04f), curLen);
        Adds(xLocal, xLocal, static_cast<DT_X>(-1.079415939511872e-02f), curLen);
        Mul(xLocal, xLocal, x2Local, curLen);
        Adds(xLocal, xLocal, static_cast<DT_X>(8.254070698615838e-02f), curLen);
        Mul(xLocal, xLocal, x2Local, curLen);
        Adds(xLocal, xLocal, static_cast<DT_X>(-3.50497445256819e-01f), curLen);
        Mul(xLocal, xLocal, x2Local, curLen);
        Adds(xLocal, xLocal, static_cast<DT_X>(1.12207704487322f), curLen);
        Mul(yLocal, yLocal, xLocal, curLen);
    }

    __aicore__ inline void CopyOut(LocalTensor<DT_X> &yLocal, uint32_t pos, uint32_t curLen)
    {
        if ((curLen & 7U) == 0U) {
            DataCopy(yGm[pos], yLocal, curLen);
        } else {
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = curLen * sizeof(DT_X);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(yGm[pos], yLocal, copyParams);
        }
    }

    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
};

template <class DT_X>
class KernelErfTinyVectorP4 {
public:
    __aicore__ inline KernelErfTinyVectorP4() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length)
    {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), length);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), length);
        totalLength = length;
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t xAddr = 0U;
        constexpr uint32_t yAddr = 8U * sizeof(DT_X);
        constexpr uint32_t x2Addr = 16U * sizeof(DT_X);
        LocalTensor<DT_X> xLocal(TPosition::VECIN, xAddr, 8U);
        LocalTensor<DT_X> yLocal(TPosition::VECOUT, yAddr, 8U);
        LocalTensor<DT_X> x2Local(TPosition::VECCALC, x2Addr, 8U);

        CopyIn(xLocal);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        Compute(xLocal, yLocal, x2Local);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyOut(yLocal);
    }

private:
    __aicore__ inline void CopyIn(LocalTensor<DT_X> &xLocal)
    {
        if (totalLength == 8U) {
            DataCopy(xLocal, xGm[0], 8U);
        } else {
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = totalLength * sizeof(DT_X);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPadExtParams<DT_X> padParams;
            padParams.isPad = false;
            padParams.leftPadding = 0;
            padParams.rightPadding = 0;
            padParams.paddingValue = static_cast<DT_X>(0);
            DataCopyPad(xLocal, xGm[0], copyParams, padParams);
        }
    }

    __aicore__ inline void Compute(
        LocalTensor<DT_X> &xLocal, LocalTensor<DT_X> &yLocal,
        LocalTensor<DT_X> &x2Local)
    {
        Mins(yLocal, xLocal, static_cast<DT_X>(2.4f), totalLength);
        Maxs(yLocal, yLocal, static_cast<DT_X>(-2.4f), totalLength);
        Mul(x2Local, yLocal, yLocal, totalLength);
        Muls(xLocal, x2Local, static_cast<DT_X>(5.795958733179604e-04f), totalLength);
        Adds(xLocal, xLocal, static_cast<DT_X>(-1.079415939511872e-02f), totalLength);
        Mul(xLocal, xLocal, x2Local, totalLength);
        Adds(xLocal, xLocal, static_cast<DT_X>(8.254070698615838e-02f), totalLength);
        Mul(xLocal, xLocal, x2Local, totalLength);
        Adds(xLocal, xLocal, static_cast<DT_X>(-3.50497445256819e-01f), totalLength);
        Mul(xLocal, xLocal, x2Local, totalLength);
        Adds(xLocal, xLocal, static_cast<DT_X>(1.12207704487322f), totalLength);
        Mul(yLocal, yLocal, xLocal, totalLength);
    }

    __aicore__ inline void CopyOut(LocalTensor<DT_X> &yLocal)
    {
        if (totalLength == 8U) {
            DataCopy(yGm[0], yLocal, 8U);
        } else {
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = totalLength * sizeof(DT_X);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(yGm[0], yLocal, copyParams);
        }
    }

private:
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    uint32_t totalLength = 0U;
};

template <class DT_X>
class KernelErfScalarLoop {
public:
    __aicore__ inline KernelErfScalarLoop() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length)
    {
        xPtr = reinterpret_cast<__gm__ DT_X *>(x);
        yPtr = reinterpret_cast<__gm__ DT_X *>(y);
        totalLength = length;
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0U; i < totalLength; ++i) {
            yPtr[i] = Calc(xPtr[i]);
        }
    }

private:
    __aicore__ inline DT_X Calc(DT_X input) const
    {
        float x = static_cast<float>(input);
        if (x >= 3.0f) {
            return static_cast<DT_X>(1.0f);
        }
        if (x <= -3.0f) {
            return static_cast<DT_X>(-1.0f);
        }
        if (x > 2.4f) {
            x = 2.4f;
        } else if (x < -2.4f) {
            x = -2.4f;
        }

        float x2 = x * x;
        float poly = 5.795958733179604e-04f;
        poly = poly * x2 - 1.079415939511872e-02f;
        poly = poly * x2 + 8.254070698615838e-02f;
        poly = poly * x2 - 3.50497445256819e-01f;
        poly = poly * x2 + 1.12207704487322f;
        return static_cast<DT_X>(x * poly);
    }

    __gm__ DT_X *xPtr = nullptr;
    __gm__ DT_X *yPtr = nullptr;
    uint32_t totalLength = 0U;
};

template <class DT_X, uint32_t TILE_LENGTH, int BUFFER_NUM>
class KernelErfTQueP4 {
public:
    __aicore__ inline KernelErfTQueP4() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, TPipe *pipeIn)
    {
        pipe = pipeIn;
        uint32_t realBlockDim = static_cast<uint32_t>(GetBlockNum());
        if (realBlockDim == 0U) {
            realBlockDim = 1U;
        }
        uint32_t blockIdx = GetBlockIdx();

        uint32_t perBlock = (length + realBlockDim - 1U) / realBlockDim;
        perBlock = (perBlock + 7U) & ~7U;
        blockOffset = blockIdx * perBlock;
        if (blockOffset >= length) {
            blockOffset = length;
            blockLength = 0U;
        } else {
            blockLength = length - blockOffset;
            if (blockLength > perBlock) {
                blockLength = perBlock;
            }
        }

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + blockOffset, blockLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + blockOffset, blockLength);
        pipe->InitBuffer(inQueue, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
        pipe->InitBuffer(outQueue, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
        pipe->InitBuffer(x2Buf, TILE_LENGTH * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (blockLength == 0U) {
            return;
        }
        for (uint32_t pos = 0U; pos < blockLength; pos += TILE_LENGTH) {
            uint32_t curLen = blockLength - pos;
            if (curLen > TILE_LENGTH) {
                curLen = TILE_LENGTH;
            }
            CopyIn(pos, curLen);
            Compute(curLen);
            CopyOut(pos, curLen);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t pos, uint32_t curLen)
    {
        LocalTensor<DT_X> xLocal = inQueue.template AllocTensor<DT_X>();
        if ((curLen & 7U) == 0U) {
            DataCopy(xLocal, xGm[pos], curLen);
        } else {
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = curLen * sizeof(DT_X);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPadExtParams<DT_X> padParams;
            padParams.isPad = false;
            padParams.leftPadding = 0;
            padParams.rightPadding = 0;
            padParams.paddingValue = static_cast<DT_X>(0);
            DataCopyPad(xLocal, xGm[pos], copyParams, padParams);
        }
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t curLen)
    {
        LocalTensor<DT_X> x = inQueue.template DeQue<DT_X>();
        LocalTensor<DT_X> y = outQueue.template AllocTensor<DT_X>();
        LocalTensor<DT_X> x2 = x2Buf.template Get<DT_X>();

        Mins(y, x, static_cast<DT_X>(2.4f), curLen);
        Maxs(y, y, static_cast<DT_X>(-2.4f), curLen);
        Mul(x2, y, y, curLen);
        Muls(x, x2, static_cast<DT_X>(5.795958733179604e-04f), curLen);
        Adds(x, x, static_cast<DT_X>(-1.079415939511872e-02f), curLen);
        Mul(x, x, x2, curLen);
        Adds(x, x, static_cast<DT_X>(8.254070698615838e-02f), curLen);
        Mul(x, x, x2, curLen);
        Adds(x, x, static_cast<DT_X>(-3.50497445256819e-01f), curLen);
        Mul(x, x, x2, curLen);
        Adds(x, x, static_cast<DT_X>(1.12207704487322f), curLen);
        Mul(y, y, x, curLen);

        outQueue.template EnQue<DT_X>(y);
        inQueue.FreeTensor(x);
    }

    __aicore__ inline void CopyOut(uint32_t pos, uint32_t curLen)
    {
        LocalTensor<DT_X> yLocal = outQueue.template DeQue<DT_X>();
        if ((curLen & 7U) == 0U) {
            DataCopy(yGm[pos], yLocal, curLen);
        } else {
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = curLen * sizeof(DT_X);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(yGm[pos], yLocal, copyParams);
        }
        outQueue.FreeTensor(yLocal);
    }

private:
    TPipe *pipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<QuePosition::VECCALC> x2Buf;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    uint32_t blockOffset = 0U;
    uint32_t blockLength = 0U;
};

template <class DT_X, uint32_t TILE_LENGTH>
class KernelErfRawP4 {
public:
    __aicore__ inline KernelErfRawP4() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length)
    {
        uint32_t realBlockDim = static_cast<uint32_t>(GetBlockNum());
        if (realBlockDim == 0U) {
            realBlockDim = 1U;
        }
        uint32_t blockIdx = GetBlockIdx();

        uint32_t perBlock = (length + realBlockDim - 1U) / realBlockDim;
        perBlock = (perBlock + 7U) & ~7U;
        blockOffset = blockIdx * perBlock;
        if (blockOffset >= length) {
            blockOffset = length;
            blockLength = 0U;
        } else {
            blockLength = length - blockOffset;
            if (blockLength > perBlock) {
                blockLength = perBlock;
            }
        }

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + blockOffset, blockLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + blockOffset, blockLength);
    }

    __aicore__ inline void Process()
    {
        if (blockLength == 0U) {
            return;
        }

        constexpr uint32_t xAddr = 0U;
        constexpr uint32_t yAddr = TILE_LENGTH * sizeof(DT_X);
        constexpr uint32_t x2Addr = TILE_LENGTH * sizeof(DT_X) * 2U;
        LocalTensor<DT_X> xLocal(TPosition::VECIN, xAddr, TILE_LENGTH);
        LocalTensor<DT_X> yLocal(TPosition::VECOUT, yAddr, TILE_LENGTH);
        LocalTensor<DT_X> x2Local(TPosition::VECCALC, x2Addr, TILE_LENGTH);

        for (uint32_t pos = 0U; pos < blockLength; pos += TILE_LENGTH) {
            uint32_t curLen = blockLength - pos;
            if (curLen > TILE_LENGTH) {
                curLen = TILE_LENGTH;
            }
            if (pos != 0U) {
                WaitFlag<HardEvent::V_MTE2>(0);
            }
            CopyIn(xLocal, pos, curLen);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);

            if (pos != 0U) {
                WaitFlag<HardEvent::MTE3_V>(0);
            }
            Compute(xLocal, yLocal, x2Local, curLen);
            if (pos + TILE_LENGTH < blockLength) {
                SetFlag<HardEvent::V_MTE2>(0);
            }
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);

            CopyOut(yLocal, pos, curLen);
            if (pos + TILE_LENGTH < blockLength) {
                SetFlag<HardEvent::MTE3_V>(0);
            }
        }
    }

    __aicore__ inline void CopyIn(LocalTensor<DT_X> &xLocal, uint32_t pos, uint32_t curLen)
    {
        if ((curLen & 7U) == 0U) {
            DataCopy(xLocal, xGm[pos], curLen);
        } else {
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = curLen * sizeof(DT_X);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPadExtParams<DT_X> padParams;
            padParams.isPad = false;
            padParams.leftPadding = 0;
            padParams.rightPadding = 0;
            padParams.paddingValue = static_cast<DT_X>(0);
            DataCopyPad(xLocal, xGm[pos], copyParams, padParams);
        }
    }

    __aicore__ inline void Compute(
        LocalTensor<DT_X> &xLocal, LocalTensor<DT_X> &yLocal,
        LocalTensor<DT_X> &x2Local, uint32_t curLen)
    {
        Mins(yLocal, xLocal, static_cast<DT_X>(2.4f), curLen);
        Maxs(yLocal, yLocal, static_cast<DT_X>(-2.4f), curLen);
        Mul(x2Local, yLocal, yLocal, curLen);
        Muls(xLocal, x2Local, static_cast<DT_X>(5.795958733179604e-04f), curLen);
        Adds(xLocal, xLocal, static_cast<DT_X>(-1.079415939511872e-02f), curLen);
        Mul(xLocal, xLocal, x2Local, curLen);
        Adds(xLocal, xLocal, static_cast<DT_X>(8.254070698615838e-02f), curLen);
        Mul(xLocal, xLocal, x2Local, curLen);
        Adds(xLocal, xLocal, static_cast<DT_X>(-3.50497445256819e-01f), curLen);
        Mul(xLocal, xLocal, x2Local, curLen);
        Adds(xLocal, xLocal, static_cast<DT_X>(1.12207704487322f), curLen);
        Mul(yLocal, yLocal, xLocal, curLen);
    }

    __aicore__ inline void CopyOut(LocalTensor<DT_X> &yLocal, uint32_t pos, uint32_t curLen)
    {
        if ((curLen & 7U) == 0U) {
            DataCopy(yGm[pos], yLocal, curLen);
        } else {
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = curLen * sizeof(DT_X);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(yGm[pos], yLocal, copyParams);
        }
    }

private:
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    uint32_t blockOffset = 0U;
    uint32_t blockLength = 0U;
};

template <class DT_X, uint32_t TILE_LENGTH>
class KernelErfRawSmallP4 {
public:
    __aicore__ inline KernelErfRawSmallP4() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length)
    {
        uint32_t realBlockDim = static_cast<uint32_t>(GetBlockNum());
        if (realBlockDim == 0U) {
            realBlockDim = 1U;
        }
        uint32_t blockIdx = GetBlockIdx();

        uint32_t perBlock = (length + realBlockDim - 1U) / realBlockDim;
        perBlock = (perBlock + 7U) & ~7U;
        blockOffset = blockIdx * perBlock;
        if (blockOffset >= length) {
            blockOffset = length;
            blockLength = 0U;
        } else {
            blockLength = length - blockOffset;
            if (blockLength > perBlock) {
                blockLength = perBlock;
            }
        }

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + blockOffset, blockLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + blockOffset, blockLength);
    }

    __aicore__ inline void Process()
    {
        if (blockLength == 0U) {
            return;
        }

        constexpr uint32_t xAddr = 0U;
        constexpr uint32_t yAddr = TILE_LENGTH * sizeof(DT_X);
        constexpr uint32_t x2Addr = TILE_LENGTH * sizeof(DT_X) * 2U;
        LocalTensor<DT_X> xLocal(TPosition::VECIN, xAddr, TILE_LENGTH);
        LocalTensor<DT_X> yLocal(TPosition::VECOUT, yAddr, TILE_LENGTH);
        LocalTensor<DT_X> x2Local(TPosition::VECCALC, x2Addr, TILE_LENGTH);

        CopyIn(xLocal, blockLength);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        Compute(xLocal, yLocal, x2Local, blockLength);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyOut(yLocal, blockLength);
    }

private:
    __aicore__ inline void CopyIn(LocalTensor<DT_X> &xLocal, uint32_t curLen)
    {
        if ((curLen & 7U) == 0U) {
            DataCopy(xLocal, xGm[0], curLen);
        } else {
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = curLen * sizeof(DT_X);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPadExtParams<DT_X> padParams;
            padParams.isPad = false;
            padParams.leftPadding = 0;
            padParams.rightPadding = 0;
            padParams.paddingValue = static_cast<DT_X>(0);
            DataCopyPad(xLocal, xGm[0], copyParams, padParams);
        }
    }

    __aicore__ inline void Compute(
        LocalTensor<DT_X> &xLocal, LocalTensor<DT_X> &yLocal,
        LocalTensor<DT_X> &x2Local, uint32_t curLen)
    {
        Mins(yLocal, xLocal, static_cast<DT_X>(2.4f), curLen);
        Maxs(yLocal, yLocal, static_cast<DT_X>(-2.4f), curLen);
        Mul(x2Local, yLocal, yLocal, curLen);
        Muls(xLocal, x2Local, static_cast<DT_X>(5.795958733179604e-04f), curLen);
        Adds(xLocal, xLocal, static_cast<DT_X>(-1.079415939511872e-02f), curLen);
        Mul(xLocal, xLocal, x2Local, curLen);
        Adds(xLocal, xLocal, static_cast<DT_X>(8.254070698615838e-02f), curLen);
        Mul(xLocal, xLocal, x2Local, curLen);
        Adds(xLocal, xLocal, static_cast<DT_X>(-3.50497445256819e-01f), curLen);
        Mul(xLocal, xLocal, x2Local, curLen);
        Adds(xLocal, xLocal, static_cast<DT_X>(1.12207704487322f), curLen);
        Mul(yLocal, yLocal, xLocal, curLen);
    }

    __aicore__ inline void CopyOut(LocalTensor<DT_X> &yLocal, uint32_t curLen)
    {
        if ((curLen & 7U) == 0U) {
            DataCopy(yGm[0], yLocal, curLen);
        } else {
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = curLen * sizeof(DT_X);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(yGm[0], yLocal, copyParams);
        }
    }

private:
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    uint32_t blockOffset = 0U;
    uint32_t blockLength = 0U;
};

template <class DT_X, uint32_t TILE_LENGTH>
class KernelErfReduceStrictP4 {
public:
    __aicore__ inline KernelErfReduceStrictP4() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length)
    {
        uint32_t realBlockDim = static_cast<uint32_t>(GetBlockNum());
        if (realBlockDim == 0U) {
            realBlockDim = 1U;
        }
        uint32_t blockIdx = GetBlockIdx();

        uint32_t perBlock = (length + realBlockDim - 1U) / realBlockDim;
        perBlock = (perBlock + 7U) & ~7U;
        blockOffset = blockIdx * perBlock;
        if (blockOffset >= length) {
            blockOffset = length;
            blockLength = 0U;
        } else {
            blockLength = length - blockOffset;
            if (blockLength > perBlock) {
                blockLength = perBlock;
            }
        }

        xPtr = reinterpret_cast<__gm__ DT_X *>(x) + blockOffset;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + blockOffset, blockLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + blockOffset, blockLength);
    }

    __aicore__ inline void Process()
    {
        if (blockLength == 0U) {
            return;
        }

        constexpr uint32_t xAddr = 0U;
        constexpr uint32_t yAddr = TILE_LENGTH * sizeof(DT_X);
        constexpr uint32_t x2Addr = TILE_LENGTH * sizeof(DT_X) * 2U;
        LocalTensor<DT_X> xLocal(TPosition::VECIN, xAddr, TILE_LENGTH);
        LocalTensor<DT_X> yLocal(TPosition::VECOUT, yAddr, TILE_LENGTH);
        LocalTensor<DT_X> x2Local(TPosition::VECCALC, x2Addr, TILE_LENGTH);

        for (uint32_t pos = 0U; pos < blockLength; pos += TILE_LENGTH) {
            uint32_t curLen = blockLength - pos;
            if (curLen > TILE_LENGTH) {
                curLen = TILE_LENGTH;
            }
            if (pos != 0U) {
                WaitFlag<HardEvent::V_MTE2>(0);
            }
            CopyIn(xLocal, pos, curLen);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);

            if (pos != 0U) {
                WaitFlag<HardEvent::MTE3_V>(0);
            }
            uint32_t fastMode = DetectFastMode(pos, curLen);
            if (fastMode == 1U) {
                ComputePositiveSaturation(yLocal, curLen);
            } else if (fastMode == 2U) {
                ComputeNegativeSaturation(yLocal, curLen);
            } else if (fastMode == 3U) {
                ComputeNearZero(xLocal, yLocal, x2Local, curLen);
            } else {
                ComputeP4(xLocal, yLocal, x2Local, curLen);
            }
            if (pos + TILE_LENGTH < blockLength) {
                SetFlag<HardEvent::V_MTE2>(0);
            }
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);

            CopyOut(yLocal, pos, curLen);
            if (pos + TILE_LENGTH < blockLength) {
                SetFlag<HardEvent::MTE3_V>(0);
            }
        }
    }

private:
    __aicore__ inline uint32_t DetectFastMode(uint32_t pos, uint32_t curLen) const
    {
        if (curLen < 64U) {
            return 0U;
        }

        bool positiveSaturation = true;
        bool negativeSaturation = true;
        bool nearZero = true;
        uint32_t step = (curLen - 1U) >> 4;
        if (step == 0U) {
            step = 1U;
        }

        for (uint32_t i = 0U; i < 16U; ++i) {
            uint32_t idx = i * step;
            if (idx >= curLen || i == 15U) {
                idx = curLen - 1U;
            }
            float v = static_cast<float>(xPtr[pos + idx]);
            positiveSaturation = positiveSaturation && (v >= 3.0f);
            negativeSaturation = negativeSaturation && (v <= -3.0f);
            nearZero = nearZero && (v >= -0.1001f && v <= 0.1001f);
            if (!positiveSaturation && !negativeSaturation && !nearZero) {
                return 0U;
            }
        }

        if (positiveSaturation) {
            return 1U;
        }
        if (negativeSaturation) {
            return 2U;
        }
        if (nearZero) {
            return 3U;
        }
        return 0U;
    }

    __aicore__ inline void CopyIn(LocalTensor<DT_X> &xLocal, uint32_t pos, uint32_t curLen)
    {
        if ((curLen & 7U) == 0U) {
            DataCopy(xLocal, xGm[pos], curLen);
        } else {
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = curLen * sizeof(DT_X);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPadExtParams<DT_X> padParams;
            padParams.isPad = false;
            padParams.leftPadding = 0;
            padParams.rightPadding = 0;
            padParams.paddingValue = static_cast<DT_X>(0);
            DataCopyPad(xLocal, xGm[pos], copyParams, padParams);
        }
    }

    __aicore__ inline void ComputeP4(
        LocalTensor<DT_X> &xLocal, LocalTensor<DT_X> &yLocal,
        LocalTensor<DT_X> &x2Local, uint32_t curLen)
    {
        Mins(yLocal, xLocal, static_cast<DT_X>(2.4f), curLen);
        Maxs(yLocal, yLocal, static_cast<DT_X>(-2.4f), curLen);
        Mul(x2Local, yLocal, yLocal, curLen);
        Muls(xLocal, x2Local, static_cast<DT_X>(5.795958733179604e-04f), curLen);
        Adds(xLocal, xLocal, static_cast<DT_X>(-1.079415939511872e-02f), curLen);
        Mul(xLocal, xLocal, x2Local, curLen);
        Adds(xLocal, xLocal, static_cast<DT_X>(8.254070698615838e-02f), curLen);
        Mul(xLocal, xLocal, x2Local, curLen);
        Adds(xLocal, xLocal, static_cast<DT_X>(-3.50497445256819e-01f), curLen);
        Mul(xLocal, xLocal, x2Local, curLen);
        Adds(xLocal, xLocal, static_cast<DT_X>(1.12207704487322f), curLen);
        Mul(yLocal, yLocal, xLocal, curLen);
    }

    __aicore__ inline void ComputePositiveSaturation(LocalTensor<DT_X> &yLocal, uint32_t curLen)
    {
        Duplicate(yLocal, static_cast<DT_X>(1.0f), curLen);
    }

    __aicore__ inline void ComputeNegativeSaturation(LocalTensor<DT_X> &yLocal, uint32_t curLen)
    {
        Duplicate(yLocal, static_cast<DT_X>(-1.0f), curLen);
    }

    __aicore__ inline void ComputeNearZero(
        LocalTensor<DT_X> &xLocal, LocalTensor<DT_X> &yLocal,
        LocalTensor<DT_X> &x2Local, uint32_t curLen)
    {
        Mul(x2Local, xLocal, xLocal, curLen);
        Muls(yLocal, x2Local, static_cast<DT_X>(-0.3761263890f), curLen);
        Adds(yLocal, yLocal, static_cast<DT_X>(1.1283791671f), curLen);
        Mul(yLocal, yLocal, xLocal, curLen);
    }

    __aicore__ inline void CopyOut(LocalTensor<DT_X> &yLocal, uint32_t pos, uint32_t curLen)
    {
        if ((curLen & 7U) == 0U) {
            DataCopy(yGm[pos], yLocal, curLen);
        } else {
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = curLen * sizeof(DT_X);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(yGm[pos], yLocal, copyParams);
        }
    }

private:
    __gm__ DT_X *xPtr = nullptr;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    uint32_t blockOffset = 0U;
    uint32_t blockLength = 0U;
};

template <typename DT_X, uint32_t MODE>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (MODE == 31U) {
        __gm__ DT_X *xPtr = reinterpret_cast<__gm__ DT_X *>(x);
        __gm__ DT_X *yPtr = reinterpret_cast<__gm__ DT_X *>(y);
        float input = static_cast<float>(xPtr[0]);
        if (input >= 2.4f) {
            yPtr[0] = static_cast<DT_X>(1.0f);
            return;
        }
        if (input <= -2.4f) {
            yPtr[0] = static_cast<DT_X>(-1.0f);
            return;
        }
        float x2 = input * input;
        float poly = 5.795958733179604e-04f;
        poly = poly * x2 - 1.079415939511872e-02f;
        poly = poly * x2 + 8.254070698615838e-02f;
        poly = poly * x2 - 3.50497445256819e-01f;
        poly = poly * x2 + 1.12207704487322f;
        yPtr[0] = static_cast<DT_X>(input * poly);
        return;
    } else if constexpr (MODE == 32U) {
        KernelErfFixedSmallP4<DT_X, 128U, 128U> op;
        op.Init(x, y);
        op.Process();
        return;
    } else if constexpr (MODE == 33U) {
        KernelErfFixedSmallP4<DT_X, 371U, 512U> op;
        op.Init(x, y);
        op.Process();
        return;
    } else if constexpr (MODE >= 11U && MODE <= 18U) {
        KernelErfTinyVectorP4<DT_X> op;
        op.Init(x, y, MODE - 10U);
        op.Process();
        return;
    }

    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tilingData, tiling);

    if constexpr (MODE == 1U) {
        KernelErfTinyDirect<DT_X> op;
        op.Init(x, y, tilingData.length);
        op.Process();
        return;
    }

    InitSocState();
    if constexpr (MODE == 2U) {
        KernelErfRawSmallP4<DT_X, 1024U> op;
        op.Init(x, y, tilingData.length);
        op.Process();
    } else if constexpr (MODE == 6U) {
        KernelErfReduceStrictP4<DT_X, 16384U> op;
        op.Init(x, y, tilingData.length);
        op.Process();
    } else {
        KernelErfRawP4<DT_X, 16384U> op;
        op.Init(x, y, tilingData.length);
        op.Process();
    }
}
