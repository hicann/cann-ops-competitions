// Kernel侧核函数实现
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

constexpr uint32_t SINGLE_BUFFER_NUM = 1;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t DOUBLE_BUFFER_MIN_LENGTH = 65536;
constexpr uint32_t TINY_DIRECT_N = 64;
constexpr uint32_t SMALL_FAST_LENGTH = 4096;
constexpr uint32_t PATH_V13C_POLY_SMALL = 0;
constexpr uint32_t PATH_TINY64_FAST = 1;
constexpr uint32_t PATH_MID_NATIVE_REVERT = 2;

template <class DT_X>
class KernelErfTinyDirect {
public:
    __aicore__ inline KernelErfTinyDirect() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t totalLength) {
        this->totalLength = totalLength;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
    }

    __aicore__ inline void Process() {
        if (this->totalLength == 0) {
            return;
        }
        for (uint32_t i = 0; i < static_cast<uint32_t>(this->totalLength); ++i) {
            DT_X x = xGm.GetValue(i);
            DT_X y = FastErfScalar(x);
            yGm.SetValue(i, y);
        }
        DataCacheCleanAndInvalid<DT_X, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(yGm);
    }

private:
    __aicore__ inline DT_X FastErfScalar(DT_X x) {
        DT_X xClip = x;
        if (xClip > static_cast<DT_X>(3.0f)) {
            xClip = static_cast<DT_X>(3.0f);
        }
        if (xClip < static_cast<DT_X>(-3.0f)) {
            xClip = static_cast<DT_X>(-3.0f);
        }
        DT_X t = xClip * xClip;
        DT_X p = static_cast<DT_X>(4.0745744378918134e-10f);
        p = p * t + static_cast<DT_X>(-2.2750374151783939e-8f);
        p = p * t + static_cast<DT_X>(5.7875516457819682e-7f);
        p = p * t + static_cast<DT_X>(-9.0015004556574117e-6f);
        p = p * t + static_cast<DT_X>(9.7410461048522186e-5f);
        p = p * t + static_cast<DT_X>(-7.9360535417300805e-4f);
        p = p * t + static_cast<DT_X>(5.1196939569625932e-3f);
        p = p * t + static_cast<DT_X>(-2.6760455816072556e-2f);
        p = p * t + static_cast<DT_X>(1.1278225992880547e-1f);
        p = p * t + static_cast<DT_X>(-3.7611487211070921e-1f);
        p = p * t + static_cast<DT_X>(1.1283787714929709f);
        return xClip * p;
    }

private:
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    uint64_t totalLength = 0;
};

template <class DT_X>
class KernelErfSmallTbuf {
public:
    __aicore__ inline KernelErfSmallTbuf() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t totalLength, uint32_t tileLength, uint32_t tmpSize) {
        this->totalLength = totalLength;
        this->tileLength = tileLength;
        this->tmpSize = tmpSize;
        if (this->totalLength == 0 || this->tileLength == 0 || this->tmpSize == 0) {
            this->thisBlockLength = 0;
            return;
        }

        this->thisBlockLength = this->totalLength;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
        pipe.InitBuffer(inBuffer, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outBuffer, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(tmpBuffer, this->tmpSize);
    }

    __aicore__ inline void Process() {
        if (this->thisBlockLength == 0) {
            return;
        }

        uint32_t curLength = static_cast<uint32_t>(this->thisBlockLength);
        LocalTensor<DT_X> xLocal = inBuffer.Get<DT_X>();
        LocalTensor<DT_X> yLocal = outBuffer.Get<DT_X>();
        if ((curLength & 7U) == 0) {
            DataCopy(xLocal, xGm[0], curLength);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, xGm[0], copyParams, padParams);
        }
        PipeBarrier<PIPE_ALL>();

        LocalTensor<uint8_t> tmpLocal = tmpBuffer.Get<uint8_t>();
        Erf<DT_X, false>(yLocal, xLocal, tmpLocal, curLength);
        PipeBarrier<PIPE_ALL>();

        if ((curLength & 7U) == 0) {
            DataCopy(yGm[0], yLocal, curLength);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(yGm[0], yLocal, copyParams);
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECIN> inBuffer;
    TBuf<TPosition::VECOUT> outBuffer;
    TBuf<TPosition::VECCALC> tmpBuffer;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    uint64_t totalLength = 0;
    uint64_t thisBlockLength = 0;
    uint32_t tileLength = 0;
    uint32_t tmpSize = 0;
};

template <class DT_X>
class KernelErfSmallPolyTbuf {
public:
    __aicore__ inline KernelErfSmallPolyTbuf() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t totalLength, uint32_t tileLength) {
        this->totalLength = totalLength;
        this->tileLength = tileLength;
        if (this->totalLength == 0 || this->tileLength == 0) {
            this->thisBlockLength = 0;
            return;
        }

        this->thisBlockLength = this->totalLength;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
        pipe.InitBuffer(inBuffer, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outBuffer, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(xClipBuffer, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(tBuffer, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (this->thisBlockLength == 0) {
            return;
        }

        uint32_t curLength = static_cast<uint32_t>(this->thisBlockLength);
        LocalTensor<DT_X> xLocal = inBuffer.Get<DT_X>();
        LocalTensor<DT_X> yLocal = outBuffer.Get<DT_X>();
        if ((curLength & 7U) == 0) {
            DataCopy(xLocal, xGm[0], curLength);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, xGm[0], copyParams, padParams);
        }
        PipeBarrier<PIPE_ALL>();

        FastErfPoly(yLocal, xLocal, curLength);
        PipeBarrier<PIPE_ALL>();

        if ((curLength & 7U) == 0) {
            DataCopy(yGm[0], yLocal, curLength);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(yGm[0], yLocal, copyParams);
        }
    }

private:
    __aicore__ inline void FastErfPoly(LocalTensor<DT_X> yLocal, LocalTensor<DT_X> xLocal, uint32_t curLength) {
        LocalTensor<DT_X> xClip = xClipBuffer.Get<DT_X>();
        LocalTensor<DT_X> tLocal = tBuffer.Get<DT_X>();

        Mins(xClip, xLocal, static_cast<DT_X>(3.0f), curLength);
        Maxs(xClip, xClip, static_cast<DT_X>(-3.0f), curLength);
        Mul(tLocal, xClip, xClip, curLength);

        Duplicate(yLocal, static_cast<DT_X>(4.0745744378918134e-10f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(-2.2750374151783939e-8f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(5.7875516457819682e-7f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(-9.0015004556574117e-6f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(9.7410461048522186e-5f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(-7.9360535417300805e-4f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(5.1196939569625932e-3f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(-2.6760455816072556e-2f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(1.1278225992880547e-1f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(-3.7611487211070921e-1f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(1.1283787714929709f), curLength);
        Mul(yLocal, yLocal, xClip, curLength);
        PipeBarrier<PIPE_V>();
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECIN> inBuffer;
    TBuf<TPosition::VECOUT> outBuffer;
    TBuf<TPosition::VECCALC> xClipBuffer;
    TBuf<TPosition::VECCALC> tBuffer;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    uint64_t totalLength = 0;
    uint64_t thisBlockLength = 0;
    uint32_t tileLength = 0;
};

template <class DT_X>
class KernelErfSingleBuffer {
public:
    __aicore__ inline KernelErfSingleBuffer() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t totalLength, uint64_t blockLength,
        uint32_t tileLength, uint32_t blockDim) {
        this->totalLength = totalLength;
        this->blockLength = blockLength;
        this->tileLength = tileLength;
        this->blockDim = blockDim;

        uint64_t start = static_cast<uint64_t>(GetBlockIdx()) * this->blockLength;
        if (this->totalLength == 0 || start >= this->totalLength || this->blockLength == 0 || this->tileLength == 0) {
            this->thisBlockLength = 0;
            return;
        }

        this->thisBlockLength = this->totalLength - start;
        if (this->thisBlockLength > this->blockLength) {
            this->thisBlockLength = this->blockLength;
        }

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + start);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + start);
        pipe.InitBuffer(inQueueX, SINGLE_BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, SINGLE_BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(xClipBuffer, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(tBuffer, this->tileLength * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        if (this->thisBlockLength == 0) {
            return;
        }

        for (uint64_t offset = 0; offset < this->thisBlockLength; offset += this->tileLength) {
            uint64_t remain = this->thisBlockLength - offset;
            uint32_t curLength = static_cast<uint32_t>(
                remain < static_cast<uint64_t>(this->tileLength) ? remain : static_cast<uint64_t>(this->tileLength));
            CopyIn(offset, curLength);
            Compute(curLength);
            CopyOut(offset, curLength);
        }

    }
private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t curLength) {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(DT_X)), 0, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t curLength) {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        FastErfPoly(yLocal, xLocal, curLength);
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t curLength) {
        LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(DT_X)), 0, 0, 0};
        DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void FastErfPoly(LocalTensor<DT_X> yLocal, LocalTensor<DT_X> xLocal, uint32_t curLength) {
        LocalTensor<DT_X> xClip = xClipBuffer.Get<DT_X>();
        LocalTensor<DT_X> tLocal = tBuffer.Get<DT_X>();

        Mins(xClip, xLocal, static_cast<DT_X>(3.0f), curLength);
        Maxs(xClip, xClip, static_cast<DT_X>(-3.0f), curLength);
        Mul(tLocal, xClip, xClip, curLength);

        Duplicate(yLocal, static_cast<DT_X>(4.0745744378918134e-10f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(-2.2750374151783939e-8f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(5.7875516457819682e-7f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(-9.0015004556574117e-6f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(9.7410461048522186e-5f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(-7.9360535417300805e-4f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(5.1196939569625932e-3f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(-2.6760455816072556e-2f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(1.1278225992880547e-1f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(-3.7611487211070921e-1f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(1.1283787714929709f), curLength);
        Mul(yLocal, yLocal, xClip, curLength);
        PipeBarrier<PIPE_V>();
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, SINGLE_BUFFER_NUM> inQueueX;
    TQue<TPosition::VECOUT, SINGLE_BUFFER_NUM> outQueueY;
    TBuf<TPosition::VECCALC> xClipBuffer;
    TBuf<TPosition::VECCALC> tBuffer;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    uint64_t totalLength = 0;
    uint64_t blockLength = 0;
    uint64_t thisBlockLength = 0;
    uint32_t tileLength = 0;
    uint32_t blockDim = 0;
};

template <class DT_X>
class KernelErfDoubleBuffer {
public:
    __aicore__ inline KernelErfDoubleBuffer() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t totalLength, uint64_t blockLength,
        uint32_t tileLength, uint32_t blockDim) {
        this->totalLength = totalLength;
        this->blockLength = blockLength;
        this->tileLength = tileLength;
        this->blockDim = blockDim;

        uint64_t start = static_cast<uint64_t>(GetBlockIdx()) * this->blockLength;
        if (this->totalLength == 0 || start >= this->totalLength || this->blockLength == 0 || this->tileLength == 0) {
            this->thisBlockLength = 0;
            return;
        }

        this->thisBlockLength = this->totalLength - start;
        if (this->thisBlockLength > this->blockLength) {
            this->thisBlockLength = this->blockLength;
        }

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + start);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + start);
        pipe.InitBuffer(inQueueX, DOUBLE_BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, DOUBLE_BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(xClipBuffer, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(tBuffer, this->tileLength * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        if (this->thisBlockLength == 0) {
            return;
        }

        for (uint64_t offset = 0; offset < this->thisBlockLength; offset += this->tileLength) {
            uint64_t remain = this->thisBlockLength - offset;
            uint32_t curLength = static_cast<uint32_t>(
                remain < static_cast<uint64_t>(this->tileLength) ? remain : static_cast<uint64_t>(this->tileLength));
            CopyIn(offset, curLength);
            Compute(curLength);
            CopyOut(offset, curLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t curLength) {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(DT_X)), 0, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t curLength) {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        FastErfPoly(yLocal, xLocal, curLength);
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t curLength) {
        LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(DT_X)), 0, 0, 0};
        DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void FastErfPoly(LocalTensor<DT_X> yLocal, LocalTensor<DT_X> xLocal, uint32_t curLength) {
        LocalTensor<DT_X> xClip = xClipBuffer.Get<DT_X>();
        LocalTensor<DT_X> tLocal = tBuffer.Get<DT_X>();

        Mins(xClip, xLocal, static_cast<DT_X>(3.0f), curLength);
        Maxs(xClip, xClip, static_cast<DT_X>(-3.0f), curLength);
        Mul(tLocal, xClip, xClip, curLength);

        Duplicate(yLocal, static_cast<DT_X>(4.0745744378918134e-10f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(-2.2750374151783939e-8f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(5.7875516457819682e-7f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(-9.0015004556574117e-6f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(9.7410461048522186e-5f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(-7.9360535417300805e-4f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(5.1196939569625932e-3f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(-2.6760455816072556e-2f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(1.1278225992880547e-1f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(-3.7611487211070921e-1f), curLength);
        Mul(yLocal, yLocal, tLocal, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(1.1283787714929709f), curLength);
        Mul(yLocal, yLocal, xClip, curLength);
        PipeBarrier<PIPE_V>();
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, DOUBLE_BUFFER_NUM> inQueueX;
    TQue<TPosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueY;
    TBuf<TPosition::VECCALC> xClipBuffer;
    TBuf<TPosition::VECCALC> tBuffer;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    uint64_t totalLength = 0;
    uint64_t blockLength = 0;
    uint64_t thisBlockLength = 0;
    uint32_t tileLength = 0;
    uint32_t blockDim = 0;
};

template <typename DT_X>
 __global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    if (tiling_data.pathType == PATH_TINY64_FAST && tiling_data.totalLength > 0 &&
        tiling_data.totalLength <= TINY_DIRECT_N) {
        KernelErfTinyDirect<DT_X> op;
        op.Init(x, y, tiling_data.totalLength);
        op.Process();
    } else if (tiling_data.pathType == PATH_MID_NATIVE_REVERT) {
        KernelErfSmallTbuf<DT_X> op;
        op.Init(x, y, tiling_data.totalLength, tiling_data.tileLength, tiling_data.tmpSize);
        op.Process();
    } else if (tiling_data.totalLength > 0 && tiling_data.totalLength <= SMALL_FAST_LENGTH) {
        KernelErfSmallPolyTbuf<DT_X> op;
        op.Init(x, y, tiling_data.totalLength, tiling_data.tileLength);
        op.Process();
    } else if (tiling_data.totalLength >= DOUBLE_BUFFER_MIN_LENGTH) {
        KernelErfDoubleBuffer<DT_X> op;
        op.Init(x, y, tiling_data.totalLength, tiling_data.blockLength, tiling_data.tileLength, tiling_data.blockDim);
        op.Process();
    } else {
        KernelErfSingleBuffer<DT_X> op;
        op.Init(x, y, tiling_data.totalLength, tiling_data.blockLength, tiling_data.tileLength, tiling_data.blockDim);
        op.Process();
    }
}
