// Kernel侧核函数实现 - best2 + CURRENT_BEST large tensor tile distribution
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t ALIGN_NUM = 8;          // float32: 8 elements = 32B
constexpr uint32_t DIRECT_TILE_MAX = 8192;

__aicore__ inline uint32_t AlignUp8(uint32_t x) {
    return (x + ALIGN_NUM - 1) & ~(ALIGN_NUM - 1);
}

template <class DT_X>
__aicore__ inline void ComputeErfInplaceSmallP5(
    AscendC::LocalTensor<DT_X> yLocal,
    AscendC::LocalTensor<DT_X> zLocal,
    AscendC::LocalTensor<DT_X> pLocal,
    uint32_t curLen) {
    AscendC::Mins(yLocal, yLocal, static_cast<DT_X>(2.45f), curLen);
    AscendC::Maxs(yLocal, yLocal, static_cast<DT_X>(-2.45f), curLen);

    AscendC::Mul(zLocal, yLocal, yLocal, curLen);

    constexpr float C0 = 1.12655014f;
    constexpr float C1 = -0.366745831f;
    constexpr float C2 = 0.0987866749f;
    constexpr float C3 = -0.0173586698f;
    constexpr float C4 = 0.00173296634f;
    constexpr float C5 = -0.0000733966480f;

    AscendC::Muls(pLocal, zLocal, static_cast<DT_X>(C5), curLen);
    AscendC::Adds(pLocal, pLocal, static_cast<DT_X>(C4), curLen);
    AscendC::Mul(pLocal, pLocal, zLocal, curLen);
    AscendC::Adds(pLocal, pLocal, static_cast<DT_X>(C3), curLen);
    AscendC::Mul(pLocal, pLocal, zLocal, curLen);
    AscendC::Adds(pLocal, pLocal, static_cast<DT_X>(C2), curLen);
    AscendC::Mul(pLocal, pLocal, zLocal, curLen);
    AscendC::Adds(pLocal, pLocal, static_cast<DT_X>(C1), curLen);
    AscendC::Mul(pLocal, pLocal, zLocal, curLen);
    AscendC::Adds(pLocal, pLocal, static_cast<DT_X>(C0), curLen);

    AscendC::Mul(yLocal, pLocal, yLocal, curLen);
}

template <class DT_X>
__aicore__ inline void ComputeErfInplaceSmallP4(
    AscendC::LocalTensor<DT_X> yLocal,
    AscendC::LocalTensor<DT_X> zLocal,
    AscendC::LocalTensor<DT_X> pLocal,
    uint32_t curLen) {
    AscendC::Mins(yLocal, yLocal, static_cast<DT_X>(2.25f), curLen);
    AscendC::Maxs(yLocal, yLocal, static_cast<DT_X>(-2.25f), curLen);

    AscendC::Mul(zLocal, yLocal, yLocal, curLen);

    constexpr float C0 = 1.1241973f;
    constexpr float C1 = -0.357146373f;
    constexpr float C2 = 0.0879198422f;
    constexpr float C3 = -0.0123576241f;
    constexpr float C4 = 0.000727836903f;

    AscendC::Muls(pLocal, zLocal, static_cast<DT_X>(C4), curLen);
    AscendC::Adds(pLocal, pLocal, static_cast<DT_X>(C3), curLen);
    AscendC::Mul(pLocal, pLocal, zLocal, curLen);
    AscendC::Adds(pLocal, pLocal, static_cast<DT_X>(C2), curLen);
    AscendC::Mul(pLocal, pLocal, zLocal, curLen);
    AscendC::Adds(pLocal, pLocal, static_cast<DT_X>(C1), curLen);
    AscendC::Mul(pLocal, pLocal, zLocal, curLen);
    AscendC::Adds(pLocal, pLocal, static_cast<DT_X>(C0), curLen);

    AscendC::Mul(yLocal, pLocal, yLocal, curLen);
}

template <class DT_X>
__aicore__ inline void ComputeErfFromX_LargeP4ReuseX(
    AscendC::LocalTensor<DT_X> xLocal,
    AscendC::LocalTensor<DT_X> yLocal,
    AscendC::LocalTensor<DT_X> pLocal,
    uint32_t curLen) {
    AscendC::Mins(yLocal, xLocal, static_cast<DT_X>(2.25f), curLen);
    AscendC::Maxs(yLocal, yLocal, static_cast<DT_X>(-2.25f), curLen);

    AscendC::Mul(xLocal, yLocal, yLocal, curLen);

    constexpr float C0 = 1.1241973f;
    constexpr float C1 = -0.357146373f;
    constexpr float C2 = 0.0879198422f;
    constexpr float C3 = -0.0123576241f;
    constexpr float C4 = 0.000727836903f;

    AscendC::Muls(pLocal, xLocal, static_cast<DT_X>(C4), curLen);
    AscendC::Adds(pLocal, pLocal, static_cast<DT_X>(C3), curLen);
    AscendC::Mul(pLocal, pLocal, xLocal, curLen);
    AscendC::Adds(pLocal, pLocal, static_cast<DT_X>(C2), curLen);
    AscendC::Mul(pLocal, pLocal, xLocal, curLen);
    AscendC::Adds(pLocal, pLocal, static_cast<DT_X>(C1), curLen);
    AscendC::Mul(pLocal, pLocal, xLocal, curLen);
    AscendC::Adds(pLocal, pLocal, static_cast<DT_X>(C0), curLen);

    AscendC::Mul(yLocal, pLocal, yLocal, curLen);
}


template <class DT_X, bool ALIGNED>
struct DirectCopy;

template <class DT_X>
struct DirectCopy<DT_X, true> {
    __aicore__ static inline void CopyIn(
        AscendC::LocalTensor<DT_X> dst,
        AscendC::GlobalTensor<DT_X>& src,
        uint32_t len) {
        AscendC::DataCopy(dst, src[0], len);
    }

    __aicore__ static inline void CopyOut(
        AscendC::GlobalTensor<DT_X>& dst,
        AscendC::LocalTensor<DT_X> src,
        uint32_t len) {
        AscendC::DataCopy(dst[0], src, len);
    }
};

template <class DT_X>
struct DirectCopy<DT_X, false> {
    __aicore__ static inline void CopyIn(
        AscendC::LocalTensor<DT_X> dst,
        AscendC::GlobalTensor<DT_X>& src,
        uint32_t len) {
        AscendC::DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(len * sizeof(DT_X)),
            0,
            0,
            0
        };
        AscendC::DataCopyPadExtParams<DT_X> padParams{
            true,
            0,
            0,
            static_cast<DT_X>(0)
        };
        AscendC::DataCopyPad(dst, src[0], copyParams, padParams);
    }

    __aicore__ static inline void CopyOut(
        AscendC::GlobalTensor<DT_X>& dst,
        AscendC::LocalTensor<DT_X> src,
        uint32_t len) {
        AscendC::DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(len * sizeof(DT_X)),
            0,
            0,
            0
        };
        AscendC::DataCopyPad(dst[0], src, copyParams);
    }
};

template <class DT_X, bool ALIGNED>
class KernelErfSingle {
public:
    __aicore__ inline KernelErfSingle() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t length,
        uint32_t blockLength,
        uint32_t tileLength) {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t start = blockIdx * blockLength;

        if (start >= length) {
            coreLength_ = 0;
            return;
        }

        const uint32_t remain = length - start;
        const uint32_t coreLength = remain < blockLength ? remain : blockLength;

        InitRange(x, y, start, coreLength, tileLength);
    }

    __aicore__ inline void InitRange(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t coreOffset,
        uint32_t coreLength,
        uint32_t tileLength) {
        if (coreLength == 0) {
            coreLength_ = 0;
            return;
        }

        coreOffset_ = coreOffset;
        coreLength_ = coreLength;
        tileLength_ = tileLength;

        xGm_.SetGlobalBuffer((__gm__ DT_X*)x + coreOffset_, coreLength_);
        yGm_.SetGlobalBuffer((__gm__ DT_X*)y + coreOffset_, coreLength_);
    }

    __aicore__ inline void Process() {
        if (coreLength_ == 0) {
            return;
        }

        AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
        AscendC::LocalTensor<DT_X> buf = ubAllocator.Alloc<DT_X>(tileLength_ * 3);
        AscendC::LocalTensor<DT_X> yLocal = buf;
        AscendC::LocalTensor<DT_X> zLocal = buf[tileLength_];
        AscendC::LocalTensor<DT_X> pLocal = buf[tileLength_ * 2];

        DirectCopy<DT_X, ALIGNED>::CopyIn(zLocal, xGm_, coreLength_);

        const uint32_t computeLen = AlignUp8(coreLength_);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

        ComputeErfFromX_LargeP4ReuseX<DT_X>(zLocal, yLocal, pLocal, computeLen);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);

        DirectCopy<DT_X, ALIGNED>::CopyOut(yGm_, yLocal, coreLength_);
    }

private:
    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;

    uint32_t coreOffset_ = 0;
    uint32_t coreLength_ = 0;
    uint32_t tileLength_ = 1;
};

template <class DT_X>
class KernelErfNormal {
public:
    __aicore__ inline KernelErfNormal() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t length,
        uint32_t blockLength,
        uint32_t tileLength) {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t start = blockIdx * blockLength;
        const uint32_t remain = start < length ? length - start : 0;
        const uint32_t coreLength = remain < blockLength ? remain : blockLength;

        InitRange(x, y, length, start, coreLength, tileLength);
    }

    __aicore__ inline void InitRange(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t length,
        uint32_t coreOffset,
        uint32_t coreLength,
        uint32_t tileLength) {
        if (coreOffset >= length || coreLength == 0) {
            coreLength_ = 0;
            return;
        }

        coreOffset_ = coreOffset;
        coreLength_ = coreLength;
        tileLength_ = tileLength;

        xGm_.SetGlobalBuffer((__gm__ DT_X*)x + coreOffset_, coreLength_);
        yGm_.SetGlobalBuffer((__gm__ DT_X*)y + coreOffset_, coreLength_);

        const uint32_t bufferBytes = tileLength_ * sizeof(DT_X);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, bufferBytes);
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, bufferBytes);

        pipe_.InitBuffer(pBuf_, bufferBytes);
    }

    __aicore__ inline void Process() {
        if (coreLength_ == 0) {
            return;
        }

        if (coreLength_ <= tileLength_) {
            CopyIn(0, coreLength_);
            Compute(AlignUp8(coreLength_));
            CopyOut(0, coreLength_);
            return;
        }

        uint32_t offset = 0;
        uint32_t curLen = tileLength_;

        CopyIn(0, curLen);

        offset += tileLength_;
        uint32_t remain = coreLength_ - offset;

        while (remain > 0) {
            const uint32_t nextLen = remain < tileLength_ ? remain : tileLength_;

            CopyIn(offset, nextLen);

            const uint32_t outOffset = offset - tileLength_;
            Compute(AlignUp8(curLen));
            CopyOut(outOffset, curLen);

            curLen = nextLen;
            offset += nextLen;
            remain = coreLength_ - offset;
        }

        Compute(AlignUp8(curLen));
        CopyOut(offset - curLen, curLen);
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t curLen) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX_.AllocTensor<DT_X>();

        if ((curLen % ALIGN_NUM) == 0) {
            AscendC::DataCopy(xLocal, xGm_[offset], curLen);
        } else {
            AscendC::DataCopyExtParams copyParams{
                1,
                static_cast<uint32_t>(curLen * sizeof(DT_X)),
                0,
                0,
                0
            };
            AscendC::DataCopyPadExtParams<DT_X> padParams{
                true,
                0,
                0,
                static_cast<DT_X>(0)
            };
            AscendC::DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        }

        inQueueX_.EnQue<DT_X>(xLocal);
    }

    __aicore__ inline void Compute(uint32_t curLen) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY_.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> pLocal = pBuf_.Get<DT_X>();

        ComputeErfFromX_LargeP4ReuseX<DT_X>(xLocal, yLocal, pLocal, curLen);

        outQueueY_.EnQue<DT_X>(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curLen) {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY_.DeQue<DT_X>();

        if ((curLen % ALIGN_NUM) == 0) {
            AscendC::DataCopy(yGm_[offset], yLocal, curLen);
        } else {
            AscendC::DataCopyExtParams copyParams{
                1,
                static_cast<uint32_t>(curLen * sizeof(DT_X)),
                0,
                0,
                0
            };
            AscendC::DataCopyPad(yGm_[offset], yLocal, copyParams);
        }

        outQueueY_.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY_;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> pBuf_;

    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;

    uint32_t coreOffset_ = 0;
    uint32_t coreLength_ = 0;
    uint32_t tileLength_ = 1;
};

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);

    const __gm__ ErfTilingData* tilingData =
        reinterpret_cast<const __gm__ ErfTilingData*>(tiling);

    const uint32_t length = tilingData->length;
    const uint32_t blockLength = tilingData->blockLength;
    const uint32_t tileLength = tilingData->tileLength;

    // Direct path 拆成 aligned / non-aligned 两个模板实例：
    // 对齐 core 里不再保留 DataCopyPad 分支，非对齐 core 只走 Pad 路径。
    if (blockLength <= DIRECT_TILE_MAX) {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t start = blockIdx * blockLength;
        const uint32_t remain = start < length ? length - start : 0;
        const uint32_t coreLength = remain < blockLength ? remain : blockLength;

        if ((coreLength & (ALIGN_NUM - 1)) == 0) {
            KernelErfSingle<DT_X, true> op;
            op.InitRange(x, y, start, coreLength, tileLength);
            op.Process();
        } else {
            KernelErfSingle<DT_X, false> op;
            op.InitRange(x, y, start, coreLength, tileLength);
            op.Process();
        }
        return;
    }

    KernelErfNormal<DT_X> op;

    if (length >= 524288 && blockLength > tileLength) {
        const uint32_t tileNum = tilingData->tileNum;
        const uint32_t blockDim = tilingData->blockDim;
        const uint32_t blockIdx = AscendC::GetBlockIdx();

        const uint32_t baseTiles = tileNum / blockDim;
        const uint32_t extraTiles = tileNum - baseTiles * blockDim;

        const uint32_t maxContigTiles = baseTiles + (extraTiles > 0 ? 1 : 0);
        const uint32_t maxContigLen = maxContigTiles * tileLength;

        if (maxContigLen <= blockLength + 2048) {
            const uint32_t coreTiles = baseTiles + (blockIdx < extraTiles ? 1 : 0);
            const uint32_t startTile =
                blockIdx * baseTiles + (blockIdx < extraTiles ? blockIdx : extraTiles);

            const uint32_t start = startTile * tileLength;
            const uint32_t maxLen = coreTiles * tileLength;
            const uint32_t remain = start < length ? length - start : 0;
            const uint32_t coreLength = remain < maxLen ? remain : maxLen;

            op.InitRange(x, y, length, start, coreLength, tileLength);
        } else {
            op.Init(x, y, length, blockLength, tileLength);
        }
    } else {
        op.Init(x, y, length, blockLength, tileLength);
    }

    op.Process();
}