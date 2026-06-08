#include "kernel_operator.h"

#include "erf_tiling.h"
using namespace AscendC;

namespace {
constexpr uint32_t TILE_SIZE      = 9760;
constexpr uint32_t BUFFER_NUM     = 2;
constexpr uint32_t BLOCK_ALIGN    = 8;
constexpr uint32_t TINY_THRESHOLD = 128;

constexpr float ERF_CLAMP = 2.44816148f;

constexpr float A0 = 1.56722970e-01f;
constexpr float A1 = 2.53287476e+00f;
constexpr float A2 = 7.92297613e+00f;
constexpr float B1 = 4.57502853e+00f;
constexpr float B2 = 7.02081808e+00f;

}  // namespace

class KernelErf {
public:
    __aicore__ inline KernelErf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t paddedTotalLength,
                                uint32_t validTotalLength) {
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockNum = GetBlockNum();

        uint32_t myStartElem;
        uint32_t myPaddedLength;
        uint32_t myValidLength;

        if (blockNum == 1) {
            myStartElem    = 0;
            myValidLength  = validTotalLength;
            myPaddedLength =
                (paddedTotalLength + BLOCK_ALIGN - 1) / BLOCK_ALIGN * BLOCK_ALIGN;
        } else {
            const uint32_t totalBlocks =
                (paddedTotalLength + BLOCK_ALIGN - 1) / BLOCK_ALIGN;
            const uint32_t blocksPerCore = totalBlocks / blockNum;
            const uint32_t extraBlocks   = totalBlocks % blockNum;

            uint32_t myBlocks;
            uint32_t myStartBlock;
            if (blockIdx < extraBlocks) {
                myBlocks     = blocksPerCore + 1;
                myStartBlock = blockIdx * myBlocks;
            } else {
                myBlocks     = blocksPerCore;
                myStartBlock = extraBlocks * (blocksPerCore + 1) +
                               (blockIdx - extraBlocks) * blocksPerCore;
            }

            myStartElem    = myStartBlock * BLOCK_ALIGN;
            myPaddedLength = myBlocks * BLOCK_ALIGN;
            myValidLength  = myPaddedLength;

            if (myStartElem >= validTotalLength) {
                myValidLength  = 0;
            } else if (myStartElem + myPaddedLength > validTotalLength) {
                myValidLength = validTotalLength - myStartElem;
            }
        }

        paddedLength_ = myPaddedLength;
        validLength_  = myValidLength;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x) + myStartElem,
                             myPaddedLength);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y) + myStartElem,
                             myPaddedLength);

        if (paddedLength_ == 0) {
            return;
        }

        isTiny_ = (blockNum == 1 && paddedLength_ <= TINY_THRESHOLD);

        if (isTiny_) {
            const uint32_t alloc =
                paddedLength_ * static_cast<uint32_t>(sizeof(float));
            pipe_.InitBuffer(xBufTiny_, alloc);
            pipe_.InitBuffer(yBufTiny_, alloc);
            pipe_.InitBuffer(tBuf_,     alloc);
        } else {
            const uint32_t alloc =
                (paddedLength_ < TILE_SIZE ? paddedLength_ : TILE_SIZE)
                * static_cast<uint32_t>(sizeof(float));
            pipe_.InitBuffer(inQueueX_,  BUFFER_NUM, alloc);
            pipe_.InitBuffer(outQueueY_, BUFFER_NUM, alloc);
            pipe_.InitBuffer(tBuf_,                 alloc);
        }
    }

    __aicore__ inline void Process() {
        if (paddedLength_ == 0) {
            return;
        }
        if (isTiny_) {
            ProcessTiny();
        } else {
            ProcessMain();
        }
    }

private:
    __aicore__ inline void ComputeDirect(LocalTensor<float> &x,
                                         LocalTensor<float> &y,
                                         LocalTensor<float> &t,
                                         uint32_t tileSize) {
        Mins(x, x, static_cast<float>(ERF_CLAMP),  tileSize);
        Maxs(x, x, static_cast<float>(-ERF_CLAMP), tileSize);
        Mul (t, x, x,                             tileSize);

        Muls(y, t, static_cast<float>(A0), tileSize);
        Adds(y, y, static_cast<float>(A1), tileSize);
        Mul (y, y, t,                     tileSize);
        Adds(y, y, static_cast<float>(A2), tileSize);
        Mul (y, y, x,                     tileSize);

        Adds(x, t, static_cast<float>(B1), tileSize);
        Mul (x, x, t,                     tileSize);
        Adds(x, x, static_cast<float>(B2), tileSize);

        Div(y, y, x, tileSize);
    }

    __aicore__ inline void ProcessTiny() {
        const uint32_t n     = paddedLength_;
        const uint32_t valid = validLength_;

        LocalTensor<float> x = xBufTiny_.Get<float>();
        LocalTensor<float> y = yBufTiny_.Get<float>();
        LocalTensor<float> t = tBuf_.Get<float>();

        DataCopy(x, xGm_[0], n);

        pipe_barrier(PIPE_ALL);
        ComputeDirect(x, y, t, n);
        pipe_barrier(PIPE_ALL);

        if (valid == n) {
            DataCopy(yGm_[0], y, n);
        } else if (valid > 0) {
            DataCopyExtParams copyParams{};
            copyParams.blockCount = 1;
            copyParams.blockLen   =
                static_cast<uint32_t>(valid * sizeof(float));
            copyParams.srcStride  = 0;
            copyParams.dstStride  = 0;
            DataCopyPad(yGm_[0], y, copyParams);
        }
    }

    __aicore__ inline void ProcessMain() {
        const uint32_t numTiles =
            (paddedLength_ + TILE_SIZE - 1) / TILE_SIZE;

        for (uint32_t i = 0; i < numTiles; ++i) {
            const uint32_t offset = i * TILE_SIZE;
            const uint32_t remain = paddedLength_ - offset;
            const uint32_t tileSize =
                (remain > TILE_SIZE) ? TILE_SIZE : remain;

            uint32_t validInTile;
            if (offset >= validLength_) {
                validInTile = 0;
            } else if (offset + tileSize > validLength_) {
                validInTile = validLength_ - offset;
            } else {
                validInTile = tileSize;
            }

            CopyIn(offset, tileSize, validInTile);
            Compute(tileSize);
            CopyOut(offset, tileSize, validInTile);
        }
    }

    __aicore__ inline void CopyIn(uint32_t offset,
                                  uint32_t tileSize,
                                  uint32_t validInTile) {
        LocalTensor<float> xLocal = inQueueX_.AllocTensor<float>();

        DataCopy(xLocal, xGm_[offset], tileSize);

        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t tileSize) {
        LocalTensor<float> x = inQueueX_.DeQue<float>();
        LocalTensor<float> y = outQueueY_.AllocTensor<float>();
        LocalTensor<float> t = tBuf_.Get<float>();

        ComputeDirect(x, y, t, tileSize);

        inQueueX_.FreeTensor(x);
        outQueueY_.EnQue(y);
    }

    __aicore__ inline void CopyOut(uint32_t offset,
                                   uint32_t tileSize,
                                   uint32_t validInTile) {
        LocalTensor<float> yLocal = outQueueY_.DeQue<float>();

        if (validInTile == tileSize) {
            DataCopy(yGm_[offset], yLocal, tileSize);
        } else if (validInTile > 0) {
            DataCopyExtParams copyParams{};
            copyParams.blockCount = 1;
            copyParams.blockLen   =
                static_cast<uint32_t>(validInTile * sizeof(float));
            copyParams.srcStride  = 0;
            copyParams.dstStride  = 0;
            DataCopyPad(yGm_[offset], yLocal, copyParams);
        }

        outQueueY_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;

    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;

    TBuf<QuePosition::VECCALC> xBufTiny_;
    TBuf<QuePosition::VECCALC> yBufTiny_;
    TBuf<QuePosition::VECCALC> tBuf_;

    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;

    uint32_t paddedLength_;
    uint32_t validLength_;
    bool     isTiny_;
};

extern "C" __global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y,
                                          GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);

    KernelErf op;
    op.Init(x, y, tiling_data.length, tiling_data.validLength);
    op.Process();
}
