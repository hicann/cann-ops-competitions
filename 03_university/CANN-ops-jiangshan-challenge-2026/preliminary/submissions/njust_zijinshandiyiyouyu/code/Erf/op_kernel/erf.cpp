#include "kernel_operator.h"
#include "erf_tiling.h"

constexpr float POLY_COEFF[5] = {
    1.1241970062f,
    -0.3571457863f,
    0.0879194960f,
    -0.0123575451f,
    0.0007278307f
};
constexpr float ERF_CLIP_BOUNDARY = 2.25f;
constexpr int32_t BUFFER_NUM = 2;

class KernelErf {
public:
    __aicore__ inline KernelErf() : coreDataSize_(0), tileLength_(0) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ErfTilingData* tiling)
    {
        const uint32_t coreId = AscendC::GetBlockIdx();

        const uint32_t startBlock =
            coreId * tiling->blocksPerCore + (coreId < tiling->tailBlocks ? coreId : tiling->tailBlocks);
        const uint32_t myBlocks = tiling->blocksPerCore + (coreId < tiling->tailBlocks ? 1U : 0U);

        const uint32_t myOffset = startBlock * ERF_ALIGN_ELEMENTS;
        uint32_t myLength = myBlocks * ERF_ALIGN_ELEMENTS;
        if (myOffset >= tiling->totalLength) {
            coreDataSize_ = 0;
            return;
        }
        if (myOffset + myLength > tiling->totalLength) {
            myLength = tiling->totalLength - myOffset;
        }

        coreDataSize_ = myLength;
        tileLength_ = tiling->tileLength;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x) + myOffset, coreDataSize_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y) + myOffset, coreDataSize_);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(float));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileLength_ * sizeof(float));
        pipe_.InitBuffer(tmpBufT_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(tmpBufT2_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(tmpBufQ1_, tileLength_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (coreDataSize_ == 0) {
            return;
        }

        const uint32_t fullTileNum = coreDataSize_ / tileLength_;
        const uint32_t tailCount = coreDataSize_ % tileLength_;

        if (fullTileNum == 0U) {
            CopyInTail(0, coreDataSize_);
            Compute(coreDataSize_);
            CopyOutTail(0, coreDataSize_);
            return;
        }

        CopyInAligned(0, tileLength_);
        for (uint32_t tileIdx = 1; tileIdx < fullTileNum; ++tileIdx) {
            CopyInAligned(tileIdx * tileLength_, tileLength_);
            Compute(tileLength_);
            CopyOutAligned((tileIdx - 1U) * tileLength_, tileLength_);
        }

        if (tailCount > 0U) {
            CopyInTail(fullTileNum * tileLength_, tailCount);
            Compute(tileLength_);
            CopyOutAligned((fullTileNum - 1U) * tileLength_, tileLength_);
            Compute(tailCount);
            CopyOutTail(fullTileNum * tileLength_, tailCount);
            return;
        }

        Compute(tileLength_);
        CopyOutAligned((fullTileNum - 1U) * tileLength_, tileLength_);
    }

private:
    __aicore__ inline bool IsThirtyTwoByteAligned(uint32_t count) const
    {
        return (count & 7U) == 0;
    }

    __aicore__ inline void CopyInAligned(uint32_t offset, uint32_t count)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX_.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm_[offset], count);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInTail(uint32_t offset, uint32_t count)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX_.AllocTensor<float>();
        if (IsThirtyTwoByteAligned(count)) {
            AscendC::DataCopy(xLocal, xGm_[offset], count);
        } else {
            AscendC::DataCopyPad(
                xLocal,
                xGm_[offset],
                {1, static_cast<uint16_t>(count * sizeof(float)), 0, 0},
                {false, 0, 0, 0});
        }
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void ComputeCore(AscendC::LocalTensor<float>& xLocal,
                                       AscendC::LocalTensor<float>& yLocal,
                                       AscendC::LocalTensor<float>& tmpT,
                                       AscendC::LocalTensor<float>& tmpT2,
                                       AscendC::LocalTensor<float>& tmpQ1,
                                       uint32_t count)
    {
        const int32_t countInt = static_cast<int32_t>(count);
        AscendC::LocalTensor<float>& xSquare = tmpT;
        AscendC::LocalTensor<float>& xQuad = tmpT2;
        AscendC::LocalTensor<float>& cubicPair = tmpQ1;

        AscendC::Mins(xSquare, xLocal, ERF_CLIP_BOUNDARY, countInt);
        AscendC::Maxs(xLocal, xSquare, -ERF_CLIP_BOUNDARY, countInt);
        AscendC::Mul(xSquare, xLocal, xLocal, countInt);

        AscendC::Muls(yLocal, xSquare, POLY_COEFF[1], countInt);
        AscendC::Adds(yLocal, yLocal, POLY_COEFF[0], countInt);

        AscendC::Muls(cubicPair, xSquare, POLY_COEFF[3], countInt);
        AscendC::Adds(cubicPair, cubicPair, POLY_COEFF[2], countInt);

        AscendC::Mul(xQuad, xSquare, xSquare, countInt);
        
        // ！！！关键修改：删掉原来的 Muls 和 Add，用 Axpy 替换，直接砍掉一个机器周期！！！
        AscendC::Axpy(cubicPair, xQuad, POLY_COEFF[4], countInt);
        
        AscendC::Mul(cubicPair, cubicPair, xQuad, countInt);
        AscendC::Add(yLocal, yLocal, cubicPair, countInt);
        AscendC::Mul(yLocal, yLocal, xLocal, countInt);
    }
    __aicore__ inline void Compute(uint32_t count)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY_.AllocTensor<float>();
        AscendC::LocalTensor<float> tmpT = tmpBufT_.Get<float>();
        AscendC::LocalTensor<float> tmpT2 = tmpBufT2_.Get<float>();
        AscendC::LocalTensor<float> tmpQ1 = tmpBufQ1_.Get<float>();
        ComputeCore(xLocal, yLocal, tmpT, tmpT2, tmpQ1, count);
        outQueueY_.EnQue(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutAligned(uint32_t offset, uint32_t count)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY_.DeQue<float>();
        AscendC::DataCopy(yGm_[offset], yLocal, count);
        outQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutTail(uint32_t offset, uint32_t count)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY_.DeQue<float>();
        if (IsThirtyTwoByteAligned(count)) {
            AscendC::DataCopy(yGm_[offset], yLocal, count);
        } else {
            AscendC::DataCopyPad(
                yGm_[offset],
                yLocal,
                {1, static_cast<uint16_t>(count * sizeof(float)), 0, 0});
        }
        outQueueY_.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufT_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufT2_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufQ1_;
    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> yGm_;

    uint32_t coreDataSize_;
    uint32_t tileLength_;
};

extern "C" __global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tilingData, tiling);
    KernelErf op;
    op.Init(x, y, &tilingData);
    op.Process();
}
