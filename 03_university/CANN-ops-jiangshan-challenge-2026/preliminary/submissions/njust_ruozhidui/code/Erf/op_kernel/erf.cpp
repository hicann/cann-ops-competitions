#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t ALIGN_NUM = 32;

template <class DT_X>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, uint32_t blockDim,
        uint32_t tileLength)
    {
        length_ = length;
        blockDim_ = blockDim == 0 ? 1 : blockDim;
        tileLength_ = tileLength < ALIGN_NUM ? ALIGN_NUM : tileLength;

        uint32_t blockIdx = GetBlockIdx();
        uint32_t baseLength = length_ / blockDim_;
        uint32_t remain = length_ % blockDim_;
        if (blockIdx < remain) {
            coreLength_ = baseLength + 1;
            coreOffset_ = blockIdx * coreLength_;
        } else {
            coreLength_ = baseLength;
            coreOffset_ = remain * (baseLength + 1) +
                (blockIdx - remain) * baseLength;
        }

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x + coreOffset_, coreLength_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y + coreOffset_, coreLength_);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(x2Buf_, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(tmpBuf_, tileLength_ * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (coreLength_ == 0) {
            return;
        }

        CopyIn(0, CalcTileLength(0));
        for (uint32_t offset = 0; offset < coreLength_; offset += tileLength_) {
            uint32_t curLength = CalcTileLength(offset);
            uint32_t nextOffset = offset + tileLength_;
            if (nextOffset < coreLength_) {
                CopyIn(nextOffset, CalcTileLength(nextOffset));
            }
            Compute(curLength);
            CopyOut(offset, curLength);
        }
    }

private:
    __aicore__ inline uint32_t CalcTileLength(uint32_t offset) const
    {
        uint32_t remain = coreLength_ - offset;
        return remain > tileLength_ ? tileLength_ : remain;
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t curLength)
    {
        LocalTensor<DT_X> xLocal = inQueueX_.AllocTensor<DT_X>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(DT_X)), 0, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t curLength)
    {
        LocalTensor<DT_X> xLocal = inQueueX_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY_.AllocTensor<DT_X>();
        LocalTensor<DT_X> x2Local = x2Buf_.Get<DT_X>();
        LocalTensor<DT_X> tmpLocal = tmpBuf_.Get<DT_X>();

        Mins(xLocal, xLocal, static_cast<DT_X>(3.92f), curLength);
        Maxs(xLocal, xLocal, static_cast<DT_X>(-3.92f), curLength);

        Mul(x2Local, xLocal, xLocal, curLength);

        Muls(yLocal, x2Local, static_cast<DT_X>(0.00000195588513f), curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(0.000294543321f), curLength);
        Mul(yLocal, yLocal, x2Local, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(0.00423350309f), curLength);
        Mul(yLocal, yLocal, x2Local, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(0.0542005573f), curLength);
        Mul(yLocal, yLocal, x2Local, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(0.201413159f), curLength);
        Mul(yLocal, yLocal, x2Local, curLength);
        Adds(yLocal, yLocal, static_cast<DT_X>(1.12837907f), curLength);
        Mul(yLocal, yLocal, xLocal, curLength);

        Muls(tmpLocal, x2Local, static_cast<DT_X>(0.0000376174671f), curLength);
        Adds(tmpLocal, tmpLocal, static_cast<DT_X>(0.00127414729f), curLength);
        Mul(tmpLocal, tmpLocal, x2Local, curLength);
        Adds(tmpLocal, tmpLocal, static_cast<DT_X>(0.0159109763f), curLength);
        Mul(tmpLocal, tmpLocal, x2Local, curLength);
        Adds(tmpLocal, tmpLocal, static_cast<DT_X>(0.118654144f), curLength);
        Mul(tmpLocal, tmpLocal, x2Local, curLength);
        Adds(tmpLocal, tmpLocal, static_cast<DT_X>(0.511828742f), curLength);
        Mul(tmpLocal, tmpLocal, x2Local, curLength);
        Adds(tmpLocal, tmpLocal, static_cast<DT_X>(1.0f), curLength);

        Div(yLocal, yLocal, tmpLocal, curLength);

        outQueueY_.EnQue(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curLength)
    {
        LocalTensor<DT_X> yLocal = outQueueY_.DeQue<DT_X>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(DT_X)), 0, 0, 0};
        DataCopyPad(yGm_[offset], yLocal, copyParams);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    TBuf<QuePosition::VECCALC> x2Buf_;
    TBuf<QuePosition::VECCALC> tmpBuf_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t length_ = 0;
    uint32_t coreOffset_ = 0;
    uint32_t coreLength_ = 0;
    uint32_t blockDim_ = 1;
    uint32_t tileLength_ = 1;
};

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tilingData, tiling);
    KernelErf<DT_X> op;
    op.Init(x, y, tilingData.length, tilingData.blockDim, tilingData.tileLength);
    op.Process();
}
