// Kernel-side implementation for erfinv.
#include "kernel_operator.h"

#include "erfinv_tiling.h"
#include "tiling_key_erfinv.h"

template <typename T>
class KernelErfinv {
public:
    __aicore__ inline KernelErfinv() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ErfinvTilingData *tiling)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t start = blockIdx * tiling->blockLength;
        uint32_t remain = (start < tiling->length) ? (tiling->length - start) : 0;
        blockLength_ = (remain < tiling->blockLength) ? remain : tiling->blockLength;
        ubLength_ = tiling->ubLength;

        xGm_.SetGlobalBuffer((__gm__ T *)x + start, blockLength_);
        yGm_.SetGlobalBuffer((__gm__ T *)y + start, blockLength_);

        uint32_t tileBytes = ubLength_ * sizeof(T);
        pipe_.InitBuffer(inQueueX_, ERFINV_BUFFER_NUM, tileBytes);
        pipe_.InitBuffer(outQueueY_, ERFINV_BUFFER_NUM, tileBytes);
        pipe_.InitBuffer(workX_, tileBytes);
        pipe_.InitBuffer(workX2_, tileBytes);
        pipe_.InitBuffer(workDen_, tileBytes);
        pipe_.InitBuffer(workTmp_, tileBytes);
    }

    __aicore__ inline void Process()
    {
        uint32_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
        for (uint32_t i = 0; i < loopCount; ++i) {
            uint32_t count = (i == loopCount - 1) ? (blockLength_ - i * ubLength_) : ubLength_;
            uint32_t offset = i * ubLength_;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX_.template AllocTensor<T>();
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = count * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(xLocal, xGm_[offset], copyParams, {false, 0, 0, 0});
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX_.template DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY_.template AllocTensor<T>();
        AscendC::LocalTensor<T> xClamp = workX_.template Get<T>();
        AscendC::LocalTensor<T> x2 = workX2_.template Get<T>();
        AscendC::LocalTensor<T> den = workDen_.template Get<T>();
        AscendC::LocalTensor<T> tmp = workTmp_.template Get<T>();

        AscendC::Duplicate<T>(den, static_cast<T>(3.92f), count);
        AscendC::Min<T>(xClamp, xLocal, den, count);
        AscendC::Duplicate<T>(den, static_cast<T>(-3.92f), count);
        AscendC::Max<T>(tmp, xClamp, den, count);

        AscendC::Mul<T>(x2, tmp, tmp, count);

        AscendC::Muls<T>(yLocal, x2, static_cast<T>(0.053443748819f), count);
        AscendC::Adds<T>(yLocal, yLocal, static_cast<T>(0.75517016694e1f), count);
        AscendC::Mul<T>(yLocal, yLocal, x2, count);
        AscendC::Adds<T>(yLocal, yLocal, static_cast<T>(0.10162808918e3f), count);
        AscendC::Mul<T>(yLocal, yLocal, x2, count);
        AscendC::Adds<T>(yLocal, yLocal, static_cast<T>(0.13938061484e4f), count);
        AscendC::Mul<T>(yLocal, yLocal, x2, count);
        AscendC::Adds<T>(yLocal, yLocal, static_cast<T>(0.50637915060e4f), count);
        AscendC::Mul<T>(yLocal, yLocal, x2, count);
        AscendC::Adds<T>(yLocal, yLocal, static_cast<T>(0.2963838468e5f), count);
        AscendC::Mul<T>(yLocal, yLocal, tmp, count);

        AscendC::Adds<T>(den, x2, static_cast<T>(0.31212858877e2f), count);
        AscendC::Mul<T>(den, den, x2, count);
        AscendC::Adds<T>(den, den, static_cast<T>(0.39856963806e3f), count);
        AscendC::Mul<T>(den, den, x2, count);
        AscendC::Adds<T>(den, den, static_cast<T>(0.30231248150e4f), count);
        AscendC::Mul<T>(den, den, x2, count);
        AscendC::Adds<T>(den, den, static_cast<T>(0.13243365831e5f), count);
        AscendC::Mul<T>(den, den, x2, count);
        AscendC::Adds<T>(den, den, static_cast<T>(0.2667224157e5f), count);

        AscendC::Div<T>(yLocal, yLocal, den, count);
        AddTorchErfinvCorrection(yLocal, tmp, x2, den, count);

        AscendC::Duplicate<T>(den, static_cast<T>(1.0f), count);
        AscendC::Min<T>(tmp, yLocal, den, count);
        AscendC::Duplicate<T>(den, static_cast<T>(-1.0f), count);
        AscendC::Max<T>(yLocal, tmp, den, count);

        outQueueY_.template EnQue<T>(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void AddTorchErfinvCorrection(AscendC::LocalTensor<T> yLocal,
        AscendC::LocalTensor<T> xClamp, AscendC::LocalTensor<T> x2,
        AscendC::LocalTensor<T> corr, uint32_t count)
    {
        AscendC::Duplicate<T>(corr, static_cast<T>(9.20419225e-12f), count);
        AscendC::Mul<T>(corr, corr, x2, count);
        AscendC::Adds<T>(corr, corr, static_cast<T>(-7.79123642e-10f), count);
        AscendC::Mul<T>(corr, corr, x2, count);
        AscendC::Adds<T>(corr, corr, static_cast<T>(2.87843488e-8f), count);
        AscendC::Mul<T>(corr, corr, x2, count);
        AscendC::Adds<T>(corr, corr, static_cast<T>(-6.11089164e-7f), count);
        AscendC::Mul<T>(corr, corr, x2, count);
        AscendC::Adds<T>(corr, corr, static_cast<T>(8.27447580e-6f), count);
        AscendC::Mul<T>(corr, corr, x2, count);
        AscendC::Adds<T>(corr, corr, static_cast<T>(-7.51985694e-5f), count);
        AscendC::Mul<T>(corr, corr, x2, count);
        AscendC::Adds<T>(corr, corr, static_cast<T>(4.72072736e-4f), count);
        AscendC::Mul<T>(corr, corr, x2, count);
        AscendC::Adds<T>(corr, corr, static_cast<T>(-2.08602746e-3f), count);
        AscendC::Mul<T>(corr, corr, x2, count);
        AscendC::Adds<T>(corr, corr, static_cast<T>(6.54588435e-3f), count);
        AscendC::Mul<T>(corr, corr, x2, count);
        AscendC::Adds<T>(corr, corr, static_cast<T>(-1.41430763e-2f), count);
        AscendC::Mul<T>(corr, corr, x2, count);
        AscendC::Adds<T>(corr, corr, static_cast<T>(1.71667465e-2f), count);
        AscendC::Mul<T>(corr, corr, xClamp, count);
        AscendC::Add<T>(yLocal, yLocal, corr, count);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        AscendC::LocalTensor<T> yLocal = outQueueY_.template DeQue<T>();
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = count * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(yGm_[offset], yLocal, copyParams);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
    AscendC::TQue<AscendC::QuePosition::VECIN, ERFINV_BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, ERFINV_BUFFER_NUM> outQueueY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workX_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workX2_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workDen_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workTmp_;
    uint32_t blockLength_ = 0;
    uint32_t ubLength_ = ERFINV_UB_ALIGN;
};

template <typename DT_X>
__global__ __aicore__ void erfinv(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(ErfinvTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfinvTilingData, tilingData, tiling);
    KernelErfinv<DT_X> op;
    op.Init(x, y, &tilingData);
    op.Process();
}
