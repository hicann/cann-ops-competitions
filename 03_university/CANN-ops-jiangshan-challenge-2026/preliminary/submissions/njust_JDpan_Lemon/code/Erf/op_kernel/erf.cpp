#include "kernel_operator.h"
#include "erf_tiling.h"
#include "tiling_key_erf.h"

template <class T>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength,
                                 uint32_t coreDataNum, uint32_t tileDataNum) {
        uint32_t coreId = AscendC::GetBlockIdx();
        uint32_t offset = coreId * coreDataNum;
        this->coreDataLen = coreDataNum;
        if (offset + this->coreDataLen > totalLength) {
            this->coreDataLen = totalLength > offset ? totalLength - offset : 0;
        }
        this->tileLen = tileDataNum;

        xGm_.SetGlobalBuffer((__gm__ T *)x + offset, this->coreDataLen);
        yGm_.SetGlobalBuffer((__gm__ T *)y + offset, this->coreDataLen);

        pipe_.InitBuffer(inQueueX, 2, this->tileLen * sizeof(T));
        pipe_.InitBuffer(outQueueY, 2, this->tileLen * sizeof(T));
        pipe_.InitBuffer(bufExp, this->tileLen * sizeof(T));
    }

    __aicore__ inline void Process() {
        if (this->coreDataLen == 0) return;

        uint32_t loopCount = this->coreDataLen / this->tileLen;
        uint32_t remainLen = this->coreDataLen % this->tileLen;

        if (loopCount == 0) {
            if (remainLen > 0) {
                CopyIn(0, remainLen);
                Compute(remainLen);
                CopyOut(0, remainLen);
            }
            return;
        }

        // Depth-2 double-buffer pipeline with remainder pre-fetch
        CopyIn(0, this->tileLen);
        uint32_t inOffset  = this->tileLen;
        uint32_t outOffset = 0;

        for (uint32_t i = 0; i < loopCount - 1; ++i) {
            CopyIn(inOffset, this->tileLen);
            inOffset += this->tileLen;
            Compute(this->tileLen);
            CopyOut(outOffset, this->tileLen);
            outOffset += this->tileLen;
        }

        if (remainLen > 0) {
            CopyIn(inOffset, remainLen);
        }

        Compute(this->tileLen);
        CopyOut(outOffset, this->tileLen);

        if (remainLen > 0) {
            Compute(remainLen);
            CopyOut(inOffset, remainLen);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t gmOffset, uint32_t len) {
        constexpr uint32_t ALIGN_NUM = 32 / sizeof(T);
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        uint32_t alignedLen = (len + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
        AscendC::DataCopy(xLocal, xGm_[gmOffset], alignedLen);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len) {
        constexpr uint32_t ALIGN_NUM = 32 / sizeof(T);
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        AscendC::LocalTensor<T> expLocal = bufExp.Get<T>();

        uint32_t alignedLen = (len + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;

        // x2 = x^2
        AscendC::Mul(expLocal, xLocal, xLocal, alignedLen);

        // Horner: P(x^2) with 7 coefficients for 1e-3, safe tail
        // y = c6*x2 + c5
        AscendC::Muls(yLocal, expLocal, (T)6.48396053e-06f, alignedLen);
        AscendC::Adds(yLocal, yLocal, (T)-2.02451265e-04f, alignedLen);
        // y = y*x2 + c4
        AscendC::Mul(yLocal, yLocal, expLocal, alignedLen);
        AscendC::Adds(yLocal, yLocal, (T)2.72023887e-03f, alignedLen);
        // y = y*x2 + c3
        AscendC::Mul(yLocal, yLocal, expLocal, alignedLen);
        AscendC::Adds(yLocal, yLocal, (T)-2.09796298e-02f, alignedLen);
        // y = y*x2 + c2
        AscendC::Mul(yLocal, yLocal, expLocal, alignedLen);
        AscendC::Adds(yLocal, yLocal, (T)1.05228491e-01f, alignedLen);
        // y = y*x2 + c1
        AscendC::Mul(yLocal, yLocal, expLocal, alignedLen);
        AscendC::Adds(yLocal, yLocal, (T)-3.71638149e-01f, alignedLen);
        // y = y*x2 + c0
        AscendC::Mul(yLocal, yLocal, expLocal, alignedLen);
        AscendC::Adds(yLocal, yLocal, (T)1.12760413e+00f, alignedLen);

        // erf = x * P(x^2)
        AscendC::Mul(yLocal, yLocal, xLocal, alignedLen);

        // Free xLocal early: not needed for clamp, unblocks next CopyIn
        inQueueX.FreeTensor(xLocal);

        // Clamp to [-1, 1]: Mins/Maxs without Duplicate overhead
        AscendC::Mins(yLocal, yLocal, (T)1.0f, alignedLen);
        AscendC::Maxs(yLocal, yLocal, (T)-1.0f, alignedLen);

        outQueueY.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t gmOffset, uint32_t len) {
        constexpr uint32_t ALIGN_NUM = 32 / sizeof(T);
        AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        uint32_t alignedLen = (len + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
        AscendC::DataCopy(yGm_[gmOffset], yLocal, alignedLen);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2>  inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC>     bufExp;

    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
    uint32_t coreDataLen;
    uint32_t tileLen;
};

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    KernelErf<DT_X> op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.coreDataNum, tiling_data.tileDataNum);
    op.Process();
}
