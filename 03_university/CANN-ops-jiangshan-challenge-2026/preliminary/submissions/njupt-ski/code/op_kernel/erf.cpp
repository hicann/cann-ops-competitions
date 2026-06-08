#include "kernel_operator.h"
#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

constexpr uint32_t ERF_ALIGN_NUM = 8; 

template <class DT_X, bool USE_POLYNOMIAL>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t usedCores, uint32_t blocksPerCore, uint32_t remainderBlocks, uint32_t tileSize) {
        this->totalLength = totalLength;
        this->tileSize = tileSize;
        
        uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= usedCores) {
            this->coreLength = 0;
            return;
        }

        uint32_t coreBlocks = blocksPerCore;
        uint32_t coreOffsetBlocks = blockIdx * blocksPerCore;
        if (blockIdx < remainderBlocks) {
            coreBlocks += 1;
            coreOffsetBlocks += blockIdx;
        } else {
            coreOffsetBlocks += remainderBlocks;
        }

        this->coreOffset = coreOffsetBlocks * ERF_ALIGN_NUM;
        uint32_t alignedCoreLength = coreBlocks * ERF_ALIGN_NUM;
        if (this->coreOffset >= this->totalLength) {
            this->coreLength = 0;
            return;
        }
        uint32_t remainingLength = this->totalLength - this->coreOffset;
        this->coreLength = alignedCoreLength < remainingLength ? alignedCoreLength : remainingLength;

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + this->coreOffset, this->coreLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + this->coreOffset, this->coreLength);

        if (USE_POLYNOMIAL) {
            pipe.InitBuffer(inQueueXPoly, 1, this->tileSize * sizeof(DT_X));
            pipe.InitBuffer(outQueueYPoly, 1, this->tileSize * sizeof(DT_X));
            pipe.InitBuffer(calcBuf, this->tileSize * 2 * sizeof(DT_X));
        } else {
            pipe.InitBuffer(inQueueXErf, 2, this->tileSize * sizeof(DT_X));
            pipe.InitBuffer(outQueueYErf, 2, this->tileSize * sizeof(DT_X));
            pipe.InitBuffer(calcBuf, this->tileSize * sizeof(DT_X));
        }
    }

    __aicore__ inline void Process() {
        if (this->coreLength == 0) return;
        if (USE_POLYNOMIAL) {
            ProcessPolynomial();
        } else {
            ProcessErf();
        }
    }

private:
    __aicore__ inline void ProcessErf() {
        uint32_t loopCount = this->coreLength / this->tileSize;
        uint32_t remainder = this->coreLength % this->tileSize;

        #pragma unroll
        for (uint32_t i = 0; i < loopCount; ++i) {
            CopyInErf(i * this->tileSize, this->tileSize);
            ComputeErf(this->tileSize);
            CopyOutErf(i * this->tileSize, this->tileSize);
        }
        if (remainder > 0) {
            uint32_t alignedRemainder = ((remainder + ERF_ALIGN_NUM - 1) / ERF_ALIGN_NUM) * ERF_ALIGN_NUM;
            CopyInErf(loopCount * this->tileSize, remainder);
            ComputeErf(alignedRemainder);
            CopyOutErf(loopCount * this->tileSize, remainder);
        }
    }

    __aicore__ inline void ProcessPolynomial() {
        uint32_t loopCount = this->coreLength / this->tileSize;
        uint32_t remainder = this->coreLength % this->tileSize;

        #pragma unroll
        for (uint32_t i = 0; i < loopCount; ++i) {
            CopyInPolynomial(i * this->tileSize, this->tileSize);
            ComputePolynomial(this->tileSize);
            CopyOutPolynomial(i * this->tileSize, this->tileSize);
        }
        if (remainder > 0) {
            uint32_t alignedRemainder = ((remainder + ERF_ALIGN_NUM - 1) / ERF_ALIGN_NUM) * ERF_ALIGN_NUM;
            CopyInPolynomial(loopCount * this->tileSize, remainder);
            ComputePolynomial(alignedRemainder);
            CopyOutPolynomial(loopCount * this->tileSize, remainder);
        }
    }

    __aicore__ inline void CopyInErf(uint32_t offset, uint32_t length) {
        uint32_t rightPadding = ((length + ERF_ALIGN_NUM - 1) / ERF_ALIGN_NUM) * ERF_ALIGN_NUM - length;
        LocalTensor<DT_X> xLocal = inQueueXErf.AllocTensor<DT_X>();
        if (rightPadding == 0) {
            DataCopy(xLocal, xGm[offset], length);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{true, 0, static_cast<uint8_t>(rightPadding), static_cast<DT_X>(0)};
            DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        }
        inQueueXErf.EnQue(xLocal);
    }

    __aicore__ inline void CopyInPolynomial(uint32_t offset, uint32_t length) {
        uint32_t rightPadding = ((length + ERF_ALIGN_NUM - 1) / ERF_ALIGN_NUM) * ERF_ALIGN_NUM - length;
        LocalTensor<DT_X> xLocal = inQueueXPoly.AllocTensor<DT_X>();
        if (rightPadding == 0) {
            DataCopy(xLocal, xGm[offset], length);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{true, 0, static_cast<uint8_t>(rightPadding), static_cast<DT_X>(0)};
            DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        }
        inQueueXPoly.EnQue(xLocal);
    }

    __aicore__ inline void ComputeErf(uint32_t length) {
        LocalTensor<DT_X> xLocal = inQueueXErf.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueYErf.AllocTensor<DT_X>();
        LocalTensor<uint8_t> sharedTmpBuffer = calcBuf.Get<uint8_t>();

        AscendC::Erf(yLocal, xLocal, sharedTmpBuffer, length);
        
        outQueueYErf.EnQue<DT_X>(yLocal);
        inQueueXErf.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputePolynomial(uint32_t length) {
        LocalTensor<DT_X> xLocal = inQueueXPoly.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueYPoly.AllocTensor<DT_X>();

        LocalTensor<DT_X> workLocal = calcBuf.Get<DT_X>();
        LocalTensor<DT_X> x2Local = workLocal;
        LocalTensor<DT_X> denominatorLocal = workLocal[this->tileSize];

        Mins(xLocal, xLocal, static_cast<DT_X>(3.92f), length);
        Maxs(xLocal, xLocal, static_cast<DT_X>(-3.92f), length);
        Mul(x2Local, xLocal, xLocal, length);

        Muls(yLocal, x2Local, static_cast<DT_X>(0.053443748819f), length);
        Adds(yLocal, yLocal, static_cast<DT_X>(7.5517016694f), length);
        Mul(yLocal, yLocal, x2Local, length);
        Adds(yLocal, yLocal, static_cast<DT_X>(101.62808918f), length);
        Mul(yLocal, yLocal, x2Local, length);
        Adds(yLocal, yLocal, static_cast<DT_X>(1393.8061484f), length);
        Mul(yLocal, yLocal, x2Local, length);
        Adds(yLocal, yLocal, static_cast<DT_X>(5063.7915060f), length);
        Mul(yLocal, yLocal, x2Local, length);
        Adds(yLocal, yLocal, static_cast<DT_X>(29638.38468f), length);
        Mul(yLocal, yLocal, xLocal, length);

        Adds(denominatorLocal, x2Local, static_cast<DT_X>(31.212858887f), length);
        Mul(denominatorLocal, denominatorLocal, x2Local, length);
        Adds(denominatorLocal, denominatorLocal, static_cast<DT_X>(398.56963806f), length);
        Mul(denominatorLocal, denominatorLocal, x2Local, length);
        Adds(denominatorLocal, denominatorLocal, static_cast<DT_X>(3023.1248150f), length);
        Mul(denominatorLocal, denominatorLocal, x2Local, length);
        Adds(denominatorLocal, denominatorLocal, static_cast<DT_X>(13243.365831f), length);
        Mul(denominatorLocal, denominatorLocal, x2Local, length);
        Adds(denominatorLocal, denominatorLocal, static_cast<DT_X>(26266.037052f), length);
        Div(yLocal, yLocal, denominatorLocal, length);
        
        outQueueYPoly.EnQue<DT_X>(yLocal);
        inQueueXPoly.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutErf(uint32_t offset, uint32_t length) {
        LocalTensor<DT_X> yLocal = outQueueYErf.DeQue<DT_X>();
        if ((length % ERF_ALIGN_NUM) == 0) {
            DataCopy(yGm[offset], yLocal, length);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(yGm[offset], yLocal, copyParams);
        }
        outQueueYErf.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutPolynomial(uint32_t offset, uint32_t length) {
        LocalTensor<DT_X> yLocal = outQueueYPoly.DeQue<DT_X>();
        if ((length % ERF_ALIGN_NUM) == 0) {
            DataCopy(yGm[offset], yLocal, length);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(yGm[offset], yLocal, copyParams);
        }
        outQueueYPoly.FreeTensor(yLocal);
    }

    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueueXErf;
    TQue<QuePosition::VECOUT, 2> outQueueYErf;
    TQue<QuePosition::VECIN, 1> inQueueXPoly;
    TQue<QuePosition::VECOUT, 1> outQueueYPoly;
    TBuf<QuePosition::VECCALC> calcBuf;
    
    uint32_t totalLength;
    uint32_t coreOffset;
    uint32_t coreLength;
    uint32_t tileSize;
};

template <typename DT_X, bool USE_POLYNOMIAL>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    KernelErf<DT_X, USE_POLYNOMIAL> op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.usedCores, tiling_data.blocksPerCore, tiling_data.remainderBlocks, tiling_data.tileSize);
    op.Process();
}
