#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

constexpr uint32_t BUFFER_NUM       = 2;
constexpr uint32_t BUFFER_NUM_SMALL = 1;

constexpr float TANH_2A = 2.25675833419103f;
constexpr float TANH_2B = 0.20085149013500f;

constexpr float CLAMP_HI =  10.0f;
constexpr float CLAMP_LO = -10.0f;

template <class DT_X, uint32_t BUF_NUM>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ErfTilingData &tiling, AscendC::TPipe* mypipe) {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t myOffset;

        if (tiling.rem > 0) {
            uint32_t base = tiling.blockLength;
            myOffset    = base * blockIdx + (blockIdx < tiling.rem ? blockIdx : tiling.rem);
            this->myLen = base + (blockIdx < tiling.rem ? 1u : 0u);
        } else {
            myOffset = tiling.blockLength * blockIdx;
            if (myOffset >= tiling.totalLength) {
                this->myLen      = 0;
                this->tileNum    = 0;
                this->tileLength = tiling.tileLength;
                return;
            }
            uint32_t remaining = tiling.totalLength - myOffset;
            this->myLen = (remaining < tiling.blockLength) ? remaining : tiling.blockLength;
        }

        this->tileLength = tiling.tileLength;
        this->tileNum    = (this->myLen + tiling.tileLength - 1) / tiling.tileLength;

        xGm.SetGlobalBuffer((__gm__ float *)x + myOffset, this->myLen);
        yGm.SetGlobalBuffer((__gm__ float *)y + myOffset, this->myLen);
        pipe = mypipe;
        pipe->InitBuffer(inQueueX,  BUF_NUM, this->tileLength * sizeof(float));
        pipe->InitBuffer(outQueueY, BUF_NUM, this->tileLength * sizeof(float));
        pipe->InitBuffer(tmpBufA, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (int32_t i = 0; i < (int32_t)this->tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline uint32_t CurTileElems(int32_t progress) {
        uint32_t processed = progress * this->tileLength;
        uint32_t remain = this->myLen - processed;
        return (remain < this->tileLength) ? remain : this->tileLength;
    }

    __aicore__ inline void CopyIn(int32_t progress) {
        AscendC::LocalTensor<float> xLocal = inQueueX.template AllocTensor<float>();
        uint32_t elems = CurTileElems(progress);

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = elems * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        AscendC::DataCopyPadExtParams<float> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0.0f;

        AscendC::DataCopyPad(xLocal, xGm[progress * this->tileLength], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        AscendC::LocalTensor<float> xLocal = inQueueX.template DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.template AllocTensor<float>();
        AscendC::LocalTensor<float> bufA = tmpBufA.Get<float>();

        uint32_t n = this->tileLength;

        AscendC::Mul (bufA, xLocal, xLocal, n);
        AscendC::Muls(bufA, bufA, TANH_2B, n);
        AscendC::Adds(bufA, bufA, TANH_2A, n);
        AscendC::Mul (bufA, bufA, xLocal, n);

        AscendC::Mins(bufA, bufA, CLAMP_HI, n);
        AscendC::Maxs(bufA, bufA, CLAMP_LO, n);

        AscendC::Exp (bufA, bufA, n);

        AscendC::Adds(yLocal, bufA, -1.0f, n);
        AscendC::Adds(bufA,   bufA,  1.0f, n);
        AscendC::Div (yLocal, yLocal, bufA, n);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        AscendC::LocalTensor<float> yLocal = outQueueY.template DeQue<float>();
        uint32_t elems = CurTileElems(progress);

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = elems * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        AscendC::DataCopyPad(yGm[progress * this->tileLength], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe* pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUF_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUF_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufA;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t myLen;
    uint32_t tileNum;
    uint32_t tileLength;
};

template <typename DT_X, bool USE_SMALL, bool USE_MID>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    constexpr uint32_t BUF = (USE_SMALL || USE_MID) ? BUFFER_NUM_SMALL : BUFFER_NUM;
    AscendC::TPipe pipe;
    KernelErf<DT_X, BUF> op;
    op.Init(x, y, tiling_data, &pipe);
    op.Process();
}
