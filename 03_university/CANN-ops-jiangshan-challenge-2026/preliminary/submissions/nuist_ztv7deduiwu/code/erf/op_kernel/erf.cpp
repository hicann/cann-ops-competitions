// Kernel侧核函数实现
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TEMP_BUF_FACTOR = 4;

template <class DT_X>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ErfTilingData* tilingData) {
        blockLength_ = tilingData->totalNum > tilingData->blockFactor * AscendC::GetBlockIdx()
                           ? (tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx() > tilingData->blockFactor
                                  ? tilingData->blockFactor
                                  : tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx())
                           : 0;
        if (blockLength_ <= 0) {
            return;
        }
        ubLength_ = tilingData->ubFactor;

        inputGM_.SetGlobalBuffer((__gm__ DT_X*)x + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
        outputGM_.SetGlobalBuffer((__gm__ DT_X*)y + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

        pipe_.InitBuffer(inputQueue_, BUFFER_NUM, ubLength_ * sizeof(DT_X));
        pipe_.InitBuffer(outputQueue_, BUFFER_NUM, ubLength_ * sizeof(DT_X));
        pipe_.InitBuffer(tempBuf_, TEMP_BUF_FACTOR * ubLength_ * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        if (blockLength_ <= 0) {
            return;
        }
        int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
        for (int64_t i = 0; i < loopCount; i++) {
            int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
            CopyIn(i, currentNum);
            Compute(currentNum);
            CopyOut(i, currentNum);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum) {
        AscendC::LocalTensor<DT_X> xLocal = inputQueue_.template AllocTensor<DT_X>();
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(DT_X));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(xLocal, inputGM_[progress * ubLength_], copyParams, {false, 0, 0, 0});
        inputQueue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum) {
        AscendC::LocalTensor<DT_X> yLocal = outputQueue_.template DeQue<DT_X>();
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(DT_X));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(outputGM_[progress * ubLength_], yLocal, copyParams);
        outputQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void Compute(int64_t currentNum) {
        constexpr float P = 0.3275911f;
        constexpr float A1 = 0.254829592f;
        constexpr float A2 = -0.284496736f;
        constexpr float A3 = 1.421413741f;
        constexpr float A4 = -1.453152027f;
        constexpr float A5 = 1.061405429f;
        constexpr float EPS = 1e-38f;

        AscendC::LocalTensor<DT_X> xLocal = inputQueue_.template DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outputQueue_.template AllocTensor<DT_X>();

        AscendC::LocalTensor<DT_X> tempAll = tempBuf_.template Get<DT_X>(TEMP_BUF_FACTOR * ubLength_);
        AscendC::LocalTensor<DT_X> tempAbs = tempAll;
        AscendC::LocalTensor<DT_X> tempT = tempAll[ubLength_];
        AscendC::LocalTensor<DT_X> tempPoly = tempAll[2 * ubLength_];
        AscendC::LocalTensor<DT_X> tempExp = tempAll[3 * ubLength_];

        AscendC::Abs(tempAbs, xLocal, currentNum);
        AscendC::Muls(tempT, tempAbs, (DT_X)P, currentNum);
        AscendC::Adds(tempT, tempT, (DT_X)1.0f, currentNum);
        AscendC::Duplicate(tempExp, (DT_X)1.0f, currentNum);
        AscendC::Div(tempT, tempExp, tempT, currentNum);

        AscendC::Muls(tempPoly, tempT, (DT_X)A5, currentNum);
        AscendC::Adds(tempPoly, tempPoly, (DT_X)A4, currentNum);
        AscendC::Mul(tempPoly, tempPoly, tempT, currentNum);
        AscendC::Adds(tempPoly, tempPoly, (DT_X)A3, currentNum);
        AscendC::Mul(tempPoly, tempPoly, tempT, currentNum);
        AscendC::Adds(tempPoly, tempPoly, (DT_X)A2, currentNum);
        AscendC::Mul(tempPoly, tempPoly, tempT, currentNum);
        AscendC::Adds(tempPoly, tempPoly, (DT_X)A1, currentNum);
        AscendC::Mul(tempPoly, tempPoly, tempT, currentNum);

        AscendC::Mul(tempExp, xLocal, xLocal, currentNum);
        AscendC::Muls(tempExp, tempExp, (DT_X)-1.0f, currentNum);
        AscendC::Exp(tempExp, tempExp, currentNum);

        AscendC::Mul(tempPoly, tempPoly, tempExp, currentNum);
        AscendC::Muls(tempPoly, tempPoly, (DT_X)-1.0f, currentNum);
        AscendC::Adds(tempPoly, tempPoly, (DT_X)1.0f, currentNum);

        AscendC::Mul(tempPoly, tempPoly, xLocal, currentNum);
        AscendC::Adds(tempAbs, tempAbs, (DT_X)EPS, currentNum);
        AscendC::Div(yLocal, tempPoly, tempAbs, currentNum);

        outputQueue_.template EnQue<DT_X>(yLocal);
        inputQueue_.FreeTensor(xLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outputQueue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tempBuf_;

    AscendC::GlobalTensor<DT_X> inputGM_;
    AscendC::GlobalTensor<DT_X> outputGM_;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    KernelErf<DT_X> op;
    op.Init(x, y, &tiling_data);
    op.Process();
}
