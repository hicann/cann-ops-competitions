#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

constexpr uint32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, uint32_t blockLength,
        uint32_t tileLength)
    {
        this->tileLength = tileLength;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t start = blockIdx * blockLength;
        if (start >= length) {
            this->blockActualLength = 0;
        } else {
            uint32_t remain = length - start;
            this->blockActualLength = remain < blockLength ? remain : blockLength;
        }

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + start, blockActualLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + start, blockActualLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(calcBuf1, tileLength * sizeof(DT_X));
        pipe.InitBuffer(calcBuf2, tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (blockActualLength == 0) {
            return;
        }

        uint32_t loopCount = (blockActualLength + tileLength - 1) / tileLength;
        for (uint32_t i = 0; i < loopCount; ++i) {
            uint32_t offset = i * tileLength;
            uint32_t copyLength = tileLength;
            if (offset + copyLength > blockActualLength) {
                copyLength = blockActualLength - offset;
            }

            CopyIn(offset, copyLength);
            Compute(copyLength);
            CopyOut(offset, copyLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t copyLength)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        if ((copyLength & 7) == 0) {
            AscendC::DataCopy(xLocal, xGm[offset], copyLength);
        } else {
            uint32_t copyBytes = static_cast<uint32_t>(copyLength * sizeof(DT_X));
            AscendC::DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        }
        inQueueX.EnQue<DT_X>(xLocal);
    }

    __aicore__ inline void Compute(uint32_t copyLength)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> absLocal = calcBuf1.Get<DT_X>();
        AscendC::LocalTensor<DT_X> tmpLocal = calcBuf2.Get<DT_X>();
        ErfApprox(yLocal, xLocal, absLocal, tmpLocal, copyLength);
        outQueueY.EnQue<DT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void ErfApprox(AscendC::LocalTensor<DT_X> yLocal,
        AscendC::LocalTensor<DT_X> xLocal, AscendC::LocalTensor<DT_X> absLocal,
        AscendC::LocalTensor<DT_X> tmpLocal, uint32_t copyLength)
    {
        constexpr float A1 = 0.0705230784f;
        constexpr float A2 = 0.0422820123f;
        constexpr float A3 = 0.0092705272f;
        constexpr float A4 = 0.0001520143f;
        constexpr float A5 = 0.0002765672f;
        constexpr float A6 = 0.0000430638f;
        constexpr float EPS = 1.0e-12f;
        AscendC::Abs(absLocal, xLocal, copyLength);

        AscendC::Muls(yLocal, absLocal, A6, copyLength);
        AscendC::Adds(yLocal, yLocal, A5, copyLength);
        AscendC::Mul(yLocal, yLocal, absLocal, copyLength);
        AscendC::Adds(yLocal, yLocal, A4, copyLength);
        AscendC::Mul(yLocal, yLocal, absLocal, copyLength);
        AscendC::Adds(yLocal, yLocal, A3, copyLength);
        AscendC::Mul(yLocal, yLocal, absLocal, copyLength);
        AscendC::Adds(yLocal, yLocal, A2, copyLength);
        AscendC::Mul(yLocal, yLocal, absLocal, copyLength);
        AscendC::Adds(yLocal, yLocal, A1, copyLength);
        AscendC::Mul(yLocal, yLocal, absLocal, copyLength);
        AscendC::Adds(yLocal, yLocal, 1.0f, copyLength);

        AscendC::Mul(tmpLocal, yLocal, yLocal, copyLength);
        AscendC::Mul(tmpLocal, tmpLocal, tmpLocal, copyLength);
        AscendC::Mul(tmpLocal, tmpLocal, tmpLocal, copyLength);
        AscendC::Mul(tmpLocal, tmpLocal, tmpLocal, copyLength);

        AscendC::Duplicate(yLocal, 1.0f, copyLength);
        AscendC::Div(tmpLocal, yLocal, tmpLocal, copyLength);
        AscendC::Sub(tmpLocal, yLocal, tmpLocal, copyLength);

        AscendC::Adds(absLocal, absLocal, EPS, copyLength);
        AscendC::Mul(tmpLocal, tmpLocal, xLocal, copyLength);
        AscendC::Div(yLocal, tmpLocal, absLocal, copyLength);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t copyLength)
    {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        if ((copyLength & 7) == 0) {
            AscendC::DataCopy(yGm[offset], yLocal, copyLength);
        } else {
            uint32_t copyBytes = static_cast<uint32_t>(copyLength * sizeof(DT_X));
            AscendC::DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
        }
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf2;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t tileLength = 0;
    uint32_t blockActualLength = 0;
};

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    KernelErf<DT_X> op;
    op.Init(x, y, tiling_data.length, tiling_data.blockLength, tiling_data.tileLength);
    op.Process();
}
