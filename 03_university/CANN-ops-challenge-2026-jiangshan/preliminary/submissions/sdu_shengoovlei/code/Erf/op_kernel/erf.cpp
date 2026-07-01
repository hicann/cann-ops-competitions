// Kernel侧核函数实现
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

template <class DT_X>
class KernelErf {
public:
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t QUEUE_DEPTH = 1;
    static constexpr uint32_t TILE_LENGTH = 6144;

    __aicore__ inline KernelErf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, uint32_t coreNum) {
        uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t blockNum = (length + 7) / 8;
        uint32_t baseBlocks = blockNum / coreNum;
        uint32_t tailBlocks = blockNum % coreNum;
        uint32_t coreBlocks = baseBlocks + (blockIdx < tailBlocks ? 1 : 0);
        uint32_t blockOffset = blockIdx * baseBlocks + (blockIdx < tailBlocks ? blockIdx : tailBlocks);

        coreOffset = blockOffset * 8;
        coreLen = coreOffset < length ? length - coreOffset : 0;
        uint32_t maxCoreLen = coreBlocks * 8;
        coreLen = coreLen > maxCoreLen ? maxCoreLen : coreLen;

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + coreOffset, coreLen);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + coreOffset, coreLen);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(tmpBuf, TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process() {
        uint32_t offset = 0;
        while (offset < coreLen) {
            uint32_t count = (coreLen - offset) > TILE_LENGTH ? TILE_LENGTH : (coreLen - offset);
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
            offset += count;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.template AllocTensor<DT_X>();
        if ((count & 7) == 0) {
            AscendC::DataCopy(xLocal, xGm[offset], count);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        }
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.template DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.template AllocTensor<DT_X>();
        AscendC::LocalTensor<float> x2Local = tmpBuf.Get<float>();

        AscendC::Mins(xLocal, xLocal, static_cast<float>(2.2f), count);
        AscendC::Maxs(xLocal, xLocal, static_cast<float>(-2.2f), count);
        AscendC::Mul(x2Local, xLocal, xLocal, count);

        AscendC::Muls(yLocal, x2Local, static_cast<float>(9.763334094e-04f), count);
        AscendC::Adds(yLocal, yLocal, static_cast<float>(-1.491714392e-02f), count);
        AscendC::Mul(yLocal, yLocal, x2Local, count);
        AscendC::Adds(yLocal, yLocal, static_cast<float>(9.659687807e-02f), count);
        AscendC::Mul(yLocal, yLocal, x2Local, count);
        AscendC::Adds(yLocal, yLocal, static_cast<float>(-3.678617894e-01f), count);
        AscendC::Mul(yLocal, yLocal, x2Local, count);
        AscendC::Adds(yLocal, yLocal, static_cast<float>(1.127689547e+00f), count);
        AscendC::Mul(yLocal, yLocal, xLocal, count);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count) {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.template DeQue<DT_X>();
        if ((count & 7) == 0) {
            AscendC::DataCopy(yGm[offset], yLocal, count);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
        }
        outQueueY.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, QUEUE_DEPTH> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, QUEUE_DEPTH> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t coreOffset = 0;
    uint32_t coreLen = 0;
};

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    KernelErf<DT_X> op;
    op.Init(x, y, tiling_data.length, tiling_data.coreNum);
    op.Process();
}
