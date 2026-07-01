// Kernel侧核函数实现
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t SMALL_BUFFER_NUM = 1;
constexpr uint32_t FLOAT_ALIGN_NUM = 8;
constexpr uint32_t FLOAT_ALIGN_MASK = FLOAT_ALIGN_NUM - 1;

template <class DT_X, int32_t QUEUE_NUM>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength,
        uint32_t blockLength, uint32_t tileLength) {
        totalLength_ = totalLength;
        blockLength_ = blockLength;
        tileLength_ = tileLength;
        uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        startOffset = blockIdx * blockLength_;
        coreLength = 0;
        if (startOffset < totalLength_) {
            uint32_t remain = totalLength_ - startOffset;
            coreLength = remain < blockLength_ ? remain : blockLength_;
        }
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + startOffset, coreLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + startOffset, coreLength);
        pipe.InitBuffer(inQueueX, QUEUE_NUM, tileLength_ * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, QUEUE_NUM, tileLength_ * sizeof(DT_X));
        pipe.InitBuffer(x2Buf, tileLength_ * sizeof(DT_X));
    }
    __aicore__ inline void ProcessSingle() {
        if (coreLength == 0) {
            return;
        }

        CopyIn(0, coreLength);
        Compute(coreLength);
        CopyOut(0, coreLength);
    }
    __aicore__ inline void ProcessSingleAligned() {
        if (coreLength == 0) {
            return;
        }

        if ((coreLength & FLOAT_ALIGN_MASK) != 0) {
            ProcessSingle();
            return;
        }

        CopyInAligned(0, coreLength);
        Compute(coreLength);
        CopyOutAligned(0, coreLength);
    }

    __aicore__ inline void ProcessGeneral() {
        if (coreLength == 0) {
            return;
        }

        if (coreLength <= tileLength_) {
            ProcessSingle();
            return;
        }

        uint32_t loopCount = (coreLength + tileLength_ - 1) / tileLength_;
        for (uint32_t i = 0; i < loopCount; ++i) {
            uint32_t offset = i * tileLength_;
            uint32_t dataNum = coreLength - offset;
            if (dataNum > tileLength_) {
                dataNum = tileLength_;
            }
            CopyIn(offset, dataNum);
            Compute(dataNum);
            CopyOut(offset, dataNum);
        }
    }
private:
    __aicore__ inline void CopyInAligned(uint32_t offset, uint32_t dataNum) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.template AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[offset], dataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t dataNum) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.template AllocTensor<DT_X>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataNum * sizeof(DT_X)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t dataNum) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.template DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.template AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> x2 = x2Buf.Get<DT_X>();

        AscendC::Maxs(xLocal, xLocal, static_cast<DT_X>(-2.1895f), dataNum);
        AscendC::Mins(xLocal, xLocal, static_cast<DT_X>(2.1895f), dataNum);
        AscendC::Mul(x2, xLocal, xLocal, dataNum);

        AscendC::Muls(yLocal, x2, static_cast<DT_X>(0.000892274083f), dataNum);
        AscendC::Adds(yLocal, yLocal, static_cast<DT_X>(-0.0140916533f), dataNum);
        AscendC::Mul(yLocal, yLocal, x2, dataNum);
        AscendC::Adds(yLocal, yLocal, static_cast<DT_X>(0.0940584244f), dataNum);
        AscendC::Mul(yLocal, yLocal, x2, dataNum);
        AscendC::Adds(yLocal, yLocal, static_cast<DT_X>(-0.365330579f), dataNum);
        AscendC::Mul(yLocal, yLocal, x2, dataNum);
        AscendC::Adds(yLocal, yLocal, static_cast<DT_X>(1.12727217f), dataNum);
        AscendC::Mul(yLocal, yLocal, xLocal, dataNum);
        outQueueY.template EnQue<DT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutAligned(uint32_t offset, uint32_t dataNum) {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.template DeQue<DT_X>();
        AscendC::DataCopy(yGm[offset], yLocal, dataNum);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t dataNum) {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.template DeQue<DT_X>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataNum * sizeof(DT_X)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, QUEUE_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, QUEUE_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> x2Buf;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t totalLength_;
    uint32_t blockLength_;
    uint32_t tileLength_;
    uint32_t startOffset;
    uint32_t coreLength;
};

template <class DT_X>
class KernelErfSmall {
public:
    __aicore__ inline KernelErfSmall() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t blockLength) {
        uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t startOffset = blockIdx * blockLength;
        coreLength = 0;
        if (startOffset < totalLength) {
            uint32_t remain = totalLength - startOffset;
            coreLength = remain < blockLength ? remain : blockLength;
        }
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + startOffset, coreLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + startOffset, coreLength);
        uint32_t bufferLength = coreLength < FLOAT_ALIGN_NUM ? FLOAT_ALIGN_NUM : coreLength;
        pipe.InitBuffer(inQueueX, SMALL_BUFFER_NUM, bufferLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, SMALL_BUFFER_NUM, bufferLength * sizeof(DT_X));
        pipe.InitBuffer(x2Buf, bufferLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (coreLength == 0) {
            return;
        }

        CopyIn(0, coreLength);
        Compute(coreLength);
        CopyOut(0, coreLength);
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t dataNum) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.template AllocTensor<DT_X>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataNum * sizeof(DT_X)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t dataNum) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.template DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.template AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> x2 = x2Buf.Get<DT_X>();

        AscendC::Maxs(xLocal, xLocal, static_cast<DT_X>(-2.1895f), dataNum);
        AscendC::Mins(xLocal, xLocal, static_cast<DT_X>(2.1895f), dataNum);
        AscendC::Mul(x2, xLocal, xLocal, dataNum);

        AscendC::Muls(yLocal, x2, static_cast<DT_X>(0.000892274083f), dataNum);
        AscendC::Adds(yLocal, yLocal, static_cast<DT_X>(-0.0140916533f), dataNum);
        AscendC::Mul(yLocal, yLocal, x2, dataNum);
        AscendC::Adds(yLocal, yLocal, static_cast<DT_X>(0.0940584244f), dataNum);
        AscendC::Mul(yLocal, yLocal, x2, dataNum);
        AscendC::Adds(yLocal, yLocal, static_cast<DT_X>(-0.365330579f), dataNum);
        AscendC::Mul(yLocal, yLocal, x2, dataNum);
        AscendC::Adds(yLocal, yLocal, static_cast<DT_X>(1.12727217f), dataNum);
        AscendC::Mul(yLocal, yLocal, xLocal, dataNum);
        outQueueY.template EnQue<DT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t dataNum) {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.template DeQue<DT_X>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataNum * sizeof(DT_X)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, SMALL_BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, SMALL_BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> x2Buf;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t coreLength;
};

template <typename DT_X, uint32_t schMode>
struct KernelErfLauncher;

template <typename DT_X>
struct KernelErfLauncher<DT_X, ERF_TPL_MODE_SMALL> {
    __aicore__ inline static void Run(GM_ADDR x, GM_ADDR y, uint32_t totalLength,
        uint32_t blockLength, uint32_t tileLength) {
        (void)tileLength;
        KernelErfSmall<DT_X> op;
        op.Init(x, y, totalLength, blockLength);
        op.Process();
    }
};

template <typename DT_X>
struct KernelErfLauncher<DT_X, ERF_TPL_MODE_GENERAL> {
    __aicore__ inline static void Run(GM_ADDR x, GM_ADDR y, uint32_t totalLength,
        uint32_t blockLength, uint32_t tileLength) {
        KernelErf<DT_X, BUFFER_NUM> op;
        op.Init(x, y, totalLength, blockLength, tileLength);
        op.ProcessGeneral();
    }
};

template <typename DT_X, uint32_t schMode>
 __global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    KernelErfLauncher<DT_X, schMode>::Run(
        x, y, tiling_data.totalLength, tiling_data.blockLength, tiling_data.tileLength);
}
