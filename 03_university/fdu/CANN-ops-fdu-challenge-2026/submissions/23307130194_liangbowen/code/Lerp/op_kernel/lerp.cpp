// Kernel侧核函数实现
#include "kernel_operator.h"

#include "lerp_tiling.h"
#include "tiling_key_lerp.h"

constexpr int32_t BUFFER_NUM = 1;

template <class DT_START>
class KernelLerp {
public:
    __aicore__ inline KernelLerp() {}
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR y, LerpTilingData &td)
    {
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t basePerCore = td.totalLength / blockNum;
        uint32_t tailCores = td.tailCoreNum;

        this->coreDataNum = basePerCore;
        if (coreIdx < tailCores) {
            this->coreDataNum = basePerCore + 1;
        }

        uint32_t offset = coreIdx * basePerCore + (coreIdx < tailCores ? coreIdx : tailCores);

        this->weightScalar = static_cast<DT_START>(td.weight);

        startGm.SetGlobalBuffer((__gm__ DT_START *)start + offset, this->coreDataNum);
        endGm.SetGlobalBuffer((__gm__ DT_START *)end + offset, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_START *)y + offset, this->coreDataNum);

        pipe.InitBuffer(inQueueStart, BUFFER_NUM, this->coreDataNum * sizeof(DT_START));
        pipe.InitBuffer(inQueueEnd, BUFFER_NUM, this->coreDataNum * sizeof(DT_START));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->coreDataNum * sizeof(DT_START));
    }

    __aicore__ inline void Process()
    {
        if (this->coreDataNum == 0) {
            return;
        }
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<DT_START> startLocal = inQueueStart.AllocTensor<DT_START>();
        AscendC::LocalTensor<DT_START> endLocal = inQueueEnd.AllocTensor<DT_START>();
        AscendC::DataCopy(startLocal, startGm, this->coreDataNum);
        AscendC::DataCopy(endLocal, endGm, this->coreDataNum);
        inQueueStart.EnQue(startLocal);
        inQueueEnd.EnQue(endLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DT_START> startLocal = inQueueStart.DeQue<DT_START>();
        AscendC::LocalTensor<DT_START> endLocal = inQueueEnd.DeQue<DT_START>();
        AscendC::LocalTensor<DT_START> yLocal = outQueueY.AllocTensor<DT_START>();

        AscendC::Sub(yLocal, endLocal, startLocal, this->coreDataNum);
        AscendC::Muls(yLocal, yLocal, this->weightScalar, this->coreDataNum);
        AscendC::Add(yLocal, startLocal, yLocal, this->coreDataNum);

        outQueueY.EnQue<DT_START>(yLocal);
        inQueueStart.FreeTensor(startLocal);
        inQueueEnd.FreeTensor(endLocal);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<DT_START> yLocal = outQueueY.DeQue<DT_START>();
        AscendC::DataCopy(yGm, yLocal, this->coreDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueStart;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueEnd;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DT_START> startGm;
    AscendC::GlobalTensor<DT_START> endGm;
    AscendC::GlobalTensor<DT_START> yGm;
    DT_START weightScalar;
    uint32_t coreDataNum;
};

template <typename DT_START>
 __global__ __aicore__ void lerp(GM_ADDR start, GM_ADDR end, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LerpTilingData);
    GET_TILING_DATA_WITH_STRUCT(LerpTilingData, tiling_data, tiling);
    KernelLerp<DT_START> op;
    op.Init(start, end, y, tiling_data);
    op.Process();
}

