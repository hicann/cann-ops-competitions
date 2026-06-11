// Kernel侧核函数实现
#include "kernel_operator.h"

#include "lerp_tiling.h"
#include "tiling_key_lerp.h"

constexpr uint32_t BUFFER_NUM = 1;

template <class DT_START>
class KernelLerp {
public:
    __aicore__ inline KernelLerp() {}
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR y, const LerpTilingData &tiling) {
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t baseBlockPerCore = tiling.totalBlockNum / blockNum;
        uint32_t tailBlockNum = tiling.tailBlockNum;
        uint32_t blockOffset = blockIdx * baseBlockPerCore + (blockIdx < tailBlockNum ? blockIdx : tailBlockNum);
        uint32_t coreBlockNum = baseBlockPerCore + (blockIdx < tailBlockNum ? 1 : 0);

        this->coreDataNum = coreBlockNum * ALIGN_NUM;
        uint32_t offset = blockOffset * ALIGN_NUM;
        this->weight = static_cast<DT_START>(tiling.weight);

        startGm.SetGlobalBuffer((__gm__ DT_START *)start + offset, this->coreDataNum);
        endGm.SetGlobalBuffer((__gm__ DT_START *)end + offset, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_START *)y + offset, this->coreDataNum);

        pipe.InitBuffer(inQueueStart, BUFFER_NUM, this->coreDataNum * sizeof(DT_START));
        pipe.InitBuffer(inQueueEnd, BUFFER_NUM, this->coreDataNum * sizeof(DT_START));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->coreDataNum * sizeof(DT_START));
    }

    __aicore__ inline void Process() {
        if (this->coreDataNum == 0) {
            return;
        }
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        AscendC::LocalTensor<DT_START> startLocal = inQueueStart.AllocTensor<DT_START>();
        AscendC::LocalTensor<DT_START> endLocal = inQueueEnd.AllocTensor<DT_START>();
        AscendC::DataCopy(startLocal, startGm, this->coreDataNum);
        AscendC::DataCopy(endLocal, endGm, this->coreDataNum);
        inQueueStart.EnQue(startLocal);
        inQueueEnd.EnQue(endLocal);
    }

    __aicore__ inline void Compute() {
        AscendC::LocalTensor<DT_START> startLocal = inQueueStart.DeQue<DT_START>();
        AscendC::LocalTensor<DT_START> endLocal = inQueueEnd.DeQue<DT_START>();
        AscendC::LocalTensor<DT_START> yLocal = outQueueY.AllocTensor<DT_START>();
        AscendC::Sub(yLocal, endLocal, startLocal, this->coreDataNum);
        AscendC::Muls(yLocal, yLocal, this->weight, this->coreDataNum);
        AscendC::Add(yLocal, startLocal, yLocal, this->coreDataNum);
        outQueueY.EnQue<DT_START>(yLocal);
        inQueueStart.FreeTensor(startLocal);
        inQueueEnd.FreeTensor(endLocal);
    }

    __aicore__ inline void CopyOut() {
        AscendC::LocalTensor<DT_START> yLocal = outQueueY.DeQue<DT_START>();
        AscendC::DataCopy(yGm, yLocal, this->coreDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    static constexpr uint32_t ALIGN_NUM = 32 / sizeof(DT_START);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueStart;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueEnd;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DT_START> startGm;
    AscendC::GlobalTensor<DT_START> endGm;
    AscendC::GlobalTensor<DT_START> yGm;
    DT_START weight;
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
