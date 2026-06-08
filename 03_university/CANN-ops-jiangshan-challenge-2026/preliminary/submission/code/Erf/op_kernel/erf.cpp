// Kernel侧核函数实现
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

constexpr uint32_t BUFFER_NUM = 2;



template <class DT_X>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t smallCoreDataNum, uint32_t bigCoreDataNum,
                                uint32_t finalBigTileNum, uint32_t finalSmallTileNum,
                                uint32_t tileDataNum, uint32_t smallTailDataNum,
                                uint32_t bigTailDataNum, uint32_t tailBlockNum) {
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * coreNum;
        this->tileDataNum = tileDataNum;

        if (coreNum < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
        }

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DT_X));
        //pipe.InitBuffer(polyBuf, this->tileDataNum * sizeof(DT_X));
        pipe.InitBuffer(squareBuf, this->tileDataNum * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        uint32_t processDataNum = this->tileDataNum;
        for (uint32_t i = 0; i < this->tileNum; i++) {
            if (i == this->tileNum - 1) {
                processDataNum = this->tailDataNum;
            }
            CopyIn(i, processDataNum);
            Compute(processDataNum);
            CopyOut(i, processDataNum);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t processDataNum) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::DataCopyParams copyParams{1, static_cast<uint16_t>(processDataNum * sizeof(DT_X)), 0, 0};
        AscendC::DataCopyPadParams padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[progress * this->tileDataNum], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t processDataNum) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> tLocal = squareBuf.Get<DT_X>();
        
          constexpr float c1 = 1.128379167f;  // 2 / sqrt(pi)
    constexpr float c3 = 0.100899f;
    // x2 = x * x
    AscendC::Mul(tLocal, xLocal, xLocal, processDataNum);
    // y = c3 * x^2 + c1
    AscendC::Muls(yLocal, tLocal, c3, processDataNum);
    AscendC::Adds(yLocal, yLocal, c1, processDataNum);
    // y = x * (c1 + c3 * x^2)
    AscendC::Mul(yLocal, yLocal, xLocal, processDataNum);
    // y = tanh(y)
    AscendC::Tanh(yLocal, yLocal, processDataNum);

      
        // // clamp to [-1,1]
        // AscendC::Mins(yLocal, yLocal, 1.0f, processDataNum);
        // AscendC::Maxs(yLocal, yLocal, -1.0f, processDataNum);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t processDataNum) {
        AscendC::LocalTensor<DT_X> outLocal = outQueueY.DeQue<DT_X>();
        AscendC::DataCopyParams copyParams{1, static_cast<uint16_t>(processDataNum * sizeof(DT_X)), 0, 0};
        AscendC::DataCopyPad(yGm[progress * this->tileDataNum], outLocal, copyParams);
        outQueueY.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> squareBuf;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
};

template <typename DT_X>
 __global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    KernelErf<DT_X> op;
    op.Init(x, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum,
            tiling_data.finalBigTileNum, tiling_data.finalSmallTileNum,
            tiling_data.tileDataNum, tiling_data.smallTailDataNum,
            tiling_data.bigTailDataNum, tiling_data.tailBlockNum);
    op.Process();
}

