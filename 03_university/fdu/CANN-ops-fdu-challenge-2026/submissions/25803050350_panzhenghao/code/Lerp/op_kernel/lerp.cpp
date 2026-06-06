#include "kernel_operator.h"

#include <cstdint>

#include "lerp_tiling.h"
#include "tiling_key_lerp.h"

constexpr uint32_t BLOCK_SIZE = 32;
constexpr int32_t BUFFER_NUM = 1;

template <typename TYPE, uint32_t LERP_MODE>
class KernelLerp {
public:
    __aicore__ inline KernelLerp() {}

    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR y,
                                uint32_t totalLength,
                                uint32_t smallCoreDataNum, uint32_t bigCoreDataNum,
                                uint32_t finalBigTileNum, uint32_t finalSmallTileNum,
                                uint32_t tileDataNum, uint32_t smallTailDataNum,
                                uint32_t bigTailDataNum, uint32_t tailBlockNum,
                                float weight)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * blockIdx;
        this->tileDataNum = tileDataNum;
        this->weight = weight;
        this->weightValue = (TYPE)weight;

        if (blockIdx < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (blockIdx - tailBlockNum);
        }

        if (globalBufferIndex >= totalLength) {
            this->validCoreDataNum = 0;
            this->tileNum = 0;
            this->tailDataNum = 0;
        } else {
            uint32_t remainingDataNum = totalLength - globalBufferIndex;
            this->validCoreDataNum = remainingDataNum < this->coreDataNum ? remainingDataNum : this->coreDataNum;
        }

        startGm.SetGlobalBuffer((__gm__ TYPE *)start + globalBufferIndex, this->coreDataNum);
        endGm.SetGlobalBuffer((__gm__ TYPE *)end + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE *)y + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(startQueue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
        if constexpr (LERP_MODE == 2) {
            pipe.InitBuffer(endQueue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
        }
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
    }

    __aicore__ inline void Process()
    {
        this->processDataNum = this->tileDataNum;
        for (uint32_t i = 0; i < this->tileNum; ++i) {
            uint32_t tileOffset = i * this->tileDataNum;
            if (tileOffset >= this->validCoreDataNum) {
                break;
            }
            uint32_t remain = this->validCoreDataNum - tileOffset;
            this->processDataNum = remain < this->tileDataNum ? remain : this->tileDataNum;
            if constexpr (LERP_MODE == 0 || LERP_MODE == 1) {
                CopyInSingle(i);
                ComputeSingle();
                CopyOut(i);
            } else {
                CopyIn(i);
                Compute();
                CopyOut(i);
            }
        }
    }

private:
    __aicore__ inline void CopyInSingle(uint32_t progress)
    {
        AscendC::LocalTensor<TYPE> local = startQueue.template AllocTensor<TYPE>();
        uint32_t copyBytes = this->processDataNum * sizeof(TYPE);
        if (copyBytes % BLOCK_SIZE == 0) {
            if constexpr (LERP_MODE == 0) {
                AscendC::DataCopy(local, startGm[progress * this->tileDataNum], this->processDataNum);
            } else {
                AscendC::DataCopy(local, endGm[progress * this->tileDataNum], this->processDataNum);
            }
        } else {
            AscendC::DataCopyExtParams copyParams = {1, copyBytes, 0, 0, 0};
            uint8_t rightPadding = (BLOCK_SIZE - copyBytes % BLOCK_SIZE) / sizeof(TYPE);
            AscendC::DataCopyPadExtParams<TYPE> padParams = {true, 0, rightPadding, 0};
            if constexpr (LERP_MODE == 0) {
                AscendC::DataCopyPad<TYPE>(local, startGm[progress * this->tileDataNum], copyParams, padParams);
            } else {
                AscendC::DataCopyPad<TYPE>(local, endGm[progress * this->tileDataNum], copyParams, padParams);
            }
        }
        startQueue.EnQue(local);
    }

    __aicore__ inline void ComputeSingle()
    {
        AscendC::LocalTensor<TYPE> inLocal = startQueue.template DeQue<TYPE>();
        AscendC::LocalTensor<TYPE> yLocal = outQueue.template AllocTensor<TYPE>();
        AscendC::Muls(yLocal, inLocal, (TYPE)1, this->processDataNum);
        outQueue.template EnQue<TYPE>(yLocal);
        startQueue.FreeTensor(inLocal);
    }

    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<TYPE> startLocal = startQueue.template AllocTensor<TYPE>();
        AscendC::LocalTensor<TYPE> endLocal = endQueue.template AllocTensor<TYPE>();
        uint32_t copyBytes = this->processDataNum * sizeof(TYPE);
        if (copyBytes % BLOCK_SIZE == 0) {
            AscendC::DataCopy(startLocal, startGm[progress * this->tileDataNum], this->processDataNum);
            AscendC::DataCopy(endLocal, endGm[progress * this->tileDataNum], this->processDataNum);
        } else {
            AscendC::DataCopyExtParams copyParams = {1, copyBytes, 0, 0, 0};
            uint8_t rightPadding = (BLOCK_SIZE - copyBytes % BLOCK_SIZE) / sizeof(TYPE);
            AscendC::DataCopyPadExtParams<TYPE> padParams = {true, 0, rightPadding, 0};
            AscendC::DataCopyPad<TYPE>(startLocal, startGm[progress * this->tileDataNum], copyParams, padParams);
            AscendC::DataCopyPad<TYPE>(endLocal, endGm[progress * this->tileDataNum], copyParams, padParams);
        }
        startQueue.EnQue(startLocal);
        endQueue.EnQue(endLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<TYPE> startLocal = startQueue.template DeQue<TYPE>();
        AscendC::LocalTensor<TYPE> endLocal = endQueue.template DeQue<TYPE>();
        AscendC::LocalTensor<TYPE> yLocal = outQueue.template AllocTensor<TYPE>();

        ComputeLerp(startLocal, endLocal, yLocal);

        outQueue.template EnQue<TYPE>(yLocal);
        startQueue.FreeTensor(startLocal);
        endQueue.FreeTensor(endLocal);
    }

    __aicore__ inline void ComputeLerp(AscendC::LocalTensor<TYPE> startLocal,
                                       AscendC::LocalTensor<TYPE> endLocal,
                                       AscendC::LocalTensor<TYPE> yLocal)
    {
        AscendC::Sub(yLocal, endLocal, startLocal, this->processDataNum);
        AscendC::Muls(yLocal, yLocal, this->weightValue, this->processDataNum);
        AscendC::Add(yLocal, startLocal, yLocal, this->processDataNum);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<TYPE> yLocal = outQueue.template DeQue<TYPE>();
        uint32_t copyBytes = this->processDataNum * sizeof(TYPE);
        if (copyBytes % BLOCK_SIZE == 0) {
            AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        } else {
            AscendC::DataCopyExtParams copyParams = {1, copyBytes, 0, 0, 0};
            AscendC::DataCopyPad<TYPE>(yGm[progress * this->tileDataNum], yLocal, copyParams);
        }
        outQueue.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> startQueue;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> endQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<TYPE> startGm;
    AscendC::GlobalTensor<TYPE> endGm;
    AscendC::GlobalTensor<TYPE> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    uint32_t validCoreDataNum;
    float weight;
    TYPE weightValue;
};

template <typename DT_START, uint32_t LERP_MODE>
__global__ __aicore__ void lerp(GM_ADDR start, GM_ADDR end, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LerpTilingData);
    GET_TILING_DATA_WITH_STRUCT(LerpTilingData, tilingData, tiling);
    KernelLerp<DT_START, LERP_MODE> op;
    op.Init(start, end, y, tilingData.totalLength, tilingData.smallCoreDataNum,
            tilingData.bigCoreDataNum, tilingData.finalBigTileNum,
            tilingData.finalSmallTileNum, tilingData.tileDataNum,
            tilingData.smallTailDataNum, tilingData.bigTailDataNum,
            tilingData.tailBlockNum, tilingData.weight);
    op.Process();
}
