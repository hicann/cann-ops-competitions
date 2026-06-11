// Kernel implementation
#include "kernel_operator.h"
#include "lerp_tiling.h"
#include "tiling_key_lerp.h"

constexpr uint32_t BUFFER_NUM = 2;

template <class DT_START>
class KernelLerp {
public:
    __aicore__ inline KernelLerp() {}

    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR y,
                                uint32_t totalLength, uint32_t tileNum, float weight)
    {
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t totalDataBlocks = totalLength / DATA_BLOCK_LENGTH;
        uint32_t dataBlocksPerCore = (totalDataBlocks + blockNum - 1) / blockNum;
        uint32_t blockStart = dataBlocksPerCore * blockIdx;

        this->blockOffset = blockStart * DATA_BLOCK_LENGTH;
        this->alignedBlockLength = 0;
        if (blockStart < totalDataBlocks) {
            uint32_t remainingBlocks = totalDataBlocks - blockStart;
            uint32_t currentBlocks = remainingBlocks < dataBlocksPerCore ? remainingBlocks : dataBlocksPerCore;
            this->alignedBlockLength = currentBlocks * DATA_BLOCK_LENGTH;
        }

        this->totalLength = totalLength;
        this->tileNum = tileNum == 0 ? 1 : tileNum;
        uint32_t loopCount = this->tileNum * BUFFER_NUM;
        this->tileLength = (this->alignedBlockLength + loopCount - 1) / loopCount;
        this->tileLength = (this->tileLength + DATA_BLOCK_LENGTH - 1) / DATA_BLOCK_LENGTH * DATA_BLOCK_LENGTH;
        if (this->tileLength == 0) {
            this->tileLength = DATA_BLOCK_LENGTH;
        }
        this->weight = weight;

        startGm.SetGlobalBuffer((__gm__ DT_START *)start, totalLength);
        endGm.SetGlobalBuffer((__gm__ DT_START *)end, totalLength);
        yGm.SetGlobalBuffer((__gm__ DT_START *)y, totalLength);

        pipe.InitBuffer(startQueue, BUFFER_NUM, this->tileLength * sizeof(DT_START));
        pipe.InitBuffer(endQueue, BUFFER_NUM, this->tileLength * sizeof(DT_START));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(DT_START));
        pipe.InitBuffer(tmpBuf0, this->tileLength * sizeof(DT_START));
    }

    __aicore__ inline void Process()
    {
        if (this->weight == 0.0f) {
            ProcessCopyStart();
            ProcessTail();
            return;
        }
        if (this->weight == 1.0f) {
            ProcessCopyEnd();
            ProcessTail();
            return;
        }

        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            uint32_t currentTileLength = GetCurrentTileLength(i);
            if (currentTileLength == 0) {
                continue;
            }

            CopyIn(i, currentTileLength);
            Compute(currentTileLength);
            CopyOut(i, currentTileLength);
        }
        ProcessTail();
    }

private:
    static constexpr uint32_t DATA_BLOCK_LENGTH = 32 / sizeof(DT_START);

    __aicore__ inline uint32_t GetCurrentTileLength(int32_t progress)
    {
        uint32_t offset = progress * this->tileLength;
        if (offset >= this->alignedBlockLength) {
            return 0;
        }

        uint32_t remainingLength = this->alignedBlockLength - offset;
        return remainingLength < this->tileLength ? remainingLength : this->tileLength;
    }

    __aicore__ inline void CopyIn(int32_t progress, uint32_t currentTileLength)
    {
        AscendC::LocalTensor<DT_START> startLocal = startQueue.AllocTensor<DT_START>();
        AscendC::DataCopy(startLocal, startGm[this->blockOffset + progress * this->tileLength], currentTileLength);
        startQueue.EnQue(startLocal);

        if (this->weight != 0.0f) {
            AscendC::LocalTensor<DT_START> endLocal = endQueue.AllocTensor<DT_START>();
            AscendC::DataCopy(endLocal, endGm[this->blockOffset + progress * this->tileLength], currentTileLength);
            endQueue.EnQue(endLocal);
        }
    }

    __aicore__ inline void ProcessCopyStart()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            uint32_t currentTileLength = GetCurrentTileLength(i);
            if (currentTileLength == 0) {
                continue;
            }

            uint32_t offset = this->blockOffset + i * this->tileLength;
            AscendC::LocalTensor<DT_START> startLocal = startQueue.AllocTensor<DT_START>();
            AscendC::DataCopy(startLocal, startGm[offset], currentTileLength);
            startQueue.EnQue(startLocal);

            startLocal = startQueue.DeQue<DT_START>();
            AscendC::DataCopy(yGm[offset], startLocal, currentTileLength);
            startQueue.FreeTensor(startLocal);
        }
    }

    __aicore__ inline void ProcessCopyEnd()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            uint32_t currentTileLength = GetCurrentTileLength(i);
            if (currentTileLength == 0) {
                continue;
            }

            uint32_t offset = this->blockOffset + i * this->tileLength;
            AscendC::LocalTensor<DT_START> endLocal = endQueue.AllocTensor<DT_START>();
            AscendC::DataCopy(endLocal, endGm[offset], currentTileLength);
            endQueue.EnQue(endLocal);

            endLocal = endQueue.DeQue<DT_START>();
            AscendC::DataCopy(yGm[offset], endLocal, currentTileLength);
            endQueue.FreeTensor(endLocal);
        }
    }

    __aicore__ inline void Compute(uint32_t currentTileLength)
    {
        AscendC::LocalTensor<DT_START> startLocal = startQueue.DeQue<DT_START>();
        AscendC::LocalTensor<DT_START> yLocal = outQueue.AllocTensor<DT_START>();

        if (this->weight == 0.0f) {
            AscendC::Adds(yLocal, startLocal, static_cast<DT_START>(0), currentTileLength);
        } else {
            AscendC::LocalTensor<DT_START> endLocal = endQueue.DeQue<DT_START>();
            if (this->weight == 1.0f) {
                AscendC::Adds(yLocal, endLocal, static_cast<DT_START>(0), currentTileLength);
            } else {
                AscendC::LocalTensor<DT_START> tmp0 = tmpBuf0.Get<DT_START>();
                AscendC::Sub(tmp0, endLocal, startLocal, currentTileLength);
                AscendC::Muls(tmp0, tmp0, static_cast<DT_START>(this->weight), currentTileLength);
                AscendC::Add(yLocal, startLocal, tmp0, currentTileLength);
            }
            endQueue.FreeTensor(endLocal);
        }

        outQueue.EnQue<DT_START>(yLocal);
        startQueue.FreeTensor(startLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t currentTileLength)
    {
        AscendC::LocalTensor<DT_START> yLocal = outQueue.DeQue<DT_START>();
        AscendC::DataCopy(yGm[this->blockOffset + progress * this->tileLength], yLocal, currentTileLength);
        outQueue.FreeTensor(yLocal);
    }

    __aicore__ inline void ProcessTail()
    {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }

        uint32_t tailOffset = this->totalLength / DATA_BLOCK_LENGTH * DATA_BLOCK_LENGTH;
        for (uint32_t i = tailOffset; i < this->totalLength; ++i) {
            if (this->weight == 0.0f) {
                yGm.SetValue(i, startGm.GetValue(i));
            } else if (this->weight == 1.0f) {
                yGm.SetValue(i, endGm.GetValue(i));
            } else {
                float startValue = static_cast<float>(startGm.GetValue(i));
                float endValue = static_cast<float>(endGm.GetValue(i));
                float result = startValue + this->weight * (endValue - startValue);
                yGm.SetValue(i, static_cast<DT_START>(result));
            }
        }
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> startQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> endQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf0;

    AscendC::GlobalTensor<DT_START> startGm;
    AscendC::GlobalTensor<DT_START> endGm;
    AscendC::GlobalTensor<DT_START> yGm;

    uint32_t totalLength;
    uint32_t blockOffset;
    uint32_t alignedBlockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    float weight;
};

template <typename DT_START>
__global__ __aicore__ void lerp(GM_ADDR start, GM_ADDR end, GM_ADDR y,
                                GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LerpTilingData);
    GET_TILING_DATA_WITH_STRUCT(LerpTilingData, tiling_data, tiling);

    KernelLerp<DT_START> op;
    op.Init(start, end, y, tiling_data.length, tiling_data.tileNum, tiling_data.weight);
    op.Process();
}
