#include "kernel_operator.h"

#include "clip_by_value_tiling.h"
#include "tiling_key_clip_by_value.h"

constexpr int32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelClipByValue {
public:
    __aicore__ inline KernelClipByValue() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, uint32_t tileLength, uint32_t tileNum,
        float minValue, float maxValue)
    {
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t alignNum = 32 / sizeof(DT_X);
        uint32_t alignedLength = length / alignNum * alignNum;
        uint32_t alignedBlockNum = alignedLength / alignNum;
        uint32_t baseBlockNum = alignedBlockNum / blockNum;
        uint32_t tailBlockNum = alignedBlockNum % blockNum;
        uint32_t startBlock = blockIdx * baseBlockNum + (blockIdx < tailBlockNum ? blockIdx : tailBlockNum);
        uint32_t currentBlockNum = baseBlockNum + (blockIdx < tailBlockNum ? 1 : 0);
        uint32_t start = startBlock * alignNum;

        this->blockLength = currentBlockNum * alignNum;
        if (blockIdx == blockNum - 1) {
            this->blockLength += length - alignedLength;
        }
        this->hasVectorData = this->blockLength >= alignNum ? 1 : 0;
        this->minValue = static_cast<DT_X>(minValue);
        this->maxValue = static_cast<DT_X>(maxValue);

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + start, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + start, this->blockLength);
        this->tileLength = GetTileLength(this->blockLength, tileLength);
        this->tileNum = tileNum == 0 ? 1 : tileNum;

        if (this->hasVectorData) {
            pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
            pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t loopCount = this->tileNum;
        for (uint32_t i = 0; i < loopCount; ++i) {
            uint32_t offset = i * this->tileLength;
            uint32_t calcLength = GetCurrentTileLength(i);
            if (calcLength == 0) {
                continue;
            }

            uint32_t alignNum = 32 / sizeof(DT_X);
            uint32_t vectorLength = calcLength / alignNum * alignNum;
            if (this->hasVectorData && vectorLength > 0) {
                CopyIn(offset, vectorLength);
                Compute(vectorLength);
                CopyOut(offset, vectorLength);
            }
            ProcessTail(offset + vectorLength, calcLength - vectorLength);
        }
    }

private:
    __aicore__ inline uint32_t GetTileLength(uint32_t length, uint32_t tilingTileLength) const
    {
        uint32_t alignNum = 32 / sizeof(DT_X);
        if (length == 0 || tilingTileLength == 0) {
            return alignNum;
        }
        uint32_t tileLength = length < tilingTileLength ? length : tilingTileLength;
        if (tileLength == 0) {
            return alignNum;
        }
        return (tileLength + alignNum - 1) / alignNum * alignNum;
    }

    __aicore__ inline uint32_t GetCurrentTileLength(uint32_t loopIdx) const
    {
        uint32_t offset = loopIdx * this->tileLength;
        if (offset >= this->blockLength) {
            return 0;
        }

        uint32_t remainLength = this->blockLength - offset;
        return remainLength < this->tileLength ? remainLength : this->tileLength;
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t calcLength)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[offset], calcLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t calcLength)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        int32_t calcCount = static_cast<int32_t>(calcLength);
        AscendC::Maxs(yLocal, xLocal, this->minValue, calcCount);
        AscendC::Mins(yLocal, yLocal, this->maxValue, calcCount);
        outQueueY.EnQue<DT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t calcLength)
    {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        AscendC::DataCopy(yGm[offset], yLocal, calcLength);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void ProcessTail(uint32_t offset, uint32_t tailLength)
    {
        for (uint32_t i = 0; i < tailLength; ++i) {
            DT_X value = xGm.GetValue(offset + i);
            float valueFloat = static_cast<float>(value);
            float minFloat = static_cast<float>(this->minValue);
            float maxFloat = static_cast<float>(this->maxValue);
            if (valueFloat < minFloat) {
                value = this->minValue;
            } else if (valueFloat > maxFloat) {
                value = this->maxValue;
            }
            yGm.SetValue(offset + i, value);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t tileNum;
    uint32_t hasVectorData;
    DT_X minValue;
    DT_X maxValue;
};

template <typename DT_X>
__global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ClipByValueTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tilingData, tiling);
    KernelClipByValue<DT_X> op;
    op.Init(x, y, tilingData.length, tilingData.tileLength, tilingData.tileNum, tilingData.minValue,
        tilingData.maxValue);
    op.Process();
}
