#include <type_traits>

#include "kernel_operator.h"

#include "clip_by_value_tiling.h"
#include "tiling_key_clip_by_value.h"

constexpr int32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelClipByValue {
public:
    __aicore__ inline KernelClipByValue() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ClipByValueTilingData &tiling)
    {
        uint32_t globalBufferIndex = tiling.bigCoreDataNum * AscendC::GetBlockIdx();
        this->tileDataNum = tiling.tileDataNum;
        this->minValue = tiling.minValue;
        this->maxValue = tiling.maxValue;

        if (AscendC::GetBlockIdx() < tiling.tailBlockNum) {
            this->coreDataNum = tiling.bigCoreDataNum;
            this->tileNum = tiling.finalBigTileNum;
            this->tailDataNum = tiling.bigTailDataNum;
        } else {
            this->coreDataNum = tiling.smallCoreDataNum;
            this->tileNum = tiling.finalSmallTileNum;
            this->tailDataNum = tiling.smallTailDataNum;
            globalBufferIndex -=
                (tiling.bigCoreDataNum - tiling.smallCoreDataNum) * (AscendC::GetBlockIdx() - tiling.tailBlockNum);
        }

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DT_X));
        if constexpr (std::is_same_v<DT_X, int32_t>) {
            pipe.InitBuffer(tmpBuffer, this->tileDataNum * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        this->processDataNum = this->tileDataNum;
        int32_t fullTileNum = static_cast<int32_t>(this->tileNum) - 1;
        for (int32_t i = 0; i < fullTileNum; ++i) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }

        this->processDataNum = this->tailDataNum;
        CopyIn(fullTileNum);
        Compute();
        CopyOut(fullTileNum);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();

        if constexpr (std::is_same_v<DT_X, int32_t>) {
            AscendC::LocalTensor<float> tmpLocal = tmpBuffer.Get<float>();
            AscendC::Cast(tmpLocal, xLocal, AscendC::RoundMode::CAST_RINT, this->processDataNum);
            AscendC::Mins(tmpLocal, tmpLocal, this->maxValue, this->processDataNum);
            AscendC::Maxs(tmpLocal, tmpLocal, this->minValue, this->processDataNum);
            AscendC::Cast(yLocal, tmpLocal, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        } else if constexpr (std::is_same_v<DT_X, half>) {
            AscendC::Mins(yLocal, xLocal, static_cast<half>(this->maxValue), this->processDataNum);
            AscendC::Maxs(yLocal, yLocal, static_cast<half>(this->minValue), this->processDataNum);
        } else {
            AscendC::Mins(yLocal, xLocal, static_cast<DT_X>(this->maxValue), this->processDataNum);
            AscendC::Maxs(yLocal, yLocal, static_cast<DT_X>(this->minValue), this->processDataNum);
        }

        outQueueY.EnQue<DT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    float minValue;
    float maxValue;
};

template <typename DT_X>
__global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ClipByValueTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tilingData, tiling);
    KernelClipByValue<DT_X> op;
    op.Init(x, y, tilingData);
    op.Process();
}
