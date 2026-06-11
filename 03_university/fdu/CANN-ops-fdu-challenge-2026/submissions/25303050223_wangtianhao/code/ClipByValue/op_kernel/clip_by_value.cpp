#include "kernel_operator.h"

#include "clip_by_value_tiling.h"
#include "tiling_key_clip_by_value.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_DATA_NUM = 4096;

template <class DT_X>
class KernelClipByValue {
public:
    __aicore__ inline KernelClipByValue() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t length,
        uint32_t blockLength,
        float minValue,
        float maxValue)
    {
        this->length = length;
        this->blockLength = blockLength;
        this->minScalar = static_cast<DT_X>(minValue);
        this->maxScalar = static_cast<DT_X>(maxValue);

        uint32_t coreId = AscendC::GetBlockIdx();
        uint32_t start = coreId * blockLength;

        if (start >= length) {
            this->coreLength = 0;
        } else {
            uint32_t remain = length - start;
            this->coreLength = remain > blockLength ? blockLength : remain;
        }

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + start, this->coreLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + start, this->coreLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_DATA_NUM * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_DATA_NUM * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (this->coreLength == 0) {
            return;
        }

        uint32_t loopCount = (this->coreLength + TILE_DATA_NUM - 1) / TILE_DATA_NUM;

        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t offset = i * TILE_DATA_NUM;
            uint32_t remain = this->coreLength - offset;
            this->processDataNum = remain > TILE_DATA_NUM ? TILE_DATA_NUM : remain;

            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline bool Is32BAligned(uint32_t dataNum)
    {
        return ((dataNum * sizeof(DT_X)) % 32) == 0;
    }

    __aicore__ inline void CopyGmToLocalPad(
        AscendC::LocalTensor<DT_X> dst,
        AscendC::GlobalTensor<DT_X> src,
        uint32_t dataNum)
    {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = dataNum * sizeof(DT_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        AscendC::DataCopyPadExtParams<DT_X> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;

        AscendC::DataCopyPad(dst, src, copyParams, padParams);
    }

    __aicore__ inline void CopyLocalToGmPad(
        AscendC::GlobalTensor<DT_X> dst,
        AscendC::LocalTensor<DT_X> src,
        uint32_t dataNum)
    {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = dataNum * sizeof(DT_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        AscendC::DataCopyPad(dst, src, copyParams);
    }

    __aicore__ inline void CopyIn(uint32_t progress)
    {
        uint32_t offset = progress * TILE_DATA_NUM;

        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();

        if (Is32BAligned(this->processDataNum)) {
            AscendC::DataCopy(xLocal, xGm[offset], this->processDataNum);
        } else {
            CopyGmToLocalPad(xLocal, xGm[offset], this->processDataNum);
        }

        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();

        AscendC::Maxs(
            yLocal,
            xLocal,
            this->minScalar,
            this->processDataNum
        );

        AscendC::Mins(
            yLocal,
            yLocal,
            this->maxScalar,
            this->processDataNum
        );

        outQueueY.EnQue<DT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        uint32_t offset = progress * TILE_DATA_NUM;

        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();

        if (Is32BAligned(this->processDataNum)) {
            AscendC::DataCopy(yGm[offset], yLocal, this->processDataNum);
        } else {
            CopyLocalToGmPad(yGm[offset], yLocal, this->processDataNum);
        }

        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;

    uint32_t length;
    uint32_t blockLength;
    uint32_t coreLength;
    uint32_t processDataNum;

    DT_X minScalar;
    DT_X maxScalar;
};

template <typename DT_X>
__global__ __aicore__ void clip_by_value(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ClipByValueTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tiling_data, tiling);

    KernelClipByValue<DT_X> op;
    op.Init(
        x,
        y,
        tiling_data.length,
        tiling_data.blockLength,
        tiling_data.minValue,
        tiling_data.maxValue
    );
    op.Process();
}