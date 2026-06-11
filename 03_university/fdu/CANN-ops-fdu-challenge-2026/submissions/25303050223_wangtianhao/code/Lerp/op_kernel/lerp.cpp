#include "kernel_operator.h"

#include "lerp_tiling.h"
#include "tiling_key_lerp.h"

namespace {
constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t TILE_DATA_NUM = 4096;
}

template <class DT_X>
class KernelLerp {
public:
    __aicore__ inline KernelLerp() {}

    __aicore__ inline void Init(
        GM_ADDR start,
        GM_ADDR end,
        GM_ADDR y,
        uint32_t length,
        uint32_t blockLength,
        float weight)
    {
        this->length = length;
        this->blockLength = blockLength;
        this->weightScalar = static_cast<DT_X>(weight);

        uint32_t coreId = AscendC::GetBlockIdx();
        uint32_t startOffset = coreId * blockLength;

        if (startOffset >= length) {
            this->coreLength = 0;
        } else {
            uint32_t remain = length - startOffset;
            this->coreLength = remain > blockLength ? blockLength : remain;
        }

        startGm.SetGlobalBuffer((__gm__ DT_X *)start + startOffset, this->coreLength);
        endGm.SetGlobalBuffer((__gm__ DT_X *)end + startOffset, this->coreLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + startOffset, this->coreLength);

        pipe.InitBuffer(inQueueStart, BUFFER_NUM, TILE_DATA_NUM * sizeof(DT_X));
        pipe.InitBuffer(inQueueEnd, BUFFER_NUM, TILE_DATA_NUM * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_DATA_NUM * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (this->coreLength == 0) {
            return;
        }

        uint32_t loopCount = (this->coreLength + TILE_DATA_NUM - 1) / TILE_DATA_NUM;

        for (uint32_t progress = 0; progress < loopCount; ++progress) {
            uint32_t dataNum = GetDataNum(progress);

            CopyIn(progress, dataNum);
            Compute(dataNum);
            CopyOut(progress, dataNum);
        }
    }

private:
    // 当前分片元素数量计算
    __aicore__ inline uint32_t GetDataNum(uint32_t progress)
    {
        uint32_t offset = progress * TILE_DATA_NUM;
        uint32_t remain = this->coreLength - offset;
        return remain > TILE_DATA_NUM ? TILE_DATA_NUM : remain;
    }

    // 32字节对齐判定
    __aicore__ inline bool Is32BAligned(uint32_t dataNum)
    {
        return ((dataNum * sizeof(DT_X)) % 32) == 0;
    }

    // 非对齐GM到UB搬入
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

    // 非对齐UB到GM搬出
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

    // start和end输入搬入
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t dataNum)
    {
        uint32_t offset = progress * TILE_DATA_NUM;

        AscendC::LocalTensor<DT_X> startLocal = inQueueStart.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> endLocal = inQueueEnd.AllocTensor<DT_X>();

        if (Is32BAligned(dataNum)) {
            AscendC::DataCopy(startLocal, startGm[offset], dataNum);
            AscendC::DataCopy(endLocal, endGm[offset], dataNum);
        } else {
            CopyGmToLocalPad(startLocal, startGm[offset], dataNum);
            CopyGmToLocalPad(endLocal, endGm[offset], dataNum);
        }

        inQueueStart.EnQue(startLocal);
        inQueueEnd.EnQue(endLocal);
    }

    // Lerp向量计算
    __aicore__ inline void Compute(uint32_t dataNum)
    {
        AscendC::LocalTensor<DT_X> startLocal = inQueueStart.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> endLocal = inQueueEnd.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();

        AscendC::Sub(yLocal, endLocal, startLocal, dataNum);
        AscendC::Muls(yLocal, yLocal, this->weightScalar, dataNum);
        AscendC::Add(yLocal, startLocal, yLocal, dataNum);

        outQueueY.EnQue<DT_X>(yLocal);

        inQueueStart.FreeTensor(startLocal);
        inQueueEnd.FreeTensor(endLocal);
    }

    // 输出搬出
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t dataNum)
    {
        uint32_t offset = progress * TILE_DATA_NUM;
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();

        if (Is32BAligned(dataNum)) {
            AscendC::DataCopy(yGm[offset], yLocal, dataNum);
        } else {
            CopyLocalToGmPad(yGm[offset], yLocal, dataNum);
        }

        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueStart;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueEnd;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    AscendC::GlobalTensor<DT_X> startGm;
    AscendC::GlobalTensor<DT_X> endGm;
    AscendC::GlobalTensor<DT_X> yGm;

    uint32_t length;
    uint32_t blockLength;
    uint32_t coreLength;

    DT_X weightScalar;
};

template <typename DT_X>
__global__ __aicore__ void lerp(
    GM_ADDR start,
    GM_ADDR end,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LerpTilingData);
    GET_TILING_DATA_WITH_STRUCT(LerpTilingData, tiling_data, tiling);

    KernelLerp<DT_X> op;
    op.Init(
        start,
        end,
        y,
        tiling_data.length,
        tiling_data.blockLength,
        tiling_data.weight);
    op.Process();
}