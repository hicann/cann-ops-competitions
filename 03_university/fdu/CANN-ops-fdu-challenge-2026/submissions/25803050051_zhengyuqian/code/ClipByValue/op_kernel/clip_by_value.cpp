// Kernel侧核函数实现
#include "kernel_operator.h"

#include "clip_by_value_tiling.h"
#include "tiling_key_clip_by_value.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t ALIGN_BYTES = 32;
constexpr uint32_t GM_ALIGN_BYTES = 512;

template <class DT_X>
class KernelClipByValue {
public:
    __aicore__ inline KernelClipByValue() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, uint32_t packedTileCore,
        float minVal, float maxVal) {
        this->length = length;
        uint32_t alignElems = ALIGN_BYTES / sizeof(DT_X);
        if (alignElems == 0) {
            alignElems = 1;
        }
        uint32_t gmAlignElems = GM_ALIGN_BYTES / sizeof(DT_X);
        if (gmAlignElems == 0) {
            gmAlignElems = 1;
        }
        uint32_t tileBlocks = packedTileCore & 0xffffU;
        uint32_t perCoreBlocks = packedTileCore >> 16;
        this->tileLength = (tileBlocks == 0 ? 1 : tileBlocks) * alignElems;
        this->perCoreLength = perCoreBlocks == 0 ? this->length : perCoreBlocks * gmAlignElems;
        this->minVal = static_cast<DT_X>(minVal);
        this->maxVal = static_cast<DT_X>(maxVal);

        uint32_t blockIdx = AscendC::GetBlockIdx();
        this->start = blockIdx * this->perCoreLength;
        if (this->start >= this->length) {
            this->coreLength = 0;
        } else {
            uint32_t remain = this->length - this->start;
            this->coreLength = remain < this->perCoreLength ? remain : this->perCoreLength;
        }

        if (this->coreLength == 0) {
            return;
        }
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + this->start, this->coreLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + this->start, this->coreLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        if (this->coreLength == 0) {
            return;
        }
        for (uint32_t offset = 0; offset < this->coreLength; offset += this->tileLength) {
            uint32_t count = this->coreLength - offset;
            count = count < this->tileLength ? count : this->tileLength;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }
private:
    __aicore__ inline bool IsAlignedCopy(uint32_t count) const {
        return count == this->tileLength && (count * sizeof(DT_X)) % ALIGN_BYTES == 0;
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        if (IsAlignedCopy(count)) {
            AscendC::DataCopy(xLocal, xGm[offset], count);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, static_cast<DT_X>(0)};
            AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        }
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        AscendC::Maxs(yLocal, xLocal, this->minVal, static_cast<int32_t>(count));
        AscendC::Mins(yLocal, yLocal, this->maxVal, static_cast<int32_t>(count));
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count) {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        if (IsAlignedCopy(count)) {
            AscendC::DataCopy(yGm[offset], yLocal, count);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
        }
        outQueueY.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t length;
    uint32_t tileLength;
    uint32_t perCoreLength;
    uint32_t start;
    uint32_t coreLength;
    DT_X minVal;
    DT_X maxVal;
};

template <typename DT_X>
 __global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ClipByValueTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tiling_data, tiling);
    KernelClipByValue<DT_X> op;
    op.Init(x, y, tiling_data.length, tiling_data.packedTileCore, tiling_data.minVal, tiling_data.maxVal);
    op.Process();
}
