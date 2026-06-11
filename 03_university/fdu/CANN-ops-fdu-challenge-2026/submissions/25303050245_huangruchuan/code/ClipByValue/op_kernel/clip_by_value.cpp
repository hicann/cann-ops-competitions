#include "kernel_operator.h"
#include "clip_by_value_tiling.h"
#include "tiling_key_clip_by_value.h"

namespace {
constexpr uint32_t kTileElems = 8192;
constexpr uint32_t kQueueDepth = 1;
}  // namespace

template <class DT_X>
class ClipByValueKernel {
public:
    __aicore__ inline ClipByValueKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ClipByValueTilingData &tiling)
    {
        MakeShard(tiling.length);
        minValue_ = static_cast<DT_X>(tiling.minValue);
        maxValue_ = static_cast<DT_X>(tiling.maxValue);
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x + begin_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y + begin_);
        pipe_.InitBuffer(xQueue_, kQueueDepth, kTileElems * sizeof(DT_X));
        pipe_.InitBuffer(yQueue_, kQueueDepth, kTileElems * sizeof(DT_X));
        padParams_.isPad = false;
        padParams_.leftPadding = 0;
        padParams_.rightPadding = 0;
    }

    __aicore__ inline void Process()
    {
        if (length_ == 0) {
            return;
        }
        for (uint32_t offset = 0; offset < length_; offset += kTileElems) {
            uint32_t elems = (offset + kTileElems <= length_) ? kTileElems : (length_ - offset);
            Load(offset, elems);
            Clamp(elems);
            Store(offset, elems);
        }
    }

private:
    __aicore__ inline void MakeShard(uint32_t total)
    {
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t span = (total + blockNum - 1) / blockNum;
        begin_ = AscendC::GetBlockIdx() * span;
        if (begin_ >= total) {
            length_ = 0;
        } else {
            uint32_t remain = total - begin_;
            length_ = remain < span ? remain : span;
        }
    }

    __aicore__ inline void Load(uint32_t offset, uint32_t len)
    {
        AscendC::LocalTensor<DT_X> xLocal = xQueue_.AllocTensor<DT_X>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(len * sizeof(DT_X));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(xLocal, xGm_[offset], copyParams, padParams_);
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void Clamp(uint32_t len)
    {
        AscendC::LocalTensor<DT_X> xLocal = xQueue_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = yQueue_.AllocTensor<DT_X>();
        AscendC::Mins(yLocal, xLocal, maxValue_, len);
        AscendC::Maxs(yLocal, yLocal, minValue_, len);
        xQueue_.FreeTensor(xLocal);
        yQueue_.EnQue(yLocal);
    }

    __aicore__ inline void Store(uint32_t offset, uint32_t len)
    {
        AscendC::LocalTensor<DT_X> yLocal = yQueue_.DeQue<DT_X>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(len * sizeof(DT_X));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(yGm_[offset], yLocal, copyParams);
        yQueue_.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, kQueueDepth> xQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kQueueDepth> yQueue_;
    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;
    AscendC::DataCopyPadExtParams<DT_X> padParams_;
    uint32_t begin_;
    uint32_t length_;
    DT_X minValue_;
    DT_X maxValue_;
};

template <typename DT_X>
__global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ClipByValueTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tilingData, tiling);
    ClipByValueKernel<DT_X> op;
    op.Init(x, y, tilingData);
    op.Process();
}
