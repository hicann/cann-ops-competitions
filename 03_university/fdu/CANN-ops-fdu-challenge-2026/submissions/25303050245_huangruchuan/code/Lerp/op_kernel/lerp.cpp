#include "kernel_operator.h"
#include "lerp_tiling.h"
#include "tiling_key_lerp.h"

namespace {
constexpr uint32_t kTileElems = 4096;
constexpr uint32_t kQueueDepth = 2;
}

template <class DT>
class LerpVectorKernel {
public:
    __aicore__ inline LerpVectorKernel() {}

    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR y, const LerpTilingData &tiling)
    {
        SplitWork(tiling.length);
        weight_ = static_cast<DT>(tiling.weight);

        startGm_.SetGlobalBuffer((__gm__ DT *)start + gmOffset_);
        endGm_.SetGlobalBuffer((__gm__ DT *)end + gmOffset_);
        yGm_.SetGlobalBuffer((__gm__ DT *)y + gmOffset_);

        pipe_.InitBuffer(startQ_, kQueueDepth, tileLen_ * sizeof(DT));
        pipe_.InitBuffer(endQ_, kQueueDepth, tileLen_ * sizeof(DT));
        pipe_.InitBuffer(outQ_, kQueueDepth, tileLen_ * sizeof(DT));
        pipe_.InitBuffer(tmpBuf_, tileLen_ * sizeof(DT));
        padParams_.isPad = false;
        padParams_.leftPadding = 0;
        padParams_.rightPadding = 0;
    }

    __aicore__ inline void Process()
    {
        if (length_ == 0) {
            return;
        }
        for (uint32_t offset = 0; offset < length_; offset += tileLen_) {
            uint32_t elems = (offset + tileLen_ <= length_) ? tileLen_ : (length_ - offset);
            Load(offset, elems);
            Formula(elems);
            Store(offset, elems);
        }
    }

private:
    __aicore__ inline void SplitWork(uint32_t totalLength)
    {
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockSpan = (totalLength + blockNum - 1) / blockNum;
        gmOffset_ = AscendC::GetBlockIdx() * blockSpan;
        if (gmOffset_ >= totalLength) {
            length_ = 0;
        } else {
            uint32_t tail = totalLength - gmOffset_;
            length_ = tail < blockSpan ? tail : blockSpan;
        }
        tileLen_ = length_ == 0 ? 1 : (length_ < kTileElems ? length_ : kTileElems);
    }

    __aicore__ inline void Load(uint32_t offset, uint32_t len)
    {
        AscendC::LocalTensor<DT> startLocal = startQ_.AllocTensor<DT>();
        AscendC::LocalTensor<DT> endLocal = endQ_.AllocTensor<DT>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(len * sizeof(DT));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(startLocal, startGm_[offset], copyParams, padParams_);
        AscendC::DataCopyPad(endLocal, endGm_[offset], copyParams, padParams_);
        startQ_.EnQue(startLocal);
        endQ_.EnQue(endLocal);
    }

    __aicore__ inline void Formula(uint32_t len)
    {
        AscendC::LocalTensor<DT> startLocal = startQ_.DeQue<DT>();
        AscendC::LocalTensor<DT> endLocal = endQ_.DeQue<DT>();
        AscendC::LocalTensor<DT> outLocal = outQ_.AllocTensor<DT>();
        AscendC::LocalTensor<DT> tmp = tmpBuf_.Get<DT>();
        AscendC::Sub(tmp, endLocal, startLocal, len);
        AscendC::Muls(tmp, tmp, weight_, len);
        AscendC::Add(outLocal, startLocal, tmp, len);
        outQ_.EnQue(outLocal);
        startQ_.FreeTensor(startLocal);
        endQ_.FreeTensor(endLocal);
    }

    __aicore__ inline void Store(uint32_t offset, uint32_t len)
    {
        AscendC::LocalTensor<DT> outLocal = outQ_.DeQue<DT>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(len * sizeof(DT));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(yGm_[offset], outLocal, copyParams);
        outQ_.FreeTensor(outLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, kQueueDepth> startQ_;
    AscendC::TQue<AscendC::QuePosition::VECIN, kQueueDepth> endQ_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kQueueDepth> outQ_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_;
    AscendC::GlobalTensor<DT> startGm_;
    AscendC::GlobalTensor<DT> endGm_;
    AscendC::GlobalTensor<DT> yGm_;
    AscendC::DataCopyPadExtParams<DT> padParams_;
    uint32_t gmOffset_;
    uint32_t length_;
    uint32_t tileLen_;
    DT weight_;
};

template <typename DT>
__global__ __aicore__ void lerp(GM_ADDR start, GM_ADDR end, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LerpTilingData);
    GET_TILING_DATA_WITH_STRUCT(LerpTilingData, tilingData, tiling);
    LerpVectorKernel<DT> op;
    op.Init(start, end, y, tilingData);
    op.Process();
}
