#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

template <class DT_X>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}

    __aicore__ inline void Init(TPipe *pipe, GM_ADDR x, GM_ADDR y, uint32_t length,
                                uint32_t blockLength, uint32_t tileLength) {
        pipe_ = pipe;
        length_ = length;
        tileLength_ = tileLength;
        if (tileLength_ == 0) {
            tileLength_ = DEFAULT_TILE_LENGTH;
        }
        blockIdx_ = GetBlockIdx();

        start_ = blockIdx_ * blockLength;
        if (start_ >= length_) {
            blockLength_ = 0;
        } else {
            uint32_t remain = length_ - start_;
            blockLength_ = remain < blockLength ? remain : blockLength;
        }

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x + start_, blockLength_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y + start_, blockLength_);

        pipe_->InitBuffer(inQueue_, 1, tileLength_ * sizeof(float));
        pipe_->InitBuffer(outQueue_, 1, tileLength_ * sizeof(float));
        pipe_->InitBuffer(zBuf_, tileLength_ * sizeof(float));
        pipe_->InitBuffer(qBuf_, tileLength_ * sizeof(float));
    }

    __aicore__ inline void Process() {
        if (blockLength_ == 0) {
            return;
        }

        for (uint32_t offset = 0; offset < blockLength_; offset += tileLength_) {
            uint32_t count = blockLength_ - offset;
            if (count > tileLength_) {
                count = tileLength_;
            }
            ComputeTile(offset, count);
        }
    }

private:
    __aicore__ inline void ComputeTile(uint32_t offset, uint32_t count) {
        LocalTensor<float> xLocal = inQueue_.AllocTensor<float>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, 0, 0.0f};
        bool alignedCopy = (((start_ + offset) & 7u) == 0) && ((count & 7u) == 0);
        if (alignedCopy) {
            DataCopy(xLocal, xGm_[offset], count);
        } else {
            DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        }
        inQueue_.EnQue(xLocal);

        xLocal = inQueue_.DeQue<float>();
        LocalTensor<float> pLocal = outQueue_.AllocTensor<float>();
        LocalTensor<float> zLocal = zBuf_.Get<float>();
        LocalTensor<float> qLocal = qBuf_.Get<float>();

        Mins(xLocal, xLocal, 2.98f, count);
        Maxs(xLocal, xLocal, -2.98f, count);

        Mul(zLocal, xLocal, xLocal, count);

        // erf(x) ~= x * P(x^2) / Q(x^2), fitted for float32 error below 1e-4.
        Muls(pLocal, zLocal, 2.98847016e-02f, count);
        Adds(pLocal, pLocal, 1.95059657e-01f, count);
        Mul(pLocal, pLocal, zLocal, count);
        Adds(pLocal, pLocal, 1.12837803f, count);

        Muls(qLocal, zLocal, 3.41858412e-03f, count);
        Adds(qLocal, qLocal, 9.72173586e-02f, count);
        Mul(qLocal, qLocal, zLocal, count);
        Adds(qLocal, qLocal, 5.05401254e-01f, count);
        Mul(qLocal, qLocal, zLocal, count);
        Adds(qLocal, qLocal, 1.0f, count);

        Div(zLocal, pLocal, qLocal, count);
        Mul(pLocal, zLocal, xLocal, count);

        outQueue_.EnQue(pLocal);
        inQueue_.FreeTensor(xLocal);

        pLocal = outQueue_.DeQue<float>();
        if (alignedCopy) {
            DataCopy(yGm_[offset], pLocal, count);
        } else {
            DataCopyPad(yGm_[offset], pLocal, copyParams);
        }
        outQueue_.FreeTensor(pLocal);
    }

private:
    static constexpr uint32_t DEFAULT_TILE_LENGTH = 16384;

    TPipe *pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TBuf<QuePosition::VECCALC> zBuf_;
    TBuf<QuePosition::VECCALC> qBuf_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t length_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t start_ = 0;
    uint32_t blockLength_ = 0;
    uint32_t tileLength_ = DEFAULT_TILE_LENGTH;
};

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    TPipe pipe;
    pipe.Init();
    KernelErf<DT_X> op;
    op.Init(&pipe, x, y, tiling_data.length, tiling_data.blockLength, tiling_data.tileLength);
    op.Process();
    pipe.Destroy();
}
