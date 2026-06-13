// PRC 2026-05-11 23:25:00 git commit: 410a3e8
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

namespace {
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t MAX_TILE_LENGTH = 16384;
constexpr uint32_t ALIGN_ELEM = 8;

constexpr float POLY_C8 = 4.311269564e-08f;
constexpr float POLY_C7 = -2.028754125e-06f;
constexpr float POLY_C6 = 4.227407408e-05f;
constexpr float POLY_C5 = -5.20253845e-04f;
constexpr float POLY_C4 = 4.274798092e-03f;
constexpr float POLY_C3 = -2.519673854e-02f;
constexpr float POLY_C2 = 1.111955419e-01f;
constexpr float POLY_C1 = -3.753775954e-01f;
constexpr float POLY_C0 = 1.128277898f;
}  // namespace

template <class DT_X>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t length, uint64_t blockLength, uint32_t tileLength) {
        tileLength_ = tileLength > MAX_TILE_LENGTH ? MAX_TILE_LENGTH : tileLength;
        tileLength_ = tileLength_ < ALIGN_ELEM ? ALIGN_ELEM : tileLength_;
        tileLength_ = tileLength_ / ALIGN_ELEM * ALIGN_ELEM;

        const uint64_t startOffset = static_cast<uint64_t>(GetBlockIdx()) * blockLength;
        if (startOffset >= length) {
            currentLength_ = 0;
            return;
        }
        const uint64_t remain = length - startOffset;
        currentLength_ = remain < blockLength ? remain : blockLength;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + startOffset, currentLength_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + startOffset, currentLength_);

        pipe_.InitBuffer(inQueue_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(outQueue_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(tmp1Buf_, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(tmp2Buf_, tileLength_ * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (currentLength_ == 0) return;

        uint64_t loadOffset = 0;
        uint64_t computeOffset = 0;
        uint32_t currentValid = currentLength_ < tileLength_ ? static_cast<uint32_t>(currentLength_) : tileLength_;
        CopyIn(loadOffset, currentValid);
        loadOffset += currentValid;

        while (loadOffset < currentLength_) {
            const uint64_t remain = currentLength_ - loadOffset;
            const uint32_t nextValid = remain < tileLength_ ? static_cast<uint32_t>(remain) : tileLength_;
            CopyIn(loadOffset, nextValid);
            Compute(currentValid);
            CopyOut(computeOffset, currentValid);
            computeOffset += currentValid;
            loadOffset += nextValid;
            currentValid = nextValid;
        }

        Compute(currentValid);
        CopyOut(computeOffset, currentValid);
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t valid) {
        LocalTensor<DT_X> xLocal = inQueue_.template AllocTensor<DT_X>();
        if (valid == tileLength_ || (valid & (ALIGN_ELEM - 1)) == 0) {
            DataCopy(xLocal, xGm_[offset], valid);
        } else {
            DataCopyExtParams params{1, static_cast<uint32_t>(valid * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> pad{false, 0, 0, static_cast<DT_X>(0)};
            DataCopyPad(xLocal, xGm_[offset], params, pad);
        }
        inQueue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t valid) {
        LocalTensor<DT_X> xLocal = inQueue_.template DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue_.template AllocTensor<DT_X>();
        LocalTensor<DT_X> tmp1 = tmp1Buf_.template Get<DT_X>();
        LocalTensor<DT_X> tmp2 = tmp2Buf_.template Get<DT_X>();

        Mins(yLocal, xLocal, static_cast<DT_X>(3.0f), valid);
        Maxs(yLocal, yLocal, static_cast<DT_X>(-3.0f), valid);

        Mul(tmp1, yLocal, yLocal, valid);

        Muls(tmp2, tmp1, POLY_C8, valid);
        Adds(tmp2, tmp2, POLY_C7, valid);
        Mul(tmp2, tmp2, tmp1, valid);
        Adds(tmp2, tmp2, POLY_C6, valid);
        Mul(tmp2, tmp2, tmp1, valid);
        Adds(tmp2, tmp2, POLY_C5, valid);
        Mul(tmp2, tmp2, tmp1, valid);
        Adds(tmp2, tmp2, POLY_C4, valid);
        Mul(tmp2, tmp2, tmp1, valid);
        Adds(tmp2, tmp2, POLY_C3, valid);
        Mul(tmp2, tmp2, tmp1, valid);
        Adds(tmp2, tmp2, POLY_C2, valid);
        Mul(tmp2, tmp2, tmp1, valid);
        Adds(tmp2, tmp2, POLY_C1, valid);
        Mul(tmp2, tmp2, tmp1, valid);
        Adds(tmp2, tmp2, POLY_C0, valid);
        Mul(yLocal, tmp2, yLocal, valid);
        Mins(yLocal, yLocal, static_cast<DT_X>(1.0f), valid);
        Maxs(yLocal, yLocal, static_cast<DT_X>(-1.0f), valid);

        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t valid) {
        LocalTensor<DT_X> yLocal = outQueue_.template DeQue<DT_X>();
        if (valid == tileLength_ || (valid & (ALIGN_ELEM - 1)) == 0) {
            DataCopy(yGm_[offset], yLocal, valid);
        } else {
            DataCopyExtParams params{1, static_cast<uint32_t>(valid * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(yGm_[offset], yLocal, params);
        }
        outQueue_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    TBuf<QuePosition::VECCALC> tmp1Buf_;
    TBuf<QuePosition::VECCALC> tmp2Buf_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint64_t currentLength_ = 0;
    uint32_t tileLength_ = MAX_TILE_LENGTH;
};

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    KernelErf<DT_X> op;
    op.Init(x, y, tiling_data.length, tiling_data.blockLength, tiling_data.tileLength);
    op.Process();
}
