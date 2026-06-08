#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;

template <class T>
class KernelErf {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ErfTilingData &tiling) {
        length_ = tiling.length;
        unitBase_ = tiling.unitBase;
        unitRemainder_ = tiling.unitRemainder;
        unitSize_ = tiling.unitSize;
        tileLength_ = tiling.tileLength;
        mode_ = tiling.mode;

        uint32_t blockIdx = GetBlockIdx();
        uint32_t unitStart = blockIdx * unitBase_;
        uint32_t unitCount = unitBase_;

        if (blockIdx < unitRemainder_) {
            unitStart += blockIdx;
            unitCount += 1U;
        } else {
            unitStart += unitRemainder_;
        }

        start_ = unitStart * unitSize_;
        count_ = 0U;

        if (start_ < length_) {
            uint32_t maxCount = unitCount * unitSize_;
            count_ = length_ - start_;
            if (count_ > maxCount) {
                count_ = maxCount;
            }
        }

        hasTail_ = ((count_ & 7U) != 0U);

        xGm_.SetGlobalBuffer((__gm__ T *)x + start_, count_);
        yGm_.SetGlobalBuffer((__gm__ T *)y + start_, count_);

        pipe_.InitBuffer(inQue_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(outQue_, BUFFER_NUM, tileLength_ * sizeof(T));

        if (hasTail_) {
            pipe_.InitBuffer(tailIn_, 8U * sizeof(T));
            pipe_.InitBuffer(tailOut_, 8U * sizeof(T));
        }

        if (mode_ != 0U) {
            pipe_.InitBuffer(tmpSquare_, tileLength_ * sizeof(T));
            pipe_.InitBuffer(tmpPoly_, tileLength_ * sizeof(T));
            pipe_.InitBuffer(tmpWork_, tileLength_ * sizeof(T));
        }
    }

    __aicore__ inline void Process() {
        uint32_t mainCount = count_ & 0xfffffff8U;

        if (mainCount > 0U && mainCount == count_ && mainCount <= tileLength_) {
            CopyIn(0U, mainCount);
            Compute(mainCount);
            CopyOut(0U, mainCount);
            return;
        }

        for (uint32_t offset = 0U; offset < mainCount;) {
            uint32_t cur = mainCount - offset;
            if (cur > tileLength_) {
                cur = tileLength_;
            }

            CopyIn(offset, cur);
            Compute(cur);
            CopyOut(offset, cur);

            offset += cur;
        }

        if (hasTail_) {
            ProcessTail(mainCount, count_ - mainCount);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t len) {
        LocalTensor<T> x = inQue_.AllocTensor<T>();
        DataCopy(x, xGm_[offset], len);
        inQue_.EnQue(x);
    }

    __aicore__ inline void HornerStep(LocalTensor<T> dst, LocalTensor<T> work,
                                      LocalTensor<T> square, float coeff, uint32_t len) {
        Mul(work, dst, square, len);
        Adds(dst, work, static_cast<T>(coeff), len);
    }

    __aicore__ inline void ErfPoly(LocalTensor<T> y, LocalTensor<T> x, uint32_t len) {
        LocalTensor<T> square = tmpSquare_.Get<T>();
        LocalTensor<T> poly = tmpPoly_.Get<T>();
        LocalTensor<T> work = tmpWork_.Get<T>();

        Mins(work, x, static_cast<T>(3.0f), len);
        Maxs(y, work, static_cast<T>(-3.0f), len);

        Mul(square, y, y, len);

        Duplicate(poly, static_cast<T>(4.75476768e-8f), len);
        HornerStep(poly, work, square, -2.20115551e-6f, len);
        HornerStep(poly, work, square, 4.50525983e-5f, len);
        HornerStep(poly, work, square, -5.44241068e-4f, len);
        HornerStep(poly, work, square, 4.39434612e-3f, len);
        HornerStep(poly, work, square, -2.55416398e-2f, len);
        HornerStep(poly, work, square, 1.11740555e-1f, len);
        HornerStep(poly, work, square, -3.75786969e-1f, len);
        HornerStep(poly, work, square, 1.12837917f, len);

        Mul(y, y, poly, len);

        Mins(work, y, static_cast<T>(1.0f), len);
        Maxs(y, work, static_cast<T>(-1.0f), len);
    }

    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<T> x = inQue_.DeQue<T>();
        LocalTensor<T> y = outQue_.AllocTensor<T>();

        if (mode_ == 0U) {
            Erf<T, true>(y, x, len);
        } else {
            ErfPoly(y, x, len);
        }

        outQue_.EnQue(y);
        inQue_.FreeTensor(x);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t len) {
        LocalTensor<T> y = outQue_.DeQue<T>();
        DataCopy(yGm_[offset], y, len);
        outQue_.FreeTensor(y);
    }

    __aicore__ inline void ProcessTail(uint32_t offset, uint32_t tail) {
        LocalTensor<T> x = tailIn_.Get<T>();
        LocalTensor<T> y = tailOut_.Get<T>();

        for (uint32_t i = 0U; i < 8U; ++i) {
            x.SetValue(i, static_cast<T>(0.0f));
        }

        for (uint32_t i = 0U; i < tail; ++i) {
            x.SetValue(i, xGm_.GetValue(offset + i));
        }

        if (mode_ == 0U) {
            Erf<T, true>(y, x, 8U);
        } else {
            ErfPoly(y, x, 8U);
        }

        for (uint32_t i = 0U; i < tail; ++i) {
            yGm_.SetValue(offset + i, y.GetValue(i));
        }
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue_;

    TBuf<QuePosition::VECCALC> tailIn_;
    TBuf<QuePosition::VECCALC> tailOut_;
    TBuf<QuePosition::VECCALC> tmpSquare_;
    TBuf<QuePosition::VECCALC> tmpPoly_;
    TBuf<QuePosition::VECCALC> tmpWork_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    uint32_t length_;
    uint32_t unitBase_;
    uint32_t unitRemainder_;
    uint32_t unitSize_;
    uint32_t tileLength_;
    uint32_t mode_;
    uint32_t start_;
    uint32_t count_;
    bool hasTail_;
};

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tilingData, tiling);

    KernelErf<DT_X> op;
    op.Init(x, y, tilingData);
    op.Process();
}