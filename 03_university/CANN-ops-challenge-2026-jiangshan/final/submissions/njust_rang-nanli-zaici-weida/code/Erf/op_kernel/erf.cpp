// Kernel implementation.
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

namespace {
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t FLOATS_PER_BLOCK = BLOCK_BYTES / sizeof(float);
constexpr uint32_t MAX_TILE_LENGTH = 8192;
constexpr uint32_t LOCAL_SCALAR_THRESHOLD = 16;
}  // namespace

template <class DT_X>
class KernelErfSmall {
public:
    __aicore__ inline KernelErfSmall() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, uint32_t tileLength, uint32_t smallMode = 0)
    {
        smallMode_ = smallMode;
        length_ = length;
        tileLength_ = NormalizeTileLength(tileLength);
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, length);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, length);
        pipe_.InitBuffer(inQueue_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(outQueue_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (length_ == 0 || GetBlockIdx() != 0) {
            return;
        }
        uint32_t calCount = length_ < tileLength_ ? length_ : tileLength_;
        CopyIn(calCount);
        Compute(calCount);
        CopyOut(calCount);
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const
    {
        return (value + align - 1) / align * align;
    }

    __aicore__ inline uint32_t NormalizeTileLength(uint32_t tileLength) const
    {
        tileLength = tileLength / FLOATS_PER_BLOCK * FLOATS_PER_BLOCK;
        return tileLength == 0 ? FLOATS_PER_BLOCK : tileLength;
    }

    __aicore__ inline void CopyIn(uint32_t calCount)
    {
        LocalTensor<DT_X> xLocal = inQueue_.AllocTensor<DT_X>();
        if ((calCount % FLOATS_PER_BLOCK) == 0) {
            DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = calCount / FLOATS_PER_BLOCK;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopy(xLocal, xGm_, copyParams);
            inQueue_.EnQue(xLocal);
            return;
        }

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = calCount * sizeof(DT_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        DataCopyPadExtParams<DT_X> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = AlignUp(calCount, FLOATS_PER_BLOCK) - calCount;
        padParams.paddingValue = 0;

        DataCopyPad(xLocal, xGm_, copyParams, padParams);
        inQueue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t calCount)
    {
        LocalTensor<DT_X> xLocal = inQueue_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue_.AllocTensor<DT_X>();
        if (smallMode_ != 0 || calCount <= LOCAL_SCALAR_THRESHOLD) {
            ComputeScalar(yLocal, xLocal, calCount);
        } else {
            Erf<DT_X, false>(yLocal, xLocal, calCount);
        }
        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeScalar(LocalTensor<DT_X> yLocal, LocalTensor<DT_X> xLocal, uint32_t calCount)
    {
        for (uint32_t i = 0; i < calCount; ++i) {
            float x = static_cast<float>(xLocal.GetValue(i));
            float y = smallMode_ == 0 ? ErfApprox(x) : ErfHermiteLut(x);
            yLocal.SetValue(i, static_cast<DT_X>(y));
        }
    }

    __aicore__ inline float ErfApprox(float x) const
    {
        if (x > 3.92f) {
            x = 3.92f;
        }
        if (x < -3.92f) {
            x = -3.92f;
        }

        float x2 = x * x;
        float p = (((((0.053443748819f * x2 + 0.75517016694e1f) * x2 + 0.10162808918e3f) * x2
            + 0.13938061484e4f) * x2 + 0.50637915060e4f) * x2 + 0.29639384698e5f) * x;
        float q = ((((x2 + 0.31212858877e2f) * x2 + 0.39856963806e3f) * x2 + 0.30231248150e4f) * x2
            + 0.13243365831e5f) * x2 + 0.26267224157e5f;
        return p / q;
    }

    __aicore__ inline float ErfHermiteLut(float x) const
    {
        static constexpr float kStep = 0.125f;
        static constexpr float kInvStep = 8.0f;
        static constexpr float kMin = -3.0f;
        static constexpr float kY[49] = {
            -9.999779095e-01f, -9.999521452e-01f, -9.998993781e-01f, -9.997946243e-01f,
            -9.995930480e-01f, -9.992170618e-01f, -9.985372834e-01f, -9.973459706e-01f,
            -9.953222650e-01f, -9.919900577e-01f, -9.866716712e-01f, -9.784437332e-01f,
            -9.661051465e-01f, -9.481700728e-01f, -9.229001283e-01f, -8.883882317e-01f,
            -8.427007929e-01f, -7.840750611e-01f, -7.111556337e-01f, -6.232408822e-01f,
            -5.204998778e-01f, -4.041169094e-01f, -2.763263902e-01f, -1.403162048e-01f,
            0.000000000e+00f, 1.403162048e-01f, 2.763263902e-01f, 4.041169094e-01f,
            5.204998778e-01f, 6.232408822e-01f, 7.111556337e-01f, 7.840750611e-01f,
            8.427007929e-01f, 8.883882317e-01f, 9.229001283e-01f, 9.481700728e-01f,
            9.661051465e-01f, 9.784437332e-01f, 9.866716712e-01f, 9.919900577e-01f,
            9.953222650e-01f, 9.973459706e-01f, 9.985372834e-01f, 9.992170618e-01f,
            9.995930480e-01f, 9.997946243e-01f, 9.998993781e-01f, 9.999521452e-01f,
            9.999779095e-01f
        };
        static constexpr float kD[49] = {
            1.392530519e-04f, 2.902282829e-04f, 5.862772471e-04f, 1.147875126e-03f,
            2.178284230e-03f, 4.006477862e-03f, 7.142319022e-03f, 1.234082061e-02f,
            2.066698535e-02f, 3.354582842e-02f, 5.277499593e-02f, 8.047225902e-02f,
            1.189302892e-01f, 1.703597737e-01f, 2.365211224e-01f, 3.182739585e-01f,
            4.151074974e-01f, 5.247450453e-01f, 6.429310692e-01f, 7.634995358e-01f,
            8.787825789e-01f, 9.803528095e-01f, 1.060014129e+00f, 1.110885270e+00f,
            1.128379167e+00f, 1.110885270e+00f, 1.060014129e+00f, 9.803528095e-01f,
            8.787825789e-01f, 7.634995358e-01f, 6.429310692e-01f, 5.247450453e-01f,
            4.151074974e-01f, 3.182739585e-01f, 2.365211224e-01f, 1.703597737e-01f,
            1.189302892e-01f, 8.047225902e-02f, 5.277499593e-02f, 3.354582842e-02f,
            2.066698535e-02f, 1.234082061e-02f, 7.142319022e-03f, 4.006477862e-03f,
            2.178284230e-03f, 1.147875126e-03f, 5.862772471e-04f, 2.902282829e-04f,
            1.392530519e-04f
        };
        if (x <= kMin) {
            return -1.0f;
        }
        if (x >= -kMin) {
            return 1.0f;
        }
        float pos = (x - kMin) * kInvStep;
        int idx = static_cast<int>(pos);
        if (idx < 0) {
            idx = 0;
        } else if (idx > 47) {
            idx = 47;
        }
        float t = pos - static_cast<float>(idx);
        float t2 = t * t;
        float t3 = t2 * t;
        float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
        float h10 = t3 - 2.0f * t2 + t;
        float h01 = -2.0f * t3 + 3.0f * t2;
        float h11 = t3 - t2;
        return h00 * kY[idx] + h10 * kStep * kD[idx] + h01 * kY[idx + 1] + h11 * kStep * kD[idx + 1];
    }

    __aicore__ inline void CopyOut(uint32_t calCount)
    {
        LocalTensor<DT_X> yLocal = outQueue_.DeQue<DT_X>();
        if ((calCount % FLOATS_PER_BLOCK) == 0) {
            DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = calCount / FLOATS_PER_BLOCK;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopy(yGm_, yLocal, copyParams);
            outQueue_.FreeTensor(yLocal);
            return;
        }

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = calCount * sizeof(DT_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        DataCopyPad(yGm_, yLocal, copyParams);
        outQueue_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueue_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t length_ = 0;
    uint32_t tileLength_ = FLOATS_PER_BLOCK;
    uint32_t smallMode_ = 0;
};

template <class DT_X>
class KernelErfSmallMultiBlock {
public:
    __aicore__ inline KernelErfSmallMultiBlock() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, uint32_t tileLength, uint32_t smallMode = 0)
    {
        smallMode_ = smallMode;
        length_ = length;
        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        uint32_t per = blockNum == 0 ? length : (length + blockNum - 1) / blockNum;
        start_ = blockIdx * per;
        if (start_ >= length) {
            calCount_ = 0;
        } else {
            uint32_t remain = length - start_;
            calCount_ = remain < per ? remain : per;
        }
        tileLength_ = NormalizeTileLength(tileLength);
        if (tileLength_ < calCount_) {
            tileLength_ = AlignUp(calCount_, FLOATS_PER_BLOCK);
        }
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, length);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, length);
        pipe_.InitBuffer(inQueue_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(outQueue_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (calCount_ == 0) {
            return;
        }
        CopyIn(calCount_);
        Compute(calCount_);
        CopyOut(calCount_);
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const
    {
        return (value + align - 1) / align * align;
    }

    __aicore__ inline uint32_t NormalizeTileLength(uint32_t tileLength) const
    {
        tileLength = tileLength / FLOATS_PER_BLOCK * FLOATS_PER_BLOCK;
        return tileLength == 0 ? FLOATS_PER_BLOCK : tileLength;
    }

    __aicore__ inline void CopyIn(uint32_t calCount)
    {
        LocalTensor<DT_X> xLocal = inQueue_.AllocTensor<DT_X>();
        if ((calCount % FLOATS_PER_BLOCK) == 0) {
            DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = calCount / FLOATS_PER_BLOCK;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopy(xLocal, xGm_[start_], copyParams);
            inQueue_.EnQue(xLocal);
            return;
        }

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = calCount * sizeof(DT_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        DataCopyPadExtParams<DT_X> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = AlignUp(calCount, FLOATS_PER_BLOCK) - calCount;
        padParams.paddingValue = 0;

        DataCopyPad(xLocal, xGm_[start_], copyParams, padParams);
        inQueue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t calCount)
    {
        LocalTensor<DT_X> xLocal = inQueue_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue_.AllocTensor<DT_X>();
        if (smallMode_ != 0 || calCount <= LOCAL_SCALAR_THRESHOLD) {
            ComputeScalar(yLocal, xLocal, calCount);
        } else {
            Erf<DT_X, false>(yLocal, xLocal, calCount);
        }
        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeScalar(LocalTensor<DT_X> yLocal, LocalTensor<DT_X> xLocal, uint32_t calCount)
    {
        for (uint32_t i = 0; i < calCount; ++i) {
            float x = static_cast<float>(xLocal.GetValue(i));
            float y = smallMode_ == 0 ? ErfApprox(x) : ErfHermiteLut(x);
            yLocal.SetValue(i, static_cast<DT_X>(y));
        }
    }

    __aicore__ inline float ErfApprox(float x) const
    {
        if (x > 3.92f) {
            x = 3.92f;
        }
        if (x < -3.92f) {
            x = -3.92f;
        }

        float x2 = x * x;
        float p = (((((0.053443748819f * x2 + 0.75517016694e1f) * x2 + 0.10162808918e3f) * x2
            + 0.13938061484e4f) * x2 + 0.50637915060e4f) * x2 + 0.29639384698e5f) * x;
        float q = ((((x2 + 0.31212858877e2f) * x2 + 0.39856963806e3f) * x2 + 0.30231248150e4f) * x2
            + 0.13243365831e5f) * x2 + 0.26267224157e5f;
        return p / q;
    }

    __aicore__ inline float ErfHermiteLut(float x) const
    {
        return ErfApprox(x);
    }

    __aicore__ inline void CopyOut(uint32_t calCount)
    {
        LocalTensor<DT_X> yLocal = outQueue_.DeQue<DT_X>();
        if ((calCount % FLOATS_PER_BLOCK) == 0) {
            DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = calCount / FLOATS_PER_BLOCK;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopy(yGm_[start_], yLocal, copyParams);
            outQueue_.FreeTensor(yLocal);
            return;
        }

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = calCount * sizeof(DT_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        DataCopyPad(yGm_[start_], yLocal, copyParams);
        outQueue_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueue_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t start_ = 0;
    uint32_t length_ = 0;
    uint32_t calCount_ = 0;
    uint32_t tileLength_ = FLOATS_PER_BLOCK;
    uint32_t smallMode_ = 0;
};

template <class DT_X>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, uint32_t tileLength, uint32_t formulaMode)
    {
        formulaMode_ = formulaMode;
        tileLength_ = NormalizeTileLength(tileLength);
        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        uint32_t totalTiles = (length + tileLength_ - 1) / tileLength_;
        uint32_t tilesPerCore = (totalTiles + blockNum - 1) / blockNum;
        uint32_t startTile = blockIdx * tilesPerCore;
        uint32_t endTile = startTile + tilesPerCore;
        if (endTile > totalTiles) {
            endTile = totalTiles;
        }

        start_ = startTile * tileLength_;
        if (start_ >= length) {
            length_ = 0;
        } else {
            uint32_t remain = length - start_;
            uint32_t perCore = (endTile > startTile) ? (endTile - startTile) * tileLength_ : 0;
            length_ = remain < perCore ? remain : perCore;
        }

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, length);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, length);

        pipe_.InitBuffer(inQueue_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(outQueue_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(tmpBuffer_, tileLength_ * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        uint32_t offset = 0;
        while (offset < length_) {
            uint32_t calCount = length_ - offset;
            if (calCount > tileLength_) {
                calCount = tileLength_;
            }
            CopyIn(offset, calCount);
            Compute(calCount);
            CopyOut(offset, calCount);
            offset += calCount;
        }
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const
    {
        return (value + align - 1) / align * align;
    }

    __aicore__ inline uint32_t NormalizeTileLength(uint32_t tileLength) const
    {
        if (tileLength == 0) {
            tileLength = FLOATS_PER_BLOCK;
        } else if (tileLength > MAX_TILE_LENGTH) {
            tileLength = MAX_TILE_LENGTH;
        }
        tileLength = tileLength / FLOATS_PER_BLOCK * FLOATS_PER_BLOCK;
        if (tileLength == 0) {
            tileLength = FLOATS_PER_BLOCK;
        }
        return tileLength;
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t calCount)
    {
        LocalTensor<DT_X> xLocal = inQueue_.AllocTensor<DT_X>();
        if ((calCount % FLOATS_PER_BLOCK) == 0) {
            DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = calCount / FLOATS_PER_BLOCK;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopy(xLocal, xGm_[start_ + offset], copyParams);
            inQueue_.EnQue(xLocal);
            return;
        }

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = calCount * sizeof(DT_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        DataCopyPadExtParams<DT_X> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = AlignUp(calCount, FLOATS_PER_BLOCK) - calCount;
        padParams.paddingValue = 0;

        DataCopyPad(xLocal, xGm_[start_ + offset], copyParams, padParams);
        inQueue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t calCount)
    {
        LocalTensor<DT_X> xLocal = inQueue_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue_.AllocTensor<DT_X>();

        if (false && calCount <= LOCAL_SCALAR_THRESHOLD) {
            ComputeScalar(yLocal, xLocal, calCount);
        } else {
            LocalTensor<DT_X> tmpLocal = tmpBuffer_.Get<DT_X>(tileLength_);
            if (formulaMode_ == 0) {
                ComputePoly(yLocal, xLocal, tmpLocal, calCount);
            } else {
                ComputePoly11(yLocal, xLocal, tmpLocal, calCount);
            }
        }

        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeScalar(LocalTensor<DT_X> yLocal, LocalTensor<DT_X> xLocal, uint32_t calCount)
    {
        for (uint32_t i = 0; i < calCount; ++i) {
            float x = static_cast<float>(xLocal.GetValue(i));
            yLocal.SetValue(i, static_cast<DT_X>(ErfApprox(x)));
        }
    }

    __aicore__ inline void ComputePoly(LocalTensor<DT_X> yLocal, LocalTensor<DT_X> xLocal,
        LocalTensor<DT_X> tmpLocal, uint32_t calCount)
    {
        UnaryRepeatParams unaryParams;
        BinaryRepeatParams binaryParams;

        SetMaskCount();
        SetVectorMask<half, MaskMode::COUNTER>(0, calCount);

        Mins<DT_X, false>(tmpLocal, xLocal, static_cast<DT_X>(2.8f), MASK_PLACEHOLDER, 1, unaryParams);
        Maxs<DT_X, false>(tmpLocal, tmpLocal, static_cast<DT_X>(-2.8f), MASK_PLACEHOLDER, 1, unaryParams);

        Mul<DT_X, false>(xLocal, tmpLocal, tmpLocal, MASK_PLACEHOLDER, 1, binaryParams);

        Duplicate<DT_X, false>(yLocal, static_cast<DT_X>(6.05204282e-06f), MASK_PLACEHOLDER, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(-1.95508637e-04f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(2.69135088e-03f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(-2.10294184e-02f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(1.05788711e-01f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(-3.72682427e-01f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(1.12807812e+00f), binaryParams, unaryParams);

        Mul<DT_X, false>(yLocal, yLocal, tmpLocal, MASK_PLACEHOLDER, 1, binaryParams);

        SetMaskNorm();
        ResetMask();
    }

    __aicore__ inline void ComputePoly11(LocalTensor<DT_X> yLocal, LocalTensor<DT_X> xLocal,
        LocalTensor<DT_X> tmpLocal, uint32_t calCount)
    {
        UnaryRepeatParams unaryParams;
        BinaryRepeatParams binaryParams;

        SetMaskCount();
        SetVectorMask<half, MaskMode::COUNTER>(0, calCount);

        Mins<DT_X, false>(tmpLocal, xLocal, static_cast<DT_X>(2.35f), MASK_PLACEHOLDER, 1, unaryParams);
        Maxs<DT_X, false>(tmpLocal, tmpLocal, static_cast<DT_X>(-2.35f), MASK_PLACEHOLDER, 1, unaryParams);

        Mul<DT_X, false>(xLocal, tmpLocal, tmpLocal, MASK_PLACEHOLDER, 1, binaryParams);

        Duplicate<DT_X, false>(yLocal, static_cast<DT_X>(-1.12372850e-04f), MASK_PLACEHOLDER, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(2.31813183e-03f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(-2.05179884e-02f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(1.06151454e-01f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(-3.73640567e-01f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(1.12832048e+00f), binaryParams, unaryParams);

        Mul<DT_X, false>(yLocal, yLocal, tmpLocal, MASK_PLACEHOLDER, 1, binaryParams);

        SetMaskNorm();
        ResetMask();
    }

    __aicore__ inline void HornerStep(LocalTensor<DT_X> yLocal, LocalTensor<DT_X> x2Local, DT_X coeff,
        BinaryRepeatParams binaryParams, UnaryRepeatParams unaryParams)
    {
        Mul<DT_X, false>(yLocal, yLocal, x2Local, MASK_PLACEHOLDER, 1, binaryParams);
        Adds<DT_X, false>(yLocal, yLocal, coeff, MASK_PLACEHOLDER, 1, unaryParams);
    }

    __aicore__ inline float ErfApprox(float x) const
    {
        if (x > 3.92f) {
            x = 3.92f;
        }
        if (x < -3.92f) {
            x = -3.92f;
        }

        float x2 = x * x;
        float p = (((((0.053443748819f * x2 + 0.75517016694e1f) * x2 + 0.10162808918e3f) * x2
            + 0.13938061484e4f) * x2 + 0.50637915060e4f) * x2 + 0.29639384698e5f) * x;
        float q = ((((x2 + 0.31212858877e2f) * x2 + 0.39856963806e3f) * x2 + 0.30231248150e4f) * x2
            + 0.13243365831e5f) * x2 + 0.26267224157e5f;
        return p / q;
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t calCount)
    {
        LocalTensor<DT_X> yLocal = outQueue_.DeQue<DT_X>();
        if ((calCount % FLOATS_PER_BLOCK) == 0) {
            DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = calCount / FLOATS_PER_BLOCK;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopy(yGm_[start_ + offset], yLocal, copyParams);
            outQueue_.FreeTensor(yLocal);
            return;
        }

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = calCount * sizeof(DT_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        DataCopyPad(yGm_[start_ + offset], yLocal, copyParams);
        outQueue_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuffer_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t start_ = 0;
    uint32_t length_ = 0;
    uint32_t tileLength_ = FLOATS_PER_BLOCK;
    uint32_t formulaMode_ = 0;
};

template <class DT_X>
class KernelErfCD {
public:
    __aicore__ inline KernelErfCD() {}

    __aicore__ inline void Init(TPipe &pipe, GM_ADDR x, GM_ADDR y, uint32_t length, uint32_t tileLength)
    {
        tileLength_ = NormalizeTileLength(tileLength);
        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        uint32_t totalTiles = (length + tileLength_ - 1) / tileLength_;
        uint32_t tilesPerCore = (totalTiles + blockNum - 1) / blockNum;
        uint32_t startTile = blockIdx * tilesPerCore;
        uint32_t endTile = startTile + tilesPerCore;
        if (endTile > totalTiles) {
            endTile = totalTiles;
        }

        start_ = startTile * tileLength_;
        if (start_ >= length) {
            length_ = 0;
        } else {
            uint32_t remain = length - start_;
            uint32_t perCore = (endTile > startTile) ? (endTile - startTile) * tileLength_ : 0;
            length_ = remain < perCore ? remain : perCore;
        }

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, length);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, length);

        pipe.InitBuffer(inQueue_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe.InitBuffer(outQueue_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe.InitBuffer(tmpBuffer_, tileLength_ * sizeof(DT_X));
        pipe.InitBuffer(tmpBuffer2_, tileLength_ * sizeof(DT_X));
        pipe.InitBuffer(tmpBuffer3_, tileLength_ * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        uint32_t offset = 0;
        while (offset < length_) {
            uint32_t calCount = length_ - offset;
            if (calCount > tileLength_) {
                calCount = tileLength_;
            }
            CopyIn(offset, calCount);
            Compute(calCount);
            CopyOut(offset, calCount);
            offset += calCount;
        }
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const
    {
        return (value + align - 1) / align * align;
    }

    __aicore__ inline uint32_t NormalizeTileLength(uint32_t tileLength) const
    {
        if (tileLength == 0) {
            tileLength = FLOATS_PER_BLOCK;
        } else if (tileLength > MAX_TILE_LENGTH) {
            tileLength = MAX_TILE_LENGTH;
        }
        tileLength = tileLength / FLOATS_PER_BLOCK * FLOATS_PER_BLOCK;
        return tileLength == 0 ? FLOATS_PER_BLOCK : tileLength;
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t calCount)
    {
        LocalTensor<DT_X> xLocal = inQueue_.AllocTensor<DT_X>();
        if ((calCount % FLOATS_PER_BLOCK) == 0) {
            DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = calCount / FLOATS_PER_BLOCK;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopy(xLocal, xGm_[start_ + offset], copyParams);
            inQueue_.EnQue(xLocal);
            return;
        }

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = calCount * sizeof(DT_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        DataCopyPadExtParams<DT_X> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = AlignUp(calCount, FLOATS_PER_BLOCK) - calCount;
        padParams.paddingValue = 0;

        DataCopyPad(xLocal, xGm_[start_ + offset], copyParams, padParams);
        inQueue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t calCount)
    {
        LocalTensor<DT_X> xLocal = inQueue_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue_.AllocTensor<DT_X>();
        LocalTensor<DT_X> tmpLocal = tmpBuffer_.Get<DT_X>(tileLength_);
        LocalTensor<DT_X> tmp2Local = tmpBuffer2_.Get<DT_X>(tileLength_);
        LocalTensor<DT_X> tmp3Local = tmpBuffer3_.Get<DT_X>(tileLength_);
        ComputePoly11(yLocal, xLocal, tmpLocal, tmp2Local, tmp3Local, calCount);
        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputePoly(LocalTensor<DT_X> yLocal, LocalTensor<DT_X> xLocal,
        LocalTensor<DT_X> tmpLocal, uint32_t calCount)
    {
        UnaryRepeatParams unaryParams;
        BinaryRepeatParams binaryParams;
        SetMaskCount();
        SetVectorMask<half, MaskMode::COUNTER>(0, calCount);
        Mins<DT_X, false>(tmpLocal, xLocal, static_cast<DT_X>(2.8f), MASK_PLACEHOLDER, 1, unaryParams);
        Maxs<DT_X, false>(tmpLocal, tmpLocal, static_cast<DT_X>(-2.8f), MASK_PLACEHOLDER, 1, unaryParams);
        Mul<DT_X, false>(xLocal, tmpLocal, tmpLocal, MASK_PLACEHOLDER, 1, binaryParams);
        Duplicate<DT_X, false>(yLocal, static_cast<DT_X>(6.05204282e-06f), MASK_PLACEHOLDER, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(-1.95508637e-04f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(2.69135088e-03f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(-2.10294184e-02f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(1.05788711e-01f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(-3.72682427e-01f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(1.12807812e+00f), binaryParams, unaryParams);
        Mul<DT_X, false>(yLocal, yLocal, tmpLocal, MASK_PLACEHOLDER, 1, binaryParams);
        SetMaskNorm();
        ResetMask();
    }

    __aicore__ inline void ComputePoly11(LocalTensor<DT_X> yLocal, LocalTensor<DT_X> xLocal,
        LocalTensor<DT_X> tmpLocal, LocalTensor<DT_X> tmp2Local, LocalTensor<DT_X> tmp3Local, uint32_t calCount)
    {
        UnaryRepeatParams unaryParams;
        BinaryRepeatParams binaryParams;
        SetMaskCount();
        SetVectorMask<half, MaskMode::COUNTER>(0, calCount);
        Mins<DT_X, false>(tmpLocal, xLocal, static_cast<DT_X>(2.35f), MASK_PLACEHOLDER, 1, unaryParams);
        Maxs<DT_X, false>(tmpLocal, tmpLocal, static_cast<DT_X>(-2.35f), MASK_PLACEHOLDER, 1, unaryParams);
        Mul<DT_X, false>(xLocal, tmpLocal, tmpLocal, MASK_PLACEHOLDER, 1, binaryParams);
        Mul<DT_X, false>(tmp2Local, xLocal, xLocal, MASK_PLACEHOLDER, 1, binaryParams);
        Mul<DT_X, false>(tmp3Local, tmp2Local, tmp2Local, MASK_PLACEHOLDER, 1, binaryParams);

        Duplicate<DT_X, false>(yLocal, static_cast<DT_X>(-1.12372850e-04f), MASK_PLACEHOLDER, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
        Mul<DT_X, false>(yLocal, yLocal, xLocal, MASK_PLACEHOLDER, 1, binaryParams);
        Adds<DT_X, false>(yLocal, yLocal, static_cast<DT_X>(2.31813183e-03f), MASK_PLACEHOLDER, 1, unaryParams);
        Mul<DT_X, false>(yLocal, yLocal, tmp3Local, MASK_PLACEHOLDER, 1, binaryParams);

        Duplicate<DT_X, false>(tmp3Local, static_cast<DT_X>(-2.05179884e-02f), MASK_PLACEHOLDER, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
        Mul<DT_X, false>(tmp3Local, tmp3Local, xLocal, MASK_PLACEHOLDER, 1, binaryParams);
        Adds<DT_X, false>(tmp3Local, tmp3Local, static_cast<DT_X>(1.06151454e-01f), MASK_PLACEHOLDER, 1, unaryParams);
        Mul<DT_X, false>(tmp3Local, tmp3Local, tmp2Local, MASK_PLACEHOLDER, 1, binaryParams);
        Add<DT_X, false>(yLocal, yLocal, tmp3Local, MASK_PLACEHOLDER, 1, binaryParams);

        Duplicate<DT_X, false>(tmp2Local, static_cast<DT_X>(-3.73640567e-01f), MASK_PLACEHOLDER, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
        Mul<DT_X, false>(tmp2Local, tmp2Local, xLocal, MASK_PLACEHOLDER, 1, binaryParams);
        Adds<DT_X, false>(tmp2Local, tmp2Local, static_cast<DT_X>(1.12832048e+00f), MASK_PLACEHOLDER, 1, unaryParams);
        Add<DT_X, false>(yLocal, yLocal, tmp2Local, MASK_PLACEHOLDER, 1, binaryParams);

        Mul<DT_X, false>(yLocal, yLocal, tmpLocal, MASK_PLACEHOLDER, 1, binaryParams);
        SetMaskNorm();
        ResetMask();
    }

    __aicore__ inline void HornerStep(LocalTensor<DT_X> yLocal, LocalTensor<DT_X> x2Local, DT_X coeff,
        BinaryRepeatParams binaryParams, UnaryRepeatParams unaryParams)
    {
        Mul<DT_X, false>(yLocal, yLocal, x2Local, MASK_PLACEHOLDER, 1, binaryParams);
        Adds<DT_X, false>(yLocal, yLocal, coeff, MASK_PLACEHOLDER, 1, unaryParams);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t calCount)
    {
        LocalTensor<DT_X> yLocal = outQueue_.DeQue<DT_X>();
        if ((calCount % FLOATS_PER_BLOCK) == 0) {
            DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = calCount / FLOATS_PER_BLOCK;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopy(yGm_[start_ + offset], yLocal, copyParams);
            outQueue_.FreeTensor(yLocal);
            return;
        }

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = calCount * sizeof(DT_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        DataCopyPad(yGm_[start_ + offset], yLocal, copyParams);
        outQueue_.FreeTensor(yLocal);
    }

private:
    TQue<TPosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuffer_;
    TBuf<TPosition::VECCALC> tmpBuffer2_;
    TBuf<TPosition::VECCALC> tmpBuffer3_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t start_ = 0;
    uint32_t length_ = 0;
    uint32_t tileLength_ = FLOATS_PER_BLOCK;
};

template <class DT_X>
class KernelErfABDirect {
public:
    __aicore__ inline KernelErfABDirect() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length)
    {
        length_ = length;
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, length);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, length);
        pipe_.InitBuffer(xBuffer_, 384 * sizeof(DT_X));
        pipe_.InitBuffer(yBuffer_, 384 * sizeof(DT_X));
        pipe_.InitBuffer(tmpBuffer_, 384 * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (length_ == 0 || GetBlockIdx() != 0) {
            return;
        }
        LocalTensor<DT_X> xLocal = xBuffer_.Get<DT_X>(384);
        LocalTensor<DT_X> yLocal = yBuffer_.Get<DT_X>(384);
        LocalTensor<DT_X> tmpLocal = tmpBuffer_.Get<DT_X>(384);
        CopyIn(xLocal, length_);
        event_t eventMte2ToV = static_cast<event_t>(0);
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);
        ComputePoly(yLocal, xLocal, tmpLocal, length_);
        event_t eventVToMte3 = static_cast<event_t>(1);
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
        CopyOut(yLocal, length_);
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const
    {
        return (value + align - 1) / align * align;
    }

    __aicore__ inline void CopyIn(LocalTensor<DT_X> xLocal, uint32_t calCount)
    {
        if ((calCount % FLOATS_PER_BLOCK) == 0) {
            DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = calCount / FLOATS_PER_BLOCK;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopy(xLocal, xGm_, copyParams);
            return;
        }

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = calCount * sizeof(DT_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        DataCopyPadExtParams<DT_X> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = AlignUp(calCount, FLOATS_PER_BLOCK) - calCount;
        padParams.paddingValue = 0;
        DataCopyPad(xLocal, xGm_, copyParams, padParams);
    }

    __aicore__ inline void ComputePoly(LocalTensor<DT_X> yLocal, LocalTensor<DT_X> xLocal,
        LocalTensor<DT_X> tmpLocal, uint32_t calCount)
    {
        UnaryRepeatParams unaryParams;
        BinaryRepeatParams binaryParams;

        SetMaskCount();
        SetVectorMask<half, MaskMode::COUNTER>(0, calCount);

        Mins<DT_X, false>(tmpLocal, xLocal, static_cast<DT_X>(2.8f), MASK_PLACEHOLDER, 1, unaryParams);
        Maxs<DT_X, false>(tmpLocal, tmpLocal, static_cast<DT_X>(-2.8f), MASK_PLACEHOLDER, 1, unaryParams);

        Mul<DT_X, false>(xLocal, tmpLocal, tmpLocal, MASK_PLACEHOLDER, 1, binaryParams);

        Duplicate<DT_X, false>(yLocal, static_cast<DT_X>(6.05204282e-06f), MASK_PLACEHOLDER, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(-1.95508637e-04f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(2.69135088e-03f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(-2.10294184e-02f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(1.05788711e-01f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(-3.72682427e-01f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(1.12807812e+00f), binaryParams, unaryParams);

        Mul<DT_X, false>(yLocal, yLocal, tmpLocal, MASK_PLACEHOLDER, 1, binaryParams);

        SetMaskNorm();
        ResetMask();
    }

    __aicore__ inline void HornerStep(LocalTensor<DT_X> yLocal, LocalTensor<DT_X> x2Local, DT_X coeff,
        BinaryRepeatParams binaryParams, UnaryRepeatParams unaryParams)
    {
        Mul<DT_X, false>(yLocal, yLocal, x2Local, MASK_PLACEHOLDER, 1, binaryParams);
        Adds<DT_X, false>(yLocal, yLocal, coeff, MASK_PLACEHOLDER, 1, unaryParams);
    }

    __aicore__ inline void CopyOut(LocalTensor<DT_X> yLocal, uint32_t calCount)
    {
        if ((calCount % FLOATS_PER_BLOCK) == 0) {
            DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = calCount / FLOATS_PER_BLOCK;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopy(yGm_, yLocal, copyParams);
            return;
        }

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = calCount * sizeof(DT_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        DataCopyPad(yGm_, yLocal, copyParams);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> xBuffer_;
    TBuf<TPosition::VECCALC> yBuffer_;
    TBuf<TPosition::VECCALC> tmpBuffer_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t length_ = 0;
};

template <class DT_X>
class KernelErfABMultiBlock {
public:
    __aicore__ inline KernelErfABMultiBlock() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length)
    {
        length_ = length;
        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        uint32_t per = blockNum == 0 ? length : (length + blockNum - 1) / blockNum;
        start_ = blockIdx * per;
        if (start_ >= length) {
            calCount_ = 0;
        } else {
            uint32_t remain = length - start_;
            calCount_ = remain < per ? remain : per;
        }
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, length);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, length);
        pipe_.InitBuffer(xBuffer_, 384 * sizeof(DT_X));
        pipe_.InitBuffer(yBuffer_, 384 * sizeof(DT_X));
        pipe_.InitBuffer(tmpBuffer_, 384 * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (calCount_ == 0) {
            return;
        }
        LocalTensor<DT_X> xLocal = xBuffer_.Get<DT_X>(384);
        LocalTensor<DT_X> yLocal = yBuffer_.Get<DT_X>(384);
        LocalTensor<DT_X> tmpLocal = tmpBuffer_.Get<DT_X>(384);
        CopyIn(xLocal, calCount_);
        event_t eventMte2ToV = static_cast<event_t>(0);
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);
        ComputePoly(yLocal, xLocal, tmpLocal, calCount_);
        event_t eventVToMte3 = static_cast<event_t>(1);
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
        CopyOut(yLocal, calCount_);
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const
    {
        return (value + align - 1) / align * align;
    }

    __aicore__ inline void CopyIn(LocalTensor<DT_X> xLocal, uint32_t calCount)
    {
        if ((calCount % FLOATS_PER_BLOCK) == 0) {
            DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = calCount / FLOATS_PER_BLOCK;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopy(xLocal, xGm_[start_], copyParams);
            return;
        }

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = calCount * sizeof(DT_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        DataCopyPadExtParams<DT_X> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = AlignUp(calCount, FLOATS_PER_BLOCK) - calCount;
        padParams.paddingValue = 0;
        DataCopyPad(xLocal, xGm_[start_], copyParams, padParams);
    }

    __aicore__ inline void ComputePoly(LocalTensor<DT_X> yLocal, LocalTensor<DT_X> xLocal,
        LocalTensor<DT_X> tmpLocal, uint32_t calCount)
    {
        UnaryRepeatParams unaryParams;
        BinaryRepeatParams binaryParams;

        SetMaskCount();
        SetVectorMask<half, MaskMode::COUNTER>(0, calCount);

        Mins<DT_X, false>(tmpLocal, xLocal, static_cast<DT_X>(2.8f), MASK_PLACEHOLDER, 1, unaryParams);
        Maxs<DT_X, false>(tmpLocal, tmpLocal, static_cast<DT_X>(-2.8f), MASK_PLACEHOLDER, 1, unaryParams);

        Mul<DT_X, false>(xLocal, tmpLocal, tmpLocal, MASK_PLACEHOLDER, 1, binaryParams);

        Duplicate<DT_X, false>(yLocal, static_cast<DT_X>(6.05204282e-06f), MASK_PLACEHOLDER, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(-1.95508637e-04f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(2.69135088e-03f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(-2.10294184e-02f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(1.05788711e-01f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(-3.72682427e-01f), binaryParams, unaryParams);
        HornerStep(yLocal, xLocal, static_cast<DT_X>(1.12807812e+00f), binaryParams, unaryParams);

        Mul<DT_X, false>(yLocal, yLocal, tmpLocal, MASK_PLACEHOLDER, 1, binaryParams);

        SetMaskNorm();
        ResetMask();
    }

    __aicore__ inline void HornerStep(LocalTensor<DT_X> yLocal, LocalTensor<DT_X> x2Local, DT_X coeff,
        BinaryRepeatParams binaryParams, UnaryRepeatParams unaryParams)
    {
        Mul<DT_X, false>(yLocal, yLocal, x2Local, MASK_PLACEHOLDER, 1, binaryParams);
        Adds<DT_X, false>(yLocal, yLocal, coeff, MASK_PLACEHOLDER, 1, unaryParams);
    }

    __aicore__ inline void CopyOut(LocalTensor<DT_X> yLocal, uint32_t calCount)
    {
        if ((calCount % FLOATS_PER_BLOCK) == 0) {
            DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = calCount / FLOATS_PER_BLOCK;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopy(yGm_[start_], yLocal, copyParams);
            return;
        }

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = calCount * sizeof(DT_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        DataCopyPad(yGm_[start_], yLocal, copyParams);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> xBuffer_;
    TBuf<TPosition::VECCALC> yBuffer_;
    TBuf<TPosition::VECCALC> tmpBuffer_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t start_ = 0;
    uint32_t length_ = 0;
    uint32_t calCount_ = 0;
};

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    if (tiling_data.mode == 0 && tiling_data.rank == 1 &&
        (tiling_data.length == 128 ||
         (tiling_data.length >= 369 && tiling_data.length <= 372))) {
        KernelErfABMultiBlock<DT_X> op;
        op.Init(x, y, tiling_data.length);
        op.Process();
        return;
    }
    if (tiling_data.mode == 0 && tiling_data.rank == 1 &&
        ((tiling_data.length > 112 && tiling_data.length <= 128) ||
         (tiling_data.length > 368 && tiling_data.length <= 384))) {
        KernelErfABDirect<DT_X> op;
        op.Init(x, y, tiling_data.length);
        op.Process();
        return;
    }
    if (tiling_data.mode == 1 && tiling_data.rank == 1 && tiling_data.length <= 8) {
        KernelErfSmallMultiBlock<DT_X> op;
        op.Init(x, y, tiling_data.length, tiling_data.tileLength, 0);
        op.Process();
        return;
    }
    if (tiling_data.mode == 1 || tiling_data.mode == 3) {
        KernelErfSmall<DT_X> op;
        op.Init(x, y, tiling_data.length, tiling_data.tileLength, tiling_data.mode == 3 ? 1 : 0);
        op.Process();
        return;
    }
    if (tiling_data.mode == 2 && tiling_data.rank == 2 &&
        tiling_data.length > 65536 && tiling_data.length <= 524288) {
        TPipe pipe;
        KernelErfCD<DT_X> op;
        op.Init(pipe, x, y, tiling_data.length, tiling_data.tileLength);
        op.Process();
        return;
    }
    KernelErf<DT_X> op;
    op.Init(x, y, tiling_data.length, tiling_data.tileLength, tiling_data.mode == 2 ? 1 : 0);
    op.Process();
}
