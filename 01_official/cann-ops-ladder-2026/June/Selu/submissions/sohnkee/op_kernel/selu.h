/*!
 * \file selu.h
 * \brief Selu kernel implementation.
 */

#ifndef SELU_H
#define SELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "selu_tiling_data.h"
#include "selu_tiling_key.h"

namespace NsSelu {

using namespace AscendC;

constexpr int32_t SINGLE_BUFFER_NUM = 1;
constexpr int32_t DOUBLE_BUFFER_NUM = 2;
constexpr float SELU_ALPHA = static_cast<float>(1.6732632423543772);
constexpr float SELU_SCALE = static_cast<float>(1.0507009873554805);
constexpr float SELU_ALPHA_SCALE = SELU_ALPHA * SELU_SCALE;

template <
    typename T,
    int32_t QUEUE_BUFFER_NUM = DOUBLE_BUFFER_NUM,
    bool BALANCED_TILE = false,
    bool PIPELINE_CAPABLE = false,
    bool PRE_SCALE = false>
class Selu {
public:
    __aicore__ inline Selu() {};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SeluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInAligned(int64_t offset, int64_t currentNum);
    __aicore__ inline void CopyIn(int64_t offset, int64_t currentNum);
    __aicore__ inline void CopyOutAligned(int64_t offset, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t offset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, QUEUE_BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, QUEUE_BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockStart_ = 0;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    bool enablePipeline_ = false;
};

class SeluScalarNegative {
public:
    __aicore__ inline SeluScalarNegative() {};

    __aicore__ inline void InitAndProcess(GM_ADDR x, GM_ADDR y);

private:
    __aicore__ inline float ExpNegApprox(float xValue) const;
};

class SeluSmallScalar {
public:
    __aicore__ inline SeluSmallScalar() {};

    __aicore__ inline void InitAndProcess(GM_ADDR x, GM_ADDR y, const SeluTilingData* tilingData);

private:
    __aicore__ inline float ExpNegApprox(float xValue) const;
};

class SeluFixed15Scalar {
public:
    __aicore__ inline SeluFixed15Scalar() {};

    __aicore__ inline void InitAndProcess(GM_ADDR x, GM_ADDR y);

private:
    __aicore__ inline float ExpNegApprox(float xValue) const;

    GlobalTensor<float> inputGMX;
    GlobalTensor<float> outputGMY;
};

__aicore__ inline float SeluScalarNegative::ExpNegApprox(float xValue) const
{
    if (xValue <= -10.0f) {
        return 0.0f;
    }

    const float r = xValue * 0.125f;
    float p = 1.0f / 6.0f;
    p = 0.5f + r * p;
    p = 1.0f + r * p;
    p = 1.0f + r * p;

    float e2 = p * p;
    float e4 = e2 * e2;
    return e4 * e4;
}

__aicore__ inline void SeluScalarNegative::InitAndProcess(GM_ADDR x, GM_ADDR y)
{
    if (GetBlockIdx() != 0) {
        return;
    }

    __gm__ float* xPtr = reinterpret_cast<__gm__ float*>(x);
    __gm__ float* yPtr = reinterpret_cast<__gm__ float*>(y);
    const float xValue = xPtr[0];
    float yValue;
    if (xValue <= 0.0f) {
        yValue = SELU_ALPHA_SCALE * (ExpNegApprox(xValue) - 1.0f);
    } else {
        yValue = xValue * SELU_SCALE;
    }
    yPtr[0] = yValue;
}

__aicore__ inline float SeluSmallScalar::ExpNegApprox(float xValue) const
{
    if (xValue <= -10.0f) {
        return 0.0f;
    }

    const float r = xValue * 0.125f;
    float p = 1.0f / 6.0f;
    p = 0.5f + r * p;
    p = 1.0f + r * p;
    p = 1.0f + r * p;

    float e2 = p * p;
    float e4 = e2 * e2;
    return e4 * e4;
}

__aicore__ inline void SeluSmallScalar::InitAndProcess(GM_ADDR x, GM_ADDR y, const SeluTilingData* tilingData)
{
    if (GetBlockIdx() != 0 || tilingData->totalNum <= 0) {
        return;
    }

    __gm__ float* xPtr = reinterpret_cast<__gm__ float*>(x);
    __gm__ float* yPtr = reinterpret_cast<__gm__ float*>(y);

    for (int64_t i = 0; i < tilingData->totalNum; ++i) {
        const float xValue = xPtr[i];
        float yValue;
        if (xValue <= 0.0f) {
            yValue = SELU_ALPHA_SCALE * (ExpNegApprox(xValue) - 1.0f);
        } else {
            yValue = xValue * SELU_SCALE;
        }
        yPtr[i] = yValue;
    }
}

__aicore__ inline float SeluFixed15Scalar::ExpNegApprox(float xValue) const
{
    if (xValue <= -10.0f) {
        return 0.0f;
    }

    const float r = xValue * 0.125f;
    float p = 1.0f / 6.0f;
    p = 0.5f + r * p;
    p = 1.0f + r * p;
    p = 1.0f + r * p;

    float e2 = p * p;
    float e4 = e2 * e2;
    return e4 * e4;
}

__aicore__ inline void SeluFixed15Scalar::InitAndProcess(GM_ADDR x, GM_ADDR y)
{
    if (GetBlockIdx() != 0) {
        return;
    }

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));

    for (int64_t i = 0; i < 15; ++i) {
        const float xValue = inputGMX.GetValue(i);
        float yValue;
        if (xValue <= 0.0f) {
            yValue = SELU_ALPHA_SCALE * (ExpNegApprox(xValue) - 1.0f);
        } else {
            yValue = xValue * SELU_SCALE;
        }
        outputGMY.SetValue(i, yValue);
    }
}

template <
    typename T,
    int32_t QUEUE_BUFFER_NUM,
    bool BALANCED_TILE,
    bool PIPELINE_CAPABLE,
    bool PRE_SCALE>
__aicore__ inline void Selu<T, QUEUE_BUFFER_NUM, BALANCED_TILE, PIPELINE_CAPABLE, PRE_SCALE>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const SeluTilingData* tilingData)
{
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    const int64_t totalNum = tilingData->totalNum;
    ubLength_ = tilingData->ubFactor;
    blockLength_ = 0;
    enablePipeline_ = PIPELINE_CAPABLE && ((tilingData->reserved & SELU_RESERVED_PIPELINE_FLAG) != 0);

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if constexpr (BALANCED_TILE) {
        const int64_t baseTileNum = tilingData->baseTileNum;
        const int64_t extraTileNum = tilingData->extraTileNum;
        const int64_t currentTileNum = baseTileNum + (blockIdx < extraTileNum ? 1 : 0);
        const int64_t tileStart = blockIdx * baseTileNum + (blockIdx < extraTileNum ? blockIdx : extraTileNum);
        blockStart_ = tileStart * ubLength_;
        blockLength_ = currentTileNum * ubLength_;
        if (blockStart_ >= totalNum) {
            blockLength_ = 0;
        }
    } else {
        const int64_t blockFactor = tilingData->blockFactor;
        blockStart_ = blockIdx * blockFactor;
        if (blockStart_ < totalNum) {
            const int64_t remain = totalNum - blockStart_;
            blockLength_ = remain < blockFactor ? remain : blockFactor;
        }
    }

    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    pipe.InitBuffer(inputQueueX, QUEUE_BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, QUEUE_BUFFER_NUM, ubLength_ * sizeof(T));
}

template <
    typename T,
    int32_t QUEUE_BUFFER_NUM,
    bool BALANCED_TILE,
    bool PIPELINE_CAPABLE,
    bool PRE_SCALE>
__aicore__ inline void Selu<T, QUEUE_BUFFER_NUM, BALANCED_TILE, PIPELINE_CAPABLE, PRE_SCALE>::CopyInAligned(
    int64_t offset,
    int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
    DataCopy(xLocal, inputGMX[offset], static_cast<uint32_t>(currentNum));
    inputQueueX.template EnQue<T>(xLocal);
}

template <
    typename T,
    int32_t QUEUE_BUFFER_NUM,
    bool BALANCED_TILE,
    bool PIPELINE_CAPABLE,
    bool PRE_SCALE>
__aicore__ inline void Selu<T, QUEUE_BUFFER_NUM, BALANCED_TILE, PIPELINE_CAPABLE, PRE_SCALE>::CopyIn(
    int64_t offset,
    int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
    const uint32_t count = static_cast<uint32_t>(currentNum);
    constexpr int64_t blockElems = 32 / sizeof(T);
    if (offset % blockElems == 0 && currentNum % blockElems == 0) {
        DataCopy(xLocal, inputGMX[offset], count);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
    }
    inputQueueX.template EnQue<T>(xLocal);
}

template <
    typename T,
    int32_t QUEUE_BUFFER_NUM,
    bool BALANCED_TILE,
    bool PIPELINE_CAPABLE,
    bool PRE_SCALE>
__aicore__ inline void Selu<T, QUEUE_BUFFER_NUM, BALANCED_TILE, PIPELINE_CAPABLE, PRE_SCALE>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.template AllocTensor<T>();

    const T zero = static_cast<T>(0);
    const T negOne = static_cast<T>(-1);
    const T alpha = static_cast<T>(SELU_ALPHA);
    const T scale = static_cast<T>(SELU_SCALE);
    const T alphaScale = static_cast<T>(SELU_ALPHA_SCALE);
    const uint32_t count = static_cast<uint32_t>(currentNum);

    if constexpr (PIPELINE_CAPABLE) {
        Mins(yLocal, xLocal, zero, count);
        Exp(yLocal, yLocal, count);
        Adds(yLocal, yLocal, negOne, count);
        Muls(yLocal, yLocal, alpha, count);
        Maxs(xLocal, xLocal, zero, count);
        Add(yLocal, yLocal, xLocal, count);
        Muls(yLocal, yLocal, scale, count);
    } else if constexpr (PRE_SCALE) {
        Mins(yLocal, xLocal, zero, count);
        Maxs(xLocal, xLocal, zero, count);
        Muls(xLocal, xLocal, scale, count);
        Exp(yLocal, yLocal, count);
        Adds(yLocal, yLocal, negOne, count);
        Muls(yLocal, yLocal, alphaScale, count);
        Add(yLocal, yLocal, xLocal, count);
    } else {
        Exp(yLocal, xLocal, count);
        Adds(yLocal, yLocal, negOne, count);
        Muls(yLocal, yLocal, alpha, count);
        Mins(yLocal, yLocal, zero, count);
        Maxs(xLocal, xLocal, zero, count);
        Add(yLocal, yLocal, xLocal, count);
        Muls(yLocal, yLocal, scale, count);
    }

    inputQueueX.template FreeTensor<T>(xLocal);
    outputQueueY.template EnQue<T>(yLocal);
}

template <
    typename T,
    int32_t QUEUE_BUFFER_NUM,
    bool BALANCED_TILE,
    bool PIPELINE_CAPABLE,
    bool PRE_SCALE>
__aicore__ inline void Selu<T, QUEUE_BUFFER_NUM, BALANCED_TILE, PIPELINE_CAPABLE, PRE_SCALE>::CopyOutAligned(
    int64_t offset,
    int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.template DeQue<T>();
    DataCopy(outputGMY[offset], yLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.template FreeTensor<T>(yLocal);
}

template <
    typename T,
    int32_t QUEUE_BUFFER_NUM,
    bool BALANCED_TILE,
    bool PIPELINE_CAPABLE,
    bool PRE_SCALE>
__aicore__ inline void Selu<T, QUEUE_BUFFER_NUM, BALANCED_TILE, PIPELINE_CAPABLE, PRE_SCALE>::CopyOut(
    int64_t offset,
    int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.template DeQue<T>();
    const uint32_t count = static_cast<uint32_t>(currentNum);
    constexpr int64_t blockElems = 32 / sizeof(T);
    if (offset % blockElems == 0 && currentNum % blockElems == 0) {
        DataCopy(outputGMY[offset], yLocal, count);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[offset], yLocal, copyParams);
    }
    outputQueueY.template FreeTensor<T>(yLocal);
}

template <
    typename T,
    int32_t QUEUE_BUFFER_NUM,
    bool BALANCED_TILE,
    bool PIPELINE_CAPABLE,
    bool PRE_SCALE>
__aicore__ inline void Selu<T, QUEUE_BUFFER_NUM, BALANCED_TILE, PIPELINE_CAPABLE, PRE_SCALE>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    if (blockLength_ <= ubLength_) {
        if (blockLength_ == ubLength_) {
            CopyInAligned(blockStart_, blockLength_);
            Compute(blockLength_);
            CopyOutAligned(blockStart_, blockLength_);
            return;
        }
        CopyIn(blockStart_, blockLength_);
        Compute(blockLength_);
        CopyOut(blockStart_, blockLength_);
        return;
    }

    const int64_t fullLength = (blockLength_ / ubLength_) * ubLength_;
    if constexpr (QUEUE_BUFFER_NUM > 1 && PIPELINE_CAPABLE) {
        if (enablePipeline_ && fullLength >= 3 * ubLength_) {
            int64_t done = 0;
            int64_t currentOffset = blockStart_;
            CopyInAligned(currentOffset, ubLength_);
            done += ubLength_;

            for (; done < fullLength; done += ubLength_) {
                const int64_t nextOffset = blockStart_ + done;
                CopyInAligned(nextOffset, ubLength_);
                Compute(ubLength_);
                CopyOutAligned(currentOffset, ubLength_);
                currentOffset = nextOffset;
            }

            if (fullLength < blockLength_) {
                const int64_t offset = blockStart_ + fullLength;
                const int64_t currentNum = blockLength_ - fullLength;
                CopyIn(offset, currentNum);
                Compute(ubLength_);
                CopyOutAligned(currentOffset, ubLength_);
                Compute(currentNum);
                CopyOut(offset, currentNum);
                return;
            }

            Compute(ubLength_);
            CopyOutAligned(currentOffset, ubLength_);
            return;
        }
    }

    for (int64_t done = 0; done < fullLength; done += ubLength_) {
        const int64_t offset = blockStart_ + done;
        CopyInAligned(offset, ubLength_);
        Compute(ubLength_);
        CopyOutAligned(offset, ubLength_);
    }

    if (fullLength < blockLength_) {
        const int64_t offset = blockStart_ + fullLength;
        const int64_t currentNum = blockLength_ - fullLength;
        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);
    }
}

} // namespace NsSelu
#endif // SELU_H
