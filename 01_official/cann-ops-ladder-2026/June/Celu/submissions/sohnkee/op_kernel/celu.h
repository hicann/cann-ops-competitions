/*!
 * \file celu.h
 * \brief Celu kernel implementation.
 */

#ifndef CELU_H
#define CELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "celu_tiling_data.h"
#include "celu_tiling_key.h"

namespace NsCelu {

using namespace AscendC;

constexpr int32_t SINGLE_BUFFER_NUM = 1;
constexpr int32_t DOUBLE_BUFFER_NUM = 2;
constexpr int32_t NON_PIPELINE_DOUBLE_TQUE_DEPTH = 2;
constexpr int32_t PIPELINE_DOUBLE_TQUE_DEPTH = 2;
constexpr int64_t PIPELINE_MIN_FULL_TILES = 3;

template <
    typename T,
    int32_t QUEUE_BUFFER_NUM = DOUBLE_BUFFER_NUM,
    bool BALANCED_TILE = false,
    bool PIPELINE_CAPABLE = false,
    int32_t QUEUE_DEPTH = QUEUE_BUFFER_NUM,
    bool CLAMP_BEFORE_EXP = false>
class Celu {
public:
    __aicore__ inline Celu() {};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const CeluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInAligned(int64_t offset, int64_t currentNum);
    __aicore__ inline void CopyIn(int64_t offset, int64_t currentNum);
    __aicore__ inline void CopyOutAligned(int64_t offset, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t offset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> inputQueueX;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockStart_;
    int64_t blockLength_;
    int64_t ubLength_;
    T alphaValue_;
    T invAlphaValue_;
    bool alphaIsOne_;
    bool enablePipeline_;
};

template <typename T>
class CeluScalar {
public:
    __aicore__ inline CeluScalar() {};

    __aicore__ inline void InitAndProcess(GM_ADDR x, GM_ADDR y, const CeluTilingData* tilingData);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inputQueueX;
    TQue<QuePosition::VECOUT, 1> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
};

template <typename T>
__aicore__ inline void CeluScalar<T>::InitAndProcess(GM_ADDR x, GM_ADDR y, const CeluTilingData* tilingData)
{
    if (GetBlockIdx() != 0 || tilingData->totalNum <= 0) {
        return;
    }

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    constexpr int64_t blockElems = 32 / sizeof(T);
    pipe.InitBuffer(inputQueueX, 1, blockElems * sizeof(T));
    pipe.InitBuffer(outputQueueY, 1, blockElems * sizeof(T));

    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(xLocal, inputGMX[0], copyInParams, padParams);
    inputQueueX.EnQue<T>(xLocal);

    xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    const T zero = static_cast<T>(0);
    const T negOne = static_cast<T>(-1);
    const float alpha = tilingData->alpha;
    if (alpha == 1.0f) {
        Exp(yLocal, xLocal, 1);
        Adds(yLocal, yLocal, negOne, 1);
    } else {
        Muls(yLocal, xLocal, static_cast<T>(1.0f / alpha), 1);
        Exp(yLocal, yLocal, 1);
        Adds(yLocal, yLocal, negOne, 1);
        Muls(yLocal, yLocal, static_cast<T>(alpha), 1);
    }
    Mins(yLocal, yLocal, zero, 1);
    Maxs(xLocal, xLocal, zero, 1);
    Add(yLocal, yLocal, xLocal, 1);

    inputQueueX.FreeTensor<T>(xLocal);
    outputQueueY.EnQue<T>(yLocal);

    yLocal = outputQueueY.DeQue<T>();
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGMY[0], yLocal, copyOutParams);
    outputQueueY.FreeTensor<T>(yLocal);
}

class CeluScalarPolyOne {
public:
    __aicore__ inline CeluScalarPolyOne() {};

    __aicore__ inline void InitAndProcess(GM_ADDR x, GM_ADDR y);

private:
    __aicore__ inline float ExpNegApprox(float xValue) const;

private:
    GlobalTensor<float> inputGMX;
    GlobalTensor<float> outputGMY;
};

class CeluSmallScalarAlpha {
public:
    __aicore__ inline CeluSmallScalarAlpha() {};

    __aicore__ inline void InitAndProcess(GM_ADDR x, GM_ADDR y, const CeluTilingData* tilingData);

private:
    __aicore__ inline float ExpNegApprox(float xValue) const;

private:
    GlobalTensor<float> inputGMX;
    GlobalTensor<float> outputGMY;
};

class CeluSmallBlockScalarAlpha {
public:
    __aicore__ inline CeluSmallBlockScalarAlpha() {};

    __aicore__ inline void InitAndProcess(GM_ADDR x, GM_ADDR y, const CeluTilingData* tilingData);

private:
    __aicore__ inline float ExpNegApprox(float xValue) const;

private:
    GlobalTensor<float> inputGMX;
    GlobalTensor<float> outputGMY;
};

class CeluTinyVectorAlpha {
public:
    __aicore__ inline CeluTinyVectorAlpha() {};

    __aicore__ inline void InitAndProcess(GM_ADDR x, GM_ADDR y, const CeluTilingData* tilingData);

private:
    __aicore__ inline float ExpNegApprox(float xValue) const;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inputQueueX;
    TQue<QuePosition::VECOUT, 1> outputQueueY;

    GlobalTensor<float> inputGMX;
    GlobalTensor<float> outputGMY;
};

class CeluTinyScalarAlpha {
public:
    __aicore__ inline CeluTinyScalarAlpha() {};
    __aicore__ inline void InitAndProcess(GM_ADDR x, GM_ADDR y, int32_t count, float alpha);

private:
    __aicore__ inline float ExpNegApprox(float xValue) const;

private:
    GlobalTensor<float> inputGMX;
    GlobalTensor<float> outputGMY;
};

__aicore__ inline float CeluScalarPolyOne::ExpNegApprox(float xValue) const
{
    if (xValue <= -10.0f) {
        return 0.0f;
    }

    const float r = xValue * 0.03125f;
    float p = 1.0f / 120.0f;
    p = 1.0f / 24.0f + r * p;
    p = 1.0f / 6.0f + r * p;
    p = 0.5f + r * p;
    p = 1.0f + r * p;
    p = 1.0f + r * p;

    float e2 = p * p;
    float e4 = e2 * e2;
    float e8 = e4 * e4;
    float e16 = e8 * e8;
    return e16 * e16;
}

__aicore__ inline void CeluScalarPolyOne::InitAndProcess(GM_ADDR x, GM_ADDR y)
{
    if (GetBlockIdx() != 0) {
        return;
    }

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));

    const float xValue = inputGMX.GetValue(0);
    float yValue = xValue;
    if (xValue <= 0.0f) {
        yValue = ExpNegApprox(xValue) - 1.0f;
    }
    outputGMY.SetValue(0, yValue);
}

__aicore__ inline float CeluSmallScalarAlpha::ExpNegApprox(float xValue) const
{
    if (xValue <= -10.0f) {
        return 0.0f;
    }

    const float r = xValue * 0.03125f;
    float p = 1.0f / 120.0f;
    p = 1.0f / 24.0f + r * p;
    p = 1.0f / 6.0f + r * p;
    p = 0.5f + r * p;
    p = 1.0f + r * p;
    p = 1.0f + r * p;

    float e2 = p * p;
    float e4 = e2 * e2;
    float e8 = e4 * e4;
    float e16 = e8 * e8;
    return e16 * e16;
}

__aicore__ inline void CeluSmallScalarAlpha::InitAndProcess(GM_ADDR x, GM_ADDR y, const CeluTilingData* tilingData)
{
    if (GetBlockIdx() != 0 || tilingData->totalNum <= 0) {
        return;
    }

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));

    const int64_t totalNum = tilingData->totalNum;
    const float alpha = tilingData->alpha;
    const float invAlpha = 1.0f / alpha;
    for (int64_t i = 0; i < totalNum; ++i) {
        const float xValue = inputGMX.GetValue(i);
        float yValue = xValue;
        if (xValue <= 0.0f) {
            yValue = alpha * (ExpNegApprox(xValue * invAlpha) - 1.0f);
        }
        outputGMY.SetValue(i, yValue);
    }
}

__aicore__ inline float CeluSmallBlockScalarAlpha::ExpNegApprox(float xValue) const
{
    if (xValue <= -10.0f) {
        return 0.0f;
    }

    const float r = xValue * 0.03125f;
    float p = 1.0f / 120.0f;
    p = 1.0f / 24.0f + r * p;
    p = 1.0f / 6.0f + r * p;
    p = 0.5f + r * p;
    p = 1.0f + r * p;
    p = 1.0f + r * p;

    float e2 = p * p;
    float e4 = e2 * e2;
    float e8 = e4 * e4;
    float e16 = e8 * e8;
    return e16 * e16;
}

__aicore__ inline void CeluSmallBlockScalarAlpha::InitAndProcess(
    GM_ADDR x,
    GM_ADDR y,
    const CeluTilingData* tilingData)
{
    if (tilingData->totalNum <= 0 || tilingData->blockFactor <= 0) {
        return;
    }

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t start = blockIdx * tilingData->blockFactor;
    if (start >= tilingData->totalNum) {
        return;
    }
    int64_t end = start + tilingData->blockFactor;
    if (end > tilingData->totalNum) {
        end = tilingData->totalNum;
    }

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));

    const float alpha = tilingData->alpha;
    const float invAlpha = 1.0f / alpha;
    for (int64_t i = start; i < end; ++i) {
        const float xValue = inputGMX.GetValue(i);
        float yValue = xValue;
        if (xValue <= 0.0f) {
            yValue = alpha * (ExpNegApprox(xValue * invAlpha) - 1.0f);
        }
        outputGMY.SetValue(i, yValue);
    }
}

__aicore__ inline float CeluTinyVectorAlpha::ExpNegApprox(float xValue) const
{
    if (xValue <= -10.0f) {
        return 0.0f;
    }

    const float r = xValue * 0.03125f;
    float p = 1.0f / 120.0f;
    p = 1.0f / 24.0f + r * p;
    p = 1.0f / 6.0f + r * p;
    p = 0.5f + r * p;
    p = 1.0f + r * p;
    p = 1.0f + r * p;

    float e2 = p * p;
    float e4 = e2 * e2;
    float e8 = e4 * e4;
    float e16 = e8 * e8;
    return e16 * e16;
}

__aicore__ inline void CeluTinyVectorAlpha::InitAndProcess(GM_ADDR x, GM_ADDR y, const CeluTilingData* tilingData)
{
    if (GetBlockIdx() != 0 || tilingData->totalNum <= 0) {
        return;
    }

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));

    constexpr int64_t blockElems = 8;
    int64_t count = tilingData->totalNum;
    if (count > 16) {
        count = 16;
    }
    if (tilingData->alpha == 1.0f) {
        for (int64_t i = 0; i < count; ++i) {
            const float xValue = inputGMX.GetValue(i);
            float yValue = xValue;
            if (xValue <= 0.0f) {
                yValue = ExpNegApprox(xValue) - 1.0f;
            }
            outputGMY.SetValue(i, yValue);
        }
        return;
    }

    pipe.InitBuffer(inputQueueX, 1, blockElems * 2 * sizeof(float));
    pipe.InitBuffer(outputQueueY, 1, blockElems * 2 * sizeof(float));

    LocalTensor<float> xLocal = inputQueueX.AllocTensor<float>();
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
    DataCopyPad(xLocal, inputGMX[0], copyInParams, padParams);
    inputQueueX.EnQue<float>(xLocal);

    xLocal = inputQueueX.DeQue<float>();
    LocalTensor<float> yLocal = outputQueueY.AllocTensor<float>();

    constexpr float zero = 0.0f;
    constexpr float negOne = -1.0f;
    const float alpha = tilingData->alpha;
    const float invAlpha = alpha == 0.5f ? 2.0f : 1.0f / alpha;
    Muls(yLocal, xLocal, invAlpha, count);
    Exp(yLocal, yLocal, count);
    Adds(yLocal, yLocal, negOne, count);
    Muls(yLocal, yLocal, alpha == 0.5f ? 0.5f : alpha, count);
    Mins(yLocal, yLocal, zero, count);
    Maxs(xLocal, xLocal, zero, count);
    Add(yLocal, yLocal, xLocal, count);

    inputQueueX.FreeTensor<float>(xLocal);
    outputQueueY.EnQue<float>(yLocal);

    yLocal = outputQueueY.DeQue<float>();
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
    DataCopyPad(outputGMY[0], yLocal, copyOutParams);
    outputQueueY.FreeTensor<float>(yLocal);
}

__aicore__ inline float CeluTinyScalarAlpha::ExpNegApprox(float xValue) const
{
    if (xValue <= -10.0f) {
        return 0.0f;
    }

    const float r = xValue * 0.03125f;
    float p = 1.0f / 120.0f;
    p = 1.0f / 24.0f + r * p;
    p = 1.0f / 6.0f + r * p;
    p = 0.5f + r * p;
    p = 1.0f + r * p;
    p = 1.0f + r * p;

    float e2 = p * p;
    float e4 = e2 * e2;
    float e8 = e4 * e4;
    float e16 = e8 * e8;
    return e16 * e16;
}

__aicore__ inline void CeluTinyScalarAlpha::InitAndProcess(GM_ADDR x, GM_ADDR y, int32_t count, float alpha)
{
    if (GetBlockIdx() != 0) {
        return;
    }

    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));

    const float invAlpha = 1.0f / alpha;
    for (int32_t i = 0; i < count; ++i) {
        const float xValue = inputGMX.GetValue(i);
        float yValue = xValue;
        if (xValue <= 0.0f) {
            yValue = alpha * (ExpNegApprox(xValue * invAlpha) - 1.0f);
        }
        outputGMY.SetValue(i, yValue);
    }
}

template <
    typename T,
    int32_t QUEUE_BUFFER_NUM,
    bool BALANCED_TILE,
    bool PIPELINE_CAPABLE,
    int32_t QUEUE_DEPTH,
    bool CLAMP_BEFORE_EXP>
__aicore__ inline void Celu<
    T,
    QUEUE_BUFFER_NUM,
    BALANCED_TILE,
    PIPELINE_CAPABLE,
    QUEUE_DEPTH,
    CLAMP_BEFORE_EXP>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const CeluTilingData* tilingData)
{
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    const int64_t totalNum = tilingData->totalNum;
    ubLength_ = tilingData->ubFactor;
    blockLength_ = 0;
    const float alpha = tilingData->alpha;
    alphaValue_ = static_cast<T>(alpha);
    invAlphaValue_ = static_cast<T>(1.0f / alpha);
    alphaIsOne_ = alpha == 1.0f;
    enablePipeline_ = PIPELINE_CAPABLE && ((tilingData->reserved & CELU_RESERVED_PIPELINE_FLAG) != 0);

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
    int32_t QUEUE_DEPTH,
    bool CLAMP_BEFORE_EXP>
__aicore__ inline void Celu<
    T,
    QUEUE_BUFFER_NUM,
    BALANCED_TILE,
    PIPELINE_CAPABLE,
    QUEUE_DEPTH,
    CLAMP_BEFORE_EXP>::CopyInAligned(
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
    int32_t QUEUE_DEPTH,
    bool CLAMP_BEFORE_EXP>
__aicore__ inline void Celu<
    T,
    QUEUE_BUFFER_NUM,
    BALANCED_TILE,
    PIPELINE_CAPABLE,
    QUEUE_DEPTH,
    CLAMP_BEFORE_EXP>::CopyIn(
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
    int32_t QUEUE_DEPTH,
    bool CLAMP_BEFORE_EXP>
__aicore__ inline void Celu<
    T,
    QUEUE_BUFFER_NUM,
    BALANCED_TILE,
    PIPELINE_CAPABLE,
    QUEUE_DEPTH,
    CLAMP_BEFORE_EXP>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.template AllocTensor<T>();

    const T zero = static_cast<T>(0);
    const T negOne = static_cast<T>(-1);
    const uint32_t count = static_cast<uint32_t>(currentNum);

    if constexpr (CLAMP_BEFORE_EXP) {
        Mins(yLocal, xLocal, zero, count);
        Muls(yLocal, yLocal, invAlphaValue_, count);
        Exp(yLocal, yLocal, count);
        Adds(yLocal, yLocal, negOne, count);
        Muls(yLocal, yLocal, alphaValue_, count);
    } else if (alphaIsOne_) {
        Exp(yLocal, xLocal, count);
        Adds(yLocal, yLocal, negOne, count);
    } else {
        Muls(yLocal, xLocal, invAlphaValue_, count);
        Exp(yLocal, yLocal, count);
        Adds(yLocal, yLocal, negOne, count);
        Muls(yLocal, yLocal, alphaValue_, count);
    }
    if constexpr (!CLAMP_BEFORE_EXP) {
        Mins(yLocal, yLocal, zero, count);
    }

    Maxs(xLocal, xLocal, zero, count);
    Add(yLocal, yLocal, xLocal, count);

    inputQueueX.template FreeTensor<T>(xLocal);
    outputQueueY.template EnQue<T>(yLocal);
}

template <
    typename T,
    int32_t QUEUE_BUFFER_NUM,
    bool BALANCED_TILE,
    bool PIPELINE_CAPABLE,
    int32_t QUEUE_DEPTH,
    bool CLAMP_BEFORE_EXP>
__aicore__ inline void Celu<
    T,
    QUEUE_BUFFER_NUM,
    BALANCED_TILE,
    PIPELINE_CAPABLE,
    QUEUE_DEPTH,
    CLAMP_BEFORE_EXP>::CopyOutAligned(
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
    int32_t QUEUE_DEPTH,
    bool CLAMP_BEFORE_EXP>
__aicore__ inline void Celu<
    T,
    QUEUE_BUFFER_NUM,
    BALANCED_TILE,
    PIPELINE_CAPABLE,
    QUEUE_DEPTH,
    CLAMP_BEFORE_EXP>::CopyOut(
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
    int32_t QUEUE_DEPTH,
    bool CLAMP_BEFORE_EXP>
__aicore__ inline void Celu<
    T,
    QUEUE_BUFFER_NUM,
    BALANCED_TILE,
    PIPELINE_CAPABLE,
    QUEUE_DEPTH,
    CLAMP_BEFORE_EXP>::Process()
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
        if (enablePipeline_ && fullLength >= PIPELINE_MIN_FULL_TILES * ubLength_) {
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

} // namespace NsCelu
#endif // CELU_H
