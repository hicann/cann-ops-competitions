/*!
 * \file hard_swish.h
 * \brief HardSwish kernel implementation
 */

#ifndef HARDSWISH_H
#define HARDSWISH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "hard_swish_tiling_data.h"
#include "hard_swish_tiling_key.h"

namespace NsHardSwish {

using namespace AscendC;

constexpr int32_t INPUT_BUFFER_NUM = 2;
constexpr int32_t OUTPUT_BUFFER_NUM = 2;
constexpr int64_t ALIGN_NUM = 8;

template <typename T, bool ALIGNED_COPY = false>
class HardSwish {
public:
    __aicore__ inline HardSwish(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const HardSwishTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, INPUT_BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, OUTPUT_BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T, bool ALIGNED_COPY>
__aicore__ inline void HardSwish<T, ALIGNED_COPY>::Init(GM_ADDR x, GM_ADDR y, const HardSwishTilingData* tilingData, TPipe* pipe)
{
    pipe_ = pipe;
    const int64_t totalNum = tilingData->totalNum;
    const int64_t blockFactor = tilingData->blockFactor;
    const int64_t startOffset = static_cast<int64_t>(GetBlockIdx()) * blockFactor;

    blockLength_ = 0;
    if (startOffset < totalNum) {
        blockLength_ = totalNum - startOffset;
        if (blockLength_ > blockFactor) {
            blockLength_ = blockFactor;
        }
    }
    ubLength_ = tilingData->ubFactor > 0 ? tilingData->ubFactor : 1;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + startOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + startOffset, blockLength_);

    pipe_->InitBuffer(inputQueueX, INPUT_BUFFER_NUM, ubLength_ * sizeof(T));
    pipe_->InitBuffer(outputQueueY, OUTPUT_BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T, bool ALIGNED_COPY>
__aicore__ inline void HardSwish<T, ALIGNED_COPY>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    if constexpr (ALIGNED_COPY) {
        DataCopy(xLocal, inputGMX[progress], static_cast<uint32_t>(currentNum));
    } else {
        if ((currentNum & (ALIGN_NUM - 1)) == 0) {
            DataCopy(xLocal, inputGMX[progress], static_cast<uint32_t>(currentNum));
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(xLocal, inputGMX[progress], copyParams, padParams);
        }
    }
    inputQueueX.EnQue(xLocal);
}

template <typename T, bool ALIGNED_COPY>
__aicore__ inline void HardSwish<T, ALIGNED_COPY>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    const uint32_t calcNum = static_cast<uint32_t>(currentNum);

    Adds(yLocal, xLocal, static_cast<T>(3.0f), calcNum);
    Maxs(yLocal, yLocal, static_cast<T>(0.0f), calcNum);
    Mins(yLocal, yLocal, static_cast<T>(6.0f), calcNum);
    Mul(yLocal, yLocal, xLocal, calcNum);
    Muls(yLocal, yLocal, static_cast<T>(0.16666666666666666f), calcNum);

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T, bool ALIGNED_COPY>
__aicore__ inline void HardSwish<T, ALIGNED_COPY>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    if constexpr (ALIGNED_COPY) {
        DataCopy(outputGMY[progress], yLocal, static_cast<uint32_t>(currentNum));
    } else {
        if ((currentNum & (ALIGN_NUM - 1)) == 0) {
            DataCopy(outputGMY[progress], yLocal, static_cast<uint32_t>(currentNum));
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
            DataCopyPad(outputGMY[progress], yLocal, copyParams);
        }
    }
    outputQueueY.FreeTensor(yLocal);
}

template <typename T, bool ALIGNED_COPY>
__aicore__ inline void HardSwish<T, ALIGNED_COPY>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }

    int64_t progress = 0;
    int64_t currentNum = ubLength_;
    if (currentNum > blockLength_) {
        currentNum = blockLength_;
    }
    CopyIn(progress, currentNum);

    while (progress < blockLength_) {
        const int64_t nextProgress = progress + currentNum;
        int64_t nextNum = 0;
        if (nextProgress < blockLength_) {
            nextNum = ubLength_;
            if (nextProgress + nextNum > blockLength_) {
                nextNum = blockLength_ - nextProgress;
            }
            CopyIn(nextProgress, nextNum);
        }

        Compute(currentNum);
        CopyOut(progress, currentNum);

        progress = nextProgress;
        currentNum = nextNum;
    }
}

template <typename T>
class HardSwishTinyQueue {
public:
    __aicore__ inline HardSwishTinyQueue(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, TPipe* pipe);
    __aicore__ inline void Process();

private:
    static constexpr uint32_t TINY_NUM = 15;
    static constexpr uint32_t TINY_CALC_NUM = 16;

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> inputQueueX;
    TQue<QuePosition::VECOUT, 1> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
};

template <typename T>
__aicore__ inline void HardSwishTinyQueue<T>::Init(GM_ADDR x, GM_ADDR y, TPipe* pipe)
{
    pipe_ = pipe;

    inputGMX.SetGlobalBuffer((__gm__ T*)x, TINY_NUM);
    outputGMY.SetGlobalBuffer((__gm__ T*)y, TINY_NUM);

    pipe_->InitBuffer(inputQueueX, 1, TINY_CALC_NUM * sizeof(T));
    pipe_->InitBuffer(outputQueueY, 1, TINY_CALC_NUM * sizeof(T));
}

template <typename T>
__aicore__ inline void HardSwishTinyQueue<T>::Process()
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(TINY_NUM * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(xLocal, inputGMX[0], copyParams, padParams);
    inputQueueX.EnQue(xLocal);

    xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    Adds(yLocal, xLocal, static_cast<T>(3.0f), TINY_CALC_NUM);
    Maxs(yLocal, yLocal, static_cast<T>(0.0f), TINY_CALC_NUM);
    Mins(yLocal, yLocal, static_cast<T>(6.0f), TINY_CALC_NUM);
    Mul(yLocal, yLocal, xLocal, TINY_CALC_NUM);
    Muls(yLocal, yLocal, static_cast<T>(0.16666666666666666f), TINY_CALC_NUM);

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);

    yLocal = outputQueueY.DeQue<T>();
    DataCopyPad(outputGMY[0], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

} // namespace NsHardSwish
#endif // HARDSWISH_H
