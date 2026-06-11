// Kernel implementation.
#include "kernel_operator.h"

#include "clip_by_value_tiling.h"
#include "tiling_key_clip_by_value.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <class T>
class KernelClipByValue {
public:
    __aicore__ inline KernelClipByValue() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ClipByValueTilingData &tilingData) {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t globalOffset = tilingData.bigCoreDataNum * blockIdx;
        tileDataNum_ = tilingData.tileDataNum;

        if (blockIdx < tilingData.tailBlockNum) {
            coreDataNum_ = tilingData.bigCoreDataNum;
            tileNum_ = tilingData.finalBigTileNum;
            tailDataNum_ = tilingData.bigTailDataNum;
        } else {
            coreDataNum_ = tilingData.smallCoreDataNum;
            tileNum_ = tilingData.finalSmallTileNum;
            tailDataNum_ = tilingData.smallTailDataNum;
            globalOffset -= (tilingData.bigCoreDataNum - tilingData.smallCoreDataNum) *
                (blockIdx - tilingData.tailBlockNum);
        }

        x_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x) + globalOffset, coreDataNum_);
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y) + globalOffset, coreDataNum_);
        minValue_ = static_cast<T>(tilingData.min);
        maxValue_ = static_cast<T>(tilingData.max);

        pipe_.InitBuffer(inQueue_, BUFFER_NUM, tileDataNum_ * sizeof(T));
        pipe_.InitBuffer(outQueue_, BUFFER_NUM, tileDataNum_ * sizeof(T));
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < tileNum_; ++i) {
            processDataNum_ = (i == tileNum_ - 1) ? tailDataNum_ : tileDataNum_;
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress) {
        LocalTensor<T> inLocal = inQueue_.AllocTensor<T>();
        DataCopy(inLocal, x_[progress * tileDataNum_], processDataNum_);
        inQueue_.EnQue(inLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<T> inLocal = inQueue_.DeQue<T>();
        LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();
        Maxs(outLocal, inLocal, minValue_, processDataNum_);
        Mins(outLocal, outLocal, maxValue_, processDataNum_);
        outQueue_.EnQue(outLocal);
        inQueue_.FreeTensor(inLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress) {
        LocalTensor<T> outLocal = outQueue_.DeQue<T>();
        DataCopy(y_[progress * tileDataNum_], outLocal, processDataNum_);
        outQueue_.FreeTensor(outLocal);
    }

    GlobalTensor<T> x_;
    GlobalTensor<T> y_;
    uint32_t coreDataNum_;
    uint32_t tileNum_;
    uint32_t tileDataNum_;
    uint32_t tailDataNum_;
    uint32_t processDataNum_;
    T minValue_;
    T maxValue_;
    TPipe pipe_;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueue_;
};

template <typename T>
__aicore__ inline void RunKernel(GM_ADDR x, GM_ADDR y, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ClipByValueTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tilingData, tiling);
    KernelClipByValue<T> op;
    op.Init(x, y, tilingData);
    op.Process();
}

template <typename D_T_X, typename D_T_Y>
__global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    RunKernel<D_T_X>(x, y, tiling);
}
