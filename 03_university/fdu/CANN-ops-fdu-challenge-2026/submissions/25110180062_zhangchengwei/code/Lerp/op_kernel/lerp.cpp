// Kernel implementation.
#include "kernel_operator.h"

#include "lerp_tiling.h"
#include "tiling_key_lerp.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <class T>
class KernelLerp {
public:
    __aicore__ inline KernelLerp() {}

    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR y, const LerpTilingData &tilingData) {
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

        start_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(start) + globalOffset, coreDataNum_);
        end_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(end) + globalOffset, coreDataNum_);
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y) + globalOffset, coreDataNum_);
        weight_ = static_cast<T>(tilingData.weight);

        pipe_.InitBuffer(startQueue_, BUFFER_NUM, tileDataNum_ * sizeof(T));
        pipe_.InitBuffer(endQueue_, BUFFER_NUM, tileDataNum_ * sizeof(T));
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
        LocalTensor<T> startLocal = startQueue_.AllocTensor<T>();
        LocalTensor<T> endLocal = endQueue_.AllocTensor<T>();
        DataCopy(startLocal, start_[progress * tileDataNum_], processDataNum_);
        DataCopy(endLocal, end_[progress * tileDataNum_], processDataNum_);
        startQueue_.EnQue(startLocal);
        endQueue_.EnQue(endLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<T> startLocal = startQueue_.DeQue<T>();
        LocalTensor<T> endLocal = endQueue_.DeQue<T>();
        LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();

        Sub(outLocal, endLocal, startLocal, processDataNum_);
        Muls(outLocal, outLocal, weight_, processDataNum_);
        Add(outLocal, startLocal, outLocal, processDataNum_);

        outQueue_.EnQue(outLocal);
        startQueue_.FreeTensor(startLocal);
        endQueue_.FreeTensor(endLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress) {
        LocalTensor<T> outLocal = outQueue_.DeQue<T>();
        DataCopy(y_[progress * tileDataNum_], outLocal, processDataNum_);
        outQueue_.FreeTensor(outLocal);
    }

    GlobalTensor<T> start_;
    GlobalTensor<T> end_;
    GlobalTensor<T> y_;
    uint32_t coreDataNum_;
    uint32_t tileNum_;
    uint32_t tileDataNum_;
    uint32_t tailDataNum_;
    uint32_t processDataNum_;
    T weight_;
    TPipe pipe_;
    TQue<TPosition::VECIN, BUFFER_NUM> startQueue_;
    TQue<TPosition::VECIN, BUFFER_NUM> endQueue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueue_;
};

template <typename T>
__aicore__ inline void RunKernel(GM_ADDR start, GM_ADDR end, GM_ADDR y, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LerpTilingData);
    GET_TILING_DATA_WITH_STRUCT(LerpTilingData, tilingData, tiling);
    KernelLerp<T> op;
    op.Init(start, end, y, tilingData);
    op.Process();
}

template <typename DT_START>
__global__ __aicore__ void lerp(GM_ADDR start, GM_ADDR end, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    RunKernel<DT_START>(start, end, y, tiling);
}
