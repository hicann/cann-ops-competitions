#include "kernel_operator.h"
#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

constexpr uint32_t kCopyAlignElems = 8U;
constexpr uint32_t kDoubleBufferDepth = 2U;

// 多项式系数与截断阈值
constexpr float kPolyC0 = 1.12419727f;
constexpr float kPolyC1 = -0.357146274f;
constexpr float kPolyC2 = 0.0879197606f;
constexpr float kPolyC3 = -0.0123576f;
constexpr float kPolyC4 = 0.00072783462f;
constexpr float kClampLow = -2.25f;
constexpr float kClampHigh = 2.25f;

// 辅助函数：判断是否满足对齐拷贝条件
template <class T>
__aicore__ inline bool UseAlignedCopy(uint64_t elemOffset, uint32_t elemCount) {
    return (elemOffset % kCopyAlignElems == 0) && (elemCount % kCopyAlignElems == 0);
}


template <class T>
__aicore__ inline void InitUnalignedCopyParams(DataCopyExtParams& copyParams, DataCopyPadExtParams<T>& padParams, uint32_t elemCount) {
    copyParams.blockCount = 1;
    copyParams.blockLen = elemCount * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = static_cast<T>(0);
}

// 辅助函数：计算 Erf 近似值（霍纳法则展开）
template <class T>
__aicore__ inline void EvaluateErfApprox(LocalTensor<T> src, LocalTensor<T> dst, LocalTensor<T> square, uint32_t elems) {
    const int32_t count = static_cast<int32_t>(elems);

    Maxs(src, src, kClampLow, count);
    Mins(src, src, kClampHigh, count);
    Mul(square, src, src, count);

    Muls(dst, square, static_cast<T>(kPolyC4), count);
    Adds(dst, dst, static_cast<T>(kPolyC3), count);
    Mul(dst, square, dst, count);
    Adds(dst, dst, static_cast<T>(kPolyC2), count);
    Mul(dst, square, dst, count);
    Adds(dst, dst, static_cast<T>(kPolyC1), count);
    Mul(dst, square, dst, count);
    Adds(dst, dst, static_cast<T>(kPolyC0), count);
    Mul(dst, src, dst, count);
}

template <class T>
class ErfSingleTileRunner {
public:
    __aicore__ inline ErfSingleTileRunner() {}

    __aicore__ inline void Run(GM_ADDR x, GM_ADDR y, uint64_t elems, uint32_t tileElems) {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));

        pipe_.InitBuffer(xQueue_, 1, tileElems * sizeof(T));
        pipe_.InitBuffer(yQueue_, 1, tileElems * sizeof(T));
        pipe_.InitBuffer(tmp_, tileElems * sizeof(T));

        const uint32_t workElems = static_cast<uint32_t>(elems);
        Load(workElems);
        Compute(workElems);
        Store(workElems);
    }

private:
    __aicore__ inline void Load(uint32_t elems) {
        LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
        if (UseAlignedCopy<T>(0ULL, elems)) {
            DataCopy(xLocal, xGm_[0], elems);
        } else {
            DataCopyExtParams copyParams;
            DataCopyPadExtParams<T> padParams;
            InitUnalignedCopyParams(copyParams, padParams, elems);
            DataCopyPad(xLocal, xGm_[0], copyParams, padParams);
        }
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t elems) {
        LocalTensor<T> xLocal = xQueue_.DeQue<T>();
        LocalTensor<T> yLocal = yQueue_.AllocTensor<T>();
        LocalTensor<T> scratch = tmp_.Get<T>();
        
        EvaluateErfApprox(xLocal, yLocal, scratch, elems);
        
        yQueue_.EnQue(yLocal);
        xQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void Store(uint32_t elems) {
        LocalTensor<T> yLocal = yQueue_.DeQue<T>();
        if (UseAlignedCopy<T>(0ULL, elems)) {
            DataCopy(yGm_[0], yLocal, elems);
        } else {
            DataCopyExtParams copyParams;
            DataCopyPadExtParams<T> padParams;
            InitUnalignedCopyParams(copyParams, padParams, elems);
            DataCopyPad(yGm_[0], yLocal, copyParams);
        }
        yQueue_.FreeTensor(yLocal);
    }

private:
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;
    TBuf<QuePosition::VECCALC> tmp_;
};

template <class T>
class ErfPipelinedRunner {
public:
    __aicore__ inline ErfPipelinedRunner() {}

    __aicore__ inline void Run(GM_ADDR x, GM_ADDR y, uint64_t elems, uint32_t tileElems) {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));

        pipe_.InitBuffer(xQueue_, kDoubleBufferDepth, tileElems * sizeof(T));
        pipe_.InitBuffer(yQueue_, kDoubleBufferDepth, tileElems * sizeof(T));
        pipe_.InitBuffer(tmp_, tileElems * sizeof(T));

        const uint64_t tileWidth = static_cast<uint64_t>(tileElems);
        const uint64_t tileNum = (elems + tileWidth - 1ULL) / tileWidth;

        // 1. 预取第一个 Tile
        EnqueueLoad(0, static_cast<uint32_t>(tileWidth < elems ? tileWidth : elems));

        // 2. 循环处理中间部分（计算当前Tile，预取下一个Tile，写出上一个Tile）
        for (uint64_t tileIdx = 1; tileIdx < tileNum; ++tileIdx) {
            const uint64_t curOffset = tileIdx * tileWidth;
            const uint64_t remain = elems - curOffset;
            const uint32_t curElems = static_cast<uint32_t>(remain < tileWidth ? remain : tileWidth);
            const uint64_t prevOffset = (tileIdx - 1) * tileWidth;
            const uint32_t prevElems = static_cast<uint32_t>(tileIdx == 1 ? (tileWidth < elems ? tileWidth : elems) : tileWidth);

            EnqueueCompute(prevElems);
            EnqueueLoad(curOffset, curElems);
            DequeueStore(prevOffset, prevElems);
        }

        // 3. 处理最后一个 Tile 的计算与写出
        const uint64_t lastOffset = (tileNum - 1) * tileWidth;
        const uint32_t lastElems = static_cast<uint32_t>(tileNum == 1 ? elems : (elems - lastOffset));
        EnqueueCompute(lastElems);
        DequeueStore(lastOffset, lastElems);
    }

private:
    __aicore__ inline void EnqueueLoad(uint64_t offset, uint32_t elems) {
        LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
        if (UseAlignedCopy<T>(offset, elems)) {
            DataCopy(xLocal, xGm_[offset], elems);
        } else {
            DataCopyExtParams copyParams;
            DataCopyPadExtParams<T> padParams;
            InitUnalignedCopyParams(copyParams, padParams, elems);
            DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        }
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void EnqueueCompute(uint32_t elems) {
        LocalTensor<T> xLocal = xQueue_.DeQue<T>();
        LocalTensor<T> yLocal = yQueue_.AllocTensor<T>();
        LocalTensor<T> scratch = tmp_.Get<T>();
        
        EvaluateErfApprox(xLocal, yLocal, scratch, elems);
        
        yQueue_.EnQue(yLocal);
        xQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void DequeueStore(uint64_t offset, uint32_t elems) {
        LocalTensor<T> yLocal = yQueue_.DeQue<T>();
        if (UseAlignedCopy<T>(offset, elems)) {
            DataCopy(yGm_[offset], yLocal, elems);
        } else {
            DataCopyExtParams copyParams;
            DataCopyPadExtParams<T> padParams;
            InitUnalignedCopyParams(copyParams, padParams, elems);
            DataCopyPad(yGm_[offset], yLocal, copyParams);
        }
        yQueue_.FreeTensor(yLocal);
    }

private:
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    TPipe pipe_;
    TQue<QuePosition::VECIN, kDoubleBufferDepth> xQueue_;
    TQue<QuePosition::VECOUT, kDoubleBufferDepth> yQueue_;
    TBuf<QuePosition::VECCALC> tmp_;
};

template <typename DT_X, bool IS_SINGLE_TILE>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    (void)workspace;
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tilingData, tiling);

    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockDim = tilingData.usedCoreNum;
    if (blockIdx >= blockDim) {
        return;
    }

    // 计算当前 Core 的偏移量和数据量
    const uint32_t bigCoreCount = tilingData.tailBlockNum;
    uint64_t coreOffset = 0ULL;
    uint64_t coreElems = 0ULL;

    if (blockIdx < bigCoreCount) {
        coreElems = tilingData.bigCoreDataNum;
        coreOffset = static_cast<uint64_t>(blockIdx) * tilingData.bigCoreDataNum;
    } else {
        coreElems = tilingData.smallCoreDataNum;
        coreOffset = static_cast<uint64_t>(bigCoreCount) * tilingData.bigCoreDataNum +
                     static_cast<uint64_t>(blockIdx - bigCoreCount) * tilingData.smallCoreDataNum;
    }

    // 处理最后一个 Core 的尾部数据
    if (tilingData.tailNum != 0U && blockIdx == blockDim - 1U) {
        coreElems += tilingData.tailNum;
    }

    if (coreElems == 0ULL) {
        return;
    }

    GM_ADDR coreX = x + coreOffset * sizeof(DT_X);
    GM_ADDR coreY = y + coreOffset * sizeof(DT_X);

    if constexpr (IS_SINGLE_TILE) {
        ErfSingleTileRunner<DT_X> runner;
        runner.Run(coreX, coreY, coreElems, tilingData.tileLength);
    } else {
        ErfPipelinedRunner<DT_X> runner;
        runner.Run(coreX, coreY, coreElems, tilingData.tileLength);
    }
}
