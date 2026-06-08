#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

namespace erf_kernel_stage {
constexpr uint32_t kAlignElems = 64;
constexpr uint32_t kDbDepth = 2;

constexpr float kClampLo = -2.50f;
constexpr float kClampHi = 2.50f;
constexpr float kPolyC0 = 1.12773312f;
constexpr float kPolyC1 = -3.69239590e-1f;
constexpr float kPolyC2 = 1.00451917e-1f;
constexpr float kPolyC3 = -1.78319906e-2f;
constexpr float kPolyC4 = 1.79231248e-3f;
constexpr float kPolyC5 = -7.60648791e-5f;

__aicore__ inline bool IsVectorAligned(uint64_t offset, uint32_t count) {
    return ((offset | static_cast<uint64_t>(count)) & (kAlignElems - 1U)) == 0;
}

template <typename T>
__aicore__ inline void EvalErfPolynomial(LocalTensor<T> x, LocalTensor<T> y, LocalTensor<T> x2, uint32_t n) {
    const int32_t count = static_cast<int32_t>(n);

    Maxs(x, x, kClampLo, count);
    Mins(x, x, kClampHi, count);

    Mul(x2, x, x, count);

    Muls(y, x2, static_cast<T>(kPolyC5), count);
    Adds(y, y, static_cast<T>(kPolyC4), count);
    Mul(y, y, x2, count);
    Adds(y, y, static_cast<T>(kPolyC3), count);
    Mul(y, y, x2, count);
    Adds(y, y, static_cast<T>(kPolyC2), count);
    Mul(y, y, x2, count);
    Adds(y, y, static_cast<T>(kPolyC1), count);
    Mul(y, y, x2, count);
    Adds(y, y, static_cast<T>(kPolyC0), count);

    Mul(y, y, x, count);
}

template <typename T>
class SingleTileStage {
public:
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR y, uint64_t length, uint32_t tileLength, TPipe *pipe) {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));

        pipe->InitBuffer(xBuf_, tileLength * sizeof(T));
        pipe->InitBuffer(yBuf_, tileLength * sizeof(T));
        pipe->InitBuffer(square_, tileLength * sizeof(T));

        const uint32_t n = static_cast<uint32_t>(length);
        LocalTensor<T> xLocal = xBuf_.Get<T>();
        LocalTensor<T> yLocal = yBuf_.Get<T>();
        LocalTensor<T> tmp = square_.Get<T>();

        Load(xLocal, 0, n);
        SyncMte2ToV();
        Compute(xLocal, yLocal, tmp, n);
        SyncVToMte3();
        Store(yLocal, 0, n);
    }

private:
    __aicore__ inline void SyncMte2ToV() {
        int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void SyncVToMte3() {
        int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void Load(LocalTensor<T> local, uint64_t offset, uint32_t n) {
        if (IsVectorAligned(offset, n)) {
            DataCopy(local, xGm_[offset], n);
        } else {
            DataCopyExtParams cp;
            cp.blockCount = 1;
            cp.blockLen = n * sizeof(T);
            cp.srcStride = 0;
            cp.dstStride = 0;
            cp.rsv = 0;

            DataCopyPadExtParams<T> pad;
            pad.isPad = false;
            pad.leftPadding = 0;
            pad.rightPadding = 0;
            pad.paddingValue = static_cast<T>(0);
            DataCopyPad(local, xGm_[offset], cp, pad);
        }
    }

    __aicore__ inline void Compute(LocalTensor<T> x, LocalTensor<T> y, LocalTensor<T> tmp, uint32_t n) {
        EvalErfPolynomial(x, y, tmp, n);
    }

    __aicore__ inline void Store(LocalTensor<T> y, uint64_t offset, uint32_t n) {
        if (IsVectorAligned(offset, n)) {
            DataCopy(yGm_[offset], y, n);
        } else {
            DataCopyExtParams cp;
            cp.blockCount = 1;
            cp.blockLen = n * sizeof(T);
            cp.srcStride = 0;
            cp.dstStride = 0;
            cp.rsv = 0;
            DataCopyPad(yGm_[offset], y, cp);
        }
    }

private:
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    TBuf<QuePosition::VECCALC> xBuf_;
    TBuf<QuePosition::VECCALC> yBuf_;
    TBuf<QuePosition::VECCALC> square_;
};

template <typename T>
class DoubleBufferStage {
public:
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR y, uint64_t length, uint32_t tileLength, TPipe *pipe) {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));

        pipe->InitBuffer(input_, kDbDepth, tileLength * sizeof(T));
        pipe->InitBuffer(output_, kDbDepth, tileLength * sizeof(T));
        pipe->InitBuffer(square_, tileLength * sizeof(T));

        const uint64_t tile = static_cast<uint64_t>(tileLength);
        const uint64_t tileNum = (length + tile - 1) / tile;

        uint64_t readyOffset = 0;
        uint32_t readyCount = static_cast<uint32_t>(tile < length ? tile : length);
        Load(readyOffset, readyCount);

        uint64_t nextOffset = tile;
        for (uint64_t i = 1; i < tileNum; ++i) {
            const uint64_t remain = length - nextOffset;
            const uint32_t nextCount = static_cast<uint32_t>(remain < tile ? remain : tile);

            Compute(readyCount);
            Load(nextOffset, nextCount);
            Store(readyOffset, readyCount);

            readyOffset = nextOffset;
            readyCount = nextCount;
            nextOffset += tile;
        }

        Compute(readyCount);
        Store(readyOffset, readyCount);
    }

private:
    __aicore__ inline void Load(uint64_t offset, uint32_t n) {
        LocalTensor<T> local = input_.AllocTensor<T>();
        if (IsVectorAligned(offset, n)) {
            DataCopy(local, xGm_[offset], n);
        } else {
            DataCopyExtParams cp;
            cp.blockCount = 1;
            cp.blockLen = n * sizeof(T);
            cp.srcStride = 0;
            cp.dstStride = 0;
            cp.rsv = 0;

            DataCopyPadExtParams<T> pad;
            pad.isPad = false;
            pad.leftPadding = 0;
            pad.rightPadding = 0;
            pad.paddingValue = static_cast<T>(0);
            DataCopyPad(local, xGm_[offset], cp, pad);
        }
        input_.EnQue(local);
    }

    __aicore__ inline void Compute(uint32_t n) {
        LocalTensor<T> x = input_.DeQue<T>();
        LocalTensor<T> y = output_.AllocTensor<T>();
        LocalTensor<T> tmp = square_.Get<T>();
        EvalErfPolynomial(x, y, tmp, n);
        output_.EnQue(y);
        input_.FreeTensor(x);
    }

    __aicore__ inline void Store(uint64_t offset, uint32_t n) {
        LocalTensor<T> y = output_.DeQue<T>();
        if (IsVectorAligned(offset, n)) {
            DataCopy(yGm_[offset], y, n);
        } else {
            DataCopyExtParams cp;
            cp.blockCount = 1;
            cp.blockLen = n * sizeof(T);
            cp.srcStride = 0;
            cp.dstStride = 0;
            cp.rsv = 0;
            DataCopyPad(yGm_[offset], y, cp);
        }
        output_.FreeTensor(y);
    }

private:
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    TQue<QuePosition::VECIN, kDbDepth> input_;
    TQue<QuePosition::VECOUT, kDbDepth> output_;
    TBuf<QuePosition::VECCALC> square_;
};

__aicore__ inline void ResolveCoreRange(const ErfTilingData &t,
                                        uint32_t blockIdx,
                                        uint64_t &offset,
                                        uint64_t &length) {
    if (blockIdx < t.tailBlockNum) {
        length = t.bigCoreDataNum;
        offset = static_cast<uint64_t>(blockIdx) * t.bigCoreDataNum;
    } else {
        length = t.smallCoreDataNum;
        offset = static_cast<uint64_t>(t.tailBlockNum) * t.bigCoreDataNum
               + static_cast<uint64_t>(blockIdx - t.tailBlockNum) * t.smallCoreDataNum;
    }

    if (t.tailNum != 0 && blockIdx == t.usedCoreNum - 1) {
        length += t.tailNum;
    }
}
}  // namespace erf_kernel_stage

template <typename DT_X, bool IS_SINGLE_TILE>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    (void)workspace;
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tilingData, tiling);

    const uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= tilingData.usedCoreNum) {
        return;
    }

    uint64_t offset = 0;
    uint64_t localLength = 0;
    erf_kernel_stage::ResolveCoreRange(tilingData, blockIdx, offset, localLength);
    if (localLength == 0) {
        return;
    }

    TPipe pipe;
    if constexpr (IS_SINGLE_TILE) {
        erf_kernel_stage::SingleTileStage<DT_X> runner;
        runner.Run(x + offset * sizeof(DT_X), y + offset * sizeof(DT_X), localLength, tilingData.tileLength, &pipe);
    } else {
        erf_kernel_stage::DoubleBufferStage<DT_X> runner;
        runner.Run(x + offset * sizeof(DT_X), y + offset * sizeof(DT_X), localLength, tilingData.tileLength, &pipe);
    }
}
