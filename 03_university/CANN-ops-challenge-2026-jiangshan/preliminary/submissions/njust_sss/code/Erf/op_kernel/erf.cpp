#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

#ifdef erf
#undef erf
#endif

using namespace AscendC;

namespace {
constexpr uint32_t kTileLength = 8192;
constexpr uint32_t kLargeTileLength = 12288;
constexpr uint32_t kLargeTensorMinLength = 327680;
constexpr uint32_t kSmallVectorLength = kLargeTileLength;
constexpr uint32_t kMaskAlignElements = 8;
constexpr uint32_t kBuiltinTmpBufCount = 3;
constexpr uint32_t kCustomTmpBufCount = 2;

// Paper 2504.05068 exp-free approximation with (M, N, K) = (0, 3, 2).
// phi(z) ~= (P(z) / Q(z))^(2^K), erf(x) = x / sqrt(x^2 + phi(x^2)).
constexpr float kErfPhiQ0 = 0x1.4e7d79f97e1a5p+3f;
constexpr float kErfPhiQ1 = 0x1.0a7c452e440c3p+6f;
constexpr float kErfPhiQ2 = 0x1.b75461d2f00c8p+8f;
constexpr float kErfPhiP0 = 0x1.9d955091d7f40p+8f;
}

__aicore__ inline uint32_t AlignVectorCount(uint32_t n) {
    uint32_t nAligned = (n + kMaskAlignElements - 1) & ~(kMaskAlignElements - 1);
    if (nAligned == 0) nAligned = kMaskAlignElements;
    return nAligned;
}

__aicore__ inline void CustomErfApproxUnmasked(
    LocalTensor<float>& y, LocalTensor<float>& x,
    LocalTensor<float>& tmp0, LocalTensor<float>& tmp1, uint32_t n) {
    Mul(tmp0, x, x, n);
    PipeBarrier<PIPE_V>();

    Adds(y, tmp0, kErfPhiQ0, n);
    PipeBarrier<PIPE_V>();

    Mul(y, y, tmp0, n);
    PipeBarrier<PIPE_V>();

    Adds(y, y, kErfPhiQ1, n);
    PipeBarrier<PIPE_V>();

    Mul(y, y, tmp0, n);
    PipeBarrier<PIPE_V>();

    Adds(y, y, kErfPhiQ2, n);
    Duplicate<float>(tmp1, kErfPhiP0, n);
    PipeBarrier<PIPE_V>();

    Div(y, tmp1, y, n);
    PipeBarrier<PIPE_V>();

    Mul(y, y, y, n);
    PipeBarrier<PIPE_V>();

    Mul(y, y, y, n);
    PipeBarrier<PIPE_V>();

    Add(tmp1, tmp0, y, n);
    PipeBarrier<PIPE_V>();

    Sqrt(tmp1, tmp1, n);
    PipeBarrier<PIPE_V>();

    Div(y, x, tmp1, n);
}

template <class DT_X>
class KernelErf {
    static constexpr uint32_t kBufBytes = kTileLength * sizeof(DT_X);
    static constexpr uint32_t kAlignElements = 32 / sizeof(DT_X);

public:
    __aicore__ inline KernelErf() : effectiveBufBytes_(kBufBytes) {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, uint32_t length, uint32_t start, uint32_t totalElems) {
        start_ = start;
        totalElems_ = totalElems;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), length);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), length);

        if (sizeof(DT_X) == sizeof(float) && length >= kLargeTensorMinLength) {
            if (totalElems_ > kTileLength && totalElems_ <= kLargeTileLength) {
                effectiveBufBytes_ = kLargeTileLength * sizeof(DT_X);
                needDoubleBuffer_ = false;
            } else {
                needDoubleBuffer_ = (totalElems_ > kTileLength);
            }
        } else {
            needDoubleBuffer_ = totalElems_ > kTileLength;
        }

        tPipe_.Init();
        tPipe_.InitBuffer(inQueue_, needDoubleBuffer_ ? 2 : 1, effectiveBufBytes_);
        tPipe_.InitBuffer(outQueue_, needDoubleBuffer_ ? 2 : 1, effectiveBufBytes_);
        tPipe_.InitBuffer(buf0_, CalcBufBytes());
    }

    __aicore__ inline void ErfVec(LocalTensor<DT_X>& y, LocalTensor<DT_X>& x, uint32_t n) {
        if constexpr (sizeof(DT_X) == sizeof(float)) {
            CustomErfApprox(y, x, n);
            return;
        }

        auto tmp = buf0_.Get<uint8_t>(effectiveBufBytes_ * kBuiltinTmpBufCount);
        Erf<DT_X>(y, x, tmp, n);
    }

    __aicore__ inline uint32_t CalcBufBytes() const {
        if constexpr (sizeof(DT_X) == sizeof(float)) {
            return effectiveBufBytes_ * kCustomTmpBufCount;
        }
        return effectiveBufBytes_ * kBuiltinTmpBufCount;
    }

    __aicore__ inline void CustomErfApproxFullTile(LocalTensor<float>& y, LocalTensor<float>& x) {
        auto tmp = buf0_.Get<uint8_t>(effectiveBufBytes_ * kCustomTmpBufCount);
        LocalTensor<float> tmp0 = tmp.ReinterpretCast<float>();
        LocalTensor<float> tmp1 = tmp0[kTileLength];
        CustomErfApproxUnmasked(y, x, tmp0, tmp1, kTileLength);
    }

    __aicore__ inline void CustomErfApproxTailWithTmp(
        LocalTensor<float>& y, LocalTensor<float>& x, LocalTensor<float>& tmp0, uint32_t n) {
        // Align tmp1 to 32 bytes: AscendC vector ops require 32-byte aligned
        // buffers. When n is not a multiple of 8, tmp0[n] is unaligned and
        // causes ACL_ERROR_RT_VECTOR_CORE_EXCEPTION (507035).
        uint32_t nAligned = ((n + kMaskAlignElements - 1) / kMaskAlignElements) * kMaskAlignElements;
        if (nAligned == 0) nAligned = kMaskAlignElements;
        LocalTensor<float> tmp1 = tmp0[nAligned];

        // Extend the processing count to nAligned: the COUNTER mask rounds up
        // in some CANN versions, and nAligned is always safe (multiple of 8,
        // bounded by effective tile length). Extra computed elements fall
        // inside the output buffer and are never copied back.
        uint32_t nSafe = nAligned;

        const UnaryRepeatParams unaryParams;
        const BinaryRepeatParams binaryParams;

        SetMaskCount();
        SetVectorMask<half, MaskMode::COUNTER>(0, nSafe);

        // Vectorized exp-free path from Eq. (14): z = x^2, phi(z) = (P/Q)^4, erf(x) = x / sqrt(z + phi).
        Mul<float, false>(tmp0, x, x, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();

        Adds<float, false>(y, tmp0, kErfPhiQ0, MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();

        Mul<float, false>(y, y, tmp0, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();

        Adds<float, false>(y, y, kErfPhiQ1, MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();

        Mul<float, false>(y, y, tmp0, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();

        Adds<float, false>(y, y, kErfPhiQ2, MASK_PLACEHOLDER, 1, unaryParams);
        // Keep Duplicate in the active COUNTER-mask mode; the count overload
        // resets mask state and corrupts larger tails on this CANN version.
        Duplicate<float, false>(tmp1, kErfPhiP0, MASK_PLACEHOLDER, 1, 1, 8);
        PipeBarrier<PIPE_V>();

        Div<float, false>(y, tmp1, y, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();

        Mul<float, false>(y, y, y, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();

        Mul<float, false>(y, y, y, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();

        Add<float, false>(tmp1, tmp0, y, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();

        Sqrt<float, false>(tmp1, tmp1, MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();

        Div<float, false>(y, x, tmp1, MASK_PLACEHOLDER, 1, binaryParams);

        SetMaskNorm();
        ResetMask();
    }

    __aicore__ inline void CustomErfApproxTail(LocalTensor<float>& y, LocalTensor<float>& x, uint32_t n) {
        auto tmp = buf0_.Get<uint8_t>(effectiveBufBytes_ * kCustomTmpBufCount);
        LocalTensor<float> tmp0 = tmp.ReinterpretCast<float>();
        CustomErfApproxTailWithTmp(y, x, tmp0, n);
    }

    __aicore__ inline void CustomErfApprox(LocalTensor<float>& y, LocalTensor<float>& x, uint32_t n) {
        if (n == kTileLength) {
            CustomErfApproxFullTile(y, x);
            return;
        }
        CustomErfApproxTail(y, x, n);
    }

    __aicore__ inline void CopyInTile(LocalTensor<DT_X>& xLocal, uint32_t offset, uint32_t elemCount) {
        if ((elemCount % kAlignElements) == 0) {
            DataCopy(xLocal, xGm_[offset], elemCount);
            return;
        }

        DataCopyExtParams p{1, static_cast<uint32_t>(elemCount * sizeof(DT_X)), 0, 0, 0};
        DataCopyPadExtParams<DT_X> pp{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[offset], p, pp);
    }

    __aicore__ inline void CopyOutTile(uint32_t offset, LocalTensor<DT_X>& yLocal, uint32_t elemCount) {
        if ((elemCount % kAlignElements) == 0) {
            DataCopy(yGm_[offset], yLocal, elemCount);
            return;
        }

        DataCopyExtParams p{1, static_cast<uint32_t>(elemCount * sizeof(DT_X)), 0, 0, 0};
        DataCopyPad(yGm_[offset], yLocal, p);
    }

    __aicore__ inline void ProcessSingleTile(uint32_t offset, uint32_t elemCount) {
        auto xLocal = inQueue_.AllocTensor<DT_X>();
        auto yLocal = outQueue_.AllocTensor<DT_X>();

        CopyInTile(xLocal, offset, elemCount);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        ErfVec(yLocal, xLocal, elemCount);

        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        CopyOutTile(offset, yLocal, elemCount);

        outQueue_.FreeTensor(yLocal);
        inQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void Process() {
        const uint32_t tileLen = effectiveBufBytes_ / sizeof(DT_X);
        const uint32_t unifStart = start_;
        const uint32_t unifElems = totalElems_;
        const uint32_t unifEnd = unifStart + unifElems;
        const uint32_t unifNumTiles = (unifElems + tileLen - 1) / tileLen;

        uint32_t start = unifStart;
        uint32_t end = unifEnd;
        uint32_t totalElems = unifElems;
        uint32_t numTiles = unifNumTiles;

        if (numTiles == 0) return;
        if (numTiles == 1) {
            ProcessSingleTile(start, totalElems);
            return;
        }

        uint32_t offset = start;
        uint32_t curCount = (offset + kTileLength < end) ? kTileLength : (end - offset);

        {
            auto xLocal = inQueue_.AllocTensor<DT_X>();
            CopyInTile(xLocal, offset, curCount);
            inQueue_.EnQue(xLocal);
        }

        for (uint32_t tile = 0; tile < numTiles; ++tile) {
            const uint32_t nextOffset = offset + curCount;
            const bool hasNextTile = (tile + 1) < numTiles;
            const uint32_t nextCount = hasNextTile
                ? ((nextOffset + kTileLength < end) ? kTileLength : (end - nextOffset))
                : 0;

            if (hasNextTile) {
                auto xLocalNext = inQueue_.AllocTensor<DT_X>();
                CopyInTile(xLocalNext, nextOffset, nextCount);
                inQueue_.EnQue(xLocalNext);
            }

            auto xIn = inQueue_.DeQue<DT_X>();
            auto yLocal = outQueue_.AllocTensor<DT_X>();
            ErfVec(yLocal, xIn, curCount);

            outQueue_.EnQue(yLocal);
            inQueue_.FreeTensor(xIn);

            auto yOut = outQueue_.DeQue<DT_X>();
            CopyOutTile(offset, yOut, curCount);
            outQueue_.FreeTensor(yOut);

            offset = nextOffset;
            curCount = nextCount;
        }
    }

    TPipe tPipe_;
    TQue<QuePosition::VECIN, 2> inQueue_;
    TQue<QuePosition::VECOUT, 2> outQueue_;
    TBuf<TPosition::VECCALC> buf0_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t start_ = 0;
    uint32_t totalElems_ = 0;
    uint32_t effectiveBufBytes_ = kBufBytes;
    bool needDoubleBuffer_ = false;
};

__aicore__ inline void ProcessSmallFloatVectorFast(
    GM_ADDR x, GM_ADDR y, uint32_t length, uint32_t start, uint32_t totalElems) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    const uint32_t nAligned = AlignVectorCount(totalElems);
    const uint32_t smallBufBytes = 4 * nAligned * sizeof(float);
    pipe.Init();
    pipe.InitBuffer(buf, smallBufBytes);

    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x), length);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y), length);

    auto tmp = buf.Get<uint8_t>(smallBufBytes);
    LocalTensor<float> xLocal = tmp.ReinterpretCast<float>();
    LocalTensor<float> yLocal = xLocal[nAligned];
    LocalTensor<float> tmp0 = yLocal[nAligned];
    LocalTensor<float> tmp1 = tmp0[nAligned];

    if ((totalElems & 7) == 0) {
        DataCopy(xLocal, xGm[start], totalElems);
    } else {
        DataCopyExtParams p{1, static_cast<uint32_t>(totalElems * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pp{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm[start], p, pp);
    }

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    CustomErfApproxUnmasked(yLocal, xLocal, tmp0, tmp1, nAligned);

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    if ((totalElems & 7) == 0) {
        DataCopy(yGm[start], yLocal, totalElems);
        return;
    }

    DataCopyExtParams p{1, static_cast<uint32_t>(totalElems * sizeof(float)), 0, 0, 0};
    DataCopyPad(yGm[start], yLocal, p);
}

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    const uint32_t length = tiling_data.length;
    if (length == 0) return;

    const uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    if (blockNum == 0) return;

    if constexpr (sizeof(DT_X) == sizeof(float)) {
        if (blockNum == 1 && length <= kSmallVectorLength) {
            ProcessSmallFloatVectorFast(x, y, length, 0, length);
            return;
        }
    }

    const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    if (blockIdx >= blockNum) return;

    const uint32_t workPerCore = (length + blockNum - 1) / blockNum;
    const uint32_t start = blockIdx * workPerCore;
    if (start >= length) return;
    const uint32_t end = (start + workPerCore < length) ? (start + workPerCore) : length;
    const uint32_t totalElems = end - start;

    if constexpr (sizeof(DT_X) == sizeof(float)) {
        if (totalElems <= kSmallVectorLength) {
            ProcessSmallFloatVectorFast(x, y, length, start, totalElems);
            return;
        }
        const uint32_t totalTiles = (length + kTileLength - 1) / kTileLength;
        const uint32_t tilesPerCoreBase = totalTiles / blockNum;
        const uint32_t extraCores = totalTiles % blockNum;
        const uint32_t tilesThisCore = tilesPerCoreBase + (blockIdx < extraCores ? 1 : 0);
        if (tilesThisCore == 0) return;

        const uint32_t cumTiles = blockIdx * tilesPerCoreBase + (blockIdx < extraCores ? blockIdx : extraCores);
        const uint32_t tileStart = cumTiles * kTileLength;
        if (tileStart >= length) return;
        uint32_t tileEnd = tileStart + tilesThisCore * kTileLength;
        if (tileEnd > length) tileEnd = length;
        const uint32_t tileElems = tileEnd - tileStart;

        if (tileElems <= kSmallVectorLength) {
            ProcessSmallFloatVectorFast(x, y, length, tileStart, tileElems);
            return;
        }

        KernelErf<DT_X> op;
        op.Init(x, y, length, tileStart, tileElems);
        op.Process();
        return;
    }

    KernelErf<DT_X> op;
    op.Init(x, y, length, start, totalElems);
    op.Process();
}
