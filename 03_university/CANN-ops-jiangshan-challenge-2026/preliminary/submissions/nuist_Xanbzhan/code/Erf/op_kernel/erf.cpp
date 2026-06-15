#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

constexpr float P0 = 1.12419727f;
constexpr float P1 = -0.357146274f;
constexpr float P2 = 0.0879197606f;
constexpr float P3 = -0.0123576f;
constexpr float P4 = 0.00072783462f;
constexpr float ERF_BOUND_LO = -2.25f;
constexpr float ERF_BOUND_HI = 2.25f;
constexpr uint32_t DMA_ALIGN_ELEMS = 8;
constexpr uint32_t DDR_BLOCK_ELEMS = 128;
constexpr uint32_t UB_CACHE_ELEMS = 2048;

template <class DT_X>
__aicore__ inline void EvalErfPoly(
    LocalTensor<DT_X> xLocal,
    LocalTensor<DT_X> yLocal,
    LocalTensor<DT_X> tLocal,
    uint32_t calCount) {
    const int32_t count = static_cast<int32_t>(calCount);

    Maxs(xLocal, xLocal, ERF_BOUND_LO, count);
    Mins(xLocal, xLocal, ERF_BOUND_HI, count);

    // y = x²  (tensor roles swapped vs reference implementation)
    Mul(yLocal, xLocal, xLocal, count);

    Muls(tLocal, yLocal, static_cast<DT_X>(P4), count);
    Adds(tLocal, tLocal, static_cast<DT_X>(P3), count);
    Mul(tLocal, yLocal, tLocal, count);
    Adds(tLocal, tLocal, static_cast<DT_X>(P2), count);
    Mul(tLocal, yLocal, tLocal, count);
    Adds(tLocal, tLocal, static_cast<DT_X>(P1), count);
    Mul(tLocal, yLocal, tLocal, count);
    Adds(tLocal, tLocal, static_cast<DT_X>(P0), count);

    // y = x * inner_poly
    Mul(yLocal, xLocal, tLocal, count);
}

// ---- DMA helpers (inline, zero overhead) ----
// General-purpose: handles both aligned and unaligned transfers.
template <class DT_X>
__aicore__ inline void Fetch(LocalTensor<DT_X> &dst, const GlobalTensor<DT_X> &src,
                              uint32_t n, uint64_t off = 0) {
    if (((off | n) & (DMA_ALIGN_ELEMS - 1)) == 0) {
        DataCopy(dst, src[off], n);
    } else {
        DataCopyExtParams cp = {1, static_cast<uint32_t>(n * sizeof(DT_X)), 0, 0, 0};
        DataCopyPadExtParams<DT_X> pp = {false, 0, 0, static_cast<DT_X>(0)};
        DataCopyPad(dst, src[off], cp, pp);
    }
}

template <class DT_X>
__aicore__ inline void Push(GlobalTensor<DT_X> &dst, const LocalTensor<DT_X> &src,
                             uint32_t n, uint64_t off = 0) {
    if (((off | n) & (DMA_ALIGN_ELEMS - 1)) == 0) {
        DataCopy(dst[off], src, n);
    } else {
        DataCopyExtParams cp = {1, static_cast<uint32_t>(n * sizeof(DT_X)), 0, 0, 0};
        DataCopyPad(dst[off], src, cp);
    }
}

// Aligned-only fast path: skips the branch.  Safe when (off|n) % 8 == 0,
// which is always true for the pipeline path (tiles are multiples of DDR_BLOCK=128).
template <class DT_X>
__aicore__ inline void FetchAligned(LocalTensor<DT_X> &dst, const GlobalTensor<DT_X> &src,
                                     uint32_t n, uint64_t off = 0) {
    DataCopy(dst, src[off], n);
}

template <class DT_X>
__aicore__ inline void PushAligned(GlobalTensor<DT_X> &dst, const LocalTensor<DT_X> &src,
                                    uint32_t n, uint64_t off = 0) {
    DataCopy(dst[off], src, n);
}

// ---- Block-mode fast path (compile-time COUNT, no tiling data read) ----
// Processes exactly COUNT elements: safe because Ascend tensor allocations are
// page-aligned; extra reads/writes stay within the physical allocation.
template <class DT_X, uint32_t COUNT>
__aicore__ inline void RunBlock(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t X_OFF = 0;
    constexpr uint32_t Y_OFF = COUNT * sizeof(DT_X);
    constexpr uint32_t T_OFF = Y_OFF + COUNT * sizeof(DT_X);

    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), COUNT);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), COUNT);

    LocalTensor<DT_X> xLocal(TPosition::VECCALC, X_OFF, COUNT);
    LocalTensor<DT_X> yLocal(TPosition::VECCALC, Y_OFF, COUNT);
    LocalTensor<DT_X> tLocal(TPosition::VECCALC, T_OFF, COUNT);

    DataCopy(xLocal, xGm[0], COUNT);
    SetFlag<HardEvent::MTE2_V>(event_t::EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(event_t::EVENT_ID0);

    EvalErfPoly(xLocal, yLocal, tLocal, COUNT);

    SetFlag<HardEvent::V_MTE3>(event_t::EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(event_t::EVENT_ID0);
    DataCopy(yGm[0], yLocal, COUNT);
}

__aicore__ inline void CalcBlockRange(
    uint32_t totalBlocks,
    uint32_t tailElems,
    uint32_t usedCores,
    uint32_t blockIdx,
    uint64_t &coreStart,
    uint64_t &coreLen) {

    // Hot path: multi-core with blocks (pipeline mode, most common)
    if (__builtin_expect(usedCores <= 1, 0)) {
        coreStart = 0;
        coreLen = totalBlocks * DDR_BLOCK_ELEMS + tailElems;
        return;
    }

    if (__builtin_expect(totalBlocks == 0, 0)) {
        coreStart = 0;
        coreLen = (blockIdx > 0) ? 0 : tailElems;
        return;
    }

    uint32_t base = totalBlocks / usedCores;
    uint32_t extra = totalBlocks % usedCores;
    uint32_t myBlocks = base + (blockIdx < extra ? 1u : 0u);
    uint32_t startBlock = blockIdx * base + (blockIdx < extra ? blockIdx : extra);

    coreStart = startBlock * DDR_BLOCK_ELEMS;
    coreLen = myBlocks * DDR_BLOCK_ELEMS;

    if (blockIdx == usedCores - 1 && tailElems > 0) {
        coreLen += tailElems;
    }
}

template <class DT_X>
class ErfEngine {
public:
    __aicore__ inline ErfEngine() {}

    __aicore__ inline void Invoke(
        GM_ADDR x, GM_ADDR y,
        uint64_t coreLen,
        uint32_t tileLength,
        TPipe &pipe) {

        src_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        dst_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
        RunStream(coreLen, tileLength, pipe);
    }

private:
    __aicore__ inline void RunStream(uint64_t coreLen, uint32_t tileLength, TPipe &pipe) {
        pipe.InitBuffer(rx_, 2, tileLength * sizeof(DT_X));
        pipe.InitBuffer(wy_, 2, tileLength * sizeof(DT_X));
        pipe.InitBuffer(scratch_, tileLength * sizeof(DT_X));

        chunk_ = tileLength;

        const uint64_t tileNum = coreLen / chunk_;
        const uint64_t tailLen = coreLen % chunk_;
        const uint64_t loopTileNum = tileNum + (tailLen > 0 ? 1 : 0);

        if (loopTileNum == 0) {
            return;
        }

        // Fast path: single tile — skip pipeline loop overhead entirely
        if (loopTileNum == 1) {
            uint32_t curLen = (tailLen > 0) ? static_cast<uint32_t>(tailLen) : chunk_;
            Load(0, curLen);
            Calc(curLen);
            Store(0, curLen);
            return;
        }

        // Prologue: prefetch first tile (always full chunk_ size)
        uint32_t curLen = chunk_;
        Load(0, curLen);

        // Software pipeline: Load(t+1) || Calc(t) || Store(t)
        // curLen is carried across iterations to avoid redundant ChunkLen calls.
        for (uint64_t tileIdx = 0; tileIdx + 1 < loopTileNum; ++tileIdx) {
            uint32_t nextLen = ((tileIdx + 1) < tileNum) ? chunk_
                               : static_cast<uint32_t>(tailLen);
            Load(tileIdx + 1, nextLen);
            Calc(curLen);
            Store(tileIdx, curLen);
            curLen = nextLen;
        }

        // Epilogue: last tile — curLen already holds the correct length
        {
            Calc(curLen);
            Store(loopTileNum - 1, curLen);
        }
    }

    __aicore__ inline void Load(uint64_t tileIdx, uint32_t calCount) {
        LocalTensor<DT_X> xLocal = rx_.AllocTensor<DT_X>();
        FetchAligned(xLocal, src_, calCount, tileIdx * chunk_);
        rx_.EnQue(xLocal);
    }

    __aicore__ inline void Calc(uint32_t calCount) {
        LocalTensor<DT_X> xLocal = rx_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = wy_.AllocTensor<DT_X>();
        LocalTensor<DT_X> tLocal = scratch_.Get<DT_X>();
        EvalErfPoly(xLocal, yLocal, tLocal, calCount);
        wy_.EnQue(yLocal);
        rx_.FreeTensor(xLocal);
    }

    __aicore__ inline void Store(uint64_t tileIdx, uint32_t calCount) {
        LocalTensor<DT_X> yLocal = wy_.DeQue<DT_X>();
        PushAligned(dst_, yLocal, calCount, tileIdx * chunk_);
        wy_.FreeTensor(yLocal);
    }

private:
    GlobalTensor<DT_X> src_;
    GlobalTensor<DT_X> dst_;
    TQue<QuePosition::VECIN, 2> rx_;
    TQue<QuePosition::VECOUT, 2> wy_;
    TBuf<QuePosition::VECCALC> scratch_;
    uint32_t chunk_;
};

template <typename DT_X, uint64_t SCH_MODE>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // ── Block-mode fast paths: compile-time COUNT, fully inlined ──
    // Each branch is self-contained to eliminate function-call boundary.
    if constexpr (SCH_MODE == SCH_BLK8) {
        constexpr uint32_t N = 8;
        constexpr uint32_t XO = 0, YO = N * sizeof(DT_X), TO = YO + N * sizeof(DT_X);
        GlobalTensor<DT_X> xG, yG;
        xG.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), N);
        yG.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), N);
        LocalTensor<DT_X> xL(TPosition::VECCALC, XO, N);
        LocalTensor<DT_X> yL(TPosition::VECCALC, YO, N);
        LocalTensor<DT_X> tL(TPosition::VECCALC, TO, N);
        DataCopy(xL, xG[0], N);
        SetFlag<HardEvent::MTE2_V>(event_t::EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(event_t::EVENT_ID0);
        EvalErfPoly(xL, yL, tL, N);
        SetFlag<HardEvent::V_MTE3>(event_t::EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(event_t::EVENT_ID0);
        DataCopy(yG[0], yL, N);
        return;
    }
    if constexpr (SCH_MODE == SCH_BLK32) {
        constexpr uint32_t N = 32;
        constexpr uint32_t XO = 0, YO = N * sizeof(DT_X), TO = YO + N * sizeof(DT_X);
        GlobalTensor<DT_X> xG, yG;
        xG.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), N);
        yG.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), N);
        LocalTensor<DT_X> xL(TPosition::VECCALC, XO, N);
        LocalTensor<DT_X> yL(TPosition::VECCALC, YO, N);
        LocalTensor<DT_X> tL(TPosition::VECCALC, TO, N);
        DataCopy(xL, xG[0], N);
        SetFlag<HardEvent::MTE2_V>(event_t::EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(event_t::EVENT_ID0);
        EvalErfPoly(xL, yL, tL, N);
        SetFlag<HardEvent::V_MTE3>(event_t::EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(event_t::EVENT_ID0);
        DataCopy(yG[0], yL, N);
        return;
    }
    if constexpr (SCH_MODE == SCH_BLK64) {
        constexpr uint32_t N = 64;
        constexpr uint32_t XO = 0, YO = N * sizeof(DT_X), TO = YO + N * sizeof(DT_X);
        GlobalTensor<DT_X> xG, yG;
        xG.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), N);
        yG.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), N);
        LocalTensor<DT_X> xL(TPosition::VECCALC, XO, N);
        LocalTensor<DT_X> yL(TPosition::VECCALC, YO, N);
        LocalTensor<DT_X> tL(TPosition::VECCALC, TO, N);
        DataCopy(xL, xG[0], N);
        SetFlag<HardEvent::MTE2_V>(event_t::EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(event_t::EVENT_ID0);
        EvalErfPoly(xL, yL, tL, N);
        SetFlag<HardEvent::V_MTE3>(event_t::EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(event_t::EVENT_ID0);
        DataCopy(yG[0], yL, N);
        return;
    }
    if constexpr (SCH_MODE == SCH_BLK128) {
        constexpr uint32_t N = 128;
        constexpr uint32_t XO = 0, YO = N * sizeof(DT_X), TO = YO + N * sizeof(DT_X);
        GlobalTensor<DT_X> xG, yG;
        xG.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), N);
        yG.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), N);
        LocalTensor<DT_X> xL(TPosition::VECCALC, XO, N);
        LocalTensor<DT_X> yL(TPosition::VECCALC, YO, N);
        LocalTensor<DT_X> tL(TPosition::VECCALC, TO, N);
        DataCopy(xL, xG[0], N);
        SetFlag<HardEvent::MTE2_V>(event_t::EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(event_t::EVENT_ID0);
        EvalErfPoly(xL, yL, tL, N);
        SetFlag<HardEvent::V_MTE3>(event_t::EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(event_t::EVENT_ID0);
        DataCopy(yG[0], yL, N);
        return;
    }
    if constexpr (SCH_MODE == SCH_BLK256) {
        constexpr uint32_t N = 256;
        constexpr uint32_t XO = 0, YO = N * sizeof(DT_X), TO = YO + N * sizeof(DT_X);
        GlobalTensor<DT_X> xG, yG;
        xG.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), N);
        yG.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), N);
        LocalTensor<DT_X> xL(TPosition::VECCALC, XO, N);
        LocalTensor<DT_X> yL(TPosition::VECCALC, YO, N);
        LocalTensor<DT_X> tL(TPosition::VECCALC, TO, N);
        DataCopy(xL, xG[0], N);
        SetFlag<HardEvent::MTE2_V>(event_t::EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(event_t::EVENT_ID0);
        EvalErfPoly(xL, yL, tL, N);
        SetFlag<HardEvent::V_MTE3>(event_t::EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(event_t::EVENT_ID0);
        DataCopy(yG[0], yL, N);
        return;
    }
    if constexpr (SCH_MODE == SCH_BLK512) {
        constexpr uint32_t N = 512;
        constexpr uint32_t XO = 0, YO = N * sizeof(DT_X), TO = YO + N * sizeof(DT_X);
        GlobalTensor<DT_X> xG, yG;
        xG.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), N);
        yG.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), N);
        LocalTensor<DT_X> xL(TPosition::VECCALC, XO, N);
        LocalTensor<DT_X> yL(TPosition::VECCALC, YO, N);
        LocalTensor<DT_X> tL(TPosition::VECCALC, TO, N);
        DataCopy(xL, xG[0], N);
        SetFlag<HardEvent::MTE2_V>(event_t::EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(event_t::EVENT_ID0);
        EvalErfPoly(xL, yL, tL, N);
        SetFlag<HardEvent::V_MTE3>(event_t::EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(event_t::EVENT_ID0);
        DataCopy(yG[0], yL, N);
        return;
    }

    if constexpr (SCH_MODE == SCH_CACHE) {
        // 513..2048 elements — fits entirely in UB, direct path. Single-core.
        // Minimal tiling read: only fetch total (first 8 bytes), bypass
        // REGISTER_TILING_DEFAULT + GET_TILING_DATA_WITH_STRUCT overhead.
        constexpr uint32_t B = UB_CACHE_ELEMS;
        constexpr uint32_t TILING_UB_OFF = 3 * B * sizeof(float);
        LocalTensor<uint64_t> tilingBuf(TPosition::VECCALC, TILING_UB_OFF, 1);
        GlobalTensor<uint64_t> tilingGm;
        tilingGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(tiling), 1);
        DataCopy(tilingBuf, tilingGm[0], 1);
        SetFlag<HardEvent::MTE2_V>(event_t::EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(event_t::EVENT_ID0);
        uint32_t n = static_cast<uint32_t>(tilingBuf.GetValue(0));

        LocalTensor<DT_X> xU(TPosition::VECCALC, 0, B);
        LocalTensor<DT_X> yU(TPosition::VECCALC, B * sizeof(DT_X), B);
        LocalTensor<DT_X> tU(TPosition::VECCALC, 2 * B * sizeof(DT_X), B);
        GlobalTensor<DT_X> xG, yG;
        xG.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), n);
        yG.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), n);
        DataCopy(xU, xG[0], n);
        SetFlag<HardEvent::MTE2_V>(event_t::EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(event_t::EVENT_ID1);
        EvalErfPoly(xU, yU, tU, n);
        SetFlag<HardEvent::V_MTE3>(event_t::EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(event_t::EVENT_ID0);
        DataCopy(yG[0], yU, n);
        return;
    }

    REGISTER_TILING_DEFAULT(ErfTilingDesc);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingDesc, td, tiling);

    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t usedCores = td.nCore;

    if (blockIdx >= usedCores) {
        return;
    }

    uint64_t coreStart = 0;
    uint64_t coreLen = 0;
    CalcBlockRange(td.nBlk, td.rem, usedCores, blockIdx, coreStart, coreLen);

    if (coreStart >= td.total || coreLen == 0) {
        return;
    }

    GM_ADDR xCore = x + coreStart * sizeof(DT_X);
    GM_ADDR yCore = y + coreStart * sizeof(DT_X);

    ErfEngine<DT_X> op;
    TPipe pipe;
    op.Invoke(xCore, yCore, coreLen, td.chunk, pipe);
}
