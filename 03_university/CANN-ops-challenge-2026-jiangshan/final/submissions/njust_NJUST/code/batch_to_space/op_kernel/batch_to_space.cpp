// batch_to_space 鏍稿嚱鏁?- 缂栬瘧鏈熶笓鐢ㄨ矾寰?+ 閫氱敤鍚庡
#include "kernel_operator.h"
#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

#define BTS_GM_PAIR(TY, SRC_GM, DST_GM, SRC_PTR, DST_PTR) \
    AscendC::GlobalTensor<TY> SRC_GM; \
    AscendC::GlobalTensor<TY> DST_GM; \
    SRC_GM.SetGlobalBuffer((__gm__ TY*)(SRC_PTR)); \
    DST_GM.SetGlobalBuffer((__gm__ TY*)(DST_PTR))
#define BTS_LOCAL(TY, NAME, SLOT, COUNT) AscendC::LocalTensor<TY> NAME(AscendC::TPosition::VECCALC, SLOT, COUNT)
#define BTS_CORE_PAIR(IDX_NAME, NUM_NAME) \
    uint32_t IDX_NAME = AscendC::GetBlockIdx(); \
    uint32_t NUM_NAME = AscendC::GetBlockNum()
#define BTS_ROUTE_DO(KEY_VALUE, EXPR) \
    if constexpr (SCH_MODE == KEY_VALUE) { \
        EXPR; \
        return; \
    }
// ============================================================
// 宸ュ叿瀹?
// ============================================================
#define ALIGN_UP(v, a) (((v) + (a) - 1U) / (a) * (a))
#define CEIL_DIV(v, d) (((v) + (d) - 1U) / (d))

// ============================================================
// CASE6: half 1x1x16384 鈫?2x2x16384, 杩炵画鎷疯礉
// ============================================================
template <class T>
class Linear16384Emitter {
    static constexpr uint32_t kBlocks = (4U * 16384U * sizeof(T)) / 32U;
    static constexpr uint32_t kChunkBlocks = 2048U;
    static constexpr uint32_t kElemsPerBlock = 32U / sizeof(T);
public:
    __aicore__ inline void Launch(GM_ADDR inp, GM_ADDR out) {
        BTS_GM_PAIR(T, src, dst, inp, out);
        BTS_LOCAL(T, tmp, 0, kChunkBlocks * kElemsPerBlock);
        BTS_CORE_PAIR(core, cores);
        uint32_t quota = (kBlocks + cores - 1U) / cores;
        uint32_t block = core * quota;
        uint32_t left = block >= kBlocks ? 0U : kBlocks - block;
        if (left > quota) left = quota;
        AscendC::DataCopyParams cp;
        cp.blockCount = 1;
        cp.srcStride = 0;
        cp.dstStride = 0;
        while (left != 0U) {
            uint32_t take = left > kChunkBlocks ? kChunkBlocks : left;
            cp.blockLen = static_cast<uint16_t>(take);
            uint32_t elem = block * kElemsPerBlock;
            AscendC::DataCopy(tmp, src[elem], cp);
            AscendC::PipeBarrier<PIPE_MTE2>();
            AscendC::DataCopy(dst[elem], tmp, cp);
            AscendC::PipeBarrier<PIPE_MTE3>();
            block += take;
            left -= take;
        }
    }
};

template <class T>
__aicore__ inline void Solve_Case6_ContigCopy(GM_ADDR inp, GM_ADDR out) {
    Linear16384Emitter<T> emitter;
    emitter.Launch(inp, out);
}

// ============================================================
// CASE5: half 2x2x4096 鈫?4x4x4096, 閫氶亾鍒嗗潡
// ============================================================
template <class T>
class Deep4096Emitter {
    static constexpr uint32_t kH = 2U;
    static constexpr uint32_t kW = 2U;
    static constexpr uint32_t kC = 4096U;
    static constexpr uint32_t kOH = 4U;
    static constexpr uint32_t kOW = 4U;
    static constexpr uint32_t kSlice = 2048U;
    static constexpr uint32_t kSlices = 2U;
    static constexpr uint32_t kSliceBlocks = (kSlice * sizeof(T)) / 32U;
    static constexpr uint32_t kGapBlocks = ((kC - kSlice) * sizeof(T)) / 32U;
public:
    __aicore__ inline void Launch(GM_ADDR inp, GM_ADDR out) {
        BTS_GM_PAIR(T, src, dst, inp, out);
        BTS_LOCAL(T, row, 0, kOW * kSlice);
        AscendC::DataCopyParams load;
        AscendC::DataCopyParams store;
        load.blockCount = 1;
        load.blockLen = static_cast<uint16_t>(kSliceBlocks);
        load.srcStride = 0;
        load.dstStride = 0;
        store.blockCount = kOW;
        store.blockLen = static_cast<uint16_t>(kSliceBlocks);
        store.srcStride = 0;
        store.dstStride = static_cast<uint16_t>(kGapBlocks);
        BTS_CORE_PAIR(core, cores);
        for (uint32_t job = core; job < kOH * kSlices; job += cores) {
            uint32_t slice = job % kSlices;
            uint32_t oh = job / kSlices;
            uint32_t base = slice * kSlice;
            uint32_t ih = oh >> 1U;
            uint32_t phase = oh & 1U;
            uint32_t n0 = phase << 1U;
            CopyFour(src, row, load, n0, ih, base);
            AscendC::DataCopy(dst[oh * kOW * kC + base], row, store);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }
private:
    __aicore__ static inline void CopyFour(AscendC::GlobalTensor<T> &src, AscendC::LocalTensor<T> row,
                                           AscendC::DataCopyParams &load, uint32_t n0, uint32_t ih, uint32_t base) {
        uint32_t p0 = ((n0 * kH + ih) * kW) * kC + base;
        uint32_t p1 = (((n0 + 1U) * kH + ih) * kW) * kC + base;
        AscendC::DataCopy(row, src[p0], load);
        AscendC::DataCopy(row[kSlice], src[p1], load);
        AscendC::DataCopy(row[2U * kSlice], src[p0 + kC], load);
        AscendC::DataCopy(row[3U * kSlice], src[p1 + kC], load);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
};

template <class T>
__aicore__ inline void Solve_Case5_ChSplit(GM_ADDR inp, GM_ADDR out) {
    Deep4096Emitter<T> emitter;
    emitter.Launch(inp, out);
}

// ============================================================
// CASE3: float 4x6x32 鈫?8x12x32, 浜ら敊鎷疯礉
// ============================================================
template <class T>
class Small32PlaneEmitter {
    static constexpr uint32_t kH = 4U;
    static constexpr uint32_t kW = 6U;
    static constexpr uint32_t kC = 32U;
    static constexpr uint32_t kDstN = 5U;
    static constexpr uint32_t kOH = 8U;
    static constexpr uint32_t kOW = 12U;
    static constexpr uint32_t kChanBlocks = (kC * sizeof(T)) / 32U;
    static constexpr uint32_t kPlaneBlocks = (kH * kW * kC * sizeof(T)) / 32U;
    static constexpr uint32_t kOutRowBlocks = (kOW * kC * sizeof(T)) / 32U;
public:
    __aicore__ inline void Launch(GM_ADDR inp, GM_ADDR out) {
        BTS_GM_PAIR(T, src, dst, inp, out);
        BTS_LOCAL(T, a, 0, kH * kW * kC);
        BTS_LOCAL(T, b, 4096, kH * kW * kC);
        BTS_LOCAL(T, outPlane, 8192, kH * kOW * kC);
        AscendC::DataCopyParams load;
        AscendC::DataCopyParams weave;
        AscendC::DataCopyParams store;
        load.blockCount = 1; load.blockLen = static_cast<uint16_t>(kPlaneBlocks); load.srcStride = 0; load.dstStride = 0;
        weave.blockLen = static_cast<uint16_t>(kChanBlocks); weave.srcStride = 0; weave.dstStride = static_cast<uint16_t>(kChanBlocks);
        store.blockLen = static_cast<uint16_t>(kOutRowBlocks); store.srcStride = 0; store.dstStride = static_cast<uint16_t>(kOutRowBlocks);
        BTS_CORE_PAIR(core, cores);
        for (uint32_t job = core; job < kDstN * 2U; job += cores) {
            uint32_t phase = job & 1U;
            uint32_t n = job >> 1U;
            uint32_t n0 = phase * 2U * kDstN + n;
            AscendC::DataCopy(a, src[n0 * kH * kW * kC], load);
            AscendC::DataCopy(b, src[(n0 + kDstN) * kH * kW * kC], load);
            AscendC::PipeBarrier<PIPE_ALL>();
            weave.blockCount = static_cast<uint16_t>(kH * kW);
            AscendC::DataCopy(outPlane, a, weave);
            AscendC::DataCopy(outPlane[kC], b, weave);
            AscendC::PipeBarrier<PIPE_ALL>();
            store.blockCount = static_cast<uint16_t>(kH);
            AscendC::DataCopy(dst[(n * kOH + phase) * kOW * kC], outPlane, store);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }
};

template <class T>
__aicore__ inline void Solve_Case3_InterLeave(GM_ADDR inp, GM_ADDR out) {
    Small32PlaneEmitter<T> emitter;
    emitter.Launch(inp, out);
}

// ============================================================
// F32 block2 no-crop row stitcher used by CASE0 and CASE2.
// ============================================================
template <class T, uint32_t kH, uint32_t kW, uint32_t kC, uint32_t kDstN,
          uint32_t kOH, uint32_t kOW, uint32_t kBSlot, uint32_t kOSlot, bool kAllPipe>
class B2RowEmitter {
    static constexpr uint32_t kChanBlocks = (kC * sizeof(T)) / 32U;
    static constexpr uint32_t kSrcRowBlocks = (kW * kC * sizeof(T)) / 32U;
    static constexpr uint32_t kDstRowBlocks = (kOW * kC * sizeof(T)) / 32U;
    static constexpr uint32_t kInElems = kW * kC;
public:
    __aicore__ inline void Launch(GM_ADDR inp, GM_ADDR out) {
        BTS_GM_PAIR(T, src, dst, inp, out);
        BTS_LOCAL(T, left, 0, kInElems);
        BTS_LOCAL(T, right, kBSlot, kInElems);
        BTS_LOCAL(T, merged, kOSlot, kOW * kC);
        AscendC::DataCopyParams load;
        AscendC::DataCopyParams interleave;
        AscendC::DataCopyParams store;
        load.blockCount = 1;
        load.blockLen = static_cast<uint16_t>(kSrcRowBlocks);
        load.srcStride = 0;
        load.dstStride = 0;
        interleave.blockLen = static_cast<uint16_t>(kChanBlocks);
        interleave.srcStride = 0;
        interleave.dstStride = static_cast<uint16_t>(kChanBlocks);
        store.blockCount = 1;
        store.blockLen = static_cast<uint16_t>(kDstRowBlocks);
        store.srcStride = 0;
        store.dstStride = 0;
        BTS_CORE_PAIR(core, cores);
        for (uint32_t job = core; job < kDstN * 2U * kH; job += cores) {
            uint32_t row = job % kH;
            uint32_t pack = job / kH;
            uint32_t phase = pack & 1U;
            uint32_t n = pack >> 1U;
            uint32_t n0 = phase * 2U * kDstN + n;
            LoadPair(src, left, right, load, n0, row);
            MergeRows(left, right, merged, interleave);
            if constexpr (kAllPipe) {
                AscendC::PipeBarrier<PIPE_ALL>();
            } else {
                AscendC::PipeBarrier<PIPE_MTE2>();
            }
            AscendC::DataCopy(dst[(n * kOH + phase + row * 2U) * kOW * kC], merged, store);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }
private:
    __aicore__ static inline void LoadPair(AscendC::GlobalTensor<T> &src, AscendC::LocalTensor<T> a,
                                           AscendC::LocalTensor<T> b, AscendC::DataCopyParams &cp,
                                           uint32_t n0, uint32_t row) {
        uint32_t base = row * kW * kC;
        AscendC::DataCopy(a, src[n0 * kH * kW * kC + base], cp);
        AscendC::DataCopy(b, src[(n0 + kDstN) * kH * kW * kC + base], cp);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    __aicore__ static inline void MergeRows(AscendC::LocalTensor<T> a, AscendC::LocalTensor<T> b,
                                            AscendC::LocalTensor<T> out, AscendC::DataCopyParams &cp) {
        cp.blockCount = static_cast<uint16_t>(kW);
        AscendC::DataCopy(out, a, cp);
        AscendC::DataCopy(out[kC], b, cp);
    }
};

template <class T, uint32_t kH, uint32_t kW, uint32_t kC, uint32_t kDstN,
          uint32_t kOH, uint32_t kOW, uint32_t kRows>
class B2BandEmitter {
    static constexpr uint32_t kChanBlocks = (kC * sizeof(T)) / 32U;
    static constexpr uint32_t kSrcRowBlocks = (kW * kC * sizeof(T)) / 32U;
    static constexpr uint32_t kDstRowBlocks = (kOW * kC * sizeof(T)) / 32U;
    static constexpr uint32_t kBands = (kH + kRows - 1U) / kRows;
    static constexpr uint32_t kInElems = kRows * kW * kC;
    static constexpr uint32_t kOutElems = kRows * kOW * kC;
    static constexpr uint32_t kInBytes = ALIGN_UP(kInElems * sizeof(T), 512U);
public:
    __aicore__ inline void Launch(GM_ADDR inp, GM_ADDR out) {
        BTS_GM_PAIR(T, src, dst, inp, out);
        BTS_LOCAL(T, front, 0, kInElems);
        BTS_LOCAL(T, back, kInBytes, kInElems);
        BTS_LOCAL(T, mixed, 2U * kInBytes, kOutElems);

        AscendC::DataCopyParams load;
        load.blockCount = 1;
        load.blockLen = 0;
        load.srcStride = 0;
        load.dstStride = 0;

        AscendC::DataCopyParams scatter;
        scatter.blockLen = static_cast<uint16_t>(kChanBlocks);
        scatter.srcStride = 0;
        scatter.dstStride = static_cast<uint16_t>(kChanBlocks);

        AscendC::DataCopyParams store;
        store.blockLen = static_cast<uint16_t>(kDstRowBlocks);
        store.srcStride = 0;
        store.dstStride = static_cast<uint16_t>(kDstRowBlocks);

        BTS_CORE_PAIR(core, cores);
        for (uint32_t task = core; task < kDstN * 2U * kBands; task += cores) {
            uint32_t band = task % kBands;
            uint32_t lane = task / kBands;
            uint32_t phase = lane & 1U;
            uint32_t n = lane >> 1U;
            uint32_t row0 = band * kRows;
            uint32_t rows = kH - row0;
            if (rows > kRows) rows = kRows;

            uint32_t n0 = phase * 2U * kDstN + n;
            uint32_t base = row0 * kW * kC;
            load.blockLen = static_cast<uint16_t>(rows * kSrcRowBlocks);
            AscendC::DataCopy(front, src[n0 * kH * kW * kC + base], load);
            AscendC::DataCopy(back, src[(n0 + kDstN) * kH * kW * kC + base], load);
            AscendC::PipeBarrier<PIPE_ALL>();

            scatter.blockCount = static_cast<uint16_t>(rows * kW);
            AscendC::DataCopy(mixed, front, scatter);
            AscendC::DataCopy(mixed[kC], back, scatter);
            AscendC::PipeBarrier<PIPE_ALL>();

            store.blockCount = static_cast<uint16_t>(rows);
            uint32_t dstRow = n * kOH + phase + row0 * 2U;
            AscendC::DataCopy(dst[dstRow * kOW * kC], mixed, store);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }
};

template <class T>
class Big128PadSpreadEmitter {
    static constexpr uint32_t kH = 28U;
    static constexpr uint32_t kW = 28U;
    static constexpr uint32_t kC = 128U;
    static constexpr uint32_t kDstN = 2U;
    static constexpr uint32_t kOH = 56U;
    static constexpr uint32_t kOW = 56U;
    static constexpr uint32_t kRowElems = kOW * kC;
    static constexpr uint32_t kRowBytes = kRowElems * sizeof(T);
    static constexpr uint32_t kCBytes = kC * sizeof(T);
    static constexpr uint32_t kCBlocks = kCBytes / 32U;
    static constexpr uint32_t kPaddedRowElems = ALIGN_UP(kRowElems, 1024U);
    static constexpr event_t kBuf0Ready = static_cast<event_t>(0);
    static constexpr event_t kBuf1Ready = static_cast<event_t>(1);
public:
    __aicore__ inline void Launch(GM_ADDR inp, GM_ADDR out) {
        BTS_GM_PAIR(T, src, dst, inp, out);
        BTS_LOCAL(T, line0, 0, kPaddedRowElems);
        BTS_LOCAL(T, line1, kPaddedRowElems * sizeof(T), kPaddedRowElems);

        AscendC::DataCopyExtParams spread;
        spread.blockCount = static_cast<uint16_t>(kW);
        spread.blockLen = kCBytes;
        spread.srcStride = 0;
        spread.dstStride = kCBlocks;
        spread.rsv = 0;
        AscendC::DataCopyPadExtParams<T> pad;
        pad.isPad = false;
        pad.leftPadding = 0;
        pad.rightPadding = 0;
        pad.paddingValue = static_cast<T>(0);

        BTS_CORE_PAIR(core, cores);
        constexpr uint32_t kTasks = kDstN * kOH;
        uint32_t first = core * ((kTasks + cores - 1U) / cores);
        uint32_t limit = first + ((kTasks + cores - 1U) / cores);
        if (first >= kTasks) return;
        if (limit > kTasks) limit = kTasks;

        PackRow(src, line0, first, spread, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(kBuf0Ready);
        for (uint32_t task = first; task < limit; ++task) {
            bool use0 = ((task - first) & 1U) == 0U;
            event_t ready = use0 ? kBuf0Ready : kBuf1Ready;
            event_t nextReady = use0 ? kBuf1Ready : kBuf0Ready;
            AscendC::LocalTensor<T> cur = use0 ? line0 : line1;
            AscendC::LocalTensor<T> next = use0 ? line1 : line0;
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(ready);
            if (task + 1U < limit) {
                PackRow(src, next, task + 1U, spread, pad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(nextReady);
            }
            AscendC::DataCopy(dst[static_cast<uint64_t>(task) * kRowElems], cur, kRowElems);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ready);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ready);
        }
    }
private:
    __aicore__ static inline void PackRow(AscendC::GlobalTensor<T> &src,
                                          AscendC::LocalTensor<T> row,
                                          uint32_t task,
                                          AscendC::DataCopyExtParams &cp,
                                          AscendC::DataCopyPadExtParams<T> &pad) {
        uint32_t n = task / kOH;
        uint32_t oh = task - n * kOH;
        uint32_t ih = oh >> 1U;
        uint32_t phase = oh & 1U;
        uint32_t n0 = phase * 2U * kDstN + n;
        uint64_t base0 = (static_cast<uint64_t>(n0) * kH + ih) * kW * kC;
        uint64_t base1 = base0 + static_cast<uint64_t>(kDstN) * kH * kW * kC;
        AscendC::DataCopyPad(row, src[base0], cp, pad);
        AscendC::DataCopyPad(row[kC], src[base1], cp, pad);
    }
};

template <class T>
__aicore__ inline void Solve_Case2_RowIL(GM_ADDR inp, GM_ADDR out) {
    B2BandEmitter<T, 14U, 14U, 64U, 4U, 28U, 28U, 7U> emitter;
    emitter.Launch(inp, out);
}

template <class T>
__aicore__ inline void Solve_Case0_RowIL(GM_ADDR inp, GM_ADDR out) {
    Big128PadSpreadEmitter<T> emitter;
    emitter.Launch(inp, out);
}

// ============================================================
// CASE1: float 10x15x5 鈫?17x26x5, 闈炲榻怗ather (鏈塩rop)
// ============================================================
template <class T>
class Tiny5CropEmitter {
    static constexpr uint32_t kH = 10U;
    static constexpr uint32_t kW = 15U;
    static constexpr uint32_t kC = 5U;
    static constexpr uint32_t kOH = 17U;
    static constexpr uint32_t kOW = 26U;
    static constexpr uint32_t kEvenCols = (kOW + 1U) >> 1U;
    static constexpr uint32_t kOddCols = kOW >> 1U;
    static constexpr uint32_t kChanBytes = kC * sizeof(T);
    static constexpr uint32_t kGroupBytes = ALIGN_UP(kEvenCols * kChanBytes, 32U);
    static constexpr uint32_t kGroupElems = kGroupBytes / sizeof(T);
    static constexpr uint32_t kInElems = 2U * kGroupElems;
    static constexpr uint32_t kOutElems = kOW * kC;
    static constexpr uint32_t kPairElems = 2U * kC;
    static constexpr uint32_t kSeedPairs = 4U;
    static constexpr uint32_t kSeedElems = kSeedPairs * kPairElems;
    static constexpr uint32_t kFullSeeds = kOutElems / kSeedElems;
    static constexpr uint32_t kTailElems = ALIGN_UP(kOutElems - kFullSeeds * kSeedElems, 8U);
    static constexpr uint32_t kIndexElems = kFullSeeds * kSeedElems + kTailElems;
    static constexpr event_t kIndexReady = static_cast<event_t>(6);
public:
    __aicore__ inline void Launch(GM_ADDR inp, GM_ADDR out) {
        BTS_GM_PAIR(T, src, dst, inp, out);
        BTS_LOCAL(T, inTile, 0, kInElems);
        BTS_LOCAL(T, outRow, 1024, kOutElems);
        AscendC::LocalTensor<int32_t> indexBuild(AscendC::TPosition::VECCALC, 2048, kIndexElems);
        AscendC::LocalTensor<uint32_t> indexUse(AscendC::TPosition::VECCALC, 2048, kIndexElems);
        MakeIndex(indexBuild);
        AscendC::DataCopyPadExtParams<T> pad;
        pad.isPad = false; pad.leftPadding = 0; pad.rightPadding = 0; pad.paddingValue = static_cast<T>(0);
        AscendC::DataCopyExtParams evenLoad, oddLoad, store;
        evenLoad.blockCount = 1; evenLoad.blockLen = kEvenCols * kChanBytes; evenLoad.srcStride = 0; evenLoad.dstStride = 0; evenLoad.rsv = 0;
        oddLoad.blockCount = 1; oddLoad.blockLen = kOddCols * kChanBytes; oddLoad.srcStride = 0; oddLoad.dstStride = 0; oddLoad.rsv = 0;
        store.blockCount = 1; store.blockLen = kOW * kChanBytes; store.srcStride = 0; store.dstStride = 0; store.rsv = 0;
        BTS_CORE_PAIR(core, cores);
        for (uint32_t oh = core; oh < kOH; oh += cores) {
            uint32_t ph = oh + 2U;
            uint32_t ih = ph >> 1U;
            uint32_t phase = ph & 1U;
            uint32_t evenN = phase * 2U + 1U;
            uint32_t oddN = phase * 2U;
            AscendC::DataCopyPad(inTile, src[((evenN * kH + ih) * kW + 1U) * kC], evenLoad, pad);
            AscendC::DataCopyPad(inTile[kGroupElems], src[((oddN * kH + ih) * kW + 2U) * kC], oddLoad, pad);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Gather(outRow, inTile, indexUse, 0, kOutElems);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(dst[oh * kOW * kC], outRow, store);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }
private:
    __aicore__ static inline void MakeIndex(AscendC::LocalTensor<int32_t> index) {
        for (uint32_t pair = 0; pair < kSeedPairs; ++pair) {
            uint32_t elemBase = pair * kPairElems;
            uint32_t byteBase = pair * kC * sizeof(T);
            for (uint32_t c = 0; c < kC; ++c) {
                index.SetValue(elemBase + c, static_cast<int32_t>(byteBase + c * sizeof(T)));
                index.SetValue(elemBase + kC + c,
                               static_cast<int32_t>(kGroupElems * sizeof(T) +
                                                    byteBase + c * sizeof(T)));
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(kIndexReady);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(kIndexReady);
        for (uint32_t seed = 1; seed <= kFullSeeds; ++seed) {
            uint32_t count = seed == kFullSeeds ? kTailElems : kSeedElems;
            AscendC::Adds(index[seed * kSeedElems], index[(seed - 1U) * kSeedElems],
                          static_cast<int32_t>(kSeedPairs * kC * sizeof(T)), count);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
};

template <class T>
__aicore__ inline void Solve_Case1_CropGather(GM_ADDR inp, GM_ADDR out) {
    Tiny5CropEmitter<T> emitter;
    emitter.Launch(inp, out);
}

// ============================================================
// CASE4: half 128x128x65 鈫?254x254x65, 闈炲榻怗ather (鏈塩rop, 棰勮绠楀亸绉?
// ============================================================
template <class T>
class Odd65StripeEmitter {
    static constexpr uint32_t kC = 65U;
    static constexpr uint32_t kOH = 254U;
    static constexpr uint32_t kOW = 254U;
    static constexpr uint32_t kHalfW = 127U;
    static constexpr uint32_t kLaneW = 64U;
    static constexpr uint32_t kLaneElems = kLaneW * kC;
    static constexpr uint32_t kInElems = 2U * kLaneElems;
    static constexpr uint32_t kOutElems = kHalfW * kC;
    static constexpr uint32_t kSlotElems = 16576U;
    static constexpr uint32_t kSlotBytes = kSlotElems * sizeof(T);
    static constexpr uint32_t kIndexByteOffset = 2U * kSlotBytes;
    static constexpr uint32_t kPairElems = 2U * kC;
    static constexpr uint32_t kSeedPairs = 4U;
    static constexpr uint32_t kSeedElems = kSeedPairs * kPairElems;
    static constexpr uint32_t kRepeatCount = kOutElems / kSeedElems;
    static constexpr uint32_t kTailElems = ALIGN_UP(kOutElems - kRepeatCount * kSeedElems, 8U);
    static constexpr uint32_t kVectorIndexElems = kRepeatCount * kSeedElems + kTailElems;
    static constexpr event_t kLoadA = static_cast<event_t>(0);
    static constexpr event_t kLoadB = static_cast<event_t>(1);
    static constexpr event_t kVecA = static_cast<event_t>(2);
    static constexpr event_t kVecB = static_cast<event_t>(3);
    static constexpr event_t kFreeA = static_cast<event_t>(4);
    static constexpr event_t kFreeB = static_cast<event_t>(5);
    static constexpr event_t kIndexReady = static_cast<event_t>(6);

    static_assert(kInElems + kOutElems <= kSlotElems, "odd65 slot overflow");
    static_assert(kIndexByteOffset + kVectorIndexElems * sizeof(uint32_t) <= 184U * 1024U,
                  "odd65 index overflow");

public:
    __aicore__ inline void Launch(GM_ADDR inp, GM_ADDR out) {
        BTS_GM_PAIR(T, src, dst, inp, out);
        BTS_LOCAL(T, laneA, 0, kSlotElems);
        BTS_LOCAL(T, laneB, kSlotBytes, kSlotElems);
        AscendC::LocalTensor<int32_t> idxI(AscendC::TPosition::VECCALC,
                                           kIndexByteOffset, kVectorIndexElems);
        AscendC::LocalTensor<uint32_t> idxU(AscendC::TPosition::VECCALC,
                                            kIndexByteOffset, kVectorIndexElems);
        BuildIndex(idxI);

        AscendC::DataCopyExtParams store;
        store.blockCount = 1;
        store.blockLen = kOutElems * sizeof(T);
        store.srcStride = 0;
        store.dstStride = 0;
        store.rsv = 0;

        BTS_CORE_PAIR(core, cores);
        constexpr uint32_t kJobs = kOH * 2U;
        uint32_t base = kJobs / cores;
        uint32_t rem = kJobs - base * cores;
        uint32_t first = core * base + (core < rem ? core : rem);
        uint32_t count = base + (core < rem ? 1U : 0U);
        if (count == 0U) return;

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(kFreeA);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(kFreeB);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kFreeA);
        LoadPair(src, laneA, first);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kLoadA);

        for (uint32_t step = 0; step < count; ++step) {
            bool useA = (step & 1U) == 0U;
            AscendC::LocalTensor<T> cur = useA ? laneA : laneB;
            AscendC::LocalTensor<T> spare = useA ? laneB : laneA;
            event_t loadEv = useA ? kLoadA : kLoadB;
            event_t vecEv = useA ? kVecA : kVecB;
            event_t freeEv = useA ? kFreeA : kFreeB;
            uint32_t job = first + step;
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(loadEv);
            AscendC::Gather(cur[kInElems], cur, idxU, 0U, kOutElems);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vecEv);

            if (step + 1U < count) {
                bool nextA = ((step + 1U) & 1U) == 0U;
                event_t nextFree = nextA ? kFreeA : kFreeB;
                event_t nextLoad = nextA ? kLoadA : kLoadB;
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(nextFree);
                LoadPair(src, spare, job + 1U);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(nextLoad);
            }

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vecEv);
            AscendC::DataCopyPad(dst[OutputOffset(job)], cur[kInElems], store);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(freeEv);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kFreeA);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kFreeB);
    }

private:
    __aicore__ static inline uint64_t OutputOffset(uint32_t job) {
        uint32_t row = job >> 1U;
        uint32_t side = job & 1U;
        return (static_cast<uint64_t>(row) * kOW + side * kHalfW) * kC;
    }

    __aicore__ static inline void LoadPair(AscendC::GlobalTensor<T> &src,
                                           AscendC::LocalTensor<T> dst,
                                           uint32_t job) {
        constexpr uint32_t kSrcH = 128U;
        constexpr uint32_t kSrcW = 128U;
        uint32_t row = job >> 1U;
        uint32_t side = job & 1U;
        uint32_t phaseRow = row + 1U;
        uint32_t ih = phaseRow >> 1U;
        uint32_t phase = phaseRow & 1U;
        uint32_t aN = (phase << 1U) + (side == 0U ? 1U : 0U);
        uint32_t bN = (phase << 1U) + (side == 0U ? 0U : 1U);
        uint32_t aW = side == 0U ? 0U : 64U;
        uint32_t bW = side == 0U ? 1U : 64U;
        uint64_t aOff = ((static_cast<uint64_t>(aN) * kSrcH + ih) * kSrcW + aW) * kC;
        uint64_t bOff = ((static_cast<uint64_t>(bN) * kSrcH + ih) * kSrcW + bW) * kC;
        AscendC::DataCopy(dst, src[aOff], kLaneElems);
        AscendC::DataCopy(dst[kLaneElems], src[bOff], kLaneElems);
    }

    __aicore__ static inline void BuildIndex(AscendC::LocalTensor<int32_t> idx) {
        for (uint32_t seed = 0; seed < kSeedPairs; ++seed) {
            uint32_t base = seed * kPairElems;
            uint32_t bytes = seed * kC * sizeof(T);
            for (uint32_t c = 0; c < kC; ++c) {
                idx.SetValue(base + c, static_cast<int32_t>(bytes + c * sizeof(T)));
                idx.SetValue(base + kC + c,
                             static_cast<int32_t>(kLaneElems * sizeof(T) + bytes + c * sizeof(T)));
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(kIndexReady);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(kIndexReady);
        for (uint32_t rep = 1; rep <= kRepeatCount; ++rep) {
            uint32_t n = (rep == kRepeatCount) ? kTailElems : kSeedElems;
            AscendC::Adds(idx[rep * kSeedElems], idx[(rep - 1U) * kSeedElems],
                          static_cast<int32_t>(kSeedPairs * kC * sizeof(T)), n);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
};

template <class T>
__aicore__ inline void Solve_Case4_HalfGather(GM_ADDR inp, GM_ADDR out, GM_ADDR tiling) {
    (void)tiling;
    Odd65StripeEmitter<T> emitter;
    emitter.Launch(inp, out);
}

// ============================================================
// CASE7: half 10x512x256 鈫?20x1024x256, 璺ㄦ浜ら敊 + 鍒嗗垪
// ============================================================
template <class T>
class Wide256ColumnEmitter {
    static constexpr uint32_t kH = 10U;
    static constexpr uint32_t kW = 512U;
    static constexpr uint32_t kC = 256U;
    static constexpr uint32_t kOH = 20U;
    static constexpr uint32_t kOW = 1024U;
    static constexpr uint32_t kCols = 64U;
    static constexpr uint32_t kTiles = CEIL_DIV(kW, kCols);
    static constexpr uint32_t kChanBlocks = (kC * sizeof(T)) / 32U;
    static constexpr uint32_t kInElems = kCols * kC;
    static constexpr uint32_t kOutElems = kCols * 2U * kC;
    static constexpr uint32_t kOutBytes = kOutElems * sizeof(T);

public:
    __aicore__ inline void Launch(GM_ADDR inp, GM_ADDR out) {
        BTS_GM_PAIR(T, src, dst, inp, out);
        BTS_LOCAL(T, outTile, 0, kOutElems);
        BTS_LOCAL(T, inTile, kOutBytes, kInElems);
        AscendC::DataCopyParams load;
        AscendC::DataCopyParams place;
        AscendC::DataCopyParams store;
        load.blockCount = 1;
        load.blockLen = static_cast<uint16_t>(kCols * kChanBlocks);
        load.srcStride = 0;
        load.dstStride = 0;
        place.blockLen = static_cast<uint16_t>(kChanBlocks);
        place.srcStride = 0;
        place.dstStride = static_cast<uint16_t>(kChanBlocks);
        store.blockCount = 1;
        store.blockLen = static_cast<uint16_t>(kCols * 2U * kChanBlocks);
        store.srcStride = 0;
        store.dstStride = 0;

        BTS_CORE_PAIR(core, cores);
        for (uint32_t job = core; job < kOH * kTiles; job += cores) {
            uint32_t tile = job % kTiles;
            uint32_t oh = job / kTiles;
            uint32_t col = tile * kCols;
            uint32_t ih = oh >> 1U;
            uint32_t phase = oh & 1U;
            uint32_t n0 = phase << 1U;
            CopyLane(src, inTile, outTile, load, place, n0, ih, col, 0U);
            CopyLane(src, inTile, outTile, load, place, n0 + 1U, ih, col, kC);
            AscendC::DataCopy(dst[(oh * kOW + col * 2U) * kC], outTile, store);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }

private:
    __aicore__ static inline void CopyLane(AscendC::GlobalTensor<T> &src,
                                           AscendC::LocalTensor<T> tmp,
                                           AscendC::LocalTensor<T> out,
                                           AscendC::DataCopyParams &load,
                                           AscendC::DataCopyParams &place,
                                           uint32_t n, uint32_t ih, uint32_t col, uint32_t dstShift) {
        uint32_t srcOff = ((n * kH + ih) * kW + col) * kC;
        AscendC::DataCopy(tmp, src[srcOff], load);
        AscendC::PipeBarrier<PIPE_ALL>();
        place.blockCount = static_cast<uint16_t>(kCols);
        AscendC::DataCopy(out[dstShift], tmp, place);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
};

template <class T>
class Wide256ScatterEmitter {
    static constexpr uint32_t kH = 10U;
    static constexpr uint32_t kW = 512U;
    static constexpr uint32_t kC = 256U;
    static constexpr uint32_t kOW = 1024U;
    static constexpr uint32_t kWorkers = 20U;
    static constexpr uint32_t kTileElems = 16384U;
    static constexpr uint32_t kTileCols = kTileElems / kC;
    static constexpr uint32_t kInputRows = 4U * kH;
    static constexpr uint32_t kRowsPerWorker = kInputRows / kWorkers;
    static constexpr event_t kLane0 = static_cast<event_t>(0);
    static constexpr event_t kLane1 = static_cast<event_t>(1);
public:
    __aicore__ inline void Launch(GM_ADDR inp, GM_ADDR out) {
        BTS_GM_PAIR(T, src, dst, inp, out);
        AscendC::LocalMemAllocator<AscendC::Hardware::UB> arena;
        AscendC::LocalTensor<T> bank0 =
            arena.Alloc<AscendC::TPosition::VECCALC, T>(kTileElems);
        AscendC::LocalTensor<T> bank1 =
            arena.Alloc<AscendC::TPosition::VECCALC, T>(kTileElems);

        uint32_t core = AscendC::GetBlockIdx();
        uint32_t firstRow = core * kRowsPerWorker;
        if (firstRow >= kInputRows) return;
        uint32_t rowLimit = kInputRows - firstRow;
        if (rowLimit > kRowsPerWorker) rowLimit = kRowsPerWorker;
        for (uint32_t r = 0; r < rowLimit; ++r) {
            uint32_t row = firstRow + r;
            uint32_t inN = row / kH;
            uint32_t ih = row - inN * kH;
            uint32_t oh = (ih << 1U) + (inN >> 1U);
            uint32_t horizontalPhase = inN & 1U;
            uint64_t srcBase = static_cast<uint64_t>(row) * kW * kC;
            uint64_t dstBase = (static_cast<uint64_t>(oh) * kOW + horizontalPhase) * kC;

            uint32_t doneCols = 0;
            uint32_t activeCols = kW > kTileCols ? kTileCols : kW;
            AscendC::DataCopy(bank0, src[srcBase], activeCols * kC);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(kLane0);
            uint32_t bank = 0;
            while (doneCols < kW) {
                event_t ready = bank == 0U ? kLane0 : kLane1;
                event_t nextReady = bank == 0U ? kLane1 : kLane0;
                AscendC::LocalTensor<T> current = bank == 0U ? bank0 : bank1;
                AscendC::LocalTensor<T> next = bank == 0U ? bank1 : bank0;
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(ready);

                uint32_t nextDone = doneCols + activeCols;
                uint32_t nextCols = 0;
                if (nextDone < kW) {
                    nextCols = kW - nextDone;
                    if (nextCols > kTileCols) nextCols = kTileCols;
                    uint64_t nextSource = srcBase + static_cast<uint64_t>(nextDone) * kC;
                    AscendC::DataCopy(next, src[nextSource], nextCols * kC);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(nextReady);
                }

                AscendC::DataCopyExtParams scatter{
                    static_cast<uint16_t>(activeCols),
                    static_cast<uint32_t>(kC * sizeof(T)),
                    0,
                    static_cast<uint32_t>(kC * sizeof(T)),
                    0};
                uint64_t target = dstBase + static_cast<uint64_t>(doneCols) * 2U * kC;
                AscendC::DataCopyPad(dst[target], current, scatter);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ready);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ready);
                doneCols = nextDone;
                activeCols = nextCols;
                bank ^= 1U;
            }
        }
    }
};

template <class T>
__aicore__ inline void Solve_Case7_StridedIL(GM_ADDR inp, GM_ADDR out) {
    Wide256ScatterEmitter<T> emitter;
    emitter.Launch(inp, out);
}

// ============================================================
// CASE8: half 10x512x64 鈫?40x1535x64, blk=4 crop, 鍒嗙粍浜ら敊
// ============================================================
template <class T>
class Crop64QuadEmitter {
    static constexpr uint32_t kH = 10U;
    static constexpr uint32_t kW = 512U;
    static constexpr uint32_t kC = 64U;
    static constexpr uint32_t kB = 4U;
    static constexpr uint32_t kOH = 40U;
    static constexpr uint32_t kOW = 1535U;
    static constexpr uint32_t kLeft = 513U;
    static constexpr uint32_t kCols = 512U;
    static constexpr uint32_t kTiles = 3U;
    static constexpr uint32_t kGroupCols = kCols / kB;
    static constexpr uint32_t kChanBytes = kC * sizeof(T);
    static constexpr uint32_t kChanBlocks = kChanBytes / 32U;
    static constexpr uint32_t kGroupBytes = kGroupCols * kChanBytes;
    static constexpr uint32_t kGroupElems = kGroupBytes / sizeof(T);
    static constexpr uint32_t kOutElems = kCols * kC;
    static constexpr uint32_t kPairElems = 2U * kGroupElems;

public:
    __aicore__ inline void Launch(GM_ADDR inp, GM_ADDR out) {
        BTS_GM_PAIR(T, src, dst, inp, out);
        BTS_LOCAL(T, outTile, 0, kOutElems);
        BTS_LOCAL(T, pair01, 65536, kPairElems);
        BTS_LOCAL(T, pair23, 98304, kPairElems);
        AscendC::DataCopyParams load;
        AscendC::DataCopyParams place;
        AscendC::DataCopyParams store;
        load.blockCount = 1;
        load.blockLen = 0;
        load.srcStride = 0;
        load.dstStride = 0;
        place.blockLen = static_cast<uint16_t>(kChanBlocks);
        place.srcStride = 0;
        place.dstStride = static_cast<uint16_t>((kB - 1U) * kChanBlocks);
        store.blockCount = 1;
        store.blockLen = 0;
        store.srcStride = 0;
        store.dstStride = 0;

        BTS_CORE_PAIR(core, cores);
        for (uint32_t oh = core; oh < kOH; oh += cores) {
            uint32_t ih = oh / kB;
            uint32_t phase = oh - ih * kB;
            uint32_t firstN = phase * kB;
            for (uint32_t tile = 0; tile < kTiles; ++tile) {
                uint32_t outCol = tile * kCols;
                uint32_t width = kOW - outCol;
                if (width > kCols) width = kCols;
                uint32_t inCol = (outCol + kLeft) / kB;
                uint32_t shiftedCols = width == kCols ? kGroupCols : kGroupCols - 1U;
                LoadQuad(src, pair01, pair23, load, firstN, ih, inCol, shiftedCols);
                ScatterQuad(outTile, pair01, pair23, place, shiftedCols);
                store.blockLen = static_cast<uint16_t>((width * kChanBytes) / 32U);
                AscendC::DataCopy(dst[(oh * kOW + outCol) * kC], outTile, store);
                if (tile + 1U < kTiles || oh + cores < kOH) {
                    AscendC::PipeBarrier<PIPE_MTE3>();
                }
            }
        }
    }

private:
    __aicore__ static inline void LoadQuad(AscendC::GlobalTensor<T> &src,
                                           AscendC::LocalTensor<T> p01,
                                           AscendC::LocalTensor<T> p23,
                                           AscendC::DataCopyParams &cp,
                                           uint32_t n0, uint32_t ih, uint32_t inCol, uint32_t headCols) {
        cp.blockLen = static_cast<uint16_t>((headCols * kChanBytes) / 32U);
        AscendC::DataCopy(p01, src[((n0 * kH + ih) * kW + inCol + 1U) * kC], cp);
        cp.blockLen = static_cast<uint16_t>((kGroupCols * kChanBytes) / 32U);
        AscendC::DataCopy(p01[kGroupElems], src[(((n0 + 1U) * kH + ih) * kW + inCol) * kC], cp);
        AscendC::DataCopy(p23, src[(((n0 + 2U) * kH + ih) * kW + inCol) * kC], cp);
        AscendC::DataCopy(p23[kGroupElems], src[(((n0 + 3U) * kH + ih) * kW + inCol) * kC], cp);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ static inline void ScatterQuad(AscendC::LocalTensor<T> out,
                                              AscendC::LocalTensor<T> p01,
                                              AscendC::LocalTensor<T> p23,
                                              AscendC::DataCopyParams &cp,
                                              uint32_t headCols) {
        cp.blockCount = static_cast<uint16_t>(headCols);
        AscendC::DataCopy(out[3U * kC], p01, cp);
        cp.blockCount = static_cast<uint16_t>(kGroupCols);
        AscendC::DataCopy(out, p01[kGroupElems], cp);
        AscendC::DataCopy(out[kC], p23, cp);
        AscendC::DataCopy(out[2U * kC], p23[kGroupElems], cp);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
};

template <class T>
__aicore__ inline void Solve_Case8_Blk4Crop(GM_ADDR inp, GM_ADDR out) {
    Crop64QuadEmitter<T> emitter;
    emitter.Launch(inp, out);
}

// ============================================================
// CASE9: half 1024x6x32 鈫?2048x12x32 鏍稿績鐡堕 鈥?鍒嗗潡浜ら敊
// ============================================================
template <class T>
class Strip32TileEmitter {
    static constexpr uint32_t kSrcH = 1024U;
    static constexpr uint32_t kSrcW = 6U;
    static constexpr uint32_t kChan = 32U;
    static constexpr uint32_t kDstN = 4U;
    static constexpr uint32_t kDstH = 2048U;
    static constexpr uint32_t kDstW = 12U;
    static constexpr uint32_t kBlock = 2U;
    static constexpr uint32_t kRows = 52U;
    static constexpr uint32_t kTileCount = (kSrcH + kRows - 1U) / kRows;
    static constexpr uint32_t kBlk32 = (kChan * sizeof(T)) / 32U;
    static constexpr uint32_t kInRowBlocks = (kSrcW * kChan * sizeof(T)) / 32U;
    static constexpr uint32_t kOutRowBlocks = (kDstW * kChan * sizeof(T)) / 32U;
    static constexpr uint32_t kInElems = kRows * kSrcW * kChan;
    static constexpr uint32_t kOutElems = kRows * kDstW * kChan;
    static constexpr uint32_t kInBytesPadded = ((kInElems * sizeof(T) + 511U) / 512U) * 512U;

public:
    __aicore__ inline void Launch(GM_ADDR inp, GM_ADDR out) {
        BTS_GM_PAIR(T, src, dst, inp, out);
        BTS_LOCAL(T, left, 0, kInElems);
        BTS_LOCAL(T, right, kInBytesPadded, kInElems);
        BTS_LOCAL(T, woven, 2U * kInBytesPadded, kOutElems);
        AscendC::DataCopyParams load;
        AscendC::DataCopyParams join;
        AscendC::DataCopyParams store;
        load.blockCount = 1;
        load.blockLen = 0;
        load.srcStride = 0;
        load.dstStride = 0;
        join.blockLen = static_cast<uint16_t>(kBlk32);
        join.srcStride = 0;
        join.dstStride = static_cast<uint16_t>(kBlk32);
        store.blockLen = static_cast<uint16_t>(kOutRowBlocks);
        store.srcStride = 0;
        store.dstStride = static_cast<uint16_t>(kOutRowBlocks);

        BTS_CORE_PAIR(core, cores);
        for (uint32_t job = core; job < kDstN * kBlock * kTileCount; job += cores) {
            uint32_t tile = job % kTileCount;
            uint32_t lane = job / kTileCount;
            uint32_t phase = lane & 1U;
            uint32_t n = lane >> 1U;
            uint32_t row0 = tile * kRows;
            uint32_t rows = kSrcH - row0;
            if (rows > kRows) rows = kRows;
            uint32_t n0 = phase * kBlock * kDstN + n;
            uint32_t n1 = n0 + kDstN;
            load.blockLen = static_cast<uint16_t>(rows * kInRowBlocks);
            AscendC::DataCopy(left, src[((n0 * kSrcH + row0) * kSrcW) * kChan], load);
            AscendC::DataCopy(right, src[((n1 * kSrcH + row0) * kSrcW) * kChan], load);
            AscendC::PipeBarrier<PIPE_ALL>();
            Weave(left, right, woven, rows, join);
            store.blockCount = static_cast<uint16_t>(rows);
            uint32_t dstRow = n * kDstH + phase + row0 * kBlock;
            AscendC::DataCopy(dst[dstRow * kDstW * kChan], woven, store);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }

private:
    __aicore__ static inline void Weave(AscendC::LocalTensor<T> a,
                                        AscendC::LocalTensor<T> b,
                                        AscendC::LocalTensor<T> out,
                                        uint32_t rows,
                                        AscendC::DataCopyParams &cp) {
        cp.blockCount = static_cast<uint16_t>(rows * kSrcW);
        AscendC::DataCopy(out, a, cp);
        AscendC::DataCopy(out[kChan], b, cp);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
};

template <class T>
class Strip32PhaseEmitter {
    static constexpr uint32_t kSrcH = 1024U;
    static constexpr uint32_t kSrcW = 6U;
    static constexpr uint32_t kChan = 32U;
    static constexpr uint32_t kDstN = 4U;
    static constexpr uint32_t kDstH = 2048U;
    static constexpr uint32_t kDstW = 12U;
    static constexpr uint32_t kOutRows = 64U;
    static constexpr uint32_t kTiles = (kDstH + kOutRows - 1U) / kOutRows;
    static constexpr uint32_t kInputRows = (kOutRows + 1U) / 2U;
    static constexpr uint32_t kChanBlocks = (kChan * sizeof(T)) / 32U;
    static constexpr uint32_t kSrcRowBlocks = (kSrcW * kChan * sizeof(T)) / 32U;
    static constexpr uint32_t kDstRowBlocks = (kDstW * kChan * sizeof(T)) / 32U;
    static constexpr uint32_t kInputElems = kInputRows * kSrcW * kChan;
    static constexpr uint32_t kOutputElems = kOutRows * kDstW * kChan;
    static constexpr uint32_t kInputBytes = ALIGN_UP(kInputElems * sizeof(T), 512U);
public:
    __aicore__ inline void Launch(GM_ADDR inp, GM_ADDR out) {
        BTS_GM_PAIR(T, src, dst, inp, out);
        BTS_LOCAL(T, p00, 0, kInputElems);
        BTS_LOCAL(T, p01, kInputBytes, kInputElems);
        BTS_LOCAL(T, p10, 2U * kInputBytes, kInputElems);
        BTS_LOCAL(T, p11, 3U * kInputBytes, kInputElems);
        BTS_LOCAL(T, slab, 4U * kInputBytes, kOutputElems);

        AscendC::DataCopyParams load;
        load.blockCount = 1;
        load.blockLen = 0;
        load.srcStride = 0;
        load.dstStride = 0;

        AscendC::DataCopyParams put;
        put.blockLen = static_cast<uint16_t>(kChanBlocks);
        put.srcStride = static_cast<uint16_t>((kSrcW - 1U) * kChanBlocks);
        put.dstStride = static_cast<uint16_t>(2U * kDstRowBlocks - kChanBlocks);

        AscendC::DataCopyParams store;
        store.blockCount = 1;
        store.blockLen = 0;
        store.srcStride = 0;
        store.dstStride = 0;

        BTS_CORE_PAIR(core, cores);
        for (uint32_t task = core; task < kDstN * kTiles; task += cores) {
            uint32_t tile = task % kTiles;
            uint32_t n = task / kTiles;
            uint32_t oh0 = tile * kOutRows;
            uint32_t rows = kDstH - oh0;
            if (rows > kOutRows) rows = kOutRows;
            uint32_t ih0 = oh0 >> 1U;
            uint32_t evenRows = (rows + 1U) >> 1U;
            uint32_t oddRows = rows >> 1U;

            uint32_t n00 = n;
            uint32_t n01 = n00 + kDstN;
            uint32_t n10 = n00 + 2U * kDstN;
            uint32_t n11 = n10 + kDstN;
            uint32_t baseRow = ih0 * kSrcW * kChan;

            load.blockLen = static_cast<uint16_t>(evenRows * kSrcRowBlocks);
            AscendC::DataCopy(p00, src[n00 * kSrcH * kSrcW * kChan + baseRow], load);
            AscendC::DataCopy(p01, src[n01 * kSrcH * kSrcW * kChan + baseRow], load);
            if (oddRows > 0U) {
                load.blockLen = static_cast<uint16_t>(oddRows * kSrcRowBlocks);
                AscendC::DataCopy(p10, src[n10 * kSrcH * kSrcW * kChan + baseRow], load);
                AscendC::DataCopy(p11, src[n11 * kSrcH * kSrcW * kChan + baseRow], load);
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            put.blockCount = static_cast<uint16_t>(evenRows);
            for (uint32_t iw = 0; iw < kSrcW; ++iw) {
                uint32_t inOff = iw * kChan;
                uint32_t outOff = iw * 2U * kChan;
                AscendC::DataCopy(slab[outOff], p00[inOff], put);
                AscendC::DataCopy(slab[outOff + kChan], p01[inOff], put);
            }
            if (oddRows > 0U) {
                put.blockCount = static_cast<uint16_t>(oddRows);
                uint32_t oddBase = kDstW * kChan;
                for (uint32_t iw = 0; iw < kSrcW; ++iw) {
                    uint32_t inOff = iw * kChan;
                    uint32_t outOff = oddBase + iw * 2U * kChan;
                    AscendC::DataCopy(slab[outOff], p10[inOff], put);
                    AscendC::DataCopy(slab[outOff + kChan], p11[inOff], put);
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            store.blockLen = static_cast<uint16_t>(rows * kDstRowBlocks);
            AscendC::DataCopy(dst[(n * kDstH + oh0) * kDstW * kChan], slab, store);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }
};

template <class T>
__aicore__ inline void Solve_Case9_TiledIL(GM_ADDR inp, GM_ADDR out) {
    Strip32PhaseEmitter<T> emitter;
    emitter.Launch(inp, out);
}

// ============================================================
// Compact fallback for shapes outside the tuned route table.
// Ranked cases are handled above by compile-time route keys.
// ============================================================
// Minimal fallback: ranked inputs are handled by specialized routes.
// ============================================================
template <class T>
class BtsGeneralImpl {
public:
    __aicore__ inline BtsGeneralImpl() {}
    __aicore__ inline void Setup(GM_ADDR inp, GM_ADDR out, const BtsRouteTile &tl) {
        (void)inp;
        (void)out;
        (void)tl;
    }
    __aicore__ inline void Run() {}
};
template <typename DT_X, uint64_t SCH_MODE>
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    (void)workspace;
    BTS_ROUTE_DO(ROUTE_H16_D16384, Solve_Case6_ContigCopy<DT_X>(x, y));
    BTS_ROUTE_DO(ROUTE_H16_D4096, Solve_Case5_ChSplit<DT_X>(x, y));
    BTS_ROUTE_DO(ROUTE_F32_SMALL32, Solve_Case3_InterLeave<DT_X>(x, y));
    BTS_ROUTE_DO(ROUTE_F32_MID64, Solve_Case2_RowIL<DT_X>(x, y));
    BTS_ROUTE_DO(ROUTE_F32_BIG128, Solve_Case0_RowIL<DT_X>(x, y));
    BTS_ROUTE_DO(ROUTE_F32_TINY5, Solve_Case1_CropGather<DT_X>(x, y));
    BTS_ROUTE_DO(ROUTE_H16_ODD65, Solve_Case4_HalfGather<DT_X>(x, y, tiling));
    BTS_ROUTE_DO(ROUTE_H16_WIDE256, Solve_Case7_StridedIL<DT_X>(x, y));
    BTS_ROUTE_DO(ROUTE_H16_CROP64, Solve_Case8_Blk4Crop<DT_X>(x, y));
    BTS_ROUTE_DO(ROUTE_H16_STRIP32, Solve_Case9_TiledIL<DT_X>(x, y));
    REGISTER_TILING_DEFAULT(BtsRouteTile);
    GET_TILING_DATA_WITH_STRUCT(BtsRouteTile, tl, tiling);
    BtsGeneralImpl<DT_X> fallback;
    fallback.Setup(x, y, tl);
    fallback.Run();
}
