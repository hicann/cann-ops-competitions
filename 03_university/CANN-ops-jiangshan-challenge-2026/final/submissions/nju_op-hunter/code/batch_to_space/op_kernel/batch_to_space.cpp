// BatchToSpace AI Core kernel.
#include "kernel_operator.h"

#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

using namespace AscendC;

constexpr uint32_t MODE_F32_SMALL_HW_LARGE_C_UNALIGNED = 9;
constexpr uint32_t MODE_F32_SMALL_HW_SMALL_C_UNALIGNED = 10;
constexpr uint32_t MODE_F32_LARGER_INPUT_BATCH_NOCROP = 11;
constexpr uint32_t MODE_F32_UNALIGNED_LARGE_BATCH_NOCROP = 12;
constexpr uint32_t MODE_F16_LARGE_C_UNALIGNED = 13;
constexpr uint32_t MODE_F16_LARGE_C_ALIGNED_NARROW_HW = 14;
constexpr uint32_t MODE_F16_HUGE_C_ALIGNED_NARROW_HW = 15;
constexpr uint32_t MODE_F16_SMALL_H_UNALIGNED_LARGE_W_ALIGNED = 16;
constexpr uint32_t MODE_F16_HUGE_CROP = 17;
constexpr uint32_t MODE_F16_HUGE_H_SMALL_W_UNALIGNED = 18;
constexpr uint32_t MODE_F16_LARGE_C_ALIGNED_NARROW_HW_BD8 = 19;
constexpr uint32_t MODE_F16_HUGE_C_ALIGNED_NARROW_HW_BD8 = 20;
constexpr uint32_t MODE_F16_HUGE_CROP_BD40 = 21;
constexpr uint32_t MODE_F16_SMALL_H_UNALIGNED_LARGE_W_ALIGNED_BD40 = 22;
constexpr uint32_t MODE_F32_UNALIGNED_LARGE_BATCH_NOCROP_BD8 = 23;
constexpr uint32_t MODE_F32_SMALL_HW_LARGE_C_UNALIGNED_P1 = 24;
constexpr uint32_t MODE_F32_LARGER_INPUT_BATCH_NOCROP_BD20 = 25;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t SPECIAL_BUFFER_BYTES = 72 * 1024;
// oj1 C128: input rows coalesced into one MTE2 burst per lane (de-interleaved dst, depth-2 ~172KB).
constexpr uint64_t kBs2C128RowsPerChunk = 3;
// oj3 C64: same coalesced de-interleave-on-read pattern as C128 (kK input rows -> one MTE2 burst per
// lane, depth-2 TQueBind holds 2*kK output rows for one row-strided MTE3 write).
constexpr uint64_t kBs2C64RowsPerChunk = 3;
// oj3 fixed descriptor-reduction path: 2 chunks/group instead of 5, trading 16 larger bursts for
// far fewer MTE descriptors (descriptor-bound on 910b).
constexpr uint64_t kBs2C64RowsPerChunkBd16 = 7;
// oj10 HLONG_C32: kK consecutive inH coalesced per lane; 4 lanes -> 2*kK CONTIGUOUS output rows
// (two-queue: MTE2 reads+col-deint, V interleaves parities, MTE3 one contiguous write). 2 queues
// depth-2 hold 2*kK rows each: 2*(2*kK*12*32*2B) -> 96KB at kK=16.
constexpr uint64_t kBs2HLongRowsPerChunk = 16;
// oj5 C65 crop1111: per-row Gather de-interleave tiled at an even pixel boundary so src+dst both
// double-buffer (3-stage MTE2->Gather->MTE3 pipeline). 254 wide -> tiles of 128 then 126; both start
// on an even output pixel so they share one reusable tile-local offset table. depth=65 is unaligned
// (130B/pixel), so element-wise Gather is the only valid de-interleave.
constexpr uint32_t kBs2C65TileW = 128;

template <class T>
class KernelBatchToSpace {
public:
    __aicore__ inline KernelBatchToSpace() {}

    template <uint32_t MODE_KEY>
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x), InputElements<MODE_KEY>());
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y), OutputElements<MODE_KEY>());

        if constexpr (MODE_KEY == MODE_F16_LARGE_C_UNALIGNED) {
            // Tiled 3-stage pipeline: read 2 lanes (MTE2) -> Gather de-interleave (V) -> contiguous
            // write (MTE3), with BOTH src and dst double-buffered so the vec-bound Gather fully
            // overlaps read+write. Tiling each output row at an even pixel boundary shrinks every
            // buffer ~4x (vs the whole-row single-buffer that left no UB room to pipeline), and every
            // even-width tile has the SAME relative gather structure -> one reusable offset table.
            constexpr uint32_t kTilePixelsPerLane = (kBs2C65TileW + 1) / 2;          // 64
            constexpr uint32_t kSrcBytes = 2 * kTilePixelsPerLane * 65 * sizeof(T);  // 2 lanes * 64px * depth
            constexpr uint32_t kDstBytes = kBs2C65TileW * 65 * sizeof(T);
            constexpr uint32_t kOffsetBytes = kBs2C65TileW * 65 * sizeof(uint32_t);
            pipe_.InitBuffer(bs4SrcQueue_, 2, kSrcBytes);
            pipe_.InitBuffer(bs4DstQueue_, 2, kDstBytes);
            pipe_.InitBuffer(specialOffsetBuf_, kOffsetBytes);
            return;
        }

        if constexpr (MODE_KEY == MODE_F16_HUGE_CROP || MODE_KEY == MODE_F16_HUGE_CROP_BD40) {
            // Double-buffered (depth 2) contiguous src lanes + de-interleaved output tile.
            constexpr uint32_t kSrcBytes = 4 * 65 * 64 * sizeof(T);  // 4 bw lanes * 65 cols * depth
            constexpr uint32_t kDstBytes = 256 * 64 * sizeof(T);     // tileW * depth
            pipe_.InitBuffer(bs4SrcQueue_, 2, kSrcBytes);
            pipe_.InitBuffer(bs4DstQueue_, 2, kDstBytes);
            return;
        }

        if constexpr (MODE_KEY == MODE_F32_SMALL_HW_LARGE_C_UNALIGNED) {
            // bs2 no-crop: coalesce kBs2C128RowsPerChunk consecutive input rows per lane into ONE
            // big contiguous MTE2 burst, de-interleaving on the read (dstStride folds even/odd cols),
            // then write the rows back (MTE3). The de-interleave-on-read drops the separate src
            // buffer + V pipe, so the de-interleaved output fits a depth-2 TQueBind: bursts are
            // kBs2C128RowsPerChunk x larger AND read(chunk i+1) still overlaps write(chunk i).
            constexpr uint32_t kDstBytes = kBs2C128RowsPerChunk * 56 * 128 * sizeof(T);  // rows * out row
            pipe_.InitBuffer(dataQueue_, 2, kDstBytes);
            return;
        }

        if constexpr (MODE_KEY == MODE_F32_SMALL_HW_LARGE_C_UNALIGNED_P1) {
            // P1 fixed 28-core row path: each core owns 4 full output rows. Two full-row UB buffers let
            // row i write while row i+1 is already loaded; each row is built by two lane reads with
            // dstStride placing even/odd output columns directly into final row layout.
            constexpr uint32_t kRowBytes = 56 * 128 * sizeof(T);
            pipe_.InitBuffer(rowPairBuf_, 2 * kRowBytes);
            return;
        }

        if constexpr (MODE_KEY == MODE_F32_LARGER_INPUT_BATCH_NOCROP) {
            // bs2 no-crop depth=64: coalesce kBs2C64RowsPerChunk consecutive input rows per lane into
            // ONE de-interleave-on-read MTE2 burst (dstStride folds even/odd cols), so the depth-2
            // TQueBind holds that many contiguous output rows for one row-strided MTE3 write.
            constexpr uint32_t kRowElems = 2 * 14 * 64;  // output row (28*64)
            constexpr uint32_t kBytes = kBs2C64RowsPerChunk * kRowElems * sizeof(T);
            pipe_.InitBuffer(dataQueue_, 2, kBytes);
            return;
        }

        if constexpr (MODE_KEY == MODE_F32_LARGER_INPUT_BATCH_NOCROP_BD20) {
            constexpr uint32_t kRowElems = 2 * 14 * 64;
            constexpr uint32_t kBytes = kBs2C64RowsPerChunkBd16 * kRowElems * sizeof(T);
            pipe_.InitBuffer(dataQueue_, 2, kBytes);
            return;
        }

        if constexpr (MODE_KEY == MODE_F32_UNALIGNED_LARGE_BATCH_NOCROP_BD8) {
            // Fixed 8-core case4 path: one core owns one output-H stripe across all 5 output batches.
            constexpr uint32_t kRowsPerCore = 5;
            constexpr uint32_t kRowElems = 2 * 6 * 32;
            pipe_.InitBuffer(rowPairBuf_, kRowsPerCore * kRowElems * sizeof(T));
            return;
        }

        if constexpr (MODE_KEY == MODE_F32_SMALL_HW_SMALL_C_UNALIGNED) {
            // dst row + two packed source lanes + replicated gather offset table, rounded for TBuf alignment.
            pipe_.InitBuffer(rowPairBuf_, 2048);
            return;
        }

        if constexpr (MODE_KEY == MODE_F16_HUGE_H_SMALL_W_UNALIGNED) {
            // 4 coalesced lane reads (col de-interleaved) into 2 parity regions, V-interleaved into
            // 2*kBs2HLongRowsPerChunk CONTIGUOUS output rows, one contiguous MTE3 write. Two depth-2
            // queues (case-9 pattern): each holds 2*kK output rows.
            constexpr uint32_t kQueueBytes = 2 * kBs2HLongRowsPerChunk * 12 * 32 * sizeof(T);
            pipe_.InitBuffer(bs4SrcQueue_, 2, kQueueBytes);
            pipe_.InitBuffer(bs4DstQueue_, 2, kQueueBytes);
            return;
        }

        if constexpr (MODE_KEY == MODE_F16_HUGE_C_ALIGNED_NARROW_HW ||
            MODE_KEY == MODE_F16_HUGE_C_ALIGNED_NARROW_HW_BD8) {
            // pure-copy: one contiguous burst per core via a plain TBuf + single MTE2->MTE3 sync
            // (leaner than the TQueBind's 4 EnQue/DeQue that round-trip through the V pipe).
            pipe_.InitBuffer(rowPairBuf_, 8192 * sizeof(T));
            return;
        }

        if constexpr (MODE_KEY == MODE_F16_LARGE_C_ALIGNED_NARROW_HW ||
            MODE_KEY == MODE_F16_LARGE_C_ALIGNED_NARROW_HW_BD8) {
            // one workItem = inputWidth*depth = 2*4096 elems; plain TBuf + single MTE2->MTE3 sync
            // (leaner than TQueBind: 1 read + 1 strided write per core, no V-pipe round-trip).
            pipe_.InitBuffer(rowPairBuf_, 2 * 4096 * sizeof(T));
            return;
        }

        if constexpr (MODE_KEY == MODE_F16_SMALL_H_UNALIGNED_LARGE_W_ALIGNED ||
            MODE_KEY == MODE_F16_SMALL_H_UNALIGNED_LARGE_W_ALIGNED_BD40) {
            // Single buffer at tileW=256 (one 128KB tile): per the DataCopy guide, 910b bandwidth peaks
            // with big bursts (>64KB), so a 64KB/lane read + 128KB contiguous write beats the depth-2
            // double-buffer that capped tileW at 128 (32KB reads). Double-buffering helped 910_93 but
            // did NOT translate to OJ; bigger bursts are the 910b lever here.
            constexpr uint32_t kTileBytes = 256 * 256 * sizeof(T);  // tileW * depth = 128KB
            pipe_.InitBuffer(rowPairBuf_, kTileBytes);
            return;
        }

        pipe_.InitBuffer(rowPairBuf_, SPECIAL_BUFFER_BYTES);
    }

    template <uint32_t MODE_KEY>
    __aicore__ inline void ProcessByMode()
    {
        if constexpr (MODE_KEY == MODE_F32_SMALL_HW_LARGE_C_UNALIGNED) {
            ProcessF32SmallHwLargeCUnaligned();
        } else if constexpr (MODE_KEY == MODE_F32_SMALL_HW_LARGE_C_UNALIGNED_P1) {
            ProcessF32SmallHwLargeCUnalignedP1();
        } else if constexpr (MODE_KEY == MODE_F32_SMALL_HW_SMALL_C_UNALIGNED) {
            ProcessF32SmallHwSmallCUnaligned();
        } else if constexpr (MODE_KEY == MODE_F32_LARGER_INPUT_BATCH_NOCROP) {
            ProcessF32LargerInputBatchNoCrop();
        } else if constexpr (MODE_KEY == MODE_F32_UNALIGNED_LARGE_BATCH_NOCROP) {
            ProcessF32UnalignedLargeBatchNoCrop();
        } else if constexpr (MODE_KEY == MODE_F32_UNALIGNED_LARGE_BATCH_NOCROP_BD8) {
            ProcessF32UnalignedLargeBatchNoCropBd8();
        } else if constexpr (MODE_KEY == MODE_F32_LARGER_INPUT_BATCH_NOCROP_BD20) {
            ProcessF32LargerInputBatchNoCropBd20();
        } else if constexpr (MODE_KEY == MODE_F16_LARGE_C_UNALIGNED) {
            ProcessF16LargeCUnaligned();
        } else if constexpr (MODE_KEY == MODE_F16_LARGE_C_ALIGNED_NARROW_HW) {
            ProcessF16LargeCAlignedNarrowHw();
        } else if constexpr (MODE_KEY == MODE_F16_LARGE_C_ALIGNED_NARROW_HW_BD8) {
            ProcessF16LargeCAlignedNarrowHwBd8();
        } else if constexpr (MODE_KEY == MODE_F16_HUGE_C_ALIGNED_NARROW_HW) {
            ProcessF16HugeCAlignedNarrowHw();
        } else if constexpr (MODE_KEY == MODE_F16_HUGE_C_ALIGNED_NARROW_HW_BD8) {
            ProcessF16HugeCAlignedNarrowHwBd8();
        } else if constexpr (MODE_KEY == MODE_F16_SMALL_H_UNALIGNED_LARGE_W_ALIGNED) {
            ProcessF16SmallHUnalignedLargeWAligned();
        } else if constexpr (MODE_KEY == MODE_F16_SMALL_H_UNALIGNED_LARGE_W_ALIGNED_BD40) {
            ProcessF16SmallHUnalignedLargeWAlignedBd40();
        } else if constexpr (MODE_KEY == MODE_F16_HUGE_CROP) {
            ProcessF16HugeCrop();
        } else if constexpr (MODE_KEY == MODE_F16_HUGE_CROP_BD40) {
            ProcessF16HugeCropBd40();
        } else if constexpr (MODE_KEY == MODE_F16_HUGE_H_SMALL_W_UNALIGNED) {
            ProcessF16HugeHSmallWUnaligned();
        }
    }

private:
    __aicore__ inline uint64_t MinU64(uint64_t a, uint64_t b) const
    {
        return a < b ? a : b;
    }

    __aicore__ inline uint64_t CeilDivU64(uint64_t value, uint64_t divisor) const
    {
        return value == 0 ? 0 : 1 + (value - 1) / divisor;
    }

    template <HardEvent EVENT>
    __aicore__ inline void PipeSync()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(EVENT));
        SetFlag<EVENT>(eventId);
        WaitFlag<EVENT>(eventId);
    }

    template <uint32_t MODE_KEY>
    __aicore__ inline uint64_t InputElements() const
    {
        if constexpr (MODE_KEY == MODE_F32_SMALL_HW_LARGE_C_UNALIGNED ||
            MODE_KEY == MODE_F32_SMALL_HW_LARGE_C_UNALIGNED_P1) {
            return 8 * 28 * 28 * 128;
        } else if constexpr (MODE_KEY == MODE_F32_SMALL_HW_SMALL_C_UNALIGNED) {
            return 4 * 10 * 15 * 5;
        } else if constexpr (MODE_KEY == MODE_F32_LARGER_INPUT_BATCH_NOCROP ||
            MODE_KEY == MODE_F32_LARGER_INPUT_BATCH_NOCROP_BD20) {
            return 16 * 14 * 14 * 64;
        } else if constexpr (MODE_KEY == MODE_F32_UNALIGNED_LARGE_BATCH_NOCROP ||
            MODE_KEY == MODE_F32_UNALIGNED_LARGE_BATCH_NOCROP_BD8) {
            return 20 * 4 * 6 * 32;
        } else if constexpr (MODE_KEY == MODE_F16_LARGE_C_UNALIGNED) {
            return 4 * 128 * 128 * 65;
        } else if constexpr (MODE_KEY == MODE_F16_LARGE_C_ALIGNED_NARROW_HW ||
            MODE_KEY == MODE_F16_LARGE_C_ALIGNED_NARROW_HW_BD8) {
            return 4 * 2 * 2 * 4096;
        } else if constexpr (MODE_KEY == MODE_F16_HUGE_C_ALIGNED_NARROW_HW ||
            MODE_KEY == MODE_F16_HUGE_C_ALIGNED_NARROW_HW_BD8) {
            return 4 * 1 * 1 * 16384;
        } else if constexpr (MODE_KEY == MODE_F16_SMALL_H_UNALIGNED_LARGE_W_ALIGNED ||
            MODE_KEY == MODE_F16_SMALL_H_UNALIGNED_LARGE_W_ALIGNED_BD40) {
            return 4 * 10 * 512 * 256;
        } else if constexpr (MODE_KEY == MODE_F16_HUGE_CROP || MODE_KEY == MODE_F16_HUGE_CROP_BD40) {
            return 16 * 10 * 512 * 64;
        } else if constexpr (MODE_KEY == MODE_F16_HUGE_H_SMALL_W_UNALIGNED) {
            return 16 * 1024 * 6 * 32;
        }
        return 0;
    }

    template <uint32_t MODE_KEY>
    __aicore__ inline uint64_t OutputElements() const
    {
        if constexpr (MODE_KEY == MODE_F32_SMALL_HW_SMALL_C_UNALIGNED) {
            return 1 * 17 * 26 * 5;
        } else if constexpr (MODE_KEY == MODE_F16_LARGE_C_UNALIGNED) {
            return 1 * 254 * 254 * 65;
        } else if constexpr (MODE_KEY == MODE_F16_HUGE_CROP || MODE_KEY == MODE_F16_HUGE_CROP_BD40) {
            return 1 * 40 * 1535 * 64;
        }
        return InputElements<MODE_KEY>();
    }

    // oj1 ([8,28,28,128] f32, bs2, no crop). For each (blockH,outB) group and input row inH, read the
    // two source lanes (blockW=0 -> even out-cols, blockW=1 -> odd out-cols) coalescing
    // kBs2C128RowsPerChunk consecutive input rows per lane into ONE contiguous MTE2 burst, folding the
    // even/odd-col de-interleave onto the read (dstStride), then write the rows back contiguously
    // (MTE3) through a depth-2 TQueBind so read(chunk i+1) overlaps write(chunk i). De-interleave-on-
    // read keeps BOTH HBM sides contiguous (scatter only in UB, on the MTE2 engine); a separate
    // contiguous-read + intra-UB de-interleave pass was tested and was WORSE (the extra UB pass costs
    // more than the scatter it removes from MTE2).
    __aicore__ inline void ProcessF32SmallHwLargeCUnaligned()
    {
        constexpr uint64_t kIW = 28;
        constexpr uint64_t kIH = 28;
        constexpr uint64_t kC = 128;
        constexpr uint64_t kOH = 56;
        constexpr uint64_t kLaneElems = kIW * kC;        // one input row per lane (3584)
        constexpr uint64_t kInBatch = kIH * kLaneElems;  // input batch stride (100352)
        constexpr uint64_t kRowElems = 2 * kIW * kC;     // output row (56*128 = 7168)
        constexpr uint64_t kLaneGap = 2 * kInBatch;      // GM gap between the two source lanes
        constexpr uint64_t kK = kBs2C128RowsPerChunk;    // input rows coalesced per MTE2 burst
        constexpr uint64_t kChunksPerGroup = (kIH + kK - 1) / kK;  // ceil(28/3) = 10
        constexpr uint64_t kTotalChunks = 4 * kChunksPerGroup;     // 4 (blockH,outB) groups
        constexpr uint16_t kDepthBlocks = static_cast<uint16_t>(kC / (BLOCK_BYTES / sizeof(T)));  // 16

        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockDim = GetBlockNum();
        const uint32_t rowBytes = static_cast<uint32_t>(kRowElems * sizeof(T));
        const uint32_t laneBytes = static_cast<uint32_t>(kC * sizeof(T));  // one pixel's depth (512B)
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        for (uint64_t chunk = blockIdx; chunk < kTotalChunks; chunk += blockDim) {
            const uint64_t g = chunk / kChunksPerGroup;        // (blockH,outB) group
            const uint64_t inHBase = (chunk - g * kChunksPerGroup) * kK;
            const uint64_t rows = MinU64(kK, kIH - inHBase);   // kK, or the tail remainder
            const uint64_t blockH = g & 1;
            const uint64_t outB = g >> 1;
            const uint64_t srcOff0 = ((blockH << 2) + outB) * kInBatch + inHBase * kLaneElems;
            const uint64_t dstRow0 = outB * kOH + (inHBase << 1) + blockH;  // consecutive inH -> stride-2 rows

            // Stage 1 (MTE2): one big contiguous burst per lane spanning `rows` input rows; dstStride
            // de-interleaves each pixel into its even-/odd-col slot across the `rows` output rows.
            LocalTensor<T> local = dataQueue_.AllocTensor<T>();
            DataCopyExtParams inParams{static_cast<uint16_t>(rows * kIW), laneBytes, 0, kDepthBlocks, 0};
            DataCopyPad(local, xGm_[srcOff0], inParams, padParams);                       // lane0 -> even cols
            DataCopyPad(local[kC], xGm_[srcOff0 + kLaneGap], inParams, padParams);        // lane1 -> odd cols
            dataQueue_.template EnQue<QuePosition::GM, QuePosition::VECIN, T>(local);

            // Stage 3 (MTE3): write the `rows` de-interleaved output rows (stride-2 apart in GM).
            local = dataQueue_.template DeQue<QuePosition::GM, QuePosition::VECIN, T>();
            dataQueue_.template EnQue<QuePosition::VECOUT, QuePosition::GM, T>(local);
            local = dataQueue_.template DeQue<QuePosition::VECOUT, QuePosition::GM, T>();
            DataCopyExtParams outParams{static_cast<uint16_t>(rows), rowBytes, 0, rowBytes, 0};
            DataCopyPad(yGm_[dstRow0 * kRowElems], local, outParams);
            dataQueue_.FreeTensor(local);
        }
    }

    template <uint32_t ROW_DELTA>
    __aicore__ inline void LoadF32SmallHwLargeCUnalignedP1Row(LocalTensor<T> local, uint32_t outB,
        uint32_t inHBase, DataCopyExtParams inParams, DataCopyPadExtParams<T> padParams)
    {
        constexpr uint32_t kIW = 28;
        constexpr uint32_t kIH = 28;
        constexpr uint32_t kC = 128;
        constexpr uint32_t kLaneElems = kIW * kC;        // 3584
        constexpr uint32_t kInBatch = kIH * kLaneElems;  // 100352
        constexpr uint32_t blockH = ROW_DELTA & 1;
        const uint32_t inH = inHBase + (ROW_DELTA >> 1);
        const uint32_t srcBatch0 = (blockH << 2) + outB;  // (blockH, blockW=0, outB)
        const uint32_t srcOff0 = srcBatch0 * kInBatch + inH * kLaneElems;
        constexpr uint32_t kLaneGap = 2 * kInBatch;
        DataCopyPad(local, xGm_[srcOff0], inParams, padParams);
        DataCopyPad(local[kC], xGm_[srcOff0 + kLaneGap], inParams, padParams);
    }

    __aicore__ inline void ProcessF32SmallHwLargeCUnalignedP1()
    {
        constexpr uint32_t kRowsPerCore = 4;
        constexpr uint32_t kRowElems = 56 * 128;
        constexpr uint32_t kC = 128;
        constexpr uint16_t kDepthBlocks = static_cast<uint16_t>(kC / (BLOCK_BYTES / sizeof(T)));
        constexpr uint32_t laneBytes = static_cast<uint32_t>(kC * sizeof(T));
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t row0 = blockIdx << 2;
        const uint32_t outB = blockIdx > 13;
        const uint32_t inHBase = (blockIdx - outB * 14) << 1;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams inParams{28, laneBytes, 0, kDepthBlocks, 0};

        LocalTensor<T> base = rowPairBuf_.Get<T>();
        LocalTensor<T> buf0 = base;
        LocalTensor<T> buf1 = base[kRowElems];

        // Prime row0, then keep MTE2(next row) overlapped with MTE3(current row).
        LoadF32SmallHwLargeCUnalignedP1Row<0>(buf0, outB, inHBase, inParams, padParams);
        PipeSync<HardEvent::MTE2_MTE3>();

        LoadF32SmallHwLargeCUnalignedP1Row<1>(buf1, outB, inHBase, inParams, padParams);
        DataCopy(yGm_[row0 * kRowElems], buf0, kRowElems);
        PipeSync<HardEvent::MTE2_MTE3>();
        PipeSync<HardEvent::MTE3_MTE2>();

        LoadF32SmallHwLargeCUnalignedP1Row<2>(buf0, outB, inHBase, inParams, padParams);
        DataCopy(yGm_[(row0 + 1) * kRowElems], buf1, kRowElems);
        PipeSync<HardEvent::MTE2_MTE3>();
        PipeSync<HardEvent::MTE3_MTE2>();

        LoadF32SmallHwLargeCUnalignedP1Row<3>(buf1, outB, inHBase, inParams, padParams);
        DataCopy(yGm_[(row0 + 2) * kRowElems], buf0, kRowElems);
        PipeSync<HardEvent::MTE2_MTE3>();

        DataCopy(yGm_[(row0 + 3) * kRowElems], buf1, kRowElems);
        PipeSync<HardEvent::MTE3_MTE2>();
    }

    // oj3 ([16,14,14,64] f32, bs2, no crop) -> out [4,28,28,64]. depth=64 = 256B/pixel (32B-aligned).
    // The old buffered path issued one read+write descriptor PER output row (224 read calls) and was
    // descriptor-count-bound on MTE2 (~5us). This is the C128 coalesced pattern at C64 dims: for each
    // (blockH,outB) group, coalesce kBs2C64RowsPerChunk consecutive input rows per lane into ONE
    // contiguous MTE2 burst, de-interleaving even/odd cols on the read (dstStride), so the depth-2
    // TQueBind holds 2*kK contiguous output rows written in ONE row-strided MTE3 burst. Far fewer,
    // larger descriptors; read(chunk i+1) overlaps write(chunk i).
    __aicore__ inline void ProcessF32LargerInputBatchNoCrop()
    {
        constexpr uint64_t kIW = 14;
        constexpr uint64_t kIH = 14;
        constexpr uint64_t kC = 64;
        constexpr uint64_t kOH = 28;
        constexpr uint64_t kOB = 4;
        constexpr uint64_t kLaneElems = kIW * kC;        // one input row per lane (896)
        constexpr uint64_t kInBatch = kIH * kLaneElems;  // input batch stride (12544)
        constexpr uint64_t kRowElems = 2 * kIW * kC;     // output row (28*64 = 1792)
        constexpr uint64_t kLaneGap = kOB * kInBatch;    // GM gap between the two source lanes (blockW)
        constexpr uint64_t kK = kBs2C64RowsPerChunk;     // input rows coalesced per MTE2 burst
        constexpr uint64_t kChunksPerGroup = (kIH + kK - 1) / kK;  // ceil(14/3) = 5
        constexpr uint64_t kTotalChunks = 2 * kOB * kChunksPerGroup;  // 2 (blockH) * kOB groups
        constexpr uint16_t kDepthBlocks = static_cast<uint16_t>(kC / (BLOCK_BYTES / sizeof(T)));  // 8

        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockDim = GetBlockNum();
        const uint32_t rowBytes = static_cast<uint32_t>(kRowElems * sizeof(T));
        const uint32_t laneBytes = static_cast<uint32_t>(kC * sizeof(T));  // one pixel's depth (256B)
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        for (uint64_t chunk = blockIdx; chunk < kTotalChunks; chunk += blockDim) {
            const uint64_t g = chunk / kChunksPerGroup;        // (blockH,outB) group
            const uint64_t inHBase = (chunk - g * kChunksPerGroup) * kK;
            const uint64_t rows = MinU64(kK, kIH - inHBase);   // kK, or the tail remainder
            const uint64_t blockH = g & 1;
            const uint64_t outB = g >> 1;
            const uint64_t srcOff0 = (((blockH << 1) * kOB) + outB) * kInBatch + inHBase * kLaneElems;
            const uint64_t dstRow0 = outB * kOH + (inHBase << 1) + blockH;  // consecutive inH -> stride-2

            // Stage 1 (MTE2): one big contiguous burst per lane spanning `rows` input rows; dstStride
            // de-interleaves each pixel into its even-/odd-col slot across the `rows` output rows.
            LocalTensor<T> local = dataQueue_.AllocTensor<T>();
            DataCopyExtParams inParams{static_cast<uint16_t>(rows * kIW), laneBytes, 0, kDepthBlocks, 0};
            DataCopyPad(local, xGm_[srcOff0], inParams, padParams);                  // lane0 -> even cols
            DataCopyPad(local[kC], xGm_[srcOff0 + kLaneGap], inParams, padParams);   // lane1 -> odd cols
            dataQueue_.template EnQue<QuePosition::GM, QuePosition::VECIN, T>(local);

            // Stage 3 (MTE3): write the `rows` de-interleaved output rows (stride-2 apart in GM).
            local = dataQueue_.template DeQue<QuePosition::GM, QuePosition::VECIN, T>();
            dataQueue_.template EnQue<QuePosition::VECOUT, QuePosition::GM, T>(local);
            local = dataQueue_.template DeQue<QuePosition::VECOUT, QuePosition::GM, T>();
            DataCopyExtParams outParams{static_cast<uint16_t>(rows), rowBytes, 0, rowBytes, 0};
            DataCopyPad(yGm_[dstRow0 * kRowElems], local, outParams);
            dataQueue_.FreeTensor(local);
        }
    }

    __aicore__ inline void ProcessF32LargerInputBatchNoCropBd20()
    {
        constexpr uint32_t kIW = 14;
        constexpr uint32_t kIH = 14;
        constexpr uint32_t kC = 64;
        constexpr uint32_t kOH = 28;
        constexpr uint32_t kOB = 4;
        constexpr uint32_t kLaneElems = kIW * kC;
        constexpr uint32_t kInBatch = kIH * kLaneElems;
        constexpr uint32_t kRowElems = 2 * kIW * kC;
        constexpr uint32_t kLaneGap = kOB * kInBatch;
        constexpr uint32_t kK = static_cast<uint32_t>(kBs2C64RowsPerChunkBd16);
        constexpr uint32_t kChunksPerGroup = (kIH + kK - 1) / kK;  // 2
        constexpr uint16_t kDepthBlocks = static_cast<uint16_t>(kC / (BLOCK_BYTES / sizeof(T)));
        constexpr uint32_t rowBytes = static_cast<uint32_t>(kRowElems * sizeof(T));
        constexpr uint32_t laneBytes = static_cast<uint32_t>(kC * sizeof(T));
        const uint32_t chunk = GetBlockIdx();
        const uint32_t g = chunk / kChunksPerGroup;
        const uint32_t inHBase = (chunk - g * kChunksPerGroup) * kK;
        constexpr uint32_t rows = kK;
        const uint32_t blockH = g & 1;
        const uint32_t outB = g >> 1;
        const uint32_t srcOff0 = (((blockH << 1) * kOB) + outB) * kInBatch + inHBase * kLaneElems;
        const uint32_t dstRow0 = outB * kOH + (inHBase << 1) + blockH;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        LocalTensor<T> local = dataQueue_.AllocTensor<T>();
        DataCopyExtParams inParams{static_cast<uint16_t>(rows * kIW), laneBytes, 0, kDepthBlocks, 0};
        DataCopyPad(local, xGm_[srcOff0], inParams, padParams);
        DataCopyPad(local[kC], xGm_[srcOff0 + kLaneGap], inParams, padParams);
        dataQueue_.template EnQue<QuePosition::GM, QuePosition::VECIN, T>(local);

        local = dataQueue_.template DeQue<QuePosition::GM, QuePosition::VECIN, T>();
        dataQueue_.template EnQue<QuePosition::VECOUT, QuePosition::GM, T>(local);
        local = dataQueue_.template DeQue<QuePosition::VECOUT, QuePosition::GM, T>();
        DataCopyExtParams outParams{static_cast<uint16_t>(rows), rowBytes, 0, rowBytes, 0};
        DataCopyPad(yGm_[dstRow0 * kRowElems], local, outParams);
        dataQueue_.FreeTensor(local);
    }

    __aicore__ inline void ProcessF32UnalignedLargeBatchNoCrop()
    {
        // C32 ([20,4,6,32]) is the same f32 bs2 no-crop family as oj3/C64. The old per-pixel strided
        // row-tile path was scalar-issue-bound (scal ~= mte2). Reuse oj3's tested
        // coalesced full-row buffered path: each lane is read as ONE contiguous burst de-interleaved
        // by dst stride, double-buffered. depth=32 is 32B-aligned -> ALIGNED_OUTPUT_COPY (plain
        // DataCopy). 40 output rows; at blockDim<=8 each core buffers <=10 rows per MTE2 burst.
        constexpr uint64_t kIH = 4;
        constexpr uint64_t kIW = 6;
        constexpr uint64_t kC = 32;
        constexpr uint64_t kOB = 5;
        constexpr uint64_t kOH = 8;
        constexpr uint64_t kRows = kOB * kOH;                       // 40
        constexpr uint64_t kInputRowElems = kIW * kC;               // 192
        constexpr uint64_t kInputBatchElems = kIH * kInputRowElems; // 768
        constexpr uint64_t kSrcLaneGap = kOB * kInputBatchElems;    // 3840
        constexpr uint64_t kRowElems = 2 * kIW * kC;                // 384
        constexpr uint32_t kBlockLenBytes = static_cast<uint32_t>(kC * sizeof(T));
        constexpr uint32_t kDstStrideBlocks = static_cast<uint32_t>(kC / (BLOCK_BYTES / sizeof(T)));

        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockDim = GetBlockNum();
        const uint64_t maxRowsPerChunk = blockDim <= 8 ? 10 : 7;
        LocalTensor<T> local = rowPairBuf_.Get<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams inParams{static_cast<uint16_t>(kIW), kBlockLenBytes, 0, kDstStrideBlocks, 0};
        bool copiedPrevChunk = false;

        for (uint64_t rowStart = blockIdx; rowStart < kRows; rowStart += maxRowsPerChunk * blockDim) {
            if (copiedPrevChunk) {
                PipeSync<HardEvent::MTE3_MTE2>();
            }

            uint64_t bufferedRows = 0;
            for (uint64_t row = rowStart; row < kRows && bufferedRows < maxRowsPerChunk;
                 row += blockDim, ++bufferedRows) {
                const uint64_t outH = row % kOH;
                const uint64_t outB = row / kOH;
                const uint64_t inH = outH >> 1;
                const uint64_t blockH = outH & 1;
                const uint64_t srcBatchBase = (blockH << 1) * kOB + outB;
                const uint64_t srcOffset = srcBatchBase * kInputBatchElems + inH * kInputRowElems;
                const uint32_t rowLocalBase = static_cast<uint32_t>(bufferedRows * kRowElems);
                DataCopyPad(local[rowLocalBase], xGm_[srcOffset], inParams, padParams);
                DataCopyPad(local[rowLocalBase + static_cast<uint32_t>(kC)], xGm_[srcOffset + kSrcLaneGap],
                    inParams, padParams);
            }

            PipeSync<HardEvent::MTE2_MTE3>();
            uint64_t localRow = 0;
            for (uint64_t row = rowStart; row < kRows && localRow < bufferedRows; row += blockDim, ++localRow) {
                const uint32_t localOffset = static_cast<uint32_t>(localRow * kRowElems);
                DataCopy(yGm_[row * kRowElems], local[localOffset], static_cast<uint32_t>(kRowElems));
            }
            copiedPrevChunk = bufferedRows != 0;
        }

        if (copiedPrevChunk) {
            PipeSync<HardEvent::MTE3_MTE2>();
        }
    }

    template <uint32_t OUT_B>
    __aicore__ inline void ProcessF32UnalignedLargeBatchNoCropBd8Row(LocalTensor<T> local, uint32_t srcGroupBase,
        uint32_t srcRowOffset, DataCopyExtParams inParams, DataCopyPadExtParams<T> padParams)
    {
        constexpr uint32_t kIH = 4;
        constexpr uint32_t kIW = 6;
        constexpr uint32_t kC = 32;
        constexpr uint32_t kOB = 5;
        constexpr uint32_t kOH = 8;
        constexpr uint32_t kInputRowElems = kIW * kC;
        constexpr uint32_t kInputBatchElems = kIH * kInputRowElems;
        constexpr uint32_t kSrcLaneGap = kOB * kInputBatchElems;
        constexpr uint32_t kRowElems = 2 * kIW * kC;
        constexpr uint32_t localOffset = OUT_B * kRowElems;
        const uint32_t srcOffset = (srcGroupBase + OUT_B) * kInputBatchElems + srcRowOffset;
        DataCopyPad(local[localOffset], xGm_[srcOffset], inParams, padParams);
        DataCopyPad(local[localOffset + kC], xGm_[srcOffset + kSrcLaneGap], inParams, padParams);
    }

    __aicore__ inline void ProcessF32UnalignedLargeBatchNoCropBd8()
    {
        constexpr uint32_t kIW = 6;
        constexpr uint32_t kC = 32;
        constexpr uint32_t kOH = 8;
        constexpr uint32_t kOB = 5;
        constexpr uint32_t kRowElems = 2 * kIW * kC;
        constexpr uint32_t kInputRowElems = kIW * kC;
        constexpr uint32_t kBlockLenBytes = static_cast<uint32_t>(kC * sizeof(T));
        constexpr uint32_t kDstStrideBlocks = kC / (BLOCK_BYTES / sizeof(T));
        constexpr uint32_t kRowBytes = static_cast<uint32_t>(kRowElems * sizeof(T));
        constexpr uint32_t kOutDstStrideBytes = (kOH - 1) * kRowBytes;
        const uint32_t outH = GetBlockIdx();
        const uint32_t srcGroupBase = ((outH & 1) << 1) * 5;
        const uint32_t srcRowOffset = (outH >> 1) * kInputRowElems;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams inParams{static_cast<uint16_t>(kIW), kBlockLenBytes, 0, kDstStrideBlocks, 0};
        LocalTensor<T> local = rowPairBuf_.Get<T>();
        ProcessF32UnalignedLargeBatchNoCropBd8Row<0>(local, srcGroupBase, srcRowOffset, inParams, padParams);
        ProcessF32UnalignedLargeBatchNoCropBd8Row<1>(local, srcGroupBase, srcRowOffset, inParams, padParams);
        ProcessF32UnalignedLargeBatchNoCropBd8Row<2>(local, srcGroupBase, srcRowOffset, inParams, padParams);
        ProcessF32UnalignedLargeBatchNoCropBd8Row<3>(local, srcGroupBase, srcRowOffset, inParams, padParams);
        ProcessF32UnalignedLargeBatchNoCropBd8Row<4>(local, srcGroupBase, srcRowOffset, inParams, padParams);
        PipeSync<HardEvent::MTE2_MTE3>();
        DataCopyExtParams outParams{kOB, kRowBytes, 0, kOutDstStrideBytes, 0};
        DataCopyPad(yGm_[outH * kRowElems], local, outParams);
    }

    __aicore__ inline void ProcessF32SmallHwSmallCUnaligned()
    {
        constexpr uint64_t kIH = 10;
        constexpr uint64_t kIW = 15;
        constexpr uint64_t kC = 5;
        constexpr uint64_t kOH = 17;
        constexpr uint64_t kOW = 26;
        constexpr uint64_t kRowElems = kOW * kC;                    // 130
        constexpr uint64_t kRowAlignedElems = 136;                  // ceil(130 / 8) * 8
        constexpr uint64_t kLanePixels = (kOW + 1) >> 1;            // 13
        constexpr uint64_t kLaneElems = kLanePixels * kC;           // 65
        constexpr uint64_t kLaneAlignedElems = 72;                  // ceil(65 / 8) * 8
        constexpr uint32_t kLaneBytes = static_cast<uint32_t>(kLaneElems * sizeof(T));
        constexpr uint32_t kRowBytes = static_cast<uint32_t>(kRowElems * sizeof(T));
        LocalTensor<T> localBase = rowPairBuf_.Get<T>();
        LocalTensor<T> dstLocal = localBase;
        LocalTensor<T> srcLocal = localBase[static_cast<uint32_t>(kRowAlignedElems)];
        constexpr uint32_t offsetTableBase = static_cast<uint32_t>(kRowAlignedElems + kLaneAlignedElems * 2);
        LocalTensor<uint32_t> offsetLocal = rowPairBuf_.Get<uint32_t>()[offsetTableBase];
        LocalTensor<int32_t> offsetI32 = rowPairBuf_.Get<int32_t>()[offsetTableBase];
        // Build the Gather byte-offset table cheaply: lay out a base block of basePairs pixel-pairs
        // (even+odd) with scalar SetValue, then replicate it across the remaining lanes with a single
        // vector Adds per group (offset advances by basePairs*depth elems). basePairs=4 keeps each
        // group a multiple of 32B (4*2*depth*4B = 160B), which the vector Adds destination requires.
        // Cuts the per-core table build from lanePixels*5*2 SetValue stores to basePairs*5*2 + a few Adds.
        constexpr uint64_t basePairs = 4;
        constexpr uint32_t groupOffsetCount = static_cast<uint32_t>(basePairs * (kC << 1));
        constexpr uint64_t offsetGroups = (kLanePixels + basePairs - 1) / basePairs;
        for (uint64_t k = 0; k < basePairs; ++k) {
            const uint64_t evenBase = (k << 1) * kC;
            const uint64_t oddBase = ((k << 1) + 1) * kC;
            for (uint64_t c = 0; c < 5; ++c) {
                offsetI32.SetValue(static_cast<uint32_t>(evenBase + c),
                    static_cast<int32_t>((k * kC + c) * sizeof(T)));
                offsetI32.SetValue(static_cast<uint32_t>(oddBase + c),
                    static_cast<int32_t>((kLaneAlignedElems + k * kC + c) * sizeof(T)));
            }
        }
        for (uint64_t group = 1; group < offsetGroups; ++group) {
            Adds(offsetI32[static_cast<uint32_t>(group * groupOffsetCount)], offsetI32,
                static_cast<int32_t>(group * basePairs * kC * sizeof(T)), groupOffsetCount);
        }

        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockDim = GetBlockNum();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        bool copiedPrevRow = false;
        for (uint64_t outH = blockIdx; outH < kOH; outH += blockDim) {
            if (copiedPrevRow) {
                PipeSync<HardEvent::MTE3_MTE2>();
            }

            const uint64_t expandedH = outH + 2;  // cropTop
            const uint64_t inH = expandedH >> 1;
            const uint64_t blockH = expandedH & 1;
            const uint64_t srcBatchBase = blockH << 1;
            const uint64_t srcBaseEvenOut = (((srcBatchBase + 1) * kIH + inH) * kIW + 1) * kC;
            const uint64_t srcBaseOddOut = ((srcBatchBase * kIH + inH) * kIW + 2) * kC;
            DataCopyExtParams laneParams{1, kLaneBytes, 0, 0, 0};
            DataCopyPad(srcLocal, xGm_[srcBaseEvenOut], laneParams, padParams);
            DataCopyPad(srcLocal[static_cast<uint32_t>(kLaneAlignedElems)], xGm_[srcBaseOddOut], laneParams,
                padParams);

            PipeSync<HardEvent::MTE2_V>();
            Gather(dstLocal, srcLocal, offsetLocal, 0, static_cast<uint32_t>(kRowElems));
            PipeSync<HardEvent::V_MTE3>();
            DataCopyExtParams outParams{1, kRowBytes, 0, 0, 0};
            DataCopyPad(yGm_[outH * kRowElems], dstLocal, outParams);
            copiedPrevRow = true;
        }

        if (copiedPrevRow) {
            PipeSync<HardEvent::MTE3_V>();
        }
    }

    // oj5 ([4,128,128,65] f16, bs2, crop 1/1/1/1 -> out [1,254,254,65]). depth=65 -> 130B/pixel, NOT
    // 32B-aligned, so the per-pixel stride-2 de-interleave can't use a block-strided DataCopy (nor a
    // strided GM write: DataCopyPad needs 32B-aligned blockLen). It must use element-wise Gather
    // (vec-bound, 13.1us). The original was fully single-buffered (PipeSync barriers, zero overlap) ->
    // 28.5us = sum of all pipes. Double-buffer the 2-lane reads (bs4SrcQueue_, depth-2) so MTE2 hides
    // under the Gather; the Gather output stays single (rowPairBuf_) so Gather and write serialize, but
    // the duration floor drops to ~Gather+write instead of read+Gather+write.
    // oj5 ([4,128,128,65] f16, bs2, crop 1111) -> out [1,254,254,65]. depth=65 -> 130B/pixel is NOT
    // 32B-aligned, so the cheap stride-based de-interleaves (intra-UB block-copy, strided GM write)
    // all fail and element-wise Gather is forced (the vec-bound bottleneck). Each output row is the
    // even/odd-column interleave of two source lanes (blockW=1 -> even out-cols, blockW=0 -> odd).
    // Tiling the row at an even pixel boundary lets src AND dst double-buffer, so the 3 pipes
    // (MTE2 read | V Gather | MTE3 write) overlap across tiles instead of running serially per row.
    __aicore__ inline void ProcessF16LargeCUnaligned()
    {
        constexpr uint64_t kIH = 128;
        constexpr uint64_t kIW = 128;
        constexpr uint64_t kC = 65;
        constexpr uint64_t kOH = 254;
        constexpr uint64_t kOW = 254;
        constexpr uint64_t kLaneElems = kIW * kC;                 // 8320
        constexpr uint64_t kRowElems = kOW * kC;                  // 16510
        constexpr uint64_t kTileW = kBs2C65TileW;
        constexpr uint64_t kTilePixelsPerLane = (kTileW + 1) >> 1;   // 64
        constexpr uint64_t kLaneTileAligned = kTilePixelsPerLane * kC;  // 4160 (mult of 16)
        constexpr uint64_t kTilesPerRow = (kOW + kTileW - 1) / kTileW;  // 2
        constexpr uint64_t kTotalTiles = kOH * kTilesPerRow;            // 508

        LocalTensor<uint32_t> offsetLocal = specialOffsetBuf_.Get<uint32_t>();
        LocalTensor<int32_t> offsetI32 = specialOffsetBuf_.Get<int32_t>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        // Build the tile-local Gather byte-offset table once (identical for every tile): even local
        // positions index lane(blockW=1) packed at src[0..]; odd index lane(blockW=0) at
        // src[laneTileAligned..]. base block of basePairs pixel-pairs, then Adds-replicate per group
        // (+basePairs*depth elems). basePairs=4 keeps the Adds dst 32B-aligned (4*2*65 elems mult of 8).
        constexpr uint64_t basePairs = 4;
        constexpr uint32_t groupOffsetCount = static_cast<uint32_t>(basePairs * (kC << 1));  // 520
        constexpr uint64_t offsetGroups = (kTilePixelsPerLane + basePairs - 1) / basePairs;  // 16
        for (uint64_t k = 0; k < basePairs; ++k) {
            const uint64_t evenBase = (k << 1) * kC;
            const uint64_t oddBase = ((k << 1) + 1) * kC;
            for (uint64_t c = 0; c < 65; ++c) {
                offsetI32.SetValue(static_cast<uint32_t>(evenBase + c),
                    static_cast<int32_t>((k * kC + c) * sizeof(T)));
                offsetI32.SetValue(static_cast<uint32_t>(oddBase + c),
                    static_cast<int32_t>((kLaneTileAligned + k * kC + c) * sizeof(T)));
            }
        }
        for (uint64_t group = 1; group < offsetGroups; ++group) {
            Adds(offsetI32[static_cast<uint32_t>(group * groupOffsetCount)], offsetI32,
                static_cast<int32_t>(group * basePairs * kC * sizeof(T)), groupOffsetCount);
        }

        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockDim = GetBlockNum();
        for (uint64_t t = blockIdx; t < kTotalTiles; t += blockDim) {
            const uint64_t outH = t >> 1;
            const uint64_t tileIdx = t & 1;
            const uint64_t startP = tileIdx * kTileW;            // even (kTileW even)
            const uint64_t tw = tileIdx == 0 ? kTileW : kOW - kTileW;
            const uint64_t evenPixels = (tw + 1) >> 1;
            const uint64_t oddPixels = tw >> 1;

            const uint64_t expandedH = outH + 1;                 // cropTop = 1
            const uint64_t inH = expandedH >> 1;
            const uint64_t blockH = expandedH & 1;
            const uint64_t rowOffset = inH * kLaneElems;
            const uint64_t srcBatchBase = blockH << 1;
            const uint64_t inWBase = startP >> 1;
            const uint64_t evenSrcGm = (srcBatchBase + 1) * kIH * kLaneElems + rowOffset + inWBase * kC;
            const uint64_t oddSrcGm = srcBatchBase * kIH * kLaneElems + rowOffset + (inWBase + 1) * kC;

            // Stage 1 (MTE2, double-buffered): two contiguous lane reads into the fixed tile layout.
            LocalTensor<T> src = bs4SrcQueue_.AllocTensor<T>();
            DataCopyExtParams evenParams{1, static_cast<uint32_t>(evenPixels * kC * sizeof(T)), 0, 0, 0};
            DataCopyExtParams oddParams{1, static_cast<uint32_t>(oddPixels * kC * sizeof(T)), 0, 0, 0};
            DataCopyPad(src, xGm_[evenSrcGm], evenParams, padParams);
            DataCopyPad(src[static_cast<uint32_t>(kLaneTileAligned)], xGm_[oddSrcGm], oddParams, padParams);
            bs4SrcQueue_.EnQue(src);

            // Stage 2 (V): Gather de-interleave into the double-buffered dst tile.
            src = bs4SrcQueue_.DeQue<T>();
            LocalTensor<T> dst = bs4DstQueue_.AllocTensor<T>();
            Gather(dst, src, offsetLocal, 0, static_cast<uint32_t>(tw * kC));
            bs4SrcQueue_.FreeTensor(src);
            bs4DstQueue_.EnQue(dst);

            // Stage 3 (MTE3, double-buffered): contiguous write of the tile's row segment.
            dst = bs4DstQueue_.DeQue<T>();
            DataCopyExtParams outParams{1, static_cast<uint32_t>(tw * kC * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[outH * kRowElems + startP * kC], dst, outParams);
            bs4DstQueue_.FreeTensor(dst);
        }
    }

    // oj6 ([4,2,2,4096] f16, bs2, no crop). Launch/scalar-bound (128KB, scalar > mte2), one workItem
    // per core at blockDim=8. Each core does one contiguous lane read + one strided write through a
    // plain TBuf with a single MTE2->MTE3 PipeSync, which is leaner than the TQueBind's four
    // EnQue/DeQue (those route the relabel through the idle V pipe, adding two events of scalar setup).
    // Shape is fixed for this exact mode, so all dims are constexpr (ScalarBound P8/P4): reading them
    // from `this->` members is a runtime Load that can't fold; constexpr folds the index math and
    // DataCopy params. Byte-identical on the 910_93 dev-env (Init/Process are all inlined so the
    // compiler already register-keeps the tiling values), but kept as official-guidance insurance in
    // case OJ's 910b toolchain spills/reloads the members.
    __aicore__ inline void ProcessF16LargeCAlignedNarrowHw()
    {
        constexpr uint64_t kIH = 2;
        constexpr uint64_t kIW = 2;
        constexpr uint64_t kC = 4096;
        constexpr uint64_t kOW = 4;
        constexpr uint64_t kWorkItems = 4 * kIH;                         // inputBatch*inputHeight = 8
        constexpr uint32_t kRowCopyElems = static_cast<uint32_t>(kIW * kC);
        constexpr uint32_t kBlockLenBytes = static_cast<uint32_t>(kC * sizeof(T));
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockDim = GetBlockNum();
        LocalTensor<T> local = rowPairBuf_.Get<T>();
        if (blockDim == 8) {
            const uint64_t srcBatch = blockIdx >> 1;
            const uint64_t inH = blockIdx & 1;
            const uint64_t blockH = srcBatch >> 1;
            const uint64_t blockW = srcBatch & 1;
            const uint64_t outH = (inH << 1) + blockH;
            const uint64_t srcOffset = static_cast<uint64_t>(blockIdx) * kRowCopyElems;
            const uint64_t dstOffset = (outH * kOW + blockW) * kC;
            DataCopy(local, xGm_[srcOffset], kRowCopyElems);
            PipeSync<HardEvent::MTE2_MTE3>();
            DataCopyExtParams outParams{static_cast<uint16_t>(kIW), kBlockLenBytes, 0, kBlockLenBytes, 0};
            DataCopyPad(yGm_[dstOffset], local, outParams);
            return;
        }

        bool wrotePrev = false;
        for (uint64_t workItem = blockIdx; workItem < kWorkItems; workItem += blockDim) {
            const uint64_t srcBatch = workItem >> 1;                     // /kIH (kIH=2)
            const uint64_t inH = workItem & 1;                           // %kIH
            const uint64_t blockH = srcBatch >> 1;
            const uint64_t blockW = srcBatch & 1;
            const uint64_t outH = (inH << 1) + blockH;
            const uint64_t srcOffset = workItem * kRowCopyElems;
            const uint64_t dstOffset = (outH * kOW + blockW) * kC;

            if (wrotePrev) {
                PipeSync<HardEvent::MTE3_MTE2>();
            }
            DataCopy(local, xGm_[srcOffset], kRowCopyElems);
            PipeSync<HardEvent::MTE2_MTE3>();
            DataCopyExtParams outParams{static_cast<uint16_t>(kIW), kBlockLenBytes, 0, kBlockLenBytes, 0};
            DataCopyPad(yGm_[dstOffset], local, outParams);
            wrotePrev = true;
        }
    }

    __aicore__ inline void ProcessF16LargeCAlignedNarrowHwBd8()
    {
        constexpr uint32_t kIW = 2;
        constexpr uint32_t kC = 4096;
        constexpr uint32_t kRowCopyElems = static_cast<uint32_t>(kIW * kC);
        constexpr uint32_t kBlockLenBytes = static_cast<uint32_t>(kC * sizeof(T));
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t srcOffset = blockIdx << 13;  // blockIdx * (2*4096)
        const uint32_t dstOffset = (((blockIdx & 1) << 3) + (blockIdx & 4) + ((blockIdx >> 1) & 1)) << 12;
        LocalTensor<T> local = rowPairBuf_.Get<T>();
        DataCopy(local, xGm_[srcOffset], kRowCopyElems);
        PipeSync<HardEvent::MTE2_MTE3>();
        DataCopyExtParams outParams{static_cast<uint16_t>(kIW), kBlockLenBytes, 0, kBlockLenBytes, 0};
        DataCopyPad(yGm_[dstOffset], local, outParams);
    }

    // oj7 ([4,1,1,16384] f16, bs2, no crop). This is a PURE contiguous identity copy (out element i
    // == in element i for all 65536), so it's entirely launch/dispatch + UB-roundtrip bound. Each
    // core copies ONE contiguous block-aligned slice (total/blockDim) in a single MTE2+MTE3 burst
    // through a plain TBuf with one MTE2->MTE3 PipeSync. blockDim=8 is the measured sweet spot (4 and
    // 16 both lose). One burst beats chunking here: the per-chunk sync setup outweighs the overlap a
    // second chunk would buy, and the leaner TBuf avoids the TQueBind V-pipe relabel events.
    __aicore__ inline void ProcessF16HugeCAlignedNarrowHw()
    {
        // Fixed shape -> constexpr total (ScalarBound P8/P4: avoids the `this->totalElements_/blockDim_`
        // member Loads); blockDim from the GetBlockNum() register read keeps it robust. Neutral on the
        // 910_93 dev-env (already inlined/folded), kept as official-guidance insurance for OJ's 910b.
        constexpr uint64_t kTotal = 65536;    // 4*1*1*16384
        constexpr uint32_t kMaxBurst = 8192;  // matches the depth-2 UB buffer (16KB/lane)
        constexpr uint32_t blockElems = BLOCK_BYTES / sizeof(T);
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockDim = GetBlockNum();
        if (blockDim == 8) {
            constexpr uint32_t kElemsPerCore = static_cast<uint32_t>(kTotal / 8);
            const uint64_t offset = static_cast<uint64_t>(blockIdx) * kElemsPerCore;
            LocalTensor<T> local = rowPairBuf_.Get<T>();
            DataCopy(local, xGm_[offset], kElemsPerCore);
            PipeSync<HardEvent::MTE2_MTE3>();
            DataCopy(yGm_[offset], local, kElemsPerCore);
            return;
        }

        // Each core owns one contiguous block-aligned slice; one MTE2+MTE3 burst (no inner loop in the
        // common case) so the per-core sync overhead is paid once instead of once per 4096-chunk.
        const uint64_t perCore = CeilDivU64(CeilDivU64(kTotal, blockDim), blockElems) * blockElems;
        const uint64_t start = static_cast<uint64_t>(blockIdx) * perCore;
        const uint64_t end = MinU64(start + perCore, kTotal);
        LocalTensor<T> local = rowPairBuf_.Get<T>();
        bool wrotePrev = false;
        for (uint64_t offset = start; offset < end; offset += kMaxBurst) {
            if (wrotePrev) {
                PipeSync<HardEvent::MTE3_MTE2>();
            }
            const uint32_t copyElems = static_cast<uint32_t>(MinU64(kMaxBurst, end - offset));
            DataCopy(local, xGm_[offset], copyElems);
            PipeSync<HardEvent::MTE2_MTE3>();
            DataCopy(yGm_[offset], local, copyElems);
            wrotePrev = true;
        }
    }

    __aicore__ inline void ProcessF16HugeCAlignedNarrowHwBd8()
    {
        constexpr uint32_t kElemsPerCore = 8192;
        const uint32_t offset = GetBlockIdx() << 13;
        LocalTensor<T> local = rowPairBuf_.Get<T>();
        DataCopy(local, xGm_[offset], kElemsPerCore);
        PipeSync<HardEvent::MTE2_MTE3>();
        DataCopy(yGm_[offset], local, kElemsPerCore);
    }

    // oj8 ([4,10,512,256] f16, bs2, no crop) -> out [1,20,1024,256]. Per (outH, wTile) the two source
    // lanes are read with the even/odd-col de-interleave folded onto the read (one big contiguous GM
    // burst each) into a single 256-col tile, then written back as ONE contiguous burst. tileW=256
    // (one 128KB tile, single-buffered + PipeSync): the DataCopy guide's 910b lever is big bursts
    // (64KB/lane read, 128KB write near peak), which beats the depth-2 double-buffer (capped tileW=128
    // = 32KB reads) -- that double-buffer helped 910_93 but regressed OJ.
    __aicore__ inline void ProcessF16SmallHUnalignedLargeWAligned()
    {
        constexpr uint64_t kIH = 10;
        constexpr uint64_t kIW = 512;
        constexpr uint64_t kC = 256;
        constexpr uint64_t kOH = 20;
        constexpr uint64_t kOW = 1024;
        constexpr uint64_t tileW = 256;
        constexpr uint64_t kWtiles = kOW / tileW;                  // 4
        constexpr uint64_t workItems = kOH * kWtiles;              // 80
        constexpr uint64_t kInputRowElems = kIW * kC;              // 131072
        constexpr uint64_t kInputBatchElems = kIH * kInputRowElems;
        constexpr uint64_t kRowElems = kOW * kC;
        constexpr uint64_t kTileElems = tileW * kC;
        constexpr uint64_t kLaneCols = tileW >> 1;                 // 128
        constexpr uint32_t kBlockLenBytes = static_cast<uint32_t>(kC * sizeof(T));
        constexpr uint32_t kDstStrideBlocks = static_cast<uint32_t>(kC / (BLOCK_BYTES / sizeof(T)));
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockDim = GetBlockNum();
        LocalTensor<T> local = rowPairBuf_.Get<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams inParams{static_cast<uint16_t>(kLaneCols), kBlockLenBytes, 0, kDstStrideBlocks, 0};
        bool wrotePrev = false;

        for (uint64_t workItem = blockIdx; workItem < workItems; workItem += blockDim) {
            const uint64_t outH = workItem >> 2;
            const uint64_t wTile = workItem & 3;
            const uint64_t startOutW = wTile * tileW;

            if (wrotePrev) {
                PipeSync<HardEvent::MTE3_MTE2>();
            }
            const uint64_t inH = outH >> 1;
            const uint64_t blockH = outH & 1;
            const uint64_t copyStartW = startOutW >> 1;
            const uint64_t srcBase = (blockH << 1) * kInputBatchElems + inH * kInputRowElems + copyStartW * kC;
            DataCopyPad(local, xGm_[srcBase], inParams, padParams);
            DataCopyPad(local[kC], xGm_[srcBase + kInputBatchElems], inParams, padParams);
            PipeSync<HardEvent::MTE2_MTE3>();
            DataCopy(yGm_[outH * kRowElems + startOutW * kC], local, static_cast<uint32_t>(kTileElems));
            wrotePrev = true;
        }
    }

    template <uint32_t WORK_DELTA>
    __aicore__ inline void ProcessF16SmallHUnalignedLargeWAlignedBd40Tile(LocalTensor<T> local)
    {
        constexpr uint32_t kIH = 10;
        constexpr uint32_t kIW = 512;
        constexpr uint32_t kC = 256;
        constexpr uint32_t kOW = 1024;
        constexpr uint32_t tileW = 256;
        constexpr uint32_t kLaneCols = tileW >> 1;
        constexpr uint32_t kInputRowElems = kIW * kC;
        constexpr uint32_t kInputBatchElems = kIH * kInputRowElems;
        constexpr uint32_t kRowElems = kOW * kC;
        constexpr uint32_t kTileElems = tileW * kC;
        constexpr uint32_t kBlockLenBytes = static_cast<uint32_t>(kC * sizeof(T));
        constexpr uint32_t kDstStrideBlocks = kC / (BLOCK_BYTES / sizeof(T));
        const uint32_t workItem = GetBlockIdx() + WORK_DELTA;
        const uint32_t outH = workItem >> 2;
        const uint32_t wTile = workItem & 3;
        const uint32_t startOutW = wTile << 8;       // wTile * 256
        const uint32_t copyStartW = startOutW >> 1;  // lane width starts at outW/2
        const uint32_t inH = outH >> 1;
        const uint32_t blockH = outH & 1;
        const uint32_t srcBase = (blockH << 1) * kInputBatchElems + inH * kInputRowElems + copyStartW * kC;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams inParams{static_cast<uint16_t>(kLaneCols), kBlockLenBytes, 0, kDstStrideBlocks, 0};
        DataCopyPad(local, xGm_[srcBase], inParams, padParams);
        DataCopyPad(local[kC], xGm_[srcBase + kInputBatchElems], inParams, padParams);
        PipeSync<HardEvent::MTE2_MTE3>();
        DataCopy(yGm_[outH * kRowElems + startOutW * kC], local, kTileElems);
    }

    __aicore__ inline void ProcessF16SmallHUnalignedLargeWAlignedBd40()
    {
        LocalTensor<T> local = rowPairBuf_.Get<T>();

        ProcessF16SmallHUnalignedLargeWAlignedBd40Tile<0>(local);
        PipeSync<HardEvent::MTE3_MTE2>();
        ProcessF16SmallHUnalignedLargeWAlignedBd40Tile<40>(local);
    }

    template <bool IS_LAST_TILE>
    __aicore__ inline void ProcessF16HugeCropTile(uint64_t outH, uint64_t tile)
    {
        constexpr uint64_t kDepth = 64;
        constexpr uint64_t kInputWidth = 512;
        constexpr uint64_t kInputHeight = 10;
        constexpr uint64_t kBatchStride = kInputHeight * kInputWidth * kDepth;
        constexpr uint64_t kRowStride = kInputWidth * kDepth;
        constexpr uint64_t kOutputWidth = 1535;
        constexpr uint64_t kRowElems = kOutputWidth * kDepth;
        constexpr uint64_t tileW = 256;
        constexpr uint64_t kReadColsFull = tileW / 4 + 1;
        constexpr uint64_t kBwStride = kReadColsFull * kDepth;
        constexpr uint64_t kTiles = (kOutputWidth + tileW - 1) / tileW;
        constexpr uint64_t kLastTile = kTiles - 1;
        constexpr uint64_t kActualTileW = IS_LAST_TILE ? (kOutputWidth - kLastTile * tileW) : tileW;
        constexpr uint64_t kReadCols = IS_LAST_TILE ? (kReadColsFull - 1) : kReadColsFull;
        constexpr uint32_t readBytes = static_cast<uint32_t>(kReadCols * kDepth * sizeof(T));
        constexpr uint16_t kBlockLen = static_cast<uint16_t>(kDepth * sizeof(T) / BLOCK_BYTES);
        constexpr uint16_t kDstStride = static_cast<uint16_t>(3 * kDepth * sizeof(T) / BLOCK_BYTES);
        constexpr uint16_t n0 = static_cast<uint16_t>((kActualTileW + 3) >> 2);
        constexpr uint16_t n1 = static_cast<uint16_t>((kActualTileW + 2) >> 2);
        constexpr uint16_t n2 = static_cast<uint16_t>((kActualTileW + 1) >> 2);
        constexpr uint16_t n3 = static_cast<uint16_t>(kActualTileW >> 2);

        const uint64_t startOutW = tile * tileW;
        const uint64_t inWBase = (startOutW + 513) >> 2;
        const uint64_t inH = outH >> 2;
        const uint64_t blockH = outH & 3;
        const uint64_t srcBase = (blockH << 2) * kBatchStride + inH * kRowStride + inWBase * kDepth;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        LocalTensor<T> srcLocal = bs4SrcQueue_.AllocTensor<T>();
        constexpr uint32_t srcStrideBytes = static_cast<uint32_t>(kBatchStride * sizeof(T)) - readBytes;
        constexpr uint32_t dstStrideBlocks = (static_cast<uint32_t>(kBwStride * sizeof(T)) - readBytes) /
            BLOCK_BYTES;
        DataCopyExtParams readParams{4, readBytes, srcStrideBytes, dstStrideBlocks, 0};
        DataCopyPad(srcLocal, xGm_[srcBase], readParams, padParams);
        bs4SrcQueue_.EnQue(srcLocal);

        srcLocal = bs4SrcQueue_.DeQue<T>();
        LocalTensor<T> dstLocal = bs4DstQueue_.AllocTensor<T>();
        DataCopy(dstLocal, srcLocal[static_cast<uint32_t>(kBwStride)],
            DataCopyParams(n0, kBlockLen, 0, kDstStride));
        DataCopy(dstLocal[kDepth], srcLocal[static_cast<uint32_t>(2 * kBwStride)],
            DataCopyParams(n1, kBlockLen, 0, kDstStride));
        DataCopy(dstLocal[2 * kDepth], srcLocal[static_cast<uint32_t>(3 * kBwStride)],
            DataCopyParams(n2, kBlockLen, 0, kDstStride));
        DataCopy(dstLocal[3 * kDepth], srcLocal[static_cast<uint32_t>(kDepth)],
            DataCopyParams(n3, kBlockLen, 0, kDstStride));
        bs4SrcQueue_.FreeTensor(srcLocal);
        bs4DstQueue_.EnQue(dstLocal);

        dstLocal = bs4DstQueue_.DeQue<T>();
        constexpr uint32_t writeBytes = static_cast<uint32_t>(kActualTileW * kDepth * sizeof(T));
        DataCopyExtParams writeParams{1, writeBytes, 0, 0, 0};
        DataCopyPad(yGm_[outH * kRowElems + startOutW * kDepth], dstLocal, writeParams);
        bs4DstQueue_.FreeTensor(dstLocal);
    }

    template <bool IS_LAST_TILE, uint32_t TILE>
    __aicore__ inline void ProcessF16HugeCropTileFixed(uint32_t outH)
    {
        constexpr uint32_t kDepth = 64;
        constexpr uint32_t kInputWidth = 512;
        constexpr uint32_t kInputHeight = 10;
        constexpr uint32_t kBatchStride = kInputHeight * kInputWidth * kDepth;
        constexpr uint32_t kRowStride = kInputWidth * kDepth;
        constexpr uint32_t kOutputWidth = 1535;
        constexpr uint32_t kRowElems = kOutputWidth * kDepth;
        constexpr uint32_t tileW = 256;
        constexpr uint32_t kReadColsFull = tileW / 4 + 1;
        constexpr uint32_t kBwStride = kReadColsFull * kDepth;
        constexpr uint32_t kTiles = (kOutputWidth + tileW - 1) / tileW;
        constexpr uint32_t kLastTile = kTiles - 1;
        constexpr uint32_t kActualTileW = IS_LAST_TILE ? (kOutputWidth - kLastTile * tileW) : tileW;
        constexpr uint32_t kReadCols = IS_LAST_TILE ? (kReadColsFull - 1) : kReadColsFull;
        constexpr uint32_t readBytes = static_cast<uint32_t>(kReadCols * kDepth * sizeof(T));
        constexpr uint16_t kBlockLen = static_cast<uint16_t>(kDepth * sizeof(T) / BLOCK_BYTES);
        constexpr uint16_t kDstStride = static_cast<uint16_t>(3 * kDepth * sizeof(T) / BLOCK_BYTES);
        constexpr uint16_t n0 = static_cast<uint16_t>((kActualTileW + 3) >> 2);
        constexpr uint16_t n1 = static_cast<uint16_t>((kActualTileW + 2) >> 2);
        constexpr uint16_t n2 = static_cast<uint16_t>((kActualTileW + 1) >> 2);
        constexpr uint16_t n3 = static_cast<uint16_t>(kActualTileW >> 2);
        constexpr uint32_t startOutW = TILE * tileW;
        constexpr uint32_t inWBase = (startOutW + 513) >> 2;

        const uint32_t inH = outH >> 2;
        const uint32_t blockH = outH & 3;
        const uint32_t srcBase = (blockH << 2) * kBatchStride + inH * kRowStride + inWBase * kDepth;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        LocalTensor<T> srcLocal = bs4SrcQueue_.AllocTensor<T>();
        constexpr uint32_t srcStrideBytes = static_cast<uint32_t>(kBatchStride * sizeof(T)) - readBytes;
        constexpr uint32_t dstStrideBlocks = (static_cast<uint32_t>(kBwStride * sizeof(T)) - readBytes) /
            BLOCK_BYTES;
        DataCopyExtParams readParams{4, readBytes, srcStrideBytes, dstStrideBlocks, 0};
        DataCopyPad(srcLocal, xGm_[srcBase], readParams, padParams);
        bs4SrcQueue_.EnQue(srcLocal);

        srcLocal = bs4SrcQueue_.DeQue<T>();
        LocalTensor<T> dstLocal = bs4DstQueue_.AllocTensor<T>();
        DataCopy(dstLocal, srcLocal[kBwStride], DataCopyParams(n0, kBlockLen, 0, kDstStride));
        DataCopy(dstLocal[kDepth], srcLocal[2 * kBwStride], DataCopyParams(n1, kBlockLen, 0, kDstStride));
        DataCopy(dstLocal[2 * kDepth], srcLocal[3 * kBwStride], DataCopyParams(n2, kBlockLen, 0, kDstStride));
        DataCopy(dstLocal[3 * kDepth], srcLocal[kDepth], DataCopyParams(n3, kBlockLen, 0, kDstStride));
        bs4SrcQueue_.FreeTensor(srcLocal);
        bs4DstQueue_.EnQue(dstLocal);

        dstLocal = bs4DstQueue_.DeQue<T>();
        constexpr uint32_t writeBytes = static_cast<uint32_t>(kActualTileW * kDepth * sizeof(T));
        DataCopyExtParams writeParams{1, writeBytes, 0, 0, 0};
        DataCopyPad(yGm_[outH * kRowElems + startOutW * kDepth], dstLocal, writeParams);
        bs4DstQueue_.FreeTensor(dstLocal);
    }

    __aicore__ inline void ProcessF16HugeCrop()
    {
        constexpr uint64_t kOutputHeight = 40;
        constexpr uint64_t kOutputWidth = 1535;
        constexpr uint64_t tileW = 256;
        constexpr uint64_t kTiles = (kOutputWidth + tileW - 1) / tileW;
        constexpr uint64_t kLastTile = kTiles - 1;
        constexpr uint64_t kFullWorkItems = kOutputHeight * kLastTile;

        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockDim = GetBlockNum();
        for (uint64_t workItem = blockIdx; workItem < kFullWorkItems; workItem += blockDim) {
            const uint64_t outH = workItem / kLastTile;
            const uint64_t tile = workItem - outH * kLastTile;
            ProcessF16HugeCropTile<false>(outH, tile);
        }
        for (uint64_t outH = blockIdx; outH < kOutputHeight; outH += blockDim) {
            ProcessF16HugeCropTile<true>(outH, kLastTile);
        }
    }

    __aicore__ inline void ProcessF16HugeCropBd40()
    {
        const uint32_t outH = GetBlockIdx();
        ProcessF16HugeCropTileFixed<false, 0>(outH);
        ProcessF16HugeCropTileFixed<false, 1>(outH);
        ProcessF16HugeCropTileFixed<false, 2>(outH);
        ProcessF16HugeCropTileFixed<false, 3>(outH);
        ProcessF16HugeCropTileFixed<false, 4>(outH);
        ProcessF16HugeCropTileFixed<true, 5>(outH);
    }


    // oj10 ([16,1024,6,32] f16, bs2, no crop). 8192 tiny 384-elem output rows. The per-row path made
    // ~16K scattered reads (scalar 7.9us); the read-coalesced de-interleave-on-read variant fixed that
    // but wrote stride-2 (64 tiny 768B writes/chunk) -> 910b-hostile. This keeps coalesced reads AND
    // contiguous writes (case-9 4-lane pattern): for one (outB) the 4 (blockH,blockW) source lanes
    // each supply kK GM-contiguous inH rows. Read each lane in ONE coalesced burst, col de-interleaved
    // (dstStride) into a parity region packed by blockH (E=even out rows, O=odd). V then interleaves
    // E,O into 2*kK CONSECUTIVE output rows, and one contiguous MTE3 burst writes them.
    __aicore__ inline void ProcessF16HugeHSmallWUnaligned()
    {
        constexpr uint64_t kIW = 6;
        constexpr uint64_t kIH = 1024;
        constexpr uint64_t kC = 32;
        constexpr uint64_t kOH = 2048;
        constexpr uint64_t kOB = 4;
        constexpr uint64_t kLaneElems = kIW * kC;        // one input row per lane (192)
        constexpr uint64_t kInBatch = kIH * kLaneElems;  // input batch stride (196608)
        constexpr uint64_t kRowElems = 2 * kIW * kC;     // output row (12*32 = 384)
        constexpr uint64_t kBatchUnit = kOB * kInBatch;  // GM step between consecutive (blockH,blockW) lanes
        constexpr uint64_t kK = kBs2HLongRowsPerChunk;   // inH rows coalesced per lane burst
        constexpr uint64_t kChunksPerGroup = kIH / kK;             // 1024/16 = 64 (exact)
        constexpr uint64_t kTotalChunks = kOB * kChunksPerGroup;   // one chunk = 2*kK output rows of one outB
        constexpr uint16_t kDepthBlocks = static_cast<uint16_t>(kC / (BLOCK_BYTES / sizeof(T)));      // 2
        constexpr uint16_t kRowBlocks = static_cast<uint16_t>(kRowElems / (BLOCK_BYTES / sizeof(T))); // 24
        constexpr uint32_t kParitySpan = static_cast<uint32_t>(kK * kRowElems);  // packed rows of one parity

        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockDim = GetBlockNum();
        const uint32_t laneBytes = static_cast<uint32_t>(kC * sizeof(T));  // one pixel's depth (64B)
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams inParams{static_cast<uint16_t>(kK * kIW), laneBytes, 0, kDepthBlocks, 0};

        for (uint64_t chunk = blockIdx; chunk < kTotalChunks; chunk += blockDim) {
            const uint64_t outB = chunk / kChunksPerGroup;
            const uint64_t inHBase = (chunk - outB * kChunksPerGroup) * kK;
            const uint64_t srcB0 = outB * kInBatch + inHBase * kLaneElems;  // lane (blockH=0,blockW=0)

            // Stage 1 (MTE2): 4 coalesced lane reads -> parity region E (even out rows) then O (odd),
            // each col de-interleaved (blockW=0 even cols, blockW=1 odd cols) via dstStride.
            LocalTensor<T> src = bs4SrcQueue_.AllocTensor<T>();
            DataCopyPad(src, xGm_[srcB0], inParams, padParams);                                   // E even cols
            DataCopyPad(src[kC], xGm_[srcB0 + kBatchUnit], inParams, padParams);                  // E odd cols
            DataCopyPad(src[kParitySpan], xGm_[srcB0 + 2 * kBatchUnit], inParams, padParams);     // O even cols
            DataCopyPad(src[kParitySpan + static_cast<uint32_t>(kC)], xGm_[srcB0 + 3 * kBatchUnit],
                inParams, padParams);                                                            // O odd cols
            bs4SrcQueue_.EnQue(src);

            // Stage 2 (V): interleave the two parities into 2*kK consecutive output rows.
            src = bs4SrcQueue_.DeQue<T>();
            LocalTensor<T> dst = bs4DstQueue_.AllocTensor<T>();
            DataCopy(dst, src, DataCopyParams(static_cast<uint16_t>(kK), kRowBlocks, 0, kRowBlocks));
            DataCopy(dst[static_cast<uint32_t>(kRowElems)], src[kParitySpan],
                DataCopyParams(static_cast<uint16_t>(kK), kRowBlocks, 0, kRowBlocks));
            bs4SrcQueue_.FreeTensor(src);
            bs4DstQueue_.EnQue(dst);

            // Stage 3 (MTE3): one contiguous write of the 2*kK output rows.
            dst = bs4DstQueue_.DeQue<T>();
            const uint64_t outRow0 = outB * kOH + (inHBase << 1);
            DataCopy(yGm_[outRow0 * kRowElems], dst, static_cast<uint32_t>(2 * kK * kRowElems));
            bs4DstQueue_.FreeTensor(dst);
        }
    }

private:
    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> dataQueue_;
    TQue<QuePosition::VECIN, 2> bs4SrcQueue_;
    TQue<QuePosition::VECOUT, 2> bs4DstQueue_;
    TBuf<TPosition::VECCALC> rowPairBuf_;
    TBuf<TPosition::VECCALC> specialOffsetBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

};

template <typename DT_X, uint32_t MODE_KEY>
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    // Pin the kernel as AIV-only (no AIC cohort at launch). The compiler already auto-detects this
    // (the 910_93 binary is coreType=VectorCore / MAGIC_ELF_AIVEC since no mode uses cube/Matmul), so
    // this is defensive insurance that OJ's 910b toolchain also launches vector-only, not MIX.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    KernelBatchToSpace<DT_X> op;
    op.template Init<MODE_KEY>(x, y);
    op.template ProcessByMode<MODE_KEY>();
}
