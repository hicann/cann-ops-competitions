// Kernel侧核函数实现
#include "kernel_operator.h"

#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

#define B2S_ENABLE_ROW_SLAB_POLICY 1

static constexpr uint32_t kB2SRuntimeChannelTaskElements = 2048U;

__aicore__ inline uint32_t B2SCoreIndex() {
    return static_cast<uint32_t>(AscendC::GetBlockIdx());
}

__aicore__ inline uint32_t B2SCoreCount() {
    return static_cast<uint32_t>(AscendC::GetBlockNum());
}

__aicore__ inline void B2SBarrierAll() {
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void B2SBarrierLoad() {
    AscendC::PipeBarrier<PIPE_MTE2>();
}

__aicore__ inline void B2SBarrierStore() {
    AscendC::PipeBarrier<PIPE_MTE3>();
}

__aicore__ inline void B2SBarrierVector() {
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline event_t B2SEventByParity(uint32_t selector, event_t even_event, event_t odd_event) {
    return (selector & 1U) == 0U ? even_event : odd_event;
}

template <class DT_X>
__aicore__ inline AscendC::LocalTensor<DT_X> B2STileByParity(uint32_t selector,
                                                             AscendC::LocalTensor<DT_X> even_tile,
                                                             AscendC::LocalTensor<DT_X> odd_tile) {
    return (selector & 1U) == 0U ? even_tile : odd_tile;
}

__aicore__ inline void B2SMarkTileLoaded(event_t event_id) {
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
}

__aicore__ inline void B2SWaitTileLoaded(event_t event_id) {
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
}

__aicore__ inline void B2SMarkTileStored(event_t event_id) {
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
}

__aicore__ inline void B2SWaitTileStored(event_t event_id) {
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
}

// Shared copy helpers keep schedule bodies focused on index mapping while
// preserving the original 32-byte aligned DataCopy/DataCopyPad behavior.
template <class DT_X>
__aicore__ inline void BindIoWindow(GM_ADDR x, GM_ADDR y,
                                          AscendC::GlobalTensor<DT_X> &gmSource,
                                          AscendC::GlobalTensor<DT_X> &gmResult) {
    gmSource.SetGlobalBuffer((__gm__ DT_X*)x);
    gmResult.SetGlobalBuffer((__gm__ DT_X*)y);
}

template <class DT_X>
__aicore__ inline void DisableCopyPadding(AscendC::DataCopyPadExtParams<DT_X> &pad) {
    pad.isPad = false;
    pad.leftPadding = 0;
    pad.rightPadding = 0;
    pad.paddingValue = static_cast<DT_X>(0);
}

__aicore__ inline void ResetBlockTransfer(AscendC::DataCopyParams &params) {
    params.blockCount = 1;
    params.blockLen = 0;
    params.srcStride = 0;
    params.dstStride = 0;
}

__aicore__ inline void ResetExtTransfer(AscendC::DataCopyExtParams &params) {
    params.blockCount = 1;
    params.blockLen = 0;
    params.srcStride = 0;
    params.dstStride = 0;
    params.rsv = 0;
}

struct LinearBlockTile {
    uint32_t first_block;
    uint32_t block_count;

    __aicore__ inline bool Active() const {
        return block_count != 0U;
    }
};

struct R2RowPhaseTile {
    uint32_t height_phase;
    uint32_t output_batch;
    uint32_t input_row;
    uint32_t row_count;
};

// Split a fixed number of 32-byte blocks evenly across AI Core workers.
template <uint32_t kTotalBlocks>
__aicore__ inline LinearBlockTile ClaimBlockSpan() {
    const uint32_t workerId = B2SCoreIndex();
    const uint32_t workerCount = B2SCoreCount();
    const uint32_t quota = (kTotalBlocks + workerCount - 1U) / workerCount;
    LinearBlockTile tile = {workerId * quota, 0U};
    if (tile.first_block >= kTotalBlocks) {
        return tile;
    }
    tile.block_count = kTotalBlocks - tile.first_block;
    if (tile.block_count > quota) {
        tile.block_count = quota;
    }
    return tile;
}

// Decode a compact row-interleave job id into output batch, height phase, and
// the input-row tile handled by this worker.
template <uint32_t kSourceHSmall, uint32_t kRowsPerTile, uint32_t kTileCount>
__aicore__ inline R2RowPhaseTile DecodeRowPhaseTile(uint32_t job) {
    const uint32_t rowTile = job % kTileCount;
    const uint32_t phaseBatch = job / kTileCount;
    R2RowPhaseTile tile = {
        phaseBatch & 1U,
        phaseBatch >> 1,
        rowTile * kRowsPerTile,
        0U,
    };
    tile.row_count = kSourceHSmall - tile.input_row;
    if (tile.row_count > kRowsPerTile) {
        tile.row_count = kRowsPerTile;
    }
    return tile;
}

// Build a gather index pattern that alternates two input banks into one output row.
template <class DT_X, uint32_t kChannels, uint32_t kBankElems,
          uint32_t kPairsPerSeed, uint32_t kSeedElems,
          uint32_t kFullSeeds, uint32_t kTailElems>
__aicore__ inline void BuildTwoBankGatherIndex(AscendC::LocalTensor<int32_t> indexTensor) {
    constexpr event_t kReadyForVector = static_cast<event_t>(6);
    constexpr uint32_t kPairElems = 2U * kChannels;
    for (uint32_t pairId = 0; pairId < kPairsPerSeed; ++pairId) {
        uint32_t dstPair = pairId * kPairElems;
        uint32_t columnBytes = pairId * kChannels * sizeof(DT_X);
        for (uint32_t channel = 0; channel < kChannels; ++channel) {
            uint32_t channelByte = channel * sizeof(DT_X);
            indexTensor.SetValue(dstPair + channel,
                                 static_cast<int32_t>(columnBytes + channelByte));
            indexTensor.SetValue(dstPair + kChannels + channel,
                                 static_cast<int32_t>(kBankElems * sizeof(DT_X) +
                                                      columnBytes + channelByte));
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(kReadyForVector);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(kReadyForVector);
    for (uint32_t seed = 1U; seed <= kFullSeeds; ++seed) {
        uint32_t activeElems = seed == kFullSeeds ? kTailElems : kSeedElems;
        AscendC::Adds(indexTensor[seed * kSeedElems],
                      indexTensor[(seed - 1U) * kSeedElems],
                      static_cast<int32_t>(kPairsPerSeed * kChannels * sizeof(DT_X)),
                      activeElems);
    }
    B2SBarrierVector();
}

// Small cropped float32 case: load two source banks and gather them into the
// cropped output row in UB, avoiding scalar per-element writes.
template <class DT_X>
__aicore__ inline void RunCompactR2CropGather(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kSourceHSmall = 10U;
    constexpr uint32_t kSourceWSmall = 15U;
    constexpr uint32_t kChannels = 5U;
    constexpr uint32_t kResultHSmall = 17U;
    constexpr uint32_t kResultWSmall = 26U;
    constexpr uint32_t kBlock = 2U;
    constexpr uint32_t kTopTrim = 2U;
    constexpr uint32_t kLeftTrim = 3U;
    constexpr uint32_t kFirstCols = 13U;
    constexpr uint32_t kChannelBytes = kChannels * sizeof(DT_X);
    constexpr uint32_t kPackedBytes = ((kFirstCols * kChannelBytes + 31U) / 32U) * 32U;
    constexpr uint32_t kPackedElems = kPackedBytes / sizeof(DT_X);
    constexpr uint32_t kStageElems = kPackedElems * 2U;
    constexpr uint32_t kRowElems = kResultWSmall * kChannels;
    constexpr uint32_t kPairsPerSeed = 4U;
    constexpr uint32_t kSeedElems = kPairsPerSeed * 2U * kChannels;
    constexpr uint32_t kFullSeeds = kRowElems / kSeedElems;
    constexpr uint32_t kTailIndexElems =
        ((kRowElems - kFullSeeds * kSeedElems + 7U) / 8U) * 8U;
    constexpr uint32_t kIndexElems = kFullSeeds * kSeedElems + kTailIndexElems;

    AscendC::GlobalTensor<DT_X> gmSource;
    AscendC::GlobalTensor<DT_X> gmResult;
    BindIoWindow(x, y, gmSource, gmResult);

    AscendC::LocalTensor<DT_X> ubBankPair(AscendC::TPosition::VECCALC, 0, kStageElems);
    AscendC::LocalTensor<DT_X> ubGatheredRow(AscendC::TPosition::VECCALC, 1024, kRowElems);
    AscendC::LocalTensor<int32_t> ubGatherSeed(AscendC::TPosition::VECCALC, 2048, kIndexElems);
    AscendC::LocalTensor<uint32_t> ubGatherMap(AscendC::TPosition::VECCALC, 2048, kRowElems);

    AscendC::DataCopyPadExtParams<DT_X> noPad;
    DisableCopyPadding(noPad);

    AscendC::DataCopyExtParams firstBankLoad;
    AscendC::DataCopyExtParams secondBankLoad;
    AscendC::DataCopyExtParams rowStoreMove;
    ResetExtTransfer(firstBankLoad);
    ResetExtTransfer(secondBankLoad);
    ResetExtTransfer(rowStoreMove);
    firstBankLoad.blockLen = kFirstCols * kChannelBytes;
    secondBankLoad.blockLen = kFirstCols * kChannelBytes;
    rowStoreMove.blockLen = kResultWSmall * kChannelBytes;

    BuildTwoBankGatherIndex<DT_X, kChannels, kPackedElems, kPairsPerSeed,
                             kSeedElems, kFullSeeds, kTailIndexElems>(ubGatherSeed);

    const uint32_t worker = B2SCoreIndex();
    const uint32_t workers = B2SCoreCount();
    for (uint32_t outRow = worker; outRow < kResultHSmall; outRow += workers) {
        uint32_t paddedRow = outRow + kTopTrim;
        uint32_t sourceRow = paddedRow >> 1;
        uint32_t rowPhase = paddedRow & 1U;
        uint32_t bank0Batch = rowPhase * kBlock + (kLeftTrim & 1U);
        uint32_t bank1Batch = rowPhase * kBlock + ((kLeftTrim + 1U) & 1U);
        uint32_t bank0Column = kLeftTrim >> 1U;
        uint32_t bank1Column = (kLeftTrim + 1U) >> 1U;
        uint32_t srcBank0 = ((bank0Batch * kSourceHSmall + sourceRow) * kSourceWSmall + bank0Column) * kChannels;
        uint32_t srcBank1 = ((bank1Batch * kSourceHSmall + sourceRow) * kSourceWSmall + bank1Column) * kChannels;

        AscendC::DataCopyPad(ubBankPair, gmSource[srcBank0], firstBankLoad, noPad);
        AscendC::DataCopyPad(ubBankPair[kPackedElems], gmSource[srcBank1], secondBankLoad, noPad);
        B2SBarrierAll();

        AscendC::Gather(ubGatheredRow, ubBankPair, ubGatherMap, 0, kRowElems);
        B2SBarrierAll();

        AscendC::DataCopyPad(gmResult[outRow * kRowElems], ubGatheredRow, rowStoreMove);
        B2SBarrierStore();
    }
}

// Degenerate no-crop shape where BatchToSpace is physically a contiguous copy.
template <class DT_X>
__aicore__ inline void RunAlignedFlatCopy(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kTotalBlocks = (4U * 1U * 1U * 16384U * sizeof(DT_X)) / 32U;
    constexpr uint32_t kMaxBlocks = 2048U;
    constexpr uint32_t kElemsPerBlock = 32U / sizeof(DT_X);
    constexpr uint32_t kMaxElems = kMaxBlocks * kElemsPerBlock;

    AscendC::GlobalTensor<DT_X> gmSource;
    AscendC::GlobalTensor<DT_X> gmResult;
    BindIoWindow(x, y, gmSource, gmResult);
    AscendC::LocalTensor<DT_X> ubLinearWindow(AscendC::TPosition::VECCALC, 0, kMaxElems);

    LinearBlockTile assigned = ClaimBlockSpan<kTotalBlocks>();
    if (!assigned.Active()) {
        return;
    }

    uint32_t elemCursor = assigned.first_block * kElemsPerBlock;
    AscendC::DataCopyParams burstMove;
    ResetBlockTransfer(burstMove);

    while (assigned.block_count != 0U) {
        const uint32_t burstBlocks = assigned.block_count < kMaxBlocks ? assigned.block_count : kMaxBlocks;
        burstMove.blockLen = static_cast<uint16_t>(burstBlocks);
        AscendC::DataCopy(ubLinearWindow, gmSource[elemCursor], burstMove);
        B2SBarrierLoad();
        AscendC::DataCopy(gmResult[elemCursor], ubLinearWindow, burstMove);
        B2SBarrierStore();
        elemCursor += burstBlocks * kElemsPerBlock;
        assigned.block_count -= burstBlocks;
    }
}

// Channel-tile path for very wide C: move a channel slice across all expanded
// columns so each DataCopy burst remains aligned and UB-bounded.
template <class DT_X, uint32_t kSourceH, uint32_t kSourceW, uint32_t kChannelCount, uint32_t kOutputBatchCount, uint32_t kTileD>
__aicore__ inline void RunR2ChannelSliceNoCrop(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kResultH = kSourceH * 2U;
    constexpr uint32_t kResultW = kSourceW * 2U;
    constexpr uint32_t kBlockSize = 2U;
    constexpr uint32_t kDTiles = kChannelCount / kTileD;
    constexpr uint32_t kChunkBlocks = (kTileD * sizeof(DT_X)) / 32U;
    constexpr uint32_t kDstStrideBlocks = ((kChannelCount - kTileD) * sizeof(DT_X)) / 32U;
    constexpr uint32_t kTileElems = kResultW * kTileD;

    AscendC::GlobalTensor<DT_X> gmInput;
    AscendC::GlobalTensor<DT_X> gmOutput;
    gmInput.SetGlobalBuffer((__gm__ DT_X*)x);
    gmOutput.SetGlobalBuffer((__gm__ DT_X*)y);
    AscendC::LocalTensor<DT_X> rowLocal(AscendC::TPosition::VECCALC, 0, kTileElems);

    AscendC::DataCopyParams channelMove;
    channelMove.blockCount = 1;
    channelMove.blockLen = static_cast<uint16_t>(kChunkBlocks);
    channelMove.srcStride = 0;
    channelMove.dstStride = 0;

    AscendC::DataCopyParams storeMove;
    storeMove.blockCount = kResultW;
    storeMove.blockLen = static_cast<uint16_t>(kChunkBlocks);
    storeMove.srcStride = 0;
    storeMove.dstStride = static_cast<uint16_t>(kDstStrideBlocks);

    const uint32_t blockIdx = B2SCoreIndex();
    const uint32_t blockNum = B2SCoreCount();
    for (uint32_t taskId = blockIdx; taskId < kOutputBatchCount * kResultH * kDTiles; taskId += blockNum) {
        uint32_t channelTileId = taskId % kDTiles;
        uint32_t rowTaskId = taskId / kDTiles;
        uint32_t dstRowIdx = rowTaskId % kResultH;
        uint32_t dstBatchIdx = rowTaskId / kResultH;
        uint32_t channelOffset = channelTileId * kTileD;
        uint32_t srcRowIdx = dstRowIdx >> 1;
        uint32_t phaseH = dstRowIdx & 1U;
        uint32_t srcBatchFirst = (phaseH * kBlockSize) * kOutputBatchCount + dstBatchIdx;
        uint32_t srcBatchSecond = srcBatchFirst + kOutputBatchCount;

#pragma unroll
        for (uint32_t srcColIdx = 0; srcColIdx < kSourceW; ++srcColIdx) {
            uint32_t srcOffsetFirst =
                ((srcBatchFirst * kSourceH + srcRowIdx) * kSourceW + srcColIdx) * kChannelCount + channelOffset;
            uint32_t srcOffsetSecond =
                ((srcBatchSecond * kSourceH + srcRowIdx) * kSourceW + srcColIdx) * kChannelCount + channelOffset;
            AscendC::DataCopy(rowLocal[(2U * srcColIdx) * kTileD],
                              gmInput[srcOffsetFirst], channelMove);
            AscendC::DataCopy(rowLocal[(2U * srcColIdx + 1U) * kTileD],
                              gmInput[srcOffsetSecond], channelMove);
        }
        B2SBarrierLoad();

        uint32_t dstLinearOffset = ((dstBatchIdx * kResultH + dstRowIdx) * kResultW) * kChannelCount + channelOffset;
        AscendC::DataCopy(gmOutput[dstLinearOffset], rowLocal, storeMove);
        B2SBarrierStore();
    }
}

// Row-interleave path: copy both width phases for a set of input rows, then
// stitch them into doubled output rows using strided UB copies.
template <class DT_X, uint32_t kSourceH, uint32_t kSourceW, uint32_t kChannelCount, uint32_t kOutputBatchCount,
          uint32_t kTileRows, uint32_t kBufferAlignBytes, bool kLoadBarrierAll,
          bool kUbBarrierAll, bool kUbBarrierVector>
__aicore__ inline void RunR2RowPhaseInterleave(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kResultH = kSourceH * 2U;
    constexpr uint32_t kResultW = kSourceW * 2U;
    constexpr uint32_t kBlockSize = 2U;
    constexpr uint32_t kTilesPerBlockH = (kSourceH + kTileRows - 1U) / kTileRows;
    constexpr uint32_t kChannelBlocks = (kChannelCount * sizeof(DT_X)) / 32U;
    constexpr uint32_t kInputRowBlocks = (kSourceW * kChannelCount * sizeof(DT_X)) / 32U;
    constexpr uint32_t kResultRowBlocks = (kResultW * kChannelCount * sizeof(DT_X)) / 32U;
    constexpr uint32_t kMaxInputElems = kTileRows * kSourceW * kChannelCount;
    constexpr uint32_t kMaxOutputElems = kTileRows * kResultW * kChannelCount;
    constexpr uint32_t kSourceByteCount =
        ((kMaxInputElems * sizeof(DT_X) + kBufferAlignBytes - 1U) / kBufferAlignBytes) * kBufferAlignBytes;

    AscendC::GlobalTensor<DT_X> gmSource;
    AscendC::GlobalTensor<DT_X> gmResult;
    BindIoWindow(x, y, gmSource, gmResult);
    AscendC::LocalTensor<DT_X> ubPhaseZeroRows(AscendC::TPosition::VECCALC, 0, kMaxInputElems);
    AscendC::LocalTensor<DT_X> ubPhaseOneRows(AscendC::TPosition::VECCALC, kSourceByteCount, kMaxInputElems);
    AscendC::LocalTensor<DT_X> ubInterleavedRows(AscendC::TPosition::VECCALC, 2U * kSourceByteCount, kMaxOutputElems);

    AscendC::DataCopyParams rowLoadMove;
    AscendC::DataCopyParams phaseShuffleMove;
    AscendC::DataCopyParams rowStoreMove;
    ResetBlockTransfer(rowLoadMove);
    ResetBlockTransfer(phaseShuffleMove);
    ResetBlockTransfer(rowStoreMove);
    phaseShuffleMove.blockLen = static_cast<uint16_t>(kChannelBlocks);
    phaseShuffleMove.dstStride = static_cast<uint16_t>(kChannelBlocks);
    rowStoreMove.blockLen = static_cast<uint16_t>(kResultRowBlocks);
    rowStoreMove.dstStride = static_cast<uint16_t>(kResultRowBlocks);

    const uint32_t workerId = B2SCoreIndex();
    const uint32_t workerCount = B2SCoreCount();
    constexpr uint32_t kJobCount = kOutputBatchCount * kBlockSize * kTilesPerBlockH;
    for (uint32_t job = workerId; job < kJobCount; job += workerCount) {
        const R2RowPhaseTile tile = DecodeRowPhaseTile<kSourceH, kTileRows, kTilesPerBlockH>(job);

        const uint32_t phaseBatch0 = tile.height_phase * kBlockSize * kOutputBatchCount + tile.output_batch;
        const uint32_t phaseBatch1 = phaseBatch0 + kOutputBatchCount;
        const uint32_t srcPhase0 = ((phaseBatch0 * kSourceH + tile.input_row) * kSourceW) * kChannelCount;
        const uint32_t srcPhase1 = ((phaseBatch1 * kSourceH + tile.input_row) * kSourceW) * kChannelCount;

        rowLoadMove.blockLen = static_cast<uint16_t>(tile.row_count * kInputRowBlocks);
        AscendC::DataCopy(ubPhaseZeroRows, gmSource[srcPhase0], rowLoadMove);
        AscendC::DataCopy(ubPhaseOneRows, gmSource[srcPhase1], rowLoadMove);
        if constexpr (kLoadBarrierAll) {
            B2SBarrierAll();
        } else {
            B2SBarrierLoad();
        }

        phaseShuffleMove.blockCount = static_cast<uint16_t>(tile.row_count * kSourceW);
        AscendC::DataCopy(ubInterleavedRows, ubPhaseZeroRows, phaseShuffleMove);
        AscendC::DataCopy(ubInterleavedRows[kChannelCount], ubPhaseOneRows, phaseShuffleMove);
        if constexpr (kUbBarrierVector) {
            B2SBarrierVector();
        } else if constexpr (kUbBarrierAll) {
            B2SBarrierAll();
        } else {
            B2SBarrierLoad();
        }

        rowStoreMove.blockCount = static_cast<uint16_t>(tile.row_count);
        const uint32_t dstRowIdx = tile.height_phase + tile.input_row * kBlockSize;
        const uint32_t dstLinearOffset = (tile.output_batch * kResultH + dstRowIdx) * kResultW * kChannelCount;
        AscendC::DataCopy(gmResult[dstLinearOffset], ubInterleavedRows, rowStoreMove);
        B2SBarrierStore();
    }
}

// Row-slab path trades more UB space for fewer GM stores on tall tensors.
template <class DT_X, uint32_t kSourceH, uint32_t kSourceW, uint32_t kChannelCount, uint32_t kOutputBatchCount,
          uint32_t kOutRows, uint32_t kBufferAlignBytes>
__aicore__ inline void RunR2TallRowSlab(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kResultH = kSourceH * 2U;
    constexpr uint32_t kResultW = kSourceW * 2U;
    constexpr uint32_t kBlockSize = 2U;
    constexpr uint32_t kTileCount = (kResultH + kOutRows - 1U) / kOutRows;
    constexpr uint32_t kChannelBlocks = (kChannelCount * sizeof(DT_X)) / 32U;
    constexpr uint32_t kInputRowBlocks = (kSourceW * kChannelCount * sizeof(DT_X)) / 32U;
    constexpr uint32_t kResultRowBlocks = (kResultW * kChannelCount * sizeof(DT_X)) / 32U;
    constexpr uint32_t kInputRows = (kOutRows + 1U) / 2U;
    constexpr uint32_t kSourceElemCount = kInputRows * kSourceW * kChannelCount;
    constexpr uint32_t kResultElemCount = kOutRows * kResultW * kChannelCount;
    constexpr uint32_t kSourceByteCount =
        ((kSourceElemCount * sizeof(DT_X) + kBufferAlignBytes - 1U) / kBufferAlignBytes) * kBufferAlignBytes;

    AscendC::GlobalTensor<DT_X> gmInput;
    AscendC::GlobalTensor<DT_X> gmOutput;
    gmInput.SetGlobalBuffer((__gm__ DT_X*)x);
    gmOutput.SetGlobalBuffer((__gm__ DT_X*)y);
    AscendC::LocalTensor<DT_X> ubPhaseFirst(AscendC::TPosition::VECCALC, 0, kSourceElemCount);
    AscendC::LocalTensor<DT_X> ubPhaseSecond(AscendC::TPosition::VECCALC,
                                              kSourceByteCount, kSourceElemCount);
    AscendC::LocalTensor<DT_X> ubOutputRows(AscendC::TPosition::VECCALC,
                                              2U * kSourceByteCount, kResultElemCount);

    AscendC::DataCopyParams loadMove;
    loadMove.blockCount = 1;
    loadMove.blockLen = 0;
    loadMove.srcStride = 0;
    loadMove.dstStride = 0;

    AscendC::DataCopyParams columnParams;
    columnParams.blockCount = 0;
    columnParams.blockLen = static_cast<uint16_t>(kChannelBlocks);
    columnParams.srcStride = static_cast<uint16_t>((kSourceW - 1U) * kChannelBlocks);
    columnParams.dstStride = static_cast<uint16_t>(2U * kResultRowBlocks - kChannelBlocks);

    AscendC::DataCopyParams storeMove;
    storeMove.blockCount = 1;
    storeMove.blockLen = 0;
    storeMove.srcStride = 0;
    storeMove.dstStride = 0;

    const uint32_t blockIdx = B2SCoreIndex();
    const uint32_t blockNum = B2SCoreCount();
    for (uint32_t slabTaskId = blockIdx; slabTaskId < kOutputBatchCount * kTileCount;
         slabTaskId += blockNum) {
        uint32_t outputTileId = slabTaskId % kTileCount;
        uint32_t dstBatchIdx = slabTaskId / kTileCount;
        uint32_t outputRowStart = outputTileId * kOutRows;
        uint32_t activeOutputRows = kResultH - outputRowStart;
        if (activeOutputRows > kOutRows) {
            activeOutputRows = kOutRows;
        }

        // A slab may start on either height phase; handle both phases and stitch
        // the two width phases into their output columns.
        uint32_t firstHeightPhase = outputRowStart & 1U;
        for (uint32_t phaseH = 0U; phaseH < kBlockSize; ++phaseH) {
            uint32_t firstLocalRow = (phaseH == firstHeightPhase) ? 0U : 1U;
            if (firstLocalRow >= activeOutputRows) {
                continue;
            }
            uint32_t activeRowsForPhase =
                (activeOutputRows - firstLocalRow + 1U) >> 1U;
            uint32_t inputRowStart = (outputRowStart + firstLocalRow) >> 1U;
            uint32_t srcBatchFirst = phaseH * kBlockSize * kOutputBatchCount + dstBatchIdx;
            uint32_t srcBatchSecond = srcBatchFirst + kOutputBatchCount;
            uint32_t srcOffsetFirst =
                ((srcBatchFirst * kSourceH + inputRowStart) * kSourceW) * kChannelCount;
            uint32_t srcOffsetSecond =
                ((srcBatchSecond * kSourceH + inputRowStart) * kSourceW) * kChannelCount;

            loadMove.blockLen =
                static_cast<uint16_t>(activeRowsForPhase * kInputRowBlocks);
            AscendC::DataCopy(ubPhaseFirst, gmInput[srcOffsetFirst], loadMove);
            AscendC::DataCopy(ubPhaseSecond, gmInput[srcOffsetSecond], loadMove);
            B2SBarrierAll();

            columnParams.blockCount = static_cast<uint16_t>(activeRowsForPhase);
            uint32_t outputRowBase = firstLocalRow * kResultW * kChannelCount;
            for (uint32_t srcColIdx = 0; srcColIdx < kSourceW; ++srcColIdx) {
                uint32_t inputColOffset = srcColIdx * kChannelCount;
                uint32_t outputColBase = outputRowBase + (srcColIdx * kBlockSize) * kChannelCount;
                AscendC::DataCopy(ubOutputRows[outputColBase],
                                  ubPhaseFirst[inputColOffset], columnParams);
                AscendC::DataCopy(ubOutputRows[outputColBase + kChannelCount],
                                  ubPhaseSecond[inputColOffset], columnParams);
            }
            B2SBarrierAll();
        }

        uint32_t dstLinearOffset = (dstBatchIdx * kResultH + outputRowStart) * kResultW * kChannelCount;
        storeMove.blockLen = static_cast<uint16_t>(activeOutputRows * kResultRowBlocks);
        AscendC::DataCopy(gmOutput[dstLinearOffset], ubOutputRows, storeMove);
        B2SBarrierStore();
    }
}

// Single-row tiling is best for compact float32 rows where two input phases fit
// comfortably in UB and can be written as one output row.
template <class DT_X, uint32_t kSourceH, uint32_t kSourceW, uint32_t kChannelCount, uint32_t kOutputBatchCount,
          uint32_t kBufferAlignBytes, bool kLoadBarrierAll, bool kUbBarrierAll>
__aicore__ inline void RunR2SingleRowTile(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kResultH = kSourceH * 2U;
    constexpr uint32_t kResultW = kSourceW * 2U;
    constexpr uint32_t kBlockSize = 2U;
    constexpr uint32_t kChannelBlocks = (kChannelCount * sizeof(DT_X)) / 32U;
    constexpr uint32_t kInputRowBlocks = (kSourceW * kChannelCount * sizeof(DT_X)) / 32U;
    constexpr uint32_t kResultRowBlocks = (kResultW * kChannelCount * sizeof(DT_X)) / 32U;
    constexpr uint32_t kSourceElemCount = kSourceW * kChannelCount;
    constexpr uint32_t kResultElemCount = kResultW * kChannelCount;
    constexpr uint32_t kSourceByteCount =
        ((kSourceElemCount * sizeof(DT_X) + kBufferAlignBytes - 1U) / kBufferAlignBytes) * kBufferAlignBytes;

    AscendC::GlobalTensor<DT_X> gmInput;
    AscendC::GlobalTensor<DT_X> gmOutput;
    gmInput.SetGlobalBuffer((__gm__ DT_X*)x);
    gmOutput.SetGlobalBuffer((__gm__ DT_X*)y);
    AscendC::LocalTensor<DT_X> ubPhaseFirst(AscendC::TPosition::VECCALC, 0, kSourceElemCount);
    AscendC::LocalTensor<DT_X> ubPhaseSecond(AscendC::TPosition::VECCALC,
                                              kSourceByteCount, kSourceElemCount);
    AscendC::LocalTensor<DT_X> ubOutputRows(AscendC::TPosition::VECCALC,
                                              2U * kSourceByteCount, kResultElemCount);

    AscendC::DataCopyParams loadMove;
    loadMove.blockCount = 1;
    loadMove.blockLen = static_cast<uint16_t>(kInputRowBlocks);
    loadMove.srcStride = 0;
    loadMove.dstStride = 0;

    AscendC::DataCopyParams ubParams;
    ubParams.blockCount = static_cast<uint16_t>(kSourceW);
    ubParams.blockLen = static_cast<uint16_t>(kChannelBlocks);
    ubParams.srcStride = 0;
    ubParams.dstStride = static_cast<uint16_t>(kChannelBlocks);

    AscendC::DataCopyParams storeMove;
    storeMove.blockCount = 1;
    storeMove.blockLen = static_cast<uint16_t>(kResultRowBlocks);
    storeMove.srcStride = 0;
    storeMove.dstStride = 0;

    const uint32_t blockIdx = B2SCoreIndex();
    const uint32_t blockNum = B2SCoreCount();
    for (uint32_t rowGroupId = blockIdx; rowGroupId < kOutputBatchCount * kBlockSize * kSourceH;
         rowGroupId += blockNum) {
        uint32_t srcRowIdx = rowGroupId % kSourceH;
        uint32_t batchPhaseId = rowGroupId / kSourceH;
        uint32_t phaseH = batchPhaseId & 1U;
        uint32_t dstBatchIdx = batchPhaseId >> 1;
        uint32_t srcBatchFirst = phaseH * kBlockSize * kOutputBatchCount + dstBatchIdx;
        uint32_t srcBatchSecond = srcBatchFirst + kOutputBatchCount;
        uint32_t srcOffsetFirst = ((srcBatchFirst * kSourceH + srcRowIdx) * kSourceW) * kChannelCount;
        uint32_t srcOffsetSecond = ((srcBatchSecond * kSourceH + srcRowIdx) * kSourceW) * kChannelCount;

        AscendC::DataCopy(ubPhaseFirst, gmInput[srcOffsetFirst], loadMove);
        AscendC::DataCopy(ubPhaseSecond, gmInput[srcOffsetSecond], loadMove);
        if constexpr (kLoadBarrierAll) {
            B2SBarrierAll();
        } else {
            B2SBarrierLoad();
        }

        // Interleave the two width phases into one output row.
        AscendC::DataCopy(ubOutputRows, ubPhaseFirst, ubParams);
        AscendC::DataCopy(ubOutputRows[kChannelCount], ubPhaseSecond, ubParams);
        if constexpr (kUbBarrierAll) {
            B2SBarrierAll();
        } else {
            B2SBarrierLoad();
        }

        uint32_t dstLinearOffset =
            (dstBatchIdx * kResultH + phaseH + srcRowIdx * kBlockSize) * kResultW * kChannelCount;
        AscendC::DataCopy(gmOutput[dstLinearOffset], ubOutputRows, storeMove);
        B2SBarrierStore();
    }
}

// Wide-row scatter keeps the original double-buffered pipeline: while one UB
// tile is stored with output stride, the next input tile can be loaded.
struct R2ScatterRowTile {
    uint32_t source_batch;
    uint32_t source_row;
    uint32_t height_phase;
    uint32_t width_phase;
    uint32_t result_row;
    uint64_t source_base;
    uint64_t result_base;
};

template <uint32_t kSourceH, uint32_t kSourceW, uint32_t kChannelCount, uint32_t kResultW>
__aicore__ inline R2ScatterRowTile MakeR2ScatterRowTile(uint32_t linearRow) {
    constexpr uint32_t kBlock = 2U;
    R2ScatterRowTile tile = {};
    tile.source_batch = linearRow / kSourceH;
    tile.source_row = linearRow - tile.source_batch * kSourceH;
    tile.height_phase = tile.source_batch >> 1;
    tile.width_phase = tile.source_batch & 1U;
    tile.result_row = tile.source_row * kBlock + tile.height_phase;
    tile.source_base = static_cast<uint64_t>(linearRow) * kSourceW * kChannelCount;
    tile.result_base = (static_cast<uint64_t>(tile.result_row) * kResultW +
                        tile.width_phase) * kChannelCount;
    return tile;
}

template <class DT_X, uint32_t kSourceH, uint32_t kSourceW, uint32_t kChannelCount, uint32_t kTileCols>
__aicore__ inline void RunR2WideInputScatter(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kResultH = kSourceH * 2U;
    constexpr uint32_t kResultW = kSourceW * 2U;
    constexpr uint32_t kBlockSize = 2U;
    constexpr uint32_t kInputBatch = 4U;
    constexpr uint32_t kTotalRows = kInputBatch * kSourceH;
    constexpr uint32_t kBlockNum = 40U;
    constexpr uint32_t kCBytes = kChannelCount * sizeof(DT_X);
    constexpr uint32_t kChannelBlocks = kCBytes / 32U;
    constexpr uint32_t kTileElems = kTileCols * kChannelCount;
    constexpr event_t kEvent0 = static_cast<event_t>(0);
    constexpr event_t kEvent1 = static_cast<event_t>(1);

    AscendC::GlobalTensor<DT_X> gmInput;
    AscendC::GlobalTensor<DT_X> gmOutput;
    gmInput.SetGlobalBuffer((__gm__ DT_X*)x);
    gmOutput.SetGlobalBuffer((__gm__ DT_X*)y);
    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub;
    AscendC::LocalTensor<DT_X> ubTileEven =
        ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(kTileElems);
    AscendC::LocalTensor<DT_X> ubTileOdd =
        ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(kTileElems);

    AscendC::DataCopyParams loadMove;
    loadMove.blockCount = 1;
    loadMove.blockLen = 0;
    loadMove.srcStride = 0;
    loadMove.dstStride = 0;

    AscendC::DataCopyExtParams storeMove;
    storeMove.blockCount = 1;
    storeMove.blockLen = kCBytes;
    storeMove.srcStride = 0;
    storeMove.dstStride = kCBytes;
    storeMove.rsv = 0;

    const uint32_t blockIdx = B2SCoreIndex();
    // This schedule copies one input row at a time and scatters columns with a
    // stride of block_size in the output row.
    if constexpr (kTotalRows <= kBlockNum) {
        if (blockIdx >= kTotalRows) {
            return;
        }
        const R2ScatterRowTile rowTile =
            MakeR2ScatterRowTile<kSourceH, kSourceW, kChannelCount, kResultW>(blockIdx);

        if constexpr ((kSourceW % kTileCols) == 0U) {
            constexpr uint32_t kTileCount = kSourceW / kTileCols;
            loadMove.blockLen = static_cast<uint16_t>(kTileCols * kChannelBlocks);
            storeMove.blockCount = static_cast<uint16_t>(kTileCols);
            AscendC::DataCopy(ubTileEven, gmInput[rowTile.source_base], loadMove);
            B2SMarkTileLoaded(kEvent0);

#pragma unroll
            for (uint32_t tile = 0; tile < kTileCount; ++tile) {
                uint32_t bufferParity = tile & 1U;
                event_t activeEvent = B2SEventByParity(bufferParity, kEvent0, kEvent1);
                event_t prefetchEvent = B2SEventByParity(bufferParity ^ 1U, kEvent0, kEvent1);
                AscendC::LocalTensor<DT_X> activeTile =
                    B2STileByParity<DT_X>(bufferParity, ubTileEven, ubTileOdd);
                AscendC::LocalTensor<DT_X> prefetchTile =
                    B2STileByParity<DT_X>(bufferParity ^ 1U, ubTileEven, ubTileOdd);

                B2SWaitTileLoaded(activeEvent);
                if (tile + 1U < kTileCount) {
                    uint64_t nextInputIndex =
                        rowTile.source_base + static_cast<uint64_t>(tile + 1U) * kTileCols * kChannelCount;
                    AscendC::DataCopy(prefetchTile, gmInput[nextInputIndex], loadMove);
                    B2SMarkTileLoaded(prefetchEvent);
                }

                uint64_t dstLinearOffset =
                    rowTile.result_base +
                    static_cast<uint64_t>(tile) * kTileCols * kBlockSize * kChannelCount;
                AscendC::DataCopyPad(gmOutput[dstLinearOffset], activeTile, storeMove);
                B2SMarkTileStored(activeEvent);
                B2SWaitTileStored(activeEvent);
            }
            return;
        }

        uint32_t finishedColumns = 0U;
        uint32_t liveColumnCount = kSourceW > kTileCols ? kTileCols : kSourceW;
        loadMove.blockLen = static_cast<uint16_t>(liveColumnCount * kChannelBlocks);
        AscendC::DataCopy(ubTileEven, gmInput[rowTile.source_base], loadMove);
        B2SMarkTileLoaded(kEvent0);

        // Ping-pong the two UB tile buffers so the next load can be issued while
        // the current tile is being scattered to GM.
        uint32_t bufferParity = 0U;
        while (finishedColumns < kSourceW) {
            event_t activeEvent = B2SEventByParity(bufferParity, kEvent0, kEvent1);
            event_t prefetchEvent = B2SEventByParity(bufferParity ^ 1U, kEvent0, kEvent1);
            AscendC::LocalTensor<DT_X> activeTile =
                    B2STileByParity<DT_X>(bufferParity, ubTileEven, ubTileOdd);
            AscendC::LocalTensor<DT_X> prefetchTile =
                    B2STileByParity<DT_X>(bufferParity ^ 1U, ubTileEven, ubTileOdd);

            B2SWaitTileLoaded(activeEvent);

            uint32_t nextProcessedCols = finishedColumns + liveColumnCount;
            uint32_t nextActiveCols = 0U;
            if (nextProcessedCols < kSourceW) {
                nextActiveCols = kSourceW - nextProcessedCols;
                if (nextActiveCols > kTileCols) {
                    nextActiveCols = kTileCols;
                }
                uint64_t nextInputIndex =
                    rowTile.source_base + static_cast<uint64_t>(nextProcessedCols) * kChannelCount;
                loadMove.blockLen = static_cast<uint16_t>(nextActiveCols * kChannelBlocks);
                AscendC::DataCopy(prefetchTile, gmInput[nextInputIndex], loadMove);
                B2SMarkTileLoaded(prefetchEvent);
            }

            uint64_t dstLinearOffset =
                rowTile.result_base + static_cast<uint64_t>(finishedColumns) * kBlockSize * kChannelCount;
            storeMove.blockCount = static_cast<uint16_t>(liveColumnCount);
            AscendC::DataCopyPad(gmOutput[dstLinearOffset], activeTile, storeMove);
            B2SMarkTileStored(activeEvent);
            B2SWaitTileStored(activeEvent);

            finishedColumns = nextProcessedCols;
            liveColumnCount = nextActiveCols;
            bufferParity ^= 1U;
        }
        return;
    }

    uint32_t rowsPerWorker = (kTotalRows + kBlockNum - 1U) / kBlockNum;
    uint32_t firstRow = blockIdx * rowsPerWorker;
    if (firstRow >= kTotalRows) {
        return;
    }
    uint32_t activeRowCount = rowsPerWorker;
    if (firstRow + activeRowCount > kTotalRows) {
        activeRowCount = kTotalRows - firstRow;
    }

    for (uint32_t rowOffset = 0U; rowOffset < activeRowCount; ++rowOffset) {
        const R2ScatterRowTile rowTile =
            MakeR2ScatterRowTile<kSourceH, kSourceW, kChannelCount, kResultW>(firstRow + rowOffset);

        if constexpr ((kSourceW % kTileCols) == 0U) {
            constexpr uint32_t kTileCount = kSourceW / kTileCols;
            loadMove.blockLen = static_cast<uint16_t>(kTileCols * kChannelBlocks);
            storeMove.blockCount = static_cast<uint16_t>(kTileCols);
            AscendC::DataCopy(ubTileEven, gmInput[rowTile.source_base], loadMove);
            B2SMarkTileLoaded(kEvent0);

#pragma unroll
            for (uint32_t tile = 0; tile < kTileCount; ++tile) {
                uint32_t bufferParity = tile & 1U;
                event_t activeEvent = B2SEventByParity(bufferParity, kEvent0, kEvent1);
                event_t prefetchEvent = B2SEventByParity(bufferParity ^ 1U, kEvent0, kEvent1);
                AscendC::LocalTensor<DT_X> activeTile =
                    B2STileByParity<DT_X>(bufferParity, ubTileEven, ubTileOdd);
                AscendC::LocalTensor<DT_X> prefetchTile =
                    B2STileByParity<DT_X>(bufferParity ^ 1U, ubTileEven, ubTileOdd);

                B2SWaitTileLoaded(activeEvent);
                if (tile + 1U < kTileCount) {
                    uint64_t nextInputIndex =
                        rowTile.source_base + static_cast<uint64_t>(tile + 1U) * kTileCols * kChannelCount;
                    AscendC::DataCopy(prefetchTile, gmInput[nextInputIndex], loadMove);
                    B2SMarkTileLoaded(prefetchEvent);
                }

                uint64_t dstLinearOffset =
                    rowTile.result_base +
                    static_cast<uint64_t>(tile) * kTileCols * kBlockSize * kChannelCount;
                AscendC::DataCopyPad(gmOutput[dstLinearOffset], activeTile, storeMove);
                B2SMarkTileStored(activeEvent);
                B2SWaitTileStored(activeEvent);
            }
            continue;
        }

        uint32_t finishedColumns = 0U;
        uint32_t liveColumnCount = kSourceW > kTileCols ? kTileCols : kSourceW;
        loadMove.blockLen = static_cast<uint16_t>(liveColumnCount * kChannelBlocks);
        AscendC::DataCopy(ubTileEven, gmInput[rowTile.source_base], loadMove);
        B2SMarkTileLoaded(kEvent0);

        // Ping-pong the two UB tile buffers for the non-divisible tail tiles too.
        uint32_t bufferParity = 0U;
        while (finishedColumns < kSourceW) {
            event_t activeEvent = B2SEventByParity(bufferParity, kEvent0, kEvent1);
            event_t prefetchEvent = B2SEventByParity(bufferParity ^ 1U, kEvent0, kEvent1);
            AscendC::LocalTensor<DT_X> activeTile =
                    B2STileByParity<DT_X>(bufferParity, ubTileEven, ubTileOdd);
            AscendC::LocalTensor<DT_X> prefetchTile =
                    B2STileByParity<DT_X>(bufferParity ^ 1U, ubTileEven, ubTileOdd);

            B2SWaitTileLoaded(activeEvent);

            uint32_t nextProcessedCols = finishedColumns + liveColumnCount;
            uint32_t nextActiveCols = 0U;
            if (nextProcessedCols < kSourceW) {
                nextActiveCols = kSourceW - nextProcessedCols;
                if (nextActiveCols > kTileCols) {
                    nextActiveCols = kTileCols;
                }
                uint64_t nextInputIndex =
                    rowTile.source_base + static_cast<uint64_t>(nextProcessedCols) * kChannelCount;
                loadMove.blockLen = static_cast<uint16_t>(nextActiveCols * kChannelBlocks);
                AscendC::DataCopy(prefetchTile, gmInput[nextInputIndex], loadMove);
                B2SMarkTileLoaded(prefetchEvent);
            }

            uint64_t dstLinearOffset =
                rowTile.result_base + static_cast<uint64_t>(finishedColumns) * kBlockSize * kChannelCount;
            storeMove.blockCount = static_cast<uint16_t>(liveColumnCount);
            AscendC::DataCopyPad(gmOutput[dstLinearOffset], activeTile, storeMove);
            B2SMarkTileStored(activeEvent);
            B2SWaitTileStored(activeEvent);

            finishedColumns = nextProcessedCols;
            liveColumnCount = nextActiveCols;
            bufferParity ^= 1U;
        }
    }
}

// Helpers for the block_size=4 cropped lane; crop-left shifts the four batch
// phases, so the first source bank starts one column ahead.
template <class DT_X, uint32_t kRows, uint32_t kCols, uint32_t kChannels,
          uint32_t kGroupElems, uint32_t kCBlocks, uint32_t kGroupsPerTile>
__aicore__ inline void LoadR4ShiftedBanks(AscendC::GlobalTensor<DT_X> &gmSource,
                                            AscendC::LocalTensor<DT_X> bank01,
                                            AscendC::LocalTensor<DT_X> bank23,
                                            AscendC::DataCopyParams &copy,
                                            uint32_t batchPhase, uint32_t row,
                                            uint32_t col, uint32_t headGroups) {
    uint32_t phaseBase = batchPhase * 4U;
    uint32_t base0 = ((phaseBase * kRows + row) * kCols + col + 1U) * kChannels;
    uint32_t base1 = (((phaseBase + 1U) * kRows + row) * kCols + col) * kChannels;
    uint32_t base2 = (((phaseBase + 2U) * kRows + row) * kCols + col) * kChannels;
    uint32_t base3 = (((phaseBase + 3U) * kRows + row) * kCols + col) * kChannels;

    copy.blockLen = static_cast<uint16_t>(headGroups * kCBlocks);
    AscendC::DataCopy(bank01, gmSource[base0], copy);
    copy.blockLen = static_cast<uint16_t>(kGroupsPerTile * kCBlocks);
    AscendC::DataCopy(bank01[kGroupElems], gmSource[base1], copy);
    AscendC::DataCopy(bank23, gmSource[base2], copy);
    AscendC::DataCopy(bank23[kGroupElems], gmSource[base3], copy);
}

template <class DT_X, uint32_t kChannels, uint32_t kGroupElems,
          uint32_t kCBlocks, uint32_t kGroupsPerTile>
__aicore__ inline void StitchR4ShiftedBanks(AscendC::LocalTensor<DT_X> rowTile,
                                               AscendC::LocalTensor<DT_X> bank01,
                                               AscendC::LocalTensor<DT_X> bank23,
                                               AscendC::DataCopyParams &copy,
                                               uint32_t headGroups) {
    copy.blockCount = static_cast<uint16_t>(headGroups);
    AscendC::DataCopy(rowTile[3U * kChannels], bank01, copy);
    copy.blockCount = static_cast<uint16_t>(kGroupsPerTile);
    AscendC::DataCopy(rowTile, bank01[kGroupElems], copy);
    AscendC::DataCopy(rowTile[kChannels], bank23, copy);
    AscendC::DataCopy(rowTile[2U * kChannels], bank23[kGroupElems], copy);
}

template <class DT_X, uint32_t kSourceH, uint32_t kSourceW, uint32_t kChannelCount, uint32_t kResultH, uint32_t kResultW,
          uint32_t kCropLeft, uint32_t kTileCols>
__aicore__ inline void RunR4LeftCropInterleave(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kBlock = 4U;
    constexpr uint32_t kTilesPerRow = (kResultW + kTileCols - 1U) / kTileCols;
    constexpr uint32_t kPhaseGroups = kTileCols / kBlock;
    constexpr uint32_t kLaneBytes = kChannelCount * sizeof(DT_X);
    constexpr uint32_t kLaneBlocks = kLaneBytes / 32U;
    constexpr uint32_t kPhaseBytes = kPhaseGroups * kLaneBytes;
    constexpr uint32_t kPhaseElems = kPhaseBytes / sizeof(DT_X);
    constexpr uint32_t kResultElemCount = kTileCols * kChannelCount;
    constexpr uint32_t kInputPairElems = 2U * kPhaseElems;

    AscendC::GlobalTensor<DT_X> gmSource;
    AscendC::GlobalTensor<DT_X> gmResult;
    BindIoWindow(x, y, gmSource, gmResult);
    AscendC::LocalTensor<DT_X> rowTile(AscendC::TPosition::VECCALC, 0, kResultElemCount);
    AscendC::LocalTensor<DT_X> bank01(AscendC::TPosition::VECCALC, 65536, kInputPairElems);
    AscendC::LocalTensor<DT_X> bank23(AscendC::TPosition::VECCALC, 98304, kInputPairElems);

    AscendC::DataCopyParams loadMove;
    ResetBlockTransfer(loadMove);

    AscendC::DataCopyParams ubParams;
    ResetBlockTransfer(ubParams);
    ubParams.blockLen = static_cast<uint16_t>(kLaneBlocks);
    ubParams.srcStride = 0;
    ubParams.dstStride = static_cast<uint16_t>((kBlock - 1U) * kLaneBlocks);

    AscendC::DataCopyParams storeMove;
    ResetBlockTransfer(storeMove);

    const uint32_t worker = B2SCoreIndex();
    const uint32_t workerCount = B2SCoreCount();
    for (uint32_t outRow = worker; outRow < kResultH; outRow += workerCount) {
        uint32_t srcRowIdx = outRow / kBlock;
        uint32_t hPhase = outRow - srcRowIdx * kBlock;
        for (uint32_t tileId = 0; tileId < kTilesPerRow; ++tileId) {
            uint32_t outCol = tileId * kTileCols;
            uint32_t liveCols = kResultW - outCol;
            if (liveCols > kTileCols) {
                liveCols = kTileCols;
            }
            uint32_t srcColIdx = (outCol + kCropLeft) / kBlock;
            uint32_t headGroups = liveCols == kTileCols ? kPhaseGroups : kPhaseGroups - 1U;

            LoadR4ShiftedBanks<DT_X, kSourceH, kSourceW, kChannelCount, kPhaseElems,
                                  kLaneBlocks, kPhaseGroups>(
                gmSource, bank01, bank23, loadMove, hPhase, srcRowIdx, srcColIdx, headGroups);
            B2SBarrierAll();

            StitchR4ShiftedBanks<DT_X, kChannelCount, kPhaseElems, kLaneBlocks, kPhaseGroups>(
                rowTile, bank01, bank23, ubParams, headGroups);
            B2SBarrierAll();

            storeMove.blockLen = static_cast<uint16_t>((liveCols * kLaneBytes) / 32U);
            AscendC::DataCopy(gmResult[(outRow * kResultW + outCol) * kChannelCount], rowTile, storeMove);
            if (tileId + 1U < kTilesPerRow || outRow + workerCount < kResultH) {
                B2SBarrierStore();
            }
        }
    }
}

// Dense cropped fp16 lane: each task produces half an output row by gathering
// two 64-column input spans into 127 visible output columns.
template <class DT_X>
__aicore__ inline void LoadR2DenseHalfRow(AscendC::GlobalTensor<DT_X> &gmInput,
                                                 AscendC::LocalTensor<DT_X> slot,
                                                 uint32_t task) {
    constexpr uint32_t kSourceH = 128;
    constexpr uint32_t kSourceW = 128;
    constexpr uint32_t kChannelCount = 65;
    constexpr uint32_t kCropTop = 1;
    constexpr uint32_t kLaneCols = 64U;
    constexpr uint32_t kLaneElems = kLaneCols * kChannelCount;

    uint32_t dstRowIdx = task >> 1U;
    uint32_t halfId = task & 1U;
    uint32_t uncroppedRow = dstRowIdx + kCropTop;
    uint32_t srcRowIdx = uncroppedRow >> 1;
    uint32_t phaseH = uncroppedRow & 1U;
    uint32_t srcBatchFirst = phaseH * 2U + (halfId == 0U ? 1U : 0U);
    uint32_t srcBatchSecond = phaseH * 2U + (halfId == 0U ? 0U : 1U);
    uint32_t firstInputCol = halfId == 0U ? 0U : 64U;
    uint32_t secondInputCol = halfId == 0U ? 1U : 64U;
    uint64_t srcOffsetFirst =
        ((static_cast<uint64_t>(srcBatchFirst) * kSourceH + srcRowIdx) * kSourceW +
         firstInputCol) * kChannelCount;
    uint64_t srcOffsetSecond =
        ((static_cast<uint64_t>(srcBatchSecond) * kSourceH + srcRowIdx) * kSourceW +
         secondInputCol) * kChannelCount;
    AscendC::DataCopy(slot, gmInput[srcOffsetFirst], kLaneElems);
    AscendC::DataCopy(slot[kLaneElems], gmInput[srcOffsetSecond], kLaneElems);
}

template <class DT_X>
__aicore__ inline uint64_t LocateR2DenseHalfOutput(uint32_t task) {
    constexpr uint32_t kChannelCount = 65;
    constexpr uint32_t kResultW = 254;
    constexpr uint32_t kHalfCols = kResultW / 2U;
    uint32_t dstRowIdx = task >> 1U;
    uint32_t halfId = task & 1U;
    return (static_cast<uint64_t>(dstRowIdx) * kResultW + halfId * kHalfCols) * kChannelCount;
}

template <class DT_X>
__aicore__ inline void RunR2DenseCropGather(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kChannelCount = 65;
    constexpr uint32_t kResultH = 254;
    constexpr uint32_t kResultW = 254;
    constexpr uint32_t kHalfCols = kResultW / 2U;
    constexpr uint32_t kLaneCols = 64U;
    constexpr uint32_t kLaneElems = kLaneCols * kChannelCount;
    constexpr uint32_t kSourceElemCount = 2U * kLaneElems;
    constexpr uint32_t kResultElemCount = kHalfCols * kChannelCount;
    constexpr uint32_t kSlotElems = 16576U;
    constexpr uint32_t kSlotBytes = kSlotElems * sizeof(DT_X);
    constexpr uint32_t kIndexBaseBytes = 2U * kSlotBytes;
    constexpr uint32_t kPairElems = 2U * kChannelCount;
    constexpr uint32_t kPatternPairs = 4U;
    constexpr uint32_t kPatternElems = kPatternPairs * kPairElems;
    constexpr uint32_t kFullPatterns = kResultElemCount / kPatternElems;
    constexpr uint32_t kTailIndexElems =
        ((kResultElemCount - kFullPatterns * kPatternElems + 7U) / 8U) * 8U;
    constexpr uint32_t kIndexElems = kFullPatterns * kPatternElems + kTailIndexElems;
    constexpr event_t kScalarToVectorEvent = static_cast<event_t>(6);
    constexpr event_t kFreeEvent0 = static_cast<event_t>(0);
    constexpr event_t kFreeEvent1 = static_cast<event_t>(1);
    constexpr event_t kLoadEvent0 = static_cast<event_t>(2);
    constexpr event_t kLoadEvent1 = static_cast<event_t>(3);
    constexpr event_t kGatherEvent0 = static_cast<event_t>(4);
    constexpr event_t kGatherEvent1 = static_cast<event_t>(5);
    static_assert(kSourceElemCount + kResultElemCount <= kSlotElems, "dense gather slot overflow");
    static_assert(kIndexBaseBytes + kIndexElems * sizeof(uint32_t) <= 184U * 1024U,
                  "dense gather UB overflow");

    AscendC::GlobalTensor<DT_X> gmInput;
    AscendC::GlobalTensor<DT_X> gmOutput;
    gmInput.SetGlobalBuffer((__gm__ DT_X*)x);
    gmOutput.SetGlobalBuffer((__gm__ DT_X*)y);

    AscendC::LocalTensor<DT_X> slot0(AscendC::TPosition::VECCALC, 0, kSlotElems);
    AscendC::LocalTensor<DT_X> slot1(AscendC::TPosition::VECCALC, kSlotBytes, kSlotElems);
    AscendC::LocalTensor<int32_t> indexLocal(AscendC::TPosition::VECCALC,
                                             kIndexBaseBytes, kIndexElems);
    AscendC::LocalTensor<uint32_t> ubGatherMap(AscendC::TPosition::VECCALC,
                                               kIndexBaseBytes, kResultElemCount);

    for (uint32_t pair = 0; pair < kPatternPairs; ++pair) {
        uint32_t elemBase = pair * kPairElems;
        uint32_t byteBase = pair * kChannelCount * sizeof(DT_X);
        for (uint32_t c = 0; c < kChannelCount; ++c) {
            indexLocal.SetValue(elemBase + c, static_cast<int32_t>(byteBase + c * sizeof(DT_X)));
            indexLocal.SetValue(elemBase + kChannelCount + c,
                                static_cast<int32_t>(kLaneElems * sizeof(DT_X) +
                                                     byteBase + c * sizeof(DT_X)));
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(kScalarToVectorEvent);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(kScalarToVectorEvent);
    for (uint32_t pat = 1; pat <= kFullPatterns; ++pat) {
        uint32_t count = (pat == kFullPatterns) ? kTailIndexElems : kPatternElems;
        AscendC::Adds(indexLocal[pat * kPatternElems],
                      indexLocal[(pat - 1U) * kPatternElems],
                      static_cast<int32_t>(kPatternPairs * kChannelCount * sizeof(DT_X)), count);
    }
    B2SBarrierVector();

    AscendC::DataCopyExtParams storeMove;
    storeMove.blockCount = 1;
    storeMove.blockLen = kResultElemCount * sizeof(DT_X);
    storeMove.srcStride = 0;
    storeMove.dstStride = 0;
    storeMove.rsv = 0;

    const uint32_t blockIdx = B2SCoreIndex();
    const uint32_t blockNum = B2SCoreCount();
    constexpr uint32_t kTaskCount = kResultH * 2U;
    uint32_t tasksPerCore = kTaskCount / blockNum;
    uint32_t extraTasks = kTaskCount - tasksPerCore * blockNum;
    uint32_t startTask = blockIdx * tasksPerCore + (blockIdx < extraTasks ? blockIdx : extraTasks);
    uint32_t taskCount = tasksPerCore + (blockIdx < extraTasks ? 1U : 0U);
    if (taskCount == 0U) {
        return;
    }

    B2SMarkTileStored(kFreeEvent0);
    B2SMarkTileStored(kFreeEvent1);
    B2SWaitTileStored(kFreeEvent0);
    LoadR2DenseHalfRow<DT_X>(gmInput, slot0, startTask);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kLoadEvent0);

    for (uint32_t idx = 0; idx < taskCount; ++idx) {
        uint32_t buf = idx & 1U;
        uint32_t task = startTask + idx;
        AscendC::LocalTensor<DT_X> slot = buf == 0U ? slot0 : slot1;
        event_t loadEvent = buf == 0U ? kLoadEvent0 : kLoadEvent1;
        event_t gatherEvent = buf == 0U ? kGatherEvent0 : kGatherEvent1;
        event_t freeEvent = buf == 0U ? kFreeEvent0 : kFreeEvent1;

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(loadEvent);
        AscendC::Gather(slot[kSourceElemCount], slot, ubGatherMap, 0U, kResultElemCount);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(gatherEvent);

        uint32_t next = idx + 1U;
        if (next < taskCount) {
            uint32_t nextBuf = next & 1U;
            event_t nextFreeEvent = nextBuf == 0U ? kFreeEvent0 : kFreeEvent1;
            event_t nextLoadEvent = nextBuf == 0U ? kLoadEvent0 : kLoadEvent1;
            AscendC::LocalTensor<DT_X> nextSlot = nextBuf == 0U ? slot0 : slot1;
            B2SWaitTileStored(nextFreeEvent);
            LoadR2DenseHalfRow<DT_X>(gmInput, nextSlot, startTask + next);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(nextLoadEvent);
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(gatherEvent);
        AscendC::DataCopyPad(gmOutput[LocateR2DenseHalfOutput<DT_X>(task)],
                             slot[kSourceElemCount], storeMove);
        B2SMarkTileStored(freeEvent);
    }
    B2SWaitTileStored(kFreeEvent0);
    B2SWaitTileStored(kFreeEvent1);
}

// Runtime fallback covers arbitrary legal NHWC shapes by processing stripes:
// fixed output batch, fixed output row, and one width phase at a time.
template <class DT_X>
class RuntimeB2SFallback {
public:
    __aicore__ inline RuntimeB2SFallback() {}

    struct FallbackStripeTile {
        uint32_t result_batch;
        uint32_t result_row;
        uint32_t first_result_col;
        uint32_t column_count;
        uint32_t source_batch;
        uint32_t source_row;
        bool valid;
    };

    __aicore__ inline void SetupRuntimePlan(GM_ADDR x, GM_ADDR y, const B2SDispatchPlan &plan) {
        gmInput.SetGlobalBuffer((__gm__ DT_X*)x);
        gmOutput.SetGlobalBuffer((__gm__ DT_X*)y);

        tempElementLimit = plan.buffer_element_quota == 0U ?
            kDefaultBufferBytes / sizeof(DT_X) : plan.buffer_element_quota;
        tempByteLimit = tempElementLimit * sizeof(DT_X);
        tempByteLimit = (tempByteLimit / kBlockBytes) * kBlockBytes;
        if (tempByteLimit < kBlockBytes) {
            tempByteLimit = kBlockBytes;
        }
        tempElementLimit = tempByteLimit / sizeof(DT_X);
        pipe.InitBuffer(tempBuffer, tempByteLimit);
        tempTensor = tempBuffer.Get<DT_X>();

        inputShape = plan.input;
        outputShape = plan.output;
        cropEdges = plan.crop;
        blockSize = plan.block_extent;
        runtimePath = plan.runtime_path;
        // Each stripe fixes output batch, output row, and width phase. The loop
        // then copies all output columns belonging to that width phase.
        stripeCount = outputShape.batch_count * outputShape.height_extent * blockSize;
        channelBytes = outputShape.channel_count * sizeof(DT_X);
        alignedChannelBytes =
            ((channelBytes + kBlockBytes - 1U) / kBlockBytes) * kBlockBytes;
    }

    __aicore__ inline void RunRuntimePath() {
        if (runtimePath == B2S_RUNTIME_FLAT_COPY) {
            RunRuntimeFlatCopy();
            return;
        }
        if (runtimePath == B2S_RUNTIME_PIXEL ||
            runtimePath == B2S_RUNTIME_WIDE_PIXEL) {
            RunRuntimePixels();
            return;
        }
        if (runtimePath == B2S_RUNTIME_PIXEL_CHANNEL) {
            RunRuntimePixelChannelTiles();
            return;
        }

        RunRuntimeStripes();
    }

    __aicore__ inline FallbackStripeTile B2SResolveStripe(uint32_t stripeId) {
        uint32_t phaseW = stripeId % blockSize;
        uint32_t rowTicket = stripeId / blockSize;
        FallbackStripeTile tile = {};
        tile.result_row = rowTicket % outputShape.height_extent;
        tile.result_batch = rowTicket / outputShape.height_extent;
        uint32_t leftPhase = cropEdges.trim_left % blockSize;
        tile.first_result_col = (phaseW + blockSize - leftPhase) % blockSize;
        if (tile.first_result_col >= outputShape.width_extent) {
            tile.valid = false;
            return tile;
        }
        tile.column_count = ((outputShape.width_extent - tile.first_result_col - 1U) / blockSize) + 1U;

        // Add crop offsets back to locate the uncropped spatial coordinate, then
        // split it into the input row and the height phase stored in batch.
        uint32_t paddedRow = tile.result_row + cropEdges.trim_top;
        tile.source_row = paddedRow / blockSize;
        uint32_t phaseH = paddedRow % blockSize;
        tile.source_batch = (phaseH * blockSize + phaseW) *
            outputShape.batch_count + tile.result_batch;
        tile.valid = true;
        return tile;
    }

    __aicore__ inline uint32_t B2SMinU32(uint32_t left, uint32_t right) const {
        return left < right ? left : right;
    }

    __aicore__ inline uint32_t B2SDivCeilU32(uint32_t value, uint32_t unit) const {
        return unit == 0U ? 0U : (value + unit - 1U) / unit;
    }

    __aicore__ inline uint32_t B2SChannelTaskQuota() const {
        uint32_t quota = tempElementLimit;
        if (quota > kB2SRuntimeChannelTaskElements) {
            quota = kB2SRuntimeChannelTaskElements;
        }
        return quota == 0U ? 1U : quota;
    }

    __aicore__ inline void CopyRuntimeSpan(uint32_t srcOffset, uint32_t dstOffset,
                                           uint32_t elementCount) {
        AscendC::DataCopyPadExtParams<DT_X> noPad;
        DisableCopyPadding<DT_X>(noPad);

        AscendC::DataCopyExtParams loadMove;
        loadMove.blockCount = 1;
        loadMove.srcStride = 0;
        loadMove.dstStride = 0;
        loadMove.rsv = 0;

        AscendC::DataCopyExtParams storeMove;
        storeMove.blockCount = 1;
        storeMove.srcStride = 0;
        storeMove.dstStride = 0;
        storeMove.rsv = 0;

        for (uint32_t copied = 0U; copied < elementCount; copied += tempElementLimit) {
            const uint32_t liveElements = B2SMinU32(elementCount - copied, tempElementLimit);
            const uint32_t liveBytes = liveElements * sizeof(DT_X);
            loadMove.blockLen = liveBytes;
            storeMove.blockLen = liveBytes;
            AscendC::DataCopyPad(tempTensor, gmInput[srcOffset + copied], loadMove, noPad);
            B2SBarrierAll();
            AscendC::DataCopyPad(gmOutput[dstOffset + copied], tempTensor, storeMove);
            B2SBarrierAll();
        }
    }

    __aicore__ inline void RunRuntimeFlatCopy() {
        const uint32_t workerId = B2SCoreIndex();
        const uint32_t workerCount = B2SCoreCount();
        const uint32_t totalElements =
            outputShape.batch_count * outputShape.height_extent *
            outputShape.width_extent * outputShape.channel_count;
        const uint32_t beginElement = static_cast<uint32_t>(
            (static_cast<uint64_t>(totalElements) * workerId) / workerCount);
        const uint32_t endElement = static_cast<uint32_t>(
            (static_cast<uint64_t>(totalElements) * (workerId + 1U)) / workerCount);

        for (uint32_t cursor = beginElement; cursor < endElement; cursor += tempElementLimit) {
            const uint32_t liveElements = B2SMinU32(endElement - cursor, tempElementLimit);
            CopyRuntimeSpan(cursor, cursor, liveElements);
        }
    }

    __aicore__ inline void CopyRuntimePixelChannels(uint32_t outputPixel,
                                                    uint32_t channelStart,
                                                    uint32_t channelCount) {
        const uint32_t resultCol = outputPixel % outputShape.width_extent;
        uint32_t rowTicket = outputPixel / outputShape.width_extent;
        const uint32_t resultRow = rowTicket % outputShape.height_extent;
        const uint32_t resultBatch = rowTicket / outputShape.height_extent;

        const uint32_t paddedRow = resultRow + cropEdges.trim_top;
        const uint32_t paddedCol = resultCol + cropEdges.trim_left;
        const uint32_t sourceRow = paddedRow / blockSize;
        const uint32_t sourceCol = paddedCol / blockSize;
        const uint32_t phaseH = paddedRow % blockSize;
        const uint32_t phaseW = paddedCol % blockSize;
        const uint32_t sourceBatch =
            (phaseH * blockSize + phaseW) * outputShape.batch_count + resultBatch;

        const uint32_t srcOffset =
            ((sourceBatch * inputShape.height_extent + sourceRow) *
             inputShape.width_extent + sourceCol) *
            inputShape.channel_count + channelStart;
        const uint32_t dstOffset = outputPixel * outputShape.channel_count + channelStart;
        CopyRuntimeSpan(srcOffset, dstOffset, channelCount);
    }

    __aicore__ inline void RunRuntimePixels() {
        const uint32_t workerId = B2SCoreIndex();
        const uint32_t workerCount = B2SCoreCount();
        const uint32_t outputPixels =
            outputShape.batch_count * outputShape.height_extent * outputShape.width_extent;
        const uint32_t beginPixel = static_cast<uint32_t>(
            (static_cast<uint64_t>(outputPixels) * workerId) / workerCount);
        const uint32_t endPixel = static_cast<uint32_t>(
            (static_cast<uint64_t>(outputPixels) * (workerId + 1U)) / workerCount);

        for (uint32_t pixel = beginPixel; pixel < endPixel; ++pixel) {
            CopyRuntimePixelChannels(pixel, 0U, outputShape.channel_count);
        }
    }

    __aicore__ inline void RunRuntimePixelChannelTiles() {
        const uint32_t workerId = B2SCoreIndex();
        const uint32_t workerCount = B2SCoreCount();
        const uint32_t outputPixels =
            outputShape.batch_count * outputShape.height_extent * outputShape.width_extent;
        const uint32_t channelQuota = B2SChannelTaskQuota();
        const uint32_t channelTaskCount =
            B2SDivCeilU32(outputShape.channel_count, channelQuota);
        const uint32_t taskTotal = outputPixels * channelTaskCount;
        const uint32_t beginTask = static_cast<uint32_t>(
            (static_cast<uint64_t>(taskTotal) * workerId) / workerCount);
        const uint32_t endTask = static_cast<uint32_t>(
            (static_cast<uint64_t>(taskTotal) * (workerId + 1U)) / workerCount);

        for (uint32_t task = beginTask; task < endTask; ++task) {
            const uint32_t channelTask = task % channelTaskCount;
            const uint32_t outputPixel = task / channelTaskCount;
            const uint32_t channelStart = channelTask * channelQuota;
            const uint32_t channelCount =
                B2SMinU32(outputShape.channel_count - channelStart, channelQuota);
            CopyRuntimePixelChannels(outputPixel, channelStart, channelCount);
        }
    }

    __aicore__ inline void RunRuntimeStripes() {
        uint32_t workerId = B2SCoreIndex();
        uint32_t workerCount = B2SCoreCount();

        AscendC::DataCopyPadExtParams<DT_X> noPad;
        noPad.isPad = false;
        noPad.leftPadding = 0;
        noPad.rightPadding = 0;
        noPad.paddingValue = static_cast<DT_X>(0);

        uint32_t maxColumnsPerCopy = tempByteLimit / alignedChannelBytes;
        if (maxColumnsPerCopy == 0U) {
            maxColumnsPerCopy = 1U;
        }

        if (channelBytes > tempByteLimit) {
            constexpr uint32_t kMaxChunkBytes = 32U * 1024U;
            uint32_t maxChunkElems = kMaxChunkBytes / sizeof(DT_X);
            if (maxChunkElems > tempElementLimit) {
                maxChunkElems = tempElementLimit;
            }
            if (maxChunkElems == 0U) {
                maxChunkElems = 1U;
            }
            // Very wide channel dimensions cannot fit one whole channel vector
            // in UB, so channels are copied in chunks before advancing columns.
            for (uint32_t stripeId = workerId; stripeId < stripeCount; stripeId += workerCount) {
                const FallbackStripeTile stripe = B2SResolveStripe(stripeId);
                if (!stripe.valid) {
                    continue;
                }

                for (uint32_t channelOffset = 0U; channelOffset < outputShape.channel_count;
                     channelOffset += maxChunkElems) {
                    uint32_t chunkElems = outputShape.channel_count - channelOffset;
                    if (chunkElems > maxChunkElems) {
                        chunkElems = maxChunkElems;
                    }
                    uint32_t chunkBytes = chunkElems * sizeof(DT_X);
                    uint32_t alignedChunkBytes =
                        ((chunkBytes + kBlockBytes - 1U) / kBlockBytes) * kBlockBytes;
                    uint32_t columnsPerChunk = tempByteLimit / alignedChunkBytes;
                    if (columnsPerChunk == 0U) {
                        columnsPerChunk = 1U;
                    }

                    AscendC::DataCopyExtParams loadMove;
                    loadMove.blockLen = chunkBytes;
                    loadMove.srcStride = channelBytes - chunkBytes;
                    loadMove.dstStride = 0;
                    loadMove.rsv = 0;

                    AscendC::DataCopyExtParams storeMove;
                    storeMove.blockLen = chunkBytes;
                    storeMove.srcStride = 0;
                    storeMove.dstStride = blockSize * channelBytes - chunkBytes;
                    storeMove.rsv = 0;

                    for (uint32_t columnBase = 0U; columnBase < stripe.column_count;
                         columnBase += columnsPerChunk) {
                        uint32_t activeColumns = stripe.column_count - columnBase;
                        if (activeColumns > columnsPerChunk) {
                            activeColumns = columnsPerChunk;
                        }
                        uint32_t dstColIdx = stripe.first_result_col + columnBase * blockSize;
                        uint32_t srcColIdx = (dstColIdx + cropEdges.trim_left) / blockSize;
                        uint32_t srcLinearOffset =
                            ((stripe.source_batch * inputShape.height_extent + stripe.source_row) *
                             inputShape.width_extent + srcColIdx) *
                            inputShape.channel_count + channelOffset;
                        uint32_t dstLinearOffset =
                            ((stripe.result_batch * outputShape.height_extent + stripe.result_row) * outputShape.width_extent +
                             dstColIdx) * outputShape.channel_count + channelOffset;

                        loadMove.blockCount = static_cast<uint16_t>(activeColumns);
                        storeMove.blockCount = static_cast<uint16_t>(activeColumns);
                        AscendC::DataCopyPad(tempTensor, gmInput[srcLinearOffset], loadMove, noPad);
                        B2SBarrierAll();
                        AscendC::DataCopyPad(gmOutput[dstLinearOffset], tempTensor, storeMove);
                        B2SBarrierAll();
                    }
                }
            }
            return;
        }

        AscendC::DataCopyExtParams loadMove;
        loadMove.blockLen = channelBytes;
        loadMove.srcStride = 0;
        loadMove.dstStride = 0;
        loadMove.rsv = 0;

        AscendC::DataCopyExtParams storeMove;
        storeMove.blockLen = channelBytes;
        storeMove.srcStride = 0;
        storeMove.dstStride = blockSize * channelBytes - channelBytes;
        storeMove.rsv = 0;

        for (uint32_t stripeId = workerId; stripeId < stripeCount; stripeId += workerCount) {
            const FallbackStripeTile stripe = B2SResolveStripe(stripeId);
            if (!stripe.valid) {
                continue;
            }

            for (uint32_t columnBase = 0U; columnBase < stripe.column_count;
                 columnBase += maxColumnsPerCopy) {
                uint32_t activeColumns = stripe.column_count - columnBase;
                if (activeColumns > maxColumnsPerCopy) {
                    activeColumns = maxColumnsPerCopy;
                }
                uint32_t dstColIdx = stripe.first_result_col + columnBase * blockSize;
                uint32_t srcColIdx = (dstColIdx + cropEdges.trim_left) / blockSize;
                uint32_t srcLinearOffset =
                    ((stripe.source_batch * inputShape.height_extent + stripe.source_row) *
                     inputShape.width_extent + srcColIdx) *
                    inputShape.channel_count;
                uint32_t dstLinearOffset =
                    ((stripe.result_batch * outputShape.height_extent + stripe.result_row) * outputShape.width_extent + dstColIdx) *
                    outputShape.channel_count;

                loadMove.blockCount = static_cast<uint16_t>(activeColumns);
                storeMove.blockCount = static_cast<uint16_t>(activeColumns);
                AscendC::DataCopyPad(tempTensor, gmInput[srcLinearOffset], loadMove, noPad);
                B2SBarrierAll();
                AscendC::DataCopyPad(gmOutput[dstLinearOffset], tempTensor, storeMove);
                B2SBarrierAll();
            }
        }
    }
private:
    static constexpr uint32_t kBlockBytes = 32U;
    static constexpr uint32_t kDefaultBufferBytes = 64U * 1024U;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> tempBuffer;
    AscendC::LocalTensor<DT_X> tempTensor;
    AscendC::GlobalTensor<DT_X> gmInput;
    AscendC::GlobalTensor<DT_X> gmOutput;
    B2SLayout4D inputShape;
    B2SLayout4D outputShape;
    B2SCropBox cropEdges;
    uint32_t blockSize;
    uint32_t runtimePath;
    uint32_t tempElementLimit;
    uint32_t tempByteLimit;
    uint32_t stripeCount;
    uint32_t channelBytes;
    uint32_t alignedChannelBytes;
};

// Generic channel imported from BatchToSpace_v4 and adapted to B2SDispatchPlan.
// The profiled schedules above still own the ten specialized lanes; only the
// generic schedule enters this runtime shape-driven implementation.
template <class DT_X>
class B2SV4GenericChannel {
public:
    __aicore__ inline B2SV4GenericChannel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const B2SDispatchPlan &plan) {
        inputBatch = plan.input.batch_count;
        inputHeight = plan.input.height_extent;
        inputWidth = plan.input.width_extent;
        depth = plan.input.channel_count;
        outputBatch = plan.output.batch_count;
        outputHeight = plan.output.height_extent;
        outputWidth = plan.output.width_extent;
        cropTop = plan.crop.trim_top;
        cropLeft = plan.crop.trim_left;
        blockSize = plan.block_extent;
        totalOutputPixels = outputBatch * outputHeight * outputWidth;
        inputElements = inputBatch * inputHeight * inputWidth * depth;
        outputElements = plan.element_total;
        tileElements = plan.buffer_element_quota == 0U ? kDefaultTileBytes / sizeof(DT_X) :
            plan.buffer_element_quota;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x, inputElements);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, outputElements);
        pipe.InitBuffer(dataQueue, kBufferNum, tileElements * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (blockSize == 1U && outputHeight == inputHeight && outputWidth == inputWidth) {
            ProcessContiguous();
            return;
        }
        if (outputWidth <= blockSize) {
            if (ShouldSplitPixels()) {
                ProcessPixelChannelTiles();
                return;
            }
            ProcessPixels();
            return;
        }
        if (ShouldUseVerticalRows()) {
            ProcessVerticalRows();
            return;
        }
        if (ShouldPackAlignedRows()) {
            ProcessPackedRows();
            return;
        }
        if (ShouldUsePixelsForWideRows()) {
            ProcessPixels();
            return;
        }

        ProcessRows();
    }

private:
    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) const {
        return a < b ? a : b;
    }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const {
        return ((value + align - 1U) / align) * align;
    }

    __aicore__ inline uint32_t AlignElements(uint32_t elements) const {
        const uint32_t bytes = elements * sizeof(DT_X);
        return AlignUp(bytes, kBlockBytes) / sizeof(DT_X);
    }

    __aicore__ inline uint32_t ChannelTaskCount() const {
        return (depth + kChannelTaskElements - 1U) / kChannelTaskElements;
    }

    __aicore__ inline bool ShouldSplitPixels() const {
        return totalOutputPixels <= kPixelSplitLimit &&
            depth >= kChannelTaskElements * 2U;
    }

    __aicore__ inline bool ShouldUsePixelsForWideRows() const {
        return outputBatch * outputHeight * blockSize <= kWidePixelRowGroupLimit &&
            outputWidth >= kWidePixelWidthLimit &&
            depth <= kWidePixelDepthLimit;
    }

    __aicore__ inline uint32_t PackedRowSegmentColumns() const {
        uint32_t segmentElements = kPackedRowTileElements;
        if (segmentElements > tileElements) {
            segmentElements = tileElements;
        }
        uint32_t segmentColumns = segmentElements / depth;
        if (segmentColumns == 0U) {
            segmentColumns = 1U;
        }
        return segmentColumns;
    }

    __aicore__ inline bool ShouldPackAlignedRows() const {
        return outputWidth >= kPackedRowMinWidth &&
            (depth * sizeof(DT_X)) % kBlockBytes == 0U &&
            PackedRowSegmentColumns() > blockSize;
    }

    __aicore__ inline bool ShouldUseVerticalRows() const {
        return outputWidth <= kVerticalWidthLimit &&
            outputHeight >= kVerticalMinHeight;
    }

    __aicore__ inline void ProcessRows() {
        const uint32_t blockIdx = B2SCoreIndex();
        const uint32_t blockNum = B2SCoreCount();
        const uint32_t totalRows = outputBatch * outputHeight;
        const uint32_t totalGroups = totalRows * blockSize;
        const uint32_t startGroup = static_cast<uint32_t>(
            (static_cast<uint64_t>(totalGroups) * blockIdx) / blockNum);
        const uint32_t endGroup = static_cast<uint32_t>(
            (static_cast<uint64_t>(totalGroups) * (blockIdx + 1U)) / blockNum);

        for (uint32_t group = startGroup; group < endGroup; ++group) {
            const uint32_t blockW = group % blockSize;
            const uint32_t row = group / blockSize;
            const uint32_t outH = row % outputHeight;
            const uint32_t outB = row / outputHeight;
            CopyRowBlock(outB, outH, blockW);
        }
    }

    __aicore__ inline void ProcessPixels() {
        const uint32_t blockIdx = B2SCoreIndex();
        const uint32_t blockNum = B2SCoreCount();
        const uint32_t startPixel = static_cast<uint32_t>(
            (static_cast<uint64_t>(totalOutputPixels) * blockIdx) / blockNum);
        const uint32_t endPixel = static_cast<uint32_t>(
            (static_cast<uint64_t>(totalOutputPixels) * (blockIdx + 1U)) / blockNum);

        for (uint32_t pixel = startPixel; pixel < endPixel; ++pixel) {
            CopyOutputPixel(pixel);
        }
    }

    __aicore__ inline void ProcessPackedRows() {
        const uint32_t blockIdx = B2SCoreIndex();
        const uint32_t blockNum = B2SCoreCount();
        const uint32_t segmentColumns = PackedRowSegmentColumns();
        const uint32_t segmentCount = (outputWidth + segmentColumns - 1U) / segmentColumns;
        const uint32_t totalRows = outputBatch * outputHeight;
        const uint32_t totalTasks = totalRows * segmentCount;
        const uint32_t startTask = static_cast<uint32_t>(
            (static_cast<uint64_t>(totalTasks) * blockIdx) / blockNum);
        const uint32_t endTask = static_cast<uint32_t>(
            (static_cast<uint64_t>(totalTasks) * (blockIdx + 1U)) / blockNum);

        for (uint32_t task = startTask; task < endTask; ++task) {
            const uint32_t segment = task % segmentCount;
            const uint32_t row = task / segmentCount;
            const uint32_t outH = row % outputHeight;
            const uint32_t outB = row / outputHeight;
            const uint32_t startW = segment * segmentColumns;
            uint32_t columns = outputWidth - startW;
            if (columns > segmentColumns) {
                columns = segmentColumns;
            }
            CopyPackedRowSegment(outB, outH, startW, columns);
        }
    }

    __aicore__ inline void ProcessVerticalRows() {
        const uint32_t blockIdx = B2SCoreIndex();
        const uint32_t blockNum = B2SCoreCount();
        const uint32_t maxRowsPerBlock = (outputHeight + blockSize - 1U) / blockSize;
        const uint32_t rowTaskCount =
            (maxRowsPerBlock + kVerticalTaskRows - 1U) / kVerticalTaskRows;
        const uint32_t totalTasks = outputBatch * outputWidth * blockSize * rowTaskCount;
        const uint32_t startTask = static_cast<uint32_t>(
            (static_cast<uint64_t>(totalTasks) * blockIdx) / blockNum);
        const uint32_t endTask = static_cast<uint32_t>(
            (static_cast<uint64_t>(totalTasks) * (blockIdx + 1U)) / blockNum);

        for (uint32_t task = startTask; task < endTask; ++task) {
            const uint32_t rowTask = task % rowTaskCount;
            uint32_t group = task / rowTaskCount;
            const uint32_t blockH = group % blockSize;
            group /= blockSize;
            const uint32_t outW = group % outputWidth;
            const uint32_t outB = group / outputWidth;
            CopyColumnBlock(outB, outW, blockH, rowTask * kVerticalTaskRows);
        }
    }

    __aicore__ inline void ProcessPixelChannelTiles() {
        const uint32_t blockIdx = B2SCoreIndex();
        const uint32_t blockNum = B2SCoreCount();
        const uint32_t channelTaskCount = ChannelTaskCount();
        const uint32_t totalTasks = totalOutputPixels * channelTaskCount;
        const uint32_t startTask = static_cast<uint32_t>(
            (static_cast<uint64_t>(totalTasks) * blockIdx) / blockNum);
        const uint32_t endTask = static_cast<uint32_t>(
            (static_cast<uint64_t>(totalTasks) * (blockIdx + 1U)) / blockNum);

        for (uint32_t task = startTask; task < endTask; ++task) {
            const uint32_t channelTask = task % channelTaskCount;
            const uint32_t outputPixel = task / channelTaskCount;
            const uint32_t channelStart = channelTask * kChannelTaskElements;
            const uint32_t channelCount =
                Min(depth - channelStart, kChannelTaskElements);
            CopyOutputPixelChannels(outputPixel, channelStart, channelCount);
        }
    }

    __aicore__ inline uint32_t FirstOutputInSegment(uint32_t baseOutW,
                                                    uint32_t startW) const {
        if (baseOutW >= startW) {
            return baseOutW;
        }
        const uint32_t delta = startW - baseOutW;
        const uint32_t steps = (delta + blockSize - 1U) / blockSize;
        return baseOutW + steps * blockSize;
    }

    __aicore__ inline void CopyPackedRowSegment(uint32_t outB, uint32_t outH,
                                                uint32_t startW, uint32_t columns) {
        AscendC::LocalTensor<DT_X> local = dataQueue.AllocTensor<DT_X>();
        const uint32_t cropLeftMod = cropLeft % blockSize;
        const uint32_t endW = startW + columns;
        const uint32_t depthBytes = depth * sizeof(DT_X);
        const uint32_t dstStride = (blockSize - 1U) * depthBytes;
        const uint32_t paddedH = outH + cropTop;
        const uint32_t inH = paddedH / blockSize;
        const uint32_t blockH = paddedH - inH * blockSize;

        for (uint32_t blockW = 0U; blockW < blockSize; ++blockW) {
            const uint32_t baseOutW = (blockW + blockSize - cropLeftMod) % blockSize;
            const uint32_t firstOutW = FirstOutputInSegment(baseOutW, startW);
            if (firstOutW >= endW) {
                continue;
            }

            const uint32_t paddedW = firstOutW + cropLeft;
            const uint32_t firstInW = paddedW / blockSize;
            const uint32_t inB = (blockH * blockSize + blockW) * outputBatch + outB;
            const uint32_t copyColumns = (endW - 1U - firstOutW) / blockSize + 1U;
            const uint32_t inputOffset =
                ((inB * inputHeight + inH) * inputWidth + firstInW) * depth;
            const uint32_t localOffset = (firstOutW - startW) * depth;

            AscendC::DataCopyParams copyInParams = {
                static_cast<uint16_t>(copyColumns),
                static_cast<uint16_t>(depthBytes / kBlockBytes),
                0,
                static_cast<uint16_t>(dstStride / kBlockBytes)};
            AscendC::DataCopy(local[localOffset], xGm[inputOffset], copyInParams);
        }

        dataQueue.EnQue(local);
        AscendC::LocalTensor<DT_X> outLocal = dataQueue.DeQue<DT_X>();
        const uint32_t outputOffset =
            ((outB * outputHeight + outH) * outputWidth + startW) * depth;
        const uint32_t copyElements = columns * depth;
        const uint32_t copyBytes = copyElements * sizeof(DT_X);
        const bool alignedOutput = ((outputOffset * sizeof(DT_X)) % kBlockBytes == 0U) &&
            (copyBytes % kBlockBytes == 0U);
        if (alignedOutput) {
            AscendC::DataCopy(yGm[outputOffset], outLocal, copyElements);
        } else {
            AscendC::DataCopyExtParams copyParams = {1, copyBytes, 0, 0, 0};
            AscendC::DataCopyPad(yGm[outputOffset], outLocal, copyParams);
        }
        dataQueue.FreeTensor(outLocal);
    }

    __aicore__ inline void CopyRowBlock(uint32_t outB, uint32_t outH, uint32_t blockW) {
        const uint32_t cropLeftMod = cropLeft % blockSize;
        const uint32_t firstOutW = (blockW + blockSize - cropLeftMod) % blockSize;
        if (firstOutW >= outputWidth) {
            return;
        }

        const uint32_t paddedH = outH + cropTop;
        const uint32_t inH = paddedH / blockSize;
        const uint32_t blockH = paddedH - inH * blockSize;
        const uint32_t paddedW = firstOutW + cropLeft;
        const uint32_t firstInW = paddedW / blockSize;
        const uint32_t inB = (blockH * blockSize + blockW) * outputBatch + outB;
        const uint32_t columnCount = (outputWidth - 1U - firstOutW) / blockSize + 1U;

        const uint32_t inputBase = ((inB * inputHeight + inH) * inputWidth + firstInW) * depth;
        const uint32_t outputBase = ((outB * outputHeight + outH) * outputWidth + firstOutW) * depth;

        for (uint32_t channel = 0U; channel < depth;) {
            uint32_t channelChunk = depth - channel;
            if (channelChunk > tileElements) {
                channelChunk = tileElements;
            }
            const uint32_t alignedChunk = AlignElements(channelChunk);
            uint32_t maxColumns = tileElements / alignedChunk;
            if (maxColumns == 0U) {
                maxColumns = 1U;
            }
            maxColumns = Min(maxColumns, 65535U);

            for (uint32_t col = 0U; col < columnCount;) {
                const uint32_t columns = Min(columnCount - col, maxColumns);
                const uint32_t inputOffset = inputBase + col * depth + channel;
                const uint32_t outputOffset = outputBase + col * blockSize * depth + channel;
                CopyStrided(inputOffset, outputOffset, columns, channelChunk);
                col += columns;
            }
            channel += channelChunk;
        }
    }

    __aicore__ inline void CopyColumnBlock(uint32_t outB, uint32_t outW,
                                           uint32_t blockH, uint32_t rowStart) {
        const uint32_t cropTopMod = cropTop % blockSize;
        uint32_t firstOutH = (blockH + blockSize - cropTopMod) % blockSize;
        if (firstOutH >= outputHeight) {
            return;
        }

        const uint32_t rowCountTotal = (outputHeight - 1U - firstOutH) / blockSize + 1U;
        if (rowStart >= rowCountTotal) {
            return;
        }

        uint32_t rowCount = rowCountTotal - rowStart;
        if (rowCount > kVerticalTaskRows) {
            rowCount = kVerticalTaskRows;
        }
        firstOutH += rowStart * blockSize;

        const uint32_t paddedH = firstOutH + cropTop;
        const uint32_t inH = paddedH / blockSize;
        const uint32_t paddedW = outW + cropLeft;
        const uint32_t inW = paddedW / blockSize;
        const uint32_t blockW = paddedW - inW * blockSize;
        const uint32_t inB = (blockH * blockSize + blockW) * outputBatch + outB;
        const uint32_t inputBase = ((inB * inputHeight + inH) * inputWidth + inW) * depth;
        const uint32_t outputBase = ((outB * outputHeight + firstOutH) * outputWidth + outW) * depth;
        const uint32_t inputRowStride = inputWidth * depth;
        const uint32_t outputRowStride = blockSize * outputWidth * depth;

        for (uint32_t channel = 0U; channel < depth;) {
            uint32_t channelChunk = depth - channel;
            if (channelChunk > tileElements) {
                channelChunk = tileElements;
            }
            const uint32_t alignedChunk = AlignElements(channelChunk);
            uint32_t maxRows = tileElements / alignedChunk;
            if (maxRows == 0U) {
                maxRows = 1U;
            }
            maxRows = Min(maxRows, 65535U);

            for (uint32_t row = 0U; row < rowCount;) {
                const uint32_t rows = Min(rowCount - row, maxRows);
                const uint32_t inputOffset = inputBase + row * inputRowStride + channel;
                const uint32_t outputOffset = outputBase + row * outputRowStride + channel;
                CopyStridedGeneral(inputOffset, outputOffset, rows, channelChunk,
                                   inputRowStride - channelChunk,
                                   outputRowStride - channelChunk);
                row += rows;
            }
            channel += channelChunk;
        }
    }

    __aicore__ inline void ProcessContiguous() {
        const uint32_t blockIdx = B2SCoreIndex();
        const uint32_t blockNum = B2SCoreCount();
        const uint32_t alignElements = kBlockBytes / sizeof(DT_X);
        const uint32_t totalBlocks = (outputElements + alignElements - 1U) / alignElements;
        const uint32_t startBlock = static_cast<uint32_t>(
            (static_cast<uint64_t>(totalBlocks) * blockIdx) / blockNum);
        const uint32_t endBlock = static_cast<uint32_t>(
            (static_cast<uint64_t>(totalBlocks) * (blockIdx + 1U)) / blockNum);
        const uint32_t start = startBlock * alignElements;
        uint32_t end = endBlock * alignElements;
        if (end > outputElements) {
            end = outputElements;
        }

        for (uint32_t offset = start; offset < end;) {
            uint32_t copyElements = end - offset;
            if (copyElements > tileElements) {
                copyElements = tileElements;
            }
            CopyChunk(offset, offset, copyElements);
            offset += copyElements;
        }
    }

    __aicore__ inline void CopyOutputPixelChannels(uint32_t outputPixel,
                                                   uint32_t channelStart,
                                                   uint32_t channelLimit) {
        const uint32_t outW = outputPixel % outputWidth;
        const uint32_t tmp = outputPixel / outputWidth;
        const uint32_t outH = tmp % outputHeight;
        const uint32_t outB = tmp / outputHeight;

        const uint32_t paddedH = outH + cropTop;
        const uint32_t paddedW = outW + cropLeft;
        const uint32_t inH = paddedH / blockSize;
        const uint32_t inW = paddedW / blockSize;
        const uint32_t blockH = paddedH - inH * blockSize;
        const uint32_t blockW = paddedW - inW * blockSize;
        const uint32_t inB = (blockH * blockSize + blockW) * outputBatch + outB;

        const uint32_t inputBase = ((inB * inputHeight + inH) * inputWidth + inW) * depth;
        const uint32_t outputBase = outputPixel * depth;

        for (uint32_t channel = 0U; channel < channelLimit;) {
            uint32_t copyElements = channelLimit - channel;
            if (copyElements > tileElements) {
                copyElements = tileElements;
            }
            CopyChunk(inputBase + channelStart + channel,
                      outputBase + channelStart + channel, copyElements);
            channel += copyElements;
        }
    }

    __aicore__ inline void CopyOutputPixel(uint32_t outputPixel) {
        const uint32_t outW = outputPixel % outputWidth;
        const uint32_t tmp = outputPixel / outputWidth;
        const uint32_t outH = tmp % outputHeight;
        const uint32_t outB = tmp / outputHeight;

        const uint32_t paddedH = outH + cropTop;
        const uint32_t paddedW = outW + cropLeft;
        const uint32_t inH = paddedH / blockSize;
        const uint32_t inW = paddedW / blockSize;
        const uint32_t blockH = paddedH - inH * blockSize;
        const uint32_t blockW = paddedW - inW * blockSize;
        const uint32_t inB = (blockH * blockSize + blockW) * outputBatch + outB;

        const uint32_t inputBase = ((inB * inputHeight + inH) * inputWidth + inW) * depth;
        const uint32_t outputBase = outputPixel * depth;

        for (uint32_t channel = 0U; channel < depth;) {
            uint32_t copyElements = depth - channel;
            if (copyElements > tileElements) {
                copyElements = tileElements;
            }
            CopyChunk(inputBase + channel, outputBase + channel, copyElements);
            channel += copyElements;
        }
    }

    __aicore__ inline void CopyChunk(uint32_t inputOffset, uint32_t outputOffset,
                                     uint32_t copyElements) {
        AscendC::LocalTensor<DT_X> local = dataQueue.AllocTensor<DT_X>();

        const uint32_t copyBytes = copyElements * sizeof(DT_X);
        const bool alignedCopy = ((inputOffset * sizeof(DT_X)) % kBlockBytes == 0U) &&
            ((outputOffset * sizeof(DT_X)) % kBlockBytes == 0U) &&
            (copyBytes % kBlockBytes == 0U);
        if (alignedCopy) {
            AscendC::DataCopy(local, xGm[inputOffset], copyElements);
        } else {
            AscendC::DataCopyExtParams copyParams = {1, copyBytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0, 0};
            AscendC::DataCopyPad(local, xGm[inputOffset], copyParams, padParams);
        }
        dataQueue.EnQue(local);

        AscendC::LocalTensor<DT_X> outLocal = dataQueue.DeQue<DT_X>();
        if (alignedCopy) {
            AscendC::DataCopy(yGm[outputOffset], outLocal, copyElements);
        } else {
            AscendC::DataCopyExtParams copyParams = {1, copyBytes, 0, 0, 0};
            AscendC::DataCopyPad(yGm[outputOffset], outLocal, copyParams);
        }
        dataQueue.FreeTensor(outLocal);
    }

    __aicore__ inline void CopyStrided(uint32_t inputOffset, uint32_t outputOffset,
                                       uint32_t blockCount, uint32_t copyElements) {
        CopyStridedGeneral(inputOffset, outputOffset, blockCount, copyElements,
                           depth - copyElements,
                           blockSize * depth - copyElements);
    }

    __aicore__ inline void CopyStridedGeneral(uint32_t inputOffset, uint32_t outputOffset,
                                              uint32_t blockCount, uint32_t copyElements,
                                              uint32_t inputStrideElements,
                                              uint32_t outputStrideElements) {
        AscendC::LocalTensor<DT_X> local = dataQueue.AllocTensor<DT_X>();

        const uint32_t copyBytes = copyElements * sizeof(DT_X);
        const uint32_t inputStride = inputStrideElements * sizeof(DT_X);
        const uint32_t outputStride = outputStrideElements * sizeof(DT_X);
        const bool alignedCopy = ((inputOffset * sizeof(DT_X)) % kBlockBytes == 0U) &&
            ((outputOffset * sizeof(DT_X)) % kBlockBytes == 0U) &&
            (copyBytes % kBlockBytes == 0U) &&
            (inputStride % kBlockBytes == 0U) &&
            (outputStride % kBlockBytes == 0U) &&
            (blockCount <= 65535U) &&
            (copyBytes / kBlockBytes <= 65535U) &&
            (inputStride / kBlockBytes <= 65535U) &&
            (outputStride / kBlockBytes <= 65535U);
        if (alignedCopy) {
            AscendC::DataCopyParams copyInParams = {
                static_cast<uint16_t>(blockCount),
                static_cast<uint16_t>(copyBytes / kBlockBytes),
                static_cast<uint16_t>(inputStride / kBlockBytes),
                0};
            AscendC::DataCopy(local, xGm[inputOffset], copyInParams);
            dataQueue.EnQue(local);

            AscendC::LocalTensor<DT_X> outLocal = dataQueue.DeQue<DT_X>();
            AscendC::DataCopyParams copyOutParams = {
                static_cast<uint16_t>(blockCount),
                static_cast<uint16_t>(copyBytes / kBlockBytes),
                0,
                static_cast<uint16_t>(outputStride / kBlockBytes)};
            AscendC::DataCopy(yGm[outputOffset], outLocal, copyOutParams);
            dataQueue.FreeTensor(outLocal);
            return;
        }

        AscendC::DataCopyExtParams copyInParams = {
            static_cast<uint16_t>(blockCount), copyBytes, inputStride, 0, 0};
        AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0, 0};
        AscendC::DataCopyPad(local, xGm[inputOffset], copyInParams, padParams);
        dataQueue.EnQue(local);

        AscendC::LocalTensor<DT_X> outLocal = dataQueue.DeQue<DT_X>();
        AscendC::DataCopyExtParams copyOutParams = {
            static_cast<uint16_t>(blockCount), copyBytes, 0, outputStride, 0};
        AscendC::DataCopyPad(yGm[outputOffset], outLocal, copyOutParams);
        dataQueue.FreeTensor(outLocal);
    }

private:
    static constexpr int32_t kBufferNum = 1;
    static constexpr uint32_t kBlockBytes = 32U;
    static constexpr uint32_t kDefaultTileBytes = 64U * 1024U;
    static constexpr uint32_t kPixelSplitLimit = 16U;
    static constexpr uint32_t kChannelTaskElements = 2048U;
    static constexpr uint32_t kWidePixelRowGroupLimit = 64U;
    static constexpr uint32_t kWidePixelWidthLimit = 128U;
    static constexpr uint32_t kWidePixelDepthLimit = 8U;
    static constexpr uint32_t kPackedRowTileElements = 32768U;
    static constexpr uint32_t kPackedRowMinWidth = 32U;
    static constexpr uint32_t kVerticalWidthLimit = 32U;
    static constexpr uint32_t kVerticalMinHeight = 64U;
    static constexpr uint32_t kVerticalTaskRows = 128U;

    AscendC::TPipe pipe;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, kBufferNum> dataQueue;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t inputBatch;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t depth;
    uint32_t outputBatch;
    uint32_t outputHeight;
    uint32_t outputWidth;
    uint32_t cropTop;
    uint32_t cropLeft;
    uint32_t blockSize;
    uint32_t totalOutputPixels;
    uint32_t inputElements;
    uint32_t outputElements;
    uint32_t tileElements;
};

// Compile-time dispatcher: profiled schedules route to constant-folded kernels;
// all unprofiled legal shapes use the compact tiling plan above.
template <typename DT_X, uint64_t SCH_MODE>
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    if constexpr (SCH_MODE == B2S_SCHEDULE_F32_R2_ROW_28X28_C128) {
        RunR2SingleRowTile<DT_X, 28U, 28U, 128U, 2U, 16384U, true, false>(x, y);
        return;
    }
    if constexpr (SCH_MODE == B2S_SCHEDULE_F32_R2_CROP_SMALL_C5) {
        RunCompactR2CropGather<DT_X>(x, y);
        return;
    }
    if constexpr (SCH_MODE == B2S_SCHEDULE_F32_R2_ROW_14X14_C64) {
        RunR2RowPhaseInterleave<DT_X, 14U, 14U, 64U, 4U, 7U, 512U, true, false, true>(x, y);
        return;
    }
    if constexpr (SCH_MODE == B2S_SCHEDULE_F32_R2_ROW_4X6_C32) {
        RunR2RowPhaseInterleave<DT_X, 4U, 6U, 32U, 5U, 4U, 4096U, false, false, false>(x, y);
        return;
    }
    if constexpr (SCH_MODE == B2S_SCHEDULE_F16_R2_CROP_DENSE_C65) {
        RunR2DenseCropGather<DT_X>(x, y);
        return;
    }
    if constexpr (SCH_MODE == B2S_SCHEDULE_F16_R2_CHANNEL_TILE_C4096) {
        RunR2ChannelSliceNoCrop<DT_X, 2U, 2U, 4096U, 1U, 2048U>(x, y);
        return;
    }
    if constexpr (SCH_MODE == B2S_SCHEDULE_F16_R2_FLAT_C16384) {
        RunAlignedFlatCopy<DT_X>(x, y);
        return;
    }
    if constexpr (SCH_MODE == B2S_SCHEDULE_F16_R2_WIDE_ROW_C256) {
        RunR2WideInputScatter<DT_X, 10U, 512U, 256U, 128U>(x, y);
        return;
    }
    if constexpr (SCH_MODE == B2S_SCHEDULE_F16_R4_LEFT_SHIFT_C64) {
        RunR4LeftCropInterleave<DT_X, 10U, 512U, 64U, 40U, 1535U, 513U, 512U>(x, y);
        return;
    }
    if constexpr (SCH_MODE == B2S_SCHEDULE_F16_R2_TALL_ROW_C32) {
#if B2S_ENABLE_ROW_SLAB_POLICY
        RunR2TallRowSlab<DT_X, 1024U, 6U, 32U, 4U, 120U, 512U>(x, y);
#else
        RunR2RowPhaseInterleave<DT_X, 1024U, 6U, 32U, 4U, 52U, 512U, true, true, false>(x, y);
#endif
        return;
    }

    REGISTER_TILING_DEFAULT(B2SDispatchPlan);
    GET_TILING_DATA_WITH_STRUCT(B2SDispatchPlan, tiling_data, tiling);
    B2SV4GenericChannel<DT_X> op;
    op.Init(x, y, tiling_data);
    op.Process();
}
