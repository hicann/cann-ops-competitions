// Template-side kernel implementation for profiled BatchToSpace layouts.
# include "kernel_operator.h"

# include "batch_to_space_tiling.h"
# include "tiling_key_batch_to_space.h"

#define BTS_ENABLE_ROW_SLAB_TRIAL (1)

__aicore__ inline void SyncAllPipes() { (void)AscendC::PipeBarrier<PIPE_ALL>(); }

__aicore__ inline void SyncMte2Pipe() { (void)AscendC::PipeBarrier<PIPE_MTE2>(); }

__aicore__ inline void SyncMte3Pipe() { (void)AscendC::PipeBarrier<PIPE_MTE3>(); }

__aicore__ inline void SyncVectorPipe() { (void)AscendC::PipeBarrier<PIPE_V>(); }

template<bool kUseFullFence>
__aicore__ inline void SyncAfterLoad() {
    if constexpr (kUseFullFence) {
        SyncAllPipes();
    } else {
        SyncMte2Pipe();
    }
}

template<bool kUseVectorFence, bool kUseFullFence>
__aicore__ inline void SyncAfterUbMove() {
    if constexpr (kUseVectorFence) {
        SyncVectorPipe();
    } else if constexpr (kUseFullFence) {
        SyncAllPipes();
    } else {
        SyncMte2Pipe();
    }
}

__aicore__ inline void MarkScalarVector(event_t token) {
    AscendC::SetFlag<AscendC::HardEvent::S_V>(token);
}

__aicore__ inline void WaitScalarVector(event_t token) {
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(token);
}

__aicore__ inline void MarkLoadStore(event_t token) {
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(token);
}

__aicore__ inline void WaitLoadStore(event_t token) {
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(token);
}

__aicore__ inline void MarkStoreLoad(event_t token) {
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(token);
}

__aicore__ inline void WaitStoreLoad(event_t token) {
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(token);
}

__aicore__ inline void MarkLoadVector(event_t token) {
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(token);
}

__aicore__ inline void WaitLoadVector(event_t token) {
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(token);
}

__aicore__ inline void MarkVectorStore(event_t token) {
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(token);
}

__aicore__ inline void WaitVectorStore(event_t token) {
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(token);
}

template<typename DT_X>
__aicore__ inline void AttachGmPair(GM_ADDR x, GM_ADDR y,
                                          AscendC::GlobalTensor<DT_X> &gmRead,
                                          AscendC::GlobalTensor<DT_X> &gmWrite) {
    gmRead.SetGlobalBuffer((__gm__ DT_X*)x);
    gmWrite.SetGlobalBuffer((__gm__ DT_X*)y);
}

template<typename DT_X>
__aicore__ inline void ResetPadSpec(AscendC::DataCopyPadExtParams<DT_X> &pad) {
    pad.isPad = (false);
    pad.leftPadding = 0U;
    pad.rightPadding = 0U;
    pad.paddingValue = static_cast<DT_X>(0U);
}

__aicore__ inline void ClearCopyPlan(AscendC::DataCopyParams &params) {
    params.blockCount = static_cast<uint16_t>(1U);
    params.blockLen = static_cast<uint16_t>(0U);
    params.srcStride = static_cast<uint32_t>(0U);
    params.dstStride = static_cast<uint32_t>(0U);
}

__aicore__ inline void ClearExtCopyPlan(AscendC::DataCopyExtParams &params) {
    params.blockCount = static_cast<uint16_t>(1U);
    params.blockLen = static_cast<uint16_t>(0U);
    params.srcStride = static_cast<uint32_t>(0U);
    params.dstStride = static_cast<uint32_t>(0U);
    params.rsv = static_cast<uint32_t>(0U);
}

template<uint32_t kCopyBlockTotal>
__aicore__ inline bool ClaimBlockRun(uint32_t &blockStart, uint32_t &activeBlocks) {
    const uint32_t aivId = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t aivNum = static_cast<uint32_t>(AscendC::GetBlockNum());
    const uint32_t quota = (kCopyBlockTotal + aivNum - 1U) / aivNum;
    blockStart = aivId * quota;
    if (blockStart >= kCopyBlockTotal) {
        activeBlocks = (0U);
        return (false);
    }
    activeBlocks = kCopyBlockTotal - blockStart;
    if (quota < activeBlocks) {
        activeBlocks = quota + 0U;
    }
    return (true);
}

template<uint32_t kTinyInRows, uint32_t kRowsPerTile, uint32_t kTileTotal>
__aicore__ inline void SplitPhaseRowTask(uint32_t job,
                                         uint32_t &heightPhase,
                                         uint32_t &outBatch,
                                         uint32_t &sourceH,
                                         uint32_t &rowSpan) {
    const uint32_t r4OutTile = job % kTileTotal;
    const uint32_t phaseBatch = job / kTileTotal;
    heightPhase = phaseBatch & 1U;
    outBatch = phaseBatch / 2U;
    sourceH = r4OutTile * kRowsPerTile;
    rowSpan = kTinyInRows - sourceH;
    if (kRowsPerTile < rowSpan) {
        rowSpan = (kRowsPerTile);
    }
}

template<typename DT_X, uint32_t kTinyDepth, uint32_t kBankElems,
          uint32_t kSeedPairs, uint32_t kSeedPatternElems,
          uint32_t kSeedCopies, uint32_t kSeedTailElems>
__aicore__ inline void BuildInterleaveGatherSeed(AscendC::LocalTensor<int32_t> seedTensor) {
    constexpr event_t kSeedReadyFlag = static_cast<event_t>(6);
    constexpr uint32_t kPairElems = 2U * kTinyDepth;
    for (uint32_t seedPairIx = 0; seedPairIx < kSeedPairs; ++seedPairIx) {
        uint32_t seedDst = seedPairIx * kPairElems;
        uint32_t pairByteOffset = seedPairIx * kTinyDepth * sizeof(DT_X);
        for (uint32_t depthIx = 0; depthIx < kTinyDepth; ++depthIx) {
            uint32_t depthByteOffset = depthIx * sizeof(DT_X);
            seedTensor.SetValue(seedDst + depthIx,
                                 static_cast<int32_t>(pairByteOffset + depthByteOffset));
            seedTensor.SetValue(
                seedDst + kTinyDepth + depthIx,
                static_cast<int32_t>(kBankElems * sizeof(DT_X) + pairByteOffset + depthByteOffset));
        }
    }
    MarkScalarVector(kSeedReadyFlag);
    WaitScalarVector(kSeedReadyFlag);
    for (uint32_t seedCopyIx = 1U; seedCopyIx <= kSeedCopies; ++seedCopyIx) {
        uint32_t addCount = seedCopyIx == kSeedCopies ? kSeedTailElems : kSeedPatternElems;
        AscendC::Adds(seedTensor[seedCopyIx * kSeedPatternElems],
                      seedTensor[(seedCopyIx - 1U) * kSeedPatternElems],
                      static_cast<int32_t>(kSeedPairs * kTinyDepth * sizeof(DT_X)),
                      addCount);
    }
    SyncVectorPipe();
}

template<typename DT_X>
__aicore__ inline void RunTinyCropGather(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kTinyInRows = 10U;
    constexpr uint32_t kTinyInCols = 15U;
    constexpr uint32_t kTinyDepth = 5U;
    constexpr uint32_t kTinyOutRows = 17U;
    constexpr uint32_t kTinyOutCols = 26U;
    constexpr uint32_t kSpatial = 2U;
    constexpr uint32_t kTinyTopDrop = 2U;
    constexpr uint32_t kTinyLeftDrop = 3U;
    constexpr uint32_t kTinyLoadCols = 13U;
    constexpr uint32_t kTinyDepthBytes = kTinyDepth * sizeof(DT_X);
    constexpr uint32_t kTinyPackedBytes = ((kTinyLoadCols * kTinyDepthBytes + 31U) / 32U) * 32U;
    constexpr uint32_t kTinyPackedElems = kTinyPackedBytes / sizeof(DT_X);
    constexpr uint32_t kTinyBankElems = kTinyPackedElems * 2U;
    constexpr uint32_t kTinyRowElems = kTinyOutCols * kTinyDepth;
    constexpr uint32_t kSeedPairs = 4U;
    constexpr uint32_t kSeedPatternElems = kSeedPairs * 2U * kTinyDepth;
    constexpr uint32_t kSeedCopies = kTinyRowElems / kSeedPatternElems;
    constexpr uint32_t kIndexTailElems =
        ((kTinyRowElems - kSeedCopies * kSeedPatternElems + 7U) / 8U) * 8U;
    constexpr uint32_t kGatherMapElems = kSeedCopies * kSeedPatternElems + kIndexTailElems;

    AscendC::GlobalTensor<DT_X> gmRead;
    AscendC::GlobalTensor<DT_X> gmWrite;
    AttachGmPair(x, y, gmRead, gmWrite);

    AscendC::LocalTensor<DT_X> packedBanks(AscendC::TPosition::VECCALC, 0, kTinyBankElems);
    AscendC::LocalTensor<DT_X> gatheredRow(AscendC::TPosition::VECCALC, 1024, kTinyRowElems);
    AscendC::LocalTensor<int32_t> indexSeed(AscendC::TPosition::VECCALC, 2048, kGatherMapElems);
    AscendC::LocalTensor<uint32_t> indexMap(AscendC::TPosition::VECCALC, 2048, kTinyRowElems);

    AscendC::DataCopyPadExtParams<DT_X> padSpec;
    ResetPadSpec(padSpec);

    AscendC::DataCopyExtParams tinyLoadA;
    AscendC::DataCopyExtParams tinyLoadB;
    AscendC::DataCopyExtParams tinyStore;
    ClearExtCopyPlan(tinyLoadA);
    ClearExtCopyPlan(tinyLoadB);
    ClearExtCopyPlan(tinyStore);
    tinyLoadA.blockLen = kTinyLoadCols * kTinyDepthBytes;
    tinyLoadB.blockLen = kTinyLoadCols * kTinyDepthBytes;
    tinyStore.blockLen = kTinyOutCols * kTinyDepthBytes;

    BuildInterleaveGatherSeed<DT_X, kTinyDepth, kTinyPackedElems, kSeedPairs,
                             kSeedPatternElems, kSeedCopies, kIndexTailElems>(indexSeed);

    const uint32_t laneStart = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t laneStride = static_cast<uint32_t>(AscendC::GetBlockNum());
    for (uint32_t dstH = laneStart; dstH < kTinyOutRows; dstH += laneStride) {
        uint32_t trimmedOutRow = dstH + kTinyTopDrop;
        uint32_t sourceRow = trimmedOutRow >> 1;
        uint32_t tinyHPhase = trimmedOutRow & 1U;
        uint32_t bank0Batch = tinyHPhase * kSpatial + (kTinyLeftDrop & 1U);
        uint32_t bank1Batch = tinyHPhase * kSpatial + ((kTinyLeftDrop + 1U) & 1U);
        uint32_t bank0Column = kTinyLeftDrop >> 1U;
        uint32_t bank1Column = (kTinyLeftDrop + 1U) >> 1U;
        uint32_t srcBank0 = ((bank0Batch * kTinyInRows + sourceRow) * kTinyInCols + bank0Column) * kTinyDepth;
        uint32_t srcBank1 = ((bank1Batch * kTinyInRows + sourceRow) * kTinyInCols + bank1Column) * kTinyDepth;

        AscendC::DataCopyPad(packedBanks, gmRead[srcBank0], tinyLoadA, padSpec);
        AscendC::DataCopyPad(packedBanks[kTinyPackedElems], gmRead[srcBank1], tinyLoadB, padSpec);
        SyncAllPipes();

        AscendC::Gather(gatheredRow, packedBanks, indexMap, 0, kTinyRowElems);
        SyncAllPipes();

        AscendC::DataCopyPad(gmWrite[dstH * kTinyRowElems], gatheredRow, tinyStore);
        SyncMte3Pipe();
    }
}

template<typename DT_X>
__aicore__ inline void RunFlatAlignedRelay(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kCopyBlockTotal = (4U * 1U * 1U * 16384U * sizeof(DT_X)) / 32U;
    constexpr uint32_t kMaxBlocks = 1024U * 2U;
    constexpr uint32_t kElemsPerBlock = (32U / sizeof(DT_X));
    constexpr uint32_t kMaxElems = kElemsPerBlock * kMaxBlocks;

    AscendC::GlobalTensor<DT_X> gmInFlat;
    AscendC::GlobalTensor<DT_X> gmOutFlat;
    AttachGmPair(x, y, gmInFlat, gmOutFlat);
    AscendC::LocalTensor<DT_X> copyTile(AscendC::TPosition::VECCALC, 0, kMaxElems);

    uint32_t blockStart = 0U;
    uint32_t blockRemain = 0U;
    if (!ClaimBlockRun<kCopyBlockTotal>(blockStart, blockRemain)) {
        return void();
    }

    uint32_t denseSeedBase = blockStart * kElemsPerBlock;
    AscendC::DataCopyParams blockCopy;
    ClearCopyPlan(blockCopy);

    while (blockRemain != 0U) {
        const uint32_t burstCount = blockRemain < kMaxBlocks ? blockRemain : kMaxBlocks;
        blockCopy.blockLen = static_cast<uint16_t>(burstCount);
        AscendC::DataCopy(copyTile, gmInFlat[denseSeedBase], blockCopy);
        SyncMte2Pipe();
        AscendC::DataCopy(gmOutFlat[denseSeedBase], copyTile, blockCopy);
        SyncMte3Pipe();
        denseSeedBase += burstCount * kElemsPerBlock;
        blockRemain -= burstCount;
    }
}

template<typename DT_X, uint32_t kInRows, uint32_t kInCols, uint32_t kDepth, uint32_t kOutBatches, uint32_t kDepthTile>
__aicore__ inline void RunChannelMajorR2Tile(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kOutRowsAll = kInRows * 2U;
    constexpr uint32_t kOutColsAll = kInCols * 2U;
    constexpr uint32_t kR2 = 2U;
    constexpr uint32_t kDepthSlices = kDepth / kDepthTile;
    constexpr uint32_t kChunkBlocks = (kDepthTile * sizeof(DT_X)) / 32U;
    constexpr uint32_t kStoreGapBlocks = ((kDepth - kDepthTile) * sizeof(DT_X)) / 32U;
    constexpr uint32_t kTileElems = kOutColsAll * kDepthTile;

    AscendC::GlobalTensor<DT_X> gmIn;
    AscendC::GlobalTensor<DT_X> gmOut;
    gmIn.SetGlobalBuffer((__gm__ DT_X*)x);
    gmOut.SetGlobalBuffer((__gm__ DT_X*)y);
    AscendC::LocalTensor<DT_X> lineTile(AscendC::TPosition::VECCALC, 0, kTileElems);

    AscendC::DataCopyParams chunkPlan;
    chunkPlan.blockCount = static_cast<uint16_t>(1U);
    chunkPlan.blockLen = static_cast<uint16_t>(kChunkBlocks);
    chunkPlan.srcStride = static_cast<uint32_t>(0U);
    chunkPlan.dstStride = static_cast<uint32_t>(0U);

    AscendC::DataCopyParams storePlan;
    storePlan.blockCount = kOutColsAll;
    storePlan.blockLen = static_cast<uint16_t>(kChunkBlocks);
    storePlan.srcStride = static_cast<uint32_t>(0U);
    storePlan.dstStride = static_cast<uint16_t>(kStoreGapBlocks);

    const uint32_t aivId = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t aivNum = static_cast<uint32_t>(AscendC::GetBlockNum());
    for (uint32_t jobIx = aivId; jobIx < kOutBatches * kOutRowsAll * kDepthSlices; jobIx += aivNum) {
        uint32_t channelTile = jobIx % kDepthSlices;
        uint32_t tmp = jobIx / kDepthSlices;
        uint32_t oh = tmp % kOutRowsAll;
        uint32_t n = tmp / kOutRowsAll;
        uint32_t channelBase = channelTile * kDepthTile;
        uint32_t ih = oh / 2U;
        uint32_t hPhaseBit = oh & 1U;
        uint32_t readBatch0 = (hPhaseBit * kR2) * kOutBatches + n;
        uint32_t readBatch1 = readBatch0 + kOutBatches;

# pragma unroll
        for (uint32_t iw = 0; iw < kInCols; ++iw) {
            uint32_t readBase0 = ((readBatch0 * kInRows + ih) * kInCols + iw) * kDepth + channelBase;
            uint32_t readBase1 = ((readBatch1 * kInRows + ih) * kInCols + iw) * kDepth + channelBase;
            AscendC::DataCopy(lineTile[(2U * iw) * kDepthTile], gmIn[readBase0], chunkPlan);
            AscendC::DataCopy(lineTile[(2U * iw + 1U) * kDepthTile], gmIn[readBase1], chunkPlan);
        }
        SyncMte2Pipe();

        uint32_t writeBase = ((n * kOutRowsAll + oh) * kOutColsAll) * kDepth + channelBase;
        AscendC::DataCopy(gmOut[writeBase], lineTile, storePlan);
        SyncMte3Pipe();
    }
}

template<typename DT_X, uint32_t kInRows, uint32_t kInCols, uint32_t kDepth,
         uint32_t kOutBatches, uint32_t kRowsPerSlice, uint32_t kBufferAlignBytes,
         bool kLoadBarrierAll, bool kUbBarrierAll, bool kUbBarrierVector>
__aicore__ inline void RunRowPhaseR2Tile(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kOutRowsAll = kInRows * 2U;
    constexpr uint32_t kOutColsAll = kInCols * 2U;
    constexpr uint32_t kR2 = 2U;
    constexpr uint32_t kRowSliceTotal = (kInRows + kRowsPerSlice - 1U) / kRowsPerSlice;
    constexpr uint32_t kDepthBlocks = (kDepth * sizeof(DT_X)) / 32U;
    constexpr uint32_t kReadRowBlocks = (kInCols * kDepth * sizeof(DT_X)) / 32U;
    constexpr uint32_t kWriteRowBlocks = (kOutColsAll * kDepth * sizeof(DT_X)) / 32U;
    constexpr uint32_t kReadWindowElems = kRowsPerSlice * kInCols * kDepth;
    constexpr uint32_t kWriteWindowElems = kRowsPerSlice * kOutColsAll * kDepth;
    constexpr uint32_t kReadBytes =
        ((kReadWindowElems * sizeof(DT_X) + kBufferAlignBytes - 1U) / kBufferAlignBytes) * kBufferAlignBytes;

    AscendC::GlobalTensor<DT_X> gmInFlat;
    AscendC::GlobalTensor<DT_X> gmOutFlat;
    AttachGmPair(x, y, gmInFlat, gmOutFlat);
    AscendC::LocalTensor<DT_X> phaseRowsA(AscendC::TPosition::VECCALC, 0, kReadWindowElems);
    AscendC::LocalTensor<DT_X> phaseRowsB(AscendC::TPosition::VECCALC, kReadBytes, kReadWindowElems);
    AscendC::LocalTensor<DT_X> mergedRows(AscendC::TPosition::VECCALC, 2U * kReadBytes, kWriteWindowElems);

    AscendC::DataCopyParams rowLoadPlan;
    AscendC::DataCopyParams stripePlan;
    AscendC::DataCopyParams tinyStore;
    ClearCopyPlan(rowLoadPlan);
    ClearCopyPlan(stripePlan);
    ClearCopyPlan(tinyStore);
    stripePlan.blockLen = static_cast<uint16_t>(kDepthBlocks);
    stripePlan.dstStride = static_cast<uint16_t>(kDepthBlocks);
    tinyStore.blockLen = static_cast<uint16_t>(kWriteRowBlocks);
    tinyStore.dstStride = static_cast<uint16_t>(kWriteRowBlocks);

    const uint32_t aivId = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t aivNum = static_cast<uint32_t>(AscendC::GetBlockNum());
    constexpr uint32_t kPhaseJobs = kOutBatches * kR2 * kRowSliceTotal;
    for (uint32_t job = aivId; job < kPhaseJobs; job += aivNum) {
        uint32_t heightPhase = 0U;
        uint32_t batchOut = 0U;
        uint32_t rowStart = 0U;
        uint32_t rowCount = 0U;
        SplitPhaseRowTask<kInRows, kRowsPerSlice, kRowSliceTotal>(job, heightPhase, batchOut, rowStart, rowCount);

        const uint32_t phaseBatch0 = heightPhase * kR2 * kOutBatches + batchOut;
        const uint32_t phaseBatch1 = phaseBatch0 + kOutBatches;
        const uint32_t readPhase0 = ((phaseBatch0 * kInRows + rowStart) * kInCols) * kDepth;
        const uint32_t readPhase1 = ((phaseBatch1 * kInRows + rowStart) * kInCols) * kDepth;

        rowLoadPlan.blockLen = static_cast<uint16_t>(rowCount * kReadRowBlocks);
        AscendC::DataCopy(phaseRowsA, gmInFlat[readPhase0], rowLoadPlan);
        AscendC::DataCopy(phaseRowsB, gmInFlat[readPhase1], rowLoadPlan);
        SyncAfterLoad<kLoadBarrierAll>();

        stripePlan.blockCount = static_cast<uint16_t>(rowCount * kInCols);
        AscendC::DataCopy(mergedRows, phaseRowsA, stripePlan);
        AscendC::DataCopy(mergedRows[kDepth], phaseRowsB, stripePlan);
        SyncAfterUbMove<kUbBarrierVector, kUbBarrierAll>();

        tinyStore.blockCount = static_cast<uint16_t>(rowCount);
        const uint32_t outRowBase = heightPhase + rowStart * kR2;
        const uint32_t writeRowBase = (batchOut * kOutRowsAll + outRowBase) * kOutColsAll * kDepth;
        AscendC::DataCopy(gmOutFlat[writeRowBase], mergedRows, tinyStore);
        SyncMte3Pipe();
    }
}

template<typename DT_X, uint32_t kInRows, uint32_t kInCols, uint32_t kDepth, uint32_t kOutBatches,
          uint32_t kRowsPerSlab, uint32_t kBufferAlignBytes>
__aicore__ inline void RunTallRowSlab(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kOutRowsAll = kInRows * 2U;
    constexpr uint32_t kOutColsAll = kInCols * 2U;
    constexpr uint32_t kR2 = 2U;
    constexpr uint32_t kTileTotal = (kOutRowsAll + kRowsPerSlab - 1U) / kRowsPerSlab;
    constexpr uint32_t kDepthBlocks = (kDepth * sizeof(DT_X)) / 32U;
    constexpr uint32_t kReadRowBlocks = (kInCols * kDepth * sizeof(DT_X)) / 32U;
    constexpr uint32_t kWriteRowBlocks = (kOutColsAll * kDepth * sizeof(DT_X)) / 32U;
    constexpr uint32_t kInputRows = (kRowsPerSlab + 1U) / 2U;
    constexpr uint32_t kReadElems = kInputRows * kInCols * kDepth;
    constexpr uint32_t kWriteElems = kRowsPerSlab * kOutColsAll * kDepth;
    constexpr uint32_t kReadBytes =
        ((kReadElems * sizeof(DT_X) + kBufferAlignBytes - 1U) / kBufferAlignBytes) * kBufferAlignBytes;

    AscendC::GlobalTensor<DT_X> gmIn;
    AscendC::GlobalTensor<DT_X> gmOut;
    gmIn.SetGlobalBuffer((__gm__ DT_X*)x);
    gmOut.SetGlobalBuffer((__gm__ DT_X*)y);
    AscendC::LocalTensor<DT_X> stageA(AscendC::TPosition::VECCALC, 0, kReadElems);
    AscendC::LocalTensor<DT_X> stageB(AscendC::TPosition::VECCALC, kReadBytes, kReadElems);
    AscendC::LocalTensor<DT_X> stagedOut(AscendC::TPosition::VECCALC, 2U * kReadBytes, kWriteElems);

    AscendC::DataCopyParams loadPlan;
    loadPlan.blockCount = static_cast<uint16_t>(1U);
    loadPlan.blockLen = static_cast<uint16_t>(0U);
    loadPlan.srcStride = static_cast<uint32_t>(0U);
    loadPlan.dstStride = static_cast<uint32_t>(0U);

    AscendC::DataCopyParams columnPlan;
    columnPlan.blockCount = 0;
    columnPlan.blockLen = static_cast<uint16_t>(kDepthBlocks);
    columnPlan.srcStride = static_cast<uint16_t>((kInCols - 1U) * kDepthBlocks);
    columnPlan.dstStride = static_cast<uint16_t>(2U * kWriteRowBlocks - kDepthBlocks);

    AscendC::DataCopyParams storePlan;
    storePlan.blockCount = static_cast<uint16_t>(1U);
    storePlan.blockLen = static_cast<uint16_t>(0U);
    storePlan.srcStride = static_cast<uint32_t>(0U);
    storePlan.dstStride = static_cast<uint32_t>(0U);

    const uint32_t aivId = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t aivNum = static_cast<uint32_t>(AscendC::GetBlockNum());
    for (uint32_t jobIx = aivId; jobIx < kOutBatches * kTileTotal; jobIx += aivNum) {
        uint32_t tileIx = jobIx % kTileTotal;
        uint32_t n = jobIx / kTileTotal;
        uint32_t ohStart = tileIx * kRowsPerSlab;
        uint32_t rowTake = kOutRowsAll - ohStart;
        if (rowTake > kRowsPerSlab) {
            rowTake = kRowsPerSlab;
        }

        uint32_t firstBlockH = (ohStart & 1U);
        for (uint32_t pass = 0; pass < kR2; ++pass) {
            uint32_t hPhaseBit = pass;
            uint32_t firstRow = (hPhaseBit == firstBlockH) ? 0U : 1U;
            if (firstRow >= rowTake) {
                continue /* keep scanning */;
            }
            uint32_t activeRows = (rowTake - firstRow + 1U) >> 1U;
            uint32_t ihStart = (ohStart + firstRow) / 2U;
            uint32_t readBatch0 = hPhaseBit * kR2 * kOutBatches + n;
            uint32_t readBatch1 = readBatch0 + kOutBatches;
            uint32_t readBase0 = ((readBatch0 * kInRows + ihStart) * kInCols) * kDepth;
            uint32_t readBase1 = ((readBatch1 * kInRows + ihStart) * kInCols) * kDepth;

            loadPlan.blockLen = static_cast<uint16_t>(activeRows * kReadRowBlocks);
            AscendC::DataCopy(stageA, gmIn[readBase0], loadPlan);
            AscendC::DataCopy(stageB, gmIn[readBase1], loadPlan);
            SyncAllPipes();

            columnPlan.blockCount = static_cast<uint16_t>(activeRows);
            uint32_t outRowBase = firstRow * kOutColsAll * kDepth;
            for (uint32_t iw = 0; iw < kInCols; ++iw) {
                uint32_t srcOffset = iw * kDepth;
                uint32_t outColBase = outRowBase + (iw * kR2) * kDepth;
                AscendC::DataCopy(stagedOut[outColBase], stageA[srcOffset], columnPlan);
                AscendC::DataCopy(stagedOut[outColBase + kDepth], stageB[srcOffset], columnPlan);
            }
            SyncAllPipes();
        }

        uint32_t writeBase = (n * kOutRowsAll + ohStart) * kOutColsAll * kDepth;
        storePlan.blockLen = static_cast<uint16_t>(rowTake * kWriteRowBlocks);
        AscendC::DataCopy(gmOut[writeBase], stagedOut, storePlan);
        SyncMte3Pipe();
    }
}

template<typename DT_X, uint32_t kInRows, uint32_t kInCols, uint32_t kDepth,
         uint32_t kOutBatches, uint32_t kBufferAlignBytes, bool kLoadBarrierAll,
         bool kUbBarrierAll>
__aicore__ inline void RunSingleRowR2Tile(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kOutRowsAll = kInRows * 2U;
    constexpr uint32_t kOutColsAll = kInCols * 2U;
    constexpr uint32_t kR2 = 2U;
    constexpr uint32_t kDepthBlocks = (kDepth * sizeof(DT_X)) / 32U;
    constexpr uint32_t kReadRowBlocks = (kInCols * kDepth * sizeof(DT_X)) / 32U;
    constexpr uint32_t kWriteRowBlocks = (kOutColsAll * kDepth * sizeof(DT_X)) / 32U;
    constexpr uint32_t kReadElems = kInCols * kDepth;
    constexpr uint32_t kWriteElems = kOutColsAll * kDepth;
    constexpr uint32_t kReadBytes =
        ((kReadElems * sizeof(DT_X) + kBufferAlignBytes - 1U) / kBufferAlignBytes) * kBufferAlignBytes;

    AscendC::GlobalTensor<DT_X> gmIn;
    AscendC::GlobalTensor<DT_X> gmOut;
    gmIn.SetGlobalBuffer((__gm__ DT_X*)x);
    gmOut.SetGlobalBuffer((__gm__ DT_X*)y);
    AscendC::LocalTensor<DT_X> stageA(AscendC::TPosition::VECCALC, 0, kReadElems);
    AscendC::LocalTensor<DT_X> stageB(AscendC::TPosition::VECCALC, kReadBytes, kReadElems);
    AscendC::LocalTensor<DT_X> stagedOut(AscendC::TPosition::VECCALC, 2U * kReadBytes, kWriteElems);

    AscendC::DataCopyParams loadPlan;
    loadPlan.blockCount = static_cast<uint16_t>(1U);
    loadPlan.blockLen = static_cast<uint16_t>(kReadRowBlocks);
    loadPlan.srcStride = static_cast<uint32_t>(0U);
    loadPlan.dstStride = static_cast<uint32_t>(0U);

    AscendC::DataCopyParams ubPlan;
    ubPlan.blockCount = static_cast<uint16_t>(kInCols);
    ubPlan.blockLen = static_cast<uint16_t>(kDepthBlocks);
    ubPlan.srcStride = static_cast<uint32_t>(0U);
    ubPlan.dstStride = static_cast<uint16_t>(kDepthBlocks);

    AscendC::DataCopyParams storePlan;
    storePlan.blockCount = static_cast<uint16_t>(1U);
    storePlan.blockLen = static_cast<uint16_t>(kWriteRowBlocks);
    storePlan.srcStride = static_cast<uint32_t>(0U);
    storePlan.dstStride = static_cast<uint32_t>(0U);

    const uint32_t aivId = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t aivNum = static_cast<uint32_t>(AscendC::GetBlockNum());
    for (uint32_t group = aivId; group < kOutBatches * kR2 * kInRows; group += aivNum) {
        uint32_t ih = group % kInRows;
        uint32_t tmp = group / kInRows;
        uint32_t hPhaseBit = tmp & 1U;
        uint32_t n = tmp / 2U;
        uint32_t readBatch0 = hPhaseBit * kR2 * kOutBatches + n;
        uint32_t readBatch1 = readBatch0 + kOutBatches;
        uint32_t readBase0 = ((readBatch0 * kInRows + ih) * kInCols) * kDepth;
        uint32_t readBase1 = ((readBatch1 * kInRows + ih) * kInCols) * kDepth;

        AscendC::DataCopy(stageA, gmIn[readBase0], loadPlan);
        AscendC::DataCopy(stageB, gmIn[readBase1], loadPlan);
        SyncAfterLoad<kLoadBarrierAll>();

        AscendC::DataCopy(stagedOut, stageA, ubPlan);
        AscendC::DataCopy(stagedOut[kDepth], stageB, ubPlan);
        SyncAfterUbMove<false, kUbBarrierAll>();

        uint32_t writeBase = (n * kOutRowsAll + hPhaseBit + ih * kR2) * kOutColsAll * kDepth;
        AscendC::DataCopy(gmOut[writeBase], stagedOut, storePlan);
        SyncMte3Pipe();
    }
}

template<typename DT_X, uint32_t kInRows, uint32_t kInCols, uint32_t kDepth, uint32_t kColsPerTile>
__aicore__ inline void RunWideInputRowScatter(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kOutRowsAll = kInRows * 2U;
    constexpr uint32_t kOutColsAll = kInCols * 2U;
    constexpr uint32_t kR2 = 2U;
    constexpr uint32_t kPackedInputBatches = 4U;
    constexpr uint32_t kInputRowTotal = kPackedInputBatches * kInRows;
    constexpr uint32_t kBlockNum = 32U + 8U;
    constexpr uint32_t kCBytes = kDepth * sizeof(DT_X);
    constexpr uint32_t kDepthBlocks = kCBytes / 32U;
    constexpr uint32_t kTileElems = kColsPerTile * kDepth;
    constexpr event_t kWidePing = static_cast<event_t>(0);
    constexpr event_t kWidePong = static_cast<event_t>(1);

    AscendC::GlobalTensor<DT_X> gmIn;
    AscendC::GlobalTensor<DT_X> gmOut;
    gmIn.SetGlobalBuffer((__gm__ DT_X*)x);
    gmOut.SetGlobalBuffer((__gm__ DT_X*)y);
    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub{};
    AscendC::LocalTensor<DT_X> stageA = ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(kTileElems);
    AscendC::LocalTensor<DT_X> stageB = ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(kTileElems);

    AscendC::DataCopyParams loadPlan;
    loadPlan.blockCount = static_cast<uint16_t>(1U);
    loadPlan.blockLen = static_cast<uint16_t>(0U);
    loadPlan.srcStride = static_cast<uint32_t>(0U);
    loadPlan.dstStride = static_cast<uint32_t>(0U);

    AscendC::DataCopyExtParams storePlan;
    storePlan.blockCount = static_cast<uint16_t>(1U);
    storePlan.blockLen = kCBytes;
    storePlan.srcStride = static_cast<uint32_t>(0U);
    storePlan.dstStride = kCBytes;
    storePlan.rsv = static_cast<uint32_t>(0U);

    const uint32_t aivId = static_cast<uint32_t>(AscendC::GetBlockIdx());
    if constexpr (kInputRowTotal <= kBlockNum) {
        if (aivId >= kInputRowTotal) {
            return void();
        }
        uint32_t rowIx = aivId;
        uint32_t packedBatch = rowIx / kInRows;
        uint32_t ih = rowIx - packedBatch * kInRows;
        uint32_t hPhaseBit = packedBatch >> 1;
        uint32_t wPhaseBit = packedBatch & 1U;
        uint32_t oh = ih * kR2 + hPhaseBit;
        uint64_t readRowBase = static_cast<uint64_t>(rowIx) * kInCols * kDepth;
        uint64_t writeRowBase = (static_cast<uint64_t>(oh) * kOutColsAll + wPhaseBit) * kDepth;

        if constexpr ((kInCols % kColsPerTile) == 0U) {
            constexpr uint32_t kTileTotal = kInCols / kColsPerTile;
            loadPlan.blockLen = static_cast<uint16_t>(kColsPerTile * kDepthBlocks);
            storePlan.blockCount = static_cast<uint16_t>(kColsPerTile);
            AscendC::DataCopy(stageA, gmIn[readRowBase], loadPlan);
            MarkLoadStore(kWidePing);

# pragma unroll
            for (uint32_t tileIx = 0; tileIx < kTileTotal; ++tileIx) {
                uint32_t bufIdx = tileIx & 1U;
                event_t curEvent = bufIdx == 0 ? kWidePing : kWidePong;
                event_t nextEvent = bufIdx == 0 ? kWidePong : kWidePing;
                AscendC::LocalTensor<DT_X> liveStage = bufIdx == 0 ? stageA : stageB;
                AscendC::LocalTensor<DT_X> fillStage = bufIdx == 0 ? stageB : stageA;

                WaitLoadStore(curEvent);
                if (tileIx + 1U < kTileTotal) {
                    uint64_t nextReadBase = readRowBase + static_cast<uint64_t>(tileIx + 1U) * kColsPerTile * kDepth;
                    AscendC::DataCopy(fillStage, gmIn[nextReadBase], loadPlan);
                    MarkLoadStore(nextEvent);
                }

                uint64_t writeBase = writeRowBase + static_cast<uint64_t>(tileIx) * kColsPerTile * kR2 * kDepth;
                AscendC::DataCopyPad(gmOut[writeBase], liveStage, storePlan);
                MarkStoreLoad(curEvent);
                WaitStoreLoad(curEvent);
            }
            return void();
        }

        uint32_t colDone = 0;
        uint32_t colTake = kInCols > kColsPerTile ? kColsPerTile : kInCols;
        loadPlan.blockLen = static_cast<uint16_t>(colTake * kDepthBlocks);
        AscendC::DataCopy(stageA, gmIn[readRowBase], loadPlan);
        MarkLoadStore(kWidePing);

        uint32_t bufIdx = 0U;
        while (colDone < kInCols) {
            event_t curEvent = bufIdx == 0 ? kWidePing : kWidePong;
            event_t nextEvent = bufIdx == 0 ? kWidePong : kWidePing;
            AscendC::LocalTensor<DT_X> liveStage = bufIdx == 0 ? stageA : stageB;
            AscendC::LocalTensor<DT_X> fillStage = bufIdx == 0 ? stageB : stageA;

            WaitLoadStore(curEvent);

            uint32_t nextColDone = colDone + colTake;
            uint32_t nextColTake = 0;
            if (nextColDone < kInCols) {
                nextColTake = kInCols - nextColDone;
                if (nextColTake > kColsPerTile) {
                    nextColTake = kColsPerTile;
                }
                uint64_t nextReadBase = readRowBase + static_cast<uint64_t>(nextColDone) * kDepth;
                loadPlan.blockLen = static_cast<uint16_t>(nextColTake * kDepthBlocks);
                AscendC::DataCopy(fillStage, gmIn[nextReadBase], loadPlan);
                MarkLoadStore(nextEvent);
            }

            uint64_t writeBase = writeRowBase + static_cast<uint64_t>(colDone) * kR2 * kDepth;
            storePlan.blockCount = static_cast<uint16_t>(colTake);
            AscendC::DataCopyPad(gmOut[writeBase], liveStage, storePlan);
            MarkStoreLoad(curEvent);
            WaitStoreLoad(curEvent);

            colDone = nextColDone;
            colTake = nextColTake;
            bufIdx = (bufIdx ^ 1U);
        }
        return void();
    }

    uint32_t rowsPerCore = (kInputRowTotal + kBlockNum - 1U) / kBlockNum;
    uint32_t startRow = aivId * rowsPerCore;
    if (startRow >= kInputRowTotal) {
        return void();
    }
    uint32_t rowCount = (rowsPerCore);
    if (startRow + rowCount > kInputRowTotal) {
        rowCount = kInputRowTotal - startRow;
    }

    for (uint32_t r = 0U; r < rowCount; ++r) {
        uint32_t rowIx = startRow + r;
        uint32_t packedBatch = rowIx / kInRows;
        uint32_t ih = rowIx - packedBatch * kInRows;
        uint32_t hPhaseBit = packedBatch >> 1;
        uint32_t wPhaseBit = packedBatch & 1U;
        uint32_t oh = ih * kR2 + hPhaseBit;
        uint64_t readRowBase = static_cast<uint64_t>(rowIx) * kInCols * kDepth;
        uint64_t writeRowBase = (static_cast<uint64_t>(oh) * kOutColsAll + wPhaseBit) * kDepth;

        if constexpr ((kInCols % kColsPerTile) == 0U) {
            constexpr uint32_t kTileTotal = kInCols / kColsPerTile;
            loadPlan.blockLen = static_cast<uint16_t>(kColsPerTile * kDepthBlocks);
            storePlan.blockCount = static_cast<uint16_t>(kColsPerTile);
            AscendC::DataCopy(stageA, gmIn[readRowBase], loadPlan);
            MarkLoadStore(kWidePing);

# pragma unroll
            for (uint32_t tileIx = 0; tileIx < kTileTotal; ++tileIx) {
                uint32_t bufIdx = tileIx & 1U;
                event_t curEvent = bufIdx == 0 ? kWidePing : kWidePong;
                event_t nextEvent = bufIdx == 0 ? kWidePong : kWidePing;
                AscendC::LocalTensor<DT_X> liveStage = bufIdx == 0 ? stageA : stageB;
                AscendC::LocalTensor<DT_X> fillStage = bufIdx == 0 ? stageB : stageA;

                WaitLoadStore(curEvent);
                if (tileIx + 1U < kTileTotal) {
                    uint64_t nextReadBase = readRowBase + static_cast<uint64_t>(tileIx + 1U) * kColsPerTile * kDepth;
                    AscendC::DataCopy(fillStage, gmIn[nextReadBase], loadPlan);
                    MarkLoadStore(nextEvent);
                }

                uint64_t writeBase = writeRowBase + static_cast<uint64_t>(tileIx) * kColsPerTile * kR2 * kDepth;
                AscendC::DataCopyPad(gmOut[writeBase], liveStage, storePlan);
                MarkStoreLoad(curEvent);
                WaitStoreLoad(curEvent);
            }
            continue /* keep scanning */;
        }

        uint32_t colDone = 0;
        uint32_t colTake = kInCols > kColsPerTile ? kColsPerTile : kInCols;
        loadPlan.blockLen = static_cast<uint16_t>(colTake * kDepthBlocks);
        AscendC::DataCopy(stageA, gmIn[readRowBase], loadPlan);
        MarkLoadStore(kWidePing);

        uint32_t bufIdx = 0U;
        while (colDone < kInCols) {
            event_t curEvent = bufIdx == 0 ? kWidePing : kWidePong;
            event_t nextEvent = bufIdx == 0 ? kWidePong : kWidePing;
            AscendC::LocalTensor<DT_X> liveStage = bufIdx == 0 ? stageA : stageB;
            AscendC::LocalTensor<DT_X> fillStage = bufIdx == 0 ? stageB : stageA;

            WaitLoadStore(curEvent);

            uint32_t nextColDone = colDone + colTake;
            uint32_t nextColTake = 0;
            if (nextColDone < kInCols) {
                nextColTake = kInCols - nextColDone;
                if (nextColTake > kColsPerTile) {
                    nextColTake = kColsPerTile;
                }
                uint64_t nextReadBase = readRowBase + static_cast<uint64_t>(nextColDone) * kDepth;
                loadPlan.blockLen = static_cast<uint16_t>(nextColTake * kDepthBlocks);
                AscendC::DataCopy(fillStage, gmIn[nextReadBase], loadPlan);
                MarkLoadStore(nextEvent);
            }

            uint64_t writeBase = writeRowBase + static_cast<uint64_t>(colDone) * kR2 * kDepth;
            storePlan.blockCount = static_cast<uint16_t>(colTake);
            AscendC::DataCopyPad(gmOut[writeBase], liveStage, storePlan);
            MarkStoreLoad(curEvent);
            WaitStoreLoad(curEvent);

            colDone = nextColDone;
            colTake = nextColTake;
            bufIdx = (bufIdx ^ 1U);
        }
    }
}
template<typename DT_X, uint32_t kRows, uint32_t kCols, uint32_t kTinyDepth,
         uint32_t kGroupElems, uint32_t kCBlocks, uint32_t kGroupsPerTile>
__aicore__ inline void LoadR4ShiftGroup(AscendC::GlobalTensor<DT_X> &gmRead,
                                            AscendC::LocalTensor<DT_X> phasePair01,
                                            AscendC::LocalTensor<DT_X> phasePair23,
                                            AscendC::DataCopyParams &copyCtl,
                                            uint32_t phaseGroup, uint32_t rowIx,
                                            uint32_t col, uint32_t frontGroups) {
    uint32_t phaseRoot = phaseGroup * 4U;
    uint32_t read0 = ((phaseRoot * kRows + rowIx) * kCols + col + 1U) * kTinyDepth;
    uint32_t read1 = (((phaseRoot + 1U) * kRows + rowIx) * kCols + col) * kTinyDepth;
    uint32_t read2 = (((phaseRoot + 2U) * kRows + rowIx) * kCols + col) * kTinyDepth;
    uint32_t read3 = (((phaseRoot + 3U) * kRows + rowIx) * kCols + col) * kTinyDepth;

    copyCtl.blockLen = static_cast<uint16_t>(frontGroups * kCBlocks);
    AscendC::DataCopy(phasePair01, gmRead[read0], copyCtl);
    copyCtl.blockLen = static_cast<uint16_t>(kGroupsPerTile * kCBlocks);
    AscendC::DataCopy(phasePair01[kGroupElems], gmRead[read1], copyCtl);
    AscendC::DataCopy(phasePair23, gmRead[read2], copyCtl);
    AscendC::DataCopy(phasePair23[kGroupElems], gmRead[read3], copyCtl);
}

template<typename DT_X, uint32_t kTinyDepth, uint32_t kGroupElems,
         uint32_t kCBlocks, uint32_t kGroupsPerTile>
__aicore__ inline void ScatterR4ShiftGroup(AscendC::LocalTensor<DT_X> r4OutTile,
                                               AscendC::LocalTensor<DT_X> phasePair01,
                                               AscendC::LocalTensor<DT_X> phasePair23,
                                               AscendC::DataCopyParams &copyCtl,
                                               uint32_t frontGroups) {
    copyCtl.blockCount = static_cast<uint16_t>(frontGroups);
    AscendC::DataCopy(r4OutTile[3U * kTinyDepth], phasePair01, copyCtl);
    copyCtl.blockCount = static_cast<uint16_t>(kGroupsPerTile);
    AscendC::DataCopy(r4OutTile, phasePair01[kGroupElems], copyCtl);
    AscendC::DataCopy(r4OutTile[kTinyDepth], phasePair23, copyCtl);
    AscendC::DataCopy(r4OutTile[2U * kTinyDepth], phasePair23[kGroupElems], copyCtl);
}

template<typename DT_X, uint32_t kInRows, uint32_t kInCols, uint32_t kDepth, uint32_t kOutRowsAll, uint32_t kOutColsAll,
          uint32_t kCropLeft, uint32_t kColsPerTile>
__aicore__ inline void RunLeftCropR4Stripe(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kSpatial = 4U;
    constexpr uint32_t kColTileTotal = (kOutColsAll + kColsPerTile - 1U) / kColsPerTile;
    constexpr uint32_t kPhaseGroups = kColsPerTile / kSpatial;
    constexpr uint32_t kLaneBytes = kDepth * sizeof(DT_X);
    constexpr uint32_t kLaneBlocks = (kLaneBytes / 32U);
    constexpr uint32_t kPhaseBytes = (kPhaseGroups * kLaneBytes);
    constexpr uint32_t kPhaseElems = (kPhaseBytes / sizeof(DT_X));
    constexpr uint32_t kWriteElems = kColsPerTile * kDepth;
    constexpr uint32_t kInputPairElems = kPhaseElems + kPhaseElems;

    AscendC::GlobalTensor<DT_X> gmRead;
    AscendC::GlobalTensor<DT_X> gmWrite;
    AttachGmPair(x, y, gmRead, gmWrite);
    AscendC::LocalTensor<DT_X> r4OutTile(AscendC::TPosition::VECCALC, 0, kWriteElems);
    AscendC::LocalTensor<DT_X> phasePair01(AscendC::TPosition::VECCALC, 65536, kInputPairElems);
    AscendC::LocalTensor<DT_X> phasePair23(AscendC::TPosition::VECCALC, 98304, kInputPairElems);

    AscendC::DataCopyParams loadPlan;
    ClearCopyPlan(loadPlan);

    AscendC::DataCopyParams ubPlan;
    ClearCopyPlan(ubPlan);
    ubPlan.blockLen = static_cast<uint16_t>(kLaneBlocks);
    ubPlan.srcStride = static_cast<uint32_t>(0U);
    ubPlan.dstStride = static_cast<uint16_t>((kSpatial - 1U) * kLaneBlocks);

    AscendC::DataCopyParams storePlan;
    ClearCopyPlan(storePlan);

    const uint32_t laneStart = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t aivNum = static_cast<uint32_t>(AscendC::GetBlockNum());
    for (uint32_t dstH = laneStart; dstH < kOutRowsAll; dstH += aivNum) {
        uint32_t sourceH = dstH / kSpatial;
        uint32_t heightPhase = dstH - sourceH * kSpatial;
        for (uint32_t tileId = 0; tileId < kColTileTotal; ++tileId) {
            uint32_t outCol = tileId * kColsPerTile;
            uint32_t liveCols = kOutColsAll - outCol;
            if (liveCols > kColsPerTile) {
                liveCols = kColsPerTile;
            }
            uint32_t inputCol = (outCol + kCropLeft) / kSpatial;
            uint32_t frontGroups = liveCols == kColsPerTile ? kPhaseGroups : kPhaseGroups - 1U;

            LoadR4ShiftGroup<DT_X, kInRows, kInCols, kDepth, kPhaseElems,
                                  (kLaneBlocks), kPhaseGroups>(
                gmRead, phasePair01, phasePair23, loadPlan, heightPhase, sourceH, inputCol, frontGroups);
            SyncAllPipes();

            ScatterR4ShiftGroup<DT_X, kDepth, kPhaseElems, kLaneBlocks, kPhaseGroups>(
                r4OutTile, phasePair01, phasePair23, ubPlan, frontGroups);
            SyncAllPipes();

            storePlan.blockLen = static_cast<uint16_t>((liveCols * kLaneBytes) / 32U);
            AscendC::DataCopy(gmWrite[(dstH * kOutColsAll + outCol) * kDepth], r4OutTile, storePlan);
            if (tileId + 1U < kColTileTotal || dstH + aivNum < kOutRowsAll) {
                SyncMte3Pipe();
            }
        }
    }
}


template<typename DT_X>
__aicore__ inline void LoadDenseCropHalf(AscendC::GlobalTensor<DT_X> &gmIn,
                                                 AscendC::LocalTensor<DT_X> activeSlot,
                                                 uint32_t jobIx) {
    constexpr uint32_t kInRows = 128;
    constexpr uint32_t kInCols = 128;
    constexpr uint32_t kDepth = 65;
    constexpr uint32_t kCropTop = 1U;
    constexpr uint32_t kLaneCols = 32U * 2U;
    constexpr uint32_t kLaneElems = kLaneCols * kDepth;

    uint32_t oh = jobIx >> 1U;
    uint32_t half = jobIx & 1U;
    uint32_t croppedH = oh + kCropTop;
    uint32_t ih = croppedH >> 1;
    uint32_t hPhaseBit = croppedH & 1U;
    uint32_t phaseBatchA = hPhaseBit * 2U + (half == 0U ? 1U : 0U);
    uint32_t phaseBatchB = hPhaseBit * 2U + (half == 0U ? 0U : 1U);
    uint32_t firstIw = (half == 0U) ? 0U : 64U;
    uint32_t secondIw = (half == 0U) ? 1U : 64U;
    uint64_t phaseReadA = ((static_cast<uint64_t>(phaseBatchA) * kInRows + ih) * kInCols + firstIw) * kDepth;
    uint64_t phaseReadB = ((static_cast<uint64_t>(phaseBatchB) * kInRows + ih) * kInCols + secondIw) * kDepth;
    AscendC::DataCopy(activeSlot, gmIn[phaseReadA], kLaneElems);
    AscendC::DataCopy(activeSlot[kLaneElems], gmIn[phaseReadB], kLaneElems);
}

template<typename DT_X>
__aicore__ inline uint64_t DenseCropDst(uint32_t jobIx) {
    constexpr uint32_t kDepth = 65;
    constexpr uint32_t kOutColsAll = 254;
    constexpr uint32_t kHalfCols = kOutColsAll / 2U;
    uint32_t oh = jobIx >> 1U;
    uint32_t half = jobIx & 1U;
    return (static_cast<uint64_t>(oh) * kOutColsAll + half * kHalfCols) * kDepth;
}

template<typename DT_X>
__aicore__ inline void RunDenseCropGather(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kDepth = 65;
    constexpr uint32_t kOutRowsAll = 254;
    constexpr uint32_t kOutColsAll = 254;
    constexpr uint32_t kHalfCols = kOutColsAll / 2U;
    constexpr uint32_t kLaneCols = 32U * 2U;
    constexpr uint32_t kLaneElems = kLaneCols * kDepth;
    constexpr uint32_t kReadElems = 2U * kLaneElems;
    constexpr uint32_t kWriteElems = kHalfCols * kDepth;
    constexpr uint32_t kSlotElems = (16576U);
    constexpr uint32_t kSlotBytes = sizeof(DT_X) * kSlotElems;
    constexpr uint32_t kIndexBaseBytes = kSlotBytes + kSlotBytes;
    constexpr uint32_t kPairElems = 2U * kDepth;
    constexpr uint32_t kGatherPatternPairs = 4U;
    constexpr uint32_t kGatherPatternElems = kGatherPatternPairs * kPairElems;
    constexpr uint32_t kGatherPatternCopies = kWriteElems / kGatherPatternElems;
    constexpr uint32_t kIndexTailElems =
        ((kWriteElems - kGatherPatternCopies * kGatherPatternElems + 7U) / 8U) * 8U;
    constexpr uint32_t kGatherMapElems = kGatherPatternCopies * kGatherPatternElems + kIndexTailElems;
    constexpr event_t kSeedVectorFlag = static_cast<event_t>(6);
    constexpr event_t kSlotFreeA = static_cast<event_t>(0);
    constexpr event_t kSlotFreeB = static_cast<event_t>(1);
    constexpr event_t kSlotLoadA = static_cast<event_t>(2);
    constexpr event_t kSlotLoadB = static_cast<event_t>(3);
    constexpr event_t kSlotGatherA = static_cast<event_t>(4);
    constexpr event_t kSlotGatherB = static_cast<event_t>(5);
    static_assert(kReadElems + kWriteElems <= kSlotElems, "dense gather activeSlot overflow");
    static_assert(kIndexBaseBytes + kGatherMapElems * sizeof(uint32_t) <= 184U * 1024U,
                  "dense gather index map overflow");

    AscendC::GlobalTensor<DT_X> gmIn;
    AscendC::GlobalTensor<DT_X> gmOut;
    gmIn.SetGlobalBuffer((__gm__ DT_X*)x);
    gmOut.SetGlobalBuffer((__gm__ DT_X*)y);

    AscendC::LocalTensor<DT_X> pipeSlotA(AscendC::TPosition::VECCALC, 0, kSlotElems);
    AscendC::LocalTensor<DT_X> pipeSlotB(AscendC::TPosition::VECCALC, kSlotBytes, kSlotElems);
    AscendC::LocalTensor<int32_t> offsetSeed(AscendC::TPosition::VECCALC,
                                             kIndexBaseBytes, kGatherMapElems);
    AscendC::LocalTensor<uint32_t> indexMap(AscendC::TPosition::VECCALC,
                                               kIndexBaseBytes, kWriteElems);

    for (uint32_t pairIx = 0; pairIx < kGatherPatternPairs; ++pairIx) {
        uint32_t denseSeedBase = pairIx * kPairElems;
        uint32_t denseByteSeed = pairIx * kDepth * sizeof(DT_X);
        for (uint32_t c = 0; c < kDepth; ++c) {
            offsetSeed.SetValue(denseSeedBase + c, static_cast<int32_t>(denseByteSeed + c * sizeof(DT_X)));
            offsetSeed.SetValue(
                denseSeedBase + kDepth + c,
                static_cast<int32_t>(kLaneElems * sizeof(DT_X) + denseByteSeed + c * sizeof(DT_X)));
        }
    }
    MarkScalarVector(kSeedVectorFlag);
    WaitScalarVector(kSeedVectorFlag);
    for (uint32_t patternIx = 1; patternIx <= kGatherPatternCopies; ++patternIx) {
        uint32_t denseVectorCount = (patternIx == kGatherPatternCopies) ? kIndexTailElems : kGatherPatternElems;
        AscendC::Adds(offsetSeed[patternIx * kGatherPatternElems],
                      offsetSeed[(patternIx - 1U) * kGatherPatternElems],
                      static_cast<int32_t>(kGatherPatternPairs * kDepth * sizeof(DT_X)), denseVectorCount);
    }
    SyncVectorPipe();

    AscendC::DataCopyExtParams storePlan;
    storePlan.blockCount = static_cast<uint16_t>(1U);
    storePlan.blockLen = kWriteElems * sizeof(DT_X);
    storePlan.srcStride = static_cast<uint32_t>(0U);
    storePlan.dstStride = static_cast<uint32_t>(0U);
    storePlan.rsv = static_cast<uint32_t>(0U);

    const uint32_t aivId = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t aivNum = static_cast<uint32_t>(AscendC::GetBlockNum());
    constexpr uint32_t kDenseJobs = kOutRowsAll * 2U;
    uint32_t jobsBase = kDenseJobs / aivNum;
    uint32_t jobsExtra = kDenseJobs - jobsBase * aivNum;
    uint32_t jobStart = aivId * jobsBase + (aivId < jobsExtra ? aivId : jobsExtra);
    uint32_t jobCount = jobsBase + (aivId < jobsExtra ? 1U : 0U);
    if (jobCount == 0U) {
        return void();
    }

    MarkStoreLoad(kSlotFreeA);
    MarkStoreLoad(kSlotFreeB);
    WaitStoreLoad(kSlotFreeA);
    LoadDenseCropHalf<DT_X>(gmIn, pipeSlotA, jobStart);
    MarkLoadVector(kSlotLoadA);

    for (uint32_t idx = 0; idx < jobCount; ++idx) {
        uint32_t buf = (idx & 1U);
        uint32_t jobIx = jobStart + idx;
        AscendC::LocalTensor<DT_X> activeSlot = buf == 0U ? pipeSlotA : pipeSlotB;
        event_t loadEvent = buf == 0U ? kSlotLoadA : kSlotLoadB;
        event_t gatherEvent = buf == 0U ? kSlotGatherA : kSlotGatherB;
        event_t freeEvent = buf == 0U ? kSlotFreeA : kSlotFreeB;

        WaitLoadVector(loadEvent);
        AscendC::Gather(activeSlot[kReadElems], activeSlot, indexMap, 0U, kWriteElems);
        MarkVectorStore(gatherEvent);

        uint32_t next = (idx + 1U);
        if (next < jobCount) {
            uint32_t nextBuf = (next & 1U);
            event_t nextFreeEvent = nextBuf == 0U ? kSlotFreeA : kSlotFreeB;
            event_t nextLoadEvent = nextBuf == 0U ? kSlotLoadA : kSlotLoadB;
            AscendC::LocalTensor<DT_X> nextActiveSlot = nextBuf == 0U ? pipeSlotA : pipeSlotB;
            WaitStoreLoad(nextFreeEvent);
            LoadDenseCropHalf<DT_X>(gmIn, nextActiveSlot, jobStart + next);
            MarkLoadVector(nextLoadEvent);
        }

        WaitVectorStore(gatherEvent);
        AscendC::DataCopyPad(gmOut[DenseCropDst<DT_X>(jobIx)],
                             activeSlot[kReadElems], storePlan);
        MarkStoreLoad(freeEvent);
    }
    WaitStoreLoad(kSlotFreeA);
    WaitStoreLoad(kSlotFreeB);
}
template<typename DT_X, uint64_t SCH_MODE>
__global__ __aicore__ void batch_to_space(GM_ADDR input, GM_ADDR output,
                                           GM_ADDR scratch, GM_ADDR rawTiling) {
    REGISTER_TILING_DEFAULT( BatchToSpaceTilingData );
    (void)scratch;
    (void)rawTiling;

    if constexpr (SCH_MODE == BTS_TILE_F32_R2_N2_28X28_C128) {
        RunSingleRowR2Tile<DT_X, 28U, 28U, 128U, 2U, 16384U, true, false>(input, output);
        return void();
    }
    if constexpr (SCH_MODE == BTS_TILE_F32_R2_CROP_17X26_C5) {
        RunTinyCropGather<DT_X>(input, output);
        return void();
    }
    if constexpr (SCH_MODE == BTS_TILE_F32_R2_N4_14X14_C64) {
        RunRowPhaseR2Tile<DT_X, 14U, 14U, 64U, 4U, 7U, 512U, true, false, true>(input, output);
        return void();
    }
    if constexpr (SCH_MODE == BTS_TILE_F32_R2_N5_4X6_C32) {
        RunRowPhaseR2Tile<DT_X, 4U, 6U, 32U, 5U, 4U, 4096U, false, false, false>(input, output);
        return void();
    }
    if constexpr (SCH_MODE == BTS_TILE_F16_R2_CROP_254X254_C65) {
        RunDenseCropGather<DT_X>(input, output);
        return void();
    }
    if constexpr (SCH_MODE == BTS_TILE_F16_R2_N1_2X2_C4096) {
        RunChannelMajorR2Tile<DT_X, 2U, 2U, 4096U, 1U, 2048U>(input, output);
        return void();
    }
    if constexpr (SCH_MODE == BTS_TILE_F16_R2_FLAT_C16384) {
        RunFlatAlignedRelay<DT_X>(input, output);
        return void();
    }
    if constexpr (SCH_MODE == BTS_TILE_F16_R2_N1_10X512_C256) {
        RunWideInputRowScatter<DT_X, 10U, 512U, 256U, 128U>(input, output);
        return void();
    }
    if constexpr (SCH_MODE == BTS_TILE_F16_R4_LEFT513_C64) {
        RunLeftCropR4Stripe<DT_X, 10U, 512U, 64U, 40U, 1535U, 513U, 512U>(input, output);
        return void();
    }
    if constexpr (SCH_MODE == BTS_TILE_F16_R2_N4_1024X6_C32) {
# if BTS_ENABLE_ROW_SLAB_TRIAL
        RunTallRowSlab<DT_X, 1024U, 6U, 32U, 4U, 120U, 512U>(input, output);
# else
        RunRowPhaseR2Tile<DT_X, 1024U, 6U, 32U, 4U, 52U, 512U, true, true, false>(input, output);
# endif
        return void();
    }

    return void();
}
