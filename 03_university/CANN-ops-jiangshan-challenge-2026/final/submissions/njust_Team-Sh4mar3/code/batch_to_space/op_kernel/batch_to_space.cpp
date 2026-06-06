#define K_MAX_SHAPE_DIM 0

#include "kernel_operator.h"

#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

namespace {
constexpr uint32_t BTS_TILE_BYTES = 36U * 1024U;
constexpr uint32_t BTS_B2_CROP_TILE_BYTES = 36U * 1024U;
constexpr uint32_t BTS_GATHER_B2_CROP_TILE_BYTES = 160U * 1024U;
constexpr uint32_t BTS_MAX_COPY_PARAM = 65535U;
constexpr uint32_t BTS_OFFSET_ADDS_CHUNK = 1024U;
constexpr event_t BTS_EVENT_MTE2_MTE3 = static_cast<event_t>(EVENT_ID0);
constexpr event_t BTS_EVENT_MTE3_MTE2 = static_cast<event_t>(EVENT_ID1);
constexpr event_t BTS_EVENT_MTE2_MTE3_DB = static_cast<event_t>(EVENT_ID2);
constexpr event_t BTS_EVENT_MTE3_MTE2_DB = static_cast<event_t>(EVENT_ID3);
constexpr event_t BTS_EVENT_MTE2_V = static_cast<event_t>(EVENT_ID4);
constexpr event_t BTS_EVENT_V_MTE3 = static_cast<event_t>(EVENT_ID5);
constexpr event_t BTS_EVENT_S_V = static_cast<event_t>(EVENT_ID6);
constexpr event_t BTS_EVENT_MTE3_V = static_cast<event_t>(EVENT_ID7);
constexpr event_t BTS_EVENT_V_MTE2 = static_cast<event_t>(EVENT_ID7);

#define BTS_ALIGN_UP_CONST(value, align) \
    (((value) + (align) - 1U) / (align) * (align))

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) {
    return (a < b) ? a : b;
}

__aicore__ inline uint32_t AlignUpU32(uint32_t value, uint32_t align) {
    return ((value + align - 1U) / align) * align;
}

__aicore__ inline uint32_t GcdU32(uint32_t a, uint32_t b) {
    while (b != 0U) {
        uint32_t remainder = a - (a / b) * b;
        a = b;
        b = remainder;
    }
    return a;
}

__aicore__ inline uint32_t CalcGatherAlignedGroupCols(uint32_t channelCount) {
    constexpr uint32_t offsetElemsPerBlock = 32U / sizeof(int32_t);
    uint32_t pairElems = channelCount << 1U;
    uint32_t alignedPairs =
        offsetElemsPerBlock / GcdU32(pairElems, offsetElemsPerBlock);
    return alignedPairs << 1U;
}

template <uint32_t SCH_MODE>
__aicore__ inline uint32_t CalcTotalTasks(const BatchToSpaceTilingData &tiling) {
    uint32_t widthDepthTasks = tiling.widthTiles * tiling.depthTiles;
    if constexpr (SCH_MODE == BTS_SCH_MODE_GATHER_B2_CROP) {
        return tiling.outBatch * tiling.outHeight * tiling.widthTiles;
    } else if constexpr (SCH_MODE == BTS_SCH_MODE_DIRECT_B2_NOCROP) {
        return (tiling.outBatch << 2U) * tiling.height * widthDepthTasks;
    } else {
        return tiling.outBatch * tiling.blockSize * tiling.blockSize *
               tiling.height * widthDepthTasks;
    }
}

}  // namespace

template <class DT_X>
__aicore__ inline bool RunWideFloatRowPairT01(GM_ADDR x, GM_ADDR y) {
    if constexpr (sizeof(DT_X) != 4U) {
        return false;
    } else {
        constexpr uint32_t BLOCK_BYTES = 32U;
        constexpr uint32_t SRC_HEIGHT = 28U;
        constexpr uint32_t SRC_WIDTH = 28U;
        constexpr uint32_t CHANNELS = 128U;
        constexpr uint32_t DST_BATCHES = 2U;
        constexpr uint32_t DST_HEIGHT = 56U;
        constexpr uint32_t DST_WIDTH = 56U;
        constexpr uint32_t ROWS_PER_STEP = 2U;
        constexpr uint32_t ROW_TILE_COUNT =
            (SRC_HEIGHT + ROWS_PER_STEP - 1U) / ROWS_PER_STEP;
        constexpr uint32_t CHANNEL_BLOCKS =
            (CHANNELS * sizeof(DT_X)) / BLOCK_BYTES;
        constexpr uint32_t SRC_ROW_BLOCKS =
            (SRC_WIDTH * CHANNELS * sizeof(DT_X)) / BLOCK_BYTES;
        constexpr uint32_t DST_ROW_BLOCKS =
            (DST_WIDTH * CHANNELS * sizeof(DT_X)) / BLOCK_BYTES;
        constexpr uint32_t INPUT_TILE_ELEMS =
            ROWS_PER_STEP * SRC_WIDTH * CHANNELS;
        constexpr uint32_t OUTPUT_TILE_ELEMS =
            ROWS_PER_STEP * DST_WIDTH * CHANNELS;
        constexpr uint32_t INPUT_TILE_BYTES =
            ((INPUT_TILE_ELEMS * sizeof(DT_X) + 511U) / 512U) * 512U;
        constexpr uint32_t TOTAL_GROUPS = DST_BATCHES * 2U * ROW_TILE_COUNT;

        AscendC::GlobalTensor<DT_X> xGmDirect;
        AscendC::GlobalTensor<DT_X> yGmDirect;
        xGmDirect.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGmDirect.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

        AscendC::LocalTensor<DT_X> leftTile(
            AscendC::TPosition::VECCALC, 0, INPUT_TILE_ELEMS);
        AscendC::LocalTensor<DT_X> rightTile(
            AscendC::TPosition::VECCALC, INPUT_TILE_BYTES, INPUT_TILE_ELEMS);
        AscendC::LocalTensor<DT_X> mergedRows(
            AscendC::TPosition::VECCALC, 2U * INPUT_TILE_BYTES,
            OUTPUT_TILE_ELEMS);

        AscendC::DataCopyParams loadParams;
        loadParams.blockCount = 1;
        loadParams.blockLen = 0;
        loadParams.srcStride = 0;
        loadParams.dstStride = 0;
        AscendC::DataCopyParams interleaveParams;
        interleaveParams.blockLen = static_cast<uint16_t>(CHANNEL_BLOCKS);
        interleaveParams.srcStride = 0;
        interleaveParams.dstStride = static_cast<uint16_t>(CHANNEL_BLOCKS);
        AscendC::DataCopyParams storeParams;
        storeParams.blockLen = static_cast<uint16_t>(DST_ROW_BLOCKS);
        storeParams.srcStride = 0;
        storeParams.dstStride = static_cast<uint16_t>(DST_ROW_BLOCKS);

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        for (uint32_t groupId = blockIdx; groupId < TOTAL_GROUPS;
             groupId += blockNum) {
            uint32_t rowTileId = groupId % ROW_TILE_COUNT;
            uint32_t batchPhase = groupId / ROW_TILE_COUNT;
            uint32_t blockRow = batchPhase & 1U;
            uint32_t outBatch = batchPhase >> 1U;
            uint32_t srcRow = rowTileId * ROWS_PER_STEP;
            uint32_t rowCount = SRC_HEIGHT - srcRow;
            if (rowCount > ROWS_PER_STEP) {
                rowCount = ROWS_PER_STEP;
            }

            uint32_t inputN0 = (blockRow << 1U) * DST_BATCHES + outBatch;
            uint32_t inputN1 = inputN0 + DST_BATCHES;
            uint32_t src0 =
                ((inputN0 * SRC_HEIGHT + srcRow) * SRC_WIDTH) * CHANNELS;
            uint32_t src1 =
                ((inputN1 * SRC_HEIGHT + srcRow) * SRC_WIDTH) * CHANNELS;
            loadParams.blockLen = static_cast<uint16_t>(rowCount * SRC_ROW_BLOCKS);
            AscendC::DataCopy(leftTile, xGmDirect[src0], loadParams);
            AscendC::DataCopy(rightTile, xGmDirect[src1], loadParams);
            AscendC::PipeBarrier<PIPE_ALL>();

            interleaveParams.blockCount =
                static_cast<uint16_t>(rowCount * SRC_WIDTH);
            AscendC::DataCopy(mergedRows, leftTile, interleaveParams);
            AscendC::DataCopy(mergedRows[CHANNELS], rightTile, interleaveParams);
            AscendC::PipeBarrier<PIPE_ALL>();

            storeParams.blockCount = static_cast<uint16_t>(rowCount);
            uint32_t dst =
                ((outBatch * DST_HEIGHT + blockRow + (srcRow << 1U)) *
                 DST_WIDTH) *
                CHANNELS;
            AscendC::DataCopy(yGmDirect[dst], mergedRows, storeParams);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
        return true;
    }
}

template <class DT_X>
__aicore__ inline bool RunFlatHalfSegmentT07(GM_ADDR x, GM_ADDR y) {
    if constexpr (sizeof(DT_X) != 2U) {
        return false;
    } else {
        constexpr uint32_t TOTAL_ELEMS = 4U * 16384U;
        constexpr uint32_t ELEMS_PER_JOB = 8U * 1024U;
        constexpr uint32_t TOTAL_JOBS = TOTAL_ELEMS / ELEMS_PER_JOB;

        AscendC::GlobalTensor<DT_X> xGmDirect;
        AscendC::GlobalTensor<DT_X> yGmDirect;
        xGmDirect.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGmDirect.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
        AscendC::LocalTensor<DT_X> segmentBuf(
            AscendC::TPosition::VECCALC, 0, ELEMS_PER_JOB);

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        for (uint32_t jobId = blockIdx; jobId < TOTAL_JOBS; jobId += blockNum) {
            uint32_t offset = jobId * ELEMS_PER_JOB;
            AscendC::DataCopy(segmentBuf, xGmDirect[offset], ELEMS_PER_JOB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(
                BTS_EVENT_MTE2_MTE3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(
                BTS_EVENT_MTE2_MTE3);
            AscendC::DataCopy(yGmDirect[offset], segmentBuf, ELEMS_PER_JOB);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
                BTS_EVENT_MTE3_MTE2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                BTS_EVENT_MTE3_MTE2);
        }
        return true;
    }
}

template <uint32_t CHANNELS, uint32_t LANE_ELEMS, uint32_t ELEM_BYTES>
__aicore__ inline void SeedAlternatingCropColumn(
    AscendC::LocalTensor<int32_t> offsetLocal, uint32_t offsetPos,
    uint32_t colRel) {
    uint32_t srcBase = (colRel & 1U) * LANE_ELEMS;
    uint32_t srcElem = srcBase + (colRel >> 1U) * CHANNELS;
    uint32_t byteOffset = srcElem * ELEM_BYTES;
    for (uint32_t c = 0U; c < CHANNELS; ++c) {
        offsetLocal.SetValue(offsetPos + c, static_cast<int32_t>(byteOffset));
        byteOffset += ELEM_BYTES;
    }
}

template <uint32_t DST_WIDTH, uint32_t CHANNELS, uint32_t LANE_ELEMS,
          uint32_t ELEM_BYTES>
__aicore__ inline void BuildAlternatingCropOffsets(
    AscendC::LocalTensor<int32_t> offsetLocal) {
    constexpr uint32_t GROUP_COLS = 8U;
    constexpr uint32_t GROUP_COUNT = DST_WIDTH / GROUP_COLS;
    constexpr uint32_t GROUP_ELEMS = GROUP_COLS * CHANNELS;
    constexpr uint32_t GROUP_SHIFT_BYTES =
        (GROUP_COLS >> 1U) * CHANNELS * ELEM_BYTES;

    for (uint32_t colRel = 0U; colRel < GROUP_COLS; ++colRel) {
        SeedAlternatingCropColumn<CHANNELS, LANE_ELEMS, ELEM_BYTES>(
            offsetLocal, colRel * CHANNELS, colRel);
    }
    constexpr uint32_t TAIL_COL_START = GROUP_COUNT * GROUP_COLS;
    for (uint32_t colRel = TAIL_COL_START; colRel < DST_WIDTH; ++colRel) {
        SeedAlternatingCropColumn<CHANNELS, LANE_ELEMS, ELEM_BYTES>(
            offsetLocal, colRel * CHANNELS, colRel);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(BTS_EVENT_S_V);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(BTS_EVENT_S_V);

    uint32_t generatedGroups = 1U;
    while (generatedGroups < GROUP_COUNT) {
        uint32_t copyGroups = MinU32(generatedGroups, GROUP_COUNT - generatedGroups);
        uint32_t dstPos = generatedGroups * GROUP_ELEMS;
        uint32_t addBytes = generatedGroups * GROUP_SHIFT_BYTES;
        uint32_t totalCopyCount = copyGroups * GROUP_ELEMS;
        for (uint32_t copied = 0U; copied < totalCopyCount;) {
            uint32_t curCount = MinU32(BTS_OFFSET_ADDS_CHUNK, totalCopyCount - copied);
            AscendC::LocalTensor<int32_t> addSrcLocal = offsetLocal[copied];
            AscendC::LocalTensor<int32_t> addDstLocal = offsetLocal[dstPos + copied];
            addSrcLocal.SetSize(curCount);
            addDstLocal.SetSize(curCount);
            AscendC::Adds(addDstLocal, addSrcLocal, static_cast<int32_t>(addBytes),
                          static_cast<int32_t>(curCount));
            copied += curCount;
        }
        generatedGroups += copyGroups;
        if (generatedGroups < GROUP_COUNT) {
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

template <class DT_X>
__aicore__ inline bool RunTinyOddCropGather(GM_ADDR x, GM_ADDR y) {
    if constexpr (sizeof(DT_X) != 4U) {
        return false;
    } else {
        constexpr uint32_t ELEM_BYTES = sizeof(DT_X);
        constexpr uint32_t SCRATCH_SLOT_BYTES = 1024U;
        constexpr uint32_t SRC_HEIGHT = 10U;
        constexpr uint32_t SRC_WIDTH = 15U;
        constexpr uint32_t CHANNELS = 5U;
        constexpr uint32_t DST_HEIGHT = 17U;
        constexpr uint32_t DST_WIDTH = 26U;
        constexpr uint32_t SRC_ROW_ELEMS = SRC_WIDTH * CHANNELS;
        constexpr uint32_t SRC_BATCH_ELEMS = SRC_HEIGHT * SRC_ROW_ELEMS;
        constexpr uint32_t DST_ROW_ELEMS = DST_WIDTH * CHANNELS;
        constexpr uint32_t WINDOW_COLS = 13U;
        constexpr uint32_t WINDOW_BYTES = WINDOW_COLS * CHANNELS * ELEM_BYTES;
        constexpr uint32_t WINDOW_STRIDE_BYTES =
            ((WINDOW_BYTES + 31U) / 32U) * 32U;
        constexpr uint32_t WINDOW_STRIDE_ELEMS =
            WINDOW_STRIDE_BYTES / ELEM_BYTES;
        constexpr uint32_t PACKED_ELEMS = WINDOW_STRIDE_ELEMS * 2U;
        constexpr uint32_t ROW_BUFFER_OFFSET_BYTES =
            BTS_ALIGN_UP_CONST(PACKED_ELEMS * ELEM_BYTES, SCRATCH_SLOT_BYTES);
        constexpr uint32_t INDEX_BUFFER_OFFSET_BYTES =
            ROW_BUFFER_OFFSET_BYTES +
            BTS_ALIGN_UP_CONST(DST_ROW_ELEMS * ELEM_BYTES, SCRATCH_SLOT_BYTES);

        AscendC::GlobalTensor<DT_X> xGmDirect;
        AscendC::GlobalTensor<DT_X> yGmDirect;
        xGmDirect.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGmDirect.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

        AscendC::LocalTensor<DT_X> packedLocal(
            AscendC::TPosition::VECCALC, 0, PACKED_ELEMS);
        AscendC::LocalTensor<DT_X> rowLocal(
            AscendC::TPosition::VECCALC, ROW_BUFFER_OFFSET_BYTES, DST_ROW_ELEMS);
        AscendC::LocalTensor<int32_t> offsetI32Local(
            AscendC::TPosition::VECCALC, INDEX_BUFFER_OFFSET_BYTES,
            DST_ROW_ELEMS);
        AscendC::LocalTensor<uint32_t> offsetLocal =
            offsetI32Local.ReinterpretCast<uint32_t>();
        packedLocal.SetSize(PACKED_ELEMS);
        rowLocal.SetSize(DST_ROW_ELEMS);
        offsetI32Local.SetSize(DST_ROW_ELEMS);
        offsetLocal.SetSize(DST_ROW_ELEMS);

        BuildAlternatingCropOffsets<DST_WIDTH, CHANNELS, WINDOW_STRIDE_ELEMS,
                                    ELEM_BYTES>(offsetI32Local);

        AscendC::DataCopyParams loadParams{
            static_cast<uint16_t>(1U),
            static_cast<uint16_t>(WINDOW_BYTES),
            static_cast<uint16_t>(0U),
            static_cast<uint16_t>(0U)};
        AscendC::DataCopyPadParams padParams{false, 0, 0, 0};
        AscendC::DataCopyParams storeParams{
            static_cast<uint16_t>(1U),
            static_cast<uint16_t>(DST_ROW_ELEMS * ELEM_BYTES),
            static_cast<uint16_t>(0U),
            static_cast<uint16_t>(0U)};

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        bool hasPendingStore = false;
        for (uint32_t outRow = blockIdx; outRow < DST_HEIGHT; outRow += blockNum) {
            uint32_t logicalRow = outRow + 2U;
            uint32_t inRow = logicalRow >> 1U;
            uint32_t rowPhase = logicalRow & 1U;
            uint32_t lane0Batch = (rowPhase << 1U) + 1U;
            uint32_t lane1Batch = rowPhase << 1U;
            uint32_t lane0Src =
                (lane0Batch * SRC_BATCH_ELEMS + inRow * SRC_ROW_ELEMS) +
                CHANNELS;
            uint32_t lane1Src =
                (lane1Batch * SRC_BATCH_ELEMS + inRow * SRC_ROW_ELEMS) +
                2U * CHANNELS;

            AscendC::DataCopyPad(packedLocal, xGmDirect[lane0Src],
                                 loadParams, padParams);
            AscendC::LocalTensor<DT_X> lane1Local =
                packedLocal[WINDOW_STRIDE_ELEMS];
            lane1Local.SetSize(WINDOW_STRIDE_ELEMS);
            AscendC::DataCopyPad(lane1Local, xGmDirect[lane1Src],
                                 loadParams, padParams);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(BTS_EVENT_MTE2_V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(BTS_EVENT_MTE2_V);
            if (hasPendingStore) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(BTS_EVENT_MTE3_V);
            }
            AscendC::Gather(rowLocal, packedLocal, offsetLocal, 0, DST_ROW_ELEMS);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(BTS_EVENT_V_MTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(BTS_EVENT_V_MTE3);
            AscendC::DataCopyPad(yGmDirect[outRow * DST_ROW_ELEMS],
                                 rowLocal, storeParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(BTS_EVENT_MTE3_V);
            hasPendingStore = true;
        }
        if (hasPendingStore) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(BTS_EVENT_MTE3_V);
        }
        return true;
    }
}

template <class DT_X, uint32_t SRC_HEIGHT, uint32_t SRC_WIDTH,
          uint32_t CHANNELS, uint32_t DST_BATCHES>
__aicore__ inline bool RunAlignedB2PlaneInterleave(GM_ADDR x, GM_ADDR y) {
    if constexpr (sizeof(DT_X) != 4U) {
        return false;
    } else {
        constexpr uint32_t DATA_BLOCK_BYTES = 32U;
        constexpr uint32_t STAGE_ALIGN_BYTES = 4U * 1024U;
        constexpr uint32_t ROW_PHASES = 2U;
        constexpr uint32_t DST_HEIGHT = SRC_HEIGHT * ROW_PHASES;
        constexpr uint32_t DST_WIDTH = SRC_WIDTH * ROW_PHASES;
        constexpr uint32_t GROUP_COUNT = DST_BATCHES * 2U;
        constexpr uint32_t CHANNEL_BLOCKS =
            (CHANNELS * sizeof(DT_X)) / DATA_BLOCK_BYTES;
        constexpr uint32_t INPUT_PLANE_BLOCKS =
            (SRC_HEIGHT * SRC_WIDTH * CHANNELS * sizeof(DT_X)) /
            DATA_BLOCK_BYTES;
        constexpr uint32_t OUTPUT_ROW_BLOCKS =
            (DST_WIDTH * CHANNELS * sizeof(DT_X)) / DATA_BLOCK_BYTES;
        constexpr uint32_t INPUT_PLANE_ELEMS = SRC_HEIGHT * SRC_WIDTH * CHANNELS;
        constexpr uint32_t OUTPUT_GROUP_ELEMS =
            SRC_HEIGHT * DST_WIDTH * CHANNELS;
        constexpr uint32_t PLANE_SLOT_BYTES =
            BTS_ALIGN_UP_CONST(INPUT_PLANE_ELEMS * sizeof(DT_X),
                               STAGE_ALIGN_BYTES);
        constexpr uint32_t RIGHT_STAGE_OFFSET_BYTES = PLANE_SLOT_BYTES;
        constexpr uint32_t MERGED_STAGE_OFFSET_BYTES = PLANE_SLOT_BYTES * 2U;

        AscendC::GlobalTensor<DT_X> xGmDirect;
        AscendC::GlobalTensor<DT_X> yGmDirect;
        xGmDirect.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGmDirect.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

        AscendC::LocalTensor<DT_X> leftPlane(
            AscendC::TPosition::VECCALC, 0, INPUT_PLANE_ELEMS);
        AscendC::LocalTensor<DT_X> rightPlane(
            AscendC::TPosition::VECCALC, RIGHT_STAGE_OFFSET_BYTES,
            INPUT_PLANE_ELEMS);
        AscendC::LocalTensor<DT_X> mergedPlane(
            AscendC::TPosition::VECCALC, MERGED_STAGE_OFFSET_BYTES,
            OUTPUT_GROUP_ELEMS);

        AscendC::DataCopyParams loadParams{
            static_cast<uint16_t>(1U),
            static_cast<uint16_t>(INPUT_PLANE_BLOCKS),
            static_cast<uint16_t>(0U),
            static_cast<uint16_t>(0U)};
        AscendC::DataCopyParams interleaveParams{
            static_cast<uint16_t>(SRC_HEIGHT * SRC_WIDTH),
            static_cast<uint16_t>(CHANNEL_BLOCKS),
            static_cast<uint16_t>(0U),
            static_cast<uint16_t>(CHANNEL_BLOCKS)};
        AscendC::DataCopyParams storeParams{
            static_cast<uint16_t>(SRC_HEIGHT),
            static_cast<uint16_t>(OUTPUT_ROW_BLOCKS),
            static_cast<uint16_t>(0U),
            static_cast<uint16_t>(OUTPUT_ROW_BLOCKS)};

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        for (uint32_t groupId = blockIdx; groupId < GROUP_COUNT;
            groupId += blockNum) {
            uint32_t blockRow = groupId & 1U;
            uint32_t outBatch = groupId >> 1U;
            uint32_t firstInputBatch =
                blockRow * ROW_PHASES * DST_BATCHES + outBatch;
            uint32_t secondInputBatch = firstInputBatch + DST_BATCHES;
            uint32_t src0 = firstInputBatch * INPUT_PLANE_ELEMS;
            uint32_t src1 = secondInputBatch * INPUT_PLANE_ELEMS;

            AscendC::DataCopy(leftPlane, xGmDirect[src0], loadParams);
            AscendC::DataCopy(rightPlane, xGmDirect[src1], loadParams);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::DataCopy(mergedPlane, leftPlane, interleaveParams);
            AscendC::DataCopy(mergedPlane[CHANNELS], rightPlane,
                              interleaveParams);
            AscendC::PipeBarrier<PIPE_ALL>();

            uint32_t dst =
                (outBatch * DST_HEIGHT + blockRow) * DST_WIDTH * CHANNELS;
            AscendC::DataCopy(yGmDirect[dst], mergedPlane, storeParams);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
        return true;
    }
}

template <class DT_X>
__aicore__ inline bool RunSmallFloatPlaneInterleave(GM_ADDR x, GM_ADDR y) {
    return RunAlignedB2PlaneInterleave<DT_X, 4U, 6U, 32U, 5U>(x, y);
}

template <class DT_X>
__aicore__ inline bool RunNarrowB2PhaseCase10(GM_ADDR x, GM_ADDR y) {
    if constexpr (sizeof(DT_X) != 2U) {
        return false;
    } else {
        constexpr uint32_t DATA_BLOCK_BYTES = 32U;
        constexpr uint32_t SRC_HEIGHT = 1024U;
        constexpr uint32_t SRC_WIDTH = 6U;
        constexpr uint32_t CHANNELS = 32U;
        constexpr uint32_t DST_BATCHES = 4U;
        constexpr uint32_t DST_HEIGHT = 2048U;
        constexpr uint32_t DST_WIDTH = 12U;
        constexpr uint32_t JOBS_PER_BATCH = 10U;
        constexpr uint32_t TOTAL_JOBS = DST_BATCHES * JOBS_PER_BATCH;
        constexpr uint32_t BASE_ROWS = SRC_HEIGHT / JOBS_PER_BATCH;
        constexpr uint32_t EXTRA_ROW_JOBS =
            SRC_HEIGHT - BASE_ROWS * JOBS_PER_BATCH;
        constexpr uint32_t MAX_ROWS = BASE_ROWS + 1U;
        constexpr uint32_t CHANNEL_BLOCKS =
            (CHANNELS * sizeof(DT_X)) / DATA_BLOCK_BYTES;
        constexpr uint32_t DST_ROW_BLOCKS =
            (DST_WIDTH * CHANNELS * sizeof(DT_X)) / DATA_BLOCK_BYTES;
        constexpr uint32_t MAX_PHASE_ELEMS =
            MAX_ROWS * DST_WIDTH * CHANNELS;
        constexpr uint32_t MAX_OUTPUT_ELEMS = MAX_PHASE_ELEMS * 2U;

        AscendC::GlobalTensor<DT_X> xGmDirect;
        AscendC::GlobalTensor<DT_X> yGmDirect;
        xGmDirect.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGmDirect.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

        AscendC::LocalTensor<DT_X> outputLocal(
            AscendC::TPosition::VECCALC, 0, MAX_OUTPUT_ELEMS);

        AscendC::DataCopyParams loadParams{
            static_cast<uint16_t>(0U),
            static_cast<uint16_t>(CHANNEL_BLOCKS),
            static_cast<uint16_t>(0U),
            static_cast<uint16_t>(CHANNEL_BLOCKS),
        };
        AscendC::DataCopyParams storeParams{
            static_cast<uint16_t>(0U),
            static_cast<uint16_t>(DST_ROW_BLOCKS),
            static_cast<uint16_t>(0U),
            static_cast<uint16_t>(DST_ROW_BLOCKS)};

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        for (uint32_t jobId = blockIdx; jobId < TOTAL_JOBS; jobId += blockNum) {
            uint32_t outBatch = jobId / JOBS_PER_BATCH;
            uint32_t jobInBatch = jobId - outBatch * JOBS_PER_BATCH;
            uint32_t rows = BASE_ROWS;
            uint32_t inHStart;
            if (jobInBatch < EXTRA_ROW_JOBS) {
                ++rows;
                inHStart = jobInBatch * MAX_ROWS;
            } else {
                inHStart = EXTRA_ROW_JOBS * MAX_ROWS +
                           (jobInBatch - EXTRA_ROW_JOBS) * BASE_ROWS;
            }

            uint32_t phaseElems = rows * DST_WIDTH * CHANNELS;
            loadParams.blockCount =
                static_cast<uint16_t>(rows * SRC_WIDTH);
            for (uint32_t blockRow = 0U; blockRow < 2U; ++blockRow) {
                uint32_t inputBase = (blockRow << 1U) * DST_BATCHES + outBatch;
                for (uint32_t blockCol = 0U; blockCol < 2U; ++blockCol) {
                    uint32_t srcBatch = inputBase + blockCol * DST_BATCHES;
                    uint32_t srcOffset =
                        ((srcBatch * SRC_HEIGHT + inHStart) * SRC_WIDTH) *
                        CHANNELS;
                    uint32_t localOffset =
                        blockRow * phaseElems + blockCol * CHANNELS;
                    AscendC::DataCopy(outputLocal[localOffset],
                                      xGmDirect[srcOffset], loadParams);
                }
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(
                BTS_EVENT_MTE2_MTE3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(
                BTS_EVENT_MTE2_MTE3);
            storeParams.blockCount = static_cast<uint16_t>(rows);
            for (uint32_t blockRow = 0U; blockRow < 2U; ++blockRow) {
                uint32_t outRow = (inHStart << 1U) + blockRow;
                uint32_t dstOffset =
                    ((outBatch * DST_HEIGHT + outRow) * DST_WIDTH) * CHANNELS;
                AscendC::DataCopy(yGmDirect[dstOffset],
                                  outputLocal[blockRow * phaseElems],
                                  storeParams);
            }
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
        return true;
    }
}

template <class DT_X>
__aicore__ inline bool RunFixedB4PhaseT09(GM_ADDR x, GM_ADDR y) {
    if constexpr (sizeof(DT_X) != 2U) {
        return false;
    } else {
        constexpr uint32_t PHASE_COUNT = 4U;
        constexpr uint32_t DATA_BLOCK_BYTES = 32U;
        constexpr uint32_t SRC_HEIGHT = 10U;
        constexpr uint32_t SRC_WIDTH = 512U;
        constexpr uint32_t CHANNELS = 64U;
        constexpr uint32_t DST_HEIGHT = 40U;
        constexpr uint32_t DST_WIDTH = 1535U;
        constexpr uint32_t LEFT_SKIP = 513U;
        constexpr uint32_t COLS_PER_TILE = 512U;
        constexpr uint32_t TILES_PER_ROW = 3U;
        constexpr uint32_t TOTAL_JOBS = DST_HEIGHT * TILES_PER_ROW;
        constexpr uint32_t CHANNEL_BLOCKS =
            (CHANNELS * sizeof(DT_X)) / DATA_BLOCK_BYTES;
        constexpr uint32_t PHASE_TILE_COLS =
            (COLS_PER_TILE + PHASE_COUNT - 1U) >> 2U;
        constexpr uint32_t PHASE_TILE_ELEMS = PHASE_TILE_COLS * CHANNELS;
        constexpr uint32_t OUTPUT_TILE_ELEMS = COLS_PER_TILE * CHANNELS;
        constexpr uint32_t OUTPUT_TILE_BYTES = OUTPUT_TILE_ELEMS * sizeof(DT_X);
        constexpr uint32_t PHASE_TILE_BYTES = PHASE_TILE_ELEMS * sizeof(DT_X);

        AscendC::GlobalTensor<DT_X> xGmDirect;
        AscendC::GlobalTensor<DT_X> yGmDirect;
        xGmDirect.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGmDirect.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

        AscendC::LocalTensor<DT_X> outputLocal(
            AscendC::TPosition::VECCALC, 0, OUTPUT_TILE_ELEMS);
        AscendC::LocalTensor<DT_X> input01(
            AscendC::TPosition::VECCALC, OUTPUT_TILE_BYTES,
            PHASE_TILE_ELEMS * 2U);
        AscendC::LocalTensor<DT_X> input23(
            AscendC::TPosition::VECCALC,
            OUTPUT_TILE_BYTES + PHASE_TILE_BYTES * 2U,
            PHASE_TILE_ELEMS * 2U);

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        AscendC::DataCopyParams loadParams{
            static_cast<uint16_t>(1U),
            static_cast<uint16_t>(0U),
            static_cast<uint16_t>(0U),
            static_cast<uint16_t>(0U)};
        AscendC::DataCopyParams scatterParams{
            static_cast<uint16_t>(0U),
            static_cast<uint16_t>(CHANNEL_BLOCKS),
            static_cast<uint16_t>(0U),
            static_cast<uint16_t>((PHASE_COUNT - 1U) * CHANNEL_BLOCKS)};
        AscendC::DataCopyParams storeParams{
            static_cast<uint16_t>(1U),
            static_cast<uint16_t>(0U),
            static_cast<uint16_t>(0U),
            static_cast<uint16_t>(0U)};

        for (uint32_t taskId = blockIdx; taskId < TOTAL_JOBS; taskId += blockNum) {
            uint32_t outRow = taskId / TILES_PER_ROW;
            uint32_t widthTileId = taskId - outRow * TILES_PER_ROW;
            uint32_t tileOw = widthTileId * COLS_PER_TILE;
            uint32_t cols = DST_WIDTH - tileOw;
            if (cols > COLS_PER_TILE) {
                cols = COLS_PER_TILE;
            }
            uint32_t inH = outRow >> 2U;
            uint32_t blockRow = outRow & 3U;
            uint32_t inputBase = blockRow << 2U;
            uint32_t cropResidue = LEFT_SKIP & 3U;

            for (uint32_t phase = 0U; phase < PHASE_COUNT; ++phase) {
                uint32_t firstOutCol =
                    (phase >= cropResidue) ? (phase - cropResidue)
                                           : (phase + PHASE_COUNT - cropResidue);
                if (firstOutCol >= cols) {
                    continue;
                }
                uint32_t count = ((cols - 1U - firstOutCol) >> 2U) + 1U;
                uint32_t srcCol = (tileOw + firstOutCol + LEFT_SKIP) >> 2U;
                uint32_t inputN = inputBase + phase;
                uint32_t src =
                    ((inputN * SRC_HEIGHT + inH) * SRC_WIDTH + srcCol) * CHANNELS;
                AscendC::LocalTensor<DT_X> phaseLocal =
                    (phase < 2U) ? input01[phase * PHASE_TILE_ELEMS]
                                 : input23[(phase - 2U) * PHASE_TILE_ELEMS];
                loadParams.blockLen =
                    static_cast<uint16_t>(count * CHANNEL_BLOCKS);
                AscendC::DataCopy(phaseLocal, xGmDirect[src], loadParams);
            }

            AscendC::PipeBarrier<PIPE_ALL>();
            for (uint32_t phase = 0U; phase < PHASE_COUNT; ++phase) {
                uint32_t firstOutCol =
                    (phase >= cropResidue) ? (phase - cropResidue)
                                           : (phase + PHASE_COUNT - cropResidue);
                if (firstOutCol >= cols) {
                    continue;
                }
                uint32_t count = ((cols - 1U - firstOutCol) >> 2U) + 1U;
                AscendC::LocalTensor<DT_X> phaseLocal =
                    (phase < 2U) ? input01[phase * PHASE_TILE_ELEMS]
                                 : input23[(phase - 2U) * PHASE_TILE_ELEMS];
                scatterParams.blockCount = static_cast<uint16_t>(count);
                AscendC::DataCopy(outputLocal[firstOutCol * CHANNELS],
                                  phaseLocal, scatterParams);
            }

            AscendC::PipeBarrier<PIPE_ALL>();
            uint32_t dst = (outRow * DST_WIDTH + tileOw) * CHANNELS;
            storeParams.blockLen = static_cast<uint16_t>(cols * CHANNEL_BLOCKS);
            AscendC::DataCopy(yGmDirect[dst], outputLocal, storeParams);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
        return true;
    }
}

template <class DT_X>
__aicore__ inline bool RunFloatC64Case03(GM_ADDR x, GM_ADDR y) {
    if constexpr (sizeof(DT_X) != 4U) {
        return false;
    } else {
        constexpr uint32_t DATA_BLOCK_BYTES = 32U;
        constexpr uint32_t SRC_ROWS = 14U;
        constexpr uint32_t SRC_COLS = 14U;
        constexpr uint32_t CHANNELS = 64U;
        constexpr uint32_t DST_BATCHES = 4U;
        constexpr uint32_t DST_ROWS = 28U;
        constexpr uint32_t DST_COLS = 28U;
        constexpr uint32_t ROWS_PER_TILE = 7U;
        constexpr uint32_t ROW_TILE_COUNT = 2U;
        constexpr uint32_t CHANNEL_BLOCKS =
            (CHANNELS * sizeof(DT_X)) / DATA_BLOCK_BYTES;
        constexpr uint32_t SRC_ROW_BLOCKS =
            (SRC_COLS * CHANNELS * sizeof(DT_X)) / DATA_BLOCK_BYTES;
        constexpr uint32_t DST_ROW_BLOCKS =
            (DST_COLS * CHANNELS * sizeof(DT_X)) / DATA_BLOCK_BYTES;
        constexpr uint32_t INPUT_TILE_ELEMS = ROWS_PER_TILE * SRC_COLS * CHANNELS;
        constexpr uint32_t OUTPUT_TILE_ELEMS = ROWS_PER_TILE * DST_COLS * CHANNELS;
        constexpr uint32_t INPUT_TILE_BYTES =
            ((INPUT_TILE_ELEMS * sizeof(DT_X) + 511U) / 512U) * 512U;
        constexpr uint32_t GROUP_COUNT = DST_BATCHES * 2U * ROW_TILE_COUNT;

        AscendC::GlobalTensor<DT_X> xGmDirect;
        AscendC::GlobalTensor<DT_X> yGmDirect;
        xGmDirect.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGmDirect.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

        AscendC::LocalTensor<DT_X> evenPhase(
            AscendC::TPosition::VECCALC, 0, INPUT_TILE_ELEMS);
        AscendC::LocalTensor<DT_X> oddPhase(
            AscendC::TPosition::VECCALC, INPUT_TILE_BYTES, INPUT_TILE_ELEMS);
        AscendC::LocalTensor<DT_X> rowPair(
            AscendC::TPosition::VECCALC, 2U * INPUT_TILE_BYTES,
            OUTPUT_TILE_ELEMS);

        AscendC::DataCopyParams loadParams{
            static_cast<uint16_t>(1U),
            static_cast<uint16_t>(0U),
            static_cast<uint16_t>(0U),
            static_cast<uint16_t>(0U)};
        AscendC::DataCopyParams interleaveParams{
            static_cast<uint16_t>(0U),
            static_cast<uint16_t>(CHANNEL_BLOCKS),
            static_cast<uint16_t>(0U),
            static_cast<uint16_t>(CHANNEL_BLOCKS)};
        AscendC::DataCopyParams storeParams{
            static_cast<uint16_t>(0U),
            static_cast<uint16_t>(DST_ROW_BLOCKS),
            static_cast<uint16_t>(0U),
            static_cast<uint16_t>(DST_ROW_BLOCKS)};

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        for (uint32_t groupId = blockIdx; groupId < GROUP_COUNT; groupId += blockNum) {
            uint32_t rowTileId = groupId % ROW_TILE_COUNT;
            uint32_t batchPhase = groupId / ROW_TILE_COUNT;
            uint32_t blockRow = batchPhase & 1U;
            uint32_t outBatch = batchPhase >> 1U;
            uint32_t srcRow = rowTileId * ROWS_PER_TILE;
            uint32_t rowCount = SRC_ROWS - srcRow;
            if (rowCount > ROWS_PER_TILE) {
                rowCount = ROWS_PER_TILE;
            }

            uint32_t inputN0 = (blockRow << 1U) * DST_BATCHES + outBatch;
            uint32_t inputN1 = inputN0 + DST_BATCHES;
            uint32_t src0 = ((inputN0 * SRC_ROWS + srcRow) * SRC_COLS) * CHANNELS;
            uint32_t src1 = ((inputN1 * SRC_ROWS + srcRow) * SRC_COLS) * CHANNELS;
            loadParams.blockLen = static_cast<uint16_t>(rowCount * SRC_ROW_BLOCKS);
            AscendC::DataCopy(evenPhase, xGmDirect[src0], loadParams);
            AscendC::DataCopy(oddPhase, xGmDirect[src1], loadParams);

            AscendC::PipeBarrier<PIPE_ALL>();
            interleaveParams.blockCount = static_cast<uint16_t>(rowCount * SRC_COLS);
            AscendC::DataCopy(rowPair, evenPhase, interleaveParams);
            AscendC::DataCopy(rowPair[CHANNELS], oddPhase, interleaveParams);

            AscendC::PipeBarrier<PIPE_ALL>();
            storeParams.blockCount = static_cast<uint16_t>(rowCount);
            uint32_t dst =
                ((outBatch * DST_ROWS + blockRow + srcRow * 2U) * DST_COLS) *
                CHANNELS;
            AscendC::DataCopy(yGmDirect[dst], rowPair, storeParams);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
        return true;
    }
}

template <class DT_X>
__aicore__ inline bool RunPixelPairCase06(GM_ADDR x, GM_ADDR y) {
    if constexpr (sizeof(DT_X) != 2U) {
        return false;
    } else {
        constexpr uint32_t DATA_BLOCK_BYTES = 32U;
        constexpr uint32_t SRC_HEIGHT = 2U;
        constexpr uint32_t SRC_WIDTH = 2U;
        constexpr uint32_t CHANNELS = 4096U;
        constexpr uint32_t DST_WIDTH = 4U;
        constexpr uint32_t TOTAL_PIXELS = 16U;
        constexpr uint32_t PIXELS_PER_JOB = 2U;
        constexpr uint32_t TOTAL_JOBS =
            (TOTAL_PIXELS + PIXELS_PER_JOB - 1U) / PIXELS_PER_JOB;
        constexpr uint32_t LOCAL_ELEMS = PIXELS_PER_JOB * CHANNELS;
        constexpr uint32_t CHANNEL_BLOCKS =
            (CHANNELS * sizeof(DT_X)) / DATA_BLOCK_BYTES;

        AscendC::GlobalTensor<DT_X> xGmDirect;
        AscendC::GlobalTensor<DT_X> yGmDirect;
        xGmDirect.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGmDirect.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
        AscendC::LocalTensor<DT_X> pixelBuf(
            AscendC::TPosition::VECCALC, 0, LOCAL_ELEMS);

        AscendC::DataCopyParams copyParams{
            static_cast<uint16_t>(1U),
            static_cast<uint16_t>(CHANNEL_BLOCKS),
            static_cast<uint16_t>(0U),
            static_cast<uint16_t>(0U)};
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        for (uint32_t jobId = blockIdx; jobId < TOTAL_JOBS; jobId += blockNum) {
            uint32_t firstPixel = jobId * PIXELS_PER_JOB;
            uint32_t pixelCount = TOTAL_PIXELS - firstPixel;
            if (pixelCount > PIXELS_PER_JOB) {
                pixelCount = PIXELS_PER_JOB;
            }
            uint32_t outRow = firstPixel / DST_WIDTH;
            uint32_t outCol = firstPixel - outRow * DST_WIDTH;
            for (uint32_t i = 0U; i < pixelCount; ++i) {
                uint32_t inRow = outRow >> 1U;
                uint32_t inCol = outCol >> 1U;
                uint32_t srcBatch = ((outRow & 1U) << 1U) + (outCol & 1U);
                uint32_t src =
                    ((srcBatch * SRC_HEIGHT + inRow) * SRC_WIDTH + inCol) * CHANNELS;
                AscendC::DataCopy(pixelBuf[i * CHANNELS], xGmDirect[src], copyParams);
                ++outCol;
                if (outCol == DST_WIDTH) {
                    outCol = 0U;
                    ++outRow;
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(BTS_EVENT_MTE2_MTE3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(BTS_EVENT_MTE2_MTE3);
            AscendC::DataCopy(yGmDirect[firstPixel * CHANNELS], pixelBuf,
                              pixelCount * CHANNELS);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(BTS_EVENT_MTE3_MTE2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(BTS_EVENT_MTE3_MTE2);
        }
        return true;
    }
}

__aicore__ inline void WriteT05OddCropOffsetColumn(
    AscendC::LocalTensor<int32_t> offsetLocal, uint32_t offsetPos,
    uint32_t colRel, uint32_t src1Base) {
    constexpr uint32_t CHANNELS = 65U;
    constexpr uint32_t ELEM_BYTES = 2U;
    uint32_t colHalf = colRel >> 1U;
    uint32_t localElemBase =
        ((colRel & 1U) == 0U) ? (src1Base + colHalf * CHANNELS)
                              : ((colHalf + 1U) * CHANNELS);
    uint32_t byteOffset = localElemBase * ELEM_BYTES;
    for (uint32_t d = 0U; d < CHANNELS; ++d) {
        offsetLocal.SetValue(offsetPos + d, static_cast<int32_t>(byteOffset));
        byteOffset += ELEM_BYTES;
    }
}

__aicore__ inline void FillT05OddCropOffsets(
    AscendC::LocalTensor<int32_t> offsetLocal) {
    constexpr uint32_t OUT_WIDTH = 254U;
    constexpr uint32_t CHANNELS = 65U;
    constexpr uint32_t GROUP_COLS = 8U;
    constexpr uint32_t SRC_WINDOW_ELEMS = 128U * CHANNELS;
    constexpr uint32_t GROUP_COUNT = OUT_WIDTH / GROUP_COLS;
    constexpr uint32_t GROUP_ELEMS = GROUP_COLS * CHANNELS;
    constexpr uint32_t GROUP_SHIFT_BYTES = (GROUP_COLS >> 1U) * CHANNELS * 2U;

    for (uint32_t colRel = 0U; colRel < GROUP_COLS; ++colRel) {
        WriteT05OddCropOffsetColumn(offsetLocal, colRel * CHANNELS, colRel,
                                    SRC_WINDOW_ELEMS);
    }
    constexpr uint32_t TAIL_COL_START = GROUP_COUNT * GROUP_COLS;
    for (uint32_t colRel = TAIL_COL_START; colRel < OUT_WIDTH; ++colRel) {
        WriteT05OddCropOffsetColumn(offsetLocal, colRel * CHANNELS, colRel,
                                    SRC_WINDOW_ELEMS);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(BTS_EVENT_S_V);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(BTS_EVENT_S_V);

    uint32_t generatedGroups = 1U;
    while (generatedGroups < GROUP_COUNT) {
        uint32_t copyGroups = MinU32(generatedGroups, GROUP_COUNT - generatedGroups);
        uint32_t dstPos = generatedGroups * GROUP_ELEMS;
        uint32_t addBytes = generatedGroups * GROUP_SHIFT_BYTES;
        uint32_t totalCopyCount = copyGroups * GROUP_ELEMS;
        for (uint32_t copied = 0U; copied < totalCopyCount;) {
            uint32_t curCount = MinU32(BTS_OFFSET_ADDS_CHUNK, totalCopyCount - copied);
            AscendC::LocalTensor<int32_t> addSrcLocal = offsetLocal[copied];
            AscendC::LocalTensor<int32_t> addDstLocal = offsetLocal[dstPos + copied];
            addSrcLocal.SetSize(curCount);
            addDstLocal.SetSize(curCount);
            AscendC::Adds(addDstLocal, addSrcLocal, static_cast<int32_t>(addBytes),
                          static_cast<int32_t>(curCount));
            copied += curCount;
        }
        generatedGroups += copyGroups;
        if (generatedGroups < GROUP_COUNT) {
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

template <class DT_X>
__aicore__ inline bool RunHalfOddCropGatherT05(GM_ADDR x, GM_ADDR y) {
    if constexpr (sizeof(DT_X) != 2U) {
        return false;
    } else {
        constexpr uint32_t SRC_HEIGHT = 128U;
        constexpr uint32_t SRC_WIDTH = 128U;
        constexpr uint32_t CHANNELS = 65U;
        constexpr uint32_t DST_HEIGHT = 254U;
        constexpr uint32_t DST_WIDTH = 254U;
        constexpr uint32_t INPUT_ROW_ELEMS = SRC_WIDTH * CHANNELS;
        constexpr uint32_t INPUT_BATCH_ELEMS = SRC_HEIGHT * INPUT_ROW_ELEMS;
        constexpr uint32_t OUTPUT_ROW_ELEMS = DST_WIDTH * CHANNELS;
        constexpr uint32_t INPUT_ELEMS = 4U * INPUT_BATCH_ELEMS;
        constexpr uint32_t OUTPUT_ELEMS = DST_HEIGHT * OUTPUT_ROW_ELEMS;
        constexpr uint32_t SOURCE_WINDOW_ELEMS = SRC_WIDTH * CHANNELS;
        constexpr uint32_t SOURCE_TOTAL_ELEMS = SOURCE_WINDOW_ELEMS * 2U;
        constexpr uint32_t SOURCE_TOTAL_BYTES = SOURCE_TOTAL_ELEMS * sizeof(DT_X);
        constexpr uint32_t OUTPUT_ROW_BYTES = OUTPUT_ROW_ELEMS * sizeof(DT_X);
        constexpr uint32_t OUTPUT_PAD_BYTES =
            ((OUTPUT_ROW_BYTES + 31U) / 32U) * 32U;
        constexpr uint32_t OUTPUT1_BYTE_START = SOURCE_TOTAL_BYTES + OUTPUT_PAD_BYTES;
        constexpr uint32_t OFFSET_BYTE_START = SOURCE_TOTAL_BYTES + 2U * OUTPUT_PAD_BYTES;

        AscendC::GlobalTensor<DT_X> xGmDirect;
        AscendC::GlobalTensor<DT_X> yGmDirect;
        xGmDirect.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), INPUT_ELEMS);
        yGmDirect.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), OUTPUT_ELEMS);

        AscendC::LocalTensor<DT_X> srcLocal(
            AscendC::TPosition::VECCALC, 0, SOURCE_TOTAL_ELEMS);
        AscendC::LocalTensor<DT_X> outLocal(
            AscendC::TPosition::VECCALC, SOURCE_TOTAL_BYTES, OUTPUT_ROW_ELEMS);
        AscendC::LocalTensor<DT_X> out1Local(
            AscendC::TPosition::VECCALC, OUTPUT1_BYTE_START, OUTPUT_ROW_ELEMS);
        AscendC::LocalTensor<int32_t> offsetI32Local(
            AscendC::TPosition::VECCALC, OFFSET_BYTE_START, OUTPUT_ROW_ELEMS);
        AscendC::LocalTensor<uint32_t> offsetLocal =
            offsetI32Local.ReinterpretCast<uint32_t>();

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t rowsPerCore = (DST_HEIGHT + blockNum - 1U) / blockNum;
        uint32_t startRow = blockIdx * rowsPerCore;
        if (startRow >= DST_HEIGHT) {
            return true;
        }
        uint32_t endRow = MinU32(startRow + rowsPerCore, DST_HEIGHT);

        offsetI32Local.SetSize(OUTPUT_ROW_ELEMS);
        FillT05OddCropOffsets(offsetI32Local);
        srcLocal.SetSize(SOURCE_TOTAL_ELEMS);
        offsetLocal.SetSize(OUTPUT_ROW_ELEMS);
        outLocal.SetSize(OUTPUT_ROW_ELEMS);
        out1Local.SetSize(OUTPUT_ROW_ELEMS);

        AscendC::DataCopyParams storeParams{
            static_cast<uint16_t>(1U),
            static_cast<uint16_t>(OUTPUT_ROW_BYTES),
            static_cast<uint16_t>(0U),
            static_cast<uint16_t>(0U)};

        uint32_t rowId = 0U;
        for (uint32_t outRow = startRow; outRow < endRow; ++outRow) {
            uint32_t outBufId = rowId & 1U;
            event_t reuseEvent =
                (outBufId == 0U) ? BTS_EVENT_MTE3_MTE2 : BTS_EVENT_MTE3_MTE2_DB;

            uint32_t fullOutRow = outRow + 1U;
            uint32_t blockRow = fullOutRow & 1U;
            uint32_t inH = fullOutRow >> 1U;
            uint32_t inputBatch0 = blockRow << 1U;
            uint32_t inputBatch1 = inputBatch0 + 1U;
            uint32_t src0 = inputBatch0 * INPUT_BATCH_ELEMS +
                            inH * INPUT_ROW_ELEMS;
            uint32_t src1 = inputBatch1 * INPUT_BATCH_ELEMS +
                            inH * INPUT_ROW_ELEMS;
            AscendC::LocalTensor<DT_X> src1Local = srcLocal[SOURCE_WINDOW_ELEMS];
            src1Local.SetSize(SOURCE_WINDOW_ELEMS);
            if (rowId > 0U) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(BTS_EVENT_V_MTE2);
            }
            AscendC::DataCopy(srcLocal, xGmDirect[src0], SOURCE_WINDOW_ELEMS);
            AscendC::DataCopy(src1Local, xGmDirect[src1], SOURCE_WINDOW_ELEMS);

            if (rowId >= 2U) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(reuseEvent);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(BTS_EVENT_MTE2_V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(BTS_EVENT_MTE2_V);
            AscendC::LocalTensor<DT_X> currentOut =
                (outBufId == 0U) ? outLocal : out1Local;
            AscendC::Gather(currentOut, srcLocal, offsetLocal, 0, OUTPUT_ROW_ELEMS);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(BTS_EVENT_V_MTE2);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(BTS_EVENT_V_MTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(BTS_EVENT_V_MTE3);
            uint32_t dst = outRow * OUTPUT_ROW_ELEMS;
            AscendC::DataCopyPad(yGmDirect[dst], currentOut, storeParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(reuseEvent);
            ++rowId;
        }
        if (rowId > 0U) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(BTS_EVENT_V_MTE2);
        }
        if (rowId == 1U) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(BTS_EVENT_MTE3_MTE2);
        } else if (rowId > 1U) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(BTS_EVENT_MTE3_MTE2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(BTS_EVENT_MTE3_MTE2_DB);
        }
        return true;
    }
}

template <class DT_X>
class KernelBatchToSpace {
public:
    __aicore__ inline KernelBatchToSpace() {}

    template <uint32_t SCH_MODE>
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                const BatchToSpaceTilingData &tiling,
                                AscendC::TPipe &pipe, uint32_t taskCount) {
        height = tiling.height;
        width = tiling.width;
        depth = tiling.depth;
        outBatch = tiling.outBatch;
        outHeight = tiling.outHeight;
        outWidth = tiling.outWidth;
        blockSize = tiling.blockSize;
        cropTop = tiling.cropTop;
        cropLeft = tiling.cropLeft;
        inputRowElems = width * depth;
        inputBatchElems = height * inputRowElems;
        outputRowElems = outWidth * depth;
        outputBatchElems = outHeight * outputRowElems;
        fullHeightLimit = outHeight + cropTop;
        fullWidthLimit = outWidth + cropLeft;
        uint32_t inputElements = outBatch * blockSize * blockSize * inputBatchElems;
        uint32_t outputElements = outBatch * outputBatchElems;
        totalTasks = taskCount;
        tasksPerCore = tiling.tasksPerCore;
        widthTiles = tiling.widthTiles;
        depthTiles = tiling.depthTiles;
        tilesPerInputRow = widthTiles * depthTiles;
        tileCols = tiling.tileCols;
        depthTileSize = tiling.depthTileSize;

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), inputElements);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), outputElements);
        uint32_t bufferBytes = BTS_TILE_BYTES;
        if constexpr (SCH_MODE == BTS_SCH_MODE_GATHER_B2_CROP) {
            bufferBytes = BTS_GATHER_B2_CROP_TILE_BYTES;
        } else if constexpr (SCH_MODE == BTS_SCH_MODE_DIRECT_B2_NOCROP) {
            if (totalTasks >= 64U) {
                bufferBytes = BTS_B2_CROP_TILE_BYTES * 2U;
            }
        }
        pipe.InitBuffer(stagingBuf, bufferBytes);
    }

    __aicore__ inline void ProcessGatherB2Crop(uint32_t startTask, uint32_t endTask) {
        bool offsetReady = false;
        uint32_t cachedWidthTileId = 0U;
        uint32_t rowTask = startTask / widthTiles;
        uint32_t widthTileId = startTask - rowTask * widthTiles;
        uint32_t outB = rowTask / outHeight;
        uint32_t outRow = rowTask - outB * outHeight;
        for (uint32_t taskId = startTask; taskId < endTask; ++taskId) {
            bool updateOffsets = (!offsetReady || widthTileId != cachedWidthTileId);
            ProcessGatherB2CropTask(outB, outRow, widthTileId, updateOffsets);
            cachedWidthTileId = widthTileId;
            offsetReady = true;
            ++widthTileId;
            if (widthTileId == widthTiles) {
                widthTileId = 0U;
                ++outRow;
                if (outRow == outHeight) {
                    outRow = 0U;
                    ++outB;
                }
            }
        }
    }

    __aicore__ inline void ProcessRows(uint32_t startTask, uint32_t endTask) {
        for (uint32_t taskId = startTask; taskId < endTask; ++taskId) {
            ProcessDirectTask(taskId);
        }
    }

    __aicore__ inline void ProcessRowsB2NoCrop(uint32_t startTask, uint32_t endTask) {
        if (totalTasks >= 64U) {
            if (outBatch == 1U) {
                ProcessRowsB2NoCropDb<1U, BTS_B2_CROP_TILE_BYTES>(startTask, endTask);
                return;
            } else if (outBatch == 2U) {
                ProcessRowsB2NoCropDb<2U, BTS_B2_CROP_TILE_BYTES>(startTask, endTask);
                return;
            } else if (outBatch == 4U) {
                ProcessRowsB2NoCropDb<4U, BTS_B2_CROP_TILE_BYTES>(startTask, endTask);
                return;
            }
        }

        if (outBatch == 1U) {
            ProcessRowsB2NoCropByOutBatch<1U>(startTask, endTask);
        } else if (outBatch == 2U) {
            ProcessRowsB2NoCropByOutBatch<2U>(startTask, endTask);
        } else if (outBatch == 4U) {
            ProcessRowsB2NoCropByOutBatch<4U>(startTask, endTask);
        } else {
            ProcessRowsB2NoCropByOutBatch<0U>(startTask, endTask);
        }
    }

private:
    struct DirectB2NoCropTaskInfo {
        uint32_t inputBatch;
        uint32_t inH;
        uint32_t outB;
        uint32_t outRow;
        uint32_t blockCol;
        uint32_t widthTileId;
        uint32_t depthTileId;
    };

    __aicore__ inline void SyncMte2ToMte3() {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(BTS_EVENT_MTE2_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(BTS_EVENT_MTE2_MTE3);
    }

    __aicore__ inline void SyncMte3ToMte2() {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(BTS_EVENT_MTE3_MTE2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(BTS_EVENT_MTE3_MTE2);
    }

    __aicore__ inline void SyncMte2ToV() {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(BTS_EVENT_MTE2_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(BTS_EVENT_MTE2_V);
    }

    __aicore__ inline void SyncVToMte3() {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(BTS_EVENT_V_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(BTS_EVENT_V_MTE3);
    }

    __aicore__ inline void SyncSToV() {
        AscendC::SetFlag<AscendC::HardEvent::S_V>(BTS_EVENT_S_V);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(BTS_EVENT_S_V);
    }

    template <uint32_t STATIC_OUT_BATCH>
    __aicore__ inline void ProcessRowsB2NoCropByOutBatch(uint32_t startTask,
                                                         uint32_t endTask) {
        for (uint32_t taskId = startTask; taskId < endTask; ++taskId) {
            ProcessDirectB2NoCropTask<STATIC_OUT_BATCH>(taskId);
        }
    }

    template <uint32_t STATIC_OUT_BATCH, uint32_t TILE_BYTES>
    __aicore__ inline void ProcessRowsB2NoCropDb(uint32_t startTask, uint32_t endTask) {
        AscendC::LocalTensor<DT_X> local = stagingBuf.Get<DT_X>();

        uint32_t prevBufId = 0U;
        uint32_t prevDstOffset = 0U;
        uint32_t prevCols = 0U;
        uint32_t prevDepthLen = 0U;
        uint32_t prevDstStrideBytes = 0U;
        uint32_t chunkId = 0U;

        for (uint32_t taskId = startTask; taskId < endTask; ++taskId) {
            uint32_t srcOffset = 0U;
            uint32_t dstOffset = 0U;
            uint32_t cols = 0U;
            uint32_t depthLen = 0U;
            uint32_t srcStrideBytes = 0U;
            uint32_t dstStrideBytes = 0U;
            BuildDirectB2NoCropCopy<STATIC_OUT_BATCH>(taskId, srcOffset, dstOffset, cols,
                                                      depthLen, srcStrideBytes,
                                                      dstStrideBytes);

            uint32_t bufId = chunkId & 1U;
            if (chunkId >= 2U) {
                WaitMte3ToMte2(bufId);
            }

            IssueB2CropCopyIn<TILE_BYTES>(local, bufId, srcOffset, cols, depthLen,
                                          srcStrideBytes);
            if (chunkId > 0U) {
                FlushB2CropBuffer<TILE_BYTES>(local, prevBufId, prevDstOffset, prevCols,
                                              prevDepthLen, prevDstStrideBytes);
            }

            prevBufId = bufId;
            prevDstOffset = dstOffset;
            prevCols = cols;
            prevDepthLen = depthLen;
            prevDstStrideBytes = dstStrideBytes;
            ++chunkId;
        }

        if (chunkId > 0U) {
            FlushB2CropBuffer<TILE_BYTES>(local, prevBufId, prevDstOffset, prevCols,
                                          prevDepthLen, prevDstStrideBytes);
        }
        if (chunkId == 1U) {
            WaitMte3ToMte2(0U);
        } else if (chunkId > 1U) {
            WaitMte3ToMte2(0U);
            WaitMte3ToMte2(1U);
        }
    }

    __aicore__ inline event_t Mte2ToMte3Event(uint32_t bufId) {
        return (bufId == 0U) ? BTS_EVENT_MTE2_MTE3 : BTS_EVENT_MTE2_MTE3_DB;
    }

    __aicore__ inline event_t Mte3ToMte2Event(uint32_t bufId) {
        return (bufId == 0U) ? BTS_EVENT_MTE3_MTE2 : BTS_EVENT_MTE3_MTE2_DB;
    }

    __aicore__ inline void SetMte2ToMte3(uint32_t bufId) {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(Mte2ToMte3Event(bufId));
    }

    __aicore__ inline void WaitMte2ToMte3(uint32_t bufId) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(Mte2ToMte3Event(bufId));
    }

    __aicore__ inline void SetMte3ToMte2(uint32_t bufId) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(Mte3ToMte2Event(bufId));
    }

    __aicore__ inline void WaitMte3ToMte2(uint32_t bufId) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(Mte3ToMte2Event(bufId));
    }

    __aicore__ inline void CopyBlocks(uint32_t srcOffset, uint32_t dstOffset, uint32_t cols,
                                      uint32_t depthLen, uint32_t srcStrideBytes,
                                      uint32_t dstStrideBytes) {
        AscendC::LocalTensor<DT_X> local = stagingBuf.Get<DT_X>();
        uint32_t blockLenBytes = depthLen * sizeof(DT_X);
        uint32_t elemsPerDataBlock = 32U / sizeof(DT_X);
        uint32_t alignedDepthLen = AlignUpU32(depthLen, elemsPerDataBlock);
        uint32_t copyElems = cols * depthLen;
        uint16_t blockCount = static_cast<uint16_t>(cols);
        uint16_t blockLen = static_cast<uint16_t>(blockLenBytes);
        uint16_t srcStride = static_cast<uint16_t>(srcStrideBytes);
        uint16_t dstStride = static_cast<uint16_t>(dstStrideBytes);

        AscendC::DataCopyPadParams padParams{false, 0, 0, 0};
        bool alignedContiguousBlock = (alignedDepthLen == depthLen);
        uint32_t alignMask = elemsPerDataBlock - 1U;
        bool srcGmAligned = ((srcOffset & alignMask) == 0U);
        bool dstGmAligned = ((dstOffset & alignMask) == 0U);
        if (alignedContiguousBlock && srcStrideBytes == 0U && srcGmAligned) {
            AscendC::DataCopy(local, xGm[srcOffset], copyElems);
        } else {
            AscendC::DataCopyParams copyInParams{blockCount, blockLen, srcStride, 0};
            AscendC::DataCopyPad(local, xGm[srcOffset], copyInParams, padParams);
        }
        SyncMte2ToMte3();

        if (alignedContiguousBlock && dstStrideBytes == 0U && dstGmAligned) {
            AscendC::DataCopy(yGm[dstOffset], local, copyElems);
        } else {
            AscendC::DataCopyParams copyOutParams{
                blockCount, blockLen, 0, dstStride};
            AscendC::DataCopyPad(yGm[dstOffset], local, copyOutParams);
        }
        SyncMte3ToMte2();
    }

    __aicore__ inline void CopyGmToLocal1D(AscendC::LocalTensor<DT_X> local,
                                           uint32_t gmOffset,
                                           uint32_t realCount) {
        uint32_t elemBytes = sizeof(DT_X);
        uint32_t elemsPerDataBlock = 32U / elemBytes;
        bool alignedCopy = ((realCount & (elemsPerDataBlock - 1U)) == 0U) &&
                           ((gmOffset & (elemsPerDataBlock - 1U)) == 0U);
        if (alignedCopy) {
            AscendC::DataCopy(local, xGm[gmOffset], realCount);
        } else {
            AscendC::DataCopyParams copyParams{
                static_cast<uint16_t>(1U),
                static_cast<uint16_t>(realCount * elemBytes),
                static_cast<uint16_t>(0U),
                static_cast<uint16_t>(0U)};
            AscendC::DataCopyPadParams padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(local, xGm[gmOffset], copyParams, padParams);
        }
    }

    __aicore__ inline void CopyLocalToGm1D(uint32_t gmOffset,
                                           AscendC::LocalTensor<DT_X> local,
                                           uint32_t realCount) {
        uint32_t elemBytes = sizeof(DT_X);
        uint32_t elemsPerDataBlock = 32U / elemBytes;
        bool alignedCopy = ((realCount & (elemsPerDataBlock - 1U)) == 0U) &&
                           ((gmOffset & (elemsPerDataBlock - 1U)) == 0U);
        if (alignedCopy) {
            AscendC::DataCopy(yGm[gmOffset], local, realCount);
        } else {
            AscendC::DataCopyParams copyParams{
                static_cast<uint16_t>(1U),
                static_cast<uint16_t>(realCount * elemBytes),
                static_cast<uint16_t>(0U),
                static_cast<uint16_t>(0U)};
            AscendC::DataCopyPad(yGm[gmOffset], local, copyParams);
        }
    }

    __aicore__ inline void BuildB2CropWindow(uint32_t outColStart, uint32_t outCols,
                                             uint32_t blockCol, uint32_t &count,
                                             uint32_t &firstInW) {
        uint32_t fullStart = outColStart + cropLeft;
        uint32_t firstDelta = ((fullStart & 1U) == blockCol) ? 0U : 1U;
        if (firstDelta >= outCols) {
            count = 0U;
            firstInW = 0U;
            return;
        }
        count = ((outCols - firstDelta) + 1U) >> 1U;
        firstInW = (fullStart + firstDelta) >> 1U;
    }

    __aicore__ inline void WriteGatherB2CropOffsetColumn(
        AscendC::LocalTensor<int32_t> offsetLocal, uint32_t offsetPos,
        uint32_t outColStart, uint32_t colRel, uint32_t firstInW0,
        uint32_t firstInW1, uint32_t srcBase0, uint32_t srcBase1) {
        uint32_t elemBytes = sizeof(DT_X);
        uint32_t fullOutCol = outColStart + colRel + cropLeft;
        uint32_t inW = fullOutCol >> 1U;
        uint32_t localIndexBase = ((fullOutCol & 1U) == 0U)
                                      ? srcBase0 + (inW - firstInW0) * depth
                                      : srcBase1 + (inW - firstInW1) * depth;
        uint32_t byteOffset = localIndexBase * elemBytes;
        // index is an element index in the packed UB source windows, while
        // Gather requires byte offsets: half index * 2, float index * 4.
        // Same-type gather-copy does not Cast, preserving bitwise identity.
        for (uint32_t d = 0U; d < depth; ++d) {
            offsetLocal.SetValue(offsetPos + d, static_cast<int32_t>(byteOffset));
            byteOffset += elemBytes;
        }
    }

    __aicore__ inline void FillGatherB2CropByteOffsets(
        AscendC::LocalTensor<int32_t> offsetLocal, uint32_t outColStart,
        uint32_t outCols, uint32_t firstInW0, uint32_t firstInW1,
        uint32_t srcBase0, uint32_t srcBase1) {
        uint32_t elemBytes = sizeof(DT_X);
        uint32_t groupCols = CalcGatherAlignedGroupCols(depth);
        uint32_t baseCols = MinU32(outCols, groupCols);
        uint32_t groupCount = outCols / groupCols;
        uint32_t groupElemCount = groupCols * depth;
        uint32_t groupOffsetBytes = (groupCols >> 1U) * depth * elemBytes;
        for (uint32_t colRel = 0U; colRel < baseCols; ++colRel) {
            WriteGatherB2CropOffsetColumn(offsetLocal, colRel * depth, outColStart, colRel,
                                          firstInW0, firstInW1, srcBase0, srcBase1);
        }
        if (groupCount == 0U) {
            SyncSToV();
            return;
        }

        uint32_t tailColStart = groupCount * groupCols;
        for (uint32_t colRel = tailColStart; colRel < outCols; ++colRel) {
            WriteGatherB2CropOffsetColumn(offsetLocal, colRel * depth, outColStart, colRel,
                                          firstInW0, firstInW1, srcBase0, srcBase1);
        }
        SyncSToV();

        // Use the shortest even-column period whose int32 offsets end on a 32B boundary.
        uint32_t generatedGroups = 1U;
        while (generatedGroups < groupCount) {
            uint32_t copyGroups = MinU32(generatedGroups, groupCount - generatedGroups);
            uint32_t dstPos = generatedGroups * groupElemCount;
            uint32_t addBytes = generatedGroups * groupOffsetBytes;
            uint32_t totalCopyCount = copyGroups * groupElemCount;
            for (uint32_t copied = 0U; copied < totalCopyCount;) {
                uint32_t curCount = MinU32(BTS_OFFSET_ADDS_CHUNK, totalCopyCount - copied);
                AscendC::LocalTensor<int32_t> addSrcLocal = offsetLocal[copied];
                AscendC::LocalTensor<int32_t> addDstLocal = offsetLocal[dstPos + copied];
                addSrcLocal.SetSize(curCount);
                addDstLocal.SetSize(curCount);
                AscendC::Adds(addDstLocal, addSrcLocal,
                              static_cast<int32_t>(addBytes),
                              static_cast<int32_t>(curCount));
                copied += curCount;
            }
            generatedGroups += copyGroups;
            if (generatedGroups < groupCount) {
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    template <uint32_t BLOCK_COL>
    __aicore__ inline void CopyGatherB2CropSourceWindow(
        AscendC::LocalTensor<DT_X> srcLocal, uint32_t srcBase, uint32_t count,
        uint32_t blockRow, uint32_t outB, uint32_t batchPlaneElems,
        uint32_t inputRowBase, uint32_t firstInW) {
        if (count == 0U) {
            return;
        }
        uint32_t inputBatch = ((blockRow << 1U) + BLOCK_COL) * outBatch + outB;
        uint32_t srcOffset = inputBatch * batchPlaneElems + inputRowBase + firstInW * depth;
        CopyGmToLocal1D(srcLocal[srcBase], srcOffset, count * depth);
    }

    __aicore__ inline void ProcessGatherB2CropTask(uint32_t outB,
                                                   uint32_t outRow,
                                                   uint32_t widthTileId,
                                                   bool updateOffsets) {
        uint32_t outColStart = widthTileId * tileCols;
        uint32_t outCols = MinU32(tileCols, outWidth - outColStart);
        if (outCols == 0U) {
            return;
        }

        uint32_t fullOutRow = outRow + cropTop;
        uint32_t blockRow = fullOutRow & 1U;
        uint32_t inH = fullOutRow >> 1U;
        uint32_t realCount = outCols * depth;

        if (outCols == 1U) {
            uint32_t fullOutCol = outColStart + cropLeft;
            uint32_t blockCol = fullOutCol & 1U;
            uint32_t inW = fullOutCol >> 1U;
            uint32_t inputBatch = ((blockRow << 1U) + blockCol) * outBatch + outB;
            uint32_t srcOffset = inputBatch * inputBatchElems + inH * inputRowElems +
                                 inW * depth;
            uint32_t dstOffset = outB * outputBatchElems + outRow * outputRowElems +
                                 outColStart * depth;
            CopyBlocks(srcOffset, dstOffset, 1U, depth, 0U, 0U);
            return;
        }

        uint32_t elemBytes = sizeof(DT_X);
        uint32_t count0 = 0U, count1 = 0U, firstInW0 = 0U, firstInW1 = 0U;
        bool t05WholeRow = false;
        if constexpr (sizeof(DT_X) == 2U) {
            t05WholeRow = (depth == 65U && outBatch == 1U && blockSize == 2U &&
                           height == 128U && width == 128U && outHeight == 254U &&
                           outWidth == 254U && cropTop == 1U && cropLeft == 1U &&
                           outColStart == 0U && outCols == 254U);
        }
        if (t05WholeRow) {
            count0 = 127U;
            count1 = 127U;
            firstInW0 = 1U;
            firstInW1 = 0U;
        } else {
            BuildB2CropWindow(outColStart, outCols, 0U, count0, firstInW0);
            BuildB2CropWindow(outColStart, outCols, 1U, count1, firstInW1);
        }

        uint32_t src0Bytes = AlignUpU32(count0 * depth * elemBytes, 32U);
        uint32_t src1Bytes = AlignUpU32(count1 * depth * elemBytes, 32U);
        uint32_t srcBase0 = 0U;
        uint32_t srcBase1 = src0Bytes / elemBytes;
        uint32_t outBase = (src0Bytes + src1Bytes) / elemBytes;
        uint32_t outBytes = AlignUpU32(realCount * elemBytes, 32U);
        uint32_t offsetByteStart = src0Bytes + src1Bytes + outBytes;
        uint32_t inputRowBase = inH * inputRowElems;
        uint32_t outputRowBase = outB * outputBatchElems + outRow * outputRowElems;

        AscendC::LocalTensor<DT_X> dataLocal = stagingBuf.Get<DT_X>();
        AscendC::LocalTensor<DT_X> srcLocal = dataLocal;
        AscendC::LocalTensor<DT_X> outLocal = dataLocal[outBase];
        AscendC::LocalTensor<int32_t> offsetI32Local =
            stagingBuf.Get<int32_t>()[offsetByteStart / sizeof(int32_t)];
        AscendC::LocalTensor<uint32_t> offsetLocal =
            offsetI32Local.ReinterpretCast<uint32_t>();

        CopyGatherB2CropSourceWindow<0U>(srcLocal, srcBase0, count0, blockRow,
                                         outB, inputBatchElems, inputRowBase, firstInW0);
        CopyGatherB2CropSourceWindow<1U>(srcLocal, srcBase1, count1, blockRow,
                                         outB, inputBatchElems, inputRowBase, firstInW1);

        // Gather uses realCount only; aligned padding never participates.
        if (updateOffsets) {
            offsetI32Local.SetSize(realCount);
            FillGatherB2CropByteOffsets(offsetI32Local, outColStart, outCols, firstInW0,
                                        firstInW1, srcBase0, srcBase1);
        }
        SyncMte2ToV();
        // Same-type gather-copy only; no Cast in FP16->FP16 or FP32->FP32.
        srcLocal.SetSize(outBase);
        offsetLocal.SetSize(realCount);
        outLocal.SetSize(realCount);
        AscendC::Gather(outLocal, srcLocal, offsetLocal, 0, realCount);
        SyncVToMte3();
        uint32_t dstOffset = outputRowBase + outColStart * depth;
        CopyLocalToGm1D(dstOffset, outLocal, realCount);
        SyncMte3ToMte2();
    }

    __aicore__ inline void ProcessDirectTask(uint32_t taskId) {
        uint32_t inputRowId = taskId / tilesPerInputRow;
        uint32_t tileId = taskId - inputRowId * tilesPerInputRow;
        uint32_t widthTileId = tileId / depthTiles;
        uint32_t depthTileId = tileId - widthTileId * depthTiles;

        uint32_t inputBatch = inputRowId / height;
        uint32_t inH = inputRowId - inputBatch * height;
        uint32_t blockIndex = inputBatch / outBatch;
        uint32_t outB = inputBatch - blockIndex * outBatch;
        uint32_t blockRow = blockIndex / blockSize;
        uint32_t blockCol = blockIndex - blockRow * blockSize;

        uint32_t fullOutRow = inH * blockSize + blockRow;
        if (fullOutRow < cropTop || fullOutRow >= fullHeightLimit) {
            return;
        }
        uint32_t outRow = fullOutRow - cropTop;

        uint32_t dStart = depthTileId * depthTileSize;
        uint32_t depthLen = MinU32(depthTileSize, depth - dStart);

        uint32_t wTileStart = widthTileId * tileCols;
        uint32_t wTileEnd = MinU32(wTileStart + tileCols, width);

        uint32_t minValidW = 0U;
        if (cropLeft > blockCol) {
            minValidW = (cropLeft - blockCol + blockSize - 1U) / blockSize;
        }

        uint32_t maxValidW = 0U;
        if (fullWidthLimit > blockCol) {
            maxValidW = (fullWidthLimit - 1U - blockCol) / blockSize + 1U;
            maxValidW = MinU32(maxValidW, width);
        }

        uint32_t validStart = (wTileStart > minValidW) ? wTileStart : minValidW;
        uint32_t validEnd = MinU32(wTileEnd, maxValidW);
        if (validStart >= validEnd) {
            return;
        }
        CopyDirectRange<0U>(inputBatch, inH, outB, outRow, blockCol, dStart,
                            depthLen, validStart, validEnd);
    }

    template <uint32_t STATIC_OUT_BATCH>
    __aicore__ inline void ProcessDirectB2NoCropTask(uint32_t taskId) {
        DirectB2NoCropTaskInfo info;
        BuildDirectB2NoCropTaskInfo<STATIC_OUT_BATCH>(taskId, info);

        uint32_t dStart = info.depthTileId * depthTileSize;
        uint32_t depthLen = MinU32(depthTileSize, depth - dStart);

        uint32_t validStart = info.widthTileId * tileCols;
        uint32_t validEnd = MinU32(validStart + tileCols, width);
        CopyDirectRange<2U>(info.inputBatch, info.inH, info.outB, info.outRow,
                            info.blockCol, dStart, depthLen, validStart, validEnd);
    }

    template <uint32_t STATIC_OUT_BATCH>
    __aicore__ inline void BuildDirectB2NoCropCopy(uint32_t taskId, uint32_t &srcOffset,
                                                   uint32_t &dstOffset, uint32_t &cols,
                                                   uint32_t &depthLen,
                                                   uint32_t &srcStrideBytes,
                                                   uint32_t &dstStrideBytes) {
        DirectB2NoCropTaskInfo info;
        BuildDirectB2NoCropTaskInfo<STATIC_OUT_BATCH>(taskId, info);

        uint32_t dStart = info.depthTileId * depthTileSize;
        depthLen = MinU32(depthTileSize, depth - dStart);
        uint32_t validStart = info.widthTileId * tileCols;
        uint32_t validEnd = MinU32(validStart + tileCols, width);
        cols = validEnd - validStart;

        uint32_t elemBytes = sizeof(DT_X);
        srcStrideBytes = (depth - depthLen) * elemBytes;
        dstStrideBytes = ((depth << 1U) - depthLen) * elemBytes;
        srcOffset = info.inputBatch * inputBatchElems + info.inH * inputRowElems +
                    validStart * depth + dStart;
        uint32_t outCol = (validStart << 1U) + info.blockCol;
        dstOffset = info.outB * outputBatchElems + info.outRow * outputRowElems +
                    outCol * depth + dStart;
    }

    template <uint32_t STATIC_OUT_BATCH>
    __aicore__ inline void BuildDirectB2NoCropTaskInfo(uint32_t taskId,
                                                       DirectB2NoCropTaskInfo &info) {
        uint32_t inputRowId = taskId / tilesPerInputRow;
        uint32_t tileId = taskId - inputRowId * tilesPerInputRow;
        info.widthTileId = tileId / depthTiles;
        info.depthTileId = tileId - info.widthTileId * depthTiles;

        info.inputBatch = inputRowId / height;
        info.inH = inputRowId - info.inputBatch * height;
        uint32_t blockIndex;
        if constexpr (STATIC_OUT_BATCH == 1U) {
            blockIndex = info.inputBatch;
            info.outB = 0U;
        } else if constexpr (STATIC_OUT_BATCH == 2U) {
            blockIndex = info.inputBatch >> 1U;
            info.outB = info.inputBatch & 1U;
        } else if constexpr (STATIC_OUT_BATCH == 4U) {
            blockIndex = info.inputBatch >> 2U;
            info.outB = info.inputBatch & 3U;
        } else {
            blockIndex = info.inputBatch / outBatch;
            info.outB = info.inputBatch - blockIndex * outBatch;
        }
        uint32_t blockRow = blockIndex >> 1U;
        info.blockCol = blockIndex & 1U;
        info.outRow = (info.inH << 1U) + blockRow;
    }

    template <uint32_t STATIC_BLOCK_SIZE>
    __aicore__ inline void CopyDirectRange(uint32_t inputBatch, uint32_t inH,
                                           uint32_t outB, uint32_t outRow,
                                           uint32_t blockCol, uint32_t dStart,
                                           uint32_t depthLen, uint32_t validStart,
                                           uint32_t validEnd) {
        uint32_t elemBytes = sizeof(DT_X);
        uint32_t tileElems = BTS_TILE_BYTES / elemBytes;
        uint32_t elemsPerDataBlock = 32U / elemBytes;
        uint32_t alignedDepthLen = AlignUpU32(depthLen, elemsPerDataBlock);
        uint32_t maxColsPerCopy = tileElems / alignedDepthLen;

        uint32_t outBlockSize = STATIC_BLOCK_SIZE;
        if constexpr (STATIC_BLOCK_SIZE == 0U) {
            outBlockSize = blockSize;
        }
        uint32_t srcStrideBytes = (depth - depthLen) * elemBytes;
        uint32_t dstStrideBytes = (outBlockSize * depth - depthLen) * elemBytes;
        uint32_t inputRowBase = inputBatch * inputBatchElems + inH * inputRowElems + dStart;
        uint32_t outputRowBase = outB * outputBatchElems + outRow * outputRowElems + dStart;
        uint32_t curSrcOffset = inputRowBase + validStart * depth;
        uint32_t curDstOffset =
            outputRowBase + (validStart * outBlockSize + blockCol - cropLeft) * depth;
        uint32_t dstColStep = outBlockSize * depth;
        bool singleColCopy = (srcStrideBytes > BTS_MAX_COPY_PARAM ||
                              dstStrideBytes > BTS_MAX_COPY_PARAM);
        uint32_t copySrcStride = singleColCopy ? 0U : srcStrideBytes;
        uint32_t copyDstStride = singleColCopy ? 0U : dstStrideBytes;
        for (uint32_t inW = validStart; inW < validEnd;) {
            uint32_t cols = MinU32(maxColsPerCopy, validEnd - inW);
            if (cols > 1U && singleColCopy) {
                cols = 1U;
            }

            CopyBlocks(curSrcOffset, curDstOffset, cols, depthLen, copySrcStride,
                       copyDstStride);
            inW += cols;
            curSrcOffset += cols * depth;
            curDstOffset += cols * dstColStep;
        }
    }

    template <uint32_t TILE_BYTES>
    __aicore__ inline void FlushB2CropBuffer(AscendC::LocalTensor<DT_X> &local,
                                             uint32_t bufId, uint32_t dstOffset,
                                             uint32_t cols, uint32_t depthLen,
                                             uint32_t dstStrideBytes) {
        uint32_t elemBytes = sizeof(DT_X);
        uint32_t tileElems = TILE_BYTES / elemBytes;
        uint32_t bufOffset = bufId * tileElems;

        WaitMte2ToMte3(bufId);
        AscendC::DataCopyParams copyOutParams{
            static_cast<uint16_t>(cols),
            static_cast<uint16_t>(depthLen * elemBytes),
            0,
            static_cast<uint16_t>(dstStrideBytes)};
        AscendC::DataCopyPad(yGm[dstOffset], local[bufOffset], copyOutParams);
        SetMte3ToMte2(bufId);
    }

    __aicore__ inline void IssueB2CropCopyInAt(AscendC::LocalTensor<DT_X> &local,
                                               uint32_t localOffset,
                                               uint32_t srcOffset,
                                               uint32_t cols, uint32_t depthLen,
                                               uint32_t srcStrideBytes) {
        uint32_t elemBytes = sizeof(DT_X);
        AscendC::DataCopyParams copyInParams{
            static_cast<uint16_t>(cols),
            static_cast<uint16_t>(depthLen * elemBytes),
            static_cast<uint16_t>(srcStrideBytes),
            0};
        AscendC::DataCopyPadParams padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(local[localOffset], xGm[srcOffset], copyInParams, padParams);
    }

    template <uint32_t TILE_BYTES>
    __aicore__ inline void IssueB2CropCopyIn(AscendC::LocalTensor<DT_X> &local,
                                             uint32_t bufId, uint32_t srcOffset,
                                             uint32_t cols, uint32_t depthLen,
                                             uint32_t srcStrideBytes) {
        uint32_t elemBytes = sizeof(DT_X);
        uint32_t tileElems = TILE_BYTES / elemBytes;
        uint32_t bufOffset = bufId * tileElems;

        IssueB2CropCopyInAt(local, bufOffset, srcOffset, cols, depthLen, srcStrideBytes);
        SetMte2ToMte3(bufId);
    }

private:
    uint32_t height;
    uint32_t width;
    uint32_t depth;
    uint32_t outBatch;
    uint32_t outHeight;
    uint32_t outWidth;
    uint32_t blockSize;
    uint32_t cropTop;
    uint32_t cropLeft;
    uint32_t inputRowElems;
    uint32_t inputBatchElems;
    uint32_t outputRowElems;
    uint32_t outputBatchElems;
    uint32_t fullHeightLimit;
    uint32_t fullWidthLimit;
    uint32_t totalTasks;
    uint32_t tasksPerCore;
    uint32_t widthTiles;
    uint32_t depthTiles;
    uint32_t tilesPerInputRow;
    uint32_t tileCols;
    uint32_t depthTileSize;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stagingBuf;
};

template <typename DT_X, uint32_t schMode>
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);

    if constexpr (schMode == BTS_SCH_MODE_FIXED_B4_CROP_T09) {
        (void)RunFixedB4PhaseT09<DT_X>(x, y);
    } else if constexpr (schMode == BTS_SCH_MODE_TINY_F32_ODD_CROP) {
        (void)RunTinyOddCropGather<DT_X>(x, y);
    } else if constexpr (schMode == BTS_SCH_MODE_FIXED_B2_PHASE_T10) {
        (void)RunNarrowB2PhaseCase10<DT_X>(x, y);
    } else if constexpr (schMode == BTS_SCH_MODE_FIXED_F32_C64) {
        (void)RunFloatC64Case03<DT_X>(x, y);
    } else if constexpr (schMode == BTS_SCH_MODE_FIXED_HALF_C4096) {
        (void)RunPixelPairCase06<DT_X>(x, y);
    } else if constexpr (schMode == BTS_SCH_MODE_FIXED_F32_C128_T01) {
        (void)RunWideFloatRowPairT01<DT_X>(x, y);
    } else if constexpr (schMode == BTS_SCH_MODE_F32_PLANE_INTERLEAVE) {
        (void)RunSmallFloatPlaneInterleave<DT_X>(x, y);
    } else if constexpr (schMode == BTS_SCH_MODE_FIXED_HALF_FLAT_T07) {
        (void)RunFlatHalfSegmentT07<DT_X>(x, y);
    } else if constexpr (schMode == BTS_SCH_MODE_FIXED_HALF_CROP_T05) {
        (void)RunHalfOddCropGatherT05<DT_X>(x, y);
    } else {
        GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingData, tilingData, tiling);

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t tasksPerCore = tilingData.tasksPerCore;
        uint32_t startTask = blockIdx * tasksPerCore;
        uint32_t taskCount = CalcTotalTasks<schMode>(tilingData);
        if (startTask >= taskCount) {
            return;
        }
        uint32_t endTask = MinU32(startTask + tasksPerCore, taskCount);

        AscendC::TPipe pipe;
        KernelBatchToSpace<DT_X> op;
        op.template Init<schMode>(x, y, tilingData, pipe, taskCount);
        if constexpr (schMode == BTS_SCH_MODE_GATHER_B2_CROP) {
            op.ProcessGatherB2Crop(startTask, endTask);
        } else if constexpr (schMode == BTS_SCH_MODE_DIRECT_B2_NOCROP) {
            op.ProcessRowsB2NoCrop(startTask, endTask);
        } else {
            op.ProcessRows(startTask, endTask);
        }
    }
}
