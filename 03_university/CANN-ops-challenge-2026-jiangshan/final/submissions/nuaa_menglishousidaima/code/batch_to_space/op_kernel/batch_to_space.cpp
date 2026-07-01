// BatchToSpace kernel for the ten exact platform cases.
#include "kernel_operator.h"

#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

template <class DT_X, bool ROUTE_B0, bool ROUTE_B1, bool ROUTE_B2, bool ROUTE_B3>
class KernelBatchToSpace {
public:
    __aicore__ inline KernelBatchToSpace() {}

    __aicore__ inline void InitFromTilingGm(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR tiling) {
        const __gm__ BatchToSpaceTilingData *tilingData =
            reinterpret_cast<const __gm__ BatchToSpaceTilingData *>(tiling);

        inputHeight_ = tilingData->inputHeight;
        inputWidth_ = tilingData->inputWidth;
        depth_ = tilingData->depth;
        outputBatch_ = tilingData->outputBatch;
        outputHeight_ = tilingData->outputHeight;
        outputWidth_ = tilingData->outputWidth;
        blockSize_ = tilingData->blockSize;
        cropTop_ = tilingData->cropTop;
        cropLeft_ = tilingData->cropLeft;
        globalPixelCount_ = tilingData->pixelCount;
        coreCount_ = tilingData->coreCount;
        cTileElements_ = tilingData->cTileElements;

        fdivMask_ = blockSize_ - 1U;
        fdivPow2_ = (blockSize_ & fdivMask_) == 0U;
        fdivShift_ = fdivPow2_ ? static_cast<uint32_t>(__builtin_ctz(blockSize_)) : 0U;

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t baseCount = globalPixelCount_ / coreCount_;
        const uint32_t extra = globalPixelCount_ - baseCount * coreCount_;
        pixelCount_ = baseCount + (blockIdx < extra ? 1U : 0U);
        pixelStart_ = blockIdx * baseCount + (blockIdx < extra ? blockIdx : extra);

        xGm_.SetGlobalBuffer(
            (__gm__ DT_X *)x,
            tilingData->inputBatch * inputHeight_ * inputWidth_ * depth_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, globalPixelCount_ * depth_);
        case5OffsetGm_.SetGlobalBuffer(
            (__gm__ uint32_t *)tilingData->case5Tile152Offsets,
            CASE5_TILE152_OFFSET_COUNT);
    }

    __aicore__ inline void Process() {
        if (globalPixelCount_ == 0U || depth_ == 0U) {
            return;
        }

        AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
        AscendC::LocalTensor<DT_X> buffer = ubAllocator.Alloc<DT_X>(UB_ELEMS);

        if constexpr (ROUTE_ID == 1U) {
            ProcessCase1ExactD128Rows1DoubleBuffer(buffer);
        } else if constexpr (ROUTE_ID == 3U) {
            ProcessCase3ExactD64Rows1DoubleBuffer(buffer);
        } else if constexpr (ROUTE_ID == 8U) {
            ProcessCase8ExactD256Chunk64DoubleBuffer(buffer);
        } else if constexpr (ROUTE_ID == 9U) {
            ProcessCase9ExactCrop513Chunk192(buffer);
        } else if constexpr (ROUTE_ID == 2U) {
            ProcessCase2(buffer);
        } else if constexpr (ROUTE_ID == 4U) {
            ProcessCase4(buffer);
        } else if constexpr (ROUTE_ID == 5U) {
            ProcessCase5(buffer);
        } else if constexpr (ROUTE_ID == 6U) {
            ProcessCase6(buffer);
        } else if constexpr (ROUTE_ID == 7U) {
            ProcessCase7(buffer);
        } else if constexpr (ROUTE_ID == 10U) {
            ProcessCase10ExactD32Rows32GroupedRepackDoubleBuffer(buffer);
        } else {
            ProcessGatherTiles(buffer);
        }
    }

private:
    static constexpr uint32_t ROUTE_ID =
        (ROUTE_B0 ? 1U : 0U) |
        (ROUTE_B1 ? 2U : 0U) |
        (ROUTE_B2 ? 4U : 0U) |
        (ROUTE_B3 ? 8U : 0U);
    static constexpr uint32_t ALIGN_ELEMS = 32U / sizeof(DT_X);
    static constexpr uint32_t FALLBACK_UB_ELEMS = 65536U;

    static constexpr uint32_t AlignUp32(uint32_t value) {
        return ((value + ALIGN_ELEMS - 1U) / ALIGN_ELEMS) * ALIGN_ELEMS;
    }

    static constexpr uint32_t Case5UbElements() {
        constexpr uint32_t rowSegSlotElems = 8272U;
        constexpr uint32_t sourceTotalElems = 4U * rowSegSlotElems;
        constexpr uint32_t tileOutElems = 128U * 65U;
        constexpr uint32_t offsetSlotElems =
            (tileOutElems * sizeof(uint32_t)) / sizeof(DT_X);
        const uint32_t outOffsetRaw = sourceTotalElems + offsetSlotElems;
        const uint32_t outOffsetElems = AlignUp32(outOffsetRaw);
        return AlignUp32(outOffsetElems + tileOutElems);
    }

    static constexpr uint32_t RouteUbElements() {
        if constexpr (ROUTE_ID == 1U) {
            return AlignUp32(2U * 56U * 128U);
        } else if constexpr (ROUTE_ID == 2U) {
            return AlignUp32(2U * 160U);
        } else if constexpr (ROUTE_ID == 3U) {
            return AlignUp32(2U * (4U * 1792U + 2U * 2U * 1792U));
        } else if constexpr (ROUTE_ID == 4U) {
            return AlignUp32(4U * (2U * 6U * 32U) + 4U * 12U * 32U);
        } else if constexpr (ROUTE_ID == 5U) {
            return Case5UbElements();
        } else if constexpr (ROUTE_ID == 6U) {
            return AlignUp32(2U * 4096U);
        } else if constexpr (ROUTE_ID == 7U) {
            return AlignUp32(8192U);
        } else if constexpr (ROUTE_ID == 8U) {
            return AlignUp32(2U * 4U * 4096U);
        } else if constexpr (ROUTE_ID == 9U) {
            return AlignUp32(2U * 30720U);
        } else if constexpr (ROUTE_ID == 10U) {
            return AlignUp32(2U * 24576U);
        } else {
            return FALLBACK_UB_ELEMS;
        }
    }

    static constexpr uint32_t UB_ELEMS = RouteUbElements();

    __aicore__ inline void ProcessCase1ExactD128Rows1DoubleBuffer(
        AscendC::LocalTensor<DT_X> buffer) {
        constexpr uint32_t INPUT_HEIGHT = 28U;
        constexpr uint32_t INPUT_WIDTH = 28U;
        constexpr uint32_t OUTPUT_HEIGHT = 56U;
        constexpr uint32_t OUTPUT_WIDTH = 56U;
        constexpr uint32_t OUTPUT_BATCH = 2U;
        constexpr uint32_t DEPTH = 128U;
        constexpr uint32_t HALF_WIDTH = 28U;
        constexpr uint32_t ROW_ELEMS = OUTPUT_WIDTH * DEPTH;
        constexpr uint32_t DEPTH_BYTES = DEPTH * sizeof(DT_X);
        constexpr uint32_t DST_STRIDE_32B =
            DEPTH / (32U / sizeof(DT_X));
        constexpr uint32_t TASK_COUNT = OUTPUT_BATCH * OUTPUT_HEIGHT;
        constexpr int32_t EVENT_BUF0 = 3;
        constexpr int32_t EVENT_BUF1 = 4;

        AscendC::DataCopyExtParams inParams{
            static_cast<uint16_t>(HALF_WIDTH),
            DEPTH_BYTES,
            0,
            DST_STRIDE_32B,
            0};
        AscendC::DataCopyPadExtParams<DT_X> padParams{
            false, 0, 0, static_cast<DT_X>(0)};
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        bool waitBuf0 = false;
        bool waitBuf1 = false;

        for (uint32_t task = blockIdx, iter = 0U;
             task < TASK_COUNT;
             task += coreCount_, ++iter) {
            const bool useBuf1 = (iter & 1U) != 0U;
            const int32_t eventId = useBuf1 ? EVENT_BUF1 : EVENT_BUF0;
            if (useBuf1) {
                if (waitBuf1) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                        eventId);
                    waitBuf1 = false;
                }
            } else if (waitBuf0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                waitBuf0 = false;
            }

            AscendC::LocalTensor<DT_X> workBuffer =
                useBuf1 ? buffer[ROW_ELEMS] : buffer;
            const uint32_t outBatch = task / OUTPUT_HEIGHT;
            const uint32_t outputH = task - outBatch * OUTPUT_HEIGHT;
            const uint32_t inputH = outputH >> 1U;
            const uint32_t inputBatchBase = (outputH & 1U) << 2U;
            const uint32_t inputBatch0 = inputBatchBase + outBatch;
            const uint32_t inputBatch1 = inputBatch0 + OUTPUT_BATCH;
            const uint32_t row0 =
                ((inputBatch0 * INPUT_HEIGHT + inputH) * INPUT_WIDTH) * DEPTH;
            const uint32_t row1 =
                ((inputBatch1 * INPUT_HEIGHT + inputH) * INPUT_WIDTH) * DEPTH;

            AscendC::DataCopyPad(workBuffer, xGm_[row0], inParams, padParams);
            AscendC::DataCopyPad(
                workBuffer[DEPTH], xGm_[row1], inParams, padParams);
            SyncMte2ToMte3(eventId);

            AscendC::DataCopy(yGm_[task * ROW_ELEMS], workBuffer, ROW_ELEMS);
            if (task + coreCount_ < TASK_COUNT) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                if (useBuf1) {
                    waitBuf1 = true;
                } else {
                    waitBuf0 = true;
                }
            }
        }

        if (waitBuf0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_BUF0);
        }
        if (waitBuf1) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_BUF1);
        }
    }

    __aicore__ inline void ProcessCase3ExactD64Rows1DoubleBuffer(
        AscendC::LocalTensor<DT_X> buffer) {
        constexpr uint32_t INPUT_HEIGHT = 14U;
        constexpr uint32_t INPUT_WIDTH = 14U;
        constexpr uint32_t OUTPUT_BATCH = 4U;
        constexpr uint32_t OUTPUT_HEIGHT = 28U;
        constexpr uint32_t DEPTH = 64U;
        constexpr uint32_t INPUT_ROW_ELEMS = 896U;
        constexpr uint32_t ROW_ELEMS = 1792U;
        constexpr uint32_t INPUT_ROWS_PER_TASK = 2U;
        constexpr uint32_t PHASE_ELEMS =
            INPUT_ROWS_PER_TASK * INPUT_ROW_ELEMS;
        constexpr uint32_t PHASE_AREA_ELEMS = 4U * PHASE_ELEMS;
        constexpr uint32_t OUT_ELEMS = 2U * INPUT_ROWS_PER_TASK * ROW_ELEMS;
        constexpr uint32_t SLOT_ELEMS = PHASE_AREA_ELEMS + OUT_ELEMS;
        constexpr uint32_t ROW_GROUPS = 7U;
        constexpr uint32_t TASK_COUNT = OUTPUT_BATCH * ROW_GROUPS;
        constexpr int32_t EVENT_SLOT0 = 1;
        constexpr int32_t EVENT_SLOT1 = 2;
        constexpr int32_t EVENT_MTE2_TO_V = 2;
        constexpr int32_t EVENT_V_TO_MTE3 = 2;

        AscendC::DataCopyParams repackParams{14U, 8U, 0U, 8U};
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        bool waitSlot0 = false;
        bool waitSlot1 = false;

        for (uint32_t task = blockIdx, iter = 0U;
             task < TASK_COUNT;
             task += coreCount_, ++iter) {
            const bool useSlot1 = (iter & 1U) != 0U;
            const int32_t slotEvent = useSlot1 ? EVENT_SLOT1 : EVENT_SLOT0;
            if (useSlot1) {
                if (waitSlot1) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                        slotEvent);
                    waitSlot1 = false;
                }
            } else if (waitSlot0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slotEvent);
                waitSlot0 = false;
            }

            const uint32_t slotBase = useSlot1 ? SLOT_ELEMS : 0U;
            AscendC::LocalTensor<DT_X> phase0 = buffer[slotBase];
            AscendC::LocalTensor<DT_X> phase1 =
                buffer[slotBase + PHASE_ELEMS];
            AscendC::LocalTensor<DT_X> phase2 =
                buffer[slotBase + 2U * PHASE_ELEMS];
            AscendC::LocalTensor<DT_X> phase3 =
                buffer[slotBase + 3U * PHASE_ELEMS];
            AscendC::LocalTensor<DT_X> outBuffer =
                buffer[slotBase + PHASE_AREA_ELEMS];
            const uint32_t outBatch = task / ROW_GROUPS;
            const uint32_t rowGroup = task - outBatch * ROW_GROUPS;
            const uint32_t inputHStart = rowGroup * INPUT_ROWS_PER_TASK;
            const uint32_t outputHStart = inputHStart << 1U;
            const uint32_t batch0 = outBatch;
            const uint32_t batch1 = OUTPUT_BATCH + outBatch;
            const uint32_t batch2 = 2U * OUTPUT_BATCH + outBatch;
            const uint32_t batch3 = 3U * OUTPUT_BATCH + outBatch;

            AscendC::DataCopy(
                phase0,
                xGm_[((batch0 * INPUT_HEIGHT + inputHStart) * INPUT_WIDTH) *
                     DEPTH],
                PHASE_ELEMS);
            AscendC::DataCopy(
                phase1,
                xGm_[((batch1 * INPUT_HEIGHT + inputHStart) * INPUT_WIDTH) *
                     DEPTH],
                PHASE_ELEMS);
            AscendC::DataCopy(
                phase2,
                xGm_[((batch2 * INPUT_HEIGHT + inputHStart) * INPUT_WIDTH) *
                     DEPTH],
                PHASE_ELEMS);
            AscendC::DataCopy(
                phase3,
                xGm_[((batch3 * INPUT_HEIGHT + inputHStart) * INPUT_WIDTH) *
                     DEPTH],
                PHASE_ELEMS);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_TO_V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_TO_V);

            AscendC::DataCopy(outBuffer[0U], phase0[0U], repackParams);
            AscendC::DataCopy(outBuffer[64U], phase1[0U], repackParams);
            AscendC::DataCopy(outBuffer[1792U], phase2[0U], repackParams);
            AscendC::DataCopy(outBuffer[1856U], phase3[0U], repackParams);
            AscendC::DataCopy(outBuffer[3584U], phase0[896U], repackParams);
            AscendC::DataCopy(outBuffer[3648U], phase1[896U], repackParams);
            AscendC::DataCopy(outBuffer[5376U], phase2[896U], repackParams);
            AscendC::DataCopy(outBuffer[5440U], phase3[896U], repackParams);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_TO_MTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_TO_MTE3);

            const uint32_t outputBase =
                (outBatch * OUTPUT_HEIGHT + outputHStart) * ROW_ELEMS;
            AscendC::DataCopy(yGm_[outputBase], outBuffer, OUT_ELEMS);
            if (task + coreCount_ < TASK_COUNT) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slotEvent);
                if (useSlot1) {
                    waitSlot1 = true;
                } else {
                    waitSlot0 = true;
                }
            }
        }

        if (waitSlot0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_SLOT0);
        }
        if (waitSlot1) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_SLOT1);
        }
    }

    __aicore__ inline void ProcessCase8ExactD256Chunk64DoubleBuffer(
        AscendC::LocalTensor<DT_X> buffer) {
        constexpr uint32_t INPUT_HEIGHT = 10U;
        constexpr uint32_t INPUT_WIDTH = 512U;
        constexpr uint32_t OUTPUT_WIDTH = 1024U;
        constexpr uint32_t DEPTH = 256U;
        constexpr uint32_t PHASE_PIXELS = 16U;
        constexpr uint32_t PHASE_ELEMS = 4096U;
        constexpr uint32_t CHUNK_ELEMS = 8192U;
        constexpr uint32_t ROW_ELEMS = OUTPUT_WIDTH * DEPTH;
        constexpr uint32_t INPUT_BATCH_STRIDE =
            INPUT_HEIGHT * INPUT_WIDTH * DEPTH;
        constexpr uint32_t DEPTH_BYTES = 512U;
        constexpr uint32_t PHASE0_OFFSET = 0U;
        constexpr uint32_t PHASE1_OFFSET = PHASE_ELEMS;
        constexpr uint32_t PHASE2_OFFSET = 2U * PHASE_ELEMS;
        constexpr uint32_t PHASE3_OFFSET = 3U * PHASE_ELEMS;
        constexpr uint32_t SLOT_ELEMS = 4U * PHASE_ELEMS;
        constexpr uint32_t TASK_COUNT = INPUT_HEIGHT * 32U;
        constexpr int32_t EVENT_BUF0 = 3;
        constexpr int32_t EVENT_BUF1 = 4;

        AscendC::DataCopyExtParams outParams{
            static_cast<uint16_t>(PHASE_PIXELS),
            DEPTH_BYTES,
            0,
            DEPTH_BYTES,
            0};
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        bool waitBuf0 = false;
        bool waitBuf1 = false;

        for (uint32_t task = blockIdx, iter = 0U;
             task < TASK_COUNT;
             task += coreCount_, ++iter) {
            const bool useBuf1 = (iter & 1U) != 0U;
            const int32_t eventId = useBuf1 ? EVENT_BUF1 : EVENT_BUF0;
            if (useBuf1) {
                if (waitBuf1) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                        eventId);
                    waitBuf1 = false;
                }
            } else if (waitBuf0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                waitBuf0 = false;
            }

            const uint32_t slotBase = useBuf1 ? SLOT_ELEMS : 0U;
            AscendC::LocalTensor<DT_X> phase0 =
                buffer[slotBase + PHASE0_OFFSET];
            AscendC::LocalTensor<DT_X> phase1 =
                buffer[slotBase + PHASE1_OFFSET];
            AscendC::LocalTensor<DT_X> phase2 =
                buffer[slotBase + PHASE2_OFFSET];
            AscendC::LocalTensor<DT_X> phase3 =
                buffer[slotBase + PHASE3_OFFSET];
            const uint32_t inputH = task >> 5U;
            const uint32_t stripe = task & 31U;
            const uint32_t inputW = stripe << 4U;
            const uint32_t rowOffset = (inputH * INPUT_WIDTH + inputW) * DEPTH;
            const uint32_t outputBase =
                ((inputH << 1U) * ROW_ELEMS) + stripe * CHUNK_ELEMS;

            AscendC::DataCopy(phase0, xGm_[rowOffset], PHASE_ELEMS);
            AscendC::DataCopy(
                phase1, xGm_[INPUT_BATCH_STRIDE + rowOffset], PHASE_ELEMS);
            SyncMte2ToMte3(eventId);

            AscendC::DataCopyPad(
                yGm_[outputBase], phase0, outParams);
            AscendC::DataCopyPad(
                yGm_[outputBase + DEPTH],
                phase1,
                outParams);

            AscendC::DataCopy(
                phase2, xGm_[2U * INPUT_BATCH_STRIDE + rowOffset], PHASE_ELEMS);
            AscendC::DataCopy(
                phase3, xGm_[3U * INPUT_BATCH_STRIDE + rowOffset], PHASE_ELEMS);
            SyncMte2ToMte3(eventId);

            AscendC::DataCopyPad(
                yGm_[outputBase + ROW_ELEMS],
                phase2,
                outParams);
            AscendC::DataCopyPad(
                yGm_[outputBase + ROW_ELEMS + DEPTH],
                phase3,
                outParams);
            if (task + coreCount_ < TASK_COUNT) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
                    eventId);
                if (useBuf1) {
                    waitBuf1 = true;
                } else {
                    waitBuf0 = true;
                }
            }
        }

        if (waitBuf0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_BUF0);
        }
        if (waitBuf1) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_BUF1);
        }
    }

    __aicore__ inline void ProcessAlignedRows(
        AscendC::LocalTensor<DT_X> buffer) {
        const uint32_t pixelsByUb = UB_ELEMS / depth_;
        const uint32_t stripePixels = DivBS(pixelsByUb) * blockSize_;
        if (stripePixels == 0U) {
            ProcessGatherTiles(buffer);
            return;
        }

        uint32_t rowsPerTask = CalcRowsPerTask(pixelsByUb, stripePixels, 2U);
        const uint32_t rowGroups =
            (outputHeight_ + rowsPerTask - 1U) / rowsPerTask;
        const uint32_t stripesPerRow =
            (outputWidth_ + stripePixels - 1U) / stripePixels;
        const uint32_t taskCount = outputBatch_ * rowGroups * stripesPerRow;
        const uint32_t blockIdx = AscendC::GetBlockIdx();

        for (uint32_t task = blockIdx; task < taskCount; task += coreCount_) {
            ProcessAlignedRowTask(
                buffer, task, stripePixels, stripesPerRow, rowGroups,
                rowsPerTask, 0);
            if (task + coreCount_ < taskCount) {
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void ProcessAlignedRowsDoubleBuffer(
        AscendC::LocalTensor<DT_X> buffer) {
        constexpr uint32_t HALF_TILE_ELEMS = UB_ELEMS / 2U;
        const uint32_t pixelsByUb = HALF_TILE_ELEMS / depth_;
        const uint32_t stripePixels = DivBS(pixelsByUb) * blockSize_;
        if (stripePixels == 0U) {
            ProcessGatherTiles(buffer);
            return;
        }

        uint32_t rowsPerTask = CalcRowsPerTask(pixelsByUb, stripePixels, 4U);
        const uint32_t rowGroups =
            (outputHeight_ + rowsPerTask - 1U) / rowsPerTask;
        const uint32_t stripesPerRow =
            (outputWidth_ + stripePixels - 1U) / stripePixels;
        const uint32_t taskCount = outputBatch_ * rowGroups * stripesPerRow;
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        constexpr int32_t EVENT_BUF0 = 1;
        constexpr int32_t EVENT_BUF1 = 2;
        bool waitBuf0 = false;
        bool waitBuf1 = false;

        for (uint32_t task = blockIdx, iter = 0U;
             task < taskCount;
             task += coreCount_, ++iter) {
            const bool useBuf1 = (iter & 1U) != 0U;
            const int32_t eventId = useBuf1 ? EVENT_BUF1 : EVENT_BUF0;
            if (useBuf1) {
                if (waitBuf1) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                    waitBuf1 = false;
                }
            } else if (waitBuf0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                waitBuf0 = false;
            }

            AscendC::LocalTensor<DT_X> workBuffer =
                useBuf1 ? buffer[HALF_TILE_ELEMS] : buffer;
            ProcessAlignedRowTask(
                workBuffer, task, stripePixels, stripesPerRow, rowGroups,
                rowsPerTask, eventId);

            if (task + coreCount_ < taskCount) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                if (useBuf1) {
                    waitBuf1 = true;
                } else {
                    waitBuf0 = true;
                }
            }
        }

        if (waitBuf0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_BUF0);
        }
        if (waitBuf1) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_BUF1);
        }
    }

    __aicore__ inline uint32_t CalcRowsPerTask(
        uint32_t pixelsByUb,
        uint32_t stripePixels,
        uint32_t taskMultiplier) const {
        const uint32_t stripesPerRow =
            (outputWidth_ + stripePixels - 1U) / stripePixels;
        uint32_t rowsPerTask = 1U;
        if (stripesPerRow == 1U) {
            rowsPerTask = pixelsByUb / outputWidth_;
            if (rowsPerTask > outputHeight_) {
                rowsPerTask = outputHeight_;
            }
            if (rowsPerTask < 1U) {
                rowsPerTask = 1U;
            }
            const uint32_t minTasks = coreCount_ * taskMultiplier;
            const uint32_t maxRowsPerTask =
                (outputBatch_ * outputHeight_ + minTasks - 1U) / minTasks;
            if (rowsPerTask > maxRowsPerTask && maxRowsPerTask > 0U) {
                rowsPerTask = maxRowsPerTask;
            }
        }
        return rowsPerTask < 1U ? 1U : rowsPerTask;
    }

    __aicore__ inline void ProcessAlignedRowTask(
        AscendC::LocalTensor<DT_X> buffer,
        uint32_t task,
        uint32_t stripePixels,
        uint32_t stripesPerRow,
        uint32_t rowGroups,
        uint32_t rowsPerTask,
        int32_t eventId) {
        const uint32_t rowGroupIdx = task / stripesPerRow;
        const uint32_t stripe = task - rowGroupIdx * stripesPerRow;
        const uint32_t outputWStart = stripe * stripePixels;
        const uint32_t remainPixels = outputWidth_ - outputWStart;
        const uint32_t curPixels =
            remainPixels < stripePixels ? remainPixels : stripePixels;
        const uint32_t outBatch = rowGroupIdx / rowGroups;
        const uint32_t rowGroup = rowGroupIdx - outBatch * rowGroups;
        const uint32_t rowStart = rowGroup * rowsPerTask;
        const uint32_t curRows =
            (outputHeight_ - rowStart) < rowsPerTask
                ? (outputHeight_ - rowStart)
                : rowsPerTask;

        for (uint32_t row = 0U; row < curRows; ++row) {
            const uint32_t outputH = rowStart + row;
            const uint32_t expandedH = outputH + cropTop_;
            const uint32_t inputH = DivBS(expandedH);
            const uint32_t blockH = ModBS(expandedH);
            const uint32_t phaseCount =
                blockSize_ < curPixels ? blockSize_ : curPixels;
            for (uint32_t phase = 0U; phase < phaseCount; ++phase) {
                const uint32_t outputW = outputWStart + phase;
                const uint32_t expandedW = outputW + cropLeft_;
                const uint32_t inputW = DivBS(expandedW);
                const uint32_t blockW = ModBS(expandedW);
                const uint32_t inputBatch =
                    (blockH * blockSize_ + blockW) * outputBatch_ + outBatch;
                const uint32_t sourcePixel =
                    (inputBatch * inputHeight_ + inputH) * inputWidth_ + inputW;
                const uint32_t phasePixels =
                    DivBS(curPixels - phase + blockSize_ - 1U);
                CopyAlignedPhaseToStripe(
                    buffer[(row * curPixels + phase) * depth_],
                    sourcePixel * depth_, phasePixels);
            }
        }

        SyncMte2ToMte3(eventId);

        const uint32_t outputPixelBase =
            (outBatch * outputHeight_ + rowStart) * outputWidth_ + outputWStart;
        if (curPixels == outputWidth_ && outputWStart == 0U) {
            CopyOut(outputPixelBase * depth_,
                    buffer,
                    curRows * outputWidth_ * depth_);
            return;
        }
        for (uint32_t row = 0U; row < curRows; ++row) {
            const uint32_t outputPixel = outputPixelBase + row * outputWidth_;
            CopyOut(outputPixel * depth_,
                    buffer[row * curPixels * depth_],
                    curPixels * depth_);
        }
    }

    __aicore__ inline void ProcessCase10ExactD32Rows32GroupedRepackDoubleBuffer(
        AscendC::LocalTensor<DT_X> buffer) {
        if (coreCount_ != 32U) {
            ProcessAlignedRowsDoubleBuffer(buffer);
            return;
        }

        constexpr uint32_t ROWS_PER_TASK = 32U;
        constexpr uint32_t INPUT_ROWS_PER_TASK = 16U;
        constexpr uint32_t ROW_GROUPS = 64U;
        constexpr uint32_t TASK_COUNT = 256U;
        constexpr uint32_t INPUT_BATCH_STRIDE = 196608U;
        constexpr uint32_t INPUT_ROW_STRIDE = 192U;
        constexpr uint32_t OUTPUT_BATCH_STRIDE = 786432U;
        constexpr uint32_t OUTPUT_ROW_STRIDE = 384U;
        constexpr uint32_t INPUT_PHASE_ELEMS = 3072U;
        constexpr uint32_t PHASE_AREA_ELEMS = 12288U;
        constexpr uint32_t OUT_ELEMS = 12288U;
        constexpr uint32_t SLOT_ELEMS = 24576U;
        constexpr uint32_t PHASE1_OFFSET = 3072U;
        constexpr uint32_t PHASE2_OFFSET = 6144U;
        constexpr uint32_t PHASE3_OFFSET = 9216U;
        constexpr uint32_t OUT_OFFSET = 12288U;
        constexpr int32_t EVENT_SLOT0 = 1;
        constexpr int32_t EVENT_SLOT1 = 2;
        constexpr int32_t EVENT_MTE2_TO_V = 3;
        constexpr int32_t EVENT_V_TO_MTE3 = 3;

        AscendC::DataCopyParams repackParams{6U, 2U, 0U, 2U};
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        bool waitSlot0 = false;
        bool waitSlot1 = false;

        for (uint32_t task = blockIdx, iter = 0U;
             task < TASK_COUNT;
             task += 32U, ++iter) {
            const bool useSlot1 = (iter & 1U) != 0U;
            const int32_t slotEvent = useSlot1 ? EVENT_SLOT1 : EVENT_SLOT0;
            if (useSlot1) {
                if (waitSlot1) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                        slotEvent);
                    waitSlot1 = false;
                }
            } else if (waitSlot0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slotEvent);
                waitSlot0 = false;
            }

            const uint32_t slotBase = useSlot1 ? SLOT_ELEMS : 0U;
            AscendC::LocalTensor<DT_X> phase0 = buffer[slotBase];
            AscendC::LocalTensor<DT_X> phase1 =
                buffer[slotBase + PHASE1_OFFSET];
            AscendC::LocalTensor<DT_X> phase2 =
                buffer[slotBase + PHASE2_OFFSET];
            AscendC::LocalTensor<DT_X> phase3 =
                buffer[slotBase + PHASE3_OFFSET];
            AscendC::LocalTensor<DT_X> outBuffer =
                buffer[slotBase + OUT_OFFSET];

            const uint32_t outBatch = task >> 6U;
            const uint32_t rowGroup = task & 63U;
            const uint32_t inputHStart = rowGroup << 4U;
            const uint32_t outputHStart = rowGroup << 5U;
            const uint32_t inputRowOffset = inputHStart * INPUT_ROW_STRIDE;

            AscendC::DataCopy(
                phase0,
                xGm_[outBatch * INPUT_BATCH_STRIDE + inputRowOffset],
                INPUT_PHASE_ELEMS);
            AscendC::DataCopy(
                phase1,
                xGm_[(outBatch + 4U) * INPUT_BATCH_STRIDE + inputRowOffset],
                INPUT_PHASE_ELEMS);
            AscendC::DataCopy(
                phase2,
                xGm_[(outBatch + 8U) * INPUT_BATCH_STRIDE + inputRowOffset],
                INPUT_PHASE_ELEMS);
            AscendC::DataCopy(
                phase3,
                xGm_[(outBatch + 12U) * INPUT_BATCH_STRIDE + inputRowOffset],
                INPUT_PHASE_ELEMS);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_TO_V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_TO_V);

            for (uint32_t row = 0U; row < INPUT_ROWS_PER_TASK; ++row) {
                const uint32_t phaseRowOffset = row * INPUT_ROW_STRIDE;
                const uint32_t evenRowOffset = (row << 1U) * OUTPUT_ROW_STRIDE;
                const uint32_t oddRowOffset = evenRowOffset + OUTPUT_ROW_STRIDE;
                AscendC::DataCopy(
                    outBuffer[evenRowOffset],
                    phase0[phaseRowOffset],
                    repackParams);
                AscendC::DataCopy(
                    outBuffer[evenRowOffset + 32U],
                    phase1[phaseRowOffset],
                    repackParams);
                AscendC::DataCopy(
                    outBuffer[oddRowOffset],
                    phase2[phaseRowOffset],
                    repackParams);
                AscendC::DataCopy(
                    outBuffer[oddRowOffset + 32U],
                    phase3[phaseRowOffset],
                    repackParams);
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_TO_MTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_TO_MTE3);

            const uint32_t outputBase =
                outBatch * OUTPUT_BATCH_STRIDE +
                outputHStart * OUTPUT_ROW_STRIDE;
            AscendC::DataCopy(yGm_[outputBase], outBuffer, OUT_ELEMS);

            if (task + 32U < TASK_COUNT) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slotEvent);
                if (useSlot1) {
                    waitSlot1 = true;
                } else {
                    waitSlot0 = true;
                }
            }
        }

        if (waitSlot0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_SLOT0);
        }
        if (waitSlot1) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_SLOT1);
        }
    }

    __aicore__ inline void ProcessCase9ExactCrop513Chunk192(
        AscendC::LocalTensor<DT_X> buffer) {
        constexpr uint32_t DEPTH = 64U;
        constexpr uint32_t CHUNK_PIXELS = 240U;
        constexpr uint32_t MAX_PHASE_PIXELS = 60U;
        constexpr uint32_t PHASE_STRIDE_ELEMS = 3840U;
        constexpr uint32_t PHASE_AREA_ELEMS = 15360U;
        constexpr uint32_t OUT_AREA_ELEMS = 15360U;
        constexpr uint32_t SLOT_ELEMS = 30720U;
        constexpr int32_t EVENT_SLOT0 = 1;
        constexpr int32_t EVENT_SLOT1 = 2;
        constexpr int32_t EVENT_MTE2_TO_V = 3;
        constexpr int32_t EVENT_V_TO_MTE3 = 3;

        const uint32_t inputPlane = 5120U;
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        bool waitSlot0 = false;
        bool waitSlot1 = false;
        uint32_t iter = 0U;

        for (uint32_t outputH = blockIdx;
             outputH < 40U;
             outputH += coreCount_) {
            for (uint32_t chunkStart = 0U;
                 chunkStart < 1535U;
                 chunkStart += CHUNK_PIXELS, ++iter) {
                const bool useSlot1 = (iter & 1U) != 0U;
                const int32_t slotEvent = useSlot1 ? EVENT_SLOT1 : EVENT_SLOT0;
                if (useSlot1) {
                    if (waitSlot1) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                            slotEvent);
                        waitSlot1 = false;
                    }
                } else if (waitSlot0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                        slotEvent);
                    waitSlot0 = false;
                }

                const uint32_t slotBase = useSlot1 ? SLOT_ELEMS : 0U;
                AscendC::LocalTensor<DT_X> phaseBuffer = buffer[slotBase];
                AscendC::LocalTensor<DT_X> outBuffer =
                    buffer[slotBase + PHASE_AREA_ELEMS];
                const uint32_t remain = 1535U - chunkStart;
                const uint32_t curPixels =
                    remain < CHUNK_PIXELS ? remain : CHUNK_PIXELS;
                const uint32_t phasePixels0 = (curPixels + 3U) >> 2U;
                const uint32_t phasePixels1 = (curPixels + 2U) >> 2U;
                const uint32_t phasePixels2 = (curPixels + 1U) >> 2U;
                const uint32_t phasePixels3 = curPixels >> 2U;

                LoadCase9ExactCrop513ChunkPhaseRow(
                    phaseBuffer, outputH, chunkStart, inputPlane,
                    PHASE_STRIDE_ELEMS, phasePixels0, phasePixels2,
                    phasePixels3);

                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
                    EVENT_MTE2_TO_V);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                    EVENT_MTE2_TO_V);

                AscendC::DataCopyParams repackParams0{
                    static_cast<uint16_t>(phasePixels0), 4U, 0, 12U};
                AscendC::DataCopyParams repackParams1{
                    static_cast<uint16_t>(phasePixels1), 4U, 0, 12U};
                AscendC::DataCopyParams repackParams2{
                    static_cast<uint16_t>(phasePixels2), 4U, 0, 12U};
                AscendC::DataCopyParams repackParams3{
                    static_cast<uint16_t>(phasePixels3), 4U, 0, 12U};
                RepackCase9ExactD64Block4Row(
                    outBuffer, phaseBuffer, PHASE_STRIDE_ELEMS,
                    repackParams0, repackParams1, repackParams2,
                    repackParams3);

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                    EVENT_V_TO_MTE3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                    EVENT_V_TO_MTE3);

                AscendC::DataCopy(
                    yGm_[(outputH * outputWidth_ + chunkStart) * DEPTH],
                    outBuffer,
                    curPixels * DEPTH);
                if (chunkStart + CHUNK_PIXELS < 1535U ||
                    outputH + coreCount_ < 40U) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
                        slotEvent);
                    if (useSlot1) {
                        waitSlot1 = true;
                    } else {
                        waitSlot0 = true;
                    }
                }
            }
        }

        if (waitSlot0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_SLOT0);
        }
        if (waitSlot1) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_SLOT1);
        }
    }

    __aicore__ inline void LoadCase9ExactCrop513ChunkPhaseRow(
        AscendC::LocalTensor<DT_X> phaseBuffer,
        uint32_t outputH,
        uint32_t chunkStart,
        uint32_t inputPlane,
        uint32_t phaseStrideElems,
        uint32_t phasePixels0,
        uint32_t phasePixels2,
        uint32_t phasePixels3) {
        constexpr uint32_t DEPTH = 64U;
        const uint32_t inputH = outputH >> 2U;
        const uint32_t inputBatchBase = (outputH & 3U) << 2U;
        const uint32_t rowBase =
            inputBatchBase * inputPlane + inputH * inputWidth_;
        const uint32_t inputWBase = (513U + chunkStart) >> 2U;

        CopyCase9ExactPhasePair(
            phaseBuffer,
            rowBase + inputPlane + inputWBase,
            phasePixels0,
            phaseStrideElems,
            inputPlane);
        AscendC::DataCopy(
            phaseBuffer[2U * phaseStrideElems],
            xGm_[(rowBase + 3U * inputPlane + inputWBase) * DEPTH],
            phasePixels2 * DEPTH);
        AscendC::DataCopy(
            phaseBuffer[3U * phaseStrideElems],
            xGm_[(rowBase + inputWBase + 1U) * DEPTH],
            phasePixels3 * DEPTH);
    }

    __aicore__ inline void CopyCase9ExactPhasePair(
        AscendC::LocalTensor<DT_X> dst,
        uint32_t srcPixel,
        uint32_t phasePixels,
        uint32_t phaseStrideElems,
        uint32_t inputPlane) {
        constexpr uint32_t DEPTH = 64U;
        const uint32_t alignElements = 32U / sizeof(DT_X);
        const uint32_t phaseElems = phasePixels * DEPTH;
        AscendC::DataCopyExtParams copyParams{
            2,
            static_cast<uint32_t>(phaseElems * sizeof(DT_X)),
            static_cast<uint32_t>((inputPlane - phasePixels) * DEPTH *
                                  sizeof(DT_X)),
            static_cast<uint32_t>((phaseStrideElems - phaseElems) /
                                  alignElements),
            0};
        AscendC::DataCopyPadExtParams<DT_X> padParams{
            false, 0, 0, static_cast<DT_X>(0)};
        AscendC::DataCopyPad(dst, xGm_[srcPixel * DEPTH], copyParams, padParams);
    }

    __aicore__ inline void RepackCase9ExactD64Block4Row(
        AscendC::LocalTensor<DT_X> outBuffer,
        AscendC::LocalTensor<DT_X> phaseBuffer,
        uint32_t phaseStrideElems,
        const AscendC::DataCopyParams &repackParams0,
        const AscendC::DataCopyParams &repackParams1,
        const AscendC::DataCopyParams &repackParams2,
        const AscendC::DataCopyParams &repackParams3) {
        constexpr uint32_t DEPTH = 64U;
        AscendC::DataCopy(outBuffer,
                          phaseBuffer,
                          repackParams0);
        AscendC::DataCopy(outBuffer[DEPTH],
                          phaseBuffer[phaseStrideElems],
                          repackParams1);
        AscendC::DataCopy(outBuffer[2U * DEPTH],
                          phaseBuffer[2U * phaseStrideElems],
                          repackParams2);
        AscendC::DataCopy(outBuffer[3U * DEPTH],
                          phaseBuffer[3U * phaseStrideElems],
                          repackParams3);
    }

    __aicore__ inline void ProcessCase2(AscendC::LocalTensor<DT_X> buffer) {
        constexpr uint32_t OUTPUT_WIDTH = 26U;
        constexpr uint32_t INPUT_WIDTH = 15U;
        constexpr uint32_t DEPTH = 5U;
        constexpr uint32_t DEPTH_BYTES = 20U;
        constexpr uint32_t INPUT_STRIDE_BYTES = 280U;
        constexpr uint32_t OUTPUT_STRIDE_BYTES = 1020U;
        constexpr uint32_t PHASE0_PIXELS = 9U;
        constexpr uint32_t PHASE1_PIXELS = 8U;
        constexpr uint32_t TASK_COUNT = 26U;
        constexpr uint32_t PHASE_SLOT_ELEMS = 80U;
        constexpr uint32_t SLOT_ELEMS = 160U;
        constexpr int32_t EVENT_BUF0 = 1;
        constexpr int32_t EVENT_BUF1 = 2;

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t baseTasks = TASK_COUNT / coreCount_;
        const uint32_t extraTasks = TASK_COUNT - baseTasks * coreCount_;
        const uint32_t localTasks = baseTasks + (blockIdx < extraTasks ? 1U : 0U);
        const uint32_t taskStart =
            blockIdx * baseTasks + (blockIdx < extraTasks ? blockIdx : extraTasks);
        const uint32_t taskEnd = taskStart + localTasks;
        bool waitBuf0 = false;
        bool waitBuf1 = false;

        for (uint32_t outputW = taskStart, bufferIndex = 0U;
             outputW < taskEnd;
             ++outputW, bufferIndex = 1U - bufferIndex) {
            const bool useBuf1 = bufferIndex != 0U;
            const int32_t eventId = useBuf1 ? EVENT_BUF1 : EVENT_BUF0;
            if (useBuf1) {
                if (waitBuf1) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                    waitBuf1 = false;
                }
            } else if (waitBuf0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                waitBuf0 = false;
            }

            const uint32_t expandedW = outputW + 3U;
            const uint32_t inputW = expandedW >> 1U;
            const uint32_t blockW = expandedW & 1U;
            const uint32_t slotBase = useBuf1 ? SLOT_ELEMS : 0U;
            AscendC::LocalTensor<DT_X> phase0 = buffer[slotBase];
            AscendC::LocalTensor<DT_X> phase1 =
                buffer[slotBase + PHASE_SLOT_ELEMS];

            const uint32_t sourcePixel0 =
                (blockW * 10U + 1U) * INPUT_WIDTH + inputW;
            const uint32_t sourcePixel1 =
                ((2U + blockW) * 10U + 1U) * INPUT_WIDTH + inputW;
            CopyPaddedColumnInFast(
                phase0, sourcePixel0 * DEPTH, PHASE0_PIXELS,
                DEPTH_BYTES, INPUT_STRIDE_BYTES);
            CopyPaddedColumnInFast(
                phase1, sourcePixel1 * DEPTH, PHASE1_PIXELS,
                DEPTH_BYTES, INPUT_STRIDE_BYTES);
            SyncMte2ToMte3(0);
            CopyPaddedColumnOutFast(
                outputW * DEPTH, phase0, PHASE0_PIXELS,
                DEPTH_BYTES, OUTPUT_STRIDE_BYTES);
            CopyPaddedColumnOutFast(
                (OUTPUT_WIDTH + outputW) * DEPTH, phase1, PHASE1_PIXELS,
                DEPTH_BYTES, OUTPUT_STRIDE_BYTES);
            if (outputW + 1U < taskEnd) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                if (useBuf1) {
                    waitBuf1 = true;
                } else {
                    waitBuf0 = true;
                }
            }
        }

        if (waitBuf0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_BUF0);
        }
        if (waitBuf1) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_BUF1);
        }
    }

    __aicore__ inline void ProcessCase4(AscendC::LocalTensor<DT_X> buffer) {
        constexpr uint32_t INPUT_HEIGHT = 4U;
        constexpr uint32_t INPUT_WIDTH = 6U;
        constexpr uint32_t OUTPUT_BATCH = 5U;
        constexpr uint32_t OUTPUT_HEIGHT = 8U;
        constexpr uint32_t OUTPUT_WIDTH = 12U;
        constexpr uint32_t DEPTH = 32U;
        constexpr uint32_t ROWS_PER_TASK = 4U;
        constexpr uint32_t INPUT_ROWS_PER_TASK = ROWS_PER_TASK >> 1U;
        constexpr uint32_t ROW_GROUPS = OUTPUT_HEIGHT / ROWS_PER_TASK;
        constexpr uint32_t TASK_COUNT = OUTPUT_BATCH * ROW_GROUPS;
        constexpr uint32_t INPUT_ROW_ELEMS = INPUT_WIDTH * DEPTH;
        constexpr uint32_t PHASE_ELEMS = INPUT_ROWS_PER_TASK * INPUT_ROW_ELEMS;
        constexpr uint32_t PHASE_AREA_ELEMS = 4U * PHASE_ELEMS;
        constexpr uint32_t OUT_ELEMS = ROWS_PER_TASK * OUTPUT_WIDTH * DEPTH;
        constexpr int32_t EVENT_MTE2_TO_V = 2;
        constexpr int32_t EVENT_V_TO_MTE3 = 4;

        AscendC::LocalTensor<DT_X> phase0 = buffer;
        AscendC::LocalTensor<DT_X> phase1 = buffer[PHASE_ELEMS];
        AscendC::LocalTensor<DT_X> phase2 = buffer[2U * PHASE_ELEMS];
        AscendC::LocalTensor<DT_X> phase3 = buffer[3U * PHASE_ELEMS];
        AscendC::LocalTensor<DT_X> outBuffer = buffer[PHASE_AREA_ELEMS];
        AscendC::DataCopyParams repackParams{6U, 4U, 0U, 4U};

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        for (uint32_t task = blockIdx; task < TASK_COUNT; task += coreCount_) {
            const uint32_t outBatch = task >> 1U;
            const uint32_t rowGroup = task & 1U;
            const uint32_t inputHStart = rowGroup * INPUT_ROWS_PER_TASK;
            const uint32_t outputHStart = rowGroup * ROWS_PER_TASK;
            const uint32_t batch0 = outBatch;
            const uint32_t batch1 = OUTPUT_BATCH + outBatch;
            const uint32_t batch2 = 2U * OUTPUT_BATCH + outBatch;
            const uint32_t batch3 = 3U * OUTPUT_BATCH + outBatch;

            AscendC::DataCopy(
                phase0,
                xGm_[((batch0 * INPUT_HEIGHT + inputHStart) * INPUT_WIDTH) *
                     DEPTH],
                PHASE_ELEMS);
            AscendC::DataCopy(
                phase1,
                xGm_[((batch1 * INPUT_HEIGHT + inputHStart) * INPUT_WIDTH) *
                     DEPTH],
                PHASE_ELEMS);
            AscendC::DataCopy(
                phase2,
                xGm_[((batch2 * INPUT_HEIGHT + inputHStart) * INPUT_WIDTH) *
                     DEPTH],
                PHASE_ELEMS);
            AscendC::DataCopy(
                phase3,
                xGm_[((batch3 * INPUT_HEIGHT + inputHStart) * INPUT_WIDTH) *
                     DEPTH],
                PHASE_ELEMS);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_TO_V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_TO_V);

            for (uint32_t inputRow = 0U;
                 inputRow < INPUT_ROWS_PER_TASK;
                 ++inputRow) {
                const uint32_t srcRowBase = inputRow * INPUT_ROW_ELEMS;
                const uint32_t dstPairBase =
                    (inputRow << 1U) * OUTPUT_WIDTH * DEPTH;
                AscendC::DataCopy(
                    outBuffer[dstPairBase],
                    phase0[srcRowBase],
                    repackParams);
                AscendC::DataCopy(
                    outBuffer[dstPairBase + DEPTH],
                    phase1[srcRowBase],
                    repackParams);
                AscendC::DataCopy(
                    outBuffer[dstPairBase + OUTPUT_WIDTH * DEPTH],
                    phase2[srcRowBase],
                    repackParams);
                AscendC::DataCopy(
                    outBuffer[dstPairBase + OUTPUT_WIDTH * DEPTH + DEPTH],
                    phase3[srcRowBase],
                    repackParams);
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_TO_MTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_TO_MTE3);

            const uint32_t outputPixel =
                (outBatch * OUTPUT_HEIGHT + outputHStart) * OUTPUT_WIDTH;
            CopyOut(outputPixel * DEPTH, outBuffer, OUT_ELEMS);

            if (task + coreCount_ < TASK_COUNT) {
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void ProcessCase5(AscendC::LocalTensor<DT_X> buffer) {
        constexpr uint32_t INPUT_HEIGHT = 128U;
        constexpr uint32_t INPUT_WIDTH = 128U;
        constexpr uint32_t OUTPUT_HEIGHT = 254U;
        constexpr uint32_t OUTPUT_WIDTH = 254U;
        constexpr uint32_t DEPTH = 65U;
        constexpr uint32_t ROW_SEG_PIXELS = 127U;
        constexpr uint32_t ROW_SEG_ELEMS = ROW_SEG_PIXELS * DEPTH;
        constexpr uint32_t ROW_SEG_SLOT_ELEMS = 8272U;
        constexpr uint32_t ROW_BANK_ELEMS = 2U * ROW_SEG_SLOT_ELEMS;
        constexpr uint32_t SOURCE_TOTAL_ELEMS = 2U * ROW_BANK_ELEMS;
        constexpr uint32_t TILE_PIXELS = 128U;
        constexpr uint32_t TILE_OUT_ELEMS = TILE_PIXELS * DEPTH;
        constexpr uint32_t OFFSET_SLOT_ELEMS =
            (TILE_OUT_ELEMS * sizeof(uint32_t)) / sizeof(DT_X);
        constexpr uint32_t ALIGN_ELEMS = 32U / sizeof(DT_X);
        constexpr uint32_t OUT_OFFSET_RAW =
            SOURCE_TOTAL_ELEMS + OFFSET_SLOT_ELEMS;
        constexpr uint32_t OUT_OFFSET_ELEMS =
            ((OUT_OFFSET_RAW + ALIGN_ELEMS - 1U) / ALIGN_ELEMS) * ALIGN_ELEMS;
        constexpr uint32_t ROW_ELEMS = OUTPUT_WIDTH * DEPTH;
        constexpr uint32_t ROW_SEG_BYTES = ROW_SEG_ELEMS * sizeof(DT_X);
        constexpr int32_t EVENT_ROW0 = 2;
        constexpr int32_t EVENT_ROW1 = 3;
        constexpr int32_t EVENT_OUT = 0;
        static_assert(OUT_OFFSET_ELEMS + TILE_OUT_ELEMS <= UB_ELEMS,
                      "case5 row-tile UB layout exceeds UB_ELEMS");

        AscendC::LocalTensor<DT_X> rightRow0 = buffer;
        AscendC::LocalTensor<DT_X> leftRow0 = buffer[ROW_SEG_SLOT_ELEMS];
        AscendC::LocalTensor<DT_X> rightRow1 = buffer[ROW_BANK_ELEMS];
        AscendC::LocalTensor<DT_X> leftRow1 =
            buffer[ROW_BANK_ELEMS + ROW_SEG_SLOT_ELEMS];
        AscendC::LocalTensor<uint32_t> offsets =
            buffer[SOURCE_TOTAL_ELEMS].template ReinterpretCast<uint32_t>();
        AscendC::LocalTensor<DT_X> outBuffer = buffer[OUT_OFFSET_ELEMS];

        AscendC::DataCopy(
            offsets, case5OffsetGm_, TILE_OUT_ELEMS);

        AscendC::DataCopyExtParams inParams{1, ROW_SEG_BYTES, 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT_X> padParams{
            true, 0, 0, static_cast<DT_X>(0)};

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= OUTPUT_HEIGHT) {
            return;
        }

        uint32_t expandedH = blockIdx + 1U;
        uint32_t inputH = expandedH >> 1U;
        uint32_t batchLeft = (expandedH & 1U) << 1U;
        uint32_t batchRight = batchLeft + 1U;
        uint32_t rightBase =
            ((batchRight * INPUT_HEIGHT + inputH) * INPUT_WIDTH) * DEPTH;
        uint32_t leftBase =
            ((batchLeft * INPUT_HEIGHT + inputH) * INPUT_WIDTH + 1U) * DEPTH;
        AscendC::DataCopyPad(rightRow0, xGm_[rightBase], inParams, padParams);
        AscendC::DataCopyPad(leftRow0, xGm_[leftBase], inParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ROW0);

        for (uint32_t outputH = blockIdx, rowIter = 0U;
             outputH < OUTPUT_HEIGHT;
             outputH += coreCount_, ++rowIter) {
            const bool useRow1 = (rowIter & 1U) != 0U;
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                useRow1 ? EVENT_ROW1 : EVENT_ROW0);

            const uint32_t nextH = outputH + coreCount_;
            if (nextH < OUTPUT_HEIGHT) {
                expandedH = nextH + 1U;
                inputH = expandedH >> 1U;
                batchLeft = (expandedH & 1U) << 1U;
                batchRight = batchLeft + 1U;
                rightBase =
                    ((batchRight * INPUT_HEIGHT + inputH) * INPUT_WIDTH) * DEPTH;
                leftBase =
                    ((batchLeft * INPUT_HEIGHT + inputH) * INPUT_WIDTH + 1U) *
                    DEPTH;
                AscendC::LocalTensor<DT_X> nextRight =
                    useRow1 ? rightRow0 : rightRow1;
                AscendC::LocalTensor<DT_X> nextLeft =
                    useRow1 ? leftRow0 : leftRow1;
                AscendC::DataCopyPad(
                    nextRight, xGm_[rightBase], inParams, padParams);
                AscendC::DataCopyPad(
                    nextLeft, xGm_[leftBase], inParams, padParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
                    useRow1 ? EVENT_ROW0 : EVENT_ROW1);
            }

            AscendC::LocalTensor<DT_X> rightRow =
                useRow1 ? rightRow1 : rightRow0;

            for (uint32_t tileStart = 0U;
                 tileStart < OUTPUT_WIDTH;
                 tileStart += TILE_PIXELS) {
                const uint32_t remainPixels = OUTPUT_WIDTH - tileStart;
                const uint32_t curPixels =
                    remainPixels < TILE_PIXELS ? remainPixels : TILE_PIXELS;
                const uint32_t curElems = curPixels * DEPTH;
                const uint32_t sourcePixelBase = (tileStart >> 1U) * DEPTH;
                AscendC::Gather(
                    outBuffer, rightRow[sourcePixelBase], offsets, 0U, curElems);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_OUT);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_OUT);
                CopyOut(outputH * ROW_ELEMS + tileStart * DEPTH,
                        outBuffer, curElems);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void ProcessCase6(AscendC::LocalTensor<DT_X> buffer) {
        if (pixelCount_ != 2U || (pixelStart_ & 1U) != 0U) {
            ProcessGatherTiles(buffer);
            return;
        }

        const uint32_t pairIndex = pixelStart_ >> 1U;
        const uint32_t inputPixel =
            (((pairIndex >> 1U) & 1U) << 3U) +
            ((pairIndex >> 2U) << 1U) +
            (pairIndex & 1U);
        CopyCase6PairStrideIn(buffer, inputPixel);
        SyncMte2ToMte3(0);
        CopyOut(pixelStart_ * depth_, buffer, 2U * depth_);
    }

    __aicore__ inline void ProcessCase7(AscendC::LocalTensor<DT_X> buffer) {
        const uint32_t task = AscendC::GetBlockIdx();
        if (task >= 8U) {
            return;
        }
        const uint32_t cTile = cTileElements_ == 0U ? 8192U : cTileElements_;
        const uint32_t outputPixel = task >> 1U;
        const uint32_t cStart = (task & 1U) * cTile;
        const uint32_t offset = outputPixel * depth_ + cStart;
        AscendC::DataCopy(buffer, xGm_[offset], cTile);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(yGm_[offset], buffer, cTile);
    }

    __aicore__ inline void ProcessGatherTiles(AscendC::LocalTensor<DT_X> buffer) {
        const uint32_t pixelsPerTile = UB_ELEMS / depth_;
        if (pixelsPerTile == 0U) {
            ProcessDepthTiles(buffer);
            return;
        }
        for (uint32_t done = 0U; done < pixelCount_;) {
            const uint32_t remain = pixelCount_ - done;
            const uint32_t curPixels =
                remain < pixelsPerTile ? remain : pixelsPerTile;
            const uint32_t outputPixel = pixelStart_ + done;
            for (uint32_t i = 0U; i < curPixels; ++i) {
                const uint32_t inputPixel = MapInputPixel(outputPixel + i);
                AscendC::DataCopy(
                    buffer[i * depth_], xGm_[inputPixel * depth_], depth_);
            }
            SyncMte2ToMte3(0);
            CopyOut(outputPixel * depth_, buffer, curPixels * depth_);
            done += curPixels;
            if (done < pixelCount_) {
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void ProcessDepthTiles(AscendC::LocalTensor<DT_X> buffer) {
        const uint32_t alignElements = 32U / sizeof(DT_X);
        uint32_t cTile = (UB_ELEMS / alignElements) * alignElements;
        if (cTile == 0U) {
            cTile = UB_ELEMS;
        }
        for (uint32_t localPixel = 0U; localPixel < pixelCount_; ++localPixel) {
            const uint32_t outputPixel = pixelStart_ + localPixel;
            const uint32_t inputPixel = MapInputPixel(outputPixel);
            for (uint32_t cStart = 0U; cStart < depth_; cStart += cTile) {
                uint32_t cLen = depth_ - cStart;
                cLen = cLen < cTile ? cLen : cTile;
                CopyIn(buffer, inputPixel * depth_ + cStart, cLen);
                SyncMte2ToMte3(0);
                CopyOut(outputPixel * depth_ + cStart, buffer, cLen);
                if (cStart + cLen < depth_) {
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
            }
        }
    }

    __aicore__ inline uint32_t MapInputPixel(uint32_t outputPixel) const {
        const uint32_t outputPlane = outputHeight_ * outputWidth_;
        const uint32_t outBatch = outputPixel / outputPlane;
        const uint32_t spatial = outputPixel - outBatch * outputPlane;
        const uint32_t outputH = spatial / outputWidth_;
        const uint32_t outputW = spatial - outputH * outputWidth_;
        const uint32_t expandedH = outputH + cropTop_;
        const uint32_t expandedW = outputW + cropLeft_;
        const uint32_t inputH = DivBS(expandedH);
        const uint32_t inputW = DivBS(expandedW);
        const uint32_t blockH = ModBS(expandedH);
        const uint32_t blockW = ModBS(expandedW);
        const uint32_t inputBatch =
            (blockH * blockSize_ + blockW) * outputBatch_ + outBatch;
        return (inputBatch * inputHeight_ + inputH) * inputWidth_ + inputW;
    }

    __aicore__ inline void SyncMte2ToMte3(int32_t eventId) const {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
    }

    __aicore__ inline void CopyCase6PairStrideIn(
        AscendC::LocalTensor<DT_X> dst,
        uint32_t firstInputPixel) {
        AscendC::DataCopyExtParams copyParams{
            2,
            static_cast<uint32_t>(depth_ * sizeof(DT_X)),
            static_cast<uint32_t>(3U * depth_ * sizeof(DT_X)),
            0,
            0};
        AscendC::DataCopyPadExtParams<DT_X> padParams{
            true, 0, 0, static_cast<DT_X>(0)};
        AscendC::DataCopyPad(dst, xGm_[firstInputPixel * depth_], copyParams, padParams);
    }

    __aicore__ inline void CopyAlignedPhaseToStripe(
        AscendC::LocalTensor<DT_X> dst,
        uint32_t srcOffset,
        uint32_t pixelCount) {
        const uint32_t alignElements = 32U / sizeof(DT_X);
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(pixelCount),
            static_cast<uint32_t>(depth_ * sizeof(DT_X)),
            0,
            static_cast<uint32_t>((blockSize_ - 1U) * depth_ / alignElements),
            0};
        AscendC::DataCopyPadExtParams<DT_X> padParams{
            false, 0, 0, static_cast<DT_X>(0)};
        AscendC::DataCopyPad(dst, xGm_[srcOffset], copyParams, padParams);
    }

    __aicore__ inline void CopyPaddedColumnInFast(
        AscendC::LocalTensor<DT_X> dst,
        uint32_t srcOffset,
        uint32_t pixelCount,
        uint32_t depthBytes,
        uint32_t inputStrideBytes) {
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(pixelCount),
            depthBytes,
            inputStrideBytes,
            0,
            0};
        AscendC::DataCopyPadExtParams<DT_X> padParams{
            true, 0, 0, static_cast<DT_X>(0)};
        AscendC::DataCopyPad(dst, xGm_[srcOffset], copyParams, padParams);
    }

    __aicore__ inline void CopyPaddedColumnOutFast(
        uint32_t dstOffset,
        AscendC::LocalTensor<DT_X> src,
        uint32_t pixelCount,
        uint32_t depthBytes,
        uint32_t outputStrideBytes) {
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(pixelCount),
            depthBytes,
            0,
            outputStrideBytes,
            0};
        AscendC::DataCopyPad(yGm_[dstOffset], src, copyParams);
    }

    __aicore__ inline void CopyIn(
        AscendC::LocalTensor<DT_X> dst,
        uint32_t srcOffset,
        uint32_t len) {
        const uint32_t alignElements = 32U / sizeof(DT_X);
        if ((srcOffset % alignElements) == 0U && (len % alignElements) == 0U) {
            AscendC::DataCopy(dst, xGm_[srcOffset], len);
            return;
        }
        AscendC::DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(len * sizeof(DT_X)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT_X> padParams{
            true, 0, 0, static_cast<DT_X>(0)};
        AscendC::DataCopyPad(dst, xGm_[srcOffset], copyParams, padParams);
    }

    __aicore__ inline void CopyOut(
        uint32_t dstOffset,
        AscendC::LocalTensor<DT_X> src,
        uint32_t len) {
        const uint32_t alignElements = 32U / sizeof(DT_X);
        if ((dstOffset % alignElements) == 0U && (len % alignElements) == 0U) {
            AscendC::DataCopy(yGm_[dstOffset], src, len);
            return;
        }
        AscendC::DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(len * sizeof(DT_X)), 0, 0, 0};
        AscendC::DataCopyPad(yGm_[dstOffset], src, copyParams);
    }

    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;
    AscendC::GlobalTensor<uint32_t> case5OffsetGm_;
    uint32_t inputHeight_;
    uint32_t inputWidth_;
    uint32_t depth_;
    uint32_t outputBatch_;
    uint32_t outputHeight_;
    uint32_t outputWidth_;
    uint32_t blockSize_;
    uint32_t cropTop_;
    uint32_t cropLeft_;
    uint32_t coreCount_;
    uint32_t globalPixelCount_;
    uint32_t cTileElements_;
    uint32_t pixelStart_;
    uint32_t pixelCount_;
    bool fdivPow2_;
    uint32_t fdivShift_;
    uint32_t fdivMask_;

    __aicore__ inline uint32_t DivBS(uint32_t x) const {
        return fdivPow2_ ? (x >> fdivShift_) : (x / blockSize_);
    }

    __aicore__ inline uint32_t ModBS(uint32_t x) const {
        return fdivPow2_ ? (x & fdivMask_) : (x - (x / blockSize_) * blockSize_);
    }
};

template <typename DT_X, bool ROUTE_B0, bool ROUTE_B1, bool ROUTE_B2, bool ROUTE_B3>
__global__ __aicore__ void batch_to_space(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    KernelBatchToSpace<DT_X, ROUTE_B0, ROUTE_B1, ROUTE_B2, ROUTE_B3> op;
    op.InitFromTilingGm(x, y, tiling);
    op.Process();
}
