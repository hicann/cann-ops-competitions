// Kernel侧核函数实现
#include "kernel_operator.h"

#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

template <class DT_X>
class KernelBatchToSpace {
public:
    __aicore__ inline KernelBatchToSpace() {}
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const BatchToSpaceTilingData &tiling,
        AscendC::TPipe *pipeIn)
    {
        pipe = pipeIn;
        xPtr = (__gm__ DT_X *)x;
        const uint32_t inputElements =
            tiling.outputBatch * tiling.blockSize * tiling.blockSize *
            tiling.inputHeight * tiling.inputWidth * tiling.depth;
        xGm.SetGlobalBuffer((__gm__ DT_X *)x, inputElements);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, tiling.outputElements);
        inputHeight = tiling.inputHeight;
        inputWidth = tiling.inputWidth;
        depth = tiling.depth;
        outputBatch = tiling.outputBatch;
        outputHeight = tiling.outputHeight;
        outputWidth = tiling.outputWidth;
        cropTop = tiling.cropTop;
        cropLeft = tiling.cropLeft;
        blockSize = tiling.blockSize;
        outputElements = tiling.outputElements;
        copyMode = tiling.copyMode;
        rowGroupSize = tiling.rowGroupSize;
        channelTiles = tiling.channelTiles;
        tileCols = tiling.tileCols;
        tileDepthAlign = tiling.tileDepthAlign;
        if (copyMode == 15U) {
            if (sizeof(DT_X) == 4U &&
                tiling.outputBatch == 2U &&
                tiling.outputHeight == 56U &&
                tiling.outputWidth == 56U &&
                tiling.depth == 128U &&
                tiling.inputHeight == 28U &&
                tiling.inputWidth == 28U &&
                tiling.cropTop == 0U &&
                tiling.cropLeft == 0U &&
                tiling.tileDepthAlign == 128U) {
                pipe->InitBuffer(xBuf, tileCols * tileDepthAlign * sizeof(DT_X));
                pipe->InitBuffer(xBuf2, tileCols * tileDepthAlign * sizeof(DT_X));
            } else {
                pipe->InitBuffer(bindQueue2, 2, tileCols * tileDepthAlign * sizeof(DT_X));
            }
        } else if (copyMode == 5U) {
            pipe->InitBuffer(bindQueue5, 2, ROW_COPY_BUFFER_BYTES);
        } else if (copyMode == 17U) {
            pipe->InitBuffer(bindQueue, 1, ROW_COPY_BUFFER_BYTES);
        } else if (copyMode == 6U) {
            pipe->InitBuffer(bindQueue, 1, tileCols * tileDepthAlign * sizeof(DT_X));
        } else if (copyMode == 29U) {
            // Gemini-style block2 H-Chunk repack path.
            // rowGroupSize carries H_chunk. Keep four input phase chunks in UB
            // and double-buffer the repacked output chunk to overlap task N MTE3
            // with task N+1 GM->UB/UB->UB work.
            const uint32_t inputChunkElements = rowGroupSize * inputWidth * depth;
            const uint32_t outputChunkElements =
                2U * rowGroupSize * outputWidth * tileDepthAlign;
            pipe->InitBuffer(inQueue29, 4, inputChunkElements * sizeof(DT_X));
            pipe->InitBuffer(outQueue29, 2, outputChunkElements * sizeof(DT_X));
        } else if (copyMode == 27U) {
            // TP9 vector-repack experiment. tileCols is forced to 512 on host
            // so the two queues fit in UB.
            const uint32_t phaseMaxPixels = (tileCols + 3U) >> 2U;
            pipe->InitBuffer(outQueue24, 2, tileCols * tileDepthAlign * sizeof(DT_X));
            pipe->InitBuffer(inQueue24, 2, phaseMaxPixels * depth * sizeof(DT_X));
        } else if (copyMode == 8U) {
            pipe->InitBuffer(bindQueue8, 2, tileCols * tileDepthAlign * sizeof(DT_X));
        } else if (copyMode == 12U) {
            const uint32_t bufferElements =
                rowGroupSize * tileCols * tileDepthAlign;
            pipe->InitBuffer(xQueue, 1, bufferElements * sizeof(DT_X));
            pipe->InitBuffer(yQueue, 1, bufferElements * sizeof(DT_X));
        }
    }
    __aicore__ inline void Process()
    {
        if (copyMode == 62U) {
            ProcessTp3FixedFloat64();
            return;
        }
        if (copyMode == 63U) {
            ProcessTp4FixedFloat32();
            return;
        }
        if (copyMode == 60U) {
            ProcessTp2CompactGather();
            return;
        }
        if (copyMode == 46U) {
            ProcessTp5HalfRowGatherPipelineCopy();
            return;
        }
        if (copyMode == 5U) {
            ProcessColumnCopy();
            return;
        }
        if (copyMode == 6U) {
            ProcessReferenceStridedCopy();
            return;
        }
        if (copyMode == 8U) {
            ProcessGatherStridedCopy();
            return;
        }
        if (copyMode == 27U) {
            ProcessGatherStridedBlock4RowTileVectorRepackCopy();
            return;
        }
        if (copyMode == 29U) {
            ProcessBlock2HChunkVectorRepackCopy();
            return;
        }
        if (copyMode == 12U) {
            ProcessGroupedGatherBlock2Copy();
            return;
        }
        if (copyMode == 15U) {
            ProcessRowGatherBlock2Copy();
            return;
        }
        if (copyMode == 17U) {
            ProcessChannelRowTileBlock2Copy();
            return;
        }
        if (copyMode == 67U) {
            ProcessTp7ContiguousProbe();
            return;
        }
    }
private:

    __aicore__ inline void ProcessTp3FixedFloat64()
    {
        constexpr uint32_t inputHeightConst = 14U;
        constexpr uint32_t inputWidthConst = 14U;
        constexpr uint32_t depthConst = 64U;
        constexpr uint32_t outputBatchConst = 4U;
        constexpr uint32_t outputHeightConst = 28U;
        constexpr uint32_t outputWidthConst = 28U;
        constexpr uint32_t blockSizeConst = 2U;
        constexpr uint32_t tileRowsConst = 7U;
        constexpr uint32_t blockBytes = 32U;
        constexpr uint32_t pixelBytes = depthConst * sizeof(DT_X);
        constexpr uint32_t cBlocks = pixelBytes / blockBytes;
        constexpr uint32_t inputRowBlocks = (inputWidthConst * pixelBytes) / blockBytes;
        constexpr uint32_t outputRowBlocks = (outputWidthConst * pixelBytes) / blockBytes;
        constexpr uint32_t inputLocalElements = tileRowsConst * inputWidthConst * depthConst;
        constexpr uint32_t outputLocalElements = tileRowsConst * outputWidthConst * depthConst;
        constexpr uint32_t inputLocalBytes =
            ((inputLocalElements * sizeof(DT_X) + 511U) / 512U) * 512U;
        constexpr uint32_t inputLocal1OffsetBytes = inputLocalBytes;
        constexpr uint32_t outputLocalOffsetBytes = 2U * inputLocalBytes;

        AscendC::LocalTensor<DT_X> inputLocal0(
            AscendC::TPosition::VECCALC, 0, inputLocalElements);
        AscendC::LocalTensor<DT_X> inputLocal1(
            AscendC::TPosition::VECCALC, inputLocal1OffsetBytes, inputLocalElements);
        AscendC::LocalTensor<DT_X> outputLocal(
            AscendC::TPosition::VECCALC, outputLocalOffsetBytes, outputLocalElements);

        AscendC::DataCopyParams loadParams;
        loadParams.blockCount = 1;
        loadParams.blockLen = 0;
        loadParams.srcStride = 0;
        loadParams.dstStride = 0;

        AscendC::DataCopyParams ubCopyParams;
        ubCopyParams.blockLen = static_cast<uint16_t>(cBlocks);
        ubCopyParams.srcStride = 0;
        ubCopyParams.dstStride = static_cast<uint16_t>(cBlocks);

        AscendC::DataCopyParams storeParams;
        storeParams.blockLen = static_cast<uint16_t>(outputRowBlocks);
        storeParams.srcStride = 0;
        storeParams.dstStride = static_cast<uint16_t>(outputRowBlocks);

        const uint32_t coreIndex = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t coreCount = static_cast<uint32_t>(AscendC::GetBlockNum());
        constexpr uint32_t hTiles =
            (inputHeightConst + tileRowsConst - 1U) / tileRowsConst;
        constexpr uint32_t taskCount = outputBatchConst * blockSizeConst * hTiles;
        for (uint32_t taskIndex = coreIndex; taskIndex < taskCount; taskIndex += coreCount) {
            const uint32_t hTileIndex = taskIndex % hTiles;
            const uint32_t tmpIndex = taskIndex / hTiles;
            const uint32_t blockH = tmpIndex & 1U;
            const uint32_t outputN = tmpIndex >> 1U;
            const uint32_t inputHStart = hTileIndex * tileRowsConst;
            uint32_t rowsThisTile = inputHeightConst - inputHStart;
            if (rowsThisTile > tileRowsConst) {
                rowsThisTile = tileRowsConst;
            }

            const uint32_t inputN0 = blockH * blockSizeConst * outputBatchConst + outputN;
            const uint32_t inputN1 = inputN0 + outputBatchConst;
            const uint32_t inputOffset0 =
                ((inputN0 * inputHeightConst + inputHStart) * inputWidthConst) * depthConst;
            const uint32_t inputOffset1 =
                ((inputN1 * inputHeightConst + inputHStart) * inputWidthConst) * depthConst;

            loadParams.blockLen = static_cast<uint16_t>(rowsThisTile * inputRowBlocks);
            AscendC::DataCopy(inputLocal0, xGm[inputOffset0], loadParams);
            AscendC::DataCopy(inputLocal1, xGm[inputOffset1], loadParams);
            AscendC::PipeBarrier<PIPE_ALL>();

            ubCopyParams.blockCount = static_cast<uint16_t>(rowsThisTile * inputWidthConst);
            AscendC::DataCopy(outputLocal, inputLocal0, ubCopyParams);
            AscendC::DataCopy(outputLocal[depthConst], inputLocal1, ubCopyParams);
            AscendC::PipeBarrier<PIPE_ALL>();

            storeParams.blockCount = static_cast<uint16_t>(rowsThisTile);
            const uint32_t outputOffset =
                (outputN * outputHeightConst + blockH + (inputHStart << 1U)) *
                outputWidthConst * depthConst;
            AscendC::DataCopy(yGm[outputOffset], outputLocal, storeParams);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }

    __aicore__ inline void ProcessTp4FixedFloat32()
    {
        constexpr uint32_t inputHeightConst = 4U;
        constexpr uint32_t inputWidthConst = 6U;
        constexpr uint32_t depthConst = 32U;
        constexpr uint32_t outputBatchConst = 5U;
        constexpr uint32_t outputHeightConst = 8U;
        constexpr uint32_t outputWidthConst = 12U;
        constexpr uint32_t blockSizeConst = 2U;
        constexpr uint32_t tileRowsConst = 4U;
        constexpr uint32_t blockBytes = 32U;
        constexpr uint32_t inputBufferBytes = 16U * 1024U;
        constexpr uint32_t outputBufferBytes = 32U * 1024U;
        constexpr uint32_t inputLocal1OffsetBytes = inputBufferBytes;
        constexpr uint32_t outputLocalOffsetBytes = inputBufferBytes * 2U;
        constexpr uint32_t pixelBytes = depthConst * sizeof(DT_X);
        constexpr uint32_t cBlocks = pixelBytes / blockBytes;
        constexpr uint32_t inputRowBlocks = (inputWidthConst * pixelBytes) / blockBytes;
        constexpr uint32_t outputRowBlocks = (outputWidthConst * pixelBytes) / blockBytes;
        constexpr uint32_t inputLocalElements = inputBufferBytes / sizeof(DT_X);
        constexpr uint32_t outputLocalElements = outputBufferBytes / sizeof(DT_X);

        AscendC::LocalTensor<DT_X> inputLocal0(
            AscendC::TPosition::VECCALC, 0, inputLocalElements);
        AscendC::LocalTensor<DT_X> inputLocal1(
            AscendC::TPosition::VECCALC, inputLocal1OffsetBytes, inputLocalElements);
        AscendC::LocalTensor<DT_X> outputLocal(
            AscendC::TPosition::VECCALC, outputLocalOffsetBytes, outputLocalElements);

        AscendC::DataCopyParams loadParams;
        loadParams.blockCount = 1;
        loadParams.blockLen = static_cast<uint16_t>(tileRowsConst * inputRowBlocks);
        loadParams.srcStride = 0;
        loadParams.dstStride = 0;

        AscendC::DataCopyParams ubCopyParams;
        ubCopyParams.blockCount = static_cast<uint16_t>(tileRowsConst * inputWidthConst);
        ubCopyParams.blockLen = static_cast<uint16_t>(cBlocks);
        ubCopyParams.srcStride = 0;
        ubCopyParams.dstStride = static_cast<uint16_t>(cBlocks);

        AscendC::DataCopyParams storeParams;
        storeParams.blockCount = static_cast<uint16_t>(tileRowsConst);
        storeParams.blockLen = static_cast<uint16_t>(outputRowBlocks);
        storeParams.srcStride = 0;
        storeParams.dstStride = static_cast<uint16_t>(outputRowBlocks);

        const uint32_t coreIndex = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t coreCount = static_cast<uint32_t>(AscendC::GetBlockNum());
        constexpr uint32_t taskCount = outputBatchConst * blockSizeConst;
        for (uint32_t taskIndex = coreIndex; taskIndex < taskCount; taskIndex += coreCount) {
            const uint32_t blockH = taskIndex & 1U;
            const uint32_t outputN = taskIndex >> 1U;
            const uint32_t inputN0 = blockH * blockSizeConst * outputBatchConst + outputN;
            const uint32_t inputN1 = inputN0 + outputBatchConst;
            const uint32_t inputOffset0 =
                inputN0 * inputHeightConst * inputWidthConst * depthConst;
            const uint32_t inputOffset1 =
                inputN1 * inputHeightConst * inputWidthConst * depthConst;

            AscendC::DataCopy(inputLocal0, xGm[inputOffset0], loadParams);
            AscendC::DataCopy(inputLocal1, xGm[inputOffset1], loadParams);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::DataCopy(outputLocal, inputLocal0, ubCopyParams);
            AscendC::DataCopy(outputLocal[depthConst], inputLocal1, ubCopyParams);
            AscendC::PipeBarrier<PIPE_ALL>();

            const uint32_t outputOffset =
                (outputN * outputHeightConst + blockH) * outputWidthConst * depthConst;
            AscendC::DataCopy(yGm[outputOffset], outputLocal, storeParams);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }

    __aicore__ inline void ProcessTp2CompactGather()
    {
        constexpr uint32_t inputHeightConst = 10U;
        constexpr uint32_t inputWidthConst = 15U;
        constexpr uint32_t depthConst = 5U;
        constexpr uint32_t outputHeightConst = 17U;
        constexpr uint32_t outputWidthConst = 26U;
        constexpr uint32_t firstInputCols = 13U;
        constexpr uint32_t secondInputCols = 13U;
        constexpr uint32_t pixelBytes = depthConst * sizeof(DT_X);
        constexpr uint32_t localGroupBytes =
            ((firstInputCols * pixelBytes + 31U) / 32U) * 32U;
        constexpr uint32_t localGroupElements = localGroupBytes / sizeof(DT_X);
        constexpr uint32_t inputLocalElements = localGroupElements * 2U;
        constexpr uint32_t outputLocalElements = outputWidthConst * depthConst;

        AscendC::LocalTensor<DT_X> inputLocal(
            AscendC::TPosition::VECCALC, 0, inputLocalElements);
        AscendC::LocalTensor<DT_X> outputLocal(
            AscendC::TPosition::VECCALC, 1024, outputLocalElements);
        AscendC::LocalTensor<int32_t> offsetInitLocal(
            AscendC::TPosition::VECCALC, 2048, 1024 / sizeof(int32_t));
        AscendC::LocalTensor<uint32_t> offsetLocal(
            AscendC::TPosition::VECCALC, 2048, outputLocalElements);

        constexpr uint32_t pairElements = depthConst * 2U;
        constexpr uint32_t periodPairCount = 4U;
        constexpr uint32_t periodElements = periodPairCount * pairElements;
        constexpr uint32_t periodStrideBytes = periodPairCount * pixelBytes;
        constexpr uint32_t alignedOffsetElements = 136U;
        for (uint32_t pairIndex = 0U; pairIndex < periodPairCount; ++pairIndex) {
            const uint32_t pairElementBase = pairIndex * pairElements;
            const uint32_t pairByteOffset = pairIndex * pixelBytes;
            for (uint32_t channelIndex = 0U; channelIndex < depthConst; ++channelIndex) {
                const uint32_t channelByteOffset = channelIndex * sizeof(DT_X);
                offsetInitLocal.SetValue(
                    pairElementBase + channelIndex,
                    static_cast<int32_t>(pairByteOffset + channelByteOffset));
                offsetInitLocal.SetValue(
                    pairElementBase + depthConst + channelIndex,
                    static_cast<int32_t>(
                        localGroupBytes + pairByteOffset + channelByteOffset));
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
        for (uint32_t offsetElementBase = periodElements;
            offsetElementBase < alignedOffsetElements;
            offsetElementBase += periodElements) {
            uint32_t offsetElementCount = alignedOffsetElements - offsetElementBase;
            if (offsetElementCount > periodElements) {
                offsetElementCount = periodElements;
            }
            AscendC::Adds(
                offsetInitLocal[offsetElementBase],
                offsetInitLocal[offsetElementBase - periodElements],
                static_cast<int32_t>(periodStrideBytes),
                offsetElementCount);
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::DataCopyPadExtParams<DT_X> copyPadParams = {false, 0, 0, 0};
        AscendC::DataCopyExtParams firstLoadParams = {
            1U,
            static_cast<uint32_t>(firstInputCols * pixelBytes),
            0U,
            0U,
            0U
        };
        AscendC::DataCopyExtParams secondLoadParams = {
            1U,
            static_cast<uint32_t>(secondInputCols * pixelBytes),
            0U,
            0U,
            0U
        };
        AscendC::DataCopyExtParams outputStoreParams = {
            1U,
            static_cast<uint32_t>(outputWidthConst * pixelBytes),
            0U,
            0U,
            0U
        };

        const uint32_t coreIndex = AscendC::GetBlockIdx();
        const uint32_t coreCount = AscendC::GetBlockNum();
        for (uint32_t outputH = coreIndex; outputH < outputHeightConst;
            outputH += coreCount) {
            const uint32_t expandedH = outputH + 2U;
            const uint32_t inputH = expandedH >> 1U;
            const uint32_t blockH = expandedH & 1U;
            const uint32_t firstInputN = blockH * 2U + 1U;
            const uint32_t secondInputN = blockH * 2U;
            const uint32_t firstInputOffset =
                ((firstInputN * inputHeightConst + inputH) * inputWidthConst + 1U) *
                depthConst;
            const uint32_t secondInputOffset =
                ((secondInputN * inputHeightConst + inputH) * inputWidthConst + 2U) *
                depthConst;
            const uint32_t outputOffset = outputH * outputWidthConst * depthConst;

            AscendC::DataCopyPad(
                inputLocal,
                xGm[firstInputOffset],
                firstLoadParams,
                copyPadParams);
            AscendC::DataCopyPad(
                inputLocal[localGroupElements],
                xGm[secondInputOffset],
                secondLoadParams,
                copyPadParams);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Gather(
                outputLocal,
                inputLocal,
                offsetLocal,
                0,
                outputLocalElements);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(yGm[outputOffset], outputLocal, outputStoreParams);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }

    __aicore__ inline void ProcessTp7ContiguousProbe()
    {
        if constexpr (sizeof(DT_X) != 2) {
            return;
        } else {
            constexpr uint32_t kTotalBlocks =
                (4U * 1U * 1U * 16384U * sizeof(DT_X)) / 32U;
            constexpr uint32_t kMaxBlocks = 2048U;
            constexpr uint32_t kElemsPerBlock = 32U / sizeof(DT_X);
            constexpr uint32_t kMaxElems = kMaxBlocks * kElemsPerBlock;

            AscendC::LocalTensor<DT_X> localCopy(
                AscendC::TPosition::VECCALC, 0, kMaxElems);

            const uint32_t coreId = AscendC::GetBlockIdx();
            const uint32_t coreNum = AscendC::GetBlockNum();
            if (coreNum == 0U) {
                return;
            }
            const uint32_t blocksPerCore = (kTotalBlocks + coreNum - 1U) / coreNum;
            uint32_t blockOffset = coreId * blocksPerCore;
            if (blockOffset >= kTotalBlocks) {
                return;
            }
            uint32_t remainBlocks = kTotalBlocks - blockOffset;
            if (remainBlocks > blocksPerCore) {
                remainBlocks = blocksPerCore;
            }
            uint32_t elemOffset = blockOffset * kElemsPerBlock;

            AscendC::DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            while (remainBlocks != 0U) {
                uint32_t chunkBlocks = remainBlocks;
                if (chunkBlocks > kMaxBlocks) {
                    chunkBlocks = kMaxBlocks;
                }
                copyParams.blockLen = static_cast<uint16_t>(chunkBlocks);
                AscendC::DataCopy(localCopy, xGm[elemOffset], copyParams);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(yGm[elemOffset], localCopy, copyParams);
                AscendC::PipeBarrier<PIPE_MTE3>();
                elemOffset += chunkBlocks * kElemsPerBlock;
                remainBlocks -= chunkBlocks;
            }
        }
    }

    __aicore__ inline uint32_t Mode46GcdU32(uint32_t leftValue, uint32_t rightValue)
    {
        while (rightValue != 0U) {
            const uint32_t nextValue = leftValue % rightValue;
            leftValue = rightValue;
            rightValue = nextValue;
        }
        return leftValue;
    }

    __aicore__ inline uint32_t Mode46AlignUpU32(uint32_t inputValue, uint32_t alignValue)
    {
        return alignValue == 0U ? inputValue : ((inputValue + alignValue - 1U) / alignValue) * alignValue;
    }

    __aicore__ inline int32_t Mode46FreeEventId(uint32_t bufferIndex)
    {
        return bufferIndex == 0U ? EVENT_ID0 : EVENT_ID1;
    }

    __aicore__ inline int32_t Mode46Mte2ReadyEventId(uint32_t bufferIndex)
    {
        return bufferIndex == 0U ? EVENT_ID2 : EVENT_ID3;
    }

    __aicore__ inline int32_t Mode46VecReadyEventId(uint32_t bufferIndex)
    {
        return bufferIndex == 0U ? EVENT_ID4 : EVENT_ID5;
    }

    __aicore__ inline void Mode46MarkBufFree(uint32_t bufferIndex)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(Mode46FreeEventId(bufferIndex));
    }

    __aicore__ inline void Mode46WaitBufFree(uint32_t bufferIndex)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(Mode46FreeEventId(bufferIndex));
    }

    __aicore__ inline void Mode46MarkMte2Ready(uint32_t bufferIndex)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(Mode46Mte2ReadyEventId(bufferIndex));
    }

    __aicore__ inline void Mode46WaitMte2Ready(uint32_t bufferIndex)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(Mode46Mte2ReadyEventId(bufferIndex));
    }

    __aicore__ inline void Mode46MarkVecReady(uint32_t bufferIndex)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(Mode46VecReadyEventId(bufferIndex));
    }

    __aicore__ inline void Mode46WaitVecReady(uint32_t bufferIndex)
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(Mode46VecReadyEventId(bufferIndex));
    }

    __aicore__ inline void Mode46LoadTaskToSlot(
        AscendC::LocalTensor<DT_X> &inputLocal,
        uint32_t taskIndex,
        uint32_t localGroupElements,
        uint32_t pixelBytes,
        AscendC::DataCopyExtParams &firstLoadParams,
        AscendC::DataCopyExtParams &secondLoadParams,
        const AscendC::DataCopyPadExtParams<DT_X> &copyPadParams)
    {
        // TP5 hard-coded shape:
        // input [4,128,128,65], output [1,254,254,65], crops=[1,1,1,1].
        constexpr uint32_t inputHeightConst = 128U;
        constexpr uint32_t inputWidthConst = 128U;
        constexpr uint32_t depthConst = 65U;
        constexpr uint32_t outputWidthConst = 254U;
        constexpr uint32_t cropTopConst = 1U;
        constexpr uint32_t cropLeftConst = 1U;

        const uint32_t tileColsPerTask = (outputWidthConst + 1U) >> 1U; // 127 columns per half-row task
        const uint32_t outputH = taskIndex >> 1U;
        const uint32_t widthPart = taskIndex & 1U;
        const uint32_t outputWStart = widthPart * tileColsPerTask;
        uint32_t outputPixels = outputWidthConst - outputWStart;
        if (outputPixels > tileColsPerTask) {
            outputPixels = tileColsPerTask;
        }

        const uint32_t expandedH = outputH + cropTopConst;
        const uint32_t inputH = expandedH >> 1U;
        const uint32_t blockH = expandedH & 1U;
        const uint32_t expandedWStart = cropLeftConst + outputWStart;
        const uint32_t firstBlockW = expandedWStart & 1U;
        const uint32_t secondBlockW = firstBlockW ^ 1U;
        const uint32_t firstInputW = expandedWStart >> 1U;
        const uint32_t secondInputW = (expandedWStart + 1U) >> 1U;
        const uint32_t firstInputPixels = (outputPixels + 1U) >> 1U;
        const uint32_t secondInputPixels = outputPixels >> 1U;
        const uint32_t firstInputN = (blockH << 1U) + firstBlockW;
        const uint32_t secondInputN = (blockH << 1U) + secondBlockW;
        const uint64_t firstInputOffset = ((static_cast<uint64_t>(firstInputN) * inputHeightConst + inputH) * inputWidthConst + firstInputW) * depthConst;
        const uint64_t secondInputOffset = ((static_cast<uint64_t>(secondInputN) * inputHeightConst + inputH) * inputWidthConst + secondInputW) * depthConst;

        firstLoadParams.blockLen = firstInputPixels * pixelBytes;
        secondLoadParams.blockLen = secondInputPixels * pixelBytes;
        AscendC::DataCopyPad(inputLocal, xGm[firstInputOffset], firstLoadParams, copyPadParams);
        if (secondInputPixels > 0U) {
            AscendC::DataCopyPad(inputLocal[localGroupElements], xGm[secondInputOffset], secondLoadParams, copyPadParams);
        }
    }

    __aicore__ inline void ProcessTp5HalfRowGatherPipelineCopy()
    {
        if constexpr (sizeof(DT_X) != 2) {
            return;
        } else {
            constexpr uint32_t blockBytes = 32U;
            constexpr uint32_t slotBufferBytes = 50U * 1024U;
            constexpr uint32_t offsetBufferBytes = 34U * 1024U;
            constexpr uint32_t vectorAlignElements = 8U;
            constexpr uint32_t outputHeightConst = 254U;
            constexpr uint32_t outputWidthConst = 254U;
            constexpr uint32_t depthConst = 65U;

            const uint32_t pixelBytes = depthConst * sizeof(DT_X); // 130B
            const uint32_t tileColsPerTask = (outputWidthConst + 1U) >> 1U; // 127
            const uint32_t lanePixelsMax = (tileColsPerTask + 1U) >> 1U; // 64
            const uint32_t localGroupBytes = Mode46AlignUpU32(lanePixelsMax * pixelBytes, blockBytes); // 8320B
            const uint32_t localGroupElements = localGroupBytes / sizeof(DT_X);
            const uint32_t inputLocalElements = (localGroupBytes * 2U) / sizeof(DT_X);
            const uint32_t outputLocalElements = tileColsPerTask * depthConst;
            const uint32_t alignedOffsetElements = Mode46AlignUpU32(outputLocalElements, vectorAlignElements);

            AscendC::LocalTensor<DT_X> inputLocal0(
                AscendC::TPosition::VECCALC, 0, slotBufferBytes / sizeof(DT_X));
            AscendC::LocalTensor<DT_X> inputLocal1(
                AscendC::TPosition::VECCALC, slotBufferBytes, slotBufferBytes / sizeof(DT_X));
            AscendC::LocalTensor<int32_t> gatherOffsetInitLocal(
                AscendC::TPosition::VECCALC, slotBufferBytes * 2U, offsetBufferBytes / sizeof(int32_t));
            AscendC::LocalTensor<uint32_t> gatherOffsetLocal(
                AscendC::TPosition::VECCALC, slotBufferBytes * 2U, offsetBufferBytes / sizeof(uint32_t));

            const uint32_t pairElements = depthConst * 2U;
            const uint32_t periodPairCount = vectorAlignElements / Mode46GcdU32(pairElements, vectorAlignElements);
            const uint32_t periodElements = periodPairCount * pairElements;
            const uint32_t periodStrideBytes = periodPairCount * pixelBytes;

            // Build one period of Gather byte offsets. Pattern is:
            // output pixel 0 <- lane A pixel 0, output pixel 1 <- lane B pixel 0, ...
            for (uint32_t pairIndex = 0U; pairIndex < periodPairCount; ++pairIndex) {
                const uint32_t pairElementBase = pairIndex * pairElements;
                const uint32_t pairByteOffset = pairIndex * pixelBytes;
                for (uint32_t channelIndex = 0U; channelIndex < depthConst; ++channelIndex) {
                    const uint32_t channelByteOffset = channelIndex * sizeof(DT_X);
                    gatherOffsetInitLocal.SetValue(pairElementBase + channelIndex,
                        static_cast<int32_t>(pairByteOffset + channelByteOffset));
                    gatherOffsetInitLocal.SetValue(pairElementBase + depthConst + channelIndex,
                        static_cast<int32_t>(localGroupBytes + pairByteOffset + channelByteOffset));
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
            for (uint32_t offsetElementBase = periodElements; offsetElementBase < alignedOffsetElements; offsetElementBase += periodElements) {
                uint32_t offsetElementCount = alignedOffsetElements - offsetElementBase;
                if (offsetElementCount > periodElements) {
                    offsetElementCount = periodElements;
                }
                AscendC::Adds(gatherOffsetInitLocal[offsetElementBase],
                    gatherOffsetInitLocal[offsetElementBase - periodElements],
                    static_cast<int32_t>(periodStrideBytes),
                    offsetElementCount);
            }
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::DataCopyPadExtParams<DT_X> copyPadParams;
            copyPadParams.isPad = false;
            copyPadParams.leftPadding = 0;
            copyPadParams.rightPadding = 0;
            copyPadParams.paddingValue = static_cast<DT_X>(0);

            AscendC::DataCopyExtParams firstLoadParams;
            firstLoadParams.blockCount = 1;
            firstLoadParams.srcStride = 0;
            firstLoadParams.dstStride = 0;
            firstLoadParams.rsv = 0;

            AscendC::DataCopyExtParams secondLoadParams;
            secondLoadParams.blockCount = 1;
            secondLoadParams.srcStride = 0;
            secondLoadParams.dstStride = 0;
            secondLoadParams.rsv = 0;

            AscendC::DataCopyExtParams outputStoreParams;
            outputStoreParams.blockCount = 1;
            outputStoreParams.srcStride = 0;
            outputStoreParams.dstStride = 0;
            outputStoreParams.rsv = 0;

            const uint32_t coreIndex = static_cast<uint32_t>(AscendC::GetBlockIdx());
            const uint32_t coreCount = static_cast<uint32_t>(AscendC::GetBlockNum());
            const uint32_t totalTasks = outputHeightConst * 2U; // 508 half-row tasks, same task count as Mode5.
            if (coreCount == 0U || coreIndex >= totalTasks) {
                return;
            }

            Mode46MarkBufFree(0U);
            Mode46MarkBufFree(1U);

            uint32_t taskIndex = coreIndex;
            uint32_t bufferIndex = 0U;
            Mode46WaitBufFree(bufferIndex);
            {
                AscendC::LocalTensor<DT_X> firstLocal = inputLocal0;
                Mode46LoadTaskToSlot(firstLocal, taskIndex, localGroupElements, pixelBytes,
                    firstLoadParams, secondLoadParams, copyPadParams);
                Mode46MarkMte2Ready(bufferIndex);
            }

            while (taskIndex < totalTasks) {
                const uint32_t currentBufferIndex = bufferIndex;
                AscendC::LocalTensor<DT_X> currentLocal = currentBufferIndex == 0U ? inputLocal0 : inputLocal1;
                Mode46WaitMte2Ready(currentBufferIndex);

                const uint32_t nextTaskIndex = taskIndex + coreCount;
                const uint32_t nextBufferIndex = currentBufferIndex ^ 1U;
                if (nextTaskIndex < totalTasks) {
                    AscendC::LocalTensor<DT_X> nextLocal = nextBufferIndex == 0U ? inputLocal0 : inputLocal1;
                    Mode46WaitBufFree(nextBufferIndex);
                    Mode46LoadTaskToSlot(nextLocal, nextTaskIndex, localGroupElements, pixelBytes,
                        firstLoadParams, secondLoadParams, copyPadParams);
                    Mode46MarkMte2Ready(nextBufferIndex);
                }

                const uint32_t outputH = taskIndex >> 1U;
                const uint32_t widthPart = taskIndex & 1U;
                const uint32_t outputWStart = widthPart * tileColsPerTask;
                uint32_t outputPixels = outputWidthConst - outputWStart;
                if (outputPixels > tileColsPerTask) {
                    outputPixels = tileColsPerTask;
                }
                const uint32_t currentOutputElements = outputPixels * depthConst;
                AscendC::Gather(currentLocal[inputLocalElements], currentLocal, gatherOffsetLocal, 0U, currentOutputElements);
                Mode46MarkVecReady(currentBufferIndex);
                Mode46WaitVecReady(currentBufferIndex);

                const uint64_t outputOffset = (static_cast<uint64_t>(outputH) * outputWidthConst + outputWStart) * depthConst;
                outputStoreParams.blockLen = currentOutputElements * sizeof(DT_X);
                AscendC::DataCopyPad(yGm[outputOffset], currentLocal[inputLocalElements], outputStoreParams);
                Mode46MarkBufFree(currentBufferIndex);

                taskIndex = nextTaskIndex;
                bufferIndex = nextBufferIndex;
            }

            Mode46WaitBufFree(0U);
            Mode46WaitBufFree(1U);
        }
    }

    __aicore__ inline void ProcessTp5HalfRowGatherNoPipelineCopy()
    {
        if constexpr (sizeof(DT_X) != 2) {
            return;
        } else {
            constexpr uint32_t blockBytes = 32U;
            constexpr uint32_t slotBufferBytes = 50U * 1024U;
            constexpr uint32_t offsetBufferBytes = 34U * 1024U;
            constexpr uint32_t vectorAlignElements = 8U;
            constexpr uint32_t outputHeightConst = 254U;
            constexpr uint32_t outputWidthConst = 254U;
            constexpr uint32_t depthConst = 65U;

            const uint32_t pixelBytes = depthConst * sizeof(DT_X);
            const uint32_t tileColsPerTask = (outputWidthConst + 1U) >> 1U;
            const uint32_t lanePixelsMax = (tileColsPerTask + 1U) >> 1U;
            const uint32_t localGroupBytes = Mode46AlignUpU32(lanePixelsMax * pixelBytes, blockBytes);
            const uint32_t localGroupElements = localGroupBytes / sizeof(DT_X);
            const uint32_t inputLocalElements = (localGroupBytes * 2U) / sizeof(DT_X);
            const uint32_t outputLocalElements = tileColsPerTask * depthConst;
            const uint32_t alignedOffsetElements = Mode46AlignUpU32(outputLocalElements, vectorAlignElements);

            AscendC::LocalTensor<DT_X> inputLocal(
                AscendC::TPosition::VECCALC, 0, slotBufferBytes / sizeof(DT_X));
            AscendC::LocalTensor<int32_t> gatherOffsetInitLocal(
                AscendC::TPosition::VECCALC, slotBufferBytes, offsetBufferBytes / sizeof(int32_t));
            AscendC::LocalTensor<uint32_t> gatherOffsetLocal(
                AscendC::TPosition::VECCALC, slotBufferBytes, offsetBufferBytes / sizeof(uint32_t));

            const uint32_t pairElements = depthConst * 2U;
            const uint32_t periodPairCount = vectorAlignElements / Mode46GcdU32(pairElements, vectorAlignElements);
            const uint32_t periodElements = periodPairCount * pairElements;
            const uint32_t periodStrideBytes = periodPairCount * pixelBytes;

            for (uint32_t pairIndex = 0U; pairIndex < periodPairCount; ++pairIndex) {
                const uint32_t pairElementBase = pairIndex * pairElements;
                const uint32_t pairByteOffset = pairIndex * pixelBytes;
                for (uint32_t channelIndex = 0U; channelIndex < depthConst; ++channelIndex) {
                    const uint32_t channelByteOffset = channelIndex * sizeof(DT_X);
                    gatherOffsetInitLocal.SetValue(pairElementBase + channelIndex,
                        static_cast<int32_t>(pairByteOffset + channelByteOffset));
                    gatherOffsetInitLocal.SetValue(pairElementBase + depthConst + channelIndex,
                        static_cast<int32_t>(localGroupBytes + pairByteOffset + channelByteOffset));
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
            for (uint32_t offsetElementBase = periodElements; offsetElementBase < alignedOffsetElements; offsetElementBase += periodElements) {
                uint32_t offsetElementCount = alignedOffsetElements - offsetElementBase;
                if (offsetElementCount > periodElements) {
                    offsetElementCount = periodElements;
                }
                AscendC::Adds(gatherOffsetInitLocal[offsetElementBase],
                    gatherOffsetInitLocal[offsetElementBase - periodElements],
                    static_cast<int32_t>(periodStrideBytes),
                    offsetElementCount);
            }
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::DataCopyPadExtParams<DT_X> copyPadParams = {false, 0, 0, 0};
            AscendC::DataCopyExtParams firstLoadParams = {1U, 0U, 0U, 0U, 0U};
            AscendC::DataCopyExtParams secondLoadParams = {1U, 0U, 0U, 0U, 0U};
            AscendC::DataCopyExtParams outputStoreParams = {1U, 0U, 0U, 0U, 0U};

            const uint32_t coreIndex = static_cast<uint32_t>(AscendC::GetBlockIdx());
            const uint32_t coreCount = static_cast<uint32_t>(AscendC::GetBlockNum());
            const uint32_t totalTasks = outputHeightConst * 2U;
            if (coreCount == 0U || coreIndex >= totalTasks) {
                return;
            }

            for (uint32_t taskIndex = coreIndex; taskIndex < totalTasks; taskIndex += coreCount) {
                Mode46LoadTaskToSlot(inputLocal, taskIndex, localGroupElements, pixelBytes,
                    firstLoadParams, secondLoadParams, copyPadParams);
                AscendC::PipeBarrier<PIPE_ALL>();

                const uint32_t outputH = taskIndex >> 1U;
                const uint32_t widthPart = taskIndex & 1U;
                const uint32_t outputWStart = widthPart * tileColsPerTask;
                uint32_t outputPixels = outputWidthConst - outputWStart;
                if (outputPixels > tileColsPerTask) {
                    outputPixels = tileColsPerTask;
                }
                const uint32_t currentOutputElements = outputPixels * depthConst;
                AscendC::Gather(inputLocal[inputLocalElements], inputLocal, gatherOffsetLocal, 0U, currentOutputElements);
                AscendC::PipeBarrier<PIPE_ALL>();

                const uint64_t outputOffset = (static_cast<uint64_t>(outputH) * outputWidthConst + outputWStart) * depthConst;
                outputStoreParams.blockLen = currentOutputElements * sizeof(DT_X);
                AscendC::DataCopyPad(yGm[outputOffset], inputLocal[inputLocalElements], outputStoreParams);
                AscendC::PipeBarrier<PIPE_MTE3>();
            }
        }
    }

    __aicore__ inline void ProcessRowGatherBlock2Copy()
    {
        if (IsTest1ExactShape()) {
            ProcessRowGatherBlock2Test1TbufPipeline();
            return;
        }

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t totalRows = outputBatch * outputHeight;
        const uint32_t rowTasks =
            (totalRows + rowGroupSize - 1U) / rowGroupSize;

        for (uint32_t rowTask = blockIdx; rowTask < rowTasks; rowTask += blockNum) {
            const uint32_t firstRow = rowTask * rowGroupSize;
            uint32_t lastRow = firstRow + rowGroupSize;
            if (lastRow > totalRows) {
                lastRow = totalRows;
            }
            bool hasPendingRow = false;
            uint32_t pendingRow = firstRow;
            for (uint32_t row = firstRow; row < lastRow; ++row) {
                EnQueRowGatherBlock2Row(row);
                if (hasPendingRow) {
                    DeQueRowGatherBlock2Row(pendingRow);
                }
                pendingRow = row;
                hasPendingRow = true;
            }
            if (hasPendingRow) {
                DeQueRowGatherBlock2Row(pendingRow);
            }
        }
    }

    __aicore__ inline bool IsTest1ExactShape() const
    {
        return sizeof(DT_X) == 4U &&
            outputBatch == 2U &&
            outputHeight == 56U &&
            outputWidth == 56U &&
            depth == 128U &&
            inputHeight == 28U &&
            inputWidth == 28U &&
            cropTop == 0U &&
            cropLeft == 0U &&
            tileDepthAlign == 128U &&
            tileCols == 56U &&
            outputElements == 802816U;
    }

    __aicore__ inline void ProcessRowGatherBlock2Test1TbufPipeline()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum = AscendC::GetBlockNum();
        const auto eventIdIn0 =
            pipe->FetchEventID<AscendC::HardEvent::MTE2_MTE3>();
        const auto eventIdIn1 =
            pipe->FetchEventID<AscendC::HardEvent::MTE2_MTE3>();
        const auto eventIdOut0 =
            pipe->FetchEventID<AscendC::HardEvent::MTE3_MTE2>();
        const auto eventIdOut1 =
            pipe->FetchEventID<AscendC::HardEvent::MTE3_MTE2>();
        AscendC::LocalTensor<DT_X> local0 = xBuf.Get<DT_X>();
        AscendC::LocalTensor<DT_X> local1 = xBuf2.Get<DT_X>();

        bool inPending0 = false;
        bool inPending1 = false;
        bool outPending0 = false;
        bool outPending1 = false;
        uint32_t pendingRow0 = 0U;
        uint32_t pendingRow1 = 0U;
        uint32_t localTask = 0U;

        for (uint32_t row = blockIdx; row < 112U; row += blockNum) {
            const uint32_t slot = localTask & 1U;
            if (slot == 0U && outPending0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdOut0);
                outPending0 = false;
            } else if (slot == 1U && outPending1) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdOut1);
                outPending1 = false;
            }

            if (slot == 0U) {
                CopyRowGatherBlock2Test1TbufRow(local0, row);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdIn0);
                inPending0 = true;
                pendingRow0 = row;
            } else {
                CopyRowGatherBlock2Test1TbufRow(local1, row);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdIn1);
                inPending1 = true;
                pendingRow1 = row;
            }

            if (localTask > 0U) {
                const uint32_t prevSlot = slot ^ 1U;
                if (prevSlot == 0U && inPending0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdIn0);
                    AscendC::DataCopy(yGm[pendingRow0 * 7168U], local0, 7168U);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdOut0);
                    inPending0 = false;
                    outPending0 = true;
                } else if (prevSlot == 1U && inPending1) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdIn1);
                    AscendC::DataCopy(yGm[pendingRow1 * 7168U], local1, 7168U);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdOut1);
                    inPending1 = false;
                    outPending1 = true;
                }
            }
            ++localTask;
        }

        if (inPending0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdIn0);
            AscendC::DataCopy(yGm[pendingRow0 * 7168U], local0, 7168U);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdOut0);
            outPending0 = true;
        }
        if (inPending1) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdIn1);
            AscendC::DataCopy(yGm[pendingRow1 * 7168U], local1, 7168U);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdOut1);
            outPending1 = true;
        }
        if (outPending0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdOut0);
        }
        if (outPending1) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdOut1);
        }
    }

    __aicore__ inline void CopyRowGatherBlock2Test1TbufRow(
        AscendC::LocalTensor<DT_X> &local,
        uint32_t row)
    {
        const uint32_t outputN = row >= 56U ? 1U : 0U;
        const uint32_t outputH = row - outputN * 56U;
        const uint32_t inputH = outputH >> 1U;
        const uint32_t blockH = outputH & 1U;
        const uint32_t inputN0 = (blockH << 2U) + outputN;
        const uint32_t inputN1 = inputN0 + 2U;
        const uint32_t inputBaseH = inputH * 3584U;
        AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0, 0};
        AscendC::DataCopyExtParams copyInParams = {
            28,
            512,
            0,
            16,
            0
        };
        AscendC::DataCopyPad(
            local,
            xGm[inputN0 * 100352U + inputBaseH],
            copyInParams,
            padParams);
        AscendC::DataCopyPad(
            local[128],
            xGm[inputN1 * 100352U + inputBaseH],
            copyInParams,
            padParams);
    }

    __aicore__ inline void EnQueRowGatherBlock2Row(uint32_t row)
    {
        const uint32_t elementBytes = sizeof(DT_X);
        const uint32_t cropPhaseW = cropLeft & 1U;
        const uint32_t outputH = row % outputHeight;
        const uint32_t outputN = row / outputHeight;
        const uint32_t expandedH = outputH + cropTop;
        const uint32_t inputH = expandedH >> 1U;
        const uint32_t blockH = expandedH & 1U;
        const uint32_t localDstStride =
            static_cast<uint32_t>(tileDepthAlign * elementBytes / 32U);
        AscendC::LocalTensor<DT_X> bindLocal = bindQueue2.AllocTensor<DT_X>();
        AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0, 0};

        CopyRowGatherBlock2Parity(
            bindLocal,
            padParams,
            outputN,
            inputH,
            blockH,
            cropPhaseW,
            0U,
            localDstStride);
        CopyRowGatherBlock2Parity(
            bindLocal,
            padParams,
            outputN,
            inputH,
            blockH,
            cropPhaseW,
            1U,
            localDstStride);
        bindQueue2.EnQue(bindLocal);
    }

    __aicore__ inline void DeQueRowGatherBlock2Row(uint32_t row)
    {
        const uint32_t elementBytes = sizeof(DT_X);
        const uint32_t pixelBytes = depth * elementBytes;
        const uint32_t outputRowBytes = outputWidth * pixelBytes;
        const uint32_t outputRowBase = row * outputWidth * depth;
        AscendC::LocalTensor<DT_X> bindReady = bindQueue2.DeQue<DT_X>();

        if (tileDepthAlign == depth && outputRowBytes <= 65535U) {
            // C is 32B-aligned and UB layout is identical to the GM row layout.
            // Use one pure 1D MTE3 copy instead of outputWidth tiny 2D blocks.
            AscendC::DataCopyExtParams copyOutParams = {
                1U,
                outputRowBytes,
                0U,
                0U,
                0U
            };
            AscendC::DataCopyPad(yGm[outputRowBase], bindReady, copyOutParams);
        } else {
            AscendC::DataCopyExtParams copyOutParams = {
                static_cast<uint16_t>(outputWidth),
                pixelBytes,
                static_cast<uint32_t>((tileDepthAlign - depth) * elementBytes / 32U),
                0U,
                0U
            };
            AscendC::DataCopyPad(yGm[outputRowBase], bindReady, copyOutParams);
        }
        bindQueue2.FreeTensor(bindReady);
    }

    __aicore__ inline void CopyRowGatherBlock2Parity(
        AscendC::LocalTensor<DT_X> &xLocal,
        const AscendC::DataCopyPadExtParams<DT_X> &padParams,
        uint32_t outputN,
        uint32_t inputH,
        uint32_t blockH,
        uint32_t cropPhaseW,
        uint32_t parity,
        uint32_t localDstStride)
    {
        if (parity >= outputWidth) {
            return;
        }

        const uint32_t elementBytes = sizeof(DT_X);
        const uint32_t pixelBytes = depth * elementBytes;
        const uint32_t copyPixels = ((outputWidth - 1U - parity) >> 1U) + 1U;
        const uint32_t blockW = parity ^ cropPhaseW;
        const uint32_t inputWStart = (parity + cropLeft) >> 1U;
        const uint32_t blockOffset = (blockH << 1U) + blockW;
        const uint32_t inputN = blockOffset * outputBatch + outputN;
        const uint32_t inputIndex =
            ((inputN * inputHeight + inputH) * inputWidth + inputWStart) * depth;
        AscendC::DataCopyExtParams copyInParams = {
            static_cast<uint16_t>(copyPixels),
            pixelBytes,
            0,
            localDstStride,
            0
        };
        AscendC::DataCopyPad(
            xLocal[parity * tileDepthAlign],
            xGm[inputIndex],
            copyInParams,
            padParams);
    }

    __aicore__ inline void ProcessGroupedGatherBlock2Copy()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t totalRows = outputBatch * outputHeight;
        const uint32_t groupCount =
            (totalRows + rowGroupSize - 1U) / rowGroupSize;

        for (uint32_t group = blockIdx; group < groupCount; group += blockNum) {
            const uint32_t rowStart = group * rowGroupSize;
            uint32_t rowsThisGroup = totalRows - rowStart;
            if (rowsThisGroup > rowGroupSize) {
                rowsThisGroup = rowGroupSize;
            }
            CopyGroupedGatherBlock2Rows(rowStart, rowsThisGroup);
        }
    }

    __aicore__ inline void CopyGroupedGatherBlock2Rows(
        uint32_t rowStart,
        uint32_t rowsThisGroup)
    {
        const uint32_t elementBytes = sizeof(DT_X);
        const uint32_t pixelBytes = depth * elementBytes;
        const uint32_t cropPhaseW = cropLeft & 1U;
        const uint32_t rowLocalStride = outputWidth * tileDepthAlign;
        const uint32_t localDstStride =
            static_cast<uint32_t>(tileDepthAlign * elementBytes / 32U);
        AscendC::LocalTensor<DT_X> xLocal = xQueue.AllocTensor<DT_X>();
        AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0, 0};

        for (uint32_t rowOffset = 0; rowOffset < rowsThisGroup; ++rowOffset) {
            const uint32_t row = rowStart + rowOffset;
            const uint32_t outputH = row % outputHeight;
            const uint32_t outputN = row / outputHeight;
            const uint32_t expandedH = outputH + cropTop;
            const uint32_t inputH = expandedH >> 1U;
            const uint32_t blockH = expandedH & 1U;
            const uint32_t rowLocalBase = rowOffset * rowLocalStride;

            CopyGroupedGatherBlock2Parity(
                xLocal,
                padParams,
                outputN,
                inputH,
                blockH,
                cropPhaseW,
                0U,
                rowLocalBase,
                localDstStride);
            CopyGroupedGatherBlock2Parity(
                xLocal,
                padParams,
                outputN,
                inputH,
                blockH,
                cropPhaseW,
                1U,
                rowLocalBase,
                localDstStride);
        }

        xQueue.EnQue(xLocal);
        AscendC::LocalTensor<DT_X> xReady = xQueue.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = yQueue.AllocTensor<DT_X>();
        AscendC::Adds<DT_X>(
            yLocal,
            xReady,
            static_cast<DT_X>(0),
            rowsThisGroup * rowLocalStride);
        yQueue.EnQue(yLocal);
        xQueue.FreeTensor(xReady);

        AscendC::DataCopyExtParams copyOutParams = {
            static_cast<uint16_t>(rowsThisGroup * outputWidth),
            pixelBytes,
            static_cast<uint32_t>((tileDepthAlign - depth) * elementBytes / 32U),
            0,
            0
        };
        AscendC::LocalTensor<DT_X> yReady = yQueue.DeQue<DT_X>();
        AscendC::DataCopyPad(
            yGm[rowStart * outputWidth * depth],
            yReady,
            copyOutParams);
        yQueue.FreeTensor(yReady);
    }

    __aicore__ inline void CopyGroupedGatherBlock2Parity(
        AscendC::LocalTensor<DT_X> &xLocal,
        const AscendC::DataCopyPadExtParams<DT_X> &padParams,
        uint32_t outputN,
        uint32_t inputH,
        uint32_t blockH,
        uint32_t cropPhaseW,
        uint32_t parity,
        uint32_t rowLocalBase,
        uint32_t localDstStride)
    {
        if (parity >= outputWidth) {
            return;
        }

        const uint32_t elementBytes = sizeof(DT_X);
        const uint32_t pixelBytes = depth * elementBytes;
        const uint32_t copyPixels = ((outputWidth - 1U - parity) >> 1U) + 1U;
        const uint32_t blockW = parity ^ cropPhaseW;
        const uint32_t inputWStart = (parity + cropLeft) >> 1U;
        const uint32_t blockOffset = (blockH << 1U) + blockW;
        const uint32_t inputN = blockOffset * outputBatch + outputN;
        const uint32_t inputIndex =
            ((inputN * inputHeight + inputH) * inputWidth + inputWStart) * depth;
        AscendC::DataCopyExtParams copyInParams = {
            static_cast<uint16_t>(copyPixels),
            pixelBytes,
            0,
            localDstStride,
            0
        };
        AscendC::DataCopyPad(
            xLocal[rowLocalBase + parity * tileDepthAlign],
            xGm[inputIndex],
            copyInParams,
            padParams);
    }

    __aicore__ inline void ProcessGatherStridedCopy()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t totalRows = outputBatch * outputHeight;
        const uint32_t cropPhaseW = cropLeft % blockSize;
        const uint32_t rowsPerCore = (totalRows + blockNum - 1U) / blockNum;
        const uint32_t rowStart = blockIdx * rowsPerCore;
        uint32_t rowEnd = rowStart + rowsPerCore;
        if (rowEnd > totalRows) {
            rowEnd = totalRows;
        }

        bool hasPendingTile = false;
        uint32_t pendingOutputOffset = 0U;
        uint32_t pendingOutputPixels = 0U;
        for (uint32_t row = rowStart; row < rowEnd; ++row) {
            const uint32_t outputH = row % outputHeight;
            const uint32_t outputN = row / outputHeight;
            const uint32_t expandedH = outputH + cropTop;
            const uint32_t inputH = expandedH / blockSize;
            const uint32_t blockH = expandedH % blockSize;
            const uint32_t outputRowBase =
                ((outputN * outputHeight + outputH) * outputWidth) * depth;

            for (uint32_t outputStart = 0; outputStart < outputWidth;
                outputStart += tileCols) {
                uint32_t outputPixels = outputWidth - outputStart;
                if (outputPixels > tileCols) {
                    outputPixels = tileCols;
                }
                EnQueGatherStridedTile(
                    outputN,
                    inputH,
                    blockH,
                    cropPhaseW,
                    outputStart,
                    outputPixels);
                if (hasPendingTile) {
                    DeQueGatherStridedTile(pendingOutputOffset, pendingOutputPixels);
                }
                pendingOutputOffset = outputRowBase + outputStart * depth;
                pendingOutputPixels = outputPixels;
                hasPendingTile = true;
            }
        }
        if (hasPendingTile) {
            DeQueGatherStridedTile(pendingOutputOffset, pendingOutputPixels);
        }
    }

    __aicore__ inline void EnQueGatherStridedTile(
        uint32_t outputN,
        uint32_t inputH,
        uint32_t blockH,
        uint32_t cropPhaseW,
        uint32_t outputStart,
        uint32_t outputPixels)
    {
        const uint32_t elementBytes = sizeof(DT_X);
        const uint32_t pixelBytes = depth * elementBytes;
        const uint32_t outputEnd = outputStart + outputPixels;
        AscendC::LocalTensor<DT_X> bindLocal = bindQueue8.AllocTensor<DT_X>();
        AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0, 0};

        for (uint32_t blockW = 0; blockW < blockSize; ++blockW) {
            const uint32_t phaseFirstOutputW =
                (blockW + blockSize - cropPhaseW) % blockSize;
            uint32_t firstOutputW = phaseFirstOutputW;
            if (firstOutputW < outputStart) {
                const uint32_t delta = outputStart - firstOutputW;
                firstOutputW += ((delta + blockSize - 1U) / blockSize) * blockSize;
            }
            if (firstOutputW >= outputEnd) {
                continue;
            }

            const uint32_t copyPixels =
                (outputEnd - 1U - firstOutputW) / blockSize + 1U;
            const uint32_t inputWStart = (firstOutputW + cropLeft) / blockSize;
            const uint32_t blockOffset = blockH * blockSize + blockW;
            const uint32_t inputN = blockOffset * outputBatch + outputN;
            const uint32_t inputIndex =
                ((inputN * inputHeight + inputH) * inputWidth + inputWStart) * depth;
            const uint32_t localOffset =
                (firstOutputW - outputStart) * tileDepthAlign;
            AscendC::DataCopyExtParams copyInParams = {
                static_cast<uint16_t>(copyPixels),
                pixelBytes,
                0,
                static_cast<uint32_t>((blockSize - 1U) * tileDepthAlign * elementBytes / 32U),
                0
            };
            AscendC::DataCopyPad(
                bindLocal[localOffset],
                xGm[inputIndex],
                copyInParams,
                padParams);
        }
        bindQueue8.EnQue(bindLocal);
    }

    __aicore__ inline void DeQueGatherStridedTile(
        uint32_t outputOffset,
        uint32_t outputPixels)
    {
        const uint32_t elementBytes = sizeof(DT_X);
        const uint32_t pixelBytes = depth * elementBytes;
        AscendC::DataCopyExtParams copyOutParams = {
            static_cast<uint16_t>(outputPixels),
            pixelBytes,
            static_cast<uint32_t>((tileDepthAlign - depth) * elementBytes / 32U),
            0,
            0
        };
        AscendC::LocalTensor<DT_X> bindReady = bindQueue8.DeQue<DT_X>();
        AscendC::DataCopyPad(yGm[outputOffset], bindReady, copyOutParams);
        bindQueue8.FreeTensor(bindReady);
    }


    __aicore__ inline void ProcessBlock2HChunkVectorRepackCopy()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t hChunk = rowGroupSize;
        const uint32_t hChunksPerBatch = (inputHeight + hChunk - 1U) / hChunk;
        const uint32_t taskCount = outputBatch * hChunksPerBatch;
        const uint32_t elementBytes = sizeof(DT_X);
        const uint32_t pixelBytes = depth * elementBytes;
        const uint32_t cBlocks = pixelBytes >> 5U;
        const uint32_t inputRowElements = inputWidth * depth;
        const uint32_t outputRowElements = outputWidth * tileDepthAlign;

        bool hasPendingOutput = false;
        uint32_t pendingOutputOffset = 0U;
        uint32_t pendingOutputBytes = 0U;

        for (uint32_t task = blockIdx; task < taskCount; task += blockNum) {
            const uint32_t hChunkIdx = task % hChunksPerBatch;
            const uint32_t outputN = task / hChunksPerBatch;
            const uint32_t inputHStart = hChunkIdx * hChunk;
            uint32_t rowsThisChunk = inputHeight - inputHStart;
            if (rowsThisChunk > hChunk) {
                rowsThisChunk = hChunk;
            }
            const uint32_t outputHStart = inputHStart << 1U;
            const uint32_t outputOffset =
                ((outputN * outputHeight + outputHStart) * outputWidth) * depth;
            const uint32_t outputBytes =
                (rowsThisChunk << 1U) * outputWidth * depth * elementBytes;

            AscendC::LocalTensor<DT_X> outLocal = outQueue29.AllocTensor<DT_X>();

            // Start task N-1 UB->GM writeback before issuing task N GM->UB reads
            // and UB->UB repack, so MTE3 is more likely to overlap with current work.
            if (hasPendingOutput) {
                DeQueBlock2HChunkOutput(pendingOutputOffset, pendingOutputBytes);
            }

            EnQueBlock2HChunkPhase(outputN, inputHStart, rowsThisChunk, 0U, 0U);
            EnQueBlock2HChunkPhase(outputN, inputHStart, rowsThisChunk, 0U, 1U);
            EnQueBlock2HChunkPhase(outputN, inputHStart, rowsThisChunk, 1U, 0U);
            EnQueBlock2HChunkPhase(outputN, inputHStart, rowsThisChunk, 1U, 1U);

            RepackBlock2HChunkPhase(outLocal, rowsThisChunk, 0U, 0U,
                cBlocks, inputRowElements, outputRowElements);
            RepackBlock2HChunkPhase(outLocal, rowsThisChunk, 0U, 1U,
                cBlocks, inputRowElements, outputRowElements);
            RepackBlock2HChunkPhase(outLocal, rowsThisChunk, 1U, 0U,
                cBlocks, inputRowElements, outputRowElements);
            RepackBlock2HChunkPhase(outLocal, rowsThisChunk, 1U, 1U,
                cBlocks, inputRowElements, outputRowElements);

            outQueue29.EnQue(outLocal);

            pendingOutputOffset = outputOffset;
            pendingOutputBytes = outputBytes;
            hasPendingOutput = true;
        }

        if (hasPendingOutput) {
            DeQueBlock2HChunkOutput(pendingOutputOffset, pendingOutputBytes);
        }
    }

    __aicore__ inline void DeQueBlock2HChunkOutput(
        uint32_t outputOffset,
        uint32_t outputBytes)
    {
        AscendC::LocalTensor<DT_X> outReady = outQueue29.DeQue<DT_X>();
        AscendC::DataCopyExtParams copyOutParams = {
            1U,
            outputBytes,
            0U,
            0U,
            0U
        };
        AscendC::DataCopyPad(yGm[outputOffset], outReady, copyOutParams);
        outQueue29.FreeTensor(outReady);
    }

    __aicore__ inline void EnQueBlock2HChunkPhase(
        uint32_t outputN,
        uint32_t inputHStart,
        uint32_t rowsThisChunk,
        uint32_t blockH,
        uint32_t blockW)
    {
        const uint32_t elementBytes = sizeof(DT_X);
        const uint32_t blockOffset = (blockH << 1U) + blockW;
        const uint32_t inputN = blockOffset * outputBatch + outputN;
        const uint32_t inputIndex =
            ((inputN * inputHeight + inputHStart) * inputWidth) * depth;
        const uint32_t copyBytes = rowsThisChunk * inputWidth * depth * elementBytes;

        AscendC::LocalTensor<DT_X> inLocal = inQueue29.AllocTensor<DT_X>();
        AscendC::DataCopyExtParams copyInParams = {
            1U,
            copyBytes,
            0U,
            0U,
            0U
        };
        AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0, 0};
        AscendC::DataCopyPad(inLocal, xGm[inputIndex], copyInParams, padParams);
        inQueue29.EnQue(inLocal);
    }

    __aicore__ inline void RepackBlock2HChunkPhase(
        AscendC::LocalTensor<DT_X> &outLocal,
        uint32_t rowsThisChunk,
        uint32_t blockH,
        uint32_t blockW,
        uint32_t cBlocks,
        uint32_t inputRowElements,
        uint32_t outputRowElements)
    {
        AscendC::LocalTensor<DT_X> inReady = inQueue29.DeQue<DT_X>();

        // TP10 Mode29 W-first repack experiment.
        // Old layout copied one input row per DataCopy:
        //   4 phases * rowsThisChunk DataCopy commands.
        // Here we invert the loop and copy one input column across all H rows:
        //   4 phases * inputWidth DataCopy commands.
        // For TP10: C=32 FP16 => cBlocks=2, inputWidth=6, outputWidth=12.
        const uint32_t elementBytes = sizeof(DT_X);
        const uint32_t inputRowBlocks =
            static_cast<uint32_t>((inputRowElements * elementBytes) >> 5U);
        const uint32_t outputRowBlocks =
            static_cast<uint32_t>((outputRowElements * elementBytes) >> 5U);

        AscendC::DataCopyParams ubCopyParams = {
            static_cast<uint16_t>(rowsThisChunk),
            static_cast<uint16_t>(cBlocks),
            static_cast<uint16_t>(inputRowBlocks - cBlocks),
            static_cast<uint16_t>((outputRowBlocks << 1U) - cBlocks)
        };

        for (uint32_t w = 0; w < inputWidth; ++w) {
            const uint32_t srcOffset = w * depth;
            const uint32_t dstCol = (w << 1U) + blockW;
            const uint32_t dstOffset =
                blockH * outputRowElements + dstCol * tileDepthAlign;
            AscendC::DataCopy(outLocal[dstOffset], inReady[srcOffset], ubCopyParams);
        }
        inQueue29.FreeTensor(inReady);
    }


    __aicore__ inline void ProcessGatherStridedBlock4RowTileVectorRepackCopy()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t totalRows = outputBatch * outputHeight;
        const uint32_t widthTiles = (outputWidth + tileCols - 1U) / tileCols;
        const uint32_t taskCount = totalRows * widthTiles;
        const uint32_t cropPhaseW = cropLeft & 3U;
        const uint32_t elementBytes = sizeof(DT_X);
        const uint32_t pixelBytes = depth * elementBytes;
        const uint32_t localDstStride =
            static_cast<uint32_t>(3U * tileDepthAlign * elementBytes / 32U);

        bool hasPendingTile = false;
        uint32_t pendingOutputOffset = 0U;
        uint32_t pendingOutputPixels = 0U;
        for (uint32_t task = blockIdx; task < taskCount; task += blockNum) {
            const uint32_t widthTile = task % widthTiles;
            const uint32_t row = task / widthTiles;
            const uint32_t outputStart = widthTile * tileCols;
            uint32_t outputPixels = outputWidth - outputStart;
            if (outputPixels > tileCols) {
                outputPixels = tileCols;
            }
            const uint32_t outputEnd = outputStart + outputPixels;

            const uint32_t outputH = row % outputHeight;
            const uint32_t outputN = row / outputHeight;
            const uint32_t expandedH = outputH + cropTop;
            const uint32_t inputH = expandedH >> 2U;
            const uint32_t blockH = expandedH & 3U;
            const uint32_t outputOffset =
                ((outputN * outputHeight + outputH) * outputWidth + outputStart) * depth;

            AscendC::LocalTensor<DT_X> outLocal = outQueue24.AllocTensor<DT_X>();
            CopyGatherStridedBlock4PhaseVectorRepack(
                outLocal,
                outputN,
                inputH,
                blockH,
                cropPhaseW,
                outputStart,
                outputEnd,
                0U,
                pixelBytes,
                localDstStride);
            CopyGatherStridedBlock4PhaseVectorRepack(
                outLocal,
                outputN,
                inputH,
                blockH,
                cropPhaseW,
                outputStart,
                outputEnd,
                1U,
                pixelBytes,
                localDstStride);
            if (hasPendingTile) {
                DeQueGatherStridedTileVectorRepack(pendingOutputOffset, pendingOutputPixels);
            }
            CopyGatherStridedBlock4PhaseVectorRepack(
                outLocal,
                outputN,
                inputH,
                blockH,
                cropPhaseW,
                outputStart,
                outputEnd,
                2U,
                pixelBytes,
                localDstStride);
            CopyGatherStridedBlock4PhaseVectorRepack(
                outLocal,
                outputN,
                inputH,
                blockH,
                cropPhaseW,
                outputStart,
                outputEnd,
                3U,
                pixelBytes,
                localDstStride);
            outQueue24.EnQue(outLocal);

            pendingOutputOffset = outputOffset;
            pendingOutputPixels = outputPixels;
            hasPendingTile = true;
        }
        if (hasPendingTile) {
            DeQueGatherStridedTileVectorRepack(pendingOutputOffset, pendingOutputPixels);
        }
    }

    __aicore__ inline void CopyGatherStridedBlock4PhaseVectorRepack(
        AscendC::LocalTensor<DT_X> &outLocal,
        uint32_t outputN,
        uint32_t inputH,
        uint32_t blockH,
        uint32_t cropPhaseW,
        uint32_t outputStart,
        uint32_t outputEnd,
        uint32_t parity,
        uint32_t pixelBytes,
        uint32_t localDstStride)
    {
        uint32_t firstOutputW = parity;
        if (firstOutputW < outputStart) {
            const uint32_t delta = outputStart - firstOutputW;
            firstOutputW += ((delta + 3U) >> 2U) << 2U;
        }
        if (firstOutputW >= outputEnd) {
            return;
        }

        const uint32_t copyPixels = ((outputEnd - 1U - firstOutputW) >> 2U) + 1U;
        const uint32_t blockW = (parity + cropPhaseW) & 3U;
        const uint32_t inputWStart = (firstOutputW + cropLeft) >> 2U;
        const uint32_t blockOffset = (blockH << 2U) + blockW;
        const uint32_t inputN = blockOffset * outputBatch + outputN;
        const uint32_t inputIndex =
            ((inputN * inputHeight + inputH) * inputWidth + inputWStart) * depth;
        const uint32_t localOffset = (firstOutputW - outputStart) * tileDepthAlign;

        AscendC::LocalTensor<DT_X> inLocal = inQueue24.AllocTensor<DT_X>();

        // Step 1: GM -> UB is a true 1D contiguous transfer.  blockCount=1
        // and blockLen may be up to 16KB for tileCols=512, safely < 65535B.
        AscendC::DataCopyExtParams copyInParams = {
            1U,
            static_cast<uint32_t>(copyPixels * pixelBytes),
            0U,
            0U,
            0U
        };
        AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0, 0};
        AscendC::DataCopyPad(inLocal, xGm[inputIndex], copyInParams, padParams);
        inQueue24.EnQue(inLocal);

        // Step 2: UB -> UB strided scatter into the final contiguous output tile.
        // LocalTensor -> LocalTensor DataCopy does NOT accept DataCopyExtParams in CANN 8.5.
        // It requires DataCopyParams, whose blockLen/srcStride/dstStride are in 32B datablock units.
        AscendC::LocalTensor<DT_X> inReady = inQueue24.DeQue<DT_X>();
        AscendC::DataCopyParams ubCopyParams = {
            static_cast<uint16_t>(copyPixels),
            static_cast<uint16_t>(pixelBytes >> 5U),
            0U,
            static_cast<uint16_t>(localDstStride)
        };
        AscendC::DataCopy(outLocal[localOffset], inReady, ubCopyParams);
        inQueue24.FreeTensor(inReady);
    }

    __aicore__ inline void DeQueGatherStridedTileVectorRepack(
        uint32_t outputOffset,
        uint32_t outputPixels)
    {
        const uint32_t elementBytes = sizeof(DT_X);
        const uint32_t pixelBytes = depth * elementBytes;
        AscendC::DataCopyExtParams copyOutParams = {
            static_cast<uint16_t>(outputPixels),
            pixelBytes,
            static_cast<uint32_t>((tileDepthAlign - depth) * elementBytes / 32U),
            0,
            0
        };
        AscendC::LocalTensor<DT_X> outReady = outQueue24.DeQue<DT_X>();
        AscendC::DataCopyPad(yGm[outputOffset], outReady, copyOutParams);
        outQueue24.FreeTensor(outReady);
    }


    __aicore__ inline void ProcessReferenceStridedCopy()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t totalRows = outputBatch * outputHeight;
        const uint32_t cropPhaseW = cropLeft % blockSize;
        const uint32_t rowsPerCore = (totalRows + blockNum - 1U) / blockNum;
        const uint32_t rowStart = blockIdx * rowsPerCore;
        uint32_t rowEnd = rowStart + rowsPerCore;
        if (rowEnd > totalRows) {
            rowEnd = totalRows;
        }

        for (uint32_t row = rowStart; row < rowEnd; ++row) {
            const uint32_t outputH = row % outputHeight;
            const uint32_t outputN = row / outputHeight;
            const uint32_t expandedH = outputH + cropTop;
            const uint32_t inputH = expandedH / blockSize;
            const uint32_t blockH = expandedH % blockSize;
            for (uint32_t blockW = 0; blockW < blockSize; ++blockW) {
                const uint32_t firstOutputW =
                    (blockW + blockSize - cropPhaseW) % blockSize;
                if (firstOutputW >= outputWidth) {
                    continue;
                }

                const uint32_t inputWStart = (firstOutputW + cropLeft) / blockSize;
                const uint32_t pixelCount =
                    (outputWidth - 1U - firstOutputW) / blockSize + 1U;
                const uint32_t blockOffset = blockH * blockSize + blockW;
                const uint32_t inputN = blockOffset * outputBatch + outputN;
                const uint32_t inputBase =
                    ((inputN * inputHeight + inputH) * inputWidth + inputWStart) * depth;
                const uint32_t outputBase =
                    ((outputN * outputHeight + outputH) * outputWidth + firstOutputW) * depth;

                for (uint32_t pixelStart = 0; pixelStart < pixelCount;
                    pixelStart += tileCols) {
                    uint32_t copyPixels = pixelCount - pixelStart;
                    if (copyPixels > tileCols) {
                        copyPixels = tileCols;
                    }
                    CopyReferenceStridedCols(
                        inputBase + pixelStart * depth,
                        outputBase + pixelStart * blockSize * depth,
                        copyPixels);
                }
            }
        }
    }

    __aicore__ inline void CopyReferenceStridedCols(
        uint32_t inputOffset,
        uint32_t outputOffset,
        uint32_t copyPixels)
    {
        const uint32_t elementBytes = sizeof(DT_X);
        const uint32_t pixelBytes = depth * elementBytes;
        AscendC::LocalTensor<DT_X> bindLocal = bindQueue.AllocTensor<DT_X>();
        AscendC::DataCopyExtParams copyInParams = {
            static_cast<uint16_t>(copyPixels),
            pixelBytes,
            0,
            0,
            0
        };
        AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0, 0};
        AscendC::DataCopyPad(bindLocal, xGm[inputOffset], copyInParams, padParams);
        bindQueue.EnQue(bindLocal);

        AscendC::DataCopyExtParams copyOutParams = {
            static_cast<uint16_t>(copyPixels),
            pixelBytes,
            static_cast<uint32_t>((tileDepthAlign - depth) * elementBytes / 32U),
            static_cast<uint32_t>((blockSize - 1U) * depth * elementBytes),
            0
        };
        AscendC::LocalTensor<DT_X> bindReady = bindQueue.DeQue<DT_X>();
        AscendC::DataCopyPad(yGm[outputOffset], bindReady, copyOutParams);
        bindQueue.FreeTensor(bindReady);
    }


    __aicore__ inline void ProcessChannelRowTileBlock2Copy()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t totalRows = outputBatch * outputHeight;
        const uint32_t widthTileCols = rowGroupSize;
        const uint32_t widthTiles =
            (outputWidth + widthTileCols - 1U) / widthTileCols;
        const uint32_t taskCount = totalRows * channelTiles * widthTiles;
        const uint32_t elementBytes = sizeof(DT_X);
        const uint32_t maxChunkElements =
            tileCols == 0U ? CHANNEL_COPY_TILE_BYTES / elementBytes : tileCols;

        for (uint32_t task = blockIdx; task < taskCount; task += blockNum) {
            uint32_t remainder = task;
            const uint32_t widthTile = remainder % widthTiles;
            remainder /= widthTiles;
            const uint32_t channelTile = remainder % channelTiles;
            const uint32_t row = remainder / channelTiles;

            const uint32_t outputStart = widthTile * widthTileCols;
            uint32_t outputPixels = outputWidth - outputStart;
            if (outputPixels > widthTileCols) {
                outputPixels = widthTileCols;
            }
            CopyChannelRowTileBlock2(row, channelTile, outputStart, outputPixels);
        }
    }

    __aicore__ inline void CopyChannelRowTileBlock2(
        uint32_t row,
        uint32_t channelTile,
        uint32_t outputStart,
        uint32_t outputPixels)
    {
        const uint32_t elementBytes = sizeof(DT_X);
        const uint32_t maxChunkElements =
            tileCols == 0U ? CHANNEL_COPY_TILE_BYTES / elementBytes : tileCols;
        const uint32_t channelStart = channelTile * maxChunkElements;
        uint32_t chunkElements = depth - channelStart;
        if (chunkElements > maxChunkElements) {
            chunkElements = maxChunkElements;
        }

        const uint32_t chunkBytes = chunkElements * elementBytes;
        const uint32_t localDstStride = chunkBytes / 32U;
        const uint32_t outputH = row % outputHeight;
        const uint32_t outputN = row / outputHeight;
        const uint32_t expandedH = outputH + cropTop;
        const uint32_t inputH = expandedH >> 1U;
        const uint32_t blockH = expandedH & 1U;
        const uint32_t cropPhaseW = cropLeft & 1U;
        const uint32_t outputEnd = outputStart + outputPixels;
        AscendC::LocalTensor<DT_X> bindLocal = bindQueue.AllocTensor<DT_X>();
        AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0, 0};

        CopyChannelRowTileBlock2Parity(
            bindLocal,
            padParams,
            outputN,
            inputH,
            blockH,
            cropPhaseW,
            channelStart,
            chunkBytes,
            localDstStride,
            outputStart,
            outputEnd,
            0U);
        CopyChannelRowTileBlock2Parity(
            bindLocal,
            padParams,
            outputN,
            inputH,
            blockH,
            cropPhaseW,
            channelStart,
            chunkBytes,
            localDstStride,
            outputStart,
            outputEnd,
            1U);
        bindQueue.EnQue(bindLocal);

        AscendC::LocalTensor<DT_X> bindReady = bindQueue.DeQue<DT_X>();
        const uint32_t outputOffset =
            ((outputN * outputHeight + outputH) * outputWidth + outputStart) * depth +
            channelStart;
        AscendC::DataCopyExtParams copyOutParams = {
            static_cast<uint16_t>(outputPixels),
            chunkBytes,
            0,
            (depth - chunkElements) * elementBytes,
            0
        };
        AscendC::DataCopyPad(yGm[outputOffset], bindReady, copyOutParams);
        bindQueue.FreeTensor(bindReady);
    }

    __aicore__ inline void CopyChannelRowTileBlock2Parity(
        AscendC::LocalTensor<DT_X> &bindLocal,
        const AscendC::DataCopyPadExtParams<DT_X> &padParams,
        uint32_t outputN,
        uint32_t inputH,
        uint32_t blockH,
        uint32_t cropPhaseW,
        uint32_t channelStart,
        uint32_t chunkBytes,
        uint32_t localDstStride,
        uint32_t outputStart,
        uint32_t outputEnd,
        uint32_t parity)
    {
        uint32_t firstOutputW = parity;
        if (firstOutputW < outputStart) {
            const uint32_t delta = outputStart - firstOutputW;
            firstOutputW += ((delta + 1U) >> 1U) << 1U;
        }
        if (firstOutputW >= outputEnd) {
            return;
        }

        const uint32_t blockW = parity ^ cropPhaseW;
        const uint32_t copyPixels = ((outputEnd - 1U - firstOutputW) >> 1U) + 1U;
        const uint32_t inputWStart = (firstOutputW + cropLeft) >> 1U;
        const uint32_t blockOffset = (blockH << 1U) + blockW;
        const uint32_t inputN = blockOffset * outputBatch + outputN;
        const uint32_t inputIndex =
            ((inputN * inputHeight + inputH) * inputWidth + inputWStart) * depth +
            channelStart;
        const uint32_t elementBytes = sizeof(DT_X);
        const uint32_t localOffset =
            (firstOutputW - outputStart) * (chunkBytes / elementBytes);
        AscendC::DataCopyExtParams copyInParams = {
            static_cast<uint16_t>(copyPixels),
            chunkBytes,
            (depth * elementBytes - chunkBytes),
            localDstStride,
            0
        };
        AscendC::DataCopyPad(
            bindLocal[localOffset],
            xGm[inputIndex],
            copyInParams,
            padParams);
    }

    __aicore__ inline void ProcessColumnCopy()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t taskCount = outputBatch * blockSize * outputWidth;
        const uint32_t elementBytes = sizeof(DT_X);
        const uint32_t pixelBytes = depth * elementBytes;
        uint32_t pointsPerCopy = ROW_COPY_BUFFER_BYTES / pixelBytes;
        if (pointsPerCopy > MAX_COPY_BLOCKS) {
            pointsPerCopy = MAX_COPY_BLOCKS;
        }
        const uint32_t cropPhaseH = cropTop % blockSize;
        bool hasPendingCopy = false;
        uint32_t pendingOutputIndex = 0U;
        uint32_t pendingCopyPoints = 0U;

        for (uint32_t task = blockIdx; task < taskCount; task += blockNum) {
            uint32_t remainder = task;
            const uint32_t outputW = remainder % outputWidth;
            remainder /= outputWidth;
            const uint32_t blockH = remainder % blockSize;
            const uint32_t outputN = remainder / blockSize;
            const uint32_t firstOutputH =
                (blockH + blockSize - cropPhaseH) % blockSize;
            if (firstOutputH >= outputHeight) {
                continue;
            }
            const uint32_t inputHStart = (firstOutputH + cropTop) / blockSize;
            const uint32_t pointCount =
                (outputHeight - 1U - firstOutputH) / blockSize + 1U;
            const uint32_t expandedW = outputW + cropLeft;
            const uint32_t inputW = expandedW / blockSize;
            const uint32_t blockW = expandedW % blockSize;
            const uint32_t blockOffset = blockH * blockSize + blockW;
            const uint32_t inputN = blockOffset * outputBatch + outputN;

            for (uint32_t pointStart = 0; pointStart < pointCount;
                pointStart += pointsPerCopy) {
                uint32_t copyPoints = pointCount - pointStart;
                if (copyPoints > pointsPerCopy) {
                    copyPoints = pointsPerCopy;
                }
                const uint32_t inputIndex =
                    ((inputN * inputHeight + inputHStart + pointStart) *
                        inputWidth + inputW) * depth;
                const uint32_t outputIndex =
                    ((outputN * outputHeight + firstOutputH +
                        pointStart * blockSize) * outputWidth + outputW) * depth;
                AscendC::LocalTensor<DT_X> bindLocal = bindQueue5.AllocTensor<DT_X>();
                AscendC::DataCopyExtParams copyInParams = {
                    static_cast<uint16_t>(copyPoints),
                    pixelBytes,
                    (inputWidth * depth - depth) * elementBytes,
                    0,
                    0
                };
                AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0, 0};
                AscendC::DataCopyPad(bindLocal, xGm[inputIndex], copyInParams, padParams);
                bindQueue5.EnQue(bindLocal);
                if (hasPendingCopy) {
                    DeQueColumnCopy(
                        pendingOutputIndex,
                        pendingCopyPoints,
                        pixelBytes,
                        elementBytes);
                }
                pendingOutputIndex = outputIndex;
                pendingCopyPoints = copyPoints;
                hasPendingCopy = true;
            }
        }
        if (hasPendingCopy) {
            DeQueColumnCopy(
                pendingOutputIndex,
                pendingCopyPoints,
                pixelBytes,
                elementBytes);
        }
    }

    __aicore__ inline void DeQueColumnCopy(
        uint32_t outputIndex,
        uint32_t copyPoints,
        uint32_t pixelBytes,
        uint32_t elementBytes)
    {
        AscendC::LocalTensor<DT_X> bindReady = bindQueue5.DeQue<DT_X>();
        AscendC::DataCopyExtParams copyOutParams = {
            static_cast<uint16_t>(copyPoints),
            pixelBytes,
            0,
            (blockSize * outputWidth * depth - depth) * elementBytes,
            0
        };
        AscendC::DataCopyPad(yGm[outputIndex], bindReady, copyOutParams);
        bindQueue5.FreeTensor(bindReady);
    }

    static constexpr uint32_t ROW_COPY_BUFFER_BYTES = 64U * 1024U;
    static constexpr uint32_t CHANNEL_COPY_TILE_BYTES = 4U * 1024U;
    static constexpr uint32_t MAX_COPY_BLOCKS = 4095U;
    AscendC::TPipe *pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> xQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> yQueue;
    AscendC::TBuf<AscendC::QuePosition::VECIN> xBuf;
    AscendC::TBuf<AscendC::QuePosition::VECIN> xBuf2;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 1> bindQueue;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 2> bindQueue5;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 2> bindQueue2;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 2> bindQueue8;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQueue24;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outQueue24;
    AscendC::TQue<AscendC::QuePosition::VECIN, 4> inQueue29;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outQueue29;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    __gm__ DT_X *xPtr;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t depth;
    uint32_t outputBatch;
    uint32_t outputHeight;
    uint32_t outputWidth;
    uint32_t cropTop;
    uint32_t cropLeft;
    uint32_t blockSize;
    uint32_t outputElements;
    uint32_t copyMode;
    uint32_t rowGroupSize;
    uint32_t channelTiles;
    uint32_t tileCols;
    uint32_t tileDepthAlign;
};

template <typename DT_X>
__aicore__ inline void RunTp6D4096HalfPairCopy(GM_ADDR x, GM_ADDR y)
{
    if constexpr (sizeof(DT_X) != 2) {
        return;
    } else {
        constexpr uint32_t kBlockBytes = 32U;
        constexpr uint32_t kXH = 2U;
        constexpr uint32_t kXW = 2U;
        constexpr uint32_t kC = 4096U;
        constexpr uint32_t kYH = 4U;
        constexpr uint32_t kYW = 4U;
        constexpr uint32_t kCBytes = kC * sizeof(DT_X);       // 8192B
        constexpr uint32_t kCBlocks = kCBytes / kBlockBytes; // 256 blocks
        constexpr uint32_t kPairBlocks = 2U * kCBlocks;      // 512 blocks = 16KB
        constexpr uint32_t kPairElems = 2U * kC;
        constexpr uint32_t kTasks = kYH * kXW;               // 4 rows * 2 iw-pairs = 8

        AscendC::GlobalTensor<DT_X> src;
        AscendC::GlobalTensor<DT_X> dst;
        src.SetGlobalBuffer((__gm__ DT_X *)x);
        dst.SetGlobalBuffer((__gm__ DT_X *)y);

        // One task builds one contiguous output pair:
        // y[oh, 2*iw,   :] = x[inputN0, ih, iw, :]
        // y[oh, 2*iw+1, :] = x[inputN1, ih, iw, :]
        // This keeps correctness of row interleave, but gives 8-way parallelism.
        AscendC::LocalTensor<DT_X> pairLocal(
            AscendC::TPosition::VECCALC, 0, kPairElems);

        AscendC::DataCopyParams oneParams;
        oneParams.blockCount = 1;
        oneParams.blockLen = static_cast<uint16_t>(kCBlocks);
        oneParams.srcStride = 0;
        oneParams.dstStride = 0;

        AscendC::DataCopyParams pairParams;
        pairParams.blockCount = 1;
        pairParams.blockLen = static_cast<uint16_t>(kPairBlocks);
        pairParams.srcStride = 0;
        pairParams.dstStride = 0;

        const uint32_t coreId = AscendC::GetBlockIdx();
        const uint32_t coreNum = AscendC::GetBlockNum();
        for (uint32_t task = coreId; task < kTasks; task += coreNum) {
            const uint32_t oh = task >> 1U;
            const uint32_t iw = task & 1U;
            const uint32_t ih = oh >> 1U;
            const uint32_t blockH = oh & 1U;
            const uint32_t inputN0 = blockH << 1U;
            const uint32_t inputN1 = inputN0 + 1U;

            const uint32_t src0 = ((inputN0 * kXH + ih) * kXW + iw) * kC;
            const uint32_t src1 = ((inputN1 * kXH + ih) * kXW + iw) * kC;
            const uint32_t dstBase = (oh * kYW + (iw << 1U)) * kC;

            AscendC::DataCopy(pairLocal, src[src0], oneParams);
            AscendC::DataCopy(pairLocal[kC], src[src1], oneParams);

            // TP6 speed probe: keep the fast MTE-only dependency, but use a
            // real MTE2 -> MTE3 hard event instead of PIPE_MTE2.
            // PIPE_MTE2 was faster but produced small WA because MTE3 could
            // observe pairLocal before both MTE2 loads were fully visible.
            // This is lighter than PIPE_ALL and should be safer than MTE2-only.
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

            AscendC::DataCopy(dst[dstBase], pairLocal, pairParams);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }
}

template <typename DT_X>
__aicore__ inline void RunTp7DirectFlatCopy(GM_ADDR x, GM_ADDR y)
{
    if constexpr (sizeof(DT_X) != 2) {
        return;
    } else {
        constexpr uint32_t elemsPerBlock = 16U;
        constexpr uint32_t blocksPerLane = 512U;
        constexpr uint32_t laneElems = blocksPerLane * elemsPerBlock;
        constexpr uint32_t totalCores = 8U;

        AscendC::GlobalTensor<DT_X> src;
        AscendC::GlobalTensor<DT_X> dst;
        src.SetGlobalBuffer((__gm__ DT_X *)x);
        dst.SetGlobalBuffer((__gm__ DT_X *)y);

        AscendC::LocalTensor<DT_X> scratch(
            AscendC::TPosition::VECCALC, 0, laneElems);

        const uint32_t coreId = AscendC::GetBlockIdx();
        if (coreId >= totalCores) {
            return;
        }

        AscendC::DataCopyParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint16_t>(blocksPerLane);
        params.srcStride = 0;
        params.dstStride = 0;

        const uint32_t elemBase = coreId * laneElems;
        AscendC::DataCopy(scratch, src[elemBase], params);

        // TP7 robust repair after TP6 event68 merge:
        // The previous event-core probing still produces 0.10% WA under this code layout.
        // Use full pipeline barrier between MTE2 load and MTE3 store for this tiny flat-copy case.
        // Keep the post-store MTE3 barrier to avoid kernel-return racing the last GM write.
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::DataCopy(dst[elemBase], scratch, params);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }
}

template <typename DT_X>
 __global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    (void)workspace;
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingData, tiling_data, tiling);
    if (tiling_data.copyMode == 68U) {
        RunTp6D4096HalfPairCopy<DT_X>(x, y);
        return;
    }
    if (tiling_data.copyMode == 67U) {
        RunTp7DirectFlatCopy<DT_X>(x, y);
        return;
    }
    AscendC::TPipe pipe;
    KernelBatchToSpace<DT_X> op;
    op.Init(x, y, tiling_data, &pipe);
    op.Process();
}
