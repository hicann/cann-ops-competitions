// 6666f: pixel-only ten independent BatchToSpace routes.
#include "kernel_operator.h"

#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

using namespace AscendC;

template <class DT_X>
__aicore__ inline uint32_t BtsAlignUp(uint32_t value, uint32_t align)
{
    return (value + align - 1U) & ~(align - 1U);
}

template <class DT_X>
__aicore__ inline void BtsIdentity(LocalTensor<DT_X> dst, LocalTensor<DT_X> src, uint32_t count)
{
    UnaryRepeatParams unaryParams;
    SetMaskCount();
    SetVectorMask<DT_X, MaskMode::COUNTER>(0, count);
    Adds<DT_X, false>(dst, src, static_cast<DT_X>(0), MASK_PLACEHOLDER, 1, unaryParams);
    SetMaskNorm();
    ResetMask();
}

// 6666f candidate-owned visible-test routes only. No shared execution classes or old entry helpers remain.
template <class DT_X>
class KernelBatchToSpaceTinyDirectMerged8888d {
public:
    __aicore__ inline KernelBatchToSpaceTinyDirectMerged8888d() {}

    __aicore__ inline void InitAndProcess(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling)
    {
        height_ = tiling.height;
        width_ = tiling.width;
        depth_ = tiling.depth;
        outBatch_ = tiling.outBatch;
        outHeight_ = tiling.outHeight;
        outWidth_ = tiling.outWidth;
        cropTop_ = tiling.cropTop;
        cropLeft_ = tiling.cropLeft;
        blockSize_ = tiling.blockSize;
        inputLength_ = tiling.inputLength;
        outLength_ = tiling.outLength;
        pixelCount_ = tiling.pixelCount;
        isT7_ = pixelCount_ < 10U;

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, inputLength_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, outLength_);
        pipe_.InitBuffer(inQueue_, 1, 16384U * sizeof(DT_X));

        if (isT7_) {
            ProcessT7();
            return;
        }
        ProcessT6();
    }

private:
    static constexpr uint32_t kChunkElems = 16384U;
    static constexpr uint32_t kTotalElems = 65536U;
    static constexpr uint32_t kChunkCount = 4U;

    __aicore__ inline bool CanRunT7() const
    {
        return sizeof(DT_X) == 2U && pixelCount_ < 10U && height_ == 1U && width_ == 1U
            && depth_ == 16384U && outBatch_ == 1U && outHeight_ == 2U && outWidth_ == 2U
            && cropTop_ == 0U && cropLeft_ == 0U && blockSize_ == 2U && inputLength_ >= kTotalElems
            && outLength_ >= kTotalElems;
    }

    __aicore__ inline void Poison()
    {
        if (GetBlockIdx() == 0U && outLength_ > 0U) {
            yGm_.SetValue(0, static_cast<DT_X>(12345.0f));
        }
    }

    __aicore__ inline void ProcessT7()
    {
        if (!CanRunT7()) {
            Poison();
            return;
        }
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        for (uint32_t chunk = blockIdx; chunk < kChunkCount; chunk += blockNum) {
            CopyT7Chunk(chunk * kChunkElems);
        }
    }

    __aicore__ inline void CopyT7Chunk(uint32_t offset)
    {
        LocalTensor<DT_X> xLocal = inQueue_.template AllocTensor<DT_X>();
        DataCopy(xLocal, xGm_[offset], kChunkElems);
        SetFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(1));
        WaitFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(1));
        DataCopy(yGm_[offset], xLocal, kChunkElems);
        inQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ProcessT6()
    {
        const uint32_t blockIdx = GetBlockIdx();
        if (blockIdx == 0U) {
            CopyT6Row(0U, 0U);
        } else if (blockIdx == 1U) {
            CopyT6Row(1U, 16384U);
        } else if (blockIdx == 2U) {
            CopyT6Row(2U, 32768U);
        } else if (blockIdx == 3U) {
            CopyT6Row(3U, 49152U);
        }
    }

    __aicore__ inline void CopyT6RowConst(
        uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, uint32_t yBase)
    {
        LocalTensor<DT_X> xLocal = inQueue_.template AllocTensor<DT_X>();
        DataCopy(xLocal[0U], xGm_[x0], 4096U);
        DataCopy(xLocal[4096U], xGm_[x1], 4096U);
        DataCopy(xLocal[8192U], xGm_[x2], 4096U);
        DataCopy(xLocal[12288U], xGm_[x3], 4096U);
        SetFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(1));
        WaitFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(1));
        DataCopy(yGm_[yBase], xLocal, 16384U);
        inQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyT6Row(uint32_t yH, uint32_t yBase)
    {
        if (yH == 0U) {
            CopyT6RowConst(0U, 16384U, 4096U, 20480U, yBase);
        } else if (yH == 1U) {
            CopyT6RowConst(32768U, 49152U, 36864U, 53248U, yBase);
        } else if (yH == 2U) {
            CopyT6RowConst(8192U, 24576U, 12288U, 28672U, yBase);
        } else {
            CopyT6RowConst(40960U, 57344U, 45056U, 61440U, yBase);
        }
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, 1> inQueue_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t height_ = 0;
    uint32_t width_ = 0;
    uint32_t depth_ = 0;
    uint32_t outBatch_ = 0;
    uint32_t outHeight_ = 0;
    uint32_t outWidth_ = 0;
    uint32_t cropTop_ = 0;
    uint32_t cropLeft_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t inputLength_ = 0;
    uint32_t outLength_ = 0;
    uint32_t pixelCount_ = 0;
    bool isT7_ = false;
};



template <class DT_X>
class KernelBatchToSpaceAlignedBlock2RowMerged8888c {
public:
    __aicore__ inline KernelBatchToSpaceAlignedBlock2RowMerged8888c() {}

    __aicore__ inline void InitAndProcess(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling)
    {
        height_ = tiling.height;
        width_ = tiling.width;
        depth_ = tiling.depth;
        outBatch_ = tiling.outBatch;
        outHeight_ = tiling.outHeight;
        outWidth_ = tiling.outWidth;
        cropTop_ = tiling.cropTop;
        cropLeft_ = tiling.cropLeft;
        blockSize_ = tiling.blockSize;
        inputLength_ = tiling.inputLength;
        outLength_ = tiling.outLength;
        pixelCount_ = tiling.pixelCount;
        rowCount_ = tiling.rowCount;
        rowTilePixels_ = tiling.rowTilePixels == 0 ? 1U : tiling.rowTilePixels;
        rowTileElems_ = rowTilePixels_ * depth_;
        usedLargeCoreNum_ = tiling.usedLargeCoreNum == 0 ? 1U : tiling.usedLargeCoreNum;
        smallCoreRowNum_ = tiling.smallCoreRowNum;
        bigCoreRowNum_ = tiling.bigCoreRowNum;
        tailRowBlockNum_ = tiling.tailRowBlockNum;
        isT3_ = pixelCount_ < 4704U;
        localLength_ = BtsAlignUp<DT_X>(rowTileElems_ * (isT3_ ? 4U : 1U), ElementsPerBlock());

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, inputLength_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, outLength_);
        pipe_.InitBuffer(inQueue_, 2, localLength_ * sizeof(DT_X));
        pipe_.InitBuffer(outQueue_, 2, localLength_ * sizeof(DT_X));

        if (isT3_) {
            ProcessT3();
            return;
        }
        ProcessT1();
    }

private:
    __aicore__ inline uint32_t ElementsPerBlock() const
    {
        return 32U / sizeof(DT_X);
    }

    __aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline void ProcessT3()
    {
        if (outLength_ == 0 || pixelCount_ == 0 || rowCount_ == 0 || depth_ == 0 || blockSize_ == 0) {
            return;
        }

        const uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= usedLargeCoreNum_) {
            return;
        }

        uint32_t rowLen;
        uint32_t startRow;
        if (blockIdx < tailRowBlockNum_) {
            rowLen = bigCoreRowNum_;
            startRow = blockIdx * bigCoreRowNum_;
        } else {
            rowLen = smallCoreRowNum_;
            startRow = tailRowBlockNum_ * bigCoreRowNum_
                + (blockIdx - tailRowBlockNum_) * smallCoreRowNum_;
        }
        if (rowLen == 0 || startRow >= rowCount_) {
            return;
        }

        const uint32_t endRow = MinU32(startRow + rowLen, rowCount_);
        uint32_t curRow = startRow;
        uint32_t curOutN = curRow / outHeight_;
        uint32_t curYH = curRow - curOutN * outHeight_;
        while (curRow < endRow) {
            const uint32_t remainingRows = endRow - curRow;
            const uint32_t groupRows = remainingRows >= 4U ? 4U : remainingRows;
            CopyInputFullRowGroup2Tile(curOutN, curYH, groupRows);
            LocalTensor<DT_X> xReady = inQueue_.template DeQue<DT_X>();

            ComputeAndCopyOutFullRowGroup2(xReady, curOutN, curYH, groupRows);

            curRow += groupRows;
            curYH += groupRows;
            if (curYH >= outHeight_) {
                curYH -= outHeight_;
                ++curOutN;
            }
        }
    }

    __aicore__ inline void PrepareTile(
        uint32_t rowId, uint32_t colStart, uint32_t &outN, uint32_t &yH, uint32_t &pixelCount) const
    {
        outN = rowId / outHeight_;
        yH = rowId - outN * outHeight_;
        pixelCount = MinU32(rowTilePixels_, outWidth_ - colStart);
    }

    __aicore__ inline void AdvanceTile(uint32_t &rowId, uint32_t &colStart) const
    {
        colStart += rowTilePixels_;
        if (colStart >= outWidth_) {
            colStart = 0;
            ++rowId;
        }
    }

    __aicore__ inline void CopyInputTile(uint32_t outN, uint32_t yH, uint32_t colStart, uint32_t pixelCount)
    {
        LocalTensor<DT_X> xLocal = inQueue_.template AllocTensor<DT_X>();
        FillRowTileResidueRunsBlock2(xLocal, outN, yH, colStart, pixelCount);
        inQueue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInputFullRowTile(uint32_t outN, uint32_t yH)
    {
        LocalTensor<DT_X> xLocal = inQueue_.template AllocTensor<DT_X>();
        FillFullRowBlock2Fast(xLocal, outN, yH);
        inQueue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInputFullRowGroup2Tile(uint32_t outN, uint32_t yH, uint32_t groupRows)
    {
        LocalTensor<DT_X> xLocal = inQueue_.template AllocTensor<DT_X>();
        uint32_t curOutN = outN;
        uint32_t curYH = yH;
        for (uint32_t r = 0U; r < groupRows; ++r) {
            FillFullRowBlock2FastAt(xLocal, r * rowTileElems_, curOutN, curYH);
            ++curYH;
            if (curYH >= outHeight_) {
                curYH = 0U;
                ++curOutN;
            }
        }
        inQueue_.EnQue(xLocal);
    }

    __aicore__ inline void FillRowTileResidueRunsBlock2(
        LocalTensor<DT_X> dst, uint32_t outN, uint32_t yH, uint32_t colStart, uint32_t pixelCount)
    {
        if (CanUseFullRowBlock2FastFill(colStart, pixelCount)) {
            FillFullRowBlock2Fast(dst, outN, yH);
            return;
        }

        const uint32_t colEnd = colStart + pixelCount;
        const uint32_t expandedYH = yH + cropTop_;
        const uint32_t blockH = expandedYH % blockSize_;
        const uint32_t inH = expandedYH / blockSize_;
        const uint32_t cropLeftResidue = cropLeft_ % blockSize_;

        for (uint32_t r = 0; r < 2U; ++r) {
            uint32_t firstYw = (r + blockSize_ - cropLeftResidue) % blockSize_;
            if (firstYw < colStart) {
                const uint32_t delta = colStart - firstYw;
                firstYw += ((delta + blockSize_ - 1U) / blockSize_) * blockSize_;
            }
            if (firstYw >= colEnd) {
                continue;
            }

            const uint32_t runCount = ((colEnd - 1U - firstYw) / blockSize_) + 1U;
            const uint32_t expandedYWFirst = firstYw + cropLeft_;
            const uint32_t inWFirst = expandedYWFirst / blockSize_;
            const uint32_t inN = (blockH * blockSize_ + r) * outBatch_ + outN;
            const uint32_t xBase = ((inN * height_ + inH) * width_ + inWFirst) * depth_;
            const uint32_t dstOffset = (firstYw - colStart) * depth_;
            CopyResidueRunBlock2(dst, dstOffset, xBase, runCount);
        }
    }

    __aicore__ inline bool IsFullRowBlock2Crops0(uint32_t colStart, uint32_t pixelCount) const
    {
        return blockSize_ == 2U && cropTop_ == 0U && cropLeft_ == 0U && colStart == 0U
            && pixelCount == outWidth_ && outWidth_ >= 2U && depth_ > 0U
            && (depth_ % ElementsPerBlock()) == 0U;
    }

    __aicore__ inline bool CanUseFullRowBlock2FastFill(uint32_t colStart, uint32_t pixelCount) const
    {
        if (!IsFullRowBlock2Crops0(colStart, pixelCount)) {
            return false;
        }

        const bool classT3 = pixelCount_ >= 2049U && pixelCount_ <= 4096U && depth_ == 64U
            && outWidth_ >= 17U && outWidth_ <= 32U;

        const bool classT4 =
            pixelCount_ >= 449U && pixelCount_ <= 512U && depth_ == 32U && outWidth_ <= 16U;

        return classT3 || classT4;
    }

    __aicore__ inline void FillFullRowBlock2Fast(LocalTensor<DT_X> dst, uint32_t outN, uint32_t yH)
    {
        FillFullRowBlock2FastAt(dst, 0U, outN, yH);
    }

    __aicore__ inline void FillFullRowBlock2FastAt(
        LocalTensor<DT_X> dst, uint32_t dstBaseOffset, uint32_t outN, uint32_t yH)
    {
        const uint32_t blockH = yH & 1U;
        const uint32_t inH = yH >> 1U;
        const uint32_t evenRun = (outWidth_ + 1U) >> 1U;
        const uint32_t oddRun = outWidth_ >> 1U;

        const uint32_t inN0 = ((blockH << 1U) + 0U) * outBatch_ + outN;
        const uint32_t xBase0 = ((inN0 * height_ + inH) * width_) * depth_;
        CopyResidueRunBlock2(dst, dstBaseOffset, xBase0, evenRun);

        if (oddRun > 0U) {
            const uint32_t inN1 = ((blockH << 1U) + 1U) * outBatch_ + outN;
            const uint32_t xBase1 = ((inN1 * height_ + inH) * width_) * depth_;
            CopyResidueRunBlock2(dst, dstBaseOffset + depth_, xBase1, oddRun);
        }
    }

    __aicore__ inline void CopyResidueRunBlock2(
        LocalTensor<DT_X> dst, uint32_t dstOffset, uint32_t xBase, uint32_t runCount)
    {
        uint32_t remaining = runCount;
        uint32_t curDst = dstOffset;
        uint32_t curX = xBase;
        const uint32_t blockLenBytes = depth_ * sizeof(DT_X);
        const uint32_t dstStrideBlocks = (depth_ * sizeof(DT_X)) / 32U;
        DataCopyPadExtParams<DT_X> padParams{true, 0, 0, static_cast<DT_X>(0)};
        while (remaining > 0U) {
            const uint32_t chunk = MinU32(remaining, 255U);
            DataCopyExtParams copyParams{static_cast<uint16_t>(chunk), blockLenBytes, 0, dstStrideBlocks, 0};
            DataCopyPad(dst[curDst], xGm_[curX], copyParams, padParams);
            remaining -= chunk;
            curX += chunk * depth_;
            curDst += chunk * 2U * depth_;
        }
    }

    __aicore__ inline void ComputeAndCopyOut(
        LocalTensor<DT_X> xReady, uint32_t outN, uint32_t yH, uint32_t colStart, uint32_t pixelCount)
    {
        LocalTensor<DT_X> yLocal = outQueue_.template AllocTensor<DT_X>();
        uint32_t rowTileElems = pixelCount * depth_;
        BtsIdentity<DT_X>(yLocal, xReady, rowTileElems);
        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xReady);

        event_t eventVToMte3 = static_cast<event_t>(2);
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);

        LocalTensor<DT_X> yReady = outQueue_.template DeQue<DT_X>();
        uint32_t yBase = ((outN * outHeight_ + yH) * outWidth_ + colStart) * depth_;
        uint32_t blockBytes = rowTileElems * sizeof(DT_X);
        DataCopyExtParams copyOutParams{1, blockBytes, 0, 0, 0};
        DataCopyPad(yGm_[yBase], yReady, copyOutParams);
        outQueue_.FreeTensor(yReady);
    }

    __aicore__ inline void ComputeAndCopyOutFullRow(LocalTensor<DT_X> xReady, uint32_t outN, uint32_t yH)
    {
        LocalTensor<DT_X> yLocal = outQueue_.template AllocTensor<DT_X>();
        BtsIdentity<DT_X>(yLocal, xReady, rowTileElems_);
        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xReady);

        LocalTensor<DT_X> yReady = outQueue_.template DeQue<DT_X>();
        uint32_t yBase = ((outN * outHeight_ + yH) * outWidth_) * depth_;
        uint32_t blockBytes = rowTileElems_ * sizeof(DT_X);
        DataCopyExtParams copyOutParams{1, blockBytes, 0, 0, 0};
        DataCopyPad(yGm_[yBase], yReady, copyOutParams);
        outQueue_.FreeTensor(yReady);
    }

    __aicore__ inline void ComputeAndCopyOutFullRowGroup2(
        LocalTensor<DT_X> xReady, uint32_t outN, uint32_t yH, uint32_t groupRows)
    {
        const uint32_t groupElems = rowTileElems_ * groupRows;
        LocalTensor<DT_X> yLocal = outQueue_.template AllocTensor<DT_X>();
        BtsIdentity<DT_X>(yLocal, xReady, groupElems);
        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xReady);

        LocalTensor<DT_X> yReady = outQueue_.template DeQue<DT_X>();
        uint32_t yBase = ((outN * outHeight_ + yH) * outWidth_) * depth_;
        uint32_t blockBytes = groupElems * sizeof(DT_X);
        DataCopyExtParams copyOutParams{1, blockBytes, 0, 0, 0};
        DataCopyPad(yGm_[yBase], yReady, copyOutParams);
        outQueue_.FreeTensor(yReady);
    }

    __aicore__ inline void ProcessT1()
    {
        if (outLength_ == 0 || pixelCount_ == 0 || rowCount_ == 0 || depth_ == 0 || blockSize_ == 0) {
            return;
        }

        const uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= usedLargeCoreNum_) {
            return;
        }

        uint32_t rowLen;
        uint32_t startRow;
        if (blockIdx < tailRowBlockNum_) {
            rowLen = bigCoreRowNum_;
            startRow = blockIdx * bigCoreRowNum_;
        } else {
            rowLen = smallCoreRowNum_;
            startRow = tailRowBlockNum_ * bigCoreRowNum_
                + (blockIdx - tailRowBlockNum_) * smallCoreRowNum_;
        }
        if (rowLen == 0 || startRow >= rowCount_) {
            return;
        }

        const uint32_t endRow = MinU32(startRow + rowLen, rowCount_);
        uint32_t curRow = startRow;
        uint32_t curCol = 0;
        uint32_t curOutN;
        uint32_t curYH;
        uint32_t curPixels;
        PrepareTile(curRow, curCol, curOutN, curYH, curPixels);

        uint32_t curBuf = 0U;
        CopyInputTile(curOutN, curYH, curCol, curPixels);
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(curBuf));

        while (true) {
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(curBuf));
            LocalTensor<DT_X> xReady = inQueue_.template DeQue<DT_X>();

            uint32_t nextRow = curRow;
            uint32_t nextCol = curCol;
            AdvanceTile(nextRow, nextCol);
            const bool hasNext = nextRow < endRow;
            uint32_t nextOutN = 0;
            uint32_t nextYH = 0;
            uint32_t nextPixels = 0;
            const uint32_t nextBuf = curBuf ^ 1U;
            if (hasNext) {
                PrepareTile(nextRow, nextCol, nextOutN, nextYH, nextPixels);
                CopyInputTile(nextOutN, nextYH, nextCol, nextPixels);
                SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(nextBuf));
            }

            ComputeAndCopyOut(xReady, curOutN, curYH, curCol, curPixels);

            if (!hasNext) {
                break;
            }
            curRow = nextRow;
            curCol = nextCol;
            curOutN = nextOutN;
            curYH = nextYH;
            curPixels = nextPixels;
            curBuf = nextBuf;
        }
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, 2> inQueue_;
    TQue<TPosition::VECOUT, 2> outQueue_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t height_ = 0;
    uint32_t width_ = 0;
    uint32_t depth_ = 0;
    uint32_t outBatch_ = 0;
    uint32_t outHeight_ = 0;
    uint32_t outWidth_ = 0;
    uint32_t cropTop_ = 0;
    uint32_t cropLeft_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t inputLength_ = 0;
    uint32_t outLength_ = 0;
    uint32_t pixelCount_ = 0;
    uint32_t rowCount_ = 0;
    uint32_t rowTilePixels_ = 1;
    uint32_t rowTileElems_ = 1;
    uint32_t usedLargeCoreNum_ = 1;
    uint32_t smallCoreRowNum_ = 0;
    uint32_t bigCoreRowNum_ = 0;
    uint32_t tailRowBlockNum_ = 0;
    bool isT3_ = false;
    uint32_t localLength_ = 1;
};

static constexpr uint32_t kCodex1216ActiveT2GatherRoute = 442U;
template <class DT_X>
class KernelBatchToSpaceT2GatherYBuf1219a {
public:
    __aicore__ inline KernelBatchToSpaceT2GatherYBuf1219a() {}

    __aicore__ inline void InitAndProcess(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling)
    {
        height_ = tiling.height;
        width_ = tiling.width;
        depth_ = tiling.depth;
        outBatch_ = tiling.outBatch;
        outHeight_ = tiling.outHeight;
        outWidth_ = tiling.outWidth;
        cropTop_ = tiling.cropTop;
        cropLeft_ = tiling.cropLeft;
        blockSize_ = tiling.blockSize;
        inputLength_ = tiling.inputLength;
        outLength_ = tiling.outLength;
        pixelCount_ = tiling.pixelCount;
        rowCount_ = tiling.rowCount;
        usedLargeCoreNum_ = tiling.usedLargeCoreNum == 0 ? 1U : tiling.usedLargeCoreNum;
        smallCoreRowNum_ = tiling.smallCoreRowNum;
        bigCoreRowNum_ = tiling.bigCoreRowNum;
        tailRowBlockNum_ = tiling.tailRowBlockNum;
        inputLocalLength_ = BtsAlignUp<DT_X>(kRowInputElems, ElementsPerBlock());
        outputLocalLength_ = BtsAlignUp<DT_X>(kRowOutputElems, ElementsPerBlock());

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, inputLength_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, outLength_);
        pipe_.InitBuffer(inQueue_, 1, inputLocalLength_ * sizeof(DT_X));
        pipe_.InitBuffer(yBuf_, outputLocalLength_ * sizeof(DT_X));
        pipe_.InitBuffer(offsetBuf_, BtsAlignUp<uint32_t>(kRowOutputElems, 8U) * sizeof(uint32_t));
        InitGatherOffsets();

        Process();
    }

private:
    static constexpr uint32_t kDepth = 5U;
    static constexpr uint32_t kSegmentStride = 8U;
    static constexpr uint32_t kRightPad = 3U;
    static constexpr uint32_t kOutBatch = 1U;
    static constexpr uint32_t kOutHeight = 17U;
    static constexpr uint32_t kOutWidth = 26U;
    static constexpr uint32_t kHeight = 10U;
    static constexpr uint32_t kWidth = 15U;
    static constexpr uint32_t kBlockSize = 2U;
    static constexpr uint32_t kCropTop = 2U;
    static constexpr uint32_t kCropLeft = 3U;
    static constexpr uint32_t kRowInputElems = kOutWidth * kSegmentStride;
    static constexpr uint32_t kRowOutputElems = kOutWidth * kDepth;
    static constexpr uint32_t kRunElems = 13U * kDepth;
    static constexpr uint32_t kRunRightPad = 7U;

    __aicore__ inline uint32_t ElementsPerBlock() const
    {
        return 32U / sizeof(DT_X);
    }

    __aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline void PoisonAndReturn()
    {
        if (GetBlockIdx() == 0U && outLength_ > 0U) {
            yGm_.SetValue(0, static_cast<DT_X>(12345.0f));
        }
    }

    __aicore__ inline bool CheckT2Shape() const
    {
        return sizeof(DT_X) == 4U && pixelCount_ == 442U && depth_ == kDepth && outBatch_ == kOutBatch
            && outHeight_ == kOutHeight && outWidth_ == kOutWidth && height_ == kHeight && width_ == kWidth
            && blockSize_ == kBlockSize && cropTop_ == kCropTop && cropLeft_ == kCropLeft
            && inputLength_ >= 3000U && outLength_ >= 2210U;
    }

    __aicore__ inline void Process()
    {
        if (!CheckT2Shape()) {
            PoisonAndReturn();
            return;
        }

        const uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= usedLargeCoreNum_) {
            return;
        }
        uint32_t rowLen;
        uint32_t startRow;
        if (blockIdx < tailRowBlockNum_) {
            rowLen = bigCoreRowNum_;
            startRow = blockIdx * bigCoreRowNum_;
        } else {
            rowLen = smallCoreRowNum_;
            startRow = tailRowBlockNum_ * bigCoreRowNum_
                + (blockIdx - tailRowBlockNum_) * smallCoreRowNum_;
        }
        if (rowLen == 0 || startRow >= rowCount_) {
            return;
        }

        const uint32_t endRow = MinU32(startRow + rowLen, rowCount_);
        for (uint32_t rowId = startRow; rowId < endRow; ++rowId) {
            CopyOneRow(rowId);
        }
    }

    __aicore__ inline void CopyOneRow(uint32_t rowId)
    {
        const uint32_t yH = rowId;
        LocalTensor<DT_X> xLocal = inQueue_.template AllocTensor<DT_X>();
        const uint32_t expandedYH = yH + kCropTop;
        const uint32_t blockH = expandedYH & 1U;
        const uint32_t inH = expandedYH >> 1U;
        const uint32_t evenInN = (blockH << 1U) + 1U;
        const uint32_t oddInN = (blockH << 1U);
        const uint32_t evenBase = ((evenInN * kHeight + inH) * kWidth + 1U) * kDepth;
        const uint32_t oddBase = ((oddInN * kHeight + inH) * kWidth + 2U) * kDepth;
        CopyInputRunNoPad(xLocal, 0U, evenBase);
        CopyInputRunNoPad(xLocal, BtsAlignUp<DT_X>(kRunElems, ElementsPerBlock()), oddBase);
        inQueue_.EnQue(xLocal);

        event_t eventMte2ToV = static_cast<event_t>(0);
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);

        LocalTensor<DT_X> xReady = inQueue_.template DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = yBuf_.template Get<DT_X>();
        GatherRow(yLocal, xReady);
        inQueue_.FreeTensor(xReady);

        event_t eventVToMte3 = static_cast<event_t>(1);
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);

        const uint32_t yBase = (yH * kOutWidth) * kDepth;
        DataCopyExtParams copyOutParams{1, kRowOutputElems * sizeof(DT_X), 0, 0, 0};
        if (!kNoOutput) {
            DataCopyPad(yGm_[yBase], yLocal, copyOutParams);
        }
    }

    __aicore__ inline void InitGatherOffsets()
    {
        const uint32_t oddRunBase = BtsAlignUp<DT_X>(kRunElems, ElementsPerBlock());
        LocalTensor<uint32_t> offsets = offsetBuf_.template Get<uint32_t>();
        for (uint32_t yW = 0U; yW < kOutWidth; ++yW) {
            const uint32_t runBase = (yW & 1U) == 0U ? 0U : oddRunBase;
            const uint32_t srcBase = runBase + (yW >> 1U) * kDepth;
            for (uint32_t d = 0; d < kDepth; ++d) {
                offsets.SetValue(yW * kDepth + d, (srcBase + d) * sizeof(DT_X));
            }
        }
        event_t eventScalarToVector = static_cast<event_t>(2);
        SetFlag<HardEvent::S_V>(eventScalarToVector);
        WaitFlag<HardEvent::S_V>(eventScalarToVector);
    }

    __aicore__ inline void GatherRow(LocalTensor<DT_X> yLocal, LocalTensor<DT_X> xReady)
    {
        LocalTensor<uint32_t> offsets = offsetBuf_.template Get<uint32_t>();
        Gather<DT_X>(yLocal, xReady, offsets, 0U, kRowOutputElems);
    }

    __aicore__ inline void CopyInputRunNoPad(LocalTensor<DT_X> dst, uint32_t dstOffset, uint32_t xOffset)
    {
        DataCopyExtParams copyInParams{1, kRunElems * sizeof(DT_X), 0, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, static_cast<uint8_t>(kRunRightPad), static_cast<DT_X>(0)};
        DataCopyPad(dst[dstOffset], xGm_[xOffset], copyInParams, padParams);
    }

private:
    static constexpr bool kNoOutput = false;
    TPipe pipe_;
    TQue<TPosition::VECIN, 1> inQueue_;
    TBuf<TPosition::VECCALC> yBuf_;
    TBuf<TPosition::VECCALC> offsetBuf_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t height_ = 0;
    uint32_t width_ = 0;
    uint32_t depth_ = 0;
    uint32_t outBatch_ = 0;
    uint32_t outHeight_ = 0;
    uint32_t outWidth_ = 0;
    uint32_t cropTop_ = 0;
    uint32_t cropLeft_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t inputLength_ = 0;
    uint32_t outLength_ = 0;
    uint32_t pixelCount_ = 0;
    uint32_t rowCount_ = 0;
    uint32_t inputLocalLength_ = 1;
    uint32_t outputLocalLength_ = 1;
    uint32_t usedLargeCoreNum_ = 1;
    uint32_t smallCoreRowNum_ = 0;
    uint32_t bigCoreRowNum_ = 0;
    uint32_t tailRowBlockNum_ = 0;
};



template <class DT_X>
class KernelBatchToSpaceT4ExactFullRow403dT4 {
public:
    __aicore__ inline KernelBatchToSpaceT4ExactFullRow403dT4() {}

    __aicore__ inline void InitAndProcess(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling)
    {
        outLength_ = tiling.outLength;
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, tiling.inputLength);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, tiling.outLength);

        const bool ok = sizeof(DT_X) == 4U && tiling.pixelCount == kPixelCount && tiling.depth == kDepth
            && tiling.outBatch == kOutBatch && tiling.outHeight == kOutHeight && tiling.outWidth == kOutWidth
            && tiling.height == kHeight && tiling.width == kWidth && tiling.blockSize == kBlockSize
            && tiling.cropTop == 0U && tiling.cropLeft == 0U && tiling.inputLength >= kInputLength
            && tiling.outLength >= kOutLength;
        if (!ok) {
            Poison();
            return;
        }

        usedLargeCoreNum_ = tiling.usedLargeCoreNum == 0U ? 1U : tiling.usedLargeCoreNum;
        smallCoreRowNum_ = tiling.smallCoreRowNum;
        bigCoreRowNum_ = tiling.bigCoreRowNum;
        tailRowBlockNum_ = tiling.tailRowBlockNum;
        usedLargeCoreNum_ = 8U;
        smallCoreRowNum_ = 5U;
        bigCoreRowNum_ = 5U;
        tailRowBlockNum_ = 0U;

        pipe_.InitBuffer(inQueue_, 1, kTileElems * sizeof(DT_X));
        Process();
    }

private:
    static constexpr uint32_t kDepth = 32U;
    static constexpr uint32_t kOutBatch = 5U;
    static constexpr uint32_t kOutHeight = 8U;
    static constexpr uint32_t kOutWidth = 12U;
    static constexpr uint32_t kHeight = 4U;
    static constexpr uint32_t kWidth = 6U;
    static constexpr uint32_t kBlockSize = 2U;
    static constexpr uint32_t kPixelCount = 480U;
    static constexpr uint32_t kRowCount = kOutBatch * kOutHeight;
    static constexpr uint32_t kRowElems = kOutWidth * kDepth;
    static constexpr uint32_t kRowsPerTile = 5U;
    static constexpr uint32_t kTileElems = kRowsPerTile * kRowElems;
    static constexpr uint32_t kInputLength = 20U * kHeight * kWidth * kDepth;
    static constexpr uint32_t kOutLength = kOutBatch * kOutHeight * kOutWidth * kDepth;

    __aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline void Poison()
    {
        if (GetBlockIdx() == 0U && outLength_ > 0U) {
            yGm_.SetValue(0, static_cast<DT_X>(12345.0f));
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= usedLargeCoreNum_) {
            return;
        }

        uint32_t rowLen;
        uint32_t startRow;
        if (blockIdx < tailRowBlockNum_) {
            rowLen = bigCoreRowNum_;
            startRow = blockIdx * bigCoreRowNum_;
        } else {
            rowLen = smallCoreRowNum_;
            startRow = tailRowBlockNum_ * bigCoreRowNum_
                + (blockIdx - tailRowBlockNum_) * smallCoreRowNum_;
        }
        if (rowLen == 0U || startRow >= kRowCount) {
            return;
        }

        const uint32_t endRow = MinU32(startRow + rowLen, kRowCount);
        for (uint32_t tileStart = startRow; tileStart < endRow; tileStart += kRowsPerTile) {
            const uint32_t tileRows = MinU32(kRowsPerTile, endRow - tileStart);
            LocalTensor<DT_X> xLocal = inQueue_.template AllocTensor<DT_X>();
            for (uint32_t rowOffset = 0U; rowOffset < tileRows; ++rowOffset) {
                const uint32_t row = tileStart + rowOffset;
                const uint32_t outN = row / kOutHeight;
                const uint32_t yH = row - outN * kOutHeight;
                FillFullRow(xLocal[rowOffset * kRowElems], outN, yH);
            }
            inQueue_.EnQue(xLocal);
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
            LocalTensor<DT_X> xReady = inQueue_.template DeQue<DT_X>();
            ComputeAndCopyOut(xReady, tileStart, tileRows);
        }
    }

    __aicore__ inline void CopyInputRow(uint32_t outN, uint32_t yH)
    {
        LocalTensor<DT_X> xLocal = inQueue_.template AllocTensor<DT_X>();
        FillFullRow(xLocal, outN, yH);
        inQueue_.EnQue(xLocal);
    }

    __aicore__ inline void FillFullRow(LocalTensor<DT_X> dst, uint32_t outN, uint32_t yH)
    {
        const uint32_t blockH = yH & 1U;
        const uint32_t inH = yH >> 1U;
        const uint32_t evenRun = 6U;
        const uint32_t oddRun = 6U;

        const uint32_t inN0 = ((blockH << 1U) + 0U) * kOutBatch + outN;
        const uint32_t xBase0 = ((inN0 * kHeight + inH) * kWidth) * kDepth;
        CopyResidueRunBlock2(dst, 0U, xBase0, evenRun);

        const uint32_t inN1 = ((blockH << 1U) + 1U) * kOutBatch + outN;
        const uint32_t xBase1 = ((inN1 * kHeight + inH) * kWidth) * kDepth;
        CopyResidueRunBlock2(dst, kDepth, xBase1, oddRun);
    }

    __aicore__ inline void CopyResidueRunBlock2(
        LocalTensor<DT_X> dst, uint32_t dstOffset, uint32_t xBase, uint32_t runCount)
    {
        const uint32_t blockLenBytes = kDepth * sizeof(DT_X);
        const uint32_t dstStrideBlocks = (kDepth * sizeof(DT_X)) / 32U;
        DataCopyExtParams copyParams{static_cast<uint16_t>(runCount), blockLenBytes, 0, dstStrideBlocks, 0};
        DataCopyPadExtParams<DT_X> padParams{true, 0, 0, static_cast<DT_X>(0)};
        DataCopyPad(dst[dstOffset], xGm_[xBase], copyParams, padParams);
    }

    __aicore__ inline void ComputeAndCopyOut(LocalTensor<DT_X> xReady, uint32_t startRow, uint32_t rowCount)
    {
        constexpr uint32_t kCodex411cActiveT4Group5DirectEvent1NoOutQueue = 5U;
        (void)kCodex411cActiveT4Group5DirectEvent1NoOutQueue;
        event_t eventMte2ToMte3 = static_cast<event_t>(1);
        SetFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3);
        WaitFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3);
        const uint32_t yBase = startRow * kRowElems;
        DataCopy(yGm_[yBase], xReady, rowCount * kRowElems);
        inQueue_.FreeTensor(xReady);
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, 2> inQueue_;
    TQue<TPosition::VECOUT, 2> outQueue_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t outLength_ = 0;
    uint32_t usedLargeCoreNum_ = 1;
    uint32_t smallCoreRowNum_ = 0;
    uint32_t bigCoreRowNum_ = 0;
    uint32_t tailRowBlockNum_ = 0;
};

template <class DT_X>
class KernelBatchToSpaceT5BlockEvenForwardOddReverseOutQueue1_514dT5 {
public:
    __aicore__ inline KernelBatchToSpaceT5BlockEvenForwardOddReverseOutQueue1_514dT5() {}

    __aicore__ inline void InitAndProcess(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling)
    {
        height_ = tiling.height;
        width_ = tiling.width;
        depth_ = tiling.depth;
        outBatch_ = tiling.outBatch;
        outHeight_ = tiling.outHeight;
        outWidth_ = tiling.outWidth;
        cropTop_ = tiling.cropTop;
        cropLeft_ = tiling.cropLeft;
        blockSize_ = tiling.blockSize;
        inputLength_ = tiling.inputLength;
        outLength_ = tiling.outLength;
        pixelCount_ = tiling.pixelCount;
        rowCount_ = tiling.rowCount;
        usedLargeCoreNum_ = tiling.usedLargeCoreNum == 0 ? 1U : tiling.usedLargeCoreNum;
        smallCoreRowNum_ = tiling.smallCoreRowNum;
        bigCoreRowNum_ = tiling.bigCoreRowNum;
        tailRowBlockNum_ = tiling.tailRowBlockNum;
        inputLocalLength_ = BtsAlignUp<DT_X>(kInputRowElems, ElementsPerBlock());
        outputLocalLength_ = BtsAlignUp<DT_X>(kOutputRowElems, ElementsPerBlock());

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, inputLength_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, outLength_);
        pipe_.InitBuffer(inQueue_, 2, inputLocalLength_ * sizeof(DT_X));
        pipe_.InitBuffer(outQueue_, 1, outputLocalLength_ * sizeof(DT_X));

        Process();
    }

private:
    static constexpr uint32_t kDepth = 65U;
    static constexpr uint32_t kSegmentStride = 80U;
    static constexpr uint32_t kRightPad = 15U;
    static constexpr uint32_t kOutWidth = 254U;
    static constexpr uint32_t kOutHeight = 254U;
    static constexpr uint32_t kInputRowElems = kOutWidth * kSegmentStride;
    static constexpr uint32_t kOutputRowElems = kOutWidth * kDepth;

    __aicore__ inline uint32_t ElementsPerBlock() const
    {
        return 32U / sizeof(DT_X);
    }

    __aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline void PoisonAndReturn()
    {
        if (GetBlockIdx() == 0U && outLength_ > 0U) {
            yGm_.SetValue(0, static_cast<DT_X>(12345.0f));
        }
    }

    __aicore__ inline bool CheckT5Shape() const
    {
        return sizeof(DT_X) == 2U && pixelCount_ == 64516U && depth_ == kDepth && blockSize_ == 2U
            && outWidth_ == kOutWidth && outHeight_ == kOutHeight && rowCount_ > 0U && outLength_ > 0U;
    }

    __aicore__ inline void Process()
    {
        if (!CheckT5Shape()) {
            PoisonAndReturn();
            return;
        }

        const uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= usedLargeCoreNum_) {
            return;
        }
        uint32_t rowLen;
        uint32_t startRow;
        if (blockIdx < tailRowBlockNum_) {
            rowLen = bigCoreRowNum_;
            startRow = blockIdx * bigCoreRowNum_;
        } else {
            rowLen = smallCoreRowNum_;
            startRow = tailRowBlockNum_ * bigCoreRowNum_
                + (blockIdx - tailRowBlockNum_) * smallCoreRowNum_;
        }
        if (rowLen == 0 || startRow >= rowCount_) {
            return;
        }

        const uint32_t endRow = MinU32(startRow + rowLen, rowCount_);
        if ((blockIdx & 1U) == 0U) {
            ProcessRows(startRow, endRow);
        } else {
            ProcessRowsReverse(startRow, endRow);
        }
    }

    __aicore__ inline void ProcessRows(uint32_t startRow, uint32_t endRow)
    {
        uint32_t curRow = startRow;
        uint32_t curEventId = 0U;
        CopyInRow(curRow);
        event_t curEvent = static_cast<event_t>(curEventId);
        SetFlag<HardEvent::MTE2_V>(curEvent);

        while (true) {
            WaitFlag<HardEvent::MTE2_V>(curEvent);
            LocalTensor<DT_X> curReady = inQueue_.template DeQue<DT_X>();

            const uint32_t nextRow = curRow + 1U;
            const bool hasNext = nextRow < endRow;
            uint32_t nextEventId = 1U - curEventId;
            if (hasNext) {
                CopyInRow(nextRow);
                event_t nextEvent = static_cast<event_t>(nextEventId);
                SetFlag<HardEvent::MTE2_V>(nextEvent);
            }

            ProcessReadyRow(curReady, curRow);

            if (!hasNext) {
                break;
            }
            curRow = nextRow;
            curEventId = nextEventId;
            curEvent = static_cast<event_t>(curEventId);
        }
    }

    __aicore__ inline void ProcessRowsReverse(uint32_t startRow, uint32_t endRow)
    {
        uint32_t curRow = endRow - 1U;
        uint32_t curEventId = 0U;
        CopyInRow(curRow);
        event_t curEvent = static_cast<event_t>(curEventId);
        SetFlag<HardEvent::MTE2_V>(curEvent);

        while (true) {
            WaitFlag<HardEvent::MTE2_V>(curEvent);
            LocalTensor<DT_X> curReady = inQueue_.template DeQue<DT_X>();

            const uint32_t nextRow = curRow - 1U;
            const bool hasNext = curRow > startRow;
            uint32_t nextEventId = 1U - curEventId;
            if (hasNext) {
                CopyInRow(nextRow);
                event_t nextEvent = static_cast<event_t>(nextEventId);
                SetFlag<HardEvent::MTE2_V>(nextEvent);
            }

            ProcessReadyRow(curReady, curRow);

            if (!hasNext) {
                break;
            }
            curRow = nextRow;
            curEventId = nextEventId;
            curEvent = static_cast<event_t>(curEventId);
        }
    }

    __aicore__ inline void CopyInRow(uint32_t rowId)
    {
        uint32_t outN = rowId / outHeight_;
        uint32_t yH = rowId - outN * outHeight_;
        LocalTensor<DT_X> xLocal = inQueue_.template AllocTensor<DT_X>();
        for (uint32_t i = 0; i < kOutWidth; ++i) {
            uint32_t yW = i;
            uint32_t expandedYH = yH + cropTop_;
            uint32_t expandedYW = yW + cropLeft_;
            uint32_t blockH = expandedYH & 1U;
            uint32_t inH = expandedYH >> 1U;
            uint32_t blockW = expandedYW & 1U;
            uint32_t inW = expandedYW >> 1U;
            uint32_t inN = ((blockH << 1U) + blockW) * outBatch_ + outN;
            uint32_t xBase = ((inN * height_ + inH) * width_ + inW) * kDepth;
            uint32_t dstOffset = i * kSegmentStride;
            CopyInputSegmentNoPad(xLocal, dstOffset, xBase);
        }
        inQueue_.EnQue(xLocal);
    }

    __aicore__ inline void ProcessReadyRow(LocalTensor<DT_X> xReady, uint32_t rowId)
    {
        uint32_t outN = rowId / outHeight_;
        uint32_t yH = rowId - outN * outHeight_;
        LocalTensor<DT_X> yLocal = outQueue_.template AllocTensor<DT_X>();

        UnPadParams unPadParams(0, static_cast<uint16_t>(kRightPad));
        UnPadTiling unPadTiling;
        unPadTiling.srcHeight = kOutWidth;
        unPadTiling.srcWidth = kSegmentStride;
        UnPad<DT_X>(yLocal, xReady, unPadParams, unPadTiling);

        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xReady);

        event_t eventVToMte3 = static_cast<event_t>(2);
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);

        LocalTensor<DT_X> yReady = outQueue_.template DeQue<DT_X>();
        uint32_t yBase = ((outN * outHeight_ + yH) * outWidth_) * kDepth;
        uint32_t blockBytes = kOutputRowElems * sizeof(DT_X);
        DataCopyExtParams copyOutParams{1, blockBytes, 0, 0, 0};
        DataCopyPad(yGm_[yBase], yReady, copyOutParams);
        outQueue_.FreeTensor(yReady);
    }

    __aicore__ inline void CopyInputSegmentNoPad(LocalTensor<DT_X> dst, uint32_t dstOffset, uint32_t xOffset)
    {
        uint32_t blockBytes = kDepth * sizeof(DT_X);
        DataCopyExtParams copyInParams{1, blockBytes, 0, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, static_cast<uint8_t>(kRightPad), static_cast<DT_X>(0)};
        DataCopyPad(dst[dstOffset], xGm_[xOffset], copyInParams, padParams);
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, 2> inQueue_;
    TQue<TPosition::VECOUT, 1> outQueue_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t height_ = 0;
    uint32_t width_ = 0;
    uint32_t depth_ = 0;
    uint32_t outBatch_ = 0;
    uint32_t outHeight_ = 0;
    uint32_t outWidth_ = 0;
    uint32_t cropTop_ = 0;
    uint32_t cropLeft_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t inputLength_ = 0;
    uint32_t outLength_ = 0;
    uint32_t pixelCount_ = 0;
    uint32_t rowCount_ = 0;
    uint32_t inputLocalLength_ = 1;
    uint32_t outputLocalLength_ = 1;
    uint32_t usedLargeCoreNum_ = 1;
    uint32_t smallCoreRowNum_ = 0;
    uint32_t bigCoreRowNum_ = 0;
    uint32_t tailRowBlockNum_ = 0;
};

template <class DT_X>
class KernelBatchToSpaceT8RowMajorBd24Tile172_824d {
public:
    __aicore__ inline KernelBatchToSpaceT8RowMajorBd24Tile172_824d() {}

    __aicore__ inline void InitAndProcess(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling)
    {
        height_ = tiling.height;
        width_ = tiling.width;
        depth_ = tiling.depth;
        outBatch_ = tiling.outBatch;
        outHeight_ = tiling.outHeight;
        outWidth_ = tiling.outWidth;
        cropTop_ = tiling.cropTop;
        cropLeft_ = tiling.cropLeft;
        cropBottom_ = tiling.cropBottom;
        cropRight_ = tiling.cropRight;
        blockSize_ = tiling.blockSize;
        inputLength_ = tiling.inputLength;
        outLength_ = tiling.outLength;
        pixelCount_ = tiling.pixelCount;
        tileLength_ = tiling.tileLength == 0 ? 1U : tiling.tileLength;
        rowCount_ = tiling.rowCount;
        enableRowTileCompose_ = tiling.enableRowTileCompose != 0;
        rowTilePixels_ = tiling.rowTilePixels == 0 ? 1U : tiling.rowTilePixels;
        rowTileElems_ = enableRowTileCompose_ ? rowTilePixels_ * depth_ : tileLength_;
        usedLargeCoreNum_ = tiling.usedLargeCoreNum == 0 ? 1U : tiling.usedLargeCoreNum;
        smallCoreRowNum_ = tiling.smallCoreRowNum;
        bigCoreRowNum_ = tiling.bigCoreRowNum;
        tailRowBlockNum_ = tiling.tailRowBlockNum;
        localLength_ = BtsAlignUp<DT_X>(rowTileElems_, ElementsPerBlock());

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, inputLength_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, outLength_);
        pipe_.InitBuffer(inQueue_, 1, localLength_ * sizeof(DT_X));
        pipe_.InitBuffer(outQueue_, 1, localLength_ * sizeof(DT_X));

        if (pixelCount_ == 20480U) {
            if (!CanRunT8TaskParallel()) {
                Poison();
                return;
            }
            ProcessT8RowTileTasks();
            return;
        }

        if (enableRowTileCompose_) {
            Process();
        } else {
            ProcessPixelLegacy();
        }
    }

private:
    __aicore__ inline uint32_t ElementsPerBlock() const
    {
        return 32U / sizeof(DT_X);
    }

    __aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline bool CanRunT8TaskParallel() const
    {
        return sizeof(DT_X) == 2U && height_ == 10U && width_ == 512U && depth_ == 256U
            && outBatch_ == 1U && outHeight_ == 20U && outWidth_ == 1024U
            && cropTop_ == 0U && cropBottom_ == 0U && cropLeft_ == 0U && cropRight_ == 0U
            && blockSize_ == 2U && rowCount_ == 20U && enableRowTileCompose_ && rowTilePixels_ > 0U
            && (depth_ % ElementsPerBlock()) == 0U;
    }

    __aicore__ inline void Poison()
    {
        if (GetBlockIdx() == 0U && outLength_ > 0U) {
            yGm_.SetValue(0, static_cast<DT_X>(12345.0f));
        }
    }

    __aicore__ inline void ProcessT8RowTileTasks()
    {
        const uint32_t tileCount = (outWidth_ + rowTilePixels_ - 1U) / rowTilePixels_;
        if (tileCount == 0U) {
            return;
        }
        const uint32_t taskCount = rowCount_ * tileCount;
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t tasksPerCore = (taskCount + blockNum - 1U) / blockNum;
        const uint32_t startTask = blockIdx * tasksPerCore;
        if (startTask >= taskCount) {
            return;
        }
        const uint32_t endTask = MinU32(startTask + tasksPerCore, taskCount);
        for (uint32_t task = startTask; task < endTask; ++task) {
            const uint32_t tileId = task / rowCount_;
            const uint32_t rowInTile = task - tileId * rowCount_;
            const uint32_t rowId = rowInTile;
            const uint32_t outN = rowId / outHeight_;
            const uint32_t yH = rowId - outN * outHeight_;
            const uint32_t colStart = tileId * rowTilePixels_;
            const uint32_t curPixels = MinU32(rowTilePixels_, outWidth_ - colStart);
            CopyRowTile(outN, yH, colStart, curPixels);
        }
    }

    __aicore__ inline void Process()
    {
        if (outLength_ == 0 || pixelCount_ == 0 || rowCount_ == 0 || depth_ == 0 || blockSize_ == 0) {
            return;
        }

        const uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= usedLargeCoreNum_) {
            return;
        }
        uint32_t rowLen;
        uint32_t startRow;
        if (blockIdx < tailRowBlockNum_) {
            rowLen = bigCoreRowNum_;
            startRow = blockIdx * bigCoreRowNum_;
        } else {
            rowLen = smallCoreRowNum_;
            startRow = tailRowBlockNum_ * bigCoreRowNum_
                + (blockIdx - tailRowBlockNum_) * smallCoreRowNum_;
        }
        if (rowLen == 0) {
            return;
        }
        if (startRow >= rowCount_) {
            return;
        }
        const uint32_t endRow = MinU32(startRow + rowLen, rowCount_);
        for (uint32_t rowId = startRow; rowId < endRow; ++rowId) {
            uint32_t out_n = rowId / outHeight_;
            uint32_t y_h = rowId - out_n * outHeight_;
            for (uint32_t colStart = 0; colStart < outWidth_; colStart += rowTilePixels_) {
                uint32_t curPixels = MinU32(rowTilePixels_, outWidth_ - colStart);
                CopyRowTile(out_n, y_h, colStart, curPixels);
            }
        }
    }

    __aicore__ inline void ProcessPixelLegacy()
    {
        if (outLength_ == 0 || pixelCount_ == 0 || depth_ == 0 || blockSize_ == 0) {
            return;
        }

        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t pixelsPerCore = (pixelCount_ + blockNum - 1U) / blockNum;
        const uint32_t startPixel = blockIdx * pixelsPerCore;
        if (startPixel >= pixelCount_) {
            return;
        }
        const uint32_t endPixel = MinU32(startPixel + pixelsPerCore, pixelCount_);
        const uint32_t outHw = outHeight_ * outWidth_;
        for (uint32_t pixel = startPixel; pixel < endPixel; ++pixel) {
            uint32_t out_n = pixel / outHw;
            uint32_t rem = pixel - out_n * outHw;
            uint32_t y_h = rem / outWidth_;
            uint32_t y_w = rem - y_h * outWidth_;

            uint32_t expanded_y_h = y_h + cropTop_;
            uint32_t expanded_y_w = y_w + cropLeft_;
            uint32_t block_h = expanded_y_h % blockSize_;
            uint32_t in_h = expanded_y_h / blockSize_;
            uint32_t block_w = expanded_y_w % blockSize_;
            uint32_t in_w = expanded_y_w / blockSize_;
            uint32_t in_n = (block_h * blockSize_ + block_w) * outBatch_ + out_n;

            uint32_t y_base = pixel * depth_;
            uint32_t x_base = ((in_n * height_ + in_h) * width_ + in_w) * depth_;
            CopyDepthSegment(x_base, y_base);
        }
    }

    __aicore__ inline void CopyDepthSegment(uint32_t xBase, uint32_t yBase)
    {
        for (uint32_t offset = 0; offset < depth_; offset += tileLength_) {
            uint32_t count = MinU32(tileLength_, depth_ - offset);
            CopyChunk(xBase + offset, yBase + offset, count);
        }
    }

    __aicore__ inline void CopyChunk(uint32_t xOffset, uint32_t yOffset, uint32_t count)
    {
        LocalTensor<DT_X> xLocal = inQueue_.template AllocTensor<DT_X>();
        uint32_t blockBytes = count * sizeof(DT_X);
        uint32_t rightPadding = BtsAlignUp<DT_X>(count, ElementsPerBlock()) - count;
        DataCopyExtParams copyInParams{1, blockBytes, 0, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{true, 0, static_cast<uint8_t>(rightPadding), static_cast<DT_X>(0)};
        DataCopyPad(xLocal, xGm_[xOffset], copyInParams, padParams);
        inQueue_.EnQue(xLocal);

        event_t eventMte2ToV = static_cast<event_t>(0);
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);

        LocalTensor<DT_X> xReady = inQueue_.template DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue_.template AllocTensor<DT_X>();
        BtsIdentity<DT_X>(yLocal, xReady, count);
        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xReady);

        event_t eventVToMte3 = static_cast<event_t>(1);
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);

        LocalTensor<DT_X> yReady = outQueue_.template DeQue<DT_X>();
        DataCopyExtParams copyOutParams{1, blockBytes, 0, 0, 0};
        DataCopyPad(yGm_[yOffset], yReady, copyOutParams);
        outQueue_.FreeTensor(yReady);
    }

    __aicore__ inline void CopyRowTile(uint32_t outN, uint32_t yH, uint32_t colStart, uint32_t pixelCount)
    {
        if (CanUseRowTileResidueRunsBlock4(pixelCount)) {
            CopyRowTileResidueRunsBlock4(outN, yH, colStart, pixelCount);
            return;
        }
        if (CanUseRowTileResidueRunsBlock2Mid(pixelCount) || CanUseRowTileResidueRunsBlock2Wide(pixelCount)
            || CanUseRowTileResidueRunsBlock2Small(pixelCount)) {
            CopyRowTileResidueRunsBlock2(outN, yH, colStart, pixelCount);
            return;
        }

        LocalTensor<DT_X> xLocal = inQueue_.template AllocTensor<DT_X>();
        for (uint32_t i = 0; i < pixelCount; ++i) {
            uint32_t y_w = colStart + i;
            uint32_t expanded_y_h = yH + cropTop_;
            uint32_t expanded_y_w = y_w + cropLeft_;
            uint32_t block_h = expanded_y_h % blockSize_;
            uint32_t in_h = expanded_y_h / blockSize_;
            uint32_t block_w = expanded_y_w % blockSize_;
            uint32_t in_w = expanded_y_w / blockSize_;
            uint32_t in_n = (block_h * blockSize_ + block_w) * outBatch_ + outN;
            uint32_t x_base = ((in_n * height_ + in_h) * width_ + in_w) * depth_;
            uint32_t dst_offset = i * depth_;
            CopyInputSegment(xLocal, dst_offset, x_base, depth_);
        }
        inQueue_.EnQue(xLocal);

        event_t eventMte2ToV = static_cast<event_t>(0);
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);

        LocalTensor<DT_X> xReady = inQueue_.template DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue_.template AllocTensor<DT_X>();
        uint32_t rowTileElems = pixelCount * depth_;
        BtsIdentity<DT_X>(yLocal, xReady, rowTileElems);
        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xReady);

        event_t eventVToMte3 = static_cast<event_t>(1);
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);

        LocalTensor<DT_X> yReady = outQueue_.template DeQue<DT_X>();
        uint32_t y_base = ((outN * outHeight_ + yH) * outWidth_ + colStart) * depth_;
        uint32_t blockBytes = rowTileElems * sizeof(DT_X);
        DataCopyExtParams copyOutParams{1, blockBytes, 0, 0, 0};
        DataCopyPad(yGm_[y_base], yReady, copyOutParams);
        outQueue_.FreeTensor(yReady);
    }

    __aicore__ inline bool CanUseRowTileResidueRunsBlock4(uint32_t pixelCount) const
    {
        return pixelCount_ >= 57345U && pixelCount_ <= 61440U && blockSize_ == 4U && depth_ > 0U
            && outWidth_ > 512U && pixelCount >= 4U && (depth_ % ElementsPerBlock()) == 0U;
    }

    __aicore__ inline bool CanUseRowTileResidueRunsBlock2Mid(uint32_t pixelCount) const
    {
        return pixelCount_ >= 6145U && pixelCount_ <= 7168U && blockSize_ == 2U
            && depth_ > 0U && outWidth_ >= 33U && outWidth_ <= 64U && pixelCount >= 2U
            && (depth_ % ElementsPerBlock()) == 0U;
    }

    __aicore__ inline bool CanUseRowTileResidueRunsBlock2Wide(uint32_t pixelCount) const
    {
        return pixelCount_ >= 16385U && pixelCount_ <= 20480U && blockSize_ == 2U
            && depth_ > 0U && outWidth_ > 512U && pixelCount >= 2U
            && (depth_ % ElementsPerBlock()) == 0U;
    }

    __aicore__ inline bool CanUseRowTileResidueRunsBlock2Small(uint32_t pixelCount) const
    {
        return pixelCount_ >= 2049U && pixelCount_ <= 4096U && blockSize_ == 2U
            && depth_ > 0U && outWidth_ >= 17U && outWidth_ <= 32U && pixelCount >= 2U
            && (depth_ % ElementsPerBlock()) == 0U;
    }

    __aicore__ inline void CopyResidueRunBlock2(
        LocalTensor<DT_X> dst, uint32_t dstOffset, uint32_t xBase, uint32_t runCount)
    {
        uint32_t remaining = runCount;
        uint32_t curDst = dstOffset;
        uint32_t curX = xBase;
        const uint32_t blockLenBytes = depth_ * sizeof(DT_X);
        const uint32_t dstStrideBlocks = (depth_ * sizeof(DT_X)) / 32U;
        DataCopyPadExtParams<DT_X> padParams{true, 0, 0, static_cast<DT_X>(0)};
        while (remaining > 0U) {
            const uint32_t chunk = MinU32(remaining, 255U);
            DataCopyExtParams copyParams{static_cast<uint16_t>(chunk), blockLenBytes, 0, dstStrideBlocks, 0};
            DataCopyPad(dst[curDst], xGm_[curX], copyParams, padParams);
            remaining -= chunk;
            curX += chunk * depth_;
            curDst += chunk * 2U * depth_;
        }
    }

    __aicore__ inline void CopyRowTileResidueRunsBlock2(
        uint32_t outN, uint32_t yH, uint32_t colStart, uint32_t pixelCount)
    {
        LocalTensor<DT_X> xLocal = inQueue_.template AllocTensor<DT_X>();

        const uint32_t colEnd = colStart + pixelCount;
        const uint32_t expandedYH = yH + cropTop_;
        const uint32_t blockH = expandedYH % blockSize_;
        const uint32_t inH = expandedYH / blockSize_;
        const uint32_t cropLeftResidue = cropLeft_ % blockSize_;

        for (uint32_t residueStep = 0; residueStep < 2U; ++residueStep) {
            const uint32_t r = 1U - residueStep;
            uint32_t firstYw = (r + blockSize_ - cropLeftResidue) % blockSize_;
            if (firstYw < colStart) {
                const uint32_t delta = colStart - firstYw;
                firstYw += ((delta + blockSize_ - 1U) / blockSize_) * blockSize_;
            }
            if (firstYw >= colEnd) {
                continue;
            }

            const uint32_t runCount = ((colEnd - 1U - firstYw) / blockSize_) + 1U;
            const uint32_t expandedYWFirst = firstYw + cropLeft_;
            const uint32_t inWFirst = expandedYWFirst / blockSize_;
            const uint32_t inN = (blockH * blockSize_ + r) * outBatch_ + outN;
            const uint32_t xBase = ((inN * height_ + inH) * width_ + inWFirst) * depth_;
            const uint32_t dstOffset = (firstYw - colStart) * depth_;
            CopyResidueRunBlock2(xLocal, dstOffset, xBase, runCount);
        }

        inQueue_.EnQue(xLocal);

        event_t eventMte2ToV = static_cast<event_t>(0);
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);

        LocalTensor<DT_X> xReady = inQueue_.template DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue_.template AllocTensor<DT_X>();
        uint32_t rowTileElems = pixelCount * depth_;
        BtsIdentity<DT_X>(yLocal, xReady, rowTileElems);
        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xReady);

        event_t eventVToMte3 = static_cast<event_t>(1);
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);

        LocalTensor<DT_X> yReady = outQueue_.template DeQue<DT_X>();
        uint32_t yBase = ((outN * outHeight_ + yH) * outWidth_ + colStart) * depth_;
        uint32_t blockBytes = rowTileElems * sizeof(DT_X);
        DataCopyExtParams copyOutParams{1, blockBytes, 0, 0, 0};
        DataCopyPad(yGm_[yBase], yReady, copyOutParams);
        outQueue_.FreeTensor(yReady);
    }

    __aicore__ inline void CopyResidueRunBlock4(
        LocalTensor<DT_X> dst, uint32_t dstOffset, uint32_t xBase, uint32_t runCount)
    {
        uint32_t remaining = runCount;
        uint32_t curDst = dstOffset;
        uint32_t curX = xBase;
        const uint32_t blockLenBytes = depth_ * sizeof(DT_X);
        const uint32_t dstStrideBlocks = (3U * depth_ * sizeof(DT_X)) / 32U;
        DataCopyPadExtParams<DT_X> padParams{true, 0, 0, static_cast<DT_X>(0)};
        while (remaining > 0U) {
            const uint32_t chunk = MinU32(remaining, 255U);
            DataCopyExtParams copyParams{static_cast<uint16_t>(chunk), blockLenBytes, 0, dstStrideBlocks, 0};
            DataCopyPad(dst[curDst], xGm_[curX], copyParams, padParams);
            remaining -= chunk;
            curX += chunk * depth_;
            curDst += chunk * 4U * depth_;
        }
    }

    __aicore__ inline void CopyRowTileResidueRunsBlock4(
        uint32_t outN, uint32_t yH, uint32_t colStart, uint32_t pixelCount)
    {
        LocalTensor<DT_X> xLocal = inQueue_.template AllocTensor<DT_X>();

        const uint32_t colEnd = colStart + pixelCount;
        const uint32_t expandedYH = yH + cropTop_;
        const uint32_t blockH = expandedYH % blockSize_;
        const uint32_t inH = expandedYH / blockSize_;
        const uint32_t cropLeftResidue = cropLeft_ % blockSize_;

        for (uint32_t r = 0; r < 4U; ++r) {
            uint32_t firstYw = (r + blockSize_ - cropLeftResidue) % blockSize_;
            if (firstYw < colStart) {
                const uint32_t delta = colStart - firstYw;
                firstYw += ((delta + blockSize_ - 1U) / blockSize_) * blockSize_;
            }
            if (firstYw >= colEnd) {
                continue;
            }

            const uint32_t runCount = ((colEnd - 1U - firstYw) / blockSize_) + 1U;
            const uint32_t expandedYWFirst = firstYw + cropLeft_;
            const uint32_t inWFirst = expandedYWFirst / blockSize_;
            const uint32_t inN = (blockH * blockSize_ + r) * outBatch_ + outN;
            const uint32_t xBase = ((inN * height_ + inH) * width_ + inWFirst) * depth_;
            const uint32_t dstOffset = (firstYw - colStart) * depth_;
            CopyResidueRunBlock4(xLocal, dstOffset, xBase, runCount);
        }

        inQueue_.EnQue(xLocal);

        event_t eventMte2ToV = static_cast<event_t>(0);
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);

        LocalTensor<DT_X> xReady = inQueue_.template DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue_.template AllocTensor<DT_X>();
        uint32_t rowTileElems = pixelCount * depth_;
        BtsIdentity<DT_X>(yLocal, xReady, rowTileElems);
        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xReady);

        event_t eventVToMte3 = static_cast<event_t>(1);
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);

        LocalTensor<DT_X> yReady = outQueue_.template DeQue<DT_X>();
        uint32_t yBase = ((outN * outHeight_ + yH) * outWidth_ + colStart) * depth_;
        uint32_t blockBytes = rowTileElems * sizeof(DT_X);
        DataCopyExtParams copyOutParams{1, blockBytes, 0, 0, 0};
        DataCopyPad(yGm_[yBase], yReady, copyOutParams);
        outQueue_.FreeTensor(yReady);
    }

    __aicore__ inline void CopyInputSegment(LocalTensor<DT_X> dst, uint32_t dstOffset, uint32_t xOffset, uint32_t count)
    {
        uint32_t blockBytes = count * sizeof(DT_X);
        uint32_t rightPadding = BtsAlignUp<DT_X>(count, ElementsPerBlock()) - count;
        DataCopyExtParams copyInParams{1, blockBytes, 0, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{true, 0, static_cast<uint8_t>(rightPadding), static_cast<DT_X>(0)};
        DataCopyPad(dst[dstOffset], xGm_[xOffset], copyInParams, padParams);
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, 1> inQueue_;
    TQue<TPosition::VECOUT, 1> outQueue_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t height_ = 0;
    uint32_t width_ = 0;
    uint32_t depth_ = 0;
    uint32_t outBatch_ = 0;
    uint32_t outHeight_ = 0;
    uint32_t outWidth_ = 0;
    uint32_t cropTop_ = 0;
    uint32_t cropBottom_ = 0;
    uint32_t cropLeft_ = 0;
    uint32_t cropRight_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t inputLength_ = 0;
    uint32_t outLength_ = 0;
    uint32_t pixelCount_ = 0;
    uint32_t tileLength_ = 1;
    uint32_t localLength_ = 1;
    uint32_t rowCount_ = 0;
    uint32_t rowTilePixels_ = 1;
    uint32_t rowTileElems_ = 1;
    uint32_t usedLargeCoreNum_ = 1;
    uint32_t smallCoreRowNum_ = 0;
    uint32_t bigCoreRowNum_ = 0;
    uint32_t tailRowBlockNum_ = 0;
    bool enableRowTileCompose_ = false;
};

template <class DT_X>
class KernelBatchToSpaceT9RowTileSingleBurstLocalCopyInterleave913aT9 {
public:
    __aicore__ inline KernelBatchToSpaceT9RowTileSingleBurstLocalCopyInterleave913aT9() {}

    __aicore__ inline void InitAndProcess(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling)
    {
        height_ = tiling.height;
        width_ = tiling.width;
        depth_ = tiling.depth;
        outBatch_ = tiling.outBatch;
        outHeight_ = tiling.outHeight;
        outWidth_ = tiling.outWidth;
        cropTop_ = tiling.cropTop;
        cropLeft_ = tiling.cropLeft;
        blockSize_ = tiling.blockSize;
        inputLength_ = tiling.inputLength;
        outLength_ = tiling.outLength;
        pixelCount_ = tiling.pixelCount;
        rowCount_ = tiling.rowCount;
        (void)tiling.rowTilePixels;
        rowTilePixels_ = 576U;
        usedLargeCoreNum_ = tiling.usedLargeCoreNum == 0 ? 1U : tiling.usedLargeCoreNum;
        smallCoreRowNum_ = tiling.smallCoreRowNum;
        bigCoreRowNum_ = tiling.bigCoreRowNum;
        tailRowBlockNum_ = tiling.tailRowBlockNum;
        localLength_ = BtsAlignUp<DT_X>(rowTilePixels_ * (depth_ == 0 ? 1U : depth_), ElementsPerBlock());

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, inputLength_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, outLength_);
        pipe_.InitBuffer(inQueue_, 1, localLength_ * sizeof(DT_X));
        pipe_.InitBuffer(outQueue_, 1, localLength_ * sizeof(DT_X));

        Process();
    }

private:
    __aicore__ inline uint32_t ElementsPerBlock() const
    {
        return 32U / sizeof(DT_X);
    }

    __aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline bool CanRunTarget() const
    {
        return sizeof(DT_X) == 2U && pixelCount_ == 61400U && height_ == 10U && width_ == 512U
            && depth_ == 64U && outBatch_ == 1U && outHeight_ == 40U && outWidth_ == 1535U
            && cropTop_ == 0U && cropLeft_ == 513U && blockSize_ == 4U && rowCount_ > 0U
            && rowTilePixels_ > 0U;
    }

    __aicore__ inline void Poison()
    {
        if (GetBlockIdx() == 0U && outLength_ > 0U) {
            yGm_.SetValue(0, static_cast<DT_X>(12345.0f));
        }
    }

    __aicore__ inline void Process()
    {
        if (!CanRunTarget()) {
            Poison();
            return;
        }

        const uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= usedLargeCoreNum_) {
            return;
        }

        uint32_t rowLen;
        uint32_t startRow;
        if (blockIdx < tailRowBlockNum_) {
            rowLen = bigCoreRowNum_;
            startRow = blockIdx * bigCoreRowNum_;
        } else {
            rowLen = smallCoreRowNum_;
            startRow = tailRowBlockNum_ * bigCoreRowNum_ + (blockIdx - tailRowBlockNum_) * smallCoreRowNum_;
        }
        if (rowLen == 0U || startRow >= rowCount_) {
            return;
        }

        const uint32_t endRow = MinU32(startRow + rowLen, rowCount_);
        for (uint32_t rowId = startRow; rowId < endRow; ++rowId) {
            const uint32_t outN = 0U;
            const uint32_t yH = rowId;
            for (uint32_t colStart = 0U; colStart < outWidth_; colStart += rowTilePixels_) {
                const uint32_t curPixels = MinU32(rowTilePixels_, outWidth_ - colStart);
                CopyRowTile(outN, yH, colStart, curPixels);
            }
        }
    }

    __aicore__ inline void CopyRowTile(uint32_t outN, uint32_t yH, uint32_t colStart, uint32_t pixelCount)
    {
        LocalTensor<DT_X> xLocal = inQueue_.template AllocTensor<DT_X>();
        FillRowTileDenseSingleBurst(xLocal, outN, yH, colStart, pixelCount);
        inQueue_.EnQue(xLocal);

        event_t eventMte2ToV = static_cast<event_t>(0);
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);

        LocalTensor<DT_X> xReady = inQueue_.template DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue_.template AllocTensor<DT_X>();
        InterleaveDenseRunsLocal(yLocal, xReady, yH, colStart, pixelCount);
        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xReady);

        event_t eventMte2ToMte3 = static_cast<event_t>(1);
        SetFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3);
        WaitFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3);

        LocalTensor<DT_X> yReady = outQueue_.template DeQue<DT_X>();
        const uint32_t yBase = ((outN * outHeight_ + yH) * outWidth_ + colStart) * depth_;
        const uint32_t blockBytes = pixelCount * depth_ * sizeof(DT_X);
        DataCopyExtParams copyOutParams{1, blockBytes, 0, 0, 0};
        DataCopyPad(yGm_[yBase], yReady, copyOutParams);
        outQueue_.FreeTensor(yReady);
    }

    __aicore__ inline void FillRowTileDenseSingleBurst(
        LocalTensor<DT_X> dst, uint32_t outN, uint32_t yH, uint32_t colStart, uint32_t pixelCount)
    {
        const uint32_t colEnd = colStart + pixelCount;
        const uint32_t expandedYH = yH + cropTop_;
        const uint32_t inH = expandedYH >> 2U;
        const uint32_t cropLeftResidue = cropLeft_ & 3U;
        uint32_t denseOffset = 0U;

        for (uint32_t r = 0U; r < 4U; ++r) {
            uint32_t firstYw = (r + 4U - cropLeftResidue) & 3U;
            if (firstYw < colStart) {
                const uint32_t delta = colStart - firstYw;
                firstYw += (((delta + 3U) >> 2U) << 2U);
            }
            if (firstYw >= colEnd) {
                continue;
            }

            const uint32_t runCount = ((colEnd - 1U - firstYw) >> 2U) + 1U;
            const uint32_t expandedYWFirst = firstYw + cropLeft_;
            const uint32_t inWFirst = expandedYWFirst >> 2U;
            const uint32_t inN = (((expandedYH & 3U) << 2U) + r) * outBatch_ + outN;
            const uint32_t xBase = ((inN * height_ + inH) * width_ + inWFirst) * depth_;
            CopyResidueRunSingleBurst(dst, denseOffset, xBase, runCount);
            denseOffset += runCount * depth_;
        }
    }

    __aicore__ inline void InterleaveDenseRunsLocal(
        LocalTensor<DT_X> dst, LocalTensor<DT_X> src, uint32_t yH, uint32_t colStart, uint32_t pixelCount)
    {
        const uint32_t colEnd = colStart + pixelCount;
        const uint32_t cropLeftResidue = cropLeft_ & 3U;
        uint32_t denseOffset = 0U;

        for (uint32_t r = 0U; r < 4U; ++r) {
            uint32_t firstYw = (r + 4U - cropLeftResidue) & 3U;
            if (firstYw < colStart) {
                const uint32_t delta = colStart - firstYw;
                firstYw += (((delta + 3U) >> 2U) << 2U);
            }
            if (firstYw >= colEnd) {
                continue;
            }

            const uint32_t runCount = ((colEnd - 1U - firstYw) >> 2U) + 1U;
            const uint32_t dstOffset = (firstYw - colStart) * depth_;
            CopyDenseRunToOutputLocal(dst, dstOffset, src, denseOffset, runCount);
            denseOffset += runCount * depth_;
        }
        (void)yH;
    }

    __aicore__ inline void CopyResidueRunSingleBurst(
        LocalTensor<DT_X> dst, uint32_t dstOffset, uint32_t xBase, uint32_t runCount)
    {
        uint32_t remaining = runCount;
        uint32_t curDst = dstOffset;
        uint32_t curX = xBase;
        DataCopyPadExtParams<DT_X> padParams{true, 0, 0, static_cast<DT_X>(0)};
        while (remaining > 0U) {
            const uint32_t chunk = MinU32(remaining, 255U);
            const uint32_t blockBytes = chunk * depth_ * sizeof(DT_X);
            DataCopyExtParams copyParams{1, blockBytes, 0, 0, 0};
            DataCopyPad(dst[curDst], xGm_[curX], copyParams, padParams);
            remaining -= chunk;
            curX += chunk * depth_;
            curDst += chunk * depth_;
        }
    }

    __aicore__ inline void CopyDenseRunToOutputLocal(
        LocalTensor<DT_X> dst, uint32_t dstOffset, LocalTensor<DT_X> src, uint32_t srcOffset, uint32_t runCount)
    {
        uint32_t remaining = runCount;
        uint32_t curDst = dstOffset;
        uint32_t curSrc = srcOffset;
        const uint32_t blockLenBlocks = depth_ / ElementsPerBlock();
        const uint32_t dstStrideBlocks = (3U * depth_ * sizeof(DT_X)) / 32U;
        while (remaining > 0U) {
            const uint32_t chunk = MinU32(remaining, 255U);
            DataCopyParams copyParams{
                static_cast<uint16_t>(chunk),
                static_cast<uint16_t>(blockLenBlocks),
                0,
                static_cast<uint16_t>(dstStrideBlocks)
            };
            DataCopy(dst[curDst], src[curSrc], copyParams);
            remaining -= chunk;
            curSrc += chunk * depth_;
            curDst += chunk * 4U * depth_;
        }
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, 1> inQueue_;
    TQue<TPosition::VECOUT, 1> outQueue_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t height_ = 0;
    uint32_t width_ = 0;
    uint32_t depth_ = 0;
    uint32_t outBatch_ = 0;
    uint32_t outHeight_ = 0;
    uint32_t outWidth_ = 0;
    uint32_t cropTop_ = 0;
    uint32_t cropLeft_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t inputLength_ = 0;
    uint32_t outLength_ = 0;
    uint32_t pixelCount_ = 0;
    uint32_t rowCount_ = 0;
    uint32_t rowTilePixels_ = 1;
    uint32_t localLength_ = 1;
    uint32_t usedLargeCoreNum_ = 1;
    uint32_t smallCoreRowNum_ = 0;
    uint32_t bigCoreRowNum_ = 0;
    uint32_t tailRowBlockNum_ = 0;
};

template <class DT_X>
class KernelBatchToSpaceT10ColumnStridedLocalCopy1002aT10 {
public:
    __aicore__ inline KernelBatchToSpaceT10ColumnStridedLocalCopy1002aT10() {}

    __aicore__ inline void InitAndProcess(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling)
    {
        height_ = tiling.height;
        width_ = tiling.width;
        depth_ = tiling.depth;
        outBatch_ = tiling.outBatch;
        outHeight_ = tiling.outHeight;
        outWidth_ = tiling.outWidth;
        cropTop_ = tiling.cropTop;
        cropBottom_ = tiling.cropBottom;
        cropLeft_ = tiling.cropLeft;
        cropRight_ = tiling.cropRight;
        blockSize_ = tiling.blockSize;
        inputLength_ = tiling.inputLength;
        outLength_ = tiling.outLength;
        pixelCount_ = tiling.pixelCount;
        rowCount_ = tiling.rowCount;
        usedLargeCoreNum_ = tiling.usedLargeCoreNum == 0 ? 1U : tiling.usedLargeCoreNum;
        smallCoreRowNum_ = tiling.smallCoreRowNum;
        bigCoreRowNum_ = tiling.bigCoreRowNum;
        tailRowBlockNum_ = tiling.tailRowBlockNum;
        stripRows_ = tiling.stripRows == 0 ? 1U : tiling.stripRows;
        stripElems_ = tiling.stripElems == 0 ? depth_ : tiling.stripElems;
        localLength_ = BtsAlignUp<DT_X>(stripElems_, ElementsPerBlock());

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, inputLength_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, outLength_);
        pipe_.InitBuffer(inQueue_, 1, localLength_ * sizeof(DT_X));
        pipe_.InitBuffer(outQueue_, 1, localLength_ * sizeof(DT_X));

        Process();
    }

private:
    __aicore__ inline uint32_t ElementsPerBlock() const
    {
        return 32U / sizeof(DT_X);
    }

    __aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline bool CanRunTarget() const
    {
        return sizeof(DT_X) == 2U && pixelCount_ == 98304U && height_ == 1024U && width_ == 6U
            && depth_ == 32U && outBatch_ == 4U && outHeight_ == 2048U && outWidth_ == 12U
            && cropTop_ == 0U && cropBottom_ == 0U && cropLeft_ == 0U && cropRight_ == 0U
            && blockSize_ == 2U && rowCount_ > 0U && stripRows_ >= 2U && stripElems_ > 0U;
    }

    __aicore__ inline void Poison()
    {
        if (GetBlockIdx() == 0U && outLength_ > 0U) {
            yGm_.SetValue(0, static_cast<DT_X>(12345.0f));
        }
    }

    __aicore__ inline void Process()
    {
        if (!CanRunTarget()) {
            Poison();
            return;
        }

        const uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= usedLargeCoreNum_) {
            return;
        }

        uint32_t rowLen;
        uint32_t startRow;
        if (blockIdx < tailRowBlockNum_) {
            rowLen = bigCoreRowNum_;
            startRow = blockIdx * bigCoreRowNum_;
        } else {
            rowLen = smallCoreRowNum_;
            startRow = tailRowBlockNum_ * bigCoreRowNum_ + (blockIdx - tailRowBlockNum_) * smallCoreRowNum_;
        }
        if (rowLen == 0U || startRow >= rowCount_) {
            return;
        }

        const uint32_t endRow = MinU32(startRow + rowLen, rowCount_);
        uint32_t rowId = startRow;
        while (rowId < endRow) {
            const uint32_t outN = rowId / outHeight_;
            const uint32_t yStart = rowId - outN * outHeight_;
            const uint32_t rowsLeftInBatch = outHeight_ - yStart;
            uint32_t curRows = MinU32(stripRows_, endRow - rowId);
            curRows = MinU32(curRows, rowsLeftInBatch);
            CopyStrip(outN, yStart, curRows);
            rowId += curRows;
        }
    }

    __aicore__ inline void CopyStrip(uint32_t outN, uint32_t yStart, uint32_t rowCount)
    {
        LocalTensor<DT_X> xLocal = inQueue_.template AllocTensor<DT_X>();
        FillStripGroupedDenseSingleBurst(xLocal, outN, yStart, rowCount);
        inQueue_.EnQue(xLocal);

        event_t eventMte2ToV = static_cast<event_t>(0);
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);

        LocalTensor<DT_X> xReady = inQueue_.template DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue_.template AllocTensor<DT_X>();
        InterleaveGroupedDenseRowsLocal(yLocal, xReady, yStart, rowCount);
        inQueue_.FreeTensor(xReady);

        event_t eventVToMte3 = static_cast<event_t>(0);
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);

        const uint32_t totalElems = rowCount * outWidth_ * depth_;
        const uint32_t yBase = ((outN * outHeight_ + yStart) * outWidth_) * depth_;
        const uint32_t blockBytes = totalElems * sizeof(DT_X);
        DataCopyExtParams copyOutParams{1, blockBytes, 0, 0, 0};
        DataCopyPad(yGm_[yBase], yLocal, copyOutParams);
        outQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void FillStripGroupedDenseSingleBurst(
        LocalTensor<DT_X> dst, uint32_t outN, uint32_t yStart, uint32_t rowCount)
    {
        uint32_t denseOffset = 0U;
        const uint32_t firstStripParity = yStart & 1U;
        for (uint32_t blockH = 0U; blockH < 2U; ++blockH) {
            const uint32_t firstRowInStrip = (blockH + 2U - firstStripParity) & 1U;
            if (firstRowInStrip >= rowCount) {
                continue;
            }
            const uint32_t groupRows = ((rowCount - 1U - firstRowInStrip) >> 1U) + 1U;
            const uint32_t firstYH = yStart + firstRowInStrip;
            const uint32_t firstInH = firstYH >> 1U;
            for (uint32_t r = 0U; r < 2U; ++r) {
                const uint32_t inN = ((blockH << 1U) + r) * outBatch_ + outN;
                const uint32_t xBase = ((inN * height_ + firstInH) * width_) * depth_;
                const uint32_t runPixels = groupRows * width_;
                if (runPixels == 0U) {
                    continue;
                }
                CopyResidueRunSingleBurst(dst, denseOffset, xBase, runPixels);
                denseOffset += runPixels * depth_;
            }
        }
    }

    __aicore__ inline void InterleaveGroupedDenseRowsLocal(
        LocalTensor<DT_X> dst, LocalTensor<DT_X> src, uint32_t yStart, uint32_t rowCount)
    {
        uint32_t denseOffset = 0U;
        const uint32_t firstStripParity = yStart & 1U;
        for (uint32_t blockH = 0U; blockH < 2U; ++blockH) {
            const uint32_t firstRowInStrip = (blockH + 2U - firstStripParity) & 1U;
            if (firstRowInStrip >= rowCount) {
                continue;
            }
            const uint32_t groupRows = ((rowCount - 1U - firstRowInStrip) >> 1U) + 1U;
            for (uint32_t r = 0U; r < 2U; ++r) {
                CopyGroupedRowsToOutputLocalT10(dst, src, denseOffset, firstRowInStrip, groupRows, r);
                denseOffset += groupRows * width_ * depth_;
            }
        }
    }

    __aicore__ inline void CopyGroupedRowsToOutputLocalT10(LocalTensor<DT_X> dst, LocalTensor<DT_X> src,
        uint32_t srcOffset, uint32_t firstRowInStrip, uint32_t groupRows, uint32_t residue)
    {
        const uint32_t rowStrideSrc = 192U;
        const uint32_t rowStrideDst = 384U;
        const uint32_t baseDst = firstRowInStrip * rowStrideDst + residue * 32U;
        const uint16_t blockLen = 2U;
        const uint16_t srcStride = 10U;
        const uint16_t dstStride = 46U;
        for (uint32_t col = 0U; col < 6U; ++col) {
            uint32_t remaining = groupRows;
            uint32_t curSrc = srcOffset + col * 32U;
            uint32_t curDst = baseDst + (col << 1U) * 32U;
            while (remaining > 0U) {
                const uint32_t chunk = MinU32(remaining, 255U);
                DataCopyParams copyParams{static_cast<uint16_t>(chunk), blockLen, srcStride, dstStride};
                DataCopy(dst[curDst], src[curSrc], copyParams);
                remaining -= chunk;
                curSrc += chunk * rowStrideSrc;
                curDst += chunk * (rowStrideDst << 1U);
            }
        }
    }

    __aicore__ inline void CopyResidueRunSingleBurst(
        LocalTensor<DT_X> dst, uint32_t dstOffset, uint32_t xBase, uint32_t runCount)
    {
        uint32_t remaining = runCount;
        uint32_t curDst = dstOffset;
        uint32_t curX = xBase;
        DataCopyPadExtParams<DT_X> padParams{true, 0, 0, static_cast<DT_X>(0)};
        while (remaining > 0U) {
            const uint32_t chunk = MinU32(remaining, 255U);
            const uint32_t blockBytes = chunk * depth_ * sizeof(DT_X);
            DataCopyExtParams copyParams{1, blockBytes, 0, 0, 0};
            DataCopyPad(dst[curDst], xGm_[curX], copyParams, padParams);
            remaining -= chunk;
            curX += chunk * depth_;
            curDst += chunk * depth_;
        }
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, 1> inQueue_;
    TQue<TPosition::VECOUT, 1> outQueue_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t height_ = 0;
    uint32_t width_ = 0;
    uint32_t depth_ = 0;
    uint32_t outBatch_ = 0;
    uint32_t outHeight_ = 0;
    uint32_t outWidth_ = 0;
    uint32_t cropTop_ = 0;
    uint32_t cropBottom_ = 0;
    uint32_t cropLeft_ = 0;
    uint32_t cropRight_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t inputLength_ = 0;
    uint32_t outLength_ = 0;
    uint32_t pixelCount_ = 0;
    uint32_t rowCount_ = 0;
    uint32_t usedLargeCoreNum_ = 1;
    uint32_t smallCoreRowNum_ = 0;
    uint32_t bigCoreRowNum_ = 0;
    uint32_t tailRowBlockNum_ = 0;
    uint32_t stripRows_ = 1;
    uint32_t stripElems_ = 1;
    uint32_t localLength_ = 1;
};


template <typename DT_X, bool IS_SINGLE_TILE>
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    (void)IS_SINGLE_TILE;
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingData, tiling_data, tiling);

    if (tiling_data.pixelCount < 229U) {
        KernelBatchToSpaceTinyDirectMerged8888d<DT_X> op;
        op.InitAndProcess(x, y, tiling_data);
        return;
    }
    if (tiling_data.pixelCount < 461U) {
        KernelBatchToSpaceT2GatherYBuf1219a<DT_X> op;
        op.InitAndProcess(x, y, tiling_data);
        return;
    }
    if (tiling_data.pixelCount < 1808U) {
        KernelBatchToSpaceT4ExactFullRow403dT4<DT_X> op;
        op.InitAndProcess(x, y, tiling_data);
        return;
    }
    if (tiling_data.pixelCount < 4704U) {
        KernelBatchToSpaceAlignedBlock2RowMerged8888c<DT_X> op;
        op.InitAndProcess(x, y, tiling_data);
        return;
    }
    if (tiling_data.pixelCount < 13376U) {
        KernelBatchToSpaceAlignedBlock2RowMerged8888c<DT_X> op;
        op.InitAndProcess(x, y, tiling_data);
        return;
    }
    if (tiling_data.pixelCount < 40940U) {
        KernelBatchToSpaceT8RowMajorBd24Tile172_824d<DT_X> op;
        op.InitAndProcess(x, y, tiling_data);
        return;
    }
    if (tiling_data.pixelCount < 62958U) {
        KernelBatchToSpaceT9RowTileSingleBurstLocalCopyInterleave913aT9<DT_X> op;
        op.InitAndProcess(x, y, tiling_data);
        return;
    }
    if (tiling_data.pixelCount < 81410U) {
        KernelBatchToSpaceT5BlockEvenForwardOddReverseOutQueue1_514dT5<DT_X> op;
        op.InitAndProcess(x, y, tiling_data);
        return;
    }
    if (tiling_data.pixelCount >= 81410U) {
        KernelBatchToSpaceT10ColumnStridedLocalCopy1002aT10<DT_X> op;
        op.InitAndProcess(x, y, tiling_data);
        return;
    }
    return;
}
