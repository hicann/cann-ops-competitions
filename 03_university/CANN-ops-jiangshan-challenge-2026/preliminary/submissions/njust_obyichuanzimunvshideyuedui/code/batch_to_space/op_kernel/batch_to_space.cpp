#include "kernel_operator.h"

#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

using namespace AscendC;

constexpr uint8_t BUFFER_NUM = 2;

template <class DT_X>
class KernelBatchToSpaceBase
{
public:
    __aicore__ inline KernelBatchToSpaceBase() {}

    __aicore__ inline void InitCommon(
        GM_ADDR x, GM_ADDR y, const __gm__ BatchToSpaceTilingData *tiling)
    {
        height_ = tiling->height;
        width_ = tiling->width;
        depth_ = tiling->depth;
        outBatch_ = tiling->outBatch;
        outHeight_ = tiling->outHeight;
        outWidth_ = tiling->outWidth;
        blockSize_ = tiling->blockSize;
        cropTop_ = tiling->cropTop;
        cropLeft_ = tiling->cropLeft;
        tileDepth_ = tiling->tileDepth;
        usedCoreNum_ = tiling->usedCoreNum;
        rowPackedTileRows_ = tiling->rowPackedTileRows;
        rowPackedTileCols_ = tiling->rowPackedTileCols;
        inputRowElements_ = width_ * depth_;
        outputRowElements_ = outWidth_ * depth_;
        outputPointsPerBatch_ = outHeight_ * outWidth_;
        alignElements_ = 32 / sizeof(DT_X);
        depthBytes_ = depth_ * sizeof(DT_X);

        const uint32_t blockIdx = GetBlockIdx();
        startPoint_ = blockIdx * tiling->pointsPerCore;
        pointCount_ = (blockIdx + 1 == tiling->usedCoreNum)
                          ? tiling->tailPoints
                          : tiling->pointsPerCore;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x),
                             tiling->inputLength);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y),
                             tiling->outputLength);
    }

    __aicore__ inline void InitMicroTileOffsetBuffer(
        const __gm__ BatchToSpaceTilingData *tiling)
    {
        microTileOffsetGm_.SetGlobalBuffer(
            (__gm__ uint32_t *)tiling->microTileGatherOffsets,
            BTS_MICRO_TILE_OFFSET_COUNT);
    }

    __aicore__ inline void InitCopyBuffer(TPipe *pipe)
    {
        const uint32_t alignElements =
            alignElements_ != 0 ? alignElements_ : 32 / sizeof(DT_X);
        uint32_t alignedTileDepth =
            (tileDepth_ + alignElements - 1) / alignElements * alignElements;
        tileBufferBytes_ = alignedTileDepth * sizeof(DT_X);
        pipe->InitBuffer(copyQ_, BUFFER_NUM, tileBufferBytes_);
    }

    __aicore__ inline void InitRowPackedBuffer(TPipe *pipe)
    {
        const bool paddedLayout = UsePaddedWideRowPackedLayout();
        const uint32_t depthStride =
            paddedLayout ? AlignUpBlockElements(depth_) : depth_;
        const uint32_t halfRowElements =
            paddedLayout ? width_ * depthStride
                         : AlignUpBlockElements(inputRowElements_);
        const uint32_t rowElements =
            paddedLayout
                ? outWidth_ * depthStride * rowPackedTileRows_
                : AlignUpBlockElements(outputRowElements_ *
                                       rowPackedTileRows_);
        pipe->InitBuffer(evenRowBuf_,
                         halfRowElements * rowPackedTileRows_ * sizeof(DT_X));
        pipe->InitBuffer(oddRowBuf_,
                         halfRowElements * rowPackedTileRows_ * sizeof(DT_X));
        pipe->InitBuffer(packedRowBuf_, rowElements * sizeof(DT_X));
        pipe->InitBuffer(evenRowBuf2_,
                         halfRowElements * rowPackedTileRows_ * sizeof(DT_X));
        pipe->InitBuffer(oddRowBuf2_,
                         halfRowElements * rowPackedTileRows_ * sizeof(DT_X));
        pipe->InitBuffer(packedRowBuf2_, rowElements * sizeof(DT_X));
    }

    __aicore__ inline void InitColumnPackedBuffer(TPipe *pipe)
    {
        const uint32_t inputCols = rowPackedTileCols_ >> 1;
        const uint32_t tileRows =
            rowPackedTileRows_ == 0 ? 1 : rowPackedTileRows_;
        const uint32_t inputElements =
            AlignUpBlockElements(inputCols * depth_) * tileRows;
        const uint32_t outputElements =
            AlignUpBlockElements(rowPackedTileCols_ * depth_) * tileRows;
        pipe->InitBuffer(evenRowBuf_, inputElements * sizeof(DT_X));
        pipe->InitBuffer(oddRowBuf_, inputElements * sizeof(DT_X));
        pipe->InitBuffer(packedRowBuf_, outputElements * sizeof(DT_X));
        pipe->InitBuffer(packedRowBuf2_, outputElements * sizeof(DT_X));
    }

    __aicore__ inline void InitHeightPackedBuffer(TPipe *pipe)
    {
        const uint32_t halfRowElements =
            AlignUpBlockElements(inputRowElements_ * rowPackedTileRows_);
        const uint32_t outputElements =
            AlignUpBlockElements(outputRowElements_ *
                                 (rowPackedTileRows_ << 1));
        pipe->InitBuffer(evenRowBuf_, halfRowElements * sizeof(DT_X));
        pipe->InitBuffer(oddRowBuf_, halfRowElements * sizeof(DT_X));
        pipe->InitBuffer(packedRowBuf_, outputElements * sizeof(DT_X));
        pipe->InitBuffer(evenRowBuf2_, halfRowElements * sizeof(DT_X));
        pipe->InitBuffer(oddRowBuf2_, halfRowElements * sizeof(DT_X));
        pipe->InitBuffer(packedRowBuf2_, outputElements * sizeof(DT_X));
    }

    __aicore__ inline void InitGenericRowPackedBuffer(TPipe *pipe)
    {
        const uint32_t effectiveWidth =
            rowPackedTileCols_ != 0 ? rowPackedTileCols_ : outWidth_;
        const uint32_t maxStreamCols =
            (effectiveWidth + blockSize_ - 1) / blockSize_;
        const uint32_t streamElements =
            AlignUpBlockElements(maxStreamCols * depth_);
        const uint32_t rowElements =
            AlignUpBlockElements(effectiveWidth * depth_);
        pipe->InitBuffer(evenRowBuf_, streamElements * sizeof(DT_X));
        pipe->InitBuffer(packedRowBuf_, rowElements * sizeof(DT_X));
    }

    __aicore__ inline void InitMicroTileBuffer(TPipe *pipe)
    {
        const uint32_t tileCols =
            rowPackedTileCols_ == 0 ? 16 : rowPackedTileCols_;
        const uint32_t streamCols = (tileCols + 1) >> 1;
        const uint32_t streamElements =
            AlignUpBlockElements(streamCols * depth_);
        const uint32_t outputElements =
            AlignUpBlockElements(tileCols * depth_);
        const uint32_t sourceElements = streamElements << 1;
        const uint32_t offsetElements =
            outputElements + AlignUpBlockElements(depth_);
        const uint32_t offsetBytes =
            AlignUpBlockBytes(offsetElements * sizeof(uint32_t));
        pipe->InitBuffer(evenRowBuf_, sourceElements * sizeof(DT_X) * BUFFER_NUM);
        pipe->InitBuffer(oddRowBuf_, offsetBytes);
        pipe->InitBuffer(packedRowBuf_, outputElements * sizeof(DT_X) * BUFFER_NUM);
    }

    __aicore__ inline void InitSmallNoCropBuffer(TPipe *pipe)
    {
        const uint32_t halfRowElements =
            AlignUpBlockElements(inputRowElements_);
        const uint32_t rowElements =
            AlignUpBlockElements(outputRowElements_);
        pipe->InitBuffer(evenRowBuf_, halfRowElements * sizeof(DT_X));
        pipe->InitBuffer(oddRowBuf_, halfRowElements * sizeof(DT_X));
        pipe->InitBuffer(packedRowBuf_, rowElements * sizeof(DT_X));
    }

    __aicore__ inline void InitLargeDepthInputPackedBuffer(TPipe *pipe)
    {
        const uint32_t tileCols =
            rowPackedTileCols_ == 0 ? outWidth_ : rowPackedTileCols_;
        const uint32_t streamCols = (tileCols + 1U) >> 1;
        const uint32_t streamElements =
            AlignUpBlockElements(streamCols * depth_);
        const uint32_t outputElements =
            AlignUpBlockElements(tileCols * depth_);
        pipe->InitBuffer(evenRowBuf_, streamElements * sizeof(DT_X));
        pipe->InitBuffer(oddRowBuf_, streamElements * sizeof(DT_X));
        pipe->InitBuffer(evenRowBuf2_, streamElements * sizeof(DT_X));
        pipe->InitBuffer(oddRowBuf2_, streamElements * sizeof(DT_X));
        pipe->InitBuffer(packedRowBuf_, outputElements * sizeof(DT_X));
        pipe->InitBuffer(packedRowBuf2_, outputElements * sizeof(DT_X));
    }

protected:
    __aicore__ inline uint32_t Nb() const { return outBatch_; }
    __aicore__ inline uint32_t Nh() const { return outHeight_; }
    __aicore__ inline uint32_t Nw() const { return outWidth_; }
    __aicore__ inline uint32_t Nd() const { return depth_; }
    __aicore__ inline uint32_t Ns() const { return startPoint_; }
    __aicore__ inline uint32_t Nc() const { return pointCount_; }
    __aicore__ inline TBuf<TPosition::VECCALC> &Sb() { return evenRowBuf_; }
    __aicore__ inline TBuf<TPosition::VECCALC> &Pb() { return packedRowBuf_; }
    __aicore__ inline uint32_t Tw() const { return rowPackedTileCols_; }

    __aicore__ inline void ProcessDirectCopy()
    {
        uint32_t pendingOffset = 0;
        uint32_t pendingCount = 0;
        bool hasPendingCopy = false;
        uint32_t done = 0;
        while (done < pointCount_)
        {
            const uint32_t remaining = pointCount_ - done;
            const uint32_t count =
                tileDepth_ < remaining ? tileDepth_ : remaining;
            const uint32_t offset = startPoint_ + done;
            CopyInSegment(offset, count);
            if (hasPendingCopy)
            {
                CopyOutSegment(pendingOffset, pendingCount);
            }
            pendingOffset = offset;
            pendingCount = count;
            hasPendingCopy = true;
            done += count;
        }
        if (hasPendingCopy)
        {
            CopyOutSegment(pendingOffset, pendingCount);
        }
    }

    __aicore__ inline void ProcessBlockSize2RowPacked()
    {
        if (cropTop_ == 0 && cropLeft_ == 0 && outHeight_ == (height_ << 1) &&
            outWidth_ == (width_ << 1))
        {
            ProcessBlockSize2RowPackedNoCrop();
            return;
        }
        if (UsePaddedWideRowPackedLayout())
        {
            ProcessBlockSize2RowPackedPadded();
            return;
        }

        LocalTensor<DT_X> evenLocal0 = evenRowBuf_.Get<DT_X>();
        LocalTensor<DT_X> oddLocal0 = oddRowBuf_.Get<DT_X>();
        LocalTensor<DT_X> rowLocal0 = packedRowBuf_.Get<DT_X>();
        LocalTensor<DT_X> evenLocal1 = evenRowBuf2_.Get<DT_X>();
        LocalTensor<DT_X> oddLocal1 = oddRowBuf2_.Get<DT_X>();
        LocalTensor<DT_X> rowLocal1 = packedRowBuf2_.Get<DT_X>();
        const uint32_t rowElements = outputRowElements_;
        const uint32_t endRow = startPoint_ + pointCount_;
        if (startPoint_ >= endRow)
        {
            return;
        }

        uint32_t currentRow = startPoint_;
        uint32_t currentRows = GetRowPackedTileRows(currentRow, endRow);
        BuildRowPackedCropTile(currentRow, currentRows, rowLocal0, evenLocal0,
                               oddLocal0);
        uint32_t currentBuffer = 0;
        uint32_t nextRow = currentRow + currentRows;

        while (nextRow < endRow)
        {
            const uint32_t outputOffset = currentRow * rowElements;
            if (currentBuffer == 0)
            {
                CopyLocalToGm(outputOffset, rowLocal0,
                              currentRows * rowElements);
            }
            else
            {
                CopyLocalToGm(outputOffset, rowLocal1,
                              currentRows * rowElements);
            }

            const uint32_t nextRows = GetRowPackedTileRows(nextRow, endRow);
            if (currentBuffer == 0)
            {
                BuildRowPackedCropTile(nextRow, nextRows, rowLocal1,
                                       evenLocal1, oddLocal1);
                currentBuffer = 1;
            }
            else
            {
                BuildRowPackedCropTile(nextRow, nextRows, rowLocal0,
                                       evenLocal0, oddLocal0);
                currentBuffer = 0;
            }
            currentRow = nextRow;
            currentRows = nextRows;
            nextRow += nextRows;
        }

        const uint32_t outputOffset = currentRow * rowElements;
        if (currentBuffer == 0)
        {
            CopyLocalToGm(outputOffset, rowLocal0, currentRows * rowElements);
        }
        else
        {
            CopyLocalToGm(outputOffset, rowLocal1, currentRows * rowElements);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessBlockSize2RowPackedNoCrop()
    {
        LocalTensor<DT_X> evenLocal0 = evenRowBuf_.Get<DT_X>();
        LocalTensor<DT_X> oddLocal0 = oddRowBuf_.Get<DT_X>();
        LocalTensor<DT_X> rowLocal0 = packedRowBuf_.Get<DT_X>();
        LocalTensor<DT_X> evenLocal1 = evenRowBuf2_.Get<DT_X>();
        LocalTensor<DT_X> oddLocal1 = oddRowBuf2_.Get<DT_X>();
        LocalTensor<DT_X> rowLocal1 = packedRowBuf2_.Get<DT_X>();
        const uint32_t rowElements = outputRowElements_;
        const uint32_t endRow = startPoint_ + pointCount_;
        if (startPoint_ >= endRow)
        {
            return;
        }

        uint32_t currentRow = startPoint_;
        uint32_t currentRows = GetRowPackedTileRows(currentRow, endRow);
        BuildRowPackedNoCropTile(currentRow, currentRows, rowLocal0,
                                 evenLocal0, oddLocal0);
        uint32_t currentBuffer = 0;
        uint32_t nextRow = currentRow + currentRows;

        while (nextRow < endRow)
        {
            const uint32_t outputOffset = currentRow * rowElements;
            if (currentBuffer == 0)
            {
                CopyLocalToGm(outputOffset, rowLocal0,
                              currentRows * rowElements);
            }
            else
            {
                CopyLocalToGm(outputOffset, rowLocal1,
                              currentRows * rowElements);
            }

            const uint32_t nextRows = GetRowPackedTileRows(nextRow, endRow);
            if (currentBuffer == 0)
            {
                BuildRowPackedNoCropTile(nextRow, nextRows, rowLocal1,
                                         evenLocal1, oddLocal1);
                currentBuffer = 1;
            }
            else
            {
                BuildRowPackedNoCropTile(nextRow, nextRows, rowLocal0,
                                         evenLocal0, oddLocal0);
                currentBuffer = 0;
            }
            currentRow = nextRow;
            currentRows = nextRows;
            nextRow += nextRows;
        }

        const uint32_t outputOffset = currentRow * rowElements;
        if (currentBuffer == 0)
        {
            CopyLocalToGm(outputOffset, rowLocal0, currentRows * rowElements);
        }
        else
        {
            CopyLocalToGm(outputOffset, rowLocal1, currentRows * rowElements);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline uint32_t GetRowPackedTileRows(
        uint32_t outputRow, uint32_t endRow) const
    {
        const uint32_t remainingRows = endRow - outputRow;
        return remainingRows < rowPackedTileRows_ ? remainingRows
                                                  : rowPackedTileRows_;
    }

    __aicore__ inline void BuildRowPackedCropTile(
        uint32_t outputRow, uint32_t tileRows, LocalTensor<DT_X> rowLocal,
        LocalTensor<DT_X> evenLocal, LocalTensor<DT_X> oddLocal)
    {
        const uint32_t rowElements = outputRowElements_;
        const uint32_t halfRowElements = inputRowElements_;
        const uint32_t halfRowStride = AlignUpBlockElements(halfRowElements);
        for (uint32_t row = 0; row < tileRows; ++row)
        {
            const uint32_t currentRow = outputRow + row;
            const uint32_t outputBatch = currentRow / outHeight_;
            const uint32_t outputH = currentRow - outputBatch * outHeight_;
            const uint32_t uncroppedH = outputH + cropTop_;
            const uint32_t inputH = uncroppedH >> 1;
            const uint32_t blockH = uncroppedH & 1;
            CopyRowPackedStream(evenLocal[row * halfRowStride], outputBatch,
                                inputH, blockH, 0);
            CopyRowPackedStream(oddLocal[row * halfRowStride], outputBatch,
                                inputH, blockH, 1);
        }
        PipeBarrier<PIPE_ALL>();

        for (uint32_t row = 0; row < tileRows; ++row)
        {
            const uint32_t inputRowOffset = row * halfRowStride;
            const uint32_t outputRowOffset = row * rowElements;
            BuildRowPackedStream(rowLocal[outputRowOffset],
                                 evenLocal[inputRowOffset], 0);
            BuildRowPackedStream(rowLocal[outputRowOffset],
                                 oddLocal[inputRowOffset], 1);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void BuildRowPackedNoCropTile(
        uint32_t outputRow, uint32_t tileRows, LocalTensor<DT_X> rowLocal,
        LocalTensor<DT_X> evenLocal, LocalTensor<DT_X> oddLocal)
    {
        const uint32_t rowElements = outputRowElements_;
        const uint32_t halfRowElements = inputRowElements_;
        const uint32_t halfRowStride = AlignUpBlockElements(halfRowElements);
        for (uint32_t row = 0; row < tileRows; ++row)
        {
            const uint32_t currentRow = outputRow + row;
            const uint32_t outputBatch = currentRow / outHeight_;
            const uint32_t outputH = currentRow - outputBatch * outHeight_;
            const uint32_t inputH = outputH >> 1;
            const uint32_t blockH = outputH & 1;
            const uint32_t inputBatchEven =
                (blockH << 1) * outBatch_ + outputBatch;
            const uint32_t inputBatchOdd = inputBatchEven + outBatch_;
            const uint32_t inputEvenBase =
                (inputBatchEven * height_ + inputH) * inputRowElements_;
            const uint32_t inputOddBase =
                (inputBatchOdd * height_ + inputH) * inputRowElements_;
            CopyGmToLocal(evenLocal[row * halfRowStride], inputEvenBase,
                          halfRowElements);
            CopyGmToLocal(oddLocal[row * halfRowStride], inputOddBase,
                          halfRowElements);
        }
        PipeBarrier<PIPE_ALL>();

        for (uint32_t row = 0; row < tileRows; ++row)
        {
            const uint32_t inputRowOffset = row * halfRowStride;
            const uint32_t outputRowOffset = row * rowElements;
            CopyLocalInterleavedColumns(rowLocal[outputRowOffset],
                                        evenLocal[inputRowOffset],
                                        oddLocal[inputRowOffset], width_);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline uint32_t GetFirstOutputWForBlock(uint32_t blockW) const
    {
        return (blockW + 2 - (cropLeft_ & 1)) & 1;
    }

    __aicore__ inline uint32_t GetRowPackedStreamCols(uint32_t blockW) const
    {
        const uint32_t firstOutputW = GetFirstOutputWForBlock(blockW);
        if (firstOutputW >= outWidth_)
        {
            return 0;
        }
        return ((outWidth_ - 1 - firstOutputW) >> 1) + 1;
    }

    __aicore__ inline void CopyRowPackedStream(LocalTensor<DT_X> local,
                                               uint32_t outputBatch,
                                               uint32_t inputH,
                                               uint32_t blockH,
                                               uint32_t blockW)
    {
        const uint32_t streamCols = GetRowPackedStreamCols(blockW);
        if (streamCols == 0)
        {
            return;
        }
        const uint32_t firstOutputW = GetFirstOutputWForBlock(blockW);
        const uint32_t inputW = (firstOutputW + cropLeft_) >> 1;
        const uint32_t inputBatch =
            ((blockH << 1) + blockW) * outBatch_ + outputBatch;
        const uint32_t inputBase =
            ((inputBatch * height_ + inputH) * width_ + inputW) * depth_;
        CopyGmToLocal(local, inputBase, streamCols * depth_);
    }

    __aicore__ inline void BuildRowPackedStream(LocalTensor<DT_X> outputLocal,
                                                LocalTensor<DT_X> streamLocal,
                                                uint32_t blockW)
    {
        const uint32_t streamCols = GetRowPackedStreamCols(blockW);
        const uint32_t firstOutputW = GetFirstOutputWForBlock(blockW);
        if (CopyLocalStridedColumns(outputLocal[firstOutputW * depth_],
                                    streamLocal, streamCols))
        {
            return;
        }
        for (uint32_t col = 0; col < streamCols; ++col)
        {
            const uint32_t srcOffset = col * depth_;
            const uint32_t dstOffset =
                (firstOutputW + (col << 1)) * depth_;
            CopyLocalSegment(outputLocal[dstOffset], streamLocal[srcOffset],
                             depth_);
        }
    }

    __aicore__ inline bool UsePaddedWideRowPackedLayout() const
    {
        const uint32_t depthBytes =
            depthBytes_ != 0 ? depthBytes_ : depth_ * sizeof(DT_X);
        return outWidth_ > 64 && (depthBytes & 31U) != 0;
    }

    __aicore__ inline void ProcessBlockSize2RowPackedPadded()
    {
        LocalTensor<DT_X> evenLocal0 = evenRowBuf_.Get<DT_X>();
        LocalTensor<DT_X> oddLocal0 = oddRowBuf_.Get<DT_X>();
        LocalTensor<DT_X> rowLocal0 = packedRowBuf_.Get<DT_X>();
        LocalTensor<DT_X> evenLocal1 = evenRowBuf2_.Get<DT_X>();
        LocalTensor<DT_X> oddLocal1 = oddRowBuf2_.Get<DT_X>();
        LocalTensor<DT_X> rowLocal1 = packedRowBuf2_.Get<DT_X>();
        const uint32_t endRow = startPoint_ + pointCount_;
        if (startPoint_ >= endRow)
        {
            return;
        }

        uint32_t currentRow = startPoint_;
        uint32_t currentRows = GetRowPackedTileRows(currentRow, endRow);
        BuildRowPackedCropTilePadded(currentRow, currentRows, rowLocal0,
                                     evenLocal0, oddLocal0);
        uint32_t currentBuffer = 0;
        uint32_t nextRow = currentRow + currentRows;

        while (nextRow < endRow)
        {
            if (currentBuffer == 0)
            {
                CopyPaddedRowPackedToGm(currentRow, currentRows, rowLocal0);
            }
            else
            {
                CopyPaddedRowPackedToGm(currentRow, currentRows, rowLocal1);
            }

            const uint32_t nextRows = GetRowPackedTileRows(nextRow, endRow);
            if (currentBuffer == 0)
            {
                BuildRowPackedCropTilePadded(nextRow, nextRows, rowLocal1,
                                             evenLocal1, oddLocal1);
                currentBuffer = 1;
            }
            else
            {
                BuildRowPackedCropTilePadded(nextRow, nextRows, rowLocal0,
                                             evenLocal0, oddLocal0);
                currentBuffer = 0;
            }
            currentRow = nextRow;
            currentRows = nextRows;
            nextRow += nextRows;
        }

        if (currentBuffer == 0)
        {
            CopyPaddedRowPackedToGm(currentRow, currentRows, rowLocal0);
        }
        else
        {
            CopyPaddedRowPackedToGm(currentRow, currentRows, rowLocal1);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void BuildRowPackedCropTilePadded(
        uint32_t outputRow, uint32_t tileRows, LocalTensor<DT_X> rowLocal,
        LocalTensor<DT_X> evenLocal, LocalTensor<DT_X> oddLocal)
    {
        const uint32_t depthStride = AlignUpBlockElements(depth_);
        const uint32_t halfRowStride = width_ * depthStride;
        const uint32_t outputRowStride = outWidth_ * depthStride;
        for (uint32_t row = 0; row < tileRows; ++row)
        {
            const uint32_t currentRow = outputRow + row;
            const uint32_t outputBatch = currentRow / outHeight_;
            const uint32_t outputH = currentRow - outputBatch * outHeight_;
            const uint32_t uncroppedH = outputH + cropTop_;
            const uint32_t inputH = uncroppedH >> 1;
            const uint32_t blockH = uncroppedH & 1;
            CopyRowPackedStreamPadded(evenLocal[row * halfRowStride],
                                      outputBatch, inputH, blockH, 0);
            CopyRowPackedStreamPadded(oddLocal[row * halfRowStride],
                                      outputBatch, inputH, blockH, 1);
        }
        PipeBarrier<PIPE_ALL>();

        for (uint32_t row = 0; row < tileRows; ++row)
        {
            const uint32_t inputRowOffset = row * halfRowStride;
            const uint32_t outputRowOffset = row * outputRowStride;
            BuildRowPackedStreamPadded(rowLocal[outputRowOffset],
                                       evenLocal[inputRowOffset], 0);
            BuildRowPackedStreamPadded(rowLocal[outputRowOffset],
                                       oddLocal[inputRowOffset], 1);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyRowPackedStreamPadded(
        LocalTensor<DT_X> local, uint32_t outputBatch, uint32_t inputH,
        uint32_t blockH, uint32_t blockW)
    {
        const uint32_t streamCols = GetRowPackedStreamCols(blockW);
        if (streamCols == 0)
        {
            return;
        }
        const uint32_t firstOutputW = GetFirstOutputWForBlock(blockW);
        const uint32_t inputW = (firstOutputW + cropLeft_) >> 1;
        const uint32_t inputBatch =
            ((blockH << 1) + blockW) * outBatch_ + outputBatch;
        const uint32_t inputBase =
            (inputBatch * height_ + inputH) * inputRowElements_ +
            inputW * depth_;
        DataCopyExtParams params{
            static_cast<uint16_t>(streamCols),
            depthBytes_, 0, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
        DataCopyPad(local, xGm_[inputBase], params, padParams);
    }

    __aicore__ inline void BuildRowPackedStreamPadded(
        LocalTensor<DT_X> outputLocal, LocalTensor<DT_X> streamLocal,
        uint32_t blockW)
    {
        const uint32_t streamCols = GetRowPackedStreamCols(blockW);
        const uint32_t firstOutputW = GetFirstOutputWForBlock(blockW);
        const uint32_t depthStride = AlignUpBlockElements(depth_);
        for (uint32_t col = 0; col < streamCols; ++col)
        {
            const uint32_t srcOffset = col * depthStride;
            const uint32_t dstOffset =
                (firstOutputW + (col << 1)) * depthStride;
            CopyLocalSegment(outputLocal[dstOffset], streamLocal[srcOffset],
                             depth_);
        }
    }

    __aicore__ inline void CopyPaddedRowPackedToGm(
        uint32_t outputRow, uint32_t tileRows, LocalTensor<DT_X> rowLocal)
    {
        const uint32_t depthStride = AlignUpBlockElements(depth_);
        const uint32_t outputRowStride = outWidth_ * depthStride;
        const uint32_t depthBytes =
            depthBytes_ != 0 ? depthBytes_ : depth_ * sizeof(DT_X);
        DataCopyExtParams params{
            static_cast<uint16_t>(outWidth_), depthBytes, 0, 0, 0};
        for (uint32_t row = 0; row < tileRows; ++row)
        {
            const uint32_t outputOffset =
                (outputRow + row) * outputRowElements_;
            DataCopyPad(yGm_[outputOffset],
                        rowLocal[row * outputRowStride], params);
        }
    }

    __aicore__ inline void ProcessBlockSize2MicroTile()
    {
        LocalTensor<DT_X> sourceLocal = evenRowBuf_.Get<DT_X>();
        LocalTensor<uint32_t> offsetLocal = oddRowBuf_.Get<uint32_t>();
        LocalTensor<DT_X> tileLocal = packedRowBuf_.Get<DT_X>();
        const uint32_t tileCols =
            rowPackedTileCols_ == 0 ? 16 : rowPackedTileCols_;
        const uint32_t streamCols = (tileCols + 1) >> 1;
        const uint32_t streamElements =
            AlignUpBlockElements(streamCols * depth_);
        const uint32_t sourceElements = streamElements << 1;
        const uint32_t outputElements =
            AlignUpBlockElements(tileCols * depth_);
        const bool useGatherMicroTile =
            tileCols == 64 && depth_ == 65 && sizeof(DT_X) == 2;
        if (useGatherMicroTile)
        {
            InitMicroTileGatherOffsets(offsetLocal, tileCols);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            ProcessBlockSize2MicroTilePingPong(sourceLocal, offsetLocal,
                                               tileLocal, tileCols,
                                               streamElements, sourceElements,
                                               outputElements);
            return;
        }
        const uint32_t outputRows = outBatch_ * outHeight_;
        const uint32_t endRow = startPoint_ + pointCount_;
        for (uint32_t outputRow = startPoint_; outputRow < endRow;
             ++outputRow)
        {
            if (outputRow >= outputRows)
            {
                return;
            }
            uint32_t outputCol = 0;
            while (outputCol < outWidth_)
            {
                const uint32_t currentTileCols =
                    (outputCol + tileCols <= outWidth_) ? tileCols
                                                        : (outWidth_ - outputCol);
                LocalTensor<DT_X> evenLocal = sourceLocal;
                LocalTensor<DT_X> oddLocal = sourceLocal[streamElements];
                BuildMicroTile(outputRow, outputCol, currentTileCols,
                               evenLocal, oddLocal, tileLocal);
                const uint32_t outputOffset =
                    (outputRow * outWidth_ + outputCol) * depth_;
                CopyLocalToGm(outputOffset, tileLocal,
                              currentTileCols * depth_);
                PipeBarrier<PIPE_ALL>();
                outputCol += currentTileCols;
            }
        }
    }

    __aicore__ inline void ProcessBlockSize2MicroTilePingPong(
        LocalTensor<DT_X> sourceBase, LocalTensor<uint32_t> offsetLocal,
        LocalTensor<DT_X> tileBase, uint32_t tileCols,
        uint32_t streamElements, uint32_t sourceElements,
        uint32_t outputElements)
    {
        uint32_t tileSerial = 0;
        const uint32_t outputRows = outBatch_ * outHeight_;
        const uint32_t endRow = startPoint_ + pointCount_;
        for (uint32_t outputRow = startPoint_; outputRow < endRow;
             ++outputRow)
        {
            if (outputRow >= outputRows)
            {
                break;
            }
            uint32_t outputCol = 0;
            while (outputCol < outWidth_)
            {
                const uint32_t currentTileCols =
                    (outputCol + tileCols <= outWidth_) ? tileCols
                                                        : (outWidth_ - outputCol);
                const uint32_t bufId = tileSerial & 1U;
                if (tileSerial >= BUFFER_NUM)
                {
                    WaitFlag<HardEvent::MTE3_MTE2>(bufId);
                }
                LocalTensor<DT_X> sourceLocal =
                    sourceBase[bufId * sourceElements];
                LocalTensor<DT_X> tileLocal =
                    tileBase[bufId * outputElements];
                BuildMicroTileGather(outputRow, outputCol, currentTileCols,
                                     streamElements, sourceLocal, offsetLocal,
                                     tileLocal, bufId);
                const uint32_t outputOffset =
                    (outputRow * outWidth_ + outputCol) * depth_;
                CopyLocalToGm(outputOffset, tileLocal,
                              currentTileCols * depth_);
                SetFlag<HardEvent::MTE3_MTE2>(bufId);
                ++tileSerial;
                outputCol += currentTileCols;
            }
        }
        if (tileSerial != 0)
        {
            WaitFlag<HardEvent::MTE3_MTE2>((tileSerial - 1U) & 1U);
        }
        if (tileSerial >= BUFFER_NUM)
        {
            WaitFlag<HardEvent::MTE3_MTE2>((tileSerial - 2U) & 1U);
        }
    }

    __aicore__ inline void InitMicroTileGatherOffsets(
        LocalTensor<uint32_t> offsetLocal, uint32_t tileCols)
    {
        DataCopy(offsetLocal, microTileOffsetGm_, tileCols * depth_);
    }

    __aicore__ inline void BuildMicroTileGather(
        uint32_t outputRow, uint32_t outputCol, uint32_t tileCols,
        uint32_t streamElements, LocalTensor<DT_X> sourceLocal,
        LocalTensor<uint32_t> offsetLocal, LocalTensor<DT_X> tileLocal,
        uint32_t eventId)
    {
        const uint32_t outputBatch = outputRow / outHeight_;
        const uint32_t outputH = outputRow - outputBatch * outHeight_;
        const uint32_t uncroppedH = outputH + cropTop_;
        const uint32_t inputH = uncroppedH >> 1;
        const uint32_t blockH = uncroppedH & 1;
        const uint32_t endCol = outputCol + tileCols;
        const uint32_t firstW0 = FirstOutputWInRange(outputCol, endCol, 0);
        const uint32_t firstW1 = FirstOutputWInRange(outputCol, endCol, 1);
        const uint32_t cols0 = OutputStreamColsInRange(firstW0, endCol);
        const uint32_t cols1 = OutputStreamColsInRange(firstW1, endCol);
        if (cols0 != 0)
        {
            CopyMicroTileInputStream(sourceLocal, outputBatch, inputH, blockH,
                                     0, firstW0, cols0);
        }
        if (cols1 != 0)
        {
            CopyMicroTileInputStream(sourceLocal[streamElements], outputBatch,
                                     inputH, blockH, 1, firstW1, cols1);
        }
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        sourceLocal.SetSize(streamElements << 1);
        offsetLocal.SetSize(tileCols * depth_);
        tileLocal.SetSize(tileCols * depth_);
        Gather(tileLocal, sourceLocal, offsetLocal, 0U, tileCols * depth_);
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline uint32_t FirstOutputWInRange(uint32_t outputCol,
                                                   uint32_t endCol,
                                                   uint32_t blockW) const
    {
        const uint32_t parity = (outputCol + cropLeft_) & 1;
        const uint32_t delta = (blockW + 2 - parity) & 1;
        const uint32_t firstW = outputCol + delta;
        return firstW < endCol ? firstW : endCol;
    }

    __aicore__ inline uint32_t OutputStreamColsInRange(uint32_t firstW,
                                                       uint32_t endCol) const
    {
        if (firstW >= endCol)
        {
            return 0;
        }
        return ((endCol - 1 - firstW) >> 1) + 1;
    }

    __aicore__ inline void BuildMicroTile(
        uint32_t outputRow, uint32_t outputCol, uint32_t tileCols,
        LocalTensor<DT_X> evenLocal, LocalTensor<DT_X> oddLocal,
        LocalTensor<DT_X> tileLocal)
    {
        const uint32_t outputBatch = outputRow / outHeight_;
        const uint32_t outputH = outputRow - outputBatch * outHeight_;
        const uint32_t uncroppedH = outputH + cropTop_;
        const uint32_t inputH = uncroppedH >> 1;
        const uint32_t blockH = uncroppedH & 1;
        const uint32_t endCol = outputCol + tileCols;
        const uint32_t firstW0 = FirstOutputWInRange(outputCol, endCol, 0);
        const uint32_t firstW1 = FirstOutputWInRange(outputCol, endCol, 1);
        const uint32_t cols0 = OutputStreamColsInRange(firstW0, endCol);
        const uint32_t cols1 = OutputStreamColsInRange(firstW1, endCol);
        if (cols0 != 0)
        {
            CopyMicroTileInputStream(evenLocal, outputBatch, inputH, blockH,
                                     0, firstW0, cols0);
        }
        if (cols1 != 0)
        {
            CopyMicroTileInputStream(oddLocal, outputBatch, inputH, blockH,
                                     1, firstW1, cols1);
        }
        PipeBarrier<PIPE_ALL>();

        for (uint32_t localCol = 0; localCol < tileCols; ++localCol)
        {
            const uint32_t outputW = outputCol + localCol;
            const uint32_t blockW = (outputW + cropLeft_) & 1;
            const uint32_t firstW = blockW == 0 ? firstW0 : firstW1;
            const uint32_t streamIndex = (outputW - firstW) >> 1;
            if (blockW == 0)
            {
                CopyLocalSegment(tileLocal[localCol * depth_],
                                 evenLocal[streamIndex * depth_], depth_);
            }
            else
            {
                CopyLocalSegment(tileLocal[localCol * depth_],
                                 oddLocal[streamIndex * depth_], depth_);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyMicroTileInputStream(
        LocalTensor<DT_X> local, uint32_t outputBatch, uint32_t inputH,
        uint32_t blockH, uint32_t blockW, uint32_t firstOutputW,
        uint32_t cols)
    {
        const uint32_t inputW = (firstOutputW + cropLeft_) >> 1;
        const uint32_t inputBatch =
            ((blockH << 1) + blockW) * outBatch_ + outputBatch;
        const uint32_t inputBase =
            ((inputBatch * height_ + inputH) * width_ + inputW) * depth_;
        CopyGmToLocal(local, inputBase, cols * depth_);
    }

    __aicore__ inline void CopyOneOutputColumn(uint32_t outputRow,
                                               uint32_t outputW,
                                               LocalTensor<DT_X> scratch)
    {
        const uint32_t outputBatch = outputRow / outHeight_;
        const uint32_t outputH = outputRow - outputBatch * outHeight_;
        const uint32_t uncroppedH = outputH + cropTop_;
        const uint32_t inputH = uncroppedH >> 1;
        const uint32_t blockH = uncroppedH & 1;
        const uint32_t uncroppedW = outputW + cropLeft_;
        const uint32_t inputW = uncroppedW >> 1;
        const uint32_t blockW = uncroppedW & 1;
        const uint32_t inputBatch =
            ((blockH << 1) + blockW) * outBatch_ + outputBatch;
        const uint32_t inputOffset =
            ((inputBatch * height_ + inputH) * width_ + inputW) * depth_;
        const uint32_t outputOffset =
            (outputRow * outWidth_ + outputW) * depth_;
        CopyGmToLocal(scratch, inputOffset, depth_);
        PipeBarrier<PIPE_ALL>();
        CopyLocalToGm(outputOffset, scratch, depth_);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessBlockSize2SmallNoCrop()
    {
        LocalTensor<DT_X> evenLocal = evenRowBuf_.Get<DT_X>();
        LocalTensor<DT_X> oddLocal = oddRowBuf_.Get<DT_X>();
        LocalTensor<DT_X> rowLocal = packedRowBuf_.Get<DT_X>();
        const uint32_t outputRows = outBatch_ * outHeight_;
        const uint32_t halfRowElements = inputRowElements_;
        const uint32_t rowElements = outputRowElements_;
        const uint32_t endRow = startPoint_ + pointCount_;
        for (uint32_t outputRow = startPoint_; outputRow < endRow;
             ++outputRow)
        {
            if (outputRow >= outputRows)
            {
                return;
            }
            const uint32_t outputBatch = outputRow / outHeight_;
            const uint32_t outputH = outputRow - outputBatch * outHeight_;
            const uint32_t inputH = outputH >> 1;
            const uint32_t blockH = outputH & 1;
            const uint32_t inputBatchEven =
                (blockH << 1) * outBatch_ + outputBatch;
            const uint32_t inputBatchOdd = inputBatchEven + outBatch_;
            const uint32_t inputEvenBase =
                (inputBatchEven * height_ + inputH) * inputRowElements_;
            const uint32_t inputOddBase =
                (inputBatchOdd * height_ + inputH) * inputRowElements_;
            CopyGmToLocal(evenLocal, inputEvenBase, halfRowElements);
            CopyGmToLocal(oddLocal, inputOddBase, halfRowElements);
            PipeBarrier<PIPE_ALL>();
            CopyLocalInterleavedColumns(rowLocal, evenLocal, oddLocal, width_);
            PipeBarrier<PIPE_ALL>();
            CopyLocalToGm(outputRow * rowElements, rowLocal, rowElements);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ProcessBlockSize2ColumnPacked()
    {
        LocalTensor<DT_X> evenLocal = evenRowBuf_.Get<DT_X>();
        LocalTensor<DT_X> oddLocal = oddRowBuf_.Get<DT_X>();
        LocalTensor<DT_X> outputLocal0 = packedRowBuf_.Get<DT_X>();
        LocalTensor<DT_X> outputLocal1 = packedRowBuf2_.Get<DT_X>();
        const uint32_t tileRows = rowPackedTileRows_ == 0 ? 1 : rowPackedTileRows_;
        const uint32_t tileCols = rowPackedTileCols_;
        const uint32_t colTiles = (outWidth_ + tileCols - 1) / tileCols;
        const uint32_t heightTiles = (height_ + tileRows - 1) / tileRows;
        const uint32_t rowTiles = outBatch_ * heightTiles;
        const uint32_t endTile = startPoint_ + pointCount_;

        for (uint32_t tile = startPoint_; tile < endTile; ++tile)
        {
            const uint32_t rowTile = tile / colTiles;
            if (rowTile >= rowTiles)
            {
                return;
            }
            const uint32_t outputBatch = rowTile / heightTiles;
            const uint32_t heightTile = rowTile - outputBatch * heightTiles;
            const uint32_t inputH = heightTile * tileRows;
            const uint32_t currentTileRows =
                height_ - inputH < tileRows ? height_ - inputH : tileRows;
            const uint32_t colTile = tile - rowTile * colTiles;
            const uint32_t outputCol = colTile * tileCols;
            const uint32_t remainingCols = outWidth_ - outputCol;
            const uint32_t currentOutputCols =
                remainingCols < tileCols ? remainingCols : tileCols;
            const uint32_t endOutputCol = outputCol + currentOutputCols;

            BuildColumnPackedOutputRows(outputLocal0, evenLocal, oddLocal,
                                        outputBatch, inputH, currentTileRows, 0,
                                        outputCol, endOutputCol,
                                        currentOutputCols);
            BuildColumnPackedOutputRows(outputLocal1, evenLocal, oddLocal,
                                        outputBatch, inputH, currentTileRows, 1,
                                        outputCol, endOutputCol,
                                        currentOutputCols);
        }
    }

    __aicore__ inline void BuildColumnPackedOutputRows(
        LocalTensor<DT_X> outputLocal, LocalTensor<DT_X> evenLocal,
        LocalTensor<DT_X> oddLocal, uint32_t outputBatch, uint32_t inputH,
        uint32_t tileRows, uint32_t blockH, uint32_t outputCol,
        uint32_t endOutputCol, uint32_t currentOutputCols)
    {
        const uint32_t inputRowStride =
            AlignUpBlockElements(((endOutputCol - outputCol + 1) >> 1) *
                                 depth_);
        const uint32_t outputRowStride =
            AlignUpBlockElements(currentOutputCols * depth_);
        const uint32_t firstOutputW0 =
            outputCol + ((2 - ((outputCol + cropLeft_) & 1)) & 1);
        const uint32_t firstOutputW1 =
            outputCol + ((3 - ((outputCol + cropLeft_) & 1)) & 1);
        uint32_t streamCols0 = 0;
        uint32_t streamCols1 = 0;
        if (firstOutputW0 < endOutputCol)
        {
            streamCols0 = ((endOutputCol - 1 - firstOutputW0) >> 1) + 1;
            const uint32_t inputW0 = (firstOutputW0 + cropLeft_) >> 1;
            const uint32_t inputBatch0 = (blockH << 1) * outBatch_ + outputBatch;
            const uint32_t inputBase0 =
                ((inputBatch0 * height_ + inputH) * width_ + inputW0) *
                depth_;
            CopyGmRowsToLocal(evenLocal, inputBase0, streamCols0 * depth_,
                              tileRows, inputRowElements_, inputRowStride);
        }
        if (firstOutputW1 < endOutputCol)
        {
            streamCols1 = ((endOutputCol - 1 - firstOutputW1) >> 1) + 1;
            const uint32_t inputW1 = (firstOutputW1 + cropLeft_) >> 1;
            const uint32_t inputBatch1 =
                ((blockH << 1) + 1) * outBatch_ + outputBatch;
            const uint32_t inputBase1 =
                ((inputBatch1 * height_ + inputH) * width_ + inputW1) *
                depth_;
            CopyGmRowsToLocal(oddLocal, inputBase1, streamCols1 * depth_,
                              tileRows, inputRowElements_, inputRowStride);
        }
        PipeBarrier<PIPE_ALL>();

        for (uint32_t row = 0; row < tileRows; ++row)
        {
            const uint32_t uncroppedH = ((inputH + row) << 1) + blockH;
            if (uncroppedH < cropTop_)
            {
                continue;
            }
            const uint32_t outputH = uncroppedH - cropTop_;
            if (outputH >= outHeight_)
            {
                continue;
            }
            if (streamCols0 != 0)
            {
                CopyColumnPackedStreamFromLocal(
                    outputLocal[row * outputRowStride],
                    evenLocal[row * inputRowStride],
                    firstOutputW0 - outputCol, streamCols0);
            }
            if (streamCols1 != 0)
            {
                CopyColumnPackedStreamFromLocal(
                    outputLocal[row * outputRowStride],
                    oddLocal[row * inputRowStride],
                    firstOutputW1 - outputCol, streamCols1);
            }
            PipeBarrier<PIPE_ALL>();

            const uint32_t outputBase =
                ((outputBatch * outHeight_ + outputH) * outWidth_ +
                 outputCol) *
                depth_;
            CopyLocalToGm(outputBase, outputLocal[row * outputRowStride],
                          currentOutputCols * depth_);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void CopyGmRowsToLocal(
        LocalTensor<DT_X> local, uint32_t inputOffset, uint32_t rowElements,
        uint32_t rows, uint32_t srcRowStrideElements,
        uint32_t dstRowStrideElements)
    {
        if (rows == 1)
        {
            CopyGmToLocal(local, inputOffset, rowElements);
            return;
        }
        const uint32_t copyBytes = rowElements * sizeof(DT_X);
        const uint32_t srcStrideBytes =
            (srcRowStrideElements - rowElements) * sizeof(DT_X);
        const uint32_t dstStrideBytes =
            (dstRowStrideElements - rowElements) * sizeof(DT_X);
        LocalTensor<DT_X> dst = local;
        if (copyBytes % 32 == 0 && CanUseAlignedCopy(inputOffset, rowElements) &&
            srcStrideBytes % 32 == 0 && dstStrideBytes % 32 == 0)
        {
            const uint16_t blockLen =
                static_cast<uint16_t>(copyBytes / 32);
            const uint16_t srcStride =
                static_cast<uint16_t>(srcStrideBytes / 32);
            const uint16_t dstStride =
                static_cast<uint16_t>(dstStrideBytes / 32);
            DataCopyParams params{
                static_cast<uint16_t>(rows), blockLen, srcStride, dstStride};
            DataCopy(dst, xGm_[inputOffset], params);
            return;
        }
        DataCopyExtParams params{
            static_cast<uint16_t>(rows), copyBytes, srcStrideBytes,
            dstStrideBytes, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
        DataCopyPad(dst, xGm_[inputOffset], params, padParams);
    }

    __aicore__ inline void BuildColumnPackedOutputRow(
        LocalTensor<DT_X> outputLocal, LocalTensor<DT_X> evenLocal,
        LocalTensor<DT_X> oddLocal, uint32_t outputBatch, uint32_t inputH,
        uint32_t blockH, uint32_t outputCol, uint32_t endOutputCol,
        uint32_t currentOutputCols)
    {
        const uint32_t uncroppedH = (inputH << 1) + blockH;
        if (uncroppedH < cropTop_)
        {
            return;
        }
        const uint32_t outputH = uncroppedH - cropTop_;
        if (outputH >= outHeight_)
        {
            return;
        }

        const uint32_t firstOutputW0 =
            outputCol + ((2 - ((outputCol + cropLeft_) & 1)) & 1);
        const uint32_t firstOutputW1 =
            outputCol + ((3 - ((outputCol + cropLeft_) & 1)) & 1);
        uint32_t streamCols0 = 0;
        uint32_t streamCols1 = 0;
        if (firstOutputW0 < endOutputCol)
        {
            streamCols0 = ((endOutputCol - 1 - firstOutputW0) >> 1) + 1;
            const uint32_t inputW0 = (firstOutputW0 + cropLeft_) >> 1;
            const uint32_t inputBatch0 = (blockH << 1) * outBatch_ + outputBatch;
            const uint32_t inputBase0 =
                ((inputBatch0 * height_ + inputH) * width_ + inputW0) *
                depth_;
            CopyGmToLocal(evenLocal, inputBase0, streamCols0 * depth_);
        }
        if (firstOutputW1 < endOutputCol)
        {
            streamCols1 = ((endOutputCol - 1 - firstOutputW1) >> 1) + 1;
            const uint32_t inputW1 = (firstOutputW1 + cropLeft_) >> 1;
            const uint32_t inputBatch1 =
                ((blockH << 1) + 1) * outBatch_ + outputBatch;
            const uint32_t inputBase1 =
                ((inputBatch1 * height_ + inputH) * width_ + inputW1) *
                depth_;
            CopyGmToLocal(oddLocal, inputBase1, streamCols1 * depth_);
        }
        PipeBarrier<PIPE_ALL>();

        if (streamCols0 != 0)
        {
            CopyColumnPackedStreamFromLocal(
                outputLocal, evenLocal, firstOutputW0 - outputCol,
                streamCols0);
        }
        if (streamCols1 != 0)
        {
            CopyColumnPackedStreamFromLocal(
                outputLocal, oddLocal, firstOutputW1 - outputCol,
                streamCols1);
        }
        PipeBarrier<PIPE_ALL>();

        const uint32_t outputBase =
            ((outputBatch * outHeight_ + outputH) * outWidth_ + outputCol) *
            depth_;
        CopyLocalToGm(outputBase, outputLocal, currentOutputCols * depth_);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyColumnPackedStreamFromLocal(
        LocalTensor<DT_X> outputLocal, LocalTensor<DT_X> streamLocal,
        uint32_t localColOffset, uint32_t streamCols)
    {
        if (CopyLocalStridedColumns(outputLocal[localColOffset * depth_],
                                    streamLocal, streamCols))
        {
            return;
        }
        for (uint32_t col = 0; col < streamCols; ++col)
        {
            const uint32_t srcOffset = col * depth_;
            const uint32_t dstOffset =
                (localColOffset + (col << 1)) * depth_;
            CopyLocalSegment(outputLocal[dstOffset], streamLocal[srcOffset],
                             depth_);
        }
    }

    __aicore__ inline void BuildColumnPackedStream(
        LocalTensor<DT_X> outputLocal, LocalTensor<DT_X> streamLocal,
        uint32_t outputBatch, uint32_t inputH, uint32_t blockH,
        uint32_t blockW, uint32_t outputCol, uint32_t endOutputCol)
    {
        const uint32_t firstOutputW =
            outputCol + ((blockW + 2 - ((outputCol + cropLeft_) & 1)) & 1);
        if (firstOutputW >= endOutputCol)
        {
            return;
        }
        const uint32_t streamCols =
            ((endOutputCol - 1 - firstOutputW) >> 1) + 1;
        const uint32_t inputW = (firstOutputW + cropLeft_) >> 1;
        const uint32_t inputBatch =
            ((blockH << 1) + blockW) * outBatch_ + outputBatch;
        const uint32_t inputBase =
            ((inputBatch * height_ + inputH) * width_ + inputW) * depth_;

        CopyGmToLocal(streamLocal, inputBase, streamCols * depth_);
        PipeBarrier<PIPE_ALL>();

        const uint32_t localColOffset = firstOutputW - outputCol;
        if (CopyLocalStridedColumns(outputLocal[localColOffset * depth_],
                                    streamLocal, streamCols))
        {
            PipeBarrier<PIPE_ALL>();
            return;
        }
        for (uint32_t col = 0; col < streamCols; ++col)
        {
            const uint32_t srcOffset = col * depth_;
            const uint32_t dstOffset =
                (localColOffset + (col << 1)) * depth_;
            CopyLocalSegment(outputLocal[dstOffset], streamLocal[srcOffset],
                             depth_);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessBlockSize2HeightPacked()
    {
        LocalTensor<DT_X> evenLocal0 = evenRowBuf_.Get<DT_X>();
        LocalTensor<DT_X> oddLocal0 = oddRowBuf_.Get<DT_X>();
        LocalTensor<DT_X> outputLocal0 = packedRowBuf_.Get<DT_X>();
        LocalTensor<DT_X> evenLocal1 = evenRowBuf2_.Get<DT_X>();
        LocalTensor<DT_X> oddLocal1 = oddRowBuf2_.Get<DT_X>();
        LocalTensor<DT_X> outputLocal1 = packedRowBuf2_.Get<DT_X>();
        const uint32_t inputRowElements = inputRowElements_;
        const uint32_t outputRowElements = outputRowElements_;
        const uint32_t endInputRow = startPoint_ + pointCount_;
        if (startPoint_ >= endInputRow)
        {
            return;
        }

        uint32_t currentInputRow = startPoint_;
        uint32_t currentTileRows =
            GetHeightPackedTileRows(currentInputRow, endInputRow);
        BuildHeightPackedTile(currentInputRow, currentTileRows, outputLocal0,
                              evenLocal0, oddLocal0);
        uint32_t currentBuffer = 0;
        uint32_t nextInputRow = currentInputRow + currentTileRows;

        while (nextInputRow < endInputRow)
        {
            const uint32_t currentOutputBase =
                GetHeightPackedOutputBase(currentInputRow);
            if (currentBuffer == 0)
            {
                CopyLocalToGm(currentOutputBase, outputLocal0,
                              (currentTileRows << 1) * outputRowElements);
            }
            else
            {
                CopyLocalToGm(currentOutputBase, outputLocal1,
                              (currentTileRows << 1) * outputRowElements);
            }

            const uint32_t nextTileRows =
                GetHeightPackedTileRows(nextInputRow, endInputRow);
            if (currentBuffer == 0)
            {
                BuildHeightPackedTile(nextInputRow, nextTileRows, outputLocal1,
                                      evenLocal1, oddLocal1);
                currentBuffer = 1;
            }
            else
            {
                BuildHeightPackedTile(nextInputRow, nextTileRows, outputLocal0,
                                      evenLocal0, oddLocal0);
                currentBuffer = 0;
            }
            currentInputRow = nextInputRow;
            currentTileRows = nextTileRows;
            nextInputRow += nextTileRows;
        }

        const uint32_t currentOutputBase =
            GetHeightPackedOutputBase(currentInputRow);
        if (currentBuffer == 0)
        {
            CopyLocalToGm(currentOutputBase, outputLocal0,
                          (currentTileRows << 1) * outputRowElements);
        }
        else
        {
            CopyLocalToGm(currentOutputBase, outputLocal1,
                          (currentTileRows << 1) * outputRowElements);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline uint32_t GetHeightPackedTileRows(
        uint32_t inputRow, uint32_t endInputRow) const
    {
        const uint32_t outputBatch = inputRow / height_;
        const uint32_t inputH = inputRow - outputBatch * height_;
        const uint32_t remainingRows = endInputRow - inputRow;
        const uint32_t rowsInBatch = height_ - inputH;
        uint32_t tileRows =
            remainingRows < rowPackedTileRows_ ? remainingRows
                                               : rowPackedTileRows_;
        return tileRows < rowsInBatch ? tileRows : rowsInBatch;
    }

    __aicore__ inline uint32_t GetHeightPackedOutputBase(
        uint32_t inputRow) const
    {
        const uint32_t outputBatch = inputRow / height_;
        const uint32_t inputH = inputRow - outputBatch * height_;
        return (outputBatch * outHeight_ + (inputH << 1)) *
               outputRowElements_;
    }

    __aicore__ inline void BuildHeightPackedTile(
        uint32_t inputRow, uint32_t tileRows, LocalTensor<DT_X> outputLocal,
        LocalTensor<DT_X> evenLocal, LocalTensor<DT_X> oddLocal)
    {
        const uint32_t inputRowElements = inputRowElements_;
        const uint32_t outputRowElements = outputRowElements_;
        const uint32_t outputBatch = inputRow / height_;
        const uint32_t inputH = inputRow - outputBatch * height_;
        const uint32_t tileInputElements = tileRows * inputRowElements;

        for (uint32_t blockH = 0; blockH < 2; ++blockH)
        {
            const uint32_t inputBatchEven =
                (blockH << 1) * outBatch_ + outputBatch;
            const uint32_t inputBatchOdd = inputBatchEven + outBatch_;
            const uint32_t inputEvenBase =
                (inputBatchEven * height_ + inputH) * inputRowElements;
            const uint32_t inputOddBase =
                (inputBatchOdd * height_ + inputH) * inputRowElements;
            CopyGmToLocal(evenLocal, inputEvenBase, tileInputElements);
            CopyGmToLocal(oddLocal, inputOddBase, tileInputElements);
            PipeBarrier<PIPE_ALL>();

            for (uint32_t row = 0; row < tileRows; ++row)
            {
                const uint32_t inputOffset = row * inputRowElements;
                const uint32_t outputOffset =
                    ((row << 1) + blockH) * outputRowElements;
                CopyLocalInterleavedColumns(outputLocal[outputOffset],
                                            evenLocal[inputOffset],
                                            oddLocal[inputOffset], width_);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ProcessBlockSize2InputMajor()
    {
        const uint32_t endRow = startPoint_ + pointCount_;
        uint32_t pendingOutputOffset = 0;
        uint32_t pendingCount = 0;
        uint32_t pendingRunBlocks = 0;
        uint32_t pendingDstStrideBytes = 0;
        bool hasPendingCopy = false;
        for (uint32_t inputRow = startPoint_; inputRow < endRow; ++inputRow)
        {
            const uint32_t inputBatch = inputRow / height_;
            const uint32_t inputH = inputRow - inputBatch * height_;
            const uint32_t blockId = inputBatch / outBatch_;
            const uint32_t outputBatch = inputBatch - blockId * outBatch_;
            const uint32_t blockH = blockId >> 1;
            const uint32_t blockW = blockId & 1;
            const uint32_t uncroppedH = (inputH << 1) + blockH;
            if (uncroppedH < cropTop_)
            {
                continue;
            }
            const uint32_t outputH = uncroppedH - cropTop_;
            if (outputH >= outHeight_)
            {
                continue;
            }
            uint32_t inputStartW = 0;
            if (cropLeft_ > blockW)
            {
                inputStartW = (cropLeft_ - blockW + 1) >> 1;
            }
            const uint32_t firstOutputW =
                (inputStartW << 1) + blockW - cropLeft_;
            if (firstOutputW >= outWidth_ || inputStartW >= width_)
            {
                continue;
            }
            const uint32_t maxInputColsByOutput =
                ((outWidth_ - 1 - firstOutputW) >> 1) + 1;
            const uint32_t maxInputColsByInput = width_ - inputStartW;
            const uint32_t inputCols =
                maxInputColsByInput < maxInputColsByOutput
                    ? maxInputColsByInput
                    : maxInputColsByOutput;
            const uint32_t inputBase =
                ((inputBatch * height_ + inputH) * width_ + inputStartW) *
                depth_;
            const uint32_t outputBase =
                ((outputBatch * outHeight_ + outputH) * outWidth_ +
                 firstOutputW) *
                depth_;
            for (uint32_t element = 0; element < depth_;
                 element += tileDepth_)
            {
                const uint32_t remaining = depth_ - element;
                const uint32_t count =
                    tileDepth_ < remaining ? tileDepth_ : remaining;
                const uint32_t maxRunBlocks = GetMaxRunBlocks(count);
                const uint32_t srcStrideBytes =
                    (depth_ - count) * sizeof(DT_X);
                const uint32_t dstStrideBytes =
                    ((depth_ << 1) - count) * sizeof(DT_X);
                uint32_t doneBlocks = 0;
                while (doneBlocks < inputCols)
                {
                    const uint32_t remainingBlocks = inputCols - doneBlocks;
                    const uint32_t runBlocks =
                        remainingBlocks < maxRunBlocks ? remainingBlocks
                                                       : maxRunBlocks;
                    CopyStridedRun(inputBase + doneBlocks * depth_ + element,
                                   outputBase +
                                       doneBlocks * (depth_ << 1) + element,
                                   count, runBlocks, srcStrideBytes,
                                   dstStrideBytes);
                    if (hasPendingCopy)
                    {
                        CopyOutStridedRun(pendingOutputOffset, pendingCount,
                                          pendingRunBlocks,
                                          pendingDstStrideBytes);
                    }
                    pendingOutputOffset =
                        outputBase + doneBlocks * (depth_ << 1) + element;
                    pendingCount = count;
                    pendingRunBlocks = runBlocks;
                    pendingDstStrideBytes = dstStrideBytes;
                    hasPendingCopy = true;
                    doneBlocks += runBlocks;
                }
            }
        }
        if (hasPendingCopy)
        {
            CopyOutStridedRun(pendingOutputOffset, pendingCount,
                              pendingRunBlocks, pendingDstStrideBytes);
        }
    }

    __aicore__ inline void ProcessStridedRowsBlockSize2()
    {
        if ((depthBytes_ & 31U) == 0 && tileDepth_ >= depth_ &&
            depthBytes_ <= 2097120U)
        {
            ProcessStridedRowsBlockSize2AlignedFullDepth();
            return;
        }
        const uint32_t endPoint = startPoint_ + pointCount_;
        uint32_t pendingOutputOffset = 0;
        uint32_t pendingCount = 0;
        uint32_t pendingRunBlocks = 0;
        uint32_t pendingDstStrideBytes = 0;
        bool hasPendingCopy = false;
        uint32_t outputPoint = startPoint_;
        while (outputPoint < endPoint)
        {
            const uint32_t outputBatch = outputPoint / outputPointsPerBatch_;
            const uint32_t outputOffset = outputPoint % outputPointsPerBatch_;
            const uint32_t outputH = outputOffset / outWidth_;
            const uint32_t outputW = outputOffset % outWidth_;
            const uint32_t rowStartPoint = outputPoint - outputW;
            const uint32_t rowEndPoint = rowStartPoint + outWidth_;
            const uint32_t segmentEndPoint =
                rowEndPoint < endPoint ? rowEndPoint : endPoint;
            const uint32_t segmentStartW = outputPoint - rowStartPoint;
            const uint32_t segmentEndW = segmentEndPoint - rowStartPoint;

            const uint32_t uncroppedH = outputH + cropTop_;
            const uint32_t inputH = uncroppedH >> 1;
            const uint32_t blockH = uncroppedH & 1;
            const uint32_t firstWindowEnd =
                segmentEndW < segmentStartW + 2 ? segmentEndW
                                                : segmentStartW + 2;
            for (uint32_t firstW = segmentStartW; firstW < firstWindowEnd;
                 ++firstW)
            {
                const uint32_t uncroppedW = firstW + cropLeft_;
                const uint32_t blockW = uncroppedW & 1;
                const uint32_t totalRunBlocks =
                    (segmentEndW - 1 - firstW) / 2 + 1;
                const uint32_t inputBatch =
                    ((blockH << 1) + blockW) * outBatch_ + outputBatch;
                const uint32_t inputW = uncroppedW >> 1;
                const uint32_t inputBase =
                    (inputBatch * height_ + inputH) * inputRowElements_ +
                    inputW * depth_;
                const uint32_t outputBase =
                    (rowStartPoint + firstW) * depth_;

                for (uint32_t element = 0; element < depth_;
                     element += tileDepth_)
                {
                    const uint32_t remaining = depth_ - element;
                    const uint32_t count =
                        tileDepth_ < remaining ? tileDepth_ : remaining;
                    const uint32_t maxRunBlocks = GetMaxRunBlocks(count);
                    const uint32_t srcStrideBytes =
                        (depth_ - count) * sizeof(DT_X);
                    const uint32_t dstStrideBytes =
                        ((depth_ << 1) - count) * sizeof(DT_X);
                    uint32_t doneBlocks = 0;
                    while (doneBlocks < totalRunBlocks)
                    {
                        const uint32_t remainingBlocks =
                            totalRunBlocks - doneBlocks;
                        const uint32_t runBlocks =
                            remainingBlocks < maxRunBlocks
                                ? remainingBlocks
                                : maxRunBlocks;
                        CopyStridedRun(inputBase + doneBlocks * depth_ +
                                           element,
                                       outputBase +
                                           doneBlocks * (depth_ << 1) +
                                           element,
                                       count, runBlocks, srcStrideBytes,
                                       dstStrideBytes);
                        if (hasPendingCopy)
                        {
                            CopyOutStridedRun(pendingOutputOffset,
                                              pendingCount,
                                              pendingRunBlocks,
                                              pendingDstStrideBytes);
                        }
                        pendingOutputOffset =
                            outputBase + doneBlocks * (depth_ << 1) + element;
                        pendingCount = count;
                        pendingRunBlocks = runBlocks;
                        pendingDstStrideBytes = dstStrideBytes;
                        hasPendingCopy = true;
                        doneBlocks += runBlocks;
                    }
                }
            }
            outputPoint = segmentEndPoint;
        }
        if (hasPendingCopy)
        {
            CopyOutStridedRun(pendingOutputOffset, pendingCount,
                              pendingRunBlocks, pendingDstStrideBytes);
        }
    }

    __aicore__ inline void ProcessStridedRowsBlockSize2AlignedFullDepth()
    {
        const uint32_t endPoint = startPoint_ + pointCount_;
        uint32_t pendingOutputOffset = 0;
        uint32_t pendingRunBlocks = 0;
        bool hasPendingCopy = false;
        const uint32_t maxRunBlocks = GetMaxRunBlocks(depth_);
        const uint32_t dstStrideBytes = depthBytes_;
        uint32_t outputPoint = startPoint_;
        while (outputPoint < endPoint)
        {
            const uint32_t outputBatch = outputPoint / outputPointsPerBatch_;
            const uint32_t outputOffset = outputPoint % outputPointsPerBatch_;
            const uint32_t outputH = outputOffset / outWidth_;
            const uint32_t outputW = outputOffset % outWidth_;
            const uint32_t rowStartPoint = outputPoint - outputW;
            const uint32_t rowEndPoint = rowStartPoint + outWidth_;
            const uint32_t segmentEndPoint =
                rowEndPoint < endPoint ? rowEndPoint : endPoint;
            const uint32_t segmentStartW = outputPoint - rowStartPoint;
            const uint32_t segmentEndW = segmentEndPoint - rowStartPoint;

            const uint32_t uncroppedH = outputH + cropTop_;
            const uint32_t inputH = uncroppedH >> 1;
            const uint32_t blockH = uncroppedH & 1;
            const uint32_t firstWindowEnd =
                segmentEndW < segmentStartW + 2 ? segmentEndW
                                                : segmentStartW + 2;
            for (uint32_t firstW = segmentStartW; firstW < firstWindowEnd;
                 ++firstW)
            {
                const uint32_t uncroppedW = firstW + cropLeft_;
                const uint32_t blockW = uncroppedW & 1;
                const uint32_t totalRunBlocks =
                    (segmentEndW - 1 - firstW) / 2 + 1;
                const uint32_t inputBatch =
                    ((blockH << 1) + blockW) * outBatch_ + outputBatch;
                const uint32_t inputW = uncroppedW >> 1;
                const uint32_t inputBase =
                    (inputBatch * height_ + inputH) * inputRowElements_ +
                    inputW * depth_;
                const uint32_t outputBase =
                    (rowStartPoint + firstW) * depth_;

                uint32_t doneBlocks = 0;
                while (doneBlocks < totalRunBlocks)
                {
                    const uint32_t remainingBlocks =
                        totalRunBlocks - doneBlocks;
                    const uint32_t runBlocks =
                        remainingBlocks < maxRunBlocks
                            ? remainingBlocks
                            : maxRunBlocks;
                    CopyStridedRunAligned(inputBase + doneBlocks * depth_,
                                          depth_, runBlocks, 0);
                    if (hasPendingCopy)
                    {
                        CopyOutStridedRunAligned(pendingOutputOffset, depth_,
                                                 pendingRunBlocks,
                                                 dstStrideBytes);
                    }
                    pendingOutputOffset =
                        outputBase + doneBlocks * (depth_ << 1);
                    pendingRunBlocks = runBlocks;
                    hasPendingCopy = true;
                    doneBlocks += runBlocks;
                }
            }
            outputPoint = segmentEndPoint;
        }
        if (hasPendingCopy)
        {
            CopyOutStridedRunAligned(pendingOutputOffset, depth_,
                                     pendingRunBlocks, dstStrideBytes);
        }
    }

    __aicore__ inline void ProcessBlockSize2LargeDepthInputPackedVectorCopy()
    {
        LocalTensor<DT_X> evenLocal0 = evenRowBuf_.Get<DT_X>();
        LocalTensor<DT_X> oddLocal0 = oddRowBuf_.Get<DT_X>();
        LocalTensor<DT_X> tileLocal0 = packedRowBuf_.Get<DT_X>();
        LocalTensor<DT_X> evenLocal1 = evenRowBuf2_.Get<DT_X>();
        LocalTensor<DT_X> oddLocal1 = oddRowBuf2_.Get<DT_X>();
        LocalTensor<DT_X> tileLocal1 = packedRowBuf2_.Get<DT_X>();
        const uint32_t outputRows = outBatch_ * outHeight_;
        const uint32_t tileCols =
            rowPackedTileCols_ == 0 ? outWidth_ : rowPackedTileCols_;
        const uint32_t colTiles = (outWidth_ + tileCols - 1U) / tileCols;
        const uint32_t endTask = startPoint_ + pointCount_;

        if (startPoint_ >= endTask)
        {
            return;
        }

        uint32_t task = startPoint_;
        uint32_t taskSerial = 0;
        bool hasPrevious = false;
        uint32_t previousTask = 0;
        uint32_t previousBuffer = 0;
        while (task < endTask)
        {
            const uint32_t bufferId = taskSerial & 1U;
            if (taskSerial >= BUFFER_NUM)
            {
                WaitFlag<HardEvent::MTE3_MTE2>(bufferId);
            }
            if (bufferId == 0)
            {
                CopyInLargeDepthInputPackedTask(task, evenLocal0, oddLocal0,
                                                outputRows, tileCols,
                                                colTiles);
            }
            else
            {
                CopyInLargeDepthInputPackedTask(task, evenLocal1, oddLocal1,
                                                outputRows, tileCols,
                                                colTiles);
            }
            SetFlag<HardEvent::MTE2_V>(bufferId);

            if (hasPrevious)
            {
                WaitFlag<HardEvent::MTE2_V>(previousBuffer);
                if (previousBuffer == 0)
                {
                    CopyInterleaveLargeDepthInputPackedTask(
                        previousTask, evenLocal0, oddLocal0, tileLocal0,
                        outputRows, tileCols, colTiles);
                    SetFlag<HardEvent::V_MTE3>(previousBuffer);
                    WaitFlag<HardEvent::V_MTE3>(previousBuffer);
                    CopyOutLargeDepthInputPackedContiguousTask(
                        previousTask, tileLocal0, outputRows, tileCols,
                        colTiles);
                }
                else
                {
                    CopyInterleaveLargeDepthInputPackedTask(
                        previousTask, evenLocal1, oddLocal1, tileLocal1,
                        outputRows, tileCols, colTiles);
                    SetFlag<HardEvent::V_MTE3>(previousBuffer);
                    WaitFlag<HardEvent::V_MTE3>(previousBuffer);
                    CopyOutLargeDepthInputPackedContiguousTask(
                        previousTask, tileLocal1, outputRows, tileCols,
                        colTiles);
                }
                SetFlag<HardEvent::MTE3_MTE2>(previousBuffer);
            }

            previousTask = task;
            previousBuffer = bufferId;
            hasPrevious = true;
            ++task;
            ++taskSerial;
        }

        if (hasPrevious)
        {
            WaitFlag<HardEvent::MTE2_V>(previousBuffer);
            if (previousBuffer == 0)
            {
                CopyInterleaveLargeDepthInputPackedTask(
                    previousTask, evenLocal0, oddLocal0, tileLocal0,
                    outputRows, tileCols, colTiles);
                SetFlag<HardEvent::V_MTE3>(previousBuffer);
                WaitFlag<HardEvent::V_MTE3>(previousBuffer);
                CopyOutLargeDepthInputPackedContiguousTask(
                    previousTask, tileLocal0, outputRows, tileCols, colTiles);
            }
            else
            {
                CopyInterleaveLargeDepthInputPackedTask(
                    previousTask, evenLocal1, oddLocal1, tileLocal1,
                    outputRows, tileCols, colTiles);
                SetFlag<HardEvent::V_MTE3>(previousBuffer);
                WaitFlag<HardEvent::V_MTE3>(previousBuffer);
                CopyOutLargeDepthInputPackedContiguousTask(
                    previousTask, tileLocal1, outputRows, tileCols, colTiles);
            }
            SetFlag<HardEvent::MTE3_MTE2>(previousBuffer);
            WaitFlag<HardEvent::MTE3_MTE2>(previousBuffer);
            if (taskSerial >= BUFFER_NUM)
            {
                WaitFlag<HardEvent::MTE3_MTE2>(previousBuffer ^ 1U);
            }
        }
    }

    __aicore__ inline void CopyInterleaveLargeDepthInputPackedTask(
        uint32_t task, LocalTensor<DT_X> evenLocal,
        LocalTensor<DT_X> oddLocal, LocalTensor<DT_X> tileLocal,
        uint32_t outputRows, uint32_t tileCols, uint32_t colTiles)
    {
        const uint32_t outputRow = task / colTiles;
        if (outputRow >= outputRows)
        {
            return;
        }
        const uint32_t tileId = task - outputRow * colTiles;
        const uint32_t outputCol = tileId * tileCols;
        const uint32_t remainingCols = outWidth_ - outputCol;
        const uint32_t currentTileCols =
            remainingCols < tileCols ? remainingCols : tileCols;
        const uint32_t colsEven = (currentTileCols + 1U) >> 1;
        const uint32_t colsOdd = currentTileCols >> 1;
        const uint64_t mask = static_cast<uint64_t>(256U / sizeof(DT_X));
        const uint8_t repeatTime = static_cast<uint8_t>(depthBytes_ >> 8);
        CopyRepeatParams params{1, 1, 8, 8};

        for (uint32_t col = 0; col < colsEven; ++col)
        {
            Copy(tileLocal[(col << 1) * depth_],
                 evenLocal[col * depth_], mask, repeatTime, params);
        }
        for (uint32_t col = 0; col < colsOdd; ++col)
        {
            Copy(tileLocal[((col << 1) + 1U) * depth_],
                 oddLocal[col * depth_], mask, repeatTime, params);
        }
    }

    __aicore__ inline void CopyOutLargeDepthInputPackedContiguousTask(
        uint32_t task, LocalTensor<DT_X> tileLocal, uint32_t outputRows,
        uint32_t tileCols, uint32_t colTiles)
    {
        const uint32_t outputRow = task / colTiles;
        if (outputRow >= outputRows)
        {
            return;
        }
        const uint32_t tileId = task - outputRow * colTiles;
        const uint32_t outputCol = tileId * tileCols;
        const uint32_t remainingCols = outWidth_ - outputCol;
        const uint32_t currentTileCols =
            remainingCols < tileCols ? remainingCols : tileCols;
        const uint32_t outputOffset =
            (outputRow * outWidth_ + outputCol) * depth_;
        CopyLocalToGm(outputOffset, tileLocal, currentTileCols * depth_);
    }

    __aicore__ inline void CopyInLargeDepthInputPackedTask(
        uint32_t task, LocalTensor<DT_X> evenLocal,
        LocalTensor<DT_X> oddLocal, uint32_t outputRows, uint32_t tileCols,
        uint32_t colTiles)
    {
        const uint32_t outputRow = task / colTiles;
        if (outputRow >= outputRows)
        {
            return;
        }
        const uint32_t tileId = task - outputRow * colTiles;
        const uint32_t outputCol = tileId * tileCols;
        const uint32_t remainingCols = outWidth_ - outputCol;
        const uint32_t currentTileCols =
            remainingCols < tileCols ? remainingCols : tileCols;
        const uint32_t endCol = outputCol + currentTileCols;

        const uint32_t outputBatch = outputRow / outHeight_;
        const uint32_t outputH = outputRow - outputBatch * outHeight_;
        const uint32_t inputH = outputH >> 1;
        const uint32_t blockH = outputH & 1U;
        const uint32_t inputBatchEven =
            (blockH << 1) * outBatch_ + outputBatch;
        const uint32_t inputBatchOdd = inputBatchEven + outBatch_;

        const uint32_t firstEvenW =
            (outputCol & 1U) == 0 ? outputCol : outputCol + 1U;
        const uint32_t firstOddW =
            (outputCol & 1U) == 0 ? outputCol + 1U : outputCol;
        const uint32_t colsEven =
            firstEvenW < endCol ? ((endCol - 1U - firstEvenW) >> 1) + 1U
                                : 0U;
        const uint32_t colsOdd =
            firstOddW < endCol ? ((endCol - 1U - firstOddW) >> 1) + 1U
                               : 0U;

        if (colsEven != 0)
        {
            const uint32_t inputEvenW = firstEvenW >> 1;
            const uint32_t inputEvenBase =
                ((inputBatchEven * height_ + inputH) * width_ + inputEvenW) *
                depth_;
            DataCopy(evenLocal, xGm_[inputEvenBase], colsEven * depth_);
        }
        if (colsOdd != 0)
        {
            const uint32_t inputOddW = firstOddW >> 1;
            const uint32_t inputOddBase =
                ((inputBatchOdd * height_ + inputH) * width_ + inputOddW) *
                depth_;
            DataCopy(oddLocal, xGm_[inputOddBase], colsOdd * depth_);
        }
    }

    __aicore__ inline void ProcessStridedRows()
    {
        const uint32_t outputPointsPerBatch = outputPointsPerBatch_;
        const uint32_t endPoint = startPoint_ + pointCount_;
        uint32_t pendingOutputOffset = 0;
        uint32_t pendingCount = 0;
        uint32_t pendingRunBlocks = 0;
        uint32_t pendingDstStrideBytes = 0;
        bool hasPendingCopy = false;
        uint32_t outputPoint = startPoint_;
        while (outputPoint < endPoint)
        {
            const uint32_t outputBatch = outputPoint / outputPointsPerBatch;
            const uint32_t outputOffset = outputPoint % outputPointsPerBatch;
            const uint32_t outputH = outputOffset / outWidth_;
            const uint32_t outputW = outputOffset % outWidth_;
            const uint32_t rowStartPoint = outputPoint - outputW;
            const uint32_t rowEndPoint = rowStartPoint + outWidth_;
            const uint32_t segmentEndPoint =
                rowEndPoint < endPoint ? rowEndPoint : endPoint;
            const uint32_t segmentStartW = outputPoint - rowStartPoint;
            const uint32_t segmentEndW = segmentEndPoint - rowStartPoint;

            const uint32_t uncroppedH = outputH + cropTop_;
            const uint32_t inputH = uncroppedH / blockSize_;
            const uint32_t blockH = uncroppedH % blockSize_;
            const uint32_t firstWindowEnd =
                segmentEndW < segmentStartW + blockSize_
                    ? segmentEndW
                    : segmentStartW + blockSize_;
            for (uint32_t firstW = segmentStartW; firstW < firstWindowEnd;
                 ++firstW)
            {
                const uint32_t blockW = (firstW + cropLeft_) % blockSize_;
                const uint32_t totalRunBlocks =
                    (segmentEndW - 1 - firstW) / blockSize_ + 1;
                const uint32_t uncroppedW = firstW + cropLeft_;
                const uint32_t inputBatch =
                    (blockH * blockSize_ + blockW) * outBatch_ + outputBatch;
                const uint32_t inputW = uncroppedW / blockSize_;
                const uint32_t inputBase =
                    (inputBatch * height_ + inputH) * inputRowElements_ +
                    inputW * depth_;
                const uint32_t outputBase =
                    (rowStartPoint + firstW) * depth_;

                for (uint32_t element = 0; element < depth_;
                     element += tileDepth_)
                {
                    const uint32_t remaining = depth_ - element;
                    const uint32_t count =
                        tileDepth_ < remaining ? tileDepth_ : remaining;
                    const uint32_t maxRunBlocks = GetMaxRunBlocks(count);
                    const uint32_t srcStrideBytes =
                        (depth_ - count) * sizeof(DT_X);
                    const uint32_t dstStrideBytes =
                        (blockSize_ * depth_ - count) * sizeof(DT_X);
                    uint32_t doneBlocks = 0;
                    while (doneBlocks < totalRunBlocks)
                    {
                        const uint32_t remainingBlocks =
                            totalRunBlocks - doneBlocks;
                        const uint32_t runBlocks =
                            remainingBlocks < maxRunBlocks
                                ? remainingBlocks
                                : maxRunBlocks;
                        CopyStridedRun(inputBase + doneBlocks * depth_ +
                                           element,
                                       outputBase +
                                           doneBlocks * blockSize_ * depth_ +
                                           element,
                                       count, runBlocks, srcStrideBytes,
                                       dstStrideBytes);
                        if (hasPendingCopy)
                        {
                            CopyOutStridedRun(pendingOutputOffset,
                                              pendingCount,
                                              pendingRunBlocks,
                                              pendingDstStrideBytes);
                        }
                        pendingOutputOffset =
                            outputBase + doneBlocks * blockSize_ * depth_ +
                            element;
                        pendingCount = count;
                        pendingRunBlocks = runBlocks;
                        pendingDstStrideBytes = dstStrideBytes;
                        hasPendingCopy = true;
                        doneBlocks += runBlocks;
                    }
                }
            }
            outputPoint = segmentEndPoint;
        }
        if (hasPendingCopy)
        {
            CopyOutStridedRun(pendingOutputOffset, pendingCount,
                              pendingRunBlocks, pendingDstStrideBytes);
        }
    }

    __aicore__ inline void ProcessBlockSizeOne()
    {
        const uint32_t outputPointsPerBatch = outputPointsPerBatch_;
        uint32_t pendingOutputOffset = 0;
        uint32_t pendingCount = 0;
        uint32_t pendingRunBlocks = 0;
        uint32_t pendingDstStrideBytes = 0;
        bool hasPendingCopy = false;
        uint32_t localPoint = 0;
        while (localPoint < pointCount_)
        {
            const uint32_t outputPoint = startPoint_ + localPoint;
            const uint32_t outputBatch = outputPoint / outputPointsPerBatch;
            const uint32_t outputOffset = outputPoint % outputPointsPerBatch;
            const uint32_t outputH = outputOffset / outWidth_;
            const uint32_t outputW = outputOffset % outWidth_;
            uint32_t runPoints = outWidth_ - outputW;
            const uint32_t remainingPoints = pointCount_ - localPoint;
            runPoints = runPoints < remainingPoints ? runPoints : remainingPoints;

            const uint32_t inputH = outputH + cropTop_;
            const uint32_t inputW = outputW + cropLeft_;
            const uint32_t inputBase =
                ((outputBatch * height_ + inputH) * width_ + inputW) * depth_;
            const uint32_t outputBase = outputPoint * depth_;
            const uint32_t rowElementCount = runPoints * depth_;

            for (uint32_t element = 0; element < rowElementCount;
                 element += tileDepth_)
            {
                const uint32_t remaining = rowElementCount - element;
                const uint32_t count =
                    tileDepth_ < remaining ? tileDepth_ : remaining;
                CopyInSegment(inputBase + element, count);
                if (hasPendingCopy)
                {
                    CopyOutSegment(pendingOutputOffset, pendingCount);
                }
                pendingOutputOffset = outputBase + element;
                pendingCount = count;
                hasPendingCopy = true;
            }
            localPoint += runPoints;
        }
        if (hasPendingCopy)
        {
            CopyOutSegment(pendingOutputOffset, pendingCount);
        }
    }

    __aicore__ inline uint32_t AlignUpBlockBytes(uint32_t bytes) const
    {
        return (bytes + 31) / 32 * 32;
    }

    __aicore__ inline uint32_t GetMaxRunBlocks(uint32_t count) const
    {
        const uint32_t blockBytes =
            AlignUpBlockBytes(count * sizeof(DT_X));
        uint32_t maxRunBlocks = tileBufferBytes_ / blockBytes;
        if (maxRunBlocks == 0)
        {
            maxRunBlocks = 1;
        }
        return maxRunBlocks < 4095 ? maxRunBlocks : 4095;
    }

    __aicore__ inline bool CanUseAlignedCopy(uint32_t offset,
                                             uint32_t count) const
    {
        const uint32_t alignElements =
            alignElements_ != 0 ? alignElements_ : 32 / sizeof(DT_X);
        return (offset % alignElements == 0) &&
               (count % alignElements == 0);
    }

    __aicore__ inline uint32_t AlignUpBlockElements(uint32_t count) const
    {
        return AlignUpBlockBytes(count * sizeof(DT_X)) / sizeof(DT_X);
    }

    __aicore__ inline void CopyGmToLocal(LocalTensor<DT_X> local,
                                         uint32_t inputOffset,
                                         uint32_t count)
    {
        if (CanUseAlignedCopy(inputOffset, count))
        {
            DataCopy(local, xGm_[inputOffset], count);
        }
        else
        {
            DataCopyExtParams params{
                1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(local, xGm_[inputOffset], params, padParams);
        }
    }

    __aicore__ inline void CopyLocalToGm(uint32_t outputOffset,
                                         LocalTensor<DT_X> local,
                                         uint32_t count)
    {
        if (CanUseAlignedCopy(outputOffset, count))
        {
            DataCopy(yGm_[outputOffset], local, count);
        }
        else
        {
            DataCopyExtParams params{
                1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(yGm_[outputOffset], local, params);
        }
    }

    __aicore__ inline void CopyLocalSegment(LocalTensor<DT_X> dst,
                                            LocalTensor<DT_X> src,
                                            uint32_t count)
    {
        const uint32_t countBytes = count * sizeof(DT_X);
        if (countBytes % 32 == 0)
        {
            DataCopy(dst, src, count);
        }
        else
        {
            Adds(dst, src, static_cast<DT_X>(0), count);
        }
    }

    __aicore__ inline void CopyLocalInterleavedColumns(
        LocalTensor<DT_X> outputLocal, LocalTensor<DT_X> evenLocal,
        LocalTensor<DT_X> oddLocal, uint32_t cols)
    {
        const uint32_t depthBytes =
            depthBytes_ != 0 ? depthBytes_ : depth_ * sizeof(DT_X);
        if (depthBytes % 32 == 0 && cols > 1 && cols <= 4095 &&
            depthBytes / 32 <= 65535)
        {
            const uint16_t blockLen =
                static_cast<uint16_t>(depthBytes / 32);
            DataCopyParams params{
                static_cast<uint16_t>(cols), blockLen, 0, blockLen};
            DataCopy(outputLocal, evenLocal, params);
            DataCopy(outputLocal[depth_], oddLocal, params);
            return;
        }
        for (uint32_t col = 0; col < cols; ++col)
        {
            const uint32_t srcOffset = col * depth_;
            const uint32_t dstEvenOffset = (col << 1) * depth_;
            const uint32_t dstOddOffset = dstEvenOffset + depth_;
            CopyLocalSegment(outputLocal[dstEvenOffset],
                             evenLocal[srcOffset], depth_);
            CopyLocalSegment(outputLocal[dstOddOffset],
                             oddLocal[srcOffset], depth_);
        }
    }

    __aicore__ inline bool CopyLocalStridedColumns(
        LocalTensor<DT_X> outputLocal, LocalTensor<DT_X> inputLocal,
        uint32_t cols)
    {
        const uint32_t depthBytes =
            depthBytes_ != 0 ? depthBytes_ : depth_ * sizeof(DT_X);
        if (depthBytes % 32 != 0 || cols <= 1 || cols > 4095 ||
            depthBytes / 32 > 65535)
        {
            return false;
        }
        const uint16_t blockLen = static_cast<uint16_t>(depthBytes / 32);
        DataCopyParams params{
            static_cast<uint16_t>(cols), blockLen, 0, blockLen};
        DataCopy(outputLocal, inputLocal, params);
        return true;
    }

    __aicore__ inline bool CopyLocalStridedColumnsGeneric(
        LocalTensor<DT_X> outputLocal, LocalTensor<DT_X> inputLocal,
        uint32_t cols)
    {
        const uint32_t depthBytes =
            depthBytes_ != 0 ? depthBytes_ : depth_ * sizeof(DT_X);
        if (depthBytes % 32 != 0 || cols <= 1 || cols > 4095 ||
            depthBytes / 32 > 65535)
        {
            return false;
        }
        const uint32_t blockLenU32 = depthBytes / 32;
        const uint32_t dstStrideU32 = (blockSize_ - 1) * blockLenU32;
        if (dstStrideU32 > 65535)
        {
            return false;
        }
        DataCopyParams params{
            static_cast<uint16_t>(cols),
            static_cast<uint16_t>(blockLenU32), 0,
            static_cast<uint16_t>(dstStrideU32)};
        DataCopy(outputLocal, inputLocal, params);
        return true;
    }

    __aicore__ inline void BuildGenericRowPackedTile(
        uint32_t outputRow, LocalTensor<DT_X> rowLocal,
        LocalTensor<DT_X> streamLocal,
        uint32_t tileStartW = 0, uint32_t tileWidth = 0)
    {
        const uint32_t endW = tileWidth != 0 ? tileStartW + tileWidth : outWidth_;
        const uint32_t outputBatch = outputRow / outHeight_;
        const uint32_t outputH = outputRow - outputBatch * outHeight_;
        const uint32_t uncroppedH = outputH + cropTop_;
        const uint32_t inputH = uncroppedH / blockSize_;
        const uint32_t blockH = uncroppedH - inputH * blockSize_;
        for (uint32_t blockW = 0; blockW < blockSize_; ++blockW)
        {
            const uint32_t firstOutputW =
                tileStartW + ((blockW + blockSize_ -
                               (tileStartW + cropLeft_) % blockSize_) %
                              blockSize_);
            if (firstOutputW >= endW)
            {
                continue;
            }
            const uint32_t streamCols =
                (endW - 1 - firstOutputW) / blockSize_ + 1;
            const uint32_t inputW =
                (firstOutputW + cropLeft_) / blockSize_;
            const uint32_t inputBatch =
                (blockH * blockSize_ + blockW) * outBatch_ + outputBatch;
            const uint32_t inputBase =
                ((inputBatch * height_ + inputH) * width_ + inputW) *
                depth_;
            CopyGmToLocal(streamLocal, inputBase, streamCols * depth_);
            PipeBarrier<PIPE_ALL>();
            if (!CopyLocalStridedColumnsGeneric(
                    rowLocal[(firstOutputW - tileStartW) * depth_],
                    streamLocal, streamCols))
            {
                const uint32_t baseOffset =
                    (firstOutputW - tileStartW) * depth_;
                for (uint32_t col = 0; col < streamCols; ++col)
                {
                    const uint32_t srcOffset = col * depth_;
                    const uint32_t dstOffset =
                        col * blockSize_ * depth_;
                    CopyLocalSegment(rowLocal[baseOffset + dstOffset],
                                     streamLocal[srcOffset], depth_);
                }
            }
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline bool MustSplitNonAlignedStridedWrite(
        uint32_t outputOffset, uint32_t count, uint32_t dstStrideBytes) const
    {
        const uint32_t blockLenBytes = count * sizeof(DT_X);
        if (CanUseAlignedCopy(outputOffset, count))
        {
            return false;
        }
        const uint32_t strideBytes = blockLenBytes + dstStrideBytes;
        const uint32_t alignElements =
            alignElements_ != 0 ? alignElements_ : 32 / sizeof(DT_X);
        const uint32_t startByteOffset =
            (outputOffset % alignElements) * sizeof(DT_X);
        if (strideBytes % 32 == 0 &&
            startByteOffset + blockLenBytes <= strideBytes)
        {
            return false;
        }
        return true;
    }

    __aicore__ inline void CopyInSegment(uint32_t inputOffset, uint32_t count)
    {
        LocalTensor<DT_X> local = copyQ_.AllocTensor<DT_X>();
        if (CanUseAlignedCopy(inputOffset, count))
        {
            DataCopy(local, xGm_[inputOffset], count);
        }
        else
        {
            DataCopyExtParams params{
                1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(local, xGm_[inputOffset], params, padParams);
        }
        copyQ_.EnQue(local);
    }

    __aicore__ inline void CopyStridedRun(uint32_t inputOffset,
                                          uint32_t outputOffset,
                                          uint32_t count,
                                          uint32_t runBlocks,
                                          uint32_t srcStrideBytes,
                                          uint32_t dstStrideBytes)
    {
        (void)outputOffset;
        (void)dstStrideBytes;
        if (runBlocks == 1)
        {
            CopyInSegment(inputOffset, count);
            return;
        }
        if (CanUseAlignedCopy(inputOffset, count) &&
            srcStrideBytes % 32 == 0)
        {
            CopyStridedRunAligned(inputOffset, count, runBlocks,
                                  srcStrideBytes);
            return;
        }
        LocalTensor<DT_X> local = copyQ_.AllocTensor<DT_X>();
        DataCopyExtParams params{
            static_cast<uint16_t>(runBlocks),
            static_cast<uint32_t>(count * sizeof(DT_X)),
            srcStrideBytes, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
        DataCopyPad(local, xGm_[inputOffset], params, padParams);
        copyQ_.EnQue(local);
    }

    __aicore__ inline void CopyStridedRunAligned(uint32_t inputOffset,
                                                 uint32_t count,
                                                 uint32_t runBlocks,
                                                 uint32_t srcStrideBytes)
    {
        LocalTensor<DT_X> local = copyQ_.AllocTensor<DT_X>();
        const uint16_t blockLen =
            static_cast<uint16_t>(count * sizeof(DT_X) / 32);
        const uint16_t srcStrideBlocks =
            static_cast<uint16_t>(srcStrideBytes / 32);
        DataCopyParams params{
            static_cast<uint16_t>(runBlocks), blockLen, srcStrideBlocks, 0};
        DataCopy(local, xGm_[inputOffset], params);
        copyQ_.EnQue(local);
    }

    __aicore__ inline void CopyOutSegment(uint32_t outputOffset,
                                          uint32_t count)
    {
        LocalTensor<DT_X> local = copyQ_.DeQue<DT_X>();
        if (CanUseAlignedCopy(outputOffset, count))
        {
            DataCopy(yGm_[outputOffset], local, count);
        }
        else
        {
            DataCopyExtParams params{
                1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(yGm_[outputOffset], local, params);
        }
        copyQ_.FreeTensor(local);
    }

    __aicore__ inline void CopyOutStridedRun(uint32_t outputOffset,
                                             uint32_t count,
                                             uint32_t runBlocks,
                                             uint32_t dstStrideBytes)
    {
        if (runBlocks == 1)
        {
            CopyOutSegment(outputOffset, count);
            return;
        }
        if (CanUseAlignedCopy(outputOffset, count) &&
            dstStrideBytes % 32 == 0)
        {
            CopyOutStridedRunAligned(outputOffset, count, runBlocks,
                                     dstStrideBytes);
            return;
        }
        if (MustSplitNonAlignedStridedWrite(outputOffset, count,
                                            dstStrideBytes))
        {
            CopyOutStridedRunSplit(outputOffset, count, runBlocks,
                                   dstStrideBytes);
            return;
        }
        LocalTensor<DT_X> local = copyQ_.DeQue<DT_X>();
        const uint32_t blockLenBytes = count * sizeof(DT_X);
        const uint32_t localStrideBytes =
            AlignUpBlockBytes(blockLenBytes) - blockLenBytes;
        DataCopyExtParams params{
            static_cast<uint16_t>(runBlocks),
            blockLenBytes, localStrideBytes, dstStrideBytes, 0};
        DataCopyPad(yGm_[outputOffset], local, params);
        copyQ_.FreeTensor(local);
    }

    __aicore__ inline void CopyOutStridedRunAligned(uint32_t outputOffset,
                                                    uint32_t count,
                                                    uint32_t runBlocks,
                                                    uint32_t dstStrideBytes)
    {
        LocalTensor<DT_X> local = copyQ_.DeQue<DT_X>();
        const uint16_t blockLen =
            static_cast<uint16_t>(count * sizeof(DT_X) / 32);
        const uint16_t dstStrideBlocks =
            static_cast<uint16_t>(dstStrideBytes / 32);
        DataCopyParams params{
            static_cast<uint16_t>(runBlocks), blockLen, 0, dstStrideBlocks};
        DataCopy(yGm_[outputOffset], local, params);
        copyQ_.FreeTensor(local);
    }

    __aicore__ inline void CopyOutStridedRunSplit(uint32_t outputOffset,
                                                  uint32_t count,
                                                  uint32_t runBlocks,
                                                  uint32_t dstStrideBytes)
    {
        LocalTensor<DT_X> local = copyQ_.DeQue<DT_X>();
        const uint32_t localStrideElements = AlignUpBlockElements(count);
        const uint32_t outputStrideElements =
            count + dstStrideBytes / sizeof(DT_X);
        DataCopyExtParams params{
            1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
        for (uint32_t blockIdx = 0; blockIdx < runBlocks; ++blockIdx)
        {
            DataCopyPad(yGm_[outputOffset + blockIdx * outputStrideElements],
                        local[blockIdx * localStrideElements], params);
        }
        copyQ_.FreeTensor(local);
    }

protected:
    // No vector compute occurs between the GM read and GM write. Bind both
    // vector positions so the queue supplies the MTE2-to-MTE3 dependency.
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> copyQ_;
    TBuf<TPosition::VECCALC> evenRowBuf_;
    TBuf<TPosition::VECCALC> oddRowBuf_;
    TBuf<TPosition::VECCALC> packedRowBuf_;
    TBuf<TPosition::VECCALC> evenRowBuf2_;
    TBuf<TPosition::VECCALC> oddRowBuf2_;
    TBuf<TPosition::VECCALC> packedRowBuf2_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    GlobalTensor<uint32_t> microTileOffsetGm_;
    uint32_t height_ = 0;
    uint32_t width_ = 0;
    uint32_t depth_ = 0;
    uint32_t outBatch_ = 0;
    uint32_t outHeight_ = 0;
    uint32_t outWidth_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t cropTop_ = 0;
    uint32_t cropLeft_ = 0;
    uint32_t tileDepth_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t rowPackedTileRows_ = 1;
    uint32_t rowPackedTileCols_ = 2;
    uint32_t inputRowElements_ = 0;
    uint32_t outputRowElements_ = 0;
    uint32_t outputPointsPerBatch_ = 0;
    uint32_t depthBytes_ = 0;
    uint32_t alignElements_ = 0;
    uint32_t tileBufferBytes_ = 0;
    uint32_t startPoint_ = 0;
    uint32_t pointCount_ = 0;
};

template <class DT_X>
class KernelBatchToSpaceDirectCopy : public KernelBatchToSpaceBase<DT_X>
{
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                const __gm__ BatchToSpaceTilingData *tiling,
                                TPipe *pipe)
    {
        this->InitCommon(x, y, tiling);
        this->InitCopyBuffer(pipe);
    }

    __aicore__ inline void Process()
    {
        this->ProcessDirectCopy();
    }
};

template <class DT_X>
class KernelBatchToSpaceBlockSizeOne : public KernelBatchToSpaceBase<DT_X>
{
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                const __gm__ BatchToSpaceTilingData *tiling,
                                TPipe *pipe)
    {
        this->InitCommon(x, y, tiling);
        this->InitCopyBuffer(pipe);
    }

    __aicore__ inline void Process()
    {
        this->ProcessBlockSizeOne();
    }
};

template <class DT_X>
class KernelBatchToSpaceBlockSize2RowPacked
    : public KernelBatchToSpaceBase<DT_X>
{
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                const __gm__ BatchToSpaceTilingData *tiling,
                                TPipe *pipe)
    {
        this->InitCommon(x, y, tiling);
        this->InitRowPackedBuffer(pipe);
    }

    __aicore__ inline void Process()
    {
        this->ProcessBlockSize2RowPacked();
    }
};

template <class DT_X>
class KernelBatchToSpaceBlockSize2HeightPacked
    : public KernelBatchToSpaceBase<DT_X>
{
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                const __gm__ BatchToSpaceTilingData *tiling,
                                TPipe *pipe)
    {
        this->InitCommon(x, y, tiling);
        this->InitHeightPackedBuffer(pipe);
    }

    __aicore__ inline void Process()
    {
        this->ProcessBlockSize2HeightPacked();
    }
};

template <class DT_X>
class KernelBatchToSpaceBlockSize2Residue
    : public KernelBatchToSpaceBase<DT_X>
{
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                const __gm__ BatchToSpaceTilingData *tiling,
                                TPipe *pipe)
    {
        this->InitCommon(x, y, tiling);
        const bool noCrop =
            this->cropTop_ == 0 && this->cropLeft_ == 0 &&
            this->outHeight_ == (this->height_ << 1) &&
            this->outWidth_ == (this->width_ << 1);
        if (!noCrop)
        {
            this->InitCopyBuffer(pipe);
            useRowPacked_ = false;
            return;
        }
        const uint32_t halfRowAligned =
            this->AlignUpBlockElements(this->inputRowElements_);
        const uint32_t rowAligned =
            this->AlignUpBlockElements(this->outputRowElements_);
        const uint32_t singleRowBytes =
            (halfRowAligned * 2U + rowAligned) * sizeof(DT_X);
        const uint32_t rowPackedBytes = singleRowBytes * BUFFER_NUM;
        if (rowPackedBytes > 0 && rowPackedBytes < 98304U)
        {
            this->rowPackedTileRows_ = 1;
            this->InitRowPackedBuffer(pipe);
            useRowPacked_ = true;
        }
        else
        {
            this->InitCopyBuffer(pipe);
            useRowPacked_ = false;
        }
    }

    __aicore__ inline void Process()
    {
        if (useRowPacked_)
        {
            ProcessStridedRowsBlockSize2WithRowPacked();
        }
        else
        {
            this->ProcessStridedRowsBlockSize2();
        }
    }

private:
    bool useRowPacked_ = false;

    __aicore__ inline void ProcessStridedRowsBlockSize2WithRowPacked()
    {
        const uint32_t endPoint = this->startPoint_ + this->pointCount_;
        if (this->startPoint_ >= endPoint)
        {
            return;
        }

        uint32_t rowStartPoint = this->startPoint_;
        const uint32_t startW = rowStartPoint % this->outWidth_;
        if (startW != 0)
        {
            rowStartPoint += this->outWidth_ - startW;
        }

        LocalTensor<DT_X> evenLocal =
            this->evenRowBuf_.template Get<DT_X>();
        LocalTensor<DT_X> oddLocal =
            this->oddRowBuf_.template Get<DT_X>();
        LocalTensor<DT_X> rowLocal =
            this->packedRowBuf_.template Get<DT_X>();

        while (rowStartPoint < endPoint)
        {
            const uint32_t outputPointOffset =
                rowStartPoint % this->outputPointsPerBatch_;
            const uint32_t outputH = outputPointOffset / this->outWidth_;
            const uint32_t outputBatch =
                rowStartPoint / this->outputPointsPerBatch_;
            const uint32_t outputRow =
                outputBatch * this->outHeight_ + outputH;

            this->BuildRowPackedNoCropTile(outputRow, 1, rowLocal,
                                           evenLocal, oddLocal);
            this->CopyLocalToGm(outputRow * this->outputRowElements_,
                                rowLocal, this->outputRowElements_);
            rowStartPoint += this->outWidth_;
        }
        PipeBarrier<PIPE_ALL>();
    }
};

template <class DT_X>
class KernelBatchToSpaceBlockSize2MicroTile
    : public KernelBatchToSpaceBase<DT_X>
{
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                const __gm__ BatchToSpaceTilingData *tiling,
                                TPipe *pipe)
    {
        this->InitCommon(x, y, tiling);
        this->InitMicroTileOffsetBuffer(tiling);
        this->InitMicroTileBuffer(pipe);
    }

    __aicore__ inline void Process()
    {
        this->ProcessBlockSize2MicroTile();
    }
};

template <class DT_X>
class KernelBatchToSpaceBlockSize2SmallNoCrop
    : public KernelBatchToSpaceBase<DT_X>
{
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                const __gm__ BatchToSpaceTilingData *tiling,
                                TPipe *pipe)
    {
        this->InitCommon(x, y, tiling);
        this->InitSmallNoCropBuffer(pipe);
    }

    __aicore__ inline void Process()
    {
        this->ProcessBlockSize2SmallNoCrop();
    }

    __aicore__ inline void InitStatic(
        GM_ADDR x, GM_ADDR y, const __gm__ BatchToSpaceTilingData *tiling)
    {
        this->InitCommon(x, y, tiling);
    }

    __aicore__ inline uint32_t GetStaticInputEvenBase(
        uint32_t outputBatch, uint32_t outputH,
        uint32_t inputBatchStride, uint32_t blockWBatchOffset,
        uint32_t halfRowElements) const
    {
        const uint32_t inputH = outputH >> 1;
        const uint32_t blockHOffset =
            (outputH & 1) == 0 ? 0 : (blockWBatchOffset << 1);
        return outputBatch * inputBatchStride + blockHOffset +
               inputH * halfRowElements;
    }

    __aicore__ inline void AdvanceStaticRow(uint32_t &outputBatch,
                                            uint32_t &outputH) const
    {
        ++outputH;
        if (outputH == this->outHeight_)
        {
            outputH = 0;
            ++outputBatch;
        }
    }

    __aicore__ inline void BuildStaticRow(
        LocalTensor<DT_X> rowLocal, LocalTensor<DT_X> evenLocal,
        LocalTensor<DT_X> oddLocal, const DataCopyParams &interleaveParams)
    {
        DataCopy(rowLocal, evenLocal, interleaveParams);
        DataCopy(rowLocal[this->depth_], oddLocal, interleaveParams);
    }

    __aicore__ inline void ProcessStaticTinyRows()
    {
        constexpr uint32_t TINY_TILE_ROWS = 4;
        LocalMemAllocator<Hardware::UB> allocator;
        const uint32_t halfRowElements = this->inputRowElements_;
        const uint32_t rowElements = this->outputRowElements_;
        const uint32_t halfRowStride =
            this->AlignUpBlockElements(halfRowElements);
        const uint32_t rowStride = this->AlignUpBlockElements(rowElements);
        LocalTensor<DT_X> evenLocal =
            allocator.Alloc<TPosition::VECIN, DT_X>(
                halfRowStride * TINY_TILE_ROWS);
        LocalTensor<DT_X> oddLocal =
            allocator.Alloc<TPosition::VECIN, DT_X>(
                halfRowStride * TINY_TILE_ROWS);
        LocalTensor<DT_X> rowLocal =
            allocator.Alloc<TPosition::VECOUT, DT_X>(
                rowStride * TINY_TILE_ROWS);

        const uint32_t inputBatchStride =
            this->height_ * halfRowElements;
        const uint32_t blockWBatchOffset =
            this->outBatch_ * inputBatchStride;
        const uint16_t blockLen =
            static_cast<uint16_t>(this->depthBytes_ >> 5);
        const uint16_t cols = static_cast<uint16_t>(this->width_);
        const DataCopyParams interleaveParams{cols, blockLen, 0, blockLen};

        uint32_t outputRow = this->startPoint_;
        const uint32_t endRow = this->startPoint_ + this->pointCount_;
        uint32_t outputBatch = outputRow / this->outHeight_;
        uint32_t outputH = outputRow - outputBatch * this->outHeight_;
        while (outputRow < endRow)
        {
            const uint32_t remainingRows = endRow - outputRow;
            const uint32_t currentRows =
                remainingRows > TINY_TILE_ROWS ? TINY_TILE_ROWS
                                               : remainingRows;
            uint32_t buildBatch = outputBatch;
            uint32_t buildH = outputH;
            for (uint32_t row = 0; row < currentRows; ++row)
            {
                const uint32_t inputEvenBase = GetStaticInputEvenBase(
                    buildBatch, buildH, inputBatchStride, blockWBatchOffset,
                    halfRowElements);
                DataCopy(evenLocal[row * halfRowStride],
                         this->xGm_[inputEvenBase], halfRowElements);
                DataCopy(oddLocal[row * halfRowStride],
                         this->xGm_[inputEvenBase + blockWBatchOffset],
                         halfRowElements);
                AdvanceStaticRow(buildBatch, buildH);
            }
            PipeBarrier<PIPE_ALL>();
            for (uint32_t row = 0; row < currentRows; ++row)
            {
                BuildStaticRow(rowLocal[row * rowStride],
                               evenLocal[row * halfRowStride],
                               oddLocal[row * halfRowStride],
                               interleaveParams);
            }
            PipeBarrier<PIPE_ALL>();
            DataCopy(this->yGm_[outputRow * rowElements], rowLocal,
                     currentRows * rowElements);
            PipeBarrier<PIPE_ALL>();
            outputRow += currentRows;
            outputBatch = buildBatch;
            outputH = buildH;
        }
    }

    __aicore__ inline void ProcessStatic()
    {
        if (this->pointCount_ == 0)
        {
            return;
        }

        InitSocState();
        if (this->depth_ == 32 && this->outWidth_ == 12)
        {
            ProcessStaticTinyRows();
            return;
        }

        this->ProcessBlockSize2SmallNoCrop();
    }
};

template <class DT_X>
class KernelBatchToSpaceBlockSize2LargeDepthInputPacked
    : public KernelBatchToSpaceBase<DT_X>
{
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                const __gm__ BatchToSpaceTilingData *tiling,
                                TPipe *pipe)
    {
        this->InitCommon(x, y, tiling);
        this->InitLargeDepthInputPackedBuffer(pipe);
    }

    __aicore__ inline void Process()
    {
        this->ProcessBlockSize2LargeDepthInputPackedVectorCopy();
    }
};

template <class DT_X>
class KernelBatchToSpaceGenericRowPacked
    : public KernelBatchToSpaceBase<DT_X>
{
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                const __gm__ BatchToSpaceTilingData *tiling,
                                TPipe *pipe)
    {
        this->InitCommon(x, y, tiling);
        this->InitGenericRowPackedBuffer(pipe);
    }

    __aicore__ inline void Process()
    {
        LocalTensor<DT_X> streamLocal =
            this->Sb().template Get<DT_X>();
        LocalTensor<DT_X> rowLocal =
            this->Pb().template Get<DT_X>();
        const uint32_t outputRows = this->Nb() * this->Nh();
        const uint32_t tileWidth = this->Tw();
        const uint32_t endRow = this->Ns() + this->Nc();
        for (uint32_t outputRow = this->Ns(); outputRow < endRow;
             ++outputRow)
        {
            if (outputRow >= outputRows)
            {
                return;
            }
            if (tileWidth != 0)
            {
                for (uint32_t tileStartW = 0; tileStartW < this->Nw();
                     tileStartW += tileWidth)
                {
                    const uint32_t currentTileWidth =
                        this->Nw() - tileStartW < tileWidth
                            ? this->Nw() - tileStartW
                            : tileWidth;
                    this->BuildGenericRowPackedTile(
                        outputRow, rowLocal, streamLocal,
                        tileStartW, currentTileWidth);
                    const uint32_t outputBase =
                        (outputRow * this->Nw() + tileStartW) *
                        this->Nd();
                    this->CopyLocalToGm(outputBase, rowLocal,
                                        currentTileWidth * this->Nd());
                    PipeBarrier<PIPE_ALL>();
                }
            }
            else
            {
                this->BuildGenericRowPackedTile(
                    outputRow, rowLocal, streamLocal);
                const uint32_t outputBase =
                    outputRow * this->Nw() * this->Nd();
                this->CopyLocalToGm(outputBase, rowLocal,
                                    this->Nw() * this->Nd());
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
};

template <class DT_X>
class KernelBatchToSpaceGeneric : public KernelBatchToSpaceBase<DT_X>
{
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                const __gm__ BatchToSpaceTilingData *tiling,
                                TPipe *pipe)
    {
        this->InitCommon(x, y, tiling);
        this->InitCopyBuffer(pipe);
    }

    __aicore__ inline void Process()
    {
        this->ProcessStridedRows();
    }
};

template <typename DT_X, uint32_t MODE>
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y,
                                          GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const __gm__ BatchToSpaceTilingData *tilingData =
        reinterpret_cast<__gm__ BatchToSpaceTilingData *>(tiling);
    if constexpr (MODE == BTS_TPL_MODE_BLOCK_SIZE_TWO_SMALL_NOCROP)
    {
        ICachePreLoad(6);
        KernelBatchToSpaceBlockSize2SmallNoCrop<DT_X> op;
        op.InitStatic(x, y, tilingData);
        op.ProcessStatic();
        return;
    }
    TPipe pipe;
    if constexpr (MODE == BTS_TPL_MODE_DIRECT_COPY)
    {
        KernelBatchToSpaceDirectCopy<DT_X> op;
        op.Init(x, y, tilingData, &pipe);
        op.Process();
        return;
    }
    else if constexpr (MODE == BTS_TPL_MODE_BLOCK_SIZE_ONE)
    {
        KernelBatchToSpaceBlockSizeOne<DT_X> op;
        op.Init(x, y, tilingData, &pipe);
        op.Process();
        return;
    }
    else if constexpr (MODE == BTS_TPL_MODE_BLOCK_SIZE_TWO_ROW_PACKED)
    {
        KernelBatchToSpaceBlockSize2RowPacked<DT_X> op;
        op.Init(x, y, tilingData, &pipe);
        op.Process();
        return;
    }
    else if constexpr (MODE == BTS_TPL_MODE_BLOCK_SIZE_TWO_HEIGHT_PACKED)
    {
        KernelBatchToSpaceBlockSize2HeightPacked<DT_X> op;
        op.Init(x, y, tilingData, &pipe);
        op.Process();
        return;
    }
    else if constexpr (MODE == BTS_TPL_MODE_BLOCK_SIZE_TWO_RESIDUE)
    {
        KernelBatchToSpaceBlockSize2Residue<DT_X> op;
        op.Init(x, y, tilingData, &pipe);
        op.Process();
        return;
    }
    else if constexpr (MODE == BTS_TPL_MODE_GENERIC_ROW_PACKED)
    {
        KernelBatchToSpaceGenericRowPacked<DT_X> op;
        op.Init(x, y, tilingData, &pipe);
        op.Process();
        return;
    }
    else if constexpr (MODE == BTS_TPL_MODE_BLOCK_SIZE_TWO_MICRO_TILE)
    {
        KernelBatchToSpaceBlockSize2MicroTile<DT_X> op;
        op.Init(x, y, tilingData, &pipe);
        op.Process();
        return;
    }
    else if constexpr (MODE ==
                       BTS_TPL_MODE_BLOCK_SIZE_TWO_LARGE_DEPTH_INPUT_PACKED)
    {
        KernelBatchToSpaceBlockSize2LargeDepthInputPacked<DT_X> op;
        op.Init(x, y, tilingData, &pipe);
        op.Process();
        return;
    }
    else
    {
        const uint32_t blockSize = tilingData->blockSize;
        if (blockSize == 2)
        {
            KernelBatchToSpaceBlockSize2Residue<DT_X> op;
            op.Init(x, y, tilingData, &pipe);
            op.Process();
            return;
        }
        KernelBatchToSpaceGeneric<DT_X> op;
        op.Init(x, y, tilingData, &pipe);
        op.Process();
    }
}
