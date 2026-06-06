#include "kernel_operator.h"

#include "batch_to_space_tiling.h"
using namespace AscendC;

namespace
{
    constexpr uint32_t MODE_FLAT = 1;
    constexpr uint32_t MODE_LINEAR = 2;
    constexpr uint32_t MODE_CHANNEL = 3;
    constexpr uint32_t MODE_ELEMENTWISE = 4;
    constexpr uint32_t MODE_ROW = 5;
    constexpr uint32_t MODE_ALIGNED = 6;
    constexpr uint32_t MODE_GENERAL = 7;
    constexpr uint32_t MODE_ROW_WIDE = 8;
    constexpr uint32_t MODE_ROW_OTHER = 10;
    constexpr uint32_t MODE_GROUPED_GENERAL = 11;
    constexpr uint32_t MODE_ALIGNED_INTERLEAVED = 12;
    constexpr uint32_t MODE_ALIGNED_BLOCK2_DIRECT = 13;
    constexpr uint32_t MODE_ALIGNED_BLOCK2_ROW_COMPACT = 14;
    constexpr uint32_t MODE_ALIGNED_BLOCK2_TINY_UB_DB = 15;
    constexpr uint32_t MODE_ALIGNED_BLOCK2_SMALL_DIRECT = 16;
    constexpr uint32_t MODE_ALIGNED_BLOCK2_FULL_ROW12_MULTIROW_DIRECT = 17;
    constexpr uint32_t MODE_ALIGNED_BLOCK4_D64_DIRECT = 18;
    constexpr uint32_t MODE_ALIGNED_BLOCK4_D64_PHASE_FAST = 19;
    constexpr uint32_t MODE_ALIGNED_BLOCK4_D64_UB_REORDER = 20;
    constexpr uint32_t MODE_GENERAL_DIRECT_DMA = 21;
    constexpr uint32_t MODE_GROUPED_GENERAL_PACK2 = 22;
    constexpr bool BTS_OPT_GROUPED_GENERAL_OVERLAP_GATHER_OUTPUT = true;
    constexpr bool BTS_OPT_GROUPED_GENERAL_OVERLAP_INPUT_OUTPUT = false;
    constexpr bool BTS_OPT_BLOCK2_GROUPED_INPUT_FAST = true;
    constexpr bool BTS_OPT_BLOCK2_GROUPED_OFFSET_FAST = true;
    constexpr bool BTS_OPT_BLOCK2_DEEP_DIRECT_LOCAL_GATHER = false;
    constexpr bool BTS_OPT_BLOCK2_DEEP_DIRECT_OUTPUT_SINGLE = false;
    constexpr bool BTS_OPT_BLOCK2_DEEP_DIRECT_PADDED_PHASE = false;
} // namespace

template <class DT_X, uint32_t MODE>
class KernelBatchToSpace
{
public:
    __aicore__ inline KernelBatchToSpace() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling)
    {
        batch_ = tiling.batch;
        height_ = tiling.height;
        width_ = tiling.width;
        depth_ = tiling.depth;
        outBatch_ = tiling.outBatch;
        outHeight_ = tiling.outHeight;
        outWidth_ = tiling.outWidth;
        cropTop_ = tiling.cropTop;
        cropLeft_ = tiling.cropLeft;
        blockSize_ = tiling.blockSize;
        totalOutputElements_ = tiling.totalOutputElements;
        tileWidth_ = tiling.tileWidth;
        tilesPerRow_ = tiling.tilesPerRow;
        totalTiles_ = tiling.totalTiles;
        tileBufferBytes_ = tiling.tileBufferBytes;
        inputGroupStrideElements_ = tiling.inputGroupStrideElements;
        inputTileBufferBytes_ = tiling.inputTileBufferBytes;
        offsetBufferBytes_ = tiling.offsetBufferBytes;
        gatherTemplateWidth_ = tiling.gatherTemplateWidth;
        rowTileHeight_ = tiling.rowTileHeight;
        flatRowMerge_ = tiling.flatRowMerge;
        linearInputOffset_ = tiling.linearInputOffset;
        usedCoreNum_ = tiling.usedCoreNum;

        const uint32_t inputElements = batch_ * height_ * width_ * depth_;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), inputElements);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), totalOutputElements_);
        if (tileBufferBytes_ != 0)
        {
            pipe_.InitBuffer(rowBuffer_, tileBufferBytes_);
        }
        if (inputTileBufferBytes_ != 0)
        {
            pipe_.InitBuffer(inputBuffer_, inputTileBufferBytes_);
        }
        if (offsetBufferBytes_ != 0)
        {
            pipe_.InitBuffer(offsetBuffer_, offsetBufferBytes_);
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId = GetBlockIdx();
        if (coreId >= usedCoreNum_)
        {
            return;
        }
        if (MODE == MODE_FLAT)
        {
            LocalTensor<DT_X> rowLocal = rowBuffer_.Get<DT_X>();
            const uint32_t pixelBytes = depth_ * static_cast<uint32_t>(sizeof(DT_X));
            if (flatRowMerge_ == 2U)
            {
                ProcessAlignedFlatChannelTiles(coreId, rowLocal);
                return;
            }
            ProcessAlignedFlatRowTiles(coreId, rowLocal, pixelBytes);
            return;
        }
        if (MODE == MODE_LINEAR)
        {
            LocalTensor<DT_X> rowLocal = rowBuffer_.Get<DT_X>();
            ProcessLinearCopy(coreId, rowLocal);
            return;
        }
        if (MODE == MODE_CHANNEL)
        {
            LocalTensor<DT_X> channelLocal = rowBuffer_.Get<DT_X>();
            ProcessChannelCopy(coreId, channelLocal);
            return;
        }
        if (MODE == MODE_ELEMENTWISE)
        {
            ProcessElementwise(coreId);
            return;
        }

        const uint32_t pixelBytes = depth_ * static_cast<uint32_t>(sizeof(DT_X));
        if (MODE == MODE_ROW)
        {
            LocalTensor<DT_X> rowLocal = rowBuffer_.Get<DT_X>();
            ProcessAlignedRowTiles(coreId, rowLocal, pixelBytes);
            return;
        }
        if (MODE == MODE_ROW_WIDE)
        {
            LocalTensor<DT_X> rowLocal = rowBuffer_.Get<DT_X>();
            ProcessAlignedRowTilesInterleavedOnly(coreId, rowLocal, pixelBytes);
            return;
        }
        if (MODE == MODE_ROW_OTHER)
        {
            LocalTensor<DT_X> rowLocal = rowBuffer_.Get<DT_X>();
            ProcessAlignedRowTilesInterleavedOnly(coreId, rowLocal, pixelBytes);
            return;
        }
        if (MODE == MODE_ALIGNED)
        {
            LocalTensor<DT_X> rowLocal = rowBuffer_.Get<DT_X>();
            ProcessAlignedTiles(coreId, rowLocal, pixelBytes);
            return;
        }
        if (MODE == MODE_ALIGNED_INTERLEAVED)
        {
            LocalTensor<DT_X> rowLocal = rowBuffer_.Get<DT_X>();
            ProcessAlignedInterleavedTiles(coreId, rowLocal, pixelBytes);
            return;
        }
        if (MODE == MODE_ALIGNED_BLOCK2_DIRECT)
        {
            LocalTensor<DT_X> rowLocal = rowBuffer_.Get<DT_X>();
            ProcessAlignedBlock2DirectTiles(coreId, rowLocal);
            return;
        }
        if (MODE == MODE_ALIGNED_BLOCK2_ROW_COMPACT)
        {
            LocalTensor<DT_X> rowLocal = rowBuffer_.Get<DT_X>();
            ProcessAlignedBlock2RowCompactTiles(coreId, rowLocal, pixelBytes);
            return;
        }
        if (MODE == MODE_ALIGNED_BLOCK2_TINY_UB_DB)
        {
            LocalTensor<DT_X> rowLocal = rowBuffer_.Get<DT_X>();
            ProcessAlignedBlock2TinyUbDoubleBufferTiles(coreId, rowLocal);
            return;
        }
        if (MODE == MODE_ALIGNED_BLOCK2_SMALL_DIRECT)
        {
            LocalTensor<DT_X> rowLocal = rowBuffer_.Get<DT_X>();
            ProcessAlignedBlock2SmallDirectTiles(coreId, rowLocal);
            return;
        }
        if (MODE == MODE_ALIGNED_BLOCK2_FULL_ROW12_MULTIROW_DIRECT)
        {
            LocalTensor<DT_X> rowLocal = rowBuffer_.Get<DT_X>();
            ProcessAlignedBlock2FullRow12MultirowDirectTiles(coreId, rowLocal);
            return;
        }
        if (MODE == MODE_ALIGNED_BLOCK4_D64_DIRECT)
        {
            LocalTensor<DT_X> rowLocal = rowBuffer_.Get<DT_X>();
            ProcessAlignedBlock4D64DirectTiles(coreId, rowLocal);
            return;
        }
        if (MODE == MODE_ALIGNED_BLOCK4_D64_PHASE_FAST)
        {
            LocalTensor<DT_X> rowLocal = rowBuffer_.Get<DT_X>();
            ProcessAlignedBlock4D64PhaseFastTiles(coreId, rowLocal, pixelBytes);
            return;
        }
        if (MODE == MODE_ALIGNED_BLOCK4_D64_UB_REORDER)
        {
            LocalTensor<DT_X> rowLocal = rowBuffer_.Get<DT_X>();
            LocalTensor<DT_X> inputLocal = inputBuffer_.Get<DT_X>();
            ProcessAlignedBlock4D64UbReorderTiles(coreId, rowLocal, inputLocal);
            return;
        }
        if (MODE == MODE_GENERAL)
        {
            LocalTensor<DT_X> rowLocal = rowBuffer_.Get<DT_X>();
            if (inputTileBufferBytes_ != 0)
            {
                BuildGroupedGatherOffsets();
            }
            ProcessGeneralTiles(coreId, rowLocal, pixelBytes);
            return;
        }
        if (MODE == MODE_GROUPED_GENERAL)
        {
            LocalTensor<DT_X> rowLocal = rowBuffer_.Get<DT_X>();
            BuildGroupedGatherOffsets();
            ProcessGroupedGeneralTiles(coreId, rowLocal, pixelBytes);
            return;
        }
        if (MODE == MODE_GROUPED_GENERAL_PACK2)
        {
            LocalTensor<DT_X> rowLocal = rowBuffer_.Get<DT_X>();
            BuildGroupedGatherOffsets();
            ProcessGroupedGeneralTiles(coreId, rowLocal, pixelBytes);
            return;
        }
        if (MODE == MODE_GENERAL_DIRECT_DMA)
        {
            LocalTensor<DT_X> rowLocal = rowBuffer_.Get<DT_X>();
            ProcessGeneralDirectDmaTiles(coreId, rowLocal, pixelBytes);
            return;
        }
    }

private:
    __aicore__ inline void ProcessAlignedTiles(
        uint32_t coreId, LocalTensor<DT_X> rowLocal, uint32_t pixelBytes)
    {
        constexpr bool alignedInterleavedMode =
            MODE == MODE_ALIGNED_INTERLEAVED;
        if (totalTiles_ <= usedCoreNum_)
        {
            const uint32_t outputRow = coreId / tilesPerRow_;
            const uint32_t tileInRow = coreId - outputRow * tilesPerRow_;
            const uint32_t ob = outputRow / outHeight_;
            const uint32_t oh = outputRow - ob * outHeight_;
            const uint32_t owBegin = tileInRow * tileWidth_;
            uint32_t currentWidth = outWidth_ - owBegin;
            if (currentWidth > tileWidth_)
            {
                currentWidth = tileWidth_;
            }
            if (!alignedInterleavedMode && blockSize_ == 1)
            {
                CopyContiguousTileAligned(rowLocal, ob, oh, owBegin, currentWidth, pixelBytes);
            }
            else
            {
                CopyInterleavedTileByDma(rowLocal, ob, oh, owBegin, currentWidth);
            }
            PipeBarrier<PIPE_ALL>();
            const uint32_t outputOffset =
                ((ob * outHeight_ + oh) * outWidth_ + owBegin) * depth_;
            const DataCopyParams outputCopy = {
                1,
                static_cast<uint16_t>(currentWidth * pixelBytes / 32),
                0,
                0};
            DataCopy(yGm_[outputOffset], rowLocal, outputCopy);
            return;
        }
        const uint32_t outputRowStart = coreId / tilesPerRow_;
        uint32_t tileInRow = coreId - outputRowStart * tilesPerRow_;
        uint32_t ob = outputRowStart / outHeight_;
        uint32_t oh = outputRowStart - ob * outHeight_;
        const uint32_t stepRows = usedCoreNum_ / tilesPerRow_;
        const uint32_t stepCols = usedCoreNum_ - stepRows * tilesPerRow_;
        const uint32_t stepOb = stepRows / outHeight_;
        const uint32_t stepOh = stepRows - stepOb * outHeight_;
        const uint32_t tileElements = tileWidth_ * depth_;
        const uint32_t singleTileBytes = tileWidth_ * pixelBytes;
        if (tileBufferBytes_ >= singleTileBytes * 2)
        {
            LocalTensor<DT_X> currentLocal = rowLocal;
            if (!alignedInterleavedMode && blockSize_ == 1)
            {
                CopyContiguousTileAligned(currentLocal, ob, oh, tileInRow * tileWidth_,
                                          ((outWidth_ - tileInRow * tileWidth_) > tileWidth_)
                                              ? tileWidth_
                                              : (outWidth_ - tileInRow * tileWidth_),
                                          pixelBytes);
            }
            else
            {
                CopyInterleavedTileByDma(currentLocal, ob, oh, tileInRow * tileWidth_,
                                         ((outWidth_ - tileInRow * tileWidth_) > tileWidth_)
                                             ? tileWidth_
                                             : (outWidth_ - tileInRow * tileWidth_));
            }
            PipeBarrier<PIPE_ALL>();
            uint32_t bufferIndex = 0;
            for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
            {
                const uint32_t owBegin = tileInRow * tileWidth_;
                uint32_t currentWidth = outWidth_ - owBegin;
                if (currentWidth > tileWidth_)
                {
                    currentWidth = tileWidth_;
                }

                const bool hasNext = tile + usedCoreNum_ < totalTiles_;
                uint32_t nextTileInRow = tileInRow;
                uint32_t nextOb = ob;
                uint32_t nextOh = oh;
                if (hasNext)
                {
                    nextTileInRow += stepCols;
                    uint32_t carryRow = 0;
                    if (nextTileInRow >= tilesPerRow_)
                    {
                        nextTileInRow -= tilesPerRow_;
                        carryRow = 1;
                    }
                    nextOb += stepOb;
                    nextOh += stepOh + carryRow;
                    if (nextOh >= outHeight_)
                    {
                        nextOh -= outHeight_;
                        ++nextOb;
                    }
                    LocalTensor<DT_X> nextLocal =
                        (bufferIndex == 0) ? rowLocal[tileElements] : rowLocal;
                    const uint32_t nextOwBegin = nextTileInRow * tileWidth_;
                    uint32_t nextWidth = outWidth_ - nextOwBegin;
                    if (nextWidth > tileWidth_)
                    {
                        nextWidth = tileWidth_;
                    }
                    if (!alignedInterleavedMode && blockSize_ == 1)
                    {
                        CopyContiguousTileAligned(
                            nextLocal, nextOb, nextOh, nextOwBegin, nextWidth, pixelBytes);
                    }
                    else
                    {
                        CopyInterleavedTileByDma(
                            nextLocal, nextOb, nextOh, nextOwBegin, nextWidth);
                    }
                }

                const uint32_t outputOffset =
                    ((ob * outHeight_ + oh) * outWidth_ + owBegin) * depth_;
                const DataCopyParams outputCopy = {
                    1,
                    static_cast<uint16_t>(currentWidth * pixelBytes / 32),
                    0,
                    0};
                DataCopy(yGm_[outputOffset], currentLocal, outputCopy);
                if (!hasNext)
                {
                    return;
                }
                PipeBarrier<PIPE_ALL>();
                tileInRow = nextTileInRow;
                ob = nextOb;
                oh = nextOh;
                bufferIndex ^= 1;
                currentLocal = (bufferIndex == 0) ? rowLocal : rowLocal[tileElements];
            }
            return;
        }
        for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
        {
            const uint32_t owBegin = tileInRow * tileWidth_;
            uint32_t currentWidth = outWidth_ - owBegin;
            if (currentWidth > tileWidth_)
            {
                currentWidth = tileWidth_;
            }

            if (!alignedInterleavedMode && blockSize_ == 1)
            {
                CopyContiguousTileAligned(rowLocal, ob, oh, owBegin, currentWidth, pixelBytes);
            }
            else
            {
                CopyInterleavedTileByDma(rowLocal, ob, oh, owBegin, currentWidth);
            }
            PipeBarrier<PIPE_ALL>();

            const uint32_t outputOffset =
                ((ob * outHeight_ + oh) * outWidth_ + owBegin) * depth_;
            const DataCopyParams outputCopy = {
                1,
                static_cast<uint16_t>(currentWidth * pixelBytes / 32),
                0,
                0};
            DataCopy(yGm_[outputOffset], rowLocal, outputCopy);
            if (tile + usedCoreNum_ < totalTiles_)
            {
                PipeBarrier<PIPE_ALL>();
            }
            tileInRow += stepCols;
            uint32_t carryRow = 0;
            if (tileInRow >= tilesPerRow_)
            {
                tileInRow -= tilesPerRow_;
                carryRow = 1;
            }
            ob += stepOb;
            oh += stepOh + carryRow;
            if (oh >= outHeight_)
            {
                oh -= outHeight_;
                ++ob;
            }
        }
    }

    __aicore__ inline void ProcessAlignedInterleavedTiles(
        uint32_t coreId, LocalTensor<DT_X> rowLocal, uint32_t pixelBytes)
    {
        if (totalTiles_ <= usedCoreNum_)
        {
            const uint32_t outputRow = coreId / tilesPerRow_;
            const uint32_t tileInRow = coreId - outputRow * tilesPerRow_;
            const uint32_t ob = outputRow / outHeight_;
            const uint32_t oh = outputRow - ob * outHeight_;
            const uint32_t owBegin = tileInRow * tileWidth_;
            uint32_t currentWidth = outWidth_ - owBegin;
            if (currentWidth > tileWidth_)
            {
                currentWidth = tileWidth_;
            }
            CopyAlignedInterleavedTileByDma(rowLocal, ob, oh, owBegin, currentWidth);
            PipeBarrier<PIPE_ALL>();
            const uint32_t outputOffset =
                ((ob * outHeight_ + oh) * outWidth_ + owBegin) * depth_;
            const DataCopyParams outputCopy = {
                1,
                static_cast<uint16_t>(currentWidth * pixelBytes / 32),
                0,
                0};
            DataCopy(yGm_[outputOffset], rowLocal, outputCopy);
            return;
        }
        const uint32_t outputRowStart = coreId / tilesPerRow_;
        uint32_t tileInRow = coreId - outputRowStart * tilesPerRow_;
        uint32_t ob = outputRowStart / outHeight_;
        uint32_t oh = outputRowStart - ob * outHeight_;
        const uint32_t stepRows = usedCoreNum_ / tilesPerRow_;
        const uint32_t stepCols = usedCoreNum_ - stepRows * tilesPerRow_;
        const uint32_t stepOb = stepRows / outHeight_;
        const uint32_t stepOh = stepRows - stepOb * outHeight_;
        const uint32_t tileElements = tileWidth_ * depth_;
        const uint32_t singleTileBytes = tileWidth_ * pixelBytes;
        if (tileBufferBytes_ >= singleTileBytes * 2)
        {
            LocalTensor<DT_X> currentLocal = rowLocal;
            CopyAlignedInterleavedTileByDma(
                currentLocal, ob, oh, tileInRow * tileWidth_,
                ((outWidth_ - tileInRow * tileWidth_) > tileWidth_)
                    ? tileWidth_
                    : (outWidth_ - tileInRow * tileWidth_));
            PipeBarrier<PIPE_ALL>();
            uint32_t bufferIndex = 0;
            for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
            {
                const uint32_t owBegin = tileInRow * tileWidth_;
                uint32_t currentWidth = outWidth_ - owBegin;
                if (currentWidth > tileWidth_)
                {
                    currentWidth = tileWidth_;
                }

                const bool hasNext = tile + usedCoreNum_ < totalTiles_;
                uint32_t nextTileInRow = tileInRow;
                uint32_t nextOb = ob;
                uint32_t nextOh = oh;
                if (hasNext)
                {
                    nextTileInRow += stepCols;
                    uint32_t carryRow = 0;
                    if (nextTileInRow >= tilesPerRow_)
                    {
                        nextTileInRow -= tilesPerRow_;
                        carryRow = 1;
                    }
                    nextOb += stepOb;
                    nextOh += stepOh + carryRow;
                    if (nextOh >= outHeight_)
                    {
                        nextOh -= outHeight_;
                        ++nextOb;
                    }
                    LocalTensor<DT_X> nextLocal =
                        (bufferIndex == 0) ? rowLocal[tileElements] : rowLocal;
                    const uint32_t nextOwBegin = nextTileInRow * tileWidth_;
                    uint32_t nextWidth = outWidth_ - nextOwBegin;
                    if (nextWidth > tileWidth_)
                    {
                        nextWidth = tileWidth_;
                    }
                    CopyAlignedInterleavedTileByDma(
                        nextLocal, nextOb, nextOh, nextOwBegin, nextWidth);
                }

                const uint32_t outputOffset =
                    ((ob * outHeight_ + oh) * outWidth_ + owBegin) * depth_;
                const DataCopyParams outputCopy = {
                    1,
                    static_cast<uint16_t>(currentWidth * pixelBytes / 32),
                    0,
                    0};
                DataCopy(yGm_[outputOffset], currentLocal, outputCopy);
                if (!hasNext)
                {
                    return;
                }
                PipeBarrier<PIPE_ALL>();
                tileInRow = nextTileInRow;
                ob = nextOb;
                oh = nextOh;
                bufferIndex ^= 1;
                currentLocal = (bufferIndex == 0) ? rowLocal : rowLocal[tileElements];
            }
            return;
        }
        for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
        {
            const uint32_t owBegin = tileInRow * tileWidth_;
            uint32_t currentWidth = outWidth_ - owBegin;
            if (currentWidth > tileWidth_)
            {
                currentWidth = tileWidth_;
            }

            CopyAlignedInterleavedTileByDma(rowLocal, ob, oh, owBegin, currentWidth);
            PipeBarrier<PIPE_ALL>();

            const uint32_t outputOffset =
                ((ob * outHeight_ + oh) * outWidth_ + owBegin) * depth_;
            const DataCopyParams outputCopy = {
                1,
                static_cast<uint16_t>(currentWidth * pixelBytes / 32),
                0,
                0};
            DataCopy(yGm_[outputOffset], rowLocal, outputCopy);
            if (tile + usedCoreNum_ < totalTiles_)
            {
                PipeBarrier<PIPE_ALL>();
            }
            tileInRow += stepCols;
            uint32_t carryRow = 0;
            if (tileInRow >= tilesPerRow_)
            {
                tileInRow -= tilesPerRow_;
                carryRow = 1;
            }
            ob += stepOb;
            oh += stepOh + carryRow;
            if (oh >= outHeight_)
            {
                oh -= outHeight_;
                ++ob;
            }
        }
    }

    __aicore__ inline void ProcessAlignedBlock2DirectTiles(
        uint32_t coreId, LocalTensor<DT_X> rowLocal)
    {
        const uint16_t copyLen =
            static_cast<uint16_t>(depth_ * static_cast<uint32_t>(sizeof(DT_X)) / 32U);
        const bool fixedBlockW = (tileWidth_ & 1U) == 0;
        if (fixedBlockW)
        {
            ProcessAlignedBlock2FixedBlockWTiles(coreId, rowLocal, copyLen);
            return;
        }
        if (totalTiles_ <= usedCoreNum_)
        {
            const uint32_t outputRow = coreId / tilesPerRow_;
            const uint32_t tileInRow = coreId - outputRow * tilesPerRow_;
            const uint32_t ob = outputRow / outHeight_;
            const uint32_t oh = outputRow - ob * outHeight_;
            const uint32_t owBegin = tileInRow * tileWidth_;
            uint32_t currentWidth = outWidth_ - owBegin;
            if (currentWidth > tileWidth_)
            {
                currentWidth = tileWidth_;
            }
            CopyBlock2TileDirectOutput(rowLocal, ob, oh, owBegin, currentWidth, copyLen);
            return;
        }
        const uint32_t outputRowStart = coreId / tilesPerRow_;
        uint32_t tileInRow = coreId - outputRowStart * tilesPerRow_;
        uint32_t ob = outputRowStart / outHeight_;
        uint32_t oh = outputRowStart - ob * outHeight_;
        const uint32_t stepRows = usedCoreNum_ / tilesPerRow_;
        const uint32_t stepCols = usedCoreNum_ - stepRows * tilesPerRow_;
        const uint32_t stepOb = stepRows / outHeight_;
        const uint32_t stepOh = stepRows - stepOb * outHeight_;
        for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
        {
            const uint32_t owBegin = tileInRow * tileWidth_;
            uint32_t currentWidth = outWidth_ - owBegin;
            if (currentWidth > tileWidth_)
            {
                currentWidth = tileWidth_;
            }
            CopyBlock2TileDirectOutput(rowLocal, ob, oh, owBegin, currentWidth, copyLen);

            tileInRow += stepCols;
            uint32_t carryRow = 0;
            if (tileInRow >= tilesPerRow_)
            {
                tileInRow -= tilesPerRow_;
                carryRow = 1;
            }
            ob += stepOb;
            oh += stepOh + carryRow;
            if (oh >= outHeight_)
            {
                oh -= outHeight_;
                ++ob;
            }
        }
    }

    __aicore__ inline void ProcessAlignedBlock4D64DirectTiles(
        uint32_t coreId, LocalTensor<DT_X> rowLocal)
    {
        const uint16_t copyLen =
            static_cast<uint16_t>(depth_ * static_cast<uint32_t>(sizeof(DT_X)) / 32U);
        if (totalTiles_ <= usedCoreNum_)
        {
            const uint32_t outputRow = coreId / tilesPerRow_;
            const uint32_t tileInRow = coreId - outputRow * tilesPerRow_;
            const uint32_t ob = outputRow / outHeight_;
            const uint32_t oh = outputRow - ob * outHeight_;
            const uint32_t owBegin = tileInRow * tileWidth_;
            uint32_t currentWidth = outWidth_ - owBegin;
            if (currentWidth > tileWidth_)
            {
                currentWidth = tileWidth_;
            }
            CopyBlock4D64TileDirectOutput(rowLocal, ob, oh, owBegin, currentWidth, copyLen);
            return;
        }
        const uint32_t outputRowStart = coreId / tilesPerRow_;
        uint32_t tileInRow = coreId - outputRowStart * tilesPerRow_;
        uint32_t ob = outputRowStart / outHeight_;
        uint32_t oh = outputRowStart - ob * outHeight_;
        const uint32_t stepRows = usedCoreNum_ / tilesPerRow_;
        const uint32_t stepCols = usedCoreNum_ - stepRows * tilesPerRow_;
        const uint32_t stepOb = stepRows / outHeight_;
        const uint32_t stepOh = stepRows - stepOb * outHeight_;
        const uint32_t tileElements = tileWidth_ * depth_;
        const uint32_t singleTileBytes =
            tileWidth_ * depth_ * static_cast<uint32_t>(sizeof(DT_X));
        if (tileBufferBytes_ >= singleTileBytes * 2U)
        {
            uint32_t owBegin = tileInRow * tileWidth_;
            uint32_t currentWidth = outWidth_ - owBegin;
            if (currentWidth > tileWidth_)
            {
                currentWidth = tileWidth_;
            }
            LocalTensor<DT_X> currentLocal = rowLocal;
            CopyBlock4D64TileDirectInput(
                currentLocal, ob, oh, owBegin, currentWidth, copyLen);
            PipeBarrier<PIPE_ALL>();

            uint32_t bufferIndex = 0;
            for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
            {
                const bool hasNext = tile + usedCoreNum_ < totalTiles_;
                uint32_t nextTileInRow = tileInRow;
                uint32_t nextOb = ob;
                uint32_t nextOh = oh;
                uint32_t nextOwBegin = 0;
                uint32_t nextWidth = 0;
                if (hasNext)
                {
                    nextTileInRow += stepCols;
                    uint32_t carryRow = 0;
                    if (nextTileInRow >= tilesPerRow_)
                    {
                        nextTileInRow -= tilesPerRow_;
                        carryRow = 1;
                    }
                    nextOb += stepOb;
                    nextOh += stepOh + carryRow;
                    if (nextOh >= outHeight_)
                    {
                        nextOh -= outHeight_;
                        ++nextOb;
                    }
                    nextOwBegin = nextTileInRow * tileWidth_;
                    nextWidth = outWidth_ - nextOwBegin;
                    if (nextWidth > tileWidth_)
                    {
                        nextWidth = tileWidth_;
                    }
                    LocalTensor<DT_X> nextLocal =
                        (bufferIndex == 0) ? rowLocal[tileElements] : rowLocal;
                    CopyBlock4D64TileDirectInput(
                        nextLocal, nextOb, nextOh, nextOwBegin, nextWidth, copyLen);
                }

                CopyBlock4D64TileDirectOutputFromUb(
                    currentLocal, ob, oh, owBegin, currentWidth, copyLen);
                if (!hasNext)
                {
                    return;
                }
                PipeBarrier<PIPE_ALL>();
                tileInRow = nextTileInRow;
                ob = nextOb;
                oh = nextOh;
                owBegin = nextOwBegin;
                currentWidth = nextWidth;
                bufferIndex ^= 1U;
                currentLocal = (bufferIndex == 0) ? rowLocal : rowLocal[tileElements];
            }
            return;
        }

        for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
        {
            const uint32_t owBegin = tileInRow * tileWidth_;
            uint32_t currentWidth = outWidth_ - owBegin;
            if (currentWidth > tileWidth_)
            {
                currentWidth = tileWidth_;
            }
            CopyBlock4D64TileDirectInput(rowLocal, ob, oh, owBegin, currentWidth, copyLen);
            PipeBarrier<PIPE_ALL>();
            CopyBlock4D64TileDirectOutputFromUb(
                rowLocal, ob, oh, owBegin, currentWidth, copyLen);
            if (tile + usedCoreNum_ < totalTiles_)
            {
                PipeBarrier<PIPE_ALL>();
            }

            tileInRow += stepCols;
            uint32_t carryRow = 0;
            if (tileInRow >= tilesPerRow_)
            {
                tileInRow -= tilesPerRow_;
                carryRow = 1;
            }
            ob += stepOb;
            oh += stepOh + carryRow;
            if (oh >= outHeight_)
            {
                oh -= outHeight_;
                ++ob;
            }
        }
    }

    __aicore__ inline void ProcessAlignedBlock4D64PhaseFastTiles(
        uint32_t coreId, LocalTensor<DT_X> rowLocal, uint32_t pixelBytes)
    {
        if (blockSize_ != 4U || depth_ != 64U || pixelBytes != 128U ||
            rowTileHeight_ != 1U || tileWidth_ != 256U || tilesPerRow_ != 6U ||
            ((cropLeft_ & 3U) != 1U) || ((cropTop_ & 3U) != 0U))
        {
            ProcessAlignedInterleavedTiles(coreId, rowLocal, pixelBytes);
            return;
        }

        const uint32_t outputRowStart = coreId / 6U;
        uint32_t tileInRow = coreId - outputRowStart * 6U;
        uint32_t ob = outputRowStart / outHeight_;
        uint32_t oh = outputRowStart - ob * outHeight_;
        const uint32_t stepRows = usedCoreNum_ / 6U;
        const uint32_t stepCols = usedCoreNum_ - stepRows * 6U;
        const uint32_t stepOb = stepRows / outHeight_;
        const uint32_t stepOh = stepRows - stepOb * outHeight_;
        const uint32_t tileElements = 256U * 64U;
        constexpr uint32_t singleTileBytes = 256U * 128U;

        if (tileBufferBytes_ >= singleTileBytes * 2U && totalTiles_ > usedCoreNum_)
        {
            uint32_t owBegin = tileInRow << 8;
            uint32_t currentWidth = outWidth_ - owBegin;
            if (currentWidth > 256U)
            {
                currentWidth = 256U;
            }
            LocalTensor<DT_X> currentLocal = rowLocal;
            CopyBlock4D64PhaseFastSingleRowInput(
                currentLocal, ob, oh, owBegin, tileInRow, currentWidth);
            PipeBarrier<PIPE_ALL>();

            uint32_t bufferIndex = 0U;
            for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
            {
                const bool hasNext = tile + usedCoreNum_ < totalTiles_;
                uint32_t nextTileInRow = tileInRow;
                uint32_t nextOb = 0U;
                uint32_t nextOh = 0U;
                uint32_t nextOwBegin = 0U;
                uint32_t nextWidth = 0U;
                if (hasNext)
                {
                    nextTileInRow += stepCols;
                    uint32_t carryRow = 0U;
                    if (nextTileInRow >= 6U)
                    {
                        nextTileInRow -= 6U;
                        carryRow = 1U;
                    }
                    nextOb = ob + stepOb;
                    nextOh = oh + stepOh + carryRow;
                    if (nextOh >= outHeight_)
                    {
                        nextOh -= outHeight_;
                        ++nextOb;
                    }
                    nextOwBegin = nextTileInRow << 8;
                    nextWidth = outWidth_ - nextOwBegin;
                    if (nextWidth > 256U)
                    {
                        nextWidth = 256U;
                    }
                    LocalTensor<DT_X> nextLocal =
                        (bufferIndex == 0U) ? rowLocal[tileElements] : rowLocal;
                    CopyBlock4D64PhaseFastSingleRowInput(
                        nextLocal, nextOb, nextOh, nextOwBegin, nextTileInRow, nextWidth);
                }

                CopyBlock4D64PhaseFastSingleRowOutput(
                    currentLocal, ob, oh, owBegin, tileInRow, currentWidth);
                if (!hasNext)
                {
                    return;
                }
                PipeBarrier<PIPE_ALL>();
                tileInRow = nextTileInRow;
                ob = nextOb;
                oh = nextOh;
                owBegin = nextOwBegin;
                currentWidth = nextWidth;
                bufferIndex ^= 1U;
                currentLocal = (bufferIndex == 0U) ? rowLocal : rowLocal[tileElements];
            }
            return;
        }

        for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
        {
            const uint32_t owBegin = tileInRow << 8;
            uint32_t currentWidth = outWidth_ - owBegin;
            if (currentWidth > 256U)
            {
                currentWidth = 256U;
            }
            CopyBlock4D64PhaseFastSingleRowInput(
                rowLocal, ob, oh, owBegin, tileInRow, currentWidth);
            PipeBarrier<PIPE_ALL>();
            CopyBlock4D64PhaseFastSingleRowOutput(
                rowLocal, ob, oh, owBegin, tileInRow, currentWidth);
            if (tile + usedCoreNum_ < totalTiles_)
            {
                PipeBarrier<PIPE_ALL>();
            }
            tileInRow += stepCols;
            uint32_t carryRow = 0U;
            if (tileInRow >= 6U)
            {
                tileInRow -= 6U;
                carryRow = 1U;
            }
            ob += stepOb;
            oh += stepOh + carryRow;
            if (oh >= outHeight_)
            {
                oh -= outHeight_;
                ++ob;
            }
        }
    }

    __aicore__ inline void ProcessAlignedBlock4D64UbReorderTiles(
        uint32_t coreId, LocalTensor<DT_X> rowLocal, LocalTensor<DT_X> inputLocal)
    {
        const uint32_t pixelBytes = depth_ * static_cast<uint32_t>(sizeof(DT_X));
        if (blockSize_ != 4U || depth_ != 64U || sizeof(DT_X) != 2U ||
            tileWidth_ == 0U || ((cropLeft_ & 3U) != 1U) ||
            ((cropTop_ & 3U) != 0U) || inputGroupStrideElements_ == 0U)
        {
            ProcessAlignedInterleavedTiles(coreId, rowLocal, pixelBytes);
            return;
        }

        const uint32_t rowTileElements = tileWidth_ * depth_;
        const uint32_t inputTileElements = 4U * inputGroupStrideElements_;
        const bool canDoubleBuffer =
            tileBufferBytes_ >= rowTileElements * static_cast<uint32_t>(sizeof(DT_X)) * 2U &&
            inputTileBufferBytes_ >= inputTileElements * static_cast<uint32_t>(sizeof(DT_X)) * 2U;

        if (totalTiles_ <= usedCoreNum_ || !canDoubleBuffer)
        {
            const uint32_t outputRowStart = coreId / tilesPerRow_;
            uint32_t tileInRow = coreId - outputRowStart * tilesPerRow_;
            uint32_t ob = outputRowStart / outHeight_;
            uint32_t oh = outputRowStart - ob * outHeight_;
            const uint32_t stepRows = usedCoreNum_ / tilesPerRow_;
            const uint32_t stepCols = usedCoreNum_ - stepRows * tilesPerRow_;
            const uint32_t stepOb = stepRows / outHeight_;
            const uint32_t stepOh = stepRows - stepOb * outHeight_;
            for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
            {
                const uint32_t owBegin = tileInRow * tileWidth_;
                uint32_t currentWidth = outWidth_ - owBegin;
                if (currentWidth > tileWidth_)
                {
                    currentWidth = tileWidth_;
                }
                CopyBlock4D64UbReorderTile(inputLocal, rowLocal, ob, oh, owBegin, currentWidth);
                PipeBarrier<PIPE_ALL>();
                CopyBlock4D64UbReorderOutput(rowLocal, ob, oh, owBegin, currentWidth);
                if (tile + usedCoreNum_ < totalTiles_)
                {
                    PipeBarrier<PIPE_ALL>();
                }
                tileInRow += stepCols;
                uint32_t carryRow = 0U;
                if (tileInRow >= tilesPerRow_)
                {
                    tileInRow -= tilesPerRow_;
                    carryRow = 1U;
                }
                ob += stepOb;
                oh += stepOh + carryRow;
                if (oh >= outHeight_)
                {
                    oh -= outHeight_;
                    ++ob;
                }
            }
            return;
        }

        const uint32_t outputRowStart = coreId / tilesPerRow_;
        uint32_t tileInRow = coreId - outputRowStart * tilesPerRow_;
        uint32_t ob = outputRowStart / outHeight_;
        uint32_t oh = outputRowStart - ob * outHeight_;
        const uint32_t stepRows = usedCoreNum_ / tilesPerRow_;
        const uint32_t stepCols = usedCoreNum_ - stepRows * tilesPerRow_;
        const uint32_t stepOb = stepRows / outHeight_;
        const uint32_t stepOh = stepRows - stepOb * outHeight_;
        uint32_t owBegin = tileInRow * tileWidth_;
        uint32_t currentWidth = outWidth_ - owBegin;
        if (currentWidth > tileWidth_)
        {
            currentWidth = tileWidth_;
        }
        LocalTensor<DT_X> currentRowLocal = rowLocal;
        CopyBlock4D64UbReorderTile(
            inputLocal, currentRowLocal, ob, oh, owBegin, currentWidth);
        PipeBarrier<PIPE_ALL>();

        uint32_t bufferIndex = 0U;
        for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
        {
            const bool hasNext = tile + usedCoreNum_ < totalTiles_;
            uint32_t nextOb = 0U;
            uint32_t nextOh = 0U;
            uint32_t nextOwBegin = 0U;
            uint32_t nextWidth = 0U;
            uint32_t nextTileInRow = tileInRow;
            if (hasNext)
            {
                nextTileInRow += stepCols;
                uint32_t carryRow = 0U;
                if (nextTileInRow >= tilesPerRow_)
                {
                    nextTileInRow -= tilesPerRow_;
                    carryRow = 1U;
                }
                nextOb = ob + stepOb;
                nextOh = oh + stepOh + carryRow;
                if (nextOh >= outHeight_)
                {
                    nextOh -= outHeight_;
                    ++nextOb;
                }
                nextOwBegin = nextTileInRow * tileWidth_;
                nextWidth = outWidth_ - nextOwBegin;
                if (nextWidth > tileWidth_)
                {
                    nextWidth = tileWidth_;
                }
                LocalTensor<DT_X> nextRowLocal =
                    (bufferIndex == 0U) ? rowLocal[rowTileElements] : rowLocal;
                LocalTensor<DT_X> nextInputLocal =
                    (bufferIndex == 0U) ? inputLocal[inputTileElements] : inputLocal;
                CopyBlock4D64UbReorderTile(
                    nextInputLocal, nextRowLocal, nextOb, nextOh, nextOwBegin, nextWidth);
            }

            CopyBlock4D64UbReorderOutput(currentRowLocal, ob, oh, owBegin, currentWidth);
            if (!hasNext)
            {
                return;
            }
            PipeBarrier<PIPE_ALL>();
            ob = nextOb;
            oh = nextOh;
            owBegin = nextOwBegin;
            currentWidth = nextWidth;
            tileInRow = nextTileInRow;
            bufferIndex ^= 1U;
            currentRowLocal = (bufferIndex == 0U) ? rowLocal : rowLocal[rowTileElements];
        }
    }

    __aicore__ inline void CopyBlock4D64UbReorderOutput(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint32_t currentWidth)
    {
        const uint32_t outputOffset =
            ((ob * outHeight_ + oh) * outWidth_ + owBegin) * 64U;
        const DataCopyParams copy = {
            1,
            static_cast<uint16_t>(currentWidth << 2),
            0,
            0};
        DataCopy(yGm_[outputOffset], rowLocal, copy);
    }

    __aicore__ inline void CopyBlock4D64UbReorderTile(
        LocalTensor<DT_X> inputLocal, LocalTensor<DT_X> rowLocal, uint32_t ob,
        uint32_t oh, uint32_t owBegin, uint32_t currentWidth)
    {
        const uint32_t spatialH = oh + cropTop_;
        const uint32_t ih = spatialH >> 2;
        const uint32_t blockH = spatialH & 3U;
        const uint32_t iw = (owBegin + cropLeft_) >> 2;
        const uint32_t ibBase = ob + outBatch_ * (blockH << 2);
        const uint32_t batchStride = height_ * width_ * 64U;
        const uint32_t outBatchStride = outBatch_ * batchStride;
        const uint32_t rowOffset = (ih * width_ + iw) * 64U;
        const uint32_t inputOffsetBase = ibBase * batchStride + rowOffset;
        if (currentWidth == tileWidth_)
        {
            const uint32_t count = tileWidth_ >> 2;
            CopyBlock4D64UbReorderInputPhaseFull(
                inputLocal, 0U, inputOffsetBase + outBatchStride, count);
            CopyBlock4D64UbReorderInputPhaseFull(
                inputLocal, 1U, inputOffsetBase + (outBatchStride << 1), count);
            CopyBlock4D64UbReorderInputPhaseFull(
                inputLocal, 2U, inputOffsetBase + outBatchStride * 3U, count);
            CopyBlock4D64UbReorderInputPhaseFull(
                inputLocal, 3U, inputOffsetBase + 64U, count);
            PipeBarrier<PIPE_ALL>();

            CopyBlock4D64UbReorderLocalPhaseFull(inputLocal, rowLocal, 0U, count);
            CopyBlock4D64UbReorderLocalPhaseFull(inputLocal, rowLocal, 1U, count);
            CopyBlock4D64UbReorderLocalPhaseFull(inputLocal, rowLocal, 2U, count);
            CopyBlock4D64UbReorderLocalPhaseFull(inputLocal, rowLocal, 3U, count);
            return;
        }

        const uint32_t count0 = (currentWidth + 3U) >> 2;
        const uint32_t count1 = (currentWidth + 2U) >> 2;
        const uint32_t count2 = (currentWidth + 1U) >> 2;
        const uint32_t count3 = currentWidth >> 2;

        CopyBlock4D64UbReorderInputPhase(
            inputLocal, 0U, inputOffsetBase + outBatchStride, count0);
        CopyBlock4D64UbReorderInputPhase(
            inputLocal, 1U, inputOffsetBase + (outBatchStride << 1), count1);
        CopyBlock4D64UbReorderInputPhase(
            inputLocal, 2U, inputOffsetBase + outBatchStride * 3U, count2);
        CopyBlock4D64UbReorderInputPhase(
            inputLocal, 3U, inputOffsetBase + 64U, count3);
        PipeBarrier<PIPE_ALL>();

        CopyBlock4D64UbReorderLocalPhase(inputLocal, rowLocal, 0U, count0);
        CopyBlock4D64UbReorderLocalPhase(inputLocal, rowLocal, 1U, count1);
        CopyBlock4D64UbReorderLocalPhase(inputLocal, rowLocal, 2U, count2);
        CopyBlock4D64UbReorderLocalPhase(inputLocal, rowLocal, 3U, count3);
    }

    __aicore__ inline void CopyBlock4D64UbReorderInputPhaseFull(
        LocalTensor<DT_X> inputLocal, uint32_t phase, uint32_t inputOffset,
        uint32_t count)
    {
        const DataCopyParams copy = {
            1,
            static_cast<uint16_t>(count << 2),
            0,
            0};
        DataCopy(inputLocal[phase * inputGroupStrideElements_], xGm_[inputOffset], copy);
    }

    __aicore__ inline void CopyBlock4D64UbReorderInputPhase(
        LocalTensor<DT_X> inputLocal, uint32_t phase, uint32_t inputOffset,
        uint32_t count)
    {
        if (count == 0U)
        {
            return;
        }
        const DataCopyParams copy = {
            1,
            static_cast<uint16_t>(count << 2),
            0,
            0};
        DataCopy(inputLocal[phase * inputGroupStrideElements_], xGm_[inputOffset], copy);
    }

    __aicore__ inline void CopyBlock4D64UbReorderLocalPhase(
        LocalTensor<DT_X> inputLocal, LocalTensor<DT_X> rowLocal, uint32_t phase,
        uint32_t count)
    {
        if (count == 0U)
        {
            return;
        }
        const DataCopyParams copy = {
            static_cast<uint16_t>(count),
            4,
            0,
            12};
        DataCopy(rowLocal[phase * 64U], inputLocal[phase * inputGroupStrideElements_], copy);
    }

    __aicore__ inline void CopyBlock4D64UbReorderLocalPhaseFull(
        LocalTensor<DT_X> inputLocal, LocalTensor<DT_X> rowLocal, uint32_t phase,
        uint32_t count)
    {
        const DataCopyParams copy = {
            static_cast<uint16_t>(count),
            4,
            0,
            12};
        DataCopy(rowLocal[phase * 64U], inputLocal[phase * inputGroupStrideElements_], copy);
    }

    __aicore__ inline void CopyBlock4D64PhaseFastSingleRowInput(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint32_t tileInRow, uint32_t currentWidth)
    {
        if (tileInRow < 5U || currentWidth == 256U)
        {
            CopyBlock4D64Phase1FullTileByDma(rowLocal, ob, oh, owBegin);
            return;
        }
        CopyBlock4D64Phase1TileByDma(rowLocal, ob, oh, owBegin, currentWidth);
    }

    __aicore__ inline void CopyBlock4D64PhaseFastSingleRowOutput(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint32_t tileInRow, uint32_t currentWidth)
    {
        if (tileInRow < 5U || currentWidth == 256U)
        {
            const uint32_t outputOffset =
                ((ob * outHeight_ + oh) * outWidth_ + owBegin) * 64U;
            const DataCopyParams outputCopy = {1, 1024, 0, 0};
            DataCopy(yGm_[outputOffset], rowLocal, outputCopy);
            return;
        }
        CopyBlock4D64PhaseFastOutput(rowLocal, ob, oh, owBegin, 1U, currentWidth, 128U);
    }

    __aicore__ inline void DecodeBlock4D64PhaseFastTile(
        uint32_t tile, uint32_t &ob, uint32_t &ohBegin, uint32_t &owBegin,
        uint32_t &currentRows, uint32_t &currentWidth)
    {
        const uint32_t rowTilesPerBatch =
            (outHeight_ + rowTileHeight_ - 1U) / rowTileHeight_;
        const uint32_t tilesPerBatch = rowTilesPerBatch * tilesPerRow_;
        ob = tile / tilesPerBatch;
        const uint32_t tileInBatch = tile - ob * tilesPerBatch;
        const uint32_t rowTile = tileInBatch / tilesPerRow_;
        const uint32_t tileInRow = tileInBatch - rowTile * tilesPerRow_;
        ohBegin = rowTile * rowTileHeight_;
        currentRows = outHeight_ - ohBegin;
        if (currentRows > rowTileHeight_)
        {
            currentRows = rowTileHeight_;
        }
        owBegin = tileInRow * tileWidth_;
        currentWidth = outWidth_ - owBegin;
        if (currentWidth > tileWidth_)
        {
            currentWidth = tileWidth_;
        }
    }

    __aicore__ inline void CopyBlock4D64PhaseFastInput(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t ohBegin, uint32_t owBegin,
        uint32_t currentRows, uint32_t currentWidth)
    {
        const uint32_t rowElements = tileWidth_ * depth_;
        for (uint32_t row = 0; row < currentRows; ++row)
        {
            CopyBlock4D64Phase1TileByDma(
                rowLocal[row * rowElements], ob, ohBegin + row, owBegin, currentWidth);
        }
    }

    __aicore__ inline void CopyBlock4D64PhaseFastOutput(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t ohBegin, uint32_t owBegin,
        uint32_t currentRows, uint32_t currentWidth, uint32_t pixelBytes)
    {
        const uint32_t outputOffset =
            ((ob * outHeight_ + ohBegin) * outWidth_ + owBegin) * depth_;
        const DataCopyParams outputCopy = {
            static_cast<uint16_t>(currentRows),
            static_cast<uint16_t>(currentWidth * pixelBytes / 32U),
            static_cast<uint16_t>((tileWidth_ - currentWidth) * pixelBytes / 32U),
            static_cast<uint16_t>((outWidth_ - currentWidth) * pixelBytes / 32U)};
        DataCopy(yGm_[outputOffset], rowLocal, outputCopy);
    }

    __aicore__ inline void ProcessAlignedBlock2FixedBlockWTiles(
        uint32_t coreId, LocalTensor<DT_X> rowLocal, uint16_t copyLen)
    {
        const uint32_t blockW = cropLeft_ & 1U;
        if (totalTiles_ <= usedCoreNum_)
        {
            const uint32_t outputRow = coreId / tilesPerRow_;
            const uint32_t tileInRow = coreId - outputRow * tilesPerRow_;
            const uint32_t ob = outputRow / outHeight_;
            const uint32_t oh = outputRow - ob * outHeight_;
            const uint32_t owBegin = tileInRow * tileWidth_;
            uint32_t currentWidth = outWidth_ - owBegin;
            if (currentWidth > tileWidth_)
            {
                currentWidth = tileWidth_;
            }
            CopyBlock2TileDirectOutputFixedBlockW(
                rowLocal, ob, oh, owBegin, currentWidth, copyLen, blockW);
            return;
        }
        const uint32_t outputRowStart = coreId / tilesPerRow_;
        uint32_t tileInRow = coreId - outputRowStart * tilesPerRow_;
        uint32_t ob = outputRowStart / outHeight_;
        uint32_t oh = outputRowStart - ob * outHeight_;
        const uint32_t stepRows = usedCoreNum_ / tilesPerRow_;
        const uint32_t stepCols = usedCoreNum_ - stepRows * tilesPerRow_;
        const uint32_t stepOb = stepRows / outHeight_;
        const uint32_t stepOh = stepRows - stepOb * outHeight_;
        const uint32_t tileElements = tileWidth_ * depth_;
        const uint32_t singleTileBytes =
            tileWidth_ * depth_ * static_cast<uint32_t>(sizeof(DT_X));
        if (tileBufferBytes_ >= singleTileBytes * 2U)
        {
            uint32_t owBegin = tileInRow * tileWidth_;
            uint32_t currentWidth = outWidth_ - owBegin;
            if (currentWidth > tileWidth_)
            {
                currentWidth = tileWidth_;
            }
            LocalTensor<DT_X> currentLocal = rowLocal;
            CopyBlock2TileDirectInputFixedBlockW(
                currentLocal, ob, oh, owBegin, currentWidth, copyLen, blockW);
            PipeBarrier<PIPE_ALL>();

            uint32_t bufferIndex = 0;
            for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
            {
                const bool hasNext = tile + usedCoreNum_ < totalTiles_;
                uint32_t nextTileInRow = tileInRow;
                uint32_t nextOb = ob;
                uint32_t nextOh = oh;
                uint32_t nextOwBegin = 0;
                uint32_t nextWidth = 0;
                if (hasNext)
                {
                    nextTileInRow += stepCols;
                    uint32_t carryRow = 0;
                    if (nextTileInRow >= tilesPerRow_)
                    {
                        nextTileInRow -= tilesPerRow_;
                        carryRow = 1;
                    }
                    nextOb += stepOb;
                    nextOh += stepOh + carryRow;
                    if (nextOh >= outHeight_)
                    {
                        nextOh -= outHeight_;
                        ++nextOb;
                    }
                    nextOwBegin = nextTileInRow * tileWidth_;
                    nextWidth = outWidth_ - nextOwBegin;
                    if (nextWidth > tileWidth_)
                    {
                        nextWidth = tileWidth_;
                    }
                    LocalTensor<DT_X> nextLocal =
                        (bufferIndex == 0) ? rowLocal[tileElements] : rowLocal;
                    CopyBlock2TileDirectInputFixedBlockW(
                        nextLocal, nextOb, nextOh, nextOwBegin, nextWidth, copyLen, blockW);
                }

                CopyBlock2TileDirectOutputFromUb(
                    currentLocal, ob, oh, owBegin, currentWidth, copyLen);
                if (!hasNext)
                {
                    return;
                }
                PipeBarrier<PIPE_ALL>();
                tileInRow = nextTileInRow;
                ob = nextOb;
                oh = nextOh;
                owBegin = nextOwBegin;
                currentWidth = nextWidth;
                bufferIndex ^= 1U;
                currentLocal = (bufferIndex == 0) ? rowLocal : rowLocal[tileElements];
            }
            return;
        }
        for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
        {
            const uint32_t owBegin = tileInRow * tileWidth_;
            uint32_t currentWidth = outWidth_ - owBegin;
            if (currentWidth > tileWidth_)
            {
                currentWidth = tileWidth_;
            }
            CopyBlock2TileDirectOutputFixedBlockW(
                rowLocal, ob, oh, owBegin, currentWidth, copyLen, blockW);

            tileInRow += stepCols;
            uint32_t carryRow = 0;
            if (tileInRow >= tilesPerRow_)
            {
                tileInRow -= tilesPerRow_;
                carryRow = 1;
            }
            ob += stepOb;
            oh += stepOh + carryRow;
            if (oh >= outHeight_)
            {
                oh -= outHeight_;
                ++ob;
            }
        }
    }

    __aicore__ inline void ProcessAlignedBlock2SmallDirectTiles(
        uint32_t coreId, LocalTensor<DT_X> rowLocal)
    {
        const uint16_t copyLen =
            static_cast<uint16_t>(depth_ * static_cast<uint32_t>(sizeof(DT_X)) / 32U);
        const uint32_t blockW = cropLeft_ & 1U;
        const uint32_t outputRowStart = coreId / tilesPerRow_;
        uint32_t tileInRow = coreId - outputRowStart * tilesPerRow_;
        uint32_t ob = outputRowStart / outHeight_;
        uint32_t oh = outputRowStart - ob * outHeight_;
        const uint32_t stepRows = usedCoreNum_ / tilesPerRow_;
        const uint32_t stepCols = usedCoreNum_ - stepRows * tilesPerRow_;
        const uint32_t stepOb = stepRows / outHeight_;
        const uint32_t stepOh = stepRows - stepOb * outHeight_;

        for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
        {
            const uint32_t owBegin = tileInRow * tileWidth_;
            uint32_t currentWidth = outWidth_ - owBegin;
            if (currentWidth > tileWidth_)
            {
                currentWidth = tileWidth_;
            }
            if (currentWidth == tileWidth_ && (tileWidth_ == 10U || tileWidth_ == 12U))
            {
                CopyBlock2SmallDirectFullTile(rowLocal, ob, oh, owBegin, copyLen, blockW);
            }
            else
            {
                CopyBlock2SmallDirectTailTile(
                    rowLocal, ob, oh, owBegin, currentWidth, copyLen);
            }

            tileInRow += stepCols;
            uint32_t carryRow = 0;
            if (tileInRow >= tilesPerRow_)
            {
                tileInRow -= tilesPerRow_;
                carryRow = 1;
            }
            ob += stepOb;
            oh += stepOh + carryRow;
            if (oh >= outHeight_)
            {
                oh -= outHeight_;
                ++ob;
            }
        }
    }

    __aicore__ inline void ProcessAlignedBlock2FullRow12MultirowDirectTiles(
        uint32_t coreId, LocalTensor<DT_X> rowLocal)
    {
        if (rowTileHeight_ == 4U)
        {
            ProcessAlignedBlock2FullRow12FourRowDirectTiles(coreId, rowLocal);
            return;
        }
        const uint16_t copyLen =
            static_cast<uint16_t>(depth_ * static_cast<uint32_t>(sizeof(DT_X)) / 32U);
        const uint32_t halfRowElements = 6U * depth_;
        const uint32_t groupElements = rowTileHeight_ * halfRowElements;
        for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
        {
            const uint32_t ob = tile / tilesPerRow_;
            const uint32_t rowTile = tile - ob * tilesPerRow_;
            const uint32_t ohBegin = rowTile * rowTileHeight_;
            uint32_t currentRows = outHeight_ - ohBegin;
            if (currentRows > rowTileHeight_)
            {
                currentRows = rowTileHeight_;
            }
            for (uint32_t row = 0; row < currentRows; ++row)
            {
                CopyBlock2FullRow12MultirowInput(
                    rowLocal, groupElements, row, ob, ohBegin + row, copyLen);
            }
            PipeBarrier<PIPE_ALL>();

            const uint32_t outputOffset =
                (ob * outHeight_ + ohBegin) * outWidth_ * depth_;
            const DataCopyParams outputCopy = {
                static_cast<uint16_t>(currentRows * 6U),
                copyLen,
                0,
                copyLen};
            DataCopy(yGm_[outputOffset], rowLocal, outputCopy);
            DataCopy(yGm_[outputOffset + depth_], rowLocal[groupElements], outputCopy);
            if (tile + usedCoreNum_ < totalTiles_)
            {
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void ProcessAlignedBlock2FullRow12FourRowDirectTiles(
        uint32_t coreId, LocalTensor<DT_X> rowLocal)
    {
        const uint16_t copyLen =
            static_cast<uint16_t>(depth_ * static_cast<uint32_t>(sizeof(DT_X)) / 32U);
        const uint32_t groupElements = 24U * depth_;
        const uint32_t tileElements = 48U * depth_;
        if (tileBufferBytes_ >=
                tileElements * static_cast<uint32_t>(sizeof(DT_X)) * 2U &&
            totalTiles_ > usedCoreNum_)
        {
            uint32_t ob = 0;
            uint32_t ohBegin = 0;
            uint32_t currentRows = 0;
            DecodeFullRow12FourRowTile(coreId, ob, ohBegin, currentRows);
            LocalTensor<DT_X> currentLocal = rowLocal;
            CopyBlock2FullRow12FourRowInput(
                currentLocal, groupElements, ob, ohBegin, currentRows, copyLen);
            PipeBarrier<PIPE_ALL>();

            uint32_t bufferIndex = 0;
            for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
            {
                const bool hasNext = tile + usedCoreNum_ < totalTiles_;
                uint32_t nextOb = 0;
                uint32_t nextOhBegin = 0;
                uint32_t nextRows = 0;
                if (hasNext)
                {
                    DecodeFullRow12FourRowTile(
                        tile + usedCoreNum_, nextOb, nextOhBegin, nextRows);
                    LocalTensor<DT_X> nextLocal =
                        (bufferIndex == 0) ? rowLocal[tileElements] : rowLocal;
                    CopyBlock2FullRow12FourRowInput(
                        nextLocal, groupElements, nextOb, nextOhBegin, nextRows, copyLen);
                }
                CopyBlock2FullRow12Output(
                    currentLocal, groupElements, ob, ohBegin, currentRows, copyLen);
                if (!hasNext)
                {
                    return;
                }
                PipeBarrier<PIPE_ALL>();
                ob = nextOb;
                ohBegin = nextOhBegin;
                currentRows = nextRows;
                bufferIndex ^= 1U;
                currentLocal = (bufferIndex == 0) ? rowLocal : rowLocal[tileElements];
            }
            return;
        }

        for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
        {
            uint32_t ob = 0;
            uint32_t ohBegin = 0;
            uint32_t currentRows = 0;
            DecodeFullRow12FourRowTile(tile, ob, ohBegin, currentRows);
            CopyBlock2FullRow12FourRowInput(
                rowLocal, groupElements, ob, ohBegin, currentRows, copyLen);
            PipeBarrier<PIPE_ALL>();
            CopyBlock2FullRow12Output(
                rowLocal, groupElements, ob, ohBegin, currentRows, copyLen);
            if (tile + usedCoreNum_ < totalTiles_)
            {
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void ProcessAlignedBlock2RowCompactTiles(
        uint32_t coreId, LocalTensor<DT_X> rowLocal, uint32_t pixelBytes)
    {
        const uint32_t rowElements = outWidth_ * depth_;
        const uint16_t rowCopyBlocks = static_cast<uint16_t>(outWidth_ * pixelBytes / 32U);
        for (uint32_t rowIndex = coreId; rowIndex < totalTiles_; rowIndex += usedCoreNum_)
        {
            const uint32_t ob = rowIndex / outHeight_;
            const uint32_t oh = rowIndex - ob * outHeight_;
            CopyInterleavedFullRowByDmaBlock2(rowLocal, ob, oh, pixelBytes);
            PipeBarrier<PIPE_ALL>();

            const DataCopyParams outputCopy = {
                1,
                rowCopyBlocks,
                0,
                0};
            DataCopy(yGm_[rowIndex * rowElements], rowLocal, outputCopy);
            if (rowIndex + usedCoreNum_ < totalTiles_)
            {
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void ProcessAlignedBlock2TinyUbDoubleBufferTiles(
        uint32_t coreId, LocalTensor<DT_X> rowLocal)
    {
        const uint16_t copyLen =
            static_cast<uint16_t>(depth_ * static_cast<uint32_t>(sizeof(DT_X)) / 32U);
        if (totalTiles_ <= usedCoreNum_)
        {
            const uint32_t outputRow = coreId / tilesPerRow_;
            const uint32_t tileInRow = coreId - outputRow * tilesPerRow_;
            const uint32_t ob = outputRow / outHeight_;
            const uint32_t oh = outputRow - ob * outHeight_;
            const uint32_t owBegin = tileInRow * tileWidth_;
            uint32_t currentWidth = outWidth_ - owBegin;
            if (currentWidth > tileWidth_)
            {
                currentWidth = tileWidth_;
            }
            CopyBlock2TileViaUb(rowLocal, ob, oh, owBegin, currentWidth, copyLen);
            return;
        }

        const uint32_t outputRowStart = coreId / tilesPerRow_;
        uint32_t tileInRow = coreId - outputRowStart * tilesPerRow_;
        uint32_t ob = outputRowStart / outHeight_;
        uint32_t oh = outputRowStart - ob * outHeight_;
        const uint32_t stepRows = usedCoreNum_ / tilesPerRow_;
        const uint32_t stepCols = usedCoreNum_ - stepRows * tilesPerRow_;
        const uint32_t stepOb = stepRows / outHeight_;
        const uint32_t stepOh = stepRows - stepOb * outHeight_;
        const uint32_t tileElements = tileWidth_ * depth_;
        const uint32_t singleTileBytes =
            tileWidth_ * depth_ * static_cast<uint32_t>(sizeof(DT_X));
        if (tileBufferBytes_ >= singleTileBytes * 2U)
        {
            uint32_t owBegin = tileInRow * tileWidth_;
            uint32_t currentWidth = outWidth_ - owBegin;
            if (currentWidth > tileWidth_)
            {
                currentWidth = tileWidth_;
            }
            LocalTensor<DT_X> currentLocal = rowLocal;
            CopyInterleavedTileByDmaBlock2(currentLocal, ob, oh, owBegin, currentWidth);
            PipeBarrier<PIPE_ALL>();

            uint32_t bufferIndex = 0;
            for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
            {
                const bool hasNext = tile + usedCoreNum_ < totalTiles_;
                uint32_t nextTileInRow = tileInRow;
                uint32_t nextOb = ob;
                uint32_t nextOh = oh;
                uint32_t nextOwBegin = 0;
                uint32_t nextWidth = 0;
                if (hasNext)
                {
                    nextTileInRow += stepCols;
                    uint32_t carryRow = 0;
                    if (nextTileInRow >= tilesPerRow_)
                    {
                        nextTileInRow -= tilesPerRow_;
                        carryRow = 1;
                    }
                    nextOb += stepOb;
                    nextOh += stepOh + carryRow;
                    if (nextOh >= outHeight_)
                    {
                        nextOh -= outHeight_;
                        ++nextOb;
                    }
                    nextOwBegin = nextTileInRow * tileWidth_;
                    nextWidth = outWidth_ - nextOwBegin;
                    if (nextWidth > tileWidth_)
                    {
                        nextWidth = tileWidth_;
                    }
                    LocalTensor<DT_X> nextLocal =
                        (bufferIndex == 0) ? rowLocal[tileElements] : rowLocal;
                    CopyInterleavedTileByDmaBlock2(
                        nextLocal, nextOb, nextOh, nextOwBegin, nextWidth);
                }

                CopyBlock2TileViaUbOutput(currentLocal, ob, oh, owBegin, currentWidth, copyLen);
                if (!hasNext)
                {
                    return;
                }
                PipeBarrier<PIPE_ALL>();
                tileInRow = nextTileInRow;
                ob = nextOb;
                oh = nextOh;
                owBegin = nextOwBegin;
                currentWidth = nextWidth;
                bufferIndex ^= 1U;
                currentLocal = (bufferIndex == 0) ? rowLocal : rowLocal[tileElements];
            }
            return;
        }

        for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
        {
            const uint32_t owBegin = tileInRow * tileWidth_;
            uint32_t currentWidth = outWidth_ - owBegin;
            if (currentWidth > tileWidth_)
            {
                currentWidth = tileWidth_;
            }
            CopyBlock2TileViaUb(rowLocal, ob, oh, owBegin, currentWidth, copyLen);

            tileInRow += stepCols;
            uint32_t carryRow = 0;
            if (tileInRow >= tilesPerRow_)
            {
                tileInRow -= tilesPerRow_;
                carryRow = 1;
            }
            ob += stepOb;
            oh += stepOh + carryRow;
            if (oh >= outHeight_)
            {
                oh -= outHeight_;
                ++ob;
            }
        }
    }

    __aicore__ inline void ProcessGeneralTiles(
        uint32_t coreId, LocalTensor<DT_X> rowLocal, uint32_t pixelBytes)
    {
        if (rowTileHeight_ != 0 && blockSize_ > 1 && tileWidth_ == outWidth_)
        {
            ProcessGeneralRowTiles(coreId, rowLocal, pixelBytes);
            return;
        }
        if (totalTiles_ <= usedCoreNum_)
        {
            const uint32_t outputRow = coreId / tilesPerRow_;
            const uint32_t tileInRow = coreId - outputRow * tilesPerRow_;
            const uint32_t ob = outputRow / outHeight_;
            const uint32_t oh = outputRow - ob * outHeight_;
            const uint32_t owBegin = tileInRow * tileWidth_;
            uint32_t currentWidth = outWidth_ - owBegin;
            if (currentWidth > tileWidth_)
            {
                currentWidth = tileWidth_;
            }
            if (blockSize_ == 1)
            {
                CopyContiguousTilePad(rowLocal, ob, oh, owBegin, currentWidth, pixelBytes);
            }
            else if (inputTileBufferBytes_ != 0)
            {
                LocalTensor<DT_X> inputLocal = inputBuffer_.Get<DT_X>();
                CopyGroupedInterleavedTileByDma(
                    inputLocal, rowLocal, ob, oh, owBegin, currentWidth);
            }
            else
            {
                GatherInterleavedTile(rowLocal, ob, oh, owBegin, currentWidth);
            }
            PipeBarrier<PIPE_ALL>();
            const uint32_t outputOffset =
                ((ob * outHeight_ + oh) * outWidth_ + owBegin) * depth_;
            const uint32_t outputBytes = currentWidth * pixelBytes;
            const uint32_t outputOffsetBytes =
                outputOffset * static_cast<uint32_t>(sizeof(DT_X));
            if (outputOffsetBytes % 32 == 0 && outputBytes % 32 == 0)
            {
                const DataCopyParams outputCopy = {
                    1,
                    static_cast<uint16_t>(outputBytes / 32),
                    0,
                    0};
                DataCopy(yGm_[outputOffset], rowLocal, outputCopy);
            }
            else
            {
                const DataCopyExtParams outputCopy = {
                    1, outputBytes, 0, 0, 0};
                DataCopyPad(yGm_[outputOffset], rowLocal, outputCopy);
            }
            return;
        }
        const uint32_t outputRowStart = coreId / tilesPerRow_;
        uint32_t tileInRow = coreId - outputRowStart * tilesPerRow_;
        uint32_t ob = outputRowStart / outHeight_;
        uint32_t oh = outputRowStart - ob * outHeight_;
        const uint32_t stepRows = usedCoreNum_ / tilesPerRow_;
        const uint32_t stepCols = usedCoreNum_ - stepRows * tilesPerRow_;
        const uint32_t stepOb = stepRows / outHeight_;
        const uint32_t stepOh = stepRows - stepOb * outHeight_;
        for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
        {
            const uint32_t owBegin = tileInRow * tileWidth_;
            uint32_t currentWidth = outWidth_ - owBegin;
            if (currentWidth > tileWidth_)
            {
                currentWidth = tileWidth_;
            }

            if (blockSize_ == 1)
            {
                CopyContiguousTilePad(rowLocal, ob, oh, owBegin, currentWidth, pixelBytes);
            }
            else if (inputTileBufferBytes_ != 0)
            {
                LocalTensor<DT_X> inputLocal = inputBuffer_.Get<DT_X>();
                CopyGroupedInterleavedTileByDma(
                    inputLocal, rowLocal, ob, oh, owBegin, currentWidth);
            }
            else
            {
                GatherInterleavedTile(rowLocal, ob, oh, owBegin, currentWidth);
            }
            PipeBarrier<PIPE_ALL>();

            const uint32_t outputOffset =
                ((ob * outHeight_ + oh) * outWidth_ + owBegin) * depth_;
            const uint32_t outputBytes = currentWidth * pixelBytes;
            const uint32_t outputOffsetBytes =
                outputOffset * static_cast<uint32_t>(sizeof(DT_X));
            if (outputOffsetBytes % 32 == 0 && outputBytes % 32 == 0)
            {
                const DataCopyParams outputCopy = {
                    1,
                    static_cast<uint16_t>(outputBytes / 32),
                    0,
                    0};
                DataCopy(yGm_[outputOffset], rowLocal, outputCopy);
            }
            else
            {
                const DataCopyExtParams outputCopy = {
                    1, outputBytes, 0, 0, 0};
                DataCopyPad(yGm_[outputOffset], rowLocal, outputCopy);
            }
            if (tile + usedCoreNum_ < totalTiles_)
            {
                PipeBarrier<PIPE_ALL>();
            }
            tileInRow += stepCols;
            uint32_t carryRow = 0;
            if (tileInRow >= tilesPerRow_)
            {
                tileInRow -= tilesPerRow_;
                carryRow = 1;
            }
            ob += stepOb;
            oh += stepOh + carryRow;
            if (oh >= outHeight_)
            {
                oh -= outHeight_;
                ++ob;
            }
        }
    }

    __aicore__ inline void ProcessGeneralRowTiles(
        uint32_t coreId, LocalTensor<DT_X> rowLocal, uint32_t pixelBytes)
    {
        const uint32_t rowElements = outWidth_ * depth_;
        const uint32_t bytesPerElement = static_cast<uint32_t>(sizeof(DT_X));
        const uint32_t rowBytes = outWidth_ * pixelBytes;
        const uint32_t rowStrideElements =
            ((rowBytes + 31U) / 32U) * (32U / bytesPerElement);
        for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
        {
            uint32_t ob = 0;
            uint32_t ohBegin = 0;
            uint32_t currentRows = 0;
            DecodeGeneralRowTile(tile, ob, ohBegin, currentRows);
            if (inputTileBufferBytes_ != 0)
            {
                LocalTensor<DT_X> inputLocal = inputBuffer_.Get<DT_X>();
                const bool useSingleGather = gatherTemplateWidth_ >= outWidth_;
                for (uint32_t row = 0; row < currentRows; ++row)
                {
                    CopyGroupedInterleavedInputByDma(inputLocal, ob, ohBegin + row, 0, outWidth_);
                    PipeBarrier<PIPE_ALL>();
                    GatherGroupedInput(
                        inputLocal, rowLocal[row * rowStrideElements], outWidth_, pixelBytes,
                        useSingleGather);
                    if (row + 1U < currentRows)
                    {
                        PipeBarrier<PIPE_ALL>();
                    }
                }
            }
            else
            {
                for (uint32_t row = 0; row < currentRows; ++row)
                {
                    GatherInterleavedTile(
                        rowLocal[row * rowStrideElements], ob, ohBegin + row, 0, outWidth_);
                }
            }
            PipeBarrier<PIPE_ALL>();
            if (rowStrideElements == rowElements)
            {
                CopyGeneralRowsOutput(rowLocal, ob, ohBegin, currentRows, pixelBytes);
            }
            else
            {
                CopyGeneralPaddedRowsOutput(
                    rowLocal, rowStrideElements, ob, ohBegin, currentRows, pixelBytes);
            }
            if (tile + usedCoreNum_ < totalTiles_)
            {
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void ProcessGroupedGeneralTiles(
        uint32_t coreId, LocalTensor<DT_X> rowLocal, uint32_t pixelBytes)
    {
        if (BTS_OPT_BLOCK2_DEEP_DIRECT_LOCAL_GATHER &&
            blockSize_ == 2U && sizeof(DT_X) == 2U &&
            pixelBytes > 128U && outWidth_ > 64U && outWidth_ <= 256U &&
            outHeight_ > 64U)
        {
            ProcessBlock2DeepDirectLocalTiles(coreId, rowLocal, pixelBytes);
            return;
        }
        LocalTensor<DT_X> inputLocal = inputBuffer_.Get<DT_X>();
        const uint32_t bytesPerElement = static_cast<uint32_t>(sizeof(DT_X));
        const uint32_t rowTileBytes =
            ((tileWidth_ * pixelBytes + 31U) / 32U) * 32U;
        const uint32_t rowTileElements = rowTileBytes / bytesPerElement;
        const uint32_t groupCount =
            (tileWidth_ < blockSize_) ? tileWidth_ : blockSize_;
        const uint32_t inputTileElements = groupCount * inputGroupStrideElements_;
        const uint32_t inputTileBytes = inputTileElements * bytesPerElement;
        const bool canDoubleBuffer =
            tileBufferBytes_ >= rowTileBytes * 2U &&
            inputTileBufferBytes_ >= inputTileBytes * 2U;
        const bool useSingleGather = gatherTemplateWidth_ >= tileWidth_;

        if (totalTiles_ <= usedCoreNum_ || !canDoubleBuffer)
        {
            for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
            {
                uint32_t ob = 0;
                uint32_t oh = 0;
                uint32_t owBegin = 0;
                uint32_t currentWidth = 0;
                DecodeGeneralTile(tile, ob, oh, owBegin, currentWidth);
                CopyGroupedInterleavedTileByDma(
                    inputLocal, rowLocal, ob, oh, owBegin, currentWidth);
                PipeBarrier<PIPE_ALL>();
                CopyGeneralOutput(rowLocal, ob, oh, owBegin, currentWidth, pixelBytes);
                if (tile + usedCoreNum_ < totalTiles_)
                {
                    PipeBarrier<PIPE_ALL>();
                }
            }
            return;
        }

        uint32_t ob = 0;
        uint32_t oh = 0;
        uint32_t owBegin = 0;
        uint32_t currentWidth = 0;
        const uint32_t outputRowStart = coreId / tilesPerRow_;
        uint32_t tileInRow = coreId - outputRowStart * tilesPerRow_;
        ob = outputRowStart / outHeight_;
        oh = outputRowStart - ob * outHeight_;
        owBegin = tileInRow * tileWidth_;
        currentWidth = outWidth_ - owBegin;
        if (currentWidth > tileWidth_)
        {
            currentWidth = tileWidth_;
        }
        const uint32_t stepRows = usedCoreNum_ / tilesPerRow_;
        const uint32_t stepCols = usedCoreNum_ - stepRows * tilesPerRow_;
        const uint32_t stepOb = stepRows / outHeight_;
        const uint32_t stepOh = stepRows - stepOb * outHeight_;
        CopyGroupedInterleavedInputByDma(inputLocal, ob, oh, owBegin, currentWidth);
        PipeBarrier<PIPE_ALL>();
        GatherGroupedInput(inputLocal, rowLocal, currentWidth, pixelBytes, useSingleGather);
        PipeBarrier<PIPE_ALL>();

        uint32_t bufferIndex = 0;
        for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
        {
            const bool hasNext = tile + usedCoreNum_ < totalTiles_;
            uint32_t nextOb = 0;
            uint32_t nextOh = 0;
            uint32_t nextOwBegin = 0;
            uint32_t nextWidth = 0;
            uint32_t nextTileInRow = tileInRow;
            LocalTensor<DT_X> nextInputLocal =
                (bufferIndex == 0) ? inputLocal[inputTileElements] : inputLocal;
            LocalTensor<DT_X> nextRowLocal =
                (bufferIndex == 0) ? rowLocal[rowTileElements] : rowLocal;
            if (hasNext)
            {
                nextTileInRow += stepCols;
                uint32_t carryRow = 0;
                if (nextTileInRow >= tilesPerRow_)
                {
                    nextTileInRow -= tilesPerRow_;
                    carryRow = 1;
                }
                nextOb = ob + stepOb;
                nextOh = oh + stepOh + carryRow;
                if (nextOh >= outHeight_)
                {
                    nextOh -= outHeight_;
                    ++nextOb;
                }
                nextOwBegin = nextTileInRow * tileWidth_;
                nextWidth = outWidth_ - nextOwBegin;
                if (nextWidth > tileWidth_)
                {
                    nextWidth = tileWidth_;
                }
                CopyGroupedInterleavedInputByDma(
                    nextInputLocal, nextOb, nextOh, nextOwBegin, nextWidth);
            }

            LocalTensor<DT_X> currentRowLocal =
                (bufferIndex == 0) ? rowLocal : rowLocal[rowTileElements];
            if (BTS_OPT_GROUPED_GENERAL_OVERLAP_INPUT_OUTPUT && hasNext)
            {
                CopyGeneralOutput(currentRowLocal, ob, oh, owBegin, currentWidth, pixelBytes);
                PipeBarrier<PIPE_ALL>();
                GatherGroupedInput(
                    nextInputLocal, nextRowLocal, nextWidth, pixelBytes, useSingleGather);
            }
            else if (BTS_OPT_GROUPED_GENERAL_OVERLAP_GATHER_OUTPUT && hasNext)
            {
                PipeBarrier<PIPE_ALL>();
                GatherGroupedInput(
                    nextInputLocal, nextRowLocal, nextWidth, pixelBytes, useSingleGather);
                CopyGeneralOutput(currentRowLocal, ob, oh, owBegin, currentWidth, pixelBytes);
            }
            else
            {
                CopyGeneralOutput(currentRowLocal, ob, oh, owBegin, currentWidth, pixelBytes);
            }
            if (!hasNext)
            {
                return;
            }
            PipeBarrier<PIPE_ALL>();
            if (!BTS_OPT_GROUPED_GENERAL_OVERLAP_INPUT_OUTPUT &&
                !BTS_OPT_GROUPED_GENERAL_OVERLAP_GATHER_OUTPUT)
            {
                GatherGroupedInput(
                    nextInputLocal, nextRowLocal, nextWidth, pixelBytes, useSingleGather);
                PipeBarrier<PIPE_ALL>();
            }
            ob = nextOb;
            oh = nextOh;
            owBegin = nextOwBegin;
            currentWidth = nextWidth;
            tileInRow = nextTileInRow;
            bufferIndex ^= 1U;
        }
    }

    __aicore__ inline void ProcessBlock2DeepDirectLocalTiles(
        uint32_t coreId, LocalTensor<DT_X> rowLocal, uint32_t pixelBytes)
    {
        const uint32_t bytesPerElement = static_cast<uint32_t>(sizeof(DT_X));
        const uint32_t rowTileBytes =
            ((tileWidth_ * pixelBytes + 31U) / 32U) * 32U;
        const uint32_t rowTileElements = rowTileBytes / bytesPerElement;
        const bool canDoubleBuffer = tileBufferBytes_ >= rowTileBytes * 2U;

        if (totalTiles_ <= usedCoreNum_ || !canDoubleBuffer)
        {
            for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
            {
                uint32_t ob = 0U;
                uint32_t oh = 0U;
                uint32_t owBegin = 0U;
                uint32_t currentWidth = 0U;
                DecodeGeneralTile(tile, ob, oh, owBegin, currentWidth);
                CopyBlock2DeepDirectLocalTile(rowLocal, ob, oh, owBegin, currentWidth);
                PipeBarrier<PIPE_ALL>();
                CopyGeneralOutput(rowLocal, ob, oh, owBegin, currentWidth, pixelBytes);
                if (tile + usedCoreNum_ < totalTiles_)
                {
                    PipeBarrier<PIPE_ALL>();
                }
            }
            return;
        }

        uint32_t ob = 0U;
        uint32_t oh = 0U;
        uint32_t owBegin = 0U;
        uint32_t currentWidth = 0U;
        DecodeGeneralTile(coreId, ob, oh, owBegin, currentWidth);
        CopyBlock2DeepDirectLocalTile(rowLocal, ob, oh, owBegin, currentWidth);
        PipeBarrier<PIPE_ALL>();

        uint32_t bufferIndex = 0U;
        for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
        {
            const bool hasNext = tile + usedCoreNum_ < totalTiles_;
            uint32_t nextOb = 0U;
            uint32_t nextOh = 0U;
            uint32_t nextOwBegin = 0U;
            uint32_t nextWidth = 0U;
            if (hasNext)
            {
                DecodeGeneralTile(
                    tile + usedCoreNum_, nextOb, nextOh, nextOwBegin, nextWidth);
                LocalTensor<DT_X> nextRowLocal =
                    (bufferIndex == 0U) ? rowLocal[rowTileElements] : rowLocal;
                CopyBlock2DeepDirectLocalTile(
                    nextRowLocal, nextOb, nextOh, nextOwBegin, nextWidth);
            }

            LocalTensor<DT_X> currentRowLocal =
                (bufferIndex == 0U) ? rowLocal : rowLocal[rowTileElements];
            CopyGeneralOutput(currentRowLocal, ob, oh, owBegin, currentWidth, pixelBytes);
            if (!hasNext)
            {
                return;
            }
            PipeBarrier<PIPE_ALL>();
            ob = nextOb;
            oh = nextOh;
            owBegin = nextOwBegin;
            currentWidth = nextWidth;
            bufferIndex ^= 1U;
        }
    }

    __aicore__ inline void CopyBlock2DeepDirectLocalTile(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint32_t currentWidth)
    {
        const uint32_t pixelBytes = depth_ * static_cast<uint32_t>(sizeof(DT_X));
        const uint32_t spatialH = oh + cropTop_;
        const uint32_t ih = spatialH >> 1;
        const uint32_t blockH = spatialH & 1U;
        const uint32_t spatialWBegin = owBegin + cropLeft_;
        const uint32_t iw = spatialWBegin >> 1;
        const uint32_t blockW = spatialWBegin & 1U;
        const uint32_t ib0 = ob + outBatch_ * ((blockH << 1) + blockW);
        const uint32_t inputOffset0 = ((ib0 * height_ + ih) * width_ + iw) * depth_;
        const uint32_t count0 = (currentWidth + 1U) >> 1;
        const DataCopyPadExtParams<DT_X> pad = {false, 0, 0, 0};
        const DataCopyExtParams copy0 = {
            static_cast<uint16_t>(count0),
            pixelBytes,
            0,
            pixelBytes,
            0};
        DataCopyPad(rowLocal, xGm_[inputOffset0], copy0, pad);
        if (currentWidth <= 1U)
        {
            return;
        }

        const uint32_t blockW1 = blockW ^ 1U;
        const uint32_t iw1 = iw + blockW;
        const uint32_t ib1 = ob + outBatch_ * ((blockH << 1) + blockW1);
        const uint32_t inputOffset1 = ((ib1 * height_ + ih) * width_ + iw1) * depth_;
        const uint32_t count1 = currentWidth >> 1;
        const DataCopyExtParams copy1 = {
            static_cast<uint16_t>(count1),
            pixelBytes,
            0,
            pixelBytes,
            0};
        DataCopyPad(rowLocal[depth_], xGm_[inputOffset1], copy1, pad);
    }

    __aicore__ inline void ProcessGeneralDirectDmaTiles(
        uint32_t coreId, LocalTensor<DT_X> rowLocal, uint32_t pixelBytes)
    {
        if (BTS_OPT_BLOCK2_DEEP_DIRECT_PADDED_PHASE &&
            blockSize_ == 2U && sizeof(DT_X) == 2U &&
            pixelBytes > 128U && outWidth_ > 64U && outWidth_ <= 256U &&
            outHeight_ > 64U)
        {
            ProcessBlock2DeepDirectPaddedPhaseTiles(coreId, rowLocal, pixelBytes);
            return;
        }
        const uint32_t bytesPerElement = static_cast<uint32_t>(sizeof(DT_X));
        const uint32_t groupCount =
            (tileWidth_ < blockSize_) ? tileWidth_ : blockSize_;
        const uint32_t inputTileElements = groupCount * inputGroupStrideElements_;
        const uint32_t inputTileBytes = inputTileElements * bytesPerElement;
        const bool canDoubleBuffer = tileBufferBytes_ >= inputTileBytes * 2U;

        if (totalTiles_ <= usedCoreNum_ || !canDoubleBuffer)
        {
            for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
            {
                uint32_t ob = 0;
                uint32_t oh = 0;
                uint32_t owBegin = 0;
                uint32_t currentWidth = 0;
                DecodeGeneralTile(tile, ob, oh, owBegin, currentWidth);
                CopyGroupedInterleavedInputByDma(rowLocal, ob, oh, owBegin, currentWidth);
                PipeBarrier<PIPE_ALL>();
                CopyGroupedInterleavedOutputDirect(
                    rowLocal, ob, oh, owBegin, currentWidth, pixelBytes);
                if (tile + usedCoreNum_ < totalTiles_)
                {
                    PipeBarrier<PIPE_ALL>();
                }
            }
            return;
        }

        uint32_t ob = 0;
        uint32_t oh = 0;
        uint32_t owBegin = 0;
        uint32_t currentWidth = 0;
        const uint32_t outputRowStart = coreId / tilesPerRow_;
        uint32_t tileInRow = coreId - outputRowStart * tilesPerRow_;
        ob = outputRowStart / outHeight_;
        oh = outputRowStart - ob * outHeight_;
        owBegin = tileInRow * tileWidth_;
        currentWidth = outWidth_ - owBegin;
        if (currentWidth > tileWidth_)
        {
            currentWidth = tileWidth_;
        }
        const uint32_t stepRows = usedCoreNum_ / tilesPerRow_;
        const uint32_t stepCols = usedCoreNum_ - stepRows * tilesPerRow_;
        const uint32_t stepOb = stepRows / outHeight_;
        const uint32_t stepOh = stepRows - stepOb * outHeight_;

        LocalTensor<DT_X> currentLocal = rowLocal;
        CopyGroupedInterleavedInputByDma(currentLocal, ob, oh, owBegin, currentWidth);
        PipeBarrier<PIPE_ALL>();

        uint32_t bufferIndex = 0;
        for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
        {
            const bool hasNext = tile + usedCoreNum_ < totalTiles_;
            uint32_t nextTileInRow = tileInRow;
            uint32_t nextOb = 0;
            uint32_t nextOh = 0;
            uint32_t nextOwBegin = 0;
            uint32_t nextWidth = 0;
            if (hasNext)
            {
                nextTileInRow += stepCols;
                uint32_t carryRow = 0;
                if (nextTileInRow >= tilesPerRow_)
                {
                    nextTileInRow -= tilesPerRow_;
                    carryRow = 1;
                }
                nextOb = ob + stepOb;
                nextOh = oh + stepOh + carryRow;
                if (nextOh >= outHeight_)
                {
                    nextOh -= outHeight_;
                    ++nextOb;
                }
                nextOwBegin = nextTileInRow * tileWidth_;
                nextWidth = outWidth_ - nextOwBegin;
                if (nextWidth > tileWidth_)
                {
                    nextWidth = tileWidth_;
                }
                LocalTensor<DT_X> nextLocal =
                    (bufferIndex == 0) ? rowLocal[inputTileElements] : rowLocal;
                CopyGroupedInterleavedInputByDma(
                    nextLocal, nextOb, nextOh, nextOwBegin, nextWidth);
            }

            CopyGroupedInterleavedOutputDirect(
                currentLocal, ob, oh, owBegin, currentWidth, pixelBytes);
            if (!hasNext)
            {
                return;
            }
            PipeBarrier<PIPE_ALL>();
            tileInRow = nextTileInRow;
            ob = nextOb;
            oh = nextOh;
            owBegin = nextOwBegin;
            currentWidth = nextWidth;
            bufferIndex ^= 1U;
            currentLocal = (bufferIndex == 0) ? rowLocal : rowLocal[inputTileElements];
        }
    }

    __aicore__ inline void ProcessBlock2DeepDirectPaddedPhaseTiles(
        uint32_t coreId, LocalTensor<DT_X> rowLocal, uint32_t pixelBytes)
    {
        const uint32_t inputTileElements = 2U * inputGroupStrideElements_;
        const uint32_t inputTileBytes =
            inputTileElements * static_cast<uint32_t>(sizeof(DT_X));
        const bool canDoubleBuffer = tileBufferBytes_ >= inputTileBytes * 2U;

        if (totalTiles_ <= usedCoreNum_ || !canDoubleBuffer)
        {
            for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
            {
                uint32_t ob = 0U;
                uint32_t oh = 0U;
                uint32_t owBegin = 0U;
                uint32_t currentWidth = 0U;
                DecodeGeneralTile(tile, ob, oh, owBegin, currentWidth);
                CopyBlock2DeepPaddedPhaseInput(
                    rowLocal, ob, oh, owBegin, currentWidth, pixelBytes);
                PipeBarrier<PIPE_ALL>();
                CopyBlock2DeepPaddedPhaseOutput(
                    rowLocal, ob, oh, owBegin, currentWidth, pixelBytes);
                if (tile + usedCoreNum_ < totalTiles_)
                {
                    PipeBarrier<PIPE_ALL>();
                }
            }
            return;
        }

        uint32_t ob = 0U;
        uint32_t oh = 0U;
        uint32_t owBegin = 0U;
        uint32_t currentWidth = 0U;
        const uint32_t outputRowStart = coreId / tilesPerRow_;
        uint32_t tileInRow = coreId - outputRowStart * tilesPerRow_;
        ob = outputRowStart / outHeight_;
        oh = outputRowStart - ob * outHeight_;
        owBegin = tileInRow * tileWidth_;
        currentWidth = outWidth_ - owBegin;
        if (currentWidth > tileWidth_)
        {
            currentWidth = tileWidth_;
        }
        const uint32_t stepRows = usedCoreNum_ / tilesPerRow_;
        const uint32_t stepCols = usedCoreNum_ - stepRows * tilesPerRow_;
        const uint32_t stepOb = stepRows / outHeight_;
        const uint32_t stepOh = stepRows - stepOb * outHeight_;

        LocalTensor<DT_X> currentLocal = rowLocal;
        CopyBlock2DeepPaddedPhaseInput(
            currentLocal, ob, oh, owBegin, currentWidth, pixelBytes);
        PipeBarrier<PIPE_ALL>();

        uint32_t bufferIndex = 0U;
        for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
        {
            const bool hasNext = tile + usedCoreNum_ < totalTiles_;
            uint32_t nextTileInRow = tileInRow;
            uint32_t nextOb = 0U;
            uint32_t nextOh = 0U;
            uint32_t nextOwBegin = 0U;
            uint32_t nextWidth = 0U;
            if (hasNext)
            {
                nextTileInRow += stepCols;
                uint32_t carryRow = 0U;
                if (nextTileInRow >= tilesPerRow_)
                {
                    nextTileInRow -= tilesPerRow_;
                    carryRow = 1U;
                }
                nextOb = ob + stepOb;
                nextOh = oh + stepOh + carryRow;
                if (nextOh >= outHeight_)
                {
                    nextOh -= outHeight_;
                    ++nextOb;
                }
                nextOwBegin = nextTileInRow * tileWidth_;
                nextWidth = outWidth_ - nextOwBegin;
                if (nextWidth > tileWidth_)
                {
                    nextWidth = tileWidth_;
                }
                LocalTensor<DT_X> nextLocal =
                    (bufferIndex == 0U) ? rowLocal[inputTileElements] : rowLocal;
                CopyBlock2DeepPaddedPhaseInput(
                    nextLocal, nextOb, nextOh, nextOwBegin, nextWidth, pixelBytes);
            }

            CopyBlock2DeepPaddedPhaseOutput(
                currentLocal, ob, oh, owBegin, currentWidth, pixelBytes);
            if (!hasNext)
            {
                return;
            }
            PipeBarrier<PIPE_ALL>();
            tileInRow = nextTileInRow;
            ob = nextOb;
            oh = nextOh;
            owBegin = nextOwBegin;
            currentWidth = nextWidth;
            bufferIndex ^= 1U;
            currentLocal = (bufferIndex == 0U) ? rowLocal : rowLocal[inputTileElements];
        }
    }

    __aicore__ inline void CopyBlock2DeepPaddedPhaseInput(
        LocalTensor<DT_X> inputLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint32_t currentWidth, uint32_t pixelBytes)
    {
        const uint32_t alignedPixelBytes = ((pixelBytes + 31U) / 32U) * 32U;
        const uint32_t spatialH = oh + cropTop_;
        const uint32_t ih = spatialH >> 1;
        const uint32_t blockH = spatialH & 1U;
        const uint32_t spatialWBegin = owBegin + cropLeft_;
        const uint32_t iw = spatialWBegin >> 1;
        const uint32_t blockW = spatialWBegin & 1U;
        const DataCopyPadExtParams<DT_X> pad = {false, 0, 0, 0};

        const uint32_t count0 = (currentWidth + 1U) >> 1;
        const uint32_t ib0 = ob + outBatch_ * ((blockH << 1) + blockW);
        const uint32_t inputOffset0 = ((ib0 * height_ + ih) * width_ + iw) * depth_;
        const DataCopyExtParams copy0 = {
            static_cast<uint16_t>(count0),
            pixelBytes,
            0,
            alignedPixelBytes - pixelBytes,
            0};
        DataCopyPad(inputLocal, xGm_[inputOffset0], copy0, pad);
        if (currentWidth <= 1U)
        {
            return;
        }

        const uint32_t blockW1 = blockW ^ 1U;
        const uint32_t iw1 = iw + blockW;
        const uint32_t ib1 = ob + outBatch_ * ((blockH << 1) + blockW1);
        const uint32_t inputOffset1 = ((ib1 * height_ + ih) * width_ + iw1) * depth_;
        const uint32_t count1 = currentWidth >> 1;
        const DataCopyExtParams copy1 = {
            static_cast<uint16_t>(count1),
            pixelBytes,
            0,
            alignedPixelBytes - pixelBytes,
            0};
        DataCopyPad(inputLocal[inputGroupStrideElements_], xGm_[inputOffset1], copy1, pad);
    }

    __aicore__ inline void CopyBlock2DeepPaddedPhaseOutput(
        LocalTensor<DT_X> inputLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint32_t currentWidth, uint32_t pixelBytes)
    {
        const uint32_t alignedPixelBytes = ((pixelBytes + 31U) / 32U) * 32U;
        const uint32_t outputBase =
            ((ob * outHeight_ + oh) * outWidth_ + owBegin) * depth_;
        const uint32_t count0 = (currentWidth + 1U) >> 1;
        const DataCopyExtParams copy0 = {
            static_cast<uint16_t>(count0),
            pixelBytes,
            alignedPixelBytes - pixelBytes,
            pixelBytes,
            0};
        DataCopyPad(yGm_[outputBase], inputLocal, copy0);
        if (currentWidth <= 1U)
        {
            return;
        }

        const uint32_t count1 = currentWidth >> 1;
        const DataCopyExtParams copy1 = {
            static_cast<uint16_t>(count1),
            pixelBytes,
            alignedPixelBytes - pixelBytes,
            pixelBytes,
            0};
        DataCopyPad(yGm_[outputBase + depth_], inputLocal[inputGroupStrideElements_], copy1);
    }

    __aicore__ inline void DecodeGeneralTile(
        uint32_t tile, uint32_t &ob, uint32_t &oh, uint32_t &owBegin,
        uint32_t &currentWidth)
    {
        const uint32_t outputRow = tile / tilesPerRow_;
        const uint32_t tileInRow = tile - outputRow * tilesPerRow_;
        ob = outputRow / outHeight_;
        oh = outputRow - ob * outHeight_;
        owBegin = tileInRow * tileWidth_;
        currentWidth = outWidth_ - owBegin;
        if (currentWidth > tileWidth_)
        {
            currentWidth = tileWidth_;
        }
    }

    __aicore__ inline void DecodeGeneralRowTile(
        uint32_t tile, uint32_t &ob, uint32_t &ohBegin, uint32_t &currentRows)
    {
        ob = tile / tilesPerRow_;
        const uint32_t rowTile = tile - ob * tilesPerRow_;
        ohBegin = rowTile * rowTileHeight_;
        currentRows = outHeight_ - ohBegin;
        if (currentRows > rowTileHeight_)
        {
            currentRows = rowTileHeight_;
        }
    }

    __aicore__ inline void CopyGeneralOutput(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint32_t currentWidth, uint32_t pixelBytes)
    {
        const uint32_t outputOffset =
            ((ob * outHeight_ + oh) * outWidth_ + owBegin) * depth_;
        const uint32_t outputBytes = currentWidth * pixelBytes;
        const uint32_t outputOffsetBytes =
            outputOffset * static_cast<uint32_t>(sizeof(DT_X));
        if (outputOffsetBytes % 32 == 0 && outputBytes % 32 == 0)
        {
            const DataCopyParams outputCopy = {
                1,
                static_cast<uint16_t>(outputBytes / 32),
                0,
                0};
            DataCopy(yGm_[outputOffset], rowLocal, outputCopy);
        }
        else
        {
            const DataCopyExtParams outputCopy = {
                1, outputBytes, 0, 0, 0};
            DataCopyPad(yGm_[outputOffset], rowLocal, outputCopy);
        }
    }

    __aicore__ inline void CopyGeneralRowsOutput(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t ohBegin,
        uint32_t currentRows, uint32_t pixelBytes)
    {
        const uint32_t rowElements = outWidth_ * depth_;
        const uint32_t outputOffset = (ob * outHeight_ + ohBegin) * rowElements;
        const uint32_t outputBytes = currentRows * outWidth_ * pixelBytes;
        const uint32_t outputOffsetBytes =
            outputOffset * static_cast<uint32_t>(sizeof(DT_X));
        if (outputOffsetBytes % 32 == 0 && outputBytes % 32 == 0)
        {
            const DataCopyParams outputCopy = {
                1,
                static_cast<uint16_t>(outputBytes / 32),
                0,
                0};
            DataCopy(yGm_[outputOffset], rowLocal, outputCopy);
        }
        else
        {
            const DataCopyExtParams outputCopy = {
                1, outputBytes, 0, 0, 0};
            DataCopyPad(yGm_[outputOffset], rowLocal, outputCopy);
        }
    }

    __aicore__ inline void CopyGeneralPaddedRowsOutput(
        LocalTensor<DT_X> rowLocal, uint32_t rowStrideElements, uint32_t ob,
        uint32_t ohBegin, uint32_t currentRows, uint32_t pixelBytes)
    {
        for (uint32_t row = 0; row < currentRows; ++row)
        {
            CopyGeneralOutput(
                rowLocal[row * rowStrideElements], ob, ohBegin + row, 0, outWidth_,
                pixelBytes);
        }
    }

    __aicore__ inline void ProcessAlignedRowTiles(
        uint32_t coreId, LocalTensor<DT_X> rowLocal, uint32_t pixelBytes)
    {
        const uint32_t rowElements = outWidth_ * depth_;
        const uint32_t rowCopyBlocks = outWidth_ * pixelBytes / 32;
        if (totalTiles_ <= usedCoreNum_)
        {
            const uint32_t ob = coreId / tilesPerRow_;
            const uint32_t rowTile = coreId - ob * tilesPerRow_;
            const uint32_t ohBegin = rowTile * rowTileHeight_;
            uint32_t currentRows = outHeight_ - ohBegin;
            if (currentRows > rowTileHeight_)
            {
                currentRows = rowTileHeight_;
            }
            if (MODE == MODE_ROW && blockSize_ == 1)
            {
                for (uint32_t row = 0; row < currentRows; ++row)
                {
                    CopyContiguousTileAligned(
                        rowLocal[row * rowElements], ob, ohBegin + row, 0, outWidth_, pixelBytes);
                }
            }
            else
            {
                for (uint32_t row = 0; row < currentRows; ++row)
                {
                    CopyInterleavedTileByDma(
                        rowLocal[row * rowElements], ob, ohBegin + row, 0, outWidth_);
                }
            }
            PipeBarrier<PIPE_ALL>();
            const uint32_t outputOffset = (ob * outHeight_ + ohBegin) * rowElements;
            const DataCopyParams outputCopy = {
                1,
                static_cast<uint16_t>(currentRows * rowCopyBlocks),
                0,
                0};
            DataCopy(yGm_[outputOffset], rowLocal, outputCopy);
            return;
        }
        uint32_t ob = coreId / tilesPerRow_;
        uint32_t rowTile = coreId - ob * tilesPerRow_;
        const uint32_t stepOb = usedCoreNum_ / tilesPerRow_;
        const uint32_t stepRowTile = usedCoreNum_ - stepOb * tilesPerRow_;
        const uint32_t tileElements = rowTileHeight_ * rowElements;
        const uint32_t singleTileBytes = rowTileHeight_ * outWidth_ * pixelBytes;
        if (tileBufferBytes_ >= singleTileBytes * 2)
        {
            LocalTensor<DT_X> currentLocal = rowLocal;
            uint32_t ohBegin = rowTile * rowTileHeight_;
            uint32_t currentRows = outHeight_ - ohBegin;
            if (currentRows > rowTileHeight_)
            {
                currentRows = rowTileHeight_;
            }
            if (MODE == MODE_ROW && blockSize_ == 1)
            {
                for (uint32_t row = 0; row < currentRows; ++row)
                {
                    CopyContiguousTileAligned(
                        currentLocal[row * rowElements], ob, ohBegin + row, 0, outWidth_,
                        pixelBytes);
                }
            }
            else
            {
                for (uint32_t row = 0; row < currentRows; ++row)
                {
                    CopyInterleavedTileByDma(
                        currentLocal[row * rowElements], ob, ohBegin + row, 0, outWidth_);
                }
            }
            PipeBarrier<PIPE_ALL>();
            uint32_t bufferIndex = 0;
            for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
            {
                const bool hasNext = tile + usedCoreNum_ < totalTiles_;
                uint32_t nextOb = ob;
                uint32_t nextRowTile = rowTile;
                if (hasNext)
                {
                    nextRowTile += stepRowTile;
                    nextOb += stepOb;
                    if (nextRowTile >= tilesPerRow_)
                    {
                        nextRowTile -= tilesPerRow_;
                        ++nextOb;
                    }
                    LocalTensor<DT_X> nextLocal =
                        (bufferIndex == 0) ? rowLocal[tileElements] : rowLocal;
                    const uint32_t nextOhBegin = nextRowTile * rowTileHeight_;
                    uint32_t nextRows = outHeight_ - nextOhBegin;
                    if (nextRows > rowTileHeight_)
                    {
                        nextRows = rowTileHeight_;
                    }
                    if (MODE == MODE_ROW && blockSize_ == 1)
                    {
                        for (uint32_t row = 0; row < nextRows; ++row)
                        {
                            CopyContiguousTileAligned(
                                nextLocal[row * rowElements], nextOb,
                                nextOhBegin + row, 0, outWidth_, pixelBytes);
                        }
                    }
                    else
                    {
                        for (uint32_t row = 0; row < nextRows; ++row)
                        {
                            CopyInterleavedTileByDma(
                                nextLocal[row * rowElements], nextOb,
                                nextOhBegin + row, 0, outWidth_);
                        }
                    }
                }

                const uint32_t outputOffset = (ob * outHeight_ + ohBegin) * rowElements;
                const DataCopyParams outputCopy = {
                    1,
                    static_cast<uint16_t>(currentRows * rowCopyBlocks),
                    0,
                    0};
                DataCopy(yGm_[outputOffset], currentLocal, outputCopy);
                if (!hasNext)
                {
                    return;
                }
                PipeBarrier<PIPE_ALL>();
                ob = nextOb;
                rowTile = nextRowTile;
                ohBegin = rowTile * rowTileHeight_;
                currentRows = outHeight_ - ohBegin;
                if (currentRows > rowTileHeight_)
                {
                    currentRows = rowTileHeight_;
                }
                bufferIndex ^= 1;
                currentLocal = (bufferIndex == 0) ? rowLocal : rowLocal[tileElements];
            }
            return;
        }
        for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
        {
            const uint32_t ohBegin = rowTile * rowTileHeight_;
            uint32_t currentRows = outHeight_ - ohBegin;
            if (currentRows > rowTileHeight_)
            {
                currentRows = rowTileHeight_;
            }

            if (MODE == MODE_ROW && blockSize_ == 1)
            {
                for (uint32_t row = 0; row < currentRows; ++row)
                {
                    CopyContiguousTileAligned(
                        rowLocal[row * rowElements], ob, ohBegin + row, 0, outWidth_, pixelBytes);
                }
            }
            else
            {
                for (uint32_t row = 0; row < currentRows; ++row)
                {
                    CopyInterleavedTileByDma(
                        rowLocal[row * rowElements], ob, ohBegin + row, 0, outWidth_);
                }
            }
            PipeBarrier<PIPE_ALL>();

            const uint32_t outputOffset = (ob * outHeight_ + ohBegin) * rowElements;
            const DataCopyParams outputCopy = {
                1,
                static_cast<uint16_t>(currentRows * rowCopyBlocks),
                0,
                0};
            DataCopy(yGm_[outputOffset], rowLocal, outputCopy);
            if (tile + usedCoreNum_ < totalTiles_)
            {
                PipeBarrier<PIPE_ALL>();
            }
            rowTile += stepRowTile;
            ob += stepOb;
            if (rowTile >= tilesPerRow_)
            {
                rowTile -= tilesPerRow_;
                ++ob;
            }
        }
    }

    __aicore__ inline void CopyInterleavedRowsByDma(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t ohBegin, uint32_t currentRows,
        uint32_t rowElements)
    {
        if (blockSize_ == 2)
        {
            for (uint32_t row = 0; row < currentRows; ++row)
            {
                CopyInterleavedTileByDmaBlock2(
                    rowLocal[row * rowElements], ob, ohBegin + row, 0, outWidth_);
            }
            return;
        }
        for (uint32_t row = 0; row < currentRows; ++row)
        {
            CopyInterleavedTileByDma(
                rowLocal[row * rowElements], ob, ohBegin + row, 0, outWidth_);
        }
    }

    __aicore__ inline void ProcessAlignedRowTilesInterleavedOnly(
        uint32_t coreId, LocalTensor<DT_X> rowLocal, uint32_t pixelBytes)
    {
        const uint32_t rowElements = outWidth_ * depth_;
        const uint32_t rowCopyBlocks = outWidth_ * pixelBytes / 32;
        if (totalTiles_ <= usedCoreNum_)
        {
            const uint32_t ob = coreId / tilesPerRow_;
            const uint32_t rowTile = coreId - ob * tilesPerRow_;
            const uint32_t ohBegin = rowTile * rowTileHeight_;
            uint32_t currentRows = outHeight_ - ohBegin;
            if (currentRows > rowTileHeight_)
            {
                currentRows = rowTileHeight_;
            }
            CopyInterleavedRowsByDma(rowLocal, ob, ohBegin, currentRows, rowElements);
            PipeBarrier<PIPE_ALL>();
            const uint32_t outputOffset = (ob * outHeight_ + ohBegin) * rowElements;
            const DataCopyParams outputCopy = {
                1,
                static_cast<uint16_t>(currentRows * rowCopyBlocks),
                0,
                0};
            DataCopy(yGm_[outputOffset], rowLocal, outputCopy);
            return;
        }

        uint32_t ob = coreId / tilesPerRow_;
        uint32_t rowTile = coreId - ob * tilesPerRow_;
        const uint32_t stepOb = usedCoreNum_ / tilesPerRow_;
        const uint32_t stepRowTile = usedCoreNum_ - stepOb * tilesPerRow_;
        const uint32_t tileElements = rowTileHeight_ * rowElements;
        const uint32_t singleTileBytes = rowTileHeight_ * outWidth_ * pixelBytes;
        if (tileBufferBytes_ >= singleTileBytes * 2)
        {
            LocalTensor<DT_X> currentLocal = rowLocal;
            uint32_t ohBegin = rowTile * rowTileHeight_;
            uint32_t currentRows = outHeight_ - ohBegin;
            if (currentRows > rowTileHeight_)
            {
                currentRows = rowTileHeight_;
            }
            CopyInterleavedRowsByDma(currentLocal, ob, ohBegin, currentRows, rowElements);
            PipeBarrier<PIPE_ALL>();
            uint32_t bufferIndex = 0;
            for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
            {
                const bool hasNext = tile + usedCoreNum_ < totalTiles_;
                uint32_t nextOb = ob;
                uint32_t nextRowTile = rowTile;
                if (hasNext)
                {
                    nextRowTile += stepRowTile;
                    nextOb += stepOb;
                    if (nextRowTile >= tilesPerRow_)
                    {
                        nextRowTile -= tilesPerRow_;
                        ++nextOb;
                    }
                    LocalTensor<DT_X> nextLocal =
                        (bufferIndex == 0) ? rowLocal[tileElements] : rowLocal;
                    const uint32_t nextOhBegin = nextRowTile * rowTileHeight_;
                    uint32_t nextRows = outHeight_ - nextOhBegin;
                    if (nextRows > rowTileHeight_)
                    {
                        nextRows = rowTileHeight_;
                    }
                    CopyInterleavedRowsByDma(
                        nextLocal, nextOb, nextOhBegin, nextRows, rowElements);
                }

                const uint32_t outputOffset = (ob * outHeight_ + ohBegin) * rowElements;
                const DataCopyParams outputCopy = {
                    1,
                    static_cast<uint16_t>(currentRows * rowCopyBlocks),
                    0,
                    0};
                DataCopy(yGm_[outputOffset], currentLocal, outputCopy);
                if (!hasNext)
                {
                    return;
                }
                PipeBarrier<PIPE_ALL>();
                ob = nextOb;
                rowTile = nextRowTile;
                ohBegin = rowTile * rowTileHeight_;
                currentRows = outHeight_ - ohBegin;
                if (currentRows > rowTileHeight_)
                {
                    currentRows = rowTileHeight_;
                }
                bufferIndex ^= 1;
                currentLocal = (bufferIndex == 0) ? rowLocal : rowLocal[tileElements];
            }
            return;
        }

        for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
        {
            const uint32_t ohBegin = rowTile * rowTileHeight_;
            uint32_t currentRows = outHeight_ - ohBegin;
            if (currentRows > rowTileHeight_)
            {
                currentRows = rowTileHeight_;
            }
            CopyInterleavedRowsByDma(rowLocal, ob, ohBegin, currentRows, rowElements);
            PipeBarrier<PIPE_ALL>();

            const uint32_t outputOffset = (ob * outHeight_ + ohBegin) * rowElements;
            const DataCopyParams outputCopy = {
                1,
                static_cast<uint16_t>(currentRows * rowCopyBlocks),
                0,
                0};
            DataCopy(yGm_[outputOffset], rowLocal, outputCopy);
            if (tile + usedCoreNum_ < totalTiles_)
            {
                PipeBarrier<PIPE_ALL>();
            }
            rowTile += stepRowTile;
            ob += stepOb;
            if (rowTile >= tilesPerRow_)
            {
                rowTile -= tilesPerRow_;
                ++ob;
            }
        }
    }

    __aicore__ inline void ProcessAlignedFlatRowTiles(
        uint32_t coreId, LocalTensor<DT_X> rowLocal, uint32_t pixelBytes)
    {
        const uint32_t rowElements = outWidth_ * depth_;
        const uint32_t rowCopyBlocks = outWidth_ * pixelBytes / 32;
        const uint32_t totalRows = outBatch_ * outHeight_;
        for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
        {
            const uint32_t rowBegin = tile * rowTileHeight_;
            uint32_t currentRows = totalRows - rowBegin;
            if (currentRows > rowTileHeight_)
            {
                currentRows = rowTileHeight_;
            }
            uint32_t ob = rowBegin / outHeight_;
            uint32_t oh = rowBegin - ob * outHeight_;
            uint32_t rowOffset = 0;
            for (uint32_t row = 0; row < currentRows; ++row)
            {
                CopyInterleavedFullRowByDma(rowLocal[rowOffset], ob, oh, pixelBytes);
                rowOffset += rowElements;
                ++oh;
                if (oh == outHeight_)
                {
                    oh = 0;
                    ++ob;
                }
            }
            PipeBarrier<PIPE_ALL>();

            const uint32_t outputOffset = rowBegin * rowElements;
            const DataCopyParams outputCopy = {
                1,
                static_cast<uint16_t>(currentRows * rowCopyBlocks),
                0,
                0};
            DataCopy(yGm_[outputOffset], rowLocal, outputCopy);
            if (tile + usedCoreNum_ < totalTiles_)
            {
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void ProcessAlignedFlatChannelTiles(
        uint32_t coreId, LocalTensor<DT_X> rowLocal)
    {
        const uint32_t bytesPerElement = static_cast<uint32_t>(sizeof(DT_X));
        const uint32_t rowElements = outWidth_ * depth_;
        for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
        {
            const uint32_t rowIndex = tile / tilesPerRow_;
            const uint32_t channelTile = tile - rowIndex * tilesPerRow_;
            const uint32_t channelBegin = channelTile * tileWidth_;
            uint32_t currentChannels = depth_ - channelBegin;
            if (currentChannels > tileWidth_)
            {
                currentChannels = tileWidth_;
            }
            const uint32_t channelBytes = currentChannels * bytesPerElement;
            const uint32_t copyBlocks = channelBytes / 32;
            const uint32_t ob = rowIndex / outHeight_;
            const uint32_t oh = rowIndex - ob * outHeight_;

            CopyInterleavedFullRowChannelByDma(
                rowLocal, ob, oh, channelBegin, currentChannels);
            PipeBarrier<PIPE_ALL>();

            const uint32_t outputOffset = rowIndex * rowElements + channelBegin;
            const uint16_t dstStride =
                static_cast<uint16_t>((depth_ - currentChannels) * bytesPerElement / 32);
            const DataCopyParams outputCopy = {
                static_cast<uint16_t>(outWidth_),
                static_cast<uint16_t>(copyBlocks),
                0,
                dstStride};
            DataCopy(yGm_[outputOffset], rowLocal, outputCopy);
            if (tile + usedCoreNum_ < totalTiles_)
            {
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void ProcessElementwise(uint32_t coreId)
    {
        if (blockSize_ == 2)
        {
            ProcessElementwiseBlock2(coreId);
            return;
        }
        constexpr uint32_t elementsPerCacheLine = 64 / sizeof(DT_X);
        const uint32_t cacheLineCount =
            (totalOutputElements_ + elementsPerCacheLine - 1) / elementsPerCacheLine;
        const uint32_t linesPerCore = cacheLineCount / usedCoreNum_;
        const uint32_t extraLines = cacheLineCount % usedCoreNum_;
        const uint32_t startLine = coreId * linesPerCore +
                                   ((coreId < extraLines) ? coreId : extraLines);
        const uint32_t ownedLines = linesPerCore + ((coreId < extraLines) ? 1 : 0);
        const uint32_t start = startLine * elementsPerCacheLine;
        uint32_t end = (startLine + ownedLines) * elementsPerCacheLine;
        if (end > totalOutputElements_)
        {
            end = totalOutputElements_;
        }

        uint32_t outOffset = start;
        uint32_t outputPixel = outOffset / depth_;
        uint32_t od = outOffset - outputPixel * depth_;
        while (outOffset < end)
        {
            const uint32_t outputRow = outputPixel / outWidth_;
            const uint32_t ow = outputPixel - outputRow * outWidth_;
            const uint32_t ob = outputRow / outHeight_;
            const uint32_t oh = outputRow - ob * outHeight_;
            const uint32_t spatialH = oh + cropTop_;
            const uint32_t spatialW = ow + cropLeft_;
            const uint32_t ih = spatialH / blockSize_;
            const uint32_t iw = spatialW / blockSize_;
            const uint32_t blockH = spatialH - ih * blockSize_;
            const uint32_t blockW = spatialW - iw * blockSize_;
            const uint32_t ib = ob + outBatch_ * (blockH * blockSize_ + blockW);
            const uint32_t inputBase = ((ib * height_ + ih) * width_ + iw) * depth_;
            uint32_t channelCount = depth_ - od;
            const uint32_t remaining = end - outOffset;
            if (channelCount > remaining)
            {
                channelCount = remaining;
            }
            for (uint32_t channel = 0; channel < channelCount; ++channel)
            {
                yGm_.SetValue(
                    outOffset + channel, xGm_.GetValue(inputBase + od + channel));
            }
            outOffset += channelCount;
            od += channelCount;
            if (od == depth_)
            {
                od = 0;
                ++outputPixel;
            }
        }
        DataCacheCleanAndInvalid<DT_X, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(yGm_);
    }

    __aicore__ inline void ProcessElementwiseBlock2(uint32_t coreId)
    {
        constexpr uint32_t elementsPerCacheLine = 64 / sizeof(DT_X);
        const uint32_t cacheLineCount =
            (totalOutputElements_ + elementsPerCacheLine - 1) / elementsPerCacheLine;
        const uint32_t linesPerCore = cacheLineCount / usedCoreNum_;
        const uint32_t extraLines = cacheLineCount - linesPerCore * usedCoreNum_;
        const uint32_t startLine = coreId * linesPerCore +
                                   ((coreId < extraLines) ? coreId : extraLines);
        const uint32_t ownedLines = linesPerCore + ((coreId < extraLines) ? 1U : 0U);
        const uint32_t start = startLine * elementsPerCacheLine;
        uint32_t end = (startLine + ownedLines) * elementsPerCacheLine;
        if (end > totalOutputElements_)
        {
            end = totalOutputElements_;
        }

        uint32_t outOffset = start;
        uint32_t outputPixel = outOffset / depth_;
        uint32_t od = outOffset - outputPixel * depth_;
        uint32_t outputRow = outputPixel / outWidth_;
        uint32_t ow = outputPixel - outputRow * outWidth_;
        uint32_t ob = outputRow / outHeight_;
        uint32_t oh = outputRow - ob * outHeight_;
        while (outOffset < end)
        {
            const uint32_t spatialH = oh + cropTop_;
            const uint32_t ih = spatialH >> 1;
            const uint32_t blockH = spatialH & 1U;
            const uint32_t ibBase = ob + outBatch_ * (blockH << 1);
            while (ow < outWidth_ && outOffset < end)
            {
                const uint32_t spatialW = ow + cropLeft_;
                const uint32_t iw = spatialW >> 1;
                const uint32_t blockW = spatialW & 1U;
                const uint32_t ib = ibBase + outBatch_ * blockW;
                const uint32_t inputBase = ((ib * height_ + ih) * width_ + iw) * depth_;
                uint32_t channelCount = depth_ - od;
                const uint32_t remaining = end - outOffset;
                if (channelCount > remaining)
                {
                    channelCount = remaining;
                }
                for (uint32_t channel = 0; channel < channelCount; ++channel)
                {
                    yGm_.SetValue(
                        outOffset + channel, xGm_.GetValue(inputBase + od + channel));
                }
                outOffset += channelCount;
                od += channelCount;
                if (od != depth_)
                {
                    break;
                }
                od = 0;
                ++ow;
            }
            if (ow == outWidth_)
            {
                ow = 0;
                ++oh;
                if (oh == outHeight_)
                {
                    oh = 0;
                    ++ob;
                }
            }
        }
        DataCacheCleanAndInvalid<DT_X, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(yGm_);
    }

    __aicore__ inline void ProcessLinearCopy(uint32_t coreId, LocalTensor<DT_X> rowLocal)
    {
        const DataCopyPadExtParams<DT_X> pad = {false, 0, 0, 0};
        const uint32_t bytesPerElement = static_cast<uint32_t>(sizeof(DT_X));
        const uint32_t singleTileBytes =
            (tileWidth_ * bytesPerElement + 31U) / 32U * 32U;
        const uint32_t singleTileElements = singleTileBytes / bytesPerElement;
        if (totalTiles_ > usedCoreNum_ && tileBufferBytes_ >= singleTileBytes * 2U)
        {
            uint32_t begin = coreId * tileWidth_;
            uint32_t currentElements = totalOutputElements_ - begin;
            if (currentElements > tileWidth_)
            {
                currentElements = tileWidth_;
            }
            LocalTensor<DT_X> currentLocal = rowLocal;
            CopyLinearInput(currentLocal, begin, currentElements, bytesPerElement, pad);
            PipeBarrier<PIPE_ALL>();

            uint32_t bufferIndex = 0U;
            for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
            {
                const bool hasNext = tile + usedCoreNum_ < totalTiles_;
                uint32_t nextBegin = 0U;
                uint32_t nextElements = 0U;
                if (hasNext)
                {
                    nextBegin = (tile + usedCoreNum_) * tileWidth_;
                    nextElements = totalOutputElements_ - nextBegin;
                    if (nextElements > tileWidth_)
                    {
                        nextElements = tileWidth_;
                    }
                    LocalTensor<DT_X> nextLocal =
                        (bufferIndex == 0U) ? rowLocal[singleTileElements] : rowLocal;
                    CopyLinearInput(nextLocal, nextBegin, nextElements, bytesPerElement, pad);
                }

                CopyLinearOutput(currentLocal, begin, currentElements, bytesPerElement);
                if (!hasNext)
                {
                    return;
                }
                PipeBarrier<PIPE_ALL>();
                begin = nextBegin;
                currentElements = nextElements;
                bufferIndex ^= 1U;
                currentLocal =
                    (bufferIndex == 0U) ? rowLocal : rowLocal[singleTileElements];
            }
            return;
        }
        for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
        {
            const uint32_t begin = tile * tileWidth_;
            uint32_t currentElements = totalOutputElements_ - begin;
            if (currentElements > tileWidth_)
            {
                currentElements = tileWidth_;
            }
            CopyLinearInput(rowLocal, begin, currentElements, bytesPerElement, pad);
            PipeBarrier<PIPE_ALL>();
            CopyLinearOutput(rowLocal, begin, currentElements, bytesPerElement);
            if (tile + usedCoreNum_ < totalTiles_)
            {
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void CopyLinearInput(
        LocalTensor<DT_X> rowLocal, uint32_t begin, uint32_t currentElements,
        uint32_t bytesPerElement, const DataCopyPadExtParams<DT_X> &pad)
    {
        const uint32_t copyBytes = currentElements * bytesPerElement;
        const uint32_t inputOffset = linearInputOffset_ + begin;
        const uint32_t inputOffsetBytes = inputOffset * bytesPerElement;
        if ((inputOffsetBytes & 31U) == 0U && (copyBytes & 31U) == 0U)
        {
            const DataCopyParams copy = {
                1,
                static_cast<uint16_t>(copyBytes >> 5),
                0,
                0};
            DataCopy(rowLocal, xGm_[inputOffset], copy);
            return;
        }
        const DataCopyExtParams copy = {
            1,
            copyBytes,
            0,
            0,
            0};
        DataCopyPad(rowLocal, xGm_[inputOffset], copy, pad);
    }

    __aicore__ inline void CopyLinearOutput(
        LocalTensor<DT_X> rowLocal, uint32_t begin, uint32_t currentElements,
        uint32_t bytesPerElement)
    {
        const uint32_t copyBytes = currentElements * bytesPerElement;
        const uint32_t outputOffsetBytes = begin * bytesPerElement;
        if ((outputOffsetBytes & 31U) == 0U && (copyBytes & 31U) == 0U)
        {
            const DataCopyParams copy = {
                1,
                static_cast<uint16_t>(copyBytes >> 5),
                0,
                0};
            DataCopy(yGm_[begin], rowLocal, copy);
            return;
        }
        const DataCopyExtParams copy = {
            1,
            copyBytes,
            0,
            0,
            0};
        DataCopyPad(yGm_[begin], rowLocal, copy);
    }

    __aicore__ inline void ProcessChannelCopy(uint32_t coreId, LocalTensor<DT_X> channelLocal)
    {
        const uint32_t chunksPerPixel = tilesPerRow_;
        const uint32_t bytesPerElement = static_cast<uint32_t>(sizeof(DT_X));
        const DataCopyPadExtParams<DT_X> pad = {false, 0, 0, 0};
        for (uint32_t tile = coreId; tile < totalTiles_; tile += usedCoreNum_)
        {
            const uint32_t outputPixel = tile / chunksPerPixel;
            const uint32_t chunk = tile - outputPixel * chunksPerPixel;
            const uint32_t channelBegin = chunk * tileWidth_;
            uint32_t currentChannels = depth_ - channelBegin;
            if (currentChannels > tileWidth_)
            {
                currentChannels = tileWidth_;
            }

            const uint32_t outputRow = outputPixel / outWidth_;
            const uint32_t ow = outputPixel - outputRow * outWidth_;
            const uint32_t ob = outputRow / outHeight_;
            const uint32_t oh = outputRow - ob * outHeight_;
            const uint32_t spatialH = oh + cropTop_;
            const uint32_t spatialW = ow + cropLeft_;
            const uint32_t ih = spatialH / blockSize_;
            const uint32_t iw = spatialW / blockSize_;
            const uint32_t blockH = spatialH - ih * blockSize_;
            const uint32_t blockW = spatialW - iw * blockSize_;
            const uint32_t ib = ob + outBatch_ * (blockH * blockSize_ + blockW);
            const uint32_t inputOffset =
                ((ib * height_ + ih) * width_ + iw) * depth_ + channelBegin;
            const uint32_t outputOffset = outputPixel * depth_ + channelBegin;
            const uint32_t copyBytes = currentChannels * bytesPerElement;
            const DataCopyExtParams copy = {
                1,
                copyBytes,
                0,
                0,
                0};
            DataCopyPad(channelLocal, xGm_[inputOffset], copy, pad);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(yGm_[outputOffset], channelLocal, copy);
            if (tile + usedCoreNum_ < totalTiles_)
            {
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void CopyContiguousTileAligned(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint32_t currentWidth, uint32_t pixelBytes)
    {
        const uint32_t ih = oh + cropTop_;
        const uint32_t iw = owBegin + cropLeft_;
        const uint32_t inputOffset = ((ob * height_ + ih) * width_ + iw) * depth_;
        const DataCopyParams copy = {
            1,
            static_cast<uint16_t>(currentWidth * pixelBytes / 32),
            0,
            0};
        DataCopy(rowLocal, xGm_[inputOffset], copy);
    }

    __aicore__ inline void CopyContiguousTilePad(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint32_t currentWidth, uint32_t pixelBytes)
    {
        const uint32_t ih = oh + cropTop_;
        const uint32_t iw = owBegin + cropLeft_;
        const uint32_t inputOffset = ((ob * height_ + ih) * width_ + iw) * depth_;
        const DataCopyExtParams copy = {
            1, currentWidth * pixelBytes, 0, 0, 0};
        const DataCopyPadExtParams<DT_X> pad = {false, 0, 0, 0};
        DataCopyPad(rowLocal, xGm_[inputOffset], copy, pad);
    }

    __aicore__ inline void CopyInterleavedTileByDmaPad(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint32_t currentWidth, uint32_t pixelBytes)
    {
        if (blockSize_ == 1)
        {
            CopyContiguousTilePad(rowLocal, ob, oh, owBegin, currentWidth, pixelBytes);
            return;
        }
        const uint32_t spatialH = oh + cropTop_;
        const uint32_t ih = spatialH / blockSize_;
        const uint32_t blockH = spatialH - ih * blockSize_;
        const uint32_t firstGroupCount =
            (currentWidth < blockSize_) ? currentWidth : blockSize_;
        const uint32_t spatialWBegin = owBegin + cropLeft_;
        uint32_t iw = spatialWBegin / blockSize_;
        uint32_t blockW = spatialWBegin - iw * blockSize_;
        const uint32_t dstStride = (blockSize_ - 1U) * pixelBytes;
        const DataCopyPadExtParams<DT_X> pad = {false, 0, 0, 0};
        for (uint32_t first = 0; first < firstGroupCount; ++first)
        {
            const uint32_t ib = ob + outBatch_ * (blockH * blockSize_ + blockW);
            const uint32_t inputOffset = ((ib * height_ + ih) * width_ + iw) * depth_;
            const uint32_t blockCount =
                (currentWidth - first + blockSize_ - 1U) / blockSize_;
            const DataCopyExtParams copy = {
                static_cast<uint16_t>(blockCount),
                pixelBytes,
                0,
                dstStride,
                0};
            DataCopyPad(rowLocal[first * depth_], xGm_[inputOffset], copy, pad);
            ++blockW;
            if (blockW == blockSize_)
            {
                blockW = 0;
                ++iw;
            }
        }
    }

    __aicore__ inline void CopyInterleavedTileByDma(LocalTensor<DT_X> rowLocal, uint32_t ob,
                                                    uint32_t oh, uint32_t owBegin,
                                                    uint32_t currentWidth)
    {
        if (blockSize_ == 2)
        {
            CopyInterleavedTileByDmaBlock2(rowLocal, ob, oh, owBegin, currentWidth);
            return;
        }
        const uint32_t pixelBytes = depth_ * static_cast<uint32_t>(sizeof(DT_X));
        const uint32_t spatialH = oh + cropTop_;
        const uint32_t ih = spatialH / blockSize_;
        const uint32_t blockH = spatialH - ih * blockSize_;
        const DataCopyPadExtParams<DT_X> pad = {false, 0, 0, 0};
        const uint32_t firstGroupCount =
            (currentWidth < blockSize_) ? currentWidth : blockSize_;
        const uint32_t spatialWBegin = owBegin + cropLeft_;
        const uint16_t copyLen = static_cast<uint16_t>(pixelBytes / 32);
        const uint16_t srcStride = static_cast<uint16_t>((blockSize_ - 1) * copyLen);
        uint32_t iw = spatialWBegin / blockSize_;
        uint32_t blockW = spatialWBegin - iw * blockSize_;
        for (uint32_t first = 0; first < firstGroupCount; ++first)
        {
            const uint32_t ib = ob + outBatch_ * (blockH * blockSize_ + blockW);
            const uint32_t inputOffset = ((ib * height_ + ih) * width_ + iw) * depth_;
            const uint32_t blockCount =
                (currentWidth - first + blockSize_ - 1) / blockSize_;
            const DataCopyParams copy = {
                static_cast<uint16_t>(blockCount),
                copyLen,
                0,
                srcStride};
            DataCopy(rowLocal[first * depth_], xGm_[inputOffset], copy);
            ++blockW;
            if (blockW == blockSize_)
            {
                blockW = 0;
                ++iw;
            }
        }
    }

    __aicore__ inline void CopyAlignedInterleavedTileByDma(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint32_t currentWidth)
    {
        if (MODE == MODE_ALIGNED_BLOCK4_D64_PHASE_FAST)
        {
            CopyBlock4D64Phase1TileByDma(rowLocal, ob, oh, owBegin, currentWidth);
            return;
        }
        CopyInterleavedTileByDma(rowLocal, ob, oh, owBegin, currentWidth);
    }

    __aicore__ inline void CopyBlock4D64Phase1FullTileByDma(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t owBegin)
    {
        const uint32_t ih = (oh >> 2) + (cropTop_ >> 2);
        const uint32_t blockH = oh & 3U;
        const uint32_t iw = (owBegin >> 2) + (cropLeft_ >> 2);
        const uint32_t ibBase = ob + outBatch_ * (blockH << 2);
        const DataCopyParams copy = {64, 4, 0, 12};
        const uint32_t batchStride = height_ * width_ * 64U;
        const uint32_t outBatchStride = outBatch_ * batchStride;
        const uint32_t rowOffset = (ih * width_ + iw) * 64U;
        const uint32_t inputOffsetBase = ibBase * batchStride + rowOffset;
        DataCopy(rowLocal, xGm_[inputOffsetBase + outBatchStride], copy);
        DataCopy(rowLocal[64U], xGm_[inputOffsetBase + (outBatchStride << 1)], copy);
        DataCopy(rowLocal[128U], xGm_[inputOffsetBase + outBatchStride * 3U], copy);
        DataCopy(rowLocal[192U], xGm_[inputOffsetBase + 64U], copy);
    }

    __aicore__ inline void CopyBlock4D64Phase1TileByDma(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint32_t currentWidth)
    {
        if (((owBegin + cropLeft_) & 3U) != 1U || blockSize_ != 4U)
        {
            CopyInterleavedTileByDma(rowLocal, ob, oh, owBegin, currentWidth);
            return;
        }
        const uint32_t pixelBytes = depth_ * static_cast<uint32_t>(sizeof(DT_X));
        const uint16_t copyLen = static_cast<uint16_t>(pixelBytes / 32U);
        const uint16_t dstStride = static_cast<uint16_t>(copyLen * 3U);
        const uint32_t spatialH = oh + cropTop_;
        const uint32_t ih = spatialH >> 2;
        const uint32_t blockH = spatialH & 3U;
        const uint32_t iw = (owBegin + cropLeft_) >> 2;
        const uint32_t ibBase = ob + outBatch_ * (blockH << 2);
        if (currentWidth == tileWidth_ && ((currentWidth & 3U) == 0U))
        {
            const uint16_t count = static_cast<uint16_t>(currentWidth >> 2);
            const uint32_t inputOffset0 =
                (((ibBase + outBatch_) * height_ + ih) * width_ + iw) * depth_;
            const DataCopyParams copy0 = {count, copyLen, 0, dstStride};
            DataCopy(rowLocal, xGm_[inputOffset0], copy0);
            const uint32_t inputOffset1 =
                (((ibBase + outBatch_ * 2U) * height_ + ih) * width_ + iw) * depth_;
            const DataCopyParams copy1 = {count, copyLen, 0, dstStride};
            DataCopy(rowLocal[depth_], xGm_[inputOffset1], copy1);
            const uint32_t inputOffset2 =
                (((ibBase + outBatch_ * 3U) * height_ + ih) * width_ + iw) * depth_;
            const DataCopyParams copy2 = {count, copyLen, 0, dstStride};
            DataCopy(rowLocal[depth_ * 2U], xGm_[inputOffset2], copy2);
            const uint32_t inputOffset3 =
                ((ibBase * height_ + ih) * width_ + iw + 1U) * depth_;
            const DataCopyParams copy3 = {count, copyLen, 0, dstStride};
            DataCopy(rowLocal[depth_ * 3U], xGm_[inputOffset3], copy3);
            return;
        }

        const uint32_t count0 = (currentWidth + 3U) >> 2;
        const uint32_t inputOffset0 =
            (((ibBase + outBatch_) * height_ + ih) * width_ + iw) * depth_;
        const DataCopyParams copy0 = {
            static_cast<uint16_t>(count0),
            copyLen,
            0,
            dstStride};
        DataCopy(rowLocal, xGm_[inputOffset0], copy0);
        if (currentWidth <= 1U)
        {
            return;
        }

        const uint32_t count1 = (currentWidth + 2U) >> 2;
        const uint32_t inputOffset1 =
            (((ibBase + outBatch_ * 2U) * height_ + ih) * width_ + iw) * depth_;
        const DataCopyParams copy1 = {
            static_cast<uint16_t>(count1),
            copyLen,
            0,
            dstStride};
        DataCopy(rowLocal[depth_], xGm_[inputOffset1], copy1);
        if (currentWidth <= 2U)
        {
            return;
        }

        const uint32_t count2 = (currentWidth + 1U) >> 2;
        const uint32_t inputOffset2 =
            (((ibBase + outBatch_ * 3U) * height_ + ih) * width_ + iw) * depth_;
        const DataCopyParams copy2 = {
            static_cast<uint16_t>(count2),
            copyLen,
            0,
            dstStride};
        DataCopy(rowLocal[depth_ * 2U], xGm_[inputOffset2], copy2);
        if (currentWidth <= 3U)
        {
            return;
        }

        const uint32_t count3 = currentWidth >> 2;
        const uint32_t inputOffset3 =
            ((ibBase * height_ + ih) * width_ + iw + 1U) * depth_;
        const DataCopyParams copy3 = {
            static_cast<uint16_t>(count3),
            copyLen,
            0,
            dstStride};
        DataCopy(rowLocal[depth_ * 3U], xGm_[inputOffset3], copy3);
    }

    __aicore__ inline void CopyInterleavedTileByDmaBlock2(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint32_t currentWidth)
    {
        const uint32_t pixelBytes = depth_ * static_cast<uint32_t>(sizeof(DT_X));
        const uint32_t spatialH = oh + cropTop_;
        const uint32_t ih = spatialH >> 1;
        const uint32_t blockH = spatialH & 1;
        const uint32_t spatialWBegin = owBegin + cropLeft_;
        const uint32_t iw = spatialWBegin >> 1;
        const uint32_t blockW = spatialWBegin & 1;
        const uint16_t copyLen = static_cast<uint16_t>(pixelBytes / 32);
        const DataCopyParams copy0 = {
            static_cast<uint16_t>((currentWidth + 1) >> 1),
            copyLen,
            0,
            copyLen};
        const uint32_t ib0 = ob + outBatch_ * ((blockH << 1) + blockW);
        const uint32_t inputOffset0 = ((ib0 * height_ + ih) * width_ + iw) * depth_;
        DataCopy(rowLocal, xGm_[inputOffset0], copy0);
        if (currentWidth > 1)
        {
            const uint32_t blockW1 = blockW ^ 1;
            const uint32_t iw1 = iw + blockW;
            const uint32_t ib1 = ob + outBatch_ * ((blockH << 1) + blockW1);
            const uint32_t inputOffset1 = ((ib1 * height_ + ih) * width_ + iw1) * depth_;
            const DataCopyParams copy1 = {
                static_cast<uint16_t>(currentWidth >> 1),
                copyLen,
                0,
                copyLen};
            DataCopy(rowLocal[depth_], xGm_[inputOffset1], copy1);
        }
    }

    __aicore__ inline void CopyBlock2TileDirectOutput(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint32_t currentWidth, uint16_t copyLen)
    {
        if (currentWidth <= 8U && totalTiles_ <= usedCoreNum_)
        {
            CopyBlock2TileViaUb(rowLocal, ob, oh, owBegin, currentWidth, copyLen);
            return;
        }
        const uint32_t spatialH = oh + cropTop_;
        const uint32_t ih = spatialH >> 1;
        const uint32_t blockH = spatialH & 1;
        const uint32_t spatialWBegin = owBegin + cropLeft_;
        const uint32_t iw = spatialWBegin >> 1;
        const uint32_t blockW = spatialWBegin & 1;
        const uint32_t count0 = (currentWidth + 1) >> 1;
        const uint32_t count1 = currentWidth >> 1;
        const uint32_t ib0 = ob + outBatch_ * ((blockH << 1) + blockW);
        const uint32_t inputOffset0 = ((ib0 * height_ + ih) * width_ + iw) * depth_;
        const DataCopyParams inputCopy0 = {
            static_cast<uint16_t>(count0),
            copyLen,
            0,
            0};
        DataCopy(rowLocal, xGm_[inputOffset0], inputCopy0);
        if (count1 != 0)
        {
            const uint32_t blockW1 = blockW ^ 1;
            const uint32_t iw1 = iw + blockW;
            const uint32_t ib1 = ob + outBatch_ * ((blockH << 1) + blockW1);
            const uint32_t inputOffset1 = ((ib1 * height_ + ih) * width_ + iw1) * depth_;
            const DataCopyParams inputCopy1 = {
                static_cast<uint16_t>(count1),
                copyLen,
                0,
                0};
            DataCopy(rowLocal[count0 * depth_], xGm_[inputOffset1], inputCopy1);
        }
        PipeBarrier<PIPE_ALL>();
        const uint32_t outputOffset =
            ((ob * outHeight_ + oh) * outWidth_ + owBegin) * depth_;
        const DataCopyParams outputCopy0 = {
            static_cast<uint16_t>(count0),
            copyLen,
            0,
            copyLen};
        DataCopy(yGm_[outputOffset], rowLocal, outputCopy0);
        if (count1 != 0)
        {
            const DataCopyParams outputCopy1 = {
                static_cast<uint16_t>(count1),
                copyLen,
                0,
                copyLen};
            DataCopy(yGm_[outputOffset + depth_], rowLocal[count0 * depth_], outputCopy1);
        }
        if (totalTiles_ > usedCoreNum_)
        {
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void CopyBlock2TileDirectOutputFixedBlockW(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint32_t currentWidth, uint16_t copyLen, uint32_t blockW)
    {
        if (currentWidth <= 8U && totalTiles_ <= usedCoreNum_)
        {
            CopyBlock2TileViaUb(rowLocal, ob, oh, owBegin, currentWidth, copyLen);
            return;
        }
        CopyBlock2TileDirectInputFixedBlockW(
            rowLocal, ob, oh, owBegin, currentWidth, copyLen, blockW);
        PipeBarrier<PIPE_ALL>();
        CopyBlock2TileDirectOutputFromUb(rowLocal, ob, oh, owBegin, currentWidth, copyLen);
        if (totalTiles_ > usedCoreNum_)
        {
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void CopyBlock2TileDirectInputFixedBlockW(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint32_t currentWidth, uint16_t copyLen, uint32_t blockW)
    {
        const uint32_t spatialH = oh + cropTop_;
        const uint32_t ih = spatialH >> 1;
        const uint32_t blockH = spatialH & 1U;
        const uint32_t iw = (owBegin + cropLeft_) >> 1;
        const uint32_t count0 = (currentWidth + 1U) >> 1;
        const uint32_t count1 = currentWidth >> 1;
        const uint32_t ib0 = ob + outBatch_ * ((blockH << 1) + blockW);
        const uint32_t inputOffset0 = ((ib0 * height_ + ih) * width_ + iw) * depth_;
        const DataCopyParams inputCopy0 = {
            static_cast<uint16_t>(count0),
            copyLen,
            0,
            0};
        DataCopy(rowLocal, xGm_[inputOffset0], inputCopy0);
        if (count1 != 0)
        {
            const uint32_t ib1 = ob + outBatch_ * ((blockH << 1) + (blockW ^ 1U));
            const uint32_t inputOffset1 =
                ((ib1 * height_ + ih) * width_ + iw + blockW) * depth_;
            const DataCopyParams inputCopy1 = {
                static_cast<uint16_t>(count1),
                copyLen,
                0,
                0};
            DataCopy(rowLocal[count0 * depth_], xGm_[inputOffset1], inputCopy1);
        }
    }

    __aicore__ inline void CopyBlock2TileDirectOutputFromUb(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint32_t currentWidth, uint16_t copyLen)
    {
        const uint32_t count0 = (currentWidth + 1U) >> 1;
        const uint32_t count1 = currentWidth >> 1;
        const uint32_t outputOffset =
            ((ob * outHeight_ + oh) * outWidth_ + owBegin) * depth_;
        const DataCopyParams outputCopy0 = {
            static_cast<uint16_t>(count0),
            copyLen,
            0,
            copyLen};
        DataCopy(yGm_[outputOffset], rowLocal, outputCopy0);
        if (count1 != 0)
        {
            const DataCopyParams outputCopy1 = {
                static_cast<uint16_t>(count1),
                copyLen,
                0,
                copyLen};
            DataCopy(yGm_[outputOffset + depth_], rowLocal[count0 * depth_], outputCopy1);
        }
    }

    __aicore__ inline void CopyBlock4D64TileDirectOutput(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint32_t currentWidth, uint16_t copyLen)
    {
        CopyBlock4D64TileDirectInput(rowLocal, ob, oh, owBegin, currentWidth, copyLen);
        PipeBarrier<PIPE_ALL>();
        CopyBlock4D64TileDirectOutputFromUb(rowLocal, ob, oh, owBegin, currentWidth, copyLen);
        if (totalTiles_ > usedCoreNum_)
        {
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void CopyBlock4D64TileDirectInput(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint32_t currentWidth, uint16_t copyLen)
    {
        const uint32_t spatialH = oh + cropTop_;
        const uint32_t ih = spatialH >> 2;
        const uint32_t blockH = spatialH & 3U;
        const uint32_t spatialWBegin = owBegin + cropLeft_;
        uint32_t iw = spatialWBegin >> 2;
        uint32_t blockW = spatialWBegin & 3U;
        uint32_t localOffset = 0;
        for (uint32_t first = 0; first < 4U && first < currentWidth; ++first)
        {
            const uint32_t count = (currentWidth - first + 3U) >> 2;
            const uint32_t ib = ob + outBatch_ * ((blockH << 2) + blockW);
            const uint32_t inputOffset = ((ib * height_ + ih) * width_ + iw) * depth_;
            const DataCopyParams inputCopy = {
                static_cast<uint16_t>(count),
                copyLen,
                0,
                0};
            DataCopy(rowLocal[localOffset], xGm_[inputOffset], inputCopy);
            localOffset += count * depth_;
            ++blockW;
            if (blockW == 4U)
            {
                blockW = 0;
                ++iw;
            }
        }
    }

    __aicore__ inline void CopyBlock4D64TileDirectOutputFromUb(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint32_t currentWidth, uint16_t copyLen)
    {
        const uint32_t outputOffset =
            ((ob * outHeight_ + oh) * outWidth_ + owBegin) * depth_;
        const uint16_t dstStride = static_cast<uint16_t>(copyLen * 3U);
        uint32_t localOffset = 0;
        for (uint32_t first = 0; first < 4U && first < currentWidth; ++first)
        {
            const uint32_t count = (currentWidth - first + 3U) >> 2;
            const DataCopyParams outputCopy = {
                static_cast<uint16_t>(count),
                copyLen,
                0,
                dstStride};
            DataCopy(yGm_[outputOffset + first * depth_], rowLocal[localOffset], outputCopy);
            localOffset += count * depth_;
        }
    }

    __aicore__ inline void DecodeFullRow12FourRowTile(
        uint32_t tile, uint32_t &ob, uint32_t &ohBegin, uint32_t &currentRows)
    {
        ob = tile / tilesPerRow_;
        const uint32_t rowTile = tile - ob * tilesPerRow_;
        ohBegin = rowTile << 2;
        currentRows = outHeight_ - ohBegin;
        if (currentRows > 4U)
        {
            currentRows = 4U;
        }
    }

    __aicore__ inline void CopyBlock2FullRow12Output(
        LocalTensor<DT_X> rowLocal, uint32_t groupElements, uint32_t ob,
        uint32_t ohBegin, uint32_t currentRows, uint16_t copyLen)
    {
        const uint32_t outputOffset =
            (ob * outHeight_ + ohBegin) * outWidth_ * depth_;
        const DataCopyParams outputCopy = {
            static_cast<uint16_t>(currentRows * 6U),
            copyLen,
            0,
            copyLen};
        DataCopy(yGm_[outputOffset], rowLocal, outputCopy);
        DataCopy(yGm_[outputOffset + depth_], rowLocal[groupElements], outputCopy);
    }

    __aicore__ inline void CopyBlock2FullRow12FourRowInput(
        LocalTensor<DT_X> rowLocal, uint32_t groupElements, uint32_t ob,
        uint32_t ohBegin, uint32_t currentRows, uint16_t copyLen)
    {
        if (currentRows == 4U)
        {
            CopyBlock2FullRow12FourRowFullInput(rowLocal, groupElements, ob, ohBegin, copyLen);
            return;
        }
        CopyBlock2FullRow12MultirowInput(rowLocal, groupElements, 0, ob, ohBegin, copyLen);
        if (currentRows <= 1U)
        {
            return;
        }
        CopyBlock2FullRow12MultirowInput(rowLocal, groupElements, 1, ob, ohBegin + 1U, copyLen);
        if (currentRows <= 2U)
        {
            return;
        }
        CopyBlock2FullRow12MultirowInput(rowLocal, groupElements, 2, ob, ohBegin + 2U, copyLen);
        if (currentRows <= 3U)
        {
            return;
        }
        CopyBlock2FullRow12MultirowInput(rowLocal, groupElements, 3, ob, ohBegin + 3U, copyLen);
    }

    __aicore__ inline void CopyBlock2FullRow12FourRowFullInput(
        LocalTensor<DT_X> rowLocal, uint32_t groupElements, uint32_t ob,
        uint32_t ohBegin, uint16_t copyLen)
    {
        CopyBlock2FullRow12MultirowInput(rowLocal, groupElements, 0, ob, ohBegin, copyLen);
        CopyBlock2FullRow12MultirowInput(rowLocal, groupElements, 1, ob, ohBegin + 1U, copyLen);
        CopyBlock2FullRow12MultirowInput(rowLocal, groupElements, 2, ob, ohBegin + 2U, copyLen);
        CopyBlock2FullRow12MultirowInput(rowLocal, groupElements, 3, ob, ohBegin + 3U, copyLen);
    }

    __aicore__ inline void CopyBlock2FullRow12MultirowInput(
        LocalTensor<DT_X> rowLocal, uint32_t groupElements, uint32_t row,
        uint32_t ob, uint32_t oh, uint16_t copyLen)
    {
        const uint32_t spatialH = oh + cropTop_;
        const uint32_t ih = spatialH >> 1;
        const uint32_t blockH = spatialH & 1U;
        const uint32_t iw = cropLeft_ >> 1;
        const uint32_t blockW = cropLeft_ & 1U;
        const uint32_t localOffset = row * 6U * depth_;
        const DataCopyParams inputCopy = {
            6,
            copyLen,
            0,
            0};
        const uint32_t ib0 = ob + outBatch_ * ((blockH << 1) + blockW);
        const uint32_t inputOffset0 = ((ib0 * height_ + ih) * width_ + iw) * depth_;
        DataCopy(rowLocal[localOffset], xGm_[inputOffset0], inputCopy);
        const uint32_t ib1 = ob + outBatch_ * ((blockH << 1) + (blockW ^ 1U));
        const uint32_t inputOffset1 =
            ((ib1 * height_ + ih) * width_ + iw + blockW) * depth_;
        DataCopy(rowLocal[groupElements + localOffset], xGm_[inputOffset1], inputCopy);
    }

    __aicore__ inline void CopyBlock2SmallDirectFullTile(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint16_t copyLen, uint32_t blockW)
    {
        const uint32_t spatialH = oh + cropTop_;
        const uint32_t ih = spatialH >> 1;
        const uint32_t blockH = spatialH & 1U;
        const uint32_t iw = (owBegin + cropLeft_) >> 1;
        const uint32_t count = tileWidth_ >> 1;
        const uint32_t ib0 = ob + outBatch_ * ((blockH << 1) + blockW);
        const uint32_t inputOffset0 = ((ib0 * height_ + ih) * width_ + iw) * depth_;
        const DataCopyParams inputCopy0 = {
            static_cast<uint16_t>(count),
            copyLen,
            0,
            0};
        DataCopy(rowLocal, xGm_[inputOffset0], inputCopy0);
        const uint32_t ib1 = ob + outBatch_ * ((blockH << 1) + (blockW ^ 1U));
        const uint32_t inputOffset1 =
            ((ib1 * height_ + ih) * width_ + iw + blockW) * depth_;
        const DataCopyParams inputCopy1 = {
            static_cast<uint16_t>(count),
            copyLen,
            0,
            0};
        DataCopy(rowLocal[count * depth_], xGm_[inputOffset1], inputCopy1);
        PipeBarrier<PIPE_ALL>();

        const uint32_t outputOffset =
            ((ob * outHeight_ + oh) * outWidth_ + owBegin) * depth_;
        const DataCopyParams outputCopy0 = {
            static_cast<uint16_t>(count),
            copyLen,
            0,
            copyLen};
        DataCopy(yGm_[outputOffset], rowLocal, outputCopy0);
        const DataCopyParams outputCopy1 = {
            static_cast<uint16_t>(count),
            copyLen,
            0,
            copyLen};
        DataCopy(yGm_[outputOffset + depth_], rowLocal[count * depth_], outputCopy1);
        if (totalTiles_ > usedCoreNum_)
        {
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void CopyBlock2SmallDirectTailTile(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint32_t currentWidth, uint16_t copyLen)
    {
        CopyBlock2TileViaUb(rowLocal, ob, oh, owBegin, currentWidth, copyLen);
        if (totalTiles_ > usedCoreNum_)
        {
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void CopyBlock2TileViaUb(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint32_t currentWidth, uint16_t copyLen)
    {
        CopyInterleavedTileByDmaBlock2(rowLocal, ob, oh, owBegin, currentWidth);
        PipeBarrier<PIPE_ALL>();
        CopyBlock2TileViaUbOutput(rowLocal, ob, oh, owBegin, currentWidth, copyLen);
    }

    __aicore__ inline void CopyBlock2TileViaUbOutput(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t owBegin,
        uint32_t currentWidth, uint16_t copyLen)
    {
        const uint32_t outputOffset =
            ((ob * outHeight_ + oh) * outWidth_ + owBegin) * depth_;
        const DataCopyParams outputCopy = {
            1,
            static_cast<uint16_t>(currentWidth * copyLen),
            0,
            0};
        DataCopy(yGm_[outputOffset], rowLocal, outputCopy);
    }

    __aicore__ inline void CopyInterleavedFullRowByDma(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t pixelBytes)
    {
        if (blockSize_ == 2)
        {
            CopyInterleavedFullRowByDmaBlock2(rowLocal, ob, oh, pixelBytes);
            return;
        }
        const uint32_t spatialH = oh + cropTop_;
        const uint32_t ih = spatialH / blockSize_;
        const uint32_t blockH = spatialH - ih * blockSize_;
        const uint32_t firstGroupCount =
            (outWidth_ < blockSize_) ? outWidth_ : blockSize_;
        const uint32_t spatialWBegin = cropLeft_;
        const uint16_t copyLen = static_cast<uint16_t>(pixelBytes / 32);
        const uint16_t srcStride = static_cast<uint16_t>((blockSize_ - 1) * copyLen);
        uint32_t iw = spatialWBegin / blockSize_;
        uint32_t blockW = spatialWBegin - iw * blockSize_;
        for (uint32_t first = 0; first < firstGroupCount; ++first)
        {
            const uint32_t ib = ob + outBatch_ * (blockH * blockSize_ + blockW);
            const uint32_t inputOffset = ((ib * height_ + ih) * width_ + iw) * depth_;
            const uint32_t blockCount =
                (outWidth_ - first + blockSize_ - 1) / blockSize_;
            const DataCopyParams copy = {
                static_cast<uint16_t>(blockCount),
                copyLen,
                0,
                srcStride};
            DataCopy(rowLocal[first * depth_], xGm_[inputOffset], copy);
            ++blockW;
            if (blockW == blockSize_)
            {
                blockW = 0;
                ++iw;
            }
        }
    }

    __aicore__ inline void CopyInterleavedFullRowByDmaBlock2(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh, uint32_t pixelBytes)
    {
        const uint32_t spatialH = oh + cropTop_;
        const uint32_t ih = spatialH >> 1;
        const uint32_t blockH = spatialH & 1;
        const uint32_t iw = cropLeft_ >> 1;
        const uint32_t blockW = cropLeft_ & 1;
        const uint16_t copyLen = static_cast<uint16_t>(pixelBytes / 32);
        const DataCopyParams copy0 = {
            static_cast<uint16_t>((outWidth_ + 1) >> 1),
            copyLen,
            0,
            copyLen};
        const uint32_t ib0 = ob + outBatch_ * ((blockH << 1) + blockW);
        const uint32_t inputOffset0 = ((ib0 * height_ + ih) * width_ + iw) * depth_;
        DataCopy(rowLocal, xGm_[inputOffset0], copy0);
        if (outWidth_ > 1)
        {
            const uint32_t blockW1 = blockW ^ 1;
            const uint32_t iw1 = iw + blockW;
            const uint32_t ib1 = ob + outBatch_ * ((blockH << 1) + blockW1);
            const uint32_t inputOffset1 = ((ib1 * height_ + ih) * width_ + iw1) * depth_;
            const DataCopyParams copy1 = {
                static_cast<uint16_t>(outWidth_ >> 1),
                copyLen,
                0,
                copyLen};
            DataCopy(rowLocal[depth_], xGm_[inputOffset1], copy1);
        }
    }

    __aicore__ inline void CopyInterleavedFullRowChannelByDma(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh,
        uint32_t channelBegin, uint32_t currentChannels)
    {
        if (blockSize_ == 2)
        {
            CopyInterleavedFullRowChannelByDmaBlock2(
                rowLocal, ob, oh, channelBegin, currentChannels);
            return;
        }
        const uint32_t bytesPerElement = static_cast<uint32_t>(sizeof(DT_X));
        const uint32_t channelBytes = currentChannels * bytesPerElement;
        const uint16_t copyLen = static_cast<uint16_t>(channelBytes / 32);
        const uint16_t srcStride =
            static_cast<uint16_t>((depth_ - currentChannels) * bytesPerElement / 32);
        const uint16_t dstStride = static_cast<uint16_t>((blockSize_ - 1) * copyLen);
        const uint32_t spatialH = oh + cropTop_;
        const uint32_t ih = spatialH / blockSize_;
        const uint32_t blockH = spatialH - ih * blockSize_;
        const uint32_t firstGroupCount =
            (outWidth_ < blockSize_) ? outWidth_ : blockSize_;
        uint32_t iw = cropLeft_ / blockSize_;
        uint32_t blockW = cropLeft_ - iw * blockSize_;
        for (uint32_t first = 0; first < firstGroupCount; ++first)
        {
            const uint32_t ib = ob + outBatch_ * (blockH * blockSize_ + blockW);
            const uint32_t inputOffset =
                ((ib * height_ + ih) * width_ + iw) * depth_ + channelBegin;
            const uint32_t blockCount =
                (outWidth_ - first + blockSize_ - 1) / blockSize_;
            const DataCopyParams copy = {
                static_cast<uint16_t>(blockCount),
                copyLen,
                srcStride,
                dstStride};
            DataCopy(rowLocal[first * currentChannels], xGm_[inputOffset], copy);
            ++blockW;
            if (blockW == blockSize_)
            {
                blockW = 0;
                ++iw;
            }
        }
    }

    __aicore__ inline void CopyInterleavedFullRowChannelByDmaBlock2(
        LocalTensor<DT_X> rowLocal, uint32_t ob, uint32_t oh,
        uint32_t channelBegin, uint32_t currentChannels)
    {
        const uint32_t bytesPerElement = static_cast<uint32_t>(sizeof(DT_X));
        const uint32_t channelBytes = currentChannels * bytesPerElement;
        const uint16_t copyLen = static_cast<uint16_t>(channelBytes / 32);
        const uint16_t srcStride =
            static_cast<uint16_t>((depth_ - currentChannels) * bytesPerElement / 32);
        const uint16_t dstStride = copyLen;
        const uint32_t spatialH = oh + cropTop_;
        const uint32_t ih = spatialH >> 1;
        const uint32_t blockH = spatialH & 1U;
        const uint32_t iw = cropLeft_ >> 1;
        const uint32_t blockW = cropLeft_ & 1U;
        const uint32_t ib0 = ob + outBatch_ * ((blockH << 1) + blockW);
        const uint32_t inputOffset0 =
            ((ib0 * height_ + ih) * width_ + iw) * depth_ + channelBegin;
        const DataCopyParams copy0 = {
            static_cast<uint16_t>((outWidth_ + 1U) >> 1),
            copyLen,
            srcStride,
            dstStride};
        DataCopy(rowLocal, xGm_[inputOffset0], copy0);
        if (outWidth_ <= 1U)
        {
            return;
        }
        const uint32_t blockW1 = blockW ^ 1U;
        const uint32_t iw1 = iw + blockW;
        const uint32_t ib1 = ob + outBatch_ * ((blockH << 1) + blockW1);
        const uint32_t inputOffset1 =
            ((ib1 * height_ + ih) * width_ + iw1) * depth_ + channelBegin;
        const DataCopyParams copy1 = {
            static_cast<uint16_t>(outWidth_ >> 1),
            copyLen,
            srcStride,
            dstStride};
        DataCopy(rowLocal[currentChannels], xGm_[inputOffset1], copy1);
    }

    __aicore__ inline void CopyGroupedInterleavedOutputDirect(
        LocalTensor<DT_X> inputLocal, uint32_t ob, uint32_t oh,
        uint32_t owBegin, uint32_t currentWidth, uint32_t pixelBytes)
    {
        if (BTS_OPT_BLOCK2_DEEP_DIRECT_OUTPUT_SINGLE &&
            blockSize_ == 2U && sizeof(DT_X) == 2U &&
            pixelBytes > 128U && outWidth_ > 64U && outWidth_ <= 256U &&
            outHeight_ > 64U)
        {
            CopyGroupedInterleavedOutputDirectSingle(
                inputLocal, ob, oh, owBegin, currentWidth, pixelBytes);
            return;
        }
        const uint32_t firstGroupCount =
            (currentWidth < blockSize_) ? currentWidth : blockSize_;
        const uint32_t dstStride = (blockSize_ - 1U) * pixelBytes;
        const uint32_t outputBase =
            ((ob * outHeight_ + oh) * outWidth_ + owBegin) * depth_;
        for (uint32_t first = 0; first < firstGroupCount; ++first)
        {
            const uint32_t blockCount =
                (currentWidth - first + blockSize_ - 1U) / blockSize_;
            const DataCopyExtParams copy = {
                static_cast<uint16_t>(blockCount),
                pixelBytes,
                0,
                dstStride,
                0};
            DataCopyPad(
                yGm_[outputBase + first * depth_],
                inputLocal[first * inputGroupStrideElements_],
                copy);
        }
    }

    __aicore__ inline void CopyGroupedInterleavedOutputDirectSingle(
        LocalTensor<DT_X> inputLocal, uint32_t ob, uint32_t oh,
        uint32_t owBegin, uint32_t currentWidth, uint32_t pixelBytes)
    {
        const uint32_t outputBase =
            ((ob * outHeight_ + oh) * outWidth_ + owBegin) * depth_;
        const DataCopyExtParams copy = {
            1,
            pixelBytes,
            0,
            0,
            0};
        const uint32_t count0 = (currentWidth + 1U) >> 1;
        for (uint32_t idx = 0U; idx < count0; ++idx)
        {
            DataCopyPad(
                yGm_[outputBase + (idx << 1) * depth_],
                inputLocal[idx * depth_],
                copy);
        }
        if (currentWidth <= 1U)
        {
            return;
        }
        const uint32_t count1 = currentWidth >> 1;
        LocalTensor<DT_X> inputLocal1 = inputLocal[inputGroupStrideElements_];
        for (uint32_t idx = 0U; idx < count1; ++idx)
        {
            DataCopyPad(
                yGm_[outputBase + ((idx << 1) + 1U) * depth_],
                inputLocal1[idx * depth_],
                copy);
        }
    }

    __aicore__ inline void CopyGroupedInterleavedTileByDma(
        LocalTensor<DT_X> inputLocal, LocalTensor<DT_X> rowLocal, uint32_t ob,
        uint32_t oh, uint32_t owBegin, uint32_t currentWidth)
    {
        const uint32_t pixelBytes = depth_ * static_cast<uint32_t>(sizeof(DT_X));
        CopyGroupedInterleavedInputByDma(inputLocal, ob, oh, owBegin, currentWidth);
        PipeBarrier<PIPE_ALL>();
        GatherGroupedInput(inputLocal, rowLocal, currentWidth, pixelBytes, false);
    }

    __aicore__ inline void CopyGroupedInterleavedInputByDma(
        LocalTensor<DT_X> inputLocal, uint32_t ob, uint32_t oh,
        uint32_t owBegin, uint32_t currentWidth)
    {
        if ((BTS_OPT_BLOCK2_GROUPED_INPUT_FAST ||
             MODE == MODE_GROUPED_GENERAL_PACK2) &&
            blockSize_ == 2U)
        {
            CopyGroupedInterleavedInputByDmaBlock2(
                inputLocal, ob, oh, owBegin, currentWidth);
            return;
        }
        const uint32_t pixelBytes = depth_ * static_cast<uint32_t>(sizeof(DT_X));
        const uint32_t spatialH = oh + cropTop_;
        const DataCopyPadExtParams<DT_X> pad = {false, 0, 0, 0};
        const uint32_t firstGroupCount =
            (currentWidth < blockSize_) ? currentWidth : blockSize_;
        const uint32_t spatialWBegin = owBegin + cropLeft_;
        const uint32_t ih = spatialH / blockSize_;
        const uint32_t blockH = spatialH - ih * blockSize_;
        uint32_t iw = spatialWBegin / blockSize_;
        uint32_t blockW = spatialWBegin - iw * blockSize_;
        for (uint32_t first = 0; first < firstGroupCount; ++first)
        {
            const uint32_t ib = ob + outBatch_ * (blockH * blockSize_ + blockW);
            const uint32_t inputOffset = ((ib * height_ + ih) * width_ + iw) * depth_;
            const uint32_t blockCount =
                (currentWidth - first + blockSize_ - 1) / blockSize_;
            const uint32_t copyBytes = blockCount * pixelBytes;
            const uint32_t inputOffsetBytes =
                inputOffset * static_cast<uint32_t>(sizeof(DT_X));
            if (inputOffsetBytes % 32 == 0 && copyBytes % 32 == 0)
            {
                const DataCopyParams copy = {
                    1,
                    static_cast<uint16_t>(copyBytes / 32),
                    0,
                    0};
                DataCopy(inputLocal[first * inputGroupStrideElements_], xGm_[inputOffset], copy);
            }
            else
            {
                const DataCopyExtParams copy = {
                    1,
                    copyBytes,
                    0,
                    0,
                    0};
                DataCopyPad(
                    inputLocal[first * inputGroupStrideElements_], xGm_[inputOffset], copy, pad);
            }
            ++blockW;
            if (blockW == blockSize_)
            {
                blockW = 0;
                ++iw;
            }
        }
    }

    __aicore__ inline void CopyGroupedInputContiguous(
        LocalTensor<DT_X> dstLocal, uint32_t inputOffset, uint32_t copyBytes)
    {
        const uint32_t inputOffsetBytes =
            inputOffset * static_cast<uint32_t>(sizeof(DT_X));
        if ((inputOffsetBytes & 31U) == 0U && (copyBytes & 31U) == 0U)
        {
            const DataCopyParams copy = {
                1,
                static_cast<uint16_t>(copyBytes >> 5),
                0,
                0};
            DataCopy(dstLocal, xGm_[inputOffset], copy);
            return;
        }
        const DataCopyExtParams copy = {
            1,
            copyBytes,
            0,
            0,
            0};
        const DataCopyPadExtParams<DT_X> pad = {false, 0, 0, 0};
        DataCopyPad(dstLocal, xGm_[inputOffset], copy, pad);
    }

    __aicore__ inline void CopyGroupedInterleavedInputByDmaBlock2(
        LocalTensor<DT_X> inputLocal, uint32_t ob, uint32_t oh,
        uint32_t owBegin, uint32_t currentWidth)
    {
        const uint32_t pixelBytes = depth_ * static_cast<uint32_t>(sizeof(DT_X));
        const uint32_t spatialH = oh + cropTop_;
        const uint32_t ih = spatialH >> 1;
        const uint32_t blockH = spatialH & 1U;
        const uint32_t spatialWBegin = owBegin + cropLeft_;
        const uint32_t iw = spatialWBegin >> 1;
        const uint32_t blockW = spatialWBegin & 1U;
        const uint32_t count0 = (currentWidth + 1U) >> 1;
        const uint32_t ib0 = ob + outBatch_ * ((blockH << 1) + blockW);
        const uint32_t inputOffset0 = ((ib0 * height_ + ih) * width_ + iw) * depth_;
        CopyGroupedInputContiguous(inputLocal, inputOffset0, count0 * pixelBytes);
        if (currentWidth <= 1U)
        {
            return;
        }

        const uint32_t blockW1 = blockW ^ 1U;
        const uint32_t iw1 = iw + blockW;
        const uint32_t ib1 = ob + outBatch_ * ((blockH << 1) + blockW1);
        const uint32_t inputOffset1 = ((ib1 * height_ + ih) * width_ + iw1) * depth_;
        const uint32_t count1 = currentWidth >> 1;
        CopyGroupedInputContiguous(
            inputLocal[inputGroupStrideElements_], inputOffset1, count1 * pixelBytes);
    }

    __aicore__ inline void GatherGroupedInput(
        LocalTensor<DT_X> inputLocal, LocalTensor<DT_X> rowLocal,
        uint32_t currentWidth, uint32_t pixelBytes, bool useSingleGather)
    {
        LocalTensor<uint32_t> offsets = offsetBuffer_.Get<uint32_t>();
        if (useSingleGather || gatherTemplateWidth_ >= currentWidth)
        {
            Gather(rowLocal, inputLocal, offsets, 0, currentWidth * depth_);
            return;
        }
        const uint32_t baseAddrStep = (gatherTemplateWidth_ / blockSize_) * pixelBytes;
        uint32_t baseAddr = 0;
        for (uint32_t start = 0; start < currentWidth; start += gatherTemplateWidth_)
        {
            uint32_t width = currentWidth - start;
            if (width > gatherTemplateWidth_)
            {
                width = gatherTemplateWidth_;
            }
            Gather(rowLocal[start * depth_], inputLocal, offsets, baseAddr, width * depth_);
            baseAddr += baseAddrStep;
        }
    }

    __aicore__ inline void BuildGroupedGatherOffsets()
    {
        if ((BTS_OPT_BLOCK2_GROUPED_OFFSET_FAST ||
             MODE == MODE_GROUPED_GENERAL_PACK2) &&
            blockSize_ == 2U)
        {
            BuildGroupedGatherOffsetsBlock2();
            return;
        }
        LocalTensor<uint32_t> offsets = offsetBuffer_.Get<uint32_t>();
        const uint32_t elementBytes = static_cast<uint32_t>(sizeof(DT_X));
        const uint32_t groupCount =
            (gatherTemplateWidth_ < blockSize_) ? gatherTemplateWidth_ : blockSize_;
        for (uint32_t group = 0; group < groupCount; ++group)
        {
            uint32_t srcElement = group * inputGroupStrideElements_;
            for (uint32_t localOw = group; localOw < gatherTemplateWidth_; localOw += blockSize_)
            {
                const uint32_t dstElement = localOw * depth_;
                for (uint32_t od = 0; od < depth_; ++od)
                {
                    offsets.SetValue(dstElement + od, (srcElement + od) * elementBytes);
                }
                srcElement += depth_;
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void BuildGroupedGatherOffsetsBlock2()
    {
        LocalTensor<uint32_t> offsets = offsetBuffer_.Get<uint32_t>();
        const uint32_t elementBytes = static_cast<uint32_t>(sizeof(DT_X));
        uint32_t srcElement = 0U;
        for (uint32_t localOw = 0U; localOw < gatherTemplateWidth_; localOw += 2U)
        {
            const uint32_t dstElement = localOw * depth_;
            for (uint32_t od = 0U; od < depth_; ++od)
            {
                offsets.SetValue(dstElement + od, (srcElement + od) * elementBytes);
            }
            srcElement += depth_;
        }
        if (gatherTemplateWidth_ > 1U)
        {
            srcElement = inputGroupStrideElements_;
            for (uint32_t localOw = 1U; localOw < gatherTemplateWidth_; localOw += 2U)
            {
                const uint32_t dstElement = localOw * depth_;
                for (uint32_t od = 0U; od < depth_; ++od)
                {
                    offsets.SetValue(dstElement + od, (srcElement + od) * elementBytes);
                }
                srcElement += depth_;
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void GatherInterleavedTile(LocalTensor<DT_X> rowLocal, uint32_t ob,
                                                 uint32_t oh, uint32_t owBegin,
                                                 uint32_t currentWidth)
    {
        if (blockSize_ == 2)
        {
            GatherInterleavedTileBlock2(rowLocal, ob, oh, owBegin, currentWidth);
            return;
        }
        const uint32_t spatialH = oh + cropTop_;
        const uint32_t ih = spatialH / blockSize_;
        const uint32_t blockH = spatialH - ih * blockSize_;
        const uint32_t spatialWBegin = owBegin + cropLeft_;
        uint32_t iw = spatialWBegin / blockSize_;
        uint32_t blockW = spatialWBegin - iw * blockSize_;
        for (uint32_t localOw = 0; localOw < currentWidth; ++localOw)
        {
            const uint32_t ib = ob + outBatch_ * (blockH * blockSize_ + blockW);
            const uint32_t inputOffset = ((ib * height_ + ih) * width_ + iw) * depth_;
            const uint32_t localOffset = localOw * depth_;
            for (uint32_t od = 0; od < depth_; ++od)
            {
                rowLocal.SetValue(localOffset + od, xGm_.GetValue(inputOffset + od));
            }
            ++blockW;
            if (blockW == blockSize_)
            {
                blockW = 0;
                ++iw;
            }
        }
    }

    __aicore__ inline void GatherInterleavedTileBlock2(LocalTensor<DT_X> rowLocal, uint32_t ob,
                                                       uint32_t oh, uint32_t owBegin,
                                                       uint32_t currentWidth)
    {
        const uint32_t spatialH = oh + cropTop_;
        const uint32_t ih = spatialH >> 1;
        const uint32_t blockH = spatialH & 1;
        const uint32_t spatialWBegin = owBegin + cropLeft_;
        const uint32_t iw = spatialWBegin >> 1;
        const uint32_t blockW = spatialWBegin & 1;
        const uint32_t count0 = (currentWidth + 1U) >> 1;
        const uint32_t count1 = currentWidth >> 1;
        const uint32_t ib0 = ob + outBatch_ * ((blockH << 1) + blockW);
        uint32_t inputOffset = ((ib0 * height_ + ih) * width_ + iw) * depth_;
        uint32_t localOffset = 0;
        for (uint32_t pixel = 0; pixel < count0; ++pixel)
        {
            for (uint32_t od = 0; od < depth_; ++od)
            {
                rowLocal.SetValue(localOffset + od, xGm_.GetValue(inputOffset + od));
            }
            inputOffset += depth_;
            localOffset += depth_ << 1;
        }
        if (count1 == 0)
        {
            return;
        }
        const uint32_t blockW1 = blockW ^ 1U;
        const uint32_t iw1 = iw + blockW;
        const uint32_t ib1 = ob + outBatch_ * ((blockH << 1) + blockW1);
        inputOffset = ((ib1 * height_ + ih) * width_ + iw1) * depth_;
        localOffset = depth_;
        for (uint32_t pixel = 0; pixel < count1; ++pixel)
        {
            for (uint32_t od = 0; od < depth_; ++od)
            {
                rowLocal.SetValue(localOffset + od, xGm_.GetValue(inputOffset + od));
            }
            inputOffset += depth_;
            localOffset += depth_ << 1;
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> rowBuffer_;
    TBuf<TPosition::VECCALC> inputBuffer_;
    TBuf<TPosition::VECCALC> offsetBuffer_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t batch_;
    uint32_t height_;
    uint32_t width_;
    uint32_t depth_;
    uint32_t outBatch_;
    uint32_t outHeight_;
    uint32_t outWidth_;
    uint32_t cropTop_;
    uint32_t cropLeft_;
    uint32_t blockSize_;
    uint32_t totalOutputElements_;
    uint32_t tileWidth_;
    uint32_t tilesPerRow_;
    uint32_t totalTiles_;
    uint32_t tileBufferBytes_;
    uint32_t inputGroupStrideElements_;
    uint32_t inputTileBufferBytes_;
    uint32_t offsetBufferBytes_;
    uint32_t gatherTemplateWidth_;
    uint32_t rowTileHeight_;
    uint32_t flatRowMerge_;
    uint32_t linearInputOffset_;
    uint32_t usedCoreNum_;
};

extern "C" __global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingData, tilingData, tiling);
    if (TILING_KEY_IS(1))
    {
        KernelBatchToSpace<half, MODE_LINEAR> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(2))
    {
        KernelBatchToSpace<float, MODE_LINEAR> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(3))
    {
        KernelBatchToSpace<half, MODE_FLAT> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(4))
    {
        KernelBatchToSpace<float, MODE_FLAT> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(5))
    {
        KernelBatchToSpace<half, MODE_CHANNEL> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(6))
    {
        KernelBatchToSpace<float, MODE_CHANNEL> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(7))
    {
        KernelBatchToSpace<half, MODE_ELEMENTWISE> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(8))
    {
        KernelBatchToSpace<float, MODE_ELEMENTWISE> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(9))
    {
        KernelBatchToSpace<half, MODE_ROW> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(10))
    {
        KernelBatchToSpace<float, MODE_ROW> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(11))
    {
        KernelBatchToSpace<half, MODE_ALIGNED> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(12))
    {
        KernelBatchToSpace<float, MODE_ALIGNED> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(13))
    {
        KernelBatchToSpace<half, MODE_GENERAL> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(14))
    {
        KernelBatchToSpace<float, MODE_GENERAL> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(15))
    {
        KernelBatchToSpace<half, MODE_ROW_WIDE> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(16))
    {
        KernelBatchToSpace<float, MODE_ROW_WIDE> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(19))
    {
        KernelBatchToSpace<half, MODE_ROW_OTHER> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(20))
    {
        KernelBatchToSpace<float, MODE_ROW_OTHER> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(21))
    {
        KernelBatchToSpace<half, MODE_GENERAL_DIRECT_DMA> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(22))
    {
        KernelBatchToSpace<float, MODE_GENERAL_DIRECT_DMA> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(23))
    {
        KernelBatchToSpace<half, MODE_GROUPED_GENERAL> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(24))
    {
        KernelBatchToSpace<float, MODE_GROUPED_GENERAL> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(25))
    {
        KernelBatchToSpace<half, MODE_ALIGNED_INTERLEAVED> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(26))
    {
        KernelBatchToSpace<float, MODE_ALIGNED_INTERLEAVED> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(27))
    {
        KernelBatchToSpace<half, MODE_ALIGNED_BLOCK2_DIRECT> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(28))
    {
        KernelBatchToSpace<float, MODE_ALIGNED_BLOCK2_DIRECT> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(29))
    {
        KernelBatchToSpace<half, MODE_ALIGNED_BLOCK2_ROW_COMPACT> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(30))
    {
        KernelBatchToSpace<float, MODE_ALIGNED_BLOCK2_ROW_COMPACT> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(31))
    {
        KernelBatchToSpace<half, MODE_ALIGNED_BLOCK2_TINY_UB_DB> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(32))
    {
        KernelBatchToSpace<float, MODE_ALIGNED_BLOCK2_TINY_UB_DB> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(33))
    {
        KernelBatchToSpace<half, MODE_ALIGNED_BLOCK2_SMALL_DIRECT> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(34))
    {
        KernelBatchToSpace<float, MODE_ALIGNED_BLOCK2_SMALL_DIRECT> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(35))
    {
        KernelBatchToSpace<half, MODE_ALIGNED_BLOCK2_FULL_ROW12_MULTIROW_DIRECT> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(36))
    {
        KernelBatchToSpace<float, MODE_ALIGNED_BLOCK2_FULL_ROW12_MULTIROW_DIRECT> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(37))
    {
        KernelBatchToSpace<half, MODE_ALIGNED_BLOCK4_D64_DIRECT> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(38))
    {
        KernelBatchToSpace<float, MODE_ALIGNED_BLOCK4_D64_DIRECT> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(39))
    {
        KernelBatchToSpace<half, MODE_ALIGNED_BLOCK4_D64_PHASE_FAST> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(40))
    {
        KernelBatchToSpace<float, MODE_ALIGNED_BLOCK4_D64_PHASE_FAST> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(41))
    {
        KernelBatchToSpace<half, MODE_ALIGNED_BLOCK4_D64_UB_REORDER> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(42))
    {
        KernelBatchToSpace<float, MODE_GROUPED_GENERAL_PACK2> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
}
