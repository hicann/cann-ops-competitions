#include "kernel_operator.h"
#include "batch_to_space_tiling.h"

namespace {
// V2_43C_pairBatch320_chunk640: test9 pairBatch320 = 640 output points, coalesced pad-read/no-DB
// V2_1 tuning interface: keep grouped DMA controls at the front.
// V89 keeps V85 chunks and uses a larger MTE2 wait cadence only for small chunks.
// No-crop and 32B-aligned crop paths still collect 16 output points per writeback.
constexpr uint32_t FAST_COPY_POINT_NUM = 16;
constexpr uint32_t SMALL_UNALIGNED_WAIT_POINT_NUM = 32;
constexpr uint32_t SMALL_UNALIGNED_CHUNK_GROUP_NUM = 4;
// Unaligned crop path uses cropChunkPointNum from host: it is a multiple that makes
// cropChunkPointNum * depthBytes exactly 32B aligned. Last remainder uses DataCopyPad.

constexpr uint32_t MODE_NO_CROP = 0;
constexpr uint32_t MODE_ROW_SCALAR = 1;
constexpr uint32_t MODE_ALIGNED_CHUNK = 2;
constexpr uint32_t MODE_BYTE_BLOCK = 3;
constexpr uint32_t MODE_ROW_ALIGNED_CHUNK = 4;
constexpr uint32_t MODE_ROW_PARALLEL_CHUNK = 5;
constexpr uint32_t MODE_UNALIGNED_FLAT_DMA = 6;
constexpr uint32_t MODE_DIRECT_BYTE_COPY = 7;
constexpr uint32_t MODE_BLOCK1_CROP_SLICE = 8;
constexpr uint32_t MODE_ROW_SEGMENT_DMA = 9;
constexpr uint32_t MODE_NO_CROP_GROUP_MEDIUM = 10;
constexpr uint32_t MODE_NO_CROP_GROUP_LARGE = 11;
constexpr uint32_t MODE_NO_CROP_DEPTH_SPLIT = 12;
constexpr uint32_t MODE_ROW_SEGMENT_DMA_DB = 13;
constexpr uint32_t MODE_NO_CROP_GROUP_LARGE_DB = 14;
constexpr uint32_t MODE_NO_CROP_BLOCK2_MTE2_INTERLEAVE_DB = 15; // V36: test10 MTE2 writes directly into UB interleaved layout
constexpr uint32_t MODE_ROW_SEGMENT_COALESCED_PADREAD = 16; // V41A: test9 GM reads interleaved into UB, continuous GM write
constexpr uint32_t MODE_NO_CROP_BLOCK2_DEPTH128_FAST = 17; // V58: exact test4-like light no-crop path
constexpr uint32_t MODE_CROP_BLOCK2_GATHER = 18; // V93: block2 non-32B crop, phase DMA + UB Gather
constexpr uint32_t MODE_TEST3_TYPED_STRIDE_MERGE = 19; // V102: V101-hit aligned block2 typed stride merge
constexpr uint32_t MODE_ROW_SEGMENT_BLOCKN_COALESCED = 20; // V130: test9 dedicated blockSize>=3 aligned row-segment, continuous GM write
constexpr uint32_t MODE_ROW_SEGMENT_BLOCKN_COALESCED_DB = 21; // V130: DB variant

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

__aicore__ inline uint32_t AlignUp32(uint32_t x)
{
    return (x + 31) & ~static_cast<uint32_t>(31);
}

__aicore__ inline uint32_t AlignUpU32(uint32_t value, uint32_t align)
{
    return ((value + align - 1) / align) * align;
}

__aicore__ inline uint32_t GcdU32(uint32_t a, uint32_t b)
{
    while (b != 0) {
        const uint32_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}
}

class KernelBatchToSpace {
public:
    __aicore__ inline KernelBatchToSpace() {}

    __aicore__ inline void Init(AscendC::TPipe *pipePtr, GM_ADDR x, GM_ADDR y,
        const BatchToSpaceTilingData &tilingData)
    {
        this->pipe = pipePtr;
        InitCoreWork(tilingData);
        InitShape(tilingData);

        const uint32_t inputBytes =
            this->batch * this->inputHeight * this->inputWidth * this->depthBytes;
        const uint32_t outputBytes =
            this->outBatch * this->outHeight * this->outWidth * this->depthBytes;

        xGm.SetGlobalBuffer((__gm__ int8_t *)x, inputBytes);
        yGm.SetGlobalBuffer((__gm__ int8_t *)y, outputBytes);
        xGm16.SetGlobalBuffer((__gm__ uint16_t *)x, (inputBytes + 1) / 2);
        yGm16.SetGlobalBuffer((__gm__ uint16_t *)y, (outputBytes + 1) / 2);
        xGm32.SetGlobalBuffer((__gm__ uint32_t *)x, (inputBytes + 3) / 4);
        yGm32.SetGlobalBuffer((__gm__ uint32_t *)y, (outputBytes + 3) / 4);
        const uint32_t inputElementNum =
            this->batch * this->inputHeight * this->inputWidth * this->depth;
        const uint32_t outputElementNum =
            this->outBatch * this->outHeight * this->outWidth * this->depth;
        xGmFp16.SetGlobalBuffer((__gm__ half *)x, inputElementNum);
        yGmFp16.SetGlobalBuffer((__gm__ half *)y, outputElementNum);
        xGmFp32.SetGlobalBuffer((__gm__ float *)x, inputElementNum);
        yGmFp32.SetGlobalBuffer((__gm__ float *)y, outputElementNum);

        xGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
        yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
        xGm16.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
        yGm16.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
        xGm32.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
        yGm32.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
        xGmFp16.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
        yGmFp16.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
        xGmFp32.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
        yGmFp32.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);

        pipe->InitBuffer(yBuf, this->tilePointNum * this->depthBytes);
        if (this->cropWorkMode == MODE_UNALIGNED_FLAT_DMA ||
            this->cropWorkMode == MODE_ROW_SEGMENT_DMA_DB ||
            this->cropWorkMode == MODE_ROW_SEGMENT_COALESCED_PADREAD ||
            this->cropWorkMode == MODE_ROW_SEGMENT_BLOCKN_COALESCED_DB ||
            this->cropWorkMode == MODE_NO_CROP_GROUP_LARGE_DB ||
            this->cropWorkMode == MODE_NO_CROP_BLOCK2_MTE2_INTERLEAVE_DB ||
            this->cropWorkMode == MODE_CROP_BLOCK2_GATHER ||
            this->cropWorkMode == MODE_TEST3_TYPED_STRIDE_MERGE) {
            pipe->InitBuffer(yBufDb, this->tilePointNum * this->depthBytes);
        }
        eventIdMte2ToV = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V);
        eventIdVToMte3 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3);
        eventIdMte3ToMte2 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2);
        if (this->cropWorkMode == MODE_TEST3_TYPED_STRIDE_MERGE) {
            eventIdMte2ToMte3 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3);
        } else {
            eventIdMte2ToMte3 = eventIdMte2ToV;
        }
        // V2_22: DB event is only needed by DB modes.  Short no-crop cases
        // such as test6 are very sensitive to setup overhead, so do not fetch
        // the second MTE3->MTE2 event on the ordinary no-crop path.
        if (this->cropWorkMode == MODE_ROW_SEGMENT_DMA_DB ||
            this->cropWorkMode == MODE_ROW_SEGMENT_COALESCED_PADREAD ||
            this->cropWorkMode == MODE_ROW_SEGMENT_BLOCKN_COALESCED_DB ||
            this->cropWorkMode == MODE_NO_CROP_GROUP_LARGE_DB ||
            this->cropWorkMode == MODE_NO_CROP_BLOCK2_MTE2_INTERLEAVE_DB ||
            this->cropWorkMode == MODE_CROP_BLOCK2_GATHER ||
            this->cropWorkMode == MODE_TEST3_TYPED_STRIDE_MERGE) {
            eventIdMte3ToMte2Db = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2);
        } else {
            eventIdMte3ToMte2Db = eventIdMte3ToMte2;
        }
    }

    __aicore__ inline void Process()
    {
        if (this->corePointNum == 0) {
            return;
        }

        if (this->cropWorkMode == MODE_NO_CROP_BLOCK2_DEPTH128_FAST) {
            CopyNoCropBlock2Depth128Fast(this->startPoint, this->corePointNum);
            return;
        }
        if (this->cropWorkMode == MODE_DIRECT_BYTE_COPY) {
            CopyDirectByteChunks(this->startPoint, this->corePointNum);
            return;
        }
        if (this->cropWorkMode == MODE_NO_CROP_DEPTH_SPLIT) {
            CopyNoCropDepthSplitChunks(this->startPoint, this->corePointNum);
            return;
        }
        if (this->cropWorkMode == MODE_NO_CROP_GROUP_MEDIUM ||
            this->cropWorkMode == MODE_NO_CROP_GROUP_LARGE ||
            this->cropWorkMode == MODE_NO_CROP_GROUP_LARGE_DB ||
            this->cropWorkMode == MODE_NO_CROP_BLOCK2_MTE2_INTERLEAVE_DB) {
            DispatchNoCropGroupedChunks();
            return;
        }
        if (this->cropWorkMode == MODE_ROW_SEGMENT_BLOCKN_COALESCED_DB) {
            CopyRowSegmentBlockNCoalescedAlignedChunksDb(this->startPoint, this->corePointNum);
            return;
        }
        if (this->cropWorkMode == MODE_ROW_SEGMENT_BLOCKN_COALESCED) {
            CopyRowSegmentBlockNCoalescedAlignedChunks(this->startPoint, this->corePointNum);
            return;
        }
        if (this->cropWorkMode == MODE_ROW_SEGMENT_COALESCED_PADREAD) {
            CopyRowSegmentCoalescedPadReadChunksDb(this->startPoint, this->corePointNum);
            return;
        }
        if (this->cropWorkMode == MODE_TEST3_TYPED_STRIDE_MERGE) {
            DispatchTest3TypedStrideMerge();
            return;
        }
        if (this->cropWorkMode == MODE_ROW_SEGMENT_DMA_DB) {
            CopyRowSegmentDmaChunksDb(this->startPoint, this->corePointNum);
            return;
        }
        if (this->cropWorkMode == MODE_ROW_SEGMENT_DMA) {
            CopyRowSegmentDmaChunks(this->startPoint, this->corePointNum);
            return;
        }
        if (this->cropWorkMode == MODE_CROP_BLOCK2_GATHER) {
            DispatchCropBlock2Gather();
            return;
        }

        if (IsNoCrop()) {
            DispatchNoCrop();
            return;
        }

        switch (this->cropWorkMode) {
            case MODE_ALIGNED_CHUNK:
                DispatchAlignedChunk();
                break;
            case MODE_BYTE_BLOCK:
                CopyCropByteBlocks(this->startPoint, this->corePointNum);
                break;
            case MODE_ROW_ALIGNED_CHUNK:
                DispatchRowAlignedChunks();
                break;
            case MODE_ROW_PARALLEL_CHUNK:
                DispatchRowParallelChunks();
                break;
            case MODE_UNALIGNED_FLAT_DMA:
                CopyCropUnalignedFlatDmaChunks(this->startPoint, this->corePointNum);
                break;
            case MODE_BLOCK1_CROP_SLICE:
                CopyBlock1CropSliceChunks(this->startPoint, this->corePointNum);
                break;
            case MODE_ROW_SCALAR:
            default:
                DispatchRows();
                break;
        }
    }

private:
    struct RowContext {
        uint32_t ob;
        uint32_t oh;
        uint32_t rowBasePoint;
        uint32_t blockH;
        uint32_t ih;
        uint32_t cropLeftMod;
    };

    __aicore__ inline void InitCoreWork(const BatchToSpaceTilingData &tilingData)
    {
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        if (coreIdx < tilingData.tailBlockNum) {
            this->corePointNum = tilingData.bigCorePointNum;
            this->startPoint = coreIdx * tilingData.bigCorePointNum;
            this->tileNum = tilingData.finalBigTileNum;
            this->tailPointNum = tilingData.bigTailPointNum;
        } else {
            this->corePointNum = tilingData.smallCorePointNum;
            this->startPoint = tilingData.tailBlockNum * tilingData.bigCorePointNum +
                (coreIdx - tilingData.tailBlockNum) * tilingData.smallCorePointNum;
            this->tileNum = tilingData.finalSmallTileNum;
            this->tailPointNum = tilingData.smallTailPointNum;
        }
        this->tilePointNum = tilingData.tilePointNum;
    }

    __aicore__ inline void InitShape(const BatchToSpaceTilingData &tilingData)
    {
        this->batch = tilingData.batch;
        this->inputHeight = tilingData.inputHeight;
        this->inputWidth = tilingData.inputWidth;
        this->depth = tilingData.depth;
        this->typeLength = tilingData.typeLength;
        this->depthBytes = this->depth * this->typeLength;

        this->outBatch = tilingData.outBatch;
        this->outHeight = tilingData.outHeight;
        this->outWidth = tilingData.outWidth;

        this->blockSize = tilingData.blockSize;
        this->cropTop = tilingData.cropTop;
        this->cropBottom = tilingData.cropBottom;
        this->cropLeft = tilingData.cropLeft;
        this->cropRight = tilingData.cropRight;
        this->cropWorkMode = tilingData.cropWorkMode;
        this->cropChunkPointNum = tilingData.cropChunkPointNum;
        this->cropChunkNumPerRow = tilingData.cropChunkNumPerRow;
    }

    __aicore__ inline bool IsNoCrop() const
    {
        return this->cropTop == 0 && this->cropBottom == 0 &&
               this->cropLeft == 0 && this->cropRight == 0;
    }

    __aicore__ inline uint32_t GetOutputAlignPointStep() const
    {
        if ((this->depthBytes & 31) == 0) {
            return 1;
        }
        if ((this->depthBytes & 15) == 0) {
            return 2;
        }
        if ((this->depthBytes & 7) == 0) {
            return 4;
        }
        if ((this->depthBytes & 3) == 0) {
            return 8;
        }
        if ((this->depthBytes & 1) == 0) {
            return 16;
        }
        return 32;
    }

    __aicore__ inline uint32_t FindFirstAlignedOw(uint32_t rowBasePoint, uint32_t step) const
    {
        for (uint32_t ow = 0; ow < step && ow < this->outWidth; ++ow) {
            if ((((rowBasePoint + ow) * this->depthBytes) & 31) == 0) {
                return ow;
            }
        }
        return this->outWidth;
    }

    __aicore__ inline uint32_t FirstOwForBlockInRange(
        uint32_t blockW, uint32_t cropLeftMod, uint32_t owBegin) const
    {
        uint32_t ow = blockW >= cropLeftMod ?
            (blockW - cropLeftMod) : (blockW + this->blockSize - cropLeftMod);
        if (ow < owBegin) {
            const uint32_t delta = owBegin - ow;
            ow += ((delta + this->blockSize - 1) / this->blockSize) * this->blockSize;
        }
        return ow;
    }

    __aicore__ inline RowContext GetRowContext(uint32_t outputRow) const
    {
        RowContext row;
        row.ob = outputRow / this->outHeight;
        row.oh = outputRow - row.ob * this->outHeight;
        row.rowBasePoint = outputRow * this->outWidth;
        const uint32_t fullH = row.oh + this->cropTop;
        row.blockH = fullH % this->blockSize;
        row.ih = fullH / this->blockSize;
        row.cropLeftMod = this->cropLeft % this->blockSize;
        return row;
    }

    __aicore__ inline uint32_t CalcInputPoint(uint32_t outputPoint)
    {
        const uint32_t outHW = this->outHeight * this->outWidth;
        const uint32_t ob = outputPoint / outHW;
        const uint32_t rem = outputPoint - ob * outHW;
        const uint32_t oh = rem / this->outWidth;
        const uint32_t ow = rem - oh * this->outWidth;
        const uint32_t fullH = oh + this->cropTop;
        const uint32_t fullW = ow + this->cropLeft;
        const uint32_t blockH = fullH % this->blockSize;
        const uint32_t blockW = fullW % this->blockSize;
        const uint32_t ih = fullH / this->blockSize;
        const uint32_t iw = fullW / this->blockSize;
        const uint32_t ib = (blockH * this->blockSize + blockW) * this->outBatch + ob;
        return ((ib * this->inputHeight + ih) * this->inputWidth + iw);
    }

    __aicore__ inline uint32_t CalcInputPointInRow(
        const RowContext &row, uint32_t blockW, uint32_t iw) const
    {
        const uint32_t ib = (row.blockH * this->blockSize + blockW) * this->outBatch + row.ob;
        return ((ib * this->inputHeight + row.ih) * this->inputWidth + iw);
    }

    template <typename T>
    __aicore__ inline void CopyPointRangeTyped(
        AscendC::GlobalTensor<T> &xTensor,
        AscendC::GlobalTensor<T> &yTensor,
        uint32_t globalPointBase,
        uint32_t pointNum,
        uint32_t unitNum)
    {
        for (uint32_t p = 0; p < pointNum; ++p) {
            const uint32_t outputPoint = globalPointBase + p;
            const uint32_t inputPoint = CalcInputPoint(outputPoint);
            const uint32_t inputOffset = inputPoint * unitNum;
            const uint32_t outputOffset = outputPoint * unitNum;
            for (uint32_t i = 0; i < unitNum; ++i) {
                yTensor.SetValue(outputOffset + i, xTensor.GetValue(inputOffset + i));
            }
        }
    }

    template <typename T>
    __aicore__ inline void CopyNoCropFastTyped(
        AscendC::GlobalTensor<T> &xTensor,
        AscendC::GlobalTensor<T> &yTensor,
        uint32_t unitNum)
    {
        auto yLocal = yBuf.Get<T>();

        for (uint32_t tileIdx = 0; tileIdx < this->tileNum; ++tileIdx) {
            const uint32_t curPointNum =
                (tileIdx == this->tileNum - 1) ? this->tailPointNum : this->tilePointNum;
            const uint32_t tilePointBase = this->startPoint + tileIdx * this->tilePointNum;

            for (uint32_t groupBase = 0; groupBase < curPointNum; groupBase += FAST_COPY_POINT_NUM) {
                const uint32_t groupPointNum =
                    MinU32(curPointNum - groupBase, FAST_COPY_POINT_NUM);
                const uint32_t globalPointBase = tilePointBase + groupBase;

                // V114/V117: no-crop row-local index recurrence.
                // Keep the proven 16-point group and the same GM access order,
                // but avoid calling CalcInputPoint() for every output point.
                uint32_t localPoint = 0;
                while (localPoint < groupPointNum) {
                    const uint32_t outputPoint = globalPointBase + localPoint;
                    const uint32_t outputRow = outputPoint / this->outWidth;
                    const uint32_t owBegin = outputPoint - outputRow * this->outWidth;
                    const uint32_t rowPointNum =
                        MinU32(groupPointNum - localPoint, this->outWidth - owBegin);

                    const uint32_t ob = outputRow / this->outHeight;
                    const uint32_t oh = outputRow - ob * this->outHeight;
                    const uint32_t blockH = oh % this->blockSize;
                    const uint32_t ih = oh / this->blockSize;
                    uint32_t blockW = owBegin % this->blockSize;
                    uint32_t iw = owBegin / this->blockSize;

                    for (uint32_t p = 0; p < rowPointNum; ++p) {
                        const uint32_t ib = (blockH * this->blockSize + blockW) * this->outBatch + ob;
                        const uint32_t inputPoint =
                            ((ib * this->inputHeight + ih) * this->inputWidth + iw);
                        AscendC::DataCopy(
                            yLocal[(localPoint + p) * unitNum],
                            xTensor[inputPoint * unitNum],
                            unitNum);

                        ++blockW;
                        if (blockW == this->blockSize) {
                            blockW = 0;
                            ++iw;
                        }
                    }
                    localPoint += rowPointNum;
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                CopyLocalToGmTyped(yTensor, yLocal, globalPointBase, groupPointNum, unitNum);
            }
        }
    }

    template <typename T>
    __aicore__ inline void CopyCropRowsTyped(
        AscendC::GlobalTensor<T> &xTensor,
        AscendC::GlobalTensor<T> &yTensor,
        uint32_t globalRowBase,
        uint32_t rowNum,
        uint32_t unitNum)
    {
        for (uint32_t r = 0; r < rowNum; ++r) {
            CopyCropRowRangeTyped(
                xTensor,
                yTensor,
                globalRowBase + r,
                0,
                this->outWidth,
                unitNum);
        }
    }

    template <typename T>
    __aicore__ inline void CopyCropRowRangeTyped(
        AscendC::GlobalTensor<T> &xTensor,
        AscendC::GlobalTensor<T> &yTensor,
        uint32_t outputRow,
        uint32_t owBegin,
        uint32_t owEnd,
        uint32_t unitNum)
    {
        if (owBegin >= owEnd) {
            return;
        }

        const RowContext row = GetRowContext(outputRow);
        for (uint32_t blockW = 0; blockW < this->blockSize; ++blockW) {
            uint32_t ow = FirstOwForBlockInRange(blockW, row.cropLeftMod, owBegin);
            if (ow >= owEnd) {
                continue;
            }

            uint32_t iw = (ow + this->cropLeft) / this->blockSize;
            for (; ow < owEnd; ow += this->blockSize, ++iw) {
                const uint32_t inputPoint = CalcInputPointInRow(row, blockW, iw);
                const uint32_t outputPoint = row.rowBasePoint + ow;
                const uint32_t inputOffset = inputPoint * unitNum;
                const uint32_t outputOffset = outputPoint * unitNum;
                for (uint32_t i = 0; i < unitNum; ++i) {
                    yTensor.SetValue(outputOffset + i, xTensor.GetValue(inputOffset + i));
                }
            }
        }
    }

    template <typename T>
    __aicore__ inline void FillLocalRowRangeTyped(
        AscendC::GlobalTensor<T> &xTensor,
        AscendC::LocalTensor<T> yLocal,
        const RowContext &row,
        uint32_t owBegin,
        uint32_t owEnd,
        uint32_t localOwBase,
        uint32_t unitNum)
    {
        for (uint32_t blockW = 0; blockW < this->blockSize; ++blockW) {
            uint32_t ow = FirstOwForBlockInRange(blockW, row.cropLeftMod, owBegin);
            if (ow >= owEnd) {
                continue;
            }

            uint32_t iw = (ow + this->cropLeft) / this->blockSize;
            for (; ow < owEnd; ow += this->blockSize, ++iw) {
                const uint32_t inputPoint = CalcInputPointInRow(row, blockW, iw);
                const uint32_t inputOffset = inputPoint * unitNum;
                const uint32_t localOffset = (ow - localOwBase) * unitNum;
                for (uint32_t i = 0; i < unitNum; ++i) {
                    yLocal.SetValue(localOffset + i, xTensor.GetValue(inputOffset + i));
                }
            }
        }
    }

    template <typename T>
    __aicore__ inline void CopyLocalToGmTyped(
        AscendC::GlobalTensor<T> &yTensor,
        AscendC::LocalTensor<T> yLocal,
        uint32_t outputPointBase,
        uint32_t pointNum,
        uint32_t unitNum)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::DataCopy(yTensor[outputPointBase * unitNum], yLocal, pointNum * unitNum);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    }

    template <typename T>
    __aicore__ inline void CopyCropChunksTyped(
        AscendC::GlobalTensor<T> &xTensor,
        AscendC::GlobalTensor<T> &yTensor,
        uint32_t globalChunkBase,
        uint32_t chunkNum,
        uint32_t unitNum)
    {
        auto yLocal = yBuf.Get<T>();

        for (uint32_t c = 0; c < chunkNum; ++c) {
            const uint32_t globalChunk = globalChunkBase + c;
            const uint32_t outputRow = globalChunk / this->cropChunkNumPerRow;
            const uint32_t chunkIdx = globalChunk - outputRow * this->cropChunkNumPerRow;
            const uint32_t owBegin = chunkIdx * this->cropChunkPointNum;
            if (owBegin >= this->outWidth) {
                continue;
            }

            const uint32_t owEnd = MinU32(owBegin + this->cropChunkPointNum, this->outWidth);
            const RowContext row = GetRowContext(outputRow);
            const uint32_t curPointNum = owEnd - owBegin;

            for (uint32_t p = 0; p < curPointNum; ++p) {
                const uint32_t outputPoint = row.rowBasePoint + owBegin + p;
                const uint32_t inputPoint = CalcInputPoint(outputPoint);
                AscendC::DataCopy(
                    yLocal[p * unitNum],
                    xTensor[inputPoint * unitNum],
                    unitNum);
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            CopyLocalToGmTyped(yTensor, yLocal, row.rowBasePoint + owBegin, curPointNum, unitNum);
        }
    }

    __aicore__ inline void CopyCropByteBlocks(uint32_t globalBlockBase, uint32_t blockNum)
    {
        auto yLocal = yBuf.Get<int8_t>();
        const uint32_t totalBytes =
            this->outBatch * this->outHeight * this->outWidth * this->depthBytes;

        for (uint32_t c = 0; c < blockNum; ++c) {
            const uint32_t outputBaseByte = (globalBlockBase + c) * 32;
            if (outputBaseByte >= totalBytes) {
                continue;
            }

            const uint32_t curBytes = MinU32(totalBytes - outputBaseByte, 32);
            if (curBytes == 32) {
                for (uint32_t i = 0; i < 32; ++i) {
                    const uint32_t outputByte = outputBaseByte + i;
                    const uint32_t outputPoint = outputByte / this->depthBytes;
                    const uint32_t d = outputByte - outputPoint * this->depthBytes;
                    const uint32_t inputPoint = CalcInputPoint(outputPoint);
                    yLocal.SetValue(i, xGm.GetValue(inputPoint * this->depthBytes + d));
                }
                CopyLocalToGmTyped(yGm, yLocal, outputBaseByte, 32, 1);
            } else {
                for (uint32_t i = 0; i < curBytes; ++i) {
                    const uint32_t outputByte = outputBaseByte + i;
                    const uint32_t outputPoint = outputByte / this->depthBytes;
                    const uint32_t d = outputByte - outputPoint * this->depthBytes;
                    const uint32_t inputPoint = CalcInputPoint(outputPoint);
                    yGm.SetValue(outputByte, xGm.GetValue(inputPoint * this->depthBytes + d));
                }
            }
        }
    }

    template <typename T>
    __aicore__ inline void CopyCropRowsAlignedChunksTyped(
        AscendC::GlobalTensor<T> &xTensor,
        AscendC::GlobalTensor<T> &yTensor,
        uint32_t globalRowBase,
        uint32_t rowNum,
        uint32_t unitNum)
    {
        const uint32_t alignStep = GetOutputAlignPointStep();
        uint32_t chunkPointNum = this->tilePointNum - (this->tilePointNum % alignStep);
        if (chunkPointNum < alignStep) {
            CopyCropRowsTyped(xTensor, yTensor, globalRowBase, rowNum, unitNum);
            return;
        }

        auto yLocal = yBuf.Get<T>();
        for (uint32_t r = 0; r < rowNum; ++r) {
            const uint32_t outputRow = globalRowBase + r;
            const RowContext row = GetRowContext(outputRow);
            uint32_t owBase = FindFirstAlignedOw(row.rowBasePoint, alignStep);

            CopyCropRowRangeTyped(xTensor, yTensor, outputRow, 0, owBase, unitNum);
            while (owBase + alignStep <= this->outWidth) {
                uint32_t curPointNum = MinU32(this->outWidth - owBase, chunkPointNum);
                curPointNum -= curPointNum % alignStep;
                if (curPointNum < alignStep) {
                    break;
                }

                const uint32_t owEnd = owBase + curPointNum;
                FillLocalRowRangeTyped(xTensor, yLocal, row, owBase, owEnd, owBase, unitNum);
                CopyLocalToGmTyped(yTensor, yLocal, row.rowBasePoint + owBase, curPointNum, unitNum);
                owBase = owEnd;
            }
            CopyCropRowRangeTyped(xTensor, yTensor, outputRow, owBase, this->outWidth, unitNum);
        }
    }

    template <typename T>
    __aicore__ inline void CopyCropRowChunksParallelTyped(
        AscendC::GlobalTensor<T> &xTensor,
        AscendC::GlobalTensor<T> &yTensor,
        uint32_t globalChunkBase,
        uint32_t chunkNum,
        uint32_t unitNum)
    {
        const uint32_t alignStep = GetOutputAlignPointStep();
        auto yLocal = yBuf.Get<T>();

        for (uint32_t c = 0; c < chunkNum; ++c) {
            const uint32_t globalChunk = globalChunkBase + c;
            const uint32_t outputRow = globalChunk / this->cropChunkNumPerRow;
            const uint32_t chunkIdx = globalChunk - outputRow * this->cropChunkNumPerRow;
            const RowContext row = GetRowContext(outputRow);
            const uint32_t alignedOw = FindFirstAlignedOw(row.rowBasePoint, alignStep);

            if (alignedOw >= this->outWidth) {
                if (chunkIdx == 0) {
                    CopyCropRowRangeTyped(xTensor, yTensor, outputRow, 0, this->outWidth, unitNum);
                }
                continue;
            }

            uint32_t middlePointNum = this->outWidth - alignedOw;
            middlePointNum -= middlePointNum % alignStep;
            const uint32_t middleEnd = alignedOw + middlePointNum;

            if (chunkIdx == 0) {
                CopyCropRowRangeTyped(xTensor, yTensor, outputRow, 0, alignedOw, unitNum);
            }
            if (middlePointNum == 0) {
                if (chunkIdx == 0) {
                    CopyCropRowRangeTyped(xTensor, yTensor, outputRow, alignedOw, this->outWidth, unitNum);
                }
                continue;
            }

            const uint32_t lastChunkIdx = (middlePointNum - 1) / this->cropChunkPointNum;
            const uint32_t owBegin = alignedOw + chunkIdx * this->cropChunkPointNum;
            if (owBegin < middleEnd) {
                const uint32_t owEnd = MinU32(owBegin + this->cropChunkPointNum, middleEnd);
                FillLocalRowRangeTyped(xTensor, yLocal, row, owBegin, owEnd, owBegin, unitNum);
                CopyLocalToGmTyped(yTensor, yLocal, row.rowBasePoint + owBegin, owEnd - owBegin, unitNum);
            }

            if (chunkIdx == lastChunkIdx) {
                CopyCropRowRangeTyped(xTensor, yTensor, outputRow, middleEnd, this->outWidth, unitNum);
            }
        }
    }


    __aicore__ inline void CopyPaddedLocalPointsToGm(
        AscendC::LocalTensor<int8_t> yLocal,
        uint32_t outputByteOffset,
        uint32_t pointNum)
    {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(pointNum);
        copyParams.blockLen = this->depthBytes;
        // Source is UB.  DataCopyPad stores each non-32B block in a 32B-padded
        // slot, so compact padded slots are represented with zero extra stride.
        copyParams.srcStride = 0;
        // Destination is GM and output points are compact/contiguous.
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::DataCopyPad(yGm[outputByteOffset], yLocal, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    }

    __aicore__ inline void IssuePaddedLocalPointsToGmNoWait(
        AscendC::LocalTensor<int8_t> yLocal,
        uint32_t outputByteOffset,
        uint32_t pointNum)
    {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(pointNum);
        copyParams.blockLen = this->depthBytes;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::DataCopyPad(yGm[outputByteOffset], yLocal, copyParams);
        // Defer this wait until just before the next MTE3 issue, so MTE3 of the
        // previous buffer can overlap with MTE2 filling of the next buffer.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    }

    __aicore__ inline void WaitPendingPaddedLocalToGm()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    }

    __aicore__ inline void CopyLocalToGmBytes(
        AscendC::LocalTensor<int8_t> yLocal,
        uint32_t outputByteOffset,
        uint32_t byteNum)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::DataCopy(yGm[outputByteOffset], yLocal, byteNum);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    }

    __aicore__ inline void CopyOnePointGmToLocalPad(
        AscendC::LocalTensor<int8_t> yLocal,
        uint32_t localByteOffset,
        uint32_t inputByteOffset)
    {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = this->depthBytes;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        AscendC::DataCopyPadExtParams<int8_t> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;

        // Destination must be a 32B-padded UB slot.  Do not pack slots by depthBytes.
        AscendC::DataCopyPad(yLocal[localByteOffset], xGm[inputByteOffset], copyParams, padParams);
    }

    __aicore__ inline void CopyCropUnalignedFlatDmaChunks(uint32_t globalChunkBase, uint32_t chunkNum)
    {
        auto yLocal0 = yBuf.Get<int8_t>();
        auto yLocal1 = yBufDb.Get<int8_t>();
        const uint32_t totalPointNum = this->outBatch * this->outHeight * this->outWidth;
        bool pendingMte3 = false;

        for (uint32_t c = 0; c < chunkNum; ++c) {
            const uint32_t globalChunk = globalChunkBase + c;
            const uint32_t outputPointBase = globalChunk * this->cropChunkPointNum;
            if (outputPointBase >= totalPointNum) {
                continue;
            }

            const uint32_t curPointNum = MinU32(this->cropChunkPointNum, totalPointNum - outputPointBase);
            const uint32_t slotBytes = AlignUp32(this->depthBytes);
            const uint32_t smallChunkLimit =
                GetOutputAlignPointStep() * SMALL_UNALIGNED_CHUNK_GROUP_NUM;
            const uint32_t waitPointNum =
                (this->cropChunkPointNum <= smallChunkLimit) ?
                    SMALL_UNALIGNED_WAIT_POINT_NUM : FAST_COPY_POINT_NUM;
            auto yLocal = ((c & 1) == 0) ? yLocal0 : yLocal1;

            uint32_t localPoint = 0;
            while (localPoint < curPointNum) {
                const uint32_t outputPoint = outputPointBase + localPoint;
                const uint32_t outputRow = outputPoint / this->outWidth;
                const uint32_t owBegin = outputPoint - outputRow * this->outWidth;
                const uint32_t rowPointNum =
                    MinU32(curPointNum - localPoint, this->outWidth - owBegin);
                const RowContext row = GetRowContext(outputRow);

                uint32_t fullW = owBegin + this->cropLeft;
                uint32_t blockW = fullW % this->blockSize;
                uint32_t iw = fullW / this->blockSize;

                for (uint32_t i = 0; i < rowPointNum; ++i) {
                    const uint32_t inputPoint = CalcInputPointInRow(row, blockW, iw);
                    CopyOnePointGmToLocalPad(
                        yLocal,
                        (localPoint + i) * slotBytes,
                        inputPoint * this->depthBytes);

                    ++blockW;
                    if (blockW == this->blockSize) {
                        blockW = 0;
                        ++iw;
                    }

                    if (((localPoint + i + 1) % waitPointNum) == 0) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                    }
                }

                localPoint += rowPointNum;
            }

            if ((curPointNum % waitPointNum) != 0) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            }

            // Before issuing this chunk's MTE3 writeback, wait for the previous
            // chunk's writeback to complete.  This preserves single outstanding
            // MTE3 event semantics, but still overlaps previous MTE3 with current MTE2.
            if (pendingMte3) {
                WaitPendingPaddedLocalToGm();
                pendingMte3 = false;
            }

            const uint32_t outputByteOffset = outputPointBase * this->depthBytes;
            IssuePaddedLocalPointsToGmNoWait(yLocal, outputByteOffset, curPointNum);
            pendingMte3 = true;
        }

        if (pendingMte3) {
            WaitPendingPaddedLocalToGm();
        }
    }

    __aicore__ inline uint32_t CalcCropBlock2GatherInputOffset(
        const RowContext &row,
        uint32_t ow,
        uint32_t unitNum) const
    {
        const uint32_t fullW = ow + this->cropLeft;
        const uint32_t blockW = fullW & 1U;
        const uint32_t iw = fullW >> 1;
        return CalcInputPointInRow(row, blockW, iw) * unitNum;
    }

    template <typename T>
    __aicore__ inline uint32_t GetBlockElementNumTyped() const
    {
        return 32 / sizeof(T);
    }

    template <typename T>
    __aicore__ inline void BuildCropBlock2GatherOffsets(
        uint32_t maxGroupNum,
        uint32_t phaseStride,
        AscendC::LocalTensor<int32_t> offsetLocal,
        uint32_t unitNum)
    {
        const uint32_t groupDataNum = 2 * unitNum;
        const uint32_t blockElementNum = GetBlockElementNumTyped<T>();
        const uint32_t groupAlign = blockElementNum / GcdU32(groupDataNum, blockElementNum);
        const uint32_t seedGroupNum = MinU32(maxGroupNum, groupAlign);

        for (uint32_t group = 0; group < seedGroupNum; ++group) {
            const uint32_t dstBase = group * groupDataNum;
            const uint32_t srcBase = group * unitNum;
            for (uint32_t d = 0; d < unitNum; ++d) {
                offsetLocal.SetValue(
                    dstBase + d,
                    static_cast<int32_t>((srcBase + d) * sizeof(T)));
                offsetLocal.SetValue(
                    dstBase + unitNum + d,
                    static_cast<int32_t>((phaseStride + srcBase + d) * sizeof(T)));
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        uint32_t readyGroupNum = seedGroupNum;
        while (readyGroupNum < maxGroupNum) {
            uint32_t copyGroupNum = readyGroupNum;
            if (copyGroupNum > maxGroupNum - readyGroupNum) {
                copyGroupNum = maxGroupNum - readyGroupNum;
            }
            const uint32_t dstOffset = readyGroupNum * groupDataNum;
            const uint32_t copyDataNum = copyGroupNum * groupDataNum;
            const int32_t byteShift =
                static_cast<int32_t>(readyGroupNum * unitNum * sizeof(T));
            AscendC::Adds(offsetLocal[dstOffset], offsetLocal, byteShift, copyDataNum);
            AscendC::PipeBarrier<PIPE_V>();
            readyGroupNum += copyGroupNum;
        }
    }

    template <typename T>
    __aicore__ inline void CopyCropBlock2GatherPackSegment(
        AscendC::LocalTensor<T> packLocal,
        uint32_t packOffset,
        AscendC::GlobalTensor<T> &xTensor,
        uint32_t inputOffset,
        uint32_t dataNum)
    {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = dataNum * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        AscendC::DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;

        AscendC::DataCopyPad(packLocal[packOffset], xTensor[inputOffset], copyParams, padParams);
    }

    template <typename T>
    __aicore__ inline void CopyCropBlock2GatherChunkTyped(
        AscendC::GlobalTensor<T> &xTensor,
        AscendC::GlobalTensor<T> &yTensor,
        uint32_t outputRow,
        uint32_t owBegin,
        uint32_t owEnd,
        uint32_t maxGroupNum,
        uint32_t phaseStride,
        uint32_t packOffset,
        uint32_t offsetWordOffset,
        uint32_t unitNum)
    {
        if (owBegin >= owEnd) {
            return;
        }

        const uint32_t groupNum = (owEnd - owBegin) >> 1;
        if (groupNum == 0 || groupNum > maxGroupNum) {
            CopyCropRowRangeTyped(xTensor, yTensor, outputRow, owBegin, owEnd, unitNum);
            return;
        }

        AscendC::LocalTensor<T> baseLocal = yBuf.Get<T>();
        AscendC::LocalTensor<T> yLocal = baseLocal;
        AscendC::LocalTensor<T> packLocal = baseLocal[packOffset];
        AscendC::LocalTensor<uint32_t> offsetLocal =
            yBuf.Get<uint32_t>()[offsetWordOffset];

        const RowContext row = GetRowContext(outputRow);
        const uint32_t phaseDataNum = groupNum * unitNum;
        const uint32_t dataNum = groupNum * 2 * unitNum;
        const uint32_t inputOffset0 = CalcCropBlock2GatherInputOffset(row, owBegin, unitNum);
        const uint32_t inputOffset1 = CalcCropBlock2GatherInputOffset(row, owBegin + 1, unitNum);

        CopyCropBlock2GatherPackSegment(packLocal, 0, xTensor, inputOffset0, phaseDataNum);
        CopyCropBlock2GatherPackSegment(packLocal, phaseStride, xTensor, inputOffset1, phaseDataNum);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);

        yLocal.SetSize(dataNum);
        packLocal.SetSize(2 * phaseStride);
        offsetLocal.SetSize(dataNum);
        AscendC::Gather(yLocal, packLocal, offsetLocal, static_cast<uint32_t>(0), dataNum);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);

        AscendC::DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = dataNum * sizeof(T);
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        outParams.rsv = 0;
        AscendC::DataCopyPad(yTensor[(row.rowBasePoint + owBegin) * unitNum], yLocal, outParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);

        const uint32_t pairedEnd = owBegin + (groupNum << 1);
        if (pairedEnd < owEnd) {
            CopyCropRowRangeTyped(xTensor, yTensor, outputRow, pairedEnd, owEnd, unitNum);
        }
    }

    template <typename T>
    __aicore__ inline void CopyCropBlock2GatherChunkTypedDb(
        AscendC::GlobalTensor<T> &xTensor,
        AscendC::GlobalTensor<T> &yTensor,
        AscendC::LocalTensor<T> baseLocal,
        AscendC::TEventID doneEvent,
        bool &pending,
        uint32_t outputRow,
        uint32_t owBegin,
        uint32_t owEnd,
        uint32_t maxGroupNum,
        uint32_t phaseStride,
        uint32_t packOffset,
        uint32_t offsetWordOffset,
        uint32_t unitNum)
    {
        if (owBegin >= owEnd) {
            return;
        }

        const uint32_t groupNum = (owEnd - owBegin) >> 1;
        if (groupNum == 0 || groupNum > maxGroupNum) {
            if (pending) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(doneEvent);
                pending = false;
            }
            CopyCropRowRangeTyped(xTensor, yTensor, outputRow, owBegin, owEnd, unitNum);
            return;
        }

        if (pending) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(doneEvent);
            pending = false;
        }

        AscendC::LocalTensor<T> yLocal = baseLocal;
        AscendC::LocalTensor<T> packLocal = baseLocal[packOffset];
        AscendC::LocalTensor<uint32_t> offsetLocal =
            yBuf.Get<uint32_t>()[offsetWordOffset];

        const RowContext row = GetRowContext(outputRow);
        const uint32_t phaseDataNum = groupNum * unitNum;
        const uint32_t dataNum = groupNum * 2 * unitNum;
        const uint32_t inputOffset0 = CalcCropBlock2GatherInputOffset(row, owBegin, unitNum);
        const uint32_t inputOffset1 = CalcCropBlock2GatherInputOffset(row, owBegin + 1, unitNum);

        CopyCropBlock2GatherPackSegment(packLocal, 0, xTensor, inputOffset0, phaseDataNum);
        CopyCropBlock2GatherPackSegment(packLocal, phaseStride, xTensor, inputOffset1, phaseDataNum);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);

        yLocal.SetSize(dataNum);
        packLocal.SetSize(2 * phaseStride);
        offsetLocal.SetSize(dataNum);
        AscendC::Gather(yLocal, packLocal, offsetLocal, static_cast<uint32_t>(0), dataNum);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);

        AscendC::DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = dataNum * sizeof(T);
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        outParams.rsv = 0;
        AscendC::DataCopyPad(yTensor[(row.rowBasePoint + owBegin) * unitNum], yLocal, outParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(doneEvent);
        pending = true;

        const uint32_t pairedEnd = owBegin + (groupNum << 1);
        if (pairedEnd < owEnd) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(doneEvent);
            pending = false;
            CopyCropRowRangeTyped(xTensor, yTensor, outputRow, pairedEnd, owEnd, unitNum);
        }
    }

    template <typename T>
    __aicore__ inline void CopyCropBlock2GatherChunksTyped(
        AscendC::GlobalTensor<T> &xTensor,
        AscendC::GlobalTensor<T> &yTensor,
        uint32_t globalChunkBase,
        uint32_t chunkNum,
        uint32_t unitNum)
    {
        const uint32_t maxGroupNum = this->cropChunkPointNum >> 1;
        if (maxGroupNum == 0) {
            for (uint32_t c = 0; c < chunkNum; ++c) {
                const uint32_t globalChunk = globalChunkBase + c;
                const uint32_t outputRow = globalChunk / this->cropChunkNumPerRow;
                const uint32_t chunkIdx = globalChunk - outputRow * this->cropChunkNumPerRow;
                const uint32_t owBegin = chunkIdx * this->cropChunkPointNum;
                if (owBegin < this->outWidth) {
                    const uint32_t owEnd = MinU32(owBegin + this->cropChunkPointNum, this->outWidth);
                    CopyCropRowRangeTyped(xTensor, yTensor, outputRow, owBegin, owEnd, unitNum);
                }
            }
            return;
        }

        const uint32_t blockElementNum = GetBlockElementNumTyped<T>();
        const uint32_t dataCapacity = maxGroupNum * 2 * unitNum;
        const uint32_t phaseStride = AlignUpU32(maxGroupNum * unitNum, blockElementNum);
        const uint32_t packOffset = AlignUpU32(dataCapacity, blockElementNum);
        const uint32_t offsetByte =
            AlignUp32((packOffset + 2 * phaseStride) * sizeof(T));
        const uint32_t offsetWordOffset = offsetByte / sizeof(uint32_t);

        AscendC::LocalTensor<int32_t> offsetBuildLocal =
            yBuf.Get<int32_t>()[offsetWordOffset];
        BuildCropBlock2GatherOffsets<T>(maxGroupNum, phaseStride, offsetBuildLocal, unitNum);

        auto yLocal0 = yBuf.Get<T>();
        auto yLocal1 = yBufDb.Get<T>();
        bool pending0 = false;
        bool pending1 = false;
        uint32_t bufferIdx = 0;
        for (uint32_t c = 0; c < chunkNum; ++c) {
            const uint32_t globalChunk = globalChunkBase + c;
            const uint32_t outputRow = globalChunk / this->cropChunkNumPerRow;
            const uint32_t chunkIdx = globalChunk - outputRow * this->cropChunkNumPerRow;
            const uint32_t owBegin = chunkIdx * this->cropChunkPointNum;
            if (owBegin >= this->outWidth) {
                continue;
            }

            const uint32_t owEnd = MinU32(owBegin + this->cropChunkPointNum, this->outWidth);
            if (bufferIdx == 0) {
                CopyCropBlock2GatherChunkTypedDb(
                    xTensor,
                    yTensor,
                    yLocal0,
                    eventIdMte3ToMte2,
                    pending0,
                    outputRow,
                    owBegin,
                    owEnd,
                    maxGroupNum,
                    phaseStride,
                    packOffset,
                    offsetWordOffset,
                    unitNum);
            } else {
                CopyCropBlock2GatherChunkTypedDb(
                    xTensor,
                    yTensor,
                    yLocal1,
                    eventIdMte3ToMte2Db,
                    pending1,
                    outputRow,
                    owBegin,
                    owEnd,
                    maxGroupNum,
                    phaseStride,
                    packOffset,
                    offsetWordOffset,
                    unitNum);
            }
            bufferIdx ^= 1;
        }
        if (pending0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        if (pending1) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2Db);
        }
    }


    __aicore__ inline void CopyInputSegmentToLocalPad(
        AscendC::LocalTensor<int8_t> yLocal,
        uint32_t inputByteOffset,
        uint32_t pointNum)
    {
        if (pointNum == 0) {
            return;
        }

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(pointNum);
        copyParams.blockLen = this->depthBytes;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        AscendC::DataCopyPadExtParams<int8_t> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;

        AscendC::DataCopyPad(yLocal, xGm[inputByteOffset], copyParams, padParams);
    }

    // V34B: test9-like input segment is contiguous and 32B aligned.
    // Use ordinary continuous DataCopy for GM->UB input when legal.
    __aicore__ inline void CopyInputSegmentToLocalNormalDc(
        AscendC::LocalTensor<int8_t> yLocal,
        uint32_t inputByteOffset,
        uint32_t pointNum)
    {
        if (pointNum == 0) {
            return;
        }
        const uint32_t byteNum = pointNum * this->depthBytes;
        if (((inputByteOffset | byteNum | this->depthBytes) & 31U) != 0U) {
            CopyInputSegmentToLocalPad(yLocal, inputByteOffset, pointNum);
            return;
        }
        AscendC::DataCopy(yLocal, xGm[inputByteOffset], byteNum);
    }

    __aicore__ inline void CopyLocalSegmentToGmStridedPad(
        AscendC::LocalTensor<int8_t> yLocal,
        uint32_t outputByteOffset,
        uint32_t pointNum)
    {
        if (pointNum == 0) {
            return;
        }

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(pointNum);
        copyParams.blockLen = this->depthBytes;
        copyParams.srcStride = 0;
        copyParams.dstStride = (this->blockSize - 1) * this->depthBytes;
        copyParams.rsv = 0;

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::DataCopyPad(yGm[outputByteOffset], yLocal, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    }


    __aicore__ inline void IssueLocalSegmentToGmStridedNoWait(
        AscendC::LocalTensor<int8_t> yLocal,
        uint32_t outputByteOffset,
        uint32_t pointNum,
        AscendC::TEventID doneEvent)
    {
        if (pointNum == 0) {
            return;
        }

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(pointNum);
        copyParams.blockLen = this->depthBytes;
        copyParams.srcStride = 0;
        copyParams.dstStride = (this->blockSize - 1) * this->depthBytes;
        copyParams.rsv = 0;

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::DataCopyPad(yGm[outputByteOffset], yLocal, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(doneEvent);
    }

    // V34A: test9-like row-segment normally has 32B-aligned channel width.
    // Try ordinary strided DataCopy for UB->GM writeback to avoid DataCopyPad overhead.
    // Fallback to the original DataCopyPad path if any byte quantity is not 32B aligned.
    __aicore__ inline void IssueLocalSegmentToGmStridedNoWaitNormalDc(
        AscendC::LocalTensor<int8_t> yLocal,
        uint32_t outputByteOffset,
        uint32_t pointNum,
        AscendC::TEventID doneEvent)
    {
        if (pointNum == 0) {
            return;
        }
        if (((this->depthBytes | outputByteOffset) & 31U) != 0U) {
            IssueLocalSegmentToGmStridedNoWait(yLocal, outputByteOffset, pointNum, doneEvent);
            return;
        }

        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(pointNum);
        copyParams.blockLen = static_cast<uint16_t>(this->depthBytes >> 5);
        copyParams.srcStride = 0;
        copyParams.dstStride = static_cast<uint16_t>(((this->blockSize - 1) * this->depthBytes) >> 5);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::DataCopy(yGm[outputByteOffset], yLocal, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(doneEvent);
    }



    // V41: test9 row-segment coalesced write path.
    // Keep GM reads as continuous per-blockW streams, but place them directly into
    // the final output layout inside UB by using MTE2 destination stride.  Then
    // write the whole [owBegin, owEnd) span back to GM as one continuous packet.
    // This removes the previous GM-side strided write from CopyRowSegmentDmaChunksDb.
    __aicore__ inline void FillRowSegmentCoalescedPadReadToLocal(
        AscendC::LocalTensor<int8_t> yLocal,
        uint32_t outputRow,
        uint32_t owBegin,
        uint32_t owEnd)
    {
        const RowContext row = GetRowContext(outputRow);
        for (uint32_t blockW = 0; blockW < 2; ++blockW) {
            const uint32_t firstOw = FirstOwForBlockInRange(blockW, row.cropLeftMod, owBegin);
            if (firstOw >= owEnd) {
                continue;
            }
            const uint32_t pointNum = ((owEnd - 1 - firstOw) >> 1) + 1;
            const uint32_t firstIw = (firstOw + this->cropLeft) >> 1;
            const uint32_t inputPoint = CalcInputPointInRow(row, blockW, firstIw);
            const uint32_t localByteOffset = (firstOw - owBegin) * this->depthBytes;

            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = static_cast<uint16_t>(pointNum);
            copyParams.blockLen = this->depthBytes;
            copyParams.srcStride = 0;
            // blockSize == 2: write one depth block, leave one depth block gap in UB.
            copyParams.dstStride = this->depthBytes;
            copyParams.rsv = 0;

            AscendC::DataCopyPadExtParams<int8_t> padParams;
            padParams.isPad = false;
            padParams.leftPadding = 0;
            padParams.rightPadding = 0;
            padParams.paddingValue = 0;

            AscendC::DataCopyPad(
                yLocal[localByteOffset],
                xGm[inputPoint * this->depthBytes],
                copyParams,
                padParams);
        }
    }

    __aicore__ inline void IssueCoalescedLocalSpanToGmNoWait(
        AscendC::LocalTensor<int8_t> yLocal,
        uint32_t outputByteOffset,
        uint32_t spanPointNum,
        AscendC::TEventID doneEvent)
    {
        if (spanPointNum == 0) {
            return;
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::DataCopy(yGm[outputByteOffset], yLocal, spanPointNum * this->depthBytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(doneEvent);
    }

    __aicore__ inline void CopyRowSegmentCoalescedPadReadChunks(uint32_t globalChunkBase, uint32_t chunkNum)
    {
        auto yLocal = yBuf.Get<int8_t>();
        for (uint32_t c = 0; c < chunkNum; ++c) {
            const uint32_t globalChunk = globalChunkBase + c;
            const uint32_t outputRow = globalChunk / this->cropChunkNumPerRow;
            const uint32_t chunkIdx = globalChunk - outputRow * this->cropChunkNumPerRow;
            const uint32_t owBegin = chunkIdx * this->cropChunkPointNum;
            if (owBegin >= this->outWidth) {
                continue;
            }
            const uint32_t owEnd = MinU32(owBegin + this->cropChunkPointNum, this->outWidth);
            const uint32_t spanPointNum = owEnd - owBegin;
            if (spanPointNum == 0 || this->blockSize != 2 || ((this->depthBytes & 31U) != 0U)) {
                CopyRowSegmentDmaChunksDb(globalChunk, 1);
                continue;
            }
            FillRowSegmentCoalescedPadReadToLocal(yLocal, outputRow, owBegin, owEnd);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            const RowContext row = GetRowContext(outputRow);
            IssueCoalescedLocalSpanToGmNoWait(
                yLocal,
                (row.rowBasePoint + owBegin) * this->depthBytes,
                spanPointNum,
                eventIdMte3ToMte2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
    }

    __aicore__ inline void CopyRowSegmentCoalescedPadReadChunksDb(uint32_t globalChunkBase, uint32_t chunkNum)
    {
        auto yLocal0 = yBuf.Get<int8_t>();
        auto yLocal1 = yBufDb.Get<int8_t>();
        bool pending0 = false;
        bool pending1 = false;
        for (uint32_t c = 0; c < chunkNum; ++c) {
            const uint32_t globalChunk = globalChunkBase + c;
            const uint32_t outputRow = globalChunk / this->cropChunkNumPerRow;
            const uint32_t chunkIdx = globalChunk - outputRow * this->cropChunkNumPerRow;
            const uint32_t owBegin = chunkIdx * this->cropChunkPointNum;
            if (owBegin >= this->outWidth) {
                continue;
            }
            const uint32_t owEnd = MinU32(owBegin + this->cropChunkPointNum, this->outWidth);
            const uint32_t spanPointNum = owEnd - owBegin;
            if (spanPointNum == 0 || this->blockSize != 2 || ((this->depthBytes & 31U) != 0U)) {
                if (pending0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                    pending0 = false;
                }
                if (pending1) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2Db);
                    pending1 = false;
                }
                CopyRowSegmentDmaChunksDb(globalChunk, 1);
                continue;
            }

            const bool useDb = (c & 1U) != 0U;
            const RowContext row = GetRowContext(outputRow);
            if (useDb) {
                if (pending1) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2Db);
                    pending1 = false;
                }
                FillRowSegmentCoalescedPadReadToLocal(yLocal1, outputRow, owBegin, owEnd);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                IssueCoalescedLocalSpanToGmNoWait(
                    yLocal1,
                    (row.rowBasePoint + owBegin) * this->depthBytes,
                    spanPointNum,
                    eventIdMte3ToMte2Db);
                pending1 = true;
            } else {
                if (pending0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                    pending0 = false;
                }
                FillRowSegmentCoalescedPadReadToLocal(yLocal0, outputRow, owBegin, owEnd);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                IssueCoalescedLocalSpanToGmNoWait(
                    yLocal0,
                    (row.rowBasePoint + owBegin) * this->depthBytes,
                    spanPointNum,
                    eventIdMte3ToMte2);
                pending0 = true;
            }
        }
        if (pending0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        if (pending1) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2Db);
        }
    }

    template <typename T>
    __aicore__ inline void FillTest3TypedStrideRowRangeToLocal(
        AscendC::GlobalTensor<T> &xTensor,
        AscendC::LocalTensor<T> yLocal,
        uint32_t outputRow,
        uint32_t owBegin,
        uint32_t owEnd,
        uint32_t unitNum,
        uint32_t localBase)
    {
        if (owBegin >= owEnd) {
            return;
        }
        const uint16_t blockLen = static_cast<uint16_t>((unitNum * sizeof(T)) >> 5);
        const uint16_t dstStride = static_cast<uint16_t>((this->blockSize - 1) * blockLen);
        const RowContext row = GetRowContext(outputRow);

        for (uint32_t blockW = 0; blockW < this->blockSize; ++blockW) {
            const uint32_t firstOw = FirstOwForBlockInRange(blockW, row.cropLeftMod, owBegin);
            if (firstOw >= owEnd) {
                continue;
            }

            const uint32_t pointNum = ((owEnd - 1 - firstOw) / this->blockSize) + 1;
            const uint32_t firstIw = (firstOw + this->cropLeft) / this->blockSize;
            const uint32_t inputPoint = CalcInputPointInRow(row, blockW, firstIw);
            const uint32_t localOffset = localBase + (firstOw - owBegin) * unitNum;

            AscendC::DataCopyParams copyParams;
            copyParams.blockCount = static_cast<uint16_t>(pointNum);
            copyParams.blockLen = blockLen;
            copyParams.srcStride = 0;
            copyParams.dstStride = dstStride;
            AscendC::DataCopy(yLocal[localOffset], xTensor[inputPoint * unitNum], copyParams);
        }
    }

    template <typename T>
    __aicore__ inline void FillTest3TypedStrideRowsToLocal(
        AscendC::GlobalTensor<T> &xTensor,
        AscendC::LocalTensor<T> yLocal,
        uint32_t outputRowBase,
        uint32_t rowNum,
        uint32_t unitNum)
    {
        const uint32_t rowDataNum = this->outWidth * unitNum;
        for (uint32_t r = 0; r < rowNum; ++r) {
            FillTest3TypedStrideRowRangeToLocal(
                xTensor,
                yLocal,
                outputRowBase + r,
                0,
                this->outWidth,
                unitNum,
                r * rowDataNum);
        }
    }

    template <typename T>
    __aicore__ inline void CopyTest3TypedStrideSegmentBlocking(
        AscendC::GlobalTensor<T> &xTensor,
        AscendC::GlobalTensor<T> &yTensor,
        AscendC::LocalTensor<T> yLocal,
        uint32_t outputRow,
        uint32_t owBegin,
        uint32_t owEnd,
        uint32_t unitNum)
    {
        if (owBegin >= owEnd) {
            return;
        }
        FillTest3TypedStrideRowRangeToLocal(xTensor, yLocal, outputRow, owBegin, owEnd, unitNum, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
        const uint32_t outputPoint = outputRow * this->outWidth + owBegin;
        const uint32_t copyDataNum = (owEnd - owBegin) * unitNum;
        AscendC::DataCopy(yTensor[outputPoint * unitNum], yLocal, copyDataNum);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    }

    template <typename T>
    __aicore__ inline void CopyTest3TypedStrideMergePointRange(
        AscendC::GlobalTensor<T> &xTensor,
        AscendC::GlobalTensor<T> &yTensor,
        uint32_t unitNum)
    {
        const uint32_t rowGroupNum = this->cropChunkPointNum;
        const uint32_t totalPointNum = this->outBatch * this->outHeight * this->outWidth;
        const uint32_t rowDataNum = this->outWidth * unitNum;
        auto yLocal0 = yBuf.Get<T>();
        auto yLocal1 = yBufDb.Get<T>();
        bool pending0 = false;
        bool pending1 = false;
        uint32_t bufferIdx = 0;
        uint32_t curPoint = this->startPoint;
        if (curPoint >= totalPointNum) {
            return;
        }
        const uint32_t pointEnd = MinU32(curPoint + this->corePointNum, totalPointNum);

        const uint32_t headOw = curPoint - (curPoint / this->outWidth) * this->outWidth;
        if (headOw != 0) {
            const uint32_t outputRow = curPoint / this->outWidth;
            const uint32_t owEnd = MinU32(this->outWidth, headOw + pointEnd - curPoint);
            CopyTest3TypedStrideSegmentBlocking(
                xTensor, yTensor, yLocal0, outputRow, headOw, owEnd, unitNum);
            curPoint += owEnd - headOw;
        }

        const uint32_t fullRowNum = (pointEnd - curPoint) / this->outWidth;
        const uint32_t fullRowBase = curPoint / this->outWidth;
        for (uint32_t rowOffset = 0; rowOffset < fullRowNum; rowOffset += rowGroupNum) {
            const uint32_t outputRowBase = fullRowBase + rowOffset;
            const uint32_t curRowNum = MinU32(rowGroupNum, fullRowNum - rowOffset);
            const uint32_t copyDataNum = curRowNum * rowDataNum;
            const uint32_t outputDataOffset = outputRowBase * rowDataNum;

            if (bufferIdx == 0) {
                if (pending0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                    pending0 = false;
                }
                FillTest3TypedStrideRowsToLocal(xTensor, yLocal0, outputRowBase, curRowNum, unitNum);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
                AscendC::DataCopy(yTensor[outputDataOffset], yLocal0, copyDataNum);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                pending0 = true;
            } else {
                if (pending1) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2Db);
                    pending1 = false;
                }
                FillTest3TypedStrideRowsToLocal(xTensor, yLocal1, outputRowBase, curRowNum, unitNum);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
                AscendC::DataCopy(yTensor[outputDataOffset], yLocal1, copyDataNum);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2Db);
                pending1 = true;
            }
            bufferIdx ^= 1;
        }
        curPoint += fullRowNum * this->outWidth;

        if (pending0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        if (pending1) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2Db);
        }

        if (curPoint < pointEnd) {
            const uint32_t outputRow = curPoint / this->outWidth;
            CopyTest3TypedStrideSegmentBlocking(
                xTensor, yTensor, yLocal0, outputRow, 0, pointEnd - curPoint, unitNum);
        }
    }

    __aicore__ inline void DispatchTest3TypedStrideMerge()
    {
        if (this->typeLength == 4) {
            CopyTest3TypedStrideMergePointRange(xGm32, yGm32, this->depthBytes / 4);
        } else if (this->typeLength == 2) {
            CopyTest3TypedStrideMergePointRange(xGm16, yGm16, this->depthBytes / 2);
        } else {
            DispatchNoCrop();
        }
    }


    // V130: generalized coalesced row-segment for test9.
    // For blockSize>=3 and 32B-aligned depth, each blockW phase reads contiguous
    // input columns.  MTE2 writes those phase streams directly into the final
    // interleaved UB layout using dstStride, then MTE3 writes one continuous
    // [owBegin, owEnd) span to GM.  This removes the GM-side strided write from
    // mode13 while avoiding Gather overhead.
    __aicore__ inline void FillRowSegmentBlockNCoalescedAlignedToLocal(
        AscendC::LocalTensor<int8_t> yLocal,
        uint32_t outputRow,
        uint32_t owBegin,
        uint32_t owEnd)
    {
        const RowContext row = GetRowContext(outputRow);
        const uint16_t depthBlockLen = static_cast<uint16_t>(this->depthBytes >> 5);
        const uint16_t dstGapBlocks = static_cast<uint16_t>((this->blockSize - 1) * depthBlockLen);

        for (uint32_t blockW = 0; blockW < this->blockSize; ++blockW) {
            const uint32_t firstOw = FirstOwForBlockInRange(blockW, row.cropLeftMod, owBegin);
            if (firstOw >= owEnd) {
                continue;
            }
            uint32_t pointNum = ((owEnd - 1 - firstOw) / this->blockSize) + 1;
            uint32_t firstIw = (firstOw + this->cropLeft) / this->blockSize;
            uint32_t inputPoint = CalcInputPointInRow(row, blockW, firstIw);
            uint32_t localByteOffset = (firstOw - owBegin) * this->depthBytes;

            // AscendC DataCopyParams blockCount is uint16_t.  The current
            // row-segment chunks are far below the limit, but keep a safe split.
            while (pointNum > 0) {
                const uint32_t curPointNum = MinU32(pointNum, static_cast<uint32_t>(4095));
                AscendC::DataCopyParams copyParams;
                copyParams.blockCount = static_cast<uint16_t>(curPointNum);
                copyParams.blockLen = depthBlockLen;
                copyParams.srcStride = 0;
                copyParams.dstStride = dstGapBlocks;
                AscendC::DataCopy(
                    yLocal[localByteOffset],
                    xGm[inputPoint * this->depthBytes],
                    copyParams);

                pointNum -= curPointNum;
                inputPoint += curPointNum;
                localByteOffset += curPointNum * this->blockSize * this->depthBytes;
            }
        }
    }

    __aicore__ inline void CopyRowSegmentBlockNCoalescedAlignedChunks(uint32_t globalChunkBase, uint32_t chunkNum)
    {
        auto yLocal = yBuf.Get<int8_t>();
        for (uint32_t c = 0; c < chunkNum; ++c) {
            const uint32_t globalChunk = globalChunkBase + c;
            const uint32_t outputRow = globalChunk / this->cropChunkNumPerRow;
            const uint32_t chunkIdx = globalChunk - outputRow * this->cropChunkNumPerRow;
            const uint32_t owBegin = chunkIdx * this->cropChunkPointNum;
            if (owBegin >= this->outWidth) {
                continue;
            }
            const uint32_t owEnd = MinU32(owBegin + this->cropChunkPointNum, this->outWidth);
            const uint32_t spanPointNum = owEnd - owBegin;
            if (spanPointNum == 0 || this->blockSize < 3 || ((this->depthBytes & 31U) != 0U)) {
                CopyRowSegmentDmaChunks(globalChunk, 1);
                continue;
            }
            FillRowSegmentBlockNCoalescedAlignedToLocal(yLocal, outputRow, owBegin, owEnd);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            const RowContext row = GetRowContext(outputRow);
            IssueCoalescedLocalSpanToGmNoWait(
                yLocal,
                (row.rowBasePoint + owBegin) * this->depthBytes,
                spanPointNum,
                eventIdMte3ToMte2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
    }

    __aicore__ inline void CopyRowSegmentBlockNCoalescedAlignedChunksDb(uint32_t globalChunkBase, uint32_t chunkNum)
    {
        auto yLocal0 = yBuf.Get<int8_t>();
        auto yLocal1 = yBufDb.Get<int8_t>();
        bool pending0 = false;
        bool pending1 = false;
        for (uint32_t c = 0; c < chunkNum; ++c) {
            const uint32_t globalChunk = globalChunkBase + c;
            const uint32_t outputRow = globalChunk / this->cropChunkNumPerRow;
            const uint32_t chunkIdx = globalChunk - outputRow * this->cropChunkNumPerRow;
            const uint32_t owBegin = chunkIdx * this->cropChunkPointNum;
            if (owBegin >= this->outWidth) {
                continue;
            }
            const uint32_t owEnd = MinU32(owBegin + this->cropChunkPointNum, this->outWidth);
            const uint32_t spanPointNum = owEnd - owBegin;
            if (spanPointNum == 0 || this->blockSize < 3 || ((this->depthBytes & 31U) != 0U)) {
                if (pending0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                    pending0 = false;
                }
                if (pending1) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2Db);
                    pending1 = false;
                }
                CopyRowSegmentDmaChunksDb(globalChunk, 1);
                continue;
            }

            const bool useDb = (c & 1U) != 0U;
            const RowContext row = GetRowContext(outputRow);
            if (useDb) {
                if (pending1) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2Db);
                    pending1 = false;
                }
                FillRowSegmentBlockNCoalescedAlignedToLocal(yLocal1, outputRow, owBegin, owEnd);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                IssueCoalescedLocalSpanToGmNoWait(
                    yLocal1,
                    (row.rowBasePoint + owBegin) * this->depthBytes,
                    spanPointNum,
                    eventIdMte3ToMte2Db);
                pending1 = true;
            } else {
                if (pending0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                    pending0 = false;
                }
                FillRowSegmentBlockNCoalescedAlignedToLocal(yLocal0, outputRow, owBegin, owEnd);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                IssueCoalescedLocalSpanToGmNoWait(
                    yLocal0,
                    (row.rowBasePoint + owBegin) * this->depthBytes,
                    spanPointNum,
                    eventIdMte3ToMte2);
                pending0 = true;
            }
        }
        if (pending0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        if (pending1) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2Db);
        }
    }

    __aicore__ inline void CopyRowSegmentDmaChunksDb(uint32_t globalChunkBase, uint32_t chunkNum)
    {
        auto yLocal0 = yBuf.Get<int8_t>();
        auto yLocal1 = yBufDb.Get<int8_t>();
        bool pending0 = false;
        bool pending1 = false;
        uint32_t ping = 0;

        for (uint32_t c = 0; c < chunkNum; ++c) {
            const uint32_t globalChunk = globalChunkBase + c;
            const uint32_t outputRow = globalChunk / this->cropChunkNumPerRow;
            const uint32_t chunkIdx = globalChunk - outputRow * this->cropChunkNumPerRow;
            const uint32_t owBegin = chunkIdx * this->cropChunkPointNum;
            if (owBegin >= this->outWidth) {
                continue;
            }

            const uint32_t owEnd = MinU32(owBegin + this->cropChunkPointNum, this->outWidth);
            const RowContext row = GetRowContext(outputRow);

            for (uint32_t blockW = 0; blockW < this->blockSize; ++blockW) {
                uint32_t firstOw = FirstOwForBlockInRange(blockW, row.cropLeftMod, owBegin);
                if (firstOw >= owEnd) {
                    continue;
                }

                const uint32_t pointNum = ((owEnd - 1 - firstOw) / this->blockSize) + 1;
                const uint32_t firstIw = (firstOw + this->cropLeft) / this->blockSize;
                const uint32_t inputPoint = CalcInputPointInRow(row, blockW, firstIw);
                const uint32_t outputPoint = row.rowBasePoint + firstOw;

                const bool useDb = (ping & 1U) != 0;
                if (useDb) {
                    if (pending1) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2Db);
                        pending1 = false;
                    }
                    CopyInputSegmentToLocalNormalDc(yLocal1, inputPoint * this->depthBytes, pointNum);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                    IssueLocalSegmentToGmStridedNoWaitNormalDc(
                        yLocal1, outputPoint * this->depthBytes, pointNum, eventIdMte3ToMte2Db);
                    pending1 = true;
                } else {
                    if (pending0) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                        pending0 = false;
                    }
                    CopyInputSegmentToLocalNormalDc(yLocal0, inputPoint * this->depthBytes, pointNum);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                    IssueLocalSegmentToGmStridedNoWaitNormalDc(
                        yLocal0, outputPoint * this->depthBytes, pointNum, eventIdMte3ToMte2);
                    pending0 = true;
                }
                ++ping;
            }
        }

        if (pending0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        if (pending1) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2Db);
        }
    }

    __aicore__ inline void CopyRowSegmentDmaChunks(uint32_t globalChunkBase, uint32_t chunkNum)
    {
        auto yLocal = yBuf.Get<int8_t>();

        for (uint32_t c = 0; c < chunkNum; ++c) {
            const uint32_t globalChunk = globalChunkBase + c;
            const uint32_t outputRow = globalChunk / this->cropChunkNumPerRow;
            const uint32_t chunkIdx = globalChunk - outputRow * this->cropChunkNumPerRow;
            const uint32_t owBegin = chunkIdx * this->cropChunkPointNum;
            if (owBegin >= this->outWidth) {
                continue;
            }

            const uint32_t owEnd = MinU32(owBegin + this->cropChunkPointNum, this->outWidth);
            const RowContext row = GetRowContext(outputRow);

            for (uint32_t blockW = 0; blockW < this->blockSize; ++blockW) {
                uint32_t firstOw = FirstOwForBlockInRange(blockW, row.cropLeftMod, owBegin);
                if (firstOw >= owEnd) {
                    continue;
                }

                const uint32_t pointNum = ((owEnd - 1 - firstOw) / this->blockSize) + 1;
                const uint32_t firstIw = (firstOw + this->cropLeft) / this->blockSize;
                const uint32_t inputPoint = CalcInputPointInRow(row, blockW, firstIw);
                const uint32_t outputPoint = row.rowBasePoint + firstOw;

                CopyInputSegmentToLocalPad(yLocal, inputPoint * this->depthBytes, pointNum);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                CopyLocalSegmentToGmStridedPad(yLocal, outputPoint * this->depthBytes, pointNum);
            }
        }
    }

    __aicore__ inline void CopyGmToLocalBytesPad(
        AscendC::LocalTensor<int8_t> yLocal,
        uint32_t inputByteOffset,
        uint32_t byteNum)
    {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = byteNum;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        AscendC::DataCopyPadExtParams<int8_t> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;

        AscendC::DataCopyPad(yLocal, xGm[inputByteOffset], copyParams, padParams);
    }

    __aicore__ inline void CopyLocalBytesToGmPad(
        AscendC::LocalTensor<int8_t> yLocal,
        uint32_t outputByteOffset,
        uint32_t byteNum)
    {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = byteNum;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::DataCopyPad(yGm[outputByteOffset], yLocal, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    }

    __aicore__ inline void CopyContinuousBytesViaUb(
        AscendC::LocalTensor<int8_t> yLocal,
        uint32_t inputByteOffset,
        uint32_t outputByteOffset,
        uint32_t byteNum)
    {
        if (byteNum == 0) {
            return;
        }

        if (((inputByteOffset | outputByteOffset | byteNum) & 31) == 0) {
            AscendC::DataCopy(yLocal, xGm[inputByteOffset], byteNum);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            CopyLocalToGmBytes(yLocal, outputByteOffset, byteNum);
        } else {
            CopyGmToLocalBytesPad(yLocal, inputByteOffset, byteNum);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            CopyLocalBytesToGmPad(yLocal, outputByteOffset, byteNum);
        }
    }

    __aicore__ inline void CopyDirectByteChunks(uint32_t globalChunkBase, uint32_t chunkNum)
    {
        auto yLocal = yBuf.Get<int8_t>();
        const uint32_t totalBytes = this->outBatch * this->outHeight * this->outWidth * this->depthBytes;
        const uint32_t chunkBytes = this->cropChunkPointNum;

        for (uint32_t c = 0; c < chunkNum; ++c) {
            const uint32_t byteBase = (globalChunkBase + c) * chunkBytes;
            if (byteBase >= totalBytes) {
                continue;
            }
            const uint32_t curBytes = MinU32(chunkBytes, totalBytes - byteBase);
            CopyContinuousBytesViaUb(yLocal, byteBase, byteBase, curBytes);
        }
    }

    __aicore__ inline void CopyBlock1CropSliceChunks(uint32_t globalChunkBase, uint32_t chunkNum)
    {
        auto yLocal = yBuf.Get<int8_t>();
        for (uint32_t c = 0; c < chunkNum; ++c) {
            const uint32_t globalChunk = globalChunkBase + c;
            const uint32_t outputRow = globalChunk / this->cropChunkNumPerRow;
            const uint32_t chunkIdx = globalChunk - outputRow * this->cropChunkNumPerRow;
            const uint32_t owBegin = chunkIdx * this->cropChunkPointNum;
            if (owBegin >= this->outWidth) {
                continue;
            }

            const uint32_t owEnd = MinU32(owBegin + this->cropChunkPointNum, this->outWidth);
            const uint32_t pointNum = owEnd - owBegin;
            if (pointNum == 0) {
                continue;
            }

            const uint32_t ob = outputRow / this->outHeight;
            const uint32_t oh = outputRow - ob * this->outHeight;
            const uint32_t ih = oh + this->cropTop;
            const uint32_t iw = owBegin + this->cropLeft;
            const uint32_t inputPoint = ((ob * this->inputHeight + ih) * this->inputWidth + iw);
            const uint32_t outputPoint = outputRow * this->outWidth + owBegin;
            const uint32_t byteNum = pointNum * this->depthBytes;

            CopyContinuousBytesViaUb(
                yLocal,
                inputPoint * this->depthBytes,
                outputPoint * this->depthBytes,
                byteNum);
        }
    }


    __aicore__ inline void CopyNoCropBlock2Depth128Fast(uint32_t globalPointBase, uint32_t pointNum)
    {
        // V59: exact test4-like path. Host guard: no-crop, blockSize==2, depthBytes==128, point [256,512).
        // Pair-unroll the blockW 0/1 recurrence inside each row. This keeps the proven 58D data movement
        // but removes the per-point blockW branch/toggle from the common even/odd pair body.
        auto yLocal = yBuf.Get<uint32_t>();
        constexpr uint32_t unitNum = 32; // 128B / sizeof(uint32_t)
        constexpr uint32_t groupPointNumConst = 96;
        const uint32_t totalPointNum = this->outBatch * this->outHeight * this->outWidth;

        for (uint32_t groupBase = 0; groupBase < pointNum; groupBase += groupPointNumConst) {
            const uint32_t groupPointNum = MinU32(pointNum - groupBase, groupPointNumConst);
            const uint32_t curGlobalPointBase = globalPointBase + groupBase;
            uint32_t localPoint = 0;
            while (localPoint < groupPointNum) {
                const uint32_t outputPoint = curGlobalPointBase + localPoint;
                if (outputPoint >= totalPointNum) {
                    break;
                }
                const uint32_t outputRow = outputPoint / this->outWidth;
                const uint32_t owBegin = outputPoint - outputRow * this->outWidth;
                const uint32_t rowPointNum = MinU32(groupPointNum - localPoint, this->outWidth - owBegin);

                const uint32_t ob = outputRow / this->outHeight;
                const uint32_t oh = outputRow - ob * this->outHeight;
                const uint32_t blockH = oh & 1U;
                const uint32_t ih = oh >> 1;
                const uint32_t ib0 = (blockH << 1) * this->outBatch + ob;
                const uint32_t ib1 = ib0 + this->outBatch;
                uint32_t iw = owBegin >> 1;
                uint32_t p = 0;

                // If this chunk starts at odd ow, handle the first blockW=1 point, then pairs are aligned.
                if (((owBegin & 1U) != 0U) && (p < rowPointNum)) {
                    const uint32_t inputPoint = ((ib1 * this->inputHeight + ih) * this->inputWidth + iw);
                    AscendC::DataCopy(
                        yLocal[(localPoint + p) * unitNum],
                        xGm32[inputPoint * unitNum],
                        unitNum);
                    ++p;
                    ++iw;
                }

                while (p + 1 < rowPointNum) {
                    const uint32_t inputPoint0 = ((ib0 * this->inputHeight + ih) * this->inputWidth + iw);
                    const uint32_t inputPoint1 = ((ib1 * this->inputHeight + ih) * this->inputWidth + iw);
                    AscendC::DataCopy(
                        yLocal[(localPoint + p) * unitNum],
                        xGm32[inputPoint0 * unitNum],
                        unitNum);
                    AscendC::DataCopy(
                        yLocal[(localPoint + p + 1) * unitNum],
                        xGm32[inputPoint1 * unitNum],
                        unitNum);
                    p += 2;
                    ++iw;
                }

                if (p < rowPointNum) {
                    const uint32_t inputPoint = ((ib0 * this->inputHeight + ih) * this->inputWidth + iw);
                    AscendC::DataCopy(
                        yLocal[(localPoint + p) * unitNum],
                        xGm32[inputPoint * unitNum],
                        unitNum);
                }

                localPoint += rowPointNum;
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            CopyLocalToGmTyped(yGm32, yLocal, curGlobalPointBase, groupPointNum, unitNum);
        }
    }

    __aicore__ inline void CopyNoCropDepthSplitChunks(uint32_t globalChunkBase, uint32_t chunkNum)
    {
        auto yLocal = yBuf.Get<int8_t>();
        const uint32_t totalPointNum = this->outBatch * this->outHeight * this->outWidth;
        const uint32_t depthChunkBytes = this->cropChunkPointNum;
        const uint32_t depthChunkNumPerPoint =
            (this->depthBytes + depthChunkBytes - 1) / depthChunkBytes;

        for (uint32_t c = 0; c < chunkNum; ++c) {
            const uint32_t globalChunk = globalChunkBase + c;
            const uint32_t outputPoint = globalChunk / depthChunkNumPerPoint;
            if (outputPoint >= totalPointNum) {
                continue;
            }
            const uint32_t depthChunkIdx = globalChunk - outputPoint * depthChunkNumPerPoint;
            const uint32_t depthByteOffset = depthChunkIdx * depthChunkBytes;
            if (depthByteOffset >= this->depthBytes) {
                continue;
            }
            const uint32_t curBytes = MinU32(depthChunkBytes, this->depthBytes - depthByteOffset);
            // V59: exact index for test7-like depth-split guard: no-crop, blockSize==2.
            const uint32_t outHW = this->outHeight * this->outWidth;
            const uint32_t ob = outputPoint / outHW;
            const uint32_t rem = outputPoint - ob * outHW;
            const uint32_t oh = rem / this->outWidth;
            const uint32_t ow = rem - oh * this->outWidth;
            const uint32_t blockH = oh & 1U;
            const uint32_t blockW = ow & 1U;
            const uint32_t ih = oh >> 1;
            const uint32_t iw = ow >> 1;
            const uint32_t ib = ((blockH << 1) + blockW) * this->outBatch + ob;
            const uint32_t inputPoint = ((ib * this->inputHeight + ih) * this->inputWidth + iw);
            const uint32_t inputByteOffset = inputPoint * this->depthBytes + depthByteOffset;
            const uint32_t outputByteOffset = outputPoint * this->depthBytes + depthByteOffset;

            // depthBytes and depthChunkBytes are 32B-aligned in this mode, so curBytes is also
            // 32B-aligned.  Use byte tensor DataCopy to split one large depth vector across cores.
            CopyContinuousBytesViaUb(yLocal, inputByteOffset, outputByteOffset, curBytes);
        }
    }


    template <typename T>
    __aicore__ inline void FillNoCropGroupedLocalTyped(
        AscendC::GlobalTensor<T> &xTensor,
        AscendC::LocalTensor<T> yLocal,
        uint32_t outputPointBase,
        uint32_t curPointNum,
        uint32_t unitNum)
    {
        uint32_t localPoint = 0;
        while (localPoint < curPointNum) {
            const uint32_t outputPoint = outputPointBase + localPoint;
            const uint32_t outputRow = outputPoint / this->outWidth;
            const uint32_t owBegin = outputPoint - outputRow * this->outWidth;
            const uint32_t rowPointNum = MinU32(curPointNum - localPoint, this->outWidth - owBegin);

            const RowContext row = GetRowContext(outputRow);

            if (this->cropWorkMode == MODE_NO_CROP_GROUP_MEDIUM) {
                uint32_t blockW = owBegin % this->blockSize;
                uint32_t iw = owBegin / this->blockSize;
                for (uint32_t i = 0; i < rowPointNum; ++i) {
                    const uint32_t inputPoint = CalcInputPointInRow(row, blockW, iw);
                    AscendC::DataCopy(
                        yLocal[(localPoint + i) * unitNum],
                        xTensor[inputPoint * unitNum],
                        unitNum);

                    ++blockW;
                    if (blockW == this->blockSize) {
                        blockW = 0;
                        ++iw;
                    }
                }
                localPoint += rowPointNum;
                continue;
            }

            const uint32_t batchBlockBase =
                (row.blockH * this->blockSize) * this->outBatch + row.ob;

            if (this->blockSize == 2) {
                const uint32_t rowBase0 =
                    ((batchBlockBase * this->inputHeight + row.ih) * this->inputWidth);
                const uint32_t rowBase1 =
                    (((batchBlockBase + this->outBatch) * this->inputHeight + row.ih) * this->inputWidth);
                uint32_t i = 0;
                uint32_t blockW = owBegin & 1U;
                uint32_t iw = owBegin >> 1;

                if (blockW == 1U && i < rowPointNum) {
                    AscendC::DataCopy(
                        yLocal[(localPoint + i) * unitNum],
                        xTensor[(rowBase1 + iw) * unitNum],
                        unitNum);
                    ++i;
                    ++iw;
                }

                for (; i + 1 < rowPointNum; i += 2, ++iw) {
                    AscendC::DataCopy(
                        yLocal[(localPoint + i) * unitNum],
                        xTensor[(rowBase0 + iw) * unitNum],
                        unitNum);
                    AscendC::DataCopy(
                        yLocal[(localPoint + i + 1) * unitNum],
                        xTensor[(rowBase1 + iw) * unitNum],
                        unitNum);
                }

                if (i < rowPointNum) {
                    AscendC::DataCopy(
                        yLocal[(localPoint + i) * unitNum],
                        xTensor[(rowBase0 + iw) * unitNum],
                        unitNum);
                }
            } else if (this->blockSize == 4) {
                const uint32_t rowBase0 =
                    ((batchBlockBase * this->inputHeight + row.ih) * this->inputWidth);
                const uint32_t rowBase1 =
                    (((batchBlockBase + this->outBatch) * this->inputHeight + row.ih) * this->inputWidth);
                const uint32_t rowBase2 =
                    (((batchBlockBase + 2 * this->outBatch) * this->inputHeight + row.ih) * this->inputWidth);
                const uint32_t rowBase3 =
                    (((batchBlockBase + 3 * this->outBatch) * this->inputHeight + row.ih) * this->inputWidth);
                uint32_t i = 0;
                uint32_t blockW = owBegin & 3U;
                uint32_t iw = owBegin >> 2;

                while (blockW != 0U && i < rowPointNum) {
                    uint32_t rowBase = rowBase0;
                    if (blockW == 1U) {
                        rowBase = rowBase1;
                    } else if (blockW == 2U) {
                        rowBase = rowBase2;
                    } else {
                        rowBase = rowBase3;
                    }
                    AscendC::DataCopy(
                        yLocal[(localPoint + i) * unitNum],
                        xTensor[(rowBase + iw) * unitNum],
                        unitNum);
                    ++i;
                    ++blockW;
                    if (blockW == 4U) {
                        blockW = 0;
                        ++iw;
                    }
                }

                for (; i + 3 < rowPointNum; i += 4, ++iw) {
                    AscendC::DataCopy(
                        yLocal[(localPoint + i) * unitNum],
                        xTensor[(rowBase0 + iw) * unitNum],
                        unitNum);
                    AscendC::DataCopy(
                        yLocal[(localPoint + i + 1) * unitNum],
                        xTensor[(rowBase1 + iw) * unitNum],
                        unitNum);
                    AscendC::DataCopy(
                        yLocal[(localPoint + i + 2) * unitNum],
                        xTensor[(rowBase2 + iw) * unitNum],
                        unitNum);
                    AscendC::DataCopy(
                        yLocal[(localPoint + i + 3) * unitNum],
                        xTensor[(rowBase3 + iw) * unitNum],
                        unitNum);
                }

                blockW = 0;
                while (i < rowPointNum) {
                    uint32_t rowBase = rowBase0;
                    if (blockW == 1U) {
                        rowBase = rowBase1;
                    } else if (blockW == 2U) {
                        rowBase = rowBase2;
                    } else if (blockW == 3U) {
                        rowBase = rowBase3;
                    }
                    AscendC::DataCopy(
                        yLocal[(localPoint + i) * unitNum],
                        xTensor[(rowBase + iw) * unitNum],
                        unitNum);
                    ++i;
                    ++blockW;
                }
            } else {
                uint32_t blockW = owBegin % this->blockSize;
                uint32_t iw = owBegin / this->blockSize;
                for (uint32_t i = 0; i < rowPointNum; ++i) {
                    const uint32_t inputPoint = CalcInputPointInRow(row, blockW, iw);
                    AscendC::DataCopy(
                        yLocal[(localPoint + i) * unitNum],
                        xTensor[inputPoint * unitNum],
                        unitNum);

                    ++blockW;
                    if (blockW == this->blockSize) {
                        blockW = 0;
                        ++iw;
                    }
                }
            }
            localPoint += rowPointNum;
        }
    }

    template <typename T>
    __aicore__ inline void IssueLocalToGmTypedNoWait(
        AscendC::GlobalTensor<T> &yTensor,
        AscendC::LocalTensor<T> yLocal,
        uint32_t outputPointBase,
        uint32_t pointNum,
        uint32_t unitNum,
        AscendC::TEventID doneEvent)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::DataCopy(yTensor[outputPointBase * unitNum], yLocal, pointNum * unitNum);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(doneEvent);
    }


    // V36: test10 experimental path.
    // Instead of per-output-point small reads, copy the two blockSize==2 input streams
    // directly into the final interleaved UB layout using MTE2 dstStride, then keep
    // the existing continuous UB->GM writeback.  Odd prefix/tail points fall back
    // to the safe single-point DataCopy used by the original grouped path.
    template <typename T>
    __aicore__ inline void FillNoCropBlock2Mte2InterleaveTyped(
        AscendC::GlobalTensor<T> &xTensor,
        AscendC::LocalTensor<T> yLocal,
        uint32_t outputPointBase,
        uint32_t curPointNum,
        uint32_t unitNum)
    {
        uint32_t localPoint = 0;
        const uint16_t depthBlockLen = static_cast<uint16_t>(this->depthBytes >> 5);
        const uint16_t ubGapBlocks = static_cast<uint16_t>(this->depthBytes >> 5);

        while (localPoint < curPointNum) {
            const uint32_t outputPoint = outputPointBase + localPoint;
            const uint32_t outputRow = outputPoint / this->outWidth;
            const uint32_t owBegin = outputPoint - outputRow * this->outWidth;
            const uint32_t rowPointNum = MinU32(curPointNum - localPoint, this->outWidth - owBegin);
            const RowContext row = GetRowContext(outputRow);

            const uint32_t batchBlockBase =
                (row.blockH * this->blockSize) * this->outBatch + row.ob;
            const uint32_t rowBase0 =
                ((batchBlockBase * this->inputHeight + row.ih) * this->inputWidth);
            const uint32_t rowBase1 =
                (((batchBlockBase + this->outBatch) * this->inputHeight + row.ih) * this->inputWidth);

            uint32_t i = 0;
            uint32_t blockW = owBegin & 1U;
            uint32_t iw = owBegin >> 1;

            // If the chunk starts from blockW1, handle that one point safely so that
            // the remaining main segment starts from a blockW0/blockW1 pair.
            if (blockW == 1U && i < rowPointNum) {
                AscendC::DataCopy(
                    yLocal[(localPoint + i) * unitNum],
                    xTensor[(rowBase1 + iw) * unitNum],
                    unitNum);
                ++i;
                ++iw;
            }

            const uint32_t pairCount = (rowPointNum - i) >> 1;
            if (pairCount > 0 && depthBlockLen > 0) {
                AscendC::DataCopyParams p;
                p.blockCount = static_cast<uint16_t>(pairCount);
                p.blockLen = depthBlockLen;
                p.srcStride = 0;
                p.dstStride = ubGapBlocks;

                // blockW0 stream -> UB slots 0,2,4...
                AscendC::DataCopy(
                    yLocal[(localPoint + i) * unitNum],
                    xTensor[(rowBase0 + iw) * unitNum],
                    p);
                // blockW1 stream -> UB slots 1,3,5...
                AscendC::DataCopy(
                    yLocal[(localPoint + i + 1) * unitNum],
                    xTensor[(rowBase1 + iw) * unitNum],
                    p);

                i += pairCount << 1;
                iw += pairCount;
            }

            if (i < rowPointNum) {
                AscendC::DataCopy(
                    yLocal[(localPoint + i) * unitNum],
                    xTensor[(rowBase0 + iw) * unitNum],
                    unitNum);
            }

            localPoint += rowPointNum;
        }
    }

    template <typename T>
    __aicore__ inline void CopyNoCropBlock2Mte2InterleaveChunksTypedDb(
        AscendC::GlobalTensor<T> &xTensor,
        AscendC::GlobalTensor<T> &yTensor,
        uint32_t unitNum)
    {
        auto yLocal0 = yBuf.Get<T>();
        auto yLocal1 = yBufDb.Get<T>();
        const uint32_t totalPointNum = this->outBatch * this->outHeight * this->outWidth;
        bool pending0 = false;
        bool pending1 = false;

        for (uint32_t c = 0; c < this->corePointNum; ++c) {
            const uint32_t globalChunk = this->startPoint + c;
            const uint32_t outputPointBase = globalChunk * this->cropChunkPointNum;
            if (outputPointBase >= totalPointNum) {
                continue;
            }
            const uint32_t curPointNum = MinU32(this->cropChunkPointNum, totalPointNum - outputPointBase);
            const bool useDb = (c & 1U) != 0;
            if (useDb) {
                if (pending1) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2Db);
                    pending1 = false;
                }
                FillNoCropBlock2Mte2InterleaveTyped(xTensor, yLocal1, outputPointBase, curPointNum, unitNum);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                IssueLocalToGmTypedNoWait(yTensor, yLocal1, outputPointBase, curPointNum, unitNum, eventIdMte3ToMte2Db);
                pending1 = true;
            } else {
                if (pending0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                    pending0 = false;
                }
                FillNoCropBlock2Mte2InterleaveTyped(xTensor, yLocal0, outputPointBase, curPointNum, unitNum);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                IssueLocalToGmTypedNoWait(yTensor, yLocal0, outputPointBase, curPointNum, unitNum, eventIdMte3ToMte2);
                pending0 = true;
            }
        }
        if (pending0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        if (pending1) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2Db);
        }
    }

    template <typename T>
    __aicore__ inline void CopyNoCropGroupedChunksTypedDb(
        AscendC::GlobalTensor<T> &xTensor,
        AscendC::GlobalTensor<T> &yTensor,
        uint32_t unitNum)
    {
        auto yLocal0 = yBuf.Get<T>();
        auto yLocal1 = yBufDb.Get<T>();
        const uint32_t totalPointNum = this->outBatch * this->outHeight * this->outWidth;
        bool pending0 = false;
        bool pending1 = false;

        for (uint32_t c = 0; c < this->corePointNum; ++c) {
            const uint32_t globalChunk = this->startPoint + c;
            const uint32_t outputPointBase = globalChunk * this->cropChunkPointNum;
            if (outputPointBase >= totalPointNum) {
                continue;
            }
            const uint32_t curPointNum = MinU32(this->cropChunkPointNum, totalPointNum - outputPointBase);
            const bool useDb = (c & 1U) != 0;
            if (useDb) {
                if (pending1) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2Db);
                    pending1 = false;
                }
                FillNoCropGroupedLocalTyped(xTensor, yLocal1, outputPointBase, curPointNum, unitNum);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                IssueLocalToGmTypedNoWait(yTensor, yLocal1, outputPointBase, curPointNum, unitNum, eventIdMte3ToMte2Db);
                pending1 = true;
            } else {
                if (pending0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                    pending0 = false;
                }
                FillNoCropGroupedLocalTyped(xTensor, yLocal0, outputPointBase, curPointNum, unitNum);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
                IssueLocalToGmTypedNoWait(yTensor, yLocal0, outputPointBase, curPointNum, unitNum, eventIdMte3ToMte2);
                pending0 = true;
            }
        }
        if (pending0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        if (pending1) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2Db);
        }
    }

    template <typename T>
    __aicore__ inline void CopyNoCropGroupedChunksTyped(
        AscendC::GlobalTensor<T> &xTensor,
        AscendC::GlobalTensor<T> &yTensor,
        uint32_t unitNum)
    {
        auto yLocal = yBuf.Get<T>();
        const uint32_t totalPointNum = this->outBatch * this->outHeight * this->outWidth;

        for (uint32_t c = 0; c < this->corePointNum; ++c) {
            const uint32_t globalChunk = this->startPoint + c;
            const uint32_t outputPointBase = globalChunk * this->cropChunkPointNum;
            if (outputPointBase >= totalPointNum) {
                continue;
            }
            const uint32_t curPointNum = MinU32(this->cropChunkPointNum, totalPointNum - outputPointBase);

            // V98: keep the V97 row-local indexing, then add blockSize==2/4 unrolled
            // no-crop inner loops.  This targets the test10-like region: no crop,
            // small depth, many output points.  For one output row, blockH/ih/ob are
            // fixed; for blockSize 2/4, each complete block group shares iw and only
            // switches input batch block.  This removes per-point function calls and
            // most per-point div/mod/branch work while keeping the same GM access order.
            uint32_t localPoint = 0;
            while (localPoint < curPointNum) {
                const uint32_t outputPoint = outputPointBase + localPoint;
                const uint32_t outputRow = outputPoint / this->outWidth;
                const uint32_t owBegin = outputPoint - outputRow * this->outWidth;
                const uint32_t rowPointNum = MinU32(curPointNum - localPoint, this->outWidth - owBegin);

                const RowContext row = GetRowContext(outputRow);

                // V101: MEDIUM uses the lighter V97 generic row-local recurrence.
                // The V98 blockSize==2/4 unrolled code helps the large test10-like
                // case, but it regressed shorter medium cases such as tests 3/4/7.
                if (this->cropWorkMode == MODE_NO_CROP_GROUP_MEDIUM) {
                    uint32_t blockW = owBegin % this->blockSize;
                    uint32_t iw = owBegin / this->blockSize;
                    for (uint32_t i = 0; i < rowPointNum; ++i) {
                        const uint32_t inputPoint = CalcInputPointInRow(row, blockW, iw);
                        AscendC::DataCopy(
                            yLocal[(localPoint + i) * unitNum],
                            xTensor[inputPoint * unitNum],
                            unitNum);

                        ++blockW;
                        if (blockW == this->blockSize) {
                            blockW = 0;
                            ++iw;
                        }
                    }
                    localPoint += rowPointNum;
                    continue;
                }

                const uint32_t batchBlockBase =
                    (row.blockH * this->blockSize) * this->outBatch + row.ob;

                if (this->blockSize == 2) {
                    const uint32_t rowBase0 =
                        ((batchBlockBase * this->inputHeight + row.ih) * this->inputWidth);
                    const uint32_t rowBase1 =
                        (((batchBlockBase + this->outBatch) * this->inputHeight + row.ih) * this->inputWidth);
                    uint32_t i = 0;
                    uint32_t blockW = owBegin & 1U;
                    uint32_t iw = owBegin >> 1;

                    if (blockW == 1U && i < rowPointNum) {
                        AscendC::DataCopy(
                            yLocal[(localPoint + i) * unitNum],
                            xTensor[(rowBase1 + iw) * unitNum],
                            unitNum);
                        ++i;
                        ++iw;
                    }

                    for (; i + 1 < rowPointNum; i += 2, ++iw) {
                        AscendC::DataCopy(
                            yLocal[(localPoint + i) * unitNum],
                            xTensor[(rowBase0 + iw) * unitNum],
                            unitNum);
                        AscendC::DataCopy(
                            yLocal[(localPoint + i + 1) * unitNum],
                            xTensor[(rowBase1 + iw) * unitNum],
                            unitNum);
                    }

                    if (i < rowPointNum) {
                        AscendC::DataCopy(
                            yLocal[(localPoint + i) * unitNum],
                            xTensor[(rowBase0 + iw) * unitNum],
                            unitNum);
                    }
                } else if (this->blockSize == 4) {
                    const uint32_t rowBase0 =
                        ((batchBlockBase * this->inputHeight + row.ih) * this->inputWidth);
                    const uint32_t rowBase1 =
                        (((batchBlockBase + this->outBatch) * this->inputHeight + row.ih) * this->inputWidth);
                    const uint32_t rowBase2 =
                        (((batchBlockBase + 2 * this->outBatch) * this->inputHeight + row.ih) * this->inputWidth);
                    const uint32_t rowBase3 =
                        (((batchBlockBase + 3 * this->outBatch) * this->inputHeight + row.ih) * this->inputWidth);
                    uint32_t i = 0;
                    uint32_t blockW = owBegin & 3U;
                    uint32_t iw = owBegin >> 2;

                    while (blockW != 0U && i < rowPointNum) {
                        uint32_t rowBase = rowBase0;
                        if (blockW == 1U) {
                            rowBase = rowBase1;
                        } else if (blockW == 2U) {
                            rowBase = rowBase2;
                        } else {
                            rowBase = rowBase3;
                        }
                        AscendC::DataCopy(
                            yLocal[(localPoint + i) * unitNum],
                            xTensor[(rowBase + iw) * unitNum],
                            unitNum);
                        ++i;
                        ++blockW;
                        if (blockW == 4U) {
                            blockW = 0;
                            ++iw;
                        }
                    }

                    for (; i + 3 < rowPointNum; i += 4, ++iw) {
                        AscendC::DataCopy(
                            yLocal[(localPoint + i) * unitNum],
                            xTensor[(rowBase0 + iw) * unitNum],
                            unitNum);
                        AscendC::DataCopy(
                            yLocal[(localPoint + i + 1) * unitNum],
                            xTensor[(rowBase1 + iw) * unitNum],
                            unitNum);
                        AscendC::DataCopy(
                            yLocal[(localPoint + i + 2) * unitNum],
                            xTensor[(rowBase2 + iw) * unitNum],
                            unitNum);
                        AscendC::DataCopy(
                            yLocal[(localPoint + i + 3) * unitNum],
                            xTensor[(rowBase3 + iw) * unitNum],
                            unitNum);
                    }

                    blockW = 0;
                    while (i < rowPointNum) {
                        uint32_t rowBase = rowBase0;
                        if (blockW == 1U) {
                            rowBase = rowBase1;
                        } else if (blockW == 2U) {
                            rowBase = rowBase2;
                        } else if (blockW == 3U) {
                            rowBase = rowBase3;
                        }
                        AscendC::DataCopy(
                            yLocal[(localPoint + i) * unitNum],
                            xTensor[(rowBase + iw) * unitNum],
                            unitNum);
                        ++i;
                        ++blockW;
                    }
                } else {
                    uint32_t blockW = owBegin % this->blockSize;
                    uint32_t iw = owBegin / this->blockSize;
                    for (uint32_t i = 0; i < rowPointNum; ++i) {
                        const uint32_t inputPoint = CalcInputPointInRow(row, blockW, iw);
                        AscendC::DataCopy(
                            yLocal[(localPoint + i) * unitNum],
                            xTensor[inputPoint * unitNum],
                            unitNum);

                        ++blockW;
                        if (blockW == this->blockSize) {
                            blockW = 0;
                            ++iw;
                        }
                    }
                }
                localPoint += rowPointNum;
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            CopyLocalToGmTyped(yTensor, yLocal, outputPointBase, curPointNum, unitNum);
        }
    }

    __aicore__ inline void DispatchNoCropGroupedChunks()
    {
        if ((this->depthBytes & 3) == 0) {
            if (this->cropWorkMode == MODE_NO_CROP_BLOCK2_MTE2_INTERLEAVE_DB) {
                CopyNoCropBlock2Mte2InterleaveChunksTypedDb(xGm32, yGm32, this->depthBytes / 4);
            } else if (this->cropWorkMode == MODE_NO_CROP_GROUP_LARGE_DB) {
                CopyNoCropGroupedChunksTypedDb(xGm32, yGm32, this->depthBytes / 4);
            } else {
                CopyNoCropGroupedChunksTyped(xGm32, yGm32, this->depthBytes / 4);
            }
        } else if ((this->depthBytes & 1) == 0) {
            if (this->cropWorkMode == MODE_NO_CROP_BLOCK2_MTE2_INTERLEAVE_DB) {
                CopyNoCropBlock2Mte2InterleaveChunksTypedDb(xGm16, yGm16, this->depthBytes / 2);
            } else if (this->cropWorkMode == MODE_NO_CROP_GROUP_LARGE_DB) {
                CopyNoCropGroupedChunksTypedDb(xGm16, yGm16, this->depthBytes / 2);
            } else {
                CopyNoCropGroupedChunksTyped(xGm16, yGm16, this->depthBytes / 2);
            }
        } else {
            DispatchNoCrop();
        }
    }

    __aicore__ inline void DispatchNoCrop()
    {
        if ((this->depthBytes & 3) == 0) {
            CopyNoCropFastTyped(xGm32, yGm32, this->depthBytes / 4);
        } else if ((this->depthBytes & 1) == 0) {
            CopyNoCropFastTyped(xGm16, yGm16, this->depthBytes / 2);
        } else {
            CopyPointRangeTyped(xGm, yGm, this->startPoint, this->corePointNum, this->depthBytes);
        }
    }

    __aicore__ inline void DispatchRows()
    {
        if ((this->depthBytes & 3) == 0) {
            CopyCropRowsTyped(xGm32, yGm32, this->startPoint, this->corePointNum, this->depthBytes / 4);
        } else if ((this->depthBytes & 1) == 0) {
            CopyCropRowsTyped(xGm16, yGm16, this->startPoint, this->corePointNum, this->depthBytes / 2);
        } else {
            CopyCropRowsTyped(xGm, yGm, this->startPoint, this->corePointNum, this->depthBytes);
        }
    }

    __aicore__ inline void DispatchAlignedChunk()
    {
        if ((this->depthBytes & 3) == 0) {
            CopyCropChunksTyped(xGm32, yGm32, this->startPoint, this->corePointNum, this->depthBytes / 4);
        } else if ((this->depthBytes & 1) == 0) {
            CopyCropChunksTyped(xGm16, yGm16, this->startPoint, this->corePointNum, this->depthBytes / 2);
        } else {
            CopyCropChunksTyped(xGm, yGm, this->startPoint, this->corePointNum, this->depthBytes);
        }
    }

    __aicore__ inline void DispatchRowAlignedChunks()
    {
        if ((this->depthBytes & 3) == 0) {
            CopyCropRowsAlignedChunksTyped(xGm32, yGm32, this->startPoint, this->corePointNum, this->depthBytes / 4);
        } else if ((this->depthBytes & 1) == 0) {
            CopyCropRowsAlignedChunksTyped(xGm16, yGm16, this->startPoint, this->corePointNum, this->depthBytes / 2);
        } else {
            CopyCropRowsAlignedChunksTyped(xGm, yGm, this->startPoint, this->corePointNum, this->depthBytes);
        }
    }

    __aicore__ inline void DispatchRowParallelChunks()
    {
        if ((this->depthBytes & 3) == 0) {
            CopyCropRowChunksParallelTyped(xGm32, yGm32, this->startPoint, this->corePointNum, this->depthBytes / 4);
        } else if ((this->depthBytes & 1) == 0) {
            CopyCropRowChunksParallelTyped(xGm16, yGm16, this->startPoint, this->corePointNum, this->depthBytes / 2);
        } else {
            CopyCropRowChunksParallelTyped(xGm, yGm, this->startPoint, this->corePointNum, this->depthBytes);
        }
    }

    __aicore__ inline void DispatchCropBlock2Gather()
    {
        if (this->typeLength == 4) {
            CopyCropBlock2GatherChunksTyped(
                xGmFp32,
                yGmFp32,
                this->startPoint,
                this->corePointNum,
                this->depth);
        } else if (this->typeLength == 2) {
            CopyCropBlock2GatherChunksTyped(
                xGmFp16,
                yGmFp16,
                this->startPoint,
                this->corePointNum,
                this->depth);
        } else {
            for (uint32_t c = 0; c < this->corePointNum; ++c) {
                const uint32_t globalChunk = this->startPoint + c;
                const uint32_t outputRow = globalChunk / this->cropChunkNumPerRow;
                const uint32_t chunkIdx = globalChunk - outputRow * this->cropChunkNumPerRow;
                const uint32_t owBegin = chunkIdx * this->cropChunkPointNum;
                if (owBegin < this->outWidth) {
                    const uint32_t owEnd = MinU32(owBegin + this->cropChunkPointNum, this->outWidth);
                    CopyCropRowRangeTyped(xGm, yGm, outputRow, owBegin, owEnd, this->depthBytes);
                }
            }
        }
    }

private:
    AscendC::TPipe *pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yBufDb;
    AscendC::TEventID eventIdMte2ToV;
    AscendC::TEventID eventIdMte2ToMte3;
    AscendC::TEventID eventIdVToMte3;
    AscendC::TEventID eventIdMte3ToMte2;
    AscendC::TEventID eventIdMte3ToMte2Db;

    AscendC::GlobalTensor<int8_t> xGm;
    AscendC::GlobalTensor<int8_t> yGm;
    AscendC::GlobalTensor<uint16_t> xGm16;
    AscendC::GlobalTensor<uint16_t> yGm16;
    AscendC::GlobalTensor<uint32_t> xGm32;
    AscendC::GlobalTensor<uint32_t> yGm32;
    AscendC::GlobalTensor<half> xGmFp16;
    AscendC::GlobalTensor<half> yGmFp16;
    AscendC::GlobalTensor<float> xGmFp32;
    AscendC::GlobalTensor<float> yGmFp32;

    uint32_t batch;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t depth;
    uint32_t typeLength;
    uint32_t depthBytes;

    uint32_t outBatch;
    uint32_t outHeight;
    uint32_t outWidth;

    uint32_t blockSize;
    uint32_t cropTop;
    uint32_t cropBottom;
    uint32_t cropLeft;
    uint32_t cropRight;
    uint32_t cropWorkMode;
    uint32_t cropChunkPointNum;
    uint32_t cropChunkNumPerRow;

    uint32_t startPoint;
    uint32_t corePointNum;
    uint32_t tileNum;
    uint32_t tilePointNum;
    uint32_t tailPointNum;
};

extern "C" __global__ __aicore__ void batch_to_space(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;

    AscendC::TPipe pipe;

    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingData, tilingData, tiling);

    KernelBatchToSpace op;
    op.Init(&pipe, x, y, tilingData);
    op.Process();
}
