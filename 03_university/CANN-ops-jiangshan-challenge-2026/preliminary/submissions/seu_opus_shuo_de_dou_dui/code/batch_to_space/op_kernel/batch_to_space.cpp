#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace {

// V12 note: the kernel paths are unchanged from V10; the difference is purely in
// host routing. row_scatter is now a *pure fallback* used only when the
// continuous-write (block_interleave) condition is not met. In particular ALL
// crops that are 32B-aligned are now routed to ProcessBlockInterleaveCrop, which
// already performs block_interleave + padding (per-phase strided MTE2 loads that
// skip the cropped columns followed by ONE contiguous MTE3 write) for any crop
// size -- so the large-crop row_scatter case is gone. Huge C that does not fit
// UB still uses ProcessChannelFallback, which moves channelTileElems per copy
// (not single elements) to keep the scalar chunk loop at the fewest iterations.
//
// V10 kernel = V9 kernel + flag-merge (batched MTE2/MTE3 events). For the no-crop
// contiguous paths the host may pick flagMergeGroup_ > 1, in which case the
// kernel processes tiles in groups that share ONE MTE2->MTE3 / MTE3->MTE2 event
// pair instead of one pair per tile -- cutting TPipe event traffic on small
// tiles. Flag-merge is mutually exclusive with the double-buffer pipe (the host
// sets only one of enableDoubleBuffer_ / flagMergeGroup_). Summary of the paths:
//  * block_interleave continuous write (mode 4/5): for pixelBytes % 32 == 0 the
//    blockSize input rows are read contiguously into strided UB slots and the
//    whole interleaved output row is flushed with ONE contiguous MTE3 write,
//    removing the strided MTE3 write that wastes ~(block-1)/block of the bus.
//    Generalized to any block_size; crop variant uses an owExpStart%block phase
//    (host only selects the crop variant when the crop is small).
//  * double-buffer software pipeline (PopStackBuffer 2nd UB): next job's MTE2
//    load overlaps current job's MTE3 store; enabled by host (enableDoubleBuffer_)
//    only when jobs/core and bytes/core clear the thresholds.
//  * row_scatter (double-buffer) fallback for small / non-32B C / large-crop;
//    channel fallback for huge C (chunk sized for the fewest scalar iterations).
template <typename T>
class KernelBatchToSpace {
public:
    __aicore__ inline KernelBatchToSpace() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const uint64_t totalJobs,
                                const uint32_t mode, const uint32_t inputHeight,
                                const uint32_t inputWidth, const uint32_t channels,
                                const uint32_t outputBatch, const uint32_t outputHeight,
                                const uint32_t outputWidth, const uint32_t blockSize,
                                const int32_t cropTop, const int32_t cropBottom,
                                const int32_t cropLeft, const int32_t cropRight,
                                const uint32_t tilePixels, const uint32_t tilesPerRow,
                                const uint32_t channelTileElems, const uint32_t bufferBytes,
                                const uint32_t enableDoubleBuffer,
                                const uint32_t flagMergeGroup)
    {
        totalJobs_ = totalJobs;
        mode_ = mode;
        inputHeight_ = inputHeight;
        inputWidth_ = inputWidth;
        channels_ = channels;
        outputBatch_ = outputBatch;
        outputHeight_ = outputHeight;
        outputWidth_ = outputWidth;
        blockSize_ = blockSize;
        cropTop_ = cropTop;
        cropBottom_ = cropBottom;
        cropLeft_ = cropLeft;
        cropRight_ = cropRight;
        tilePixels_ = tilePixels;
        tilesPerRow_ = tilesPerRow;
        channelTileElems_ = channelTileElems;
        bufferBytes_ = bufferBytes;
        enableDoubleBuffer_ = enableDoubleBuffer;
        flagMergeGroup_ = (flagMergeGroup == 0U) ? 1U : flagMergeGroup;
        // bufferBytes_ holds flagMergeGroup_ tiles back-to-back; this is the
        // per-tile element stride used to address tile k at tileStrideElems_*k.
        tileStrideElems_ = (bufferBytes_ / flagMergeGroup_) / static_cast<uint32_t>(sizeof(T));

        pixelBytes_ = channels_ * static_cast<uint32_t>(sizeof(T));
        blocksPerPixel_ = pixelBytes_ / 32U;
        rowDstGapBytes_ = (blockSize_ - 1U) * pixelBytes_;
        expandedBottom_ = static_cast<int32_t>(inputHeight_ * blockSize_) - cropBottom_;
        expandedRight_ = static_cast<int32_t>(inputWidth_ * blockSize_) - cropRight_;
        noCrop_ = (cropTop_ == 0 && cropBottom_ == 0 && cropLeft_ == 0 && cropRight_ == 0);

        xGm_.SetGlobalBuffer((__gm__ T *)x);
        yGm_.SetGlobalBuffer((__gm__ T *)y);
        pipe_.InitBuffer(dataBuffer_, bufferBytes_);

        eventMte2ToMte3_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
        eventMte3ToMte2_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();

        const uint64_t coreCount = static_cast<uint64_t>(GetBlockNum());
        const uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t base = totalJobs_ / coreCount;
        const uint64_t extra = totalJobs_ - base * coreCount;
        jobStart_ = coreIdx * base + ((coreIdx < extra) ? coreIdx : extra);
        jobCount_ = base + ((coreIdx < extra) ? 1ULL : 0ULL);

        // V8: the double-buffer cost/benefit decision (pipeline-eligible mode,
        // enough jobs/core and bytes/core) is now made in the host tiling and
        // delivered as enableDoubleBuffer_. The kernel only has to confirm a
        // second UB tile can actually be claimed via PopStackBuffer; if so it
        // ping-pongs dbuf1_/dataBuffer_ so job i+1's load overlaps job i's store.
        // When the host disabled it we never allocate the extra MTE2/MTE3 events.
        if (enableDoubleBuffer_ != 0U) {
            LocalTensor<T> rest;
            if (PopStackBuffer<T, TPosition::VECIN>(rest) &&
                rest.GetSize() * sizeof(T) >= static_cast<uint64_t>(bufferBytes_)) {
                dbuf1_ = rest;
                doubleBuf_ = true;
                eventMte2ToMte3b_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
                eventMte3ToMte2b_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
            }
        }
    }

    __aicore__ inline void Process()
    {
        if (mode_ == BATCH_TO_SPACE_MODE_BLOCK_INTERLEAVE_NO_CROP) {
            if (flagMergeGroup_ > 1U) {
                ProcessBlockInterleaveNoCropMerged();
            } else if (doubleBuf_) {
                ProcessBlockInterleaveNoCropPipe();
            } else {
                for (uint64_t i = 0ULL; i < jobCount_; ++i) {
                    ProcessBlockInterleaveNoCrop(jobStart_ + i);
                }
            }
        } else if (mode_ == BATCH_TO_SPACE_MODE_BLOCK_INTERLEAVE_CROP) {
            for (uint64_t i = 0ULL; i < jobCount_; ++i) {
                ProcessBlockInterleaveCrop(jobStart_ + i);
            }
        } else if (mode_ == BATCH_TO_SPACE_MODE_ROW_SCATTER) {
            if (noCrop_) {
                if (flagMergeGroup_ > 1U) {
                    ProcessRowScatterNoCropMerged();
                } else if (doubleBuf_) {
                    ProcessRowScatterNoCropPipe();
                } else {
                    for (uint64_t i = 0ULL; i < jobCount_; ++i) {
                        ProcessRowScatterNoCrop(jobStart_ + i);
                    }
                }
            } else {
                for (uint64_t i = 0ULL; i < jobCount_; ++i) {
                    ProcessRowScatter(jobStart_ + i);
                }
            }
        } else {
            for (uint64_t i = 0ULL; i < jobCount_; ++i) {
                ProcessChannelFallback(jobStart_ + i);
            }
        }

        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(eventMte2ToMte3_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
        if (doubleBuf_) {
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(eventMte2ToMte3b_);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventMte3ToMte2b_);
        }
    }

private:
    __aicore__ inline uint32_t MinU32(const uint32_t a, const uint32_t b) const
    {
        return (a < b) ? a : b;
    }

    __aicore__ inline uint32_t MaxU32(const uint32_t a, const uint32_t b) const
    {
        return (a > b) ? a : b;
    }

    // Strided copy helper (row_scatter / channel_fallback): one MTE2 contiguous
    // read into UB, then one MTE3 write with dstGap stride.
    __aicore__ inline void CopyPixelBlocks(const uint64_t srcElementOffset,
                                           const uint64_t dstElementOffset,
                                           const uint32_t blockCount,
                                           const uint32_t blockBytes,
                                           const uint32_t dstGapBytes)
    {
        LocalTensor<T> local = dataBuffer_.Get<T>();
        DataCopyExtParams inParams{
            static_cast<uint16_t>(blockCount), blockBytes, 0U, 0U, 0U};
        DataCopyPadExtParams<T> padParams{false, 0U, 0U, static_cast<T>(0)};
        DataCopyPad(local, xGm_[srcElementOffset], inParams, padParams);

        SetFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3_);
        WaitFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3_);

        DataCopyExtParams outParams{
            static_cast<uint16_t>(blockCount), blockBytes, 0U, dstGapBytes, 0U};
        DataCopyPad(yGm_[dstElementOffset], local, outParams);

        SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    }

    // ---- block_interleave (continuous write) ----
    __aicore__ inline void ProcessBlockInterleaveNoCrop(const uint64_t job)
    {
        const uint32_t tileId = static_cast<uint32_t>(job % tilesPerRow_);
        const uint64_t rowId = job / tilesPerRow_;
        const uint32_t bh = static_cast<uint32_t>(rowId % blockSize_);
        const uint64_t tmp = rowId / blockSize_;
        const uint32_t ih = static_cast<uint32_t>(tmp % inputHeight_);
        const uint32_t ob = static_cast<uint32_t>(tmp / inputHeight_);
        const uint32_t oh = ih * blockSize_ + bh;

        const uint32_t iwBegin = tileId * tilePixels_;
        const uint32_t iwEnd = MinU32(inputWidth_, iwBegin + tilePixels_);
        if (iwBegin >= iwEnd) {
            return;
        }
        const uint32_t n = iwEnd - iwBegin;

        LocalTensor<T> out = dataBuffer_.Get<T>();
        DataCopyPadExtParams<T> padParams{false, 0U, 0U, static_cast<T>(0)};
        const uint32_t interStride = (blockSize_ - 1U) * blocksPerPixel_;
        DataCopyExtParams inParams{
            static_cast<uint16_t>(n), pixelBytes_, 0U, interStride, 0U};
        for (uint32_t bw = 0U; bw < blockSize_; ++bw) {
            const uint32_t ib = (bh * blockSize_ + bw) * outputBatch_ + ob;
            const uint64_t srcPixel =
                (static_cast<uint64_t>(ib) * inputHeight_ + ih) * inputWidth_ + iwBegin;
            DataCopyPad(out[bw * channels_], xGm_[srcPixel * channels_], inParams, padParams);
        }

        SetFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3_);
        WaitFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3_);

        const uint64_t dstPixel =
            (static_cast<uint64_t>(ob) * outputHeight_ + oh) * outputWidth_ +
            static_cast<uint64_t>(iwBegin) * blockSize_;
        DataCopyExtParams outParams{1U, n * blockSize_ * pixelBytes_, 0U, 0U, 0U};
        DataCopyPad(yGm_[dstPixel * channels_], out, outParams);

        SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    }

    __aicore__ inline void ProcessBlockInterleaveCrop(const uint64_t job)
    {
        const uint32_t tileId = static_cast<uint32_t>(job % tilesPerRow_);
        const uint64_t rowId = job / tilesPerRow_;
        const uint32_t bh = static_cast<uint32_t>(rowId % blockSize_);
        const uint64_t tmp = rowId / blockSize_;
        const uint32_t ih = static_cast<uint32_t>(tmp % inputHeight_);
        const uint32_t ob = static_cast<uint32_t>(tmp / inputHeight_);

        const int32_t expandedOh = static_cast<int32_t>(ih * blockSize_ + bh);
        if (expandedOh < cropTop_ || expandedOh >= expandedBottom_) {
            return;
        }
        const uint32_t oh = static_cast<uint32_t>(expandedOh - cropTop_);

        const uint32_t iwBegin = tileId * tilePixels_;
        const uint32_t iwEnd = MinU32(inputWidth_, iwBegin + tilePixels_);
        if (iwBegin >= iwEnd) {
            return;
        }

        const int32_t tileExpBegin = static_cast<int32_t>(iwBegin * blockSize_);
        const int32_t tileExpEnd = static_cast<int32_t>(iwEnd * blockSize_);
        const int32_t owExpStart = (tileExpBegin > cropLeft_) ? tileExpBegin : cropLeft_;
        const int32_t owExpEnd = (tileExpEnd < expandedRight_) ? tileExpEnd : expandedRight_;
        if (owExpStart >= owExpEnd) {
            return;
        }

        const uint32_t sliceCount = static_cast<uint32_t>(owExpEnd - owExpStart);
        const uint32_t startExp = static_cast<uint32_t>(owExpStart);

        LocalTensor<T> out = dataBuffer_.Get<T>();
        DataCopyPadExtParams<T> padParams{false, 0U, 0U, static_cast<T>(0)};
        const uint32_t interStride = (blockSize_ - 1U) * blocksPerPixel_;
        for (uint32_t bw = 0U; bw < blockSize_; ++bw) {
            const uint32_t startPos =
                (bw + blockSize_ - (startExp % blockSize_)) % blockSize_;
            if (startPos >= sliceCount) {
                continue;
            }
            const uint32_t count = (sliceCount - startPos + blockSize_ - 1U) / blockSize_;
            const uint32_t iw0 = (startExp + startPos) / blockSize_;
            const uint32_t ib = (bh * blockSize_ + bw) * outputBatch_ + ob;
            const uint64_t srcPixel =
                (static_cast<uint64_t>(ib) * inputHeight_ + ih) * inputWidth_ + iw0;
            DataCopyExtParams inParams{
                static_cast<uint16_t>(count), pixelBytes_, 0U, interStride, 0U};
            DataCopyPad(out[startPos * channels_], xGm_[srcPixel * channels_],
                        inParams, padParams);
        }

        SetFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3_);
        WaitFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3_);

        const uint32_t owBegin = static_cast<uint32_t>(owExpStart - cropLeft_);
        const uint64_t dstPixel =
            (static_cast<uint64_t>(ob) * outputHeight_ + oh) * outputWidth_ + owBegin;
        DataCopyExtParams outParams{1U, sliceCount * pixelBytes_, 0U, 0U, 0U};
        DataCopyPad(yGm_[dstPixel * channels_], out, outParams);

        SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    }

    __aicore__ inline void BlockInterleaveNoCropLoad(const uint64_t job,
                                                     const LocalTensor<T> &buf)
    {
        const uint32_t tileId = static_cast<uint32_t>(job % tilesPerRow_);
        const uint64_t rowId = job / tilesPerRow_;
        const uint32_t bh = static_cast<uint32_t>(rowId % blockSize_);
        const uint64_t tmp = rowId / blockSize_;
        const uint32_t ih = static_cast<uint32_t>(tmp % inputHeight_);
        const uint32_t ob = static_cast<uint32_t>(tmp / inputHeight_);
        const uint32_t iwBegin = tileId * tilePixels_;
        const uint32_t iwEnd = MinU32(inputWidth_, iwBegin + tilePixels_);
        const uint32_t n = iwEnd - iwBegin;
        const uint32_t interStride = (blockSize_ - 1U) * blocksPerPixel_;
        DataCopyExtParams inParams{
            static_cast<uint16_t>(n), pixelBytes_, 0U, interStride, 0U};
        DataCopyPadExtParams<T> padParams{false, 0U, 0U, static_cast<T>(0)};
        for (uint32_t bw = 0U; bw < blockSize_; ++bw) {
            const uint32_t ib = (bh * blockSize_ + bw) * outputBatch_ + ob;
            const uint64_t srcPixel =
                (static_cast<uint64_t>(ib) * inputHeight_ + ih) * inputWidth_ + iwBegin;
            DataCopyPad(buf[bw * channels_], xGm_[srcPixel * channels_], inParams, padParams);
        }
    }

    __aicore__ inline void BlockInterleaveNoCropStore(const uint64_t job,
                                                      const LocalTensor<T> &buf)
    {
        const uint32_t tileId = static_cast<uint32_t>(job % tilesPerRow_);
        const uint64_t rowId = job / tilesPerRow_;
        const uint32_t bh = static_cast<uint32_t>(rowId % blockSize_);
        const uint64_t tmp = rowId / blockSize_;
        const uint32_t ih = static_cast<uint32_t>(tmp % inputHeight_);
        const uint32_t ob = static_cast<uint32_t>(tmp / inputHeight_);
        const uint32_t oh = ih * blockSize_ + bh;
        const uint32_t iwBegin = tileId * tilePixels_;
        const uint32_t iwEnd = MinU32(inputWidth_, iwBegin + tilePixels_);
        const uint32_t n = iwEnd - iwBegin;
        const uint64_t dstPixel =
            (static_cast<uint64_t>(ob) * outputHeight_ + oh) * outputWidth_ +
            static_cast<uint64_t>(iwBegin) * blockSize_;
        DataCopyExtParams outParams{1U, n * blockSize_ * pixelBytes_, 0U, 0U, 0U};
        DataCopyPad(yGm_[dstPixel * channels_], buf, outParams);
    }

    __aicore__ inline void ProcessBlockInterleaveNoCropPipe()
    {
        const uint64_t n = jobCount_;
        const uint64_t base = jobStart_;
        LocalTensor<T> buf0 = dataBuffer_.Get<T>();

        BlockInterleaveNoCropLoad(base, buf0);
        SetFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3_);

        for (uint64_t i = 0ULL; i < n; ++i) {
            const bool curIs0 = ((i & 1ULL) == 0ULL);
            const LocalTensor<T> curBuf = curIs0 ? buf0 : dbuf1_;
            const TEventID e2e3cur = curIs0 ? eventMte2ToMte3_ : eventMte2ToMte3b_;
            const TEventID e3e2cur = curIs0 ? eventMte3ToMte2_ : eventMte3ToMte2b_;

            if (i + 1ULL < n) {
                const bool nxtIs0 = (((i + 1ULL) & 1ULL) == 0ULL);
                const LocalTensor<T> nxtBuf = nxtIs0 ? buf0 : dbuf1_;
                const TEventID e2e3nxt = nxtIs0 ? eventMte2ToMte3_ : eventMte2ToMte3b_;
                const TEventID e3e2nxt = nxtIs0 ? eventMte3ToMte2_ : eventMte3ToMte2b_;
                if (i >= 1ULL) {
                    WaitFlag<HardEvent::MTE3_MTE2>(e3e2nxt);
                }
                BlockInterleaveNoCropLoad(base + i + 1ULL, nxtBuf);
                SetFlag<HardEvent::MTE2_MTE3>(e2e3nxt);
            }

            WaitFlag<HardEvent::MTE2_MTE3>(e2e3cur);
            BlockInterleaveNoCropStore(base + i, curBuf);
            SetFlag<HardEvent::MTE3_MTE2>(e3e2cur);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(
            (((n - 1ULL) & 1ULL) == 0ULL) ? eventMte3ToMte2_ : eventMte3ToMte2b_);
        WaitFlag<HardEvent::MTE3_MTE2>(
            (((n - 2ULL) & 1ULL) == 0ULL) ? eventMte3ToMte2_ : eventMte3ToMte2b_);
    }

    // V10 flag-merge: load a whole group of tiles into back-to-back UB slots,
    // raise ONE MTE2->MTE3 flag for the group, store the group, then ONE
    // MTE3->MTE2 flag. Events drop from 4/tile to 4/group for small tiles.
    __aicore__ inline void ProcessBlockInterleaveNoCropMerged()
    {
        LocalTensor<T> big = dataBuffer_.Get<T>();
        const uint32_t group = flagMergeGroup_;
        for (uint64_t i = 0ULL; i < jobCount_; i += group) {
            const uint32_t g =
                (jobCount_ - i < group) ? static_cast<uint32_t>(jobCount_ - i) : group;
            for (uint32_t k = 0U; k < g; ++k) {
                BlockInterleaveNoCropLoad(jobStart_ + i + k, big[k * tileStrideElems_]);
            }
            SetFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3_);
            WaitFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3_);
            for (uint32_t k = 0U; k < g; ++k) {
                BlockInterleaveNoCropStore(jobStart_ + i + k, big[k * tileStrideElems_]);
            }
            SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
            WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
        }
    }

    // ---- row_scatter (strided write) ----
    __aicore__ inline void ProcessRowScatterNoCrop(const uint64_t job)
    {
        const uint32_t tileId = static_cast<uint32_t>(job % tilesPerRow_);
        const uint64_t rowId = job / tilesPerRow_;
        const uint32_t ih = static_cast<uint32_t>(rowId % inputHeight_);
        const uint32_t ib = static_cast<uint32_t>(rowId / inputHeight_);

        const uint32_t blockId = ib / outputBatch_;
        const uint32_t ob = ib - blockId * outputBatch_;
        const uint32_t bh = blockId / blockSize_;
        const uint32_t bw = blockId - bh * blockSize_;

        const uint32_t iwBegin = tileId * tilePixels_;
        const uint32_t iwEnd = MinU32(inputWidth_, iwBegin + tilePixels_);
        const uint32_t count = iwEnd - iwBegin;

        const uint64_t srcPixel =
            (static_cast<uint64_t>(ib) * inputHeight_ + ih) * inputWidth_ + iwBegin;
        const uint64_t dstPixel =
            (static_cast<uint64_t>(ob) * outputHeight_ + ih * blockSize_ + bh) *
                outputWidth_ + iwBegin * blockSize_ + bw;
        CopyPixelBlocks(srcPixel * channels_, dstPixel * channels_, count,
                        pixelBytes_, rowDstGapBytes_);
    }

    __aicore__ inline void RowScatterNoCropLoad(const uint64_t job,
                                                const LocalTensor<T> &buf)
    {
        const uint32_t tileId = static_cast<uint32_t>(job % tilesPerRow_);
        const uint64_t rowId = job / tilesPerRow_;
        const uint32_t ih = static_cast<uint32_t>(rowId % inputHeight_);
        const uint32_t ib = static_cast<uint32_t>(rowId / inputHeight_);
        const uint32_t iwBegin = tileId * tilePixels_;
        const uint32_t iwEnd = MinU32(inputWidth_, iwBegin + tilePixels_);
        const uint32_t count = iwEnd - iwBegin;
        const uint64_t srcPixel =
            (static_cast<uint64_t>(ib) * inputHeight_ + ih) * inputWidth_ + iwBegin;
        DataCopyExtParams inParams{
            static_cast<uint16_t>(count), pixelBytes_, 0U, 0U, 0U};
        DataCopyPadExtParams<T> padParams{false, 0U, 0U, static_cast<T>(0)};
        DataCopyPad(buf, xGm_[srcPixel * channels_], inParams, padParams);
    }

    __aicore__ inline void RowScatterNoCropStore(const uint64_t job,
                                                 const LocalTensor<T> &buf)
    {
        const uint32_t tileId = static_cast<uint32_t>(job % tilesPerRow_);
        const uint64_t rowId = job / tilesPerRow_;
        const uint32_t ih = static_cast<uint32_t>(rowId % inputHeight_);
        const uint32_t ib = static_cast<uint32_t>(rowId / inputHeight_);
        const uint32_t blockId = ib / outputBatch_;
        const uint32_t ob = ib - blockId * outputBatch_;
        const uint32_t bh = blockId / blockSize_;
        const uint32_t bw = blockId - bh * blockSize_;
        const uint32_t iwBegin = tileId * tilePixels_;
        const uint32_t iwEnd = MinU32(inputWidth_, iwBegin + tilePixels_);
        const uint32_t count = iwEnd - iwBegin;
        const uint64_t dstPixel =
            (static_cast<uint64_t>(ob) * outputHeight_ + ih * blockSize_ + bh) *
                outputWidth_ + iwBegin * blockSize_ + bw;
        DataCopyExtParams outParams{
            static_cast<uint16_t>(count), pixelBytes_, 0U, rowDstGapBytes_, 0U};
        DataCopyPad(yGm_[dstPixel * channels_], buf, outParams);
    }

    __aicore__ inline void ProcessRowScatterNoCropPipe()
    {
        const uint64_t n = jobCount_;
        const uint64_t base = jobStart_;
        LocalTensor<T> buf0 = dataBuffer_.Get<T>();

        RowScatterNoCropLoad(base, buf0);
        SetFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3_);

        for (uint64_t i = 0ULL; i < n; ++i) {
            const bool curIs0 = ((i & 1ULL) == 0ULL);
            const LocalTensor<T> curBuf = curIs0 ? buf0 : dbuf1_;
            const TEventID e2e3cur = curIs0 ? eventMte2ToMte3_ : eventMte2ToMte3b_;
            const TEventID e3e2cur = curIs0 ? eventMte3ToMte2_ : eventMte3ToMte2b_;

            if (i + 1ULL < n) {
                const bool nxtIs0 = (((i + 1ULL) & 1ULL) == 0ULL);
                const LocalTensor<T> nxtBuf = nxtIs0 ? buf0 : dbuf1_;
                const TEventID e2e3nxt = nxtIs0 ? eventMte2ToMte3_ : eventMte2ToMte3b_;
                const TEventID e3e2nxt = nxtIs0 ? eventMte3ToMte2_ : eventMte3ToMte2b_;
                if (i >= 1ULL) {
                    WaitFlag<HardEvent::MTE3_MTE2>(e3e2nxt);
                }
                RowScatterNoCropLoad(base + i + 1ULL, nxtBuf);
                SetFlag<HardEvent::MTE2_MTE3>(e2e3nxt);
            }

            WaitFlag<HardEvent::MTE2_MTE3>(e2e3cur);
            RowScatterNoCropStore(base + i, curBuf);
            SetFlag<HardEvent::MTE3_MTE2>(e3e2cur);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(
            (((n - 1ULL) & 1ULL) == 0ULL) ? eventMte3ToMte2_ : eventMte3ToMte2b_);
        WaitFlag<HardEvent::MTE3_MTE2>(
            (((n - 2ULL) & 1ULL) == 0ULL) ? eventMte3ToMte2_ : eventMte3ToMte2b_);
    }

    // V10 flag-merge variant of the no-crop row_scatter path (see the block
    // interleave merged loop above): one event pair per group of tiles.
    __aicore__ inline void ProcessRowScatterNoCropMerged()
    {
        LocalTensor<T> big = dataBuffer_.Get<T>();
        const uint32_t group = flagMergeGroup_;
        for (uint64_t i = 0ULL; i < jobCount_; i += group) {
            const uint32_t g =
                (jobCount_ - i < group) ? static_cast<uint32_t>(jobCount_ - i) : group;
            for (uint32_t k = 0U; k < g; ++k) {
                RowScatterNoCropLoad(jobStart_ + i + k, big[k * tileStrideElems_]);
            }
            SetFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3_);
            WaitFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3_);
            for (uint32_t k = 0U; k < g; ++k) {
                RowScatterNoCropStore(jobStart_ + i + k, big[k * tileStrideElems_]);
            }
            SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
            WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
        }
    }

    __aicore__ inline void ProcessRowScatter(const uint64_t job)
    {
        const uint32_t tileId = static_cast<uint32_t>(job % tilesPerRow_);
        const uint64_t rowId = job / tilesPerRow_;
        const uint32_t ih = static_cast<uint32_t>(rowId % inputHeight_);
        const uint32_t ib = static_cast<uint32_t>(rowId / inputHeight_);

        const uint32_t blockId = ib / outputBatch_;
        const uint32_t ob = ib - blockId * outputBatch_;
        const uint32_t bh = blockId / blockSize_;
        const uint32_t bw = blockId - bh * blockSize_;

        const int32_t expandedOh = static_cast<int32_t>(ih * blockSize_ + bh);
        if (expandedOh < cropTop_ || expandedOh >= expandedBottom_) {
            return;
        }
        const uint32_t oh = static_cast<uint32_t>(expandedOh - cropTop_);

        uint32_t firstIw = 0U;
        if (cropLeft_ > static_cast<int32_t>(bw)) {
            firstIw = static_cast<uint32_t>(
                (cropLeft_ - static_cast<int32_t>(bw) +
                 static_cast<int32_t>(blockSize_) - 1) /
                static_cast<int32_t>(blockSize_));
        }
        uint32_t lastIw = 0U;
        if (expandedRight_ > static_cast<int32_t>(bw)) {
            lastIw = static_cast<uint32_t>(
                (expandedRight_ - static_cast<int32_t>(bw) +
                 static_cast<int32_t>(blockSize_) - 1) /
                static_cast<int32_t>(blockSize_));
            lastIw = MinU32(lastIw, inputWidth_);
        }

        const uint32_t tileBegin = tileId * tilePixels_;
        const uint32_t tileEnd = MinU32(inputWidth_, tileBegin + tilePixels_);
        const uint32_t iwBegin = MaxU32(tileBegin, firstIw);
        const uint32_t iwEnd = MinU32(tileEnd, lastIw);
        if (iwBegin >= iwEnd) {
            return;
        }

        const uint32_t count = iwEnd - iwBegin;
        const uint32_t owBegin = static_cast<uint32_t>(
            static_cast<int32_t>(iwBegin * blockSize_ + bw) - cropLeft_);

        const uint64_t srcPixel =
            (static_cast<uint64_t>(ib) * inputHeight_ + ih) * inputWidth_ + iwBegin;
        const uint64_t dstPixel =
            (static_cast<uint64_t>(ob) * outputHeight_ + oh) * outputWidth_ + owBegin;
        CopyPixelBlocks(srcPixel * channels_, dstPixel * channels_, count,
                        pixelBytes_, rowDstGapBytes_);
    }

    // ---- channel fallback (very large C or unsupported stride) ----
    __aicore__ inline void ProcessChannelFallback(const uint64_t job)
    {
        const uint32_t ow = static_cast<uint32_t>(job % outputWidth_);
        const uint64_t rowTmp = job / outputWidth_;
        const uint32_t oh = static_cast<uint32_t>(rowTmp % outputHeight_);
        const uint32_t ob = static_cast<uint32_t>(rowTmp / outputHeight_);

        const uint32_t expandedH = oh + static_cast<uint32_t>(cropTop_);
        const uint32_t expandedW = ow + static_cast<uint32_t>(cropLeft_);
        const uint32_t ih = expandedH / blockSize_;
        const uint32_t iw = expandedW / blockSize_;
        const uint32_t bh = expandedH - ih * blockSize_;
        const uint32_t bw = expandedW - iw * blockSize_;
        const uint32_t ib = (bh * blockSize_ + bw) * outputBatch_ + ob;

        const uint64_t srcPixel =
            (static_cast<uint64_t>(ib) * inputHeight_ + ih) * inputWidth_ + iw;
        const uint64_t dstPixel = job;

        for (uint32_t c0 = 0U; c0 < channels_; c0 += channelTileElems_) {
            const uint32_t curElems = MinU32(channelTileElems_, channels_ - c0);
            CopyPixelBlocks(srcPixel * channels_ + c0, dstPixel * channels_ + c0,
                            1U, curElems * static_cast<uint32_t>(sizeof(T)), 0U);
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECIN> dataBuffer_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    TEventID eventMte2ToMte3_;
    TEventID eventMte3ToMte2_;
    TEventID eventMte2ToMte3b_;
    TEventID eventMte3ToMte2b_;
    LocalTensor<T> dbuf1_;
    bool doubleBuf_ = false;
    uint32_t bufferBytes_ = 0U;
    uint32_t enableDoubleBuffer_ = 0U;
    uint32_t flagMergeGroup_ = 1U;
    uint32_t tileStrideElems_ = 0U;

    uint64_t totalJobs_ = 0ULL;
    uint64_t jobStart_ = 0ULL;
    uint64_t jobCount_ = 0ULL;
    uint32_t mode_ = 0U;
    uint32_t inputHeight_ = 0U;
    uint32_t inputWidth_ = 0U;
    uint32_t channels_ = 0U;
    uint32_t outputBatch_ = 0U;
    uint32_t outputHeight_ = 0U;
    uint32_t outputWidth_ = 0U;
    uint32_t blockSize_ = 0U;
    int32_t cropTop_ = 0;
    int32_t cropBottom_ = 0;
    int32_t cropLeft_ = 0;
    int32_t cropRight_ = 0;
    uint32_t tilePixels_ = 0U;
    uint32_t tilesPerRow_ = 0U;
    uint32_t channelTileElems_ = 0U;
    uint32_t pixelBytes_ = 0U;
    uint32_t blocksPerPixel_ = 0U;
    uint32_t rowDstGapBytes_ = 0U;
    int32_t expandedBottom_ = 0;
    int32_t expandedRight_ = 0;
    bool noCrop_ = false;
};

template <typename T>
__aicore__ inline void RunBatchToSpace(GM_ADDR x, GM_ADDR y,
                                       const BatchToSpaceTilingData &t)
{
    KernelBatchToSpace<T> op;
    op.Init(x, y, t.totalJobs, t.mode, t.inputHeight, t.inputWidth, t.channels,
            t.outputBatch, t.outputHeight, t.outputWidth, t.blockSize, t.cropTop,
            t.cropBottom, t.cropLeft, t.cropRight, t.tilePixels, t.tilesPerRow,
            t.channelTileElems, t.bufferBytes, t.enableDoubleBuffer,
            t.flagMergeGroup);
    op.Process();
}

}  // namespace

extern "C" __global__ __aicore__ void batch_to_space(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (TILING_KEY_IS(BATCH_TO_SPACE_TILING_KEY_FP16)) {
        RunBatchToSpace<half>(x, y, tilingData);
    } else if (TILING_KEY_IS(BATCH_TO_SPACE_TILING_KEY_FP32)) {
        RunBatchToSpace<float>(x, y, tilingData);
    }
}
