// Kernel implementation for BatchToSpace.
#include "kernel_operator.h"

#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

using namespace AscendC;

namespace {
constexpr uint32_t kDataBlockBytes = 32U;

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) {
    return (a < b) ? a : b;
}

__aicore__ inline uint32_t MaxU32(uint32_t a, uint32_t b) {
    return (a > b) ? a : b;
}
}  // namespace

template <uint32_t MODE_KIND>
struct CompactBsValue {
    static constexpr uint32_t value = 0U;
};

template <> struct CompactBsValue<BTS_MODE_F16_B2_STRIPE> { static constexpr uint32_t value = 2U; };
template <> struct CompactBsValue<BTS_MODE_F16_B3_STRIPE> { static constexpr uint32_t value = 3U; };
template <> struct CompactBsValue<BTS_MODE_F16_B4_STRIPE> { static constexpr uint32_t value = 4U; };
template <> struct CompactBsValue<BTS_MODE_F16_B5_STRIPE> { static constexpr uint32_t value = 5U; };
template <> struct CompactBsValue<BTS_MODE_F16_B6_STRIPE> { static constexpr uint32_t value = 6U; };
template <> struct CompactBsValue<BTS_MODE_F16_B7_STRIPE> { static constexpr uint32_t value = 7U; };
template <> struct CompactBsValue<BTS_MODE_F16_B8_STRIPE> { static constexpr uint32_t value = 8U; };
template <> struct CompactBsValue<BTS_MODE_F32_B2_STRIPE> { static constexpr uint32_t value = 2U; };
template <> struct CompactBsValue<BTS_MODE_F32_B3_STRIPE> { static constexpr uint32_t value = 3U; };
template <> struct CompactBsValue<BTS_MODE_F32_B4_STRIPE> { static constexpr uint32_t value = 4U; };
template <> struct CompactBsValue<BTS_MODE_F32_B5_STRIPE> { static constexpr uint32_t value = 5U; };
template <> struct CompactBsValue<BTS_MODE_F32_B6_STRIPE> { static constexpr uint32_t value = 6U; };
template <> struct CompactBsValue<BTS_MODE_F32_B7_STRIPE> { static constexpr uint32_t value = 7U; };
template <> struct CompactBsValue<BTS_MODE_F32_B8_STRIPE> { static constexpr uint32_t value = 8U; };

class KernelBatchToSpaceF16 {
public:
    __aicore__ inline KernelBatchToSpaceF16() {}

    template <uint32_t MODE_KIND>
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling) {
        batch_ = tiling.batch;
        height_ = tiling.height;
        width_ = tiling.width;
        depth_ = tiling.depth;
        outBatch_ = tiling.out_batch;
        outHeight_ = tiling.out_height;
        outWidth_ = tiling.out_width;
        totalElems_ = tiling.total_elems;
        totalGroups_ = tiling.total_groups;
        blockSize_ = tiling.block_size;
        cropTop_ = tiling.crop_top;
        cropBottom_ = tiling.crop_bottom;
        cropLeft_ = tiling.crop_left;
        cropRight_ = tiling.crop_right;
        mode_ = tiling.mode;
        strategy_ = tiling.strategy;
        depthBytes_ = tiling.depth_bytes;
        depthAlignedBytes_ = tiling.depth_aligned_bytes;
        tileElems_ = tiling.tile_elems;
        tileRows_ = tiling.tile_rows;
        tileIw_ = tiling.tile_iw;
        ubBytes_ = tiling.ub_bytes;
        usedCoreNum_ = tiling.used_core_num;
        bigCoreNum_ = tiling.big_core_num;
        bigCoreWorkNum_ = tiling.big_core_work_num;
        smallCoreWorkNum_ = tiling.small_core_work_num;
        if (tileElems_ == 0U) { tileElems_ = 1U; }
        if (tileRows_ == 0U) { tileRows_ = 1U; }
        if (tileIw_ == 0U) { tileIw_ = 1U; }
        if (ubBytes_ < kDataBlockBytes) { ubBytes_ = kDataBlockBytes; }

        compactPhaseNum_ = tiling.compact_phase_num;
        compactUniform_ = tiling.compact_uniform;
        compactPhaseGroupSize_ = tiling.compact_phase_group_size;
        compactRowAssembly_ = tiling.compact_row_assembly;
        compactRowTilesPerWidth_ = tiling.compact_row_tiles_per_width;
        compactRowGroupsPerBatch_ = tiling.compact_row_groups_per_batch;
        if (compactRowTilesPerWidth_ == 0U) {
            compactRowTilesPerWidth_ = (outWidth_ + tileIw_ - 1U) / tileIw_;
        }
        if (compactRowGroupsPerBatch_ == 0U) {
            const uint32_t rowGroupsPerBatch = (outHeight_ + tileRows_ - 1U) / tileRows_;
            compactRowGroupsPerBatch_ = rowGroupsPerBatch * compactRowTilesPerWidth_;
        }
        constexpr uint32_t kCompactBs = CompactBsValue<MODE_KIND>::value;
        if constexpr (kCompactBs != 0U) {
            if (compactRowAssembly_ == 0U) {
                for (uint32_t i = 0; i < kCompactBs; ++i) {
                hBegin_[i] = tiling.h_begin[i];
                hCount_[i] = tiling.h_count[i];
                wBegin_[i] = tiling.w_begin[i];
                wCount_[i] = tiling.w_count[i];
                tilesIw_[i] = tiling.tiles_iw[i];
                }
                if (compactUniform_ == 0U) {
                    constexpr uint32_t kPhaseLimit = kCompactBs * kCompactBs;
                    for (uint32_t i = 0; i <= kPhaseLimit; ++i) {
                        phasePrefix_[i] = tiling.phase_prefix[i];
                    }
                }
            }
        }

        xWStride_ = depth_;
        xHStride_ = width_ * depth_;
        xNStride_ = height_ * xHStride_;
        yWStride_ = depth_;
        yHStride_ = outWidth_ * depth_;
        yNStride_ = outHeight_ * yHStride_;

        xGm_.SetGlobalBuffer((__gm__ half *)x, batch_ * height_ * width_ * depth_);
        yGm_.SetGlobalBuffer((__gm__ half *)y, totalElems_);
        pipe_.InitBuffer(buf0_, ubBytes_);
        pipe_.InitBuffer(buf1_, ubBytes_);
        local0_ = buf0_.template Get<half>();
        local1_ = buf1_.template Get<half>();
        local_ = local0_;
    }

    template <uint32_t MODE_KIND>
    __aicore__ inline void Process() {
        if (totalGroups_ == 0U) {
            return;
        }
        if (MODE_KIND == BTS_MODE_F16_B1_LINEAR) {
            ProcessB1Linear();
        } else if (MODE_KIND == BTS_MODE_F16_B1_SLAB) {
            ProcessB1Slab();
        } else if (MODE_KIND == BTS_MODE_F16_B1_ROW2D) {
            ProcessB1Row2D();
        } else if (MODE_KIND == BTS_MODE_F16_B2_STRIPE) {
            ProcessCompactStripe<2U>();
        } else if (MODE_KIND == BTS_MODE_F16_B3_STRIPE) {
            ProcessCompactStripe<3U>();
        } else if (MODE_KIND == BTS_MODE_F16_B4_STRIPE) {
            ProcessCompactStripe<4U>();
        } else if (MODE_KIND == BTS_MODE_F16_B5_STRIPE) {
            ProcessCompactStripe<5U>();
        } else if (MODE_KIND == BTS_MODE_F16_B6_STRIPE) {
            ProcessCompactStripe<6U>();
        } else if (MODE_KIND == BTS_MODE_F16_B7_STRIPE) {
            ProcessCompactStripe<7U>();
        } else if (MODE_KIND == BTS_MODE_F16_B8_STRIPE) {
            ProcessCompactStripe<8U>();
        } else {
            ProcessGenericStripe();
        }
    }

private:
    __aicore__ inline DataCopyPadExtParams<half> NoPadParams() const {
        DataCopyPadExtParams<half> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = static_cast<half>(0);
        return padParams;
    }


    struct CopyJob2D {
        uint32_t srcOffset;
        uint32_t dstOffset;
        uint32_t blockLenBytes;
        uint32_t srcStrideBytes;
        uint32_t dstStrideBytes;
        uint16_t blockCount;
    };

    __aicore__ inline LocalTensor<half> GetLocalById(uint32_t bufId) {
        return (bufId == 0U) ? local0_ : local1_;
    }

    __aicore__ inline void SetFreeFlag(uint32_t bufId) const {
        if (bufId == 0U) { SetFlag<HardEvent::MTE3_MTE2>(0); }
        else { SetFlag<HardEvent::MTE3_MTE2>(1); }
    }

    __aicore__ inline void WaitFreeFlag(uint32_t bufId) const {
        if (bufId == 0U) { WaitFlag<HardEvent::MTE3_MTE2>(0); }
        else { WaitFlag<HardEvent::MTE3_MTE2>(1); }
    }

    __aicore__ inline void SetInReadyFlag(uint32_t bufId) const {
        if (bufId == 0U) { SetFlag<HardEvent::MTE2_MTE3>(0); }
        else { SetFlag<HardEvent::MTE2_MTE3>(1); }
    }

    __aicore__ inline void WaitInReadyFlag(uint32_t bufId) const {
        if (bufId == 0U) { WaitFlag<HardEvent::MTE2_MTE3>(0); }
        else { WaitFlag<HardEvent::MTE2_MTE3>(1); }
    }

    __aicore__ inline void InitDoubleCopyFlags() const {
        SetFreeFlag(0U);
        SetFreeFlag(1U);
    }

    __aicore__ inline void DrainDoubleCopyFlags() const {
        WaitFreeFlag(0U);
        WaitFreeFlag(1U);
    }

    __aicore__ inline void StartCopyInJob(const CopyJob2D &job, uint32_t bufId) {
        WaitFreeFlag(bufId);
        LocalTensor<half> local = GetLocalById(bufId);
        DataCopyExtParams inParams;
        inParams.blockCount = job.blockCount;
        inParams.blockLen = job.blockLenBytes;
        inParams.srcStride = job.srcStrideBytes;
        inParams.dstStride = 0;
        inParams.rsv = 0;
        DataCopyPad(local, xGm_[job.srcOffset], inParams, NoPadParams());
        SetInReadyFlag(bufId);
    }

    __aicore__ inline void FinishCopyOutJob(const CopyJob2D &job, uint32_t bufId) {
        WaitInReadyFlag(bufId);
        LocalTensor<half> local = GetLocalById(bufId);
        DataCopyExtParams outParams;
        outParams.blockCount = job.blockCount;
        outParams.blockLen = job.blockLenBytes;
        outParams.srcStride = 0;
        outParams.dstStride = job.dstStrideBytes;
        outParams.rsv = 0;
        DataCopyPad(yGm_[job.dstOffset], local, outParams);
        SetFreeFlag(bufId);
    }

    __aicore__ inline void RunSingleJob(const CopyJob2D &job) {
        if (job.blockCount == 0U || job.blockLenBytes == 0U) { return; }
        LocalTensor<half> local = local0_;
        DataCopyExtParams inParams;
        inParams.blockCount = job.blockCount;
        inParams.blockLen = job.blockLenBytes;
        inParams.srcStride = job.srcStrideBytes;
        inParams.dstStride = 0;
        inParams.rsv = 0;
        DataCopyPad(local, xGm_[job.srcOffset], inParams, NoPadParams());
        SetFlag<HardEvent::MTE2_MTE3>(0);
        WaitFlag<HardEvent::MTE2_MTE3>(0);
        DataCopyExtParams outParams;
        outParams.blockCount = job.blockCount;
        outParams.blockLen = job.blockLenBytes;
        outParams.srcStride = 0;
        outParams.dstStride = job.dstStrideBytes;
        outParams.rsv = 0;
        DataCopyPad(yGm_[job.dstOffset], local, outParams);
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }

    __aicore__ inline CopyJob2D Make1DJob(uint32_t srcOffset, uint32_t dstOffset, uint32_t elemCount) const {
        CopyJob2D job;
        job.srcOffset = srcOffset;
        job.dstOffset = dstOffset;
        job.blockCount = 1;
        job.blockLenBytes = elemCount * 2U;
        job.srcStrideBytes = 0;
        job.dstStrideBytes = 0;
        return job;
    }


    __aicore__ inline void BeginCopyLoop() const {
        SetFlag<HardEvent::MTE3_MTE2>(0);
    }

    __aicore__ inline void EndCopyLoop() const {
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }

    __aicore__ inline void GetWorkRange(uint32_t &start, uint32_t &end) const {
        const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
        if (coreIdx >= usedCoreNum_) {
            start = 0U;
            end = 0U;
            return;
        }
        if (coreIdx < bigCoreNum_) {
            start = coreIdx * bigCoreWorkNum_;
            end = start + bigCoreWorkNum_;
        } else {
            start = bigCoreNum_ * bigCoreWorkNum_ +
                    (coreIdx - bigCoreNum_) * smallCoreWorkNum_;
            end = start + smallCoreWorkNum_;
        }
    }

    __aicore__ inline void GetGroupRange(uint32_t &start, uint32_t &end) const {
        GetWorkRange(start, end);
        if (end > totalGroups_) { end = totalGroups_; }
    }

    __aicore__ inline void Copy1D(uint32_t srcOffset, uint32_t dstOffset, uint32_t elemCount) {
        if (elemCount == 0U) { return; }
        const uint32_t bytes = elemCount * 2U;
        WaitFlag<HardEvent::MTE3_MTE2>(0);
        DataCopyExtParams inParams;
        inParams.blockCount = 1;
        inParams.blockLen = bytes;
        inParams.srcStride = 0;
        inParams.dstStride = 0;
        inParams.rsv = 0;
        DataCopyPad(local_, xGm_[srcOffset], inParams, NoPadParams());
        SetFlag<HardEvent::MTE2_MTE3>(0);
        WaitFlag<HardEvent::MTE2_MTE3>(0);
        DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = bytes;
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        outParams.rsv = 0;
        DataCopyPad(yGm_[dstOffset], local_, outParams);
        SetFlag<HardEvent::MTE3_MTE2>(0);
    }

    __aicore__ inline void Copy2D(uint32_t srcOffset,
                                  uint32_t dstOffset,
                                  uint16_t blockCount,
                                  uint32_t blockLenBytes,
                                  uint32_t srcStrideBytes,
                                  uint32_t dstStrideBytes) {
        if (blockCount == 0U || blockLenBytes == 0U) { return; }
        WaitFlag<HardEvent::MTE3_MTE2>(0);
        DataCopyExtParams inParams;
        inParams.blockCount = blockCount;
        inParams.blockLen = blockLenBytes;
        inParams.srcStride = srcStrideBytes;
        inParams.dstStride = 0;
        inParams.rsv = 0;
        DataCopyPad(local_, xGm_[srcOffset], inParams, NoPadParams());
        SetFlag<HardEvent::MTE2_MTE3>(0);
        WaitFlag<HardEvent::MTE2_MTE3>(0);
        DataCopyExtParams outParams;
        outParams.blockCount = blockCount;
        outParams.blockLen = blockLenBytes;
        outParams.srcStride = 0;
        outParams.dstStride = dstStrideBytes;
        outParams.rsv = 0;
        DataCopyPad(yGm_[dstOffset], local_, outParams);
        SetFlag<HardEvent::MTE3_MTE2>(0);
    }

    __aicore__ inline void ProcessB1Linear() {
        uint32_t offset = 0U;
        uint32_t end = 0U;
        GetWorkRange(offset, end);
        if (end > totalElems_) { end = totalElems_; }
        if (offset >= end) { return; }

        InitDoubleCopyFlags();
        uint32_t len = tileElems_;
        if (offset + len > end) { len = end - offset; }
        CopyJob2D curJob = Make1DJob(offset, offset, len);
        offset += len;
        if (offset >= end) {
            RunSingleJob(curJob);
            return;
        }
        uint32_t curBuf = 0U;
        StartCopyInJob(curJob, curBuf);

        while (offset < end) {
            len = tileElems_;
            if (offset + len > end) { len = end - offset; }
            const uint32_t nextBuf = 1U - curBuf;
            CopyJob2D nextJob = Make1DJob(offset, offset, len);
            StartCopyInJob(nextJob, nextBuf);
            FinishCopyOutJob(curJob, curBuf);
            curJob = nextJob;
            curBuf = nextBuf;
            offset += len;
        }
        FinishCopyOutJob(curJob, curBuf);
        DrainDoubleCopyFlags();
    }

    __aicore__ inline bool MakeB1SlabJob(uint32_t groupIdx, CopyJob2D &job) const {
        const uint32_t slabElems = outHeight_ * outWidth_ * depth_;
        const uint32_t tilesPerSlab = (slabElems + tileElems_ - 1U) / tileElems_;
        if (tilesPerSlab == 0U) { return false; }
        const uint32_t ob = groupIdx / tilesPerSlab;
        const uint32_t tileIdx = groupIdx - ob * tilesPerSlab;
        if (ob >= outBatch_) { return false; }
        const uint32_t localOffset = tileIdx * tileElems_;
        if (localOffset >= slabElems) { return false; }
        uint32_t len = tileElems_;
        if (localOffset + len > slabElems) { len = slabElems - localOffset; }
        const uint32_t srcBase = ob * xNStride_ + cropTop_ * xHStride_;
        const uint32_t dstBase = ob * yNStride_;
        job = Make1DJob(srcBase + localOffset, dstBase + localOffset, len);
        return len != 0U;
    }

    __aicore__ inline void ProcessB1Slab() {
        uint32_t start = 0U, end = 0U;
        GetGroupRange(start, end);
        if (start >= end) { return; }

        InitDoubleCopyFlags();
        CopyJob2D curJob;
        uint32_t groupIdx = start;
        while (groupIdx < end && !MakeB1SlabJob(groupIdx, curJob)) { ++groupIdx; }
        if (groupIdx >= end) {
            DrainDoubleCopyFlags();
            return;
        }
        uint32_t curBuf = 0U;
        StartCopyInJob(curJob, curBuf);
        ++groupIdx;

        while (groupIdx < end) {
            CopyJob2D nextJob;
            bool hasNext = false;
            while (groupIdx < end) {
                if (MakeB1SlabJob(groupIdx, nextJob)) {
                    hasNext = true;
                    ++groupIdx;
                    break;
                }
                ++groupIdx;
            }
            if (!hasNext) { break; }
            const uint32_t nextBuf = 1U - curBuf;
            StartCopyInJob(nextJob, nextBuf);
            FinishCopyOutJob(curJob, curBuf);
            curJob = nextJob;
            curBuf = nextBuf;
        }
        FinishCopyOutJob(curJob, curBuf);
        DrainDoubleCopyFlags();
    }

    __aicore__ inline bool MakeB1Row2DJob(uint32_t groupIdx, CopyJob2D &job) const {
        if (tileIw_ >= outWidth_) {
            const uint32_t rowGroupsPerBatch = (outHeight_ + tileRows_ - 1U) / tileRows_;
            if (rowGroupsPerBatch == 0U) { return false; }
            const uint32_t rowBytes = outWidth_ * depthBytes_;
            const uint32_t srcRowGapBytes = (cropLeft_ + cropRight_) * depthBytes_;
            const uint32_t ob = groupIdx / rowGroupsPerBatch;
            const uint32_t rowTile = groupIdx - ob * rowGroupsPerBatch;
            if (ob >= outBatch_) { return false; }
            const uint32_t oh0 = rowTile * tileRows_;
            if (oh0 >= outHeight_) { return false; }
            uint32_t rows = tileRows_;
            if (oh0 + rows > outHeight_) { rows = outHeight_ - oh0; }
            job.srcOffset = ob * xNStride_ + (oh0 + cropTop_) * xHStride_ + cropLeft_ * xWStride_;
            job.dstOffset = ob * yNStride_ + oh0 * yHStride_;
            job.blockCount = static_cast<uint16_t>(rows);
            job.blockLenBytes = rowBytes;
            job.srcStrideBytes = srcRowGapBytes;
            job.dstStrideBytes = 0U;
            return job.blockCount != 0U && job.blockLenBytes != 0U;
        }

        const uint32_t tilesPerRow = (outWidth_ + tileIw_ - 1U) / tileIw_;
        const uint32_t rowGroupsPerBatch = (outHeight_ + tileRows_ - 1U) / tileRows_;
        const uint32_t groupsPerBatch = rowGroupsPerBatch * tilesPerRow;
        if (groupsPerBatch == 0U) { return false; }
        const uint32_t ob = groupIdx / groupsPerBatch;
        const uint32_t localGroup = groupIdx - ob * groupsPerBatch;
        const uint32_t rowTile = localGroup / tilesPerRow;
        const uint32_t tileIdx = localGroup - rowTile * tilesPerRow;
        if (ob >= outBatch_) { return false; }
        const uint32_t oh0 = rowTile * tileRows_;
        if (oh0 >= outHeight_) { return false; }
        uint32_t rows = tileRows_;
        if (oh0 + rows > outHeight_) { rows = outHeight_ - oh0; }
        const uint32_t ow0 = tileIdx * tileIw_;
        if (ow0 >= outWidth_) { return false; }
        uint32_t curOw = tileIw_;
        if (ow0 + curOw > outWidth_) { curOw = outWidth_ - ow0; }
        job.srcOffset = ob * xNStride_ + (oh0 + cropTop_) * xHStride_ + (ow0 + cropLeft_) * xWStride_;
        job.dstOffset = ob * yNStride_ + oh0 * yHStride_ + ow0 * yWStride_;
        job.blockCount = static_cast<uint16_t>(rows);
        job.blockLenBytes = curOw * depthBytes_;
        job.srcStrideBytes = (width_ - curOw) * depthBytes_;
        job.dstStrideBytes = (outWidth_ - curOw) * depthBytes_;
        return job.blockCount != 0U && job.blockLenBytes != 0U;
    }

    __aicore__ inline void ProcessB1Row2D() {
        uint32_t start = 0U, end = 0U;
        GetGroupRange(start, end);
        if (start >= end) { return; }

        InitDoubleCopyFlags();
        CopyJob2D curJob;
        uint32_t groupIdx = start;
        while (groupIdx < end && !MakeB1Row2DJob(groupIdx, curJob)) { ++groupIdx; }
        if (groupIdx >= end) {
            DrainDoubleCopyFlags();
            return;
        }
        uint32_t curBuf = 0U;
        StartCopyInJob(curJob, curBuf);
        ++groupIdx;

        while (groupIdx < end) {
            CopyJob2D nextJob;
            bool hasNext = false;
            while (groupIdx < end) {
                if (MakeB1Row2DJob(groupIdx, nextJob)) {
                    hasNext = true;
                    ++groupIdx;
                    break;
                }
                ++groupIdx;
            }
            if (!hasNext) { break; }
            const uint32_t nextBuf = 1U - curBuf;
            StartCopyInJob(nextJob, nextBuf);
            FinishCopyOutJob(curJob, curBuf);
            curJob = nextJob;
            curBuf = nextBuf;
        }
        FinishCopyOutJob(curJob, curBuf);
        DrainDoubleCopyFlags();
    }

    template <uint32_t BS>
    __aicore__ inline bool MakeCompactJob(uint32_t groupIdx, CopyJob2D &job) const {
        if (groupIdx >= totalGroups_) { return false; }
        const uint32_t phaseLimit = BS * BS;
        uint32_t phase = 0U;
        uint32_t local = 0U;
        if (compactUniform_ != 0U && compactPhaseGroupSize_ != 0U) {
            phase = groupIdx / compactPhaseGroupSize_;
            if (phase >= phaseLimit) { return false; }
            local = groupIdx - phase * compactPhaseGroupSize_;
        } else {
            uint32_t lo = 0U;
            uint32_t hi = phaseLimit;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) >> 1;
                if (phasePrefix_[mid + 1U] <= groupIdx) { lo = mid + 1U; }
                else { hi = mid; }
            }
            phase = lo;
            if (phase >= phaseLimit) { return false; }
            local = groupIdx - phasePrefix_[phase];
        }
        uint32_t bh = 0U;
        uint32_t bw = 0U;
        if (BS == 2U) {
            bh = phase >> 1;
            bw = phase & 1U;
        } else if (BS == 4U) {
            bh = phase >> 2;
            bw = phase & 3U;
        } else if (BS == 8U) {
            bh = phase >> 3;
            bw = phase & 7U;
        } else {
            bh = phase / BS;
            bw = phase - bh * BS;
        }
        const uint32_t tilesIw = tilesIw_[bw];
        const uint32_t ihCount = hCount_[bh];
        if (tilesIw == 0U || ihCount == 0U) { return false; }
        const uint32_t tileIdx = local % tilesIw;
        uint32_t tmp = local / tilesIw;
        const uint32_t ihRel = tmp % ihCount;
        const uint32_t ob = tmp / ihCount;
        if (ob >= outBatch_) { return false; }
        const uint32_t iwRel = tileIdx * tileIw_;
        if (iwRel >= wCount_[bw]) { return false; }
        uint32_t count = tileIw_;
        if (iwRel + count > wCount_[bw]) { count = wCount_[bw] - iwRel; }
        const uint32_t ih = hBegin_[bh] + ihRel;
        const uint32_t iw0 = wBegin_[bw] + iwRel;
        const uint32_t ib = phase * outBatch_ + ob;
        uint32_t oh = 0U;
        uint32_t ow0 = 0U;
        if (BS == 2U) {
            oh = (ih << 1) + bh - cropTop_;
            ow0 = (iw0 << 1) + bw - cropLeft_;
        } else if (BS == 4U) {
            oh = (ih << 2) + bh - cropTop_;
            ow0 = (iw0 << 2) + bw - cropLeft_;
        } else if (BS == 8U) {
            oh = (ih << 3) + bh - cropTop_;
            ow0 = (iw0 << 3) + bw - cropLeft_;
        } else {
            oh = ih * BS + bh - cropTop_;
            ow0 = iw0 * BS + bw - cropLeft_;
        }
        job.srcOffset = ib * xNStride_ + ih * xHStride_ + iw0 * xWStride_;
        job.dstOffset = ob * yNStride_ + oh * yHStride_ + ow0 * yWStride_;
        job.blockCount = static_cast<uint16_t>(count);
        job.blockLenBytes = depthBytes_;
        job.srcStrideBytes = 0U;
        job.dstStrideBytes = (BS - 1U) * depthBytes_;
        return job.blockCount != 0U;
    }


    template <uint32_t BS>
    __aicore__ inline bool MakeCompactPhaseTileJob(uint32_t phase,
                                                    uint32_t ob,
                                                    uint32_t ihRel,
                                                    uint32_t tileIdx,
                                                    CopyJob2D &job) const {
        const uint32_t phaseLimit = BS * BS;
        if (phase >= phaseLimit || ob >= outBatch_) { return false; }
        uint32_t bh = 0U;
        uint32_t bw = 0U;
        if constexpr (BS == 2U) {
            bh = phase >> 1;
            bw = phase & 1U;
        } else if constexpr (BS == 4U) {
            bh = phase >> 2;
            bw = phase & 3U;
        } else if constexpr (BS == 8U) {
            bh = phase >> 3;
            bw = phase & 7U;
        } else {
            bh = phase / BS;
            bw = phase - bh * BS;
        }
        const uint32_t tilesIw = tilesIw_[bw];
        const uint32_t ihCount = hCount_[bh];
        if (tilesIw == 0U || ihCount == 0U || tileIdx >= tilesIw || ihRel >= ihCount) {
            return false;
        }
        const uint32_t iwRel = tileIdx * tileIw_;
        if (iwRel >= wCount_[bw]) { return false; }
        uint32_t count = tileIw_;
        if (iwRel + count > wCount_[bw]) { count = wCount_[bw] - iwRel; }
        if (count == 0U) { return false; }

        const uint32_t ih = hBegin_[bh] + ihRel;
        const uint32_t iw0 = wBegin_[bw] + iwRel;
        const uint32_t ib = phase * outBatch_ + ob;
        uint32_t oh = 0U;
        uint32_t ow0 = 0U;
        if constexpr (BS == 2U) {
            oh = (ih << 1) + bh - cropTop_;
            ow0 = (iw0 << 1) + bw - cropLeft_;
        } else if constexpr (BS == 4U) {
            oh = (ih << 2) + bh - cropTop_;
            ow0 = (iw0 << 2) + bw - cropLeft_;
        } else if constexpr (BS == 8U) {
            oh = (ih << 3) + bh - cropTop_;
            ow0 = (iw0 << 3) + bw - cropLeft_;
        } else {
            oh = ih * BS + bh - cropTop_;
            ow0 = iw0 * BS + bw - cropLeft_;
        }
        job.srcOffset = ib * xNStride_ + ih * xHStride_ + iw0 * xWStride_;
        job.dstOffset = ob * yNStride_ + oh * yHStride_ + ow0 * yWStride_;
        job.blockCount = static_cast<uint16_t>(count);
        job.blockLenBytes = depthBytes_;
        job.srcStrideBytes = 0U;
        job.dstStrideBytes = (BS - 1U) * depthBytes_;
        return true;
    }

    template <uint32_t BS>
    __aicore__ inline void ProcessCompactTinyPhase() {
        uint32_t start = 0U, end = 0U;
        GetGroupRange(start, end);
        if (start >= end) { return; }
        const uint32_t phaseLimit = BS * BS;
        if (end > phaseLimit) { end = phaseLimit; }
        for (uint32_t phase = start; phase < end; ++phase) {
            uint32_t bh = 0U;
            uint32_t bw = 0U;
            if constexpr (BS == 2U) {
                bh = phase >> 1;
                bw = phase & 1U;
            } else if constexpr (BS == 4U) {
                bh = phase >> 2;
                bw = phase & 3U;
            } else if constexpr (BS == 8U) {
                bh = phase >> 3;
                bw = phase & 7U;
            } else {
                bh = phase / BS;
                bw = phase - bh * BS;
            }
            const uint32_t tilesIw = tilesIw_[bw];
            const uint32_t ihCount = hCount_[bh];
            if (tilesIw == 0U || ihCount == 0U) { continue; }
            for (uint32_t ob = 0U; ob < outBatch_; ++ob) {
                for (uint32_t ihRel = 0U; ihRel < ihCount; ++ihRel) {
                    for (uint32_t tileIdx = 0U; tileIdx < tilesIw; ++tileIdx) {
                        CopyJob2D job;
                        if (MakeCompactPhaseTileJob<BS>(phase, ob, ihRel, tileIdx, job)) {
                            RunSingleJob(job);
                        }
                    }
                }
            }
        }
    }


    template <uint32_t BS>
    __aicore__ inline bool MakeB2NoCropFastJob(uint32_t groupIdx, CopyJob2D &job) const {
        if constexpr (BS != 2U) {
            return false;
        } else {
            const uint32_t tilesPerRow = (width_ + tileIw_ - 1U) / tileIw_;
            if (tilesPerRow == 0U) { return false; }
            const uint32_t phaseGroupSize = outBatch_ * height_ * tilesPerRow;
            if (phaseGroupSize == 0U) { return false; }

            const uint32_t phase = groupIdx / phaseGroupSize;
            if (phase >= 4U) { return false; }
            const uint32_t local = groupIdx - phase * phaseGroupSize;
            const uint32_t tileIdx = local % tilesPerRow;
            uint32_t tmp = local / tilesPerRow;
            const uint32_t ih = tmp % height_;
            const uint32_t ob = tmp / height_;
            if (ob >= outBatch_) { return false; }

            const uint32_t iw0 = tileIdx * tileIw_;
            if (iw0 >= width_) { return false; }
            uint32_t count = tileIw_;
            if (iw0 + count > width_) { count = width_ - iw0; }
            if (count == 0U) { return false; }

            const uint32_t bh = phase >> 1;
            const uint32_t bw = phase & 1U;
            const uint32_t ib = phase * outBatch_ + ob;
            const uint32_t oh = (ih << 1) + bh;
            const uint32_t ow0 = (iw0 << 1) + bw;

            job.srcOffset = ib * xNStride_ + ih * xHStride_ + iw0 * xWStride_;
            job.dstOffset = ob * yNStride_ + oh * yHStride_ + ow0 * yWStride_;
            job.blockCount = static_cast<uint16_t>(count);
            job.blockLenBytes = depthBytes_;
            job.srcStrideBytes = 0U;
            job.dstStrideBytes = depthBytes_;
            return true;
        }
    }

    template <uint32_t BS>
    __aicore__ inline void ProcessB2NoCropFast() {
        if constexpr (BS != 2U) {
            return;
        } else {
            uint32_t start = 0U, end = 0U;
            GetGroupRange(start, end);
            if (start >= end) { return; }

            InitDoubleCopyFlags();
            CopyJob2D curJob;
            uint32_t groupIdx = start;
            while (groupIdx < end && !MakeB2NoCropFastJob<BS>(groupIdx, curJob)) { ++groupIdx; }
            if (groupIdx >= end) {
                DrainDoubleCopyFlags();
                return;
            }
            uint32_t curBuf = 0U;
            StartCopyInJob(curJob, curBuf);
            ++groupIdx;

            while (groupIdx < end) {
                CopyJob2D nextJob;
                bool hasNext = false;
                while (groupIdx < end) {
                    if (MakeB2NoCropFastJob<BS>(groupIdx, nextJob)) {
                        hasNext = true;
                        ++groupIdx;
                        break;
                    }
                    ++groupIdx;
                }
                if (!hasNext) { break; }
                const uint32_t nextBuf = 1U - curBuf;
                StartCopyInJob(nextJob, nextBuf);
                FinishCopyOutJob(curJob, curBuf);
                curJob = nextJob;
                curBuf = nextBuf;
            }
            FinishCopyOutJob(curJob, curBuf);
            DrainDoubleCopyFlags();
        }
    }

    template <uint32_t BS>
    __aicore__ inline void ProcessCompactSmall() {
        uint32_t start = 0U, end = 0U;
        GetGroupRange(start, end);
        if (start >= end) { return; }
        for (uint32_t groupIdx = start; groupIdx < end; ++groupIdx) {
            CopyJob2D job;
            if (MakeCompactJob<BS>(groupIdx, job)) {
                RunSingleJob(job);
            }
        }
    }

    template <uint32_t BS>
    __aicore__ inline void ProcessCompactStripe() {
        if (compactRowAssembly_ != 0U || strategy_ == BTS_STRATEGY_COMPACT_ROWASM) {
            ProcessCompactRowAssemble<BS>();
            return;
        }
        if (strategy_ == BTS_STRATEGY_B2_NOCROP_FAST) {
            ProcessB2NoCropFast<BS>();
            return;
        }
        if (strategy_ == BTS_STRATEGY_COMPACT_TINY_PHASE) {
            ProcessCompactTinyPhase<BS>();
            return;
        }
        if (strategy_ == BTS_STRATEGY_COMPACT_SMALL) {
            ProcessCompactSmall<BS>();
            return;
        }
        uint32_t start = 0U, end = 0U;
        GetGroupRange(start, end);
        if (start >= end) { return; }

        InitDoubleCopyFlags();
        CopyJob2D curJob;
        uint32_t groupIdx = start;
        while (groupIdx < end && !MakeCompactJob<BS>(groupIdx, curJob)) { ++groupIdx; }
        if (groupIdx >= end) {
            DrainDoubleCopyFlags();
            return;
        }
        uint32_t curBuf = 0U;
        StartCopyInJob(curJob, curBuf);
        ++groupIdx;

        while (groupIdx < end) {
            CopyJob2D nextJob;
            bool hasNext = false;
            while (groupIdx < end) {
                if (MakeCompactJob<BS>(groupIdx, nextJob)) {
                    hasNext = true;
                    ++groupIdx;
                    break;
                }
                ++groupIdx;
            }
            if (!hasNext) { break; }
            const uint32_t nextBuf = 1U - curBuf;
            StartCopyInJob(nextJob, nextBuf);
            FinishCopyOutJob(curJob, curBuf);
            curJob = nextJob;
            curBuf = nextBuf;
        }
        FinishCopyOutJob(curJob, curBuf);
        DrainDoubleCopyFlags();
    }


    template <uint32_t BS>
    __aicore__ inline uint32_t ModBs(uint32_t value) const {
        if constexpr (BS == 2U) { return value & 1U; }
        else if constexpr (BS == 4U) { return value & 3U; }
        else if constexpr (BS == 8U) { return value & 7U; }
        else { return value % BS; }
    }

    template <uint32_t BS>
    __aicore__ inline uint32_t DivBs(uint32_t value) const {
        if constexpr (BS == 2U) { return value >> 1; }
        else if constexpr (BS == 4U) { return value >> 2; }
        else if constexpr (BS == 8U) { return value >> 3; }
        else { return value / BS; }
    }

    template <uint32_t BS>
    __aicore__ inline uint32_t MulBs(uint32_t value) const {
        if constexpr (BS == 2U) { return value << 1; }
        else if constexpr (BS == 4U) { return value << 2; }
        else if constexpr (BS == 8U) { return value << 3; }
        else { return value * BS; }
    }

    struct CompactRowJob {
        uint32_t ob;
        uint32_t oh0;
        uint32_t ohCount;
        uint32_t ow0;
        uint32_t owCount;
    };

    __aicore__ inline bool MakeCompactRowJob(uint32_t groupIdx, CompactRowJob &job) const {
        const uint32_t tilesPerRow = compactRowTilesPerWidth_;
        const uint32_t groupsPerBatch = compactRowGroupsPerBatch_;
        if (tilesPerRow == 0U || groupsPerBatch == 0U) { return false; }

        const uint32_t ob = groupIdx / groupsPerBatch;
        const uint32_t localGroup = groupIdx - ob * groupsPerBatch;
        uint32_t rowTile;
        uint32_t tileIdx;
        if (tilesPerRow == 1U) {
            rowTile = localGroup;
            tileIdx = 0U;
        } else {
            rowTile = localGroup / tilesPerRow;
            tileIdx = localGroup - rowTile * tilesPerRow;
        }

        const uint32_t oh0 = rowTile * tileRows_;
        if (ob >= outBatch_ || oh0 >= outHeight_) { return false; }
        uint32_t ohCount = tileRows_;
        if (oh0 + ohCount > outHeight_) { ohCount = outHeight_ - oh0; }
        const uint32_t ow0 = tileIdx * tileIw_;
        if (ow0 >= outWidth_) { return false; }
        uint32_t owCount = tileIw_;
        if (ow0 + owCount > outWidth_) { owCount = outWidth_ - ow0; }
        if (ohCount == 0U || owCount == 0U) { return false; }
        job.ob = ob;
        job.oh0 = oh0;
        job.ohCount = ohCount;
        job.ow0 = ow0;
        job.owCount = owCount;
        return true;
    }

    template <uint32_t BS>
    __aicore__ inline void StartCompactRowAssembleJob(const CompactRowJob &job, uint32_t bufId) {
        WaitFreeFlag(bufId);
        LocalTensor<half> local = GetLocalById(bufId);
        const uint32_t rowStride = job.owCount * depth_;
        const uint32_t ubDstStride = ((BS - 1U) * depthBytes_) >> 5;

        // v23: common full-width/no-W-crop row-assembly fast path.
        // In this case every phase contributes exactly width_ contiguous depth blocks
        // starting from iw=0, so the inner loop avoids per-phase delta/count/div
        // computations.  The vertical (bh, ih) decode is advanced incrementally
        // across bundled rows to avoid repeated div/mod, especially for BS=3/5/6/7.
        if (job.ow0 == 0U && job.owCount == outWidth_ && cropLeft_ == 0U && cropRight_ == 0U) {
            uint32_t fh = job.oh0 + cropTop_;
            uint32_t bh = ModBs<BS>(fh);
            uint32_t ih = DivBs<BS>(fh);
            for (uint32_t r = 0U; r < job.ohCount; ++r) {
                const uint32_t rowLocalBase = r * rowStride;
                const uint32_t phaseBase = bh * BS;
                const uint32_t srcRowBase = ih * xHStride_;
                for (uint32_t bw = 0U; bw < BS; ++bw) {
                    const uint32_t ib = (phaseBase + bw) * outBatch_ + job.ob;
                    const uint32_t localOffset = rowLocalBase + bw * depth_;
                    const uint32_t srcOffset = ib * xNStride_ + srcRowBase;

                    DataCopyExtParams inParams;
                    inParams.blockCount = static_cast<uint16_t>(width_);
                    inParams.blockLen = depthBytes_;
                    inParams.srcStride = 0U;
                    inParams.dstStride = ubDstStride;
                    inParams.rsv = 0;
                    DataCopyPad(local[localOffset], xGm_[srcOffset], inParams, NoPadParams());
                }
                ++bh;
                if (bh == BS) {
                    bh = 0U;
                    ++ih;
                }
            }
            SetInReadyFlag(bufId);
            return;
        }

        const uint32_t fw0Base = job.ow0 + cropLeft_;
        const uint32_t fwEnd = fw0Base + job.owCount;
        const uint32_t fw0Mod = ModBs<BS>(fw0Base);
        uint32_t fh = job.oh0 + cropTop_;
        uint32_t bh = ModBs<BS>(fh);
        uint32_t ih = DivBs<BS>(fh);

        for (uint32_t r = 0U; r < job.ohCount; ++r) {
            const uint32_t rowLocalBase = r * rowStride;

            for (uint32_t bw = 0U; bw < BS; ++bw) {
                uint32_t delta = (bw >= fw0Mod) ? (bw - fw0Mod) : (bw + BS - fw0Mod);
                const uint32_t firstFw = fw0Base + delta;
                if (firstFw >= fwEnd) { continue; }
                const uint32_t count = (fwEnd - 1U - firstFw) / BS + 1U;
                const uint32_t iw0 = DivBs<BS>(firstFw);
                const uint32_t phase = bh * BS + bw;
                const uint32_t ib = phase * outBatch_ + job.ob;
                const uint32_t localOw = firstFw - cropLeft_ - job.ow0;
                const uint32_t localOffset = rowLocalBase + localOw * depth_;
                const uint32_t srcOffset = ib * xNStride_ + ih * xHStride_ + iw0 * xWStride_;

                DataCopyExtParams inParams;
                inParams.blockCount = static_cast<uint16_t>(count);
                inParams.blockLen = depthBytes_;
                inParams.srcStride = 0U;
                inParams.dstStride = ubDstStride;
                inParams.rsv = 0;
                DataCopyPad(local[localOffset], xGm_[srcOffset], inParams, NoPadParams());
            }

            ++bh;
            if (bh == BS) {
                bh = 0U;
                ++ih;
            }
        }
        SetInReadyFlag(bufId);
    }

    __aicore__ inline void FinishCompactRowAssembleJob(const CompactRowJob &job, uint32_t bufId) {
        WaitInReadyFlag(bufId);
        LocalTensor<half> local = GetLocalById(bufId);
        const uint32_t dstOffset = job.ob * yNStride_ + job.oh0 * yHStride_ + job.ow0 * yWStride_;
        if (job.ow0 == 0U && job.owCount == outWidth_) {
            DataCopy(yGm_[dstOffset], local, job.ohCount * job.owCount * depth_);
        } else {
            DataCopyExtParams outParams;
            outParams.blockCount = static_cast<uint16_t>(job.ohCount);
            outParams.blockLen = job.owCount * depthBytes_;
            outParams.srcStride = 0U;
            outParams.dstStride = (outWidth_ - job.owCount) * depthBytes_;
            outParams.rsv = 0;
            DataCopyPad(yGm_[dstOffset], local, outParams);
        }
        SetFreeFlag(bufId);
    }

    template <uint32_t BS>
    __aicore__ inline void ProcessCompactRowAssemble() {
        uint32_t start = 0U, end = 0U;
        GetGroupRange(start, end);
        if (start >= end) { return; }

        InitDoubleCopyFlags();
        CompactRowJob curJob;
        uint32_t groupIdx = start;
        while (groupIdx < end && !MakeCompactRowJob(groupIdx, curJob)) { ++groupIdx; }
        if (groupIdx >= end) {
            DrainDoubleCopyFlags();
            return;
        }

        uint32_t curBuf = 0U;
        StartCompactRowAssembleJob<BS>(curJob, curBuf);
        ++groupIdx;

        while (groupIdx < end) {
            CompactRowJob nextJob;
            bool hasNext = false;
            while (groupIdx < end) {
                if (MakeCompactRowJob(groupIdx, nextJob)) {
                    hasNext = true;
                    ++groupIdx;
                    break;
                }
                ++groupIdx;
            }
            if (!hasNext) { break; }
            const uint32_t nextBuf = 1U - curBuf;
            StartCompactRowAssembleJob<BS>(nextJob, nextBuf);
            FinishCompactRowAssembleJob(curJob, curBuf);
            curJob = nextJob;
            curBuf = nextBuf;
        }

        FinishCompactRowAssembleJob(curJob, curBuf);
        DrainDoubleCopyFlags();
    }

    __aicore__ inline int32_t CeilDivS32(int32_t a, int32_t b) const {
        if (a >= 0) { return (a + b - 1) / b; }
        return a / b;
    }

    __aicore__ inline uint32_t ClampBeginKernel(int32_t crop, int32_t phase, int32_t block, uint32_t dim) const {
        int32_t v = CeilDivS32(crop - phase, block);
        if (v < 0) { v = 0; }
        const uint32_t uv = static_cast<uint32_t>(v);
        return MinU32(uv, dim);
    }

    __aicore__ inline uint32_t ClampEndKernel(int32_t crop, int32_t outDim, int32_t phase, int32_t block, uint32_t dim) const {
        int32_t v = CeilDivS32(crop + outDim - phase, block);
        if (v < 0) { v = 0; }
        const uint32_t uv = static_cast<uint32_t>(v);
        return MinU32(uv, dim);
    }

    __aicore__ inline bool MakeGenericStripeJob(uint32_t groupIdx, CopyJob2D &job) const {
        const uint32_t tilesPerInputRow = (width_ + tileIw_ - 1U) / tileIw_;
        if (tilesPerInputRow == 0U) { return false; }
        const uint32_t bsz = blockSize_;
        const uint32_t tileIdx = groupIdx % tilesPerInputRow;
        uint32_t tmp = groupIdx / tilesPerInputRow;
        const uint32_t ih = tmp % height_;
        tmp /= height_;
        const uint32_t phaseArea = bsz * bsz;
        const uint32_t phase = tmp % phaseArea;
        tmp /= phaseArea;
        const uint32_t ob = tmp;
        if (ob >= outBatch_) { return false; }
        const uint32_t bh = phase / bsz;
        const uint32_t bw = phase - bh * bsz;
        const uint32_t ihBegin = ClampBeginKernel(static_cast<int32_t>(cropTop_), static_cast<int32_t>(bh),
                                                  static_cast<int32_t>(bsz), height_);
        const uint32_t ihEnd = ClampEndKernel(static_cast<int32_t>(cropTop_), static_cast<int32_t>(outHeight_),
                                              static_cast<int32_t>(bh), static_cast<int32_t>(bsz), height_);
        if (ih < ihBegin || ih >= ihEnd) { return false; }
        const uint32_t iwTile0 = tileIdx * tileIw_;
        const uint32_t iwTile1 = MinU32(width_, iwTile0 + tileIw_);
        const uint32_t iwBegin = ClampBeginKernel(static_cast<int32_t>(cropLeft_), static_cast<int32_t>(bw),
                                                  static_cast<int32_t>(bsz), width_);
        const uint32_t iwEnd = ClampEndKernel(static_cast<int32_t>(cropLeft_), static_cast<int32_t>(outWidth_),
                                              static_cast<int32_t>(bw), static_cast<int32_t>(bsz), width_);
        const uint32_t validIw0 = MaxU32(iwTile0, iwBegin);
        const uint32_t validIw1 = MinU32(iwTile1, iwEnd);
        if (validIw0 >= validIw1) { return false; }
        const uint32_t count = validIw1 - validIw0;
        const uint32_t ib = phase * outBatch_ + ob;
        const uint32_t oh = ih * bsz + bh - cropTop_;
        const uint32_t ow0 = validIw0 * bsz + bw - cropLeft_;
        job.srcOffset = ib * xNStride_ + ih * xHStride_ + validIw0 * xWStride_;
        job.dstOffset = ob * yNStride_ + oh * yHStride_ + ow0 * yWStride_;
        job.blockCount = static_cast<uint16_t>(count);
        job.blockLenBytes = depthBytes_;
        job.srcStrideBytes = 0U;
        job.dstStrideBytes = (bsz - 1U) * depthBytes_;
        return job.blockCount != 0U && job.blockLenBytes != 0U;
    }

    __aicore__ inline void ProcessGenericStripe() {
        uint32_t start = 0U, end = 0U;
        GetGroupRange(start, end);
        if (start >= end) { return; }

        InitDoubleCopyFlags();
        CopyJob2D curJob;
        uint32_t groupIdx = start;
        while (groupIdx < end && !MakeGenericStripeJob(groupIdx, curJob)) { ++groupIdx; }
        if (groupIdx >= end) {
            DrainDoubleCopyFlags();
            return;
        }
        uint32_t curBuf = 0U;
        StartCopyInJob(curJob, curBuf);
        ++groupIdx;

        while (groupIdx < end) {
            CopyJob2D nextJob;
            bool hasNext = false;
            while (groupIdx < end) {
                if (MakeGenericStripeJob(groupIdx, nextJob)) {
                    hasNext = true;
                    ++groupIdx;
                    break;
                }
                ++groupIdx;
            }
            if (!hasNext) { break; }
            const uint32_t nextBuf = 1U - curBuf;
            StartCopyInJob(nextJob, nextBuf);
            FinishCopyOutJob(curJob, curBuf);
            curJob = nextJob;
            curBuf = nextBuf;
        }
        FinishCopyOutJob(curJob, curBuf);
        DrainDoubleCopyFlags();
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECIN> buf0_;
    TBuf<TPosition::VECIN> buf1_;
    LocalTensor<half> local0_;
    LocalTensor<half> local1_;
    LocalTensor<half> local_;
    GlobalTensor<half> xGm_;
    GlobalTensor<half> yGm_;

    uint32_t batch_;
    uint32_t height_;
    uint32_t width_;
    uint32_t depth_;
    uint32_t outBatch_;
    uint32_t outHeight_;
    uint32_t outWidth_;
    uint32_t totalElems_;
    uint32_t totalGroups_;
    uint32_t xWStride_;
    uint32_t xHStride_;
    uint32_t xNStride_;
    uint32_t yWStride_;
    uint32_t yHStride_;
    uint32_t yNStride_;

    uint32_t hBegin_[BTS_MAX_COMPACT_BS];
    uint32_t hCount_[BTS_MAX_COMPACT_BS];
    uint32_t wBegin_[BTS_MAX_COMPACT_BS];
    uint32_t wCount_[BTS_MAX_COMPACT_BS];
    uint32_t tilesIw_[BTS_MAX_COMPACT_BS];
    uint32_t phasePrefix_[BTS_MAX_COMPACT_PHASE + 1U];
    uint32_t compactPhaseNum_;
    uint32_t compactUniform_;
    uint32_t compactPhaseGroupSize_;
    uint32_t compactRowAssembly_;
    uint32_t compactRowTilesPerWidth_;
    uint32_t compactRowGroupsPerBatch_;

    uint32_t blockSize_;
    uint32_t cropTop_;
    uint32_t cropBottom_;
    uint32_t cropLeft_;
    uint32_t cropRight_;
    uint32_t mode_;
    uint32_t strategy_;
    uint32_t depthBytes_;
    uint32_t depthAlignedBytes_;
    uint32_t tileElems_;
    uint32_t tileRows_;
    uint32_t tileIw_;
    uint32_t ubBytes_;
    uint32_t usedCoreNum_;
    uint32_t bigCoreNum_;
    uint32_t bigCoreWorkNum_;
    uint32_t smallCoreWorkNum_;
};

class KernelBatchToSpaceF32 {
public:
    __aicore__ inline KernelBatchToSpaceF32() {}

    template <uint32_t MODE_KIND>
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling) {
        batch_ = tiling.batch;
        height_ = tiling.height;
        width_ = tiling.width;
        depth_ = tiling.depth;
        outBatch_ = tiling.out_batch;
        outHeight_ = tiling.out_height;
        outWidth_ = tiling.out_width;
        totalElems_ = tiling.total_elems;
        totalGroups_ = tiling.total_groups;
        blockSize_ = tiling.block_size;
        cropTop_ = tiling.crop_top;
        cropBottom_ = tiling.crop_bottom;
        cropLeft_ = tiling.crop_left;
        cropRight_ = tiling.crop_right;
        mode_ = tiling.mode;
        strategy_ = tiling.strategy;
        depthBytes_ = tiling.depth_bytes;
        depthAlignedBytes_ = tiling.depth_aligned_bytes;
        tileElems_ = tiling.tile_elems;
        tileRows_ = tiling.tile_rows;
        tileIw_ = tiling.tile_iw;
        ubBytes_ = tiling.ub_bytes;
        usedCoreNum_ = tiling.used_core_num;
        bigCoreNum_ = tiling.big_core_num;
        bigCoreWorkNum_ = tiling.big_core_work_num;
        smallCoreWorkNum_ = tiling.small_core_work_num;
        if (tileElems_ == 0U) { tileElems_ = 1U; }
        if (tileRows_ == 0U) { tileRows_ = 1U; }
        if (tileIw_ == 0U) { tileIw_ = 1U; }
        if (ubBytes_ < kDataBlockBytes) { ubBytes_ = kDataBlockBytes; }

        compactPhaseNum_ = tiling.compact_phase_num;
        compactUniform_ = tiling.compact_uniform;
        compactPhaseGroupSize_ = tiling.compact_phase_group_size;
        compactRowAssembly_ = tiling.compact_row_assembly;
        compactRowTilesPerWidth_ = tiling.compact_row_tiles_per_width;
        compactRowGroupsPerBatch_ = tiling.compact_row_groups_per_batch;
        if (compactRowTilesPerWidth_ == 0U) {
            compactRowTilesPerWidth_ = (outWidth_ + tileIw_ - 1U) / tileIw_;
        }
        if (compactRowGroupsPerBatch_ == 0U) {
            const uint32_t rowGroupsPerBatch = (outHeight_ + tileRows_ - 1U) / tileRows_;
            compactRowGroupsPerBatch_ = rowGroupsPerBatch * compactRowTilesPerWidth_;
        }
        constexpr uint32_t kCompactBs = CompactBsValue<MODE_KIND>::value;
        if constexpr (kCompactBs != 0U) {
            if (compactRowAssembly_ == 0U) {
                for (uint32_t i = 0; i < kCompactBs; ++i) {
                hBegin_[i] = tiling.h_begin[i];
                hCount_[i] = tiling.h_count[i];
                wBegin_[i] = tiling.w_begin[i];
                wCount_[i] = tiling.w_count[i];
                tilesIw_[i] = tiling.tiles_iw[i];
                }
                if (compactUniform_ == 0U) {
                    constexpr uint32_t kPhaseLimit = kCompactBs * kCompactBs;
                    for (uint32_t i = 0; i <= kPhaseLimit; ++i) {
                        phasePrefix_[i] = tiling.phase_prefix[i];
                    }
                }
            }
        }

        xWStride_ = depth_;
        xHStride_ = width_ * depth_;
        xNStride_ = height_ * xHStride_;
        yWStride_ = depth_;
        yHStride_ = outWidth_ * depth_;
        yNStride_ = outHeight_ * yHStride_;

        xGm_.SetGlobalBuffer((__gm__ float *)x, batch_ * height_ * width_ * depth_);
        yGm_.SetGlobalBuffer((__gm__ float *)y, totalElems_);
        pipe_.InitBuffer(buf0_, ubBytes_);
        pipe_.InitBuffer(buf1_, ubBytes_);
        local0_ = buf0_.template Get<float>();
        local1_ = buf1_.template Get<float>();
        local_ = local0_;
    }

    template <uint32_t MODE_KIND>
    __aicore__ inline void Process() {
        if (totalGroups_ == 0U) {
            return;
        }
        if (MODE_KIND == BTS_MODE_F32_B1_LINEAR) {
            ProcessB1Linear();
        } else if (MODE_KIND == BTS_MODE_F32_B1_SLAB) {
            ProcessB1Slab();
        } else if (MODE_KIND == BTS_MODE_F32_B1_ROW2D) {
            ProcessB1Row2D();
        } else if (MODE_KIND == BTS_MODE_F32_B2_STRIPE) {
            ProcessCompactStripe<2U>();
        } else if (MODE_KIND == BTS_MODE_F32_B3_STRIPE) {
            ProcessCompactStripe<3U>();
        } else if (MODE_KIND == BTS_MODE_F32_B4_STRIPE) {
            ProcessCompactStripe<4U>();
        } else if (MODE_KIND == BTS_MODE_F32_B5_STRIPE) {
            ProcessCompactStripe<5U>();
        } else if (MODE_KIND == BTS_MODE_F32_B6_STRIPE) {
            ProcessCompactStripe<6U>();
        } else if (MODE_KIND == BTS_MODE_F32_B7_STRIPE) {
            ProcessCompactStripe<7U>();
        } else if (MODE_KIND == BTS_MODE_F32_B8_STRIPE) {
            ProcessCompactStripe<8U>();
        } else {
            ProcessGenericStripe();
        }
    }

private:
    __aicore__ inline DataCopyPadExtParams<float> NoPadParams() const {
        DataCopyPadExtParams<float> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = static_cast<float>(0);
        return padParams;
    }

    struct CopyJob2D {
        uint32_t srcOffset;
        uint32_t dstOffset;
        uint32_t blockLenBytes;
        uint32_t srcStrideBytes;
        uint32_t dstStrideBytes;
        uint16_t blockCount;
    };

    __aicore__ inline LocalTensor<float> GetLocalById(uint32_t bufId) {
        return (bufId == 0U) ? local0_ : local1_;
    }

    __aicore__ inline void SetFreeFlag(uint32_t bufId) const {
        if (bufId == 0U) { SetFlag<HardEvent::MTE3_MTE2>(0); }
        else { SetFlag<HardEvent::MTE3_MTE2>(1); }
    }

    __aicore__ inline void WaitFreeFlag(uint32_t bufId) const {
        if (bufId == 0U) { WaitFlag<HardEvent::MTE3_MTE2>(0); }
        else { WaitFlag<HardEvent::MTE3_MTE2>(1); }
    }

    __aicore__ inline void SetInReadyFlag(uint32_t bufId) const {
        if (bufId == 0U) { SetFlag<HardEvent::MTE2_MTE3>(0); }
        else { SetFlag<HardEvent::MTE2_MTE3>(1); }
    }

    __aicore__ inline void WaitInReadyFlag(uint32_t bufId) const {
        if (bufId == 0U) { WaitFlag<HardEvent::MTE2_MTE3>(0); }
        else { WaitFlag<HardEvent::MTE2_MTE3>(1); }
    }

    __aicore__ inline void InitDoubleCopyFlags() const {
        SetFreeFlag(0U);
        SetFreeFlag(1U);
    }

    __aicore__ inline void DrainDoubleCopyFlags() const {
        WaitFreeFlag(0U);
        WaitFreeFlag(1U);
    }

    __aicore__ inline void StartCopyInJob(const CopyJob2D &job, uint32_t bufId) {
        WaitFreeFlag(bufId);
        LocalTensor<float> local = GetLocalById(bufId);
        DataCopyExtParams inParams;
        inParams.blockCount = job.blockCount;
        inParams.blockLen = job.blockLenBytes;
        inParams.srcStride = job.srcStrideBytes;
        inParams.dstStride = 0;
        inParams.rsv = 0;
        DataCopyPad(local, xGm_[job.srcOffset], inParams, NoPadParams());
        SetInReadyFlag(bufId);
    }

    __aicore__ inline void FinishCopyOutJob(const CopyJob2D &job, uint32_t bufId) {
        WaitInReadyFlag(bufId);
        LocalTensor<float> local = GetLocalById(bufId);
        DataCopyExtParams outParams;
        outParams.blockCount = job.blockCount;
        outParams.blockLen = job.blockLenBytes;
        outParams.srcStride = 0;
        outParams.dstStride = job.dstStrideBytes;
        outParams.rsv = 0;
        DataCopyPad(yGm_[job.dstOffset], local, outParams);
        SetFreeFlag(bufId);
    }

    __aicore__ inline void RunSingleJob(const CopyJob2D &job) {
        if (job.blockCount == 0U || job.blockLenBytes == 0U) { return; }
        LocalTensor<float> local = local0_;
        DataCopyExtParams inParams;
        inParams.blockCount = job.blockCount;
        inParams.blockLen = job.blockLenBytes;
        inParams.srcStride = job.srcStrideBytes;
        inParams.dstStride = 0;
        inParams.rsv = 0;
        DataCopyPad(local, xGm_[job.srcOffset], inParams, NoPadParams());
        SetFlag<HardEvent::MTE2_MTE3>(0);
        WaitFlag<HardEvent::MTE2_MTE3>(0);
        DataCopyExtParams outParams;
        outParams.blockCount = job.blockCount;
        outParams.blockLen = job.blockLenBytes;
        outParams.srcStride = 0;
        outParams.dstStride = job.dstStrideBytes;
        outParams.rsv = 0;
        DataCopyPad(yGm_[job.dstOffset], local, outParams);
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }

    __aicore__ inline CopyJob2D Make1DJob(uint32_t srcOffset, uint32_t dstOffset, uint32_t elemCount) const {
        CopyJob2D job;
        job.srcOffset = srcOffset;
        job.dstOffset = dstOffset;
        job.blockCount = 1;
        job.blockLenBytes = elemCount * 4U;
        job.srcStrideBytes = 0;
        job.dstStrideBytes = 0;
        return job;
    }

    __aicore__ inline void BeginCopyLoop() const {
        SetFlag<HardEvent::MTE3_MTE2>(0);
    }

    __aicore__ inline void EndCopyLoop() const {
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }

    __aicore__ inline void GetWorkRange(uint32_t &start, uint32_t &end) const {
        const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
        if (coreIdx >= usedCoreNum_) {
            start = 0U;
            end = 0U;
            return;
        }
        if (coreIdx < bigCoreNum_) {
            start = coreIdx * bigCoreWorkNum_;
            end = start + bigCoreWorkNum_;
        } else {
            start = bigCoreNum_ * bigCoreWorkNum_ +
                    (coreIdx - bigCoreNum_) * smallCoreWorkNum_;
            end = start + smallCoreWorkNum_;
        }
    }

    __aicore__ inline void GetGroupRange(uint32_t &start, uint32_t &end) const {
        GetWorkRange(start, end);
        if (end > totalGroups_) { end = totalGroups_; }
    }

    __aicore__ inline void Copy1D(uint32_t srcOffset, uint32_t dstOffset, uint32_t elemCount) {
        if (elemCount == 0U) { return; }
        const uint32_t bytes = elemCount * 4U;
        WaitFlag<HardEvent::MTE3_MTE2>(0);
        DataCopyExtParams inParams;
        inParams.blockCount = 1;
        inParams.blockLen = bytes;
        inParams.srcStride = 0;
        inParams.dstStride = 0;
        inParams.rsv = 0;
        DataCopyPad(local_, xGm_[srcOffset], inParams, NoPadParams());
        SetFlag<HardEvent::MTE2_MTE3>(0);
        WaitFlag<HardEvent::MTE2_MTE3>(0);
        DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = bytes;
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        outParams.rsv = 0;
        DataCopyPad(yGm_[dstOffset], local_, outParams);
        SetFlag<HardEvent::MTE3_MTE2>(0);
    }

    __aicore__ inline void Copy2D(uint32_t srcOffset,
                                  uint32_t dstOffset,
                                  uint16_t blockCount,
                                  uint32_t blockLenBytes,
                                  uint32_t srcStrideBytes,
                                  uint32_t dstStrideBytes) {
        if (blockCount == 0U || blockLenBytes == 0U) { return; }
        WaitFlag<HardEvent::MTE3_MTE2>(0);
        DataCopyExtParams inParams;
        inParams.blockCount = blockCount;
        inParams.blockLen = blockLenBytes;
        inParams.srcStride = srcStrideBytes;
        inParams.dstStride = 0;
        inParams.rsv = 0;
        DataCopyPad(local_, xGm_[srcOffset], inParams, NoPadParams());
        SetFlag<HardEvent::MTE2_MTE3>(0);
        WaitFlag<HardEvent::MTE2_MTE3>(0);
        DataCopyExtParams outParams;
        outParams.blockCount = blockCount;
        outParams.blockLen = blockLenBytes;
        outParams.srcStride = 0;
        outParams.dstStride = dstStrideBytes;
        outParams.rsv = 0;
        DataCopyPad(yGm_[dstOffset], local_, outParams);
        SetFlag<HardEvent::MTE3_MTE2>(0);
    }

    __aicore__ inline void ProcessB1Linear() {
        uint32_t offset = 0U;
        uint32_t end = 0U;
        GetWorkRange(offset, end);
        if (end > totalElems_) { end = totalElems_; }
        if (offset >= end) { return; }

        InitDoubleCopyFlags();
        uint32_t len = tileElems_;
        if (offset + len > end) { len = end - offset; }
        CopyJob2D curJob = Make1DJob(offset, offset, len);
        offset += len;
        if (offset >= end) {
            RunSingleJob(curJob);
            return;
        }
        uint32_t curBuf = 0U;
        StartCopyInJob(curJob, curBuf);

        while (offset < end) {
            len = tileElems_;
            if (offset + len > end) { len = end - offset; }
            const uint32_t nextBuf = 1U - curBuf;
            CopyJob2D nextJob = Make1DJob(offset, offset, len);
            StartCopyInJob(nextJob, nextBuf);
            FinishCopyOutJob(curJob, curBuf);
            curJob = nextJob;
            curBuf = nextBuf;
            offset += len;
        }
        FinishCopyOutJob(curJob, curBuf);
        DrainDoubleCopyFlags();
    }

    __aicore__ inline bool MakeB1SlabJob(uint32_t groupIdx, CopyJob2D &job) const {
        const uint32_t slabElems = outHeight_ * outWidth_ * depth_;
        const uint32_t tilesPerSlab = (slabElems + tileElems_ - 1U) / tileElems_;
        if (tilesPerSlab == 0U) { return false; }
        const uint32_t ob = groupIdx / tilesPerSlab;
        const uint32_t tileIdx = groupIdx - ob * tilesPerSlab;
        if (ob >= outBatch_) { return false; }
        const uint32_t localOffset = tileIdx * tileElems_;
        if (localOffset >= slabElems) { return false; }
        uint32_t len = tileElems_;
        if (localOffset + len > slabElems) { len = slabElems - localOffset; }
        const uint32_t srcBase = ob * xNStride_ + cropTop_ * xHStride_;
        const uint32_t dstBase = ob * yNStride_;
        job = Make1DJob(srcBase + localOffset, dstBase + localOffset, len);
        return len != 0U;
    }

    __aicore__ inline void ProcessB1Slab() {
        uint32_t start = 0U, end = 0U;
        GetGroupRange(start, end);
        if (start >= end) { return; }

        InitDoubleCopyFlags();
        CopyJob2D curJob;
        uint32_t groupIdx = start;
        while (groupIdx < end && !MakeB1SlabJob(groupIdx, curJob)) { ++groupIdx; }
        if (groupIdx >= end) {
            DrainDoubleCopyFlags();
            return;
        }
        uint32_t curBuf = 0U;
        StartCopyInJob(curJob, curBuf);
        ++groupIdx;

        while (groupIdx < end) {
            CopyJob2D nextJob;
            bool hasNext = false;
            while (groupIdx < end) {
                if (MakeB1SlabJob(groupIdx, nextJob)) {
                    hasNext = true;
                    ++groupIdx;
                    break;
                }
                ++groupIdx;
            }
            if (!hasNext) { break; }
            const uint32_t nextBuf = 1U - curBuf;
            StartCopyInJob(nextJob, nextBuf);
            FinishCopyOutJob(curJob, curBuf);
            curJob = nextJob;
            curBuf = nextBuf;
        }
        FinishCopyOutJob(curJob, curBuf);
        DrainDoubleCopyFlags();
    }

    __aicore__ inline bool MakeB1Row2DJob(uint32_t groupIdx, CopyJob2D &job) const {
        if (tileIw_ >= outWidth_) {
            const uint32_t rowGroupsPerBatch = (outHeight_ + tileRows_ - 1U) / tileRows_;
            if (rowGroupsPerBatch == 0U) { return false; }
            const uint32_t rowBytes = outWidth_ * depthBytes_;
            const uint32_t srcRowGapBytes = (cropLeft_ + cropRight_) * depthBytes_;
            const uint32_t ob = groupIdx / rowGroupsPerBatch;
            const uint32_t rowTile = groupIdx - ob * rowGroupsPerBatch;
            if (ob >= outBatch_) { return false; }
            const uint32_t oh0 = rowTile * tileRows_;
            if (oh0 >= outHeight_) { return false; }
            uint32_t rows = tileRows_;
            if (oh0 + rows > outHeight_) { rows = outHeight_ - oh0; }
            job.srcOffset = ob * xNStride_ + (oh0 + cropTop_) * xHStride_ + cropLeft_ * xWStride_;
            job.dstOffset = ob * yNStride_ + oh0 * yHStride_;
            job.blockCount = static_cast<uint16_t>(rows);
            job.blockLenBytes = rowBytes;
            job.srcStrideBytes = srcRowGapBytes;
            job.dstStrideBytes = 0U;
            return job.blockCount != 0U && job.blockLenBytes != 0U;
        }

        const uint32_t tilesPerRow = (outWidth_ + tileIw_ - 1U) / tileIw_;
        const uint32_t rowGroupsPerBatch = (outHeight_ + tileRows_ - 1U) / tileRows_;
        const uint32_t groupsPerBatch = rowGroupsPerBatch * tilesPerRow;
        if (groupsPerBatch == 0U) { return false; }
        const uint32_t ob = groupIdx / groupsPerBatch;
        const uint32_t localGroup = groupIdx - ob * groupsPerBatch;
        const uint32_t rowTile = localGroup / tilesPerRow;
        const uint32_t tileIdx = localGroup - rowTile * tilesPerRow;
        if (ob >= outBatch_) { return false; }
        const uint32_t oh0 = rowTile * tileRows_;
        if (oh0 >= outHeight_) { return false; }
        uint32_t rows = tileRows_;
        if (oh0 + rows > outHeight_) { rows = outHeight_ - oh0; }
        const uint32_t ow0 = tileIdx * tileIw_;
        if (ow0 >= outWidth_) { return false; }
        uint32_t curOw = tileIw_;
        if (ow0 + curOw > outWidth_) { curOw = outWidth_ - ow0; }
        job.srcOffset = ob * xNStride_ + (oh0 + cropTop_) * xHStride_ + (ow0 + cropLeft_) * xWStride_;
        job.dstOffset = ob * yNStride_ + oh0 * yHStride_ + ow0 * yWStride_;
        job.blockCount = static_cast<uint16_t>(rows);
        job.blockLenBytes = curOw * depthBytes_;
        job.srcStrideBytes = (width_ - curOw) * depthBytes_;
        job.dstStrideBytes = (outWidth_ - curOw) * depthBytes_;
        return job.blockCount != 0U && job.blockLenBytes != 0U;
    }

    __aicore__ inline void ProcessB1Row2D() {
        uint32_t start = 0U, end = 0U;
        GetGroupRange(start, end);
        if (start >= end) { return; }

        InitDoubleCopyFlags();
        CopyJob2D curJob;
        uint32_t groupIdx = start;
        while (groupIdx < end && !MakeB1Row2DJob(groupIdx, curJob)) { ++groupIdx; }
        if (groupIdx >= end) {
            DrainDoubleCopyFlags();
            return;
        }
        uint32_t curBuf = 0U;
        StartCopyInJob(curJob, curBuf);
        ++groupIdx;

        while (groupIdx < end) {
            CopyJob2D nextJob;
            bool hasNext = false;
            while (groupIdx < end) {
                if (MakeB1Row2DJob(groupIdx, nextJob)) {
                    hasNext = true;
                    ++groupIdx;
                    break;
                }
                ++groupIdx;
            }
            if (!hasNext) { break; }
            const uint32_t nextBuf = 1U - curBuf;
            StartCopyInJob(nextJob, nextBuf);
            FinishCopyOutJob(curJob, curBuf);
            curJob = nextJob;
            curBuf = nextBuf;
        }
        FinishCopyOutJob(curJob, curBuf);
        DrainDoubleCopyFlags();
    }

    template <uint32_t BS>
    __aicore__ inline bool MakeCompactJob(uint32_t groupIdx, CopyJob2D &job) const {
        if (groupIdx >= totalGroups_) { return false; }
        const uint32_t phaseLimit = BS * BS;
        uint32_t phase = 0U;
        uint32_t local = 0U;
        if (compactUniform_ != 0U && compactPhaseGroupSize_ != 0U) {
            phase = groupIdx / compactPhaseGroupSize_;
            if (phase >= phaseLimit) { return false; }
            local = groupIdx - phase * compactPhaseGroupSize_;
        } else {
            uint32_t lo = 0U;
            uint32_t hi = phaseLimit;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) >> 1;
                if (phasePrefix_[mid + 1U] <= groupIdx) { lo = mid + 1U; }
                else { hi = mid; }
            }
            phase = lo;
            if (phase >= phaseLimit) { return false; }
            local = groupIdx - phasePrefix_[phase];
        }
        uint32_t bh = 0U;
        uint32_t bw = 0U;
        if (BS == 2U) {
            bh = phase >> 1;
            bw = phase & 1U;
        } else if (BS == 4U) {
            bh = phase >> 2;
            bw = phase & 3U;
        } else if (BS == 8U) {
            bh = phase >> 3;
            bw = phase & 7U;
        } else {
            bh = phase / BS;
            bw = phase - bh * BS;
        }
        const uint32_t tilesIw = tilesIw_[bw];
        const uint32_t ihCount = hCount_[bh];
        if (tilesIw == 0U || ihCount == 0U) { return false; }
        const uint32_t tileIdx = local % tilesIw;
        uint32_t tmp = local / tilesIw;
        const uint32_t ihRel = tmp % ihCount;
        const uint32_t ob = tmp / ihCount;
        if (ob >= outBatch_) { return false; }
        const uint32_t iwRel = tileIdx * tileIw_;
        if (iwRel >= wCount_[bw]) { return false; }
        uint32_t count = tileIw_;
        if (iwRel + count > wCount_[bw]) { count = wCount_[bw] - iwRel; }
        const uint32_t ih = hBegin_[bh] + ihRel;
        const uint32_t iw0 = wBegin_[bw] + iwRel;
        const uint32_t ib = phase * outBatch_ + ob;
        uint32_t oh = 0U;
        uint32_t ow0 = 0U;
        if (BS == 2U) {
            oh = (ih << 1) + bh - cropTop_;
            ow0 = (iw0 << 1) + bw - cropLeft_;
        } else if (BS == 4U) {
            oh = (ih << 2) + bh - cropTop_;
            ow0 = (iw0 << 2) + bw - cropLeft_;
        } else if (BS == 8U) {
            oh = (ih << 3) + bh - cropTop_;
            ow0 = (iw0 << 3) + bw - cropLeft_;
        } else {
            oh = ih * BS + bh - cropTop_;
            ow0 = iw0 * BS + bw - cropLeft_;
        }
        job.srcOffset = ib * xNStride_ + ih * xHStride_ + iw0 * xWStride_;
        job.dstOffset = ob * yNStride_ + oh * yHStride_ + ow0 * yWStride_;
        job.blockCount = static_cast<uint16_t>(count);
        job.blockLenBytes = depthBytes_;
        job.srcStrideBytes = 0U;
        job.dstStrideBytes = (BS - 1U) * depthBytes_;
        return job.blockCount != 0U;
    }


    template <uint32_t BS>
    __aicore__ inline bool MakeCompactPhaseTileJob(uint32_t phase,
                                                    uint32_t ob,
                                                    uint32_t ihRel,
                                                    uint32_t tileIdx,
                                                    CopyJob2D &job) const {
        const uint32_t phaseLimit = BS * BS;
        if (phase >= phaseLimit || ob >= outBatch_) { return false; }
        uint32_t bh = 0U;
        uint32_t bw = 0U;
        if constexpr (BS == 2U) {
            bh = phase >> 1;
            bw = phase & 1U;
        } else if constexpr (BS == 4U) {
            bh = phase >> 2;
            bw = phase & 3U;
        } else if constexpr (BS == 8U) {
            bh = phase >> 3;
            bw = phase & 7U;
        } else {
            bh = phase / BS;
            bw = phase - bh * BS;
        }
        const uint32_t tilesIw = tilesIw_[bw];
        const uint32_t ihCount = hCount_[bh];
        if (tilesIw == 0U || ihCount == 0U || tileIdx >= tilesIw || ihRel >= ihCount) {
            return false;
        }
        const uint32_t iwRel = tileIdx * tileIw_;
        if (iwRel >= wCount_[bw]) { return false; }
        uint32_t count = tileIw_;
        if (iwRel + count > wCount_[bw]) { count = wCount_[bw] - iwRel; }
        if (count == 0U) { return false; }

        const uint32_t ih = hBegin_[bh] + ihRel;
        const uint32_t iw0 = wBegin_[bw] + iwRel;
        const uint32_t ib = phase * outBatch_ + ob;
        uint32_t oh = 0U;
        uint32_t ow0 = 0U;
        if constexpr (BS == 2U) {
            oh = (ih << 1) + bh - cropTop_;
            ow0 = (iw0 << 1) + bw - cropLeft_;
        } else if constexpr (BS == 4U) {
            oh = (ih << 2) + bh - cropTop_;
            ow0 = (iw0 << 2) + bw - cropLeft_;
        } else if constexpr (BS == 8U) {
            oh = (ih << 3) + bh - cropTop_;
            ow0 = (iw0 << 3) + bw - cropLeft_;
        } else {
            oh = ih * BS + bh - cropTop_;
            ow0 = iw0 * BS + bw - cropLeft_;
        }
        job.srcOffset = ib * xNStride_ + ih * xHStride_ + iw0 * xWStride_;
        job.dstOffset = ob * yNStride_ + oh * yHStride_ + ow0 * yWStride_;
        job.blockCount = static_cast<uint16_t>(count);
        job.blockLenBytes = depthBytes_;
        job.srcStrideBytes = 0U;
        job.dstStrideBytes = (BS - 1U) * depthBytes_;
        return true;
    }

    template <uint32_t BS>
    __aicore__ inline void ProcessCompactTinyPhase() {
        uint32_t start = 0U, end = 0U;
        GetGroupRange(start, end);
        if (start >= end) { return; }
        const uint32_t phaseLimit = BS * BS;
        if (end > phaseLimit) { end = phaseLimit; }
        for (uint32_t phase = start; phase < end; ++phase) {
            uint32_t bh = 0U;
            uint32_t bw = 0U;
            if constexpr (BS == 2U) {
                bh = phase >> 1;
                bw = phase & 1U;
            } else if constexpr (BS == 4U) {
                bh = phase >> 2;
                bw = phase & 3U;
            } else if constexpr (BS == 8U) {
                bh = phase >> 3;
                bw = phase & 7U;
            } else {
                bh = phase / BS;
                bw = phase - bh * BS;
            }
            const uint32_t tilesIw = tilesIw_[bw];
            const uint32_t ihCount = hCount_[bh];
            if (tilesIw == 0U || ihCount == 0U) { continue; }
            for (uint32_t ob = 0U; ob < outBatch_; ++ob) {
                for (uint32_t ihRel = 0U; ihRel < ihCount; ++ihRel) {
                    for (uint32_t tileIdx = 0U; tileIdx < tilesIw; ++tileIdx) {
                        CopyJob2D job;
                        if (MakeCompactPhaseTileJob<BS>(phase, ob, ihRel, tileIdx, job)) {
                            RunSingleJob(job);
                        }
                    }
                }
            }
        }
    }


    template <uint32_t BS>
    __aicore__ inline bool MakeB2NoCropFastJob(uint32_t groupIdx, CopyJob2D &job) const {
        if constexpr (BS != 2U) {
            return false;
        } else {
            const uint32_t tilesPerRow = (width_ + tileIw_ - 1U) / tileIw_;
            if (tilesPerRow == 0U) { return false; }
            const uint32_t phaseGroupSize = outBatch_ * height_ * tilesPerRow;
            if (phaseGroupSize == 0U) { return false; }

            const uint32_t phase = groupIdx / phaseGroupSize;
            if (phase >= 4U) { return false; }
            const uint32_t local = groupIdx - phase * phaseGroupSize;
            const uint32_t tileIdx = local % tilesPerRow;
            uint32_t tmp = local / tilesPerRow;
            const uint32_t ih = tmp % height_;
            const uint32_t ob = tmp / height_;
            if (ob >= outBatch_) { return false; }

            const uint32_t iw0 = tileIdx * tileIw_;
            if (iw0 >= width_) { return false; }
            uint32_t count = tileIw_;
            if (iw0 + count > width_) { count = width_ - iw0; }
            if (count == 0U) { return false; }

            const uint32_t bh = phase >> 1;
            const uint32_t bw = phase & 1U;
            const uint32_t ib = phase * outBatch_ + ob;
            const uint32_t oh = (ih << 1) + bh;
            const uint32_t ow0 = (iw0 << 1) + bw;

            job.srcOffset = ib * xNStride_ + ih * xHStride_ + iw0 * xWStride_;
            job.dstOffset = ob * yNStride_ + oh * yHStride_ + ow0 * yWStride_;
            job.blockCount = static_cast<uint16_t>(count);
            job.blockLenBytes = depthBytes_;
            job.srcStrideBytes = 0U;
            job.dstStrideBytes = depthBytes_;
            return true;
        }
    }

    template <uint32_t BS>
    __aicore__ inline void ProcessB2NoCropFast() {
        if constexpr (BS != 2U) {
            return;
        } else {
            uint32_t start = 0U, end = 0U;
            GetGroupRange(start, end);
            if (start >= end) { return; }

            InitDoubleCopyFlags();
            CopyJob2D curJob;
            uint32_t groupIdx = start;
            while (groupIdx < end && !MakeB2NoCropFastJob<BS>(groupIdx, curJob)) { ++groupIdx; }
            if (groupIdx >= end) {
                DrainDoubleCopyFlags();
                return;
            }
            uint32_t curBuf = 0U;
            StartCopyInJob(curJob, curBuf);
            ++groupIdx;

            while (groupIdx < end) {
                CopyJob2D nextJob;
                bool hasNext = false;
                while (groupIdx < end) {
                    if (MakeB2NoCropFastJob<BS>(groupIdx, nextJob)) {
                        hasNext = true;
                        ++groupIdx;
                        break;
                    }
                    ++groupIdx;
                }
                if (!hasNext) { break; }
                const uint32_t nextBuf = 1U - curBuf;
                StartCopyInJob(nextJob, nextBuf);
                FinishCopyOutJob(curJob, curBuf);
                curJob = nextJob;
                curBuf = nextBuf;
            }
            FinishCopyOutJob(curJob, curBuf);
            DrainDoubleCopyFlags();
        }
    }

    template <uint32_t BS>
    __aicore__ inline void ProcessCompactSmall() {
        uint32_t start = 0U, end = 0U;
        GetGroupRange(start, end);
        if (start >= end) { return; }
        for (uint32_t groupIdx = start; groupIdx < end; ++groupIdx) {
            CopyJob2D job;
            if (MakeCompactJob<BS>(groupIdx, job)) {
                RunSingleJob(job);
            }
        }
    }

    template <uint32_t BS>
    __aicore__ inline void ProcessCompactStripe() {
        if (compactRowAssembly_ != 0U || strategy_ == BTS_STRATEGY_COMPACT_ROWASM) {
            ProcessCompactRowAssemble<BS>();
            return;
        }
        if (strategy_ == BTS_STRATEGY_B2_NOCROP_FAST) {
            ProcessB2NoCropFast<BS>();
            return;
        }
        if (strategy_ == BTS_STRATEGY_COMPACT_TINY_PHASE) {
            ProcessCompactTinyPhase<BS>();
            return;
        }
        if (strategy_ == BTS_STRATEGY_COMPACT_SMALL) {
            ProcessCompactSmall<BS>();
            return;
        }
        uint32_t start = 0U, end = 0U;
        GetGroupRange(start, end);
        if (start >= end) { return; }

        InitDoubleCopyFlags();
        CopyJob2D curJob;
        uint32_t groupIdx = start;
        while (groupIdx < end && !MakeCompactJob<BS>(groupIdx, curJob)) { ++groupIdx; }
        if (groupIdx >= end) {
            DrainDoubleCopyFlags();
            return;
        }
        uint32_t curBuf = 0U;
        StartCopyInJob(curJob, curBuf);
        ++groupIdx;

        while (groupIdx < end) {
            CopyJob2D nextJob;
            bool hasNext = false;
            while (groupIdx < end) {
                if (MakeCompactJob<BS>(groupIdx, nextJob)) {
                    hasNext = true;
                    ++groupIdx;
                    break;
                }
                ++groupIdx;
            }
            if (!hasNext) { break; }
            const uint32_t nextBuf = 1U - curBuf;
            StartCopyInJob(nextJob, nextBuf);
            FinishCopyOutJob(curJob, curBuf);
            curJob = nextJob;
            curBuf = nextBuf;
        }
        FinishCopyOutJob(curJob, curBuf);
        DrainDoubleCopyFlags();
    }


    template <uint32_t BS>
    __aicore__ inline uint32_t ModBs(uint32_t value) const {
        if constexpr (BS == 2U) { return value & 1U; }
        else if constexpr (BS == 4U) { return value & 3U; }
        else if constexpr (BS == 8U) { return value & 7U; }
        else { return value % BS; }
    }

    template <uint32_t BS>
    __aicore__ inline uint32_t DivBs(uint32_t value) const {
        if constexpr (BS == 2U) { return value >> 1; }
        else if constexpr (BS == 4U) { return value >> 2; }
        else if constexpr (BS == 8U) { return value >> 3; }
        else { return value / BS; }
    }

    template <uint32_t BS>
    __aicore__ inline uint32_t MulBs(uint32_t value) const {
        if constexpr (BS == 2U) { return value << 1; }
        else if constexpr (BS == 4U) { return value << 2; }
        else if constexpr (BS == 8U) { return value << 3; }
        else { return value * BS; }
    }

    struct CompactRowJob {
        uint32_t ob;
        uint32_t oh0;
        uint32_t ohCount;
        uint32_t ow0;
        uint32_t owCount;
    };

    __aicore__ inline bool MakeCompactRowJob(uint32_t groupIdx, CompactRowJob &job) const {
        const uint32_t tilesPerRow = compactRowTilesPerWidth_;
        const uint32_t groupsPerBatch = compactRowGroupsPerBatch_;
        if (tilesPerRow == 0U || groupsPerBatch == 0U) { return false; }

        const uint32_t ob = groupIdx / groupsPerBatch;
        const uint32_t localGroup = groupIdx - ob * groupsPerBatch;
        uint32_t rowTile;
        uint32_t tileIdx;
        if (tilesPerRow == 1U) {
            rowTile = localGroup;
            tileIdx = 0U;
        } else {
            rowTile = localGroup / tilesPerRow;
            tileIdx = localGroup - rowTile * tilesPerRow;
        }

        const uint32_t oh0 = rowTile * tileRows_;
        if (ob >= outBatch_ || oh0 >= outHeight_) { return false; }
        uint32_t ohCount = tileRows_;
        if (oh0 + ohCount > outHeight_) { ohCount = outHeight_ - oh0; }
        const uint32_t ow0 = tileIdx * tileIw_;
        if (ow0 >= outWidth_) { return false; }
        uint32_t owCount = tileIw_;
        if (ow0 + owCount > outWidth_) { owCount = outWidth_ - ow0; }
        if (ohCount == 0U || owCount == 0U) { return false; }
        job.ob = ob;
        job.oh0 = oh0;
        job.ohCount = ohCount;
        job.ow0 = ow0;
        job.owCount = owCount;
        return true;
    }

    template <uint32_t BS>
    __aicore__ inline void StartCompactRowAssembleJob(const CompactRowJob &job, uint32_t bufId) {
        WaitFreeFlag(bufId);
        LocalTensor<float> local = GetLocalById(bufId);
        const uint32_t rowStride = job.owCount * depth_;
        const uint32_t ubDstStride = ((BS - 1U) * depthBytes_) >> 5;

        // v23: common full-width/no-W-crop row-assembly fast path.
        // In this case every phase contributes exactly width_ contiguous depth blocks
        // starting from iw=0, so the inner loop avoids per-phase delta/count/div
        // computations.  The vertical (bh, ih) decode is advanced incrementally
        // across bundled rows to avoid repeated div/mod, especially for BS=3/5/6/7.
        if (job.ow0 == 0U && job.owCount == outWidth_ && cropLeft_ == 0U && cropRight_ == 0U) {
            uint32_t fh = job.oh0 + cropTop_;
            uint32_t bh = ModBs<BS>(fh);
            uint32_t ih = DivBs<BS>(fh);
            for (uint32_t r = 0U; r < job.ohCount; ++r) {
                const uint32_t rowLocalBase = r * rowStride;
                const uint32_t phaseBase = bh * BS;
                const uint32_t srcRowBase = ih * xHStride_;
                for (uint32_t bw = 0U; bw < BS; ++bw) {
                    const uint32_t ib = (phaseBase + bw) * outBatch_ + job.ob;
                    const uint32_t localOffset = rowLocalBase + bw * depth_;
                    const uint32_t srcOffset = ib * xNStride_ + srcRowBase;

                    DataCopyExtParams inParams;
                    inParams.blockCount = static_cast<uint16_t>(width_);
                    inParams.blockLen = depthBytes_;
                    inParams.srcStride = 0U;
                    inParams.dstStride = ubDstStride;
                    inParams.rsv = 0;
                    DataCopyPad(local[localOffset], xGm_[srcOffset], inParams, NoPadParams());
                }
                ++bh;
                if (bh == BS) {
                    bh = 0U;
                    ++ih;
                }
            }
            SetInReadyFlag(bufId);
            return;
        }

        const uint32_t fw0Base = job.ow0 + cropLeft_;
        const uint32_t fwEnd = fw0Base + job.owCount;
        const uint32_t fw0Mod = ModBs<BS>(fw0Base);
        uint32_t fh = job.oh0 + cropTop_;
        uint32_t bh = ModBs<BS>(fh);
        uint32_t ih = DivBs<BS>(fh);

        for (uint32_t r = 0U; r < job.ohCount; ++r) {
            const uint32_t rowLocalBase = r * rowStride;

            for (uint32_t bw = 0U; bw < BS; ++bw) {
                uint32_t delta = (bw >= fw0Mod) ? (bw - fw0Mod) : (bw + BS - fw0Mod);
                const uint32_t firstFw = fw0Base + delta;
                if (firstFw >= fwEnd) { continue; }
                const uint32_t count = (fwEnd - 1U - firstFw) / BS + 1U;
                const uint32_t iw0 = DivBs<BS>(firstFw);
                const uint32_t phase = bh * BS + bw;
                const uint32_t ib = phase * outBatch_ + job.ob;
                const uint32_t localOw = firstFw - cropLeft_ - job.ow0;
                const uint32_t localOffset = rowLocalBase + localOw * depth_;
                const uint32_t srcOffset = ib * xNStride_ + ih * xHStride_ + iw0 * xWStride_;

                DataCopyExtParams inParams;
                inParams.blockCount = static_cast<uint16_t>(count);
                inParams.blockLen = depthBytes_;
                inParams.srcStride = 0U;
                inParams.dstStride = ubDstStride;
                inParams.rsv = 0;
                DataCopyPad(local[localOffset], xGm_[srcOffset], inParams, NoPadParams());
            }

            ++bh;
            if (bh == BS) {
                bh = 0U;
                ++ih;
            }
        }
        SetInReadyFlag(bufId);
    }

    __aicore__ inline void FinishCompactRowAssembleJob(const CompactRowJob &job, uint32_t bufId) {
        WaitInReadyFlag(bufId);
        LocalTensor<float> local = GetLocalById(bufId);
        const uint32_t dstOffset = job.ob * yNStride_ + job.oh0 * yHStride_ + job.ow0 * yWStride_;
        if (job.ow0 == 0U && job.owCount == outWidth_) {
            DataCopy(yGm_[dstOffset], local, job.ohCount * job.owCount * depth_);
        } else {
            DataCopyExtParams outParams;
            outParams.blockCount = static_cast<uint16_t>(job.ohCount);
            outParams.blockLen = job.owCount * depthBytes_;
            outParams.srcStride = 0U;
            outParams.dstStride = (outWidth_ - job.owCount) * depthBytes_;
            outParams.rsv = 0;
            DataCopyPad(yGm_[dstOffset], local, outParams);
        }
        SetFreeFlag(bufId);
    }

    template <uint32_t BS>
    __aicore__ inline void ProcessCompactRowAssemble() {
        uint32_t start = 0U, end = 0U;
        GetGroupRange(start, end);
        if (start >= end) { return; }

        InitDoubleCopyFlags();
        CompactRowJob curJob;
        uint32_t groupIdx = start;
        while (groupIdx < end && !MakeCompactRowJob(groupIdx, curJob)) { ++groupIdx; }
        if (groupIdx >= end) {
            DrainDoubleCopyFlags();
            return;
        }

        uint32_t curBuf = 0U;
        StartCompactRowAssembleJob<BS>(curJob, curBuf);
        ++groupIdx;

        while (groupIdx < end) {
            CompactRowJob nextJob;
            bool hasNext = false;
            while (groupIdx < end) {
                if (MakeCompactRowJob(groupIdx, nextJob)) {
                    hasNext = true;
                    ++groupIdx;
                    break;
                }
                ++groupIdx;
            }
            if (!hasNext) { break; }
            const uint32_t nextBuf = 1U - curBuf;
            StartCompactRowAssembleJob<BS>(nextJob, nextBuf);
            FinishCompactRowAssembleJob(curJob, curBuf);
            curJob = nextJob;
            curBuf = nextBuf;
        }

        FinishCompactRowAssembleJob(curJob, curBuf);
        DrainDoubleCopyFlags();
    }

    __aicore__ inline int32_t CeilDivS32(int32_t a, int32_t b) const {
        if (a >= 0) { return (a + b - 1) / b; }
        return a / b;
    }

    __aicore__ inline uint32_t ClampBeginKernel(int32_t crop, int32_t phase, int32_t block, uint32_t dim) const {
        int32_t v = CeilDivS32(crop - phase, block);
        if (v < 0) { v = 0; }
        const uint32_t uv = static_cast<uint32_t>(v);
        return MinU32(uv, dim);
    }

    __aicore__ inline uint32_t ClampEndKernel(int32_t crop, int32_t outDim, int32_t phase, int32_t block, uint32_t dim) const {
        int32_t v = CeilDivS32(crop + outDim - phase, block);
        if (v < 0) { v = 0; }
        const uint32_t uv = static_cast<uint32_t>(v);
        return MinU32(uv, dim);
    }

    __aicore__ inline bool MakeGenericStripeJob(uint32_t groupIdx, CopyJob2D &job) const {
        const uint32_t tilesPerInputRow = (width_ + tileIw_ - 1U) / tileIw_;
        if (tilesPerInputRow == 0U) { return false; }
        const uint32_t bsz = blockSize_;
        const uint32_t tileIdx = groupIdx % tilesPerInputRow;
        uint32_t tmp = groupIdx / tilesPerInputRow;
        const uint32_t ih = tmp % height_;
        tmp /= height_;
        const uint32_t phaseArea = bsz * bsz;
        const uint32_t phase = tmp % phaseArea;
        tmp /= phaseArea;
        const uint32_t ob = tmp;
        if (ob >= outBatch_) { return false; }
        const uint32_t bh = phase / bsz;
        const uint32_t bw = phase - bh * bsz;
        const uint32_t ihBegin = ClampBeginKernel(static_cast<int32_t>(cropTop_), static_cast<int32_t>(bh),
                                                  static_cast<int32_t>(bsz), height_);
        const uint32_t ihEnd = ClampEndKernel(static_cast<int32_t>(cropTop_), static_cast<int32_t>(outHeight_),
                                              static_cast<int32_t>(bh), static_cast<int32_t>(bsz), height_);
        if (ih < ihBegin || ih >= ihEnd) { return false; }
        const uint32_t iwTile0 = tileIdx * tileIw_;
        const uint32_t iwTile1 = MinU32(width_, iwTile0 + tileIw_);
        const uint32_t iwBegin = ClampBeginKernel(static_cast<int32_t>(cropLeft_), static_cast<int32_t>(bw),
                                                  static_cast<int32_t>(bsz), width_);
        const uint32_t iwEnd = ClampEndKernel(static_cast<int32_t>(cropLeft_), static_cast<int32_t>(outWidth_),
                                              static_cast<int32_t>(bw), static_cast<int32_t>(bsz), width_);
        const uint32_t validIw0 = MaxU32(iwTile0, iwBegin);
        const uint32_t validIw1 = MinU32(iwTile1, iwEnd);
        if (validIw0 >= validIw1) { return false; }
        const uint32_t count = validIw1 - validIw0;
        const uint32_t ib = phase * outBatch_ + ob;
        const uint32_t oh = ih * bsz + bh - cropTop_;
        const uint32_t ow0 = validIw0 * bsz + bw - cropLeft_;
        job.srcOffset = ib * xNStride_ + ih * xHStride_ + validIw0 * xWStride_;
        job.dstOffset = ob * yNStride_ + oh * yHStride_ + ow0 * yWStride_;
        job.blockCount = static_cast<uint16_t>(count);
        job.blockLenBytes = depthBytes_;
        job.srcStrideBytes = 0U;
        job.dstStrideBytes = (bsz - 1U) * depthBytes_;
        return job.blockCount != 0U && job.blockLenBytes != 0U;
    }

    __aicore__ inline void ProcessGenericStripe() {
        uint32_t start = 0U, end = 0U;
        GetGroupRange(start, end);
        if (start >= end) { return; }

        InitDoubleCopyFlags();
        CopyJob2D curJob;
        uint32_t groupIdx = start;
        while (groupIdx < end && !MakeGenericStripeJob(groupIdx, curJob)) { ++groupIdx; }
        if (groupIdx >= end) {
            DrainDoubleCopyFlags();
            return;
        }
        uint32_t curBuf = 0U;
        StartCopyInJob(curJob, curBuf);
        ++groupIdx;

        while (groupIdx < end) {
            CopyJob2D nextJob;
            bool hasNext = false;
            while (groupIdx < end) {
                if (MakeGenericStripeJob(groupIdx, nextJob)) {
                    hasNext = true;
                    ++groupIdx;
                    break;
                }
                ++groupIdx;
            }
            if (!hasNext) { break; }
            const uint32_t nextBuf = 1U - curBuf;
            StartCopyInJob(nextJob, nextBuf);
            FinishCopyOutJob(curJob, curBuf);
            curJob = nextJob;
            curBuf = nextBuf;
        }
        FinishCopyOutJob(curJob, curBuf);
        DrainDoubleCopyFlags();
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECIN> buf0_;
    TBuf<TPosition::VECIN> buf1_;
    LocalTensor<float> local0_;
    LocalTensor<float> local1_;
    LocalTensor<float> local_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;

    uint32_t batch_;
    uint32_t height_;
    uint32_t width_;
    uint32_t depth_;
    uint32_t outBatch_;
    uint32_t outHeight_;
    uint32_t outWidth_;
    uint32_t totalElems_;
    uint32_t totalGroups_;
    uint32_t xWStride_;
    uint32_t xHStride_;
    uint32_t xNStride_;
    uint32_t yWStride_;
    uint32_t yHStride_;
    uint32_t yNStride_;

    uint32_t hBegin_[BTS_MAX_COMPACT_BS];
    uint32_t hCount_[BTS_MAX_COMPACT_BS];
    uint32_t wBegin_[BTS_MAX_COMPACT_BS];
    uint32_t wCount_[BTS_MAX_COMPACT_BS];
    uint32_t tilesIw_[BTS_MAX_COMPACT_BS];
    uint32_t phasePrefix_[BTS_MAX_COMPACT_PHASE + 1U];
    uint32_t compactPhaseNum_;
    uint32_t compactUniform_;
    uint32_t compactPhaseGroupSize_;
    uint32_t compactRowAssembly_;
    uint32_t compactRowTilesPerWidth_;
    uint32_t compactRowGroupsPerBatch_;

    uint32_t blockSize_;
    uint32_t cropTop_;
    uint32_t cropBottom_;
    uint32_t cropLeft_;
    uint32_t cropRight_;
    uint32_t mode_;
    uint32_t strategy_;
    uint32_t depthBytes_;
    uint32_t depthAlignedBytes_;
    uint32_t tileElems_;
    uint32_t tileRows_;
    uint32_t tileIw_;
    uint32_t ubBytes_;
    uint32_t usedCoreNum_;
    uint32_t bigCoreNum_;
    uint32_t bigCoreWorkNum_;
    uint32_t smallCoreWorkNum_;
};

template <typename DT_X, uint32_t MODE_KIND>
class KernelSelector;

template <uint32_t MODE_KIND>
class KernelSelector<half, MODE_KIND> {
public:
    __aicore__ inline static void Run(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling) {
        KernelBatchToSpaceF16 op;
        op.template Init<MODE_KIND>(x, y, tiling);
        op.template Process<MODE_KIND>();
    }
};

template <uint32_t MODE_KIND>
class KernelSelector<float, MODE_KIND> {
public:
    __aicore__ inline static void Run(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling) {
        KernelBatchToSpaceF32 op;
        op.template Init<MODE_KIND>(x, y, tiling);
        op.template Process<MODE_KIND>();
    }
};

template <typename DT_X, uint32_t MODE_KIND>
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingData, tiling_data, tiling);
    KernelSelector<DT_X, MODE_KIND>::Run(x, y, tiling_data);
}
