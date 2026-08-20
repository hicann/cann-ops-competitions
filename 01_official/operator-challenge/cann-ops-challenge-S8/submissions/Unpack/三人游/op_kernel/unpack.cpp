#include "kernel_inc.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

#include <limits>
#include <type_traits>

using namespace AscendC;

namespace {

constexpr uint32_t kBufferNum = 2;
constexpr uint32_t kAlignBytes = 32;
constexpr int64_t kGatherSeedCount = 8;
constexpr int64_t kTaskModeDefault = 0;
constexpr int64_t kTaskModeSplitOuterByOutput = 1;
constexpr int64_t kTaskModeLastAxisInt32PackedRows = 2;
constexpr int64_t kTaskModeSmallInnerFloat32PackedRows = 3;
constexpr int64_t kTaskModeWideNonAlignedBfloat16Rows = 4;
constexpr int64_t kTaskModeLargeAlignedFloatOutputSlabs = 5;
constexpr int64_t kTaskModeWideAlignedBfloat16BoundRows = 6;
constexpr int64_t kTaskModeSmallInnerFloat16PackedRows = 8;

template <typename T>
class KernelUnpack {
  public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const UnpackTilingData &tiling) {
        num_ = tiling.num;
        outer_ = tiling.outer;
        inner_ = tiling.inner;
        taskMode_ = tiling.task_mode;
        totalTasks_ = tiling.total_tasks;
        tasksPerCore_ = tiling.tasks_per_core;
        tileElems_ = tiling.tile_elems;
        directRowBatch_ = (tiling.direct_row_batch != 0);

        inputGm_.SetGlobalBuffer((__gm__ T *)x, static_cast<uint32_t>(tiling.outer * tiling.num * tiling.inner));
        outputList_.Init((__gm__ void *)y);
        const bool useFloatOutputSlab = (taskMode_ == kTaskModeLargeAlignedFloatOutputSlabs);
        const bool useSmallInnerPackedRows = (taskMode_ == kTaskModeSmallInnerFloat16PackedRows);
        const bool useWideBfloat16BoundRows =
            (taskMode_ == kTaskModeWideAlignedBfloat16BoundRows ||
             taskMode_ == kTaskModeWideNonAlignedBfloat16Rows);
        const bool useBoundQueue = useWideBfloat16BoundRows || useSmallInnerPackedRows;
        const uint32_t inputBufferCount =
            (useFloatOutputSlab || useSmallInnerPackedRows) ? 1U : kBufferNum;
        const uint32_t outputBufferCount =
            (useFloatOutputSlab || useSmallInnerPackedRows) ? 1U : kBufferNum;
        const uint32_t inputBufferElems =
            (taskMode_ == kTaskModeLastAxisInt32PackedRows || taskMode_ == kTaskModeSmallInnerFloat32PackedRows ||
             taskMode_ == kTaskModeSmallInnerFloat16PackedRows)
                ? (taskMode_ == kTaskModeSmallInnerFloat16PackedRows)
                      ? static_cast<uint32_t>(GetSmallInnerPaddedRowElems() * GetSmallInnerRowsPerTask())
                      : static_cast<uint32_t>(tileElems_ * num_)
                : (taskMode_ == kTaskModeWideNonAlignedBfloat16Rows)
                      ? static_cast<uint32_t>(GetWideNonAlignedPaddedRowElems() * GetWideNonAlignedRowsPerTask())
                : (taskMode_ == kTaskModeWideAlignedBfloat16BoundRows)
                      ? static_cast<uint32_t>(tileElems_)
                : static_cast<uint32_t>(tileElems_);
        if (useBoundQueue) {
            pipe_.InitBuffer(boundQueue_, kBufferNum, inputBufferElems * sizeof(T));
        } else {
            pipe_.InitBuffer(inQueue_, inputBufferCount, inputBufferElems * sizeof(T));
        }
        if (!directRowBatch_ && taskMode_ != kTaskModeWideNonAlignedBfloat16Rows &&
            taskMode_ != kTaskModeWideAlignedBfloat16BoundRows &&
            taskMode_ != kTaskModeLargeAlignedFloatOutputSlabs) {
            const uint32_t outputBufferElems =
                (taskMode_ == kTaskModeWideNonAlignedBfloat16Rows)
                    ? static_cast<uint32_t>(GetWideNonAlignedPaddedRowElems() * GetWideNonAlignedRowsPerTask())
                : (taskMode_ == kTaskModeLargeAlignedFloatOutputSlabs)
                    ? static_cast<uint32_t>(GetLargeAlignedFloatRowsPerTask() * inner_)
                : static_cast<uint32_t>(tileElems_);
            pipe_.InitBuffer(outQueue_, outputBufferCount, outputBufferElems * sizeof(T));
        }
        if (taskMode_ == kTaskModeLastAxisInt32PackedRows) {
            pipe_.InitBuffer(indexBuf_, static_cast<uint32_t>(tileElems_ * 2 * sizeof(int32_t)));
        } else if (taskMode_ == kTaskModeSmallInnerFloat32PackedRows) {
            pipe_.InitBuffer(indexBuf_, static_cast<uint32_t>(tileElems_ * sizeof(int32_t)));
        }
    }

    __aicore__ inline void Process() {
        if (num_ <= 0 || outer_ <= 0 || inner_ <= 0 || totalTasks_ <= 0 || tileElems_ <= 0) {
            return;
        }

        int64_t startTask = 0;
        int64_t endTask = 0;
        if (!ComputeBlockTaskRange(startTask, endTask)) {
            return;
        }

        if (taskMode_ == kTaskModeLastAxisInt32PackedRows && CanUseLastAxisInt32PackedRows()) {
            ProcessLastAxisInt32PackedRows(startTask, endTask);
            return;
        }
        if (taskMode_ == kTaskModeSmallInnerFloat16PackedRows && CanUseSmallInnerFloat16PackedRows()) {
            ProcessSmallInnerFloat16PadRows(startTask, endTask);
            return;
        }
        if (taskMode_ == kTaskModeSmallInnerFloat32PackedRows && CanUseSmallInnerFloat32PackedRows()) {
            ProcessSmallInnerFloat32PackedRows(startTask, endTask);
            return;
        }
        if (taskMode_ == kTaskModeWideNonAlignedBfloat16Rows && CanUseWideNonAlignedBfloat16Rows()) {
            ProcessWideNonAlignedBfloat16Rows(startTask, endTask);
            return;
        }
        if (taskMode_ == kTaskModeWideAlignedBfloat16BoundRows && CanUseWideAlignedBfloat16BoundRows()) {
            ProcessWideAlignedBfloat16BoundRows(startTask, endTask);
            return;
        }
        if (taskMode_ == kTaskModeLargeAlignedFloatOutputSlabs && CanUseLargeAlignedFloatOutputSlabs()) {
            ProcessLargeAlignedFloatOutputSlabs(startTask, endTask);
            return;
        }
        for (int64_t taskId = startTask; taskId < endTask; ++taskId) {
            int64_t outIdx = taskId;
            int64_t outerBegin = 0;
            int64_t outerEnd = outer_;
            if (taskMode_ == kTaskModeSplitOuterByOutput) {
                const int64_t rowsPerTask = tileElems_ >= inner_ ? (tileElems_ / inner_) : 1;
                outIdx = taskId % num_;
                outerBegin = (taskId / num_) * rowsPerTask;
                outerEnd = outerBegin + rowsPerTask;
                if (outerEnd > outer_) {
                    outerEnd = outer_;
                }
                if (outerBegin >= outer_) {
                    continue;
                }
            }

            __gm__ T *outAddr = outputList_.GetDataPtr<__gm__ T>(static_cast<uint32_t>(outIdx));
            if (outAddr == nullptr) {
                continue;
            }

            GlobalTensor<T> outputGm;
            outputGm.SetGlobalBuffer(outAddr, static_cast<uint32_t>(outer_ * inner_));

            if (CanUseRowBatchCopy()) {
                const int64_t rowsPerBatch = tileElems_ / inner_;
                for (int64_t outerIdx = outerBegin; outerIdx < outerEnd; outerIdx += rowsPerBatch) {
                    int64_t currentRows = outerEnd - outerIdx;
                    if (currentRows > rowsPerBatch) {
                        currentRows = rowsPerBatch;
                    }
                    if (UseDirectRowBatch()) {
                        CopyRowBatchDirect(outputGm, outIdx, outerIdx, currentRows);
                    } else {
                        CopyRowBatch(outputGm, outIdx, outerIdx, currentRows);
                    }
                }
            } else {
                for (int64_t outerIdx = outerBegin; outerIdx < outerEnd; ++outerIdx) {
                    const int64_t inputBase = (outerIdx * num_ + outIdx) * inner_;
                    const int64_t outputBase = outerIdx * inner_;
                    for (int64_t offset = 0; offset < inner_; offset += tileElems_) {
                        int64_t currentTileElems = inner_ - offset;
                        if (currentTileElems > tileElems_) {
                            currentTileElems = tileElems_;
                        }
                        CopyTile(outputGm, inputBase + offset, outputBase + offset, currentTileElems);
                    }
                }
            }
        }
    }

  private:
    __aicore__ inline bool ComputeBlockTaskRange(int64_t &startTask, int64_t &endTask) const {
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        if (blockNum <= 0) {
            return false;
        }
        const int64_t formerTasks = totalTasks_ / blockNum;
        const int64_t tailCoreCount = totalTasks_ % blockNum;
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        int64_t blockTaskCount = formerTasks;
        startTask = blockIdx * formerTasks;
        if (blockIdx < tailCoreCount) {
            blockTaskCount += 1;
            startTask += blockIdx;
        } else {
            startTask += tailCoreCount;
        }
        if (blockTaskCount <= 0 || startTask >= totalTasks_) {
            return false;
        }
        endTask = startTask + blockTaskCount;
        if (endTask > totalTasks_) {
            endTask = totalTasks_;
        }
        return true;
    }

    __aicore__ inline bool CanUseLastAxisInt32PackedRows() const {
        if constexpr (std::is_same_v<T, bool>) {
            return false;
        }
        return sizeof(T) == 4 && num_ == 33 && inner_ == 1;
    }

    __aicore__ inline bool CanUseSmallInnerFloat16PackedRows() const {
        if constexpr (std::is_same_v<T, bool>) {
            return false;
        }
        return sizeof(T) == 2 && num_ == 63 && inner_ > 0 && inner_ < 16;
    }

    __aicore__ inline bool CanUseSmallInnerFloat32PackedRows() const {
        if constexpr (std::is_same_v<T, bool>) {
            return false;
        }
        return sizeof(T) == 4 && num_ >= 129 && inner_ > 0 && inner_ <= 4;
    }

    __aicore__ inline bool CanUseWideNonAlignedBfloat16Rows() const {
        if constexpr (std::is_same_v<T, bool>) {
            return false;
        }
        return sizeof(T) == 2 && num_ > 17 &&
               (((inner_ >= 1024 && inner_ < 2048) || inner_ == 3073) && ((inner_ % 16) != 0));
    }

    __aicore__ inline bool CanUseWideAlignedBfloat16BoundRows() const {
        if constexpr (std::is_same_v<T, bool>) {
            return false;
        }
        return sizeof(T) == 2 && num_ > 17 && inner_ >= 1024 && inner_ < 4096 && ((inner_ % 16) == 0);
    }

    __aicore__ inline bool CanUseLargeAlignedFloatOutputSlabs() const {
        if constexpr (std::is_same_v<T, bool>) {
            return false;
        }
        if (!(sizeof(T) == 4 && num_ == 255 && inner_ >= 4096 && ((inner_ % 8) == 0) &&
              GetLargeAlignedFloatSlabOutputs() > 1)) {
            return false;
        }
        if (inner_ == 4096) {
            return outer_ > 64;
        }
        return false;
    }

    __aicore__ inline int64_t GetWideNonAlignedPaddedRowElems() const {
        const uint64_t rowBytes = static_cast<uint64_t>(inner_) * sizeof(T);
        const uint64_t paddedRowBytes = ((rowBytes + kAlignBytes - 1ULL) / kAlignBytes) * kAlignBytes;
        return static_cast<int64_t>(paddedRowBytes / sizeof(T));
    }

    __aicore__ inline int64_t GetWideNonAlignedRowsPerTask() const {
        return tileElems_ >= inner_ ? (tileElems_ / inner_) : 1;
    }

    __aicore__ inline int64_t GetSmallInnerPaddedRowElems() const {
        const uint64_t rowBytes = static_cast<uint64_t>(inner_) * sizeof(T);
        const uint64_t paddedRowBytes = ((rowBytes + kAlignBytes - 1ULL) / kAlignBytes) * kAlignBytes;
        return static_cast<int64_t>(paddedRowBytes / sizeof(T));
    }

    __aicore__ inline int64_t GetSmallInnerRowsPerTask() const {
        return tileElems_ >= inner_ ? (tileElems_ / inner_) : 1;
    }

    __aicore__ inline int64_t GetLargeAlignedFloatSlabOutputs() const {
        if (inner_ <= 4096) {
            return 4;
        }
        if (inner_ <= 8192) {
            return 2;
        }
        return 2;
    }

    __aicore__ inline int64_t GetLargeAlignedFloatRowsPerTask() const {
        if (inner_ <= 8192) {
            return 2;
        }
        return 1;
    }

    __aicore__ inline void ProcessLastAxisInt32PackedRows(int64_t startTask, int64_t endTask) {
        const int64_t rowsPerTask = tileElems_ > 0 ? tileElems_ : 1;
        for (int64_t taskId = startTask; taskId < endTask; ++taskId) {
            const int64_t outerBegin = taskId * rowsPerTask;
            if (outerBegin >= outer_) {
                break;
            }
            int64_t rowCount = outer_ - outerBegin;
            if (rowCount > rowsPerTask) {
                rowCount = rowsPerTask;
            }
            CopyLastAxisInt32PackedRows(outerBegin, rowCount);
        }
    }

    __aicore__ inline void CopyLastAxisInt32PackedRows(int64_t outerBegin, int64_t rowCount) {
        __gm__ T *outAddrs[33];
        for (int64_t outIdx = 0; outIdx < num_; ++outIdx) {
            outAddrs[outIdx] = outputList_.GetDataPtr<__gm__ T>(static_cast<uint32_t>(outIdx));
        }
        const int64_t inputBase = outerBegin * num_;
        const int64_t packedElems = rowCount * num_;
        const uint32_t inputBytes = static_cast<uint32_t>(packedElems * sizeof(T));
        const uint32_t outputBytes = static_cast<uint32_t>(rowCount * sizeof(T));
        const bool outputAligned = ((outputBytes % kAlignBytes) == 0);

        LocalTensor<T> inputLocal = inQueue_.AllocTensor<T>();
        if ((inputBytes % kAlignBytes) == 0) {
            DataCopy(inputLocal, inputGm_[inputBase], static_cast<uint32_t>(packedElems));
        } else {
            DataCopyExtParams gmToUbParams = {1, inputBytes, 0, 0, 0};
            DataCopyPadExtParams<T> padParams = {false, 0, 0, static_cast<T>(0)};
            DataCopyPad(inputLocal, inputGm_[inputBase], gmToUbParams, padParams);
        }
        inQueue_.EnQue(inputLocal);

        inputLocal = inQueue_.DeQue<T>();
        const int32_t alignedRows = (rowCount >= kGatherSeedCount)
                                        ? static_cast<int32_t>((rowCount / kGatherSeedCount) * kGatherSeedCount)
                                        : 0;
        LocalTensor<int32_t> currentGatherOffsetsI32;
        LocalTensor<int32_t> nextGatherOffsetsI32;
        LocalTensor<uint32_t> currentGatherOffsetsU32;
        LocalTensor<uint32_t> nextGatherOffsetsU32;
        if (alignedRows > 0) {
            PrepareLastAxisInt32GatherOffsets(0, alignedRows);
            currentGatherOffsetsI32 = indexBuf_.Get<int32_t>();
            nextGatherOffsetsI32 = currentGatherOffsetsI32[alignedRows];
            currentGatherOffsetsU32 = indexBuf_.Get<uint32_t>();
            nextGatherOffsetsU32 = currentGatherOffsetsU32[alignedRows];
        }
        for (int64_t outIdx = 0; outIdx < num_; ++outIdx) {
            __gm__ T *outAddr = outAddrs[outIdx];
            if (outAddr == nullptr) {
                if (alignedRows > 0 && (outIdx + 1) < num_) {
                    Adds(nextGatherOffsetsI32, currentGatherOffsetsI32, static_cast<int32_t>(sizeof(T)), alignedRows);
                    PipeBarrier<PIPE_V>();
                    LocalTensor<int32_t> tmpGatherOffsetsI32 = currentGatherOffsetsI32;
                    currentGatherOffsetsI32 = nextGatherOffsetsI32;
                    nextGatherOffsetsI32 = tmpGatherOffsetsI32;
                    LocalTensor<uint32_t> tmpGatherOffsetsU32 = currentGatherOffsetsU32;
                    currentGatherOffsetsU32 = nextGatherOffsetsU32;
                    nextGatherOffsetsU32 = tmpGatherOffsetsU32;
                }
                continue;
            }

            GlobalTensor<T> outputGm;
            outputGm.SetGlobalBuffer(outAddr, static_cast<uint32_t>(outer_));
            LocalTensor<T> outputLocal = outQueue_.AllocTensor<T>();
            if (alignedRows > 0) {
                Gather(outputLocal, inputLocal, currentGatherOffsetsU32, 0, static_cast<uint32_t>(alignedRows));
                PipeBarrier<PIPE_V>();
            }
            for (int64_t row = alignedRows; row < rowCount; ++row) {
                outputLocal.SetValue(row, inputLocal.GetValue(row * num_ + outIdx));
            }

            if (outputAligned) {
                DataCopy(outputGm[outerBegin], outputLocal, static_cast<uint32_t>(rowCount));
            } else {
                DataCopyExtParams ubToGmParams = {1, outputBytes, 0, 0, 0};
                DataCopyPad(outputGm[outerBegin], outputLocal, ubToGmParams);
            }
            PipeBarrier<PIPE_MTE3>();
            outQueue_.FreeTensor(outputLocal);
            if (alignedRows > 0 && (outIdx + 1) < num_) {
                Adds(nextGatherOffsetsI32, currentGatherOffsetsI32, static_cast<int32_t>(sizeof(T)), alignedRows);
                PipeBarrier<PIPE_V>();
                LocalTensor<int32_t> tmpGatherOffsetsI32 = currentGatherOffsetsI32;
                currentGatherOffsetsI32 = nextGatherOffsetsI32;
                nextGatherOffsetsI32 = tmpGatherOffsetsI32;
                LocalTensor<uint32_t> tmpGatherOffsetsU32 = currentGatherOffsetsU32;
                currentGatherOffsetsU32 = nextGatherOffsetsU32;
                nextGatherOffsetsU32 = tmpGatherOffsetsU32;
            }
        }
        inQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void PrepareLastAxisInt32GatherOffsets(int64_t outIdx, int32_t count) {
        LocalTensor<int32_t> gatherOffsets = indexBuf_.Get<int32_t>();
        const int32_t seedCount = count < kGatherSeedCount ? count : static_cast<int32_t>(kGatherSeedCount);
        for (int32_t i = 0; i < seedCount; ++i) {
            gatherOffsets.SetValue(i, static_cast<int32_t>(outIdx + i * num_));
        }

        int32_t filled = seedCount;
        while (filled < count) {
            const int32_t chunk = ((count - filled) < filled) ? (count - filled) : filled;
            Adds(gatherOffsets[filled], gatherOffsets, filled * static_cast<int32_t>(num_), chunk);
            filled += chunk;
            PipeBarrier<PIPE_V>();
        }
        Muls(gatherOffsets, gatherOffsets, static_cast<int32_t>(sizeof(T)), count);
        PipeBarrier<PIPE_V>();
    }


    __aicore__ inline void PrepareStridedGatherOffsets(int64_t baseOffset, int32_t stride, int32_t count) {
        LocalTensor<int32_t> gatherOffsets = indexBuf_.Get<int32_t>();
        const int32_t seedCount = count < kGatherSeedCount ? count : static_cast<int32_t>(kGatherSeedCount);
        for (int32_t i = 0; i < seedCount; ++i) {
            gatherOffsets.SetValue(i, static_cast<int32_t>(baseOffset) + i * stride);
        }

        int32_t filled = seedCount;
        while (filled < count) {
            const int32_t chunk = ((count - filled) < filled) ? (count - filled) : filled;
            Adds(gatherOffsets[filled], gatherOffsets, filled * stride, chunk);
            filled += chunk;
            PipeBarrier<PIPE_V>();
        }
        Muls(gatherOffsets, gatherOffsets, static_cast<int32_t>(sizeof(T)), count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PrepareSmallInnerFloat32GatherOffsets(int64_t outIdx, int64_t rowCount, int64_t elemCount) {
        LocalTensor<int32_t> gatherOffsets = indexBuf_.Get<int32_t>();
        int32_t writeIdx = 0;
        for (int64_t row = 0; row < rowCount && writeIdx < elemCount; ++row) {
            const int32_t srcBase = static_cast<int32_t>((row * num_ + outIdx) * inner_);
            for (int64_t j = 0; j < inner_ && writeIdx < elemCount; ++j) {
                gatherOffsets.SetValue(writeIdx++, srcBase + static_cast<int32_t>(j));
            }
        }
        Muls(gatherOffsets, gatherOffsets, static_cast<int32_t>(sizeof(T)), static_cast<int32_t>(elemCount));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessSmallInnerFloat16PadRows(int64_t startTask, int64_t endTask) {
        const int64_t rowsPerTask = GetSmallInnerRowsPerTask();
        for (int64_t taskId = startTask; taskId < endTask; ++taskId) {
            const int64_t outIdx = taskId % num_;
            const int64_t outerBegin = (taskId / num_) * rowsPerTask;
            if (outerBegin >= outer_) {
                continue;
            }
            int64_t rowCount = outer_ - outerBegin;
            if (rowCount > rowsPerTask) {
                rowCount = rowsPerTask;
            }
            CopySmallInnerFloat16PadRows(outIdx, outerBegin, rowCount);
        }
    }

    __aicore__ inline void CopySmallInnerFloat16PadRows(int64_t outIdx, int64_t outerBegin, int64_t rowCount) {
        __gm__ T *outAddr = outputList_.GetDataPtr<__gm__ T>(static_cast<uint32_t>(outIdx));
        if (outAddr == nullptr) {
            return;
        }

        const uint32_t rowBytes = static_cast<uint32_t>(inner_ * sizeof(T));
        const uint32_t srcStrideBytes = static_cast<uint32_t>((num_ - 1) * inner_ * sizeof(T));
        const DataCopyExtParams gmToUbParams = {static_cast<uint16_t>(rowCount), rowBytes, srcStrideBytes, 0, 0};
        const DataCopyExtParams ubToGmParams = {static_cast<uint16_t>(rowCount), rowBytes, 0, 0, 0};
        const DataCopyPadExtParams<T> padParams = {false, 0, 0, static_cast<T>(0)};
        const int64_t inputBase = (outerBegin * num_ + outIdx) * inner_;
        const int64_t outputBase = outerBegin * inner_;

        LocalTensor<T> rowLocal = boundQueue_.AllocTensor<T>();
        DataCopyPad(rowLocal, inputGm_[inputBase], gmToUbParams, padParams);
        boundQueue_.EnQue(rowLocal);

        rowLocal = boundQueue_.DeQue<T>();
        GlobalTensor<T> outputGm;
        outputGm.SetGlobalBuffer(outAddr, static_cast<uint32_t>(outer_ * inner_));
        DataCopyPad(outputGm[outputBase], rowLocal, ubToGmParams);
        boundQueue_.FreeTensor(rowLocal);
    }

    __aicore__ inline void ProcessSmallInnerFloat32PackedRows(int64_t startTask, int64_t endTask) {
        const int64_t rowsPerTask = tileElems_ >= inner_ ? (tileElems_ / inner_) : 1;
        for (int64_t taskId = startTask; taskId < endTask; ++taskId) {
            const int64_t outerBegin = taskId * rowsPerTask;
            if (outerBegin >= outer_) {
                break;
            }
            int64_t rowCount = outer_ - outerBegin;
            if (rowCount > rowsPerTask) {
                rowCount = rowsPerTask;
            }
            CopySmallInnerFloat32PackedRows(outerBegin, rowCount);
        }
    }

    __aicore__ inline void CopySmallInnerFloat32PackedRows(int64_t outerBegin, int64_t rowCount) {
        const int64_t inputBase = outerBegin * num_ * inner_;
        const int64_t packedElems = rowCount * num_ * inner_;
        const uint32_t inputBytes = static_cast<uint32_t>(packedElems * sizeof(T));
        const uint32_t outputElems = static_cast<uint32_t>(rowCount * inner_);

        LocalTensor<T> inputLocal = inQueue_.AllocTensor<T>();
        if ((inputBytes % kAlignBytes) == 0) {
            DataCopy(inputLocal, inputGm_[inputBase], static_cast<uint32_t>(packedElems));
        } else {
            DataCopyExtParams gmToUbParams = {1, inputBytes, 0, 0, 0};
            DataCopyPadExtParams<T> padParams = {false, 0, 0, static_cast<T>(0)};
            DataCopyPad(inputLocal, inputGm_[inputBase], gmToUbParams, padParams);
        }
        inQueue_.EnQue(inputLocal);

        inputLocal = inQueue_.DeQue<T>();
        for (int64_t outIdx = 0; outIdx < num_; ++outIdx) {
            __gm__ T *outAddr = outputList_.GetDataPtr<__gm__ T>(static_cast<uint32_t>(outIdx));
            if (outAddr == nullptr) {
                continue;
            }

            GlobalTensor<T> outputGm;
            outputGm.SetGlobalBuffer(outAddr, static_cast<uint32_t>(outer_ * inner_));
            LocalTensor<T> outputLocal = outQueue_.AllocTensor<T>();
            int64_t alignedElems = outputElems;
            if (alignedElems >= kGatherSeedCount) {
                alignedElems = (alignedElems / kGatherSeedCount) * kGatherSeedCount;
            } else {
                alignedElems = 0;
            }
            if (alignedElems > 0) {
                if (inner_ == 1) {
                    PrepareStridedGatherOffsets(outIdx, static_cast<int32_t>(num_), static_cast<int32_t>(alignedElems));
                } else {
                    PrepareSmallInnerFloat32GatherOffsets(outIdx, rowCount, alignedElems);
                }
                LocalTensor<uint32_t> gatherOffsets = indexBuf_.Get<uint32_t>();
                Gather(outputLocal, inputLocal, gatherOffsets, 0, static_cast<uint32_t>(alignedElems));
                PipeBarrier<PIPE_V>();
            }
            for (int64_t elem = alignedElems; elem < outputElems; ++elem) {
                const int64_t row = elem / inner_;
                const int64_t j = elem - row * inner_;
                const int64_t srcBase = (row * num_ + outIdx) * inner_;
                outputLocal.SetValue(elem, inputLocal.GetValue(srcBase + j));
            }

            const int64_t outputBase = outerBegin * inner_;
            const uint32_t outputBytes = outputElems * sizeof(T);
            if ((outputBytes % kAlignBytes) == 0) {
                DataCopy(outputGm[outputBase], outputLocal, outputElems);
            } else {
                DataCopyExtParams ubToGmParams = {1, outputBytes, 0, 0, 0};
                DataCopyPad(outputGm[outputBase], outputLocal, ubToGmParams);
            }
            outQueue_.FreeTensor(outputLocal);
        }
        inQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void ProcessWideNonAlignedBfloat16Rows(int64_t startTask, int64_t endTask) {
        const int64_t rowsPerTask = GetWideNonAlignedRowsPerTask();
        int64_t currentOutIdx = 0;
        int64_t currentOuterBegin = 0;
        int64_t currentRowCount = 0;
        if (!DecodeWideRowsTask(startTask, rowsPerTask, currentOutIdx, currentOuterBegin, currentRowCount)) {
            return;
        }
        LoadWideNonAlignedBfloat16Rows(currentOutIdx, currentOuterBegin, currentRowCount);
        for (int64_t taskId = startTask + 1; taskId < endTask; ++taskId) {
            int64_t nextOutIdx = 0;
            int64_t nextOuterBegin = 0;
            int64_t nextRowCount = 0;
            if (!DecodeWideRowsTask(taskId, rowsPerTask, nextOutIdx, nextOuterBegin, nextRowCount)) {
                continue;
            }
            LoadWideNonAlignedBfloat16Rows(nextOutIdx, nextOuterBegin, nextRowCount);
            StoreWideNonAlignedBfloat16Rows(currentOutIdx, currentOuterBegin, currentRowCount);
            currentOutIdx = nextOutIdx;
            currentOuterBegin = nextOuterBegin;
            currentRowCount = nextRowCount;
        }
        StoreWideNonAlignedBfloat16Rows(currentOutIdx, currentOuterBegin, currentRowCount);
    }

    __aicore__ inline bool DecodeWideRowsTask(int64_t taskId, int64_t rowsPerTask, int64_t &outIdx,
                                              int64_t &outerBegin, int64_t &rowCount) const {
        outIdx = taskId % num_;
        outerBegin = (taskId / num_) * rowsPerTask;
        if (outerBegin >= outer_) {
            return false;
        }
        rowCount = outer_ - outerBegin;
        if (rowCount > rowsPerTask) {
            rowCount = rowsPerTask;
        }
        return rowCount > 0;
    }

    __aicore__ inline void LoadWideNonAlignedBfloat16Rows(int64_t outIdx, int64_t outerBegin, int64_t rowCount) {
        const uint32_t rowBytes = static_cast<uint32_t>(inner_ * sizeof(T));
        const uint32_t srcStrideBytes = static_cast<uint32_t>((num_ - 1) * inner_ * sizeof(T));
        const DataCopyExtParams gmToUbParams = {static_cast<uint16_t>(rowCount), rowBytes, srcStrideBytes, 0, 0};
        const DataCopyPadExtParams<T> padParams = {false, 0, 0, static_cast<T>(0)};
        const int64_t inputBase = (outerBegin * num_ + outIdx) * inner_;

        LocalTensor<T> inputLocal = boundQueue_.AllocTensor<T>();
        DataCopyPad(inputLocal, inputGm_[inputBase], gmToUbParams, padParams);
        boundQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void StoreWideNonAlignedBfloat16Rows(int64_t outIdx, int64_t outerBegin, int64_t rowCount) {
        const uint32_t rowBytes = static_cast<uint32_t>(inner_ * sizeof(T));
        const DataCopyExtParams ubToGmParams = {static_cast<uint16_t>(rowCount), rowBytes, 0, 0, 0};
        const int64_t outputBase = outerBegin * inner_;

        LocalTensor<T> inputLocal = boundQueue_.DeQue<T>();
        __gm__ T *outAddr = outputList_.GetDataPtr<__gm__ T>(static_cast<uint32_t>(outIdx));
        if (outAddr != nullptr) {
            GlobalTensor<T> outputGm;
            outputGm.SetGlobalBuffer(outAddr, static_cast<uint32_t>(outer_ * inner_));
            DataCopyPad(outputGm[outputBase], inputLocal, ubToGmParams);
        }
        boundQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void ProcessWideAlignedBfloat16BoundRows(int64_t startTask, int64_t endTask) {
        const int64_t rowsPerTask = tileElems_ >= inner_ ? (tileElems_ / inner_) : 1;
        for (int64_t taskId = startTask; taskId < endTask; ++taskId) {
            const int64_t outIdx = taskId % num_;
            const int64_t outerBegin = (taskId / num_) * rowsPerTask;
            if (outerBegin >= outer_) {
                continue;
            }
            int64_t rowCount = outer_ - outerBegin;
            if (rowCount > rowsPerTask) {
                rowCount = rowsPerTask;
            }
            CopyWideAlignedBfloat16BoundRows(outIdx, outerBegin, rowCount);
        }
    }

    __aicore__ inline void CopyWideAlignedBfloat16BoundRows(int64_t outIdx, int64_t outerBegin, int64_t rowCount) {
        __gm__ T *outAddr = outputList_.GetDataPtr<__gm__ T>(static_cast<uint32_t>(outIdx));
        if (outAddr == nullptr) {
            return;
        }

        const uint64_t rowBytes = static_cast<uint64_t>(inner_) * sizeof(T);
        const uint16_t blockLen = static_cast<uint16_t>(rowBytes / kAlignBytes);
        const uint16_t srcGap = static_cast<uint16_t>(((num_ - 1) * rowBytes) / kAlignBytes);
        const DataCopyParams gmToUbParams = {static_cast<uint16_t>(rowCount), blockLen, srcGap, 0};
        const DataCopyParams ubToGmParams = {static_cast<uint16_t>(rowCount), blockLen, 0, 0};

        GlobalTensor<T> outputGm;
        outputGm.SetGlobalBuffer(outAddr, static_cast<uint32_t>(outer_ * inner_));
        const int64_t inputBase = (outerBegin * num_ + outIdx) * inner_;
        const int64_t outputBase = outerBegin * inner_;

        LocalTensor<T> rowLocal = boundQueue_.AllocTensor<T>();
        DataCopy(rowLocal, inputGm_[inputBase], gmToUbParams);
        boundQueue_.EnQue(rowLocal);

        rowLocal = boundQueue_.DeQue<T>();
        DataCopy(outputGm[outputBase], rowLocal, ubToGmParams);
        boundQueue_.FreeTensor(rowLocal);
    }

    __aicore__ inline void ProcessLargeAlignedFloatOutputSlabs(int64_t startTask, int64_t endTask) {
        const int64_t slabOutputs = GetLargeAlignedFloatSlabOutputs();
        const int64_t rowsPerTask = GetLargeAlignedFloatRowsPerTask();
        const int64_t outGroups = (num_ + slabOutputs - 1) / slabOutputs;
        for (int64_t taskId = startTask; taskId < endTask; ++taskId) {
            const int64_t outerBegin = (taskId / outGroups) * rowsPerTask;
            if (outerBegin >= outer_) {
                break;
            }
            const int64_t outGroup = taskId % outGroups;
            int64_t rowCount = outer_ - outerBegin;
            if (rowCount > rowsPerTask) {
                rowCount = rowsPerTask;
            }
            CopyLargeAlignedFloatOutputSlab(outerBegin, rowCount, outGroup * slabOutputs, slabOutputs);
        }
    }

    __aicore__ inline void CopyLargeAlignedFloatOutputSlab(int64_t outerBegin, int64_t rowCount,
                                                           int64_t outBlockBegin, int64_t slabOutputs) {
        if (outBlockBegin >= num_) {
            return;
        }
        int64_t blockOutputs = num_ - outBlockBegin;
        if (blockOutputs > slabOutputs) {
            blockOutputs = slabOutputs;
        }
        const int64_t inputBase = (outerBegin * num_ + outBlockBegin) * inner_;
        const uint32_t groupBytes = static_cast<uint32_t>(blockOutputs * inner_ * sizeof(T));
        const uint32_t inputSrcStride = static_cast<uint32_t>((num_ - blockOutputs) * inner_ * sizeof(T));

        LocalTensor<T> inputLocal = inQueue_.AllocTensor<T>();
        DataCopyExtParams gmToUbParams = {static_cast<uint16_t>(rowCount), groupBytes, inputSrcStride, 0, 0};
        DataCopyPadExtParams<T> padParams = {false, 0, 0, static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm_[inputBase], gmToUbParams, padParams);
        inQueue_.EnQue(inputLocal);

        inputLocal = inQueue_.DeQue<T>();
        const uint16_t innerBlocks = static_cast<uint16_t>((inner_ * sizeof(T)) / kAlignBytes);
        const uint16_t outputSrcGap = static_cast<uint16_t>(((blockOutputs - 1) * inner_ * sizeof(T)) / kAlignBytes);
        for (int64_t localOut = 0; localOut < blockOutputs; ++localOut) {
            __gm__ T *outAddr = outputList_.GetDataPtr<__gm__ T>(static_cast<uint32_t>(outBlockBegin + localOut));
            if (outAddr == nullptr) {
                continue;
            }
            GlobalTensor<T> outputGm;
            outputGm.SetGlobalBuffer(outAddr, static_cast<uint32_t>(outer_ * inner_));
            const DataCopyParams ubToGmParams = {static_cast<uint16_t>(rowCount), innerBlocks, outputSrcGap, 0};
            DataCopy(outputGm[outerBegin * inner_], inputLocal[localOut * inner_], ubToGmParams);
        }
        PipeBarrier<PIPE_MTE3>();
        inQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline bool CanUseRowBatchCopy() const {
        if constexpr (std::is_same_v<T, bool>) {
            return false;
        }
        if (inner_ <= 0 || tileElems_ < inner_) {
            return false;
        }
        const uint64_t rowBytes = static_cast<uint64_t>(inner_) * sizeof(T);
        const uint64_t srcStrideBytes = static_cast<uint64_t>(num_) * rowBytes;
        const uint64_t rowBlocks = rowBytes / kAlignBytes;
        const uint64_t srcGapBlocks = (static_cast<uint64_t>(num_) - 1ULL) * rowBlocks;
        return rowBytes >= kAlignBytes && (rowBytes % kAlignBytes) == 0 &&
               (srcStrideBytes % kAlignBytes) == 0 &&
               rowBlocks <= static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) &&
               srcGapBlocks <= static_cast<uint64_t>(std::numeric_limits<uint16_t>::max());
    }

    __aicore__ inline bool UseDirectRowBatch() const {
        if constexpr (std::is_same_v<T, bool>) {
            return false;
        }
        return directRowBatch_;
    }

    __aicore__ inline void CopyRowBatchDirect(GlobalTensor<T> &outputGm, int64_t outIdx, int64_t outerBegin,
                                              int64_t rowCount) {
        const uint64_t rowBytes = static_cast<uint64_t>(inner_) * sizeof(T);
        const uint16_t blockLen = static_cast<uint16_t>(rowBytes / kAlignBytes);
        const uint16_t srcGap = static_cast<uint16_t>(((num_ - 1) * rowBytes) / kAlignBytes);
        const DataCopyParams gmToUbParams = {static_cast<uint16_t>(rowCount), blockLen, srcGap, 0};
        const DataCopyParams ubToGmParams = {static_cast<uint16_t>(rowCount), blockLen, 0, 0};

        const int64_t inputBase = (outerBegin * num_ + outIdx) * inner_;
        const int64_t outputBase = outerBegin * inner_;

        LocalTensor<T> inputLocal = inQueue_.AllocTensor<T>();
        DataCopy(inputLocal, inputGm_[inputBase], gmToUbParams);
        inQueue_.EnQue(inputLocal);

        inputLocal = inQueue_.DeQue<T>();
        DataCopy(outputGm[outputBase], inputLocal, ubToGmParams);
        inQueue_.FreeTensor(inputLocal);
    }
    __aicore__ inline void CopyRowBatch(GlobalTensor<T> &outputGm, int64_t outIdx, int64_t outerBegin, int64_t rowCount) {
        const int64_t batchElems = rowCount * inner_;
        const uint64_t rowBytes = static_cast<uint64_t>(inner_) * sizeof(T);
        const uint16_t blockLen = static_cast<uint16_t>(rowBytes / kAlignBytes);
        const uint16_t srcGap = static_cast<uint16_t>(((num_ - 1) * rowBytes) / kAlignBytes);
        const DataCopyParams gmToUbParams = {static_cast<uint16_t>(rowCount), blockLen, srcGap, 0};
        const DataCopyParams ubToGmParams = {static_cast<uint16_t>(rowCount), blockLen, 0, 0};

        const int64_t inputBase = (outerBegin * num_ + outIdx) * inner_;
        const int64_t outputBase = outerBegin * inner_;

        LocalTensor<T> inputLocal = inQueue_.AllocTensor<T>();
        DataCopy(inputLocal, inputGm_[inputBase], gmToUbParams);
        inQueue_.EnQue(inputLocal);

        inputLocal = inQueue_.DeQue<T>();
        LocalTensor<T> outputLocal = outQueue_.AllocTensor<T>();
        CopyToOutput(outputLocal, inputLocal, batchElems);
        outQueue_.EnQue(outputLocal);
        inQueue_.FreeTensor(inputLocal);

        outputLocal = outQueue_.DeQue<T>();
        DataCopy(outputGm[outputBase], outputLocal, ubToGmParams);
        outQueue_.FreeTensor(outputLocal);
    }

    __aicore__ inline void CopyTile(GlobalTensor<T> &outputGm, int64_t inputOffset, int64_t outputOffset,
                                    int64_t elemCount) {
        const uint32_t copyBytes = static_cast<uint32_t>(elemCount * sizeof(T));
        if (copyBytes < kAlignBytes) {
            CopyTileScalar(outputGm, inputOffset, outputOffset, elemCount);
            return;
        }

        LocalTensor<T> inputLocal = inQueue_.AllocTensor<T>();
        if ((copyBytes % kAlignBytes) == 0) {
            DataCopy(inputLocal, inputGm_[inputOffset], static_cast<uint32_t>(elemCount));
        } else {
            DataCopyExtParams gmToUbParams = {1, copyBytes, 0, 0, 0};
            DataCopyPadExtParams<T> padParams = {false, 0, 0, static_cast<T>(0)};
            DataCopyPad(inputLocal, inputGm_[inputOffset], gmToUbParams, padParams);
        }
        inQueue_.EnQue(inputLocal);

        inputLocal = inQueue_.DeQue<T>();
        LocalTensor<T> outputLocal = outQueue_.AllocTensor<T>();
        CopyToOutput(outputLocal, inputLocal, elemCount);
        outQueue_.EnQue(outputLocal);
        inQueue_.FreeTensor(inputLocal);

        outputLocal = outQueue_.DeQue<T>();
        if ((copyBytes % kAlignBytes) == 0) {
            DataCopy(outputGm[outputOffset], outputLocal, static_cast<uint32_t>(elemCount));
        } else {
            DataCopyExtParams ubToGmParams = {1, copyBytes, 0, 0, 0};
            DataCopyPad(outputGm[outputOffset], outputLocal, ubToGmParams);
        }
        outQueue_.FreeTensor(outputLocal);
    }

    __aicore__ inline void CopyTileScalar(GlobalTensor<T> &outputGm, int64_t inputOffset, int64_t outputOffset,
                                          int64_t elemCount) {
        for (int64_t i = 0; i < elemCount; ++i) {
            outputGm.SetValue(outputOffset + i, inputGm_.GetValue(inputOffset + i));
        }
    }

    __aicore__ inline void CopyToOutput(const LocalTensor<T> &dst, const LocalTensor<T> &src, int64_t elemCount) {
        const uint32_t copyBytes = static_cast<uint32_t>(elemCount * sizeof(T));
        if constexpr (!std::is_same_v<T, bool>) {
            if (copyBytes >= kAlignBytes) {
                const uint32_t alignedBytes = ((copyBytes + kAlignBytes - 1) / kAlignBytes) * kAlignBytes;
                const uint32_t alignedElems = alignedBytes / sizeof(T);
                DataCopy(dst, src, alignedElems);
                return;
            }
        }
        for (int64_t i = 0; i < elemCount; ++i) {
            dst.SetValue(i, src.GetValue(i));
        }
    }

  private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, kBufferNum> inQueue_;
    TQue<QuePosition::VECOUT, kBufferNum> outQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, kBufferNum> boundQueue_;
    TBuf<QuePosition::VECCALC> indexBuf_;
    GlobalTensor<T> inputGm_;
    AscendC::ListTensorDesc outputList_;
    int64_t num_ = 0;
    int64_t outer_ = 0;
    int64_t inner_ = 0;
    int64_t taskMode_ = 0;
    int64_t totalTasks_ = 0;
    int64_t tasksPerCore_ = 0;
    int64_t tileElems_ = 0;
    bool directRowBatch_ = false;
};

} // namespace

extern "C" __global__ __aicore__ void unpack(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelUnpack<DTYPE_X> op;
    op.Init(x, y, tiling_data);
    op.Process();
}
