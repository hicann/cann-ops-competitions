#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

#include <limits>
#include <type_traits>

using namespace AscendC;

namespace {

constexpr uint32_t kBufferNum = 2;
constexpr uint32_t kAlignBytes = 32;
constexpr uint32_t kMtePreferredAlignBytes = 512;
constexpr int64_t kGatherSeedCount = 8;
constexpr int64_t kRouteLinear = 0;
constexpr int64_t kRouteOuterFan = 1;
constexpr int64_t kRouteInt32Needle = 2;
constexpr int64_t kRouteFloatTiny = 3;
constexpr int64_t kRouteBf16RaggedRelay = 4;
constexpr int64_t kRouteFloatSlab = 5;
constexpr int64_t kRouteBf16AlignedRelay = 6;
constexpr int64_t kRouteFloatSingleRowBalance = 7;
constexpr int64_t kRouteFloatSingleRowBodyTail = 8;

template <typename T>
class UnpackFlowCore {
  public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const UnpackTilingData &tiling) {
        splitCount_ = tiling.splitCount;
        frontSpan_ = tiling.frontSpan;
        laneWidth_ = tiling.laneWidth;
        route_ = tiling.route;
        ticketCount_ = tiling.ticketCount;
        ticketsPerCore_ = tiling.ticketsPerCore;
        stageElems_ = tiling.stageElems;
        bodyTailCores_ = tiling.bodyTailCores;
        directLaneRelay_ = (tiling.directLaneRelay != 0);

        srcHub_.SetGlobalBuffer((__gm__ T *)x, static_cast<uint32_t>(tiling.frontSpan * tiling.splitCount * tiling.laneWidth));
        dstRoster_.Init((__gm__ void *)y);
        const bool useFloatOutputSlab = (route_ == kRouteFloatSlab);
        const bool useWideBfloat16BoundRows =
            (route_ == kRouteBf16AlignedRelay ||
             route_ == kRouteBf16RaggedRelay);
        const bool useFloatSingleRowBodyTail = (route_ == kRouteFloatSingleRowBodyTail);
        const bool useBoundQueue = useWideBfloat16BoundRows || useFloatSingleRowBodyTail;
        const uint32_t inputBufferCount = useFloatOutputSlab ? 1U : kBufferNum;
        const uint32_t outputBufferCount = useFloatOutputSlab ? 1U : kBufferNum;
        const uint32_t inputBufferElems =
            (route_ == kRouteInt32Needle || route_ == kRouteFloatTiny)
                ? static_cast<uint32_t>(stageElems_ * splitCount_)
                : (route_ == kRouteBf16RaggedRelay)
                      ? static_cast<uint32_t>(Bf16RaggedPaddedElems() * Bf16RaggedRowsPerTicket())
                : (route_ == kRouteBf16AlignedRelay)
                      ? static_cast<uint32_t>(stageElems_)
                : static_cast<uint32_t>(stageElems_);
        if (useBoundQueue) {
            fabric_.InitBuffer(relayPipe_, kBufferNum, inputBufferElems * sizeof(T));
            if (useFloatSingleRowBodyTail) {
                const uint32_t bodyBytes = static_cast<uint32_t>(AlignedBodyElems(tiling.laneWidth) * sizeof(T));
                fabric_.InitBuffer(manualBodyBuf0_, bodyBytes);
                fabric_.InitBuffer(manualBodyBuf1_, bodyBytes);
            }
        } else {
            fabric_.InitBuffer(readPipe_, inputBufferCount, inputBufferElems * sizeof(T));
        }
        if (!directLaneRelay_ && route_ != kRouteBf16RaggedRelay &&
            route_ != kRouteBf16AlignedRelay &&
            route_ != kRouteFloatSingleRowBodyTail &&
            route_ != kRouteFloatSlab) {
            const uint32_t outputBufferElems =
                (route_ == kRouteBf16RaggedRelay)
                    ? static_cast<uint32_t>(Bf16RaggedPaddedElems() * Bf16RaggedRowsPerTicket())
                : (route_ == kRouteFloatSlab)
                    ? static_cast<uint32_t>(FloatSlabRowsPerTicket() * laneWidth_)
                : static_cast<uint32_t>(stageElems_);
            fabric_.InitBuffer(writePipe_, outputBufferCount, outputBufferElems * sizeof(T));
        }
        if (route_ == kRouteInt32Needle) {
            fabric_.InitBuffer(offsetScratch_, static_cast<uint32_t>(stageElems_ * 2 * sizeof(int32_t)));
        } else if (route_ == kRouteFloatTiny) {
            fabric_.InitBuffer(offsetScratch_, static_cast<uint32_t>(stageElems_ * sizeof(int32_t)));
        }
    }

    __aicore__ inline void Run() {
        if (splitCount_ <= 0 || frontSpan_ <= 0 || laneWidth_ <= 0 || ticketCount_ <= 0 || stageElems_ <= 0) {
            return;
        }

        if (route_ == kRouteFloatSingleRowBalance && IsFloatSingleRowBalanceRoute()) {
            RunFloatSingleRowBalance();
            return;
        }
        if (route_ == kRouteFloatSingleRowBodyTail && IsFloatSingleRowBodyTailRoute()) {
            RunFloatSingleRowBodyTail();
            return;
        }
        int64_t startTask = 0;
        int64_t endTask = 0;
        if (!ClaimTicketSpan(startTask, endTask)) {
            return;
        }

        if (route_ == kRouteInt32Needle && IsInt32NeedleGatherRoute()) {
            RunInt32NeedleGather(startTask, endTask);
            return;
        }
        if (route_ == kRouteFloatTiny && IsFloatTinyTailGatherRoute()) {
            RunFloatTinyTailGather(startTask, endTask);
            return;
        }
        if (route_ == kRouteBf16RaggedRelay && IsBf16RaggedRelayRoute()) {
            RunBf16RaggedRelay(startTask, endTask);
            return;
        }
        if (route_ == kRouteBf16AlignedRelay && IsBf16AlignedRelayRoute()) {
            RunBf16AlignedRelay(startTask, endTask);
            return;
        }
        if (route_ == kRouteFloatSlab && IsFloatSlabFanoutRoute()) {
            RunFloatSlabFanout(startTask, endTask);
            return;
        }
        for (int64_t taskId = startTask; taskId < endTask; ++taskId) {
            DrainLinearTicket(taskId);
        }
    }

  private:
    struct LaneTicket {
        int64_t split = 0;
        int64_t rowBegin = 0;
        int64_t rowEnd = 0;
    };

    struct WindowPlan {
        uint32_t bytes = 0;
        uint32_t headElems = 0;
        uint32_t tailBytes = 0;
        uint32_t tailAlignedElems = 0;
        int64_t tailElems = 0;
        bool split512 = false;
        bool blockAligned = false;
    };

    __aicore__ inline bool DecodeLinearTicket(int64_t taskId, LaneTicket &ticket) const {
        ticket.split = taskId;
        ticket.rowBegin = 0;
        ticket.rowEnd = frontSpan_;
        if (route_ == kRouteOuterFan) {
            const int64_t rowsPerTicket = stageElems_ >= laneWidth_ ? (stageElems_ / laneWidth_) : 1;
            ticket.split = taskId % splitCount_;
            ticket.rowBegin = (taskId / splitCount_) * rowsPerTicket;
            ticket.rowEnd = ticket.rowBegin + rowsPerTicket;
            if (ticket.rowEnd > frontSpan_) {
                ticket.rowEnd = frontSpan_;
            }
        }
        return ticket.split >= 0 && ticket.split < splitCount_ && ticket.rowBegin < ticket.rowEnd;
    }

    __aicore__ inline void DrainLinearTicket(int64_t taskId) {
        LaneTicket ticket;
        if (!DecodeLinearTicket(taskId, ticket)) {
            return;
        }

        __gm__ T *outAddr = dstRoster_.GetDataPtr<__gm__ T>(static_cast<uint32_t>(ticket.split));
        if (outAddr == nullptr) {
            return;
        }

        GlobalTensor<T> outputGm;
        outputGm.SetGlobalBuffer(outAddr, static_cast<uint32_t>(frontSpan_ * laneWidth_));
        if (IsStridedMteRoute()) {
            DrainStridedRows(outputGm, ticket);
        } else {
            DrainScalarWindows(outputGm, ticket);
        }
    }

    __aicore__ inline void DrainStridedRows(GlobalTensor<T> &outputGm, const LaneTicket &ticket) {
        const int64_t rowsPerBatch = stageElems_ / laneWidth_;
        for (int64_t row = ticket.rowBegin; row < ticket.rowEnd; row += rowsPerBatch) {
            int64_t currentRows = ticket.rowEnd - row;
            if (currentRows > rowsPerBatch) {
                currentRows = rowsPerBatch;
            }
            if (UseMteRelayRows()) {
                RelayRowsNoVecCopy(outputGm, ticket.split, row, currentRows);
            } else {
                RelayRowsViaVecOut(outputGm, ticket.split, row, currentRows);
            }
        }
    }

    __aicore__ inline void DrainScalarWindows(GlobalTensor<T> &outputGm, const LaneTicket &ticket) {
        for (int64_t row = ticket.rowBegin; row < ticket.rowEnd; ++row) {
            const int64_t inputBase = (row * splitCount_ + ticket.split) * laneWidth_;
            const int64_t outputBase = row * laneWidth_;
            for (int64_t offset = 0; offset < laneWidth_; offset += stageElems_) {
                int64_t currentElems = laneWidth_ - offset;
                if (currentElems > stageElems_) {
                    currentElems = stageElems_;
                }
                StageWindow(outputGm, inputBase + offset, outputBase + offset, currentElems);
            }
        }
    }

    __aicore__ inline bool PrefersPlainMte(uint64_t bytes) const {
        return bytes <= kMtePreferredAlignBytes || (bytes % kMtePreferredAlignBytes) == 0;
    }

    __aicore__ inline bool NeedsMte512Split(uint32_t copyBytes) const {
        return copyBytes > kMtePreferredAlignBytes && (copyBytes % kMtePreferredAlignBytes) != 0;
    }

    __aicore__ inline bool ClaimTicketSpan(int64_t &startTask, int64_t &endTask) const {
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        if (blockNum <= 0) {
            return false;
        }
        const int64_t formerTasks = ticketCount_ / blockNum;
        const int64_t tailCoreCount = ticketCount_ % blockNum;
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        int64_t blockTaskCount = formerTasks;
        startTask = blockIdx * formerTasks;
        if (blockIdx < tailCoreCount) {
            blockTaskCount += 1;
            startTask += blockIdx;
        } else {
            startTask += tailCoreCount;
        }
        if (blockTaskCount <= 0 || startTask >= ticketCount_) {
            return false;
        }
        endTask = startTask + blockTaskCount;
        if (endTask > ticketCount_) {
            endTask = ticketCount_;
        }
        return true;
    }

    __aicore__ inline bool IsInt32NeedleGatherRoute() const {
        if constexpr (std::is_same_v<T, bool>) {
            return false;
        }
        return sizeof(T) == 4 && splitCount_ >= 33 && splitCount_ <= 256 && laneWidth_ == 1;
    }

    __aicore__ inline bool IsFloatTinyTailGatherRoute() const {
        if constexpr (std::is_same_v<T, bool>) {
            return false;
        }
        return sizeof(T) == 4 && splitCount_ >= 129 && laneWidth_ > 0 && laneWidth_ <= 4;
    }

    __aicore__ inline bool IsBf16RaggedRelayRoute() const {
        if constexpr (std::is_same_v<T, bool>) {
            return false;
        }
        return sizeof(T) == 2 && splitCount_ > 17 &&
               (((laneWidth_ >= 1024 && laneWidth_ < 2048) || laneWidth_ == 3073) && ((laneWidth_ % 16) != 0));
    }

    __aicore__ inline bool IsBf16AlignedRelayRoute() const {
        if constexpr (std::is_same_v<T, bool>) {
            return false;
        }
        return sizeof(T) == 2 && splitCount_ > 17 && laneWidth_ >= 1024 && laneWidth_ < 4096 && ((laneWidth_ % 16) == 0);
    }

    __aicore__ inline bool IsFloatSlabFanoutRoute() const {
        if constexpr (std::is_same_v<T, bool>) {
            return false;
        }
        if (!(sizeof(T) == 4 && splitCount_ == 255 && laneWidth_ >= 4096 && ((laneWidth_ % 8) == 0) &&
              FloatSlabWidth() > 1)) {
            return false;
        }
        if (laneWidth_ == 4096) {
            return frontSpan_ > 64;
        }
        return false;
    }

    __aicore__ inline bool IsFloatSingleRowBalanceRoute() const {
        if constexpr (std::is_same_v<T, bool>) {
            return false;
        }
        return sizeof(T) == 4 && frontSpan_ == 1 && splitCount_ == 255 &&
               laneWidth_ >= 8193 && laneWidth_ <= 8255 && ((laneWidth_ % 8) != 0);
    }

    __aicore__ inline bool IsFloatSingleRowBodyTailRoute() const {
        if constexpr (std::is_same_v<T, bool>) {
            return false;
        }
        const int64_t bodyElems = AlignedBodyElems(laneWidth_);
        const int64_t tailBytes = (laneWidth_ - bodyElems) * static_cast<int64_t>(sizeof(T));
        return sizeof(T) == 4 && frontSpan_ == 1 && splitCount_ == 255 &&
               laneWidth_ >= 8193 && laneWidth_ <= 8255 && ((laneWidth_ % 8) != 0) &&
               bodyTailCores_ > 0 && bodyTailCores_ < static_cast<int64_t>(GetBlockNum()) &&
               bodyElems > 0 && tailBytes >= static_cast<int64_t>(kAlignBytes) &&
               tailBytes < static_cast<int64_t>(kMtePreferredAlignBytes);
    }

    __aicore__ inline int64_t Bf16RaggedPaddedElems() const {
        const uint64_t rowBytes = static_cast<uint64_t>(laneWidth_) * sizeof(T);
        const uint64_t paddedRowBytes = ((rowBytes + kAlignBytes - 1ULL) / kAlignBytes) * kAlignBytes;
        return static_cast<int64_t>(paddedRowBytes / sizeof(T));
    }

    __aicore__ inline int64_t Bf16RaggedRowsPerTicket() const {
        return stageElems_ >= laneWidth_ ? (stageElems_ / laneWidth_) : 1;
    }

    __aicore__ inline int64_t FloatSlabWidth() const {
        if (laneWidth_ <= 4096) {
            return 4;
        }
        if (laneWidth_ <= 8192) {
            return 2;
        }
        return 2;
    }

    __aicore__ inline int64_t FloatSlabRowsPerTicket() const {
        if (laneWidth_ <= 8192) {
            return 2;
        }
        return 1;
    }

    __aicore__ inline void RunInt32NeedleGather(int64_t startTask, int64_t endTask) {
        const int64_t rowsPerTicket = stageElems_ > 0 ? stageElems_ : 1;
        for (int64_t taskId = startTask; taskId < endTask; ++taskId) {
            const int64_t outerBegin = taskId * rowsPerTicket;
            if (outerBegin >= frontSpan_) {
                break;
            }
            int64_t rowCount = frontSpan_ - outerBegin;
            if (rowCount > rowsPerTicket) {
                rowCount = rowsPerTicket;
            }
            GatherInt32Needles(outerBegin, rowCount);
        }
    }

    __aicore__ inline void GatherInt32Needles(int64_t outerBegin, int64_t rowCount) {
        const int64_t inputBase = outerBegin * splitCount_;
        const int64_t packedElems = rowCount * splitCount_;
        const uint32_t inputBytes = static_cast<uint32_t>(packedElems * sizeof(T));

        LocalTensor<T> inputLocal = readPipe_.AllocTensor<T>();
        if ((inputBytes % kAlignBytes) == 0) {
            DataCopy(inputLocal, srcHub_[inputBase], static_cast<uint32_t>(packedElems));
        } else {
            DataCopyExtParams gmToUbParams = {1, inputBytes, 0, 0, 0};
            DataCopyPadExtParams<T> padParams = {false, 0, 0, static_cast<T>(0)};
            DataCopyPad(inputLocal, srcHub_[inputBase], gmToUbParams, padParams);
        }
        readPipe_.EnQue(inputLocal);

        inputLocal = readPipe_.DeQue<T>();
        const int32_t alignedRows = (rowCount >= kGatherSeedCount)
                                        ? static_cast<int32_t>((rowCount / kGatherSeedCount) * kGatherSeedCount)
                                        : 0;
        LocalTensor<int32_t> currentGatherOffsetsI32;
        LocalTensor<int32_t> nextGatherOffsetsI32;
        LocalTensor<uint32_t> currentGatherOffsetsU32;
        LocalTensor<uint32_t> nextGatherOffsetsU32;
        if (alignedRows > 0) {
            SeedNeedleOffsets(0, alignedRows);
            currentGatherOffsetsI32 = offsetScratch_.Get<int32_t>();
            nextGatherOffsetsI32 = currentGatherOffsetsI32[alignedRows];
            currentGatherOffsetsU32 = offsetScratch_.Get<uint32_t>();
            nextGatherOffsetsU32 = currentGatherOffsetsU32[alignedRows];
        }
        for (int64_t outIdx = 0; outIdx < splitCount_; ++outIdx) {
            __gm__ T *outAddr = dstRoster_.GetDataPtr<__gm__ T>(static_cast<uint32_t>(outIdx));
            if (outAddr == nullptr) {
                if (alignedRows > 0 && (outIdx + 1) < splitCount_) {
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
            outputGm.SetGlobalBuffer(outAddr, static_cast<uint32_t>(frontSpan_));
            LocalTensor<T> outputLocal = writePipe_.AllocTensor<T>();
            if (alignedRows > 0) {
                Gather(outputLocal, inputLocal, currentGatherOffsetsU32, 0, static_cast<uint32_t>(alignedRows));
                PipeBarrier<PIPE_V>();
            }
            for (int64_t row = alignedRows; row < rowCount; ++row) {
                outputLocal.SetValue(row, inputLocal.GetValue(row * splitCount_ + outIdx));
            }

            const uint32_t outputBytes = static_cast<uint32_t>(rowCount * sizeof(T));
            if ((outputBytes % kAlignBytes) == 0) {
                DataCopy(outputGm[outerBegin], outputLocal, static_cast<uint32_t>(rowCount));
            } else {
                DataCopyExtParams ubToGmParams = {1, outputBytes, 0, 0, 0};
                DataCopyPad(outputGm[outerBegin], outputLocal, ubToGmParams);
            }
            writePipe_.FreeTensor(outputLocal);
            if (alignedRows > 0 && (outIdx + 1) < splitCount_) {
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
        readPipe_.FreeTensor(inputLocal);
    }

    __aicore__ inline void SeedNeedleOffsets(int64_t outIdx, int32_t count) {
        LocalTensor<int32_t> gatherOffsets = offsetScratch_.Get<int32_t>();
        const int32_t seedCount = count < kGatherSeedCount ? count : static_cast<int32_t>(kGatherSeedCount);
        for (int32_t i = 0; i < seedCount; ++i) {
            gatherOffsets.SetValue(i, static_cast<int32_t>(outIdx + i * splitCount_));
        }

        int32_t filled = seedCount;
        while (filled < count) {
            const int32_t chunk = ((count - filled) < filled) ? (count - filled) : filled;
            Adds(gatherOffsets[filled], gatherOffsets, filled * static_cast<int32_t>(splitCount_), chunk);
            filled += chunk;
            PipeBarrier<PIPE_V>();
        }
        Muls(gatherOffsets, gatherOffsets, static_cast<int32_t>(sizeof(T)), count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SeedStridedOffsets(int64_t baseOffset, int32_t stride, int32_t count) {
        LocalTensor<int32_t> gatherOffsets = offsetScratch_.Get<int32_t>();
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

    __aicore__ inline void SeedTinyTailOffsets(int64_t outIdx, int64_t rowCount, int64_t elemCount) {
        LocalTensor<int32_t> gatherOffsets = offsetScratch_.Get<int32_t>();
        int32_t writeIdx = 0;
        for (int64_t row = 0; row < rowCount && writeIdx < elemCount; ++row) {
            const int32_t srcBase = static_cast<int32_t>((row * splitCount_ + outIdx) * laneWidth_);
            for (int64_t j = 0; j < laneWidth_ && writeIdx < elemCount; ++j) {
                gatherOffsets.SetValue(writeIdx++, srcBase + static_cast<int32_t>(j));
            }
        }
        Muls(gatherOffsets, gatherOffsets, static_cast<int32_t>(sizeof(T)), static_cast<int32_t>(elemCount));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RunFloatTinyTailGather(int64_t startTask, int64_t endTask) {
        const int64_t rowsPerTicket = stageElems_ >= laneWidth_ ? (stageElems_ / laneWidth_) : 1;
        for (int64_t taskId = startTask; taskId < endTask; ++taskId) {
            const int64_t outerBegin = taskId * rowsPerTicket;
            if (outerBegin >= frontSpan_) {
                break;
            }
            int64_t rowCount = frontSpan_ - outerBegin;
            if (rowCount > rowsPerTicket) {
                rowCount = rowsPerTicket;
            }
            GatherFloatTinyTail(outerBegin, rowCount);
        }
    }

    __aicore__ inline void GatherFloatTinyTail(int64_t outerBegin, int64_t rowCount) {
        const int64_t inputBase = outerBegin * splitCount_ * laneWidth_;
        const int64_t packedElems = rowCount * splitCount_ * laneWidth_;
        const uint32_t inputBytes = static_cast<uint32_t>(packedElems * sizeof(T));
        const uint32_t outputElems = static_cast<uint32_t>(rowCount * laneWidth_);

        LocalTensor<T> inputLocal = readPipe_.AllocTensor<T>();
        if ((inputBytes % kAlignBytes) == 0) {
            DataCopy(inputLocal, srcHub_[inputBase], static_cast<uint32_t>(packedElems));
        } else {
            DataCopyExtParams gmToUbParams = {1, inputBytes, 0, 0, 0};
            DataCopyPadExtParams<T> padParams = {false, 0, 0, static_cast<T>(0)};
            DataCopyPad(inputLocal, srcHub_[inputBase], gmToUbParams, padParams);
        }
        readPipe_.EnQue(inputLocal);

        inputLocal = readPipe_.DeQue<T>();
        for (int64_t outIdx = 0; outIdx < splitCount_; ++outIdx) {
            __gm__ T *outAddr = dstRoster_.GetDataPtr<__gm__ T>(static_cast<uint32_t>(outIdx));
            if (outAddr == nullptr) {
                continue;
            }

            GlobalTensor<T> outputGm;
            outputGm.SetGlobalBuffer(outAddr, static_cast<uint32_t>(frontSpan_ * laneWidth_));
            LocalTensor<T> outputLocal = writePipe_.AllocTensor<T>();
            int64_t alignedElems = outputElems;
            if (alignedElems >= kGatherSeedCount) {
                alignedElems = (alignedElems / kGatherSeedCount) * kGatherSeedCount;
            } else {
                alignedElems = 0;
            }
            if (alignedElems > 0) {
                if (laneWidth_ == 1) {
                    SeedStridedOffsets(outIdx, static_cast<int32_t>(splitCount_), static_cast<int32_t>(alignedElems));
                } else {
                    SeedTinyTailOffsets(outIdx, rowCount, alignedElems);
                }
                LocalTensor<uint32_t> gatherOffsets = offsetScratch_.Get<uint32_t>();
                Gather(outputLocal, inputLocal, gatherOffsets, 0, static_cast<uint32_t>(alignedElems));
                PipeBarrier<PIPE_V>();
            }
            for (int64_t elem = alignedElems; elem < outputElems; ++elem) {
                const int64_t row = elem / laneWidth_;
                const int64_t j = elem - row * laneWidth_;
                const int64_t srcBase = (row * splitCount_ + outIdx) * laneWidth_;
                outputLocal.SetValue(elem, inputLocal.GetValue(srcBase + j));
            }

            const int64_t outputBase = outerBegin * laneWidth_;
            const uint32_t outputBytes = outputElems * sizeof(T);
            if ((outputBytes % kAlignBytes) == 0) {
                DataCopy(outputGm[outputBase], outputLocal, outputElems);
            } else {
                DataCopyExtParams ubToGmParams = {1, outputBytes, 0, 0, 0};
                DataCopyPad(outputGm[outputBase], outputLocal, ubToGmParams);
            }
            writePipe_.FreeTensor(outputLocal);
        }
        readPipe_.FreeTensor(inputLocal);
    }

    __aicore__ inline void RunBf16RaggedRelay(int64_t startTask, int64_t endTask) {
        const int64_t rowsPerTicket = Bf16RaggedRowsPerTicket();
        int64_t currentOutIdx = 0;
        int64_t currentOuterBegin = 0;
        int64_t currentRowCount = 0;
        if (!DecodeRelayTicket(startTask, rowsPerTicket, currentOutIdx, currentOuterBegin, currentRowCount)) {
            return;
        }
        PrimeBf16RaggedRelay(currentOutIdx, currentOuterBegin, currentRowCount);
        for (int64_t taskId = startTask + 1; taskId < endTask; ++taskId) {
            int64_t nextOutIdx = 0;
            int64_t nextOuterBegin = 0;
            int64_t nextRowCount = 0;
            if (!DecodeRelayTicket(taskId, rowsPerTicket, nextOutIdx, nextOuterBegin, nextRowCount)) {
                continue;
            }
            PrimeBf16RaggedRelay(nextOutIdx, nextOuterBegin, nextRowCount);
            DrainBf16RaggedRelay(currentOutIdx, currentOuterBegin, currentRowCount);
            currentOutIdx = nextOutIdx;
            currentOuterBegin = nextOuterBegin;
            currentRowCount = nextRowCount;
        }
        DrainBf16RaggedRelay(currentOutIdx, currentOuterBegin, currentRowCount);
    }

    __aicore__ inline bool DecodeRelayTicket(int64_t taskId, int64_t rowsPerTicket, int64_t &outIdx,
                                              int64_t &outerBegin, int64_t &rowCount) const {
        outIdx = taskId % splitCount_;
        outerBegin = (taskId / splitCount_) * rowsPerTicket;
        if (outerBegin >= frontSpan_) {
            return false;
        }
        rowCount = frontSpan_ - outerBegin;
        if (rowCount > rowsPerTicket) {
            rowCount = rowsPerTicket;
        }
        return rowCount > 0;
    }

    __aicore__ inline void PrimeBf16RaggedRelay(int64_t outIdx, int64_t outerBegin, int64_t rowCount) {
        const uint32_t rowBytes = static_cast<uint32_t>(laneWidth_ * sizeof(T));
        const uint32_t srcStrideBytes = static_cast<uint32_t>((splitCount_ - 1) * laneWidth_ * sizeof(T));
        const DataCopyExtParams gmToUbParams = {static_cast<uint16_t>(rowCount), rowBytes, srcStrideBytes, 0, 0};
        const DataCopyPadExtParams<T> padParams = {false, 0, 0, static_cast<T>(0)};
        const int64_t inputBase = (outerBegin * splitCount_ + outIdx) * laneWidth_;

        LocalTensor<T> inputLocal = relayPipe_.AllocTensor<T>();
        DataCopyPad(inputLocal, srcHub_[inputBase], gmToUbParams, padParams);
        relayPipe_.EnQue(inputLocal);
    }

    __aicore__ inline void DrainBf16RaggedRelay(int64_t outIdx, int64_t outerBegin, int64_t rowCount) {
        const uint32_t rowBytes = static_cast<uint32_t>(laneWidth_ * sizeof(T));
        const DataCopyExtParams ubToGmParams = {static_cast<uint16_t>(rowCount), rowBytes, 0, 0, 0};
        const int64_t outputBase = outerBegin * laneWidth_;

        LocalTensor<T> inputLocal = relayPipe_.DeQue<T>();
        __gm__ T *outAddr = dstRoster_.GetDataPtr<__gm__ T>(static_cast<uint32_t>(outIdx));
        if (outAddr != nullptr) {
            GlobalTensor<T> outputGm;
            outputGm.SetGlobalBuffer(outAddr, static_cast<uint32_t>(frontSpan_ * laneWidth_));
            DataCopyPad(outputGm[outputBase], inputLocal, ubToGmParams);
        }
        relayPipe_.FreeTensor(inputLocal);
    }

    __aicore__ inline void RunBf16AlignedRelay(int64_t startTask, int64_t endTask) {
        const int64_t rowsPerTicket = stageElems_ >= laneWidth_ ? (stageElems_ / laneWidth_) : 1;
        for (int64_t taskId = startTask; taskId < endTask; ++taskId) {
            const int64_t outIdx = taskId % splitCount_;
            const int64_t outerBegin = (taskId / splitCount_) * rowsPerTicket;
            if (outerBegin >= frontSpan_) {
                continue;
            }
            int64_t rowCount = frontSpan_ - outerBegin;
            if (rowCount > rowsPerTicket) {
                rowCount = rowsPerTicket;
            }
            RelayBf16AlignedRows(outIdx, outerBegin, rowCount);
        }
    }

    __aicore__ inline void RelayBf16AlignedRows(int64_t outIdx, int64_t outerBegin, int64_t rowCount) {
        __gm__ T *outAddr = dstRoster_.GetDataPtr<__gm__ T>(static_cast<uint32_t>(outIdx));
        if (outAddr == nullptr) {
            return;
        }

        const uint64_t rowBytes = static_cast<uint64_t>(laneWidth_) * sizeof(T);
        const uint16_t blockLen = static_cast<uint16_t>(rowBytes / kAlignBytes);
        const uint16_t srcGap = static_cast<uint16_t>(((splitCount_ - 1) * rowBytes) / kAlignBytes);
        const DataCopyParams gmToUbParams = {static_cast<uint16_t>(rowCount), blockLen, srcGap, 0};
        const DataCopyParams ubToGmParams = {static_cast<uint16_t>(rowCount), blockLen, 0, 0};

        GlobalTensor<T> outputGm;
        outputGm.SetGlobalBuffer(outAddr, static_cast<uint32_t>(frontSpan_ * laneWidth_));
        const int64_t inputBase = (outerBegin * splitCount_ + outIdx) * laneWidth_;
        const int64_t outputBase = outerBegin * laneWidth_;

        LocalTensor<T> rowLocal = relayPipe_.AllocTensor<T>();
        DataCopy(rowLocal, srcHub_[inputBase], gmToUbParams);
        relayPipe_.EnQue(rowLocal);

        rowLocal = relayPipe_.DeQue<T>();
        DataCopy(outputGm[outputBase], rowLocal, ubToGmParams);
        relayPipe_.FreeTensor(rowLocal);
    }

    __aicore__ inline void RunFloatSlabFanout(int64_t startTask, int64_t endTask) {
        const int64_t slabWidth = FloatSlabWidth();
        const int64_t rowsPerTicket = FloatSlabRowsPerTicket();
        const int64_t outputGroups = (splitCount_ + slabWidth - 1) / slabWidth;
        for (int64_t taskId = startTask; taskId < endTask; ++taskId) {
            const int64_t outerBegin = (taskId / outputGroups) * rowsPerTicket;
            if (outerBegin >= frontSpan_) {
                break;
            }
            const int64_t outGroup = taskId % outputGroups;
            int64_t rowCount = frontSpan_ - outerBegin;
            if (rowCount > rowsPerTicket) {
                rowCount = rowsPerTicket;
            }
            FanoutFloatSlab(outerBegin, rowCount, outGroup * slabWidth, slabWidth);
        }
    }

    __aicore__ inline void FanoutFloatSlab(int64_t outerBegin, int64_t rowCount,
                                                           int64_t outBlockBegin, int64_t slabWidth) {
        if (outBlockBegin >= splitCount_) {
            return;
        }
        int64_t blockOutputs = splitCount_ - outBlockBegin;
        if (blockOutputs > slabWidth) {
            blockOutputs = slabWidth;
        }
        const int64_t inputBase = (outerBegin * splitCount_ + outBlockBegin) * laneWidth_;
        const uint32_t groupBytes = static_cast<uint32_t>(blockOutputs * laneWidth_ * sizeof(T));
        const uint32_t inputSrcStride = static_cast<uint32_t>((splitCount_ - blockOutputs) * laneWidth_ * sizeof(T));

        LocalTensor<T> inputLocal = readPipe_.AllocTensor<T>();
        DataCopyExtParams gmToUbParams = {static_cast<uint16_t>(rowCount), groupBytes, inputSrcStride, 0, 0};
        DataCopyPadExtParams<T> padParams = {false, 0, 0, static_cast<T>(0)};
        DataCopyPad(inputLocal, srcHub_[inputBase], gmToUbParams, padParams);
        readPipe_.EnQue(inputLocal);

        inputLocal = readPipe_.DeQue<T>();
        const uint16_t innerBlocks = static_cast<uint16_t>((laneWidth_ * sizeof(T)) / kAlignBytes);
        const uint16_t outputSrcGap = static_cast<uint16_t>(((blockOutputs - 1) * laneWidth_ * sizeof(T)) / kAlignBytes);
        for (int64_t localOut = 0; localOut < blockOutputs; ++localOut) {
            __gm__ T *outAddr = dstRoster_.GetDataPtr<__gm__ T>(static_cast<uint32_t>(outBlockBegin + localOut));
            if (outAddr == nullptr) {
                continue;
            }
            GlobalTensor<T> outputGm;
            outputGm.SetGlobalBuffer(outAddr, static_cast<uint32_t>(frontSpan_ * laneWidth_));
            const DataCopyParams ubToGmParams = {static_cast<uint16_t>(rowCount), innerBlocks, outputSrcGap, 0};
            DataCopy(outputGm[outerBegin * laneWidth_], inputLocal[localOut * laneWidth_], ubToGmParams);
        }
        PipeBarrier<PIPE_MTE3>();
        readPipe_.FreeTensor(inputLocal);
    }

    __aicore__ inline void RunFloatSingleRowBalance() {
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        if (blockNum <= 0 || blockIdx >= blockNum) {
            return;
        }

        const int64_t fullSplitsPerCore = splitCount_ / blockNum;
        const int64_t balancedSplitCount = fullSplitsPerCore * blockNum;
        const int64_t tailSplitCount = splitCount_ - balancedSplitCount;
        const int64_t baseSplit = blockIdx * fullSplitsPerCore;
        for (int64_t i = 0; i < fullSplitsPerCore; ++i) {
            CopySingleRowWindow(baseSplit + i, 0, laneWidth_);
        }

        if (tailSplitCount <= 0) {
            return;
        }
        if (blockIdx < tailSplitCount) {
            const int64_t bodyElems = AlignedBodyElems(laneWidth_);
            CopySingleRowWindow(balancedSplitCount + blockIdx, 0, bodyElems);
        } else if (blockIdx < 2 * tailSplitCount) {
            const int64_t tailIdx = blockIdx - tailSplitCount;
            const int64_t bodyElems = AlignedBodyElems(laneWidth_);
            CopySingleRowWindow(balancedSplitCount + tailIdx, bodyElems, laneWidth_ - bodyElems);
        }
    }

    __aicore__ inline void RunFloatSingleRowBodyTail() {
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        if (blockNum <= 1 || blockIdx >= blockNum) {
            return;
        }

        int64_t tailCoreCount = bodyTailCores_;
        if (tailCoreCount < 1) {
            tailCoreCount = 1;
        }
        if (tailCoreCount >= blockNum) {
            tailCoreCount = blockNum - 1;
        }
        const int64_t bodyCoreCount = blockNum - tailCoreCount;
        const int64_t bodyElems = AlignedBodyElems(laneWidth_);
        const int64_t tailElems = laneWidth_ - bodyElems;
        if (bodyCoreCount <= 0 || bodyElems <= 0 || tailElems <= 0) {
            return;
        }

        if (blockIdx < bodyCoreCount) {
            CopySingleRowManualBodyRangeByWorker(blockIdx, bodyCoreCount, bodyElems);
        } else {
            CopySingleRowTailRelayBatchByWorker(blockIdx - bodyCoreCount, tailCoreCount, bodyElems, tailElems);
        }
    }

    __aicore__ inline LocalTensor<T> ManualBodyBuffer(uint32_t slot) {
        return slot == 0 ? manualBodyBuf0_.Get<T>() : manualBodyBuf1_.Get<T>();
    }

    __aicore__ inline int32_t ManualBodyEvent(uint32_t slot) const {
        return slot == 0 ? EVENT_ID0 : EVENT_ID1;
    }

    __aicore__ inline void CopySingleRowManualBodyRangeByWorker(int64_t workerIdx, int64_t workerCount,
                                                                int64_t bodyElems) {
        if (workerCount <= 0 || workerIdx < 0 || workerIdx >= workerCount || bodyElems <= 0) {
            return;
        }
        const int64_t firstSplit = (splitCount_ * workerIdx) / workerCount;
        const int64_t endSplit = (splitCount_ * (workerIdx + 1)) / workerCount;
        if (firstSplit >= endSplit) {
            return;
        }

        const uint32_t copyElems = static_cast<uint32_t>(bodyElems);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        PrimeManualBodySplit(firstSplit, 0U, copyElems);
        for (int64_t outIdx = firstSplit + 1; outIdx < endSplit; ++outIdx) {
            const uint32_t nextSlot = static_cast<uint32_t>(outIdx - firstSplit) & 1U;
            const uint32_t readySlot = nextSlot ^ 1U;
            PrimeManualBodySplit(outIdx, nextSlot, copyElems);
            DrainManualBodySplit(outIdx - 1, readySlot, copyElems);
        }
        DrainManualBodySplit(endSplit - 1, static_cast<uint32_t>(endSplit - 1 - firstSplit) & 1U, copyElems);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void PrimeManualBodySplit(int64_t outIdx, uint32_t slot, uint32_t copyElems) {
        const int32_t eventId = ManualBodyEvent(slot);
        LocalTensor<T> local = ManualBodyBuffer(slot);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        DataCopy(local, srcHub_[outIdx * laneWidth_], copyElems);
        SetFlag<HardEvent::MTE2_MTE3>(eventId);
    }

    __aicore__ inline void DrainManualBodySplit(int64_t outIdx, uint32_t slot, uint32_t copyElems) {
        const int32_t eventId = ManualBodyEvent(slot);
        LocalTensor<T> local = ManualBodyBuffer(slot);
        WaitFlag<HardEvent::MTE2_MTE3>(eventId);
        __gm__ T *outAddr = dstRoster_.GetDataPtr<__gm__ T>(static_cast<uint32_t>(outIdx));
        if (outAddr != nullptr) {
            GlobalTensor<T> outputGm;
            outputGm.SetGlobalBuffer(outAddr, static_cast<uint32_t>(laneWidth_));
            DataCopy(outputGm[0], local, copyElems);
        }
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
    }

    __aicore__ inline void CopySingleRowRangeByWorker(int64_t workerIdx, int64_t workerCount,
                                                      int64_t innerOffset, int64_t elemCount) {
        if (workerCount <= 0 || workerIdx < 0 || workerIdx >= workerCount || elemCount <= 0) {
            return;
        }
        const int64_t firstSplit = (splitCount_ * workerIdx) / workerCount;
        const int64_t endSplit = (splitCount_ * (workerIdx + 1)) / workerCount;
        for (int64_t outIdx = firstSplit; outIdx < endSplit; ++outIdx) {
            CopySingleRowWindow(outIdx, innerOffset, elemCount);
        }
    }

    __aicore__ inline void CopySingleRowAlignedRangeByWorker(int64_t workerIdx, int64_t workerCount,
                                                             int64_t innerOffset, int64_t elemCount) {
        if (workerCount <= 0 || workerIdx < 0 || workerIdx >= workerCount || elemCount <= 0) {
            return;
        }
        const int64_t firstSplit = (splitCount_ * workerIdx) / workerCount;
        const int64_t endSplit = (splitCount_ * (workerIdx + 1)) / workerCount;
        for (int64_t outIdx = firstSplit; outIdx < endSplit; ++outIdx) {
            CopySingleRowAlignedWindow(outIdx, innerOffset, elemCount);
        }
    }

    __aicore__ inline void CopySingleRowRelayRangeByWorker(int64_t workerIdx, int64_t workerCount,
                                                           int64_t innerOffset, int64_t elemCount) {
        if (workerCount <= 0 || workerIdx < 0 || workerIdx >= workerCount || elemCount <= 0) {
            return;
        }
        const int64_t firstSplit = (splitCount_ * workerIdx) / workerCount;
        const int64_t endSplit = (splitCount_ * (workerIdx + 1)) / workerCount;
        for (int64_t outIdx = firstSplit; outIdx < endSplit; ++outIdx) {
            CopySingleRowRelayWindow(outIdx, innerOffset, static_cast<uint32_t>(elemCount));
        }
    }

    __aicore__ inline int64_t AlignedBodyElems(int64_t elemCount) const {
        const int64_t preferredElems = static_cast<int64_t>(kMtePreferredAlignBytes / sizeof(T));
        if (preferredElems <= 0) {
            return elemCount;
        }
        return (elemCount / preferredElems) * preferredElems;
    }

    __aicore__ inline void CopySingleRowWindow(int64_t outIdx, int64_t innerOffset, int64_t elemCount) {
        if (outIdx < 0 || outIdx >= splitCount_ || elemCount <= 0) {
            return;
        }
        __gm__ T *outAddr = dstRoster_.GetDataPtr<__gm__ T>(static_cast<uint32_t>(outIdx));
        if (outAddr == nullptr) {
            return;
        }
        GlobalTensor<T> outputGm;
        outputGm.SetGlobalBuffer(outAddr, static_cast<uint32_t>(laneWidth_));
        StageWindow(outputGm, outIdx * laneWidth_ + innerOffset, innerOffset, elemCount);
    }

    __aicore__ inline void CopySingleRowAlignedWindow(int64_t outIdx, int64_t innerOffset, int64_t elemCount) {
        if (outIdx < 0 || outIdx >= splitCount_ || elemCount <= 0) {
            return;
        }
        __gm__ T *outAddr = dstRoster_.GetDataPtr<__gm__ T>(static_cast<uint32_t>(outIdx));
        if (outAddr == nullptr) {
            return;
        }
        GlobalTensor<T> outputGm;
        outputGm.SetGlobalBuffer(outAddr, static_cast<uint32_t>(laneWidth_));
        StageAlignedWindow(outputGm, outIdx * laneWidth_ + innerOffset, innerOffset,
                           static_cast<uint32_t>(elemCount));
    }

    __aicore__ inline void CopySingleRowRelayWindow(int64_t outIdx, int64_t innerOffset, uint32_t elemCount) {
        if (outIdx < 0 || outIdx >= splitCount_ || elemCount == 0) {
            return;
        }
        __gm__ T *outAddr = dstRoster_.GetDataPtr<__gm__ T>(static_cast<uint32_t>(outIdx));
        if (outAddr == nullptr) {
            return;
        }
        GlobalTensor<T> outputGm;
        outputGm.SetGlobalBuffer(outAddr, static_cast<uint32_t>(laneWidth_));

        LocalTensor<T> laneLocal = relayPipe_.AllocTensor<T>();
        DataCopy(laneLocal, srcHub_[outIdx * laneWidth_ + innerOffset], elemCount);
        relayPipe_.EnQue(laneLocal);

        laneLocal = relayPipe_.DeQue<T>();
        DataCopy(outputGm[innerOffset], laneLocal, elemCount);
        relayPipe_.FreeTensor(laneLocal);
    }

    __aicore__ inline int64_t PaddedWindowElems(int64_t elemCount) const {
        const int64_t copyBytes = elemCount * static_cast<int64_t>(sizeof(T));
        const int64_t paddedBytes = ((copyBytes + static_cast<int64_t>(kAlignBytes) - 1) /
                                     static_cast<int64_t>(kAlignBytes)) *
                                    static_cast<int64_t>(kAlignBytes);
        return paddedBytes / static_cast<int64_t>(sizeof(T));
    }

    __aicore__ inline void CopySingleRowTailBatchByWorker(int64_t workerIdx, int64_t workerCount,
                                                          int64_t innerOffset, int64_t elemCount) {
        if (workerCount <= 0 || workerIdx < 0 || workerIdx >= workerCount || elemCount <= 0) {
            return;
        }
        const int64_t firstSplit = (splitCount_ * workerIdx) / workerCount;
        const int64_t endSplit = (splitCount_ * (workerIdx + 1)) / workerCount;
        if (firstSplit >= endSplit) {
            return;
        }

        const int64_t paddedElems = PaddedWindowElems(elemCount);
        if (paddedElems <= 0) {
            return;
        }
        int64_t maxRowsPerBatch = stageElems_ / paddedElems;
        if (maxRowsPerBatch <= 0) {
            maxRowsPerBatch = 1;
        }
        const uint32_t tailBytes = static_cast<uint32_t>(elemCount * static_cast<int64_t>(sizeof(T)));
        const uint32_t srcGapBytes = static_cast<uint32_t>((laneWidth_ - elemCount) * static_cast<int64_t>(sizeof(T)));
        const DataCopyExtParams ubToGmParams = {1, tailBytes, 0, 0, 0};

        for (int64_t splitBase = firstSplit; splitBase < endSplit; splitBase += maxRowsPerBatch) {
            int64_t currentRows = endSplit - splitBase;
            if (currentRows > maxRowsPerBatch) {
                currentRows = maxRowsPerBatch;
            }

            const DataCopyExtParams gmToUbParams = {
                static_cast<uint16_t>(currentRows), tailBytes, srcGapBytes, 0, 0};
            const DataCopyPadExtParams<T> padParams = {false, 0, 0, static_cast<T>(0)};

            LocalTensor<T> inputLocal = readPipe_.AllocTensor<T>();
            DataCopyPad(inputLocal, srcHub_[splitBase * laneWidth_ + innerOffset], gmToUbParams, padParams);
            readPipe_.EnQue(inputLocal);

            inputLocal = readPipe_.DeQue<T>();
            LocalTensor<T> outputLocal = writePipe_.AllocTensor<T>();
            DataCopy(outputLocal, inputLocal, static_cast<uint32_t>(currentRows * paddedElems));
            writePipe_.EnQue(outputLocal);
            readPipe_.FreeTensor(inputLocal);

            outputLocal = writePipe_.DeQue<T>();
            for (int64_t row = 0; row < currentRows; ++row) {
                const int64_t outIdx = splitBase + row;
                __gm__ T *outAddr = dstRoster_.GetDataPtr<__gm__ T>(static_cast<uint32_t>(outIdx));
                if (outAddr == nullptr) {
                    continue;
                }
                GlobalTensor<T> outputGm;
                outputGm.SetGlobalBuffer(outAddr, static_cast<uint32_t>(laneWidth_));
                DataCopyPad(outputGm[innerOffset], outputLocal[row * paddedElems], ubToGmParams);
            }
            writePipe_.FreeTensor(outputLocal);
        }
    }

    __aicore__ inline void CopySingleRowTailRelayBatchByWorker(int64_t workerIdx, int64_t workerCount,
                                                               int64_t innerOffset, int64_t elemCount) {
        if (workerCount <= 0 || workerIdx < 0 || workerIdx >= workerCount || elemCount <= 0) {
            return;
        }
        const int64_t firstSplit = (splitCount_ * workerIdx) / workerCount;
        const int64_t endSplit = (splitCount_ * (workerIdx + 1)) / workerCount;
        if (firstSplit >= endSplit) {
            return;
        }

        const int64_t paddedElems = PaddedWindowElems(elemCount);
        if (paddedElems <= 0) {
            return;
        }
        int64_t maxRowsPerBatch = stageElems_ / paddedElems;
        if (maxRowsPerBatch <= 0) {
            maxRowsPerBatch = 1;
        }
        const uint32_t tailBytes = static_cast<uint32_t>(elemCount * static_cast<int64_t>(sizeof(T)));
        const uint32_t srcGapBytes = static_cast<uint32_t>((laneWidth_ - elemCount) * static_cast<int64_t>(sizeof(T)));
        const DataCopyExtParams ubToGmParams = {1, tailBytes, 0, 0, 0};

        for (int64_t splitBase = firstSplit; splitBase < endSplit; splitBase += maxRowsPerBatch) {
            int64_t currentRows = endSplit - splitBase;
            if (currentRows > maxRowsPerBatch) {
                currentRows = maxRowsPerBatch;
            }

            const DataCopyExtParams gmToUbParams = {
                static_cast<uint16_t>(currentRows), tailBytes, srcGapBytes, 0, 0};
            const DataCopyPadExtParams<T> padParams = {false, 0, 0, static_cast<T>(0)};

            LocalTensor<T> inputLocal = relayPipe_.AllocTensor<T>();
            DataCopyPad(inputLocal, srcHub_[splitBase * laneWidth_ + innerOffset], gmToUbParams, padParams);
            relayPipe_.EnQue(inputLocal);

            inputLocal = relayPipe_.DeQue<T>();
            for (int64_t row = 0; row < currentRows; ++row) {
                const int64_t outIdx = splitBase + row;
                __gm__ T *outAddr = dstRoster_.GetDataPtr<__gm__ T>(static_cast<uint32_t>(outIdx));
                if (outAddr == nullptr) {
                    continue;
                }
                GlobalTensor<T> outputGm;
                outputGm.SetGlobalBuffer(outAddr, static_cast<uint32_t>(laneWidth_));
                DataCopyPad(outputGm[innerOffset], inputLocal[row * paddedElems], ubToGmParams);
            }
            relayPipe_.FreeTensor(inputLocal);
        }
    }

    __aicore__ inline void StageAlignedWindow(GlobalTensor<T> &outputGm, int64_t inputOffset,
                                              int64_t outputOffset, uint32_t elemCount) {
        LocalTensor<T> inputLocal = readPipe_.AllocTensor<T>();
        DataCopy(inputLocal, srcHub_[inputOffset], elemCount);
        readPipe_.EnQue(inputLocal);

        inputLocal = readPipe_.DeQue<T>();
        LocalTensor<T> outputLocal = writePipe_.AllocTensor<T>();
        DataCopy(outputLocal, inputLocal, elemCount);
        writePipe_.EnQue(outputLocal);
        readPipe_.FreeTensor(inputLocal);

        outputLocal = writePipe_.DeQue<T>();
        DataCopy(outputGm[outputOffset], outputLocal, elemCount);
        writePipe_.FreeTensor(outputLocal);
    }

    __aicore__ inline bool IsStridedMteRoute() const {
        if constexpr (std::is_same_v<T, bool>) {
            return false;
        }
        if (laneWidth_ <= 0 || stageElems_ < laneWidth_) {
            return false;
        }
        const uint64_t rowBytes = static_cast<uint64_t>(laneWidth_) * sizeof(T);
        const uint64_t srcStrideBytes = static_cast<uint64_t>(splitCount_) * rowBytes;
        const uint64_t rowBlocks = rowBytes / kAlignBytes;
        const uint64_t srcGapBlocks = (static_cast<uint64_t>(splitCount_) - 1ULL) * rowBlocks;
        return rowBytes >= kAlignBytes && (rowBytes % kAlignBytes) == 0 &&
               (srcStrideBytes % kAlignBytes) == 0 &&
               PrefersPlainMte(rowBytes) &&
               rowBlocks <= static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) &&
               srcGapBlocks <= static_cast<uint64_t>(std::numeric_limits<uint16_t>::max());
    }

    __aicore__ inline bool UseMteRelayRows() const {
        if constexpr (std::is_same_v<T, bool>) {
            return false;
        }
        return directLaneRelay_;
    }

    __aicore__ inline void RelayRowsNoVecCopy(GlobalTensor<T> &outputGm, int64_t outIdx, int64_t outerBegin,
                                              int64_t rowCount) {
        const uint64_t rowBytes = static_cast<uint64_t>(laneWidth_) * sizeof(T);
        const uint16_t blockLen = static_cast<uint16_t>(rowBytes / kAlignBytes);
        const uint16_t srcGap = static_cast<uint16_t>(((splitCount_ - 1) * rowBytes) / kAlignBytes);
        const DataCopyParams gmToUbParams = {static_cast<uint16_t>(rowCount), blockLen, srcGap, 0};
        const DataCopyParams ubToGmParams = {static_cast<uint16_t>(rowCount), blockLen, 0, 0};

        const int64_t inputBase = (outerBegin * splitCount_ + outIdx) * laneWidth_;
        const int64_t outputBase = outerBegin * laneWidth_;

        LocalTensor<T> inputLocal = readPipe_.AllocTensor<T>();
        DataCopy(inputLocal, srcHub_[inputBase], gmToUbParams);
        readPipe_.EnQue(inputLocal);

        inputLocal = readPipe_.DeQue<T>();
        DataCopy(outputGm[outputBase], inputLocal, ubToGmParams);
        readPipe_.FreeTensor(inputLocal);
    }
    __aicore__ inline void RelayRowsViaVecOut(GlobalTensor<T> &outputGm, int64_t outIdx, int64_t outerBegin, int64_t rowCount) {
        const int64_t batchElems = rowCount * laneWidth_;
        const uint64_t rowBytes = static_cast<uint64_t>(laneWidth_) * sizeof(T);
        const uint16_t blockLen = static_cast<uint16_t>(rowBytes / kAlignBytes);
        const uint16_t srcGap = static_cast<uint16_t>(((splitCount_ - 1) * rowBytes) / kAlignBytes);
        const DataCopyParams gmToUbParams = {static_cast<uint16_t>(rowCount), blockLen, srcGap, 0};
        const DataCopyParams ubToGmParams = {static_cast<uint16_t>(rowCount), blockLen, 0, 0};

        const int64_t inputBase = (outerBegin * splitCount_ + outIdx) * laneWidth_;
        const int64_t outputBase = outerBegin * laneWidth_;

        LocalTensor<T> inputLocal = readPipe_.AllocTensor<T>();
        DataCopy(inputLocal, srcHub_[inputBase], gmToUbParams);
        readPipe_.EnQue(inputLocal);

        inputLocal = readPipe_.DeQue<T>();
        LocalTensor<T> outputLocal = writePipe_.AllocTensor<T>();
        MirrorUbWindow(outputLocal, inputLocal, batchElems);
        writePipe_.EnQue(outputLocal);
        readPipe_.FreeTensor(inputLocal);

        outputLocal = writePipe_.DeQue<T>();
        DataCopy(outputGm[outputBase], outputLocal, ubToGmParams);
        writePipe_.FreeTensor(outputLocal);
    }

    __aicore__ inline WindowPlan PlanWindow(int64_t elemCount) const {
        WindowPlan plan;
        plan.bytes = static_cast<uint32_t>(elemCount * sizeof(T));
        plan.split512 = NeedsMte512Split(plan.bytes);
        plan.blockAligned = ((plan.bytes % kAlignBytes) == 0);
        if (plan.split512) {
            const uint32_t elemBytes = static_cast<uint32_t>(sizeof(T));
            const uint32_t headBytes = (plan.bytes / kMtePreferredAlignBytes) * kMtePreferredAlignBytes;
            plan.headElems = headBytes / elemBytes;
            plan.tailBytes = plan.bytes - headBytes;
            plan.tailElems = elemCount - static_cast<int64_t>(plan.headElems);
            plan.tailAlignedElems = ((plan.tailBytes / kAlignBytes) * kAlignBytes) / elemBytes;
        }
        return plan;
    }

    __aicore__ inline void PrimeWindowFromGm(const LocalTensor<T> &inputLocal, int64_t inputOffset,
                                             int64_t elemCount, const WindowPlan &plan) {
        if (plan.split512) {
            DataCopy(inputLocal, srcHub_[inputOffset], plan.headElems);
            DataCopyExtParams gmToUbParams = {1, plan.tailBytes, 0, 0, 0};
            DataCopyPadExtParams<T> padParams = {false, 0, 0, static_cast<T>(0)};
            DataCopyPad(inputLocal[plan.headElems], srcHub_[inputOffset + plan.headElems], gmToUbParams, padParams);
        } else if (plan.blockAligned) {
            DataCopy(inputLocal, srcHub_[inputOffset], static_cast<uint32_t>(elemCount));
        } else {
            DataCopyExtParams gmToUbParams = {1, plan.bytes, 0, 0, 0};
            DataCopyPadExtParams<T> padParams = {false, 0, 0, static_cast<T>(0)};
            DataCopyPad(inputLocal, srcHub_[inputOffset], gmToUbParams, padParams);
        }
    }

    __aicore__ inline void ShuttleWindowInUb(const LocalTensor<T> &outputLocal, const LocalTensor<T> &inputLocal,
                                             int64_t elemCount, const WindowPlan &plan) {
        if (!plan.split512) {
            MirrorUbWindow(outputLocal, inputLocal, elemCount);
            return;
        }
        DataCopy(outputLocal, inputLocal, plan.headElems);
        if (plan.tailAlignedElems > 0) {
            DataCopy(outputLocal[plan.headElems], inputLocal[plan.headElems], plan.tailAlignedElems);
        }
        for (int64_t i = static_cast<int64_t>(plan.tailAlignedElems); i < plan.tailElems; ++i) {
            outputLocal.SetValue(static_cast<int64_t>(plan.headElems) + i,
                                 inputLocal.GetValue(static_cast<int64_t>(plan.headElems) + i));
        }
    }

    __aicore__ inline void CommitWindowToGm(GlobalTensor<T> &outputGm, const LocalTensor<T> &outputLocal,
                                           int64_t outputOffset, int64_t elemCount, const WindowPlan &plan) {
        if (plan.split512) {
            DataCopy(outputGm[outputOffset], outputLocal, plan.headElems);
            DataCopyExtParams ubToGmParams = {1, plan.tailBytes, 0, 0, 0};
            DataCopyPad(outputGm[outputOffset + plan.headElems], outputLocal[plan.headElems], ubToGmParams);
        } else if (plan.blockAligned) {
            DataCopy(outputGm[outputOffset], outputLocal, static_cast<uint32_t>(elemCount));
        } else {
            DataCopyExtParams ubToGmParams = {1, plan.bytes, 0, 0, 0};
            DataCopyPad(outputGm[outputOffset], outputLocal, ubToGmParams);
        }
    }

    __aicore__ inline void StageWindow(GlobalTensor<T> &outputGm, int64_t inputOffset, int64_t outputOffset,
                                       int64_t elemCount) {
        const WindowPlan plan = PlanWindow(elemCount);
        if (plan.bytes < kAlignBytes) {
            ScalarDribble(outputGm, inputOffset, outputOffset, elemCount);
            return;
        }

        LocalTensor<T> inputLocal = readPipe_.AllocTensor<T>();
        PrimeWindowFromGm(inputLocal, inputOffset, elemCount, plan);
        readPipe_.EnQue(inputLocal);

        inputLocal = readPipe_.DeQue<T>();
        LocalTensor<T> outputLocal = writePipe_.AllocTensor<T>();
        ShuttleWindowInUb(outputLocal, inputLocal, elemCount, plan);
        writePipe_.EnQue(outputLocal);
        readPipe_.FreeTensor(inputLocal);

        outputLocal = writePipe_.DeQue<T>();
        CommitWindowToGm(outputGm, outputLocal, outputOffset, elemCount, plan);
        writePipe_.FreeTensor(outputLocal);
    }

    __aicore__ inline void ScalarDribble(GlobalTensor<T> &outputGm, int64_t inputOffset, int64_t outputOffset,
                                          int64_t elemCount) {
        for (int64_t i = 0; i < elemCount; ++i) {
            outputGm.SetValue(outputOffset + i, srcHub_.GetValue(inputOffset + i));
        }
    }

    __aicore__ inline void MirrorUbWindow(const LocalTensor<T> &dst, const LocalTensor<T> &src, int64_t elemCount) {
        const uint32_t copyBytes = static_cast<uint32_t>(elemCount * sizeof(T));
        if constexpr (!std::is_same_v<T, bool>) {
            if (copyBytes >= kAlignBytes) {
                uint32_t alignedBytes = ((copyBytes + kAlignBytes - 1) / kAlignBytes) * kAlignBytes;
                if (copyBytes > kMtePreferredAlignBytes &&
                    (copyBytes % kMtePreferredAlignBytes) == 0) {
                    alignedBytes = copyBytes;
                }
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
    TPipe fabric_;
    TQue<QuePosition::VECIN, kBufferNum> readPipe_;
    TQue<QuePosition::VECOUT, kBufferNum> writePipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, kBufferNum> relayPipe_;
    TBuf<QuePosition::VECCALC> manualBodyBuf0_;
    TBuf<QuePosition::VECCALC> manualBodyBuf1_;
    TBuf<QuePosition::VECCALC> offsetScratch_;
    GlobalTensor<T> srcHub_;
    AscendC::ListTensorDesc dstRoster_;
    int64_t splitCount_ = 0;
    int64_t frontSpan_ = 0;
    int64_t laneWidth_ = 0;
    int64_t route_ = 0;
    int64_t ticketCount_ = 0;
    int64_t ticketsPerCore_ = 0;
    int64_t stageElems_ = 0;
    int64_t bodyTailCores_ = 0;
    bool directLaneRelay_ = false;
};

} // namespace

extern "C" __global__ __aicore__ void unpack(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    UnpackFlowCore<DTYPE_X> op;
    op.Init(x, y, tiling_data);
    op.Run();
}
