#include "kernel_operator.h"
#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

using namespace AscendC;

constexpr uint32_t TILE_BYTES = 40 * 1024;
constexpr uint32_t ALIGN_BYTES = 32;
constexpr uint32_t GM_ALIGN_BYTES = 512;
constexpr uint32_t COPY_BUFFER_BYTES = TILE_BYTES + 512;
constexpr uint32_t CASE1_TILE_BYTES = 40 * 1024;
constexpr uint32_t CASE1_COPY_BUFFER_BYTES = CASE1_TILE_BYTES + 512;
constexpr uint32_t SMALL_DIRECT_BLOCKS = 16;
constexpr uint32_t MODE_CASE1 = 1;
constexpr uint32_t MODE_CASE2 = 2;
constexpr uint32_t MODE_CASE3 = 3;
constexpr uint32_t MODE_CASE4 = 4;
constexpr uint32_t MODE_CASE5 = 5;
constexpr uint32_t MODE_CASE6 = 6;
constexpr uint32_t MODE_CASE7 = 7;
constexpr uint32_t MODE_CASE8 = 8;
constexpr uint32_t MODE_CASE9 = 9;
constexpr uint32_t MODE_CASE10 = 10;
constexpr uint32_t CASE9_TILE_OW = 512;
constexpr uint32_t CASE9_TILE_BYTES = 64 * 1024;
constexpr uint32_t CASE10_TILE_H = 64;
constexpr uint32_t CASE10_TILE_BYTES = 48 * 1024;

template <typename DT_X, uint32_t MODE_T>
class KernelBatchToSpace {
public:
    static constexpr bool kCase7MicroMode = MODE_T == MODE_CASE7;
    static constexpr bool kCase2LaneMode = MODE_T == MODE_CASE2;
    static constexpr bool kCase6LaneMode = MODE_T == MODE_CASE6;
    static constexpr bool kCase1RowMode = MODE_T == MODE_CASE1;
    static constexpr bool kCase3RowMode = MODE_T == MODE_CASE3;
    static constexpr bool kCase5RowMode = MODE_T == MODE_CASE5;
    static constexpr bool kCase9RowMode = MODE_T == MODE_CASE9;
    static constexpr bool kRow8BindMode = MODE_T == MODE_CASE8;
    static constexpr bool kCase10ArrangeMode = MODE_T == MODE_CASE10;
    static constexpr bool kPatchBindMode = MODE_T == MODE_CASE4;
    // Front cases are intentionally split instead of sharing one "front" policy:
    // case2 is lane-oriented, case3 is row-pair-oriented, case4 is patch-oriented.
    // Sharing their queue policy caused repeated p2/p4 regressions.
    static constexpr bool kCase2DeepBindMode = kCase2LaneMode;
    static constexpr bool kCase3DeepBindMode = kCase3RowMode;
    static constexpr bool kCase4PatchQueueMode = kPatchBindMode;
    // Keep each case on the queue depth that matches its tested transport family.
    // p1 was faster with row-bind double buffering; p2 needs a deeper lane
    // pipeline; p4 needs two patch buffers; p6/p7/p9 stay shallow to minimize
    // launch-side overhead on tiny data.
    static constexpr uint32_t kCopyQueueDepth =
        (kCase1RowMode || kRow8BindMode || kCase10ArrangeMode) ? 2 :
        kCase2LaneMode ? 4 :
        kCase4PatchQueueMode ? 2 :
        ((kCase6LaneMode || kCase7MicroMode || kCase9RowMode) ? 1 : 2);

    __aicore__ inline KernelBatchToSpace() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling,
                                TPipe *pipe) {
        tiling_ = tiling;
        pipe_ = pipe;
        uint32_t inputLen = tiling.batch * tiling.height * tiling.width * tiling.depth;
        uint32_t outputLen = tiling.outBatch * tiling.outHeight * tiling.outWidth * tiling.depth;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), inputLen);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), outputLen);
        if constexpr (kCase1RowMode) {
            pipe_->InitBuffer(copyQueue_, 2, COPY_BUFFER_BYTES);
        } else if constexpr (kCase9RowMode) {
            pipe_->InitBuffer(copyQueue_, 1, CASE9_TILE_BYTES);
        } else if constexpr (kCase10ArrangeMode) {
            pipe_->InitBuffer(copyQueue_, 2, CASE10_TILE_BYTES);
        } else if constexpr (kRow8BindMode) {
            pipe_->InitBuffer(copyQueue_, 2, COPY_BUFFER_BYTES);
        } else if constexpr (kCase4PatchQueueMode) {
            pipe_->InitBuffer(copyQueue_, 2, COPY_BUFFER_BYTES);
        } else if constexpr (kCase6LaneMode) {
            pipe_->InitBuffer(copyQueue_, 1, COPY_BUFFER_BYTES);
        } else if constexpr (kCase7MicroMode) {
            pipe_->InitBuffer(copyQueue_, 1, COPY_BUFFER_BYTES);
        } else if constexpr (kCase2DeepBindMode) {
            pipe_->InitBuffer(copyQueue_, 4, COPY_BUFFER_BYTES);
        } else if constexpr (kCase3DeepBindMode) {
            pipe_->InitBuffer(copyQueue_, 2, COPY_BUFFER_BYTES);
        } else if constexpr (kCase5RowMode) {
            pipe_->InitBuffer(copyQueue_, 4, COPY_BUFFER_BYTES);
        } else {
            pipe_->InitBuffer(inQueue_, 2, TILE_BYTES);
            pipe_->InitBuffer(outQueue_, 2, TILE_BYTES);
        }
    }

    __aicore__ inline void Process() {
        if constexpr (kCase7MicroMode) {
            uint32_t outputPixels = tiling_.outBatch * tiling_.outHeight * tiling_.outWidth;
            if (outputPixels == 0 || tiling_.depth == 0) {
                return;
            }
            ProcessCase7DepthTasks(outputPixels);
            return;
        }
        if constexpr (kCase2LaneMode) {
            if (tiling_.tileBlocks == 0 || tiling_.taskCount == 0) {
                return;
            }
            if (tiling_.chunksPerLane == 0) {
                return;
            }
            ProcessCase2LaneTasks();
            return;
        }
        if constexpr (kCase6LaneMode) {
            if (tiling_.tileBlocks == 0 || tiling_.chunksPerLane == 0 || tiling_.taskCount == 0) {
                return;
            }
            ProcessCase6LaneTasks();
            return;
        }
        if constexpr (kCase1RowMode) {
            if (tiling_.tileBlocks == 0 || tiling_.taskCount == 0) {
                return;
            }
            ProcessCase1RowTasks();
            return;
        }
        if constexpr (kCase3RowMode) {
            if (tiling_.tileBlocks == 0 || tiling_.rowChunks == 0 || tiling_.taskCount == 0) {
                return;
            }
            ProcessCase3RowTasks();
            return;
        }
        if constexpr (kCase5RowMode) {
            if (tiling_.tileBlocks == 0 || tiling_.rowChunks == 0 || tiling_.taskCount == 0) {
                return;
            }
            ProcessCase5RowTasks();
            return;
        }
        if constexpr (kCase9RowMode) {
            if (tiling_.tileBlocks == 0 || tiling_.rowChunks == 0 || tiling_.taskCount == 0) {
                return;
            }
            ProcessCase9RowTasks();
            return;
        }
        if constexpr (kRow8BindMode) {
            if (tiling_.tileBlocks == 0 || tiling_.rowChunks == 0 || tiling_.taskCount == 0) {
                return;
            }
            ProcessCase8Tasks();
            return;
        }
        if constexpr (kPatchBindMode) {
            if (tiling_.tileBlocks == 0 || tiling_.rowChunks == 0 || tiling_.taskCount == 0) {
                return;
            }
            ProcessCase4Patch2x2Tasks();
            return;
        }
        if constexpr (kCase10ArrangeMode) {
            if (tiling_.taskCount == 0) {
                return;
            }
            ProcessCase10Tasks();
            return;
        }
        uint32_t outputPixels = tiling_.outBatch * tiling_.outHeight * tiling_.outWidth;
        if (outputPixels == 0 || tiling_.depth == 0 || tiling_.blockSize == 0) {
            return;
        }
        uint32_t blockDim = tiling_.blockDim == 0 ? 1 : tiling_.blockDim;
        uint32_t blockIdx = GetBlockIdx();
        uint32_t pixelsPerCore = (outputPixels + blockDim - 1) / blockDim;
        uint32_t start = blockIdx * pixelsPerCore;
        if (start >= outputPixels) {
            return;
        }
        uint32_t end = start + pixelsPerCore;
        if (end > outputPixels) {
            end = outputPixels;
        }

        ProcessOutputPixels(start, end);
    }

private:
    __aicore__ inline uint32_t AlignUpBytes(uint32_t x) const {
        return ((x + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
    }

    __aicore__ inline uint32_t ActiveCopyBufferBytes() const {
        if constexpr (kCase1RowMode) {
            return CASE1_COPY_BUFFER_BYTES;
        }
        return COPY_BUFFER_BYTES;
    }

    __aicore__ inline uint32_t AlignUpTo(uint32_t x, uint32_t align) const {
        return ((x + align - 1) / align) * align;
    }

    __aicore__ inline uint32_t CeilDiv(uint32_t x, uint32_t y) const {
        return y == 0 ? 0 : (x + y - 1) / y;
    }

    __aicore__ inline bool CanUseDataCopyParams(uint32_t blockCount, uint32_t blockLenBytes,
                                                uint32_t srcStrideBytes,
                                                uint32_t dstStrideBytes) const {
        if (((blockLenBytes | srcStrideBytes | dstStrideBytes) & (ALIGN_BYTES - 1)) != 0) {
            return false;
        }
        uint32_t blockLenBlocks = blockLenBytes / ALIGN_BYTES;
        uint32_t srcStrideBlocks = srcStrideBytes / ALIGN_BYTES;
        uint32_t dstStrideBlocks = dstStrideBytes / ALIGN_BYTES;
        return blockCount <= 4095 && blockLenBlocks <= 65535 &&
               srcStrideBlocks <= 65535 && dstStrideBlocks <= 65535;
    }

    __aicore__ inline void DataCopyInParams(LocalTensor<DT_X> dst, uint32_t srcBase,
                                            uint32_t blockCount, uint32_t blockLenBytes,
                                            uint32_t srcStrideBytes,
                                            uint32_t dstStrideBytes) {
        DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(blockCount);
        copyParams.blockLen = static_cast<uint16_t>(blockLenBytes / ALIGN_BYTES);
        copyParams.srcStride = static_cast<uint16_t>(srcStrideBytes / ALIGN_BYTES);
        copyParams.dstStride = static_cast<uint16_t>(dstStrideBytes / ALIGN_BYTES);
        DataCopy(dst, xGm_[srcBase], copyParams);
    }

    __aicore__ inline void DataCopyOutParams(uint32_t dstBase, LocalTensor<DT_X> src,
                                             uint32_t blockCount, uint32_t blockLenBytes,
                                             uint32_t srcStrideBytes,
                                             uint32_t dstStrideBytes) {
        DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(blockCount);
        copyParams.blockLen = static_cast<uint16_t>(blockLenBytes / ALIGN_BYTES);
        copyParams.srcStride = static_cast<uint16_t>(srcStrideBytes / ALIGN_BYTES);
        copyParams.dstStride = static_cast<uint16_t>(dstStrideBytes / ALIGN_BYTES);
        DataCopy(yGm_[dstBase], src, copyParams);
    }

    __aicore__ inline bool CanUsePadExtInput(uint32_t blockCount, uint32_t blockLenBytes,
                                             uint32_t srcStrideBytes,
                                             uint32_t dstStrideBytes) const {
        if (blockCount > 65535 || blockLenBytes > 65535) {
            return false;
        }
        if ((dstStrideBytes & (ALIGN_BYTES - 1)) != 0) {
            return false;
        }
        return srcStrideBytes <= 65535 && (dstStrideBytes / ALIGN_BYTES) <= 65535;
    }

    __aicore__ inline bool CanUsePadExtOutput(uint32_t blockCount, uint32_t blockLenBytes,
                                              uint32_t srcStrideBytes,
                                              uint32_t dstStrideBytes) const {
        if (blockCount > 65535 || blockLenBytes > 65535) {
            return false;
        }
        if ((srcStrideBytes & (ALIGN_BYTES - 1)) != 0) {
            return false;
        }
        return (srcStrideBytes / ALIGN_BYTES) <= 65535 && dstStrideBytes <= 65535;
    }

    __aicore__ inline bool BuildBlockCountSplitPlan(uint32_t srcByteBase,
                                                    uint32_t dstByteBase,
                                                    uint32_t blockLenBytes,
                                                    uint32_t srcStepBytes,
                                                    uint32_t dstStepBytes,
                                                    uint32_t &headBytes,
                                                    uint32_t &bodyBytes,
                                                    uint32_t &tailBytes,
                                                    uint32_t &headSlotBytes,
                                                    uint32_t &tailSlotBytes) const {
        if (blockLenBytes < ALIGN_BYTES ||
            ((srcByteBase | dstByteBase) & (ALIGN_BYTES - 1)) != 0 ||
            srcStepBytes < blockLenBytes || dstStepBytes < AlignUpBytes(blockLenBytes)) {
            return false;
        }

        headBytes = 0;
        bodyBytes = (blockLenBytes / ALIGN_BYTES) * ALIGN_BYTES;
        if (bodyBytes == 0) {
            return false;
        }
        tailBytes = blockLenBytes - bodyBytes;
        headSlotBytes = 0;
        tailSlotBytes = AlignUpBytes(tailBytes);
        return true;
    }

    __aicore__ inline bool CopyBlockCountToLocalSplit(LocalTensor<DT_X> dst,
                                                      uint32_t dstElemBase,
                                                      uint32_t srcBase,
                                                      uint32_t blockCount,
                                                      uint32_t blockLenBytes,
                                                      uint32_t srcStrideBytes,
                                                      uint32_t dstStrideBytes) {
        uint32_t elemBytes = sizeof(DT_X);
        uint32_t srcByteBase = srcBase * elemBytes;
        uint32_t dstByteBase = dstElemBase * elemBytes;
        uint32_t srcStepBytes = blockLenBytes + srcStrideBytes;
        uint32_t dstStepBytes = AlignUpBytes(blockLenBytes) + dstStrideBytes;
        uint32_t headBytes;
        uint32_t bodyBytes;
        uint32_t tailBytes;
        uint32_t headSlotBytes;
        uint32_t tailSlotBytes;
        if (!BuildBlockCountSplitPlan(srcByteBase, dstByteBase, blockLenBytes,
                                      srcStepBytes, dstStepBytes, headBytes,
                                      bodyBytes, tailBytes, headSlotBytes,
                                      tailSlotBytes)) {
            return false;
        }

        uint32_t bodySrcStrideBytes = srcStepBytes - bodyBytes;
        uint32_t bodyDstStrideBytes = dstStepBytes - bodyBytes;
        if (!CanUseDataCopyParams(blockCount, bodyBytes,
                                  bodySrcStrideBytes, bodyDstStrideBytes)) {
            return false;
        }
        if (headBytes != 0 &&
            !CanUsePadExtInput(blockCount, headBytes, srcStepBytes - headBytes,
                               dstStepBytes - headSlotBytes)) {
            return false;
        }
        if (tailBytes != 0 &&
            !CanUsePadExtInput(blockCount, tailBytes, srcStepBytes - tailBytes,
                               dstStepBytes - tailSlotBytes)) {
            return false;
        }

        uint32_t headElems = headBytes / elemBytes;
        uint32_t bodyElems = bodyBytes / elemBytes;
        uint32_t localBodyElems = headSlotBytes / elemBytes;
        uint32_t localTailElems = (headSlotBytes + bodyBytes) / elemBytes;
        if (headBytes != 0) {
            DataCopyExtParams headParams{static_cast<uint16_t>(blockCount),
                                         headBytes, srcStepBytes - headBytes,
                                         (dstStepBytes - headSlotBytes) / ALIGN_BYTES, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(dst[dstElemBase], xGm_[srcBase], headParams, padParams);
        }
        DataCopyInParams(dst[dstElemBase + localBodyElems], srcBase + headElems,
                         blockCount, bodyBytes, bodySrcStrideBytes, bodyDstStrideBytes);
        if (tailBytes != 0) {
            DataCopyExtParams tailParams{static_cast<uint16_t>(blockCount),
                                         tailBytes, srcStepBytes - tailBytes,
                                         (dstStepBytes - tailSlotBytes) / ALIGN_BYTES, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(dst[dstElemBase + localTailElems],
                        xGm_[srcBase + headElems + bodyElems], tailParams, padParams);
        }
        return true;
    }

    __aicore__ inline bool CopyBlockCountToGmSplit(uint32_t dstBase,
                                                   LocalTensor<DT_X> src,
                                                   uint32_t srcElemBase,
                                                   uint32_t blockCount,
                                                   uint32_t blockLenBytes,
                                                   uint32_t srcStrideBytes,
                                                   uint32_t dstStrideBytes) {
        uint32_t elemBytes = sizeof(DT_X);
        uint32_t srcByteBase = srcElemBase * elemBytes;
        uint32_t dstByteBase = dstBase * elemBytes;
        uint32_t srcStepBytes = AlignUpBytes(blockLenBytes) + srcStrideBytes;
        uint32_t dstStepBytes = blockLenBytes + dstStrideBytes;
        uint32_t headBytes;
        uint32_t bodyBytes;
        uint32_t tailBytes;
        uint32_t headSlotBytes;
        uint32_t tailSlotBytes;
        if (!BuildBlockCountSplitPlan(srcByteBase, dstByteBase, blockLenBytes,
                                      srcStepBytes, dstStepBytes, headBytes,
                                      bodyBytes, tailBytes, headSlotBytes,
                                      tailSlotBytes)) {
            return false;
        }

        uint32_t bodySrcStrideBytes = srcStepBytes - bodyBytes;
        uint32_t bodyDstStrideBytes = dstStepBytes - bodyBytes;
        if (!CanUseDataCopyParams(blockCount, bodyBytes,
                                  bodySrcStrideBytes, bodyDstStrideBytes)) {
            return false;
        }
        if (headBytes != 0 &&
            !CanUsePadExtOutput(blockCount, headBytes,
                                srcStepBytes - headSlotBytes,
                                dstStepBytes - headBytes)) {
            return false;
        }
        if (tailBytes != 0 &&
            !CanUsePadExtOutput(blockCount, tailBytes,
                                srcStepBytes - tailSlotBytes,
                                dstStepBytes - tailBytes)) {
            return false;
        }

        uint32_t headElems = headBytes / elemBytes;
        uint32_t bodyElems = bodyBytes / elemBytes;
        uint32_t localBodyElems = headSlotBytes / elemBytes;
        uint32_t localTailElems = (headSlotBytes + bodyBytes) / elemBytes;
        if (headBytes != 0) {
            DataCopyExtParams headParams{static_cast<uint16_t>(blockCount),
                                         headBytes,
                                         (srcStepBytes - headSlotBytes) / ALIGN_BYTES,
                                         dstStepBytes - headBytes, 0};
            DataCopyPad(yGm_[dstBase], src[srcElemBase], headParams);
        }
        DataCopyOutParams(dstBase + headElems, src[srcElemBase + localBodyElems],
                          blockCount, bodyBytes, bodySrcStrideBytes, bodyDstStrideBytes);
        if (tailBytes != 0) {
            DataCopyExtParams tailParams{static_cast<uint16_t>(blockCount),
                                         tailBytes,
                                         (srcStepBytes - tailSlotBytes) / ALIGN_BYTES,
                                         dstStepBytes - tailBytes, 0};
            DataCopyPad(yGm_[dstBase + headElems + bodyElems],
                        src[srcElemBase + localTailElems], tailParams);
        }
        return true;
    }

    __aicore__ inline bool TryBlockCountToLocalSplit(LocalTensor<DT_X> dst,
                                                     uint32_t dstElemBase,
                                                     uint32_t srcBase,
                                                     uint32_t blockCount,
                                                     uint32_t blockLenBytes,
                                                     uint32_t srcStrideBytes,
                                                     uint32_t dstStrideBytes) {
        return CopyBlockCountToLocalSplit(dst, dstElemBase, srcBase, blockCount,
                                          blockLenBytes, srcStrideBytes,
                                          dstStrideBytes);
    }

    __aicore__ inline bool TryBlockCountToGmSplit(uint32_t dstBase,
                                                  LocalTensor<DT_X> src,
                                                  uint32_t srcElemBase,
                                                  uint32_t blockCount,
                                                  uint32_t blockLenBytes,
                                                  uint32_t srcStrideBytes,
                                                  uint32_t dstStrideBytes) {
        return CopyBlockCountToGmSplit(dstBase, src, srcElemBase, blockCount,
                                       blockLenBytes, srcStrideBytes,
                                       dstStrideBytes);
    }

    __aicore__ inline void ProcessCase2LaneTasks() {
        uint32_t blockDim = tiling_.blockDim == 0 ? 1 : tiling_.blockDim;
        uint32_t blockIdx = GetBlockIdx();
        uint32_t tasksPerCore = (tiling_.taskCount + blockDim - 1) / blockDim;
        uint32_t start = blockIdx * tasksPerCore;
        if (start >= tiling_.taskCount) {
            return;
        }
        uint32_t end = start + tasksPerCore;
        if (end > tiling_.taskCount) {
            end = tiling_.taskCount;
        }
        for (uint32_t task = start; task < end; ++task) {
            ProcessCase2LaneTaskBs2(task);
        }
    }

    __aicore__ inline void ProcessCase6LaneTasks() {
        uint32_t blockDim = tiling_.blockDim == 0 ? 1 : tiling_.blockDim;
        uint32_t blockIdx = GetBlockIdx();
        uint32_t tasksPerCore = (tiling_.taskCount + blockDim - 1) / blockDim;
        uint32_t start = blockIdx * tasksPerCore;
        if (start >= tiling_.taskCount) {
            return;
        }
        uint32_t end = start + tasksPerCore;
        if (end > tiling_.taskCount) {
            end = tiling_.taskCount;
        }
        if (tiling_.blockSize != 2) {
            return;
        }
        for (uint32_t task = start; task < end; ++task) {
            ProcessCase6LaneTaskBs2(task);
        }
    }

    __aicore__ inline void ProcessCase2LaneTaskBs2(uint32_t task) {
        uint32_t chunk = task % tiling_.chunksPerLane;
        uint32_t rowLane = task / tiling_.chunksPerLane;
        uint32_t laneW = rowLane & 1;
        uint32_t outRow = rowLane >> 1;
        if (outRow >= tiling_.outBatch * tiling_.outHeight) {
            return;
        }

        uint32_t firstOutW = FirstOutWForLaneBs2(laneW, 0);
        if (firstOutW >= tiling_.outWidth) {
            return;
        }
        uint32_t laneBlocks = (tiling_.outWidth - firstOutW + 1) >> 1;
        uint32_t startBlock = chunk * tiling_.tileBlocks;
        if (startBlock >= laneBlocks) {
            return;
        }
        uint32_t copyBlocks = laneBlocks - startBlock;
        if (copyBlocks > tiling_.tileBlocks) {
            copyBlocks = tiling_.tileBlocks;
        }

        uint32_t outH = outRow % tiling_.outHeight;
        uint32_t outN = outRow / tiling_.outHeight;
        uint32_t fullH = outH + tiling_.cropTop;
        uint32_t inH = fullH >> 1;
        uint32_t blockH = fullH & 1;
        uint32_t fullW0 = firstOutW + tiling_.cropLeft;
        uint32_t inW0 = (fullW0 >> 1) + startBlock;
        uint32_t inN = ((blockH << 1) + laneW) * tiling_.outBatch + outN;
        uint32_t inBase = ((inN * tiling_.height + inH) * tiling_.width + inW0) * tiling_.depth;
        uint32_t outBase = (outRow * tiling_.outWidth + firstOutW + (startBlock << 1)) * tiling_.depth;
        CopyCase2LaneBlocksBind(inBase, outBase, copyBlocks);
    }

    __aicore__ inline void ProcessCase6LaneTaskBs2(uint32_t task) {
        uint32_t chunk = task % tiling_.chunksPerLane;
        uint32_t rowLane = task / tiling_.chunksPerLane;
        uint32_t laneW = rowLane & 1;
        uint32_t outRow = rowLane >> 1;
        if (outRow >= tiling_.outBatch * tiling_.outHeight) {
            return;
        }

        uint32_t firstOutW = FirstOutWForLaneBs2(laneW, 0);
        if (firstOutW >= tiling_.outWidth) {
            return;
        }
        uint32_t laneBlocks = (tiling_.outWidth - firstOutW + 1) >> 1;
        uint32_t startBlock = chunk * tiling_.tileBlocks;
        if (startBlock >= laneBlocks) {
            return;
        }
        uint32_t copyBlocks = laneBlocks - startBlock;
        if (copyBlocks > tiling_.tileBlocks) {
            copyBlocks = tiling_.tileBlocks;
        }

        uint32_t outH = outRow % tiling_.outHeight;
        uint32_t outN = outRow / tiling_.outHeight;
        uint32_t fullH = outH + tiling_.cropTop;
        uint32_t inH = fullH >> 1;
        uint32_t blockH = fullH & 1;
        uint32_t fullW0 = firstOutW + tiling_.cropLeft;
        uint32_t inW0 = (fullW0 >> 1) + startBlock;
        uint32_t inN = ((blockH << 1) + laneW) * tiling_.outBatch + outN;
        uint32_t inBase = ((inN * tiling_.height + inH) * tiling_.width + inW0) * tiling_.depth;
        uint32_t outBase = (outRow * tiling_.outWidth + firstOutW + (startBlock << 1)) * tiling_.depth;
        CopyCase6LaneBlocksBind(inBase, outBase, copyBlocks);
    }

    __aicore__ inline void CopyCase2LaneBlocksBind(uint32_t inBase, uint32_t outBase,
                                                   uint32_t blocks) {
        uint32_t blockLenBytes = tiling_.depth * sizeof(DT_X);
        uint32_t alignedBlockBytes = AlignUpBytes(blockLenBytes);
        LocalTensor<DT_X> xLocal = copyQueue_.template AllocTensor<DT_X>();
        if (blockLenBytes == alignedBlockBytes) {
            DataCopy(xLocal, xGm_[inBase], blocks * tiling_.depth);
        } else if (!TryBlockCountToLocalSplit(xLocal, 0, inBase, blocks,
                                              blockLenBytes, 0, 0)) {
            DataCopyExtParams copyInParams{static_cast<uint16_t>(blocks), blockLenBytes, 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, xGm_[inBase], copyInParams, padParams);
        }
        copyQueue_.EnQue(xLocal);

        LocalTensor<DT_X> yOut = copyQueue_.template DeQue<DT_X>();
        uint32_t dstStrideBytes = blockLenBytes;
        if (blockLenBytes == alignedBlockBytes &&
            CanUseDataCopyParams(blocks, blockLenBytes, 0, dstStrideBytes)) {
            DataCopyOutParams(outBase, yOut, blocks, blockLenBytes, 0, dstStrideBytes);
        } else if (!TryBlockCountToGmSplit(outBase, yOut, 0, blocks,
                                           blockLenBytes, 0, dstStrideBytes)) {
            DataCopyExtParams copyOutParams{static_cast<uint16_t>(blocks), blockLenBytes,
                                            0, dstStrideBytes, 0};
            DataCopyPad(yGm_[outBase], yOut, copyOutParams);
        }
        copyQueue_.FreeTensor(yOut);
    }

    __aicore__ inline void CopyCase6LaneBlocksBind(uint32_t inBase, uint32_t outBase,
                                                   uint32_t blocks) {
        uint32_t blockLenBytes = tiling_.depth * sizeof(DT_X);
        uint32_t alignedBlockBytes = AlignUpBytes(blockLenBytes);
        if (blockLenBytes != alignedBlockBytes &&
            TryCopyLaneDirectBind(inBase, outBase, blocks)) {
            return;
        }

        LocalTensor<DT_X> xLocal = copyQueue_.template AllocTensor<DT_X>();
        if (blockLenBytes == alignedBlockBytes) {
            DataCopy(xLocal, xGm_[inBase], blocks * tiling_.depth);
        } else if (!TryBlockCountToLocalSplit(xLocal, 0, inBase, blocks,
                                              blockLenBytes, 0, 0)) {
            DataCopyExtParams copyInParams{static_cast<uint16_t>(blocks), blockLenBytes, 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, xGm_[inBase], copyInParams, padParams);
        }
        copyQueue_.EnQue(xLocal);

        LocalTensor<DT_X> yOut = copyQueue_.template DeQue<DT_X>();
        uint32_t dstStrideBytes = blockLenBytes;
        if (blockLenBytes == alignedBlockBytes &&
            CanUseDataCopyParams(blocks, blockLenBytes, 0, dstStrideBytes)) {
            DataCopyOutParams(outBase, yOut, blocks, blockLenBytes, 0, dstStrideBytes);
        } else if (!TryBlockCountToGmSplit(outBase, yOut, 0, blocks,
                                           blockLenBytes, 0, dstStrideBytes)) {
            DataCopyExtParams copyOutParams{static_cast<uint16_t>(blocks), blockLenBytes,
                                            0, dstStrideBytes, 0};
            DataCopyPad(yGm_[outBase], yOut, copyOutParams);
        }
        copyQueue_.FreeTensor(yOut);
    }

    __aicore__ inline bool TryCopyLaneDirectBind(uint32_t inBase, uint32_t outBase,
                                                 uint32_t blocks) {
        if (blocks == 0 || blocks > SMALL_DIRECT_BLOCKS) {
            return false;
        }

        uint32_t elemBytes = sizeof(DT_X);
        uint32_t blockBytes = tiling_.depth * elemBytes;
        uint32_t localStrideBytes = AlignUpTo(blockBytes + ALIGN_BYTES, ALIGN_BYTES);
        if (localStrideBytes == 0 || localStrideBytes * blocks > ActiveCopyBufferBytes()) {
            return false;
        }

        LocalTensor<DT_X> xLocal = copyQueue_.template AllocTensor<DT_X>();
        uint32_t localStrideElems = localStrideBytes / elemBytes;
        for (uint32_t block = 0; block < blocks; ++block) {
            uint32_t inIndex = inBase + block * tiling_.depth;
            uint32_t outIndex = outBase + block * tiling_.blockSize * tiling_.depth;
            uint32_t localBase = block * localStrideElems;
            if (!CopyLinearToLocalSplit(xLocal, localBase, localStrideBytes,
                                        inIndex, outIndex, tiling_.depth)) {
                CopyLinearToLocalWhole(xLocal, localBase, inIndex, tiling_.depth);
            }
        }
        copyQueue_.EnQue(xLocal);

        LocalTensor<DT_X> yOut = copyQueue_.template DeQue<DT_X>();
        for (uint32_t block = 0; block < blocks; ++block) {
            uint32_t inIndex = inBase + block * tiling_.depth;
            uint32_t outIndex = outBase + block * tiling_.blockSize * tiling_.depth;
            uint32_t localBase = block * localStrideElems;
            if (!CopyLocalToLinearSplit(outIndex, yOut, localBase, localStrideBytes,
                                        inIndex, tiling_.depth)) {
                CopyLocalToLinearWhole(outIndex, yOut, localBase, tiling_.depth);
            }
        }
        copyQueue_.FreeTensor(yOut);
        return true;
    }

    __aicore__ inline bool BuildLinearSplitPlan(uint32_t inIndex, uint32_t outIndex,
                                                uint32_t elemCount, uint32_t localStrideBytes,
                                                uint32_t &headBytes, uint32_t &bodyBytes,
                                                uint32_t &tailBytes, uint32_t &headElems,
                                                uint32_t &bodyElems, uint32_t &tailElems,
                                                uint32_t &localBodyElems,
                                                uint32_t &localTailElems) const {
        uint32_t elemBytes = sizeof(DT_X);
        uint32_t totalBytes = elemCount * elemBytes;
        uint32_t inBytes = inIndex * elemBytes;
        uint32_t outBytes = outIndex * elemBytes;
        uint32_t inResidue = inBytes & (ALIGN_BYTES - 1);
        uint32_t outResidue = outBytes & (ALIGN_BYTES - 1);
        if (totalBytes < ALIGN_BYTES || inResidue != outResidue) {
            return false;
        }

        headBytes = inResidue == 0 ? 0 : (ALIGN_BYTES - inResidue);
        if (headBytes >= totalBytes) {
            return false;
        }
        bodyBytes = ((totalBytes - headBytes) / ALIGN_BYTES) * ALIGN_BYTES;
        if (bodyBytes == 0) {
            return false;
        }
        tailBytes = totalBytes - headBytes - bodyBytes;

        uint32_t localBodyBytes = AlignUpTo(headBytes, ALIGN_BYTES);
        uint32_t localTailBytes = localBodyBytes + bodyBytes;
        uint32_t requiredBytes = localTailBytes + AlignUpTo(tailBytes, ALIGN_BYTES);
        if (requiredBytes > localStrideBytes) {
            return false;
        }

        headElems = headBytes / elemBytes;
        bodyElems = bodyBytes / elemBytes;
        tailElems = tailBytes / elemBytes;
        localBodyElems = localBodyBytes / elemBytes;
        localTailElems = localTailBytes / elemBytes;
        return true;
    }

    __aicore__ inline bool CopyLinearToLocalSplit(LocalTensor<DT_X> xLocal,
                                                  uint32_t localBase, uint32_t localStrideBytes,
                                                  uint32_t inIndex, uint32_t outIndex,
                                                  uint32_t elemCount) {
        uint32_t headBytes;
        uint32_t bodyBytes;
        uint32_t tailBytes;
        uint32_t headElems;
        uint32_t bodyElems;
        uint32_t tailElems;
        uint32_t localBodyElems;
        uint32_t localTailElems;
        if (!BuildLinearSplitPlan(inIndex, outIndex, elemCount, localStrideBytes,
                                  headBytes, bodyBytes, tailBytes, headElems, bodyElems,
                                  tailElems, localBodyElems, localTailElems)) {
            return false;
        }

        if (headElems != 0) {
            DataCopyExtParams copyInParams{1, headBytes, 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal[localBase], xGm_[inIndex], copyInParams, padParams);
        }
        DataCopy(xLocal[localBase + localBodyElems], xGm_[inIndex + headElems], bodyElems);
        if (tailElems != 0) {
            DataCopyExtParams copyInParams{1, tailBytes, 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal[localBase + localTailElems],
                        xGm_[inIndex + headElems + bodyElems], copyInParams, padParams);
        }
        return true;
    }

    __aicore__ inline bool CopyLocalToLinearSplit(uint32_t outIndex, LocalTensor<DT_X> yOut,
                                                  uint32_t localBase, uint32_t localStrideBytes,
                                                  uint32_t inIndex, uint32_t elemCount) {
        uint32_t headBytes;
        uint32_t bodyBytes;
        uint32_t tailBytes;
        uint32_t headElems;
        uint32_t bodyElems;
        uint32_t tailElems;
        uint32_t localBodyElems;
        uint32_t localTailElems;
        if (!BuildLinearSplitPlan(inIndex, outIndex, elemCount, localStrideBytes,
                                  headBytes, bodyBytes, tailBytes, headElems, bodyElems,
                                  tailElems, localBodyElems, localTailElems)) {
            return false;
        }

        if (headElems != 0) {
            DataCopyExtParams copyOutParams{1, headBytes, 0, 0, 0};
            DataCopyPad(yGm_[outIndex], yOut[localBase], copyOutParams);
        }
        DataCopy(yGm_[outIndex + headElems], yOut[localBase + localBodyElems], bodyElems);
        if (tailElems != 0) {
            DataCopyExtParams copyOutParams{1, tailBytes, 0, 0, 0};
            DataCopyPad(yGm_[outIndex + headElems + bodyElems],
                        yOut[localBase + localTailElems], copyOutParams);
        }
        return true;
    }

    __aicore__ inline void CopyLinearToLocalWhole(LocalTensor<DT_X> xLocal, uint32_t localBase,
                                                  uint32_t inIndex, uint32_t elemCount) {
        uint32_t elemBytes = sizeof(DT_X);
        uint32_t totalBytes = elemCount * elemBytes;
        uint32_t inBytes = inIndex * elemBytes;
        if (((inBytes | totalBytes) & (ALIGN_BYTES - 1)) == 0) {
            DataCopy(xLocal[localBase], xGm_[inIndex], elemCount);
        } else {
            DataCopyExtParams copyInParams{1, totalBytes, 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal[localBase], xGm_[inIndex], copyInParams, padParams);
        }
    }

    __aicore__ inline void CopyLocalToLinearWhole(uint32_t outIndex, LocalTensor<DT_X> yOut,
                                                  uint32_t localBase, uint32_t elemCount) {
        uint32_t elemBytes = sizeof(DT_X);
        uint32_t totalBytes = elemCount * elemBytes;
        uint32_t outBytes = outIndex * elemBytes;
        if (((outBytes | totalBytes) & (ALIGN_BYTES - 1)) == 0) {
            DataCopy(yGm_[outIndex], yOut[localBase], elemCount);
        } else {
            DataCopyExtParams copyOutParams{1, totalBytes, 0, 0, 0};
            DataCopyPad(yGm_[outIndex], yOut[localBase], copyOutParams);
        }
    }

    __aicore__ inline void ProcessCase1RowTasks() {
        if (tiling_.rowChunks == 0) {
            ProcessCase1LinearTasks();
            return;
        }
        if (tiling_.rowChunks == 1) {
            ProcessRowFullWidthTasksBs2Only();
            return;
        }
        ProcessRowBucketTasksBs2Only();
    }

    __aicore__ inline void ProcessRowBucketTasksBs2Only() {
        uint32_t blockDim = tiling_.blockDim == 0 ? 1 : tiling_.blockDim;
        uint32_t blockIdx = GetBlockIdx();
        uint32_t tasksPerCore = (tiling_.taskCount + blockDim - 1) / blockDim;
        uint32_t start = blockIdx * tasksPerCore;
        if (start >= tiling_.taskCount) {
            return;
        }
        uint32_t end = start + tasksPerCore;
        if (end > tiling_.taskCount) {
            end = tiling_.taskCount;
        }
        for (uint32_t task = start; task < end; ++task) {
            ProcessCase1RowBucketTaskBs2(task);
        }
    }

    __aicore__ inline void ProcessRowFullWidthTasksBs2Only() {
        uint32_t outputRows = tiling_.outBatch * tiling_.outHeight;
        uint32_t blockDim = tiling_.blockDim == 0 ? 1 : tiling_.blockDim;
        uint32_t blockIdx = GetBlockIdx();
        uint32_t tasksPerCore = (outputRows + blockDim - 1) / blockDim;
        uint32_t start = blockIdx * tasksPerCore;
        if (start >= outputRows) {
            return;
        }
        uint32_t end = start + tasksPerCore;
        if (end > outputRows) {
            end = outputRows;
        }
        for (uint32_t outRow = start; outRow < end; ++outRow) {
            ProcessCase1RowFullWidthTaskBs2(outRow);
        }
    }

    __aicore__ inline void ProcessCase3RowTasks() {
        uint32_t blockDim = tiling_.blockDim == 0 ? 1 : tiling_.blockDim;
        uint32_t blockIdx = GetBlockIdx();
        uint32_t tasksPerCore = (tiling_.taskCount + blockDim - 1) / blockDim;
        uint32_t start = blockIdx * tasksPerCore;
        if (start >= tiling_.taskCount) {
            return;
        }
        uint32_t end = start + tasksPerCore;
        if (end > tiling_.taskCount) {
            end = tiling_.taskCount;
        }
        for (uint32_t task = start; task < end; ++task) {
            ProcessCase3NoCropAlignedPairTask(task);
        }
    }

    __aicore__ inline void ProcessCase5RowTasks() {
        if (tiling_.blockSize == 2 && tiling_.cropLeft == 1 && tiling_.cropTop == 1 &&
            tiling_.outWidth >= 224 && tiling_.outWidth <= 288 &&
            tiling_.outHeight >= 224 && tiling_.outHeight <= 288 &&
            tiling_.depth >= 32 && tiling_.depth <= 96 &&
            (tiling_.depth & 1) != 0) {
            uint32_t blockDim = tiling_.blockDim == 0 ? 1 : tiling_.blockDim;
            uint32_t blockIdx = GetBlockIdx();
            uint32_t tasksPerCore = (tiling_.taskCount + blockDim - 1) / blockDim;
            uint32_t start = blockIdx * tasksPerCore;
            if (start >= tiling_.taskCount) {
                return;
            }
            uint32_t end = start + tasksPerCore;
            if (end > tiling_.taskCount) {
                end = tiling_.taskCount;
            }
            for (uint32_t task = start; task < end; ++task) {
                ProcessCompactLaneBs2Task(task);
            }
            return;
        }
        uint32_t blockDim = tiling_.blockDim == 0 ? 1 : tiling_.blockDim;
        uint32_t blockIdx = GetBlockIdx();
        uint32_t tasksPerCore = (tiling_.taskCount + blockDim - 1) / blockDim;
        uint32_t start = blockIdx * tasksPerCore;
        if (start >= tiling_.taskCount) {
            return;
        }
        uint32_t end = start + tasksPerCore;
        if (end > tiling_.taskCount) {
            end = tiling_.taskCount;
        }
        if (tiling_.blockSize == 2) {
            for (uint32_t task = start; task < end; ++task) {
                ProcessCase5RowBucketTaskBs2(task);
            }
        } else {
            for (uint32_t task = start; task < end; ++task) {
                ProcessRowBucketTask(task);
            }
        }
    }

    __aicore__ inline void ProcessCase9RowTasks() {
        ProcessCase9ChunkTasks();
    }

    __aicore__ inline void ProcessCase9ChunkTasks() {
        uint32_t blockDim = tiling_.blockDim == 0 ? 1 : tiling_.blockDim;
        uint32_t blockIdx = GetBlockIdx();
        uint32_t tasksPerCore = (tiling_.taskCount + blockDim - 1) / blockDim;
        uint32_t start = blockIdx * tasksPerCore;
        if (start >= tiling_.taskCount) {
            return;
        }
        uint32_t end = start + tasksPerCore;
        if (end > tiling_.taskCount) {
            end = tiling_.taskCount;
        }

        for (uint32_t task = start; task < end; ++task) {
            ProcessCase9ChunkTask(task);
        }
    }

    __aicore__ inline void ProcessCase9ChunkTask(uint32_t task) {
        uint32_t outRow = 0;
        uint32_t chunk = 0;
        if (tiling_.rowChunks == 3) {
            outRow = task / 3;
            chunk = task - outRow * 3;
        } else {
            outRow = task / tiling_.rowChunks;
            chunk = task - outRow * tiling_.rowChunks;
        }
        uint32_t outputRows = tiling_.outBatch * tiling_.outHeight;
        if (outRow >= outputRows) {
            return;
        }

        LocalTensor<DT_X> tile = copyQueue_.template AllocTensor<DT_X>();
        CopyInArrangeCase9Tile(tile, outRow, chunk);
        copyQueue_.EnQue(tile);

        LocalTensor<DT_X> out = copyQueue_.template DeQue<DT_X>();
        CopyOutCase9Tile(out, outRow, chunk);
        copyQueue_.FreeTensor(out);
    }

    __aicore__ inline void CopyInArrangeCase9Tile(LocalTensor<DT_X> ubTile,
                                                  uint32_t outRow,
                                                  uint32_t chunk) {
        uint32_t owStart = chunk * CASE9_TILE_OW;
        if (owStart >= tiling_.outWidth) {
            return;
        }
        uint32_t tileOW = tiling_.outWidth - owStart;
        if (tileOW > CASE9_TILE_OW) {
            tileOW = CASE9_TILE_OW;
        }
        uint32_t outN = outRow / tiling_.outHeight;
        uint32_t outH = outRow - outN * tiling_.outHeight;
        uint32_t fullH = outH + tiling_.cropTop;
        uint32_t inH = fullH >> 2;
        uint32_t offsetH = fullH & 3;
        uint32_t depthBytes = tiling_.depth * sizeof(DT_X);
        uint32_t dstStrideBytes = (tiling_.blockSize - 1) * depthBytes;

        for (uint32_t rel = 0; rel < 4; ++rel) {
            if (rel >= tileOW) {
                break;
            }
            uint32_t fullW = owStart + rel + tiling_.cropLeft;
            uint32_t offsetW = fullW & 3;
            uint32_t groupCount = (tileOW + 3 - rel) >> 2;
            uint32_t inN = (offsetH * tiling_.blockSize + offsetW) *
                           tiling_.outBatch + outN;
            uint32_t inWStart = fullW >> 2;
            uint32_t inBase = ((inN * tiling_.height + inH) * tiling_.width + inWStart) *
                              tiling_.depth;
            uint32_t dstOffset = rel * tiling_.depth;
            DataCopyInParams(ubTile[dstOffset], inBase, groupCount,
                             depthBytes, 0, dstStrideBytes);
        }
    }

    __aicore__ inline void CopyOutCase9Tile(LocalTensor<DT_X> ubTile,
                                            uint32_t outRow,
                                            uint32_t chunk) {
        uint32_t owStart = chunk * CASE9_TILE_OW;
        if (owStart >= tiling_.outWidth) {
            return;
        }
        uint32_t tileOW = tiling_.outWidth - owStart;
        if (tileOW > CASE9_TILE_OW) {
            tileOW = CASE9_TILE_OW;
        }
        uint32_t outBase = (outRow * tiling_.outWidth + owStart) * tiling_.depth;
        DataCopy(yGm_[outBase], ubTile, tileOW * tiling_.depth);
    }

    __aicore__ inline void ProcessCase10CopyInArrangeDbTasks() {
        uint32_t blockDim = tiling_.blockDim == 0 ? 1 : tiling_.blockDim;
        uint32_t blockIdx = GetBlockIdx();
        uint32_t tasksPerCore = (tiling_.taskCount + blockDim - 1) / blockDim;
        uint32_t start = blockIdx * tasksPerCore;
        if (start >= tiling_.taskCount) {
            return;
        }
        uint32_t end = start + tasksPerCore;
        if (end > tiling_.taskCount) {
            end = tiling_.taskCount;
        }

        LocalTensor<DT_X> first = copyQueue_.template AllocTensor<DT_X>();
        CopyInArrangeCase10Tile(first, start);
        copyQueue_.EnQue(first);

        for (uint32_t task = start + 1; task < end; ++task) {
            LocalTensor<DT_X> cur = copyQueue_.template AllocTensor<DT_X>();
            CopyInArrangeCase10Tile(cur, task);
            copyQueue_.EnQue(cur);

            LocalTensor<DT_X> prev = copyQueue_.template DeQue<DT_X>();
            CopyOutCase10Tile(prev, task - 1);
            copyQueue_.FreeTensor(prev);
        }

        LocalTensor<DT_X> last = copyQueue_.template DeQue<DT_X>();
        CopyOutCase10Tile(last, end - 1);
        copyQueue_.FreeTensor(last);
    }

    __aicore__ inline void CopyInArrangeCase10Tile(LocalTensor<DT_X> ubTile,
                                                   uint32_t task) {
        uint32_t ob = 0;
        uint32_t hTile = 0;
        if (tiling_.rowChunks == 32) {
            ob = task >> 5;
            hTile = task & 31;
        } else {
            ob = task / tiling_.rowChunks;
            hTile = task - ob * tiling_.rowChunks;
        }
        uint32_t hStart = hTile * CASE10_TILE_H;
        if (ob >= tiling_.outBatch || hStart >= tiling_.outHeight) {
            return;
        }
        uint32_t tileH = tiling_.outHeight - hStart;
        if (tileH > CASE10_TILE_H) {
            tileH = CASE10_TILE_H;
        }
        uint32_t depthBytes = tiling_.depth * sizeof(DT_X);
        uint32_t dstStrideBytes = depthBytes;

        for (uint32_t rowRel = 0; rowRel < tileH; ++rowRel) {
            uint32_t outH = hStart + rowRel;
            uint32_t inH = outH >> 1;
            uint32_t offsetH = rowRel & 1;
            for (uint32_t offsetW = 0; offsetW < 2; ++offsetW) {
                uint32_t inN = ((offsetH << 1) + offsetW) * tiling_.outBatch + ob;
                uint32_t inBase = ((inN * tiling_.height + inH) * tiling_.width) *
                                  tiling_.depth;
                uint32_t dstOffset = (rowRel * tiling_.outWidth + offsetW) *
                                     tiling_.depth;
                DataCopyInParams(ubTile[dstOffset], inBase, tiling_.width,
                                 depthBytes, 0, dstStrideBytes);
            }
        }
    }

    __aicore__ inline void CopyOutCase10Tile(LocalTensor<DT_X> ubTile,
                                             uint32_t task) {
        uint32_t ob = 0;
        uint32_t hTile = 0;
        if (tiling_.rowChunks == 32) {
            ob = task >> 5;
            hTile = task & 31;
        } else {
            ob = task / tiling_.rowChunks;
            hTile = task - ob * tiling_.rowChunks;
        }
        uint32_t hStart = hTile * CASE10_TILE_H;
        if (ob >= tiling_.outBatch || hStart >= tiling_.outHeight) {
            return;
        }
        uint32_t tileH = tiling_.outHeight - hStart;
        if (tileH > CASE10_TILE_H) {
            tileH = CASE10_TILE_H;
        }
        uint32_t outBase = ((ob * tiling_.outHeight + hStart) * tiling_.outWidth) *
                           tiling_.depth;
        DataCopy(yGm_[outBase], ubTile, tileH * tiling_.outWidth * tiling_.depth);
    }

    __aicore__ inline void ProcessCase1LinearTasks() {
        uint32_t blockDim = tiling_.blockDim == 0 ? 1 : tiling_.blockDim;
        uint32_t blockIdx = GetBlockIdx();
        uint32_t tasksPerCore = (tiling_.taskCount + blockDim - 1) / blockDim;
        uint32_t start = blockIdx * tasksPerCore;
        if (start >= tiling_.taskCount) {
            return;
        }
        uint32_t end = start + tasksPerCore;
        if (end > tiling_.taskCount) {
            end = tiling_.taskCount;
        }
        uint32_t totalElems = tiling_.outBatch * tiling_.outHeight *
                              tiling_.outWidth * tiling_.depth;
        for (uint32_t task = start; task < end; ++task) {
            uint32_t elemStart = task * tiling_.tileBlocks;
            if (elemStart >= totalElems) {
                return;
            }
            uint32_t elemCount = totalElems - elemStart;
            if (elemCount > tiling_.tileBlocks) {
                elemCount = tiling_.tileBlocks;
            }
            CopyLinearBind(elemStart, elemStart, elemCount);
        }
    }

    __aicore__ inline void ProcessCase8Tasks() {
        if (tiling_.chunksPerLane == 2) {
            uint32_t blockDim = tiling_.blockDim == 0 ? 1 : tiling_.blockDim;
            uint32_t blockIdx = GetBlockIdx();
            uint32_t tasksPerCore = (tiling_.taskCount + blockDim - 1) / blockDim;
            uint32_t start = blockIdx * tasksPerCore;
            if (start >= tiling_.taskCount) {
                return;
            }
            uint32_t end = start + tasksPerCore;
            if (end > tiling_.taskCount) {
                end = tiling_.taskCount;
            }
            for (uint32_t task = start; task < end; ++task) {
                ProcessCase8RowBucketTaskBs2(task);
            }
        } else {
            uint32_t blockDim = tiling_.blockDim == 0 ? 1 : tiling_.blockDim;
            uint32_t blockIdx = GetBlockIdx();
            uint32_t tasksPerCore = (tiling_.taskCount + blockDim - 1) / blockDim;
            uint32_t start = blockIdx * tasksPerCore;
            if (start >= tiling_.taskCount) {
                return;
            }
            uint32_t end = start + tasksPerCore;
            if (end > tiling_.taskCount) {
                end = tiling_.taskCount;
            }
            ProcessRow8BucketDbTasks(start, end);
        }
    }

    __aicore__ inline void ProcessRow8BucketDbTasks(uint32_t start, uint32_t end) {
        uint32_t firstRow0 = 0;
        uint32_t firstRowCount = 0;
        uint32_t firstWStart = 0;
        uint32_t firstWCount = 0;
        if (!GetRow8BucketTask(start, firstRow0, firstRowCount, firstWStart, firstWCount)) {
            return;
        }

        LocalTensor<DT_X> first = copyQueue_.template AllocTensor<DT_X>();
        FillCase8Row8BucketBs2(first, firstRow0, firstRowCount, firstWStart, firstWCount);
        copyQueue_.EnQue(first);
        uint32_t prevRow0 = firstRow0;
        uint32_t prevRowCount = firstRowCount;
        uint32_t prevWStart = firstWStart;
        uint32_t prevWCount = firstWCount;

        for (uint32_t task = start + 1; task < end; ++task) {
            uint32_t row0 = 0;
            uint32_t rowCount = 0;
            uint32_t wStart = 0;
            uint32_t wCount = 0;
            if (!GetRow8BucketTask(task, row0, rowCount, wStart, wCount)) {
                break;
            }

            LocalTensor<DT_X> cur = copyQueue_.template AllocTensor<DT_X>();
            FillCase8Row8BucketBs2(cur, row0, rowCount, wStart, wCount);
            copyQueue_.EnQue(cur);

            LocalTensor<DT_X> prev = copyQueue_.template DeQue<DT_X>();
            CopyOutCase8Row8BucketBs2(prev, prevRow0, prevRowCount, prevWStart, prevWCount);
            copyQueue_.FreeTensor(prev);

            prevRow0 = row0;
            prevRowCount = rowCount;
            prevWStart = wStart;
            prevWCount = wCount;
        }

        LocalTensor<DT_X> last = copyQueue_.template DeQue<DT_X>();
        CopyOutCase8Row8BucketBs2(last, prevRow0, prevRowCount, prevWStart, prevWCount);
        copyQueue_.FreeTensor(last);
    }

    __aicore__ inline void ProcessCase10Tasks() {
        if (tiling_.blockSize == 2 && tiling_.cropLeft == 0 && tiling_.cropTop == 0 &&
            tiling_.outWidth == tiling_.width * 2 &&
            tiling_.outHeight == tiling_.height * 2 &&
            tiling_.outBatch >= 1 && tiling_.outBatch <= 8 &&
            tiling_.outWidth >= 8 && tiling_.outWidth <= 16 &&
            tiling_.outHeight >= 1024 && tiling_.outHeight <= 4096 &&
            tiling_.depth >= 16 && tiling_.depth <= 64 &&
            (tiling_.depth * sizeof(DT_X)) == AlignUpBytes(tiling_.depth * sizeof(DT_X)) &&
            tiling_.tileBlocks == CASE10_TILE_H &&
            tiling_.rowChunks == CeilDiv(tiling_.outHeight, CASE10_TILE_H) &&
            tiling_.taskCount == tiling_.outBatch * tiling_.rowChunks) {
            ProcessCase10CopyInArrangeDbTasks();
            return;
        }
    }

    __aicore__ inline void ProcessCase4Patch2x2Tasks() {
        uint32_t blockDim = tiling_.blockDim == 0 ? 1 : tiling_.blockDim;
        uint32_t blockIdx = GetBlockIdx();
        uint32_t tasksPerCore = (tiling_.taskCount + blockDim - 1) / blockDim;
        uint32_t start = blockIdx * tasksPerCore;
        if (start >= tiling_.taskCount) {
            return;
        }
        uint32_t end = start + tasksPerCore;
        if (end > tiling_.taskCount) {
            end = tiling_.taskCount;
        }
        for (uint32_t task = start; task < end; ++task) {
            ProcessCase4Patch2x2Task(task);
        }
    }

    __aicore__ inline void ProcessCase4Patch2x2Task(uint32_t task) {
        uint32_t chunk = task % tiling_.rowChunks;
        uint32_t base = task / tiling_.rowChunks;
        uint32_t inH = base % tiling_.height;
        uint32_t outN = base / tiling_.height;
        if (outN >= tiling_.outBatch) {
            return;
        }
        uint32_t inWStart = chunk * tiling_.tileBlocks;
        if (inWStart >= tiling_.width) {
            return;
        }
        uint32_t inWCount = tiling_.width - inWStart;
        if (inWCount > tiling_.tileBlocks) {
            inWCount = tiling_.tileBlocks;
        }
        CopyCase4Patch2x2Bind(outN, inH, inWStart, inWCount);
    }

    __aicore__ inline void ProcessCompactLaneBs2Task(uint32_t task) {
        uint32_t chunk = task % tiling_.rowChunks;
        uint32_t outRow = task / tiling_.rowChunks;
        if (outRow >= tiling_.outBatch * tiling_.outHeight) {
            return;
        }

        uint32_t outWStart = chunk * tiling_.tileBlocks;
        uint32_t outWEnd = outWStart + tiling_.tileBlocks;
        if (outWEnd > tiling_.outWidth) {
            outWEnd = tiling_.outWidth;
        }
        uint32_t outWCount = outWEnd - outWStart;
        if (outWCount == 0) {
            return;
        }

        uint32_t tmp = outRow;
        uint32_t outH = tmp % tiling_.outHeight;
        uint32_t outN = tmp / tiling_.outHeight;
        uint32_t fullH = outH + tiling_.cropTop;
        uint32_t inH = fullH >> 1;
        uint32_t blockH = fullH & 1;
        CopyCompactLaneBucketBs2(outRow, outWStart, outWEnd, outN, inH, blockH);
    }

    __aicore__ inline void ProcessRowBucketTask(uint32_t task) {
        uint32_t chunk = task % tiling_.rowChunks;
        uint32_t outRow = task / tiling_.rowChunks;
        if (outRow >= tiling_.outBatch * tiling_.outHeight) {
            return;
        }

        uint32_t outWStart = chunk * tiling_.tileBlocks;
        uint32_t outWEnd = outWStart + tiling_.tileBlocks;
        if (outWEnd > tiling_.outWidth) {
            outWEnd = tiling_.outWidth;
        }
        uint32_t outWCount = outWEnd - outWStart;

        uint32_t tmp = outRow;
        uint32_t outH = tmp % tiling_.outHeight;
        uint32_t outN = tmp / tiling_.outHeight;
        uint32_t fullH = outH + tiling_.cropTop;
        uint32_t inH = fullH / tiling_.blockSize;
        uint32_t blockH = fullH - inH * tiling_.blockSize;

        CopyRowBucket(outRow, outWStart, outWCount, outN, inH, blockH);
    }


    __aicore__ inline void ProcessCase1RowBucketTaskBs2(uint32_t task) {
        uint32_t chunk = task % tiling_.rowChunks;
        uint32_t outRow = task / tiling_.rowChunks;
        if (outRow >= tiling_.outBatch * tiling_.outHeight) {
            return;
        }

        uint32_t outWStart = chunk * tiling_.tileBlocks;
        uint32_t outWEnd = outWStart + tiling_.tileBlocks;
        if (outWEnd > tiling_.outWidth) {
            outWEnd = tiling_.outWidth;
        }
        uint32_t outWCount = outWEnd - outWStart;

        uint32_t tmp = outRow;
        uint32_t outH = tmp % tiling_.outHeight;
        uint32_t outN = tmp / tiling_.outHeight;
        uint32_t fullH = outH + tiling_.cropTop;
        uint32_t inH = fullH >> 1;
        uint32_t blockH = fullH & 1;

        CopyCase1RowBucketBs2Bind(outRow, outWStart, outWCount, outN, inH, blockH);
    }

    __aicore__ inline void ProcessCase5RowBucketTaskBs2(uint32_t task) {
        uint32_t chunk = task % tiling_.rowChunks;
        uint32_t outRow = task / tiling_.rowChunks;
        if (outRow >= tiling_.outBatch * tiling_.outHeight) {
            return;
        }

        uint32_t outWStart = chunk * tiling_.tileBlocks;
        uint32_t outWEnd = outWStart + tiling_.tileBlocks;
        if (outWEnd > tiling_.outWidth) {
            outWEnd = tiling_.outWidth;
        }
        uint32_t outWCount = outWEnd - outWStart;

        uint32_t tmp = outRow;
        uint32_t outH = tmp % tiling_.outHeight;
        uint32_t outN = tmp / tiling_.outHeight;
        uint32_t fullH = outH + tiling_.cropTop;
        uint32_t inH = fullH >> 1;
        uint32_t blockH = fullH & 1;

        CopyCase5RowBucketBs2Bind(outRow, outWStart, outWCount, outN, inH, blockH);
    }

    __aicore__ inline void ProcessCase8RowBucketTaskBs2(uint32_t task) {
        uint32_t chunk = task % tiling_.rowChunks;
        uint32_t outRow = task / tiling_.rowChunks;
        if (outRow >= tiling_.outBatch * tiling_.outHeight) {
            return;
        }

        uint32_t outWStart = chunk * tiling_.tileBlocks;
        uint32_t outWEnd = outWStart + tiling_.tileBlocks;
        if (outWEnd > tiling_.outWidth) {
            outWEnd = tiling_.outWidth;
        }
        uint32_t outWCount = outWEnd - outWStart;

        uint32_t tmp = outRow;
        uint32_t outH = tmp % tiling_.outHeight;
        uint32_t outN = tmp / tiling_.outHeight;
        uint32_t fullH = outH + tiling_.cropTop;
        uint32_t inH = fullH >> 1;
        uint32_t blockH = fullH & 1;

        CopyCase8RowBucketBs2Bind(outRow, outWStart, outWCount, outN, inH, blockH);
    }

    __aicore__ inline void ProcessCase3NoCropAlignedPairTask(uint32_t task) {
        uint32_t chunk = task % tiling_.rowChunks;
        uint32_t rowPair = task / tiling_.rowChunks;
        uint32_t outRow0 = rowPair << 1;
        uint32_t outputRows = tiling_.outBatch * tiling_.outHeight;
        if (outRow0 >= outputRows) {
            return;
        }
        uint32_t rowCount = (outRow0 + 1 < outputRows) ? 2 : 1;
        uint32_t outWStart = chunk * tiling_.tileBlocks;
        if (outWStart >= tiling_.outWidth) {
            return;
        }
        uint32_t outWEnd = outWStart + tiling_.tileBlocks;
        if (outWEnd > tiling_.outWidth) {
            outWEnd = tiling_.outWidth;
        }
        uint32_t outWCount = outWEnd - outWStart;
        if (rowCount == 2 && tiling_.blockSize == 2 && tiling_.cropTop == 0 &&
            tiling_.cropLeft == 0 && (outWStart & 1) == 0 && (outWCount & 1) == 0 &&
            CopyCase3NoCropAlignedPairBs2(outRow0, outWStart, outWCount)) {
            return;
        }
        CopyCase3RowPairBucketBs2Bind(outRow0, rowCount, outWStart, outWCount);
    }

    __aicore__ inline void ProcessCase1RowFullWidthTaskBs2(uint32_t outRow) {
        uint32_t tmp = outRow;
        uint32_t outH = tmp % tiling_.outHeight;
        uint32_t outN = tmp / tiling_.outHeight;
        uint32_t fullH = outH + tiling_.cropTop;
        uint32_t inH = fullH >> 1;
        uint32_t blockH = fullH & 1;
        CopyCase1RowBucketBs2Bind(outRow, 0, tiling_.outWidth, outN, inH, blockH);
    }


    __aicore__ inline uint32_t FirstOutWForLane(uint32_t laneW, uint32_t outWStart) const {
        uint32_t cropMod = tiling_.cropLeft % tiling_.blockSize;
        uint32_t first = laneW >= cropMod ?
            (laneW - cropMod) : (laneW + tiling_.blockSize - cropMod);
        if (first < outWStart) {
            first += CeilDiv(outWStart - first, tiling_.blockSize) * tiling_.blockSize;
        }
        return first;
    }

    __aicore__ inline uint32_t FirstOutWForLaneBs2(uint32_t laneW, uint32_t outWStart) const {
        uint32_t cropMod = tiling_.cropLeft & 1;
        uint32_t first = laneW >= cropMod ? (laneW - cropMod) : (laneW + 2 - cropMod);
        if (first < outWStart) {
            uint32_t delta = outWStart - first;
            first += ((delta + 1) >> 1) << 1;
        }
        return first;
    }

    __aicore__ inline void CopyRowBucket(uint32_t outRow, uint32_t outWStart, uint32_t outWCount,
                                         uint32_t outN, uint32_t inH, uint32_t blockH) {
        uint32_t blockLenBytes = tiling_.depth * sizeof(DT_X);
        uint32_t alignedBlockBytes = AlignUpBytes(blockLenBytes);
        uint32_t alignedElemsPerBlock = alignedBlockBytes / sizeof(DT_X);
        uint32_t totalElems = outWCount * alignedElemsPerBlock;

        LocalTensor<DT_X> xLocal = inQueue_.template AllocTensor<DT_X>();
        for (uint32_t laneW = 0; laneW < tiling_.blockSize; ++laneW) {
            uint32_t firstOutW = FirstOutWForLane(laneW, outWStart);
            uint32_t outWEnd = outWStart + outWCount;
            if (firstOutW >= outWEnd) {
                continue;
            }
            uint32_t blocks = (outWEnd - firstOutW + tiling_.blockSize - 1) / tiling_.blockSize;
            uint32_t fullW0 = firstOutW + tiling_.cropLeft;
            uint32_t inW0 = fullW0 / tiling_.blockSize;
            uint32_t inN = (blockH * tiling_.blockSize + laneW) * tiling_.outBatch + outN;
            uint32_t inBase = ((inN * tiling_.height + inH) * tiling_.width + inW0) * tiling_.depth;
            uint32_t dstOffset = (firstOutW - outWStart) * alignedElemsPerBlock;
            uint32_t dstStride = (tiling_.blockSize - 1) * alignedBlockBytes / ALIGN_BYTES;
            uint32_t dstStrideBytes = dstStride * ALIGN_BYTES;
            if (blockLenBytes == alignedBlockBytes &&
                CanUseDataCopyParams(blocks, blockLenBytes, 0, dstStrideBytes)) {
                DataCopyInParams(xLocal[dstOffset], inBase, blocks, blockLenBytes,
                                 0, dstStrideBytes);
            } else if (!TryBlockCountToLocalSplit(xLocal, dstOffset, inBase, blocks,
                                                  blockLenBytes, 0, dstStrideBytes)) {
                DataCopyExtParams copyInParams{static_cast<uint16_t>(blocks), blockLenBytes, 0, dstStride, 0};
                DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
                DataCopyPad(xLocal[dstOffset], xGm_[inBase], copyInParams, padParams);
            }
        }
        inQueue_.EnQue(xLocal);

        LocalTensor<DT_X> xIn = inQueue_.template DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue_.template AllocTensor<DT_X>();
        Adds(yLocal, xIn, static_cast<DT_X>(0), totalElems);
        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xIn);

        LocalTensor<DT_X> yOut = outQueue_.template DeQue<DT_X>();
        uint32_t outBase = (outRow * tiling_.outWidth + outWStart) * tiling_.depth;
        if (blockLenBytes == alignedBlockBytes) {
            DataCopy(yGm_[outBase], yOut, outWCount * tiling_.depth);
        } else if (!TryBlockCountToGmSplit(outBase, yOut, 0, outWCount,
                                           blockLenBytes, 0, 0)) {
            DataCopyExtParams copyOutParams{static_cast<uint16_t>(outWCount), blockLenBytes, 0, 0, 0};
            DataCopyPad(yGm_[outBase], yOut, copyOutParams);
        }
        outQueue_.FreeTensor(yOut);
    }




    __aicore__ inline void CopyCase1RowBucketBs2Bind(uint32_t outRow, uint32_t outWStart,
                                                     uint32_t outWCount, uint32_t outN,
                                                     uint32_t inH, uint32_t blockH) {
        uint32_t blockLenBytes = tiling_.depth * sizeof(DT_X);
        uint32_t alignedBlockBytes = AlignUpBytes(blockLenBytes);
        uint32_t alignedElemsPerBlock = alignedBlockBytes / sizeof(DT_X);
        uint32_t outWEnd = outWStart + outWCount;

        LocalTensor<DT_X> xLocal = copyQueue_.template AllocTensor<DT_X>();
        CopyCase1RowLaneBs2(xLocal, 0, outWStart, outWEnd, outN, inH, blockH,
                            blockLenBytes, alignedBlockBytes, alignedElemsPerBlock);
        CopyCase1RowLaneBs2(xLocal, 1, outWStart, outWEnd, outN, inH, blockH,
                            blockLenBytes, alignedBlockBytes, alignedElemsPerBlock);
        copyQueue_.EnQue(xLocal);

        LocalTensor<DT_X> yOut = copyQueue_.template DeQue<DT_X>();
        uint32_t outBase = (outRow * tiling_.outWidth + outWStart) * tiling_.depth;
        if (blockLenBytes == alignedBlockBytes) {
            DataCopy(yGm_[outBase], yOut, outWCount * tiling_.depth);
        } else if (!TryBlockCountToGmSplit(outBase, yOut, 0, outWCount,
                                           blockLenBytes, 0, 0)) {
            DataCopyExtParams copyOutParams{static_cast<uint16_t>(outWCount),
                                            blockLenBytes, 0, 0, 0};
            DataCopyPad(yGm_[outBase], yOut, copyOutParams);
        }
        copyQueue_.FreeTensor(yOut);
    }

    __aicore__ inline void CopyCase5RowBucketBs2Bind(uint32_t outRow, uint32_t outWStart,
                                                     uint32_t outWCount, uint32_t outN,
                                                     uint32_t inH, uint32_t blockH) {
        uint32_t blockLenBytes = tiling_.depth * sizeof(DT_X);
        uint32_t alignedBlockBytes = AlignUpBytes(blockLenBytes);
        uint32_t alignedElemsPerBlock = alignedBlockBytes / sizeof(DT_X);
        uint32_t outWEnd = outWStart + outWCount;

        LocalTensor<DT_X> xLocal = copyQueue_.template AllocTensor<DT_X>();
        CopyCase5RowLaneBs2(xLocal, 0, outWStart, outWEnd, outN, inH, blockH,
                            blockLenBytes, alignedBlockBytes, alignedElemsPerBlock);
        CopyCase5RowLaneBs2(xLocal, 1, outWStart, outWEnd, outN, inH, blockH,
                            blockLenBytes, alignedBlockBytes, alignedElemsPerBlock);
        copyQueue_.EnQue(xLocal);

        LocalTensor<DT_X> yOut = copyQueue_.template DeQue<DT_X>();
        uint32_t outBase = (outRow * tiling_.outWidth + outWStart) * tiling_.depth;
        if (blockLenBytes == alignedBlockBytes) {
            DataCopy(yGm_[outBase], yOut, outWCount * tiling_.depth);
        } else if (!TryBlockCountToGmSplit(outBase, yOut, 0, outWCount,
                                           blockLenBytes, 0, 0)) {
            DataCopyExtParams copyOutParams{static_cast<uint16_t>(outWCount),
                                            blockLenBytes, 0, 0, 0};
            DataCopyPad(yGm_[outBase], yOut, copyOutParams);
        }
        copyQueue_.FreeTensor(yOut);
    }

    __aicore__ inline void CopyCase8RowBucketBs2Bind(uint32_t outRow, uint32_t outWStart,
                                                     uint32_t outWCount, uint32_t outN,
                                                     uint32_t inH, uint32_t blockH) {
        uint32_t blockLenBytes = tiling_.depth * sizeof(DT_X);
        uint32_t alignedBlockBytes = AlignUpBytes(blockLenBytes);
        uint32_t alignedElemsPerBlock = alignedBlockBytes / sizeof(DT_X);
        uint32_t outWEnd = outWStart + outWCount;

        LocalTensor<DT_X> xLocal = copyQueue_.template AllocTensor<DT_X>();
        CopyCase8RowLaneBs2(xLocal, 0, outWStart, outWEnd, outN, inH, blockH,
                            blockLenBytes, alignedBlockBytes, alignedElemsPerBlock);
        CopyCase8RowLaneBs2(xLocal, 1, outWStart, outWEnd, outN, inH, blockH,
                            blockLenBytes, alignedBlockBytes, alignedElemsPerBlock);
        copyQueue_.EnQue(xLocal);

        LocalTensor<DT_X> yOut = copyQueue_.template DeQue<DT_X>();
        uint32_t outBase = (outRow * tiling_.outWidth + outWStart) * tiling_.depth;
        if (blockLenBytes == alignedBlockBytes) {
            DataCopy(yGm_[outBase], yOut, outWCount * tiling_.depth);
        } else if (!TryBlockCountToGmSplit(outBase, yOut, 0, outWCount,
                                           blockLenBytes, 0, 0)) {
            DataCopyExtParams copyOutParams{static_cast<uint16_t>(outWCount),
                                            blockLenBytes, 0, 0, 0};
            DataCopyPad(yGm_[outBase], yOut, copyOutParams);
        }
        copyQueue_.FreeTensor(yOut);
    }





    __aicore__ inline void CopyCase3RowPairBucketBs2Bind(uint32_t outRow0, uint32_t rowCount,
                                                         uint32_t outWStart,
                                                         uint32_t outWCount) {
        uint32_t blockLenBytes = tiling_.depth * sizeof(DT_X);
        uint32_t alignedBlockBytes = blockLenBytes;
        uint32_t alignedElemsPerBlock = tiling_.depth;
        uint32_t rowElems = outWCount * tiling_.depth;
        uint32_t outWEnd = outWStart + outWCount;

        LocalTensor<DT_X> xLocal = copyQueue_.template AllocTensor<DT_X>();
        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t outRow = outRow0 + r;
            uint32_t tmp = outRow;
            uint32_t outH = tmp % tiling_.outHeight;
            uint32_t outN = tmp / tiling_.outHeight;
            uint32_t fullH = outH + tiling_.cropTop;
            uint32_t inH = fullH >> 1;
            uint32_t blockH = fullH & 1;
            uint32_t rowOffset = r * rowElems;
            CopyCase3RowLaneBs2(xLocal, 0, outWStart, outWEnd, outN, inH, blockH,
                                blockLenBytes, alignedBlockBytes, alignedElemsPerBlock,
                                rowOffset);
            CopyCase3RowLaneBs2(xLocal, 1, outWStart, outWEnd, outN, inH, blockH,
                                blockLenBytes, alignedBlockBytes, alignedElemsPerBlock,
                                rowOffset);
        }
        copyQueue_.EnQue(xLocal);

        LocalTensor<DT_X> yOut = copyQueue_.template DeQue<DT_X>();
        uint32_t outBase = (outRow0 * tiling_.outWidth + outWStart) * tiling_.depth;
        if (rowCount == 1) {
            DataCopy(yGm_[outBase], yOut, rowElems);
        } else if (outWStart == 0 && outWCount == tiling_.outWidth) {
            DataCopy(yGm_[outBase], yOut, rowElems * rowCount);
        } else {
            uint32_t dstStrideBytes = (tiling_.outWidth - outWCount) * blockLenBytes;
            uint32_t rowBytes = rowElems * sizeof(DT_X);
            if (CanUseDataCopyParams(rowCount, rowBytes, 0, dstStrideBytes)) {
                DataCopyOutParams(outBase, yOut, rowCount, rowBytes, 0, dstStrideBytes);
            } else if (!TryBlockCountToGmSplit(outBase, yOut, 0, rowCount,
                                               rowBytes, 0, dstStrideBytes)) {
                DataCopyExtParams copyOutParams{static_cast<uint16_t>(rowCount),
                                                rowBytes, 0, dstStrideBytes, 0};
                DataCopyPad(yGm_[outBase], yOut, copyOutParams);
            }
        }
        copyQueue_.FreeTensor(yOut);
    }

    __aicore__ inline bool CopyCase3NoCropAlignedPairBs2(uint32_t outRow0,
                                                         uint32_t outWStart,
                                                         uint32_t outWCount) {
        uint32_t blockLenBytes = tiling_.depth * sizeof(DT_X);
        if (blockLenBytes == 0 || blockLenBytes != AlignUpBytes(blockLenBytes)) {
            return false;
        }
        uint32_t laneBlocks = outWCount >> 1;
        if (laneBlocks == 0) {
            return false;
        }
        uint32_t rowElems = outWCount * tiling_.depth;
        uint32_t totalElems = rowElems << 1;
        if (totalElems * sizeof(DT_X) > ActiveCopyBufferBytes()) {
            return false;
        }
        uint32_t rowBytes = rowElems * sizeof(DT_X);
        uint32_t outBase = (outRow0 * tiling_.outWidth + outWStart) * tiling_.depth;
        uint32_t outBase1 = outBase + tiling_.outWidth * tiling_.depth;
        uint32_t outBytes0 = outBase * sizeof(DT_X);
        uint32_t outBytes1 = outBase1 * sizeof(DT_X);
        if (((rowBytes | outBytes0 | outBytes1) & (ALIGN_BYTES - 1)) != 0) {
            return false;
        }

        LocalTensor<DT_X> xLocal = copyQueue_.template AllocTensor<DT_X>();
        CopyCase3NoCropPairRowToLocal(xLocal, outRow0, outWStart, outWCount, 0);
        CopyCase3NoCropPairRowToLocal(xLocal, outRow0 + 1, outWStart, outWCount, rowElems);
        copyQueue_.EnQue(xLocal);

        LocalTensor<DT_X> yOut = copyQueue_.template DeQue<DT_X>();
        if (outWStart == 0 && outWCount == tiling_.outWidth) {
            DataCopy(yGm_[outBase], yOut, rowElems << 1);
        } else {
            DataCopy(yGm_[outBase], yOut, rowElems);
            DataCopy(yGm_[outBase1], yOut[rowElems], rowElems);
        }
        copyQueue_.FreeTensor(yOut);
        return true;
    }

    __aicore__ inline void CopyCase3NoCropPairRowToLocal(LocalTensor<DT_X> xLocal,
                                                         uint32_t outRow,
                                                         uint32_t outWStart,
                                                         uint32_t outWCount,
                                                         uint32_t rowOffset) {
        uint32_t outH = outRow % tiling_.outHeight;
        uint32_t outN = outRow / tiling_.outHeight;
        uint32_t inH = outH >> 1;
        uint32_t blockH = outH & 1;
        uint32_t inW0 = outWStart >> 1;
        uint32_t laneBlocks = outWCount >> 1;
        uint32_t blockLenBytes = tiling_.depth * sizeof(DT_X);
        uint32_t dstStrideBytes = blockLenBytes;

        uint32_t inN0 = (blockH << 1) * tiling_.outBatch + outN;
        uint32_t inBase0 = ((inN0 * tiling_.height + inH) * tiling_.width + inW0) * tiling_.depth;
        DataCopyInParams(xLocal[rowOffset], inBase0, laneBlocks, blockLenBytes, 0, dstStrideBytes);

        uint32_t inN1 = inN0 + tiling_.outBatch;
        uint32_t inBase1 = ((inN1 * tiling_.height + inH) * tiling_.width + inW0) * tiling_.depth;
        DataCopyInParams(xLocal[rowOffset + tiling_.depth], inBase1, laneBlocks,
                         blockLenBytes, 0, dstStrideBytes);
    }

    __aicore__ inline bool GetRow8BucketTask(uint32_t task, uint32_t &outRow0,
                                             uint32_t &rowCount, uint32_t &outWStart,
                                             uint32_t &outWCount) {
        uint32_t chunk = task % tiling_.rowChunks;
        uint32_t rowGroup = task / tiling_.rowChunks;
        outRow0 = rowGroup << 3;
        uint32_t outputRows = tiling_.outBatch * tiling_.outHeight;
        if (outRow0 >= outputRows) {
            return false;
        }

        rowCount = outputRows - outRow0;
        if (rowCount > 8) {
            rowCount = 8;
        }

        outWStart = chunk * tiling_.tileBlocks;
        uint32_t outWEnd = outWStart + tiling_.tileBlocks;
        if (outWEnd > tiling_.outWidth) {
            outWEnd = tiling_.outWidth;
        }
        outWCount = outWEnd - outWStart;
        return outWCount != 0;
    }


    __aicore__ inline void FillCase8Row8BucketBs2(LocalTensor<DT_X> xLocal,
                                                  uint32_t outRow0, uint32_t rowCount,
                                                  uint32_t outWStart,
                                                  uint32_t outWCount) {
        uint32_t blockLenBytes = tiling_.depth * sizeof(DT_X);
        uint32_t alignedBlockBytes = blockLenBytes;
        uint32_t alignedElemsPerBlock = tiling_.depth;
        uint32_t rowElems = outWCount * tiling_.depth;
        uint32_t outWEnd = outWStart + outWCount;

        for (uint32_t r = 0; r < rowCount; ++r) {
            uint32_t outRow = outRow0 + r;
            uint32_t tmp = outRow;
            uint32_t outH = tmp % tiling_.outHeight;
            uint32_t outN = tmp / tiling_.outHeight;
            uint32_t fullH = outH + tiling_.cropTop;
            uint32_t inH = fullH >> 1;
            uint32_t blockH = fullH & 1;
            uint32_t rowOffset = r * rowElems;
            CopyCase8RowLaneBs2(xLocal, 0, outWStart, outWEnd, outN, inH, blockH,
                                blockLenBytes, alignedBlockBytes, alignedElemsPerBlock,
                                rowOffset);
            CopyCase8RowLaneBs2(xLocal, 1, outWStart, outWEnd, outN, inH, blockH,
                                blockLenBytes, alignedBlockBytes, alignedElemsPerBlock,
                                rowOffset);
        }
    }


    __aicore__ inline void CopyOutCase8Row8BucketBs2(LocalTensor<DT_X> yOut,
                                                     uint32_t outRow0, uint32_t rowCount,
                                                     uint32_t outWStart,
                                                     uint32_t outWCount) {
        uint32_t blockLenBytes = tiling_.depth * sizeof(DT_X);
        uint32_t rowElems = outWCount * tiling_.depth;
        uint32_t outBase = (outRow0 * tiling_.outWidth + outWStart) * tiling_.depth;
        if (rowCount == 1) {
            DataCopy(yGm_[outBase], yOut, rowElems);
        } else {
            uint32_t dstStrideBytes = (tiling_.outWidth - outWCount) * blockLenBytes;
            uint32_t rowBytes = rowElems * sizeof(DT_X);
            if (CanUseDataCopyParams(rowCount, rowBytes, 0, dstStrideBytes)) {
                DataCopyOutParams(outBase, yOut, rowCount, rowBytes, 0, dstStrideBytes);
            } else if (!TryBlockCountToGmSplit(outBase, yOut, 0, rowCount,
                                               rowBytes, 0, dstStrideBytes)) {
                DataCopyExtParams copyOutParams{static_cast<uint16_t>(rowCount),
                                                rowBytes, 0, dstStrideBytes, 0};
                DataCopyPad(yGm_[outBase], yOut, copyOutParams);
            }
        }
    }



    __aicore__ inline void CopyCompactLaneBucketBs2(uint32_t outRow, uint32_t outWStart,
                                                    uint32_t outWEnd, uint32_t outN,
                                                    uint32_t inH, uint32_t blockH) {
        uint32_t blockLenBytes = tiling_.depth * sizeof(DT_X);
        uint32_t alignedBlockBytes = AlignUpBytes(blockLenBytes);
        uint32_t alignedElemsPerBlock = alignedBlockBytes / sizeof(DT_X);
        uint32_t maxLaneBlocks = ((outWEnd - outWStart) + 1) >> 1;
        if (maxLaneBlocks == 0) {
            return;
        }
        uint32_t laneCapacityElems = maxLaneBlocks * alignedElemsPerBlock;
        uint32_t requiredElems = laneCapacityElems << 1;
        if (requiredElems * sizeof(DT_X) > ActiveCopyBufferBytes()) {
            CopyCase5RowBucketBs2Bind(outRow, outWStart, outWEnd - outWStart, outN, inH, blockH);
            return;
        }

        LocalTensor<DT_X> xLocal = copyQueue_.template AllocTensor<DT_X>();
        uint32_t lane0Blocks = CopyCompactLaneToLocalBs2(xLocal, 0, 0, outWStart, outWEnd,
                                                         outN, inH, blockH, blockLenBytes,
                                                         alignedBlockBytes);
        uint32_t lane1Blocks = CopyCompactLaneToLocalBs2(xLocal, 1, laneCapacityElems, outWStart,
                                                         outWEnd, outN, inH, blockH,
                                                         blockLenBytes, alignedBlockBytes);
        copyQueue_.EnQue(xLocal);

        LocalTensor<DT_X> yOut = copyQueue_.template DeQue<DT_X>();
        CopyCompactLaneToGmBs2(yOut, 0, 0, lane0Blocks, outRow, outWStart, blockLenBytes);
        CopyCompactLaneToGmBs2(yOut, 1, laneCapacityElems, lane1Blocks, outRow, outWStart,
                               blockLenBytes);
        copyQueue_.FreeTensor(yOut);
    }

    __aicore__ inline uint32_t CopyCompactLaneToLocalBs2(LocalTensor<DT_X> xLocal,
                                                         uint32_t laneW,
                                                         uint32_t localOffset,
                                                         uint32_t outWStart,
                                                         uint32_t outWEnd,
                                                         uint32_t outN, uint32_t inH,
                                                         uint32_t blockH,
                                                         uint32_t blockLenBytes,
                                                         uint32_t alignedBlockBytes) {
        uint32_t firstOutW = FirstOutWForLaneBs2(laneW, outWStart);
        if (firstOutW >= outWEnd) {
            return 0;
        }
        uint32_t blocks = (outWEnd - firstOutW + 1) >> 1;
        uint32_t fullW0 = firstOutW + tiling_.cropLeft;
        uint32_t inW0 = fullW0 >> 1;
        uint32_t inN = ((blockH << 1) + laneW) * tiling_.outBatch + outN;
        uint32_t inBase = ((inN * tiling_.height + inH) * tiling_.width + inW0) * tiling_.depth;
        if (blockLenBytes == alignedBlockBytes) {
            DataCopy(xLocal[localOffset], xGm_[inBase], blocks * tiling_.depth);
        } else if (!TryBlockCountToLocalSplit(xLocal, localOffset, inBase, blocks,
                                              blockLenBytes, 0, 0)) {
            DataCopyExtParams copyInParams{static_cast<uint16_t>(blocks), blockLenBytes, 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal[localOffset], xGm_[inBase], copyInParams, padParams);
        }
        return blocks;
    }

    __aicore__ inline void CopyCompactLaneToGmBs2(LocalTensor<DT_X> yOut, uint32_t laneW,
                                                  uint32_t localOffset, uint32_t blocks,
                                                  uint32_t outRow, uint32_t outWStart,
                                                  uint32_t blockLenBytes) {
        if (blocks == 0) {
            return;
        }
        uint32_t firstOutW = FirstOutWForLaneBs2(laneW, outWStart);
        uint32_t outBase = (outRow * tiling_.outWidth + firstOutW) * tiling_.depth;
        uint32_t dstStrideBytes = blockLenBytes;
        if (((blockLenBytes | dstStrideBytes) & (ALIGN_BYTES - 1)) == 0 &&
            CanUseDataCopyParams(blocks, blockLenBytes, 0, dstStrideBytes)) {
            DataCopyOutParams(outBase, yOut[localOffset], blocks, blockLenBytes,
                              0, dstStrideBytes);
        } else if (!TryBlockCountToGmSplit(outBase, yOut, localOffset, blocks,
                                           blockLenBytes, 0, dstStrideBytes)) {
            DataCopyExtParams copyOutParams{static_cast<uint16_t>(blocks), blockLenBytes,
                                            0, dstStrideBytes, 0};
            DataCopyPad(yGm_[outBase], yOut[localOffset], copyOutParams);
        }
    }

    __aicore__ inline void CopyCase4Patch2x2Bind(uint32_t outN, uint32_t inH,
                                                 uint32_t inWStart, uint32_t inWCount) {
        uint32_t blockLenBytes = tiling_.depth * sizeof(DT_X);
        uint32_t alignedBlockBytes = blockLenBytes;
        uint32_t alignedElemsPerBlock = tiling_.depth;
        uint32_t rowOutWCount = inWCount << 1;
        uint32_t rowElems = rowOutWCount * tiling_.depth;
        uint32_t totalElems = rowElems << 1;

        LocalTensor<DT_X> xLocal = copyQueue_.template AllocTensor<DT_X>();
        uint32_t row1Offset = rowElems;
        uint32_t lane1Offset = alignedElemsPerBlock;
        CopyCase4PatchLane2x2(xLocal, outN, inH, inWStart, inWCount, 0,
                              blockLenBytes, alignedBlockBytes);
        CopyCase4PatchLane2x2(xLocal, tiling_.outBatch + outN, inH, inWStart, inWCount,
                              lane1Offset, blockLenBytes, alignedBlockBytes);
        CopyCase4PatchLane2x2(xLocal, (tiling_.outBatch << 1) + outN, inH, inWStart, inWCount,
                              row1Offset, blockLenBytes, alignedBlockBytes);
        CopyCase4PatchLane2x2(xLocal, tiling_.outBatch * 3 + outN, inH, inWStart, inWCount,
                              row1Offset + lane1Offset, blockLenBytes, alignedBlockBytes);
        copyQueue_.EnQue(xLocal);

        LocalTensor<DT_X> yOut = copyQueue_.template DeQue<DT_X>();
        uint32_t outRow0 = (outN * tiling_.outHeight) + (inH << 1);
        uint32_t outWStart = inWStart << 1;
        uint32_t outBase = (outRow0 * tiling_.outWidth + outWStart) * tiling_.depth;
        if (inWStart == 0 && inWCount == tiling_.width) {
            DataCopy(yGm_[outBase], yOut, totalElems);
        } else {
            uint32_t dstStrideBytes = (tiling_.outWidth - rowOutWCount) * blockLenBytes;
            uint32_t rowBytes = rowElems * sizeof(DT_X);
            if (CanUseDataCopyParams(2, rowBytes, 0, dstStrideBytes)) {
                DataCopyOutParams(outBase, yOut, 2, rowBytes, 0, dstStrideBytes);
            } else if (!TryBlockCountToGmSplit(outBase, yOut, 0, 2,
                                               rowBytes, 0, dstStrideBytes)) {
                DataCopyExtParams copyOutParams{static_cast<uint16_t>(2),
                                                rowBytes, 0, dstStrideBytes, 0};
                DataCopyPad(yGm_[outBase], yOut, copyOutParams);
            }
        }
        copyQueue_.FreeTensor(yOut);
    }

    __aicore__ inline void CopyCase4PatchLane2x2(LocalTensor<DT_X> xLocal, uint32_t inN,
                                                 uint32_t inH, uint32_t inWStart,
                                                 uint32_t inWCount, uint32_t dstOffset,
                                                 uint32_t blockLenBytes,
                                                 uint32_t alignedBlockBytes) {
        uint32_t inBase = ((inN * tiling_.height + inH) * tiling_.width + inWStart) * tiling_.depth;
        uint32_t dstStrideBytes = alignedBlockBytes;
        if (blockLenBytes == alignedBlockBytes &&
            CanUseDataCopyParams(inWCount, blockLenBytes, 0, dstStrideBytes)) {
            DataCopyInParams(xLocal[dstOffset], inBase, inWCount, blockLenBytes,
                             0, dstStrideBytes);
        } else if (!TryBlockCountToLocalSplit(xLocal, dstOffset, inBase, inWCount,
                                              blockLenBytes, 0, dstStrideBytes)) {
            uint32_t dstStride = alignedBlockBytes / ALIGN_BYTES;
            DataCopyExtParams copyInParams{static_cast<uint16_t>(inWCount),
                                           blockLenBytes, 0, dstStride, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal[dstOffset], xGm_[inBase], copyInParams, padParams);
        }
    }


    __aicore__ inline void CopyCase1RowLaneBs2(LocalTensor<DT_X> xLocal, uint32_t laneW,
                                               uint32_t outWStart, uint32_t outWEnd,
                                               uint32_t outN, uint32_t inH,
                                               uint32_t blockH,
                                               uint32_t blockLenBytes,
                                               uint32_t alignedBlockBytes,
                                               uint32_t alignedElemsPerBlock,
                                               uint32_t baseOffset = 0) {
        uint32_t firstOutW = FirstOutWForLaneBs2(laneW, outWStart);
        if (firstOutW >= outWEnd) {
            return;
        }
        uint32_t blocks = (outWEnd - firstOutW + 1) >> 1;
        uint32_t fullW0 = firstOutW + tiling_.cropLeft;
        uint32_t inW0 = fullW0 >> 1;
        uint32_t inN = ((blockH << 1) + laneW) * tiling_.outBatch + outN;
        uint32_t inBase = ((inN * tiling_.height + inH) * tiling_.width + inW0) * tiling_.depth;
        uint32_t dstOffset = baseOffset + (firstOutW - outWStart) * alignedElemsPerBlock;
        uint32_t dstStride = alignedBlockBytes / ALIGN_BYTES;
        uint32_t dstStrideBytes = dstStride * ALIGN_BYTES;
        if (blockLenBytes == alignedBlockBytes &&
            CanUseDataCopyParams(blocks, blockLenBytes, 0, dstStrideBytes)) {
            DataCopyInParams(xLocal[dstOffset], inBase, blocks, blockLenBytes, 0, dstStrideBytes);
        } else if (!TryBlockCountToLocalSplit(xLocal, dstOffset, inBase, blocks,
                                              blockLenBytes, 0, dstStrideBytes)) {
            DataCopyExtParams copyInParams{static_cast<uint16_t>(blocks),
                                           blockLenBytes, 0, dstStride, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal[dstOffset], xGm_[inBase], copyInParams, padParams);
        }
    }

    __aicore__ inline void CopyCase3RowLaneBs2(LocalTensor<DT_X> xLocal, uint32_t laneW,
                                               uint32_t outWStart, uint32_t outWEnd,
                                               uint32_t outN, uint32_t inH,
                                               uint32_t blockH,
                                               uint32_t blockLenBytes,
                                               uint32_t alignedBlockBytes,
                                               uint32_t alignedElemsPerBlock,
                                               uint32_t baseOffset = 0) {
        uint32_t firstOutW = FirstOutWForLaneBs2(laneW, outWStart);
        if (firstOutW >= outWEnd) {
            return;
        }
        uint32_t blocks = (outWEnd - firstOutW + 1) >> 1;
        uint32_t fullW0 = firstOutW + tiling_.cropLeft;
        uint32_t inW0 = fullW0 >> 1;
        uint32_t inN = ((blockH << 1) + laneW) * tiling_.outBatch + outN;
        uint32_t inBase = ((inN * tiling_.height + inH) * tiling_.width + inW0) * tiling_.depth;
        uint32_t dstOffset = baseOffset + (firstOutW - outWStart) * alignedElemsPerBlock;
        uint32_t dstStride = alignedBlockBytes / ALIGN_BYTES;
        uint32_t dstStrideBytes = dstStride * ALIGN_BYTES;
        if (blockLenBytes == alignedBlockBytes &&
            CanUseDataCopyParams(blocks, blockLenBytes, 0, dstStrideBytes)) {
            DataCopyInParams(xLocal[dstOffset], inBase, blocks, blockLenBytes, 0, dstStrideBytes);
        } else if (!TryBlockCountToLocalSplit(xLocal, dstOffset, inBase, blocks,
                                              blockLenBytes, 0, dstStrideBytes)) {
            DataCopyExtParams copyInParams{static_cast<uint16_t>(blocks),
                                           blockLenBytes, 0, dstStride, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal[dstOffset], xGm_[inBase], copyInParams, padParams);
        }
    }

    __aicore__ inline void CopyCase5RowLaneBs2(LocalTensor<DT_X> xLocal, uint32_t laneW,
                                               uint32_t outWStart, uint32_t outWEnd,
                                               uint32_t outN, uint32_t inH,
                                               uint32_t blockH,
                                               uint32_t blockLenBytes,
                                               uint32_t alignedBlockBytes,
                                               uint32_t alignedElemsPerBlock,
                                               uint32_t baseOffset = 0) {
        uint32_t firstOutW = FirstOutWForLaneBs2(laneW, outWStart);
        if (firstOutW >= outWEnd) {
            return;
        }
        uint32_t blocks = (outWEnd - firstOutW + 1) >> 1;
        uint32_t fullW0 = firstOutW + tiling_.cropLeft;
        uint32_t inW0 = fullW0 >> 1;
        uint32_t inN = ((blockH << 1) + laneW) * tiling_.outBatch + outN;
        uint32_t inBase = ((inN * tiling_.height + inH) * tiling_.width + inW0) * tiling_.depth;
        uint32_t dstOffset = baseOffset + (firstOutW - outWStart) * alignedElemsPerBlock;
        uint32_t dstStride = alignedBlockBytes / ALIGN_BYTES;
        uint32_t dstStrideBytes = dstStride * ALIGN_BYTES;
        if (blockLenBytes == alignedBlockBytes &&
            CanUseDataCopyParams(blocks, blockLenBytes, 0, dstStrideBytes)) {
            DataCopyInParams(xLocal[dstOffset], inBase, blocks, blockLenBytes, 0, dstStrideBytes);
        } else if (!TryBlockCountToLocalSplit(xLocal, dstOffset, inBase, blocks,
                                              blockLenBytes, 0, dstStrideBytes)) {
            DataCopyExtParams copyInParams{static_cast<uint16_t>(blocks),
                                           blockLenBytes, 0, dstStride, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal[dstOffset], xGm_[inBase], copyInParams, padParams);
        }
    }

    __aicore__ inline void CopyCase8RowLaneBs2(LocalTensor<DT_X> xLocal, uint32_t laneW,
                                               uint32_t outWStart, uint32_t outWEnd,
                                               uint32_t outN, uint32_t inH,
                                               uint32_t blockH,
                                               uint32_t blockLenBytes,
                                               uint32_t alignedBlockBytes,
                                               uint32_t alignedElemsPerBlock,
                                               uint32_t baseOffset = 0) {
        uint32_t firstOutW = FirstOutWForLaneBs2(laneW, outWStart);
        if (firstOutW >= outWEnd) {
            return;
        }
        uint32_t blocks = (outWEnd - firstOutW + 1) >> 1;
        uint32_t fullW0 = firstOutW + tiling_.cropLeft;
        uint32_t inW0 = fullW0 >> 1;
        uint32_t inN = ((blockH << 1) + laneW) * tiling_.outBatch + outN;
        uint32_t inBase = ((inN * tiling_.height + inH) * tiling_.width + inW0) * tiling_.depth;
        uint32_t dstOffset = baseOffset + (firstOutW - outWStart) * alignedElemsPerBlock;
        uint32_t dstStride = alignedBlockBytes / ALIGN_BYTES;
        uint32_t dstStrideBytes = dstStride * ALIGN_BYTES;
        if (blockLenBytes == alignedBlockBytes &&
            CanUseDataCopyParams(blocks, blockLenBytes, 0, dstStrideBytes)) {
            DataCopyInParams(xLocal[dstOffset], inBase, blocks, blockLenBytes, 0, dstStrideBytes);
        } else if (!TryBlockCountToLocalSplit(xLocal, dstOffset, inBase, blocks,
                                              blockLenBytes, 0, dstStrideBytes)) {
            DataCopyExtParams copyInParams{static_cast<uint16_t>(blocks),
                                           blockLenBytes, 0, dstStride, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal[dstOffset], xGm_[inBase], copyInParams, padParams);
        }
    }

    __aicore__ inline bool IsAlignedSegment(uint32_t inIndex, uint32_t outIndex, uint32_t count) const {
        uint32_t bytes = count * sizeof(DT_X);
        uint32_t inBytes = inIndex * sizeof(DT_X);
        uint32_t outBytes = outIndex * sizeof(DT_X);
        return ((bytes | inBytes | outBytes) & (ALIGN_BYTES - 1)) == 0;
    }

    __aicore__ inline void CopyLinearBind(uint32_t inIndex, uint32_t outIndex, uint32_t elemCount) {
        if (elemCount == 0) {
            return;
        }
        bool aligned = IsAlignedSegment(inIndex, outIndex, elemCount);
        if (!aligned && CopyLinearBindAlignedBody(inIndex, outIndex, elemCount)) {
            return;
        }

        LocalTensor<DT_X> xLocal = copyQueue_.template AllocTensor<DT_X>();
        if (aligned) {
            DataCopy(xLocal, xGm_[inIndex], elemCount);
        } else {
            DataCopyExtParams copyInParams{1, static_cast<uint32_t>(elemCount * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, xGm_[inIndex], copyInParams, padParams);
        }
        copyQueue_.EnQue(xLocal);

        LocalTensor<DT_X> yOut = copyQueue_.template DeQue<DT_X>();
        if (aligned) {
            DataCopy(yGm_[outIndex], yOut, elemCount);
        } else {
            DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(elemCount * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(yGm_[outIndex], yOut, copyOutParams);
        }
        copyQueue_.FreeTensor(yOut);
    }

    __aicore__ inline bool CopyLinearBindAlignedBody(uint32_t inIndex, uint32_t outIndex,
                                                     uint32_t elemCount) {
        uint32_t elemBytes = sizeof(DT_X);
        uint32_t totalBytes = elemCount * elemBytes;
        uint32_t inBytes = inIndex * elemBytes;
        uint32_t outBytes = outIndex * elemBytes;
        uint32_t inResidue = inBytes & (ALIGN_BYTES - 1);
        uint32_t outResidue = outBytes & (ALIGN_BYTES - 1);
        if (totalBytes < ALIGN_BYTES || inResidue != outResidue) {
            return false;
        }

        uint32_t headBytes = inResidue == 0 ? 0 : (ALIGN_BYTES - inResidue);
        if (headBytes >= totalBytes) {
            return false;
        }
        uint32_t bodyBytes = ((totalBytes - headBytes) / ALIGN_BYTES) * ALIGN_BYTES;
        if (bodyBytes == 0) {
            return false;
        }
        uint32_t tailBytes = totalBytes - headBytes - bodyBytes;
        uint32_t localBodyBytes = AlignUpTo(headBytes, ALIGN_BYTES);
        uint32_t localTailBytes = localBodyBytes + bodyBytes;
        uint32_t requiredBytes = localTailBytes + AlignUpTo(tailBytes, ALIGN_BYTES);
        if (requiredBytes > ActiveCopyBufferBytes()) {
            return false;
        }

        uint32_t headElems = headBytes / elemBytes;
        uint32_t bodyElems = bodyBytes / elemBytes;
        uint32_t tailElems = tailBytes / elemBytes;
        uint32_t localBodyElems = localBodyBytes / elemBytes;
        uint32_t localTailElems = localTailBytes / elemBytes;

        LocalTensor<DT_X> xLocal = copyQueue_.template AllocTensor<DT_X>();
        if (headElems != 0) {
            DataCopyExtParams copyInParams{1, headBytes, 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, xGm_[inIndex], copyInParams, padParams);
        }
        DataCopy(xLocal[localBodyElems], xGm_[inIndex + headElems], bodyElems);
        if (tailElems != 0) {
            DataCopyExtParams copyInParams{1, tailBytes, 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal[localTailElems], xGm_[inIndex + headElems + bodyElems],
                        copyInParams, padParams);
        }
        copyQueue_.EnQue(xLocal);

        LocalTensor<DT_X> yOut = copyQueue_.template DeQue<DT_X>();
        if (headElems != 0) {
            DataCopyExtParams copyOutParams{1, headBytes, 0, 0, 0};
            DataCopyPad(yGm_[outIndex], yOut, copyOutParams);
        }
        DataCopy(yGm_[outIndex + headElems], yOut[localBodyElems], bodyElems);
        if (tailElems != 0) {
            DataCopyExtParams copyOutParams{1, tailBytes, 0, 0, 0};
            DataCopyPad(yGm_[outIndex + headElems + bodyElems], yOut[localTailElems],
                        copyOutParams);
        }
        copyQueue_.FreeTensor(yOut);
        return true;
    }

    __aicore__ inline void CopyCase7LinearBind(uint32_t inIndex, uint32_t outIndex,
                                               uint32_t elemCount) {
        if (elemCount == 0) {
            return;
        }
        bool aligned = IsAlignedSegment(inIndex, outIndex, elemCount);
        if (!aligned && CopyCase7LinearBindAlignedBody(inIndex, outIndex, elemCount)) {
            return;
        }

        LocalTensor<DT_X> xLocal = copyQueue_.template AllocTensor<DT_X>();
        if (aligned) {
            DataCopy(xLocal, xGm_[inIndex], elemCount);
        } else {
            DataCopyExtParams copyInParams{1,
                                           static_cast<uint32_t>(elemCount * sizeof(DT_X)),
                                           0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, xGm_[inIndex], copyInParams, padParams);
        }
        copyQueue_.EnQue(xLocal);

        LocalTensor<DT_X> yOut = copyQueue_.template DeQue<DT_X>();
        if (aligned) {
            DataCopy(yGm_[outIndex], yOut, elemCount);
        } else {
            DataCopyExtParams copyOutParams{1,
                                            static_cast<uint32_t>(elemCount * sizeof(DT_X)),
                                            0, 0, 0};
            DataCopyPad(yGm_[outIndex], yOut, copyOutParams);
        }
        copyQueue_.FreeTensor(yOut);
    }

    __aicore__ inline bool CopyCase7LinearBindAlignedBody(uint32_t inIndex,
                                                          uint32_t outIndex,
                                                          uint32_t elemCount) {
        uint32_t elemBytes = sizeof(DT_X);
        uint32_t totalBytes = elemCount * elemBytes;
        uint32_t inBytes = inIndex * elemBytes;
        uint32_t outBytes = outIndex * elemBytes;
        uint32_t inResidue = inBytes & (ALIGN_BYTES - 1);
        uint32_t outResidue = outBytes & (ALIGN_BYTES - 1);
        if (totalBytes < ALIGN_BYTES || inResidue != outResidue) {
            return false;
        }

        uint32_t headBytes = inResidue == 0 ? 0 : (ALIGN_BYTES - inResidue);
        if (headBytes >= totalBytes) {
            return false;
        }
        uint32_t bodyBytes = ((totalBytes - headBytes) / ALIGN_BYTES) * ALIGN_BYTES;
        if (bodyBytes == 0) {
            return false;
        }
        uint32_t tailBytes = totalBytes - headBytes - bodyBytes;
        uint32_t localBodyBytes = AlignUpTo(headBytes, ALIGN_BYTES);
        uint32_t localTailBytes = localBodyBytes + bodyBytes;
        uint32_t requiredBytes = localTailBytes + AlignUpTo(tailBytes, ALIGN_BYTES);
        if (requiredBytes > ActiveCopyBufferBytes()) {
            return false;
        }

        uint32_t headElems = headBytes / elemBytes;
        uint32_t bodyElems = bodyBytes / elemBytes;
        uint32_t tailElems = tailBytes / elemBytes;
        uint32_t localBodyElems = localBodyBytes / elemBytes;
        uint32_t localTailElems = localTailBytes / elemBytes;

        LocalTensor<DT_X> xLocal = copyQueue_.template AllocTensor<DT_X>();
        if (headElems != 0) {
            DataCopyExtParams copyInParams{1, headBytes, 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, xGm_[inIndex], copyInParams, padParams);
        }
        DataCopy(xLocal[localBodyElems], xGm_[inIndex + headElems], bodyElems);
        if (tailElems != 0) {
            DataCopyExtParams copyInParams{1, tailBytes, 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal[localTailElems],
                        xGm_[inIndex + headElems + bodyElems],
                        copyInParams, padParams);
        }
        copyQueue_.EnQue(xLocal);

        LocalTensor<DT_X> yOut = copyQueue_.template DeQue<DT_X>();
        if (headElems != 0) {
            DataCopyExtParams copyOutParams{1, headBytes, 0, 0, 0};
            DataCopyPad(yGm_[outIndex], yOut, copyOutParams);
        }
        DataCopy(yGm_[outIndex + headElems], yOut[localBodyElems], bodyElems);
        if (tailElems != 0) {
            DataCopyExtParams copyOutParams{1, tailBytes, 0, 0, 0};
            DataCopyPad(yGm_[outIndex + headElems + bodyElems],
                        yOut[localTailElems], copyOutParams);
        }
        copyQueue_.FreeTensor(yOut);
        return true;
    }

    __aicore__ inline void CopyDepthSegmentBind(uint32_t inBase, uint32_t outBase) {
        uint32_t offset = 0;
        while (offset < tiling_.depth) {
            uint32_t count = tiling_.depth - offset;
            uint32_t maxElems = TILE_BYTES / sizeof(DT_X);
            if (count > maxElems) {
                count = maxElems;
            }
            CopyLinearBind(inBase + offset, outBase + offset, count);
            offset += count;
        }
    }

    __aicore__ inline uint32_t InputBaseForOutputPixelFast(uint32_t outPixel) const {
        uint32_t tmp = outPixel;
        uint32_t outW = tmp % tiling_.outWidth;
        tmp /= tiling_.outWidth;
        uint32_t outH = tmp % tiling_.outHeight;
        uint32_t outN = tmp / tiling_.outHeight;

        uint32_t fullH = outH + tiling_.cropTop;
        uint32_t fullW = outW + tiling_.cropLeft;
        uint32_t inH;
        uint32_t inW;
        uint32_t blockH;
        uint32_t blockW;
        if (tiling_.blockSize == 2) {
            inH = fullH >> 1;
            inW = fullW >> 1;
            blockH = fullH & 1;
            blockW = fullW & 1;
        } else if (tiling_.blockSize == 1) {
            inH = fullH;
            inW = fullW;
            blockH = 0;
            blockW = 0;
        } else {
            inH = fullH / tiling_.blockSize;
            inW = fullW / tiling_.blockSize;
            blockH = fullH - inH * tiling_.blockSize;
            blockW = fullW - inW * tiling_.blockSize;
        }
        uint32_t inN = (blockH * tiling_.blockSize + blockW) * tiling_.outBatch + outN;
        return ((inN * tiling_.height + inH) * tiling_.width + inW) * tiling_.depth;
    }

    __aicore__ inline void ProcessCase7DepthTasks(uint32_t outputPixels) {
        ProcessDepthMicroTasks(outputPixels);
    }

    __aicore__ inline void ProcessDepthMicroTasks(uint32_t outputPixels) {
        uint32_t depthChunks = tiling_.rowChunks == 0 ? 1 : tiling_.rowChunks;
        uint32_t blockDim = tiling_.blockDim == 0 ? 1 : tiling_.blockDim;
        uint32_t blockIdx = GetBlockIdx();
        uint32_t taskCount = tiling_.taskCount == 0 ? outputPixels : tiling_.taskCount;
        uint32_t tasksPerCore = (taskCount + blockDim - 1) / blockDim;
        uint32_t start = blockIdx * tasksPerCore;
        if (start >= taskCount) {
            return;
        }
        uint32_t end = start + tasksPerCore;
        if (end > taskCount) {
            end = taskCount;
        }
        uint32_t chunkElems = tiling_.tileBlocks == 0 ? tiling_.depth : tiling_.tileBlocks;
        for (uint32_t task = start; task < end; ++task) {
            uint32_t outPixel = task / depthChunks;
            if (outPixel >= outputPixels) {
                return;
            }
            uint32_t depthChunk = task - outPixel * depthChunks;
            uint32_t depthOffset = depthChunk * chunkElems;
            if (depthOffset >= tiling_.depth) {
                continue;
            }
            uint32_t count = tiling_.depth - depthOffset;
            if (count > chunkElems) {
                count = chunkElems;
            }
            uint32_t inBase = InputBaseForOutputPixelFast(outPixel) + depthOffset;
            uint32_t outBase = outPixel * tiling_.depth + depthOffset;
            CopyCase7LinearBind(inBase, outBase, count);
        }
    }

    __aicore__ inline void ProcessOutputPixels(uint32_t start, uint32_t end) {
        for (uint32_t outPixel = start; outPixel < end; ++outPixel) {
            uint32_t tmp = outPixel;
            uint32_t outW = tmp % tiling_.outWidth;
            tmp /= tiling_.outWidth;
            uint32_t outH = tmp % tiling_.outHeight;
            uint32_t outN = tmp / tiling_.outHeight;

            uint32_t fullH = outH + tiling_.cropTop;
            uint32_t fullW = outW + tiling_.cropLeft;
            uint32_t inH = fullH / tiling_.blockSize;
            uint32_t inW = fullW / tiling_.blockSize;
            uint32_t blockH = fullH - inH * tiling_.blockSize;
            uint32_t blockW = fullW - inW * tiling_.blockSize;
            uint32_t inN = (blockH * tiling_.blockSize + blockW) * tiling_.outBatch + outN;

            uint32_t inBase = ((inN * tiling_.height + inH) * tiling_.width + inW) * tiling_.depth;
            uint32_t outBase = outPixel * tiling_.depth;
            CopyDepthSegment(inBase, outBase);
        }
    }

    __aicore__ inline void CopyDepthSegment(uint32_t inBase, uint32_t outBase) {
        if constexpr (kCase7MicroMode) {
            CopyDepthSegmentBind(inBase, outBase);
            return;
        }
        uint32_t offset = 0;
        while (offset < tiling_.depth) {
            uint32_t count = tiling_.depth - offset;
            uint32_t maxElems = TILE_BYTES / sizeof(DT_X);
            if (count > maxElems) {
                count = maxElems;
            }

            uint32_t inIndex = inBase + offset;
            uint32_t outIndex = outBase + offset;
            bool aligned = IsAlignedSegment(inIndex, outIndex, count);
            LocalTensor<DT_X> xLocal = inQueue_.template AllocTensor<DT_X>();
            if (aligned) {
                DataCopy(xLocal, xGm_[inIndex], count);
            } else {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
                DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
                DataCopyPad(xLocal, xGm_[inIndex], copyParams, padParams);
            }
            inQueue_.EnQue(xLocal);

            LocalTensor<DT_X> xIn = inQueue_.template DeQue<DT_X>();
            LocalTensor<DT_X> yLocal = outQueue_.template AllocTensor<DT_X>();
            Adds(yLocal, xIn, static_cast<DT_X>(0), count);
            outQueue_.EnQue(yLocal);
            inQueue_.FreeTensor(xIn);

            LocalTensor<DT_X> yOut = outQueue_.template DeQue<DT_X>();
            if (aligned) {
                DataCopy(yGm_[outIndex], yOut, count);
            } else {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
                DataCopyPad(yGm_[outIndex], yOut, copyParams);
            }
            outQueue_.FreeTensor(yOut);
            offset += count;
        }
    }

private:
    BatchToSpaceTilingData tiling_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    // Template depth is queue dependency depth, not double buffer depth.
    // p9/p10 enqueue the next tile before dequeuing the previous tile, so they
    // require depth=2. All other modes keep the lighter depth=1 path.
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, kCopyQueueDepth> copyQueue_;
};

template <typename DT_X, uint32_t MODE_T>
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    InitSocState();
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingData, tilingData, tiling);
    TPipe pipe;
    KernelBatchToSpace<DT_X, MODE_T> op;
    op.Init(x, y, tilingData, &pipe);
    op.Process();
}
