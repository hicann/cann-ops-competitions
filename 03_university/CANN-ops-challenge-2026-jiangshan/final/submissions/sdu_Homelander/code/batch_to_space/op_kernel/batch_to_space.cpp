#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"

#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

template <class DT_X>
class KernelBatchToSpace {
public:
    __aicore__ inline KernelBatchToSpace() {}

    __aicore__ inline void InitAndProcess(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling) {
        xGm.SetGlobalBuffer((__gm__ DT_X *)x, tiling.inputLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, tiling.outputLength);

        height = tiling.height;
        width = tiling.width;
        depth = tiling.depth;

        blockSize = tiling.blockSize;

        cropTop = tiling.cropTop;
        cropBottom = tiling.cropBottom;
        cropLeft = tiling.cropLeft;
        cropRight = tiling.cropRight;

        outBatch = tiling.outBatch;
        outHeight = tiling.outHeight;
        outWidth = tiling.outWidth;

        outputLength = tiling.outputLength;
        rowTaskNum = tiling.rowTaskNum;

        // 第7点专用：width=1 + large aligned depth + DirectFlag。
        // 注意：这里仍然保留 inQueue 的 BUFFER_NUM=2。
        // 原因是 CopyOut(local -> GM) 是异步的，如果 inQueue 只给 1 块 UB，
        // 下一次 CopyIn(GM -> local) 可能过早复用同一块 UB，覆盖尚未写回完成的数据，导致答案错误。
        // 该路径只去掉 outQueue 初始化和 UB->UB 中转，不压掉 inQueue 双缓冲。
        if (CanUseBlock2NoCropWidth1LargeDepthAlignedDirect()) {
            pipe.InitBuffer(inQueue, BUFFER_NUM, POINT7_TILE_ELEMS * sizeof(DT_X));
            ProcessBlock2NoCropWidth1LargeDepthAlignedDirect();
            return;
        }

        pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_ELEMS * sizeof(DT_X));
        pipe.InitBuffer(outQueue, BUFFER_NUM, TILE_ELEMS * sizeof(DT_X));

        if (CanUseBlock2NoCropPairGroupAssembleAligned()) {    //第10个点
            ProcessBlock2NoCropPairGroupAssembleAligned();
            return;
        }

        if (CanUseBlock2NoCropTwoRowsAssembleAligned()) {     //第1、3、4、8个点
            ProcessBlock2NoCropTwoRowsAssembleAligned();
            return;
        }

        if (blockSize == 4) {    //第9个点：blockSize=4 + crop + aligned 专用
            ProcessBlock4CropMultiRowsAssembleAligned();
            return;
        }

        if (CanUseBlock2CropColumnChunkNonAlignedPadded()) { //第5个点：blockSize=2 + crop + non-aligned 专用
            // Gather 版本需要 uint32 offset 表；只在第5点路径初始化，避免影响其它点 UB。
            pipe.InitBuffer(gatherOffsetBuf, GATHER_OFFSET_ELEMS * sizeof(uint32_t));
            ProcessBlock2CropColumnChunkNonAlignedGather();
            return;
        }

        if (CanUseGeneralNoCropAssembleAligned()) {    //第6个点
            ProcessGeneralNoCropAssembleAligned();
            return;
        }

        //第2个点：blockSize=2 + 小 crop 专用。
        // 第2点已确认为 non-aligned：使用 Gather offset 表做 UB 内 interleave，GM 侧保持连续搬入/连续写出。
        pipe.InitBuffer(gatherOffsetBuf, GATHER_OFFSET_ELEMS * sizeof(uint32_t));
        ProcessSmallBlock2CropFast();
        
    }

private:
    static constexpr uint32_t BUFFER_NUM = 2;
    static constexpr uint32_t TILE_ELEMS = 8192;
    // 第7点专用 tile：第7点已测为 width=1，depth 在 (8192, 16384]，
    // 少创建 outQueue 后保留 inQueue 双缓冲，把单 buffer 提到 16384，
    // 让每个 task 从 2 个 tile 降到 1 个 tile。
    static constexpr uint32_t POINT7_TILE_ELEMS = 16384;
    // 第5点历史测试 CHUNK_COLS_NONALIGNED=32 更优；Gather 路径也需要控制 offset 表大小。
    static constexpr uint32_t CHUNK_COLS_NONALIGNED = 32;
    static constexpr uint32_t GATHER_OFFSET_ELEMS = 8192;

    __aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) {
        return a < b ? a : b;
    }

    __aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
        return (a + b - 1) / b;
    }

    __aicore__ inline uint32_t AlignUpU32(uint32_t x, uint32_t align) {
        return (x + align - 1) / align * align;
    }

    __aicore__ inline bool CanUseNonAlignedPaddedLocal() {
        const uint32_t depthBytes = depth * sizeof(DT_X);
        if ((depthBytes & 31) == 0) {
            return false;
        }

        const uint32_t alignElems = 32 / sizeof(DT_X);
        return AlignUpU32(depth, alignElems) <= TILE_ELEMS;
    }

    __aicore__ inline uint32_t CalcNonAlignedChunkCols() {
        const uint32_t depthBytes = depth * sizeof(DT_X);
        uint32_t g = depthBytes & (0U - depthBytes);
        if (g > 32) {
            g = 32;
        }
        uint32_t colsAlign = 32 / g;

        uint32_t chunkCols = CHUNK_COLS_NONALIGNED;
        if (chunkCols < colsAlign) {
            chunkCols = colsAlign;
        }

        chunkCols = chunkCols / colsAlign * colsAlign;
        if (chunkCols == 0) {
            chunkCols = colsAlign;
        }

        return chunkCols;
    }


    __aicore__ inline bool CanUseBlock2NoCropPairGroupAssembleAligned() {
        if (cropTop != 0 || cropBottom != 0 || cropLeft != 0 || cropRight != 0) {
            return false;
        }

        const uint64_t totalBytes = static_cast<uint64_t>(outputLength) * sizeof(DT_X);
        if (totalBytes <= 4ULL * 1024ULL * 1024ULL) {
            return false;
        }

        const uint32_t depthBytes = depth * sizeof(DT_X);
        if ((depthBytes & 31) != 0) {
            return false;
        }

        const uint32_t depthBlocks = depthBytes >> 5;
        if (depthBlocks > 65535 || width > 4095) {
            return false;
        }

        const uint32_t pairElems = width * 4 * depth;
        return (pairElems <= TILE_ELEMS) && ((TILE_ELEMS / pairElems) >= 2);
    }

    __aicore__ inline bool CanUseBlock2CropColumnChunkNonAlignedPadded() {
        if (cropTop == 0 && cropBottom == 0 && cropLeft == 0 && cropRight == 0) {
            return false;
        }

        const uint64_t totalBytes = static_cast<uint64_t>(outputLength) * sizeof(DT_X);
        if (totalBytes <= 4ULL * 1024ULL * 1024ULL) {
            return false;
        }

        return CanUseNonAlignedPaddedLocal();
    }

    __aicore__ inline bool CanUseBlock2NoCropTwoRowsAssembleAligned() {
        if (blockSize != 2) {
            return false;
        }

        if (cropTop != 0 || cropBottom != 0 || cropLeft != 0 || cropRight != 0) {
            return false;
        }

        const uint32_t depthBytes = depth * sizeof(DT_X);
        if ((depthBytes & 31) != 0) {
            return false;
        }

        const uint32_t depthBlocks = depthBytes >> 5;
        if (depthBlocks > 65535) {
            return false;
        }

        return (depth << 2) <= TILE_ELEMS;
    }

    __aicore__ inline bool CanUseBlock2NoCropWidth1LargeDepthAlignedDirect() {
        if (blockSize != 2) {
            return false;
        }

        if (cropTop != 0 || cropBottom != 0 || cropLeft != 0 || cropRight != 0) {
            return false;
        }

        // 第7点已测特征：width=1，depth 32B 对齐，且 depth > TILE_ELEMS。
        // 这种场景不能走 GeneralNoCropAssembleAligned，也不适配 small fast/combined 路线。
        if (width != 1 || depth <= TILE_ELEMS || depth > POINT7_TILE_ELEMS) {
            return false;
        }

        const uint32_t depthBytes = depth * sizeof(DT_X);
        if ((depthBytes & 31) != 0) {
            return false;
        }

        const uint32_t depthBlocks = depthBytes >> 5;
        if (depthBlocks == 0 || depthBlocks > 65535) {
            return false;
        }

        // 标准 blockSize=2 no-crop fine 映射：每个 (ob,h) 有 4 个 blockIndex task。
        return rowTaskNum == outBatch * height * 4;
    }

    __aicore__ inline bool CanUseGeneralNoCropAssembleAligned() {
        if (cropTop != 0 || cropBottom != 0 || cropLeft != 0 || cropRight != 0) {
            return false;
        }

        const uint32_t depthBytes = depth * sizeof(DT_X);
        if ((depthBytes & 31) != 0) {
            return false;
        }

        const uint32_t depthBlocks = depthBytes >> 5;
        if (depthBlocks > 65535) {
            return false;
        }

        return (depth << 1) <= TILE_ELEMS;
    }

    __aicore__ inline void CopySpanAlignedDirectFlag(
        uint64_t srcOffset,
        uint64_t dstOffset,
        uint32_t elemCount,
        int32_t eventIdMte2ToMte3
    ) {
        uint32_t remain = elemCount;
        uint64_t src = srcOffset;
        uint64_t dst = dstOffset;

        while (remain > 0) {
            const uint32_t cur = remain > POINT7_TILE_ELEMS ? POINT7_TILE_ELEMS : remain;

            AscendC::LocalTensor<DT_X> local = inQueue.AllocTensor<DT_X>();
            AscendC::DataCopy(local, xGm[src], cur);

            // 直接用同一个 UB 从 MTE2 读后交给 MTE3 写，避免 CopySpanAligned 中的
            // inLocal -> outLocal UB 中转和 outQueue 管理开销。
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);

            AscendC::DataCopy(yGm[dst], local, cur);
            inQueue.FreeTensor(local);

            src += cur;
            dst += cur;
            remain -= cur;
        }
    }


    __aicore__ inline void CopyWidthStridedNonAlignedPaddedLocal(
        uint64_t srcBase,
        uint64_t dstBase,
        uint32_t wCount
    ) {
        const uint32_t depthBytes = depth * sizeof(DT_X);
        const uint32_t alignElems = 32 / sizeof(DT_X);
        const uint32_t alignedDepthElems = AlignUpU32(depth, alignElems);
        const uint32_t dstStepElems = depth << 1;
        const uint32_t dstStrideBytes = depthBytes;

        uint32_t maxBlocksByUb = TILE_ELEMS / alignedDepthElems;
        if (maxBlocksByUb > 4095) {
            maxBlocksByUb = 4095;
        }

        uint32_t remainBlocks = wCount;
        uint32_t curWOffset = 0;

        uint32_t curBlocks = remainBlocks > maxBlocksByUb ? maxBlocksByUb : remainBlocks;
        uint64_t srcOffset = srcBase + static_cast<uint64_t>(curWOffset) * depth;
        uint64_t dstOffset = dstBase + static_cast<uint64_t>(curWOffset) * dstStepElems;
        uint32_t localElems = curBlocks * alignedDepthElems;
        uint32_t rightPad = alignedDepthElems - depth;

        AscendC::LocalTensor<DT_X> inLocal = inQueue.AllocTensor<DT_X>();

        AscendC::DataCopyExtParams inParams{
            static_cast<uint16_t>(curBlocks),
            static_cast<uint32_t>(depthBytes),
            0,
            0,
            0
        };

        AscendC::DataCopyPadExtParams<DT_X> padParams{
            true,
            0,
            static_cast<uint8_t>(rightPad),
            static_cast<DT_X>(0)
        };

        AscendC::DataCopyPad(inLocal, xGm[srcOffset], inParams, padParams);
        inQueue.EnQue(inLocal);

        uint32_t prevBlocks = curBlocks;
        uint64_t prevDstOffset = dstOffset;
        uint32_t prevLocalElems = localElems;

        curWOffset += curBlocks;
        remainBlocks -= curBlocks;

        while (prevBlocks > 0) {
            AscendC::LocalTensor<DT_X> readyIn = inQueue.DeQue<DT_X>();

            uint32_t nextBlocks = 0;
            uint64_t nextDstOffset = 0;
            uint32_t nextLocalElems = 0;

            if (remainBlocks > 0) {
               
                nextBlocks = remainBlocks > maxBlocksByUb ? maxBlocksByUb : remainBlocks;

                uint64_t nextSrcOffset = srcBase + static_cast<uint64_t>(curWOffset) * depth;
                nextDstOffset = dstBase + static_cast<uint64_t>(curWOffset) * dstStepElems;
                nextLocalElems = nextBlocks * alignedDepthElems;

                AscendC::LocalTensor<DT_X> nextIn = inQueue.AllocTensor<DT_X>();

                AscendC::DataCopyExtParams nextInParams{
                    static_cast<uint16_t>(nextBlocks),
                    static_cast<uint32_t>(depthBytes),
                    0,
                    0,
                    0
                };

                AscendC::DataCopyPad(nextIn, xGm[nextSrcOffset], nextInParams, padParams);
                inQueue.EnQue(nextIn);

                curWOffset += nextBlocks;
                remainBlocks -= nextBlocks;
            }

            AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
            AscendC::DataCopy(outLocal, readyIn, prevLocalElems);
            inQueue.FreeTensor(readyIn);

            outQueue.EnQue(outLocal);
            outLocal = outQueue.DeQue<DT_X>();

            AscendC::DataCopyExtParams outParams{
                static_cast<uint16_t>(prevBlocks),
                static_cast<uint32_t>(depthBytes),
                0,
                static_cast<uint32_t>(dstStrideBytes),
                0
            };

            AscendC::DataCopyPad(yGm[prevDstOffset], outLocal, outParams);
            outQueue.FreeTensor(outLocal);

            prevBlocks = nextBlocks;
            prevDstOffset = nextDstOffset;
            prevLocalElems = nextLocalElems;
        }
    }

    // 第2个点专用：把同一输出行的奇偶两段合并到一次队列流程中。
    // 目的不是改变 GM 搬运次数，而是减少每行两次 Alloc/EnQue/DeQue/Free 的固定开销。
    __aicore__ inline void CopySmallBlock2CropRowNonAlignedCombined(
        uint64_t srcBase0,
        uint64_t srcBase1,
        uint64_t dstRowBase,
        uint32_t count0,
        uint32_t count1
    ) {
        const uint32_t depthBytes = depth * sizeof(DT_X);
        const uint32_t alignElems = 32 / sizeof(DT_X);
        const uint32_t alignedDepthElems = AlignUpU32(depth, alignElems);

        const uint32_t maxBlocksByUb = TILE_ELEMS / alignedDepthElems;
        const uint32_t totalBlocks = count0 + count1;
        const uint32_t totalLocalElems = totalBlocks * alignedDepthElems;
        if (maxBlocksByUb == 0 || count0 > maxBlocksByUb || count1 > maxBlocksByUb ||
            totalLocalElems > TILE_ELEMS || count0 > 4095 || count1 > 4095) {
            CopyWidthStridedNonAlignedPaddedLocal(srcBase0, dstRowBase, count0);
            if (count1 != 0) {
                CopyWidthStridedNonAlignedPaddedLocal(srcBase1, dstRowBase + depth, count1);
            }
            return;
        }

        const uint32_t rightPad = alignedDepthElems - depth;
        AscendC::DataCopyPadExtParams<DT_X> padParams{
            true,
            0,
            static_cast<uint8_t>(rightPad),
            static_cast<DT_X>(0)
        };

        AscendC::LocalTensor<DT_X> inLocal = inQueue.AllocTensor<DT_X>();

        AscendC::DataCopyExtParams inParams0{
            static_cast<uint16_t>(count0),
            static_cast<uint32_t>(depthBytes),
            0,
            0,
            0
        };
        AscendC::DataCopyPad(inLocal, xGm[srcBase0], inParams0, padParams);

        const uint32_t part1LocalOffset = count0 * alignedDepthElems;
        if (count1 != 0) {
            AscendC::DataCopyExtParams inParams1{
                static_cast<uint16_t>(count1),
                static_cast<uint32_t>(depthBytes),
                0,
                0,
                0
            };
            AscendC::DataCopyPad(inLocal[part1LocalOffset], xGm[srcBase1], inParams1, padParams);
        }

        inQueue.EnQue(inLocal);
        inLocal = inQueue.DeQue<DT_X>();

        AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
        AscendC::DataCopy(outLocal, inLocal, totalLocalElems);
        inQueue.FreeTensor(inLocal);

        outQueue.EnQue(outLocal);
        outLocal = outQueue.DeQue<DT_X>();

        AscendC::DataCopyExtParams outParams0{
            static_cast<uint16_t>(count0),
            static_cast<uint32_t>(depthBytes),
            0,
            static_cast<uint32_t>(depthBytes),
            0
        };
        AscendC::DataCopyPad(yGm[dstRowBase], outLocal, outParams0);

        if (count1 != 0) {
            AscendC::DataCopyExtParams outParams1{
                static_cast<uint16_t>(count1),
                static_cast<uint32_t>(depthBytes),
                0,
                static_cast<uint32_t>(depthBytes),
                0
            };
            AscendC::DataCopyPad(yGm[dstRowBase + depth], outLocal[part1LocalOffset], outParams1);
        }

        outQueue.FreeTensor(outLocal);
    }



    // 第2/5点 Gather interleave 公共判断：
    // tempLocal 布局为 [part0 compact][对齐 pad][part1 compact][对齐 pad]，
    // outLocal 通过 Gather offset 表形成最终连续输出行 [col0][col1][col2]...
    __aicore__ inline bool CanUseBlock2InterleaveGather(
        uint32_t count0,
        uint32_t count1
    ) {
        if (depth == 0 || count0 == 0) {
            return false;
        }
        const uint32_t alignElems = 32 / sizeof(DT_X);
        const uint32_t part0Elems = count0 * depth;
        const uint32_t part1Elems = count1 * depth;
        const uint32_t part0AlignedElems = AlignUpU32(part0Elems, alignElems);
        const uint32_t part1AlignedElems = AlignUpU32(part1Elems, alignElems);
        const uint32_t tempElems = part0AlignedElems + part1AlignedElems;
        const uint32_t compactElems = (count0 + count1) * depth;

        return (compactElems <= TILE_ELEMS) &&
               (tempElems <= TILE_ELEMS) &&
               (compactElems <= GATHER_OFFSET_ELEMS);
    }

    // 第2/5点 Gather offset 表周期优化版：
    // output chunk 的 interleave offset 每 8 个输出列重复一次，只是源端整体向后平移 4*depth。
    // 对 full cycles 使用一次 scalar pattern + Adds 批量扩展；尾部列保守 scalar 构造。
    // 这样既保留第5点已经验证有效的优化，也能覆盖第2点 outWidth 不是 8 倍数的情况。
    __aicore__ inline void BuildBlock2InterleaveGatherOffsetsCycleI32(
        AscendC::LocalTensor<int32_t> offsetI32,
        uint32_t count0,
        uint32_t count1
    ) {
        const uint32_t alignElems = 32 / sizeof(DT_X);
        const uint32_t part0Elems = count0 * depth;
        const uint32_t part1Base = AlignUpU32(part0Elems, alignElems);
        const uint32_t colCount = count0 + count1;
        static constexpr uint32_t CYCLE_COLS = 8;

        if (colCount == 0 || depth == 0) {
            return;
        }

        const uint32_t fullCycles = colCount / CYCLE_COLS;
        const uint32_t cycleCols = (fullCycles > 0) ? CYCLE_COLS : colCount;
        const uint32_t baseCount = cycleCols * depth;

        // 先构造第一个周期；如果不足 8 列，就直接构造全部。
        uint32_t outIdx = 0;
        for (uint32_t c = 0; c < cycleCols; ++c) {
            const uint32_t inCol = c >> 1;
            const bool usePart0 = ((c & 1U) == 0U);
            const uint32_t srcBase = usePart0 ? (inCol * depth) : (part1Base + inCol * depth);
            for (uint32_t d = 0; d < depth; ++d) {
                offsetI32.SetValue(outIdx, static_cast<int32_t>((srcBase + d) * sizeof(DT_X)));
                ++outIdx;
            }
        }

        // 不足一个完整周期时，到这里就结束。
        if (fullCycles == 0) {
            return;
        }

        // 后续完整 8 列周期：直接在第一个周期基础上加固定 byte delta。
        // 每 8 个输出列对应每个 part 消耗 4 个输入列。
        const uint32_t fullBaseCount = CYCLE_COLS * depth;
        const int32_t groupDeltaBytes = static_cast<int32_t>((CYCLE_COLS >> 1) * depth * sizeof(DT_X));
        for (uint32_t g = 1; g < fullCycles; ++g) {
            AscendC::Adds(offsetI32[g * fullBaseCount], offsetI32[0], static_cast<int32_t>(g) * groupDeltaBytes, fullBaseCount);
        }

        // 尾部不足 8 列继续 scalar 构造；起点为 fullCycles * fullBaseCount，天然 32B 对齐。
        outIdx = fullCycles * fullBaseCount;
        for (uint32_t c = fullCycles * CYCLE_COLS; c < colCount; ++c) {
            const uint32_t inCol = c >> 1;
            const bool usePart0 = ((c & 1U) == 0U);
            const uint32_t srcBase = usePart0 ? (inCol * depth) : (part1Base + inCol * depth);
            for (uint32_t d = 0; d < depth; ++d) {
                offsetI32.SetValue(outIdx, static_cast<int32_t>((srcBase + d) * sizeof(DT_X)));
                ++outIdx;
            }
        }
    }



    // 第5点 full chunk=32 专用 offset 构造：
    // count0=count1=16，输出列固定 32。先构造第一个 8 列周期，后 3 个周期直接 Adds 平移。
    // 这比通用 CycleI32 少掉 colCount/fullCycles/tail 分支，专门服务第5点热路径。
    __aicore__ inline void BuildPoint5Full32GatherOffsetsI32(
        AscendC::LocalTensor<int32_t> offsetI32
    ) {
        static constexpr uint32_t CYCLE_COLS = 8;
        static constexpr uint32_t FULL_CYCLES = 4;
        static constexpr uint32_t COUNT_PER_PART = 16;

        const uint32_t alignElems = 32 / sizeof(DT_X);
        const uint32_t part0Elems = COUNT_PER_PART * depth;
        const uint32_t part1Base = AlignUpU32(part0Elems, alignElems);

        uint32_t outIdx = 0;
        for (uint32_t c = 0; c < CYCLE_COLS; ++c) {
            const uint32_t inCol = c >> 1;
            const bool usePart0 = ((c & 1U) == 0U);
            const uint32_t srcBase = usePart0 ? (inCol * depth) : (part1Base + inCol * depth);
            for (uint32_t d = 0; d < depth; ++d) {
                offsetI32.SetValue(outIdx, static_cast<int32_t>((srcBase + d) * sizeof(DT_X)));
                ++outIdx;
            }
        }

        const uint32_t fullBaseCount = CYCLE_COLS * depth;
        const int32_t groupDeltaBytes = static_cast<int32_t>((CYCLE_COLS >> 1) * depth * sizeof(DT_X));
        AscendC::Adds(offsetI32[1 * fullBaseCount], offsetI32[0], groupDeltaBytes, fullBaseCount);
        AscendC::Adds(offsetI32[2 * fullBaseCount], offsetI32[0], static_cast<int32_t>(2) * groupDeltaBytes, fullBaseCount);
        AscendC::Adds(offsetI32[3 * fullBaseCount], offsetI32[0], static_cast<int32_t>(3) * groupDeltaBytes, fullBaseCount);
    }
    __aicore__ inline void CopyBlock2CropRowNonAlignedGatherPrepared(
        uint64_t srcBase0,
        uint64_t srcBase1,
        uint64_t dstRowBase,
        uint32_t count0,
        uint32_t count1
    ) {
        const uint32_t alignElems = 32 / sizeof(DT_X);
        const uint32_t depthBytes = depth * sizeof(DT_X);
        const uint32_t part0Elems = count0 * depth;
        const uint32_t part1Elems = count1 * depth;
        const uint32_t part0AlignedElems = AlignUpU32(part0Elems, alignElems);
        const uint32_t part1AlignedElems = AlignUpU32(part1Elems, alignElems);
        const uint32_t tempElems = part0AlignedElems + part1AlignedElems;
        const uint32_t compactElems = part0Elems + part1Elems;

        AscendC::LocalTensor<DT_X> tempLocal = inQueue.AllocTensor<DT_X>();

        if (count0 != 0) {
            AscendC::DataCopyExtParams inParams0{
                1,
                static_cast<uint32_t>(part0Elems * sizeof(DT_X)),
                0,
                0,
                0
            };
            AscendC::DataCopyPadExtParams<DT_X> padParams0{
                true,
                0,
                static_cast<uint8_t>(part0AlignedElems - part0Elems),
                static_cast<DT_X>(0)
            };
            AscendC::DataCopyPad(tempLocal, xGm[srcBase0], inParams0, padParams0);
        }

        if (count1 != 0) {
            AscendC::DataCopyExtParams inParams1{
                1,
                static_cast<uint32_t>(part1Elems * sizeof(DT_X)),
                0,
                0,
                0
            };
            AscendC::DataCopyPadExtParams<DT_X> padParams1{
                true,
                0,
                static_cast<uint8_t>(part1AlignedElems - part1Elems),
                static_cast<DT_X>(0)
            };
            AscendC::DataCopyPad(tempLocal[part0AlignedElems], xGm[srcBase1], inParams1, padParams1);
        }

        inQueue.EnQue(tempLocal);
        tempLocal = inQueue.DeQue<DT_X>();
        tempLocal.SetSize(tempElems);

        AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
        outLocal.SetSize(compactElems);
        AscendC::LocalTensor<int32_t> offsetI32 = gatherOffsetBuf.Get<int32_t>();
        AscendC::LocalTensor<uint32_t> offsetLocal = offsetI32.template ReinterpretCast<uint32_t>();
        offsetLocal.SetSize(compactElems);

        AscendC::Gather(outLocal, tempLocal, offsetLocal, static_cast<uint32_t>(0), compactElems);
        inQueue.FreeTensor(tempLocal);

        outQueue.EnQue(outLocal);
        outLocal = outQueue.DeQue<DT_X>();

        AscendC::DataCopyExtParams outParams{
            1,
            static_cast<uint32_t>(compactElems * sizeof(DT_X)),
            0,
            0,
            0
        };
        AscendC::DataCopyPad(yGm[dstRowBase], outLocal, outParams);
        outQueue.FreeTensor(outLocal);
    }

    __aicore__ inline void CopyBlock2CropRowNonAlignedGatherWithBuiltOffsets(
        uint64_t srcBase0,
        uint64_t srcBase1,
        uint64_t dstRowBase,
        uint32_t count0,
        uint32_t count1
    ) {
        if (!CanUseBlock2InterleaveGather(count0, count1)) {
            CopySmallBlock2CropRowNonAlignedCombined(srcBase0, srcBase1, dstRowBase, count0, count1);
            return;
        }

        const uint32_t alignElems = 32 / sizeof(DT_X);
        const uint32_t depthBytes = depth * sizeof(DT_X);
        const uint32_t part0Elems = count0 * depth;
        const uint32_t part1Elems = count1 * depth;
        const uint32_t part0AlignedElems = AlignUpU32(part0Elems, alignElems);
        const uint32_t part1AlignedElems = AlignUpU32(part1Elems, alignElems);
        const uint32_t tempElems = part0AlignedElems + part1AlignedElems;
        const uint32_t compactElems = part0Elems + part1Elems;

        AscendC::LocalTensor<DT_X> tempLocal = inQueue.AllocTensor<DT_X>();

        if (count0 != 0) {
            AscendC::DataCopyExtParams inParams0{
                1,
                static_cast<uint32_t>(part0Elems * sizeof(DT_X)),
                0,
                0,
                0
            };
            AscendC::DataCopyPadExtParams<DT_X> padParams0{
                true,
                0,
                static_cast<uint8_t>(part0AlignedElems - part0Elems),
                static_cast<DT_X>(0)
            };
            AscendC::DataCopyPad(tempLocal, xGm[srcBase0], inParams0, padParams0);
        }

        if (count1 != 0) {
            AscendC::DataCopyExtParams inParams1{
                1,
                static_cast<uint32_t>(part1Elems * sizeof(DT_X)),
                0,
                0,
                0
            };
            AscendC::DataCopyPadExtParams<DT_X> padParams1{
                true,
                0,
                static_cast<uint8_t>(part1AlignedElems - part1Elems),
                static_cast<DT_X>(0)
            };
            AscendC::DataCopyPad(tempLocal[part0AlignedElems], xGm[srcBase1], inParams1, padParams1);
        }

        inQueue.EnQue(tempLocal);
        tempLocal = inQueue.DeQue<DT_X>();
        tempLocal.SetSize(tempElems);

        AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
        outLocal.SetSize(compactElems);
        AscendC::LocalTensor<int32_t> offsetI32 = gatherOffsetBuf.Get<int32_t>();
        AscendC::LocalTensor<uint32_t> offsetLocal = offsetI32.template ReinterpretCast<uint32_t>();
        offsetLocal.SetSize(compactElems);

        AscendC::Gather(outLocal, tempLocal, offsetLocal, static_cast<uint32_t>(0), compactElems);
        inQueue.FreeTensor(tempLocal);

        outQueue.EnQue(outLocal);
        outLocal = outQueue.DeQue<DT_X>();

        AscendC::DataCopyExtParams outParams{
            1,
            static_cast<uint32_t>(compactElems * sizeof(DT_X)),
            0,
            0,
            0
        };
        AscendC::DataCopyPad(yGm[dstRowBase], outLocal, outParams);
        outQueue.FreeTensor(outLocal);
    }

    // 第2个点 aligned-depth 专用：不用 DataCopyPad，保留多核并行，只减少 Pad 调度成本。
    __aicore__ inline void ProcessBlock2NoCropPairGroupAssembleAligned() {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        const uint32_t depthBytes = depth * sizeof(DT_X);
        const uint32_t depthBlocks = depthBytes >> 5;
        const uint32_t rowElems = width * depth;
        const uint32_t rowBlocks = width * depthBlocks;
        const uint64_t batchBlockStride = static_cast<uint64_t>(outBatch) * height * width * depth;

        uint32_t outRowElems = width * 2 * depth;
        uint32_t pairElems = outRowElems * 2;
        uint32_t pairsPerTile = TILE_ELEMS / pairElems;

        uint32_t groupsPerBatch = CeilDivU32(height, pairsPerTile);
        uint32_t totalTasks = outBatch * groupsPerBatch;

        uint32_t tasksPerCore = CeilDivU32(totalTasks, blockNum);
        uint32_t task = blockIdx * tasksPerCore;
        uint32_t endTask = MinU32(task + tasksPerCore, totalTasks);

        // 第10点优化实验：和第9点同一思想。
        // GM -> tempLocal 连续搬入四个 input batch row，避免 MTE2 直接写 UB 的 0,2,4... strided 位置；
        // tempLocal -> outLocal 在 UB 内完成 2 路交错；
        // outLocal -> GM 保持连续写回。
        AscendC::DataCopyParams rowInParams{
            1,
            static_cast<uint16_t>(rowBlocks),
            0,
            0
        };

        AscendC::DataCopyParams scatterParams{
            static_cast<uint16_t>(width),
            static_cast<uint16_t>(depthBlocks),
            0,
            static_cast<uint16_t>(depthBlocks)
        };

        uint32_t ob = task / groupsPerBatch;
        uint32_t groupIdx = task - ob * groupsPerBatch;
        for (; task < endTask; ++task, ++groupIdx) {
            if (groupIdx == groupsPerBatch) {
                groupIdx = 0;
                ++ob;
            }

            uint32_t hStart = groupIdx * pairsPerTile;
            if (hStart >= height) {
                continue;
            }

            uint32_t curPairs = height - hStart;
            if (curPairs > pairsPerTile) {
                curPairs = pairsPerTile;
            }

            uint32_t totalElems = curPairs * pairElems;
            AscendC::LocalTensor<DT_X> tempLocal = inQueue.AllocTensor<DT_X>();

            for (uint32_t hp = 0; hp < curPairs; ++hp) {
                uint32_t h = hStart + hp;
                uint32_t tempBase = hp * pairElems;

                uint64_t src0 = (((static_cast<uint64_t>(ob) * height + h) * width) * depth);
                uint64_t src1 = src0 + batchBlockStride;
                uint64_t src2 = src1 + batchBlockStride;
                uint64_t src3 = src2 + batchBlockStride;

                AscendC::DataCopy(tempLocal[tempBase], xGm[src0], rowInParams);
                AscendC::DataCopy(tempLocal[tempBase + rowElems], xGm[src1], rowInParams);
                AscendC::DataCopy(tempLocal[tempBase + (rowElems << 1)], xGm[src2], rowInParams);
                AscendC::DataCopy(tempLocal[tempBase + rowElems * 3], xGm[src3], rowInParams);
            }

            inQueue.EnQue(tempLocal);
            tempLocal = inQueue.DeQue<DT_X>();

            AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();

            for (uint32_t hp = 0; hp < curPairs; ++hp) {
                uint32_t tempBase = hp * pairElems;
                uint32_t outBase = hp * pairElems;

                // output row 2*h: [src0_w0][src1_w0][src0_w1][src1_w1]...
                AscendC::DataCopy(outLocal[outBase], tempLocal[tempBase], scatterParams);
                AscendC::DataCopy(outLocal[outBase + depth], tempLocal[tempBase + rowElems], scatterParams);

                // output row 2*h+1: [src2_w0][src3_w0][src2_w1][src3_w1]...
                AscendC::DataCopy(outLocal[outBase + outRowElems], tempLocal[tempBase + (rowElems << 1)], scatterParams);
                AscendC::DataCopy(outLocal[outBase + outRowElems + depth], tempLocal[tempBase + rowElems * 3], scatterParams);
            }

            inQueue.FreeTensor(tempLocal);

            outQueue.EnQue(outLocal);
            outLocal = outQueue.DeQue<DT_X>();

            uint64_t dstOffset =
                (((static_cast<uint64_t>(ob) * outHeight + (hStart << 1)) * outWidth) * depth);

            AscendC::DataCopy(yGm[dstOffset], outLocal, totalElems);
            outQueue.FreeTensor(outLocal);
        }
    }

    // 第9点实验版：GM -> tempLocal 时不再直接写到 strided local 位置，
    // 而是每个 bw/part 先紧凑连续写入 tempLocal。
    // depth 已 32B 对齐，因此 temp 内每个 depth block 起点仍然 32B 对齐，
    // 后续 Local -> Local strided scatter 是安全的。
    __aicore__ inline void CopyBlock4CropPartToCompactTemp(
        AscendC::LocalTensor<DT_X> tempLocal,
        uint32_t rowTempBase,
        uint32_t bh,
        uint32_t part,
        uint32_t startMod,
        uint32_t fullWStart,
        uint32_t partCount,
        uint32_t partPrefix,
        uint32_t depthBlocks,
        uint32_t ob,
        uint32_t h
    ) {
        if (partCount == 0) {
            return;
        }

        const uint32_t bw = (startMod + part) & 3;
        const uint32_t firstFullW = fullWStart + part;
        const uint32_t wStart = firstFullW >> 2;

        const uint32_t blockIndex = (bh << 2) + bw;
        const uint32_t ib = blockIndex * outBatch + ob;

        const uint64_t srcOffset =
            (((static_cast<uint64_t>(ib) * height + h) * width + wStart) * depth);

        const uint32_t tempOffset = rowTempBase + partPrefix * depth;

        AscendC::DataCopyParams inParams{
            static_cast<uint16_t>(partCount),
            static_cast<uint16_t>(depthBlocks),
            0,
            0
        };

        AscendC::DataCopy(tempLocal[tempOffset], xGm[srcOffset], inParams);
    }

    __aicore__ inline void ScatterBlock4CompactPartToRow(
        AscendC::LocalTensor<DT_X> outLocal,
        AscendC::LocalTensor<DT_X> tempLocal,
        uint32_t rowLocalBase,
        uint32_t rowTempBase,
        uint32_t part,
        uint32_t partCount,
        uint32_t partPrefix,
        uint32_t depthBlocks,
        uint32_t dstStrideBlocks
    ) {
        if (partCount == 0) {
            return;
        }

        const uint32_t srcOffset = rowTempBase + partPrefix * depth;
        const uint32_t dstOffset = rowLocalBase + part * depth;

        AscendC::DataCopyParams localParams{
            static_cast<uint16_t>(partCount),
            static_cast<uint16_t>(depthBlocks),
            0,
            static_cast<uint16_t>(dstStrideBlocks)
        };

        AscendC::DataCopy(outLocal[dstOffset], tempLocal[srcOffset], localParams);
    }

    __aicore__ inline void ProcessBlock4CropMultiRowsAssembleAligned() {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        const uint32_t depthBytes = depth * sizeof(DT_X);
        const uint32_t depthBlocks = depthBytes >> 5;
        const uint32_t dstStrideBlocks = 3 * depthBlocks;

        uint32_t chunkCols = TILE_ELEMS / (4 * depth);
        const uint32_t maxColsByBlockCount = 4095 * 4;
        if (chunkCols > maxColsByBlockCount) {
            chunkCols = maxColsByBlockCount;
        }

        const uint32_t numChunks = CeilDivU32(outWidth, chunkCols);
        const uint32_t totalTasks = outBatch * height * numChunks;

        const uint32_t tasksPerCore = CeilDivU32(totalTasks, blockNum);
        uint32_t task = blockIdx * tasksPerCore;
        const uint32_t endTask = MinU32(task + tasksPerCore, totalTasks);

        const uint32_t lastOutFullH = cropTop + outHeight - 1;
        uint32_t tmp = task / numChunks;
        uint32_t chunkIdx = task - tmp * numChunks;
        uint32_t ob = tmp / height;
        uint32_t h = tmp - ob * height;

        for (; task < endTask; ++task, ++chunkIdx) {
            if (chunkIdx == numChunks) {
                chunkIdx = 0;
                ++h;
                if (h == height) {
                    h = 0;
                    ++ob;
                }
            }

            const uint32_t fullHBase = h << 2;
            if (fullHBase > lastOutFullH) {
                continue;
            }

            uint32_t bhStart = 0;
            if (cropTop > fullHBase) {
                bhStart = cropTop - fullHBase;
            }
            if (bhStart >= 4) {
                continue;
            }

            uint32_t bhEnd = lastOutFullH - fullHBase;
            if (bhEnd >= 4) {
                bhEnd = 3;
            }
            if (bhStart > bhEnd) {
                continue;
            }

            const uint32_t owBase = chunkIdx * chunkCols;
            uint32_t curCols = outWidth - owBase;
            if (curCols > chunkCols) {
                curCols = chunkCols;
            }
            if (curCols == 0) {
                continue;
            }

            const uint32_t rowElems = curCols * depth;
            const uint32_t fullWStart = cropLeft + owBase;
            const uint32_t startMod = fullWStart & 3;
            const uint32_t partCount0 = (curCols + 3) >> 2;
            const uint32_t partCount1 = (curCols + 2) >> 2;
            const uint32_t partCount2 = (curCols + 1) >> 2;
            const uint32_t partCount3 = curCols >> 2;
            const uint32_t partPrefix0 = 0;
            const uint32_t partPrefix1 = partCount0;
            const uint32_t partPrefix2 = partPrefix1 + partCount1;
            const uint32_t partPrefix3 = partPrefix2 + partCount2;

            // 大多数第9点数据都处在 crop 的中间稳定区域：一个 input h 对应4条完整输出行。
            // 这里把 row/bw 两层循环完全展开，只保留顶部/底部边界行走小 fallback。
            if (bhStart == 0 && bhEnd == 3) {
                const uint32_t totalElems = rowElems << 2;
                AscendC::LocalTensor<DT_X> tempLocal = inQueue.AllocTensor<DT_X>();

                const uint32_t row0 = 0;
                const uint32_t row1 = rowElems;
                const uint32_t row2 = rowElems << 1;
                const uint32_t row3 = row2 + rowElems;

                // Phase 1: GM -> tempLocal compact，避免 MTE2 直接写 strided UB 位置。
                CopyBlock4CropPartToCompactTemp(tempLocal, row0, 0, 0, startMod, fullWStart, partCount0, partPrefix0, depthBlocks, ob, h);
                CopyBlock4CropPartToCompactTemp(tempLocal, row0, 0, 1, startMod, fullWStart, partCount1, partPrefix1, depthBlocks, ob, h);
                CopyBlock4CropPartToCompactTemp(tempLocal, row0, 0, 2, startMod, fullWStart, partCount2, partPrefix2, depthBlocks, ob, h);
                CopyBlock4CropPartToCompactTemp(tempLocal, row0, 0, 3, startMod, fullWStart, partCount3, partPrefix3, depthBlocks, ob, h);

                CopyBlock4CropPartToCompactTemp(tempLocal, row1, 1, 0, startMod, fullWStart, partCount0, partPrefix0, depthBlocks, ob, h);
                CopyBlock4CropPartToCompactTemp(tempLocal, row1, 1, 1, startMod, fullWStart, partCount1, partPrefix1, depthBlocks, ob, h);
                CopyBlock4CropPartToCompactTemp(tempLocal, row1, 1, 2, startMod, fullWStart, partCount2, partPrefix2, depthBlocks, ob, h);
                CopyBlock4CropPartToCompactTemp(tempLocal, row1, 1, 3, startMod, fullWStart, partCount3, partPrefix3, depthBlocks, ob, h);

                CopyBlock4CropPartToCompactTemp(tempLocal, row2, 2, 0, startMod, fullWStart, partCount0, partPrefix0, depthBlocks, ob, h);
                CopyBlock4CropPartToCompactTemp(tempLocal, row2, 2, 1, startMod, fullWStart, partCount1, partPrefix1, depthBlocks, ob, h);
                CopyBlock4CropPartToCompactTemp(tempLocal, row2, 2, 2, startMod, fullWStart, partCount2, partPrefix2, depthBlocks, ob, h);
                CopyBlock4CropPartToCompactTemp(tempLocal, row2, 2, 3, startMod, fullWStart, partCount3, partPrefix3, depthBlocks, ob, h);

                CopyBlock4CropPartToCompactTemp(tempLocal, row3, 3, 0, startMod, fullWStart, partCount0, partPrefix0, depthBlocks, ob, h);
                CopyBlock4CropPartToCompactTemp(tempLocal, row3, 3, 1, startMod, fullWStart, partCount1, partPrefix1, depthBlocks, ob, h);
                CopyBlock4CropPartToCompactTemp(tempLocal, row3, 3, 2, startMod, fullWStart, partCount2, partPrefix2, depthBlocks, ob, h);
                CopyBlock4CropPartToCompactTemp(tempLocal, row3, 3, 3, startMod, fullWStart, partCount3, partPrefix3, depthBlocks, ob, h);

                inQueue.EnQue(tempLocal);
                tempLocal = inQueue.DeQue<DT_X>();

                AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();

                // Phase 2: tempLocal compact -> outLocal final layout。
                ScatterBlock4CompactPartToRow(outLocal, tempLocal, row0, row0, 0, partCount0, partPrefix0, depthBlocks, dstStrideBlocks);
                ScatterBlock4CompactPartToRow(outLocal, tempLocal, row0, row0, 1, partCount1, partPrefix1, depthBlocks, dstStrideBlocks);
                ScatterBlock4CompactPartToRow(outLocal, tempLocal, row0, row0, 2, partCount2, partPrefix2, depthBlocks, dstStrideBlocks);
                ScatterBlock4CompactPartToRow(outLocal, tempLocal, row0, row0, 3, partCount3, partPrefix3, depthBlocks, dstStrideBlocks);

                ScatterBlock4CompactPartToRow(outLocal, tempLocal, row1, row1, 0, partCount0, partPrefix0, depthBlocks, dstStrideBlocks);
                ScatterBlock4CompactPartToRow(outLocal, tempLocal, row1, row1, 1, partCount1, partPrefix1, depthBlocks, dstStrideBlocks);
                ScatterBlock4CompactPartToRow(outLocal, tempLocal, row1, row1, 2, partCount2, partPrefix2, depthBlocks, dstStrideBlocks);
                ScatterBlock4CompactPartToRow(outLocal, tempLocal, row1, row1, 3, partCount3, partPrefix3, depthBlocks, dstStrideBlocks);

                ScatterBlock4CompactPartToRow(outLocal, tempLocal, row2, row2, 0, partCount0, partPrefix0, depthBlocks, dstStrideBlocks);
                ScatterBlock4CompactPartToRow(outLocal, tempLocal, row2, row2, 1, partCount1, partPrefix1, depthBlocks, dstStrideBlocks);
                ScatterBlock4CompactPartToRow(outLocal, tempLocal, row2, row2, 2, partCount2, partPrefix2, depthBlocks, dstStrideBlocks);
                ScatterBlock4CompactPartToRow(outLocal, tempLocal, row2, row2, 3, partCount3, partPrefix3, depthBlocks, dstStrideBlocks);

                ScatterBlock4CompactPartToRow(outLocal, tempLocal, row3, row3, 0, partCount0, partPrefix0, depthBlocks, dstStrideBlocks);
                ScatterBlock4CompactPartToRow(outLocal, tempLocal, row3, row3, 1, partCount1, partPrefix1, depthBlocks, dstStrideBlocks);
                ScatterBlock4CompactPartToRow(outLocal, tempLocal, row3, row3, 2, partCount2, partPrefix2, depthBlocks, dstStrideBlocks);
                ScatterBlock4CompactPartToRow(outLocal, tempLocal, row3, row3, 3, partCount3, partPrefix3, depthBlocks, dstStrideBlocks);

                inQueue.FreeTensor(tempLocal);

                outQueue.EnQue(outLocal);
                outLocal = outQueue.DeQue<DT_X>();

                const uint32_t ohBase = fullHBase - cropTop;
                const uint64_t dstOffset =
                    (((static_cast<uint64_t>(ob) * outHeight + ohBase) * outWidth + owBase) * depth);

                if (curCols == outWidth) {
                    AscendC::DataCopy(yGm[dstOffset], outLocal, totalElems);
                } else {
                    const uint64_t rowStride = static_cast<uint64_t>(outWidth) * depth;
                    AscendC::DataCopy(yGm[dstOffset], outLocal, rowElems);
                    AscendC::DataCopy(yGm[dstOffset + rowStride], outLocal[rowElems], rowElems);
                    AscendC::DataCopy(yGm[dstOffset + (rowStride << 1)], outLocal[rowElems << 1], rowElems);
                    AscendC::DataCopy(yGm[dstOffset + rowStride * 3], outLocal[rowElems * 3], rowElems);
                }

                outQueue.FreeTensor(outLocal);
                continue;
            }

            // 仅 crop 顶部/底部边界会进入这里，范围很小，保留通用但仍然 block4 的逻辑。
            const uint32_t rowCount = bhEnd - bhStart + 1;
            const uint32_t totalElems = rowCount * rowElems;
            const uint32_t ohBase = fullHBase + bhStart - cropTop;

            AscendC::LocalTensor<DT_X> tempLocal = inQueue.AllocTensor<DT_X>();

            uint32_t rowTempBase = 0;
            for (uint32_t bh = bhStart; bh <= bhEnd; ++bh) {
                CopyBlock4CropPartToCompactTemp(tempLocal, rowTempBase, bh, 0, startMod, fullWStart, partCount0, partPrefix0, depthBlocks, ob, h);
                CopyBlock4CropPartToCompactTemp(tempLocal, rowTempBase, bh, 1, startMod, fullWStart, partCount1, partPrefix1, depthBlocks, ob, h);
                CopyBlock4CropPartToCompactTemp(tempLocal, rowTempBase, bh, 2, startMod, fullWStart, partCount2, partPrefix2, depthBlocks, ob, h);
                CopyBlock4CropPartToCompactTemp(tempLocal, rowTempBase, bh, 3, startMod, fullWStart, partCount3, partPrefix3, depthBlocks, ob, h);
                rowTempBase += rowElems;
            }

            inQueue.EnQue(tempLocal);
            tempLocal = inQueue.DeQue<DT_X>();

            AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();

            uint32_t rowLocalBase = 0;
            rowTempBase = 0;
            for (uint32_t bh = bhStart; bh <= bhEnd; ++bh) {
                (void)bh;
                ScatterBlock4CompactPartToRow(outLocal, tempLocal, rowLocalBase, rowTempBase, 0, partCount0, partPrefix0, depthBlocks, dstStrideBlocks);
                ScatterBlock4CompactPartToRow(outLocal, tempLocal, rowLocalBase, rowTempBase, 1, partCount1, partPrefix1, depthBlocks, dstStrideBlocks);
                ScatterBlock4CompactPartToRow(outLocal, tempLocal, rowLocalBase, rowTempBase, 2, partCount2, partPrefix2, depthBlocks, dstStrideBlocks);
                ScatterBlock4CompactPartToRow(outLocal, tempLocal, rowLocalBase, rowTempBase, 3, partCount3, partPrefix3, depthBlocks, dstStrideBlocks);
                rowLocalBase += rowElems;
                rowTempBase += rowElems;
            }

            inQueue.FreeTensor(tempLocal);

            outQueue.EnQue(outLocal);
            outLocal = outQueue.DeQue<DT_X>();

            const uint64_t dstOffset =
                (((static_cast<uint64_t>(ob) * outHeight + ohBase) * outWidth + owBase) * depth);

            if (curCols == outWidth) {
                AscendC::DataCopy(yGm[dstOffset], outLocal, totalElems);
            } else {
                const uint64_t rowStride = static_cast<uint64_t>(outWidth) * depth;
                uint32_t localBase = 0;
                uint64_t rowDst = dstOffset;
                for (uint32_t r = 0; r < rowCount; ++r) {
                    AscendC::DataCopy(yGm[rowDst], outLocal[localBase], rowElems);
                    localBase += rowElems;
                    rowDst += rowStride;
                }
            }

            outQueue.FreeTensor(outLocal);
        }
    }

__aicore__ inline void ProcessBlock2NoCropTwoRowsAssembleAligned() {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        const uint32_t depthBytes = depth * sizeof(DT_X);
        const uint32_t depthBlocks = depthBytes >> 5;
        const uint64_t batchBlockStride = static_cast<uint64_t>(outBatch) * height * width * depth;

        uint32_t chunkW = TILE_ELEMS / (4 * depth);
        if (chunkW > 4095) {
            chunkW = 4095;
        }

        uint32_t numChunks = CeilDivU32(width, chunkW);
        uint32_t pairRows = outBatch * height;
        uint32_t totalTasks = pairRows * numChunks;

        uint32_t tasksPerCore = CeilDivU32(totalTasks, blockNum);
        uint32_t task = blockIdx * tasksPerCore;
        uint32_t endTask = MinU32(task + tasksPerCore, totalTasks);

        uint32_t pairTask = task / numChunks;
        uint32_t chunkIdx = task - pairTask * numChunks;
        uint32_t ob = pairTask / height;
        uint32_t h = pairTask - ob * height;

        for (; task < endTask; ++task, ++chunkIdx) {
            if (chunkIdx == numChunks) {
                chunkIdx = 0;
                ++h;
                if (h == height) {
                    h = 0;
                    ++ob;
                }
            }

            uint32_t wBase = chunkIdx * chunkW;
            uint32_t curW = width - wBase;
            if (curW > chunkW) {
                curW = chunkW;
            }

            const uint32_t onePartElems = curW * depth;
            uint32_t outRowElems = onePartElems << 1;
            uint32_t totalElems = outRowElems << 1;

            // 第1/3/4/8点优化：复用第9点已经验证有效的搬运思想。
            // 原实现 GM->UB 时直接写到 0,2,4... / 1,3,5... 的 strided UB 位置；
            // 这里改为先把四路输入连续搬进 tempLocal，再在 UB 内 scatter 到最终 two-row 布局，
            // 最后仍然连续写回 GM。depth 已 32B 对齐，因此 Local->Local strided scatter 是安全的。
            AscendC::LocalTensor<DT_X> tempLocal = inQueue.AllocTensor<DT_X>();

            AscendC::DataCopyParams compactParams{
                static_cast<uint16_t>(curW),
                static_cast<uint16_t>(depthBlocks),
                0,
                0
            };

            uint64_t src0 = (((static_cast<uint64_t>(ob) * height + h) * width + wBase) * depth);
            uint64_t src1 = src0 + batchBlockStride;
            uint64_t src2 = src1 + batchBlockStride;
            uint64_t src3 = src2 + batchBlockStride;

            const uint32_t temp0 = 0;
            const uint32_t temp1 = onePartElems;
            const uint32_t temp2 = onePartElems << 1;
            const uint32_t temp3 = temp2 + onePartElems;

            AscendC::DataCopy(tempLocal[temp0], xGm[src0], compactParams);
            AscendC::DataCopy(tempLocal[temp1], xGm[src1], compactParams);
            AscendC::DataCopy(tempLocal[temp2], xGm[src2], compactParams);
            AscendC::DataCopy(tempLocal[temp3], xGm[src3], compactParams);

            inQueue.EnQue(tempLocal);
            tempLocal = inQueue.DeQue<DT_X>();

            AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();

            AscendC::DataCopyParams scatterParams{
                static_cast<uint16_t>(curW),
                static_cast<uint16_t>(depthBlocks),
                0,
                static_cast<uint16_t>(depthBlocks)
            };

            // row0: [src0_w0][src1_w0][src0_w1][src1_w1]...
            AscendC::DataCopy(outLocal[0], tempLocal[temp0], scatterParams);
            AscendC::DataCopy(outLocal[depth], tempLocal[temp1], scatterParams);
            // row1: [src2_w0][src3_w0][src2_w1][src3_w1]...
            AscendC::DataCopy(outLocal[outRowElems], tempLocal[temp2], scatterParams);
            AscendC::DataCopy(outLocal[outRowElems + depth], tempLocal[temp3], scatterParams);

            inQueue.FreeTensor(tempLocal);

            outQueue.EnQue(outLocal);
            outLocal = outQueue.DeQue<DT_X>();

            uint64_t dstRow0 =
                (((static_cast<uint64_t>(ob) * outHeight + (h << 1)) * outWidth + (wBase << 1)) * depth);

            if (curW == width) {
                AscendC::DataCopy(yGm[dstRow0], outLocal, totalElems);
            } else {
                uint64_t dstRow1 =
                    (((static_cast<uint64_t>(ob) * outHeight + (h << 1) + 1) * outWidth + (wBase << 1)) * depth);

                AscendC::DataCopy(yGm[dstRow0], outLocal, outRowElems);
                AscendC::DataCopy(yGm[dstRow1], outLocal[outRowElems], outRowElems);
            }

            outQueue.FreeTensor(outLocal);
        }
    }


    __aicore__ inline void ProcessGeneralNoCropAssembleAligned() {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        uint32_t totalRows = outBatch * outHeight;
        uint32_t rowsPerCore = CeilDivU32(totalRows, blockNum);
        uint32_t startRow = blockIdx * rowsPerCore;
        uint32_t endRow = MinU32(startRow + rowsPerCore, totalRows);

        const uint32_t depthBytes = depth * sizeof(DT_X);
        const uint32_t depthBlocks = depthBytes >> 5;
        const uint32_t dstStrideBlocks = depthBlocks;
        const uint32_t dstStepElems = depth << 1;

        uint32_t maxWByUb = TILE_ELEMS / dstStepElems;
        if (maxWByUb > 4095) {
            maxWByUb = 4095;
        }

        uint32_t ob = startRow / outHeight;
        uint32_t oh = startRow - ob * outHeight;
        uint32_t h = oh >> 1;
        uint32_t bh = oh & 1;

        for (uint32_t rowTask = startRow; rowTask < endRow; ++rowTask) {
            uint64_t dstOffset =
                (((static_cast<uint64_t>(ob) * outHeight + oh) * outWidth) * depth);

            const uint32_t blockIndexBase = bh << 1;
            const uint32_t ib0 = blockIndexBase * outBatch + ob;
            const uint32_t ib1 = ib0 + outBatch;
            const uint64_t srcRowBase0 =
                (((static_cast<uint64_t>(ib0) * height + h) * width) * depth);
            const uint64_t srcRowBase1 =
                (((static_cast<uint64_t>(ib1) * height + h) * width) * depth);

            uint32_t wBase = 0;
            while (wBase < width) {
                uint32_t curW = width - wBase;
                if (curW > maxWByUb) {
                    curW = maxWByUb;
                }

                const uint32_t copyElems = curW * dstStepElems;
                const uint64_t wBaseDepthOffset = static_cast<uint64_t>(wBase) * depth;

                AscendC::LocalTensor<DT_X> inLocal = inQueue.AllocTensor<DT_X>();
                AscendC::DataCopyParams inParams{
                    static_cast<uint16_t>(curW),
                    static_cast<uint16_t>(depthBlocks),
                    0,
                    static_cast<uint16_t>(dstStrideBlocks)
                };

                AscendC::DataCopy(inLocal[0], xGm[srcRowBase0 + wBaseDepthOffset], inParams);
                AscendC::DataCopy(inLocal[depth], xGm[srcRowBase1 + wBaseDepthOffset], inParams);

                inQueue.EnQue(inLocal);
                inLocal = inQueue.DeQue<DT_X>();

                AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
                AscendC::DataCopy(outLocal, inLocal, copyElems);
                inQueue.FreeTensor(inLocal);

                outQueue.EnQue(outLocal);
                outLocal = outQueue.DeQue<DT_X>();

                AscendC::DataCopy(yGm[dstOffset], outLocal, copyElems);
                outQueue.FreeTensor(outLocal);

                wBase += curW;
                dstOffset += static_cast<uint64_t>(curW) * dstStepElems;
            }

            ++oh;
            bh ^= 1;
            if (bh == 0) {
                ++h;
            }

            if (oh == outHeight) {
                oh = 0;
                h = 0;
                bh = 0;
                ++ob;
            }
        }
    }




    __aicore__ inline void ProcessBlock2CropColumnChunkNonAlignedGather() {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        const uint32_t chunkCols = CalcNonAlignedChunkCols();
        const uint32_t fullCount0 = (chunkCols + 1) >> 1;
        const uint32_t fullCount1 = chunkCols >> 1;
        const bool useFullGather = CanUseBlock2InterleaveGather(fullCount0, fullCount1);
        const bool usePoint5Full32Gather = (chunkCols == 32U && fullCount0 == 16U && fullCount1 == 16U && useFullGather);
        if (useFullGather) {
            AscendC::LocalTensor<int32_t> offsetI32 = gatherOffsetBuf.Get<int32_t>();
            if (usePoint5Full32Gather) {
                BuildPoint5Full32GatherOffsetsI32(offsetI32);
            } else {
                BuildBlock2InterleaveGatherOffsetsCycleI32(offsetI32, fullCount0, fullCount1);
            }
        }

        const uint32_t numChunks = CeilDivU32(outWidth, chunkCols);
        const uint32_t totalRows = outBatch * outHeight;
        const uint32_t totalTasks = totalRows * numChunks;

        const uint32_t tasksPerCore = CeilDivU32(totalTasks, blockNum);
        uint32_t task = blockIdx * tasksPerCore;
        const uint32_t endTask = MinU32(task + tasksPerCore, totalTasks);

        uint32_t rowTask = task / numChunks;
        uint32_t chunkIdx = task - rowTask * numChunks;
        uint32_t ob = rowTask / outHeight;
        uint32_t oh = rowTask - ob * outHeight;

        for (; task < endTask; ++task, ++chunkIdx) {
            if (chunkIdx == numChunks) {
                chunkIdx = 0;
                ++oh;
                if (oh == outHeight) {
                    oh = 0;
                    ++ob;
                }
            }

            const uint32_t owBase = chunkIdx * chunkCols;
            uint32_t curCols = outWidth - owBase;
            if (curCols > chunkCols) {
                curCols = chunkCols;
            }
            if (curCols == 0) {
                continue;
            }

            const uint32_t fullH = oh + cropTop;
            const uint32_t h = fullH >> 1;
            const uint32_t bh = fullH & 1;
            const uint32_t fullWStart = cropLeft + owBase;
            const uint32_t startMod = fullWStart & 1;

            const uint32_t blockIndexBase = bh << 1;
            const uint64_t dstRowBase =
                (((static_cast<uint64_t>(ob) * outHeight + oh) * outWidth + owBase) * depth);

            const uint32_t bw0 = startMod;
            const uint32_t count0 = (curCols + 1) >> 1;
            const uint32_t wStart0 = fullWStart >> 1;
            const uint32_t ib0 = (blockIndexBase + bw0) * outBatch + ob;
            const uint64_t srcBase0 =
                (((static_cast<uint64_t>(ib0) * height + h) * width + wStart0) * depth);

            const uint32_t bw1 = startMod ^ 1;
            const uint32_t count1 = curCols >> 1;
            const uint32_t wStart1 = (fullWStart + 1) >> 1;
            const uint32_t ib1 = (blockIndexBase + bw1) * outBatch + ob;
            const uint64_t srcBase1 =
                (((static_cast<uint64_t>(ib1) * height + h) * width + wStart1) * depth);

            // Gather offset 表按 full chunk 构造；尾 chunk 保守回退原正确路径，避免每个尾块重建 offset 表。
            if (curCols == chunkCols && useFullGather) {
                CopyBlock2CropRowNonAlignedGatherPrepared(srcBase0, srcBase1, dstRowBase, count0, count1);
            } else {
                CopyWidthStridedNonAlignedPaddedLocal(srcBase0, dstRowBase, count0);
                if (count1 != 0) {
                    CopyWidthStridedNonAlignedPaddedLocal(srcBase1, dstRowBase + depth, count1);
                }
            }
        }
    }

    __aicore__ inline void ProcessSmallBlock2CropFast() {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        const uint32_t totalRows = outBatch * outHeight;
        const uint32_t rowsPerCore = CeilDivU32(totalRows, blockNum);
        uint32_t rowTask = blockIdx * rowsPerCore;
        const uint32_t endRow = MinU32(rowTask + rowsPerCore, totalRows);

        const uint32_t startMod = cropLeft & 1;
        const uint32_t count0 = (outWidth + 1) >> 1;
        const uint32_t count1 = outWidth >> 1;
        const uint32_t wStart0 = cropLeft >> 1;
        const uint32_t wStart1 = (cropLeft + 1) >> 1;
        const bool useGather = CanUseBlock2InterleaveGather(count0, count1);
        if (useGather) {
            AscendC::LocalTensor<int32_t> offsetI32 = gatherOffsetBuf.Get<int32_t>();
            BuildBlock2InterleaveGatherOffsetsCycleI32(offsetI32, count0, count1);
        }

        uint32_t ob = rowTask / outHeight;
        uint32_t oh = rowTask - ob * outHeight;
        for (; rowTask < endRow; ++rowTask, ++oh) {
            if (oh == outHeight) {
                oh = 0;
                ++ob;
            }

            const uint32_t fullH = oh + cropTop;
            const uint32_t h = fullH >> 1;
            const uint32_t bh = fullH & 1;
            const uint32_t blockIndexBase = bh << 1;

            const uint32_t ib0 = (blockIndexBase + startMod) * outBatch + ob;
            const uint32_t ib1 = (blockIndexBase + (startMod ^ 1)) * outBatch + ob;

            const uint64_t srcBase0 =
                (((static_cast<uint64_t>(ib0) * height + h) * width + wStart0) * depth);
            const uint64_t srcBase1 =
                (((static_cast<uint64_t>(ib1) * height + h) * width + wStart1) * depth);
            const uint64_t dstRowBase =
                (((static_cast<uint64_t>(ob) * outHeight + oh) * outWidth) * depth);

            if (useGather) {
                CopyBlock2CropRowNonAlignedGatherWithBuiltOffsets(srcBase0, srcBase1, dstRowBase, count0, count1);
            } else {
                CopySmallBlock2CropRowNonAlignedCombined(srcBase0, srcBase1, dstRowBase, count0, count1);
            }
        }
    }

    __aicore__ inline void ProcessBlock2NoCropWidth1LargeDepthAlignedDirect() {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        const uint32_t taskPerCore = CeilDivU32(rowTaskNum, blockNum);
        uint32_t startTask = blockIdx * taskPerCore;
        const uint32_t endTask = MinU32(startTask + taskPerCore, rowTaskNum);
        if (startTask >= endTask) {
            return;
        }

        const int32_t eventIdMte2ToMte3 = static_cast<int32_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));

        uint32_t tmp = startTask / height;
        uint32_t h = startTask - tmp * height;
        for (uint32_t task = startTask; task < endTask; ++task, ++h) {
            if (h == height) {
                h = 0;
                ++tmp;
            }

            const uint32_t blockIndex = tmp & 3;
            const uint32_t ob = tmp >> 2;

            const uint32_t bh = blockIndex >> 1;
            const uint32_t bw = blockIndex & 1;
            const uint32_t ib = blockIndex * outBatch + ob;

            const uint32_t oh = (h << 1) + bh;
            const uint32_t ow = bw;  // width=1, outWidth=2

            const uint64_t srcOffset =
                (((static_cast<uint64_t>(ib) * height + h) * width) * depth);
            const uint64_t dstOffset =
                (((static_cast<uint64_t>(ob) * outHeight + oh) * outWidth + ow) * depth);

            CopySpanAlignedDirectFlag(srcOffset, dstOffset, depth, eventIdMte2ToMte3);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatherOffsetBuf;

    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;

    uint32_t height;
    uint32_t width;
    uint32_t depth;

    uint32_t blockSize;

    uint32_t cropTop;
    uint32_t cropBottom;
    uint32_t cropLeft;
    uint32_t cropRight;

    uint32_t outBatch;
    uint32_t outHeight;
    uint32_t outWidth;

    uint32_t outputLength;
    uint32_t rowTaskNum;
};

template <typename DT_X>
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    (void)workspace;

    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelBatchToSpace<DT_X> op;
    op.InitAndProcess(x, y, tilingData);
}
