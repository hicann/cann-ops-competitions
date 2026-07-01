// Kernel侧核函数实现
#include "kernel_operator.h"

#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

namespace {
    constexpr uint32_t MAX_CHUNK = 8192;
}

template <class DT_X>
class KernelBatchToSpace {
public:
    __aicore__ inline KernelBatchToSpace() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling) {
        inN_ = tiling.inN; inH_ = tiling.inH; inW_ = tiling.inW;
        outN_ = tiling.outN; outH_ = tiling.outH; outW_ = tiling.outW;
        channels_ = tiling.channels; blockSize_ = tiling.blockSize;
        cropTop_ = tiling.cropTop; cropLeft_ = tiling.cropLeft;
        alignedMode_ = tiling.alignedMode;
        totalUnits_ = tiling.totalUnits; unitsPerCore_ = tiling.unitsPerCore;

        const uint32_t totalInElems = inN_ * inH_ * inW_ * channels_;
        const uint32_t totalOutElems = outN_ * outH_ * outW_ * channels_;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), totalInElems);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), totalOutElems);

        colsPerTile_ = (channels_ > 0 && channels_ <= MAX_CHUNK) ? (MAX_CHUNK / channels_) : 1;
        if (colsPerTile_ == 0) colsPerTile_ = 1;
        numColChunks_ = (outW_ + colsPerTile_ - 1) / colsPerTile_;

        // 安全的 8192 元素双队列
        pipe_.InitBuffer(inQueue_, 2, MAX_CHUNK * sizeof(DT_X));
        pipe_.InitBuffer(outQueue_, 2, MAX_CHUNK * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (totalUnits_ == 0 || unitsPerCore_ == 0) return;

        const uint32_t coreIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t begin = coreIdx * unitsPerCore_;
        if (begin >= totalUnits_) return;
        uint32_t end = begin + unitsPerCore_;
        if (end > totalUnits_) end = totalUnits_;

        if (alignedMode_ == 1) {
            uint32_t chunk = begin % numColChunks_;
            uint32_t rowIdx = begin / numColChunks_;
            uint32_t h = rowIdx % outH_;
            uint32_t n = rowIdx / outH_;
            
            if (blockSize_ ==2){
                for (uint32_t unit = begin; unit < end; ++unit) {
                    ProcessChunkOpt2(n, h, chunk);
                    // 纯加法状态机
                    chunk++;
                    if (chunk == numColChunks_) {
                        chunk = 0; h++;
                        if (h == outH_) { h = 0; n++; }
                    }
                }
            } else if (blockSize_ ==4){
                for (uint32_t unit = begin; unit < end; ++unit) {
                    ProcessChunkOpt4(n, h, chunk);
                    // 纯加法状态机
                    chunk++;
                    if (chunk == numColChunks_) {
                        chunk = 0; h++;
                        if (h == outH_) { h = 0; n++; }
                    }
                }
            } else {
                for (uint32_t unit = begin; unit < end; ++unit) {
                    ProcessChunkOpt(n, h, chunk);
                    // 纯加法状态机
                    chunk++;
                    if (chunk == numColChunks_) {
                        chunk = 0; h++;
                        if (h == outH_) { h = 0; n++; }
                    }
                }
            }
        } else {
            ProcessAllCellsOpt(begin, end);
        }
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t n) const {
        const uint32_t elemsPerBlock = 32 / sizeof(DT_X);
        return (n + elemsPerBlock - 1) / elemsPerBlock * elemsPerBlock;
    }

    __aicore__ inline void ProcessChunkOpt(uint32_t n, uint32_t h, uint32_t chunk) {
        const uint32_t cw0 = chunk * colsPerTile_;
        if (cw0 >= outW_) return;
        uint32_t cw1 = cw0 + colsPerTile_;
        if (cw1 > outW_) cw1 = outW_;
        const uint32_t segCols = cw1 - cw0;

        const uint32_t oh = (h + cropTop_) % blockSize_;
        const uint32_t hIn = (h + cropTop_) / blockSize_;

        const uint32_t cBytesDb = static_cast<uint32_t>(channels_ * sizeof(DT_X)) / 32;
        const uint32_t dstStrideDb = (blockSize_ - 1) * cBytesDb;

        const uint32_t cw0Pad = cw0 + cropLeft_;
        const uint32_t cw0In = cw0Pad / blockSize_;
        const uint32_t cw0Offset = cw0Pad % blockSize_;

        const uint32_t cw1Pad = cw1 - 1 + cropLeft_;
        const uint32_t cw1In = cw1Pad / blockSize_;
        const uint32_t cw1Offset = cw1Pad % blockSize_;

        // 【优化核心】：标量计算外提。提前算好常量，消灭 for 循环内沉重的乘法和加法开销
        const uint32_t inH_inW = inH_ * inW_;
        const uint32_t baseBatch = (oh * blockSize_) * outN_ + n;
        const uint32_t baseH = hIn * inW_;
        const uint32_t inBatchBase = (baseBatch * inH_inW) + baseH;
        const uint32_t step = outN_ * inH_inW; // 每循环一次 ow，基础地址跨越的固定步长

        AscendC::LocalTensor<DT_X> obuf = outQueue_.template AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> ibuf = inQueue_.template AllocTensor<DT_X>();
        uint32_t ibufOffset = 0;

        // ================= 阶段一：纯 DMA 连续聚合读取 (MTE2) =================
        for (uint32_t ow = 0; ow < blockSize_; ++ow) {
            const uint32_t wInFirst = cw0In + (ow < cw0Offset ? 1 : 0);
            if (wInFirst > cw1In || (wInFirst == cw1In && ow > cw1Offset)) continue;
            
            const uint32_t k = cw1In - wInFirst + (ow <= cw1Offset ? 1 : 0);
            
            // 使用外提后的算式，极大减轻标量核调度压力
            const uint32_t inBase = (inBatchBase + ow * step + wInFirst) * channels_;

            AscendC::DataCopy(ibuf[ibufOffset], xGm_[inBase], k * channels_);
            ibufOffset += k * channels_;
        }

        // 把屏障移到循环外，等待当前块的所有 GM 读取完成
        inQueue_.EnQue(ibuf);
        ibuf = inQueue_.template DeQue<DT_X>();

        // ================= 阶段二：纯 Vector 高速重排拼图 (UB -> UB) =================
        ibufOffset = 0;
        AscendC::DataCopyParams params;
        params.srcStride = 0;
        params.dstStride = static_cast<uint16_t>(dstStrideDb);
        
        for (uint32_t ow = 0; ow < blockSize_; ++ow) {
            const uint32_t wInFirst = cw0In + (ow < cw0Offset ? 1 : 0);
            if (wInFirst > cw1In || (wInFirst == cw1In && ow > cw1Offset)) continue;
            
            const uint32_t k = cw1In - wInFirst + (ow <= cw1Offset ? 1 : 0);
            const uint32_t wFirst = wInFirst * blockSize_ + ow - cropLeft_;
            const uint32_t localOff = (wFirst - cw0) * channels_;

            const uint32_t inBatch = baseBatch + ow * outN_;
            const uint32_t inBase = (inBatch * inH_inW + baseH + wInFirst) * channels_;

            params.blockCount = static_cast<uint16_t>(k);
            params.blockLen = static_cast<uint16_t>(cBytesDb);
            
            // Vector 引擎执行跨步写入
            AscendC::DataCopy(obuf[localOff], ibuf[ibufOffset], params);
            ibufOffset += k * channels_;
        }
        
        inQueue_.FreeTensor(ibuf);

        // ================= 阶段三：连续大块写回 GM (MTE3) =================
        outQueue_.EnQue(obuf);
        obuf = outQueue_.template DeQue<DT_X>();
        
        const uint32_t outBase = ((n * outH_ + h) * outW_ + cw0) * channels_;
        AscendC::DataCopy(yGm_[outBase], obuf, segCols * channels_);
        outQueue_.FreeTensor(obuf);
    }

    __aicore__ inline void ProcessChunkOpt2(uint32_t n, uint32_t h, uint32_t chunk) {
        const uint32_t cw0 = chunk * colsPerTile_;
        if (cw0 >= outW_) return;
        uint32_t cw1 = cw0 + colsPerTile_;
        if (cw1 > outW_) cw1 = outW_;
        const uint32_t segCols = cw1 - cw0;

        const uint32_t oh = (h + cropTop_) & 1;
        const uint32_t hIn = (h + cropTop_) >> 1;

        const uint32_t cBytesDb = static_cast<uint32_t>(channels_ * sizeof(DT_X)) / 32;
        const uint32_t dstStrideDb = cBytesDb;

        const uint32_t cw0Pad = cw0 + cropLeft_;
        const uint32_t cw0In = cw0Pad >> 1;
        const uint32_t cw0Offset = cw0Pad & 1;

        const uint32_t cw1Pad = cw1 - 1 + cropLeft_;
        const uint32_t cw1In = cw1Pad  >> 1;
        const uint32_t cw1Offset = cw1Pad & 1;

        // 【优化核心】：标量计算外提。提前算好常量，消灭 for 循环内沉重的乘法和加法开销
        const uint32_t inH_inW = inH_ * inW_;
        const uint32_t baseBatch = (oh * 2) * outN_ + n;
        const uint32_t baseH = hIn * inW_;
        const uint32_t inBatchBase = (baseBatch * inH_inW) + baseH;
        const uint32_t step = outN_ * inH_inW; // 每循环一次 ow，基础地址跨越的固定步长

        AscendC::LocalTensor<DT_X> obuf = outQueue_.template AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> ibuf = inQueue_.template AllocTensor<DT_X>();
        uint32_t ibufOffset = 0;

        // ================= 阶段一：纯 DMA 连续聚合读取 (MTE2) =================
        for (uint32_t ow = 0; ow < 2; ++ow) {
            const uint32_t wInFirst = cw0In + (ow < cw0Offset ? 1 : 0);
            if (wInFirst > cw1In || (wInFirst == cw1In && ow > cw1Offset)) continue;
            
            const uint32_t k = cw1In - wInFirst + (ow <= cw1Offset ? 1 : 0);
            
            // 使用外提后的算式，极大减轻标量核调度压力
            const uint32_t inBase = (inBatchBase + ow * step + wInFirst) * channels_;

            AscendC::DataCopy(ibuf[ibufOffset], xGm_[inBase], k * channels_);
            ibufOffset += k * channels_;
        }

        // 把屏障移到循环外，等待当前块的所有 GM 读取完成
        inQueue_.EnQue(ibuf);
        ibuf = inQueue_.template DeQue<DT_X>();

        // ================= 阶段二：纯 Vector 高速重排拼图 (UB -> UB) =================
        ibufOffset = 0;
        AscendC::DataCopyParams params;
        params.srcStride = 0;
        params.dstStride = static_cast<uint16_t>(dstStrideDb);
        
        for (uint32_t ow = 0; ow < 2; ++ow) {
            const uint32_t wInFirst = cw0In + (ow < cw0Offset ? 1 : 0);
            if (wInFirst > cw1In || (wInFirst == cw1In && ow > cw1Offset)) continue;
            
            const uint32_t k = cw1In - wInFirst + (ow <= cw1Offset ? 1 : 0);
            const uint32_t wFirst = wInFirst * 2 + ow - cropLeft_;
            const uint32_t localOff = (wFirst - cw0) * channels_;

            const uint32_t inBatch = baseBatch + ow * outN_;
            const uint32_t inBase = (inBatch * inH_inW + baseH + wInFirst) * channels_;

            params.blockCount = static_cast<uint16_t>(k);
            params.blockLen = static_cast<uint16_t>(cBytesDb);
            
            // Vector 引擎执行跨步写入
            AscendC::DataCopy(obuf[localOff], ibuf[ibufOffset], params);
            ibufOffset += k * channels_;
        }
        
        inQueue_.FreeTensor(ibuf);

        // ================= 阶段三：连续大块写回 GM (MTE3) =================
        outQueue_.EnQue(obuf);
        obuf = outQueue_.template DeQue<DT_X>();
        
        const uint32_t outBase = ((n * outH_ + h) * outW_ + cw0) * channels_;
        AscendC::DataCopy(yGm_[outBase], obuf, segCols * channels_);
        outQueue_.FreeTensor(obuf);
    }

    __aicore__ inline void ProcessChunkOpt4(uint32_t n, uint32_t h, uint32_t chunk) {
        const uint32_t cw0 = chunk * colsPerTile_;
        if (cw0 >= outW_) return;
        uint32_t cw1 = cw0 + colsPerTile_;
        if (cw1 > outW_) cw1 = outW_;
        const uint32_t segCols = cw1 - cw0;

        const uint32_t oh = (h + cropTop_) & 3;
        const uint32_t hIn = (h + cropTop_) >> 2;

        const uint32_t cBytesDb = static_cast<uint32_t>(channels_ * sizeof(DT_X)) / 32;
        const uint32_t dstStrideDb = 3 * cBytesDb;

        const uint32_t cw0Pad = cw0 + cropLeft_;
        const uint32_t cw0In = cw0Pad >> 2;
        const uint32_t cw0Offset = cw0Pad & 3;

        const uint32_t cw1Pad = cw1 - 1 + cropLeft_;
        const uint32_t cw1In = cw1Pad  >> 2;
        const uint32_t cw1Offset = cw1Pad & 3;

        // 【优化核心】：标量计算外提。提前算好常量，消灭 for 循环内沉重的乘法和加法开销
        const uint32_t inH_inW = inH_ * inW_;
        const uint32_t baseBatch = (oh * 4) * outN_ + n;
        const uint32_t baseH = hIn * inW_;
        const uint32_t inBatchBase = (baseBatch * inH_inW) + baseH;
        const uint32_t step = outN_ * inH_inW; // 每循环一次 ow，基础地址跨越的固定步长

        AscendC::LocalTensor<DT_X> obuf = outQueue_.template AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> ibuf = inQueue_.template AllocTensor<DT_X>();
        uint32_t ibufOffset = 0;

        // ================= 阶段一：纯 DMA 连续聚合读取 (MTE2) =================
        for (uint32_t ow = 0; ow < 4; ++ow) {
            const uint32_t wInFirst = cw0In + (ow < cw0Offset ? 1 : 0);
            if (wInFirst > cw1In || (wInFirst == cw1In && ow > cw1Offset)) continue;
            
            const uint32_t k = cw1In - wInFirst + (ow <= cw1Offset ? 1 : 0);
            
            // 使用外提后的算式，极大减轻标量核调度压力
            const uint32_t inBase = (inBatchBase + ow * step + wInFirst) * channels_;

            AscendC::DataCopy(ibuf[ibufOffset], xGm_[inBase], k * channels_);
            ibufOffset += k * channels_;
        }

        // 把屏障移到循环外，等待当前块的所有 GM 读取完成
        inQueue_.EnQue(ibuf);
        ibuf = inQueue_.template DeQue<DT_X>();

        // ================= 阶段二：纯 Vector 高速重排拼图 (UB -> UB) =================
        ibufOffset = 0;
        AscendC::DataCopyParams params;
        params.srcStride = 0;
        params.dstStride = static_cast<uint16_t>(dstStrideDb);
        
        for (uint32_t ow = 0; ow < 4; ++ow) {
            const uint32_t wInFirst = cw0In + (ow < cw0Offset ? 1 : 0);
            if (wInFirst > cw1In || (wInFirst == cw1In && ow > cw1Offset)) continue;
            
            const uint32_t k = cw1In - wInFirst + (ow <= cw1Offset ? 1 : 0);
            const uint32_t wFirst = wInFirst * 4 + ow - cropLeft_;
            const uint32_t localOff = (wFirst - cw0) * channels_;

            const uint32_t inBatch = baseBatch + ow * outN_;
            const uint32_t inBase = (inBatch * inH_inW + baseH + wInFirst) * channels_;

            params.blockCount = static_cast<uint16_t>(k);
            params.blockLen = static_cast<uint16_t>(cBytesDb);
            
            // Vector 引擎执行跨步写入
            AscendC::DataCopy(obuf[localOff], ibuf[ibufOffset], params);
            ibufOffset += k * channels_;
        }
        
        inQueue_.FreeTensor(ibuf);

        // ================= 阶段三：连续大块写回 GM (MTE3) =================
        outQueue_.EnQue(obuf);
        obuf = outQueue_.template DeQue<DT_X>();
        
        const uint32_t outBase = ((n * outH_ + h) * outW_ + cw0) * channels_;
        AscendC::DataCopy(yGm_[outBase], obuf, segCols * channels_);
        outQueue_.FreeTensor(obuf);
    }

    __aicore__ inline void ProcessAllCellsOpt(uint32_t begin, uint32_t end) {
        uint32_t w = begin % outW_;
        uint32_t tmpIdx = begin / outW_;
        uint32_t h = tmpIdx % outH_;
        uint32_t n = tmpIdx / outH_;

        uint32_t hPad = h + cropTop_;
        uint32_t wPad = w + cropLeft_;
        
        uint32_t hIn = hPad / blockSize_;
        uint32_t offsetH = hPad % blockSize_;
        uint32_t wIn = wPad / blockSize_;
        uint32_t offsetW = wPad % blockSize_;

        const uint32_t resetWPad = cropLeft_;
        const uint32_t resetWIn = resetWPad / blockSize_;
        const uint32_t resetOffsetW = resetWPad % blockSize_;

        const uint32_t resetHPad = cropTop_;
        const uint32_t resetHIn = resetHPad / blockSize_;
        const uint32_t resetOffsetH = resetHPad % blockSize_;

        // --- 核心优化 1：计算 32B 对齐的插槽大小 (Padded Slot) ---
        const uint32_t cBytes = channels_ * sizeof(DT_X);
        const uint32_t alignedBytes = ((cBytes + 31) / 32) * 32;
        const uint32_t alignedElems = alignedBytes / sizeof(DT_X);

        // 计算单次能容纳的最大批量 (Batch Size)
        uint32_t maxBatch = MAX_CHUNK / alignedElems;
        if (maxBatch == 0) maxBatch = 1;

        uint32_t batchCount = 0;
        uint32_t batchStartCell = begin;

        // 整个 Batch 只分配一次 UB 内存
        AscendC::LocalTensor<DT_X> tmp = inQueue_.template AllocTensor<DT_X>();

        AscendC::DataCopyExtParams copyParams{1, cBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};

        for (uint32_t cell = begin; cell < end; ++cell) {
            uint32_t inBatch = (offsetH * blockSize_ + offsetW) * outN_ + n;
            uint32_t inBase = ((inBatch * inH_ + hIn) * inW_ + wIn) * channels_;
            
            // --- 核心优化 2：异步批量读取 (MTE2) ---
            // 连发读指令，将像素放入对齐的插槽 tmp[batchCount * alignedElems]
            AscendC::DataCopyPad(tmp[batchCount * alignedElems], xGm_[inBase], copyParams, padParams);

            batchCount++;

            // 纯加法进位状态机 (性能极佳，保持不变)
            w++; wPad++; offsetW++;
            if (offsetW == blockSize_) { offsetW = 0; wIn++; }
            if (w == outW_) {
                w = 0; wPad = resetWPad; wIn = resetWIn; offsetW = resetOffsetW;
                h++; hPad++; offsetH++;
                if (offsetH == blockSize_) { offsetH = 0; hIn++; }
                if (h == outH_) {
                    h = 0; hPad = resetHPad; hIn = resetHIn; offsetH = resetOffsetH;
                    n++;
                }
            }

            // --- 核心优化 3：Batch 攒满或到达末尾时，统一同步并批量写出 ---
            if (batchCount == maxBatch || cell == end - 1) {
                // 1. 等待这几百个读指令全部完成
                AscendC::PipeBarrier<PIPE_ALL>();

                // 2. 异步批量写回 (MTE3)
                // 输出地址由于是线性连续的，可以通过 batchStartCell 简单推导
                for (uint32_t i = 0; i < batchCount; ++i) {
                    uint32_t outBase = (batchStartCell + i) * channels_;
                    // 从带 padding 的插槽中精确写回 channels 长度到 GM，完美避开重排开销
                    AscendC::DataCopyPad(yGm_[outBase], tmp[i * alignedElems], copyParams);
                }

                // 3. 等待这几百个写指令全部完成
                AscendC::PipeBarrier<PIPE_ALL>();

                // 更新下一个 Batch 的起始输出位置
                batchStartCell = cell + 1;
                batchCount = 0;
            }
        }
        
        // 循环结束释放内存
        inQueue_.FreeTensor(tmp);
    }

private:
    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueue_;

    uint32_t inN_ = 0, inH_ = 0, inW_ = 0;
    uint32_t outN_ = 0, outH_ = 0, outW_ = 0;
    uint32_t channels_ = 0, blockSize_ = 0, cropTop_ = 0, cropLeft_ = 0;
    uint32_t alignedMode_ = 0, totalUnits_ = 0, unitsPerCore_ = 0;
    uint32_t colsPerTile_ = 1, numColChunks_ = 1;
};

template <typename DT_X>
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingData, tiling_data, tiling);
    KernelBatchToSpace<DT_X> op;
    op.Init(x, y, tiling_data);
    op.Process();
}