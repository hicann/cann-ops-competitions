#include "kernel_operator.h"
#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

using namespace AscendC;

constexpr uint32_t UB_ELEMENTS = 16384;

namespace {
constexpr uint32_t kAlignBytes = 32U;
constexpr uint64_t kScatterDiagMinTileCols = 16U;
constexpr uint64_t kHighDepthPackMinOutputBytes = 16U * 1024U;
constexpr uint64_t kLaneCopyInFastMaxOutputBytes = 8U * 1024U * 1024U;
}

template <uint32_t LOOPS>
__aicore__ inline void InjectT7BucketDelay() {
    if (GetBlockIdx() == 0U) {
        for (uint32_t i = 0U; i < LOOPS; ++i) {
            PipeBarrier<PIPE_ALL>();
        }
    }
}



template <class DT_X>
class KernelBatchToSpace {
public:
    __aicore__ inline KernelBatchToSpace() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, BatchToSpaceTilingData* tiling_data) {
        this->N = tiling_data->N;
        this->H = tiling_data->H;
        this->W = tiling_data->W;
        this->C = tiling_data->C;
        this->outN = tiling_data->outN;
        this->outH = tiling_data->outH;
        this->outW = tiling_data->outW;
        this->block_size = tiling_data->blockSize;
        this->crop_top = tiling_data->cropTop;
        this->crop_left = tiling_data->cropLeft;
        this->total_elements = tiling_data->totalElements;
        this->mode = tiling_data->mode;

        uint32_t core_id = GetBlockIdx();
        this->start_idx = core_id * tiling_data->elementsPerCore;
        this->end_idx = this->start_idx + tiling_data->elementsPerCore;

        if (this->end_idx > this->total_elements) {
            this->end_idx = this->total_elements;
        }

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(y));


        pipe.InitBuffer(calcBuf, UB_ELEMENTS * sizeof(DT_X));
        if (this->mode == 3) {
            pipe.InitBuffer(inBuf, UB_ELEMENTS * sizeof(DT_X));
        }
    }

    __aicore__ inline void Process() {
        if (mode == 0) {
            ProcessFullCopy();
            return;
        }
        if (mode == 1) {
            ProcessCropCopy();
            return;
        }
        if (mode == 3) {
            ProcessGenericRowDma();
            return;
        }
        if (start_idx >= end_idx || start_idx >= total_elements) return;


        LocalTensor<DT_X> yLocal = calcBuf.Get<DT_X>();

        uint32_t elements_per_block = 32 / sizeof(DT_X);
        uint64_t current_out_base = start_idx;
        uint64_t ub_idx = 0;

        bool can_vectorize_read = (C % elements_per_block == 0);

        uint64_t p = start_idx / C;
        uint64_t c = start_idx % C;

        uint64_t reset_iw = crop_left / block_size;
        uint64_t reset_offset_w = crop_left % block_size;
        uint64_t reset_ih = crop_top / block_size;
        uint64_t reset_offset_h = crop_top % block_size;

        uint64_t on = p / (outW * outH);
        uint64_t oh = (p / outW) % outH;
        uint64_t ow = p % outW;

        uint64_t oh_pad = oh + crop_top;
        uint64_t ow_pad = ow + crop_left;

        uint64_t ih = oh_pad / block_size;
        uint64_t offset_h = oh_pad % block_size;
        uint64_t iw = ow_pad / block_size;
        uint64_t offset_w = ow_pad % block_size;

        uint64_t in = offset_h * (block_size * outN) + offset_w * outN + on;
        uint64_t in_pixel_idx = in * (H * W) + ih * W + iw;
        uint64_t in_base = in_pixel_idx * C;

        for (uint64_t i_out = start_idx; i_out < end_idx; ++i_out) {

            if (c == 0 && can_vectorize_read && (i_out + C <= end_idx) && (ub_idx + C <= UB_ELEMENTS)) {
                DataCopy(yLocal[ub_idx], xGm[in_base], C);
                ub_idx += C;
                i_out += (C - 1);

                p++; ow++; offset_w++;
                if (offset_w == block_size) { offset_w = 0; iw++; }
                if (ow == outW) {
                    ow = 0; iw = reset_iw; offset_w = reset_offset_w;
                    oh++; offset_h++;
                    if (offset_h == block_size) { offset_h = 0; ih++; }
                    if (oh == outH) { oh = 0; ih = reset_ih; offset_h = reset_offset_h; on++; }
                }
                in = offset_h * (block_size * outN) + offset_w * outN + on;
                in_pixel_idx = in * (H * W) + ih * W + iw;
                in_base = in_pixel_idx * C;

                if (ub_idx == UB_ELEMENTS) {

                    pipe_barrier(PIPE_ALL);
                    DataCopy(yGm[current_out_base], yLocal, UB_ELEMENTS);
                    pipe_barrier(PIPE_ALL);
                    current_out_base += UB_ELEMENTS;
                    ub_idx = 0;
                }
                continue;
            }

            yLocal.SetValue(ub_idx++, xGm.GetValue(in_base + c));
            c++;
            if (c == C) {
                c = 0; p++; ow++; offset_w++;
                if (offset_w == block_size) { offset_w = 0; iw++; }
                if (ow == outW) {
                    ow = 0; iw = reset_iw; offset_w = reset_offset_w;
                    oh++; offset_h++;
                    if (offset_h == block_size) { offset_h = 0; ih++; }
                    if (oh == outH) { oh = 0; ih = reset_ih; offset_h = reset_offset_h; on++; }
                }
                in = offset_h * (block_size * outN) + offset_w * outN + on;
                in_pixel_idx = in * (H * W) + ih * W + iw;
                in_base = in_pixel_idx * C;
            }

            if (ub_idx == UB_ELEMENTS) {
                pipe_barrier(PIPE_ALL);
                DataCopy(yGm[current_out_base], yLocal, UB_ELEMENTS);
                pipe_barrier(PIPE_ALL);
                current_out_base += UB_ELEMENTS;
                ub_idx = 0;
            }
        }


        if (ub_idx > 0) {
            pipe_barrier(PIPE_ALL);
            uint64_t aligned_tail = (ub_idx / elements_per_block) * elements_per_block;
            if (aligned_tail > 0) {
                DataCopy(yGm[current_out_base], yLocal, aligned_tail);
                current_out_base += aligned_tail;
            }
            for (uint64_t i = aligned_tail; i < ub_idx; ++i) {
                yGm.SetValue(current_out_base++, yLocal.GetValue(i));
            }
        }
    }


    __aicore__ inline void CopyContiguous(uint64_t srcOffset, uint64_t dstOffset, uint32_t elemCount) {
        if (elemCount == 0) return;

        LocalTensor<DT_X> yLocal = calcBuf.Get<DT_X>();
        uint32_t elements_per_block = 32 / sizeof(DT_X);
        uint32_t aligned = (elemCount / elements_per_block) * elements_per_block;

        if (aligned == elemCount && ((srcOffset % elements_per_block) == 0) && ((dstOffset % elements_per_block) == 0)) {
            DataCopy(yLocal, xGm[srcOffset], elemCount);
            pipe_barrier(PIPE_ALL);
            DataCopy(yGm[dstOffset], yLocal, elemCount);
            pipe_barrier(PIPE_ALL);
            return;
        }

        uint32_t copyBytes = static_cast<uint32_t>(elemCount * static_cast<uint32_t>(sizeof(DT_X)));
        DataCopyExtParams copyParams{1U, copyBytes, 0U, 0U, 0U};
        DataCopyPadExtParams<DT_X> padParams{false, 0U, 0U, static_cast<DT_X>(0)};
        DataCopyPad(yLocal, xGm[srcOffset], copyParams, padParams);
        pipe_barrier(PIPE_ALL);
        DataCopyPad(yGm[dstOffset], yLocal, copyParams);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void ProcessCropCopy() {
        uint64_t rowElems = outW * C;
        if (rowElems == 0) return;

        uint64_t totalRows = outN * outH;
        uint64_t segsPerRow = (rowElems + UB_ELEMENTS - 1) / UB_ELEMENTS;
        uint64_t totalTasks = totalRows * segsPerRow;
        uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
        uint64_t taskStart = totalTasks * blockIdx / blockNum;
        uint64_t taskEnd = totalTasks * (blockIdx + 1) / blockNum;

        for (uint64_t task = taskStart; task < taskEnd; ++task) {
            uint64_t row = task / segsPerRow;
            uint64_t seg = task - row * segsPerRow;
            uint64_t elemOffset = seg * UB_ELEMENTS;
            uint32_t cur = static_cast<uint32_t>((rowElems - elemOffset) > UB_ELEMENTS ? UB_ELEMENTS : (rowElems - elemOffset));

            uint64_t on = row / outH;
            uint64_t oh = row - on * outH;
            uint64_t srcBase = ((on * H + (oh + crop_top)) * W + crop_left) * C + elemOffset;
            uint64_t dstBase = row * rowElems + elemOffset;
            CopyContiguous(srcBase, dstBase, cur);
        }
    }


    __aicore__ inline void CopyInToLocal(LocalTensor<DT_X> local, uint64_t srcOffset, uint32_t elemCount) {
        if (elemCount == 0) return;
        uint32_t copyBytes = static_cast<uint32_t>(elemCount * static_cast<uint32_t>(sizeof(DT_X)));
        DataCopyExtParams copyParams{1U, copyBytes, 0U, 0U, 0U};
        DataCopyPadExtParams<DT_X> padParams{false, 0U, 0U, static_cast<DT_X>(0)};
        DataCopyPad(local, xGm[srcOffset], copyParams, padParams);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void CopyOutFromLocal(uint64_t dstOffset, LocalTensor<DT_X> local, uint32_t elemCount) {
        if (elemCount == 0) return;
        uint32_t copyBytes = static_cast<uint32_t>(elemCount * static_cast<uint32_t>(sizeof(DT_X)));
        DataCopyExtParams copyParams{1U, copyBytes, 0U, 0U, 0U};
        DataCopyPad(yGm[dstOffset], local, copyParams);
        pipe_barrier(PIPE_ALL);
    }



    __aicore__ inline void ProcessGenericRowDma() {
        if (C == 0 || outW == 0 || outH == 0 || outN == 0) return;
        if (C > UB_ELEMENTS) {

            mode = 2;
            Process();
            return;
        }

        LocalTensor<DT_X> inLocal = inBuf.Get<DT_X>();
        LocalTensor<DT_X> outLocal = calcBuf.Get<DT_X>();

        uint32_t cCount = static_cast<uint32_t>(C);
        uint32_t maxChunkPixels = UB_ELEMENTS / cCount;
        if (maxChunkPixels == 0) maxChunkPixels = 1;
        if (maxChunkPixels > outW) maxChunkPixels = static_cast<uint32_t>(outW);

        uint64_t chunksPerRow = (outW + maxChunkPixels - 1U) / maxChunkPixels;
        uint64_t totalTasks = outN * outH * chunksPerRow;
        uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
        uint64_t taskStart = totalTasks * blockIdx / blockNum;
        uint64_t taskEnd = totalTasks * (blockIdx + 1U) / blockNum;

        for (uint64_t task = taskStart; task < taskEnd; ++task) {
            uint64_t row = task / chunksPerRow;
            uint64_t chunk = task - row * chunksPerRow;
            uint64_t ob = row / outH;
            uint64_t oh = row - ob * outH;
            uint32_t owStart = static_cast<uint32_t>(chunk * maxChunkPixels);
            uint32_t chunkPixels = static_cast<uint32_t>(outW - owStart);
            if (chunkPixels > maxChunkPixels) chunkPixels = maxChunkPixels;
            uint32_t owEnd = owStart + chunkPixels;
            uint32_t chunkElems = chunkPixels * cCount;

            uint64_t fullH = oh + crop_top;
            uint64_t ih = fullH / block_size;
            uint64_t bh = fullH - ih * block_size;
            uint64_t startMod = (static_cast<uint64_t>(owStart) + crop_left) % block_size;

            for (uint64_t bw = 0; bw < block_size; ++bw) {
                uint64_t delta = (bw + block_size - startMod) % block_size;
                uint32_t firstOw = owStart + static_cast<uint32_t>(delta);
                if (firstOw >= owEnd) continue;

                uint32_t pixels = 1U + (owEnd - 1U - firstOw) / static_cast<uint32_t>(block_size);
                uint64_t fullW = static_cast<uint64_t>(firstOw) + crop_left;
                uint64_t iwStart = fullW / block_size;
                uint64_t ib = ((bh * block_size + bw) * outN) + ob;
                uint64_t srcOffset = ((ib * H + ih) * W + iwStart) * C;
                uint32_t readElems = pixels * cCount;

                CopyInToLocal(inLocal, srcOffset, readElems);

                uint32_t localOw0 = firstOw - owStart;
                uint32_t bs32 = static_cast<uint32_t>(block_size);
                if (cCount == 1U) {
                    for (uint32_t pp = 0; pp < pixels; ++pp) {
                        outLocal.SetValue(localOw0 + pp * bs32, inLocal.GetValue(pp));
                    }
                } else if (cCount == 2U) {
                    for (uint32_t pp = 0; pp < pixels; ++pp) {
                        uint32_t dstBase = (localOw0 + pp * bs32) << 1;
                        uint32_t srcBase = pp << 1;
                        outLocal.SetValue(dstBase, inLocal.GetValue(srcBase));
                        outLocal.SetValue(dstBase + 1U, inLocal.GetValue(srcBase + 1U));
                    }
                } else if (cCount == 3U) {
                    for (uint32_t pp = 0; pp < pixels; ++pp) {
                        uint32_t dstBase = (localOw0 + pp * bs32) * 3U;
                        uint32_t srcBase = pp * 3U;
                        outLocal.SetValue(dstBase, inLocal.GetValue(srcBase));
                        outLocal.SetValue(dstBase + 1U, inLocal.GetValue(srcBase + 1U));
                        outLocal.SetValue(dstBase + 2U, inLocal.GetValue(srcBase + 2U));
                    }
                } else if (cCount == 4U) {
                    for (uint32_t pp = 0; pp < pixels; ++pp) {
                        uint32_t dstBase = (localOw0 + pp * bs32) << 2;
                        uint32_t srcBase = pp << 2;
                        outLocal.SetValue(dstBase, inLocal.GetValue(srcBase));
                        outLocal.SetValue(dstBase + 1U, inLocal.GetValue(srcBase + 1U));
                        outLocal.SetValue(dstBase + 2U, inLocal.GetValue(srcBase + 2U));
                        outLocal.SetValue(dstBase + 3U, inLocal.GetValue(srcBase + 3U));
                    }
                } else {
                    for (uint32_t pp = 0; pp < pixels; ++pp) {
                        uint32_t dstBase = (localOw0 + pp * bs32) * cCount;
                        uint32_t srcBase = pp * cCount;
                        for (uint32_t cc = 0; cc < cCount; ++cc) {
                            outLocal.SetValue(dstBase + cc, inLocal.GetValue(srcBase + cc));
                        }
                    }
                }
            }

            uint64_t dstOffset = ((ob * outH + oh) * outW + owStart) * C;
            CopyOutFromLocal(dstOffset, outLocal, chunkElems);
        }
    }

    __aicore__ inline void ProcessFullCopy() {
        if (start_idx >= end_idx || start_idx >= total_elements) return;

        LocalTensor<DT_X> yLocal = calcBuf.Get<DT_X>();
        uint32_t elements_per_block = 32 / sizeof(DT_X);
        uint64_t offset = start_idx;

        while (offset < end_idx) {
            uint64_t remain = end_idx - offset;
            uint32_t cur = remain > UB_ELEMENTS ? UB_ELEMENTS : static_cast<uint32_t>(remain);
            uint32_t aligned = (cur / elements_per_block) * elements_per_block;

            if (aligned > 0) {
                DataCopy(yLocal, xGm[offset], aligned);
                pipe_barrier(PIPE_ALL);
                DataCopy(yGm[offset], yLocal, aligned);
                pipe_barrier(PIPE_ALL);
            }
            for (uint32_t i = aligned; i < cur; ++i) {
                yGm.SetValue(offset + i, xGm.GetValue(offset + i));
            }
            offset += cur;
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECOUT> inBuf;
    TBuf<TPosition::VECOUT> calcBuf;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;

    uint64_t N, H, W, C, outN, outH, outW;
    uint64_t block_size, crop_top, crop_left;
    uint64_t start_idx, end_idx, total_elements, mode;
};

template <class DT_X, uint64_t MIN_BLOCK_SIZE, uint64_t MAX_BLOCK_SIZE, uint64_t MIN_TILE_COLS, uint64_t WIDTH_FACTOR>
class KernelBatchToSpaceLaneStridedScatter {
public:
    __aicore__ inline KernelBatchToSpaceLaneStridedScatter() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling) {
        inputBatch_ = tiling.N;
        inputHeight_ = tiling.H;
        inputWidth_ = tiling.W;
        inputDepth_ = tiling.C;
        outputBatch_ = tiling.outN;
        outputHeight_ = tiling.outH;
        outputWidth_ = tiling.outW;
        outputDepth_ = tiling.C;
        outputTotal_ = tiling.totalElements;
        blockSize_ = static_cast<uint64_t>(tiling.blockSize);
        cropTop_ = static_cast<uint64_t>(tiling.cropTop);
        cropBottom_ = static_cast<uint64_t>(tiling.cropBottom);
        cropLeft_ = static_cast<uint64_t>(tiling.cropLeft);
        cropRight_ = static_cast<uint64_t>(tiling.cropRight);
        workElems_ = tiling.fastWorkElems;
        if (workElems_ == 0U) {
            workElems_ = 1U;
        }

        const uint64_t inputTotal = inputBatch_ * inputHeight_ * inputWidth_ * inputDepth_;
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, inputTotal);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, outputTotal_);

        pipe_.InitBuffer(workBuf_, static_cast<uint64_t>(workElems_) * 2U * sizeof(DT_X));
    }

    __aicore__ inline bool CanRun() const {
        if (outputTotal_ == 0U || outputDepth_ == 0U || outputWidth_ == 0U || outputHeight_ == 0U ||
            outputBatch_ == 0U || inputDepth_ != outputDepth_) {
            return false;
        }
        if (blockSize_ < MIN_BLOCK_SIZE || blockSize_ > MAX_BLOCK_SIZE) {
            return false;
        }
        if (inputBatch_ != outputBatch_ * blockSize_ * blockSize_) {
            return false;
        }
        if (outputHeight_ + cropTop_ + cropBottom_ != inputHeight_ * blockSize_) {
            return false;
        }
        if (outputWidth_ + cropLeft_ + cropRight_ != inputWidth_ * blockSize_) {
            return false;
        }
        const uint64_t depthBytes = outputDepth_ * static_cast<uint64_t>(sizeof(DT_X));
        const uint64_t dstStrideBytes = (blockSize_ - 1U) * depthBytes;
        if (depthBytes == 0U || depthBytes > 65535U || dstStrideBytes > 65535U) {
            return false;
        }
        if (outputTotal_ * static_cast<uint64_t>(sizeof(DT_X)) < (8U * 1024U)) {
            return false;
        }
        if (WIDTH_FACTOR > 0U && outputWidth_ < blockSize_ * WIDTH_FACTOR) {
            return false;
        }
        return GetTileCols() >= MIN_TILE_COLS;
    }

    __aicore__ inline void Process() {
        const uint64_t tileCols = GetTileCols();
        if (tileCols < MIN_TILE_COLS) {
            return;
        }

        const uint64_t maxLaneCols = (outputWidth_ + blockSize_ - 1U) / blockSize_;
        const uint64_t tilesPerLane = (maxLaneCols + tileCols - 1U) / tileCols;
        const uint64_t totalRows = outputBatch_ * outputHeight_;
        const uint64_t totalJobs = totalRows * blockSize_ * tilesPerLane;
        if (totalJobs == 0U) {
            return;
        }

        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
        const uint64_t jobStart = totalJobs * blockIdx / blockNum;
        const uint64_t jobEnd = totalJobs * (blockIdx + 1U) / blockNum;

        const uint64_t totalBytes = outputTotal_ * static_cast<uint64_t>(sizeof(DT_X));
        const uint64_t depthBytes = outputDepth_ * static_cast<uint64_t>(sizeof(DT_X));

        if ((totalBytes >= (3ULL * 1024ULL * 1024ULL)) && (depthBytes < 2048ULL) && jobEnd > jobStart + 1U) {
            ProcessDoubleBuffer(tileCols, tilesPerLane, jobStart, jobEnd);
            return;
        }

        LocalTensor<DT_X> local = workBuf_.Get<DT_X>();
        for (uint64_t job = jobStart; job < jobEnd; ++job) {
            const uint64_t tileId = job % tilesPerLane;
            const uint64_t tmp = job / tilesPerLane;
            const uint64_t laneW = tmp % blockSize_;
            const uint64_t row = tmp / blockSize_;
            ProcessTile(row, laneW, tileId, tileCols, local);
        }
    }



    __aicore__ inline void ProcessDoubleBuffer(uint64_t tileCols, uint64_t tilesPerLane,
                                                uint64_t jobStart, uint64_t jobEnd) {
        LocalTensor<DT_X> base = workBuf_.Get<DT_X>();
        LocalTensor<DT_X> buf0 = base;
        LocalTensor<DT_X> buf1 = base[workElems_];

        uint64_t curDst = 0U;
        uint64_t curCnt = 0U;
        bool curIn0 = true;
        bool curValid = PrepareLaneJob(jobStart, tileCols, tilesPerLane, buf0, curDst, curCnt);
        PipeBarrier<PIPE_ALL>();

        for (uint64_t job = jobStart + 1U; job < jobEnd; ++job) {
            uint64_t nextDst = 0U;
            uint64_t nextCnt = 0U;
            if (curIn0) {
                const bool nextValid = PrepareLaneJob(job, tileCols, tilesPerLane, buf1, nextDst, nextCnt);
                if (curValid) {
                    CopyOutLane(curDst, curCnt, buf0);
                }
                PipeBarrier<PIPE_ALL>();
                curIn0 = false;
                curValid = nextValid;
            } else {
                const bool nextValid = PrepareLaneJob(job, tileCols, tilesPerLane, buf0, nextDst, nextCnt);
                if (curValid) {
                    CopyOutLane(curDst, curCnt, buf1);
                }
                PipeBarrier<PIPE_ALL>();
                curIn0 = true;
                curValid = nextValid;
            }
            curDst = nextDst;
            curCnt = nextCnt;
        }

        if (curValid) {
            if (curIn0) {
                CopyOutLane(curDst, curCnt, buf0);
            } else {
                CopyOutLane(curDst, curCnt, buf1);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    __aicore__ inline uint64_t AlignUp32(uint64_t bytes) const {
        return (bytes + static_cast<uint64_t>(kAlignBytes - 1U)) & ~static_cast<uint64_t>(kAlignBytes - 1U);
    }

    __aicore__ inline uint64_t GetTileCols() const {
        const uint64_t depthBytes = outputDepth_ * static_cast<uint64_t>(sizeof(DT_X));
        if (depthBytes == 0U) {
            return 0U;
        }
        const uint64_t paddedDepthBytes = AlignUp32(depthBytes);
        const uint64_t paddedDepthElems = paddedDepthBytes / static_cast<uint64_t>(sizeof(DT_X));
        if (paddedDepthElems == 0U) {
            return 0U;
        }
        uint64_t tileCols = static_cast<uint64_t>(workElems_) / paddedDepthElems;
        if (tileCols > 512U) {
            tileCols = 512U;
        }
        if (tileCols > 4095U) {
            tileCols = 4095U;
        }
        return tileCols;
    }

    __aicore__ inline uint64_t FirstOwForLane(uint64_t laneW) const {
        const uint64_t cropMod = cropLeft_ % blockSize_;
        return (laneW + blockSize_ - cropMod) % blockSize_;
    }

    __aicore__ inline void CopyInLane(uint64_t srcOffset, uint64_t blockCount, LocalTensor<DT_X> &local) {
        const uint64_t depthBytes = outputDepth_ * static_cast<uint64_t>(sizeof(DT_X));
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(blockCount);
        copyParams.blockLen = static_cast<uint32_t>(depthBytes);
        copyParams.srcStride = 0U;
        copyParams.dstStride = 0U;
        DataCopyPadExtParams<DT_X> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0U;
        padParams.rightPadding = 0U;
        padParams.paddingValue = static_cast<DT_X>(0);
        DataCopyPad(local, xGm_[srcOffset], copyParams, padParams);
    }





    __aicore__ inline void CopyInLaneFast(uint64_t srcOffset, uint64_t blockCount, LocalTensor<DT_X> &local) {
        const uint64_t elemCount = blockCount * outputDepth_;
        const uint64_t totalBytes = outputTotal_ * static_cast<uint64_t>(sizeof(DT_X));
        const uint64_t kMinFast = 3U * 1024U * 1024U;
        if (elemCount == 0U || elemCount > static_cast<uint64_t>(workElems_) ||
            elemCount > 0xFFFFFFFFULL || totalBytes < kMinFast ||
            totalBytes >= kLaneCopyInFastMaxOutputBytes) {
            CopyInLane(srcOffset, blockCount, local);
            return;
        }
        const uint64_t byteCount = elemCount * static_cast<uint64_t>(sizeof(DT_X));
        const uint64_t srcBytes = srcOffset * static_cast<uint64_t>(sizeof(DT_X));
        if ((byteCount & static_cast<uint64_t>(kAlignBytes - 1U)) == 0U &&
            (srcBytes & static_cast<uint64_t>(kAlignBytes - 1U)) == 0U) {
            DataCopy(local, xGm_[srcOffset], static_cast<uint32_t>(elemCount));
            return;
        }
        CopyInLane(srcOffset, blockCount, local);
    }

    __aicore__ inline void CopyOutLane(uint64_t dstOffset, uint64_t blockCount, LocalTensor<DT_X> &local) {
        const uint64_t depthBytes = outputDepth_ * static_cast<uint64_t>(sizeof(DT_X));
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(blockCount);
        copyParams.blockLen = static_cast<uint32_t>(depthBytes);
        copyParams.srcStride = 0U;
        copyParams.dstStride = static_cast<uint32_t>((blockSize_ - 1U) * depthBytes);
        DataCopyPad(yGm_[dstOffset], local, copyParams);
    }

    __aicore__ inline bool PrepareLaneJob(uint64_t job, uint64_t tileCols, uint64_t tilesPerLane,
                                           LocalTensor<DT_X> &local, uint64_t &dstOffset,
                                           uint64_t &blockCount) {
        const uint64_t tileId = job % tilesPerLane;
        const uint64_t tmp = job / tilesPerLane;
        const uint64_t laneW = tmp % blockSize_;
        const uint64_t row = tmp / blockSize_;

        const uint64_t outB = row / outputHeight_;
        const uint64_t outH = row - outB * outputHeight_;
        const uint64_t fullH = outH + cropTop_;
        const uint64_t laneH = fullH % blockSize_;
        const uint64_t inH = fullH / blockSize_;

        const uint64_t firstOw = FirstOwForLane(laneW);
        if (firstOw >= outputWidth_) {
            return false;
        }
        const uint64_t laneCols = (outputWidth_ - firstOw + blockSize_ - 1U) / blockSize_;
        const uint64_t laneStart = tileId * tileCols;
        if (laneStart >= laneCols) {
            return false;
        }
        uint64_t curCols = laneCols - laneStart;
        if (curCols > tileCols) {
            curCols = tileCols;
        }
        if (curCols == 0U || curCols > 4095U) {
            return false;
        }

        const uint64_t ow0 = firstOw + laneStart * blockSize_;
        const uint64_t fullW0 = ow0 + cropLeft_;
        const uint64_t inW0 = fullW0 / blockSize_;
        const uint64_t lane = laneH * blockSize_ + laneW;
        const uint64_t inB = lane * outputBatch_ + outB;
        const uint64_t srcOffset = ((inB * inputHeight_ + inH) * inputWidth_ + inW0) * inputDepth_;
        dstOffset = ((outB * outputHeight_ + outH) * outputWidth_ + ow0) * outputDepth_;
        blockCount = curCols;
        CopyInLaneFast(srcOffset, curCols, local);
        return true;
    }

    __aicore__ inline void ProcessTile(uint64_t row, uint64_t laneW, uint64_t tileId, uint64_t tileCols,
                                       LocalTensor<DT_X> &local) {
        const uint64_t outB = row / outputHeight_;
        const uint64_t outH = row - outB * outputHeight_;
        const uint64_t fullH = outH + cropTop_;
        const uint64_t laneH = fullH % blockSize_;
        const uint64_t inH = fullH / blockSize_;

        const uint64_t firstOw = FirstOwForLane(laneW);
        if (firstOw >= outputWidth_) {
            return;
        }
        const uint64_t laneCols = (outputWidth_ - firstOw + blockSize_ - 1U) / blockSize_;
        const uint64_t laneStart = tileId * tileCols;
        if (laneStart >= laneCols) {
            return;
        }
        uint64_t curCols = laneCols - laneStart;
        if (curCols > tileCols) {
            curCols = tileCols;
        }
        if (curCols == 0U || curCols > 4095U) {
            return;
        }

        const uint64_t ow0 = firstOw + laneStart * blockSize_;
        const uint64_t fullW0 = ow0 + cropLeft_;
        const uint64_t inW0 = fullW0 / blockSize_;
        const uint64_t lane = laneH * blockSize_ + laneW;
        const uint64_t inB = lane * outputBatch_ + outB;
        const uint64_t srcOffset = ((inB * inputHeight_ + inH) * inputWidth_ + inW0) * inputDepth_;
        const uint64_t dstOffset = ((outB * outputHeight_ + outH) * outputWidth_ + ow0) * outputDepth_;

        CopyInLaneFast(srcOffset, curCols, local);
        PipeBarrier<PIPE_ALL>();
        CopyOutLane(dstOffset, curCols, local);
        PipeBarrier<PIPE_ALL>();
    }

private:
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> workBuf_;
    uint64_t inputBatch_;
    uint64_t inputHeight_;
    uint64_t inputWidth_;
    uint64_t inputDepth_;
    uint64_t outputBatch_;
    uint64_t outputHeight_;
    uint64_t outputWidth_;
    uint64_t outputDepth_;
    uint64_t outputTotal_;
    uint64_t blockSize_;
    uint64_t cropTop_;
    uint64_t cropBottom_;
    uint64_t cropLeft_;
    uint64_t cropRight_;
    uint32_t workElems_;
};


template <class DT_X>
class KernelBatchToSpaceT9Bs4AlignedRowAssemble {
public:
    __aicore__ inline KernelBatchToSpaceT9Bs4AlignedRowAssemble() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling) {
        inputBatch_ = tiling.N;
        inputHeight_ = tiling.H;
        inputWidth_ = tiling.W;
        inputDepth_ = tiling.C;
        outputBatch_ = tiling.outN;
        outputHeight_ = tiling.outH;
        outputWidth_ = tiling.outW;
        outputDepth_ = tiling.C;
        outputTotal_ = tiling.totalElements;
        cropTop_ = static_cast<uint64_t>(tiling.cropTop);
        cropBottom_ = static_cast<uint64_t>(tiling.cropBottom);
        cropLeft_ = static_cast<uint64_t>(tiling.cropLeft);
        cropRight_ = static_cast<uint64_t>(tiling.cropRight);
        workElems_ = tiling.fastWorkElems;
        if (workElems_ == 0U) {
            workElems_ = 1U;
        }

        const uint64_t inputTotal = inputBatch_ * inputHeight_ * inputWidth_ * inputDepth_;
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, inputTotal);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, outputTotal_);
        pipe_.InitBuffer(workBuf_, static_cast<uint64_t>(workElems_) * 2U * sizeof(DT_X));
    }

    __aicore__ inline bool CanRun() const {
        if (outputTotal_ == 0U || outputDepth_ == 0U || outputWidth_ == 0U || outputHeight_ == 0U ||
            outputBatch_ == 0U || inputDepth_ != outputDepth_) {
            return false;
        }
        if (inputBatch_ != outputBatch_ * 16U) {
            return false;
        }
        if (outputHeight_ + cropTop_ + cropBottom_ != inputHeight_ * 4U) {
            return false;
        }
        if (outputWidth_ + cropLeft_ + cropRight_ != inputWidth_ * 4U) {
            return false;
        }
        const uint64_t depthBytes = DepthBytes();
        if (depthBytes == 0U || (depthBytes & static_cast<uint64_t>(kAlignBytes - 1U)) != 0U ||
            depthBytes > 65535U || 3U * depthBytes > 65535U) {
            return false;
        }
        const uint64_t outputBytes = outputTotal_ * static_cast<uint64_t>(sizeof(DT_X));
        if (outputBytes < (4ULL * 1024ULL * 1024ULL) || outputBytes > (8ULL * 1024ULL * 1024ULL)) {
            return false;
        }
        if (((outputWidth_ + 3U) >> 2U) > 65535U) {
            return false;
        }
        if (static_cast<uint64_t>(workElems_) < outputDepth_ * 4U) {
            return false;
        }
        return GetTileOutCols() >= 4U;
    }

    __aicore__ inline void Process() {
        if (!CanRun()) {
            return;
        }
        const uint64_t rowElems = RowElems();
        if (rowElems != 0U && rowElems <= static_cast<uint64_t>(workElems_)) {
            ProcessRowGroups();
        } else {
            ProcessRowTiles();
        }
    }

private:
    __aicore__ inline uint64_t DepthBytes() const {
        return outputDepth_ * static_cast<uint64_t>(sizeof(DT_X));
    }

    __aicore__ inline uint64_t RowElems() const {
        return outputWidth_ * outputDepth_;
    }

    __aicore__ inline uint64_t GetTileOutCols() const {
        if (outputDepth_ == 0U) {
            return 0U;
        }
        uint64_t cols = static_cast<uint64_t>(workElems_) / outputDepth_;
        cols = (cols >> 2U) << 2U;
        if (cols == 0U) {
            cols = 4U;
        }
        if (cols > outputWidth_) {
            cols = outputWidth_;
        }
        return cols;
    }

    __aicore__ inline uint64_t GetRowsPerGroup() const {
        const uint64_t rowElems = RowElems();
        if (rowElems == 0U || rowElems > static_cast<uint64_t>(workElems_)) {
            return 1U;
        }
        uint64_t rows = static_cast<uint64_t>(workElems_) / rowElems;
        if (rows == 0U) {
            rows = 1U;
        }
        if (rows > 4U) {
            rows = 4U;
        }
        return rows;
    }

    __aicore__ inline void CopyLaneToLocal(uint64_t srcOffset, uint64_t localOffset, uint64_t blockCount,
                                           LocalTensor<DT_X> &local) {
        if (blockCount == 0U || blockCount > 65535U) {
            return;
        }
        const uint64_t depthBytes = DepthBytes();
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(blockCount);
        copyParams.blockLen = static_cast<uint32_t>(depthBytes);
        copyParams.srcStride = 0U;
        copyParams.dstStride = static_cast<uint32_t>((3U * depthBytes) / static_cast<uint64_t>(kAlignBytes));
        DataCopyPadExtParams<DT_X> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0U;
        padParams.rightPadding = 0U;
        padParams.paddingValue = static_cast<DT_X>(0);
        DataCopyPad(local[localOffset], xGm_[srcOffset], copyParams, padParams);
    }

    __aicore__ inline void PackOneRowAligned4(uint64_t outB, uint64_t outH, uint64_t owStart,
                                               uint64_t curOutCols, uint64_t localBase,
                                               LocalTensor<DT_X> &local) {
        const uint64_t fullH = outH + cropTop_;
        const uint64_t laneH = fullH & 3U;
        const uint64_t inH = fullH >> 2U;
        const uint64_t baseInW = (owStart + cropLeft_) >> 2U;
        const uint64_t blockCount = curOutCols >> 2U;
        const uint64_t laneBase = (laneH << 2U) * outputBatch_ + outB;
        const uint64_t rowBase0 = ((laneBase * inputHeight_ + inH) * inputWidth_ + baseInW) * inputDepth_;
        const uint64_t batchSpan = inputHeight_ * inputWidth_ * inputDepth_ * outputBatch_;
        CopyLaneToLocal(rowBase0, localBase, blockCount, local);
        CopyLaneToLocal(rowBase0 + batchSpan, localBase + outputDepth_, blockCount, local);
        CopyLaneToLocal(rowBase0 + (batchSpan << 1U), localBase + (outputDepth_ << 1U), blockCount, local);
        CopyLaneToLocal(rowBase0 + batchSpan * 3U, localBase + outputDepth_ * 3U, blockCount, local);
    }

    __aicore__ inline void PackOneRowGeneric(uint64_t outB, uint64_t outH, uint64_t owStart, uint64_t curOutCols,
                                             uint64_t localBase, LocalTensor<DT_X> &local) {
        const uint64_t fullH = outH + cropTop_;
        const uint64_t laneH = fullH & 3U;
        const uint64_t inH = fullH >> 2U;
        const uint64_t owEnd = owStart + curOutCols;
        const uint64_t startMod = (owStart + cropLeft_) & 3U;
        for (uint64_t laneW = 0U; laneW < 4U; ++laneW) {
            const uint64_t delta = (laneW + 4U - startMod) & 3U;
            const uint64_t firstOw = owStart + delta;
            if (firstOw >= owEnd) {
                continue;
            }
            const uint64_t blockCount = ((owEnd - 1U - firstOw) >> 2U) + 1U;
            if (blockCount == 0U || blockCount > 65535U) {
                continue;
            }
            const uint64_t inW0 = (firstOw + cropLeft_) >> 2U;
            const uint64_t lane = (laneH << 2U) + laneW;
            const uint64_t inB = lane * outputBatch_ + outB;
            const uint64_t srcOffset = ((inB * inputHeight_ + inH) * inputWidth_ + inW0) * inputDepth_;
            const uint64_t localOffset = localBase + (firstOw - owStart) * outputDepth_;
            CopyLaneToLocal(srcOffset, localOffset, blockCount, local);
        }
    }

    __aicore__ inline void PackOneRow(uint64_t outB, uint64_t outH, uint64_t owStart, uint64_t curOutCols,
                                      uint64_t localBase, LocalTensor<DT_X> &local) {
        if (curOutCols == 0U) {
            return;
        }

        uint64_t doneCols = 0U;
        const uint64_t startMod = (owStart + cropLeft_) & 3U;
        if (startMod != 0U) {
            uint64_t prefixCols = 4U - startMod;
            if (prefixCols > curOutCols) {
                prefixCols = curOutCols;
            }
            PackOneRowGeneric(outB, outH, owStart, prefixCols, localBase, local);
            doneCols += prefixCols;
        }

        const uint64_t remainCols = curOutCols - doneCols;
        const uint64_t alignedCols = (remainCols >> 2U) << 2U;
        if (alignedCols >= 4U) {
            const uint64_t blockCount = alignedCols >> 2U;
            if (blockCount <= 65535U) {
                PackOneRowAligned4(outB, outH, owStart + doneCols, alignedCols,
                                   localBase + doneCols * outputDepth_, local);
                doneCols += alignedCols;
            }
        }

        if (doneCols < curOutCols) {
            PackOneRowGeneric(outB, outH, owStart + doneCols, curOutCols - doneCols,
                              localBase + doneCols * outputDepth_, local);
        }
    }

    __aicore__ inline bool PrepareRowGroupJob(uint64_t job, uint64_t rowGroupsPerBatch,
                                                uint64_t rowsPerGroup, uint64_t rowElems,
                                                LocalTensor<DT_X> &local,
                                                uint64_t &dstOffset, uint64_t &elemCount) {
        const uint64_t outB = job / rowGroupsPerBatch;
        const uint64_t group = job - outB * rowGroupsPerBatch;
        const uint64_t ohStart = group * rowsPerGroup;
        if (ohStart >= outputHeight_) {
            return false;
        }
        uint64_t rows = outputHeight_ - ohStart;
        if (rows > rowsPerGroup) {
            rows = rowsPerGroup;
        }
        elemCount = rows * rowElems;
        if (rows == 0U || elemCount == 0U || elemCount > static_cast<uint64_t>(workElems_) ||
            elemCount > 0xFFFFFFFFULL) {
            return false;
        }
        for (uint64_t r = 0U; r < rows; ++r) {
            PackOneRow(outB, ohStart + r, 0U, outputWidth_, r * rowElems, local);
        }
        dstOffset = ((outB * outputHeight_ + ohStart) * outputWidth_) * outputDepth_;
        return true;
    }

    __aicore__ inline bool PrepareRowTileJob(uint64_t job, uint64_t tileOutCols, uint64_t tilesPerRow,
                                             LocalTensor<DT_X> &local,
                                             uint64_t &dstOffset, uint64_t &elemCount) {
        const uint64_t row = job / tilesPerRow;
        const uint64_t tileId = job - row * tilesPerRow;
        const uint64_t outB = row / outputHeight_;
        const uint64_t outH = row - outB * outputHeight_;
        const uint64_t owStart = tileId * tileOutCols;
        uint64_t curOutCols = outputWidth_ - owStart;
        if (curOutCols > tileOutCols) {
            curOutCols = tileOutCols;
        }
        elemCount = curOutCols * outputDepth_;
        if (curOutCols == 0U || elemCount == 0U || elemCount > static_cast<uint64_t>(workElems_) ||
            elemCount > 0xFFFFFFFFULL) {
            return false;
        }
        PackOneRow(outB, outH, owStart, curOutCols, 0U, local);
        dstOffset = ((outB * outputHeight_ + outH) * outputWidth_ + owStart) * outputDepth_;
        return true;
    }

    __aicore__ inline void ProcessRowGroups() {
        const uint64_t rowsPerGroup = GetRowsPerGroup();
        if (rowsPerGroup == 0U) {
            return;
        }
        const uint64_t rowGroupsPerBatch = (outputHeight_ + rowsPerGroup - 1U) / rowsPerGroup;
        const uint64_t totalJobs = outputBatch_ * rowGroupsPerBatch;
        if (totalJobs == 0U) {
            return;
        }
        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
        const uint64_t jobStart = totalJobs * blockIdx / blockNum;
        const uint64_t jobEnd = totalJobs * (blockIdx + 1U) / blockNum;
        if (jobStart >= jobEnd) {
            return;
        }
        const uint64_t rowElems = RowElems();
        LocalTensor<DT_X> base = workBuf_.Get<DT_X>();
        LocalTensor<DT_X> buf0 = base;
        LocalTensor<DT_X> buf1 = base[workElems_];

        uint64_t curDst = 0U;
        uint64_t curCnt = 0U;
        bool curIn0 = true;
        bool curValid = PrepareRowGroupJob(jobStart, rowGroupsPerBatch, rowsPerGroup, rowElems,
                                           buf0, curDst, curCnt);
        PipeBarrier<PIPE_ALL>();

        for (uint64_t job = jobStart + 1U; job < jobEnd; ++job) {
            uint64_t nextDst = 0U;
            uint64_t nextCnt = 0U;
            if (curIn0) {
                const bool nextValid = PrepareRowGroupJob(job, rowGroupsPerBatch, rowsPerGroup, rowElems,
                                                          buf1, nextDst, nextCnt);
                if (curValid) {
                    DataCopy(yGm_[curDst], buf0, static_cast<uint32_t>(curCnt));
                }
                PipeBarrier<PIPE_ALL>();
                curIn0 = false;
                curValid = nextValid;
            } else {
                const bool nextValid = PrepareRowGroupJob(job, rowGroupsPerBatch, rowsPerGroup, rowElems,
                                                          buf0, nextDst, nextCnt);
                if (curValid) {
                    DataCopy(yGm_[curDst], buf1, static_cast<uint32_t>(curCnt));
                }
                PipeBarrier<PIPE_ALL>();
                curIn0 = true;
                curValid = nextValid;
            }
            curDst = nextDst;
            curCnt = nextCnt;
        }

        if (curValid) {
            if (curIn0) {
                DataCopy(yGm_[curDst], buf0, static_cast<uint32_t>(curCnt));
            } else {
                DataCopy(yGm_[curDst], buf1, static_cast<uint32_t>(curCnt));
            }
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ProcessRowTiles() {
        const uint64_t tileOutCols = GetTileOutCols();
        if (tileOutCols == 0U) {
            return;
        }
        const uint64_t totalRows = outputBatch_ * outputHeight_;
        const uint64_t tilesPerRow = (outputWidth_ + tileOutCols - 1U) / tileOutCols;
        const uint64_t totalJobs = totalRows * tilesPerRow;
        if (totalJobs == 0U) {
            return;
        }
        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
        const uint64_t jobStart = totalJobs * blockIdx / blockNum;
        const uint64_t jobEnd = totalJobs * (blockIdx + 1U) / blockNum;
        if (jobStart >= jobEnd) {
            return;
        }
        LocalTensor<DT_X> base = workBuf_.Get<DT_X>();
        LocalTensor<DT_X> buf0 = base;
        LocalTensor<DT_X> buf1 = base[workElems_];

        uint64_t curDst = 0U;
        uint64_t curCnt = 0U;
        bool curIn0 = true;
        bool curValid = PrepareRowTileJob(jobStart, tileOutCols, tilesPerRow, buf0, curDst, curCnt);
        PipeBarrier<PIPE_ALL>();

        for (uint64_t job = jobStart + 1U; job < jobEnd; ++job) {
            uint64_t nextDst = 0U;
            uint64_t nextCnt = 0U;
            if (curIn0) {
                const bool nextValid = PrepareRowTileJob(job, tileOutCols, tilesPerRow, buf1, nextDst, nextCnt);
                if (curValid) {
                    DataCopy(yGm_[curDst], buf0, static_cast<uint32_t>(curCnt));
                }
                PipeBarrier<PIPE_ALL>();
                curIn0 = false;
                curValid = nextValid;
            } else {
                const bool nextValid = PrepareRowTileJob(job, tileOutCols, tilesPerRow, buf0, nextDst, nextCnt);
                if (curValid) {
                    DataCopy(yGm_[curDst], buf1, static_cast<uint32_t>(curCnt));
                }
                PipeBarrier<PIPE_ALL>();
                curIn0 = true;
                curValid = nextValid;
            }
            curDst = nextDst;
            curCnt = nextCnt;
        }

        if (curValid) {
            if (curIn0) {
                DataCopy(yGm_[curDst], buf0, static_cast<uint32_t>(curCnt));
            } else {
                DataCopy(yGm_[curDst], buf1, static_cast<uint32_t>(curCnt));
            }
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> workBuf_;
    uint64_t inputBatch_;
    uint64_t inputHeight_;
    uint64_t inputWidth_;
    uint64_t inputDepth_;
    uint64_t outputBatch_;
    uint64_t outputHeight_;
    uint64_t outputWidth_;
    uint64_t outputDepth_;
    uint64_t outputTotal_;
    uint64_t cropTop_;
    uint64_t cropBottom_;
    uint64_t cropLeft_;
    uint64_t cropRight_;
    uint32_t workElems_;
};


template <class DT_X, uint32_t MAX_OUTW>
class KernelBatchToSpaceT7Bs2Fp16NarrowRowContract {
public:
    __aicore__ inline KernelBatchToSpaceT7Bs2Fp16NarrowRowContract() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling) {
        inputBatch_ = tiling.N;
        inputHeight_ = tiling.H;
        inputWidth_ = tiling.W;
        inputDepth_ = tiling.C;
        outputBatch_ = tiling.outN;
        outputHeight_ = tiling.outH;
        outputWidth_ = tiling.outW;
        outputDepth_ = tiling.C;
        outputTotal_ = tiling.totalElements;
        blockSize_ = static_cast<uint64_t>(tiling.blockSize);
        cropTop_ = static_cast<uint64_t>(tiling.cropTop);
        cropBottom_ = static_cast<uint64_t>(tiling.cropBottom);
        cropLeft_ = static_cast<uint64_t>(tiling.cropLeft);
        cropRight_ = static_cast<uint64_t>(tiling.cropRight);
        workElems_ = tiling.fastWorkElems;
        if (workElems_ == 0U) workElems_ = 1U;
        const uint64_t inputTotal = inputBatch_ * inputHeight_ * inputWidth_ * inputDepth_;
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, inputTotal);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, outputTotal_);
        pipe_.InitBuffer(workBuf_, static_cast<uint64_t>(workElems_) * sizeof(DT_X));
    }

    __aicore__ inline bool CanRun() const {
        return CanRunFailCode() == 0U;
    }













    __aicore__ inline uint32_t CanRunFailCode() const {
        if (outputTotal_ == 0U || inputDepth_ == 0U || outputDepth_ == 0U ||
            outputWidth_ == 0U || outputHeight_ == 0U || outputBatch_ == 0U) {
            return 1U;
        }
        if (blockSize_ != 2U || inputDepth_ != outputDepth_) return 2U;
        if (inputBatch_ != outputBatch_ * 4U) return 2U;
        if (outputHeight_ + cropTop_ + cropBottom_ != inputHeight_ * 2U) return 2U;
        if (outputWidth_ + cropLeft_ + cropRight_ != inputWidth_ * 2U) return 2U;
        if (outputWidth_ > static_cast<uint64_t>(MAX_OUTW)) return 3U;
        const uint64_t depthBytes = outputDepth_ * static_cast<uint64_t>(sizeof(DT_X));
        const uint64_t rowBytes = outputWidth_ * depthBytes;
        const uint64_t inputRowBytes = inputWidth_ * depthBytes;
        if (depthBytes == 0U || rowBytes == 0U || inputRowBytes < depthBytes) return 4U;
        if ((depthBytes & static_cast<uint64_t>(kAlignBytes - 1U)) != 0U) return 4U;
        if ((rowBytes & static_cast<uint64_t>(kAlignBytes - 1U)) != 0U) return 4U;


        if (depthBytes > 0xFFFFFFFFULL) return 5U;
        const uint64_t srcGap = inputRowBytes - depthBytes;
        const uint64_t dstGap = 2U * rowBytes - depthBytes;
        if (srcGap > 0xFFFFFFFFULL) return 6U;
        if ((dstGap & static_cast<uint64_t>(kAlignBytes - 1U)) != 0U) return 8U;
        const uint64_t dstStride32 = dstGap / static_cast<uint64_t>(kAlignBytes);
        if (dstStride32 > 0xFFFFFFFFULL) return 7U;
        const uint64_t rowElems = outputWidth_ * outputDepth_;
        if (rowElems == 0U || rowElems > static_cast<uint64_t>(workElems_)) return 9U;
        const uint64_t rowsPerTile = GetRowsPerTile();
        if (rowsPerTile == 0U) return 10U;
        return 0U;
    }

    __aicore__ inline void Process() {
        if (!CanRun()) return;
        const uint64_t rowsPerTile = GetRowsPerTile();
        const uint64_t rowGroupsPerBatch = (outputHeight_ + rowsPerTile - 1U) / rowsPerTile;
        const uint64_t totalJobs = outputBatch_ * rowGroupsPerBatch;
        if (totalJobs == 0U) return;
        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
        const uint64_t jobStart = totalJobs * blockIdx / blockNum;
        const uint64_t jobEnd = totalJobs * (blockIdx + 1U) / blockNum;
        LocalTensor<DT_X> local = workBuf_.Get<DT_X>();
        for (uint64_t job = jobStart; job < jobEnd; ++job) {
            const uint64_t outB = job / rowGroupsPerBatch;
            const uint64_t group = job - outB * rowGroupsPerBatch;
            const uint64_t ohStart = group * rowsPerTile;
            if (ohStart >= outputHeight_) continue;
            uint64_t tileRows = outputHeight_ - ohStart;
            if (tileRows > rowsPerTile) tileRows = rowsPerTile;
            ProcessTile(outB, ohStart, tileRows, local);
        }
    }

private:
    __aicore__ inline uint64_t GetRowsPerTile() const {
        const uint64_t rowElems = outputWidth_ * outputDepth_;
        if (rowElems == 0U) return 0U;
        uint64_t rows = static_cast<uint64_t>(workElems_) / rowElems;
        if (rows == 0U) rows = 1U;
        if (rows > 1024U) rows = 1024U;
        return rows;
    }

    __aicore__ inline uint64_t FirstLocalRowForLaneH(uint64_t ohStart, uint64_t tileRows, uint64_t laneH) const {
        const uint64_t firstFullH = ohStart + cropTop_;
        const uint64_t curLaneH = firstFullH & 1U;
        uint64_t first = 0U;
        if (curLaneH != laneH) first = 1U;
        if (first >= tileRows) return tileRows;
        return first;
    }

    __aicore__ inline void CopyRowsForLaneOw(uint64_t outB, uint64_t ohStart, uint64_t tileRows,
                                             uint64_t laneH, uint64_t ow, LocalTensor<DT_X> &local) {
        const uint64_t firstLocalRow = FirstLocalRowForLaneH(ohStart, tileRows, laneH);
        if (firstLocalRow >= tileRows) return;
        const uint64_t blockCount = ((tileRows - 1U - firstLocalRow) / 2U) + 1U;
        if (blockCount == 0U || blockCount > 65535U) return;

        const uint64_t fullH0 = ohStart + firstLocalRow + cropTop_;
        const uint64_t inH0 = fullH0 >> 1U;
        const uint64_t fullW = ow + cropLeft_;
        const uint64_t laneW = fullW & 1U;
        const uint64_t inW = fullW >> 1U;
        const uint64_t lane = (laneH << 1U) + laneW;
        const uint64_t inB = lane * outputBatch_ + outB;
        const uint64_t srcOffset = ((inB * inputHeight_ + inH0) * inputWidth_ + inW) * inputDepth_;

        const uint64_t depthBytes = outputDepth_ * static_cast<uint64_t>(sizeof(DT_X));
        const uint64_t rowBytes = outputWidth_ * depthBytes;
        const uint64_t inputRowBytes = inputWidth_ * depthBytes;
        const uint64_t srcGap = inputRowBytes - depthBytes;
        const uint64_t dstGap = 2U * rowBytes - depthBytes;
        const uint64_t localOffset = firstLocalRow * outputWidth_ * outputDepth_ + ow * outputDepth_;

        DataCopyExtParams copyParams;
        if (depthBytes > 0xFFFFFFFFULL || srcGap > 0xFFFFFFFFULL) return;
        if ((dstGap & static_cast<uint64_t>(kAlignBytes - 1U)) != 0U) return;
        const uint64_t dstStride32 = dstGap / static_cast<uint64_t>(kAlignBytes);
        if (dstStride32 > 0xFFFFFFFFULL) return;
        copyParams.blockCount = static_cast<uint16_t>(blockCount);
        copyParams.blockLen = static_cast<uint32_t>(depthBytes);
        copyParams.srcStride = static_cast<uint32_t>(srcGap);

        copyParams.dstStride = static_cast<uint32_t>(dstStride32);
        DataCopyPadExtParams<DT_X> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0U;
        padParams.rightPadding = 0U;
        padParams.paddingValue = static_cast<DT_X>(0);
        DataCopyPad(local[localOffset], xGm_[srcOffset], copyParams, padParams);
    }

    __aicore__ inline void ProcessTile(uint64_t outB, uint64_t ohStart, uint64_t tileRows,
                                       LocalTensor<DT_X> &local) {
        if (tileRows == 0U) return;
        for (uint64_t laneH = 0U; laneH < 2U; ++laneH) {
            for (uint64_t ow = 0U; ow < outputWidth_; ++ow) {
                CopyRowsForLaneOw(outB, ohStart, tileRows, laneH, ow, local);
            }
        }
        PipeBarrier<PIPE_ALL>();
        const uint64_t rowElems = outputWidth_ * outputDepth_;
        const uint64_t dstOffset = (outB * outputHeight_ + ohStart) * rowElems;
        const uint64_t elemCount = tileRows * rowElems;
        DataCopy(yGm_[dstOffset], local, static_cast<uint32_t>(elemCount));
        PipeBarrier<PIPE_ALL>();
    }

private:
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> workBuf_;
    uint64_t inputBatch_;
    uint64_t inputHeight_;
    uint64_t inputWidth_;
    uint64_t inputDepth_;
    uint64_t outputBatch_;
    uint64_t outputHeight_;
    uint64_t outputWidth_;
    uint64_t outputDepth_;
    uint64_t outputTotal_;
    uint64_t blockSize_;
    uint64_t cropTop_;
    uint64_t cropBottom_;
    uint64_t cropLeft_;
    uint64_t cropRight_;
    uint32_t workElems_;
};


template <class DT_X, uint64_t TARGET_BLOCK_SIZE>
class KernelBatchToSpaceHighDepthContigPack {
public:
    __aicore__ inline KernelBatchToSpaceHighDepthContigPack() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling) {
        inputBatch_ = tiling.N;
        inputHeight_ = tiling.H;
        inputWidth_ = tiling.W;
        inputDepth_ = tiling.C;
        outputBatch_ = tiling.outN;
        outputHeight_ = tiling.outH;
        outputWidth_ = tiling.outW;
        outputDepth_ = tiling.C;
        outputTotal_ = tiling.totalElements;
        blockSize_ = static_cast<uint64_t>(tiling.blockSize);
        cropTop_ = static_cast<uint64_t>(tiling.cropTop);
        cropBottom_ = static_cast<uint64_t>(tiling.cropBottom);
        cropLeft_ = static_cast<uint64_t>(tiling.cropLeft);
        cropRight_ = static_cast<uint64_t>(tiling.cropRight);
        workElems_ = tiling.fastWorkElems;
        if (workElems_ == 0U) {
            workElems_ = 1U;
        }

        const uint64_t inputTotal = inputBatch_ * inputHeight_ * inputWidth_ * inputDepth_;
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, inputTotal);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, outputTotal_);
        pipe_.InitBuffer(workBuf_, workElems_ * sizeof(DT_X));
    }

    __aicore__ inline bool CanRun() const {
        if (outputTotal_ == 0U || outputDepth_ == 0U || outputWidth_ == 0U || outputHeight_ == 0U ||
            outputBatch_ == 0U || inputDepth_ != outputDepth_) {
            return false;
        }
        if (blockSize_ != TARGET_BLOCK_SIZE) {
            return false;
        }
        if (inputBatch_ != outputBatch_ * TARGET_BLOCK_SIZE * TARGET_BLOCK_SIZE) {
            return false;
        }
        if (outputHeight_ + cropTop_ + cropBottom_ != inputHeight_ * TARGET_BLOCK_SIZE) {
            return false;
        }
        if (outputWidth_ + cropLeft_ + cropRight_ != inputWidth_ * TARGET_BLOCK_SIZE) {
            return false;
        }
        const uint64_t depthBytes = outputDepth_ * static_cast<uint64_t>(sizeof(DT_X));
        if (depthBytes == 0U || (depthBytes & static_cast<uint64_t>(kAlignBytes - 1U)) != 0U ||
            depthBytes > 65535U) {
            return false;
        }
        if (outputTotal_ * static_cast<uint64_t>(sizeof(DT_X)) < kHighDepthPackMinOutputBytes) {
            return false;
        }
        if (outputWidth_ < 8U) {
            return false;
        }
        const uint64_t tileCols = GetTileCols();
        return tileCols >= 2U && tileCols < kScatterDiagMinTileCols;
    }

    __aicore__ inline void Process() {
        const uint64_t tileCols = GetTileCols();
        if (tileCols < 2U || tileCols >= kScatterDiagMinTileCols) {
            return;
        }
        const uint64_t tilesPerRow = (outputWidth_ + tileCols - 1U) / tileCols;
        const uint64_t totalRows = outputBatch_ * outputHeight_;
        const uint64_t totalJobs = totalRows * tilesPerRow;
        if (totalJobs == 0U) {
            return;
        }

        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
        const uint64_t jobStart = totalJobs * blockIdx / blockNum;
        const uint64_t jobEnd = totalJobs * (blockIdx + 1U) / blockNum;

        LocalTensor<DT_X> local = workBuf_.Get<DT_X>();
        for (uint64_t job = jobStart; job < jobEnd; ++job) {
            const uint64_t tileId = job % tilesPerRow;
            const uint64_t row = job / tilesPerRow;
            ProcessTile(row, tileId, tileCols, local);
        }
    }

private:
    __aicore__ inline uint64_t AlignUp32(uint64_t bytes) const {
        return (bytes + static_cast<uint64_t>(kAlignBytes - 1U)) & ~static_cast<uint64_t>(kAlignBytes - 1U);
    }

    __aicore__ inline uint64_t GetTileCols() const {
        const uint64_t depthBytes = outputDepth_ * static_cast<uint64_t>(sizeof(DT_X));
        if (depthBytes == 0U) {
            return 0U;
        }
        const uint64_t paddedDepthBytes = AlignUp32(depthBytes);
        const uint64_t paddedDepthElems = paddedDepthBytes / static_cast<uint64_t>(sizeof(DT_X));
        if (paddedDepthElems == 0U) {
            return 0U;
        }
        uint64_t tileCols = static_cast<uint64_t>(workElems_) / paddedDepthElems;
        if (tileCols > 15U) {
            tileCols = 15U;
        }
        return tileCols;
    }

    __aicore__ inline void CopyLaneToLocal(uint64_t srcOffset, uint64_t localOffset, uint64_t blockCount,
                                           LocalTensor<DT_X> &local) {
        if (blockCount == 0U) {
            return;
        }
        const uint64_t depthBytes = outputDepth_ * static_cast<uint64_t>(sizeof(DT_X));
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(blockCount);
        copyParams.blockLen = static_cast<uint32_t>(depthBytes);
        copyParams.srcStride = 0U;


        copyParams.dstStride = static_cast<uint32_t>(((TARGET_BLOCK_SIZE - 1U) * depthBytes) /
                                                    static_cast<uint64_t>(kAlignBytes));
        DataCopyPadExtParams<DT_X> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0U;
        padParams.rightPadding = 0U;
        padParams.paddingValue = static_cast<DT_X>(0);
        DataCopyPad(local[localOffset], xGm_[srcOffset], copyParams, padParams);
    }

    __aicore__ inline void CopyOutTile(uint64_t dstOffset, uint64_t count, LocalTensor<DT_X> &local) {
        if (count == 0U) {
            return;
        }
        DataCopy(yGm_[dstOffset], local, static_cast<uint32_t>(count));
    }

    __aicore__ inline void ProcessTile(uint64_t row, uint64_t tileId, uint64_t tileCols,
                                       LocalTensor<DT_X> &local) {
        const uint64_t outB = row / outputHeight_;
        const uint64_t outH = row - outB * outputHeight_;
        const uint64_t fullH = outH + cropTop_;
        const uint64_t laneH = fullH % TARGET_BLOCK_SIZE;
        const uint64_t inH = fullH / TARGET_BLOCK_SIZE;

        const uint64_t startOw = tileId * tileCols;
        if (startOw >= outputWidth_) {
            return;
        }
        uint64_t curCols = outputWidth_ - startOw;
        if (curCols > tileCols) {
            curCols = tileCols;
        }
        if (curCols == 0U) {
            return;
        }
        const uint64_t depthBytes = outputDepth_ * static_cast<uint64_t>(sizeof(DT_X));
        const uint64_t tileBytes = curCols * depthBytes;
        if (tileBytes > static_cast<uint64_t>(workElems_) * static_cast<uint64_t>(sizeof(DT_X))) {
            return;
        }
        const uint64_t endOw = startOw + curCols;

        for (uint64_t laneW = 0U; laneW < TARGET_BLOCK_SIZE; ++laneW) {
            const uint64_t startMod = (startOw + cropLeft_) % TARGET_BLOCK_SIZE;
            const uint64_t delta = (laneW + TARGET_BLOCK_SIZE - startMod) % TARGET_BLOCK_SIZE;
            const uint64_t firstOw = startOw + delta;
            if (firstOw >= endOw) {
                continue;
            }
            const uint64_t laneCols = ((endOw - 1U - firstOw) / TARGET_BLOCK_SIZE) + 1U;
            const uint64_t inW0 = (firstOw + cropLeft_) / TARGET_BLOCK_SIZE;
            const uint64_t lane = laneH * TARGET_BLOCK_SIZE + laneW;
            const uint64_t inB = lane * outputBatch_ + outB;
            const uint64_t srcOffset = ((inB * inputHeight_ + inH) * inputWidth_ + inW0) * inputDepth_;
            const uint64_t localOffset = (firstOw - startOw) * outputDepth_;
            CopyLaneToLocal(srcOffset, localOffset, laneCols, local);
        }

        PipeBarrier<PIPE_ALL>();
        const uint64_t dstOffset = ((outB * outputHeight_ + outH) * outputWidth_ + startOw) * outputDepth_;
        CopyOutTile(dstOffset, curCols * outputDepth_, local);
        PipeBarrier<PIPE_ALL>();
    }

private:
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> workBuf_;
    uint64_t inputBatch_;
    uint64_t inputHeight_;
    uint64_t inputWidth_;
    uint64_t inputDepth_;
    uint64_t outputBatch_;
    uint64_t outputHeight_;
    uint64_t outputWidth_;
    uint64_t outputDepth_;
    uint64_t outputTotal_;
    uint64_t blockSize_;
    uint64_t cropTop_;
    uint64_t cropBottom_;
    uint64_t cropLeft_;
    uint64_t cropRight_;
    uint32_t workElems_;
};




template <class DT_X, uint32_t ROW_GROUP>

class KernelBatchToSpaceBs2RowGroupPack {
public:
    __aicore__ inline KernelBatchToSpaceBs2RowGroupPack() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling) {
        inputBatch_ = tiling.N;
        inputHeight_ = tiling.H;
        inputWidth_ = tiling.W;
        inputDepth_ = tiling.C;
        outputBatch_ = tiling.outN;
        outputHeight_ = tiling.outH;
        outputWidth_ = tiling.outW;
        outputDepth_ = tiling.C;
        outputTotal_ = tiling.totalElements;
        blockSize_ = static_cast<uint64_t>(tiling.blockSize);
        cropTop_ = static_cast<uint64_t>(tiling.cropTop);
        cropBottom_ = static_cast<uint64_t>(tiling.cropBottom);
        cropLeft_ = static_cast<uint64_t>(tiling.cropLeft);
        cropRight_ = static_cast<uint64_t>(tiling.cropRight);
        workElems_ = tiling.fastWorkElems;
        if (workElems_ == 0U) workElems_ = 1U;
        const uint64_t inputTotal = inputBatch_ * inputHeight_ * inputWidth_ * inputDepth_;
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, inputTotal);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, outputTotal_);
        pipe_.InitBuffer(workBuf_, workElems_ * sizeof(DT_X));
    }

    __aicore__ inline bool CanRun() const {
        if (outputTotal_ == 0U || outputDepth_ == 0U || outputWidth_ == 0U || outputHeight_ == 0U || outputBatch_ == 0U) return false;
        if (blockSize_ != 2U || inputDepth_ != outputDepth_) return false;
        if (inputBatch_ != outputBatch_ * 4U) return false;
        if (outputHeight_ + cropTop_ + cropBottom_ != inputHeight_ * 2U) return false;
        if (outputWidth_ + cropLeft_ + cropRight_ != inputWidth_ * 2U) return false;
        const uint64_t depthBytes = outputDepth_ * static_cast<uint64_t>(sizeof(DT_X));
        if (depthBytes == 0U || (depthBytes & static_cast<uint64_t>(kAlignBytes - 1U)) != 0U || depthBytes > 65535U) return false;
        const uint64_t rowElems = outputWidth_ * outputDepth_;
        if (rowElems == 0U) return false;
        const uint64_t elemsPerBlock = static_cast<uint64_t>(kAlignBytes / sizeof(DT_X));
        if ((rowElems & (elemsPerBlock - 1U)) != 0U) return false;
        if (rowElems * static_cast<uint64_t>(ROW_GROUP) > static_cast<uint64_t>(workElems_)) return false;
        return true;
    }

    __aicore__ inline void Process() {
        if (!CanRun()) return;
        const uint64_t rowGroupsPerBatch = (outputHeight_ + static_cast<uint64_t>(ROW_GROUP) - 1U) / static_cast<uint64_t>(ROW_GROUP);
        const uint64_t totalJobs = outputBatch_ * rowGroupsPerBatch;
        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
        const uint64_t jobStart = totalJobs * blockIdx / blockNum;
        const uint64_t jobEnd = totalJobs * (blockIdx + 1U) / blockNum;
        LocalTensor<DT_X> local = workBuf_.Get<DT_X>();
        const uint64_t rowElems = outputWidth_ * outputDepth_;
        for (uint64_t job = jobStart; job < jobEnd; ++job) {
            const uint64_t outB = job / rowGroupsPerBatch;
            const uint64_t group = job - outB * rowGroupsPerBatch;
            const uint64_t startH = group * static_cast<uint64_t>(ROW_GROUP);
            if (startH >= outputHeight_) continue;
            uint64_t rows = outputHeight_ - startH;
            if (rows > static_cast<uint64_t>(ROW_GROUP)) rows = static_cast<uint64_t>(ROW_GROUP);
            for (uint64_t r = 0U; r < rows; ++r) {
                PackFullRow(outB, startH + r, r * rowElems, local);
            }
            PipeBarrier<PIPE_ALL>();
            const uint64_t dstOffset = ((outB * outputHeight_ + startH) * outputWidth_) * outputDepth_;
            DataCopy(yGm_[dstOffset], local, static_cast<uint32_t>(rows * rowElems));
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    __aicore__ inline void PackFullRow(uint64_t outB, uint64_t outH, uint64_t localBase, LocalTensor<DT_X> &local) {
        const uint64_t fullH = outH + cropTop_;
        const uint64_t laneH = fullH & 1U;
        const uint64_t inH = fullH >> 1U;
        const uint64_t cropMod = cropLeft_ & 1U;
        const uint64_t depthBytes = outputDepth_ * static_cast<uint64_t>(sizeof(DT_X));
        for (uint64_t laneW = 0U; laneW < 2U; ++laneW) {
            const uint64_t firstOw = (laneW + 2U - cropMod) & 1U;
            if (firstOw >= outputWidth_) continue;
            const uint64_t laneCols = ((outputWidth_ - 1U - firstOw) >> 1U) + 1U;
            const uint64_t inW0 = (firstOw + cropLeft_) >> 1U;
            const uint64_t lane = (laneH << 1U) + laneW;
            const uint64_t inB = lane * outputBatch_ + outB;
            const uint64_t srcOffset = ((inB * inputHeight_ + inH) * inputWidth_ + inW0) * inputDepth_;
            const uint64_t localOffset = localBase + firstOw * outputDepth_;
            DataCopyExtParams copyParams;
            copyParams.blockCount = static_cast<uint16_t>(laneCols);
            copyParams.blockLen = static_cast<uint32_t>(depthBytes);
            copyParams.srcStride = 0U;
            copyParams.dstStride = static_cast<uint32_t>(depthBytes / static_cast<uint64_t>(kAlignBytes));
            DataCopyPadExtParams<DT_X> padParams;
            padParams.isPad = false;
            padParams.leftPadding = 0U;
            padParams.rightPadding = 0U;
            padParams.paddingValue = static_cast<DT_X>(0);
            DataCopyPad(local[localOffset], xGm_[srcOffset], copyParams, padParams);
        }
    }

private:
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> workBuf_;
    uint64_t inputBatch_, inputHeight_, inputWidth_, inputDepth_;
    uint64_t outputBatch_, outputHeight_, outputWidth_, outputDepth_, outputTotal_;
    uint64_t blockSize_, cropTop_, cropBottom_, cropLeft_, cropRight_;
    uint32_t workElems_;
};



template <class DT_X>
class KernelBatchToSpaceBlock1VerticalCropContig {
public:
    __aicore__ inline KernelBatchToSpaceBlock1VerticalCropContig() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling) {
        inputBatch_ = tiling.N;
        inputHeight_ = tiling.H;
        inputWidth_ = tiling.W;
        inputDepth_ = tiling.C;
        outputBatch_ = tiling.outN;
        outputHeight_ = tiling.outH;
        outputWidth_ = tiling.outW;
        outputDepth_ = tiling.C;
        outputTotal_ = tiling.totalElements;
        blockSize_ = static_cast<uint64_t>(tiling.blockSize);
        cropTop_ = static_cast<uint64_t>(tiling.cropTop);
        cropBottom_ = static_cast<uint64_t>(tiling.cropBottom);
        cropLeft_ = static_cast<uint64_t>(tiling.cropLeft);
        cropRight_ = static_cast<uint64_t>(tiling.cropRight);
        workElems_ = tiling.fastWorkElems;
        if (workElems_ == 0U) workElems_ = 1U;
        const uint64_t inputTotal = inputBatch_ * inputHeight_ * inputWidth_ * inputDepth_;
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, inputTotal);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, outputTotal_);
        pipe_.InitBuffer(workBuf_, static_cast<uint64_t>(workElems_) * sizeof(DT_X));
    }

    __aicore__ inline bool CanRun() const {
        if (blockSize_ != 1U || outputTotal_ == 0U || outputDepth_ == 0U ||
            outputWidth_ == 0U || outputHeight_ == 0U || outputBatch_ == 0U) {
            return false;
        }
        if (inputBatch_ != outputBatch_) return false;
        if (cropLeft_ != 0U || cropRight_ != 0U) return false;
        if ((cropTop_ | cropBottom_) == 0U) return false;
        if (outputHeight_ + cropTop_ + cropBottom_ != inputHeight_) return false;
        if (outputWidth_ != inputWidth_) return false;

        const uint64_t elemBlock = static_cast<uint64_t>(kAlignBytes / sizeof(DT_X));
        const uint64_t batchElems = outputHeight_ * outputWidth_ * outputDepth_;
        const uint64_t inputBatchElems = inputHeight_ * inputWidth_ * inputDepth_;
        const uint64_t cropTopElems = cropTop_ * inputWidth_ * inputDepth_;
        if (batchElems == 0U || workElems_ == 0U) return false;

        if ((batchElems % elemBlock) != 0U) return false;
        if ((inputBatchElems % elemBlock) != 0U) return false;
        if ((cropTopElems % elemBlock) != 0U) return false;
        return true;
    }

    __aicore__ inline void Process() {
        if (!CanRun()) return;
        const uint64_t batchElems = outputHeight_ * outputWidth_ * outputDepth_;
        const uint64_t inputBatchElems = inputHeight_ * inputWidth_ * inputDepth_;
        const uint64_t cropTopElems = cropTop_ * inputWidth_ * inputDepth_;
        const uint64_t workElems = static_cast<uint64_t>(workElems_);
        const uint64_t segsPerBatch = (batchElems + workElems - 1U) / workElems;
        const uint64_t totalJobs = outputBatch_ * segsPerBatch;
        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
        const uint64_t jobStart = totalJobs * blockIdx / blockNum;
        const uint64_t jobEnd = totalJobs * (blockIdx + 1U) / blockNum;
        LocalTensor<DT_X> local = workBuf_.Get<DT_X>();

        for (uint64_t job = jobStart; job < jobEnd; ++job) {
            const uint64_t outB = job / segsPerBatch;
            const uint64_t seg = job - outB * segsPerBatch;
            const uint64_t elemOffset = seg * workElems;
            if (elemOffset >= batchElems) continue;
            uint64_t cur = batchElems - elemOffset;
            if (cur > workElems) cur = workElems;

            const uint64_t srcOffset = outB * inputBatchElems + cropTopElems + elemOffset;
            const uint64_t dstOffset = outB * batchElems + elemOffset;
            DataCopy(local, xGm_[srcOffset], static_cast<uint32_t>(cur));
            PipeBarrier<PIPE_ALL>();
            DataCopy(yGm_[dstOffset], local, static_cast<uint32_t>(cur));
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> workBuf_;
    uint64_t inputBatch_, inputHeight_, inputWidth_, inputDepth_;
    uint64_t outputBatch_, outputHeight_, outputWidth_, outputDepth_, outputTotal_;
    uint64_t blockSize_, cropTop_, cropBottom_, cropLeft_, cropRight_;
    uint32_t workElems_;
};



template <class DT_X>
class KernelBatchToSpaceBlock1CropRowContract {
public:
    __aicore__ inline KernelBatchToSpaceBlock1CropRowContract() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling) {
        inputBatch_ = tiling.N;
        inputHeight_ = tiling.H;
        inputWidth_ = tiling.W;
        inputDepth_ = tiling.C;
        outputBatch_ = tiling.outN;
        outputHeight_ = tiling.outH;
        outputWidth_ = tiling.outW;
        outputDepth_ = tiling.C;
        outputTotal_ = tiling.totalElements;
        blockSize_ = static_cast<uint64_t>(tiling.blockSize);
        cropTop_ = static_cast<uint64_t>(tiling.cropTop);
        cropBottom_ = static_cast<uint64_t>(tiling.cropBottom);
        cropLeft_ = static_cast<uint64_t>(tiling.cropLeft);
        cropRight_ = static_cast<uint64_t>(tiling.cropRight);
        workElems_ = tiling.fastWorkElems;
        if (workElems_ == 0U) workElems_ = 1U;
        const uint64_t inputTotal = inputBatch_ * inputHeight_ * inputWidth_ * inputDepth_;
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, inputTotal);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, outputTotal_);
        pipe_.InitBuffer(workBuf_, static_cast<uint64_t>(workElems_) * sizeof(DT_X));
    }

    __aicore__ inline bool CanRun() const {
        if (blockSize_ != 1U || outputTotal_ == 0U || outputDepth_ == 0U ||
            outputWidth_ == 0U || outputHeight_ == 0U || outputBatch_ == 0U) {
            return false;
        }
        if (inputBatch_ != outputBatch_) return false;
        if (outputHeight_ + cropTop_ + cropBottom_ != inputHeight_) return false;
        if (outputWidth_ + cropLeft_ + cropRight_ != inputWidth_) return false;
        if ((cropTop_ | cropBottom_ | cropLeft_ | cropRight_) == 0U) return false;

        const uint64_t rowElems = outputWidth_ * outputDepth_;
        const uint64_t inputRowElems = inputWidth_ * inputDepth_;
        if (rowElems == 0U || rowElems > static_cast<uint64_t>(workElems_)) return false;
        const uint64_t rowBytes = rowElems * static_cast<uint64_t>(sizeof(DT_X));
        const uint64_t inputRowBytes = inputRowElems * static_cast<uint64_t>(sizeof(DT_X));
        const uint64_t cropLeftBytes = cropLeft_ * outputDepth_ * static_cast<uint64_t>(sizeof(DT_X));
        if (rowBytes == 0U || rowBytes > 65535U || inputRowBytes < rowBytes) return false;

        if ((rowBytes & static_cast<uint64_t>(kAlignBytes - 1U)) != 0U) return false;
        if ((inputRowBytes & static_cast<uint64_t>(kAlignBytes - 1U)) != 0U) return false;
        if ((cropLeftBytes & static_cast<uint64_t>(kAlignBytes - 1U)) != 0U) return false;
        if ((inputRowBytes - rowBytes) > 65535U) return false;
        return true;
    }

    __aicore__ inline void Process() {
        if (!CanRun()) return;
        const uint64_t rowElems = outputWidth_ * outputDepth_;
        const uint64_t inputRowElems = inputWidth_ * inputDepth_;
        uint64_t rowsPerGroup = static_cast<uint64_t>(workElems_) / rowElems;
        if (rowsPerGroup == 0U) rowsPerGroup = 1U;
        if (rowsPerGroup > 16U) rowsPerGroup = 16U;

        const uint64_t rowGroupsPerBatch = (outputHeight_ + rowsPerGroup - 1U) / rowsPerGroup;
        const uint64_t totalJobs = outputBatch_ * rowGroupsPerBatch;
        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
        const uint64_t jobStart = totalJobs * blockIdx / blockNum;
        const uint64_t jobEnd = totalJobs * (blockIdx + 1U) / blockNum;
        LocalTensor<DT_X> local = workBuf_.Get<DT_X>();

        for (uint64_t job = jobStart; job < jobEnd; ++job) {
            const uint64_t outB = job / rowGroupsPerBatch;
            const uint64_t group = job - outB * rowGroupsPerBatch;
            const uint64_t startH = group * rowsPerGroup;
            if (startH >= outputHeight_) continue;
            uint64_t rows = outputHeight_ - startH;
            if (rows > rowsPerGroup) rows = rowsPerGroup;

            const uint64_t srcOffset = ((outB * inputHeight_ + (startH + cropTop_)) * inputWidth_ + cropLeft_) * inputDepth_;
            const uint64_t dstOffset = ((outB * outputHeight_ + startH) * outputWidth_) * outputDepth_;
            const uint32_t totalCopyElems = static_cast<uint32_t>(rows * rowElems);

            if (inputRowElems == rowElems) {
                DataCopy(local, xGm_[srcOffset], totalCopyElems);
                PipeBarrier<PIPE_ALL>();
                DataCopy(yGm_[dstOffset], local, totalCopyElems);
                PipeBarrier<PIPE_ALL>();
            } else {
                CopyInRows2D(srcOffset, rows, local);
                PipeBarrier<PIPE_ALL>();
                DataCopy(yGm_[dstOffset], local, totalCopyElems);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

private:
    __aicore__ inline void CopyInRows2D(uint64_t srcOffset, uint64_t rows, LocalTensor<DT_X> &local) {
        const uint64_t rowBytes = outputWidth_ * outputDepth_ * static_cast<uint64_t>(sizeof(DT_X));
        const uint64_t inputRowBytes = inputWidth_ * inputDepth_ * static_cast<uint64_t>(sizeof(DT_X));
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(rows);
        copyParams.blockLen = static_cast<uint32_t>(rowBytes);
        copyParams.srcStride = static_cast<uint32_t>(inputRowBytes - rowBytes);
        copyParams.dstStride = 0U;
        DataCopyPadExtParams<DT_X> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0U;
        padParams.rightPadding = 0U;
        padParams.paddingValue = static_cast<DT_X>(0);
        DataCopyPad(local, xGm_[srcOffset], copyParams, padParams);
    }

private:
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> workBuf_;
    uint64_t inputBatch_, inputHeight_, inputWidth_, inputDepth_;
    uint64_t outputBatch_, outputHeight_, outputWidth_, outputDepth_, outputTotal_;
    uint64_t blockSize_, cropTop_, cropBottom_, cropLeft_, cropRight_;
    uint32_t workElems_;
};


template <class DT_X, uint64_t TARGET_BLOCK_SIZE, uint64_t BLOCK_SHIFT>
class KernelBatchToSpaceSmallCRowOwner {
public:
    __aicore__ inline KernelBatchToSpaceSmallCRowOwner() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling) {
        inputBatch_ = tiling.N;
        inputHeight_ = tiling.H;
        inputWidth_ = tiling.W;
        inputDepth_ = tiling.C;
        outputBatch_ = tiling.outN;
        outputHeight_ = tiling.outH;
        outputWidth_ = tiling.outW;
        outputDepth_ = tiling.C;
        outputTotal_ = tiling.totalElements;
        blockSize_ = static_cast<uint64_t>(tiling.blockSize);
        cropTop_ = static_cast<uint64_t>(tiling.cropTop);
        cropBottom_ = static_cast<uint64_t>(tiling.cropBottom);
        cropLeft_ = static_cast<uint64_t>(tiling.cropLeft);
        cropRight_ = static_cast<uint64_t>(tiling.cropRight);
        workElems_ = tiling.fastWorkElems;
        if (workElems_ == 0U) workElems_ = 1U;
        const uint64_t inputTotal = inputBatch_ * inputHeight_ * inputWidth_ * inputDepth_;
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, inputTotal);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, outputTotal_);
        pipe_.InitBuffer(workBuf_, static_cast<uint64_t>(workElems_) * sizeof(DT_X));
    }

    __aicore__ inline bool CanRun() const {
        if (blockSize_ != TARGET_BLOCK_SIZE || outputTotal_ == 0U || outputDepth_ == 0U ||
            outputWidth_ == 0U || outputHeight_ == 0U || outputBatch_ == 0U) return false;
        if (TARGET_BLOCK_SIZE != (1ULL << BLOCK_SHIFT)) return false;
        if (inputDepth_ != outputDepth_) return false;
        if (inputBatch_ != outputBatch_ * TARGET_BLOCK_SIZE * TARGET_BLOCK_SIZE) return false;
        if (outputHeight_ + cropTop_ + cropBottom_ != inputHeight_ * TARGET_BLOCK_SIZE) return false;
        if (outputWidth_ + cropLeft_ + cropRight_ != inputWidth_ * TARGET_BLOCK_SIZE) return false;

        const uint64_t depthBytes = outputDepth_ * static_cast<uint64_t>(sizeof(DT_X));
        if (depthBytes == 0U || depthBytes > 64U || depthBytes > 65535U) return false;
        const uint64_t dstStrideBytes = (TARGET_BLOCK_SIZE - 1U) * depthBytes;
        if (dstStrideBytes > 65535U) return false;
        const uint64_t laneCols = (outputWidth_ + TARGET_BLOCK_SIZE - 1U) >> BLOCK_SHIFT;
        if (laneCols == 0U || laneCols > 4095U) return false;
        const uint64_t laneElems = laneCols * outputDepth_;
        if (laneElems == 0U || laneElems > static_cast<uint64_t>(workElems_)) return false;

        if (outputTotal_ * static_cast<uint64_t>(sizeof(DT_X)) > (16U * 1024U)) return false;
        return true;
    }

    __aicore__ inline void Process() {
        if (!CanRun()) return;
        const uint64_t totalRows = outputBatch_ * outputHeight_;
        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
        const uint64_t rowStart = totalRows * blockIdx / blockNum;
        const uint64_t rowEnd = totalRows * (blockIdx + 1U) / blockNum;
        LocalTensor<DT_X> local = workBuf_.Get<DT_X>();

        for (uint64_t row = rowStart; row < rowEnd; ++row) {
            ProcessRow(row, local);
        }
    }

private:
    __aicore__ inline void ProcessRow(uint64_t row, LocalTensor<DT_X> &local) {
        constexpr uint64_t kMask = TARGET_BLOCK_SIZE - 1U;
        const uint64_t outB = row / outputHeight_;
        const uint64_t outH = row - outB * outputHeight_;
        const uint64_t fullH = outH + cropTop_;
        const uint64_t laneH = fullH & kMask;
        const uint64_t inH = fullH >> BLOCK_SHIFT;
        const uint64_t cropMod = cropLeft_ & kMask;

        for (uint64_t laneW = 0U; laneW < TARGET_BLOCK_SIZE; ++laneW) {
            const uint64_t firstOw = (laneW + TARGET_BLOCK_SIZE - cropMod) & kMask;
            if (firstOw >= outputWidth_) continue;
            const uint64_t laneCols = ((outputWidth_ - 1U - firstOw) >> BLOCK_SHIFT) + 1U;
            if (laneCols == 0U || laneCols > 4095U) continue;
            const uint64_t inW0 = (firstOw + cropLeft_) >> BLOCK_SHIFT;
            const uint64_t lane = (laneH << BLOCK_SHIFT) + laneW;
            const uint64_t inB = lane * outputBatch_ + outB;
            const uint64_t srcOffset = ((inB * inputHeight_ + inH) * inputWidth_ + inW0) * inputDepth_;
            const uint64_t dstOffset = ((outB * outputHeight_ + outH) * outputWidth_ + firstOw) * outputDepth_;
            CopyInLane(srcOffset, laneCols, local);
            PipeBarrier<PIPE_ALL>();
            CopyOutLane(dstOffset, laneCols, local);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void CopyInLane(uint64_t srcOffset, uint64_t blockCount, LocalTensor<DT_X> &local) {
        const uint64_t depthBytes = outputDepth_ * static_cast<uint64_t>(sizeof(DT_X));
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(blockCount);
        copyParams.blockLen = static_cast<uint32_t>(depthBytes);
        copyParams.srcStride = 0U;
        copyParams.dstStride = 0U;
        DataCopyPadExtParams<DT_X> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0U;
        padParams.rightPadding = 0U;
        padParams.paddingValue = static_cast<DT_X>(0);
        DataCopyPad(local, xGm_[srcOffset], copyParams, padParams);
    }

    __aicore__ inline void CopyOutLane(uint64_t dstOffset, uint64_t blockCount, LocalTensor<DT_X> &local) {
        const uint64_t depthBytes = outputDepth_ * static_cast<uint64_t>(sizeof(DT_X));
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(blockCount);
        copyParams.blockLen = static_cast<uint32_t>(depthBytes);
        copyParams.srcStride = 0U;
        copyParams.dstStride = static_cast<uint32_t>((TARGET_BLOCK_SIZE - 1U) * depthBytes);
        DataCopyPad(yGm_[dstOffset], local, copyParams);
    }

private:
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> workBuf_;
    uint64_t inputBatch_, inputHeight_, inputWidth_, inputDepth_;
    uint64_t outputBatch_, outputHeight_, outputWidth_, outputDepth_, outputTotal_;
    uint64_t blockSize_, cropTop_, cropBottom_, cropLeft_, cropRight_;
    uint32_t workElems_;
};

template <class DT_X>
class KernelBatchToSpaceLightweightGeneric {
public:
    __aicore__ inline KernelBatchToSpaceLightweightGeneric() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, BatchToSpaceTilingData *tiling_data) {
        inputBatch_ = tiling_data->N;
        inputHeight_ = tiling_data->H;
        inputWidth_ = tiling_data->W;
        inputDepth_ = tiling_data->C;
        outputBatch_ = tiling_data->outN;
        outputHeight_ = tiling_data->outH;
        outputWidth_ = tiling_data->outW;
        outputDepth_ = tiling_data->C;
        outputTotal_ = tiling_data->totalElements;
        blockSize_ = tiling_data->blockSize;
        cropTop_ = tiling_data->cropTop;
        cropBottom_ = tiling_data->cropBottom;
        cropLeft_ = tiling_data->cropLeft;
        cropRight_ = tiling_data->cropRight;
        workElems_ = tiling_data->fastWorkElems;
        if (workElems_ == 0U) {
            workElems_ = 1U;
        }

        const uint64_t inputTotal = inputBatch_ * inputHeight_ * inputWidth_ * inputDepth_;
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, inputTotal);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, outputTotal_);
        pipe_.InitBuffer(workBuf_, workElems_ * sizeof(DT_X));
    }

    __aicore__ inline bool CanRun() const {
        if (outputTotal_ == 0U || outputDepth_ == 0U || outputWidth_ == 0U || outputHeight_ == 0U ||
            outputBatch_ == 0U || inputDepth_ != outputDepth_) {
            return false;
        }
        if (blockSize_ < 2U || blockSize_ > 64U) {
            return false;
        }
        if (inputBatch_ != outputBatch_ * blockSize_ * blockSize_) {
            return false;
        }
        if (outputHeight_ + cropTop_ + cropBottom_ != inputHeight_ * blockSize_) {
            return false;
        }
        if (outputWidth_ + cropLeft_ + cropRight_ != inputWidth_ * blockSize_) {
            return false;
        }
        return true;
    }

    __aicore__ inline void Process() {
        if (!CanRun()) {
            return;
        }

        const uint32_t elemsPerBlock = static_cast<uint32_t>(kAlignBytes / sizeof(DT_X));
        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
        uint64_t elementsPerCore = (outputTotal_ + blockNum - 1U) / blockNum;
        uint64_t rangeAlign = static_cast<uint64_t>(elemsPerBlock);
        const uint64_t rowElemsForAlign = outputWidth_ * outputDepth_;
        const uint64_t totalRowsForAlign = outputBatch_ * outputHeight_;
        if ((blockSize_ == 2U || blockSize_ == 4U) && outputDepth_ > 0U &&
            outputDepth_ <= static_cast<uint64_t>(workElems_) &&
            (outputDepth_ % static_cast<uint64_t>(elemsPerBlock)) == 0U) {
            if (rowElemsForAlign > 0U && rowElemsForAlign <= static_cast<uint64_t>(workElems_) &&
                totalRowsForAlign >= blockNum) {
                rangeAlign = rowElemsForAlign;
            } else {
                rangeAlign = outputDepth_;
            }
        }
        elementsPerCore = ((elementsPerCore + rangeAlign - 1U) / rangeAlign) * rangeAlign;
        uint64_t startIdx = blockIdx * elementsPerCore;
        uint64_t endIdx = startIdx + elementsPerCore;
        if (endIdx > outputTotal_) {
            endIdx = outputTotal_;
        }
        if (startIdx >= endIdx || startIdx >= outputTotal_) {
            return;
        }

        LocalTensor<DT_X> yLocal = workBuf_.Get<DT_X>();
        if (blockSize_ == 2U) {
            ProcessRangePower2<1U>(startIdx, endIdx, yLocal, elemsPerBlock);
            return;
        }
        if (blockSize_ == 4U) {
            ProcessRangePower2<2U>(startIdx, endIdx, yLocal, elemsPerBlock);
            return;
        }

        const uint64_t ubCapacity = static_cast<uint64_t>(workElems_);
        uint64_t currentOutBase = startIdx;
        uint64_t ubIdx = 0U;
        const bool canVectorizeRead = (outputDepth_ % static_cast<uint64_t>(elemsPerBlock)) == 0U;

        uint64_t p = startIdx / outputDepth_;
        uint64_t c = startIdx - p * outputDepth_;

        const uint64_t resetIw = cropLeft_ / blockSize_;
        const uint64_t resetOffsetW = cropLeft_ - resetIw * blockSize_;
        const uint64_t resetIh = cropTop_ / blockSize_;
        const uint64_t resetOffsetH = cropTop_ - resetIh * blockSize_;

        uint64_t outB = p / (outputWidth_ * outputHeight_);
        uint64_t rem = p - outB * outputWidth_ * outputHeight_;
        uint64_t outH = rem / outputWidth_;
        uint64_t outW = rem - outH * outputWidth_;

        uint64_t fullH = outH + cropTop_;
        uint64_t fullW = outW + cropLeft_;
        uint64_t inH = fullH / blockSize_;
        uint64_t laneH = fullH - inH * blockSize_;
        uint64_t inW = fullW / blockSize_;
        uint64_t laneW = fullW - inW * blockSize_;
        uint64_t inB = laneH * (blockSize_ * outputBatch_) + laneW * outputBatch_ + outB;
        uint64_t inBase = ((inB * inputHeight_ + inH) * inputWidth_ + inW) * inputDepth_;

        for (uint64_t iOut = startIdx; iOut < endIdx; ++iOut) {
            if (c == 0U && canVectorizeRead && (iOut + outputDepth_ <= endIdx) &&
                (ubIdx + outputDepth_ <= ubCapacity)) {
                DataCopy(yLocal[ubIdx], xGm_[inBase], static_cast<uint32_t>(outputDepth_));
                ubIdx += outputDepth_;
                iOut += outputDepth_ - 1U;
                AdvancePixel(outB, outH, outW, inB, inH, inW, laneH, laneW, inBase,
                             resetIw, resetOffsetW, resetIh, resetOffsetH);
                if (ubIdx == ubCapacity) {
                    FlushFull(currentOutBase, yLocal);
                    ubIdx = 0U;
                }
                continue;
            }

            yLocal.SetValue(ubIdx++, xGm_.GetValue(inBase + c));
            ++c;
            if (c == outputDepth_) {
                c = 0U;
                AdvancePixel(outB, outH, outW, inB, inH, inW, laneH, laneW, inBase,
                             resetIw, resetOffsetW, resetIh, resetOffsetH);
            }
            if (ubIdx == ubCapacity) {
                FlushFull(currentOutBase, yLocal);
                ubIdx = 0U;
            }
        }

        if (ubIdx > 0U) {
            FlushTail(currentOutBase, ubIdx, yLocal, elemsPerBlock);
        }
    }

private:
    template <uint64_t BLOCK_SHIFT>
    __aicore__ inline void ProcessRangePower2(uint64_t startIdx, uint64_t endIdx,
                                               LocalTensor<DT_X> &yLocal,
                                               uint32_t elemsPerBlock) {
        constexpr uint64_t kBlock = (1ULL << BLOCK_SHIFT);
        constexpr uint64_t kMask = kBlock - 1ULL;
        const uint64_t ubCapacity = static_cast<uint64_t>(workElems_);
        uint64_t currentOutBase = startIdx;
        uint64_t ubIdx = 0U;
        const bool canVectorizeRead = (outputDepth_ % static_cast<uint64_t>(elemsPerBlock)) == 0U;

        uint64_t p = startIdx / outputDepth_;
        uint64_t c = startIdx - p * outputDepth_;

        const uint64_t outArea = outputWidth_ * outputHeight_;
        uint64_t outB = p / outArea;
        uint64_t rem = p - outB * outArea;
        uint64_t outH = rem / outputWidth_;
        uint64_t outW = rem - outH * outputWidth_;

        const uint64_t resetIw = cropLeft_ >> BLOCK_SHIFT;
        const uint64_t resetLaneW = cropLeft_ & kMask;
        const uint64_t resetIh = cropTop_ >> BLOCK_SHIFT;
        const uint64_t resetLaneH = cropTop_ & kMask;

        const uint64_t batchStride = inputHeight_ * inputWidth_ * inputDepth_;
        const uint64_t rowStride = inputWidth_ * inputDepth_;
        const uint64_t laneBatchDelta = outputBatch_ * batchStride;
        const uint64_t laneWrapDelta = inputDepth_ - (kBlock - 1ULL) * laneBatchDelta;

        uint64_t fullH = outH + cropTop_;
        uint64_t fullW = outW + cropLeft_;
        uint64_t inH = fullH >> BLOCK_SHIFT;
        uint64_t laneH = fullH & kMask;
        uint64_t inW = fullW >> BLOCK_SHIFT;
        uint64_t laneW = fullW & kMask;
        uint64_t inB = ((laneH << BLOCK_SHIFT) + laneW) * outputBatch_ + outB;
        uint64_t inBase = inB * batchStride + inH * rowStride + inW * inputDepth_;

        if (c == 0U && canVectorizeRead && outputDepth_ <= ubCapacity &&
            (endIdx % outputDepth_) == 0U) {
            ProcessPixelAlignedPower2<BLOCK_SHIFT>(startIdx, endIdx, yLocal);
            return;
        }

        for (uint64_t iOut = startIdx; iOut < endIdx; ++iOut) {
            if (c == 0U && canVectorizeRead && (iOut + outputDepth_ <= endIdx) &&
                (ubIdx + outputDepth_ <= ubCapacity)) {
                DataCopy(yLocal[ubIdx], xGm_[inBase], static_cast<uint32_t>(outputDepth_));
                ubIdx += outputDepth_;
                iOut += outputDepth_ - 1U;
                AdvancePixelPower2<BLOCK_SHIFT>(outB, outH, outW, inH, inW, laneH, laneW, inBase,
                                                resetIw, resetLaneW, resetIh, resetLaneH,
                                                batchStride, rowStride, laneBatchDelta, laneWrapDelta);
                if (ubIdx == ubCapacity) {
                    FlushFull(currentOutBase, yLocal);
                    ubIdx = 0U;
                }
                continue;
            }

            yLocal.SetValue(ubIdx++, xGm_.GetValue(inBase + c));
            ++c;
            if (c == outputDepth_) {
                c = 0U;
                AdvancePixelPower2<BLOCK_SHIFT>(outB, outH, outW, inH, inW, laneH, laneW, inBase,
                                                resetIw, resetLaneW, resetIh, resetLaneH,
                                                batchStride, rowStride, laneBatchDelta, laneWrapDelta);
            }
            if (ubIdx == ubCapacity) {
                FlushFull(currentOutBase, yLocal);
                ubIdx = 0U;
            }
        }

        if (ubIdx > 0U) {
            FlushTail(currentOutBase, ubIdx, yLocal, elemsPerBlock);
        }
    }

    __aicore__ inline void FlushAlignedAndWait(uint64_t &currentOutBase, uint64_t ubIdx,
                                               LocalTensor<DT_X> &yLocal) {
        if (ubIdx == 0U) {
            return;
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(yGm_[currentOutBase], yLocal, static_cast<uint32_t>(ubIdx));
        PipeBarrier<PIPE_ALL>();
        currentOutBase += ubIdx;
    }

    __aicore__ inline void FlushAlignedNoWait(uint64_t &currentOutBase, uint64_t ubIdx,
                                             LocalTensor<DT_X> &yLocal) {
        if (ubIdx == 0U) {
            return;
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(yGm_[currentOutBase], yLocal, static_cast<uint32_t>(ubIdx));
        currentOutBase += ubIdx;
    }

    template <uint64_t BLOCK_SHIFT>
    __aicore__ inline void ProcessPixelAlignedPower2(uint64_t startIdx, uint64_t endIdx,
                                                     LocalTensor<DT_X> &yLocal) {
        constexpr uint64_t kBlock = (1ULL << BLOCK_SHIFT);
        constexpr uint64_t kMask = kBlock - 1ULL;
        const uint64_t ubCapacity = static_cast<uint64_t>(workElems_);
        uint64_t currentOutBase = startIdx;
        uint64_t ubIdx = 0U;

        uint64_t pixel = startIdx / outputDepth_;
        const uint64_t pixelEnd = endIdx / outputDepth_;
        const uint64_t outArea = outputWidth_ * outputHeight_;
        uint64_t outB = pixel / outArea;
        uint64_t rem = pixel - outB * outArea;
        uint64_t outH = rem / outputWidth_;
        uint64_t outW = rem - outH * outputWidth_;

        const uint64_t batchStride = inputHeight_ * inputWidth_ * inputDepth_;
        const uint64_t rowStride = inputWidth_ * inputDepth_;
        const uint64_t laneBatchDelta = outputBatch_ * batchStride;
        const uint64_t laneWrapDelta = inputDepth_ - (kBlock - 1ULL) * laneBatchDelta;

        while (pixel < pixelEnd) {
            const uint64_t rowRemain = outputWidth_ - outW;
            const uint64_t totalRemain = pixelEnd - pixel;
            const uint64_t pixelsThisRow = (rowRemain < totalRemain) ? rowRemain : totalRemain;

            const uint64_t fullH = outH + cropTop_;
            const uint64_t fullW = outW + cropLeft_;
            const uint64_t inH = fullH >> BLOCK_SHIFT;
            const uint64_t laneH = fullH & kMask;
            uint64_t inW = fullW >> BLOCK_SHIFT;
            uint64_t laneW = fullW & kMask;
            uint64_t inB = ((laneH << BLOCK_SHIFT) + laneW) * outputBatch_ + outB;
            uint64_t inBase = inB * batchStride + inH * rowStride + inW * inputDepth_;

            for (uint64_t j = 0U; j < pixelsThisRow; ++j) {
                if (ubIdx + outputDepth_ > ubCapacity) {
                    FlushAlignedAndWait(currentOutBase, ubIdx, yLocal);
                    ubIdx = 0U;
                }
                DataCopy(yLocal[ubIdx], xGm_[inBase], static_cast<uint32_t>(outputDepth_));
                ubIdx += outputDepth_;

                ++laneW;
                if (laneW == kBlock) {
                    laneW = 0U;
                    ++inW;
                    inBase += laneWrapDelta;
                } else {
                    inBase += laneBatchDelta;
                }
            }

            pixel += pixelsThisRow;
            outW += pixelsThisRow;
            if (outW == outputWidth_) {
                outW = 0U;
                ++outH;
                if (outH == outputHeight_) {
                    outH = 0U;
                    ++outB;
                }
            }
        }
        if (ubIdx > 0U) {
            FlushAlignedNoWait(currentOutBase, ubIdx, yLocal);
        }
    }

    template <uint64_t BLOCK_SHIFT>
    __aicore__ inline void AdvancePixelPower2(uint64_t &outB, uint64_t &outH, uint64_t &outW,
                                              uint64_t &inH, uint64_t &inW,
                                              uint64_t &laneH, uint64_t &laneW,
                                              uint64_t &inBase,
                                              uint64_t resetIw, uint64_t resetLaneW,
                                              uint64_t resetIh, uint64_t resetLaneH,
                                              uint64_t batchStride, uint64_t rowStride,
                                              uint64_t laneBatchDelta, uint64_t laneWrapDelta) {
        constexpr uint64_t kBlock = (1ULL << BLOCK_SHIFT);
        constexpr uint64_t kMask = kBlock - 1ULL;
        ++outW;
        if (outW == outputWidth_) {
            outW = 0U;
            ++outH;
            if (outH == outputHeight_) {
                outH = 0U;
                ++outB;
            }
            const uint64_t fullH = outH + cropTop_;
            inH = fullH >> BLOCK_SHIFT;
            laneH = fullH & kMask;
            inW = resetIw;
            laneW = resetLaneW;
            const uint64_t inB = ((laneH << BLOCK_SHIFT) + laneW) * outputBatch_ + outB;
            inBase = inB * batchStride + inH * rowStride + inW * inputDepth_;
            return;
        }

        ++laneW;
        if (laneW == kBlock) {
            laneW = 0U;
            ++inW;
            inBase += laneWrapDelta;
        } else {
            inBase += laneBatchDelta;
        }
    }

private:
    __aicore__ inline void AdvancePixel(uint64_t &outB, uint64_t &outH, uint64_t &outW,
                                        uint64_t &inB, uint64_t &inH, uint64_t &inW,
                                        uint64_t &laneH, uint64_t &laneW, uint64_t &inBase,
                                        uint64_t resetIw, uint64_t resetOffsetW,
        uint64_t resetIh, uint64_t resetOffsetH) {
        ++outW;
        ++laneW;
        if (laneW == blockSize_) {
            laneW = 0U;
            ++inW;
        }
        if (outW == outputWidth_) {
            outW = 0U;
            inW = resetIw;
            laneW = resetOffsetW;
            ++outH;
            ++laneH;
            if (laneH == blockSize_) {
                laneH = 0U;
                ++inH;
            }
            if (outH == outputHeight_) {
                outH = 0U;
                inH = resetIh;
                laneH = resetOffsetH;
                ++outB;
            }
        }
        inB = laneH * (blockSize_ * outputBatch_) + laneW * outputBatch_ + outB;
        inBase = ((inB * inputHeight_ + inH) * inputWidth_ + inW) * inputDepth_;
    }

    __aicore__ inline void FlushFull(uint64_t &currentOutBase, LocalTensor<DT_X> &yLocal) {
        PipeBarrier<PIPE_ALL>();
        DataCopy(yGm_[currentOutBase], yLocal, workElems_);
        PipeBarrier<PIPE_ALL>();
        currentOutBase += static_cast<uint64_t>(workElems_);
    }

    __aicore__ inline void FlushTail(uint64_t &currentOutBase, uint64_t ubIdx,
                                     LocalTensor<DT_X> &yLocal, uint32_t elemsPerBlock) {
        PipeBarrier<PIPE_ALL>();
        const uint64_t aligned = (ubIdx / static_cast<uint64_t>(elemsPerBlock)) * static_cast<uint64_t>(elemsPerBlock);
        if (aligned > 0U) {
            DataCopy(yGm_[currentOutBase], yLocal, static_cast<uint32_t>(aligned));
            currentOutBase += aligned;
        }
        for (uint64_t i = aligned; i < ubIdx; ++i) {
            yGm_.SetValue(currentOutBase++, yLocal.GetValue(i));
        }
    }

private:
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> workBuf_;
    uint64_t inputBatch_;
    uint64_t inputHeight_;
    uint64_t inputWidth_;
    uint64_t inputDepth_;
    uint64_t outputBatch_;
    uint64_t outputHeight_;
    uint64_t outputWidth_;
    uint64_t outputDepth_;
    uint64_t outputTotal_;
    uint64_t blockSize_;
    uint64_t cropTop_;
    uint64_t cropBottom_;
    uint64_t cropLeft_;
    uint64_t cropRight_;
    uint32_t workElems_;
};









template <class DT_X>
class KernelBatchToSpaceT7C1_8x8SmallCropMc8 {
public:
    __aicore__ inline KernelBatchToSpaceT7C1_8x8SmallCropMc8() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling) {
        outputWidth_ = tiling.outW;
        outputTotal_ = tiling.totalElements;
        cropTop_ = tiling.cropTop;
        cropLeft_ = tiling.cropLeft;
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, 256U);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, outputTotal_);
    }

    __aicore__ inline bool CanRun() const {
        return outputTotal_ >= 224U && outputTotal_ <= 256U &&
               outputWidth_ >= 14U && outputWidth_ <= 16U;
    }

    __aicore__ inline void Process() {
        const uint64_t start = static_cast<uint64_t>(GetBlockIdx()) << 5U;
        if (start >= outputTotal_) {
            return;
        }
        uint64_t end = start + 32U;
        if (end > outputTotal_) {
            end = outputTotal_;
        }
        if (outputWidth_ == 16U) {
            for (uint64_t idx = start; idx < end; ++idx) {
                ProcessOneW16(idx);
            }
        } else {
            for (uint64_t idx = start; idx < end; ++idx) {
                ProcessOneDynamicW(idx);
            }
        }
    }

private:
    __aicore__ inline void ProcessOneW16(const uint64_t idx) {
        const uint64_t oh = idx >> 4U;
        const uint64_t ow = idx & 15U;
        const uint64_t fullH = oh + cropTop_;
        const uint64_t fullW = ow + cropLeft_;
        const uint64_t laneH = fullH & 1U;
        const uint64_t laneW = fullW & 1U;
        const uint64_t inH = fullH >> 1U;
        const uint64_t inW = fullW >> 1U;
        const uint64_t inB = (laneH << 1U) | laneW;
        const uint64_t src = (inB << 6U) + (inH << 3U) + inW;
        yGm_.SetValue(idx, xGm_.GetValue(src));
    }

    __aicore__ inline void ProcessOneDynamicW(const uint64_t idx) {
        const uint64_t oh = idx / outputWidth_;
        const uint64_t ow = idx - oh * outputWidth_;
        const uint64_t fullH = oh + cropTop_;
        const uint64_t fullW = ow + cropLeft_;
        const uint64_t laneH = fullH & 1U;
        const uint64_t laneW = fullW & 1U;
        const uint64_t inH = fullH >> 1U;
        const uint64_t inW = fullW >> 1U;
        const uint64_t inB = (laneH << 1U) | laneW;
        const uint64_t src = (inB << 6U) + (inH << 3U) + inW;
        yGm_.SetValue(idx, xGm_.GetValue(src));
    }

    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint64_t outputWidth_;
    uint64_t outputTotal_;
    uint64_t cropTop_;
    uint64_t cropLeft_;
};






template <class DT_X>
class KernelBatchToSpaceT7Bs2SmallCTinyWideOwner {
public:
    __aicore__ inline KernelBatchToSpaceT7Bs2SmallCTinyWideOwner() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling) {
        N_ = tiling.N;
        H_ = tiling.H;
        W_ = tiling.W;
        C_ = tiling.C;
        outN_ = tiling.outN;
        outH_ = tiling.outH;
        outW_ = tiling.outW;
        cropTop_ = tiling.cropTop;
        cropLeft_ = tiling.cropLeft;
        total_ = tiling.totalElements;
        elemsPerCore_ = tiling.elementsPerCore;
        const uint64_t inputTotal = N_ * H_ * W_ * C_;
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, inputTotal);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, total_);
    }

    __aicore__ inline bool CanRun() const {
        return C_ >= 1U && C_ <= 4U && outN_ >= 1U && outN_ <= 2U &&
               outH_ >= 2U && outW_ >= 2U && total_ > 0U && total_ <= 4096U &&
               elemsPerCore_ > 0U;
    }

    __aicore__ inline void Process() {
        const uint64_t start = static_cast<uint64_t>(GetBlockIdx()) * elemsPerCore_;
        if (start >= total_) {
            return;
        }
        uint64_t end = start + elemsPerCore_;
        if (end > total_) {
            end = total_;
        }
        if (C_ == 1U) {
            ProcessC1(start, end);
        } else if (C_ == 2U) {
            ProcessC2(start, end);
        } else if (C_ == 4U) {
            ProcessC4(start, end);
        } else {
            ProcessGenericC(start, end);
        }
    }

private:
    __aicore__ inline void StoreOne(const uint64_t outB, const uint64_t oh, const uint64_t ow,
                                    const uint64_t c, const uint64_t dst) {
        const uint64_t fullH = oh + cropTop_;
        const uint64_t fullW = ow + cropLeft_;
        const uint64_t laneH = fullH & 1U;
        const uint64_t laneW = fullW & 1U;
        const uint64_t inH = fullH >> 1U;
        const uint64_t inW = fullW >> 1U;
        const uint64_t lane = (laneH << 1U) | laneW;
        const uint64_t inB = lane * outN_ + outB;
        const uint64_t src = (((inB * H_ + inH) * W_ + inW) * C_) + c;
        yGm_.SetValue(dst, xGm_.GetValue(src));
    }

    __aicore__ inline void ProcessC1(const uint64_t start, const uint64_t end) {
        if (outW_ == 16U) {
            for (uint64_t idx = start; idx < end; ++idx) {
                const uint64_t ow = idx & 15U;
                uint64_t t = idx >> 4U;
                const uint64_t oh = t % outH_;
                const uint64_t outB = t / outH_;
                StoreOne(outB, oh, ow, 0U, idx);
            }
        } else {
            for (uint64_t idx = start; idx < end; ++idx) {
                const uint64_t ow = idx % outW_;
                uint64_t t = idx / outW_;
                const uint64_t oh = t % outH_;
                const uint64_t outB = t / outH_;
                StoreOne(outB, oh, ow, 0U, idx);
            }
        }
    }

    __aicore__ inline void ProcessC2(const uint64_t start, const uint64_t end) {
        for (uint64_t idx = start; idx < end; ++idx) {
            const uint64_t c = idx & 1U;
            uint64_t t = idx >> 1U;
            const uint64_t ow = t % outW_;
            t /= outW_;
            const uint64_t oh = t % outH_;
            const uint64_t outB = t / outH_;
            StoreOne(outB, oh, ow, c, idx);
        }
    }

    __aicore__ inline void ProcessC4(const uint64_t start, const uint64_t end) {
        for (uint64_t idx = start; idx < end; ++idx) {
            const uint64_t c = idx & 3U;
            uint64_t t = idx >> 2U;
            const uint64_t ow = t % outW_;
            t /= outW_;
            const uint64_t oh = t % outH_;
            const uint64_t outB = t / outH_;
            StoreOne(outB, oh, ow, c, idx);
        }
    }

    __aicore__ inline void ProcessGenericC(const uint64_t start, const uint64_t end) {
        for (uint64_t idx = start; idx < end; ++idx) {
            const uint64_t c = idx % C_;
            uint64_t t = idx / C_;
            const uint64_t ow = t % outW_;
            t /= outW_;
            const uint64_t oh = t % outH_;
            const uint64_t outB = t / outH_;
            StoreOne(outB, oh, ow, c, idx);
        }
    }

    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint64_t N_;
    uint64_t H_;
    uint64_t W_;
    uint64_t C_;
    uint64_t outN_;
    uint64_t outH_;
    uint64_t outW_;
    uint64_t cropTop_;
    uint64_t cropLeft_;
    uint64_t total_;
    uint64_t elemsPerCore_;
};


template <class DT_X>
class KernelBatchToSpaceT7B5W12AlignedPerf {
public:
    __aicore__ inline KernelBatchToSpaceT7B5W12AlignedPerf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling) {
        inputBatch_ = tiling.N;
        inputHeight_ = tiling.H;
        inputWidth_ = tiling.W;
        inputDepth_ = tiling.C;
        outputBatch_ = tiling.outN;
        outputHeight_ = tiling.outH;
        outputWidth_ = tiling.outW;
        outputDepth_ = tiling.C;
        outputTotal_ = tiling.totalElements;
        blockSize_ = static_cast<uint64_t>(tiling.blockSize);
        cropTop_ = static_cast<uint64_t>(tiling.cropTop);
        cropBottom_ = static_cast<uint64_t>(tiling.cropBottom);
        cropLeft_ = static_cast<uint64_t>(tiling.cropLeft);
        cropRight_ = static_cast<uint64_t>(tiling.cropRight);
        workElems_ = tiling.fastWorkElems;
        if (workElems_ == 0U) workElems_ = 1U;
        const uint64_t inputTotal = inputBatch_ * inputHeight_ * inputWidth_ * inputDepth_;
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, inputTotal);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, outputTotal_);
        pipe_.InitBuffer(workBuf_, static_cast<uint64_t>(workElems_) * sizeof(DT_X));
    }

    __aicore__ inline bool CanRun() const {
        if (outputTotal_ == 0U || inputDepth_ == 0U || outputDepth_ == 0U ||
            outputWidth_ == 0U || outputHeight_ == 0U || outputBatch_ == 0U) return false;
        if (blockSize_ != 2U || inputDepth_ != outputDepth_) return false;
        if (inputBatch_ != outputBatch_ * 4U) return false;
        if (outputHeight_ + cropTop_ + cropBottom_ != inputHeight_ * 2U) return false;
        if (outputWidth_ + cropLeft_ + cropRight_ != inputWidth_ * 2U) return false;
        if (outputWidth_ > 2U) return false;
        const uint64_t elemsPerBlock = static_cast<uint64_t>(kAlignBytes / sizeof(DT_X));
        if (elemsPerBlock == 0U) return false;
        const uint64_t depthBytes = outputDepth_ * static_cast<uint64_t>(sizeof(DT_X));
        const uint64_t rowBytes = outputWidth_ * depthBytes;
        if (depthBytes == 0U || rowBytes <= 128U) return false;
        if ((depthBytes & static_cast<uint64_t>(kAlignBytes - 1U)) != 0U) return false;
        if ((rowBytes & static_cast<uint64_t>(kAlignBytes - 1U)) != 0U) return false;
        if ((static_cast<uint64_t>(workElems_) % elemsPerBlock) != 0U) return false;
        if (static_cast<uint64_t>(workElems_) < elemsPerBlock) return false;
        if (depthBytes > 0xFFFFFFFFULL) return false;
        const uint64_t inputRowBytes = inputWidth_ * depthBytes;
        if (inputRowBytes < depthBytes) return false;
        const uint64_t srcGap = inputRowBytes - depthBytes;
        if (srcGap > 0xFFFFFFFFULL) return false;
        const uint64_t dstGap = 2U * rowBytes - depthBytes;
        if ((dstGap & static_cast<uint64_t>(kAlignBytes - 1U)) != 0U) return false;
        if ((dstGap / static_cast<uint64_t>(kAlignBytes)) > 0xFFFFFFFFULL) return false;
        return true;
    }

    __aicore__ inline void Process() {
        if (!CanRun()) return;
        const uint64_t rowElems = outputWidth_ * outputDepth_;
        if (static_cast<uint64_t>(workElems_) >= rowElems) {
            ProcessRowGroups();
        } else {
            ProcessDepthChunks();
        }
    }

private:
    __aicore__ inline uint64_t RowsPerTile() const {
        const uint64_t rowElems = outputWidth_ * outputDepth_;
        if (rowElems == 0U) return 1U;
        uint64_t rows = static_cast<uint64_t>(workElems_) / rowElems;
        if (rows == 0U) rows = 1U;
        return rows;
    }

    __aicore__ inline uint64_t FirstLocalRowForLaneH(uint64_t ohStart, uint64_t tileRows, uint64_t laneH) const {
        const uint64_t firstFullH = ohStart + cropTop_;
        const uint64_t curLaneH = firstFullH & 1U;
        uint64_t first = 0U;
        if (curLaneH != laneH) first = 1U;
        if (first >= tileRows) return tileRows;
        return first;
    }

    __aicore__ inline void CopyRowsForLaneOw(uint64_t outB, uint64_t ohStart, uint64_t tileRows,
                                             uint64_t laneH, uint64_t ow, LocalTensor<DT_X> &local) {
        const uint64_t firstLocalRow = FirstLocalRowForLaneH(ohStart, tileRows, laneH);
        if (firstLocalRow >= tileRows) return;
        const uint64_t blockCount = ((tileRows - 1U - firstLocalRow) >> 1U) + 1U;
        if (blockCount == 0U || blockCount > 65535U) return;
        const uint64_t fullH0 = ohStart + firstLocalRow + cropTop_;
        const uint64_t inH0 = fullH0 >> 1U;
        const uint64_t fullW = ow + cropLeft_;
        const uint64_t laneW = fullW & 1U;
        const uint64_t inW = fullW >> 1U;
        const uint64_t lane = (laneH << 1U) + laneW;
        const uint64_t inB = lane * outputBatch_ + outB;
        const uint64_t srcOffset = ((inB * inputHeight_ + inH0) * inputWidth_ + inW) * inputDepth_;
        const uint64_t depthBytes = outputDepth_ * static_cast<uint64_t>(sizeof(DT_X));
        const uint64_t rowBytes = outputWidth_ * depthBytes;
        const uint64_t inputRowBytes = inputWidth_ * depthBytes;
        const uint64_t srcGap = inputRowBytes - depthBytes;
        const uint64_t dstGap = 2U * rowBytes - depthBytes;
        const uint64_t localOffset = firstLocalRow * outputWidth_ * outputDepth_ + ow * outputDepth_;
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(blockCount);
        copyParams.blockLen = static_cast<uint32_t>(depthBytes);
        copyParams.srcStride = static_cast<uint32_t>(srcGap);
        copyParams.dstStride = static_cast<uint32_t>(dstGap / static_cast<uint64_t>(kAlignBytes));
        DataCopyPadExtParams<DT_X> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0U;
        padParams.rightPadding = 0U;
        padParams.paddingValue = static_cast<DT_X>(0);
        DataCopyPad(local[localOffset], xGm_[srcOffset], copyParams, padParams);
    }

    __aicore__ inline void ProcessTile(uint64_t outB, uint64_t ohStart, uint64_t tileRows,
                                       LocalTensor<DT_X> &local) {
        for (uint64_t laneH = 0U; laneH < 2U; ++laneH) {
            for (uint64_t ow = 0U; ow < outputWidth_; ++ow) {
                CopyRowsForLaneOw(outB, ohStart, tileRows, laneH, ow, local);
            }
        }
        PipeBarrier<PIPE_ALL>();
        const uint64_t rowElems = outputWidth_ * outputDepth_;
        const uint64_t dstOffset = (outB * outputHeight_ + ohStart) * rowElems;
        const uint64_t elemCount = tileRows * rowElems;
        DataCopy(yGm_[dstOffset], local, static_cast<uint32_t>(elemCount));
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessRowGroups() {
        const uint64_t rowsPerTile = RowsPerTile();
        const uint64_t rowGroupsPerBatch = (outputHeight_ + rowsPerTile - 1U) / rowsPerTile;
        const uint64_t totalJobs = outputBatch_ * rowGroupsPerBatch;
        if (totalJobs == 0U) return;
        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
        const uint64_t jobStart = totalJobs * blockIdx / blockNum;
        const uint64_t jobEnd = totalJobs * (blockIdx + 1U) / blockNum;
        LocalTensor<DT_X> local = workBuf_.Get<DT_X>();
        for (uint64_t job = jobStart; job < jobEnd; ++job) {
            const uint64_t outB = job / rowGroupsPerBatch;
            const uint64_t group = job - outB * rowGroupsPerBatch;
            const uint64_t ohStart = group * rowsPerTile;
            if (ohStart >= outputHeight_) continue;
            uint64_t tileRows = outputHeight_ - ohStart;
            if (tileRows > rowsPerTile) tileRows = rowsPerTile;
            ProcessTile(outB, ohStart, tileRows, local);
        }
    }

    __aicore__ inline void ProcessOneDepthChunk(uint64_t outB, uint64_t oh, uint64_t ow,
                                                uint64_t chunkStart, uint64_t chunkElems,
                                                LocalTensor<DT_X> &local) {
        const uint64_t fullH = oh + cropTop_;
        const uint64_t fullW = ow + cropLeft_;
        const uint64_t laneH = fullH & 1U;
        const uint64_t laneW = fullW & 1U;
        const uint64_t inH = fullH >> 1U;
        const uint64_t inW = fullW >> 1U;
        const uint64_t lane = (laneH << 1U) + laneW;
        const uint64_t inB = lane * outputBatch_ + outB;
        const uint64_t srcOffset = ((inB * inputHeight_ + inH) * inputWidth_ + inW) * inputDepth_ + chunkStart;
        const uint64_t dstOffset = ((outB * outputHeight_ + oh) * outputWidth_ + ow) * outputDepth_ + chunkStart;
        DataCopy(local, xGm_[srcOffset], static_cast<uint32_t>(chunkElems));
        PipeBarrier<PIPE_ALL>();
        DataCopy(yGm_[dstOffset], local, static_cast<uint32_t>(chunkElems));
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessDepthChunks() {
        const uint64_t work = static_cast<uint64_t>(workElems_);
        const uint64_t chunksPerDepth = (outputDepth_ + work - 1U) / work;
        const uint64_t totalJobs = outputBatch_ * outputHeight_ * outputWidth_ * chunksPerDepth;
        if (totalJobs == 0U) return;
        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
        const uint64_t jobStart = totalJobs * blockIdx / blockNum;
        const uint64_t jobEnd = totalJobs * (blockIdx + 1U) / blockNum;
        LocalTensor<DT_X> local = workBuf_.Get<DT_X>();
        for (uint64_t job = jobStart; job < jobEnd; ++job) {
            uint64_t t = job;
            const uint64_t chunk = t % chunksPerDepth;
            t /= chunksPerDepth;
            const uint64_t ow = t % outputWidth_;
            t /= outputWidth_;
            const uint64_t oh = t % outputHeight_;
            const uint64_t outB = t / outputHeight_;
            const uint64_t chunkStart = chunk * work;
            if (chunkStart >= outputDepth_) continue;
            uint64_t chunkElems = outputDepth_ - chunkStart;
            if (chunkElems > work) chunkElems = work;
            ProcessOneDepthChunk(outB, oh, ow, chunkStart, chunkElems, local);
        }
    }

private:
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> workBuf_;
    uint64_t inputBatch_;
    uint64_t inputHeight_;
    uint64_t inputWidth_;
    uint64_t inputDepth_;
    uint64_t outputBatch_;
    uint64_t outputHeight_;
    uint64_t outputWidth_;
    uint64_t outputDepth_;
    uint64_t outputTotal_;
    uint64_t blockSize_;
    uint64_t cropTop_;
    uint64_t cropBottom_;
    uint64_t cropLeft_;
    uint64_t cropRight_;
    uint32_t workElems_;
};

template <typename DT_X, uint32_t MODE>
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    (void)workspace;
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingData, tiling_data, tiling);
    if constexpr (MODE == BATCH_TO_SPACE_MODE_BLOCK1_VERTICAL_CROP_CONTIG) {
        KernelBatchToSpaceBlock1VerticalCropContig<DT_X> verticalCrop;
        verticalCrop.Init(x, y, tiling_data);
        if (verticalCrop.CanRun()) {
            verticalCrop.Process();
            return;
        }
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_BLOCK1_CROP_ROW_CONTRACT) {
        KernelBatchToSpaceBlock1CropRowContract<DT_X> cropContract;
        cropContract.Init(x, y, tiling_data);
        if (cropContract.CanRun()) {
            cropContract.Process();
            return;
        }
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_SMALLC_ROW_OWNER_BS2) {
        KernelBatchToSpaceSmallCRowOwner<DT_X, 2U, 1U> rowOwner2;
        rowOwner2.Init(x, y, tiling_data);
        if (rowOwner2.CanRun()) {
            rowOwner2.Process();
            return;
        }
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_SMALLC_ROW_OWNER_BS4) {
        KernelBatchToSpaceSmallCRowOwner<DT_X, 4U, 2U> rowOwner4;
        rowOwner4.Init(x, y, tiling_data);
        if (rowOwner4.CanRun()) {
            rowOwner4.Process();
            return;
        }
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_T4_BS2_MID_LANE_CONTRACT) {



        const uint64_t outBytes = tiling_data.totalElements * static_cast<uint64_t>(sizeof(DT_X));
        const uint64_t rowBytes = tiling_data.outW * tiling_data.C * static_cast<uint64_t>(sizeof(DT_X));
        if (tiling_data.blockSize == 2U &&
            outBytes > (16ULL * 1024ULL) && outBytes <= (64ULL * 1024ULL) &&
            rowBytes > 256ULL && rowBytes <= (4ULL * 1024ULL) &&
            tiling_data.C > 0U && tiling_data.C <= 128U) {








            KernelBatchToSpaceBs2RowGroupPack<DT_X, 6U> t4Row6;
            t4Row6.Init(x, y, tiling_data);
            if (t4Row6.CanRun()) {
                t4Row6.Process();
                return;
            }


            KernelBatchToSpaceLaneStridedScatter<DT_X, 2U, 2U, 1U, 0U> t4Lane;
            t4Lane.Init(x, y, tiling_data);
            if (t4Lane.CanRun()) {
                t4Lane.Process();
                return;
            }
        }
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_T7_C1_8X8_SMALLCROP_MC8) {
        KernelBatchToSpaceT7C1_8x8SmallCropMc8<DT_X> t7;
        t7.Init(x, y, tiling_data);
        if (t7.CanRun()) {
            t7.Process();
            return;
        }
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_T7_BS2_SMALLC_TINY_WIDE_OWNER) {
        KernelBatchToSpaceT7Bs2SmallCTinyWideOwner<DT_X> t7wide;
        t7wide.Init(x, y, tiling_data);
        if (t7wide.CanRun()) {
            t7wide.Process();
            return;
        }
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_T7_BUCKET_OUTW1_DELAY) {
        InjectT7BucketDelay<262144U>();
        KernelBatchToSpace<DT_X> core;
        core.Init(x, y, &tiling_data);
        core.Process();
        return;
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_T7_BUCKET_OUTW2_DELAY) {
        InjectT7BucketDelay<524288U>();
        KernelBatchToSpace<DT_X> core;
        core.Init(x, y, &tiling_data);
        core.Process();
        return;
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_T7_BUCKET_OUTW3_4_DELAY) {
        InjectT7BucketDelay<1048576U>();
        KernelBatchToSpace<DT_X> core;
        core.Init(x, y, &tiling_data);
        core.Process();
        return;
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_T7_BUCKET_OUTW5_7_DELAY) {
        InjectT7BucketDelay<1572864U>();
        KernelBatchToSpace<DT_X> core;
        core.Init(x, y, &tiling_data);
        core.Process();
        return;
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_T7_BUCKET_OTHER_DELAY) {
        InjectT7BucketDelay<2097152U>();
        KernelBatchToSpace<DT_X> core;
        core.Init(x, y, &tiling_data);
        core.Process();
        return;
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_T7_OTHER_B1_W3_4_ROW256_512_OUT64_128_DELAY) {
        InjectT7BucketDelay<262144U>();
        KernelBatchToSpace<DT_X> core;
        core.Init(x, y, &tiling_data);
        core.Process();
        return;
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_T7_OTHER_B2_W3_4_ROW256_512_OUT128_512_DELAY) {
        InjectT7BucketDelay<786432U>();
        KernelBatchToSpace<DT_X> core;
        core.Init(x, y, &tiling_data);
        core.Process();
        return;
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_T7_OTHER_B3_W3_4_ROW_GT512_DELAY) {
        InjectT7BucketDelay<1310720U>();
        KernelBatchToSpace<DT_X> core;
        core.Init(x, y, &tiling_data);
        core.Process();
        return;
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_T7_OTHER_B4_W5_7_ROW_GT256_DELAY) {
        InjectT7BucketDelay<1835008U>();
        KernelBatchToSpace<DT_X> core;
        core.Init(x, y, &tiling_data);
        core.Process();
        return;
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_T7_OTHER_B5_W1_2_ROW_GT128_PERF) {
        KernelBatchToSpaceT7B5W12AlignedPerf<DT_X> t7B5;
        t7B5.Init(x, y, tiling_data);
        t7B5.Process();
        return;
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_T7_OTHER_B6_ROWNARROW_OUTW_GE8_DELAY) {
        InjectT7BucketDelay<2883584U>();
        KernelBatchToSpace<DT_X> core;
        core.Init(x, y, &tiling_data);
        core.Process();
        return;
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_T7_OTHER_B7_NONALIGNED_CGT16_DELAY) {
        InjectT7BucketDelay<3407872U>();
        KernelBatchToSpace<DT_X> core;
        core.Init(x, y, &tiling_data);
        core.Process();
        return;
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_T7_OTHER_B8_OTHER_DELAY) {
        InjectT7BucketDelay<3932160U>();
        KernelBatchToSpace<DT_X> core;
        core.Init(x, y, &tiling_data);
        core.Process();
        return;
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_T6_BS2_FP16_HIGHDEPTH_MEDIUM_ROW2) {



        KernelBatchToSpace<DT_X> t6GeneralCore;
        t6GeneralCore.Init(x, y, &tiling_data);
        t6GeneralCore.Process();
        return;
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_T7_BS2_FP16_MEDIUM_NARROW) {

        KernelBatchToSpace<DT_X> t7GeneralCore;
        t7GeneralCore.Init(x, y, &tiling_data);
        t7GeneralCore.Process();
        return;
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_T7_OUTW1_ALIGNED_ROWCONTRACT) {


        KernelBatchToSpaceT7Bs2Fp16NarrowRowContract<DT_X, 1U> t7OutW1;
        t7OutW1.Init(x, y, tiling_data);
        if (t7OutW1.CanRun()) {
            t7OutW1.Process();
            return;
        }
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_T7_OUTW2_ALIGNED_ROWCONTRACT) {


        KernelBatchToSpaceT7Bs2Fp16NarrowRowContract<DT_X, 2U> t7OutW2;
        t7OutW2.Init(x, y, tiling_data);
        if (t7OutW2.CanRun()) {
            t7OutW2.Process();
            return;
        }
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_T7_OUTW4_ALIGNED_ROWCONTRACT) {


        KernelBatchToSpaceT7Bs2Fp16NarrowRowContract<DT_X, 4U> t7OutW4;
        t7OutW4.Init(x, y, tiling_data);
        if (t7OutW4.CanRun()) {
            t7OutW4.Process();
            return;
        }
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_T7_OUTW7_ALIGNED_ROWCONTRACT ||
                             MODE == BATCH_TO_SPACE_MODE_T7_BS2_FP16_MEDIUM_NARROW_LANE) {


        KernelBatchToSpaceLaneStridedScatter<DT_X, 2U, 2U, 1U, 1U> t7NarrowLane;
        t7NarrowLane.Init(x, y, tiling_data);
        if (t7NarrowLane.CanRun()) {
            t7NarrowLane.Process();
            return;
        }
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_T9_BS4_ALIGNED_ROW_ASSEMBLE) {
        KernelBatchToSpaceT9Bs4AlignedRowAssemble<DT_X> t9Row;
        t9Row.Init(x, y, tiling_data);
        if (t9Row.CanRun()) {
            t9Row.Process();
            return;
        }
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_LANE_STRIDED_SCATTER_SAFE) {
        KernelBatchToSpaceLaneStridedScatter<DT_X, 2U, 4U, kScatterDiagMinTileCols, 4U> fast;
        fast.Init(x, y, tiling_data);
        if (fast.CanRun()) {
            fast.Process();
            return;
        }
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_BS2_ROWGROUP_PACK) {
        KernelBatchToSpaceBs2RowGroupPack<DT_X, 32U> rowPack;
        rowPack.Init(x, y, tiling_data);
        if (rowPack.CanRun()) {
            rowPack.Process();
            return;
        }
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_BS2_ROWGROUP_PACK_16) {
        KernelBatchToSpaceBs2RowGroupPack<DT_X, 16U> rowPack16;
        rowPack16.Init(x, y, tiling_data);
        if (rowPack16.CanRun()) {
            rowPack16.Process();
            return;
        }
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_BS2_ROWGROUP_PACK_8) {
        KernelBatchToSpaceBs2RowGroupPack<DT_X, 8U> rowPack8;
        rowPack8.Init(x, y, tiling_data);
        if (rowPack8.CanRun()) {
            rowPack8.Process();
            return;
        }
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_HIGH_DEPTH_CONTIG_PACK) {
        KernelBatchToSpaceHighDepthContigPack<DT_X, 2U> pack;
        pack.Init(x, y, tiling_data);
        if (pack.CanRun()) {
            pack.Process();
            return;
        }
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_BS4_HIGH_DEPTH_CONTIG_PACK) {
        KernelBatchToSpaceHighDepthContigPack<DT_X, 4U> pack4;
        pack4.Init(x, y, tiling_data);
        if (pack4.CanRun()) {
            pack4.Process();
            return;
        }
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_HIGH_DILATION_LANE_SCATTER) {
        KernelBatchToSpaceLaneStridedScatter<DT_X, 5U, 24U, 1U, 0U> highDil;
        highDil.Init(x, y, tiling_data);
        if (highDil.CanRun()) {
            highDil.Process();
            return;
        }
    } else if constexpr (MODE == BATCH_TO_SPACE_MODE_LIGHTWEIGHT_GENERIC) {
        KernelBatchToSpaceLightweightGeneric<DT_X> light;
        light.Init(x, y, &tiling_data);
        if (light.CanRun()) {
            light.Process();
            return;
        }
    }
    KernelBatchToSpace<DT_X> op;
    op.Init(x, y, &tiling_data);
    op.Process();
}
