#include "kernel_operator.h"

#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

template <class DT_X, int COPY_MODE>
class KernelBatchToSpace {
public:
    __aicore__ inline KernelBatchToSpace() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tilingData) {
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, tilingData.inputLength);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, tilingData.outputLength);
        height_ = tilingData.height;
        width_ = tilingData.width;
        depth_ = tilingData.depth;
        outBatch_ = tilingData.outBatch;
        outHeight_ = tilingData.outHeight;
        outWidth_ = tilingData.outWidth;
        cropTop_ = tilingData.cropTop;
        cropLeft_ = tilingData.cropLeft;
        blockSize_ = tilingData.blockSize;
        usedCoreNum_ = tilingData.usedCoreNum;
        outputLength_ = tilingData.outputLength;
        coreStride_ = tilingData.coreStride;
        corePixelStride_ = tilingData.corePixelStride;
        rowGroupLength_ = tilingData.rowGroupLength;
        depthSplit_ = tilingData.canUseBlockCopy;
        if (COPY_MODE == 72 || COPY_MODE == 73 || COPY_MODE == 78) {
            pipe_.InitBuffer(gatherInQueue_, 3, 8320);
            pipe_.InitBuffer(gatherOutQueue_, 1, 4160);
        } else if (COPY_MODE == 59) {
            pipe_.InitBuffer(gatherInQueue_, 2, 8320);
            pipe_.InitBuffer(gatherOutQueue_, 1, 4160);
        } else if (COPY_MODE == 15 || COPY_MODE == 49) {
            pipe_.InitBuffer(gatherInQueue_, 2, 32768);
            pipe_.InitBuffer(gatherOutQueue_, 1, 32768);
        } else if (COPY_MODE == 2 || COPY_MODE == 3 || COPY_MODE == 5 || COPY_MODE == 6 ||
            COPY_MODE == 7 || COPY_MODE == 77 || COPY_MODE == 8 || COPY_MODE == 9 || COPY_MODE == 10 ||
            COPY_MODE == 18 ||
            COPY_MODE == 11 || COPY_MODE == 12 || COPY_MODE == 14 || COPY_MODE == 16 ||
            COPY_MODE == 20 ||
            COPY_MODE == 17) {
            pipe_.InitBuffer(copyQueue_, 1, copyUbBytes_);
        }
    }
    __aicore__ inline void Process() {
        if (COPY_MODE == 2) {
            ProcessB2RowCopy();
        } else if (COPY_MODE == 17) {
            ProcessB2DepthSplitRowCopy();
        } else if (COPY_MODE == 20) {
            ProcessB2RowtileAssembleCopy();
        } else if (COPY_MODE == 16) {
            ProcessB2RowsAssembleCopy();
        } else if (COPY_MODE == 15) {
            ProcessB2GatherInterleaveCopy();
        } else if (COPY_MODE == 14) {
            ProcessB2LocalInterleaveCopy();
        } else if (COPY_MODE == 59) {
            ProcessB2Case5Gather32SmallUbChunks();
        } else if (COPY_MODE == 72) {
            ProcessB2Case5InputPipelineChunks();
        } else if (COPY_MODE == 73) {
            ProcessB2Case5InputPipelineNoFlushChunks();
        } else if (COPY_MODE == 78) {
            ProcessB2Case5Exact65NoFlushChunks();
        } else if (COPY_MODE == 11) {
            ProcessB2AssembleCopy();
        } else if (COPY_MODE == 12) {
            ProcessB2PadAssembleCopy();
        } else if (COPY_MODE == 3) {
            ProcessB1RowCopy();
        } else if (COPY_MODE == 5) {
            ProcessB2PadCopy();
        } else if (COPY_MODE == 6) {
            ProcessB1PadCopy();
        } else if (COPY_MODE == 7) {
            ProcessBnRowCopy();
        } else if (COPY_MODE == 77) {
            ProcessBnAssembleRowCopy();
        } else if (COPY_MODE == 8) {
            ProcessBnPadCopy();
        } else if (COPY_MODE == 9) {
            ProcessLargeDepthCopy();
        } else if (COPY_MODE == 10) {
            ProcessB1ContigPadCopy();
        } else if (COPY_MODE == 18) {
            ProcessB1ContigCopy();
        } else if (COPY_MODE == 4) {
            ProcessDepth1RowScalar();
        } else if (COPY_MODE == 1) {
            ProcessScalarByDepth();
        } else {
            ProcessScalarByElement();
        }
    }

private:
    static constexpr uint32_t copyUbBytes_ = 32768;

    __aicore__ inline uint32_t GetInputBaseByPixel(uint32_t pixel_index) {
        uint32_t tmp = pixel_index;
        uint32_t ow = tmp % outWidth_;
        tmp = tmp / outWidth_;
        uint32_t oh = tmp % outHeight_;
        uint32_t on = tmp / outHeight_;

        uint32_t padded_h = oh + cropTop_;
        uint32_t padded_w = ow + cropLeft_;
        uint32_t ih;
        uint32_t iw;
        uint32_t block_h;
        uint32_t block_w;
        if (blockSize_ == 2) {
            ih = padded_h >> 1;
            iw = padded_w >> 1;
            block_h = padded_h & 1;
            block_w = padded_w & 1;
        } else if (blockSize_ == 1) {
            ih = padded_h;
            iw = padded_w;
            block_h = 0;
            block_w = 0;
        } else {
            ih = padded_h / blockSize_;
            iw = padded_w / blockSize_;
            block_h = padded_h - ih * blockSize_;
            block_w = padded_w - iw * blockSize_;
        }

        uint32_t in_n = (block_h * blockSize_ + block_w) * outBatch_ + on;
        return ((in_n * height_ + ih) * width_ + iw) * depth_;
    }

    __aicore__ inline void ProcessScalarByDepth() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        uint32_t output_pixels = outBatch_ * outHeight_ * outWidth_;
        uint32_t start_pixel = core_idx * corePixelStride_;
        uint32_t end_pixel = start_pixel + corePixelStride_;
        if (end_pixel > output_pixels) {
            end_pixel = output_pixels;
        }

        for (uint32_t pixel = start_pixel; pixel < end_pixel; ++pixel) {
            uint32_t input_base = GetInputBaseByPixel(pixel);
            uint32_t output_base = pixel * depth_;
            for (uint32_t c = 0; c < depth_; ++c) {
                yGm_.SetValue(output_base + c, xGm_.GetValue(input_base + c));
            }
        }

        AscendC::DataCacheCleanAndInvalid<DT_X, AscendC::CacheLine::ENTIRE_DATA_CACHE>(yGm_);
    }

    __aicore__ inline void ProcessB2AssembleCopy() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        uint32_t depth_blocks = depth_ * sizeof(DT_X) / 32;
        uint32_t max_copy_width = copyUbBytes_ / (depth_ * sizeof(DT_X));
        if (max_copy_width > 4095) {
            max_copy_width = 4095;
        }

        AscendC::DataCopyParams copy_in;
        copy_in.blockLen = static_cast<uint16_t>(depth_blocks);
        copy_in.srcStride = 0;
        copy_in.dstStride = static_cast<uint16_t>(depth_blocks);

        AscendC::DataCopyParams copy_out;
        copy_out.blockCount = 1;
        copy_out.srcStride = 0;
        copy_out.dstStride = 0;

        uint32_t row_count = outBatch_ * outHeight_;
        uint32_t unit_count = row_count * rowGroupLength_;
        uint32_t start_unit = core_idx * corePixelStride_;
        uint32_t end_unit = start_unit + corePixelStride_;
        if (end_unit > unit_count) {
            end_unit = unit_count;
        }

        for (uint32_t unit = start_unit; unit < end_unit; ++unit) {
            uint32_t chunk = unit % rowGroupLength_;
            uint32_t row = unit / rowGroupLength_;
            uint32_t on = row / outHeight_;
            uint32_t out_h = row - on * outHeight_;
            uint32_t padded_h = out_h + cropTop_;
            uint32_t h = padded_h >> 1;
            uint32_t bh = padded_h & 1;
            uint32_t out_w0 = chunk * max_copy_width;
            if (out_w0 >= outWidth_) {
                continue;
            }
            uint32_t cur_out = outWidth_ - out_w0;
            if (cur_out > max_copy_width) {
                cur_out = max_copy_width;
            }

            AscendC::LocalTensor<DT_X> local = copyQueue_.AllocTensor<DT_X>();
            uint32_t padded_w0 = out_w0 + cropLeft_;
            for (uint32_t bw = 0; bw < 2; ++bw) {
                uint32_t first_rel = (padded_w0 & 1U) == bw ? 0 : 1;
                if (first_rel >= cur_out) {
                    continue;
                }
                uint32_t count = ((cur_out - 1 - first_rel) >> 1) + 1;
                uint32_t input_w0 = (padded_w0 + first_rel) >> 1;
                uint32_t in_n = ((bh << 1) + bw) * outBatch_ + on;
                uint32_t in_base = ((in_n * height_ + h) * width_ + input_w0) * depth_;
                copy_in.blockCount = static_cast<uint16_t>(count);
                AscendC::DataCopy(local[first_rel * depth_], xGm_[in_base], copy_in);
            }
            copyQueue_.EnQue(local);
            local = copyQueue_.DeQue<DT_X>();
            uint32_t out_base = ((on * outHeight_ + out_h) * outWidth_ + out_w0) * depth_;
            copy_out.blockLen = static_cast<uint16_t>(cur_out * depth_blocks);
            AscendC::DataCopy(yGm_[out_base], local, copy_out);
            copyQueue_.FreeTensor(local);
        }

    }

    __aicore__ inline void ProcessB2PadAssembleCopy() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        uint32_t depth_bytes = depth_ * sizeof(DT_X);
        uint32_t padded_bytes = ((depth_bytes + 31U) >> 5) << 5;
        uint32_t max_copy_width = copyUbBytes_ / padded_bytes;
        if (max_copy_width > 4095) {
            max_copy_width = 4095;
        }

        AscendC::DataCopyExtParams copy_in;
        copy_in.blockCount = 1;
        copy_in.blockLen = depth_bytes;
        copy_in.srcStride = 0;
        copy_in.dstStride = 0;
        copy_in.rsv = 0;

        AscendC::DataCopyExtParams copy_out;
        copy_out.blockCount = 1;
        copy_out.dstStride = 0;
        copy_out.srcStride = 0;
        copy_out.rsv = 0;

        AscendC::DataCopyPadExtParams<DT_X> pad_params;
        pad_params.isPad = false;
        pad_params.leftPadding = 0;
        pad_params.rightPadding = 0;
        pad_params.paddingValue = 0;

        uint32_t row_count = outBatch_ * outHeight_;
        uint32_t unit_count = row_count * rowGroupLength_;
        uint32_t start_unit = core_idx * corePixelStride_;
        uint32_t end_unit = start_unit + corePixelStride_;
        if (end_unit > unit_count) {
            end_unit = unit_count;
        }

        for (uint32_t unit = start_unit; unit < end_unit; ++unit) {
            uint32_t chunk = unit % rowGroupLength_;
            uint32_t row = unit / rowGroupLength_;
            uint32_t on = row / outHeight_;
            uint32_t out_h = row - on * outHeight_;
            uint32_t padded_h = out_h + cropTop_;
            uint32_t h = padded_h >> 1;
            uint32_t bh = padded_h & 1;
            uint32_t out_w0 = chunk * max_copy_width;
            if (out_w0 >= outWidth_) {
                continue;
            }
            uint32_t cur_out = outWidth_ - out_w0;
            if (cur_out > max_copy_width) {
                cur_out = max_copy_width;
            }

            AscendC::LocalTensor<DT_X> local = copyQueue_.AllocTensor<DT_X>();
            for (uint32_t rel = 0; rel < cur_out; ++rel) {
                uint32_t ow = out_w0 + rel;
                uint32_t padded_w = ow + cropLeft_;
                uint32_t iw = padded_w >> 1;
                uint32_t bw = padded_w & 1;
                uint32_t in_n = ((bh << 1) + bw) * outBatch_ + on;
                uint32_t in_base = ((in_n * height_ + h) * width_ + iw) * depth_;
                AscendC::DataCopyPad(local[rel * depth_], xGm_[in_base], copy_in, pad_params);
            }
            copyQueue_.EnQue(local);
            local = copyQueue_.DeQue<DT_X>();
            uint32_t out_base = ((on * outHeight_ + out_h) * outWidth_ + out_w0) * depth_;
            copy_out.blockLen = cur_out * depth_bytes;
            AscendC::DataCopyPad(yGm_[out_base], local, copy_out);
            copyQueue_.FreeTensor(local);
        }

    }

    __aicore__ inline void ProcessB2LocalInterleaveCopy() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        uint32_t depth_blocks = depth_ * sizeof(DT_X) / 32;
        uint32_t depth_bytes = depth_ * sizeof(DT_X);
        uint32_t max_copy_width = copyUbBytes_ / (3U * depth_bytes);
        if (max_copy_width == 0) {
            max_copy_width = 1;
        }
        if (max_copy_width > 1024) {
            max_copy_width = 1024;
        }
        uint32_t lane0_off = 0;
        uint32_t lane1_off = max_copy_width * depth_;
        uint32_t out_off = lane1_off + max_copy_width * depth_;

        AscendC::DataCopyParams copy_in;
        copy_in.blockCount = 1;
        copy_in.blockLen = 0;
        copy_in.srcStride = 0;
        copy_in.dstStride = 0;

        AscendC::DataCopyParams copy_out;
        copy_out.blockCount = 1;
        copy_out.blockLen = 0;
        copy_out.srcStride = 0;
        copy_out.dstStride = 0;

        uint32_t row_count = outBatch_ * outHeight_;
        uint32_t unit_count = row_count * rowGroupLength_;
        uint32_t start_unit = core_idx * corePixelStride_;
        uint32_t end_unit = start_unit + corePixelStride_;
        if (end_unit > unit_count) {
            end_unit = unit_count;
        }

        for (uint32_t unit = start_unit; unit < end_unit; ++unit) {
            uint32_t chunk = unit % rowGroupLength_;
            uint32_t row = unit / rowGroupLength_;
            uint32_t on = row / outHeight_;
            uint32_t out_h = row - on * outHeight_;
            uint32_t padded_h = out_h + cropTop_;
            uint32_t h = padded_h >> 1;
            uint32_t bh = padded_h & 1;
            uint32_t out_w0 = chunk * max_copy_width;
            if (out_w0 >= outWidth_) {
                continue;
            }
            uint32_t cur_out = outWidth_ - out_w0;
            if (cur_out > max_copy_width) {
                cur_out = max_copy_width;
            }

            AscendC::LocalTensor<DT_X> local = copyQueue_.AllocTensor<DT_X>();
            uint32_t padded_w0 = out_w0 + cropLeft_;
            uint32_t first_rel0 = (padded_w0 & 1U) == 0 ? 0 : 1;
            uint32_t first_rel1 = (padded_w0 & 1U) == 1 ? 0 : 1;
            uint32_t count0 = 0;
            uint32_t count1 = 0;
            if (first_rel0 < cur_out) {
                count0 = ((cur_out - 1 - first_rel0) >> 1) + 1;
                uint32_t iw0 = (padded_w0 + first_rel0) >> 1;
                uint32_t in_n0 = (bh << 1) * outBatch_ + on;
                uint32_t in_base0 = ((in_n0 * height_ + h) * width_ + iw0) * depth_;
                copy_in.blockLen = static_cast<uint16_t>(count0 * depth_blocks);
                AscendC::DataCopy(local[lane0_off], xGm_[in_base0], copy_in);
            }
            if (first_rel1 < cur_out) {
                count1 = ((cur_out - 1 - first_rel1) >> 1) + 1;
                uint32_t iw1 = (padded_w0 + first_rel1) >> 1;
                uint32_t in_n1 = ((bh << 1) + 1) * outBatch_ + on;
                uint32_t in_base1 = ((in_n1 * height_ + h) * width_ + iw1) * depth_;
                copy_in.blockLen = static_cast<uint16_t>(count1 * depth_blocks);
                AscendC::DataCopy(local[lane1_off], xGm_[in_base1], copy_in);
            }

            copyQueue_.EnQue(local);
            local = copyQueue_.DeQue<DT_X>();
            for (uint32_t rel = 0; rel < cur_out; ++rel) {
                uint32_t padded_w = padded_w0 + rel;
                uint32_t src_off;
                if ((padded_w & 1U) == 0) {
                    uint32_t lane_pos = rel >= first_rel0 ? ((rel - first_rel0) >> 1) : 0;
                    src_off = lane0_off + lane_pos * depth_;
                } else {
                    uint32_t lane_pos = rel >= first_rel1 ? ((rel - first_rel1) >> 1) : 0;
                    src_off = lane1_off + lane_pos * depth_;
                }
                uint32_t dst_off = out_off + rel * depth_;
                for (uint32_t c = 0; c < depth_; ++c) {
                    local.SetValue(dst_off + c, local.GetValue(src_off + c));
                }
            }

            uint32_t out_base = ((on * outHeight_ + out_h) * outWidth_ + out_w0) * depth_;
            copy_out.blockLen = static_cast<uint16_t>(cur_out * depth_blocks);
            AscendC::DataCopy(yGm_[out_base], local[out_off], copy_out);
            copyQueue_.FreeTensor(local);
        }

    }

    __aicore__ inline void ProcessB2GatherInterleaveCopy() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        uint32_t depth_blocks = depth_ * sizeof(DT_X) / 32;
        uint32_t depth_bytes = depth_ * sizeof(DT_X);
        uint32_t max_copy_width = 32768 / depth_bytes;
        uint32_t max_offset_width = 32768 / (depth_ * 4U);
        if (max_copy_width > max_offset_width) {
            max_copy_width = max_offset_width;
        }
        if (max_copy_width == 0) {
            max_copy_width = 1;
        }
        if (max_copy_width > 1024) {
            max_copy_width = 1024;
        }
        uint32_t lane1_off = ((max_copy_width + 1) >> 1) * depth_;

        AscendC::DataCopyParams copy_in;
        copy_in.blockCount = 1;
        copy_in.blockLen = 0;
        copy_in.srcStride = 0;
        copy_in.dstStride = 0;

        AscendC::DataCopyParams copy_out;
        copy_out.blockCount = 1;
        copy_out.blockLen = 0;
        copy_out.srcStride = 0;
        copy_out.dstStride = 0;

        uint32_t row_count = outBatch_ * outHeight_;
        uint32_t unit_count = row_count * rowGroupLength_;
        uint32_t start_unit = core_idx * corePixelStride_;
        uint32_t end_unit = start_unit + corePixelStride_;
        if (end_unit > unit_count) {
            end_unit = unit_count;
        }

        for (uint32_t unit = start_unit; unit < end_unit; ++unit) {
            uint32_t chunk = unit % rowGroupLength_;
            uint32_t row = unit / rowGroupLength_;
            uint32_t on = row / outHeight_;
            uint32_t out_h = row - on * outHeight_;
            uint32_t padded_h = out_h + cropTop_;
            uint32_t h = padded_h >> 1;
            uint32_t bh = padded_h & 1;
            uint32_t out_w0 = chunk * max_copy_width;
            if (out_w0 >= outWidth_) {
                continue;
            }
            uint32_t cur_out = outWidth_ - out_w0;
            if (cur_out > max_copy_width) {
                cur_out = max_copy_width;
            }

            AscendC::LocalTensor<DT_X> src = gatherInQueue_.AllocTensor<DT_X>();
            AscendC::LocalTensor<uint32_t> offsets = gatherInQueue_.AllocTensor<uint32_t>();
            AscendC::LocalTensor<DT_X> dst = gatherOutQueue_.AllocTensor<DT_X>();

            uint32_t padded_w0 = out_w0 + cropLeft_;
            uint32_t first_rel0 = (padded_w0 & 1U) == 0 ? 0 : 1;
            uint32_t first_rel1 = (padded_w0 & 1U) == 1 ? 0 : 1;
            if (first_rel0 < cur_out) {
                uint32_t count0 = ((cur_out - 1 - first_rel0) >> 1) + 1;
                uint32_t iw0 = (padded_w0 + first_rel0) >> 1;
                uint32_t in_n0 = (bh << 1) * outBatch_ + on;
                uint32_t in_base0 = ((in_n0 * height_ + h) * width_ + iw0) * depth_;
                copy_in.blockLen = static_cast<uint16_t>(count0 * depth_blocks);
                AscendC::DataCopy(src, xGm_[in_base0], copy_in);
            }
            if (first_rel1 < cur_out) {
                uint32_t count1 = ((cur_out - 1 - first_rel1) >> 1) + 1;
                uint32_t iw1 = (padded_w0 + first_rel1) >> 1;
                uint32_t in_n1 = ((bh << 1) + 1) * outBatch_ + on;
                uint32_t in_base1 = ((in_n1 * height_ + h) * width_ + iw1) * depth_;
                copy_in.blockLen = static_cast<uint16_t>(count1 * depth_blocks);
                AscendC::DataCopy(src[lane1_off], xGm_[in_base1], copy_in);
            }

            uint32_t elem_count = cur_out * depth_;
            for (uint32_t rel = 0; rel < cur_out; ++rel) {
                uint32_t padded_w = padded_w0 + rel;
                uint32_t lane_base;
                if ((padded_w & 1U) == 0) {
                    uint32_t lane_pos = rel >= first_rel0 ? ((rel - first_rel0) >> 1) : 0;
                    lane_base = lane_pos * depth_;
                } else {
                    uint32_t lane_pos = rel >= first_rel1 ? ((rel - first_rel1) >> 1) : 0;
                    lane_base = lane1_off + lane_pos * depth_;
                }
                uint32_t out_elem = rel * depth_;
                for (uint32_t c = 0; c < depth_; ++c) {
                    offsets.SetValue(out_elem + c, (lane_base + c) * sizeof(DT_X));
                }
            }

            gatherInQueue_.EnQue(src);
            gatherInQueue_.EnQue(offsets);
            src = gatherInQueue_.DeQue<DT_X>();
            offsets = gatherInQueue_.DeQue<uint32_t>();
            AscendC::Gather(dst, src, offsets, static_cast<uint32_t>(0), elem_count);
            gatherInQueue_.FreeTensor(src);
            gatherInQueue_.FreeTensor(offsets);

            gatherOutQueue_.EnQue(dst);
            dst = gatherOutQueue_.DeQue<DT_X>();
            uint32_t out_base = ((on * outHeight_ + out_h) * outWidth_ + out_w0) * depth_;
            copy_out.blockLen = static_cast<uint16_t>(cur_out * depth_blocks);
            AscendC::DataCopy(yGm_[out_base], dst, copy_out);
            gatherOutQueue_.FreeTensor(dst);
        }

    }

    __aicore__ inline void ProcessB2Case5InputPipelineChunks() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        uint32_t depth_bytes = depth_ * sizeof(DT_X);
        uint32_t chunk_width = 32U;
        uint32_t lane_capacity = 16U;
        uint32_t lane1_off_bytes = ((lane_capacity * depth_bytes + 31U) >> 5) << 5;
        uint32_t lane1_off = lane1_off_bytes / sizeof(DT_X);

        AscendC::DataCopyExtParams copy_in;
        copy_in.blockCount = 1U;
        copy_in.srcStride = 0;
        copy_in.dstStride = 0;
        copy_in.rsv = 0;

        AscendC::DataCopyExtParams copy_out;
        copy_out.blockCount = 1U;
        copy_out.srcStride = 0;
        copy_out.dstStride = 0;
        copy_out.rsv = 0;

        AscendC::DataCopyPadExtParams<DT_X> pad_params;
        pad_params.isPad = false;
        pad_params.leftPadding = 0;
        pad_params.rightPadding = 0;
        pad_params.paddingValue = 0;

        AscendC::LocalTensor<uint32_t> offsets = gatherInQueue_.AllocTensor<uint32_t>();
        for (uint32_t rel = 0; rel < chunk_width; ++rel) {
            uint32_t lane_pos = rel >> 1;
            uint32_t lane_base = (rel & 1U) == 0U ? (lane1_off + lane_pos * depth_) :
                (lane_pos * depth_);
            uint32_t out_elem = rel * depth_;
            for (uint32_t c = 0; c < depth_; ++c) {
                offsets.SetValue(out_elem + c, (lane_base + c) * sizeof(DT_X));
            }
        }
        gatherInQueue_.EnQue(offsets);
        offsets = gatherInQueue_.DeQue<uint32_t>();

        bool has_pending = false;
        uint32_t pending_cur_out = 0;
        uint32_t pending_out_base = 0;

        uint32_t unit_count = outHeight_ * rowGroupLength_;
        uint32_t start_unit = core_idx * corePixelStride_;
        uint32_t end_unit = start_unit + corePixelStride_;
        if (end_unit > unit_count) {
            end_unit = unit_count;
        }

        for (uint32_t unit = start_unit; unit < end_unit; ++unit) {
            uint32_t chunk = unit % rowGroupLength_;
            uint32_t out_h = unit / rowGroupLength_;
            uint32_t out_w0 = chunk * chunk_width;
            if (out_w0 >= outWidth_) {
                continue;
            }
            uint32_t cur_out = outWidth_ - out_w0;
            if (cur_out > chunk_width) {
                cur_out = chunk_width;
            }

            uint32_t padded_h = out_h + 1U;
            uint32_t h = padded_h >> 1;
            uint32_t bh = padded_h & 1U;
            uint32_t padded_w0 = out_w0 + 1U;

            AscendC::LocalTensor<DT_X> src = gatherInQueue_.AllocTensor<DT_X>();

            uint32_t count1 = (cur_out + 1U) >> 1;
            if (count1 != 0U) {
                uint32_t in_n1 = (bh << 1) + 1U;
                uint32_t input_w1 = padded_w0 >> 1;
                uint32_t in_base1 = ((in_n1 * height_ + h) * width_ + input_w1) * depth_;
                copy_in.blockLen = count1 * depth_bytes;
                AscendC::DataCopyPad(src[lane1_off], xGm_[in_base1], copy_in, pad_params);
            }

            uint32_t count0 = cur_out >> 1;
            if (count0 != 0U) {
                uint32_t in_n0 = bh << 1;
                uint32_t input_w0 = (padded_w0 + 1U) >> 1;
                uint32_t in_base0 = ((in_n0 * height_ + h) * width_ + input_w0) * depth_;
                copy_in.blockLen = count0 * depth_bytes;
                AscendC::DataCopyPad(src, xGm_[in_base0], copy_in, pad_params);
            }

            gatherInQueue_.EnQue(src);

            if (has_pending) {
                AscendC::LocalTensor<DT_X> prev = gatherInQueue_.DeQue<DT_X>();
                AscendC::LocalTensor<DT_X> dst = gatherOutQueue_.AllocTensor<DT_X>();
                uint32_t elem_count = pending_cur_out * depth_;
                AscendC::Gather(dst, prev, offsets, static_cast<uint32_t>(0), elem_count);
                gatherInQueue_.FreeTensor(prev);

                gatherOutQueue_.EnQue(dst);
                dst = gatherOutQueue_.DeQue<DT_X>();
                copy_out.blockLen = pending_cur_out * depth_bytes;
                AscendC::DataCopyPad(yGm_[pending_out_base], dst, copy_out);
                gatherOutQueue_.FreeTensor(dst);
            }

            pending_cur_out = cur_out;
            pending_out_base = (out_h * outWidth_ + out_w0) * depth_;
            has_pending = true;
        }

        if (has_pending) {
            AscendC::LocalTensor<DT_X> prev = gatherInQueue_.DeQue<DT_X>();
            AscendC::LocalTensor<DT_X> dst = gatherOutQueue_.AllocTensor<DT_X>();
            uint32_t elem_count = pending_cur_out * depth_;
            AscendC::Gather(dst, prev, offsets, static_cast<uint32_t>(0), elem_count);
            gatherInQueue_.FreeTensor(prev);

            gatherOutQueue_.EnQue(dst);
            dst = gatherOutQueue_.DeQue<DT_X>();
            copy_out.blockLen = pending_cur_out * depth_bytes;
            AscendC::DataCopyPad(yGm_[pending_out_base], dst, copy_out);
            gatherOutQueue_.FreeTensor(dst);
        }

        gatherInQueue_.FreeTensor(offsets);
    }

    __aicore__ inline void ProcessB2Case5InputPipelineNoFlushChunks() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        uint32_t depth_bytes = depth_ * sizeof(DT_X);
        uint32_t chunk_width = 32U;
        uint32_t lane_capacity = 16U;
        uint32_t lane1_off_bytes = ((lane_capacity * depth_bytes + 31U) >> 5) << 5;
        uint32_t lane1_off = lane1_off_bytes / sizeof(DT_X);

        AscendC::DataCopyExtParams copy_in;
        copy_in.blockCount = 1U;
        copy_in.srcStride = 0;
        copy_in.dstStride = 0;
        copy_in.rsv = 0;

        AscendC::DataCopyExtParams copy_out;
        copy_out.blockCount = 1U;
        copy_out.srcStride = 0;
        copy_out.dstStride = 0;
        copy_out.rsv = 0;

        AscendC::DataCopyPadExtParams<DT_X> pad_params;
        pad_params.isPad = false;
        pad_params.leftPadding = 0;
        pad_params.rightPadding = 0;
        pad_params.paddingValue = 0;

        AscendC::LocalTensor<uint32_t> offsets = gatherInQueue_.AllocTensor<uint32_t>();
        for (uint32_t rel = 0; rel < chunk_width; ++rel) {
            uint32_t lane_pos = rel >> 1;
            uint32_t lane_base = (rel & 1U) == 0U ? (lane1_off + lane_pos * depth_) :
                (lane_pos * depth_);
            uint32_t out_elem = rel * depth_;
            for (uint32_t c = 0; c < depth_; ++c) {
                offsets.SetValue(out_elem + c, (lane_base + c) * sizeof(DT_X));
            }
        }
        gatherInQueue_.EnQue(offsets);
        offsets = gatherInQueue_.DeQue<uint32_t>();

        bool has_pending = false;
        uint32_t pending_cur_out = 0;
        uint32_t pending_out_base = 0;

        uint32_t unit_count = outHeight_ * rowGroupLength_;
        uint32_t start_unit = core_idx * corePixelStride_;
        uint32_t end_unit = start_unit + corePixelStride_;
        if (end_unit > unit_count) {
            end_unit = unit_count;
        }

        for (uint32_t unit = start_unit; unit < end_unit; ++unit) {
            uint32_t chunk = unit % rowGroupLength_;
            uint32_t out_h = unit / rowGroupLength_;
            uint32_t out_w0 = chunk * chunk_width;
            if (out_w0 >= outWidth_) {
                continue;
            }
            uint32_t cur_out = outWidth_ - out_w0;
            if (cur_out > chunk_width) {
                cur_out = chunk_width;
            }

            uint32_t padded_h = out_h + 1U;
            uint32_t h = padded_h >> 1;
            uint32_t bh = padded_h & 1U;
            uint32_t padded_w0 = out_w0 + 1U;

            AscendC::LocalTensor<DT_X> src = gatherInQueue_.AllocTensor<DT_X>();

            uint32_t count1 = (cur_out + 1U) >> 1;
            if (count1 != 0U) {
                uint32_t in_n1 = (bh << 1) + 1U;
                uint32_t input_w1 = padded_w0 >> 1;
                uint32_t in_base1 = ((in_n1 * height_ + h) * width_ + input_w1) * depth_;
                copy_in.blockLen = count1 * depth_bytes;
                AscendC::DataCopyPad(src[lane1_off], xGm_[in_base1], copy_in, pad_params);
            }

            uint32_t count0 = cur_out >> 1;
            if (count0 != 0U) {
                uint32_t in_n0 = bh << 1;
                uint32_t input_w0 = (padded_w0 + 1U) >> 1;
                uint32_t in_base0 = ((in_n0 * height_ + h) * width_ + input_w0) * depth_;
                copy_in.blockLen = count0 * depth_bytes;
                AscendC::DataCopyPad(src, xGm_[in_base0], copy_in, pad_params);
            }

            gatherInQueue_.EnQue(src);

            if (has_pending) {
                AscendC::LocalTensor<DT_X> prev = gatherInQueue_.DeQue<DT_X>();
                AscendC::LocalTensor<DT_X> dst = gatherOutQueue_.AllocTensor<DT_X>();
                uint32_t elem_count = pending_cur_out * depth_;
                AscendC::Gather(dst, prev, offsets, static_cast<uint32_t>(0), elem_count);
                gatherInQueue_.FreeTensor(prev);

                gatherOutQueue_.EnQue(dst);
                dst = gatherOutQueue_.DeQue<DT_X>();
                copy_out.blockLen = pending_cur_out * depth_bytes;
                AscendC::DataCopyPad(yGm_[pending_out_base], dst, copy_out);
                gatherOutQueue_.FreeTensor(dst);
            }

            pending_cur_out = cur_out;
            pending_out_base = (out_h * outWidth_ + out_w0) * depth_;
            has_pending = true;
        }

        if (has_pending) {
            AscendC::LocalTensor<DT_X> prev = gatherInQueue_.DeQue<DT_X>();
            AscendC::LocalTensor<DT_X> dst = gatherOutQueue_.AllocTensor<DT_X>();
            uint32_t elem_count = pending_cur_out * depth_;
            AscendC::Gather(dst, prev, offsets, static_cast<uint32_t>(0), elem_count);
            gatherInQueue_.FreeTensor(prev);

            gatherOutQueue_.EnQue(dst);
            dst = gatherOutQueue_.DeQue<DT_X>();
            copy_out.blockLen = pending_cur_out * depth_bytes;
            AscendC::DataCopyPad(yGm_[pending_out_base], dst, copy_out);
            gatherOutQueue_.FreeTensor(dst);
        }

        gatherInQueue_.FreeTensor(offsets);
    }

    __aicore__ inline void ProcessB2Case5Exact65NoFlushChunks() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        constexpr uint32_t kDepth = 65U;
        constexpr uint32_t kDepthBytes = 130U;
        constexpr uint32_t kChunkWidth = 32U;
        constexpr uint32_t kLanePixels = 16U;
        constexpr uint32_t kLaneElems = kLanePixels * kDepth;
        constexpr uint32_t kFullInBytes = kLanePixels * kDepthBytes;
        constexpr uint32_t kTailInBytes = 15U * kDepthBytes;
        constexpr uint32_t kFullOutBytes = kChunkWidth * kDepthBytes;
        constexpr uint32_t kTailOutBytes = 30U * kDepthBytes;

        AscendC::DataCopyExtParams copy_in;
        copy_in.blockCount = 1U;
        copy_in.srcStride = 0;
        copy_in.dstStride = 0;
        copy_in.rsv = 0;

        AscendC::DataCopyExtParams copy_out;
        copy_out.blockCount = 1U;
        copy_out.srcStride = 0;
        copy_out.dstStride = 0;
        copy_out.rsv = 0;

        AscendC::DataCopyPadExtParams<DT_X> pad_params;
        pad_params.isPad = false;
        pad_params.leftPadding = 0;
        pad_params.rightPadding = 0;
        pad_params.paddingValue = 0;

        AscendC::LocalTensor<uint32_t> offsets = gatherInQueue_.AllocTensor<uint32_t>();
        for (uint32_t rel = 0; rel < kChunkWidth; ++rel) {
            uint32_t lane_pos = rel >> 1;
            uint32_t lane_base = (rel & 1U) == 0U ? (kLaneElems + lane_pos * kDepth) :
                (lane_pos * kDepth);
            uint32_t out_elem = rel * kDepth;
            for (uint32_t c = 0; c < kDepth; ++c) {
                offsets.SetValue(out_elem + c, (lane_base + c) * sizeof(DT_X));
            }
        }
        gatherInQueue_.EnQue(offsets);
        offsets = gatherInQueue_.DeQue<uint32_t>();

        bool has_pending = false;
        uint32_t pending_is_tail = 0U;
        uint32_t pending_out_base = 0U;

        uint32_t start_unit = core_idx * corePixelStride_;
        uint32_t end_unit = start_unit + corePixelStride_;
        constexpr uint32_t kUnitCount = 254U * 8U;
        if (end_unit > kUnitCount) {
            end_unit = kUnitCount;
        }

        for (uint32_t unit = start_unit; unit < end_unit; ++unit) {
            uint32_t chunk = unit & 7U;
            uint32_t out_h = unit >> 3;
            uint32_t padded_h = out_h + 1U;
            uint32_t h = padded_h >> 1;
            uint32_t bh = padded_h & 1U;
            uint32_t input_w1 = chunk << 4;
            uint32_t input_w0 = input_w1 + 1U;
            uint32_t is_tail = chunk == 7U ? 1U : 0U;

            AscendC::LocalTensor<DT_X> src = gatherInQueue_.AllocTensor<DT_X>();

            uint32_t in_n1 = (bh << 1) + 1U;
            uint32_t input_index1 = (in_n1 << 14) + (h << 7) + input_w1;
            uint32_t in_base1 = (input_index1 << 6) + input_index1;
            copy_in.blockLen = is_tail == 0U ? kFullInBytes : kTailInBytes;
            AscendC::DataCopyPad(src[kLaneElems], xGm_[in_base1], copy_in, pad_params);

            uint32_t in_n0 = bh << 1;
            uint32_t input_index0 = (in_n0 << 14) + (h << 7) + input_w0;
            uint32_t in_base0 = (input_index0 << 6) + input_index0;
            AscendC::DataCopyPad(src, xGm_[in_base0], copy_in, pad_params);

            gatherInQueue_.EnQue(src);

            if (has_pending) {
                AscendC::LocalTensor<DT_X> prev = gatherInQueue_.DeQue<DT_X>();
                AscendC::LocalTensor<DT_X> dst = gatherOutQueue_.AllocTensor<DT_X>();
                uint32_t elem_count = pending_is_tail == 0U ? (kChunkWidth * kDepth) : (30U * kDepth);
                AscendC::Gather(dst, prev, offsets, static_cast<uint32_t>(0), elem_count);
                gatherInQueue_.FreeTensor(prev);

                gatherOutQueue_.EnQue(dst);
                dst = gatherOutQueue_.DeQue<DT_X>();
                copy_out.blockLen = pending_is_tail == 0U ? kFullOutBytes : kTailOutBytes;
                AscendC::DataCopyPad(yGm_[pending_out_base], dst, copy_out);
                gatherOutQueue_.FreeTensor(dst);
            }

            pending_is_tail = is_tail;
            pending_out_base = ((out_h * 254U) + (chunk << 5)) * kDepth;
            has_pending = true;
        }

        if (has_pending) {
            AscendC::LocalTensor<DT_X> prev = gatherInQueue_.DeQue<DT_X>();
            AscendC::LocalTensor<DT_X> dst = gatherOutQueue_.AllocTensor<DT_X>();
            uint32_t elem_count = pending_is_tail == 0U ? (kChunkWidth * kDepth) : (30U * kDepth);
            AscendC::Gather(dst, prev, offsets, static_cast<uint32_t>(0), elem_count);
            gatherInQueue_.FreeTensor(prev);

            gatherOutQueue_.EnQue(dst);
            dst = gatherOutQueue_.DeQue<DT_X>();
            copy_out.blockLen = pending_is_tail == 0U ? kFullOutBytes : kTailOutBytes;
            AscendC::DataCopyPad(yGm_[pending_out_base], dst, copy_out);
            gatherOutQueue_.FreeTensor(dst);
        }

        gatherInQueue_.FreeTensor(offsets);
    }

    __aicore__ inline void ProcessB2Case5Gather32SmallUbChunks() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        uint32_t depth_bytes = depth_ * sizeof(DT_X);
        uint32_t chunk_width = 32U;
        uint32_t lane_capacity = 16U;
        uint32_t lane1_off_bytes = ((lane_capacity * depth_bytes + 31U) >> 5) << 5;
        uint32_t lane1_off = lane1_off_bytes / sizeof(DT_X);

        AscendC::DataCopyExtParams copy_in;
        copy_in.blockCount = 1U;
        copy_in.srcStride = 0;
        copy_in.dstStride = 0;
        copy_in.rsv = 0;

        AscendC::DataCopyExtParams copy_out;
        copy_out.blockCount = 1U;
        copy_out.srcStride = 0;
        copy_out.dstStride = 0;
        copy_out.rsv = 0;

        AscendC::DataCopyPadExtParams<DT_X> pad_params;
        pad_params.isPad = false;
        pad_params.leftPadding = 0;
        pad_params.rightPadding = 0;
        pad_params.paddingValue = 0;

        AscendC::LocalTensor<uint32_t> offsets = gatherInQueue_.AllocTensor<uint32_t>();
        for (uint32_t rel = 0; rel < chunk_width; ++rel) {
            uint32_t lane_pos = rel >> 1;
            uint32_t lane_base = (rel & 1U) == 0U ? (lane1_off + lane_pos * depth_) :
                (lane_pos * depth_);
            uint32_t out_elem = rel * depth_;
            for (uint32_t c = 0; c < depth_; ++c) {
                offsets.SetValue(out_elem + c, (lane_base + c) * sizeof(DT_X));
            }
        }
        gatherInQueue_.EnQue(offsets);
        offsets = gatherInQueue_.DeQue<uint32_t>();

        uint32_t unit_count = outHeight_ * rowGroupLength_;
        uint32_t start_unit = core_idx * corePixelStride_;
        uint32_t end_unit = start_unit + corePixelStride_;
        if (end_unit > unit_count) {
            end_unit = unit_count;
        }

        for (uint32_t unit = start_unit; unit < end_unit; ++unit) {
            uint32_t chunk = unit % rowGroupLength_;
            uint32_t out_h = unit / rowGroupLength_;
            uint32_t out_w0 = chunk * chunk_width;
            if (out_w0 >= outWidth_) {
                continue;
            }
            uint32_t cur_out = outWidth_ - out_w0;
            if (cur_out > chunk_width) {
                cur_out = chunk_width;
            }

            uint32_t padded_h = out_h + 1U;
            uint32_t h = padded_h >> 1;
            uint32_t bh = padded_h & 1U;
            uint32_t padded_w0 = out_w0 + 1U;

            AscendC::LocalTensor<DT_X> src = gatherInQueue_.AllocTensor<DT_X>();
            AscendC::LocalTensor<DT_X> dst = gatherOutQueue_.AllocTensor<DT_X>();

            uint32_t count1 = (cur_out + 1U) >> 1;
            if (count1 != 0U) {
                uint32_t in_n1 = (bh << 1) + 1U;
                uint32_t input_w1 = padded_w0 >> 1;
                uint32_t in_base1 = ((in_n1 * height_ + h) * width_ + input_w1) * depth_;
                copy_in.blockLen = count1 * depth_bytes;
                AscendC::DataCopyPad(src[lane1_off], xGm_[in_base1], copy_in, pad_params);
            }

            uint32_t count0 = cur_out >> 1;
            if (count0 != 0U) {
                uint32_t in_n0 = bh << 1;
                uint32_t input_w0 = (padded_w0 + 1U) >> 1;
                uint32_t in_base0 = ((in_n0 * height_ + h) * width_ + input_w0) * depth_;
                copy_in.blockLen = count0 * depth_bytes;
                AscendC::DataCopyPad(src, xGm_[in_base0], copy_in, pad_params);
            }

            gatherInQueue_.EnQue(src);
            src = gatherInQueue_.DeQue<DT_X>();
            uint32_t elem_count = cur_out * depth_;
            AscendC::Gather(dst, src, offsets, static_cast<uint32_t>(0), elem_count);
            gatherInQueue_.FreeTensor(src);

            gatherOutQueue_.EnQue(dst);
            dst = gatherOutQueue_.DeQue<DT_X>();
            uint32_t out_base = (out_h * outWidth_ + out_w0) * depth_;
            copy_out.blockLen = cur_out * depth_bytes;
            AscendC::DataCopyPad(yGm_[out_base], dst, copy_out);
            gatherOutQueue_.FreeTensor(dst);
        }

        gatherInQueue_.FreeTensor(offsets);
    }

    __aicore__ inline void ProcessB2RowtileAssembleCopy() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        uint32_t depth_blocks = depth_ * sizeof(DT_X) / 32U;
        uint32_t rows_per_group = rowGroupLength_ == 0 ? 1U : rowGroupLength_;
        uint32_t tile_width = depthSplit_ == 0 ? 1U : depthSplit_;
        uint32_t groups_per_batch = (outHeight_ + rows_per_group - 1U) / rows_per_group;
        uint32_t chunks_per_row = (outWidth_ + tile_width - 1U) / tile_width;

        AscendC::DataCopyParams copy_in;
        copy_in.blockCount = 1;
        copy_in.blockLen = static_cast<uint16_t>(depth_blocks);
        copy_in.srcStride = 0;
        copy_in.dstStride = static_cast<uint16_t>(depth_blocks);

        AscendC::DataCopyParams copy_out;
        copy_out.blockCount = 1;
        copy_out.srcStride = 0;
        copy_out.dstStride = 0;

        uint32_t unit_count = outBatch_ * groups_per_batch * chunks_per_row;
        uint32_t start_unit = core_idx * corePixelStride_;
        uint32_t end_unit = start_unit + corePixelStride_;
        if (end_unit > unit_count) {
            end_unit = unit_count;
        }

        for (uint32_t unit = start_unit; unit < end_unit; ++unit) {
            uint32_t tmp = unit;
            uint32_t chunk = tmp % chunks_per_row;
            tmp = tmp / chunks_per_row;
            uint32_t on = tmp / groups_per_batch;
            uint32_t group = tmp - on * groups_per_batch;
            uint32_t out_h0 = group * rows_per_group;
            uint32_t out_w0 = chunk * tile_width;
            if (out_h0 >= outHeight_ || out_w0 >= outWidth_) {
                continue;
            }

            uint32_t cur_rows = outHeight_ - out_h0;
            if (cur_rows > rows_per_group) {
                cur_rows = rows_per_group;
            }
            uint32_t cur_out = outWidth_ - out_w0;
            if (cur_out > tile_width) {
                cur_out = tile_width;
            }

            AscendC::LocalTensor<DT_X> local = copyQueue_.AllocTensor<DT_X>();
            uint32_t row_elems = cur_out * depth_;
            uint32_t padded_w0 = out_w0 + cropLeft_;
            for (uint32_t rr = 0; rr < cur_rows; ++rr) {
                uint32_t out_h = out_h0 + rr;
                uint32_t padded_h = out_h + cropTop_;
                uint32_t h = padded_h >> 1;
                uint32_t bh = padded_h & 1U;
                uint32_t row_off = rr * row_elems;
                for (uint32_t bw = 0; bw < 2U; ++bw) {
                    uint32_t first_rel = (padded_w0 & 1U) == bw ? 0U : 1U;
                    if (first_rel >= cur_out) {
                        continue;
                    }
                    uint32_t count = ((cur_out - 1U - first_rel) >> 1) + 1U;
                    uint32_t input_w0 = (padded_w0 + first_rel) >> 1;
                    uint32_t in_n = ((bh << 1) + bw) * outBatch_ + on;
                    uint32_t in_base = ((in_n * height_ + h) * width_ + input_w0) * depth_;
                    copy_in.blockCount = static_cast<uint16_t>(count);
                    AscendC::DataCopy(local[row_off + first_rel * depth_], xGm_[in_base], copy_in);
                }
            }
            copyQueue_.EnQue(local);
            local = copyQueue_.DeQue<DT_X>();
            for (uint32_t rr = 0; rr < cur_rows; ++rr) {
                uint32_t out_base = ((on * outHeight_ + out_h0 + rr) * outWidth_ + out_w0) * depth_;
                copy_out.blockLen = static_cast<uint16_t>(cur_out * depth_blocks);
                AscendC::DataCopy(yGm_[out_base], local[rr * row_elems], copy_out);
            }
            copyQueue_.FreeTensor(local);
        }

    }

    __aicore__ inline void ProcessB2RowsAssembleCopy() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        uint32_t depth_blocks = depth_ * sizeof(DT_X) / 32;
        uint32_t row_blocks = outWidth_ * depth_blocks;
        uint32_t row_bytes = outWidth_ * depth_ * sizeof(DT_X);
        uint32_t rows_per_group = copyUbBytes_ / row_bytes;
        if (rows_per_group == 0) {
            rows_per_group = 1;
        }
        if (rows_per_group > 16) {
            rows_per_group = 16;
        }

        AscendC::DataCopyParams copy_in;
        copy_in.blockCount = 1;
        copy_in.blockLen = static_cast<uint16_t>(depth_blocks);
        copy_in.srcStride = 0;
        copy_in.dstStride = static_cast<uint16_t>(depth_blocks);

        AscendC::DataCopyParams copy_out;
        copy_out.blockCount = 1;
        copy_out.blockLen = 0;
        copy_out.srcStride = 0;
        copy_out.dstStride = 0;

        uint32_t groups_per_batch = (outHeight_ + rows_per_group - 1) / rows_per_group;
        uint32_t unit_count = outBatch_ * groups_per_batch;
        uint32_t start_unit = core_idx * corePixelStride_;
        uint32_t end_unit = start_unit + corePixelStride_;
        if (end_unit > unit_count) {
            end_unit = unit_count;
        }

        for (uint32_t unit = start_unit; unit < end_unit; ++unit) {
            uint32_t on = unit / groups_per_batch;
            uint32_t group = unit - on * groups_per_batch;
            uint32_t out_h0 = group * rows_per_group;
            if (out_h0 >= outHeight_) {
                continue;
            }
            uint32_t cur_rows = outHeight_ - out_h0;
            if (cur_rows > rows_per_group) {
                cur_rows = rows_per_group;
            }

            AscendC::LocalTensor<DT_X> local = copyQueue_.AllocTensor<DT_X>();
            for (uint32_t rr = 0; rr < cur_rows; ++rr) {
                uint32_t out_h = out_h0 + rr;
                uint32_t padded_h = out_h + cropTop_;
                uint32_t h = padded_h >> 1;
                uint32_t bh = padded_h & 1;
                uint32_t row_off = rr * outWidth_ * depth_;
                for (uint32_t bw = 0; bw < 2; ++bw) {
                    uint32_t start_ow = (cropLeft_ & 1U) == bw ? 0 : 1;
                    if (start_ow >= outWidth_) {
                        continue;
                    }
                    uint32_t count = ((outWidth_ - 1 - start_ow) >> 1) + 1;
                    uint32_t input_w0 = (start_ow + cropLeft_) >> 1;
                    uint32_t in_n = ((bh << 1) + bw) * outBatch_ + on;
                    uint32_t in_base = ((in_n * height_ + h) * width_ + input_w0) * depth_;
                    copy_in.blockCount = static_cast<uint16_t>(count);
                    AscendC::DataCopy(local[row_off + start_ow * depth_], xGm_[in_base], copy_in);
                }
            }
            copyQueue_.EnQue(local);
            local = copyQueue_.DeQue<DT_X>();
            uint32_t out_base = ((on * outHeight_ + out_h0) * outWidth_) * depth_;
            copy_out.blockLen = static_cast<uint16_t>(cur_rows * row_blocks);
            AscendC::DataCopy(yGm_[out_base], local, copy_out);
            copyQueue_.FreeTensor(local);
        }

    }

    __aicore__ inline void ProcessB2RowCopy() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        uint32_t depth_blocks = depth_ * sizeof(DT_X) / 32;
        uint32_t max_copy_width = copyUbBytes_ / (depth_ * sizeof(DT_X));
        if (max_copy_width > 4095) {
            max_copy_width = 4095;
        }

        AscendC::DataCopyParams copy_in;
        copy_in.blockLen = static_cast<uint16_t>(depth_blocks);
        copy_in.srcStride = 0;
        copy_in.dstStride = 0;

        AscendC::DataCopyParams copy_out;
        copy_out.blockLen = static_cast<uint16_t>(depth_blocks);
        copy_out.srcStride = 0;
        copy_out.dstStride = static_cast<uint16_t>(depth_blocks);

        uint32_t row_count = outBatch_ * outHeight_;
        uint32_t unit_count = row_count * 2 * rowGroupLength_;
        uint32_t start_unit = core_idx * corePixelStride_;
        uint32_t end_unit = start_unit + corePixelStride_;
        if (end_unit > unit_count) {
            end_unit = unit_count;
        }

        for (uint32_t unit = start_unit; unit < end_unit; ++unit) {
            uint32_t tmp = unit;
            uint32_t chunk = tmp % rowGroupLength_;
            tmp = tmp / rowGroupLength_;
            uint32_t bw = tmp & 1;
            uint32_t row = tmp >> 1;
            uint32_t on = row / outHeight_;
            uint32_t out_h = row - on * outHeight_;
            uint32_t padded_h = out_h + cropTop_;
            uint32_t h = padded_h >> 1;
            uint32_t bh = padded_h & 1;

            uint32_t start_ow = 0;
            if ((cropLeft_ & 1) != bw) {
                start_ow = 1;
            }
            if (start_ow >= outWidth_) {
                continue;
            }
            uint32_t copy_width = ((outWidth_ - 1 - start_ow) >> 1) + 1;
            uint32_t lane_offset = chunk * max_copy_width;
            if (lane_offset >= copy_width) {
                continue;
            }
            uint32_t cur = copy_width - lane_offset;
            if (cur > max_copy_width) {
                cur = max_copy_width;
            }
            uint32_t padded_w0 = start_ow + cropLeft_ + lane_offset * 2;
            uint32_t input_w0 = padded_w0 >> 1;
            uint32_t in_n = ((bh << 1) + bw) * outBatch_ + on;
            uint32_t in_row = ((in_n * height_ + h) * width_ + input_w0) * depth_;
            uint32_t out_row = ((on * outHeight_ + out_h) * outWidth_ + start_ow + lane_offset * 2) * depth_;

            copy_in.blockCount = static_cast<uint16_t>(cur);
            copy_out.blockCount = static_cast<uint16_t>(cur);
            AscendC::LocalTensor<DT_X> local = copyQueue_.AllocTensor<DT_X>();
            AscendC::DataCopy(local, xGm_[in_row], copy_in);
            copyQueue_.EnQue(local);
            local = copyQueue_.DeQue<DT_X>();
            AscendC::DataCopy(yGm_[out_row], local, copy_out);
            copyQueue_.FreeTensor(local);
        }

    }

    __aicore__ inline void ProcessB2DepthSplitRowCopy() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        uint32_t elems_per_block = 32U / sizeof(DT_X);
        uint32_t depth_blocks = depth_ * sizeof(DT_X) / 32U;
        uint32_t depth_chunks = depthSplit_ == 0 ? 1 : depthSplit_;
        uint32_t blocks_per_chunk = (depth_blocks + depth_chunks - 1) / depth_chunks;
        uint32_t max_copy_width = copyUbBytes_ / (blocks_per_chunk * 32U);
        if (max_copy_width == 0) {
            max_copy_width = 1;
        }
        if (max_copy_width > 4095) {
            max_copy_width = 4095;
        }

        AscendC::DataCopyParams copy_in;
        copy_in.srcStride = 0;
        copy_in.dstStride = 0;

        AscendC::DataCopyParams copy_out;
        copy_out.srcStride = 0;
        copy_out.dstStride = 0;

        uint32_t row_count = outBatch_ * outHeight_;
        uint32_t unit_count = row_count * 2 * rowGroupLength_;
        uint32_t start_unit = core_idx * corePixelStride_;
        uint32_t end_unit = start_unit + corePixelStride_;
        if (end_unit > unit_count) {
            end_unit = unit_count;
        }

        for (uint32_t unit = start_unit; unit < end_unit; ++unit) {
            uint32_t tmp = unit;
            uint32_t packed_chunk = tmp % rowGroupLength_;
            tmp = tmp / rowGroupLength_;
            uint32_t bw = tmp & 1U;
            uint32_t row = tmp >> 1;
            uint32_t depth_chunk = packed_chunk % depth_chunks;
            uint32_t width_chunk = packed_chunk / depth_chunks;
            uint32_t block_start = depth_chunk * blocks_per_chunk;
            if (block_start >= depth_blocks) {
                continue;
            }
            uint32_t cur_blocks = depth_blocks - block_start;
            if (cur_blocks > blocks_per_chunk) {
                cur_blocks = blocks_per_chunk;
            }

            uint32_t on = row / outHeight_;
            uint32_t out_h = row - on * outHeight_;
            uint32_t padded_h = out_h + cropTop_;
            uint32_t h = padded_h >> 1;
            uint32_t bh = padded_h & 1U;

            uint32_t start_ow = (cropLeft_ & 1U) == bw ? 0 : 1;
            if (start_ow >= outWidth_) {
                continue;
            }
            uint32_t copy_width = ((outWidth_ - 1 - start_ow) >> 1) + 1;
            uint32_t lane_offset = width_chunk * max_copy_width;
            if (lane_offset >= copy_width) {
                continue;
            }
            uint32_t cur = copy_width - lane_offset;
            if (cur > max_copy_width) {
                cur = max_copy_width;
            }

            uint32_t padded_w0 = start_ow + cropLeft_ + lane_offset * 2;
            uint32_t input_w0 = padded_w0 >> 1;
            uint32_t in_n = ((bh << 1) + bw) * outBatch_ + on;
            uint32_t channel_offset = block_start * elems_per_block;
            uint32_t in_row = ((in_n * height_ + h) * width_ + input_w0) * depth_ + channel_offset;
            uint32_t out_row = ((on * outHeight_ + out_h) * outWidth_ + start_ow +
                lane_offset * 2) * depth_ + channel_offset;

            copy_in.blockCount = static_cast<uint16_t>(cur);
            copy_in.blockLen = static_cast<uint16_t>(cur_blocks);
            copy_in.srcStride = static_cast<uint16_t>(depth_blocks - cur_blocks);
            copy_out.blockCount = static_cast<uint16_t>(cur);
            copy_out.blockLen = static_cast<uint16_t>(cur_blocks);
            copy_out.dstStride = static_cast<uint16_t>(2U * depth_blocks - cur_blocks);

            AscendC::LocalTensor<DT_X> local = copyQueue_.AllocTensor<DT_X>();
            AscendC::DataCopy(local, xGm_[in_row], copy_in);
            copyQueue_.EnQue(local);
            local = copyQueue_.DeQue<DT_X>();
            AscendC::DataCopy(yGm_[out_row], local, copy_out);
            copyQueue_.FreeTensor(local);
        }

    }

    __aicore__ inline void ProcessBnAssembleRowCopy() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        uint32_t depth_blocks = depth_ * sizeof(DT_X) / 32U;

        AscendC::DataCopyParams copy_in;
        copy_in.blockLen = static_cast<uint16_t>(depth_blocks);
        copy_in.srcStride = 0;
        copy_in.dstStride = static_cast<uint16_t>((blockSize_ - 1U) * depth_blocks);

        uint32_t max_copy_width = copyUbBytes_ / (depth_ * sizeof(DT_X));
        if (max_copy_width > 4095U) {
            max_copy_width = 4095U;
        }

        AscendC::DataCopyParams copy_out;
        copy_out.blockCount = 1;
        copy_out.blockLen = 0;
        copy_out.srcStride = 0;
        copy_out.dstStride = 0;

        uint32_t row_count = outBatch_ * outHeight_;
        uint32_t unit_count = row_count * rowGroupLength_;
        uint32_t start_unit = core_idx * corePixelStride_;
        uint32_t end_unit = start_unit + corePixelStride_;
        if (end_unit > unit_count) {
            end_unit = unit_count;
        }

        for (uint32_t unit = start_unit; unit < end_unit; ++unit) {
            uint32_t chunk = unit % rowGroupLength_;
            uint32_t row = unit / rowGroupLength_;
            uint32_t on = row / outHeight_;
            uint32_t out_h = row - on * outHeight_;
            uint32_t padded_h = out_h + cropTop_;
            uint32_t ih = padded_h / blockSize_;
            uint32_t bh = padded_h - ih * blockSize_;
            uint32_t out_w0 = chunk * max_copy_width;
            if (out_w0 >= outWidth_) {
                continue;
            }
            uint32_t cur_out = outWidth_ - out_w0;
            if (cur_out > max_copy_width) {
                cur_out = max_copy_width;
            }

            AscendC::LocalTensor<DT_X> local = copyQueue_.AllocTensor<DT_X>();
            uint32_t padded_w_base = out_w0 + cropLeft_;
            uint32_t crop_mod = padded_w_base % blockSize_;
            for (uint32_t bw = 0; bw < blockSize_; ++bw) {
                uint32_t start_ow = bw >= crop_mod ? (bw - crop_mod) : (bw + blockSize_ - crop_mod);
                if (start_ow >= cur_out) {
                    continue;
                }
                uint32_t count = ((cur_out - 1U - start_ow) / blockSize_) + 1U;
                uint32_t padded_w0 = out_w0 + start_ow + cropLeft_;
                uint32_t input_w0 = padded_w0 / blockSize_;
                uint32_t in_n = (bh * blockSize_ + bw) * outBatch_ + on;
                uint32_t in_row = ((in_n * height_ + ih) * width_ + input_w0) * depth_;
                copy_in.blockCount = static_cast<uint16_t>(count);
                AscendC::DataCopy(local[start_ow * depth_], xGm_[in_row], copy_in);
            }

            copyQueue_.EnQue(local);
            local = copyQueue_.DeQue<DT_X>();
            uint32_t out_base = (row * outWidth_ + out_w0) * depth_;
            copy_out.blockLen = static_cast<uint16_t>(cur_out * depth_blocks);
            AscendC::DataCopy(yGm_[out_base], local, copy_out);
            copyQueue_.FreeTensor(local);
        }
    }

    __aicore__ inline void ProcessBnRowCopy() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        uint32_t depth_blocks = depth_ * sizeof(DT_X) / 32;
        uint32_t max_copy_width = copyUbBytes_ / (depth_ * sizeof(DT_X));
        if (max_copy_width > 4095) {
            max_copy_width = 4095;
        }

        AscendC::DataCopyParams copy_in;
        copy_in.blockLen = static_cast<uint16_t>(depth_blocks);
        copy_in.srcStride = 0;
        copy_in.dstStride = 0;

        AscendC::DataCopyParams copy_out;
        copy_out.blockLen = static_cast<uint16_t>(depth_blocks);
        copy_out.srcStride = 0;
        copy_out.dstStride = static_cast<uint16_t>((blockSize_ - 1) * depth_blocks);

        uint32_t row_count = outBatch_ * outHeight_;
        uint32_t unit_count = row_count * blockSize_ * rowGroupLength_;
        uint32_t start_unit = core_idx * corePixelStride_;
        uint32_t end_unit = start_unit + corePixelStride_;
        if (end_unit > unit_count) {
            end_unit = unit_count;
        }

        for (uint32_t unit = start_unit; unit < end_unit; ++unit) {
            uint32_t tmp = unit;
            uint32_t chunk = tmp % rowGroupLength_;
            tmp = tmp / rowGroupLength_;
            uint32_t bw = tmp % blockSize_;
            uint32_t row = tmp / blockSize_;
            uint32_t on = row / outHeight_;
            uint32_t out_h = row - on * outHeight_;
            uint32_t padded_h = out_h + cropTop_;
            uint32_t ih = padded_h / blockSize_;
            uint32_t bh = padded_h - ih * blockSize_;

            uint32_t crop_mod = cropLeft_ % blockSize_;
            uint32_t start_ow = bw >= crop_mod ? (bw - crop_mod) : (bw + blockSize_ - crop_mod);
            if (start_ow >= outWidth_) {
                continue;
            }
            uint32_t copy_width = ((outWidth_ - 1 - start_ow) / blockSize_) + 1;
            uint32_t lane_offset = chunk * max_copy_width;
            if (lane_offset >= copy_width) {
                continue;
            }
            uint32_t cur = copy_width - lane_offset;
            if (cur > max_copy_width) {
                cur = max_copy_width;
            }

            uint32_t padded_w0 = start_ow + cropLeft_ + lane_offset * blockSize_;
            uint32_t input_w0 = padded_w0 / blockSize_;
            uint32_t in_n = (bh * blockSize_ + bw) * outBatch_ + on;
            uint32_t in_row = ((in_n * height_ + ih) * width_ + input_w0) * depth_;
            uint32_t out_row = ((on * outHeight_ + out_h) * outWidth_ + start_ow +
                lane_offset * blockSize_) * depth_;

            copy_in.blockCount = static_cast<uint16_t>(cur);
            copy_out.blockCount = static_cast<uint16_t>(cur);
            AscendC::LocalTensor<DT_X> local = copyQueue_.AllocTensor<DT_X>();
            AscendC::DataCopy(local, xGm_[in_row], copy_in);
            copyQueue_.EnQue(local);
            local = copyQueue_.DeQue<DT_X>();
            AscendC::DataCopy(yGm_[out_row], local, copy_out);
            copyQueue_.FreeTensor(local);
        }

    }

    __aicore__ inline void ProcessLargeDepthCopy() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        uint32_t max_copy_elems = copyUbBytes_ / sizeof(DT_X);
        uint32_t depth_chunks = rowGroupLength_;
        uint32_t unit_count = outBatch_ * outHeight_ * outWidth_ * depth_chunks;
        uint32_t start_unit = core_idx * corePixelStride_;
        uint32_t end_unit = start_unit + corePixelStride_;
        if (end_unit > unit_count) {
            end_unit = unit_count;
        }

        AscendC::DataCopyParams copy_params;
        copy_params.blockCount = 1;
        copy_params.srcStride = 0;
        copy_params.dstStride = 0;

        for (uint32_t unit = start_unit; unit < end_unit; ++unit) {
            uint32_t chunk = unit % depth_chunks;
            uint32_t pixel = unit / depth_chunks;
            uint32_t c_start = chunk * max_copy_elems;
            if (c_start >= depth_) {
                continue;
            }
            uint32_t cur = depth_ - c_start;
            if (cur > max_copy_elems) {
                cur = max_copy_elems;
            }
            uint32_t copy_bytes = cur * sizeof(DT_X);

            uint32_t input_base = GetInputBaseByPixel(pixel) + c_start;
            uint32_t output_base = pixel * depth_ + c_start;
            if ((copy_bytes & 31U) == 0) {
                copy_params.blockLen = static_cast<uint16_t>(copy_bytes / 32U);
                AscendC::LocalTensor<DT_X> local = copyQueue_.AllocTensor<DT_X>();
                AscendC::DataCopy(local, xGm_[input_base], copy_params);
                copyQueue_.EnQue(local);
                local = copyQueue_.DeQue<DT_X>();
                AscendC::DataCopy(yGm_[output_base], local, copy_params);
                copyQueue_.FreeTensor(local);
            } else {
                for (uint32_t i = 0; i < cur; ++i) {
                    yGm_.SetValue(output_base + i, xGm_.GetValue(input_base + i));
                }
            }
        }

        AscendC::DataCacheCleanAndInvalid<DT_X, AscendC::CacheLine::ENTIRE_DATA_CACHE>(yGm_);
    }

    __aicore__ inline uint32_t GetPadMaxCopyWidth() {
        uint32_t depth_bytes = depth_ * sizeof(DT_X);
        uint32_t padded_bytes = ((depth_bytes + 31) >> 5) << 5;
        uint32_t max_copy_width = copyUbBytes_ / padded_bytes;
        if (max_copy_width > 4095) {
            max_copy_width = 4095;
        }
        return max_copy_width;
    }

    __aicore__ inline void ProcessB2PadCopy() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        uint32_t depth_bytes = depth_ * sizeof(DT_X);
        uint32_t max_copy_width = GetPadMaxCopyWidth();

        AscendC::DataCopyExtParams copy_in;
        copy_in.blockLen = depth_bytes;
        copy_in.srcStride = 0;
        copy_in.dstStride = 0;
        copy_in.rsv = 0;

        AscendC::DataCopyExtParams copy_out;
        copy_out.blockLen = depth_bytes;
        copy_out.srcStride = 0;
        copy_out.dstStride = depth_bytes;
        copy_out.rsv = 0;

        AscendC::DataCopyPadExtParams<DT_X> pad_params;
        pad_params.isPad = false;
        pad_params.leftPadding = 0;
        pad_params.rightPadding = 0;
        pad_params.paddingValue = 0;

        uint32_t row_count = outBatch_ * outHeight_;
        uint32_t unit_count = row_count * 2 * rowGroupLength_;
        uint32_t start_unit = core_idx * corePixelStride_;
        uint32_t end_unit = start_unit + corePixelStride_;
        if (end_unit > unit_count) {
            end_unit = unit_count;
        }

        for (uint32_t unit = start_unit; unit < end_unit; ++unit) {
            uint32_t tmp = unit;
            uint32_t chunk = tmp % rowGroupLength_;
            tmp = tmp / rowGroupLength_;
            uint32_t bw = tmp & 1;
            uint32_t row = tmp >> 1;
            uint32_t on = row / outHeight_;
            uint32_t out_h = row - on * outHeight_;
            uint32_t padded_h = out_h + cropTop_;
            uint32_t h = padded_h >> 1;
            uint32_t bh = padded_h & 1;

            uint32_t start_ow = 0;
            if ((cropLeft_ & 1) != bw) {
                start_ow = 1;
            }
            if (start_ow >= outWidth_) {
                continue;
            }
            uint32_t copy_width = ((outWidth_ - 1 - start_ow) >> 1) + 1;
            uint32_t lane_offset = chunk * max_copy_width;
            if (lane_offset >= copy_width) {
                continue;
            }
            uint32_t cur = copy_width - lane_offset;
            if (cur > max_copy_width) {
                cur = max_copy_width;
            }
            uint32_t padded_w0 = start_ow + cropLeft_ + lane_offset * 2;
            uint32_t input_w0 = padded_w0 >> 1;
            uint32_t in_n = ((bh << 1) + bw) * outBatch_ + on;
            uint32_t in_row = ((in_n * height_ + h) * width_ + input_w0) * depth_;
            uint32_t out_row = ((on * outHeight_ + out_h) * outWidth_ + start_ow + lane_offset * 2) * depth_;

            copy_in.blockCount = static_cast<uint16_t>(cur);
            copy_out.blockCount = static_cast<uint16_t>(cur);
            AscendC::LocalTensor<DT_X> local = copyQueue_.AllocTensor<DT_X>();
            AscendC::DataCopyPad(local, xGm_[in_row], copy_in, pad_params);
            copyQueue_.EnQue(local);
            local = copyQueue_.DeQue<DT_X>();
            AscendC::DataCopyPad(yGm_[out_row], local, copy_out);
            copyQueue_.FreeTensor(local);
        }

    }

    __aicore__ inline void ProcessBnPadCopy() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        uint32_t depth_bytes = depth_ * sizeof(DT_X);
        uint32_t max_copy_width = GetPadMaxCopyWidth();

        AscendC::DataCopyExtParams copy_in;
        copy_in.blockLen = depth_bytes;
        copy_in.srcStride = 0;
        copy_in.dstStride = 0;
        copy_in.rsv = 0;

        AscendC::DataCopyExtParams copy_out;
        copy_out.blockLen = depth_bytes;
        copy_out.srcStride = 0;
        copy_out.dstStride = (blockSize_ - 1) * depth_bytes;
        copy_out.rsv = 0;

        AscendC::DataCopyPadExtParams<DT_X> pad_params;
        pad_params.isPad = false;
        pad_params.leftPadding = 0;
        pad_params.rightPadding = 0;
        pad_params.paddingValue = 0;

        uint32_t row_count = outBatch_ * outHeight_;
        uint32_t unit_count = row_count * blockSize_ * rowGroupLength_;
        uint32_t start_unit = core_idx * corePixelStride_;
        uint32_t end_unit = start_unit + corePixelStride_;
        if (end_unit > unit_count) {
            end_unit = unit_count;
        }

        for (uint32_t unit = start_unit; unit < end_unit; ++unit) {
            uint32_t tmp = unit;
            uint32_t chunk = tmp % rowGroupLength_;
            tmp = tmp / rowGroupLength_;
            uint32_t bw = tmp % blockSize_;
            uint32_t row = tmp / blockSize_;
            uint32_t on = row / outHeight_;
            uint32_t out_h = row - on * outHeight_;
            uint32_t padded_h = out_h + cropTop_;
            uint32_t ih = padded_h / blockSize_;
            uint32_t bh = padded_h - ih * blockSize_;

            uint32_t crop_mod = cropLeft_ % blockSize_;
            uint32_t start_ow = bw >= crop_mod ? (bw - crop_mod) : (bw + blockSize_ - crop_mod);
            if (start_ow >= outWidth_) {
                continue;
            }
            uint32_t copy_width = ((outWidth_ - 1 - start_ow) / blockSize_) + 1;
            uint32_t lane_offset = chunk * max_copy_width;
            if (lane_offset >= copy_width) {
                continue;
            }
            uint32_t cur = copy_width - lane_offset;
            if (cur > max_copy_width) {
                cur = max_copy_width;
            }

            uint32_t padded_w0 = start_ow + cropLeft_ + lane_offset * blockSize_;
            uint32_t input_w0 = padded_w0 / blockSize_;
            uint32_t in_n = (bh * blockSize_ + bw) * outBatch_ + on;
            uint32_t in_row = ((in_n * height_ + ih) * width_ + input_w0) * depth_;
            uint32_t out_row = ((on * outHeight_ + out_h) * outWidth_ + start_ow +
                lane_offset * blockSize_) * depth_;

            copy_in.blockCount = static_cast<uint16_t>(cur);
            copy_out.blockCount = static_cast<uint16_t>(cur);
            AscendC::LocalTensor<DT_X> local = copyQueue_.AllocTensor<DT_X>();
            AscendC::DataCopyPad(local, xGm_[in_row], copy_in, pad_params);
            copyQueue_.EnQue(local);
            local = copyQueue_.DeQue<DT_X>();
            AscendC::DataCopyPad(yGm_[out_row], local, copy_out);
            copyQueue_.FreeTensor(local);
        }

    }

    __aicore__ inline void ProcessB1ContigPadCopy() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        uint32_t max_copy_elems = copyUbBytes_ / sizeof(DT_X);
        uint32_t elems_per_batch = outHeight_ * outWidth_ * depth_;
        uint32_t unit_count = outBatch_ * rowGroupLength_;
        uint32_t start_unit = core_idx * corePixelStride_;
        uint32_t end_unit = start_unit + corePixelStride_;
        if (end_unit > unit_count) {
            end_unit = unit_count;
        }

        AscendC::DataCopyExtParams copy_params;
        copy_params.blockCount = 1;
        copy_params.srcStride = 0;
        copy_params.dstStride = 0;
        copy_params.rsv = 0;

        AscendC::DataCopyPadExtParams<DT_X> pad_params;
        pad_params.isPad = false;
        pad_params.leftPadding = 0;
        pad_params.rightPadding = 0;
        pad_params.paddingValue = 0;

        for (uint32_t unit = start_unit; unit < end_unit; ++unit) {
            uint32_t chunk = unit % rowGroupLength_;
            uint32_t on = unit / rowGroupLength_;
            uint32_t offset = chunk * max_copy_elems;
            if (offset >= elems_per_batch) {
                continue;
            }
            uint32_t cur = elems_per_batch - offset;
            if (cur > max_copy_elems) {
                cur = max_copy_elems;
            }

            uint32_t input_base = ((on * height_ + cropTop_) * width_) * depth_ + offset;
            uint32_t output_base = on * elems_per_batch + offset;
            copy_params.blockLen = cur * sizeof(DT_X);
            AscendC::LocalTensor<DT_X> local = copyQueue_.AllocTensor<DT_X>();
            AscendC::DataCopyPad(local, xGm_[input_base], copy_params, pad_params);
            copyQueue_.EnQue(local);
            local = copyQueue_.DeQue<DT_X>();
            AscendC::DataCopyPad(yGm_[output_base], local, copy_params);
            copyQueue_.FreeTensor(local);
        }

    }

    __aicore__ inline void ProcessB1ContigCopy() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        uint32_t max_copy_elems = copyUbBytes_ / sizeof(DT_X);
        uint32_t elems_per_batch = outHeight_ * outWidth_ * depth_;
        uint32_t unit_count = outBatch_ * rowGroupLength_;
        uint32_t start_unit = core_idx * corePixelStride_;
        uint32_t end_unit = start_unit + corePixelStride_;
        if (end_unit > unit_count) {
            end_unit = unit_count;
        }

        AscendC::DataCopyParams copy_params;
        copy_params.blockCount = 1;
        copy_params.srcStride = 0;
        copy_params.dstStride = 0;

        for (uint32_t unit = start_unit; unit < end_unit; ++unit) {
            uint32_t chunk = unit % rowGroupLength_;
            uint32_t on = unit / rowGroupLength_;
            uint32_t offset = chunk * max_copy_elems;
            if (offset >= elems_per_batch) {
                continue;
            }
            uint32_t cur = elems_per_batch - offset;
            if (cur > max_copy_elems) {
                cur = max_copy_elems;
            }

            uint32_t input_base = ((on * height_ + cropTop_) * width_) * depth_ + offset;
            uint32_t output_base = on * elems_per_batch + offset;
            copy_params.blockLen = static_cast<uint16_t>((cur * sizeof(DT_X)) / 32U);
            AscendC::LocalTensor<DT_X> local = copyQueue_.AllocTensor<DT_X>();
            AscendC::DataCopy(local, xGm_[input_base], copy_params);
            copyQueue_.EnQue(local);
            local = copyQueue_.DeQue<DT_X>();
            AscendC::DataCopy(yGm_[output_base], local, copy_params);
            copyQueue_.FreeTensor(local);
        }

    }

    __aicore__ inline void ProcessB1PadCopy() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        uint32_t depth_bytes = depth_ * sizeof(DT_X);
        uint32_t max_copy_width = copyUbBytes_ / depth_bytes;

        AscendC::DataCopyExtParams copy_params;
        copy_params.blockLen = 0;
        copy_params.srcStride = 0;
        copy_params.dstStride = 0;
        copy_params.rsv = 0;

        AscendC::DataCopyPadExtParams<DT_X> pad_params;
        pad_params.isPad = false;
        pad_params.leftPadding = 0;
        pad_params.rightPadding = 0;
        pad_params.paddingValue = 0;

        uint32_t row_count = outBatch_ * outHeight_;
        uint32_t unit_count = row_count * rowGroupLength_;
        uint32_t start_unit = core_idx * corePixelStride_;
        uint32_t end_unit = start_unit + corePixelStride_;
        if (end_unit > unit_count) {
            end_unit = unit_count;
        }

        for (uint32_t unit = start_unit; unit < end_unit; ++unit) {
            uint32_t chunk = unit % rowGroupLength_;
            uint32_t row = unit / rowGroupLength_;
            uint32_t on = row / outHeight_;
            uint32_t oh = row - on * outHeight_;
            uint32_t lane_offset = chunk * max_copy_width;
            if (lane_offset >= outWidth_) {
                continue;
            }
            uint32_t cur = outWidth_ - lane_offset;
            if (cur > max_copy_width) {
                cur = max_copy_width;
            }

            uint32_t input_base = ((on * height_ + oh + cropTop_) * width_ + cropLeft_ + lane_offset) * depth_;
            uint32_t output_base = ((on * outHeight_ + oh) * outWidth_ + lane_offset) * depth_;
            copy_params.blockCount = 1;
            copy_params.blockLen = cur * depth_bytes;
            AscendC::LocalTensor<DT_X> local = copyQueue_.AllocTensor<DT_X>();
            AscendC::DataCopyPad(local, xGm_[input_base], copy_params, pad_params);
            copyQueue_.EnQue(local);
            local = copyQueue_.DeQue<DT_X>();
            AscendC::DataCopyPad(yGm_[output_base], local, copy_params);
            copyQueue_.FreeTensor(local);
        }

    }

    __aicore__ inline void ProcessDepth1RowScalar() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        uint32_t start = core_idx * coreStride_;
        uint32_t end = start + coreStride_;
        if (end > outputLength_) {
            end = outputLength_;
        }

        uint32_t pixel = start;
        while (pixel < end) {
            uint32_t tmp = pixel;
            uint32_t ow = tmp % outWidth_;
            tmp = tmp / outWidth_;
            uint32_t oh = tmp % outHeight_;
            uint32_t on = tmp / outHeight_;

            uint32_t row_remain = outWidth_ - ow;
            uint32_t count = end - pixel;
            if (count > row_remain) {
                count = row_remain;
            }

            uint32_t padded_h = oh + cropTop_;
            uint32_t padded_w = ow + cropLeft_;
            if (blockSize_ == 2) {
                uint32_t ih = padded_h >> 1;
                uint32_t block_h = padded_h & 1;
                for (uint32_t i = 0; i < count; ++i) {
                    uint32_t cur_w = padded_w + i;
                    uint32_t iw = cur_w >> 1;
                    uint32_t block_w = cur_w & 1;
                    uint32_t in_n = ((block_h << 1) + block_w) * outBatch_ + on;
                    uint32_t input_index = ((in_n * height_ + ih) * width_ + iw);
                    yGm_.SetValue(pixel + i, xGm_.GetValue(input_index));
                }
            } else if (blockSize_ == 1) {
                uint32_t input_index = ((on * height_ + padded_h) * width_ + padded_w);
                for (uint32_t i = 0; i < count; ++i) {
                    yGm_.SetValue(pixel + i, xGm_.GetValue(input_index + i));
                }
            } else {
                uint32_t ih = padded_h / blockSize_;
                uint32_t block_h = padded_h - ih * blockSize_;
                for (uint32_t i = 0; i < count; ++i) {
                    uint32_t cur_w = padded_w + i;
                    uint32_t iw = cur_w / blockSize_;
                    uint32_t block_w = cur_w - iw * blockSize_;
                    uint32_t in_n = (block_h * blockSize_ + block_w) * outBatch_ + on;
                    uint32_t input_index = ((in_n * height_ + ih) * width_ + iw);
                    yGm_.SetValue(pixel + i, xGm_.GetValue(input_index));
                }
            }
            pixel += count;
        }

        AscendC::DataCacheCleanAndInvalid<DT_X, AscendC::CacheLine::ENTIRE_DATA_CACHE>(yGm_);
    }

    __aicore__ inline void ProcessB1RowCopy() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        uint32_t depth_blocks = depth_ * sizeof(DT_X) / 32;
        uint32_t max_copy_width = copyUbBytes_ / (depth_ * sizeof(DT_X));
        if (max_copy_width > 4095) {
            max_copy_width = 4095;
        }

        AscendC::DataCopyParams copy_params;
        copy_params.blockLen = 0;
        copy_params.blockCount = 1;
        copy_params.srcStride = 0;
        copy_params.dstStride = 0;

        uint32_t row_count = outBatch_ * outHeight_;
        uint32_t unit_count = row_count * rowGroupLength_;
        uint32_t start_unit = core_idx * corePixelStride_;
        uint32_t end_unit = start_unit + corePixelStride_;
        if (end_unit > unit_count) {
            end_unit = unit_count;
        }

        for (uint32_t unit = start_unit; unit < end_unit; ++unit) {
            uint32_t chunk = unit % rowGroupLength_;
            uint32_t row = unit / rowGroupLength_;
            uint32_t on = row / outHeight_;
            uint32_t oh = row - on * outHeight_;
            uint32_t lane_offset = chunk * max_copy_width;
            if (lane_offset >= outWidth_) {
                continue;
            }
            uint32_t cur = outWidth_ - lane_offset;
            if (cur > max_copy_width) {
                cur = max_copy_width;
            }

            uint32_t input_base = ((on * height_ + oh + cropTop_) * width_ + cropLeft_ + lane_offset) * depth_;
            uint32_t output_base = ((on * outHeight_ + oh) * outWidth_ + lane_offset) * depth_;
            copy_params.blockLen = static_cast<uint16_t>(cur * depth_blocks);
            AscendC::LocalTensor<DT_X> local = copyQueue_.AllocTensor<DT_X>();
            AscendC::DataCopy(local, xGm_[input_base], copy_params);
            copyQueue_.EnQue(local);
            local = copyQueue_.DeQue<DT_X>();
            AscendC::DataCopy(yGm_[output_base], local, copy_params);
            copyQueue_.FreeTensor(local);
        }

    }

    __aicore__ inline void ProcessScalarByElement() {
        uint32_t core_idx = AscendC::GetBlockIdx();
        uint32_t start = core_idx * coreStride_;
        uint32_t end = start + coreStride_;
        if (end > outputLength_) {
            end = outputLength_;
        }

        for (uint32_t index = start; index < end; ++index) {
            uint32_t tmp = index;
            uint32_t c = 0;
            if (depth_ != 1) {
                c = tmp % depth_;
                tmp = tmp / depth_;
            }
            uint32_t ow = tmp % outWidth_;
            tmp = tmp / outWidth_;
            uint32_t oh = tmp % outHeight_;
            uint32_t on = tmp / outHeight_;

            uint32_t padded_h = oh + cropTop_;
            uint32_t padded_w = ow + cropLeft_;
            uint32_t ih;
            uint32_t iw;
            uint32_t block_h;
            uint32_t block_w;
            if (blockSize_ == 2) {
                ih = padded_h >> 1;
                iw = padded_w >> 1;
                block_h = padded_h & 1;
                block_w = padded_w & 1;
            } else if (blockSize_ == 1) {
                ih = padded_h;
                iw = padded_w;
                block_h = 0;
                block_w = 0;
            } else {
                ih = padded_h / blockSize_;
                iw = padded_w / blockSize_;
                block_h = padded_h - ih * blockSize_;
                block_w = padded_w - iw * blockSize_;
            }

            uint32_t in_n = (block_h * blockSize_ + block_w) * outBatch_ + on;
            uint32_t input_index = ((in_n * height_ + ih) * width_ + iw) * depth_ + c;
            yGm_.SetValue(index, xGm_.GetValue(input_index));
        }

        AscendC::DataCacheCleanAndInvalid<DT_X, AscendC::CacheLine::ENTIRE_DATA_CACHE>(yGm_);
    }

    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;
    AscendC::TPipe pipe_;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 1> copyQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 3> gatherInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> gatherOutQueue_;
    uint32_t height_;
    uint32_t width_;
    uint32_t depth_;
    uint32_t outBatch_;
    uint32_t outHeight_;
    uint32_t outWidth_;
    uint32_t cropTop_;
    uint32_t cropLeft_;
    uint32_t blockSize_;
    uint32_t usedCoreNum_;
    uint32_t outputLength_;
    uint32_t coreStride_;
    uint32_t corePixelStride_;
    uint32_t rowGroupLength_;
    uint32_t depthSplit_;
};

template <typename DT_X, int COPY_MODE>
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingData, tiling_data, tiling);
    KernelBatchToSpace<DT_X, COPY_MODE> op;
    op.Init(x, y, tiling_data);
    op.Process();
}
