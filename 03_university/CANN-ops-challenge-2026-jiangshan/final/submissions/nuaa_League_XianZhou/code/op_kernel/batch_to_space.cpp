// Kernel-side BatchToSpace implementation.
#include "kernel_operator.h"

#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

using namespace AscendC;

template <class DT_X>
class KernelBatchToSpace {
public:
    __aicore__ inline KernelBatchToSpace() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &td) {
        copy_mode_ = td.use_stride_copy;
        if (copy_mode_ >= 67u && copy_mode_ <= 73u) {
            InitPoint7Linear(x, y);
            return;
        }

        height_ = td.height;
        width_ = td.width;
        depth_ = td.depth;
        block_size_ = td.block_size;
        crop_top_ = td.crop_top;
        crop_left_ = td.crop_left;
        new_batch_ = td.new_batch;
        out_height_ = td.out_height;
        out_width_ = td.out_width;
        depth_bytes_ = td.depth_bytes;
        depth_aligned_ = td.depth_aligned;
        ppb_ = td.pixels_per_batch;
        total_units_ = td.total_rows;
        per_core_units_ = td.per_core_rows;
        tile_rows_ = td.per_core_pixels;
        depth_bytes_32_ = td.depth_bytes_32;

        in_stride_b_ = static_cast<uint64_t>(height_) * width_ * depth_;
        in_stride_h_ = static_cast<uint64_t>(width_) * depth_;
        out_stride_b_ = static_cast<uint64_t>(out_height_) * out_width_ * depth_;
        out_stride_h_ = static_cast<uint64_t>(out_width_) * depth_;

        uint32_t block_idx = GetBlockIdx();
        uint32_t start = block_idx * per_core_units_;
        uint32_t end = start + per_core_units_;
        if (start > total_units_) {
            start = total_units_;
        }
        if (end > total_units_) {
            end = total_units_;
        }
        my_start_ = start;
        my_count_ = (end > start) ? (end - start) : 0u;
        if ((copy_mode_ == 16u || copy_mode_ == 26u || copy_mode_ == 27u ||
             copy_mode_ == 28u || copy_mode_ == 29u || copy_mode_ == 30u ||
             copy_mode_ == 31u || copy_mode_ == 32u || copy_mode_ == 33u ||
             copy_mode_ == 34u || copy_mode_ == 35u || copy_mode_ == 36u ||
             copy_mode_ == 37u || copy_mode_ == 38u || copy_mode_ == 39u ||
             copy_mode_ == 40u ||
             copy_mode_ == 43u || copy_mode_ == 55u || copy_mode_ == 108u) &&
            tile_rows_ > 0u) {
            uint32_t base_units = total_units_ / tile_rows_;
            uint32_t extra_units = total_units_ - base_units * tile_rows_;
            my_start_ = block_idx * base_units + ((block_idx < extra_units) ? block_idx : extra_units);
            my_count_ = base_units + ((block_idx < extra_units) ? 1u : 0u);
        }

        uint64_t input_total = static_cast<uint64_t>(td.batch) * height_ * width_ * depth_;
        uint64_t output_total = static_cast<uint64_t>(new_batch_) * out_height_ * out_width_ * depth_;
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), input_total);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), output_total);

        uint32_t ub_elems = ppb_ * depth_aligned_;
        if (ub_elems == 0u) {
            ub_elems = 32u / static_cast<uint32_t>(sizeof(DT_X));
        }
        if (copy_mode_ == 2u || copy_mode_ == 7u || copy_mode_ == 8u ||
            copy_mode_ == 16u || copy_mode_ == 26u || copy_mode_ == 27u || copy_mode_ == 43u) {
            ub_elems *= 2u;
        }
        if (copy_mode_ == 28u) {
            ub_elems = 66048u;
        }
        if (copy_mode_ == 29u) {
            ub_elems = 82688u;
        }
        if (copy_mode_ == 30u) {
            ub_elems = 33024u;
        }
        if (copy_mode_ == 31u) {
            ub_elems = 82560u;
        }
        if (copy_mode_ == 32u) {
            ub_elems = 82560u;
        }
        if (copy_mode_ == 33u) {
            ub_elems = 49920u;
        }
        if (copy_mode_ == 34u) {
            ub_elems = 24960u;
        }
        if (copy_mode_ == 35u) {
            ub_elems = 33584u;
        }
        if (copy_mode_ == 36u) {
            ub_elems = 83200u;
        }
        if (copy_mode_ == 37u) {
            ub_elems = 49920u;
        }
        if (copy_mode_ == 38u) {
            ub_elems = 49920u;
        }
        if (copy_mode_ == 39u) {
            ub_elems = 49920u;
        }
        if (copy_mode_ == 40u) {
            ub_elems = 66560u;
        }
        if (copy_mode_ == 108u) {
            ub_elems = 82560u;
        }
        if (copy_mode_ == 55u) {
            ub_elems = 1024u;
        }
        if (copy_mode_ == 67u) {
            ub_elems = 8192u;
        }
        if (copy_mode_ == 68u) {
            ub_elems = 4096u;
        }
        if (copy_mode_ == 69u) {
            ub_elems = 16384u;
        }
        if (copy_mode_ == 70u) {
            ub_elems = 8192u;
        }
        if (copy_mode_ == 71u) {
            ub_elems = 8192u;
        }
        if (copy_mode_ == 72u) {
            ub_elems = 8192u;
        }
        if (copy_mode_ == 73u) {
            ub_elems = 8192u;
        }
        if (copy_mode_ == 94u) {
            ub_elems = 92160u;
        }
        if (copy_mode_ == 102u) {
            ub_elems = 24576u;
        }
        if (copy_mode_ == 103u) {
            ub_elems = 12288u;
        }
        if (copy_mode_ == 104u) {
            ub_elems = 6144u;
        }
        if (copy_mode_ == 111u) {
            ub_elems = 30720u;
        }
        pipe_.InitBuffer(buf_, ub_elems * static_cast<uint32_t>(sizeof(DT_X)));
    }

    __aicore__ inline void InitPoint7Linear(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kTotalElems = 65536u;
        constexpr uint32_t kDepth = 16384u;
        constexpr uint32_t kChunks8 = 8u;
        constexpr uint32_t kChunks16 = 16u;
        constexpr uint32_t kChunks4 = 4u;

        depth_ = kDepth;
        uint32_t chunks = kChunks8;
        uint32_t ub_elems = 8192u;
        if (copy_mode_ == 68u) {
            chunks = kChunks16;
            ub_elems = 4096u;
        } else if (copy_mode_ == 69u) {
            chunks = kChunks4;
            ub_elems = 16384u;
        }

        uint32_t block_idx = GetBlockIdx();
        my_start_ = block_idx;
        my_count_ = (block_idx < chunks) ? 1u : 0u;
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), kTotalElems);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), kTotalElems);
        pipe_.InitBuffer(buf_, ub_elems * static_cast<uint32_t>(sizeof(DT_X)));
    }

    __aicore__ inline void Process() {
        if (depth_ == 0u || my_count_ == 0u) {
            return;
        }
        if (copy_mode_ == 67u) {
            ProcessPoint7Linear8Core();
        } else if (copy_mode_ == 68u) {
            ProcessPoint7Linear16Core();
        } else if (copy_mode_ == 69u) {
            ProcessPoint7Linear4Core();
        } else if (copy_mode_ == 70u) {
            ProcessPoint7Linear8CorePipe2();
        } else if (copy_mode_ == 71u) {
            ProcessPoint7Linear8CoreFast();
        } else if (copy_mode_ == 72u) {
            ProcessPoint7Linear8CoreFastNoWait();
        } else if (copy_mode_ == 73u) {
            ProcessPoint7Linear8CoreCountCopy();
        } else if (copy_mode_ == 43u) {
            ProcessPoint1RowsGrouped2();
        } else if (copy_mode_ == 111u) {
            ProcessPoint1BetaCase0();
        } else if (copy_mode_ == 104u) {
            ProcessPoint10Tile8Pipeline();
        } else if (copy_mode_ == 102u) {
            ProcessPoint10Tile32Pipeline();
        } else if (copy_mode_ == 103u) {
            ProcessPoint10Tile16Pipeline();
        } else if (copy_mode_ == 94u) {
            ProcessPoint9GatherTile240Pipeline();
        } else if (copy_mode_ == 55u) {
            ProcessPoint2GatherPipeline();
        } else if (copy_mode_ == 108u) {
            ProcessScalarPoint5GatherPipelineSafe();
        } else if (copy_mode_ == 32u) {
            ProcessScalarPoint5GatherPipelineDstEvent();
        } else if (copy_mode_ == 33u) {
            ProcessScalarPoint5GatherHalfPipeline();
        } else if (copy_mode_ == 34u) {
            ProcessScalarPoint5GatherQuarterPipeline();
        } else if (copy_mode_ == 35u) {
            ProcessScalarPoint5GatherThirdPipeline();
        } else if (copy_mode_ == 36u) {
            ProcessScalarPoint5GatherHalf4SlotPipeline();
        } else if (copy_mode_ == 37u) {
            ProcessScalarPoint5GatherHalfPipeline();
        } else if (copy_mode_ == 38u) {
            ProcessScalarPoint5GatherHalfFastPipeline();
        } else if (copy_mode_ == 39u) {
            ProcessScalarPoint5GatherHalfFastPipeline();
        } else if (copy_mode_ == 40u) {
            ProcessScalarPoint5GatherHalfFast3StagePipeline();
        } else if (copy_mode_ == 31u) {
            ProcessScalarPoint5GatherPipelineBarrier();
        } else if (copy_mode_ == 30u) {
            ProcessScalarPoint5InterleaveReadPipeline();
        } else if (copy_mode_ == 29u) {
            ProcessScalarPoint5Gather64Pipeline();
        } else if (copy_mode_ == 26u) {
            ProcessScalarPoint5Fixed();
        } else if (copy_mode_ == 27u) {
            ProcessScalarPoint5RowGroups();
        } else if (copy_mode_ == 28u) {
            ProcessScalarPoint5Gather();
        } else if (copy_mode_ == 25u) {
            ProcessStrideTinyFixed();
        } else if (copy_mode_ == 16u) {
            ProcessStrideRowsPipeline();
        } else if (copy_mode_ == 9u) {
            ProcessShortWidthTile();
        } else if (copy_mode_ == 8u) {
            ProcessStrideTaskPipeline();
        } else if (copy_mode_ == 7u) {
            ProcessStrideStreamsPipeline();
        } else if (copy_mode_ == 4u) {
            ProcessFullCopy();
        } else if (copy_mode_ == 3u) {
            ProcessScalarCropRows();
        } else if (copy_mode_ == 1u) {
            ProcessStrideRows();
        } else if (copy_mode_ == 2u) {
            ProcessStrideRowsDoubleBuffer();
        } else {
            if (block_size_ > 1u) {
                ProcessScalarStreamRows();
            } else {
                ProcessScalarRows();
            }
        }
    }

private:
    __aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) const {
        return (a + b - 1u) / b;
    }

    __aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) const {
        return (a < b) ? a : b;
    }

    __aicore__ inline void ProcessStrideRows() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr event_t event_id = EVENT_ID0;
        uint16_t block_len_32 = static_cast<uint16_t>(depth_bytes_32_);
        uint16_t skip_32 = static_cast<uint16_t>((block_size_ - 1u) * depth_bytes_32_);

        DataCopyParams read_params;
        read_params.blockLen = block_len_32;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyParams write_params;
        write_params.blockLen = block_len_32;
        write_params.srcStride = 0u;
        write_params.dstStride = skip_32;

        uint32_t crop_left_mod = crop_left_ - (crop_left_ / block_size_) * block_size_;
        SetFlag<HardEvent::MTE3_MTE2>(event_id);

        for (uint32_t r = 0u; r < my_count_; ++r) {
            uint32_t row_idx = my_start_ + r;
            if (row_idx >= total_units_) {
                break;
            }

            uint32_t bi = row_idx / out_height_;
            uint32_t h_out_c = row_idx - bi * out_height_;
            uint32_t h_out = h_out_c + crop_top_;
            uint32_t h_in = h_out / block_size_;
            uint32_t b1 = h_out - h_in * block_size_;

            uint64_t out_row_base = static_cast<uint64_t>(bi) * out_stride_b_
                                  + static_cast<uint64_t>(h_out_c) * out_stride_h_;
            uint64_t in_h_base = static_cast<uint64_t>(h_in) * in_stride_h_;

            for (uint32_t b2 = 0u; b2 < block_size_; ++b2) {
                uint32_t first_w_out_c = b2 + block_size_ - crop_left_mod;
                first_w_out_c -= (first_w_out_c / block_size_) * block_size_;
                if (first_w_out_c >= out_width_) {
                    continue;
                }

                uint32_t total_w = (out_width_ - first_w_out_c + block_size_ - 1u) / block_size_;
                uint32_t first_w_in = (first_w_out_c + crop_left_) / block_size_;
                uint32_t input_batch = (b1 * block_size_ + b2) * new_batch_ + bi;

                uint64_t in_base = static_cast<uint64_t>(input_batch) * in_stride_b_
                                 + in_h_base
                                 + static_cast<uint64_t>(first_w_in) * depth_;
                uint64_t out_base = out_row_base + static_cast<uint64_t>(first_w_out_c) * depth_;

                for (uint32_t done = 0u; done < total_w; done += ppb_) {
                    uint32_t chunk = MinU32(total_w - done, ppb_);
                    read_params.blockCount = static_cast<uint16_t>(chunk);
                    write_params.blockCount = static_cast<uint16_t>(chunk);

                    WaitFlag<HardEvent::MTE3_MTE2>(event_id);
                    DataCopy(ubuf[0], xGM_[in_base + static_cast<uint64_t>(done) * depth_], read_params);
                    SetFlag<HardEvent::MTE2_MTE3>(event_id);
                    WaitFlag<HardEvent::MTE2_MTE3>(event_id);
                    DataCopy(yGM_[out_base + static_cast<uint64_t>(done) * block_size_ * depth_], ubuf[0], write_params);
                    SetFlag<HardEvent::MTE3_MTE2>(event_id);
                }
            }
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
    }

    __aicore__ inline void ProcessStrideRowsDoubleBuffer() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        const uint32_t buf_elems = ppb_ * depth_aligned_;
        uint16_t block_len_32 = static_cast<uint16_t>(depth_bytes_32_);
        uint16_t skip_32 = static_cast<uint16_t>((block_size_ - 1u) * depth_bytes_32_);

        DataCopyParams read_params;
        read_params.blockLen = block_len_32;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyParams write_params;
        write_params.blockLen = block_len_32;
        write_params.srcStride = 0u;
        write_params.dstStride = skip_32;

        uint32_t crop_left_mod = crop_left_ - (crop_left_ / block_size_) * block_size_;
        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);

        for (uint32_t r = 0u; r < my_count_; ++r) {
            uint32_t row_idx = my_start_ + r;
            if (row_idx >= total_units_) {
                break;
            }

            uint32_t bi = row_idx / out_height_;
            uint32_t h_out_c = row_idx - bi * out_height_;
            uint32_t h_out = h_out_c + crop_top_;
            uint32_t h_in = h_out / block_size_;
            uint32_t b1 = h_out - h_in * block_size_;

            uint64_t out_row_base = static_cast<uint64_t>(bi) * out_stride_b_
                                  + static_cast<uint64_t>(h_out_c) * out_stride_h_;
            uint64_t in_h_base = static_cast<uint64_t>(h_in) * in_stride_h_;

            for (uint32_t b2 = 0u; b2 < block_size_; ++b2) {
                uint32_t first_w_out_c = b2 + block_size_ - crop_left_mod;
                first_w_out_c -= (first_w_out_c / block_size_) * block_size_;
                if (first_w_out_c >= out_width_) {
                    continue;
                }

                uint32_t total_w = (out_width_ - first_w_out_c + block_size_ - 1u) / block_size_;
                uint32_t first_w_in = (first_w_out_c + crop_left_) / block_size_;
                uint32_t input_batch = (b1 * block_size_ + b2) * new_batch_ + bi;

                uint64_t in_base = static_cast<uint64_t>(input_batch) * in_stride_b_
                                 + in_h_base
                                 + static_cast<uint64_t>(first_w_in) * depth_;
                uint64_t out_base = out_row_base + static_cast<uint64_t>(first_w_out_c) * depth_;

                uint32_t chunks = CeilDiv(total_w, ppb_);
                if (chunks == 0u) {
                    continue;
                }

                uint32_t first_chunk = MinU32(total_w, ppb_);
                read_params.blockCount = static_cast<uint16_t>(first_chunk);
                WaitFlag<HardEvent::MTE3_MTE2>(event0);
                DataCopy(ubuf[0], xGM_[in_base], read_params);
                SetFlag<HardEvent::MTE2_MTE3>(event0);

                for (uint32_t chunk_idx = 1u; chunk_idx < chunks; ++chunk_idx) {
                    uint32_t curr_slot = chunk_idx & 1u;
                    uint32_t prev_slot = curr_slot ^ 1u;
                    event_t curr_event = (curr_slot == 0u) ? event0 : event1;
                    event_t prev_event = (prev_slot == 0u) ? event0 : event1;
                    uint32_t done = chunk_idx * ppb_;
                    uint32_t chunk = MinU32(total_w - done, ppb_);

                    read_params.blockCount = static_cast<uint16_t>(chunk);
                    WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
                    DataCopy(ubuf[curr_slot * buf_elems], xGM_[in_base + static_cast<uint64_t>(done) * depth_],
                             read_params);
                    SetFlag<HardEvent::MTE2_MTE3>(curr_event);

                    uint32_t prev_done = (chunk_idx - 1u) * ppb_;
                    uint32_t prev_chunk = MinU32(total_w - prev_done, ppb_);
                    write_params.blockCount = static_cast<uint16_t>(prev_chunk);
                    WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
                    DataCopy(yGM_[out_base + static_cast<uint64_t>(prev_done) * block_size_ * depth_],
                             ubuf[prev_slot * buf_elems], write_params);
                    SetFlag<HardEvent::MTE3_MTE2>(prev_event);
                }

                uint32_t last_slot = (chunks - 1u) & 1u;
                event_t last_event = (last_slot == 0u) ? event0 : event1;
                uint32_t last_done = (chunks - 1u) * ppb_;
                uint32_t last_chunk = MinU32(total_w - last_done, ppb_);
                write_params.blockCount = static_cast<uint16_t>(last_chunk);
                WaitFlag<HardEvent::MTE2_MTE3>(last_event);
                DataCopy(yGM_[out_base + static_cast<uint64_t>(last_done) * block_size_ * depth_],
                         ubuf[last_slot * buf_elems], write_params);
                SetFlag<HardEvent::MTE3_MTE2>(last_event);

                WaitFlag<HardEvent::MTE3_MTE2>(event0);
                WaitFlag<HardEvent::MTE3_MTE2>(event1);
                SetFlag<HardEvent::MTE3_MTE2>(event0);
                SetFlag<HardEvent::MTE3_MTE2>(event1);
            }
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);

    }

    __aicore__ inline void ProcessStrideStreamsPipeline() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        const uint32_t buf_elems = ppb_ * depth_aligned_;
        uint16_t block_len_32 = static_cast<uint16_t>(depth_bytes_32_);
        uint16_t skip_32 = static_cast<uint16_t>((block_size_ - 1u) * depth_bytes_32_);

        DataCopyParams read_params;
        read_params.blockLen = block_len_32;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyParams write_params;
        write_params.blockLen = block_len_32;
        write_params.srcStride = 0u;
        write_params.dstStride = skip_32;

        uint32_t crop_left_mod = crop_left_ - (crop_left_ / block_size_) * block_size_;
        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);

        bool has_prev = false;
        uint32_t prev_slot = 0u;
        uint32_t prev_count = 0u;
        uint64_t prev_out_base = 0u;

        for (uint32_t s = 0u; s < my_count_; ++s) {
            uint32_t stream_idx = my_start_ + s;
            uint32_t row_idx = stream_idx / block_size_;
            if (row_idx >= total_units_) {
                break;
            }
            uint32_t b2 = stream_idx - row_idx * block_size_;

            uint32_t bi = row_idx / out_height_;
            uint32_t h_out_c = row_idx - bi * out_height_;
            uint32_t h_out = h_out_c + crop_top_;
            uint32_t h_in = h_out / block_size_;
            uint32_t b1 = h_out - h_in * block_size_;

            uint32_t first_w_out_c = b2 + block_size_ - crop_left_mod;
            first_w_out_c -= (first_w_out_c / block_size_) * block_size_;
            if (first_w_out_c >= out_width_) {
                continue;
            }

            uint32_t count = (out_width_ - first_w_out_c + block_size_ - 1u) / block_size_;
            if (count == 0u || count > ppb_) {
                continue;
            }

            uint32_t first_w_in = (first_w_out_c + crop_left_) / block_size_;
            uint32_t input_batch = (b1 * block_size_ + b2) * new_batch_ + bi;
            uint64_t in_base = static_cast<uint64_t>(input_batch) * in_stride_b_
                             + static_cast<uint64_t>(h_in) * in_stride_h_
                             + static_cast<uint64_t>(first_w_in) * depth_;
            uint64_t out_base = static_cast<uint64_t>(bi) * out_stride_b_
                              + static_cast<uint64_t>(h_out_c) * out_stride_h_
                              + static_cast<uint64_t>(first_w_out_c) * depth_;

            uint32_t curr_slot = has_prev ? (prev_slot ^ 1u) : 0u;
            event_t curr_event = (curr_slot == 0u) ? event0 : event1;
            read_params.blockCount = static_cast<uint16_t>(count);

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            DataCopy(ubuf[curr_slot * buf_elems], xGM_[in_base], read_params);
            SetFlag<HardEvent::MTE2_MTE3>(curr_event);

            if (has_prev) {
                event_t prev_event = (prev_slot == 0u) ? event0 : event1;
                write_params.blockCount = static_cast<uint16_t>(prev_count);
                WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
                DataCopy(yGM_[prev_out_base], ubuf[prev_slot * buf_elems], write_params);
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            has_prev = true;
            prev_slot = curr_slot;
            prev_count = count;
            prev_out_base = out_base;
        }

        if (has_prev) {
            event_t prev_event = (prev_slot == 0u) ? event0 : event1;
            write_params.blockCount = static_cast<uint16_t>(prev_count);
            WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
            DataCopy(yGM_[prev_out_base], ubuf[prev_slot * buf_elems], write_params);
            SetFlag<HardEvent::MTE3_MTE2>(prev_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
    }

    __aicore__ inline void ProcessStrideTaskPipeline() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        const uint32_t buf_elems = ppb_ * depth_aligned_;
        uint16_t block_len_32 = static_cast<uint16_t>(depth_bytes_32_);
        uint16_t skip_32 = static_cast<uint16_t>((block_size_ - 1u) * depth_bytes_32_);

        DataCopyParams read_params;
        read_params.blockLen = block_len_32;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyParams write_params;
        write_params.blockLen = block_len_32;
        write_params.srcStride = 0u;
        write_params.dstStride = skip_32;

        uint32_t actual_rows = new_batch_ * out_height_;
        uint32_t max_stream_pixels = CeilDiv(out_width_, block_size_);
        uint32_t chunks_per_stream = CeilDiv(max_stream_pixels, ppb_);
        uint32_t tasks_per_row = block_size_ * chunks_per_stream;
        uint32_t crop_left_mod = crop_left_ - (crop_left_ / block_size_) * block_size_;

        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);

        bool has_prev = false;
        uint32_t prev_slot = 0u;
        uint32_t prev_count = 0u;
        uint64_t prev_out_base = 0u;

        for (uint32_t t = 0u; t < my_count_; ++t) {
            uint32_t task = my_start_ + t;
            uint32_t row_idx = task / tasks_per_row;
            if (row_idx >= actual_rows) {
                break;
            }
            uint32_t rem = task - row_idx * tasks_per_row;
            uint32_t b2 = rem / chunks_per_stream;
            uint32_t chunk_idx = rem - b2 * chunks_per_stream;

            uint32_t first_w_out_c = b2 + block_size_ - crop_left_mod;
            first_w_out_c -= (first_w_out_c / block_size_) * block_size_;
            if (first_w_out_c >= out_width_) {
                continue;
            }

            uint32_t total_w = (out_width_ - first_w_out_c + block_size_ - 1u) / block_size_;
            uint32_t done = chunk_idx * ppb_;
            if (done >= total_w) {
                continue;
            }
            uint32_t count = MinU32(total_w - done, ppb_);

            uint32_t bi = row_idx / out_height_;
            uint32_t h_out_c = row_idx - bi * out_height_;
            uint32_t h_out = h_out_c + crop_top_;
            uint32_t h_in = h_out / block_size_;
            uint32_t b1 = h_out - h_in * block_size_;
            uint32_t first_w_in = (first_w_out_c + crop_left_) / block_size_;
            uint32_t input_batch = (b1 * block_size_ + b2) * new_batch_ + bi;

            uint64_t in_base = static_cast<uint64_t>(input_batch) * in_stride_b_
                             + static_cast<uint64_t>(h_in) * in_stride_h_
                             + static_cast<uint64_t>(first_w_in + done) * depth_;
            uint64_t out_base = static_cast<uint64_t>(bi) * out_stride_b_
                              + static_cast<uint64_t>(h_out_c) * out_stride_h_
                              + static_cast<uint64_t>(first_w_out_c + done * block_size_) * depth_;

            uint32_t curr_slot = has_prev ? (prev_slot ^ 1u) : 0u;
            event_t curr_event = (curr_slot == 0u) ? event0 : event1;
            read_params.blockCount = static_cast<uint16_t>(count);

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            DataCopy(ubuf[curr_slot * buf_elems], xGM_[in_base], read_params);
            SetFlag<HardEvent::MTE2_MTE3>(curr_event);

            if (has_prev) {
                event_t prev_event = (prev_slot == 0u) ? event0 : event1;
                write_params.blockCount = static_cast<uint16_t>(prev_count);
                WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
                DataCopy(yGM_[prev_out_base], ubuf[prev_slot * buf_elems], write_params);
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            has_prev = true;
            prev_slot = curr_slot;
            prev_count = count;
            prev_out_base = out_base;
        }

        if (has_prev) {
            event_t prev_event = (prev_slot == 0u) ? event0 : event1;
            write_params.blockCount = static_cast<uint16_t>(prev_count);
            WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
            DataCopy(yGM_[prev_out_base], ubuf[prev_slot * buf_elems], write_params);
            SetFlag<HardEvent::MTE3_MTE2>(prev_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
    }

    __aicore__ inline void ProcessPoint1RowsGrouped2() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        constexpr uint32_t kDepth = 128u;
        constexpr uint32_t kHalfWidth = 28u;
        constexpr uint32_t kOutHeight = 56u;
        constexpr uint32_t kInBatchStride = 28u * 28u * kDepth;
        constexpr uint32_t kInRowStride = 28u * kDepth;
        constexpr uint32_t kOutRowStride = 56u * kDepth;
        constexpr uint32_t kHalfRowElems = kHalfWidth * kDepth;
        constexpr uint32_t kRowElems = 2u * kHalfRowElems;
        constexpr uint32_t kGroupRows = 2u;
        constexpr uint32_t kGroupElems = kGroupRows * kRowElems;
        constexpr uint32_t kHalfRowBlocks = kHalfWidth * 16u;
        constexpr uint32_t kReadGapBlocks =
            (2u * kInBatchStride - kHalfRowElems) * sizeof(DT_X) / 32u;

        DataCopyParams read_params;
        read_params.blockCount = 2u;
        read_params.blockLen = static_cast<uint16_t>(kHalfRowBlocks);
        read_params.srcStride = static_cast<uint16_t>(kReadGapBlocks);
        read_params.dstStride = 0u;

        DataCopyParams write_params;
        write_params.blockCount = static_cast<uint16_t>(kHalfWidth);
        write_params.blockLen = 16u;
        write_params.srcStride = 0u;
        write_params.dstStride = 16u;

        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);

        bool has_prev = false;
        uint32_t prev_slot = 0u;
        uint32_t prev_rows = 0u;
        uint64_t prev_out_base = 0u;

        for (uint32_t done = 0u; done < my_count_; done += kGroupRows) {
            uint32_t rows = MinU32(my_count_ - done, kGroupRows);
            uint32_t row_idx = my_start_ + done;
            uint32_t curr_slot = has_prev ? (prev_slot ^ 1u) : 0u;
            event_t curr_event = (curr_slot == 0u) ? event0 : event1;
            uint32_t curr_ub_base = curr_slot * kGroupElems;

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            for (uint32_t r = 0u; r < rows; ++r) {
                uint32_t curr_row_idx = row_idx + r;
                uint32_t bi = (curr_row_idx >= kOutHeight) ? 1u : 0u;
                uint32_t h_out = curr_row_idx - bi * kOutHeight;
                uint32_t input_batch = ((h_out & 1u) << 2u) + bi;
                uint64_t in_base = static_cast<uint64_t>(input_batch) * kInBatchStride
                                 + static_cast<uint64_t>(h_out >> 1u) * kInRowStride;
                DataCopy(ubuf[curr_ub_base + r * kRowElems], xGM_[in_base], read_params);
            }
            SetFlag<HardEvent::MTE2_MTE3>(curr_event);

            if (has_prev) {
                event_t prev_event = (prev_slot == 0u) ? event0 : event1;
                uint32_t prev_ub_base = prev_slot * kGroupElems;
                WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
                for (uint32_t r = 0u; r < prev_rows; ++r) {
                    uint64_t out_base = prev_out_base + static_cast<uint64_t>(r) * kOutRowStride;
                    uint32_t ub_row_base = prev_ub_base + r * kRowElems;
                    DataCopy(yGM_[out_base], ubuf[ub_row_base], write_params);
                    DataCopy(yGM_[out_base + kDepth],
                             ubuf[ub_row_base + kHalfRowElems], write_params);
                }
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            has_prev = true;
            prev_slot = curr_slot;
            prev_rows = rows;
            prev_out_base = static_cast<uint64_t>(row_idx) * kOutRowStride;
        }

        if (has_prev) {
            event_t prev_event = (prev_slot == 0u) ? event0 : event1;
            uint32_t prev_ub_base = prev_slot * kGroupElems;
            WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
            for (uint32_t r = 0u; r < prev_rows; ++r) {
                uint64_t out_base = prev_out_base + static_cast<uint64_t>(r) * kOutRowStride;
                uint32_t ub_row_base = prev_ub_base + r * kRowElems;
                DataCopy(yGM_[out_base], ubuf[ub_row_base], write_params);
                DataCopy(yGM_[out_base + kDepth],
                         ubuf[ub_row_base + kHalfRowElems], write_params);
            }
            SetFlag<HardEvent::MTE3_MTE2>(prev_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
    }

    __aicore__ inline void ProcessPoint1BetaCase0() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr uint32_t kXH = 28u;
        constexpr uint32_t kXW = 28u;
        constexpr uint32_t kXC = 128u;
        constexpr uint32_t kYN = 2u;
        constexpr uint32_t kYH = 56u;
        constexpr uint32_t kYW = 56u;
        constexpr uint32_t kDBlocks = (kXC * static_cast<uint32_t>(sizeof(DT_X))) / 32u;
        constexpr uint32_t kSourceRowBlocks = (kXW * kXC * static_cast<uint32_t>(sizeof(DT_X))) / 32u;
        constexpr uint32_t kOutRowBlocks = (kYW * kXC * static_cast<uint32_t>(sizeof(DT_X))) / 32u;
        constexpr uint32_t kInputElems = kXW * kXC;
        constexpr uint32_t kInput1Offset = (16u * 1024u) / static_cast<uint32_t>(sizeof(DT_X));
        constexpr uint32_t kInput2Offset = (32u * 1024u) / static_cast<uint32_t>(sizeof(DT_X));
        constexpr uint32_t kInput3Offset = (48u * 1024u) / static_cast<uint32_t>(sizeof(DT_X));
        constexpr uint32_t kOutputOffset = (64u * 1024u) / static_cast<uint32_t>(sizeof(DT_X));
        constexpr uint32_t kInBatchStride = kXH * kXW * kXC;
        constexpr uint32_t kInRowStride = kXW * kXC;

        DataCopyParams load_params;
        load_params.blockCount = 1u;
        load_params.blockLen = static_cast<uint16_t>(kSourceRowBlocks);
        load_params.srcStride = 0u;
        load_params.dstStride = 0u;

        DataCopyParams ub_params;
        ub_params.blockCount = static_cast<uint16_t>(kXW);
        ub_params.blockLen = static_cast<uint16_t>(kDBlocks);
        ub_params.srcStride = 0u;
        ub_params.dstStride = static_cast<uint16_t>(kDBlocks);

        DataCopyParams store_params;
        store_params.blockCount = 1u;
        store_params.blockLen = static_cast<uint16_t>(2u * kOutRowBlocks);
        store_params.srcStride = 0u;
        store_params.dstStride = 0u;

        uint32_t block_idx = GetBlockIdx();
        uint32_t block_num = GetBlockNum();
        for (uint32_t group = block_idx; group < kYN * kXH; group += block_num) {
            uint32_t ih = group % kXH;
            uint32_t n = group / kXH;
            uint32_t input_n0 = n;
            uint32_t input_n1 = input_n0 + kYN;
            uint32_t input_n2 = input_n0 + 2u * kYN;
            uint32_t input_n3 = input_n2 + kYN;
            uint32_t src0 = input_n0 * kInBatchStride + ih * kInRowStride;
            uint32_t src1 = input_n1 * kInBatchStride + ih * kInRowStride;
            uint32_t src2 = input_n2 * kInBatchStride + ih * kInRowStride;
            uint32_t src3 = input_n3 * kInBatchStride + ih * kInRowStride;

            DataCopy(ubuf[0u], xGM_[src0], load_params);
            DataCopy(ubuf[kInput1Offset], xGM_[src1], load_params);
            DataCopy(ubuf[kInput2Offset], xGM_[src2], load_params);
            DataCopy(ubuf[kInput3Offset], xGM_[src3], load_params);
            PipeBarrier<PIPE_ALL>();

            DataCopy(ubuf[kOutputOffset], ubuf[0u], ub_params);
            DataCopy(ubuf[kOutputOffset + kXC], ubuf[kInput1Offset], ub_params);
            DataCopy(ubuf[kOutputOffset + kYW * kXC], ubuf[kInput2Offset], ub_params);
            DataCopy(ubuf[kOutputOffset + kYW * kXC + kXC], ubuf[kInput3Offset], ub_params);
            PipeBarrier<PIPE_MTE2>();

            uint32_t dst = (n * kYH + ih * 2u) * kYW * kXC;
            DataCopy(yGM_[dst], ubuf[kOutputOffset], store_params);
            PipeBarrier<PIPE_MTE3>();
        }
    }

    __aicore__ inline void ProcessShortWidthTile() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr event_t event_id = EVENT_ID0;
        uint16_t block_len_32 = static_cast<uint16_t>(depth_bytes_32_);
        uint16_t dst_skip_32 = static_cast<uint16_t>(depth_bytes_32_);

        DataCopyParams read_params;
        read_params.blockLen = block_len_32;
        read_params.srcStride = 0u;
        read_params.dstStride = dst_skip_32;

        DataCopyParams write_params;
        write_params.blockCount = 1u;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        uint32_t rows_per_tile = tile_rows_;
        if (rows_per_tile == 0u) {
            rows_per_tile = 1u;
        }
        uint32_t tiles_per_batch = CeilDiv(out_height_, rows_per_tile);
        uint32_t crop_left_mod = crop_left_ - (crop_left_ / block_size_) * block_size_;

        SetFlag<HardEvent::MTE3_MTE2>(event_id);

        for (uint32_t t = 0u; t < my_count_; ++t) {
            uint32_t tile_idx = my_start_ + t;
            uint32_t bi = tile_idx / tiles_per_batch;
            if (bi >= new_batch_) {
                break;
            }
            uint32_t tile_in_batch = tile_idx - bi * tiles_per_batch;
            uint32_t h_start_c = tile_in_batch * rows_per_tile;
            if (h_start_c >= out_height_) {
                continue;
            }
            uint32_t tile_rows = MinU32(rows_per_tile, out_height_ - h_start_c);

            WaitFlag<HardEvent::MTE3_MTE2>(event_id);

            for (uint32_t tr = 0u; tr < tile_rows; ++tr) {
                uint32_t h_out_c = h_start_c + tr;
                uint32_t h_out = h_out_c + crop_top_;
                uint32_t h_in = h_out / block_size_;
                uint32_t b1 = h_out - h_in * block_size_;
                uint64_t in_h_base = static_cast<uint64_t>(h_in) * in_stride_h_;
                uint32_t ub_row_base = tr * out_width_ * depth_;

                for (uint32_t b2 = 0u; b2 < 2u; ++b2) {
                    uint32_t first_w_out_c = b2 + block_size_ - crop_left_mod;
                    first_w_out_c -= (first_w_out_c / block_size_) * block_size_;
                    if (first_w_out_c >= out_width_) {
                        continue;
                    }

                    uint32_t count = (out_width_ - first_w_out_c + block_size_ - 1u) / block_size_;
                    uint32_t first_w_in = (first_w_out_c + crop_left_) / block_size_;
                    uint32_t input_batch = (b1 * block_size_ + b2) * new_batch_ + bi;
                    uint64_t in_base = static_cast<uint64_t>(input_batch) * in_stride_b_
                                     + in_h_base
                                     + static_cast<uint64_t>(first_w_in) * depth_;

                    read_params.blockCount = static_cast<uint16_t>(count);
                    DataCopy(ubuf[ub_row_base + first_w_out_c * depth_], xGM_[in_base], read_params);
                }
            }

            SetFlag<HardEvent::MTE2_MTE3>(event_id);
            WaitFlag<HardEvent::MTE2_MTE3>(event_id);

            write_params.blockLen =
                static_cast<uint16_t>(tile_rows * out_width_ * depth_bytes_32_);
            uint64_t out_base = static_cast<uint64_t>(bi) * out_stride_b_
                              + static_cast<uint64_t>(h_start_c) * out_stride_h_;
            DataCopy(yGM_[out_base], ubuf[0], write_params);
            SetFlag<HardEvent::MTE3_MTE2>(event_id);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
    }

    __aicore__ inline void ProcessPoint10Tile32Pipeline() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        constexpr uint32_t kTileRows = 32u;
        constexpr uint32_t kOutHeight = 2048u;
        constexpr uint32_t kOutWidth = 12u;
        constexpr uint32_t kDepth = 32u;
        constexpr uint32_t kDepthBlocks = 2u;
        constexpr uint32_t kInputWidth = 6u;
        constexpr uint32_t kStreamPixels = 6u;
        constexpr uint32_t kRowElems = kOutWidth * kDepth;
        constexpr uint32_t kTileElems = kTileRows * kRowElems;
        constexpr uint64_t kInStrideB = 1024ull * kInputWidth * kDepth;
        constexpr uint64_t kInStrideH = kInputWidth * kDepth;
        constexpr uint64_t kOutStrideB = kOutHeight * kRowElems;
        constexpr uint64_t kOutStrideH = kRowElems;

        DataCopyParams read_params;
        read_params.blockCount = kStreamPixels;
        read_params.blockLen = kDepthBlocks;
        read_params.srcStride = 0u;
        read_params.dstStride = kDepthBlocks;

        DataCopyParams write_params;
        write_params.blockCount = 1u;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        uint32_t block_dim = tile_rows_;
        if (block_dim == 0u) {
            return;
        }

        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);

        bool has_prev = false;
        uint32_t prev_slot = 0u;
        uint32_t prev_rows = 0u;
        uint64_t prev_out_base = 0u;

        for (uint32_t task_idx = GetBlockIdx(); task_idx < 256u; task_idx += block_dim) {
            uint32_t bi = task_idx >> 6u;
            if (bi >= 4u) {
                break;
            }
            uint32_t tile_in_batch = task_idx & 63u;
            uint32_t h_start_c = tile_in_batch * kTileRows;
            uint32_t tile_rows = MinU32(kTileRows, kOutHeight - h_start_c);

            uint32_t curr_slot = has_prev ? (prev_slot ^ 1u) : 0u;
            event_t curr_event = (curr_slot == 0u) ? event0 : event1;
            uint32_t curr_base = curr_slot * kTileElems;

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            for (uint32_t tr = 0u; tr < tile_rows; ++tr) {
                uint32_t h_out_c = h_start_c + tr;
                uint32_t h_in = h_out_c >> 1u;
                uint32_t b1 = h_out_c & 1u;
                uint64_t in_h_base = static_cast<uint64_t>(h_in) * kInStrideH;
                uint64_t input_batch_0 = static_cast<uint64_t>(b1 << 1u) * 4ull + bi;
                uint64_t in_base_0 = input_batch_0 * kInStrideB + in_h_base;
                uint32_t ub_row = curr_base + tr * kRowElems;

                DataCopy(ubuf[ub_row],
                         xGM_[in_base_0],
                         read_params);
                DataCopy(ubuf[ub_row + kDepth],
                         xGM_[in_base_0 + 4ull * kInStrideB],
                         read_params);
            }
            SetFlag<HardEvent::MTE2_MTE3>(curr_event);

            if (has_prev) {
                event_t prev_event = (prev_slot == 0u) ? event0 : event1;
                WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
                write_params.blockLen = static_cast<uint16_t>(prev_rows * kOutWidth * kDepthBlocks);
                DataCopy(yGM_[prev_out_base], ubuf[prev_slot * kTileElems], write_params);
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            has_prev = true;
            prev_slot = curr_slot;
            prev_rows = tile_rows;
            prev_out_base = static_cast<uint64_t>(bi) * kOutStrideB
                          + static_cast<uint64_t>(h_start_c) * kOutStrideH;
        }

        if (has_prev) {
            event_t prev_event = (prev_slot == 0u) ? event0 : event1;
            WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
            write_params.blockLen = static_cast<uint16_t>(prev_rows * kOutWidth * kDepthBlocks);
            DataCopy(yGM_[prev_out_base], ubuf[prev_slot * kTileElems], write_params);
            SetFlag<HardEvent::MTE3_MTE2>(prev_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
    }

    __aicore__ inline void ProcessPoint10Tile16Pipeline() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        constexpr uint32_t kTileRows = 16u;
        constexpr uint32_t kOutHeight = 2048u;
        constexpr uint32_t kOutWidth = 12u;
        constexpr uint32_t kDepth = 32u;
        constexpr uint32_t kDepthBlocks = 2u;
        constexpr uint32_t kInputWidth = 6u;
        constexpr uint32_t kStreamPixels = 6u;
        constexpr uint32_t kRowElems = kOutWidth * kDepth;
        constexpr uint32_t kTileElems = kTileRows * kRowElems;
        constexpr uint64_t kInStrideB = 1024ull * kInputWidth * kDepth;
        constexpr uint64_t kInStrideH = kInputWidth * kDepth;
        constexpr uint64_t kOutStrideB = kOutHeight * kRowElems;
        constexpr uint64_t kOutStrideH = kRowElems;

        DataCopyParams read_params;
        read_params.blockCount = kStreamPixels;
        read_params.blockLen = kDepthBlocks;
        read_params.srcStride = 0u;
        read_params.dstStride = kDepthBlocks;

        DataCopyParams write_params;
        write_params.blockCount = 1u;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        uint32_t block_dim = tile_rows_;
        if (block_dim == 0u) {
            return;
        }

        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);

        bool has_prev = false;
        uint32_t prev_slot = 0u;
        uint32_t prev_rows = 0u;
        uint64_t prev_out_base = 0u;

        for (uint32_t task_idx = GetBlockIdx(); task_idx < 512u; task_idx += block_dim) {
            uint32_t bi = task_idx >> 7u;
            if (bi >= 4u) {
                break;
            }
            uint32_t tile_in_batch = task_idx & 127u;
            uint32_t h_start_c = tile_in_batch * kTileRows;
            uint32_t tile_rows = MinU32(kTileRows, kOutHeight - h_start_c);

            uint32_t curr_slot = has_prev ? (prev_slot ^ 1u) : 0u;
            event_t curr_event = (curr_slot == 0u) ? event0 : event1;
            uint32_t curr_base = curr_slot * kTileElems;

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            for (uint32_t tr = 0u; tr < tile_rows; ++tr) {
                uint32_t h_out_c = h_start_c + tr;
                uint32_t h_in = h_out_c >> 1u;
                uint32_t b1 = h_out_c & 1u;
                uint64_t in_h_base = static_cast<uint64_t>(h_in) * kInStrideH;
                uint64_t input_batch_0 = static_cast<uint64_t>(b1 << 1u) * 4ull + bi;
                uint64_t in_base_0 = input_batch_0 * kInStrideB + in_h_base;
                uint32_t ub_row = curr_base + tr * kRowElems;

                DataCopy(ubuf[ub_row],
                         xGM_[in_base_0],
                         read_params);
                DataCopy(ubuf[ub_row + kDepth],
                         xGM_[in_base_0 + 4ull * kInStrideB],
                         read_params);
            }
            SetFlag<HardEvent::MTE2_MTE3>(curr_event);

            if (has_prev) {
                event_t prev_event = (prev_slot == 0u) ? event0 : event1;
                WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
                write_params.blockLen = static_cast<uint16_t>(prev_rows * kOutWidth * kDepthBlocks);
                DataCopy(yGM_[prev_out_base], ubuf[prev_slot * kTileElems], write_params);
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            has_prev = true;
            prev_slot = curr_slot;
            prev_rows = tile_rows;
            prev_out_base = static_cast<uint64_t>(bi) * kOutStrideB
                          + static_cast<uint64_t>(h_start_c) * kOutStrideH;
        }

        if (has_prev) {
            event_t prev_event = (prev_slot == 0u) ? event0 : event1;
            WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
            write_params.blockLen = static_cast<uint16_t>(prev_rows * kOutWidth * kDepthBlocks);
            DataCopy(yGM_[prev_out_base], ubuf[prev_slot * kTileElems], write_params);
            SetFlag<HardEvent::MTE3_MTE2>(prev_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
    }

    __aicore__ inline void ProcessPoint10Tile8Pipeline() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        constexpr uint32_t kTileRows = 8u;
        constexpr uint32_t kOutHeight = 2048u;
        constexpr uint32_t kOutWidth = 12u;
        constexpr uint32_t kDepth = 32u;
        constexpr uint32_t kDepthBlocks = 2u;
        constexpr uint32_t kInputWidth = 6u;
        constexpr uint32_t kStreamPixels = 6u;
        constexpr uint32_t kRowElems = kOutWidth * kDepth;
        constexpr uint32_t kTileElems = kTileRows * kRowElems;
        constexpr uint64_t kInStrideB = 1024ull * kInputWidth * kDepth;
        constexpr uint64_t kInStrideH = kInputWidth * kDepth;
        constexpr uint64_t kOutStrideB = kOutHeight * kRowElems;
        constexpr uint64_t kOutStrideH = kRowElems;

        DataCopyParams read_params;
        read_params.blockCount = kStreamPixels;
        read_params.blockLen = kDepthBlocks;
        read_params.srcStride = 0u;
        read_params.dstStride = kDepthBlocks;

        DataCopyParams write_params;
        write_params.blockCount = 1u;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        uint32_t block_dim = tile_rows_;
        if (block_dim == 0u) {
            return;
        }

        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);

        bool has_prev = false;
        uint32_t prev_slot = 0u;
        uint32_t prev_rows = 0u;
        uint64_t prev_out_base = 0u;

        for (uint32_t task_idx = GetBlockIdx(); task_idx < 1024u; task_idx += block_dim) {
            uint32_t bi = task_idx >> 8u;
            if (bi >= 4u) {
                break;
            }
            uint32_t tile_in_batch = task_idx & 255u;
            uint32_t h_start_c = tile_in_batch * kTileRows;
            uint32_t tile_rows = MinU32(kTileRows, kOutHeight - h_start_c);

            uint32_t curr_slot = has_prev ? (prev_slot ^ 1u) : 0u;
            event_t curr_event = (curr_slot == 0u) ? event0 : event1;
            uint32_t curr_base = curr_slot * kTileElems;

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            for (uint32_t tr = 0u; tr < tile_rows; ++tr) {
                uint32_t h_out_c = h_start_c + tr;
                uint32_t h_in = h_out_c >> 1u;
                uint32_t b1 = h_out_c & 1u;
                uint64_t in_h_base = static_cast<uint64_t>(h_in) * kInStrideH;
                uint64_t input_batch_0 = static_cast<uint64_t>(b1 << 1u) * 4ull + bi;
                uint64_t in_base_0 = input_batch_0 * kInStrideB + in_h_base;
                uint32_t ub_row = curr_base + tr * kRowElems;

                DataCopy(ubuf[ub_row],
                         xGM_[in_base_0],
                         read_params);
                DataCopy(ubuf[ub_row + kDepth],
                         xGM_[in_base_0 + 4ull * kInStrideB],
                         read_params);
            }
            SetFlag<HardEvent::MTE2_MTE3>(curr_event);

            if (has_prev) {
                event_t prev_event = (prev_slot == 0u) ? event0 : event1;
                WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
                write_params.blockLen = static_cast<uint16_t>(prev_rows * kOutWidth * kDepthBlocks);
                DataCopy(yGM_[prev_out_base], ubuf[prev_slot * kTileElems], write_params);
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            has_prev = true;
            prev_slot = curr_slot;
            prev_rows = tile_rows;
            prev_out_base = static_cast<uint64_t>(bi) * kOutStrideB
                          + static_cast<uint64_t>(h_start_c) * kOutStrideH;
        }

        if (has_prev) {
            event_t prev_event = (prev_slot == 0u) ? event0 : event1;
            WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
            write_params.blockLen = static_cast<uint16_t>(prev_rows * kOutWidth * kDepthBlocks);
            DataCopy(yGM_[prev_out_base], ubuf[prev_slot * kTileElems], write_params);
            SetFlag<HardEvent::MTE3_MTE2>(prev_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
    }

    __aicore__ inline void ProcessStrideTinyFixed() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr event_t event_id = EVENT_ID0;
        DataCopyParams read_params;
        read_params.blockCount = 1u;
        read_params.blockLen = static_cast<uint16_t>(depth_bytes_32_);
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyParams write_params;
        write_params.blockCount = 1u;
        write_params.blockLen =
            static_cast<uint16_t>(new_batch_ * out_height_ * out_width_ * depth_bytes_32_);
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        uint32_t crop_left_mod = crop_left_ & 1u;
        uint32_t b2_0 = crop_left_mod;
        uint32_t b2_1 = crop_left_mod ^ 1u;
        uint32_t w_in_0 = crop_left_ >> 1u;
        uint32_t w_in_1 = (crop_left_ + 1u) >> 1u;

        uint32_t h_out_0 = crop_top_;
        uint32_t h_in_0 = h_out_0 >> 1u;
        uint32_t b1_0 = h_out_0 & 1u;
        uint64_t in_h_base_0 = static_cast<uint64_t>(h_in_0) * in_stride_h_;
        uint32_t input_batch_00 = ((b1_0 << 1u) + b2_0) * new_batch_;
        uint32_t input_batch_01 = ((b1_0 << 1u) + b2_1) * new_batch_;

        uint32_t bi_1 = (out_height_ == 1u) ? 1u : 0u;
        uint32_t h_out_c_1 = (out_height_ == 1u) ? 0u : 1u;
        uint32_t h_out_1 = h_out_c_1 + crop_top_;
        uint32_t h_in_1 = h_out_1 >> 1u;
        uint32_t b1_1 = h_out_1 & 1u;
        uint64_t in_h_base_1 = static_cast<uint64_t>(h_in_1) * in_stride_h_;
        uint32_t input_batch_10 = ((b1_1 << 1u) + b2_0) * new_batch_ + bi_1;
        uint32_t input_batch_11 = ((b1_1 << 1u) + b2_1) * new_batch_ + bi_1;

        DataCopy(ubuf[0u],
                 xGM_[static_cast<uint64_t>(input_batch_00) * in_stride_b_
                    + in_h_base_0 + static_cast<uint64_t>(w_in_0) * depth_],
                 read_params);
        DataCopy(ubuf[depth_],
                 xGM_[static_cast<uint64_t>(input_batch_01) * in_stride_b_
                    + in_h_base_0 + static_cast<uint64_t>(w_in_1) * depth_],
                 read_params);
        DataCopy(ubuf[2u * depth_],
                 xGM_[static_cast<uint64_t>(input_batch_10) * in_stride_b_
                    + in_h_base_1 + static_cast<uint64_t>(w_in_0) * depth_],
                 read_params);
        DataCopy(ubuf[3u * depth_],
                 xGM_[static_cast<uint64_t>(input_batch_11) * in_stride_b_
                    + in_h_base_1 + static_cast<uint64_t>(w_in_1) * depth_],
                 read_params);

        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);
        DataCopy(yGM_[0u], ubuf[0u], write_params);
        SetFlag<HardEvent::MTE3_MTE2>(event_id);
        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
    }

    __aicore__ inline void ProcessPoint7Linear8Core() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr uint32_t kChunkElems = 8192u;
        constexpr uint16_t kChunkBlocks = 512u;

        DataCopyParams copy_params;
        copy_params.blockCount = 1u;
        copy_params.blockLen = kChunkBlocks;
        copy_params.srcStride = 0u;
        copy_params.dstStride = 0u;

        constexpr event_t event_id = EVENT_ID0;
        SetFlag<HardEvent::MTE3_MTE2>(event_id);

        for (uint32_t r = 0u; r < my_count_; ++r) {
            uint32_t chunk_idx = my_start_ + r;
            if (chunk_idx >= 8u) {
                break;
            }
            uint64_t offset = static_cast<uint64_t>(chunk_idx) * kChunkElems;
            WaitFlag<HardEvent::MTE3_MTE2>(event_id);
            DataCopy(ubuf[0u], xGM_[offset], copy_params);
            SetFlag<HardEvent::MTE2_MTE3>(event_id);
            WaitFlag<HardEvent::MTE2_MTE3>(event_id);
            DataCopy(yGM_[offset], ubuf[0u], copy_params);
            SetFlag<HardEvent::MTE3_MTE2>(event_id);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
    }

    __aicore__ inline void ProcessPoint7Linear16Core() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr uint32_t kChunkElems = 4096u;
        constexpr uint16_t kChunkBlocks = 256u;

        DataCopyParams copy_params;
        copy_params.blockCount = 1u;
        copy_params.blockLen = kChunkBlocks;
        copy_params.srcStride = 0u;
        copy_params.dstStride = 0u;

        constexpr event_t event_id = EVENT_ID0;
        SetFlag<HardEvent::MTE3_MTE2>(event_id);

        for (uint32_t r = 0u; r < my_count_; ++r) {
            uint32_t chunk_idx = my_start_ + r;
            if (chunk_idx >= 16u) {
                break;
            }
            uint64_t offset = static_cast<uint64_t>(chunk_idx) * kChunkElems;
            WaitFlag<HardEvent::MTE3_MTE2>(event_id);
            DataCopy(ubuf[0u], xGM_[offset], copy_params);
            SetFlag<HardEvent::MTE2_MTE3>(event_id);
            WaitFlag<HardEvent::MTE2_MTE3>(event_id);
            DataCopy(yGM_[offset], ubuf[0u], copy_params);
            SetFlag<HardEvent::MTE3_MTE2>(event_id);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
    }

    __aicore__ inline void ProcessPoint7Linear4Core() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr uint32_t kChunkElems = 16384u;
        constexpr uint16_t kChunkBlocks = 1024u;

        DataCopyParams copy_params;
        copy_params.blockCount = 1u;
        copy_params.blockLen = kChunkBlocks;
        copy_params.srcStride = 0u;
        copy_params.dstStride = 0u;

        constexpr event_t event_id = EVENT_ID0;
        SetFlag<HardEvent::MTE3_MTE2>(event_id);

        for (uint32_t r = 0u; r < my_count_; ++r) {
            uint32_t chunk_idx = my_start_ + r;
            if (chunk_idx >= 4u) {
                break;
            }
            uint64_t offset = static_cast<uint64_t>(chunk_idx) * kChunkElems;
            WaitFlag<HardEvent::MTE3_MTE2>(event_id);
            DataCopy(ubuf[0u], xGM_[offset], copy_params);
            SetFlag<HardEvent::MTE2_MTE3>(event_id);
            WaitFlag<HardEvent::MTE2_MTE3>(event_id);
            DataCopy(yGM_[offset], ubuf[0u], copy_params);
            SetFlag<HardEvent::MTE3_MTE2>(event_id);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
    }

    __aicore__ inline void ProcessPoint7Linear8CorePipe2() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr uint32_t kCoreElems = 8192u;
        constexpr uint32_t kHalfElems = 4096u;
        constexpr uint16_t kHalfBlocks = 256u;

        DataCopyParams copy_params;
        copy_params.blockCount = 1u;
        copy_params.blockLen = kHalfBlocks;
        copy_params.srcStride = 0u;
        copy_params.dstStride = 0u;

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);

        for (uint32_t r = 0u; r < my_count_; ++r) {
            uint32_t chunk_idx = my_start_ + r;
            if (chunk_idx >= 8u) {
                break;
            }

            uint64_t base_offset = static_cast<uint64_t>(chunk_idx) * kCoreElems;
            WaitFlag<HardEvent::MTE3_MTE2>(event0);
            DataCopy(ubuf[0u], xGM_[base_offset], copy_params);
            SetFlag<HardEvent::MTE2_MTE3>(event0);

            WaitFlag<HardEvent::MTE3_MTE2>(event1);
            DataCopy(ubuf[kHalfElems], xGM_[base_offset + kHalfElems], copy_params);
            SetFlag<HardEvent::MTE2_MTE3>(event1);

            WaitFlag<HardEvent::MTE2_MTE3>(event0);
            DataCopy(yGM_[base_offset], ubuf[0u], copy_params);
            SetFlag<HardEvent::MTE3_MTE2>(event0);

            WaitFlag<HardEvent::MTE2_MTE3>(event1);
            DataCopy(yGM_[base_offset + kHalfElems], ubuf[kHalfElems], copy_params);
            SetFlag<HardEvent::MTE3_MTE2>(event1);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
    }

    __aicore__ inline void ProcessPoint7Linear8CoreFast() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr uint32_t kChunkElems = 8192u;
        constexpr uint16_t kChunkBlocks = 512u;

        uint32_t chunk_idx = GetBlockIdx();
        if (chunk_idx >= 8u) {
            return;
        }

        DataCopyParams copy_params;
        copy_params.blockCount = 1u;
        copy_params.blockLen = kChunkBlocks;
        copy_params.srcStride = 0u;
        copy_params.dstStride = 0u;

        uint64_t offset = static_cast<uint64_t>(chunk_idx) * kChunkElems;
        constexpr event_t event_id = EVENT_ID0;

        DataCopy(ubuf[0u], xGM_[offset], copy_params);
        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);
        DataCopy(yGM_[offset], ubuf[0u], copy_params);
        SetFlag<HardEvent::MTE3_MTE2>(event_id);
        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
    }

    __aicore__ inline void ProcessPoint7Linear8CoreFastNoWait() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr uint32_t kChunkElems = 8192u;
        constexpr uint16_t kChunkBlocks = 512u;

        uint32_t chunk_idx = GetBlockIdx();
        if (chunk_idx >= 8u) {
            return;
        }

        DataCopyParams copy_params;
        copy_params.blockCount = 1u;
        copy_params.blockLen = kChunkBlocks;
        copy_params.srcStride = 0u;
        copy_params.dstStride = 0u;

        uint64_t offset = static_cast<uint64_t>(chunk_idx) * kChunkElems;
        constexpr event_t event_id = EVENT_ID0;

        DataCopy(ubuf[0u], xGM_[offset], copy_params);
        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);
        DataCopy(yGM_[offset], ubuf[0u], copy_params);
    }

    __aicore__ inline void ProcessPoint7Linear8CoreCountCopy() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr uint32_t kChunkElems = 8192u;

        uint32_t chunk_idx = GetBlockIdx();
        if (chunk_idx >= 8u) {
            return;
        }

        uint64_t offset = static_cast<uint64_t>(chunk_idx) * kChunkElems;
        constexpr event_t event_id = EVENT_ID0;

        DataCopy(ubuf[0u], xGM_[offset], kChunkElems);
        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);
        DataCopy(yGM_[offset], ubuf[0u], kChunkElems);
        SetFlag<HardEvent::MTE3_MTE2>(event_id);
        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
    }

    __aicore__ inline void BuildPoint9Gather240OffsetsPipeline(LocalTensor<DT_X> ubuf) {
        constexpr uint32_t kDepth = 64u;
        constexpr uint32_t kStreams = 4u;
        constexpr uint32_t kGroups = 60u;
        constexpr uint32_t kGroupElems = kStreams * kDepth;
        constexpr uint32_t kStreamElems = kGroups * kDepth;
        constexpr uint32_t kSrcElems = kStreams * kStreamElems;
        constexpr uint32_t kTileElems = kGroupElems * kGroups;
        constexpr uint32_t kSlotElems = kSrcElems + kTileElems;
        constexpr uint32_t kOffsetOffset = 2u * kSlotElems;
        LocalTensor<int32_t> offsets_i32 = ubuf[kOffsetOffset].template ReinterpretCast<int32_t>();

        for (uint32_t r = 0u; r < kStreams; ++r) {
            for (uint32_t c = 0u; c < kDepth; ++c) {
                offsets_i32.SetValue(r * kDepth + c,
                                     static_cast<int32_t>((r * kStreamElems + c) * sizeof(DT_X)));
            }
        }
        PipeBarrier<PIPE_ALL>();

        uint32_t built_groups = 1u;
        while (built_groups < kGroups) {
            uint32_t copy_groups = MinU32(built_groups, kGroups - built_groups);
            uint32_t dst_index = built_groups * kGroupElems;
            Adds(offsets_i32[dst_index], offsets_i32[0u],
                 static_cast<int32_t>(built_groups * kDepth * sizeof(DT_X)),
                 copy_groups * kGroupElems);
            PipeBarrier<PIPE_V>();
            built_groups += copy_groups;
        }
    }

    __aicore__ inline void ProcessPoint9GatherTile240Pipeline() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        constexpr uint32_t kDepth = 64u;
        constexpr uint32_t kDepthBlocks = 4u;
        constexpr uint32_t kTilePixels = 240u;
        constexpr uint32_t kTileGroups = 60u;
        constexpr uint32_t kTileElems = kTilePixels * kDepth;
        constexpr uint32_t kStreamElems = kTileGroups * kDepth;
        constexpr uint32_t kSrcElems = 4u * kStreamElems;
        constexpr uint32_t kDstInSlotOffset = kSrcElems;
        constexpr uint32_t kSlotElems = kSrcElems + kTileElems;
        constexpr uint32_t kOffsetOffset = 2u * kSlotElems;
        constexpr uint64_t kInStrideB = 327680ull;
        constexpr uint64_t kInStrideH = 32768ull;
        constexpr uint64_t kOutStrideH = 98240ull;

        BuildPoint9Gather240OffsetsPipeline(ubuf);
        LocalTensor<uint32_t> offsets = ubuf[kOffsetOffset].template ReinterpretCast<uint32_t>();

        DataCopyParams read_params;
        read_params.blockCount = 1u;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyParams write_params;
        write_params.blockCount = 1u;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        uint32_t block_dim = tile_rows_;
        if (block_dim == 0u) {
            return;
        }

        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);

        bool has_prev = false;
        uint32_t prev_slot = 0u;
        uint32_t prev_pixels = 0u;
        uint32_t prev_gather_elems = 0u;
        uint64_t prev_out_base = 0u;

        for (uint32_t task_idx = GetBlockIdx(); task_idx < 280u; task_idx += block_dim) {
            uint32_t row_idx = task_idx / 7u;
            if (row_idx >= 40u) {
                break;
            }
            uint32_t tile_idx = task_idx - row_idx * 7u;
            uint32_t tile_pixels = (tile_idx < 6u) ? kTilePixels : 95u;
            uint32_t count012 = (tile_idx < 6u) ? kTileGroups : 24u;
            uint32_t count3 = (tile_idx < 6u) ? kTileGroups : 23u;
            uint32_t gather_elems = tile_pixels * kDepth;
            uint32_t in012 = 128u + tile_idx * kTileGroups;
            uint32_t in3 = 129u + tile_idx * kTileGroups;

            uint32_t h_in = row_idx >> 2u;
            uint32_t b1 = row_idx & 3u;
            uint64_t in_batch_base = static_cast<uint64_t>(b1 << 2u) * kInStrideB
                                   + static_cast<uint64_t>(h_in) * kInStrideH;
            uint64_t out_base = static_cast<uint64_t>(row_idx) * kOutStrideH
                              + static_cast<uint64_t>(tile_idx) * kTileElems;

            uint32_t curr_slot = has_prev ? (prev_slot ^ 1u) : 0u;
            event_t curr_event = (curr_slot == 0u) ? event0 : event1;
            uint32_t curr_base = curr_slot * kSlotElems;

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            read_params.blockLen = static_cast<uint16_t>(count012 * kDepthBlocks);
            DataCopy(ubuf[curr_base + 0u * kStreamElems],
                     xGM_[in_batch_base + kInStrideB + static_cast<uint64_t>(in012) * kDepth],
                     read_params);
            DataCopy(ubuf[curr_base + 1u * kStreamElems],
                     xGM_[in_batch_base + 2u * kInStrideB + static_cast<uint64_t>(in012) * kDepth],
                     read_params);
            DataCopy(ubuf[curr_base + 2u * kStreamElems],
                     xGM_[in_batch_base + 3u * kInStrideB + static_cast<uint64_t>(in012) * kDepth],
                     read_params);
            read_params.blockLen = static_cast<uint16_t>(count3 * kDepthBlocks);
            DataCopy(ubuf[curr_base + 3u * kStreamElems],
                     xGM_[in_batch_base + static_cast<uint64_t>(in3) * kDepth],
                     read_params);
            SetFlag<HardEvent::MTE2_V>(curr_event);

            if (has_prev) {
                event_t prev_event = (prev_slot == 0u) ? event0 : event1;
                uint32_t prev_base = prev_slot * kSlotElems;
                uint32_t prev_dst = prev_base + kDstInSlotOffset;
                WaitFlag<HardEvent::MTE2_V>(prev_event);
                Gather(ubuf[prev_dst], ubuf[prev_base], offsets, 0u, prev_gather_elems);
                SetFlag<HardEvent::V_MTE3>(prev_event);
                WaitFlag<HardEvent::V_MTE3>(prev_event);
                write_params.blockLen = static_cast<uint16_t>(prev_pixels * kDepthBlocks);
                DataCopy(yGM_[prev_out_base], ubuf[prev_dst], write_params);
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            has_prev = true;
            prev_slot = curr_slot;
            prev_pixels = tile_pixels;
            prev_gather_elems = gather_elems;
            prev_out_base = out_base;
        }

        if (has_prev) {
            event_t prev_event = (prev_slot == 0u) ? event0 : event1;
            uint32_t prev_base = prev_slot * kSlotElems;
            uint32_t prev_dst = prev_base + kDstInSlotOffset;
            WaitFlag<HardEvent::MTE2_V>(prev_event);
            Gather(ubuf[prev_dst], ubuf[prev_base], offsets, 0u, prev_gather_elems);
            SetFlag<HardEvent::V_MTE3>(prev_event);
            WaitFlag<HardEvent::V_MTE3>(prev_event);
            write_params.blockLen = static_cast<uint16_t>(prev_pixels * kDepthBlocks);
            DataCopy(yGM_[prev_out_base], ubuf[prev_dst], write_params);
            SetFlag<HardEvent::MTE3_MTE2>(prev_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
    }

    __aicore__ inline void ProcessStrideRowsPipeline() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        const uint32_t row_elems = out_width_ * depth_;
        uint16_t block_len_32 = static_cast<uint16_t>(depth_bytes_32_);

        DataCopyParams read_params;
        read_params.blockLen = block_len_32;
        read_params.srcStride = 0u;
        read_params.dstStride = block_len_32;

        DataCopyParams write_params;
        write_params.blockCount = 1u;
        write_params.blockLen = static_cast<uint16_t>(out_width_ * depth_bytes_32_);
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        const uint32_t crop_left_mod = crop_left_ & 1u;
        const uint32_t first_w_out_0 = crop_left_mod;
        const uint32_t first_w_out_1 = crop_left_mod ^ 1u;
        const uint32_t first_w_in_0 = (first_w_out_0 + crop_left_) >> 1u;
        const uint32_t first_w_in_1 = (first_w_out_1 + crop_left_) >> 1u;
        const uint32_t stream_count = out_width_ >> 1u;
        const uint64_t first_in_offset_0 = static_cast<uint64_t>(first_w_in_0) * depth_;
        const uint64_t first_in_offset_1 = static_cast<uint64_t>(first_w_in_1) * depth_;
        read_params.blockCount = static_cast<uint16_t>(stream_count);
        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);

        bool has_prev = false;
        uint32_t prev_slot = 0u;
        uint64_t prev_out_base = 0u;

        for (uint32_t r = 0u; r < my_count_; ++r) {
            uint32_t row_idx = my_start_ + r;
            if (row_idx >= total_units_) {
                break;
            }

            uint32_t bi = row_idx / out_height_;
            uint32_t h_out_c = row_idx - bi * out_height_;
            uint32_t h_out = h_out_c + crop_top_;
            uint32_t h_in = h_out >> 1u;
            uint32_t b1 = h_out & 1u;
            uint32_t curr_slot = has_prev ? (prev_slot ^ 1u) : 0u;
            event_t curr_event = (curr_slot == 0u) ? event0 : event1;
            uint32_t ub_base = curr_slot * row_elems;

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            uint32_t input_batch_0 = (b1 << 1u) * new_batch_ + bi;
            uint64_t in_base_0 = static_cast<uint64_t>(input_batch_0) * in_stride_b_
                               + static_cast<uint64_t>(h_in) * in_stride_h_;
            DataCopy(ubuf[ub_base + first_w_out_0 * depth_],
                     xGM_[in_base_0 + first_in_offset_0], read_params);
            DataCopy(ubuf[ub_base + first_w_out_1 * depth_],
                     xGM_[in_base_0 + in_stride_b_ * new_batch_ + first_in_offset_1], read_params);
            SetFlag<HardEvent::MTE2_MTE3>(curr_event);

            if (has_prev) {
                event_t prev_event = (prev_slot == 0u) ? event0 : event1;
                WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
                DataCopy(yGM_[prev_out_base], ubuf[prev_slot * row_elems], write_params);
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            has_prev = true;
            prev_slot = curr_slot;
            prev_out_base = static_cast<uint64_t>(bi) * out_stride_b_
                          + static_cast<uint64_t>(h_out_c) * out_stride_h_;
        }

        if (has_prev) {
            event_t prev_event = (prev_slot == 0u) ? event0 : event1;
            WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
            DataCopy(yGM_[prev_out_base], ubuf[prev_slot * row_elems], write_params);
            SetFlag<HardEvent::MTE3_MTE2>(prev_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
    }

    __aicore__ inline void ProcessScalarPoint5Gather() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr uint32_t fixed_depth = 65u;
        constexpr uint32_t fixed_depth_bytes = 130u;
        constexpr uint32_t stream_pixels = 127u;
        constexpr uint32_t stream_elems = stream_pixels * fixed_depth;
        constexpr uint32_t stream_bytes = stream_elems * sizeof(DT_X);
        constexpr uint32_t aligned_stream_elems = 8256u;
        constexpr uint32_t row_elems = 254u * fixed_depth;
        constexpr uint32_t row_bytes = row_elems * sizeof(DT_X);
        constexpr uint32_t input_stride_h = 128u * fixed_depth;
        constexpr uint32_t input_stride_b = 128u * input_stride_h;
        constexpr uint32_t output_stride_h = row_elems;
        constexpr uint32_t dst_offset = 2u * aligned_stream_elems;
        constexpr uint32_t offset_table_offset = dst_offset + 16512u;

        LocalTensor<DT_X> src = ubuf[0u];
        LocalTensor<DT_X> dst = ubuf[dst_offset];
        LocalTensor<int32_t> offsets_i32 = ubuf[offset_table_offset].template ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> offsets = ubuf[offset_table_offset].template ReinterpretCast<uint32_t>();

        // Build offsets for four pixel pairs with scalar stores, then replicate aligned groups in V.
        for (uint32_t p = 0u; p < 4u; ++p) {
            for (uint32_t c = 0u; c < fixed_depth; ++c) {
                offsets_i32.SetValue(p * 2u * fixed_depth + c,
                                     static_cast<int32_t>(p * fixed_depth_bytes + c * sizeof(DT_X)));
                offsets_i32.SetValue(p * 2u * fixed_depth + fixed_depth + c,
                                     static_cast<int32_t>(stream_bytes + 2u +
                                                          p * fixed_depth_bytes + c * sizeof(DT_X)));
            }
        }
        PipeBarrier<PIPE_ALL>();

        uint32_t built_pairs = 4u;
        while (built_pairs < 124u) {
            uint32_t copy_pairs = MinU32(built_pairs, 124u - built_pairs);
            uint32_t dst_index = built_pairs * 2u * fixed_depth;
            Adds(offsets_i32[dst_index], offsets_i32[0u],
                 static_cast<int32_t>(built_pairs * fixed_depth_bytes),
                 copy_pairs * 2u * fixed_depth);
            PipeBarrier<PIPE_V>();
            built_pairs += copy_pairs;
        }
        Adds(offsets_i32[124u * 2u * fixed_depth], offsets_i32[0u],
             static_cast<int32_t>(124u * fixed_depth_bytes),
             3u * 2u * fixed_depth);
        PipeBarrier<PIPE_V>();

        DataCopyExtParams read_params;
        read_params.blockCount = 1u;
        read_params.blockLen = stream_bytes;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyExtParams write_params;
        write_params.blockCount = 1u;
        write_params.blockLen = row_bytes;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        DataCopyPadExtParams<DT_X> pad_params;
        pad_params.isPad = false;
        pad_params.leftPadding = 0u;
        pad_params.rightPadding = 1u;
        pad_params.paddingValue = static_cast<DT_X>(0);

        constexpr event_t event_id = EVENT_ID0;
        SetFlag<HardEvent::MTE3_MTE2>(event_id);
        for (uint32_t r = 0u; r < my_count_; ++r) {
            uint32_t row_idx = my_start_ + r;
            if (row_idx >= total_units_) {
                break;
            }

            uint32_t h_out = row_idx + 1u;
            uint32_t h_in = h_out >> 1u;
            uint32_t b1 = h_out & 1u;
            uint32_t even_batch = (b1 << 1u) + 1u;
            uint32_t odd_batch = b1 << 1u;
            uint64_t even_in_offset = static_cast<uint64_t>(even_batch) * input_stride_b
                                    + static_cast<uint64_t>(h_in) * input_stride_h;
            uint64_t odd_in_offset = static_cast<uint64_t>(odd_batch) * input_stride_b
                                   + static_cast<uint64_t>(h_in) * input_stride_h
                                   + fixed_depth;

            WaitFlag<HardEvent::MTE3_MTE2>(event_id);
            DataCopyPad(src[0u], xGM_[even_in_offset], read_params, pad_params);
            DataCopyPad(src[aligned_stream_elems], xGM_[odd_in_offset], read_params, pad_params);
            SetFlag<HardEvent::MTE2_V>(event_id);
            WaitFlag<HardEvent::MTE2_V>(event_id);

            Gather(dst, src, offsets, 0u, row_elems);
            SetFlag<HardEvent::V_MTE3>(event_id);
            WaitFlag<HardEvent::V_MTE3>(event_id);
            DataCopyPad(yGM_[static_cast<uint64_t>(row_idx) * output_stride_h], dst, write_params);
            SetFlag<HardEvent::MTE3_MTE2>(event_id);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
    }

    __aicore__ inline void ProcessScalarPoint5InterleaveReadPipeline() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr uint32_t fixed_depth = 65u;
        constexpr uint32_t fixed_depth_bytes = 130u;
        constexpr uint32_t stream_pixels = 127u;
        constexpr uint32_t row_elems = 254u * fixed_depth;
        constexpr uint32_t row_bytes = row_elems * sizeof(DT_X);
        constexpr uint32_t row_aligned_elems = 16512u;
        constexpr uint32_t input_stride_h = 128u * fixed_depth;
        constexpr uint32_t input_stride_b = 128u * input_stride_h;
        constexpr uint32_t output_stride_h = row_elems;

        DataCopyExtParams read_params;
        read_params.blockCount = static_cast<uint16_t>(stream_pixels);
        read_params.blockLen = fixed_depth_bytes;
        read_params.srcStride = 0u;
        read_params.dstStride = fixed_depth_bytes;

        DataCopyExtParams write_params;
        write_params.blockCount = 1u;
        write_params.blockLen = row_bytes;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        DataCopyPadExtParams<DT_X> pad_params;
        pad_params.isPad = false;
        pad_params.leftPadding = 0u;
        pad_params.rightPadding = 0u;
        pad_params.paddingValue = static_cast<DT_X>(0);

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);

        bool has_prev = false;
        uint32_t prev_slot = 0u;
        uint64_t prev_out_offset = 0u;

        for (uint32_t r = 0u; r < my_count_; ++r) {
            uint32_t row_idx = my_start_ + r;
            if (row_idx >= total_units_) {
                break;
            }

            uint32_t curr_slot = has_prev ? (prev_slot ^ 1u) : 0u;
            event_t curr_event = (curr_slot == 0u) ? event0 : event1;
            uint32_t curr_base = curr_slot * row_aligned_elems;

            uint32_t h_out = row_idx + 1u;
            uint32_t h_in = h_out >> 1u;
            uint32_t b1 = h_out & 1u;
            uint32_t even_batch = (b1 << 1u) + 1u;
            uint32_t odd_batch = b1 << 1u;
            uint64_t even_in_offset = static_cast<uint64_t>(even_batch) * input_stride_b
                                    + static_cast<uint64_t>(h_in) * input_stride_h;
            uint64_t odd_in_offset = static_cast<uint64_t>(odd_batch) * input_stride_b
                                   + static_cast<uint64_t>(h_in) * input_stride_h
                                   + fixed_depth;

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            DataCopyPad(ubuf[curr_base], xGM_[even_in_offset], read_params, pad_params);
            DataCopyPad(ubuf[curr_base + fixed_depth], xGM_[odd_in_offset], read_params, pad_params);
            SetFlag<HardEvent::MTE2_MTE3>(curr_event);

            if (has_prev) {
                event_t prev_event = (prev_slot == 0u) ? event0 : event1;
                WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
                DataCopyPad(yGM_[prev_out_offset], ubuf[prev_slot * row_aligned_elems], write_params);
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            has_prev = true;
            prev_slot = curr_slot;
            prev_out_offset = static_cast<uint64_t>(row_idx) * output_stride_h;
        }

        if (has_prev) {
            event_t prev_event = (prev_slot == 0u) ? event0 : event1;
            WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
            DataCopyPad(yGM_[prev_out_offset], ubuf[prev_slot * row_aligned_elems], write_params);
            SetFlag<HardEvent::MTE3_MTE2>(prev_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
    }

    __aicore__ inline void ProcessScalarPoint5Gather64Pipeline() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr uint32_t fixed_depth = 65u;
        constexpr uint32_t fixed_depth_bytes = 130u;
        constexpr uint32_t stream_pixels = 127u;
        constexpr uint32_t stream_elems = stream_pixels * fixed_depth;
        constexpr uint32_t stream_bytes = stream_elems * sizeof(DT_X);
        constexpr uint32_t aligned_stream_elems = 8256u;
        constexpr uint32_t src_slot_elems = 2u * aligned_stream_elems;
        constexpr uint32_t row_elems = 254u * fixed_depth;
        constexpr uint32_t row_bytes = row_elems * sizeof(DT_X);
        constexpr uint32_t row_aligned_elems = 16512u;
        constexpr uint32_t input_stride_h = 128u * fixed_depth;
        constexpr uint32_t input_stride_b = 128u * input_stride_h;
        constexpr uint32_t output_stride_h = row_elems;
        constexpr uint32_t dst_offset = src_slot_elems;
        constexpr uint32_t slot_elems = dst_offset + row_aligned_elems;
        constexpr uint32_t offset_table_offset = 2u * slot_elems;
        constexpr uint32_t gather_pairs = 64u;
        constexpr uint32_t gather_elems = gather_pairs * 2u * fixed_depth;
        constexpr uint32_t second_pairs = 63u;
        constexpr uint32_t second_src_shift = gather_pairs * fixed_depth;
        constexpr uint32_t second_dst_shift = gather_pairs * 2u * fixed_depth;

        LocalTensor<int32_t> offsets_i32 = ubuf[offset_table_offset].template ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> offsets = ubuf[offset_table_offset].template ReinterpretCast<uint32_t>();

        for (uint32_t p = 0u; p < gather_pairs; ++p) {
            for (uint32_t c = 0u; c < fixed_depth; ++c) {
                offsets_i32.SetValue(p * 2u * fixed_depth + c,
                                     static_cast<int32_t>(p * fixed_depth_bytes + c * sizeof(DT_X)));
                offsets_i32.SetValue(p * 2u * fixed_depth + fixed_depth + c,
                                     static_cast<int32_t>(stream_bytes + 2u +
                                                          p * fixed_depth_bytes + c * sizeof(DT_X)));
            }
        }
        PipeBarrier<PIPE_ALL>();

        DataCopyExtParams read_params;
        read_params.blockCount = 1u;
        read_params.blockLen = stream_bytes;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyExtParams write_params;
        write_params.blockCount = 1u;
        write_params.blockLen = row_bytes;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        DataCopyPadExtParams<DT_X> pad_params;
        pad_params.isPad = false;
        pad_params.leftPadding = 0u;
        pad_params.rightPadding = 1u;
        pad_params.paddingValue = static_cast<DT_X>(0);

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);

        bool has_prev = false;
        uint32_t prev_slot = 0u;
        uint64_t prev_out_offset = 0u;

        for (uint32_t r = 0u; r < my_count_; ++r) {
            uint32_t row_idx = my_start_ + r;
            if (row_idx >= total_units_) {
                break;
            }

            uint32_t curr_slot = has_prev ? (prev_slot ^ 1u) : 0u;
            event_t curr_event = (curr_slot == 0u) ? event0 : event1;
            uint32_t curr_base = curr_slot * slot_elems;
            LocalTensor<DT_X> src = ubuf[curr_base];
            LocalTensor<DT_X> dst = ubuf[curr_base + dst_offset];

            uint32_t h_out = row_idx + 1u;
            uint32_t h_in = h_out >> 1u;
            uint32_t b1 = h_out & 1u;
            uint32_t even_batch = (b1 << 1u) + 1u;
            uint32_t odd_batch = b1 << 1u;
            uint64_t even_in_offset = static_cast<uint64_t>(even_batch) * input_stride_b
                                    + static_cast<uint64_t>(h_in) * input_stride_h;
            uint64_t odd_in_offset = static_cast<uint64_t>(odd_batch) * input_stride_b
                                   + static_cast<uint64_t>(h_in) * input_stride_h
                                   + fixed_depth;
            uint64_t out_offset = static_cast<uint64_t>(row_idx) * output_stride_h;

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            DataCopyPad(src[0u], xGM_[even_in_offset], read_params, pad_params);
            DataCopyPad(src[aligned_stream_elems], xGM_[odd_in_offset], read_params, pad_params);
            SetFlag<HardEvent::MTE2_V>(curr_event);

            if (has_prev) {
                event_t prev_event = (prev_slot == 0u) ? event0 : event1;
                WaitFlag<HardEvent::V_MTE3>(prev_event);
                DataCopyPad(yGM_[prev_out_offset],
                            ubuf[prev_slot * slot_elems + dst_offset], write_params);
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            WaitFlag<HardEvent::MTE2_V>(curr_event);
            Gather(dst, src, offsets, 0u, gather_elems);
            Gather(dst[second_dst_shift], src[second_src_shift], offsets, 0u,
                   second_pairs * 2u * fixed_depth);
            SetFlag<HardEvent::V_MTE3>(curr_event);

            has_prev = true;
            prev_slot = curr_slot;
            prev_out_offset = out_offset;
        }

        if (has_prev) {
            event_t prev_event = (prev_slot == 0u) ? event0 : event1;
            WaitFlag<HardEvent::V_MTE3>(prev_event);
            DataCopyPad(yGM_[prev_out_offset],
                        ubuf[prev_slot * slot_elems + dst_offset], write_params);
            SetFlag<HardEvent::MTE3_MTE2>(prev_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
    }

    __aicore__ inline void ProcessScalarPoint5GatherPipelineSafe() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr uint32_t fixed_depth = 65u;
        constexpr uint32_t fixed_depth_bytes = 130u;
        constexpr uint32_t stream_pixels = 127u;
        constexpr uint32_t stream_elems = stream_pixels * fixed_depth;
        constexpr uint32_t stream_bytes = stream_elems * sizeof(DT_X);
        constexpr uint32_t aligned_stream_elems = 8256u;
        constexpr uint32_t src_slot_elems = 2u * aligned_stream_elems;
        constexpr uint32_t row_elems = 254u * fixed_depth;
        constexpr uint32_t row_bytes = row_elems * sizeof(DT_X);
        constexpr uint32_t row_aligned_elems = 16512u;
        constexpr uint32_t input_stride_h = 128u * fixed_depth;
        constexpr uint32_t input_stride_b = 128u * input_stride_h;
        constexpr uint32_t output_stride_h = row_elems;
        constexpr uint32_t dst0_offset = src_slot_elems;
        constexpr uint32_t dst1_offset = dst0_offset + row_aligned_elems;
        constexpr uint32_t offset_table_offset = dst1_offset + row_aligned_elems;

        LocalTensor<int32_t> offsets_i32 = ubuf[offset_table_offset].template ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> offsets = ubuf[offset_table_offset].template ReinterpretCast<uint32_t>();

        for (uint32_t p = 0u; p < 4u; ++p) {
            for (uint32_t c = 0u; c < fixed_depth; ++c) {
                offsets_i32.SetValue(p * 2u * fixed_depth + c,
                                     static_cast<int32_t>(p * fixed_depth_bytes + c * sizeof(DT_X)));
                offsets_i32.SetValue(p * 2u * fixed_depth + fixed_depth + c,
                                     static_cast<int32_t>(stream_bytes + 2u +
                                                          p * fixed_depth_bytes + c * sizeof(DT_X)));
            }
        }
        PipeBarrier<PIPE_ALL>();

        uint32_t built_pairs = 4u;
        while (built_pairs < 124u) {
            uint32_t copy_pairs = MinU32(built_pairs, 124u - built_pairs);
            uint32_t dst_index = built_pairs * 2u * fixed_depth;
            Adds(offsets_i32[dst_index], offsets_i32[0u],
                 static_cast<int32_t>(built_pairs * fixed_depth_bytes),
                 copy_pairs * 2u * fixed_depth);
            PipeBarrier<PIPE_V>();
            built_pairs += copy_pairs;
        }
        Adds(offsets_i32[124u * 2u * fixed_depth], offsets_i32[0u],
             static_cast<int32_t>(124u * fixed_depth_bytes),
             3u * 2u * fixed_depth);
        PipeBarrier<PIPE_V>();

        DataCopyExtParams read_params;
        read_params.blockCount = 1u;
        read_params.blockLen = stream_bytes;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyExtParams write_params;
        write_params.blockCount = 1u;
        write_params.blockLen = row_bytes;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        DataCopyPadExtParams<DT_X> pad_params;
        pad_params.isPad = false;
        pad_params.leftPadding = 0u;
        pad_params.rightPadding = 1u;
        pad_params.paddingValue = static_cast<DT_X>(0);

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);

        bool has_prev = false;
        uint32_t prev_slot = 0u;
        uint64_t prev_out_offset = 0u;

        for (uint32_t r = 0u; r < my_count_; ++r) {
            uint32_t row_idx = my_start_ + r;
            if (row_idx >= total_units_) {
                break;
            }

            uint32_t curr_slot = has_prev ? (prev_slot ^ 1u) : 0u;
            event_t curr_event = (curr_slot == 0u) ? event0 : event1;

            if (has_prev) {
                event_t prev_event = (prev_slot == 0u) ? event0 : event1;
                uint32_t prev_dst_offset = (prev_slot == 0u) ? dst0_offset : dst1_offset;
                WaitFlag<HardEvent::V_MTE3>(prev_event);
                DataCopyPad(yGM_[prev_out_offset], ubuf[prev_dst_offset], write_params);
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            uint32_t h_out = row_idx + 1u;
            uint32_t h_in = h_out >> 1u;
            uint32_t b1 = h_out & 1u;
            uint32_t even_batch = (b1 << 1u) + 1u;
            uint32_t odd_batch = b1 << 1u;
            uint64_t even_in_offset = static_cast<uint64_t>(even_batch) * input_stride_b
                                    + static_cast<uint64_t>(h_in) * input_stride_h;
            uint64_t odd_in_offset = static_cast<uint64_t>(odd_batch) * input_stride_b
                                   + static_cast<uint64_t>(h_in) * input_stride_h
                                   + fixed_depth;

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            DataCopyPad(ubuf[0u], xGM_[even_in_offset], read_params, pad_params);
            DataCopyPad(ubuf[aligned_stream_elems], xGM_[odd_in_offset],
                        read_params, pad_params);
            SetFlag<HardEvent::MTE2_V>(curr_event);
            WaitFlag<HardEvent::MTE2_V>(curr_event);

            uint32_t curr_dst_offset = (curr_slot == 0u) ? dst0_offset : dst1_offset;
            Gather(ubuf[curr_dst_offset], ubuf[0u], offsets, 0u, row_elems);
            SetFlag<HardEvent::V_MTE3>(curr_event);

            has_prev = true;
            prev_slot = curr_slot;
            prev_out_offset = static_cast<uint64_t>(row_idx) * output_stride_h;
        }

        if (has_prev) {
            event_t prev_event = (prev_slot == 0u) ? event0 : event1;
            uint32_t prev_dst_offset = (prev_slot == 0u) ? dst0_offset : dst1_offset;
            WaitFlag<HardEvent::V_MTE3>(prev_event);
            DataCopyPad(yGM_[prev_out_offset], ubuf[prev_dst_offset], write_params);
            SetFlag<HardEvent::MTE3_MTE2>(prev_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
    }

    __aicore__ inline void ProcessScalarPoint5GatherHalf4SlotPipeline() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr uint32_t fixed_depth = 65u;
        constexpr uint32_t fixed_depth_bytes = 130u;
        constexpr uint32_t first_pairs = 64u;
        constexpr uint32_t second_pairs = 63u;
        constexpr uint32_t max_pairs = first_pairs;
        constexpr uint32_t max_stream_elems = max_pairs * fixed_depth;
        constexpr uint32_t max_stream_bytes = max_stream_elems * sizeof(DT_X);
        constexpr uint32_t max_row_elems = 2u * max_stream_elems;
        constexpr uint32_t input_stride_h = 128u * fixed_depth;
        constexpr uint32_t input_stride_b = 128u * input_stride_h;
        constexpr uint32_t output_stride_h = 254u * fixed_depth;
        constexpr uint32_t src_slot_elems = 2u * max_stream_elems;
        constexpr uint32_t dst_offset = src_slot_elems;
        constexpr uint32_t slot_elems = src_slot_elems + max_row_elems;
        constexpr uint32_t offset_table_offset = 4u * slot_elems;

        LocalTensor<int32_t> offsets_i32 = ubuf[offset_table_offset].template ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> offsets = ubuf[offset_table_offset].template ReinterpretCast<uint32_t>();

        for (uint32_t p = 0u; p < 4u; ++p) {
            for (uint32_t c = 0u; c < fixed_depth; ++c) {
                offsets_i32.SetValue(p * 2u * fixed_depth + c,
                                     static_cast<int32_t>(p * fixed_depth_bytes + c * sizeof(DT_X)));
                offsets_i32.SetValue(p * 2u * fixed_depth + fixed_depth + c,
                                     static_cast<int32_t>(max_stream_bytes +
                                                          p * fixed_depth_bytes + c * sizeof(DT_X)));
            }
        }
        PipeBarrier<PIPE_ALL>();

        uint32_t built_pairs = 4u;
        while (built_pairs < max_pairs) {
            uint32_t copy_pairs = MinU32(built_pairs, max_pairs - built_pairs);
            uint32_t dst_index = built_pairs * 2u * fixed_depth;
            Adds(offsets_i32[dst_index], offsets_i32[0u],
                 static_cast<int32_t>(built_pairs * fixed_depth_bytes),
                 copy_pairs * 2u * fixed_depth);
            PipeBarrier<PIPE_V>();
            built_pairs += copy_pairs;
        }

        DataCopyExtParams read_params;
        read_params.blockCount = 1u;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyExtParams write_params;
        write_params.blockCount = 1u;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        DataCopyPadExtParams<DT_X> pad_params;
        pad_params.leftPadding = 0u;
        pad_params.paddingValue = static_cast<DT_X>(0);

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        constexpr event_t event2 = EVENT_ID2;
        constexpr event_t event3 = EVENT_ID3;
        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);
        SetFlag<HardEvent::MTE3_MTE2>(event2);
        SetFlag<HardEvent::MTE3_MTE2>(event3);

        bool has_prev = false;
        uint32_t prev_slot = 0u;
        uint32_t prev_row_elems = 0u;
        uint32_t prev_row_bytes = 0u;
        uint64_t prev_out_offset = 0u;

        for (uint32_t t = 0u; t < my_count_; ++t) {
            uint32_t task_idx = my_start_ + t;
            if (task_idx >= total_units_) {
                break;
            }

            uint32_t row_idx = task_idx >> 1u;
            uint32_t chunk_id = task_idx & 1u;
            uint32_t chunk_pairs = (chunk_id == 0u) ? first_pairs : second_pairs;
            uint32_t pair_start = (chunk_id == 0u) ? 0u : first_pairs;
            uint32_t col_start = (chunk_id == 0u) ? 0u : (2u * first_pairs);
            uint32_t stream_elems = chunk_pairs * fixed_depth;
            uint32_t stream_bytes = stream_elems * sizeof(DT_X);
            uint32_t row_elems = 2u * stream_elems;
            uint32_t row_bytes = row_elems * sizeof(DT_X);

            uint32_t curr_slot = has_prev ? ((prev_slot + 1u) & 3u) : 0u;
            event_t curr_event = (curr_slot == 0u) ? event0
                               : (curr_slot == 1u) ? event1
                               : (curr_slot == 2u) ? event2
                                                   : event3;
            uint32_t curr_base = curr_slot * slot_elems;

            uint32_t h_out = row_idx + 1u;
            uint32_t h_in = h_out >> 1u;
            uint32_t b1 = h_out & 1u;
            uint32_t even_batch = (b1 << 1u) + 1u;
            uint32_t odd_batch = b1 << 1u;
            uint64_t even_in_offset = static_cast<uint64_t>(even_batch) * input_stride_b
                                    + static_cast<uint64_t>(h_in) * input_stride_h
                                    + static_cast<uint64_t>(pair_start) * fixed_depth;
            uint64_t odd_in_offset = static_cast<uint64_t>(odd_batch) * input_stride_b
                                   + static_cast<uint64_t>(h_in) * input_stride_h
                                   + static_cast<uint64_t>(pair_start + 1u) * fixed_depth;
            uint64_t out_offset = static_cast<uint64_t>(row_idx) * output_stride_h
                                + static_cast<uint64_t>(col_start) * fixed_depth;

            read_params.blockLen = stream_bytes;
            pad_params.isPad = stream_elems != max_stream_elems;
            pad_params.rightPadding = static_cast<uint8_t>(max_stream_elems - stream_elems);

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            DataCopyPad(ubuf[curr_base], xGM_[even_in_offset], read_params, pad_params);
            DataCopyPad(ubuf[curr_base + max_stream_elems], xGM_[odd_in_offset],
                        read_params, pad_params);
            SetFlag<HardEvent::MTE2_V>(curr_event);

            if (has_prev) {
                event_t prev_event = (prev_slot == 0u) ? event0
                                   : (prev_slot == 1u) ? event1
                                   : (prev_slot == 2u) ? event2
                                                       : event3;
                uint32_t prev_base = prev_slot * slot_elems;
                WaitFlag<HardEvent::MTE2_V>(prev_event);
                Gather(ubuf[prev_base + dst_offset], ubuf[prev_base], offsets, 0u, prev_row_elems);
                SetFlag<HardEvent::V_MTE3>(prev_event);
                WaitFlag<HardEvent::V_MTE3>(prev_event);
                write_params.blockLen = prev_row_bytes;
                DataCopyPad(yGM_[prev_out_offset], ubuf[prev_base + dst_offset], write_params);
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            has_prev = true;
            prev_slot = curr_slot;
            prev_row_elems = row_elems;
            prev_row_bytes = row_bytes;
            prev_out_offset = out_offset;
        }

        if (has_prev) {
            event_t prev_event = (prev_slot == 0u) ? event0
                               : (prev_slot == 1u) ? event1
                               : (prev_slot == 2u) ? event2
                                                   : event3;
            uint32_t prev_base = prev_slot * slot_elems;
            WaitFlag<HardEvent::MTE2_V>(prev_event);
            Gather(ubuf[prev_base + dst_offset], ubuf[prev_base], offsets, 0u, prev_row_elems);
            SetFlag<HardEvent::V_MTE3>(prev_event);
            WaitFlag<HardEvent::V_MTE3>(prev_event);
            write_params.blockLen = prev_row_bytes;
            DataCopyPad(yGM_[prev_out_offset], ubuf[prev_base + dst_offset], write_params);
            SetFlag<HardEvent::MTE3_MTE2>(prev_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
        WaitFlag<HardEvent::MTE3_MTE2>(event2);
        WaitFlag<HardEvent::MTE3_MTE2>(event3);
    }

    __aicore__ inline void ProcessScalarPoint5GatherThirdPipeline() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr uint32_t fixed_depth = 65u;
        constexpr uint32_t fixed_depth_bytes = 130u;
        constexpr uint32_t first_pairs = 43u;
        constexpr uint32_t other_pairs = 42u;
        constexpr uint32_t max_stream_elems = 2800u;
        constexpr uint32_t max_stream_bytes = max_stream_elems * sizeof(DT_X);
        constexpr uint32_t max_row_elems = 5600u;
        constexpr uint32_t input_stride_h = 128u * fixed_depth;
        constexpr uint32_t input_stride_b = 128u * input_stride_h;
        constexpr uint32_t output_stride_h = 254u * fixed_depth;
        constexpr uint32_t src_slot_elems = 2u * max_stream_elems;
        constexpr uint32_t dst_offset = src_slot_elems;
        constexpr uint32_t slot_elems = src_slot_elems + max_row_elems;
        constexpr uint32_t offset_table_offset = 2u * slot_elems;

        LocalTensor<int32_t> offsets_i32 = ubuf[offset_table_offset].template ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> offsets = ubuf[offset_table_offset].template ReinterpretCast<uint32_t>();

        for (uint32_t p = 0u; p < 4u; ++p) {
            for (uint32_t c = 0u; c < fixed_depth; ++c) {
                offsets_i32.SetValue(p * 2u * fixed_depth + c,
                                     static_cast<int32_t>(p * fixed_depth_bytes + c * sizeof(DT_X)));
                offsets_i32.SetValue(p * 2u * fixed_depth + fixed_depth + c,
                                     static_cast<int32_t>(max_stream_bytes +
                                                          p * fixed_depth_bytes + c * sizeof(DT_X)));
            }
        }
        PipeBarrier<PIPE_ALL>();

        uint32_t built_pairs = 4u;
        while (built_pairs < first_pairs) {
            uint32_t copy_pairs = MinU32(built_pairs, first_pairs - built_pairs);
            uint32_t dst_index = built_pairs * 2u * fixed_depth;
            Adds(offsets_i32[dst_index], offsets_i32[0u],
                 static_cast<int32_t>(built_pairs * fixed_depth_bytes),
                 copy_pairs * 2u * fixed_depth);
            PipeBarrier<PIPE_V>();
            built_pairs += copy_pairs;
        }

        DataCopyExtParams read_params;
        read_params.blockCount = 1u;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyExtParams write_params;
        write_params.blockCount = 1u;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        DataCopyPadExtParams<DT_X> pad_params;
        pad_params.leftPadding = 0u;
        pad_params.paddingValue = static_cast<DT_X>(0);

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);

        bool has_prev = false;
        uint32_t prev_slot = 0u;
        uint32_t prev_row_elems = 0u;
        uint32_t prev_row_bytes = 0u;
        uint64_t prev_out_offset = 0u;

        for (uint32_t t = 0u; t < my_count_; ++t) {
            uint32_t task_idx = my_start_ + t;
            if (task_idx >= total_units_) {
                break;
            }

            uint32_t row_idx = task_idx / 3u;
            uint32_t chunk_id = task_idx - row_idx * 3u;
            uint32_t chunk_pairs = (chunk_id == 0u) ? first_pairs : other_pairs;
            uint32_t pair_start = (chunk_id == 0u) ? 0u : ((chunk_id == 1u) ? first_pairs : (first_pairs + other_pairs));
            uint32_t col_start = 2u * pair_start;
            uint32_t stream_elems = chunk_pairs * fixed_depth;
            uint32_t stream_bytes = stream_elems * sizeof(DT_X);
            uint32_t row_elems = 2u * stream_elems;
            uint32_t row_bytes = row_elems * sizeof(DT_X);

            uint32_t curr_slot = has_prev ? (prev_slot ^ 1u) : 0u;
            event_t curr_event = (curr_slot == 0u) ? event0 : event1;
            uint32_t curr_base = curr_slot * slot_elems;

            uint32_t h_out = row_idx + 1u;
            uint32_t h_in = h_out >> 1u;
            uint32_t b1 = h_out & 1u;
            uint32_t even_batch = (b1 << 1u) + 1u;
            uint32_t odd_batch = b1 << 1u;
            uint64_t even_in_offset = static_cast<uint64_t>(even_batch) * input_stride_b
                                    + static_cast<uint64_t>(h_in) * input_stride_h
                                    + static_cast<uint64_t>(pair_start) * fixed_depth;
            uint64_t odd_in_offset = static_cast<uint64_t>(odd_batch) * input_stride_b
                                   + static_cast<uint64_t>(h_in) * input_stride_h
                                   + static_cast<uint64_t>(pair_start + 1u) * fixed_depth;
            uint64_t out_offset = static_cast<uint64_t>(row_idx) * output_stride_h
                                + static_cast<uint64_t>(col_start) * fixed_depth;

            read_params.blockLen = stream_bytes;
            pad_params.isPad = stream_elems != max_stream_elems;
            pad_params.rightPadding = static_cast<uint8_t>(max_stream_elems - stream_elems);

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            DataCopyPad(ubuf[curr_base], xGM_[even_in_offset], read_params, pad_params);
            DataCopyPad(ubuf[curr_base + max_stream_elems], xGM_[odd_in_offset],
                        read_params, pad_params);
            SetFlag<HardEvent::MTE2_V>(curr_event);

            if (has_prev) {
                event_t prev_event = (prev_slot == 0u) ? event0 : event1;
                uint32_t prev_base = prev_slot * slot_elems;
                WaitFlag<HardEvent::MTE2_V>(prev_event);
                Gather(ubuf[prev_base + dst_offset], ubuf[prev_base], offsets, 0u, prev_row_elems);
                SetFlag<HardEvent::V_MTE3>(prev_event);
                WaitFlag<HardEvent::V_MTE3>(prev_event);
                write_params.blockLen = prev_row_bytes;
                DataCopyPad(yGM_[prev_out_offset], ubuf[prev_base + dst_offset], write_params);
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            has_prev = true;
            prev_slot = curr_slot;
            prev_row_elems = row_elems;
            prev_row_bytes = row_bytes;
            prev_out_offset = out_offset;
        }

        if (has_prev) {
            event_t prev_event = (prev_slot == 0u) ? event0 : event1;
            uint32_t prev_base = prev_slot * slot_elems;
            WaitFlag<HardEvent::MTE2_V>(prev_event);
            Gather(ubuf[prev_base + dst_offset], ubuf[prev_base], offsets, 0u, prev_row_elems);
            SetFlag<HardEvent::V_MTE3>(prev_event);
            WaitFlag<HardEvent::V_MTE3>(prev_event);
            write_params.blockLen = prev_row_bytes;
            DataCopyPad(yGM_[prev_out_offset], ubuf[prev_base + dst_offset], write_params);
            SetFlag<HardEvent::MTE3_MTE2>(prev_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
    }

    __aicore__ inline void ProcessScalarPoint5GatherQuarterPipeline() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr uint32_t fixed_depth = 65u;
        constexpr uint32_t fixed_depth_bytes = 130u;
        constexpr uint32_t chunk_pairs_fixed = 32u;
        constexpr uint32_t last_pairs = 31u;
        constexpr uint32_t max_stream_elems = chunk_pairs_fixed * fixed_depth;
        constexpr uint32_t max_stream_bytes = max_stream_elems * sizeof(DT_X);
        constexpr uint32_t max_row_elems = 2u * max_stream_elems;
        constexpr uint32_t input_stride_h = 128u * fixed_depth;
        constexpr uint32_t input_stride_b = 128u * input_stride_h;
        constexpr uint32_t output_stride_h = 254u * fixed_depth;
        constexpr uint32_t src_slot_elems = 2u * max_stream_elems;
        constexpr uint32_t dst_offset = src_slot_elems;
        constexpr uint32_t slot_elems = src_slot_elems + max_row_elems;
        constexpr uint32_t offset_table_offset = 2u * slot_elems;

        LocalTensor<int32_t> offsets_i32 = ubuf[offset_table_offset].template ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> offsets = ubuf[offset_table_offset].template ReinterpretCast<uint32_t>();

        for (uint32_t p = 0u; p < 4u; ++p) {
            for (uint32_t c = 0u; c < fixed_depth; ++c) {
                offsets_i32.SetValue(p * 2u * fixed_depth + c,
                                     static_cast<int32_t>(p * fixed_depth_bytes + c * sizeof(DT_X)));
                offsets_i32.SetValue(p * 2u * fixed_depth + fixed_depth + c,
                                     static_cast<int32_t>(max_stream_bytes +
                                                          p * fixed_depth_bytes + c * sizeof(DT_X)));
            }
        }
        PipeBarrier<PIPE_ALL>();

        uint32_t built_pairs = 4u;
        while (built_pairs < chunk_pairs_fixed) {
            uint32_t copy_pairs = MinU32(built_pairs, chunk_pairs_fixed - built_pairs);
            uint32_t dst_index = built_pairs * 2u * fixed_depth;
            Adds(offsets_i32[dst_index], offsets_i32[0u],
                 static_cast<int32_t>(built_pairs * fixed_depth_bytes),
                 copy_pairs * 2u * fixed_depth);
            PipeBarrier<PIPE_V>();
            built_pairs += copy_pairs;
        }

        DataCopyExtParams read_params;
        read_params.blockCount = 1u;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyExtParams write_params;
        write_params.blockCount = 1u;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        DataCopyPadExtParams<DT_X> pad_params;
        pad_params.leftPadding = 0u;
        pad_params.paddingValue = static_cast<DT_X>(0);

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);

        bool has_prev = false;
        uint32_t prev_slot = 0u;
        uint32_t prev_row_elems = 0u;
        uint32_t prev_row_bytes = 0u;
        uint64_t prev_out_offset = 0u;

        for (uint32_t t = 0u; t < my_count_; ++t) {
            uint32_t task_idx = my_start_ + t;
            if (task_idx >= total_units_) {
                break;
            }

            uint32_t row_idx = task_idx >> 2u;
            uint32_t chunk_id = task_idx & 3u;
            uint32_t chunk_pairs = (chunk_id == 3u) ? last_pairs : chunk_pairs_fixed;
            uint32_t pair_start = chunk_id * chunk_pairs_fixed;
            uint32_t col_start = 2u * pair_start;
            uint32_t stream_elems = chunk_pairs * fixed_depth;
            uint32_t stream_bytes = stream_elems * sizeof(DT_X);
            uint32_t row_elems = 2u * stream_elems;
            uint32_t row_bytes = row_elems * sizeof(DT_X);

            uint32_t curr_slot = has_prev ? (prev_slot ^ 1u) : 0u;
            event_t curr_event = (curr_slot == 0u) ? event0 : event1;
            uint32_t curr_base = curr_slot * slot_elems;

            uint32_t h_out = row_idx + 1u;
            uint32_t h_in = h_out >> 1u;
            uint32_t b1 = h_out & 1u;
            uint32_t even_batch = (b1 << 1u) + 1u;
            uint32_t odd_batch = b1 << 1u;
            uint64_t even_in_offset = static_cast<uint64_t>(even_batch) * input_stride_b
                                    + static_cast<uint64_t>(h_in) * input_stride_h
                                    + static_cast<uint64_t>(pair_start) * fixed_depth;
            uint64_t odd_in_offset = static_cast<uint64_t>(odd_batch) * input_stride_b
                                   + static_cast<uint64_t>(h_in) * input_stride_h
                                   + static_cast<uint64_t>(pair_start + 1u) * fixed_depth;
            uint64_t out_offset = static_cast<uint64_t>(row_idx) * output_stride_h
                                + static_cast<uint64_t>(col_start) * fixed_depth;

            read_params.blockLen = stream_bytes;
            pad_params.isPad = stream_elems != max_stream_elems;
            pad_params.rightPadding = static_cast<uint8_t>(max_stream_elems - stream_elems);

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            DataCopyPad(ubuf[curr_base], xGM_[even_in_offset], read_params, pad_params);
            DataCopyPad(ubuf[curr_base + max_stream_elems], xGM_[odd_in_offset],
                        read_params, pad_params);
            SetFlag<HardEvent::MTE2_V>(curr_event);

            if (has_prev) {
                event_t prev_event = (prev_slot == 0u) ? event0 : event1;
                uint32_t prev_base = prev_slot * slot_elems;
                WaitFlag<HardEvent::MTE2_V>(prev_event);
                Gather(ubuf[prev_base + dst_offset], ubuf[prev_base], offsets, 0u, prev_row_elems);
                SetFlag<HardEvent::V_MTE3>(prev_event);
                WaitFlag<HardEvent::V_MTE3>(prev_event);
                write_params.blockLen = prev_row_bytes;
                DataCopyPad(yGM_[prev_out_offset], ubuf[prev_base + dst_offset], write_params);
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            has_prev = true;
            prev_slot = curr_slot;
            prev_row_elems = row_elems;
            prev_row_bytes = row_bytes;
            prev_out_offset = out_offset;
        }

        if (has_prev) {
            event_t prev_event = (prev_slot == 0u) ? event0 : event1;
            uint32_t prev_base = prev_slot * slot_elems;
            WaitFlag<HardEvent::MTE2_V>(prev_event);
            Gather(ubuf[prev_base + dst_offset], ubuf[prev_base], offsets, 0u, prev_row_elems);
            SetFlag<HardEvent::V_MTE3>(prev_event);
            WaitFlag<HardEvent::V_MTE3>(prev_event);
            write_params.blockLen = prev_row_bytes;
            DataCopyPad(yGM_[prev_out_offset], ubuf[prev_base + dst_offset], write_params);
            SetFlag<HardEvent::MTE3_MTE2>(prev_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
    }

    __aicore__ inline void ProcessScalarPoint5GatherHalfFast3StagePipeline() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr uint32_t fixed_depth = 65u;
        constexpr uint32_t fixed_depth_bytes = 130u;
        constexpr uint32_t first_stream_elems = 4160u;
        constexpr uint32_t second_stream_elems = 4095u;
        constexpr uint32_t first_stream_bytes = first_stream_elems * sizeof(DT_X);
        constexpr uint32_t second_stream_bytes = second_stream_elems * sizeof(DT_X);
        constexpr uint32_t first_row_elems = 2u * first_stream_elems;
        constexpr uint32_t second_row_elems = 2u * second_stream_elems;
        constexpr uint32_t first_row_bytes = first_row_elems * sizeof(DT_X);
        constexpr uint32_t second_row_bytes = second_row_elems * sizeof(DT_X);
        constexpr uint32_t input_stride_h = 128u * fixed_depth;
        constexpr uint32_t input_stride_b = 128u * input_stride_h;
        constexpr uint32_t output_stride_h = 254u * fixed_depth;
        constexpr uint32_t src_slot_elems = 2u * first_stream_elems;
        constexpr uint32_t dst_offset = src_slot_elems;
        constexpr uint32_t slot_elems = src_slot_elems + first_row_elems;
        constexpr uint32_t offset_table_offset = 3u * slot_elems;

        LocalTensor<int32_t> offsets_i32 = ubuf[offset_table_offset].template ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> offsets = ubuf[offset_table_offset].template ReinterpretCast<uint32_t>();

        for (uint32_t p = 0u; p < 4u; ++p) {
            for (uint32_t c = 0u; c < fixed_depth; ++c) {
                offsets_i32.SetValue(p * 2u * fixed_depth + c,
                                     static_cast<int32_t>(p * fixed_depth_bytes + c * sizeof(DT_X)));
                offsets_i32.SetValue(p * 2u * fixed_depth + fixed_depth + c,
                                     static_cast<int32_t>(first_stream_bytes +
                                                          p * fixed_depth_bytes + c * sizeof(DT_X)));
            }
        }
        PipeBarrier<PIPE_ALL>();

        uint32_t built_pairs = 4u;
        while (built_pairs < 64u) {
            uint32_t copy_pairs = MinU32(built_pairs, 64u - built_pairs);
            uint32_t dst_index = built_pairs * 2u * fixed_depth;
            Adds(offsets_i32[dst_index], offsets_i32[0u],
                 static_cast<int32_t>(built_pairs * fixed_depth_bytes),
                 copy_pairs * 2u * fixed_depth);
            PipeBarrier<PIPE_V>();
            built_pairs += copy_pairs;
        }

        DataCopyExtParams read_params;
        read_params.blockCount = 1u;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyExtParams write_params;
        write_params.blockCount = 1u;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        DataCopyPadExtParams<DT_X> pad_params;
        pad_params.leftPadding = 0u;
        pad_params.paddingValue = static_cast<DT_X>(0);

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        constexpr event_t event2 = EVENT_ID2;
        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);
        SetFlag<HardEvent::MTE3_MTE2>(event2);

        uint32_t row_elems0 = 0u;
        uint32_t row_elems1 = 0u;
        uint32_t row_elems2 = 0u;
        uint32_t row_bytes0 = 0u;
        uint32_t row_bytes1 = 0u;
        uint32_t row_bytes2 = 0u;
        uint64_t out_offset0 = 0u;
        uint64_t out_offset1 = 0u;
        uint64_t out_offset2 = 0u;

        uint32_t task_idx = my_start_;
        uint32_t row_idx = task_idx >> 1u;
        uint32_t chunk_id = task_idx & 1u;
        uint32_t curr_slot = 0u;
        uint32_t processed = 0u;
        for (uint32_t t = 0u; t < my_count_; ++t) {
            if (task_idx >= total_units_) {
                break;
            }

            bool is_second = chunk_id != 0u;
            uint32_t stream_elems = is_second ? second_stream_elems : first_stream_elems;
            uint32_t stream_bytes = is_second ? second_stream_bytes : first_stream_bytes;
            uint32_t row_elems = is_second ? second_row_elems : first_row_elems;
            uint32_t row_bytes = is_second ? second_row_bytes : first_row_bytes;
            uint32_t pair_start = is_second ? 64u : 0u;
            uint32_t col_start = is_second ? 128u : 0u;

            event_t curr_event = (curr_slot == 0u) ? event0 : (curr_slot == 1u) ? event1 : event2;
            uint32_t curr_base = curr_slot * slot_elems;

            uint32_t h_out = row_idx + 1u;
            uint32_t h_in = h_out >> 1u;
            uint32_t b1 = h_out & 1u;
            uint32_t even_batch = (b1 << 1u) + 1u;
            uint32_t odd_batch = b1 << 1u;
            uint64_t even_in_offset = static_cast<uint64_t>(even_batch) * input_stride_b
                                    + static_cast<uint64_t>(h_in) * input_stride_h
                                    + static_cast<uint64_t>(pair_start) * fixed_depth;
            uint64_t odd_in_offset = static_cast<uint64_t>(odd_batch) * input_stride_b
                                   + static_cast<uint64_t>(h_in) * input_stride_h
                                   + static_cast<uint64_t>(pair_start + 1u) * fixed_depth;
            uint64_t out_offset = static_cast<uint64_t>(row_idx) * output_stride_h
                                + static_cast<uint64_t>(col_start) * fixed_depth;

            read_params.blockLen = stream_bytes;
            pad_params.isPad = is_second;
            pad_params.rightPadding = is_second ? 65u : 0u;

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            DataCopyPad(ubuf[curr_base], xGM_[even_in_offset], read_params, pad_params);
            DataCopyPad(ubuf[curr_base + first_stream_elems], xGM_[odd_in_offset],
                        read_params, pad_params);
            SetFlag<HardEvent::MTE2_V>(curr_event);

            if (curr_slot == 0u) {
                row_elems0 = row_elems;
                row_bytes0 = row_bytes;
                out_offset0 = out_offset;
            } else if (curr_slot == 1u) {
                row_elems1 = row_elems;
                row_bytes1 = row_bytes;
                out_offset1 = out_offset;
            } else {
                row_elems2 = row_elems;
                row_bytes2 = row_bytes;
                out_offset2 = out_offset;
            }

            if (processed >= 1u) {
                uint32_t gather_slot = (curr_slot == 0u) ? 2u : (curr_slot - 1u);
                event_t gather_event = (gather_slot == 0u) ? event0
                                     : (gather_slot == 1u) ? event1
                                                           : event2;
                uint32_t gather_base = gather_slot * slot_elems;
                uint32_t gather_row_elems = (gather_slot == 0u) ? row_elems0
                                          : (gather_slot == 1u) ? row_elems1
                                                                : row_elems2;
                WaitFlag<HardEvent::MTE2_V>(gather_event);
                Gather(ubuf[gather_base + dst_offset], ubuf[gather_base], offsets, 0u, gather_row_elems);
                SetFlag<HardEvent::V_MTE3>(gather_event);
            }

            if (processed >= 2u) {
                uint32_t write_slot = curr_slot + 1u;
                if (write_slot == 3u) {
                    write_slot = 0u;
                }
                event_t write_event = (write_slot == 0u) ? event0
                                    : (write_slot == 1u) ? event1
                                                         : event2;
                uint32_t write_base = write_slot * slot_elems;
                uint32_t write_row_bytes = (write_slot == 0u) ? row_bytes0
                                         : (write_slot == 1u) ? row_bytes1
                                                              : row_bytes2;
                uint64_t write_out_offset = (write_slot == 0u) ? out_offset0
                                          : (write_slot == 1u) ? out_offset1
                                                               : out_offset2;
                WaitFlag<HardEvent::V_MTE3>(write_event);
                write_params.blockLen = write_row_bytes;
                DataCopyPad(yGM_[write_out_offset], ubuf[write_base + dst_offset], write_params);
                SetFlag<HardEvent::MTE3_MTE2>(write_event);
            }

            ++task_idx;
            chunk_id ^= 1u;
            if (chunk_id == 0u) {
                ++row_idx;
            }
            ++processed;
            ++curr_slot;
            if (curr_slot == 3u) {
                curr_slot = 0u;
            }
        }

        if (processed > 0u) {
            uint32_t last_slot = (curr_slot == 0u) ? 2u : (curr_slot - 1u);
            event_t last_event = (last_slot == 0u) ? event0 : (last_slot == 1u) ? event1 : event2;
            uint32_t last_base = last_slot * slot_elems;
            uint32_t last_row_elems = (last_slot == 0u) ? row_elems0
                                    : (last_slot == 1u) ? row_elems1
                                                        : row_elems2;
            uint32_t last_row_bytes = (last_slot == 0u) ? row_bytes0
                                    : (last_slot == 1u) ? row_bytes1
                                                        : row_bytes2;
            uint64_t last_out_offset = (last_slot == 0u) ? out_offset0
                                     : (last_slot == 1u) ? out_offset1
                                                         : out_offset2;

            WaitFlag<HardEvent::MTE2_V>(last_event);
            Gather(ubuf[last_base + dst_offset], ubuf[last_base], offsets, 0u, last_row_elems);
            SetFlag<HardEvent::V_MTE3>(last_event);

            if (processed >= 2u) {
                uint32_t prev_slot = (last_slot == 0u) ? 2u : (last_slot - 1u);
                event_t prev_event = (prev_slot == 0u) ? event0
                                   : (prev_slot == 1u) ? event1
                                                       : event2;
                uint32_t prev_base = prev_slot * slot_elems;
                uint32_t prev_row_bytes = (prev_slot == 0u) ? row_bytes0
                                      : (prev_slot == 1u) ? row_bytes1
                                                          : row_bytes2;
                uint64_t prev_out_offset = (prev_slot == 0u) ? out_offset0
                                         : (prev_slot == 1u) ? out_offset1
                                                             : out_offset2;
                WaitFlag<HardEvent::V_MTE3>(prev_event);
                write_params.blockLen = prev_row_bytes;
                DataCopyPad(yGM_[prev_out_offset], ubuf[prev_base + dst_offset], write_params);
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            WaitFlag<HardEvent::V_MTE3>(last_event);
            write_params.blockLen = last_row_bytes;
            DataCopyPad(yGM_[last_out_offset], ubuf[last_base + dst_offset], write_params);
            SetFlag<HardEvent::MTE3_MTE2>(last_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
        WaitFlag<HardEvent::MTE3_MTE2>(event2);
    }

    __aicore__ inline void ProcessScalarPoint5GatherHalfFastPipeline() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr uint32_t fixed_depth = 65u;
        constexpr uint32_t fixed_depth_bytes = 130u;
        constexpr uint32_t first_stream_elems = 4160u;
        constexpr uint32_t second_stream_elems = 4095u;
        constexpr uint32_t first_stream_bytes = first_stream_elems * sizeof(DT_X);
        constexpr uint32_t second_stream_bytes = second_stream_elems * sizeof(DT_X);
        constexpr uint32_t first_row_elems = 2u * first_stream_elems;
        constexpr uint32_t second_row_elems = 2u * second_stream_elems;
        constexpr uint32_t first_row_bytes = first_row_elems * sizeof(DT_X);
        constexpr uint32_t second_row_bytes = second_row_elems * sizeof(DT_X);
        constexpr uint32_t input_stride_h = 128u * fixed_depth;
        constexpr uint32_t input_stride_b = 128u * input_stride_h;
        constexpr uint32_t output_stride_h = 254u * fixed_depth;
        constexpr uint32_t src_slot_elems = 2u * first_stream_elems;
        constexpr uint32_t dst_offset = src_slot_elems;
        constexpr uint32_t slot_elems = src_slot_elems + first_row_elems;
        constexpr uint32_t offset_table_offset = 2u * slot_elems;

        LocalTensor<int32_t> offsets_i32 = ubuf[offset_table_offset].template ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> offsets = ubuf[offset_table_offset].template ReinterpretCast<uint32_t>();

        for (uint32_t p = 0u; p < 4u; ++p) {
            for (uint32_t c = 0u; c < fixed_depth; ++c) {
                offsets_i32.SetValue(p * 2u * fixed_depth + c,
                                     static_cast<int32_t>(p * fixed_depth_bytes + c * sizeof(DT_X)));
                offsets_i32.SetValue(p * 2u * fixed_depth + fixed_depth + c,
                                     static_cast<int32_t>(first_stream_bytes +
                                                          p * fixed_depth_bytes + c * sizeof(DT_X)));
            }
        }
        PipeBarrier<PIPE_ALL>();

        uint32_t built_pairs = 4u;
        while (built_pairs < 64u) {
            uint32_t copy_pairs = MinU32(built_pairs, 64u - built_pairs);
            uint32_t dst_index = built_pairs * 2u * fixed_depth;
            Adds(offsets_i32[dst_index], offsets_i32[0u],
                 static_cast<int32_t>(built_pairs * fixed_depth_bytes),
                 copy_pairs * 2u * fixed_depth);
            PipeBarrier<PIPE_V>();
            built_pairs += copy_pairs;
        }

        DataCopyExtParams read_params;
        read_params.blockCount = 1u;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyExtParams write_params;
        write_params.blockCount = 1u;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        DataCopyPadExtParams<DT_X> pad_params;
        pad_params.leftPadding = 0u;
        pad_params.paddingValue = static_cast<DT_X>(0);

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);

        bool has_prev = false;
        uint32_t prev_slot = 0u;
        uint32_t prev_row_elems = 0u;
        uint32_t prev_row_bytes = 0u;
        uint64_t prev_out_offset = 0u;

        uint32_t task_idx = my_start_;
        uint32_t row_idx = task_idx >> 1u;
        uint32_t chunk_id = task_idx & 1u;
        for (uint32_t t = 0u; t < my_count_; ++t) {
            if (task_idx >= total_units_) {
                break;
            }

            bool is_second = chunk_id != 0u;
            uint32_t stream_elems = is_second ? second_stream_elems : first_stream_elems;
            uint32_t stream_bytes = is_second ? second_stream_bytes : first_stream_bytes;
            uint32_t row_elems = is_second ? second_row_elems : first_row_elems;
            uint32_t row_bytes = is_second ? second_row_bytes : first_row_bytes;
            uint32_t pair_start = is_second ? 64u : 0u;
            uint32_t col_start = is_second ? 128u : 0u;

            uint32_t curr_slot = has_prev ? (prev_slot ^ 1u) : 0u;
            event_t curr_event = (curr_slot == 0u) ? event0 : event1;
            uint32_t curr_base = curr_slot * slot_elems;

            uint32_t h_out = row_idx + 1u;
            uint32_t h_in = h_out >> 1u;
            uint32_t b1 = h_out & 1u;
            uint32_t even_batch = (b1 << 1u) + 1u;
            uint32_t odd_batch = b1 << 1u;
            uint64_t even_in_offset = static_cast<uint64_t>(even_batch) * input_stride_b
                                    + static_cast<uint64_t>(h_in) * input_stride_h
                                    + static_cast<uint64_t>(pair_start) * fixed_depth;
            uint64_t odd_in_offset = static_cast<uint64_t>(odd_batch) * input_stride_b
                                   + static_cast<uint64_t>(h_in) * input_stride_h
                                   + static_cast<uint64_t>(pair_start + 1u) * fixed_depth;
            uint64_t out_offset = static_cast<uint64_t>(row_idx) * output_stride_h
                                + static_cast<uint64_t>(col_start) * fixed_depth;

            read_params.blockLen = stream_bytes;
            pad_params.isPad = is_second;
            pad_params.rightPadding = is_second ? 65u : 0u;

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            DataCopyPad(ubuf[curr_base], xGM_[even_in_offset], read_params, pad_params);
            DataCopyPad(ubuf[curr_base + first_stream_elems], xGM_[odd_in_offset],
                        read_params, pad_params);
            SetFlag<HardEvent::MTE2_V>(curr_event);

            if (has_prev) {
                event_t prev_event = (prev_slot == 0u) ? event0 : event1;
                uint32_t prev_base = prev_slot * slot_elems;
                WaitFlag<HardEvent::MTE2_V>(prev_event);
                Gather(ubuf[prev_base + dst_offset], ubuf[prev_base], offsets, 0u, prev_row_elems);
                SetFlag<HardEvent::V_MTE3>(prev_event);
                WaitFlag<HardEvent::V_MTE3>(prev_event);
                write_params.blockLen = prev_row_bytes;
                DataCopyPad(yGM_[prev_out_offset], ubuf[prev_base + dst_offset], write_params);
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            has_prev = true;
            prev_slot = curr_slot;
            prev_row_elems = row_elems;
            prev_row_bytes = row_bytes;
            prev_out_offset = out_offset;

            ++task_idx;
            chunk_id ^= 1u;
            if (chunk_id == 0u) {
                ++row_idx;
            }
        }

        if (has_prev) {
            event_t prev_event = (prev_slot == 0u) ? event0 : event1;
            uint32_t prev_base = prev_slot * slot_elems;
            WaitFlag<HardEvent::MTE2_V>(prev_event);
            Gather(ubuf[prev_base + dst_offset], ubuf[prev_base], offsets, 0u, prev_row_elems);
            SetFlag<HardEvent::V_MTE3>(prev_event);
            WaitFlag<HardEvent::V_MTE3>(prev_event);
            write_params.blockLen = prev_row_bytes;
            DataCopyPad(yGM_[prev_out_offset], ubuf[prev_base + dst_offset], write_params);
            SetFlag<HardEvent::MTE3_MTE2>(prev_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
    }

    __aicore__ inline void ProcessScalarPoint5GatherHalfPipeline() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr uint32_t fixed_depth = 65u;
        constexpr uint32_t fixed_depth_bytes = 130u;
        constexpr uint32_t first_pairs = 64u;
        constexpr uint32_t second_pairs = 63u;
        constexpr uint32_t max_pairs = first_pairs;
        constexpr uint32_t max_stream_elems = max_pairs * fixed_depth;
        constexpr uint32_t max_stream_bytes = max_stream_elems * sizeof(DT_X);
        constexpr uint32_t max_row_elems = 2u * max_stream_elems;
        constexpr uint32_t input_stride_h = 128u * fixed_depth;
        constexpr uint32_t input_stride_b = 128u * input_stride_h;
        constexpr uint32_t output_stride_h = 254u * fixed_depth;
        constexpr uint32_t src_slot_elems = 2u * max_stream_elems;
        constexpr uint32_t dst_offset = src_slot_elems;
        constexpr uint32_t slot_elems = src_slot_elems + max_row_elems;
        constexpr uint32_t offset_table_offset = 2u * slot_elems;

        LocalTensor<int32_t> offsets_i32 = ubuf[offset_table_offset].template ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> offsets = ubuf[offset_table_offset].template ReinterpretCast<uint32_t>();

        for (uint32_t p = 0u; p < 4u; ++p) {
            for (uint32_t c = 0u; c < fixed_depth; ++c) {
                offsets_i32.SetValue(p * 2u * fixed_depth + c,
                                     static_cast<int32_t>(p * fixed_depth_bytes + c * sizeof(DT_X)));
                offsets_i32.SetValue(p * 2u * fixed_depth + fixed_depth + c,
                                     static_cast<int32_t>(max_stream_bytes +
                                                          p * fixed_depth_bytes + c * sizeof(DT_X)));
            }
        }
        PipeBarrier<PIPE_ALL>();

        uint32_t built_pairs = 4u;
        while (built_pairs < max_pairs) {
            uint32_t copy_pairs = MinU32(built_pairs, max_pairs - built_pairs);
            uint32_t dst_index = built_pairs * 2u * fixed_depth;
            Adds(offsets_i32[dst_index], offsets_i32[0u],
                 static_cast<int32_t>(built_pairs * fixed_depth_bytes),
                 copy_pairs * 2u * fixed_depth);
            PipeBarrier<PIPE_V>();
            built_pairs += copy_pairs;
        }

        DataCopyExtParams read_params;
        read_params.blockCount = 1u;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyExtParams write_params;
        write_params.blockCount = 1u;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        DataCopyPadExtParams<DT_X> pad_params;
        pad_params.leftPadding = 0u;
        pad_params.paddingValue = static_cast<DT_X>(0);

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);

        bool has_prev = false;
        uint32_t prev_slot = 0u;
        uint32_t prev_row_elems = 0u;
        uint32_t prev_row_bytes = 0u;
        uint64_t prev_out_offset = 0u;

        for (uint32_t t = 0u; t < my_count_; ++t) {
            uint32_t task_idx = my_start_ + t;
            if (task_idx >= total_units_) {
                break;
            }

            uint32_t row_idx = task_idx >> 1u;
            uint32_t chunk_id = task_idx & 1u;
            uint32_t chunk_pairs = (chunk_id == 0u) ? first_pairs : second_pairs;
            uint32_t pair_start = (chunk_id == 0u) ? 0u : first_pairs;
            uint32_t col_start = (chunk_id == 0u) ? 0u : (2u * first_pairs);
            uint32_t stream_elems = chunk_pairs * fixed_depth;
            uint32_t stream_bytes = stream_elems * sizeof(DT_X);
            uint32_t row_elems = 2u * stream_elems;
            uint32_t row_bytes = row_elems * sizeof(DT_X);

            uint32_t curr_slot = has_prev ? (prev_slot ^ 1u) : 0u;
            event_t curr_event = (curr_slot == 0u) ? event0 : event1;
            uint32_t curr_base = curr_slot * slot_elems;

            uint32_t h_out = row_idx + 1u;
            uint32_t h_in = h_out >> 1u;
            uint32_t b1 = h_out & 1u;
            uint32_t even_batch = (b1 << 1u) + 1u;
            uint32_t odd_batch = b1 << 1u;
            uint64_t even_in_offset = static_cast<uint64_t>(even_batch) * input_stride_b
                                    + static_cast<uint64_t>(h_in) * input_stride_h
                                    + static_cast<uint64_t>(pair_start) * fixed_depth;
            uint64_t odd_in_offset = static_cast<uint64_t>(odd_batch) * input_stride_b
                                   + static_cast<uint64_t>(h_in) * input_stride_h
                                   + static_cast<uint64_t>(pair_start + 1u) * fixed_depth;
            uint64_t out_offset = static_cast<uint64_t>(row_idx) * output_stride_h
                                + static_cast<uint64_t>(col_start) * fixed_depth;

            read_params.blockLen = stream_bytes;
            pad_params.isPad = stream_elems != max_stream_elems;
            pad_params.rightPadding = static_cast<uint8_t>(max_stream_elems - stream_elems);

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            DataCopyPad(ubuf[curr_base], xGM_[even_in_offset], read_params, pad_params);
            DataCopyPad(ubuf[curr_base + max_stream_elems], xGM_[odd_in_offset],
                        read_params, pad_params);
            SetFlag<HardEvent::MTE2_V>(curr_event);

            if (has_prev) {
                event_t prev_event = (prev_slot == 0u) ? event0 : event1;
                uint32_t prev_base = prev_slot * slot_elems;
                WaitFlag<HardEvent::MTE2_V>(prev_event);
                Gather(ubuf[prev_base + dst_offset], ubuf[prev_base], offsets, 0u, prev_row_elems);
                SetFlag<HardEvent::V_MTE3>(prev_event);
                WaitFlag<HardEvent::V_MTE3>(prev_event);
                write_params.blockLen = prev_row_bytes;
                DataCopyPad(yGM_[prev_out_offset], ubuf[prev_base + dst_offset], write_params);
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            has_prev = true;
            prev_slot = curr_slot;
            prev_row_elems = row_elems;
            prev_row_bytes = row_bytes;
            prev_out_offset = out_offset;
        }

        if (has_prev) {
            event_t prev_event = (prev_slot == 0u) ? event0 : event1;
            uint32_t prev_base = prev_slot * slot_elems;
            WaitFlag<HardEvent::MTE2_V>(prev_event);
            Gather(ubuf[prev_base + dst_offset], ubuf[prev_base], offsets, 0u, prev_row_elems);
            SetFlag<HardEvent::V_MTE3>(prev_event);
            WaitFlag<HardEvent::V_MTE3>(prev_event);
            write_params.blockLen = prev_row_bytes;
            DataCopyPad(yGM_[prev_out_offset], ubuf[prev_base + dst_offset], write_params);
            SetFlag<HardEvent::MTE3_MTE2>(prev_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
    }

    __aicore__ inline void ProcessScalarPoint5GatherPipelineDstEvent() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr uint32_t fixed_depth = 65u;
        constexpr uint32_t fixed_depth_bytes = 130u;
        constexpr uint32_t stream_pixels = 127u;
        constexpr uint32_t stream_elems = stream_pixels * fixed_depth;
        constexpr uint32_t stream_bytes = stream_elems * sizeof(DT_X);
        constexpr uint32_t aligned_stream_elems = 8256u;
        constexpr uint32_t row_elems = 254u * fixed_depth;
        constexpr uint32_t row_bytes = row_elems * sizeof(DT_X);
        constexpr uint32_t row_aligned_elems = 16512u;
        constexpr uint32_t input_stride_h = 128u * fixed_depth;
        constexpr uint32_t input_stride_b = 128u * input_stride_h;
        constexpr uint32_t output_stride_h = row_elems;
        constexpr uint32_t src_slot_elems = 2u * aligned_stream_elems;
        constexpr uint32_t dst_offset = 2u * src_slot_elems;
        constexpr uint32_t offset_table_offset = dst_offset + row_aligned_elems;

        LocalTensor<DT_X> dst = ubuf[dst_offset];
        LocalTensor<int32_t> offsets_i32 = ubuf[offset_table_offset].template ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> offsets = ubuf[offset_table_offset].template ReinterpretCast<uint32_t>();

        for (uint32_t p = 0u; p < 4u; ++p) {
            for (uint32_t c = 0u; c < fixed_depth; ++c) {
                offsets_i32.SetValue(p * 2u * fixed_depth + c,
                                     static_cast<int32_t>(p * fixed_depth_bytes + c * sizeof(DT_X)));
                offsets_i32.SetValue(p * 2u * fixed_depth + fixed_depth + c,
                                     static_cast<int32_t>(stream_bytes + 2u +
                                                          p * fixed_depth_bytes + c * sizeof(DT_X)));
            }
        }
        PipeBarrier<PIPE_ALL>();

        uint32_t built_pairs = 4u;
        while (built_pairs < 124u) {
            uint32_t copy_pairs = MinU32(built_pairs, 124u - built_pairs);
            uint32_t dst_index = built_pairs * 2u * fixed_depth;
            Adds(offsets_i32[dst_index], offsets_i32[0u],
                 static_cast<int32_t>(built_pairs * fixed_depth_bytes),
                 copy_pairs * 2u * fixed_depth);
            PipeBarrier<PIPE_V>();
            built_pairs += copy_pairs;
        }
        Adds(offsets_i32[124u * 2u * fixed_depth], offsets_i32[0u],
             static_cast<int32_t>(124u * fixed_depth_bytes),
             3u * 2u * fixed_depth);
        PipeBarrier<PIPE_V>();

        DataCopyExtParams read_params;
        read_params.blockCount = 1u;
        read_params.blockLen = stream_bytes;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyExtParams write_params;
        write_params.blockCount = 1u;
        write_params.blockLen = row_bytes;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        DataCopyPadExtParams<DT_X> pad_params;
        pad_params.isPad = false;
        pad_params.leftPadding = 0u;
        pad_params.rightPadding = 1u;
        pad_params.paddingValue = static_cast<DT_X>(0);

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        constexpr event_t dst_event = EVENT_ID2;
        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);
        SetFlag<HardEvent::MTE3_V>(dst_event);

        bool has_prev = false;
        uint32_t prev_slot = 0u;
        uint64_t prev_out_offset = 0u;

        for (uint32_t r = 0u; r < my_count_; ++r) {
            uint32_t row_idx = my_start_ + r;
            if (row_idx >= total_units_) {
                break;
            }

            uint32_t curr_slot = has_prev ? (prev_slot ^ 1u) : 0u;
            event_t curr_event = (curr_slot == 0u) ? event0 : event1;
            uint32_t curr_base = curr_slot * src_slot_elems;

            uint32_t h_out = row_idx + 1u;
            uint32_t h_in = h_out >> 1u;
            uint32_t b1 = h_out & 1u;
            uint32_t even_batch = (b1 << 1u) + 1u;
            uint32_t odd_batch = b1 << 1u;
            uint64_t even_in_offset = static_cast<uint64_t>(even_batch) * input_stride_b
                                    + static_cast<uint64_t>(h_in) * input_stride_h;
            uint64_t odd_in_offset = static_cast<uint64_t>(odd_batch) * input_stride_b
                                   + static_cast<uint64_t>(h_in) * input_stride_h
                                   + fixed_depth;

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            DataCopyPad(ubuf[curr_base], xGM_[even_in_offset], read_params, pad_params);
            DataCopyPad(ubuf[curr_base + aligned_stream_elems], xGM_[odd_in_offset],
                        read_params, pad_params);
            SetFlag<HardEvent::MTE2_V>(curr_event);

            if (has_prev) {
                event_t prev_event = (prev_slot == 0u) ? event0 : event1;
                uint32_t prev_base = prev_slot * src_slot_elems;
                WaitFlag<HardEvent::MTE2_V>(prev_event);
                WaitFlag<HardEvent::MTE3_V>(dst_event);
                Gather(dst, ubuf[prev_base], offsets, 0u, row_elems);
                SetFlag<HardEvent::V_MTE3>(dst_event);
                WaitFlag<HardEvent::V_MTE3>(dst_event);
                DataCopyPad(yGM_[prev_out_offset], dst, write_params);
                SetFlag<HardEvent::MTE3_V>(dst_event);
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            has_prev = true;
            prev_slot = curr_slot;
            prev_out_offset = static_cast<uint64_t>(row_idx) * output_stride_h;
        }

        if (has_prev) {
            event_t prev_event = (prev_slot == 0u) ? event0 : event1;
            uint32_t prev_base = prev_slot * src_slot_elems;
            WaitFlag<HardEvent::MTE2_V>(prev_event);
            WaitFlag<HardEvent::MTE3_V>(dst_event);
            Gather(dst, ubuf[prev_base], offsets, 0u, row_elems);
            SetFlag<HardEvent::V_MTE3>(dst_event);
            WaitFlag<HardEvent::V_MTE3>(dst_event);
            DataCopyPad(yGM_[prev_out_offset], dst, write_params);
            SetFlag<HardEvent::MTE3_V>(dst_event);
            SetFlag<HardEvent::MTE3_MTE2>(prev_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
        WaitFlag<HardEvent::MTE3_V>(dst_event);
    }

    __aicore__ inline void ProcessScalarPoint5GatherPipelineBarrier() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr uint32_t fixed_depth = 65u;
        constexpr uint32_t fixed_depth_bytes = 130u;
        constexpr uint32_t stream_pixels = 127u;
        constexpr uint32_t stream_elems = stream_pixels * fixed_depth;
        constexpr uint32_t stream_bytes = stream_elems * sizeof(DT_X);
        constexpr uint32_t aligned_stream_elems = 8256u;
        constexpr uint32_t row_elems = 254u * fixed_depth;
        constexpr uint32_t row_bytes = row_elems * sizeof(DT_X);
        constexpr uint32_t row_aligned_elems = 16512u;
        constexpr uint32_t input_stride_h = 128u * fixed_depth;
        constexpr uint32_t input_stride_b = 128u * input_stride_h;
        constexpr uint32_t output_stride_h = row_elems;
        constexpr uint32_t src_slot_elems = 2u * aligned_stream_elems;
        constexpr uint32_t dst_offset = 2u * src_slot_elems;
        constexpr uint32_t offset_table_offset = dst_offset + row_aligned_elems;

        LocalTensor<DT_X> dst = ubuf[dst_offset];
        LocalTensor<int32_t> offsets_i32 = ubuf[offset_table_offset].template ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> offsets = ubuf[offset_table_offset].template ReinterpretCast<uint32_t>();

        for (uint32_t p = 0u; p < 4u; ++p) {
            for (uint32_t c = 0u; c < fixed_depth; ++c) {
                offsets_i32.SetValue(p * 2u * fixed_depth + c,
                                     static_cast<int32_t>(p * fixed_depth_bytes + c * sizeof(DT_X)));
                offsets_i32.SetValue(p * 2u * fixed_depth + fixed_depth + c,
                                     static_cast<int32_t>(stream_bytes + 2u +
                                                          p * fixed_depth_bytes + c * sizeof(DT_X)));
            }
        }
        PipeBarrier<PIPE_ALL>();

        uint32_t built_pairs = 4u;
        while (built_pairs < 124u) {
            uint32_t copy_pairs = MinU32(built_pairs, 124u - built_pairs);
            uint32_t dst_index = built_pairs * 2u * fixed_depth;
            Adds(offsets_i32[dst_index], offsets_i32[0u],
                 static_cast<int32_t>(built_pairs * fixed_depth_bytes),
                 copy_pairs * 2u * fixed_depth);
            PipeBarrier<PIPE_V>();
            built_pairs += copy_pairs;
        }
        Adds(offsets_i32[124u * 2u * fixed_depth], offsets_i32[0u],
             static_cast<int32_t>(124u * fixed_depth_bytes),
             3u * 2u * fixed_depth);
        PipeBarrier<PIPE_V>();

        DataCopyExtParams read_params;
        read_params.blockCount = 1u;
        read_params.blockLen = stream_bytes;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyExtParams write_params;
        write_params.blockCount = 1u;
        write_params.blockLen = row_bytes;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        DataCopyPadExtParams<DT_X> pad_params;
        pad_params.isPad = false;
        pad_params.leftPadding = 0u;
        pad_params.rightPadding = 1u;
        pad_params.paddingValue = static_cast<DT_X>(0);

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);

        bool has_prev = false;
        uint32_t prev_slot = 0u;
        uint64_t prev_out_offset = 0u;

        for (uint32_t r = 0u; r < my_count_; ++r) {
            uint32_t row_idx = my_start_ + r;
            if (row_idx >= total_units_) {
                break;
            }

            uint32_t curr_slot = has_prev ? (prev_slot ^ 1u) : 0u;
            event_t curr_event = (curr_slot == 0u) ? event0 : event1;
            uint32_t curr_base = curr_slot * src_slot_elems;

            uint32_t h_out = row_idx + 1u;
            uint32_t h_in = h_out >> 1u;
            uint32_t b1 = h_out & 1u;
            uint32_t even_batch = (b1 << 1u) + 1u;
            uint32_t odd_batch = b1 << 1u;
            uint64_t even_in_offset = static_cast<uint64_t>(even_batch) * input_stride_b
                                    + static_cast<uint64_t>(h_in) * input_stride_h;
            uint64_t odd_in_offset = static_cast<uint64_t>(odd_batch) * input_stride_b
                                   + static_cast<uint64_t>(h_in) * input_stride_h
                                   + fixed_depth;

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            DataCopyPad(ubuf[curr_base], xGM_[even_in_offset], read_params, pad_params);
            DataCopyPad(ubuf[curr_base + aligned_stream_elems], xGM_[odd_in_offset],
                        read_params, pad_params);
            SetFlag<HardEvent::MTE2_V>(curr_event);

            if (has_prev) {
                event_t prev_event = (prev_slot == 0u) ? event0 : event1;
                uint32_t prev_base = prev_slot * src_slot_elems;
                WaitFlag<HardEvent::MTE2_V>(prev_event);
                Gather(dst, ubuf[prev_base], offsets, 0u, row_elems);
                SetFlag<HardEvent::V_MTE3>(prev_event);
                WaitFlag<HardEvent::V_MTE3>(prev_event);
                DataCopyPad(yGM_[prev_out_offset], dst, write_params);
                PipeBarrier<PIPE_ALL>();
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            has_prev = true;
            prev_slot = curr_slot;
            prev_out_offset = static_cast<uint64_t>(row_idx) * output_stride_h;
        }

        if (has_prev) {
            event_t prev_event = (prev_slot == 0u) ? event0 : event1;
            uint32_t prev_base = prev_slot * src_slot_elems;
            WaitFlag<HardEvent::MTE2_V>(prev_event);
            Gather(dst, ubuf[prev_base], offsets, 0u, row_elems);
            SetFlag<HardEvent::V_MTE3>(prev_event);
            WaitFlag<HardEvent::V_MTE3>(prev_event);
            DataCopyPad(yGM_[prev_out_offset], dst, write_params);
            PipeBarrier<PIPE_ALL>();
            SetFlag<HardEvent::MTE3_MTE2>(prev_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
    }

    __aicore__ inline void ProcessScalarPoint5RowGroups() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr uint32_t group_rows = 4u;
        constexpr uint32_t groups_per_batch = 32u;
        constexpr uint32_t valid_rows_per_batch = 127u;
        constexpr uint32_t fixed_width = 128u;
        constexpr uint32_t stream_pixels = 127u;
        constexpr uint32_t fixed_depth = 65u;
        constexpr uint32_t fixed_depth_bytes = 130u;
        constexpr uint32_t fixed_depth_aligned = 80u;
        constexpr uint32_t input_stride_h = fixed_width * fixed_depth;
        constexpr uint32_t input_stride_b = fixed_width * input_stride_h;
        constexpr uint32_t output_stride_h = 254u * fixed_depth;
        constexpr uint32_t ub_row_elems = fixed_width * fixed_depth_aligned;
        constexpr uint32_t buf_elems = group_rows * ub_row_elems;

        DataCopyExtParams read_params;
        read_params.blockLen = fixed_depth_bytes;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyExtParams write_params;
        write_params.blockCount = static_cast<uint16_t>(stream_pixels);
        write_params.blockLen = fixed_depth_bytes;
        write_params.srcStride = 0u;
        write_params.dstStride = fixed_depth_bytes;

        DataCopyPadExtParams<DT_X> pad_params;
        pad_params.isPad = false;
        pad_params.leftPadding = 0u;
        pad_params.rightPadding = static_cast<uint8_t>(fixed_depth_aligned - fixed_depth);
        pad_params.paddingValue = static_cast<DT_X>(0);

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);

        bool has_prev = false;
        uint32_t prev_slot = 0u;
        uint32_t prev_batch = 0u;
        uint32_t prev_first_h = 0u;
        uint32_t prev_rows = 0u;

        for (uint32_t t = 0u; t < my_count_; ++t) {
            uint32_t task = my_start_ + t;
            if (task >= total_units_) {
                break;
            }

            uint32_t input_batch = task / groups_per_batch;
            uint32_t group_idx = task - input_batch * groups_per_batch;
            uint32_t first_h = ((input_batch < 2u) ? 1u : 0u) + group_idx * group_rows;
            uint32_t rows = MinU32(group_rows, valid_rows_per_batch - group_idx * group_rows);
            uint32_t curr_slot = has_prev ? (prev_slot ^ 1u) : 0u;
            event_t curr_event = (curr_slot == 0u) ? event0 : event1;

            uint64_t in_offset = static_cast<uint64_t>(input_batch) * input_stride_b
                               + static_cast<uint64_t>(first_h) * input_stride_h;

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            read_params.blockCount = static_cast<uint16_t>(rows * fixed_width);
            DataCopyPad(ubuf[curr_slot * buf_elems], xGM_[in_offset], read_params, pad_params);
            SetFlag<HardEvent::MTE2_MTE3>(curr_event);

            if (has_prev) {
                event_t prev_event = (prev_slot == 0u) ? event0 : event1;
                uint32_t b1 = prev_batch >> 1u;
                uint32_t b2 = prev_batch & 1u;
                uint32_t first_w = b2 ^ 1u;
                uint32_t ub_offset = prev_slot * buf_elems + first_w * fixed_depth_aligned;
                uint32_t row_idx = (prev_first_h << 1u) + b1 - 1u;
                uint64_t out_offset = static_cast<uint64_t>(row_idx) * output_stride_h
                                    + static_cast<uint64_t>(first_w) * fixed_depth;

                WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
                for (uint32_t r = 0u; r < prev_rows; ++r) {
                    DataCopyPad(yGM_[out_offset], ubuf[ub_offset], write_params);
                    ub_offset += ub_row_elems;
                    out_offset += 2u * output_stride_h;
                }
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            has_prev = true;
            prev_slot = curr_slot;
            prev_batch = input_batch;
            prev_first_h = first_h;
            prev_rows = rows;
        }

        if (has_prev) {
            event_t prev_event = (prev_slot == 0u) ? event0 : event1;
            uint32_t b1 = prev_batch >> 1u;
            uint32_t b2 = prev_batch & 1u;
            uint32_t first_w = b2 ^ 1u;
            uint32_t ub_offset = prev_slot * buf_elems + first_w * fixed_depth_aligned;
            uint32_t row_idx = (prev_first_h << 1u) + b1 - 1u;
            uint64_t out_offset = static_cast<uint64_t>(row_idx) * output_stride_h
                                + static_cast<uint64_t>(first_w) * fixed_depth;

            WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
            for (uint32_t r = 0u; r < prev_rows; ++r) {
                DataCopyPad(yGM_[out_offset], ubuf[ub_offset], write_params);
                ub_offset += ub_row_elems;
                out_offset += 2u * output_stride_h;
            }
            SetFlag<HardEvent::MTE3_MTE2>(prev_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
    }

    __aicore__ inline void ProcessScalarPoint5Fixed() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr uint32_t stream_pixels = 127u;
        constexpr uint32_t fixed_depth = 65u;
        constexpr uint32_t fixed_depth_bytes = 130u;
        constexpr uint32_t fixed_depth_aligned = 80u;
        constexpr uint32_t input_stride_h = 128u * fixed_depth;
        constexpr uint32_t input_stride_b = 128u * input_stride_h;
        constexpr uint32_t output_stride_h = 254u * fixed_depth;

        DataCopyExtParams read_params;
        read_params.blockCount = static_cast<uint16_t>(stream_pixels);
        read_params.blockLen = fixed_depth_bytes;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyExtParams write_params;
        write_params.blockCount = static_cast<uint16_t>(stream_pixels);
        write_params.blockLen = fixed_depth_bytes;
        write_params.srcStride = 0u;
        write_params.dstStride = fixed_depth_bytes;

        DataCopyPadExtParams<DT_X> pad_params;
        pad_params.isPad = false;
        pad_params.leftPadding = 0u;
        pad_params.rightPadding = static_cast<uint8_t>(fixed_depth_aligned - fixed_depth);
        pad_params.paddingValue = static_cast<DT_X>(0);

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        constexpr uint32_t buf_elems = stream_pixels * fixed_depth_aligned;
        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);

        bool has_prev = false;
        uint32_t prev_slot = 0u;
        uint64_t prev_out_offset = 0u;

        for (uint32_t s = 0u; s < my_count_; ++s) {
            uint32_t stream_idx = my_start_ + s;
            if (stream_idx >= total_units_) {
                break;
            }

            uint32_t row_idx = stream_idx >> 1u;
            uint32_t b2 = stream_idx & 1u;
            uint32_t h_out = row_idx + 1u;
            uint32_t h_in = h_out >> 1u;
            uint32_t input_batch = ((h_out & 1u) << 1u) + b2;
            uint32_t first_w = b2 ^ 1u;

            uint64_t in_offset = static_cast<uint64_t>(input_batch) * input_stride_b
                               + static_cast<uint64_t>(h_in) * input_stride_h
                               + static_cast<uint64_t>(first_w) * fixed_depth;
            uint64_t out_offset = static_cast<uint64_t>(row_idx) * output_stride_h
                                + static_cast<uint64_t>(first_w) * fixed_depth;

            uint32_t curr_slot = has_prev ? (prev_slot ^ 1u) : 0u;
            event_t curr_event = (curr_slot == 0u) ? event0 : event1;

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            DataCopyPad(ubuf[curr_slot * buf_elems], xGM_[in_offset], read_params, pad_params);
            SetFlag<HardEvent::MTE2_MTE3>(curr_event);

            if (has_prev) {
                event_t prev_event = (prev_slot == 0u) ? event0 : event1;
                WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
                DataCopyPad(yGM_[prev_out_offset], ubuf[prev_slot * buf_elems], write_params);
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            has_prev = true;
            prev_slot = curr_slot;
            prev_out_offset = out_offset;
        }

        if (has_prev) {
            event_t prev_event = (prev_slot == 0u) ? event0 : event1;
            WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
            DataCopyPad(yGM_[prev_out_offset], ubuf[prev_slot * buf_elems], write_params);
            SetFlag<HardEvent::MTE3_MTE2>(prev_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
    }

    __aicore__ inline void ProcessPoint2GatherPipeline() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr uint32_t fixed_depth = 5u;
        constexpr uint32_t fixed_width = 15u;
        constexpr uint32_t fixed_height = 10u;
        constexpr uint32_t out_width = 26u;
        constexpr uint32_t crop_top = 2u;
        constexpr uint32_t even_first_w_in = 1u;
        constexpr uint32_t odd_first_w_in = 2u;
        constexpr uint32_t stream_pixels = 13u;
        constexpr uint32_t stream_elems = stream_pixels * fixed_depth;
        constexpr uint32_t fixed_depth_bytes = fixed_depth * static_cast<uint32_t>(sizeof(DT_X));
        constexpr uint32_t stream_bytes = stream_elems * static_cast<uint32_t>(sizeof(DT_X));
        constexpr uint32_t aligned_stream_elems =
            ((stream_bytes + 31u) / 32u) * (32u / static_cast<uint32_t>(sizeof(DT_X)));
        constexpr uint32_t row_elems = out_width * fixed_depth;
        constexpr uint32_t row_bytes = row_elems * static_cast<uint32_t>(sizeof(DT_X));
        constexpr uint32_t row_aligned_elems =
            ((row_bytes + 31u) / 32u) * (32u / static_cast<uint32_t>(sizeof(DT_X)));
        constexpr uint32_t input_stride_h = fixed_width * fixed_depth;
        constexpr uint32_t input_stride_b = fixed_height * input_stride_h;
        constexpr uint32_t output_stride_h = row_elems;
        constexpr uint32_t dst_offset = 2u * aligned_stream_elems;
        constexpr uint32_t slot_elems = dst_offset + row_aligned_elems;
        constexpr uint32_t offset_table_offset = 2u * slot_elems;

        LocalTensor<int32_t> offsets_i32 = ubuf[offset_table_offset].template ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> offsets = ubuf[offset_table_offset].template ReinterpretCast<uint32_t>();

        for (uint32_t p = 0u; p < stream_pixels; ++p) {
            for (uint32_t c = 0u; c < fixed_depth; ++c) {
                offsets_i32.SetValue((2u * p) * fixed_depth + c,
                                     static_cast<int32_t>(p * fixed_depth_bytes +
                                                          c * static_cast<uint32_t>(sizeof(DT_X))));
                offsets_i32.SetValue((2u * p + 1u) * fixed_depth + c,
                                     static_cast<int32_t>(aligned_stream_elems *
                                                              static_cast<uint32_t>(sizeof(DT_X)) +
                                                          p * fixed_depth_bytes +
                                                          c * static_cast<uint32_t>(sizeof(DT_X))));
            }
        }
        PipeBarrier<PIPE_ALL>();

        DataCopyExtParams read_params;
        read_params.blockCount = 1u;
        read_params.blockLen = stream_bytes;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyExtParams write_params;
        write_params.blockCount = 1u;
        write_params.blockLen = row_bytes;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        DataCopyPadExtParams<DT_X> pad_params;
        pad_params.isPad = (stream_bytes % 32u) != 0u;
        pad_params.leftPadding = 0u;
        pad_params.rightPadding = static_cast<uint8_t>(aligned_stream_elems - stream_elems);
        pad_params.paddingValue = static_cast<DT_X>(0);

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);

        bool has_prev = false;
        uint32_t prev_slot = 0u;
        uint64_t prev_out_offset = 0u;

        for (uint32_t r = 0u; r < my_count_; ++r) {
            uint32_t row_idx = my_start_ + r;
            if (row_idx >= total_units_) {
                break;
            }

            uint32_t curr_slot = has_prev ? (prev_slot ^ 1u) : 0u;
            event_t curr_event = (curr_slot == 0u) ? event0 : event1;
            uint32_t curr_base = curr_slot * slot_elems;
            LocalTensor<DT_X> src = ubuf[curr_base];
            LocalTensor<DT_X> dst = ubuf[curr_base + dst_offset];

            uint32_t h_out = row_idx + crop_top;
            uint32_t h_in = h_out >> 1u;
            uint32_t b1 = h_out & 1u;
            uint32_t even_batch = (b1 << 1u) + 1u;
            uint32_t odd_batch = b1 << 1u;
            uint64_t even_in_offset = static_cast<uint64_t>(even_batch) * input_stride_b
                                    + static_cast<uint64_t>(h_in) * input_stride_h
                                    + static_cast<uint64_t>(even_first_w_in) * fixed_depth;
            uint64_t odd_in_offset = static_cast<uint64_t>(odd_batch) * input_stride_b
                                   + static_cast<uint64_t>(h_in) * input_stride_h
                                   + static_cast<uint64_t>(odd_first_w_in) * fixed_depth;
            uint64_t out_offset = static_cast<uint64_t>(row_idx) * output_stride_h;

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            DataCopyPad(src[0u], xGM_[even_in_offset], read_params, pad_params);
            DataCopyPad(src[aligned_stream_elems], xGM_[odd_in_offset], read_params, pad_params);
            SetFlag<HardEvent::MTE2_V>(curr_event);

            if (has_prev) {
                event_t prev_event = (prev_slot == 0u) ? event0 : event1;
                WaitFlag<HardEvent::V_MTE3>(prev_event);
                DataCopyPad(yGM_[prev_out_offset],
                            ubuf[prev_slot * slot_elems + dst_offset], write_params);
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            WaitFlag<HardEvent::MTE2_V>(curr_event);
            Gather(dst, src, offsets, 0u, row_elems);
            SetFlag<HardEvent::V_MTE3>(curr_event);

            has_prev = true;
            prev_slot = curr_slot;
            prev_out_offset = out_offset;
        }

        if (has_prev) {
            event_t prev_event = (prev_slot == 0u) ? event0 : event1;
            WaitFlag<HardEvent::V_MTE3>(prev_event);
            DataCopyPad(yGM_[prev_out_offset],
                        ubuf[prev_slot * slot_elems + dst_offset], write_params);
            SetFlag<HardEvent::MTE3_MTE2>(prev_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
    }

    __aicore__ inline void ProcessScalarRows() {
        if (block_size_ == 1u) {
            ProcessScalarCropRows();
            return;
        }

        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();
        DataCopyExtParams read_params;
        read_params.blockCount = 1u;
        read_params.blockLen = depth_bytes_;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyExtParams write_params;
        write_params.blockCount = 1u;
        write_params.blockLen = depth_bytes_;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        DataCopyPadExtParams<DT_X> pad_params;
        pad_params.isPad = (depth_bytes_ % 32u) != 0u;
        pad_params.leftPadding = 0u;
        pad_params.rightPadding = static_cast<uint8_t>(depth_aligned_ - depth_);
        pad_params.paddingValue = static_cast<DT_X>(0);

        constexpr event_t event_id = EVENT_ID0;
        SetFlag<HardEvent::MTE3_MTE2>(event_id);

        for (uint32_t r = 0u; r < my_count_; ++r) {
            uint32_t row_idx = my_start_ + r;
            if (row_idx >= total_units_) {
                break;
            }
            uint32_t bi = row_idx / out_height_;
            uint32_t h_out_c = row_idx - bi * out_height_;
            uint32_t h_out = h_out_c + crop_top_;
            uint32_t h_in = h_out / block_size_;
            uint32_t b1 = h_out - h_in * block_size_;
            uint64_t out_row_base = static_cast<uint64_t>(bi) * out_stride_b_
                                  + static_cast<uint64_t>(h_out_c) * out_stride_h_;

            ProcessScalarRowChunk(ubuf, read_params, write_params, pad_params,
                                  bi, h_in, b1, out_row_base, 0u, out_width_, event_id);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
    }

    __aicore__ inline void ProcessScalarStreamRows() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        DataCopyExtParams read_params;
        read_params.blockCount = 1u;
        read_params.blockLen = depth_bytes_;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyExtParams write_params;
        write_params.blockCount = 1u;
        write_params.blockLen = depth_bytes_;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        DataCopyPadExtParams<DT_X> pad_params;
        pad_params.isPad = (depth_bytes_ % 32u) != 0u;
        pad_params.leftPadding = 0u;
        pad_params.rightPadding = static_cast<uint8_t>(depth_aligned_ - depth_);
        pad_params.paddingValue = static_cast<DT_X>(0);

        constexpr event_t event_id = EVENT_ID0;
        SetFlag<HardEvent::MTE3_MTE2>(event_id);

        uint32_t crop_left_mod = crop_left_ - (crop_left_ / block_size_) * block_size_;

        for (uint32_t r = 0u; r < my_count_; ++r) {
            uint32_t row_idx = my_start_ + r;
            if (row_idx >= total_units_) {
                break;
            }

            uint32_t bi = row_idx / out_height_;
            uint32_t h_out_c = row_idx - bi * out_height_;
            uint32_t h_out = h_out_c + crop_top_;
            uint32_t h_in = h_out / block_size_;
            uint32_t b1 = h_out - h_in * block_size_;

            uint64_t out_row_base = static_cast<uint64_t>(bi) * out_stride_b_
                                  + static_cast<uint64_t>(h_out_c) * out_stride_h_;
            uint64_t in_h_base = static_cast<uint64_t>(h_in) * in_stride_h_;

            for (uint32_t b2 = 0u; b2 < block_size_; ++b2) {
                uint32_t first_w_out_c = b2 + block_size_ - crop_left_mod;
                first_w_out_c -= (first_w_out_c / block_size_) * block_size_;
                if (first_w_out_c >= out_width_) {
                    continue;
                }

                uint32_t total_w = (out_width_ - first_w_out_c + block_size_ - 1u) / block_size_;
                uint32_t first_w_in = (first_w_out_c + crop_left_) / block_size_;
                uint32_t input_batch = (b1 * block_size_ + b2) * new_batch_ + bi;

                uint64_t in_base = static_cast<uint64_t>(input_batch) * in_stride_b_
                                 + in_h_base
                                 + static_cast<uint64_t>(first_w_in) * depth_;
                uint64_t out_base = out_row_base + static_cast<uint64_t>(first_w_out_c) * depth_;

                for (uint32_t done = 0u; done < total_w; done += ppb_) {
                    uint32_t chunk = MinU32(total_w - done, ppb_);

                    WaitFlag<HardEvent::MTE3_MTE2>(event_id);
                    for (uint32_t i = 0u; i < chunk; ++i) {
                        DataCopyPad(ubuf[i * depth_aligned_],
                                    xGM_[in_base + static_cast<uint64_t>(done + i) * depth_],
                                    read_params, pad_params);
                    }
                    SetFlag<HardEvent::MTE2_MTE3>(event_id);
                    WaitFlag<HardEvent::MTE2_MTE3>(event_id);

                    for (uint32_t i = 0u; i < chunk; ++i) {
                        DataCopyPad(yGM_[out_base + static_cast<uint64_t>(done + i) * block_size_ * depth_],
                                    ubuf[i * depth_aligned_], write_params);
                    }
                    SetFlag<HardEvent::MTE3_MTE2>(event_id);
                }
            }
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
    }

    __aicore__ inline void ProcessScalarRowChunk(LocalTensor<DT_X> ubuf,
                                                 DataCopyExtParams read_params,
                                                 DataCopyExtParams write_params,
                                                 DataCopyPadExtParams<DT_X> pad_params,
                                                 uint32_t bi,
                                                 uint32_t h_in,
                                                 uint32_t b1,
                                                 uint64_t out_row_base,
                                                 uint32_t w_start,
                                                 uint32_t count,
                                                 event_t event_id) {
        WaitFlag<HardEvent::MTE3_MTE2>(event_id);

        for (uint32_t i = 0u; i < count; ++i) {
            uint32_t w_out_c = w_start + i;
            uint32_t w_out = w_out_c + crop_left_;
            uint32_t w_in = w_out / block_size_;
            uint32_t b2 = w_out - w_in * block_size_;
            uint32_t input_batch = (b1 * block_size_ + b2) * new_batch_ + bi;
            uint64_t in_offset = static_cast<uint64_t>(input_batch) * in_stride_b_
                               + static_cast<uint64_t>(h_in) * in_stride_h_
                               + static_cast<uint64_t>(w_in) * depth_;
            DataCopyPad(ubuf[i * depth_aligned_], xGM_[in_offset], read_params, pad_params);
        }

        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);

        for (uint32_t i = 0u; i < count; ++i) {
            uint32_t w_out_c = w_start + i;
            uint64_t out_offset = out_row_base + static_cast<uint64_t>(w_out_c) * depth_;
            DataCopyPad(yGM_[out_offset], ubuf[i * depth_aligned_], write_params);
        }

        SetFlag<HardEvent::MTE3_MTE2>(event_id);
    }

    __aicore__ inline void ProcessScalarCropRows() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr event_t event_id = EVENT_ID0;
        SetFlag<HardEvent::MTE3_MTE2>(event_id);

        for (uint32_t r = 0u; r < my_count_; ++r) {
            uint32_t row_idx = my_start_ + r;
            if (row_idx >= total_units_) {
                break;
            }
            uint32_t bi = row_idx / out_height_;
            uint32_t h_out_c = row_idx - bi * out_height_;
            uint32_t h_in = h_out_c + crop_top_;

            for (uint32_t w_start = 0u; w_start < out_width_; w_start += ppb_) {
                uint32_t chunk = MinU32(out_width_ - w_start, ppb_);
                uint64_t in_offset = static_cast<uint64_t>(bi) * in_stride_b_
                                   + static_cast<uint64_t>(h_in) * in_stride_h_
                                   + static_cast<uint64_t>(crop_left_ + w_start) * depth_;
                uint64_t out_offset = static_cast<uint64_t>(bi) * out_stride_b_
                                    + static_cast<uint64_t>(h_out_c) * out_stride_h_
                                    + static_cast<uint64_t>(w_start) * depth_;
                CopyLinearChunk(ubuf, in_offset, out_offset, chunk, event_id);
            }
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
    }

    __aicore__ inline void ProcessFullCopy() {
        LocalTensor<DT_X> ubuf = buf_.template Get<DT_X>();

        constexpr event_t event_id = EVENT_ID0;
        SetFlag<HardEvent::MTE3_MTE2>(event_id);

        uint32_t done = 0u;
        while (done < my_count_) {
            uint32_t chunk = MinU32(my_count_ - done, ppb_);
            uint32_t pixel_start = my_start_ + done;
            uint64_t offset = static_cast<uint64_t>(pixel_start) * depth_;
            CopyLinearChunk(ubuf, offset, offset, chunk, event_id);
            done += chunk;
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
    }

    __aicore__ inline void CopyLinearChunk(LocalTensor<DT_X> ubuf,
                                           uint64_t in_offset,
                                           uint64_t out_offset,
                                           uint32_t pixels,
                                           event_t event_id) {
        uint32_t elem_count = pixels * depth_;
        uint32_t block_bytes = pixels * depth_bytes_;
        uint32_t aligned_bytes = ((block_bytes + 31u) / 32u) * 32u;
        uint32_t aligned_elems = aligned_bytes / static_cast<uint32_t>(sizeof(DT_X));

        DataCopyExtParams copy_params;
        copy_params.blockCount = 1u;
        copy_params.blockLen = block_bytes;
        copy_params.srcStride = 0u;
        copy_params.dstStride = 0u;

        DataCopyPadExtParams<DT_X> pad_params;
        pad_params.isPad = aligned_elems != elem_count;
        pad_params.leftPadding = 0u;
        pad_params.rightPadding = static_cast<uint8_t>(aligned_elems - elem_count);
        pad_params.paddingValue = static_cast<DT_X>(0);

        if ((depth_bytes_ % 32u) == 0u) {
            DataCopyParams copy_params_32;
            copy_params_32.blockCount = 1u;
            copy_params_32.blockLen = static_cast<uint16_t>(block_bytes / 32u);
            copy_params_32.srcStride = 0u;
            copy_params_32.dstStride = 0u;

            WaitFlag<HardEvent::MTE3_MTE2>(event_id);
            DataCopy(ubuf[0], xGM_[in_offset], copy_params_32);
            SetFlag<HardEvent::MTE2_MTE3>(event_id);
            WaitFlag<HardEvent::MTE2_MTE3>(event_id);
            DataCopy(yGM_[out_offset], ubuf[0], copy_params_32);
            SetFlag<HardEvent::MTE3_MTE2>(event_id);
            return;
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
        DataCopyPad(ubuf[0], xGM_[in_offset], copy_params, pad_params);
        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);
        DataCopyPad(yGM_[out_offset], ubuf[0], copy_params);
        SetFlag<HardEvent::MTE3_MTE2>(event_id);
    }

    GlobalTensor<DT_X> xGM_;
    GlobalTensor<DT_X> yGM_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> buf_;

    uint32_t height_;
    uint32_t width_;
    uint32_t depth_;
    uint32_t block_size_;
    uint32_t crop_top_;
    uint32_t crop_left_;
    uint32_t new_batch_;
    uint32_t out_height_;
    uint32_t out_width_;
    uint32_t depth_bytes_;
    uint32_t depth_aligned_;
    uint32_t ppb_;
    uint32_t total_units_;
    uint32_t per_core_units_;
    uint32_t tile_rows_;
    uint32_t copy_mode_;
    uint32_t depth_bytes_32_;
    uint32_t my_start_;
    uint32_t my_count_;
    uint64_t in_stride_b_;
    uint64_t in_stride_h_;
    uint64_t out_stride_b_;
    uint64_t out_stride_h_;
};

template <typename DT_X>
class KernelPoint3Tile4 {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kDepth = 64u;
        constexpr uint32_t kInWidth = 14u;
        constexpr uint32_t kOutHeight = 28u;
        constexpr uint32_t kOutWidth = 28u;
        constexpr uint32_t kTileRows = 4u;
        constexpr uint32_t kTilesPerBatch = 7u;
        constexpr uint32_t kRowElems = kOutWidth * kDepth;
        constexpr uint32_t kTileElems = kTileRows * kRowElems;
        constexpr uint32_t kInStrideB = 14u * kInWidth * kDepth;
        constexpr uint32_t kInStrideH = kInWidth * kDepth;
        constexpr uint32_t kOutStrideB = kOutHeight * kRowElems;
        constexpr uint32_t kDepthBlocks = 8u;
        constexpr uint32_t kStreamPixels = 14u;
        constexpr uint32_t kTotalElems = 16u * 14u * kInWidth * kDepth;

        uint32_t task_idx = GetBlockIdx();
        if (task_idx >= 28u) {
            return;
        }

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), kTotalElems);
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), kTotalElems);

        TPipe pipe;
        TBuf<TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, kTileElems * static_cast<uint32_t>(sizeof(DT_X)));
        LocalTensor<DT_X> ubuf = buf.template Get<DT_X>();

        DataCopyParams read_params;
        read_params.blockCount = kStreamPixels;
        read_params.blockLen = kDepthBlocks;
        read_params.srcStride = 0u;
        read_params.dstStride = kDepthBlocks;

        DataCopyParams write_params;
        write_params.blockCount = 1u;
        write_params.blockLen = kTileRows * kOutWidth * kDepthBlocks;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        uint32_t bi = task_idx / kTilesPerBatch;
        uint32_t tile_in_batch = task_idx - bi * kTilesPerBatch;
        uint32_t h_start = tile_in_batch * kTileRows;
        uint64_t out_base = static_cast<uint64_t>(bi) * kOutStrideB
                          + static_cast<uint64_t>(h_start) * kRowElems;

        for (uint32_t tr = 0u; tr < kTileRows; ++tr) {
            uint32_t h_out = h_start + tr;
            uint32_t h_in = h_out >> 1u;
            uint32_t b1 = h_out & 1u;
            uint32_t ub_row = tr * kRowElems;
            uint64_t in_base0 = (static_cast<uint64_t>(b1 << 1u) * 4u + bi) * kInStrideB
                              + static_cast<uint64_t>(h_in) * kInStrideH;
            DataCopy(ubuf[ub_row], x_gm[in_base0], read_params);
            DataCopy(ubuf[ub_row + kDepth], x_gm[in_base0 + 4ull * kInStrideB], read_params);
        }

        constexpr event_t event_id = EVENT_ID0;
        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);
        DataCopy(y_gm[out_base], ubuf[0u], write_params);
    }
};

template <typename DT_X>
class KernelPoint3Tile2 {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kDepth = 64u;
        constexpr uint32_t kInWidth = 14u;
        constexpr uint32_t kOutHeight = 28u;
        constexpr uint32_t kOutWidth = 28u;
        constexpr uint32_t kTileRows = 2u;
        constexpr uint32_t kTilesPerBatch = 14u;
        constexpr uint32_t kTotalTasks = 56u;
        constexpr uint32_t kLaunchBlocks = 40u;
        constexpr uint32_t kRowElems = kOutWidth * kDepth;
        constexpr uint32_t kTileElems = kTileRows * kRowElems;
        constexpr uint32_t kInStrideB = 14u * kInWidth * kDepth;
        constexpr uint32_t kInStrideH = kInWidth * kDepth;
        constexpr uint32_t kOutStrideB = kOutHeight * kRowElems;
        constexpr uint32_t kDepthBlocks = 8u;
        constexpr uint32_t kStreamPixels = 14u;
        constexpr uint32_t kTotalElems = 16u * 14u * kInWidth * kDepth;

        uint32_t block_idx = GetBlockIdx();
        if (block_idx >= kLaunchBlocks) {
            return;
        }

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), kTotalElems);
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), kTotalElems);

        TPipe pipe;
        TBuf<TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, kTileElems * static_cast<uint32_t>(sizeof(DT_X)));
        LocalTensor<DT_X> ubuf = buf.template Get<DT_X>();

        DataCopyParams read_params;
        read_params.blockCount = kStreamPixels;
        read_params.blockLen = kDepthBlocks;
        read_params.srcStride = 0u;
        read_params.dstStride = kDepthBlocks;

        DataCopyParams write_params;
        write_params.blockCount = 1u;
        write_params.blockLen = kTileRows * kOutWidth * kDepthBlocks;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        constexpr event_t event_id = EVENT_ID0;
        bool has_written = false;
        for (uint32_t task_idx = block_idx; task_idx < kTotalTasks; task_idx += kLaunchBlocks) {
            if (has_written) {
                WaitFlag<HardEvent::MTE3_MTE2>(event_id);
            }
            uint32_t bi = task_idx / kTilesPerBatch;
            uint32_t tile_in_batch = task_idx - bi * kTilesPerBatch;
            uint32_t h_start = tile_in_batch * kTileRows;
            uint64_t out_base = static_cast<uint64_t>(bi) * kOutStrideB
                              + static_cast<uint64_t>(h_start) * kRowElems;

            for (uint32_t tr = 0u; tr < kTileRows; ++tr) {
                uint32_t h_out = h_start + tr;
                uint32_t h_in = h_out >> 1u;
                uint32_t b1 = h_out & 1u;
                uint32_t ub_row = tr * kRowElems;
                uint64_t in_base0 = (static_cast<uint64_t>(b1 << 1u) * 4u + bi) * kInStrideB
                                  + static_cast<uint64_t>(h_in) * kInStrideH;
                DataCopy(ubuf[ub_row], x_gm[in_base0], read_params);
                DataCopy(ubuf[ub_row + kDepth], x_gm[in_base0 + 4ull * kInStrideB], read_params);
            }

            SetFlag<HardEvent::MTE2_MTE3>(event_id);
            WaitFlag<HardEvent::MTE2_MTE3>(event_id);
            DataCopy(y_gm[out_base], ubuf[0u], write_params);
            SetFlag<HardEvent::MTE3_MTE2>(event_id);
            has_written = true;
        }
        if (has_written) {
            WaitFlag<HardEvent::MTE3_MTE2>(event_id);
        }
    }
};

template <typename DT_X>
class KernelPoint3Tile2DoubleBuffer {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kDepth = 64u;
        constexpr uint32_t kInWidth = 14u;
        constexpr uint32_t kOutHeight = 28u;
        constexpr uint32_t kOutWidth = 28u;
        constexpr uint32_t kTileRows = 2u;
        constexpr uint32_t kTilesPerBatch = 14u;
        constexpr uint32_t kTotalTasks = 56u;
        constexpr uint32_t kLaunchBlocks = 40u;
        constexpr uint32_t kRowElems = kOutWidth * kDepth;
        constexpr uint32_t kTileElems = kTileRows * kRowElems;
        constexpr uint32_t kInStrideB = 14u * kInWidth * kDepth;
        constexpr uint32_t kInStrideH = kInWidth * kDepth;
        constexpr uint32_t kOutStrideB = kOutHeight * kRowElems;
        constexpr uint32_t kDepthBlocks = 8u;
        constexpr uint32_t kStreamPixels = 14u;
        constexpr uint32_t kTotalElems = 16u * 14u * kInWidth * kDepth;

        uint32_t block_idx = GetBlockIdx();
        if (block_idx >= kLaunchBlocks) {
            return;
        }

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), kTotalElems);
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), kTotalElems);

        TPipe pipe;
        TBuf<TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, 2u * kTileElems * static_cast<uint32_t>(sizeof(DT_X)));
        LocalTensor<DT_X> ubuf = buf.template Get<DT_X>();

        DataCopyParams read_params;
        read_params.blockCount = kStreamPixels;
        read_params.blockLen = kDepthBlocks;
        read_params.srcStride = 0u;
        read_params.dstStride = kDepthBlocks;

        DataCopyParams write_params;
        write_params.blockCount = 1u;
        write_params.blockLen = kTileRows * kOutWidth * kDepthBlocks;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);

        bool has_prev = false;
        uint32_t prev_slot = 0u;
        uint64_t prev_out_base = 0u;

        for (uint32_t task_idx = block_idx, iter = 0u; task_idx < kTotalTasks;
             task_idx += kLaunchBlocks, ++iter) {
            uint32_t curr_slot = iter & 1u;
            event_t curr_event = (curr_slot == 0u) ? event0 : event1;
            uint32_t curr_base = curr_slot * kTileElems;

            uint32_t bi = task_idx / kTilesPerBatch;
            uint32_t tile_in_batch = task_idx - bi * kTilesPerBatch;
            uint32_t h_start = tile_in_batch * kTileRows;
            uint64_t out_base = static_cast<uint64_t>(bi) * kOutStrideB
                              + static_cast<uint64_t>(h_start) * kRowElems;

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            for (uint32_t tr = 0u; tr < kTileRows; ++tr) {
                uint32_t h_out = h_start + tr;
                uint32_t h_in = h_out >> 1u;
                uint32_t b1 = h_out & 1u;
                uint32_t ub_row = curr_base + tr * kRowElems;
                uint64_t in_base0 = (static_cast<uint64_t>(b1 << 1u) * 4u + bi) * kInStrideB
                                  + static_cast<uint64_t>(h_in) * kInStrideH;
                DataCopy(ubuf[ub_row], x_gm[in_base0], read_params);
                DataCopy(ubuf[ub_row + kDepth], x_gm[in_base0 + 4ull * kInStrideB], read_params);
            }
            SetFlag<HardEvent::MTE2_MTE3>(curr_event);

            if (has_prev) {
                event_t prev_event = (prev_slot == 0u) ? event0 : event1;
                WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
                DataCopy(y_gm[prev_out_base], ubuf[prev_slot * kTileElems], write_params);
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            has_prev = true;
            prev_slot = curr_slot;
            prev_out_base = out_base;
        }

        if (has_prev) {
            event_t prev_event = (prev_slot == 0u) ? event0 : event1;
            WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
            DataCopy(y_gm[prev_out_base], ubuf[prev_slot * kTileElems], write_params);
            SetFlag<HardEvent::MTE3_MTE2>(prev_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
    }
};

template <typename DT_X>
class KernelPoint3Phase7 {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kDepth = 64u;
        constexpr uint32_t kInWidth = 14u;
        constexpr uint32_t kOutHeight = 28u;
        constexpr uint32_t kOutWidth = 28u;
        constexpr uint32_t kRows = 7u;
        constexpr uint32_t kRowElems = kOutWidth * kDepth;
        constexpr uint32_t kTileElems = kRows * kRowElems;
        constexpr uint32_t kInStrideB = 14u * kInWidth * kDepth;
        constexpr uint32_t kInStrideH = kInWidth * kDepth;
        constexpr uint32_t kOutStrideB = kOutHeight * kRowElems;
        constexpr uint32_t kDepthBlocks = 8u;
        constexpr uint32_t kStreamPixels = 14u;
        constexpr uint32_t kOutRowBlocks = kOutWidth * kDepthBlocks;
        constexpr uint32_t kTotalElems = 16u * 14u * kInWidth * kDepth;

        uint32_t task_idx = GetBlockIdx();
        if (task_idx >= 16u) {
            return;
        }

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), kTotalElems);
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), kTotalElems);

        TPipe pipe;
        TBuf<TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, kTileElems * static_cast<uint32_t>(sizeof(DT_X)));
        LocalTensor<DT_X> ubuf = buf.template Get<DT_X>();

        DataCopyParams read_params;
        read_params.blockCount = kStreamPixels;
        read_params.blockLen = kDepthBlocks;
        read_params.srcStride = 0u;
        read_params.dstStride = kDepthBlocks;

        DataCopyParams write_params;
        write_params.blockCount = kRows;
        write_params.blockLen = kOutRowBlocks;
        write_params.srcStride = 0u;
        write_params.dstStride = kOutRowBlocks;

        uint32_t bi = task_idx >> 2u;
        uint32_t local = task_idx & 3u;
        uint32_t tile_hi = local >> 1u;
        uint32_t row_phase = local & 1u;
        uint32_t h_in_start = tile_hi * kRows;
        uint32_t h_out_start = (h_in_start << 1u) + row_phase;
        uint64_t out_base = static_cast<uint64_t>(bi) * kOutStrideB
                          + static_cast<uint64_t>(h_out_start) * kRowElems;
        uint64_t in_batch0_base = (static_cast<uint64_t>(row_phase << 1u) * 4u + bi) * kInStrideB
                                + static_cast<uint64_t>(h_in_start) * kInStrideH;
        uint64_t in_batch1_base = in_batch0_base + 4ull * kInStrideB;

        for (uint32_t r = 0u; r < kRows; ++r) {
            uint32_t ub_row = r * kRowElems;
            uint64_t row_offset = static_cast<uint64_t>(r) * kInStrideH;
            DataCopy(ubuf[ub_row], x_gm[in_batch0_base + row_offset], read_params);
            DataCopy(ubuf[ub_row + kDepth], x_gm[in_batch1_base + row_offset], read_params);
        }

        constexpr event_t event_id = EVENT_ID0;
        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);
        DataCopy(y_gm[out_base], ubuf[0u], write_params);
        SetFlag<HardEvent::MTE3_MTE2>(event_id);
        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
    }
};

template <typename DT_X>
class KernelPoint10Tile8Tpl {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kTileRows = 8u;
        constexpr uint32_t kOutHeight = 2048u;
        constexpr uint32_t kOutWidth = 12u;
        constexpr uint32_t kDepth = 32u;
        constexpr uint32_t kInputHeight = 1024u;
        constexpr uint32_t kInputWidth = 6u;
        constexpr uint32_t kOutBatch = 4u;
        constexpr uint32_t kTilesPerBatch = kOutHeight / kTileRows;
        constexpr uint32_t kTotalTasks = kOutBatch * kTilesPerBatch;
        constexpr uint32_t kLaunchBlocks = 40u;
        constexpr uint32_t kDepthBlocks = 2u;
        constexpr uint32_t kStreamPixels = 6u;
        constexpr uint32_t kRowElems = kOutWidth * kDepth;
        constexpr uint32_t kTileElems = kTileRows * kRowElems;
        constexpr uint64_t kInStrideB = static_cast<uint64_t>(kInputHeight) * kInputWidth * kDepth;
        constexpr uint64_t kInStrideH = static_cast<uint64_t>(kInputWidth) * kDepth;
        constexpr uint64_t kOutStrideB = static_cast<uint64_t>(kOutHeight) * kRowElems;
        constexpr uint32_t kTotalElems = 16u * kInputHeight * kInputWidth * kDepth;

        uint32_t block_idx = GetBlockIdx();
        if (block_idx >= kLaunchBlocks) {
            return;
        }

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), kTotalElems);
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), kTotalElems);

        TPipe pipe;
        TBuf<TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, 2u * kTileElems * static_cast<uint32_t>(sizeof(DT_X)));
        LocalTensor<DT_X> ubuf = buf.template Get<DT_X>();

        DataCopyParams read_params;
        read_params.blockCount = kStreamPixels;
        read_params.blockLen = kDepthBlocks;
        read_params.srcStride = 0u;
        read_params.dstStride = kDepthBlocks;

        DataCopyParams write_params;
        write_params.blockCount = 1u;
        write_params.blockLen = kTileRows * kOutWidth * kDepthBlocks;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        constexpr event_t event0 = EVENT_ID0;
        constexpr event_t event1 = EVENT_ID1;
        SetFlag<HardEvent::MTE3_MTE2>(event0);
        SetFlag<HardEvent::MTE3_MTE2>(event1);

        bool has_prev = false;
        uint32_t prev_slot = 0u;
        uint64_t prev_out_base = 0u;

        for (uint32_t task_idx = block_idx, iter = 0u; task_idx < kTotalTasks;
             task_idx += kLaunchBlocks, ++iter) {
            uint32_t curr_slot = iter & 1u;
            event_t curr_event = (curr_slot == 0u) ? event0 : event1;
            uint32_t curr_base = curr_slot * kTileElems;

            uint32_t bi = task_idx >> 8u;
            uint32_t tile_in_batch = task_idx & 255u;
            uint32_t h_start = tile_in_batch * kTileRows;
            uint64_t out_base = static_cast<uint64_t>(bi) * kOutStrideB
                              + static_cast<uint64_t>(h_start) * kRowElems;

            WaitFlag<HardEvent::MTE3_MTE2>(curr_event);
            for (uint32_t tr = 0u; tr < kTileRows; ++tr) {
                uint32_t h_out = h_start + tr;
                uint32_t h_in = h_out >> 1u;
                uint32_t b1 = h_out & 1u;
                uint32_t ub_row = curr_base + tr * kRowElems;
                uint64_t in_base0 = (static_cast<uint64_t>(b1 << 1u) * kOutBatch + bi) * kInStrideB
                                  + static_cast<uint64_t>(h_in) * kInStrideH;
                DataCopy(ubuf[ub_row], x_gm[in_base0], read_params);
                DataCopy(ubuf[ub_row + kDepth], x_gm[in_base0 + static_cast<uint64_t>(kOutBatch) * kInStrideB],
                         read_params);
            }
            SetFlag<HardEvent::MTE2_MTE3>(curr_event);

            if (has_prev) {
                event_t prev_event = (prev_slot == 0u) ? event0 : event1;
                WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
                DataCopy(y_gm[prev_out_base], ubuf[prev_slot * kTileElems], write_params);
                SetFlag<HardEvent::MTE3_MTE2>(prev_event);
            }

            has_prev = true;
            prev_slot = curr_slot;
            prev_out_base = out_base;
        }

        if (has_prev) {
            event_t prev_event = (prev_slot == 0u) ? event0 : event1;
            WaitFlag<HardEvent::MTE2_MTE3>(prev_event);
            DataCopy(y_gm[prev_out_base], ubuf[prev_slot * kTileElems], write_params);
            SetFlag<HardEvent::MTE3_MTE2>(prev_event);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(event0);
        WaitFlag<HardEvent::MTE3_MTE2>(event1);
    }
};

template <typename DT_X>
__aicore__ inline void RunPoint10BetaCase9(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kXH = 1024u;
    constexpr uint32_t kXW = 6u;
    constexpr uint32_t kXC = 32u;
    constexpr uint32_t kYN = 4u;
    constexpr uint32_t kYH = 2048u;
    constexpr uint32_t kYW = 12u;
    constexpr uint32_t kBlockSize = 2u;
    constexpr uint32_t kTileRows = 52u;
    constexpr uint32_t kTilesPerBlockH = (kXH + kTileRows - 1u) / kTileRows;
    constexpr uint32_t kMaxCopyRows = 52u;
    constexpr uint32_t kDBlocks = (kXC * static_cast<uint32_t>(sizeof(DT_X))) / 32u;
    constexpr uint32_t kSourceRowBlocks = (kXW * kXC * static_cast<uint32_t>(sizeof(DT_X))) / 32u;
    constexpr uint32_t kOutRowBlocks = (kYW * kXC * static_cast<uint32_t>(sizeof(DT_X))) / 32u;
    constexpr uint32_t kMaxInputElems = kTileRows * kXW * kXC;
    constexpr uint32_t kMaxOutputElems = kTileRows * kYW * kXC;
    constexpr uint32_t kInputBytes =
        ((kMaxInputElems * static_cast<uint32_t>(sizeof(DT_X)) + 511u) / 512u) * 512u;

    GlobalTensor<DT_X> x_gm;
    GlobalTensor<DT_X> y_gm;
    x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
    y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
    LocalTensor<DT_X> input0(TPosition::VECCALC, 0, kMaxInputElems);
    LocalTensor<DT_X> input1(TPosition::VECCALC, kInputBytes, kMaxInputElems);
    LocalTensor<DT_X> output(TPosition::VECCALC, 2u * kInputBytes, kMaxOutputElems);

    DataCopyParams load_params;
    load_params.blockCount = 1u;
    load_params.blockLen = 0u;
    load_params.srcStride = 0u;
    load_params.dstStride = 0u;

    DataCopyParams ub_params;
    ub_params.blockLen = static_cast<uint16_t>(kDBlocks);
    ub_params.srcStride = 0u;
    ub_params.dstStride = static_cast<uint16_t>(kDBlocks);

    DataCopyParams store_params;
    store_params.blockLen = static_cast<uint16_t>(kOutRowBlocks);
    store_params.srcStride = 0u;
    store_params.dstStride = static_cast<uint16_t>(kOutRowBlocks);

    uint32_t block_idx = GetBlockIdx();
    uint32_t block_num = GetBlockNum();
    for (uint32_t group = block_idx; group < kYN * kBlockSize * kTilesPerBlockH; group += block_num) {
        uint32_t tile = group % kTilesPerBlockH;
        uint32_t tmp = group / kTilesPerBlockH;
        uint32_t block_h = tmp & 1u;
        uint32_t n = tmp >> 1u;
        uint32_t ih = tile * kTileRows;
        uint32_t rows = kXH - ih;
        if (rows > kTileRows) {
            rows = kTileRows;
        }
        uint32_t input_n0 = block_h * kBlockSize * kYN + n;
        uint32_t input_n1 = input_n0 + kYN;
        uint32_t src0 = ((input_n0 * kXH + ih) * kXW) * kXC;
        uint32_t src1 = ((input_n1 * kXH + ih) * kXW) * kXC;

        load_params.blockLen = static_cast<uint16_t>(rows * kSourceRowBlocks);
        DataCopy(input0, x_gm[src0], load_params);
        DataCopy(input1, x_gm[src1], load_params);
        PipeBarrier<PIPE_ALL>();

        for (uint32_t row_base = 0u; row_base < rows; row_base += kMaxCopyRows) {
            uint32_t copy_rows = rows - row_base;
            if (copy_rows > kMaxCopyRows) {
                copy_rows = kMaxCopyRows;
            }
            uint32_t in_offset = row_base * kXW * kXC;
            uint32_t out_offset = row_base * kYW * kXC;
            ub_params.blockCount = static_cast<uint16_t>(copy_rows * kXW);
            DataCopy(output[out_offset], input0[in_offset], ub_params);
            DataCopy(output[out_offset + kXC], input1[in_offset], ub_params);
            PipeBarrier<PIPE_ALL>();
        }

        store_params.blockCount = static_cast<uint16_t>(rows);
        uint32_t dst = (n * kYH + block_h + ih * kBlockSize) * kYW * kXC;
        DataCopy(y_gm[dst], output, store_params);
        PipeBarrier<PIPE_MTE3>();
    }
}

template <typename DT_X>
__aicore__ inline void RunPoint8BetaCase7(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kXH = 10u;
    constexpr uint32_t kXW = 512u;
    constexpr uint32_t kXC = 256u;
    constexpr uint32_t kYH = 20u;
    constexpr uint32_t kYW = 1024u;
    constexpr uint32_t kBlockSize = 2u;
    constexpr uint32_t kTileCols = 64u;
    constexpr uint32_t kTileCount = (kXW + kTileCols - 1u) / kTileCols;
    constexpr uint32_t kDBlocks = (kXC * static_cast<uint32_t>(sizeof(DT_X))) / 32u;
    constexpr uint32_t kInputElems = kTileCols * kXC;
    constexpr uint32_t kOutputElems = kTileCols * kBlockSize * kXC;
    constexpr uint32_t kOutputBytes = kOutputElems * static_cast<uint32_t>(sizeof(DT_X));

    GlobalTensor<DT_X> x_gm;
    GlobalTensor<DT_X> y_gm;
    x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
    y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
    LocalTensor<DT_X> output_local(TPosition::VECCALC, 0, kOutputElems);
    LocalTensor<DT_X> input_local(TPosition::VECCALC, kOutputBytes, kInputElems);

    DataCopyParams load_params;
    load_params.blockCount = 1u;
    load_params.blockLen = 0u;
    load_params.srcStride = 0u;
    load_params.dstStride = 0u;

    DataCopyParams ub_params;
    ub_params.blockLen = static_cast<uint16_t>(kDBlocks);
    ub_params.srcStride = 0u;
    ub_params.dstStride = static_cast<uint16_t>(kDBlocks);

    DataCopyParams store_params;
    store_params.blockCount = 1u;
    store_params.blockLen = 0u;
    store_params.srcStride = 0u;
    store_params.dstStride = 0u;

    uint32_t block_idx = GetBlockIdx();
    uint32_t block_num = GetBlockNum();
    for (uint32_t task = block_idx; task < kYH * kTileCount; task += block_num) {
        uint32_t tile = task & 7u;
        uint32_t oh = task >> 3u;
        uint32_t col_base = tile * kTileCols;
        uint32_t ih = oh >> 1u;
        uint32_t block_h = oh & 1u;
        uint32_t input_n0 = block_h * kBlockSize;
        uint32_t input_n1 = input_n0 + 1u;
        uint32_t src0 = ((input_n0 * kXH + ih) * kXW + col_base) * kXC;
        uint32_t src1 = ((input_n1 * kXH + ih) * kXW + col_base) * kXC;

        load_params.blockLen = static_cast<uint16_t>(kTileCols * kDBlocks);
        DataCopy(input_local, x_gm[src0], load_params);
        PipeBarrier<PIPE_ALL>();

        ub_params.blockCount = static_cast<uint16_t>(kTileCols);
        DataCopy(output_local, input_local, ub_params);
        PipeBarrier<PIPE_ALL>();

        DataCopy(input_local, x_gm[src1], load_params);
        PipeBarrier<PIPE_ALL>();

        DataCopy(output_local[kXC], input_local, ub_params);
        PipeBarrier<PIPE_ALL>();

        uint32_t dst = (oh * kYW + col_base * kBlockSize) * kXC;
        store_params.blockLen = static_cast<uint16_t>(kTileCols * kBlockSize * kDBlocks);
        DataCopy(y_gm[dst], output_local, store_params);
        PipeBarrier<PIPE_MTE3>();
    }
}

template <typename DT_X>
__aicore__ inline void RunPoint9BetaCase8(GM_ADDR x, GM_ADDR y) {
    constexpr uint32_t kXH = 10u;
    constexpr uint32_t kXW = 512u;
    constexpr uint32_t kXC = 64u;
    constexpr uint32_t kYH = 40u;
    constexpr uint32_t kYW = 1535u;
    constexpr uint32_t kBlockSize = 4u;
    constexpr uint32_t kCropLeft = 513u;
    constexpr uint32_t kTileCols = 512u;
    constexpr uint32_t kTileCount = 3u;
    constexpr uint32_t kGroupCols = kTileCols / kBlockSize;
    constexpr uint32_t kCBytes = kXC * static_cast<uint32_t>(sizeof(DT_X));
    constexpr uint32_t kDBlocks = kCBytes / 32u;
    constexpr uint32_t kGroupBytes = kGroupCols * kCBytes;
    constexpr uint32_t kGroupElems = kGroupBytes / static_cast<uint32_t>(sizeof(DT_X));
    constexpr uint32_t kTileElems = kTileCols * kXC;
    constexpr uint32_t kTwoGroupElems = 2u * kGroupElems;

    GlobalTensor<DT_X> x_gm;
    GlobalTensor<DT_X> y_gm;
    x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
    y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
    LocalTensor<DT_X> output_local(TPosition::VECCALC, 0, kTileElems);
    LocalTensor<DT_X> input01(TPosition::VECCALC, 65536u, kTwoGroupElems);
    LocalTensor<DT_X> input23(TPosition::VECCALC, 98304u, kTwoGroupElems);

    DataCopyParams load_params;
    load_params.blockCount = 1u;
    load_params.blockLen = 0u;
    load_params.srcStride = 0u;
    load_params.dstStride = 0u;

    DataCopyParams ub_params;
    ub_params.blockLen = static_cast<uint16_t>(kDBlocks);
    ub_params.srcStride = 0u;
    ub_params.dstStride = static_cast<uint16_t>((kBlockSize - 1u) * kDBlocks);

    DataCopyParams store_params;
    store_params.blockCount = 1u;
    store_params.blockLen = 0u;
    store_params.srcStride = 0u;
    store_params.dstStride = 0u;

    uint32_t block_idx = GetBlockIdx();
    uint32_t block_num = GetBlockNum();
    for (uint32_t oh = block_idx; oh < kYH; oh += block_num) {
        uint32_t ih = oh / kBlockSize;
        uint32_t block_h = oh - ih * kBlockSize;
        uint32_t input_base = block_h * kBlockSize;

        for (uint32_t tile = 0u; tile < kTileCount; ++tile) {
            uint32_t tile_ow = tile * kTileCols;
            uint32_t cols = kYW - tile_ow;
            if (cols > kTileCols) {
                cols = kTileCols;
            }
            uint32_t iw_base = (tile_ow + kCropLeft) / kBlockSize;
            uint32_t count0 = cols == kTileCols ? kGroupCols : kGroupCols - 1u;
            uint32_t src_index0 = ((input_base * kXH + ih) * kXW + iw_base + 1u) * kXC;
            uint32_t src_index1 = (((input_base + 1u) * kXH + ih) * kXW + iw_base) * kXC;
            uint32_t src_index2 = (((input_base + 2u) * kXH + ih) * kXW + iw_base) * kXC;
            uint32_t src_index3 = (((input_base + 3u) * kXH + ih) * kXW + iw_base) * kXC;

            load_params.blockLen = static_cast<uint16_t>((count0 * kCBytes) / 32u);
            DataCopy(input01, x_gm[src_index0], load_params);
            load_params.blockLen = static_cast<uint16_t>((kGroupCols * kCBytes) / 32u);
            DataCopy(input01[kGroupElems], x_gm[src_index1], load_params);
            DataCopy(input23, x_gm[src_index2], load_params);
            DataCopy(input23[kGroupElems], x_gm[src_index3], load_params);
            PipeBarrier<PIPE_ALL>();

            ub_params.blockCount = static_cast<uint16_t>(count0);
            DataCopy(output_local[3u * kXC], input01, ub_params);
            ub_params.blockCount = static_cast<uint16_t>(kGroupCols);
            DataCopy(output_local, input01[kGroupElems], ub_params);
            DataCopy(output_local[kXC], input23, ub_params);
            DataCopy(output_local[2u * kXC], input23[kGroupElems], ub_params);
            PipeBarrier<PIPE_ALL>();

            uint32_t dst_index = (oh * kYW + tile_ow) * kXC;
            store_params.blockLen = static_cast<uint16_t>((cols * kCBytes) / 32u);
            DataCopy(y_gm[dst_index], output_local, store_params);
            if (tile + 1u < kTileCount || oh + block_num < kYH) {
                PipeBarrier<PIPE_MTE3>();
            }
        }
    }
}

template <typename DT_X>
class KernelPoint1BetaCase0 {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kXH = 28u;
        constexpr uint32_t kXW = 28u;
        constexpr uint32_t kXC = 128u;
        constexpr uint32_t kYN = 2u;
        constexpr uint32_t kYH = 56u;
        constexpr uint32_t kYW = 56u;
        constexpr uint32_t kDBlocks = (kXC * static_cast<uint32_t>(sizeof(DT_X))) / 32u;
        constexpr uint32_t kSourceRowBlocks = (kXW * kXC * static_cast<uint32_t>(sizeof(DT_X))) / 32u;
        constexpr uint32_t kOutRowBlocks = (kYW * kXC * static_cast<uint32_t>(sizeof(DT_X))) / 32u;
        constexpr uint32_t kInputElems = kXW * kXC;
        constexpr uint32_t kOutputElems = 2u * kYW * kXC;
        constexpr uint32_t kInput1OffsetBytes = 16u * 1024u;
        constexpr uint32_t kInput2OffsetBytes = 32u * 1024u;
        constexpr uint32_t kInput3OffsetBytes = 48u * 1024u;
        constexpr uint32_t kOutputOffsetBytes = 64u * 1024u;
        constexpr uint32_t kInBatchStride = kXH * kXW * kXC;
        constexpr uint32_t kInRowStride = kXW * kXC;

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

        LocalTensor<DT_X> input0(TPosition::VECCALC, 0, kInputElems);
        LocalTensor<DT_X> input1(TPosition::VECCALC, kInput1OffsetBytes, kInputElems);
        LocalTensor<DT_X> input2(TPosition::VECCALC, kInput2OffsetBytes, kInputElems);
        LocalTensor<DT_X> input3(TPosition::VECCALC, kInput3OffsetBytes, kInputElems);
        LocalTensor<DT_X> output(TPosition::VECCALC, kOutputOffsetBytes, kOutputElems);

        DataCopyParams load_params;
        load_params.blockCount = 1u;
        load_params.blockLen = static_cast<uint16_t>(kSourceRowBlocks);
        load_params.srcStride = 0u;
        load_params.dstStride = 0u;

        DataCopyParams ub_params;
        ub_params.blockCount = static_cast<uint16_t>(kXW);
        ub_params.blockLen = static_cast<uint16_t>(kDBlocks);
        ub_params.srcStride = 0u;
        ub_params.dstStride = static_cast<uint16_t>(kDBlocks);

        DataCopyParams store_params;
        store_params.blockCount = 1u;
        store_params.blockLen = static_cast<uint16_t>(2u * kOutRowBlocks);
        store_params.srcStride = 0u;
        store_params.dstStride = 0u;

        uint32_t block_idx = GetBlockIdx();
        uint32_t block_num = GetBlockNum();
        for (uint32_t group = block_idx; group < kYN * kXH; group += block_num) {
            uint32_t ih = group % kXH;
            uint32_t n = group / kXH;
            uint32_t input_n0 = n;
            uint32_t input_n1 = input_n0 + kYN;
            uint32_t input_n2 = 2u * kYN + n;
            uint32_t input_n3 = input_n2 + kYN;
            uint32_t src0 = input_n0 * kInBatchStride + ih * kInRowStride;
            uint32_t src1 = input_n1 * kInBatchStride + ih * kInRowStride;
            uint32_t src2 = input_n2 * kInBatchStride + ih * kInRowStride;
            uint32_t src3 = input_n3 * kInBatchStride + ih * kInRowStride;

            DataCopy(input0, x_gm[src0], load_params);
            DataCopy(input1, x_gm[src1], load_params);
            DataCopy(input2, x_gm[src2], load_params);
            DataCopy(input3, x_gm[src3], load_params);
            PipeBarrier<PIPE_ALL>();

            DataCopy(output, input0, ub_params);
            DataCopy(output[kXC], input1, ub_params);
            DataCopy(output[kYW * kXC], input2, ub_params);
            DataCopy(output[kYW * kXC + kXC], input3, ub_params);
            PipeBarrier<PIPE_MTE2>();

            uint32_t dst = (n * kYH + ih * 2u) * kYW * kXC;
            DataCopy(y_gm[dst], output, store_params);
            PipeBarrier<PIPE_MTE3>();
        }
    }
};

template <typename DT_X>
class KernelPoint1BetaHalfWidth {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kXH = 28u;
        constexpr uint32_t kXW = 28u;
        constexpr uint32_t kHalfXW = 14u;
        constexpr uint32_t kXC = 128u;
        constexpr uint32_t kYN = 2u;
        constexpr uint32_t kYH = 56u;
        constexpr uint32_t kYW = 56u;
        constexpr uint32_t kHalfYW = 28u;
        constexpr uint32_t kDBlocks = (kXC * static_cast<uint32_t>(sizeof(DT_X))) / 32u;
        constexpr uint32_t kHalfSourceRowBlocks = (kHalfXW * kXC * static_cast<uint32_t>(sizeof(DT_X))) / 32u;
        constexpr uint32_t kHalfOutRowBlocks = (kHalfYW * kXC * static_cast<uint32_t>(sizeof(DT_X))) / 32u;
        constexpr uint32_t kHalfInputElems = kHalfXW * kXC;
        constexpr uint32_t kHalfOutRowElems = kHalfYW * kXC;
        constexpr uint32_t kInput1OffsetBytes = 8u * 1024u;
        constexpr uint32_t kInput2OffsetBytes = 16u * 1024u;
        constexpr uint32_t kInput3OffsetBytes = 24u * 1024u;
        constexpr uint32_t kOutputOffsetBytes = 32u * 1024u;
        constexpr uint32_t kInBatchStride = kXH * kXW * kXC;
        constexpr uint32_t kInRowStride = kXW * kXC;
        constexpr uint32_t kOutRowStride = kYW * kXC;

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

        LocalTensor<DT_X> input0(TPosition::VECCALC, 0, kHalfInputElems);
        LocalTensor<DT_X> input1(TPosition::VECCALC, kInput1OffsetBytes, kHalfInputElems);
        LocalTensor<DT_X> input2(TPosition::VECCALC, kInput2OffsetBytes, kHalfInputElems);
        LocalTensor<DT_X> input3(TPosition::VECCALC, kInput3OffsetBytes, kHalfInputElems);
        LocalTensor<DT_X> output(TPosition::VECCALC, kOutputOffsetBytes, 2u * kHalfOutRowElems);

        DataCopyParams load_params;
        load_params.blockCount = 1u;
        load_params.blockLen = static_cast<uint16_t>(kHalfSourceRowBlocks);
        load_params.srcStride = 0u;
        load_params.dstStride = 0u;

        DataCopyParams ub_params;
        ub_params.blockCount = static_cast<uint16_t>(kHalfXW);
        ub_params.blockLen = static_cast<uint16_t>(kDBlocks);
        ub_params.srcStride = 0u;
        ub_params.dstStride = static_cast<uint16_t>(kDBlocks);

        DataCopyParams store_params;
        store_params.blockCount = 2u;
        store_params.blockLen = static_cast<uint16_t>(kHalfOutRowBlocks);
        store_params.srcStride = 0u;
        store_params.dstStride = static_cast<uint16_t>(kHalfOutRowBlocks);

        uint32_t block_idx = GetBlockIdx();
        uint32_t block_num = GetBlockNum();
        for (uint32_t task = block_idx; task < kYN * kXH * 2u; task += block_num) {
            uint32_t half = task & 1u;
            uint32_t group = task >> 1u;
            uint32_t ih = group % kXH;
            uint32_t n = group / kXH;
            uint32_t input_n0 = n;
            uint32_t input_n1 = input_n0 + kYN;
            uint32_t input_n2 = 2u * kYN + n;
            uint32_t input_n3 = input_n2 + kYN;
            uint32_t half_input_offset = half * kHalfInputElems;
            uint32_t src0 = input_n0 * kInBatchStride + ih * kInRowStride + half_input_offset;
            uint32_t src1 = input_n1 * kInBatchStride + ih * kInRowStride + half_input_offset;
            uint32_t src2 = input_n2 * kInBatchStride + ih * kInRowStride + half_input_offset;
            uint32_t src3 = input_n3 * kInBatchStride + ih * kInRowStride + half_input_offset;

            DataCopy(input0, x_gm[src0], load_params);
            DataCopy(input1, x_gm[src1], load_params);
            DataCopy(input2, x_gm[src2], load_params);
            DataCopy(input3, x_gm[src3], load_params);
            PipeBarrier<PIPE_ALL>();

            DataCopy(output, input0, ub_params);
            DataCopy(output[kXC], input1, ub_params);
            DataCopy(output[kHalfOutRowElems], input2, ub_params);
            DataCopy(output[kHalfOutRowElems + kXC], input3, ub_params);
            PipeBarrier<PIPE_MTE2>();

            uint32_t dst = (n * kYH + ih * 2u) * kOutRowStride + half * kHalfOutRowElems;
            DataCopy(y_gm[dst], output, store_params);
            PipeBarrier<PIPE_MTE3>();
        }
    }
};

template <typename DT_X, uint32_t SPLITS>
class KernelPoint1BetaSplitWidth {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kXH = 28u;
        constexpr uint32_t kXW = 28u;
        constexpr uint32_t kChunkXW = kXW / SPLITS;
        constexpr uint32_t kXC = 128u;
        constexpr uint32_t kYN = 2u;
        constexpr uint32_t kYH = 56u;
        constexpr uint32_t kYW = 56u;
        constexpr uint32_t kChunkYW = 2u * kChunkXW;
        constexpr uint32_t kDBlocks = (kXC * static_cast<uint32_t>(sizeof(DT_X))) / 32u;
        constexpr uint32_t kChunkSourceRowBlocks =
            (kChunkXW * kXC * static_cast<uint32_t>(sizeof(DT_X))) / 32u;
        constexpr uint32_t kChunkOutRowBlocks =
            (kChunkYW * kXC * static_cast<uint32_t>(sizeof(DT_X))) / 32u;
        constexpr uint32_t kChunkInputElems = kChunkXW * kXC;
        constexpr uint32_t kChunkOutRowElems = kChunkYW * kXC;
        constexpr uint32_t kInput1OffsetBytes = 4u * 1024u;
        constexpr uint32_t kInput2OffsetBytes = 8u * 1024u;
        constexpr uint32_t kInput3OffsetBytes = 12u * 1024u;
        constexpr uint32_t kOutputOffsetBytes = 16u * 1024u;
        constexpr uint32_t kInBatchStride = kXH * kXW * kXC;
        constexpr uint32_t kInRowStride = kXW * kXC;
        constexpr uint32_t kOutRowStride = kYW * kXC;

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

        LocalTensor<DT_X> input0(TPosition::VECCALC, 0, kChunkInputElems);
        LocalTensor<DT_X> input1(TPosition::VECCALC, kInput1OffsetBytes, kChunkInputElems);
        LocalTensor<DT_X> input2(TPosition::VECCALC, kInput2OffsetBytes, kChunkInputElems);
        LocalTensor<DT_X> input3(TPosition::VECCALC, kInput3OffsetBytes, kChunkInputElems);
        LocalTensor<DT_X> output(TPosition::VECCALC, kOutputOffsetBytes, 2u * kChunkOutRowElems);

        DataCopyParams load_params;
        load_params.blockCount = 1u;
        load_params.blockLen = static_cast<uint16_t>(kChunkSourceRowBlocks);
        load_params.srcStride = 0u;
        load_params.dstStride = 0u;

        DataCopyParams ub_params;
        ub_params.blockCount = static_cast<uint16_t>(kChunkXW);
        ub_params.blockLen = static_cast<uint16_t>(kDBlocks);
        ub_params.srcStride = 0u;
        ub_params.dstStride = static_cast<uint16_t>(kDBlocks);

        DataCopyParams store_params;
        store_params.blockCount = 2u;
        store_params.blockLen = static_cast<uint16_t>(kChunkOutRowBlocks);
        store_params.srcStride = 0u;
        store_params.dstStride = static_cast<uint16_t>((kOutRowStride - kChunkOutRowElems) *
                                                       static_cast<uint32_t>(sizeof(DT_X)) / 32u);

        uint32_t block_idx = GetBlockIdx();
        uint32_t block_num = GetBlockNum();
        for (uint32_t task = block_idx; task < kYN * kXH * SPLITS; task += block_num) {
            uint32_t split = task % SPLITS;
            uint32_t group = task / SPLITS;
            uint32_t ih = group % kXH;
            uint32_t n = group / kXH;
            uint32_t input_n0 = n;
            uint32_t input_n1 = input_n0 + kYN;
            uint32_t input_n2 = 2u * kYN + n;
            uint32_t input_n3 = input_n2 + kYN;
            uint32_t input_offset = split * kChunkInputElems;
            uint32_t output_offset = split * kChunkOutRowElems;
            uint32_t src0 = input_n0 * kInBatchStride + ih * kInRowStride + input_offset;
            uint32_t src1 = input_n1 * kInBatchStride + ih * kInRowStride + input_offset;
            uint32_t src2 = input_n2 * kInBatchStride + ih * kInRowStride + input_offset;
            uint32_t src3 = input_n3 * kInBatchStride + ih * kInRowStride + input_offset;

            DataCopy(input0, x_gm[src0], load_params);
            DataCopy(input1, x_gm[src1], load_params);
            DataCopy(input2, x_gm[src2], load_params);
            DataCopy(input3, x_gm[src3], load_params);
            PipeBarrier<PIPE_ALL>();

            DataCopy(output, input0, ub_params);
            DataCopy(output[kXC], input1, ub_params);
            DataCopy(output[kChunkOutRowElems], input2, ub_params);
            DataCopy(output[kChunkOutRowElems + kXC], input3, ub_params);
            PipeBarrier<PIPE_MTE2>();

            uint32_t dst = (n * kYH + ih * 2u) * kOutRowStride + output_offset;
            DataCopy(y_gm[dst], output, store_params);
            PipeBarrier<PIPE_MTE3>();
        }
    }
};

template <typename DT_X>
class KernelPoint2BetaCase1 {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kXH = 10u;
        constexpr uint32_t kXW = 15u;
        constexpr uint32_t kXC = 5u;
        constexpr uint32_t kYH = 17u;
        constexpr uint32_t kYW = 26u;
        constexpr uint32_t kYC = 5u;
        constexpr uint32_t kEvenCols = 13u;
        constexpr uint32_t kOddCols = 13u;
        constexpr uint32_t kCBytes = kYC * static_cast<uint32_t>(sizeof(DT_X));
        constexpr uint32_t kGroupBytes = ((kEvenCols * kCBytes + 31u) / 32u) * 32u;
        constexpr uint32_t kGroupElems = kGroupBytes / static_cast<uint32_t>(sizeof(DT_X));
        constexpr uint32_t kInputElems = kGroupElems * 2u;
        constexpr uint32_t kOutputElems = kYW * kYC;

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

        LocalTensor<DT_X> input_local(TPosition::VECCALC, 0, kInputElems);
        LocalTensor<DT_X> output_local(TPosition::VECCALC, 1024, kOutputElems);
        LocalTensor<uint32_t> offset_local(TPosition::VECCALC, 2048, kOutputElems);

        DataCopyPadExtParams<DT_X> pad_params;
        pad_params.isPad = false;
        pad_params.leftPadding = 0;
        pad_params.rightPadding = 0;
        pad_params.paddingValue = static_cast<DT_X>(0);

        DataCopyExtParams even_load_params;
        even_load_params.blockCount = 1;
        even_load_params.blockLen = kEvenCols * kCBytes;
        even_load_params.srcStride = 0;
        even_load_params.dstStride = 0;
        even_load_params.rsv = 0;

        DataCopyExtParams odd_load_params;
        odd_load_params.blockCount = 1;
        odd_load_params.blockLen = kOddCols * kCBytes;
        odd_load_params.srcStride = 0;
        odd_load_params.dstStride = 0;
        odd_load_params.rsv = 0;

        DataCopyExtParams row_store_params;
        row_store_params.blockCount = 1;
        row_store_params.blockLen = kYW * kCBytes;
        row_store_params.srcStride = 0;
        row_store_params.dstStride = 0;
        row_store_params.rsv = 0;

        for (uint32_t ow = 0; ow < kYW; ++ow) {
            uint32_t group_base = ((ow & 1u) == 0u) ? 0u : kGroupElems;
            uint32_t src_elem = group_base + (ow >> 1u) * kYC;
            uint32_t dst_elem = ow * kYC;
            for (uint32_t c = 0; c < kYC; ++c) {
                offset_local.SetValue(dst_elem + c, (src_elem + c) * static_cast<uint32_t>(sizeof(DT_X)));
            }
        }

        uint32_t block_idx = GetBlockIdx();
        uint32_t block_num = GetBlockNum();
        for (uint32_t oh = block_idx; oh < kYH; oh += block_num) {
            uint32_t padded_h = oh + 2u;
            uint32_t ih = padded_h >> 1u;
            uint32_t block_h = padded_h & 1u;

            uint32_t even_input_n = block_h * 2u + 1u;
            uint32_t even_src = ((even_input_n * kXH + ih) * kXW + 1u) * kXC;
            uint32_t odd_input_n = block_h * 2u;
            uint32_t odd_src = ((odd_input_n * kXH + ih) * kXW + 2u) * kXC;
            uint32_t dst = oh * kYW * kYC;

            DataCopyPad(input_local, x_gm[even_src], even_load_params, pad_params);
            DataCopyPad(input_local[kGroupElems], x_gm[odd_src], odd_load_params, pad_params);
            PipeBarrier<PIPE_ALL>();

            Gather(output_local, input_local, offset_local, 0, kOutputElems);
            PipeBarrier<PIPE_ALL>();

            DataCopyPad(y_gm[dst], output_local, row_store_params);
            PipeBarrier<PIPE_MTE3>();
        }
    }
};

template <typename DT_X>
class KernelPoint3BetaCase2 {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kXH = 14u;
        constexpr uint32_t kXW = 14u;
        constexpr uint32_t kXC = 64u;
        constexpr uint32_t kYN = 4u;
        constexpr uint32_t kYH = 28u;
        constexpr uint32_t kYW = 28u;
        constexpr uint32_t kBlockSize = 2u;
        constexpr uint32_t kTileRows = 7u;
        constexpr uint32_t kTilesPerBlockH = 2u;
        constexpr uint32_t kDBlocks = (kXC * static_cast<uint32_t>(sizeof(DT_X))) / 32u;
        constexpr uint32_t kSourceRowBlocks = (kXW * kXC * static_cast<uint32_t>(sizeof(DT_X))) / 32u;
        constexpr uint32_t kOutRowBlocks = (kYW * kXC * static_cast<uint32_t>(sizeof(DT_X))) / 32u;
        constexpr uint32_t kMaxInputElems = kTileRows * kXW * kXC;
        constexpr uint32_t kMaxOutputElems = kTileRows * kYW * kXC;
        constexpr uint32_t kInputBytes =
            ((kMaxInputElems * static_cast<uint32_t>(sizeof(DT_X)) + 511u) / 512u) * 512u;

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

        LocalTensor<DT_X> input0(TPosition::VECCALC, 0, kMaxInputElems);
        LocalTensor<DT_X> input1(TPosition::VECCALC, kInputBytes, kMaxInputElems);
        LocalTensor<DT_X> output(TPosition::VECCALC, 2u * kInputBytes, kMaxOutputElems);

        DataCopyParams load_params;
        load_params.blockCount = 1u;
        load_params.blockLen = 0u;
        load_params.srcStride = 0u;
        load_params.dstStride = 0u;

        DataCopyParams ub_params;
        ub_params.blockLen = static_cast<uint16_t>(kDBlocks);
        ub_params.srcStride = 0u;
        ub_params.dstStride = static_cast<uint16_t>(kDBlocks);

        DataCopyParams store_params;
        store_params.blockLen = static_cast<uint16_t>(kOutRowBlocks);
        store_params.srcStride = 0u;
        store_params.dstStride = static_cast<uint16_t>(kOutRowBlocks);

        uint32_t block_idx = GetBlockIdx();
        uint32_t block_num = GetBlockNum();
        for (uint32_t group = block_idx; group < kYN * kBlockSize * kTilesPerBlockH; group += block_num) {
            uint32_t tile = group & 1u;
            uint32_t tmp = group >> 1u;
            uint32_t block_h = tmp & 1u;
            uint32_t n = tmp >> 1u;
            uint32_t ih = tile * kTileRows;
            uint32_t rows = kXH - ih;
            if (rows > kTileRows) {
                rows = kTileRows;
            }

            uint32_t input_n0 = block_h * kBlockSize * kYN + n;
            uint32_t input_n1 = input_n0 + kYN;
            uint32_t src0 = ((input_n0 * kXH + ih) * kXW) * kXC;
            uint32_t src1 = ((input_n1 * kXH + ih) * kXW) * kXC;

            load_params.blockLen = static_cast<uint16_t>(rows * kSourceRowBlocks);
            DataCopy(input0, x_gm[src0], load_params);
            DataCopy(input1, x_gm[src1], load_params);
            PipeBarrier<PIPE_ALL>();

            ub_params.blockCount = static_cast<uint16_t>(rows * kXW);
            DataCopy(output, input0, ub_params);
            DataCopy(output[kXC], input1, ub_params);
            PipeBarrier<PIPE_MTE2>();

            store_params.blockCount = static_cast<uint16_t>(rows);
            uint32_t dst = (n * kYH + block_h + ih * kBlockSize) * kYW * kXC;
            DataCopy(y_gm[dst], output, store_params);
            PipeBarrier<PIPE_MTE3>();
        }
    }
};

template <typename DT_X>
class KernelPoint4Tile4Row {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kDepth = 32u;
        constexpr uint32_t kOutWidth = 12u;
        constexpr uint32_t kTileRows = 4u;
        constexpr uint32_t kRowElems = kOutWidth * kDepth;
        constexpr uint32_t kTileElems = kTileRows * kRowElems;
        constexpr uint32_t kInStrideB = 4u * 6u * kDepth;
        constexpr uint32_t kInStrideH = 6u * kDepth;
        constexpr uint32_t kOutStrideB = 8u * kRowElems;
        constexpr uint32_t kDepthBlocks = 4u;
        constexpr uint32_t kStreamPixels = 6u;
        constexpr uint32_t kTotalElems = 20u * 4u * 6u * kDepth;

        uint32_t task_idx = GetBlockIdx();
        if (task_idx >= 10u) {
            return;
        }

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), kTotalElems);
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), kTotalElems);

        TPipe pipe;
        TBuf<TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, kTileElems * static_cast<uint32_t>(sizeof(DT_X)));
        LocalTensor<DT_X> ubuf = buf.template Get<DT_X>();

        DataCopyParams read_params;
        read_params.blockCount = kStreamPixels;
        read_params.blockLen = kDepthBlocks;
        read_params.srcStride = 0u;
        read_params.dstStride = kDepthBlocks;

        DataCopyParams write_params;
        write_params.blockCount = 1u;
        write_params.blockLen = kTileRows * kOutWidth * kDepthBlocks;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        uint32_t bi = task_idx >> 1u;
        uint32_t h_start = (task_idx & 1u) << 2u;
        uint64_t out_base = static_cast<uint64_t>(bi) * kOutStrideB
                          + static_cast<uint64_t>(h_start) * kRowElems;

        for (uint32_t tr = 0u; tr < kTileRows; ++tr) {
            uint32_t h_out = h_start + tr;
            uint32_t h_in = h_out >> 1u;
            uint32_t b1 = h_out & 1u;
            uint32_t ub_row = tr * kRowElems;
            uint64_t in_base0 = (static_cast<uint64_t>(b1 << 1u) * 5u + bi) * kInStrideB
                              + static_cast<uint64_t>(h_in) * kInStrideH;
            uint64_t in_base1 = (static_cast<uint64_t>((b1 << 1u) + 1u) * 5u + bi) * kInStrideB
                              + static_cast<uint64_t>(h_in) * kInStrideH;
            DataCopy(ubuf[ub_row], x_gm[in_base0], read_params);
            DataCopy(ubuf[ub_row + kDepth], x_gm[in_base1], read_params);
        }

        constexpr event_t event_id = EVENT_ID0;
        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);
        DataCopy(y_gm[out_base], ubuf[0u], write_params);
    }
};

template <typename DT_X>
class KernelPoint4BetaCase3 {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kBlockBytes = 32u;
        constexpr uint32_t kInputBufferBytes = 16u * 1024u;
        constexpr uint32_t kOutputBufferBytes = 32u * 1024u;
        constexpr uint32_t kInput1OffsetBytes = kInputBufferBytes;
        constexpr uint32_t kOutputOffsetBytes = kInputBufferBytes * 2u;
        constexpr uint32_t kXH = 4u;
        constexpr uint32_t kXW = 6u;
        constexpr uint32_t kXC = 32u;
        constexpr uint32_t kYN = 5u;
        constexpr uint32_t kYH = 8u;
        constexpr uint32_t kYW = 12u;
        constexpr uint32_t kCBytes = kXC * static_cast<uint32_t>(sizeof(DT_X));
        constexpr uint32_t kCBlocks = kCBytes / kBlockBytes;
        constexpr uint32_t kSourceRowBlocks = (kXW * kCBytes) / kBlockBytes;
        constexpr uint32_t kOutRowBlocks = (kYW * kCBytes) / kBlockBytes;
        constexpr uint32_t kSourceRowBytes = kXW * kCBytes;
        constexpr uint32_t kOutRowBytes = kYW * kCBytes;
        constexpr uint32_t kMaxRowsBySrc = kInputBufferBytes / kSourceRowBytes;
        constexpr uint32_t kMaxRowsByOut = kOutputBufferBytes / kOutRowBytes;
        constexpr uint32_t kTileRows = kMaxRowsBySrc < kMaxRowsByOut ? kMaxRowsBySrc : kMaxRowsByOut;
        constexpr uint32_t kTilesPerBlockH = (kXH + kTileRows - 1u) / kTileRows;
        constexpr uint32_t kInterleaveGroups = kYN * 2u * kTilesPerBlockH;

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

        LocalTensor<DT_X> input0(TPosition::VECCALC, 0, kInputBufferBytes / sizeof(DT_X));
        LocalTensor<DT_X> input1(TPosition::VECCALC, kInput1OffsetBytes, kInputBufferBytes / sizeof(DT_X));
        LocalTensor<DT_X> output(TPosition::VECCALC, kOutputOffsetBytes, kOutputBufferBytes / sizeof(DT_X));

        DataCopyParams load_params;
        load_params.blockCount = 1u;
        load_params.blockLen = 0u;
        load_params.srcStride = 0u;
        load_params.dstStride = 0u;

        DataCopyParams ub_params;
        ub_params.blockLen = static_cast<uint16_t>(kCBlocks);
        ub_params.srcStride = 0u;
        ub_params.dstStride = static_cast<uint16_t>(kCBlocks);

        DataCopyParams store_params;
        store_params.blockLen = static_cast<uint16_t>(kOutRowBlocks);
        store_params.srcStride = 0u;
        store_params.dstStride = static_cast<uint16_t>(kOutRowBlocks);

        uint32_t block_idx = GetBlockIdx();
        uint32_t block_num = GetBlockNum();
        for (uint32_t group = block_idx; group < kInterleaveGroups; group += block_num) {
            uint32_t block_h = group & 1u;
            uint32_t n = group >> 1u;
            uint32_t rows = kXH;

            uint32_t input_n0 = (block_h * 2u) * kYN + n;
            uint32_t input_n1 = input_n0 + kYN;
            uint32_t src0 = (input_n0 * kXH * kXW) * kXC;
            uint32_t src1 = (input_n1 * kXH * kXW) * kXC;

            load_params.blockLen = static_cast<uint16_t>(rows * kSourceRowBlocks);
            DataCopy(input0, x_gm[src0], load_params);
            DataCopy(input1, x_gm[src1], load_params);
            PipeBarrier<PIPE_ALL>();

            ub_params.blockCount = static_cast<uint16_t>(rows * kXW);
            DataCopy(output, input0, ub_params);
            DataCopy(output[kXC], input1, ub_params);
            PipeBarrier<PIPE_ALL>();

            store_params.blockCount = static_cast<uint16_t>(rows);
            uint32_t dst = ((n * kYH + block_h) * kYW) * kXC;
            DataCopy(y_gm[dst], output, store_params);
            PipeBarrier<PIPE_MTE3>();
        }
    }
};

template <typename DT_X>
class KernelPoint5BetaCase4 {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kDepth = 65u;
        constexpr uint32_t kDepthBytes = 130u;
        constexpr uint32_t kXH = 128u;
        constexpr uint32_t kXW = 128u;
        constexpr uint32_t kYW = 254u;
        constexpr uint32_t kCropTop = 1u;
        constexpr uint32_t kCropLeft = 1u;
        constexpr uint32_t kSlotBytes = 50u * 1024u;
        constexpr uint32_t kSlotElems = kSlotBytes / static_cast<uint32_t>(sizeof(DT_X));
        constexpr uint32_t kOffsetBytes = 34u * 1024u;
        constexpr uint32_t kMaxTaskCols = 127u;
        constexpr uint32_t kMaxLaneCols = 64u;
        constexpr uint32_t kGroupBytes = kMaxLaneCols * kDepthBytes;
        constexpr uint32_t kGroupElems = kGroupBytes / static_cast<uint32_t>(sizeof(DT_X));
        constexpr uint32_t kInputElems = 2u * kGroupElems;
        constexpr uint32_t kOffsetElemsAligned = 8256u;
        constexpr uint32_t kPairElems = 2u * kDepth;
        constexpr uint32_t kPeriodPairs = 4u;
        constexpr uint32_t kPeriodElems = kPeriodPairs * kPairElems;
        constexpr uint32_t kPeriodIncBytes = kPeriodPairs * kDepthBytes;
        constexpr uint32_t kTotalTasks = 254u * 2u;

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

        LocalTensor<DT_X> slot0(TPosition::VECCALC, 0, kSlotElems);
        LocalTensor<DT_X> slot1(TPosition::VECCALC, kSlotBytes, kSlotElems);
        LocalTensor<int32_t> offsets_i32(TPosition::VECCALC, kSlotBytes * 2u,
                                         kOffsetBytes / sizeof(int32_t));
        LocalTensor<uint32_t> offsets(TPosition::VECCALC, kSlotBytes * 2u,
                                      kOffsetBytes / sizeof(uint32_t));

        for (uint32_t pair = 0u; pair < kPeriodPairs; ++pair) {
            uint32_t pair_base = pair * kPairElems;
            uint32_t pair_byte_base = pair * kDepthBytes;
            for (uint32_t c = 0u; c < kDepth; ++c) {
                uint32_t channel_byte = c * static_cast<uint32_t>(sizeof(DT_X));
                offsets_i32.SetValue(pair_base + c, static_cast<int32_t>(pair_byte_base + channel_byte));
                offsets_i32.SetValue(pair_base + kDepth + c,
                                     static_cast<int32_t>(kGroupBytes + pair_byte_base + channel_byte));
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID6);
        WaitFlag<HardEvent::S_V>(EVENT_ID6);
        for (uint32_t built = kPeriodElems; built < kOffsetElemsAligned; built += kPeriodElems) {
            uint32_t count = kOffsetElemsAligned - built;
            if (count > kPeriodElems) {
                count = kPeriodElems;
            }
            Adds(offsets_i32[built], offsets_i32[built - kPeriodElems],
                 static_cast<int32_t>(kPeriodIncBytes), count);
        }
        PipeBarrier<PIPE_V>();

        DataCopyExtParams lane_a_params;
        lane_a_params.blockCount = 1u;
        lane_a_params.srcStride = 0u;
        lane_a_params.dstStride = 0u;
        lane_a_params.rsv = 0u;

        DataCopyExtParams lane_b_params;
        lane_b_params.blockCount = 1u;
        lane_b_params.srcStride = 0u;
        lane_b_params.dstStride = 0u;
        lane_b_params.rsv = 0u;

        DataCopyExtParams store_params;
        store_params.blockCount = 1u;
        store_params.srcStride = 0u;
        store_params.dstStride = 0u;
        store_params.rsv = 0u;

        DataCopyPadExtParams<DT_X> pad_params;
        pad_params.isPad = false;
        pad_params.leftPadding = 0u;
        pad_params.rightPadding = 0u;
        pad_params.paddingValue = static_cast<DT_X>(0);

        uint32_t block_idx = GetBlockIdx();
        uint32_t block_num = GetBlockNum();
        if (block_num == 0u || block_idx >= kTotalTasks) {
            return;
        }

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

        uint32_t task_id = block_idx;
        uint32_t buf_idx = 0u;
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        {
            LocalTensor<DT_X> buf = slot0;
            uint32_t oh = task_id >> 1u;
            uint32_t half = task_id & 1u;
            uint32_t ow_start = half * kMaxTaskCols;
            uint32_t task_cols = kYW - ow_start;
            if (task_cols > kMaxTaskCols) {
                task_cols = kMaxTaskCols;
            }
            uint32_t full_h = oh + kCropTop;
            uint32_t ih = full_h >> 1u;
            uint32_t block_h = full_h & 1u;
            uint32_t full_w_start = kCropLeft + ow_start;
            uint32_t phase_a = full_w_start & 1u;
            uint32_t phase_b = phase_a ^ 1u;
            uint32_t iw_a = full_w_start >> 1u;
            uint32_t iw_b = (full_w_start + 1u) >> 1u;
            uint32_t lane_a_cols = (task_cols + 1u) >> 1u;
            uint32_t lane_b_cols = task_cols >> 1u;
            uint32_t input_na = block_h * 2u + phase_a;
            uint32_t input_nb = block_h * 2u + phase_b;
            uint64_t src_a = ((static_cast<uint64_t>(input_na) * kXH + ih) * kXW + iw_a) * kDepth;
            uint64_t src_b = ((static_cast<uint64_t>(input_nb) * kXH + ih) * kXW + iw_b) * kDepth;
            lane_a_params.blockLen = lane_a_cols * kDepthBytes;
            lane_b_params.blockLen = lane_b_cols * kDepthBytes;
            DataCopyPad(buf, x_gm[src_a], lane_a_params, pad_params);
            if (lane_b_cols > 0u) {
                DataCopyPad(buf[kGroupElems], x_gm[src_b], lane_b_params, pad_params);
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        }

        while (task_id < kTotalTasks) {
            uint32_t cur_buf_idx = buf_idx;
            LocalTensor<DT_X> cur_buf = cur_buf_idx == 0u ? slot0 : slot1;
            event_t cur_ready_event = cur_buf_idx == 0u ? EVENT_ID2 : EVENT_ID3;
            event_t cur_pack_event = cur_buf_idx == 0u ? EVENT_ID4 : EVENT_ID5;
            WaitFlag<HardEvent::MTE2_V>(cur_ready_event);

            uint32_t next_task_id = task_id + block_num;
            uint32_t next_buf_idx = cur_buf_idx ^ 1u;
            if (next_task_id < kTotalTasks) {
                LocalTensor<DT_X> next_buf = next_buf_idx == 0u ? slot0 : slot1;
                event_t next_free_event = next_buf_idx == 0u ? EVENT_ID0 : EVENT_ID1;
                event_t next_ready_event = next_buf_idx == 0u ? EVENT_ID2 : EVENT_ID3;
                WaitFlag<HardEvent::MTE3_MTE2>(next_free_event);

                uint32_t oh = next_task_id >> 1u;
                uint32_t half = next_task_id & 1u;
                uint32_t ow_start = half * kMaxTaskCols;
                uint32_t task_cols = kYW - ow_start;
                if (task_cols > kMaxTaskCols) {
                    task_cols = kMaxTaskCols;
                }
                uint32_t full_h = oh + kCropTop;
                uint32_t ih = full_h >> 1u;
                uint32_t block_h = full_h & 1u;
                uint32_t full_w_start = kCropLeft + ow_start;
                uint32_t phase_a = full_w_start & 1u;
                uint32_t phase_b = phase_a ^ 1u;
                uint32_t iw_a = full_w_start >> 1u;
                uint32_t iw_b = (full_w_start + 1u) >> 1u;
                uint32_t lane_a_cols = (task_cols + 1u) >> 1u;
                uint32_t lane_b_cols = task_cols >> 1u;
                uint32_t input_na = block_h * 2u + phase_a;
                uint32_t input_nb = block_h * 2u + phase_b;
                uint64_t src_a = ((static_cast<uint64_t>(input_na) * kXH + ih) * kXW + iw_a) * kDepth;
                uint64_t src_b = ((static_cast<uint64_t>(input_nb) * kXH + ih) * kXW + iw_b) * kDepth;
                lane_a_params.blockLen = lane_a_cols * kDepthBytes;
                lane_b_params.blockLen = lane_b_cols * kDepthBytes;
                DataCopyPad(next_buf, x_gm[src_a], lane_a_params, pad_params);
                if (lane_b_cols > 0u) {
                    DataCopyPad(next_buf[kGroupElems], x_gm[src_b], lane_b_params, pad_params);
                }
                SetFlag<HardEvent::MTE2_V>(next_ready_event);
            }

            uint32_t oh = task_id >> 1u;
            uint32_t half = task_id & 1u;
            uint32_t ow_start = half * kMaxTaskCols;
            uint32_t task_cols = kYW - ow_start;
            if (task_cols > kMaxTaskCols) {
                task_cols = kMaxTaskCols;
            }
            uint32_t cur_output_elems = task_cols * kDepth;
            Gather(cur_buf[kInputElems], cur_buf, offsets, 0u, cur_output_elems);
            SetFlag<HardEvent::V_MTE3>(cur_pack_event);
            WaitFlag<HardEvent::V_MTE3>(cur_pack_event);

            uint64_t dst = (static_cast<uint64_t>(oh) * kYW + ow_start) * kDepth;
            store_params.blockLen = cur_output_elems * static_cast<uint32_t>(sizeof(DT_X));
            DataCopyPad(y_gm[dst], cur_buf[kInputElems], store_params);
            event_t cur_free_event = cur_buf_idx == 0u ? EVENT_ID0 : EVENT_ID1;
            SetFlag<HardEvent::MTE3_MTE2>(cur_free_event);

            task_id = next_task_id;
            buf_idx = next_buf_idx;
        }

        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
};

template <typename DT_X>
class KernelPoint4Tile4RowUnroll {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kDepth = 32u;
        constexpr uint32_t kOutWidth = 12u;
        constexpr uint32_t kTileRows = 4u;
        constexpr uint32_t kRowElems = kOutWidth * kDepth;
        constexpr uint32_t kTileElems = kTileRows * kRowElems;
        constexpr uint32_t kInStrideB = 4u * 6u * kDepth;
        constexpr uint32_t kInStrideH = 6u * kDepth;
        constexpr uint32_t kOutStrideB = 8u * kRowElems;
        constexpr uint32_t kDepthBlocks = 4u;
        constexpr uint32_t kStreamPixels = 6u;
        constexpr uint32_t kTotalElems = 20u * 4u * 6u * kDepth;

        uint32_t task_idx = GetBlockIdx();
        if (task_idx >= 10u) {
            return;
        }

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), kTotalElems);
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), kTotalElems);

        TPipe pipe;
        TBuf<TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, kTileElems * static_cast<uint32_t>(sizeof(DT_X)));
        LocalTensor<DT_X> ubuf = buf.template Get<DT_X>();

        DataCopyParams read_params;
        read_params.blockCount = kStreamPixels;
        read_params.blockLen = kDepthBlocks;
        read_params.srcStride = 0u;
        read_params.dstStride = kDepthBlocks;

        DataCopyParams write_params;
        write_params.blockCount = 1u;
        write_params.blockLen = kTileRows * kOutWidth * kDepthBlocks;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        uint32_t bi = task_idx >> 1u;
        uint32_t tile_hi = task_idx & 1u;
        uint64_t out_base = static_cast<uint64_t>(bi) * kOutStrideB
                          + static_cast<uint64_t>(tile_hi) * (4u * kRowElems);
        uint64_t b0_base = static_cast<uint64_t>(bi) * kInStrideB;
        uint64_t b1_base = static_cast<uint64_t>(5u + bi) * kInStrideB;
        uint64_t b2_base = static_cast<uint64_t>(10u + bi) * kInStrideB;
        uint64_t b3_base = static_cast<uint64_t>(15u + bi) * kInStrideB;

        if (tile_hi == 0u) {
            DataCopy(ubuf[0u * kRowElems], x_gm[b0_base], read_params);
            DataCopy(ubuf[0u * kRowElems + kDepth], x_gm[b1_base], read_params);
            DataCopy(ubuf[1u * kRowElems], x_gm[b2_base], read_params);
            DataCopy(ubuf[1u * kRowElems + kDepth], x_gm[b3_base], read_params);
            DataCopy(ubuf[2u * kRowElems], x_gm[b0_base + kInStrideH], read_params);
            DataCopy(ubuf[2u * kRowElems + kDepth], x_gm[b1_base + kInStrideH], read_params);
            DataCopy(ubuf[3u * kRowElems], x_gm[b2_base + kInStrideH], read_params);
            DataCopy(ubuf[3u * kRowElems + kDepth], x_gm[b3_base + kInStrideH], read_params);
        } else {
            constexpr uint32_t kH2Offset = 2u * kInStrideH;
            DataCopy(ubuf[0u * kRowElems], x_gm[b0_base + kH2Offset], read_params);
            DataCopy(ubuf[0u * kRowElems + kDepth], x_gm[b1_base + kH2Offset], read_params);
            DataCopy(ubuf[1u * kRowElems], x_gm[b2_base + kH2Offset], read_params);
            DataCopy(ubuf[1u * kRowElems + kDepth], x_gm[b3_base + kH2Offset], read_params);
            DataCopy(ubuf[2u * kRowElems], x_gm[b0_base + kH2Offset + kInStrideH], read_params);
            DataCopy(ubuf[2u * kRowElems + kDepth], x_gm[b1_base + kH2Offset + kInStrideH], read_params);
            DataCopy(ubuf[3u * kRowElems], x_gm[b2_base + kH2Offset + kInStrideH], read_params);
            DataCopy(ubuf[3u * kRowElems + kDepth], x_gm[b3_base + kH2Offset + kInStrideH], read_params);
        }

        constexpr event_t event_id = EVENT_ID0;
        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);
        DataCopy(y_gm[out_base], ubuf[0u], write_params);
    }
};

template <typename DT_X>
class KernelPoint4Row40 {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kDepth = 32u;
        constexpr uint32_t kOutWidth = 12u;
        constexpr uint32_t kRowElems = kOutWidth * kDepth;
        constexpr uint32_t kInStrideB = 4u * 6u * kDepth;
        constexpr uint32_t kInStrideH = 6u * kDepth;
        constexpr uint32_t kOutStrideB = 8u * kRowElems;
        constexpr uint32_t kDepthBlocks = 4u;
        constexpr uint32_t kStreamPixels = 6u;
        constexpr uint32_t kTotalElems = 20u * 4u * 6u * kDepth;

        uint32_t row_task = GetBlockIdx();
        if (row_task >= 40u) {
            return;
        }

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), kTotalElems);
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), kTotalElems);

        TPipe pipe;
        TBuf<TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, kRowElems * static_cast<uint32_t>(sizeof(DT_X)));
        LocalTensor<DT_X> ubuf = buf.template Get<DT_X>();

        DataCopyParams read_params;
        read_params.blockCount = kStreamPixels;
        read_params.blockLen = kDepthBlocks;
        read_params.srcStride = 0u;
        read_params.dstStride = kDepthBlocks;

        DataCopyParams write_params;
        write_params.blockCount = 1u;
        write_params.blockLen = kOutWidth * kDepthBlocks;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        uint32_t bi = row_task >> 3u;
        uint32_t h_out = row_task & 7u;
        uint32_t h_in = h_out >> 1u;
        uint32_t b1 = h_out & 1u;
        uint64_t in_base0 = (static_cast<uint64_t>(b1 << 1u) * 5u + bi) * kInStrideB
                          + static_cast<uint64_t>(h_in) * kInStrideH;
        uint64_t in_base1 = (static_cast<uint64_t>((b1 << 1u) + 1u) * 5u + bi) * kInStrideB
                          + static_cast<uint64_t>(h_in) * kInStrideH;
        uint64_t out_base = static_cast<uint64_t>(bi) * kOutStrideB
                          + static_cast<uint64_t>(h_out) * kRowElems;

        DataCopy(ubuf[0u], x_gm[in_base0], read_params);
        DataCopy(ubuf[kDepth], x_gm[in_base1], read_params);
        constexpr event_t event_id = EVENT_ID0;
        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);
        DataCopy(y_gm[out_base], ubuf[0u], write_params);
    }
};

template <typename DT_X>
class KernelPoint4ReadContig {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kDepth = 32u;
        constexpr uint32_t kInWidth = 6u;
        constexpr uint32_t kOutWidth = 12u;
        constexpr uint32_t kTileRows = 4u;
        constexpr uint32_t kRowElems = kOutWidth * kDepth;
        constexpr uint32_t kReadRowElems = kInWidth * kDepth;
        constexpr uint32_t kTileElems = 2u * kTileRows * kReadRowElems;
        constexpr uint32_t kInStrideB = 4u * kInWidth * kDepth;
        constexpr uint32_t kInStrideH = kInWidth * kDepth;
        constexpr uint32_t kOutStrideB = 8u * kRowElems;
        constexpr uint32_t kReadBlocks = kReadRowElems / 8u;
        constexpr uint32_t kDepthBlocks = 4u;
        constexpr uint32_t kTotalElems = 20u * 4u * kInWidth * kDepth;

        uint32_t task_idx = GetBlockIdx();
        if (task_idx >= 10u) {
            return;
        }

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), kTotalElems);
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), kTotalElems);

        TPipe pipe;
        TBuf<TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, kTileElems * static_cast<uint32_t>(sizeof(DT_X)));
        LocalTensor<DT_X> ubuf = buf.template Get<DT_X>();

        DataCopyParams read_params;
        read_params.blockCount = 1u;
        read_params.blockLen = kReadBlocks;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyParams write_params;
        write_params.blockCount = kInWidth;
        write_params.blockLen = kDepthBlocks;
        write_params.srcStride = 0u;
        write_params.dstStride = kDepthBlocks;

        uint32_t bi = task_idx >> 1u;
        uint32_t h_start = (task_idx & 1u) << 2u;
        uint64_t out_base = static_cast<uint64_t>(bi) * kOutStrideB
                          + static_cast<uint64_t>(h_start) * kRowElems;

        for (uint32_t tr = 0u; tr < kTileRows; ++tr) {
            uint32_t h_out = h_start + tr;
            uint32_t h_in = h_out >> 1u;
            uint32_t b1 = h_out & 1u;
            uint32_t ub_row = tr * 2u * kReadRowElems;
            uint64_t in_base0 = (static_cast<uint64_t>(b1 << 1u) * 5u + bi) * kInStrideB
                              + static_cast<uint64_t>(h_in) * kInStrideH;
            uint64_t in_base1 = (static_cast<uint64_t>((b1 << 1u) + 1u) * 5u + bi) * kInStrideB
                              + static_cast<uint64_t>(h_in) * kInStrideH;
            DataCopy(ubuf[ub_row], x_gm[in_base0], read_params);
            DataCopy(ubuf[ub_row + kReadRowElems], x_gm[in_base1], read_params);
        }

        constexpr event_t event_id = EVENT_ID0;
        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);

        for (uint32_t tr = 0u; tr < kTileRows; ++tr) {
            uint32_t ub_row = tr * 2u * kReadRowElems;
            uint64_t out_row = out_base + static_cast<uint64_t>(tr) * kRowElems;
            DataCopy(y_gm[out_row], ubuf[ub_row], write_params);
            DataCopy(y_gm[out_row + kDepth], ubuf[ub_row + kReadRowElems], write_params);
        }
    }
};

template <typename DT_X>
class KernelPoint4Batch5Core {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kDepth = 32u;
        constexpr uint32_t kInputWidth = 6u;
        constexpr uint32_t kOutHeight = 8u;
        constexpr uint32_t kOutWidth = 12u;
        constexpr uint32_t kRowElems = kOutWidth * kDepth;
        constexpr uint32_t kBatchElems = kOutHeight * kRowElems;
        constexpr uint32_t kInStrideB = 4u * kInputWidth * kDepth;
        constexpr uint32_t kInStrideH = kInputWidth * kDepth;
        constexpr uint32_t kDepthBlocks = 4u;
        constexpr uint32_t kStreamPixels = 6u;
        constexpr uint32_t kTotalElems = 20u * 4u * kInputWidth * kDepth;

        uint32_t bi = GetBlockIdx();
        if (bi >= 5u) {
            return;
        }

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), kTotalElems);
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), kTotalElems);

        TPipe pipe;
        TBuf<TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, kBatchElems * static_cast<uint32_t>(sizeof(DT_X)));
        LocalTensor<DT_X> ubuf = buf.template Get<DT_X>();

        DataCopyParams read_params;
        read_params.blockCount = kStreamPixels;
        read_params.blockLen = kDepthBlocks;
        read_params.srcStride = 0u;
        read_params.dstStride = kDepthBlocks;

        DataCopyParams write_params;
        write_params.blockCount = 1u;
        write_params.blockLen = kOutHeight * kOutWidth * kDepthBlocks;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        for (uint32_t h_out = 0u; h_out < kOutHeight; ++h_out) {
            uint32_t h_in = h_out >> 1u;
            uint32_t b1 = h_out & 1u;
            uint32_t ub_row = h_out * kRowElems;
            uint64_t in_base0 = (static_cast<uint64_t>(b1 << 1u) * 5u + bi) * kInStrideB
                              + static_cast<uint64_t>(h_in) * kInStrideH;
            uint64_t in_base1 = (static_cast<uint64_t>((b1 << 1u) + 1u) * 5u + bi) * kInStrideB
                              + static_cast<uint64_t>(h_in) * kInStrideH;
            DataCopy(ubuf[ub_row], x_gm[in_base0], read_params);
            DataCopy(ubuf[ub_row + kDepth], x_gm[in_base1], read_params);
        }

        constexpr event_t event_id = EVENT_ID0;
        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);
        DataCopy(y_gm[static_cast<uint64_t>(bi) * kBatchElems], ubuf[0u], write_params);
    }
};

template <typename DT_X>
class KernelPoint4Tile2Row {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kDepth = 32u;
        constexpr uint32_t kOutWidth = 12u;
        constexpr uint32_t kTileRows = 2u;
        constexpr uint32_t kRowElems = kOutWidth * kDepth;
        constexpr uint32_t kTileElems = kTileRows * kRowElems;
        constexpr uint32_t kInStrideB = 4u * 6u * kDepth;
        constexpr uint32_t kInStrideH = 6u * kDepth;
        constexpr uint32_t kOutStrideB = 8u * kRowElems;
        constexpr uint32_t kDepthBlocks = 4u;
        constexpr uint32_t kStreamPixels = 6u;
        constexpr uint32_t kTotalElems = 20u * 4u * 6u * kDepth;

        uint32_t task_idx = GetBlockIdx();
        if (task_idx >= 20u) {
            return;
        }

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), kTotalElems);
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), kTotalElems);

        TPipe pipe;
        TBuf<TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, kTileElems * static_cast<uint32_t>(sizeof(DT_X)));
        LocalTensor<DT_X> ubuf = buf.template Get<DT_X>();

        DataCopyParams read_params;
        read_params.blockCount = kStreamPixels;
        read_params.blockLen = kDepthBlocks;
        read_params.srcStride = 0u;
        read_params.dstStride = kDepthBlocks;

        DataCopyParams write_params;
        write_params.blockCount = 1u;
        write_params.blockLen = kTileRows * kOutWidth * kDepthBlocks;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        uint32_t bi = task_idx / 4u;
        uint32_t tile_in_batch = task_idx - bi * 4u;
        uint32_t h_start = tile_in_batch << 1u;
        uint64_t out_base = static_cast<uint64_t>(bi) * kOutStrideB
                          + static_cast<uint64_t>(h_start) * kRowElems;

        for (uint32_t tr = 0u; tr < kTileRows; ++tr) {
            uint32_t h_out = h_start + tr;
            uint32_t h_in = h_out >> 1u;
            uint32_t b1 = h_out & 1u;
            uint32_t ub_row = tr * kRowElems;
            uint64_t in_base0 = (static_cast<uint64_t>(b1 << 1u) * 5u + bi) * kInStrideB
                              + static_cast<uint64_t>(h_in) * kInStrideH;
            uint64_t in_base1 = (static_cast<uint64_t>((b1 << 1u) + 1u) * 5u + bi) * kInStrideB
                              + static_cast<uint64_t>(h_in) * kInStrideH;
            DataCopy(ubuf[ub_row], x_gm[in_base0], read_params);
            DataCopy(ubuf[ub_row + kDepth], x_gm[in_base1], read_params);
        }

        constexpr event_t event_id = EVENT_ID0;
        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);
        DataCopy(y_gm[out_base], ubuf[0u], write_params);
    }
};

template <typename DT_X>
class KernelPoint6Stream8Core {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kDepth = 4096u;
        constexpr uint32_t kTotalElems = 65536u;
        constexpr uint32_t kStreamElems = 2u * kDepth;
        constexpr uint16_t kDepthBlocks = 256u;
        constexpr uint16_t kDstStrideBlocks = 256u;

        uint32_t stream_idx = GetBlockIdx();
        if (stream_idx >= 8u) {
            return;
        }

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), kTotalElems);
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), kTotalElems);

        TPipe pipe;
        TBuf<TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, kStreamElems * static_cast<uint32_t>(sizeof(DT_X)));
        LocalTensor<DT_X> ubuf = buf.template Get<DT_X>();

        DataCopyParams read_params;
        read_params.blockCount = 2u;
        read_params.blockLen = kDepthBlocks;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyParams write_params;
        write_params.blockCount = 2u;
        write_params.blockLen = kDepthBlocks;
        write_params.srcStride = 0u;
        write_params.dstStride = kDstStrideBlocks;

        uint32_t row_idx = stream_idx >> 1u;
        uint32_t b2 = stream_idx & 1u;
        uint32_t h_in = row_idx >> 1u;
        uint32_t b1 = row_idx & 1u;
        uint32_t input_batch = (b1 << 1u) + b2;

        uint64_t in_base = (static_cast<uint64_t>(input_batch) * 4u
                         + static_cast<uint64_t>(h_in) * 2u) * kDepth;
        uint64_t out_base = (static_cast<uint64_t>(row_idx) * 4u + b2) * kDepth;

        constexpr event_t event_id = EVENT_ID0;
        DataCopy(ubuf[0u], x_gm[in_base], read_params);
        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);
        DataCopy(y_gm[out_base], ubuf[0u], write_params);
    }
};

template <typename DT_X>
class KernelPoint6BetaCase5 {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kBlockBytes = 32u;
        constexpr uint32_t kBufferBytes = 64u * 1024u;
        constexpr uint32_t kXH = 2u;
        constexpr uint32_t kXW = 2u;
        constexpr uint32_t kXC = 4096u;
        constexpr uint32_t kYH = 4u;
        constexpr uint32_t kYW = 4u;
        constexpr uint32_t kCBytes = kXC * static_cast<uint32_t>(sizeof(DT_X));
        constexpr uint32_t kCBlocks = kCBytes / kBlockBytes;
        constexpr uint32_t kRowBlocks = (kYW * kCBytes) / kBlockBytes;

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

        LocalTensor<DT_X> row_local(TPosition::VECCALC, 0, kBufferBytes / sizeof(DT_X));

        DataCopyParams chunk_params;
        chunk_params.blockCount = 1u;
        chunk_params.blockLen = static_cast<uint16_t>(kCBlocks);
        chunk_params.srcStride = 0u;
        chunk_params.dstStride = 0u;

        DataCopyParams row_params;
        row_params.blockCount = 1u;
        row_params.blockLen = static_cast<uint16_t>(kRowBlocks);
        row_params.srcStride = 0u;
        row_params.dstStride = 0u;

        uint32_t block_idx = GetBlockIdx();
        uint32_t block_num = GetBlockNum();
        for (uint32_t oh = block_idx; oh < kYH; oh += block_num) {
            uint32_t ih = oh >> 1u;
            uint32_t block_h = oh & 1u;
            for (uint32_t iw = 0u; iw < kXW; ++iw) {
                uint32_t input_n0 = block_h * 2u;
                uint32_t input_n1 = input_n0 + 1u;
                uint32_t src0 = ((input_n0 * kXH + ih) * kXW + iw) * kXC;
                uint32_t src1 = ((input_n1 * kXH + ih) * kXW + iw) * kXC;
                DataCopy(row_local[(2u * iw) * kXC], x_gm[src0], chunk_params);
                DataCopy(row_local[(2u * iw + 1u) * kXC], x_gm[src1], chunk_params);
            }
            PipeBarrier<PIPE_ALL>();

            uint32_t dst = (oh * kYW) * kXC;
            DataCopy(y_gm[dst], row_local, row_params);
            PipeBarrier<PIPE_MTE3>();
        }
    }
};

template <typename DT_X>
class KernelPoint6Pixel16Core {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kDepth = 4096u;
        constexpr uint32_t kTotalElems = 65536u;
        constexpr uint16_t kDepthBlocks = 256u;

        uint32_t out_pixel = GetBlockIdx();
        if (out_pixel >= 16u) {
            return;
        }

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), kTotalElems);
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), kTotalElems);

        TPipe pipe;
        TBuf<TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, kDepth * static_cast<uint32_t>(sizeof(DT_X)));
        LocalTensor<DT_X> ubuf = buf.template Get<DT_X>();

        DataCopyParams copy_params;
        copy_params.blockCount = 1u;
        copy_params.blockLen = kDepthBlocks;
        copy_params.srcStride = 0u;
        copy_params.dstStride = 0u;

        uint32_t h_out = out_pixel >> 2u;
        uint32_t w_out = out_pixel & 3u;
        uint32_t h_in = h_out >> 1u;
        uint32_t w_in = w_out >> 1u;
        uint32_t input_batch = ((h_out & 1u) << 1u) + (w_out & 1u);

        uint64_t in_base = (static_cast<uint64_t>(input_batch) * 4u
                         + static_cast<uint64_t>(h_in) * 2u
                         + static_cast<uint64_t>(w_in)) * kDepth;
        uint64_t out_base = static_cast<uint64_t>(out_pixel) * kDepth;

        constexpr event_t event_id = EVENT_ID0;
        DataCopy(ubuf[0u], x_gm[in_base], copy_params);
        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);
        DataCopy(y_gm[out_base], ubuf[0u], copy_params);
    }
};

template <typename DT_X>
class KernelPoint6Row4Core {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kDepth = 4096u;
        constexpr uint32_t kTotalElems = 65536u;
        constexpr uint16_t kDepthBlocks = 256u;

        uint32_t row_idx = GetBlockIdx();
        if (row_idx >= 4u) {
            return;
        }

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), kTotalElems);
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), kTotalElems);

        TPipe pipe;
        TBuf<TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, 4u * kDepth * static_cast<uint32_t>(sizeof(DT_X)));
        LocalTensor<DT_X> ubuf = buf.template Get<DT_X>();

        DataCopyParams read_params;
        read_params.blockCount = 1u;
        read_params.blockLen = kDepthBlocks;
        read_params.srcStride = 0u;
        read_params.dstStride = 0u;

        DataCopyParams write_params;
        write_params.blockCount = 4u;
        write_params.blockLen = kDepthBlocks;
        write_params.srcStride = 0u;
        write_params.dstStride = 0u;

        uint32_t h_in = row_idx >> 1u;
        uint32_t b1 = row_idx & 1u;
        uint64_t in_row_base = static_cast<uint64_t>(h_in) * 2u * kDepth;
        uint64_t out_base = static_cast<uint64_t>(row_idx) * 4u * kDepth;
        uint64_t in_base0 = (static_cast<uint64_t>(b1 << 1u) * 4u) * kDepth + in_row_base;
        uint64_t in_base1 = (static_cast<uint64_t>((b1 << 1u) + 1u) * 4u) * kDepth + in_row_base;

        constexpr event_t event_id = EVENT_ID0;
        DataCopy(ubuf[0u], x_gm[in_base0], read_params);
        DataCopy(ubuf[kDepth], x_gm[in_base1], read_params);
        DataCopy(ubuf[2u * kDepth], x_gm[in_base0 + kDepth], read_params);
        DataCopy(ubuf[3u * kDepth], x_gm[in_base1 + kDepth], read_params);
        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);
        DataCopy(y_gm[out_base], ubuf[0u], write_params);
    }
};

template <typename DT_X, uint32_t CORES>
class KernelPoint7Linear8Core {
public:
    template <bool WAIT_WRITE>
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kTotalElems = 65536u;
        constexpr uint32_t kChunkElems = kTotalElems / CORES;
        constexpr uint16_t kChunkBlocks =
            static_cast<uint16_t>((kChunkElems * static_cast<uint32_t>(sizeof(DT_X))) / 32u);

        uint32_t chunk_idx = GetBlockIdx();
        if (chunk_idx >= CORES) {
            return;
        }

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), kTotalElems);
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), kTotalElems);

        TPipe pipe;
        TBuf<TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, kChunkElems * static_cast<uint32_t>(sizeof(DT_X)));
        LocalTensor<DT_X> ubuf = buf.template Get<DT_X>();

        DataCopyParams copy_params;
        copy_params.blockCount = 1u;
        copy_params.blockLen = kChunkBlocks;
        copy_params.srcStride = 0u;
        copy_params.dstStride = 0u;

        uint64_t offset = static_cast<uint64_t>(chunk_idx) * kChunkElems;
        constexpr event_t event_id = EVENT_ID0;

        DataCopy(ubuf[0u], x_gm[offset], copy_params);
        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);
        DataCopy(y_gm[offset], ubuf[0u], copy_params);
        if constexpr (WAIT_WRITE) {
            SetFlag<HardEvent::MTE3_MTE2>(event_id);
            WaitFlag<HardEvent::MTE3_MTE2>(event_id);
        }
    }
};

template <typename DT_X>
class KernelPoint7BetaCase6 {
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y) {
        constexpr uint32_t kBlockBytes = 32u;
        constexpr uint32_t kMaxBlocks = 2048u;
        constexpr uint32_t kElemsPerBlock = kBlockBytes / static_cast<uint32_t>(sizeof(DT_X));
        constexpr uint32_t kMaxElems = kMaxBlocks * kElemsPerBlock;
        constexpr uint32_t kTotalElems = 1u * 2u * 2u * 16384u;
        constexpr uint32_t kTotalBlocks = (kTotalElems * static_cast<uint32_t>(sizeof(DT_X))) / kBlockBytes;

        GlobalTensor<DT_X> x_gm;
        GlobalTensor<DT_X> y_gm;
        x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

        LocalTensor<DT_X> tmp_local(TPosition::VECCALC, 0, kMaxElems);

        uint32_t block_idx = GetBlockIdx();
        uint32_t block_num = GetBlockNum();
        uint32_t blocks_per_core = (kTotalBlocks + block_num - 1u) / block_num;
        uint32_t start_block = block_idx * blocks_per_core;
        if (start_block >= kTotalBlocks) {
            return;
        }

        uint32_t copy_blocks = kTotalBlocks - start_block;
        if (copy_blocks > blocks_per_core) {
            copy_blocks = blocks_per_core;
        }
        uint32_t elem_offset = start_block * kElemsPerBlock;

        DataCopyParams copy_params;
        copy_params.blockCount = 1u;
        copy_params.srcStride = 0u;
        copy_params.dstStride = 0u;

        while (copy_blocks > 0u) {
            uint32_t blocks = copy_blocks > kMaxBlocks ? kMaxBlocks : copy_blocks;
            uint32_t remaining_blocks = copy_blocks - blocks;
            copy_params.blockLen = static_cast<uint16_t>(blocks);
            DataCopy(tmp_local, x_gm[elem_offset], copy_params);
            PipeBarrier<PIPE_ALL>();
            DataCopy(y_gm[elem_offset], tmp_local, copy_params);
            if (remaining_blocks > 0u) {
                PipeBarrier<PIPE_ALL>();
            }
            elem_offset += blocks * kElemsPerBlock;
            copy_blocks = remaining_blocks;
        }
    }
};

template <typename DT_X, uint32_t BTS_POINT>
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    if constexpr (BTS_POINT == BTS_TPL_POINT3) {
        KernelPoint3Tile4<DT_X> op;
        op.Process(x, y);
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT3_TILE2) {
        KernelPoint3Tile2<DT_X> op;
        op.Process(x, y);
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT3_TILE2_DB) {
        KernelPoint3Tile2DoubleBuffer<DT_X> op;
        op.Process(x, y);
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT3_PHASE7) {
        KernelPoint3Phase7<DT_X> op;
        op.Process(x, y);
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT3_BETA) {
        KernelPoint3BetaCase2<DT_X> op;
        op.Process(x, y);
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT10) {
        KernelPoint10Tile8Tpl<DT_X> op;
        op.Process(x, y);
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT8_BETA) {
        if constexpr (sizeof(DT_X) == 2u) {
            RunPoint8BetaCase7<DT_X>(x, y);
        }
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT9_BETA) {
        if constexpr (sizeof(DT_X) == 2u) {
            RunPoint9BetaCase8<DT_X>(x, y);
        }
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT10_BETA) {
        if constexpr (sizeof(DT_X) == 2u) {
            RunPoint10BetaCase9<DT_X>(x, y);
        }
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT1) {
        KernelPoint1BetaCase0<DT_X> op;
        op.Process(x, y);
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT1_HALF) {
        KernelPoint1BetaHalfWidth<DT_X> op;
        op.Process(x, y);
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT1_QUARTER) {
        KernelPoint1BetaSplitWidth<DT_X, 4u> op;
        op.Process(x, y);
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT2_BETA) {
        KernelPoint2BetaCase1<DT_X> op;
        op.Process(x, y);
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT4_BETA) {
        if constexpr (sizeof(DT_X) == 4u) {
            KernelPoint4BetaCase3<DT_X> op;
            op.Process(x, y);
        }
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT5_BETA) {
        if constexpr (sizeof(DT_X) == 2u) {
            KernelPoint5BetaCase4<DT_X> op;
            op.Process(x, y);
        }
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT4) {
        KernelPoint4Tile4Row<DT_X> op;
        op.Process(x, y);
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT4_BATCH) {
        KernelPoint4Batch5Core<DT_X> op;
        op.Process(x, y);
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT4_2ROW) {
        KernelPoint4Tile2Row<DT_X> op;
        op.Process(x, y);
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT4_UNROLL) {
        KernelPoint4Tile4RowUnroll<DT_X> op;
        op.Process(x, y);
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT4_ROW40) {
        KernelPoint4Row40<DT_X> op;
        op.Process(x, y);
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT4_READ_CONTIG) {
        KernelPoint4ReadContig<DT_X> op;
        op.Process(x, y);
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT6) {
        KernelPoint6Stream8Core<DT_X> op;
        op.Process(x, y);
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT6_BETA) {
        if constexpr (sizeof(DT_X) == 2u) {
            KernelPoint6BetaCase5<DT_X> op;
            op.Process(x, y);
        }
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT6_16CORE) {
        KernelPoint6Pixel16Core<DT_X> op;
        op.Process(x, y);
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT6_4CORE_ROW) {
        KernelPoint6Row4Core<DT_X> op;
        op.Process(x, y);
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT7) {
        KernelPoint7Linear8Core<DT_X, 8u> op;
        op.template Process<false>(x, y);
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT7_BETA) {
        if constexpr (sizeof(DT_X) == 2u) {
            KernelPoint7BetaCase6<DT_X> op;
            op.Process(x, y);
        }
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT7_4CORE) {
        KernelPoint7Linear8Core<DT_X, 4u> op;
        op.template Process<false>(x, y);
        return;
    }
    if constexpr (BTS_POINT == BTS_TPL_POINT7_16CORE) {
        KernelPoint7Linear8Core<DT_X, 16u> op;
        op.template Process<false>(x, y);
        return;
    }

    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingData, tiling_data, tiling);
    if (tiling_data.use_stride_copy == 67u) {
        KernelPoint7Linear8Core<DT_X, 8u> op;
        op.template Process<true>(x, y);
        return;
    }
    if (tiling_data.use_stride_copy == 74u) {
        KernelPoint7Linear8Core<DT_X, 8u> op;
        op.template Process<false>(x, y);
        return;
    }
    KernelBatchToSpace<DT_X> op;
    op.Init(x, y, tiling_data);
    op.Process();
}
