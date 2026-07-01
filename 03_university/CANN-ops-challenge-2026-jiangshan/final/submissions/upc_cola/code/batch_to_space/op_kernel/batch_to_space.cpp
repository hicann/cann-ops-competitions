// Kernel侧核函数实现
#include "kernel_operator.h"

#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

using namespace AscendC;

// Keep Host buf_budget consistent with the common paths.
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t MODE3_BUFFER_NUM = 1;
constexpr int32_t MODE5_BUFFER_NUM = 1;
constexpr int32_t MODE6_BUFFER_NUM = 4;
constexpr uint32_t MODE3_BUF_BYTES = 8192u;

template <class DT_X>
class KernelBatchToSpaceMode0 {
public:
    __aicore__ inline KernelBatchToSpaceMode0() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &t) {
        H_ = t.height;
        W_ = t.width;
        D_ = t.depth;
        bs_ = t.block_size;
        ct_ = t.crop_top;
        cl_ = t.crop_left;
        OB_ = t.out_batch;
        OH_ = t.out_height;
        OW_ = t.out_width;
        w_chunk_ = t.w_chunk;
        h_chunk_ = t.h_chunk;
        num_chunks_ = t.num_chunks;

        InitUnitRange(t);

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y);

        d_bytes_ = D_ * sizeof(DT_X);
        aligned_d_bytes_ = (d_bytes_ + 31) / 32 * 32;

        uint32_t buf_bytes = w_chunk_ * aligned_d_bytes_;
        if (buf_bytes < 32) buf_bytes = 32;
        pipe_.InitBuffer(queue_, BUFFER_NUM, buf_bytes);
    }

    __aicore__ inline void Process() {
        if (num_units_ == 0) return;

        bool has_pending = false;
        uint64_t pending_out_off = 0;
        uint32_t pending_w_count = 0;
        uint32_t pending_src_ub_elem_off = 0;
        uint32_t h_tiles = (H_ + h_chunk_ - 1u) / h_chunk_;

        for (uint32_t i = 0; i < num_units_; ++i) {
            uint32_t unit = start_unit_ + i;
            uint32_t bh_tile_idx = unit / num_chunks_;
            uint32_t chunk_idx = unit - bh_tile_idx * num_chunks_;
            uint32_t b = bh_tile_idx / h_tiles;
            uint32_t h_tile = bh_tile_idx - b * h_tiles;
            uint32_t h_begin = h_tile * h_chunk_;
            uint32_t h_end = h_begin + h_chunk_;
            if (h_end > H_) h_end = H_;

            uint32_t bb = b / OB_;
            uint32_t n_out = b - bb * OB_;
            uint32_t bh = bb / bs_;
            uint32_t bw = bb - bh * bs_;

            uint32_t chunk_start = chunk_idx * w_chunk_;
            uint32_t chunk_end = chunk_start + w_chunk_;
            if (chunk_end > W_) chunk_end = W_;

            uint32_t w_valid_start;
            if (cl_ <= bw) {
                w_valid_start = 0;
            } else {
                w_valid_start = (cl_ - bw + bs_ - 1) / bs_;
            }
            uint32_t w_valid_end = 0;
            if (cl_ + OW_ > bw) {
                uint32_t e = (cl_ + OW_ - bw + bs_ - 1) / bs_;
                w_valid_end = (e > W_) ? W_ : e;
            }

            uint32_t local_start = (w_valid_start > chunk_start) ? w_valid_start : chunk_start;
            uint32_t local_end = (w_valid_end < chunk_end) ? w_valid_end : chunk_end;
            if (local_start >= local_end) continue;

            uint32_t w_count = local_end - local_start;
            int32_t w_out_start = (int32_t)(local_start * bs_) + (int32_t)bw - (int32_t)cl_;

            for (uint32_t h = h_begin; h < h_end; ++h) {
                uint32_t real_h = h * bs_ + bh;
                if (real_h < ct_ || real_h >= ct_ + OH_) continue;

                uint32_t h_out = real_h - ct_;
                uint64_t in_off = ((uint64_t)b * H_ + h) * W_ * D_ + (uint64_t)local_start * D_;
                uint64_t out_off = ((uint64_t)n_out * OH_ + h_out) * OW_ * D_ +
                                   (uint64_t)w_out_start * D_;

                LocalTensor<DT_X> buf = queue_.AllocTensor<DT_X>();
                DataCopyExtParams inP;
                inP.blockCount = (uint16_t)w_count;
                inP.blockLen = d_bytes_;
                inP.srcStride = 0;
                inP.dstStride = 0;
                inP.rsv = 0;
                DataCopyPadExtParams<DT_X> padP{false, 0, 0, 0};
                DataCopyPad(buf, xGm_[in_off], inP, padP);
                queue_.EnQue(buf);

                uint32_t src_ub_elem_off = 0;
                if (has_pending) {
                    LocalTensor<DT_X> prev = queue_.DeQue<DT_X>();
                    DataCopyExtParams op;
                    op.blockCount = (uint16_t)pending_w_count;
                    op.blockLen = d_bytes_;
                    op.srcStride = 0;
                    op.dstStride = (bs_ - 1) * d_bytes_;
                    op.rsv = 0;
                    DataCopyPad(yGm_[pending_out_off], prev[pending_src_ub_elem_off], op);
                    queue_.FreeTensor(prev);
                }

                pending_out_off = out_off;
                pending_w_count = w_count;
                pending_src_ub_elem_off = src_ub_elem_off;
                has_pending = true;
            }
        }

        if (has_pending) {
            LocalTensor<DT_X> last = queue_.DeQue<DT_X>();
            DataCopyExtParams op;
            op.blockCount = (uint16_t)pending_w_count;
            op.blockLen = d_bytes_;
            op.srcStride = 0;
            op.dstStride = (bs_ - 1) * d_bytes_;
            op.rsv = 0;
            DataCopyPad(yGm_[pending_out_off], last[pending_src_ub_elem_off], op);
            queue_.FreeTensor(last);
        }
    }

private:
    __aicore__ inline void InitUnitRange(const BatchToSpaceTilingData &t) {
        uint32_t bid = GetBlockIdx();
        uint32_t used = t.used_cores;
        uint32_t per = t.units_per_core;
        uint32_t tail = t.tail_units;
        if (bid >= used) {
            start_unit_ = 0;
            num_units_ = 0;
        } else {
            uint32_t extra = (bid < tail) ? 1 : 0;
            uint32_t prefix = (bid < tail) ? bid : tail;
            start_unit_ = bid * per + prefix;
            num_units_ = per + extra;
        }
    }

    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> queue_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;

    uint32_t H_, W_, D_, bs_;
    uint32_t ct_, cl_;
    uint32_t OB_, OH_, OW_;
    uint32_t start_unit_, num_units_;
    uint32_t d_bytes_, aligned_d_bytes_;
    uint32_t w_chunk_, h_chunk_, num_chunks_;
};

template <class DT_X>
class KernelBatchToSpaceMode1 {
public:
    __aicore__ inline KernelBatchToSpaceMode1() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &t) {
        H_ = t.height;
        W_ = t.width;
        D_ = t.depth;
        bs_ = t.block_size;
        ct_ = t.crop_top;
        cl_ = t.crop_left;
        OB_ = t.out_batch;
        OH_ = t.out_height;
        OW_ = t.out_width;
        w_chunk_ = t.w_chunk;
        h_chunk_ = t.h_chunk;
        num_chunks_ = t.num_chunks;

        InitUnitRange(t);

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y);

        d_bytes_ = D_ * sizeof(DT_X);
        uint32_t buf_bytes = w_chunk_ * d_bytes_;
        if (buf_bytes < 32) buf_bytes = 32;
        pipe_.InitBuffer(queue_, BUFFER_NUM, buf_bytes);
    }

    __aicore__ inline void Process() {
        if (num_units_ == 0) return;

        uint32_t cl_mod = cl_ % bs_;
        uint32_t d_b32_units = d_bytes_ / 32u;
        uint32_t h_tiles = (OH_ + h_chunk_ - 1u) / h_chunk_;

        for (uint32_t i = 0; i < num_units_; ++i) {
            uint32_t unit = start_unit_ + i;
            uint32_t nh_tile_idx = unit / num_chunks_;
            uint32_t chunk_idx = unit - nh_tile_idx * num_chunks_;
            uint32_t n_out = nh_tile_idx / h_tiles;
            uint32_t h_tile = nh_tile_idx - n_out * h_tiles;
            uint32_t h_out_begin = h_tile * h_chunk_;
            uint32_t h_out_end_tile = h_out_begin + h_chunk_;
            if (h_out_end_tile > OH_) h_out_end_tile = OH_;

            uint32_t w_out_start = chunk_idx * w_chunk_;
            uint32_t w_out_end = w_out_start + w_chunk_;
            if (w_out_end > OW_) w_out_end = OW_;
            if (w_out_start >= w_out_end) continue;
            uint32_t w_out_count = w_out_end - w_out_start;

            uint32_t real_w_start = w_out_start + cl_;
            uint32_t bw0 = real_w_start & 1u;
            uint32_t first_w_in0 = real_w_start >> 1;
            uint32_t count0 = (w_out_count + 1u) >> 1;
            uint32_t bw1 = bw0 ^ 1u;
            uint32_t first_w_in1 = (real_w_start + 1u) >> 1;
            uint32_t count1 = w_out_count >> 1;
            uint32_t w_out_start_mod = w_out_start % bs_;

            for (uint32_t h_out = h_out_begin; h_out < h_out_end_tile; ++h_out) {
                uint32_t real_h = h_out + ct_;
                uint32_t bh = real_h % bs_;
                uint32_t h = real_h / bs_;

                uint64_t out_off = ((uint64_t)n_out * OH_ + h_out) * OW_ * D_ +
                                   (uint64_t)w_out_start * D_;

                LocalTensor<DT_X> buf = queue_.AllocTensor<DT_X>();

                if (bs_ == 2u) {
                    uint32_t b0 = (bh * 2u + bw0) * OB_ + n_out;
                    uint64_t in_off0 = ((uint64_t)b0 * H_ + h) * W_ * D_ +
                                       (uint64_t)first_w_in0 * D_;

                    DataCopyParams inP0;
                    inP0.blockCount = (uint16_t)count0;
                    inP0.blockLen = (uint16_t)d_b32_units;
                    inP0.srcStride = 0;
                    inP0.dstStride = d_b32_units;
                    DataCopy(buf, xGm_[in_off0], inP0);

                    if (w_out_count > 1u) {
                        uint32_t b1 = (bh * 2u + bw1) * OB_ + n_out;
                        uint64_t in_off1 = ((uint64_t)b1 * H_ + h) * W_ * D_ +
                                           (uint64_t)first_w_in1 * D_;

                        DataCopyParams inP1;
                        inP1.blockCount = (uint16_t)count1;
                        inP1.blockLen = (uint16_t)d_b32_units;
                        inP1.srcStride = 0;
                        inP1.dstStride = d_b32_units;
                        DataCopy(buf[D_], xGm_[in_off1], inP1);
                    }
                } else {
                    for (uint32_t bw = 0; bw < bs_; ++bw) {
                        uint32_t b = (bh * bs_ + bw) * OB_ + n_out;
                        uint32_t target_mod = (bw + bs_ - cl_mod) % bs_;
                        uint32_t offset;
                        if (target_mod >= w_out_start_mod) {
                            offset = target_mod - w_out_start_mod;
                        } else {
                            offset = target_mod + bs_ - w_out_start_mod;
                        }
                        if (offset >= w_out_count) continue;

                        uint32_t first_w_out = w_out_start + offset;
                        uint32_t count_for_bw = (w_out_end - first_w_out + bs_ - 1) / bs_;
                        uint32_t first_real_w = first_w_out + cl_;
                        uint32_t first_w_in = first_real_w / bs_;
                        uint64_t in_off = ((uint64_t)b * H_ + h) * W_ * D_ +
                                          (uint64_t)first_w_in * D_;
                        uint32_t dst_off_elems = offset * D_;

                        DataCopyParams inP;
                        inP.blockCount = (uint16_t)count_for_bw;
                        inP.blockLen = (uint16_t)d_b32_units;
                        inP.srcStride = 0;
                        inP.dstStride = (bs_ - 1) * d_b32_units;
                        DataCopy(buf[dst_off_elems], xGm_[in_off], inP);
                    }
                }

                queue_.EnQue(buf);

                LocalTensor<DT_X> out = queue_.DeQue<DT_X>();
                DataCopyParams op;
                op.blockCount = 1;
                op.blockLen = (uint16_t)(w_out_count * d_b32_units);
                op.srcStride = 0;
                op.dstStride = 0;
                DataCopy(yGm_[out_off], out, op);
                queue_.FreeTensor(out);
            }
        }
    }

private:
    __aicore__ inline void InitUnitRange(const BatchToSpaceTilingData &t) {
        uint32_t bid = GetBlockIdx();
        uint32_t used = t.used_cores;
        uint32_t per = t.units_per_core;
        uint32_t tail = t.tail_units;
        if (bid >= used) {
            start_unit_ = 0;
            num_units_ = 0;
        } else {
            uint32_t extra = (bid < tail) ? 1 : 0;
            uint32_t prefix = (bid < tail) ? bid : tail;
            start_unit_ = bid * per + prefix;
            num_units_ = per + extra;
        }
    }

    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> queue_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;

    uint32_t H_, W_, D_, bs_;
    uint32_t ct_, cl_;
    uint32_t OB_, OH_, OW_;
    uint32_t start_unit_, num_units_;
    uint32_t d_bytes_;
    uint32_t w_chunk_, h_chunk_, num_chunks_;
};

template <class DT_X>
class KernelBatchToSpaceMode2 {
public:
    __aicore__ inline KernelBatchToSpaceMode2() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &t) {
        H_ = t.height;
        W_ = t.width;
        D_ = t.depth;
        ct_ = t.crop_top;
        cl_ = t.crop_left;
        OB_ = t.out_batch;
        OH_ = t.out_height;
        OW_ = t.out_width;
        w_chunk_ = t.w_chunk;
        h_chunk_ = t.h_chunk;
        num_chunks_ = t.num_chunks;

        InitUnitRange(t);

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y);

        d_bytes_ = D_ * sizeof(DT_X);
        uint32_t buf_bytes = w_chunk_ * h_chunk_ * d_bytes_;
        if (buf_bytes < 32) buf_bytes = 32;
        pipe_.InitBuffer(queue_, BUFFER_NUM, buf_bytes);
    }

    __aicore__ inline void Process() {
        if (num_units_ == 0) return;

        uint32_t d_b32_units = d_bytes_ / 32u;
        uint32_t h_tiles = (OH_ + h_chunk_ - 1u) / h_chunk_;

        for (uint32_t i = 0; i < num_units_; ++i) {
            uint32_t unit = start_unit_ + i;
            uint32_t nh_tile_idx = unit / num_chunks_;
            uint32_t chunk_idx = unit - nh_tile_idx * num_chunks_;
            uint32_t n_out = nh_tile_idx / h_tiles;
            uint32_t h_tile = nh_tile_idx - n_out * h_tiles;
            uint32_t h_out_begin = h_tile * h_chunk_;
            uint32_t h_out_end_tile = h_out_begin + h_chunk_;
            if (h_out_end_tile > OH_) h_out_end_tile = OH_;
            uint32_t h_out_count = h_out_end_tile - h_out_begin;

            uint32_t w_out_start = chunk_idx * w_chunk_;
            uint32_t w_out_end = w_out_start + w_chunk_;
            if (w_out_end > OW_) w_out_end = OW_;
            if (w_out_start >= w_out_end || h_out_count == 0u) continue;
            uint32_t w_out_count = w_out_end - w_out_start;

            uint32_t real_w_start = w_out_start + cl_;
            uint32_t bw0 = real_w_start & 1u;
            uint32_t first_w_in0 = real_w_start >> 1;
            uint32_t count0 = (w_out_count + 1u) >> 1;
            uint32_t bw1 = bw0 ^ 1u;
            uint32_t first_w_in1 = (real_w_start + 1u) >> 1;
            uint32_t count1 = w_out_count >> 1;

            LocalTensor<DT_X> buf = queue_.AllocTensor<DT_X>();

            for (uint32_t row = 0; row < h_out_count; ++row) {
                uint32_t h_out = h_out_begin + row;
                uint32_t real_h = h_out + ct_;
                uint32_t bh = real_h & 1u;
                uint32_t h = real_h >> 1;
                uint32_t row_base = row * w_out_count * D_;

                uint32_t b0 = (bh * 2u + bw0) * OB_ + n_out;
                uint64_t in_off0 = ((uint64_t)b0 * H_ + h) * W_ * D_ +
                                   (uint64_t)first_w_in0 * D_;

                DataCopyParams inP0;
                inP0.blockCount = (uint16_t)count0;
                inP0.blockLen = (uint16_t)d_b32_units;
                inP0.srcStride = 0;
                inP0.dstStride = d_b32_units;
                DataCopy(buf[row_base], xGm_[in_off0], inP0);

                if (w_out_count > 1u) {
                    uint32_t b1 = (bh * 2u + bw1) * OB_ + n_out;
                    uint64_t in_off1 = ((uint64_t)b1 * H_ + h) * W_ * D_ +
                                       (uint64_t)first_w_in1 * D_;

                    DataCopyParams inP1;
                    inP1.blockCount = (uint16_t)count1;
                    inP1.blockLen = (uint16_t)d_b32_units;
                    inP1.srcStride = 0;
                    inP1.dstStride = d_b32_units;
                    DataCopy(buf[row_base + D_], xGm_[in_off1], inP1);
                }
            }

            queue_.EnQue(buf);

            uint64_t out_off = ((uint64_t)n_out * OH_ + h_out_begin) * OW_ * D_ +
                               (uint64_t)w_out_start * D_;
            LocalTensor<DT_X> out = queue_.DeQue<DT_X>();
            DataCopyParams op;
            op.srcStride = 0;
            if (w_out_count == OW_) {
                op.blockCount = 1;
                op.blockLen = (uint16_t)(h_out_count * w_out_count * d_b32_units);
                op.dstStride = 0;
            } else {
                op.blockCount = (uint16_t)h_out_count;
                op.blockLen = (uint16_t)(w_out_count * d_b32_units);
                op.dstStride = (uint16_t)((OW_ - w_out_count) * d_b32_units);
            }
            DataCopy(yGm_[out_off], out, op);
            queue_.FreeTensor(out);
        }
    }

private:
    __aicore__ inline void InitUnitRange(const BatchToSpaceTilingData &t) {
        uint32_t bid = GetBlockIdx();
        uint32_t used = t.used_cores;
        uint32_t per = t.units_per_core;
        uint32_t tail = t.tail_units;
        if (bid >= used) {
            start_unit_ = 0;
            num_units_ = 0;
        } else {
            uint32_t extra = (bid < tail) ? 1 : 0;
            uint32_t prefix = (bid < tail) ? bid : tail;
            start_unit_ = bid * per + prefix;
            num_units_ = per + extra;
        }
    }

    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> queue_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;

    uint32_t H_, W_, D_;
    uint32_t ct_, cl_;
    uint32_t OB_, OH_, OW_;
    uint32_t start_unit_, num_units_;
    uint32_t d_bytes_;
    uint32_t w_chunk_, h_chunk_, num_chunks_;
};

template <class DT_X>
class KernelBatchToSpaceMode6 {
public:
    __aicore__ inline KernelBatchToSpaceMode6() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &t) {
        H_ = t.height;
        W_ = t.width;
        D_ = t.depth;
        ct_ = t.crop_top;
        cl_ = t.crop_left;
        OB_ = t.out_batch;
        OH_ = t.out_height;
        OW_ = t.out_width;
        w_chunk_ = t.w_chunk;
        h_chunk_ = t.h_chunk;
        num_chunks_ = t.num_chunks;

        InitUnitRange(t);

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y);

        d_bytes_ = D_ * sizeof(DT_X);
        uint32_t buf_bytes = w_chunk_ * h_chunk_ * d_bytes_;
        if (buf_bytes < 32) buf_bytes = 32;
        pipe_.InitBuffer(queue_, MODE6_BUFFER_NUM, buf_bytes);
    }

    __aicore__ inline void Process() {
        if (num_units_ == 0) return;

        uint32_t d_b32_units = d_bytes_ / 32u;
        uint32_t h_tiles = (OH_ + h_chunk_ - 1u) / h_chunk_;
        uint32_t batch_stride = H_ * W_ * D_;
        uint32_t h_stride = W_ * D_;
        uint32_t out_h_stride = OW_ * D_;
        uint32_t out_n_stride = OH_ * OW_ * D_;
        uint32_t two_ob = OB_ << 1;

        for (uint32_t i = 0; i < num_units_; ++i) {
            uint32_t unit = start_unit_ + i;
            uint32_t nh_tile_idx = unit / num_chunks_;
            uint32_t chunk_idx = unit - nh_tile_idx * num_chunks_;
            uint32_t n_out = nh_tile_idx / h_tiles;
            uint32_t h_tile = nh_tile_idx - n_out * h_tiles;
            uint32_t h_out_begin = h_tile * h_chunk_;
            uint32_t h_out_end_tile = h_out_begin + h_chunk_;
            if (h_out_end_tile > OH_) h_out_end_tile = OH_;
            uint32_t h_out_count = h_out_end_tile - h_out_begin;

            uint32_t w_out_start = chunk_idx * w_chunk_;
            uint32_t w_out_end = w_out_start + w_chunk_;
            if (w_out_end > OW_) w_out_end = OW_;
            if (w_out_start >= w_out_end || h_out_count == 0u) continue;
            uint32_t w_out_count = w_out_end - w_out_start;

            uint32_t real_w_start = w_out_start + cl_;
            uint32_t bw0 = real_w_start & 1u;
            uint32_t first_w_in0 = real_w_start >> 1;
            uint32_t count0 = (w_out_count + 1u) >> 1;
            uint32_t bw1 = bw0 ^ 1u;
            uint32_t first_w_in1 = (real_w_start + 1u) >> 1;
            uint32_t count1 = w_out_count >> 1;
            uint32_t w_in0_off = first_w_in0 * D_;
            uint32_t w_in1_off = first_w_in1 * D_;
            uint32_t row_elems = w_out_count * D_;
            uint32_t b0_base = bw0 * OB_ + n_out;
            uint32_t b1_base = bw1 * OB_ + n_out;

            DataCopyParams inP0;
            inP0.blockCount = (uint16_t)count0;
            inP0.blockLen = (uint16_t)d_b32_units;
            inP0.srcStride = 0;
            inP0.dstStride = d_b32_units;

            DataCopyParams inP1;
            inP1.blockCount = (uint16_t)count1;
            inP1.blockLen = (uint16_t)d_b32_units;
            inP1.srcStride = 0;
            inP1.dstStride = d_b32_units;

            LocalTensor<DT_X> buf = queue_.AllocTensor<DT_X>();

            for (uint32_t row = 0; row < h_out_count; ++row) {
                uint32_t h_out = h_out_begin + row;
                uint32_t real_h = h_out + ct_;
                uint32_t bh = real_h & 1u;
                uint32_t h = real_h >> 1;
                uint32_t row_base = row * row_elems;
                uint32_t bh_batch_off = bh * two_ob;
                uint64_t row_in_base = (uint64_t)h * h_stride;

                uint32_t b0 = b0_base + bh_batch_off;
                uint64_t in_off0 = (uint64_t)b0 * batch_stride + row_in_base + w_in0_off;
                DataCopy(buf[row_base], xGm_[in_off0], inP0);

                if (w_out_count > 1u) {
                    uint32_t b1 = b1_base + bh_batch_off;
                    uint64_t in_off1 = (uint64_t)b1 * batch_stride + row_in_base + w_in1_off;
                    DataCopy(buf[row_base + D_], xGm_[in_off1], inP1);
                }
            }

            queue_.EnQue(buf);

            uint64_t out_off = (uint64_t)n_out * out_n_stride +
                               (uint64_t)h_out_begin * out_h_stride +
                               (uint64_t)w_out_start * D_;
            LocalTensor<DT_X> out = queue_.DeQue<DT_X>();
            DataCopyParams op;
            op.srcStride = 0;
            if (w_out_count == OW_) {
                op.blockCount = 1;
                op.blockLen = (uint16_t)(h_out_count * w_out_count * d_b32_units);
                op.dstStride = 0;
            } else {
                op.blockCount = (uint16_t)h_out_count;
                op.blockLen = (uint16_t)(w_out_count * d_b32_units);
                op.dstStride = (uint16_t)((OW_ - w_out_count) * d_b32_units);
            }
            DataCopy(yGm_[out_off], out, op);
            queue_.FreeTensor(out);
        }
    }

private:
    __aicore__ inline void InitUnitRange(const BatchToSpaceTilingData &t) {
        uint32_t bid = GetBlockIdx();
        uint32_t used = t.used_cores;
        uint32_t per = t.units_per_core;
        uint32_t tail = t.tail_units;
        if (bid >= used) {
            start_unit_ = 0;
            num_units_ = 0;
        } else {
            uint32_t extra = (bid < tail) ? 1 : 0;
            uint32_t prefix = (bid < tail) ? bid : tail;
            start_unit_ = bid * per + prefix;
            num_units_ = per + extra;
        }
    }

    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, MODE6_BUFFER_NUM> queue_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;

    uint32_t H_, W_, D_;
    uint32_t ct_, cl_;
    uint32_t OB_, OH_, OW_;
    uint32_t start_unit_, num_units_;
    uint32_t d_bytes_;
    uint32_t w_chunk_, h_chunk_, num_chunks_;
};

template <class DT_X>
class KernelBatchToSpaceMode3 {
public:
    __aicore__ inline KernelBatchToSpaceMode3() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &t) {
        H_ = t.height;
        W_ = t.width;
        D_ = t.depth;
        ct_ = t.crop_top;
        cl_ = t.crop_left;
        OB_ = t.out_batch;
        OH_ = t.out_height;
        num_chunks_ = t.num_chunks;
        d_chunk_ = t.d_chunk;
        d_num_chunks_ = t.d_num_chunks;

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y);

        d_b32_units_ = D_ * sizeof(DT_X) / 32u;
        if (d_chunk_ == 0u) d_chunk_ = d_b32_units_;
        if (d_num_chunks_ == 0u) d_num_chunks_ = 1u;

        pipe_.InitBuffer(queue_, MODE3_BUFFER_NUM, MODE3_BUF_BYTES);
    }

    __aicore__ inline void Process() {
        uint32_t elem_per_32b = 32u / sizeof(DT_X);
        uint32_t unit = GetBlockIdx();
        uint32_t outer_unit = unit / d_num_chunks_;
        uint32_t d_chunk_idx = unit - outer_unit * d_num_chunks_;
        uint32_t d_start_b32 = d_chunk_idx * d_chunk_;
        if (d_start_b32 >= d_b32_units_) return;

        uint32_t d_len_b32 = d_chunk_;
        if (d_start_b32 + d_len_b32 > d_b32_units_) {
            d_len_b32 = d_b32_units_ - d_start_b32;
        }
        uint32_t d_elem_off = d_start_b32 * elem_per_32b;

        uint32_t nh_idx = outer_unit / num_chunks_;
        uint32_t w_out_start = outer_unit - nh_idx * num_chunks_;
        uint32_t n_out = nh_idx / OH_;
        uint32_t h_out = nh_idx - n_out * OH_;

        uint32_t real_h = h_out + ct_;
        uint32_t bh = real_h & 1u;
        uint32_t h = real_h >> 1;
        uint32_t real_w_start = w_out_start + cl_;
        uint32_t bw = real_w_start & 1u;
        uint32_t w_in = real_w_start >> 1;

        uint64_t out_off = (((uint64_t)n_out * OH_ + h_out) * num_chunks_ + w_out_start) * D_ +
                           d_elem_off;

        LocalTensor<DT_X> buf = queue_.AllocTensor<DT_X>();
        uint32_t b = (bh * 2u + bw) * OB_ + n_out;
        uint64_t in_off = (((uint64_t)b * H_ + h) * W_ + w_in) * D_ + d_elem_off;

        DataCopyParams p;
        p.blockCount = 1;
        p.blockLen = (uint16_t)d_len_b32;
        p.srcStride = 0;
        p.dstStride = 0;
        DataCopy(buf, xGm_[in_off], p);

        queue_.EnQue(buf);

        LocalTensor<DT_X> out = queue_.DeQue<DT_X>();
        DataCopy(yGm_[out_off], out, p);
        queue_.FreeTensor(out);
    }

private:
    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, MODE3_BUFFER_NUM> queue_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;

    uint32_t H_, W_, D_;
    uint32_t ct_, cl_;
    uint32_t OB_, OH_;
    uint32_t d_b32_units_;
    uint32_t num_chunks_;
    uint32_t d_chunk_, d_num_chunks_;
};

template <class DT_X>
class KernelBatchToSpaceMode4 {
public:
    __aicore__ inline KernelBatchToSpaceMode4() {}

    __aicore__ inline void InitAndProcess(GM_ADDR x, GM_ADDR y) {
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y);

        pipe_.InitBuffer(queue_, MODE3_BUFFER_NUM, kBufBytes);
        uint32_t unit = GetBlockIdx();
        uint32_t b = ((unit >> 1) & 2u) | (unit & 1u);
        uint32_t h = unit >> 3;
        uint32_t w = (unit >> 1) & 1u;
        uint64_t in_off = (uint64_t)((((b << 1) + h) << 1) + w) << 12;
        uint64_t out_off = (uint64_t)unit << 12;

        LocalTensor<DT_X> buf = queue_.AllocTensor<DT_X>();
        DataCopy(buf, xGm_[in_off], kD);
        queue_.EnQue(buf);

        LocalTensor<DT_X> out = queue_.DeQue<DT_X>();
        DataCopy(yGm_[out_off], out, kD);
        queue_.FreeTensor(out);
    }

private:
    static constexpr uint32_t kD = 4096u;
    static constexpr uint32_t kBufBytes = kD * sizeof(DT_X);

    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, MODE3_BUFFER_NUM> queue_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
};

template <class DT_X>
class KernelBatchToSpaceMode7 {
public:
    __aicore__ inline KernelBatchToSpaceMode7() {}

    __aicore__ inline void InitAndProcess(GM_ADDR x, GM_ADDR y) {
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y);

        pipe_.InitBuffer(queue_, MODE3_BUFFER_NUM, kBufBytes);
        uint32_t unit = GetBlockIdx();
        uint64_t off = (uint64_t)unit << 12;

        LocalTensor<DT_X> buf = queue_.AllocTensor<DT_X>();
        DataCopy(buf, xGm_[off], kChunkElems);
        queue_.EnQue(buf);

        LocalTensor<DT_X> out = queue_.DeQue<DT_X>();
        DataCopy(yGm_[off], out, kChunkElems);
        queue_.FreeTensor(out);
    }

private:
    static constexpr uint32_t kChunkElems = 4096u;
    static constexpr uint32_t kBufBytes = kChunkElems * sizeof(DT_X);

    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, MODE3_BUFFER_NUM> queue_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
};

template <class DT_X>
class KernelBatchToSpaceMode5 {
public:
    __aicore__ inline KernelBatchToSpaceMode5() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &t) {
        H_ = t.height;
        W_ = t.width;
        D_ = t.depth;
        ct_ = t.crop_top;
        cl_ = t.crop_left;
        OB_ = t.out_batch;
        OH_ = t.out_height;
        OW_ = t.out_width;
        h_chunk_ = t.h_chunk;

        InitUnitRange(t);

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y);

        d_bytes_ = D_ * sizeof(DT_X);
        aligned_d_bytes_ = (d_bytes_ + 31u) / 32u * 32u;
        uint32_t buf_bytes = h_chunk_ * W_ * aligned_d_bytes_;
        if (buf_bytes < 32u) buf_bytes = 32u;
        pipe_.InitBuffer(queue_, MODE5_BUFFER_NUM, buf_bytes);
    }

    __aicore__ inline void Process() {
        if (num_units_ == 0) return;

        uint32_t h_tiles = (H_ + h_chunk_ - 1u) / h_chunk_;
        uint32_t aligned_d_elems = aligned_d_bytes_ / sizeof(DT_X);
        uint32_t crop_h_end = ct_ + OH_;

        for (uint32_t i = 0; i < num_units_; ++i) {
            uint32_t unit = start_unit_ + i;
            uint32_t b = unit / h_tiles;
            uint32_t h_tile = unit - b * h_tiles;
            uint32_t h_begin = h_tile * h_chunk_;
            uint32_t h_end = h_begin + h_chunk_;
            if (h_end > H_) h_end = H_;
            uint32_t h_count = h_end - h_begin;
            if (h_count == 0u) continue;

            uint32_t bb = b / OB_;
            uint32_t n_out = b - bb * OB_;
            uint32_t bh = bb >> 1;
            uint32_t bw = bb & 1u;

            uint32_t w_valid_start;
            if (cl_ <= bw) {
                w_valid_start = 0;
            } else {
                w_valid_start = (cl_ - bw + 1u) >> 1;
            }
            uint32_t w_valid_end = 0;
            if (cl_ + OW_ > bw) {
                uint32_t e = (cl_ + OW_ - bw + 1u) >> 1;
                w_valid_end = (e > W_) ? W_ : e;
            }
            if (w_valid_start >= w_valid_end) continue;
            uint32_t w_count = w_valid_end - w_valid_start;
            uint32_t w_out_start = (w_valid_start << 1) + bw - cl_;

            uint64_t in_off = ((uint64_t)b * H_ + h_begin) * W_ * D_;
            LocalTensor<DT_X> buf = queue_.AllocTensor<DT_X>();
            DataCopyExtParams inP;
            inP.blockCount = (uint16_t)(h_count * W_);
            inP.blockLen = d_bytes_;
            inP.srcStride = 0;
            inP.dstStride = 0;
            inP.rsv = 0;
            DataCopyPadExtParams<DT_X> padP{false, 0, 0, 0};
            DataCopyPad(buf, xGm_[in_off], inP, padP);
            queue_.EnQue(buf);

            LocalTensor<DT_X> out = queue_.DeQue<DT_X>();
            for (uint32_t row = 0; row < h_count; ++row) {
                uint32_t real_h = ((h_begin + row) << 1) + bh;
                if (real_h < ct_ || real_h >= crop_h_end) continue;
                uint32_t h_out = real_h - ct_;
                uint64_t out_off = ((uint64_t)n_out * OH_ + h_out) * OW_ * D_ +
                                   (uint64_t)w_out_start * D_;
                uint32_t src_off = row * W_ * aligned_d_elems + w_valid_start * aligned_d_elems;

                DataCopyExtParams op;
                op.blockCount = (uint16_t)w_count;
                op.blockLen = d_bytes_;
                op.srcStride = 0;
                op.dstStride = d_bytes_;
                op.rsv = 0;
                DataCopyPad(yGm_[out_off], out[src_off], op);
            }
            queue_.FreeTensor(out);
        }
    }

private:
    __aicore__ inline void InitUnitRange(const BatchToSpaceTilingData &t) {
        uint32_t bid = GetBlockIdx();
        uint32_t used = t.used_cores;
        uint32_t per = t.units_per_core;
        uint32_t tail = t.tail_units;
        if (bid >= used) {
            start_unit_ = 0;
            num_units_ = 0;
        } else {
            uint32_t extra = (bid < tail) ? 1 : 0;
            uint32_t prefix = (bid < tail) ? bid : tail;
            start_unit_ = bid * per + prefix;
            num_units_ = per + extra;
        }
    }

    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, MODE5_BUFFER_NUM> queue_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;

    uint32_t H_, W_, D_;
    uint32_t ct_, cl_;
    uint32_t OB_, OH_, OW_;
    uint32_t start_unit_, num_units_;
    uint32_t d_bytes_, aligned_d_bytes_;
    uint32_t h_chunk_;
};

template <class DT_X>
class KernelBatchToSpaceMode8 {
public:
    __aicore__ inline KernelBatchToSpaceMode8() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &t) {
        H_ = t.height;
        W_ = t.width;
        D_ = t.depth;
        ct_ = t.crop_top;
        cl_ = t.crop_left;
        OB_ = t.out_batch;
        OH_ = t.out_height;
        OW_ = t.out_width;
        w_chunk_ = t.w_chunk;
        num_chunks_ = t.num_chunks;

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y);

        InitUnitRange(t);
        pipe_.InitBuffer(queue_, BUFFER_NUM, kBufBytes);
    }

    __aicore__ inline void Process() {
        if (num_units_ == 0) return;

        for (uint32_t i = 0; i < num_units_; ++i) {
            uint32_t h_out = start_unit_ + i;
            uint32_t real_h = h_out + 1u;
            uint32_t bh = real_h & 1u;
            uint32_t h = real_h >> 1;
            uint32_t b_even_col = (bh << 1) + 1u;
            uint32_t b_odd_col = bh << 1;

            uint64_t row_out_off = (uint64_t)h_out * 254u * 65u;
            uint64_t in_even_off = ((uint64_t)b_even_col * 128u + h) * 128u * 65u;
            uint64_t in_odd_off = ((uint64_t)b_odd_col * 128u + h) * 128u * 65u + 65u;

            DataCopyExtParams inP;
            inP.blockCount = 127u;
            inP.blockLen = kDBytes;
            inP.srcStride = 0;
            inP.dstStride = 0;
            inP.rsv = 0;
            DataCopyPadExtParams<DT_X> padP{false, 0, 0, 0};

            LocalTensor<DT_X> evenBuf = queue_.AllocTensor<DT_X>();
            DataCopyPad(evenBuf, xGm_[in_even_off], inP, padP);
            queue_.EnQue(evenBuf);

            LocalTensor<DT_X> oddBuf = queue_.AllocTensor<DT_X>();
            DataCopyPad(oddBuf, xGm_[in_odd_off], inP, padP);
            queue_.EnQue(oddBuf);

            DataCopyExtParams op;
            op.blockCount = 127u;
            op.blockLen = kDBytes;
            op.srcStride = 0;
            op.dstStride = kDBytes;
            op.rsv = 0;

            LocalTensor<DT_X> evenOut = queue_.DeQue<DT_X>();
            DataCopyPad(yGm_[row_out_off], evenOut, op);
            queue_.FreeTensor(evenOut);

            LocalTensor<DT_X> oddOut = queue_.DeQue<DT_X>();
            DataCopyPad(yGm_[row_out_off + 65u], oddOut, op);
            queue_.FreeTensor(oddOut);
        }
    }

private:
    __aicore__ inline void InitUnitRange(const BatchToSpaceTilingData &t) {
        uint32_t bid = GetBlockIdx();
        uint32_t used = t.used_cores;
        uint32_t per = t.units_per_core;
        uint32_t tail = t.tail_units;
        if (bid >= used) {
            start_unit_ = 0;
            num_units_ = 0;
        } else {
            uint32_t extra = (bid < tail) ? 1 : 0;
            uint32_t prefix = (bid < tail) ? bid : tail;
            start_unit_ = bid * per + prefix;
            num_units_ = per + extra;
        }
    }

    static constexpr uint32_t kDBytes = 130u;
    static constexpr uint32_t kAlignedDBytes = 160u;
    static constexpr uint32_t kBufBytes = 127u * kAlignedDBytes;

    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> queue_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;

    uint32_t H_, W_, D_;
    uint32_t ct_, cl_;
    uint32_t OB_, OH_, OW_;
    uint32_t start_unit_, num_units_;
    uint32_t w_chunk_, num_chunks_;
};

template <typename DT_X, uint32_t MODE>
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);

    if constexpr (MODE == 4u) {
        KernelBatchToSpaceMode4<DT_X> op;
        op.InitAndProcess(x, y);
    } else if constexpr (MODE == 7u) {
        KernelBatchToSpaceMode7<DT_X> op;
        op.InitAndProcess(x, y);
    } else {
        GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingData, tiling_data, tiling);
        if constexpr (MODE == 3u) {
            KernelBatchToSpaceMode3<DT_X> op;
            op.Init(x, y, tiling_data);
            op.Process();
        } else if constexpr (MODE == 2u) {
            KernelBatchToSpaceMode2<DT_X> op;
            op.Init(x, y, tiling_data);
            op.Process();
        } else if constexpr (MODE == 6u) {
            KernelBatchToSpaceMode6<DT_X> op;
            op.Init(x, y, tiling_data);
            op.Process();
        } else if constexpr (MODE == 1u) {
            KernelBatchToSpaceMode1<DT_X> op;
            op.Init(x, y, tiling_data);
            op.Process();
        } else if constexpr (MODE == 5u) {
            KernelBatchToSpaceMode5<DT_X> op;
            op.Init(x, y, tiling_data);
            op.Process();
        } else if constexpr (MODE == 8u) {
            KernelBatchToSpaceMode8<DT_X> op;
            op.Init(x, y, tiling_data);
            op.Process();
        } else {
            KernelBatchToSpaceMode0<DT_X> op;
            op.Init(x, y, tiling_data);
            op.Process();
        }
    }
}
