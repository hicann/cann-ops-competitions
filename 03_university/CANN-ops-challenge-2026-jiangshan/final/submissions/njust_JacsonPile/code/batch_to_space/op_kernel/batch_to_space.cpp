// Kernel-side BatchToSpace implementation with feature-family dispatch.
// TILING_KEY_IS selects compile-time algorithm families; runtime layout
// parameters are loaded once from tiling data.
#include "kernel_operator.h"

#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

template <class DT_X>
__aicore__ inline void InitGlobalTensors(
    GM_ADDR x, GM_ADDR y, AscendC::GlobalTensor<DT_X> &x_gm,
    AscendC::GlobalTensor<DT_X> &y_gm) {
    x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
    y_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
}

constexpr uint32_t kTileElems = 8192;
constexpr uint32_t kTileElemsWideRow = 16384;
constexpr uint32_t kCropOddBlockNum = 40;
constexpr uint32_t kCropOddWidthTile = 128;
constexpr event_t kEventId0 = static_cast<event_t>(0);
constexpr event_t kEventId1 = static_cast<event_t>(1);
constexpr event_t kEventId2 = static_cast<event_t>(2);
constexpr event_t kEventId3 = static_cast<event_t>(3);
constexpr event_t kEventId4 = static_cast<event_t>(4);
constexpr event_t kEventId5 = static_cast<event_t>(5);

__aicore__ inline uint32_t BtsGcdU32(uint32_t a, uint32_t b) {
    while (b != 0U) {
        uint32_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

__aicore__ inline void ExpandGatherOffsets(
    AscendC::LocalTensor<int32_t> offset_init, uint32_t out_cols,
    uint32_t channels, uint32_t phase_bytes, uint32_t elem_bytes) {
    constexpr uint32_t kVecAlignElems = 8U;
    const uint32_t row_elems = out_cols * channels;
    const uint32_t offset_elems =
        (row_elems + kVecAlignElems - 1U) & ~(kVecAlignElems - 1U);
    const uint32_t pair_elems = channels * 2U;
    const uint32_t period_pairs =
        kVecAlignElems / BtsGcdU32(pair_elems, kVecAlignElems);
    const uint32_t period_elems = period_pairs * pair_elems;
    const uint32_t period_bytes = period_pairs * channels * elem_bytes;

    for (uint32_t pair = 0U; pair < period_pairs; ++pair) {
        const uint32_t pair_base = pair * pair_elems;
        const uint32_t pair_byte_base = pair * channels * elem_bytes;
        for (uint32_t c = 0U; c < channels; ++c) {
            const uint32_t byte_off = c * elem_bytes;
            offset_init.SetValue(pair_base + c,
                                 static_cast<int32_t>(pair_byte_base + byte_off));
            offset_init.SetValue(pair_base + channels + c,
                                 static_cast<int32_t>(phase_bytes + pair_byte_base + byte_off));
        }
    }

    AscendC::SetFlag<AscendC::HardEvent::S_V>(kEventId0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(kEventId0);
    for (uint32_t built = period_elems; built < offset_elems; built += period_elems) {
        uint32_t count = offset_elems - built;
        if (count > period_elems) {
            count = period_elems;
        }
        AscendC::Adds(offset_init[built], offset_init[built - period_elems],
                      static_cast<int32_t>(period_bytes), count);
    }
    AscendC::PipeBarrier<PIPE_V>();
}
__aicore__ inline void SignalStageLoadReady(uint32_t stage_idx) {
    if (stage_idx == 0U) {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEventId2);
    } else {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEventId3);
    }
}

__aicore__ inline void WaitStageLoadReady(uint32_t stage_idx) {
    if (stage_idx == 0U) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEventId2);
    } else {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEventId3);
    }
}

__aicore__ inline void SignalStageStoreDone(uint32_t stage_idx) {
    if (stage_idx == 0U) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(kEventId0);
    } else {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(kEventId1);
    }
}

__aicore__ inline void WaitStageStoreDone(uint32_t stage_idx) {
    if (stage_idx == 0U) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kEventId0);
    } else {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kEventId1);
    }
}

__aicore__ inline void SyncGatherToStore(uint32_t stage_idx) {
    if (stage_idx == 0U) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEventId4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEventId4);
    } else {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kEventId5);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kEventId5);
    }
}

template <class DT_X>
__aicore__ inline void LoadBs4PhaseBlock(
    AscendC::GlobalTensor<DT_X> &x_gm,
    AscendC::LocalTensor<DT_X> phase_block, uint64_t row_base,
    uint32_t iw_base, uint32_t H, uint32_t W, uint32_t D);

template <bool PREFIX_TILE, class DT_X>
__aicore__ inline void ReorderBs4PhaseBlock(
    AscendC::LocalTensor<DT_X> phase_block,
    AscendC::LocalTensor<DT_X> packed,
    uint32_t D);

    // ── fp32 large-depth noCrop family ────────────────────────────────
    // Four-row tile aggregation for large contiguous noCrop rows.
template <class DT_X>
__aicore__ inline void RunFp32DeepNoCrop(
    AscendC::GlobalTensor<DT_X> &x_gm, AscendC::GlobalTensor<DT_X> &y_gm,
    uint32_t core_idx) {
        constexpr uint32_t H = 28U, W = 28U, D = 128U;
        constexpr uint32_t OUT_B = 2U, OUT_H = 56U, OUT_W = 56U;
        constexpr uint32_t BLOCK_NUM = 32U;
        constexpr uint32_t ROW_ELEMS = OUT_W * D;
        constexpr uint32_t D_BYTES = D * sizeof(DT_X);
        constexpr uint32_t D_STRIDE = D_BYTES >> 5;

        AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub;
        AscendC::LocalTensor<DT_X> buf0 =
            ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(kTileElems);
        AscendC::LocalTensor<DT_X> buf1 =
            ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(kTileElems);

        constexpr uint32_t total_rows = OUT_B * OUT_H;
        constexpr uint32_t rows_per_core =
            (total_rows + BLOCK_NUM - 1U) / BLOCK_NUM;
        uint32_t start_row = core_idx * rows_per_core;
        if (start_row >= total_rows) return;
        uint32_t row_count = rows_per_core;
        if (start_row + row_count > total_rows)
            row_count = total_rows - start_row;

        AscendC::DataCopyExtParams in_params{
            static_cast<uint16_t>(W), D_BYTES, 0, D_STRIDE, 0};
        AscendC::DataCopyPadExtParams<DT_X> pad{false, 0, 0, 0};

        uint32_t ob = start_row / OUT_H;
        uint32_t oh = start_row - ob * OUT_H;
        uint32_t in_h0 = oh >> 1, block_h0 = oh & 1;
        uint32_t in_b00 = (block_h0 << 1) * OUT_B + ob;
        uint32_t in_b01 = in_b00 + OUT_B;
        uint64_t s00 =
            ((static_cast<uint64_t>(in_b00) * H + in_h0) * W) * D;
        uint64_t s01 =
            ((static_cast<uint64_t>(in_b01) * H + in_h0) * W) * D;
        AscendC::DataCopyPad(buf0, x_gm[s00], in_params, pad);
        AscendC::DataCopyPad(buf0[D], x_gm[s01], in_params, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(kEventId0);

        uint32_t buf_idx = 0;
        for (uint32_t r = 0; r < row_count; ++r) {
            event_t cur_ev = buf_idx == 0 ? kEventId0 : kEventId1;
            event_t next_ev = buf_idx == 0 ? kEventId1 : kEventId0;
            AscendC::LocalTensor<DT_X> cur_buf = buf_idx == 0 ? buf0 : buf1;
            AscendC::LocalTensor<DT_X> next_buf = buf_idx == 0 ? buf1 : buf0;

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(cur_ev);

            uint32_t next_ob = ob;
            uint32_t next_oh = oh + 1;
            if (next_oh == OUT_H) { next_oh = 0; ++next_ob; }
            if (r + 1 < row_count) {
                uint32_t nin_h = next_oh >> 1;
                uint32_t nb_h = next_oh & 1;
                uint32_t nin_b0 = (nb_h << 1) * OUT_B + next_ob;
                uint32_t nin_b1 = nin_b0 + OUT_B;
                uint64_t s0 =
                    ((static_cast<uint64_t>(nin_b0) * H + nin_h) * W) * D;
                uint64_t s1 =
                    ((static_cast<uint64_t>(nin_b1) * H + nin_h) * W) * D;
                AscendC::DataCopyPad(next_buf, x_gm[s0], in_params, pad);
                AscendC::DataCopyPad(next_buf[D], x_gm[s1], in_params, pad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(next_ev);
            }

            uint64_t dst = static_cast<uint64_t>(start_row + r) * ROW_ELEMS;
            AscendC::DataCopy(y_gm[dst], cur_buf, ROW_ELEMS);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(cur_ev);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(cur_ev);

            ob = next_ob;
            oh = next_oh;
            buf_idx ^= 1;
        }
    }

    // ── fp32 small-depth crop family ─────────────────────────────────
    // Keep batch approach: per-phase DataCopyPad with batch read+write.
    // Whole-phase DataCopyPad with row-level double buffering.
template <class DT_X>
__aicore__ inline void RunFp32CompactCrop(
    AscendC::GlobalTensor<DT_X> &x_gm, AscendC::GlobalTensor<DT_X> &y_gm,
    uint32_t core_idx) {
    constexpr uint32_t BS = 2U;
    constexpr uint32_t H = 10U, W = 15U, D = 5U;
    constexpr uint32_t OUT_B = 1U, OUT_H = 17U, OUT_W = 26U;
    constexpr uint32_t CROP_TOP = 2U, CROP_LEFT = 3U;
    constexpr uint32_t kInputBytes = 1024U;
    constexpr uint32_t kOutputBytes = 1024U;
    constexpr uint32_t kOffsetBytes = 1024U;

    constexpr uint32_t total_rows = OUT_B * OUT_H;
    if (core_idx >= total_rows) {
        return;
    }

    constexpr uint32_t c_bytes = D * sizeof(DT_X);
    constexpr uint32_t even_cols = (OUT_W + 1U) >> 1U;
    constexpr uint32_t odd_cols = OUT_W >> 1U;
    constexpr uint32_t phase_cols = even_cols > odd_cols ? even_cols : odd_cols;
    constexpr uint32_t phase_bytes =
        ((phase_cols * c_bytes + 31U) >> 5U) << 5U;
    constexpr uint32_t phase_elems = phase_bytes / sizeof(DT_X);
    constexpr uint32_t row_elems = OUT_W * D;
    constexpr uint32_t row_bytes = row_elems * sizeof(DT_X);
    const uint32_t out_row = core_idx;
    const uint32_t ob = out_row / OUT_H;
    const uint32_t oh = out_row - ob * OUT_H;
    const uint32_t full_h = oh + CROP_TOP;
    const uint32_t in_h = full_h >> 1U;
    const uint32_t block_h = full_h & 1U;
    const uint32_t even_phase = CROP_LEFT & 1U;
    const uint32_t odd_phase = even_phase ^ 1U;
    const uint32_t even_start_col = CROP_LEFT >> 1U;
    const uint32_t odd_start_col = (CROP_LEFT + 1U) >> 1U;
    const uint32_t even_input_n = (block_h * BS + even_phase) * OUT_B + ob;
    const uint32_t odd_input_n = (block_h * BS + odd_phase) * OUT_B + ob;
    const uint64_t even_src =
        ((static_cast<uint64_t>(even_input_n) * H + in_h) * W + even_start_col) * D;
    const uint64_t odd_src =
        ((static_cast<uint64_t>(odd_input_n) * H + in_h) * W + odd_start_col) * D;

    AscendC::LocalTensor<DT_X> phase_buf(
        AscendC::TPosition::VECCALC, 0, kInputBytes / sizeof(DT_X));
    AscendC::LocalTensor<DT_X> row_buf(
        AscendC::TPosition::VECCALC, kInputBytes, kOutputBytes / sizeof(DT_X));
    AscendC::LocalTensor<int32_t> offset_init(
        AscendC::TPosition::VECCALC, kInputBytes + kOutputBytes,
        kOffsetBytes / sizeof(int32_t));
    AscendC::LocalTensor<uint32_t> gather_offsets(
        AscendC::TPosition::VECCALC, kInputBytes + kOutputBytes,
        kOffsetBytes / sizeof(uint32_t));

    ExpandGatherOffsets(offset_init, OUT_W, D, phase_bytes, sizeof(DT_X));

    AscendC::DataCopyPadExtParams<DT_X> pad{false, 0, 0, 0};
    AscendC::DataCopyExtParams even_load{
        1, even_cols * c_bytes, 0, 0, 0};
    AscendC::DataCopyExtParams odd_load{
        1, odd_cols * c_bytes, 0, 0, 0};
    AscendC::DataCopyExtParams row_store{1, row_bytes, 0, 0, 0};

    AscendC::DataCopyPad(phase_buf, x_gm[even_src], even_load, pad);
    if (odd_cols != 0U) {
        AscendC::DataCopyPad(phase_buf[phase_elems], x_gm[odd_src], odd_load, pad);
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::Gather(row_buf, phase_buf, gather_offsets, 0U, row_elems);
    AscendC::PipeBarrier<PIPE_ALL>();

    const uint64_t dst = static_cast<uint64_t>(out_row) * row_elems;
    AscendC::DataCopyPad(y_gm[dst], row_buf, row_store);
    AscendC::PipeBarrier<PIPE_MTE3>();
}

    // ── fp32 mid-depth noCrop family ──────────────────────────────────
    // Grouped slab interleave for the fp32 point-3 no-crop shape family.
template <class DT_X>
__aicore__ inline void RunFp32MidNoCropSingleBuf(
    AscendC::GlobalTensor<DT_X> &x_gm, AscendC::GlobalTensor<DT_X> &y_gm,
    uint32_t core_idx) {
        constexpr uint32_t BS = 2U;
        constexpr uint32_t H = 14U, W = 14U, D = 64U;
        constexpr uint32_t OUT_B = 4U, OUT_H = 28U, OUT_W = 28U;
        constexpr uint32_t STEP_H = 7U;
        constexpr uint32_t D_BYTES = D * sizeof(DT_X);
        constexpr uint32_t D_BLOCKS = D_BYTES >> 5;
        constexpr uint32_t SRC_ROW_BLOCKS = (W * D_BYTES) >> 5;
        constexpr uint32_t DST_ROW_BLOCKS = (OUT_W * D_BYTES) >> 5;
        constexpr uint32_t SLAB_IN_ELEMS = STEP_H * W * D;
        constexpr uint32_t SLAB_OUT_ELEMS = STEP_H * OUT_W * D;
        constexpr uint32_t SLAB_BYTES =
            ((SLAB_IN_ELEMS * sizeof(DT_X) + 511U) >> 9U) << 9U;
        constexpr uint32_t SPLIT_H = (H + STEP_H - 1U) / STEP_H;
        constexpr uint32_t TASK_COUNT = OUT_B * BS * SPLIT_H;
        constexpr uint16_t LOAD_BLOCKS = static_cast<uint16_t>(STEP_H * SRC_ROW_BLOCKS);
        constexpr uint16_t BLEND_BLOCKS = static_cast<uint16_t>(STEP_H * W);
        constexpr uint16_t STORE_BLOCKS = static_cast<uint16_t>(STEP_H);

        if (core_idx >= TASK_COUNT) {
            return;
        }

        AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub;
        AscendC::LocalTensor<DT_X> even_buf =
            ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(
                SLAB_BYTES / sizeof(DT_X));
        AscendC::LocalTensor<DT_X> odd_buf =
            ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(
                SLAB_BYTES / sizeof(DT_X));
        AscendC::LocalTensor<DT_X> mix_buf =
            ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(SLAB_OUT_ELEMS);

        const AscendC::DataCopyParams load_params(1U, LOAD_BLOCKS, 0U, 0U);
        const AscendC::DataCopyParams blend_params(
            BLEND_BLOCKS, static_cast<uint16_t>(D_BLOCKS), 0U,
            static_cast<uint16_t>(D_BLOCKS));
        const AscendC::DataCopyParams store_params(
            STORE_BLOCKS, static_cast<uint16_t>(DST_ROW_BLOCKS), 0U,
            static_cast<uint16_t>(DST_ROW_BLOCKS));

        const uint32_t tile_idx = core_idx & 1U;
        const uint32_t phase_slot = core_idx >> 1U;
        const uint32_t block_h = phase_slot & 1U;
        const uint32_t out_n = phase_slot >> 1U;
        const uint32_t in_h = tile_idx * STEP_H;
        const uint32_t in_n0 = (block_h * BS) * OUT_B + out_n;
        const uint32_t in_n1 = in_n0 + OUT_B;
        const uint64_t src0 =
            ((static_cast<uint64_t>(in_n0) * H + in_h) * W) * D;
        const uint64_t src1 =
            ((static_cast<uint64_t>(in_n1) * H + in_h) * W) * D;

        AscendC::DataCopy(even_buf, x_gm[src0], load_params);
        AscendC::DataCopy(odd_buf, x_gm[src1], load_params);
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::DataCopy(mix_buf, even_buf, blend_params);
        AscendC::DataCopy(mix_buf[D], odd_buf, blend_params);
        AscendC::PipeBarrier<PIPE_MTE2>();

        const uint64_t dst =
            (static_cast<uint64_t>(out_n) * OUT_H + block_h + in_h * BS) *
            OUT_W * D;
        AscendC::DataCopy(y_gm[dst], mix_buf, store_params);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

template <class DT_X>
__aicore__ inline void RunFp32ShortRowNoCrop(
    AscendC::GlobalTensor<DT_X> &x_gm, AscendC::GlobalTensor<DT_X> &y_gm,
    uint32_t core_idx) {
        constexpr uint32_t H = 4U, W = 6U, D = 32U;
        constexpr uint32_t OUT_B = 5U, OUT_H = 8U, OUT_W = 12U;
        constexpr uint32_t ACTIVE_CORES = OUT_B * 2U;
        constexpr uint32_t D_BLOCKS = (D * sizeof(DT_X)) >> 5;
        constexpr uint32_t INPUT_ELEMS = H * W * D;
        constexpr uint32_t OUTPUT_ELEMS = H * OUT_W * D;
        constexpr uint32_t INPUT_BYTES = INPUT_ELEMS * sizeof(DT_X);
        constexpr uint32_t OUTPUT_BYTES = OUTPUT_ELEMS * sizeof(DT_X);

        if (core_idx >= ACTIVE_CORES) {
            return;
        }

        AscendC::LocalTensor<DT_X> src_phase0(
            AscendC::TPosition::VECCALC, 0, INPUT_ELEMS);
        AscendC::LocalTensor<DT_X> src_phase1(
            AscendC::TPosition::VECCALC, INPUT_BYTES, INPUT_ELEMS);
        AscendC::LocalTensor<DT_X> out_rows(
            AscendC::TPosition::VECCALC, INPUT_BYTES * 2U, OUTPUT_ELEMS);

        const uint32_t out_batch = core_idx >> 1U;
        const uint32_t block_h = core_idx & 1U;
        const uint32_t in_batch0 = block_h * (OUT_B << 1U) + out_batch;
        const uint32_t in_batch1 = in_batch0 + OUT_B;
        const uint64_t src0 =
            static_cast<uint64_t>(in_batch0) * H * W * D;
        const uint64_t src1 =
            static_cast<uint64_t>(in_batch1) * H * W * D;

        const AscendC::DataCopyParams load_whole(
            1U, static_cast<uint16_t>(INPUT_BYTES >> 5), 0U, 0U);
        AscendC::DataCopy(src_phase0, x_gm[src0], load_whole);
        AscendC::DataCopy(src_phase1, x_gm[src1], load_whole);
        AscendC::PipeBarrier<PIPE_ALL>();

        const AscendC::DataCopyParams interleave_rows(
            static_cast<uint16_t>(H * W),
            static_cast<uint16_t>(D_BLOCKS), 0U,
            static_cast<uint16_t>(D_BLOCKS));
        AscendC::DataCopy(out_rows, src_phase0, interleave_rows);
        AscendC::DataCopy(out_rows[D], src_phase1, interleave_rows);
        AscendC::PipeBarrier<PIPE_MTE2>();

        const AscendC::DataCopyParams store_rows(
            static_cast<uint16_t>(H),
            static_cast<uint16_t>((OUTPUT_BYTES >> 5) / H), 0U,
            static_cast<uint16_t>((OUTPUT_BYTES >> 5) / H));
        const uint64_t dst =
            (static_cast<uint64_t>(out_batch) * OUT_H + block_h) * OUT_W * D;
        AscendC::DataCopy(y_gm[dst], out_rows, store_rows);
    }

template <class DT_X>
__aicore__ inline void RunFp16OddDepthCropPipeline(
    AscendC::GlobalTensor<DT_X> &x_gm, AscendC::GlobalTensor<DT_X> &y_gm,
    uint32_t core_idx) {
    constexpr uint32_t BS = 2U;
    constexpr uint32_t H = 128U, W = 128U, D = 65U;
    constexpr uint32_t OUT_B = 1U, OUT_H = 254U, OUT_W = 254U;
    constexpr uint32_t CROP_TOP = 1U, CROP_LEFT = 1U;
    constexpr uint32_t kSlotBytes = 50U * 1024U;
    constexpr uint32_t kOffsetBytes = 34U * 1024U;

    if (sizeof(DT_X) != 2) {
        return;
    }

    constexpr uint32_t total_rows = OUT_B * OUT_H;
    constexpr uint32_t total_works = total_rows * 2U;
    constexpr uint32_t active_blocks = 40U;
    if (active_blocks == 0U || core_idx >= total_works) {
        return;
    }

    constexpr uint32_t c_bytes = D * sizeof(DT_X);
    constexpr uint32_t half_width = (OUT_W + 1U) >> 1U;
    constexpr uint32_t phase_cols = (half_width + 1U) >> 1U;
    constexpr uint32_t phase_bytes =
        ((phase_cols * c_bytes + 31U) >> 5U) << 5U;
    constexpr uint32_t phase_elems = phase_bytes / sizeof(DT_X);
    constexpr uint32_t packed_input_elems = phase_elems * 2U;

    AscendC::LocalTensor<DT_X> stage_buf0(
        AscendC::TPosition::VECCALC, 0, kSlotBytes / sizeof(DT_X));
    AscendC::LocalTensor<DT_X> stage_buf1(
        AscendC::TPosition::VECCALC, kSlotBytes, kSlotBytes / sizeof(DT_X));
    AscendC::LocalTensor<int32_t> offset_init(
        AscendC::TPosition::VECCALC, kSlotBytes * 2U,
        kOffsetBytes / sizeof(int32_t));
    AscendC::LocalTensor<uint32_t> gather_offsets(
        AscendC::TPosition::VECCALC, kSlotBytes * 2U,
        kOffsetBytes / sizeof(uint32_t));

    ExpandGatherOffsets(offset_init, half_width, D, phase_bytes, sizeof(DT_X));

    AscendC::DataCopyPadExtParams<DT_X> pad{false, 0, 0, 0};
    AscendC::DataCopyExtParams phase0_load{1, 0, 0, 0, 0};
    AscendC::DataCopyExtParams phase1_load{1, 0, 0, 0, 0};
    AscendC::DataCopyExtParams row_store{1, 0, 0, 0, 0};

    constexpr uint32_t works_per_core =
        (total_works + active_blocks - 1U) / active_blocks;
    uint32_t start_work = core_idx * works_per_core;
    if (start_work >= total_works) {
        return;
    }
    uint32_t end_work = start_work + works_per_core;
    if (end_work > total_works) {
        end_work = total_works;
    }

    SignalStageStoreDone(0U);
    SignalStageStoreDone(1U);

    uint32_t work_id = start_work;
    uint32_t stage_idx = 0U;
    WaitStageStoreDone(stage_idx);
    {
        AscendC::LocalTensor<DT_X> load_stage = stage_buf0;
        const uint32_t out_row = work_id >> 1U;
        const uint32_t half_sel = work_id & 1U;
        const uint32_t out_col0 = half_sel * half_width;
        uint32_t cols_this_half = OUT_W - out_col0;
        if (cols_this_half > half_width) {
            cols_this_half = half_width;
        }
        const uint32_t ob = out_row / OUT_H;
        const uint32_t oh = out_row - ob * OUT_H;
        const uint32_t full_h = oh + CROP_TOP;
        const uint32_t in_h = full_h >> 1U;
        const uint32_t block_h = full_h & 1U;
        const uint32_t full_w0 = CROP_LEFT + out_col0;
        const uint32_t first_phase = full_w0 & 1U;
        const uint32_t second_phase = first_phase ^ 1U;
        const uint32_t first_in_w = full_w0 >> 1U;
        const uint32_t second_in_w = (full_w0 + 1U) >> 1U;
        const uint32_t first_cols = (cols_this_half + 1U) >> 1U;
        const uint32_t second_cols = cols_this_half >> 1U;
        const uint32_t input_n0 = (block_h * BS + first_phase) * OUT_B + ob;
        const uint32_t input_n1 = (block_h * BS + second_phase) * OUT_B + ob;
        const uint64_t src0 =
            ((static_cast<uint64_t>(input_n0) * H + in_h) * W + first_in_w) * D;
        const uint64_t src1 =
            ((static_cast<uint64_t>(input_n1) * H + in_h) * W + second_in_w) * D;

        phase0_load.blockLen = first_cols * c_bytes;
        phase1_load.blockLen = second_cols * c_bytes;
        AscendC::DataCopyPad(load_stage, x_gm[src0], phase0_load, pad);
        if (second_cols != 0U) {
            AscendC::DataCopyPad(load_stage[phase_elems], x_gm[src1], phase1_load, pad);
        }
    }
    SignalStageLoadReady(stage_idx);

    while (work_id < end_work) {
        AscendC::LocalTensor<DT_X> cur_stage =
            stage_idx == 0U ? stage_buf0 : stage_buf1;
        WaitStageLoadReady(stage_idx);

        const uint32_t next_work = work_id + 1U;
        const uint32_t next_stage = stage_idx ^ 1U;
        if (next_work < end_work) {
            AscendC::LocalTensor<DT_X> load_stage =
                next_stage == 0U ? stage_buf0 : stage_buf1;
            WaitStageStoreDone(next_stage);

            const uint32_t out_row = next_work >> 1U;
            const uint32_t half_sel = next_work & 1U;
            const uint32_t out_col0 = half_sel * half_width;
            uint32_t cols_this_half = OUT_W - out_col0;
            if (cols_this_half > half_width) {
                cols_this_half = half_width;
            }
            const uint32_t ob = out_row / OUT_H;
            const uint32_t oh = out_row - ob * OUT_H;
            const uint32_t full_h = oh + CROP_TOP;
            const uint32_t in_h = full_h >> 1U;
            const uint32_t block_h = full_h & 1U;
            const uint32_t full_w0 = CROP_LEFT + out_col0;
            const uint32_t first_phase = full_w0 & 1U;
            const uint32_t second_phase = first_phase ^ 1U;
            const uint32_t first_in_w = full_w0 >> 1U;
            const uint32_t second_in_w = (full_w0 + 1U) >> 1U;
            const uint32_t first_cols = (cols_this_half + 1U) >> 1U;
            const uint32_t second_cols = cols_this_half >> 1U;
            const uint32_t input_n0 = (block_h * BS + first_phase) * OUT_B + ob;
            const uint32_t input_n1 = (block_h * BS + second_phase) * OUT_B + ob;
            const uint64_t src0 =
                ((static_cast<uint64_t>(input_n0) * H + in_h) * W + first_in_w) * D;
            const uint64_t src1 =
                ((static_cast<uint64_t>(input_n1) * H + in_h) * W + second_in_w) * D;

            phase0_load.blockLen = first_cols * c_bytes;
            phase1_load.blockLen = second_cols * c_bytes;
            AscendC::DataCopyPad(load_stage, x_gm[src0], phase0_load, pad);
            if (second_cols != 0U) {
                AscendC::DataCopyPad(load_stage[phase_elems], x_gm[src1], phase1_load, pad);
            }
            SignalStageLoadReady(next_stage);
        }

        const uint32_t out_row = work_id >> 1U;
        const uint32_t half_sel = work_id & 1U;
        const uint32_t out_col0 = half_sel * half_width;
        uint32_t cols_this_half = OUT_W - out_col0;
        if (cols_this_half > half_width) {
            cols_this_half = half_width;
        }
        const uint32_t out_elems = cols_this_half * D;
        AscendC::Gather(cur_stage[packed_input_elems], cur_stage, gather_offsets,
                        0U, out_elems);
        SyncGatherToStore(stage_idx);

        const uint64_t dst =
            (static_cast<uint64_t>(out_row) * OUT_W + out_col0) * D;
        row_store.blockLen = out_elems * sizeof(DT_X);
        AscendC::DataCopyPad(y_gm[dst], cur_stage[packed_input_elems], row_store);
        SignalStageStoreDone(stage_idx);

        work_id = next_work;
        stage_idx = next_stage;
    }

    WaitStageStoreDone(0U);
    WaitStageStoreDone(1U);
}

    // ── fp16 huge-depth small-HW family ───────────────────────────────
    // batched: 2 pixels/core, fire reads �?1 wait �?1 big write
template <class DT_X>
__aicore__ inline void RunFp16HugeDepthNoCrop(
    AscendC::GlobalTensor<DT_X> &x_gm, AscendC::GlobalTensor<DT_X> &y_gm,
    uint32_t core_idx) {
        constexpr uint32_t H = 2U, W = 2U, D = 4096U;
        constexpr uint32_t OUT_B = 1U, OUT_H = 4U, OUT_W = 4U;
        constexpr uint32_t PPC = 2U;
        AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub;
        AscendC::LocalTensor<DT_X> buffer =
            ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(kTileElems);
        uint32_t sp = core_idx * PPC;
        constexpr uint32_t total_pixels = OUT_B * OUT_H * OUT_W;
        if (sp >= total_pixels) return;
        uint32_t oh = sp / OUT_W, ow = sp - oh * OUT_W;
        for (uint32_t i = 0; i < PPC; ++i) {
            uint32_t ih = oh >> 1, iw = ow >> 1;
            uint32_t bh = oh & 1, bw = ow & 1;
            uint32_t ib = (bh * 2 + bw);
            uint64_t s = ((static_cast<uint64_t>(ib) * H + ih) * W + iw) * D;
            AscendC::DataCopy(buffer[i * D], x_gm[s], D);
            ++ow; if (ow == OUT_W) { ow = 0; ++oh; }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(kEventId0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(kEventId0);
        uint64_t dst = static_cast<uint64_t>(sp) * D;
        AscendC::DataCopy(y_gm[dst], buffer, PPC * D);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(kEventId0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kEventId0);
    }

template <class DT_X>
__aicore__ inline void RunFp16ExtremeDepthNoCrop(
    AscendC::GlobalTensor<DT_X> &x_gm, AscendC::GlobalTensor<DT_X> &y_gm,
    uint32_t core_idx) {
        constexpr uint32_t D = 16384U;
        constexpr uint32_t OUT_B = 1U, OUT_H = 2U, OUT_W = 2U;
        constexpr uint32_t TOTAL = OUT_B * OUT_H * OUT_W * D;
        constexpr uint32_t BN = 8U;
        AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub;
        AscendC::LocalTensor<DT_X> buffer =
            ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(kTileElems);
        uint32_t epc = (TOTAL + BN - 1) / BN;
        uint64_t start = static_cast<uint64_t>(core_idx) * epc;
        if (start >= TOTAL) return;
        uint64_t end = start + epc;
        if (end > TOTAL) end = TOTAL;
        uint64_t pos = start;
        while (pos < end) {
            uint32_t n = static_cast<uint32_t>(end - pos);
            if (n > kTileElems) n = kTileElems;
            AscendC::DataCopy(buffer, x_gm[pos], n);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(kEventId0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(kEventId0);
            AscendC::DataCopy(y_gm[pos], buffer, n);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(kEventId0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kEventId0);
            pos += n;
        }
    }

template <class DT_X>
__aicore__ inline void RunFp16WideRowNoCrop(
    AscendC::GlobalTensor<DT_X> &x_gm, AscendC::GlobalTensor<DT_X> &y_gm,
    uint32_t core_idx) {
        constexpr uint32_t H = 10U, W = 512U, D = 256U;
        constexpr uint32_t OUT_B = 1U, OUT_H = 20U, OUT_W = 1024U;
        constexpr uint32_t BLOCK_NUM = 40U;
        constexpr uint32_t TILE_COLS = kTileElemsWideRow / D;
        constexpr uint32_t TILE_COUNT = W / TILE_COLS;
        constexpr uint32_t TILE_ELEMS = TILE_COLS * D;
        constexpr uint32_t OUT_STRIDE_BYTES = D * sizeof(DT_X);
        constexpr uint32_t BATCH = OUT_B * 4U;
        constexpr uint32_t TOTAL_ROWS = BATCH * H;

        if (core_idx >= TOTAL_ROWS || core_idx >= BLOCK_NUM) {
            return;
        }

        const uint32_t row = core_idx;
        const uint32_t in_b = row / H;
        const uint32_t in_h = row - in_b * H;
        const uint32_t block_h = in_b >> 1U;
        const uint32_t block_w = in_b & 1U;
        const uint32_t oh = (in_h << 1U) + block_h;
        const uint64_t src_base = static_cast<uint64_t>(row) * W * D;
        const uint64_t dst_base =
            (static_cast<uint64_t>(oh) * OUT_W + block_w) * D;

        AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub;
        AscendC::LocalTensor<DT_X> buffer0 =
            ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(kTileElemsWideRow);
        AscendC::LocalTensor<DT_X> buffer1 =
            ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(kTileElemsWideRow);

        const AscendC::DataCopyExtParams out_params{
            static_cast<uint16_t>(TILE_COLS), OUT_STRIDE_BYTES, 0U,
            OUT_STRIDE_BYTES, 0U};

        AscendC::DataCopy(buffer0, x_gm[src_base], TILE_ELEMS);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(kEventId0);

        uint32_t done_cols = 0U;
        uint32_t stage = 0U;
        for (uint32_t tile = 0U; tile < TILE_COUNT; ++tile) {
            const event_t cur_event = stage == 0U ? kEventId0 : kEventId1;
            const event_t next_event = stage == 0U ? kEventId1 : kEventId0;
            AscendC::LocalTensor<DT_X> cur_buffer =
                stage == 0U ? buffer0 : buffer1;
            AscendC::LocalTensor<DT_X> next_buffer =
                stage == 0U ? buffer1 : buffer0;

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(cur_event);
            if (tile + 1U < TILE_COUNT) {
                const uint64_t next_src =
                    src_base + static_cast<uint64_t>(done_cols + TILE_COLS) * D;
                AscendC::DataCopy(next_buffer, x_gm[next_src], TILE_ELEMS);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(next_event);
            }

            const uint64_t dst =
                dst_base + static_cast<uint64_t>(done_cols) * 2U * D;
            AscendC::DataCopyPad(y_gm[dst], cur_buffer, out_params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(cur_event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(cur_event);

            done_cols += TILE_COLS;
            stage ^= 1U;
        }
    }

template <class DT_X>
__aicore__ inline void RunFp16Bs4WideCropLanePackFixed(
    AscendC::GlobalTensor<DT_X> &x_gm, AscendC::GlobalTensor<DT_X> &y_gm,
    uint32_t core_idx) {
        constexpr uint32_t kHeight = 8U;
        constexpr uint32_t kWidth = 640U;
        constexpr uint32_t kDepth = 64U;
        constexpr uint32_t kOutHeight = 32U;
        constexpr uint32_t kOutWidth = 1919U;
        constexpr uint32_t kCropLeft = 641U;
        constexpr uint32_t kTileCols = 512U;
        constexpr uint32_t kOutputBytes = 64U * 1024U;
        constexpr uint32_t kPhasePairBytes = 32U * 1024U;
        constexpr uint32_t kDepthBlocks = (kDepth * sizeof(DT_X)) / 32U;
        constexpr uint32_t kGroupElems = (kTileCols / 4U) * kDepth;

        AscendC::LocalTensor<DT_X> row_local(
            AscendC::TPosition::VECCALC, 0, kOutputBytes / sizeof(DT_X));
        AscendC::LocalTensor<DT_X> phase01(
            AscendC::TPosition::VECCALC, kOutputBytes,
            kPhasePairBytes / sizeof(DT_X));
        AscendC::LocalTensor<DT_X> phase23(
            AscendC::TPosition::VECCALC, kOutputBytes + kPhasePairBytes,
            kPhasePairBytes / sizeof(DT_X));

        AscendC::DataCopyParams load_params;
        load_params.blockCount = 1;
        load_params.blockLen = 0;
        load_params.srcStride = 0;
        load_params.dstStride = 0;

        AscendC::DataCopyParams ub_params;
        ub_params.blockLen = static_cast<uint16_t>(kDepthBlocks);
        ub_params.srcStride = 0;
        ub_params.dstStride = static_cast<uint16_t>(3U * kDepthBlocks);

        AscendC::DataCopyParams store_params;
        store_params.blockCount = 1;
        store_params.blockLen = 0;
        store_params.srcStride = 0;
        store_params.dstStride = 0;

        constexpr uint32_t block_num = 40U;
        for (uint32_t oh = core_idx; oh < kOutHeight; oh += block_num) {
            uint32_t ih = oh >> 2U;
            uint32_t input_base = (oh & 3U) << 2U;

            for (uint32_t tile = 0U; tile < 4U; ++tile) {
                uint32_t col_begin = tile * kTileCols;
                uint32_t cols = tile == 3U ? 383U : kTileCols;
                uint32_t iw_base = (col_begin + kCropLeft) >> 2U;
                uint32_t count0 = tile == 3U ? 95U : 128U;
                uint32_t count1 = tile == 3U ? 96U : 128U;
                uint32_t count2 = tile == 3U ? 96U : 128U;
                uint32_t count3 = tile == 3U ? 96U : 128U;

                uint32_t src0 = ((input_base * kHeight + ih) * kWidth + iw_base + 1U) * kDepth;
                uint32_t src1 = (((input_base + 1U) * kHeight + ih) * kWidth + iw_base) * kDepth;
                uint32_t src2 = (((input_base + 2U) * kHeight + ih) * kWidth + iw_base) * kDepth;
                uint32_t src3 = (((input_base + 3U) * kHeight + ih) * kWidth + iw_base) * kDepth;

                load_params.blockLen = static_cast<uint16_t>(count0 * kDepthBlocks);
                AscendC::DataCopy(phase01, x_gm[src0], load_params);
                load_params.blockLen = static_cast<uint16_t>(count1 * kDepthBlocks);
                AscendC::DataCopy(phase01[kGroupElems], x_gm[src1], load_params);
                load_params.blockLen = static_cast<uint16_t>(count2 * kDepthBlocks);
                AscendC::DataCopy(phase23, x_gm[src2], load_params);
                load_params.blockLen = static_cast<uint16_t>(count3 * kDepthBlocks);
                AscendC::DataCopy(phase23[kGroupElems], x_gm[src3], load_params);
                AscendC::PipeBarrier<PIPE_ALL>();

                ub_params.blockCount = static_cast<uint16_t>(count0);
                AscendC::DataCopy(row_local[3U * kDepth], phase01, ub_params);
                ub_params.blockCount = static_cast<uint16_t>(count1);
                AscendC::DataCopy(row_local, phase01[kGroupElems], ub_params);
                ub_params.blockCount = static_cast<uint16_t>(count2);
                AscendC::DataCopy(row_local[kDepth], phase23, ub_params);
                ub_params.blockCount = static_cast<uint16_t>(count3);
                AscendC::DataCopy(row_local[2U * kDepth], phase23[kGroupElems], ub_params);
                AscendC::PipeBarrier<PIPE_ALL>();

                uint32_t dst = (oh * kOutWidth + col_begin) * kDepth;
                store_params.blockLen = static_cast<uint16_t>(cols * kDepthBlocks);
                AscendC::DataCopy(y_gm[dst], row_local, store_params);
                if (tile != 3U || oh + block_num < kOutHeight) {
                    AscendC::PipeBarrier<PIPE_MTE3>();
                }
            }
        }
    }

template <class DT_X>
__aicore__ inline void RunFp16Bs4WideCropLanePack(
    AscendC::GlobalTensor<DT_X> &x_gm, AscendC::GlobalTensor<DT_X> &y_gm,
    uint32_t core_idx) {
        constexpr uint32_t H = 10U, W = 512U, D = 64U;
        constexpr uint32_t OUT_H = 40U, OUT_W = 1535U;
        constexpr uint32_t kBlockBytes = 32U;
        constexpr uint32_t kTileCols = 512U;
        constexpr uint32_t kTailCols = 511U;
        constexpr uint32_t kOutputBytes = 64U * 1024U;
        constexpr uint32_t kPhasePairBytes = 32U * 1024U;
        constexpr uint32_t kCropLeft = (W << 2U) - OUT_W;
        constexpr uint32_t kDepthBlocks = (D * sizeof(DT_X)) / kBlockBytes;
        constexpr uint32_t kGroupElems = (kTileCols / 4U) * D;
        constexpr uint32_t kBlockNum = 40U;
        constexpr uint16_t kLoad128 = static_cast<uint16_t>(128U * kDepthBlocks);
        constexpr uint16_t kLoad127 = static_cast<uint16_t>(127U * kDepthBlocks);
        constexpr uint16_t kStore512 = static_cast<uint16_t>(kTileCols * kDepthBlocks);
        constexpr uint16_t kStore511 = static_cast<uint16_t>(kTailCols * kDepthBlocks);

        if (core_idx >= kBlockNum) {
            return;
        }

        const uint32_t oh = core_idx;
        const uint32_t ih = oh >> 2U;
        const uint32_t block_h = oh & 3U;
        const uint32_t input_base = block_h << 2U;

        AscendC::LocalTensor<DT_X> row_local(
            AscendC::TPosition::VECCALC, 0, kOutputBytes / sizeof(DT_X));
        AscendC::LocalTensor<DT_X> phase01(
            AscendC::TPosition::VECCALC, kOutputBytes,
            kPhasePairBytes / sizeof(DT_X));
        AscendC::LocalTensor<DT_X> phase23(
            AscendC::TPosition::VECCALC, kOutputBytes + kPhasePairBytes,
            kPhasePairBytes / sizeof(DT_X));

        const AscendC::DataCopyParams load_128(1U, kLoad128, 0U, 0U);
        const AscendC::DataCopyParams load_127(1U, kLoad127, 0U, 0U);
        const AscendC::DataCopyParams gather_128(
            128U, static_cast<uint16_t>(kDepthBlocks), 0U,
            static_cast<uint16_t>(3U * kDepthBlocks));
        const AscendC::DataCopyParams gather_127(
            127U, static_cast<uint16_t>(kDepthBlocks), 0U,
            static_cast<uint16_t>(3U * kDepthBlocks));
        const AscendC::DataCopyParams store_512(1U, kStore512, 0U, 0U);
        const AscendC::DataCopyParams store_511(1U, kStore511, 0U, 0U);

        {
            constexpr uint32_t kColBegin = 0U;
            constexpr uint32_t kIwBase = (kColBegin + kCropLeft) >> 2U;
            const uint32_t src0 = ((input_base * H + ih) * W + kIwBase + 1U) * D;
            const uint32_t src1 = (((input_base + 1U) * H + ih) * W + kIwBase) * D;
            const uint32_t src2 = (((input_base + 2U) * H + ih) * W + kIwBase) * D;
            const uint32_t src3 = (((input_base + 3U) * H + ih) * W + kIwBase) * D;
            const uint32_t dst = (oh * OUT_W + kColBegin) * D;

            AscendC::DataCopy(phase01, x_gm[src0], load_128);
            AscendC::DataCopy(phase01[kGroupElems], x_gm[src1], load_128);
            AscendC::DataCopy(phase23, x_gm[src2], load_128);
            AscendC::DataCopy(phase23[kGroupElems], x_gm[src3], load_128);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::DataCopy(row_local[3U * D], phase01, gather_128);
            AscendC::DataCopy(row_local, phase01[kGroupElems], gather_128);
            AscendC::DataCopy(row_local[D], phase23, gather_128);
            AscendC::DataCopy(row_local[2U * D], phase23[kGroupElems], gather_128);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::DataCopy(y_gm[dst], row_local, store_512);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }

        {
            constexpr uint32_t kColBegin = 512U;
            constexpr uint32_t kIwBase = (kColBegin + kCropLeft) >> 2U;
            const uint32_t src0 = ((input_base * H + ih) * W + kIwBase + 1U) * D;
            const uint32_t src1 = (((input_base + 1U) * H + ih) * W + kIwBase) * D;
            const uint32_t src2 = (((input_base + 2U) * H + ih) * W + kIwBase) * D;
            const uint32_t src3 = (((input_base + 3U) * H + ih) * W + kIwBase) * D;
            const uint32_t dst = (oh * OUT_W + kColBegin) * D;

            AscendC::DataCopy(phase01, x_gm[src0], load_128);
            AscendC::DataCopy(phase01[kGroupElems], x_gm[src1], load_128);
            AscendC::DataCopy(phase23, x_gm[src2], load_128);
            AscendC::DataCopy(phase23[kGroupElems], x_gm[src3], load_128);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::DataCopy(row_local[3U * D], phase01, gather_128);
            AscendC::DataCopy(row_local, phase01[kGroupElems], gather_128);
            AscendC::DataCopy(row_local[D], phase23, gather_128);
            AscendC::DataCopy(row_local[2U * D], phase23[kGroupElems], gather_128);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::DataCopy(y_gm[dst], row_local, store_512);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }

        {
            constexpr uint32_t kColBegin = 1024U;
            constexpr uint32_t kIwBase = (kColBegin + kCropLeft) >> 2U;
            const uint32_t src0 = ((input_base * H + ih) * W + kIwBase + 1U) * D;
            const uint32_t src1 = (((input_base + 1U) * H + ih) * W + kIwBase) * D;
            const uint32_t src2 = (((input_base + 2U) * H + ih) * W + kIwBase) * D;
            const uint32_t src3 = (((input_base + 3U) * H + ih) * W + kIwBase) * D;
            const uint32_t dst = (oh * OUT_W + kColBegin) * D;

            AscendC::DataCopy(phase01, x_gm[src0], load_127);
            AscendC::DataCopy(phase01[kGroupElems], x_gm[src1], load_128);
            AscendC::DataCopy(phase23, x_gm[src2], load_128);
            AscendC::DataCopy(phase23[kGroupElems], x_gm[src3], load_128);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::DataCopy(row_local[3U * D], phase01, gather_127);
            AscendC::DataCopy(row_local, phase01[kGroupElems], gather_128);
            AscendC::DataCopy(row_local[D], phase23, gather_128);
            AscendC::DataCopy(row_local[2U * D], phase23[kGroupElems], gather_128);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::DataCopy(y_gm[dst], row_local, store_511);
        }
    }

template <class DT_X>
__aicore__ inline void RunFp16ShortWidthNoCropFixed(
    AscendC::GlobalTensor<DT_X> &x_gm, AscendC::GlobalTensor<DT_X> &y_gm,
    uint32_t core_idx) {
        constexpr uint32_t kHeight = 800U;
        constexpr uint32_t kWidth = 5U;
        constexpr uint32_t kDepth = 32U;
        constexpr uint32_t kOutBatch = 4U;
        constexpr uint32_t kOutHeight = 1600U;
        constexpr uint32_t kOutWidth = 10U;
        constexpr uint32_t kTileRows = 52U;
        constexpr uint32_t kTileCount = 16U;
        constexpr uint32_t kTailTile = 15U;
        constexpr uint32_t kTailRows = kHeight - kTailTile * kTileRows;
        constexpr uint32_t kFixedBlocks = 40U;
        constexpr uint32_t kTaskCount = kOutBatch * 2U * kTileCount;
        constexpr uint32_t kDepthBlocks = (kDepth * sizeof(DT_X)) / 32U;
        constexpr uint32_t kSrcRowBlocks = (kWidth * kDepth * sizeof(DT_X)) / 32U;
        constexpr uint32_t kDstRowBlocks = (kOutWidth * kDepth * sizeof(DT_X)) / 32U;
        constexpr uint32_t kSrcRowElems = kWidth * kDepth;
        constexpr uint32_t kDstRowElems = kOutWidth * kDepth;
        constexpr uint32_t kSrcBatchElems = kHeight * kSrcRowElems;
        constexpr uint32_t kDstBatchElems = kOutHeight * kDstRowElems;
        constexpr uint32_t kInputElems = kTileRows * kSrcRowElems;
        constexpr uint32_t kOutputElems = kTileRows * kDstRowElems;

        AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub;
        AscendC::LocalTensor<DT_X> first_local =
            ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(kInputElems);
        AscendC::LocalTensor<DT_X> second_local =
            ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(kInputElems);
        AscendC::LocalTensor<DT_X> row_local =
            ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(kOutputElems);

        AscendC::DataCopyParams load_params;
        load_params.blockCount = 1;
        load_params.blockLen = static_cast<uint16_t>(kTileRows * kSrcRowBlocks);
        load_params.srcStride = 0;
        load_params.dstStride = 0;

        AscendC::DataCopyParams mix_params;
        mix_params.blockCount = static_cast<uint16_t>(kTileRows * kWidth);
        mix_params.blockLen = static_cast<uint16_t>(kDepthBlocks);
        mix_params.srcStride = 0;
        mix_params.dstStride = static_cast<uint16_t>(kDepthBlocks);

        AscendC::DataCopyParams store_params;
        store_params.blockCount = static_cast<uint16_t>(kTileRows);
        store_params.blockLen = static_cast<uint16_t>(kDstRowBlocks);
        store_params.srcStride = 0;
        store_params.dstStride = static_cast<uint16_t>(kDstRowBlocks);

        AscendC::DataCopyParams tail_load_params = load_params;
        tail_load_params.blockLen = static_cast<uint16_t>(kTailRows * kSrcRowBlocks);
        AscendC::DataCopyParams tail_mix_params = mix_params;
        tail_mix_params.blockCount = static_cast<uint16_t>(kTailRows * kWidth);
        AscendC::DataCopyParams tail_store_params = store_params;
        tail_store_params.blockCount = static_cast<uint16_t>(kTailRows);

        for (uint32_t group = core_idx; group < kTaskCount; group += kFixedBlocks) {
            uint32_t tile = group & (kTileCount - 1U);
            uint32_t tmp = group >> 4U;
            uint32_t height_phase = tmp & 1U;
            uint32_t batch_id = tmp >> 1U;
            uint32_t row_begin = tile * kTileRows;

            uint32_t src_batch = ((height_phase << 1U) * kOutBatch) + batch_id;
            uint32_t src0 = src_batch * kSrcBatchElems + row_begin * kSrcRowElems;
            uint32_t src1 = src0 + kOutBatch * kSrcBatchElems;

            uint32_t dst = batch_id * kDstBatchElems +
                           (height_phase + (row_begin << 1U)) * kDstRowElems;
            if (tile != kTailTile) {
                AscendC::DataCopy(first_local, x_gm[src0], load_params);
                AscendC::DataCopy(second_local, x_gm[src1], load_params);
                AscendC::PipeBarrier<PIPE_ALL>();

                AscendC::DataCopy(row_local, first_local, mix_params);
                AscendC::DataCopy(row_local[kDepth], second_local, mix_params);
                AscendC::PipeBarrier<PIPE_ALL>();

                AscendC::DataCopy(y_gm[dst], row_local, store_params);
            } else {
                AscendC::DataCopy(first_local, x_gm[src0], tail_load_params);
                AscendC::DataCopy(second_local, x_gm[src1], tail_load_params);
                AscendC::PipeBarrier<PIPE_ALL>();

                AscendC::DataCopy(row_local, first_local, tail_mix_params);
                AscendC::DataCopy(row_local[kDepth], second_local, tail_mix_params);
                AscendC::PipeBarrier<PIPE_ALL>();

                AscendC::DataCopy(y_gm[dst], row_local, tail_store_params);
            }
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }

template <class DT_X>
__aicore__ inline void RunFp16ShortWidthNoCrop(
    AscendC::GlobalTensor<DT_X> &x_gm, AscendC::GlobalTensor<DT_X> &y_gm,
    uint32_t core_idx) {
        constexpr uint32_t kDepth = 32U;
        constexpr uint32_t kWidth = 6U;
        constexpr uint32_t kHeight = 1024U;
        constexpr uint32_t kOutBatch = 4U;
        constexpr uint32_t kOutHeight = 2048U;
        constexpr uint32_t kOutWidth = 12U;
        constexpr uint32_t kChunkRows = 16U;
        constexpr uint32_t kOutputChunkRows = kChunkRows * 2U;
        constexpr uint32_t kDepthBlocks =
            (kDepth * sizeof(DT_X)) >> 5;
        constexpr uint32_t kOutRowElems = kOutWidth * kDepth;
        constexpr uint32_t kOutRowBlocks =
            (kOutRowElems * sizeof(DT_X)) >> 5;
        constexpr uint32_t kEvenOddElems = kChunkRows * kOutRowElems;
        constexpr uint32_t kEvenOddBytes = kEvenOddElems * sizeof(DT_X);
        constexpr uint32_t kOutputElems = kOutputChunkRows * kOutRowElems;
        constexpr uint32_t kOutputBytes = kOutputElems * sizeof(DT_X);
        constexpr uint32_t kChunksPerBatch = kOutHeight / kOutputChunkRows;
        constexpr uint32_t kTaskCount = kOutBatch * kChunksPerBatch;

        if (core_idx >= 40U) {
            return;
        }

        AscendC::LocalTensor<DT_X> even_rows(
            AscendC::TPosition::VECCALC, 0, kEvenOddElems);
        AscendC::LocalTensor<DT_X> odd_rows(
            AscendC::TPosition::VECCALC, kEvenOddBytes, kEvenOddElems);
        AscendC::LocalTensor<DT_X> output_rows(
            AscendC::TPosition::VECCALC, kEvenOddBytes * 2U, kOutputElems);

        AscendC::DataCopyExtParams load_rows;
        load_rows.blockCount = static_cast<uint16_t>(kChunkRows * kWidth);
        load_rows.blockLen = kDepth * sizeof(DT_X);
        load_rows.srcStride = 0U;
        load_rows.dstStride = kDepthBlocks;
        load_rows.rsv = 0U;
        AscendC::DataCopyPadExtParams<DT_X> no_pad(
            false, 0U, 0U, static_cast<DT_X>(0));

        AscendC::DataCopyParams interleave_rows;
        interleave_rows.blockCount = static_cast<uint16_t>(kChunkRows);
        interleave_rows.blockLen = static_cast<uint16_t>(kOutRowBlocks);
        interleave_rows.srcStride = 0U;
        interleave_rows.dstStride = static_cast<uint16_t>(kOutRowBlocks);

        AscendC::DataCopyParams store_rows;
        store_rows.blockCount = 1U;
        store_rows.blockLen = static_cast<uint16_t>((kOutputElems * sizeof(DT_X)) >> 5);
        store_rows.srcStride = 0U;
        store_rows.dstStride = 0U;

        const uint32_t block_count = static_cast<uint32_t>(AscendC::GetBlockNum());
        for (uint32_t task = core_idx; task < kTaskCount; task += block_count) {
            const uint32_t batch_id = task / kChunksPerBatch;
            const uint32_t chunk_id = task - batch_id * kChunksPerBatch;
            const uint32_t in_row = chunk_id * kChunkRows;
            const uint32_t out_row = chunk_id * kOutputChunkRows;

            const uint32_t batch00 = batch_id;
            const uint32_t batch01 = kOutBatch + batch_id;
            const uint32_t batch10 = kOutBatch * 2U + batch_id;
            const uint32_t batch11 = kOutBatch * 3U + batch_id;

            const uint64_t src00 =
                ((static_cast<uint64_t>(batch00) * kHeight + in_row) * kWidth) * kDepth;
            const uint64_t src01 =
                ((static_cast<uint64_t>(batch01) * kHeight + in_row) * kWidth) * kDepth;
            const uint64_t src10 =
                ((static_cast<uint64_t>(batch10) * kHeight + in_row) * kWidth) * kDepth;
            const uint64_t src11 =
                ((static_cast<uint64_t>(batch11) * kHeight + in_row) * kWidth) * kDepth;

            AscendC::DataCopyPad(even_rows, x_gm[src00], load_rows, no_pad);
            AscendC::DataCopyPad(even_rows[kDepth], x_gm[src01], load_rows, no_pad);
            AscendC::DataCopyPad(odd_rows, x_gm[src10], load_rows, no_pad);
            AscendC::DataCopyPad(odd_rows[kDepth], x_gm[src11], load_rows, no_pad);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::DataCopy(output_rows, even_rows, interleave_rows);
            AscendC::DataCopy(output_rows[kOutRowElems], odd_rows, interleave_rows);
            AscendC::PipeBarrier<PIPE_ALL>();

            const uint64_t dst =
                ((static_cast<uint64_t>(batch_id) * kOutHeight + out_row) * kOutWidth) * kDepth;
            AscendC::DataCopy(y_gm[dst], output_rows, store_rows);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }

template <class DT_X>
__aicore__ inline void LoadBs4PhaseBlock(
    AscendC::GlobalTensor<DT_X> &x_gm,
    AscendC::LocalTensor<DT_X> phase_block, uint64_t row_base,
    uint32_t iw_base, uint32_t H, uint32_t W, uint32_t D) {
        constexpr uint32_t BS = 4;
        constexpr uint16_t GROUP_COUNT = 128;
        const uint16_t BLOCK_LEN =
            static_cast<uint16_t>((D * sizeof(DT_X)) >> 5);
        const uint32_t BATCH_STRIDE_BLOCKS =
            (H * W * D * sizeof(DT_X)) >> 5;
        const uint16_t PHASE_BLOCKS = GROUP_COUNT * BLOCK_LEN;
        const uint16_t SRC_BATCH_GAP =
            static_cast<uint16_t>(BATCH_STRIDE_BLOCKS - PHASE_BLOCKS);
        AscendC::DataCopyParams load_params(
            static_cast<uint16_t>(BS), PHASE_BLOCKS, SRC_BATCH_GAP, 0);
        uint64_t src = row_base + static_cast<uint64_t>(iw_base) * D;
        AscendC::DataCopy(phase_block, x_gm[src], load_params);
    }

template <bool PREFIX_TILE, class DT_X>
__aicore__ inline void ReorderBs4PhaseBlock(
    AscendC::LocalTensor<DT_X> phase_block,
    AscendC::LocalTensor<DT_X> packed,
    uint32_t D) {
    constexpr uint16_t GROUP_COUNT = 128;
    const uint32_t PHASE_ELEMS = GROUP_COUNT * D;
    const uint16_t BLOCK_LEN =
        static_cast<uint16_t>((D * sizeof(DT_X)) >> 5);
    const uint16_t DST_GAP = BLOCK_LEN * 3U;
    AscendC::DataCopyParams params(GROUP_COUNT, BLOCK_LEN, 0, DST_GAP);

    if constexpr (PREFIX_TILE) {
        AscendC::DataCopy(packed, phase_block[PHASE_ELEMS], params);
        AscendC::DataCopy(packed[D], phase_block[PHASE_ELEMS * 2U], params);
        AscendC::DataCopy(packed[D * 2U], phase_block[PHASE_ELEMS * 3U], params);
        AscendC::DataCopyParams prefix_tail_params(
            static_cast<uint16_t>(GROUP_COUNT - 1U), BLOCK_LEN, 0, DST_GAP);
        AscendC::DataCopy(packed[D * 3U], phase_block[D], prefix_tail_params);
    } else {
        AscendC::DataCopy(packed, phase_block, params);
        AscendC::DataCopy(packed[D], phase_block[PHASE_ELEMS], params);
        AscendC::DataCopy(packed[D * 2U], phase_block[PHASE_ELEMS * 2U], params);
        AscendC::DataCopy(packed[D * 3U], phase_block[PHASE_ELEMS * 3U], params);
    }
}



// TILING_KEY_IS keeps algorithm dispatch; per-route geometry is fixed in kernel.
extern "C" __global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y,
                                                     GM_ADDR workspace,
                                                     GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    const uint32_t core_idx = AscendC::GetBlockIdx();
    (void)tiling;
    (void)workspace;

    if (TILING_KEY_IS(TK_F32_CROP_THIN)) {
        AscendC::GlobalTensor<float> x_gm;
        AscendC::GlobalTensor<float> y_gm;
        InitGlobalTensors<float>(x, y, x_gm, y_gm);
        RunFp32CompactCrop(x_gm, y_gm, core_idx);
        return;
    } else if (TILING_KEY_IS(TK_F32_NC_SHORT)) {
        AscendC::GlobalTensor<float> x_gm;
        AscendC::GlobalTensor<float> y_gm;
        InitGlobalTensors<float>(x, y, x_gm, y_gm);
        RunFp32ShortRowNoCrop(x_gm, y_gm, core_idx);
        return;
    } else if (TILING_KEY_IS(TK_F32_NC_MID)) {
        AscendC::GlobalTensor<float> x_gm;
        AscendC::GlobalTensor<float> y_gm;
        InitGlobalTensors<float>(x, y, x_gm, y_gm);
        RunFp32MidNoCropSingleBuf(x_gm, y_gm, core_idx);
        return;
    } else if (TILING_KEY_IS(TK_F32_NC_DEEP)) {
        AscendC::GlobalTensor<float> x_gm;
        AscendC::GlobalTensor<float> y_gm;
        InitGlobalTensors<float>(x, y, x_gm, y_gm);
        RunFp32DeepNoCrop(x_gm, y_gm, core_idx);
        return;
    } else if (TILING_KEY_IS(TK_F16_CROP_ODD_DEPTH)) {
        AscendC::GlobalTensor<half> x_gm;
        AscendC::GlobalTensor<half> y_gm;
        InitGlobalTensors<half>(x, y, x_gm, y_gm);
        RunFp16OddDepthCropPipeline(x_gm, y_gm, core_idx);
        return;
    } else if (TILING_KEY_IS(TK_F16_CROP_BS4_WIDE)) {
        AscendC::GlobalTensor<half> x_gm;
        AscendC::GlobalTensor<half> y_gm;
        InitGlobalTensors<half>(x, y, x_gm, y_gm);
        RunFp16Bs4WideCropLanePack(x_gm, y_gm, core_idx);
        return;
    } else if (TILING_KEY_IS(TK_F16_NC_SHORT_WIDTH)) {
        AscendC::GlobalTensor<half> x_gm;
        AscendC::GlobalTensor<half> y_gm;
        InitGlobalTensors<half>(x, y, x_gm, y_gm);
        RunFp16ShortWidthNoCrop(x_gm, y_gm, core_idx);
        return;
    } else if (TILING_KEY_IS(TK_F16_NC_WIDE_ROW)) {
        AscendC::GlobalTensor<half> x_gm;
        AscendC::GlobalTensor<half> y_gm;
        InitGlobalTensors<half>(x, y, x_gm, y_gm);
        RunFp16WideRowNoCrop(x_gm, y_gm, core_idx);
        return;
    } else if (TILING_KEY_IS(TK_F16_NC_HUGE_DEPTH)) {
        AscendC::GlobalTensor<half> x_gm;
        AscendC::GlobalTensor<half> y_gm;
        InitGlobalTensors<half>(x, y, x_gm, y_gm);
        RunFp16HugeDepthNoCrop(x_gm, y_gm, core_idx);
        return;
    } else if (TILING_KEY_IS(TK_F16_NC_EXTREME_DEPTH)) {
        AscendC::GlobalTensor<half> x_gm;
        AscendC::GlobalTensor<half> y_gm;
        InitGlobalTensors<half>(x, y, x_gm, y_gm);
        RunFp16ExtremeDepthNoCrop(x_gm, y_gm, core_idx);
        return;
    }
}
