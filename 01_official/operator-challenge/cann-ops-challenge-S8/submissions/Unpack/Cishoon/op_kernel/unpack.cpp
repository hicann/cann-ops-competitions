#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
using namespace AscendC;

#define SCAST static_cast

template <class T> struct block { constexpr static int32_t value = 32 / sizeof(T); };
template <class T> constexpr static int32_t block_n = block<T>::value;

template <class T>
class KernelUnpack {
public:
    __aicore__ inline KernelUnpack() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                uint32_t M, uint32_t K, uint32_t L,
                                uint32_t tile_length, uint32_t iter_per_row,
                                uint32_t real_residue, uint32_t pad_residue,
                                uint32_t iterations,
                                uint32_t pack_mode, uint32_t pack_tile_m) {
        this->M            = M;
        this->K            = K;
        this->L            = L;
        this->tile_length  = tile_length;
        this->iter_per_row = iter_per_row;
        this->real_residue = real_residue;
        this->pad_residue  = pad_residue;
        this->iterations   = iterations;
        this->pack_mode    = pack_mode;
        this->pack_tile_m  = pack_tile_m;

        uint64_t total_in = SCAST<uint64_t>(M) * SCAST<uint64_t>(K) * SCAST<uint64_t>(L);
        input_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input), total_in);

        this->output_desc = output;
    }

    __aicore__ inline void Process() {
        if (pack_mode == 2)
            ProcessPackGather();
        else if (pack_mode == 1)
            ProcessPackStrided();
        else
            ProcessScatter();
    }

private:
    // ── Pack path: gather (l_bytes < 32) ─────────────────────────────────────
    // Strided DMA-in hits HBM bad path when blockLen < 32 B: each block touches
    // a 32-B cache line while using only a fraction of it. Replace with:
    //   (a) one contiguous DMA-in of the whole [tile_m × K × L] slab
    //   (b) K on-chip Gather ops, one per output column
    //   (c) K contiguous DMA-outs, one per output tensor
    // HBM util goes to ~100%. DMA count per tile = 1 in + K out; no strided
    // reads. offset_table is built once (S pipe) and shared across all tiles.
    __aicore__ inline void ProcessPackGather() {
        uint32_t ts = SCAST<uint32_t>(sizeof(T));
        uint32_t tm = pack_tile_m;

        uint32_t m_begin = M * SCAST<uint32_t>(GetBlockIdx())       / SCAST<uint32_t>(GetBlockNum());
        uint32_t m_end   = M * (SCAST<uint32_t>(GetBlockIdx()) + 1) / SCAST<uint32_t>(GetBlockNum());
        if (m_begin >= m_end) return;

        // UB layout: [ offset_table | ub_packed | ub_col ] — each 32-B aligned.
        // ub_col holds `count = tm*L` contiguous elements written by Gather; no
        // per-row padding needed since all rows are produced in one Gather call.
        uint32_t offset_elems = tm * L;
        uint32_t offset_bytes = ((offset_elems * 4 + 31) / 32) * 32;
        uint32_t packed_elems = tm * K * L;
        uint32_t packed_bytes = ((packed_elems * ts + 31) / 32) * 32;
        uint32_t col_elems    = tm * L;

        LocalTensor<uint32_t> offset_table(TPosition::VECCALC, 0, offset_elems);
        LocalTensor<T>        ub_packed(TPosition::VECIN,  offset_bytes, packed_elems);
        LocalTensor<T>        ub_col(TPosition::VECOUT,    offset_bytes + packed_bytes, col_elems);

        // offset_table[i*L + j] = i*K*L*ts + j*ts — byte offset into ub_packed
        // for element [i, 0, j]. Gather uses offset_base = k*L*ts to select column k.
        uint32_t row_stride_bytes = K * L * ts;
        for (uint32_t i = 0; i < tm; i++) {
            uint32_t row_off = i * row_stride_bytes;
            for (uint32_t j = 0; j < L; j++) {
                offset_table.SetValue(i * L + j, row_off + j * ts);
            }
        }
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);

        ListTensorDesc output_list(reinterpret_cast<__gm__ void*>(output_desc));

        bool pending_mte3_mte2 = false;
        bool pending_mte3_v    = false;  // ub_col was last written by DMA-out

        for (uint32_t m_base = m_begin; m_base < m_end; m_base += tm) {
            uint32_t this_tm = (m_base + tm <= m_end) ? tm : (m_end - m_base);
            uint32_t count   = this_tm * L;

            if (pending_mte3_mte2) {
                WaitFlag<HardEvent::MTE3_MTE2>(0);
                pending_mte3_mte2 = false;
            }

            // Contiguous DMA-in: blockLen = this_tm*K*L*ts bytes, 1 block.
            uint64_t in_off   = SCAST<uint64_t>(m_base) * K * L;
            uint32_t in_bytes = this_tm * K * L * ts;
            DataCopyExtParams       cp_in{1, in_bytes, 0, 0, 0};
            DataCopyPadExtParams<T> pp_in{false, 0, 0, 0};
            DataCopyPad(ub_packed, input_global[in_off], cp_in, pp_in);

            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);

            bool more_tile = (m_base + tm < m_end);

            for (uint32_t k = 0; k < K; k++) {
                if (pending_mte3_v) {
                    WaitFlag<HardEvent::MTE3_V>(0);
                    pending_mte3_v = false;
                }

                uint32_t col_base_bytes = k * L * ts;
                Gather(ub_col, ub_packed, offset_table, col_base_bytes, count);

                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);

                __gm__ T*       out_raw = output_list.template GetDataPtr<T>(k);
                GlobalTensor<T> out_k;
                out_k.SetGlobalBuffer(out_raw, SCAST<uint64_t>(M) * L);

                uint32_t          out_bytes = count * ts;
                DataCopyExtParams cp_out{1, out_bytes, 0, 0, 0};
                DataCopyPad(out_k[SCAST<uint64_t>(m_base) * L], ub_col, cp_out);

                if (k + 1 < K || more_tile) {
                    SetFlag<HardEvent::MTE3_V>(0);
                    pending_mte3_v = true;
                }
            }

            if (more_tile) {
                SetFlag<HardEvent::MTE3_MTE2>(0);
                pending_mte3_mte2 = true;
            }
        }

        if (pending_mte3_v) {
            WaitFlag<HardEvent::MTE3_V>(0);
        }
    }

    // ── Pack path: strided (l_bytes >= 32) ───────────────────────────────────
    // Triggered when L*ts >= 32 B. Each core owns a disjoint m-slice [m_begin, m_end).
    // Inner loop over k issues one strided DMA-in (column k of [M,K,L]) and one
    // DMA-out per (tile, k). No Gather: input is already laid out [m,k,l] in GM.
    //
    // UB row pitch = padded_L*ts, always a 32-B multiple, so every row offset is
    // 32-B aligned (required by DataCopyPad src).
    __aicore__ inline void ProcessPackStrided() {
        constexpr uint32_t BN = SCAST<uint32_t>(block_n<T>);
        uint32_t type_size    = SCAST<uint32_t>(sizeof(T));
        uint32_t padded_L     = ((L + BN - 1) / BN) * BN;
        uint8_t  right_pad    = SCAST<uint8_t>(padded_L - L);

        uint32_t m_begin = M * SCAST<uint32_t>(GetBlockIdx())       / SCAST<uint32_t>(GetBlockNum());
        uint32_t m_end   = M * (SCAST<uint32_t>(GetBlockIdx()) + 1) / SCAST<uint32_t>(GetBlockNum());
        if (m_begin >= m_end) return;

        LocalTensor<T> ub_buf(TPosition::VECIN, 0, pack_tile_m * padded_L);

        ListTensorDesc output_list(reinterpret_cast<__gm__ void*>(output_desc));

        bool pending_mte3_mte2 = false;

        for (uint32_t m_base = m_begin; m_base < m_end; m_base += pack_tile_m) {
            uint32_t this_tile = (m_base + pack_tile_m <= m_end) ? pack_tile_m : (m_end - m_base);
            uint64_t in_base   = SCAST<uint64_t>(m_base) * K * L;

            for (uint32_t k = 0; k < K; k++) {
                if (pending_mte3_mte2) {
                    WaitFlag<HardEvent::MTE3_MTE2>(0);
                    pending_mte3_mte2 = false;
                }

                // DMA-in: stride across GM to pick column k, one block per m.
                DataCopyExtParams cp_in{
                    SCAST<uint16_t>(this_tile),
                    L * type_size,
                    (K - 1) * L * type_size,   // GM srcStride, bytes
                    0,                          // UB dstStride, dataBlock
                    0
                };
                DataCopyPadExtParams<T> pp{false, 0, right_pad, 0};
                DataCopyPad(ub_buf, input_global[in_base + k * L], cp_in, pp);

                SetFlag<HardEvent::MTE2_MTE3>(0);
                WaitFlag<HardEvent::MTE2_MTE3>(0);

                __gm__ T* out_raw = output_list.template GetDataPtr<T>(k);
                GlobalTensor<T> out_k;
                out_k.SetGlobalBuffer(out_raw, SCAST<uint64_t>(M) * L);

                DataCopyExtParams cp_out{
                    SCAST<uint16_t>(this_tile),
                    L * type_size,
                    0,                          // UB srcStride, dataBlock (blocks adjacent)
                    0,                          // GM dstStride, bytes
                    0
                };
                DataCopyPad(out_k[SCAST<uint64_t>(m_base) * L], ub_buf, cp_out);

                bool more_k    = (k + 1 < K);
                bool more_tile = (m_base + pack_tile_m < m_end);
                if (more_k || more_tile) {
                    SetFlag<HardEvent::MTE3_MTE2>(0);
                    pending_mte3_mte2 = true;
                }
            }
        }
    }

    // ── Scatter path (original, L >= block_n) ────────────────────────────────
    __aicore__ inline void ProcessScatter() {
        constexpr int32_t buffer_num = 2;
        int32_t           type_size  = SCAST<int32_t>(sizeof(T));

        ListTensorDesc output_list(reinterpret_cast<__gm__ void*>(output_desc));

        LocalTensor<T> in_local[buffer_num];
        for (int j = 0; j < buffer_num; j++) {
            int32_t base_off = j * SCAST<int32_t>(tile_length) * type_size;
            in_local[j]      = LocalTensor<T>(TPosition::VECIN, base_off, tile_length);
        }

        uint32_t iter_begin  = SCAST<uint32_t>(GetBlockIdx());
        uint32_t iter_stride = SCAST<uint32_t>(GetBlockNum());
        uint32_t total_step  = iter_stride * SCAST<uint32_t>(buffer_num);
        uint32_t mr_per_out  = M * iter_per_row;

        for (uint32_t iter = iter_begin; iter < iterations; iter += total_step) {
            for (int j = 0; j < buffer_num; j++) {
                uint32_t cur_iter = iter + SCAST<uint32_t>(j) * iter_stride;
                if (cur_iter >= iterations) break;

                uint32_t output_idx = cur_iter / mr_per_out;
                uint32_t rem        = cur_iter - output_idx * mr_per_out;
                uint32_t m          = rem / iter_per_row;
                uint32_t tile_l_idx = rem - m * iter_per_row;

                bool     is_last = (tile_l_idx == iter_per_row - 1);
                uint32_t cur_l   = is_last ? real_residue : tile_length;
                uint32_t pad_l   = is_last ? pad_residue  : tile_length;
                uint32_t l_start = tile_l_idx * tile_length;

                uint64_t in_off  = SCAST<uint64_t>(m) * K * L +
                                   SCAST<uint64_t>(output_idx) * L +
                                   l_start;
                uint64_t out_off = SCAST<uint64_t>(m) * L + l_start;

                bool wait_back = (cur_iter >= iter_begin + total_step);
                bool set_back  = (cur_iter + total_step < iterations);

                if (wait_back) WaitFlag<HardEvent::MTE3_MTE2>(j);

                if (cur_l == pad_l) {
                    DataCopy(in_local[j], input_global[in_off], pad_l);
                } else {
                    DataCopyExtParams       copyParams{SCAST<uint16_t>(1),
                                                SCAST<uint32_t>(cur_l * sizeof(T)),
                                                0, 0, 0};
                    DataCopyPadExtParams<T> padParams{false, 0,
                                                      SCAST<uint8_t>(pad_l - cur_l),
                                                      0};
                    DataCopyPad(in_local[j], input_global[in_off], copyParams, padParams);
                }

                SetFlag<HardEvent::MTE2_MTE3>(j);
                WaitFlag<HardEvent::MTE2_MTE3>(j);

                __gm__ T*       out_raw = output_list.template GetDataPtr<T>(output_idx);
                GlobalTensor<T> out_i;
                out_i.SetGlobalBuffer(out_raw, SCAST<uint64_t>(M) * L);

                if (cur_l == pad_l) {
                    DataCopy(out_i[out_off], in_local[j], pad_l);
                } else {
                    DataCopyExtParams outParams{SCAST<uint16_t>(1),
                                                SCAST<uint32_t>(cur_l * sizeof(T)),
                                                0, 0, 0};
                    DataCopyPad(out_i[out_off], in_local[j], outParams);
                }

                if (set_back) SetFlag<HardEvent::MTE3_MTE2>(j);
            }
        }
    }

    GlobalTensor<T> input_global;
    GM_ADDR         output_desc;
    uint32_t        M, K, L;
    uint32_t        tile_length, iter_per_row, real_residue, pad_residue, iterations;
    uint32_t        pack_mode, pack_tile_m;
};

extern "C" __global__ __aicore__ void unpack(GM_ADDR input, GM_ADDR output,
                                             GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);

    KernelUnpack<DTYPE_INPUT> op;
    op.Init(input, output,
            tiling_data.M, tiling_data.K, tiling_data.L,
            tiling_data.tile_length, tiling_data.iter_per_row,
            tiling_data.real_residue, tiling_data.pad_residue,
            tiling_data.iterations,
            tiling_data.pack_mode, tiling_data.pack_tile_m);
    op.Process();
}
