#include "kernel_operator.h"
#include "kernel_inc.h"

template <typename T, typename U> class KernelTensorEqual {
  public:
    __aicore__ inline KernelTensorEqual() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR out, int64_t iterations,
                                int64_t tile_len, int64_t iter_per_vec, int64_t real_res,
                                int64_t pad_res, int64_t *input_shape, int64_t *other_shape) {

        InitSocState();

        for (int i = 0; i < MAX_SHAPE; i++) {
            input_stride[i] = (input_shape[i] < other_shape[i] ? 0 : 1);
            other_stride[i] = (other_shape[i] < input_shape[i] ? 0 : 1);
            out_shape[i]    = max(input_shape[i], other_shape[i]);
        }

        int64_t xprod = 1, yprod = 1, oprod = 1;
        for (int i = DIM_LAST - 1; i >= 0; i--) {
            input_prod[i] = xprod;
            other_prod[i] = yprod;
            out_prod[i]   = oprod;
            xprod *= input_shape[i];
            yprod *= other_shape[i];
            oprod *= out_shape[i];
        }

        int64_t iprod = 1;
        for (int i = DIM_LAST - 1; i >= 0; i--) {
            if (!input_stride[i]) {
                iter_prod_input_order[i] = iprod;
                iprod *= out_shape[i];
            }
        }
        iter_prod_input_order[DIM_LAST] = iprod;
        iprod *= iter_per_vec;
        for (int i = DIM_LAST - 1; i >= 0; i--) {
            if (input_stride[i]) {
                iter_prod_input_order[i] = iprod;
                iprod *= out_shape[i];
            }
        }

        input_prod[DIM_LAST] = input_shape[DIM_LAST];
        other_prod[DIM_LAST] = other_shape[DIM_LAST];
        out_prod[DIM_LAST]   = max(input_shape[DIM_LAST], other_shape[DIM_LAST]);

        this->total_len = oprod * out_prod[DIM_LAST];
        int64_t out_buf = total_len;
        this->tot_div   = 1.0f / SCAST<float>(total_len);

        input_global.SetGlobalBuffer((__gm__ T *)input, xprod * input_prod[DIM_LAST]);
        other_global.SetGlobalBuffer((__gm__ T *)other, yprod * other_prod[DIM_LAST]);
        out_global.SetGlobalBuffer((__gm__ U *)out, out_buf);

        this->tile_len     = tile_len;
        this->iterations   = iterations;
        this->iter_per_vec = iter_per_vec;
        this->real_res     = real_res;
        this->pad_res      = pad_res;
    }
    __aicore__ inline void Process() {
        auto res_float = LocalTensor<float>(TPosition::VECCALC, 0, 256);
        Duplicate(res_float, 0.0f, 256);
        PipeBarrier<PIPE_ALL>();

        int64_t scale = 4 / sizeof(T), buffer_num = 2;

        LocalTensor<T> input_local[2] = {
            LocalTensor<T>(TPosition::VECIN, 1024 + 0 * tile_len * 8, tile_len * scale),
            LocalTensor<T>(TPosition::VECIN, 1024 + 2 * tile_len * 8, tile_len * scale),
        };
        LocalTensor<T> other_local[2] = {
            LocalTensor<T>(TPosition::VECIN, 1024 + 1 * tile_len * 8, tile_len * scale),
            LocalTensor<T>(TPosition::VECIN, 1024 + 3 * tile_len * 8, tile_len * scale),
        };
        LocalTensor<uint8_t> out_local[2] = {
            LocalTensor<uint8_t>(TPosition::VECOUT, 1024 + 4 * tile_len * 8, tile_len * scale),
            LocalTensor<uint8_t>(TPosition::VECOUT, 1024 + 5 * tile_len * 8, tile_len * scale),
        };

        LocalTensor<uint8_t> tmp_local[2] = {
            LocalTensor<uint8_t>(TPosition::VECCALC, 1024 + 6 * tile_len * 8, tile_len * scale),
            LocalTensor<uint8_t>(TPosition::VECCALC, 1024 + 7 * tile_len * 8, tile_len * scale),
        };
        int64_t pre_input_iter[2] = {-1, -1};

        int64_t iter_begin      = GetBlockIdx();
        int64_t iter_buf_step   = GetBlockNum();
        int64_t iter_total_step = iter_buf_step * buffer_num;

        for (int64_t i = iter_begin; i < iterations; i += iter_total_step) {
            for (int j = 0; j < buffer_num; j++) {
                // ================= BEGIN Tiling =================
                int64_t iter = i + j * iter_buf_step;
                if (iter >= iterations) break;

                int64_t input_vec, other_vec, out_vec, tile;
                calc_vec(iter, input_vec, other_vec, out_vec, tile);
                //printf("input_vec: %ld, other_vec: %ld, out_vec: %ld\n", input_vec, other_vec, out_vec);
                int64_t input_iter = iter_per_vec * input_vec + tile * input_stride[DIM_LAST];

                bool    is_last  = (tile == iter_per_vec - 1);
                int64_t cur_len  = is_last ? pad_res : tile_len;
                int64_t real_len = is_last ? real_res : tile_len;
                int64_t offset   = tile * tile_len;

                int64_t input_len   = (input_stride[DIM_LAST] ? cur_len : block_n<T>);
                int64_t other_len   = (other_stride[DIM_LAST] ? cur_len : block_n<T>);
                int64_t cast_offset = 0;
                if constexpr (is_casting<T>) cast_offset = tile_len;
                else if constexpr(is_casting_8<T>) cast_offset = 7 * tile_len;

                bool wait_backward = (iter >= iter_total_step);
                bool set_backward  = (iter + iter_total_step < iterations);
                // ================= END Tiling =================
                // ================= BEGIN CopyIn =================
                if (wait_backward) WaitFlag<HardEvent::MTE3_MTE2>(j);
                if (input_iter != pre_input_iter[j]) {
                    DataCopy(input_local[j][cast_offset],
                             input_global[input_vec * input_prod[DIM_LAST] +
                                          offset * input_stride[DIM_LAST]],
                             input_len);
                }
                DataCopy(other_local[j][cast_offset],
                         other_global[other_vec * other_prod[DIM_LAST] +
                                      offset * other_stride[DIM_LAST]],
                         other_len);
                // ================= END CopyIn =================
                SetFlag<HardEvent::MTE2_V>(j);
                WaitFlag<HardEvent::MTE2_V>(j);
                // ================= BEGIN Compute =================
                auto input_calc = VCAST(float, input_local[j]);
                auto other_calc = VCAST(float, other_local[j]);
                auto out_calc   = VCAST(uint8_t, out_local[j]);
                if constexpr (is_casting<T>) {
                    if (input_iter != pre_input_iter[j]) {
                        Cast(input_calc, input_local[j][tile_len], RoundMode::CAST_NONE, input_len);
                    }
                    Cast(other_calc, other_local[j][tile_len], RoundMode::CAST_NONE, other_len);
                }
                else if constexpr (is_casting_8<T>) {
                    auto input_tmp = VCAST(half, input_local[j][4 * tile_len]);
                    auto other_tmp = VCAST(half, other_local[j][4 * tile_len]);
                    if (input_iter != pre_input_iter[j]) {
                        Cast(input_tmp, input_local[j][7 * tile_len], RoundMode::CAST_NONE, input_len);
                        Cast(input_calc, input_tmp, RoundMode::CAST_NONE, input_len);
                    }
                    Cast(other_tmp, other_local[j][7 * tile_len], RoundMode::CAST_NONE, other_len);
                    Cast(other_calc, other_tmp, RoundMode::CAST_NONE, other_len);
                }
                // for (int k = 0; k < cur_len; k++) {
                //     printf("input_calc[%d] = %f, other_calc[%d] = %f\n", k, input_calc(k), k, other_calc(k));
                // }
                auto cmp_res = VCAST(uint8_t, tmp_local[j]);
                if (input_stride[DIM_LAST] && other_stride[DIM_LAST])
                    Compare(cmp_res, input_calc, other_calc, CMPMODE::EQ, cur_len);
                else if (input_stride[DIM_LAST])
                    CompareScalar(cmp_res, input_calc, SCAST<float>(other_calc(0)), CMPMODE::EQ, cur_len);
                else CompareScalar(cmp_res, other_calc, SCAST<float>(input_calc(0)), CMPMODE::EQ, cur_len);
                auto clean_res = VCAST(half, out_calc);
                Duplicate(clean_res, SCAST<half>(1.0), cur_len);
                //auto clean_res_half = VCAST(half, clean_res);
                Select(clean_res, cmp_res, clean_res, SCAST<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, cur_len);
                Cast(out_calc, clean_res, RoundMode::CAST_RINT, cur_len);
                // ================= END Compute =================
                SetFlag<HardEvent::V_MTE3>(j);
                WaitFlag<HardEvent::V_MTE3>(j);
                // ================= BEGIN CopyOut =================
                int64_t contentSize = real_len * sizeof(T);
                int64_t alignSize   = cur_len * sizeof(T);
                int64_t srcStride   = (alignSize - (contentSize + 31) / 32 * 32) / 32;
                int64_t pos         = out_vec * out_prod[DIM_LAST] + offset;
                // for (int k = 0; k < cur_len; k++) {
                //     printf("out_calc: %d\n", (int)out_calc(k));
                // }
                // printf("datacopypad: %ld %ld\n", contentSize, srcStride);
                DataCopyExtParams copyParams{SCAST<uint16_t>(1), SCAST<uint32_t>(contentSize),
                                                SCAST<uint32_t>(srcStride), 0, 0};
                DataCopyPad(out_global[pos], out_local[j], copyParams);
                // ================= END CopyOut =================
                pre_input_iter[j] = input_iter;
                if (set_backward) SetFlag<HardEvent::MTE3_MTE2>(j);
            }
        }
    }

  private:
    __aicore__ inline void calc_vec(int64_t iter, int64_t &input_vec, int64_t &other_vec,
                                    int64_t &out_vec, int64_t &tile) {
        input_vec = other_vec = out_vec = 0;
        for (int i = 0; i < DIM_LAST; i++) {
            if (input_stride[i]) {
                int64_t id = iter / iter_prod_input_order[i];
                iter %= iter_prod_input_order[i];
                input_vec += id * input_prod[i];
                other_vec += id * other_stride[i] * other_prod[i];
                out_vec += id * out_prod[i];
            }
        }
        tile = iter / iter_prod_input_order[DIM_LAST];
        iter %= iter_prod_input_order[DIM_LAST];
        for (int i = 0; i < DIM_LAST; i++) {
            if (!input_stride[i]) {
                int64_t id = iter / iter_prod_input_order[i];
                iter %= iter_prod_input_order[i];
                other_vec += id * other_stride[i] * other_prod[i];
                out_vec += id * out_prod[i];
            }
        }
    }

  private:
    GlobalTensor<T> input_global, other_global;
    GlobalTensor<U> out_global;

    int64_t tile_len;
    int64_t iterations, iter_per_vec;
    int64_t real_res, pad_res;
    int64_t total_len;
    float   tot_div;
    int64_t out_prod[MAX_SHAPE], input_prod[MAX_SHAPE], other_prod[MAX_SHAPE],
        iter_prod_input_order[MAX_SHAPE], out_shape[MAX_SHAPE];
    int8_t input_stride[MAX_SHAPE], other_stride[MAX_SHAPE];
};

extern "C" __global__ __aicore__ void tensor_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelTensorEqual<DTYPE_X1, uint8_t> op;

    if (!tiling_data.iter_idx) {
        op.Init(x1, x2, y, tiling_data.iterations, tiling_data.tile_length,
                tiling_data.iter_per_vector, tiling_data.real_residue, tiling_data.pad_residue,
                tiling_data.input_shape, tiling_data.other_shape);
    } else {
        op.Init(x2, x1, y, tiling_data.iterations, tiling_data.tile_length,
                tiling_data.iter_per_vector, tiling_data.real_residue, tiling_data.pad_residue,
                tiling_data.other_shape, tiling_data.input_shape);
    }
    op.Process();
}