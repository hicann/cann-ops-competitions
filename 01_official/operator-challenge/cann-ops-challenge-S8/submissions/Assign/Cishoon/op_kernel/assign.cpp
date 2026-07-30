
#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;

#define SCAST static_cast
#define VCAST(T, vec) vec.template ReinterpretCast<T>()

constexpr int MAX_SHAPE(4);
constexpr int DIM_LAST(MAX_SHAPE - 1);

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

template <class T>
inline constexpr bool is_bit_8 = std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>;

template <class T> struct reduce { using type = T; };
template <> struct reduce<bool> { using type = uint8_t; };
template <> struct reduce<bfloat16_t> { using type = float16_t; };
template <class T> using reduce_t = typename reduce<T>::type;

template <class T> 
struct block {
    constexpr static int32_t value = 32 / sizeof(T);
};
template <class T> 
constexpr static int32_t block_n = block<T>::value;

__aicore__ inline int32_t Low32(uint64_t packed) {
    return static_cast<int32_t>(packed);
}

__aicore__ inline int32_t High32(uint64_t packed) {
    return static_cast<int32_t>(packed >> 32);
}

template <class T> 
class KernelAssign {
  public:
    __aicore__ inline KernelAssign() {}
    __aicore__ inline void Init(GM_ADDR other, GM_ADDR out, int32_t tile_len,
                                int32_t *other_shape, int32_t *out_shape_arr) {

        InitSocState();

        for (int i = 0; i < MAX_SHAPE; i++) {
            other_stride[i] = (other_shape[i] < out_shape_arr[i] ? 0 : 1);
            this->out_shape[i] = out_shape_arr[i];
        }

        int32_t yprod = 1, oprod = 1;
        for (int i = DIM_LAST - 1; i >= 0; i--) {
            other_prod[i] = yprod;
            out_prod[i]   = oprod;
            yprod *= other_shape[i];
            oprod *= out_shape_arr[i];
        }

        int32_t length_per_block  = 64 / sizeof(T);
        int32_t length_per_vector = ((out_shape_arr[DIM_LAST] + length_per_block - 1) /
                                     length_per_block) *
                                    length_per_block;

        this->tile_len     = tile_len;
        this->iter_per_vec = (length_per_vector + tile_len - 1) / tile_len;
        this->iterations   = oprod * this->iter_per_vec;
        this->real_res     = (out_shape_arr[DIM_LAST] - 1) % tile_len + 1;
        this->pad_res      = (length_per_vector - 1) % tile_len + 1;

        // Build iteration order: other_stride=1 dims outer, other_stride=0 dims inner
        int32_t iprod = 1;
        for (int i = DIM_LAST - 1; i >= 0; i--) {
            if (!other_stride[i]) {
                iter_prod_order[i] = iprod;
                iprod *= out_shape_arr[i];
            }
        }
        iter_prod_order[DIM_LAST] = iprod;
        iprod *= this->iter_per_vec;
        for (int i = DIM_LAST - 1; i >= 0; i--) {
            if (other_stride[i]) {
                iter_prod_order[i] = iprod;
                iprod *= out_shape_arr[i];
            }
        }

        other_prod[DIM_LAST] = other_shape[DIM_LAST];
        out_prod[DIM_LAST]   = out_shape_arr[DIM_LAST];

        other_global.SetGlobalBuffer((__gm__ T *)other, yprod * other_prod[DIM_LAST]);
        out_global.SetGlobalBuffer((__gm__ T *)out, oprod * out_prod[DIM_LAST]);
    }

    __aicore__ inline void Process() {
        int32_t type_size = max(sizeof(T), 2), scale = type_size / sizeof(T);
        int32_t buffer_num = (out_shape[DIM_LAST] > 8192 ? 1 : 2);

        LocalTensor<T> other_local[2] = {
            LocalTensor<T>(TPosition::VECIN, 0 * tile_len * type_size, tile_len * scale),
            LocalTensor<T>(TPosition::VECIN, (buffer_num - 1) * tile_len * type_size,
                           tile_len * scale),
        };
        LocalTensor<T> out_local[2] = {
            LocalTensor<T>(TPosition::VECOUT, 0 * tile_len * type_size, tile_len * scale),
            LocalTensor<T>(TPosition::VECOUT, (buffer_num - 1) * tile_len * type_size,
                           tile_len * scale),
        };
        int32_t pre_other_iter[2] = {-1, -1};

        int32_t iter_begin      = GetBlockIdx();
        int32_t iter_buf_step   = GetBlockNum();
        int32_t iter_total_step = iter_buf_step * buffer_num;

        for (int32_t i = iter_begin; i < iterations; i += iter_total_step) {
            for (int j = 0; j < buffer_num; j++) {
                int32_t iter = i + j * iter_buf_step;
                if (iter >= iterations) break;

                int32_t other_vec, out_vec, tile;
                calc_vec(iter, other_vec, out_vec, tile);
                int32_t other_iter = iter_per_vec * other_vec + tile * other_stride[DIM_LAST];

                bool    is_last  = (tile == iter_per_vec - 1);
                int32_t cur_len  = is_last ? pad_res : tile_len;
                int32_t real_len = is_last ? real_res : tile_len;
                int32_t offset   = tile * tile_len;

                int32_t other_len = (other_stride[DIM_LAST] ? cur_len : block_n<T>);

                bool wait_backward = (iter >= iter_total_step);
                bool set_backward  = (iter + iter_total_step < iterations);

                // CopyIn
                if (wait_backward) WaitFlag<HardEvent::MTE3_MTE2>(j);
                if (other_iter != pre_other_iter[j]) {
                    DataCopy(other_local[j],
                             other_global[other_vec * other_prod[DIM_LAST] +
                                          offset * other_stride[DIM_LAST]],
                             other_len);
                }
                SetFlag<HardEvent::MTE2_V>(j);
                WaitFlag<HardEvent::MTE2_V>(j);

                // Compute: broadcast copy other -> out
                if (other_stride[DIM_LAST]) {
                    // No broadcast on last dim, direct copy
                    DataCopy(out_local[j], other_local[j], cur_len);
                } else {
                    // Last dim broadcast: fill with scalar
                    if constexpr (is_bit_8<T>) {
                        // Duplicate doesn't support int8/uint8, pack to int16
                        T raw = other_local[j].GetValue(0);
                        int16_t packed = static_cast<int16_t>(static_cast<uint8_t>(raw)) |
                                         (static_cast<int16_t>(static_cast<uint8_t>(raw)) << 8);
                        auto out16 = out_local[j].template ReinterpretCast<int16_t>();
                        Duplicate(out16, packed, cur_len / 2 + 1);
                    } else {
                        T val = other_local[j].GetValue(0);
                        Duplicate(out_local[j], val, cur_len);
                    }
                }

                SetFlag<HardEvent::V_MTE3>(j);
                WaitFlag<HardEvent::V_MTE3>(j);

                // CopyOut
                int32_t contentSize = real_len * sizeof(T);
                int32_t alignSize   = cur_len * sizeof(T);
                int32_t srcStride   = (alignSize - (contentSize + 31) / 32 * 32) / 32;
                int32_t pos         = out_vec * out_prod[DIM_LAST] + offset;

                DataCopyExtParams copyParams{SCAST<uint16_t>(1), SCAST<uint32_t>(contentSize),
                                             SCAST<uint32_t>(srcStride), 0, 0};
                DataCopyPad(out_global[pos], out_local[j], copyParams);

                pre_other_iter[j] = other_iter;
                if (set_backward) SetFlag<HardEvent::MTE3_MTE2>(j);
            }
        }
    }

  private:
    __aicore__ inline void calc_vec(int32_t iter, int32_t &other_vec, int32_t &out_vec,
                                    int32_t &tile) {
        other_vec = out_vec = 0;
        for (int i = 0; i < DIM_LAST; i++) {
            if (other_stride[i]) {
                int32_t id = iter / iter_prod_order[i];
                iter %= iter_prod_order[i];
                other_vec += id * other_prod[i];
                out_vec += id * out_prod[i];
            }
        }
        tile = iter / iter_prod_order[DIM_LAST];
        iter %= iter_prod_order[DIM_LAST];
        for (int i = 0; i < DIM_LAST; i++) {
            if (!other_stride[i]) {
                int32_t id = iter / iter_prod_order[i];
                iter %= iter_prod_order[i];
                other_vec += id * other_stride[i] * other_prod[i];
                out_vec += id * out_prod[i];
            }
        }
    }

  private:
    GlobalTensor<T> other_global;
    GlobalTensor<T> out_global;

    int32_t tile_len;
    int32_t iterations, iter_per_vec;
    int32_t real_res, pad_res;
    int32_t out_prod[MAX_SHAPE], other_prod[MAX_SHAPE],
        iter_prod_order[MAX_SHAPE], out_shape[MAX_SHAPE];
    int8_t other_stride[MAX_SHAPE];
};

template <class T>
__aicore__ inline void ProcessSameShapeCopy(GM_ADDR other, GM_ADDR output,
                                            int32_t total_len, int32_t tile_len) {
    GlobalTensor<T> other_global;
    GlobalTensor<T> out_global;
    other_global.SetGlobalBuffer((__gm__ T *)other, total_len);
    out_global.SetGlobalBuffer((__gm__ T *)output, total_len);

    int32_t type_size = max(sizeof(T), 2), scale = type_size / sizeof(T);
    int32_t length_per_block  = 64 / sizeof(T);
    int32_t length_per_vector = ((total_len + length_per_block - 1) / length_per_block) *
                                length_per_block;
    int32_t tile_count = (length_per_vector + tile_len - 1) / tile_len;
    int32_t buffer_num = (total_len > 8192 ? 1 : 2);

    LocalTensor<T> local[2] = {
        LocalTensor<T>(TPosition::VECIN, 0 * tile_len * type_size, tile_len * scale),
        LocalTensor<T>(TPosition::VECIN, (buffer_num - 1) * tile_len * type_size,
                       tile_len * scale),
    };

    int32_t tile_begin = tile_count * GetBlockIdx() / GetBlockNum();
    int32_t tile_end   = tile_count * (GetBlockIdx() + 1) / GetBlockNum();

    for (int32_t i = tile_begin; i < tile_end; i += buffer_num) {
        for (int32_t j = 0; j < buffer_num; ++j) {
            int32_t tile = i + j;
            if (tile >= tile_end) break;

            int32_t offset = tile * tile_len;
            int32_t pad_len = min(tile_len, length_per_vector - offset);
            int32_t real_len = min(pad_len, total_len - offset);

            bool wait_backward = (tile >= tile_begin + buffer_num);
            bool set_backward  = (tile + buffer_num < tile_end);

            if (wait_backward) WaitFlag<HardEvent::MTE3_MTE2>(j);

            // Match the original generic path: input is read at padded length, output drops tail
            // bytes with DataCopyPad. DataCopyPad GM->UB cannot pad more than 32B.
            DataCopy(local[j], other_global[offset], pad_len);

            SetFlag<HardEvent::MTE2_MTE3>(j);
            WaitFlag<HardEvent::MTE2_MTE3>(j);

            if (real_len == pad_len) {
                DataCopy(out_global[offset], local[j], pad_len);
            } else {
                DataCopyExtParams copyParams{SCAST<uint16_t>(1),
                                             SCAST<uint32_t>(real_len * sizeof(T)),
                                             0, 0, 0};
                DataCopyPad(out_global[offset], local[j], copyParams);
            }

            if (set_backward) SetFlag<HardEvent::MTE3_MTE2>(j);
        }
    }
}

template <class T>
__aicore__ inline void ProcessScalarFill(GM_ADDR other, GM_ADDR output,
                                         int32_t total_len, int32_t tile_len) {
    GlobalTensor<T> other_global;
    GlobalTensor<T> out_global;
    other_global.SetGlobalBuffer((__gm__ T *)other, block_n<T>);
    out_global.SetGlobalBuffer((__gm__ T *)output, total_len);

    int32_t type_size = max(sizeof(T), 2), scale = type_size / sizeof(T);
    int32_t length_per_block  = 64 / sizeof(T);
    int32_t length_per_vector = ((total_len + length_per_block - 1) / length_per_block) *
                                length_per_block;
    int32_t tile_count = (length_per_vector + tile_len - 1) / tile_len;
    int32_t tile_begin = tile_count * GetBlockIdx() / GetBlockNum();
    int32_t tile_end   = tile_count * (GetBlockIdx() + 1) / GetBlockNum();
    if (tile_begin >= tile_end) return;

    LocalTensor<T> scalar_local(TPosition::VECIN, 0, block_n<T>);
    DataCopy(scalar_local, other_global[0], block_n<T>);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    LocalTensor<T> stamp(TPosition::VECOUT, 0, tile_len * scale);
    int32_t stamp_len = min(tile_len, length_per_vector);

    if constexpr (is_bit_8<T>) {
        T raw = scalar_local.GetValue(0);
        int16_t packed = static_cast<int16_t>(static_cast<uint8_t>(raw)) |
                         (static_cast<int16_t>(static_cast<uint8_t>(raw)) << 8);
        auto stamp16 = stamp.template ReinterpretCast<int16_t>();
        Duplicate(stamp16, packed, stamp_len / 2 + 1);
    } else {
        T val = scalar_local.GetValue(0);
        Duplicate(stamp, val, stamp_len);
    }

    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);

    for (int32_t tile = tile_begin; tile < tile_end; ++tile) {
        int32_t offset = tile * tile_len;
        int32_t pad_len = min(tile_len, length_per_vector - offset);
        int32_t real_len = min(pad_len, total_len - offset);

        if (real_len == pad_len) {
            DataCopy(out_global[offset], stamp, pad_len);
        } else {
            DataCopyExtParams copyParams{SCAST<uint16_t>(1),
                                         SCAST<uint32_t>(real_len * sizeof(T)),
                                         0, 0, 0};
            DataCopyPad(out_global[offset], stamp, copyParams);
        }
    }
}

template <bool SAME_SHAPE, class T>
__aicore__ inline void RunOneShapeAssign(GM_ADDR other, GM_ADDR output, GM_ADDR tiling) {
    uint64_t packed = *((const __gm__ uint64_t *)tiling);
    int32_t tile_length = Low32(packed);
    int32_t out_shape_last = High32(packed);

    if constexpr (SAME_SHAPE) {
        ProcessSameShapeCopy<T>(other, output, out_shape_last, tile_length);
    } else {
        ProcessScalarFill<T>(other, output, out_shape_last, tile_length);
    }
}

template <int COMPACT_SHAPE_NUM, class T>
__aicore__ inline void RunAssign(GM_ADDR other, GM_ADDR output, GM_ADDR tiling) {
    constexpr int32_t tiling_word_num = 1 + COMPACT_SHAPE_NUM * 2;
    int32_t tiling_words[1 + MAX_SHAPE * 2];

    constexpr int32_t packed_num = tiling_word_num / 2;
    const __gm__ uint64_t *tiling64 = (const __gm__ uint64_t *)tiling;
    for (int32_t i = 0; i < packed_num; ++i) {
        uint64_t packed = tiling64[i];
        tiling_words[i * 2] = Low32(packed);
        tiling_words[i * 2 + 1] = High32(packed);
    }
    if constexpr ((tiling_word_num & 1) != 0) {
        tiling_words[tiling_word_num - 1] =
            *((const __gm__ int32_t *)tiling + tiling_word_num - 1);
    }

    int32_t other_shape[MAX_SHAPE];
    int32_t out_shape[MAX_SHAPE];
    for (int32_t i = 0; i < MAX_SHAPE; ++i) {
        other_shape[i] = 1;
        out_shape[i]   = 1;
    }

    constexpr int32_t start = MAX_SHAPE - COMPACT_SHAPE_NUM;
    for (int32_t i = 0; i < COMPACT_SHAPE_NUM; ++i) {
        other_shape[start + i] = tiling_words[1 + i];
        out_shape[start + i]   = tiling_words[1 + COMPACT_SHAPE_NUM + i];
    }

    KernelAssign<T> op;
    op.Init(other, output, tiling_words[0], other_shape, out_shape);
    op.Process();
}

extern "C" __global__ __aicore__ void assign(GM_ADDR input, GM_ADDR other,
                                             GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using input_t = reduce_t<DTYPE_INPUT>;

    // key 0: no broadcast, key 1: scalar broadcast, key 2..5: compact_shape_num 1..4.
    if (TILING_KEY_IS(0)) {
        RunOneShapeAssign<true, input_t>(other, output, tiling);
    } else if (TILING_KEY_IS(1)) {
        RunOneShapeAssign<false, input_t>(other, output, tiling);
    } else if (TILING_KEY_IS(2)) {
        RunAssign<1, input_t>(other, output, tiling);
    } else if (TILING_KEY_IS(3)) {
        RunAssign<2, input_t>(other, output, tiling);
    } else if (TILING_KEY_IS(4)) {
        RunAssign<3, input_t>(other, output, tiling);
    } else if (TILING_KEY_IS(5)) {
        RunAssign<4, input_t>(other, output, tiling);
    }
}
