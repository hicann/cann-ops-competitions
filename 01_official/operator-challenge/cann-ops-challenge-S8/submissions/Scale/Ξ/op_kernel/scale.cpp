#include "kernel_operator.h"

using namespace AscendC;

template<typename T>
__aicore__ inline constexpr T ceil_div(T x, T y)
{
    return (x - 1) / y + 1;
}

template<typename T>
__aicore__ inline constexpr T ceil_round(T x, T y)
{
    return ceil_div(x, y) * y;
}

template<typename T>
__aicore__ inline constexpr T floor_round(T x, T y)
{
    return x / y * y;
}

#define scale_0_0() \
    int block_index = GetBlockIdx(); \
    int block_dim = GetBlockNum(); \
    constexpr int MTE_BLOCK_SIZE = 32 / sizeof(T); \
    int mid_blocks = ceil_div(tiling.mid_size, MTE_BLOCK_SIZE); \
    int mid_start = mid_blocks * block_index / block_dim * MTE_BLOCK_SIZE; \
    int mid_end = min(mid_blocks * (block_index + 1) / block_dim * MTE_BLOCK_SIZE, tiling.mid_size);

#define scale_0_1() \
    int block_index = GetBlockIdx(); \
    int block_dim = GetBlockNum(); \
    int high_start = tiling.high_size * block_index / block_dim; \
    int high_end = tiling.high_size * (block_index + 1) / block_dim;

#define scale_0_2() \
    int block_index = GetBlockIdx(); \
    constexpr int DATA_BLOCK_SIZE = 32 / sizeof(T); \
    constexpr int high_cut = 10; \
    constexpr int mid_cut = 4; \
    int high_index = block_index / mid_cut; \
    int high_start = tiling.high_size * high_index / high_cut; \
    int high_end = tiling.high_size * (high_index + 1) / high_cut; \
    int mid_index = block_index % mid_cut; \
    int mid_blocks = ceil_div(tiling.mid_size, DATA_BLOCK_SIZE); \
    int mid_offset = mid_blocks * mid_index / mid_cut * DATA_BLOCK_SIZE; \
    int _ = min(mid_blocks * (mid_index + 1) / mid_cut * DATA_BLOCK_SIZE, tiling.mid_size) - mid_offset;

#define scale_0_3() \
    int block_index = GetBlockIdx(); \
    int block_dim = GetBlockNum(); \
    int compute_size = tiling.high_size * tiling.mid_size; \
    int compute_start = compute_size * block_index / block_dim; \
    int compute_end = compute_size * (block_index + 1) / block_dim;

#define scale_0_4() \
    int block_index = GetBlockIdx(); \
    int block_dim = GetBlockNum(); \
    constexpr int DATA_BLOCK_SIZE = 32 / sizeof(T); \
    int mid_blocks = tiling.mid_size / DATA_BLOCK_SIZE; \
    int mid_start = mid_blocks * block_index / block_dim * DATA_BLOCK_SIZE; \
    int mid_end = mid_blocks * (block_index + 1) / block_dim * DATA_BLOCK_SIZE;

#define scale_1_0() \
    GlobalTensor<T> input_global_tensor, scale_global_tensor, output_global_tensor; \
    input_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input)); \
    scale_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(scale)); \
    output_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output));

#define scale_1_1() \
    GlobalTensor<T> input_global_tensor, scale_global_tensor, bias_global_tensor, output_global_tensor; \
    input_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input)); \
    scale_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(scale)); \
    bias_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias)); \
    output_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output));

#define scale_2_0() \
    TPipe t_pipe; \
    TQue<TPosition::VECIN, 1> input_t_que, scale_t_que; \
    TQue<TPosition::VECOUT, 1> output_t_que;

#define scale_2_1() \
    TPipe t_pipe; \
    TQue<TPosition::VECIN, 1> input_t_que, scale_t_que, bias_t_que; \
    TQue<TPosition::VECOUT, 1> output_t_que;

#define scale_2_2() \
    TPipe t_pipe; \
    TQue<TPosition::VECIN, 1> input_t_que; \
    TQue<TPosition::VECOUT, 1> output_t_que;

#define scale_3_0() \
    constexpr int MAX_TILE_SIZE = (63 << 10) / sizeof(T); \
    t_pipe.InitBuffer(input_t_que, 1, 64512); \
    t_pipe.InitBuffer(scale_t_que, 1, 64512); \
    t_pipe.InitBuffer(output_t_que, 1, 64512);

#define scale_3_1() \
    constexpr int MAX_TILE_SIZE = (63 << 10) / sizeof(U); \
    t_pipe.InitBuffer(input_t_que, 1, 64512); \
    t_pipe.InitBuffer(scale_t_que, 1, 64512); \
    t_pipe.InitBuffer(output_t_que, 1, 64512);

#define scale_3_2() \
    constexpr int MAX_TILE_SIZE = (47 << 10) / sizeof(T); \
    t_pipe.InitBuffer(input_t_que, 1, 48128); \
    t_pipe.InitBuffer(scale_t_que, 1, 48128); \
    t_pipe.InitBuffer(bias_t_que, 1, 48128); \
    t_pipe.InitBuffer(output_t_que, 1, 48128);

#define scale_3_3() \
    constexpr int MAX_TILE_SIZE = (47 << 10) / sizeof(U); \
    t_pipe.InitBuffer(input_t_que, 1, 48128); \
    t_pipe.InitBuffer(scale_t_que, 1, 48128); \
    t_pipe.InitBuffer(bias_t_que, 1, 48128); \
    t_pipe.InitBuffer(output_t_que, 1, 48128);

#define scale_3_4() \
    constexpr int MAX_TILE_SIZE = (47 << 10) / sizeof(T); \
    int round_mid_size = ceil_round(tiling.mid_size, static_cast<int>(32 / sizeof(T))); \
    int max_high_tile_size = MAX_TILE_SIZE / round_mid_size; \
    t_pipe.InitBuffer(input_t_que, 1, 48128); \
    t_pipe.InitBuffer(scale_t_que, 1, 48128); \
    t_pipe.InitBuffer(bias_t_que, 1, 48128); \
    t_pipe.InitBuffer(output_t_que, 1, 48128);

#define scale_3_5() \
    constexpr int MAX_TILE_SIZE = (31 << 10) / sizeof(U); \
    t_pipe.InitBuffer(input_t_que, 2, 31744); \
    t_pipe.InitBuffer(scale_t_que, 1, 32000); \
    t_pipe.InitBuffer(bias_t_que, 1, 31744); \
    t_pipe.InitBuffer(output_t_que, 2, 31744);

#define scale_3_6() \
    constexpr int MAX_TILE_SIZE = (95 << 10) / sizeof(T); \
    int max_mid_tile_size = MAX_TILE_SIZE / tiling.low_size; \
    t_pipe.InitBuffer(input_t_que, 1, 97280); \
    t_pipe.InitBuffer(output_t_que, 1, 97280);

#define scale_3_7() \
    constexpr int MAX_TILE_SIZE = (47 << 10) / sizeof(T); \
    int max_mid_tile_size = floor_round(MAX_TILE_SIZE / tiling.broadcast_size, DATA_BLOCK_SIZE); \
    t_pipe.InitBuffer(input_t_que, 1, 48384); \
    t_pipe.InitBuffer(scale_t_que, 1, 48384); \
    t_pipe.InitBuffer(bias_t_que, 1, 48384); \
    t_pipe.InitBuffer(output_t_que, 1, 48384);

#define scale_4_0()

#define scale_4_1() \
    int loop_start, loop_end, loop_step; \
    uint16_t atomic_type, atomic_op; \
    GetStoreAtomicConfig(atomic_type, atomic_op); \
    if (atomic_type) \
    { \
        loop_start = high_start + ceil_round(high_end - high_start, max_high_tile_size) - max_high_tile_size; \
        loop_end = high_start - max_high_tile_size; \
        loop_step = -max_high_tile_size; \
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_NONE, AtomicOp::ATOMIC_SUM>(); \
    } \
    else \
    { \
        loop_start = high_start; \
        loop_end = high_start + ceil_round(high_end - high_start, max_high_tile_size); \
        loop_step = max_high_tile_size; \
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_S32, AtomicOp::ATOMIC_SUM>(); \
    }

#define scale_4_2() \
    int loop_start, loop_end, loop_step; \
    uint16_t atomic_type, atomic_op; \
    GetStoreAtomicConfig(atomic_type, atomic_op); \
    if (atomic_type) \
    { \
        loop_start = high_end - 1; \
        loop_end = high_start - 1; \
        loop_step = -1; \
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_NONE, AtomicOp::ATOMIC_SUM>(); \
    } \
    else \
    { \
        loop_start = high_start; \
        loop_end = high_end; \
        loop_step = 1; \
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_S32, AtomicOp::ATOMIC_SUM>(); \
    }

#define scale_4_3() \
    int loop_start, loop_end, loop_step; \
    uint16_t atomic_type, atomic_op; \
    GetStoreAtomicConfig(atomic_type, atomic_op); \
    if (atomic_type) \
    { \
        loop_start = compute_start + ceil_round(compute_end - compute_start, max_mid_tile_size) - max_mid_tile_size; \
        loop_end = compute_start - max_mid_tile_size; \
        loop_step = -max_mid_tile_size; \
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_NONE, AtomicOp::ATOMIC_SUM>(); \
    } \
    else \
    { \
        loop_start = compute_start; \
        loop_end = compute_start + ceil_round(compute_end - compute_start, max_mid_tile_size); \
        loop_step = max_mid_tile_size; \
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_S32, AtomicOp::ATOMIC_SUM>(); \
    }

#define scale_4_4() \
    int mid_loop_start, mid_loop_end, mid_loop_step, high_loop_start, high_loop_end, high_loop_step; \
    uint16_t atomic_type, atomic_op; \
    GetStoreAtomicConfig(atomic_type, atomic_op); \
    if (atomic_type) \
    { \
        mid_loop_start = mid_start + ceil_round(mid_end - mid_start, max_mid_tile_size) - max_mid_tile_size; \
        mid_loop_end = mid_start - max_mid_tile_size; \
        mid_loop_step = -max_mid_tile_size; \
        high_loop_start = tiling.high_size - 1; \
        high_loop_end = -1; \
        high_loop_step = -1; \
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_NONE, AtomicOp::ATOMIC_SUM>(); \
    } \
    else \
    { \
        mid_loop_start = mid_start; \
        mid_loop_end =  mid_start + ceil_round(mid_end - mid_start, max_mid_tile_size); \
        mid_loop_step = max_mid_tile_size; \
        high_loop_start = 0; \
        high_loop_end = tiling.high_size; \
        high_loop_step = 1; \
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_S32, AtomicOp::ATOMIC_SUM>(); \
    }

#define scale_5_0() \
    if (tiling.low_size == 1) \
    { \
        for (int i = mid_start; i < mid_end; i += MAX_TILE_SIZE) \
        { \
            int _ = min(mid_end - i, MAX_TILE_SIZE); \
            DataCopyParams data_copy_params; \
            data_copy_params.blockLen = _ * sizeof(T); \
            { \
                LocalTensor<T> scale = scale_t_que.AllocTensor<T>(); \
                DataCopyPad(scale, scale_global_tensor[i], data_copy_params, {}); \
                scale_t_que.EnQue(scale); \
            } \
            LocalTensor<T> scale = scale_t_que.DeQue<T>(); \
            for (int j = 0; j < tiling.high_size; j++) \
            { \
                { \
                    LocalTensor<T> input = input_t_que.AllocTensor<T>(); \
                    DataCopyPad(input, input_global_tensor[j * tiling.mid_size + i], data_copy_params, {}); \
                    input_t_que.EnQue(input); \
                } \
                { \
                    LocalTensor<T> input = input_t_que.DeQue<T>(); \
                    LocalTensor<T> output = output_t_que.AllocTensor<T>(); \
                    Mul(output, input, scale, _); \
                    input_t_que.FreeTensor(input); \
                    output_t_que.EnQue(output); \
                } \
                { \
                    LocalTensor<T> output = output_t_que.DeQue<T>(); \
                    DataCopyPad(output_global_tensor[j * tiling.mid_size + i], output, data_copy_params); \
                    output_t_que.FreeTensor(output); \
                } \
            } \
            scale_t_que.FreeTensor(scale); \
        } \
    } \
    else \
    { \
        for (int i = mid_start; i < mid_end; i += MAX_TILE_SIZE) \
        { \
            int _ = min(mid_end - i, MAX_TILE_SIZE); \
            { \
                LocalTensor<T> scale = scale_t_que.AllocTensor<T>(); \
                DataCopyParams data_copy_params; \
                data_copy_params.blockLen = _ * sizeof(T); \
                DataCopyPad(scale, scale_global_tensor[i], data_copy_params, {}); \
                scale_t_que.EnQue(scale); \
            } \
            LocalTensor<T> scale = scale_t_que.DeQue<T>(); \
            for (int j = 0; j < _; j++) \
            { \
                T scale_value = scale(j); \
                for (int k = 0; k < tiling.high_size; k++) \
                { \
                    for (int l = 0; l < tiling.low_size; l += MAX_TILE_SIZE) \
                    { \
                        int _ = min(tiling.low_size - l, MAX_TILE_SIZE); \
                        DataCopyParams data_copy_params; \
                        data_copy_params.blockLen = _ * sizeof(T); \
                        { \
                            LocalTensor<T> input = input_t_que.AllocTensor<T>(); \
                            DataCopyPad(input, input_global_tensor[(k * tiling.mid_size + i + j) * tiling.low_size + l], data_copy_params, {}); \
                            input_t_que.EnQue(input); \
                        } \
                        { \
                            LocalTensor<T> input = input_t_que.DeQue<T>(); \
                            LocalTensor<T> output = output_t_que.AllocTensor<T>(); \
                            Muls(output, input, scale_value, _); \
                            input_t_que.FreeTensor(input); \
                            output_t_que.EnQue(output); \
                        } \
                        { \
                            LocalTensor<T> output = output_t_que.DeQue<T>(); \
                            DataCopyPad(output_global_tensor[(k * tiling.mid_size + i + j) * tiling.low_size + l], output, data_copy_params); \
                            output_t_que.FreeTensor(output); \
                        } \
                    } \
                } \
            } \
            scale_t_que.FreeTensor(scale); \
        } \
    }

#define scale_5_1() \
    if (tiling.low_size == 1) \
    { \
        for (int i = mid_start; i < mid_end; i += MAX_TILE_SIZE) \
        { \
            int _ = min(mid_end - i, MAX_TILE_SIZE); \
            DataCopyParams data_copy_params; \
            data_copy_params.blockLen = _ * sizeof(T); \
            { \
                LocalTensor<T> scale = scale_t_que.AllocTensor<T>(); \
                DataCopyPad(scale[MAX_TILE_SIZE], scale_global_tensor[i], data_copy_params, {}); \
                scale_t_que.EnQue(scale); \
            } \
            LocalTensor<U> scale = scale_t_que.DeQue<U>(); \
            Cast(scale, scale.ReinterpretCast<T>()[MAX_TILE_SIZE], RoundMode::CAST_NONE, _); \
            for (int j = 0; j < tiling.high_size; j++) \
            { \
                { \
                    LocalTensor<T> input = input_t_que.AllocTensor<T>(); \
                    DataCopyPad(input, input_global_tensor[j * tiling.mid_size + i], data_copy_params, {}); \
                    input_t_que.EnQue(input); \
                } \
                { \
                    LocalTensor<U> input = input_t_que.DeQue<U>(); \
                    LocalTensor<U> output = output_t_que.AllocTensor<U>(); \
                    Cast(output, input.ReinterpretCast<T>(), RoundMode::CAST_NONE, _); \
                    Mul(input, output, scale, _); \
                    Cast(output.ReinterpretCast<T>(), input, RoundMode::CAST_RINT, _); \
                    input_t_que.FreeTensor(input); \
                    output_t_que.EnQue(output); \
                } \
                { \
                    LocalTensor<T> output = output_t_que.DeQue<T>(); \
                    DataCopyPad(output_global_tensor[j * tiling.mid_size + i], output, data_copy_params); \
                    output_t_que.FreeTensor(output); \
                } \
            } \
            scale_t_que.FreeTensor(scale); \
        } \
    } \
    else \
    { \
        for (int i = mid_start; i < mid_end; i += MAX_TILE_SIZE) \
        { \
            int _ = min(mid_end - i, MAX_TILE_SIZE); \
            { \
                LocalTensor<T> scale = scale_t_que.AllocTensor<T>(); \
                DataCopyParams data_copy_params; \
                data_copy_params.blockLen = _ * sizeof(T); \
                DataCopyPad(scale[MAX_TILE_SIZE], scale_global_tensor[i], data_copy_params, {}); \
                scale_t_que.EnQue(scale); \
            } \
            LocalTensor<U> scale = scale_t_que.DeQue<U>(); \
            Cast(scale, scale.ReinterpretCast<T>()[MAX_TILE_SIZE], RoundMode::CAST_NONE, _); \
            for (int j = 0; j < _; j++) \
            { \
                U scale_value = scale(j); \
                for (int k = 0; k < tiling.high_size; k++) \
                { \
                    for (int l = 0; l < tiling.low_size; l += MAX_TILE_SIZE) \
                    { \
                        int _ = min(tiling.low_size - l, MAX_TILE_SIZE); \
                        DataCopyParams data_copy_params; \
                        data_copy_params.blockLen = _ * sizeof(T); \
                        { \
                            LocalTensor<T> input = input_t_que.AllocTensor<T>(); \
                            DataCopyPad(input, input_global_tensor[(k * tiling.mid_size + i + j) * tiling.low_size + l], data_copy_params, {}); \
                            input_t_que.EnQue(input); \
                        } \
                        { \
                            LocalTensor<U> input = input_t_que.DeQue<U>(); \
                            LocalTensor<U> output = output_t_que.AllocTensor<U>(); \
                            Cast(output, input.ReinterpretCast<T>(), RoundMode::CAST_NONE, _); \
                            Muls(input, output, scale_value, _); \
                            Cast(output.ReinterpretCast<T>(), input, RoundMode::CAST_RINT, _); \
                            input_t_que.FreeTensor(input); \
                            output_t_que.EnQue(output); \
                        } \
                        { \
                            LocalTensor<T> output = output_t_que.DeQue<T>(); \
                            DataCopyPad(output_global_tensor[(k * tiling.mid_size + i + j) * tiling.low_size + l], output, data_copy_params); \
                            output_t_que.FreeTensor(output); \
                        } \
                    } \
                } \
            } \
            scale_t_que.FreeTensor(scale); \
        } \
    }

#define scale_5_2() \
    if (tiling.low_size == 1) \
    { \
        for (int i = mid_start; i < mid_end; i += MAX_TILE_SIZE) \
        { \
            int _ = min(mid_end - i, MAX_TILE_SIZE); \
            DataCopyParams data_copy_params; \
            data_copy_params.blockLen = _ * sizeof(T); \
            { \
                LocalTensor<T> scale = scale_t_que.AllocTensor<T>(); \
                LocalTensor<T> bias = bias_t_que.AllocTensor<T>(); \
                DataCopyPad(scale, scale_global_tensor[i], data_copy_params, {}); \
                DataCopyPad(bias, bias_global_tensor[i], data_copy_params, {}); \
                scale_t_que.EnQue(scale); \
                bias_t_que.EnQue(bias); \
            } \
            LocalTensor<T> scale = scale_t_que.DeQue<T>(); \
            LocalTensor<T> bias = bias_t_que.DeQue<T>(); \
            for (int j = 0; j < tiling.high_size; j++) \
            { \
                { \
                    LocalTensor<T> input = input_t_que.AllocTensor<T>(); \
                    DataCopyPad(input, input_global_tensor[j * tiling.mid_size + i], data_copy_params, {}); \
                    input_t_que.EnQue(input); \
                } \
                { \
                    LocalTensor<T> input = input_t_que.DeQue<T>(); \
                    LocalTensor<T> output = output_t_que.AllocTensor<T>(); \
                    Mul(output, input, scale, _); \
                    Add(output, output, bias, _); \
                    input_t_que.FreeTensor(input); \
                    output_t_que.EnQue(output); \
                } \
                { \
                    LocalTensor<T> output = output_t_que.DeQue<T>(); \
                    DataCopyPad(output_global_tensor[j * tiling.mid_size + i], output, data_copy_params); \
                    output_t_que.FreeTensor(output); \
                } \
            } \
            scale_t_que.FreeTensor(scale); \
            bias_t_que.FreeTensor(bias); \
        } \
    } \
    else \
    { \
        for (int i = mid_start; i < mid_end; i += MAX_TILE_SIZE) \
        { \
            int _ = min(mid_end - i, MAX_TILE_SIZE); \
            { \
                LocalTensor<T> scale = scale_t_que.AllocTensor<T>(); \
                LocalTensor<T> bias = bias_t_que.AllocTensor<T>(); \
                DataCopyParams data_copy_params; \
                data_copy_params.blockLen = _ * sizeof(T); \
                DataCopyPad(scale, scale_global_tensor[i], data_copy_params, {}); \
                DataCopyPad(bias, bias_global_tensor[i], data_copy_params, {}); \
                scale_t_que.EnQue(scale); \
                bias_t_que.EnQue(bias); \
            } \
            LocalTensor<T> scale = scale_t_que.DeQue<T>(); \
            LocalTensor<T> bias = bias_t_que.DeQue<T>(); \
            for (int j = 0; j < _; j++) \
            { \
                T scale_value = scale(j); \
                T bias_value = bias(j); \
                for (int k = 0; k < tiling.high_size; k++) \
                { \
                    for (int l = 0; l < tiling.low_size; l += MAX_TILE_SIZE) \
                    { \
                        int _ = min(tiling.low_size - l, MAX_TILE_SIZE); \
                        DataCopyParams data_copy_params; \
                        data_copy_params.blockLen = _ * sizeof(T); \
                        { \
                            LocalTensor<T> input = input_t_que.AllocTensor<T>(); \
                            DataCopyPad(input, input_global_tensor[(k * tiling.mid_size + i + j) * tiling.low_size + l], data_copy_params, {}); \
                            input_t_que.EnQue(input); \
                        } \
                        { \
                            LocalTensor<T> input = input_t_que.DeQue<T>(); \
                            LocalTensor<T> output = output_t_que.AllocTensor<T>(); \
                            Muls(output, input, scale_value, _); \
                            Adds(output, output, bias_value, _); \
                            input_t_que.FreeTensor(input); \
                            output_t_que.EnQue(output); \
                        } \
                        { \
                            LocalTensor<T> output = output_t_que.DeQue<T>(); \
                            DataCopyPad(output_global_tensor[(k * tiling.mid_size + i + j) * tiling.low_size + l], output, data_copy_params); \
                            output_t_que.FreeTensor(output); \
                        } \
                    } \
                } \
            } \
            scale_t_que.FreeTensor(scale); \
            bias_t_que.FreeTensor(bias); \
        } \
    }

#define scale_5_3() \
    if (tiling.low_size == 1) \
    { \
        for (int i = mid_start; i < mid_end; i += MAX_TILE_SIZE) \
        { \
            int _ = min(mid_end - i, MAX_TILE_SIZE); \
            DataCopyParams data_copy_params; \
            data_copy_params.blockLen = _ * sizeof(T); \
            { \
                LocalTensor<T> scale = scale_t_que.AllocTensor<T>(); \
                LocalTensor<T> bias = bias_t_que.AllocTensor<T>(); \
                DataCopyPad(scale[MAX_TILE_SIZE], scale_global_tensor[i], data_copy_params, {}); \
                DataCopyPad(bias[MAX_TILE_SIZE], bias_global_tensor[i], data_copy_params, {}); \
                scale_t_que.EnQue(scale); \
                bias_t_que.EnQue(bias); \
            } \
            LocalTensor<U> scale = scale_t_que.DeQue<U>(); \
            LocalTensor<U> bias = bias_t_que.DeQue<U>(); \
            Cast(scale, scale.ReinterpretCast<T>()[MAX_TILE_SIZE], RoundMode::CAST_NONE, _); \
            Cast(bias, bias.ReinterpretCast<T>()[MAX_TILE_SIZE], RoundMode::CAST_NONE, _); \
            for (int j = 0; j < tiling.high_size; j++) \
            { \
                { \
                    LocalTensor<T> input = input_t_que.AllocTensor<T>(); \
                    DataCopyPad(input, input_global_tensor[j * tiling.mid_size + i], data_copy_params, {}); \
                    input_t_que.EnQue(input); \
                } \
                { \
                    LocalTensor<U> input = input_t_que.DeQue<U>(); \
                    LocalTensor<U> output = output_t_que.AllocTensor<U>(); \
                    LocalTensor<T> output_T = output.ReinterpretCast<T>(); \
                    Cast(output, input.ReinterpretCast<T>(), RoundMode::CAST_NONE, _); \
                    Mul(input, output, scale, _); \
                    Cast(output_T, input, RoundMode::CAST_RINT, _); \
                    Cast(input, output_T, RoundMode::CAST_NONE, _); \
                    Add(input, input, bias, _); \
                    Cast(output_T, input, RoundMode::CAST_RINT, _); \
                    input_t_que.FreeTensor(input); \
                    output_t_que.EnQue(output); \
                } \
                { \
                    LocalTensor<T> output = output_t_que.DeQue<T>(); \
                    DataCopyPad(output_global_tensor[j * tiling.mid_size + i], output, data_copy_params); \
                    output_t_que.FreeTensor(output); \
                } \
            } \
            scale_t_que.FreeTensor(scale); \
            bias_t_que.FreeTensor(bias); \
        } \
    } \
    else \
    { \
        for (int i = mid_start; i < mid_end; i += MAX_TILE_SIZE) \
        { \
            int _ = min(mid_end - i, MAX_TILE_SIZE); \
            { \
                LocalTensor<T> scale = scale_t_que.AllocTensor<T>(); \
                LocalTensor<T> bias = bias_t_que.AllocTensor<T>(); \
                DataCopyParams data_copy_params; \
                data_copy_params.blockLen = _ * sizeof(T); \
                DataCopyPad(scale[MAX_TILE_SIZE], scale_global_tensor[i], data_copy_params, {}); \
                DataCopyPad(bias[MAX_TILE_SIZE], bias_global_tensor[i], data_copy_params, {}); \
                scale_t_que.EnQue(scale); \
                bias_t_que.EnQue(bias); \
            } \
            LocalTensor<U> scale = scale_t_que.DeQue<U>(); \
            LocalTensor<U> bias = bias_t_que.DeQue<U>(); \
            Cast(scale, scale.ReinterpretCast<T>()[MAX_TILE_SIZE], RoundMode::CAST_NONE, _); \
            Cast(bias, bias.ReinterpretCast<T>()[MAX_TILE_SIZE], RoundMode::CAST_NONE, _); \
            for (int j = 0; j < _; j++) \
            { \
                U scale_value = scale(j); \
                U bias_value = bias(j); \
                for (int k = 0; k < tiling.high_size; k++) \
                { \
                    for (int l = 0; l < tiling.low_size; l += MAX_TILE_SIZE) \
                    { \
                        int _ = min(tiling.low_size - l, MAX_TILE_SIZE); \
                        DataCopyParams data_copy_params; \
                        data_copy_params.blockLen = _ * sizeof(T); \
                        { \
                            LocalTensor<T> input = input_t_que.AllocTensor<T>(); \
                            DataCopyPad(input, input_global_tensor[(k * tiling.mid_size + i + j) * tiling.low_size + l], data_copy_params, {}); \
                            input_t_que.EnQue(input); \
                        } \
                        { \
                            LocalTensor<U> input = input_t_que.DeQue<U>(); \
                            LocalTensor<U> output = output_t_que.AllocTensor<U>(); \
                            LocalTensor<T> output_T = output.ReinterpretCast<T>(); \
                            Cast(output, input.ReinterpretCast<T>(), RoundMode::CAST_NONE, _); \
                            Muls(input, output, scale_value, _); \
                            Cast(output_T, input, RoundMode::CAST_RINT, _); \
                            Cast(input, output_T, RoundMode::CAST_NONE, _); \
                            Adds(input, input, bias_value, _); \
                            Cast(output.ReinterpretCast<T>(), input, RoundMode::CAST_RINT, _); \
                            input_t_que.FreeTensor(input); \
                            output_t_que.EnQue(output); \
                        } \
                        { \
                            LocalTensor<T> output = output_t_que.DeQue<T>(); \
                            DataCopyPad(output_global_tensor[(k * tiling.mid_size + i + j) * tiling.low_size + l], output, data_copy_params); \
                            output_t_que.FreeTensor(output); \
                        } \
                    } \
                } \
            } \
            scale_t_que.FreeTensor(scale); \
            bias_t_que.FreeTensor(bias); \
        } \
    }

#define scale_5_4() \
    { \
        LocalTensor<T> scale = scale_t_que.AllocTensor<T>(); \
        LocalTensor<T> bias = bias_t_que.AllocTensor<T>(); \
        DataCopy(scale, scale_global_tensor, round_mid_size); \
        DataCopy(bias, bias_global_tensor, round_mid_size); \
        scale_t_que.EnQue(scale); \
        bias_t_que.EnQue(bias); \
    } \
    { \
        LocalTensor<T> scale = scale_t_que.DeQue<T>(); \
        LocalTensor<T> bias = bias_t_que.DeQue<T>(); \
        if (max_high_tile_size > 1) \
        { \
            int _ = max_high_tile_size * round_mid_size; \
            CopyRepeatParams copy_repeat_params{1, 1, 8, 8}; \
            constexpr int mask = 256 / sizeof(T); \
            int current_size; \
            for (current_size = round_mid_size; current_size < _ / 2; current_size *= 2) \
            { \
                int repeat_time = ceil_div(current_size, mask); \
                Copy(scale[current_size], scale, mask, repeat_time, copy_repeat_params); \
                Copy(bias[current_size], bias, mask, repeat_time, copy_repeat_params); \
            } \
            int repeat_time = ceil_div(_ - current_size, mask); \
            Copy(scale[current_size], scale, mask, repeat_time, copy_repeat_params); \
            Copy(bias[current_size], bias, mask, repeat_time, copy_repeat_params); \
        } \
        for (int i = loop_start; i != loop_end; i += loop_step) \
        { \
            int high_tile_size = min(high_end - i, max_high_tile_size); \
            int _ = high_tile_size * round_mid_size; \
            DataCopyParams data_copy_params; \
            data_copy_params.blockCount = high_tile_size; \
            data_copy_params.blockLen = tiling.mid_size * sizeof(T); \
            { \
                LocalTensor<T> input = input_t_que.AllocTensor<T>(); \
                DataCopyPad(input, input_global_tensor[i * tiling.mid_size], data_copy_params, {}); \
                input_t_que.EnQue(input); \
            } \
            { \
                LocalTensor<T> input = input_t_que.DeQue<T>(); \
                LocalTensor<T> output = output_t_que.AllocTensor<T>(); \
                Mul(output, input, scale, _); \
                Add(output, output, bias, _); \
                input_t_que.FreeTensor(input); \
                output_t_que.EnQue(output); \
            } \
            { \
                LocalTensor<T> output = output_t_que.DeQue<T>(); \
                DataCopyPad(output_global_tensor[i * tiling.mid_size], output, data_copy_params); \
                output_t_que.FreeTensor(output); \
            } \
        } \
        scale_t_que.FreeTensor(scale); \
        bias_t_que.FreeTensor(bias); \
    }

#define scale_5_5() \
    { \
        LocalTensor<T> scale = scale_t_que.AllocTensor<T>(); \
        LocalTensor<T> bias = bias_t_que.AllocTensor<T>(); \
        int round_mid_size = ceil_round(_, DATA_BLOCK_SIZE); \
        DataCopy(scale[MAX_TILE_SIZE], scale_global_tensor[mid_offset], round_mid_size); \
        DataCopy(bias, bias_global_tensor[mid_offset], round_mid_size); \
        scale_t_que.EnQue(scale); \
        bias_t_que.EnQue(bias); \
    } \
    { \
        LocalTensor<U> scale = scale_t_que.DeQue<U>(); \
        LocalTensor<T> bias = bias_t_que.DeQue<T>(); \
        Cast(scale, scale.ReinterpretCast<T>()[MAX_TILE_SIZE], RoundMode::CAST_NONE, _); \
        for (int i = loop_start; i != loop_end; i += loop_step) \
        { \
            DataCopyParams data_copy_params; \
            data_copy_params.blockLen = _ * sizeof(T); \
            { \
                LocalTensor<T> input = input_t_que.AllocTensor<T>(); \
                DataCopyPad(input, input_global_tensor[i * tiling.mid_size + mid_offset], data_copy_params, {}); \
                input_t_que.EnQue(input); \
            } \
            { \
                LocalTensor<U> input = input_t_que.DeQue<U>(); \
                LocalTensor<U> output = output_t_que.AllocTensor<U>(); \
                LocalTensor<T> output_T = output.ReinterpretCast<T>(); \
                Cast(output, input.ReinterpretCast<T>(), RoundMode::CAST_NONE, _); \
                Mul(input, output, scale, _); \
                Cast(output.ReinterpretCast<T>(), input, RoundMode::CAST_RINT, _); \
                input_t_que.FreeTensor(input); \
                output_t_que.EnQue(output); \
            } \
            { \
                LocalTensor<T> output = output_t_que.DeQue<T>(); \
                DataCopyPad(output_global_tensor[i * tiling.mid_size + mid_offset], output, data_copy_params); \
                output_t_que.FreeTensor(output); \
                PipeBarrier<PIPE_MTE3>(); \
                SetAtomicAdd<T>(); \
                DataCopyPad(output_global_tensor[i * tiling.mid_size + mid_offset], bias, data_copy_params); \
                SetAtomicNone(); \
            } \
        } \
        scale_t_que.FreeTensor(scale); \
        bias_t_que.FreeTensor(bias); \
    }

#define scale_5_6() \
    for (int i = loop_start; i != loop_end; i += loop_step) \
    { \
        int mid_tile_size = min(compute_end - i, max_mid_tile_size); \
        int _ = mid_tile_size * tiling.low_size; \
        { \
            LocalTensor<T> input = input_t_que.AllocTensor<T>(); \
            DataCopy(input, input_global_tensor[i * tiling.low_size], _); \
            input_t_que.EnQue(input); \
        } \
        { \
            LocalTensor<T> input = input_t_que.DeQue<T>(); \
            LocalTensor<T> output = output_t_que.AllocTensor<T>(); \
            for (int j = 0; j < mid_tile_size; j++) \
            { \
                T scale = scale_global_tensor((i + j) % tiling.mid_size); \
                Muls(output[j * tiling.low_size], input[j * tiling.low_size], scale, tiling.low_size); \
            } \
            for (int j = 0; j < mid_tile_size; j++) \
            { \
                T bias = bias_global_tensor((i + j) % tiling.mid_size); \
                Adds(output[j * tiling.low_size], output[j * tiling.low_size], bias, tiling.low_size); \
            } \
            input_t_que.FreeTensor(input); \
            output_t_que.EnQue(output); \
        } \
        { \
            LocalTensor<T> output = output_t_que.DeQue<T>(); \
            DataCopy(output_global_tensor[i * tiling.low_size], output, _); \
            output_t_que.FreeTensor(output); \
        } \
    }

#define scale_5_7() \
    for (int i = mid_loop_start; i != mid_loop_end; i += mid_loop_step) \
    { \
        int mid_tile_size = min(mid_end - i, max_mid_tile_size); \
        int _ = mid_tile_size * tiling.low_size; \
        { \
            LocalTensor<T> scale = scale_t_que.AllocTensor<T>(); \
            LocalTensor<T> bias = bias_t_que.AllocTensor<T>(); \
            DataCopy(scale, scale_global_tensor[i], mid_tile_size); \
            DataCopy(bias, bias_global_tensor[i], mid_tile_size); \
            scale_t_que.EnQue(scale); \
            bias_t_que.EnQue(bias); \
        } \
        LocalTensor<T> scale = scale_t_que.DeQue<T>(); \
        LocalTensor<T> bias = bias_t_que.DeQue<T>(); \
        for (int j = high_loop_start; j != high_loop_end; j += high_loop_step) \
        { \
            LocalTensor<T> output = output_t_que.AllocTensor<T>(); \
            DataCopyParams data_copy_params; \
            data_copy_params.blockLen = _ * sizeof(T); \
            { \
                LocalTensor<T> input = input_t_que.AllocTensor<T>(); \
                DataCopy(input, input_global_tensor[(j * tiling.mid_size + i) * tiling.low_size], _); \
                input_t_que.EnQue(input); \
            } \
            if (j == high_loop_start) \
            { \
                int brcb_repeat_time = ceil_div(mid_tile_size, 8); \
                BrcbRepeatParams brcb_repeat_params; \
                Brcb(output, scale, brcb_repeat_time, brcb_repeat_params); \
                Brcb(scale, bias, brcb_repeat_time, brcb_repeat_params); \
                CopyRepeatParams copy_repeat_params_1{1, 0, static_cast<uint16_t>(tiling.broadcast_stride[0]), 1}; \
                constexpr int mask = 256 / sizeof(T); \
                Copy(bias, output, mask, mid_tile_size, copy_repeat_params_1); \
                Copy(output, scale, mask, mid_tile_size, copy_repeat_params_1); \
                int copy_repeat_time_2 = mid_tile_size * tiling.broadcast_stride[0]; \
                CopyRepeatParams copy_repeat_params_2{1, 0, static_cast<uint16_t>(tiling.broadcast_stride[1]), 1}; \
                Copy(scale, bias, mask, copy_repeat_time_2, copy_repeat_params_2); \
                Copy(bias, output, mask, copy_repeat_time_2, copy_repeat_params_2); \
                GatherMaskParams gather_mask_params; \
                gather_mask_params.repeatTimes = mid_tile_size; \
                gather_mask_params.src0RepeatStride = tiling.broadcast_size / DATA_BLOCK_SIZE; \
                uint64_t rsvd_cnt; \
                GatherMask(scale, scale, 7, true, tiling.low_size, gather_mask_params, rsvd_cnt); \
                GatherMask(bias, bias, 7, true, tiling.low_size, gather_mask_params, rsvd_cnt); \
            } \
            { \
                LocalTensor<T> input = input_t_que.DeQue<T>(); \
                Mul(input, input, scale, _); \
                Add(output, input, bias, _); \
                input_t_que.FreeTensor(input); \
                output_t_que.EnQue(output); \
            } \
            { \
                LocalTensor<T> output = output_t_que.DeQue<T>(); \
                DataCopy(output_global_tensor[(j * tiling.mid_size + i) * tiling.low_size], output, _); \
                output_t_que.FreeTensor(output); \
            } \
        } \
        scale_t_que.FreeTensor(scale); \
        bias_t_que.FreeTensor(bias); \
    }

#define scale_exec(a, b, c, d, e, f) \
    scale_0_##a() \
    scale_1_##b() \
    scale_2_##c() \
    scale_3_##d() \
    scale_4_##e() \
    scale_5_##f()

template<typename T, int EnBias>
__aicore__ inline void scale(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, ScaleTilingData &tiling);

template<>
__aicore__ inline void scale<half, 0>(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, ScaleTilingData &tiling)
{
    using T = half;
    scale_exec(0, 0, 0, 0, 0, 0);
}

template<>
__aicore__ inline void scale<bfloat16_t, 0>(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, ScaleTilingData &tiling)
{
    using T = bfloat16_t;
    using U = float;
    scale_exec(0, 0, 0, 1, 0, 1);
}

template<>
__aicore__ inline void scale<float, 0>(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, ScaleTilingData &tiling)
{
    using T = float;
    scale_exec(0, 0, 0, 0, 0, 0);
}

template<>
__aicore__ inline void scale<half, 1>(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, ScaleTilingData &tiling)
{
    using T = half;
    scale_exec(0, 1, 1, 2, 0, 2);
}

template<>
__aicore__ inline void scale<bfloat16_t, 1>(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, ScaleTilingData &tiling)
{
    using T = bfloat16_t;
    using U = float;
    scale_exec(0, 1, 1, 3, 0, 3);
}

template<>
__aicore__ inline void scale<float, 1>(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, ScaleTilingData &tiling)
{
    using T = float;
    scale_exec(0, 1, 1, 2, 0, 2);
}

template<>
__aicore__ inline void scale<half, 2>(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, ScaleTilingData &tiling)
{
    using T = half;
    scale_exec(0, 1, 1, 2, 0, 2);
}

template<>
__aicore__ inline void scale<bfloat16_t, 2>(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, ScaleTilingData &tiling)
{
    using T = bfloat16_t;
    using U = float;
    scale_exec(0, 1, 1, 3, 0, 3);
}

template<>
__aicore__ inline void scale<float, 2>(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, ScaleTilingData &tiling)
{
    using T = float;
    scale_exec(1, 1, 1, 4, 1, 4);
}

template<>
__aicore__ inline void scale<half, 3>(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, ScaleTilingData &tiling)
{
    using T = half;
    scale_exec(0, 1, 1, 2, 0, 2);
}

template<>
__aicore__ inline void scale<bfloat16_t, 3>(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, ScaleTilingData &tiling)
{
    using T = bfloat16_t;
    using U = float;
    scale_exec(2, 1, 1, 5, 2, 5);
}

template<>
__aicore__ inline void scale<float, 3>(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, ScaleTilingData &tiling)
{
    using T = float;
    scale_exec(0, 1, 1, 2, 0, 2);
}

template<>
__aicore__ inline void scale<half, 4>(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, ScaleTilingData &tiling)
{
    using T = half;
    scale_exec(3, 1, 2, 6, 3, 6);
}

template<>
__aicore__ inline void scale<bfloat16_t, 4>(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, ScaleTilingData &tiling)
{
    using T = bfloat16_t;
    using U = float;
    scale_exec(0, 1, 1, 3, 0, 3);
}

template<>
__aicore__ inline void scale<float, 4>(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, ScaleTilingData &tiling)
{
    using T = float;
    scale_exec(0, 1, 1, 2, 0, 2);
}

template<>
__aicore__ inline void scale<half, 5>(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, ScaleTilingData &tiling)
{
    using T = half;
    scale_exec(4, 1, 1, 7, 4, 7);
}

template<>
__aicore__ inline void scale<bfloat16_t, 5>(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, ScaleTilingData &tiling)
{
    using T = bfloat16_t;
    using U = float;
    scale_exec(0, 1, 1, 3, 0, 3);
}

template<>
__aicore__ inline void scale<float, 5>(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, ScaleTilingData &tiling)
{
    using T = float;
    scale_exec(0, 1, 1, 2, 0, 2);
}

extern "C" __global__ __aicore__ void scale(GM_ADDR input, GM_ADDR scale, GM_ADDR bias, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(0))
        ::scale<DTYPE_INPUT, 0>(input, scale, bias, output, tiling_data);
    else if (TILING_KEY_IS(1))
        ::scale<DTYPE_INPUT, 1>(input, scale, bias, output, tiling_data);
    else if (TILING_KEY_IS(2))
        ::scale<DTYPE_INPUT, 2>(input, scale, bias, output, tiling_data);
    else if (TILING_KEY_IS(3))
        ::scale<DTYPE_INPUT, 3>(input, scale, bias, output, tiling_data);
    else if (TILING_KEY_IS(4))
        ::scale<DTYPE_INPUT, 4>(input, scale, bias, output, tiling_data);
    else if (TILING_KEY_IS(5))
        ::scale<DTYPE_INPUT, 5>(input, scale, bias, output, tiling_data);
}
