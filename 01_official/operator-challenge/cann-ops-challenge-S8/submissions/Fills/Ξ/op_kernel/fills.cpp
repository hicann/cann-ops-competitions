#include "kernel_operator.h"

using namespace AscendC;

template<typename T, typename... Ts>
struct is_one_of : std::false_type {};

template<typename T, typename U, typename... Ts>
struct is_one_of<T, U, Ts...> : std::conditional_t<std::is_same_v<T, U>, std::true_type, is_one_of<T, Ts...>> {};

template<typename T, typename... Ts>
constexpr bool is_one_of_v = is_one_of<T, Ts...>::value;

template<typename T>
__aicore__ inline constexpr T ceil_div(T x, T y)
{
    return (x - 1) / y + 1;
}

#define fills_0_0() \
    int block_index = GetBlockIdx(); \
    int block_dim = GetBlockNum(); \
    constexpr int MTE_BLOCK_SIZE = 512 / sizeof(T); \
    int compute_blocks = ceil_div(tiling.size, MTE_BLOCK_SIZE); \
    int compute_start = compute_blocks * block_index / block_dim * MTE_BLOCK_SIZE; \
    int compute_end = min(compute_blocks * (block_index + 1) / block_dim * MTE_BLOCK_SIZE, tiling.size);

#define fills_1_0() \
    GlobalTensor<T> output_global_tensor; \
    output_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output));

#define fills_2_0() \
    TPipe t_pipe; \
    TQue<TPosition::VECOUT, 1> output_t_que;

#define fills_3_0() \
    constexpr int MAX_TILE_SIZE = (191 << 10) / sizeof(T); \
    t_pipe.InitBuffer(output_t_que, 1, MAX_TILE_SIZE * sizeof(T));

#define fills_4_0() \
    { \
        LocalTensor<T> output = output_t_que.AllocTensor<T>(); \
        Duplicate(output, static_cast<T>(tiling.value), MAX_TILE_SIZE); \
        output_t_que.EnQue(output); \
    }

#define fills_4_1() \
    { \
        LocalTensor<T> output = output_t_que.AllocTensor<T>(); \
        LocalTensor<U> output_U = output.template ReinterpretCast<U>(); \
        output_U(0) = tiling.value; \
        Cast(output, output_U, RoundMode::CAST_RINT, 1); \
        T value = output(0); \
        Duplicate(output, value, MAX_TILE_SIZE); \
        output_t_que.EnQue(output); \
    }

#define fills_4_2() \
    { \
        LocalTensor<U> output = output_t_que.AllocTensor<U>(); \
        T value = static_cast<T>(static_cast<int>(tiling.value)); \
        Duplicate(output, static_cast<U>(value << 8 | value), MAX_TILE_SIZE / 2); \
        output_t_que.EnQue(output); \
    }

#define fills_5_0() \
    { \
        LocalTensor<T> output = output_t_que.DeQue<T>(); \
        for (int i = compute_start; i < compute_end; i += MAX_TILE_SIZE) \
        { \
            int _ = min(compute_end - i, MAX_TILE_SIZE); \
            DataCopy(output_global_tensor[i], output, _); \
        } \
        output_t_que.FreeTensor(output); \
    }

#define fills_5_1() \
    TEventID event_id = t_pipe.AllocEventID<HardEvent::MTE3_S>(); \
    { \
        LocalTensor<T> output = output_t_que.DeQue<T>(); \
        for (int i = 0; i < 1000; i++) \
        { \
            long start = GetSystemCycle(); \
            for (int j = compute_start; j < compute_end; j += MAX_TILE_SIZE) \
            { \
                int _ = min(compute_end - j, MAX_TILE_SIZE); \
                DataCopy(output_global_tensor[j], output, _); \
            } \
            long threshold = tiling.size * sizeof(T) * 1.7e-5f; \
            SetFlag<HardEvent::MTE3_S>(event_id); \
            WaitFlag<HardEvent::MTE3_S>(event_id); \
            long end = GetSystemCycle(); \
            if (end - start < threshold) \
                break; \
        } \
        output_t_que.FreeTensor(output); \
    } \
    t_pipe.ReleaseEventID<HardEvent::MTE3_S>(event_id);

#define fills_exec(a, b, c, d, e, f) \
    fills_0_##a() \
    fills_1_##b() \
    fills_2_##c() \
    fills_3_##d() \
    fills_4_##e() \
    fills_5_##f()

template<typename T, int TilingKey>
__aicore__ inline void fills(GM_ADDR output, FillsTilingData &tiling);

template<>
__aicore__ inline void fills<half, 0>(GM_ADDR output, FillsTilingData &tiling)
{
    using T = half;
    fills_exec(0, 0, 0, 0, 0, 0);
}

template<>
__aicore__ inline void fills<bfloat16_t, 0>(GM_ADDR output, FillsTilingData &tiling)
{
    using T = bfloat16_t;
    using U = float;
    fills_exec(0, 0, 0, 0, 1, 0);
}

template<>
__aicore__ inline void fills<float, 0>(GM_ADDR output, FillsTilingData &tiling)
{
    using T = float;
    fills_exec(0, 0, 0, 0, 0, 0);
}

template<>
__aicore__ inline void fills<int, 0>(GM_ADDR output, FillsTilingData &tiling)
{
    using T = int;
    fills_exec(0, 0, 0, 0, 0, 0);
}

template<>
__aicore__ inline void fills<short, 0>(GM_ADDR output, FillsTilingData &tiling)
{
    using T = short;
    fills_exec(0, 0, 0, 0, 0, 0);
}

template<>
__aicore__ inline void fills<uint8_t, 0>(GM_ADDR output, FillsTilingData &tiling)
{
    using T = uint8_t;
    using U = short;
    fills_exec(0, 0, 0, 0, 2, 0);
}

template<>
__aicore__ inline void fills<int8_t, 0>(GM_ADDR output, FillsTilingData &tiling)
{
    using T = int8_t;
    using U = short;
    fills_exec(0, 0, 0, 0, 2, 0);
}

template<>
__aicore__ inline void fills<half, 1>(GM_ADDR output, FillsTilingData &tiling)
{
    using T = half;
    fills_exec(0, 0, 0, 0, 0, 1);
}

template<>
__aicore__ inline void fills<bfloat16_t, 1>(GM_ADDR output, FillsTilingData &tiling)
{
    using T = bfloat16_t;
    using U = float;
    fills_exec(0, 0, 0, 0, 1, 1);
}

template<>
__aicore__ inline void fills<float, 1>(GM_ADDR output, FillsTilingData &tiling)
{
    using T = float;
    fills_exec(0, 0, 0, 0, 0, 1);
}

template<>
__aicore__ inline void fills<int, 1>(GM_ADDR output, FillsTilingData &tiling)
{
    using T = int;
    fills_exec(0, 0, 0, 0, 0, 1);
}

template<>
__aicore__ inline void fills<short, 1>(GM_ADDR output, FillsTilingData &tiling)
{
    using T = short;
    fills_exec(0, 0, 0, 0, 0, 1);
}

template<>
__aicore__ inline void fills<uint8_t, 1>(GM_ADDR output, FillsTilingData &tiling)
{
    using T = uint8_t;
    using U = short;
    fills_exec(0, 0, 0, 0, 2, 1);
}

template<>
__aicore__ inline void fills<int8_t, 1>(GM_ADDR output, FillsTilingData &tiling)
{
    using T = int8_t;
    using U = short;
    fills_exec(0, 0, 0, 0, 2, 1);
}

extern "C" __global__ __aicore__ void fills(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(0))
        fills<DTYPE_INPUT, 0>(output, tiling_data);
    else if (TILING_KEY_IS(1))
        fills<DTYPE_INPUT, 1>(output, tiling_data);
}
