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

#define atanh_0_0() \
    int block_index = GetBlockIdx(); \
    int block_dim = GetBlockNum(); \
    constexpr int MTE_BLOCK_SIZE = 512 / sizeof(T); \
    int compute_blocks = ceil_div(tiling.size, MTE_BLOCK_SIZE); \
    int compute_start = compute_blocks * block_index / block_dim * MTE_BLOCK_SIZE; \
    int compute_end = min(compute_blocks * (block_index + 1) / block_dim * MTE_BLOCK_SIZE, tiling.size);

#define atanh_1_0() \
    GlobalTensor<T> input_global_tensor, output_global_tensor; \
    input_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input)); \
    output_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output));

#define atanh_2_0() \
    TPipe t_pipe; \
    TQue<TPosition::VECIN, 1> input_t_que; \
    TQue<TPosition::VECOUT, 1> output_t_que;

#define atanh_3_0() \
    constexpr int MAX_TILE_SIZE = 24064; \
    t_pipe.InitBuffer(input_t_que, 2, 48256); \
    t_pipe.InitBuffer(output_t_que, 2, 48256);

#define atanh_3_1() \
    constexpr int MAX_TILE_SIZE = 12032; \
    t_pipe.InitBuffer(input_t_que, 2, 48256); \
    t_pipe.InitBuffer(output_t_que, 2, 48256);

#define atanh_4_0() \
    int loop_start, loop_end, loop_step; \
    uint16_t atomic_type, atomic_op; \
    GetStoreAtomicConfig(atomic_type, atomic_op); \
    if (atomic_type) \
    { \
        loop_start = compute_start + ceil_round(compute_end - compute_start, MAX_TILE_SIZE) - MAX_TILE_SIZE; \
        loop_end = compute_start - MAX_TILE_SIZE; \
        loop_step = -MAX_TILE_SIZE; \
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_NONE, AtomicOp::ATOMIC_SUM>(); \
    } \
    else \
    { \
        loop_start = compute_start; \
        loop_end = compute_start + ceil_round(compute_end - compute_start, MAX_TILE_SIZE); \
        loop_step = MAX_TILE_SIZE; \
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_S32, AtomicOp::ATOMIC_SUM>(); \
    }

#define atanh_4_1() \
    TEventID event_id = t_pipe.AllocEventID<HardEvent::MTE3_S>(); \
    for (int h = 0; h < 1000; h++) \
    { \
        int loop_start, loop_end, loop_step; \
        uint16_t atomic_type, atomic_op; \
        GetStoreAtomicConfig(atomic_type, atomic_op); \
        if (atomic_type) \
        { \
            loop_start = compute_start + ceil_round(compute_end - compute_start, MAX_TILE_SIZE) - MAX_TILE_SIZE; \
            loop_end = compute_start - MAX_TILE_SIZE; \
            loop_step = -MAX_TILE_SIZE; \
            SetStoreAtomicConfig<AtomicDtype::ATOMIC_NONE, AtomicOp::ATOMIC_SUM>(); \
        } \
        else \
        { \
            loop_start = compute_start; \
            loop_end = compute_start + ceil_round(compute_end - compute_start, MAX_TILE_SIZE); \
            loop_step = MAX_TILE_SIZE; \
            SetStoreAtomicConfig<AtomicDtype::ATOMIC_S32, AtomicOp::ATOMIC_SUM>(); \
        }

#define atanh_5_0() \
    for (int i = loop_start; i != loop_end; i += loop_step) \
    { \
        int _ = min(compute_end - i, MAX_TILE_SIZE); \
        { \
            LocalTensor<T> input = input_t_que.AllocTensor<T>(); \
            DataCopy(input, input_global_tensor[i], _); \
            input_t_que.EnQue(input); \
        } \
        { \
            LocalTensor<T> input = input_t_que.DeQue<T>(); \
            LocalTensor<T> output = output_t_que.AllocTensor<T>(); \
            Duplicate(output, static_cast<T>(1), _); \
            Sub(output, output, input, _); \
            Adds(input, input, static_cast<T>(1), _); \
            Div(output, input, output, _); \
            Ln(input, output, _); \
            Muls(output, input, static_cast<T>(0.5), _); \
            input_t_que.FreeTensor(input); \
            output_t_que.EnQue(output); \
        } \
        { \
            LocalTensor<T> output = output_t_que.DeQue<T>(); \
            DataCopy(output_global_tensor[i], output, _); \
            output_t_que.FreeTensor(output); \
        } \
    }

#define atanh_5_1() \
    for (int i = loop_start; i != loop_end; i += loop_step) \
    { \
        int _ = min(compute_end - i, MAX_TILE_SIZE); \
        { \
            LocalTensor<T> input = input_t_que.AllocTensor<T>(); \
            DataCopy(input, input_global_tensor[i], _); \
            input_t_que.EnQue(input); \
        } \
        { \
            LocalTensor<U> input = input_t_que.DeQue<U>(); \
            LocalTensor<U> output = output_t_que.AllocTensor<U>(); \
            Cast(output, input.ReinterpretCast<T>(), RoundMode::CAST_NONE, _); \
            Duplicate(input, static_cast<U>(1), _); \
            Sub(input, input, output, _); \
            Adds(output, output, static_cast<U>(1), _); \
            Div(input, output, input, _); \
            Ln(output, input, _); \
            Muls(input, output, static_cast<U>(0.5), _); \
            Cast(output.ReinterpretCast<T>(), input, RoundMode::CAST_RINT, _); \
            input_t_que.FreeTensor(input); \
            output_t_que.EnQue(output); \
        } \
        { \
            LocalTensor<T> output = output_t_que.DeQue<T>(); \
            DataCopy(output_global_tensor[i], output, _); \
            output_t_que.FreeTensor(output); \
        } \
    }

#define atanh_5_2() \
        long start = GetSystemCycle(); \
        for (int i = loop_start; i != loop_end; i += loop_step) \
        { \
            int _ = min(compute_end - i, MAX_TILE_SIZE); \
            { \
                LocalTensor<T> input = input_t_que.AllocTensor<T>(); \
                DataCopy(input, input_global_tensor[i], _); \
                input_t_que.EnQue(input); \
            } \
            { \
                LocalTensor<T> input = input_t_que.DeQue<T>(); \
                LocalTensor<T> output = output_t_que.AllocTensor<T>(); \
                Duplicate(output, static_cast<T>(1), _); \
                Sub(output, output, input, _); \
                Adds(input, input, static_cast<T>(1), _); \
                Div(output, input, output, _); \
                Ln(input, output, _); \
                Muls(output, input, static_cast<T>(0.5), _); \
                input_t_que.FreeTensor(input); \
                output_t_que.EnQue(output); \
            } \
            { \
                LocalTensor<T> output = output_t_que.DeQue<T>(); \
                DataCopy(output_global_tensor[i], output, _); \
                output_t_que.FreeTensor(output); \
            } \
        } \
        long threshold = tiling.size * sizeof(T) * 6.4e-5f; \
        SetFlag<HardEvent::MTE3_S>(event_id); \
        WaitFlag<HardEvent::MTE3_S>(event_id); \
        long end = GetSystemCycle(); \
        if (end - start < threshold) \
            break; \
    } \
    t_pipe.ReleaseEventID<HardEvent::MTE3_MTE2>(event_id);

#define atanh_5_3() \
        long start = GetSystemCycle(); \
        for (int i = loop_start; i != loop_end; i += loop_step) \
        { \
            int _ = min(compute_end - i, MAX_TILE_SIZE); \
            { \
                LocalTensor<T> input = input_t_que.AllocTensor<T>(); \
                DataCopy(input, input_global_tensor[i], _); \
                input_t_que.EnQue(input); \
            } \
            { \
                LocalTensor<U> input = input_t_que.DeQue<U>(); \
                LocalTensor<U> output = output_t_que.AllocTensor<U>(); \
                Cast(output, input.ReinterpretCast<T>(), RoundMode::CAST_NONE, _); \
                Duplicate(input, static_cast<U>(1), _); \
                Sub(input, input, output, _); \
                Adds(output, output, static_cast<U>(1), _); \
                Div(input, output, input, _); \
                Ln(output, input, _); \
                Muls(input, output, static_cast<U>(0.5), _); \
                Cast(output.ReinterpretCast<T>(), input, RoundMode::CAST_RINT, _); \
                input_t_que.FreeTensor(input); \
                output_t_que.EnQue(output); \
            } \
            { \
                LocalTensor<T> output = output_t_que.DeQue<T>(); \
                DataCopy(output_global_tensor[i], output, _); \
                output_t_que.FreeTensor(output); \
            } \
        } \
        long threshold = tiling.size * sizeof(T) * 1.2e-4f; \
        SetFlag<HardEvent::MTE3_S>(event_id); \
        WaitFlag<HardEvent::MTE3_S>(event_id); \
        long end = GetSystemCycle(); \
        if (end - start < threshold) \
            break; \
    } \
    t_pipe.ReleaseEventID<HardEvent::MTE3_MTE2>(event_id);

#define atanh_5_4() \
        long start = GetSystemCycle(); \
        for (int i = loop_start; i != loop_end; i += loop_step) \
        { \
            int _ = min(compute_end - i, MAX_TILE_SIZE); \
            { \
                LocalTensor<T> input = input_t_que.AllocTensor<T>(); \
                DataCopy(input, input_global_tensor[i], _); \
                input_t_que.EnQue(input); \
            } \
            { \
                LocalTensor<T> input = input_t_que.DeQue<T>(); \
                LocalTensor<T> output = output_t_que.AllocTensor<T>(); \
                Duplicate(output, static_cast<T>(1), _); \
                Sub(output, output, input, _); \
                Adds(input, input, static_cast<T>(1), _); \
                Div(output, input, output, _); \
                Ln(input, output, _); \
                Muls(output, input, static_cast<T>(0.5), _); \
                input_t_que.FreeTensor(input); \
                output_t_que.EnQue(output); \
            } \
            { \
                LocalTensor<T> output = output_t_que.DeQue<T>(); \
                DataCopy(output_global_tensor[i], output, _); \
                output_t_que.FreeTensor(output); \
            } \
        } \
        long threshold = tiling.size * sizeof(T) * 5.1e-5f; \
        SetFlag<HardEvent::MTE3_S>(event_id); \
        WaitFlag<HardEvent::MTE3_S>(event_id); \
        long end = GetSystemCycle(); \
        if (end - start < threshold) \
            break; \
    } \
    t_pipe.ReleaseEventID<HardEvent::MTE3_MTE2>(event_id);

#define atanh_exec(a, b, c, d, e, f) \
    atanh_0_##a() \
    atanh_1_##b() \
    atanh_2_##c() \
    atanh_3_##d() \
    atanh_4_##e() \
    atanh_5_##f()

template<typename T, int TilingKey>
__aicore__ inline void atanh(GM_ADDR input, GM_ADDR output, AtanhTilingData &tiling);

template<>
__aicore__ inline void atanh<half, 0>(GM_ADDR input, GM_ADDR output, AtanhTilingData &tiling)
{
    using T = half;
    atanh_exec(0, 0, 0, 0, 0, 0);
}

template<>
__aicore__ inline void atanh<bfloat16_t, 0>(GM_ADDR input, GM_ADDR output, AtanhTilingData &tiling)
{
    using T = bfloat16_t;
    using U = float;
    atanh_exec(0, 0, 0, 1, 0, 1);
}

template<>
__aicore__ inline void atanh<float, 0>(GM_ADDR input, GM_ADDR output, AtanhTilingData &tiling)
{
    using T = float;
    atanh_exec(0, 0, 0, 1, 0, 0);
}

template<>
__aicore__ inline void atanh<half, 1>(GM_ADDR input, GM_ADDR output, AtanhTilingData &tiling)
{
    using T = half;
    atanh_exec(0, 0, 0, 0, 1, 2);
}

template<>
__aicore__ inline void atanh<bfloat16_t, 1>(GM_ADDR input, GM_ADDR output, AtanhTilingData &tiling)
{
    using T = bfloat16_t;
    using U = float;
    atanh_exec(0, 0, 0, 1, 1, 3);
}

template<>
__aicore__ inline void atanh<float, 1>(GM_ADDR input, GM_ADDR output, AtanhTilingData &tiling)
{
    using T = float;
    atanh_exec(0, 0, 0, 1, 1, 4);
}

extern "C" __global__ __aicore__ void atanh(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(0))
        atanh<DTYPE_INPUT, 0>(input, output, tiling_data);
    else if (TILING_KEY_IS(1))
        atanh<DTYPE_INPUT, 1>(input, output, tiling_data);
}
