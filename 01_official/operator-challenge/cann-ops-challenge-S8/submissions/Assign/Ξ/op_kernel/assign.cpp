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
__aicore__ inline void assign(GM_ADDR input, GM_ADDR other, AssignTilingData &tiling)
{
    int block_index = GetBlockIdx();
    int block_dim = GetBlockNum();
    constexpr int MTE_BLOCK_SIZE = 512 / sizeof(T);
    int compute_blocks = ceil_div(tiling.size, MTE_BLOCK_SIZE);
    int compute_start = compute_blocks * block_index / block_dim * MTE_BLOCK_SIZE;
    int compute_end = min(compute_blocks * (block_index + 1) / block_dim * MTE_BLOCK_SIZE, tiling.size);

    GlobalTensor<T> input_global_tensor, other_global_tensor;
    input_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
    other_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(other));

    TPipe t_pipe;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> t_que_bind;

    constexpr int MAX_TILE_SIZE = (191 << 10) / sizeof(T);
    t_pipe.InitBuffer(t_que_bind, 1, MAX_TILE_SIZE * sizeof(T));

    int loop_start, loop_end, loop_step;
    uint16_t atomic_type, atomic_op;
    GetStoreAtomicConfig(atomic_type, atomic_op);
    if (atomic_type)
    {
        loop_start = compute_start + ceil_round(compute_end - compute_start, MAX_TILE_SIZE) - MAX_TILE_SIZE;
        loop_end = compute_start - MAX_TILE_SIZE;
        loop_step = -MAX_TILE_SIZE;
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_S16, AtomicOp::ATOMIC_SUM>();
    }
    else
    {
        loop_start = compute_start;
        loop_end = compute_start + ceil_round(compute_end - compute_start, MAX_TILE_SIZE);
        loop_step = MAX_TILE_SIZE;
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_S32, AtomicOp::ATOMIC_SUM>();
    }

    for (int i = loop_start; i != loop_end; i += loop_step)
    {
        int _ = min(compute_end - i, MAX_TILE_SIZE);
        {
            LocalTensor<T> other = t_que_bind.AllocTensor<T>();
            DataCopy(other, other_global_tensor[i], _);
            t_que_bind.EnQue(other);
        }
        {
            LocalTensor<T> input = t_que_bind.DeQue<T>();
            DataCopy(input_global_tensor[i], input, _);
            t_que_bind.FreeTensor(input);
        }
    }
}

template<typename T>
__aicore__ inline void assign_broadcast(GM_ADDR input, GM_ADDR other, GM_ADDR workspace, AssignTilingData &tiling)
{
    int block_index = GetBlockIdx();
    int block_dim = GetBlockNum();
    constexpr int MTE_BLOCK_SIZE = 512 / sizeof(T);
    int broadcast_count = 0;
    for (int i = 0; i < 4; i++)
    {
        if (tiling.other_n[i] != tiling.input_n[i])
            broadcast_count++;
    }

    GlobalTensor<T> input_global_tensor, workspace_global_tensor;
    input_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(broadcast_count % 2 == 1 ? input : workspace));
    workspace_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(other));

    TPipe t_pipe;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> t_que_bind;

    constexpr int MAX_TILE_SIZE = (191 << 10) / sizeof(T);
    t_pipe.InitBuffer(t_que_bind, 1, MAX_TILE_SIZE * sizeof(T));

    int high_size = tiling.other_n[0] * tiling.other_n[1] * tiling.other_n[2] * tiling.other_n[3];
    int low_size = 1;
    for (int i = 0; i < 4; i++)
    {
        high_size /= tiling.other_n[i];
        if (tiling.other_n[i] != tiling.input_n[i])
        {
            int mid_size = tiling.input_n[i];
            int high_from, high_to, low_from, low_to;
            if (low_size > MAX_TILE_SIZE)
            {
                high_from = 0;
                high_to = high_size;
                int low_blocks = ceil_div(low_size, MTE_BLOCK_SIZE);
                low_from = low_blocks * block_index / block_dim * MTE_BLOCK_SIZE;
                low_to = min(low_blocks * (block_index + 1) / block_dim * MTE_BLOCK_SIZE, low_size);
            }
            else
            {
                high_from = high_size * block_index / block_dim;
                high_to = high_size * (block_index + 1) / block_dim;
                low_from = 0;
                low_to = low_size;
            }
            for (int j = high_from; j < high_to; j++)
            {
                for (int k = low_from; k < low_to; k += MAX_TILE_SIZE)
                {
                    int _ = min(low_to - k, MAX_TILE_SIZE);
                    {
                        LocalTensor<T> workspace = t_que_bind.AllocTensor<T>();
                        DataCopyExtParams data_copy_ext_params;
                        data_copy_ext_params.blockLen = _ * sizeof(T);
                        DataCopyPad(workspace, workspace_global_tensor[low_size * j + k], data_copy_ext_params, {});
                        t_que_bind.EnQue(workspace);
                    }
                    {
                        LocalTensor<T> input = t_que_bind.DeQue<T>();
                        DataCopyExtParams data_copy_ext_params;
                        data_copy_ext_params.blockLen = _ * sizeof(T);
                        for (int l = 0; l < mid_size; l++)
                            DataCopyPad(input_global_tensor[(j * mid_size + l) * low_size + k], input, data_copy_ext_params);
                        t_que_bind.FreeTensor(input);
                    }
                }
            }
            if (--broadcast_count % 2 == 0)
            {
                input_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace));
                workspace_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
            }
            else
            {
                input_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
                workspace_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace));
            }
            CrossCoreSetFlag<0, PIPE_MTE3>(0);
            CrossCoreWaitFlag(0);
        }
        low_size *= tiling.input_n[i];
    }
}

extern "C" __global__ __aicore__ void assign(GM_ADDR input, GM_ADDR other, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(0))
        assign<DTYPE_INPUT>(input, other, tiling_data);
    else if (TILING_KEY_IS(1))
    {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIV_1_0);
        assign_broadcast<DTYPE_INPUT>(input, other, workspace, tiling_data);
    }
}
