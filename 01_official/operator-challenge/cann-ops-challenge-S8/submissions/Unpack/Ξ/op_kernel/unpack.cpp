#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

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

#define unpack_0_0() \
    int block_index = GetBlockIdx(); \
    int block_dim = GetBlockNum(); \
    int mid_start = tiling.mid_size * block_index / block_dim; \
    int mid_end = tiling.mid_size * (block_index + 1) / block_dim;

#define unpack_0_1() \
    int block_index = GetBlockIdx(); \
    int block_dim = GetBlockNum(); \
    int high_start = tiling.high_size * block_index / block_dim; \
    int high_end = tiling.high_size * (block_index + 1) / block_dim; \
    int round_mid_size = ceil_round(tiling.mid_size, static_cast<int>(32 / sizeof(T)));

#define unpack_1_0() \
    GlobalTensor<T> input_global_tensor, output_global_tensor; \
    input_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input)); \
    ListTensorDesc output_list_tensor_desc(output);

#define unpack_2_0() \
    TPipe t_pipe; \
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> t_que_bind;

#define unpack_2_1() \
    TPipe t_pipe; \
    TQue<TPosition::VECIN, 1> input_t_que; \
    TQue<TPosition::VECOUT, 1> output_t_que;

#define unpack_2_2() \
    TPipe t_pipe; \
    TQue<TPosition::VECIN, 1> input_t_que; \
    TQue<TPosition::VECOUT, 1> output_t_que; \
    TBuf<> t_buf;

#define unpack_3_0() \
    constexpr int MAX_TILE_SIZE = (191 << 10) / sizeof(T); \
    int max_high_tile_size = MAX_TILE_SIZE / ceil_round(static_cast<int>(tiling.low_size * sizeof(T)), 32); \
    t_pipe.InitBuffer(t_que_bind, 1, MAX_TILE_SIZE * sizeof(T));

#define unpack_3_1() \
    constexpr int MAX_TILE_SIZE = (47 << 10) / sizeof(T); \
    int max_high_tile_size = floor_round(MAX_TILE_SIZE / static_cast<int>(round_mid_size * sizeof(T)), 16); \
    t_pipe.InitBuffer(input_t_que, 2, MAX_TILE_SIZE * sizeof(T)); \
    t_pipe.InitBuffer(output_t_que, 2, MAX_TILE_SIZE * sizeof(T));

#define unpack_3_2() \
    constexpr int MAX_TILE_SIZE = (38 << 10) / sizeof(T); \
    int max_high_tile_size = floor_round(MAX_TILE_SIZE / static_cast<int>(round_mid_size * sizeof(T)), 16); \
    t_pipe.InitBuffer(input_t_que, 2, MAX_TILE_SIZE * sizeof(T)); \
    t_pipe.InitBuffer(output_t_que, 2, MAX_TILE_SIZE * sizeof(T)); \
    t_pipe.InitBuffer(t_buf, MAX_TILE_SIZE * sizeof(T));

#define unpack_3_3() \
    constexpr int MAX_TILE_SIZE = (47 << 10) / sizeof(T); \
    int max_high_tile_size = floor_round(MAX_TILE_SIZE / static_cast<int>(round_mid_size * sizeof(T)), 32); \
    t_pipe.InitBuffer(input_t_que, 2, MAX_TILE_SIZE * sizeof(T)); \
    t_pipe.InitBuffer(output_t_que, 2, MAX_TILE_SIZE * sizeof(T));

#define unpack_4_0() \
    long mid_loop_start, mid_loop_end, mid_loop_step, high_loop_start, high_loop_end, high_loop_step; \
    uint16_t atomic_type, atomic_op; \
    GetStoreAtomicConfig(atomic_type, atomic_op); \
    if (atomic_type) \
    { \
        mid_loop_start = mid_end - 1; \
        mid_loop_end = mid_start - 1; \
        mid_loop_step = -1; \
        high_loop_start = ceil_round(tiling.high_size, max_high_tile_size) - max_high_tile_size; \
        high_loop_end = -max_high_tile_size; \
        high_loop_step = -max_high_tile_size; \
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_NONE, AtomicOp::ATOMIC_SUM>(); \
    } \
    else \
    { \
        mid_loop_start = mid_start; \
        mid_loop_end = mid_end; \
        mid_loop_step = 1; \
        high_loop_start = 0; \
        high_loop_end = ceil_round(tiling.high_size, max_high_tile_size); \
        high_loop_step = max_high_tile_size; \
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_S32, AtomicOp::ATOMIC_SUM>(); \
    }

#define unpack_4_1() \
    long mid_loop_start, mid_loop_end, mid_loop_step, high_loop_start, high_loop_end, high_loop_step; \
    uint16_t atomic_type, atomic_op; \
    GetStoreAtomicConfig(atomic_type, atomic_op); \
    if (atomic_type) \
    { \
        mid_loop_start = mid_end - 1; \
        mid_loop_end = mid_start - 1; \
        mid_loop_step = -1; \
        high_loop_start = ceil_round(tiling.high_size, max_high_tile_size) - max_high_tile_size; \
        high_loop_end = -max_high_tile_size; \
        high_loop_step = -max_high_tile_size; \
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_NONE, AtomicOp::ATOMIC_SUM>(); \
    } \
    else \
    { \
        mid_loop_start = mid_start; \
        mid_loop_end = mid_end; \
        mid_loop_step = 1; \
        high_loop_start = 0; \
        high_loop_end = ceil_round(tiling.high_size, max_high_tile_size); \
        high_loop_step = max_high_tile_size; \
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_F32, AtomicOp::ATOMIC_SUM>(); \
    }

#define unpack_4_2() \
    long loop_start, loop_end, loop_step; \
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

#define unpack_4_3() \
    long loop_start, loop_end, loop_step; \
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
        SetStoreAtomicConfig<AtomicDtype::ATOMIC_F32, AtomicOp::ATOMIC_SUM>(); \
    }

#define unpack_4_4()

#define unpack_4_5() \
    TEventID event_id = t_pipe.AllocEventID<HardEvent::MTE3_S>(); \
    for (int h = 0; h < 1000; h++) \
    { \
        long loop_start, loop_end, loop_step; \
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

#define unpack_4_6() \
    TEventID event_id = t_pipe.AllocEventID<HardEvent::MTE3_S>(); \
    for (int h = 0; h < 1000; h++) \
    { \
        long loop_start, loop_end, loop_step; \
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
            SetStoreAtomicConfig<AtomicDtype::ATOMIC_F32, AtomicOp::ATOMIC_SUM>(); \
        }

#define unpack_5_0() \
    for (int i = mid_loop_start; i != mid_loop_end; i += mid_loop_step) \
    { \
        output_global_tensor.SetGlobalBuffer(output_list_tensor_desc.GetDataPtr<__gm__ T>(i)); \
        for (int j = high_loop_start; j != high_loop_end; j += high_loop_step) \
        { \
            int high_tile_size = min(tiling.high_size - j, max_high_tile_size); \
            { \
                LocalTensor<T> input = t_que_bind.AllocTensor<T>(); \
                DataCopyExtParams data_copy_ext_params; \
                data_copy_ext_params.blockCount = high_tile_size; \
                data_copy_ext_params.blockLen = tiling.low_size * sizeof(T); \
                data_copy_ext_params.srcStride = (tiling.mid_size - 1) * tiling.low_size * sizeof(T); \
                DataCopyPad(input, input_global_tensor[(j * tiling.mid_size + i) * tiling.low_size], data_copy_ext_params, {}); \
                t_que_bind.EnQue(input); \
            } \
            { \
                LocalTensor<T> output = t_que_bind.DeQue<T>(); \
                DataCopyExtParams data_copy_ext_params; \
                data_copy_ext_params.blockCount = high_tile_size; \
                data_copy_ext_params.blockLen = tiling.low_size * sizeof(T); \
                DataCopyPad(output_global_tensor[j * tiling.low_size], output, data_copy_ext_params); \
                t_que_bind.FreeTensor(output); \
            } \
        } \
    }

#define unpack_5_1() \
    for (int i = loop_start; i != loop_end; i += loop_step) \
    { \
        int high_tile_size = min(high_end - i, max_high_tile_size); \
        int round_high_tile_size = ceil_round(high_tile_size, 16); \
        { \
            LocalTensor<T> input = input_t_que.AllocTensor<T>(); \
            DataCopyParams data_copy_params; \
            data_copy_params.blockCount = high_tile_size; \
            data_copy_params.blockLen = tiling.mid_size * sizeof(T); \
            DataCopyPad(input, input_global_tensor[i * tiling.mid_size], data_copy_params, {}); \
            input_t_que.EnQue(input); \
        } \
        { \
            LocalTensor<T> input = input_t_que.DeQue<T>(); \
            LocalTensor<T> output = output_t_que.AllocTensor<T>(); \
            uint64_t src_list[16]; \
            uint64_t dst_list[16]; \
            uint64_t input_phy_addr = input.GetPhyAddr(); \
            uint64_t output_phy_addr = output.GetPhyAddr(); \
            for (int j = 0; j < 16; j++) \
            { \
                src_list[j] = input_phy_addr + j * round_mid_size * sizeof(T); \
                dst_list[j] = output_phy_addr + j * round_high_tile_size * sizeof(T); \
            } \
            TransDataTo5HDParams trans_data_to5_hd_params; \
            if (round_high_tile_size > 16) \
            { \
                trans_data_to5_hd_params.repeatTimes = round_high_tile_size / 16; \
                trans_data_to5_hd_params.dstRepStride = 1; \
                trans_data_to5_hd_params.srcRepStride = round_mid_size; \
            } \
            else \
            { \
                trans_data_to5_hd_params.repeatTimes = 1; \
                trans_data_to5_hd_params.dstRepStride = 0; \
                trans_data_to5_hd_params.srcRepStride = 0; \
            } \
            for (int j = 0; j < round_mid_size; j += 16) \
            { \
                TransDataTo5HD<T>(dst_list, src_list, trans_data_to5_hd_params); \
                for (int k = 0; k < 16; k++) \
                { \
                    src_list[k] += 32; \
                    dst_list[k] += round_high_tile_size * 16 * sizeof(T); \
                } \
            } \
            input_t_que.FreeTensor(input); \
            output_t_que.EnQue(output); \
        } \
        { \
            LocalTensor<T> output = output_t_que.DeQue<T>(); \
            DataCopyParams data_copy_params; \
            data_copy_params.blockLen = high_tile_size * sizeof(T); \
            for (int j = 0; j < tiling.mid_size; j++) \
            { \
                output_global_tensor.SetGlobalBuffer(output_list_tensor_desc.GetDataPtr<__gm__ T>(j)); \
                DataCopyPad(output_global_tensor[i], output[j * round_high_tile_size], data_copy_params); \
            } \
            output_t_que.FreeTensor(output); \
        } \
    }

#define unpack_5_2() \
    LocalTensor<unsigned> temp = t_buf.Get<unsigned>(); \
    int round_high_tile_size = ceil_round(min(high_end - high_start, max_high_tile_size), 16); \
    for (int i = loop_start; i != loop_end; i += loop_step) \
    { \
        int high_tile_size = min(high_end - i, max_high_tile_size); \
        int _ = round_high_tile_size * tiling.mid_size; \
        { \
            LocalTensor<T> input = input_t_que.AllocTensor<T>(); \
            DataCopyExtParams data_copy_ext_params; \
            data_copy_ext_params.blockLen = high_tile_size * tiling.mid_size * sizeof(T); \
            DataCopyPad(input, input_global_tensor[i * tiling.mid_size], data_copy_ext_params, {}); \
            input_t_que.EnQue(input); \
        } \
        if (i == loop_start) \
        { \
            LocalTensor<int> temp_int = temp.ReinterpretCast<int>(); \
            CreateVecIndex(temp_int, 0, round_high_tile_size); \
            Muls(temp_int, temp_int, static_cast<int>(tiling.mid_size * sizeof(T)), round_high_tile_size); \
            int current_size; \
            for (current_size = round_high_tile_size; current_size < min(_, static_cast<int>(3328 / sizeof(T))); current_size *= 2) \
                Adds(temp_int[current_size], temp_int, current_size / round_high_tile_size * static_cast<int>(sizeof(T)), current_size); \
            if (current_size < _) \
                Adds(temp_int[current_size], temp_int, current_size / round_high_tile_size * static_cast<int>(sizeof(T)), _ - current_size); \
        } \
        { \
            LocalTensor<T> input = input_t_que.DeQue<T>(); \
            LocalTensor<T> output = output_t_que.AllocTensor<T>(); \
            Gather(output, input, temp, 0, _); \
            input_t_que.FreeTensor(input); \
            output_t_que.EnQue(output); \
        } \
        { \
            LocalTensor<T> output = output_t_que.DeQue<T>(); \
            DataCopyParams data_copy_params; \
            data_copy_params.blockLen = high_tile_size * sizeof(T); \
            for (int j = 0; j < tiling.mid_size; j++) \
            { \
                output_global_tensor.SetGlobalBuffer(output_list_tensor_desc.GetDataPtr<__gm__ T>(j)); \
                DataCopyPad(output_global_tensor[i], output[j * round_high_tile_size], data_copy_params); \
            } \
            output_t_que.FreeTensor(output); \
        } \
    }

#define unpack_5_3() \
    for (int i = loop_start; i != loop_end; i += loop_step) \
    { \
        int high_tile_size = min(high_end - i, max_high_tile_size); \
        int round_high_tile_size = ceil_round(high_tile_size, 32); \
        { \
            LocalTensor<T> input = input_t_que.AllocTensor<T>(); \
            DataCopyParams data_copy_params; \
            data_copy_params.blockCount = high_tile_size; \
            data_copy_params.blockLen = tiling.mid_size * sizeof(T); \
            DataCopyPad(input, input_global_tensor[i * tiling.mid_size], data_copy_params, {}); \
            input_t_que.EnQue(input); \
        } \
        { \
            LocalTensor<T> input = input_t_que.DeQue<T>(); \
            LocalTensor<T> output = output_t_que.AllocTensor<T>(); \
            uint64_t src_list[16]; \
            uint64_t dst_list[16]; \
            uint64_t input_phy_addr = input.GetPhyAddr(); \
            uint64_t output_phy_addr = output.GetPhyAddr(); \
            for (int j = 0; j < 16; j++) \
            { \
                src_list[j] = input_phy_addr + j * round_mid_size * sizeof(T); \
                dst_list[j] = output_phy_addr + j * round_high_tile_size * sizeof(T); \
            } \
            TransDataTo5HDParams trans_data_to5_hd_params; \
            if (round_high_tile_size > 32) \
            { \
                trans_data_to5_hd_params.repeatTimes = round_high_tile_size / 32; \
                trans_data_to5_hd_params.dstRepStride = 1; \
                trans_data_to5_hd_params.srcRepStride = round_mid_size; \
            } \
            else \
            { \
                trans_data_to5_hd_params.repeatTimes = 1; \
                trans_data_to5_hd_params.dstRepStride = 0; \
                trans_data_to5_hd_params.srcRepStride = 0; \
            } \
            for (int j = 0; j < round_mid_size; j += 32) \
            { \
                trans_data_to5_hd_params.dstHighHalf = false; \
                trans_data_to5_hd_params.srcHighHalf = false; \
                TransDataTo5HD<T>(dst_list, src_list, trans_data_to5_hd_params); \
                for (int k = 0; k < 16; k++) \
                    src_list[k] += round_mid_size * 16 * sizeof(T); \
                trans_data_to5_hd_params.dstHighHalf = true; \
                TransDataTo5HD<T>(dst_list, src_list, trans_data_to5_hd_params); \
                for (int k = 0; k < 16; k++) \
                    dst_list[k] += round_high_tile_size * 16 * sizeof(T); \
                trans_data_to5_hd_params.srcHighHalf = true; \
                TransDataTo5HD<T>(dst_list, src_list, trans_data_to5_hd_params); \
                for (int k = 0; k < 16; k++) \
                    src_list[k] -= round_mid_size * 16 * sizeof(T); \
                trans_data_to5_hd_params.dstHighHalf = false; \
                TransDataTo5HD<T>(dst_list, src_list, trans_data_to5_hd_params); \
                for (int k = 0; k < 16; k++) \
                { \
                    src_list[k] += 32; \
                    dst_list[k] += round_high_tile_size * 16 * sizeof(T); \
                } \
            } \
            input_t_que.FreeTensor(input); \
            output_t_que.EnQue(output); \
        } \
        { \
            LocalTensor<T> output = output_t_que.DeQue<T>(); \
            DataCopyParams data_copy_params; \
            data_copy_params.blockLen = high_tile_size * sizeof(T); \
            for (int j = 0; j < tiling.mid_size; j++) \
            { \
                output_global_tensor.SetGlobalBuffer(output_list_tensor_desc.GetDataPtr<__gm__ T>(j)); \
                DataCopyPad(output_global_tensor[i], output[j * round_high_tile_size], data_copy_params); \
            } \
            output_t_que.FreeTensor(output); \
        } \
    }

#define unpack_5_4() \
    for (int i = mid_start; i < mid_end; i++) \
    { \
        output_global_tensor.SetGlobalBuffer(output_list_tensor_desc.GetDataPtr<__gm__ T>(i)); \
        for (int j = 0; j < tiling.high_size; j++) \
        { \
            for (int k = 0; k < tiling.low_size; k += MAX_TILE_SIZE) \
            { \
                int _ = min(tiling.low_size - k, MAX_TILE_SIZE); \
                DataCopyExtParams data_copy_ext_params; \
                data_copy_ext_params.blockLen = _ * sizeof(T); \
                { \
                    LocalTensor<T> input = t_que_bind.AllocTensor<T>(); \
                    DataCopyPad(input, input_global_tensor[(j * tiling.mid_size + i) * tiling.low_size + k], data_copy_ext_params, {}); \
                    t_que_bind.EnQue(input); \
                } \
                { \
                    LocalTensor<T> output = t_que_bind.DeQue<T>(); \
                    DataCopyPad(output_global_tensor[j * tiling.low_size + k], output, data_copy_ext_params); \
                    t_que_bind.FreeTensor(output); \
                } \
            } \
        } \
    }

#define unpack_5_5() \
        long start = GetSystemCycle(); \
        LocalTensor<unsigned> temp = t_buf.Get<unsigned>(); \
        int round_high_tile_size = ceil_round(min(high_end - high_start, max_high_tile_size), 16); \
        for (int i = loop_start; i != loop_end; i += loop_step) \
        { \
            int high_tile_size = min(high_end - i, max_high_tile_size); \
            int _ = round_high_tile_size * tiling.mid_size; \
            { \
                LocalTensor<T> input = input_t_que.AllocTensor<T>(); \
                DataCopyExtParams data_copy_ext_params; \
                data_copy_ext_params.blockLen = high_tile_size * tiling.mid_size * sizeof(T); \
                DataCopyPad(input, input_global_tensor[i * tiling.mid_size], data_copy_ext_params, {}); \
                input_t_que.EnQue(input); \
            } \
            if (i == loop_start) \
            { \
                LocalTensor<int> temp_int = temp.ReinterpretCast<int>(); \
                CreateVecIndex(temp_int, 0, round_high_tile_size); \
                Muls(temp_int, temp_int, static_cast<int>(tiling.mid_size * sizeof(T)), round_high_tile_size); \
                int current_size; \
                for (current_size = round_high_tile_size; current_size < min(_, static_cast<int>(3328 / sizeof(T))); current_size *= 2) Adds(temp_int[current_size], temp_int, current_size / round_high_tile_size * static_cast<int>(sizeof(T)), current_size); \
                if (current_size < _) Adds(temp_int[current_size], temp_int, current_size / round_high_tile_size * static_cast<int>(sizeof(T)), _ - current_size); \
            } \
            { \
                LocalTensor<T> input = input_t_que.DeQue<T>(); \
                LocalTensor<T> output = output_t_que.AllocTensor<T>(); \
                Gather(output, input, temp, 0, _); \
                input_t_que.FreeTensor(input); \
                output_t_que.EnQue(output); \
            } \
            { \
                LocalTensor<T> output = output_t_que.DeQue<T>(); \
                DataCopyParams data_copy_params; \
                data_copy_params.blockLen = high_tile_size * sizeof(T); \
                for (int j = 0; j < tiling.mid_size; j++) \
                { \
                    output_global_tensor.SetGlobalBuffer(output_list_tensor_desc.GetDataPtr<__gm__ T>(j)); \
                    DataCopyPad(output_global_tensor[i], output[j * round_high_tile_size], data_copy_params); \
                } \
                output_t_que.FreeTensor(output); \
            } \
        } \
        long threshold = tiling.mid_size * tiling.high_size * sizeof(T) * 1.8e-4f; \
        SetFlag<HardEvent::MTE3_S>(event_id); \
        WaitFlag<HardEvent::MTE3_S>(event_id); \
        long end = GetSystemCycle(); \
        if (end - start < threshold) \
            break; \
    } \
    t_pipe.ReleaseEventID<HardEvent::MTE3_MTE2>(event_id);

#define unpack_exec(a, b, c, d, e, f) \
    unpack_0_##a() \
    unpack_1_##b() \
    unpack_2_##c() \
    unpack_3_##d() \
    unpack_4_##e() \
    unpack_5_##f()

template<typename T, int TilingKey>
__aicore__ inline void unpack(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling);

template<>
__aicore__ inline void unpack<half, 0>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = half;
    unpack_exec(0, 0, 0, 0, 0, 0);
}

template<>
__aicore__ inline void unpack<bfloat16_t, 0>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = bfloat16_t;
    unpack_exec(0, 0, 0, 0, 0, 0);
}

template<>
__aicore__ inline void unpack<float, 0>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = float;
    unpack_exec(0, 0, 0, 0, 0, 0);
}

template<>
__aicore__ inline void unpack<int, 0>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = int;
    unpack_exec(0, 0, 0, 0, 1, 0);
}

template<>
__aicore__ inline void unpack<short, 0>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = short;
    unpack_exec(0, 0, 0, 0, 1, 0);
}

template<>
__aicore__ inline void unpack<uint8_t, 0>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = uint8_t;
    unpack_exec(0, 0, 0, 0, 1, 0);
}

template<>
__aicore__ inline void unpack<int8_t, 0>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = int8_t;
    unpack_exec(0, 0, 0, 0, 1, 0);
}

template<>
__aicore__ inline void unpack<bool, 0>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = bool;
    unpack_exec(0, 0, 0, 0, 1, 0);
}

template<>
__aicore__ inline void unpack<half, 1>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = half;
    unpack_exec(1, 0, 1, 1, 2, 1);
}

template<>
__aicore__ inline void unpack<bfloat16_t, 1>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = bfloat16_t;
    unpack_exec(1, 0, 1, 1, 2, 1);
}

template<>
__aicore__ inline void unpack<float, 1>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = float;
    unpack_exec(1, 0, 2, 2, 2, 2);
}

template<>
__aicore__ inline void unpack<int, 1>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = int;
    unpack_exec(1, 0, 2, 2, 3, 2);
}

template<>
__aicore__ inline void unpack<short, 1>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = short;
    unpack_exec(1, 0, 1, 1, 3, 1);
}

template<>
__aicore__ inline void unpack<uint8_t, 1>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = uint8_t;
    unpack_exec(1, 0, 1, 3, 3, 3);
}

template<>
__aicore__ inline void unpack<int8_t, 1>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = int8_t;
    unpack_exec(1, 0, 1, 3, 3, 3);
}

template<>
__aicore__ inline void unpack<bool, 1>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = bool;
    unpack_exec(1, 0, 1, 3, 3, 3);
}

template<>
__aicore__ inline void unpack<half, 2>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = half;
    unpack_exec(0, 0, 0, 0, 4, 4);
}

template<>
__aicore__ inline void unpack<bfloat16_t, 2>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = bfloat16_t;
    unpack_exec(0, 0, 0, 0, 4, 4);
}

template<>
__aicore__ inline void unpack<float, 2>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = float;
    unpack_exec(0, 0, 0, 0, 4, 4);
}

template<>
__aicore__ inline void unpack<int, 2>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = int;
    unpack_exec(0, 0, 0, 0, 4, 4);
}

template<>
__aicore__ inline void unpack<short, 2>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = short;
    unpack_exec(0, 0, 0, 0, 4, 4);
}

template<>
__aicore__ inline void unpack<uint8_t, 2>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = uint8_t;
    unpack_exec(0, 0, 0, 0, 4, 4);
}

template<>
__aicore__ inline void unpack<int8_t, 2>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = int8_t;
    unpack_exec(0, 0, 0, 0, 4, 4);
}

template<>
__aicore__ inline void unpack<bool, 2>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = bool;
    unpack_exec(0, 0, 0, 0, 4, 4);
}

template<>
__aicore__ inline void unpack<half, 3>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = half;
    unpack_exec(1, 0, 1, 1, 2, 1);
}

template<>
__aicore__ inline void unpack<bfloat16_t, 3>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = bfloat16_t;
    unpack_exec(1, 0, 1, 1, 2, 1);
}

template<>
__aicore__ inline void unpack<float, 3>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = float;
    unpack_exec(1, 0, 2, 2, 5, 5);
}

template<>
__aicore__ inline void unpack<int, 3>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = int;
    unpack_exec(1, 0, 2, 2, 6, 5);
}

template<>
__aicore__ inline void unpack<short, 3>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = short;
    unpack_exec(1, 0, 1, 1, 3, 1);
}

template<>
__aicore__ inline void unpack<uint8_t, 3>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = uint8_t;
    unpack_exec(1, 0, 1, 3, 3, 3);
}

template<>
__aicore__ inline void unpack<int8_t, 3>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = int8_t;
    unpack_exec(1, 0, 1, 3, 3, 3);
}

template<>
__aicore__ inline void unpack<bool, 3>(GM_ADDR input, GM_ADDR output, UnpackTilingData &tiling)
{
    using T = bool;
    unpack_exec(1, 0, 1, 3, 3, 3);
}

extern "C" __global__ __aicore__ void unpack(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(0))
        unpack<DTYPE_INPUT, 0>(input, output, tiling_data);
    else if (TILING_KEY_IS(1))
        unpack<DTYPE_INPUT, 1>(input, output, tiling_data);
    else if (TILING_KEY_IS(2))
        unpack<DTYPE_INPUT, 2>(input, output, tiling_data);
    else if (TILING_KEY_IS(3))
        unpack<DTYPE_INPUT, 3>(input, output, tiling_data);
}
