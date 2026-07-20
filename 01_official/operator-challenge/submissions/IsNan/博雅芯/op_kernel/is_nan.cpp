#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"

using namespace AscendC;

#define QUEUE_SIZE 2

class KernelIsNanScalar {
    using T = DTYPE_X;
public:
    __aicore__ inline KernelIsNanScalar() {}
    __aicore__ inline void Init(
        GM_ADDR gm_x, GM_ADDR gm_y,
        int size
    ) {
        global_x.SetGlobalBuffer((__gm__ T *)gm_x);
        global_y.SetGlobalBuffer((__gm__ bool *)gm_y);
        this->size = size;
    }
    __aicore__ inline void Process() {
        for(int i = 0; i < size; ++i) {
            T v = global_x.GetValue(i);
            bool res;
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                float f = ToFloat(v);
                res = f != f;
            } else {
                float f = (float)v;
                res = f != f;
            }
            global_y.SetValue(i, res);
        }
    }
private:
    GlobalTensor<T> global_x;
    GlobalTensor<bool> global_y;
    int size;
};

template<typename T>
class KernelIsNanVector {
public:
    __aicore__ inline KernelIsNanVector() {}
    __aicore__ inline void Init(
        GM_ADDR gm_x, GM_ADDR gm_y,
        TPipe *pipe,
        int total, int each, int buffer
    ) {
        // 当前核负责区间[start, end)的子任务
        int32_t core_id = GetBlockIdx();
        start = each * core_id;
        end = start + each;
        if(end > total) end = total;
        this->buffer = buffer;
        // 分配空间
        pipe->InitBuffer(queue_x, QUEUE_SIZE, buffer * sizeof(T));
        pipe->InitBuffer(queue_y, QUEUE_SIZE, buffer);
        pipe->InitBuffer(buf_tmp, buffer * 3); // 合并所有的TBuf
        // 全局tensor绑定
        global_x.SetGlobalBuffer((__gm__ T *)gm_x);
        global_y.SetGlobalBuffer((__gm__ int8_t *)gm_y);
    }
    __aicore__ inline void Process() {
        for(int32_t i = start; i < end; i += buffer) {
            if(i + buffer > end) {
                CopyInTail(i, end - i);
                Compute();
                CopyOutTail(i, end - i);
                break;
            }
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }
    __aicore__ inline void Compute() {
        LocalTensor<T> x = queue_x.DeQue<T>();
        LocalTensor<int8_t> y = queue_y.AllocTensor<int8_t>();
        LocalTensor<uint8_t> mask = buf_tmp.Get<uint8_t>();
        LocalTensor<half> res = buf_tmp.GetWithOffset<half>(buffer, buffer);
        // 计算
        // for(int i = 0; i < 32; ++i) printf("%f ", x.GetValue(i));
        // printf("\n");
        Compare(mask, x, x, CMPMODE::EQ, buffer);
        // CompareScalar(mask, x, (float)0, CMPMODE::GT, buffer);
        // for(int i = 0; i < 32; ++i) printf("%d ", mask.GetValue(i / 8) >> (i % 8) & 1);
        // printf("\n");
        Duplicate(res, (half)0, buffer);
        Select(res, mask, res, (half)1, SELMODE::VSEL_TENSOR_SCALAR_MODE, buffer);
        // for(int i = 0; i < 32; ++i) printf("%f ", res.GetValue(i));
        // printf("\n");
        Cast(y, res, RoundMode::CAST_NONE, buffer);
        // for(int i = 0; i < 32; ++i) printf("%d ", y.GetValue(i));
        // printf("\n");
        
        queue_x.FreeTensor(x);
        queue_y.EnQue(y);
    }
    __aicore__ inline void CopyIn(int32_t start) {
        LocalTensor<T> x = queue_x.AllocTensor<T>();
        DataCopy(x, global_x[start], buffer);
        queue_x.EnQue(x);
    }
    __aicore__ inline void CopyInTail(int32_t start, int32_t len) {
        LocalTensor<T> x = queue_x.AllocTensor<T>();
        DataCopyPad(x, global_x[start], DataCopyExtParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0}, DataCopyPadExtParams{false, 0, 0, (T)0});
        queue_x.EnQue(x);
    }
    __aicore__ inline void CopyOut(int32_t start) {
        LocalTensor<int8_t> y = queue_y.DeQue<int8_t>();
        DataCopy(global_y[start], y, buffer);
        queue_y.FreeTensor(y);
    }
    __aicore__ inline void CopyOutTail(int32_t start, int32_t len) {
        LocalTensor<int8_t> y = queue_y.DeQue<int8_t>();
        DataCopyPad(global_y[start], y, DataCopyExtParams{1, static_cast<uint32_t>(len), 0, 0, 0});
        queue_y.FreeTensor(y);
    }
private:
    TQue<QuePosition::VECIN, QUEUE_SIZE> queue_x;
    TQue<QuePosition::VECOUT, QUEUE_SIZE> queue_y;
    TBuf<QuePosition::VECCALC> buf_tmp;
    GlobalTensor<T> global_x;
    GlobalTensor<int8_t> global_y;
    int32_t start, end, buffer;
};

template<>
class KernelIsNanVector<bfloat16_t> {
    using T = bfloat16_t;
public:
    __aicore__ inline KernelIsNanVector() {}
    __aicore__ inline void Init(
        GM_ADDR gm_x, GM_ADDR gm_y,
        TPipe *pipe,
        int total, int each, int buffer
    ) {
        // 当前核负责区间[start, end)的子任务
        int32_t core_id = GetBlockIdx();
        start = each * core_id;
        end = start + each;
        if(end > total) end = total;
        this->buffer = buffer;
        // 分配空间
        pipe->InitBuffer(queue_x, QUEUE_SIZE, buffer * sizeof(T));
        pipe->InitBuffer(queue_y, QUEUE_SIZE, buffer);
        pipe->InitBuffer(buf_tmp, buffer * 7); // 合并所有的TBuf
        // 全局tensor绑定
        global_x.SetGlobalBuffer((__gm__ T *)gm_x);
        global_y.SetGlobalBuffer((__gm__ int8_t *)gm_y);
    }
    __aicore__ inline void Process() {
        for(int32_t i = start; i < end; i += buffer) {
            if(i + buffer > end) {
                CopyInTail(i, end - i);
                Compute();
                CopyOutTail(i, end - i);
                break;
            }
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }
    __aicore__ inline void Compute() {
        LocalTensor<T> x = queue_x.DeQue<T>();
        LocalTensor<int8_t> y = queue_y.AllocTensor<int8_t>();
        LocalTensor<uint8_t> mask = buf_tmp.Get<uint8_t>();
        LocalTensor<half> res = buf_tmp.GetWithOffset<half>(buffer, buffer);
        LocalTensor<float> cast = buf_tmp.GetWithOffset<float>(buffer, buffer * 3);
        // 计算
        Cast(cast, x, RoundMode::CAST_NONE, buffer);
        Compare(mask, cast, cast, CMPMODE::EQ, buffer);
        Duplicate(res, (half)0, buffer);
        Select(res, mask, res, (half)1, SELMODE::VSEL_TENSOR_SCALAR_MODE, buffer);
        Cast(y, res, RoundMode::CAST_NONE, buffer);
        
        queue_x.FreeTensor(x);
        queue_y.EnQue(y);
    }
    __aicore__ inline void CopyIn(int32_t offset) {
        LocalTensor<T> x = queue_x.AllocTensor<T>();
        DataCopy(x, global_x[offset], buffer);
        queue_x.EnQue(x);
    }
    __aicore__ inline void CopyInTail(int32_t offset, int32_t len) {
        LocalTensor<T> x = queue_x.AllocTensor<T>();
        DataCopyPad(x, global_x[offset], DataCopyExtParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0}, DataCopyPadExtParams{false, 0, 0, (T)0});
        queue_x.EnQue(x);
    }
    __aicore__ inline void CopyOut(int32_t offset) {
        LocalTensor<int8_t> y = queue_y.DeQue<int8_t>();
        DataCopy(global_y[offset], y, buffer);
        queue_y.FreeTensor(y);
    }
    __aicore__ inline void CopyOutTail(int32_t offset, int32_t len) {
        LocalTensor<int8_t> y = queue_y.DeQue<int8_t>();
        DataCopyPad(global_y[offset], y, DataCopyExtParams{1, static_cast<uint32_t>(len), 0, 0, 0});
        queue_y.FreeTensor(y);
    }
private:
    TQue<QuePosition::VECIN, QUEUE_SIZE> queue_x;
    TQue<QuePosition::VECOUT, QUEUE_SIZE> queue_y;
    TBuf<QuePosition::VECCALC> buf_tmp;
    GlobalTensor<T> global_x;
    GlobalTensor<int8_t> global_y;
    int32_t start, end, buffer;
};

extern "C" __global__ __aicore__ void is_nan(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    KernelIsNanVector<DTYPE_X> op;
    op.Init(x, y, &pipe, tiling_data.total, tiling_data.each, tiling_data.buffer);
    op.Process();
}