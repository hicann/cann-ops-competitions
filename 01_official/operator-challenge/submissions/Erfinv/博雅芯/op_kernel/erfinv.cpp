#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"

using namespace AscendC;

#define QUEUE_SIZE 2

#define C0 (float)2.6614387
#define C1 (float)-0.176356971
#define C2 (float)-0.00505544385
#define C3 (float)-0.000245757634
#define C4 (float)-8.90277624e-06
#define C5 (float)5.18547211e-07
#define C6 (float)1.60569613e-07
#define C7 (float)1.87987332e-08
#define C8 (float)1.94082128e-09
#define C9 (float)2.2907938e-10
#define C10 (float)3.42406794e-13
#define C11 (float)-4.20382305e-12
#define C12 (float)-2.99225949e-13
#define C13 (float)1.61046425e-14
#define C14 (float)1.61937781e-15

template<typename T>
class KernelErfinv {
public:
    __aicore__ inline KernelErfinv() {}
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
        pipe->InitBuffer(queue_y, QUEUE_SIZE, buffer * sizeof(T));
        pipe->InitBuffer(buf_tmp, buffer * (sizeof(float) * 2));
        // 全局tensor绑定
        global_x.SetGlobalBuffer((__gm__ T *)gm_x);
        global_y.SetGlobalBuffer((__gm__ T *)gm_y);
    }
    __aicore__ inline void Process() {
        for(int32_t i = start; i < end; i += buffer) {
            if(i + buffer > end) {
                CopyInTail(i, end - i);
                Compute(end - i);
                CopyOutTail(i, end - i);
                break;
            }
            CopyIn(i);
            Compute(buffer);
            CopyOut(i);
        }
    }
    __aicore__ inline void Compute(int32_t len) {
        LocalTensor<T> x = queue_x.DeQue<T>();
        LocalTensor<T> y = queue_y.AllocTensor<T>();
        LocalTensor<float> p = buf_tmp.Get<float>();
        LocalTensor<float> w = buf_tmp.GetWithOffset<float>(buffer, buffer * sizeof(float));
        Duplicate(w, (float)1, len); // w = 1
        Cast(p, x, RoundMode::CAST_NONE, len); // p = x
        Mul(p, p, p, len); // p = x*x
        Sub(p, w, p, len); // p = w - p
        Ln(p, p, len); // p = ln(p)
        Adds(w, p, (float)8, len); // w = p + 8
        Muls(p, w, C14, len);
        Adds(p, p, C13, len);
        Mul(p, p, w, len);
        Adds(p, p, C12, len);
        Mul(p, p, w, len);
        Adds(p, p, C11, len);
        Mul(p, p, w, len);
        Adds(p, p, C10, len);
        Mul(p, p, w, len);
        Adds(p, p, C9, len);
        Mul(p, p, w, len);
        Adds(p, p, C8, len);
        Mul(p, p, w, len);
        Adds(p, p, C7, len);
        Mul(p, p, w, len);
        Adds(p, p, C6, len);
        Mul(p, p, w, len);
        Adds(p, p, C5, len);
        Mul(p, p, w, len);
        Adds(p, p, C4, len);
        Mul(p, p, w, len);
        Adds(p, p, C3, len);
        Mul(p, p, w, len);
        Adds(p, p, C2, len);
        Mul(p, p, w, len);
        Adds(p, p, C1, len);
        Mul(p, p, w, len);
        Adds(p, p, C0, len);
        Cast(w, x, RoundMode::CAST_NONE, len); // w = x
        Mul(p, p, w, len); // p = p * x
        Cast(y, p, RoundMode::CAST_RINT, len);
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
        LocalTensor<T> y = queue_y.DeQue<T>();
        DataCopy(global_y[start], y, buffer);
        queue_y.FreeTensor(y);
    }
    __aicore__ inline void CopyOutTail(int32_t start, int32_t len) {
        LocalTensor<T> y = queue_y.DeQue<T>();
        DataCopyPad(global_y[start], y, DataCopyExtParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0});
        queue_y.FreeTensor(y);
    }
private:
    TQue<QuePosition::VECIN, QUEUE_SIZE> queue_x;
    TQue<QuePosition::VECOUT, QUEUE_SIZE> queue_y;
    TBuf<QuePosition::VECCALC> buf_tmp;
    GlobalTensor<T> global_x;
    GlobalTensor<T> global_y;
    int32_t start, end, buffer;
};

template<>
class KernelErfinv<half> {
    using T = half;
public:
    __aicore__ inline KernelErfinv() {}
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
        pipe->InitBuffer(queue_y, QUEUE_SIZE, buffer * sizeof(T));
        pipe->InitBuffer(buf_tmp, buffer * (sizeof(float) * 2));
        // 全局tensor绑定
        global_x.SetGlobalBuffer((__gm__ T *)gm_x);
        global_y.SetGlobalBuffer((__gm__ T *)gm_y);
    }
    __aicore__ inline void Process() {
        for(int32_t i = start; i < end; i += buffer) {
            if(i + buffer > end) {
                CopyInTail(i, end - i);
                Compute(end - i);
                CopyOutTail(i, end - i);
                break;
            }
            CopyIn(i);
            Compute(buffer);
            CopyOut(i);
        }
    }
    __aicore__ inline void Compute(int32_t len) {
        LocalTensor<T> x = queue_x.DeQue<T>();
        LocalTensor<T> y = queue_y.AllocTensor<T>();
        LocalTensor<float> p = buf_tmp.Get<float>();
        LocalTensor<float> w = buf_tmp.GetWithOffset<float>(buffer, buffer * sizeof(float));
        Duplicate(w, (float)1, len); // w = 1
        Cast(p, x, RoundMode::CAST_NONE, len); // p = x
        Mul(p, p, p, len); // p = x*x
        Sub(p, w, p, len); // p = w - p
        Ln(p, p, len); // p = ln(p)
        Adds(w, p, (float)8, len); // w = p + 8
        Muls(p, w, C14, len);
        Adds(p, p, C13, len);
        Mul(p, p, w, len);
        Adds(p, p, C12, len);
        Mul(p, p, w, len);
        Adds(p, p, C11, len);
        Mul(p, p, w, len);
        Adds(p, p, C10, len);
        Mul(p, p, w, len);
        Adds(p, p, C9, len);
        Mul(p, p, w, len);
        Adds(p, p, C8, len);
        Mul(p, p, w, len);
        Adds(p, p, C7, len);
        Mul(p, p, w, len);
        Adds(p, p, C6, len);
        Mul(p, p, w, len);
        Adds(p, p, C5, len);
        Mul(p, p, w, len);
        Adds(p, p, C4, len);
        Mul(p, p, w, len);
        Adds(p, p, C3, len);
        Mul(p, p, w, len);
        Adds(p, p, C2, len);
        Mul(p, p, w, len);
        Adds(p, p, C1, len);
        Mul(p, p, w, len);
        Adds(p, p, C0, len);
        Cast(y, p, RoundMode::CAST_RINT, len);
        Mul(y, y, x, len);
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
        LocalTensor<T> y = queue_y.DeQue<T>();
        DataCopy(global_y[start], y, buffer);
        queue_y.FreeTensor(y);
    }
    __aicore__ inline void CopyOutTail(int32_t start, int32_t len) {
        LocalTensor<T> y = queue_y.DeQue<T>();
        DataCopyPad(global_y[start], y, DataCopyExtParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0});
        queue_y.FreeTensor(y);
    }
private:
    TQue<QuePosition::VECIN, QUEUE_SIZE> queue_x;
    TQue<QuePosition::VECOUT, QUEUE_SIZE> queue_y;
    TBuf<QuePosition::VECCALC> buf_tmp;
    GlobalTensor<T> global_x;
    GlobalTensor<T> global_y;
    int32_t start, end, buffer;
};

template<>
class KernelErfinv<float> {
    using T = float;
public:
    __aicore__ inline KernelErfinv() {}
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
        pipe->InitBuffer(queue_y, QUEUE_SIZE, buffer * sizeof(T));
        pipe->InitBuffer(buf_tmp, buffer * sizeof(T) * 2);
        // 全局tensor绑定
        global_x.SetGlobalBuffer((__gm__ T *)gm_x);
        global_y.SetGlobalBuffer((__gm__ T *)gm_y);
    }
    __aicore__ inline void Process() {
        for(int32_t i = start; i < end; i += buffer) {
            if(i + buffer > end) {
                CopyInTail(i, end - i);
                Compute(end - i);
                CopyOutTail(i, end - i);
                break;
            }
            CopyIn(i);
            Compute(buffer);
            CopyOut(i);
        }
    }
    __aicore__ inline void Compute(int32_t len) {
        LocalTensor<T> x = queue_x.DeQue<T>();
        LocalTensor<T> y = queue_y.AllocTensor<T>();
        LocalTensor<float> p = buf_tmp.Get<float>();
        LocalTensor<float> w = buf_tmp.GetWithOffset<float>(buffer, buffer * sizeof(float));
        Duplicate(w, (float)1, len); // w = 1
        Mul(p, x, x, len); // p = x*x
        Sub(p, w, p, len); // p = w - p
        Ln(p, p, len); // p = ln(p)
        Adds(w, p, (float)8, len); // w = p + 8        
        Muls(p, w, C14, len);
        Adds(p, p, C13, len);
        Mul(p, p, w, len);
        Adds(p, p, C12, len);
        Mul(p, p, w, len);
        Adds(p, p, C11, len);
        Mul(p, p, w, len);
        Adds(p, p, C10, len);
        Mul(p, p, w, len);
        Adds(p, p, C9, len);
        Mul(p, p, w, len);
        Adds(p, p, C8, len);
        Mul(p, p, w, len);
        Adds(p, p, C7, len);
        Mul(p, p, w, len);
        Adds(p, p, C6, len);
        Mul(p, p, w, len);
        Adds(p, p, C5, len);
        Mul(p, p, w, len);
        Adds(p, p, C4, len);
        Mul(p, p, w, len);
        Adds(p, p, C3, len);
        Mul(p, p, w, len);
        Adds(p, p, C2, len);
        Mul(p, p, w, len);
        Adds(p, p, C1, len);
        Mul(p, p, w, len);
        Adds(p, p, C0, len);
        Mul(y, p, x, len);
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
        LocalTensor<T> y = queue_y.DeQue<T>();
        DataCopy(global_y[start], y, buffer);
        queue_y.FreeTensor(y);
    }
    __aicore__ inline void CopyOutTail(int32_t start, int32_t len) {
        LocalTensor<T> y = queue_y.DeQue<T>();
        DataCopyPad(global_y[start], y, DataCopyExtParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0});
        queue_y.FreeTensor(y);
    }
private:
    TQue<QuePosition::VECIN, QUEUE_SIZE> queue_x;
    TQue<QuePosition::VECOUT, QUEUE_SIZE> queue_y;
    TBuf<QuePosition::VECCALC> buf_tmp;
    GlobalTensor<T> global_x;
    GlobalTensor<T> global_y;
    int32_t start, end, buffer;
};

extern "C" __global__ __aicore__ void erfinv(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    KernelErfinv<DTYPE_X> op;
    op.Init(x, y, &pipe, tiling_data.total, tiling_data.each, tiling_data.buffer);
    op.Process();
}