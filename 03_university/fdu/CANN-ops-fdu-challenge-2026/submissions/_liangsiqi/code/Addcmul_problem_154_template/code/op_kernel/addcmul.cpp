// Kernel侧核函数实现
// 本算子不使用 ShapeInfo(SetShapeInfo/GetShapeInfo), 关闭其内嵌数组以缩栈空间/减 scalar 指令与 cache miss
// 注意: 必须在包含 Ascend C 头文件之前定义
#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"

#include "addcmul_tiling.h"

#include "tiling_key_addcmul.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // 双缓冲, 搬运与计算流水并行

// 计算工作类型: fp32/int32 原生; fp16 升 float(评测参考=PyTorch 升 float opmath, 实测原生会变差); int8 经 half 中转到 float
template <typename T>
struct WorkType { using type = T; };
template <>
struct WorkType<half> { using type = float; };
template <>
struct WorkType<int8_t> { using type = float; }; // int8 经 half 中转到 float(int8 无法直转 int32/float)

template <class T>
class KernelAddcmul {
    using W = typename WorkType<T>::type;
public:
    __aicore__ inline KernelAddcmul() {}
    __aicore__ inline void Init(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
                                const AddcmulTilingData &t) {
        tile = t;
        uint32_t blockIdx = GetBlockIdx();
        inputGm.SetGlobalBuffer((__gm__ T *)input_data);
        x1Gm.SetGlobalBuffer((__gm__ T *)x1);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2);
        yGm.SetGlobalBuffer((__gm__ T *)y);
        valueGm.SetGlobalBuffer((__gm__ T *)value, 1);

        if (t.bcast == 0) {
            // 扁平 elementwise: 512 块余数均摊(blockRows=每核基础块数, 前 remainder 核多 1 块)
            constexpr uint32_t BLK = 512u;
            uint32_t myBlocks, blockStart;
            if (blockIdx < t.remainder) { myBlocks = t.blockRows + 1; blockStart = blockIdx * (t.blockRows + 1); }
            else { myBlocks = t.blockRows; blockStart = t.remainder * (t.blockRows + 1) + (blockIdx - t.remainder) * t.blockRows; }
            flatStart = blockStart * BLK;
            myFlat = (flatStart >= t.totalLength) ? 0u : (t.totalLength - flatStart);
            if (myFlat > myBlocks * BLK) myFlat = myBlocks * BLK;
            if (myFlat == 0) { return; }
        } else {
            // 按行多核切分: 前 remainder 个核各多 1 行
            if (blockIdx < t.remainder) { myRows = t.blockRows + 1; rowStart = blockIdx * (t.blockRows + 1); }
            else { myRows = t.blockRows; rowStart = t.remainder * (t.blockRows + 1) + (blockIdx - t.remainder) * t.blockRows; }
            if (myRows == 0) { return; }
        }

        pipe.InitBuffer(inQueueInput, BUFFER_NUM, tile.tileLength * sizeof(T));
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, tile.tileLength * sizeof(T));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, tile.tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tile.tileLength * sizeof(T));
        pipe.InitBuffer(valueBuf, 32);
        if constexpr (IsSameType<T, W>::value) {
            pipe.InitBuffer(tmpBuf, tile.tileLength * sizeof(W));
        } else {
            pipe.InitBuffer(wBuf0, tile.tileLength * sizeof(W));
            pipe.InitBuffer(wBuf1, tile.tileLength * sizeof(W));
            // halfBuf 仅 int8 的 int8->half->float 中转需要; fp16 是 half->float 直转, 不分配以腾 UB 用更大 tile
            if constexpr (IsSameType<T, int8_t>::value) pipe.InitBuffer(halfBuf, tile.tileLength * sizeof(half));
        }
    }

    __aicore__ inline void Process() {
        if (tile.bcast == 0) { if (myFlat == 0) return; }
        else { if (myRows == 0) return; }
        LocalTensor<T> vLocal = valueBuf.Get<T>();
        DataCopyExtParams vp; vp.blockCount = 1; vp.blockLen = sizeof(T); vp.srcStride = 0; vp.dstStride = 0;
        DataCopyPadExtParams<T> vpad; vpad.isPad = false; vpad.leftPadding = 0; vpad.rightPadding = 0; vpad.paddingValue = 0;
        DataCopyPad(vLocal, valueGm, vp, vpad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0); WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        valueScalar = static_cast<W>(vLocal.GetValue(0));

        uint32_t cap = tile.tileLength;
        if (tile.bcast == 0) {
            // 扁平快路径: 三输入同一连续偏移, 大 tile 连续搬运(last0/1/2=2 -> CopyIn 读满 len, 不 Duplicate)
            for (uint32_t off = 0; off < myFlat; off += cap) {
                uint32_t len = (myFlat - off < cap) ? (myFlat - off) : cap;
                uint32_t base = flatStart + off;
                CopyIn(base, base, base, len);
                Compute(len);
                CopyOut(base, len);
            }
            return;
        }
        // 广播路径: 按行处理, 末维分块
        uint32_t D = tile.lastDim;
        for (uint32_t r = 0; r < myRows; r++) {
            uint32_t row = rowStart + r;
            // 计算各输入本行起始偏移(外层维广播步长已 0)
            uint32_t off0 = 0, off1 = 0, off2 = 0, rem = row;
            for (int i = ADDCMUL_MAX_DIM - 2; i >= 0; i--) {
                uint32_t dim = tile.yDim[i] == 0 ? 1 : tile.yDim[i];
                uint32_t idx = rem % dim; rem /= dim;
                off0 += idx * tile.s0[i]; off1 += idx * tile.s1[i]; off2 += idx * tile.s2[i];
            }
            for (uint32_t c = 0; c < D; c += cap) {
                uint32_t len = (D - c < cap) ? (D - c) : cap;
                uint32_t c0 = (tile.last0 == 1) ? 0 : c, c1 = (tile.last1 == 1) ? 0 : c, c2 = (tile.last2 == 1) ? 0 : c;
                CopyIn(off0 + c0, off1 + c1, off2 + c2, len);
                Compute(len);
                CopyOut(row * D + c, len);
            }
        }
    }

private:
    __aicore__ inline void LoadRow(LocalTensor<T> dst, GlobalTensor<T> gm, uint32_t off, uint32_t lastK, uint32_t D) {
        DataCopyExtParams p; p.blockCount = 1; p.srcStride = 0; p.dstStride = 0;
        DataCopyPadExtParams<T> pad; pad.isPad = false; pad.leftPadding = 0; pad.rightPadding = 0; pad.paddingValue = 0;
        if (lastK == 1) { p.blockLen = sizeof(T); DataCopyPad(dst, gm[off], p, pad); } // 行内广播: 取1元素
        // 起始偏移 off 与长度 D 均 32B 对齐才可用轻量 DataCopy(其对 GM 起址有对齐要求); 否则 DataCopyPad
        else if ((off * sizeof(T)) % 32u == 0u && (D * sizeof(T)) % 32u == 0u) { DataCopy(dst, gm[off], D); }
        else { p.blockLen = D * sizeof(T); DataCopyPad(dst, gm[off], p, pad); }
    }
    // 把 T(fp16/int8) 整行转到 float 缓冲; lastK==1 时按行广播 Duplicate; int8 经 half 中转
    __aicore__ inline void ToFloat(LocalTensor<W> dst, LocalTensor<T> src, LocalTensor<half> h, uint32_t lastK, uint32_t D) {
        if (lastK == 1) { Duplicate(dst, static_cast<W>(src.GetValue(0)), D); return; }
        if constexpr (IsSameType<T, int8_t>::value) { Cast(h, src, RoundMode::CAST_NONE, D); Cast(dst, h, RoundMode::CAST_NONE, D); }
        else { Cast(dst, src, RoundMode::CAST_NONE, D); }
    }
    __aicore__ inline void CopyIn(uint32_t off0, uint32_t off1, uint32_t off2, uint32_t D) {
        LocalTensor<T> inLocal = inQueueInput.AllocTensor<T>();
        LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
        LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();
        LoadRow(inLocal, inputGm, off0, tile.last0, D);
        LoadRow(x1Local, x1Gm, off1, tile.last1, D);
        LoadRow(x2Local, x2Gm, off2, tile.last2, D);
        inQueueInput.EnQue(inLocal); inQueueX1.EnQue(x1Local); inQueueX2.EnQue(x2Local);
    }
    __aicore__ inline void Compute(uint32_t D) {
        LocalTensor<T> inLocal = inQueueInput.DeQue<T>();
        LocalTensor<T> x1Local = inQueueX1.DeQue<T>();
        LocalTensor<T> x2Local = inQueueX2.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        if constexpr (IsSameType<T, W>::value) {
            // 行内广播: 末维=1 时首元素复制满整行(half/float/int32 支持 Duplicate)
            if (tile.last0 == 1) Duplicate(inLocal, inLocal.GetValue(0), D);
            if (tile.last1 == 1) Duplicate(x1Local, x1Local.GetValue(0), D);
            if (tile.last2 == 1) Duplicate(x2Local, x2Local.GetValue(0), D);
            LocalTensor<W> tmp = tmpBuf.Get<W>();
            // 对齐 torch: input + (value*x1)*x2, 乘法配对顺序与 torch 一致以减小舍入差
            Muls(tmp, x1Local, valueScalar, D); Mul(tmp, tmp, x2Local, D); Add(yLocal, inLocal, tmp, D);
        } else {
            // fp16/int8 升 float 计算; int8 经 half 中转(int8 无法直转 float/int32)
            LocalTensor<W> w0 = wBuf0.Get<W>(); LocalTensor<W> w1 = wBuf1.Get<W>();
            LocalTensor<half> h; // fp16 不用(half->float 直转); 仅 int8 中转需要, 故仅 int8 取已分配的 halfBuf
            if constexpr (IsSameType<T, int8_t>::value) h = halfBuf.Get<half>();
            ToFloat(w0, x1Local, h, tile.last1, D);
            ToFloat(w1, x2Local, h, tile.last2, D);
            Mul(w1, w0, w1, D);                  // w1 = x1*x2
            ToFloat(w0, inLocal, h, tile.last0, D); // w0 = input
            // Axpy 融合: w0 = input + value*(x1*x2), 单次舍入(2 op vs 3, 更快且对齐 torch FMA 舍入)
            Axpy(w0, w1, valueScalar, D);
            if constexpr (IsSameType<T, int8_t>::value) {
                // int8 溢出按 torch 回绕: w0 -= 256*floor((w0+128)/256), 落回[-128,127]
                Adds(w1, w0, 128.0f, D); Muls(w1, w1, 1.0f / 256.0f, D); Floor(w1, w1, D); Muls(w1, w1, 256.0f, D); Sub(w0, w0, w1, D);
                Cast(h, w0, RoundMode::CAST_RINT, D); Cast(yLocal, h, RoundMode::CAST_RINT, D);
            }
            else { Cast(yLocal, w0, RoundMode::CAST_RINT, D); } // fp16: float->half 就近舍入对齐 torch
        }
        outQueueY.EnQue<T>(yLocal);
        inQueueInput.FreeTensor(inLocal); inQueueX1.FreeTensor(x1Local); inQueueX2.FreeTensor(x2Local);
    }
    __aicore__ inline void CopyOut(uint32_t off, uint32_t D) {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        // off 与 D 均 32B 对齐才用轻量 DataCopy(GM 起址需对齐); 否则 DataCopyPad
        if ((off * sizeof(T)) % 32u == 0u && (D * sizeof(T)) % 32u == 0u) { DataCopy(yGm[off], yLocal, D); }
        else { DataCopyExtParams p; p.blockCount = 1; p.blockLen = D * sizeof(T); p.srcStride = 0; p.dstStride = 0; DataCopyPad(yGm[off], yLocal, p); }
        outQueueY.FreeTensor(yLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueInput, inQueueX1, inQueueX2;  // depth=1: 编译器特殊优化; 双缓冲由 InitBuffer num=2 开启
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<QuePosition::VECCALC> tmpBuf, wBuf0, wBuf1, halfBuf, valueBuf;
    GlobalTensor<T> inputGm, x1Gm, x2Gm, yGm, valueGm;
    AddcmulTilingData tile;
    uint32_t myRows = 0, rowStart = 0;   // 广播路径用
    uint32_t myFlat = 0, flatStart = 0;  // 扁平路径用(每核元素数/起始)
    W valueScalar;
};

template <typename DT_INPUT_DATA>
 __global__ __aicore__ void addcmul(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 纯 Vector 算子: 声明 AIV-only, 避免混合启动空转 Cube 核的头开销
    REGISTER_TILING_DEFAULT(AddcmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddcmulTilingData, tiling_data, tiling);
    KernelAddcmul<DT_INPUT_DATA> op;
    op.Init(input_data, x1, x2, value, y, tiling_data);
    op.Process();
}
