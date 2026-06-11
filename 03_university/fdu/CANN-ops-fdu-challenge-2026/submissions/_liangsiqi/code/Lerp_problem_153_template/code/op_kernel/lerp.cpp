// Kernel侧核函数实现
#include "kernel_operator.h"

#include "lerp_tiling.h"
#include "tiling_key_lerp.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // 双缓冲, 搬运与计算流水并行

template <class T>
class KernelLerp {
public:
    __aicore__ inline KernelLerp() {}
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR y, const LerpTilingData &t) {
        tile = t;
        uint32_t blockIdx = GetBlockIdx();
        // 512 元素块均分 + 余数分摊到前 remainder 个核(负载均衡): 每核起始 512 元素对齐, 末核裁到边界
        constexpr uint32_t BLK = 512u;
        uint32_t myBlocks, blockStart;
        if (blockIdx < t.remainder) { myBlocks = t.blockEle + 1; blockStart = blockIdx * (t.blockEle + 1); }
        else { myBlocks = t.blockEle; blockStart = t.remainder * (t.blockEle + 1) + (blockIdx - t.remainder) * t.blockEle; }
        eleStart = blockStart * BLK;
        if (eleStart >= t.length) { myEle = 0; return; }
        myEle = t.length - eleStart;
        if (myEle > myBlocks * BLK) myEle = myBlocks * BLK;

        sGm.SetGlobalBuffer((__gm__ T *)start + eleStart);
        eGm.SetGlobalBuffer((__gm__ T *)end + eleStart);
        yGm.SetGlobalBuffer((__gm__ T *)y + eleStart);
        // 原生 T 精度计算, 对齐参考实现 y = start + weight*(end-start) 的逐步舍入
        w = (T)tile.weight;

        pipe.InitBuffer(inQueueS, BUFFER_NUM, tile.tileLength * sizeof(T));
        pipe.InitBuffer(inQueueE, BUFFER_NUM, tile.tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tile.tileLength * sizeof(T));
    }

    __aicore__ inline void Process() {
        if (myEle == 0) return;
        for (uint32_t off = 0; off < myEle; off += tile.tileLength) {
            uint32_t len = (myEle - off < tile.tileLength) ? (myEle - off) : tile.tileLength;
            CopyIn(off, len);
            Compute(len);
            CopyOut(off, len);
        }
    }

private:
    __aicore__ inline void Load(LocalTensor<T> dst, GlobalTensor<T> gm, uint32_t off, uint32_t len) {
        DataCopyExtParams p; p.blockCount = 1; p.blockLen = len * sizeof(T); p.srcStride = 0; p.dstStride = 0;
        DataCopyPadExtParams<T> pad; pad.isPad = false; pad.leftPadding = 0; pad.rightPadding = 0; pad.paddingValue = 0;
        DataCopyPad(dst, gm[off], p, pad);
    }
    __aicore__ inline void CopyIn(uint32_t off, uint32_t len) {
        LocalTensor<T> s = inQueueS.AllocTensor<T>(); LocalTensor<T> e = inQueueE.AllocTensor<T>();
        Load(s, sGm, off, len); Load(e, eGm, off, len);
        inQueueS.EnQue(s); inQueueE.EnQue(e);
    }
    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<T> s = inQueueS.DeQue<T>(); LocalTensor<T> e = inQueueE.DeQue<T>();
        LocalTensor<T> yL = outQueueY.AllocTensor<T>();
        // y = s + w*(e - s), 原生 T 精度逐步计算, 对齐参考实现
        Sub(yL, e, s, len);   // d = e - s
        Muls(yL, yL, w, len); // w*d
        Add(yL, yL, s, len);  // s + w*d
        outQueueY.EnQue<T>(yL);
        inQueueS.FreeTensor(s); inQueueE.FreeTensor(e);
    }
    __aicore__ inline void CopyOut(uint32_t off, uint32_t len) {
        LocalTensor<T> yL = outQueueY.DeQue<T>();
        DataCopyExtParams p; p.blockCount = 1; p.blockLen = len * sizeof(T); p.srcStride = 0; p.dstStride = 0;
        DataCopyPad(yGm[off], yL, p);
        outQueueY.FreeTensor(yL);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueS, inQueueE;  // depth=1: 编译器特殊优化; 双缓冲由 InitBuffer num=2 开启
    TQue<QuePosition::VECOUT, 1> outQueueY;
    GlobalTensor<T> sGm, eGm, yGm;
    LerpTilingData tile;
    uint32_t myEle = 0, eleStart = 0;
    T w;
};

template <typename DT_START>
 __global__ __aicore__ void lerp(GM_ADDR start, GM_ADDR end, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 纯 Vector 算子: 声明 AIV-only, 避免混合启动空转 Cube 核的头开销
    REGISTER_TILING_DEFAULT(LerpTilingData);
    GET_TILING_DATA_WITH_STRUCT(LerpTilingData, tiling_data, tiling);
    KernelLerp<DT_START> op;
    op.Init(start, end, y, tiling_data);
    op.Process();
}
