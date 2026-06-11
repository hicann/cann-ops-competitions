// Kernel侧核函数实现
#include "kernel_operator.h"

#include "clip_by_value_tiling.h"
#include "tiling_key_clip_by_value.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // 双缓冲, 搬运与计算流水并行

template <class T>
class KernelClipByValue {
public:
    __aicore__ inline KernelClipByValue() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ClipByValueTilingData &t) {
        tile = t;
        uint32_t blockIdx = GetBlockIdx();
        // blockFormer 连续切分: 每核 blockEle 个元素(512 元素对齐 -> 起始 256B 对齐), 末核裁到边界
        eleStart = t.blockEle * blockIdx;
        if (eleStart >= t.length) { myEle = 0; return; }
        myEle = t.length - eleStart;
        if (myEle > t.blockEle) myEle = t.blockEle;

        xGm.SetGlobalBuffer((__gm__ T *)x + eleStart);
        yGm.SetGlobalBuffer((__gm__ T *)y + eleStart);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tile.tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tile.tileLength * sizeof(T));

        // int32 用预算好的整型边界(ceil/floor), 浮点类型直接按 T 截断标量
        if constexpr (IsSameType<T, int32_t>::value) { loVal = (T)tile.minI; hiVal = (T)tile.maxI; }
        else { loVal = (T)tile.minVal; hiVal = (T)tile.maxVal; }
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
    __aicore__ inline void CopyIn(uint32_t off, uint32_t len) {
        LocalTensor<T> xL = inQueueX.AllocTensor<T>();
        DataCopyExtParams p; p.blockCount = 1; p.blockLen = len * sizeof(T); p.srcStride = 0; p.dstStride = 0;
        DataCopyPadExtParams<T> pad; pad.isPad = false; pad.leftPadding = 0; pad.rightPadding = 0; pad.paddingValue = 0;
        DataCopyPad(xL, xGm[off], p, pad);
        inQueueX.EnQue(xL);
    }
    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<T> xL = inQueueX.DeQue<T>();
        LocalTensor<T> yL = outQueueY.AllocTensor<T>();
        // y = min(max(x, lo), hi)
        Maxs(yL, xL, loVal, len);
        Mins(yL, yL, hiVal, len);
        outQueueY.EnQue<T>(yL);
        inQueueX.FreeTensor(xL);
    }
    __aicore__ inline void CopyOut(uint32_t off, uint32_t len) {
        LocalTensor<T> yL = outQueueY.DeQue<T>();
        DataCopyExtParams p; p.blockCount = 1; p.blockLen = len * sizeof(T); p.srcStride = 0; p.dstStride = 0;
        DataCopyPad(yGm[off], yL, p);
        outQueueY.FreeTensor(yL);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;  // depth=1: 编译器特殊优化; 双缓冲由 InitBuffer num=2 开启
    TQue<QuePosition::VECOUT, 1> outQueueY;
    GlobalTensor<T> xGm, yGm;
    ClipByValueTilingData tile;
    uint32_t myEle = 0, eleStart = 0;
    T loVal, hiVal;
};

template <typename DT_X>
 __global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 纯 Vector 算子: 声明 AIV-only, 避免混合启动空转 Cube 核的头开销
    REGISTER_TILING_DEFAULT(ClipByValueTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tiling_data, tiling);
    KernelClipByValue<DT_X> op;
    op.Init(x, y, tiling_data);
    op.Process();
}
