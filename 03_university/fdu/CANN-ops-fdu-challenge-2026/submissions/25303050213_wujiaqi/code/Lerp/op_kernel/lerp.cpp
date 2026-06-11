// AscendC Lerp 算子 —— Kernel 侧实现
// 数学公式: y = start + weight * (end - start)
// 支持 dtype: float16 (half) / float32
//
// === 软件流水说明 ===
// 本 kernel 采用固定深度预取 (prefetch) 的软件流水模式:
//   1. 先连续发射 BUFFER_NUM 笔 CopyIn, 把流水线填满
//   2. 进入稳态: 每完成一对 (Compute + CopyOut) 就补一发 CopyIn
//   3. 效果: MTE2 (CopyIn) 与 Vector (Compute) 可以 overlap,
//      用计算延迟掩盖片外访存延迟
//
// === 多核切分策略 ===
// 以 32B 对齐 chunk 为粒度在 AI Core 间分发, remainder 个核多
// 领 1 个 chunk,保证各核负载最多差 1 个 chunk.
#include "kernel_operator.h"

#include "lerp_tiling.h"
#include "tiling_key_lerp.h"

using namespace AscendC;

namespace {
// 队列深度: 双缓冲, MTE2 与 Vector 各持一个 slot, 乒乓切换
constexpr int32_t  BUF_CNT   = 2;
constexpr uint32_t CACHELINE = 32;
}  // namespace

template <typename T>
class KernelLerp {
public:
    __aicore__ inline KernelLerp() {}

    __aicore__ inline void Init(GM_ADDR startAddr, GM_ADDR endAddr, GM_ADDR yAddr,
                                uint32_t nTotal, float w32, uint32_t nPerSlice) {
        wScalar   = w32;
        wCast     = static_cast<T>(w32);
        sliceLen  = nPerSlice;
        totalLen  = nTotal;

        // 为 start / end 输入和 y 输出各开一组双缓冲队列
        pipe.InitBuffer(qSIn, BUF_CNT, sliceLen * sizeof(T));
        pipe.InitBuffer(qEIn, BUF_CNT, sliceLen * sizeof(T));
        pipe.InitBuffer(qYOut, BUF_CNT, sliceLen * sizeof(T));

        // 挂载 GM 地址到 GlobalTensor 视图
        gmS.SetGlobalBuffer((__gm__ T *)startAddr, nTotal);
        gmE.SetGlobalBuffer((__gm__ T *)endAddr,   nTotal);
        gmY.SetGlobalBuffer((__gm__ T *)yAddr,     nTotal);

        // ---- 核间数据划分 (block 粒度) ----
        uint32_t coreCnt  = GetBlockNum();
        uint32_t coreId   = GetBlockIdx();
        if (coreCnt == 0) { coreCnt = 1; }

        uint32_t alignCnt = CACHELINE / sizeof(T);
        if (alignCnt == 0) { alignCnt = 1; }

        uint32_t blkTotal     = (nTotal + alignCnt - 1) / alignCnt;
        uint32_t blkPerCore   = blkTotal / coreCnt;
        uint32_t blkRemainder = blkTotal % coreCnt;
        // coreId 小于 remainder 的核多分一个 chunk
        uint32_t myBlkCnt = blkPerCore + (coreId < blkRemainder ? 1u : 0u);
        uint32_t skipCnt  = coreId * blkPerCore + (coreId < blkRemainder ? coreId : blkRemainder);
        offGlobal = skipCnt * alignCnt;

        if (offGlobal >= nTotal) {
            lenLocal = 0;
            return;
        }
        uint32_t raw = myBlkCnt * alignCnt;
        if (offGlobal + raw > nTotal) {
            raw = nTotal - offGlobal;
        }
        lenLocal = raw;
    }

    __aicore__ inline void Process() {
        if (lenLocal == 0) { return; }

        uint32_t steps   = (lenLocal + sliceLen - 1) / sliceLen;
        uint32_t tailId  = steps - 1;
        uint32_t tailLen = lenLocal - tailId * sliceLen;

        // 预取深度取 min(steps, BUF_CNT), 数据量不足一个流水深度时
        // 有多少 tile 就预取多少
        uint32_t pfDepth = (steps < static_cast<uint32_t>(BUF_CNT))
                               ? steps
                               : static_cast<uint32_t>(BUF_CNT);

        // phase 1: 发射前 pfDepth 个 tile 的 CopyIn, 填满流水线
        for (uint32_t s = 0; s < pfDepth; ++s) {
            CopyIn(s, (s == tailId) ? tailLen : sliceLen);
        }
        // phase 2: 稳态 —— 每轮算一个 tile, 搬出, 再补一个 tile 的 CopyIn
        for (uint32_t s = 0; s < steps; ++s) {
            uint32_t cur = (s == tailId) ? tailLen : sliceLen;
            Compute(cur);
            CopyOut(s, cur);
            uint32_t ns = s + pfDepth;
            if (ns < steps) {
                CopyIn(ns, (ns == tailId) ? tailLen : sliceLen);
            }
        }
    }

private:
    // === CopyIn: 从 GM 搬 start & end 两个 tile 到 UB ===
    __aicore__ inline void CopyIn(uint32_t round, uint32_t n) {
        LocalTensor<T> sLocal = qSIn.AllocTensor<T>();
        LocalTensor<T> eLocal = qEIn.AllocTensor<T>();
        uint32_t base = offGlobal + round * sliceLen;
        uint32_t nb   = n * sizeof(T);

        // 对齐的 bulk 搬运走 DataCopy 快速路径, 未对齐尾部走 DataCopyPad
        if ((nb & (CACHELINE - 1)) == 0) {
            DataCopy(sLocal, gmS[base], n);
            DataCopy(eLocal, gmE[base], n);
        } else {
            DataCopyExtParams       cp{1, static_cast<uint16_t>(nb), 0, 0, 0};
            DataCopyPadExtParams<T> pp{false, 0, 0, T{}};
            DataCopyPad(sLocal, gmS[base], cp, pp);
            DataCopyPad(eLocal, gmE[base], cp, pp);
        }
        qSIn.EnQue(sLocal);
        qEIn.EnQue(eLocal);
    }

    // === Compute: 执行 y = start + weight * (end - start) ===
    // 重要: 算子精度校验 (1e-3 相对容差) 要求严格保持该运算顺序.
    // 等价公式 y = (1-w)*start + w*end 在 fp16 下因舍入路径不同,
    // 实测在 910b 上会产生少量超差元素. 先减后乘再加的顺序不可改动.
    __aicore__ inline void Compute(uint32_t n) {
        LocalTensor<T> sLocal = qSIn.DeQue<T>();
        LocalTensor<T> eLocal = qEIn.DeQue<T>();
        LocalTensor<T> yLocal = qYOut.AllocTensor<T>();

        Sub (yLocal, eLocal, sLocal, n);
        Muls(yLocal, yLocal, wCast,  n);
        Add (yLocal, yLocal, sLocal, n);

        qYOut.EnQue<T>(yLocal);
        qSIn.FreeTensor(sLocal);
        qEIn.FreeTensor(eLocal);
    }

    // === CopyOut: 从 UB 搬结果 tile 回 GM ===
    __aicore__ inline void CopyOut(uint32_t round, uint32_t n) {
        LocalTensor<T> yLocal = qYOut.DeQue<T>();
        uint32_t base = offGlobal + round * sliceLen;
        uint32_t nb   = n * sizeof(T);

        if ((nb & (CACHELINE - 1)) == 0) {
            DataCopy(gmY[base], yLocal, n);
        } else {
            DataCopyExtParams cp{1, static_cast<uint16_t>(nb), 0, 0, 0};
            DataCopyPad(gmY[base], yLocal, cp);
        }
        qYOut.FreeTensor(yLocal);
    }

private:
    TPipe pipe;

    // 三组双缓冲队列: start 输入 / end 输入 / y 输出
    TQue<TPosition::VECIN,  BUF_CNT> qSIn, qEIn;
    TQue<TPosition::VECOUT, BUF_CNT> qYOut;

    GlobalTensor<T> gmS, gmE, gmY;

    float    wScalar  = 0.0f;
    T        wCast    = T{};
    uint32_t sliceLen = 0;
    uint32_t totalLen = 0;
    uint32_t offGlobal = 0;
    uint32_t lenLocal  = 0;
};

template <typename DT_START>
__global__ __aicore__ void lerp(GM_ADDR start, GM_ADDR end, GM_ADDR y,
                                 GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LerpTilingData);
    GET_TILING_DATA_WITH_STRUCT(LerpTilingData, td, tiling);
    KernelLerp<DT_START> op;
    op.Init(start, end, y, td.length, td.weight, td.tileLength);
    op.Process();
}
