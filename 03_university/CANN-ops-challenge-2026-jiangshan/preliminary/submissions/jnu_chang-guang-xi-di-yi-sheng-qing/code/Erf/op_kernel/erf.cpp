#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t PIPE_LOWER    = 10000u;
constexpr uint32_t TINY_THRESHOLD = 32768u;

template <class DT_X>
class KernelErfLarge {
public:
    __aicore__ inline KernelErfLarge() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t startElem, uint32_t coreLen,
                                uint32_t tileLen, TPipe* pipeIn) {
        pipe = pipeIn;
        coreLength = coreLen;
        tileLength = tileLen;
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + startElem);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + startElem);
        xGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        yGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        pipe->InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        uint32_t loopCount  = coreLength / tileLength;
        uint32_t tailLength = coreLength - loopCount * tileLength;
        uint32_t alignedTail = tailLength > 0u
                             ? (tailLength + 7u) / 8u * 8u : 0u;

        CopyIn(0, tileLength);

        for (uint32_t i = 0; i < loopCount - 1; i++) {
            CopyIn((i + 1) * tileLength, tileLength);
            Compute(tileLength);
            CopyOut(i * tileLength, tileLength);
        }

        if (alignedTail > 0u) CopyIn(loopCount * tileLength, alignedTail);
        Compute(tileLength);
        CopyOut((loopCount - 1) * tileLength, tileLength);
        if (alignedTail > 0u) {
            Compute(alignedTail);
            CopyOut(loopCount * tileLength, alignedTail);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count) {
        LocalTensor<DT_X> xLocal = inQueueX.template AllocTensor<DT_X>();
        DataCopy(xLocal, xGm[offset], count);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(uint32_t count) {
        LocalTensor<DT_X> xLocal = inQueueX.template DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY.template AllocTensor<DT_X>();
        Erf(yLocal, xLocal, count);
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count) {
        LocalTensor<DT_X> yLocal = outQueueY.template DeQue<DT_X>();
        DataCopy(yGm[offset], yLocal, count);
        outQueueY.FreeTensor(yLocal);
    }

    TPipe* pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<DT_X> xGm, yGm;
    uint32_t coreLength = 0;
    uint32_t tileLength = 0;
};

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    uint32_t coreIdx   = GetBlockIdx();
    uint32_t startElem = coreIdx * tiling_data.perCoreLength;

    if (startElem >= tiling_data.totalLength) return;

    uint32_t remaining  = tiling_data.totalLength - startElem;
    uint32_t coreLength = remaining < tiling_data.perCoreLength
                        ? remaining : tiling_data.perCoreLength;

    // VECCALC 判定：小数据 + (核数多 或 极小微张量) → 串行 Padé
    // 核数少且数据量适中 → 跳过 VECCALC，走 pipeline/medium 尝试重叠
    if (coreLength <= PIPE_LOWER && (GetBlockNum() > 8u || coreLength <= 288u)) {
        // 小数据路径：无 TPipe、无 LMA，使用固定 VECCALC 地址 + 多项式 erf。
        // 硬件 Erf() 是 SFU 指令，需要 TPipe；Mul/Add 是纯 ALU 指令，无需 TPipe。
        // VECCALC 布局: 4 个缓冲区 × PIPE_LOWER × 4B  (PIPE_LOWER=12000 → 192KB, 经验 VECCALC 上限)
        // 注意: 无 TPipe 模式下 VECCALC 实际可用约 192KB (非完整 256KB UB)
        InitSocState();

        constexpr uint32_t VECCALC_X = 0u;
        constexpr uint32_t VECCALC_Y = PIPE_LOWER * 4u;        // 分子 x*N(u)
        constexpr uint32_t VECCALC_T = PIPE_LOWER * 8u;        // u = x²
        constexpr uint32_t VECCALC_V = PIPE_LOWER * 12u;       // 分母 D(u)

        GlobalTensor<DT_X> xGm, yGm;
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + startElem);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + startElem);
        uint32_t calcLen = (coreLength + 7u) / 8u * 8u;

        // max_elements 用 calcLen (而非 PIPE_LOWER): 小张量用小 max, 实测能减少
        // 多项式路径每个 Mul/Add 的内部边界检查或寻址开销。字节偏移仍按 PIPE_LOWER 分区。
        LocalTensor<DT_X> xL(TPosition::VECCALC, VECCALC_X, calcLen);
        LocalTensor<DT_X> yL(TPosition::VECCALC, VECCALC_Y, calcLen);
        LocalTensor<DT_X> tL(TPosition::VECCALC, VECCALC_T, calcLen);
        LocalTensor<DT_X> vL(TPosition::VECCALC, VECCALC_V, calcLen);

        DataCopy(xL, xGm[0], calcLen);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        // 奇函数 Padé[2,2] minimax 近似 (scipy DE+local fit)：max_err = 2.5e-4 (优于 BS 5e-4)
        // erf(x) ≈ clamp(x * N(u) / D(u), -1, 1), u = x²
        //   N(u) = a0 + a1*u + a2*u²
        //   D(u) = 1 + b1*u + b2*u²
        // 奇函数 *x 自动保留符号，省去 Abs/sign 计算；13 ops，只需 1 个 Div。

        // Step 1: tL = u = x²
        Mul(tL, xL, xL, calcLen);

        // Step 2-6: yL = x * N(u), Horner: N = a0 + u*(a1 + u*a2)
        Muls(yL, tL, (DT_X)(0.0339832977f), calcLen);
        Adds(yL, yL, (DT_X)(0.5957600023f), calcLen);
        Mul (yL, yL, tL, calcLen);
        Adds(yL, yL, (DT_X)(1.1266513665f), calcLen);
        Mul (yL, yL, xL, calcLen);

        // Step 7-10: vL = D(u), Horner: D = 1 + u*(b1 + u*b2)
        Muls(vL, tL, (DT_X)(0.2338672098f), calcLen);
        Adds(vL, vL, (DT_X)(0.8505337612f), calcLen);
        Mul (vL, vL, tL, calcLen);
        Adds(vL, vL, (DT_X)(1.0f),          calcLen);

        // Step 11: yL = x*N(u) / D(u)
        Div(yL, yL, vL, calcLen);

        // Step 12-13: clamp 到 [-1, 1]
        Mins(yL, yL, (DT_X)( 1.0f), calcLen);
        Maxs(yL, yL, (DT_X)(-1.0f), calcLen);

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(yGm[0], yL, calcLen);
    } else {
        // 大/中等数据：判断是否走流水线
        bool use_pipeline = coreLength > TINY_THRESHOLD ||
                            GetBlockNum() <= 16u;
        if (use_pipeline) {
            TPipe pipe;
            KernelErfLarge<DT_X> op;
            op.Init(x, y, startElem, coreLength, tiling_data.tileLength, &pipe);
            op.Process();
        } else {
            // 中等数据但核数多：退化为小数据路径
            TPipe pipe;
            LocalMemAllocator allocator;
            GlobalTensor<DT_X> xGm, yGm;
            xGm.SetGlobalBuffer((__gm__ DT_X *)x + startElem);
            yGm.SetGlobalBuffer((__gm__ DT_X *)y + startElem);
            uint32_t calcLen = (coreLength + 7u) / 8u * 8u;
            LocalTensor<DT_X> in  = allocator.Alloc<TPosition::VECIN,  DT_X>(calcLen);
            LocalTensor<DT_X> out = allocator.Alloc<TPosition::VECOUT, DT_X>(calcLen);
            DataCopy(in, xGm[0], calcLen);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Erf(out, in, calcLen);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(yGm[0], out, calcLen);
        }
    }
}