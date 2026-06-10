#pragma GCC optimize("Ofast")
#pragma GCC optimize("fast-math")
#pragma GCC optimize("unroll-loops")

#include "kernel_operator.h"
#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

constexpr int32_t LARGE_IN_QUEUE_DEPTH = 2;
constexpr int32_t LARGE_OUT_QUEUE_DEPTH = 2;

__aicore__ inline uint32_t AlignUpLocal(uint32_t x, uint32_t align)
{
    return (x + align - 1) & ~(align - 1);
}

// 针对 float (4 bytes) 的全局硬件对齐常数
constexpr uint32_t HW_ALIGN_FLOAT = 32 / sizeof(float);   // 8
constexpr uint32_t CALC_ALIGN_FLOAT = 256 / sizeof(float); // 64
#define ERF_USE_RISKY_P4_APPROX 1

// Erf 计算引擎：默认启用风险 P4 短公式，降低一组 Mul/Add 向量指令。
// 风险点：P4 最大绝对误差离线约 8.28e-4，但 0 附近相对误差高于稳定 P5。
__attribute__((always_inline)) __aicore__ inline void ErfApproxNative(
    LocalTensor<float>& yLocal,
    LocalTensor<float>& xLocal,
    LocalTensor<float>& poly,
    uint32_t calcLength
) {
#if ERF_USE_RISKY_P4_APPROX
    Maxs(yLocal, xLocal, -2.23674319f, calcLength);
    Mins(yLocal, yLocal, 2.23674319f, calcLength);
    Mul(xLocal, yLocal, yLocal, calcLength);
    Muls(poly, xLocal, 7.44074512e-04f, calcLength);
    Adds(poly, poly, -1.25173407e-02f, calcLength);
    Mul(poly, poly, xLocal, calcLength);
    Adds(poly, poly, 8.84319239e-02f, calcLength);
    Mul(poly, poly, xLocal, calcLength);
    Adds(poly, poly, -3.57735114e-01f, calcLength);
    Mul(poly, poly, xLocal, calcLength);
    Adds(poly, poly, 1.12437125e+00f, calcLength);
    Mul(yLocal, poly, yLocal, calcLength);
#else
    Maxs(yLocal, xLocal, -2.345f, calcLength);
    Mins(yLocal, yLocal, 2.345f, calcLength);
    Mul(xLocal, yLocal, yLocal, calcLength);
    Muls(poly, xLocal, -1.05458649e-04f, calcLength);
    Adds(poly, poly, 2.22270468e-03f, calcLength);
    Mul(poly, poly, xLocal, calcLength);
    Adds(poly, poly, -2.00621334e-02f, calcLength);
    Mul(poly, poly, xLocal, calcLength);
    Adds(poly, poly, 1.05292204e-01f, calcLength);
    Mul(poly, poly, xLocal, calcLength);
    Adds(poly, poly, -3.73161159e-01f, calcLength);
    Mul(poly, poly, xLocal, calcLength);
    Adds(poly, poly, 1.12837917e+00f, calcLength);
    Mul(yLocal, poly, yLocal, calcLength);
#endif
}

// 小数据量核函数 
class KernelErfSmall {
public:
    __aicore__ inline KernelErfSmall() {}
    __attribute__((always_inline)) __aicore__ inline void Process(GM_ADDR x, GM_ADDR y, ErfTilingData *tilingData)
    {
        uint32_t realLength = tilingData->tailLength;
        uint32_t calcLength = tilingData->maxCalcLength;
				GlobalTensor<float> xGm;
        GlobalTensor<float> yGm;
        xGm.SetGlobalBuffer((__gm__ float *)x, calcLength);
        yGm.SetGlobalBuffer((__gm__ float *)y, calcLength);

        TPipe pipe;
        TBuf<QuePosition::VECCALC> myMemPool;
        pipe.InitBuffer(myMemPool, calcLength * 3 * sizeof(float));

        LocalTensor<float> rawMem = myMemPool.Get<float>();
        LocalTensor<float> xLocal = rawMem[0];
        LocalTensor<float> yLocal = rawMem[calcLength];
        LocalTensor<float> poly   = rawMem[calcLength * 2];

        uint32_t safeLength = AlignUpLocal(realLength, HW_ALIGN_FLOAT);

        DataCopy(xLocal, xGm, safeLength);
        SetFlag<HardEvent::MTE2_V>(0); 
				WaitFlag<HardEvent::MTE2_V>(0);
        ErfApproxNative(yLocal, xLocal, poly, calcLength);
        SetFlag<HardEvent::V_MTE3>(1); 
				WaitFlag<HardEvent::V_MTE3>(1);
        DataCopy(yGm, yLocal, safeLength);
    }
};

// 多核单 tile direct：每核只处理一个 block，避免无收益的 queue 双缓冲开销
class KernelErfDirectMulti {
public:
    __aicore__ inline KernelErfDirectMulti() {}
    __attribute__((always_inline)) __aicore__ inline void Process(GM_ADDR x, GM_ADDR y, ErfTilingData *tilingData)
    {
        uint32_t coreIdx = GetBlockIdx();
        uint32_t usedCoreNum = tilingData->usedCoreNum;
        if (coreIdx >= usedCoreNum) return;

        uint32_t realLength = (coreIdx == usedCoreNum - 1) ? tilingData->tailLength : tilingData->blockLength;
        if (realLength == 0) return;

        uint32_t calcLength = tilingData->maxCalcLength;
        uint32_t offset = coreIdx * tilingData->blockLength;

        GlobalTensor<float> xGm;
        GlobalTensor<float> yGm;
        xGm.SetGlobalBuffer((__gm__ float *)x + offset, calcLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + offset, calcLength);

        TPipe pipe;
        TBuf<QuePosition::VECCALC> myMemPool;
        pipe.InitBuffer(myMemPool, calcLength * 3 * sizeof(float));

        LocalTensor<float> rawMem = myMemPool.Get<float>();
        LocalTensor<float> xLocal = rawMem[0];
        LocalTensor<float> yLocal = rawMem[calcLength];
        LocalTensor<float> poly   = rawMem[calcLength * 2];

        uint32_t safeLength = AlignUpLocal(realLength, HW_ALIGN_FLOAT);
        uint32_t cLength = AlignUpLocal(realLength, CALC_ALIGN_FLOAT);

        DataCopy(xLocal, xGm, safeLength);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        ErfApproxNative(yLocal, xLocal, poly, cLength);
        SetFlag<HardEvent::V_MTE3>(1);
        WaitFlag<HardEvent::V_MTE3>(1);
        DataCopy(yGm, yLocal, safeLength);
    }
};

// 大数据核函数 v2
class KernelErfLarge {
public:
    __aicore__ inline KernelErfLarge() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, ErfTilingData *tilingData)
    {
        uint32_t coreIdx = GetBlockIdx();
        uint32_t usedCoreNum = tilingData->usedCoreNum;
        if (coreIdx >= usedCoreNum) {
            this->realBlockLength = 0;
            return;
        }
        this->realBlockLength = (coreIdx == usedCoreNum - 1) ? tilingData->tailLength : tilingData->blockLength;
        this->tileLength = tilingData->tileLength;

        uint32_t offset = coreIdx * tilingData->blockLength;
        xGm.SetGlobalBuffer((__gm__ float *)x + offset, this->realBlockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + offset, this->realBlockLength);

        pipe.InitBuffer(inQueueX, LARGE_IN_QUEUE_DEPTH, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, LARGE_OUT_QUEUE_DEPTH, this->tileLength * sizeof(float));
        pipe.InitBuffer(bufNum, this->tileLength * sizeof(float));
    }
    __attribute__((always_inline)) __aicore__ inline void Process()
    {
        uint32_t totalLen = this->realBlockLength;
        if (totalLen == 0) return;
        uint32_t tLen = this->tileLength;
        if (totalLen <= tLen) {
            uint32_t safeLen = AlignUpLocal(totalLen, HW_ALIGN_FLOAT);
            uint32_t calcLen = AlignUpLocal(totalLen, CALC_ALIGN_FLOAT);
            CopyIn(0, safeLen);
            Compute(calcLen);
            CopyOut(0, safeLen);
            return;
        }

        uint32_t totalTileNum = (totalLen + tLen - 1) / tLen;
        uint32_t mainTiles = totalTileNum - 1;               // 保证是完整块的数量
        uint32_t lastLen = totalLen - mainTiles * tLen;      // 最后一块的尾料长度

        // 预先算好最后一块的对齐长度，绝不带入循环内部
        uint32_t lastSafe = AlignUpLocal(lastLen, HW_ALIGN_FLOAT);
        uint32_t lastCalc = AlignUpLocal(lastLen, CALC_ALIGN_FLOAT);

        CopyIn(0, tLen); 
				#pragma GCC unroll 4
        for (uint32_t i = 0; i < mainTiles - 1; ++i) {
            CopyIn(i + 1, tLen);  // 后台异步：搬入下一块
            Compute(tLen);        // 前台狂暴：计算当前块
            CopyOut(i, tLen);     // 后台异步：搬出当前块
        }
				CopyIn(mainTiles, lastSafe); // 搬入最后一块尾料
        Compute(tLen);               // 计算倒数第二块 (满载)
        CopyOut(mainTiles - 1, tLen);// 搬出倒数第二块 (满载)
				Compute(lastCalc);           // 计算最后一块尾料
        CopyOut(mainTiles, lastSafe);// 搬出最后一块尾料
    }
private:
    // 函数签名改造：直接接收确定的 safeLen/calcLen，彻底干掉运行时的 AlignUpLocal
    __attribute__((always_inline)) __aicore__ inline void CopyIn(uint32_t progress, uint32_t safeLen)
    {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[progress * this->tileLength], safeLen);
        inQueueX.EnQue(xLocal);
    }

    __attribute__((always_inline)) __aicore__ inline void Compute(uint32_t calcLen)
    {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<float> poly = bufNum.Get<float>();

        ErfApproxNative(yLocal, xLocal, poly, calcLen);

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __attribute__((always_inline)) __aicore__ inline void CopyOut(uint32_t progress, uint32_t safeLen)
    {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[progress * this->tileLength], yLocal, safeLen);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, LARGE_IN_QUEUE_DEPTH> inQueueX;
    TQue<QuePosition::VECOUT, LARGE_OUT_QUEUE_DEPTH> outQueueY;
    TBuf<QuePosition::VECCALC> bufNum;

    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;

    uint32_t realBlockLength = 0;
    uint32_t tileLength = 0;
};

template <typename DT_X>
__global__ __aicore__ void erf(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling
)
{
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tilingData, tiling);

    if (tilingData.mode == ERF_MODE_SMALL) {
        KernelErfSmall op;
        op.Process(x, y, &tilingData);
    } else if (tilingData.mode == ERF_MODE_DIRECT_MULTI) {
        KernelErfDirectMulti op;
        op.Process(x, y, &tilingData);
    } else {
        KernelErfLarge op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
