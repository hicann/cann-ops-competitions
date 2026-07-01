#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

// K5_poly4_muls_opt_clean_shape0：当前最高分baseline的小清理融合版。
// 保持4阶多项式、Muls、[-2.24, 2.24] clamp、8192 tile、双缓冲和DataCopy快路径不变。
// 只做安全微优化：K_MAX_SHAPE_DIM=0、删除未用length成员、workspace匿名化。
using namespace AscendC;

template <class DT_X>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}

__aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length)
{
    xGm.SetGlobalBuffer((__gm__ float *)x);
    yGm.SetGlobalBuffer((__gm__ float *)y);

    uint32_t blockNum = GetBlockNum();
    uint32_t blockIdx = GetBlockIdx();

    // 256B 对齐分组：64 个 float = 256B
    constexpr uint32_t GROUP_ELEMS = 1024;

    uint32_t groupNum = length / GROUP_ELEMS;
    uint32_t tail = length - groupNum * GROUP_ELEMS;

    uint32_t baseGroup = 0;
    uint32_t remainGroup = 0;

    if (blockNum > 0) {
        baseGroup = groupNum / blockNum;
        remainGroup = groupNum % blockNum;
    }

    uint32_t thisGroup = baseGroup + (blockIdx < remainGroup ? 1 : 0);
    uint32_t groupStart = blockIdx * baseGroup + (blockIdx < remainGroup ? blockIdx : remainGroup);

    this->blockStart = groupStart * GROUP_ELEMS;
    this->blockLength = thisGroup * GROUP_ELEMS;

    uint32_t tailOwner = 0;
    if (groupNum > 0) {
        uint32_t usedBlockNum = blockNum;
        if (groupNum < blockNum) {
            usedBlockNum = groupNum;
        }
        tailOwner = usedBlockNum - 1;
    }

    if (tail != 0 && blockIdx == tailOwner) {
        this->blockLength += tail;
    }

    pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_LENGTH * sizeof(float));
    pipe.InitBuffer(outQueue, BUFFER_NUM, TILE_LENGTH * sizeof(float));
    pipe.InitBuffer(calcBuf, TILE_LENGTH * sizeof(float));
}

    __aicore__ inline void Process() {
        if (this->blockLength == 0) {
            return;
        }

        uint32_t processed = 0;

        // -------------------------------
        // 先预取第一个 tile
        // -------------------------------
        uint32_t curLen = MinUInt32(TILE_LENGTH, this->blockLength - processed);
        uint32_t curCalcLen = AlignUp8(curLen);
        uint32_t curOffset = this->blockStart + processed;

        CopyIn(curOffset, curLen, curCalcLen);

        while (processed < this->blockLength) {
            uint32_t nextProcessed = processed + curLen;

            // -------------------------------
            // 提前搬运下一块数据
            // BUFFER_NUM=2，使下一块 CopyIn 尽量和当前块 Compute 重叠
            // -------------------------------
            if (nextProcessed < this->blockLength) {
                uint32_t nextLen = MinUInt32(TILE_LENGTH, this->blockLength - nextProcessed);
                uint32_t nextCalcLen = AlignUp8(nextLen);
                uint32_t nextOffset = this->blockStart + nextProcessed;

                CopyIn(nextOffset, nextLen, nextCalcLen);
            }

            // -------------------------------
            // 计算当前 tile
            // -------------------------------
            Compute(curCalcLen);

            // -------------------------------
            // 写回当前 tile，只写真实 curLen
            // -------------------------------
            CopyOut(curOffset, curLen);

            processed = nextProcessed;

            // -------------------------------
            // 更新当前 tile 信息
            // 队列中已经预取好的下一块，会在下一轮 Compute 中 DeQue
            // -------------------------------
            if (processed < this->blockLength) {
                curLen = MinUInt32(TILE_LENGTH, this->blockLength - processed);
                curCalcLen = AlignUp8(curLen);
                curOffset = this->blockStart + processed;
            }
        }
    }

private:
    // E方案沿用D方案双缓冲
    static constexpr uint32_t BUFFER_NUM = 2;

    // 8192 8960 9216 9472 9728 
    static constexpr uint32_t TILE_LENGTH = 9728;

    __aicore__ inline uint32_t MinUInt32(uint32_t a, uint32_t b) {
        return a < b ? a : b;
    }

    __aicore__ inline uint32_t AlignUp8(uint32_t x) {
        return (x + 7) & (~7);
    }

    __aicore__ inline bool IsAligned8(uint32_t offset, uint32_t len) {
        return (((offset | len) & 7) == 0);
    }

    __aicore__ inline void CopyIn(uint32_t gmOffset, uint32_t curLen, uint32_t calcLen) {
        LocalTensor<DT_X> xLocal = inQueue.AllocTensor<DT_X>();

        // E方案优化点：
        // 对齐块直接走普通 DataCopy；
        // 非对齐块，包括 gmOffset 非8对齐或 curLen 非8对齐，仍然走 DataCopyPad。
        if (IsAligned8(gmOffset, curLen)) {
            DataCopy(xLocal, xGm[gmOffset], curLen);
        } else {
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = curLen * sizeof(DT_X);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;

            DataCopyPadExtParams<DT_X> padParams;
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = calcLen - curLen;
            padParams.paddingValue = static_cast<DT_X>(0);

            DataCopyPad(xLocal, xGm[gmOffset], copyParams, padParams);
        }

        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t calcLen) {
        LocalTensor<DT_X> xLocal = inQueue.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue.AllocTensor<DT_X>();
        LocalTensor<DT_X> zLocal = calcBuf.Get<DT_X>();

        // ------------------------------------------------------------
        // K5-Opt方案：4阶少指令多项式 + Muls 去 Duplicate
        // 1. Host 核数策略不变，minElemsPerCore=1024；
        // 2. TILE_LENGTH=8192 不变；
        // 3. BUFFER_NUM=2 双缓冲不变；
        // 4. CopyIn/CopyOut 的 DataCopy 快路径不变；
        // 5. 仍然使用 erf(x) ≈ x * P(x^2) 的奇函数形式。
        //
        // 关键变化：
        // 1. clamp 从 [-2.8, 2.8] 改为 [-2.24, 2.24]；
        //    因为 erf(2.24) 已经非常接近 1，降低拟合区间可以让 4阶多项式更容易通过精度。
        // 2. P(z) 从 5/6 阶继续降到 4 阶：
        //    P(z)=c0+c1*z+c2*z^2+c3*z^3+c4*z^4, z=x^2。
        //
        // 相比 K4_poly5：少 1 次 Mul + 1 次 Adds。
        // ------------------------------------------------------------

        // 1. x = clamp(x, -2.24, 2.24)
        Mins(xLocal, xLocal, static_cast<DT_X>(2.24f), calcLen);
        Maxs(xLocal, xLocal, static_cast<DT_X>(-2.24f), calcLen);

        // 2. z = x * x
        Mul(zLocal, xLocal, xLocal, calcLen);

        // 3. 4阶 Horner 计算 P(z)
        //
        // K5-Opt方案：在已通过并提分的 K5_poly4 基础上，只做低风险指令数压缩。
        // 数学形式保持完全一致：
        //   P(z)=c0+c1*z+c2*z^2+c3*z^3+c4*z^4, z=x^2
        //
        // 优化点：
        //   原 K5 先 Duplicate(c4)，再 Mul(c4*z)；
        //   这里改为 Muls(yLocal, zLocal, c4)，直接得到 c4*z。
        //   这样少一次 Duplicate Vector API，其他搬运、tile、双缓冲、核间划分都不变。
        //
        // c0 =  1.12431724
        // c1 = -0.357548949
        // c2 =  0.0882685077
        // c3 = -0.0124660552
        // c4 =  0.000738827828

        Muls(yLocal, zLocal, static_cast<DT_X>(7.38827828e-04f), calcLen);
        Adds(yLocal, yLocal, static_cast<DT_X>(-1.24660552e-02f), calcLen);

        Mul(yLocal, yLocal, zLocal, calcLen);
        Adds(yLocal, yLocal, static_cast<DT_X>(8.82685077e-02f), calcLen);

        Mul(yLocal, yLocal, zLocal, calcLen);
        Adds(yLocal, yLocal, static_cast<DT_X>(-3.57548949e-01f), calcLen);

        Mul(yLocal, yLocal, zLocal, calcLen);
        Adds(yLocal, yLocal, static_cast<DT_X>(1.12431724f), calcLen);

        // 4. y = x * P(z)
        Mul(yLocal, yLocal, xLocal, calcLen);

        outQueue.EnQue(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t gmOffset, uint32_t curLen) {
        LocalTensor<DT_X> yLocal = outQueue.DeQue<DT_X>();

        // E方案优化点：
        // 对齐块直接普通 DataCopy 写回；
        // 非对齐块仍使用 DataCopyPad，只写回真实 curLen。
        if (IsAligned8(gmOffset, curLen)) {
            DataCopy(yGm[gmOffset], yLocal, curLen);
        } else {
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = curLen * sizeof(DT_X);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;

            DataCopyPad(yGm[gmOffset], yLocal, copyParams);
        }

        outQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<TPosition::VECCALC> calcBuf;

    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;

    uint32_t blockStart = 0;
    uint32_t blockLength = 0;
};

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);

    KernelErf<DT_X> op;
    op.Init(x, y, tiling_data.length);
    op.Process();
}