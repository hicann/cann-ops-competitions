#include "kernel_operator.h"
#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

static constexpr uint32_t BUFFER_NUM = 2;
static constexpr uint32_t ALIGN_ELEM = 8;

static constexpr float ERF_CLAMP_B = 2.5f;

// 7 项 Horner 多项式系数（q(z) 的 6 次）
static constexpr float ERF_Q0 =  1.1283278415444710e+00f;
static constexpr float ERF_Q1 = -3.7521613750121097e-01f;
static constexpr float ERF_Q2 =  1.1017227860008497e-01f;
static constexpr float ERF_Q3 = -2.3862963685707575e-02f;
static constexpr float ERF_Q4 =  3.5287012564340996e-03f;
static constexpr float ERF_Q5 = -3.1035556189440617e-04f;
static constexpr float ERF_Q6 =  1.1987494547599188e-05f;

__aicore__ inline uint32_t AlignUp8(uint32_t n) {
    return (n + ALIGN_ELEM - 1) / ALIGN_ELEM * ALIGN_ELEM;
}

template <class DT_X>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ErfTilingData &tiling) {
        uint32_t blockIdx = GetBlockIdx();
        tileLength_       = tiling.tileLength;

        if (blockIdx < tiling.coreNum - 1) {
            coreOffset_ = blockIdx * tiling.lengthPerCore;
            coreLength_ = tiling.lengthPerCore;
        } else {
            coreOffset_ = blockIdx * tiling.lengthPerCore;
            coreLength_ = tiling.tailLength;
        }

        tileNum_  =(coreLength_>0)?(coreLength_ / tileLength_):0;
        lastTileLength_ =(coreLength_>0)?(coreLength_ % tileLength_):0;
       

        // GM buffer：AlignUp8 保证 DataCopy 对齐访问安全
        uint32_t gmLen = AlignUp8(coreLength_);
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x + coreOffset_, gmLen);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y + coreOffset_, gmLen);

        uint32_t slotBytes = AlignUp8(tileLength_) * sizeof(DT_X);
        pipe_.InitBuffer(inQueueX_,  BUFFER_NUM, slotBytes);
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, slotBytes);
        pipe_.InitBuffer(bufZ_,slotBytes);
    }

    __aicore__ inline void Process() {
        // host 侧保证 tileLength_ = AlignDown8(...)，已是 8 的倍数
        // 无需在此调用 AlignUp8(tileLength_)
        uint32_t offset = 0;
        for (uint32_t i = 0; i < tileNum_; ++i) {
            CopyIn(offset, tileLength_);
            Compute(tileLength_);
            CopyOut(offset, tileLength_);
            offset += tileLength_;
        }
        if (lastTileLength_ > 0) {
            // 尾块长度可能不是 8 的整数倍（非对齐场景）
            // AlignUp8 保证 DataCopy 安全；Compute 内部也用 AlignUp8(len)
            uint32_t tailLen = AlignUp8(lastTileLength_);
            CopyIn(offset, tailLen);
            Compute(tailLen);
            CopyOut(offset, tailLen);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t gmOffset, uint32_t len) {
        LocalTensor<DT_X> xLocal = inQueueX_.template AllocTensor<DT_X>();
        DataCopy(xLocal, xGm_[gmOffset], len);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t n) {
        LocalTensor<DT_X> xLocal = inQueueX_.template DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY_.template AllocTensor<DT_X>();
        LocalTensor<DT_X> zBuf   = bufZ_.template Get<DT_X>();

        // Step1: xClamp = clamp(x, -2.5, 2.5)
        // 覆盖写回 xLocal（DeQue 后 xLocal 可任意修改，省去 bufPoly1_）
        Mins(xLocal, xLocal, ERF_CLAMP_B, n);
        Maxs(xLocal, xLocal, -ERF_CLAMP_B, n);

        // Step2: z = xClamp^2
        Mul(zBuf, xLocal, xLocal, n);

        // Step3: Horner 展开 q(z)，从最高阶 Q6 开始
        Muls(yLocal, zBuf, ERF_Q6, n);
        Adds(yLocal, yLocal, ERF_Q5, n);
        Mul(yLocal, yLocal, zBuf, n);
        Adds(yLocal, yLocal, ERF_Q4, n);
        Mul(yLocal, yLocal, zBuf, n);
        Adds(yLocal, yLocal, ERF_Q3, n);
        Mul(yLocal, yLocal, zBuf, n);
        Adds(yLocal, yLocal, ERF_Q2, n);
        Mul(yLocal, yLocal, zBuf, n);
        Adds(yLocal, yLocal, ERF_Q1, n);
        Mul(yLocal, yLocal, zBuf, n);
        Adds(yLocal, yLocal, ERF_Q0, n);

        // Step4: y = xClamp * q(z)
        Mul(yLocal, yLocal, xLocal, n);

        outQueueY_.EnQue(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t gmOffset, uint32_t len) {
        LocalTensor<DT_X> yLocal = outQueueY_.template DeQue<DT_X>();
        DataCopy(yGm_[gmOffset], yLocal, len);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;

    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;

    // 只需 1 个 TBuf（原来 2 个），节省 1 个 slot
    TBuf<QuePosition::VECCALC> bufZ_;

    GlobalTensor<DT_X> xGm_, yGm_;

    uint32_t coreOffset_    = 0;
    uint32_t coreLength_    = 0;
    uint32_t tileLength_    = 0;
    uint32_t tileNum_       = 0;
    uint32_t lastTileLength_= 0;
};

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    KernelErf<DT_X> op;
    op.Init(x, y, tiling_data);
    op.Process();
}