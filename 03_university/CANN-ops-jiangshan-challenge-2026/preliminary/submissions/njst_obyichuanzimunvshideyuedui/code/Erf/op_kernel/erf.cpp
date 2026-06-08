// AI Core kernel implementation for y = erf(x), float32.
// Two decoupled kernels:
//   Single path: inlined at entry, LocalMemAllocator, SetFlag/WaitFlag, no TPipe/TQue
//   Multi path:  TPipe + TQue + Counter mode + conditional double-buffer
// Entry dispatches based on tilingData.isSingle.
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

// ============================================================
// KernelErfMulti — multi-tile, Counter mode, conditional double-buffer
// ============================================================
template <class DT_X>
class KernelErfMulti
{
public:
    __aicore__ inline KernelErfMulti() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ErfTilingData &tiling, TPipe *pipeIn)
    {
        tileLength = tiling.tileLength;

        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockOff = blockIdx * tiling.blockLength;
        bool isTailCore = (blockIdx + 1 == tiling.formerNum);
        blockLength_ = isTailCore ? tiling.tailLength : tiling.blockLength;
        fullTiles_ = isTailCore ? tiling.tailFullTileCnt : tiling.fullTileCount;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + blockOff, blockLength_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + blockOff, blockLength_);
        xGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        yGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

        bp_ = AscendC::BinaryRepeatParams();
        up_ = AscendC::UnaryRepeatParams();

        int inDepth = (fullTiles_ >= 10) ? 2 : 1;
        pipeIn->InitBuffer(inQ_, inDepth, tileLength * sizeof(DT_X));
        pipeIn->InitBuffer(outQ_, 1, tileLength * sizeof(DT_X));
        pipeIn->InitBuffer(tmp_, tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ == 0)
            return;

        LocalTensor<DT_X> xLocal;
        LocalTensor<DT_X> yLocal;
        uint32_t offset = 0;
        for (uint32_t i = 0; i < fullTiles_; i++)
        {
            CopyIn(offset, tileLength);
            Compute(tileLength, xLocal, yLocal);
            CopyOut(offset, tileLength);
            offset += tileLength;
        }
        if (offset < blockLength_)
        {
            uint32_t n = blockLength_ - offset;
            xLocal = inQ_.AllocTensor<DT_X>();
            DataCopyExtParams cp{1, static_cast<uint32_t>(n * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> pp{false, 0, 0, 0};
            DataCopyPad(xLocal, xGm_[offset], cp, pp);
            inQ_.EnQue(xLocal);

            Compute(n, xLocal, yLocal);

            yLocal = outQ_.DeQue<DT_X>();
            DataCopyExtParams cp2{1, static_cast<uint32_t>(n * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(yGm_[offset], yLocal, cp2);
            outQ_.FreeTensor(yLocal);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count)
    {
        LocalTensor<DT_X> xLocal = inQ_.AllocTensor<DT_X>();
        if ((count & (ALIGN_NUM - 1)) == 0)
        {
            DataCopy(xLocal, xGm_[offset], count);
        }
        else
        {
            DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> pp{false, 0, 0, 0};
            DataCopyPad(xLocal, xGm_[offset], cp, pp);
        }
        inQ_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count, LocalTensor<DT_X> &xLocal, LocalTensor<DT_X> &yLocal)
    {
        xLocal = inQ_.DeQue<DT_X>();
        yLocal = outQ_.AllocTensor<DT_X>();
        LocalTensor<float> t = tmp_.Get<float>();

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(count);

        AscendC::Mins<float, false>(xLocal, xLocal, 2.238962216f, AscendC::MASK_PLACEHOLDER, 1, up_);
        AscendC::Maxs<float, false>(xLocal, xLocal, -2.238962216f, AscendC::MASK_PLACEHOLDER, 1, up_);

        AscendC::Mul<float, false>(t, xLocal, xLocal, AscendC::MASK_PLACEHOLDER, 1, bp_);

        AscendC::Muls<float, false>(yLocal, t, 7.400368329e-04f, AscendC::MASK_PLACEHOLDER, 1, up_);
        AscendC::Adds<float, false>(yLocal, yLocal, -1.247796912e-02f, AscendC::MASK_PLACEHOLDER, 1, up_);
        AscendC::Mul<float, false>(yLocal, yLocal, t, AscendC::MASK_PLACEHOLDER, 1, bp_);
        AscendC::Adds<float, false>(yLocal, yLocal, 8.830686063e-02f, AscendC::MASK_PLACEHOLDER, 1, up_);
        AscendC::Mul<float, false>(yLocal, yLocal, t, AscendC::MASK_PLACEHOLDER, 1, bp_);
        AscendC::Adds<float, false>(yLocal, yLocal, -3.575934732e-01f, AscendC::MASK_PLACEHOLDER, 1, up_);
        AscendC::Mul<float, false>(yLocal, yLocal, t, AscendC::MASK_PLACEHOLDER, 1, bp_);
        AscendC::Adds<float, false>(yLocal, yLocal, 1.124330570e+00f, AscendC::MASK_PLACEHOLDER, 1, up_);

        AscendC::Mul<float, false>(yLocal, yLocal, xLocal, AscendC::MASK_PLACEHOLDER, 1, bp_);

        AscendC::ResetMask();

        outQ_.EnQue(yLocal);
        inQ_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        LocalTensor<DT_X> yLocal = outQ_.DeQue<DT_X>();
        if ((count & (ALIGN_NUM - 1)) == 0)
        {
            DataCopy(yGm_[offset], yLocal, count);
        }
        else
        {
            DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(yGm_[offset], yLocal, cp);
        }
        outQ_.FreeTensor(yLocal);
    }

private:
    TQue<QuePosition::VECIN, 2> inQ_;
    TQue<QuePosition::VECOUT, 1> outQ_;
    TBuf<QuePosition::VECCALC> tmp_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    AscendC::BinaryRepeatParams bp_;
    AscendC::UnaryRepeatParams up_;
    uint32_t tileLength = 0;
    uint32_t blockLength_ = 0;
    uint32_t fullTiles_ = 0;
    static constexpr uint32_t ALIGN_NUM = 32 / sizeof(DT_X);
};

// ============================================================
// Entry — dispatch: inlined Single (no class overhead) / Multi (class)
// ============================================================
template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::ICachePreLoad(6);
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tilingData.isSingle)
    {
        InitSocState();

        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockOff = blockIdx * tilingData.blockLength;
        uint32_t n = (blockIdx + 1 == tilingData.formerNum) ? tilingData.tailLength : tilingData.blockLength;

        LocalMemAllocator<Hardware::UB> alloc;
        uint32_t allocN = (n + 7) >> 3 << 3;
        LocalTensor<float> xLocal = alloc.Alloc<TPosition::VECIN,   float>(allocN);
        LocalTensor<float> yLocal = alloc.Alloc<TPosition::VECOUT,  float>(allocN);
        LocalTensor<float> tmp    = alloc.Alloc<TPosition::VECCALC, float>(allocN);

        GlobalTensor<float> xGm;
        xGm.SetGlobalBuffer((__gm__ float *)x + blockOff, n);
        DataCopy(xLocal, xGm, n);

        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        AscendC::Mins(xLocal, xLocal, 2.238962216f, n);
        AscendC::Maxs(xLocal, xLocal, -2.238962216f, n);
        AscendC::Mul(tmp, xLocal, xLocal, n);
        AscendC::Muls(yLocal, tmp, 7.400368329e-04f, n);
        AscendC::Adds(yLocal, yLocal, -1.247796912e-02f, n);
        AscendC::Mul(yLocal, yLocal, tmp, n);
        AscendC::Adds(yLocal, yLocal, 8.830686063e-02f, n);
        AscendC::Mul(yLocal, yLocal, tmp, n);
        AscendC::Adds(yLocal, yLocal, -3.575934732e-01f, n);
        AscendC::Mul(yLocal, yLocal, tmp, n);
        AscendC::Adds(yLocal, yLocal, 1.124330570e+00f, n);
        AscendC::Mul(yLocal, yLocal, xLocal, n);

        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);

        GlobalTensor<float> yGm;
        yGm.SetGlobalBuffer((__gm__ float *)y + blockOff, n);
        DataCopy(yGm, yLocal, n);
    }
    else
    {
        TPipe pipe;
        KernelErfMulti<DT_X> op;
        op.Init(x, y, tilingData, &pipe);
        op.Process();
    }
}