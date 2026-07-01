// Kernel implementation for erf.
#include "kernel_operator.h"

#include "erf_tiling.h"
#include "tiling_key_erf.h"

constexpr uint32_t ALIGN_NUM = 8;
constexpr uint32_t TINY_TILE_LENGTH = 2048;

template <class DT_X, bool IS_ALIGNED>
class KernelErfTiny {
public:
    __aicore__ inline KernelErfTiny() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ErfTilingData &tilingData) {
        uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t coreUnits = tilingData.baseUnits;
        uint32_t startUnits = blockIdx * tilingData.baseUnits;
        if (tilingData.remainUnits != 0) {
            uint32_t beforeExtra = blockIdx < tilingData.remainUnits ? blockIdx : tilingData.remainUnits;
            coreUnits += blockIdx < tilingData.remainUnits ? 1U : 0U;
            startUnits += beforeExtra;
        }

        uint32_t coreOffset = startUnits * 16U;
        uint32_t maxLength = tilingData.length > coreOffset ? tilingData.length - coreOffset : 0U;
        uint32_t expectLength = coreUnits * 16U;
        coreLength = expectLength < maxLength ? expectLength : maxLength;

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + coreOffset, coreLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + coreOffset, coreLength);
    }

    __aicore__ inline void Process() {
        if (coreLength == 0) {
            return;
        }

        constexpr uint32_t xAddr = 0;
        constexpr uint32_t yAddr = xAddr + TINY_TILE_LENGTH * sizeof(DT_X);
        constexpr uint32_t x2Addr = yAddr + TINY_TILE_LENGTH * sizeof(DT_X);

        AscendC::LocalTensor<DT_X> xLocal(AscendC::TPosition::VECIN, xAddr, TINY_TILE_LENGTH);
        AscendC::LocalTensor<DT_X> yLocal(AscendC::TPosition::VECOUT, yAddr, TINY_TILE_LENGTH);
        AscendC::LocalTensor<DT_X> x2Local(AscendC::TPosition::VECCALC, x2Addr, TINY_TILE_LENGTH);

        if constexpr (IS_ALIGNED) {
            AscendC::DataCopy(xLocal, xGm, coreLength);
        } else {
            AscendC::DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(coreLength * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, static_cast<DT_X>(0)};
            AscendC::DataCopyPad(xLocal, xGm, copyParams, padParams);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

        Compute(yLocal, xLocal, x2Local, coreLength);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        if constexpr (IS_ALIGNED) {
            AscendC::DataCopy(yGm, yLocal, coreLength);
        } else {
            AscendC::DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(coreLength * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPad(yGm, yLocal, copyParams);
        }
    }

private:
    __aicore__ inline void Compute(const AscendC::LocalTensor<DT_X> &yLocal,
                                   const AscendC::LocalTensor<DT_X> &xLocal,
                                   const AscendC::LocalTensor<DT_X> &x2Local,
                                   uint32_t calCount) {
        constexpr DT_X CLIP = static_cast<DT_X>(2.17f);
        constexpr DT_X NEG_CLIP = static_cast<DT_X>(-2.17f);

        constexpr DT_X C0 = static_cast<DT_X>(1.1283791671f);
        constexpr DT_X C1 = static_cast<DT_X>(-0.368614014825f);
        constexpr DT_X C2 = static_cast<DT_X>(0.0966817963789f);
        constexpr DT_X C3 = static_cast<DT_X>(-0.0148689537407f);
        constexpr DT_X C4 = static_cast<DT_X>(0.000969055670676f);

        AscendC::Mins(xLocal, xLocal, CLIP, calCount);
        AscendC::Maxs(xLocal, xLocal, NEG_CLIP, calCount);
        AscendC::Mul(x2Local, xLocal, xLocal, calCount);

        AscendC::Muls(yLocal, x2Local, C4, calCount);
        AscendC::Adds(yLocal, yLocal, C3, calCount);
        AscendC::Mul(yLocal, yLocal, x2Local, calCount);
        AscendC::Adds(yLocal, yLocal, C2, calCount);
        AscendC::Mul(yLocal, yLocal, x2Local, calCount);
        AscendC::Adds(yLocal, yLocal, C1, calCount);
        AscendC::Mul(yLocal, yLocal, x2Local, calCount);
        AscendC::Adds(yLocal, yLocal, C0, calCount);
        AscendC::Mul(yLocal, yLocal, xLocal, calCount);
    }

private:
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t coreLength = 0;
};

template <class DT_X>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ErfTilingData &tilingData, AscendC::TPipe *pipeIn) {
        pipe = pipeIn;
        length = tilingData.length;
        blockLength = tilingData.blockLength;
        tileLength = tilingData.tileLength;

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x), length);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), length);
    }

    __aicore__ inline void Process() {
        pipe->InitBuffer(inQueueX, 2, tileLength * sizeof(DT_X));
        pipe->InitBuffer(outQueueY, 2, tileLength * sizeof(DT_X));
        pipe->InitBuffer(tmpBuf0, tileLength * sizeof(DT_X));

        uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t start = blockIdx * blockLength;
        if (start >= length) {
            return;
        }

        uint32_t coreLength = length - start;
        coreLength = coreLength > blockLength ? blockLength : coreLength;

        if (coreLength <= tileLength) {
            CopyIn(start, coreLength);
            Compute(coreLength);
            CopyOut(start, coreLength);
            return;
        }

        for (uint32_t offset = 0; offset < coreLength; offset += tileLength) {
            uint32_t calCount = coreLength - offset;
            calCount = calCount > tileLength ? tileLength : calCount;
            CopyIn(start + offset, calCount);
            Compute(calCount);
            CopyOut(start + offset, calCount);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t calCount) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        if ((calCount & (ALIGN_NUM - 1)) == 0) {
            AscendC::DataCopy(xLocal, xGm[offset], calCount);
        } else {
            AscendC::DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(calCount * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, static_cast<DT_X>(0)};
            AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        }
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t calCount) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> xClip = tmpBuf0.Get<DT_X>();

        constexpr DT_X CLIP = static_cast<DT_X>(2.17f);
        constexpr DT_X NEG_CLIP = static_cast<DT_X>(-2.17f);

        constexpr DT_X C0 = static_cast<DT_X>(1.1283791671f);
        constexpr DT_X C1 = static_cast<DT_X>(-0.368614014825f);
        constexpr DT_X C2 = static_cast<DT_X>(0.0966817963789f);
        constexpr DT_X C3 = static_cast<DT_X>(-0.0148689537407f);
        constexpr DT_X C4 = static_cast<DT_X>(0.000969055670676f);

        AscendC::Mins(xClip, xLocal, CLIP, calCount);
        AscendC::Maxs(xClip, xClip, NEG_CLIP, calCount);
        AscendC::Mul(yLocal, xClip, xClip, calCount);

        AscendC::Muls(xLocal, yLocal, C4, calCount);
        AscendC::Adds(xLocal, xLocal, C3, calCount);
        AscendC::Mul(xLocal, xLocal, yLocal, calCount);
        AscendC::Adds(xLocal, xLocal, C2, calCount);
        AscendC::Mul(xLocal, xLocal, yLocal, calCount);
        AscendC::Adds(xLocal, xLocal, C1, calCount);
        AscendC::Mul(xLocal, xLocal, yLocal, calCount);
        AscendC::Adds(xLocal, xLocal, C0, calCount);

        AscendC::Mul(yLocal, xClip, xLocal, calCount);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t calCount) {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        if ((calCount & (ALIGN_NUM - 1)) == 0) {
            AscendC::DataCopy(yGm[offset], yLocal, calCount);
        } else {
            AscendC::DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(calCount * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
        }
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::TPipe *pipe = nullptr;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0;
    uint32_t length = 0;
    uint32_t blockLength = 0;
    uint32_t tileLength = 0;
};

template <typename DT_X, auto MODE, auto... TILING_ARGS>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tilingData, tiling);
    if constexpr (MODE == 1) {
        AscendC::InitSocState();
        KernelErfTiny<DT_X, false> op;
        op.Init(x, y, tilingData);
        op.Process();
        return;
    } else if constexpr (MODE == 2) {
        AscendC::InitSocState();
        KernelErfTiny<DT_X, true> op;
        op.Init(x, y, tilingData);
        op.Process();
        return;
    }
    AscendC::TPipe pipe;
    KernelErf<DT_X> op;
    op.Init(x, y, tilingData, &pipe);
    op.Process();
}
