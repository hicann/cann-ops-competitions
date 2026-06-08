#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include "erf_tiling.h"
#include "tiling_key_erf.h"

// (3,3) rational fit over [-3.5657, 3.5657], float32 max_err = 9.5e-6
// erf(x) ~ x * P(x^2) / Q(x^2)
// P(z) = P0*z^3 + P1*z^2 + P2*z + P3
// Q(z) =    z^3 + Q0*z^2 + Q1*z + Q2   (leading 1 implicit)
constexpr float P0 = 7.8318489868e-02f;
constexpr float P1 = 4.7889168319e+00f;
constexpr float P2 = 1.8177672933e+01f;
constexpr float P3 = 1.2778993852e+02f;
constexpr float Q0 = 1.0893102865e+01f;
constexpr float Q1 = 5.3844448567e+01f;
constexpr float Q2 = 1.1325262336e+02f;
constexpr float C0 = 1.1280359266f;
constexpr float C1 = -3.7381502966e-01f;
constexpr float C2 = 1.0826675093e-01f;
constexpr float C3 = -2.2759529936e-02f;
constexpr float C4 = 3.2197281044e-03f;
constexpr float C5 = -2.6900824174e-04f;
constexpr float C6 = 9.8670905534e-06f;
constexpr float D0 = 1.1251893262f;
constexpr float D1 = -3.6061971277e-01f;
constexpr float D2 = 9.1030703722e-02f;
constexpr float D3 = -1.3353736032e-02f;
constexpr float D4 = 8.3155293218e-04f;
constexpr float E0 = 1.10009617f;
constexpr float E1 = -2.9413564e-01f;
constexpr float E2 = 4.500437e-02f;
constexpr float E3 = -2.11869e-03f;

template <class DT_X>
__aicore__ inline void FastErfPoly9Approx(const AscendC::LocalTensor<DT_X> &dst,
                                          const AscendC::LocalTensor<DT_X> &src,
                                          const AscendC::LocalTensor<DT_X> &z,
                                          uint32_t n)
{
    AscendC::Mul(z, src, src, n);
    AscendC::Muls(dst, z, static_cast<DT_X>(D4), n);
    AscendC::Adds(dst, dst, static_cast<DT_X>(D3), n);
    AscendC::Mul(dst, dst, z, n);
    AscendC::Adds(dst, dst, static_cast<DT_X>(D2), n);
    AscendC::Mul(dst, dst, z, n);
    AscendC::Adds(dst, dst, static_cast<DT_X>(D1), n);
    AscendC::Mul(dst, dst, z, n);
    AscendC::Adds(dst, dst, static_cast<DT_X>(D0), n);
    AscendC::Mul(dst, dst, src, n);
}

template <class DT_X>
__aicore__ inline void FastErfPoly7Approx(const AscendC::LocalTensor<DT_X> &dst,
                                          const AscendC::LocalTensor<DT_X> &src,
                                          const AscendC::LocalTensor<DT_X> &z,
                                          uint32_t n)
{
    AscendC::Mul(z, src, src, n);
    AscendC::Muls(dst, z, static_cast<DT_X>(E3), n);
    AscendC::Adds(dst, dst, static_cast<DT_X>(E2), n);
    AscendC::Mul(dst, dst, z, n);
    AscendC::Adds(dst, dst, static_cast<DT_X>(E1), n);
    AscendC::Mul(dst, dst, z, n);
    AscendC::Adds(dst, dst, static_cast<DT_X>(E0), n);
    AscendC::Mul(dst, dst, src, n);
}

template <class DT_X>
__aicore__ inline void ErfRationalApprox(const AscendC::LocalTensor<DT_X> &dst,
                                         const AscendC::LocalTensor<DT_X> &src,
                                         const AscendC::LocalTensor<DT_X> &den,
                                         const AscendC::LocalTensor<DT_X> &z,
                                         uint32_t n)
{
    AscendC::Mul(z, src, src, n);
    AscendC::Muls(dst, z, static_cast<DT_X>(P0), n);
    AscendC::Adds(dst, dst, static_cast<DT_X>(P1), n);
    AscendC::Adds(den, z, static_cast<DT_X>(Q0), n);
    AscendC::Mul(dst, dst, z, n);
    AscendC::Adds(dst, dst, static_cast<DT_X>(P2), n);
    AscendC::Mul(den, den, z, n);
    AscendC::Adds(den, den, static_cast<DT_X>(Q1), n);
    AscendC::Mul(dst, dst, z, n);
    AscendC::Adds(dst, dst, static_cast<DT_X>(P3), n);
    AscendC::Mul(den, den, z, n);
    AscendC::Adds(den, den, static_cast<DT_X>(Q2), n);
    AscendC::Div(dst, dst, den, n);
    AscendC::Mul(dst, dst, src, n);
}

template <class DT_X>
class KernelErfMedium {
public:
    __aicore__ inline KernelErfMedium() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileDataNum) {
        this->totalLength = totalLength; this->tileDataNum = tileDataNum;
        uint32_t bd = AscendC::GetBlockNum(), bi = AscendC::GetBlockIdx(), bs = totalLength / bd, r = totalLength % bd;
        this->coreDataNum = bs + (bi < r ? 1u : 0u); this->coreStart = bi * bs + (bi < r ? bi : r);
        if (this->coreDataNum == 0) return;
        this->tileNum = (this->coreDataNum + tileDataNum - 1) / tileDataNum;
        this->tailDataNum = this->coreDataNum - (this->tileNum - 1) * tileDataNum;
        if (this->tailDataNum == 0) this->tailDataNum = tileDataNum;
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + this->coreStart, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + this->coreStart, this->coreDataNum);
        pipe.InitBuffer(bufIn, this->tileDataNum * sizeof(DT_X));
        pipe.InitBuffer(bufX2, this->tileDataNum * sizeof(DT_X) + 256);
        pipe.InitBuffer(bufAcc, this->tileDataNum * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (this->coreDataNum == 0) return;
        for (uint32_t i = 0; i < this->tileNum; i++) {
            uint32_t n = (i == this->tileNum - 1) ? this->tailDataNum : this->tileDataNum;
            uint32_t off = i * this->tileDataNum;
            AscendC::DataCopyParams cp{1, static_cast<uint16_t>(n * sizeof(DT_X)), 0, 0};
            AscendC::DataCopyPadParams pp{false, 0, 0, 0};
            auto xL = bufIn.Get<DT_X>();
            AscendC::DataCopyPad(xL, xGm[off], cp, pp);
            event_t eI = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eI); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eI);
            auto pX = bufX2.Get<DT_X>(), pC = bufAcc.Get<DT_X>();
            FastErfPoly9Approx(pC, xL, pX, n);
            AscendC::Mins(pC, pC, static_cast<DT_X>(1.0f), n); AscendC::Maxs(pC, pC, static_cast<DT_X>(-1.0f), n);
            event_t eO = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eO); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eO);
            AscendC::DataCopyPad(yGm[off], pC, cp);
        }
    }
private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bufIn, bufX2, bufAcc;
    AscendC::GlobalTensor<DT_X> xGm, yGm;
    uint32_t totalLength, tileDataNum, coreDataNum, coreStart, tileNum, tailDataNum;
};

template <class DT_X>
class KernelErfMediumDiv {
public:
    __aicore__ inline KernelErfMediumDiv() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileDataNum) {
        this->totalLength = totalLength; this->tileDataNum = tileDataNum;
        uint32_t bd = AscendC::GetBlockNum(), bi = AscendC::GetBlockIdx(), bs = totalLength / bd, r = totalLength % bd;
        this->coreDataNum = bs + (bi < r ? 1u : 0u); this->coreStart = bi * bs + (bi < r ? bi : r);
        if (this->coreDataNum == 0) return;
        this->tileNum = (this->coreDataNum + tileDataNum - 1) / tileDataNum;
        this->tailDataNum = this->coreDataNum - (this->tileNum - 1) * tileDataNum;
        if (this->tailDataNum == 0) this->tailDataNum = tileDataNum;
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + this->coreStart, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + this->coreStart, this->coreDataNum);
    }

    __aicore__ inline void Process() {
        if (this->coreDataNum == 0) return;
        uint32_t tileBytes = this->tileDataNum * sizeof(DT_X);
        uint32_t xAddr = 0;
        uint32_t aAddr = tileBytes;
        uint32_t x2Addr = tileBytes * 2;
        uint32_t cAddr = tileBytes * 3 + 256;

        AscendC::LocalTensor<DT_X> xL(AscendC::TPosition::VECIN, xAddr, this->tileDataNum);
        AscendC::LocalTensor<DT_X> pA(AscendC::TPosition::VECCALC, aAddr, this->tileDataNum);
        AscendC::LocalTensor<DT_X> pX(AscendC::TPosition::VECCALC, x2Addr, this->tileDataNum);
        AscendC::LocalTensor<DT_X> pC(AscendC::TPosition::VECOUT, cAddr, this->tileDataNum);

        for (uint32_t i = 0; i < this->tileNum; i++) {
            uint32_t n = (i == this->tileNum - 1) ? this->tailDataNum : this->tileDataNum;
            uint32_t off = i * this->tileDataNum;
            AscendC::DataCopyParams cp{1, static_cast<uint16_t>(n * sizeof(DT_X)), 0, 0};
            AscendC::DataCopyPadParams pp{false, 0, 0, 0};
            AscendC::DataCopyPad(xL, xGm[off], cp, pp);
            event_t eI = static_cast<event_t>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eI);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eI);
            FastErfPoly9Approx(pC, xL, pX, n);
            AscendC::Mins(pC, pC, static_cast<DT_X>(1.0f), n); AscendC::Maxs(pC, pC, static_cast<DT_X>(-1.0f), n);
            event_t eO = static_cast<event_t>(0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eO);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eO);
            AscendC::DataCopyPad(yGm[off], pC, cp);
        }
    }
private:
    AscendC::GlobalTensor<DT_X> xGm, yGm;
    uint32_t totalLength, tileDataNum, coreDataNum, coreStart, tileNum, tailDataNum;
};

template <class DT_X>
class KernelErfMediumQueue {
public:
    __aicore__ inline KernelErfMediumQueue() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileDataNum) {
        this->totalLength = totalLength; this->tileDataNum = tileDataNum;
        uint32_t bd = AscendC::GetBlockNum(), bi = AscendC::GetBlockIdx(), bs = totalLength / bd, r = totalLength % bd;
        this->coreDataNum = bs + (bi < r ? 1u : 0u); this->coreStart = bi * bs + (bi < r ? bi : r);
        if (this->coreDataNum == 0) return;
        this->tileNum = (this->coreDataNum + tileDataNum - 1) / tileDataNum;
        this->tailDataNum = this->coreDataNum - (this->tileNum - 1) * tileDataNum;
        if (this->tailDataNum == 0) this->tailDataNum = tileDataNum;
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + this->coreStart, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + this->coreStart, this->coreDataNum);
        pipe.InitBuffer(inQueueX, 1, this->tileDataNum * sizeof(DT_X));
        pipe.InitBuffer(bufX2, this->tileDataNum * sizeof(DT_X) + 256);
        pipe.InitBuffer(bufAcc, this->tileDataNum * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (this->coreDataNum == 0) return;
        for (uint32_t i = 0; i < this->tileNum; i++) {
            uint32_t n = (i == this->tileNum - 1) ? this->tailDataNum : this->tileDataNum;
            uint32_t off = i * this->tileDataNum;
            auto xL = inQueueX.template AllocTensor<DT_X>();
            AscendC::DataCopyParams cp{1, static_cast<uint16_t>(n * sizeof(DT_X)), 0, 0};
            AscendC::DataCopyPadParams pp{false, 0, 0, 0};
            AscendC::DataCopyPad(xL, xGm[off], cp, pp); inQueueX.EnQue(xL);
            xL = inQueueX.template DeQue<DT_X>();
            auto pX = bufX2.Get<DT_X>(), pC = bufAcc.Get<DT_X>();
            FastErfPoly9Approx(pC, xL, pX, n);
            AscendC::Mins(pC, pC, static_cast<DT_X>(1.0f), n); AscendC::Maxs(pC, pC, static_cast<DT_X>(-1.0f), n);
            AscendC::DataCopyPad(yGm[off], pC, cp); inQueueX.FreeTensor(xL);
        }
    }
private:
    AscendC::TPipe pipe; AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bufX2, bufAcc;
    AscendC::GlobalTensor<DT_X> xGm, yGm;
    uint32_t totalLength, tileDataNum, coreDataNum, coreStart, tileNum, tailDataNum;
};

template <class DT_X>
class KernelErfMediumAligned {
public:
    __aicore__ inline KernelErfMediumAligned() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileDataNum) {
        this->totalLength = totalLength; this->tileDataNum = tileDataNum;
        uint32_t bd = AscendC::GetBlockNum(), bi = AscendC::GetBlockIdx(), bs = totalLength / bd;
        this->coreDataNum = bs; this->coreStart = bi * bs;
        if (this->coreDataNum == 0) return;
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + this->coreStart, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + this->coreStart, this->coreDataNum);
    }

    __aicore__ inline void Process() {
        if (this->coreDataNum == 0) return;
        uint32_t n = this->coreDataNum;
        uint32_t tileBytes = this->tileDataNum * sizeof(DT_X);
        uint32_t xAddr = 0;
        uint32_t x2Addr = tileBytes;
        uint32_t cAddr = tileBytes * 2 + 256;

        AscendC::LocalTensor<DT_X> xL(AscendC::TPosition::VECIN, xAddr, this->tileDataNum);
        AscendC::LocalTensor<DT_X> pX(AscendC::TPosition::VECCALC, x2Addr, this->tileDataNum);
        AscendC::LocalTensor<DT_X> pC(AscendC::TPosition::VECOUT, cAddr, this->tileDataNum);

        AscendC::DataCopy(xL, xGm[0], n);
        event_t eI = static_cast<event_t>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eI);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eI);
        FastErfPoly9Approx(pC, xL, pX, n);
        AscendC::Mins(pC, pC, static_cast<DT_X>(1.0f), n); AscendC::Maxs(pC, pC, static_cast<DT_X>(-1.0f), n);
        event_t eO = static_cast<event_t>(0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eO);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eO);
        AscendC::DataCopy(yGm[0], pC, n);
    }
private:
    AscendC::GlobalTensor<DT_X> xGm, yGm;
    uint32_t totalLength, tileDataNum, coreDataNum, coreStart;
};

template <class DT_X>
class KernelErfLarge {
public:
    __aicore__ inline KernelErfLarge() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileDataNum) {
        this->totalLength = totalLength; this->tileDataNum = tileDataNum;
        uint32_t bd = AscendC::GetBlockNum(), bi = AscendC::GetBlockIdx(), bs = totalLength / bd, r = totalLength % bd;
        this->coreDataNum = bs + (bi < r ? 1u : 0u); this->coreStart = bi * bs + (bi < r ? bi : r);
        if (this->coreDataNum == 0) return;
        this->tileNum = (this->coreDataNum + tileDataNum - 1) / tileDataNum;
        this->tailDataNum = this->coreDataNum - (this->tileNum - 1) * tileDataNum;
        if (this->tailDataNum == 0) this->tailDataNum = tileDataNum;
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + this->coreStart, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + this->coreStart, this->coreDataNum);
        pipe.InitBuffer(inQueueX, 2, this->tileDataNum * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, 2, this->tileDataNum * sizeof(DT_X));
        pipe.InitBuffer(bufAbs, this->tileDataNum * sizeof(DT_X));
        pipe.InitBuffer(bufX2, this->tileDataNum * sizeof(DT_X) + 256);
        pipe.InitBuffer(bufAcc, this->tileDataNum * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (this->coreDataNum == 0) return;
        for (uint32_t i = 0; i < this->tileNum; i++) { CopyIn(i); Compute(i); CopyOut(i); }
    }
private:
    __aicore__ inline void CopyIn(int32_t p) {
        uint32_t n = (p == this->tileNum - 1) ? this->tailDataNum : this->tileDataNum;
        uint32_t o = p * this->tileDataNum;
        auto xL = inQueueX.template AllocTensor<DT_X>();
        AscendC::DataCopyParams cp{1, static_cast<uint16_t>(n * sizeof(DT_X)), 0, 0};
        AscendC::DataCopyPadParams pp{false, 0, 0, 0};
        AscendC::DataCopyPad(xL, xGm[o], cp, pp); inQueueX.EnQue(xL);
    }

    __aicore__ inline void Compute(int32_t p) {
        uint32_t n = (p == this->tileNum - 1) ? this->tailDataNum : this->tileDataNum;
        auto xL = inQueueX.template DeQue<DT_X>();
        auto pA = bufAbs.Get<DT_X>(), pX = bufX2.Get<DT_X>(), pC = bufAcc.Get<DT_X>();
        auto zL = outQueueY.template AllocTensor<DT_X>();
        FastErfPoly9Approx(pC, xL, pX, n);
        AscendC::Mins(pC, pC, static_cast<DT_X>(1.0f), n); AscendC::Maxs(zL, pC, static_cast<DT_X>(-1.0f), n);
        outQueueY.template EnQue<DT_X>(zL); inQueueX.FreeTensor(xL);
    }

    __aicore__ inline void CopyOut(int32_t p) {
        auto zL = outQueueY.template DeQue<DT_X>();
        uint32_t n = (p == this->tileNum - 1) ? this->tailDataNum : this->tileDataNum;
        uint32_t o = p * this->tileDataNum;
        AscendC::DataCopyParams cp{1, static_cast<uint16_t>(n * sizeof(DT_X)), 0, 0};
        AscendC::DataCopyPad(yGm[o], zL, cp); outQueueY.FreeTensor(zL);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bufAbs, bufX2, bufAcc;
    AscendC::GlobalTensor<DT_X> xGm, yGm;
    uint32_t totalLength, tileDataNum, coreDataNum, coreStart, tileNum, tailDataNum;
};

template <class DT_X>
class KernelErfDirect {
public:
    __aicore__ inline KernelErfDirect() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileDataNum) {
        this->dataNum = totalLength; this->bufSize = tileDataNum;
        xGm.SetGlobalBuffer((__gm__ DT_X *)x, this->dataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, this->dataNum);
        pipe.InitBuffer(bufX, this->bufSize * sizeof(DT_X));
        pipe.InitBuffer(bufX2, this->bufSize * sizeof(DT_X) + 256);
        pipe.InitBuffer(bufAcc, this->bufSize * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (this->dataNum == 0) return;
        uint32_t n = this->dataNum;
        auto pX = bufX.Get<DT_X>(), pX2 = bufX2.Get<DT_X>(), pC = bufAcc.Get<DT_X>();
        AscendC::DataCopyParams cp{1, static_cast<uint16_t>(n * sizeof(DT_X)), 0, 0};
        AscendC::DataCopyPadParams pp{false, 0, 0, 0};
        AscendC::DataCopyPad(pX, xGm[0], cp, pp);
        event_t eI = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eI); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eI);
        FastErfPoly9Approx(pC, pX, pX2, n);
        AscendC::Mins(pC, pC, static_cast<DT_X>(1.0f), n); AscendC::Maxs(pC, pC, static_cast<DT_X>(-1.0f), n);
        event_t eO = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eO); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eO);
        AscendC::DataCopyPad(yGm[0], pC, cp);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bufX, bufX2, bufAcc;
    AscendC::GlobalTensor<DT_X> xGm, yGm;
    uint32_t dataNum, bufSize;
};

template <class DT_X>
class KernelErfDirectStatic {
public:
    __aicore__ inline KernelErfDirectStatic() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileDataNum) {
        this->dataNum = totalLength; this->bufSize = tileDataNum;
        xGm.SetGlobalBuffer((__gm__ DT_X *)x, this->dataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, this->dataNum);
    }

    __aicore__ inline void Process() {
        if (this->dataNum == 0) return;
        uint32_t n = this->dataNum;
        uint32_t tileBytes = this->bufSize * sizeof(DT_X);
        uint32_t xAddr = 0;
        uint32_t aAddr = tileBytes;
        uint32_t x2Addr = tileBytes * 2;
        uint32_t cAddr = tileBytes * 3 + 256;

        AscendC::LocalTensor<DT_X> pX(AscendC::TPosition::VECIN, xAddr, this->bufSize);
        AscendC::LocalTensor<DT_X> pA(AscendC::TPosition::VECCALC, aAddr, this->bufSize);
        AscendC::LocalTensor<DT_X> pX2(AscendC::TPosition::VECCALC, x2Addr, this->bufSize);
        AscendC::LocalTensor<DT_X> pC(AscendC::TPosition::VECOUT, cAddr, this->bufSize);

        AscendC::DataCopyParams cp{1, static_cast<uint16_t>(n * sizeof(DT_X)), 0, 0};
        AscendC::DataCopyPadParams pp{false, 0, 0, 0};
        AscendC::DataCopyPad(pX, xGm[0], cp, pp);
        event_t eI = static_cast<event_t>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eI);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eI);
        ErfRationalApprox(pC, pX, pA, pX2, n);
        AscendC::Mins(pC, pC, static_cast<DT_X>(1.0f), n); AscendC::Maxs(pC, pC, static_cast<DT_X>(-1.0f), n);
        event_t eO = static_cast<event_t>(0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eO);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eO);
        AscendC::DataCopyPad(yGm[0], pC, cp);
    }
private:
    AscendC::GlobalTensor<DT_X> xGm, yGm;
    uint32_t dataNum, bufSize;
};

template <class DT_X>
class KernelErfDirectStaticAligned {
public:
    __aicore__ inline KernelErfDirectStaticAligned() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileDataNum) {
        this->dataNum = totalLength; this->bufSize = tileDataNum;
        xGm.SetGlobalBuffer((__gm__ DT_X *)x, this->dataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, this->dataNum);
    }

    __aicore__ inline void Process() {
        if (this->dataNum == 0) return;
        uint32_t n = this->dataNum;
        uint32_t tileBytes = this->bufSize * sizeof(DT_X);
        uint32_t xAddr = 0;
        uint32_t aAddr = tileBytes;
        uint32_t x2Addr = tileBytes * 2;
        uint32_t cAddr = tileBytes * 3 + 256;

        AscendC::LocalTensor<DT_X> pX(AscendC::TPosition::VECIN, xAddr, this->bufSize);
        AscendC::LocalTensor<DT_X> pA(AscendC::TPosition::VECCALC, aAddr, this->bufSize);
        AscendC::LocalTensor<DT_X> pX2(AscendC::TPosition::VECCALC, x2Addr, this->bufSize);
        AscendC::LocalTensor<DT_X> pC(AscendC::TPosition::VECOUT, cAddr, this->bufSize);

        AscendC::DataCopy(pX, xGm[0], n);
        event_t eI = static_cast<event_t>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eI);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eI);
        ErfRationalApprox(pC, pX, pA, pX2, n);
        AscendC::Mins(pC, pC, static_cast<DT_X>(1.0f), n); AscendC::Maxs(pC, pC, static_cast<DT_X>(-1.0f), n);
        event_t eO = static_cast<event_t>(0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eO);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eO);
        AscendC::DataCopy(yGm[0], pC, n);
    }
private:
    AscendC::GlobalTensor<DT_X> xGm, yGm;
    uint32_t dataNum, bufSize;
};

template <class DT_X>
class KernelErfDirectRepeatAligned {
public:
    __aicore__ inline KernelErfDirectRepeatAligned() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileDataNum) {
        this->dataNum = totalLength; this->bufSize = tileDataNum;
        xGm.SetGlobalBuffer((__gm__ DT_X *)x, this->dataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, this->dataNum);
    }

    __aicore__ inline void Process() {
        if (this->dataNum == 0) return;
        uint32_t n = this->dataNum;
        uint32_t tileBytes = this->bufSize * sizeof(DT_X);
        uint32_t xAddr = 0;
        uint32_t aAddr = tileBytes;
        uint32_t x2Addr = tileBytes * 2;
        uint32_t cAddr = tileBytes * 3 + 256;

        AscendC::LocalTensor<DT_X> pX(AscendC::TPosition::VECIN, xAddr, this->bufSize);
        AscendC::LocalTensor<DT_X> pA(AscendC::TPosition::VECCALC, aAddr, this->bufSize);
        AscendC::LocalTensor<DT_X> pX2(AscendC::TPosition::VECCALC, x2Addr, this->bufSize);
        AscendC::LocalTensor<DT_X> pC(AscendC::TPosition::VECOUT, cAddr, this->bufSize);

        AscendC::DataCopy(pX, xGm[0], n);
        event_t eI = static_cast<event_t>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eI);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eI);

        uint8_t rpt = static_cast<uint8_t>(n >> 6);
        AscendC::SetVectorMask<DT_X, AscendC::MaskMode::NORMAL>(64);
        AscendC::Mul<DT_X, false>(pX2, pX, pX, AscendC::MASK_PLACEHOLDER, rpt, {1, 1, 1, 8, 8, 8});
        AscendC::Muls<DT_X, false>(pC, pX2, static_cast<DT_X>(P0), AscendC::MASK_PLACEHOLDER, rpt, {1, 1, 8, 8});
        AscendC::Adds<DT_X, false>(pC, pC, static_cast<DT_X>(P1), AscendC::MASK_PLACEHOLDER, rpt, {1, 1, 8, 8});
        AscendC::Adds<DT_X, false>(pA, pX2, static_cast<DT_X>(Q0), AscendC::MASK_PLACEHOLDER, rpt, {1, 1, 8, 8});
        AscendC::Mul<DT_X, false>(pC, pC, pX2, AscendC::MASK_PLACEHOLDER, rpt, {1, 1, 1, 8, 8, 8});
        AscendC::Adds<DT_X, false>(pC, pC, static_cast<DT_X>(P2), AscendC::MASK_PLACEHOLDER, rpt, {1, 1, 8, 8});
        AscendC::Mul<DT_X, false>(pA, pA, pX2, AscendC::MASK_PLACEHOLDER, rpt, {1, 1, 1, 8, 8, 8});
        AscendC::Adds<DT_X, false>(pA, pA, static_cast<DT_X>(Q1), AscendC::MASK_PLACEHOLDER, rpt, {1, 1, 8, 8});
        AscendC::Mul<DT_X, false>(pC, pC, pX2, AscendC::MASK_PLACEHOLDER, rpt, {1, 1, 1, 8, 8, 8});
        AscendC::Adds<DT_X, false>(pC, pC, static_cast<DT_X>(P3), AscendC::MASK_PLACEHOLDER, rpt, {1, 1, 8, 8});
        AscendC::Mul<DT_X, false>(pA, pA, pX2, AscendC::MASK_PLACEHOLDER, rpt, {1, 1, 1, 8, 8, 8});
        AscendC::Adds<DT_X, false>(pA, pA, static_cast<DT_X>(Q2), AscendC::MASK_PLACEHOLDER, rpt, {1, 1, 8, 8});
        AscendC::Div<DT_X, false>(pC, pC, pA, AscendC::MASK_PLACEHOLDER, rpt, {1, 1, 1, 8, 8, 8});
        AscendC::Mul<DT_X, false>(pC, pC, pX, AscendC::MASK_PLACEHOLDER, rpt, {1, 1, 1, 8, 8, 8});
        AscendC::Mins<DT_X, false>(pC, pC, static_cast<DT_X>(1.0f), AscendC::MASK_PLACEHOLDER, rpt, {1, 1, 8, 8});
        AscendC::Maxs<DT_X, false>(pC, pC, static_cast<DT_X>(-1.0f), AscendC::MASK_PLACEHOLDER, rpt, {1, 1, 8, 8});

        event_t eO = static_cast<event_t>(0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eO);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eO);
        AscendC::DataCopy(yGm[0], pC, n);
    }
private:
    AscendC::GlobalTensor<DT_X> xGm, yGm;
    uint32_t dataNum, bufSize;
};

template <class DT_X>
class KernelErfDirectAligned {
public:
    __aicore__ inline KernelErfDirectAligned() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileDataNum) {
        this->dataNum = totalLength; this->bufSize = tileDataNum;
        xGm.SetGlobalBuffer((__gm__ DT_X *)x, this->dataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, this->dataNum);
        pipe.InitBuffer(bufX, this->bufSize * sizeof(DT_X));
        pipe.InitBuffer(bufX2, this->bufSize * sizeof(DT_X) + 256);
        pipe.InitBuffer(bufAcc, this->bufSize * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (this->dataNum == 0) return;
        uint32_t n = this->dataNum;
        auto pX = bufX.Get<DT_X>(), pX2 = bufX2.Get<DT_X>(), pC = bufAcc.Get<DT_X>();
        AscendC::DataCopy(pX, xGm[0], n);
        event_t eI = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eI); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eI);
        FastErfPoly9Approx(pC, pX, pX2, n);
        AscendC::Mins(pC, pC, static_cast<DT_X>(1.0f), n); AscendC::Maxs(pC, pC, static_cast<DT_X>(-1.0f), n);
        event_t eO = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eO); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eO);
        AscendC::DataCopy(yGm[0], pC, n);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bufX, bufX2, bufAcc;
    AscendC::GlobalTensor<DT_X> xGm, yGm;
    uint32_t dataNum, bufSize;
};

template <typename DT_X, int IS_SPLIT>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (IS_SPLIT == 0) { KernelErfLarge<DT_X> op; op.Init(x,y,tiling_data.totalLength,tiling_data.tileDataNum); op.Process(); }
    else if constexpr (IS_SPLIT == 1) { KernelErfMediumQueue<DT_X> op; op.Init(x,y,tiling_data.totalLength,tiling_data.tileDataNum); op.Process(); }
    else if constexpr (IS_SPLIT == 4) { AscendC::InitSocState(); KernelErfMediumAligned<DT_X> op; op.Init(x,y,tiling_data.totalLength,tiling_data.tileDataNum); op.Process(); }
    else if constexpr (IS_SPLIT == 5) { AscendC::InitSocState(); KernelErfDirectStatic<DT_X> op; op.Init(x,y,tiling_data.totalLength,tiling_data.tileDataNum); op.Process(); }
    else if constexpr (IS_SPLIT == 7) { AscendC::InitSocState(); KernelErfDirectStaticAligned<DT_X> op; op.Init(x,y,tiling_data.totalLength,tiling_data.tileDataNum); op.Process(); }
    else { KernelErfDirect<DT_X> op; op.Init(x,y,tiling_data.totalLength,tiling_data.tileDataNum); op.Process(); }
}
