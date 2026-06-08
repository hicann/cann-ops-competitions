#include "kernel_operator.h"
#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

constexpr float C1  =  1.127609275346750e+00f;
constexpr float C3  = -3.685559188684842e-01f;
constexpr float C5  =  9.942597347522031e-02f;
constexpr float C7  = -1.730947175481838e-02f;
constexpr float C9  =  1.687210840969595e-03f;
constexpr float C11 = -6.882775119509469e-05f;
constexpr float CLAMP_HI = 2.6f, CLAMP_LO = -2.6f;

__aicore__ inline void ProbeScalarSpin() {
    volatile uint32_t acc = 0x12345678u;
#pragma unroll 1
    for (uint32_t i = 0; i < kErfProbeSpin; ++i) {
        acc = acc * 1664525u + 1013904223u;
    }
}

template <class DT_X>
class KernelErf {
public:
    __aicore__ inline KernelErf() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, ErfTilingData* td) {
        uint32_t coreId = GetBlockIdx();
        uint64_t cd = td->coreDataLength, tl = td->totalLength;
        uint64_t off = coreId * cd;
        if (off >= tl) { processLength = 0; return; }
        processLength = cd; if (off + cd > tl) processLength = tl - off;
        xGm.SetGlobalBuffer((__gm__ DT_X*)x + off, processLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + off, processLength);
        tileLength = td->tileLength;
        fullTileNum = processLength / tileLength;
        tailTileLength = processLength - static_cast<uint64_t>(fullTileNum) * tileLength;
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(calcBuf, 2 * tileLength * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        if (processLength == 0) return;
        if (fullTileNum == 0 && tailTileLength > 0) { ProcessFlat(); return; }
        if constexpr (kErfProbeMode == ERF_PROBE_FIXED_SCALAR) ProbeScalarSpin();
        uint64_t off = 0;
        for (uint32_t i = 0; i < fullTileNum; ++i, off += tileLength) { CopyInFull(off); ComputeFull(); CopyOutFull(off); }
        if (tailTileLength > 0) { CopyInTail(off); ComputeTail(); CopyOutTail(off); }
    }
private:
    __aicore__ inline void ProcessFlat() {
        uint32_t N = (processLength + 7) / 8 * 8;
        if constexpr (kErfProbeMode == ERF_PROBE_PER_TILE_SCALAR) ProbeScalarSpin();
        LocalTensor<DT_X> xL = inQueueX.AllocTensor<DT_X>();
        LocalTensor<DT_X> yL = outQueueY.AllocTensor<DT_X>();
        auto zL = calcBuf.Get<DT_X>(); auto pL = zL[tileLength];
        uint32_t al = (processLength / 8) * 8;
        if (al > 0) DataCopy(xL, xGm[0], al);
        for (uint32_t j = al; j < processLength; ++j) xL.SetValue(j, xGm.GetValue(j));
        auto id = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(id); WaitFlag<HardEvent::MTE2_V>(id);
        Mins(xL, xL, (DT_X)CLAMP_HI, N); Maxs(xL, xL, (DT_X)CLAMP_LO, N);
        Mul(zL, xL, xL, N);
        Muls(pL, zL, (DT_X)C11, N); Adds(pL, pL, (DT_X)C9, N);
        Mul(pL, zL, pL, N); Adds(pL, pL, (DT_X)C7, N);
        Mul(pL, zL, pL, N); Adds(pL, pL, (DT_X)C5, N);
        Mul(pL, zL, pL, N); Adds(pL, pL, (DT_X)C3, N);
        Mul(pL, zL, pL, N); Adds(pL, pL, (DT_X)C1, N);
        Mul(yL, xL, pL, N);
        auto idV = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(idV); WaitFlag<HardEvent::V_MTE3>(idV);
        if (al > 0) DataCopy(yGm[0], yL, al);
        for (uint32_t j = al; j < processLength; ++j) yGm.SetValue(j, yL.GetValue(j));
    }
    __aicore__ inline void RunPerTileProbe() { if constexpr (kErfProbeMode == ERF_PROBE_PER_TILE_SCALAR) ProbeScalarSpin(); }
    __aicore__ inline void RunVectorProbe(LocalTensor<DT_X> zL, LocalTensor<DT_X> pL, LocalTensor<DT_X> qL, uint32_t N) {
        if constexpr (kErfProbeMode == ERF_PROBE_EXTRA_VECTOR) {
            constexpr uint32_t kR = (ERF_PROBE_SPIN / 64) > 0 ? (ERF_PROBE_SPIN / 64) : 1;
#pragma unroll 1
            for (uint32_t i = 0; i < kR; ++i) { Muls(qL, zL, (DT_X)0.03125f, N); Adds(pL, pL, (DT_X)0.125f, N); Mul(zL, qL, pL, N); }
        }
    }
    __aicore__ inline void CopyInRange(LocalTensor<DT_X> xL, uint64_t off, uint32_t len) {
        if constexpr (kErfProbeMode == ERF_PROBE_SCALAR_COPY) {
            for (uint32_t j = 0; j < len; ++j) xL.SetValue(j, xGm.GetValue(off + j)); return;
        }
        uint32_t a = (len / 8) * 8; if (a > 0) DataCopy(xL, xGm[off], a);
        for (uint32_t j = a; j < len; ++j) xL.SetValue(j, xGm.GetValue(off + j));
    }
    __aicore__ inline void CopyOutRange(LocalTensor<DT_X> yL, uint64_t off, uint32_t len) {
        if constexpr (kErfProbeMode == ERF_PROBE_SCALAR_COPY) {
            for (uint32_t j = 0; j < len; ++j) yGm.SetValue(off + j, yL.GetValue(j)); return;
        }
        uint32_t a = (len / 8) * 8;
        for (uint32_t j = a; j < len; ++j) yGm.SetValue(off + j, yL.GetValue(j));
        if (a > 0) DataCopy(yGm[off], yL, a);
    }
    __aicore__ inline void CopyInFull(uint64_t off) {
        auto xL = inQueueX.AllocTensor<DT_X>(); CopyInRange(xL, off, tileLength); inQueueX.EnQue(xL);
    }
    __aicore__ inline void ComputeFull() {
        RunPerTileProbe();
        auto xL = inQueueX.DeQue<DT_X>(); auto yL = outQueueY.AllocTensor<DT_X>();
        auto zL = calcBuf.Get<DT_X>(); auto pL = zL[tileLength]; uint32_t N = tileLength;
        Mins(xL, xL, (DT_X)CLAMP_HI, N); Maxs(xL, xL, (DT_X)CLAMP_LO, N);
        Mul(zL, xL, xL, N);
        Muls(pL, zL, (DT_X)C11, N); Adds(pL, pL, (DT_X)C9, N);
        Mul(pL, zL, pL, N); Adds(pL, pL, (DT_X)C7, N);
        Mul(pL, zL, pL, N); Adds(pL, pL, (DT_X)C5, N);
        Mul(pL, zL, pL, N); Adds(pL, pL, (DT_X)C3, N);
        Mul(pL, zL, pL, N); Adds(pL, pL, (DT_X)C1, N);
        Mul(yL, xL, pL, N);
        RunVectorProbe(zL, pL, zL, N);
        outQueueY.EnQue(yL); inQueueX.FreeTensor(xL);
    }
    __aicore__ inline void CopyOutFull(uint64_t off) {
        auto yL = outQueueY.DeQue<DT_X>(); CopyOutRange(yL, off, tileLength); outQueueY.FreeTensor(yL);
    }
    __aicore__ inline void CopyInTail(uint64_t off) {
        auto xL = inQueueX.AllocTensor<DT_X>(); CopyInRange(xL, off, tailTileLength); inQueueX.EnQue(xL);
    }
    __aicore__ inline void ComputeTail() {
        RunPerTileProbe(); uint32_t N = (tailTileLength + 7) / 8 * 8;
        auto xL = inQueueX.DeQue<DT_X>(); auto yL = outQueueY.AllocTensor<DT_X>();
        auto zL = calcBuf.Get<DT_X>(); auto pL = zL[tileLength];
        Mins(xL, xL, (DT_X)CLAMP_HI, N); Maxs(xL, xL, (DT_X)CLAMP_LO, N);
        Mul(zL, xL, xL, N);
        Muls(pL, zL, (DT_X)C11, N); Adds(pL, pL, (DT_X)C9, N);
        Mul(pL, zL, pL, N); Adds(pL, pL, (DT_X)C7, N);
        Mul(pL, zL, pL, N); Adds(pL, pL, (DT_X)C5, N);
        Mul(pL, zL, pL, N); Adds(pL, pL, (DT_X)C3, N);
        Mul(pL, zL, pL, N); Adds(pL, pL, (DT_X)C1, N);
        Mul(yL, xL, pL, N);
        RunVectorProbe(zL, pL, zL, N);
        outQueueY.EnQue(yL); inQueueX.FreeTensor(xL);
    }
    __aicore__ inline void CopyOutTail(uint64_t off) {
        auto yL = outQueueY.DeQue<DT_X>(); CopyOutRange(yL, off, tailTileLength); outQueueY.FreeTensor(yL);
    }
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<TPosition::VECCALC> calcBuf;
    GlobalTensor<DT_X> xGm, yGm;
    uint64_t processLength;
    uint32_t tileLength, fullTileNum, tailTileLength;
};

template <typename DT_X>
__global__ __aicore__ void erf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GlobalTensor<uint64_t> tg;
    tg.SetGlobalBuffer((__gm__ uint64_t*)tiling, 3);
    uint64_t totalLength = tg.GetValue(0);
    uint64_t coreDataLength = tg.GetValue(1);
    uint32_t coreId = GetBlockIdx();
    uint64_t off = coreId * coreDataLength;
    if (off >= totalLength) return;
    uint64_t length = coreDataLength;
    if (off + coreDataLength > totalLength) length = totalLength - off;
    if (length <= 32) {
        __gm__ DT_X* xP = (__gm__ DT_X*)x + off;
        __gm__ DT_X* yP = (__gm__ DT_X*)y + off;
        for (uint32_t j = 0; j < length; ++j) {
            DT_X v = xP[j];
            if (v > (DT_X)CLAMP_HI) v = (DT_X)CLAMP_HI;
            if (v < (DT_X)CLAMP_LO) v = (DT_X)CLAMP_LO;
            DT_X z = v * v; DT_X p = C11;
            p = p * z + C9; p = p * z + C7; p = p * z + C5; p = p * z + C3; p = p * z + C1;
            yP[j] = v * p;
        }
        return;
    }
    if (length <= 48 && (length % 8 == 0)) {
        uint32_t NK = length;
        LocalMemAllocator alloc;
        auto xL = alloc.Alloc<TPosition::VECIN, DT_X>(NK);
        auto tL = alloc.Alloc<TPosition::VECCALC, DT_X>(NK);
        auto yL = alloc.Alloc<TPosition::VECOUT, DT_X>(NK);
        GlobalTensor<DT_X> xG, yG;
        xG.SetGlobalBuffer((__gm__ DT_X*)x + off, length);
        yG.SetGlobalBuffer((__gm__ DT_X*)y + off, length);
        DataCopy(xL, xG[0], NK);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::ALL>();
        Mins(xL, xL, (DT_X)CLAMP_HI, NK); Maxs(xL, xL, (DT_X)CLAMP_LO, NK);
        Mul(tL, xL, xL, NK);
        Muls(yL, tL, (DT_X)C11, NK); Adds(yL, yL, (DT_X)C9, NK);
        Mul(yL, tL, yL, NK); Adds(yL, yL, (DT_X)C7, NK);
        Mul(yL, tL, yL, NK); Adds(yL, yL, (DT_X)C5, NK);
        Mul(yL, tL, yL, NK); Adds(yL, yL, (DT_X)C3, NK);
        Mul(yL, tL, yL, NK); Adds(yL, yL, (DT_X)C1, NK);
        Mul(yL, xL, yL, NK);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::ALL>();
        DataCopy(yG[0], yL, NK);
        return;
    }
    if (length <= 128) {
        ErfTilingData td;
        td.totalLength = totalLength; td.coreDataLength = coreDataLength;
        td.coreNum = 8; td.tileLength = ((length + 7) / 8 + 1) * 8;
        KernelErf<DT_X> op; op.Init(x, y, &td); op.Process();
        return;
    }
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, tiling_data, tiling);
    KernelErf<DT_X> op; op.Init(x, y, &tiling_data); op.Process();
}
