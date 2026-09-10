#include "kernel_operator.h"

using namespace AscendC;

// ---------------------------------------------------------------------------
// Scale: output = input * scale[j] + bias[j]   (layout [outer, S, I])
//
//   fp16 : hardware Mul + Add in fp16 (bit-exact to torch's 2-step rounding)
//   fp32 : single FusedMulAdd in fp32 (1-step; within 1e-4)
//   bf16 : fp32 2-step rounding (no hw bf16 mul); round_bf16(round_bf16(x*s)+b)
//
// Vector-op counts rounded up to 64 (fp32 SIMD repeat) to avoid the non-64
// tail miscompute. DataCopyPad for arbitrary GM alignment. Multi-core / outer.
// ---------------------------------------------------------------------------

constexpr uint32_t SCALE_CACHE_MAX = 8192u;
constexpr uint32_t TILE_ELEMS      = 2048u;
constexpr uint32_t VREP            = 64u;
constexpr uint32_t EXP_TILE = 3840u;   // legacy ScaleCached per-j tile reference
constexpr uint32_t S_PERJ_MAX = 256u;  // S below which per-j vectorized run beats expansion
__aicore__ inline uint32_t RU64(uint32_t n) { return (n + VREP - 1u) / VREP * VREP; }

template <typename T>
__aicore__ inline float SToF(T v) {
    if constexpr (Std::is_same<T, bfloat16_t>::value) return ToFloat(v);
    else return static_cast<float>(v);
}
template <typename T>
__aicore__ inline T FToS(float v) {
    if constexpr (Std::is_same<T, bfloat16_t>::value) return ToBfloat16(v);
    else return static_cast<T>(v);
}

template <typename T>
__aicore__ inline void ScaleCached(GM_ADDR input, GM_ADDR scale_gm, GM_ADDR bias_gm,
                                    GM_ADDR output, uint32_t outer, uint32_t S,
                                    uint32_t I, uint32_t has_bias)
{
    constexpr uint32_t BLK = 32u / static_cast<uint32_t>(sizeof(T));
    constexpr bool isF32 = Std::is_same<T, float>::value;
    constexpr bool isF16 = Std::is_same<T, half>::value;

    uint32_t S_load = (S + BLK - 1u) / BLK * BLK;
    uint32_t S_pad  = RU64(S);
    if (S_load > S_pad) S_pad = S_load;
    uint64_t total = static_cast<uint64_t>(outer) * S * I;

    GlobalTensor<T> inGm, outGm, scaleGm, biasGm;
    inGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input), total);
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output), total);
    // input is read once and output written once: caching them in L2 only causes eviction
    // write-backs that steal HBM bandwidth, so bypass L2 for these two streams. scale/bias are
    // reused per p-tile across batches, so they stay cache-NORMAL.
    inGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    outGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale_gm), S_load);
    if (has_bias) biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias_gm), S_load);

    constexpr bool isBf16 = Std::is_same<T, bfloat16_t>::value;
    TPipe pipe;
    TBuf<TPosition::VECCALC> inMem, outMem, sNatMem, bNatMem, sFMem, bFMem, xfMem, rtMem, zMem, oMem;
    pipe.InitBuffer(inMem,   TILE_ELEMS * sizeof(T));
    pipe.InitBuffer(outMem,  TILE_ELEMS * sizeof(T));
    pipe.InitBuffer(sNatMem, S_pad * sizeof(T));
    pipe.InitBuffer(bNatMem, S_pad * sizeof(T));
    pipe.InitBuffer(sFMem,   S_pad * sizeof(float));
    pipe.InitBuffer(bFMem,   S_pad * sizeof(float));
    pipe.InitBuffer(xfMem,   TILE_ELEMS * sizeof(float));
    pipe.InitBuffer(rtMem,   TILE_ELEMS * sizeof(T));
    pipe.InitBuffer(zMem,    TILE_ELEMS * sizeof(float));
    pipe.InitBuffer(oMem,    TILE_ELEMS * sizeof(float));

    LocalTensor<T>     inBuf  = inMem.Get<T>();
    LocalTensor<T>     outBuf = outMem.Get<T>();
    LocalTensor<T>     scaleN = sNatMem.Get<T>();
    LocalTensor<T>     biasN  = bNatMem.Get<T>();
    LocalTensor<float> scaleF = sFMem.Get<float>();
    LocalTensor<float> biasF  = bFMem.Get<float>();
    LocalTensor<float> xf     = xfMem.Get<float>();
    LocalTensor<T>     rt     = rtMem.Get<T>();
    LocalTensor<float> zeroB  = zMem.Get<float>();
    LocalTensor<float> oneB   = oMem.Get<float>();

    // Init padding: scale tail = 1, bias tail = 0
    Duplicate(scaleN, static_cast<T>(1), S_pad);
    Duplicate(biasN, static_cast<T>(0), S_pad);
    if constexpr (isBf16) { Duplicate(zeroB, 0.0f, TILE_ELEMS); Duplicate(oneB, 1.0f, TILE_ELEMS); }
    pipe_barrier(PIPE_V);

    DataCopy(scaleN, scaleGm[0], S_load);
    if (has_bias) DataCopy(biasN, biasGm[0], S_load);
    SetFlag<HardEvent::MTE2_V>(static_cast<TEventID>(2));
    WaitFlag<HardEvent::MTE2_V>(static_cast<TEventID>(2));

    if constexpr (!isF16) {
        // fp32 / bf16 need fp32 copies of scale/bias
        if constexpr (isF32) { Adds(scaleF, scaleN, 0.0f, S_pad); Adds(biasF, biasN, 0.0f, S_pad); }
        else { Cast(scaleF, scaleN, RoundMode::CAST_NONE, S_pad); Cast(biasF, biasN, RoundMode::CAST_NONE, S_pad); }
    }
    pipe_barrier(PIPE_V);

    uint32_t bIdx = GetBlockIdx();
    uint32_t bNum = GetBlockNum();
    DataCopyPadExtParams<T> padp{false, 0, 0, 0};
    bool first = true;

    #define STORE(goff_, t_)                                                       \
        do {                                                                       \
            SetFlag<HardEvent::V_MTE3>(static_cast<TEventID>(0));                  \
            WaitFlag<HardEvent::V_MTE3>(static_cast<TEventID>(0));                 \
            DataCopyExtParams cpO{1, (t_) * static_cast<uint32_t>(sizeof(T)), 0, 0, 0}; \
            DataCopyPad(outGm[(goff_)], outBuf, cpO);                              \
            SetFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));               \
            first = false;                                                         \
        } while (0)

    if (I == 1u) {
        for (uint32_t b = bIdx; b < outer; b += bNum) {
            uint32_t base = b * S;
            uint32_t s = 0u;
            while (s < S) {
                uint32_t t = ((S - s) < TILE_ELEMS) ? (S - s) : TILE_ELEMS;
                uint32_t t64 = RU64(t);
                if (!first) WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
                DataCopyExtParams cpI{1, t * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                DataCopyPad(inBuf, inGm[base + s], cpI, padp);
                SetFlag<HardEvent::MTE2_V>(static_cast<TEventID>(0));
                WaitFlag<HardEvent::MTE2_V>(static_cast<TEventID>(0));

                if constexpr (isF16) {
                    Mul(outBuf, inBuf, scaleN[s], t64);
                    if (has_bias) Add(outBuf, outBuf, biasN[s], t64);
                } else if constexpr (isF32) {
                    // inBuf = scaleF*inBuf + biasF (single fused; fp32 fine at 1e-4)
                    FusedMulAdd(inBuf, scaleF[s], biasF[s], static_cast<int32_t>(t64));
                    Adds(outBuf, inBuf, 0.0f, t64);
                } else {  // bf16 — 2-step rounding to match torch
                    Cast(xf, inBuf, RoundMode::CAST_NONE, t64);
                    Mul(xf, xf, scaleF[s], t64);                                  // xf = scale*x
                    Cast(rt, xf, RoundMode::CAST_RINT, t64); Cast(xf, rt, RoundMode::CAST_NONE, t64);
                    if (has_bias) Add(xf, xf, biasF[s], t64);                     // + bias
                    Cast(outBuf, xf, RoundMode::CAST_RINT, t64);
                }
                STORE(base + s, t);
                s += t;
            }
        }
    } else {
        for (uint32_t b = bIdx; b < outer; b += bNum) {
            uint32_t base = b * S * I;
            for (uint32_t j = 0u; j < S; j++) {
                uint32_t jb = base + j * I;
                uint32_t k = 0u;
                while (k < I) {
                    uint32_t t = ((I - k) < TILE_ELEMS) ? (I - k) : TILE_ELEMS;
                    uint32_t t64 = RU64(t);
                    if (!first) WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
                    DataCopyExtParams cpI{1, t * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                    DataCopyPad(inBuf, inGm[jb + k], cpI, padp);
                    SetFlag<HardEvent::MTE2_V>(static_cast<TEventID>(0));
                    WaitFlag<HardEvent::MTE2_V>(static_cast<TEventID>(0));

                    if constexpr (isF16) {
                        half sv = scaleN.GetValue(j);
                        Muls(outBuf, inBuf, sv, t64);
                        if (has_bias) { half bv = biasN.GetValue(j); Adds(outBuf, outBuf, bv, t64); }
                    } else if constexpr (isF32) {
                        float sv = scaleF.GetValue(j); float bv = biasF.GetValue(j);
                        Muls(inBuf, inBuf, sv, t64);
                        Adds(outBuf, inBuf, bv, t64);
                    } else {  // bf16 — 2-step rounding
                        float sv = scaleF.GetValue(j); float bv = biasF.GetValue(j);
                        Cast(xf, inBuf, RoundMode::CAST_NONE, t64);
                        Muls(xf, xf, sv, t64);
                        Cast(rt, xf, RoundMode::CAST_RINT, t64); Cast(xf, rt, RoundMode::CAST_NONE, t64);
                        if (has_bias) Adds(xf, xf, bv, t64);
                        Cast(outBuf, xf, RoundMode::CAST_RINT, t64);
                    }
                    STORE(jb + k, t);
                    k += t;
                }
            }
        }
    }
    if (!first) WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
    #undef STORE
}

// ===========================================================================
// BuildExpandCoef: expand scale/bias for p-tile [pt*ET, pt*ET+t) into per-element
// coefficient vectors (esF/ebF fp32 for fp32/bf16; esH/ebH for fp16). Built ONCE
// per (core, p-tile) and reused across all batches. Returns t (valid elems) and
// t64 (RU64 padded count) by reference.
// ===========================================================================
template <typename T>
__aicore__ inline void BuildExpandCoef(uint32_t pt, uint32_t P, uint32_t I, uint32_t ET,
                                       uint32_t has_bias,
                                       GlobalTensor<T>& scaleGm, GlobalTensor<T>& biasGm,
                                       LocalTensor<T>& ssBuf, LocalTensor<T>& sbBuf,
                                       LocalTensor<T>& esH, LocalTensor<T>& ebH,
                                       LocalTensor<float>& esF, LocalTensor<float>& ebF,
                                       uint32_t& t, uint32_t& t64)
{
    constexpr bool isF16 = Std::is_same<T, half>::value;
    constexpr bool isF32 = Std::is_same<T, float>::value;
    constexpr uint32_t CALN = isF16 ? 16u : 8u;   // 32B in coef elements (fp16 coef=2B, else 4B)
    DataCopyPadExtParams<T> padp{false, 0, 0, 0};

    uint32_t ps = pt * ET;
    t = ((P - ps) < ET) ? (P - ps) : ET;
    t64 = RU64(t);
    uint32_t j0 = ps / I, j1 = (ps + t - 1u) / I;
    uint32_t jc = j1 - j0 + 1u;
    DataCopyExtParams cpS{1, jc * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPad(ssBuf, scaleGm[j0], cpS, padp);
    if (has_bias) DataCopyPad(sbBuf, biasGm[j0], cpS, padp);
    SetFlag<HardEvent::MTE2_V>(static_cast<TEventID>(1));
    WaitFlag<HardEvent::MTE2_V>(static_cast<TEventID>(1));

    if (I == 1u) {
        // I==1: coef IS the scale slice (j-range == [ps,ps+t)). Vectorized copy/cast.
        if constexpr (isF16) {
            Adds(esH, ssBuf, static_cast<half>(0), t64);
            if (has_bias) Adds(ebH, sbBuf, static_cast<half>(0), t64);
        } else if constexpr (isF32) {
            Adds(esF, ssBuf, 0.0f, t64);
            if (has_bias) Adds(ebF, sbBuf, 0.0f, t64);
        } else { // bf16 -> fp32 coef
            Cast(esF, ssBuf, RoundMode::CAST_NONE, t64);
            if (has_bias) Cast(ebF, sbBuf, RoundMode::CAST_NONE, t64);
        }
    } else {
        // I>1: fill each j-segment of the coef with a vector Duplicate (off the scalar pipe).
        // Duplicate needs a 32B-aligned start, so SetValue only the (<CALN) misaligned leading
        // elements of each segment, then Duplicate the aligned bulk.
        for (uint32_t j = j0; j <= j1; j++) {
            uint32_t pj0 = j * I;        if (pj0 < ps) pj0 = ps;
            uint32_t pj1 = (j + 1u) * I; if (pj1 > ps + t) pj1 = ps + t;
            uint32_t so = pj0 - ps;                    // segment offset within tile
            uint32_t sl = pj1 - pj0;                   // segment length
            uint32_t a  = so % CALN;
            uint32_t head = (a == 0u) ? 0u : (CALN - a);
            if (head > sl) head = sl;
            if constexpr (isF16) {
                half sh = ssBuf.GetValue(j - j0);
                half bh = has_bias ? sbBuf.GetValue(j - j0) : static_cast<half>(0);
                for (uint32_t k = 0u; k < head; k++) { esH.SetValue(so + k, sh); if (has_bias) ebH.SetValue(so + k, bh); }
                if (sl > head) { Duplicate(esH[so + head], sh, sl - head); if (has_bias) Duplicate(ebH[so + head], bh, sl - head); }
            } else {
                float sv = SToF<T>(ssBuf.GetValue(j - j0));
                float bv = has_bias ? SToF<T>(sbBuf.GetValue(j - j0)) : 0.0f;
                for (uint32_t k = 0u; k < head; k++) { esF.SetValue(so + k, sv); if (has_bias) ebF.SetValue(so + k, bv); }
                if (sl > head) { Duplicate(esF[so + head], sv, sl - head); if (has_bias) Duplicate(ebF[so + head], bv, sl - head); }
            }
        }
        // pad coef tail [t, t64): scale=1, bias=0 (<=63 elems)
        for (uint32_t p = t; p < t64; p++) {
            if constexpr (isF16) { esH.SetValue(p, static_cast<half>(1)); if (has_bias) ebH.SetValue(p, static_cast<half>(0)); }
            else { esF.SetValue(p, 1.0f); if (has_bias) ebF.SetValue(p, 0.0f); }
        }
    }
    pipe_barrier(PIPE_V);
}

// ===========================================================================
// ScaleExpand (inner>1, or inner==1 large S): expand scale/bias into per-batch
// coefficient vectors so each batch becomes a plain vectorized in*coef_scale +
// coef_bias over big tiles. Coef built ONCE per (core, p-tile) and reused across
// all batches. 2D core distribution over (p-tile × batch-group).
//
// bf16: the 2-step rounding is an irreducible 6-op dependent vector chain
//   (Cast NONE; Mul; Cast RINT; Cast NONE; Add; Cast RINT) — no fused intrinsic
//   on dav_c220 accepts a bf16 dst. Profiling shows this chain is LATENCY-bound
//   (aiv_vec≈0.51, aiv_mte2≈0.50, neither saturated), so we run TWO independent
//   tiles (A,B) interleaved through the chain to hide dependent-op latency, with
//   TQue depth 4 to also overlap the next pair's MTE2 loads under the VEC chain.
// ===========================================================================
template <typename T>
__aicore__ inline void ScaleExpand(GM_ADDR input, GM_ADDR scale_gm, GM_ADDR bias_gm,
                                    GM_ADDR output, uint32_t outer, uint32_t S,
                                    uint32_t I, uint32_t has_bias)
{
    constexpr bool isF16 = Std::is_same<T, half>::value;
    constexpr bool isF32 = Std::is_same<T, float>::value;
    constexpr bool isBf16 = Std::is_same<T, bfloat16_t>::value;

    // Per-dtype EXPAND tile. bf16 uses two-stream scratch + depth-4 queues, so it
    // carries the most buffers/elem and gets the smaller tile.
    //   fp16: 6 bufs (depth2)  -> 10240 ; fp32: depth2 -> 5120 ; bf16: depth4 two-stream -> 4608.
    constexpr uint32_t ET = isF16 ? 10240u : (isF32 ? 5120u : 4608u);
    constexpr uint32_t BLK = 32u / static_cast<uint32_t>(sizeof(T));  // 32B block in elements
    constexpr int32_t  QD = isBf16 ? 4 : 2;                           // queue depth

    uint32_t P = S * I;                       // per-batch element count
    uint64_t total = static_cast<uint64_t>(outer) * P;
    GlobalTensor<T> inGm, outGm, scaleGm, biasGm;
    inGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input), total);
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output), total);
    // input read once / output written once: bypass L2 (avoid eviction write-backs stealing HBM BW).
    inGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    outGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale_gm), S);
    if (has_bias) biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias_gm), S);

    TPipe pipe;
    TQue<TPosition::VECIN, QD>  inQ;
    TQue<TPosition::VECOUT, QD> outQ;
    TBuf<TPosition::VECCALC> ssMem, sbMem, esMem, ebMem, xaMem, raMem, xbMem, rbMem;
    pipe.InitBuffer(inQ, QD, ET * sizeof(T));
    pipe.InitBuffer(outQ, QD, ET * sizeof(T));
    pipe.InitBuffer(ssMem, ET * sizeof(T));     // scale slice (raw)
    pipe.InitBuffer(sbMem, ET * sizeof(T));     // bias slice (raw)
    pipe.InitBuffer(esMem, ET * (isF16 ? sizeof(T) : sizeof(float)));  // coef scale
    pipe.InitBuffer(ebMem, ET * (isF16 ? sizeof(T) : sizeof(float)));  // coef bias
    // bf16 two-stream scratch: xfA/rtA (stream A), xfB/rtB (stream B). 32B stubs otherwise.
    pipe.InitBuffer(xaMem, isBf16 ? (ET * sizeof(float)) : 32u);
    pipe.InitBuffer(raMem, isBf16 ? (ET * sizeof(T))     : 32u);
    pipe.InitBuffer(xbMem, isBf16 ? (ET * sizeof(float)) : 32u);
    pipe.InitBuffer(rbMem, isBf16 ? (ET * sizeof(T))     : 32u);

    LocalTensor<T> ssBuf = ssMem.Get<T>(), sbBuf = sbMem.Get<T>();
    LocalTensor<T> esH = esMem.Get<T>(), ebH = ebMem.Get<T>();          // fp16 coef view
    LocalTensor<float> esF = esMem.Get<float>(), ebF = ebMem.Get<float>(); // fp32 coef view
    LocalTensor<float> xfA = xaMem.Get<float>(), xfB = xbMem.Get<float>();
    LocalTensor<T> rtA = raMem.Get<T>(), rtB = rbMem.Get<T>();

    uint32_t nPT = (P + ET - 1u) / ET;
    uint32_t bIdx = GetBlockIdx(), bNum = GetBlockNum();
    DataCopyPadExtParams<T> padp{false,0,0,0};

    // Balanced flatten: spread the nPT*outer (p-tile, batch) work units evenly across cores as
    // contiguous pt-major slices, so each core rebuilds the coef at most once per contiguous p-tile run.
    uint64_t W = static_cast<uint64_t>(nPT) * outer;
    uint64_t qbase = W / bNum, qrem = W % bNum;
    uint64_t u0 = static_cast<uint64_t>(bIdx) * qbase + ((bIdx < qrem) ? bIdx : qrem);
    uint64_t cnt = qbase + ((bIdx < qrem) ? 1u : 0u);
    uint64_t u1 = u0 + cnt;
    if (u0 >= W) return;                        // only when W < bNum (tiny shapes)

    uint32_t curPt = 0xFFFFFFFFu;
    uint32_t t = 0u, t64 = 0u;

    if constexpr (isBf16) {
        // ---- bf16: two-stream interleaved 6-op chain, depth-4 queues ----
        uint64_t u = u0;
        while (u < u1) {
            uint32_t pt = static_cast<uint32_t>(u / outer);
            if (pt != curPt) {
                curPt = pt;
                BuildExpandCoef<T>(pt, P, I, ET, has_bias, scaleGm, biasGm,
                                   ssBuf, sbBuf, esH, ebH, esF, ebF, t, t64);
            }
            uint32_t ps = pt * ET;
            uint64_t runEnd = static_cast<uint64_t>(pt + 1) * outer;
            if (runEnd > u1) runEnd = u1;

            // process pairs of batches (both share this p-tile's coef)
            while (u + 1 < runEnd) {
                uint32_t bA = static_cast<uint32_t>(u % outer);
                uint32_t bB = static_cast<uint32_t>((u + 1) % outer);
                uint64_t goffA = static_cast<uint64_t>(bA) * P + ps;
                uint64_t goffB = static_cast<uint64_t>(bB) * P + ps;

                LocalTensor<T> xA = inQ.template AllocTensor<T>();
                if ((goffA % BLK == 0u) && (t % BLK == 0u)) { DataCopy(xA, inGm[goffA], t); }
                else { DataCopyExtParams cA{1, t*static_cast<uint32_t>(sizeof(T)),0,0,0}; DataCopyPad(xA, inGm[goffA], cA, padp); }
                inQ.EnQue(xA);
                LocalTensor<T> xB = inQ.template AllocTensor<T>();
                if ((goffB % BLK == 0u) && (t % BLK == 0u)) { DataCopy(xB, inGm[goffB], t); }
                else { DataCopyExtParams cB{1, t*static_cast<uint32_t>(sizeof(T)),0,0,0}; DataCopyPad(xB, inGm[goffB], cB, padp); }
                inQ.EnQue(xB);

                LocalTensor<T> xdA = inQ.template DeQue<T>();
                LocalTensor<T> xdB = inQ.template DeQue<T>();
                LocalTensor<T> yA = outQ.template AllocTensor<T>();
                LocalTensor<T> yB = outQ.template AllocTensor<T>();

                // interleaved two-step rounding (A and B independent -> hides latency)
                Cast(xfA, xdA, RoundMode::CAST_NONE, t64);  Cast(xfB, xdB, RoundMode::CAST_NONE, t64);
                Mul(xfA, xfA, esF, t64);                    Mul(xfB, xfB, esF, t64);
                Cast(rtA, xfA, RoundMode::CAST_RINT, t64);  Cast(rtB, xfB, RoundMode::CAST_RINT, t64);
                Cast(xfA, rtA, RoundMode::CAST_NONE, t64);  Cast(xfB, rtB, RoundMode::CAST_NONE, t64);
                if (has_bias) { Add(xfA, xfA, ebF, t64);    Add(xfB, xfB, ebF, t64); }
                Cast(yA, xfA, RoundMode::CAST_RINT, t64);   Cast(yB, xfB, RoundMode::CAST_RINT, t64);

                outQ.template EnQue<T>(yA);
                outQ.template EnQue<T>(yB);
                inQ.FreeTensor(xdA);
                inQ.FreeTensor(xdB);

                LocalTensor<T> ydA = outQ.template DeQue<T>();
                LocalTensor<T> ydB = outQ.template DeQue<T>();
                if ((goffA % BLK == 0u) && (t % BLK == 0u)) { DataCopy(outGm[goffA], ydA, t); }
                else { DataCopyExtParams cA{1, t*static_cast<uint32_t>(sizeof(T)),0,0,0}; DataCopyPad(outGm[goffA], ydA, cA); }
                if ((goffB % BLK == 0u) && (t % BLK == 0u)) { DataCopy(outGm[goffB], ydB, t); }
                else { DataCopyExtParams cB{1, t*static_cast<uint32_t>(sizeof(T)),0,0,0}; DataCopyPad(outGm[goffB], ydB, cB); }
                outQ.FreeTensor(ydA);
                outQ.FreeTensor(ydB);
                u += 2;
            }
            // odd leftover batch in this p-tile run -> single-stream
            if (u < runEnd) {
                uint32_t b = static_cast<uint32_t>(u % outer);
                uint64_t goff = static_cast<uint64_t>(b) * P + ps;
                LocalTensor<T> x = inQ.template AllocTensor<T>();
                if ((goff % BLK == 0u) && (t % BLK == 0u)) { DataCopy(x, inGm[goff], t); }
                else { DataCopyExtParams c{1, t*static_cast<uint32_t>(sizeof(T)),0,0,0}; DataCopyPad(x, inGm[goff], c, padp); }
                inQ.EnQue(x);
                LocalTensor<T> xd = inQ.template DeQue<T>();
                LocalTensor<T> y = outQ.template AllocTensor<T>();
                Cast(xfA, xd, RoundMode::CAST_NONE, t64);
                Mul(xfA, xfA, esF, t64);
                Cast(rtA, xfA, RoundMode::CAST_RINT, t64);
                Cast(xfA, rtA, RoundMode::CAST_NONE, t64);
                if (has_bias) Add(xfA, xfA, ebF, t64);
                Cast(y, xfA, RoundMode::CAST_RINT, t64);
                outQ.template EnQue<T>(y);
                inQ.FreeTensor(xd);
                LocalTensor<T> yd = outQ.template DeQue<T>();
                if ((goff % BLK == 0u) && (t % BLK == 0u)) { DataCopy(outGm[goff], yd, t); }
                else { DataCopyExtParams c{1, t*static_cast<uint32_t>(sizeof(T)),0,0,0}; DataCopyPad(outGm[goff], yd, c); }
                outQ.FreeTensor(yd);
                u += 1;
            }
        }
    } else {
        // ---- fp16 / fp32: single-stream (one fused/Mul pass; not latency-bound) ----
        for (uint64_t u = u0; u < u1; u++) {
            uint32_t pt = static_cast<uint32_t>(u / outer);
            uint32_t b  = static_cast<uint32_t>(u % outer);
            uint32_t ps = pt * ET;
            if (pt != curPt) {
                curPt = pt;
                BuildExpandCoef<T>(pt, P, I, ET, has_bias, scaleGm, biasGm,
                                   ssBuf, sbBuf, esH, ebH, esF, ebF, t, t64);
            }
            uint64_t goff = static_cast<uint64_t>(b) * P + ps;
            LocalTensor<T> x = inQ.template AllocTensor<T>();
            if ((goff % BLK == 0u) && (t % BLK == 0u)) { DataCopy(x, inGm[goff], t); }
            else { DataCopyExtParams cpI{1, t*static_cast<uint32_t>(sizeof(T)),0,0,0}; DataCopyPad(x, inGm[goff], cpI, padp); }
            inQ.EnQue(x);
            LocalTensor<T> xd = inQ.template DeQue<T>();
            LocalTensor<T> y = outQ.template AllocTensor<T>();

            if constexpr (isF16) {
                Mul(y, xd, esH, t64);
                if (has_bias) Add(y, y, ebH, t64);
            } else { // fp32
                if (has_bias) {
                    FusedMulAdd(xd, esF, ebF, static_cast<int32_t>(t64));  // xd = esF*xd + ebF
                    Adds(y, xd, 0.0f, t64);
                } else {
                    Mul(y, xd, esF, t64);
                }
            }
            outQ.template EnQue<T>(y);
            inQ.FreeTensor(xd);

            LocalTensor<T> yd = outQ.template DeQue<T>();
            if ((goff % BLK == 0u) && (t % BLK == 0u)) { DataCopy(outGm[goff], yd, t); }
            else { DataCopyExtParams cpO{1, t*static_cast<uint32_t>(sizeof(T)),0,0,0}; DataCopyPad(outGm[goff], yd, cpO); }
            outQ.FreeTensor(yd);
        }
    }
}

template <typename T>
__aicore__ inline void ScaleProcess(GM_ADDR input, GM_ADDR scale_gm, GM_ADDR bias_gm,
                                     GM_ADDR output, uint32_t outer, uint32_t S,
                                     uint32_t I, uint32_t has_bias)
{
    if (outer == 0u || S == 0u || I == 0u) return;
    // Route by S = number of distinct coefficients (NOT inner):
    //   - inner==1, small S: cached/per-element. large S: expand.
    //   - inner>1, small S: per-j vectorized run (ScaleCached). large S: expand.
    if (I == 1u) {
        if (S <= SCALE_CACHE_MAX) ScaleCached<T>(input, scale_gm, bias_gm, output, outer, S, I, has_bias);
        else                      ScaleExpand<T>(input, scale_gm, bias_gm, output, outer, S, I, has_bias);
    } else if (S <= S_PERJ_MAX) {
        ScaleCached<T>(input, scale_gm, bias_gm, output, outer, S, I, has_bias);
    } else {
        ScaleExpand<T>(input, scale_gm, bias_gm, output, outer, S, I, has_bias);
    }
}

extern "C" __global__ __aicore__ void scale(GM_ADDR input, GM_ADDR scale_gm,
                                             GM_ADDR bias_gm, GM_ADDR output,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    switch (tiling_data.dtype) {
        case 0:
            ScaleProcess<float>(input, scale_gm, bias_gm, output, tiling_data.outer_size,
                                tiling_data.scale_size, tiling_data.inner_size, tiling_data.has_bias);
            break;
        case 1:
            ScaleProcess<half>(input, scale_gm, bias_gm, output, tiling_data.outer_size,
                               tiling_data.scale_size, tiling_data.inner_size, tiling_data.has_bias);
            break;
        case 27:
            ScaleProcess<bfloat16_t>(input, scale_gm, bias_gm, output, tiling_data.outer_size,
                                     tiling_data.scale_size, tiling_data.inner_size, tiling_data.has_bias);
            break;
        default:
            ScaleProcess<float>(input, scale_gm, bias_gm, output, tiling_data.outer_size,
                                tiling_data.scale_size, tiling_data.inner_size, tiling_data.has_bias);
            break;
    }
}
