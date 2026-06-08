// erf kernel — self-distributed per-core, runtime single-tile fast path
#include "kernel_operator.h"
#include "erf_tiling.h"
#include "tiling_key_erf.h"

using namespace AscendC;

constexpr uint32_t kAlignElems = 8;
constexpr uint32_t kQueueSlots = 2;

template<typename T>
__aicore__ inline void ApplyOddP4(LocalTensor<T> x, LocalTensor<T> y,
                                    LocalTensor<T> t, uint32_t cnt) {
    int32_t c = (int32_t)cnt;
    Maxs(x, x, (T)(-2.22f), c);
    Mins(x, x, (T)2.22f, c);
    Mul(t, x, x, c);
    Muls(y, t, (T)0.000761244780f, c);
    Adds(y, y, (T)(-0.0126843134f), c);
    Mul(y, t, y, c);
    Adds(y, y, (T)0.0889612101f, c);
    Mul(y, t, y, c);
    Adds(y, y, (T)(-0.358338483f), c);
    Mul(y, t, y, c);
    Adds(y, y, (T)1.12454948f, c);
    Mul(y, x, y, c);
}

template<typename T>
__aicore__ inline bool CanUsePlainCopy(uint64_t o, uint32_t n) {
    return ((o | (uint64_t)n) & (kAlignElems - 1U)) == 0ULL;
}

template<typename T>
__aicore__ inline void ReadChunk(LocalTensor<T> d, const GlobalTensor<T>& s,
                             uint64_t o, uint32_t n) {
    if (CanUsePlainCopy<T>(o, n)) {
        DataCopy(d, s[o], n);
    } else {
        DataCopyExtParams p{1, (uint32_t)(n * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pp{false, 0, 0, (T)0};
        DataCopyPad(d, s[o], p, pp);
    }
}

template<typename T>
__aicore__ inline void WriteChunk(const GlobalTensor<T>& d, uint64_t o,
                              LocalTensor<T> s, uint32_t n) {
    if (CanUsePlainCopy<T>(o, n)) {
        DataCopy(d[o], s, n);
    } else {
        DataCopyExtParams p{1, (uint32_t)(n * sizeof(T)), 0, 0, 0};
        DataCopyPad(d[o], s, p);
    }
}

template<typename T>
__global__ __aicore__ void erf(__gm__ uint8_t* x, __gm__ uint8_t* y,
                                __gm__ uint8_t* ws, __gm__ uint8_t* tl) {
    (void)ws;
    REGISTER_TILING_DEFAULT(ErfTilingData);
    GET_TILING_DATA_WITH_STRUCT(ErfTilingData, d, tl);

    uint64_t N = d.length;
    uint32_t me = GetBlockIdx();
    uint32_t nc = GetBlockNum();
    if (me >= nc || N == 0) return;

    // Distribute: aligned blocks evenly, tail to last core
    uint64_t aligned = (N / kAlignElems) * kAlignElems;
    uint32_t tail = (uint32_t)(N - aligned);
    uint64_t blk = aligned / kAlignElems;
    uint64_t bpk = blk / nc;
    uint64_t rem = blk % nc;

    uint64_t off, len;
    if (me < rem) {
        len = (bpk + 1) * kAlignElems;
        off = me * len;
    } else {
        len = bpk * kAlignElems;
        off = rem * (bpk + 1) * kAlignElems + (me - rem) * len;
    }
    if (tail && me == nc - 1) len += tail;
    if (len == 0) return;

    GlobalTensor<T> X, Y;
    X.SetGlobalBuffer((__gm__ T*)x, (int64_t)N);
    Y.SetGlobalBuffer((__gm__ T*)y, (int64_t)N);

    uint32_t tile = (len <= 1024) ? 1024 :
                    (len <= 4096) ? 4096 :
                    (len <= 8192) ? 8192 : 8192;

    if (len <= tile) {
        // Single tile
        TPipe p;
        TQue<QuePosition::VECIN, 1> qi;
        TQue<QuePosition::VECOUT, 1> qo;
        TBuf<QuePosition::VECCALC> qt;
        p.InitBuffer(qi, 1, len * sizeof(T));
        p.InitBuffer(qo, 1, len * sizeof(T));
        p.InitBuffer(qt, len * sizeof(T));

        auto a = qi.AllocTensor<T>();
        ReadChunk(a, X, off, (uint32_t)len); qi.EnQue(a);
        a = qi.DeQue<T>();
        auto b = qo.AllocTensor<T>();
        ApplyOddP4(a, b, qt.Get<T>(), (uint32_t)len);
        qo.EnQue(b); qi.FreeTensor(a);
        b = qo.DeQue<T>();
        WriteChunk(Y, off, b, (uint32_t)len); qo.FreeTensor(b);
        return;
    }

    // Pipelined double buffer
    TPipe p;
    TQue<QuePosition::VECIN, kQueueSlots> qi;
    TQue<QuePosition::VECOUT, kQueueSlots> qo;
    TBuf<QuePosition::VECCALC> qt;
    p.InitBuffer(qi, kQueueSlots, tile * sizeof(T));
    p.InitBuffer(qo, kQueueSlots, tile * sizeof(T));
    p.InitBuffer(qt, tile * sizeof(T));

    uint64_t po = off;
    uint64_t co = off + tile;
    uint32_t pl = (uint32_t)((len < tile) ? len : tile);
    uint64_t rm = len - pl;

    {
        auto a = qi.AllocTensor<T>();
        ReadChunk(a, X, po, pl); qi.EnQue(a);
    }

    while (rm > 0) {
        uint32_t cl = (uint32_t)((rm > tile) ? tile : rm);
        {
            auto xi = qi.DeQue<T>();
            auto yi = qo.AllocTensor<T>();
            ApplyOddP4(xi, yi, qt.Get<T>(), pl);
            qo.EnQue(yi); qi.FreeTensor(xi);
        }
        {
            auto xj = qi.AllocTensor<T>();
            ReadChunk(xj, X, co, cl); qi.EnQue(xj);
        }
        {
            auto yo = qo.DeQue<T>();
            WriteChunk(Y, po, yo, pl);
            qo.FreeTensor(yo);
        }
        po = co; pl = cl; co += cl; rm -= cl;
    }

    {
        auto xi = qi.DeQue<T>();
        auto yi = qo.AllocTensor<T>();
        ApplyOddP4(xi, yi, qt.Get<T>(), pl);
        qo.EnQue(yi); qi.FreeTensor(xi);
    }
    {
        auto yo = qo.DeQue<T>();
        WriteChunk(Y, po, yo, pl);
        qo.FreeTensor(yo);
    }
}
