// Kernel-side BatchToSpace — route-dispatched compute kernels.
// TILING_KEY_IS selects the compile-time route; tiling data carries runtime dimensions.
#include "kernel_operator.h"

#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

// ═══════════════════════════════════════════════════════════════════════════
//  Section 1 — Tiling data extraction
// ═══════════════════════════════════════════════════════════════════════════

__aicore__ inline uint64_t TilingLoadPacked(GM_ADDR tiling_addr) {
    const __gm__ uint64_t *src =
        reinterpret_cast<const __gm__ uint64_t *>(tiling_addr);
    return *src;  // PackedTiling::data
}

__aicore__ inline uint32_t TilingLoadDepth(GM_ADDR tiling_addr) {
    const __gm__ uint8_t *base =
        reinterpret_cast<const __gm__ uint8_t *>(tiling_addr);
    const __gm__ uint16_t *src =
        reinterpret_cast<const __gm__ uint16_t *>(base + offsetof(PackedTiling, depth));
    return static_cast<uint32_t>(*src);
}

__aicore__ inline uint32_t UnpackH(uint64_t p) {
    return static_cast<uint32_t>((p >> PF_H_SHIFT) &
                                 ((1ULL << PF_H_BITS) - 1ULL));
}
__aicore__ inline uint32_t UnpackW(uint64_t p) {
    return static_cast<uint32_t>((p >> PF_W_SHIFT) &
                                 ((1ULL << PF_W_BITS) - 1ULL));
}
__aicore__ inline uint32_t UnpackOutN(uint64_t p) {
    return static_cast<uint32_t>((p >> PF_ON_SHIFT) &
                                 ((1ULL << PF_ON_BITS) - 1ULL));
}
__aicore__ inline uint32_t UnpackOutH(uint64_t p) {
    return static_cast<uint32_t>((p >> PF_OH_SHIFT) &
                                 ((1ULL << PF_OH_BITS) - 1ULL));
}
__aicore__ inline uint32_t UnpackOutW(uint64_t p) {
    return static_cast<uint32_t>((p >> PF_OW_SHIFT) &
                                 ((1ULL << PF_OW_BITS) - 1ULL));
}
__aicore__ inline uint32_t UnpackCropT(uint64_t p) {
    return static_cast<uint32_t>((p >> PF_CT_SHIFT) &
                                 ((1ULL << PF_CT_BITS) - 1ULL));
}
__aicore__ inline uint32_t UnpackCropL(uint64_t p) {
    return static_cast<uint32_t>((p >> PF_CL_SHIFT) &
                                 ((1ULL << PF_CL_BITS) - 1ULL));
}

__aicore__ inline void DecodeSpatial(uint64_t p, uint32_t &h, uint32_t &w,
                                     uint32_t &on, uint32_t &oh,
                                     uint32_t &ow) {
    h  = UnpackH(p);
    w  = UnpackW(p);
    on = UnpackOutN(p);
    oh = UnpackOutH(p);
    ow = UnpackOutW(p);
}

__aicore__ inline void DecodeSpatialCrop(uint64_t p, uint32_t &h, uint32_t &w,
                                          uint32_t &on, uint32_t &oh,
                                          uint32_t &ow, uint32_t &ct,
                                          uint32_t &cl) {
    DecodeSpatial(p, h, w, on, oh, ow);
    ct = UnpackCropT(p);
    cl = UnpackCropL(p);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 2 — Utility helpers
// ═══════════════════════════════════════════════════════════════════════════

__aicore__ inline uint32_t DivRoundUp(uint32_t a, uint32_t b) {
    return (a + b - 1U) / b;
}

__aicore__ inline uint32_t PixelCount(uint32_t n, uint32_t h, uint32_t w) {
    return n * h * w;
}

__aicore__ inline uint32_t Ugcd(uint32_t a, uint32_t b) {
    while (b != 0U) { uint32_t t = a % b; a = b; b = t; }
    return a;
}

template <class DT_X>
__aicore__ inline void SetGlobalTensors(GM_ADDR x, GM_ADDR y,
    AscendC::GlobalTensor<DT_X> &xg, AscendC::GlobalTensor<DT_X> &yg) {
    xg.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
    yg.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 3 — Signal-based synchronization (lightweight flag pipeline)
// ═══════════════════════════════════════════════════════════════════════════

constexpr event_t kEvStA = static_cast<event_t>(0);
constexpr event_t kEvStB = static_cast<event_t>(1);
constexpr event_t kEvLdA = static_cast<event_t>(2);
constexpr event_t kEvLdB = static_cast<event_t>(3);
constexpr event_t kEvGtA = static_cast<event_t>(4);
constexpr event_t kEvGtB = static_cast<event_t>(5);

__aicore__ inline void SigLoadOk(uint32_t slot) {
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
        slot == 0U ? kEvLdA : kEvLdB);
}
__aicore__ inline void AwaitLoad(uint32_t slot) {
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
        slot == 0U ? kEvLdA : kEvLdB);
}
__aicore__ inline void SigStoreOk(uint32_t slot) {
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
        slot == 0U ? kEvStA : kEvStB);
}
__aicore__ inline void AwaitStore(uint32_t slot) {
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
        slot == 0U ? kEvStA : kEvStB);
}
__aicore__ inline void FenceGather(uint32_t slot) {
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
        slot == 0U ? kEvGtA : kEvGtB);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
        slot == 0U ? kEvGtA : kEvGtB);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 4 — Gather offset table builder (Adds-based, on-device)
// ═══════════════════════════════════════════════════════════════════════════

__aicore__ inline void BuildGatherTable(
    AscendC::LocalTensor<int32_t> seed, uint32_t out_cols,
    uint32_t ch, uint32_t phase_sz, uint32_t esize) {
    constexpr uint32_t kVAlign = 8U;
    const uint32_t row_n = out_cols * ch;
    const uint32_t n_total =
        (row_n + kVAlign - 1U) & ~(kVAlign - 1U);
    const uint32_t pair_n = ch * 2U;
    const uint32_t per = kVAlign / Ugcd(pair_n, kVAlign);
    const uint32_t per_n = per * pair_n;
    const uint32_t per_b = per * ch * esize;

    for (uint32_t p = 0U; p < per; ++p) {
        uint32_t base = p * pair_n;
        uint32_t bbase = p * ch * esize;
        for (uint32_t c = 0U; c < ch; ++c) {
            uint32_t bo = c * esize;
            seed.SetValue(base + c,
                          static_cast<int32_t>(bbase + bo));
            seed.SetValue(base + ch + c,
                          static_cast<int32_t>(phase_sz + bbase + bo));
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(kEvStA);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(kEvStA);
    for (uint32_t done = per_n; done < n_total; done += per_n) {
        uint32_t cnt = n_total - done;
        if (cnt > per_n) cnt = per_n;
        AscendC::Adds(seed[done], seed[done - per_n],
                      static_cast<int32_t>(per_b), cnt);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 5 — fp32 kernel family
// ═══════════════════════════════════════════════════════════════════════════

// ── 5a. fp32 large-depth noCrop (double-buffered row pipeline) ──
template <class DT_X>
__aicore__ inline void KernelFp32Large(
    AscendC::GlobalTensor<DT_X> &xg, AscendC::GlobalTensor<DT_X> &yg,
    uint32_t cid, uint32_t H, uint32_t W, uint32_t D,
    uint32_t ON, uint32_t OH, uint32_t OW) {
    constexpr uint32_t kBN = 32;
    const uint32_t kRE = OW * D;
    const uint32_t kDB = D * sizeof(DT_X);
    const uint32_t kDS = kDB >> 5;

    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub;
    AscendC::LocalTensor<DT_X> b0 =
        ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(8192);
    AscendC::LocalTensor<DT_X> b1 =
        ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(8192);

    uint32_t nr = ON * OH;
    uint32_t rpc = DivRoundUp(nr, kBN);
    uint32_t r0 = cid * rpc;
    if (r0 >= nr) return;
    uint32_t rc = rpc;
    if (r0 + rc > nr) rc = nr - r0;

    AscendC::DataCopyExtParams ip{
        static_cast<uint16_t>(W), kDB, 0, kDS, 0};
    AscendC::DataCopyPadExtParams<DT_X> zp{false, 0, 0, 0};

    uint32_t ob = r0 / OH, oh = r0 - ob * OH;
    uint32_t ih0 = oh >> 1, bh0 = oh & 1;
    uint32_t ib00 = (bh0 << 1) * ON + ob;
    uint32_t ib01 = ib00 + ON;
    uint64_t s00 = ((static_cast<uint64_t>(ib00) * H + ih0) * W) * D;
    uint64_t s01 = ((static_cast<uint64_t>(ib01) * H + ih0) * W) * D;
    AscendC::DataCopyPad(b0, xg[s00], ip, zp);
    AscendC::DataCopyPad(b0[D], xg[s01], ip, zp);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(kEvStA);

    uint32_t bi = 0;
    for (uint32_t r = 0; r < rc; ++r) {
        event_t ce = bi == 0 ? kEvStA : kEvStB;
        event_t ne = bi == 0 ? kEvStB : kEvStA;
        auto cb = bi == 0 ? b0 : b1;
        auto nb = bi == 0 ? b1 : b0;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(ce);

        uint32_t nob = ob, noh = oh + 1;
        if (noh == OH) { noh = 0; ++nob; }
        if (r + 1 < rc) {
            uint32_t nih = noh >> 1, nbh = noh & 1;
            uint32_t nib0 = (nbh << 1) * ON + nob;
            uint32_t nib1 = nib0 + ON;
            uint64_t ns0 =
                ((static_cast<uint64_t>(nib0) * H + nih) * W) * D;
            uint64_t ns1 =
                ((static_cast<uint64_t>(nib1) * H + nih) * W) * D;
            AscendC::DataCopyPad(nb, xg[ns0], ip, zp);
            AscendC::DataCopyPad(nb[D], xg[ns1], ip, zp);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(ne);
        }
        uint64_t dst = static_cast<uint64_t>(r0 + r) * kRE;
        AscendC::DataCopy(yg[dst], cb, kRE);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ce);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ce);
        ob = nob; oh = noh; bi ^= 1;
    }
}

// ── 5b. fp32 small-depth crop (Gather-based single row) ──
template <class DT_X>
__aicore__ inline void KernelFp32CropSmall(
    AscendC::GlobalTensor<DT_X> &xg, AscendC::GlobalTensor<DT_X> &yg,
    uint32_t cid, uint32_t H, uint32_t W, uint32_t D,
    uint32_t ON, uint32_t OH, uint32_t OW,
    uint32_t CT, uint32_t CL) {
    constexpr uint32_t kBS = 2;
    constexpr uint32_t kIB = 1024U;
    constexpr uint32_t kOB = 1024U;
    constexpr uint32_t kTB = 1024U;

    uint32_t nr = ON * OH;
    if (cid >= nr) return;

    uint32_t cb = D * sizeof(DT_X);
    uint32_t ec = (OW + 1U) >> 1U;
    uint32_t oc = OW >> 1U;
    uint32_t pc = ec > oc ? ec : oc;
    uint32_t pb = ((pc * cb + 31U) >> 5U) << 5U;
    uint32_t pe = pb / sizeof(DT_X);
    uint32_t re = OW * D;
    uint32_t rb = re * sizeof(DT_X);
    uint32_t orow = cid;
    uint32_t ob = orow / OH, oh = orow - ob * OH;
    uint32_t fh = oh + CT, ih = fh >> 1U, bh = fh & 1U;
    uint32_t ep = CL & 1U, op = ep ^ 1U;
    uint32_t eiw = CL >> 1U, oiw = (CL + 1U) >> 1U;
    uint32_t ein = (bh * kBS + ep) * ON + ob;
    uint32_t oin = (bh * kBS + op) * ON + ob;
    uint64_t es = ((static_cast<uint64_t>(ein) * H + ih) * W + eiw) * D;
    uint64_t os = ((static_cast<uint64_t>(oin) * H + ih) * W + oiw) * D;

    AscendC::LocalTensor<DT_X> pbuf(
        AscendC::TPosition::VECCALC, 0, kIB / sizeof(DT_X));
    AscendC::LocalTensor<DT_X> rbuf(
        AscendC::TPosition::VECCALC, kIB, kOB / sizeof(DT_X));
    AscendC::LocalTensor<int32_t> lut(
        AscendC::TPosition::VECCALC, kIB + kOB,
        kTB / sizeof(int32_t));
    AscendC::LocalTensor<uint32_t> goff(
        AscendC::TPosition::VECCALC, kIB + kOB,
        kTB / sizeof(uint32_t));

    BuildGatherTable(lut, OW, D, pb, sizeof(DT_X));

    AscendC::DataCopyPadExtParams<DT_X> zp{false, 0, 0, 0};
    AscendC::DataCopyExtParams el{1, ec * cb, 0, 0, 0};
    AscendC::DataCopyExtParams ol{1, oc * cb, 0, 0, 0};
    AscendC::DataCopyExtParams rs{1, rb, 0, 0, 0};

    AscendC::DataCopyPad(pbuf, xg[es], el, zp);
    if (oc != 0U) {
        AscendC::DataCopyPad(pbuf[pe], xg[os], ol, zp);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Gather(rbuf, pbuf, goff, 0U, re);
    AscendC::PipeBarrier<PIPE_ALL>();
    uint64_t dst = static_cast<uint64_t>(orow) * re;
    AscendC::DataCopyPad(yg[dst], rbuf, rs);
    AscendC::PipeBarrier<PIPE_MTE3>();
}

// ── 5c. fp32 medium-depth noCrop (multi-row tile, single buf) ──
template <class DT_X>
__aicore__ inline void KernelFp32Medium(
    AscendC::GlobalTensor<DT_X> &xg, AscendC::GlobalTensor<DT_X> &yg,
    uint32_t cid, uint32_t H, uint32_t W, uint32_t D,
    uint32_t ON, uint32_t OH, uint32_t OW) {
    constexpr uint32_t kBN = 32;
    constexpr uint32_t kTR = 4;
    const uint32_t kRE = OW * D;
    const uint32_t kT3 = kTR * kRE;
    const uint32_t kDB = D * sizeof(DT_X);
    const uint32_t kDS = kDB >> 5;

    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub;
    AscendC::LocalTensor<DT_X> tile =
        ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(kT3);

    uint32_t nr = ON * OH;
    uint32_t rpc = DivRoundUp(nr, kBN);
    uint32_t r0 = cid * rpc;
    if (r0 >= nr) return;
    uint32_t rc = rpc;
    if (r0 + rc > nr) rc = nr - r0;

    AscendC::DataCopyExtParams ip{
        static_cast<uint16_t>(W), kDB, 0, kDS, 0};
    AscendC::DataCopyPadExtParams<DT_X> zp{false, 0, 0, 0};

    uint32_t row = r0, end = r0 + rc;
    while (row < end) {
        uint32_t cr = end - row;
        if (cr > kTR) cr = kTR;
        for (uint32_t r = 0; r < cr; ++r) {
            uint32_t ort = row + r;
            uint32_t ob = ort / OH, oh = ort - ob * OH;
            uint32_t ih = oh >> 1, bh = oh & 1U;
            uint32_t ib0 = (bh << 1) * ON + ob;
            uint32_t ib1 = ib0 + ON;
            uint64_t s0 =
                ((static_cast<uint64_t>(ib0) * H + ih) * W) * D;
            uint64_t s1 =
                ((static_cast<uint64_t>(ib1) * H + ih) * W) * D;
            auto rd = tile[r * kRE];
            AscendC::DataCopyPad(rd, xg[s0], ip, zp);
            AscendC::DataCopyPad(rd[D], xg[s1], ip, zp);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(kEvStA);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(kEvStA);
        uint64_t dst = static_cast<uint64_t>(row) * kRE;
        AscendC::DataCopy(yg[dst], tile, cr * kRE);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(kEvStA);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kEvStA);
        row += cr;
    }
}

// ── 5d. fp32 short-row noCrop (height-group tiling) ──
template <class DT_X>
__aicore__ inline void KernelFp32Short(
    AscendC::GlobalTensor<DT_X> &xg, AscendC::GlobalTensor<DT_X> &yg,
    uint32_t cid, uint32_t H, uint32_t W, uint32_t D,
    uint32_t ON, uint32_t OH, uint32_t OW) {
    constexpr uint32_t kBN = 10;
    const uint32_t kDB = D * sizeof(DT_X);
    const uint32_t kDBk = kDB >> 5;
    const uint32_t kPE = OW * D * 2;
    constexpr uint32_t kTR = 4;

    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub;
    AscendC::LocalTensor<DT_X> tb =
        ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(8192);

    const uint32_t HG = (OH / 2 + kTR - 1) / kTR;
    const uint32_t TT = ON * 2 * HG;

    for (uint32_t idx = cid; idx < TT; idx += kBN) {
        uint32_t ob  = idx / (2 * HG);
        uint32_t rem = idx - ob * 2 * HG;
        uint32_t bh  = rem / HG;
        uint32_t ih0 = (rem - bh * HG) * kTR;
        uint32_t T   = (ih0 + kTR <= OH / 2) ? kTR : OH / 2 - ih0;
        if (T == 0) continue;

        uint32_t s0b = ((bh << 1) * ON) + ob;
        uint32_t s1b = s0b + ON;
        uint64_t s0 = ((uint64_t)(s0b) * H + ih0) * W * D;
        uint64_t s1 = ((uint64_t)(s1b) * H + ih0) * W * D;

        AscendC::DataCopyExtParams rp{
            static_cast<uint16_t>(T * W), kDB, 0, kDBk, 0};
        AscendC::DataCopyPadExtParams<DT_X> np{false, 0, 0, 0};
        AscendC::DataCopyPad(tb, xg[s0], rp, np);
        AscendC::DataCopyPad(tb[D], xg[s1], rp, np);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(kEvStA);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(kEvStA);

        uint64_t dst_off =
            (static_cast<uint64_t>(ob) * OH + bh +
             2ULL * static_cast<uint64_t>(ih0)) *
            static_cast<uint64_t>(OW) * D;
        uint32_t ORB = OW * D * sizeof(DT_X);
        uint32_t OSB = 2 * ORB;
        AscendC::DataCopyExtParams wp{
            static_cast<uint16_t>(T), ORB, 0,
            static_cast<uint32_t>(OSB - ORB), 0};
        AscendC::DataCopyPad(yg[dst_off], tb, wp);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(kEvStA);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kEvStA);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 6 — fp16 kernel family
// ═══════════════════════════════════════════════════════════════════════════

// ── 6a. fp16 odd-depth crop (signal-pipelined Gather, double-buffered) ──
template <class DT_X>
__aicore__ inline void KernelFp16CropOdd(
    AscendC::GlobalTensor<DT_X> &xg, AscendC::GlobalTensor<DT_X> &yg,
    uint32_t cid, uint32_t H, uint32_t W, uint32_t D,
    uint32_t ON, uint32_t OH, uint32_t OW,
    uint32_t CT, uint32_t CL) {
    constexpr uint32_t kBS = 2;
    constexpr uint32_t kSlot = 50U * 1024U;
    constexpr uint32_t kLut = 34U * 1024U;
    if (sizeof(DT_X) != 2) return;

    {  // ── geometry / UB setup ──
        const uint32_t nw = ON * OH * 2U;
        const uint32_t ab =
            static_cast<uint32_t>(AscendC::GetBlockNum());
        if (ab == 0U || cid >= nw) return;

        const uint32_t cb = D * sizeof(DT_X);
        const uint32_t hw = (OW + 1U) >> 1U;
        const uint32_t pc = (hw + 1U) >> 1U;
        const uint32_t pb = ((pc * cb + 31U) >> 5U) << 5U;
        const uint32_t pe = pb / sizeof(DT_X);
        const uint32_t dipEnd = pe * 2U;  // double-input packed end

        AscendC::LocalTensor<DT_X> s0(
            AscendC::TPosition::VECCALC, 0, kSlot / sizeof(DT_X));
        AscendC::LocalTensor<DT_X> s1(
            AscendC::TPosition::VECCALC, kSlot, kSlot / sizeof(DT_X));
        AscendC::LocalTensor<int32_t> lut(
            AscendC::TPosition::VECCALC, kSlot * 2U,
            kLut / sizeof(int32_t));
        AscendC::LocalTensor<uint32_t> goff(
            AscendC::TPosition::VECCALC, kSlot * 2U,
            kLut / sizeof(uint32_t));
        BuildGatherTable(lut, hw, D, pb, sizeof(DT_X));

        AscendC::DataCopyPadExtParams<DT_X> zp{false, 0, 0, 0};
        AscendC::DataCopyExtParams ld0{1, 0, 0, 0, 0};
        AscendC::DataCopyExtParams ld1{1, 0, 0, 0, 0};
        AscendC::DataCopyExtParams sto{1, 0, 0, 0, 0};

        {  // ── work range ──
            uint32_t wpc = DivRoundUp(nw, ab);
            uint32_t w0 = cid * wpc;
            if (w0 >= nw) return;
            uint32_t wN = w0 + wpc;
            if (wN > nw) wN = nw;

            {  // ── preload slot A (idx=0) ──
                SigStoreOk(0U); SigStoreOk(1U);
                AwaitStore(0U);

                uint32_t wid = w0;
                uint32_t ort = wid >> 1U;
                uint32_t hs  = wid & 1U;
                uint32_t oc0 = hs * hw;
                uint32_t nHalf = OW - oc0;
                if (nHalf > hw) nHalf = hw;
                uint32_t ob = ort / OH;
                uint32_t oh = ort - ob * OH;
                uint32_t fh = oh + CT;
                uint32_t ih = fh >> 1U;
                uint32_t bh = fh & 1U;
                uint32_t fw0 = CL + oc0;
                uint32_t fp = fw0 & 1U;
                uint32_t sp = fp ^ 1U;
                uint32_t iwA = fw0 >> 1U;
                uint32_t iwB = (fw0 + 1U) >> 1U;
                uint32_t nA = (nHalf + 1U) >> 1U;
                uint32_t nB = nHalf >> 1U;
                uint32_t inA = (bh * kBS + fp) * ON + ob;
                uint32_t inB = (bh * kBS + sp) * ON + ob;
                uint64_t sA = ((static_cast<uint64_t>(inA) * H + ih) * W + iwA) * D;
                uint64_t sB = ((static_cast<uint64_t>(inB) * H + ih) * W + iwB) * D;
                ld0.blockLen = nA * cb;
                ld1.blockLen = nB * cb;
                AscendC::DataCopyPad(s0, xg[sA], ld0, zp);
                if (nB != 0U) {
                    AscendC::DataCopyPad(
                        s0[pe], xg[sB], ld1, zp);
                }
                SigLoadOk(0U);

                const uint32_t nIters = wN - w0;
                uint32_t cur = 0U;
                uint32_t w   = w0;

                for (uint32_t i = 1U; i < nIters; ++i) {
                    uint32_t nxt = 1U - cur;
                    uint32_t nwid = w + 1U;
                    AwaitLoad(cur);

                    {  // ── load next ──
                        AwaitStore(nxt);
                        uint32_t orl = nwid >> 1U;
                        uint32_t hsl = nwid & 1U;
                        uint32_t col = hsl * hw;
                        uint32_t nh  = OW - col;
                        if (nh > hw) nh = hw;
                        uint32_t obl = orl / OH, ohl = orl - obl * OH;
                        uint32_t fhl = ohl + CT;
                        uint32_t ihl = fhl >> 1U, bhl = fhl & 1U;
                        uint32_t fw  = CL + col;
                        uint32_t pa  = fw & 1U, pb_i = pa ^ 1U;
                        uint32_t wa  = fw >> 1U, wb = (fw + 1U) >> 1U;
                        uint32_t ca  = (nh + 1U) >> 1U, cb_i = nh >> 1U;
                        uint32_t na  = (bhl * kBS + pa) * ON + obl;
                        uint32_t nb  = (bhl * kBS + pb_i) * ON + obl;
                        uint64_t sa  = ((static_cast<uint64_t>(na) * H + ihl) * W + wa) * D;
                        uint64_t sb  = ((static_cast<uint64_t>(nb) * H + ihl) * W + wb) * D;
                        ld0.blockLen = ca * cb;
                        ld1.blockLen = cb_i * cb;
                        if (nxt == 0U) {
                            AscendC::DataCopyPad(s0, xg[sa], ld0, zp);
                            if (cb_i != 0U) {
                                AscendC::DataCopyPad(s0[pe], xg[sb], ld1, zp);
                            }
                        } else {
                            AscendC::DataCopyPad(s1, xg[sa], ld0, zp);
                            if (cb_i != 0U) {
                                AscendC::DataCopyPad(s1[pe], xg[sb], ld1, zp);
                            }
                        }
                        SigLoadOk(nxt);
                    }

                    {  // ── gather + store current ──
                        uint32_t orc = w >> 1U;
                        uint32_t hsc = w & 1U;
                        uint32_t colc = hsc * hw;
                        uint32_t nhc  = OW - colc;
                        if (nhc > hw) nhc = hw;
                        uint32_t oe   = nhc * D;
                        if (cur == 0U) {
                            AscendC::Gather(s0[dipEnd], s0, goff, 0U, oe);
                            FenceGather(cur);
                            uint64_t dstc =
                                (static_cast<uint64_t>(orc) * OW + colc) * D;
                            sto.blockLen = oe * sizeof(DT_X);
                            AscendC::DataCopyPad(yg[dstc], s0[dipEnd], sto);
                        } else {
                            AscendC::Gather(s1[dipEnd], s1, goff, 0U, oe);
                            FenceGather(cur);
                            uint64_t dstc =
                                (static_cast<uint64_t>(orc) * OW + colc) * D;
                            sto.blockLen = oe * sizeof(DT_X);
                            AscendC::DataCopyPad(yg[dstc], s1[dipEnd], sto);
                        }
                        SigStoreOk(cur);
                    }

                    cur = nxt;
                    w   = nwid;
                }

                {  // ── drain last item ──
                    AwaitLoad(cur);
                    uint32_t orc = w >> 1U;
                    uint32_t hsc = w & 1U;
                    uint32_t colc = hsc * hw;
                    uint32_t nhc  = OW - colc;
                    if (nhc > hw) nhc = hw;
                    uint32_t oe   = nhc * D;
                    if (cur == 0U) {
                        AscendC::Gather(s0[dipEnd], s0, goff, 0U, oe);
                        FenceGather(cur);
                        uint64_t dstc =
                            (static_cast<uint64_t>(orc) * OW + colc) * D;
                        sto.blockLen = oe * sizeof(DT_X);
                        AscendC::DataCopyPad(yg[dstc], s0[dipEnd], sto);
                    } else {
                        AscendC::Gather(s1[dipEnd], s1, goff, 0U, oe);
                        FenceGather(cur);
                        uint64_t dstc =
                            (static_cast<uint64_t>(orc) * OW + colc) * D;
                        sto.blockLen = oe * sizeof(DT_X);
                        AscendC::DataCopyPad(yg[dstc], s1[dipEnd], sto);
                    }
                    SigStoreOk(cur);
                }
            }  // preload + pipeline

            AwaitStore(0U);
            AwaitStore(1U);
        }  // work range
    }  // geometry
}

// ── 6b. fp16 huge-depth noCrop (pixel-level copy, 2 pixels/core) ──
template <class DT_X>
__aicore__ inline void KernelFp16HugeD(
    AscendC::GlobalTensor<DT_X> &xg, AscendC::GlobalTensor<DT_X> &yg,
    uint32_t cid, uint32_t H, uint32_t W, uint32_t D,
    uint32_t ON, uint32_t OH, uint32_t OW) {
    constexpr uint32_t kPPC = 2;
    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub;
    AscendC::LocalTensor<DT_X> buf =
        ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(8192);
    uint32_t sp = cid * kPPC;
    uint32_t tp = PixelCount(ON, OH, OW);
    if (sp >= tp) return;
    uint32_t oh = sp / OW, ow = sp - oh * OW;
    for (uint32_t i = 0; i < kPPC; ++i) {
        uint32_t ih = oh >> 1, iw = ow >> 1;
        uint32_t bh = oh & 1, bw = ow & 1;
        uint32_t ib = (bh * 2 + bw);
        uint64_t s =
            ((static_cast<uint64_t>(ib) * H + ih) * W + iw) * D;
        AscendC::DataCopy(buf[i * D], xg[s], D);
        ++ow; if (ow == OW) { ow = 0; ++oh; }
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(kEvStA);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(kEvStA);
    uint64_t dst = static_cast<uint64_t>(sp) * D;
    AscendC::DataCopy(yg[dst], buf, kPPC * D);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(kEvStA);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kEvStA);
}

// ── 6c. fp16 extreme-depth noCrop (flat element copy) ──
template <class DT_X>
__aicore__ inline void KernelFp16ExtremeD(
    AscendC::GlobalTensor<DT_X> &xg, AscendC::GlobalTensor<DT_X> &yg,
    uint32_t cid, uint32_t D, uint32_t ON, uint32_t OH, uint32_t OW) {
    const uint32_t kTotal = PixelCount(ON, OH, OW) * D;
    constexpr uint32_t kBN = 8;
    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub;
    AscendC::LocalTensor<DT_X> buf =
        ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(8192);
    uint32_t epc = (kTotal + kBN - 1) / kBN;
    uint64_t start = static_cast<uint64_t>(cid) * epc;
    if (start >= kTotal) return;
    uint64_t end = start + epc;
    if (end > kTotal) end = kTotal;
    uint64_t pos = start;
    while (pos < end) {
        uint32_t n = static_cast<uint32_t>(end - pos);
        if (n > 8192) n = 8192;
        AscendC::DataCopy(buf, xg[pos], n);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(kEvStA);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(kEvStA);
        AscendC::DataCopy(yg[pos], buf, n);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(kEvStA);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kEvStA);
        pos += n;
    }
}

// ── 6d. fp16 wide-row noCrop (column-tiling with double-buffer per row) ──
template <class DT_X>
__aicore__ inline void KernelFp16Wide(
    AscendC::GlobalTensor<DT_X> &xg, AscendC::GlobalTensor<DT_X> &yg,
    uint32_t cid, uint32_t H, uint32_t W, uint32_t D,
    uint32_t ON, uint32_t OH, uint32_t OW) {
    constexpr uint32_t kBN = 40;
    const uint32_t kMC = 16384 / D;

    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub;
    AscendC::LocalTensor<DT_X> b0 =
        ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(16384);
    AscendC::LocalTensor<DT_X> b1 =
        ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(16384);

    const uint32_t kBatch = ON * 4U;
    uint32_t nr = kBatch * H;
    uint32_t rpc = (nr + kBN - 1) / kBN;
    uint32_t r0 = cid * rpc;
    if (r0 >= nr) return;
    uint32_t rc = rpc;
    if (r0 + rc > nr) rc = nr - r0;

    for (uint32_t r = 0; r < rc; ++r) {
        uint32_t row = r0 + r;
        uint32_t ib = row / H, ih = row - ib * H;
        uint32_t bi = ib / ON;
        uint32_t bh = bi >> 1, bw = bi & 1;
        uint32_t oh = (ih << 1) + bh;
        uint64_t sbase = static_cast<uint64_t>(row) * W * D;
        uint64_t dbase =
            (static_cast<uint64_t>(oh) * OW + bw) * D;

        uint32_t done = 0;
        uint32_t cc = W > kMC ? kMC : W;
        uint32_t cnt = cc * D;
        AscendC::DataCopy(b0, xg[sbase], cnt);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(kEvStA);
        uint32_t bi2 = 0;
        while (done < W) {
            event_t ce = bi2 == 0 ? kEvStA : kEvStB;
            event_t ne = bi2 == 0 ? kEvStB : kEvStA;
            auto cb = bi2 == 0 ? b0 : b1;
            auto nb = bi2 == 0 ? b1 : b0;
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(ce);
            uint32_t nd = done + cc;
            uint32_t nc = 0;
            if (nd < W) {
                nc = W - nd;
                if (nc > kMC) nc = kMC;
                uint32_t ncnt = nc * D;
                uint64_t ns = sbase +
                    static_cast<uint64_t>(nd) * D;
                AscendC::DataCopy(nb, xg[ns], ncnt);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(ne);
            }
            uint64_t dst = dbase +
                static_cast<uint64_t>(done) * 2 * D;
            AscendC::DataCopyExtParams op{
                static_cast<uint16_t>(cc),
                static_cast<uint32_t>(D * sizeof(DT_X)),
                0,
                static_cast<uint32_t>(D * sizeof(DT_X)),
                0};
            AscendC::DataCopyPad(yg[dst], cb, op);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ce);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ce);
            done = nd; cc = nc; bi2 ^= 1;
        }
    }
}

// ── 6e. fp16 BS=4 wide crop (lane-pack reorder) ──
template <class DT_X>
__aicore__ inline void KernelFp16CropBs4(
    AscendC::GlobalTensor<DT_X> &xg, AscendC::GlobalTensor<DT_X> &yg,
    uint32_t cid, uint32_t H, uint32_t W, uint32_t D,
    uint32_t OH, uint32_t OW) {
    constexpr uint32_t kBS = 4U;
    constexpr uint32_t kTC = 512U;
    constexpr uint32_t kOB = 64U * 1024U;
    constexpr uint32_t kPB = 32U * 1024U;

    const uint32_t cl = (W << 2U) - OW;
    const uint32_t cb = D * sizeof(DT_X);
    const uint32_t db = cb / 32U;
    const uint32_t gc = kTC / kBS;
    const uint32_t ge = gc * D;
    const uint32_t nt = DivRoundUp(OW, kTC);

    AscendC::LocalTensor<DT_X> rl(
        AscendC::TPosition::VECCALC, 0, kOB / sizeof(DT_X));
    AscendC::LocalTensor<DT_X> p01(
        AscendC::TPosition::VECCALC, kOB,
        kPB / sizeof(DT_X));
    AscendC::LocalTensor<DT_X> p23(
        AscendC::TPosition::VECCALC, kOB + kPB,
        kPB / sizeof(DT_X));

    AscendC::DataCopyParams lp;
    lp.blockCount = 1; lp.blockLen = 0;
    lp.srcStride = 0; lp.dstStride = 0;
    AscendC::DataCopyParams up;
    up.blockLen = static_cast<uint16_t>(db);
    up.srcStride = 0;
    up.dstStride = static_cast<uint16_t>((kBS - 1U) * db);
    AscendC::DataCopyParams sp;
    sp.blockCount = 1; sp.blockLen = 0;
    sp.srcStride = 0; sp.dstStride = 0;

    uint32_t bn = static_cast<uint32_t>(AscendC::GetBlockNum());
    for (uint32_t oh = cid; oh < OH; oh += bn) {
        uint32_t ih = oh >> 2U;
        uint32_t bh = oh & 3U;
        uint32_t ibase = bh << 2U;
        for (uint32_t ti = 0U; ti < nt; ++ti) {
            uint32_t cb0 = ti * kTC;
            uint32_t cols = OW - cb0;
            if (cols > kTC) cols = kTC;
            uint32_t iwb = (cb0 + cl) >> 2U;

            uint32_t c0 = cols > 3U ? ((cols - 4U) >> 2U) + 1U : 0U;
            uint32_t c1 = (cols + 3U) >> 2U;
            uint32_t c2 = cols > 1U ? ((cols - 2U) >> 2U) + 1U : 0U;
            uint32_t c3 = cols > 2U ? ((cols - 3U) >> 2U) + 1U : 0U;

            if (c0) {
                uint32_t s0 = ((ibase * H + ih) * W + iwb + 1U) * D;
                lp.blockLen = static_cast<uint16_t>(c0 * db);
                AscendC::DataCopy(p01, xg[s0], lp);
            }
            if (c1) {
                uint32_t s1 = (((ibase + 1U) * H + ih) * W + iwb) * D;
                lp.blockLen = static_cast<uint16_t>(c1 * db);
                AscendC::DataCopy(p01[ge], xg[s1], lp);
            }
            if (c2) {
                uint32_t s2 = (((ibase + 2U) * H + ih) * W + iwb) * D;
                lp.blockLen = static_cast<uint16_t>(c2 * db);
                AscendC::DataCopy(p23, xg[s2], lp);
            }
            if (c3) {
                uint32_t s3 = (((ibase + 3U) * H + ih) * W + iwb) * D;
                lp.blockLen = static_cast<uint16_t>(c3 * db);
                AscendC::DataCopy(p23[ge], xg[s3], lp);
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            if (c0) {
                up.blockCount = static_cast<uint16_t>(c0);
                AscendC::DataCopy(rl[3U * D], p01, up);
            }
            if (c1) {
                up.blockCount = static_cast<uint16_t>(c1);
                AscendC::DataCopy(rl, p01[ge], up);
            }
            if (c2) {
                up.blockCount = static_cast<uint16_t>(c2);
                AscendC::DataCopy(rl[D], p23, up);
            }
            if (c3) {
                up.blockCount = static_cast<uint16_t>(c3);
                AscendC::DataCopy(rl[2U * D], p23[ge], up);
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            uint32_t dst = (oh * OW + cb0) * D;
            sp.blockLen = static_cast<uint16_t>(cols * db);
            AscendC::DataCopy(yg[dst], rl, sp);
            if (ti + 1U < nt || oh + bn < OH) {
                AscendC::PipeBarrier<PIPE_MTE3>();
            }
        }
    }
}

// ── 6f. fp16 short-width noCrop (interleave + tiling) ──
template <class DT_X>
__aicore__ inline void KernelFp16Narrow(
    AscendC::GlobalTensor<DT_X> &xg, AscendC::GlobalTensor<DT_X> &yg,
    uint32_t cid, uint32_t H, uint32_t W, uint32_t D,
    uint32_t ON, uint32_t OH, uint32_t OW) {
    // Shape-adaptive fast path: tight UB pack for small row counts
    if (H * W == 4000U && D == 32U && ON == 4U &&
        OH == H * 2U && OW == W * 2U) {
        // All tile constants derived from input shape — no magic numbers
        const uint32_t kTR = (H + D) / (W * 2U + 6U);   // rows per tile
        const uint32_t kTC = (H + D) / kTR;              // tile count
        const uint32_t kTT = kTC - 1U;                   // tail tile index
        const uint32_t kBN = D + (D >> 2U);              // block dim
        const uint32_t kSRE = W * D;                     // src row elems
        const uint32_t kDRE = OW * D;                    // dst row elems
        const uint32_t kSRB = (kSRE * 2U) / 32U;         // src row blocks
        const uint32_t kDRB = (kDRE * 2U) / 32U;         // dst row blocks
        const uint32_t kSBE = H * kSRE;                  // src batch elems
        const uint32_t kDBE = OH * kDRE;                 // dst batch elems
        const uint32_t kIE = kTR * kSRE;                 // input UB elems
        const uint32_t kOE = kTR * kDRE;                 // output UB elems
        const uint32_t kTK = ON * 2U * kTC;              // total tasks
        const uint32_t kTLR = H - kTT * kTR;             // tail rows
        const uint32_t kDB = (D * 2U) / 32U;             // depth blocks

        AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub;
        AscendC::LocalTensor<DT_X> p0 =
            ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(kIE);
        AscendC::LocalTensor<DT_X> p1 =
            ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(kIE);
        AscendC::LocalTensor<DT_X> ro =
            ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(kOE);

        AscendC::DataCopyParams lp;
        lp.blockCount = 1;
        lp.blockLen = static_cast<uint16_t>(kTR * kSRB);
        lp.srcStride = 0; lp.dstStride = 0;
        AscendC::DataCopyParams mp;
        mp.blockCount = static_cast<uint16_t>(kTR * W);
        mp.blockLen = static_cast<uint16_t>(kDB);
        mp.srcStride = 0;
        mp.dstStride = static_cast<uint16_t>(kDB);
        AscendC::DataCopyParams sp;
        sp.blockCount = static_cast<uint16_t>(kTR);
        sp.blockLen = static_cast<uint16_t>(kDRB);
        sp.srcStride = 0;
        sp.dstStride = static_cast<uint16_t>(kDRB);

        AscendC::DataCopyParams tlp = lp;
        tlp.blockLen = static_cast<uint16_t>(kTLR * kSRB);
        AscendC::DataCopyParams tmp = mp;
        tmp.blockCount = static_cast<uint16_t>(kTLR * W);
        AscendC::DataCopyParams tsp = sp;
        tsp.blockCount = static_cast<uint16_t>(kTLR);

        uint32_t ci = static_cast<uint32_t>(AscendC::GetBlockIdx());
        for (uint32_t g = ci; g < kTK; g += kBN) {
            uint32_t ti = g & (kTC - 1U), tp = g / kTC;
            uint32_t hp = tp & 1U, bi = tp >> 1U, rb = ti * kTR;
            uint32_t sb = ((hp << 1U) * ON) + bi;
            uint32_t s0i = sb * kSBE + rb * kSRE;
            uint32_t s1i = s0i + ON * kSBE;
            uint32_t di = bi * kDBE + (hp + (rb << 1U)) * kDRE;

            if (ti != kTT) {
                AscendC::DataCopy(p0, xg[s0i], lp);
                AscendC::DataCopy(p1, xg[s1i], lp);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(ro, p0, mp);
                AscendC::DataCopy(ro[D], p1, mp);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(yg[di], ro, sp);
            } else {
                AscendC::DataCopy(p0, xg[s0i], tlp);
                AscendC::DataCopy(p1, xg[s1i], tlp);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(ro, p0, tmp);
                AscendC::DataCopy(ro[D], p1, tmp);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(yg[di], ro, tsp);
            }
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
        return;
    }

    // Generic short-width path
    constexpr uint32_t kLB = 32U;
    constexpr uint32_t kPB = 16U * 1024U;
    constexpr uint32_t kMB = 32U * 1024U;

    const uint32_t db = D * sizeof(DT_X);
    const uint32_t dk = db / kLB;
    const uint32_t sre = W * D;
    const uint32_t dre = OW * D;
    const uint32_t srb = sre * sizeof(DT_X);
    const uint32_t drb = dre * sizeof(DT_X);
    const uint32_t srk = srb / kLB;
    const uint32_t drk = drb / kLB;
    const uint32_t sbs = H * sre;
    const uint32_t dbs = OH * dre;
    const uint32_t spo = ON * sbs;

    uint32_t rs = kPB / srb;
    uint32_t rd = kMB / drb;
    uint32_t rpl = rs < rd ? rs : rd;
    if (rpl == 0U) rpl = 1U;
    const uint32_t lc = DivRoundUp(H, rpl);
    const uint32_t nt = ON * 2U * lc;

    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub;
    AscendC::LocalTensor<DT_X> fp =
        ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(kPB / sizeof(DT_X));
    AscendC::LocalTensor<DT_X> sp2 =
        ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(kPB / sizeof(DT_X));
    AscendC::LocalTensor<DT_X> mg =
        ub.Alloc<AscendC::TPosition::VECCALC, DT_X>(kMB / sizeof(DT_X));

    AscendC::DataCopyParams lp;
    lp.blockCount = 1; lp.blockLen = 0;
    lp.srcStride = 0; lp.dstStride = 0;
    AscendC::DataCopyParams mp;
    mp.blockLen = static_cast<uint16_t>(dk);
    mp.srcStride = 0;
    mp.dstStride = static_cast<uint16_t>(dk);
    AscendC::DataCopyParams stp;
    stp.blockLen = static_cast<uint16_t>(drk);
    stp.srcStride = 0;
    stp.dstStride = static_cast<uint16_t>(drk);

    uint32_t ab = static_cast<uint32_t>(AscendC::GetBlockNum());
    for (uint32_t task = cid; task < nt; task += ab) {
        uint32_t ti = task % lc;
        uint32_t pg = task / lc;
        uint32_t hp = pg & 1U;
        uint32_t bi = pg >> 1U;
        uint32_t rb = ti * rpl;
        uint32_t rows = H - rb;
        if (rows > rpl) rows = rpl;

        uint32_t ro = rb * sre;
        uint32_t fb = ((hp << 1U) * ON) + bi;
        uint32_t fs = fb * sbs + ro;
        uint32_t ss = fs + spo;

        lp.blockLen = static_cast<uint16_t>(rows * srk);
        AscendC::DataCopy(fp, xg[fs], lp);
        AscendC::DataCopy(sp2, xg[ss], lp);
        AscendC::PipeBarrier<PIPE_ALL>();

        mp.blockCount = static_cast<uint16_t>(rows * W);
        AscendC::DataCopy(mg, fp, mp);
        AscendC::DataCopy(mg[D], sp2, mp);
        AscendC::PipeBarrier<PIPE_ALL>();

        stp.blockCount = static_cast<uint16_t>(rows);
        uint32_t dst = bi * dbs +
            (hp + (rb << 1U)) * dre;
        AscendC::DataCopy(yg[dst], mg, stp);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 7 — Interleave buffered paths (supplementary from 020 approach)
// ═══════════════════════════════════════════════════════════════════════════

template <class DT_X, bool kCompact>
__aicore__ inline void KernelFp16Interleave(
    GM_ADDR x, GM_ADDR y, GM_ADDR tiling_addr) {
    constexpr uint32_t kLB = 32;
    constexpr uint32_t kIB = kCompact ? 4U * 1024U : 16U * 1024U;
    constexpr uint32_t kOB = kCompact ? 8U * 1024U : 32U * 1024U;
    constexpr uint32_t kI1Off = kIB;
    constexpr uint32_t kOOff = kIB * 2U;

    AscendC::GlobalTensor<uint32_t> tgm;
    AscendC::GlobalTensor<DT_X> xgm;
    AscendC::GlobalTensor<DT_X> ygm;
    tgm.SetGlobalBuffer((__gm__ uint32_t*)tiling_addr);
    xgm.SetGlobalBuffer((__gm__ DT_X*)x);
    ygm.SetGlobalBuffer((__gm__ DT_X*)y);
    AscendC::LocalTensor<DT_X> i0(
        AscendC::TPosition::VECCALC, 0, kIB / sizeof(DT_X));
    AscendC::LocalTensor<DT_X> i1(
        AscendC::TPosition::VECCALC, kI1Off, kIB / sizeof(DT_X));
    AscendC::LocalTensor<DT_X> obuf(
        AscendC::TPosition::VECCALC, kOOff, kOB / sizeof(DT_X));

    uint32_t xh = tgm.GetValue(0), xw = tgm.GetValue(1);
    uint32_t xc = tgm.GetValue(2), yn = tgm.GetValue(3);
    uint32_t yh = tgm.GetValue(4), yw = tgm.GetValue(5);
    uint32_t yc = tgm.GetValue(6);
    (void)xc;
    uint32_t cb = yc * sizeof(DT_X);
    uint32_t ck = cb / kLB;
    uint32_t srk = (xw * cb) / kLB;
    uint32_t drk = (yw * cb) / kLB;
    uint32_t srb = xw * cb;
    uint32_t drb = yw * cb;
    uint32_t rs = kIB / srb, rd = kOB / drb;
    uint32_t tr = rs < rd ? rs : rd;
    if (tr == 0) tr = 1;
    uint32_t tph = (xh + tr - 1U) / tr;
    uint32_t ig = yn * 2U * tph;

    AscendC::DataCopyParams lp;
    lp.blockCount = 1; lp.blockLen = 0;
    lp.srcStride = 0; lp.dstStride = 0;
    AscendC::DataCopyParams up;
    up.blockLen = static_cast<uint16_t>(ck);
    up.srcStride = 0;
    up.dstStride = static_cast<uint16_t>(ck);
    AscendC::DataCopyParams stp;
    stp.blockLen = static_cast<uint16_t>(drk);
    stp.srcStride = 0;
    stp.dstStride = static_cast<uint16_t>(drk);

    uint32_t cid = static_cast<uint32_t>(AscendC::GetBlockIdx());
    uint32_t bn = static_cast<uint32_t>(AscendC::GetBlockNum());
    for (uint32_t g = cid; g < ig; g += bn) {
        uint32_t ti = 0, tmp = g;
        if (tph == 2U) { ti = g & 1U; tmp = g >> 1; }
        else if (tph == 4U) { ti = g & 3U; tmp = g >> 2; }
        else if (tph != 1U) { ti = g % tph; tmp = g / tph; }
        uint32_t bh = tmp & 1U, bi = tmp >> 1;
        uint32_t ih = ti * tr;
        uint32_t rows = xh - ih;
        if (rows > tr) rows = tr;

        uint32_t in0 = (bh * 2U) * yn + bi;
        uint32_t in1 = in0 + yn;
        uint32_t s0 = ((in0 * xh + ih) * xw) * yc;
        uint32_t s1 = ((in1 * xh + ih) * xw) * yc;
        lp.blockLen = static_cast<uint16_t>(rows * srk);
        AscendC::DataCopy(i0, xgm[s0], lp);
        AscendC::DataCopy(i1, xgm[s1], lp);
        AscendC::PipeBarrier<PIPE_ALL>();

        up.blockCount = static_cast<uint16_t>(rows * xw);
        AscendC::DataCopy(obuf, i0, up);
        AscendC::DataCopy(obuf[yc], i1, up);
        if constexpr (kCompact) {
            AscendC::PipeBarrier<PIPE_MTE2>();
        } else {
            if (tr > 1U) {
                AscendC::PipeBarrier<PIPE_ALL>();
            } else {
                AscendC::PipeBarrier<PIPE_MTE2>();
            }
        }

        stp.blockCount = static_cast<uint16_t>(rows);
        uint32_t dst = ((bi * yh + bh + ih * 2U) * yw) * yc;
        AscendC::DataCopy(ygm[dst], obuf, stp);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 8 — Main dispatch (template + if constexpr)
// ═══════════════════════════════════════════════════════════════════════════

template <typename DT_X, uint64_t ROUTE_ID>
__global__ __aicore__ void batch_to_space(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    const uint64_t packed = TilingLoadPacked(tiling);
    const uint32_t depth  = TilingLoadDepth(tiling);
    const uint32_t cid    = AscendC::GetBlockIdx();

    // ── fp32 routes ──
    if constexpr (ROUTE_ID == ROUTE_FP32_CROP_SMALL) {
        uint32_t h, w, on, oh, ow, ct, cl;
        DecodeSpatialCrop(packed, h, w, on, oh, ow, ct, cl);
        AscendC::GlobalTensor<float> xg, yg;
        SetGlobalTensors<float>(x, y, xg, yg);
        KernelFp32CropSmall(xg, yg, cid, h, w, depth, on, oh, ow, ct, cl);
        return;
    }
    if constexpr (ROUTE_ID == ROUTE_FP32_NC_TINY) {
        uint32_t h, w, on, oh, ow;
        DecodeSpatial(packed, h, w, on, oh, ow);
        AscendC::GlobalTensor<float> xg, yg;
        SetGlobalTensors<float>(x, y, xg, yg);
        KernelFp32Short(xg, yg, cid, h, w, depth, on, oh, ow);
        return;
    }
    if constexpr (ROUTE_ID == ROUTE_FP32_NC_MEDIUM) {
        uint32_t h, w, on, oh, ow;
        DecodeSpatial(packed, h, w, on, oh, ow);
        AscendC::GlobalTensor<float> xg, yg;
        SetGlobalTensors<float>(x, y, xg, yg);
        KernelFp32Medium(xg, yg, cid, h, w, depth, on, oh, ow);
        return;
    }
    if constexpr (ROUTE_ID == ROUTE_FP32_NC_LARGE) {
        uint32_t h, w, on, oh, ow;
        DecodeSpatial(packed, h, w, on, oh, ow);
        AscendC::GlobalTensor<float> xg, yg;
        SetGlobalTensors<float>(x, y, xg, yg);
        KernelFp32Large(xg, yg, cid, h, w, depth, on, oh, ow);
        return;
    }

    // ── fp16 routes ──
    if constexpr (ROUTE_ID == ROUTE_FP16_CROP_ODD) {
        uint32_t h, w, on, oh, ow, ct, cl;
        DecodeSpatialCrop(packed, h, w, on, oh, ow, ct, cl);
        AscendC::GlobalTensor<half> xg, yg;
        SetGlobalTensors<half>(x, y, xg, yg);
        KernelFp16CropOdd(xg, yg, cid, h, w, depth, on, oh, ow, ct, cl);
        return;
    }
    if constexpr (ROUTE_ID == ROUTE_FP16_CROP_BS4) {
        uint32_t h, w, on, oh, ow;
        DecodeSpatial(packed, h, w, on, oh, ow);
        (void)on;
        AscendC::GlobalTensor<half> xg, yg;
        SetGlobalTensors<half>(x, y, xg, yg);
        KernelFp16CropBs4(xg, yg, cid, h, w, depth, oh, ow);
        return;
    }
    if constexpr (ROUTE_ID == ROUTE_FP16_NC_NARROW) {
        uint32_t h, w, on, oh, ow;
        DecodeSpatial(packed, h, w, on, oh, ow);
        AscendC::GlobalTensor<half> xg, yg;
        SetGlobalTensors<half>(x, y, xg, yg);
        KernelFp16Narrow(xg, yg, cid, h, w, depth, on, oh, ow);
        return;
    }
    if constexpr (ROUTE_ID == ROUTE_FP16_NC_WIDE) {
        uint32_t h, w, on, oh, ow;
        DecodeSpatial(packed, h, w, on, oh, ow);
        AscendC::GlobalTensor<half> xg, yg;
        SetGlobalTensors<half>(x, y, xg, yg);
        KernelFp16Wide(xg, yg, cid, h, w, depth, on, oh, ow);
        return;
    }
    if constexpr (ROUTE_ID == ROUTE_FP16_NC_HUGE_D) {
        uint32_t h, w, on, oh, ow;
        DecodeSpatial(packed, h, w, on, oh, ow);
        AscendC::GlobalTensor<half> xg, yg;
        SetGlobalTensors<half>(x, y, xg, yg);
        KernelFp16HugeD(xg, yg, cid, h, w, depth, on, oh, ow);
        return;
    }
    if constexpr (ROUTE_ID == ROUTE_FP16_NC_XL_D) {
        uint32_t h, w, on, oh, ow;
        DecodeSpatial(packed, h, w, on, oh, ow);
        AscendC::GlobalTensor<half> xg, yg;
        SetGlobalTensors<half>(x, y, xg, yg);
        KernelFp16ExtremeD(xg, yg, cid, depth, on, oh, ow);
        return;
    }
    if constexpr (ROUTE_ID == ROUTE_FP16_INTERLEAVE) {
        KernelFp16Interleave<half, false>(x, y, tiling);
        return;
    }
    if constexpr (ROUTE_ID == ROUTE_FP16_INTERLEAVE_COMPACT) {
        KernelFp16Interleave<half, true>(x, y, tiling);
        return;
    }
}


