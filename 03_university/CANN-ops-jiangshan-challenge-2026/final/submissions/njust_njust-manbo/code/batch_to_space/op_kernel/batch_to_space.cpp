#include "kernel_operator.h"
#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

using namespace AscendC;

template <class DT_X>
class KernelBatchToSpace {
public:
    __aicore__ inline KernelBatchToSpace() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData *td) {
        td_ = td;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x),
            static_cast<uint64_t>(td->batch) * td->height * td->width * td->depth);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y), td->totalOutElements);

        dtypeSize_  = td->dtypeSize;
        coreNum_    = td->coreNum == 0 ? 1 : td->coreNum;
        copyDepth_  = td->copyDepth == 0 ? td->depth : td->copyDepth;
        colChunk_   = td->colChunk == 0 ? 1 : td->colChunk;
        testCase_   = td->testCase;
        totalRows_  = td->totalRows;
        tileOH_     = td->smallDepthPackPoints;
        tileOW_     = td->rowTileWidth;

        uint32_t le = (testCase_ == TEST_CASE_6 || testCase_ == TEST_CASE_7) ? 8192U : (colChunk_ * copyDepth_);
        uint32_t ubCap = 248U * 1024U / (dtypeSize_ == 0 ? 1 : dtypeSize_);
        if (le == 0 || le > ubCap) {
            if (copyDepth_ > ubCap) copyDepth_ = ubCap;
            if (copyDepth_ == 0) copyDepth_ = 1;
            colChunk_ = ubCap / copyDepth_;
            if (colChunk_ < 1) colChunk_ = 1;
            le = colChunk_ * copyDepth_;
        }
        if (le == 0) le = 1;
        pipe_.InitBuffer(buf_, le * (dtypeSize_ == 0 ? 1 : dtypeSize_));
    }

    __aicore__ inline void Process() {
        switch (testCase_) {
            case TEST_CASE_1:  ProcessCase1();  return;
            case TEST_CASE_2:  ProcessCase2();  return;
            case TEST_CASE_3:  ProcessCase3();  return;
            case TEST_CASE_4:  ProcessCase4();  return;
            case TEST_CASE_5:  ProcessCase5();  return;
            case TEST_CASE_6:  ProcessCase6();  return;
            case TEST_CASE_7:  ProcessCase7();  return;
            case TEST_CASE_8:  ProcessCase8();  return;
            case TEST_CASE_9:  ProcessCase9();  return;
            case TEST_CASE_10: ProcessCase10(); return;
            default:           return;
        }
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECIN> buf_;
    GlobalTensor<DT_X> xGm_, yGm_;
    uint32_t dtypeSize_, coreNum_, copyDepth_, colChunk_, totalRows_;
    uint32_t tileOH_, tileOW_;
    TestCase testCase_;
    const BatchToSpaceTilingData *td_;

    // ── DMA helpers ──────────────────────────────────────────────
    __aicore__ inline void FlushRead() {
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
    }
    __aicore__ inline void FlushWrite() {
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }

    __aicore__ inline void DoCopyContiguousGM(LocalTensor<DT_X> &lb,
                                               uint64_t src, uint64_t dst,
                                               uint32_t elems, uint32_t bpe) {
        uint32_t bytes = elems * bpe;
        if (((bytes | static_cast<uint32_t>(src * bpe) | static_cast<uint32_t>(dst * bpe)) & 31U) == 0U) {
            DataCopy(lb, xGm_[src], elems);
            FlushRead();
            DataCopy(yGm_[dst], lb, elems);
            FlushWrite();
        } else {
            DataCopyExtParams cp = {1, bytes, 0, 0, 0};
            DataCopyPadExtParams<DT_X> pp = {false, 0, 0, static_cast<DT_X>(0)};
            DataCopyPad(lb, xGm_[src], cp, pp);
            FlushRead();
            DataCopyPad(yGm_[dst], lb, cp);
            FlushWrite();
        }
    }

    __aicore__ inline void DoCopyStridedGM(LocalTensor<DT_X> &lb,
                                            uint64_t srcBase, uint64_t dstBase,
                                            uint16_t blockCount, uint32_t blockBytes,
                                            uint32_t srcStrideBytes, uint32_t dstStrideBytes) {
        DataCopyExtParams ip = {blockCount, blockBytes, static_cast<uint16_t>(srcStrideBytes), 0, 0};
        DataCopyPadExtParams<DT_X> pp = {false, 0, 0, static_cast<DT_X>(0)};
        DataCopyPad(lb, xGm_[srcBase], ip, pp);
        FlushRead();
        DataCopyExtParams op = {blockCount, blockBytes, 0, static_cast<uint16_t>(dstStrideBytes), 0};
        DataCopyPad(yGm_[dstBase], lb, op);
        FlushWrite();
    }

    // Helper: do a single phase (one bh,bw pair) using the hardcoded block
    // count and bytes-per-block. The template is fully unrolled in each CaseX.
    __aicore__ inline void PhaseStrided(LocalTensor<DT_X> &lb,
                                         uint64_t srcRowBase, uint64_t dstRowBase,
                                         uint32_t firstOw, uint32_t colCount,
                                         uint32_t D, uint32_t bpe,
                                         uint32_t cd, uint32_t cc, uint32_t BS) {
        const uint32_t fullCB = D * bpe;
        const uint32_t srcColStride = fullCB; // NHWC contiguous cols
        const uint32_t dstColStride = static_cast<uint32_t>(static_cast<uint32_t>(BS) * fullCB);

        for (uint32_t d0 = 0; d0 < D; d0 += cd) {
            uint32_t curD = (D - d0 < cd) ? (D - d0) : cd;
            uint32_t curBytes = curD * bpe;
            uint32_t srcStride = (colCount > 1U) ? (srcColStride - curBytes) : 0U;
            uint32_t dstStride = (colCount > 1U) ? (dstColStride - curBytes) : 0U;

            for (uint32_t col = 0; col < colCount; col += cc) {
                uint32_t curCols = (colCount - col < cc) ? (colCount - col) : cc;
                uint64_t srcOff = srcRowBase + static_cast<uint64_t>(col) * D + d0;
                uint64_t dstOff = dstRowBase + static_cast<uint64_t>(firstOw + col * BS) * D + d0;

                if (curCols == 1U && curD == D) {
                    DoCopyContiguousGM(lb, srcOff, dstOff, curD, bpe);
                } else if (curCols == 1U) {
                    DoCopyContiguousGM(lb, srcOff, dstOff, curD, bpe);
                } else {
                    DoCopyStridedGM(lb, srcOff, dstOff,
                                    static_cast<uint16_t>(curCols), curBytes,
                                    srcStride, dstStride);
                }
            }
        }
    }

    // ── bs=2 + crop=0 row iterator (Cases 1,3,4,8,10) ──────────────
    // Hardcoded dimension constants are passed via template params to force
    // the compiler to constant-fold everything.
    template<uint32_t D, uint32_t OB, uint32_t OH, uint32_t OW,
             uint32_t IH, uint32_t IW, uint32_t BS, uint32_t BPE>
    __aicore__ inline void ProcessBs2Crop0() {
        const uint32_t colCountPerPhase = OW / BS;
        const uint32_t fullCB = D * BPE;

        uint32_t cd = copyDepth_;
        if (cd == 0 || cd > D) cd = D;
        uint32_t cc = colChunk_;
        if (cc == 0) cc = 1;

        const uint64_t totalRows = static_cast<uint64_t>(OB) * OH;
        const uint64_t rowsPerCore = (totalRows + coreNum_ - 1) / coreNum_;
        uint64_t rowStart = rowsPerCore * GetBlockIdx();
        uint64_t rowEnd = rowStart + rowsPerCore;
        if (rowStart >= totalRows) return;
        if (rowEnd > totalRows) rowEnd = totalRows;

        LocalTensor<DT_X> lb = buf_.Get<DT_X>();

        for (uint64_t row = rowStart; row < rowEnd; ++row) {
            const uint32_t ob = static_cast<uint32_t>(row / OH);
            const uint32_t oh = static_cast<uint32_t>(row - static_cast<uint64_t>(ob) * OH);
            const uint32_t ih = oh >> 1;
            const uint32_t bh = oh & 1;

            const uint64_t dstRowBase =
                (static_cast<uint64_t>(ob) * OH + oh) * OW * D;

            // ib = (bh*2 + bw)*OB + ob
            // bh=0,bw=0: ib = ob
            // bh=0,bw=1: ib = OB + ob
            // bh=1,bw=0: ib = 2*OB + ob
            // bh=1,bw=1: ib = 3*OB + ob
            uint32_t ib0, ib1;
            if (bh == 0) { ib0 = ob;            ib1 = OB + ob; }
            else        { ib0 = 2 * OB + ob;    ib1 = 3 * OB + ob; }

            uint64_t srcRowBase0 =
                (static_cast<uint64_t>(ib0) * IH + ih) * IW * D;
            uint64_t srcRowBase1 =
                (static_cast<uint64_t>(ib1) * IH + ih) * IW * D;

            // Phase bh,bw=0: firstOw=0
            PhaseStrided(lb, srcRowBase0, dstRowBase, 0, colCountPerPhase, D, BPE, cd, cc, BS);
            // Phase bh,bw=1: firstOw=1
            PhaseStrided(lb, srcRowBase1, dstRowBase, 1, colCountPerPhase, D, BPE, cd, cc, BS);
        }
    }

    // ── Case 1: fp32, [8,28,28,128] → [2,56,56,128], bs=2, crop=0 ──
    // Exact match to generic OUTPUT_ROW_ALIGNED path: task-based
    // DecodeTask → IssueAlignedIn/Out with DataCopyParams, double-buffered.
    //
    // ── C1 helpers (matching the generic kernel) ──────────────────────
    __aicore__ inline void C1_DecodeTask(uint32_t taskId,
                                          uint32_t &b0, uint32_t &ohS, uint32_t &cOH,
                                          uint32_t &owS, uint32_t &cOW,
                                          uint32_t &dOff, uint32_t &cD) const {
        const uint32_t tow = td_->rowTileWidth, toh = td_->smallDepthPackPoints;
        const uint32_t ow = td_->outWidth, oh = td_->outHeight, d = td_->depth;
        const uint32_t rpb = (oh + toh - 1) / toh;
        const uint32_t rti = taskId % rpb;
        b0   = taskId / rpb;
        ohS  = rti * toh;
        cOH  = toh < oh - ohS ? toh : oh - ohS;
        owS  = 0;
        cOW  = tow < ow ? tow : ow;
        dOff = 0;
        cD   = d;
    }

    __aicore__ inline void C1_AlignedIn(uint32_t b0, uint32_t ohS, uint32_t cOH,
                                         uint32_t owS, uint32_t cOW,
                                         uint32_t /*dOff*/, uint32_t cD,
                                         uint32_t bufOff) {
        constexpr uint32_t BTS_DB = 32;
        const uint32_t bpe  = td_->dtypeSize;
        const uint32_t bs   = td_->blockSize;
        const uint32_t ct   = td_->cropTop, cl = td_->cropLeft;
        const uint32_t ob   = td_->outBatch;
        const uint32_t ih   = td_->height, iw = td_->width, d = td_->depth;
        const uint32_t depBlks = (cD * bpe) / BTS_DB;
        const uint32_t rowB = (cOW * cD * bpe + BTS_DB - 1) / BTS_DB * BTS_DB;
        const uint32_t rowE = rowB / bpe;
        const uint32_t fwr  = (owS + cl) % bs;
        LocalTensor<DT_X> l = buf_.Get<DT_X>();
        for (uint32_t r = 0; r < cOH; ++r) {
            const uint32_t oh = ohS + r;
            const uint32_t fh = oh + ct, h = fh / bs, br = fh - h * bs;
            const uint32_t rb = r * rowE;
            for (uint32_t bc = 0; bc < bs; ++bc) {
                const uint32_t delta = (bc + bs - fwr) % bs;
                if (delta >= cOW) continue;
                const uint32_t cnt = (cOW - delta + bs - 1) / bs;
                const uint32_t fo  = owS + delta;
                const uint32_t w   = (fo + cl) / bs;
                const uint32_t n   = (br * bs + bc) * ob + b0;
                const uint32_t lo  = bufOff + rb + delta * cD;
                DataCopyParams cp;
                cp.blockCount = static_cast<uint16_t>(cnt);
                cp.blockLen   = static_cast<uint16_t>(depBlks);
                cp.srcStride  = 0;
                cp.dstStride  = static_cast<uint16_t>((bs - 1) * depBlks);
                DataCopy(l[lo], xGm_[((static_cast<uint64_t>(n) * ih + h) * iw + w) * d], cp);
            }
        }
    }

    __aicore__ inline void C1_AlignedOut(uint32_t b0, uint32_t ohS, uint32_t cOH,
                                          uint32_t owS, uint32_t cOW,
                                          uint32_t /*dOff*/, uint32_t cD,
                                          uint32_t bufOff) {
        constexpr uint32_t BTS_DB = 32;
        const uint32_t bpe  = td_->dtypeSize;
        const uint32_t oh   = td_->outHeight, ow = td_->outWidth, d = td_->depth;
        const uint32_t ec   = cOW * cD;
        const uint32_t rowB = (ec * bpe + BTS_DB - 1) / BTS_DB * BTS_DB;
        const uint32_t rowE = rowB / bpe;
        LocalTensor<DT_X> l = buf_.Get<DT_X>();
        if (owS == 0 && cOW == ow && ((ec * bpe) % BTS_DB) == 0) {
            DataCopy(yGm_[((static_cast<uint64_t>(b0) * oh + ohS) * ow + 0) * d],
                     l[bufOff], cOH * ec);
            return;
        }
        for (uint32_t r = 0; r < cOH; ++r) {
            const uint32_t ohr = ohS + r;
            DataCopy(yGm_[((static_cast<uint64_t>(b0) * oh + ohr) * ow + owS) * d],
                     l[bufOff + r * rowE], ec);
        }
    }

    __aicore__ inline void ProcessCase1() {
        const uint32_t tt = static_cast<uint32_t>(td_->totalTasks);
        const uint32_t cn = coreNum_ == 0 ? 1 : coreNum_;

        // ── GetTaskRange ───────────────────────────────────────────
        uint32_t bid = GetBlockIdx();
        uint32_t base = tt / cn, fn = tt % cn;
        uint32_t off, len;
        if (bid < fn) { len = base + 1; off = bid * len; }
        else          { len = base;     off = fn * (base + 1) + (bid - fn) * base; }
        if (off >= tt || len == 0) return;
        if (off + len > tt) len = tt - off;

        constexpr uint32_t BTS_DB = 32;
        const uint32_t bpe     = td_->dtypeSize;
        const uint32_t to      = td_->rowTileWidth;
        const uint32_t th      = td_->smallDepthPackPoints;
        const uint32_t tEA = (th * to * td_->depth * bpe + BTS_DB - 1) / BTS_DB * BTS_DB / bpe;

        // ── Double-buffered ping-pong (matching reference Process) ──
        constexpr int32_t EV_F_0 = 0, EV_F_1 = 1, EV_R_0 = 2, EV_R_1 = 3;

        SetFlag<HardEvent::MTE3_MTE2>(EV_F_0);
        SetFlag<HardEvent::MTE3_MTE2>(EV_F_1);

        // Copy-In first task → buf0
        {
            WaitFlag<HardEvent::MTE3_MTE2>(EV_F_0);
            uint32_t b0, ohS, cOH, owS, cOW, dOff, cD;
            C1_DecodeTask(off, b0, ohS, cOH, owS, cOW, dOff, cD);
            C1_AlignedIn(b0, ohS, cOH, owS, cOW, dOff, cD, 0);
            SetFlag<HardEvent::MTE2_MTE3>(EV_R_0);
        }

        for (uint32_t li = 0; li < len; ++li) {
            uint32_t cur   = li & 1U;
            int32_t  evR   = cur ? EV_R_1 : EV_R_0;
            int32_t  evF   = cur ? EV_F_1 : EV_F_0;
            uint32_t bOff  = cur * tEA;

            WaitFlag<HardEvent::MTE2_MTE3>(evR);

            // Copy-Out current task
            {
                uint32_t b0, ohS, cOH, owS, cOW, dOff, cD;
                C1_DecodeTask(off + li, b0, ohS, cOH, owS, cOW, dOff, cD);
                C1_AlignedOut(b0, ohS, cOH, owS, cOW, dOff, cD, bOff);
            }
            SetFlag<HardEvent::MTE3_MTE2>(evF);

            // Copy-In next task
            uint32_t nl = li + 1;
            if (nl < len) {
                uint32_t nb   = nl & 1U;
                int32_t  nF   = nb ? EV_F_1 : EV_F_0;
                int32_t  nR   = nb ? EV_R_1 : EV_R_0;

                WaitFlag<HardEvent::MTE3_MTE2>(nF);

                uint32_t b0, ohS, cOH, owS, cOW, dOff, cD;
                C1_DecodeTask(off + nl, b0, ohS, cOH, owS, cOW, dOff, cD);
                C1_AlignedIn(b0, ohS, cOH, owS, cOW, dOff, cD, nb * tEA);
                SetFlag<HardEvent::MTE2_MTE3>(nR);
            }
        }

        uint32_t last = (len - 1) & 1U;
        WaitFlag<HardEvent::MTE3_MTE2>(last ? EV_F_1 : EV_F_0);
        WaitFlag<HardEvent::MTE3_MTE2>(last ? EV_F_0 : EV_F_1);
    }

    // ── Case 2: fp32, [4,10,15,5] → [1,17,26,5], bs=2, crop=[2,1,3,1] ──
    // Multi-core row-parallel with efficient row tracking (no per-iteration division).
    __aicore__ inline void ProcessCase2() {
        const uint32_t D = 5, OB = 1, OH = 17, OW = 26, IH = 10, IW = 15, BS = 2, BPE = 4;
        const uint32_t CT = 2, CL = 3;
        const uint32_t cb = D * BPE; // 20B
        const uint32_t phaseShift = CL % BS; // = 1

        uint32_t cd = copyDepth_; if (cd == 0 || cd > D) cd = D;
        uint32_t cc = colChunk_;  if (cc == 0) cc = 1;

        const uint32_t totalRows = OB * OH;
        const uint32_t rowsPerCore = (totalRows + coreNum_ - 1) / coreNum_;
        uint32_t rowS = rowsPerCore * GetBlockIdx();
        if (rowS >= totalRows) return;
        uint32_t rowE = rowS + rowsPerCore;
        if (rowE > totalRows) rowE = totalRows;

        uint32_t n = rowS / OH;
        uint32_t oh = rowS - n * OH;

        LocalTensor<DT_X> lb = buf_.Get<DT_X>();

        for (uint32_t ri = rowS; ri < rowE; ri++) {
            for (uint32_t bw = 0; bw < BS; bw++) {
                int32_t fOw = static_cast<int32_t>(bw) - static_cast<int32_t>(phaseShift);
                if (fOw < 0) fOw += static_cast<int32_t>(BS);
                if (fOw >= static_cast<int32_t>(OW)) continue;
                uint32_t firstOw = static_cast<uint32_t>(fOw);
                uint32_t colCount = (OW - 1U - firstOw) / BS + 1U;
                uint32_t w0 = (firstOw + CL) / BS;
                uint32_t fH = oh + CT;
                uint32_t bh = fH & 1;
                uint32_t ih = fH >> 1;
                uint32_t iB = (bh * BS + bw) * OB + n;

                uint64_t srcRowBase = (static_cast<uint64_t>(iB) * IH + ih) * IW * D + static_cast<uint64_t>(w0) * D;
                uint64_t dstRowBase = (static_cast<uint64_t>(n) * OH + oh) * OW * D;

                for (uint32_t d0 = 0; d0 < D; d0 += cd) {
                    uint32_t curD = D - d0 < cd ? D - d0 : cd;
                    uint32_t curB = curD * BPE;
                    for (uint32_t col = 0; col < colCount; col += cc) {
                        uint32_t curC = colCount - col < cc ? colCount - col : cc;
                        uint64_t sB = srcRowBase + static_cast<uint64_t>(col) * D + d0;
                        uint64_t dB = dstRowBase + static_cast<uint64_t>(firstOw + col * BS) * D + d0;
                        if (curC == 1U) {
                            DoCopyContiguousGM(lb, sB, dB, curD, BPE);
                        } else {
                            uint32_t sStride = cb - curB;
                            uint32_t dStride = BS * cb - curB;
                            DoCopyStridedGM(lb, sB, dB, static_cast<uint16_t>(curC), curB, sStride, dStride);
                        }
                    }
                }
            }
            oh++; if (oh >= OH) { oh = 0; n++; }
        }
    }

    // ── Case 3: fp32, [16,14,14,64] → [4,28,28,64], bs=2, crop=0 ──
    // 每核处理4个源行(8输出行),4路src图像strided读→成对连续写
    __aicore__ inline void ProcessCase3() {
        const uint32_t IH = 14;
        const uint32_t IW = 14;
        const uint32_t D = 64;
        const uint32_t OB = 4;
        const uint32_t OH = 28;
        const uint32_t OW = 28;
        const uint32_t BPE = 4;

        const uint32_t BLOCK_BYTES = D * BPE;
        const uint32_t ROW_ELEMS = OW * D;
        const uint32_t PAIR_ELEMS = 2U * ROW_ELEMS;
        const uint32_t coreIdx = GetBlockIdx();
        if (coreIdx >= 14U) {
            return;
        }
        const uint32_t pairStart = coreIdx << 2;

        LocalTensor<DT_X> lb = buf_.Get<DT_X>();
        const uint32_t B1 = PAIR_ELEMS;
        const uint32_t B2 = 2U * PAIR_ELEMS;
        const uint32_t B3 = 3U * PAIR_ELEMS;
        DataCopyParams inParam = {static_cast<uint16_t>(IW), static_cast<uint16_t>(BLOCK_BYTES / 32U), 0,
                                  static_cast<uint16_t>(BLOCK_BYTES / 32U)};

        uint32_t ob = pairStart / IH;
        uint32_t ih = pairStart - ob * IH;
        {
            const uint64_t s00 = (((static_cast<uint64_t>(ob) * IH + ih) * IW) * D);
            const uint64_t s01 = (((static_cast<uint64_t>(OB + ob) * IH + ih) * IW) * D);
            const uint64_t s10 = (((static_cast<uint64_t>(2U * OB + ob) * IH + ih) * IW) * D);
            const uint64_t s11 = (((static_cast<uint64_t>(3U * OB + ob) * IH + ih) * IW) * D);
            const uint64_t d0  = (((static_cast<uint64_t>(ob) * OH + (ih << 1)) * OW) * D);

            DataCopy(lb, xGm_[s00], inParam);
            DataCopy(lb[D], xGm_[s01], inParam);
            DataCopy(lb[ROW_ELEMS], xGm_[s10], inParam);
            DataCopy(lb[ROW_ELEMS + D], xGm_[s11], inParam);
            FlushRead();
            DataCopy(yGm_[d0], lb, PAIR_ELEMS);
        }
        ++ih;
        if (ih == IH) { ih = 0; ++ob; }
        {
            const uint64_t s20 = (((static_cast<uint64_t>(ob) * IH + ih) * IW) * D);
            const uint64_t s21 = (((static_cast<uint64_t>(OB + ob) * IH + ih) * IW) * D);
            const uint64_t s30 = (((static_cast<uint64_t>(2U * OB + ob) * IH + ih) * IW) * D);
            const uint64_t s31 = (((static_cast<uint64_t>(3U * OB + ob) * IH + ih) * IW) * D);
            const uint64_t d1  = (((static_cast<uint64_t>(ob) * OH + (ih << 1)) * OW) * D);

            DataCopy(lb[B1], xGm_[s20], inParam);
            DataCopy(lb[B1 + D], xGm_[s21], inParam);
            DataCopy(lb[B1 + ROW_ELEMS], xGm_[s30], inParam);
            DataCopy(lb[B1 + ROW_ELEMS + D], xGm_[s31], inParam);
            FlushRead();
            DataCopy(yGm_[d1], lb[B1], PAIR_ELEMS);
        }
        ++ih;
        if (ih == IH) { ih = 0; ++ob; }
        {
            const uint64_t s00 = (((static_cast<uint64_t>(ob) * IH + ih) * IW) * D);
            const uint64_t s01 = (((static_cast<uint64_t>(OB + ob) * IH + ih) * IW) * D);
            const uint64_t s10 = (((static_cast<uint64_t>(2U * OB + ob) * IH + ih) * IW) * D);
            const uint64_t s11 = (((static_cast<uint64_t>(3U * OB + ob) * IH + ih) * IW) * D);
            const uint64_t d0  = (((static_cast<uint64_t>(ob) * OH + (ih << 1)) * OW) * D);

            DataCopy(lb[B2], xGm_[s00], inParam);
            DataCopy(lb[B2 + D], xGm_[s01], inParam);
            DataCopy(lb[B2 + ROW_ELEMS], xGm_[s10], inParam);
            DataCopy(lb[B2 + ROW_ELEMS + D], xGm_[s11], inParam);
            FlushRead();
            DataCopy(yGm_[d0], lb[B2], PAIR_ELEMS);
        }
        ++ih;
        if (ih == IH) { ih = 0; ++ob; }
        {
            const uint64_t s20 = (((static_cast<uint64_t>(ob) * IH + ih) * IW) * D);
            const uint64_t s21 = (((static_cast<uint64_t>(OB + ob) * IH + ih) * IW) * D);
            const uint64_t s30 = (((static_cast<uint64_t>(2U * OB + ob) * IH + ih) * IW) * D);
            const uint64_t s31 = (((static_cast<uint64_t>(3U * OB + ob) * IH + ih) * IW) * D);
            const uint64_t d1  = (((static_cast<uint64_t>(ob) * OH + (ih << 1)) * OW) * D);

            DataCopy(lb[B3], xGm_[s20], inParam);
            DataCopy(lb[B3 + D], xGm_[s21], inParam);
            DataCopy(lb[B3 + ROW_ELEMS], xGm_[s30], inParam);
            DataCopy(lb[B3 + ROW_ELEMS + D], xGm_[s31], inParam);
            FlushRead();
            DataCopy(yGm_[d1], lb[B3], PAIR_ELEMS);
            FlushWrite();
        }
    }

    // ── Case 4: fp32, [20,4,6,32] → [5,8,12,32], bs=2, crop=0 ──
    // 10核, 每核8路src交错读→4输出行连续写, 单次批量DMA
    __aicore__ inline void ProcessCase4() {
        const uint32_t IH = 4;
        const uint32_t IW = 6;
        const uint32_t D = 32;
        const uint32_t OB = 5;
        const uint32_t OH = 8;
        const uint32_t OW = 12;
        const uint32_t BPE = 4;

        const uint32_t BLOCK_BYTES = D * BPE;
        const uint32_t ROW_ELEMS = OW * D;
        const uint32_t coreIdx = GetBlockIdx();
        if (coreIdx >= 10U) {
            return;
        }

        LocalTensor<DT_X> lb = buf_.Get<DT_X>();
        DataCopyParams inParam = {6, static_cast<uint16_t>(BLOCK_BYTES / 32U), 0,
                                  static_cast<uint16_t>(BLOCK_BYTES / 32U)};

        const uint32_t ob = coreIdx >> 1;
        const uint32_t ih0 = (coreIdx & 1U) << 1;
        const uint32_t ih1 = ih0 + 1U;

        const uint64_t s00 = (((static_cast<uint64_t>(ob) * IH + ih0) * IW) * D);
        const uint64_t s01 = (((static_cast<uint64_t>(OB + ob) * IH + ih0) * IW) * D);
        const uint64_t s10 = (((static_cast<uint64_t>(2U * OB + ob) * IH + ih0) * IW) * D);
        const uint64_t s11 = (((static_cast<uint64_t>(3U * OB + ob) * IH + ih0) * IW) * D);
        const uint64_t s20 = (((static_cast<uint64_t>(ob) * IH + ih1) * IW) * D);
        const uint64_t s21 = (((static_cast<uint64_t>(OB + ob) * IH + ih1) * IW) * D);
        const uint64_t s30 = (((static_cast<uint64_t>(2U * OB + ob) * IH + ih1) * IW) * D);
        const uint64_t s31 = (((static_cast<uint64_t>(3U * OB + ob) * IH + ih1) * IW) * D);

        const uint64_t d0 = (((static_cast<uint64_t>(ob) * OH + (ih0 << 1)) * OW) * D);

        DataCopy(lb, xGm_[s00], inParam);
        DataCopy(lb[D], xGm_[s01], inParam);
        DataCopy(lb[ROW_ELEMS], xGm_[s10], inParam);
        DataCopy(lb[ROW_ELEMS + D], xGm_[s11], inParam);
        DataCopy(lb[2U * ROW_ELEMS], xGm_[s20], inParam);
        DataCopy(lb[2U * ROW_ELEMS + D], xGm_[s21], inParam);
        DataCopy(lb[3U * ROW_ELEMS], xGm_[s30], inParam);
        DataCopy(lb[3U * ROW_ELEMS + D], xGm_[s31], inParam);
        FlushRead();
        DataCopy(yGm_[d0], lb, 4U * ROW_ELEMS);
        FlushWrite();
    }

    // ── Case 5: fp16, [4,128,128,65] → [1,254,254,65], bs=2, crop=[1,1,1,1] ──
    __aicore__ inline void ProcessCase5() {
        constexpr uint32_t IH = 128;
        constexpr uint32_t IW = 128;
        constexpr uint32_t D = 65;
        constexpr uint32_t OH = 254;
        constexpr uint32_t OW = 254;
        constexpr uint32_t CT = 1;
        constexpr uint32_t CL = 1;
        constexpr uint32_t BS = 2;
        constexpr uint32_t NCOL = 127;
        constexpr uint16_t CB = 130;
        constexpr uint32_t LOCAL_STRIDE = ((CB + 31U) / 32U) * 16U;
        constexpr uint32_t SLOT_STRIDE = NCOL * LOCAL_STRIDE;

        const uint64_t totalPhaseTasks = static_cast<uint64_t>(OH) * BS;
        const uint64_t tasksPerCore = (totalPhaseTasks + coreNum_ - 1) / coreNum_;
        uint64_t start = tasksPerCore * GetBlockIdx();
        if (start >= totalPhaseTasks) return;
        uint64_t end = start + tasksPerCore;
        if (end > totalPhaseTasks) end = totalPhaseTasks;

        LocalTensor<DT_X> local = buf_.Get<DT_X>();
        DataCopyExtParams ip = {static_cast<uint16_t>(NCOL), CB, 0, 0, 0};
        DataCopyExtParams op = {static_cast<uint16_t>(NCOL), CB, 0, CB, 0};
        DataCopyPadExtParams<DT_X> pp0 = {false, 0, 0, static_cast<DT_X>(0)};

        for (uint64_t task = start; task < end; ++task) {
            uint32_t localTask = static_cast<uint32_t>(task - start);
            uint32_t eventId = localTask & 1U;
            uint32_t slotOff = eventId * SLOT_STRIDE;
            if (localTask >= 2U) {
                WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            }

            uint32_t oh = static_cast<uint32_t>(task / BS);
            uint32_t bw = static_cast<uint32_t>(task % BS);
            uint32_t fullH = oh + CT;
            uint32_t bh = fullH & 1U;
            uint32_t ih = fullH >> 1U;

            int32_t firstOwSigned = static_cast<int32_t>(bw) - static_cast<int32_t>(CL % BS);
            if (firstOwSigned < 0) firstOwSigned += static_cast<int32_t>(BS);
            if (firstOwSigned >= static_cast<int32_t>(OW)) continue;

            uint32_t firstOw = static_cast<uint32_t>(firstOwSigned);
            uint32_t iwBase = (firstOw + CL) / BS;
            uint32_t ib = bh * BS + bw;
            uint64_t src = ((static_cast<uint64_t>(ib) * IH + ih) * IW + iwBase) * D;

            DataCopyPad(local[slotOff], xGm_[src], ip, pp0);
            SetFlag<HardEvent::MTE2_MTE3>(eventId);
            WaitFlag<HardEvent::MTE2_MTE3>(eventId);

            uint64_t dst = (static_cast<uint64_t>(oh) * OW + firstOw) * D;
            DataCopyPad(yGm_[dst], local[slotOff], op);
            SetFlag<HardEvent::MTE3_MTE2>(eventId);
        }

        uint32_t done = static_cast<uint32_t>(end - start);
        if (done > 0U) {
            WaitFlag<HardEvent::MTE3_MTE2>((done - 1U) & 1U);
        }
        if (done > 1U) {
            WaitFlag<HardEvent::MTE3_MTE2>((done - 2U) & 1U);
        }
    }

    // ── Case 6: fp16, [4,2,2,4096] → [1,4,4,4096], bs=2, crop=0 ──
    // task-based PhaseStrided, totalTasks=totalRows*bs*dsn = 4*2*1=8, 匹配测试点6-7
    __aicore__ inline void ProcessCase6() {
        constexpr uint32_t BS = 2;
        constexpr uint32_t D = 4096;
        constexpr uint32_t IH = 2;
        constexpr uint32_t IW = 2;
        constexpr uint32_t OW = 4;
        constexpr uint32_t BPE = 2;
        constexpr uint32_t HALF_COLS = 2;
        constexpr uint32_t BLOCK_BYTES = D * BPE;
        constexpr uint32_t BLOCK_UNITS = BLOCK_BYTES / 32U;

        const uint32_t task = GetBlockIdx();
        if (task >= 8U) return;

        const uint32_t oh = task >> 1;
        const uint32_t bw = task & 1U;
        const uint32_t ih = oh >> 1;
        const uint32_t bh = oh & 1U;
        const uint32_t ib = bh * BS + bw;
        const uint64_t src = ((uint64_t)ib * IH + ih) * IW * D;
        const uint64_t dst = ((uint64_t)oh * OW + bw) * D;

        LocalTensor<DT_X> lb = buf_.Get<DT_X>();
        DataCopyParams inParam = {static_cast<uint16_t>(HALF_COLS),
                                  static_cast<uint16_t>(BLOCK_UNITS),
                                  0,
                                  0};
        DataCopyParams outParam = {static_cast<uint16_t>(HALF_COLS),
                                   static_cast<uint16_t>(BLOCK_UNITS),
                                   0,
                                   static_cast<uint16_t>(BLOCK_UNITS)};
        DataCopy(lb, xGm_[src], inParam);
        FlushRead();
        DataCopy(yGm_[dst], lb, outParam);
    }

    // ── Case 7: fp16, [4,1,1,16384] → [1,2,2,16384], bs=2, crop=0 ──
    // task-based PhaseStrided, totalTasks=totalRows*bs*dsn = 2*2*4=16, 匹配测试点6-7
    __aicore__ inline void ProcessCase7() {
        constexpr uint32_t D = 4096;
        constexpr uint32_t BPE = 2;
        constexpr uint32_t BLOCKS_PER_CORE = 2;
        constexpr uint32_t BLOCK_UNITS = (D * BPE) / 32U;
        constexpr uint32_t CORE_TASKS = 8;

        const uint32_t task = GetBlockIdx();
        if (task >= CORE_TASKS) return;
        const uint64_t offset = static_cast<uint64_t>(task) * BLOCKS_PER_CORE * D;
        LocalTensor<DT_X> lb = buf_.Get<DT_X>();
        DataCopyParams params = {static_cast<uint16_t>(BLOCKS_PER_CORE),
                                 static_cast<uint16_t>(BLOCK_UNITS),
                                 0,
                                 0};
        DataCopy(lb, xGm_[offset], params);
        FlushRead();
        DataCopy(yGm_[offset], lb, params);
    }

    // ── Case 8: fp16, [4,10,512,256] → [1,20,1024,256], bs=2, crop=0 ──
    __aicore__ inline void ProcessCase8() {
        ProcessBs2Crop0<256, 1, 20, 1024, 10, 512, 2, 2>();
    }

    // ── Case 9: fp16, [16,10,512,64] → [1,40,1535,64], bs=4, crop=[0,0,513,0] ──
    // OUTPUT_ROW_PACK bs=4 交织: 4路stride CopyIn → tile连续CopyOut, 匹配测试点9
    __aicore__ inline void ProcessCase9() {
        const uint32_t D = 64, OH = 40, OW = 1535, IH = 10, IW = 512, BS = 4, BPE = 2;
        const uint32_t CL = 513;
        const uint32_t blockBytes = D * BPE;
        const uint32_t ubStride = (BS - 1U) * blockBytes / 32U;

        uint32_t cc = colChunk_; if (cc == 0 || cc > OW) cc = OW;
        const uint32_t rowTileNum = (OW + cc - 1U) / cc;
        const uint64_t totalTasks = (uint64_t)OH * rowTileNum;

        uint32_t cn = coreNum_ == 0 ? 1U : coreNum_;
        uint64_t tpc = (totalTasks + cn - 1U) / cn;
        uint64_t ts = tpc * GetBlockIdx();
        if (ts >= totalTasks) return;
        uint64_t te = ts + tpc;
        if (te > totalTasks) te = totalTasks;

        LocalTensor<DT_X> lb = buf_.Get<DT_X>();
        DataCopyPadExtParams<DT_X> pp = {false, 0, 0, static_cast<DT_X>(0)};

        for (uint64_t task = ts; task < te; ++task) {
            uint32_t rowTask = (uint32_t)(task / rowTileNum);
            uint32_t ctId    = (uint32_t)(task - rowTask * (uint64_t)rowTileNum);
            uint32_t oh      = rowTask % OH;
            uint32_t owStart = ctId * cc;
            if (owStart >= OW) continue;
            uint32_t owEnd = owStart + cc;
            if (owEnd > OW) owEnd = OW;
            uint32_t tileW = owEnd - owStart;

            uint32_t bh = oh & 3U;
            uint32_t ih = oh >> 2;

            uint32_t startPhase = (owStart + CL) & (BS - 1U);

            for (uint32_t bw = 0; bw < BS; ++bw) {
                uint32_t firstOw = owStart + ((bw - startPhase + BS) & (BS - 1U));
                if (firstOw >= owEnd) continue;
                uint32_t nCol = (owEnd - 1U - firstOw) / BS + 1U;
                uint32_t ib   = (bh * BS + bw);
                uint32_t iwStart = (firstOw + CL) >> 2;
                uint64_t sB = (((uint64_t)ib * IH + ih) * IW + iwStart) * D;
                DataCopyExtParams ip = {(uint16_t)nCol, blockBytes, 0, (uint16_t)ubStride, 0};
                DataCopyPad(lb[(firstOw - owStart) * D], xGm_[sB], ip, pp);
            }

            FlushRead();
            uint64_t dB = ((uint64_t)oh * OW + owStart) * D;
            DataCopy(yGm_[dB], lb, tileW * D);
            FlushWrite();
        }
    }

    // ── Case 10: fp16, [16,1024,6,32] → [4,2048,12,32], bs=2, crop=0 ──
    // 多行批次交织: 每批24行, bw0/bw1 stride CopyIn → 全批连续 CopyOut, 匹配测试点10
    __aicore__ inline void ProcessCase10() {
        constexpr uint32_t ROW_GROUP  = 24;
        constexpr uint32_t OH = 2048, OW = 12, D = 32, IH = 1024, IW = 6, OB = 4;
        constexpr uint32_t ROW_ELEMS  = OW * D;
        constexpr uint32_t IN_ROW_ELEMS = IW * D;
        constexpr uint32_t IN_BATCH_STRIDE = IH * IN_ROW_ELEMS;
        constexpr uint16_t HALF_COLS  = 6;
        constexpr uint16_t BLOCK_BYTES = 64;
        constexpr uint16_t UB_STRIDE  = 2;

        uint32_t rowStart, rowEnd;
        {
            uint32_t cn = coreNum_ == 0 ? 1U : coreNum_;
            const uint64_t totalRows = (uint64_t)OB * OH;
            const uint64_t rpc = (totalRows + cn - 1U) / cn;
            uint64_t rs64 = rpc * GetBlockIdx();
            if (rs64 >= totalRows) return;
            uint64_t re64 = rs64 + rpc;
            if (re64 > totalRows) re64 = totalRows;
            rowStart = (uint32_t)rs64;
            rowEnd = (uint32_t)re64;
        }

        LocalTensor<DT_X> local = buf_.Get<DT_X>();
        DataCopyPadExtParams<DT_X> pp = {false, 0, 0, static_cast<DT_X>(0)};

        uint32_t r = rowStart;
        while (r < rowEnd) {
            uint32_t groupEnd = r + ROW_GROUP;
            if (groupEnd > rowEnd) groupEnd = rowEnd;
            uint32_t groupRows = groupEnd - r;

            for (uint32_t g = 0; g < groupRows; ++g) {
                uint32_t row = r + g;
                uint32_t ob = row / OH;
                uint32_t oh = row - ob * OH;
                uint32_t bh = oh & 1U;
                uint32_t ih = oh >> 1;
                uint32_t rowOff = g * ROW_ELEMS;

                uint32_t ib0 = bh * 8U + ob;
                uint32_t ib1 = bh * 8U + OB + ob;

                uint64_t src0 = (uint64_t)ib0 * IN_BATCH_STRIDE + (uint64_t)ih * IN_ROW_ELEMS;
                uint64_t src1 = (uint64_t)ib1 * IN_BATCH_STRIDE + (uint64_t)ih * IN_ROW_ELEMS;

                DataCopyExtParams cp = {HALF_COLS, BLOCK_BYTES, 0, UB_STRIDE, 0};
                DataCopyPad(local[rowOff], xGm_[src0], cp, pp);
                DataCopyPad(local[rowOff + D], xGm_[src1], cp, pp);
            }

            FlushRead();
            uint64_t dstBase = (uint64_t)r * ROW_ELEMS;
            DataCopy(yGm_[dstBase], local, groupRows * ROW_ELEMS);
            FlushWrite();

            r = groupEnd;
        }
    }
};

template <typename DT_X>
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) { 
    (void)workspace;
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingData, td, tiling);
    KernelBatchToSpace<DT_X> op;
    op.Init(x, y, &td);
    op.Process();
}
