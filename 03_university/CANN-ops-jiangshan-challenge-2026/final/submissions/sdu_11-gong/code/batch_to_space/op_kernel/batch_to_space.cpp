#include "kernel_operator.h"
#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

using namespace AscendC;

constexpr uint32_t kUbReserveBytes = 1024;
constexpr uint32_t kMinUbTileOutputWidth = 1;
constexpr uint32_t kAlign32 = 32;
constexpr uint32_t kCacheLineSize = 64;
constexpr uint32_t kMinRowBufOutputElems = 32;

template <class DT_X>
class KernelBatchToSpace {
private:
    const BatchToSpaceTilingData *m_tiling = nullptr;
    TPipe m_pipe;
    TBuf<TPosition::VECCALC> m_srcBuf;
    TBuf<TPosition::VECOUT> m_dstBuf;
    TBuf<TPosition::VECCALC> m_rowBuf;
    GlobalTensor<DT_X> m_xGm;
    GlobalTensor<DT_X> m_yGm;

    uint32_t m_alignElems = 0;
    uint32_t m_depth = 0;
    uint32_t m_bs = 0;
    uint32_t m_obMax = 0;
    uint32_t m_iH = 0;
    uint32_t m_iW = 0;
    uint32_t m_oH = 0;
    uint32_t m_oW = 0;
    uint32_t m_ct = 0;
    uint32_t m_cl = 0;
    uint32_t m_ubTileW = 0;

    uint32_t m_rowElems = 0;
    uint32_t m_rowBytes = 0;
    uint32_t m_alignRows = 1;
    uint64_t m_startRow = 0;
    uint64_t m_endRow = 0;
    uint64_t m_rowWidth = 0;
    uint64_t m_bStride = 0;
    uint64_t m_inputBatchStride = 0;
    uint64_t m_totalInputElements = 0;

    bool m_isDepthAligned = false;
    bool m_useUbPath = false;
    bool m_hasRowBuf = false;
    bool m_useVecGather = false;
    bool m_vecGatherUniform = false;
    bool m_useDoubleBuf = false;
    bool m_useScalarPrefetch = false;
    bool m_scalarPrefetchUniform = false;
    bool m_useScalarGroupPrefetch = false;
    bool m_useBatchWrite = false;

    uint32_t m_batchWriteRows = 0;
    uint32_t m_batchWriteElems = 0;

    uint32_t m_prefetchAlignedCount = 0;
    bool m_useStridedDma = false;
    bool m_useBulkStridedDma = false;
    uint32_t m_stridedBufElems = 0;
    uint32_t m_stridedSlotElems = 0;
    uint32_t m_bulkStridedTotalSlots = 0;

    uint32_t m_stridedBufHalfElems = 0;
    uint32_t m_rowBufHalfElems = 0;
    uint32_t m_groupBufStride = 0;
    uint32_t m_groupRowCounts[32] = {0};

    uint32_t m_ubDirectComboCount = 0;
    uint32_t m_ubDirectSingleB1 = 0;
    uint32_t m_ubDirectFirstOh = 0;

    TBuf<TPosition::VECCALC> m_loadBuf;
    TBuf<TPosition::VECCALC> m_gatherOffsets;
    TBuf<TPosition::VECCALC> m_batchBuf;

public:
    __aicore__ inline KernelBatchToSpace() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR ws, const BatchToSpaceTilingData *tilingData) {
        if (tilingData == nullptr) { this->m_startRow = this->m_endRow = 0; return; }

        // ── Phase 1: minimum fields for routing decision ──
        uint32_t coreId = GetBlockIdx();
        uint32_t coreNum = tilingData->coreNum == 0 ? 1 : tilingData->coreNum;
        uint32_t depth = tilingData->depth;
        uint32_t bs = tilingData->blockSize;
        uint64_t totalRows = static_cast<uint64_t>(tilingData->outBatch) * tilingData->outHeight;
        uint64_t sRow = (totalRows * coreId) / coreNum;
        uint64_t eRow = (totalRows * (coreId + 1)) / coreNum;

        // TinyPath check — early return avoids 25+ unnecessary member writes
        // and 64-bit stride chain for the simplest path.
        constexpr uint32_t kAlignElems = kAlign32 / sizeof(DT_X);
        if (tilingData->totalOutputElements < 512 &&
            tilingData->ubTileW >= kMinUbTileOutputWidth && (depth % kAlignElems == 0)) {
            // Set only the members ProcessTiny actually needs (16 vs 40+)
            this->m_tiling = tilingData;
            this->m_depth = depth; this->m_bs = bs;
            this->m_obMax = tilingData->outBatch;
            this->m_iH = tilingData->inputHeight; this->m_iW = tilingData->inputWidth;
            this->m_oH = tilingData->outHeight; this->m_oW = tilingData->outWidth;
            this->m_ct = tilingData->cropTop; this->m_cl = tilingData->cropLeft;
            this->m_rowElems = tilingData->rowElems;
            this->m_startRow = sRow; this->m_endRow = eRow;
            (void)ws;
            this->m_xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
            this->m_yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
            if (this->m_startRow >= this->m_endRow) return;
            this->m_bStride = static_cast<uint64_t>(this->m_obMax) * this->m_iH * this->m_iW * this->m_depth;
            return;
        }

        // ── Phase 2: full Init for non-Tiny paths ──
        this->m_tiling = tilingData;
        this->m_alignElems = kAlignElems;
        this->m_depth = depth;
        this->m_bs = bs;
        this->m_obMax = tilingData->outBatch;
        this->m_iH = tilingData->inputHeight;
        this->m_iW = tilingData->inputWidth;
        this->m_oH = tilingData->outHeight;
        this->m_oW = tilingData->outWidth;
        this->m_ct = tilingData->cropTop;
        this->m_cl = tilingData->cropLeft;

        this->m_rowElems = this->m_tiling->rowElems;
        this->m_rowWidth = static_cast<uint64_t>(this->m_oW) * this->m_depth;
        this->m_rowBytes = this->m_tiling->rowBytes;

        this->m_alignRows = this->m_tiling->alignRows;
        if (this->m_alignRows > 1) {
            uint64_t perCore = (totalRows + coreNum - 1) / coreNum;
            uint64_t alignedPerCore = ((perCore + this->m_alignRows - 1) / this->m_alignRows) * this->m_alignRows;
            this->m_startRow = coreId * alignedPerCore;
            this->m_endRow = (coreId + 1) * alignedPerCore;
            if (this->m_endRow > totalRows) this->m_endRow = totalRows;
            if (this->m_startRow >= this->m_endRow) this->m_startRow = this->m_endRow = 0;
        } else {
            this->m_startRow = (totalRows * coreId) / coreNum;
            this->m_endRow = (totalRows * (coreId + 1)) / coreNum;
        }

        (void)ws;
        this->m_xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        this->m_yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

        if (this->m_startRow >= this->m_endRow) return;

        this->m_bStride = static_cast<uint64_t>(this->m_obMax) * this->m_iH * this->m_iW * this->m_depth;
        this->m_inputBatchStride = this->m_bStride * this->m_bs;
        this->m_totalInputElements = this->m_inputBatchStride * this->m_bs;

        this->m_useUbPath = (this->m_tiling->ubTileW >= kMinUbTileOutputWidth)
            && (this->m_depth % this->m_alignElems == 0);
        this->m_ubTileW = this->m_tiling->ubTileW;
        this->m_isDepthAligned = (this->m_tiling->isDepthAligned != 0);

        uint64_t availUb = this->m_tiling->ubSizeBytes > kUbReserveBytes
            ? this->m_tiling->ubSizeBytes - kUbReserveBytes : 0;

        if (this->m_useUbPath) {
            uint32_t maxSpanW = (this->m_ubTileW + this->m_bs - 1) / this->m_bs + 1;
            uint32_t tileInputW = maxSpanW * this->m_depth;
            uint32_t tileOutputW = this->m_ubTileW * this->m_depth;
            uint32_t alignedInput = this->AlignUp(tileInputW, this->m_alignElems);
            uint32_t srcBufBytes = static_cast<uint32_t>(this->m_bs) * alignedInput * sizeof(DT_X);
            uint32_t dstBufBytes = this->AlignUp(tileOutputW, this->m_alignElems) * sizeof(DT_X);
            uint32_t ubTilesUsed = srcBufBytes + dstBufBytes;

            // Use host pre-computed estimate when available, else fallback estimate
            uint32_t estInputElems = this->m_tiling->ubTotalInEst;
            if (estInputElems == 0) {
                uint64_t rowsPerCore = this->m_endRow - this->m_startRow;
                estInputElems = static_cast<uint32_t>(
                    rowsPerCore * static_cast<uint64_t>(this->m_iW) * this->m_depth);
            }
            uint32_t ubDirectNeed = this->AlignUp(estInputElems * sizeof(DT_X), kAlign32);
            uint32_t uRem = this->AlignUp(static_cast<uint32_t>(
                availUb > ubTilesUsed + kUbReserveBytes ? availUb - ubTilesUsed - kUbReserveBytes : 0), kAlign32);

            if (ubDirectNeed > 0 && ubDirectNeed <= uRem) {
                this->m_pipe.InitBuffer(this->m_loadBuf, ubDirectNeed);
                this->m_prefetchAlignedCount = 0xFFFFFFFFu;
                // Count combos for UbDirect routing
                {
                    uint32_t uOb0 = static_cast<uint32_t>(this->m_startRow / this->m_oH);
                    uint32_t uOb1 = static_cast<uint32_t>((this->m_endRow - 1) / this->m_oH);
                    uint32_t uOh0 = static_cast<uint32_t>(this->m_startRow % this->m_oH);
                    uint32_t uOh1 = static_cast<uint32_t>((this->m_endRow - 1) % this->m_oH);
                    uint32_t uClMod = this->m_cl % this->m_bs;
                    uint32_t cCnt = 0;
                    for (uint32_t ob = uOb0; ob <= uOb1 && cCnt <= 2; ob++) {
                        uint32_t ohS = (ob == uOb0) ? uOh0 : 0;
                        uint32_t ohE = (ob == uOb1) ? uOh1 : this->m_oH - 1;
                        for (uint32_t b0 = 0; b0 < this->m_bs && cCnt <= 2; b0++) {
                            uint32_t fm = (ohS + this->m_ct) % this->m_bs;
                            uint32_t ohF = ohS + ((b0 - fm + this->m_bs) % this->m_bs);
                            if (ohF > ohE) continue;
                            for (uint32_t b1 = 0; b1 < this->m_bs && cCnt <= 2; b1++) {
                                uint32_t fow = (b1 + this->m_bs - uClMod) % this->m_bs;
                                if (fow >= this->m_oW) continue;
                                if (cCnt == 0) {
                                    this->m_ubDirectSingleB1 = b1;
                                    this->m_ubDirectFirstOh = ohF;
                                }
                                cCnt++;
                            }
                        }
                    }
                    this->m_ubDirectComboCount = cCnt;
                }
            } else if (this->m_tiling->ubTotalInEst > 0) {
                // Host estimate failed, try exact per-core enumeration
                this->m_pipe.InitBuffer(this->m_srcBuf, srcBufBytes);
                this->m_pipe.InitBuffer(this->m_dstBuf, dstBufBytes);
                uint32_t uClMod = this->m_cl % this->m_bs;
                uint32_t uFirstOb = static_cast<uint32_t>(this->m_startRow / this->m_oH);
                uint32_t uLastOb = static_cast<uint32_t>((this->m_endRow - 1) / this->m_oH);
                uint32_t uFirstOh = static_cast<uint32_t>(this->m_startRow % this->m_oH);
                uint32_t uLastOh = static_cast<uint32_t>((this->m_endRow - 1) % this->m_oH);
                uint64_t uTotalIn = 0;
                uint32_t uIwd = this->m_iW * this->m_depth;
                uint32_t cCnt = 0;
                for (uint32_t ob = uFirstOb; ob <= uLastOb; ob++) {
                    uint32_t ohS = (ob == uFirstOb) ? uFirstOh : 0, ohE = (ob == uLastOb) ? uLastOh : this->m_oH - 1;
                    for (uint32_t b0 = 0; b0 < this->m_bs; b0++) {
                        uint32_t uFirstM = (ohS + this->m_ct) % this->m_bs;
                        uint32_t uOhFirst = ohS + ((b0 - uFirstM + this->m_bs) % this->m_bs);
                        if (uOhFirst > ohE) continue;
                        uint32_t uLastM = (ohE + this->m_ct) % this->m_bs;
                        uint32_t uOhLast = ohE - ((uLastM - b0 + this->m_bs) % this->m_bs);
                        for (uint32_t b1 = 0; b1 < this->m_bs; b1++) {
                            if (((b1 + this->m_bs - uClMod) % this->m_bs) >= this->m_oW) continue;
                            if (cCnt == 0) {
                                this->m_ubDirectSingleB1 = b1;
                                this->m_ubDirectFirstOh = uOhFirst;
                            }
                            cCnt++;
                            uTotalIn += static_cast<uint64_t>(((uOhLast + this->m_ct) / this->m_bs) - ((uOhFirst + this->m_ct) / this->m_bs) + 1) * uIwd;
                        }
                    }
                }
                uint32_t uNeed = this->AlignUp(static_cast<uint32_t>(uTotalIn * sizeof(DT_X)), kAlign32);
                if (uNeed > 0 && uNeed <= uRem) {
                    this->m_pipe.InitBuffer(this->m_loadBuf, uNeed);
                    this->m_prefetchAlignedCount = 0xFFFFFFFFu;
                    this->m_ubDirectComboCount = cCnt;
                }
            } else {
                this->m_pipe.InitBuffer(this->m_srcBuf, srcBufBytes);
                this->m_pipe.InitBuffer(this->m_dstBuf, dstBufBytes);
            }
        }

        uint32_t rowBufBytes = 0;
        if (false) {
            uint32_t alignedRowBytes = this->AlignUp(this->m_rowBytes, kAlign32);
            rowBufBytes = alignedRowBytes + kAlign32;
            uint32_t rowBufHalfElems = rowBufBytes / sizeof(DT_X);
            if (2 * rowBufBytes <= availUb) {
                this->m_pipe.InitBuffer(this->m_rowBuf, 2 * rowBufBytes);
                this->m_hasRowBuf = true;
                this->m_useDoubleBuf = true;
                this->m_rowBufHalfElems = rowBufHalfElems;
            } else if (rowBufBytes <= availUb) {
                this->m_pipe.InitBuffer(this->m_rowBuf, rowBufBytes);
                this->m_hasRowBuf = true;
                this->m_useDoubleBuf = false;
                this->m_rowBufHalfElems = rowBufHalfElems;
            }
        }

        if (false) {
            bool canVecGather = false;
            bool uniform = (this->m_cl % this->m_bs == 0) && (this->m_oW % this->m_bs == 0);
            if (uniform) {
                uint32_t pbb = this->m_rowBytes / this->m_bs;
                canVecGather = (pbb > 0) && (pbb % kAlign32 == 0);
            } else {
                canVecGather = true;
            }
            if (canVecGather) {
                uint32_t offsetBytes = this->m_rowElems * sizeof(uint32_t);
                uint32_t loadBytes = this->m_rowBytes;
                uint32_t actualRowBufBytes = this->m_useDoubleBuf ? 2 * rowBufBytes : rowBufBytes;
                if (loadBytes + offsetBytes + actualRowBufBytes <= availUb) {
                    this->m_pipe.InitBuffer(this->m_loadBuf, this->AlignUp(loadBytes, kAlign32));
                    this->m_pipe.InitBuffer(this->m_gatherOffsets, this->AlignUp(offsetBytes, kAlign32));
                    this->m_useVecGather = true;
                    this->m_vecGatherUniform = uniform;
                }
            }
        }

        if (this->m_useVecGather) {
            uint32_t clMod = this->m_cl % this->m_bs;
            uint32_t batchStartElems[32];
            uint32_t batchNumBlks[32];
            uint32_t pos = 0;
            for (uint32_t b1 = 0; b1 < this->m_bs; b1++) {
                uint32_t firstOw = (b1 + this->m_bs - clMod) % this->m_bs;
                if (firstOw >= this->m_oW) { batchNumBlks[b1] = 0; batchStartElems[b1] = pos; continue; }
                uint32_t nb = (this->m_oW - firstOw + this->m_bs - 1) / this->m_bs;
                batchNumBlks[b1] = nb;
                batchStartElems[b1] = pos;
                pos += nb * this->m_depth;
            }
            uint32_t elemSize = sizeof(DT_X);
            LocalTensor<uint32_t> offs = this->m_gatherOffsets.template Get<uint32_t>();
            for (uint32_t i = 0; i < this->m_rowElems; i++) {
                uint32_t ow = i / this->m_depth;
                uint32_t c = i % this->m_depth;
                uint32_t b = (ow + clMod) % this->m_bs;
                uint32_t blk = (ow - ((b + this->m_bs - clMod) % this->m_bs)) / this->m_bs;
                offs.SetValue(i, (batchStartElems[b] + blk * this->m_depth + c) * elemSize);
            }
        }

        // StridedDma: per-block DataCopyPad through 32B-aligned UB slots.
        // Each block (depth elems) occupies one aligned slot so both the GM->UB
        // load dst and UB->GM store src are 32B-aligned (required by DataCopyPad).
        // Preempts all scalar-heavy paths (works for any depth alignment / row size).
        if (!this->m_useUbPath && !this->m_useVecGather && this->m_startRow < this->m_endRow && this->m_depth > 0) {
            uint32_t clMod = this->m_cl % this->m_bs;
            // StridedDma needs enough blocks per row to amortize per-block overhead
            // 32 blocks * 3 elems/block = 96 rowElems minimum
            constexpr uint32_t kMinStridedBlocks = 1;
            uint32_t maxNumBlocks = 0;
            // Skip StridedDma for small shapes — BatchWrite is faster
            for (uint32_t b1 = 0; b1 < this->m_bs; b1++) {
                uint32_t firstOw = (b1 + this->m_bs - clMod) % this->m_bs;
                if (firstOw >= this->m_oW) continue;
                uint32_t nb = (this->m_oW - firstOw + this->m_bs - 1) / this->m_bs;
                if (nb > maxNumBlocks) maxNumBlocks = nb;
            }
            uint32_t slotElems = this->AlignUp(this->m_depth, this->m_alignElems);
            uint32_t availElems = static_cast<uint32_t>(availUb / sizeof(DT_X));
            if (maxNumBlocks >= kMinStridedBlocks && maxNumBlocks > 0 && slotElems > 0 && availElems >= slotElems) {
                // ── StridedDma buffer sizing ──────────────────────────
                // Strategy:
                // 1. Full bulk: all (row,b1) blocks fit UB → canBulk=true, 2 barriers
                // 2. Minimum: one b1's blocks → minBuf (maxNumBlocks * slotElems)

                // Step A: minimum needed for one (row,b1)
                uint64_t minDesired = static_cast<uint64_t>(maxNumBlocks) * slotElems;
                uint32_t minBuf = minDesired < availElems ? static_cast<uint32_t>(minDesired) : availElems;
                minBuf = (minBuf / slotElems) * slotElems;
                if (minBuf < slotElems) minBuf = slotElems;

                // Step B: compute per-core block count for bulk check
                uint32_t blocksPerRow = 0;
                for (uint32_t b1 = 0; b1 < this->m_bs; b1++) {
                    uint32_t fow = (b1 + this->m_bs - clMod) % this->m_bs;
                    if (fow >= this->m_oW) continue;
                    blocksPerRow += (this->m_oW - fow + this->m_bs - 1) / this->m_bs;
                }
                uint64_t rowsPerCore = this->m_endRow - this->m_startRow;
                uint32_t estTotalBlocks = static_cast<uint32_t>(rowsPerCore * blocksPerRow);
                // Tight packing for Bulk: depth stride avoids DataCopyPad 32B waste
                // Bulk stride = aligned slot for 32B-safe DataCopyPad P2 store.
                //   ceil(depth*elemSize/32)*32/elemSize
                uint32_t bulkStride = ((depth * sizeof(DT_X) + 31) / 32) * (32 / sizeof(DT_X));
                uint32_t bulkBuf = estTotalBlocks * bulkStride;

                // Step C: determine final buffer size
                //   canBulk  — all rows fit in UB → ProcessBulkStridedDma (2 barriers)
                //   else     — try multi-row buffer → fewer ProcessStridedDma iterations
                bool canBulk = false;
                uint32_t bufElems = minBuf;
                if (estTotalBlocks > 0) {
                    if (static_cast<uint64_t>(bulkBuf) * sizeof(DT_X) <= availUb) {
                        // ── Full bulk fits → enable canBulk ──────────────
                        canBulk = true;
                        bufElems = bulkBuf;
                    }
                    // else: keep minBuf (multi-row fallback removed: 75% UB
                    // allocation caused InitBuffer failure → silent fallback
                    // to ProcessScalar with multi-core correctness bug).
                }

                // Step D: double-buffer toggle (bulk doesn't need it)
                uint32_t dbBufElems = bufElems;
                bool useDoubleBuf = false;
                if (!canBulk && static_cast<uint64_t>(bufElems * 2) * sizeof(DT_X) <= availUb) {
                    useDoubleBuf = true;
                    dbBufElems = bufElems * 2;
                }

                // Step E: allocate and commit
                if (static_cast<uint64_t>(dbBufElems) * sizeof(DT_X) <= availUb) {
                    this->m_pipe.InitBuffer(this->m_loadBuf, dbBufElems * sizeof(DT_X));
                    this->m_useStridedDma = true;
                    this->m_useBulkStridedDma = canBulk;
                    this->m_stridedBufElems = dbBufElems;
                    this->m_stridedBufHalfElems = bufElems;
                    this->m_stridedSlotElems = slotElems;
                    if (canBulk) this->m_bulkStridedTotalSlots = estTotalBlocks;
                }

        // DEBUG: trace canBulk decision (uncomment for OJ diagnosis)
        // printf("DBG[%u] StridedDma blkPerRow=%u rowsPerCore=%lu estTotalBlk=%u "
        //        "bulkBuf=%u availUb=%lu slotElems=%u canBulk=%d\n",
        //        GetBlockIdx(), blocksPerRow, rowsPerCore, estTotalBlocks,
        //        static_cast<uint32_t>(bulkBuf * sizeof(DT_X)),
        //        availUb, slotElems, canBulk);
            }
        }

        // BatchWrite: accumulate multiple rows then DataCopy (for misaligned rowBytes)
        if (!this->m_useUbPath && !this->m_useVecGather && !this->m_useStridedDma && !this->m_hasRowBuf && this->m_rowElems >= 4) {
            this->m_batchWriteRows = this->m_tiling->batchWriteRows;
            if (this->m_batchWriteRows > 1) {
                uint32_t batchElems = this->m_batchWriteRows * this->m_rowElems;
                uint32_t batchBytes = this->AlignUp(batchElems * sizeof(DT_X), kAlign32);
                if (batchBytes <= availUb) {
                    this->m_pipe.InitBuffer(this->m_batchBuf, batchBytes);
                    this->m_useBatchWrite = true;
                    this->m_batchWriteElems = batchBytes / sizeof(DT_X);
                }
            }
        }

        // ScalarGroupPrefetch: load all rows of a b0 group into UB, then write back
        if (false) {
            // Fast estimation: skip the row traversal loop when the worst-case
            // group element count is below the threshold, avoiding O(rowsPerCore)
            // overhead for small shapes where GroupPrefetch would never activate.
            uint64_t rowsPerCore = this->m_endRow - this->m_startRow;
            if (rowsPerCore * this->m_rowElems >= 65536) {
                this->m_groupBufStride = this->AlignUp(this->m_rowElems, this->m_alignElems);
                for (uint32_t i = 0; i < this->m_bs; i++) this->m_groupRowCounts[i] = 0;
                for (uint64_t row = this->m_startRow; row < this->m_endRow; row++) {
                    uint32_t oh = static_cast<uint32_t>(row % this->m_oH);
                    uint32_t fullOh = oh + this->m_ct;
                    uint32_t b0 = fullOh % this->m_bs;
                    this->m_groupRowCounts[b0]++;
                }
                uint32_t maxGroupRows = 0;
                for (uint32_t i = 0; i < this->m_bs; i++) {
                    if (this->m_groupRowCounts[i] > maxGroupRows)
                        maxGroupRows = this->m_groupRowCounts[i];
                }
                if (maxGroupRows > 0) {
                    uint64_t groupTotalElems = static_cast<uint64_t>(maxGroupRows) * this->m_rowElems;
                    uint32_t maxBatchElems = 0;
                    uint32_t clMod = this->m_cl % this->m_bs;
                    for (uint32_t b1 = 0; b1 < this->m_bs; b1++) {
                        uint32_t firstOw = (b1 + this->m_bs - clMod) % this->m_bs;
                        if (firstOw >= this->m_oW) continue;
                        uint32_t nb = (this->m_oW - firstOw + this->m_bs - 1) / this->m_bs;
                        uint32_t be = nb * this->m_depth;
                        if (be > maxBatchElems) maxBatchElems = be;
                    }
                    uint32_t groupBufBytes = (maxGroupRows * this->m_groupBufStride + maxBatchElems) * sizeof(DT_X);
                    if (groupBufBytes <= availUb && groupTotalElems >= 65536) {
                        this->m_pipe.InitBuffer(this->m_loadBuf, groupBufBytes);
                        this->m_useScalarGroupPrefetch = true;
                    }
                }
            }
        }

        // ScalarPrefetch (fallback if Group not viable)
        if (false) {
            this->m_scalarPrefetchUniform = (this->m_cl % this->m_bs == 0) && (this->m_oW % this->m_bs == 0);
            uint32_t clMod = this->m_cl % this->m_bs;
            uint32_t totalAligned = 0;
            uint32_t perBatchElems = 0;
            bool allSameSize = true;
            for (uint32_t b1 = 0; b1 < this->m_bs; b1++) {
                uint32_t firstOw = (b1 + this->m_bs - clMod) % this->m_bs;
                if (firstOw >= this->m_oW) continue;
                uint32_t nb = (this->m_oW - firstOw + this->m_bs - 1) / this->m_bs;
                uint32_t batchElems = nb * this->m_depth;
                if (b1 == 0) perBatchElems = batchElems;
                else if (batchElems != perBatchElems) allSameSize = false;
                totalAligned += this->AlignUp(batchElems, this->m_alignElems);
            }
            this->m_scalarPrefetchUniform = this->m_scalarPrefetchUniform && allSameSize;
            uint32_t loadBufBytes = totalAligned * sizeof(DT_X);
            if (loadBufBytes > 0 && loadBufBytes <= availUb) {
                this->m_pipe.InitBuffer(this->m_loadBuf, loadBufBytes);
                this->m_useScalarPrefetch = true;
            }
        }
    }

    __aicore__ inline void Process() {
        if (this->m_tiling == nullptr || this->m_tiling->totalOutputElements == 0 ||
            this->m_depth == 0 || this->m_oW == 0 || this->m_bs == 0 || this->m_startRow >= this->m_endRow)
            return;

        if (this->m_useUbPath) {
            if (this->m_prefetchAlignedCount == 0xFFFFFFFFu) {
                if (this->m_ubDirectComboCount <= 1) {
                    this->ProcessUbDirectSingle();
                } else if (this->m_ubDirectComboCount > 2) {
                    this->ProcessUbDirectPipelined();
                } else {
                    this->ProcessUbDirect();
                }
            } else {
                this->ProcessUbTiles();
            }
        } else if (false) {
            this->ProcessVecGather();
        } else if (this->m_useStridedDma) {
            if (this->m_useBulkStridedDma) {
                this->ProcessBulkStridedDma();
            } else {
                this->ProcessStridedDma();
            }
        } else if (false) {
            this->ProcessScalarRowBuf();
        } else if (this->m_useBatchWrite) {
            this->ProcessBatchWrite();
        } else if (false) {
            this->ProcessScalarGroupPrefetch();
        } else if (false) {
            this->ProcessScalarPrefetch();
        } else {
            this->ProcessScalar();
        }
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t v, uint32_t align) {
        return (v + align - 1) / align * align;
    }

    __aicore__ inline uint32_t ComputeTileWidth(uint64_t availUb, uint32_t elemSize) {
        uint32_t maxSingleBuf = static_cast<uint32_t>(availUb / ((this->m_bs + 1) * elemSize));
        uint32_t cand = maxSingleBuf > this->m_oW ? this->m_oW : maxSingleBuf;
        while (cand > 0) {
            uint32_t spanW = (cand + this->m_bs - 1) / this->m_bs;
            uint32_t inElems = spanW * this->m_depth;
            uint32_t outElems = cand * this->m_depth;
            uint32_t aIn = this->AlignUp(inElems, this->m_alignElems);
            uint64_t need = (static_cast<uint64_t>(this->m_bs) * aIn + this->AlignUp(outElems, this->m_alignElems)) * elemSize;
            if (need <= availUb) return cand;
            cand--;
        }
        return 0;
    }

    __aicore__ inline void ProcessUbTiles() {
        LocalTensor<DT_X> loadBuf = this->m_srcBuf.template Get<DT_X>();
        LocalTensor<DT_X> outBuf = this->m_dstBuf.template Get<DT_X>();

        for (uint64_t row = this->m_startRow; row < this->m_endRow; row++) {
            uint32_t oh = static_cast<uint32_t>(row % this->m_oH);
            uint32_t ob = static_cast<uint32_t>(row / this->m_oH);
            uint32_t fullOh = oh + this->m_ct;
            uint32_t b0 = fullOh % this->m_bs;
            uint32_t ih = fullOh / this->m_bs;

            for (uint32_t twStart = 0; twStart < this->m_oW; twStart += this->m_ubTileW) {
                uint32_t tw = this->m_ubTileW;
                if (twStart + tw > this->m_oW) tw = this->m_oW - twStart;

                uint32_t clMod = this->m_cl % this->m_bs;
                uint32_t iwStart = (twStart + this->m_cl) / this->m_bs;
                uint32_t iwEnd = (twStart + tw - 1 + this->m_cl) / this->m_bs;
                uint32_t spanW = iwEnd - iwStart + 1;

                uint32_t elemSize = sizeof(DT_X);
                uint32_t aIn = this->AlignUp(spanW * this->m_depth, this->m_alignElems);

                // Phase 1: Load (MTE2)
                if ((spanW * this->m_depth) % this->m_alignElems == 0) {
                    uint64_t rowBase = (static_cast<uint64_t>(b0) * this->m_bs * this->m_obMax + ob) * this->m_iH * this->m_iW * this->m_depth
                                     + static_cast<uint64_t>(ih) * this->m_iW * this->m_depth + iwStart * this->m_depth;
                    uint32_t blkBytes = spanW * this->m_depth * elemSize;
                    uint64_t bStrideBytes = this->m_bStride * elemSize;
                    int32_t srcGap = static_cast<int32_t>(bStrideBytes - blkBytes);

                    DataCopyExtParams cp;
                    cp.blockCount = static_cast<uint16_t>(this->m_bs);
                    cp.blockLen = blkBytes;
                    cp.srcStride = srcGap;
                    cp.dstStride = 0;
                    DataCopyPadExtParams<DT_X> pp;
                    DataCopyPad(loadBuf, this->m_xGm[rowBase], cp, pp);
                }

                // Phase 1: MTE2 load done → Phase 2 V can read loadBuf.
                // 实测 3510 AIC 上 PIPE_MTE2 不足以保证 MTE2→V 可见性，需 PIPE_ALL。
                pipe_barrier(PIPE_ALL);

                // Phase 2: Scatter (V pipeline) — batch DataCopy via DataCopyParams.
                // DataCopyParams fields are in 32B C0 blocks (not elements/bytes):
                //   elemPerC0 = kAlign32 / sizeof(DT_X), c0BlkLen = depth / elemPerC0.
                // See kernel_struct_data_copy.h and dav_3510/kernel_operator_data_copy_impl.h:163.
                {
                    uint32_t elemPerC0 = kAlign32 / sizeof(DT_X);
                    uint16_t c0BlkLen = this->m_depth / elemPerC0;
                    for (uint32_t b1 = 0; b1 < this->m_bs; b1++) {
                        uint32_t targetMod = (b1 + this->m_bs - clMod) % this->m_bs;
                        uint32_t twStartMod = twStart % this->m_bs;
                        uint32_t firstDelta = (targetMod + this->m_bs - twStartMod) % this->m_bs;
                        if (firstDelta >= tw) continue;

                        uint32_t nblks = (tw - firstDelta + this->m_bs - 1) / this->m_bs;
                        if (nblks == 0) continue;

                        uint32_t absOw = twStart + firstDelta;
                        uint32_t firstIwRel = (absOw + this->m_cl) / this->m_bs - iwStart;

                        DataCopyParams ubDp;
                        ubDp.blockCount = static_cast<uint16_t>(nblks);
                        ubDp.blockLen   = c0BlkLen;
                        ubDp.srcGap     = 0;
                        ubDp.dstGap     = static_cast<uint16_t>((this->m_bs - 1) * c0BlkLen);
                        DataCopy(outBuf[firstDelta * this->m_depth],
                                 loadBuf[b1 * aIn + firstIwRel * this->m_depth], ubDp);
                    }
                }

                // TQueSync: V→MTE3 (scatter done → MTE3 can read VECOUT)
                {
                    AscendC::TQueSync<PIPE_V, PIPE_MTE3> sync;
                    sync.SetFlag(0);
                    sync.WaitFlag(0);
                }

                // Phase 3: Write (MTE3)
                uint64_t gmOff = (static_cast<uint64_t>(ob) * this->m_oH + oh) * this->m_oW * this->m_depth
                               + static_cast<uint64_t>(twStart) * this->m_depth;
                uint32_t outCount = tw * this->m_depth;
                if (outCount > 0) {
                    DataCopy(this->m_yGm[gmOff], outBuf[0], outCount);
                }
            }
        }
    }

    __aicore__ inline void ProcessScalarRowBuf() {
        uint32_t rowElems = this->m_rowElems;
        uint32_t elemSize = sizeof(DT_X);
        uint32_t clMod = this->m_cl % this->m_bs;
        LocalTensor<DT_X> rowBuf = this->m_rowBuf.template Get<DT_X>();
        uint32_t halfElems = this->m_rowBufHalfElems;

        if (this->m_useDoubleBuf) {
            uint64_t row = this->m_startRow;
            // Row 0 into buf0
            {
                uint32_t oh = static_cast<uint32_t>(row % this->m_oH);
                uint32_t ob = static_cast<uint32_t>(row / this->m_oH);
                uint32_t fullOh = oh + this->m_ct;
                uint32_t b0 = fullOh % this->m_bs;
                uint32_t ih = fullOh / this->m_bs;
                uint64_t rowBase = (static_cast<uint64_t>(b0) * this->m_bs * this->m_obMax + ob) * this->m_iH * this->m_iW * this->m_depth
                                 + static_cast<uint64_t>(ih) * this->m_iW * this->m_depth;
                for (uint32_t b1 = 0; b1 < this->m_bs; b1++) {
                    uint32_t firstOw = (b1 + this->m_bs - clMod) % this->m_bs;
                    if (firstOw >= this->m_oW) continue;
                    uint32_t iw = (this->m_cl + firstOw) / this->m_bs;
                    uint64_t srcBase = rowBase + b1 * this->m_bStride + static_cast<uint64_t>(iw) * this->m_depth;
                    uint32_t numBlocks = (this->m_oW - firstOw + this->m_bs - 1) / this->m_bs;
                    for (uint32_t n = 0; n < numBlocks; n++) {
                        uint32_t dstOff = (firstOw + n * this->m_bs) * this->m_depth;
                        for (uint32_t c = 0; c < this->m_depth; c++)
                            rowBuf.SetValue(dstOff + c, this->m_xGm.GetValue(srcBase + c));
                        srcBase += this->m_depth;
                    }
                }
            }
            pipe_barrier(PIPE_ALL);

            for (row = this->m_startRow + 1; row < this->m_endRow; row++) {
                uint32_t bufIdx = (row - this->m_startRow) & 1;
                uint32_t prevBufIdx = bufIdx ^ 1;
                uint32_t bufOff = bufIdx * halfElems;
                uint32_t prevBufOff = prevBufIdx * halfElems;
                uint64_t prevOutStart = (row - 1) * rowElems;

                DataCopy(this->m_yGm[prevOutStart], rowBuf[prevBufOff], rowElems);

                uint32_t oh = static_cast<uint32_t>(row % this->m_oH);
                uint32_t ob = static_cast<uint32_t>(row / this->m_oH);
                uint32_t fullOh = oh + this->m_ct;
                uint32_t b0 = fullOh % this->m_bs;
                uint32_t ih = fullOh / this->m_bs;
                uint64_t rowBase = (static_cast<uint64_t>(b0) * this->m_bs * this->m_obMax + ob) * this->m_iH * this->m_iW * this->m_depth
                                 + static_cast<uint64_t>(ih) * this->m_iW * this->m_depth;
                for (uint32_t b1 = 0; b1 < this->m_bs; b1++) {
                    uint32_t firstOw = (b1 + this->m_bs - clMod) % this->m_bs;
                    if (firstOw >= this->m_oW) continue;
                    uint32_t iw = (this->m_cl + firstOw) / this->m_bs;
                    uint64_t srcBase = rowBase + b1 * this->m_bStride + static_cast<uint64_t>(iw) * this->m_depth;
                    uint32_t numBlocks = (this->m_oW - firstOw + this->m_bs - 1) / this->m_bs;
                    for (uint32_t n = 0; n < numBlocks; n++) {
                        uint32_t dstOff = bufOff + (firstOw + n * this->m_bs) * this->m_depth;
                        for (uint32_t c = 0; c < this->m_depth; c++)
                            rowBuf.SetValue(dstOff + c, this->m_xGm.GetValue(srcBase + c));
                        srcBase += this->m_depth;
                    }
                }

                pipe_barrier(PIPE_ALL);
            }

            // Last row write-back
            uint32_t lastBufIdx = (this->m_endRow - 1 - this->m_startRow) & 1;
            uint64_t lastOutStart = (this->m_endRow - 1) * rowElems;
            DataCopy(this->m_yGm[lastOutStart], rowBuf[lastBufIdx * halfElems], rowElems);
            pipe_barrier(PIPE_ALL);
        } else {
            for (uint64_t row = this->m_startRow; row < this->m_endRow; row++) {
                uint32_t oh = static_cast<uint32_t>(row % this->m_oH);
                uint32_t ob = static_cast<uint32_t>(row / this->m_oH);
                uint32_t fullOh = oh + this->m_ct;
                uint32_t b0 = fullOh % this->m_bs;
                uint32_t ih = fullOh / this->m_bs;

                uint64_t rowBase = (static_cast<uint64_t>(b0) * this->m_bs * this->m_obMax + ob) * this->m_iH * this->m_iW * this->m_depth
                                 + static_cast<uint64_t>(ih) * this->m_iW * this->m_depth;

                for (uint32_t b1 = 0; b1 < this->m_bs; b1++) {
                    uint32_t firstOw = (b1 + this->m_bs - clMod) % this->m_bs;
                    if (firstOw >= this->m_oW) continue;
                    uint32_t iw = (this->m_cl + firstOw) / this->m_bs;
                    uint64_t srcBase = rowBase + b1 * this->m_bStride + static_cast<uint64_t>(iw) * this->m_depth;

                    uint32_t numBlocks = (this->m_oW - firstOw + this->m_bs - 1) / this->m_bs;
                    for (uint32_t n = 0; n < numBlocks; n++) {
                        uint32_t dstOff = (firstOw + n * this->m_bs) * this->m_depth;
                        for (uint32_t c = 0; c < this->m_depth; c++)
                            rowBuf.SetValue(dstOff + c, this->m_xGm.GetValue(srcBase + c));
                        srcBase += this->m_depth;
                    }
                }

                pipe_barrier(PIPE_ALL);

                uint64_t outStart = row * this->m_oW * this->m_depth;
                DataCopy(this->m_yGm[outStart], rowBuf, rowElems);
                pipe_barrier(PIPE_ALL);
            }
        }
    }

    __aicore__ inline void ProcessVecGather() {
        uint32_t rowElems = this->m_rowElems;
        uint32_t iw = this->m_cl / this->m_bs;
        uint64_t bStrideBytes = this->m_bStride * sizeof(DT_X);
        uint32_t clMod = this->m_cl % this->m_bs;
        uint32_t alignElems = this->m_alignElems;

        LocalTensor<DT_X> loadBuf = this->m_loadBuf.template Get<DT_X>();
        LocalTensor<DT_X> rowBuf = this->m_rowBuf.template Get<DT_X>();
        LocalTensor<uint32_t> offsets = this->m_gatherOffsets.template Get<uint32_t>();

        for (uint64_t row = this->m_startRow; row < this->m_endRow; row++) {
            uint32_t oh = static_cast<uint32_t>(row % this->m_oH);
            uint32_t ob = static_cast<uint32_t>(row / this->m_oH);
            uint32_t fullOh = oh + this->m_ct;
            uint32_t b0 = fullOh % this->m_bs;
            uint32_t ih = fullOh / this->m_bs;

            uint64_t rowBase = (static_cast<uint64_t>(b0) * this->m_bs * this->m_obMax + ob) * this->m_iH * this->m_iW * this->m_depth
                             + static_cast<uint64_t>(ih) * this->m_iW * this->m_depth;

            if (this->m_vecGatherUniform) {
                uint32_t perBatchElems = rowElems / this->m_bs;
                uint32_t batchDataBytes = perBatchElems * sizeof(DT_X);
                DataCopyExtParams cp;
                cp.blockCount = static_cast<uint16_t>(this->m_bs);
                cp.blockLen = batchDataBytes;
                cp.srcStride = static_cast<int32_t>(bStrideBytes - batchDataBytes);
                cp.dstStride = 0;
                DataCopyPadExtParams<DT_X> pp;
                DataCopyPad(loadBuf, this->m_xGm[rowBase + static_cast<uint64_t>(iw) * this->m_depth], cp, pp);
            } else {
                uint32_t loadOff = 0;
                for (uint32_t b1 = 0; b1 < this->m_bs; b1++) {
                    uint32_t firstOw = (b1 + this->m_bs - clMod) % this->m_bs;
                    if (firstOw >= this->m_oW) continue;
                    uint32_t bbIw = (this->m_cl + firstOw) / this->m_bs;
                    uint64_t srcBase = rowBase + b1 * this->m_bStride + static_cast<uint64_t>(bbIw) * this->m_depth;
                    uint32_t numBlocks = (this->m_oW - firstOw + this->m_bs - 1) / this->m_bs;
                    uint32_t count = numBlocks * this->m_depth;
                    if ((srcBase % alignElems == 0) && (count % alignElems == 0)) {
                        DataCopy(loadBuf[loadOff], this->m_xGm[srcBase], count);
                    } else {
                        for (uint32_t n = 0; n < numBlocks; n++) {
                            for (uint32_t c = 0; c < this->m_depth; c++)
                                loadBuf.SetValue(loadOff + c, this->m_xGm.GetValue(srcBase + c));
                            srcBase += this->m_depth;
                            loadOff += this->m_depth;
                        }
                        continue;
                    }
                    loadOff += count;
                }
            }

            pipe_barrier(PIPE_ALL);

            Gather(rowBuf, loadBuf, offsets, 0, rowElems);

            pipe_barrier(PIPE_ALL);

            uint64_t outStart = row * rowElems;
            DataCopy(this->m_yGm[outStart], rowBuf, rowElems);

            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline void ProcessBatchWrite() {
        if (this->m_startRow >= this->m_endRow) return;
        uint32_t rowElems = this->m_rowElems;
        uint32_t clMod = this->m_cl % this->m_bs;
        uint32_t batchWriteRows = this->m_batchWriteRows;
        uint32_t depth = this->m_depth;
        uint32_t bs = this->m_bs;
        uint32_t dstStep = bs * depth;      // pre-computed block stride in output
        LocalTensor<DT_X> batchBuf = this->m_batchBuf.template Get<DT_X>();

        for (uint64_t batchStart = this->m_startRow; batchStart < this->m_endRow; batchStart += batchWriteRows) {
            uint64_t batchEnd = batchStart + batchWriteRows;
            if (batchEnd > this->m_endRow) batchEnd = this->m_endRow;
            uint32_t rowsInBatch = static_cast<uint32_t>(batchEnd - batchStart);

            for (uint32_t ri = 0; ri < rowsInBatch; ri++) {
                uint64_t row = batchStart + ri;
                uint32_t oh = static_cast<uint32_t>(row % this->m_oH);
                uint32_t ob = static_cast<uint32_t>(row / this->m_oH);
                uint32_t fullOh = oh + this->m_ct;
                uint32_t b0 = fullOh % bs;
                uint32_t ih = fullOh / bs;

                uint64_t rowBase = (static_cast<uint64_t>(b0) * bs * this->m_obMax + ob)
                                 * this->m_iH * this->m_iW * depth
                                 + static_cast<uint64_t>(ih) * this->m_iW * depth;
                uint32_t bufOff = ri * rowElems;

                for (uint32_t b1 = 0; b1 < bs; b1++) {
                    uint32_t firstOw = (b1 + bs - clMod) % bs;
                    if (firstOw >= this->m_oW) continue;
                    uint32_t iw = (this->m_cl + firstOw) / bs;
                    uint64_t srcBase = rowBase + b1 * this->m_bStride
                                     + static_cast<uint64_t>(iw) * depth;
                    uint32_t numBlocks = (this->m_oW - firstOw + bs - 1) / bs;

                    if (depth == 3) {
                        // depth=3 manual unrolling: 3 explicit SetValue/GetValue pairs
                        // eliminates inner `for(c)` loop overhead + address recompute
                        uint32_t dOff = bufOff + firstOw * depth;
                        for (uint32_t n = 0; n < numBlocks; n++) {
                            batchBuf.SetValue(dOff,     this->m_xGm.GetValue(srcBase));
                            batchBuf.SetValue(dOff + 1, this->m_xGm.GetValue(srcBase + 1));
                            batchBuf.SetValue(dOff + 2, this->m_xGm.GetValue(srcBase + 2));
                            dOff += dstStep;
                            srcBase += depth;
                        }
                    } else {
                        // generic depth: running dstOff eliminates per-iteration multiply
                        uint32_t dOff = bufOff + firstOw * depth;
                        for (uint32_t n = 0; n < numBlocks; n++) {
                            for (uint32_t c = 0; c < depth; c++)
                                batchBuf.SetValue(dOff + c, this->m_xGm.GetValue(srcBase + c));
                            dOff += dstStep;
                            srcBase += depth;
                        }
                    }
                }
            }

            // TQueSync: PIPE_S→MTE3 cross-pipeline (scalar fill → DataCopy write)
            {
                AscendC::TQueSync<PIPE_S, PIPE_MTE3> sync;
                sync.SetFlag(0);
                sync.WaitFlag(0);
            }

            uint64_t outBase = batchStart * rowElems;
            if (rowsInBatch == batchWriteRows) {
                DataCopy(this->m_yGm[outBase], batchBuf, batchWriteRows * rowElems);
            } else {
                for (uint32_t ri = 0; ri < rowsInBatch; ri++) {
                    uint64_t rowOutBase = (batchStart + ri) * rowElems;
                    uint32_t bufOff = ri * rowElems;
                    for (uint32_t i = 0; i < rowElems; i++)
                        this->m_yGm.SetValue(rowOutBase + i, batchBuf.GetValue(bufOff + i));
                }
            }

            // TQueSync: MTE3→PIPE_S cross-pipeline (store done → next scalar fill, buffer reuse)
            {
                AscendC::TQueSync<PIPE_MTE3, PIPE_S> sync;
                sync.SetFlag(1);
                sync.WaitFlag(1);
            }
        }
    }

    __aicore__ inline void ProcessScalarPrefetch() {
        if (this->m_startRow >= this->m_endRow) return;
        uint32_t elemSize = sizeof(DT_X);
        uint32_t alignElems = this->m_alignElems;
        uint32_t clMod = this->m_cl % this->m_bs;
        LocalTensor<DT_X> loadBuf = this->m_loadBuf.template Get<DT_X>();

        for (uint64_t row = this->m_startRow; row < this->m_endRow; row++) {
            uint32_t oh = static_cast<uint32_t>(row % this->m_oH);
            uint32_t ob = static_cast<uint32_t>(row / this->m_oH);
            uint32_t fullOh = oh + this->m_ct;
            uint32_t b0 = fullOh % this->m_bs;
            uint32_t ih = fullOh / this->m_bs;

            uint64_t rowBase = (static_cast<uint64_t>(b0) * this->m_bs * this->m_obMax + ob) * this->m_iH * this->m_iW * this->m_depth
                             + static_cast<uint64_t>(ih) * this->m_iW * this->m_depth;
            uint64_t outRowBase = (static_cast<uint64_t>(ob) * this->m_oH + oh) * this->m_oW * this->m_depth;

            uint32_t loadOff = 0;
            if (this->m_scalarPrefetchUniform) {
                uint32_t perBatchCount = (this->m_oW / this->m_bs) * this->m_depth;
                uint32_t perBatchBytes = perBatchCount * elemSize;
                uint32_t iw = this->m_cl / this->m_bs;
                uint64_t firstSrcBase = rowBase + static_cast<uint64_t>(iw) * this->m_depth;
                uint64_t bStrideBytes = this->m_bStride * elemSize;
                if ((firstSrcBase % alignElems == 0) && (perBatchCount % alignElems == 0) &&
                    (bStrideBytes % kAlign32 == 0)) {
                    DataCopyExtParams cp;
                    cp.blockCount = static_cast<uint16_t>(this->m_bs);
                    cp.blockLen = perBatchBytes;
                    cp.srcStride = static_cast<int32_t>(bStrideBytes - perBatchBytes);
                    cp.dstStride = 0;
                    DataCopyPadExtParams<DT_X> pp;
                    DataCopyPad(loadBuf, this->m_xGm[firstSrcBase], cp, pp);
                    loadOff = this->AlignUp(perBatchCount, alignElems) * this->m_bs;
                } else {
                    for (uint32_t b1 = 0; b1 < this->m_bs; b1++) {
                        uint64_t srcBase = rowBase + b1 * this->m_bStride + static_cast<uint64_t>(iw) * this->m_depth;
                        uint32_t alignedCount = this->AlignUp(perBatchCount, alignElems);
                        if ((srcBase % alignElems == 0) && (perBatchCount % alignElems == 0)) {
                            DataCopy(loadBuf[loadOff], this->m_xGm[srcBase], perBatchCount);
                        } else {
                            uint32_t alignedPart = (perBatchCount / alignElems) * alignElems;
                            if (alignedPart > 0)
                                DataCopy(loadBuf[loadOff], this->m_xGm[srcBase], alignedPart);
                            for (uint32_t i = alignedPart; i < perBatchCount; i++)
                                loadBuf.SetValue(loadOff + i, this->m_xGm.GetValue(srcBase + i));
                        }
                        loadOff += alignedCount;
                    }
                }
            } else {
                for (uint32_t b1 = 0; b1 < this->m_bs; b1++) {
                    uint32_t firstOw = (b1 + this->m_bs - clMod) % this->m_bs;
                    if (firstOw >= this->m_oW) continue;
                    uint32_t iw = (this->m_cl + firstOw) / this->m_bs;
                    uint64_t srcBase = rowBase + b1 * this->m_bStride + static_cast<uint64_t>(iw) * this->m_depth;
                    uint32_t numBlocks = (this->m_oW - firstOw + this->m_bs - 1) / this->m_bs;
                    uint32_t count = numBlocks * this->m_depth;
                    uint32_t alignedCount = this->AlignUp(count, alignElems);
                    if ((srcBase % alignElems == 0) && (count % alignElems == 0)) {
                        DataCopy(loadBuf[loadOff], this->m_xGm[srcBase], count);
                    } else {
                        uint32_t alignedPart = (count / alignElems) * alignElems;
                        if (alignedPart > 0)
                            DataCopy(loadBuf[loadOff], this->m_xGm[srcBase], alignedPart);
                        for (uint32_t i = alignedPart; i < count; i++)
                            loadBuf.SetValue(loadOff + i, this->m_xGm.GetValue(srcBase + i));
                    }
                    loadOff += alignedCount;
                }
            }

            pipe_barrier(PIPE_ALL);

            loadOff = 0;
            for (uint32_t b1 = 0; b1 < this->m_bs; b1++) {
                uint32_t firstOw = (b1 + this->m_bs - clMod) % this->m_bs;
                if (firstOw >= this->m_oW) continue;
                uint32_t numBlocks = (this->m_oW - firstOw + this->m_bs - 1) / this->m_bs;
                uint32_t count = numBlocks * this->m_depth;
                for (uint32_t n = 0; n < numBlocks; n++) {
                    uint32_t ow = firstOw + n * this->m_bs;
                    uint64_t dstBase = outRowBase + static_cast<uint64_t>(ow) * this->m_depth;
                    for (uint32_t c = 0; c < this->m_depth; c++)
                        this->m_yGm.SetValue(dstBase + c, loadBuf.GetValue(loadOff + c));
                    loadOff += this->m_depth;
                }
                uint32_t alignedCount = this->AlignUp(count, alignElems);
                loadOff += (alignedCount - count);
            }

            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline void ProcessScalarGroupPrefetch() {
        if (this->m_startRow >= this->m_endRow) return;
        uint32_t clMod = this->m_cl % this->m_bs;
        uint32_t groupBufStride = this->m_groupBufStride;
        LocalTensor<DT_X> loadBuf = this->m_loadBuf.template Get<DT_X>();

        for (uint32_t g = 0; g < this->m_bs; g++) {
            uint32_t groupRowCount = this->m_groupRowCounts[g];
            if (groupRowCount == 0) continue;

            uint32_t rowInGroup = 0;
            for (uint64_t row = this->m_startRow; row < this->m_endRow; row++) {
                uint32_t oh = static_cast<uint32_t>(row % this->m_oH);
                uint32_t ob = static_cast<uint32_t>(row / this->m_oH);
                uint32_t fullOh = oh + this->m_ct;
                uint32_t b0 = fullOh % this->m_bs;
                if (b0 != g) continue;

                uint32_t ih = fullOh / this->m_bs;
                uint64_t rowBase = (static_cast<uint64_t>(b0) * this->m_bs * this->m_obMax + ob) * this->m_iH * this->m_iW * this->m_depth
                                 + static_cast<uint64_t>(ih) * this->m_iW * this->m_depth;
                uint64_t outRowBase = (static_cast<uint64_t>(ob) * this->m_oH + oh) * this->m_oW * this->m_depth;
                uint32_t baseOff = rowInGroup * groupBufStride;

                for (uint32_t b1 = 0; b1 < this->m_bs; b1++) {
                    uint32_t firstOw = (b1 + this->m_bs - clMod) % this->m_bs;
                    if (firstOw >= this->m_oW) continue;
                    uint32_t iw = (this->m_cl + firstOw) / this->m_bs;
                    uint64_t srcBase = rowBase + b1 * this->m_bStride + static_cast<uint64_t>(iw) * this->m_depth;
                    uint64_t dstBase = outRowBase + static_cast<uint64_t>(firstOw) * this->m_depth;
                    uint32_t numBlocks = (this->m_oW - firstOw + this->m_bs - 1) / this->m_bs;
                    for (uint32_t n = 0; n < numBlocks; n++) {
                        uint32_t ow = firstOw + n * this->m_bs;
                        for (uint32_t c = 0; c < this->m_depth; c++)
                            loadBuf.SetValue(baseOff + ow * this->m_depth + c,
                                             this->m_xGm.GetValue(srcBase + c));
                        srcBase += this->m_depth;
                    }
                }
                rowInGroup++;
            }

            pipe_barrier(PIPE_ALL);

            rowInGroup = 0;
            for (uint64_t row = this->m_startRow; row < this->m_endRow; row++) {
                uint32_t oh = static_cast<uint32_t>(row % this->m_oH);
                uint32_t ob = static_cast<uint32_t>(row / this->m_oH);
                uint32_t fullOh = oh + this->m_ct;
                uint32_t b0 = fullOh % this->m_bs;
                if (b0 != g) continue;

                uint64_t outRowBase = (static_cast<uint64_t>(ob) * this->m_oH + oh) * this->m_oW * this->m_depth;
                uint32_t baseOff = rowInGroup * groupBufStride;

                for (uint32_t b1 = 0; b1 < this->m_bs; b1++) {
                    uint32_t firstOw = (b1 + this->m_bs - clMod) % this->m_bs;
                    if (firstOw >= this->m_oW) continue;
                    uint64_t dstBase = outRowBase + static_cast<uint64_t>(firstOw) * this->m_depth;
                    uint32_t numBlocks = (this->m_oW - firstOw + this->m_bs - 1) / this->m_bs;
                    for (uint32_t n = 0; n < numBlocks; n++) {
                        uint32_t ow = firstOw + n * this->m_bs;
                        for (uint32_t c = 0; c < this->m_depth; c++)
                            this->m_yGm.SetValue(dstBase + c,
                                                  loadBuf.GetValue(baseOff + ow * this->m_depth + c));
                        dstBase += static_cast<uint64_t>(this->m_bs) * this->m_depth;
                    }
                }
                rowInGroup++;
            }

            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline void ProcessStridedDma() {
        if (this->m_startRow >= this->m_endRow) return;
        uint32_t elemSize = sizeof(DT_X);
        uint32_t clMod = this->m_cl % this->m_bs;
        uint32_t depth = this->m_depth;
        uint32_t rowElems = this->m_rowElems;
        uint32_t slotElems = this->m_stridedSlotElems;

        // Pre-computed byte strides for batch DataCopyPad
        uint32_t gapBytes = (slotElems - depth) * elemSize;       // UB gap between slots
        int32_t dstBlkStride = static_cast<int32_t>((this->m_bs - 1) * depth * elemSize); // GM output stride

        // Blocks per chunk: bounded by aligned-slot count and hw blockCount limit.
        uint32_t halfSlots = this->m_stridedBufHalfElems / slotElems;
        uint32_t slotsPerChunk = this->m_stridedBufElems / slotElems;
        if (slotsPerChunk == 0) slotsPerChunk = 1;
        if (slotsPerChunk > 4095) slotsPerChunk = 4095;
        bool useDb = (halfSlots > 0 && halfSlots != slotsPerChunk); // double buffer active

        LocalTensor<DT_X> buf = this->m_loadBuf.template Get<DT_X>();
        DataCopyPadExtParams<DT_X> pad;
        DataCopyExtParams lp, sp;
        lp.blockLen = depth * elemSize;
        lp.srcStride = 0;
        sp.blockLen = depth * elemSize;
        sp.dstStride = dstBlkStride;

        for (uint64_t row = this->m_startRow; row < this->m_endRow; row++) {
            uint32_t oh = static_cast<uint32_t>(row % this->m_oH);
            uint32_t ob = static_cast<uint32_t>(row / this->m_oH);
            uint32_t fullOh = oh + this->m_ct;
            uint32_t b0 = fullOh % this->m_bs;
            uint32_t ih = fullOh / this->m_bs;

            uint64_t rowBase = (static_cast<uint64_t>(b0) * this->m_bs * this->m_obMax + ob)
                             * this->m_iH * this->m_iW * depth
                             + static_cast<uint64_t>(ih) * this->m_iW * depth;
            uint64_t outRowBase = row * rowElems;

            for (uint32_t b1 = 0; b1 < this->m_bs; b1++) {
                uint32_t firstOw = (b1 + this->m_bs - clMod) % this->m_bs;
                if (firstOw >= this->m_oW) continue;
                uint32_t iw = (this->m_cl + firstOw) / this->m_bs;
                uint64_t srcBatchBase = rowBase + b1 * this->m_bStride + static_cast<uint64_t>(iw) * depth;
                uint32_t numBlocks = (this->m_oW - firstOw + this->m_bs - 1) / this->m_bs;

                uint32_t blkDone = 0;
                uint32_t prevBlkDone = 0;
                uint32_t prevChunk = 0;
                uint32_t bufHalf = 0;

                // ── Double-buffered pipeline (stores MTE3 + loads MTE2 overlap) ──
                // Phase 1: Load first chunk to bufA (batch DataCopyPad)
                uint32_t firstChunk = numBlocks > halfSlots ? halfSlots : numBlocks;
                if (useDb && firstChunk > 0) {
                    lp.blockCount = firstChunk;
                    lp.dstStride = gapBytes;
                    DataCopyPad(buf[0], this->m_xGm[srcBatchBase], lp, pad);
                    blkDone = firstChunk;
                    prevBlkDone = 0;
                    prevChunk = firstChunk;
                    pipe_barrier(PIPE_ALL);  // MTE2 load done → buf readable by MTE3
                }

                while (blkDone < numBlocks) {
                    uint32_t chunk = numBlocks - blkDone;
                    if (chunk > slotsPerChunk) chunk = slotsPerChunk;

                    if (useDb && prevChunk > 0) {
                        // ── Double-buffer pipeline ───────────────────────
                        // Store prev chunk from buf[prevBufOff] (MTE3)
                        // Load current chunk to buf[curBufOff] (MTE2)
                        // MTE3 and MTE2 run IN PARALLEL (independent pipes)
                        uint32_t prevBufOff = bufHalf * halfSlots * slotElems;
                        bufHalf ^= 1;
                        uint32_t curBufOff = bufHalf * halfSlots * slotElems;

                        // Store prev chunk (MTE3) — batch write with dstStride
                        sp.srcStride = gapBytes;
                        sp.blockCount = prevChunk;
                        {
                            uint64_t dstBase = outRowBase + firstOw * depth
                                             + prevBlkDone * this->m_bs * depth;
                            DataCopyPad(this->m_yGm[dstBase], buf[prevBufOff], sp);
                        }
                        // Load current chunk (MTE2) — starts while MTE3 is storing!
                        lp.dstStride = gapBytes;
                        lp.blockCount = chunk;
                        DataCopyPad(buf[curBufOff], this->m_xGm[srcBatchBase + blkDone * depth], lp, pad);

                        // pipe_barrier(PIPE_ALL): MTE3 store done + MTE2 load done → buf reusable/readable
                        pipe_barrier(PIPE_ALL);
                    } else {
                        // ── Single-buffer (chunked) ───────────────────
                        lp.dstStride = gapBytes;
                        lp.blockCount = chunk;
                        DataCopyPad(buf[0], this->m_xGm[srcBatchBase + blkDone * depth], lp, pad);
                        pipe_barrier(PIPE_ALL);  // MTE2 load done → buf readable

                        sp.srcStride = gapBytes;
                        sp.blockCount = chunk;
                        {
                            uint64_t dstBase = outRowBase + firstOw * depth
                                             + blkDone * this->m_bs * depth;
                            DataCopyPad(this->m_yGm[dstBase], buf[0], sp);
                        }
                        pipe_barrier(PIPE_ALL);  // MTE3 store done → buf reusable
                    }

                    prevBlkDone = blkDone;
                    prevChunk = chunk;
                    blkDone += chunk;
                }

                // ── Last chunk store (double buffer only) ────────────
                if (useDb && prevChunk > 0) {
                    uint32_t lastBufOff = bufHalf * halfSlots * slotElems;
                    sp.srcStride = gapBytes;
                    sp.blockCount = prevChunk;
                    {
                        uint64_t dstBase = outRowBase + firstOw * depth
                                         + prevBlkDone * this->m_bs * depth;
                        DataCopyPad(this->m_yGm[dstBase], buf[lastBufOff], sp);
                    }
                    pipe_barrier(PIPE_ALL);  // MTE3 store done before next b1/row
                }
            }
        }
    }

    __aicore__ inline void ProcessBulkStridedDma() {
        if (this->m_startRow >= this->m_endRow) return;
        uint32_t clMod = this->m_cl % this->m_bs;
        uint32_t depth = this->m_depth;
        uint32_t rowElems = this->m_rowElems;
        // Aligned slot stride for 32B-safe DataCopyPad P2 store.
        uint32_t bulkStride = ((depth * sizeof(DT_X) + 31) / 32) * (32 / sizeof(DT_X));
        LocalTensor<DT_X> buf = this->m_loadBuf.template Get<DT_X>();

        // ── P1: Load per-block, each at aligned bulkStride slot ──────
        // Each block gets its own 32B-aligned slot in UB.  Slots within
        // a (row,b1) group are indexed by (slotPos + j), NOT jumbled.
        // Waste = (bulkStride - depth) elements per block, but ensures
        // all UB→GM DataCopy addresses are 32B-aligned for MTE3 DMA.
        uint32_t slotPos = 0;
        uint32_t bs_oh = 0, bs_ob = 0;
        for (uint64_t row = this->m_startRow; row < this->m_endRow; row++) {
            if (row == this->m_startRow) {
                bs_oh = static_cast<uint32_t>(row % this->m_oH);
                bs_ob = static_cast<uint32_t>(row / this->m_oH);
            } else {
                bs_oh++; if (bs_oh >= this->m_oH) { bs_oh = 0; bs_ob++; }
            }
            uint32_t fullOh = bs_oh + this->m_ct;
            uint32_t b0 = fullOh % this->m_bs;
            uint32_t ih = fullOh / this->m_bs;

            uint64_t rowBase = (static_cast<uint64_t>(b0) * this->m_bs * this->m_obMax + bs_ob)
                             * this->m_iH * this->m_iW * depth
                             + static_cast<uint64_t>(ih) * this->m_iW * depth;

            for (uint32_t b1 = 0; b1 < this->m_bs; b1++) {
                uint32_t firstOw = (b1 + this->m_bs - clMod) % this->m_bs;
                if (firstOw >= this->m_oW) continue;
                uint32_t iw = (this->m_cl + firstOw) / this->m_bs;
                uint64_t srcBatchBase = rowBase + b1 * this->m_bStride + static_cast<uint64_t>(iw) * depth;
                uint32_t numBlocks = (this->m_oW - firstOw + this->m_bs - 1) / this->m_bs;

                for (uint32_t j = 0; j < numBlocks; j++) {
                    // Copy bulkStride elements (padded to 32B multiple).
                    // Extra elements go into slot padding between blocks.
                    DataCopy(buf[(slotPos + j) * bulkStride],
                             this->m_xGm[srcBatchBase + j * depth],
                             static_cast<uint32_t>(bulkStride));
                }
                slotPos += numBlocks;
            }
        }
        pipe_barrier(PIPE_ALL);

        // ── P2: Store per-block via DataCopyPad (handles 32B datablock ─
        // granularity for non-32B-multiple blockLen, e.g. depth=65→260B).
        DataCopyExtParams sp;
        sp.blockLen = depth * sizeof(DT_X);
        sp.blockCount = 1;
        sp.srcStride = 0;
        sp.dstStride = 0;
        slotPos = 0;
        bs_oh = 0; bs_ob = 0;
        for (uint64_t row = this->m_startRow; row < this->m_endRow; row++) {
            if (row == this->m_startRow) {
                bs_oh = static_cast<uint32_t>(row % this->m_oH);
                bs_ob = static_cast<uint32_t>(row / this->m_oH);
            } else {
                bs_oh++; if (bs_oh >= this->m_oH) { bs_oh = 0; bs_ob++; }
            }
            uint64_t outRowBase = row * rowElems;

            for (uint32_t b1 = 0; b1 < this->m_bs; b1++) {
                uint32_t firstOw = (b1 + this->m_bs - clMod) % this->m_bs;
                if (firstOw >= this->m_oW) continue;
                uint32_t numBlocks = (this->m_oW - firstOw + this->m_bs - 1) / this->m_bs;
                for (uint32_t j = 0; j < numBlocks; j++) {
                    uint64_t dstOff = outRowBase + (static_cast<uint64_t>(firstOw) + static_cast<uint64_t>(j) * this->m_bs) * depth;
                    DataCopyPad(this->m_yGm[dstOff], buf[(slotPos + j) * bulkStride], sp);
                }
                slotPos += numBlocks;
            }
        }
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void ProcessScalar() {
        if (this->m_startRow >= this->m_endRow) return;

        for (uint64_t row = this->m_startRow; row < this->m_endRow; row++) {
            uint32_t oh = static_cast<uint32_t>(row % this->m_oH);
            uint32_t ob = static_cast<uint32_t>(row / this->m_oH);
            uint32_t fullOh = oh + this->m_ct;
            uint32_t b0 = fullOh % this->m_bs;
            uint32_t ih = fullOh / this->m_bs;
            uint32_t clMod = this->m_cl % this->m_bs;

            uint64_t rowBase = (static_cast<uint64_t>(b0) * this->m_bs * this->m_obMax + ob) * this->m_iH * this->m_iW * this->m_depth
                             + static_cast<uint64_t>(ih) * this->m_iW * this->m_depth;

            uint64_t outRowBase = (static_cast<uint64_t>(ob) * this->m_oH + oh) * this->m_oW * this->m_depth;

            for (uint32_t b1 = 0; b1 < this->m_bs; b1++) {
                uint32_t firstOw = (b1 + this->m_bs - clMod) % this->m_bs;
                if (firstOw >= this->m_oW) continue;

                uint32_t iw = (this->m_cl + firstOw) / this->m_bs;
                uint64_t srcBase = rowBase + b1 * this->m_bStride + static_cast<uint64_t>(iw) * this->m_depth;
                uint64_t dstBase = outRowBase + static_cast<uint64_t>(firstOw) * this->m_depth;

                for (uint32_t ow = firstOw; ow < this->m_oW; ow += this->m_bs) {
                    for (uint32_t c = 0; c < this->m_depth; c++) {
                        this->m_yGm.SetValue(dstBase + c, this->m_xGm.GetValue(srcBase + c));
                    }
                    srcBase += this->m_depth;
                    dstBase += static_cast<uint64_t>(this->m_bs) * this->m_depth;
                }
            }
        }
        pipe_barrier(PIPE_ALL);
    }
    __aicore__ inline void ProcessUbDirectSingle() {
        // Fast path for single-combo UbDirect: ALL rows map to same (ob, b0, b1).
        // Input data is one contiguous chunk → single DataCopy load.
        // Common for cores with 1 row where only 1 b1 produces output (small oW).
        if (this->m_startRow >= this->m_endRow) return;
        uint32_t clMod = this->m_cl % this->m_bs;
        uint32_t iWd = this->m_iW * this->m_depth;
        uint32_t depth = this->m_depth;
        uint32_t bs = this->m_bs;
        uint32_t ohF = this->m_ubDirectFirstOh;
        uint32_t b0 = (ohF + this->m_ct) % bs;
        uint32_t fi = (ohF + this->m_ct) / bs;
        uint32_t b1 = this->m_ubDirectSingleB1;
        uint32_t ob = static_cast<uint32_t>(this->m_startRow / this->m_oH);
        uint32_t numIh = static_cast<uint32_t>(this->m_endRow - this->m_startRow);
        uint32_t fow = (b1 + bs - clMod) % bs;
        uint32_t iw0 = (this->m_cl + fow) / bs;
        uint32_t nb = (this->m_oW - fow + bs - 1) / bs;
        uint32_t bo = b0 * bs + b1;

        // Phase 1: single contiguous load
        LocalTensor<DT_X> ldBuf = this->m_loadBuf.template Get<DT_X>();
        uint64_t gmBase = (static_cast<uint64_t>(bo) * this->m_obMax + ob)
                        * this->m_iH * iWd + static_cast<uint64_t>(fi) * iWd;
        uint32_t totalCnt = numIh * iWd;
        DataCopy(ldBuf[0], this->m_xGm[gmBase], totalCnt);

        // TQueSync: MTE2→MTE3
        { AscendC::TQueSync<PIPE_MTE2, PIPE_MTE3> sync; sync.SetFlag(0); sync.WaitFlag(0); }

        // Phase 2: write each row with dstStride
        DataCopyExtParams wcp;
        wcp.blockLen = depth * sizeof(DT_X);
        wcp.srcStride = 0;
        wcp.dstStride = static_cast<int32_t>((bs - 1) * depth * sizeof(DT_X));
        wcp.blockCount = static_cast<uint16_t>(nb);

        uint64_t ri_gmBase = (static_cast<uint64_t>(ob) * this->m_oH + ohF) * this->m_oW * depth
                           + static_cast<uint64_t>(fow) * depth;
        uint64_t ri_srcBase = static_cast<uint64_t>(iw0) * depth;
        uint64_t gmInc = static_cast<uint64_t>(bs) * this->m_oW * depth;

        for (uint32_t ri = 0; ri < numIh; ri++) {
            DataCopyPad(this->m_yGm[ri_gmBase], ldBuf[ri_srcBase], wcp);
            ri_gmBase += gmInc;
            ri_srcBase += iWd;
        }
    }
    __aicore__ inline void ProcessUbDirect() {
        if (this->m_startRow >= this->m_endRow) return;
        uint32_t clMod = this->m_cl % this->m_bs;
        uint32_t iWd = this->m_iW * this->m_depth;
        uint32_t elemSize = sizeof(DT_X);
        uint32_t ob0 = static_cast<uint32_t>(this->m_startRow / this->m_oH);
        uint32_t ob1 = static_cast<uint32_t>((this->m_endRow - 1) / this->m_oH);
        uint32_t oh0 = static_cast<uint32_t>(this->m_startRow % this->m_oH);
        uint32_t oh1 = static_cast<uint32_t>((this->m_endRow - 1) % this->m_oH);
        uint32_t bs = this->m_bs;

        // Pre-compute all (ob, b0, b1) combos into local arrays [128]
        uint32_t cOb[128], cB0[128], cB1[128], cOhF[128], cNumIh[128], cFi[128];
        uint32_t cCnt = 0;
        for (uint32_t ob = ob0; ob <= ob1; ob++) {
            uint32_t ohS = (ob == ob0) ? oh0 : 0;
            uint32_t ohE = (ob == ob1) ? oh1 : this->m_oH - 1;
            for (uint32_t b0 = 0; b0 < bs; b0++) {
                uint32_t fm = (ohS + this->m_ct) % bs;
                uint32_t ohF = ohS + ((b0 - fm + bs) % bs);
                if (ohF > ohE) continue;
                uint32_t lm = (ohE + this->m_ct) % bs;
                uint32_t ohL = ohE - ((lm - b0 + bs) % bs);
                uint32_t fi = (ohF + this->m_ct) / bs;
                uint32_t li = (ohL + this->m_ct) / bs;
                uint32_t numIh = li - fi + 1;
                for (uint32_t b1 = 0; b1 < bs; b1++) {
                    uint32_t fow = (b1 + bs - clMod) % bs;
                    if (fow >= this->m_oW) continue;
                    if (cCnt >= 128) break;
                    cOb[cCnt] = ob; cB0[cCnt] = b0; cB1[cCnt] = b1;
                    cOhF[cCnt] = ohF; cNumIh[cCnt] = numIh; cFi[cCnt] = fi;
                    cCnt++;
                }
            }
        }

        LocalTensor<DT_X> ldBuf = this->m_loadBuf.template Get<DT_X>();

        // Phase 1: load all input data (MTE2)
        uint32_t loadOff = 0;
        for (uint32_t ci = 0; ci < cCnt; ci++) {
            uint32_t bo = cB0[ci] * bs + cB1[ci];
            uint64_t gmOff = (static_cast<uint64_t>(bo) * this->m_obMax + cOb[ci]) * this->m_iH * iWd
                           + static_cast<uint64_t>(cFi[ci]) * iWd;
            uint32_t cnt = cNumIh[ci] * iWd;
            if (cnt > 0) {
                DataCopy(ldBuf[loadOff], this->m_xGm[gmOff], cnt);
                loadOff += cnt;
            }
        }
        // TQueSync: MTE2→MTE3 (load done → MTE3 can read loadBuf)
        {
            AscendC::TQueSync<PIPE_MTE2, PIPE_MTE3> sync;
            sync.SetFlag(0);
            sync.WaitFlag(0);
        }
        // Phase 2: write all output (MTE3)
        DataCopyExtParams wcp;
        wcp.blockLen = this->m_depth * sizeof(DT_X);
        wcp.srcStride = 0;
        wcp.dstStride = static_cast<int32_t>((bs - 1) * this->m_depth * sizeof(DT_X));
        loadOff = 0;
        for (uint32_t ci = 0; ci < cCnt; ci++) {
            uint32_t fow = (cB1[ci] + bs - clMod) % bs;
            uint32_t nb = (this->m_oW - fow + bs - 1) / bs;
            if (nb == 0) { loadOff += cNumIh[ci] * iWd; continue; }
            uint32_t iw0 = (this->m_cl + fow) / bs;
            uint64_t ri_gmBase = (static_cast<uint64_t>(cOb[ci]) * this->m_oH + cOhF[ci]) * this->m_oW * this->m_depth
                                + static_cast<uint64_t>(fow) * this->m_depth;
            uint64_t ri_srcBase = static_cast<uint64_t>(loadOff) + static_cast<uint64_t>(iw0) * this->m_depth;
            uint64_t gmInc = static_cast<uint64_t>(bs) * this->m_oW * this->m_depth;
            uint32_t numIh = cNumIh[ci];
            wcp.blockCount = static_cast<uint16_t>(nb);
            for (uint32_t ri = 0; ri < numIh; ri++) {
                DataCopyPad(this->m_yGm[ri_gmBase], ldBuf[ri_srcBase], wcp);
                ri_gmBase += gmInc;
                ri_srcBase += iWd;
            }
            loadOff += numIh * iWd;
        }
        // barrier ② removed: MTE3 writes are naturally ordered before kernel exit
    }

    // ── Pipelined UbDirect: overlap MTE2 load with MTE3 write ──────────
    // For cCnt ≥ 3, issue the next combo's DataCopy on MTE2 while the
    // current combo's DataCopyPad runs on MTE3.  Hides DMA setup latency.
    __aicore__ inline void ProcessUbDirectPipelined() {
        if (this->m_startRow >= this->m_endRow) return;
        uint32_t clMod = this->m_cl % this->m_bs;
        uint32_t iWd = this->m_iW * this->m_depth;
        uint32_t elemSize = sizeof(DT_X);
        uint32_t ob0 = static_cast<uint32_t>(this->m_startRow / this->m_oH);
        uint32_t ob1 = static_cast<uint32_t>((this->m_endRow - 1) / this->m_oH);
        uint32_t oh0 = static_cast<uint32_t>(this->m_startRow % this->m_oH);
        uint32_t oh1 = static_cast<uint32_t>((this->m_endRow - 1) % this->m_oH);
        uint32_t bs = this->m_bs;

        // Combo arrays
        uint32_t cOb[32], cB0[32], cB1[32], cOhF[32], cNumIh[32], cFi[32];
        uint32_t cCnt = 0;
        for (uint32_t ob = ob0; ob <= ob1; ob++) {
            for (uint32_t b0 = 0; b0 < bs; b0++) {
                uint32_t ohS = (ob == ob0) ? oh0 : 0, ohE = (ob == ob1) ? oh1 : this->m_oH - 1;
                uint32_t fm = (ohS + this->m_ct) % bs;
                uint32_t ohF = ohS + ((b0 - fm + bs) % bs);
                if (ohF > ohE) continue;
                uint32_t lm = (ohE + this->m_ct) % bs;
                uint32_t ohL = ohE - ((lm - b0 + bs) % bs);
                uint32_t fi = (ohF + this->m_ct) / bs;
                uint32_t li = (ohL + this->m_ct) / bs;
                uint32_t numIh = li - fi + 1;
                for (uint32_t b1 = 0; b1 < bs; b1++) {
                    if (((b1 + bs - clMod) % bs) >= this->m_oW) continue;
                    cOb[cCnt] = ob; cB0[cCnt] = b0; cB1[cCnt] = b1;
                    cOhF[cCnt] = ohF; cNumIh[cCnt] = numIh; cFi[cCnt] = fi;
                    cCnt++;
                }
            }
        }

        LocalTensor<DT_X> ldBuf = this->m_loadBuf.template Get<DT_X>();

        // Phase 1: load combo 0 (kick off MTE2)
        {
            uint32_t bo = cB0[0] * bs + cB1[0];
            uint64_t gmOff = (static_cast<uint64_t>(bo) * this->m_obMax + cOb[0])
                           * this->m_iH * iWd + static_cast<uint64_t>(cFi[0]) * iWd;
            DataCopy(ldBuf[0], this->m_xGm[gmOff], cNumIh[0] * iWd);
        }

        uint32_t loadOff = cNumIh[0] * iWd;  // next load offset in loadBuf
        DataCopyExtParams wcp;
        wcp.blockLen = this->m_depth * sizeof(DT_X);
        wcp.srcStride = 0;
        wcp.dstStride = static_cast<int32_t>((bs - 1) * this->m_depth * sizeof(DT_X));

        for (uint32_t ci = 0; ci < cCnt; ci++) {
            // Sync: wait for this combo's MTE2 load to finish → MTE3 can start write
            {
                AscendC::TQueSync<PIPE_MTE2, PIPE_MTE3> sync;
                sync.SetFlag(0);
                sync.WaitFlag(0);
            }

            // Write this combo (MTE3)
            uint32_t fow = (cB1[ci] + bs - clMod) % bs;
            uint32_t nb = (this->m_oW - fow + bs - 1) / bs;
            if (nb > 0) {
                uint32_t iw0 = (this->m_cl + fow) / bs;
                uint64_t ri_gmBase = (static_cast<uint64_t>(cOb[ci]) * this->m_oH + cOhF[ci])
                                   * this->m_oW * this->m_depth + static_cast<uint64_t>(fow) * this->m_depth;
                uint64_t ri_srcBase = static_cast<uint64_t>(ci == 0 ? 0 : loadOff - cNumIh[ci] * iWd)
                                    + static_cast<uint64_t>(iw0) * this->m_depth;
                uint64_t gmInc = static_cast<uint64_t>(bs) * this->m_oW * this->m_depth;
                uint32_t numIh = cNumIh[ci];
                wcp.blockCount = static_cast<uint16_t>(nb);
                for (uint32_t ri = 0; ri < numIh; ri++) {
                    DataCopyPad(this->m_yGm[ri_gmBase], ldBuf[ri_srcBase], wcp);
                    ri_gmBase += gmInc;
                    ri_srcBase += iWd;
                }
            }

            // Kick off next combo load on MTE2 (parallel with MTE3 write!)
            if (ci + 1 < cCnt) {
                uint32_t next = ci + 1;
                uint32_t bo = cB0[next] * bs + cB1[next];
                uint64_t gmOff = (static_cast<uint64_t>(bo) * this->m_obMax + cOb[next])
                               * this->m_iH * iWd + static_cast<uint64_t>(cFi[next]) * iWd;
                uint32_t cnt = cNumIh[next] * iWd;
                DataCopy(ldBuf[loadOff], this->m_xGm[gmOff], cnt);
                loadOff += cnt;
            }
        }
        pipe_barrier(PIPE_MTE3);
    }

};

// ── CopyDepthTiny: batch depth elements via wider GM access ──────────
// TinyPath shapes have small depth (typically ≤16).  Instead of N scalar
// GM transactions (GetValue/SetValue, one per element), pack into wider
// uint32_t/uint64_t loads/stores when possible.
//
//   sizeof(DT_X)=2 (half/bf16):
//     depth=5 → 2x uint32_t (4 half) + 1x scalar (tail half) = 10B ✓
//
//   sizeof(DT_X)=4 (float/int32):
//     depth=3 → 1x uint64_t (2 float) + 1x scalar (tail) = 12B ✓
//     depth=5 → 2x uint64_t (4 float) + 1x scalar (tail) = 20B ✓
//
// NOTE: GetPhyAddr() returns a typed __gm__ pointer, so pointer
// arithmetic on it already scales by sizeof(DT_X).  The element
// offset srcOff/dstOff is added directly — no * sizeof() needed.
template <typename DT_X>
__aicore__ inline void CopyDepthTiny(
    GlobalTensor<DT_X> &dst, GlobalTensor<DT_X> &src,
    uint64_t dstOff, uint64_t srcOff, uint32_t depth)
{
    if constexpr (sizeof(DT_X) == 2) {
        // half/bf16: pack pairs into uint32_t (4B), tail odd via scalar.
        // Require even srcOff/dstOff for 4-byte aligned uint32_t access.
        if ((srcOff & 1) == 0 && (dstOff & 1) == 0 && depth >= 2) {
            uint32_t numU32 = depth / 2;
            auto *s = reinterpret_cast<__gm__ uint32_t *>(
                reinterpret_cast<uint64_t>(src.GetPhyAddr() + srcOff));
            auto *d = reinterpret_cast<__gm__ uint32_t *>(
                reinterpret_cast<uint64_t>(dst.GetPhyAddr() + dstOff));
            for (uint32_t i = 0; i < numU32; i++)
                *(d + i) = *(s + i);
            if (depth & 1)
                dst.SetValue(dstOff + depth - 1, src.GetValue(srcOff + depth - 1));
            return;
        }
    } else {
        // fp32/int32: pack pairs into uint64_t, tail odd via scalar.
        // Require even srcOff/dstOff (= 8-byte aligned for fp32).
        if (depth >= 2 && (srcOff & 1) == 0 && (dstOff & 1) == 0) {
            uint32_t numU64 = depth / 2;
            auto *s = reinterpret_cast<__gm__ uint64_t *>(
                reinterpret_cast<uint64_t>(src.GetPhyAddr() + srcOff));
            auto *d = reinterpret_cast<__gm__ uint64_t *>(
                reinterpret_cast<uint64_t>(dst.GetPhyAddr() + dstOff));
            for (uint32_t i = 0; i < numU64; i++)
                *(d + i) = *(s + i);
            if (depth & 1)
                dst.SetValue(dstOff + depth - 1, src.GetValue(srcOff + depth - 1));
            return;
        }
    }
    // fallback: depth=1 or misaligned offset
    for (uint32_t c = 0; c < depth; c++)
        dst.SetValue(dstOff + c, src.GetValue(srcOff + c));
}

// ── UbDirect standalone: no class, no multi-path ─────────────────────
// For UbPath shapes where all data fits in UB (but too large for TinyKernel).
// Bypasses the entire KernelBatchToSpace Init (40+ member writes, path
// selection, buffer allocations for unused paths).
template <typename DT_X>
__aicore__ inline void UbDirectStandalone(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &t) {
    uint32_t coreId = GetBlockIdx();
    uint32_t coreNum = t.coreNum > 0 ? t.coreNum : 1;
    uint64_t totalRows = static_cast<uint64_t>(t.outBatch) * t.outHeight;
    uint64_t startRow = (totalRows * coreId) / coreNum;
    uint64_t endRow = (totalRows * (coreId + 1)) / coreNum;
    if (startRow >= endRow) { pipe_barrier(PIPE_ALL); return; }

    GlobalTensor<DT_X> xGm, yGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

    uint32_t clMod = t.cropLeft % t.blockSize;
    uint32_t iWd = t.inputWidth * t.depth;
    uint32_t elemSize = sizeof(DT_X);
    uint32_t ob0 = static_cast<uint32_t>(startRow / t.outHeight);
    uint32_t ob1 = static_cast<uint32_t>((endRow - 1) / t.outHeight);
    uint32_t oh0 = static_cast<uint32_t>(startRow % t.outHeight);
    uint32_t oh1 = static_cast<uint32_t>((endRow - 1) % t.outHeight);
    uint32_t bs = t.blockSize;

    // Combo arrays [128]
    uint32_t cOb[128], cB0[128], cB1[128], cOhF[128], cNumIh[128], cFi[128];
    uint32_t cCnt = 0, totalInElems = 0;
    for (uint32_t ob = ob0; ob <= ob1; ob++) {
        uint32_t ohS = (ob == ob0) ? oh0 : 0, ohE = (ob == ob1) ? oh1 : t.outHeight - 1;
        for (uint32_t b0 = 0; b0 < bs; b0++) {
            uint32_t fm = (ohS + t.cropTop) % bs;
            uint32_t ohF = ohS + ((b0 - fm + bs) % bs);
            if (ohF > ohE) continue;
            uint32_t lm = (ohE + t.cropTop) % bs;
            uint32_t ohL = ohE - ((lm - b0 + bs) % bs);
            uint32_t fi = (ohF + t.cropTop) / bs;
            uint32_t li = (ohL + t.cropTop) / bs;
            uint32_t numIh = li - fi + 1;
            for (uint32_t b1 = 0; b1 < bs; b1++) {
                if (((b1 + bs - clMod) % bs) >= t.outWidth) continue;
                if (cCnt >= 128) break;
                cOb[cCnt] = ob; cB0[cCnt] = b0; cB1[cCnt] = b1;
                cOhF[cCnt] = ohF; cNumIh[cCnt] = numIh; cFi[cCnt] = fi;
                totalInElems += numIh * iWd;
                cCnt++;
            }
        }
    }
    if (cCnt == 0) return;

    // Allocate loadBuf (VECCALC) + outBuf (VECOUT)
    TPipe pipe;
    TBuf<TPosition::VECCALC> loadBuf;
    TBuf<TPosition::VECOUT> outBuf;
    uint32_t rowElems = static_cast<uint32_t>(t.outWidth) * t.depth;
    uint32_t bufBytes = ((totalInElems * elemSize + 31) / 32) * 32;
    uint32_t outBytes = ((rowElems * elemSize + 31) / 32) * 32;
    pipe.InitBuffer(loadBuf, bufBytes);
    pipe.InitBuffer(outBuf, outBytes);
    LocalTensor<DT_X> ldBuf = loadBuf.template Get<DT_X>();
    LocalTensor<DT_X> oBuf = outBuf.template Get<DT_X>();

    // Phase 1: load all combo data (MTE2)
    uint32_t loadOff = 0;
    for (uint32_t ci = 0; ci < cCnt; ci++) {
        uint32_t bo = cB0[ci] * bs + cB1[ci];
        uint64_t gmOff = (static_cast<uint64_t>(bo) * t.outBatch + cOb[ci]) * t.inputHeight * iWd
                       + static_cast<uint64_t>(cFi[ci]) * iWd;
        uint32_t cnt = cNumIh[ci] * iWd;
        DataCopy(ldBuf[loadOff], xGm[gmOff], cnt);
        loadOff += cnt;
    }

    // TQS: MTE2→V (load data visible to PIPE_V)
    { AscendC::TQueSync<PIPE_MTE2, PIPE_V> sync; sync.SetFlag(0); sync.WaitFlag(0); }

    // Phase 2: row-by-row, Duplicate+scatter+continuous write
    // For each row: fill oBuf with zeros → scatter ALL b1 blocks from loadBuf
    //   → TQS<V,MTE3> → one contiguous DataCopy to GM.
    // This avoids strided MTE3 write (50% bandwidth waste for bs=2).
    for (uint64_t row = startRow; row < endRow; row++) {
        uint32_t oh = static_cast<uint32_t>(row % t.outHeight);
        uint32_t ob = static_cast<uint32_t>(row / t.outHeight);
        uint32_t b0 = (oh + t.cropTop) % bs;

        // Zero oBuf: all gap positions become 0
        Duplicate(oBuf, static_cast<DT_X>(0), rowElems);

        // Scatter blocks from all matching combos
        for (uint32_t ci = 0; ci < cCnt; ci++) {
            if (cOb[ci] != ob || cB0[ci] != b0) continue;
            if (oh < cOhF[ci]) continue;
            uint32_t ri = (oh - cOhF[ci]) / bs;
            if (ri >= cNumIh[ci]) continue;

            uint32_t fow = (cB1[ci] + bs - clMod) % bs;
            uint32_t nb = (t.outWidth - fow + bs - 1) / bs;
            uint32_t iw0 = (t.cropLeft + fow) / bs;
            // Find this combo's data offset in loadBuf
            uint32_t baseOff = 0;
            for (uint32_t cj = 0; cj < ci; cj++)
                baseOff += cNumIh[cj] * iWd;
            uint64_t rowSrcBase = static_cast<uint64_t>(baseOff) + static_cast<uint64_t>(ri) * iWd;

            // P1: merge nb DataCopy(UB,UB) via DataCopyParams (datablock units)
            constexpr uint16_t ELEMS_PER_BLK = 32 / sizeof(DT_X);
            uint16_t blkLen = static_cast<uint16_t>(t.depth / ELEMS_PER_BLK);
            if (blkLen > 0) {
                DataCopyParams ubp;
                ubp.blockCount = static_cast<uint16_t>(nb);
                ubp.blockLen   = blkLen;
                ubp.srcGap     = 0;
                ubp.dstGap     = static_cast<uint16_t>((t.blockSize - 1) * blkLen);
                DataCopy(oBuf[static_cast<uint32_t>(fow * t.depth)],
                         ldBuf[static_cast<uint32_t>(rowSrcBase + iw0 * t.depth)], ubp);
            } else {
                for (uint32_t n = 0; n < nb; n++)
                    DataCopy(oBuf[static_cast<uint32_t>(fow + n * bs) * t.depth],
                             ldBuf[static_cast<uint32_t>(rowSrcBase + (iw0 + n) * t.depth)], t.depth);
            }
        }

        // Sync: VECOUT → MTE3
        { AscendC::TQueSync<PIPE_V, PIPE_MTE3> sync; sync.SetFlag(0); sync.WaitFlag(0); }

        // One contiguous DataCopy to GM (full bandwidth, no stride waste)
        uint64_t gmBase = (static_cast<uint64_t>(ob) * t.outHeight + oh) * t.outWidth * t.depth;
        DataCopy(yGm[gmBase], oBuf[0], rowElems);
        // MTE3 done → V can reuse oBuf for next row
        { AscendC::TQueSync<PIPE_MTE3, PIPE_V> sync; sync.SetFlag(0); sync.WaitFlag(0); }
    }
    // barrier ② removed: MTE3 drains on kernel exit
}

template <typename DT_X>
// ── AlignedBulkDirect standalone: interleaved DataCopyPad load + contiguous
// DataCopy store.  For aligned depth, this replaces UbDirect's combo
// enumeration + scatter, eliminating the V-pipeline scatter overhead.
__aicore__ inline void AlignedBulkDirectStandalone(GM_ADDR x, GM_ADDR y,
                                                    const BatchToSpaceTilingData &t) {
    uint32_t cid = GetBlockIdx();
    uint32_t cn = t.coreNum > 0 ? t.coreNum : 1;
    uint64_t totalRows = static_cast<uint64_t>(t.outBatch) * t.outHeight;
    uint64_t sRow = (totalRows * cid) / cn;
    uint64_t eRow = (totalRows * (cid + 1)) / cn;
    if (sRow >= eRow) { pipe_barrier(PIPE_ALL); return; }

    GlobalTensor<DT_X> xGm, yGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));

    uint32_t depth = t.depth, bs = t.blockSize, elemSz = sizeof(DT_X);
    uint32_t clMod = t.cropLeft % bs;
    uint32_t rowElems = t.rowElems;
    constexpr uint16_t ELEMS_PER_BLK = 32 / sizeof(DT_X);
    uint16_t blkLenLocal = static_cast<uint16_t>(depth / ELEMS_PER_BLK);
    uint16_t interleave = static_cast<uint16_t>((bs - 1) * blkLenLocal);
    uint32_t rowBytes = ((rowElems * elemSz + 31) / 32) * 32;

    // Fit as many rows as UB allows (single batch, two barriers total).
    uint32_t availE = (t.ubSizeBytes > 1024 ? t.ubSizeBytes - 1024 : 0) / elemSz;
    uint32_t maxNRows = availE / (rowBytes / elemSz);
    if (maxNRows < 1) maxNRows = 1;

    TPipe pipe;
    TBuf<TPosition::VECOUT> rowBuf;
    pipe.InitBuffer(rowBuf, maxNRows * rowBytes);
    LocalTensor<DT_X> buf = rowBuf.template Get<DT_X>();
    DataCopyPadExtParams<DT_X> pad;

    for (uint64_t batchStart = sRow; batchStart < eRow; batchStart += maxNRows) {
        uint64_t batchEnd = batchStart + maxNRows;
        if (batchEnd > eRow) batchEnd = eRow;
        uint32_t nRows = static_cast<uint32_t>(batchEnd - batchStart);

        // P1: interleaved load for all rows in the batch
        for (uint64_t row = batchStart; row < batchEnd; row++) {
            uint32_t oh = static_cast<uint32_t>(row % t.outHeight);
            uint32_t ob = static_cast<uint32_t>(row / t.outHeight);
            uint32_t fullOh = oh + t.cropTop;
            uint32_t b0 = fullOh % bs;
            uint32_t ih = fullOh / bs;
            uint64_t rBase = (static_cast<uint64_t>(b0) * bs * t.outBatch + ob)
                           * t.inputHeight * t.inputWidth * depth
                           + static_cast<uint64_t>(ih) * t.inputWidth * depth;
            uint32_t ro = static_cast<uint32_t>(row - batchStart) * rowElems;

            for (uint32_t b1 = 0; b1 < bs; b1++) {
                uint32_t fo = (b1 + bs - clMod) % bs;
                if (fo >= t.outWidth) continue;
                uint32_t iw = (t.cropLeft + fo) / bs;
                uint64_t sBase = rBase + b1 * static_cast<uint64_t>(t.outBatch)
                               * t.inputHeight * t.inputWidth * depth
                               + static_cast<uint64_t>(iw) * depth;
                uint32_t nb = (t.outWidth - fo + bs - 1) / bs;
                DataCopyExtParams lp;
                lp.blockLen   = depth * elemSz;
                lp.blockCount = static_cast<uint16_t>(nb);
                lp.srcStride  = 0;
                lp.dstStride  = interleave;
                DataCopyPad(buf[ro + fo * depth], xGm[sBase], lp, pad);
            }
        }
        pipe_barrier(PIPE_ALL);

        // P2: contiguous store for all rows in the batch
        for (uint64_t row = batchStart; row < batchEnd; row++) {
            uint32_t ro = static_cast<uint32_t>(row - batchStart) * rowElems;
            DataCopy(yGm[row * rowElems], buf[ro], rowElems);
        }
        pipe_barrier(PIPE_ALL);
    }
}

template <typename DT_X>
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingData, tiling_data, tiling);
    if (tiling_data.useAlignedBulkDirect) {
        AlignedBulkDirectStandalone<DT_X>(x, y, tiling_data);
        return;
    }
    if (tiling_data.useSimpleUbDirect) {
        UbDirectStandalone<DT_X>(x, y, tiling_data);
        return;
    }
    KernelBatchToSpace<DT_X> op;
    op.Init(x, y, workspace, &tiling_data);
    op.Process();
}