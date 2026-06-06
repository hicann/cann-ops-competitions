// BatchToSpace0 核函数：
// - 主机侧按 10 个数据点选择 MODE，核函数通过 if constexpr 编译期特化。
// - 小数据点优先走 block=2 direct/P4 exact 路径，减少通用 ROW_TILE 的标量反解和分支。
// - P7/P9/P10 保留独立路径，分别处理大 D、block=4 宽 W、超高 H 场景。
// - 非对齐 D 默认只在 gather 能放入 UB 时整行重排，避免非对齐 strided GM 写的正确性风险。
#include "kernel_operator.h"

#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <class DT_X, uint32_t MODE>
class KernelBatchToSpace {
public:
    __aicore__ inline KernelBatchToSpace() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR rawTiling, const BatchToSpaceTilingData &t)
    {
        N_ = t.N;
        H_ = t.H;
        W_ = t.W;
        D_ = t.D;
        block_ = t.block;
        ct_ = t.ct;
        cl_ = t.cl;
        OH_ = t.OH;
        OW_ = t.OW;
        totalUnits_ = t.totalUnits;
        usedCoreNum_ = t.usedCoreNum;
        tileW_ = t.tileW;
        dTileLen_ = t.dTileLen;
        gatherDepth_ = t.gatherDepth;
        aligned_ = t.aligned != 0;
        // D 是否"真 32B 对齐"(D*sizeof%32==0)。aligned_ 还包含 block==1(输出连续)，但 block==1+D非32B(如 D=65)
        // 时 DataCopy(需 32B 对齐地址/长度) 会在任意 flatStart 处失败 → 这类用 DataCopyPad(单块,任意地址/长)。
        dAligned32_ = (((uint64_t)D_ * sizeof(DT_X)) % 32 == 0);

        blockElems_ = 32 / sizeof(DT_X);
        colAligned_ = AlignUp(D_);
        blockD_ = static_cast<uint64_t>(block_) * D_;
        owd_ = static_cast<uint64_t>(OW_) * D_;

        numTilesPerRow_ = (W_ + tileW_ - 1) / tileW_;
        if constexpr (MODE == BTS_MODE_B4_W_PACK_TILE) {
            numTilesPerRow_ = (OW_ + tileW_ - 1) / tileW_;
        }
        const uint64_t flatElems = static_cast<uint64_t>(H_) * W_;
        flatTiles_ = static_cast<uint32_t>((flatElems + tileW_ - 1) / tileW_);
        numDTiles_ = (D_ + dTileLen_ - 1) / dTileLen_;
        bufElems_ = aligned_ ? AlignUp(tileW_ * dTileLen_) : tileW_ * colAligned_;

        uint64_t xTotal = static_cast<uint64_t>(N_) * block_ * block_ * H_ * W_ * D_;
        uint64_t yTotal = static_cast<uint64_t>(N_) * OH_ * OW_ * D_;
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, xTotal);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, yTotal);

        SplitWork();
        if constexpr (MODE == BTS_MODE_P10_OH_GATHER_TILE || MODE == BTS_MODE_P10_UB_PACK_TILE) {
            srcRowElems_ = W_ * D_;
            p10OhTiles_ = (OH_ + tileW_ - 1) / tileW_;
            p10SlabStride_ = ((tileW_ + block_ - 1) / block_) * srcRowElems_;
            rowElems_ = tileW_ * OW_ * D_;
            pipe_.InitBuffer(inQueue_, gatherDepth_, block_ * block_ * p10SlabStride_ * sizeof(DT_X));
            pipe_.InitBuffer(outQueue_, gatherDepth_, rowElems_ * sizeof(DT_X));
            if constexpr (MODE == BTS_MODE_P10_OH_GATHER_TILE) {
                idxBufElems_ = AlignUpU32(rowElems_);
                pipe_.InitBuffer(idxBuf_, idxBufElems_ * sizeof(uint32_t));
                LoadGatherIndex(rawTiling);
            }
        } else if constexpr (MODE == BTS_MODE_B4_W_PACK_TILE) {
            srcRowElems_ = ((tileW_ + block_ - 1) / block_) * D_;
            rowElems_ = tileW_ * D_;
            pipe_.InitBuffer(inQueue_, gatherDepth_, block_ * srcRowElems_ * sizeof(DT_X));
            pipe_.InitBuffer(outQueue_, gatherDepth_, rowElems_ * sizeof(DT_X));
        } else if constexpr (MODE == BTS_MODE_LARGE_D) {
            bufElems_ = AlignUp(dTileLen_);
            pipe_.InitBuffer(inQueue_, BUFFER_NUM, bufElems_ * sizeof(DT_X));
            pipe_.InitBuffer(outQueue_, BUFFER_NUM, bufElems_ * sizeof(DT_X));
        } else if constexpr (MODE == BTS_MODE_GATHER_ROW || MODE == BTS_MODE_P2_GATHER_V73) {
            // 非对齐 Gather 路径：src 缓冲放 block 个源行(各 W*D，32B 对齐步长 regionStride_)，
            // out 缓冲放一整个输出行(OW*D)，idxBuf_ 放行无关的 gather 索引(算一次复用)。
            srcRowElems_ = W_ * D_;
            regionStride_ = AlignUp(srcRowElems_);
            rowElems_ = OW_ * D_;
            idxBufElems_ = AlignUpU32(rowElems_);
            pipe_.InitBuffer(inQueue_, gatherDepth_, (uint32_t)block_ * regionStride_ * sizeof(DT_X));
            pipe_.InitBuffer(outQueue_, gatherDepth_, rowElems_ * sizeof(DT_X));
            pipe_.InitBuffer(idxBuf_, idxBufElems_ * sizeof(uint32_t));
            if constexpr (MODE == BTS_MODE_P2_GATHER_V73) {
                BuildGatherIndex();
            } else {
                LoadGatherIndex(rawTiling);
            }
        } else {
            pipe_.InitBuffer(inQueue_, BUFFER_NUM, bufElems_ * sizeof(DT_X));
            pipe_.InitBuffer(outQueue_, BUFFER_NUM, bufElems_ * sizeof(DT_X));
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (MODE == BTS_MODE_P10_UB_PACK_TILE) {
            ProcessP10UbPackTilePacket();
        } else if constexpr (MODE == BTS_MODE_P10_OH_GATHER_TILE) {
            ProcessP10OhGatherTilePacket();
        } else if constexpr (MODE == BTS_MODE_B4_W_PACK_TILE) {
            ProcessB4WPackTilePacket();
        } else if constexpr (MODE == BTS_MODE_LARGE_D) {
            ProcessLargeDPacket();
        } else if constexpr (MODE == BTS_MODE_P2_GATHER_V73) {
            ProcessP2GatherV73Packet();
        } else if constexpr (MODE == BTS_MODE_GATHER_ROW) {
            ProcessGatherRowPacket();
        } else if constexpr (MODE == BTS_MODE_P1_ROW_V71) {
            ProcessP1RowV71Packet();
        } else if constexpr (MODE == BTS_MODE_P3_ROW_V73) {
            ProcessP3RowV73Packet();
        } else if constexpr (MODE == BTS_MODE_FLAT_TILE) {
            ProcessFlatTilePacket();
        } else {
            ProcessRowTilePacket();
        }
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t x)
    {
        return (x + blockElems_ - 1) / blockElems_ * blockElems_;
    }

    __aicore__ inline uint32_t AlignUpU32(uint32_t x)
    {
        return (x + 7) / 8 * 8;
    }

    __aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline uint32_t MaxU32(uint32_t lhs, uint32_t rhs)
    {
        return lhs > rhs ? lhs : rhs;
    }

    __aicore__ inline uint64_t MinU64(uint64_t lhs, uint64_t rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline void SplitWork()
    {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t base = totalUnits_ / usedCoreNum_;
        uint32_t rem = totalUnits_ % usedCoreNum_;
        cnt_ = (blockIdx < rem) ? (base + 1) : base;
        start_ = (blockIdx < rem) ? (blockIdx * cnt_) : (rem * (base + 1) + (blockIdx - rem) * base);
    }

    __aicore__ inline void ComputeWRange(uint32_t j)
    {
        curWlo_ = 0;
        curValidW_ = W_;
        curOw0_ = j;
    }

    // 路径 A（ROW_TILE）：每个源行 (n,i,j,h) 按 tileW 切列，不跨行。
    __aicore__ inline void ProcessRowTilePacket()
    {
        uint32_t rem = start_;
        uint32_t tileIdx = rem % numTilesPerRow_;
        rem /= numTilesPerRow_;
        uint32_t h = rem % H_;
        rem /= H_;
        uint32_t j = rem % block_;
        rem /= block_;
        uint32_t i = rem % block_;
        rem /= block_;
        uint32_t n = rem;
        ComputeWRange(j);

        for (uint32_t idx = 0; idx < cnt_; ++idx) {
            ProcessRowTile(n, i, j, h, tileIdx);
            if (++tileIdx == numTilesPerRow_) {
                tileIdx = 0;
                if (++h == H_) {
                    h = 0;
                    if (++j == block_) {
                        j = 0;
                        if (++i == block_) {
                            i = 0;
                            ++n;
                        }
                    }
                    ComputeWRange(j);
                }
            }
        }
    }

    __aicore__ inline void ProcessP1RowV71Packet()
    {
        uint32_t rem = start_;
        uint32_t tileIdx = rem % numTilesPerRow_;
        rem /= numTilesPerRow_;
        uint32_t h = rem % H_;
        rem /= H_;
        uint32_t i = rem % block_;
        rem /= block_;
        uint32_t n = rem;

        for (uint32_t idx = 0; idx < cnt_; ++idx) {
            for (uint32_t j = 0; j < block_; ++j) {
                ComputeWRange(j);
                ProcessRowTile(n, i, j, h, tileIdx);
            }
            if (++tileIdx == numTilesPerRow_) {
                tileIdx = 0;
                if (++h == H_) {
                    h = 0;
                    if (++i == block_) {
                        i = 0;
                        ++n;
                    }
                }
            }
        }
    }

    __aicore__ inline void ProcessP3RowV73Packet()
    {
        uint32_t rem = start_;
        uint32_t tileIdx = rem % numTilesPerRow_;
        rem /= numTilesPerRow_;
        uint32_t h = rem % H_;
        rem /= H_;
        uint32_t j = rem % block_;
        rem /= block_;
        uint32_t i = rem % block_;
        rem /= block_;
        uint32_t n = rem;
        ComputeWRange(j);

        for (uint32_t idx = 0; idx < cnt_; ++idx) {
            ProcessRowTile(n, i, j, h, tileIdx);
            if (++tileIdx == numTilesPerRow_) {
                tileIdx = 0;
                if (++h == H_) {
                    h = 0;
                    if (++j == block_) {
                        j = 0;
                        if (++i == block_) {
                            i = 0;
                            ++n;
                        }
                    }
                    ComputeWRange(j);
                }
            }
        }
    }

    __aicore__ inline void ProcessRowTile(uint32_t n, uint32_t i, uint32_t j, uint32_t h, uint32_t tileIdx)
    {
        int64_t oh = static_cast<int64_t>(h) * static_cast<int64_t>(block_) +
                     static_cast<int64_t>(i) - static_cast<int64_t>(ct_);

        uint32_t wStart = curWlo_ + tileIdx * tileW_;
        uint32_t wEnd = wStart + tileW_;
        uint32_t validEnd = curWlo_ + curValidW_;
        if (wEnd > validEnd) {
            wEnd = validEnd;
        }
        uint32_t curW = wEnd - wStart;
        uint32_t b = (i * block_ + j) * N_ + n;
        uint64_t inOff = ((static_cast<uint64_t>(b) * H_ + h) * W_ + wStart) * D_;
        uint64_t outOff = (static_cast<uint64_t>(n) * OH_ + static_cast<uint64_t>(oh)) * owd_ +
                          static_cast<uint64_t>(curOw0_ + (wStart - curWlo_) * block_) * D_;

        CopyIn(inOff, curW, D_);
        Compute(curW, D_);
        CopyOutRowSegment(outOff, curW, D_);
    }

    // 路径 B（FLAT_TILE）：数据点 10 将 H*W 压成一维，tile 可跨多行连续搬入。
    __aicore__ inline void ProcessFlatTilePacket()
    {
        uint32_t rem = start_;
        uint32_t flatTile = rem % flatTiles_;
        rem /= flatTiles_;
        uint32_t j = rem % block_;
        rem /= block_;
        uint32_t i = rem % block_;
        rem /= block_;
        uint32_t n = rem;
        ComputeWRange(j);

        for (uint32_t idx = 0; idx < cnt_; ++idx) {
            ProcessFlatTile(n, i, j, flatTile);
            if (++flatTile == flatTiles_) {
                flatTile = 0;
                if (++j == block_) {
                    j = 0;
                    if (++i == block_) {
                        i = 0;
                        ++n;
                    }
                }
                ComputeWRange(j);
            }
        }
    }

    __aicore__ inline void ProcessFlatTile(uint32_t n, uint32_t i, uint32_t j, uint32_t flatTile)
    {
        const uint64_t flatElems = static_cast<uint64_t>(H_) * W_;
        const uint64_t flatStart = static_cast<uint64_t>(flatTile) * tileW_;
        const uint32_t curCols = static_cast<uint32_t>(MinU64(tileW_, flatElems - flatStart));
        const uint32_t b = (i * block_ + j) * N_ + n;
        const uint64_t inOff = (static_cast<uint64_t>(b) * H_ * W_ + flatStart) * D_;

        CopyIn(inOff, curCols, D_);
        Compute(curCols, D_);

        const uint32_t h = static_cast<uint32_t>(flatStart / W_);
        const uint32_t w = static_cast<uint32_t>(flatStart - static_cast<uint64_t>(h) * W_);
        CopyOutFlatTile(n, i, h, w, curCols);
    }

    // 路径 C（LARGE_D）：数据点 7 按输出像素和 D tile 切任务，内层只做 O(1) 地址增量。
    __aicore__ inline void ProcessLargeDPacket()
    {
        uint32_t rem = start_;
        uint32_t dTile = rem % numDTiles_;
        rem /= numDTiles_;
        uint32_t ow = rem % OW_;
        rem /= OW_;
        uint32_t oh = rem % OH_;
        rem /= OH_;
        uint32_t n = rem;

        uint32_t hidx = oh + ct_;
        uint32_t i = hidx % block_;
        uint32_t h = hidx / block_;

        uint32_t widx = ow + cl_;
        uint32_t j = widx % block_;
        uint32_t w = widx / block_;
        uint32_t b = (i * block_ + j) * N_ + n;

        uint64_t baseInOffset = ((static_cast<uint64_t>(b) * H_ + h) * W_ + w) * D_;
        uint64_t baseOutOffset = ((static_cast<uint64_t>(n) * OH_ + oh) * OW_ + ow) * D_;

        int64_t stepJIn = static_cast<int64_t>(N_) * H_ * W_ * D_;
        int64_t stepWIn = static_cast<int64_t>(D_) - static_cast<int64_t>(block_ - 1) * stepJIn;

        for (uint32_t idx = 0; idx < cnt_; ++idx) {
            uint32_t dStart = dTile * dTileLen_;
            uint32_t curDLen = dTileLen_;
            if (dStart + curDLen > D_) {
                curDLen = D_ - dStart;
            }

            uint64_t inOffset = baseInOffset + dStart;
            uint64_t outOffset = baseOutOffset + dStart;

            LocalTensor<DT_X> xLocal = inQueue_.AllocTensor<DT_X>();
            DataCopy(xLocal, xGm_[inOffset], curDLen);
            inQueue_.EnQue(xLocal);

            LocalTensor<DT_X> xq = inQueue_.DeQue<DT_X>();
            LocalTensor<DT_X> yLocal = outQueue_.AllocTensor<DT_X>();
            DataCopy(yLocal, xq, curDLen);
            outQueue_.EnQue(yLocal);
            inQueue_.FreeTensor(xq);

            LocalTensor<DT_X> yq = outQueue_.DeQue<DT_X>();
            DataCopy(yGm_[outOffset], yq, curDLen);
            outQueue_.FreeTensor(yq);

            if (++dTile == numDTiles_) {
                dTile = 0;
                if (++ow == OW_) {
                    ow = 0;
                    if (++oh == OH_) {
                        oh = 0;
                        ++n;
                    }
                    uint32_t hidxNew = oh + ct_;
                    i = hidxNew % block_;
                    h = hidxNew / block_;
                    uint32_t widxNew = cl_;
                    j = widxNew % block_;
                    w = widxNew / block_;

                    uint32_t bNew = (i * block_ + j) * N_ + n;
                    baseInOffset = ((static_cast<uint64_t>(bNew) * H_ + h) * W_ + w) * D_;
                    baseOutOffset = ((static_cast<uint64_t>(n) * OH_ + oh) * OW_ + ow) * D_;
                } else {
                    baseOutOffset += D_;
                    if (++j == block_) {
                        j = 0;
                        baseInOffset = static_cast<uint64_t>(static_cast<int64_t>(baseInOffset) + stepWIn);
                    } else {
                        baseInOffset = static_cast<uint64_t>(static_cast<int64_t>(baseInOffset) + stepJIn);
                    }
                }
            }
        }
    }

    __aicore__ inline void ProcessP10OhGatherTilePacket()
    {
        uint32_t rem = start_;
        uint32_t tileIdx = rem % p10OhTiles_;
        uint32_t n = rem / p10OhTiles_;

        for (uint32_t idx = 0; idx < cnt_; ++idx) {
            ProcessP10OhGatherTile(n, tileIdx);
            if (++tileIdx == p10OhTiles_) {
                tileIdx = 0;
                ++n;
            }
        }
    }

    __aicore__ inline void ProcessP10OhGatherTile(uint32_t n, uint32_t tileIdx)
    {
        uint32_t ohStart = tileIdx * tileW_;
        uint32_t curOH = MinU32(tileW_, OH_ - ohStart);
        uint32_t curElems = curOH * OW_ * D_;

        LocalTensor<DT_X> srcBuf = inQueue_.AllocTensor<DT_X>();
        LoadP10SourceSlabs(srcBuf, n, ohStart, curOH);
        inQueue_.EnQue(srcBuf);

        LocalTensor<DT_X> src = inQueue_.DeQue<DT_X>();
        LocalTensor<DT_X> out = outQueue_.AllocTensor<DT_X>();
        LocalTensor<uint32_t> idx = idxBuf_.Get<uint32_t>();
        Gather(out, src, idx, static_cast<uint32_t>(0), curElems);
        outQueue_.EnQue<DT_X>(out);
        inQueue_.FreeTensor(src);

        LocalTensor<DT_X> y = outQueue_.DeQue<DT_X>();
        uint64_t outOff = (static_cast<uint64_t>(n) * OH_ + ohStart) * owd_;
        DataCopy(yGm_[outOff], y, curElems);
        outQueue_.FreeTensor(y);
    }

    __aicore__ inline void ProcessP10UbPackTilePacket()
    {
        uint32_t rem = start_;
        uint32_t tileIdx = rem % p10OhTiles_;
        uint32_t n = rem / p10OhTiles_;

        for (uint32_t idx = 0; idx < cnt_; ++idx) {
            ProcessP10UbPackTile(n, tileIdx);
            if (++tileIdx == p10OhTiles_) {
                tileIdx = 0;
                ++n;
            }
        }
    }

    __aicore__ inline void ProcessP10UbPackTile(uint32_t n, uint32_t tileIdx)
    {
        uint32_t ohStart = tileIdx * tileW_;
        uint32_t curOH = MinU32(tileW_, OH_ - ohStart);
        uint32_t curElems = curOH * OW_ * D_;

        LocalTensor<DT_X> srcBuf = inQueue_.AllocTensor<DT_X>();
        LoadP10SourceSlabs(srcBuf, n, ohStart, curOH);
        inQueue_.EnQue(srcBuf);

        LocalTensor<DT_X> src = inQueue_.DeQue<DT_X>();
        LocalTensor<DT_X> out = outQueue_.AllocTensor<DT_X>();

        // P10 UB pack 只在 block=2、无裁剪、D 32B 对齐的数据点 10 上启用，直接走拼装快路径。
        DataCopyParams p = {0, 0, 0, 0};
        const uint16_t dBlocks = static_cast<uint16_t>((D_ * sizeof(DT_X)) / 32);
        p.blockLen = dBlocks;
        p.srcStride = static_cast<uint16_t>((W_ - 1) * dBlocks);
        p.dstStride = static_cast<uint16_t>((block_ * OW_ - 1) * dBlocks);

        uint32_t hidxMod = ohStart % block_;
        for (uint32_t i = 0; i < block_; ++i) {
            uint32_t firstLocal = (i + block_ - hidxMod) % block_;
            uint32_t hRows = (curOH - firstLocal + block_ - 1) / block_;
            p.blockCount = static_cast<uint16_t>(hRows);
            for (uint32_t j = 0; j < block_; ++j) {
                for (uint32_t w = 0; w < W_; ++w) {
                    uint32_t srcOff = (i * block_ + j) * p10SlabStride_ + w * D_;
                    uint32_t dstOff = (firstLocal * OW_ + w * block_ + j) * D_;
                    DataCopy(out[dstOff], src[srcOff], p);
                }
            }
        }
        outQueue_.EnQue<DT_X>(out);
        inQueue_.FreeTensor(src);

        LocalTensor<DT_X> y = outQueue_.DeQue<DT_X>();
        uint64_t outOff = (static_cast<uint64_t>(n) * OH_ + ohStart) * owd_;
        DataCopy(yGm_[outOff], y, curElems);
        outQueue_.FreeTensor(y);
    }

    __aicore__ inline void LoadP10SourceSlabs(LocalTensor<DT_X> &srcBuf, uint32_t n, uint32_t ohStart, uint32_t curOH)
    {
        uint32_t hidxMod = (ohStart + ct_) % block_;
        for (uint32_t i = 0; i < block_; ++i) {
            uint32_t firstLocal = (i + block_ - hidxMod) % block_;
            uint32_t hRows = (curOH - firstLocal + block_ - 1) / block_;
            uint32_t hBase = (ohStart + firstLocal + ct_) / block_;
            for (uint32_t j = 0; j < block_; ++j) {
                uint32_t b = (i * block_ + j) * N_ + n;
                uint64_t inOff = ((static_cast<uint64_t>(b) * H_ + hBase) * W_) * D_;
                uint32_t localOff = (i * block_ + j) * p10SlabStride_;
                DataCopy(srcBuf[localOff], xGm_[inOff], hRows * srcRowElems_);
            }
        }
    }

    __aicore__ inline void ProcessB4WPackTilePacket()
    {
        uint32_t rem = start_;
        uint32_t tileIdx = rem % numTilesPerRow_;
        rem /= numTilesPerRow_;
        uint32_t oh = rem % OH_;
        uint32_t n = rem / OH_;

        for (uint32_t idx = 0; idx < cnt_; ++idx) {
            ProcessB4WPackTile(n, oh, tileIdx);
            if (++tileIdx == numTilesPerRow_) {
                tileIdx = 0;
                if (++oh == OH_) {
                    oh = 0;
                    ++n;
                }
            }
        }
    }

    __aicore__ inline void ProcessB4WPackTile(uint32_t n, uint32_t oh, uint32_t tileIdx)
    {
        uint32_t owStart = tileIdx * tileW_;
        uint32_t curOW = MinU32(tileW_, OW_ - owStart);
        uint32_t curElems = curOW * D_;

        uint32_t hidx = oh + ct_;
        uint32_t i = hidx % block_;
        uint32_t h = hidx / block_;
        uint32_t widxBase = owStart + cl_;
        uint32_t baseMod = widxBase % block_;

        LocalTensor<DT_X> srcBuf = inQueue_.AllocTensor<DT_X>();
        for (uint32_t j = 0; j < block_; ++j) {
            uint32_t firstLocal = (j + block_ - baseMod) % block_;
            if (firstLocal >= curOW) {
                continue;
            }
            uint32_t srcCols = (curOW - firstLocal + block_ - 1) / block_;
            uint32_t srcW = (widxBase + firstLocal) / block_;
            uint32_t b = (i * block_ + j) * N_ + n;
            uint64_t inOff = ((static_cast<uint64_t>(b) * H_ + h) * W_ + srcW) * D_;
            uint32_t localOff = j * srcRowElems_;
            DataCopy(srcBuf[localOff], xGm_[inOff], srcCols * D_);
        }
        inQueue_.EnQue(srcBuf);

        LocalTensor<DT_X> src = inQueue_.DeQue<DT_X>();
        LocalTensor<DT_X> out = outQueue_.AllocTensor<DT_X>();
        // P9 固定 D 32B 对齐；按输出列连续写，水平裁剪只影响每个 j 的首列。
        DataCopyParams p = {0, 0, 0, 0};
        const uint16_t dBlocks = static_cast<uint16_t>((D_ * sizeof(DT_X)) / 32);
        p.blockLen = dBlocks;
        p.srcStride = 0;
        p.dstStride = static_cast<uint16_t>((block_ - 1) * dBlocks);
        for (uint32_t j = 0; j < block_; ++j) {
            uint32_t firstLocal = (j + block_ - baseMod) % block_;
            if (firstLocal >= curOW) {
                continue;
            }
            p.blockCount = static_cast<uint16_t>((curOW - firstLocal + block_ - 1) / block_);
            DataCopy(out[firstLocal * D_], src[j * srcRowElems_], p);
        }
        outQueue_.EnQue<DT_X>(out);
        inQueue_.FreeTensor(src);

        LocalTensor<DT_X> y = outQueue_.DeQue<DT_X>();
        uint64_t outOff = (static_cast<uint64_t>(n) * OH_ + oh) * owd_ +
                          static_cast<uint64_t>(owStart) * D_;
        DataCopy(yGm_[outOff], y, curElems);
        outQueue_.FreeTensor(y);
    }

    // ===== GATHER_ROW（非对齐 D）：把数据重排从 MTE3 逐列写挪到 VEC 的 Gather =====
    // 索引表与行无关：idx[ow*D+d] = (j*regionStride_ + w*D + d)*sizeof，j=(ow+cl)%block, w=(ow+cl)/block。
    // Init 算一次，所有输出行复用。
    __aicore__ inline void BuildGatherIndex()
    {
        LocalTensor<uint32_t> idx = idxBuf_.Get<uint32_t>();
        uint32_t k = 0;
        for (uint32_t ow = 0; ow < OW_; ++ow) {
            uint32_t widx = ow + cl_;
            uint32_t j = widx % block_;
            uint32_t w = widx / block_;
            uint32_t base = j * regionStride_ + w * D_;        // 源拼接缓冲里的元素下标
            for (uint32_t d = 0; d < D_; ++d) {
                idx.SetValue(k, (base + d) * static_cast<uint32_t>(sizeof(DT_X)));  // 字节偏移
                ++k;
            }
        }
    }

    __aicore__ inline void BuildP10GatherIndex()
    {
        LocalTensor<uint32_t> idx = idxBuf_.Get<uint32_t>();
        uint32_t k = 0;
        uint32_t ctMod = ct_ % block_;
        for (uint32_t localOh = 0; localOh < tileW_; ++localOh) {
            uint32_t hidx = localOh + ct_;
            uint32_t i = hidx % block_;
            uint32_t firstLocal = (i + block_ - ctMod) % block_;
            uint32_t hRel = (localOh - firstLocal) / block_;
            for (uint32_t ow = 0; ow < OW_; ++ow) {
                uint32_t widx = ow + cl_;
                uint32_t j = widx % block_;
                uint32_t w = widx / block_;
                uint32_t base = (i * block_ + j) * p10SlabStride_ + hRel * srcRowElems_ + w * D_;
                for (uint32_t d = 0; d < D_; ++d) {
                    idx.SetValue(k, (base + d) * static_cast<uint32_t>(sizeof(DT_X)));
                    ++k;
                }
            }
        }
    }

    __aicore__ inline void LoadGatherIndex(GM_ADDR rawTiling)
    {
        GlobalTensor<uint32_t> idxGm;
        __gm__ uint8_t *raw = reinterpret_cast<__gm__ uint8_t *>(rawTiling);
        idxGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(raw + sizeof(BatchToSpaceTilingData)), rowElems_);

        LocalTensor<uint32_t> idx = idxBuf_.Get<uint32_t>();
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(rowElems_ * sizeof(uint32_t));
        p.srcStride = 0;
        p.dstStride = 0;
        DataCopyPadExtParams<uint32_t> pad;
        pad.isPad = true;
        pad.leftPadding = 0;
        pad.rightPadding = static_cast<uint8_t>(idxBufElems_ - rowElems_);
        pad.paddingValue = 0;
        DataCopyPad(idx, idxGm[0], p, pad);
        PipeBarrier<PIPE_ALL>();
    }

    // 任务 = 每个输出行 (n,oh)；start_/cnt_ 在 N*OH 上均分。
    __aicore__ inline void ProcessGatherRowPacket()
    {
        uint32_t oh = start_ % OH_;
        uint32_t n = start_ / OH_;
        for (uint32_t idx = 0; idx < cnt_; ++idx) {
            ProcessGatherRow(n, oh);
            if (++oh == OH_) {
                oh = 0;
                ++n;
            }
        }
    }

    __aicore__ inline void ProcessP2GatherV73Packet()
    {
        uint32_t oh = start_ % OH_;
        uint32_t n = start_ / OH_;
        for (uint32_t idx = 0; idx < cnt_; ++idx) {
            ProcessGatherRow(n, oh);
            if (++oh == OH_) {
                oh = 0;
                ++n;
            }
        }
    }

    __aicore__ inline void ProcessGatherRow(uint32_t n, uint32_t oh)
    {
        // oh 反解 i,h（oh∈[0,OH) 必有效 → h<H）
        uint32_t hidx = oh + ct_;
        uint32_t i = hidx % block_;
        uint32_t h = hidx / block_;

        // 1) 读 block 个源行 x[b(j),h,0:W,:]（各 W*D 连续，一次单块读，非逐列）进 src 的 block 个 32B 对齐区
        LocalTensor<DT_X> srcBuf = inQueue_.AllocTensor<DT_X>();
        for (uint32_t j = 0; j < block_; ++j) {
            uint32_t b = (i * block_ + j) * N_ + n;
            uint64_t inOff = (static_cast<uint64_t>(b) * H_ + h) * srcRowElems_;
            DataCopyExtParams p;
            p.blockCount = 1;
            p.blockLen = static_cast<uint32_t>(srcRowElems_ * sizeof(DT_X));
            p.srcStride = 0;
            p.dstStride = 0;
            DataCopyPadExtParams<DT_X> pad;
            pad.isPad = false;
            pad.leftPadding = 0;
            pad.rightPadding = 0;
            pad.paddingValue = 0;
            DataCopyPad(srcBuf[j * regionStride_], xGm_[inOff], p, pad);
        }
        inQueue_.EnQue(srcBuf);

        // 2) Gather 交织成输出行序 packed（核心：VEC 重排，破 sub-32B 写墙）
        LocalTensor<DT_X> src = inQueue_.DeQue<DT_X>();
        LocalTensor<DT_X> out = outQueue_.AllocTensor<DT_X>();
        LocalTensor<uint32_t> idx = idxBuf_.Get<uint32_t>();
        // 需确认 CANN8.5/arch22：按索引 Gather 的签名和语义。常见语义为 out[k]=src[字节偏移 idx[k]]。
        Gather(out, src, idx, static_cast<uint32_t>(0), rowElems_);
        outQueue_.EnQue<DT_X>(out);
        inQueue_.FreeTensor(src);

        // 3) 整行一次写出（OW*D 连续 ≥32B；单块 DataCopyPad，blockLen 任意长）
        LocalTensor<DT_X> y = outQueue_.DeQue<DT_X>();
        uint64_t outOff = (static_cast<uint64_t>(n) * OH_ + oh) * rowElems_;
        DataCopyExtParams wp;
        wp.blockCount = 1;
        wp.blockLen = static_cast<uint32_t>(rowElems_ * sizeof(DT_X));
        wp.srcStride = 0;
        wp.dstStride = 0;
        DataCopyPad(yGm_[outOff], y, wp);
        outQueue_.FreeTensor(y);
    }

    __aicore__ inline void CopyIn(uint64_t inOff, uint32_t curCols, uint32_t curDLen)
    {
        LocalTensor<DT_X> xLocal = inQueue_.AllocTensor<DT_X>();

        if (aligned_) {
            uint32_t total = curCols * curDLen;
            if (dAligned32_) {
                // D 真 32B 对齐：整段连续，用轻量 DataCopy（无 pad 逻辑，scalar 更低）。
                DataCopy(xLocal, xGm_[inOff], total);
            } else {
                // block==1 但 D 非 32B（如 D=65）：单块 DataCopyPad（任意地址/长度，尾部 0 填充到 32B）。
                DataCopyExtParams p;
                p.blockCount = 1;
                p.blockLen = static_cast<uint32_t>(total * sizeof(DT_X));
                p.srcStride = 0;
                p.dstStride = 0;
                DataCopyPadExtParams<DT_X> pad;
                pad.isPad = true;
                pad.leftPadding = 0;
                pad.rightPadding = static_cast<uint8_t>(AlignUp(total) - total);
                pad.paddingValue = 0;
                DataCopyPad(xLocal, xGm_[inOff], p, pad);
            }
        } else {
            // 非对齐：curCols 列一次性搬入 curCols 个 32B 对齐槽（单条多块，源连续 → srcStride=0）。
            DataCopyExtParams params;
            DataCopyPadExtParams<DT_X> padParams;
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.paddingValue = 0;
            params.blockCount = static_cast<uint16_t>(curCols);
            params.blockLen = static_cast<uint32_t>(D_ * sizeof(DT_X));
            params.srcStride = 0;
            params.dstStride = 0;
            padParams.rightPadding = static_cast<uint8_t>(colAligned_ - D_);
            DataCopyPad(xLocal, xGm_[inOff], params, padParams);
        }
        inQueue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t curCols, uint32_t curDLen)
    {
        LocalTensor<DT_X> xLocal = inQueue_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue_.AllocTensor<DT_X>();
        uint32_t len = aligned_ ? AlignUp(curCols * curDLen) : curCols * colAligned_;
        DataCopy(yLocal, xLocal, len);
        outQueue_.EnQue<DT_X>(yLocal);
        inQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutRowSegment(uint64_t outOff, uint32_t curCols, uint32_t curDLen)
    {
        LocalTensor<DT_X> yLocal = outQueue_.DeQue<DT_X>();
        if (aligned_) {
            CopyOutAligned(yLocal, outOff, 0, curCols, curDLen);
        } else {
            CopyOutUnaligned(yLocal, outOff, 0, curCols);
        }
        outQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutFlatTile(uint32_t n, uint32_t i, uint32_t startH, uint32_t startW, uint32_t curCols)
    {
        LocalTensor<DT_X> yLocal = outQueue_.DeQue<DT_X>();

        // 数据点 10 在 W=6 时 tileW=384，即 64 个完整输入行。
        // 每个 flat tile 都从第 0 列开始，可跳过通用行边界循环。
        if (startW == 0 && curCols % W_ == 0) {
            uint32_t rows = curCols / W_;
            uint32_t segCols = curValidW_;
            uint32_t localOff = aligned_ ? curWlo_ * D_ : curWlo_ * colAligned_;

            for (uint32_t step = 0; step < rows; ++step) {
                int64_t oh = static_cast<int64_t>(startH + step) * static_cast<int64_t>(block_) +
                             static_cast<int64_t>(i) - static_cast<int64_t>(ct_);
                if (oh >= 0 && oh < static_cast<int64_t>(OH_)) {
                    uint64_t outOff = (static_cast<uint64_t>(n) * OH_ + static_cast<uint64_t>(oh)) * owd_ +
                                      static_cast<uint64_t>(curOw0_) * D_;
                    if (aligned_) {
                        CopyOutAligned(yLocal, outOff, localOff, segCols, D_);
                    } else {
                        CopyOutUnaligned(yLocal, outOff, localOff, segCols);
                    }
                }
                localOff += aligned_ ? W_ * D_ : W_ * colAligned_;
            }
        } else {
            const uint32_t validEnd = curWlo_ + curValidW_;
            uint32_t remain = curCols;
            uint32_t localCol = 0;
            uint32_t h = startH;
            uint32_t w = startW;

            while (remain != 0 && h < H_) {
                uint32_t rowCols = MinU32(remain, W_ - w);
                uint32_t rowEnd = w + rowCols;
                uint32_t segStart = MaxU32(w, curWlo_);
                uint32_t segEnd = MinU32(rowEnd, validEnd);

                if (segStart < segEnd) {
                    int64_t oh = static_cast<int64_t>(h) * static_cast<int64_t>(block_) +
                                 static_cast<int64_t>(i) - static_cast<int64_t>(ct_);
                    if (oh >= 0 && oh < static_cast<int64_t>(OH_)) {
                        uint32_t skipCols = segStart - w;
                        uint32_t segCols = segEnd - segStart;
                        uint64_t outOff = (static_cast<uint64_t>(n) * OH_ + static_cast<uint64_t>(oh)) * owd_ +
                                          static_cast<uint64_t>(curOw0_ + (segStart - curWlo_) * block_) * D_;
                        uint32_t localOff = aligned_ ? (localCol + skipCols) * D_ :
                                                       (localCol + skipCols) * colAligned_;
                        if (aligned_) {
                            CopyOutAligned(yLocal, outOff, localOff, segCols, D_);
                        } else {
                            CopyOutUnaligned(yLocal, outOff, localOff, segCols);
                        }
                    }
                }

                remain -= rowCols;
                localCol += rowCols;
                ++h;
                w = 0;
            }
        }
        outQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutAligned(LocalTensor<DT_X> &yLocal, uint64_t outOff, uint32_t localOff,
                                          uint32_t curCols, uint32_t curDLen)
    {
        DataCopyExtParams params;
        if (block_ == 1) {
            params.blockCount = 1;
            params.blockLen = static_cast<uint32_t>(curCols * curDLen * sizeof(DT_X));
            params.srcStride = 0;
            params.dstStride = 0;
        } else {
            params.blockCount = static_cast<uint16_t>(curCols);
            params.blockLen = static_cast<uint32_t>(curDLen * sizeof(DT_X));
            params.srcStride = 0;
            params.dstStride = static_cast<uint32_t>((blockD_ - curDLen) * sizeof(DT_X));
        }
        DataCopyPad(yGm_[outOff], yLocal[localOff], params);
    }

    __aicore__ inline void CopyOutUnaligned(LocalTensor<DT_X> &yLocal, uint64_t outOff, uint32_t localOff,
                                            uint32_t curCols)
    {
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(D_ * sizeof(DT_X));
        params.srcStride = 0;
        params.dstStride = 0;

        uint32_t ubOff = localOff;
        uint64_t gmOff = outOff;
        for (uint32_t col = 0; col < curCols; ++col) {
            DataCopyPad(yGm_[gmOff], yLocal[ubOff], params);
            ubOff += colAligned_;
            gmOff += blockD_;
        }
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    TBuf<QuePosition::VECCALC> idxBuf_;       // GATHER_ROW: 行无关的 gather 索引(算一次复用)
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;

    uint32_t N_ = 0, H_ = 0, W_ = 0, D_ = 0, block_ = 0;
    uint32_t ct_ = 0, cl_ = 0, OH_ = 0, OW_ = 0;
    uint32_t totalUnits_ = 0, usedCoreNum_ = 0, tileW_ = 1;
    uint32_t numTilesPerRow_ = 1, flatTiles_ = 1, dTileLen_ = 1, numDTiles_ = 1;
    uint32_t blockElems_ = 8, colAligned_ = 0, bufElems_ = 0;
    uint32_t regionStride_ = 0, rowElems_ = 0, srcRowElems_ = 0;  // GATHER_ROW 用
    uint32_t p10OhTiles_ = 0, p10SlabStride_ = 0;
    uint32_t gatherDepth_ = 1, idxBufElems_ = 0;
    uint64_t blockD_ = 0, owd_ = 0;
    bool aligned_ = true;
    bool dAligned32_ = true;   // D*sizeof 是否 32B 整数倍（决定 CopyIn 用 DataCopy 还是 DataCopyPad）
    uint32_t start_ = 0, cnt_ = 0;
    uint32_t curWlo_ = 0, curValidW_ = 0, curOw0_ = 0;
};

__aicore__ inline bool IsP4ExactTiling(const BatchToSpaceTilingData &t)
{
    return t.aligned == 2;
}

template <typename DT_X>
class KernelBatchToSpaceP4Exact {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &t)
    {
        totalUnits_ = t.totalUnits;
        usedCoreNum_ = t.usedCoreNum;
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, 15360);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, 15360);
        pipe_.InitBuffer(ubBuf_, 768 * sizeof(DT_X));
        SplitWork();
    }

    __aicore__ inline void Process()
    {
        for (uint32_t idx = 0; idx < cnt_; ++idx) {
            ProcessUnit(start_ + idx);
        }
    }

private:
    __aicore__ inline void SplitWork()
    {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t base = totalUnits_ / usedCoreNum_;
        uint32_t rem = totalUnits_ % usedCoreNum_;
        cnt_ = (blockIdx < rem) ? (base + 1) : base;
        start_ = (blockIdx < rem) ? (blockIdx * cnt_) : (rem * (base + 1) + (blockIdx - rem) * base);
    }

    __aicore__ inline void ProcessUnit(uint32_t unit)
    {
        const uint32_t inBase = (unit << 7) + (unit << 6);
        const uint32_t outBase = inBase << 2;

        LocalTensor<DT_X> ub = ubBuf_.Get<DT_X>();
        constexpr uint16_t dBlocks = static_cast<uint16_t>(sizeof(DT_X));
        DataCopyParams p = {6, dBlocks, 0, dBlocks};
        // P4 固定 W=6/D=32，一核处理一个 (n,h)，四个 batch 分块直接交错写入 UB。
        DataCopy(ub[0], xGm_[inBase], p);
        DataCopy(ub[32], xGm_[inBase + 3840], p);
        DataCopy(ub[384], xGm_[inBase + 7680], p);
        DataCopy(ub[416], xGm_[inBase + 11520], p);
        int32_t readEventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(readEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(readEventId);
        DataCopy(yGm_[outBase], ub, 768);
        int32_t writeEventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(writeEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(writeEventId);
    }

private:
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> ubBuf_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t totalUnits_ = 0, usedCoreNum_ = 1, start_ = 0, cnt_ = 0;
};

template <typename DT_X>
class KernelBatchToSpaceB2PairPackDirect {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &t)
    {
        N_ = t.N;
        H_ = t.H;
        W_ = t.W;
        D_ = t.D;
        OH_ = t.OH;
        OW_ = t.OW;
        totalUnits_ = t.totalUnits;
        usedCoreNum_ = t.usedCoreNum;
        rowElems_ = W_ * D_;
        pairElems_ = rowElems_ << 1;

        uint64_t xTotal = static_cast<uint64_t>(N_) * 4 * H_ * W_ * D_;
        uint64_t yTotal = static_cast<uint64_t>(N_) * OH_ * OW_ * D_;
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, xTotal);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, yTotal);

        pipe_.InitBuffer(outBuf_, pairElems_ * sizeof(DT_X));
        SplitWork();
    }

    __aicore__ inline void Process()
    {
        uint32_t rem = start_;
        uint32_t h = rem % H_;
        rem /= H_;
        uint32_t i = rem & 1;
        uint32_t n = rem >> 1;

        for (uint32_t idx = 0; idx < cnt_; ++idx) {
            ProcessPair(n, i, h);
            if (++h == H_) {
                h = 0;
                i ^= 1;
                if (i == 0) {
                    ++n;
                }
            }
        }
    }

private:
    __aicore__ inline void SplitWork()
    {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t base = totalUnits_ / usedCoreNum_;
        uint32_t rem = totalUnits_ % usedCoreNum_;
        cnt_ = (blockIdx < rem) ? (base + 1) : base;
        start_ = (blockIdx < rem) ? (blockIdx * cnt_) : (rem * (base + 1) + (blockIdx - rem) * base);
    }

    __aicore__ inline void ProcessPair(uint32_t n, uint32_t i, uint32_t h)
    {
        LocalTensor<DT_X> out = outBuf_.Get<DT_X>();
        const uint16_t dBlocks = static_cast<uint16_t>((D_ * sizeof(DT_X)) / 32);
        DataCopyParams packParams = {static_cast<uint16_t>(W_), dBlocks, 0, dBlocks};

        for (uint32_t j = 0; j < 2; ++j) {
            const uint32_t b = ((i << 1) + j) * N_ + n;
            const uint64_t inOff = ((static_cast<uint64_t>(b) * H_ + h) * W_) * D_;
            DataCopy(out[j * D_], xGm_[inOff], packParams);
        }

        const uint64_t outOff = (static_cast<uint64_t>(n) * OH_ + ((h << 1) + i)) *
                                static_cast<uint64_t>(OW_) * D_;
        int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        DataCopy(yGm_[outOff], out, pairElems_);
        int32_t reuseEventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(reuseEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(reuseEventId);
    }

private:
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> outBuf_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t N_ = 0, H_ = 0, W_ = 0, D_ = 0, OH_ = 0, OW_ = 0;
    uint32_t totalUnits_ = 0, usedCoreNum_ = 1, rowElems_ = 0, pairElems_ = 0;
    uint32_t start_ = 0, cnt_ = 0;
};

template <typename DT_X, uint32_t MODE>
class KernelBatchToSpaceB2Direct {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &t)
    {
        N_ = t.N;
        H_ = t.H;
        W_ = t.W;
        D_ = t.D;
        OH_ = t.OH;
        OW_ = t.OW;
        dTileLen_ = t.dTileLen;
        numDTiles_ = (D_ + dTileLen_ - 1) / dTileLen_;
        totalUnits_ = t.totalUnits;
        usedCoreNum_ = t.usedCoreNum;
        rowElems_ = W_ * D_;

        uint64_t xTotal = static_cast<uint64_t>(N_) * 4 * H_ * W_ * D_;
        uint64_t yTotal = static_cast<uint64_t>(N_) * OH_ * OW_ * D_;
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, xTotal);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, yTotal);

        SplitWork();
        uint32_t inElems = rowElems_;
        uint32_t outElems = rowElems_;
        if constexpr (MODE == BTS_MODE_P1_B2_DIRECT || MODE == BTS_MODE_P3_B2_DIRECT) {
            outElems = rowElems_ << 1;
        } else if constexpr (MODE == BTS_MODE_P6_B2_DIRECT) {
            inElems = W_ * dTileLen_;
            outElems = inElems;
        }
        pipe_.InitBuffer(inQueue_, BUFFER_NUM, inElems * sizeof(DT_X));
        pipe_.InitBuffer(outQueue_, BUFFER_NUM, outElems * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if constexpr (MODE == BTS_MODE_P6_B2_DIRECT) {
            ProcessDtilePacket();
        } else if constexpr (MODE == BTS_MODE_P1_B2_DIRECT || MODE == BTS_MODE_P3_B2_DIRECT) {
            ProcessPairPackedJPacket();
        } else if constexpr (MODE == BTS_MODE_P4_B2_DIRECT) {
            ProcessPairJPacket();
        } else {
            ProcessSingleJPacket();
        }
    }

private:
    __aicore__ inline void SplitWork()
    {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t base = totalUnits_ / usedCoreNum_;
        uint32_t rem = totalUnits_ % usedCoreNum_;
        cnt_ = (blockIdx < rem) ? (base + 1) : base;
        start_ = (blockIdx < rem) ? (blockIdx * cnt_) : (rem * (base + 1) + (blockIdx - rem) * base);
    }

    __aicore__ inline void ProcessPairJPacket()
    {
        uint32_t rem = start_;
        uint32_t h = rem % H_;
        rem /= H_;
        uint32_t i = rem & 1;
        uint32_t n = rem >> 1;

        for (uint32_t idx = 0; idx < cnt_; ++idx) {
            ProcessRow(n, i, 0, h);
            ProcessRow(n, i, 1, h);
            if (++h == H_) {
                h = 0;
                i ^= 1;
                if (i == 0) {
                    ++n;
                }
            }
        }
    }

    __aicore__ inline void ProcessPairPackedJPacket()
    {
        uint32_t rem = start_;
        uint32_t h = rem % H_;
        rem /= H_;
        uint32_t i = rem & 1;
        uint32_t n = rem >> 1;

        for (uint32_t idx = 0; idx < cnt_; ++idx) {
            ProcessPairPackedRow(n, i, h);
            if (++h == H_) {
                h = 0;
                i ^= 1;
                if (i == 0) {
                    ++n;
                }
            }
        }
    }

    __aicore__ inline void ProcessSingleJPacket()
    {
        uint32_t rem = start_;
        uint32_t h = rem % H_;
        rem /= H_;
        uint32_t j = rem & 1;
        rem >>= 1;
        uint32_t i = rem & 1;
        uint32_t n = rem >> 1;

        for (uint32_t idx = 0; idx < cnt_; ++idx) {
            ProcessRow(n, i, j, h);
            if (++h == H_) {
                h = 0;
                if (++j == 2) {
                    j = 0;
                    if (++i == 2) {
                        i = 0;
                        ++n;
                    }
                }
            }
        }
    }

    __aicore__ inline void ProcessDtilePacket()
    {
        uint32_t rem = start_;
        uint32_t dTile = rem % numDTiles_;
        rem /= numDTiles_;
        uint32_t h = rem % H_;
        rem /= H_;
        uint32_t j = rem & 1;
        rem >>= 1;
        uint32_t i = rem & 1;
        uint32_t n = rem >> 1;

        for (uint32_t idx = 0; idx < cnt_; ++idx) {
            ProcessDtileRow(n, i, j, h, dTile);
            if (++dTile == numDTiles_) {
                dTile = 0;
                if (++h == H_) {
                    h = 0;
                    if (++j == 2) {
                        j = 0;
                        if (++i == 2) {
                            i = 0;
                            ++n;
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void ProcessPairPackedRow(uint32_t n, uint32_t i, uint32_t h)
    {
        const uint32_t pairElems = rowElems_ << 1;
        const uint16_t dBlocks = static_cast<uint16_t>((D_ * sizeof(DT_X)) / 32);
        DataCopyParams packParams = {static_cast<uint16_t>(W_), dBlocks, 0, dBlocks};

        LocalTensor<DT_X> yLocal = outQueue_.AllocTensor<DT_X>();
        for (uint32_t j = 0; j < 2; ++j) {
            const uint32_t b = ((i << 1) + j) * N_ + n;
            const uint64_t inOff = ((static_cast<uint64_t>(b) * H_ + h) * W_) * D_;

            LocalTensor<DT_X> xLocal = inQueue_.AllocTensor<DT_X>();
            DataCopy(xLocal, xGm_[inOff], rowElems_);
            inQueue_.EnQue(xLocal);

            LocalTensor<DT_X> xq = inQueue_.DeQue<DT_X>();
            DataCopy(yLocal[j * D_], xq, packParams);
            inQueue_.FreeTensor(xq);
        }
        outQueue_.EnQue<DT_X>(yLocal);

        const uint64_t outOff = (static_cast<uint64_t>(n) * OH_ + ((h << 1) + i)) *
                                static_cast<uint64_t>(OW_) * D_;
        LocalTensor<DT_X> yq = outQueue_.DeQue<DT_X>();
        DataCopy(yGm_[outOff], yq, pairElems);
        outQueue_.FreeTensor(yq);
    }

    __aicore__ inline void ProcessDtileRow(uint32_t n, uint32_t i, uint32_t j, uint32_t h, uint32_t dTile)
    {
        const uint32_t dStart = dTile * dTileLen_;
        uint32_t curDLen = dTileLen_;
        if (dStart + curDLen > D_) {
            curDLen = D_ - dStart;
        }
        const uint32_t curElems = W_ * curDLen;
        const uint16_t dBlocks = static_cast<uint16_t>((curDLen * sizeof(DT_X)) / 32);
        const uint16_t srcStride = static_cast<uint16_t>(((D_ - curDLen) * sizeof(DT_X)) / 32);
        const uint16_t dstStride = static_cast<uint16_t>(((2 * D_ - curDLen) * sizeof(DT_X)) / 32);

        const uint32_t b = ((i << 1) + j) * N_ + n;
        const uint64_t inOff = ((static_cast<uint64_t>(b) * H_ + h) * W_) * D_ + dStart;
        const uint64_t outOff = (static_cast<uint64_t>(n) * OH_ + ((h << 1) + i)) *
                                static_cast<uint64_t>(OW_) * D_ + static_cast<uint64_t>(j) * D_ + dStart;

        LocalTensor<DT_X> xLocal = inQueue_.AllocTensor<DT_X>();
        DataCopyParams rp = {static_cast<uint16_t>(W_), dBlocks, srcStride, 0};
        DataCopy(xLocal, xGm_[inOff], rp);
        inQueue_.EnQue(xLocal);

        LocalTensor<DT_X> xq = inQueue_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue_.AllocTensor<DT_X>();
        DataCopy(yLocal, xq, curElems);
        outQueue_.EnQue<DT_X>(yLocal);
        inQueue_.FreeTensor(xq);

        LocalTensor<DT_X> yq = outQueue_.DeQue<DT_X>();
        DataCopyParams wp = {static_cast<uint16_t>(W_), dBlocks, 0, dstStride};
        DataCopy(yGm_[outOff], yq, wp);
        outQueue_.FreeTensor(yq);
    }

    __aicore__ inline void ProcessRow(uint32_t n, uint32_t i, uint32_t j, uint32_t h)
    {
        // 小点 direct 只服务 block=2、no-crop、D 32B 对齐数据，省掉通用裁剪和 tail 判断。
        uint32_t b = ((i << 1) + j) * N_ + n;
        uint64_t inOff = ((static_cast<uint64_t>(b) * H_ + h) * W_) * D_;
        uint64_t outOff = (static_cast<uint64_t>(n) * OH_ + ((h << 1) + i)) *
                          static_cast<uint64_t>(OW_) * D_ + static_cast<uint64_t>(j) * D_;

        LocalTensor<DT_X> xLocal = inQueue_.AllocTensor<DT_X>();
        DataCopy(xLocal, xGm_[inOff], rowElems_);
        inQueue_.EnQue(xLocal);

        LocalTensor<DT_X> xq = inQueue_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue_.AllocTensor<DT_X>();
        DataCopy(yLocal, xq, rowElems_);
        outQueue_.EnQue<DT_X>(yLocal);
        inQueue_.FreeTensor(xq);

        LocalTensor<DT_X> yq = outQueue_.DeQue<DT_X>();
        DataCopyExtParams p;
        p.blockCount = static_cast<uint16_t>(W_);
        p.blockLen = static_cast<uint32_t>(D_ * sizeof(DT_X));
        p.srcStride = 0;
        p.dstStride = static_cast<uint32_t>(D_ * sizeof(DT_X));
        DataCopyPad(yGm_[outOff], yq, p);
        outQueue_.FreeTensor(yq);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;

    uint32_t N_ = 0, H_ = 0, W_ = 0, D_ = 0, OH_ = 0, OW_ = 0;
    uint32_t totalUnits_ = 0, usedCoreNum_ = 0, rowElems_ = 0, dTileLen_ = 1, numDTiles_ = 1;
    uint32_t start_ = 0, cnt_ = 0;
};

template <typename DT_X>
class KernelBatchToSpaceP7DDirect {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &t)
    {
        N_ = t.N;
        H_ = t.H;
        D_ = t.D;
        OH_ = t.OH;
        OW_ = t.OW;
        totalUnits_ = t.totalUnits;
        usedCoreNum_ = t.usedCoreNum;
        dTileLen_ = t.dTileLen;
        numDTiles_ = (D_ + dTileLen_ - 1) / dTileLen_;

        uint64_t xTotal = static_cast<uint64_t>(N_) * 4 * H_ * D_;
        uint64_t yTotal = static_cast<uint64_t>(N_) * OH_ * OW_ * D_;
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, xTotal);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, yTotal);

        SplitWork();
        pipe_.InitBuffer(inQueue_, BUFFER_NUM, dTileLen_ * sizeof(DT_X));
        pipe_.InitBuffer(outQueue_, BUFFER_NUM, dTileLen_ * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        uint32_t rem = start_;
        uint32_t dTile = rem % numDTiles_;
        rem /= numDTiles_;
        uint32_t ow = rem & 1;
        rem >>= 1;
        uint32_t oh = rem % OH_;
        uint32_t n = rem / OH_;
        uint64_t baseInOffset = 0;
        uint64_t baseOutOffset = 0;
        CalcBaseOffsets(n, oh, ow, baseInOffset, baseOutOffset);

        for (uint32_t idx = 0; idx < cnt_; ++idx) {
            ProcessTile(baseInOffset, baseOutOffset, dTile);
            if (++dTile == numDTiles_) {
                dTile = 0;
                ow ^= 1;
                if (ow == 0 && ++oh == OH_) {
                    oh = 0;
                    ++n;
                }
                CalcBaseOffsets(n, oh, ow, baseInOffset, baseOutOffset);
            }
        }
    }

private:
    __aicore__ inline void SplitWork()
    {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t base = totalUnits_ / usedCoreNum_;
        uint32_t rem = totalUnits_ % usedCoreNum_;
        cnt_ = (blockIdx < rem) ? (base + 1) : base;
        start_ = (blockIdx < rem) ? (blockIdx * cnt_) : (rem * (base + 1) + (blockIdx - rem) * base);
    }

    __aicore__ inline void CalcBaseOffsets(uint32_t n, uint32_t oh, uint32_t ow,
                                           uint64_t &inOff, uint64_t &outOff)
    {
        uint32_t i = oh & 1;
        uint32_t h = oh >> 1;
        uint32_t b = ((i << 1) + ow) * N_ + n;
        inOff = (static_cast<uint64_t>(b) * H_ + h) * D_;
        outOff = ((static_cast<uint64_t>(n) * OH_ + oh) * OW_ + ow) * D_;
    }

    __aicore__ inline void ProcessTile(uint64_t baseInOffset, uint64_t baseOutOffset, uint32_t dTile)
    {
        uint32_t dStart = dTile * dTileLen_;
        uint32_t curDLen = dTileLen_;
        if (dStart + curDLen > D_) {
            curDLen = D_ - dStart;
        }

        uint64_t inOff = baseInOffset + dStart;
        uint64_t outOff = baseOutOffset + dStart;

        LocalTensor<DT_X> xLocal = inQueue_.AllocTensor<DT_X>();
        DataCopy(xLocal, xGm_[inOff], curDLen);
        inQueue_.EnQue(xLocal);

        LocalTensor<DT_X> xq = inQueue_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue_.AllocTensor<DT_X>();
        DataCopy(yLocal, xq, curDLen);
        outQueue_.EnQue<DT_X>(yLocal);
        inQueue_.FreeTensor(xq);

        LocalTensor<DT_X> yq = outQueue_.DeQue<DT_X>();
        DataCopy(yGm_[outOff], yq, curDLen);
        outQueue_.FreeTensor(yq);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;

    uint32_t N_ = 0, H_ = 0, D_ = 0, OH_ = 0, OW_ = 0;
    uint32_t totalUnits_ = 0, usedCoreNum_ = 0, dTileLen_ = 1, numDTiles_ = 1;
    uint32_t start_ = 0, cnt_ = 0;
};

template <typename DT_X>
class KernelBatchToSpaceP7FlatCopy {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &t)
    {
        totalElems_ = static_cast<uint32_t>(static_cast<uint64_t>(t.N) * t.OH * t.OW * t.D);
        tileElems_ = t.tileW;
        totalUnits_ = t.totalUnits;
        usedCoreNum_ = t.usedCoreNum;

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, totalElems_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, totalElems_);

        SplitWork();
        pipe_.InitBuffer(copyBuf_, tileElems_ * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        uint32_t tileIdx = start_;
        for (uint32_t idx = 0; idx < cnt_; ++idx, ++tileIdx) {
            const uint32_t offset = tileIdx * tileElems_;
            uint32_t len = tileElems_;
            if (offset + len > totalElems_) {
                len = totalElems_ - offset;
            }
            LocalTensor<DT_X> local = copyBuf_.Get<DT_X>();
            DataCopy(local, xGm_[offset], len);
            int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
            DataCopy(yGm_[offset], local, len);
            int32_t reuseEventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(reuseEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(reuseEventId);
        }
    }

private:
    __aicore__ inline void SplitWork()
    {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t base = totalUnits_ / usedCoreNum_;
        uint32_t rem = totalUnits_ % usedCoreNum_;
        cnt_ = (blockIdx < rem) ? (base + 1) : base;
        start_ = (blockIdx < rem) ? (blockIdx * cnt_) : (rem * (base + 1) + (blockIdx - rem) * base);
    }

private:
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> copyBuf_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;

    uint32_t totalElems_ = 0, tileElems_ = 1, totalUnits_ = 0, usedCoreNum_ = 0;
    uint32_t start_ = 0, cnt_ = 0;
};

template <typename DT_X>
class KernelBatchToSpaceP1V71 {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &t)
    {
        N_ = t.N;
        H_ = t.H;
        W_ = t.W;
        D_ = t.D;
        block_ = t.block;
        ct_ = t.ct;
        cl_ = t.cl;
        OH_ = t.OH;
        OW_ = t.OW;
        totalUnits_ = t.totalUnits;
        usedCoreNum_ = t.usedCoreNum;
        tileW_ = t.tileW;
        dTileLen_ = t.dTileLen;
        aligned_ = t.aligned != 0;
        dAligned32_ = (((uint64_t)D_ * sizeof(DT_X)) % 32 == 0);

        blockElems_ = 32 / sizeof(DT_X);
        colAligned_ = AlignUp(D_);
        blockD_ = static_cast<uint64_t>(block_) * D_;
        owd_ = static_cast<uint64_t>(OW_) * D_;
        numTilesPerRow_ = (W_ + tileW_ - 1) / tileW_;
        bufElems_ = aligned_ ? AlignUp(tileW_ * dTileLen_) : tileW_ * colAligned_;

        uint64_t xTotal = static_cast<uint64_t>(N_) * block_ * block_ * H_ * W_ * D_;
        uint64_t yTotal = static_cast<uint64_t>(N_) * OH_ * OW_ * D_;
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, xTotal);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, yTotal);

        SplitWork();
        pipe_.InitBuffer(inQueue_, BUFFER_NUM, bufElems_ * sizeof(DT_X));
        pipe_.InitBuffer(outQueue_, BUFFER_NUM, bufElems_ * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        ProcessRowPacket();
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t x)
    {
        return (x + blockElems_ - 1) / blockElems_ * blockElems_;
    }

    __aicore__ inline void SplitWork()
    {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t base = totalUnits_ / usedCoreNum_;
        uint32_t rem = totalUnits_ % usedCoreNum_;
        cnt_ = (blockIdx < rem) ? (base + 1) : base;
        start_ = (blockIdx < rem) ? (blockIdx * cnt_) : (rem * (base + 1) + (blockIdx - rem) * base);
    }

    __aicore__ inline void ComputeWRange(uint32_t j)
    {
        curWlo_ = 0;
        curValidW_ = W_;
        curOw0_ = j;
    }

    __aicore__ inline void ProcessRowPacket()
    {
        uint32_t rem = start_;
        uint32_t tileIdx = rem % numTilesPerRow_;
        rem /= numTilesPerRow_;
        uint32_t h = rem % H_;
        rem /= H_;
        uint32_t i = rem % block_;
        rem /= block_;
        uint32_t n = rem;

        for (uint32_t idx = 0; idx < cnt_; ++idx) {
            for (uint32_t j = 0; j < block_; ++j) {
                ComputeWRange(j);
                ProcessRowTile(n, i, j, h, tileIdx);
            }
            if (++tileIdx == numTilesPerRow_) {
                tileIdx = 0;
                if (++h == H_) {
                    h = 0;
                    if (++i == block_) {
                        i = 0;
                        ++n;
                    }
                }
            }
        }
    }

    __aicore__ inline void ProcessRowTile(uint32_t n, uint32_t i, uint32_t j, uint32_t h, uint32_t tileIdx)
    {
        int64_t oh = static_cast<int64_t>(h) * static_cast<int64_t>(block_) +
                     static_cast<int64_t>(i) - static_cast<int64_t>(ct_);

        uint32_t wStart = curWlo_ + tileIdx * tileW_;
        uint32_t wEnd = wStart + tileW_;
        uint32_t validEnd = curWlo_ + curValidW_;
        if (wEnd > validEnd) {
            wEnd = validEnd;
        }
        uint32_t curW = wEnd - wStart;
        uint32_t b = (i * block_ + j) * N_ + n;
        uint64_t inOff = ((static_cast<uint64_t>(b) * H_ + h) * W_ + wStart) * D_;
        uint64_t outOff = (static_cast<uint64_t>(n) * OH_ + static_cast<uint64_t>(oh)) * owd_ +
                          static_cast<uint64_t>(curOw0_ + (wStart - curWlo_) * block_) * D_;

        CopyIn(inOff, curW, D_);
        Compute(curW, D_);
        CopyOutRowSegment(outOff, curW, D_);
    }

    __aicore__ inline void CopyIn(uint64_t inOff, uint32_t curCols, uint32_t curDLen)
    {
        LocalTensor<DT_X> xLocal = inQueue_.AllocTensor<DT_X>();
        if (aligned_) {
            uint32_t total = curCols * curDLen;
            if (dAligned32_) {
                DataCopy(xLocal, xGm_[inOff], total);
            } else {
                DataCopyExtParams p;
                p.blockCount = 1;
                p.blockLen = static_cast<uint32_t>(total * sizeof(DT_X));
                p.srcStride = 0;
                p.dstStride = 0;
                DataCopyPadExtParams<DT_X> pad;
                pad.isPad = true;
                pad.leftPadding = 0;
                pad.rightPadding = static_cast<uint8_t>(AlignUp(total) - total);
                pad.paddingValue = 0;
                DataCopyPad(xLocal, xGm_[inOff], p, pad);
            }
        } else {
            DataCopyExtParams params;
            DataCopyPadExtParams<DT_X> padParams;
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.paddingValue = 0;
            params.blockCount = static_cast<uint16_t>(curCols);
            params.blockLen = static_cast<uint32_t>(D_ * sizeof(DT_X));
            params.srcStride = 0;
            params.dstStride = 0;
            padParams.rightPadding = static_cast<uint8_t>(colAligned_ - D_);
            DataCopyPad(xLocal, xGm_[inOff], params, padParams);
        }
        inQueue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t curCols, uint32_t curDLen)
    {
        LocalTensor<DT_X> xLocal = inQueue_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue_.AllocTensor<DT_X>();
        uint32_t len = aligned_ ? AlignUp(curCols * curDLen) : curCols * colAligned_;
        DataCopy(yLocal, xLocal, len);
        outQueue_.EnQue<DT_X>(yLocal);
        inQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutRowSegment(uint64_t outOff, uint32_t curCols, uint32_t curDLen)
    {
        LocalTensor<DT_X> yLocal = outQueue_.DeQue<DT_X>();
        if (aligned_) {
            CopyOutAligned(yLocal, outOff, 0, curCols, curDLen);
        } else {
            CopyOutUnaligned(yLocal, outOff, 0, curCols);
        }
        outQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutAligned(LocalTensor<DT_X> &yLocal, uint64_t outOff, uint32_t localOff,
                                          uint32_t curCols, uint32_t curDLen)
    {
        DataCopyExtParams params;
        if (block_ == 1) {
            params.blockCount = 1;
            params.blockLen = static_cast<uint32_t>(curCols * curDLen * sizeof(DT_X));
            params.srcStride = 0;
            params.dstStride = 0;
        } else {
            params.blockCount = static_cast<uint16_t>(curCols);
            params.blockLen = static_cast<uint32_t>(curDLen * sizeof(DT_X));
            params.srcStride = 0;
            params.dstStride = static_cast<uint32_t>((blockD_ - curDLen) * sizeof(DT_X));
        }
        DataCopyPad(yGm_[outOff], yLocal[localOff], params);
    }

    __aicore__ inline void CopyOutUnaligned(LocalTensor<DT_X> &yLocal, uint64_t outOff, uint32_t localOff,
                                            uint32_t curCols)
    {
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(D_ * sizeof(DT_X));
        params.srcStride = 0;
        params.dstStride = 0;

        uint32_t ubOff = localOff;
        uint64_t gmOff = outOff;
        for (uint32_t col = 0; col < curCols; ++col) {
            DataCopyPad(yGm_[gmOff], yLocal[ubOff], params);
            ubOff += colAligned_;
            gmOff += blockD_;
        }
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;

    uint32_t N_ = 0, H_ = 0, W_ = 0, D_ = 0, block_ = 0;
    uint32_t ct_ = 0, cl_ = 0, OH_ = 0, OW_ = 0;
    uint32_t totalUnits_ = 0, usedCoreNum_ = 0, tileW_ = 1;
    uint32_t numTilesPerRow_ = 1, dTileLen_ = 1;
    uint32_t blockElems_ = 8, colAligned_ = 0, bufElems_ = 0;
    uint64_t blockD_ = 0, owd_ = 0;
    bool aligned_ = true;
    bool dAligned32_ = true;
    uint32_t start_ = 0, cnt_ = 0;
    uint32_t curWlo_ = 0, curValidW_ = 0, curOw0_ = 0;
};

template <typename DT_X, uint32_t MODE>
class KernelBatchToSpaceV73 {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &t)
    {
        N_ = t.N;
        H_ = t.H;
        W_ = t.W;
        D_ = t.D;
        block_ = t.block;
        ct_ = t.ct;
        cl_ = t.cl;
        OH_ = t.OH;
        OW_ = t.OW;
        totalUnits_ = t.totalUnits;
        usedCoreNum_ = t.usedCoreNum;
        tileW_ = t.tileW;
        dTileLen_ = t.dTileLen;
        aligned_ = t.aligned != 0;
        dAligned32_ = (((uint64_t)D_ * sizeof(DT_X)) % 32 == 0);

        blockElems_ = 32 / sizeof(DT_X);
        colAligned_ = AlignUp(D_);
        blockD_ = static_cast<uint64_t>(block_) * D_;
        owd_ = static_cast<uint64_t>(OW_) * D_;
        numTilesPerRow_ = (W_ + tileW_ - 1) / tileW_;

        uint64_t xTotal = static_cast<uint64_t>(N_) * block_ * block_ * H_ * W_ * D_;
        uint64_t yTotal = static_cast<uint64_t>(N_) * OH_ * OW_ * D_;
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, xTotal);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, yTotal);

        SplitWork();
        if constexpr (MODE == BTS_MODE_P2_GATHER_V73) {
            srcRowElems_ = W_ * D_;
            regionStride_ = AlignUp(srcRowElems_);
            rowElems_ = OW_ * D_;
            pipe_.InitBuffer(inQueue_, BUFFER_NUM, (uint32_t)block_ * regionStride_ * sizeof(DT_X));
            pipe_.InitBuffer(outQueue_, BUFFER_NUM, rowElems_ * sizeof(DT_X));
            pipe_.InitBuffer(idxBuf_, rowElems_ * sizeof(uint32_t));
            BuildGatherIndex();
        } else {
            bufElems_ = aligned_ ? AlignUp(tileW_ * dTileLen_) : tileW_ * colAligned_;
            pipe_.InitBuffer(inQueue_, BUFFER_NUM, bufElems_ * sizeof(DT_X));
            pipe_.InitBuffer(outQueue_, BUFFER_NUM, bufElems_ * sizeof(DT_X));
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (MODE == BTS_MODE_P2_GATHER_V73) {
            ProcessGatherRowPacket();
        } else {
            ProcessRowTilePacket();
        }
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t x)
    {
        return (x + blockElems_ - 1) / blockElems_ * blockElems_;
    }

    __aicore__ inline void SplitWork()
    {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t base = totalUnits_ / usedCoreNum_;
        uint32_t rem = totalUnits_ % usedCoreNum_;
        cnt_ = (blockIdx < rem) ? (base + 1) : base;
        start_ = (blockIdx < rem) ? (blockIdx * cnt_) : (rem * (base + 1) + (blockIdx - rem) * base);
    }

    __aicore__ inline void ComputeWRange(uint32_t j)
    {
        curWlo_ = 0;
        curValidW_ = W_;
        curOw0_ = j;
    }

    __aicore__ inline void ProcessRowTilePacket()
    {
        uint32_t rem = start_;
        uint32_t tileIdx = rem % numTilesPerRow_;
        rem /= numTilesPerRow_;
        uint32_t h = rem % H_;
        rem /= H_;
        uint32_t j = rem % block_;
        rem /= block_;
        uint32_t i = rem % block_;
        rem /= block_;
        uint32_t n = rem;
        ComputeWRange(j);

        for (uint32_t idx = 0; idx < cnt_; ++idx) {
            ProcessRowTile(n, i, j, h, tileIdx);
            if (++tileIdx == numTilesPerRow_) {
                tileIdx = 0;
                if (++h == H_) {
                    h = 0;
                    if (++j == block_) {
                        j = 0;
                        if (++i == block_) {
                            i = 0;
                            ++n;
                        }
                    }
                    ComputeWRange(j);
                }
            }
        }
    }

    __aicore__ inline void ProcessRowTile(uint32_t n, uint32_t i, uint32_t j, uint32_t h, uint32_t tileIdx)
    {
        int64_t oh = static_cast<int64_t>(h) * static_cast<int64_t>(block_) +
                     static_cast<int64_t>(i) - static_cast<int64_t>(ct_);

        uint32_t wStart = curWlo_ + tileIdx * tileW_;
        uint32_t wEnd = wStart + tileW_;
        uint32_t validEnd = curWlo_ + curValidW_;
        if (wEnd > validEnd) {
            wEnd = validEnd;
        }

        uint32_t curW = wEnd - wStart;
        uint32_t b = (i * block_ + j) * N_ + n;
        uint64_t inOff = ((static_cast<uint64_t>(b) * H_ + h) * W_ + wStart) * D_;
        uint64_t outOff = (static_cast<uint64_t>(n) * OH_ + static_cast<uint64_t>(oh)) * owd_ +
                          static_cast<uint64_t>(curOw0_ + (wStart - curWlo_) * block_) * D_;

        CopyIn(inOff, curW, D_);
        Compute(curW, D_);
        CopyOutRowSegment(outOff, curW, D_);
    }

    __aicore__ inline void BuildGatherIndex()
    {
        LocalTensor<uint32_t> idx = idxBuf_.Get<uint32_t>();
        uint32_t k = 0;
        uint32_t j = cl_ & 1;
        uint32_t w = cl_ >> 1;
        for (uint32_t ow = 0; ow < OW_; ++ow) {
            uint32_t base = j * regionStride_ + w * D_;
            for (uint32_t d = 0; d < D_; ++d) {
                idx.SetValue(k, (base + d) * static_cast<uint32_t>(sizeof(DT_X)));
                ++k;
            }
            j ^= 1;
            if (j == 0) {
                ++w;
            }
        }
    }

    __aicore__ inline void ProcessGatherRowPacket()
    {
        uint32_t oh = start_ % OH_;
        uint32_t n = start_ / OH_;
        for (uint32_t idx = 0; idx < cnt_; ++idx) {
            ProcessGatherRow(n, oh);
            if (++oh == OH_) {
                oh = 0;
                ++n;
            }
        }
    }

    __aicore__ inline void ProcessGatherRow(uint32_t n, uint32_t oh)
    {
        uint32_t hidx = oh + ct_;
        uint32_t i = hidx % block_;
        uint32_t h = hidx / block_;

        LocalTensor<DT_X> srcBuf = inQueue_.AllocTensor<DT_X>();
        for (uint32_t j = 0; j < block_; ++j) {
            uint32_t b = (i * block_ + j) * N_ + n;
            uint64_t inOff = (static_cast<uint64_t>(b) * H_ + h) * srcRowElems_;
            DataCopyExtParams p;
            p.blockCount = 1;
            p.blockLen = static_cast<uint32_t>(srcRowElems_ * sizeof(DT_X));
            p.srcStride = 0;
            p.dstStride = 0;
            DataCopyPadExtParams<DT_X> pad;
            pad.isPad = false;
            pad.leftPadding = 0;
            pad.rightPadding = 0;
            pad.paddingValue = 0;
            DataCopyPad(srcBuf[j * regionStride_], xGm_[inOff], p, pad);
        }
        inQueue_.EnQue(srcBuf);

        LocalTensor<DT_X> src = inQueue_.DeQue<DT_X>();
        LocalTensor<DT_X> out = outQueue_.AllocTensor<DT_X>();
        LocalTensor<uint32_t> idx = idxBuf_.Get<uint32_t>();
        Gather(out, src, idx, static_cast<uint32_t>(0), rowElems_);
        outQueue_.EnQue<DT_X>(out);
        inQueue_.FreeTensor(src);

        LocalTensor<DT_X> y = outQueue_.DeQue<DT_X>();
        uint64_t outOff = (static_cast<uint64_t>(n) * OH_ + oh) * rowElems_;
        DataCopyExtParams wp;
        wp.blockCount = 1;
        wp.blockLen = static_cast<uint32_t>(rowElems_ * sizeof(DT_X));
        wp.srcStride = 0;
        wp.dstStride = 0;
        DataCopyPad(yGm_[outOff], y, wp);
        outQueue_.FreeTensor(y);
    }

    __aicore__ inline void CopyIn(uint64_t inOff, uint32_t curCols, uint32_t curDLen)
    {
        LocalTensor<DT_X> xLocal = inQueue_.AllocTensor<DT_X>();
        if (aligned_) {
            uint32_t total = curCols * curDLen;
            if (dAligned32_) {
                DataCopy(xLocal, xGm_[inOff], total);
            } else {
                DataCopyExtParams p;
                p.blockCount = 1;
                p.blockLen = static_cast<uint32_t>(total * sizeof(DT_X));
                p.srcStride = 0;
                p.dstStride = 0;
                DataCopyPadExtParams<DT_X> pad;
                pad.isPad = true;
                pad.leftPadding = 0;
                pad.rightPadding = static_cast<uint8_t>(AlignUp(total) - total);
                pad.paddingValue = 0;
                DataCopyPad(xLocal, xGm_[inOff], p, pad);
            }
        } else {
            DataCopyExtParams params;
            DataCopyPadExtParams<DT_X> padParams;
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.paddingValue = 0;
            params.blockCount = static_cast<uint16_t>(curCols);
            params.blockLen = static_cast<uint32_t>(D_ * sizeof(DT_X));
            params.srcStride = 0;
            params.dstStride = 0;
            padParams.rightPadding = static_cast<uint8_t>(colAligned_ - D_);
            DataCopyPad(xLocal, xGm_[inOff], params, padParams);
        }
        inQueue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t curCols, uint32_t curDLen)
    {
        LocalTensor<DT_X> xLocal = inQueue_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueue_.AllocTensor<DT_X>();
        uint32_t len = aligned_ ? AlignUp(curCols * curDLen) : curCols * colAligned_;
        DataCopy(yLocal, xLocal, len);
        outQueue_.EnQue<DT_X>(yLocal);
        inQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutRowSegment(uint64_t outOff, uint32_t curCols, uint32_t curDLen)
    {
        LocalTensor<DT_X> yLocal = outQueue_.DeQue<DT_X>();
        if (aligned_) {
            CopyOutAligned(yLocal, outOff, 0, curCols, curDLen);
        } else {
            CopyOutUnaligned(yLocal, outOff, 0, curCols);
        }
        outQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutAligned(LocalTensor<DT_X> &yLocal, uint64_t outOff, uint32_t localOff,
                                          uint32_t curCols, uint32_t curDLen)
    {
        DataCopyExtParams params;
        if (block_ == 1) {
            params.blockCount = 1;
            params.blockLen = static_cast<uint32_t>(curCols * curDLen * sizeof(DT_X));
            params.srcStride = 0;
            params.dstStride = 0;
        } else {
            params.blockCount = static_cast<uint16_t>(curCols);
            params.blockLen = static_cast<uint32_t>(curDLen * sizeof(DT_X));
            params.srcStride = 0;
            params.dstStride = static_cast<uint32_t>((blockD_ - curDLen) * sizeof(DT_X));
        }
        DataCopyPad(yGm_[outOff], yLocal[localOff], params);
    }

    __aicore__ inline void CopyOutUnaligned(LocalTensor<DT_X> &yLocal, uint64_t outOff, uint32_t localOff,
                                            uint32_t curCols)
    {
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(D_ * sizeof(DT_X));
        params.srcStride = 0;
        params.dstStride = 0;

        uint32_t ubOff = localOff;
        uint64_t gmOff = outOff;
        for (uint32_t col = 0; col < curCols; ++col) {
            DataCopyPad(yGm_[gmOff], yLocal[ubOff], params);
            ubOff += colAligned_;
            gmOff += blockD_;
        }
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    TBuf<QuePosition::VECCALC> idxBuf_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;

    uint32_t N_ = 0, H_ = 0, W_ = 0, D_ = 0, block_ = 0;
    uint32_t ct_ = 0, cl_ = 0, OH_ = 0, OW_ = 0;
    uint32_t totalUnits_ = 0, usedCoreNum_ = 0, tileW_ = 1;
    uint32_t numTilesPerRow_ = 1, dTileLen_ = 1;
    uint32_t blockElems_ = 8, colAligned_ = 0, bufElems_ = 0;
    uint32_t regionStride_ = 0, rowElems_ = 0, srcRowElems_ = 0;
    uint64_t blockD_ = 0, owd_ = 0;
    bool aligned_ = true;
    bool dAligned32_ = true;
    uint32_t start_ = 0, cnt_ = 0;
    uint32_t curWlo_ = 0, curValidW_ = 0, curOw0_ = 0;
};

template <typename DT_X, uint32_t MODE>
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GM_ADDR rawTiling = tiling;
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingDataBuf);
    GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingData, tiling_data, tiling);
    if constexpr (MODE == BTS_MODE_P1_B2_DIRECT || MODE == BTS_MODE_P3_B2_DIRECT) {
        KernelBatchToSpaceB2PairPackDirect<DT_X> op;
        op.Init(x, y, tiling_data);
        op.Process();
    } else if constexpr (MODE == BTS_MODE_P4_B2_DIRECT || MODE == BTS_MODE_P6_B2_DIRECT) {
        KernelBatchToSpaceB2Direct<DT_X, MODE> op;
        op.Init(x, y, tiling_data);
        op.Process();
    } else if constexpr (MODE == BTS_MODE_P7_D_DIRECT) {
        KernelBatchToSpaceP7DDirect<DT_X> op;
        op.Init(x, y, tiling_data);
        op.Process();
    } else if constexpr (MODE == BTS_MODE_P7_FLAT_COPY) {
        KernelBatchToSpaceP7FlatCopy<DT_X> op;
        op.Init(x, y, tiling_data);
        op.Process();
    } else if constexpr (MODE == BTS_MODE_P1_ROW_V71) {
        KernelBatchToSpaceP1V71<DT_X> op;
        op.Init(x, y, tiling_data);
        op.Process();
    } else if constexpr (MODE == BTS_MODE_P2_GATHER_V73 || MODE == BTS_MODE_P3_ROW_V73) {
        KernelBatchToSpaceV73<DT_X, MODE> op;
        op.Init(x, y, tiling_data);
        op.Process();
    } else if constexpr (MODE == BTS_MODE_B4_W_PACK_TILE) {
        if (IsP4ExactTiling(tiling_data)) {
            KernelBatchToSpaceP4Exact<DT_X> op;
            op.Init(x, y, tiling_data);
            op.Process();
        } else {
            KernelBatchToSpace<DT_X, MODE> op;
            op.Init(x, y, rawTiling, tiling_data);
            op.Process();
        }
    } else {
        KernelBatchToSpace<DT_X, MODE> op;
        op.Init(x, y, rawTiling, tiling_data);
        op.Process();
    }
}
