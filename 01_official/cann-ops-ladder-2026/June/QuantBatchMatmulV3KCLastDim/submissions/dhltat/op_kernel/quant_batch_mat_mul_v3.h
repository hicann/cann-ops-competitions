/*!
 * \file quant_batch_mat_mul_v3.h
 * \brief QuantBatchMatMulV3 kernel implementation.
 */

#ifndef QUANTBATCHMATMULV3_H
#define QUANTBATCHMATMULV3_H

#include "quant_batch_mat_mul_v3_tiling_data.h"
#include "quant_batch_mat_mul_v3_tiling_key.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/quantization/ascend_dequant.h"

namespace NsQuantBatchMatMulV3 {

using namespace AscendC;

constexpr auto QBM_CFG_MDL_INTRINSICS = GetMDLConfig(true);

template <bool TRANS_X1, bool TRANS_X2, bool VECTOR_LARGE_M = false>
class QuantBatchMatMulV3Matmul {
public:
    using AMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t, TRANS_X1>;
    using BMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t, TRANS_X2>;
    using CMatmulType = matmul::MatmulType<TPosition::VECIN, CubeFormat::ND, int32_t>;
    using BiasMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;

    matmul::Matmul<AMatmulType, BMatmulType, CMatmulType, BiasMatmulType, QBM_CFG_MDL_INTRINSICS> mm;

    __aicore__ inline QuantBatchMatMulV3Matmul() {}

    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR scale,
        GM_ADDR pertokenScale,
        GM_ADDR out,
        GM_ADDR userWorkspace,
        TPipe* pipe,
        const QuantBatchMatMulV3TilingData* tilingData)
    {
        blockIdx_ = static_cast<uint32_t>(GetBlockIdx());
        pipe_ = pipe;
        tilingData_ = tilingData;
        if (GetSubBlockIdx() > 0) {
            return;
        }
        m_ = static_cast<uint32_t>(tilingData->m);
        n_ = static_cast<uint32_t>(tilingData->n);
        k_ = static_cast<uint32_t>(tilingData->k);
        scaleLen_ = static_cast<uint32_t>(tilingData->scaleLen);
        vectorLargeM_ = VECTOR_LARGE_M && TRANS_X1 && !TRANS_X2 && m_ == 65536U && n_ == 16U && k_ == 256U;
        singleCoreM_ = static_cast<uint32_t>(tilingData->matmulTiling.singleCoreM);
        singleCoreN_ = static_cast<uint32_t>(tilingData->matmulTiling.singleCoreN);
        singleCoreK_ = static_cast<uint32_t>(tilingData->matmulTiling.singleCoreK);
        baseM_ = static_cast<uint32_t>(tilingData->matmulTiling.baseM);
        baseN_ = static_cast<uint32_t>(tilingData->matmulTiling.baseN);
        usedCoreNum_ = static_cast<uint32_t>(tilingData->matmulTiling.usedCoreNum);
        isMOuter_ = tilingData->matmulTiling.iterateOrder == 0;
        workspacePerCore_ = tilingData->workspacePerCore;

        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1),
            static_cast<uint64_t>(tilingData->x1Dim0) * static_cast<uint64_t>(tilingData->x1Dim1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x2),
            static_cast<uint64_t>(tilingData->x2Dim0) * static_cast<uint64_t>(tilingData->x2Dim1));
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale), scaleLen_);
        pertokenScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pertokenScale), m_);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out), static_cast<uint64_t>(m_) * n_);

        if (singleCoreK_ == 0) {
            singleCoreK_ = k_;
        }
        if (blockIdx_ < usedCoreNum_) {
            GM_ADDR coreWorkspace = userWorkspace + static_cast<uint64_t>(blockIdx_) * workspacePerCore_;
            const uint64_t normalWorkspace =
                AlignUp64(static_cast<uint64_t>(baseM_) * baseN_ * sizeof(int32_t), WORKSPACE_ALIGN);
            matmulWorkspacePerCore_ = normalWorkspace;
            cWorkspacePerCore_ = matmulWorkspacePerCore_;
            aCompactStride_ = AlignUp(baseM_, GM_BLOCK_BYTES);
            if constexpr (!VECTOR_LARGE_M) {
                mm.SetWorkspace(coreWorkspace, matmulWorkspacePerCore_);
            }
            mmOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(coreWorkspace + matmulWorkspacePerCore_),
                cWorkspacePerCore_ / sizeof(int32_t));
            aCompactGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ int8_t*>(coreWorkspace + matmulWorkspacePerCore_ + cWorkspacePerCore_),
                static_cast<uint64_t>(A_COMPACT_K_STEP) * aCompactStride_);
        }

        ubNAligned_ = AlignUp(baseN_, BF16_BLOCK_ELEMS);
        if (pipe_ != nullptr && blockIdx_ < usedCoreNum_ && ubNAligned_ > 0) {
            const bool singleColumnVector = !TRANS_X1 && !TRANS_X2 && n_ == 1U && k_ > MATMUL_K_LIMIT;
            const bool largeKTransBVector = !TRANS_X1 && TRANS_X2 && k_ > MATMUL_K_LIMIT &&
                m_ <= LARGE_K_TRANS_B_VECTOR_M_LIMIT && n_ <= LARGE_K_TRANS_B_VECTOR_N_LIMIT;
            const bool largeMVector = TRANS_X1 && m_ > MATMUL_M_LIMIT && k_ <= LARGE_M_VECTOR_K_LIMIT &&
                n_ <= LARGE_M_VECTOR_N_LIMIT;
            const bool directOriginalStride = NeedCase10OriginalStride() || NeedCase11OriginalStride() ||
                NeedCase12OriginalStride() || NeedCase13OriginalStride() || NeedCase14OriginalStride() ||
                NeedCase15OriginalStride();
            const uint32_t vectorKTile = singleColumnVector ? SINGLE_COLUMN_VECTOR_K_TILE :
                (largeKTransBVector ? LARGE_K_TRANS_B_VECTOR_K_TILE : 0U);
            const uint32_t compactCopyRows = VECTOR_LARGE_M ? 1U : A_COMPACT_COPY_ROWS;
            const uint32_t directElems = directOriginalStride ? DIRECT_AIC_ROW_STEP * ubNAligned_ : 0U;
            const uint32_t srcElems = Max(Max(Max(ubNAligned_, AlignUp(baseM_, FP32_BLOCK_ELEMS)), vectorKTile),
                directElems);
            const uint32_t int8Elems = Max(Max(compactCopyRows * AlignUp(baseM_, GM_BLOCK_BYTES),
                AlignUp(baseN_, GM_BLOCK_BYTES)), AlignUp(vectorKTile, GM_BLOCK_BYTES));
            pipe_->InitBuffer(vecQueA_, BUFFER_NUM,
                int8Elems * sizeof(int8_t));
            const bool largeMTransBCache = TRANS_X2 && largeMVector &&
                n_ * k_ <= LARGE_M_TRANS_B_CACHE_ELEMS;
            const uint32_t srcBytes = Max(srcElems * static_cast<uint32_t>(sizeof(int32_t)),
                largeMTransBCache ? LARGE_M_TRANS_B_CACHE_ELEMS * static_cast<uint32_t>(sizeof(int8_t)) : 0U);
            pipe_->InitBuffer(vecQueSrc_, BUFFER_NUM, srcBytes);
            pipe_->InitBuffer(vecQueScale_, BUFFER_NUM, ubNAligned_ * sizeof(float));
            uint32_t outElems = (VECTOR_LARGE_M || largeMVector) ?
                Max(ubNAligned_, AlignUp(baseM_, BF16_BLOCK_ELEMS)) : ubNAligned_;
            if (largeMVector && n_ >= 64U) {
                outElems = Max(outElems, LARGE_M_BLOCK16_OUT_ELEMS);
            }
            if (directOriginalStride) {
                outElems = Max(outElems, directElems);
            }
            pipe_->InitBuffer(vecQueOut_, BUFFER_NUM, outElems * sizeof(bfloat16_t));
            pipe_->InitBuffer(vecFp32Tmp_, srcElems * sizeof(float));
            pipe_->InitBuffer(vecHalfTmp_, Max(Max(AlignUp(baseM_, BF16_BLOCK_ELEMS), ubNAligned_), vectorKTile) *
                sizeof(half));
            if (!TRANS_X2 && n_ > MATMUL_N_LIMIT && k_ <= LARGE_N_VECTOR_K_LIMIT) {
                pipe_->InitBuffer(vecLargeNB_, ubNAligned_ * sizeof(float));
            }
            if constexpr (TRANS_X1) {
                pipe_->InitBuffer(vecTokenScale_, AlignUp(baseM_, FP32_BLOCK_ELEMS) * sizeof(float));
            } else {
                if (NeedCase15OriginalStride()) {
                    pipe_->InitBuffer(vecTokenScale_, AlignUp(baseM_, FP32_BLOCK_ELEMS) * sizeof(float));
                }
            }
            uint64_t accumElems = static_cast<uint64_t>(AlignUp(baseM_, FP32_BLOCK_ELEMS)) * ubNAligned_;
            if (vectorKTile > 0U) {
                accumElems += static_cast<uint64_t>(vectorKTile) * 3U + FP32_BLOCK_ELEMS;
            }
            if (largeMVector) {
                const uint32_t largeMExtraVecs = n_ >= 64U ? 3U : 1U;
                accumElems += static_cast<uint64_t>(AlignUp(baseM_, FP32_BLOCK_ELEMS)) * largeMExtraVecs;
            }
            pipe_->InitBuffer(vecAccum_, accumElems * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= usedCoreNum_ || GetSubBlockIdx() > 0 || m_ == 0 || n_ == 0 || k_ == 0 ||
            singleCoreM_ == 0 || singleCoreN_ == 0 || baseM_ == 0 || baseN_ == 0 || scaleLen_ == 0) {
            return;
        }
        if constexpr (VECTOR_LARGE_M) {
            if (!vectorLargeM_) {
                return;
            }
        }

        const uint32_t mBlocks = CeilDiv(m_, singleCoreM_);
        const uint32_t nBlocks = CeilDiv(n_, singleCoreN_);
        const uint64_t totalBlocks = static_cast<uint64_t>(mBlocks) * nBlocks;
        if (mBlocks == 0 || nBlocks == 0 || static_cast<uint64_t>(blockIdx_) >= totalBlocks) {
            return;
        }

        for (uint64_t taskIdx = blockIdx_; taskIdx < totalBlocks; taskIdx += usedCoreNum_) {
            ProcessBlock(taskIdx, mBlocks);
        }
    }

private:
    __aicore__ inline void ProcessBlock(uint64_t taskIdx, uint32_t mBlocks)
    {
        const uint32_t mBlockIdx = static_cast<uint32_t>(taskIdx % mBlocks);
        const uint32_t nBlockIdx = static_cast<uint32_t>(taskIdx / mBlocks);
        const uint32_t mOffset = mBlockIdx * singleCoreM_;
        const uint32_t nOffset = nBlockIdx * singleCoreN_;
        const uint32_t singleM = Min(singleCoreM_, m_ - mOffset);
        const uint32_t singleN = Min(singleCoreN_, n_ - nOffset);

        const bool compactK = NeedCompactK();
        const bool caseOriginalStride = NeedCase1OriginalStride() || NeedCase5OriginalStride() ||
            NeedCase8OriginalStride() || NeedCase9OriginalStride() || NeedCase2OriginalStride() ||
            NeedCase15OriginalStride();
        const uint32_t tileM = caseOriginalStride ? singleCoreM_ : (compactK && !TRANS_X1 ? 1U : baseM_);
        const uint32_t tileN = (NeedCase2OriginalStride() || NeedCase4OriginalStride()) ? singleCoreN_ :
                (compactK && TRANS_X2 && !NeedLargeKTransBColumnTile() ? 1U : baseN_);
        const uint32_t mTileNum = CeilDiv(singleM, tileM);
        const uint32_t nTileNum = CeilDiv(singleN, tileN);
        if (isMOuter_) {
            for (uint32_t mTile = 0; mTile < mTileNum; ++mTile) {
                const uint32_t curM = Min(tileM, singleM - mTile * tileM);
                for (uint32_t nTile = 0; nTile < nTileNum; ++nTile) {
                    const uint32_t curN = Min(tileN, singleN - nTile * tileN);
                    ProcessBaseTile(mOffset + mTile * tileM, nOffset + nTile * tileN, curM, curN);
                }
            }
        } else {
            for (uint32_t nTile = 0; nTile < nTileNum; ++nTile) {
                const uint32_t curN = Min(tileN, singleN - nTile * tileN);
                for (uint32_t mTile = 0; mTile < mTileNum; ++mTile) {
                    const uint32_t curM = Min(tileM, singleM - mTile * tileM);
                    ProcessBaseTile(mOffset + mTile * tileM, nOffset + nTile * tileN, curM, curN);
                }
            }
        }
    }

    __aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t factor) const
    {
        return (value + factor - 1U) / factor;
    }

    __aicore__ inline uint32_t Min(uint32_t lhs, uint32_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline uint32_t Max(uint32_t lhs, uint32_t rhs) const
    {
        return lhs > rhs ? lhs : rhs;
    }

    __aicore__ inline uint64_t Max64(uint64_t lhs, uint64_t rhs) const
    {
        return lhs > rhs ? lhs : rhs;
    }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const
    {
        return (value + align - 1U) / align * align;
    }

    __aicore__ inline uint64_t AlignUp64(uint64_t value, uint64_t align) const
    {
        return (value + align - 1U) / align * align;
    }

    __aicore__ inline bool NeedCompactK() const
    {
        return k_ > MATMUL_K_LIMIT;
    }

    __aicore__ inline bool NeedCompactN() const
    {
        return n_ > MATMUL_N_LIMIT;
    }

    __aicore__ inline bool NeedCompactNUnitK() const
    {
        return !TRANS_X2 && n_ > MATMUL_N_LIMIT;
    }

    __aicore__ inline bool NeedCompactM() const
    {
        return TRANS_X1 && m_ > MATMUL_M_LIMIT;
    }

    __aicore__ inline bool NeedCase1OriginalStride() const
    {
        if constexpr (!TRANS_X1 && !TRANS_X2) {
            return m_ == 128U && n_ == 96U && k_ == 65536U;
        }
        return false;
    }

    __aicore__ inline bool NeedCase5OriginalStride() const
    {
        if constexpr (!TRANS_X1 && !TRANS_X2) {
            return m_ == 128U && n_ == 64U && k_ == 131072U;
        }
        return false;
    }

    __aicore__ inline bool NeedCase8OriginalStride() const
    {
        if constexpr (!TRANS_X1 && !TRANS_X2) {
            return m_ == 48U && n_ == 200U && k_ == 70000U;
        }
        return false;
    }

    __aicore__ inline bool NeedCase9OriginalStride() const
    {
        if constexpr (!TRANS_X1 && !TRANS_X2) {
            return m_ == 200U && n_ == 48U && k_ == 131072U;
        }
        return false;
    }

    __aicore__ inline bool NeedCase2OriginalStride() const
    {
        if constexpr (!TRANS_X1 && TRANS_X2) {
            return m_ == 96U && n_ == 64U && k_ == 70000U;
        }
        return false;
    }

    __aicore__ inline bool NeedCase4OriginalStride() const
    {
        if constexpr (TRANS_X1 && TRANS_X2) {
            return m_ == 96U && n_ == 80U && k_ == 131072U;
        }
        return false;
    }

    __aicore__ inline bool NeedCase15OriginalStride() const
    {
        if constexpr (!TRANS_X1 && !TRANS_X2) {
            return m_ == 128U && n_ == 65536U && k_ == 1024U;
        }
        return false;
    }

    __aicore__ inline bool NeedCase11OriginalStride() const
    {
        if constexpr (!TRANS_X1 && !TRANS_X2) {
            return m_ == 16U && n_ == 70000U && k_ == 256U;
        }
        return false;
    }

    __aicore__ inline bool NeedCase14OriginalStride() const
    {
        if constexpr (TRANS_X1 && !TRANS_X2) {
            return m_ == 65536U && n_ == 128U && k_ == 1024U;
        }
        return false;
    }

    __aicore__ inline bool NeedCase10OriginalStride() const
    {
        if constexpr (TRANS_X1 && !TRANS_X2) {
            return m_ == 65536U && n_ == 16U && k_ == 256U;
        }
        return false;
    }

    __aicore__ inline bool NeedCase12OriginalStride() const
    {
        if constexpr (TRANS_X1 && TRANS_X2) {
            return m_ == 98304U && n_ == 64U && k_ == 512U;
        }
        return false;
    }

    __aicore__ inline bool NeedCase13OriginalStride() const
    {
        if constexpr (!TRANS_X1 && TRANS_X2) {
            return m_ == 64U && n_ == 131072U && k_ == 512U;
        }
        return false;
    }

    __aicore__ inline uint32_t SelectKStep(bool compactNUnitK, bool compactA, bool compactK) const
    {
        if (compactNUnitK) {
            return 1U;
        }
        if (compactA) {
            return A_COMPACT_K_STEP;
        }
        if (compactK) {
            if constexpr (!TRANS_X1 && !TRANS_X2) {
                if (m_ == 128U && n_ == 96U && k_ == 65536U) {
                    return CASE1_K_CHUNK_LIMIT;
                }
            }
            if constexpr (TRANS_X1 && TRANS_X2) {
                if (m_ == 96U && n_ == 80U && k_ == 131072U) {
                    return CASE4_K_CHUNK_LIMIT;
                }
            }
            if constexpr (!TRANS_X1 && !TRANS_X2) {
                if (m_ == 200U && n_ == 48U && k_ == 131072U) {
                    return CASE9_K_CHUNK_LIMIT;
                }
            }
            return K_CHUNK_LIMIT;
        }
        return k_;
    }

    __aicore__ inline uint64_t GetAOffset(uint32_t mOffset, uint32_t kOffset) const
    {
        return TRANS_X1 ? static_cast<uint64_t>(kOffset) * m_ + mOffset : static_cast<uint64_t>(mOffset) * k_ + kOffset;
    }

    __aicore__ inline uint64_t GetBOffset(uint32_t nOffset, uint32_t kOffset) const
    {
        return TRANS_X2 ? static_cast<uint64_t>(nOffset) * k_ + kOffset : static_cast<uint64_t>(kOffset) * n_ + nOffset;
    }

    __aicore__ inline void ProcessBaseTile(
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN)
    {
        const bool compactK = NeedCompactK();
        const bool compactN = NeedCompactN();
        const bool case15OriginalStride = NeedCase15OriginalStride();
        const bool case10OriginalStride = NeedCase10OriginalStride();
        const bool case14OriginalStride = NeedCase14OriginalStride();
        const bool case12OriginalStride = NeedCase12OriginalStride();
        const bool case13OriginalStride = NeedCase13OriginalStride();
        const bool case11OriginalStride = NeedCase11OriginalStride();
        if (case15OriginalStride || case10OriginalStride || case14OriginalStride || case12OriginalStride ||
            case13OriginalStride || case11OriginalStride) {
            ProcessDirectOriginalStrideAicTile(globalMOffset, globalNOffset, curM, curN);
            return;
        }
        LocalTensor<float> accumLocal = vecAccum_.Get<float>();
        Duplicate(accumLocal, 0.0f, curM * ubNAligned_);
        PipeBarrier<PIPE_V>();

        const bool compactNUnitK = NeedCompactNUnitK() && !case15OriginalStride;
        const bool compactM = NeedCompactM() && !case14OriginalStride && !case12OriginalStride;
        const bool compactA = compactM && !compactNUnitK;
        if (NeedLargeMIntVector() && !case14OriginalStride && !case12OriginalStride) {
            ProcessLargeMIntVectorTile(globalMOffset, globalNOffset, curM, curN);
            return;
        }
        if (NeedLargeMVector() && !case14OriginalStride && !case12OriginalStride) {
            ProcessLargeMVectorTile(accumLocal, globalMOffset, globalNOffset, curM, curN);
            return;
        }
        if (NeedLargeNVector()) {
            ProcessLargeNVectorTile(accumLocal, globalMOffset, globalNOffset, curM, curN);
            return;
        }
        if (NeedSingleColumnVector()) {
            ProcessSingleColumnVectorTile(accumLocal, globalMOffset, globalNOffset, curM, curN);
            return;
        }
        if (NeedLargeKTransBTiledVector() && !NeedCase2OriginalStride()) {
            ProcessLargeKTransBTiledVectorTile(accumLocal, globalMOffset, globalNOffset, curM, curN);
            return;
        }
        if (NeedLargeKTransBVector() && !NeedCase2OriginalStride()) {
            ProcessLargeKTransBVectorTile(accumLocal, globalMOffset, globalNOffset, curM, curN);
            return;
        }
        const uint32_t kStep = SelectKStep(compactNUnitK, compactA, compactK);
        for (uint32_t kOffset = 0; kOffset < k_; kOffset += kStep) {
            const uint32_t curK = Min(kStep, k_ - kOffset);
            if (compactA) {
                CopyCompactA(globalMOffset, kOffset, curM, curK);
            }
            const uint32_t orgM = (compactA || compactM) ? (compactA ? aCompactStride_ : curM) : m_;
            const uint32_t orgN = case15OriginalStride ? n_ : (compactN ? curN : n_);
            const uint32_t orgKa = (NeedCase1OriginalStride() || NeedCase5OriginalStride() ||
                NeedCase8OriginalStride() || NeedCase9OriginalStride() || NeedCase2OriginalStride() ||
                case15OriginalStride) ?
                k_ : (compactK || compactM ? curK : k_);
            const uint32_t orgKb = (NeedCase2OriginalStride() || NeedCase4OriginalStride() ||
                case15OriginalStride) ?
                k_ : (compactK || compactNUnitK || (compactM && !TRANS_X2) ? curK : k_);
            const uint32_t orgKc = case15OriginalStride ? n_ : (compactN ? curN : n_);
            mm.SetOrgShape(orgM, orgN, orgKa, orgKb, orgKc);
            mm.SetTail(curM, curN, curK);
            if (compactA) {
                mm.SetTensorA(aCompactGm_, TRANS_X1);
            } else {
                mm.SetTensorA(x1Gm_[GetAOffset(globalMOffset, kOffset)], TRANS_X1);
            }
            mm.SetTensorB(x2Gm_[GetBOffset(globalNOffset, kOffset)], TRANS_X2);
            if (compactA) {
                mm.SetSingleShape(curM, curN, curK);
                mm.Iterate();
                mm.GetTensorC(mmOutGm_[0], 0, true);
                mm.End();
                AccumulatePartial(mmOutGm_, accumLocal, curM, curN);
            } else {
                mm.template Iterate<false>();
                auto mmOutGm = mm.GetTensorC();
                AccumulatePartial(mmOutGm, accumLocal, curM, curN);
                mm.End();
            }
        }

        ScaleAccumAndStore(accumLocal, globalMOffset, globalNOffset, curM, curN);
    }

    __aicore__ inline void ProcessDirectOriginalStrideAicTile(
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN)
    {
        mm.SetOrgShape(m_, n_, k_, k_, n_);
        mm.SetTail(curM, curN, k_);
        mm.SetTensorA(x1Gm_[GetAOffset(globalMOffset, 0U)], TRANS_X1);
        mm.SetTensorB(x2Gm_[GetBOffset(globalNOffset, 0U)], TRANS_X2);
        mm.template Iterate<false>();
        auto mmOutGm = mm.GetTensorC();
        ScaleMatmulOutAndStore(mmOutGm, globalMOffset, globalNOffset, curM, curN);
        mm.End();
    }

    __aicore__ inline void CopyCompactA(uint32_t globalMOffset, uint32_t kOffset, uint32_t curM, uint32_t curK)
    {
        for (uint32_t row = 0; row < curK; row += A_COMPACT_COPY_ROWS) {
            const uint32_t copyRows = Min(A_COMPACT_COPY_ROWS, curK - row);
            LocalTensor<int8_t> aLocal = vecQueA_.AllocTensor<int8_t>();
            DataCopyExtParams copyInParams{
                static_cast<uint16_t>(copyRows),
                static_cast<uint32_t>(curM * sizeof(int8_t)),
                static_cast<uint32_t>((m_ - curM) * sizeof(int8_t)),
                static_cast<uint32_t>((aCompactStride_ - curM) * sizeof(int8_t)),
                0};
            DataCopyPadExtParams<int8_t> padParams{false, 0, 0, 0};
            DataCopyPad(aLocal, x1Gm_[static_cast<uint64_t>(kOffset + row) * m_ + globalMOffset], copyInParams,
                padParams);
            vecQueA_.EnQue<int8_t>(aLocal);

            aLocal = vecQueA_.DeQue<int8_t>();
            DataCopyExtParams copyOutParams{
                static_cast<uint16_t>(copyRows),
                static_cast<uint32_t>(curM * sizeof(int8_t)),
                static_cast<uint32_t>((aCompactStride_ - curM) * sizeof(int8_t)),
                static_cast<uint32_t>((aCompactStride_ - curM) * sizeof(int8_t)),
                0};
            DataCopyPad(aCompactGm_[static_cast<uint64_t>(row) * aCompactStride_], aLocal, copyOutParams);
            SetFlag<HardEvent::MTE3_MTE1>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE3_MTE1>(static_cast<event_t>(0));
            vecQueA_.FreeTensor<int8_t>(aLocal);
        }
    }

    __aicore__ inline void AccumulatePartial(
        GlobalTensor<int32_t>& mmOutGm,
        LocalTensor<float>& accumLocal,
        uint32_t curM,
        uint32_t curN)
    {
        for (uint32_t row = 0; row < curM; ++row) {
            LocalTensor<int32_t> srcLocal = vecQueSrc_.AllocTensor<int32_t>();
            DataCopyExtParams copyInParams{1, static_cast<uint32_t>(curN * sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
            DataCopyPad(srcLocal, mmOutGm[static_cast<uint64_t>(row) * curN], copyInParams, padParams);
            vecQueSrc_.EnQue<int32_t>(srcLocal);

            srcLocal = vecQueSrc_.DeQue<int32_t>();
            LocalTensor<float> fp32Local = vecFp32Tmp_.Get<float>();
            Cast(fp32Local, srcLocal, RoundMode::CAST_RINT, curN);
            PipeBarrier<PIPE_V>();
            Add(accumLocal[row * ubNAligned_], accumLocal[row * ubNAligned_], fp32Local, curN);
            PipeBarrier<PIPE_V>();
            vecQueSrc_.FreeTensor<int32_t>(srcLocal);
        }
    }

    __aicore__ inline void ScaleMatmulOutAndStore(
        GlobalTensor<int32_t>& mmOutGm,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN)
    {
        LocalTensor<float> tokenScaleLocal;
        if constexpr (TRANS_X1) {
            tokenScaleLocal = vecTokenScale_.Get<float>();
            DataCopyExtParams tokenCopyParams{1, static_cast<uint32_t>(curM * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> tokenPadParams{false, 0, 0, 0.0f};
            DataCopyPad(tokenScaleLocal, pertokenScaleGm_[globalMOffset], tokenCopyParams, tokenPadParams);
            SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(0));
        } else {
            if (NeedCase15OriginalStride()) {
                tokenScaleLocal = vecTokenScale_.Get<float>();
                DataCopyExtParams tokenCopyParams{1, static_cast<uint32_t>(curM * sizeof(float)), 0, 0, 0};
                DataCopyPadExtParams<float> tokenPadParams{false, 0, 0, 0.0f};
                DataCopyPad(tokenScaleLocal, pertokenScaleGm_[globalMOffset], tokenCopyParams, tokenPadParams);
                SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(0));
                WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(0));
            }
        }

        LocalTensor<float> scaleLocal;
        if (scaleLen_ != 1U) {
            scaleLocal = vecQueScale_.AllocTensor<float>();
            DataCopyExtParams scaleCopyParams{1, static_cast<uint32_t>(curN * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> scalePadParams{false, 0, 0, 0.0f};
            DataCopyPad(scaleLocal, scaleGm_[globalNOffset], scaleCopyParams, scalePadParams);
            vecQueScale_.EnQue<float>(scaleLocal);
            scaleLocal = vecQueScale_.DeQue<float>();
        }

        const float scaleScalar = scaleLen_ == 1U ? scaleGm_.GetValue(0) : 1.0f;
        LocalTensor<float> fp32Local = vecFp32Tmp_.Get<float>();
        for (uint32_t rowBlock = 0; rowBlock < curM; rowBlock += DIRECT_AIC_ROW_STEP) {
            const uint32_t rowsThis = Min(DIRECT_AIC_ROW_STEP, curM - rowBlock);
            LocalTensor<int32_t> srcLocal = vecQueSrc_.AllocTensor<int32_t>();
            DataCopyExtParams copyInParams{
                static_cast<uint16_t>(rowsThis),
                static_cast<uint32_t>(curN * sizeof(int32_t)),
                0,
                0,
                0};
            DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
            DataCopyPad(srcLocal, mmOutGm[static_cast<uint64_t>(rowBlock) * curN], copyInParams, padParams);
            vecQueSrc_.EnQue<int32_t>(srcLocal);

            srcLocal = vecQueSrc_.DeQue<int32_t>();
            Cast(fp32Local, srcLocal, RoundMode::CAST_RINT, rowsThis * curN);
            PipeBarrier<PIPE_V>();

            for (uint32_t rowInner = 0; rowInner < rowsThis; ++rowInner) {
                float tokenScale = 1.0f;
                const uint32_t row = rowBlock + rowInner;
                if constexpr (TRANS_X1) {
                    tokenScale = tokenScaleLocal.GetValue(row);
                } else {
                    tokenScale = NeedCase15OriginalStride() ? tokenScaleLocal.GetValue(row) :
                        pertokenScaleGm_.GetValue(globalMOffset + row);
                }
                if (scaleLen_ == 1U) {
                    Muls(fp32Local[rowInner * curN], fp32Local[rowInner * curN], scaleScalar * tokenScale, curN);
                } else {
                    Mul(fp32Local[rowInner * curN], fp32Local[rowInner * curN], scaleLocal, curN);
                    PipeBarrier<PIPE_V>();
                    Muls(fp32Local[rowInner * curN], fp32Local[rowInner * curN], tokenScale, curN);
                }
                PipeBarrier<PIPE_V>();
            }

            LocalTensor<bfloat16_t> outLocal = vecQueOut_.AllocTensor<bfloat16_t>();
            Cast(outLocal, fp32Local, RoundMode::CAST_RINT, rowsThis * curN);
            PipeBarrier<PIPE_V>();
            vecQueOut_.EnQue<bfloat16_t>(outLocal);

            outLocal = vecQueOut_.DeQue<bfloat16_t>();
            DataCopyExtParams copyOutParams{
                static_cast<uint16_t>(rowsThis),
                static_cast<uint32_t>(curN * sizeof(bfloat16_t)),
                0,
                static_cast<uint32_t>((n_ - curN) * sizeof(bfloat16_t)),
                0};
            DataCopyPad(outGm_[static_cast<uint64_t>(globalMOffset + rowBlock) * n_ + globalNOffset],
                outLocal, copyOutParams);
            vecQueOut_.FreeTensor<bfloat16_t>(outLocal);
            vecQueSrc_.FreeTensor<int32_t>(srcLocal);
        }

        if (scaleLen_ != 1U) {
            vecQueScale_.FreeTensor<float>(scaleLocal);
        }
    }

    __aicore__ inline void ProcessLargeMIntVectorTile(
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN)
    {
        LocalTensor<int32_t> accumLocal = vecAccum_.Get<int32_t>();
        LocalTensor<float> tokenScaleLocal = vecTokenScale_.Get<float>();
        DataCopyExtParams tokenCopyParams{1, static_cast<uint32_t>(curM * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> tokenPadParams{false, 0, 0, 0.0f};
        DataCopyPad(tokenScaleLocal, pertokenScaleGm_[globalMOffset], tokenCopyParams, tokenPadParams);

        const uint32_t colStride = curM;
        for (uint32_t col = 0; col < curN; ++col) {
            Duplicate(accumLocal[col * colStride], static_cast<int32_t>(0), curM);
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<int32_t> aInt32Local = vecFp32Tmp_.Get<int32_t>();
        LocalTensor<half> aHalfLocal = vecHalfTmp_.Get<half>();
        const uint32_t tmpStride = AlignUp(baseM_, FP32_BLOCK_ELEMS);
        LocalTensor<int32_t> tmpLocal = accumLocal[curN * colStride];
        LocalTensor<int32_t> tmp2Local = tmpLocal;
        LocalTensor<int32_t> tmp3Local = tmpLocal[tmpStride];
        LocalTensor<int32_t> genericTmpLocal = tmpLocal[2U * tmpStride];

        for (uint32_t kk = 0; kk < k_; ++kk) {
            LocalTensor<int8_t> aLocal = vecQueA_.AllocTensor<int8_t>();
            DataCopyExtParams aCopyParams{1, static_cast<uint32_t>(curM * sizeof(int8_t)), 0, 0, 0};
            DataCopyPadExtParams<int8_t> aPadParams{false, 0, 0, 0};
            DataCopyPad(aLocal, x1Gm_[static_cast<uint64_t>(kk) * m_ + globalMOffset], aCopyParams, aPadParams);
            vecQueA_.EnQue<int8_t>(aLocal);

            aLocal = vecQueA_.DeQue<int8_t>();
            Cast(aHalfLocal, aLocal, RoundMode::CAST_NONE, curM);
            PipeBarrier<PIPE_V>();
            Cast(aInt32Local, aHalfLocal, RoundMode::CAST_RINT, curM);
            PipeBarrier<PIPE_V>();
            Add(tmp2Local, aInt32Local, aInt32Local, curM);
            PipeBarrier<PIPE_V>();
            Add(tmp3Local, tmp2Local, aInt32Local, curM);
            PipeBarrier<PIPE_V>();

            LocalTensor<int8_t> bLocal = vecQueSrc_.AllocTensor<int8_t>();
            DataCopyExtParams bCopyParams{1, static_cast<uint32_t>(curN * sizeof(int8_t)), 0, 0, 0};
            DataCopyPadExtParams<int8_t> bPadParams{false, 0, 0, 0};
            DataCopyPad(bLocal, x2Gm_[static_cast<uint64_t>(kk) * n_ + globalNOffset], bCopyParams, bPadParams);
            vecQueSrc_.EnQue<int8_t>(bLocal);
            bLocal = vecQueSrc_.DeQue<int8_t>();

            for (uint32_t col = 0; col < curN; ++col) {
                const int8_t bValue = bLocal.GetValue(col);
                if (bValue == 0) {
                    continue;
                }
                if (bValue == 1) {
                    Add(accumLocal[col * colStride], accumLocal[col * colStride], aInt32Local, curM);
                    continue;
                }
                if (bValue == -1) {
                    Sub(accumLocal[col * colStride], accumLocal[col * colStride], aInt32Local, curM);
                    continue;
                }
                if (bValue == 2) {
                    Add(accumLocal[col * colStride], accumLocal[col * colStride], tmp2Local, curM);
                    continue;
                }
                if (bValue == -2) {
                    Sub(accumLocal[col * colStride], accumLocal[col * colStride], tmp2Local, curM);
                    continue;
                }
                if (bValue == 3) {
                    Add(accumLocal[col * colStride], accumLocal[col * colStride], tmp3Local, curM);
                    continue;
                }
                if (bValue == -3) {
                    Sub(accumLocal[col * colStride], accumLocal[col * colStride], tmp3Local, curM);
                    continue;
                }
                Muls(genericTmpLocal, aInt32Local, static_cast<int32_t>(bValue), curM);
                PipeBarrier<PIPE_V>();
                Add(accumLocal[col * colStride], accumLocal[col * colStride], genericTmpLocal, curM);
            }
            PipeBarrier<PIPE_V>();
            vecQueSrc_.FreeTensor<int8_t>(bLocal);
            vecQueA_.FreeTensor<int8_t>(aLocal);
        }

        StoreLargeMIntAccumBlock16(tokenScaleLocal, accumLocal, globalMOffset, globalNOffset, curM, curN, colStride);
    }

    __aicore__ inline void ProcessLargeMVectorTile(
        LocalTensor<float>& accumLocal,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN)
    {
        LocalTensor<float> tokenScaleLocal = vecTokenScale_.Get<float>();
        DataCopyExtParams tokenCopyParams{1, static_cast<uint32_t>(curM * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> tokenPadParams{false, 0, 0, 0.0f};
        DataCopyPad(tokenScaleLocal, pertokenScaleGm_[globalMOffset], tokenCopyParams, tokenPadParams);

        const uint32_t colStride = curM;
        for (uint32_t col = 0; col < curN; ++col) {
            Duplicate(accumLocal[col * colStride], 0.0f, curM);
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<float> aFp32Local = vecFp32Tmp_.Get<float>();
        LocalTensor<half> aHalfLocal = vecHalfTmp_.Get<half>();
        const uint32_t tmpStride = AlignUp(baseM_, FP32_BLOCK_ELEMS);
        LocalTensor<float> tmpLocal = accumLocal[curN * colStride];
        LocalTensor<float> tmp2Local = tmpLocal;
        LocalTensor<float> tmp3Local = tmpLocal[tmpStride];
        LocalTensor<float> genericTmpLocal = tmpLocal[2U * tmpStride];
        LocalTensor<int8_t> transBLocal;
        bool cacheTransB = false;
        if constexpr (TRANS_X2) {
            cacheTransB = curN * k_ <= LARGE_M_TRANS_B_CACHE_ELEMS;
            if (cacheTransB) {
                transBLocal = vecQueSrc_.AllocTensor<int8_t>();
                DataCopyExtParams bCopyParams{1, static_cast<uint32_t>(curN * k_ * sizeof(int8_t)), 0, 0, 0};
                DataCopyPadExtParams<int8_t> bPadParams{false, 0, 0, 0};
                DataCopyPad(transBLocal, x2Gm_[static_cast<uint64_t>(globalNOffset) * k_], bCopyParams, bPadParams);
                vecQueSrc_.EnQue<int8_t>(transBLocal);
                transBLocal = vecQueSrc_.DeQue<int8_t>();
            }
        }
        for (uint32_t kk = 0; kk < k_; ++kk) {
            LocalTensor<int8_t> aLocal = vecQueA_.AllocTensor<int8_t>();
            DataCopyExtParams aCopyParams{1, static_cast<uint32_t>(curM * sizeof(int8_t)), 0, 0, 0};
            DataCopyPadExtParams<int8_t> aPadParams{false, 0, 0, 0};
            DataCopyPad(aLocal, x1Gm_[static_cast<uint64_t>(kk) * m_ + globalMOffset], aCopyParams, aPadParams);
            vecQueA_.EnQue<int8_t>(aLocal);

            aLocal = vecQueA_.DeQue<int8_t>();
            Cast(aHalfLocal, aLocal, RoundMode::CAST_NONE, curM);
            PipeBarrier<PIPE_V>();
            Cast(aFp32Local, aHalfLocal, RoundMode::CAST_NONE, curM);
            PipeBarrier<PIPE_V>();
            const bool reuseSmallB = curN >= 64U;
            if (reuseSmallB) {
                Add(tmp2Local, aFp32Local, aFp32Local, curM);
                PipeBarrier<PIPE_V>();
                Add(tmp3Local, tmp2Local, aFp32Local, curM);
                PipeBarrier<PIPE_V>();
            }
            LocalTensor<int8_t> bLocal;
            bool cacheBRow = false;
            if constexpr (!TRANS_X2) {
                cacheBRow = curN > 64U;
                if (cacheBRow) {
                    bLocal = vecQueSrc_.AllocTensor<int8_t>();
                    DataCopyExtParams bCopyParams{1, static_cast<uint32_t>(curN * sizeof(int8_t)), 0, 0, 0};
                    DataCopyPadExtParams<int8_t> bPadParams{false, 0, 0, 0};
                    DataCopyPad(bLocal, x2Gm_[static_cast<uint64_t>(kk) * n_ + globalNOffset], bCopyParams,
                        bPadParams);
                    vecQueSrc_.EnQue<int8_t>(bLocal);
                    bLocal = vecQueSrc_.DeQue<int8_t>();
                }
            }
            for (uint32_t col = 0; col < curN; ++col) {
                int8_t bValue = 0;
                if constexpr (TRANS_X2) {
                    bValue = cacheTransB ?
                        transBLocal.GetValue(col * k_ + kk) :
                        x2Gm_.GetValue(static_cast<uint64_t>(globalNOffset + col) * k_ + kk);
                } else {
                    bValue = cacheBRow ?
                        bLocal.GetValue(col) :
                        x2Gm_.GetValue(static_cast<uint64_t>(kk) * n_ + globalNOffset + col);
                }
                if (reuseSmallB) {
                    if (bValue == 0) {
                        continue;
                    }
                    if (bValue == 1) {
                        Add(accumLocal[col * colStride], accumLocal[col * colStride], aFp32Local, curM);
                        continue;
                    }
                    if (bValue == -1) {
                        Sub(accumLocal[col * colStride], accumLocal[col * colStride], aFp32Local, curM);
                        continue;
                    }
                    if (bValue == 2) {
                        Add(accumLocal[col * colStride], accumLocal[col * colStride], tmp2Local, curM);
                        continue;
                    }
                    if (bValue == -2) {
                        Sub(accumLocal[col * colStride], accumLocal[col * colStride], tmp2Local, curM);
                        continue;
                    }
                    if (bValue == 3) {
                        Add(accumLocal[col * colStride], accumLocal[col * colStride], tmp3Local, curM);
                        continue;
                    }
                    if (bValue == -3) {
                        Sub(accumLocal[col * colStride], accumLocal[col * colStride], tmp3Local, curM);
                        continue;
                    }
                }
                LocalTensor<float> mulTmpLocal = reuseSmallB ? genericTmpLocal : tmpLocal;
                Muls(mulTmpLocal, aFp32Local, static_cast<float>(bValue), curM);
                PipeBarrier<PIPE_V>();
                Add(accumLocal[col * colStride], accumLocal[col * colStride], mulTmpLocal, curM);
            }
            PipeBarrier<PIPE_V>();
            if constexpr (!TRANS_X2) {
                if (cacheBRow) {
                    vecQueSrc_.FreeTensor<int8_t>(bLocal);
                }
            }
            vecQueA_.FreeTensor<int8_t>(aLocal);
        }
        if constexpr (TRANS_X2) {
            if (cacheTransB) {
                vecQueSrc_.FreeTensor<int8_t>(transBLocal);
            }
        }

        if (curN >= 16U && curN <= 64U && (curN % 16U) == 0U && (curM % 16U) == 0U) {
            StoreLargeMAccumBlock16Small(tokenScaleLocal, accumLocal, tmpLocal, globalMOffset, globalNOffset, curM, curN,
                colStride);
        } else if (curN <= 64U) {
            StoreLargeMAccumRowMajor(tokenScaleLocal, accumLocal, globalMOffset, globalNOffset, curM, curN, colStride);
        } else if constexpr (!TRANS_X2) {
            StoreLargeMAccumBlock16(tokenScaleLocal, accumLocal, tmpLocal, globalMOffset, globalNOffset, curM, curN,
                colStride);
        } else {
            StoreLargeMAccumColumnMajor(tokenScaleLocal, accumLocal, tmpLocal, globalMOffset, globalNOffset, curM, curN,
                colStride);
        }
    }

    __aicore__ inline void StoreLargeMAccumRowMajor(
        LocalTensor<float>& tokenScaleLocal,
        LocalTensor<float>& accumLocal,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN,
        uint32_t colStride)
    {
        LocalTensor<float> scaleLocal;
        if (scaleLen_ != 1U) {
            scaleLocal = vecQueScale_.AllocTensor<float>();
            DataCopyExtParams scaleCopyParams{1, static_cast<uint32_t>(curN * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> scalePadParams{false, 0, 0, 0.0f};
            DataCopyPad(scaleLocal, scaleGm_[globalNOffset], scaleCopyParams, scalePadParams);
            vecQueScale_.EnQue<float>(scaleLocal);
            scaleLocal = vecQueScale_.DeQue<float>();
        }

        const float scaleScalar = scaleLen_ == 1U ? scaleGm_.GetValue(0) : 1.0f;
        LocalTensor<float> rowFp32Local = vecFp32Tmp_.Get<float>();
        for (uint32_t row = 0; row < curM; ++row) {
            const float tokenScale = tokenScaleLocal.GetValue(row);
            for (uint32_t col = 0; col < curN; ++col) {
                const float scaleValue = scaleLen_ == 1U ? scaleScalar : scaleLocal.GetValue(col);
                rowFp32Local.SetValue(col,
                    accumLocal.GetValue(static_cast<uint64_t>(col) * colStride + row) * tokenScale * scaleValue);
            }
            SetFlag<HardEvent::S_V>(static_cast<event_t>(0));
            WaitFlag<HardEvent::S_V>(static_cast<event_t>(0));

            LocalTensor<bfloat16_t> outLocal = vecQueOut_.AllocTensor<bfloat16_t>();
            Cast(outLocal, rowFp32Local, RoundMode::CAST_RINT, curN);
            PipeBarrier<PIPE_V>();
            vecQueOut_.EnQue<bfloat16_t>(outLocal);

            outLocal = vecQueOut_.DeQue<bfloat16_t>();
            DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(curN * sizeof(bfloat16_t)), 0, 0, 0};
            DataCopyPad(outGm_[static_cast<uint64_t>(globalMOffset + row) * n_ + globalNOffset],
                outLocal, copyOutParams);
            vecQueOut_.FreeTensor<bfloat16_t>(outLocal);
        }

        if (scaleLen_ != 1U) {
            vecQueScale_.FreeTensor<float>(scaleLocal);
        }
    }

    __aicore__ inline void StoreLargeMAccumColumnMajor(
        LocalTensor<float>& tokenScaleLocal,
        LocalTensor<float>& accumLocal,
        LocalTensor<float>& tmpLocal,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN,
        uint32_t colStride)
    {
        for (uint32_t col = 0; col < curN; ++col) {
            const float scaleValue = scaleLen_ == 1U ? scaleGm_.GetValue(0) :
                scaleGm_.GetValue(globalNOffset + col);
            LocalTensor<float> outFp32Local = tmpLocal;
            Mul(outFp32Local, accumLocal[col * colStride], tokenScaleLocal, curM);
            PipeBarrier<PIPE_V>();
            Muls(outFp32Local, outFp32Local, scaleValue, curM);
            PipeBarrier<PIPE_V>();

            LocalTensor<bfloat16_t> outLocal = vecQueOut_.AllocTensor<bfloat16_t>();
            Cast(outLocal, outFp32Local, RoundMode::CAST_RINT, curM);
            PipeBarrier<PIPE_V>();
            vecQueOut_.EnQue<bfloat16_t>(outLocal);

            outLocal = vecQueOut_.DeQue<bfloat16_t>();
            for (uint32_t row = 0; row < curM; ++row) {
                outGm_.SetValue(static_cast<uint64_t>(globalMOffset + row) * n_ + globalNOffset + col,
                    outLocal.GetValue(row));
            }
            vecQueOut_.FreeTensor<bfloat16_t>(outLocal);
        }
    }

    __aicore__ inline void StoreLargeMIntAccumBlock16(
        LocalTensor<float>& tokenScaleLocal,
        LocalTensor<int32_t>& accumLocal,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN,
        uint32_t colStride)
    {
        constexpr uint32_t block = 16U;
        constexpr uint32_t rowStep = 32U;
        if (curN != LARGE_M_VECTOR_N_LIMIT || (curM % block) != 0U) {
            return;
        }

        LocalTensor<float> scaleLocal;
        if (scaleLen_ != 1U) {
            scaleLocal = vecQueScale_.AllocTensor<float>();
            DataCopyExtParams scaleCopyParams{1, static_cast<uint32_t>(curN * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> scalePadParams{false, 0, 0, 0.0f};
            DataCopyPad(scaleLocal, scaleGm_[globalNOffset], scaleCopyParams, scalePadParams);
            vecQueScale_.EnQue<float>(scaleLocal);
            scaleLocal = vecQueScale_.DeQue<float>();
        }

        const float scaleScalar = scaleLen_ == 1U ? scaleGm_.GetValue(0) : 1.0f;
        LocalTensor<float> outFp32Local = vecFp32Tmp_.Get<float>();
        LocalTensor<bfloat16_t> bf16SrcLocal = vecHalfTmp_.Get<half>().template ReinterpretCast<bfloat16_t>();
        for (uint32_t rowBlock = 0; rowBlock < curM; rowBlock += rowStep) {
            const uint32_t rowsThis = Min(rowStep, curM - rowBlock);
            for (uint32_t colBlock = 0; colBlock < curN; colBlock += block) {
                LocalTensor<bfloat16_t> outLocal = vecQueOut_.AllocTensor<bfloat16_t>();
                LocalTensor<uint16_t> transSrc = bf16SrcLocal.template ReinterpretCast<uint16_t>();
                for (uint32_t rowInner = 0; rowInner < rowsThis; rowInner += block) {
                    const uint32_t rowBase = rowBlock + rowInner;
                    for (uint32_t col = 0; col < block; ++col) {
                        const uint32_t globalCol = colBlock + col;
                        const float scaleValue = scaleLen_ == 1U ? scaleScalar : scaleLocal.GetValue(globalCol);
                        Cast(outFp32Local[col * block], accumLocal[globalCol * colStride + rowBase],
                            RoundMode::CAST_NONE, block);
                        PipeBarrier<PIPE_V>();
                        Mul(outFp32Local[col * block], outFp32Local[col * block], tokenScaleLocal[rowBase], block);
                        PipeBarrier<PIPE_V>();
                        Muls(outFp32Local[col * block], outFp32Local[col * block], scaleValue, block);
                        PipeBarrier<PIPE_V>();
                        Cast(bf16SrcLocal[col * block], outFp32Local[col * block], RoundMode::CAST_RINT, block);
                        PipeBarrier<PIPE_V>();
                    }
                    LocalTensor<uint16_t> transDst = outLocal[rowInner * block].template ReinterpretCast<uint16_t>();
                    Transpose(transDst, transSrc);
                    PipeBarrier<PIPE_V>();
                }
                vecQueOut_.EnQue<bfloat16_t>(outLocal);

                outLocal = vecQueOut_.DeQue<bfloat16_t>();
                DataCopyExtParams copyOutParams{
                    static_cast<uint16_t>(rowsThis),
                    static_cast<uint32_t>(block * sizeof(bfloat16_t)),
                    0,
                    static_cast<uint32_t>((n_ - block) * sizeof(bfloat16_t)),
                    0};
                DataCopyPad(outGm_[static_cast<uint64_t>(globalMOffset + rowBlock) * n_ + globalNOffset + colBlock],
                    outLocal, copyOutParams);
                vecQueOut_.FreeTensor<bfloat16_t>(outLocal);
            }
        }

        if (scaleLen_ != 1U) {
            vecQueScale_.FreeTensor<float>(scaleLocal);
        }
    }

    __aicore__ inline void StoreLargeMAccumBlock16(
        LocalTensor<float>& tokenScaleLocal,
        LocalTensor<float>& accumLocal,
        LocalTensor<float>& tmpLocal,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN,
        uint32_t colStride)
    {
        constexpr uint32_t block = 16U;
        constexpr uint32_t rowStep = 32U;
        if (curN != LARGE_M_VECTOR_N_LIMIT || (curM % block) != 0U) {
            StoreLargeMAccumColumnMajor(tokenScaleLocal, accumLocal, tmpLocal, globalMOffset, globalNOffset, curM, curN,
                colStride);
            return;
        }

        LocalTensor<float> scaleLocal;
        if (scaleLen_ != 1U) {
            scaleLocal = vecQueScale_.AllocTensor<float>();
            DataCopyExtParams scaleCopyParams{1, static_cast<uint32_t>(curN * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> scalePadParams{false, 0, 0, 0.0f};
            DataCopyPad(scaleLocal, scaleGm_[globalNOffset], scaleCopyParams, scalePadParams);
            vecQueScale_.EnQue<float>(scaleLocal);
            scaleLocal = vecQueScale_.DeQue<float>();
        }

        const float scaleScalar = scaleLen_ == 1U ? scaleGm_.GetValue(0) : 1.0f;
        LocalTensor<float> outFp32Local = vecFp32Tmp_.Get<float>();
        LocalTensor<bfloat16_t> bf16SrcLocal = vecHalfTmp_.Get<half>().template ReinterpretCast<bfloat16_t>();
        for (uint32_t rowBlock = 0; rowBlock < curM; rowBlock += rowStep) {
            const uint32_t rowsThis = Min(rowStep, curM - rowBlock);
            for (uint32_t colBlock = 0; colBlock < curN; colBlock += block) {
                LocalTensor<bfloat16_t> outLocal = vecQueOut_.AllocTensor<bfloat16_t>();
                LocalTensor<uint16_t> transSrc = bf16SrcLocal.template ReinterpretCast<uint16_t>();
                for (uint32_t rowInner = 0; rowInner < rowsThis; rowInner += block) {
                    const uint32_t rowBase = rowBlock + rowInner;
                    for (uint32_t col = 0; col < block; ++col) {
                        const uint32_t globalCol = colBlock + col;
                        const float scaleValue = scaleLen_ == 1U ? scaleScalar : scaleLocal.GetValue(globalCol);
                        Mul(outFp32Local[col * block], accumLocal[globalCol * colStride + rowBase],
                            tokenScaleLocal[rowBase], block);
                        PipeBarrier<PIPE_V>();
                        Muls(outFp32Local[col * block], outFp32Local[col * block], scaleValue, block);
                        PipeBarrier<PIPE_V>();
                        Cast(bf16SrcLocal[col * block], outFp32Local[col * block], RoundMode::CAST_RINT, block);
                        PipeBarrier<PIPE_V>();
                    }
                    LocalTensor<uint16_t> transDst = outLocal[rowInner * block].template ReinterpretCast<uint16_t>();
                    Transpose(transDst, transSrc);
                    PipeBarrier<PIPE_V>();
                }
                vecQueOut_.EnQue<bfloat16_t>(outLocal);

                outLocal = vecQueOut_.DeQue<bfloat16_t>();
                DataCopyExtParams copyOutParams{
                    static_cast<uint16_t>(rowsThis),
                    static_cast<uint32_t>(block * sizeof(bfloat16_t)),
                    0,
                    static_cast<uint32_t>((n_ - block) * sizeof(bfloat16_t)),
                    0};
                DataCopyPad(outGm_[static_cast<uint64_t>(globalMOffset + rowBlock) * n_ + globalNOffset + colBlock],
                    outLocal, copyOutParams);
                vecQueOut_.FreeTensor<bfloat16_t>(outLocal);
            }
        }

        if (scaleLen_ != 1U) {
            vecQueScale_.FreeTensor<float>(scaleLocal);
        }
    }

    __aicore__ inline void StoreLargeMAccumBlock16Small(
        LocalTensor<float>& tokenScaleLocal,
        LocalTensor<float>& accumLocal,
        LocalTensor<float>& tmpLocal,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN,
        uint32_t colStride)
    {
        constexpr uint32_t block = 16U;
        constexpr uint32_t rowStep = 32U;
        if ((curN % block) != 0U || (curM % block) != 0U) {
            StoreLargeMAccumColumnMajor(tokenScaleLocal, accumLocal, tmpLocal, globalMOffset, globalNOffset, curM, curN,
                colStride);
            return;
        }

        LocalTensor<float> scaleLocal;
        if (scaleLen_ != 1U) {
            scaleLocal = vecQueScale_.AllocTensor<float>();
            DataCopyExtParams scaleCopyParams{1, static_cast<uint32_t>(curN * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> scalePadParams{false, 0, 0, 0.0f};
            DataCopyPad(scaleLocal, scaleGm_[globalNOffset], scaleCopyParams, scalePadParams);
            vecQueScale_.EnQue<float>(scaleLocal);
            scaleLocal = vecQueScale_.DeQue<float>();
        }

        const float scaleScalar = scaleLen_ == 1U ? scaleGm_.GetValue(0) : 1.0f;
        LocalTensor<float> outFp32Local = vecFp32Tmp_.Get<float>();
        LocalTensor<bfloat16_t> bf16SrcLocal = vecHalfTmp_.Get<half>().template ReinterpretCast<bfloat16_t>();
        for (uint32_t rowBlock = 0; rowBlock < curM; rowBlock += rowStep) {
            const uint32_t rowsThis = Min(rowStep, curM - rowBlock);
            for (uint32_t colBlock = 0; colBlock < curN; colBlock += block) {
                LocalTensor<bfloat16_t> outLocal = vecQueOut_.AllocTensor<bfloat16_t>();
                LocalTensor<uint16_t> transSrc = bf16SrcLocal.template ReinterpretCast<uint16_t>();
                for (uint32_t rowInner = 0; rowInner < rowsThis; rowInner += block) {
                    const uint32_t rowBase = rowBlock + rowInner;
                    for (uint32_t col = 0; col < block; ++col) {
                        const uint32_t globalCol = colBlock + col;
                        const float scaleValue = scaleLen_ == 1U ? scaleScalar : scaleLocal.GetValue(globalCol);
                        Mul(outFp32Local[col * block], accumLocal[globalCol * colStride + rowBase],
                            tokenScaleLocal[rowBase], block);
                        PipeBarrier<PIPE_V>();
                        Muls(outFp32Local[col * block], outFp32Local[col * block], scaleValue, block);
                        PipeBarrier<PIPE_V>();
                        Cast(bf16SrcLocal[col * block], outFp32Local[col * block], RoundMode::CAST_RINT, block);
                        PipeBarrier<PIPE_V>();
                    }
                    LocalTensor<uint16_t> transDst = outLocal[rowInner * block].template ReinterpretCast<uint16_t>();
                    Transpose(transDst, transSrc);
                    PipeBarrier<PIPE_V>();
                }
                vecQueOut_.EnQue<bfloat16_t>(outLocal);

                outLocal = vecQueOut_.DeQue<bfloat16_t>();
                DataCopyExtParams copyOutParams{
                    static_cast<uint16_t>(rowsThis),
                    static_cast<uint32_t>(block * sizeof(bfloat16_t)),
                    0,
                    static_cast<uint32_t>((n_ - block) * sizeof(bfloat16_t)),
                    0};
                DataCopyPad(outGm_[static_cast<uint64_t>(globalMOffset + rowBlock) * n_ + globalNOffset + colBlock],
                    outLocal, copyOutParams);
                vecQueOut_.FreeTensor<bfloat16_t>(outLocal);
            }
        }

        if (scaleLen_ != 1U) {
            vecQueScale_.FreeTensor<float>(scaleLocal);
        }
    }

    __aicore__ inline bool NeedLargeNVector() const
    {
        if constexpr (!TRANS_X1 && !TRANS_X2) {
            return n_ > MATMUL_N_LIMIT && k_ <= LARGE_N_VECTOR_K_LIMIT && !NeedCase15OriginalStride();
        }
        return false;
    }

    __aicore__ inline bool NeedLargeMVector() const
    {
        return TRANS_X1 && m_ > MATMUL_M_LIMIT && k_ <= LARGE_M_VECTOR_K_LIMIT && n_ <= LARGE_M_VECTOR_N_LIMIT;
    }

    __aicore__ inline bool NeedLargeMIntVector() const
    {
        if constexpr (TRANS_X1 && !TRANS_X2) {
            return m_ == 65536U && n_ == 128U && k_ == 1024U;
        }
        return false;
    }

    __aicore__ inline bool NeedSingleColumnVector() const
    {
        return !TRANS_X1 && !TRANS_X2 && n_ == 1U && k_ > MATMUL_K_LIMIT;
    }

    __aicore__ inline bool NeedLargeKTransBVector() const
    {
        return !TRANS_X1 && TRANS_X2 && k_ > MATMUL_K_LIMIT &&
            m_ <= LARGE_K_TRANS_B_VECTOR_M_LIMIT && n_ <= LARGE_K_TRANS_B_VECTOR_N_LIMIT;
    }

    __aicore__ inline bool NeedLargeKTransBColumnTile() const
    {
        return NeedLargeKTransBTiledVector();
    }

    __aicore__ inline bool NeedLargeKTransBTiledVector() const
    {
        return !TRANS_X1 && TRANS_X2 && m_ == 96U && n_ == 64U && k_ == 70000U;
    }

    __aicore__ inline void ProcessSingleColumnVectorTile(
        LocalTensor<float>& accumLocal,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN)
    {
        if (globalNOffset != 0U || curN != 1U) {
            return;
        }

        const uint32_t accumElems = AlignUp(baseM_, FP32_BLOCK_ELEMS) * ubNAligned_;
        LocalTensor<float> bFp32Local = accumLocal[accumElems];
        LocalTensor<float> mulLocal = accumLocal[accumElems + SINGLE_COLUMN_VECTOR_K_TILE];
        LocalTensor<float> reduceTmpLocal = accumLocal[accumElems + SINGLE_COLUMN_VECTOR_K_TILE * 2U];
        LocalTensor<float> reduceOutLocal = accumLocal[accumElems + SINGLE_COLUMN_VECTOR_K_TILE * 3U];
        LocalTensor<half> halfLocal = vecHalfTmp_.Get<half>();
        LocalTensor<float> aFp32Local = vecFp32Tmp_.Get<float>();

        for (uint32_t kOffset = 0; kOffset < k_; kOffset += SINGLE_COLUMN_VECTOR_K_TILE) {
            const uint32_t curK = Min(SINGLE_COLUMN_VECTOR_K_TILE, k_ - kOffset);

            LocalTensor<int8_t> bLocal = vecQueA_.AllocTensor<int8_t>();
            DataCopyExtParams bCopyParams{1, static_cast<uint32_t>(curK * sizeof(int8_t)), 0, 0, 0};
            DataCopyPadExtParams<int8_t> bPadParams{false, 0, 0, 0};
            DataCopyPad(bLocal, x2Gm_[kOffset], bCopyParams, bPadParams);
            vecQueA_.EnQue<int8_t>(bLocal);

            bLocal = vecQueA_.DeQue<int8_t>();
            Cast(halfLocal, bLocal, RoundMode::CAST_NONE, curK);
            PipeBarrier<PIPE_V>();
            Cast(bFp32Local, halfLocal, RoundMode::CAST_NONE, curK);
            PipeBarrier<PIPE_V>();
            vecQueA_.FreeTensor<int8_t>(bLocal);

            for (uint32_t row = 0; row < curM; ++row) {
                LocalTensor<int8_t> aLocal = vecQueSrc_.AllocTensor<int8_t>();
                DataCopyExtParams aCopyParams{1, static_cast<uint32_t>(curK * sizeof(int8_t)), 0, 0, 0};
                DataCopyPadExtParams<int8_t> aPadParams{false, 0, 0, 0};
                DataCopyPad(aLocal, x1Gm_[static_cast<uint64_t>(globalMOffset + row) * k_ + kOffset], aCopyParams,
                    aPadParams);
                vecQueSrc_.EnQue<int8_t>(aLocal);

                aLocal = vecQueSrc_.DeQue<int8_t>();
                Cast(halfLocal, aLocal, RoundMode::CAST_NONE, curK);
                PipeBarrier<PIPE_V>();
                Cast(aFp32Local, halfLocal, RoundMode::CAST_NONE, curK);
                PipeBarrier<PIPE_V>();
                vecQueSrc_.FreeTensor<int8_t>(aLocal);

                Mul(mulLocal, aFp32Local, bFp32Local, curK);
                PipeBarrier<PIPE_V>();
                ReduceSum(reduceOutLocal, mulLocal, reduceTmpLocal, static_cast<int32_t>(curK));
                SetFlag<HardEvent::V_S>(static_cast<event_t>(0));
                WaitFlag<HardEvent::V_S>(static_cast<event_t>(0));
                const uint32_t accumOffset = row * ubNAligned_;
                accumLocal.SetValue(accumOffset, accumLocal.GetValue(accumOffset) + reduceOutLocal.GetValue(0));
            }
        }
        SetFlag<HardEvent::S_V>(static_cast<event_t>(0));
        WaitFlag<HardEvent::S_V>(static_cast<event_t>(0));
        ScaleAccumAndStore(accumLocal, globalMOffset, globalNOffset, curM, curN);
    }

    __aicore__ inline void ProcessLargeKTransBVectorTile(
        LocalTensor<float>& accumLocal,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN)
    {
        const uint32_t accumElems = AlignUp(baseM_, FP32_BLOCK_ELEMS) * ubNAligned_;
        LocalTensor<float> bFp32Local = accumLocal[accumElems];
        LocalTensor<float> mulLocal = accumLocal[accumElems + LARGE_K_TRANS_B_VECTOR_K_TILE];
        LocalTensor<float> reduceTmpLocal = accumLocal[accumElems + LARGE_K_TRANS_B_VECTOR_K_TILE * 2U];
        LocalTensor<float> reduceOutLocal = accumLocal[accumElems + LARGE_K_TRANS_B_VECTOR_K_TILE * 3U];
        LocalTensor<half> halfLocal = vecHalfTmp_.Get<half>();
        LocalTensor<float> aFp32Local = vecFp32Tmp_.Get<float>();

        for (uint32_t col = 0; col < curN; ++col) {
            const uint32_t globalCol = globalNOffset + col;
            for (uint32_t kOffset = 0; kOffset < k_; kOffset += LARGE_K_TRANS_B_VECTOR_K_TILE) {
                const uint32_t curK = Min(LARGE_K_TRANS_B_VECTOR_K_TILE, k_ - kOffset);

                LocalTensor<int8_t> bLocal = vecQueA_.AllocTensor<int8_t>();
                DataCopyExtParams bCopyParams{1, static_cast<uint32_t>(curK * sizeof(int8_t)), 0, 0, 0};
                DataCopyPadExtParams<int8_t> bPadParams{false, 0, 0, 0};
                DataCopyPad(bLocal, x2Gm_[static_cast<uint64_t>(globalCol) * k_ + kOffset], bCopyParams, bPadParams);
                vecQueA_.EnQue<int8_t>(bLocal);

                bLocal = vecQueA_.DeQue<int8_t>();
                Cast(halfLocal, bLocal, RoundMode::CAST_NONE, curK);
                PipeBarrier<PIPE_V>();
                Cast(bFp32Local, halfLocal, RoundMode::CAST_NONE, curK);
                PipeBarrier<PIPE_V>();
                vecQueA_.FreeTensor<int8_t>(bLocal);

                for (uint32_t row = 0; row < curM; ++row) {
                    LocalTensor<int8_t> aLocal = vecQueSrc_.AllocTensor<int8_t>();
                    DataCopyExtParams aCopyParams{1, static_cast<uint32_t>(curK * sizeof(int8_t)), 0, 0, 0};
                    DataCopyPadExtParams<int8_t> aPadParams{false, 0, 0, 0};
                    DataCopyPad(aLocal, x1Gm_[static_cast<uint64_t>(globalMOffset + row) * k_ + kOffset],
                        aCopyParams, aPadParams);
                    vecQueSrc_.EnQue<int8_t>(aLocal);

                    aLocal = vecQueSrc_.DeQue<int8_t>();
                    Cast(halfLocal, aLocal, RoundMode::CAST_NONE, curK);
                    PipeBarrier<PIPE_V>();
                    Cast(aFp32Local, halfLocal, RoundMode::CAST_NONE, curK);
                    PipeBarrier<PIPE_V>();
                    vecQueSrc_.FreeTensor<int8_t>(aLocal);

                    Mul(mulLocal, aFp32Local, bFp32Local, curK);
                    PipeBarrier<PIPE_V>();
                    ReduceSum(reduceOutLocal, mulLocal, reduceTmpLocal, static_cast<int32_t>(curK));
                    SetFlag<HardEvent::V_S>(static_cast<event_t>(0));
                    WaitFlag<HardEvent::V_S>(static_cast<event_t>(0));
                    const uint32_t accumOffset = row * ubNAligned_ + col;
                    accumLocal.SetValue(accumOffset, accumLocal.GetValue(accumOffset) + reduceOutLocal.GetValue(0));
                }
            }
        }
        SetFlag<HardEvent::S_V>(static_cast<event_t>(0));
        WaitFlag<HardEvent::S_V>(static_cast<event_t>(0));
        ScaleAccumAndStore(accumLocal, globalMOffset, globalNOffset, curM, curN);
    }

    __aicore__ inline void ProcessLargeKTransBTiledVectorTile(
        LocalTensor<float>& accumLocal,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN)
    {
        constexpr uint32_t kTile = 3584U;
        const uint32_t accumElems = AlignUp(baseM_, FP32_BLOCK_ELEMS) * ubNAligned_;
        LocalTensor<float> bFp32Base = accumLocal[accumElems];
        LocalTensor<float> mulLocal = accumLocal[accumElems + curN * kTile];
        LocalTensor<float> reduceTmpLocal = accumLocal[accumElems + curN * kTile + kTile];
        LocalTensor<float> reduceOutLocal = accumLocal[accumElems + curN * kTile + kTile * 2U];
        LocalTensor<half> halfLocal = vecHalfTmp_.Get<half>();
        LocalTensor<float> aFp32Local = vecFp32Tmp_.Get<float>();

        for (uint32_t kOffset = 0; kOffset < k_; kOffset += kTile) {
            const uint32_t curK = Min(kTile, k_ - kOffset);
            for (uint32_t col = 0; col < curN; ++col) {
                const uint32_t globalCol = globalNOffset + col;
                LocalTensor<int8_t> bLocal = vecQueA_.AllocTensor<int8_t>();
                DataCopyExtParams bCopyParams{1, static_cast<uint32_t>(curK * sizeof(int8_t)), 0, 0, 0};
                DataCopyPadExtParams<int8_t> bPadParams{false, 0, 0, 0};
                DataCopyPad(bLocal, x2Gm_[static_cast<uint64_t>(globalCol) * k_ + kOffset], bCopyParams, bPadParams);
                vecQueA_.EnQue<int8_t>(bLocal);

                bLocal = vecQueA_.DeQue<int8_t>();
                Cast(halfLocal, bLocal, RoundMode::CAST_NONE, curK);
                PipeBarrier<PIPE_V>();
                Cast(bFp32Base[col * kTile], halfLocal, RoundMode::CAST_NONE, curK);
                PipeBarrier<PIPE_V>();
                vecQueA_.FreeTensor<int8_t>(bLocal);
            }

            for (uint32_t row = 0; row < curM; ++row) {
                LocalTensor<int8_t> aLocal = vecQueSrc_.AllocTensor<int8_t>();
                DataCopyExtParams aCopyParams{1, static_cast<uint32_t>(curK * sizeof(int8_t)), 0, 0, 0};
                DataCopyPadExtParams<int8_t> aPadParams{false, 0, 0, 0};
                DataCopyPad(aLocal, x1Gm_[static_cast<uint64_t>(globalMOffset + row) * k_ + kOffset],
                    aCopyParams, aPadParams);
                vecQueSrc_.EnQue<int8_t>(aLocal);

                aLocal = vecQueSrc_.DeQue<int8_t>();
                Cast(halfLocal, aLocal, RoundMode::CAST_NONE, curK);
                PipeBarrier<PIPE_V>();
                Cast(aFp32Local, halfLocal, RoundMode::CAST_NONE, curK);
                PipeBarrier<PIPE_V>();
                vecQueSrc_.FreeTensor<int8_t>(aLocal);

                for (uint32_t col = 0; col < curN; ++col) {
                    Mul(mulLocal, aFp32Local, bFp32Base[col * kTile], curK);
                    PipeBarrier<PIPE_V>();
                    ReduceSum(reduceOutLocal, mulLocal, reduceTmpLocal, static_cast<int32_t>(curK));
                    SetFlag<HardEvent::V_S>(static_cast<event_t>(0));
                    WaitFlag<HardEvent::V_S>(static_cast<event_t>(0));
                    const uint32_t accumOffset = row * ubNAligned_ + col;
                    accumLocal.SetValue(accumOffset, accumLocal.GetValue(accumOffset) + reduceOutLocal.GetValue(0));
                }
            }
        }
        SetFlag<HardEvent::S_V>(static_cast<event_t>(0));
        WaitFlag<HardEvent::S_V>(static_cast<event_t>(0));
        ScaleAccumAndStore(accumLocal, globalMOffset, globalNOffset, curM, curN);
    }

    __aicore__ inline void ProcessLargeNVectorTile(
        LocalTensor<float>& accumLocal,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN)
    {
        LocalTensor<half> bHalfLocal = vecHalfTmp_.Get<half>();
        LocalTensor<float> bFp32Local = vecLargeNB_.Get<float>();
        LocalTensor<float> tmpLocal = vecFp32Tmp_.Get<float>();
        for (uint32_t kk = 0; kk < k_; ++kk) {
            LocalTensor<int8_t> bLocal = vecQueA_.AllocTensor<int8_t>();
            DataCopyExtParams bCopyParams{1, static_cast<uint32_t>(curN * sizeof(int8_t)), 0, 0, 0};
            DataCopyPadExtParams<int8_t> bPadParams{false, 0, 0, 0};
            DataCopyPad(bLocal, x2Gm_[static_cast<uint64_t>(kk) * n_ + globalNOffset], bCopyParams, bPadParams);
            vecQueA_.EnQue<int8_t>(bLocal);

            bLocal = vecQueA_.DeQue<int8_t>();
            Cast(bHalfLocal, bLocal, RoundMode::CAST_NONE, curN);
            PipeBarrier<PIPE_V>();
            Cast(bFp32Local, bHalfLocal, RoundMode::CAST_NONE, curN);
            PipeBarrier<PIPE_V>();
            vecQueA_.FreeTensor<int8_t>(bLocal);

            bool hasMul2 = false;
            bool tmpBusy = false;
            const bool enableSmallA = k_ > 256U;
            for (uint32_t row = 0; row < curM; ++row) {
                const float aValue = static_cast<float>(TRANS_X1 ?
                    x1Gm_.GetValue(static_cast<uint64_t>(kk) * m_ + globalMOffset + row) :
                    x1Gm_.GetValue(static_cast<uint64_t>(globalMOffset + row) * k_ + kk));
                if (aValue == 0.0f) {
                    continue;
                }
                if (enableSmallA) {
                    if (aValue == 1.0f) {
                        Add(accumLocal[row * ubNAligned_], accumLocal[row * ubNAligned_], bFp32Local, curN);
                        continue;
                    }
                    if (aValue == -1.0f) {
                        Sub(accumLocal[row * ubNAligned_], accumLocal[row * ubNAligned_], bFp32Local, curN);
                        continue;
                    }
                    if (aValue == 2.0f || aValue == -2.0f) {
                        if (!hasMul2) {
                            if (tmpBusy) {
                                PipeBarrier<PIPE_V>();
                                tmpBusy = false;
                            }
                            Muls(tmpLocal, bFp32Local, 2.0f, curN);
                            PipeBarrier<PIPE_V>();
                            hasMul2 = true;
                        }
                        if (aValue == 2.0f) {
                            Add(accumLocal[row * ubNAligned_], accumLocal[row * ubNAligned_], tmpLocal, curN);
                        } else {
                            Sub(accumLocal[row * ubNAligned_], accumLocal[row * ubNAligned_], tmpLocal, curN);
                        }
                        tmpBusy = true;
                        continue;
                    }
                }
                if (tmpBusy) {
                    PipeBarrier<PIPE_V>();
                    tmpBusy = false;
                }
                hasMul2 = false;
                Muls(tmpLocal, bFp32Local, aValue, curN);
                PipeBarrier<PIPE_V>();
                Add(accumLocal[row * ubNAligned_], accumLocal[row * ubNAligned_], tmpLocal, curN);
                tmpBusy = true;
            }
            PipeBarrier<PIPE_V>();
        }
        ScaleAccumAndStore(accumLocal, globalMOffset, globalNOffset, curM, curN);
    }

    __aicore__ inline void ScaleAccumAndStore(
        LocalTensor<float>& accumLocal,
        uint32_t globalMOffset,
        uint32_t globalNOffset,
        uint32_t curM,
        uint32_t curN)
    {
        LocalTensor<float> tokenScaleLocal;
        if constexpr (TRANS_X1 && !TRANS_X2) {
            tokenScaleLocal = vecTokenScale_.Get<float>();
            DataCopyExtParams tokenCopyParams{1, static_cast<uint32_t>(curM * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> tokenPadParams{false, 0, 0, 0.0f};
            DataCopyPad(tokenScaleLocal, pertokenScaleGm_[globalMOffset], tokenCopyParams, tokenPadParams);
            SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(0));
        }

        const float scaleScalar = scaleLen_ == 1U ? scaleGm_.GetValue(0) : 1.0f;
        for (uint32_t colOffset = 0; colOffset < curN; colOffset += baseN_) {
            const uint32_t tileN = Min(baseN_, curN - colOffset);
            LocalTensor<float> scaleLocal;
            if (scaleLen_ != 1U) {
                scaleLocal = vecQueScale_.AllocTensor<float>();
                DataCopyExtParams scaleCopyParams{1, static_cast<uint32_t>(tileN * sizeof(float)), 0, 0, 0};
                DataCopyPadExtParams<float> scalePadParams{false, 0, 0, 0.0f};
                DataCopyPad(scaleLocal, scaleGm_[globalNOffset + colOffset], scaleCopyParams, scalePadParams);
                vecQueScale_.EnQue<float>(scaleLocal);
                scaleLocal = vecQueScale_.DeQue<float>();
            }

            for (uint32_t row = 0; row < curM; ++row) {
                const uint32_t globalRow = globalMOffset + row;
                float tokenScale = 1.0f;
                if constexpr (TRANS_X1 && !TRANS_X2) {
                    tokenScale = tokenScaleLocal.GetValue(row);
                } else {
                    tokenScale = pertokenScaleGm_.GetValue(globalRow);
                }
                LocalTensor<float> fp32Local = vecFp32Tmp_.Get<float>();
                if (scaleLen_ == 1U) {
                    Muls(fp32Local, accumLocal[row * ubNAligned_ + colOffset], scaleScalar * tokenScale, tileN);
                } else {
                    Mul(fp32Local, accumLocal[row * ubNAligned_ + colOffset], scaleLocal, tileN);
                    PipeBarrier<PIPE_V>();
                    Muls(fp32Local, fp32Local, tokenScale, tileN);
                }
                PipeBarrier<PIPE_V>();

                LocalTensor<bfloat16_t> outLocal = vecQueOut_.AllocTensor<bfloat16_t>();
                Cast(outLocal, fp32Local, RoundMode::CAST_RINT, tileN);
                PipeBarrier<PIPE_V>();
                vecQueOut_.EnQue<bfloat16_t>(outLocal);

                outLocal = vecQueOut_.DeQue<bfloat16_t>();
                DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(tileN * sizeof(bfloat16_t)), 0, 0, 0};
                DataCopyPad(outGm_[static_cast<uint64_t>(globalRow) * n_ + globalNOffset + colOffset],
                    outLocal, copyOutParams);
                vecQueOut_.FreeTensor<bfloat16_t>(outLocal);
            }

            if (scaleLen_ != 1U) {
                vecQueScale_.FreeTensor<float>(scaleLocal);
            }
        }
    }

private:
    static constexpr uint32_t BUFFER_NUM = 1;
    static constexpr uint32_t GM_BLOCK_BYTES = 32;
    static constexpr uint64_t WORKSPACE_ALIGN = 32;
    static constexpr uint32_t FP32_BLOCK_ELEMS = 8;
    static constexpr uint32_t BF16_BLOCK_ELEMS = 16;
    static constexpr uint32_t MATMUL_M_LIMIT = 65535;
    static constexpr uint32_t MATMUL_K_LIMIT = 65535;
    static constexpr uint32_t MATMUL_N_LIMIT = 65535;
    static constexpr uint32_t K_CHUNK_LIMIT = 64512;
    static constexpr uint32_t CASE1_K_CHUNK_LIMIT = 32768;
    static constexpr uint32_t CASE4_K_CHUNK_LIMIT = 65536;
    static constexpr uint32_t CASE9_K_CHUNK_LIMIT = K_CHUNK_LIMIT;
    static constexpr uint32_t LARGE_N_VECTOR_K_LIMIT = 1024;
    static constexpr uint32_t LARGE_M_VECTOR_K_LIMIT = 1024;
    static constexpr uint32_t LARGE_M_VECTOR_N_LIMIT = 128;
    static constexpr uint32_t LARGE_M_TRANS_B_CACHE_ELEMS = 32768;
    static constexpr uint32_t LARGE_M_BLOCK16_OUT_ELEMS = 512;
    static constexpr uint32_t DIRECT_AIC_ROW_STEP = 32;
    static constexpr uint32_t SINGLE_COLUMN_VECTOR_K_TILE = 6144;
    static constexpr uint32_t LARGE_K_TRANS_B_VECTOR_K_TILE = 7168;
    static constexpr uint32_t LARGE_K_TRANS_B_VECTOR_M_LIMIT = 128;
    static constexpr uint32_t LARGE_K_TRANS_B_VECTOR_N_LIMIT = 128;
    static constexpr uint32_t A_COMPACT_K_STEP = 1024;
    static constexpr uint32_t A_COMPACT_COPY_ROWS = 64;

    const QuantBatchMatMulV3TilingData* tilingData_ = nullptr;
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> vecQueA_;
    TQue<QuePosition::VECIN, BUFFER_NUM> vecQueSrc_;
    TQue<QuePosition::VECIN, BUFFER_NUM> vecQueScale_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> vecQueOut_;
    TBuf<TPosition::VECCALC> vecFp32Tmp_;
    TBuf<TPosition::VECCALC> vecLargeNB_;
    TBuf<TPosition::VECCALC> vecHalfTmp_;
    TBuf<TPosition::VECCALC> vecTokenScale_;
    TBuf<TPosition::VECCALC> vecAccum_;

    GlobalTensor<int8_t> x1Gm_;
    GlobalTensor<int8_t> x2Gm_;
    GlobalTensor<int8_t> aCompactGm_;
    GlobalTensor<int32_t> mmOutGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> pertokenScaleGm_;
    GlobalTensor<bfloat16_t> outGm_;

    uint32_t blockIdx_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t m_ = 0;
    uint32_t n_ = 0;
    uint32_t k_ = 0;
    uint32_t scaleLen_ = 0;
    uint32_t singleCoreM_ = 0;
    uint32_t singleCoreN_ = 0;
    uint32_t singleCoreK_ = 0;
    uint32_t baseM_ = 0;
    uint32_t baseN_ = 0;
    uint32_t ubNAligned_ = 0;
    uint32_t aCompactStride_ = 0;
    uint64_t matmulWorkspacePerCore_ = 0;
    uint64_t cWorkspacePerCore_ = 0;
    uint64_t workspacePerCore_ = 0;
    bool isMOuter_ = true;
    bool vectorLargeM_ = false;
};

} // namespace NsQuantBatchMatMulV3
#endif // QUANTBATCHMATMULV3_H
