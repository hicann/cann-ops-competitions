#define K_MAX_SHAPE_DIM 0

#include "kernel_operator.h"

#include <type_traits>

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TQUE_NUM = 1;
constexpr int32_t DIM_MAX_NUM = 4;
constexpr uint32_t UB_FALLBACK_BYTES = 184U * 1024U;
constexpr uint32_t ROW_GROUP_UB_BYTES = 190U * 1024U;
constexpr uint32_t ROW_GROUP_OUTER_TILE = 3U;

#include "scale_v36_fallback.inc"

template <typename T>
class SharedScaleInplaceKernel {
public:
    __aicore__ inline SharedScaleInplaceKernel() = default;

    __aicore__ inline void Init(GM_ADDR x,
                                GM_ADDR scale,
                                GM_ADDR bias,
                                GM_ADDR y,
                                const ScaleTilingData& tilingData,
                                TPipe* pipeIn)
    {
        pipe = pipeIn;
        blockNum = GetBlockNum();
        blockIdx = GetBlockIdx();
        hasBias = tilingData.has_bias;
        elemBytes = static_cast<uint32_t>(sizeof(T));
        alignCount = 32U / elemBytes;
        if (alignCount == 0U) {
            alignCount = 1U;
        }

        outerCount = static_cast<uint32_t>(tilingData.x_dims[0]);
        scaleLen = static_cast<uint32_t>(tilingData.x_dims[1]);
        scaleLenAlign = AlignUp(scaleLen);
        const uint32_t rowBytes = scaleLenAlign * elemBytes;
        const uint32_t coeffVectors = 1U + ((hasBias != 0U) ? 1U : 0U);
        const uint32_t coeffBytes = coeffVectors * rowBytes;
        const uint32_t ubBytes = ROW_GROUP_UB_BYTES;
        uint32_t usableBytes = (ubBytes > coeffBytes + 2U * rowBytes) ? (ubBytes - coeffBytes) : (2U * rowBytes);
        rowBatch = usableBytes / (2U * rowBytes);
        if (rowBatch == 0U) {
            rowBatch = 1U;
        }
        if (rowBatch > outerCount) {
            rowBatch = outerCount;
        }

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), static_cast<uint64_t>(outerCount) * scaleLen);
        scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale), scaleLen);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), static_cast<uint64_t>(outerCount) * scaleLen);
        if (hasBias != 0U) {
            biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias), scaleLen);
        }

        pipe->InitBuffer(inQueueX, 1, rowBatch * rowBytes);
        pipe->InitBuffer(outQueueY, 1, rowBatch * rowBytes);
        pipe->InitBuffer(scaleBuf, rowBytes);
        if (hasBias != 0U) {
            pipe->InitBuffer(biasBuf, rowBytes);
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (!std::is_same_v<T, float>) {
            return;
        } else {
            if (outerCount == 0U || scaleLen == 0U) {
                return;
            }
            LoadScaleBias();

            const uint64_t baseRows = static_cast<uint64_t>(outerCount) / static_cast<uint64_t>(blockNum);
            const uint64_t extraRows = static_cast<uint64_t>(outerCount) % static_cast<uint64_t>(blockNum);
            const uint64_t startRow = static_cast<uint64_t>(blockIdx) * baseRows +
                ((static_cast<uint64_t>(blockIdx) < extraRows) ? static_cast<uint64_t>(blockIdx) : extraRows);
            const uint64_t localRows = baseRows + ((static_cast<uint64_t>(blockIdx) < extraRows) ? 1ULL : 0ULL);

            for (uint64_t row = 0U; row < localRows; row += static_cast<uint64_t>(rowBatch)) {
                uint32_t rowsInTile = rowBatch;
                if (row + static_cast<uint64_t>(rowsInTile) > localRows) {
                    rowsInTile = static_cast<uint32_t>(localRows - row);
                }
                ProcessRows((startRow + row) * static_cast<uint64_t>(scaleLen), rowsInTile);
            }
        }
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t value) const
    {
        return ((value + alignCount - 1U) / alignCount) * alignCount;
    }

    __aicore__ inline void LoadScaleBias()
    {
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copy = {
            static_cast<uint16_t>(1),
            scaleLen * elemBytes,
            0,
            0,
            0
        };

        LocalTensor<T> scaleTmp = inQueueX.AllocTensor<T>();
        if (scaleLen == scaleLenAlign) {
            DataCopy(scaleTmp, scaleGm, scaleLenAlign);
        } else {
            DataCopyPad(scaleTmp, scaleGm, copy, padParams);
        }
        inQueueX.EnQue(scaleTmp);
        scaleTmp = inQueueX.DeQue<T>();
        for (uint32_t i = scaleLen; i < scaleLenAlign; ++i) {
            scaleTmp.SetValue(i, static_cast<T>(0));
        }
        LocalTensor<T> scaleLocal = scaleBuf.Get<T>();
        DataCopy(scaleLocal, scaleTmp, scaleLenAlign);
        inQueueX.FreeTensor(scaleTmp);

        if (hasBias != 0U) {
            LocalTensor<T> biasTmp = inQueueX.AllocTensor<T>();
            if (scaleLen == scaleLenAlign) {
                DataCopy(biasTmp, biasGm, scaleLenAlign);
            } else {
                DataCopyPad(biasTmp, biasGm, copy, padParams);
            }
            inQueueX.EnQue(biasTmp);
            biasTmp = inQueueX.DeQue<T>();
            for (uint32_t i = scaleLen; i < scaleLenAlign; ++i) {
                biasTmp.SetValue(i, static_cast<T>(0));
            }
            LocalTensor<T> biasLocal = biasBuf.Get<T>();
            DataCopy(biasLocal, biasTmp, scaleLenAlign);
            inQueueX.FreeTensor(biasTmp);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessRows(uint64_t xOffset, uint32_t rowsInTile)
    {
        const bool aligned = (scaleLen == scaleLenAlign);
        const uint32_t tileElems = rowsInTile * scaleLenAlign;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copy = {
            static_cast<uint16_t>(rowsInTile),
            scaleLen * elemBytes,
            0,
            0,
            0
        };

        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        if (aligned) {
            DataCopy(xLocal, xGm[xOffset], tileElems);
        } else {
            DataCopyPad(xLocal, xGm[xOffset], copy, padParams);
        }
        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.DeQue<T>();

        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        LocalTensor<T> scaleLocal = scaleBuf.Get<T>();
        for (uint32_t i = 0U; i < rowsInTile; ++i) {
            const uint32_t off = i * scaleLenAlign;
            Mul(yLocal[off], xLocal[off], scaleLocal, scaleLenAlign);
        }
        inQueueX.FreeTensor(xLocal);
        if (hasBias != 0U) {
            LocalTensor<T> biasLocal = biasBuf.Get<T>();
            for (uint32_t i = 0U; i < rowsInTile; ++i) {
                const uint32_t off = i * scaleLenAlign;
                Add(yLocal[off], yLocal[off], biasLocal, scaleLenAlign);
            }
        }

        outQueueY.EnQue(yLocal);
        yLocal = outQueueY.DeQue<T>();
        if (aligned) {
            DataCopy(yGm[xOffset], yLocal, tileElems);
        } else {
            DataCopyPad(yGm[xOffset], yLocal, copy);
        }
        outQueueY.FreeTensor(yLocal);
    }

    TPipe* pipe = nullptr;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<QuePosition::VECCALC> scaleBuf;
    TBuf<QuePosition::VECCALC> biasBuf;

    GlobalTensor<T> xGm;
    GlobalTensor<T> scaleGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<T> yGm;

    uint32_t hasBias = 0U;
    uint32_t blockNum = 1U;
    uint32_t blockIdx = 0U;
    uint32_t elemBytes = 0U;
    uint32_t alignCount = 1U;
    uint32_t outerCount = 0U;
    uint32_t scaleLen = 0U;
    uint32_t scaleLenAlign = 0U;
    uint32_t rowBatch = 1U;
};

template <typename T>
class CompressedRowGroupScaleKernel {
public:
    __aicore__ inline CompressedRowGroupScaleKernel() = default;

    __aicore__ inline void Init(GM_ADDR x,
                                GM_ADDR scale,
                                GM_ADDR bias,
                                GM_ADDR y,
                                const ScaleTilingData& tilingData,
                                TPipe* pipeIn)
    {
        pipe = pipeIn;
        hasBias = tilingData.has_bias;
        ubBytes = ROW_GROUP_UB_BYTES;
        blockNum = GetBlockNum();
        blockIdx = GetBlockIdx();
        elemBytes = static_cast<uint32_t>(sizeof(T));
        alignCount = 32U / elemBytes;
        if (alignCount == 0U) {
            alignCount = 1U;
        }

        ReadCompressedDims(tilingData);
        DeriveCompressedRowGroupLayout(tilingData);

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), totalLengthX);
        scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale), totalLengthScale);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), totalLengthX);
        if (hasBias != 0U) {
            biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias), totalLengthScale);
        }

        const uint32_t tileBytes = maxGroupElemsAlign * elemBytes;
        pipe->InitBuffer(inQueueX, BUFFER_NUM, tileBytes);
        pipe->InitBuffer(outQueueY, TQUE_NUM, tileBytes);
        pipe->InitBuffer(scaleBuf, tileBytes);
        pipe->InitBuffer(scalarScaleBuf, scalarCacheElemsAlign * elemBytes);
        if (hasBias != 0U) {
            pipe->InitBuffer(biasBuf, tileBytes);
            pipe->InitBuffer(scalarBiasBuf, scalarCacheElemsAlign * elemBytes);
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (!std::is_same_v<T, bfloat16_t>) {
            ProcessCompressedRowGroups();
        }
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t value) const
    {
        return ((value + alignCount - 1U) / alignCount) * alignCount;
    }

    __aicore__ inline void ReadCompressedDims(const ScaleTilingData& tilingData)
    {
        dimNum = 0;
        totalLengthX = 1U;
        totalLengthScale = 1U;

        for (int i = 0; i < DIM_MAX_NUM; ++i) {
            if (tilingData.x_dims[i] == 0) {
                break;
            }
            dimX[dimNum] = tilingData.x_dims[i];
            dimScale[dimNum] = tilingData.scale_dims[i];
            totalLengthX *= static_cast<uint64_t>(dimX[dimNum]);
            totalLengthScale *= static_cast<uint64_t>(dimScale[dimNum]);
            ++dimNum;
        }

        for (int i = dimNum; i < DIM_MAX_NUM; ++i) {
            dimX[i] = 0;
            dimScale[i] = 0;
        }
    }

    __aicore__ inline void DeriveCompressedRowGroupLayout(const ScaleTilingData& tilingData)
    {
        outerCount = (dimNum > 0) ? static_cast<uint32_t>(dimX[0]) : 1U;
        scaleRows = (dimNum > 1) ? static_cast<uint32_t>(dimX[1]) : 1U;
        innerCount = (dimNum > 2) ? static_cast<uint32_t>(dimX[2]) : 1U;
        rowCount = totalLengthX / static_cast<uint64_t>(innerCount);

        lastDim = innerCount;
        rowStrideAlign = AlignUp(innerCount);
        compactRowBytes = innerCount * elemBytes;
        paddedRowBytes = rowStrideAlign * elemBytes;
        DeriveCoreRowRange();
        scalarCacheRows = CalcScalarCacheRows();
        scalarCacheElemsAlign = AlignUp((scalarCacheRows == 0U) ? 1U : scalarCacheRows);
        maxGroupRows = CalcMaxGroupRows();
        maxGroupElems = maxGroupRows * innerCount;
        maxGroupElemsAlign = AlignUp(maxGroupElems);
        groupCount = (coreRowCount + maxGroupRows - 1U) / maxGroupRows;
        tileLength = tilingData.tile_length;
        if (tileLength == 0U || tileLength > innerCount) {
            tileLength = innerCount;
        }
        tileLengthAlign = AlignUp(tileLength);

        scaleLastBroadcast = (dimNum > 2 && dimScale[2] == 1);
        outerBroadcast = (dimNum > 0 && dimScale[0] == 1);
        compressedRowGroupPath = (dimNum == 3 && outerBroadcast && scaleLastBroadcast &&
            dimScale[1] == dimX[1] && hasBias != 0U);
    }

    __aicore__ inline void ProcessCompressedRowGroups()
    {
        if (!compressedRowGroupPath || groupCount == 0U) {
            return;
        }

        const uint32_t outerTile = MinU32(outerCount, ROW_GROUP_OUTER_TILE);
        if (scalarCacheRows >= coreRowCount) {
            LoadScaleBiasScalars(coreRowStart, coreRowCount);
            for (uint32_t outerBase = 0U; outerBase < outerCount; outerBase += outerTile) {
                const uint32_t curOuterTile = MinU32(outerTile, outerCount - outerBase);
                const uint32_t cacheGroupCount = (coreRowCount + maxGroupRows - 1U) / maxGroupRows;
                for (uint32_t group = 0U; group < cacheGroupCount; ++group) {
                    const uint32_t scalarStart = group * maxGroupRows;
                    const uint32_t scaleStart = coreRowStart + scalarStart;
                    const uint32_t curRows = MinU32(maxGroupRows, coreRowCount - scalarStart);
                    const uint32_t flatCount = curRows * innerCount;
                    const uint32_t calcCount = AlignUp(flatCount);

                    BuildScaleBiasVectors(scalarStart, curRows, calcCount);
                    ProcessOuterTile(outerBase, curOuterTile, scaleStart, curRows, flatCount, calcCount);
                }
            }
            return;
        }

        for (uint32_t outerBase = 0U; outerBase < outerCount; outerBase += outerTile) {
            const uint32_t curOuterTile = MinU32(outerTile, outerCount - outerBase);
            for (uint32_t cacheOffset = 0U; cacheOffset < coreRowCount; cacheOffset += scalarCacheRows) {
                const uint32_t cacheRows = MinU32(scalarCacheRows, coreRowCount - cacheOffset);
                const uint32_t cacheScaleStart = coreRowStart + cacheOffset;
                LoadScaleBiasScalars(cacheScaleStart, cacheRows);

                const uint32_t cacheGroupCount = (cacheRows + maxGroupRows - 1U) / maxGroupRows;
                for (uint32_t group = 0U; group < cacheGroupCount; ++group) {
                    const uint32_t scalarStart = group * maxGroupRows;
                    const uint32_t scaleStart = cacheScaleStart + scalarStart;
                    const uint32_t curRows = MinU32(maxGroupRows, cacheRows - scalarStart);
                    const uint32_t flatCount = curRows * innerCount;
                    const uint32_t calcCount = AlignUp(flatCount);

                    BuildScaleBiasVectors(scalarStart, curRows, calcCount);
                    ProcessOuterTile(outerBase, curOuterTile, scaleStart, curRows, flatCount, calcCount);
                }
            }
        }
    }

    __aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs) const
    {
        return (lhs < rhs) ? lhs : rhs;
    }

    __aicore__ inline void DeriveCoreRowRange()
    {
        const uint32_t safeBlockNum = (blockNum == 0U) ? 1U : blockNum;
        const uint32_t baseRows = scaleRows / safeBlockNum;
        const uint32_t extraRows = scaleRows % safeBlockNum;
        coreRowStart = blockIdx * baseRows + ((blockIdx < extraRows) ? blockIdx : extraRows);
        coreRowCount = baseRows + ((blockIdx < extraRows) ? 1U : 0U);
        if (coreRowStart >= scaleRows) {
            coreRowStart = scaleRows;
            coreRowCount = 0U;
        } else {
            coreRowCount = MinU32(coreRowCount, scaleRows - coreRowStart);
        }
    }

    __aicore__ inline uint32_t ScalarVectorCount() const
    {
        return 2U;
    }

    __aicore__ inline uint32_t ExpandedVectorCount() const
    {
        return BUFFER_NUM + TQUE_NUM + 2U;
    }

    __aicore__ inline uint64_t ScalarCacheBytesFor(uint32_t rows) const
    {
        const uint32_t alignedRows = AlignUp((rows == 0U) ? 1U : rows);
        return static_cast<uint64_t>(alignedRows) * static_cast<uint64_t>(elemBytes) *
            static_cast<uint64_t>(ScalarVectorCount());
    }

    __aicore__ inline uint64_t ExpandedBytesForRows(uint32_t rows) const
    {
        const uint32_t alignedElems = AlignUp(rows * innerCount);
        return static_cast<uint64_t>(alignedElems) * static_cast<uint64_t>(elemBytes) *
            static_cast<uint64_t>(ExpandedVectorCount());
    }

    __aicore__ inline uint32_t CalcScalarCacheRows()
    {
        if (coreRowCount == 0U || elemBytes == 0U) {
            return 1U;
        }

        const uint64_t minExpandedBytes = ExpandedBytesForRows(1U);
        if (static_cast<uint64_t>(ubBytes) <= minExpandedBytes) {
            return 1U;
        }

        const uint64_t scalarBudget = static_cast<uint64_t>(ubBytes) - minExpandedBytes;
        uint32_t rows = static_cast<uint32_t>(scalarBudget /
            (static_cast<uint64_t>(elemBytes) * static_cast<uint64_t>(ScalarVectorCount())));
        rows = MinU32(rows, coreRowCount);
        if (rows == 0U) {
            rows = 1U;
        }

        while (rows > 1U && ScalarCacheBytesFor(rows) + minExpandedBytes > static_cast<uint64_t>(ubBytes)) {
            --rows;
        }
        return rows;
    }

    __aicore__ inline uint32_t CalcMaxGroupRows()
    {
        if (innerCount == 0U || elemBytes == 0U || scaleRows == 0U) {
            return 1U;
        }

        const uint64_t scalarBytes = ScalarCacheBytesFor(scalarCacheRows);
        if (static_cast<uint64_t>(ubBytes) <= scalarBytes) {
            return 1U;
        }

        const uint64_t vectorBudget = static_cast<uint64_t>(ubBytes) - scalarBytes;
        const uint32_t liveVectors = ExpandedVectorCount();
        const uint64_t compactBytesPerRow = static_cast<uint64_t>(liveVectors) *
            static_cast<uint64_t>(innerCount) * static_cast<uint64_t>(elemBytes);
        if (compactBytesPerRow == 0U) {
            return 1U;
        }

        uint32_t rows = static_cast<uint32_t>(vectorBudget / compactBytesPerRow);
        rows = MinU32(rows, scalarCacheRows);
        if (rows == 0U) {
            rows = 1U;
        }

        while (rows > 1U) {
            const uint64_t actualBytes = ExpandedBytesForRows(rows);
            if (actualBytes <= vectorBudget) {
                break;
            }
            --rows;
        }
        return rows;
    }

    __aicore__ inline void LoadScaleBiasScalars(uint32_t scaleStart, uint32_t rowCount)
    {
        LocalTensor<T> scalarScaleLocal = scalarScaleBuf.Get<T>();
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        DataCopyExtParams scalarCopy = {
            static_cast<uint16_t>(1),
            rowCount * elemBytes,
            0,
            0,
            0
        };
        DataCopyPad(scalarScaleLocal, scaleGm[scaleStart], scalarCopy, padParams);
        LocalTensor<T> scalarBiasLocal = scalarBiasBuf.Get<T>();
        DataCopyPad(scalarBiasLocal, biasGm[scaleStart], scalarCopy, padParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void BuildScaleBiasVectors(uint32_t scalarStart, uint32_t rowBatch, uint32_t calcCount)
    {
        LocalTensor<T> scaleLocal = scaleBuf.Get<T>();
        LocalTensor<T> scalarScaleLocal = scalarScaleBuf.Get<T>();
        LocalTensor<uint16_t> scalarScaleBits = scalarScaleLocal.template ReinterpretCast<uint16_t>();
        LocalTensor<T> biasLocal = biasBuf.Get<T>();
        LocalTensor<T> scalarBiasLocal = scalarBiasBuf.Get<T>();
        LocalTensor<uint16_t> scalarBiasBits = scalarBiasLocal.template ReinterpretCast<uint16_t>();

        for (int32_t row = static_cast<int32_t>(rowBatch) - 1; row >= 0; --row) {
            const uint32_t rowIndex = static_cast<uint32_t>(row);
            const uint32_t rowStart = rowIndex * innerCount;
            const uint32_t alignedStart = (rowStart / alignCount) * alignCount;
            const uint32_t rowEnd = rowStart + innerCount;
            const uint32_t fillEnd = (rowIndex + 1U == rowBatch) ? calcCount : rowEnd;
            const uint32_t fillLen = fillEnd - alignedStart;

            const uint32_t scalarOffset = scalarStart + rowIndex;
            DuplicateHalfVector(scaleLocal, scalarScaleLocal, scalarScaleBits, scalarOffset, alignedStart, fillLen);
            DuplicateHalfVector(biasLocal, scalarBiasLocal, scalarBiasBits, scalarOffset, alignedStart, fillLen);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DuplicateHalfVector(LocalTensor<T> dst,
                                               LocalTensor<T> scalarLocal,
                                               LocalTensor<uint16_t> scalarBits,
                                               uint32_t scalarOffset,
                                               uint32_t alignedStart,
                                               uint32_t fillLen)
    {
        if constexpr (sizeof(T) == 2) {
            LocalTensor<uint16_t> dstBits = dst.template ReinterpretCast<uint16_t>();
            Duplicate(dstBits[alignedStart], scalarBits.GetValue(scalarOffset), fillLen);
        } else {
            Duplicate(dst[alignedStart], scalarLocal.GetValue(scalarOffset), fillLen);
        }
    }

    __aicore__ inline void ProcessOuterTile(uint32_t outerBase,
                                            uint32_t outerTile,
                                            uint32_t scaleStart,
                                            uint32_t rowBatch,
                                            uint32_t flatCount,
                                            uint32_t calcCount)
    {
        LocalTensor<T> scaleLocal = scaleBuf.Get<T>();
        LocalTensor<T> biasLocal = biasBuf.Get<T>();

        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        DataCopyExtParams flatCopy = {
            static_cast<uint16_t>(1),
            flatCount * elemBytes,
            0,
            0,
            0
        };

        CopyOuterInput(scaleStart, outerBase, flatCopy, padParams);
        for (uint32_t outerOffset = 0U; outerOffset < outerTile; ++outerOffset) {
            const uint32_t outer = outerBase + outerOffset;
            const uint64_t rowBase = static_cast<uint64_t>(outer) * static_cast<uint64_t>(scaleRows) +
                static_cast<uint64_t>(scaleStart);
            const uint64_t xOffset = rowBase * static_cast<uint64_t>(innerCount);

            LocalTensor<T> xLocal = inQueueX.DeQue<T>();
            if (outerOffset + 1U < outerTile) {
                CopyOuterInput(scaleStart, outer + 1U, flatCopy, padParams);
            }

            LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
            Mul(yLocal, xLocal, scaleLocal, calcCount);
            Add(yLocal, yLocal, biasLocal, calcCount);
            PipeBarrier<PIPE_V>();
            inQueueX.FreeTensor(xLocal);

            outQueueY.EnQue(yLocal);
            yLocal = outQueueY.DeQue<T>();
            DataCopyPad(yGm[xOffset], yLocal, flatCopy);
            outQueueY.FreeTensor(yLocal);
        }

        (void)rowBatch;
    }

    __aicore__ inline void CopyOuterInput(uint32_t scaleStart,
                                          uint32_t outer,
                                          DataCopyExtParams flatCopy,
                                          DataCopyPadExtParams<T> padParams)
    {
        const uint64_t rowBase = static_cast<uint64_t>(outer) * static_cast<uint64_t>(scaleRows) +
            static_cast<uint64_t>(scaleStart);
        const uint64_t xOffset = rowBase * static_cast<uint64_t>(innerCount);
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopyPad(xLocal, xGm[xOffset], flatCopy, padParams);
        inQueueX.EnQue(xLocal);
    }

    TPipe* pipe = nullptr;
    TQue<QuePosition::VECIN, TQUE_NUM> inQueueX;
    TQue<QuePosition::VECOUT, TQUE_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> scaleBuf;
    TBuf<QuePosition::VECCALC> biasBuf;
    TBuf<QuePosition::VECCALC> scalarScaleBuf;
    TBuf<QuePosition::VECCALC> scalarBiasBuf;

    GlobalTensor<T> xGm;
    GlobalTensor<T> scaleGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<T> yGm;

    int dimNum = 0;
    int dimX[DIM_MAX_NUM] = {};
    int dimScale[DIM_MAX_NUM] = {};

    uint64_t totalLengthX = 0U;
    uint64_t totalLengthScale = 0U;
    uint64_t rowCount = 0U;

    uint32_t hasBias = 0U;
    uint32_t blockNum = 1U;
    uint32_t blockIdx = 0U;
    uint32_t elemBytes = 0U;
    uint32_t alignCount = 1U;
    uint32_t ubBytes = UB_FALLBACK_BYTES;

    uint32_t outerCount = 1U;
    uint32_t scaleRows = 1U;
    uint32_t innerCount = 1U;
    uint32_t lastDim = 1U;
    uint32_t rowStrideAlign = 1U;
    uint32_t compactRowBytes = 0U;
    uint32_t paddedRowBytes = 0U;
    uint32_t coreRowStart = 0U;
    uint32_t coreRowCount = 0U;
    uint32_t scalarCacheRows = 1U;
    uint32_t scalarCacheElemsAlign = 1U;
    uint32_t maxGroupRows = 1U;
    uint32_t maxGroupElems = 1U;
    uint32_t maxGroupElemsAlign = 1U;
    uint32_t groupCount = 0U;
    uint32_t tileLength = 1U;
    uint32_t tileLengthAlign = 1U;

    bool outerBroadcast = false;
    bool scaleLastBroadcast = false;
    bool compressedRowGroupPath = false;
};

template <typename T>
__aicore__ inline bool UseSharedScaleInplacePath(const ScaleTilingData& tilingData)
{
    if constexpr (!std::is_same_v<T, float>) {
        return false;
    }

    int dimNum = 0;
    int xDims[DIM_MAX_NUM] = {};
    int scaleDims[DIM_MAX_NUM] = {};
    for (int i = 0; i < DIM_MAX_NUM; ++i) {
        if (tilingData.x_dims[i] == 0) {
            break;
        }
        xDims[dimNum] = tilingData.x_dims[i];
        scaleDims[dimNum] = tilingData.scale_dims[i];
        ++dimNum;
    }

    if (dimNum != 2) {
        return false;
    }
    return xDims[0] > 1 && xDims[1] > 0 && scaleDims[0] == 1 && scaleDims[1] == xDims[1];
}

template <typename T>
__aicore__ inline bool UseCompressedRowGroupPath(const ScaleTilingData& tilingData)
{
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        return false;
    }
    if (tilingData.has_bias == 0U) {
        return false;
    }

    int dimNum = 0;
    int xDims[DIM_MAX_NUM] = {};
    int scaleDims[DIM_MAX_NUM] = {};
    for (int i = 0; i < DIM_MAX_NUM; ++i) {
        if (tilingData.x_dims[i] == 0) {
            break;
        }
        xDims[dimNum] = tilingData.x_dims[i];
        scaleDims[dimNum] = tilingData.scale_dims[i];
        ++dimNum;
    }

    if (dimNum != 3) {
        return false;
    }
    return scaleDims[0] == 1 && scaleDims[1] == xDims[1] && scaleDims[2] == 1;
}

extern "C" __global__ __aicore__ void scale(GM_ADDR xGm,
                                             GM_ADDR scaleGm,
                                             GM_ADDR biasGm,
                                             GM_ADDR yGm,
                                             GM_ADDR workspace,
                                             GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;

    if (UseSharedScaleInplacePath<DTYPE_X>(tilingData)) {
        SharedScaleInplaceKernel<DTYPE_X> kernel;
        kernel.Init(xGm, scaleGm, biasGm, yGm, tilingData, &pipe);
        kernel.Process();
    } else if (UseCompressedRowGroupPath<DTYPE_X>(tilingData)) {
        CompressedRowGroupScaleKernel<DTYPE_X> kernel;
        kernel.Init(xGm, scaleGm, biasGm, yGm, tilingData, &pipe);
        kernel.Process();
    } else {
        ScaleFallbackV36::FallbackScaleKernel<DTYPE_X> kernel;
        kernel.Init(xGm, scaleGm, biasGm, yGm, tilingData, &pipe);
        kernel.Process();
    }
}
