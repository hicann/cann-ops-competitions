namespace ScaleFallbackV36 {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t CONTIG_BUFFER_NUM = 1;
constexpr int32_t TQUE_NUM = 1;
constexpr int32_t DIM_MAX_NUM = 4;
constexpr uint32_t UB_FALLBACK_BYTES = 184U * 1024U;
constexpr uint32_t MAX_CONTIG_ROWS = 1024U;
constexpr uint32_t SCALAR_CACHE_TARGET = 512U;

template <typename T>
class FallbackScaleKernel {
private:
    static constexpr bool kBf16 = std::is_same_v<T, bfloat16_t>;

public:
    __aicore__ inline FallbackScaleKernel() = default;

    __aicore__ inline void Init(GM_ADDR x,
                                GM_ADDR scale,
                                GM_ADDR bias,
                                GM_ADDR y,
                                const ScaleTilingData& tilingData,
                                TPipe* pipeIn)
    {
        pipe = pipeIn;
        hasBias = tilingData.has_bias;
        ubBytes = tilingData.ub_bytes == 0U ? UB_FALLBACK_BYTES : tilingData.ub_bytes;
        blockNum = GetBlockNum();
        blockIdx = GetBlockIdx();
        alignCount = 32U / static_cast<uint32_t>(sizeof(T));
        if (alignCount == 0U) {
            alignCount = 1U;
        }

        dimNum = 0;
        totalLengthX = 1U;
        totalLengthScale = 1U;
        totalLengthBC = 1U;
        for (int i = 0; i < DIM_MAX_NUM; ++i) {
            if (tilingData.x_dims[i] == 0) {
                break;
            }
            dimX[dimNum] = tilingData.x_dims[i];
            dimScale[dimNum] = tilingData.scale_dims[i];
            dimBC[dimNum] = dimX[dimNum];
            totalLengthX *= static_cast<uint64_t>(dimX[dimNum]);
            totalLengthScale *= static_cast<uint64_t>(dimScale[dimNum]);
            totalLengthBC *= static_cast<uint64_t>(dimBC[dimNum]);
            ++dimNum;
        }

        if (dimNum == 0) {
            dimNum = 1;
            dimX[0] = 1;
            dimScale[0] = 1;
            dimBC[0] = 1;
            totalLengthX = 1U;
            totalLengthScale = 1U;
            totalLengthBC = 1U;
        }

        for (int i = dimNum; i < DIM_MAX_NUM; ++i) {
            dimX[i] = 0;
            dimScale[i] = 0;
            dimBC[i] = 0;
            strideX[i] = 0U;
            strideScale[i] = 0U;
        }

        strideX[dimNum - 1] = 1U;
        strideScale[dimNum - 1] = 1U;
        for (int i = dimNum - 2; i >= 0; --i) {
            strideX[i] = strideX[i + 1] * static_cast<uint64_t>(dimX[i + 1]);
            strideScale[i] = strideScale[i + 1] * static_cast<uint64_t>(dimScale[i + 1]);
        }

        lastDim = static_cast<uint32_t>(dimBC[dimNum - 1]);
        tileLength = tilingData.tile_length;
        if (tileLength == 0U || tileLength > lastDim) {
            tileLength = lastDim;
        }
        tileLengthAlign = AlignUp(tileLength);
        rowCount = totalLengthBC / static_cast<uint64_t>(lastDim);
        chunkCount = (static_cast<uint64_t>(lastDim) + static_cast<uint64_t>(tileLength) - 1U) /
            static_cast<uint64_t>(tileLength);
        totalTaskCount = rowCount * chunkCount;
        scaleLastBroadcast = (dimScale[dimNum - 1] == 1);
        TryEnableContiguousRowPath();
        TryEnableSharedScalePath();
        TryEnablePackedScalarRowsPath();
        TryEnableScalarCachePath();
        TryEnableSequentialScaleRowsPath();

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), totalLengthX);
        scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scale), totalLengthScale);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), totalLengthX);
        if (hasBias != 0U) {
            biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias), totalLengthScale);
        }

        if (contiguousRowPath) {
            pipe->InitBuffer(inQueueX, CONTIG_BUFFER_NUM, contiguousTileAlign * sizeof(T));
            if (!contiguousScalarPath) {
                pipe->InitBuffer(outQueueY, CONTIG_BUFFER_NUM, contiguousTileAlign * sizeof(T));
            }
            pipe->InitBuffer(contiguousScaleBuf, contiguousTileAlign * sizeof(T));
            if (hasBias != 0U) {
                pipe->InitBuffer(contiguousBiasBuf, contiguousTileAlign * sizeof(T));
            }
            return;
        }

        if (packedScalarRowsPath) {
            pipe->InitBuffer(inQueueX,
                             packedOuterPipelinePath ? BUFFER_NUM : CONTIG_BUFFER_NUM,
                             packedRowBatch * packedRowStrideAlign * sizeof(T));
            if (!packedFlatVectorPath) {
                pipe->InitBuffer(outQueueY, CONTIG_BUFFER_NUM, packedRowBatch * packedRowStrideAlign * sizeof(T));
            }
            pipe->InitBuffer(scalarScaleBuf, packedRowBatch * sizeof(T));
            if (hasBias != 0U) {
                pipe->InitBuffer(scalarBiasBuf, packedRowBatch * sizeof(T));
            }
            return;
        }

        if (sharedScalePath) {
            const uint32_t tileElems = sharedRowBatch * sharedScaleLenAlign;
            pipe->InitBuffer(inQueueX, BUFFER_NUM, tileElems * sizeof(T));
            pipe->InitBuffer(outQueueY, BUFFER_NUM, tileElems * sizeof(T));
            if constexpr (kBf16) {
                pipe->InitBuffer(sharedScaleBuf, sharedScaleLenAlign * sizeof(float));
                if (hasBias != 0U) {
                    pipe->InitBuffer(sharedBiasBuf, sharedScaleLenAlign * sizeof(float));
                }
                pipe->InitBuffer(sharedXCalcBuf, tileElems * sizeof(float));
                pipe->InitBuffer(sharedYCalcBuf, tileElems * sizeof(float));
            } else {
                pipe->InitBuffer(sharedScaleBuf, sharedScaleLenAlign * sizeof(T));
                if (hasBias != 0U) {
                    pipe->InitBuffer(sharedBiasBuf, sharedScaleLenAlign * sizeof(T));
                }
            }
            return;
        }

        if (scalarCachePath) {
            pipe->InitBuffer(inQueueX, BUFFER_NUM, tileLengthAlign * sizeof(T));
            pipe->InitBuffer(outQueueY, BUFFER_NUM, tileLengthAlign * sizeof(T));
            pipe->InitBuffer(scalarScaleBuf, scalarCacheTile * sizeof(T));
            if (hasBias != 0U) {
                pipe->InitBuffer(scalarBiasBuf, scalarCacheTile * sizeof(T));
            }
            return;
        }

        pipe->InitBuffer(inQueueX, BUFFER_NUM, tileLengthAlign * sizeof(T));
        pipe->InitBuffer(inQueueScale, BUFFER_NUM, tileLengthAlign * sizeof(T));
        if (hasBias != 0U) {
            pipe->InitBuffer(inQueueBias, BUFFER_NUM, tileLengthAlign * sizeof(T));
        }
        pipe->InitBuffer(outQueueY, BUFFER_NUM, tileLengthAlign * sizeof(T));

        if constexpr (kBf16) {
            pipe->InitBuffer(calcQueueX, BUFFER_NUM, tileLengthAlign * sizeof(float));
            pipe->InitBuffer(calcQueueScale, BUFFER_NUM, tileLengthAlign * sizeof(float));
            if (hasBias != 0U) {
                pipe->InitBuffer(calcQueueBias, BUFFER_NUM, tileLengthAlign * sizeof(float));
            }
            pipe->InitBuffer(calcQueueY, BUFFER_NUM, tileLengthAlign * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (!kBf16) {
            if (contiguousRowPath) {
                ProcessContiguousRows();
                return;
            }
            if (packedScalarRowsPath) {
                ProcessPackedScalarRows();
                return;
            }
        }

        if (sharedScalePath) {
            ProcessSharedScale();
            return;
        }

        if (sequentialScaleRowsPath) {
            ProcessSequentialScaleRows();
            return;
        }

        if (totalTaskCount == 0U) {
            return;
        }

        const uint64_t baseTasks = totalTaskCount / static_cast<uint64_t>(blockNum);
        const uint64_t extraTasks = totalTaskCount % static_cast<uint64_t>(blockNum);
        const uint64_t startTask = static_cast<uint64_t>(blockIdx) * baseTasks +
            ((static_cast<uint64_t>(blockIdx) < extraTasks) ? static_cast<uint64_t>(blockIdx) : extraTasks);
        const uint64_t localTasks = baseTasks + ((static_cast<uint64_t>(blockIdx) < extraTasks) ? 1ULL : 0ULL);
        const uint64_t endTask = startTask + localTasks;

        for (uint64_t task = startTask; task < endTask; ++task) {
            ProcessTask(task);
        }
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t count) const
    {
        return ((count + alignCount - 1U) / alignCount) * alignCount;
    }

    __aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs) const
    {
        return (lhs < rhs) ? lhs : rhs;
    }

    __aicore__ inline uint64_t MinU64(uint64_t lhs, uint64_t rhs) const
    {
        return (lhs < rhs) ? lhs : rhs;
    }

    __aicore__ inline void TryEnableContiguousRowPath()
    {
        contiguousRowPath = false;
        contiguousScalarPath = false;
        contiguousScaleRows = 0U;
        contiguousRowBatch = 1U;
        contiguousTileAlign = 0U;
        contiguousRowStride = 0U;

        if constexpr (kBf16) {
            return;
        }
        if (!scaleLastBroadcast || chunkCount != 1U || dimNum < 2) {
            return;
        }
        if (lastDim == 0U || lastDim != tileLength) {
            return;
        }

        for (int i = 0; i < dimNum - 2; ++i) {
            if (dimScale[i] != 1) {
                return;
            }
        }
        if (dimScale[dimNum - 2] <= 1) {
            return;
        }

        contiguousScaleRows = static_cast<uint32_t>(dimScale[dimNum - 2]);
        const uint32_t rowStride = lastDim;
        const bool rowAligned = (AlignUp(lastDim) == lastDim);
        if (!rowAligned) {
            return;
        }
        const uint32_t elemBytes = static_cast<uint32_t>(sizeof(T));
        const uint32_t biasFactor = (hasBias != 0U) ? 1U : 0U;
        const uint32_t expandBytesPerElem = (3U + biasFactor) * elemBytes;
        if (expandBytesPerElem == 0U) {
            return;
        }

        uint32_t expandRows = 0U;
        uint32_t expandTileAlign = 0U;
        uint32_t maxRows = ubBytes / (rowStride * expandBytesPerElem);
        if (maxRows > contiguousScaleRows) {
            maxRows = contiguousScaleRows;
        }
        if (maxRows > MAX_CONTIG_ROWS) {
            maxRows = MAX_CONTIG_ROWS;
        }

        while (maxRows > 1U) {
            uint64_t rawTileElems = static_cast<uint64_t>(rowStride) * static_cast<uint64_t>(maxRows);
            uint64_t tileElems = static_cast<uint64_t>(AlignUp(static_cast<uint32_t>(rawTileElems)));
            if (tileElems > static_cast<uint64_t>(UINT32_MAX)) {
                --maxRows;
                continue;
            }

            uint32_t alignedCount = static_cast<uint32_t>(tileElems);
            uint64_t requiredBytes = static_cast<uint64_t>(alignedCount) * static_cast<uint64_t>(expandBytesPerElem);
            if (requiredBytes <= static_cast<uint64_t>(ubBytes)) {
                expandRows = maxRows;
                expandTileAlign = alignedCount;
                break;
            }
            --maxRows;
        }

        uint32_t scalarRows = 0U;
        uint32_t scalarTileAlign = 0U;
        if (rowAligned) {
            const uint64_t scalarPerRowBytes =
                static_cast<uint64_t>(rowStride) * static_cast<uint64_t>(2U * BUFFER_NUM) * static_cast<uint64_t>(elemBytes) +
                static_cast<uint64_t>(1U + biasFactor) * static_cast<uint64_t>(elemBytes);
            if (scalarPerRowBytes != 0U) {
                maxRows = static_cast<uint32_t>(static_cast<uint64_t>(ubBytes) / scalarPerRowBytes);
                if (maxRows > contiguousScaleRows) {
                    maxRows = contiguousScaleRows;
                }
                if (maxRows > MAX_CONTIG_ROWS) {
                    maxRows = MAX_CONTIG_ROWS;
                }
                while (maxRows > 1U) {
                    uint64_t rawTileElems = static_cast<uint64_t>(rowStride) * static_cast<uint64_t>(maxRows);
                    uint64_t tileElems = static_cast<uint64_t>(AlignUp(static_cast<uint32_t>(rawTileElems)));
                    if (tileElems > static_cast<uint64_t>(UINT32_MAX)) {
                        --maxRows;
                        continue;
                    }

                    const uint32_t alignedCount = static_cast<uint32_t>(tileElems);
                    const uint64_t requiredBytes =
                        static_cast<uint64_t>(alignedCount) * static_cast<uint64_t>(2U * BUFFER_NUM) * static_cast<uint64_t>(elemBytes) +
                        static_cast<uint64_t>(maxRows) * static_cast<uint64_t>(1U + biasFactor) * static_cast<uint64_t>(elemBytes);
                    if (requiredBytes <= static_cast<uint64_t>(ubBytes)) {
                        scalarRows = maxRows;
                        scalarTileAlign = alignedCount;
                        break;
                    }
                    --maxRows;
                }
            }
        }

        if (scalarRows > expandRows && scalarRows > 1U) {
            contiguousScalarPath = true;
            contiguousRowBatch = scalarRows;
            contiguousTileAlign = scalarTileAlign;
            contiguousRowStride = rowStride;
            contiguousRowPath = true;
            return;
        }

        if (expandRows > 1U) {
            contiguousRowBatch = expandRows;
            contiguousTileAlign = expandTileAlign;
            contiguousRowStride = rowStride;
            contiguousRowPath = true;
        }
    }

    __aicore__ inline void TryEnableSharedScalePath()
    {
        sharedScalePath = false;
        sharedOuterCount = 0U;
        sharedScaleLen = 0U;
        sharedScaleLenAlign = 0U;
        sharedRowBatch = 1U;

        if (contiguousRowPath) {
            return;
        }
        if (dimNum <= 0) {
            return;
        }

        int splitIdx = dimNum;
        for (int i = 0; i < dimNum; ++i) {
            if (dimScale[i] != 1) {
                splitIdx = i;
                break;
            }
        }
        for (int i = splitIdx; i < dimNum; ++i) {
            if (dimScale[i] != dimX[i]) {
                return;
            }
        }

        uint64_t outerCount = 1U;
        for (int i = 0; i < splitIdx; ++i) {
            outerCount *= static_cast<uint64_t>(dimX[i]);
        }
        uint64_t scaleLen = 1U;
        for (int i = splitIdx; i < dimNum; ++i) {
            scaleLen *= static_cast<uint64_t>(dimX[i]);
        }
        if (scaleLen == 0U || scaleLen > static_cast<uint64_t>(UINT32_MAX)) {
            return;
        }

        const uint32_t scaleLenU = static_cast<uint32_t>(scaleLen);
        const uint32_t scaleLenAlign = AlignUp(scaleLenU);
        const uint32_t biasFactor = (hasBias != 0U) ? 1U : 0U;
        if (outerCount < static_cast<uint64_t>(blockNum)) {
            return;
        }

        const uint32_t scaleBytes = scaleLenU * static_cast<uint32_t>(sizeof(T));
        if (scaleBytes < 64U) {
            return;
        }

        constexpr uint32_t TARGET_TILE_BYTES = 16384U;
        uint32_t rowBytes = scaleLenAlign * static_cast<uint32_t>(sizeof(T));
        uint32_t rowBatch = 1U;
        if (scaleLenU == scaleLenAlign && rowBytes > 0U && rowBytes < TARGET_TILE_BYTES) {
            rowBatch = TARGET_TILE_BYTES / rowBytes;
            if (rowBatch == 0U) {
                rowBatch = 1U;
            }
        }
        const uint64_t baseRows = outerCount / static_cast<uint64_t>(blockNum);
        if (baseRows > 0U && rowBatch > static_cast<uint32_t>(baseRows)) {
            rowBatch = static_cast<uint32_t>(baseRows);
        }
        if (rowBatch == 0U) {
            rowBatch = 1U;
        }

        uint64_t persistBytes = 0U;
        if constexpr (kBf16) {
            persistBytes = static_cast<uint64_t>(scaleLenAlign) * sizeof(float)
                         * (1ULL + biasFactor);
        } else {
            persistBytes = static_cast<uint64_t>(scaleLenAlign) * sizeof(T)
                         * (1ULL + biasFactor);
        }
        uint64_t perRowBytes = static_cast<uint64_t>(scaleLenAlign) * sizeof(T)
                             * 2ULL * BUFFER_NUM;
        if constexpr (kBf16) {
            perRowBytes += static_cast<uint64_t>(scaleLenAlign) * sizeof(float) * 2ULL;
        }
        while (rowBatch > 1U &&
               persistBytes + static_cast<uint64_t>(rowBatch) * perRowBytes >
                   static_cast<uint64_t>(ubBytes)) {
            --rowBatch;
        }
        if (persistBytes + static_cast<uint64_t>(rowBatch) * perRowBytes >
                static_cast<uint64_t>(ubBytes)) {
            return;
        }

        
        if (static_cast<uint64_t>(rowBatch) * baseRows < 4U) {
            return;
        }

        sharedScalePath = true;
        sharedOuterCount = outerCount;
        sharedScaleLen = scaleLenU;
        sharedScaleLenAlign = scaleLenAlign;
        sharedRowBatch = rowBatch;
    }

    __aicore__ inline void TryEnableScalarCachePath()
    {
        scalarCachePath = false;
        scalarCacheBase = 0U;
        scalarCacheCount = 0U;
        scalarCacheTile = 0U;

        if constexpr (kBf16) {
            return;
        }
        if (contiguousRowPath || sharedScalePath || !scaleLastBroadcast) {
            return;
        }
        if (tileLengthAlign == 0U) {
            return;
        }

        uint64_t cacheBytes = static_cast<uint64_t>(ubBytes);
        const uint64_t queueBytes = static_cast<uint64_t>(tileLengthAlign) * sizeof(T) * 2ULL * BUFFER_NUM;
        if (cacheBytes <= queueBytes) {
            return;
        }
        cacheBytes -= queueBytes;

        const uint32_t biasFactor = (hasBias != 0U) ? 1U : 0U;
        const uint32_t scalarBytes = static_cast<uint32_t>(sizeof(T)) * (1U + biasFactor);
        if (scalarBytes == 0U) {
            return;
        }

        uint32_t cacheTile = static_cast<uint32_t>(cacheBytes / static_cast<uint64_t>(scalarBytes));
        if (cacheTile < alignCount) {
            return;
        }
        if (cacheTile > totalLengthScale) {
            cacheTile = static_cast<uint32_t>(totalLengthScale);
        }
        if (cacheTile > SCALAR_CACHE_TARGET) {
            cacheTile = SCALAR_CACHE_TARGET;
        }
        if (cacheTile < alignCount) {
            cacheTile = alignCount;
        }

        scalarCacheTile = cacheTile;
        scalarCachePath = true;
    }

    __aicore__ inline void TryEnablePackedScalarRowsPath()
    {
        packedScalarRowsPath = false;
        packedFlatVectorPath = (lastDim >= (alignCount * 4U));
        packedOuterPipelinePath = packedFlatVectorPath;
        packedScaleRows = 0U;
        packedRowBatch = 1U;
        packedRowStrideAlign = 0U;

        if constexpr (kBf16) {
            return;
        }
        if (contiguousRowPath || sharedScalePath || !scaleLastBroadcast || chunkCount != 1U || dimNum < 2) {
            return;
        }

        for (int i = 0; i < dimNum - 2; ++i) {
            if (dimScale[i] != 1) {
                return;
            }
        }
        if (dimScale[dimNum - 2] <= 1) {
            return;
        }

        const uint32_t rowStrideAlign = AlignUp(lastDim);
        if (lastDim == 0U || rowStrideAlign == lastDim || lastDim > 1024U) {
            return;
        }

        packedFlatVectorPath = (lastDim >= (alignCount * 4U));
        packedOuterPipelinePath = packedFlatVectorPath;
        const uint32_t biasFactor = (hasBias != 0U) ? 1U : 0U;
        const uint32_t elemBytes = static_cast<uint32_t>(sizeof(T));
        uint64_t bytesPerRow =
            static_cast<uint64_t>(rowStrideAlign) * elemBytes * static_cast<uint64_t>(packedOuterPipelinePath ? 2U : 1U);
        if (packedFlatVectorPath) {
            bytesPerRow += static_cast<uint64_t>(1U + biasFactor) * elemBytes;
        } else {
            bytesPerRow += static_cast<uint64_t>(rowStrideAlign) * elemBytes;
            bytesPerRow += static_cast<uint64_t>(1U + biasFactor) * elemBytes;
        }
        if (bytesPerRow == 0U) {
            return;
        }

        uint32_t rowBatch = static_cast<uint32_t>(static_cast<uint64_t>(ubBytes) / bytesPerRow);
        if (rowBatch > static_cast<uint32_t>(dimScale[dimNum - 2])) {
            rowBatch = static_cast<uint32_t>(dimScale[dimNum - 2]);
        }
        if (rowBatch > MAX_CONTIG_ROWS) {
            rowBatch = MAX_CONTIG_ROWS;
        }
        if (packedFlatVectorPath && hasBias != 0U && rowBatch > 240U) {
            rowBatch = 240U;
        }
        if (rowBatch < 8U) {
            return;
        }

        packedScaleRows = static_cast<uint32_t>(dimScale[dimNum - 2]);
        packedRowBatch = rowBatch;
        packedRowStrideAlign = rowStrideAlign;
        packedScalarRowsPath = true;
    }

    __aicore__ inline void TryEnableSequentialScaleRowsPath()
    {
        sequentialScaleRowsPath = false;
        sequentialScaleRows = 0U;

        if (contiguousRowPath || packedScalarRowsPath || sharedScalePath || !scaleLastBroadcast || chunkCount != 1U || dimNum < 2) {
            return;
        }

        for (int i = 0; i < dimNum - 2; ++i) {
            if (dimScale[i] != 1) {
                return;
            }
        }
        if (dimScale[dimNum - 2] <= 1) {
            return;
        }

        sequentialScaleRows = static_cast<uint32_t>(dimScale[dimNum - 2]);
        sequentialScaleRowsPath = true;
    }

    __aicore__ inline T GetCachedScaleValue(uint32_t scaleOffset)
    {
        if (scaleOffset < scalarCacheBase || scaleOffset >= (scalarCacheBase + scalarCacheCount)) {
            LoadScalarCacheRange(scaleOffset, scalarCacheTile);
        }

        LocalTensor<T> scaleCache = scalarScaleBuf.Get<T>();
        return scaleCache.GetValue(scaleOffset - scalarCacheBase);
    }

    __aicore__ inline T GetCachedBiasValue(uint32_t scaleOffset)
    {
        LocalTensor<T> biasCache = scalarBiasBuf.Get<T>();
        return biasCache.GetValue(scaleOffset - scalarCacheBase);
    }

    __aicore__ inline void LoadScalarCacheRange(uint32_t scaleOffset, uint32_t requestedCount)
    {
        const uint32_t remain = static_cast<uint32_t>(totalLengthScale) - scaleOffset;
        scalarCacheBase = scaleOffset;
        scalarCacheCount = MinU32(requestedCount, remain);

        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copy = {
            static_cast<uint16_t>(1),
            scalarCacheCount * static_cast<uint32_t>(sizeof(T)),
            0,
            0,
            0
        };

        LocalTensor<T> scaleCache = scalarScaleBuf.Get<T>();
        DataCopyPad(scaleCache, scaleGm[scalarCacheBase], copy, padParams);
        if (hasBias != 0U) {
            LocalTensor<T> biasCache = scalarBiasBuf.Get<T>();
            DataCopyPad(biasCache, biasGm[scalarCacheBase], copy, padParams);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessContiguousRows()
    {
        const uint64_t baseRows = rowCount / static_cast<uint64_t>(blockNum);
        const uint64_t extraRows = rowCount % static_cast<uint64_t>(blockNum);
        const uint64_t startRow = static_cast<uint64_t>(blockIdx) * baseRows +
            ((static_cast<uint64_t>(blockIdx) < extraRows) ? static_cast<uint64_t>(blockIdx) : extraRows);
        const uint64_t localRows = baseRows + ((static_cast<uint64_t>(blockIdx) < extraRows) ? 1ULL : 0ULL);
        const uint64_t endRow = startRow + localRows;

        uint64_t currentRow = startRow;
        while (currentRow < endRow) {
            const uint32_t scaleStart = static_cast<uint32_t>(currentRow % static_cast<uint64_t>(contiguousScaleRows));
            const uint64_t remainOuter = static_cast<uint64_t>(contiguousScaleRows - scaleStart);
            const uint64_t remainAssigned = MinU64(endRow - currentRow, remainOuter);

            uint64_t handled = 0U;
            while (handled < remainAssigned) {
                const uint32_t curRows = MinU32(contiguousRowBatch, static_cast<uint32_t>(remainAssigned - handled));
                const uint64_t rowBase = currentRow + handled;
                ExecuteContiguousRows(rowBase * static_cast<uint64_t>(lastDim), scaleStart + static_cast<uint32_t>(handled), curRows);
                handled += static_cast<uint64_t>(curRows);
            }

            currentRow += remainAssigned;
        }
    }

    __aicore__ inline void ProcessSequentialScaleRows()
    {
        const uint64_t baseRows = rowCount / static_cast<uint64_t>(blockNum);
        const uint64_t extraRows = rowCount % static_cast<uint64_t>(blockNum);
        const uint64_t startRow = static_cast<uint64_t>(blockIdx) * baseRows +
            ((static_cast<uint64_t>(blockIdx) < extraRows) ? static_cast<uint64_t>(blockIdx) : extraRows);
        const uint64_t localRows = baseRows + ((static_cast<uint64_t>(blockIdx) < extraRows) ? 1ULL : 0ULL);
        const uint64_t endRow = startRow + localRows;

        uint64_t currentRow = startRow;
        uint64_t xOffset = currentRow * static_cast<uint64_t>(lastDim);
        while (currentRow < endRow) {
            const uint32_t scaleStart = static_cast<uint32_t>(currentRow % static_cast<uint64_t>(sequentialScaleRows));
            const uint64_t remainOuter = static_cast<uint64_t>(sequentialScaleRows - scaleStart);
            const uint64_t remainAssigned = MinU64(endRow - currentRow, remainOuter);

            uint64_t handled = 0U;
            while (handled < remainAssigned) {
                if constexpr (!kBf16) {
                    if (scalarCachePath) {
                        const uint32_t chunkRows = MinU32(
                            scalarCacheTile,
                            static_cast<uint32_t>(remainAssigned - handled));
                        LoadScalarCacheRange(scaleStart + static_cast<uint32_t>(handled), chunkRows);
                        LocalTensor<T> scaleCache = scalarScaleBuf.Get<T>();
                        LocalTensor<T> biasCache;
                        if (hasBias != 0U) {
                            biasCache = scalarBiasBuf.Get<T>();
                        }

                        for (uint32_t localRow = 0U; localRow < chunkRows; ++localRow) {
                            const T scaleValue = scaleCache.GetValue(localRow);
                            const T biasValue = (hasBias != 0U) ? biasCache.GetValue(localRow) : static_cast<T>(0);
                            ExecuteChunkWithScalarValue(xOffset, scaleValue, biasValue, lastDim);
                            xOffset += static_cast<uint64_t>(lastDim);
                        }
                        handled += static_cast<uint64_t>(chunkRows);
                        continue;
                    }
                }

                ExecuteChunk(xOffset,
                             static_cast<uint64_t>(scaleStart) + handled,
                             lastDim);
                xOffset += static_cast<uint64_t>(lastDim);
                ++handled;
            }
            currentRow += remainAssigned;
        }
    }

    __aicore__ inline void ProcessPackedScalarRows()
    {
        if (packedScaleRows == 0U) {
            return;
        }

        const uint64_t outerCount = rowCount / static_cast<uint64_t>(packedScaleRows);
        if (outerCount <= 1U || (rowCount % static_cast<uint64_t>(packedScaleRows)) != 0U) {
            const uint64_t baseRows = rowCount / static_cast<uint64_t>(blockNum);
            const uint64_t extraRows = rowCount % static_cast<uint64_t>(blockNum);
            const uint64_t startRow = static_cast<uint64_t>(blockIdx) * baseRows +
                ((static_cast<uint64_t>(blockIdx) < extraRows) ? static_cast<uint64_t>(blockIdx) : extraRows);
            const uint64_t localRows = baseRows + ((static_cast<uint64_t>(blockIdx) < extraRows) ? 1ULL : 0ULL);
            const uint64_t endRow = startRow + localRows;

            uint64_t currentRow = startRow;
            uint64_t xOffset = currentRow * static_cast<uint64_t>(lastDim);
            while (currentRow < endRow) {
                const uint32_t scaleStart = static_cast<uint32_t>(currentRow % static_cast<uint64_t>(packedScaleRows));
                const uint64_t remainOuter = static_cast<uint64_t>(packedScaleRows - scaleStart);
                const uint64_t remainAssigned = MinU64(endRow - currentRow, remainOuter);

                uint64_t handled = 0U;
                while (handled < remainAssigned) {
                    const uint32_t curRows = MinU32(packedRowBatch, static_cast<uint32_t>(remainAssigned - handled));
                    ExecutePackedScalarRows(xOffset, scaleStart + static_cast<uint32_t>(handled), curRows);
                    xOffset += static_cast<uint64_t>(curRows) * static_cast<uint64_t>(lastDim);
                    handled += static_cast<uint64_t>(curRows);
                }
                currentRow += remainAssigned;
            }
            return;
        }

        const uint64_t baseScaleRows = static_cast<uint64_t>(packedScaleRows) / static_cast<uint64_t>(blockNum);
        const uint64_t extraScaleRows = static_cast<uint64_t>(packedScaleRows) % static_cast<uint64_t>(blockNum);
        const uint64_t startScaleRow = static_cast<uint64_t>(blockIdx) * baseScaleRows +
            ((static_cast<uint64_t>(blockIdx) < extraScaleRows) ? static_cast<uint64_t>(blockIdx) : extraScaleRows);
        const uint64_t localScaleRows = baseScaleRows + ((static_cast<uint64_t>(blockIdx) < extraScaleRows) ? 1ULL : 0ULL);
        const uint64_t endScaleRow = startScaleRow + localScaleRows;

        uint64_t scaleRow = startScaleRow;
        while (scaleRow < endScaleRow) {
            const uint32_t curRows = MinU32(packedRowBatch, static_cast<uint32_t>(endScaleRow - scaleRow));
            LoadScalarCacheRange(static_cast<uint32_t>(scaleRow), curRows);

            if (packedFlatVectorPath && packedOuterPipelinePath) {
                ExecutePackedScalarRowsExpandedOuterLoop(scaleRow, curRows, outerCount);
            } else {
                for (uint64_t outerIdx = 0U; outerIdx < outerCount; ++outerIdx) {
                    const uint64_t rowBase = outerIdx * static_cast<uint64_t>(packedScaleRows) + scaleRow;
                    if (packedFlatVectorPath) {
                        ExecutePackedScalarRowsExpanded(rowBase * static_cast<uint64_t>(lastDim), curRows);
                    } else {
                        ExecutePackedScalarRowsFromCache(rowBase * static_cast<uint64_t>(lastDim), curRows);
                    }
                }
            }

            scaleRow += static_cast<uint64_t>(curRows);
        }
    }

    __aicore__ inline void ProcessPackedScalarRowsOuterRepeat(uint64_t startScaleRow,
                                                              uint64_t endScaleRow,
                                                              uint64_t outerCount)
    {
        uint64_t scaleRow = startScaleRow;
        while (scaleRow < endScaleRow) {
            const uint32_t curRows = MinU32(packedRowBatch, static_cast<uint32_t>(endScaleRow - scaleRow));
            LoadScalarCacheRange(static_cast<uint32_t>(scaleRow), curRows);
            for (uint32_t localRow = 0U; localRow < curRows; ++localRow) {
                ExecutePackedScaleRowOuterRepeat(scaleRow + static_cast<uint64_t>(localRow), localRow, outerCount);
            }
            scaleRow += static_cast<uint64_t>(curRows);
        }
    }

    __aicore__ inline void ExecutePackedScaleRowOuterRepeat(uint64_t scaleRow,
                                                            uint32_t localScaleIndex,
                                                            uint64_t outerCount)
    {
        LocalTensor<T> scaleCache = scalarScaleBuf.Get<T>();
        LocalTensor<T> biasCache;
        if (hasBias != 0U) {
            biasCache = scalarBiasBuf.Get<T>();
        }
        const T scaleValue = scaleCache.GetValue(localScaleIndex);
        const T biasValue = (hasBias != 0U) ? biasCache.GetValue(localScaleIndex) : static_cast<T>(0);
        const uint32_t outerRows = static_cast<uint32_t>(outerCount);
        const uint32_t blockLenBytes = lastDim * static_cast<uint32_t>(sizeof(T));
        const uint32_t gmStrideBytes = static_cast<uint32_t>(
            (static_cast<uint64_t>(packedScaleRows - 1U) * static_cast<uint64_t>(lastDim)) *
            static_cast<uint64_t>(sizeof(T)));

        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyRows = {
            static_cast<uint16_t>(outerRows),
            blockLenBytes,
            gmStrideBytes,
            0,
            0
        };
        DataCopyExtParams storeRows = {
            static_cast<uint16_t>(outerRows),
            blockLenBytes,
            0,
            gmStrideBytes,
            0
        };

        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopyPad(xLocal, xGm[scaleRow * static_cast<uint64_t>(lastDim)], copyRows, padParams);
        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.DeQue<T>();

        for (uint32_t outerRow = 0U; outerRow < outerRows; ++outerRow) {
            const uint32_t rowOffset = outerRow * packedRowStrideAlign;
            Muls(xLocal[rowOffset], xLocal[rowOffset], scaleValue, lastDim);
            if (hasBias != 0U) {
                Adds(xLocal[rowOffset], xLocal[rowOffset], biasValue, lastDim);
            }
        }
        PipeBarrier<PIPE_V>();

        DataCopyPad(yGm[scaleRow * static_cast<uint64_t>(lastDim)], xLocal, storeRows);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void ExecutePackedScalarRows(uint64_t xOffset, uint32_t scaleOffset, uint32_t rowBatch)
    {
        LoadScalarCacheRange(scaleOffset, rowBatch);
        if (packedFlatVectorPath) {
            ExecutePackedScalarRowsExpanded(xOffset, rowBatch);
            return;
        }
        ExecutePackedScalarRowsFromCache(xOffset, rowBatch);
    }

    __aicore__ inline void CopyPackedRowsToLocal(LocalTensor<T>& xLocal, uint64_t xOffset, uint32_t rowBatch)
    {
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyRows = {
            static_cast<uint16_t>(rowBatch),
            lastDim * static_cast<uint32_t>(sizeof(T)),
            0,
            0,
            0
        };
        xLocal = inQueueX.AllocTensor<T>();
        DataCopyPad(xLocal, xGm[xOffset], copyRows, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void ComputePackedRowsExpandedInplace(LocalTensor<T>& xLocal, uint64_t xOffset, uint32_t rowBatch)
    {
        LocalTensor<T> scaleCache = scalarScaleBuf.Get<T>();
        LocalTensor<T> biasCache;
        if (hasBias != 0U) {
            biasCache = scalarBiasBuf.Get<T>();
        }

        for (uint32_t row = 0U; row < rowBatch; ++row) {
            const uint32_t rowOffset = row * packedRowStrideAlign;
            const T scaleValue = scaleCache.GetValue(row);
            Muls(xLocal[rowOffset], xLocal[rowOffset], scaleValue, packedRowStrideAlign);
            if (hasBias != 0U) {
                const T biasValue = biasCache.GetValue(row);
                Adds(xLocal[rowOffset], xLocal[rowOffset], biasValue, packedRowStrideAlign);
            }
        }
        PipeBarrier<PIPE_V>();

        DataCopyExtParams copyRows = {
            static_cast<uint16_t>(rowBatch),
            lastDim * static_cast<uint32_t>(sizeof(T)),
            0,
            0,
            0
        };
        DataCopyPad(yGm[xOffset], xLocal, copyRows);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void ExecutePackedScalarRowsExpandedOuterLoop(uint64_t scaleRow,
                                                                    uint32_t rowBatch,
                                                                    uint64_t outerCount)
    {
        if (outerCount == 0U) {
            return;
        }

        uint64_t xOffset = scaleRow * static_cast<uint64_t>(lastDim);
        LocalTensor<T> xLocal;
        CopyPackedRowsToLocal(xLocal, xOffset, rowBatch);

        for (uint64_t outerIdx = 0U; outerIdx < outerCount; ++outerIdx) {
            xLocal = inQueueX.DeQue<T>();
            const uint64_t currentOffset =
                (outerIdx * static_cast<uint64_t>(packedScaleRows) + scaleRow) * static_cast<uint64_t>(lastDim);
            if (outerIdx + 1U < outerCount) {
                const uint64_t nextOffset =
                    ((outerIdx + 1U) * static_cast<uint64_t>(packedScaleRows) + scaleRow) *
                    static_cast<uint64_t>(lastDim);
                LocalTensor<T> nextLocal;
                CopyPackedRowsToLocal(nextLocal, nextOffset, rowBatch);
            }
            ComputePackedRowsExpandedInplace(xLocal, currentOffset, rowBatch);
        }
    }

    __aicore__ inline void ExpandPackedScaleBiasFromCache(uint32_t rowBatch)
    {
        LocalTensor<T> scaleCache = scalarScaleBuf.Get<T>();
        LocalTensor<T> biasCache;
        if (hasBias != 0U) {
            biasCache = scalarBiasBuf.Get<T>();
        }

        for (int32_t row = static_cast<int32_t>(rowBatch) - 1; row >= 0; --row) {
            const uint32_t rowIndex = static_cast<uint32_t>(row);
            const uint32_t rowOffset = rowIndex * packedRowStrideAlign;
            const T scaleValue = scaleCache.GetValue(rowIndex);
            Duplicate(scaleCache[rowOffset], scaleValue, packedRowStrideAlign);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ExecutePackedScalarRowsExpanded(uint64_t xOffset, uint32_t rowBatch)
    {
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        LocalTensor<T> scaleCache = scalarScaleBuf.Get<T>();
        LocalTensor<T> biasCache;
        if (hasBias != 0U) {
            biasCache = scalarBiasBuf.Get<T>();
        }

        DataCopyExtParams copyRows = {
            static_cast<uint16_t>(rowBatch),
            lastDim * static_cast<uint32_t>(sizeof(T)),
            0,
            0,
            0
        };

        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopyPad(xLocal, xGm[xOffset], copyRows, padParams);
        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.DeQue<T>();

        for (uint32_t row = 0U; row < rowBatch; ++row) {
            const uint32_t rowOffset = row * packedRowStrideAlign;
            const T scaleValue = scaleCache.GetValue(row);
            Muls(xLocal[rowOffset], xLocal[rowOffset], scaleValue, packedRowStrideAlign);
            if (hasBias != 0U) {
                const T biasValue = biasCache.GetValue(row);
                Adds(xLocal[rowOffset], xLocal[rowOffset], biasValue, packedRowStrideAlign);
            }
        }
        PipeBarrier<PIPE_V>();

        DataCopyPad(yGm[xOffset], xLocal, copyRows);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void ExecutePackedScalarRowsFromCache(uint64_t xOffset, uint32_t rowBatch)
    {
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        LocalTensor<T> scaleCache = scalarScaleBuf.Get<T>();
        LocalTensor<T> biasCache;
        if (hasBias != 0U) {
            biasCache = scalarBiasBuf.Get<T>();
        }

        DataCopyExtParams copyRows = {
            static_cast<uint16_t>(rowBatch),
            lastDim * static_cast<uint32_t>(sizeof(T)),
            0,
            0,
            0
        };

        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopyPad(xLocal, xGm[xOffset], copyRows, padParams);
        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.DeQue<T>();

        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        const uint32_t rowChunk = MinU32(lastDim, alignCount * 16U);
        for (uint32_t row = 0U; row < rowBatch; ++row) {
            const uint32_t rowOffset = row * packedRowStrideAlign;
            const T scaleValue = scaleCache.GetValue(row);
            const T biasValue = (hasBias != 0U) ? biasCache.GetValue(row) : static_cast<T>(0);
            Muls(yLocal[rowOffset], xLocal[rowOffset], scaleValue, lastDim);
            if (hasBias != 0U) {
                for (uint32_t col = 0U; col < lastDim; col += rowChunk) {
                    const uint32_t curLen = MinU32(rowChunk, lastDim - col);
                    Adds(yLocal[rowOffset + col], yLocal[rowOffset + col], biasValue, curLen);
                }
            }
        }
        PipeBarrier<PIPE_V>();

        DataCopyPad(yGm[xOffset], yLocal, copyRows);
        inQueueX.FreeTensor(xLocal);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void ExecuteChunkWithScalarValue(uint64_t xOffset,
                                                       T scaleValue,
                                                       T biasValue,
                                                       uint32_t calcLength)
    {
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyVec = {
            static_cast<uint16_t>(1),
            calcLength * static_cast<uint32_t>(sizeof(T)),
            0,
            0,
            0
        };

        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopyPad(xLocal, xGm[xOffset], copyVec, padParams);
        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.DeQue<T>();

        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        if (AlignUp(calcLength) == calcLength) {
            Muls(yLocal, xLocal, scaleValue, calcLength);
            if (hasBias != 0U) {
                Adds(yLocal, yLocal, biasValue, calcLength);
            }
        } else {
            const uint32_t chunkLen = MinU32(calcLength, alignCount * 16U);
            Muls(yLocal, xLocal, scaleValue, calcLength);
            if (hasBias != 0U) {
                for (uint32_t offset = 0U; offset < calcLength; offset += chunkLen) {
                    const uint32_t curLen = MinU32(chunkLen, calcLength - offset);
                    Adds(yLocal[offset], yLocal[offset], biasValue, curLen);
                }
            }
        }
        inQueueX.FreeTensor(xLocal);

        outQueueY.EnQue(yLocal);
        yLocal = outQueueY.DeQue<T>();
        DataCopyPad(yGm[xOffset], yLocal, copyVec);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void ExecuteContiguousRows(uint64_t xOffset, uint32_t scaleOffset, uint32_t rowBatch)
    {
        const uint32_t flatCount = rowBatch * lastDim;
        const uint32_t alignedCount = contiguousTileAlign;
        const bool rowAligned = (AlignUp(lastDim) == lastDim);
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        DataCopyExtParams copyFlat = {
            static_cast<uint16_t>(1),
            flatCount * static_cast<uint32_t>(sizeof(T)),
            0,
            0,
            0
        };
        DataCopyExtParams copyScale = {
            static_cast<uint16_t>(1),
            rowBatch * static_cast<uint32_t>(sizeof(T)),
            0,
            0,
            0
        };

        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        if (alignedCount == flatCount) {
            DataCopy(xLocal, xGm[xOffset], flatCount);
        } else {
            DataCopyPad(xLocal, xGm[xOffset], copyFlat, padParams);
        }
        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.DeQue<T>();

        LocalTensor<T> scaleExpanded = contiguousScaleBuf.Get<T>();
        DataCopyPad(scaleExpanded, scaleGm[scaleOffset], copyScale, padParams);

        LocalTensor<T> biasExpanded;
        if (hasBias != 0U) {
            biasExpanded = contiguousBiasBuf.Get<T>();
            DataCopyPad(biasExpanded, biasGm[scaleOffset], copyScale, padParams);
        }
        PipeBarrier<PIPE_ALL>();

        if (contiguousScalarPath) {
            for (uint32_t row = 0U; row < rowBatch; ++row) {
                const uint32_t rowOffset = row * lastDim;
                const T scaleValue = scaleExpanded.GetValue(row);
                Muls(xLocal[rowOffset], xLocal[rowOffset], scaleValue, lastDim);
                if (hasBias != 0U) {
                    const T biasValue = biasExpanded.GetValue(row);
                    Adds(xLocal[rowOffset], xLocal[rowOffset], biasValue, lastDim);
                }
            }
            PipeBarrier<PIPE_V>();
            DataCopy(yGm[xOffset], xLocal, flatCount);
            inQueueX.FreeTensor(xLocal);
            return;
        }

        if (rowAligned) {
            for (int row = static_cast<int>(rowBatch) - 1; row >= 0; --row) {
                const uint32_t rowIndex = static_cast<uint32_t>(row);
                const uint32_t rowOffset = rowIndex * lastDim;
                const T scaleValue = scaleExpanded.GetValue(rowIndex);
                Duplicate(scaleExpanded[rowOffset], scaleValue, lastDim);
                if (hasBias != 0U) {
                    const T biasValue = biasExpanded.GetValue(rowIndex);
                    Duplicate(biasExpanded[rowOffset], biasValue, lastDim);
                }
            }
            if (alignedCount > flatCount) {
                Duplicate(scaleExpanded[flatCount], static_cast<T>(0), alignedCount - flatCount);
                if (hasBias != 0U) {
                    Duplicate(biasExpanded[flatCount], static_cast<T>(0), alignedCount - flatCount);
                }
            }
            PipeBarrier<PIPE_V>();
        } else {
            for (int row = static_cast<int>(rowBatch) - 1; row >= 0; --row) {
                const uint32_t rowIndex = static_cast<uint32_t>(row);
                const uint32_t rowOffset = rowIndex * lastDim;
                const T scaleValue = scaleExpanded.GetValue(rowIndex);
                for (uint32_t col = 0U; col < lastDim; ++col) {
                    scaleExpanded.SetValue(rowOffset + col, scaleValue);
                }
                if (hasBias != 0U) {
                    const T biasValue = biasExpanded.GetValue(rowIndex);
                    for (uint32_t col = 0U; col < lastDim; ++col) {
                        biasExpanded.SetValue(rowOffset + col, biasValue);
                    }
                }
            }
            for (uint32_t pad = flatCount; pad < alignedCount; ++pad) {
                scaleExpanded.SetValue(pad, static_cast<T>(0));
                if (hasBias != 0U) {
                    biasExpanded.SetValue(pad, static_cast<T>(0));
                }
            }
            PipeBarrier<PIPE_ALL>();
        }

        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        Mul(yLocal, xLocal, scaleExpanded, alignedCount);
        if (hasBias != 0U) {
            Add(yLocal, yLocal, biasExpanded, alignedCount);
        }
        inQueueX.FreeTensor(xLocal);

        outQueueY.EnQue(yLocal);
        yLocal = outQueueY.DeQue<T>();
        if (alignedCount == flatCount) {
            DataCopy(yGm[xOffset], yLocal, flatCount);
        } else {
            DataCopyPad(yGm[xOffset], yLocal, copyFlat);
        }
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void ProcessSharedScale()
    {
        if (sharedOuterCount == 0U) {
            return;
        }
        LoadSharedScaleBias();

        const uint64_t baseRows = sharedOuterCount / static_cast<uint64_t>(blockNum);
        const uint64_t extraRows = sharedOuterCount % static_cast<uint64_t>(blockNum);
        const uint64_t startRow = static_cast<uint64_t>(blockIdx) * baseRows +
            ((static_cast<uint64_t>(blockIdx) < extraRows) ? static_cast<uint64_t>(blockIdx) : extraRows);
        const uint64_t localRows = baseRows + ((static_cast<uint64_t>(blockIdx) < extraRows) ? 1ULL : 0ULL);

        for (uint64_t r = 0U; r < localRows; r += static_cast<uint64_t>(sharedRowBatch)) {
            uint32_t rowsInTile = sharedRowBatch;
            if (r + static_cast<uint64_t>(rowsInTile) > localRows) {
                rowsInTile = static_cast<uint32_t>(localRows - r);
            }
            ExecuteSharedScaleTile((startRow + r) * static_cast<uint64_t>(sharedScaleLen),
                                   rowsInTile);
        }
    }

    __aicore__ inline void LoadSharedScaleBias()
    {
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copy = {
            static_cast<uint16_t>(1),
            sharedScaleLen * static_cast<uint32_t>(sizeof(T)),
            0,
            0,
            0
        };

        LocalTensor<T> scaleTmp = inQueueX.AllocTensor<T>();
        if (sharedScaleLen == sharedScaleLenAlign) {
            DataCopy(scaleTmp, scaleGm, sharedScaleLenAlign);
        } else {
            DataCopyPad(scaleTmp, scaleGm, copy, padParams);
        }
        inQueueX.EnQue(scaleTmp);
        scaleTmp = inQueueX.DeQue<T>();
        for (uint32_t pad = sharedScaleLen; pad < sharedScaleLenAlign; ++pad) {
            scaleTmp.SetValue(pad, static_cast<T>(0));
        }
        if constexpr (kBf16) {
            LocalTensor<float> scaleFp = sharedScaleBuf.Get<float>();
            Cast(scaleFp, scaleTmp, RoundMode::CAST_NONE, sharedScaleLenAlign);
        } else {
            LocalTensor<T> scaleNative = sharedScaleBuf.Get<T>();
            DataCopy(scaleNative, scaleTmp, sharedScaleLenAlign);
        }
        inQueueX.FreeTensor(scaleTmp);

        if (hasBias != 0U) {
            LocalTensor<T> biasTmp = inQueueX.AllocTensor<T>();
            if (sharedScaleLen == sharedScaleLenAlign) {
                DataCopy(biasTmp, biasGm, sharedScaleLenAlign);
            } else {
                DataCopyPad(biasTmp, biasGm, copy, padParams);
            }
            inQueueX.EnQue(biasTmp);
            biasTmp = inQueueX.DeQue<T>();
            for (uint32_t pad = sharedScaleLen; pad < sharedScaleLenAlign; ++pad) {
                biasTmp.SetValue(pad, static_cast<T>(0));
            }
            if constexpr (kBf16) {
                LocalTensor<float> biasFp = sharedBiasBuf.Get<float>();
                Cast(biasFp, biasTmp, RoundMode::CAST_NONE, sharedScaleLenAlign);
            } else {
                LocalTensor<T> biasNative = sharedBiasBuf.Get<T>();
                DataCopy(biasNative, biasTmp, sharedScaleLenAlign);
            }
            inQueueX.FreeTensor(biasTmp);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ExecuteSharedScaleTile(uint64_t xOffset, uint32_t rowsInTile)
    {
        const uint32_t scaleLen = sharedScaleLen;
        const uint32_t scaleLenAlign = sharedScaleLenAlign;
        const bool aligned = (scaleLen == scaleLenAlign);
        const uint32_t tileElems = rowsInTile * scaleLenAlign;

        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        // Per-row strided copy when scaleLen is not aligned: blockCount=rowsInTile,
        // blockLen=scaleLen*sizeof(T), src/dst stride 0 (rows already contiguous in GM,
        // and target lays them out contiguously per-row at scaleLenAlign stride).
        DataCopyExtParams copyIn = {
            static_cast<uint16_t>(rowsInTile),
            scaleLen * static_cast<uint32_t>(sizeof(T)),
            0,
            0,
            0
        };
        DataCopyExtParams copyOut = {
            static_cast<uint16_t>(rowsInTile),
            scaleLen * static_cast<uint32_t>(sizeof(T)),
            0,
            0,
            0
        };

        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        if (aligned) {
            DataCopy(xLocal, xGm[xOffset], tileElems);
        } else {
            DataCopyPad(xLocal, xGm[xOffset], copyIn, padParams);
        }
        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.DeQue<T>();

        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

        if constexpr (kBf16) {
            LocalTensor<float> xCalc = sharedXCalcBuf.Get<float>();
            LocalTensor<float> yCalc = sharedYCalcBuf.Get<float>();
            LocalTensor<float> scaleFp = sharedScaleBuf.Get<float>();
            Cast(xCalc, xLocal, RoundMode::CAST_NONE, tileElems);
            for (uint32_t i = 0U; i < rowsInTile; ++i) {
                const uint32_t off = i * scaleLenAlign;
                Mul(yCalc[off], xCalc[off], scaleFp, scaleLenAlign);
            }
            if (hasBias != 0U) {
                LocalTensor<float> biasFp = sharedBiasBuf.Get<float>();
                Cast(yLocal, yCalc, RoundMode::CAST_RINT, tileElems);
                Cast(yCalc, yLocal, RoundMode::CAST_NONE, tileElems);
                for (uint32_t i = 0U; i < rowsInTile; ++i) {
                    const uint32_t off = i * scaleLenAlign;
                    Add(yCalc[off], yCalc[off], biasFp, scaleLenAlign);
                }
            }
            Cast(yLocal, yCalc, RoundMode::CAST_RINT, tileElems);
        } else {
            LocalTensor<T> scaleNative = sharedScaleBuf.Get<T>();
            for (uint32_t i = 0U; i < rowsInTile; ++i) {
                const uint32_t off = i * scaleLenAlign;
                Mul(yLocal[off], xLocal[off], scaleNative, scaleLenAlign);
            }
            if (hasBias != 0U) {
                LocalTensor<T> biasNative = sharedBiasBuf.Get<T>();
                for (uint32_t i = 0U; i < rowsInTile; ++i) {
                    const uint32_t off = i * scaleLenAlign;
                    Add(yLocal[off], yLocal[off], biasNative, scaleLenAlign);
                }
            }
        }

        inQueueX.FreeTensor(xLocal);
        outQueueY.EnQue(yLocal);
        yLocal = outQueueY.DeQue<T>();
        if (aligned) {
            DataCopy(yGm[xOffset], yLocal, tileElems);
        } else {
            DataCopyPad(yGm[xOffset], yLocal, copyOut);
        }
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void ResolveRowOffsets(uint64_t rowIdx, uint64_t& xOffset, uint64_t& scaleOffset) const
    {
        int indices[DIM_MAX_NUM] = {};
        for (int i = dimNum - 2; i >= 0; --i) {
            indices[i] = static_cast<int>(rowIdx % static_cast<uint64_t>(dimBC[i]));
            rowIdx /= static_cast<uint64_t>(dimBC[i]);
        }

        xOffset = 0U;
        scaleOffset = 0U;
        for (int i = 0; i < dimNum - 1; ++i) {
            xOffset += static_cast<uint64_t>((dimX[i] == 1) ? 0 : indices[i]) * strideX[i];
            scaleOffset += static_cast<uint64_t>((dimScale[i] == 1) ? 0 : indices[i]) * strideScale[i];
        }
    }

    __aicore__ inline void ProcessTask(uint64_t taskIdx)
    {
        const uint64_t rowIdx = taskIdx / chunkCount;
        const uint32_t chunkIdx = static_cast<uint32_t>(taskIdx % chunkCount);
        const uint32_t elemOffset = chunkIdx * tileLength;
        const uint32_t calcLength = MinU32(tileLength, lastDim - elemOffset);

        uint64_t xBase = 0U;
        uint64_t scaleBase = 0U;
        ResolveRowOffsets(rowIdx, xBase, scaleBase);

        ExecuteChunk(
            xBase + static_cast<uint64_t>(elemOffset),
            scaleBase + static_cast<uint64_t>(scaleLastBroadcast ? 0U : elemOffset),
            calcLength);
    }

    __aicore__ inline void ExecuteChunk(uint64_t xOffset, uint64_t scaleOffset, uint32_t calcLength)
    {
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyVec = {
            static_cast<uint16_t>(1),
            calcLength * static_cast<uint32_t>(sizeof(T)),
            0,
            0,
            0
        };
        DataCopyExtParams copyScalar = {
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(sizeof(T)),
            0,
            0,
            0
        };

        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopyPad(xLocal, xGm[xOffset], copyVec, padParams);
        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.DeQue<T>();

        if constexpr (!kBf16) {
            if (scalarCachePath && scaleLastBroadcast) {
                const T scaleValue = GetCachedScaleValue(static_cast<uint32_t>(scaleOffset));
                const T biasValue = (hasBias != 0U) ? GetCachedBiasValue(static_cast<uint32_t>(scaleOffset)) : static_cast<T>(0);
                LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
                Muls(yLocal, xLocal, scaleValue, calcLength);
                if (hasBias != 0U) {
                    Adds(yLocal, yLocal, biasValue, calcLength);
                }
                inQueueX.FreeTensor(xLocal);
                outQueueY.EnQue(yLocal);
                yLocal = outQueueY.DeQue<T>();
                DataCopyPad(yGm[xOffset], yLocal, copyVec);
                outQueueY.FreeTensor(yLocal);
                return;
            }
        }

        LocalTensor<T> scaleLocal = inQueueScale.AllocTensor<T>();
        if (scaleLastBroadcast) {
            DataCopyPad(scaleLocal, scaleGm[scaleOffset], copyScalar, padParams);
        } else {
            DataCopyPad(scaleLocal, scaleGm[scaleOffset], copyVec, padParams);
        }
        inQueueScale.EnQue(scaleLocal);
        scaleLocal = inQueueScale.DeQue<T>();

        LocalTensor<T> biasLocal;
        if (hasBias != 0U) {
            biasLocal = inQueueBias.AllocTensor<T>();
            if (scaleLastBroadcast) {
                DataCopyPad(biasLocal, biasGm[scaleOffset], copyScalar, padParams);
            } else {
                DataCopyPad(biasLocal, biasGm[scaleOffset], copyVec, padParams);
            }
            inQueueBias.EnQue(biasLocal);
            biasLocal = inQueueBias.DeQue<T>();
        }

        if constexpr (kBf16) {
            ComputeBf16(xLocal, scaleLocal, biasLocal, xOffset, copyVec, calcLength);
        } else {
            ComputeNative(xLocal, scaleLocal, biasLocal, xOffset, copyVec, calcLength);
        }
    }

    __aicore__ inline void ComputeNative(LocalTensor<T>& xLocal,
                                         LocalTensor<T>& scaleLocal,
                                         LocalTensor<T>& biasLocal,
                                         uint64_t xOffset,
                                         const DataCopyExtParams& copyVec,
                                         uint32_t calcLength)
    {
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        if (scaleLastBroadcast) {
            T scaleValue = scaleLocal.GetValue(0);
            Muls(yLocal, xLocal, scaleValue, calcLength);
        } else {
            Mul(yLocal, xLocal, scaleLocal, calcLength);
        }
        inQueueScale.FreeTensor(scaleLocal);

        if (hasBias != 0U) {
            if (scaleLastBroadcast) {
                T biasValue = biasLocal.GetValue(0);
                Adds(yLocal, yLocal, biasValue, calcLength);
            } else {
                Add(yLocal, yLocal, biasLocal, calcLength);
            }
            inQueueBias.FreeTensor(biasLocal);
        }
        inQueueX.FreeTensor(xLocal);

        outQueueY.EnQue(yLocal);
        yLocal = outQueueY.DeQue<T>();
        DataCopyPad(yGm[xOffset], yLocal, copyVec);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void ComputeBf16(LocalTensor<T>& xLocal,
                                       LocalTensor<T>& scaleLocal,
                                       LocalTensor<T>& biasLocal,
                                       uint64_t xOffset,
                                       const DataCopyExtParams& copyVec,
                                       uint32_t calcLength)
    {
        LocalTensor<float> xCalc = calcQueueX.AllocTensor<float>();
        Cast(xCalc, xLocal, RoundMode::CAST_NONE, calcLength);
        inQueueX.FreeTensor(xLocal);

        LocalTensor<float> yCalc = calcQueueY.AllocTensor<float>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

        if (scaleLastBroadcast) {
            float scaleValue = ToFloat(scaleLocal.GetValue(0));
            inQueueScale.FreeTensor(scaleLocal);
            Muls(yCalc, xCalc, scaleValue, calcLength);
            if (hasBias != 0U) {
                float biasValue = ToFloat(biasLocal.GetValue(0));
                inQueueBias.FreeTensor(biasLocal);
                Cast(yLocal, yCalc, RoundMode::CAST_RINT, calcLength);
                Cast(yCalc, yLocal, RoundMode::CAST_NONE, calcLength);
                Adds(yCalc, yCalc, biasValue, calcLength);
            }
        } else {
            LocalTensor<float> scaleCalc = calcQueueScale.AllocTensor<float>();
            Cast(scaleCalc, scaleLocal, RoundMode::CAST_NONE, calcLength);
            inQueueScale.FreeTensor(scaleLocal);
            Mul(yCalc, xCalc, scaleCalc, calcLength);
            calcQueueScale.FreeTensor(scaleCalc);

            if (hasBias != 0U) {
                LocalTensor<float> biasCalc = calcQueueBias.AllocTensor<float>();
                Cast(biasCalc, biasLocal, RoundMode::CAST_NONE, calcLength);
                inQueueBias.FreeTensor(biasLocal);
                Cast(yLocal, yCalc, RoundMode::CAST_RINT, calcLength);
                Cast(yCalc, yLocal, RoundMode::CAST_NONE, calcLength);
                Add(yCalc, yCalc, biasCalc, calcLength);
                calcQueueBias.FreeTensor(biasCalc);
            }
        }

        Cast(yLocal, yCalc, RoundMode::CAST_RINT, calcLength);
        calcQueueY.FreeTensor(yCalc);
        calcQueueX.FreeTensor(xCalc);

        outQueueY.EnQue(yLocal);
        yLocal = outQueueY.DeQue<T>();
        DataCopyPad(yGm[xOffset], yLocal, copyVec);
        outQueueY.FreeTensor(yLocal);
    }

    TPipe* pipe = nullptr;
    TQue<QuePosition::VECIN, TQUE_NUM> inQueueX;
    TQue<QuePosition::VECIN, TQUE_NUM> inQueueScale;
    TQue<QuePosition::VECIN, TQUE_NUM> inQueueBias;
    TQue<QuePosition::VECOUT, TQUE_NUM> outQueueY;
    TQue<QuePosition::VECCALC, TQUE_NUM> calcQueueX;
    TQue<QuePosition::VECCALC, TQUE_NUM> calcQueueScale;
    TQue<QuePosition::VECCALC, TQUE_NUM> calcQueueBias;
    TQue<QuePosition::VECCALC, TQUE_NUM> calcQueueY;
    TBuf<QuePosition::VECCALC> contiguousScaleBuf;
    TBuf<QuePosition::VECCALC> contiguousBiasBuf;
    TBuf<QuePosition::VECCALC> sharedScaleBuf;
    TBuf<QuePosition::VECCALC> sharedBiasBuf;
    TBuf<QuePosition::VECCALC> sharedXCalcBuf;
    TBuf<QuePosition::VECCALC> sharedYCalcBuf;
    TBuf<QuePosition::VECCALC> scalarScaleBuf;
    TBuf<QuePosition::VECCALC> scalarBiasBuf;
    GlobalTensor<T> xGm;
    GlobalTensor<T> scaleGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<T> yGm;

    int dimNum = 0;
    int dimX[DIM_MAX_NUM] = {};
    int dimScale[DIM_MAX_NUM] = {};
    int dimBC[DIM_MAX_NUM] = {};
    uint64_t strideX[DIM_MAX_NUM] = {};
    uint64_t strideScale[DIM_MAX_NUM] = {};
    uint64_t totalLengthX = 0U;
    uint64_t totalLengthScale = 0U;
    uint64_t totalLengthBC = 0U;
    uint64_t rowCount = 0U;
    uint64_t chunkCount = 0U;
    uint64_t totalTaskCount = 0U;
    uint32_t lastDim = 0U;
    uint32_t tileLength = 0U;
    uint32_t tileLengthAlign = 0U;
    uint32_t hasBias = 0U;
    uint32_t blockNum = 1U;
    uint32_t blockIdx = 0U;
    uint32_t alignCount = 1U;
    uint32_t ubBytes = UB_FALLBACK_BYTES;
    uint32_t contiguousScaleRows = 0U;
    uint32_t contiguousRowBatch = 1U;
    uint32_t contiguousTileAlign = 0U;
    uint32_t contiguousRowStride = 0U;
    uint64_t sharedOuterCount = 0U;
    uint32_t sharedScaleLen = 0U;
    uint32_t sharedScaleLenAlign = 0U;
    uint32_t sharedRowBatch = 1U;
    uint32_t scalarCacheBase = 0U;
    uint32_t scalarCacheCount = 0U;
    uint32_t scalarCacheTile = 0U;
    uint32_t packedScaleRows = 0U;
    uint32_t packedRowBatch = 1U;
    uint32_t packedRowStrideAlign = 0U;
    uint32_t sequentialScaleRows = 0U;
    bool scaleLastBroadcast = false;
    bool contiguousRowPath = false;
    bool contiguousScalarPath = false;
    bool packedScalarRowsPath = false;
    bool packedFlatVectorPath = false;
    bool packedOuterPipelinePath = false;
    bool sharedScalePath = false;
    bool scalarCachePath = false;
    bool sequentialScaleRowsPath = false;
};

}  // namespace ScaleFallbackV36
