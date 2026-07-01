// Kernel implementation. Optimized MTE2 with strided DataCopy.
#include "kernel_operator.h"

#include "batch_to_space_tiling.h"
#include "tiling_key_batch_to_space.h"

template <class DT_X>
class KernelBatchToSpace {
public:
    __aicore__ inline KernelBatchToSpace() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling) {
        xGm.SetGlobalBuffer((__gm__ DT_X *)x, tiling.inputLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, tiling.outputLength);
        outputLength = tiling.outputLength;
        height = tiling.height;
        width = tiling.width;
        depth = tiling.depth;
        outBatch = tiling.outBatch;
        outHeight = tiling.outHeight;
        outWidth = tiling.outWidth;
        blockSize = tiling.blockSize;
        cropTop = tiling.cropTop;
        cropBottom = tiling.cropBottom;
        cropLeft = tiling.cropLeft;
        cropRight = tiling.cropRight;
        blockDim = tiling.blockDim;
        depthAligned = (depth % elemsPer32B) == 0;
        pipe.InitBuffer(inQueue, 2, tileElems * sizeof(DT_X));
        pipe.InitBuffer(outQueue, 2, tileElems * sizeof(DT_X));
        pipe.InitBuffer(tmpBuf, tc5GatherWorkOffsets * sizeof(int32_t));
        allocElems = tileElems;
    }

__aicore__ inline void Process() {
        if (blockSize == 4) {
            ProcessBs4PixelChunks();
            return;
        }
        if (blockSize == 2) {

if (outWidth * depth * 2 <= tileElems) {
                if (!depthAligned && width * depth * sizeof(DT_X) <= 65535) {
                    if (outputLength <= 131072) {
                        ProcessBs2RowScalar();
                    } else {
                        ProcessBs2PadRowChunks();
                    }
                    return;
                }
                if (depthAligned && width * (depth / elemsPer32B) <= 65535) {
                    ProcessBs2MultiRow();
                    return;
                }
            }
 if (!depthAligned && depth * sizeof(DT_X) <= 65535 && depth <= tileElems
                && outWidth * depth * 2 <= tileElems) {
                if (outputLength <= 131072) {
                    ProcessBs2RowScalar();
                } else {
                    ProcessBs2PadRowChunks();
                }
                return;
            }
            if (depth * sizeof(DT_X) <= 65535 && depth <= tileElems) {
                ProcessBs2RowScalar();
                return;
            }
        }
        ProcessPixelChunks();
    }



    template <uint32_t D>
    __aicore__ inline void ProcessBs2StaticDepth() {
        constexpr uint32_t blockLen32B = D / elemsPer32B;
        uint32_t core = AscendC::GetBlockIdx();
        uint32_t outputPixels = outputLength / D;
        uint32_t chunk = (outputPixels + blockDim - 1) / blockDim;
        uint32_t start = core * chunk;
        uint32_t end = start + chunk;
        if (end > outputPixels) {
            end = outputPixels;
        }

        uint32_t pixel = start;
        while (pixel < end) {
            uint32_t tmp = pixel;
            uint32_t ow = tmp % outWidth;
            tmp = tmp / outWidth;
            uint32_t oh = tmp % outHeight;
            uint32_t ob = tmp / outHeight;

            uint32_t paddedH = oh + cropTop;
            uint32_t bh = paddedH & 1;
            uint32_t ih = paddedH >> 1;
            uint32_t rowRun = outWidth - ow;
            if (rowRun > end - pixel) {
                rowRun = end - pixel;
            }

            for (uint32_t r = 0; r < rowRun; ++r) {
                uint32_t paddedW = ow + r + cropLeft;
                if (((paddedW & 1) == 0) && r + 63 < rowRun) {
                    uint32_t iwPair = paddedW >> 1;
                    uint32_t inBatch0 = (bh << 1) * outBatch + ob;
                    uint32_t inBatch1 = ((bh << 1) + 1) * outBatch + ob;
                    uint32_t inBase0 = (((inBatch0 * height + ih) * width + iwPair) * D);
                    uint32_t inBase1 = (((inBatch1 * height + ih) * width + iwPair) * D);
                    uint32_t outBase = (pixel + r) * D;

                    AscendC::LocalTensor<DT_X> inAll = inQueue.AllocTensor<DT_X>();
                    AscendC::DataCopy(inAll[0], xGm[inBase0], D * 32);
                    AscendC::DataCopy(inAll[D * 32], xGm[inBase1], D * 32);
                    inQueue.EnQue(inAll);
                    AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();
                    
                    AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
                    AscendC::DataCopyParams zipParams;
                    zipParams.blockCount = 32;
                    zipParams.blockLen = blockLen32B;
                    zipParams.srcStride = 0;
                    zipParams.dstStride = blockLen32B;
                    AscendC::DataCopy(outLocal, inDe[0], zipParams);
                    AscendC::DataCopy(outLocal[D], inDe[D * 32], zipParams);
                    inQueue.FreeTensor(inDe);

                    outQueue.EnQue(outLocal);
                    AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                    AscendC::DataCopy(yGm[outBase], finalLocal, D * 64);
                    outQueue.FreeTensor(finalLocal);
                    r += 63;
                    continue;
                }
                if (((paddedW & 1) == 0) && r + 31 < rowRun) {
                    uint32_t iwPair = paddedW >> 1;
                    uint32_t inBatch0 = (bh << 1) * outBatch + ob;
                    uint32_t inBatch1 = ((bh << 1) + 1) * outBatch + ob;
                    uint32_t inBase0 = (((inBatch0 * height + ih) * width + iwPair) * D);
                    uint32_t inBase1 = (((inBatch1 * height + ih) * width + iwPair) * D);
                    uint32_t outBase = (pixel + r) * D;

                    AscendC::LocalTensor<DT_X> inAll = inQueue.AllocTensor<DT_X>();
                    AscendC::DataCopy(inAll[0], xGm[inBase0], D * 16);
                    AscendC::DataCopy(inAll[D * 16], xGm[inBase1], D * 16);
                    inQueue.EnQue(inAll);
                    AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();
                    
                    AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
                    AscendC::DataCopyParams zipParams;
                    zipParams.blockCount = 16;
                    zipParams.blockLen = blockLen32B;
                    zipParams.srcStride = 0;
                    zipParams.dstStride = blockLen32B;
                    AscendC::DataCopy(outLocal, inDe[0], zipParams);
                    AscendC::DataCopy(outLocal[D], inDe[D * 16], zipParams);
                    inQueue.FreeTensor(inDe);

                    outQueue.EnQue(outLocal);
                    AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                    AscendC::DataCopy(yGm[outBase], finalLocal, D * 32);
                    outQueue.FreeTensor(finalLocal);
                    r += 31;
                    continue;
                }
                if (((paddedW & 1) == 0) && r + 15 < rowRun) {
                    uint32_t iwPair = paddedW >> 1;
                    uint32_t inBatch0 = (bh << 1) * outBatch + ob;
                    uint32_t inBatch1 = ((bh << 1) + 1) * outBatch + ob;
                    uint32_t inBase0 = (((inBatch0 * height + ih) * width + iwPair) * D);
                    uint32_t inBase1 = (((inBatch1 * height + ih) * width + iwPair) * D);
                    uint32_t outBase = (pixel + r) * D;

                    AscendC::LocalTensor<DT_X> inAll = inQueue.AllocTensor<DT_X>();
                    AscendC::DataCopy(inAll[0], xGm[inBase0], D * 8);
                    AscendC::DataCopy(inAll[D * 8], xGm[inBase1], D * 8);
                    inQueue.EnQue(inAll);
                    AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();
                    
                    AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
                    AscendC::DataCopyParams zipParams;
                    zipParams.blockCount = 8;
                    zipParams.blockLen = blockLen32B;
                    zipParams.srcStride = 0;
                    zipParams.dstStride = blockLen32B;
                    AscendC::DataCopy(outLocal, inDe[0], zipParams);
                    AscendC::DataCopy(outLocal[D], inDe[D * 8], zipParams);
                    inQueue.FreeTensor(inDe);

                    outQueue.EnQue(outLocal);
                    AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                    AscendC::DataCopy(yGm[outBase], finalLocal, D * 16);
                    outQueue.FreeTensor(finalLocal);
                    r += 15;
                    continue;
                }
                if (((paddedW & 1) == 0) && r + 7 < rowRun) {
                    uint32_t iwPair = paddedW >> 1;
                    uint32_t inBatch0 = (bh << 1) * outBatch + ob;
                    uint32_t inBatch1 = ((bh << 1) + 1) * outBatch + ob;
                    uint32_t inBase0 = (((inBatch0 * height + ih) * width + iwPair) * D);
                    uint32_t inBase1 = (((inBatch1 * height + ih) * width + iwPair) * D);
                    uint32_t outBase = (pixel + r) * D;

                    AscendC::LocalTensor<DT_X> inAll = inQueue.AllocTensor<DT_X>();
                    AscendC::DataCopy(inAll[0], xGm[inBase0], D * 4);
                    AscendC::DataCopy(inAll[D * 4], xGm[inBase1], D * 4);
                    inQueue.EnQue(inAll);
                    AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();
                    
                    AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
                    AscendC::DataCopyParams zipParams;
                    zipParams.blockCount = 4;
                    zipParams.blockLen = blockLen32B;
                    zipParams.srcStride = 0;
                    zipParams.dstStride = blockLen32B;
                    AscendC::DataCopy(outLocal, inDe[0], zipParams);
                    AscendC::DataCopy(outLocal[D], inDe[D * 4], zipParams);
                    inQueue.FreeTensor(inDe);

                    outQueue.EnQue(outLocal);
                    AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                    AscendC::DataCopy(yGm[outBase], finalLocal, D * 8);
                    outQueue.FreeTensor(finalLocal);
                    r += 7;
                    continue;
                }

                uint32_t bw = paddedW & 1;
                uint32_t iw = paddedW >> 1;
                uint32_t inBatch = ((bh << 1) + bw) * outBatch + ob;
                uint32_t inBase = (((inBatch * height + ih) * width + iw) * D);
                uint32_t outBase = (pixel + r) * D;
                AscendC::LocalTensor<DT_X> local = inQueue.AllocTensor<DT_X>();
                AscendC::DataCopy(local, xGm[inBase], D);
                inQueue.EnQue(local);
                AscendC::LocalTensor<DT_X> inLocal = inQueue.DeQue<DT_X>();
                AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
                AscendC::DataCopy(outLocal, inLocal, D);
                outQueue.EnQue(outLocal);
                inQueue.FreeTensor(inLocal);
                AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                AscendC::DataCopy(yGm[outBase], finalLocal, D);
                outQueue.FreeTensor(finalLocal);
            }
            pixel += rowRun;
        }
    }

    __aicore__ inline void ProcessBs2MultiRow() {
        uint32_t blockLen32B = depth / elemsPer32B;
        uint32_t core = AscendC::GetBlockIdx();
        uint32_t rowPixels = outWidth;
        uint32_t rowElems = rowPixels * depth;
        uint32_t totalRows = (outputLength / depth) / rowPixels;
        uint32_t rowChunk = (totalRows + blockDim - 1) / blockDim;
        uint32_t rowStart = core * rowChunk;
        uint32_t rowEnd = rowStart + rowChunk;
        if (rowEnd > totalRows) rowEnd = totalRows;
        if (rowStart >= rowEnd) return;

        uint32_t maxBatchRows = tileElems / rowElems;
        if (maxBatchRows > 64) maxBatchRows = 64;
        if (maxBatchRows < 2) maxBatchRows = 2;
        maxBatchRows &= ~1u; // ensure even

        bool cropLeftEven = ((cropLeft & 1) == 0);
        uint32_t bw0_iwStart = cropLeftEven ? (cropLeft >> 1) : ((cropLeft + 1) >> 1);
        uint32_t bw1_iwStart = cropLeftEven ? (cropLeft >> 1) : (cropLeft >> 1);
        uint32_t w0 = cropLeftEven ? ((outWidth + 1) >> 1) : (outWidth >> 1);
        uint32_t w1 = cropLeftEven ? (outWidth >> 1) : ((outWidth + 1) >> 1);
        uint32_t inputRowElems = width * depth;

        uint32_t row = rowStart;
        while (row < rowEnd) {
            uint32_t batchRows = rowEnd - row;
            if (batchRows > maxBatchRows) batchRows = maxBatchRows;

            uint32_t ohStart = row % outHeight;
            uint32_t ob = row / outHeight;
            uint32_t rowsInThisOb = outHeight - ohStart;
            if (batchRows > rowsInThisOb) batchRows = rowsInThisOb;
            if (batchRows < 2) batchRows = 1; // handle tail

            uint32_t outBase = (ob * outHeight + ohStart) * rowElems;
            uint32_t paddedH0 = ohStart + cropTop;
            bool firstIsBh0 = ((paddedH0 & 1) == 0);
            uint32_t nBh0 = firstIsBh0 ? ((batchRows + 1) >> 1) : (batchRows >> 1);
            uint32_t nBh1 = batchRows - nBh0;
            uint32_t ih0Start = firstIsBh0 ? (paddedH0 >> 1) : ((paddedH0 + 1) >> 1);
            uint32_t ih1Start = firstIsBh0 ? ((paddedH0 + 1) >> 1) : (paddedH0 >> 1);

            AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();

            for (uint32_t bh = 0; bh < 2; ++bh) {
                uint32_t nRows = (bh == 0) ? nBh0 : nBh1;
                if (nRows == 0) continue;
                uint32_t ihStart = (bh == 0) ? ih0Start : ih1Start;

                AscendC::LocalTensor<DT_X> inAll = inQueue.AllocTensor<DT_X>();
                for (uint32_t bwIdx = 0; bwIdx < 2; ++bwIdx) {
                    uint32_t curW = (bwIdx == 0) ? w0 : w1;
                    if (curW == 0) continue;
                    uint32_t iwStart = (bwIdx == 0) ? bw0_iwStart : bw1_iwStart;
                    uint32_t inBatch = (bh * 2 + bwIdx) * outBatch + ob;
                    uint32_t gmBase = ((inBatch * height + ihStart) * width + iwStart) * depth;

                    uint32_t readLen32B = curW * blockLen32B;
                    uint32_t readGap32B = (inputRowElems - curW * depth) / elemsPer32B;
                    uint32_t offsetInAll = (bwIdx == 0) ? 0 : (nRows * w0 * depth);

                    if (nRows == 1) {
                        AscendC::DataCopy(inAll[offsetInAll], xGm[gmBase], curW * depth);
                    } else {
                        AscendC::DataCopyParams readP;
                        readP.blockCount = static_cast<uint16_t>(nRows);
                        readP.blockLen = static_cast<uint16_t>(readLen32B);
                        readP.srcStride = static_cast<uint16_t>(readGap32B);
                        readP.dstStride = 0;
                        AscendC::DataCopy(inAll[offsetInAll], xGm[gmBase], readP);
                    }
                }
                inQueue.EnQue(inAll);
                AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();

                for (uint32_t bwIdx = 0; bwIdx < 2; ++bwIdx) {
                    uint32_t curW = (bwIdx == 0) ? w0 : w1;
                    if (curW == 0) continue;
                    uint32_t offsetInAll = (bwIdx == 0) ? 0 : (nRows * w0 * depth);
                    
                    AscendC::DataCopyParams zipP;
                    zipP.blockCount = static_cast<uint16_t>(curW);
                    zipP.blockLen = blockLen32B;
                    zipP.srcStride = 0;
                    zipP.dstStride = blockLen32B;

                    for (uint32_t r = 0; r < nRows; ++r) {
                        uint32_t outRowIdx;
                        if (bh == 0) {
                            outRowIdx = firstIsBh0 ? (2 * r) : (2 * r + 1);
                        } else {
                            outRowIdx = firstIsBh0 ? (2 * r + 1) : (2 * r);
                        }
                        uint32_t rowOff = outRowIdx * rowElems;
                        uint32_t srcOff = offsetInAll + r * curW * depth;
                        bool isEvenBw = (bwIdx == 0);
                        uint32_t dstOff = (cropLeftEven == isEvenBw) ? rowOff : (rowOff + depth);
                        AscendC::DataCopy(outLocal[dstOff], inDe[srcOff], zipP);
                    }
                }
                inQueue.FreeTensor(inDe);
            }

            outQueue.EnQue(outLocal);
            AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
            AscendC::DataCopy(yGm[outBase], finalLocal, batchRows * rowElems);
            outQueue.FreeTensor(finalLocal);

            row += batchRows;
        }
    }

    __aicore__ inline void ProcessTC5GatherPath() {
        constexpr uint32_t tc5DepthBytes = 130;
        constexpr uint32_t tc5RowWidth = 128;
        constexpr uint32_t tc5RowElems = tc5RowWidth * tc5Depth;
        constexpr uint32_t tc5RowBytes = tc5RowElems * sizeof(DT_X);

        uint32_t core = AscendC::GetBlockIdx();
        uint32_t outputRows = outputLength / (outWidth * tc5Depth);
        uint32_t rowsPerCore = (outputRows + blockDim - 1) / blockDim;
        uint32_t startRow = core * rowsPerCore;
        uint32_t endRow = startRow + rowsPerCore;
        if (endRow > outputRows) endRow = outputRows;
        if (startRow >= endRow) return;

        AscendC::LocalTensor<int32_t> offsetBase = tmpBuf.Get<int32_t>();
        constexpr uint32_t tc5OffsetSeedPixels = 8;
        for (uint32_t k = 0; k < tc5OffsetSeedPixels; k++) {
            uint32_t pixelIn = k >> 1;
            uint32_t pixelBaseBytes = ((k & 1) == 0)
                ? (pixelIn * tc5DepthBytes)
                : (tc5RowBytes + (pixelIn + 1) * tc5DepthBytes);
            for (uint32_t j = 0; j < tc5Depth; j++) {
                offsetBase.SetValue(k * tc5Depth + j, static_cast<int32_t>(pixelBaseBytes + j * sizeof(DT_X)));
            }
        }
        for (uint32_t group = 1; group * tc5OffsetSeedPixels < tc5MaxPixels; group++) {
            uint32_t dstPixel = group * tc5OffsetSeedPixels;
            uint32_t remainPixels = tc5MaxPixels - dstPixel;
            uint32_t copyPixels = remainPixels > tc5OffsetSeedPixels ? tc5OffsetSeedPixels : remainPixels;
            AscendC::Adds(offsetBase[dstPixel * tc5Depth], offsetBase,
                          static_cast<int32_t>(group * (tc5OffsetSeedPixels >> 1) * tc5DepthBytes),
                          copyPixels * tc5Depth);
        }

        AscendC::DataCopyPadParams padP{true, 0, 0, 0};
        for (uint32_t row = startRow; row < endRow; row++) {
            uint32_t oh = row % outHeight;
            uint32_t ob = row / outHeight;
            uint32_t paddedH = oh + cropTop;
            uint32_t bh = paddedH & 1;
            uint32_t ih = paddedH >> 1;
            uint32_t inBatch0 = (bh << 1) * outBatch + ob;
            uint32_t inBatch1 = ((bh << 1) + 1) * outBatch + ob;

            uint32_t inBase1Row = ((inBatch1 * height + ih) * width) * tc5Depth;
            uint32_t inBase0Row = ((inBatch0 * height + ih) * width) * tc5Depth;
            uint32_t rowOutBase = row * outWidth * tc5Depth;

            AscendC::LocalTensor<DT_X> inAlloc = inQueue.AllocTensor<DT_X>();
            AscendC::DataCopy(inAlloc, xGm[inBase1Row], tc5RowElems);
            AscendC::DataCopy(inAlloc[tc5RowElems], xGm[inBase0Row], tc5RowElems);
            inQueue.EnQue(inAlloc);

            uint32_t r = 0;
            uint32_t currentBaseBytes = 0;
            AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();

            while (r < outWidth) {
                uint32_t batchPixels = outWidth - r;
                if (batchPixels > tc5MaxPixels) batchPixels = tc5MaxPixels;
                uint32_t totalGatherCount = batchPixels * tc5Depth;
                uint32_t targetBaseBytes = (r >> 1) * tc5DepthBytes;
                if (targetBaseBytes != currentBaseBytes) {
                    AscendC::Adds(offsetBase, offsetBase,
                                  static_cast<int32_t>(targetBaseBytes) - static_cast<int32_t>(currentBaseBytes),
                                  tc5GatherMaxOffsets);
                    currentBaseBytes = targetBaseBytes;
                }
                AscendC::LocalTensor<DT_X> outAlloc = outQueue.AllocTensor<DT_X>();
                AscendC::Gather(outAlloc, inDe, offsetBase.template ReinterpretCast<uint32_t>(), 0, totalGatherCount);
                outQueue.EnQue(outAlloc);

                AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                uint32_t writeBytes = batchPixels * tc5DepthBytes;
                AscendC::DataCopyParams copyPOut{1, static_cast<uint16_t>(writeBytes), 0, 0};
                AscendC::DataCopyPad(yGm[rowOutBase + r * tc5Depth], finalLocal, copyPOut);
                outQueue.FreeTensor(finalLocal);

                r += batchPixels;
            }
            inQueue.FreeTensor(inDe);
            if (currentBaseBytes != 0) {
                AscendC::Adds(offsetBase, offsetBase, -static_cast<int32_t>(currentBaseBytes), tc5GatherMaxOffsets);
            }
        }
    }

    __aicore__ inline void ProcessTC5DirectScatter() {
        constexpr uint32_t tc5Depth = 65;
        constexpr uint32_t tc5DepthBytes = 130;
        constexpr uint32_t tc5PaddedDepthElems = 80;
        constexpr uint32_t tc5MaxPixels = 36;

        uint32_t core = AscendC::GetBlockIdx();
        uint32_t outputPixels = outputLength / tc5Depth;
        uint32_t pixelsPerCore = (outputPixels + blockDim - 1) / blockDim;
        uint32_t startPixel = core * pixelsPerCore;
        uint32_t endPixel = startPixel + pixelsPerCore;
        if (endPixel > outputPixels) endPixel = outputPixels;
        if (startPixel >= endPixel) return;

        AscendC::DataCopyPadParams padP{true, 0, 0, 0};
        uint32_t pixel = startPixel;
        while (pixel < endPixel) {
            uint32_t tmp = pixel;
            uint32_t ow = tmp % outWidth;
            tmp = tmp / outWidth;
            uint32_t oh = tmp % outHeight;
            uint32_t ob = tmp / outHeight;

            uint32_t rowRun = outWidth - ow;
            if (rowRun > endPixel - pixel) rowRun = endPixel - pixel;

            uint32_t paddedH = oh + cropTop;
            uint32_t bh = paddedH & 1;
            uint32_t ih = paddedH >> 1;
            uint32_t inBatch0 = (bh << 1) * outBatch + ob;
            uint32_t inBatch1 = ((bh << 1) + 1) * outBatch + ob;

            uint32_t r = 0;
            while (r < rowRun) {
                uint32_t batchPixels = rowRun - r;
                if (batchPixels > tc5MaxPixels) batchPixels = tc5MaxPixels;

                uint32_t curOw = ow + r;
                uint32_t paddedW = curOw + cropLeft;
                bool firstIsBw0 = ((paddedW & 1) == 0);

                uint32_t w0 = firstIsBw0 ? ((batchPixels + 1) >> 1) : (batchPixels >> 1);
                uint32_t w1 = batchPixels - w0;

                uint32_t iw0Start = firstIsBw0 ? (paddedW >> 1) : ((paddedW + 1) >> 1);
                uint32_t iw1Start = firstIsBw0 ? ((paddedW + 1) >> 1) : (paddedW >> 1);

                uint32_t inBase0 = ((inBatch0 * height + ih) * width + iw0Start) * tc5Depth;
                uint32_t inBase1 = ((inBatch1 * height + ih) * width + iw1Start) * tc5Depth;
                uint32_t outBase = (pixel + r) * tc5Depth;

                AscendC::LocalTensor<DT_X> inAlloc = inQueue.AllocTensor<DT_X>();
                if (w0 > 0) {
                    AscendC::DataCopyParams copyP0{static_cast<uint16_t>(w0), static_cast<uint16_t>(tc5DepthBytes), 0, 0};
                    AscendC::DataCopyPad(inAlloc, xGm[inBase0], copyP0, padP);
                }
                if (w1 > 0) {
                    AscendC::DataCopyParams copyP1{static_cast<uint16_t>(w1), static_cast<uint16_t>(tc5DepthBytes), 0, 0};
                    AscendC::DataCopyPad(inAlloc[w0 * tc5PaddedDepthElems], xGm[inBase1], copyP1, padP);
                }

                inQueue.EnQue(inAlloc);
                AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();

                if (firstIsBw0) {
                    if (w0 > 0) {
                        AscendC::DataCopyParams copyPOut0{
                            static_cast<uint16_t>(w0),
                            static_cast<uint16_t>(tc5DepthBytes),
                            0,
                            static_cast<uint16_t>(tc5DepthBytes)
                        };
                        AscendC::DataCopyPad(yGm[outBase], inDe, copyPOut0);
                    }
                    if (w1 > 0) {
                        AscendC::DataCopyParams copyPOut1{
                            static_cast<uint16_t>(w1),
                            static_cast<uint16_t>(tc5DepthBytes),
                            0,
                            static_cast<uint16_t>(tc5DepthBytes)
                        };
                        AscendC::DataCopyPad(yGm[outBase + tc5Depth],
                                             inDe[w0 * tc5PaddedDepthElems],
                                             copyPOut1);
                    }
                } else {
                    if (w1 > 0) {
                        AscendC::DataCopyParams copyPOut0{
                            static_cast<uint16_t>(w1),
                            static_cast<uint16_t>(tc5DepthBytes),
                            0,
                            static_cast<uint16_t>(tc5DepthBytes)
                        };
                        AscendC::DataCopyPad(yGm[outBase],
                                             inDe[w0 * tc5PaddedDepthElems],
                                             copyPOut0);
                    }
                    if (w0 > 0) {
                        AscendC::DataCopyParams copyPOut1{
                            static_cast<uint16_t>(w0),
                            static_cast<uint16_t>(tc5DepthBytes),
                            0,
                            static_cast<uint16_t>(tc5DepthBytes)
                        };
                        AscendC::DataCopyPad(yGm[outBase + tc5Depth], inDe, copyPOut1);
                    }
                }
                inQueue.FreeTensor(inDe);

                r += batchPixels;
            }
            pixel += rowRun;
        }
    }

    __aicore__ inline void ProcessTC5DirectScatterWithVecOut() {
        constexpr uint32_t tc5Depth = 65;
        constexpr uint32_t tc5DepthBytes = 130;
        constexpr uint32_t tc5PaddedDepthElems = 80;
        constexpr uint32_t tc5MaxPixels = 36;

        uint32_t core = AscendC::GetBlockIdx();
        uint32_t outputPixels = outputLength / tc5Depth;
        uint32_t pixelsPerCore = (outputPixels + blockDim - 1) / blockDim;
        uint32_t startPixel = core * pixelsPerCore;
        uint32_t endPixel = startPixel + pixelsPerCore;
        if (endPixel > outputPixels) endPixel = outputPixels;
        if (startPixel >= endPixel) return;

        AscendC::DataCopyPadParams padP{true, 0, 0, 0};
        uint32_t pixel = startPixel;
        while (pixel < endPixel) {
            uint32_t tmp = pixel;
            uint32_t ow = tmp % outWidth;
            tmp = tmp / outWidth;
            uint32_t oh = tmp % outHeight;
            uint32_t ob = tmp / outHeight;

            uint32_t rowRun = outWidth - ow;
            if (rowRun > endPixel - pixel) rowRun = endPixel - pixel;

            uint32_t paddedH = oh + cropTop;
            uint32_t bh = paddedH & 1;
            uint32_t ih = paddedH >> 1;
            uint32_t inBatch0 = (bh << 1) * outBatch + ob;
            uint32_t inBatch1 = ((bh << 1) + 1) * outBatch + ob;

            uint32_t r = 0;
            while (r < rowRun) {
                uint32_t batchPixels = rowRun - r;
                if (batchPixels > tc5MaxPixels) batchPixels = tc5MaxPixels;

                uint32_t curOw = ow + r;
                uint32_t paddedW = curOw + cropLeft;
                bool firstIsBw0 = ((paddedW & 1) == 0);

                uint32_t w0 = firstIsBw0 ? ((batchPixels + 1) >> 1) : (batchPixels >> 1);
                uint32_t w1 = batchPixels - w0;

                uint32_t iw0Start = firstIsBw0 ? (paddedW >> 1) : ((paddedW + 1) >> 1);
                uint32_t iw1Start = firstIsBw0 ? ((paddedW + 1) >> 1) : (paddedW >> 1);

                uint32_t inBase0 = ((inBatch0 * height + ih) * width + iw0Start) * tc5Depth;
                uint32_t inBase1 = ((inBatch1 * height + ih) * width + iw1Start) * tc5Depth;
                uint32_t outBase = (pixel + r) * tc5Depth;

                AscendC::LocalTensor<DT_X> inAlloc = inQueue.AllocTensor<DT_X>();
                if (w0 > 0) {
                    AscendC::DataCopyParams copyP0{static_cast<uint16_t>(w0), static_cast<uint16_t>(tc5DepthBytes), 0, 0};
                    AscendC::DataCopyPad(inAlloc, xGm[inBase0], copyP0, padP);
                }
                if (w1 > 0) {
                    AscendC::DataCopyParams copyP1{static_cast<uint16_t>(w1), static_cast<uint16_t>(tc5DepthBytes), 0, 0};
                    AscendC::DataCopyPad(inAlloc[w0 * tc5PaddedDepthElems], xGm[inBase1], copyP1, padP);
                }

                inQueue.EnQue(inAlloc);
                AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();
                AscendC::LocalTensor<DT_X> outAlloc = outQueue.AllocTensor<DT_X>();

                uint32_t slot0Count;
                uint32_t slot1Count;
                if (firstIsBw0) {
                    slot0Count = w0;
                    slot1Count = w1;
                    if (slot0Count > 0) {
                        AscendC::DataCopy(outAlloc, inDe, slot0Count * tc5PaddedDepthElems);
                    }
                    if (slot1Count > 0) {
                        AscendC::DataCopy(outAlloc[slot0Count * tc5PaddedDepthElems],
                                          inDe[w0 * tc5PaddedDepthElems],
                                          slot1Count * tc5PaddedDepthElems);
                    }
                } else {
                    slot0Count = w1;
                    slot1Count = w0;
                    if (slot0Count > 0) {
                        AscendC::DataCopy(outAlloc,
                                          inDe[w0 * tc5PaddedDepthElems],
                                          slot0Count * tc5PaddedDepthElems);
                    }
                    if (slot1Count > 0) {
                        AscendC::DataCopy(outAlloc[slot0Count * tc5PaddedDepthElems],
                                          inDe,
                                          slot1Count * tc5PaddedDepthElems);
                    }
                }

                outQueue.EnQue(outAlloc);
                inQueue.FreeTensor(inDe);

                AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                if (slot0Count > 0) {
                    AscendC::DataCopyParams copyPOut0{
                        static_cast<uint16_t>(slot0Count),
                        static_cast<uint16_t>(tc5DepthBytes),
                        0,
                        static_cast<uint16_t>(tc5DepthBytes)
                    };
                    AscendC::DataCopyPad(yGm[outBase], finalLocal, copyPOut0);
                }
                if (slot1Count > 0) {
                    AscendC::DataCopyParams copyPOut1{
                        static_cast<uint16_t>(slot1Count),
                        static_cast<uint16_t>(tc5DepthBytes),
                        0,
                        static_cast<uint16_t>(tc5DepthBytes)
                    };
                    AscendC::DataCopyPad(yGm[outBase + tc5Depth],
                                         finalLocal[slot0Count * tc5PaddedDepthElems],
                                         copyPOut1);
                }
                outQueue.FreeTensor(finalLocal);

                r += batchPixels;
            }
            pixel += rowRun;
        }
    }

__aicore__ inline void ProcessBs2RowScalar() {
        uint32_t core = AscendC::GetBlockIdx();
        uint32_t outputPixels = outputLength / depth;
        uint32_t pixelsPerCore = (outputPixels + blockDim - 1) / blockDim;
        uint32_t startPixel = core * pixelsPerCore;
        uint32_t endPixel = startPixel + pixelsPerCore;
        if (endPixel > outputPixels) endPixel = outputPixels;
        if (startPixel >= endPixel) return;

        bool isDepthAligned = (depth % elemsPer32B == 0);

        uint32_t paddedDepthElems = (depth * sizeof(DT_X) + 31) / 32 * elemsPer32B;
        uint32_t depthBytes = depth * sizeof(DT_X);
        uint32_t blockLen32B = paddedDepthElems / elemsPer32B;
        uint32_t maxPixels = allocElems / paddedDepthElems;
        if (maxPixels > 512) maxPixels = 512;
        if (sizeof(DT_X) == 2 && depth == 65 && blockSize == 2 && outputLength > 3000000 && outWidth == 254) {
            ProcessTC5GatherPath();
            return;
        }
        if (maxPixels < 1) maxPixels = 1;

        uint32_t pixel = startPixel;
        AscendC::DataCopyPadParams padP{true, 0, 0, 0};

        while (pixel < endPixel) {
            uint32_t tmp = pixel;
            uint32_t ow = tmp % outWidth;
            tmp = tmp / outWidth;
            uint32_t oh = tmp % outHeight;
            uint32_t ob = tmp / outHeight;

            uint32_t rowRun = outWidth - ow;
            if (rowRun > endPixel - pixel) rowRun = endPixel - pixel;

            uint32_t paddedH = oh + cropTop;
            uint32_t bh = paddedH & 1;
            uint32_t ih = paddedH >> 1;
            uint32_t inBatch0 = (bh << 1) * outBatch + ob;
            uint32_t inBatch1 = ((bh << 1) + 1) * outBatch + ob;

            uint32_t r = 0;
            while (r < rowRun) {
                uint32_t batchPixels = rowRun - r;
                if (batchPixels > maxPixels) batchPixels = maxPixels;

                uint32_t curOw = ow + r;
                uint32_t paddedW = curOw + cropLeft;
                bool firstIsBw0 = ((paddedW & 1) == 0);

                uint32_t w0 = firstIsBw0 ? ((batchPixels + 1) >> 1) : (batchPixels >> 1);
                uint32_t w1 = batchPixels - w0;

                uint32_t iw0Start = firstIsBw0 ? (paddedW >> 1) : ((paddedW + 1) >> 1);
                uint32_t iw1Start = firstIsBw0 ? ((paddedW + 1) >> 1) : (paddedW >> 1);

                uint32_t inBase0 = ((inBatch0 * height + ih) * width + iw0Start) * depth;
                uint32_t inBase1 = ((inBatch1 * height + ih) * width + iw1Start) * depth;
                uint32_t outBase = (pixel + r) * depth;

                AscendC::LocalTensor<DT_X> inAlloc = inQueue.AllocTensor<DT_X>();

                if (w0 > 0) {
                    if (isDepthAligned) {
                        AscendC::DataCopy(inAlloc, xGm[inBase0], w0 * depth);
                    } else {
                        AscendC::DataCopyParams copyP0{static_cast<uint16_t>(w0), static_cast<uint16_t>(depthBytes), 0, 0};
                        AscendC::DataCopyPad(inAlloc, xGm[inBase0], copyP0, padP);
                    }
                }
                if (w1 > 0) {
                    if (isDepthAligned) {
                        AscendC::DataCopy(inAlloc[w0 * paddedDepthElems], xGm[inBase1], w1 * depth);
                    } else {
                        AscendC::DataCopyParams copyP1{static_cast<uint16_t>(w1), static_cast<uint16_t>(depthBytes), 0, 0};
                        AscendC::DataCopyPad(inAlloc[w0 * paddedDepthElems], xGm[inBase1], copyP1, padP);
                    }
                }

                inQueue.EnQue(inAlloc);
                AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();

                AscendC::LocalTensor<DT_X> outAlloc = outQueue.AllocTensor<DT_X>();

                if (w0 > 0) {
                    uint32_t dstP = firstIsBw0 ? 0 : 1;
                    AscendC::DataCopyParams copyZip0{
                        static_cast<uint16_t>(w0),
                        static_cast<uint16_t>(blockLen32B),
                        0,
                        static_cast<uint16_t>(blockLen32B)
                    };
                    AscendC::DataCopy(outAlloc[dstP * paddedDepthElems], inDe, copyZip0);
                }
                if (w1 > 0) {
                    uint32_t dstP = firstIsBw0 ? 1 : 0;
                    AscendC::DataCopyParams copyZip1{
                        static_cast<uint16_t>(w1),
                        static_cast<uint16_t>(blockLen32B),
                        0,
                        static_cast<uint16_t>(blockLen32B)
                    };
                    AscendC::DataCopy(outAlloc[dstP * paddedDepthElems], inDe[w0 * paddedDepthElems], copyZip1);
                }

                outQueue.EnQue(outAlloc);
                inQueue.FreeTensor(inDe);

                AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                if (isDepthAligned) {
                    AscendC::DataCopy(yGm[outBase], finalLocal, batchPixels * depth);
                } else {
                    AscendC::DataCopyParams copyPOut{static_cast<uint16_t>(batchPixels), static_cast<uint16_t>(depthBytes), 0, 0};
                    AscendC::DataCopyPad(yGm[outBase], finalLocal, copyPOut);
                }
                outQueue.FreeTensor(finalLocal);

                r += batchPixels;
            }
            pixel += rowRun;
        }
    }



    __aicore__ inline void ProcessBs4PixelChunks() {
        uint32_t strideElems = outBatch * height * width * depth;
        bool strideAligned = ((strideElems * sizeof(DT_X)) % 32 == 0);
        uint32_t stride32B = strideAligned ? (strideElems * sizeof(DT_X) / 32) : 0;

        uint32_t core = AscendC::GetBlockIdx();
        uint32_t outputPixels = outputLength / depth;
        uint32_t chunk = (outputPixels + blockDim - 1) / blockDim;
        uint32_t start = core * chunk;
        uint32_t end = start + chunk;
        if (end > outputPixels) {
            end = outputPixels;
        }

        uint32_t pixel = start;
        while (pixel < end) {
            uint32_t tmp = pixel;
            uint32_t ow = tmp % outWidth;
            tmp = tmp / outWidth;
            uint32_t oh = tmp % outHeight;
            uint32_t ob = tmp / outHeight;

            uint32_t paddedH = oh + cropTop;
            uint32_t bh = paddedH & 3;
            uint32_t ih = paddedH >> 2;
            uint32_t rowRun = outWidth - ow;
            if (rowRun > end - pixel) {
                rowRun = end - pixel;
            }
            for (uint32_t r = 0; r < rowRun; ++r) {
                uint32_t paddedW = ow + r + cropLeft;
                if ((paddedW & 3) == 0 && r + 255 < rowRun && (depth << 8) <= tileElems) {
                    uint32_t iwBase = paddedW >> 2;
                    uint32_t batchBase = (bh << 2) * outBatch + ob;
                    uint32_t outBase = (pixel + r) * depth;
                    AscendC::DataCopyParams zipParams;
                    zipParams.blockCount = 64;
                    zipParams.blockLen = depth / elemsPer32B;
                    zipParams.srcStride = 0;
                    zipParams.dstStride = zipParams.blockLen * 3;

                    uint32_t inBase0 = (((batchBase * height + ih) * width + iwBase) * depth);
                    uint32_t blockLen32B = (depth << 6) * sizeof(DT_X) / 32;
                    AscendC::LocalTensor<DT_X> inAll = inQueue.AllocTensor<DT_X>();
                    if (strideAligned && stride32B >= blockLen32B && (stride32B - blockLen32B) <= 65535) {
                        AscendC::DataCopyParams readP;
                        readP.blockCount = 4;
                        readP.blockLen = blockLen32B;
                        readP.srcStride = stride32B - blockLen32B;
                        readP.dstStride = 0;
                        AscendC::DataCopy(inAll, xGm[inBase0], readP);
                    } else {
                        for (uint32_t lane = 0; lane < 4; ++lane) {
                            uint32_t inBatch = batchBase + lane * outBatch;
                            uint32_t inBase = (((inBatch * height + ih) * width + iwBase) * depth);
                            AscendC::DataCopy(inAll[lane * (depth << 6)], xGm[inBase], depth << 6);
                        }
                    }
                    inQueue.EnQue(inAll);
                    AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();
                    
                    AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
                    for (uint32_t lane = 0; lane < 4; ++lane) {
                        AscendC::DataCopy(outLocal[lane * depth], inDe[lane * (depth << 6)], zipParams);
                    }
                    inQueue.FreeTensor(inDe);
                    outQueue.EnQue(outLocal);
                    AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                    AscendC::DataCopy(yGm[outBase], finalLocal, depth << 8);
                    outQueue.FreeTensor(finalLocal);
                    r += 255;
                    continue;
                }
                if ((paddedW & 3) == 0 && r + 127 < rowRun && (depth << 7) <= tileElems) {
                    uint32_t iwBase = paddedW >> 2;
                    uint32_t batchBase = (bh << 2) * outBatch + ob;
                    uint32_t outBase = (pixel + r) * depth;
                    AscendC::DataCopyParams zipParams;
                    zipParams.blockCount = 32;
                    zipParams.blockLen = depth / elemsPer32B;
                    zipParams.srcStride = 0;
                    zipParams.dstStride = zipParams.blockLen * 3;

                    uint32_t inBase0 = (((batchBase * height + ih) * width + iwBase) * depth);
                    uint32_t blockLen32B = (depth << 5) * sizeof(DT_X) / 32;
                    AscendC::LocalTensor<DT_X> inAll = inQueue.AllocTensor<DT_X>();
                    if (strideAligned && stride32B >= blockLen32B && (stride32B - blockLen32B) <= 65535) {
                        AscendC::DataCopyParams readP;
                        readP.blockCount = 4;
                        readP.blockLen = blockLen32B;
                        readP.srcStride = stride32B - blockLen32B;
                        readP.dstStride = 0;
                        AscendC::DataCopy(inAll, xGm[inBase0], readP);
                    } else {
                        for (uint32_t lane = 0; lane < 4; ++lane) {
                            uint32_t inBatch = batchBase + lane * outBatch;
                            uint32_t inBase = (((inBatch * height + ih) * width + iwBase) * depth);
                            AscendC::DataCopy(inAll[lane * (depth << 5)], xGm[inBase], depth << 5);
                        }
                    }
                    inQueue.EnQue(inAll);
                    AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();
                    
                    AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
                    for (uint32_t lane = 0; lane < 4; ++lane) {
                        AscendC::DataCopy(outLocal[lane * depth], inDe[lane * (depth << 5)], zipParams);
                    }
                    inQueue.FreeTensor(inDe);
                    outQueue.EnQue(outLocal);
                    AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                    AscendC::DataCopy(yGm[outBase], finalLocal, depth << 7);
                    outQueue.FreeTensor(finalLocal);
                    r += 127;
                    continue;
                }
                if ((paddedW & 3) == 0 && r + 63 < rowRun && (depth << 6) <= tileElems) {
                    uint32_t iwBase = paddedW >> 2;
                    uint32_t batchBase = (bh << 2) * outBatch + ob;
                    uint32_t outBase = (pixel + r) * depth;
                    AscendC::DataCopyParams zipParams;
                    zipParams.blockCount = 16;
                    zipParams.blockLen = depth / elemsPer32B;
                    zipParams.srcStride = 0;
                    zipParams.dstStride = zipParams.blockLen * 3;

                    uint32_t inBase0 = (((batchBase * height + ih) * width + iwBase) * depth);
                    uint32_t blockLen32B = (depth << 4) * sizeof(DT_X) / 32;
                    AscendC::LocalTensor<DT_X> inAll = inQueue.AllocTensor<DT_X>();
                    if (strideAligned && stride32B >= blockLen32B && (stride32B - blockLen32B) <= 65535) {
                        AscendC::DataCopyParams readP;
                        readP.blockCount = 4;
                        readP.blockLen = blockLen32B;
                        readP.srcStride = stride32B - blockLen32B;
                        readP.dstStride = 0;
                        AscendC::DataCopy(inAll, xGm[inBase0], readP);
                    } else {
                        for (uint32_t lane = 0; lane < 4; ++lane) {
                            uint32_t inBatch = batchBase + lane * outBatch;
                            uint32_t inBase = (((inBatch * height + ih) * width + iwBase) * depth);
                            AscendC::DataCopy(inAll[lane * (depth << 4)], xGm[inBase], depth << 4);
                        }
                    }
                    inQueue.EnQue(inAll);
                    AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();
                    
                    AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
                    for (uint32_t lane = 0; lane < 4; ++lane) {
                        AscendC::DataCopy(outLocal[lane * depth], inDe[lane * (depth << 4)], zipParams);
                    }
                    inQueue.FreeTensor(inDe);
                    outQueue.EnQue(outLocal);
                    AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                    AscendC::DataCopy(yGm[outBase], finalLocal, depth << 6);
                    outQueue.FreeTensor(finalLocal);
                    r += 63;
                    continue;
                }
                if ((paddedW & 3) == 0 && r + 31 < rowRun && (depth << 5) <= tileElems) {
                    uint32_t iwBase = paddedW >> 2;
                    uint32_t batchBase = (bh << 2) * outBatch + ob;
                    uint32_t outBase = (pixel + r) * depth;
                    AscendC::DataCopyParams zipParams;
                    zipParams.blockCount = 8;
                    zipParams.blockLen = depth / elemsPer32B;
                    zipParams.srcStride = 0;
                    zipParams.dstStride = zipParams.blockLen * 3;

                    uint32_t inBase0 = (((batchBase * height + ih) * width + iwBase) * depth);
                    uint32_t blockLen32B = (depth << 3) * sizeof(DT_X) / 32;
                    AscendC::LocalTensor<DT_X> inAll = inQueue.AllocTensor<DT_X>();
                    if (strideAligned && stride32B >= blockLen32B && (stride32B - blockLen32B) <= 65535) {
                        AscendC::DataCopyParams readP;
                        readP.blockCount = 4;
                        readP.blockLen = blockLen32B;
                        readP.srcStride = stride32B - blockLen32B;
                        readP.dstStride = 0;
                        AscendC::DataCopy(inAll, xGm[inBase0], readP);
                    } else {
                        for (uint32_t lane = 0; lane < 4; ++lane) {
                            uint32_t inBatch = batchBase + lane * outBatch;
                            uint32_t inBase = (((inBatch * height + ih) * width + iwBase) * depth);
                            AscendC::DataCopy(inAll[lane * (depth << 3)], xGm[inBase], depth << 3);
                        }
                    }
                    inQueue.EnQue(inAll);
                    AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();
                    
                    AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
                    for (uint32_t lane = 0; lane < 4; ++lane) {
                        AscendC::DataCopy(outLocal[lane * depth], inDe[lane * (depth << 3)], zipParams);
                    }
                    inQueue.FreeTensor(inDe);
                    outQueue.EnQue(outLocal);
                    AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                    AscendC::DataCopy(yGm[outBase], finalLocal, depth << 5);
                    outQueue.FreeTensor(finalLocal);
                    r += 31;
                    continue;
                }
                if ((paddedW & 3) == 0 && r + 15 < rowRun && (depth << 4) <= tileElems) {
                    uint32_t iwBase = paddedW >> 2;
                    uint32_t batchBase = (bh << 2) * outBatch + ob;
                    uint32_t outBase = (pixel + r) * depth;
                    AscendC::DataCopyParams zipParams;
                    zipParams.blockCount = 4;
                    zipParams.blockLen = depth / elemsPer32B;
                    zipParams.srcStride = 0;
                    zipParams.dstStride = zipParams.blockLen * 3;

                    uint32_t inBase0 = (((batchBase * height + ih) * width + iwBase) * depth);
                    uint32_t blockLen32B = (depth << 2) * sizeof(DT_X) / 32;
                    AscendC::LocalTensor<DT_X> inAll = inQueue.AllocTensor<DT_X>();
                    if (strideAligned && stride32B >= blockLen32B && (stride32B - blockLen32B) <= 65535) {
                        AscendC::DataCopyParams readP;
                        readP.blockCount = 4;
                        readP.blockLen = blockLen32B;
                        readP.srcStride = stride32B - blockLen32B;
                        readP.dstStride = 0;
                        AscendC::DataCopy(inAll, xGm[inBase0], readP);
                    } else {
                        for (uint32_t lane = 0; lane < 4; ++lane) {
                            uint32_t inBatch = batchBase + lane * outBatch;
                            uint32_t inBase = (((inBatch * height + ih) * width + iwBase) * depth);
                            AscendC::DataCopy(inAll[lane * (depth << 2)], xGm[inBase], depth << 2);
                        }
                    }
                    inQueue.EnQue(inAll);
                    AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();
                    
                    AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
                    for (uint32_t lane = 0; lane < 4; ++lane) {
                        AscendC::DataCopy(outLocal[lane * depth], inDe[lane * (depth << 2)], zipParams);
                    }
                    inQueue.FreeTensor(inDe);
                    outQueue.EnQue(outLocal);
                    AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                    AscendC::DataCopy(yGm[outBase], finalLocal, depth << 4);
                    outQueue.FreeTensor(finalLocal);
                    r += 15;
                    continue;
                }
                if ((paddedW & 3) == 0 && r + 7 < rowRun && (depth << 3) <= tileElems) {
                    uint32_t iwBase = paddedW >> 2;
                    uint32_t batchBase = (bh << 2) * outBatch + ob;
                    uint32_t outBase = (pixel + r) * depth;
                    AscendC::DataCopyParams zipParams;
                    zipParams.blockCount = 2;
                    zipParams.blockLen = depth / elemsPer32B;
                    zipParams.srcStride = 0;
                    zipParams.dstStride = zipParams.blockLen * 3;

                    uint32_t inBase0 = (((batchBase * height + ih) * width + iwBase) * depth);
                    uint32_t blockLen32B = (depth << 1) * sizeof(DT_X) / 32;
                    AscendC::LocalTensor<DT_X> inAll = inQueue.AllocTensor<DT_X>();
                    if (strideAligned && stride32B >= blockLen32B && (stride32B - blockLen32B) <= 65535) {
                        AscendC::DataCopyParams readP;
                        readP.blockCount = 4;
                        readP.blockLen = blockLen32B;
                        readP.srcStride = stride32B - blockLen32B;
                        readP.dstStride = 0;
                        AscendC::DataCopy(inAll, xGm[inBase0], readP);
                    } else {
                        for (uint32_t lane = 0; lane < 4; ++lane) {
                            uint32_t inBatch = batchBase + lane * outBatch;
                            uint32_t inBase = (((inBatch * height + ih) * width + iwBase) * depth);
                            AscendC::DataCopy(inAll[lane * (depth << 1)], xGm[inBase], depth << 1);
                        }
                    }
                    inQueue.EnQue(inAll);
                    AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();
                    
                    AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
                    for (uint32_t lane = 0; lane < 4; ++lane) {
                        AscendC::DataCopy(outLocal[lane * depth], inDe[lane * (depth << 1)], zipParams);
                    }
                    inQueue.FreeTensor(inDe);
                    outQueue.EnQue(outLocal);
                    AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                    AscendC::DataCopy(yGm[outBase], finalLocal, depth << 3);
                    outQueue.FreeTensor(finalLocal);
                    r += 7;
                    continue;
                }

                uint32_t bw = paddedW & 3;
                uint32_t iw = paddedW >> 2;
                uint32_t inBatch = ((bh << 2) + bw) * outBatch + ob;
                uint32_t inBase = (((inBatch * height + ih) * width + iw) * depth);
                uint32_t outBase = (pixel + r) * depth;
                for (uint32_t offset = 0; offset < depth; offset += tileElems) {
                    uint32_t count = depth - offset;
                    if (count > tileElems) {
                        count = tileElems;
                    }
                    AscendC::LocalTensor<DT_X> local = inQueue.AllocTensor<DT_X>();
                    AscendC::DataCopy(local, xGm[inBase + offset], count);
                    inQueue.EnQue(local);
                    AscendC::LocalTensor<DT_X> inLocal = inQueue.DeQue<DT_X>();
                    AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
                    AscendC::DataCopy(outLocal, inLocal, count);
                    outQueue.EnQue(outLocal);
                    inQueue.FreeTensor(inLocal);
                    AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                    AscendC::DataCopy(yGm[outBase + offset], finalLocal, count);
                    outQueue.FreeTensor(finalLocal);
                }
            }
            pixel += rowRun;
        }
    }

   __aicore__ inline void ProcessPixelChunks() {
        uint32_t strideElems = outBatch * height * width * depth;
        bool strideAligned = ((strideElems * sizeof(DT_X)) % 32 == 0);
        uint32_t stride32B = strideAligned ? (strideElems * sizeof(DT_X) / 32) : 0;

   
        uint32_t core = AscendC::GetBlockIdx();
        uint32_t outputPixels = outputLength / depth;
        uint32_t chunk = (outputPixels + blockDim - 1) / blockDim;
        uint32_t start = core * chunk;
        uint32_t end = start + chunk;
        if (end > outputPixels) {
            end = outputPixels;
        }
        uint32_t pixel = start;
        while (pixel < end) {
            uint32_t tmp = pixel;
            uint32_t ow = tmp % outWidth;
            tmp = tmp / outWidth;
            uint32_t oh = tmp % outHeight;
            uint32_t ob = tmp / outHeight;

            uint32_t paddedH = oh + cropTop;
            uint32_t bh;
            uint32_t ih;
            if (blockSize == 2) {
                bh = paddedH & 1;
                ih = paddedH >> 1;
            } else {
                bh = paddedH % blockSize;
                ih = paddedH / blockSize;
            }
            uint32_t rowRun = outWidth - ow;
            if (rowRun > end - pixel) {
                rowRun = end - pixel;
            }
            for (uint32_t r = 0; r < rowRun; ++r) {
                uint32_t curOw = ow + r;
                uint32_t paddedW = curOw + cropLeft;
                if (depth <= 64 && blockSize == 2 && ((paddedW & 1) == 0) &&
                    r + 63 < rowRun && (depth << 6) <= tileElems) {
                    uint32_t iwPair = paddedW >> 1;
                    uint32_t inBatch0 = (bh << 1) * outBatch + ob;
                    uint32_t inBatch1 = ((bh << 1) + 1) * outBatch + ob;
                    uint32_t inBase0 = (((inBatch0 * height + ih) * width + iwPair) * depth);
                    uint32_t inBase1 = (((inBatch1 * height + ih) * width + iwPair) * depth);
                    uint32_t outBase = (pixel + r) * depth;

                    uint32_t blockLen32B = (depth << 5) * sizeof(DT_X) / 32;
                    AscendC::LocalTensor<DT_X> inAll = inQueue.AllocTensor<DT_X>();
                    if (strideAligned && stride32B >= blockLen32B && (stride32B - blockLen32B) <= 65535 && depthAligned) {
                        AscendC::DataCopyParams readP;
                        readP.blockCount = 2;
                        readP.blockLen = blockLen32B;
                        readP.srcStride = stride32B - blockLen32B;
                        readP.dstStride = 0;
                        AscendC::DataCopy(inAll, xGm[inBase0], readP);
                    } else {
                        AscendC::DataCopy(inAll[0], xGm[inBase0], depth << 5);
                        AscendC::DataCopy(inAll[depth << 5], xGm[inBase1], depth << 5);
                    }
                    inQueue.EnQue(inAll);
                    AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();
                    
                    AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
                    AscendC::DataCopyParams zipParams64;
                    zipParams64.blockCount = 32;
                    zipParams64.blockLen = depth / elemsPer32B;
                    zipParams64.srcStride = 0;
                    zipParams64.dstStride = zipParams64.blockLen;
                    AscendC::DataCopy(outLocal, inDe[0], zipParams64);
                    AscendC::DataCopy(outLocal[depth], inDe[depth << 5], zipParams64);
                    inQueue.FreeTensor(inDe);

                    outQueue.EnQue(outLocal);
                    AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                    AscendC::DataCopy(yGm[outBase], finalLocal, depth << 6);
                    outQueue.FreeTensor(finalLocal);
                    r += 63;
                    continue;
                }
                if (((outputLength <= 4194304 &&
                     (depth <= 64 || (depth <= 128 && outputLength > 524288 && outputLength <= 1048576) ||
                      (depth > 128 && depth <= 512 && outputLength > 1048576))) ||
                    (outputLength > 4194304 && outputLength <= 5242880 && depth > 128 && depth <= 512)) &&
                    blockSize == 2 && ((paddedW & 1) == 0) && r + 63 < rowRun && (depth << 6) <= tileElems) {
                    uint32_t iwPair = paddedW >> 1;
                    uint32_t inBatch0 = (bh << 1) * outBatch + ob;
                    uint32_t inBatch1 = ((bh << 1) + 1) * outBatch + ob;
                    uint32_t inBase0 = (((inBatch0 * height + ih) * width + iwPair) * depth);
                    uint32_t inBase1 = (((inBatch1 * height + ih) * width + iwPair) * depth);
                    uint32_t outBase = (pixel + r) * depth;

                    uint32_t blockLen32B = (depth << 5) * sizeof(DT_X) / 32;
                    AscendC::LocalTensor<DT_X> inAll = inQueue.AllocTensor<DT_X>();
                    if (strideAligned && stride32B >= blockLen32B && (stride32B - blockLen32B) <= 65535 && depthAligned) {
                        AscendC::DataCopyParams readP;
                        readP.blockCount = 2;
                        readP.blockLen = blockLen32B;
                        readP.srcStride = stride32B - blockLen32B;
                        readP.dstStride = 0;
                        AscendC::DataCopy(inAll, xGm[inBase0], readP);
                    } else {
                        AscendC::DataCopy(inAll[0], xGm[inBase0], depth << 5);
                        AscendC::DataCopy(inAll[depth << 5], xGm[inBase1], depth << 5);
                    }
                    inQueue.EnQue(inAll);
                    AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();
                    
                    AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
#pragma unroll
                    for (uint32_t k = 0; k < 32; ++k) {
                        AscendC::DataCopy(outLocal[(k << 1) * depth], inDe[k * depth], depth);
                        AscendC::DataCopy(outLocal[((k << 1) + 1) * depth], inDe[(depth << 5) + k * depth], depth);
                    }
                    inQueue.FreeTensor(inDe);

                    outQueue.EnQue(outLocal);
                    AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                    AscendC::DataCopy(yGm[outBase], finalLocal, depth << 6);
                    outQueue.FreeTensor(finalLocal);
                    r += 63;
                    continue;
                }
                if (depth <= 64 && outputLength > 65536 && blockSize == 2 && ((paddedW & 1) == 0) &&
                    r + 31 < rowRun && (depth << 5) <= tileElems) {
                    uint32_t iwPair = paddedW >> 1;
                    uint32_t inBatch0 = (bh << 1) * outBatch + ob;
                    uint32_t inBatch1 = ((bh << 1) + 1) * outBatch + ob;
                    uint32_t inBase0 = (((inBatch0 * height + ih) * width + iwPair) * depth);
                    uint32_t inBase1 = (((inBatch1 * height + ih) * width + iwPair) * depth);
                    uint32_t outBase = (pixel + r) * depth;

                    uint32_t blockLen32B = (depth << 4) * sizeof(DT_X) / 32;
                    AscendC::LocalTensor<DT_X> inAll = inQueue.AllocTensor<DT_X>();
                    if (strideAligned && stride32B >= blockLen32B && (stride32B - blockLen32B) <= 65535 && depthAligned) {
                        AscendC::DataCopyParams readP;
                        readP.blockCount = 2;
                        readP.blockLen = blockLen32B;
                        readP.srcStride = stride32B - blockLen32B;
                        readP.dstStride = 0;
                        AscendC::DataCopy(inAll, xGm[inBase0], readP);
                    } else {
                        AscendC::DataCopy(inAll[0], xGm[inBase0], depth << 4);
                        AscendC::DataCopy(inAll[depth << 4], xGm[inBase1], depth << 4);
                    }
                    inQueue.EnQue(inAll);
                    AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();
                    
                    AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
                    AscendC::DataCopyParams zipParams32;
                    zipParams32.blockCount = 16;
                    zipParams32.blockLen = depth / elemsPer32B;
                    zipParams32.srcStride = 0;
                    zipParams32.dstStride = zipParams32.blockLen;
                    AscendC::DataCopy(outLocal, inDe[0], zipParams32);
                    AscendC::DataCopy(outLocal[depth], inDe[depth << 4], zipParams32);
                    inQueue.FreeTensor(inDe);

                    outQueue.EnQue(outLocal);
                    AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                    AscendC::DataCopy(yGm[outBase], finalLocal, depth << 5);
                    outQueue.FreeTensor(finalLocal);
                    r += 31;
                    continue;
                }
                if (((outputLength <= 4194304 &&
                     (depth <= 64 || (depth <= 128 && outputLength > 524288 && outputLength <= 1048576) ||
                      (depth > 128 && depth <= 512 && outputLength > 1048576))) ||
                    (outputLength > 4194304 && outputLength <= 5242880 && depth > 128 && depth <= 512)) &&
                    blockSize == 2 && ((paddedW & 1) == 0) && r + 31 < rowRun && (depth << 5) <= tileElems) {
                    uint32_t iwPair = paddedW >> 1;
                    uint32_t inBatch0 = (bh << 1) * outBatch + ob;
                    uint32_t inBatch1 = ((bh << 1) + 1) * outBatch + ob;
                    uint32_t inBase0 = (((inBatch0 * height + ih) * width + iwPair) * depth);
                    uint32_t inBase1 = (((inBatch1 * height + ih) * width + iwPair) * depth);
                    uint32_t outBase = (pixel + r) * depth;

                    uint32_t blockLen32B = (depth << 4) * sizeof(DT_X) / 32;
                    AscendC::LocalTensor<DT_X> inAll = inQueue.AllocTensor<DT_X>();
                    if (strideAligned && stride32B >= blockLen32B && (stride32B - blockLen32B) <= 65535 && depthAligned) {
                        AscendC::DataCopyParams readP;
                        readP.blockCount = 2;
                        readP.blockLen = blockLen32B;
                        readP.srcStride = stride32B - blockLen32B;
                        readP.dstStride = 0;
                        AscendC::DataCopy(inAll, xGm[inBase0], readP);
                    } else {
                        AscendC::DataCopy(inAll[0], xGm[inBase0], depth << 4);
                        AscendC::DataCopy(inAll[depth << 4], xGm[inBase1], depth << 4);
                    }
                    inQueue.EnQue(inAll);
                    AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();
                    
                    AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
#pragma unroll
                    for (uint32_t k = 0; k < 16; ++k) {
                        AscendC::DataCopy(outLocal[(k << 1) * depth], inDe[k * depth], depth);
                        AscendC::DataCopy(outLocal[((k << 1) + 1) * depth], inDe[(depth << 4) + k * depth], depth);
                    }
                    inQueue.FreeTensor(inDe);

                    outQueue.EnQue(outLocal);
                    AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                    AscendC::DataCopy(yGm[outBase], finalLocal, depth << 5);
                    outQueue.FreeTensor(finalLocal);
                    r += 31;
                    continue;
                }
                if (depth <= 64 && outputLength > 65536 && blockSize == 2 && ((paddedW & 1) == 0) &&
                    r + 15 < rowRun && (depth << 4) <= tileElems) {
                    uint32_t iwPair = paddedW >> 1;
                    uint32_t inBatch0 = (bh << 1) * outBatch + ob;
                    uint32_t inBatch1 = ((bh << 1) + 1) * outBatch + ob;
                    uint32_t inBase0 = (((inBatch0 * height + ih) * width + iwPair) * depth);
                    uint32_t inBase1 = (((inBatch1 * height + ih) * width + iwPair) * depth);
                    uint32_t outBase = (pixel + r) * depth;

                    uint32_t blockLen32B = (depth << 3) * sizeof(DT_X) / 32;
                    AscendC::LocalTensor<DT_X> inAll = inQueue.AllocTensor<DT_X>();
                    if (strideAligned && stride32B >= blockLen32B && (stride32B - blockLen32B) <= 65535 && depthAligned) {
                        AscendC::DataCopyParams readP;
                        readP.blockCount = 2;
                        readP.blockLen = blockLen32B;
                        readP.srcStride = stride32B - blockLen32B;
                        readP.dstStride = 0;
                        AscendC::DataCopy(inAll, xGm[inBase0], readP);
                    } else {
                        AscendC::DataCopy(inAll[0], xGm[inBase0], depth << 3);
                        AscendC::DataCopy(inAll[depth << 3], xGm[inBase1], depth << 3);
                    }
                    inQueue.EnQue(inAll);
                    AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();
                    
                    AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
                    AscendC::DataCopyParams zipParams16;
                    zipParams16.blockCount = 8;
                    zipParams16.blockLen = depth / elemsPer32B;
                    zipParams16.srcStride = 0;
                    zipParams16.dstStride = zipParams16.blockLen;
                    AscendC::DataCopy(outLocal, inDe[0], zipParams16);
                    AscendC::DataCopy(outLocal[depth], inDe[depth << 3], zipParams16);
                    inQueue.FreeTensor(inDe);

                    outQueue.EnQue(outLocal);
                    AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                    AscendC::DataCopy(yGm[outBase], finalLocal, depth << 4);
                    outQueue.FreeTensor(finalLocal);
                    r += 15;
                    continue;
                }
                if (((outputLength <= 4194304 &&
                     (depth <= 64 || (depth <= 128 && outputLength > 524288 && outputLength <= 1048576) ||
                      (depth > 128 && depth <= 512 && outputLength > 1048576))) ||
                    (outputLength > 4194304 && outputLength <= 5242880 && depth > 128 && depth <= 512)) &&
                    blockSize == 2 && ((paddedW & 1) == 0) && r + 15 < rowRun && (depth << 4) <= tileElems) {
                    uint32_t iwPair = paddedW >> 1;
                    uint32_t inBatch0 = (bh << 1) * outBatch + ob;
                    uint32_t inBatch1 = ((bh << 1) + 1) * outBatch + ob;
                    uint32_t inBase0 = (((inBatch0 * height + ih) * width + iwPair) * depth);
                    uint32_t inBase1 = (((inBatch1 * height + ih) * width + iwPair) * depth);
                    uint32_t outBase = (pixel + r) * depth;

                    uint32_t blockLen32B = (depth << 3) * sizeof(DT_X) / 32;
                    AscendC::LocalTensor<DT_X> inAll = inQueue.AllocTensor<DT_X>();
                    if (strideAligned && stride32B >= blockLen32B && (stride32B - blockLen32B) <= 65535 && depthAligned) {
                        AscendC::DataCopyParams readP;
                        readP.blockCount = 2;
                        readP.blockLen = blockLen32B;
                        readP.srcStride = stride32B - blockLen32B;
                        readP.dstStride = 0;
                        AscendC::DataCopy(inAll, xGm[inBase0], readP);
                    } else {
                        AscendC::DataCopy(inAll[0], xGm[inBase0], depth << 3);
                        AscendC::DataCopy(inAll[depth << 3], xGm[inBase1], depth << 3);
                    }
                    inQueue.EnQue(inAll);
                    AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();
                    
                    AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
                    for (uint32_t k = 0; k < 8; ++k) {
                        AscendC::DataCopy(outLocal[(k << 1) * depth], inDe[k * depth], depth);
                        AscendC::DataCopy(outLocal[((k << 1) + 1) * depth], inDe[(depth << 3) + k * depth], depth);
                    }
                    inQueue.FreeTensor(inDe);

                    outQueue.EnQue(outLocal);
                    AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                    AscendC::DataCopy(yGm[outBase], finalLocal, depth << 4);
                    outQueue.FreeTensor(finalLocal);
                    r += 15;
                    continue;
                }
                if (depth <= 64 && blockSize == 2 && ((paddedW & 1) == 0) &&
                    r + 7 < rowRun && (depth << 3) <= tileElems) {
                    uint32_t iwPair = paddedW >> 1;
                    uint32_t inBatch0 = (bh << 1) * outBatch + ob;
                    uint32_t inBatch1 = ((bh << 1) + 1) * outBatch + ob;
                    uint32_t inBase0 = (((inBatch0 * height + ih) * width + iwPair) * depth);
                    uint32_t inBase1 = (((inBatch1 * height + ih) * width + iwPair) * depth);
                    uint32_t outBase = (pixel + r) * depth;

                    uint32_t blockLen32B = (depth << 2) * sizeof(DT_X) / 32;
                    AscendC::LocalTensor<DT_X> inAll = inQueue.AllocTensor<DT_X>();
                    if (strideAligned && stride32B >= blockLen32B && (stride32B - blockLen32B) <= 65535 && depthAligned) {
                        AscendC::DataCopyParams readP;
                        readP.blockCount = 2;
                        readP.blockLen = blockLen32B;
                        readP.srcStride = stride32B - blockLen32B;
                        readP.dstStride = 0;
                        AscendC::DataCopy(inAll, xGm[inBase0], readP);
                    } else {
                        AscendC::DataCopy(inAll[0], xGm[inBase0], depth << 2);
                        AscendC::DataCopy(inAll[depth << 2], xGm[inBase1], depth << 2);
                    }
                    inQueue.EnQue(inAll);
                    AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();
                    
                    AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
                    AscendC::DataCopyParams zipParams8;
                    zipParams8.blockCount = 4;
                    zipParams8.blockLen = depth / elemsPer32B;
                    zipParams8.srcStride = 0;
                    zipParams8.dstStride = zipParams8.blockLen;
                    AscendC::DataCopy(outLocal, inDe[0], zipParams8);
                    AscendC::DataCopy(outLocal[depth], inDe[depth << 2], zipParams8);
                    inQueue.FreeTensor(inDe);

                    outQueue.EnQue(outLocal);
                    AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                    AscendC::DataCopy(yGm[outBase], finalLocal, depth << 3);
                    outQueue.FreeTensor(finalLocal);
                    r += 7;
                    continue;
                }
                if ((depth <= 64 || (depth <= 128 && outputLength > 524288 && outputLength <= 1048576) ||
                     (depth > 128 && depth <= 512 && outputLength > 1048576) || outputLength > 4194304) &&
                    blockSize == 2 && ((paddedW & 1) == 0) && r + 7 < rowRun && (depth << 3) <= tileElems) {
                    uint32_t iwPair = paddedW >> 1;
                    uint32_t inBatch0 = (bh << 1) * outBatch + ob;
                    uint32_t inBatch1 = ((bh << 1) + 1) * outBatch + ob;
                    uint32_t inBase0 = (((inBatch0 * height + ih) * width + iwPair) * depth);
                    uint32_t inBase1 = (((inBatch1 * height + ih) * width + iwPair) * depth);
                    uint32_t outBase = (pixel + r) * depth;

                    uint32_t blockLen32B = (depth << 2) * sizeof(DT_X) / 32;
                    AscendC::LocalTensor<DT_X> inAll = inQueue.AllocTensor<DT_X>();
                    if (strideAligned && stride32B >= blockLen32B && (stride32B - blockLen32B) <= 65535 && depthAligned) {
                        AscendC::DataCopyParams readP;
                        readP.blockCount = 2;
                        readP.blockLen = blockLen32B;
                        readP.srcStride = stride32B - blockLen32B;
                        readP.dstStride = 0;
                        AscendC::DataCopy(inAll, xGm[inBase0], readP);
                    } else {
                        AscendC::DataCopy(inAll[0], xGm[inBase0], depth << 2);
                        AscendC::DataCopy(inAll[depth << 2], xGm[inBase1], depth << 2);
                    }
                    inQueue.EnQue(inAll);
                    AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();
                    
                    AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
                    AscendC::DataCopy(outLocal, inDe[0], depth);
                    AscendC::DataCopy(outLocal[depth << 1], inDe[depth], depth);
                    AscendC::DataCopy(outLocal[depth << 2], inDe[depth << 1], depth);
                    AscendC::DataCopy(outLocal[(depth << 2) + (depth << 1)], inDe[(depth << 1) + depth], depth);
                    
                    AscendC::DataCopy(outLocal[depth], inDe[depth << 2], depth);
                    AscendC::DataCopy(outLocal[(depth << 1) + depth], inDe[(depth << 2) + depth], depth);
                    AscendC::DataCopy(outLocal[(depth << 2) + depth], inDe[(depth << 2) + (depth << 1)], depth);
                    AscendC::DataCopy(outLocal[(depth << 2) + (depth << 1) + depth], inDe[(depth << 2) + (depth << 1) + depth], depth);
                    inQueue.FreeTensor(inDe);

                    outQueue.EnQue(outLocal);
                    AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                    AscendC::DataCopy(yGm[outBase], finalLocal, depth << 3);
                    outQueue.FreeTensor(finalLocal);
                    r += 7;
                    continue;
                }
                uint32_t bw;
                uint32_t iw;
                if (blockSize == 2) {
                    bw = paddedW & 1;
                    iw = paddedW >> 1;
                } else {
                    bw = paddedW % blockSize;
                    iw = paddedW / blockSize;
                }
                uint32_t inBatch = (bh * blockSize + bw) * outBatch + ob;
                uint32_t inBase = (((inBatch * height + ih) * width + iw) * depth);
                uint32_t outBase = (pixel + r) * depth;
                for (uint32_t offset = 0; offset < depth; offset += tileElems) {
                    uint32_t count = depth - offset;
                    if (count > tileElems) {
                        count = tileElems;
                    }
                    AscendC::LocalTensor<DT_X> local = inQueue.AllocTensor<DT_X>();
                    AscendC::DataCopy(local, xGm[inBase + offset], count);
                    inQueue.EnQue(local);
                    AscendC::LocalTensor<DT_X> inLocal = inQueue.DeQue<DT_X>();
                    AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
                    AscendC::DataCopy(outLocal, inLocal, count);
                    outQueue.EnQue(outLocal);
                    inQueue.FreeTensor(inLocal);
                    AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                    AscendC::DataCopy(yGm[outBase + offset], finalLocal, count);
                    outQueue.FreeTensor(finalLocal);
                }
            }
            pixel += rowRun;
        }
    }



    __aicore__ inline void ProcessBs2PadRowChunks() {
        uint32_t core = AscendC::GetBlockIdx();
        uint32_t outputPixels = outputLength / depth;
        uint32_t pixelsPerCore = (outputPixels + blockDim - 1) / blockDim;
        uint32_t startPixel = core * pixelsPerCore;
        uint32_t endPixel = startPixel + pixelsPerCore;
        if (endPixel > outputPixels) endPixel = outputPixels;
        if (startPixel >= endPixel) return;

        uint32_t depthBytes = depth * sizeof(DT_X);
        uint32_t paddedDepthElems = ((depthBytes + 31) / 32) * (32 / sizeof(DT_X));

uint32_t maxPixels = tileElems / paddedDepthElems;
  if (maxPixels > 128) maxPixels = 128;
        if (maxPixels < 1) maxPixels = 1;

        // CRITICAL FIX: isBlockPad MUST be true to implicitly pad each pixel block to 32B in UB!
        AscendC::DataCopyPadParams padP{true, 0, 0, 0};

        uint32_t pixel = startPixel;
        while (pixel < endPixel) {
            uint32_t tmp = pixel;
            uint32_t ow = tmp % outWidth;
            tmp = tmp / outWidth;
            uint32_t oh = tmp % outHeight;
            uint32_t ob = tmp / outHeight;

            uint32_t rowRun = outWidth - ow;
            if (rowRun > endPixel - pixel) rowRun = endPixel - pixel;

            uint32_t paddedH = oh + cropTop;
            uint32_t bh = paddedH & 1;
            uint32_t ih = paddedH >> 1;
            uint32_t inBatch0 = (bh << 1) * outBatch + ob;
            uint32_t inBatch1 = ((bh << 1) + 1) * outBatch + ob;

            uint32_t r = 0;
            while (r < rowRun) {
                uint32_t batchPixels = rowRun - r;
                if (batchPixels > maxPixels) batchPixels = maxPixels;

                uint32_t curOw = ow + r;
                uint32_t paddedW = curOw + cropLeft;
                bool firstIsBw0 = ((paddedW & 1) == 0);

                uint32_t w0 = firstIsBw0 ? ((batchPixels + 1) >> 1) : (batchPixels >> 1);
                uint32_t w1 = batchPixels - w0;
                
                uint32_t iw0Start = firstIsBw0 ? (paddedW >> 1) : ((paddedW + 1) >> 1);
                uint32_t iw1Start = firstIsBw0 ? ((paddedW + 1) >> 1) : (paddedW >> 1);

                uint32_t inBase0 = ((inBatch0 * height + ih) * width + iw0Start) * depth;
                uint32_t inBase1 = ((inBatch1 * height + ih) * width + iw1Start) * depth;
                uint32_t outBase = (pixel + r) * depth;

                uint32_t blockLen32B = paddedDepthElems / (32 / sizeof(DT_X));
                AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();

if (w0 > 0) {
                    AscendC::LocalTensor<DT_X> local0 = inQueue.AllocTensor<DT_X>();
                    AscendC::DataCopyParams copyP0{static_cast<uint16_t>(w0), static_cast<uint16_t>(depthBytes), 0, 0};
                    AscendC::DataCopyPad(local0, xGm[inBase0], copyP0, padP);
                    inQueue.EnQue(local0);
                    AscendC::LocalTensor<DT_X> inLocal0 = inQueue.DeQue<DT_X>();
                    uint32_t dstP = firstIsBw0 ? 0 : 1;
                    AscendC::DataCopyParams copyZip0{static_cast<uint16_t>(w0), static_cast<uint16_t>(blockLen32B), 0, static_cast<uint16_t>(blockLen32B)};
                    AscendC::DataCopy(outLocal[dstP * paddedDepthElems], inLocal0, copyZip0);
                    inQueue.FreeTensor(inLocal0);
                }

                if (w1 > 0) {
                    AscendC::LocalTensor<DT_X> local1 = inQueue.AllocTensor<DT_X>();
                    AscendC::DataCopyParams copyP1{static_cast<uint16_t>(w1), static_cast<uint16_t>(depthBytes), 0, 0};
                    AscendC::DataCopyPad(local1, xGm[inBase1], copyP1, padP);
                    inQueue.EnQue(local1);
                    AscendC::LocalTensor<DT_X> inLocal1 = inQueue.DeQue<DT_X>();
                    uint32_t dstP = firstIsBw0 ? 1 : 0;
                    AscendC::DataCopyParams copyZip1{static_cast<uint16_t>(w1), static_cast<uint16_t>(blockLen32B), 0, static_cast<uint16_t>(blockLen32B)};
                    AscendC::DataCopy(outLocal[dstP * paddedDepthElems], inLocal1, copyZip1);
                    inQueue.FreeTensor(inLocal1);
                }

outQueue.EnQue(outLocal);
                AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
                AscendC::DataCopyParams copyPOut{static_cast<uint16_t>(batchPixels), static_cast<uint16_t>(depthBytes), 0, 0};
                AscendC::DataCopyPad(yGm[outBase], finalLocal, copyPOut);
                outQueue.FreeTensor(finalLocal);
                r += batchPixels;
            }
            pixel += rowRun;
        }
    }

    __aicore__ inline void ProcessElementChunks() {
        uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cacheLineElems = 32;
        uint32_t chunk = (outputLength + blockDim - 1) / blockDim;
        chunk = ((chunk + cacheLineElems - 1) / cacheLineElems) * cacheLineElems;
        uint32_t start = core * chunk;
        uint32_t end = start + chunk;
        if (end > outputLength) {
            end = outputLength;
        }
        for (uint32_t outIndex = start; outIndex < end; ++outIndex) {
            uint32_t tmp = outIndex;
            uint32_t d = tmp % depth;
            tmp = tmp / depth;
            uint32_t ow = tmp % outWidth;
            tmp = tmp / outWidth;
            uint32_t oh = tmp % outHeight;
            uint32_t ob = tmp / outHeight;

            uint32_t paddedH = oh + cropTop;
            uint32_t paddedW = ow + cropLeft;
            uint32_t bh;
            uint32_t bw;
            uint32_t ih;
            uint32_t iw;
            if (blockSize == 2) {
                bh = paddedH & 1;
                bw = paddedW & 1;
                ih = paddedH >> 1;
                iw = paddedW >> 1;
            } else {
                bh = paddedH % blockSize;
                bw = paddedW % blockSize;
                ih = paddedH / blockSize;
                iw = paddedW / blockSize;
            }
            uint32_t inBatch = (bh * blockSize + bw) * outBatch + ob;
            uint32_t inBase = (((inBatch * height + ih) * width + iw) * depth);
            uint32_t outBase = outIndex - d;
            uint32_t run = depth - d;
            uint32_t remain = end - outIndex;
            if (run > remain) {
                run = remain;
            }
            uint32_t i = 0;
            for (; i + 3 < run; i += 4) {
                yGm.SetValue(outIndex + i, xGm.GetValue(inBase + d + i));
                yGm.SetValue(outIndex + i + 1, xGm.GetValue(inBase + d + i + 1));
                yGm.SetValue(outIndex + i + 2, xGm.GetValue(inBase + d + i + 2));
                yGm.SetValue(outIndex + i + 3, xGm.GetValue(inBase + d + i + 3));
            }
            for (; i < run; ++i) {
                yGm.SetValue(outIndex + i, xGm.GetValue(inBase + d + i));
            }
            outIndex += run - 1;
        }
    }



private:
    static constexpr uint32_t elemsPer32B = 32 / sizeof(DT_X);
    static constexpr uint32_t tileElems = (sizeof(DT_X) == 2) ? 22528 : 8192;
    static constexpr uint32_t tc5MaxPixels = 44;
    static constexpr uint32_t tc5Depth = 65;
    static constexpr uint32_t tc5GatherMaxOffsets = tc5MaxPixels * tc5Depth;
    static constexpr uint32_t tc5GatherStrideOffsets = ((tc5GatherMaxOffsets + 7) / 8) * 8;
    static constexpr uint32_t tc5GatherWorkOffsets = tc5GatherStrideOffsets;
    AscendC::TPipe pipe;
    uint32_t allocElems;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t outputLength;
    uint32_t height;
    uint32_t width;
    uint32_t depth;
    uint32_t outBatch;
    uint32_t outHeight;
    uint32_t outWidth;
    uint32_t blockSize;
    uint32_t cropTop;
    uint32_t cropBottom;
    uint32_t cropLeft;
    uint32_t cropRight;
    uint32_t blockDim;
    bool depthAligned;
};

template <typename DT_X>
class KernelBatchToSpaceDepth5Gather {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling) {
        xGm.SetGlobalBuffer((__gm__ DT_X *)x, tiling.inputLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, tiling.outputLength);
        height = tiling.height;
        width = tiling.width;
        outBatch = tiling.outBatch;
        outHeight = tiling.outHeight;
        outWidth = tiling.outWidth;
        cropTop = tiling.cropTop;
        cropLeft = tiling.cropLeft;
        blockDim = tiling.blockDim;
        pipe.InitBuffer(inQueue, 2, smallTileElems * sizeof(DT_X));
        pipe.InitBuffer(outQueue, 2, smallTileElems * sizeof(DT_X));
        pipe.InitBuffer(offsetBuf, smallTileElems * sizeof(int32_t));
    }

    __aicore__ inline void Process() {
        uint32_t core = AscendC::GetBlockIdx();
        uint32_t totalRows = outBatch * outHeight;
        uint32_t rowsPerCore = (totalRows + blockDim - 1) / blockDim;
        uint32_t rowStart = core * rowsPerCore;
        uint32_t rowEnd = rowStart + rowsPerCore;
        if (rowEnd > totalRows) rowEnd = totalRows;
        if (rowStart >= rowEnd) return;

        uint32_t iwStart = cropLeft >> 1;
        uint32_t iwEnd = (cropLeft + outWidth - 1) >> 1;
        uint32_t lanePixels = iwEnd - iwStart + 1;
        uint32_t laneBytes = lanePixels * depth5Bytes;
        uint32_t paddedLaneElems =
            ((laneBytes + 31) / 32) * elemsPer32B;
        uint32_t rowElems = outWidth * depth5;
        uint32_t rowBytes = rowElems * sizeof(DT_X);

        AscendC::LocalTensor<int32_t> offsets = offsetBuf.Get<int32_t>();
        for (uint32_t ow = 0; ow < outWidth; ++ow) {
            uint32_t paddedW = cropLeft + ow;
            uint32_t bw = paddedW & 1;
            uint32_t iw = (paddedW >> 1) - iwStart;
            uint32_t pixelBaseBytes =
                bw * paddedLaneElems * sizeof(DT_X) + iw * depth5Bytes;
            uint32_t offsetBase = ow * depth5;
            offsets.SetValue(offsetBase, pixelBaseBytes);
            offsets.SetValue(
                offsetBase + 1, pixelBaseBytes + sizeof(DT_X));
            offsets.SetValue(
                offsetBase + 2, pixelBaseBytes + 2 * sizeof(DT_X));
            offsets.SetValue(
                offsetBase + 3, pixelBaseBytes + 3 * sizeof(DT_X));
            offsets.SetValue(
                offsetBase + 4, pixelBaseBytes + 4 * sizeof(DT_X));
        }

        AscendC::DataCopyPadParams padP{true, 0, 0, 0};
        AscendC::DataCopyParams readP{
            1, static_cast<uint16_t>(laneBytes), 0, 0
        };
        AscendC::DataCopyParams writeP{
            1, static_cast<uint16_t>(rowBytes), 0, 0
        };

        for (uint32_t row = rowStart; row < rowEnd; ++row) {
            uint32_t oh = row % outHeight;
            uint32_t ob = row / outHeight;
            uint32_t paddedH = oh + cropTop;
            uint32_t bh = paddedH & 1;
            uint32_t ih = paddedH >> 1;
            uint32_t inBatch0 = (bh << 1) * outBatch + ob;
            uint32_t inBatch1 = inBatch0 + outBatch;
            uint32_t inBase0 =
                ((inBatch0 * height + ih) * width + iwStart) * depth5;
            uint32_t inBase1 =
                ((inBatch1 * height + ih) * width + iwStart) * depth5;
            uint32_t outBase = row * rowElems;

            AscendC::LocalTensor<DT_X> inAlloc = inQueue.AllocTensor<DT_X>();
            AscendC::DataCopyPad(inAlloc, xGm[inBase0], readP, padP);
            AscendC::DataCopyPad(
                inAlloc[paddedLaneElems], xGm[inBase1], readP, padP);
            inQueue.EnQue(inAlloc);
            AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();

            AscendC::LocalTensor<DT_X> outAlloc = outQueue.AllocTensor<DT_X>();
            AscendC::Gather(
                outAlloc, inDe, offsets.template ReinterpretCast<uint32_t>(),
                0, rowElems);
            outQueue.EnQue(outAlloc);
            inQueue.FreeTensor(inDe);

            AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
            if ((rowBytes & 31) == 0) {
                AscendC::DataCopy(yGm[outBase], finalLocal, rowElems);
            } else {
                AscendC::DataCopyPad(yGm[outBase], finalLocal, writeP);
            }
            outQueue.FreeTensor(finalLocal);
        }
    }

private:
    static constexpr uint32_t depth5 = 5;
    static constexpr uint32_t depth5Bytes = depth5 * sizeof(DT_X);
    static constexpr uint32_t elemsPer32B = 32 / sizeof(DT_X);
    static constexpr uint32_t smallTileElems = 4096;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> offsetBuf;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t height;
    uint32_t width;
    uint32_t outBatch;
    uint32_t outHeight;
    uint32_t outWidth;
    uint32_t cropTop;
    uint32_t cropLeft;
    uint32_t blockDim;
};

template <typename DT_X>
class KernelBatchToSpaceSmallAligned {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData &tiling) {
        xGm.SetGlobalBuffer((__gm__ DT_X *)x, tiling.inputLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, tiling.outputLength);
        outputLength = tiling.outputLength;
        height = tiling.height;
        width = tiling.width;
        depth = tiling.depth;
        outBatch = tiling.outBatch;
        outHeight = tiling.outHeight;
        outWidth = tiling.outWidth;
        cropTop = tiling.cropTop;
        cropLeft = tiling.cropLeft;
        blockDim = tiling.blockDim;
        pipe.InitBuffer(inQueue, 2, smallTileElems * sizeof(DT_X));
        pipe.InitBuffer(outQueue, 2, smallTileElems * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        uint32_t blockLen32B = depth / elemsPer32B;
        uint32_t core = AscendC::GetBlockIdx();
        uint32_t rowElems = outWidth * depth;
        uint32_t totalRows = (outputLength / depth) / outWidth;
        uint32_t rowChunk = (totalRows + blockDim - 1) / blockDim;
        uint32_t rowStart = core * rowChunk;
        uint32_t rowEnd = rowStart + rowChunk;
        if (rowEnd > totalRows) rowEnd = totalRows;
        if (rowStart >= rowEnd) return;

        uint32_t maxBatchRows = smallTileElems / rowElems;
        if (maxBatchRows > 64) maxBatchRows = 64;
        if (maxBatchRows < 2) maxBatchRows = 2;
        maxBatchRows &= ~1u;

        bool cropLeftEven = ((cropLeft & 1) == 0);
        uint32_t bw0IwStart =
            cropLeftEven ? (cropLeft >> 1) : ((cropLeft + 1) >> 1);
        uint32_t bw1IwStart = cropLeft >> 1;
        uint32_t w0 = cropLeftEven ? ((outWidth + 1) >> 1) : (outWidth >> 1);
        uint32_t w1 = cropLeftEven ? (outWidth >> 1) : ((outWidth + 1) >> 1);
        uint32_t inputRowElems = width * depth;

        uint32_t row = rowStart;
        while (row < rowEnd) {
            uint32_t batchRows = rowEnd - row;
            if (batchRows > maxBatchRows) batchRows = maxBatchRows;

            uint32_t ohStart = row % outHeight;
            uint32_t ob = row / outHeight;
            uint32_t rowsInThisOb = outHeight - ohStart;
            if (batchRows > rowsInThisOb) batchRows = rowsInThisOb;
            if (batchRows < 2) batchRows = 1;

            uint32_t outBase = (ob * outHeight + ohStart) * rowElems;
            uint32_t paddedH0 = ohStart + cropTop;
            bool firstIsBh0 = ((paddedH0 & 1) == 0);
            uint32_t nBh0 = firstIsBh0 ? ((batchRows + 1) >> 1) : (batchRows >> 1);
            uint32_t nBh1 = batchRows - nBh0;
            uint32_t ih0Start =
                firstIsBh0 ? (paddedH0 >> 1) : ((paddedH0 + 1) >> 1);
            uint32_t ih1Start =
                firstIsBh0 ? ((paddedH0 + 1) >> 1) : (paddedH0 >> 1);

            AscendC::LocalTensor<DT_X> outLocal = outQueue.AllocTensor<DT_X>();
            for (uint32_t bh = 0; bh < 2; ++bh) {
                uint32_t nRows = (bh == 0) ? nBh0 : nBh1;
                if (nRows == 0) continue;
                uint32_t ihStart = (bh == 0) ? ih0Start : ih1Start;

                AscendC::LocalTensor<DT_X> inAll = inQueue.AllocTensor<DT_X>();
                for (uint32_t bwIdx = 0; bwIdx < 2; ++bwIdx) {
                    uint32_t curW = (bwIdx == 0) ? w0 : w1;
                    if (curW == 0) continue;
                    uint32_t iwStart = (bwIdx == 0) ? bw0IwStart : bw1IwStart;
                    uint32_t inBatch = (bh * 2 + bwIdx) * outBatch + ob;
                    uint32_t gmBase =
                        ((inBatch * height + ihStart) * width + iwStart) * depth;
                    uint32_t readLen32B = curW * blockLen32B;
                    uint32_t readGap32B =
                        (inputRowElems - curW * depth) / elemsPer32B;
                    uint32_t offsetInAll =
                        (bwIdx == 0) ? 0 : (nRows * w0 * depth);

                    if (nRows == 1) {
                        AscendC::DataCopy(
                            inAll[offsetInAll], xGm[gmBase], curW * depth);
                    } else {
                        AscendC::DataCopyParams readP;
                        readP.blockCount = static_cast<uint16_t>(nRows);
                        readP.blockLen = static_cast<uint16_t>(readLen32B);
                        readP.srcStride = static_cast<uint16_t>(readGap32B);
                        readP.dstStride = 0;
                        AscendC::DataCopy(inAll[offsetInAll], xGm[gmBase], readP);
                    }
                }
                inQueue.EnQue(inAll);
                AscendC::LocalTensor<DT_X> inDe = inQueue.DeQue<DT_X>();

                for (uint32_t bwIdx = 0; bwIdx < 2; ++bwIdx) {
                    uint32_t curW = (bwIdx == 0) ? w0 : w1;
                    if (curW == 0) continue;
                    uint32_t offsetInAll =
                        (bwIdx == 0) ? 0 : (nRows * w0 * depth);
                    AscendC::DataCopyParams zipP;
                    zipP.blockCount = static_cast<uint16_t>(curW);
                    zipP.blockLen = blockLen32B;
                    zipP.srcStride = 0;
                    zipP.dstStride = blockLen32B;

                    for (uint32_t r = 0; r < nRows; ++r) {
                        uint32_t outRowIdx;
                        if (bh == 0) {
                            outRowIdx = firstIsBh0 ? (2 * r) : (2 * r + 1);
                        } else {
                            outRowIdx = firstIsBh0 ? (2 * r + 1) : (2 * r);
                        }
                        uint32_t rowOff = outRowIdx * rowElems;
                        uint32_t srcOff = offsetInAll + r * curW * depth;
                        bool isEvenBw = (bwIdx == 0);
                        uint32_t dstOff =
                            (cropLeftEven == isEvenBw) ? rowOff : (rowOff + depth);
                        AscendC::DataCopy(outLocal[dstOff], inDe[srcOff], zipP);
                    }
                }
                inQueue.FreeTensor(inDe);
            }

            outQueue.EnQue(outLocal);
            AscendC::LocalTensor<DT_X> finalLocal = outQueue.DeQue<DT_X>();
            AscendC::DataCopy(yGm[outBase], finalLocal, batchRows * rowElems);
            outQueue.FreeTensor(finalLocal);
            row += batchRows;
        }
    }

private:
    static constexpr uint32_t elemsPer32B = 32 / sizeof(DT_X);
    static constexpr uint32_t smallTileElems = 8192;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outQueue;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t outputLength;
    uint32_t height;
    uint32_t width;
    uint32_t depth;
    uint32_t outBatch;
    uint32_t outHeight;
    uint32_t outWidth;
    uint32_t cropTop;
    uint32_t cropLeft;
    uint32_t blockDim;
};

template <typename DT_X>
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingData, tiling_data, tiling);
    if (tiling_data.blockSize == 2 && tiling_data.depth == 5 &&
        tiling_data.outWidth * 5 + 64 <= 4096) {
        KernelBatchToSpaceDepth5Gather<DT_X> depth5Op;
        depth5Op.Init(x, y, tiling_data);
        depth5Op.Process();
        return;
    }
    if (sizeof(DT_X) == 2 && tiling_data.blockSize == 2 &&
        tiling_data.outputLength > 32768 && tiling_data.outputLength <= 65536 &&
        (tiling_data.depth % 16) == 0 && tiling_data.outWidth <= 16 &&
        tiling_data.outWidth * tiling_data.depth * 2 <= 8192) {
        KernelBatchToSpaceSmallAligned<DT_X> smallOp;
        smallOp.Init(x, y, tiling_data);
        smallOp.Process();
        return;
    }
    KernelBatchToSpace<DT_X> op;
    op.Init(x, y, tiling_data);
    op.Process();
}

