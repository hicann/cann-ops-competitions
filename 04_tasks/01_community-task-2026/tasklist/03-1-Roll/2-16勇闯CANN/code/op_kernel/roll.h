/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file roll.h
 * \brief Roll kernel implementation - unified per-region copy with double buffer
 */

#ifndef ROLL_H
#define ROLL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "roll_tiling_data.h"
#include "roll_tiling_key.h"

namespace NsRoll {

using namespace AscendC;

template <typename T>
class Roll {
public:
    __aicore__ inline Roll(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const RollTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyChunk(int64_t srcOffset, int64_t dstOffset, int64_t numElements);
    __aicore__ inline void CopyChunkSB(int64_t srcOffset, int64_t dstOffset, int64_t numElements);
    __aicore__ inline void CopyChunkDB(int64_t srcOffset, int64_t dstOffset, int64_t numElements);
    __aicore__ inline void ProcessMode0UBReorder();
    __aicore__ inline void ProcessMode0UBReorderGather();
    __aicore__ inline void ProcessMode0TailSmall();
    __aicore__ inline void ProcessMode0TinyWholeGather();
    __aicore__ inline void GatherBatch(int64_t g, LocalTensor<T> buf, int64_t n,
                                       int64_t mainLen, int64_t wrapLen, int64_t groupSize,
                                       int64_t alignedMainElems);
    __aicore__ inline void StoreBatch(int64_t g, LocalTensor<T> buf, int64_t n,
                                      int64_t mainLen, int64_t wrapLen, int64_t groupSize,
                                      int64_t alignedMainElems);
    __aicore__ inline void ProcessMultiDim();
    __aicore__ inline void ProcessMultiDimUBReorder();
    __aicore__ inline void ProcessMultiDim2DReorder();
    __aicore__ inline void ProcessMultiDimNDReorder();
    __aicore__ inline void ProcessPairedRowTile();
    __aicore__ inline void ProcessMultiDimNDUnit();
    __aicore__ inline void ProcessMultiDimPlaneGather();
    __aicore__ inline void ProcessMultiDimTileGather();
    __aicore__ inline void ProcessGatherSupplement();
    __aicore__ inline void ProcessChunkedTileGather();
    __aicore__ inline void ProcessFusedGather();
    __aicore__ inline int64_t ComputeSrcBaseOffset(int64_t outPos);
    __aicore__ inline void SyncMTE2ToMTE3();
    __aicore__ inline void SyncMTE3ToMTE2();
    __aicore__ inline void SyncMTE2ToV();
    __aicore__ inline void SyncVToMTE3();
    __aicore__ inline bool IsPairedRowTile() const;

    TPipe* pipe_;
    TQue<QuePosition::VECIN, 2> inQueue;      // unified UB queue: depth=2 for DB, depth=1 for SB
    TQue<QuePosition::VECOUT, 2> outQueue_;    // Gather output queue for pipelined special paths
    TBuf<QuePosition::VECCALC> calcBufA_;     // multi-dim UBReorder / special path buffer
    TQue<QuePosition::VECCALC, 2> calcQueue_; // MODE0 UBReorder double-buffer queue
    TBuf<QuePosition::VECCALC> offsetBuf_;    // Gather srcOffset array for MODE0 Gather path
    AscendC::DataCopyExtParams copyParams;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> workspaceGm;

    int64_t needCoreNum;
    int64_t numEachCore;
    int64_t lastCoreNum;
    int64_t inNum;        // dimSize for the rolled dim
    int64_t afterNum;     // stride after the rolled dim
    int64_t shift;        // normalized shift in [0, inNum)
    int64_t ubTensorSize; // UB capacity in elements
    int64_t tilingMode;   // 0=MODE0, 2=MODE2
    int64_t subMode;      // Unified MODE0/2 structural sub-mode selector

    // MODE2: multi-dim fields
    int64_t dimNum;
    int64_t shapes[8];
    int64_t strides[8];
    int64_t shifts[8];

    // Multi-dim UBReorder fields
    int64_t innerShiftIdx;
    int64_t rowMultiplier;
    int64_t continuousRows;
    int64_t lenL;
    int64_t lenR;
    int64_t outerBlockSize;
    int64_t subBlockSize;
    int64_t totalElements;
    int64_t remCoreNum;       // first N cores get one extra alignUnit

    // Precomputed move parameters (filled in tiling, consumed in kernel)
    int64_t moveCount;
    int64_t moveRowCount[4];
    int64_t moveBlockLen[4];
    int64_t moveSrcStride[4];
    int64_t moveDstStride[4];
    int64_t moveSrcOffset[4];
    int64_t moveDstOffset[4];

    // NDReorder outer-level precomputed parameters (avoid kernel recomputation)
    int64_t regionOuterLen[4];
    int64_t regionSrcOuterOff[4];
    int64_t regionDstOuterOff[4];
    int64_t workspaceOffset;
    int64_t workspaceBytes;

    int64_t coreId;
};

template <typename T>
__aicore__ inline void Roll<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const RollTilingData* tilingData, TPipe* pipe)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    this->pipe_ = pipe;
    this->needCoreNum = tilingData->needCoreNum;
    this->numEachCore = tilingData->numEachCore;
    this->lastCoreNum = tilingData->lastCoreNum;
    this->inNum = tilingData->inNum;
    this->afterNum = tilingData->afterNum;
    this->shift = tilingData->shift;
    this->ubTensorSize = tilingData->ubTensorSize;
    this->tilingMode = tilingData->tilingMode;
    this->subMode = tilingData->subMode;
    this->dimNum = tilingData->dimNum;
    for (int i = 0; i < 8; i++) {
        this->shapes[i] = tilingData->shapes[i];
        this->strides[i] = tilingData->strides[i];
        this->shifts[i] = tilingData->shifts[i];
    }
    this->innerShiftIdx = tilingData->innerShiftIdx;
    this->rowMultiplier = tilingData->rowMultiplier;
    this->continuousRows = tilingData->continuousRows;
    this->lenL = tilingData->lenL;
    this->lenR = tilingData->lenR;
    this->outerBlockSize = tilingData->outerBlockSize;
    this->subBlockSize = tilingData->subBlockSize;
    this->totalElements = tilingData->totalElements;
    this->remCoreNum = tilingData->remCoreNum;

    // Copy precomputed move parameters
    this->moveCount = tilingData->moveCount;
    for (int i = 0; i < 4; i++) {
        this->moveRowCount[i] = tilingData->moveRowCount[i];
        this->moveBlockLen[i] = tilingData->moveBlockLen[i];
        this->moveSrcStride[i] = tilingData->moveSrcStride[i];
        this->moveDstStride[i] = tilingData->moveDstStride[i];
        this->moveSrcOffset[i] = tilingData->moveSrcOffset[i];
        this->moveDstOffset[i] = tilingData->moveDstOffset[i];
        this->regionOuterLen[i] = tilingData->regionOuterLen[i];
        this->regionSrcOuterOff[i] = tilingData->regionSrcOuterOff[i];
        this->regionDstOuterOff[i] = tilingData->regionDstOuterOff[i];
    }
    this->workspaceOffset = tilingData->useMultiDimReorder_;
    this->workspaceBytes = tilingData->useGatherReorder_;
    coreId = GetBlockIdx();
    xGm.SetGlobalBuffer((__gm__ T*)x);
    yGm.SetGlobalBuffer((__gm__ T*)y);
    workspaceGm.SetGlobalBuffer((__gm__ T*)((__gm__ uint8_t*)workspace + this->workspaceOffset));

    if (coreId < needCoreNum) {
        if (tilingMode == 0 && subMode == 8) {
            this->pipe_->InitBuffer(calcBufA_, 2 * ubTensorSize * sizeof(T));
            this->pipe_->InitBuffer(offsetBuf_, totalElements * sizeof(uint32_t));
        } else if (tilingMode == 0 && (subMode == 1 || subMode == 2)) {
            // MODE0 UBReorder/Gather: TQue double-buffer for implicit MTE2/MTE3 overlap
            this->pipe_->InitBuffer(calcQueue_, 2, ubTensorSize * sizeof(T));
            if (subMode == 2) {
                this->pipe_->InitBuffer(offsetBuf_, 32 * inNum * sizeof(uint32_t));
            }
        } else if (tilingMode == 2 && subMode == 1) {
            // Multi-dim UBReorder: use calcBuf for batch gather/scatter
            this->pipe_->InitBuffer(calcBufA_, 2 * ubTensorSize * sizeof(T));
        } else if (tilingMode == 2 && subMode == 5) {
            this->pipe_->InitBuffer(inQueue, 2, ubTensorSize * sizeof(T));
            this->pipe_->InitBuffer(outQueue_, 2, ubTensorSize * sizeof(T));
            this->pipe_->InitBuffer(offsetBuf_, ubTensorSize * sizeof(uint32_t));
        } else if (tilingMode == 2 && subMode == 6) {
            this->pipe_->InitBuffer(inQueue, 2, ubTensorSize * sizeof(T));
            this->pipe_->InitBuffer(outQueue_, 2, ubTensorSize * sizeof(T));
            this->pipe_->InitBuffer(offsetBuf_, ubTensorSize * sizeof(uint32_t));
        } else if (tilingMode == 2 && subMode == 7) {
            this->pipe_->InitBuffer(inQueue, 2, ubTensorSize * sizeof(T));
            this->pipe_->InitBuffer(outQueue_, 2, ubTensorSize * sizeof(T));
            this->pipe_->InitBuffer(offsetBuf_, outerBlockSize * sizeof(uint32_t));
        } else if (tilingMode == 2 && subMode == 13) {
            this->pipe_->InitBuffer(calcBufA_, ubTensorSize * sizeof(T));
        } else if (tilingMode == 2 && subMode == 11 && subBlockSize > 0 && rowMultiplier > 1) {
            this->pipe_->InitBuffer(calcBufA_, ubTensorSize * sizeof(T));
        } else if (tilingMode == 2 && subMode == 11) {
            this->pipe_->InitBuffer(calcQueue_, 2, ubTensorSize * sizeof(T));
            this->pipe_->InitBuffer(offsetBuf_, outerBlockSize * sizeof(uint32_t));
        } else if (tilingMode == 2 && subMode == 12) {
            // Single buffer keeps the paired row tile large.
            this->pipe_->InitBuffer(inQueue, 1, ubTensorSize * sizeof(T));
        } else if (tilingMode == 2 && subMode == 10) {
            this->pipe_->InitBuffer(inQueue, 2, ubTensorSize * sizeof(T));
            this->pipe_->InitBuffer(outQueue_, 2, ubTensorSize * sizeof(T));
            this->pipe_->InitBuffer(offsetBuf_, outerBlockSize * sizeof(uint32_t));
        } else {
            // MODE0/2: depth=2 for double-buffer ping-pong.
            this->pipe_->InitBuffer(inQueue, 2, ubTensorSize * sizeof(T));
        }
    }
    this->copyParams.blockCount = 1;
    this->copyParams.blockLen = 0;
    this->copyParams.srcStride = 0;
    this->copyParams.dstStride = 0;
}

template <typename T>
__aicore__ inline void Roll<T>::GatherBatch(int64_t g, LocalTensor<T> buf, int64_t n,
                                            int64_t mainLen, int64_t wrapLen, int64_t groupSize,
                                            int64_t alignedMainElems)
{
    if (n <= 0) return;
    auto bufMain = buf;
    auto bufWrap = buf[n * alignedMainElems];
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

    AscendC::DataCopyExtParams mainGather;
    mainGather.blockCount = static_cast<uint32_t>(n);
    mainGather.blockLen = static_cast<uint32_t>(mainLen * sizeof(T));
    mainGather.srcStride = static_cast<uint32_t>((groupSize - mainLen) * sizeof(T));
    mainGather.dstStride = 0;
    DataCopyPad(bufMain, xGm[g + wrapLen], mainGather, padParams);

    AscendC::DataCopyExtParams wrapGather;
    wrapGather.blockCount = static_cast<uint32_t>(n);
    wrapGather.blockLen = static_cast<uint32_t>(wrapLen * sizeof(T));
    wrapGather.srcStride = static_cast<uint32_t>((groupSize - wrapLen) * sizeof(T));
    wrapGather.dstStride = 0;
    DataCopyPad(bufWrap, xGm[g], wrapGather, padParams);
}

template <typename T>
__aicore__ inline void Roll<T>::StoreBatch(int64_t g, LocalTensor<T> buf, int64_t n,
                                           int64_t mainLen, int64_t wrapLen, int64_t groupSize,
                                           int64_t alignedMainElems)
{
    if (n <= 0) return;
    auto bufMain = buf;
    auto bufWrap = buf[n * alignedMainElems];

    AscendC::DataCopyExtParams mainStore;
    mainStore.blockCount = static_cast<uint32_t>(n);
    mainStore.blockLen = static_cast<uint32_t>(mainLen * sizeof(T));
    mainStore.srcStride = 0;
    mainStore.dstStride = static_cast<uint32_t>((groupSize - mainLen) * sizeof(T));
    DataCopyPad(yGm[g], bufMain, mainStore);

    AscendC::DataCopyExtParams wrapStore;
    wrapStore.blockCount = static_cast<uint32_t>(n);
    wrapStore.blockLen = static_cast<uint32_t>(wrapLen * sizeof(T));
    wrapStore.srcStride = 0;
    wrapStore.dstStride = static_cast<uint32_t>((groupSize - wrapLen) * sizeof(T));
    DataCopyPad(yGm[g + mainLen], bufWrap, wrapStore);
}

template <typename T>
__aicore__ inline void Roll<T>::ProcessMode0UBReorder()
{
    int64_t numElements;
    int64_t coreStart;
    int64_t groupSize = inNum * afterNum;
    if (coreId < remCoreNum) {
        coreStart = coreId * (numEachCore + groupSize);
        numElements = numEachCore + groupSize;
    } else {
        coreStart = remCoreNum * (numEachCore + groupSize) + (coreId - remCoreNum) * numEachCore;
        numElements = numEachCore;
    }
    if (numElements <= 0) {
        return;
    }
    int64_t coreEnd = coreStart + numElements;
    int64_t wrapLen = (inNum - shift) * afterNum;
    int64_t mainLen = shift * afterNum;

    // 32B alignment
    int64_t alignedMainBytes = ((mainLen * static_cast<int64_t>(sizeof(T)) + 31) / 32) * 32;
    int64_t alignedWrapBytes = ((wrapLen * static_cast<int64_t>(sizeof(T)) + 31) / 32) * 32;
    int64_t alignedMainElems = alignedMainBytes / sizeof(T);
    int64_t alignedWrapElems = alignedWrapBytes / sizeof(T);
    int64_t elemsPerBatch = alignedMainElems + alignedWrapElems;

    constexpr int64_t MAX_BATCH_GROUPS = 256;
    int64_t batchSize = ubTensorSize / elemsPerBatch;
    if (batchSize > MAX_BATCH_GROUPS) {
        batchSize = MAX_BATCH_GROUPS;
    }
    if (batchSize < 1) {
        batchSize = 1;
    }

    int64_t g = coreStart;
    if (g >= coreEnd) {
        return;
    }

    // First batch size (may be smaller)
    int64_t firstBatch = (coreEnd - g) / groupSize;
    if (firstBatch > batchSize) {
        firstBatch = batchSize;
    }
    if (firstBatch < 1) {
        firstBatch = 1;
    }

    // Warm-up: gather first batch into queue (EnQue implicitly syncs MTE2)
    auto bufPrev = calcQueue_.AllocTensor<T>();
    GatherBatch(g, bufPrev, firstBatch, mainLen, wrapLen, groupSize, alignedMainElems);
    calcQueue_.EnQue(bufPrev);
    g += firstBatch * groupSize;
    int64_t currBatch = firstBatch;

    // Main loop: store previous batch + gather next batch, overlapped via queue
    while (g < coreEnd) {
        int64_t nextBatch = (coreEnd - g) / groupSize;
        if (nextBatch > batchSize) {
            nextBatch = batchSize;
        }
        if (nextBatch < 1) {
            nextBatch = 1;
        }

        // Gather next batch into new buffer (MTE2, background)
        auto bufNext = calcQueue_.AllocTensor<T>();
        GatherBatch(g, bufNext, nextBatch, mainLen, wrapLen, groupSize, alignedMainElems);

        // Store previous batch from ready buffer (MTE3, foreground)
        auto bufReady = calcQueue_.DeQue<T>();
        int64_t storeStart = g - currBatch * groupSize;
        StoreBatch(storeStart, bufReady, currBatch, mainLen, wrapLen, groupSize, alignedMainElems);
        calcQueue_.FreeTensor(bufReady);

        calcQueue_.EnQue(bufNext);
        g += nextBatch * groupSize;
        currBatch = nextBatch;
    }

    // Drain: store last batch
    auto bufLast = calcQueue_.DeQue<T>();
    int64_t lastStart = g - currBatch * groupSize;
    StoreBatch(lastStart, bufLast, currBatch, mainLen, wrapLen, groupSize, alignedMainElems);
    calcQueue_.FreeTensor(bufLast);
}

template <typename T>
__aicore__ inline void Roll<T>::ProcessMode0UBReorderGather()
{
    int64_t groupSize = inNum * afterNum;
    int64_t alignGroups = 32 / static_cast<int64_t>(sizeof(T));
    if (alignGroups < 1) alignGroups = 1;
    int64_t totalGroups = totalElements / groupSize;
    int64_t totalAlignGroups = (totalGroups + alignGroups - 1) / alignGroups;
    int64_t baseAlignGroups = totalAlignGroups / needCoreNum;
    int64_t remAlignGroups = totalAlignGroups % needCoreNum;
    int64_t myAlignGroups = baseAlignGroups + (coreId < remAlignGroups ? 1 : 0);
    if (myAlignGroups <= 0) {
        return;
    }
    int64_t alignGroupStart = coreId < remAlignGroups
        ? coreId * (baseAlignGroups + 1)
        : remAlignGroups * (baseAlignGroups + 1) + (coreId - remAlignGroups) * baseAlignGroups;
    int64_t groupStart = alignGroupStart * alignGroups;
    int64_t groupCount = myAlignGroups * alignGroups;
    if (groupStart + groupCount > totalGroups) {
        groupCount = totalGroups - groupStart;
    }
    if (groupCount <= 0) {
        return;
    }
    int64_t coreStart = groupStart * groupSize;
    int64_t coreEnd = coreStart + groupCount * groupSize;
    int64_t wrapLen = (inNum - shift) * afterNum;

    // Each queue buffer holds input+output for batchSize groups.
    // ubTensorSize = 2 * batchSize * inNum  (set in tiling)
    int64_t batchSize = ubTensorSize / (2 * inNum);
    batchSize = (batchSize / alignGroups) * alignGroups;
    if (batchSize < alignGroups) batchSize = alignGroups;
    const int64_t SUPER_GROUP = 32;

    // Precompute srcOffset pattern once: for SUPER_GROUP groups, rotate by wrapLen within each group.
    // offsetTensor stores BYTE offsets relative to the start of each super-group,
    // so the same table can be reused across all super-groups by varying byteOff.
    auto offsetTensor = offsetBuf_.Get<uint32_t>();
    int inNum_i = static_cast<int>(inNum);
    int wrap_i = inNum_i - static_cast<int>(shift);
    int ts = static_cast<int>(sizeof(T));
    for (int g = 0; g < static_cast<int>(SUPER_GROUP); g++) {
        int base = g * inNum_i;
        offsetTensor.SetValue(base, static_cast<uint32_t>((base + wrap_i) * ts));
        for (int e = 1; e < inNum_i; e++) {
            offsetTensor.SetValue(base + e, static_cast<uint32_t>((base + e - 1) * ts));
        }
    }

    // Bulk DMA copy params (single contiguous block per batch)
    AscendC::DataCopyExtParams bulkCopy;
    bulkCopy.blockCount = 1;
    bulkCopy.blockLen = 0;  // set per-batch
    bulkCopy.srcStride = 0;
    bulkCopy.dstStride = 0;
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

    int64_t g = coreStart;
    if (g >= coreEnd) {
        return;
    }

    // First batch size (may be smaller)
    int64_t firstBatch = (coreEnd - g) / groupSize;
    if (firstBatch > batchSize) firstBatch = batchSize;
    if (firstBatch < 1) firstBatch = 1;

    // Warm-up: gather first batch into queue
    auto bufPrev = calcQueue_.AllocTensor<T>();
    auto bufPrevIn = bufPrev;
    auto bufPrevOut = bufPrev[batchSize * inNum];
    bulkCopy.blockLen = static_cast<uint32_t>(firstBatch * groupSize * sizeof(T));
    DataCopyPad(bufPrevIn, xGm[g], bulkCopy, padParams);
    calcQueue_.EnQue(bufPrev);
    g += firstBatch * groupSize;
    int64_t currBatch = firstBatch;

    // Main loop: process previous batch + gather next batch, overlapped via queue
    while (g < coreEnd) {
        int64_t nextBatch = (coreEnd - g) / groupSize;
        if (nextBatch > batchSize) nextBatch = batchSize;
        if (nextBatch < 1) nextBatch = 1;

        // Gather next batch (MTE2)
        auto bufNext = calcQueue_.AllocTensor<T>();
        auto bufNextIn = bufNext;
        auto bufNextOut = bufNext[batchSize * inNum];
        bulkCopy.blockLen = static_cast<uint32_t>(nextBatch * groupSize * sizeof(T));
        DataCopyPad(bufNextIn, xGm[g], bulkCopy, padParams);

        // Process previous batch
        auto bufReady = calcQueue_.DeQue<T>();
        auto bufReadyIn = bufReady;
        auto bufReadyOut = bufReady[batchSize * inNum];

        SyncMTE2ToV();
        int64_t fullSG = currBatch / SUPER_GROUP;
        int64_t remGroups = currBatch % SUPER_GROUP;
        int64_t elemOff = 0;
        int64_t sgStep = SUPER_GROUP * inNum;
        for (int64_t sg = 0; sg < fullSG; sg++) {
            uint32_t byteOff = static_cast<uint32_t>(elemOff * sizeof(T));
            auto dstPtr = bufReadyOut[elemOff];
            auto srcPtr = bufReadyIn;
            Gather(dstPtr, srcPtr, offsetTensor, byteOff,
                   static_cast<uint32_t>(sgStep));
            elemOff += sgStep;
        }
        if (remGroups > 0) {
            uint32_t byteOff = static_cast<uint32_t>(elemOff * sizeof(T));
            auto dstPtr = bufReadyOut[elemOff];
            auto srcPtr = bufReadyIn;
            uint32_t count = static_cast<uint32_t>(remGroups * inNum);
            Gather(dstPtr, srcPtr, offsetTensor, byteOff, count);
        }
        SyncVToMTE3();

        // Store processed batch (MTE3)
        int64_t storeStart = g - currBatch * groupSize;
        bulkCopy.blockLen = static_cast<uint32_t>(currBatch * groupSize * sizeof(T));
        DataCopyPad(yGm[storeStart], bufReadyOut, bulkCopy);
        calcQueue_.FreeTensor(bufReady);

        calcQueue_.EnQue(bufNext);
        g += nextBatch * groupSize;
        currBatch = nextBatch;
    }

    // Drain: process and store last batch
    auto bufLast = calcQueue_.DeQue<T>();
    auto bufLastIn = bufLast;
    auto bufLastOut = bufLast[batchSize * inNum];

    SyncMTE2ToV();
    int64_t fullSG = currBatch / SUPER_GROUP;
    int64_t remGroups = currBatch % SUPER_GROUP;
    int64_t elemOff = 0;
    int64_t sgStep = SUPER_GROUP * inNum;
    for (int64_t sg = 0; sg < fullSG; sg++) {
        uint32_t byteOff = static_cast<uint32_t>(elemOff * sizeof(T));
        auto dstPtr = bufLastOut[elemOff];
        auto srcPtr = bufLastIn;
        Gather(dstPtr, srcPtr, offsetTensor, byteOff,
               static_cast<uint32_t>(sgStep));
        elemOff += sgStep;
    }
    if (remGroups > 0) {
        uint32_t byteOff = static_cast<uint32_t>(elemOff * sizeof(T));
        auto dstPtr = bufLastOut[elemOff];
        auto srcPtr = bufLastIn;
        uint32_t count = static_cast<uint32_t>(remGroups * inNum);
        Gather(dstPtr, srcPtr, offsetTensor, byteOff, count);
    }
    SyncVToMTE3();

    int64_t lastStart = g - currBatch * groupSize;
    bulkCopy.blockLen = static_cast<uint32_t>(currBatch * groupSize * sizeof(T));
    DataCopyPad(yGm[lastStart], bufLastOut, bulkCopy);
    calcQueue_.FreeTensor(bufLast);
}

template <typename T>
__aicore__ inline void Roll<T>::ProcessMode0TailSmall()
{
    int64_t numElements;
    int64_t coreStart;
    int64_t groupSize = inNum * afterNum;
    if (coreId < remCoreNum) {
        coreStart = coreId * (numEachCore + groupSize);
        numElements = numEachCore + groupSize;
    } else {
        coreStart = remCoreNum * (numEachCore + groupSize) + (coreId - remCoreNum) * numEachCore;
        numElements = numEachCore;
    }
    if (numElements <= 0) {
        return;
    }

    int64_t coreEnd = coreStart + numElements;
    if (groupSize <= 0 || shift <= 0 || shift >= inNum) {
        return;
    }

    int64_t leftLen = shift * afterNum;
    int64_t rightLen = (inNum - shift) * afterNum;
    int64_t outPos = coreStart;

    while (outPos < coreEnd) {
        int64_t groupBase = (outPos / groupSize) * groupSize;
        int64_t inGroup = outPos - groupBase;
        int64_t regionEnd = (inGroup < leftLen) ? leftLen : groupSize;
        int64_t chunkSize = regionEnd - inGroup;
        int64_t remain = coreEnd - outPos;
        if (chunkSize > remain) {
            chunkSize = remain;
        }

        int64_t srcPos;
        if (inGroup < leftLen) {
            srcPos = groupBase + rightLen + inGroup;
        } else {
            srcPos = groupBase + (inGroup - leftLen);
        }

        CopyChunk(srcPos, outPos, chunkSize);
        outPos += chunkSize;
    }
}

template <typename T>
__aicore__ inline void Roll<T>::ProcessMode0TinyWholeGather()
{
    if (coreId != 0 || totalElements <= 0 || shift <= 0 || shift >= inNum) {
        return;
    }

    int64_t groupSize = inNum * afterNum;
    if (groupSize <= 0 || totalElements > ubTensorSize) {
        return;
    }

    auto inputLocal = calcBufA_.Get<T>();
    auto outputLocal = inputLocal[ubTensorSize];
    auto offsetTensor = offsetBuf_.Get<uint32_t>();
    int ts = static_cast<int>(sizeof(T));

    for (int64_t outPos = 0; outPos < totalElements; ++outPos) {
        int64_t groupBase = (outPos / groupSize) * groupSize;
        int64_t rem = outPos - groupBase;
        int64_t outI = rem / afterNum;
        int64_t k = rem - outI * afterNum;
        int64_t srcI = outI - shift;
        if (srcI < 0) {
            srcI += inNum;
        }
        int64_t srcPos = groupBase + srcI * afterNum + k;
        offsetTensor.SetValue(outPos, static_cast<uint32_t>(srcPos * ts));
    }

    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(totalElements * static_cast<int64_t>(sizeof(T)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

    DataCopyPad(inputLocal, xGm[0], copyParams, padParams);
    SyncMTE2ToV();
    Gather(outputLocal, inputLocal, offsetTensor, static_cast<uint32_t>(0), static_cast<uint32_t>(totalElements));
    SyncVToMTE3();
    DataCopyPad(yGm[0], outputLocal, copyParams);
}

template <typename T>
__aicore__ inline void Roll<T>::SyncMTE2ToMTE3()
{
    event_t e = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(e);
    WaitFlag<HardEvent::MTE2_MTE3>(e);
}

template <typename T>
__aicore__ inline void Roll<T>::SyncMTE3ToMTE2()
{
    event_t e = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e);
    WaitFlag<HardEvent::MTE3_MTE2>(e);
}

template <typename T>
__aicore__ inline void Roll<T>::SyncMTE2ToV()
{
    event_t e = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e);
    WaitFlag<HardEvent::MTE2_V>(e);
}

template <typename T>
__aicore__ inline void Roll<T>::SyncVToMTE3()
{
    event_t e = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e);
    WaitFlag<HardEvent::V_MTE3>(e);
}

template <typename T>
__aicore__ inline void Roll<T>::CopyChunkSB(int64_t srcOffset, int64_t dstOffset, int64_t numElements)
{
    int64_t copied = 0;
    while (copied < numElements) {
        int64_t chunkSize = numElements - copied;
        if (chunkSize > ubTensorSize) {
            chunkSize = ubTensorSize;
        }

        auto tmpUB = inQueue.AllocTensor<T>();
        this->copyParams.blockLen = static_cast<uint32_t>(chunkSize * sizeof(T));
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(tmpUB, xGm[srcOffset + copied], this->copyParams, padParams);
        SyncMTE2ToMTE3();
        DataCopyPad(yGm[dstOffset + copied], tmpUB, this->copyParams);
        SyncMTE3ToMTE2();
        inQueue.FreeTensor(tmpUB);

        copied += chunkSize;
    }
}

template <typename T>
__aicore__ inline void Roll<T>::CopyChunkDB(int64_t srcOffset, int64_t dstOffset, int64_t numElements)
{
    int64_t copied = 0;
    int64_t chunkSize = numElements;
    if (chunkSize > ubTensorSize) {
        chunkSize = ubTensorSize;
    } else {
        // Total data fits in one ubTensorSize buffer.
        // Force split into 2 aligned chunks so the DB pipeline actually overlaps MTE2/MTE3.
        int64_t alignElems = 32 / static_cast<int64_t>(sizeof(T));
        if (alignElems < 1) alignElems = 1;
        chunkSize = (numElements / 2 / alignElems) * alignElems;
        if (chunkSize < alignElems) chunkSize = alignElems;
        if (chunkSize > ubTensorSize) chunkSize = ubTensorSize;
    }

    // Preheat: copy first chunk and enqueue it as ready
    this->copyParams.blockLen = static_cast<uint32_t>(chunkSize * sizeof(T));
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    auto ubFirst = inQueue.AllocTensor<T>();
    DataCopyPad(ubFirst, xGm[srcOffset], this->copyParams, padParams);
    SyncMTE2ToMTE3();
    inQueue.EnQue(ubFirst);

    AscendC::LocalTensor<T> ubNext;

    while (copied < numElements) {
        int64_t nextChunkSize = numElements - copied - chunkSize;
        if (nextChunkSize > ubTensorSize) {
            nextChunkSize = ubTensorSize;
        }

        // Background: start copying next chunk into a fresh buffer (MTE2)
        if (nextChunkSize > 0) {
            ubNext = inQueue.AllocTensor<T>();
            this->copyParams.blockLen = static_cast<uint32_t>(nextChunkSize * sizeof(T));
            DataCopyPad(ubNext, xGm[srcOffset + copied + chunkSize], this->copyParams, padParams);
        }

        // Foreground: dequeue ready buffer and copy out to GM (MTE3)
        auto ubCurrent = inQueue.DeQue<T>();
        this->copyParams.blockLen = static_cast<uint32_t>(chunkSize * sizeof(T));
        DataCopyPad(yGm[dstOffset + copied], ubCurrent, this->copyParams);
        SyncMTE3ToMTE2();
        inQueue.FreeTensor(ubCurrent);

        // Wait for background copy-in and enqueue the next ready buffer
        if (nextChunkSize > 0) {
            SyncMTE2ToMTE3();
            inQueue.EnQue(ubNext);
        }

        copied += chunkSize;
        chunkSize = nextChunkSize;
    }
}

template <typename T>
__aicore__ inline void Roll<T>::CopyChunk(int64_t srcOffset, int64_t dstOffset, int64_t numElements)
{
    // Enable double-buffer as long as total data exceeds 2KB.
    // CopyChunkDB internally splits into at least 2 chunks when numElements <= ubTensorSize.
    constexpr int64_t DB_MIN_BYTES = 1024;
    if ((tilingMode == 0 || tilingMode == 2) && numElements * static_cast<int64_t>(sizeof(T)) > DB_MIN_BYTES) {
        CopyChunkDB(srcOffset, dstOffset, numElements);
    } else {
        CopyChunkSB(srcOffset, dstOffset, numElements);
    }
}

template <typename T>
__aicore__ inline void Roll<T>::ProcessMultiDim()
{
    int64_t numElements = (coreId != needCoreNum - 1) ? numEachCore : lastCoreNum;
    if (numElements <= 0) {
        return;
    }
    int64_t coreStart = numEachCore * coreId;
    int64_t coreEnd = coreStart + numElements;

    // Decompose coreStart into multi-dimensional coordinates
    int64_t coords[8] = {0};
    int64_t tmp = coreStart;
    for (int i = 0; i < dimNum; i++) {
        coords[i] = tmp / strides[i];
        tmp %= strides[i];
    }

    int64_t outPos = coreStart;
    while (outPos < coreEnd) {
        // 1. Compute source position by reverse-mapping output coords to input coords
        int64_t srcPos = 0;
        for (int i = 0; i < dimNum; i++) {
            int64_t srcCoord = coords[i] - shifts[i];
            if (srcCoord < 0) {
                srcCoord += shapes[i];
            }
            srcPos += srcCoord * strides[i];
        }

        // 2. Compute max contiguous chunk where src is also contiguous.
        //    Continuity breaks when crossing the split point of any shifted dim.
        int64_t maxChunk = coreEnd - outPos;
        for (int i = dimNum - 1; i >= 0; i--) {
            if (shifts[i] != 0) {
                int64_t split = shifts[i];
                int64_t offsetInSlice = outPos % strides[i];
                if (coords[i] < split) {
                    int64_t limit = (split - coords[i]) * strides[i] - offsetInSlice;
                    if (limit < maxChunk) maxChunk = limit;
                } else {
                    int64_t limit = (shapes[i] - coords[i]) * strides[i] - offsetInSlice;
                    if (limit < maxChunk) maxChunk = limit;
                }
                break;
            }
        }
        if (maxChunk > ubTensorSize) {
            maxChunk = ubTensorSize;
        }

        // Align maxChunk to 32B boundary for efficient MTE copy
        int64_t alignElems = 32 / sizeof(T);
        if (maxChunk >= alignElems) {
            maxChunk = (maxChunk / alignElems) * alignElems;
        }

        // 3. Core copy: CopyChunk internally handles UB tiling and double-buffering
        CopyChunk(srcPos, outPos, maxChunk);

        // 4. Incrementally update coordinates by additive carry (no division in steady state)
        outPos += maxChunk;
        coords[dimNum - 1] += maxChunk;
        for (int i = dimNum - 1; i > 0 && coords[i] >= shapes[i]; i--) {
            int64_t carry = coords[i] / shapes[i];
            coords[i] %= shapes[i];
            coords[i - 1] += carry;
        }
    }
}

template <typename T>
__aicore__ inline int64_t Roll<T>::ComputeSrcBaseOffset(int64_t outPos)
{
    int64_t coords[8] = {0};
    int64_t tmp = outPos;
    for (int i = 0; i < dimNum; i++) {
        coords[i] = tmp / strides[i];
        tmp %= strides[i];
    }

    int64_t srcPos = 0;
    for (int i = 0; i < dimNum; i++) {
        int64_t srcCoord = coords[i] - shifts[i];
        if (srcCoord < 0) {
            srcCoord += shapes[i];
        }
        srcPos += srcCoord * strides[i];
    }
    return srcPos;
}

template <typename T>
__aicore__ inline void Roll<T>::ProcessMultiDimUBReorder()
{
    int64_t innerIdx = this->innerShiftIdx;
    int64_t rowMult = this->rowMultiplier;
    int64_t maxRows = this->continuousRows;
    int64_t lLen = this->lenL;
    int64_t rLen = this->lenR;
    int64_t blockSize = this->outerBlockSize;

    int64_t innerSliceSize = shapes[innerIdx] * rowMult;
    int64_t totalRows = totalElements / innerSliceSize;
    int64_t baseRows = totalRows / needCoreNum;
    int64_t remRows = totalRows % needCoreNum;

    int64_t myRows = baseRows + (coreId < remRows ? 1 : 0);
    if (myRows <= 0) {
        return;
    }

    int64_t myRowStart = coreId < remRows
        ? coreId * (baseRows + 1)
        : remRows * (baseRows + 1) + (coreId - remRows) * baseRows;
    int64_t myRowEnd = myRowStart + myRows;

    int64_t lBytes = lLen * rowMult * static_cast<int64_t>(sizeof(T));
    int64_t rBytes = rLen * rowMult * static_cast<int64_t>(sizeof(T));
    int64_t alignedLBytes = ((lBytes + 31) / 32) * 32;
    int64_t alignedRBytes = ((rBytes + 31) / 32) * 32;
    int64_t alignedLElems = alignedLBytes / sizeof(T);
    int64_t alignedRElems = alignedRBytes / sizeof(T);

    int64_t totalBufBytes = 2 * ubTensorSize * static_cast<int64_t>(sizeof(T));
    int64_t bytesPerRow = alignedLBytes + alignedRBytes;
    int64_t rowsPerBatch = totalBufBytes / bytesPerRow;
    if (rowsPerBatch < 1) rowsPerBatch = 1;
    if (rowsPerBatch > maxRows) rowsPerBatch = maxRows;

    auto ubBase = calcBufA_.Get<T>();
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

    // Iterate over my assigned rows, grouped by block
    int64_t row = myRowStart;
    while (row < myRowEnd) {
        int64_t blockIdx = row / maxRows;
        int64_t inBlockRowStart = row % maxRows;
        int64_t inBlockRowEnd = inBlockRowStart + (myRowEnd - row);
        if (inBlockRowEnd > maxRows) {
            inBlockRowEnd = maxRows;
        }

        int64_t outPos = blockIdx * blockSize;
        int64_t srcPos = ComputeSrcBaseOffset(outPos);

        int64_t rowOffset = inBlockRowStart;
        while (rowOffset < inBlockRowEnd) {
            int64_t currentRows = inBlockRowEnd - rowOffset;
            if (currentRows > rowsPerBatch) {
                currentRows = rowsPerBatch;
            }

            auto ubL = ubBase;
            auto ubR = ubBase[currentRows * alignedLElems];

            int64_t srcRowOff = rowOffset * shapes[innerIdx] * rowMult;
            int64_t dstRowOff = rowOffset * shapes[innerIdx] * rowMult;

            // Gather L: srcPos already sits at the split point (innerIdx coord = lenR)
            AscendC::DataCopyExtParams copyL;
            copyL.blockCount = currentRows;
            copyL.blockLen = lBytes;
            copyL.srcStride = rBytes;
            copyL.dstStride = 0;
            DataCopyPad(ubL, xGm[srcPos + srcRowOff], copyL, padParams);

            // Gather R: need to rewind from split point back to true row start
            AscendC::DataCopyExtParams copyR;
            copyR.blockCount = currentRows;
            copyR.blockLen = rBytes;
            copyR.srcStride = lBytes;
            copyR.dstStride = 0;
            DataCopyPad(ubR, xGm[srcPos + srcRowOff - rLen * rowMult], copyR, padParams);
            SyncMTE2ToMTE3();

            // Scatter L to output left
            AscendC::DataCopyExtParams storeL;
            storeL.blockCount = currentRows;
            storeL.blockLen = lBytes;
            storeL.srcStride = 0;
            storeL.dstStride = rBytes;
            DataCopyPad(yGm[outPos + dstRowOff], ubL, storeL);

            // Scatter R to output right
            AscendC::DataCopyExtParams storeR;
            storeR.blockCount = currentRows;
            storeR.blockLen = rBytes;
            storeR.srcStride = 0;
            storeR.dstStride = lBytes;
            DataCopyPad(yGm[outPos + dstRowOff + lLen * rowMult], ubR, storeR);
            SyncMTE3ToMTE2();

            rowOffset += currentRows;
        }

        row = (blockIdx + 1) * maxRows;
    }
}

template <typename T>
__aicore__ inline void Roll<T>::ProcessMultiDim2DReorder()
{
    int64_t numElements;
    int64_t coreStart;
    int64_t alignUnit = outerBlockSize;
    if (coreId < remCoreNum) {
        coreStart = coreId * (numEachCore + alignUnit);
        numElements = numEachCore + alignUnit;
    } else {
        coreStart = remCoreNum * (numEachCore + alignUnit) + (coreId - remCoreNum) * numEachCore;
        numElements = numEachCore;
    }
    if (numElements <= 0) {
        return;
    }
    int64_t coreEnd = coreStart + numElements;

    int64_t W = shapes[innerShiftIdx];
    int64_t C = rowMultiplier;
    int64_t planeSize = outerBlockSize;
    int64_t lineStrideElems = W * C;

    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

    for (int64_t pos = coreStart; pos < coreEnd; pos += planeSize) {
        int64_t base = pos;

        for (int q = 0; q < moveCount; q++) {
            int64_t h = moveRowCount[q];
            int64_t wBytes = moveBlockLen[q];
            if (h <= 0 || wBytes <= 0) {
                continue;
            }

            int64_t alignedBlockBytes = ((wBytes + 31) / 32) * 32;
            int64_t alignedBlockElems = alignedBlockBytes / sizeof(T);
            int64_t rowsPerTile = ubTensorSize / alignedBlockElems;
            if (rowsPerTile < 1) {
                rowsPerTile = 1;
            }
            if (rowsPerTile > h) {
                rowsPerTile = h;
            }

            int64_t srcOff = base + moveSrcOffset[q];
            int64_t dstOff = base + moveDstOffset[q];

            for (int64_t row = 0; row < h; row += rowsPerTile) {
                int64_t currentRows = h - row;
                if (currentRows > rowsPerTile) {
                    currentRows = rowsPerTile;
                }

                auto ubBuf = inQueue.AllocTensor<T>();

                // Gather: strided read from GM → contiguous UB
                AscendC::DataCopyExtParams gather;
                gather.blockCount = currentRows;
                gather.blockLen = wBytes;
                gather.srcStride = moveSrcStride[q];
                gather.dstStride = 0;
                DataCopyPad(ubBuf, xGm[srcOff + row * lineStrideElems], gather, padParams);
                SyncMTE2ToMTE3();

                // Scatter: contiguous UB → strided write to GM
                AscendC::DataCopyExtParams scatter;
                scatter.blockCount = currentRows;
                scatter.blockLen = wBytes;
                scatter.srcStride = 0;
                scatter.dstStride = moveDstStride[q];
                DataCopyPad(yGm[dstOff + row * lineStrideElems], ubBuf, scatter);
                SyncMTE3ToMTE2();

                inQueue.FreeTensor(ubBuf);
            }
        }
    }
}

template <typename T>
__aicore__ inline void Roll<T>::ProcessMultiDimNDReorder()
{
    int64_t totalSubBlocks = totalElements / subBlockSize;
    int64_t baseBlocks = totalSubBlocks / needCoreNum;
    int64_t remBlocks = totalSubBlocks % needCoreNum;

    int64_t myBlocks = baseBlocks + (coreId < remBlocks ? 1 : 0);
    if (myBlocks <= 0) {
        return;
    }

    int64_t coreStart = coreId < remBlocks
        ? coreId * (baseBlocks + 1) * subBlockSize
        : (remBlocks * (baseBlocks + 1) + (coreId - remBlocks) * baseBlocks) * subBlockSize;
    int64_t coreEnd = coreStart + myBlocks * subBlockSize;

    int64_t F = shapes[innerShiftIdx - 1];
    int64_t blockSize = outerBlockSize;
    int64_t outerStride = subBlockSize;
    int64_t fillStride = strides[innerShiftIdx - 1];

    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

    for (int64_t pos = coreStart; pos < coreEnd; ) {
        int64_t blockBase = (pos / blockSize) * blockSize;
        int64_t inBlockStart = pos - blockBase;
        int64_t inBlockEnd = coreEnd - blockBase;
        if (inBlockEnd > blockSize) {
            inBlockEnd = blockSize;
        }

        int64_t outerStart = inBlockStart / outerStride;
        int64_t outerEnd = inBlockEnd / outerStride;

        for (int r = 0; r < moveCount; r++) {
            int64_t blockLenBytes = moveBlockLen[r];
            int64_t strideBytes = moveSrcStride[r];
            if (blockLenBytes <= 0) {
                continue;
            }

            int64_t roOuterLen = regionOuterLen[r];
            if (roOuterLen <= 0) {
                continue;
            }

            int64_t regDstStart = regionDstOuterOff[r];
            int64_t regDstEnd = regDstStart + roOuterLen;

            int64_t oStart = outerStart > regDstStart ? outerStart : regDstStart;
            int64_t oEnd = outerEnd < regDstEnd ? outerEnd : regDstEnd;
            if (oStart >= oEnd) {
                continue;
            }

            int64_t alignedBlockBytes = ((blockLenBytes + 31) / 32) * 32;
            int64_t alignedBlockElems = alignedBlockBytes / sizeof(T);
            int64_t rowsPerTile = ubTensorSize / alignedBlockElems;
            if (rowsPerTile < 1) {
                rowsPerTile = 1;
            }
            if (rowsPerTile > F) {
                rowsPerTile = F;
            }
            int64_t tailRows = F % rowsPerTile;
            int64_t balancedRows = (F + 1) / 2;
            if (tailRows > 0 && tailRows <= 2 && rowsPerTile > 2
                && balancedRows * alignedBlockElems <= ubTensorSize) {
                rowsPerTile = balancedRows;
            }

            // Cross-position double-buffer pipeline: preheat first tile of region,
            // then pipeline all remaining tiles across all positions in this region.
            int64_t firstO = oStart;
            int64_t firstSrcOuter = regionSrcOuterOff[r] + (firstO - regionDstOuterOff[r]);
            int64_t firstSrcBase = blockBase + firstSrcOuter * outerStride + moveSrcOffset[r];
            int64_t firstRow = 0;
            int64_t firstRows = (F - firstRow < rowsPerTile) ? (F - firstRow) : rowsPerTile;

            auto bufPrev = inQueue.AllocTensor<T>();
            AscendC::DataCopyExtParams gatherFirst;
            gatherFirst.blockCount = firstRows;
            gatherFirst.blockLen = blockLenBytes;
            gatherFirst.srcStride = strideBytes;
            gatherFirst.dstStride = 0;
            DataCopyPad(bufPrev, xGm[firstSrcBase + firstRow * fillStride], gatherFirst, padParams);
            SyncMTE2ToMTE3();
            inQueue.EnQue(bufPrev);

            int64_t prevO = firstO;
            int64_t prevT = 0;

            // Main loop: gather next tile (MTE2) while store prev tile (MTE3) in parallel
            for (int64_t o = oStart; o < oEnd; o++) {
                int64_t numTiles = (F + rowsPerTile - 1) / rowsPerTile;
                int64_t tStart = (o == oStart) ? 1 : 0;
                for (int64_t t = tStart; t < numTiles; t++) {
                    // Gather next tile (MTE2, background)
                    int64_t curSrcOuter = regionSrcOuterOff[r] + (o - regionDstOuterOff[r]);
                    int64_t curSrcBase = blockBase + curSrcOuter * outerStride + moveSrcOffset[r];
                    int64_t row = t * rowsPerTile;
                    int64_t curRows = (F - row < rowsPerTile) ? (F - row) : rowsPerTile;

                    auto bufNext = inQueue.AllocTensor<T>();
                    AscendC::DataCopyExtParams gatherNext;
                    gatherNext.blockCount = curRows;
                    gatherNext.blockLen = blockLenBytes;
                    gatherNext.srcStride = strideBytes;
                    gatherNext.dstStride = 0;
                    DataCopyPad(bufNext, xGm[curSrcBase + row * fillStride], gatherNext, padParams);

                    // Store previous tile (MTE3, foreground)
                    auto bufReady = inQueue.DeQue<T>();
                    int64_t prevDstOuter = prevO;
                    int64_t prevDstBase = blockBase + prevDstOuter * outerStride + moveDstOffset[r];
                    int64_t prevRow = prevT * rowsPerTile;
                    int64_t prevRows = (F - prevRow < rowsPerTile) ? (F - prevRow) : rowsPerTile;

                    AscendC::DataCopyExtParams scatterPrev;
                    scatterPrev.blockCount = prevRows;
                    scatterPrev.blockLen = blockLenBytes;
                    scatterPrev.srcStride = 0;
                    scatterPrev.dstStride = strideBytes;
                    DataCopyPad(yGm[prevDstBase + prevRow * fillStride], bufReady, scatterPrev);

                    SyncMTE3ToMTE2();
                    inQueue.FreeTensor(bufReady);

                    SyncMTE2ToMTE3();
                    inQueue.EnQue(bufNext);

                    prevO = o;
                    prevT = t;
                }
            }

            // Drain last tile
            auto bufLast = inQueue.DeQue<T>();
            int64_t lastDstOuter = prevO;
            int64_t lastDstBase = blockBase + lastDstOuter * outerStride + moveDstOffset[r];
            int64_t lastRow = prevT * rowsPerTile;
            int64_t lastRows = (F - lastRow < rowsPerTile) ? (F - lastRow) : rowsPerTile;

            AscendC::DataCopyExtParams scatterLast;
            scatterLast.blockCount = lastRows;
            scatterLast.blockLen = blockLenBytes;
            scatterLast.srcStride = 0;
            scatterLast.dstStride = strideBytes;
            DataCopyPad(yGm[lastDstBase + lastRow * fillStride], bufLast, scatterLast);
            SyncMTE3ToMTE2();
            inQueue.FreeTensor(bufLast);
        }
        pos = blockBase + blockSize;
    }
}

template <typename T>
__aicore__ inline bool Roll<T>::IsPairedRowTile() const
{
    int64_t innerIdx = innerShiftIdx;
    int64_t outerIdx = innerIdx - 2;
    return (tilingMode == 2 && subMode == 12 && outerIdx >= 0
            && shifts[outerIdx] != 0 && shifts[innerIdx] != 0
            && shifts[innerIdx - 1] == 0
            && subBlockSize == strides[outerIdx]);
}

template <typename T>
__aicore__ inline void Roll<T>::ProcessPairedRowTile()
{
    int64_t innerIdx = innerShiftIdx;
    int64_t outerIdx = innerIdx - 2;
    int64_t rowIdx = innerIdx - 1;
    int64_t outerDim = shapes[outerIdx];
    int64_t rowCount = shapes[rowIdx];
    int64_t rowStride = shapes[innerIdx] * rowMultiplier;
    int64_t totalSubBlocks = totalElements / subBlockSize;
    int64_t totalRows = totalSubBlocks * rowCount;
    int64_t baseRows = totalRows / needCoreNum;
    int64_t remRows = totalRows % needCoreNum;

    int64_t myRows = baseRows + (coreId < remRows ? 1 : 0);
    if (myRows <= 0) {
        return;
    }

    int64_t rowTask = coreId < remRows
        ? coreId * (baseRows + 1)
        : remRows * (baseRows + 1) + (coreId - remRows) * baseRows;
    int64_t rowTaskEnd = rowTask + myRows;

    int64_t leftElems = shifts[innerIdx] * rowMultiplier;
    int64_t rightElems = (shapes[innerIdx] - shifts[innerIdx]) * rowMultiplier;
    int64_t leftBytes = leftElems * static_cast<int64_t>(sizeof(T));
    int64_t rightBytes = rightElems * static_cast<int64_t>(sizeof(T));
    int64_t alignedLeftBytes = ((leftBytes + 31) / 32) * 32;
    int64_t alignedRightBytes = ((rightBytes + 31) / 32) * 32;
    int64_t alignedLeftElems = alignedLeftBytes / static_cast<int64_t>(sizeof(T));
    int64_t alignedRightElems = alignedRightBytes / static_cast<int64_t>(sizeof(T));
    int64_t pairRows = ubTensorSize / (alignedLeftElems + alignedRightElems);
    if (pairRows < 1) {
        pairRows = 1;
    }
    if (pairRows > rowCount) {
        pairRows = rowCount;
    }

    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

    int64_t sub = rowTask / rowCount;
    int64_t row = rowTask - sub * rowCount;
    int64_t prefix = sub / outerDim;
    int64_t outerCoord = sub - prefix * outerDim;
    int64_t srcOuterCoord = outerCoord - shifts[outerIdx];
    if (srcOuterCoord < 0) {
        srcOuterCoord += outerDim;
    }
    int64_t srcBase = (prefix * outerDim + srcOuterCoord) * subBlockSize;
    int64_t dstBase = sub * subBlockSize;

    while (rowTask < rowTaskEnd) {
        int64_t curRows = rowCount - row;
        if (curRows > pairRows) {
            curRows = pairRows;
        }
        if (curRows > rowTaskEnd - rowTask) {
            curRows = rowTaskEnd - rowTask;
        }

        auto ubBuf = inQueue.AllocTensor<T>();

        AscendC::DataCopyExtParams leftGather;
        leftGather.blockCount = static_cast<uint32_t>(curRows);
        leftGather.blockLen = static_cast<uint32_t>(leftBytes);
        leftGather.srcStride = static_cast<uint32_t>(rightBytes);
        leftGather.dstStride = 0;
        DataCopyPad(ubBuf, xGm[srcBase + row * rowStride + rightElems], leftGather, padParams);

        AscendC::DataCopyExtParams rightGather;
        rightGather.blockCount = static_cast<uint32_t>(curRows);
        rightGather.blockLen = static_cast<uint32_t>(rightBytes);
        rightGather.srcStride = static_cast<uint32_t>(leftBytes);
        rightGather.dstStride = 0;
        DataCopyPad(ubBuf[curRows * alignedLeftElems], xGm[srcBase + row * rowStride],
                    rightGather, padParams);

        SyncMTE2ToMTE3();

        AscendC::DataCopyExtParams leftStore;
        leftStore.blockCount = static_cast<uint32_t>(curRows);
        leftStore.blockLen = static_cast<uint32_t>(leftBytes);
        leftStore.srcStride = 0;
        leftStore.dstStride = static_cast<uint32_t>(rightBytes);
        DataCopyPad(yGm[dstBase + row * rowStride], ubBuf, leftStore);

        AscendC::DataCopyExtParams rightStore;
        rightStore.blockCount = static_cast<uint32_t>(curRows);
        rightStore.blockLen = static_cast<uint32_t>(rightBytes);
        rightStore.srcStride = 0;
        rightStore.dstStride = static_cast<uint32_t>(leftBytes);
        DataCopyPad(yGm[dstBase + row * rowStride + leftElems], ubBuf[curRows * alignedLeftElems],
                    rightStore);

        SyncMTE3ToMTE2();
        inQueue.FreeTensor(ubBuf);
        rowTask += curRows;
        row += curRows;
        if (row == rowCount && rowTask < rowTaskEnd) {
            row = 0;
            sub++;
            outerCoord++;
            if (outerCoord == outerDim) {
                outerCoord = 0;
                prefix++;
            }
            srcOuterCoord = outerCoord - shifts[outerIdx];
            if (srcOuterCoord < 0) {
                srcOuterCoord += outerDim;
            }
            srcBase = (prefix * outerDim + srcOuterCoord) * subBlockSize;
            dstBase = sub * subBlockSize;
        }
    }
}

template <typename T>
__aicore__ inline void Roll<T>::ProcessMultiDimNDUnit()
{
    constexpr int64_t BATCH_UNITS = 8;
    int64_t innerIdx = this->innerShiftIdx;
    int64_t outerIdx = innerIdx - 2;
    int64_t gapIdx = innerIdx - 1;
    int64_t rowMult = this->rowMultiplier;
    int64_t unitSize = shapes[innerIdx] * rowMult;
    int64_t outerDim = shapes[outerIdx];
    int64_t gapDim = shapes[gapIdx];
    int64_t innerDim = shapes[innerIdx];
    int64_t totalBlocks = totalElements / (gapDim * unitSize);
    int64_t baseBlocks = totalBlocks / needCoreNum;
    int64_t remBlocks = totalBlocks % needCoreNum;

    int64_t myBlocks = baseBlocks + (coreId < remBlocks ? 1 : 0);
    if (myBlocks <= 0) {
        return;
    }

    int64_t blockStart = coreId < remBlocks
        ? coreId * (baseBlocks + 1)
        : remBlocks * (baseBlocks + 1) + (coreId - remBlocks) * baseBlocks;
    int64_t batchesPerBlock = (gapDim + BATCH_UNITS - 1) / BATCH_UNITS;
    int64_t totalBatches = myBlocks * batchesPerBlock;

    auto offsetTensor = offsetBuf_.Get<uint32_t>();
    int ts = static_cast<int>(sizeof(T));
    for (int64_t b = 0; b < BATCH_UNITS; b++) {
        int64_t unitBase = b * unitSize;
        for (int64_t h = 0; h < innerDim; h++) {
            int64_t srcH = h - shifts[innerIdx];
            if (srcH < 0) {
                srcH += innerDim;
            }
            for (int64_t c = 0; c < rowMult; c++) {
                int64_t idx = unitBase + h * rowMult + c;
                int64_t srcIdx = unitBase + srcH * rowMult + c;
                offsetTensor.SetValue(idx, static_cast<uint32_t>(srcIdx * ts));
            }
        }
    }

    AscendC::DataCopyExtParams bulkCopy;
    bulkCopy.blockCount = 1;
    bulkCopy.srcStride = 0;
    bulkCopy.dstStride = 0;
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

    int64_t batch = 0;
    int64_t block = blockStart;
    int64_t batchInBlock = 0;
    int64_t axis2 = 0;
    int64_t batchUnits = gapDim;
    if (batchUnits > BATCH_UNITS) batchUnits = BATCH_UNITS;
    int64_t axis1 = block % outerDim;
    int64_t axis0 = block / outerDim;
    int64_t srcAxis1 = axis1 - shifts[outerIdx];
    if (srcAxis1 < 0) {
        srcAxis1 += outerDim;
    }
    int64_t readyUnit = block * gapDim + axis2;
    int64_t srcUnit = (axis0 * outerDim + srcAxis1) * gapDim + axis2;
    int64_t readyCount = batchUnits * unitSize;

    auto inputFirst = inQueue.template AllocTensor<T>();
    bulkCopy.blockLen = static_cast<uint32_t>(readyCount * static_cast<int64_t>(sizeof(T)));
    DataCopyPad(inputFirst, xGm[srcUnit * unitSize], bulkCopy, padParams);
    inQueue.EnQue(inputFirst);

    int64_t outputUnit = 0;
    int64_t outputCount = 0;
    bool hasOutput = false;
    batch++;

    while (batch < totalBatches) {
        block = blockStart + batch / batchesPerBlock;
        batchInBlock = batch - (batch / batchesPerBlock) * batchesPerBlock;
        axis2 = batchInBlock * BATCH_UNITS;
        batchUnits = gapDim - axis2;
        if (batchUnits > BATCH_UNITS) batchUnits = BATCH_UNITS;
        axis1 = block % outerDim;
        axis0 = block / outerDim;
        srcAxis1 = axis1 - shifts[outerIdx];
        if (srcAxis1 < 0) {
            srcAxis1 += outerDim;
        }
        srcUnit = (axis0 * outerDim + srcAxis1) * gapDim + axis2;
        int64_t nextUnit = block * gapDim + axis2;
        int64_t nextCount = batchUnits * unitSize;

        auto inputNext = inQueue.template AllocTensor<T>();
        bulkCopy.blockLen = static_cast<uint32_t>(nextCount * static_cast<int64_t>(sizeof(T)));
        DataCopyPad(inputNext, xGm[srcUnit * unitSize], bulkCopy, padParams);
        inQueue.EnQue(inputNext);

        auto inputReady = inQueue.template DeQue<T>();
        auto outputReady = outQueue_.template AllocTensor<T>();
        Gather(outputReady, inputReady, offsetTensor, static_cast<uint32_t>(0), static_cast<uint32_t>(readyCount));
        outQueue_.template EnQue<T>(outputReady);
        inQueue.FreeTensor(inputReady);

        if (hasOutput) {
            auto outputLocal = outQueue_.template DeQue<T>();
            bulkCopy.blockLen = static_cast<uint32_t>(outputCount * static_cast<int64_t>(sizeof(T)));
            DataCopyPad(yGm[outputUnit * unitSize], outputLocal, bulkCopy);
            outQueue_.FreeTensor(outputLocal);
        }

        outputUnit = readyUnit;
        outputCount = readyCount;
        hasOutput = true;
        readyUnit = nextUnit;
        readyCount = nextCount;
        batch++;
    }

    auto inputLast = inQueue.template DeQue<T>();
    auto outputLast = outQueue_.template AllocTensor<T>();
    Gather(outputLast, inputLast, offsetTensor, static_cast<uint32_t>(0), static_cast<uint32_t>(readyCount));
    outQueue_.template EnQue<T>(outputLast);
    inQueue.FreeTensor(inputLast);

    if (hasOutput) {
        auto outputLocal = outQueue_.template DeQue<T>();
        bulkCopy.blockLen = static_cast<uint32_t>(outputCount * static_cast<int64_t>(sizeof(T)));
        DataCopyPad(yGm[outputUnit * unitSize], outputLocal, bulkCopy);
        outQueue_.FreeTensor(outputLocal);
    }

    auto outputLocal = outQueue_.template DeQue<T>();
    bulkCopy.blockLen = static_cast<uint32_t>(readyCount * static_cast<int64_t>(sizeof(T)));
    DataCopyPad(yGm[readyUnit * unitSize], outputLocal, bulkCopy);
    outQueue_.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Roll<T>::ProcessMultiDimPlaneGather()
{
    int64_t planeSize = outerBlockSize;
    int64_t planeCount = totalElements / planeSize;
    if (coreId >= planeCount) {
        return;
    }

    int64_t innerIdx = innerShiftIdx;
    int64_t outerIdx = innerIdx - 1;
    int64_t H = shapes[outerIdx];
    int64_t W = shapes[innerIdx] * rowMultiplier;

    auto offsetTensor = offsetBuf_.Get<uint32_t>();
    int ts = static_cast<int>(sizeof(T));
    for (int64_t h = 0; h < H; h++) {
        int64_t srcH = h - shifts[outerIdx];
        if (srcH < 0) {
            srcH += H;
        }
        for (int64_t w = 0; w < W; w++) {
            int64_t innerCoord = w / rowMultiplier;
            int64_t lane = w - innerCoord * rowMultiplier;
            int64_t srcInner = innerCoord - shifts[innerIdx];
            if (srcInner < 0) {
                srcInner += shapes[innerIdx];
            }
            int64_t idx = h * W + w;
            int64_t srcIdx = srcH * W + srcInner * rowMultiplier + lane;
            offsetTensor.SetValue(idx, static_cast<uint32_t>(srcIdx * ts));
        }
    }

    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(planeSize * static_cast<int64_t>(sizeof(T)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

    int64_t plane = coreId;
    auto inputFirst = inQueue.template AllocTensor<T>();
    int64_t firstBase = plane * planeSize;
    DataCopyPad(inputFirst, xGm[firstBase], copyParams, padParams);
    inQueue.EnQue(inputFirst);

    int64_t readyPlane = plane;
    int64_t outputPlane = 0;
    bool hasOutput = false;
    plane += needCoreNum;

    while (plane < planeCount) {
        auto inputNext = inQueue.template AllocTensor<T>();
        int64_t nextBase = plane * planeSize;
        DataCopyPad(inputNext, xGm[nextBase], copyParams, padParams);
        inQueue.EnQue(inputNext);

        auto inputReady = inQueue.template DeQue<T>();
        auto outputReady = outQueue_.template AllocTensor<T>();
        Gather(outputReady, inputReady, offsetTensor, static_cast<uint32_t>(0), static_cast<uint32_t>(planeSize));
        outQueue_.template EnQue<T>(outputReady);
        inQueue.FreeTensor(inputReady);

        if (hasOutput) {
            auto outputLocal = outQueue_.template DeQue<T>();
            DataCopyPad(yGm[outputPlane * planeSize], outputLocal, copyParams);
            outQueue_.FreeTensor(outputLocal);
        }

        outputPlane = readyPlane;
        hasOutput = true;
        readyPlane = plane;
        plane += needCoreNum;
    }

    auto inputLast = inQueue.template DeQue<T>();
    auto outputLast = outQueue_.template AllocTensor<T>();
    Gather(outputLast, inputLast, offsetTensor, static_cast<uint32_t>(0), static_cast<uint32_t>(planeSize));
    outQueue_.template EnQue<T>(outputLast);
    inQueue.FreeTensor(inputLast);

    if (hasOutput) {
        auto outputLocal = outQueue_.template DeQue<T>();
        DataCopyPad(yGm[outputPlane * planeSize], outputLocal, copyParams);
        outQueue_.FreeTensor(outputLocal);
    }

    auto outputLocal = outQueue_.template DeQue<T>();
    DataCopyPad(yGm[readyPlane * planeSize], outputLocal, copyParams);
    outQueue_.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Roll<T>::ProcessMultiDimTileGather()
{
    int64_t tileSize = outerBlockSize;
    int64_t totalTiles = totalElements / tileSize;
    int64_t innerIdx = innerShiftIdx;
    int64_t outerIdx = innerIdx - 2;
    int64_t gapIdx = innerIdx - 1;
    int64_t W = shapes[innerIdx] * rowMultiplier;
    int64_t tileRows = tileSize / W;
    int64_t axis2Dim = shapes[gapIdx] / tileRows;
    int64_t outerDim = shapes[outerIdx];
    int64_t chunkSize = subBlockSize > 0 ? subBlockSize : tileSize;
    if (chunkSize > tileSize) {
        chunkSize = tileSize;
    }
    int64_t chunkRows = chunkSize / W;
    if (tileRows <= 0 || axis2Dim <= 0 || chunkRows <= 0) {
        return;
    }
    int64_t chunksPerTile = (tileRows + chunkRows - 1) / chunkRows;
    int64_t totalChunks = totalTiles * chunksPerTile;
    int64_t baseChunks = totalChunks / needCoreNum;
    int64_t remChunks = totalChunks % needCoreNum;
    int64_t myChunks = baseChunks + (coreId < remChunks ? 1 : 0);
    if (myChunks <= 0) {
        return;
    }
    int64_t chunkStart = coreId < remChunks
        ? coreId * (baseChunks + 1)
        : remChunks * (baseChunks + 1) + (coreId - remChunks) * baseChunks;
    int64_t chunkEnd = chunkStart + myChunks;

    auto offsetTensor = offsetBuf_.Get<uint32_t>();
    int ts = static_cast<int>(sizeof(T));
    for (int64_t h = 0; h < chunkRows; h++) {
        for (int64_t w = 0; w < W; w++) {
            int64_t innerCoord = w / rowMultiplier;
            int64_t lane = w - innerCoord * rowMultiplier;
            int64_t srcInner = innerCoord - shifts[innerIdx];
            if (srcInner < 0) {
                srcInner += shapes[innerIdx];
            }
            int64_t idx = h * W + w;
            int64_t srcIdx = h * W + srcInner * rowMultiplier + lane;
            offsetTensor.SetValue(idx, static_cast<uint32_t>(srcIdx * ts));
        }
    }

    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

    int64_t chunk = chunkStart;
    int64_t tile = chunk / chunksPerTile;
    int64_t chunkInTile = chunk - tile * chunksPerTile;
    int64_t hBase = chunkInTile * chunkRows;
    int64_t rows = tileRows - hBase;
    if (rows > chunkRows) rows = chunkRows;
    int64_t count = rows * W;
    int64_t curAxis2 = tile % axis2Dim;
    int64_t tmp = tile / axis2Dim;
    int64_t outerCoord = tmp % outerDim;
    int64_t prefix = tmp / outerDim;
    int64_t srcOuter = outerCoord - shifts[outerIdx];
    if (srcOuter < 0) {
        srcOuter += outerDim;
    }
    int64_t srcTile = (prefix * outerDim + srcOuter) * axis2Dim + curAxis2;
    auto inputFirst = inQueue.template AllocTensor<T>();
    copyParams.blockLen = static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(T)));
    DataCopyPad(inputFirst, xGm[srcTile * tileSize + hBase * W], copyParams, padParams);
    inQueue.EnQue(inputFirst);

    int64_t readyChunk = chunk;
    int64_t readyCount = count;
    int64_t outputChunk = 0;
    int64_t outputCount = 0;
    bool hasOutput = false;
    chunk++;

    while (chunk < chunkEnd) {
        tile = chunk / chunksPerTile;
        chunkInTile = chunk - tile * chunksPerTile;
        hBase = chunkInTile * chunkRows;
        rows = tileRows - hBase;
        if (rows > chunkRows) rows = chunkRows;
        count = rows * W;
        curAxis2 = tile % axis2Dim;
        tmp = tile / axis2Dim;
        outerCoord = tmp % outerDim;
        prefix = tmp / outerDim;
        srcOuter = outerCoord - shifts[outerIdx];
        if (srcOuter < 0) {
            srcOuter += outerDim;
        }
        srcTile = (prefix * outerDim + srcOuter) * axis2Dim + curAxis2;
        auto inputNext = inQueue.template AllocTensor<T>();
        copyParams.blockLen = static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(T)));
        DataCopyPad(inputNext, xGm[srcTile * tileSize + hBase * W], copyParams, padParams);
        inQueue.EnQue(inputNext);

        auto inputReady = inQueue.template DeQue<T>();
        auto outputReady = outQueue_.template AllocTensor<T>();
        Gather(outputReady, inputReady, offsetTensor, static_cast<uint32_t>(0), static_cast<uint32_t>(readyCount));
        outQueue_.template EnQue<T>(outputReady);
        inQueue.FreeTensor(inputReady);

        if (hasOutput) {
            int64_t outTile = outputChunk / chunksPerTile;
            int64_t outChunkInTile = outputChunk - outTile * chunksPerTile;
            int64_t outHBase = outChunkInTile * chunkRows;
            auto outputLocal = outQueue_.template DeQue<T>();
            copyParams.blockLen = static_cast<uint32_t>(outputCount * static_cast<int64_t>(sizeof(T)));
            DataCopyPad(yGm[outTile * tileSize + outHBase * W], outputLocal, copyParams);
            outQueue_.FreeTensor(outputLocal);
        }

        outputChunk = readyChunk;
        outputCount = readyCount;
        hasOutput = true;
        readyChunk = chunk;
        readyCount = count;
        chunk++;
    }

    auto inputLast = inQueue.template DeQue<T>();
    auto outputLast = outQueue_.template AllocTensor<T>();
    Gather(outputLast, inputLast, offsetTensor, static_cast<uint32_t>(0), static_cast<uint32_t>(readyCount));
    outQueue_.template EnQue<T>(outputLast);
    inQueue.FreeTensor(inputLast);

    if (hasOutput) {
        int64_t outTile = outputChunk / chunksPerTile;
        int64_t outChunkInTile = outputChunk - outTile * chunksPerTile;
        int64_t outHBase = outChunkInTile * chunkRows;
        auto outputLocal = outQueue_.template DeQue<T>();
        copyParams.blockLen = static_cast<uint32_t>(outputCount * static_cast<int64_t>(sizeof(T)));
        DataCopyPad(yGm[outTile * tileSize + outHBase * W], outputLocal, copyParams);
        outQueue_.FreeTensor(outputLocal);
    }

    int64_t outTile = readyChunk / chunksPerTile;
    int64_t outChunkInTile = readyChunk - outTile * chunksPerTile;
    int64_t outHBase = outChunkInTile * chunkRows;
    auto outputLocal = outQueue_.template DeQue<T>();
    copyParams.blockLen = static_cast<uint32_t>(readyCount * static_cast<int64_t>(sizeof(T)));
    DataCopyPad(yGm[outTile * tileSize + outHBase * W], outputLocal, copyParams);
    outQueue_.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Roll<T>::ProcessGatherSupplement()
{
    int ts = static_cast<int>(sizeof(T));
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

    if (subBlockSize > 0 && rowMultiplier > 1) {
        int64_t CHUNK_D = subBlockSize;
        int64_t innerIdx = innerShiftIdx;
        int64_t outerIdx = innerIdx - 2;
        int64_t gapIdx = innerIdx - 1;
        int64_t D = shapes[gapIdx];
        int64_t H = shapes[innerIdx];
        int64_t rowSize = H * rowMultiplier;
        int64_t mainH = H - shifts[innerIdx];
        int64_t wrapH = shifts[innerIdx];
        int64_t mainElems = mainH * rowMultiplier;
        int64_t wrapElems = wrapH * rowMultiplier;
        uint32_t mainBytes = static_cast<uint32_t>(mainElems * static_cast<int64_t>(sizeof(T)));
        uint32_t wrapBytes = static_cast<uint32_t>(wrapElems * static_cast<int64_t>(sizeof(T)));
        int64_t mainBlockBytes = ((static_cast<int64_t>(mainBytes) + 31) / 32) * 32;
        int64_t wrapBlockBytes = ((static_cast<int64_t>(wrapBytes) + 31) / 32) * 32;
        int64_t mainSegmentElems = CHUNK_D * (mainBlockBytes / static_cast<int64_t>(sizeof(T)));
        int64_t wrapSegmentElems = CHUNK_D * (wrapBlockBytes / static_cast<int64_t>(sizeof(T)));
        int64_t slotElems = mainSegmentElems + wrapSegmentElems;
        auto ubBase = calcBufA_.Get<T>();
        int64_t axis1Size = shapes[outerIdx];

        int64_t prefixCount = (outerIdx > 0) ? (totalElements / strides[outerIdx - 1]) : 1;
        if (CHUNK_D >= D && needCoreNum == prefixCount * axis1Size) {
            if (coreId >= needCoreNum) {
                return;
            }
            int64_t axis0 = coreId / axis1Size;
            int64_t axis1 = coreId - axis0 * axis1Size;
            int64_t srcAxis1 = axis1 - shifts[outerIdx];
            if (srcAxis1 < 0) {
                srcAxis1 += axis1Size;
            }
            int64_t srcBase = (axis0 * axis1Size + srcAxis1) * D * rowSize;
            int64_t dstBase = (axis0 * axis1Size + axis1) * D * rowSize;

            AscendC::DataCopyExtParams gatherParams;
            AscendC::DataCopyExtParams scatterParams;
            gatherParams.blockCount = static_cast<uint32_t>(D);
            gatherParams.dstStride = 0;
            gatherParams.blockLen = mainBytes;
            gatherParams.srcStride = wrapBytes;
            DataCopyPad(ubBase, xGm[srcBase], gatherParams, padParams);
            gatherParams.blockLen = wrapBytes;
            gatherParams.srcStride = mainBytes;
            DataCopyPad(ubBase[mainSegmentElems], xGm[srcBase + mainElems], gatherParams, padParams);
            SyncMTE2ToMTE3();

            scatterParams.blockCount = static_cast<uint32_t>(D);
            scatterParams.srcStride = 0;
            scatterParams.blockLen = mainBytes;
            scatterParams.dstStride = wrapBytes;
            DataCopyPad(yGm[dstBase + wrapElems], ubBase, scatterParams);
            scatterParams.blockLen = wrapBytes;
            scatterParams.dstStride = mainBytes;
            DataCopyPad(yGm[dstBase], ubBase[mainSegmentElems], scatterParams);
            SyncMTE3ToMTE2();
            return;
        }

        int64_t chunkCount = (D + CHUNK_D - 1) / CHUNK_D;
        int64_t axis1Shift = shifts[outerIdx];
        int64_t totalTasks = prefixCount * axis1Size * chunkCount;
        int64_t baseTasks = totalTasks / needCoreNum;
        int64_t remTasks = totalTasks % needCoreNum;
        int64_t myTasks = baseTasks + (coreId < remTasks ? 1 : 0);
        if (myTasks <= 0) {
            return;
        }
        int64_t taskStart = coreId < remTasks
            ? coreId * (baseTasks + 1)
            : remTasks * (baseTasks + 1) + (coreId - remTasks) * baseTasks;
        int64_t taskEnd = taskStart + myTasks;

        AscendC::DataCopyExtParams gatherParams;
        AscendC::DataCopyExtParams scatterParams;
        gatherParams.dstStride = 0;
        scatterParams.srcStride = 0;

        int64_t task = taskStart;
        int64_t tmp = taskStart / chunkCount;
        int64_t chunkIdx = taskStart - tmp * chunkCount;
        int64_t axis1 = tmp % axis1Size;
        int64_t axis0 = tmp / axis1Size;
        int64_t dStart = chunkIdx * CHUNK_D;
        int64_t dCount = D - dStart;
        if (dCount > CHUNK_D) dCount = CHUNK_D;
        int64_t srcAxis1 = axis1 - axis1Shift;
        if (srcAxis1 < 0) {
            srcAxis1 += axis1Size;
        }
        int64_t srcPlaneBase = (axis0 * axis1Size + srcAxis1) * D * rowSize;
        int64_t dstPlaneBase = (axis0 * axis1Size + axis1) * D * rowSize;
        int64_t srcBase = srcPlaneBase + dStart * rowSize;
        int64_t readyDst = dstPlaneBase + dStart * rowSize;
        int64_t readyCount = dCount;
        auto firstMain = ubBase;
        auto firstWrap = firstMain[mainSegmentElems];
        gatherParams.blockCount = static_cast<uint32_t>(readyCount);
        gatherParams.blockLen = mainBytes;
        gatherParams.srcStride = wrapBytes;
        DataCopyPad(firstMain, xGm[srcBase], gatherParams, padParams);
        gatherParams.blockLen = wrapBytes;
        gatherParams.srcStride = mainBytes;
        DataCopyPad(firstWrap, xGm[srcBase + mainElems], gatherParams, padParams);
        SyncMTE2ToMTE3();

        int64_t readySlot = 0;
        task++;
        while (task < taskEnd) {
            chunkIdx++;
            if (chunkIdx == chunkCount) {
                chunkIdx = 0;
                axis1++;
                if (axis1 == axis1Size) {
                    axis1 = 0;
                    axis0++;
                }
                srcAxis1 = axis1 - axis1Shift;
                if (srcAxis1 < 0) {
                    srcAxis1 += axis1Size;
                }
                srcPlaneBase = (axis0 * axis1Size + srcAxis1) * D * rowSize;
                dstPlaneBase = (axis0 * axis1Size + axis1) * D * rowSize;
            }
            dStart = chunkIdx * CHUNK_D;
            dCount = D - dStart;
            if (dCount > CHUNK_D) dCount = CHUNK_D;
            srcBase = srcPlaneBase + dStart * rowSize;
            int64_t nextDst = dstPlaneBase + dStart * rowSize;
            int64_t nextCount = dCount;
            int64_t nextSlot = 1 - readySlot;
            auto nextMain = ubBase[nextSlot * slotElems];
            auto nextWrap = nextMain[mainSegmentElems];
            gatherParams.blockCount = static_cast<uint32_t>(nextCount);
            gatherParams.blockLen = mainBytes;
            gatherParams.srcStride = wrapBytes;
            DataCopyPad(nextMain, xGm[srcBase], gatherParams, padParams);
            gatherParams.blockLen = wrapBytes;
            gatherParams.srcStride = mainBytes;
            DataCopyPad(nextWrap, xGm[srcBase + mainElems], gatherParams, padParams);

            auto readyMain = ubBase[readySlot * slotElems];
            auto readyWrap = readyMain[mainSegmentElems];
            scatterParams.blockCount = static_cast<uint32_t>(readyCount);
            scatterParams.blockLen = mainBytes;
            scatterParams.dstStride = wrapBytes;
            DataCopyPad(yGm[readyDst + wrapElems], readyMain, scatterParams);
            scatterParams.blockLen = wrapBytes;
            scatterParams.dstStride = mainBytes;
            DataCopyPad(yGm[readyDst], readyWrap, scatterParams);
            SyncMTE3ToMTE2();
            SyncMTE2ToMTE3();

            readySlot = nextSlot;
            readyDst = nextDst;
            readyCount = nextCount;
            task++;
        }

        auto readyMain = ubBase[readySlot * slotElems];
        auto readyWrap = readyMain[mainSegmentElems];
        scatterParams.blockCount = static_cast<uint32_t>(readyCount);
        scatterParams.blockLen = mainBytes;
        scatterParams.dstStride = wrapBytes;
        DataCopyPad(yGm[readyDst + wrapElems], readyMain, scatterParams);
        scatterParams.blockLen = wrapBytes;
        scatterParams.dstStride = mainBytes;
        DataCopyPad(yGm[readyDst], readyWrap, scatterParams);
        SyncMTE3ToMTE2();
        return;
    }

    if (subMode == 13) {
        int64_t innerIdx = innerShiftIdx;
        int64_t outerIdx = innerIdx - 2;
        int64_t gapIdx = innerIdx - 1;
        int64_t D = shapes[gapIdx];
        int64_t rowSize = shapes[innerIdx] * rowMultiplier;
        int64_t mainH = shapes[innerIdx] - shifts[innerIdx];
        int64_t wrapH = shifts[innerIdx];
        int64_t mainElems = mainH * rowMultiplier;
        int64_t wrapElems = wrapH * rowMultiplier;
        uint32_t mainBytes = static_cast<uint32_t>(mainElems * static_cast<int64_t>(sizeof(T)));
        uint32_t wrapBytes = static_cast<uint32_t>(wrapElems * static_cast<int64_t>(sizeof(T)));
        int64_t mainBlockBytes = ((static_cast<int64_t>(mainBytes) + 31) / 32) * 32;
        int64_t wrapBlockBytes = ((static_cast<int64_t>(wrapBytes) + 31) / 32) * 32;
        int64_t chunkRows = outerBlockSize / rowSize;
        if (chunkRows <= 0) {
            return;
        }
        int64_t mainSegmentElems = chunkRows * (mainBlockBytes / static_cast<int64_t>(sizeof(T)));
        int64_t wrapSegmentElems = chunkRows * (wrapBlockBytes / static_cast<int64_t>(sizeof(T)));
        int64_t slotElems = mainSegmentElems + wrapSegmentElems;
        auto ubBase = calcBufA_.Get<T>();
        int64_t chunkCount = (D + chunkRows - 1) / chunkRows;
        int64_t outerDim = shapes[outerIdx];
        int64_t prefixCount = (outerIdx > 0) ? (totalElements / strides[outerIdx - 1]) : 1;
        int64_t totalTasks = prefixCount * outerDim * chunkCount;
        int64_t baseTasks = totalTasks / needCoreNum;
        int64_t remTasks = totalTasks % needCoreNum;
        int64_t myTasks = baseTasks + (coreId < remTasks ? 1 : 0);
        if (myTasks <= 0) {
            return;
        }
        int64_t taskStart = coreId < remTasks
            ? coreId * (baseTasks + 1)
            : remTasks * (baseTasks + 1) + (coreId - remTasks) * baseTasks;
        int64_t taskEnd = taskStart + myTasks;

        AscendC::DataCopyExtParams gatherParams;
        AscendC::DataCopyExtParams scatterParams;
        gatherParams.dstStride = 0;
        scatterParams.srcStride = 0;

        int64_t task = taskStart;
        int64_t chunkIdx = task % chunkCount;
        int64_t tmp = task / chunkCount;
        int64_t axis1 = tmp % outerDim;
        int64_t axis0 = tmp / outerDim;
        int64_t dStart = chunkIdx * chunkRows;
        int64_t dCount = D - dStart;
        if (dCount > chunkRows) dCount = chunkRows;
        int64_t srcAxis1 = axis1 - shifts[outerIdx];
        if (srcAxis1 < 0) {
            srcAxis1 += outerDim;
        }
        int64_t srcBase = ((axis0 * outerDim + srcAxis1) * D + dStart) * rowSize;
        int64_t readyDst = ((axis0 * outerDim + axis1) * D + dStart) * rowSize;
        int64_t readyCount = dCount;
        auto firstMain = ubBase;
        auto firstWrap = firstMain[mainSegmentElems];
        gatherParams.blockCount = static_cast<uint32_t>(readyCount);
        gatherParams.blockLen = mainBytes;
        gatherParams.srcStride = wrapBytes;
        DataCopyPad(firstMain, xGm[srcBase], gatherParams, padParams);
        gatherParams.blockLen = wrapBytes;
        gatherParams.srcStride = mainBytes;
        DataCopyPad(firstWrap, xGm[srcBase + mainElems], gatherParams, padParams);
        SyncMTE2ToMTE3();

        int64_t readySlot = 0;
        task++;
        while (task < taskEnd) {
            chunkIdx = task % chunkCount;
            tmp = task / chunkCount;
            axis1 = tmp % outerDim;
            axis0 = tmp / outerDim;
            dStart = chunkIdx * chunkRows;
            dCount = D - dStart;
            if (dCount > chunkRows) dCount = chunkRows;
            srcAxis1 = axis1 - shifts[outerIdx];
            if (srcAxis1 < 0) {
                srcAxis1 += outerDim;
            }
            srcBase = ((axis0 * outerDim + srcAxis1) * D + dStart) * rowSize;
            int64_t nextDst = ((axis0 * outerDim + axis1) * D + dStart) * rowSize;
            int64_t nextCount = dCount;
            int64_t nextSlot = 1 - readySlot;
            auto nextMain = ubBase[nextSlot * slotElems];
            auto nextWrap = nextMain[mainSegmentElems];
            gatherParams.blockCount = static_cast<uint32_t>(nextCount);
            gatherParams.blockLen = mainBytes;
            gatherParams.srcStride = wrapBytes;
            DataCopyPad(nextMain, xGm[srcBase], gatherParams, padParams);
            gatherParams.blockLen = wrapBytes;
            gatherParams.srcStride = mainBytes;
            DataCopyPad(nextWrap, xGm[srcBase + mainElems], gatherParams, padParams);

            auto readyMain = ubBase[readySlot * slotElems];
            auto readyWrap = readyMain[mainSegmentElems];
            scatterParams.blockCount = static_cast<uint32_t>(readyCount);
            scatterParams.blockLen = mainBytes;
            scatterParams.dstStride = wrapBytes;
            DataCopyPad(yGm[readyDst + wrapElems], readyMain, scatterParams);
            scatterParams.blockLen = wrapBytes;
            scatterParams.dstStride = mainBytes;
            DataCopyPad(yGm[readyDst], readyWrap, scatterParams);
            SyncMTE3ToMTE2();
            SyncMTE2ToMTE3();

            readySlot = nextSlot;
            readyDst = nextDst;
            readyCount = nextCount;
            task++;
        }

        auto readyMain = ubBase[readySlot * slotElems];
        auto readyWrap = readyMain[mainSegmentElems];
        scatterParams.blockCount = static_cast<uint32_t>(readyCount);
        scatterParams.blockLen = mainBytes;
        scatterParams.dstStride = wrapBytes;
        DataCopyPad(yGm[readyDst + wrapElems], readyMain, scatterParams);
        scatterParams.blockLen = wrapBytes;
        scatterParams.dstStride = mainBytes;
        DataCopyPad(yGm[readyDst], readyWrap, scatterParams);
        SyncMTE3ToMTE2();
        return;
    }

    auto offsetTensor = offsetBuf_.Get<uint32_t>();
    constexpr int64_t BATCH_TILES = 16;
    int64_t tileStartIdx = dimNum - 2;
    int64_t H = shapes[tileStartIdx];
    int64_t W = shapes[tileStartIdx + 1];
    int64_t tileSize = outerBlockSize;
    for (int64_t h = 0; h < H; h++) {
        for (int64_t w = 0; w < W; w++) {
            int64_t srcW = w - shifts[tileStartIdx + 1];
            if (srcW < 0) {
                srcW += W;
            }
            int64_t idx = h * W + w;
            int64_t srcIdx = h * W + srcW;
            offsetTensor.SetValue(idx, static_cast<uint32_t>(srcIdx * ts));
        }
    }

    int64_t totalTiles = totalElements / tileSize;
    int64_t baseTiles = totalTiles / needCoreNum;
    int64_t remTiles = totalTiles % needCoreNum;
    int64_t myTiles = baseTiles + (coreId < remTiles ? 1 : 0);
    if (myTiles <= 0) {
        return;
    }
    int64_t tileStart = coreId < remTiles
        ? coreId * (baseTiles + 1)
        : remTiles * (baseTiles + 1) + (coreId - remTiles) * baseTiles;
    int64_t tileEnd = tileStart + myTiles;

    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(tileSize * static_cast<int64_t>(sizeof(T)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    int64_t alignElems = 32 / static_cast<int64_t>(sizeof(T));
    int64_t segmentElems = ((tileSize + alignElems - 1) / alignElems) * alignElems;
    int64_t batchElems = BATCH_TILES * segmentElems;
    int64_t tile = tileStart;
    int64_t batchTiles = tileEnd - tile;
    if (batchTiles > BATCH_TILES) batchTiles = BATCH_TILES;
    auto bufFirst = calcQueue_.AllocTensor<T>();
    for (int64_t b = 0; b < batchTiles; b++) {
        int64_t outBase = (tile + b) * tileSize;
        int64_t srcBase = 0;
        for (int64_t d = 0; d < tileStartIdx; d++) {
            int64_t coord = (outBase / strides[d]) % shapes[d];
            int64_t srcCoord = coord - shifts[d];
            if (srcCoord < 0) {
                srcCoord += shapes[d];
            }
            srcBase += srcCoord * strides[d];
        }
        DataCopyPad(bufFirst[b * segmentElems], xGm[srcBase], copyParams, padParams);
    }
    calcQueue_.EnQue(bufFirst);
    int64_t readyTile = tile;
    int64_t readyBatch = batchTiles;
    tile += batchTiles;

    while (tile < tileEnd) {
        batchTiles = tileEnd - tile;
        if (batchTiles > BATCH_TILES) batchTiles = BATCH_TILES;
        auto bufNext = calcQueue_.AllocTensor<T>();
        for (int64_t b = 0; b < batchTiles; b++) {
            int64_t outBase = (tile + b) * tileSize;
            int64_t srcBase = 0;
            for (int64_t d = 0; d < tileStartIdx; d++) {
                int64_t coord = (outBase / strides[d]) % shapes[d];
                int64_t srcCoord = coord - shifts[d];
                if (srcCoord < 0) {
                    srcCoord += shapes[d];
                }
                srcBase += srcCoord * strides[d];
            }
            DataCopyPad(bufNext[b * segmentElems], xGm[srcBase], copyParams, padParams);
        }

        auto bufReady = calcQueue_.DeQue<T>();
        auto ubInBase = bufReady;
        auto ubOutBase = bufReady[batchElems];
        SyncMTE2ToV();
        for (int64_t b = 0; b < readyBatch; b++) {
            Gather(ubOutBase[b * segmentElems], ubInBase[b * segmentElems], offsetTensor,
                   static_cast<uint32_t>(0), static_cast<uint32_t>(tileSize));
        }
        SyncVToMTE3();
        for (int64_t b = 0; b < readyBatch; b++) {
            DataCopyPad(yGm[(readyTile + b) * tileSize], ubOutBase[b * segmentElems], copyParams);
        }
        calcQueue_.FreeTensor(bufReady);
        calcQueue_.EnQue(bufNext);
        readyTile = tile;
        readyBatch = batchTiles;
        tile += batchTiles;
    }

    auto bufLast = calcQueue_.DeQue<T>();
    auto ubInBase = bufLast;
    auto ubOutBase = bufLast[batchElems];
    SyncMTE2ToV();
    for (int64_t b = 0; b < readyBatch; b++) {
        Gather(ubOutBase[b * segmentElems], ubInBase[b * segmentElems], offsetTensor,
               static_cast<uint32_t>(0), static_cast<uint32_t>(tileSize));
    }
    SyncVToMTE3();
    for (int64_t b = 0; b < readyBatch; b++) {
        DataCopyPad(yGm[(readyTile + b) * tileSize], ubOutBase[b * segmentElems], copyParams);
    }
    calcQueue_.FreeTensor(bufLast);
}

template <typename T>
__aicore__ inline void Roll<T>::ProcessChunkedTileGather()
{
    int64_t innerIdx = innerShiftIdx;
    int64_t outerIdx = innerIdx - 2;
    int64_t gapIdx = innerIdx - 1;
    if (outerIdx >= 0 && gapIdx >= 0 && shapes[gapIdx] > shapes[innerIdx]
        && shapes[gapIdx] % shapes[innerIdx] == 0 && rowMultiplier > 1) {
        int64_t splitDim2B = shapes[innerIdx];
        int64_t splitDim2A = shapes[gapIdx] / splitDim2B;
        int64_t tileSize = outerBlockSize;
        if (subBlockSize <= 0 || splitDim2B <= 0 || splitDim2A <= 0) {
            return;
        }

        int64_t batchTilesMax = subBlockSize;
        int64_t innerDim = shapes[innerIdx];
        int64_t axis1Size = shapes[outerIdx];
        int64_t axis1Shift = shifts[outerIdx];
        int64_t totalTiles = totalElements / tileSize;
        int64_t baseTiles = totalTiles / needCoreNum;
        int64_t remTiles = totalTiles % needCoreNum;
        int64_t myTiles = baseTiles + (coreId < remTiles ? 1 : 0);
        if (myTiles <= 0) {
            return;
        }
        int64_t tileStart = coreId < remTiles
            ? coreId * (baseTiles + 1)
            : remTiles * (baseTiles + 1) + (coreId - remTiles) * baseTiles;
        int64_t tileEnd = tileStart + myTiles;

        auto offsetTensor = offsetBuf_.Get<uint32_t>();
        int ts = static_cast<int>(sizeof(T));
        for (int64_t i2b = 0; i2b < splitDim2B; i2b++) {
            for (int64_t inner = 0; inner < innerDim; inner++) {
                int64_t srcInner = inner - shifts[innerIdx];
                if (srcInner < 0) {
                    srcInner += innerDim;
                }
                for (int64_t tail = 0; tail < rowMultiplier; tail++) {
                    int64_t idx = (i2b * innerDim + inner) * rowMultiplier + tail;
                    int64_t srcIdx = (i2b * innerDim + srcInner) * rowMultiplier + tail;
                    offsetTensor.SetValue(idx, static_cast<uint32_t>(srcIdx * ts));
                }
            }
        }

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(tileSize * static_cast<int64_t>(sizeof(T)));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        int64_t alignElems = 32 / static_cast<int64_t>(sizeof(T));
        int64_t segmentElems = ((tileSize + alignElems - 1) / alignElems) * alignElems;

        int64_t tile = tileStart;
        int64_t batchTiles = tileEnd - tile;
        if (batchTiles > batchTilesMax) batchTiles = batchTilesMax;
        auto inputFirst = inQueue.template AllocTensor<T>();
        int64_t i2a = tile % splitDim2A;
        int64_t tmp = tile / splitDim2A;
        int64_t boundaryTiles = splitDim2A - i2a;
        if (batchTiles > boundaryTiles) batchTiles = boundaryTiles;
        int64_t i1 = tmp % axis1Size;
        int64_t i0 = tmp / axis1Size;
        int64_t src1 = i1 - axis1Shift;
        if (src1 < 0) {
            src1 += axis1Size;
        }
        int64_t srcTileBase = (i0 * axis1Size + src1) * splitDim2A;
        int64_t srcTile = srcTileBase + i2a;
        copyParams.blockCount = static_cast<uint32_t>(batchTiles);
        DataCopyPad(inputFirst, xGm[srcTile * tileSize], copyParams, padParams);
        inQueue.EnQue(inputFirst);

        int64_t readyTile = tile;
        int64_t readyBatch = batchTiles;
        int64_t outputTile = 0;
        int64_t outputBatch = 0;
        bool hasOutput = false;
        tile += batchTiles;

        while (tile < tileEnd) {
            batchTiles = tileEnd - tile;
            if (batchTiles > batchTilesMax) batchTiles = batchTilesMax;
            auto inputNext = inQueue.template AllocTensor<T>();
            i2a += readyBatch;
            if (i2a == splitDim2A) {
                i2a = 0;
                i1++;
                if (i1 == axis1Size) {
                    i1 = 0;
                    i0++;
                }
                src1 = i1 - axis1Shift;
                if (src1 < 0) {
                    src1 += axis1Size;
                }
                srcTileBase = (i0 * axis1Size + src1) * splitDim2A;
            }
            boundaryTiles = splitDim2A - i2a;
            if (batchTiles > boundaryTiles) batchTiles = boundaryTiles;
            srcTile = srcTileBase + i2a;
            copyParams.blockCount = static_cast<uint32_t>(batchTiles);
            DataCopyPad(inputNext, xGm[srcTile * tileSize], copyParams, padParams);
            inQueue.EnQue(inputNext);

            auto inputReady = inQueue.template DeQue<T>();
            auto outputReady = outQueue_.template AllocTensor<T>();
            for (int64_t b = 0; b < readyBatch; b++) {
                Gather(outputReady[b * segmentElems], inputReady[b * segmentElems], offsetTensor,
                       static_cast<uint32_t>(0), static_cast<uint32_t>(tileSize));
            }
            outQueue_.template EnQue<T>(outputReady);
            inQueue.FreeTensor(inputReady);

            if (hasOutput) {
                auto outputLocal = outQueue_.template DeQue<T>();
                copyParams.blockCount = static_cast<uint32_t>(outputBatch);
                DataCopyPad(yGm[outputTile * tileSize], outputLocal, copyParams);
                outQueue_.FreeTensor(outputLocal);
            }

            outputTile = readyTile;
            outputBatch = readyBatch;
            hasOutput = true;
            readyTile = tile;
            readyBatch = batchTiles;
            tile += batchTiles;
        }

        auto inputLast = inQueue.template DeQue<T>();
        auto outputLast = outQueue_.template AllocTensor<T>();
        for (int64_t b = 0; b < readyBatch; b++) {
            Gather(outputLast[b * segmentElems], inputLast[b * segmentElems], offsetTensor,
                   static_cast<uint32_t>(0), static_cast<uint32_t>(tileSize));
        }
        outQueue_.template EnQue<T>(outputLast);
        inQueue.FreeTensor(inputLast);

        if (hasOutput) {
            auto outputLocal = outQueue_.template DeQue<T>();
            copyParams.blockCount = static_cast<uint32_t>(outputBatch);
            DataCopyPad(yGm[outputTile * tileSize], outputLocal, copyParams);
            outQueue_.FreeTensor(outputLocal);
        }

        auto outputLocal = outQueue_.template DeQue<T>();
        copyParams.blockCount = static_cast<uint32_t>(readyBatch);
        DataCopyPad(yGm[readyTile * tileSize], outputLocal, copyParams);
        outQueue_.FreeTensor(outputLocal);
        return;
    }

    int64_t splitDim2B = shapes[innerIdx];
    int64_t splitDim2A = shapes[gapIdx] / splitDim2B;
    int64_t tileSize = outerBlockSize;
    if (subBlockSize <= 0 || splitDim2B <= 0 || splitDim2A <= 0) {
        return;
    }

    int64_t batchTilesMax = subBlockSize;
    int64_t d3 = shapes[innerIdx];
    int64_t d4 = rowMultiplier;
    int64_t outerDim = shapes[outerIdx];
    int64_t totalTiles = totalElements / tileSize;
    int64_t baseTiles = totalTiles / needCoreNum;
    int64_t remTiles = totalTiles % needCoreNum;
    int64_t myTiles = baseTiles + (coreId < remTiles ? 1 : 0);
    if (myTiles <= 0) {
        return;
    }
    int64_t tileStart = coreId < remTiles
        ? coreId * (baseTiles + 1)
        : remTiles * (baseTiles + 1) + (coreId - remTiles) * baseTiles;
    int64_t tileEnd = tileStart + myTiles;

    auto offsetTensor = offsetBuf_.Get<uint32_t>();
    int ts = static_cast<int>(sizeof(T));
    for (int64_t i2b = 0; i2b < splitDim2B; i2b++) {
        for (int64_t i3 = 0; i3 < d3; i3++) {
            int64_t src3 = i3 - shifts[innerIdx];
            if (src3 < 0) {
                src3 += d3;
            }
            for (int64_t i4 = 0; i4 < d4; i4++) {
                int64_t idx = (i2b * d3 + i3) * d4 + i4;
                int64_t srcIdx = (i2b * d3 + src3) * d4 + i4;
                offsetTensor.SetValue(idx, static_cast<uint32_t>(srcIdx * ts));
            }
        }
    }

    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(tileSize * static_cast<int64_t>(sizeof(T)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    int64_t alignElems = 32 / static_cast<int64_t>(sizeof(T));
    int64_t segmentElems = ((tileSize + alignElems - 1) / alignElems) * alignElems;

    int64_t tile = tileStart;
    int64_t batchTiles = tileEnd - tile;
    if (batchTiles > batchTilesMax) batchTiles = batchTilesMax;
    int64_t boundaryTiles = splitDim2A - (tile % splitDim2A);
    if (batchTiles > boundaryTiles) batchTiles = boundaryTiles;
    auto inputFirst = inQueue.template AllocTensor<T>();
    int64_t i2a = tile % splitDim2A;
    int64_t tmp = tile / splitDim2A;
    int64_t i1 = tmp % outerDim;
    int64_t i0 = tmp / outerDim;
    int64_t src1 = i1 - shifts[outerIdx];
    if (src1 < 0) {
        src1 += outerDim;
    }
    int64_t srcTile = (i0 * outerDim + src1) * splitDim2A + i2a;
    copyParams.blockCount = static_cast<uint32_t>(batchTiles);
    DataCopyPad(inputFirst, xGm[srcTile * tileSize], copyParams, padParams);
    inQueue.EnQue(inputFirst);

    int64_t readyTile = tile;
    int64_t readyBatch = batchTiles;
    int64_t outputTile = 0;
    int64_t outputBatch = 0;
    bool hasOutput = false;
    tile += batchTiles;

    while (tile < tileEnd) {
        batchTiles = tileEnd - tile;
        if (batchTiles > batchTilesMax) batchTiles = batchTilesMax;
        boundaryTiles = splitDim2A - (tile % splitDim2A);
        if (batchTiles > boundaryTiles) batchTiles = boundaryTiles;
        auto inputNext = inQueue.template AllocTensor<T>();
        i2a = tile % splitDim2A;
        tmp = tile / splitDim2A;
        i1 = tmp % outerDim;
        i0 = tmp / outerDim;
        src1 = i1 - shifts[outerIdx];
        if (src1 < 0) {
            src1 += outerDim;
        }
        srcTile = (i0 * outerDim + src1) * splitDim2A + i2a;
        copyParams.blockCount = static_cast<uint32_t>(batchTiles);
        DataCopyPad(inputNext, xGm[srcTile * tileSize], copyParams, padParams);
        inQueue.EnQue(inputNext);

        auto inputReady = inQueue.template DeQue<T>();
        auto outputReady = outQueue_.template AllocTensor<T>();
        for (int64_t b = 0; b < readyBatch; b++) {
            Gather(outputReady[b * segmentElems], inputReady[b * segmentElems], offsetTensor,
                   static_cast<uint32_t>(0), static_cast<uint32_t>(tileSize));
        }
        outQueue_.template EnQue<T>(outputReady);
        inQueue.FreeTensor(inputReady);

        if (hasOutput) {
            auto outputLocal = outQueue_.template DeQue<T>();
            copyParams.blockCount = static_cast<uint32_t>(outputBatch);
            DataCopyPad(yGm[outputTile * tileSize], outputLocal, copyParams);
            outQueue_.FreeTensor(outputLocal);
        }

        outputTile = readyTile;
        outputBatch = readyBatch;
        hasOutput = true;
        readyTile = tile;
        readyBatch = batchTiles;
        tile += batchTiles;
    }

    auto inputLast = inQueue.template DeQue<T>();
    auto outputLast = outQueue_.template AllocTensor<T>();
    for (int64_t b = 0; b < readyBatch; b++) {
        Gather(outputLast[b * segmentElems], inputLast[b * segmentElems], offsetTensor,
               static_cast<uint32_t>(0), static_cast<uint32_t>(tileSize));
    }
    outQueue_.template EnQue<T>(outputLast);
    inQueue.FreeTensor(inputLast);

    if (hasOutput) {
        auto outputLocal = outQueue_.template DeQue<T>();
        copyParams.blockCount = static_cast<uint32_t>(outputBatch);
        DataCopyPad(yGm[outputTile * tileSize], outputLocal, copyParams);
        outQueue_.FreeTensor(outputLocal);
    }

    auto outputLocal = outQueue_.template DeQue<T>();
    copyParams.blockCount = static_cast<uint32_t>(readyBatch);
    DataCopyPad(yGm[readyTile * tileSize], outputLocal, copyParams);
    outQueue_.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Roll<T>::ProcessFusedGather()
{
    switch (subMode) {
        case 13: ProcessGatherSupplement(); break;
        case 11: ProcessGatherSupplement(); break;
        case 10: ProcessChunkedTileGather(); break;
        case 7: ProcessMultiDimTileGather(); break;
        case 6: ProcessMultiDimPlaneGather(); break;
        case 5: ProcessMultiDimNDUnit(); break;
        case 3:
            ProcessMultiDimNDReorder();
            break;
        case 12: ProcessPairedRowTile(); break;
        case 2: ProcessMultiDim2DReorder(); break;
        case 1: ProcessMultiDimUBReorder(); break;
        default: ProcessMultiDim(); break;
    }
}

template <typename T>
__aicore__ inline void Roll<T>::Process()
{
    if (coreId >= needCoreNum) {
        return;
    }

    if (tilingMode == 2) {
        ProcessFusedGather();
        return;
    }

    if (tilingMode == 0) {
        switch (subMode) {
            case 8: ProcessMode0TinyWholeGather(); return;
            case 4: ProcessMode0TailSmall(); return;
            case 2: ProcessMode0UBReorderGather(); return;
            case 1: ProcessMode0UBReorder(); return;
            default: break;
        }
    }

    int64_t numElements = (coreId != needCoreNum - 1) ? numEachCore : lastCoreNum;
    int64_t coreStart = numEachCore * coreId;
    int64_t coreEnd = coreStart + numElements;

    int64_t groupSize = inNum * afterNum;

    int64_t outPos = coreStart;
    while (outPos < coreEnd) {
        int64_t b = outPos / groupSize;
        int64_t rem = outPos - b * groupSize;
        int64_t outI = rem / afterNum;
        int64_t k = rem - outI * afterNum;

        int64_t srcI = (outI >= shift) ? (outI - shift) : (outI - shift + inNum);
        int64_t srcPos = (b * inNum + srcI) * afterNum + k;

        // Region boundary: outI in [shift, inNum) maps to srcI in [0, inNum-shift)
        //                  outI in [0, shift) maps to srcI in [inNum-shift, inNum)
        // Within a region, srcI advances 1:1 with outI, so a contiguous output run
        // corresponds to a contiguous input run.
        int64_t outIEnd = (outI >= shift) ? inNum : shift;
        int64_t maxChunkInGroup = (outIEnd * afterNum) - rem;

        int64_t chunkSize = maxChunkInGroup;
        if (chunkSize > coreEnd - outPos) {
            chunkSize = coreEnd - outPos;
        }
        // Do NOT cap chunkSize by ubTensorSize here; let CopyChunk internally split
        // into ubTensorSize tiles so that double-buffer can be applied across tiles.

        CopyChunk(srcPos, outPos, chunkSize);
        outPos += chunkSize;
    }
}

} // namespace NsRoll

#endif // ROLL_H
