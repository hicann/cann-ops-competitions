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
 * \file roll_tiling.cpp
 * \brief Roll tiling - unified per-position output distribution
 */

#include "log/log.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "../op_kernel/roll_tiling_data.h"
#include "../op_kernel/roll_tiling_key.h"

namespace optiling {

constexpr uint32_t BLOCK_SIZE = 32;
constexpr int64_t MAX_USE_CORE_NUM = 40;
constexpr uint32_t WS_SYS_SIZE = 512U;
constexpr int64_t MAX_UBREORDER_GROUP_SIZE = 65536;
constexpr int64_t TINY_WHOLE_GATHER_MAX_ELEMS = 64;
constexpr int64_t UB_GUARD_BYTES = 1024;
constexpr int64_t BATCH_UNITS = 8;
constexpr int64_t CHUNKED_BATCH_TILES = 10;
constexpr int64_t SUPPLEMENT_BATCH_TILES = 16;
constexpr int64_t SEGMENT_ROTATE_MAX_ROWS = 72;

struct RollCompileInfo {};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum > MAX_USE_CORE_NUM) {
        coreNum = MAX_USE_CORE_NUM;
    }
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "Invalid coreNum: %ld", coreNum), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= 0, OP_LOGE(context, "Invalid ubSize: %lu", ubSize), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static uint32_t TypeSizeOf(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT: return 4;
        case ge::DT_INT32: return 4;
        case ge::DT_UINT32: return 4;
        case ge::DT_FLOAT16: return 2;
        case ge::DT_BF16: return 2;
        case ge::DT_INT8: return 1;
        case ge::DT_UINT8: return 1;
        default: return 0;
    }
}

static uint64_t TilingKeyForDtype(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT:    return GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
        case ge::DT_FLOAT16:  return GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
        case ge::DT_INT32:    return GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_2);
        case ge::DT_UINT32:   return GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_3);
        case ge::DT_INT8:     return GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_4);
        case ge::DT_UINT8:    return GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_5);
        case ge::DT_BF16:     return GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_6);
        default:              return UINT64_MAX;
    }
}

static inline int64_t Mod(int64_t a, int64_t b)
{
    int64_t r = a % b;
    if (r < 0) {
        r += (b < 0) ? -b : b;
    }
    return r;
}

static void MergeAxes(int64_t shape[], int64_t shifts[], int64_t& dimNum)
{
    if (dimNum <= 0) {
        return;
    }
    if (dimNum == 1) {
        return;
    }
    int64_t newShape[8] = {0};
    int64_t newShifts[8] = {0};
    int64_t newSize = 0;
    int64_t currentMergeSize = 1;
    bool inMergeZone = false;
    for (int32_t i = 0; i < dimNum; ++i) {
        if (shifts[i] == 0) {
            if (!inMergeZone) {
                inMergeZone = true;
                currentMergeSize = shape[i];
            } else {
                currentMergeSize *= shape[i];
            }
            if (i == dimNum - 1 || shifts[i + 1] != 0) {
                newShape[newSize] = currentMergeSize;
                newShifts[newSize] = 0;
                newSize++;
                inMergeZone = false;
            }
        } else {
            if (inMergeZone) {
                newShape[newSize] = currentMergeSize;
                newShifts[newSize] = 0;
                newSize++;
                inMergeZone = false;
            }
            newShape[newSize] = shape[i];
            newShifts[newSize] = shifts[i];
            newSize++;
        }
    }
    for (int32_t i = 0; i < newSize; ++i) {
        shape[i] = newShape[i];
        shifts[i] = newShifts[i];
    }
    dimNum = newSize;
}

static void RemoveShapeOne(int64_t shape[], int64_t shifts[], int64_t& dimNum)
{
    if (dimNum <= 1) {
        return;
    }
    int64_t newShape[8] = {0};
    int64_t newShift[8] = {0};
    int64_t newSize = 0;
    for (int32_t i = 0; i < dimNum; ++i) {
        if (shape[i] > 1) {
            newShape[newSize] = shape[i];
            newShift[newSize] = shifts[i];
            newSize++;
        }
    }
    for (int32_t i = 0; i < newSize; ++i) {
        shape[i] = newShape[i];
        shifts[i] = newShift[i];
    }
    dimNum = newSize;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context, RollTilingData* tiling = nullptr,
                                        int64_t userWorkspaceSize = 0)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    int64_t workspaceOffset = WS_SYS_SIZE + static_cast<int64_t>(sysWorkspaceSize);
    currentWorkspace[0] = workspaceOffset + userWorkspaceSize;
    if (tiling != nullptr) {
        tiling->useMultiDimReorder_ = workspaceOffset;
        tiling->useGatherReorder_ = userWorkspaceSize;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseShapeAndAttrs(gert::TilingContext* context,
    int64_t shapes[], int64_t shifts[], int64_t& dimNum, int64_t& totalElements,
    ge::DataType& dataType, uint32_t& typeSize)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShape = inputX->GetStorageShape();
    size_t origDimNum = inputShape.GetDimNum();

    totalElements = 1;
    for (size_t i = 0; i < origDimNum; ++i) {
        int64_t d = inputShape.GetDim(i);
        OP_CHECK_IF(d < 0, OP_LOGE(context, "Invalid dim size at %zu: %ld", i, d), return ge::GRAPH_FAILED);
        totalElements *= d;
    }

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    typeSize = TypeSizeOf(dataType);
    OP_CHECK_IF(typeSize == 0, OP_LOGE(context, "Roll: unsupported data type %d", dataType), return ge::GRAPH_FAILED);

    for (size_t i = 0; i < origDimNum; ++i) {
        shapes[i] = inputShape.GetDim(i);
    }

    auto attrPtr = context->GetAttrs();
    OP_LOGI(context, "Roll attrs ptr=%p", attrPtr);

    if (attrPtr != nullptr) {
        auto shiftsCv = attrPtr->GetAttrPointer<gert::ContinuousVector>(0);
        auto dimsCv = attrPtr->GetAttrPointer<gert::ContinuousVector>(1);
        OP_LOGI(context, "Roll shiftsCv=%p size=%lu dimsCv=%p size=%lu",
                shiftsCv, shiftsCv ? shiftsCv->GetSize() : 0,
                dimsCv, dimsCv ? dimsCv->GetSize() : 0);
        if (shiftsCv != nullptr && dimsCv != nullptr) {
            int64_t shiftsSize = static_cast<int64_t>(shiftsCv->GetSize());
            int64_t dimsSize = static_cast<int64_t>(dimsCv->GetSize());
            const int64_t* shiftsData = reinterpret_cast<const int64_t*>(shiftsCv->GetData());
            const int64_t* dimsData = reinterpret_cast<const int64_t*>(dimsCv->GetData());

            if (dimsSize == 0) {
                if (shiftsSize > 0 && totalElements > 0) {
                    shifts[0] = Mod(shiftsData[0], totalElements);
                }
            } else {
                for (int64_t i = 0; i < dimsSize; i++) {
                    int64_t dim = dimsData[i];
                    if (dim < 0) {
                        dim += static_cast<int64_t>(origDimNum);
                    }
                    if (dim >= 0 && dim < static_cast<int64_t>(origDimNum) && shapes[dim] > 0) {
                        shifts[dim] = Mod(shifts[dim] + shiftsData[i], shapes[dim]);
                    }
                }
            }
        }
    }

    dimNum = static_cast<int64_t>(origDimNum);
    MergeAxes(shapes, shifts, dimNum);
    RemoveShapeOne(shapes, shifts, dimNum);
    if (dimNum == 0 && totalElements > 0) {
        dimNum = 1;
        shapes[0] = totalElements;
        shifts[0] = 0;
    }

    return ge::GRAPH_SUCCESS;
}

static void ComputeStrides(int64_t strides[], const int64_t shapes[], int64_t dimNum)
{
    if (dimNum > 0) {
        strides[dimNum - 1] = 1;
        for (int64_t i = dimNum - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shapes[i + 1];
        }
    }
}

static void FillCommonTiling(RollTilingData* tiling, int64_t totalElements, int64_t coreNum,
    uint64_t ubSize, uint32_t typeSize, int64_t dimNum, const int64_t shapes[],
    const int64_t strides[], const int64_t shifts[])
{
    tiling->moveCount = 0;
    for (int i = 0; i < 4; i++) {
        tiling->moveRowCount[i] = 0;
        tiling->moveBlockLen[i] = 0;
        tiling->moveSrcStride[i] = 0;
        tiling->moveDstStride[i] = 0;
        tiling->moveSrcOffset[i] = 0;
        tiling->moveDstOffset[i] = 0;
    }

    tiling->totalElements = totalElements;
    tiling->aiCoreNum = coreNum;
    tiling->ubTensorSize = ((static_cast<int64_t>(ubSize) - 128) / static_cast<int64_t>(typeSize) / 4 / 32) * 32;
    if (tiling->ubTensorSize <= 0) {
        tiling->ubTensorSize = 32;
    }
    tiling->dataEachBlock = BLOCK_SIZE / typeSize;
    tiling->alignNum = (typeSize == 1) ? 32 : 16;

    tiling->dimNum = dimNum;
    for (int64_t i = 0; i < dimNum; i++) {
        tiling->shapes[i] = shapes[i];
        tiling->strides[i] = strides[i];
        tiling->shifts[i] = shifts[i];
    }
}

static void DistributeCoresFlat(int64_t totalElements, int64_t coreNum,
    int64_t& needCoreNum, int64_t& numEachCore, int64_t& lastCoreNum)
{
    needCoreNum = std::min(coreNum, totalElements);
    numEachCore = totalElements / needCoreNum;
    if (numEachCore == 0) numEachCore = 1;
    if (numEachCore * needCoreNum > totalElements) {
        needCoreNum = totalElements / numEachCore;
        if (needCoreNum == 0) needCoreNum = 1;
    }
    lastCoreNum = totalElements - numEachCore * (needCoreNum - 1);
}

static void DistributeCoresByBlock(int64_t totalElements, int64_t coreNum, int64_t blockSize,
    int64_t& needCoreNum, int64_t& numEachCore, int64_t& lastCoreNum, int64_t& remCoreNum)
{
    int64_t totalBlocks = totalElements / blockSize;
    int64_t baseBlocks = totalBlocks / coreNum;
    int64_t remBlocks = totalBlocks % coreNum;

    if (baseBlocks == 0) {
        needCoreNum = totalBlocks;
        if (needCoreNum == 0) needCoreNum = 1;
        numEachCore = blockSize;
        lastCoreNum = blockSize;
        remCoreNum = 0;
    } else {
        needCoreNum = coreNum;
        numEachCore = baseBlocks * blockSize;
        lastCoreNum = numEachCore;
        remCoreNum = remBlocks;
    }
}

static bool IsUniformShape(const int64_t shapes[], int64_t dimNum)
{
    if (dimNum <= 1) {
        return true;
    }
    for (int64_t i = 1; i < dimNum; ++i) {
        if (shapes[i] != shapes[0]) {
            return false;
        }
    }
    return true;
}

static inline int64_t AlignUp(int64_t value, int64_t align)
{
    if (align <= 0) {
        return value;
    }
    return ((value + align - 1) / align) * align;
}

static inline int64_t AlignDown(int64_t value, int64_t align)
{
    if (align <= 0) {
        return value;
    }
    return (value / align) * align;
}

static inline int64_t AlignBytes(int64_t bytes)
{
    return AlignUp(bytes, static_cast<int64_t>(BLOCK_SIZE));
}

static inline int64_t AlignElemsByBytes(int64_t elems, uint32_t typeSize)
{
    return AlignBytes(elems * static_cast<int64_t>(typeSize)) / static_cast<int64_t>(typeSize);
}

static inline bool FitsUb(int64_t bytes, uint64_t ubSize)
{
    return bytes + UB_GUARD_BYTES <= static_cast<int64_t>(ubSize);
}

static int64_t FindPrevShiftIdx(const int64_t shifts[], int64_t from)
{
    for (int64_t i = from - 1; i >= 0; --i) {
        if (shifts[i] != 0) {
            return i;
        }
    }
    return -1;
}

static int64_t LargestDivisorAtMost(int64_t value, int64_t limit)
{
    if (value <= 0 || limit <= 0) {
        return 0;
    }
    if (limit > value) {
        limit = value;
    }
    for (int64_t d = limit; d >= 1; --d) {
        if (value % d == 0) {
            return d;
        }
    }
    return 1;
}

static ge::graphStatus SetupMode0(gert::TilingContext* context, RollTilingData* tiling,
    const int64_t shapes[], const int64_t shifts[], int64_t dimNum, int64_t singleShiftDim,
    int64_t totalElements, uint32_t typeSize, uint64_t ubSize, int64_t coreNum, ge::DataType dt)
{
    constexpr int64_t TAIL_SHIFT_SMALL_INNUM_GATE = 32;
    constexpr int64_t UBREORDER_MIN_SEG_BYTES = 128;

    int64_t inNumDim = (singleShiftDim >= 0) ? shapes[singleShiftDim] : totalElements;
    int64_t afterNum = 1;
    if (singleShiftDim >= 0) {
        for (int64_t i = singleShiftDim + 1; i < dimNum; i++) {
            afterNum *= shapes[i];
        }
    }
    int64_t shiftVal = (singleShiftDim >= 0) ? shifts[singleShiftDim] : 0;

    tiling->inNum = inNumDim;
    tiling->afterNum = afterNum;
    tiling->shift = shiftVal;
    tiling->tilingMode = 0;

    int64_t groupSize = inNumDim * afterNum;
    int64_t ubTensorSize = tiling->ubTensorSize;

    bool isTailShift = (singleShiftDim >= 0 && singleShiftDim == dimNum - 1);
    bool smallTailGroup = (isTailShift && afterNum == 1 && inNumDim <= TAIL_SHIFT_SMALL_INNUM_GATE);
    int64_t segMainElems = shiftVal * afterNum;
    int64_t segWrapElems = (inNumDim - shiftVal) * afterNum;
    int64_t minSegBytes = std::min(segMainElems, segWrapElems) * static_cast<int64_t>(typeSize);
    bool tinySeg = (shiftVal != 0 && minSegBytes < UBREORDER_MIN_SEG_BYTES);

    bool useTailSmallMode = (shiftVal != 0 && smallTailGroup);
    bool useTinyWholeGather = (shiftVal != 0 && typeSize > 1 && totalElements > 0
                               && totalElements <= TINY_WHOLE_GATHER_MAX_ELEMS
                               && (shiftVal == 1
                                   || (typeSize == 2 && afterNum == 1
                                       && totalElements >= 7 && totalElements <= 8))
                               && 2 * totalElements <= ubTensorSize);
    bool canUBReorder = (groupSize <= ubTensorSize
                         && groupSize <= MAX_UBREORDER_GROUP_SIZE
                         && shiftVal != 0);
    if (!canUBReorder && shiftVal != 0 && (smallTailGroup || tinySeg)) {
        OP_LOGI(context,
                "Roll MODE0 UBReorder gated: tail=%d smallTail=%d tinySeg=%d inNum=%ld after=%ld shift=%ld minSegBytes=%ld",
                isTailShift ? 1 : 0,
                smallTailGroup ? 1 : 0,
                tinySeg ? 1 : 0,
                inNumDim,
                afterNum,
                shiftVal,
                minSegBytes);
    }
    if (useTinyWholeGather) {
        tiling->subMode = 8;
        int64_t alignElems = BLOCK_SIZE / static_cast<int64_t>(typeSize);
        if (alignElems < 1) alignElems = 1;
        ubTensorSize = AlignUp(totalElements, alignElems);
    } else if (canUBReorder) {
        bool useGatherMode = (isTailShift && afterNum == 1 && shiftVal == 1);
        tiling->subMode = useGatherMode ? 2 : 1;
        constexpr int64_t DB_TARGET_COPY_BYTES = 2048;
        constexpr int64_t DB_MIN_BATCH_GROUPS = 16;
        constexpr int64_t DB_MAX_BATCH_GROUPS = 256;
        int64_t mainLen = shiftVal * afterNum;
        int64_t wrapLen = (inNumDim - shiftVal) * afterNum;
        int64_t mainBytes = mainLen * static_cast<int64_t>(typeSize);
        int64_t wrapBytes = wrapLen * static_cast<int64_t>(typeSize);
        int64_t maxCopyBytes = std::max(mainBytes, wrapBytes);
        int64_t batchGroups = DB_MIN_BATCH_GROUPS;
        if (maxCopyBytes > 0) {
            batchGroups = DB_TARGET_COPY_BYTES / maxCopyBytes;
            if (batchGroups < DB_MIN_BATCH_GROUPS) batchGroups = DB_MIN_BATCH_GROUPS;
            if (batchGroups > DB_MAX_BATCH_GROUPS) batchGroups = DB_MAX_BATCH_GROUPS;
        }
        int64_t bufSizeElems = 0;
        if (useGatherMode) {
            int64_t alignGroups = BLOCK_SIZE / (static_cast<int64_t>(typeSize) * inNumDim);
            if (alignGroups < 1 || (alignGroups * inNumDim * typeSize) % BLOCK_SIZE != 0) {
                alignGroups = BLOCK_SIZE / typeSize;
            }
            batchGroups = AlignDown(batchGroups, alignGroups);
            if (batchGroups < alignGroups) batchGroups = alignGroups;
            bufSizeElems = 2 * batchGroups * inNumDim;
        } else {
            int64_t bufMainBytes = AlignBytes(mainBytes);
            int64_t bufWrapBytes = AlignBytes(wrapBytes);
            int64_t elemsPerBatch = (bufMainBytes + bufWrapBytes) / typeSize;
            bufSizeElems = batchGroups * elemsPerBatch;
        }
        int64_t totalBufBytes = 2 * bufSizeElems * typeSize;
        if (FitsUb(totalBufBytes, ubSize)) {
            ubTensorSize = bufSizeElems;
        } else {
            int64_t maxBufElems = (static_cast<int64_t>(ubSize) - UB_GUARD_BYTES) / 2 / typeSize;
            ubTensorSize = AlignDown(maxBufElems, 32);
            if (ubTensorSize <= 0) ubTensorSize = 32;
        }
    } else if (useTailSmallMode) {
        tiling->subMode = 4;
    } else {
        tiling->subMode = 0;
    }

    tiling->ubTensorSize = ubTensorSize;

    int64_t needCoreNum, numEachCore, lastCoreNum, remCoreNum = 0;
    if (tiling->subMode == 8) {
        needCoreNum = 1;
        numEachCore = totalElements;
        lastCoreNum = totalElements;
        remCoreNum = 0;
    } else if (tiling->subMode >= 1 && groupSize > 0) {
        DistributeCoresByBlock(totalElements, coreNum, groupSize,
            needCoreNum, numEachCore, lastCoreNum, remCoreNum);
    } else {
        DistributeCoresFlat(totalElements, coreNum,
            needCoreNum, numEachCore, lastCoreNum);
    }

    tiling->needCoreNum = needCoreNum;
    tiling->numEachCore = numEachCore;
    tiling->lastCoreNum = lastCoreNum;
    tiling->remCoreNum = remCoreNum;

    context->SetBlockDim(needCoreNum);

    if (GetWorkspaceSize(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint64_t tilingKey = TilingKeyForDtype(dt);
    OP_CHECK_IF(tilingKey == UINT64_MAX, OP_LOGE(context, "Unsupported data type for tiling key"), return ge::GRAPH_FAILED);
    context->SetTilingKey(tilingKey);

    OP_LOGI(context, "Roll Tiling MODE0: total=%ld inNum=%ld afterNum=%ld shift=%ld subMode=%ld cores=%ld each=%ld last=%ld ubT=%ld",
            totalElements, tiling->inNum, tiling->afterNum, tiling->shift, tiling->subMode,
            tiling->needCoreNum, tiling->numEachCore, tiling->lastCoreNum, tiling->ubTensorSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetupMode2(gert::TilingContext* context, RollTilingData* tiling,
    const int64_t shapes[], const int64_t shifts[], const int64_t strides[], int64_t dimNum,
    int64_t totalElements, uint32_t typeSize, uint64_t ubSize, int64_t coreNum,
    ge::DataType dt, int64_t nonZeroShiftCount)
{
    tiling->tilingMode = 2;
    tiling->inNum = shapes[0];
    tiling->afterNum = (dimNum > 1) ? strides[0] / shapes[0] : 1;
    tiling->shift = 0;

    int64_t innerIdx = -1;
    for (int64_t i = dimNum - 1; i >= 0; i--) {
        if (shifts[i] != 0) {
            innerIdx = i;
            break;
        }
    }
    tiling->innerShiftIdx = innerIdx;

    int64_t rowMult = 1;
    if (innerIdx != -1) {
        for (int64_t i = innerIdx + 1; i < dimNum; i++) {
            rowMult *= shapes[i];
        }
    }
    tiling->rowMultiplier = rowMult;

    bool use2DReorder = false;
    if (innerIdx >= 1 && shifts[innerIdx - 1] != 0 && nonZeroShiftCount == 2) {
        int64_t H = shapes[innerIdx - 1];
        int64_t W = shapes[innerIdx];
        int64_t sW = shifts[innerIdx];
        int64_t maxWidthElems = std::max((W - sW) * rowMult, sW * rowMult);
        if (maxWidthElems > 0 && maxWidthElems <= tiling->ubTensorSize) {
            use2DReorder = true;
        }
    }

    bool useNDReorder = false;
    int64_t outerShiftIdx = -1;
    if (!use2DReorder && nonZeroShiftCount == 2 && innerIdx >= 2) {
        outerShiftIdx = FindPrevShiftIdx(shifts, innerIdx);
        if (outerShiftIdx != -1 && innerIdx - outerShiftIdx == 2) {
            int64_t fillCount = shapes[innerIdx - 1];
            int64_t maxWidthElems = std::max(shifts[innerIdx] * rowMult,
                                             (shapes[innerIdx] - shifts[innerIdx]) * rowMult);
            bool canParallel = (outerShiftIdx > 0);
            bool worthSingleCore = (fillCount >= 32 && totalElements >= coreNum * tiling->ubTensorSize);
            if (fillCount > 1 && maxWidthElems > 0 && maxWidthElems <= tiling->ubTensorSize
                && (canParallel || worthSingleCore)) {
                useNDReorder = true;
            }
        }
    }

    bool usePairedRowTile = false;
    if (!use2DReorder && useNDReorder && outerShiftIdx >= 0 && innerIdx - outerShiftIdx == 2) {
        int64_t rowStrideElems = shapes[innerIdx] * rowMult;
        int64_t leftElems = shifts[innerIdx] * rowMult;
        int64_t rightElems = (shapes[innerIdx] - shifts[innerIdx]) * rowMult;
        int64_t leftBytes = leftElems * static_cast<int64_t>(typeSize);
        int64_t rightBytes = rightElems * static_cast<int64_t>(typeSize);
        bool balancedInner = (leftElems > 0 && rightElems > 0
                              && leftElems * 2 >= rowStrideElems
                              && rightElems * 2 >= rowStrideElems);
        bool rowsUseful = (shapes[innerIdx - 1] >= coreNum || totalElements >= coreNum * tiling->ubTensorSize);
        usePairedRowTile = (balancedInner && rowsUseful
                            && leftBytes <= 65535 && rightBytes <= 65535);
    }

    int64_t ubTensorSize = tiling->ubTensorSize;
    if (use2DReorder) {
        int64_t W = shapes[innerIdx];
        int64_t sW = shifts[innerIdx];
        int64_t C = rowMult;
        int64_t ts = static_cast<int64_t>(typeSize);
        int64_t scBytes = sW * C * ts;
        int64_t wscBytes = (W - sW) * C * ts;
        int64_t alignedSC = AlignBytes(scBytes);
        int64_t alignedWSC = AlignBytes(wscBytes);
        bool scWaste = (scBytes > 0) && ((alignedSC - scBytes) * 100 > alignedSC * 30);
        bool wscWaste = (wscBytes > 0) && ((alignedWSC - wscBytes) * 100 > alignedWSC * 30);
        if (scWaste || wscWaste) {
            OP_LOGI(context, "Roll 2DReorder alignment gate blocked: sc=%ld/%ld wsc=%ld/%ld",
                    scBytes, alignedSC, wscBytes, alignedWSC);
            use2DReorder = false;
        }
    }
    if (useNDReorder) {
        int64_t W = shapes[innerIdx];
        int64_t sW = shifts[innerIdx];
        int64_t C = rowMult;
        int64_t ts = static_cast<int64_t>(typeSize);
        int64_t lBytes = sW * C * ts;
        int64_t rBytes = (W - sW) * C * ts;
        int64_t alignedL = AlignBytes(lBytes);
        int64_t alignedR = AlignBytes(rBytes);
        bool lWaste = (lBytes > 0) && ((alignedL - lBytes) * 100 > alignedL * 30);
        bool rWaste = (rBytes > 0) && ((alignedR - rBytes) * 100 > alignedR * 30);
        if (lWaste || rWaste) {
            OP_LOGI(context, "Roll NDReorder alignment gate blocked: l=%ld/%ld r=%ld/%ld",
                    lBytes, alignedL, rBytes, alignedR);
            useNDReorder = false;
        }
    }

    bool useNDUnit = false;
    bool usePlaneGather = false;
    bool useTileGather = false;
    bool useSupplementSmallTileGather = false;
    bool useSupplementLargeTileGather = false;
    bool useSupplementWholeGapRotate = false;
    bool useChunkedTileGather = false;
    if (nonZeroShiftCount == 2 && innerIdx >= 0) {
        int64_t outerIdx = -1;
        for (int64_t i = innerIdx - 1; i >= 0; --i) {
            if (shifts[i] != 0) {
                outerIdx = i;
                break;
            }
        }
        int64_t innerPlane = shapes[innerIdx] * rowMult;
        int64_t tilePlane = (innerIdx > 0) ? shapes[innerIdx - 1] * innerPlane : innerPlane;
        bool separatedByOneAxis = (outerIdx >= 0 && innerIdx - outerIdx == 2);
        int64_t gapRows = separatedByOneAxis ? shapes[innerIdx - 1] : 0;
        int64_t maxTileRows = (innerPlane > 0) ? (tiling->ubTensorSize / innerPlane) : 0;
        int64_t tileRows = LargestDivisorAtMost(gapRows, maxTileRows);
        int64_t tileElems = tileRows * innerPlane;
        int64_t chunkedTileElems = separatedByOneAxis ? shapes[innerIdx] * innerPlane : 0;
        int64_t chunkedSegmentElems = (chunkedTileElems > 0) ? AlignElemsByBytes(chunkedTileElems, typeSize) : 0;
        int64_t chunkedBatchElems = CHUNKED_BATCH_TILES * chunkedSegmentElems;
        int64_t chunkedUbBytes = 4 * chunkedBatchElems * static_cast<int64_t>(typeSize)
                                 + chunkedTileElems * static_cast<int64_t>(sizeof(uint32_t));
        int64_t tileChunkRows = ((tileRows + 2) / 3);
        int64_t tileChunkElems = tileChunkRows * innerPlane;
        int64_t tileSegmentElems = (tileChunkElems > 0) ? AlignElemsByBytes(tileChunkElems, typeSize) : 0;
        int64_t tileUbBytes = 4 * tileSegmentElems * static_cast<int64_t>(typeSize)
                              + tileElems * static_cast<int64_t>(sizeof(uint32_t));
        int64_t planeUbBytes = 4 * tilePlane * static_cast<int64_t>(typeSize)
                               + tilePlane * static_cast<int64_t>(sizeof(uint32_t));
        int64_t supplementSegmentElems = (tilePlane > 0) ? AlignElemsByBytes(tilePlane, typeSize) : 0;
        int64_t supplementUbBytes = 4 * SUPPLEMENT_BATCH_TILES * supplementSegmentElems
                                    * static_cast<int64_t>(typeSize)
                                    + tilePlane * static_cast<int64_t>(sizeof(uint32_t));
        int64_t chunkRows = std::min<int64_t>(SEGMENT_ROTATE_MAX_ROWS, gapRows);
        int64_t mainElems = (innerIdx >= 0) ? (shapes[innerIdx] - shifts[innerIdx]) * rowMult : 0;
        int64_t wrapElems = (innerIdx >= 0) ? shifts[innerIdx] * rowMult : 0;
        int64_t mainBytes = mainElems * static_cast<int64_t>(typeSize);
        int64_t wrapBytes = wrapElems * static_cast<int64_t>(typeSize);
        int64_t mainBlockElems = AlignBytes(mainBytes) / typeSize;
        int64_t wrapBlockElems = AlignBytes(wrapBytes) / typeSize;
        int64_t segmentRotateUbBytes = 2 * chunkRows * (mainBlockElems + wrapBlockElems)
                                       * static_cast<int64_t>(typeSize);
        int64_t wholeGapRotateUbBytes = gapRows * (mainBlockElems + wrapBlockElems)
                                        * static_cast<int64_t>(typeSize);
        int64_t wholeGapPrefix = (separatedByOneAxis && outerIdx > 0 && strides[outerIdx - 1] > 0)
                                 ? (totalElements / strides[outerIdx - 1]) : 1;
        int64_t wholeGapTasks = (separatedByOneAxis && outerIdx >= 0)
                                ? (wholeGapPrefix * shapes[outerIdx]) : 0;
        useNDUnit = (separatedByOneAxis && IsUniformShape(shapes, dimNum)
                     && shapes[innerIdx - 1] >= 8
                     && BATCH_UNITS * innerPlane <= tiling->ubTensorSize);
        usePlaneGather = (dt == ge::DT_BF16
                          && dimNum == 3
                          && innerIdx == dimNum - 1
                          && outerIdx == innerIdx - 1
                          && rowMult == 1
                          && shifts[0] == 0
                          && shifts[outerIdx] != 0
                          && shifts[innerIdx] != 0
                          && tilePlane <= tiling->ubTensorSize
                          && FitsUb(planeUbBytes, ubSize));
        useTileGather = (separatedByOneAxis && tileRows > 1
                         && tileRows < gapRows
                         && tileElems <= tiling->ubTensorSize
                         && shapes[innerIdx] > 1
                         && FitsUb(tileUbBytes, ubSize));
        useChunkedTileGather = (separatedByOneAxis && rowMult > 1
                                && shapes[innerIdx - 1] % shapes[innerIdx] == 0
                                && innerPlane <= tiling->ubTensorSize
                                && FitsUb(chunkedUbBytes, ubSize));
        useSupplementSmallTileGather = (separatedByOneAxis
                                        && innerIdx == dimNum - 1
                                        && rowMult == 1
                                        && shapes[innerIdx] <= 9
                                        && tilePlane <= tiling->ubTensorSize
                                        && totalElements <= coreNum * tilePlane * 8
                                        && FitsUb(supplementUbBytes, ubSize));
        useSupplementLargeTileGather = (typeSize == 4
                                        && separatedByOneAxis
                                        && rowMult > 1
                                        && shapes[innerIdx - 1] > 2 * coreNum
                                        && std::max(shifts[innerIdx] * rowMult,
                                                    (shapes[innerIdx] - shifts[innerIdx]) * rowMult)
                                           <= tiling->ubTensorSize
                                        && FitsUb(segmentRotateUbBytes, ubSize));
        useSupplementWholeGapRotate = (useSupplementLargeTileGather
                                       && gapRows > 0
                                       && wholeGapTasks > 0
                                       && wholeGapTasks <= coreNum
                                       && FitsUb(wholeGapRotateUbBytes, ubSize));
        if (useSupplementLargeTileGather) {
            useTileGather = false;
        }
    }

    if (useChunkedTileGather) {
        // Chunked tile gather family: keep the current whole-tile probe with a larger UB batch.
        tiling->subMode = 10;
        tiling->outerBlockSize = shapes[innerIdx] * shapes[innerIdx] * rowMult;
        tiling->subBlockSize = 10;
        int64_t segmentElems = AlignElemsByBytes(tiling->outerBlockSize, typeSize);
        tiling->ubTensorSize = tiling->subBlockSize * segmentElems;
    } else if (useTileGather) {
        tiling->subMode = 7;
        tiling->lenL = shifts[innerIdx];
        tiling->lenR = shifts[innerIdx - 2];
        int64_t innerPlane = shapes[innerIdx] * rowMult;
        int64_t maxTileRows = tiling->ubTensorSize / innerPlane;
        int64_t tileRows = LargestDivisorAtMost(shapes[innerIdx - 1], maxTileRows);
        tiling->outerBlockSize = tileRows * innerPlane;
        tiling->subBlockSize = ((tileRows + 2) / 3) * innerPlane;
        int64_t segmentElems = AlignElemsByBytes(tiling->subBlockSize, typeSize);
        tiling->ubTensorSize = segmentElems;
    } else if (useSupplementSmallTileGather) {
        tiling->subMode = 11;
        tiling->outerBlockSize = shapes[dimNum - 2] * shapes[dimNum - 1];
        tiling->subBlockSize = 2;
        int64_t segmentElems = AlignElemsByBytes(tiling->outerBlockSize, typeSize);
        tiling->ubTensorSize = 2 * SUPPLEMENT_BATCH_TILES * segmentElems;
    } else if (useSupplementWholeGapRotate) {
        tiling->subMode = 11;
        int64_t chunkD = shapes[innerIdx - 1];
        tiling->outerBlockSize = chunkD * shapes[innerIdx] * rowMult;
        tiling->subBlockSize = chunkD;
        int64_t mainElems = (shapes[innerIdx] - shifts[innerIdx]) * rowMult;
        int64_t wrapElems = shifts[innerIdx] * rowMult;
        int64_t mainBytes = mainElems * static_cast<int64_t>(typeSize);
        int64_t wrapBytes = wrapElems * static_cast<int64_t>(typeSize);
        int64_t mainBlockElems = AlignBytes(mainBytes) / typeSize;
        int64_t wrapBlockElems = AlignBytes(wrapBytes) / typeSize;
        tiling->ubTensorSize = chunkD * (mainBlockElems + wrapBlockElems);
    } else if (useSupplementLargeTileGather) {
        tiling->subMode = 11;
        int64_t chunkD = std::min<int64_t>(SEGMENT_ROTATE_MAX_ROWS, shapes[innerIdx - 1]);
        tiling->outerBlockSize = chunkD * shapes[innerIdx] * rowMult;
        tiling->subBlockSize = chunkD;
        int64_t mainElems = (shapes[innerIdx] - shifts[innerIdx]) * rowMult;
        int64_t wrapElems = shifts[innerIdx] * rowMult;
        int64_t mainBytes = mainElems * static_cast<int64_t>(typeSize);
        int64_t wrapBytes = wrapElems * static_cast<int64_t>(typeSize);
        int64_t mainBlockElems = AlignBytes(mainBytes) / typeSize;
        int64_t wrapBlockElems = AlignBytes(wrapBytes) / typeSize;
        tiling->ubTensorSize = 2 * chunkD * (mainBlockElems + wrapBlockElems);
    } else if (usePlaneGather) {
        tiling->subMode = 6;
        tiling->lenL = shifts[innerIdx - 1];
        tiling->lenR = shifts[innerIdx];
        tiling->outerBlockSize = shapes[innerIdx - 1] * shapes[innerIdx] * rowMult;
        tiling->subBlockSize = 8;
        tiling->ubTensorSize = tiling->outerBlockSize;
    } else if (use2DReorder) {
        tiling->subMode = 2;
        tiling->lenL = shifts[innerIdx];
        tiling->lenR = shapes[innerIdx] - shifts[innerIdx];
        tiling->continuousRows = 1;
        tiling->outerBlockSize = shapes[innerIdx - 1] * shapes[innerIdx] * rowMult;

        int64_t H = shapes[innerIdx - 1];
        int64_t W = shapes[innerIdx];
        int64_t sH = shifts[innerIdx - 1];
        int64_t sW = shifts[innerIdx];
        int64_t C = rowMult;
        int64_t lineStride = W * C;
        int64_t ts = static_cast<int64_t>(typeSize);

        tiling->moveCount = 4;
        tiling->moveRowCount[0] = H - sH;
        tiling->moveBlockLen[0] = (W - sW) * C * ts;
        tiling->moveSrcStride[0] = sW * C * ts;
        tiling->moveDstStride[0] = sW * C * ts;
        tiling->moveSrcOffset[0] = 0;
        tiling->moveDstOffset[0] = sH * lineStride + sW * C;

        tiling->moveRowCount[1] = H - sH;
        tiling->moveBlockLen[1] = sW * C * ts;
        tiling->moveSrcStride[1] = (W - sW) * C * ts;
        tiling->moveDstStride[1] = (W - sW) * C * ts;
        tiling->moveSrcOffset[1] = (W - sW) * C;
        tiling->moveDstOffset[1] = sH * lineStride;

        tiling->moveRowCount[2] = sH;
        tiling->moveBlockLen[2] = (W - sW) * C * ts;
        tiling->moveSrcStride[2] = sW * C * ts;
        tiling->moveDstStride[2] = sW * C * ts;
        tiling->moveSrcOffset[2] = (H - sH) * lineStride;
        tiling->moveDstOffset[2] = sW * C;

        tiling->moveRowCount[3] = sH;
        tiling->moveBlockLen[3] = sW * C * ts;
        tiling->moveSrcStride[3] = (W - sW) * C * ts;
        tiling->moveDstStride[3] = (W - sW) * C * ts;
        tiling->moveSrcOffset[3] = (H - sH) * lineStride + (W - sW) * C;
        tiling->moveDstOffset[3] = 0;
    } else if (useNDReorder) {
        tiling->subMode = usePairedRowTile ? 12 : 3;
        tiling->lenL = shifts[innerIdx];
        tiling->lenR = shapes[innerIdx] - shifts[innerIdx];
        tiling->continuousRows = shapes[innerIdx - 1];
        if (outerShiftIdx > 0) {
            tiling->outerBlockSize = strides[outerShiftIdx - 1];
        } else {
            tiling->outerBlockSize = totalElements;
        }
        tiling->subBlockSize = strides[outerShiftIdx];

        int64_t H = shapes[outerShiftIdx];
        int64_t W = shapes[innerIdx];
        int64_t sH = shifts[outerShiftIdx];
        int64_t sW = shifts[innerIdx];
        int64_t C = rowMult;
        int64_t ts = static_cast<int64_t>(typeSize);
        int64_t lBytes = sW * C * ts;
        int64_t rBytes = (W - sW) * C * ts;

        tiling->moveCount = 4;
        tiling->moveBlockLen[0] = lBytes;
        tiling->moveSrcStride[0] = rBytes;
        tiling->moveDstStride[0] = rBytes;
        tiling->moveSrcOffset[0] = (W - sW) * C;
        tiling->moveDstOffset[0] = 0;

        tiling->moveBlockLen[1] = rBytes;
        tiling->moveSrcStride[1] = lBytes;
        tiling->moveDstStride[1] = lBytes;
        tiling->moveSrcOffset[1] = 0;
        tiling->moveDstOffset[1] = sW * C;

        tiling->moveBlockLen[2] = lBytes;
        tiling->moveSrcStride[2] = rBytes;
        tiling->moveDstStride[2] = rBytes;
        tiling->moveSrcOffset[2] = (W - sW) * C;
        tiling->moveDstOffset[2] = 0;

        tiling->moveBlockLen[3] = rBytes;
        tiling->moveSrcStride[3] = lBytes;
        tiling->moveDstStride[3] = lBytes;
        tiling->moveSrcOffset[3] = 0;
        tiling->moveDstOffset[3] = sW * C;

        tiling->regionOuterLen[0] = sH;
        tiling->regionSrcOuterOff[0] = H - sH;
        tiling->regionDstOuterOff[0] = 0;

        tiling->regionOuterLen[1] = sH;
        tiling->regionSrcOuterOff[1] = H - sH;
        tiling->regionDstOuterOff[1] = 0;

        tiling->regionOuterLen[2] = H - sH;
        tiling->regionSrcOuterOff[2] = 0;
        tiling->regionDstOuterOff[2] = sH;

        tiling->regionOuterLen[3] = H - sH;
        tiling->regionSrcOuterOff[3] = 0;
        tiling->regionDstOuterOff[3] = sH;
    } else if (useNDUnit) {
        tiling->subMode = 5;
        tiling->lenL = shifts[innerIdx];
        tiling->lenR = shapes[innerIdx] - shifts[innerIdx];
        tiling->continuousRows = shapes[innerIdx - 1];
        tiling->outerBlockSize = shapes[innerIdx] * rowMult;
        tiling->subBlockSize = 8;
        tiling->ubTensorSize = BATCH_UNITS * tiling->outerBlockSize;
    } else if (innerIdx != -1) {
        tiling->lenL = shifts[innerIdx];
        tiling->lenR = shapes[innerIdx] - shifts[innerIdx];

        int64_t rows = 1;
        for (int64_t i = innerIdx - 1; i >= 0; i--) {
            if (shifts[i] != 0) break;
            rows *= shapes[i];
        }
        tiling->continuousRows = rows;
        tiling->outerBlockSize = rows * shapes[innerIdx] * rowMult;

        int64_t lBytes = tiling->lenL * rowMult * typeSize;
        int64_t rBytes = tiling->lenR * rowMult * typeSize;
        int64_t alignedL = AlignBytes(lBytes);
        int64_t alignedR = AlignBytes(rBytes);

        if (tiling->lenL > 0 && tiling->lenR > 0
            && (alignedL + alignedR) <= (static_cast<int64_t>(ubSize) / 4)) {
            tiling->subMode = 1;
        } else {
            tiling->subMode = 0;
        }
    } else {
        tiling->subMode = 0;
    }

    if (tiling->subMode == 3) {
        int64_t bufferDivisor = 2;
        ubTensorSize = ((static_cast<int64_t>(ubSize) - UB_GUARD_BYTES) /
                        static_cast<int64_t>(typeSize) / bufferDivisor / 32) * 32;
        if (ubTensorSize <= 0) ubTensorSize = 32;
        tiling->ubTensorSize = ubTensorSize;
    } else if (tiling->subMode == 12) {
        int64_t bufferDivisor = 1;
        ubTensorSize = ((static_cast<int64_t>(ubSize) - UB_GUARD_BYTES) /
                        static_cast<int64_t>(typeSize) / bufferDivisor / 32) * 32;
        if (ubTensorSize <= 0) ubTensorSize = 32;
        tiling->ubTensorSize = ubTensorSize;
    }

    int64_t needCoreNum, numEachCore, lastCoreNum, remCoreNum = 0;
    DistributeCoresFlat(totalElements, coreNum, needCoreNum, numEachCore, lastCoreNum);

    if (tiling->subMode == 11 && tiling->subBlockSize > 0 && rowMult > 1) {
        int64_t chunkRows = tiling->outerBlockSize / (shapes[innerIdx] * rowMult);
        if (chunkRows <= 0) chunkRows = 1;
        int64_t outerIdx = innerIdx - 2;
        int64_t prefixCount = (outerIdx > 0) ? (totalElements / strides[outerIdx - 1]) : 1;
        int64_t totalPlanes = prefixCount * shapes[outerIdx] *
                              ((shapes[innerIdx - 1] + chunkRows - 1) / chunkRows);
        needCoreNum = std::min(coreNum, totalPlanes);
        if (needCoreNum <= 0) needCoreNum = 1;
        numEachCore = tiling->outerBlockSize;
        lastCoreNum = tiling->outerBlockSize;
        remCoreNum = 0;
    } else if (tiling->subMode == 13) {
        int64_t chunkRows = tiling->outerBlockSize / (shapes[innerIdx] * rowMult);
        if (chunkRows <= 0) chunkRows = 1;
        int64_t totalPlanes = (totalElements / strides[innerIdx - 2]) * ((shapes[innerIdx - 1] + chunkRows - 1) / chunkRows);
        needCoreNum = std::min(coreNum, totalPlanes);
        if (needCoreNum <= 0) needCoreNum = 1;
        numEachCore = tiling->outerBlockSize;
        lastCoreNum = tiling->outerBlockSize;
        remCoreNum = 0;
    } else if ((tiling->subMode == 6 || tiling->subMode == 7 || tiling->subMode == 10 || tiling->subMode == 11) && tiling->outerBlockSize > 0) {
        int64_t totalPlanes = totalElements / tiling->outerBlockSize;
        needCoreNum = std::min(coreNum, totalPlanes);
        if (needCoreNum <= 0) needCoreNum = 1;
        numEachCore = tiling->outerBlockSize;
        lastCoreNum = tiling->outerBlockSize;
        remCoreNum = 0;
    } else if ((tiling->subMode == 1 || tiling->subMode == 2 || tiling->subMode == 3 || tiling->subMode == 12)
        && tiling->outerBlockSize > 0) {
        if (tiling->subMode == 1) {
            needCoreNum = coreNum;
            remCoreNum = 0;
        } else {
            int64_t alignSize = tiling->outerBlockSize;
            if ((tiling->subMode == 3 || tiling->subMode == 12) && tiling->subBlockSize > 0) {
                alignSize = tiling->subBlockSize;
            }

            if (tiling->subMode == 2) {
                int64_t totalBlocks = totalElements / alignSize;
                int64_t baseBlocks = totalBlocks / coreNum;
                int64_t remBlocks = totalBlocks % coreNum;

                if (baseBlocks == 0) {
                    needCoreNum = totalBlocks;
                    if (needCoreNum == 0) needCoreNum = 1;
                    numEachCore = alignSize;
                    lastCoreNum = alignSize;
                    remCoreNum = 0;
                } else {
                    needCoreNum = coreNum;
                    numEachCore = baseBlocks * alignSize;
                    lastCoreNum = numEachCore;
                    remCoreNum = remBlocks;
                }
            } else {
                numEachCore = (numEachCore / alignSize) * alignSize;
                if (numEachCore == 0) numEachCore = alignSize;
                needCoreNum = totalElements / numEachCore;
                if (needCoreNum == 0) needCoreNum = 1;
                if (needCoreNum > coreNum) {
                    numEachCore = (totalElements / coreNum / alignSize) * alignSize;
                    if (numEachCore == 0) numEachCore = alignSize;
                    needCoreNum = coreNum;
                }
                if (numEachCore * needCoreNum > totalElements) {
                    needCoreNum = totalElements / numEachCore;
                    if (needCoreNum == 0) needCoreNum = 1;
                }
                lastCoreNum = totalElements - numEachCore * (needCoreNum - 1);
                remCoreNum = 0;
            }
        }
    }

    tiling->needCoreNum = needCoreNum;
    tiling->numEachCore = numEachCore;
    tiling->lastCoreNum = lastCoreNum;
    tiling->remCoreNum = remCoreNum;

    context->SetBlockDim(needCoreNum);

    int64_t userWorkspaceSize = 0;

    if (GetWorkspaceSize(context, tiling, userWorkspaceSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint64_t tilingKey = TilingKeyForDtype(dt);
    OP_CHECK_IF(tilingKey == UINT64_MAX, OP_LOGE(context, "Unsupported data type for tiling key"), return ge::GRAPH_FAILED);
    context->SetTilingKey(tilingKey);

    OP_LOGI(context, "Roll Tiling MODE2: total=%ld dimNum=%ld subMode=%ld cores=%ld each=%ld last=%ld ubT=%ld",
            totalElements, dimNum, tiling->subMode,
            tiling->needCoreNum, tiling->numEachCore, tiling->lastCoreNum, tiling->ubTensorSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RollTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"),
                return ge::GRAPH_FAILED);

    int64_t shapes[8] = {0};
    int64_t shifts[8] = {0};
    int64_t strides[8] = {0};
    int64_t dimNum = 0;
    int64_t totalElements = 0;
    ge::DataType dataType;
    uint32_t typeSize = 0;

    OP_CHECK_IF(ParseShapeAndAttrs(context, shapes, shifts, dimNum, totalElements, dataType, typeSize)
                != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ParseShapeAndAttrs error"),
                return ge::GRAPH_FAILED);

    ComputeStrides(strides, shapes, dimNum);

    RollTilingData* tiling = context->GetTilingData<RollTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(RollTilingData), 0, sizeof(RollTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"),
                return ge::GRAPH_FAILED);

    FillCommonTiling(tiling, totalElements, coreNum, ubSize, typeSize, dimNum, shapes, strides, shifts);

    if (totalElements == 0) {
        tiling->subMode = 0;
        tiling->needCoreNum = 1;
        tiling->numEachCore = 0;
        tiling->lastCoreNum = 0;
        context->SetBlockDim(1);
        if (GetWorkspaceSize(context) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        context->SetTilingKey(TilingKeyForDtype(dataType));
        return ge::GRAPH_SUCCESS;
    }

    int64_t nonZeroShiftCount = 0;
    int64_t singleShiftDim = -1;
    for (int64_t i = 0; i < dimNum; i++) {
        if (shifts[i] != 0) {
            nonZeroShiftCount++;
            singleShiftDim = i;
        }
    }

    if (nonZeroShiftCount <= 1) {
        OP_CHECK_IF(SetupMode0(context, tiling, shapes, shifts, dimNum, singleShiftDim,
                               totalElements, typeSize, ubSize, coreNum, dataType)
                    != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "SetupMode0 error"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(SetupMode2(context, tiling, shapes, shifts, strides, dimNum,
                               totalElements, typeSize, ubSize, coreNum, dataType, nonZeroShiftCount)
                    != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "SetupMode2 error"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRoll([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Roll).Tiling(RollTilingFunc).TilingParse<RollCompileInfo>(TilingParseForRoll);

} // namespace optiling
