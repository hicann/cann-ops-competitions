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
 * \file roll_tiling_data.h
 * \brief roll tiling data struct, align with TBE roll.py tiling params
 */

#ifndef __ROLL_TILING_DATA_H__
#define __ROLL_TILING_DATA_H__

struct RollTilingData {
    // Tiling params from TBE (indices 0-11 used, 12-13 reserved)
    int64_t tilingMode;       // 0=TILING_MODE0, 1=TILING_MODE1, 2=TILING_MODE2
    int64_t needCoreNum;      // cores needed for this task
    int64_t numEachCore;      // elements per core (non-last)
    int64_t lastCoreNum;      // elements for last core
    int64_t inNum;            // total elements in target dimension
    int64_t afterNum;         // stride / elements after target dimension
    int64_t shift;            // shift value
    int64_t n;                // N dimension size
    int64_t chwNum;           // C * H * W
    int64_t subMode;          // Unified sub-mode:
                              //   MODE0: 0=normal, 1=UBReorder DMA, 2=UBReorder Gather
                              //   MODE2: 0=normal, 1=UBReorder, 2=2DReorder, 3=NDReorder
    int64_t ubTensorSize;     // UB tensor size in elements
    int64_t dataEachBlock;    // elements per 32B block
    int64_t alignNum;         // alignment number (16 or 32)
    int64_t totalElements;    // total elements
    int64_t aiCoreNum;        // total AI core num
    int64_t remCoreNum;       // first N cores get one extra alignUnit (flat distribution)

    // Multi-dim roll fields (MODE2)
    int64_t dimNum;           // effective dim count after merge (0~8)
    int64_t shapes[8];        // per-dim size
    int64_t strides[8];       // per-dim stride
    int64_t shifts[8];        // per-dim normalized shift

    // Multi-dim UBReorder fields
    int64_t innerShiftIdx;    // innermost shift dim index, -1 if none
    int64_t rowMultiplier;    // product of dims after innerShiftIdx
    int64_t continuousRows;   // product of contiguous non-shift dims above innerShiftIdx
    int64_t lenL;             // left segment length in innerShiftIdx dim
    int64_t lenR;             // right segment length in innerShiftIdx dim
    int64_t outerBlockSize;   // total elements per outer block
    int64_t subBlockSize;     // alignment granularity for core workload split

    // Precomputed move parameters (for 2DReorder / NDReorder)
    // Each group corresponds to one quad/region. Filled in tiling, consumed in kernel.
    int64_t moveCount;           // valid groups (0~4)
    int64_t moveRowCount[4];     // blockCount for DataCopyPad (number of rows)
    int64_t moveBlockLen[4];     // blockLen for DataCopyPad (bytes)
    int64_t moveSrcStride[4];    // srcStride for DataCopyPad (bytes)
    int64_t moveDstStride[4];    // dstStride for DataCopyPad (bytes)
    int64_t moveSrcOffset[4];    // source offset (elements, relative to plane base)
    int64_t moveDstOffset[4];    // destination offset (elements, relative to plane base)

    // Deprecated: merged into subMode, kept for layout compatibility
    int64_t useMultiDimReorder_; // no longer used
    int64_t useGatherReorder_;   // no longer used

    // NDReorder outer-level precomputed parameters (filled by tiling, consumed by kernel)
    // Avoids recomputing regionOuters[4] in ProcessMultiDimNDReorder.
    int64_t regionOuterLen[4];
    int64_t regionSrcOuterOff[4];
    int64_t regionDstOuterOff[4];
};

#endif
