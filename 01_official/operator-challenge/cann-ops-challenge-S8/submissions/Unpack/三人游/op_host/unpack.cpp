#include "unpack_tiling.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t kMaxRank = 4;
constexpr uint32_t kAlignBytes = 32;
constexpr uint32_t kDefaultTileBytes = 64 * 1024;
constexpr uint64_t kCompetitionUbBytes = 192 * 1024;
constexpr int64_t kTaskModeDefault = 0;
constexpr int64_t kTaskModeSplitOuterByOutput = 1;
constexpr int64_t kTaskModeLastAxisInt32PackedRows = 2;
constexpr int64_t kTaskModeSmallInnerFloat32PackedRows = 3;
constexpr int64_t kTaskModeWideNonAlignedBfloat16Rows = 4;
constexpr int64_t kTaskModeLargeAlignedFloatOutputSlabs = 5;
constexpr int64_t kTaskModeWideAlignedBfloat16BoundRows = 6;
constexpr int64_t kTaskModeSmallInnerFloat16PackedRows = 8;
constexpr ge::DataType kSupportedTypes[] = {
    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32,
    ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL};
constexpr ge::Format kSupportedFormats[] = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

inline int64_t GetElementSize(ge::DataType dtype) {
    switch (dtype) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        case ge::DT_UINT8:
        case ge::DT_INT8:
        case ge::DT_BOOL:
            return 1;
        default:
            return -1;
    }
}

inline bool IsSupportedType(ge::DataType dtype) {
    for (const auto supportedType : kSupportedTypes) {
        if (dtype == supportedType) {
            return true;
        }
    }
    return false;
}

inline bool CanUseFastRowBatch(ge::DataType dtype, int64_t num, int64_t inner, int64_t elemSize) {
    if (dtype == ge::DT_BOOL || num <= 0 || inner <= 0 || elemSize <= 0) {
        return false;
    }
    const uint64_t rowBytes = static_cast<uint64_t>(inner) * static_cast<uint64_t>(elemSize);
    const uint64_t srcStrideBytes = static_cast<uint64_t>(num) * rowBytes;
    const uint64_t rowBlocks = rowBytes / kAlignBytes;
    const uint64_t srcGapBlocks = (static_cast<uint64_t>(num) - 1ULL) * rowBlocks;
    return rowBytes >= kAlignBytes && (rowBytes % kAlignBytes) == 0 &&
           (srcStrideBytes % kAlignBytes) == 0 &&
           rowBlocks <= static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) &&
           srcGapBlocks <= static_cast<uint64_t>(std::numeric_limits<uint16_t>::max());
}

inline bool ShouldUseDirectRowBatch(ge::DataType dtype, int64_t axis, int64_t num, int64_t inner) {
    return dtype == ge::DT_FLOAT && axis == 1 && num >= 129 && inner >= 4096;
}

inline uint32_t ChoosePreferredRowBatch(int64_t inner, uint32_t maxRowsByUb) {
    if (maxRowsByUb <= 1) {
        return 1;
    }
    if (inner == 1024) {
        return std::min<uint32_t>(maxRowsByUb, 64U);
    }
    if (inner == 4096) {
        return std::min<uint32_t>(maxRowsByUb, 32U);
    }
    if (inner == 8192) {
        return std::min<uint32_t>(maxRowsByUb, 2U);
    }
    if (inner <= 2048) {
        return std::min<uint32_t>(maxRowsByUb, 32U);
    }
    if (inner <= 8192) {
        return std::min<uint32_t>(maxRowsByUb, 16U);
    }
    return std::min<uint32_t>(maxRowsByUb, 8U);
}

inline bool ShouldSplitByOuterTask(ge::DataType dtype, int64_t axis, int64_t outer, int64_t num, int64_t inner,
                                   uint32_t coreNum) {
    if (outer <= 1 || num <= 0 || inner <= 0 || coreNum == 0) {
        return false;
    }
    const int64_t totalTasks = outer * num;
    if (totalTasks < static_cast<int64_t>(coreNum) * 4) {
        return false;
    }
    if (dtype == ge::DT_FLOAT) {
        if (axis == 1 && num == 255 && inner == 8192 && outer >= 64) {
            return true;
        }
        return num < static_cast<int64_t>(coreNum) && outer >= static_cast<int64_t>(coreNum) &&
               num <= static_cast<int64_t>(coreNum / 2);
    }
    if (dtype == ge::DT_BF16) {
        if (axis != 1 || inner < 1024) {
            return false;
        }
        if (outer >= static_cast<int64_t>(coreNum) && num >= static_cast<int64_t>(coreNum)) {
            return true;
        }
        return num <= static_cast<int64_t>(coreNum) * 2;
    }
    if (dtype == ge::DT_INT32) {
        return axis == 3 && inner <= 8 && outer >= static_cast<int64_t>(coreNum);
    }
    return false;
}

inline uint32_t ChoosePreferredTaskRows(ge::DataType dtype, int64_t axis, int64_t inner, uint32_t maxRowsByUb) {
    if (maxRowsByUb <= 1 || inner <= 0) {
        return 1;
    }
    if (dtype == ge::DT_FLOAT) {
        return ChoosePreferredRowBatch(inner, maxRowsByUb);
    }
    if (dtype == ge::DT_BF16 && axis == 1) {
        if (inner % 16 != 0) {
            if (inner >= 2048) {
                return std::min<uint32_t>(maxRowsByUb, 4U);
            }
            return std::min<uint32_t>(maxRowsByUb, 8U);
        }
        if (inner >= 2048) {
            return std::min<uint32_t>(maxRowsByUb, 8U);
        }
        return std::min<uint32_t>(maxRowsByUb, 16U);
    }
    if (dtype == ge::DT_INT32 && axis == 3 && inner <= 8) {
        return std::min<uint32_t>(maxRowsByUb, 128U);
    }
    return 1;
}

inline bool CanUseLastAxisInt32PackedRows(ge::DataType dtype, int64_t axis, int64_t num, int64_t inner) {
    return dtype == ge::DT_INT32 && axis == 3 && num == 33 && inner == 1;
}

inline bool CanUseSmallInnerFloat16PackedRows(ge::DataType dtype, int64_t axis, int64_t num, int64_t inner) {
    (void)dtype;
    (void)axis;
    (void)num;
    (void)inner;
    return false;
}

inline bool CanUseSmallInnerFloat32PackedRows(ge::DataType dtype, int64_t axis, int64_t num, int64_t inner) {
    return dtype == ge::DT_FLOAT && axis == 1 && num >= 129 && inner > 0 && inner <= 4;
}

inline bool CanUseWideNonAlignedBfloat16Rows(ge::DataType dtype, int64_t axis, int64_t num, int64_t inner) {
    return dtype == ge::DT_BF16 && axis == 1 && num > 17 &&
           (((inner >= 1024 && inner < 2048) || inner == 3073) && ((inner % 16) != 0));
}

inline bool CanUseWideAlignedBfloat16BoundRows(ge::DataType dtype, int64_t axis, int64_t num, int64_t inner,
                                               int64_t elemSize) {
    return dtype == ge::DT_BF16 && axis == 1 && num > 17 && inner >= 1024 && inner < 4096 &&
           ((inner % 16) == 0) && CanUseFastRowBatch(dtype, num, inner, elemSize);
}

inline bool CanUseLargeAlignedFloatOutputSlabs(ge::DataType dtype, int64_t axis, int64_t outer, int64_t num,
                                               int64_t inner) {
    if (!(dtype == ge::DT_FLOAT && axis == 1 && num == 255 && inner >= 4096 && ((inner % 8) == 0))) {
        return false;
    }
    if (inner == 4096) {
        return outer > 64;
    }
    return false;
}

inline uint32_t ChooseLargeAlignedFloatSlabOutputs(int64_t inner) {
    if (inner <= 4096) {
        return 4U;
    }
    if (inner <= 8192) {
        return 2U;
    }
    return 2U;
}

inline uint32_t ChooseLargeAlignedFloatRowsPerTask(int64_t inner) {
    if (inner <= 8192) {
        return 2U;
    }
    return 1U;
}

inline uint32_t ChooseWideNonAlignedBfloat16PackedRowsCap(int64_t inner) {
    if (inner > 0 && inner < 2048) {
        return 32U;
    }
    if (inner >= 3072 && inner < 4096) {
        return 16U;
    }
    return 24U;
}

inline uint32_t ChooseWideAlignedBfloat16BoundRowsCap(int64_t inner) {
    if (inner <= 1024) {
        return 32U;
    }
    if (inner <= 2048) {
        return 24U;
    }
    return 16U;
}

inline bool LoadAttrs(const gert::RuntimeAttrs *attrs, int64_t &num, int64_t &axis) {
    if (attrs == nullptr) {
        return false;
    }
    const int64_t *numPtr = attrs->GetInt(0);
    if (numPtr == nullptr) {
        return false;
    }
    num = *numPtr;

    const int64_t *axisPtr = attrs->GetInt(1);
    axis = axisPtr == nullptr ? 0 : *axisPtr;
    return true;
}

inline bool NormalizeAxis(const gert::Shape &shape, int64_t &axis) {
    const int64_t rank = shape.GetDimNum();
    if (rank <= 0 || rank > static_cast<int64_t>(kMaxRank)) {
        return false;
    }
    if (axis < 0) {
        axis += rank;
    }
    return axis >= 0 && axis < rank;
}

inline bool CheckShapeAndAttrs(const gert::Shape &shape, int64_t &num, int64_t &axis) {
    if (!NormalizeAxis(shape, axis)) {
        return false;
    }
    const int64_t axisDim = shape.GetDim(axis);
    if (axisDim <= 0 || num <= 0 || num != axisDim) {
        return false;
    }
    return true;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    UnpackTilingData tiling;

    const gert::StorageShape *inputShapeHolder = context->GetInputShape(0);
    const gert::Tensor *inputTensor = context->GetInputTensor(0);
    if (inputShapeHolder == nullptr || inputTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape inputShape = inputShapeHolder->GetStorageShape();
    int64_t num = 0;
    int64_t axis = 0;
    if (!LoadAttrs(context->GetAttrs(), num, axis) || !CheckShapeAndAttrs(inputShape, num, axis)) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtype = inputTensor->GetDataType();
    const int64_t elemSize = GetElementSize(dtype);
    if (!IsSupportedType(dtype) || elemSize <= 0) {
        return ge::GRAPH_FAILED;
    }

    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        const int64_t dim = inputShape.GetDim(i);
        if (dim <= 0) {
            return ge::GRAPH_FAILED;
        }
        outer *= dim;
    }

    int64_t inner = 1;
    for (int64_t i = axis + 1; i < inputShape.GetDimNum(); ++i) {
        const int64_t dim = inputShape.GetDim(i);
        if (dim <= 0) {
            return ge::GRAPH_FAILED;
        }
        inner *= dim;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = std::max(1U, ascendcPlatform.GetCoreNumAiv());
    tiling.set_axis(axis);
    tiling.set_num(num);
    tiling.set_elem_size(elemSize);
    tiling.set_outer(outer);
    tiling.set_inner(inner);
    tiling.set_output_elems(outer * inner);
    const uint32_t elemPerBlock = std::max<uint32_t>(1U, kAlignBytes / static_cast<uint32_t>(elemSize));
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0 || ubSize > kCompetitionUbBytes) {
        ubSize = kCompetitionUbBytes;
    }
    const bool canUseFastRowBatch = CanUseFastRowBatch(dtype, num, inner, elemSize);
    const bool directRowBatch = canUseFastRowBatch && ShouldUseDirectRowBatch(dtype, axis, num, inner);
    const uint64_t queueBufferFactor = directRowBatch ? 2ULL : 4ULL;
    uint32_t maxTileByUb = static_cast<uint32_t>(std::max<uint64_t>(
        elemPerBlock, ubSize / (queueBufferFactor * static_cast<uint64_t>(elemSize))));
    maxTileByUb = (maxTileByUb / elemPerBlock) * elemPerBlock;
    uint32_t defaultTile = std::max<uint32_t>(elemPerBlock, kDefaultTileBytes / static_cast<uint32_t>(elemSize));
    defaultTile = (defaultTile / elemPerBlock) * elemPerBlock;
    uint32_t tileElems = 0;
    const bool splitByOuterTask = ShouldSplitByOuterTask(dtype, axis, outer, num, inner, coreNum);
    if (canUseFastRowBatch) {
        const uint32_t innerU32 = static_cast<uint32_t>(inner);
        const uint32_t maxRowsByUb = std::max<uint32_t>(1U, maxTileByUb / innerU32);
        uint32_t preferredRows = std::max<uint32_t>(1U, defaultTile / innerU32);
        preferredRows = std::max<uint32_t>(preferredRows, ChoosePreferredRowBatch(inner, maxRowsByUb));
        const uint32_t rowBatch = std::max<uint32_t>(1U, std::min(maxRowsByUb, preferredRows));
        tileElems = rowBatch * innerU32;
    } else {
        tileElems = std::max<uint32_t>(
            elemPerBlock, std::min<uint32_t>(static_cast<uint32_t>(inner), std::min(defaultTile, maxTileByUb)));
    }

    if (splitByOuterTask && inner > 0) {
        const uint32_t innerU32 = static_cast<uint32_t>(inner);
        const uint32_t maxRowsByUb = std::max<uint32_t>(1U, maxTileByUb / innerU32);
        const uint32_t preferredTaskRows = ChoosePreferredTaskRows(dtype, axis, inner, maxRowsByUb);
        tileElems = std::max<uint32_t>(tileElems, preferredTaskRows * innerU32);
        if (dtype == ge::DT_FLOAT && axis == 1 && num == 255 && inner == 8192 && outer >= 64) {
            tileElems = innerU32;
        }
    }

    int64_t taskMode = splitByOuterTask ? kTaskModeSplitOuterByOutput : kTaskModeDefault;
    int64_t totalTasks = num;
    if (CanUseSmallInnerFloat16PackedRows(dtype, axis, num, inner)) {
        taskMode = kTaskModeSmallInnerFloat16PackedRows;
        const uint64_t rowBytes = static_cast<uint64_t>(inner) * static_cast<uint64_t>(elemSize);
        const uint64_t paddedRowBytes = ((rowBytes + kAlignBytes - 1ULL) / kAlignBytes) * kAlignBytes;
        uint32_t packedRows =
            static_cast<uint32_t>(std::max<uint64_t>(1, ubSize / std::max<uint64_t>(1, 2ULL * paddedRowBytes)));
        packedRows = std::min<uint32_t>(packedRows, static_cast<uint32_t>(outer));
        tileElems = std::max<uint32_t>(static_cast<uint32_t>(inner), packedRows * static_cast<uint32_t>(inner));
        const int64_t rowsPerTask = std::max<int64_t>(1, tileElems / static_cast<uint32_t>(inner));
        totalTasks = ((outer + rowsPerTask - 1) / rowsPerTask) * num;
    } else if (CanUseLastAxisInt32PackedRows(dtype, axis, num, inner)) {
        taskMode = kTaskModeLastAxisInt32PackedRows;
        const uint64_t packedRowBytes = static_cast<uint64_t>(num) * static_cast<uint64_t>(elemSize);
        const uint64_t indexBytesPerRow = static_cast<uint64_t>(2 * sizeof(int32_t));
        const uint64_t bytesPerRow = packedRowBytes + static_cast<uint64_t>(elemSize) + indexBytesPerRow;
        uint32_t packedRows = static_cast<uint32_t>(std::max<uint64_t>(1, ubSize / bytesPerRow));
        packedRows = std::min<uint32_t>(packedRows, 512U);
        tileElems = std::max<uint32_t>(static_cast<uint32_t>(inner), packedRows * static_cast<uint32_t>(inner));
        const int64_t rowsPerTask = std::max<int64_t>(1, tileElems / static_cast<uint32_t>(inner));
        totalTasks = (outer + rowsPerTask - 1) / rowsPerTask;
    } else if (CanUseSmallInnerFloat32PackedRows(dtype, axis, num, inner)) {
        taskMode = kTaskModeSmallInnerFloat32PackedRows;
        const uint64_t packedRowBytes =
            static_cast<uint64_t>(num + 1) * static_cast<uint64_t>(inner) * static_cast<uint64_t>(elemSize);
        uint32_t packedRows = static_cast<uint32_t>(std::max<uint64_t>(1, ubSize / (4ULL * packedRowBytes)));
        packedRows = std::min<uint32_t>(packedRows, 16U);
        tileElems = std::max<uint32_t>(static_cast<uint32_t>(inner), packedRows * static_cast<uint32_t>(inner));
        const int64_t rowsPerTask = std::max<int64_t>(1, tileElems / static_cast<uint32_t>(inner));
        totalTasks = (outer + rowsPerTask - 1) / rowsPerTask;
    } else if (CanUseWideAlignedBfloat16BoundRows(dtype, axis, num, inner, elemSize)) {
        taskMode = kTaskModeWideAlignedBfloat16BoundRows;
        const uint32_t innerU32 = static_cast<uint32_t>(inner);
        const uint32_t maxRowsByUb = static_cast<uint32_t>(std::max<uint64_t>(
            1, ubSize / (2ULL * static_cast<uint64_t>(inner) * static_cast<uint64_t>(elemSize))));
        const uint32_t packedRows = std::min<uint32_t>(maxRowsByUb, ChooseWideAlignedBfloat16BoundRowsCap(inner));
        tileElems = std::max<uint32_t>(innerU32, packedRows * innerU32);
        const int64_t rowsPerTask = std::max<int64_t>(1, tileElems / innerU32);
        totalTasks = ((outer + rowsPerTask - 1) / rowsPerTask) * num;
    } else if (CanUseWideNonAlignedBfloat16Rows(dtype, axis, num, inner)) {
        taskMode = kTaskModeWideNonAlignedBfloat16Rows;
        const uint64_t rowBytes = static_cast<uint64_t>(inner) * static_cast<uint64_t>(elemSize);
        const uint64_t paddedRowBytes = ((rowBytes + kAlignBytes - 1ULL) / kAlignBytes) * kAlignBytes;
        const uint32_t paddedRowElems = static_cast<uint32_t>(paddedRowBytes / static_cast<uint64_t>(elemSize));
        uint32_t packedRows = static_cast<uint32_t>(std::max<uint64_t>(
            1, ubSize / (2ULL * static_cast<uint64_t>(paddedRowElems) * static_cast<uint64_t>(elemSize))));
        packedRows = std::min<uint32_t>(packedRows, ChooseWideNonAlignedBfloat16PackedRowsCap(inner));
        tileElems = std::max<uint32_t>(static_cast<uint32_t>(inner), packedRows * static_cast<uint32_t>(inner));
        const int64_t rowsPerTask = std::max<int64_t>(1, tileElems / static_cast<uint32_t>(inner));
        totalTasks = ((outer + rowsPerTask - 1) / rowsPerTask) * num;
    } else if (CanUseLargeAlignedFloatOutputSlabs(dtype, axis, outer, num, inner)) {
        taskMode = kTaskModeLargeAlignedFloatOutputSlabs;
        const uint32_t slabOutputs = ChooseLargeAlignedFloatSlabOutputs(inner);
        const uint32_t rowsPerTask = ChooseLargeAlignedFloatRowsPerTask(inner);
        tileElems = static_cast<uint32_t>(rowsPerTask * slabOutputs * static_cast<uint32_t>(inner));
        const int64_t outGroups = (num + static_cast<int64_t>(slabOutputs) - 1) / static_cast<int64_t>(slabOutputs);
        totalTasks = ((outer + static_cast<int64_t>(rowsPerTask) - 1) / static_cast<int64_t>(rowsPerTask)) * outGroups;
    } else if (taskMode == kTaskModeSplitOuterByOutput) {
        const int64_t rowsPerTask = std::max<int64_t>(1, tileElems / static_cast<uint32_t>(inner));
        totalTasks = ((outer + rowsPerTask - 1) / rowsPerTask) * num;
    }
    if (totalTasks <= 0) {
        return ge::GRAPH_FAILED;
    }

    tiling.set_task_mode(taskMode);
    uint32_t blockDim = std::min<uint32_t>(coreNum, static_cast<uint32_t>(totalTasks));
    if (dtype == ge::DT_FLOAT16 && axis == 1 && num == 63 && inner > 0 && inner < 16) {
        blockDim = std::min<uint32_t>(blockDim, 8U);
    }
    if (blockDim == 0) {
        blockDim = 1;
    }

    const int64_t tasksPerCore = (totalTasks + static_cast<int64_t>(blockDim) - 1) / static_cast<int64_t>(blockDim);
    tiling.set_total_tasks(totalTasks);
    tiling.set_tasks_per_core(tasksPerCore);
    tiling.set_tile_elems(tileElems);
    tiling.set_direct_row_batch(directRowBatch ? 1 : 0);

    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t num = 0;
    int64_t axis = 0;
    if (!LoadAttrs(context->GetAttrs(), num, axis) || !CheckShapeAndAttrs(*inputShape, num, axis)) {
        return ge::GRAPH_FAILED;
    }

    const size_t outputCount = context->GetComputeNodeOutputNum();
    if (outputCount != static_cast<size_t>(num)) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape singleShape;
    for (int64_t i = 0; i < inputShape->GetDimNum(); ++i) {
        if (i != axis) {
            singleShape.AppendDim(inputShape->GetDim(i));
        }
    }

    for (size_t i = 0; i < outputCount; ++i) {
        gert::Shape *outputShape = context->GetOutputShape(i);
        if (outputShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *outputShape = singleShape;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const ge::DataType inputDtype = context->GetInputDataType(0);
    if (!IsSupportedType(inputDtype)) {
        return ge::GRAPH_FAILED;
    }

    int64_t num = 0;
    int64_t axis = 0;
    if (!LoadAttrs(context->GetAttrs(), num, axis)) {
        return ge::GRAPH_FAILED;
    }

    const size_t outputCount = context->GetComputeNodeOutputNum();
    if (outputCount != static_cast<size_t>(num)) {
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < outputCount; ++i) {
        if (context->SetOutputDataType(i, inputDtype) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Unpack : public OpDef {
  public:
    explicit Unpack(const char *name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({kSupportedTypes[0], kSupportedTypes[1], kSupportedTypes[2], kSupportedTypes[3],
                       kSupportedTypes[4], kSupportedTypes[5], kSupportedTypes[6], kSupportedTypes[7]})
            .Format({kSupportedFormats[0], kSupportedFormats[1], kSupportedFormats[2], kSupportedFormats[3],
                     kSupportedFormats[4], kSupportedFormats[5], kSupportedFormats[6], kSupportedFormats[7]})
            .UnknownShapeFormat({kSupportedFormats[0], kSupportedFormats[1], kSupportedFormats[2], kSupportedFormats[3],
                                 kSupportedFormats[4], kSupportedFormats[5], kSupportedFormats[6], kSupportedFormats[7]});

        this->Attr("num").AttrType(REQUIRED).Int();
        this->Attr("axis").AttrType(OPTIONAL).Int(0);

        this->Output("y")
            .ParamType(DYNAMIC)
            .DataType({kSupportedTypes[0], kSupportedTypes[1], kSupportedTypes[2], kSupportedTypes[3],
                       kSupportedTypes[4], kSupportedTypes[5], kSupportedTypes[6], kSupportedTypes[7]})
            .Format({kSupportedFormats[0], kSupportedFormats[1], kSupportedFormats[2], kSupportedFormats[3],
                     kSupportedFormats[4], kSupportedFormats[5], kSupportedFormats[6], kSupportedFormats[7]})
            .UnknownShapeFormat({kSupportedFormats[0], kSupportedFormats[1], kSupportedFormats[2], kSupportedFormats[3],
                                 kSupportedFormats[4], kSupportedFormats[5], kSupportedFormats[6], kSupportedFormats[7]});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(Unpack);
} // namespace ops
