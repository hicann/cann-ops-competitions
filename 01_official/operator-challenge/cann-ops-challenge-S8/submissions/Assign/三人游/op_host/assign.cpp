#include "assign_tiling.h"

#include <algorithm>
#include <cstdlib>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t kMaxRank = 8;
constexpr int64_t kBroadcastModeGeneric = 0;
constexpr int64_t kBroadcastModeScalar = 1;
constexpr int64_t kBroadcastModeLastDimContiguous = 2;
constexpr int64_t kSameShapeMinBytesPerCore = 6144;
constexpr int64_t kSameShapeReservedUbBytes = 8 * 1024;
constexpr ge::DataType kSupportedTypes[] = {
    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32,
    ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL};
constexpr ge::Format kSupportedFormats[] = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

inline int64_t GetDimFromRight(const gert::Shape &shape, size_t alignedIndex, size_t rank) {
    const size_t shapeRank = static_cast<size_t>(shape.GetDimNum());
    if (alignedIndex + shapeRank < rank) {
        return 1;
    }
    return shape.GetDim(static_cast<int64_t>(alignedIndex + shapeRank - rank));
}

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

inline int64_t GetCoreNum(gert::TilingContext *context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }
    return coreNum;
}

inline int64_t GetUbSize(gert::TilingContext *context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    return ubSize == 0 ? 196352 : static_cast<int64_t>(ubSize);
}

inline int64_t ClampCoreNum(int64_t coreNum, int64_t maxCoreNum, int64_t totalElems) {
    if (maxCoreNum > 0 && coreNum > maxCoreNum) {
        coreNum = maxCoreNum;
    }
    if (totalElems > 0 && coreNum > totalElems) {
        coreNum = totalElems;
    }
    if (coreNum <= 0) {
        coreNum = 1;
    }
    return coreNum;
}

inline int64_t GetEnvInt64(const char *name, int64_t defaultValue) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    char *end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0) {
        return defaultValue;
    }
    return static_cast<int64_t>(parsed);
}

inline int64_t CeilDiv(int64_t value, int64_t divisor) {
    return (value + divisor - 1) / divisor;
}

inline int64_t AlignDown(int64_t value, int64_t align) {
    return (value / align) * align;
}

inline int64_t SelectBalancedCoreNum(int64_t totalUbBlocks, int64_t targetCoreNum) {
    if (totalUbBlocks <= 0 || targetCoreNum <= 1) {
        return 1;
    }
    int64_t bestCoreNum = 1;
    int64_t bestCost = (totalUbBlocks - 1) * 4 + (targetCoreNum - 1);
    for (int64_t candidate = 2; candidate <= targetCoreNum; ++candidate) {
        const int64_t remainder = totalUbBlocks % candidate;
        const int64_t corePenalty = targetCoreNum - candidate;
        const int64_t cost = remainder * 4 + corePenalty;
        if (cost < bestCost || (cost == bestCost && candidate > bestCoreNum)) {
            bestCost = cost;
            bestCoreNum = candidate;
        }
    }
    return bestCoreNum;
}

inline bool CanCollapseDims(int64_t outDim0, int64_t outDim1, int64_t otherDim0, int64_t otherDim1) {
    const bool firstBroadcast = (otherDim0 == 1);
    const bool secondBroadcast = (otherDim1 == 1);
    if (firstBroadcast != secondBroadcast) {
        return false;
    }
    if (!firstBroadcast && (otherDim0 != outDim0 || otherDim1 != outDim1)) {
        return false;
    }
    return true;
}

} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    AssignTilingData tiling;

    const gert::Shape inputShape = context->GetInputShape(0)->GetStorageShape();
    const gert::Shape otherShape = context->GetInputShape(1)->GetStorageShape();
    const size_t inputRank = static_cast<size_t>(inputShape.GetDimNum());
    const size_t otherRank = static_cast<size_t>(otherShape.GetDimNum());
    const size_t rank = inputRank > otherRank ? inputRank : otherRank;
    if (rank == 0 || rank > kMaxRank) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType inputDtype = context->GetInputTensor(0)->GetDataType();
    const ge::DataType otherDtype = context->GetInputTensor(1)->GetDataType();
    if (inputDtype != otherDtype) {
        return ge::GRAPH_FAILED;
    }

    const int64_t elemSize = GetElementSize(inputDtype);
    if (elemSize <= 0) {
        return ge::GRAPH_FAILED;
    }

    int64_t dims[kMaxRank] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t otherDims[kMaxRank] = {1, 1, 1, 1, 1, 1, 1, 1};
    bool sameShape = inputRank == otherRank;
    int64_t totalElems = 1;
    int64_t otherTotalElems = 1;
    bool hasPrefixBroadcast = false;
    for (size_t i = 0; i < rank; ++i) {
        dims[i] = GetDimFromRight(inputShape, i, rank);
        otherDims[i] = GetDimFromRight(otherShape, i, rank);
        if (dims[i] <= 0 || otherDims[i] <= 0) {
            return ge::GRAPH_FAILED;
        }
        if (otherDims[i] != 1 && otherDims[i] != dims[i]) {
            return ge::GRAPH_FAILED;
        }
        if (dims[i] != otherDims[i]) {
            sameShape = false;
        }
        if (i + 1 < rank && otherDims[i] == 1 && dims[i] != 1) {
            hasPrefixBroadcast = true;
        }
        totalElems *= dims[i];
        otherTotalElems *= otherDims[i];
    }

    int64_t outStrides[kMaxRank] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t otherStrides[kMaxRank] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t running = 1;
    for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
        outStrides[i] = running;
        running *= dims[i];
    }

    int64_t otherRunning = 1;
    for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
        if (otherDims[i] == 1) {
            otherStrides[i] = 0;
        } else {
            otherStrides[i] = otherRunning;
            otherRunning *= otherDims[i];
        }
    }

    const int64_t outLastDim = dims[rank - 1];
    const int64_t otherLastDim = otherDims[rank - 1];
    const int64_t outerElems = totalElems / outLastDim;

    const int64_t totalBytes = totalElems * elemSize;
    const int64_t maxCoreNum = GetCoreNum(context);
    int64_t coreNum = ClampCoreNum(maxCoreNum, maxCoreNum, totalElems);
    int64_t sameShapeUsedCoreNum = coreNum;
    int64_t sameShapeUbFactor = 1;
    int64_t sameShapeBlockFactor = 1;
    int64_t sameShapeTailBlockFactor = 1;
    int64_t sameShapeTailBlockTailUbFactor = totalElems > 0 ? totalElems : 1;
    if (sameShape && totalBytes > 0) {
        const int64_t alignElems = std::max<int64_t>(1, 32 / elemSize);
        const int64_t ubSize = GetUbSize(context);
        int64_t usableUbBytes = ubSize > kSameShapeReservedUbBytes ? (ubSize - kSameShapeReservedUbBytes) : ubSize;
        usableUbBytes = AlignDown(usableUbBytes, 32);
        if (usableUbBytes < 64) {
            usableUbBytes = 64;
        }
        int64_t ubFactor = usableUbBytes / 2 / elemSize;
        ubFactor = AlignDown(ubFactor, alignElems);
        if (ubFactor < alignElems) {
            ubFactor = alignElems;
        }

        const int64_t totalUbBlocks = std::max<int64_t>(1, CeilDiv(totalElems, ubFactor));
        int64_t usedCoreNum = totalUbBlocks < coreNum ? totalUbBlocks : coreNum;
        if (usedCoreNum <= 0) {
            usedCoreNum = 1;
        }

        const int64_t forcedSameShapeCoreNum = GetEnvInt64("ASSIGN_FORCE_SAMESHAPE_BLOCKDIM", 0);
        if (forcedSameShapeCoreNum > 0) {
            usedCoreNum = ClampCoreNum(forcedSameShapeCoreNum, maxCoreNum, totalUbBlocks);
        } else {
            usedCoreNum = SelectBalancedCoreNum(totalUbBlocks, usedCoreNum);
        }
        if (elemSize == 1 && totalBytes < kSameShapeMinBytesPerCore) {
            usedCoreNum = 1;
        }

        int64_t blockFactor = totalUbBlocks / usedCoreNum;
        if (blockFactor <= 0) {
            blockFactor = 1;
        }
        int64_t tailBlockFactor = totalUbBlocks - blockFactor * (usedCoreNum - 1);
        int64_t tailBlockTailUbFactor = totalElems - (totalUbBlocks - 1) * ubFactor;
        if (tailBlockTailUbFactor <= 0) {
            tailBlockTailUbFactor = ubFactor;
        }

        sameShapeUsedCoreNum = usedCoreNum;
        sameShapeUbFactor = ubFactor;
        sameShapeBlockFactor = blockFactor;
        sameShapeTailBlockFactor = tailBlockFactor;
        sameShapeTailBlockTailUbFactor = tailBlockTailUbFactor;
        coreNum = usedCoreNum;
    }
    int64_t broadcastMode = kBroadcastModeGeneric;
    if (!sameShape && otherTotalElems == 1) {
        broadcastMode = kBroadcastModeScalar;
    } else if (!sameShape && hasPrefixBroadcast && otherLastDim == outLastDim &&
               ((outLastDim * elemSize) % 32 == 0)) {
        broadcastMode = kBroadcastModeLastDimContiguous;
        if (outerElems > 0 && coreNum > outerElems) {
            coreNum = outerElems;
        }
    }
    if (!sameShape && inputDtype == ge::DT_BF16) {
        coreNum = 1;
    }
    if (!sameShape && broadcastMode == kBroadcastModeGeneric) {
        coreNum = 1;
    }

    coreNum = ClampCoreNum(coreNum, maxCoreNum, totalElems);

    int64_t elemsPerCore = (totalElems + coreNum - 1) / coreNum;
    const int64_t alignElems = std::max<int64_t>(1, 32 / elemSize);
    if (sameShape || broadcastMode == kBroadcastModeScalar) {
        elemsPerCore = ((elemsPerCore + alignElems - 1) / alignElems) * alignElems;
    }
    const int64_t bytesPerCore = (totalBytes + coreNum - 1) / coreNum;
    const int64_t outerElemsPerCore = outerElems > 0 ? (outerElems + coreNum - 1) / coreNum : 1;

    tiling.set_total_elems(totalElems);
    tiling.set_total_bytes(totalBytes);
    tiling.set_elem_size(elemSize);
    tiling.set_rank(static_cast<int64_t>(rank));
    tiling.set_same_shape(sameShape ? 1 : 0);
    tiling.set_block_dim(coreNum);
    tiling.set_total_core_num(maxCoreNum);
    tiling.set_used_core_num(sameShapeUsedCoreNum);
    tiling.set_ub_factor(sameShapeUbFactor);
    tiling.set_block_factor(sameShapeBlockFactor);
    tiling.set_tail_block_factor(sameShapeTailBlockFactor);
    tiling.set_tail_block_tail_ub_factor(sameShapeTailBlockTailUbFactor);
    tiling.set_elems_per_core(elemsPerCore);
    tiling.set_bytes_per_core(bytesPerCore);
    tiling.set_dim0(dims[0]);
    tiling.set_dim1(dims[1]);
    tiling.set_dim2(dims[2]);
    tiling.set_dim3(dims[3]);
    tiling.set_dim4(dims[4]);
    tiling.set_dim5(dims[5]);
    tiling.set_dim6(dims[6]);
    tiling.set_dim7(dims[7]);
    tiling.set_other_dim0(otherDims[0]);
    tiling.set_other_dim1(otherDims[1]);
    tiling.set_other_dim2(otherDims[2]);
    tiling.set_other_dim3(otherDims[3]);
    tiling.set_other_dim4(otherDims[4]);
    tiling.set_other_dim5(otherDims[5]);
    tiling.set_other_dim6(otherDims[6]);
    tiling.set_other_dim7(otherDims[7]);
    tiling.set_out_stride0(outStrides[0]);
    tiling.set_out_stride1(outStrides[1]);
    tiling.set_out_stride2(outStrides[2]);
    tiling.set_out_stride3(outStrides[3]);
    tiling.set_out_stride4(outStrides[4]);
    tiling.set_out_stride5(outStrides[5]);
    tiling.set_out_stride6(outStrides[6]);
    tiling.set_out_stride7(outStrides[7]);
    tiling.set_other_stride0(otherStrides[0]);
    tiling.set_other_stride1(otherStrides[1]);
    tiling.set_other_stride2(otherStrides[2]);
    tiling.set_other_stride3(otherStrides[3]);
    tiling.set_other_stride4(otherStrides[4]);
    tiling.set_other_stride5(otherStrides[5]);
    tiling.set_other_stride6(otherStrides[6]);
    tiling.set_other_stride7(otherStrides[7]);
    tiling.set_broadcast_mode(broadcastMode);
    tiling.set_outer_elems(outerElems);
    tiling.set_outer_elems_per_core(outerElemsPerCore);
    tiling.set_out_last_dim(outLastDim);
    tiling.set_other_last_dim(otherLastDim);

    context->SetBlockDim(static_cast<uint32_t>(coreNum));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *inputShape = context->GetInputShape(0);
    const gert::Shape *otherShape = context->GetInputShape(1);
    const size_t inputRank = static_cast<size_t>(inputShape->GetDimNum());
    const size_t otherRank = static_cast<size_t>(otherShape->GetDimNum());
    const size_t rank = inputRank > otherRank ? inputRank : otherRank;
    if (rank == 0 || rank > kMaxRank) {
        return GRAPH_FAILED;
    }

    for (size_t i = 0; i < rank; ++i) {
        const int64_t inputDim = GetDimFromRight(*inputShape, i, rank);
        const int64_t otherDim = GetDimFromRight(*otherShape, i, rank);
        if (otherDim != 1 && otherDim != inputDim) {
            return GRAPH_FAILED;
        }
    }

    gert::Shape *outShape = context->GetOutputShape(0);
    *outShape = *inputShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const auto inputDtype = context->GetInputDataType(0);
    const auto otherDtype = context->GetInputDataType(1);
    if (inputDtype != otherDtype) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, inputDtype);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Assign : public OpDef {
  public:
    explicit Assign(const char *name) : OpDef(name) {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({kSupportedTypes[0], kSupportedTypes[1], kSupportedTypes[2], kSupportedTypes[3],
                       kSupportedTypes[4], kSupportedTypes[5], kSupportedTypes[6], kSupportedTypes[7]})
            .Format({kSupportedFormats[0], kSupportedFormats[1], kSupportedFormats[2], kSupportedFormats[3],
                     kSupportedFormats[4], kSupportedFormats[5], kSupportedFormats[6], kSupportedFormats[7]})
            .UnknownShapeFormat({kSupportedFormats[0], kSupportedFormats[1], kSupportedFormats[2], kSupportedFormats[3],
                                 kSupportedFormats[4], kSupportedFormats[5], kSupportedFormats[6], kSupportedFormats[7]});

        this->Input("other")
            .ParamType(REQUIRED)
            .DataType({kSupportedTypes[0], kSupportedTypes[1], kSupportedTypes[2], kSupportedTypes[3],
                       kSupportedTypes[4], kSupportedTypes[5], kSupportedTypes[6], kSupportedTypes[7]})
            .Format({kSupportedFormats[0], kSupportedFormats[1], kSupportedFormats[2], kSupportedFormats[3],
                     kSupportedFormats[4], kSupportedFormats[5], kSupportedFormats[6], kSupportedFormats[7]})
            .UnknownShapeFormat({kSupportedFormats[0], kSupportedFormats[1], kSupportedFormats[2], kSupportedFormats[3],
                                 kSupportedFormats[4], kSupportedFormats[5], kSupportedFormats[6], kSupportedFormats[7]});

        this->Attr("validate_shape").AttrType(OPTIONAL).Bool(true);
        this->Attr("use_locking").AttrType(OPTIONAL).Bool(false);

        this->Output("input")
            .ParamType(REQUIRED)
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

OP_ADD(Assign);
} // namespace ops
