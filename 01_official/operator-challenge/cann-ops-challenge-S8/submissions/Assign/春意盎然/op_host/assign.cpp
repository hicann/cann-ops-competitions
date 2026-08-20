
#include "../op_kernel/assign_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace optiling {
namespace {
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_USE_COEF = 4;
constexpr uint32_t MIN_CORE_BYTES = 16 * 1024;
constexpr uint32_t SMALL_COPY_BYTES = 8 * 1024;
constexpr uint32_t GM_ALIGN_BYTES = 512;

uint32_t GetTypeSize(const ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
            return 2;
        case ge::DT_UINT8:
        case ge::DT_INT8:
        case ge::DT_BOOL:
        default:
            return 1;
    }
}

uint64_t GetShapeSize(const gert::Shape &shape)
{
    uint64_t size = 1;
    for (int32_t i = 0; i < shape.GetDimNum(); ++i) {
        size *= static_cast<uint64_t>(shape.GetDim(i));
    }
    return size;
}

std::vector<uint32_t> NormalizeShape(const gert::Shape &shape, const uint32_t dimNum)
{
    std::vector<uint32_t> result(dimNum, 1);
    const int32_t srcDimNum = shape.GetDimNum();
    for (int32_t i = srcDimNum - 1, j = static_cast<int32_t>(dimNum) - 1; i >= 0 && j >= 0; --i, --j) {
        result[j] = static_cast<uint32_t>(shape.GetDim(i));
    }
    return result;
}

bool IsBroadcastCompatible(const std::vector<uint32_t> &inputShape, const std::vector<uint32_t> &otherShape)
{
    for (size_t i = 0; i < inputShape.size(); ++i) {
        if (otherShape[i] != inputShape[i] && otherShape[i] != 1) {
            return false;
        }
    }
    return true;
}

uint32_t ComputeBlockLength(const uint32_t totalLength, const uint32_t blockDim)
{
    if (blockDim == 0) {
        return totalLength;
    }
    return (totalLength + blockDim - 1) / blockDim;
}

uint32_t AlignUp(const uint32_t value, const uint32_t align)
{
    if (align == 0) {
        return value;
    }
    return ((value + align - 1) / align) * align;
}
} // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    AssignTilingData tiling;
    const gert::StorageShape *inputStorageShape = context->GetInputShape(0);
    const gert::StorageShape *otherStorageShape = context->GetInputShape(1);
    if (inputStorageShape == nullptr || otherStorageShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &inputShape = inputStorageShape->GetStorageShape();
    const gert::Shape &otherShape = otherStorageShape->GetStorageShape();
    const uint32_t inputDimNum = static_cast<uint32_t>(inputShape.GetDimNum());
    const uint32_t otherDimNum = static_cast<uint32_t>(otherShape.GetDimNum());
    const uint32_t dimNum = std::max(inputDimNum, otherDimNum);
    if (dimNum == 0 || dimNum > ASSIGN_MAX_DIMS) {
        return ge::GRAPH_FAILED;
    }

    const std::vector<uint32_t> inputShapeVec = NormalizeShape(inputShape, dimNum);
    const std::vector<uint32_t> otherShapeVec = NormalizeShape(otherShape, dimNum);
    if (!IsBroadcastCompatible(inputShapeVec, otherShapeVec)) {
        return ge::GRAPH_FAILED;
    }

    uint64_t totalLength64 = GetShapeSize(inputShape);
    uint64_t otherLength64 = GetShapeSize(otherShape);
    if (totalLength64 == 0 || totalLength64 > UINT32_MAX || otherLength64 == 0 || otherLength64 > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t coreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    if (coreNum == 0) {
        coreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNum());
        if (coreNum == 0) {
            coreNum = 1;
        }
    }

    const uint32_t typeSize = GetTypeSize(context->GetInputDesc(0)->GetDataType());
    const uint32_t alignNum = std::max<uint32_t>(1, BLOCK_SIZE / typeSize);
    uint32_t maxTileLength = static_cast<uint32_t>(ubSize / UB_USE_COEF / BUFFER_NUM / typeSize);
    maxTileLength = std::max<uint32_t>(alignNum, maxTileLength / alignNum * alignNum);
    const uint32_t alignedTotalLength = (static_cast<uint32_t>(totalLength64) + alignNum - 1) / alignNum * alignNum;
    uint32_t tileLength = std::min<uint32_t>(maxTileLength, alignedTotalLength);
    tileLength = std::max<uint32_t>(alignNum, tileLength / alignNum * alignNum);

    const uint32_t totalLength = static_cast<uint32_t>(totalLength64);
    // For contiguous path: each core's startOffset must be 32B-aligned in bytes.
    // blockLength (in elements) * typeSize must be a multiple of 32.
    // So blockLength must be a multiple of alignNum.
    const uint64_t totalBytes = totalLength64 * typeSize;
    const bool isBroadcastEarly = (totalLength64 != otherLength64);
    const bool isScalarBroadcast = isBroadcastEarly && otherLength64 == 1;
    const uint32_t gmAlignNum = std::max<uint32_t>(alignNum, GM_ALIGN_BYTES / typeSize);
    uint32_t usedCoreNum;
    if (isBroadcastEarly && !isScalarBroadcast) {
        if (totalBytes < MIN_CORE_BYTES) {
            usedCoreNum = 1;
        } else {
            usedCoreNum = std::max<uint32_t>(1, std::min<uint32_t>(coreNum, totalLength));
        }
    } else {
        const uint32_t coresByBytes =
            totalBytes < MIN_CORE_BYTES ? 1U : static_cast<uint32_t>(std::min<uint64_t>(coreNum, totalBytes / MIN_CORE_BYTES));
        uint32_t desiredCores = std::min<uint32_t>(coresByBytes, totalLength / gmAlignNum);
        if (desiredCores == 0) {
            desiredCores = 1;
        }
        const uint32_t blockLenAligned =
            AlignUp((totalLength + desiredCores - 1) / desiredCores, gmAlignNum);
        usedCoreNum = (totalLength + blockLenAligned - 1) / blockLenAligned;
        if (usedCoreNum == 0) {
            usedCoreNum = 1;
        }
    }
    const uint32_t blockLength =
        (isBroadcastEarly && !isScalarBroadcast) ? ComputeBlockLength(totalLength, usedCoreNum) :
                                                   AlignUp((totalLength + usedCoreNum - 1) / usedCoreNum, gmAlignNum);

    uint32_t outShape[ASSIGN_MAX_DIMS] = {0};
    uint32_t normalizedOtherShape[ASSIGN_MAX_DIMS] = {0};
    uint32_t otherStride[ASSIGN_MAX_DIMS] = {0};
    uint32_t stride = 1;
    bool isBroadcast = (totalLength64 != otherLength64);
    for (int32_t i = static_cast<int32_t>(dimNum) - 1; i >= 0; --i) {
        outShape[i] = inputShapeVec[i];
        normalizedOtherShape[i] = otherShapeVec[i];
        otherStride[i] = (otherShapeVec[i] == 1 && inputShapeVec[i] != 1) ? 0 : stride;
        if (otherShapeVec[i] != 1 || inputShapeVec[i] == 1) {
            stride *= otherShapeVec[i];
        }
        if (otherShapeVec[i] != inputShapeVec[i]) {
            isBroadcast = true;
        }
    }

    tiling.set_totalLength(totalLength);
    tiling.set_otherLength(static_cast<uint32_t>(otherLength64));
    tiling.set_blockLength(blockLength);
    tiling.set_tileLength(tileLength);
    tiling.set_alignNum(alignNum);
    tiling.set_dimNum(dimNum);
    tiling.set_isBroadcast(isBroadcast ? 1U : 0U);
    tiling.set_outShape(outShape);
    tiling.set_otherShape(normalizedOtherShape);
    tiling.set_otherStride(otherStride);

    const bool isRowRepeatBroadcast =
        isBroadcast && normalizedOtherShape[dimNum - 1] == outShape[dimNum - 1] &&
        otherLength64 == static_cast<uint64_t>(outShape[dimNum - 1]);
    bool isPrefixBlockRepeatBroadcast =
        isBroadcast && normalizedOtherShape[dimNum - 1] == outShape[dimNum - 1] &&
        otherLength64 > static_cast<uint64_t>(outShape[dimNum - 1]);
    bool seenBlockDim = false;
    for (uint32_t i = 0; i + 1 < dimNum; ++i) {
        if (normalizedOtherShape[i] == outShape[i]) {
            seenBlockDim = true;
        } else if (normalizedOtherShape[i] == 1 && outShape[i] != 1 && !seenBlockDim) {
            continue;
        } else {
            isPrefixBlockRepeatBroadcast = false;
            break;
        }
    }
    bool isPrefixLastDimScalarBroadcast =
        isBroadcast && normalizedOtherShape[dimNum - 1] == 1 && outShape[dimNum - 1] != 1 && otherLength64 > 1;
    bool seenOuterEqualDim = false;
    for (uint32_t i = 0; i + 1 < dimNum; ++i) {
        if (normalizedOtherShape[i] == outShape[i]) {
            seenOuterEqualDim = true;
        } else if (normalizedOtherShape[i] == 1 && outShape[i] != 1 && !seenOuterEqualDim) {
            continue;
        } else {
            isPrefixLastDimScalarBroadcast = false;
            break;
        }
    }
    bool isMiddleLastDimScalarBroadcast =
        isBroadcast && normalizedOtherShape[dimNum - 1] == 1 && outShape[dimNum - 1] != 1 && otherLength64 > 1;
    bool seenPrefixEqual = false;
    bool seenMiddleBroadcast = false;
    bool seenSuffixEqual = false;
    for (uint32_t i = 0; i + 1 < dimNum; ++i) {
        const bool isEqualDim = normalizedOtherShape[i] == outShape[i];
        const bool isBroadcastDim = normalizedOtherShape[i] == 1 && outShape[i] != 1;
        if (!seenMiddleBroadcast) {
            if (isEqualDim) {
                seenPrefixEqual = true;
            } else if (seenPrefixEqual && isBroadcastDim) {
                seenMiddleBroadcast = true;
            } else {
                isMiddleLastDimScalarBroadcast = false;
                break;
            }
        } else if (!seenSuffixEqual) {
            if (isBroadcastDim) {
                continue;
            } else if (isEqualDim) {
                seenSuffixEqual = true;
            } else {
                isMiddleLastDimScalarBroadcast = false;
                break;
            }
        } else if (!isEqualDim) {
            isMiddleLastDimScalarBroadcast = false;
            break;
        }
    }
    isMiddleLastDimScalarBroadcast =
        isMiddleLastDimScalarBroadcast && seenPrefixEqual && seenMiddleBroadcast;

    context->SetBlockDim(usedCoreNum);
    if (!isBroadcast && totalBytes <= SMALL_COPY_BYTES) {
        context->SetBlockDim(1);
        context->SetTilingKey(0);
    } else if (!isBroadcast) {
        context->SetTilingKey(1);
    } else if (isScalarBroadcast && totalBytes <= SMALL_COPY_BYTES) {
        context->SetBlockDim(1);
        context->SetTilingKey(4);
    } else if (isScalarBroadcast) {
        context->SetTilingKey(2);
    } else if (isRowRepeatBroadcast) {
        context->SetTilingKey(5);
    } else if (isPrefixBlockRepeatBroadcast) {
        context->SetTilingKey(7);
    } else if (isPrefixLastDimScalarBroadcast) {
        context->SetTilingKey(6);
    } else if (isMiddleLastDimScalarBroadcast) {
        context->SetTilingKey(8);
    } else {
        context->SetTilingKey(3);
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    const gert::Shape *otherShape = context->GetInputShape(1);
    if (inputShape == nullptr || otherShape == nullptr) {
        return GRAPH_FAILED;
    }

    const uint32_t dimNum = static_cast<uint32_t>(std::max(inputShape->GetDimNum(), otherShape->GetDimNum()));
    if (dimNum > optiling::ASSIGN_MAX_DIMS) {
        return GRAPH_FAILED;
    }
    const std::vector<uint32_t> inputShapeVec = optiling::NormalizeShape(*inputShape, dimNum);
    const std::vector<uint32_t> otherShapeVec = optiling::NormalizeShape(*otherShape, dimNum);
    if (!optiling::IsBroadcastCompatible(inputShapeVec, otherShapeVec)) {
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Assign : public OpDef {
public:
    explicit Assign(const char *name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8,
                       ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("other")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8,
                       ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("use_locking").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Assign);
} // namespace ops
