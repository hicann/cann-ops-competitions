#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace {
constexpr uint32_t ERF_FLOAT_BYTES = 4U;
constexpr uint32_t ERF_ALIGN_ELEMENTS = 64U;
constexpr uint32_t ERF_RESERVED_UB_BYTES = 8192U;

uint32_t CeilDiv(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1U) / divisor;
}

uint32_t AlignUp(uint32_t value, uint32_t align) {
    return ((value + align - 1U) / align) * align;
}

uint32_t CalcTileLength(uint64_t ubBytes, uint32_t bufferCount) {
    uint64_t usableBytes = ubBytes > ERF_RESERVED_UB_BYTES ? ubBytes - ERF_RESERVED_UB_BYTES : ubBytes;
    uint64_t rawElements = usableBytes / (static_cast<uint64_t>(bufferCount) * ERF_FLOAT_BYTES);
    uint32_t tileLength = (static_cast<uint32_t>(rawElements) / ERF_ALIGN_ELEMENTS) * ERF_ALIGN_ELEMENTS;
    return tileLength == 0U ? ERF_ALIGN_ELEMENTS : tileLength;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t platformCoreNum = platform.GetCoreNumAiv();
    uint64_t ubBytes = 0U;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);

    const gert::Tensor *inputTensor = context->GetRequiredInputTensor(0);
    uint32_t totalLength = static_cast<uint32_t>(inputTensor->GetShapeSize());
    uint32_t maxCoreNum = platformCoreNum > 0 ? static_cast<uint32_t>(platformCoreNum) : 1U;

    uint32_t maxTileLengthDirect = CalcTileLength(ubBytes, 3U);

    uint32_t mode = 0U;
    uint32_t coreNum = 1U;
    uint32_t tileLength = maxTileLengthDirect;

    if (totalLength > 0U) {
        if (totalLength <= 8192U) {
            mode = 0U;
            coreNum = 1U;
            tileLength = AlignUp(totalLength, ERF_ALIGN_ELEMENTS);
        } else if (totalLength <= maxTileLengthDirect * maxCoreNum) {
            mode = 0U;
            uint32_t preferredElementsPerCore = 8192U;
            uint32_t neededCores = CeilDiv(totalLength, preferredElementsPerCore);
            coreNum = neededCores < maxCoreNum ? neededCores : maxCoreNum;
            if (coreNum == 0U) {
                coreNum = 1U;
            }

            uint32_t blockStride = CeilDiv(totalLength, coreNum);
            blockStride = AlignUp(blockStride, ERF_ALIGN_ELEMENTS);

            tileLength = blockStride;
            coreNum = CeilDiv(totalLength, tileLength);
        } else {
            mode = 1U;
            uint32_t maxTileLengthPipeline = CalcTileLength(ubBytes, 5U);
            tileLength = maxTileLengthPipeline;
            coreNum = maxCoreNum;
        }
    }

    if (coreNum == 0U) {
        coreNum = 1U;
    }

    ErfTilingData *tilingData = context->GetTilingData<ErfTilingData>();
    tilingData->totalLength = totalLength;
    tilingData->tileLength = tileLength;
    tilingData->coreNum = coreNum;
    tilingData->mode = mode;

    context->GetRawTilingData()->SetDataSize(sizeof(ErfTilingData));
    context->SetBlockDim(coreNum);

    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = 0U;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Erf : public OpDef {
public:
    explicit Erf(const char *name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(Erf);
}
