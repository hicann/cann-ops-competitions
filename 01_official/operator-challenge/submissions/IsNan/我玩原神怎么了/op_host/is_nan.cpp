#include "is_nan_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    IsNanTilingData tiling;
    const auto shape = context->GetInputTensor(0)->GetOriginShape();

    uint32_t totalSize = 1;
    for (int i = 0; i < shape.GetDimNum(); ++i) {
        totalSize *= shape.GetDim(i);
    }

    const auto dt = context->GetInputTensor(0)->GetDataType();
    uint32_t dataTypeSize = 2;
    if (dt == ge::DT_FLOAT) {
        dataTypeSize = 4;
        context->SetTilingKey(3);
    } else if (dt == ge::DT_BF16) {
        context->SetTilingKey(2);
    } else {
        context->SetTilingKey(1);
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivMax = ascendcPlatform.GetCoreNumAiv();
    if (aivMax == 0) {
        aivMax = 40;
    }
    constexpr uint32_t blockSize = 512;
    uint32_t elementAlign = blockSize / dataTypeSize;
    uint32_t blockNum = (totalSize * dataTypeSize + blockSize - 1) / blockSize;
    uint32_t aivNum = std::min(aivMax, blockNum);
    if (aivNum == 0) {
        aivNum = 1;
    }

    uint32_t alignedSize = totalSize / elementAlign * elementAlign;
    uint32_t alignedBlocks = alignedSize / elementAlign;
    uint32_t smallSize = alignedBlocks == 0 ? 0 : alignedBlocks / aivNum * elementAlign;
    uint32_t incSize = alignedBlocks == 0 ? 0 : elementAlign;
    uint16_t formerNum = alignedBlocks == 0 ? 0 : alignedBlocks % aivNum;

    tiling.set_smallSize(smallSize);
    tiling.set_incSize(incSize);
    tiling.set_totalSize(totalSize);
    tiling.set_formerNum(formerNum);
    context->SetBlockDim(aivNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, ge::DT_BOOL);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class IsNan : public OpDef {
public:
    explicit IsNan(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(IsNan);
} // namespace ops