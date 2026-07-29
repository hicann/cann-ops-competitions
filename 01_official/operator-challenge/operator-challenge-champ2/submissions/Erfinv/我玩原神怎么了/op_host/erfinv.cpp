#include "erfinv_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    ErfinvTilingData tiling;
    const auto shape = context->GetInputTensor(0)->GetOriginShape();

    uint32_t totalSize = 1;
    for (int i = 0; i < shape.GetDimNum(); ++i) {
        totalSize *= shape.GetDim(i);
    }

    const auto dt = context->GetInputTensor(0)->GetDataType();
    if (dt == ge::DT_FLOAT) {
        context->SetTilingKey(3);
    } else if (dt == ge::DT_BF16) {
        context->SetTilingKey(2);
    } else {
        context->SetTilingKey(1);
    }

    constexpr uint32_t aivMax = 40;
    constexpr uint32_t elementsPerCore = 16;
    uint32_t blockNum = (totalSize + elementsPerCore - 1) / elementsPerCore;
    uint32_t aivNum = std::min(aivMax, blockNum);
    if (aivNum == 0) {
        aivNum = 1;
    }

    uint32_t smallSize = totalSize / aivNum;
    uint32_t incSize = 1;
    uint16_t formerNum = totalSize % aivNum;

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
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Erfinv : public OpDef {
public:
    explicit Erfinv(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Erfinv);
} // namespace ops
