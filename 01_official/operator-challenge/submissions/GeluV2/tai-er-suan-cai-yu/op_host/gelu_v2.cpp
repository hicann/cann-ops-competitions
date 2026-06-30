#include "gelu_v2_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t BLOCK_DIM = 1;

bool GetTotalSize(const gert::Shape &shape, uint32_t &totalSize)
{
    const size_t dimNum = shape.GetDimNum();
    if (dimNum == 0 || dimNum > 4) {
        return false;
    }
    uint64_t total = 1;
    for (size_t i = 0; i < dimNum; ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim <= 0) {
            return false;
        }
        total *= static_cast<uint64_t>(dim);
    }
    if (total == 0 || total > 0xffffffffULL) {
        return false;
    }
    totalSize = static_cast<uint32_t>(total);
    return true;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *xShape = context->GetInputShape(0);
    if (xShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t totalSize = 0;
    if (!GetTotalSize(xShape->GetStorageShape(), totalSize)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t approximate = 0;
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t *attrPtr = attrs->GetAttrPointer<int64_t>(0);
        if (attrPtr != nullptr && *attrPtr != 0) {
            approximate = 1;
        }
    }

    GeluV2TilingData tiling;
    tiling.set_totalSize(totalSize);
    tiling.set_approximate(approximate);
    context->SetBlockDim(BLOCK_DIM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr) {
        return GRAPH_FAILED;
    }
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class GeluV2 : public OpDef {
public:
    explicit GeluV2(const char *name) : OpDef(name)
    {
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
        this->Attr("approximate").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(GeluV2);
}
