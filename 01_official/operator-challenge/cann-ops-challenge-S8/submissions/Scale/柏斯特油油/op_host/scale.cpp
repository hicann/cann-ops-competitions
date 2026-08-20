
#include "scale_tiling.h"
#include "register/op_def_registry.h"

using namespace std;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ScaleTilingData tiling;

    int attrNum = context->GetAttrs()->GetAttrNum();
    int axis = 1;
    int num_axes = 1;
    bool scale_from_blob = true;
    if (attrNum > 0) {
        const int64_t *axis_attr = context->GetAttrs()->GetInt(0);
        if (axis_attr) {
            axis = (int)*axis_attr;
        }
    }
    if (attrNum > 1) {
        const int64_t *num_axes_attr = context->GetAttrs()->GetInt(1);
        if (num_axes_attr) {
            num_axes = (int)*num_axes_attr;
        }
    }
    if (attrNum > 2) {
        const bool *scale_from_blob_attr = context->GetAttrs()->GetBool(2);
        if (scale_from_blob_attr) {
            scale_from_blob = *scale_from_blob_attr;
        }
    }
    auto input_shape = context->GetInputShape(0)->GetOriginShape();
    int input_rank = input_shape.GetDimNum();
    auto scale_shape = context->GetInputShape(1)->GetOriginShape();
    auto bias_shape = context->GetInputShape(2)->GetOriginShape();
    int scale_rank = scale_shape.GetDimNum();
    int bias_rank = bias_shape.GetDimNum();
    auto tensor = context->GetInputTensor(0);
    auto type = tensor->GetDataType();
    if (axis < 0) {
        axis += input_rank;
    }
    int end_axis = axis;
    if (scale_from_blob) {
        end_axis = (num_axes == -1) ? input_rank : (axis + num_axes);
    } else {
        end_axis = axis + scale_rank;
    }
    uint64_t preLength = 1;
    uint64_t midLength = 1;
    uint64_t postLength = 1;
    for (int i = 0; i < axis; i++) {
        preLength *= input_shape.GetDim(i);
    }
    for (int i = axis; i < end_axis; i++) {
        midLength *= input_shape.GetDim(i);
    }
    for (int i = end_axis; i < input_rank; i++) {
        postLength *= input_shape.GetDim(i);
    }
    tiling.set_preLength(preLength);
    tiling.set_midLength(midLength);
    tiling.set_postLength(postLength);
    
    if (postLength == 1) {
        context->SetTilingKey(1);
    } else {
        if (postLength <= 2048) {
            context->SetTilingKey(2);
        } else {
            context->SetTilingKey(0);
        }
    }

    context->SetBlockDim(40);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class Scale : public OpDef {
public:
    explicit Scale(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("axis").AttrType(OPTIONAL).Int(1);
        this->Attr("num_axes").AttrType(OPTIONAL).Int(1);
        this->Attr("scale_from_blob").AttrType(OPTIONAL).Bool(true);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Scale);
}
