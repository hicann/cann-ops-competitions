#include "scale_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t BLOCK_DIM = 40;
constexpr uint32_t SCALE_TILE_SIZE = 4096;
constexpr uint32_t ALIGN_BYTES = 32;

static uint32_t GetTypeBytes(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
            return 4;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2;
        default:
            return 1;
    }
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ScaleTilingData tiling;

    const gert::StorageShape* input_shape = context->GetInputShape(0);
    const gert::StorageShape* scale_shape = context->GetInputShape(1);

    int dim_num = input_shape->GetStorageShape().GetDimNum();
    int scale_dim_num = scale_shape->GetStorageShape().GetDimNum();

    const gert::RuntimeAttrs* attrs = context->GetAttrs();

    int axis = *attrs->GetAttrPointer<int>(0);
    int num_axes = *attrs->GetAttrPointer<int>(1);
    bool scale_from_blob = *attrs->GetAttrPointer<bool>(2);

    if (axis < 0) {
        axis += dim_num;
    }

    if (scale_from_blob) {
        if (num_axes == -1) {
            num_axes = dim_num - axis;
        }
    } else {
        num_axes = scale_dim_num;
    }

    uint32_t batch_size = 1;
    uint32_t length = 1;
    uint32_t batch_size_vec = 1;

    for (int i = 0; i < axis; ++i) {
        batch_size *= input_shape->GetStorageShape().GetDim(i);
    }

    for (int i = axis; i < axis + num_axes; ++i) {
        length *= input_shape->GetStorageShape().GetDim(i);
    }

    for (int i = axis + num_axes; i < dim_num; ++i) {
        batch_size_vec *= input_shape->GetStorageShape().GetDim(i);
    }

    uint32_t has_bias = 1;
    const gert::StorageShape* bias_shape = context->GetInputShape(2);
    if (bias_shape == nullptr) {
        has_bias = 0;
    }

    tiling.set_length(length);
    tiling.set_batch_size(batch_size);
    tiling.set_batch_size_vec(batch_size_vec);
    tiling.set_has_bias(has_bias);

    const ge::DataType input_dtype = context->GetInputDesc(0)->GetDataType();
    const uint32_t type_bytes = GetTypeBytes(input_dtype);
    const uint32_t vec_bytes = batch_size_vec * type_bytes;
    const bool is_vec_32b_aligned = (vec_bytes % ALIGN_BYTES == 0);

    if (batch_size_vec == 1) {
        context->SetTilingKey(1);
    } else if ((batch_size_vec * type_bytes) % 32 == 0&&batch_size_vec*type_bytes < 20*1024) {
        context->SetTilingKey(3);
    } else if (batch_size_vec < SCALE_TILE_SIZE) {
        context->SetTilingKey(2);
    } else {
        context->SetTilingKey(0);
    }

    context->SetBlockDim(BLOCK_DIM);

    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());

    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

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
            .ParamType(OPTIONAL)
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

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Scale);
} // namespace ops
