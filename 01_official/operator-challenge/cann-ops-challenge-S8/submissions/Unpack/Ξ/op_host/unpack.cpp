#include "unpack_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling
{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        auto &input = context->GetInputTensor(0)->GetStorageShape();
        auto axis = *context->GetAttrs()->GetInt(1);
        auto dim = axis >= 0 ? axis : axis + input.GetDimNum();
        auto high_size = 1;
        for (auto i = 0; i < dim; i++)
            high_size *= input[i];
        auto mid_size = input[dim];
        auto low_size = 1;
        for (auto i = dim + 1; i < input.GetDimNum(); i++)
            low_size *= input[i];

        UnpackTilingData tiling;
        tiling.set_low_size(low_size);
        tiling.set_mid_size(mid_size);
        tiling.set_high_size(high_size);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        auto plaform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto block_dim = plaform.GetCoreNumAiv();
        auto dtype = context->GetInputTensor(0)->GetDataType();
        auto size_in_bytes = ge::GetSizeInBytes(input.GetShapeSize(), dtype);
        if (size_in_bytes < 65536)
            block_dim = 1;
        context->SetBlockDim(block_dim);
        auto tiling_key = 0;
        if (low_size == 1)
            tiling_key = size_in_bytes > 15000000 && size_in_bytes < 20000000 ? 3 : 1;
        else if (ge::GetSizeInBytes(low_size, dtype) > 98304)
            tiling_key = 2;
        context->SetTilingKey(tiling_key);

        return ge::GRAPH_SUCCESS;
    }
}


namespace ge
{
    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *x1_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);
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


namespace ops
{
    class Unpack : public OpDef
    {
    public:
        explicit Unpack(const char *name) : OpDef(name)
        {
            this->Input("input")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("output")
                .ParamType(DYNAMIC)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("num").Int();
            this->Attr("axis").AttrType(OPTIONAL).Int(0);

            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

            this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(Unpack);
}
