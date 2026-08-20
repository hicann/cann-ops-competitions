#include "fills_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

template<typename T>
constexpr T ceil_div(T x, T y)
{
    return (x - 1) / y + 1;
}

template<typename T>
constexpr T ceil_round(T x, T y)
{
    return ceil_div(x, y) * y;
}

namespace optiling
{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        auto &input = context->GetInputTensor(0)->GetStorageShape();
        auto value = *context->GetAttrs()->GetFloat(0);
        auto dtype = context->GetInputTensor(0)->GetDataType();
        auto size = ceil_round(input.GetShapeSize(), 32L / ge::GetSizeByDataType(dtype));

        FillsTilingData tiling;
        tiling.set_size(size);
        tiling.set_value(value);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        auto plaform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto block_dim = plaform.GetCoreNumAiv();
        auto size_in_bytes = ge::GetSizeInBytes(size, dtype);
        if (size_in_bytes < 65536)
            block_dim = 1;
        context->SetBlockDim(block_dim);
        auto tiling_key = size_in_bytes > 15000000 && size_in_bytes < 40000000 ? 1 : 0;
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
    class Fills : public OpDef
    {
    public:
        explicit Fills(const char *name) : OpDef(name)
        {
            this->Input("input")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("output")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("value").Float();

            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

            this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(Fills);
}
