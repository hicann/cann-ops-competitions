#include "scale_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling
{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        auto &input = context->GetInputShape(0)->GetStorageShape();
        auto &scale = context->GetInputShape(1)->GetStorageShape();
        auto has_bias = context->GetOptionalInputTensor(2) != nullptr;
        auto axis = *context->GetAttrs()->GetInt(0);
        auto dim = axis >= 0 ? axis : axis + input.GetDimNum();
        auto high_size = 1;
        for (auto i = 0; i < dim; i++)
            high_size *= input[i];
        auto mid_size = 1;
        for (auto i = 0; i < scale.GetDimNum(); i++)
            mid_size *= scale[i];
        auto low_size = 1;
        for (auto i = dim + scale.GetDimNum(); i < input.GetDimNum(); i++)
            low_size *= input[i];
        auto broadcast_size = -1;
        std::vector<int> broadcast_stride{-1, -1};
        auto dtype = context->GetInputTensor(0)->GetDataType();
        int data_block_size = 32 / ge::GetSizeByDataType(dtype);
        if (low_size > 1 && mid_size % data_block_size == 0)
        {
            for (int i = 1; i <= 8; i++)
            {
                for (int j = 1; j <= 8; j++)
                {
                    int round_size = data_block_size * i * j;
                    if (round_size >= low_size && (broadcast_size == -1 || round_size < broadcast_size))
                    {
                        broadcast_size = round_size;
                        broadcast_stride[0] = i;
                        broadcast_stride[1] = j;
                    }
                }
            }
        }

        ScaleTilingData tiling;
        tiling.set_low_size(low_size);
        tiling.set_mid_size(mid_size);
        tiling.set_high_size(high_size);
        tiling.set_broadcast_size(broadcast_size);
        tiling.set_broadcast_stride(broadcast_stride.data());
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        auto plaform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto block_dim = plaform.GetCoreNumAiv();
        context->SetBlockDim(block_dim);
        auto tiling_key = 0;
        if (has_bias)
        {
            if (low_size == 1)
            {
                if (mid_size < 6000)
                    tiling_key = 2;
                else if (mid_size < 30000)
                    tiling_key = 3;
                else
                    tiling_key = 1;
            }
            else
            {
                if (low_size > 512 && low_size % data_block_size == 0)
                    tiling_key = 4;
                else if (low_size <= 512 && mid_size % data_block_size == 0)
                    tiling_key = 5;
                else
                    tiling_key = 1;
            }
        }
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
    class Scale : public OpDef
    {
    public:
        explicit Scale(const char *name) : OpDef(name)
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

            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

            this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(Scale);
}
