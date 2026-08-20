#include "assign_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

static std::vector<int> get_canon_dims(const gert::Shape &shape)
{
    std::vector<int> dims{1, 1, 1, 1};
    for (auto i = 0; i < shape.GetDimNum(); i++)
        dims[i] = shape[shape.GetDimNum() - i - 1];
    while (!dims.empty() && dims.back() == 1)
        dims.pop_back();
    return dims;
}

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
        auto &input = context->GetInputShape(0)->GetStorageShape();
        auto &other = context->GetInputShape(1)->GetStorageShape();
        auto input_dims = get_canon_dims(input);
        auto other_dims = get_canon_dims(other);
        if (input_dims.size() != other_dims.size())
        {
            auto high_dim = 1;
            for (auto i = other_dims.size(); i < input_dims.size(); i++)
                high_dim *= input_dims[i];
            input_dims.resize(other_dims.size() + 1);
            input_dims[other_dims.size()] = high_dim;
        }
        input_dims.resize(4, 1);
        other_dims.resize(4, 1);
        auto size = 1;
        for (auto i = 0; i < 4; i++)
            size *= input_dims[i];
        auto tiling_key = input_dims == other_dims ? 0 : 1;
        auto dtype = context->GetInputTensor(0)->GetDataType();
        if (tiling_key == 0)
            size = ceil_round(size, 32 / ge::GetSizeByDataType(dtype));
        auto size_in_bytes = ge::GetSizeInBytes(size, dtype);

        AssignTilingData tiling;
        tiling.set_size(size);
        tiling.set_input_n(input_dims.data());
        tiling.set_other_n(other_dims.data());
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        auto plaform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto block_dim = plaform.GetCoreNumAiv();
        if (size_in_bytes <= 65536)
            block_dim = 1;
        context->SetBlockDim(block_dim);
        context->SetTilingKey(tiling_key);
        if (tiling_key == 1)
        {
            auto single_size = ceil_round(size_in_bytes, 512L);
            auto workspace_sizes = context->GetWorkspaceSizes(1);
            workspace_sizes[0] = plaform.GetLibApiWorkSpaceSize() + single_size;
        }

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
    class Assign : public OpDef
    {
    public:
        explicit Assign(const char *name) : OpDef(name)
        {
            this->Input("input")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("other")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("use_locking").AttrType(OPTIONAL).Bool(false);

            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

            this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(Assign);
}
