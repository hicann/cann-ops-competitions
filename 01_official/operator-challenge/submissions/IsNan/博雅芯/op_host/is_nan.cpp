
#include "is_nan_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 获取平台信息
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t num_cores = platform.GetCoreNumAiv();
    // uint64_t ub_size;
    // platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    // 任务划分
    int32_t total = context->GetInputTensor(0)->GetShapeSize();
    auto dtype = context->GetInputDesc(0)->GetDataType();
    int32_t bytes = ge::GetSizeByDataType(dtype);
    int32_t elements_per_task = 2048 / bytes;
    while (elements_per_task < 8192 && elements_per_task * num_cores < total) elements_per_task <<= 1;
    int32_t tasks_total = (total + elements_per_task - 1) / elements_per_task;
    int32_t tasks_per_core = (tasks_total + num_cores - 1) / num_cores;
    num_cores = (tasks_total + tasks_per_core - 1) / tasks_per_core;
    int32_t elements_per_core = tasks_per_core * elements_per_task;
    IsNanTilingData tiling;
    tiling.set_total(total);
    tiling.set_each(elements_per_core);
    tiling.set_buffer(elements_per_task);
    context->SetBlockDim(num_cores);
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
class IsNan : public OpDef {
public:
    explicit IsNan(const char* name) : OpDef(name)
    {
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

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(IsNan);
}
