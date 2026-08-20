
#include "atanh_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  AtanhTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  int32_t data_sz = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    data_sz *= x1_shape->GetStorageShape().GetDim(i);

  size_t usr_size = 1024;
  auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t sys_workspace_size = ascendc_platform.GetLibApiWorkSpaceSize();
  size_t *current_workspace = context->GetWorkspaceSizes(1);
  current_workspace[0] = usr_size + sys_workspace_size;

  auto dt = context->GetInputTensor(0)->GetDataType();
    if (dt == ge::DT_FLOAT) {
        context->SetTilingKey(1);
    } else if(dt == ge::DT_FLOAT16) {
        context->SetTilingKey(2);
    } else if(dt == ge::DT_BF16) {
        context->SetTilingKey(3);
    } 
  int32_t chunk = int((int(data_sz / 40) + 255) / 256) * 256;
  tiling.set_size(data_sz);
  tiling.set_chunk(chunk);
  int cores = (data_sz + chunk - 1) / chunk;
  if(cores % 2) cores += 1;
  context->SetBlockDim(cores);
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
class Atanh : public OpDef {
public:
    explicit Atanh(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Atanh);
}
