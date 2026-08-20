
#include "unpack_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  UnpackTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  const gert::RuntimeAttrs *attrs = context->GetAttrs();
  int64_t axis, num, mode, dims;
  const int64_t *num_attr = attrs->GetInt(0);
  const int64_t *axis_attr = attrs->GetInt(1);
  num = *num_attr;
  axis = *axis_attr;
  int m = 1, n = 1, k = 1;
  dims = x1_shape->GetStorageShape().GetDimNum();
  if(axis == -1) axis = dims - 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
  {
    if(i < axis) m *= x1_shape->GetStorageShape().GetDim(i);
    else if(i > axis) k *= x1_shape->GetStorageShape().GetDim(i);
  }
  n = num;
  std::cout << "num: " << n << " input[axis]: " << x1_shape->GetStorageShape().GetDim(axis) << std::endl;
  
  size_t usr_size = 1024;
  auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t sys_workspace_size = ascendc_platform.GetLibApiWorkSpaceSize();
  size_t *current_workspace = context->GetWorkspaceSizes(1);
  current_workspace[0] = usr_size + sys_workspace_size;
  
  auto dt = context->GetInputTensor(0)->GetDataType();
  int dt_size;
  if (dt == ge::DT_FLOAT || dt == ge::DT_INT32) { if(dt == ge::DT_INT32 && n <= 40 && n > 32) context->SetTilingKey(5); else context->SetTilingKey(1);}
  else if(dt == ge::DT_BOOL) {context->SetTilingKey(4);}
  else if(dt == ge::DT_BF16 || dt == ge::DT_INT16 || dt == ge::DT_FLOAT16) {context->SetTilingKey(2);}
  else {context->SetTilingKey(3);}

  tiling.set_m(m);
  tiling.set_n(n);
  tiling.set_k(k);
  if(axis == 0) mode = 1;
  else if(axis == dims - 1) mode = 2;
  else mode = 3;
  std::cout << "m: " << m << " n: " << n << " k: " << k << " mode: " << mode << std::endl;
  tiling.set_mode(mode);
  int cores = m*n*k / 256;
  if(cores > 40) cores = 40;
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
class Unpack : public OpDef {
public:
    explicit Unpack(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT16, ge::DT_INT8, ge::DT_UINT8, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT16, ge::DT_INT8, ge::DT_UINT8, ge::DT_BOOL})
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
