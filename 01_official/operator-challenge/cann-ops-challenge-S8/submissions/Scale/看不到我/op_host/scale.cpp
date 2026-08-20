
#include "scale_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  ScaleTilingData tiling;
  
  // 1. 获取输入 Shape 和 属性
  const gert::StorageShape* input_shape = context->GetInputShape(0); // Input
  const gert::StorageShape* scale_shape = context->GetInputShape(1); // Scale
  const gert::RuntimeAttrs *attrs = context->GetAttrs();
  
  int64_t axis = 1;
  // CANN 规范中通常通过 Attrs 获取，假设索引 0 是 axis
  if (attrs != nullptr) {
      const int64_t *axis_attr = attrs->GetInt(0);
      if (axis_attr != nullptr) axis = *axis_attr;
  }

  // 2. 核心数学降维：将任意维度的 Tensor 展平成 [M, N, K]
  int dims = input_shape->GetStorageShape().GetDimNum();
  if (axis < 0) axis += dims; // 处理 Python 风格的负数轴

  uint32_t m = 1, n = 1, k = 1;
  uint32_t total_size = 1;

  // 2.1 计算总数据量 和 M (axis 之前的维度乘积)
  for (int i = 0; i < dims; i++) {
    int dim_val = input_shape->GetStorageShape().GetDim(i);
    total_size *= dim_val;
    if (i < axis) {
      m *= dim_val;
    }
  }

  // 2.2 计算 N (严格按照 CPU 的逻辑，N 就是 scale tensor 自身的总大小)
  int scale_dims = scale_shape->GetStorageShape().GetDimNum();
  for (int i = 0; i < scale_dims; i++) {
      n *= scale_shape->GetStorageShape().GetDim(i);
  }

  // 2.3 计算 K (剩余的内部广播维度)
  // 防御性编程：避免除以 0（虽管理论上不会出现）
  if (m * n != 0) {
      k = total_size / (m * n);
  } else {
      k = 1;
  }

  std::cout << "M: " << m << " N: " << n << " K: " << k << " Total: " << total_size << std::endl;

  // 3. 设置数据类型 TilingKey
  auto dt = context->GetInputTensor(0)->GetDataType();
  if (dt == ge::DT_FLOAT) { 
      context->SetTilingKey(1); // Float32
  } else if (dt == ge::DT_FLOAT16) { 
      context->SetTilingKey(2); // Float16
  } else {
      context->SetTilingKey(3); // BF16
  }

  // 4. 计算并行度 (多核切分)
  auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t sys_workspace_size = ascendc_platform.GetLibApiWorkSpaceSize();
  size_t *current_workspace = context->GetWorkspaceSizes(1);
  current_workspace[0] = sys_workspace_size + 1024; 

  uint32_t core_num = ascendc_platform.GetCoreNumAiv(); 
  if (core_num == 0) core_num = 1;

  uint32_t cores = (total_size + 255) / 256; 
  if (cores > core_num) cores = core_num; 
  if (cores % 2) cores += 1;              
  context->SetBlockDim(cores);

  // 5. 写入 Tiling 数据
  tiling.set_m(m);
  tiling.set_n(n);
  tiling.set_k(k);
  tiling.set_total_size(total_size);

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
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("axis").Int();
        this->Attr("num_axes").Int();
        this->Attr("scale_from_blob").Bool();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Scale);
}
