
#include "assign_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  AssignTilingData tiling;
  const gert::StorageShape* input_shape = context->GetInputShape(0);
  const gert::StorageShape* other_shape = context->GetInputShape(1);
  int input_dim_num = input_shape->GetStorageShape().GetDimNum();
  int other_dim_num = other_shape->GetStorageShape().GetDimNum();
  std::vector<int32_t> input_dims_raw(input_dim_num);
  std::vector<int32_t> other_dims_raw(other_dim_num);
  for (uint32_t i = 0; i < input_dim_num; ++i) {
      input_dims_raw[i] = input_shape->GetStorageShape().GetDim(i);
  }
  for (uint32_t i = 0; i < other_dim_num; ++i) {
      other_dims_raw[i] = other_shape->GetStorageShape().GetDim(i);
  }

    int max_dim = std::max(input_dim_num, other_dim_num);
    std::vector<int64_t> align_input(max_dim, 1);
    std::vector<int64_t> align_other(max_dim, 1);

    // 将实际维度填入尾部
    for(int i = 0; i < input_dim_num; ++i) {
        align_input[max_dim - input_dim_num + i] = input_dims_raw[i];
    }
    for(int i = 0; i < other_dim_num; ++i) {
        align_other[max_dim - other_dim_num + i] = other_dims_raw[i];
    }

    // 2. 连续维度合并算法 (Flattening)
    std::vector<int64_t> merged_input;
    std::vector<int64_t> merged_other;

    if (max_dim > 0) {
        merged_input.push_back(align_input[0]);
        merged_other.push_back(align_other[0]);
    }

    for (int i = 1; i < max_dim; ++i) {
        // 判断前一维的广播属性
        bool prev_is_broadcast = (align_other[i-1] == 1 && align_input[i-1] > 1);
        // 判断当前维的广播属性
        bool curr_is_broadcast = (align_other[i] == 1 && align_input[i] > 1);
        
        // 如果两者的广播属性完全一致，就可以无脑合并！
        // 场景 A: 连续两维都不需要广播 (other 对应维度 == input 对应维度)
        // 场景 B: 连续两维 other 都为 1 (都需要广播)
        if (prev_is_broadcast == curr_is_broadcast) {
            merged_input.back() *= align_input[i];
            merged_other.back() *= align_other[i];
        } else {
            // 广播边界发生切换，不能合并，开启新的一维
            merged_input.push_back(align_input[i]);
            merged_other.push_back(align_other[i]);
        }
    }
    bool is_broadcast = false;
    uint64_t total_elements = 1;
    for (size_t i = 0; i < merged_input.size(); ++i) {
        total_elements *= merged_input[i];
        if (merged_input[i] != merged_other[i]) {
            is_broadcast = true;
        }
    }
  
  uint32_t final_input_shape[8] = {0};
  uint32_t final_other_shape[8] = {0};
  for(size_t i = 0; i < merged_input.size(); ++i) {
    final_input_shape[i] = merged_input[i];
    final_other_shape[i] = merged_other[i];
  }

  // ================= 加入的打印代码 =================
  std::cout << "[Tiling Debug] Merged Input Shape: [";
  for(size_t i = 0; i < merged_input.size(); ++i) {
      std::cout << merged_input[i] << (i == merged_input.size() - 1 ? "" : ", ");
  }
  std::cout << "]" << std::endl;

  std::cout << "[Tiling Debug] Merged Other Shape: [";
  for(size_t i = 0; i < merged_other.size(); ++i) {
      std::cout << merged_other[i] << (i == merged_other.size() - 1 ? "" : ", ");
  }
  std::cout << "]" << std::endl;
  // =================================================

  size_t usr_size = 1024;
  auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t sys_workspace_size = ascendc_platform.GetLibApiWorkSpaceSize();
  size_t *current_workspace = context->GetWorkspaceSizes(1);
  current_workspace[0] = usr_size + sys_workspace_size;

  auto dt = context->GetInputTensor(0)->GetDataType();
  int dt_size;
  int tiling_key;
  if (dt == ge::DT_FLOAT || dt == ge::DT_INT32) { tiling_key = 1;}
  else if(dt == ge::DT_BOOL) {tiling_key = 4;}
  else if(dt == ge::DT_BF16 || dt == ge::DT_INT16 || dt == ge::DT_FLOAT16) {tiling_key = 2;}
  else {tiling_key = 3;}
  if (is_broadcast) {
    if (tiling_key == 1) tiling_key = 5;        // 4B 广播
    else if (tiling_key == 2) tiling_key = 6;   // 2B 广播
    else tiling_key = 7;                         // 1B 广播 (int8/uint8/bool)
  }
  context->SetTilingKey(tiling_key);
  tiling.set_size(total_elements);
  tiling.set_dims(merged_input.size());
  tiling.set_input_shape(final_input_shape);
  tiling.set_other_shape(final_other_shape);
  int cores = int(total_elements / 256) > 40 ? 40 : int(total_elements / 256);
  if (cores % 2) cores += 1;          // 核数必须偶数，奇数向上加一
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
class Assign : public OpDef {
public:
    explicit Assign(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT16, ge::DT_INT8, ge::DT_UINT8, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("other")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT16, ge::DT_INT8, ge::DT_UINT8, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("use_locking").Bool();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Assign);
}
