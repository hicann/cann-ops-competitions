#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
  static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t num_cores_aiv = platform.GetCoreNumAiv();
    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    uint32_t length_x = tensor_x->GetShapeSize();
    ge::DataType dtype_x = tensor_x->GetDataType();

    uint32_t ALIGN_NUM = 32 / ge::GetSizeByDataType(dtype_x); 
    uint32_t used_cores = num_cores_aiv;

    if (length_x < ALIGN_NUM) {
      used_cores = 1;
    } else {
      used_cores = length_x / ALIGN_NUM;
      if (used_cores > num_cores_aiv) {
        used_cores = num_cores_aiv;
      }
    }
    if (used_cores == 0) { used_cores = 1; }

    uint32_t blockLength = (length_x / used_cores / ALIGN_NUM) * ALIGN_NUM;
    if (blockLength == 0) {
      blockLength = ALIGN_NUM;
    }
    uint32_t blockTail = length_x - blockLength * (used_cores - 1);

    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    tiling->totalLength = length_x;
    tiling->blockLength = blockLength;
    tiling->blockTail = blockTail;
    tiling->usedCores = used_cores;

    context->SetBlockDim(used_cores);
     
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
  }
} // namespace optiling

namespace ge {
  static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *x_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
  }
  static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const ge::DataType x_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, x_dtype);
    return ge::GRAPH_SUCCESS;
  }
} // namespace ge

namespace ops {
  class Erf : public OpDef {
  public:
    explicit Erf(const char *name) : OpDef(name) {
      this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND});
      this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND});
      this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
      this->AICore()
        .SetTiling(optiling::TilingFunc)
        .AddConfig("ascend910b");
    }
  };
  OP_ADD(Erf);
} // namespace ops