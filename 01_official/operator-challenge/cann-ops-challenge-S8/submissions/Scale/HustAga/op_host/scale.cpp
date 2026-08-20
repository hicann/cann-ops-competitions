
#include "scale_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
const int BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  ScaleTilingData tiling;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint64_t ubSize;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  auto num_cores = ascendcPlatform.GetCoreNum();
  auto dt = context->GetInputTensor(0)->GetDataType();
  uint32_t sizeofdata = 2;
  if (dt == ge::DT_FLOAT)
    sizeofdata = 4;
  const gert::StorageShape* input_shape = context->GetInputShape(0);
  const gert::StorageShape* scale_shape = context->GetInputShape(1);
  const gert::StorageShape* bias_shape = context->GetInputShape(2);
  const int64_t* axis = context->GetAttrs()->GetInt(0);
  const int64_t* num_axes = context->GetAttrs()->GetInt(1);
  const bool* scale_from_blob = context->GetAttrs()->GetBool(2);
  int start_axis = *axis, end_axis, batch_sz = 1, useCoreNum, tiling_key, tileDataNum, innerStride = 1, outerStride = 1;
  if (start_axis < 0) start_axis += input_shape->GetStorageShape().GetDimNum();
  if (*scale_from_blob)
  {
    if (*num_axes == -1)
      end_axis = input_shape->GetStorageShape().GetDimNum();
    else
      end_axis = start_axis + *num_axes;
  }
  else
    end_axis = start_axis + scale_shape->GetStorageShape().GetDimNum();
  if (end_axis == input_shape->GetStorageShape().GetDimNum())
  {
    if (start_axis == 0)
    {
      for (int i = 0; i < input_shape->GetStorageShape().GetDimNum(); i ++)
        batch_sz *= input_shape->GetStorageShape().GetDim(i);
      tiling_key = 0;
    }
    else
    {
      for (int i = 0; i < start_axis; i ++)
        batch_sz *= input_shape->GetStorageShape().GetDim(i);
      for (int i = start_axis; i < end_axis; i ++)
        innerStride *= input_shape->GetStorageShape().GetDim(i);
      tiling_key = 1;
    }
    int tileBlockNum = ubSize / BLOCK_SIZE;
    if (dt == ge::DT_FLOAT || dt == ge::DT_FLOAT16)
      tileBlockNum /= 4;
    else
      tileBlockNum /= 10;
    tileDataNum = tileBlockNum * BLOCK_SIZE / sizeofdata;
  }
  else
  {
    for (int i = 0; i < end_axis; i ++)
      batch_sz *= input_shape->GetStorageShape().GetDim(i);
    for (int i = end_axis; i < input_shape->GetStorageShape().GetDimNum(); i ++)
      innerStride *= input_shape->GetStorageShape().GetDim(i);
    for (int i = start_axis; i < end_axis; i ++)
      outerStride *= input_shape->GetStorageShape().GetDim(i);
    tiling_key = 2;
    int tileBlockNum = ubSize / BLOCK_SIZE;
    if (dt == ge::DT_BF16)
      tileBlockNum /= 4;
    else
    {
      // tileBlockNum -= innerStride * sizeofdata / BLOCK_SIZE;
      tileBlockNum /= 3;
    }
    tileDataNum = tileBlockNum * BLOCK_SIZE / sizeofdata;
  }
  useCoreNum = (batch_sz < num_cores) ? batch_sz : num_cores;
  int bigCoreNum = batch_sz % useCoreNum;
  int smallCoreProcessNum = batch_sz / useCoreNum;
  int bigCoreProcessNum = (bigCoreNum == 0) ? smallCoreProcessNum : smallCoreProcessNum + 1;
  tiling.set_tileDataNum(tileDataNum);
  tiling.set_innerStride(innerStride);
  tiling.set_outerStride(outerStride);
  tiling.set_bigCoreNum(bigCoreNum);
  tiling.set_bigCoreProcessNum(bigCoreProcessNum);
  tiling.set_smallCoreProcessNum(smallCoreProcessNum);
  context->SetTilingKey(tiling_key);
  context->SetBlockDim(useCoreNum);
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
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
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
