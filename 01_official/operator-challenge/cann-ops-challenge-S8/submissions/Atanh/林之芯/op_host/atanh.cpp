
#include "atanh_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

const uint32_t BLOCK_SIZE = 32;
const uint32_t BUFFER_NUM = 2;
const uint32_t TILE_NUM = 8;


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  AtanhTilingData tiling;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  auto coreNum = ascendcPlatform.GetCoreNum();   // 获取AI Core数量

  uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
  uint32_t typeBytes = 0;
  typeBytes = ge::GetSizeByDataType(context->GetInputDesc(0)->GetDataType());
  uint32_t inputLength = inputNum * typeBytes;

  //UB
  uint64_t ubSize;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  uint32_t uB_BlockNum = (ubSize / BLOCK_SIZE / BUFFER_NUM);

  AtanhDType mappedDType;
  uint32_t scale = 4;
  ge::DataType dataType = context->GetInputDesc(0)->GetDataType();
  switch (dataType) {
    case ge::DT_FLOAT:   mappedDType = DTYPE_FP32;  scale = 2;  break;
    case ge::DT_FLOAT16: mappedDType = DTYPE_FP16;  scale = 2;  break;
    case ge::DT_BF16:    mappedDType = DTYPE_BF16;  break;
    case ge::DT_INT32:   mappedDType = DTYPE_INT32; break;
    case ge::DT_INT16:   mappedDType = DTYPE_INT16; break;
    case ge::DT_INT8:    mappedDType = DTYPE_INT8;  break;
    case ge::DT_UINT8:   mappedDType = DTYPE_UINT8; break;
    default: return ge::GRAPH_FAILED;
  }
  context->SetTilingKey(mappedDType);
  
  
  uint32_t uselessBlock = uB_BlockNum % scale;
  uint32_t useBlock = uB_BlockNum - uselessBlock;
  uint32_t smallestVariableAllocBlock = useBlock / scale;

  uint32_t tileBlockNum = smallestVariableAllocBlock;
  uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / typeBytes;


  uint32_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);

  coreNum = (coreNum < inputLengthAlgin32 / BLOCK_SIZE) ? coreNum : inputLengthAlgin32 / BLOCK_SIZE;
  coreNum = (coreNum >= 1) ? coreNum : 1;

  uint32_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
  uint32_t tailBlockNum = inputLengthAlgin32 / BLOCK_SIZE % coreNum;
  uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeBytes;   //小核需要处理的数据量
  uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
  uint32_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
  uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
  smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;
  everyCoreInputBlockNum += 1;
  uint32_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeBytes;
  uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
  uint32_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
  uint32_t bigTailDataNum = bigCoreDataNum - (tileDataNum * bigTileNum);
  bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

  tiling.set_size(inputNum);
  tiling.set_tileDataNum(tileDataNum);
  tiling.set_tailBlockNum(tailBlockNum);
  tiling.set_smallCoreDataNum(smallCoreDataNum);
  tiling.set_finalSmallTileNum(finalSmallTileNum);
  tiling.set_smallTailDataNum(smallTailDataNum);
  tiling.set_bigCoreDataNum(bigCoreDataNum);
  tiling.set_finalBigTileNum(finalBigTileNum);
  tiling.set_bigTailDataNum(bigTailDataNum);

  context->SetBlockDim(coreNum);

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
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Atanh);
}
