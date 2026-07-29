
#include "erfinv_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  ErfinvTilingData tiling;
  auto input_shape = context->GetInputShape(0)->GetStorageShape();

    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize(); //输入数量
    uint32_t inputBytes = GetSizeByDataType(context->GetInputDesc(0)->GetDataType()); //输入类型
    uint32_t inputLength = inputBytes * inputNum; //输入长度

    uint32_t BLOCK_DIM = (inputLength+255)/256;
    if(BLOCK_DIM > 40)
        BLOCK_DIM = 40;
    
    uint32_t ALIGN_NUM = 512 / inputBytes;
    if(inputLength < 1024*1000*100)
    {
        ALIGN_NUM = 256 / inputBytes;
    }
    uint32_t totalLengthAligned = ((inputNum + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
    uint32_t bigNum = (totalLengthAligned / ALIGN_NUM) % BLOCK_DIM;
    uint32_t bigLength = ((totalLengthAligned / BLOCK_DIM + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
    uint32_t smallLength = (totalLengthAligned / BLOCK_DIM / ALIGN_NUM) * ALIGN_NUM;


    tiling.set_bigNum(bigNum);
    tiling.set_bigLength(bigLength);
    tiling.set_smallLength(smallLength);

    context->SetBlockDim(BLOCK_DIM);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
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
class Erfinv : public OpDef {
public:
    explicit Erfinv(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Erfinv);
}
