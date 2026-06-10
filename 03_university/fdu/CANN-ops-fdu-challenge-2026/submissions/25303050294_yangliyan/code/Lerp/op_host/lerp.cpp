// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/lerp_tiling.h"
#include "../op_kernel/tiling_key_lerp.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t num_cores_aiv = platform.GetCoreNumAiv();

    uint64_t ub_size;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    const gert::Tensor *tensor_start = context->GetRequiredInputTensor(0);
    const gert::Tensor *tensor_end = context->GetRequiredInputTensor(1);

    ge::DataType dtype_start = tensor_start->GetDataType();
    int dtype_size_start = ge::GetSizeByDataType(dtype_start);
    uint32_t length_start = tensor_start->GetShapeSize();

    uint32_t inputLength = length_start * dtype_size_start;

    const uint32_t BLOCK_SIZE = 32;
    uint32_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    num_cores_aiv = std::min(num_cores_aiv, static_cast<int32_t>(inputLengthAlgin32 / BLOCK_SIZE));
    num_cores_aiv = std::max(num_cores_aiv, static_cast<int32_t>(1));

    context->SetBlockDim(num_cores_aiv);

    uint32_t totalBlockNum = inputLengthAlgin32 / BLOCK_SIZE;
    uint32_t tailBlockNum = totalBlockNum % static_cast<uint32_t>(num_cores_aiv);
    uint32_t everyCoreInputBlockNum = totalBlockNum / static_cast<uint32_t>(num_cores_aiv);

    uint32_t ubDataNumber = 3;
    uint32_t tileBlockNum = (ub_size / BLOCK_SIZE) / ubDataNumber;
    if (tileBlockNum == 0) {
        tileBlockNum = 1;
    }
    uint32_t tileDataNum = tileBlockNum * BLOCK_SIZE / dtype_size_start;

    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / dtype_size_start;
    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    everyCoreInputBlockNum += 1;
    uint32_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / dtype_size_start;
    uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *attr_weight = attrs->GetFloat(0);

    uint32_t DT_START = static_cast<uint32_t>(dtype_start);
    ASCENDC_TPL_SEL_PARAM(context, DT_START);

    LerpTilingData *tiling = context->GetTilingData<LerpTilingData>();
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->tailBlockNum = tailBlockNum;
    tiling->weight = *attr_weight;

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *intputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    *outputShape = *intputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Lerp : public OpDef {
public:
    explicit Lerp(const char *name) : OpDef(name)
    {
        this->Input("start")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("end")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("weight").AttrType(REQUIRED).Float();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Lerp);
}  // namespace ops
