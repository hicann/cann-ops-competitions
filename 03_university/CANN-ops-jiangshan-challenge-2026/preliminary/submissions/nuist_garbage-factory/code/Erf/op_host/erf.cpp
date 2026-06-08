// Host侧Tiling实现
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNum();
    
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint32_t inputLength = inputNum * typeLength;
    // BLOCK_SIZE 自适应：小数据避免padding开销，大数据提升DMA效率
    uint32_t BLOCK_SIZE;
    if (inputNum > 65536) {
        BLOCK_SIZE = 512;     // ms级大数据

    } else {
        BLOCK_SIZE = 64;      // μs级小数据
    }

    uint32_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    coreNum = std::min(coreNum, inputLengthAlgin32 / BLOCK_SIZE);
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));
    uint32_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
    uint32_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;
    context->SetBlockDim(coreNum);


    
	uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint32_t BUFFER_NUM = 2;
    // 单缓冲：总 buffer = inQueueX(1个) + outQueueZ(1个) = 2
    // 省下的UB空间让每个tile更大，减少循环次数
    uint32_t tileBlockNum = (ubSize * 8 / 10) / (BLOCK_SIZE * BUFFER_NUM * 2);
    if (tileBlockNum == 0) tileBlockNum = 1;
    uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / typeLength;
    
    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;
    
    // The total length of a large block of data is 32B larger than that of a small block of data
    everyCoreInputBlockNum += 1;
    uint32_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;
    // 示例: 获取算子输入数组信息
    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x = tensor_x->GetDataType(); // 获取数据类型
    // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);
    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->tailBlockNum = tailBlockNum;
    
    // 配置workspace大小
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


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
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

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
}