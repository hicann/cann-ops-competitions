// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/clip_by_value_tiling.h"
#include "../op_kernel/tiling_key_clip_by_value.h"


namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 获取平台信息
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t num_cores_aiv = platform.GetCoreNumAiv();                      // 可用于 AIV（AI Vector）计算 的核心数量
    uint64_t ub_size;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);    // 获取单个 AI Core 上 Unified Buffer 的大小

    // 获取算子输入数组信息
    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x = tensor_x->GetDataType();     // 获取数据类型
    int dtype_size = ge::GetSizeByDataType(dtype_x);    // 获取数据类型的字长
    uint32_t input_length = tensor_x->GetShapeSize();   // 获取元素个数
    uint32_t input_size = tensor_x->GetSize();          // 获取内存大小

    // 获取算子输入属性
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *attr_min = attrs->GetFloat(0);
    const float *attr_max = attrs->GetFloat(1);

    // 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    /* 计算tiling方案并填充tiling结构体 */

    // 将输入长度对齐至 32 字节
    const uint32_t BLOCK_SIZE = 32;
    uint32_t input_size_align32 = (((input_size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    
    // 以每个核至少分配1个32B数据块为基准，判断是否启用全部核数：若对齐后的数据块总量过小，核数不足时则至少启用1个核完成计算；
    num_cores_aiv = std::min(num_cores_aiv, input_size_align32 / BLOCK_SIZE);
    num_cores_aiv = std::max(num_cores_aiv, static_cast<uint32_t>(1));
    // 配置启动核数
    context->SetBlockDim(num_cores_aiv);

    // 确定实际使用的核数后，计算核均基础处理的 32B 数据块数，对无法均分的剩余数据块，采用前N核补块的方式分配
    // 即让前 tailBlockNum 个核各多处理 1 个 32B 数据块，其余核按基础数处理
    uint32_t everyCoreInputBlockNum = input_size_align32 / BLOCK_SIZE / num_cores_aiv;
    uint32_t tailBlockNum = (input_size_align32 / BLOCK_SIZE) % num_cores_aiv;

    // 计算出单个Buffer最多可容纳的32B数据块个数，以及对应的元素个数。
    uint32_t ubDataNumber = 3;  // x, y, tmp
    uint32_t tileBlockNum = (ub_size / BLOCK_SIZE ) / ubDataNumber;
    uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / dtype_size;

    // 对于小核
    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / dtype_size;
    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    // 对于大核
    everyCoreInputBlockNum += 1;
    uint32_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / dtype_size;
    uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    // 填充 tiling 结构体
    ClipByValueTilingData *tiling = context->GetTilingData<ClipByValueTilingData>();
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->tailBlockNum = tailBlockNum;
    tiling->min = *attr_min;
    tiling->max = *attr_max;

    // 配置workspace大小
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

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
class ClipByValue : public OpDef {

public:
    explicit ClipByValue(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("min").Float();
        this->Attr("max").Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(ClipByValue);

}
