// Host侧Tiling实现
#include <algorithm>
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    // 1. 获取可用AI Core数量
    uint32_t coreNum = ascendcPlatform.GetCoreNum();

    // 2. 获取输入元素个数
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();

    // 3. 获取输入数据类型字节数
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);

    // Erf当前只注册了float，一般typeLength为4
    if (typeLength == 0) {
        return ge::GRAPH_FAILED;
    }

    // 4. 设置模板选择参数
    ge::DataType dtypeX = context->GetInputDesc(0)->GetDataType();
    uint32_t DT_X = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();

    // 如果erf_tiling.h中保留了length字段，建议写入原始元素个数
    //tiling->length = inputNum;

    constexpr uint32_t BLOCK_SIZE = 32;

    /*
     * Erf是单输入单输出算子，但如果Kernel侧计算Erf时还申请了多个中间LocalTensor，
     * 这里的BUFFER_NUM需要包含输入、输出、中间计算Buffer的数量。
     *
     * 你原来使用的是7，这里继续保留。
     * 如果Kernel侧只申请xLocal和yLocal，则可以改成2。
     */
    constexpr uint32_t BUFFER_NUM = 7;

    /*
     * 如果你的Kernel侧DataCopy或某些API存在单次搬运上限，可以保留该限制。
     * 注意：必须以32B为单位限制，不能直接按元素数截断，否则会破坏32B对齐。
     */
    constexpr uint32_t MAX_COPY_BYTES = 65535;
    constexpr uint32_t MAX_TILE_BLOCK_NUM = MAX_COPY_BYTES / BLOCK_SIZE;

    // 5. 处理空输入
    if (inputNum == 0) {
        context->SetBlockDim(1);

        tiling->smallCoreDataNum = 0;
        tiling->bigCoreDataNum = 0;
        tiling->finalBigTileNum = 0;
        tiling->finalSmallTileNum = 0;
        tiling->tileDataNum = BLOCK_SIZE / typeLength;
        tiling->smallTailDataNum = 0;
        tiling->bigTailDataNum = 0;
        tiling->tailBlockNum = 0;

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }

    // 6. 按字节数计算输入总长度，并向上对齐到32B
    uint32_t inputLength = inputNum * typeLength;
    uint32_t inputLengthAlign32 =
        ((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

    // 7. 计算总的32B数据块数量
    uint32_t totalBlockNum = inputLengthAlign32 / BLOCK_SIZE;

    // 8. 实际使用的核数：不能超过32B数据块数量
    coreNum = std::min(coreNum, totalBlockNum);
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));

    context->SetBlockDim(coreNum);

    /*
     * 9. 核间切分
     *
     * everyCoreInputBlockNum:
     *   每个小核处理的32B块数量
     *
     * tailBlockNum:
     *   前tailBlockNum个核为大核，每个大核多处理1个32B块
     */
    uint32_t everyCoreInputBlockNum = totalBlockNum / coreNum;
    uint32_t tailBlockNum = totalBlockNum % coreNum;

    /*
     * 10. 核内切分
     *
     * UB空间按照BUFFER_NUM等分。
     * tileBlockNum表示单次最多处理多少个32B块。
     */
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    uint32_t tileBlockNum =
        static_cast<uint32_t>((ubSize / BLOCK_SIZE) / BUFFER_NUM);

    tileBlockNum = std::max(tileBlockNum, static_cast<uint32_t>(1));

    // 限制单次DataCopy大小，同时保持32B对齐
    tileBlockNum = std::min(tileBlockNum, MAX_TILE_BLOCK_NUM);
    tileBlockNum = std::max(tileBlockNum, static_cast<uint32_t>(1));

    // 单次处理的元素个数
    uint32_t tileDataNum = tileBlockNum * BLOCK_SIZE / typeLength;

    /*
     * 11. 小核Tiling参数
     *
     * 小核处理 everyCoreInputBlockNum 个32B块。
     */
    uint32_t smallCoreDataNum =
        everyCoreInputBlockNum * BLOCK_SIZE / typeLength;

    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;

    uint32_t finalSmallTileNum =
        (everyCoreInputBlockNum % tileBlockNum) == 0
            ? smallTileNum
            : smallTileNum + 1;

    uint32_t smallTailDataNum =
        smallCoreDataNum - tileDataNum * smallTileNum;

    smallTailDataNum =
        smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    /*
     * 12. 大核Tiling参数
     *
     * 大核比小核多处理1个32B块。
     * 当前tailBlockNum个核为大核。
     */
    uint32_t bigCoreBlockNum = everyCoreInputBlockNum + 1;

    uint32_t bigCoreDataNum =
        bigCoreBlockNum * BLOCK_SIZE / typeLength;

    uint32_t bigTileNum = bigCoreBlockNum / tileBlockNum;

    uint32_t finalBigTileNum =
        (bigCoreBlockNum % tileBlockNum) == 0
            ? bigTileNum
            : bigTileNum + 1;

    uint32_t bigTailDataNum =
        bigCoreDataNum - tileDataNum * bigTileNum;

    bigTailDataNum =
        bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    // 13. 写入TilingData
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->tailBlockNum = tailBlockNum;

    // 14. workspace大小
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *inputShape = context->GetInputShape(0);
        gert::Shape *outputShape = context->GetOutputShape(0);
        *outputShape = *inputShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
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
}  // namespace ops

