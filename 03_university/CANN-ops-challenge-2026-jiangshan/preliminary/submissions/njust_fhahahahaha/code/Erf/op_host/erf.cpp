// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"
#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

#define typeLength      4
#define BLOCK_SIZE      512
#define ubDataNumber    3
#define TOTAL_BUF_NUM   7

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

        // 获取算子输入信息
        uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        uint32_t inputLength = inputNum * typeLength;

        // 512B对齐
        uint32_t inputBlockNum = inputLength / BLOCK_SIZE;  // 对齐512B
        uint32_t intputTailNum = (inputLength % BLOCK_SIZE) / typeLength;
        intputTailNum = (intputTailNum + 7)& ~7;  // 32B 对齐

        /* 核数 */
        uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv()) - 1;     // 留一个作为尾核

        /* 获取UB大小，确定tile切分粒度.
           ubDataNumber=T缓冲数=3, 但tileDataNum需保证全部物理buffer装入UB:
           Pipeline: 3*TBuf + 2*TQue(各BUFFER_NUM=2物理buf) = 7物理buf */
        uint64_t ubSize;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        uint32_t TileBlockNum = (ubSize-2048) / TOTAL_BUF_NUM / BLOCK_SIZE;    // 一个buf最多存多少个block(预留一点空间)

        uint32_t EachCoreBlockNum = inputBlockNum / coreNum;   // 数据均分给每个核
        uint32_t BigCoreNum = inputBlockNum % coreNum;     // 多少个核要多处理一个block

        uint32_t TileNum = EachCoreBlockNum / TileBlockNum;         // 满buf块的搬运次数
        uint32_t TailBlockNum = EachCoreBlockNum % TileBlockNum;    // 尾块，小于一个buf容量。

        // 一次搬运不完再启用pipeline
        uint32_t usePipeline = (TileNum >= 1);
        uint32_t useCoreNum = std::min(coreNum + 1, inputBlockNum+1);  // 每个核至少搬运一个512B，否则不启用 
        context->SetBlockDim(useCoreNum); 
        context->SetTilingKey(GET_TPL_TILING_KEY(usePipeline));

        /* 填充结构体 (全部元素计数, kernel侧无除法) */
        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        tiling->intputTailNum = static_cast<uint16_t>(intputTailNum);       // 凑不够512B的尾块数据量
        tiling->TailCoreIdx = static_cast<uint16_t>(useCoreNum - 1);
        tiling->processDataNum   = TileBlockNum * BLOCK_SIZE / typeLength;
        tiling->tailDataNum      = TailBlockNum * BLOCK_SIZE / typeLength;
        tiling->smallCoreDataNum = EachCoreBlockNum * BLOCK_SIZE / typeLength;
        tiling->tileNum          = TileNum;
        tiling->bigCoreNum       = BigCoreNum;

        // 配置workspace大小 (本算子无需额外workspace)
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

