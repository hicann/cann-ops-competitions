#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/lerp_tiling.h"
#include "../op_kernel/tiling_key_lerp.h"


namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 获取算子输入数组信息
        const gert::Tensor *tensor_start = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_end = context->GetRequiredInputTensor(1);
        ge::DataType dtype_start = tensor_start->GetDataType(); // 获取数据类型
        int dtype_size_start = ge::GetSizeByDataType(dtype_start); // 获取数据类型的字长
        uint32_t length_start = tensor_start->GetShapeSize(); // 获取元素个数
        uint32_t size_start = tensor_start->GetSize(); // 获取内存大小

        // 获取算子输入属性
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const float attr_weight = *attrs->GetFloat(0);

        // 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_START = static_cast<uint32_t>(dtype_start);
        ASCENDC_TPL_SEL_PARAM(context, DT_START);

        /* 计算tiling方案并填充tiling结构体 */
       
        const uint32_t BLOCK_SIZE = 32;
        uint32_t size_start_align32 = ((size_start + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

        // 以每个核至少分配1个32B数据块为基准，判断是否启用全部核数：若对齐后的数据块总量过小，核数不足时则至少启用1个核完成计算；
        num_cores_aiv = std::min(num_cores_aiv, size_start_align32 / BLOCK_SIZE);
        num_cores_aiv = std::max(num_cores_aiv, static_cast<uint32_t>(1));

        // 确定实际使用的核数后，计算核均基础处理的 32B 数据块数，对无法均分的剩余数据块，采用前 N 核补块的方式分配
        // 即让前 tailBlockNum 个核各多处理 1 个 32B 数据块，其余核按基础数处理
        uint32_t everyCoreInputBlockNum = size_start_align32 / BLOCK_SIZE / num_cores_aiv;
        uint32_t tailBlockNum = (size_start_align32 / BLOCK_SIZE) % num_cores_aiv;


        // 基于核可用 UB 总大小与 Buffer 的数量，即可计算出单个 Buffer 最多可容纳的 32B 数据块个数，以及对应的元素个数。
        // 在这里，我们将 ubDataNumber 乘以 2，以支持后续设置 BUFFER_NUM = 2
        uint32_t ubDataNumber = 0;
        if (dtype_start == ge::DT_FLOAT) {
            ubDataNumber = 3 * 2;  // start, end, output
        } else {
            ubDataNumber = 3 * 2 + 3 * 2; // 补上 3 个 buffer
        }
        uint32_t tileBlockNum = (ub_size / BLOCK_SIZE ) / ubDataNumber;
        uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / dtype_size_start;

        // 对于小核
        uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / dtype_size_start;
        uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
        uint32_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
        uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
        smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

        // 对于大核
        everyCoreInputBlockNum += 1;
        uint32_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / dtype_size_start;
        uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
        uint32_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
        uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
        bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

        // 填充 tiling 结构体
        LerpTilingData *tiling = context->GetTilingData<LerpTilingData>();
        tiling->smallCoreDataNum = smallCoreDataNum;
        tiling->bigCoreDataNum = bigCoreDataNum;
        tiling->tileDataNum = tileDataNum;
        tiling->smallTailDataNum = smallTailDataNum;
        tiling->bigTailDataNum = bigTailDataNum;
        tiling->finalSmallTileNum = finalSmallTileNum;
        tiling->finalBigTileNum = finalBigTileNum;
        tiling->tailBlockNum = tailBlockNum;
        tiling->weight = attr_weight;
        tiling->oneMinusW = 1.0 - attr_weight;

        tiling->w_is_0 = tiling->w_is_1 = false;
        if (std::abs(attr_weight) < 1e-6) {
            tiling->w_is_0 = true;
        } else if (std::abs(attr_weight - 1.0f) < 1e-6) {
            tiling->w_is_1 = true;
        }

        // 配置启动核数
        context->SetBlockDim(num_cores_aiv);

        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape* start_shape = context->GetInputShape(0);
        gert::Shape* out_shape = context->GetOutputShape(0);
        *out_shape = *start_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const auto inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
class Lerp : public OpDef {
public:
    explicit Lerp(const char* name) : OpDef(name)
    {
        this->Input("start")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("end")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("weight").Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Lerp);
}
