// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/lerp_tiling.h"
#include "../op_kernel/tiling_key_lerp.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        // 获取算子输入数组信息
        const gert::Tensor *tensor_start = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_end = context->GetRequiredInputTensor(1);
        ge::DataType dtype_start = tensor_start->GetDataType();
        uint32_t length_start = tensor_start->GetShapeSize(); // 元素个数
        // 获取算子输入属性
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const float *attr_weight = attrs->GetFloat(0);
        // 配置tiling key, 从而实现kernel侧不同数据的区分
        uint32_t DT_START = static_cast<uint32_t>(dtype_start);
        ASCENDC_TPL_SEL_PARAM(context, DT_START);
        // 计算tiling方案并填充tiling结构体
        LerpTilingData *tiling = context->GetTilingData<LerpTilingData>();
        tiling->length = length_start;
        tiling->weight = *attr_weight;

        // 多核切分: 按 TILE_LENGTH 粒度分配
        uint32_t totalTiles = (length_start + TILE_LENGTH - 1) / TILE_LENGTH;
        uint32_t blockNum = (totalTiles < static_cast<uint32_t>(num_cores_aiv)) ?
                             totalTiles : static_cast<uint32_t>(num_cores_aiv);
        if (blockNum == 0) blockNum = 1;
        uint32_t tilesPerCore = totalTiles / blockNum;
        uint32_t tailTiles = totalTiles - tilesPerCore * (blockNum - 1);

        tiling->blockNum = blockNum;
        tiling->numPerCore = tilesPerCore * TILE_LENGTH;
        tiling->tailNumLastCore = tailTiles * TILE_LENGTH;

        // 配置启动核数
        context->SetBlockDim(blockNum);
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class Lerp : public OpDef {
    public:
        explicit Lerp(const char *name) : OpDef(name) {
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

