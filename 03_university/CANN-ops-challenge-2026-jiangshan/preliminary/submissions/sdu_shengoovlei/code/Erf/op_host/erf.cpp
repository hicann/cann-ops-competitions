// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t num_cores_aiv = static_cast<uint32_t>(platform.GetCoreNumAiv());
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType(); // 获取数据类型
        uint32_t length_x = tensor_x->GetShapeSize(); // 获取元素个数
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        uint32_t block_num = length_x == 0 ? 1 : ((length_x + 7) / 8);
        uint32_t task_core_num;
        if (block_num <= 256) {
            task_core_num = 8;
        } else if (block_num <= 384) {
            task_core_num = (block_num + 41) / 42;
        } else if (block_num <= 512) {
            task_core_num = (block_num + 39) / 40;
        } else {
            task_core_num = (block_num + 51) / 52;
        }
        uint32_t core_num = task_core_num < num_cores_aiv ? task_core_num : num_cores_aiv;
        core_num = core_num == 0 ? 1 : core_num;
        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        tiling->length = length_x;
        tiling->coreNum = core_num;
        // 配置启动核数
        context->SetBlockDim(core_num);
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
