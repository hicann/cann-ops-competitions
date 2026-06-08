#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取当前硬件平台的属性，获取可用核数
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t num_cores = static_cast<uint32_t>(platform.GetCoreNumAiv()); // 获取实际可用的 AIV (Vector Core) 核心数
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        // 计算输入张量总的元素数量
        uint32_t totalLength = static_cast<uint32_t>(tensor_x->GetShapeSize());

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 128B alignment: float32 = 4B, 32 elements per burst
        uint32_t ALIGN_NUM = 128 / sizeof(float);

        uint32_t coreDataNum = (totalLength + num_cores - 1) / num_cores;
        coreDataNum = (coreDataNum + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;

        uint32_t usedCoreNum = (totalLength + coreDataNum - 1) / coreDataNum;
        if (usedCoreNum == 0) usedCoreNum = 1;

        // tile = 9824 floats, 5 slots x 9824 x 4B = 191.9KB (UB 192KB)
        uint32_t tileDataNum = 9824;

        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        tiling->totalLength = totalLength;
        tiling->coreDataNum = coreDataNum;
        tiling->tileDataNum = tileDataNum;
        tiling->usedCoreNum = usedCoreNum;

        // 设置并发调度的Block维度（发给多少个核同时跑）
        context->SetBlockDim(usedCoreNum);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *x_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);
        *y_shape = *x_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }
}

namespace ops {
    class Erf : public OpDef {
    public:
        explicit Erf(const char *name) : OpDef(name) {
            this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
            this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
        }
    };
    OP_ADD(Erf);
}
