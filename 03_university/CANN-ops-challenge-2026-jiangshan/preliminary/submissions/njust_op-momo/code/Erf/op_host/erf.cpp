#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aivCoreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    const uint32_t length = static_cast<uint32_t>(tensorX->GetShapeSize());

    uint32_t blockNum = 1;

    // 小 tensor 强制单核，避免调度开销 > 计算收益
    // 每核至少需要处理 MIN_ELEMS_PER_CORE 个元素才值得开多核
    // 从测试数据反推：optimal 1.42us 是单核最快，说明小 tensor 就该单核
    // 阈值设 8192：低于此值单核跑更快（调度省下的时间 > 并行节省的时间）
    constexpr uint32_t MIN_ELEMS_PER_CORE = 8192;
    if (length > MIN_ELEMS_PER_CORE) {
        // 计算实际需要的核数（每核至少处理 MIN_ELEMS_PER_CORE 个元素）
        uint32_t neededCores = (length + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE;
        blockNum = std::min(aivCoreNum, neededCores);
    }

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    tiling->length = length;
    tiling->blockNum = blockNum;

    context->SetBlockDim(blockNum);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
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