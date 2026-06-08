#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace {
constexpr uint32_t FLOAT_BYTES = 4;
constexpr uint32_t PERF_BUFFER_NUM = 2;
constexpr uint32_t LOCAL_TENSOR_COUNT = PERF_BUFFER_NUM * 2 + 2;  // x/y queues + z/tmp buffers
constexpr uint32_t ALIGN_NUM = 32;
constexpr uint32_t PERF_TILE_LENGTH = 8192;
constexpr uint32_t MIN_ELEMENTS_PER_CORE = 1024;
constexpr uint32_t MAX_CORE_NUM_LIMIT = 0;  // 0 means use all AIV cores from platform.

uint32_t AlignDown(uint32_t value, uint32_t align)
{
    return value / align * align;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t maxCoreNum = static_cast<uint32_t>(std::max<int32_t>(platform.GetCoreNumAiv(), 1));
    if (MAX_CORE_NUM_LIMIT > 0) {
        maxCoreNum = std::min<uint32_t>(maxCoreNum, MAX_CORE_NUM_LIMIT);
    }

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    ge::DataType dtypeX = tensorX->GetDataType();
    uint32_t length = static_cast<uint32_t>(tensorX->GetShapeSize());

    uint32_t blockDim = 1;
    if (length > 0) {
        uint32_t targetBlockDim = (length + MIN_ELEMENTS_PER_CORE - 1) / MIN_ELEMENTS_PER_CORE;
        blockDim = std::min<uint32_t>(maxCoreNum, std::max<uint32_t>(targetBlockDim, 1));
    }

    uint32_t tileLength = PERF_TILE_LENGTH;
    if (ubSize > 0) {
        uint64_t maxTileByUb = ubSize / (LOCAL_TENSOR_COUNT * FLOAT_BYTES);
        uint32_t ubTile = static_cast<uint32_t>(
            std::min<uint64_t>(maxTileByUb, PERF_TILE_LENGTH));
        ubTile = AlignDown(ubTile, ALIGN_NUM);
        if (ubTile >= ALIGN_NUM) {
            tileLength = ubTile;
        }
    }

    uint32_t DT_X = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    tiling->length = length;
    tiling->blockDim = blockDim;
    tiling->tileLength = tileLength;

    context->SetBlockDim(blockDim);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = *xShape;
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
    explicit Erf(const char *name) : OpDef(name)
    {
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
