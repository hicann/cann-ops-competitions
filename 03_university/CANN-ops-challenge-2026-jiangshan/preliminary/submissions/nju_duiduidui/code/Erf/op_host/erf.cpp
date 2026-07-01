#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
namespace {
constexpr uint32_t ALIGN_ELEMS = 32 / sizeof(float);
constexpr uint32_t BUFFER_NUMS = 5;
constexpr uint32_t MIN_CORE_LENGTH = 4096;
constexpr uint64_t UB_RESERVE_BYTES = 1024;

inline uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return (value + align - 1U) / align * align;
}

inline uint32_t AlignDown(uint32_t value, uint32_t align)
{
    return value / align * align;
}
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t numCoresAiv = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (numCoresAiv == 0) {
        numCoresAiv = 1;
    }

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    if (tensorX == nullptr || tensorX->GetDataType() != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    uint32_t totalLength = static_cast<uint32_t>(tensorX->GetShapeSize());
    uint32_t usedCoreNum = 1;
    uint32_t formerLength = totalLength;
    uint32_t tailLength = totalLength;

    if (totalLength > 0) {
        uint32_t idealCoreLength = (totalLength + numCoresAiv - 1U) / numCoresAiv;
        if (idealCoreLength < MIN_CORE_LENGTH) {
            idealCoreLength = MIN_CORE_LENGTH;
        }
        formerLength = AlignUp(idealCoreLength, ALIGN_ELEMS);
        if (formerLength == 0) {
            formerLength = ALIGN_ELEMS;
        }

        usedCoreNum = (totalLength + formerLength - 1U) / formerLength;
        if (usedCoreNum == 0) {
            usedCoreNum = 1;
        }

        if (usedCoreNum == 1) {
            formerLength = totalLength;
            tailLength = totalLength;
        } else {
            tailLength = totalLength - (usedCoreNum - 1U) * formerLength;
        }
    }

    uint64_t usableUbBytes = (ubSize > UB_RESERVE_BYTES) ? (ubSize - UB_RESERVE_BYTES) : ubSize;
    uint32_t tileLength = static_cast<uint32_t>(usableUbBytes / (BUFFER_NUMS * sizeof(float)));
    tileLength = AlignDown(tileLength, ALIGN_ELEMS);
    if (tileLength == 0) {
        tileLength = ALIGN_ELEMS;
    }

    uint32_t DT_X = static_cast<uint32_t>(tensorX->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    tiling->totalLength = totalLength;
    tiling->usedCoreNum = usedCoreNum;
    tiling->formerLength = formerLength;
    tiling->tailLength = tailLength;
    tiling->tileLength = tileLength;

    context->SetBlockDim(usedCoreNum);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto *inputDesc = context->GetInputDesc(0);
    if (inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    return context->SetOutputDataType(0, inputDesc->GetDataType());
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
