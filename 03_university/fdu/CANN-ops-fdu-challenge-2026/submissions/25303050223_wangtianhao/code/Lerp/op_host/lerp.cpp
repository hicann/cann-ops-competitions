#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/lerp_tiling.h"
#include "../op_kernel/tiling_key_lerp.h"

namespace {
constexpr uint32_t SINGLE_CORE_THRESHOLD = 8192;

static uint32_t AlignUp(uint32_t x, uint32_t align)
{
    return (x + align - 1) / align * align;
}

static uint32_t GetDataTypeBytes(ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT) {
        return 4;
    }
    return 2;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t maxCoreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());

    const gert::Tensor *tensor_start = context->GetRequiredInputTensor(0);
    ge::DataType dtype_start = tensor_start->GetDataType();
    uint32_t length = static_cast<uint32_t>(tensor_start->GetShapeSize());

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *attr_weight = attrs->GetFloat(0);

    uint32_t DT_X = static_cast<uint32_t>(dtype_start);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    uint32_t elemBytes = GetDataTypeBytes(dtype_start);
    uint32_t alignNum = 32 / elemBytes;
    if (alignNum == 0) {
        alignNum = 1;
    }

    uint32_t coreNum = 1;
    if (length > SINGLE_CORE_THRESHOLD && maxCoreNum > 0) {
        coreNum = maxCoreNum;
        if (length < coreNum) {
            coreNum = length;
        }
    }

    uint32_t blockLength = 0;
    if (length == 0) {
        coreNum = 1;
        blockLength = 0;
    } else {
        uint32_t baseBlockLength = (length + coreNum - 1) / coreNum;
        blockLength = AlignUp(baseBlockLength, alignNum);
    }

    LerpTilingData *tiling = context->GetTilingData<LerpTilingData>();
    tiling->length = length;
    tiling->blockLength = blockLength;
    tiling->coreNum = coreNum;
    tiling->weight = *attr_weight;

    context->SetBlockDim(coreNum);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *start_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *start_shape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Lerp : public OpDef {
public:
    explicit Lerp(const char *name) : OpDef(name)
    {
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
}