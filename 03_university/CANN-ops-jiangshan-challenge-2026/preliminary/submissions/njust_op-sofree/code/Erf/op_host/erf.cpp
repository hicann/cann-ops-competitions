#include <algorithm>
#include <cstdint>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/erf_tiling.h"
#include "../op_kernel/tiling_key_erf.h"

namespace optiling {
constexpr uint32_t MODE_TINY8 = 0;
constexpr uint32_t MODE_LEN31 = 1;
constexpr uint32_t MODE_LEN33 = 2;
constexpr uint32_t MODE_LEN127 = 3;
constexpr uint32_t MODE_SMALL = 4;
constexpr uint32_t MODE_MID = 5;
constexpr uint32_t MODE_LARGE = 6;
constexpr uint32_t ALIGN_NUM = 8;
constexpr uint32_t SPLIT_ALIGN_NUM = 128;

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t coreNum = platform.GetCoreNumAiv();
    const gert::Tensor *x = context->GetRequiredInputTensor(0);
    ge::DataType dtype = x->GetDataType();
    uint32_t length = x->GetShapeSize();
    uint32_t DT_X = static_cast<uint32_t>(dtype);

    uint32_t mode = MODE_LARGE;
    uint32_t tileLength = 8192;
    uint32_t granularity = 8192;

    if (length <= 8) {
        mode = MODE_TINY8;
        tileLength = 1024;
        granularity = 4096;
    } else if (length <= 31) {
        mode = MODE_LEN31;
        tileLength = 1024;
        granularity = 4096;
    } else if (length <= 33) {
        mode = MODE_LEN33;
        tileLength = 1024;
        granularity = 4096;
    } else if (length <= 127) {
        mode = MODE_LEN127;
        tileLength = 1024;
        granularity = 4096;
    } else if (length <= 512) {
        mode = MODE_SMALL;
        tileLength = 1024;
        granularity = 4096;
    } else if (length <= 65536) {
        mode = MODE_MID;
        tileLength = 4096;
        granularity = 4096;
    }
    bool isAligned = (length & (ALIGN_NUM - 1)) == 0;
    ASCENDC_TPL_SEL_PARAM(context, DT_X, mode, isAligned);

    uint32_t maxBlockDim = coreNum > 0 ? static_cast<uint32_t>(coreNum) : 1;
    uint32_t blockDim = 1;
    if (length > 4096) {
        blockDim = (length + granularity - 1) / granularity;
        blockDim = std::max<uint32_t>(1, std::min(maxBlockDim, blockDim));
    }
    uint32_t groupNum = (length + SPLIT_ALIGN_NUM - 1) / SPLIT_ALIGN_NUM;
    if (groupNum != 0) {
        blockDim = std::min(blockDim, groupNum);
    }
    uint32_t alignedGroups = length / SPLIT_ALIGN_NUM;
    uint32_t baseGroups = blockDim == 0 ? 0 : alignedGroups / blockDim;
    uint32_t tailGroups = blockDim == 0 ? 0 : alignedGroups - baseGroups * blockDim;

    ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
    tiling->length = length;
    tiling->blockDim = blockDim;
    tiling->tileLength = baseGroups;
    tiling->mode = mode | (tailGroups << 2);

    uint32_t launchBlockDim = platform.CalcTschBlockDim(blockDim, 0, coreNum);
    context->SetBlockDim(launchBlockDim);
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

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
