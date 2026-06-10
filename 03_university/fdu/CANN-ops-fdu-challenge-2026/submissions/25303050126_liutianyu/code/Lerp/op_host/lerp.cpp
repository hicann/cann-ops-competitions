// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>

#include "../op_kernel/lerp_tiling.h"
#include "../op_kernel/tiling_key_lerp.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t numCoresAiv = platform.GetCoreNumAiv();
    if (numCoresAiv <= 0) {
        numCoresAiv = 1;
    }

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const gert::Tensor *tensorStart = context->GetRequiredInputTensor(0);
    if (tensorStart == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const int dtypeSize = ge::GetSizeByDataType(tensorStart->GetDataType());
    if (dtypeSize <= 0) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t length = static_cast<uint64_t>(tensorStart->GetShapeSize());

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const float *attrWeight = attrs->GetFloat(0);
    if (attrWeight == nullptr) {
        return ge::GRAPH_FAILED;
    }

    LerpTilingData *tiling = context->GetTilingData<LerpTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t dataBlockElems = std::max<uint64_t>(1ULL,
        LERP_DATA_BLOCK_BYTES / static_cast<uint64_t>(dtypeSize));
    const uint64_t cacheLineElems = std::max<uint64_t>(1ULL,
        64ULL / static_cast<uint64_t>(dtypeSize));

    constexpr uint64_t DEFAULT_TILE = 8192;
    constexpr uint64_t UB_RESERVED = 8192;
    uint64_t tileLength = DEFAULT_TILE;
    if (ubSize > UB_RESERVED) {
        const uint64_t usableUb = ubSize - UB_RESERVED;
        const uint64_t ubLimitedTile = usableUb /
            (static_cast<uint64_t>(dtypeSize) * LERP_LOCAL_BUFFER_NUM);
        tileLength = std::min<uint64_t>(tileLength, ubLimitedTile);
    }
    // Maxs/Mins calCount上限: 16bit 32640(128*255), 32bit 16320(64*255)
    uint64_t maxVecCount = (dtypeSize == static_cast<int>(sizeof(uint16_t)))
        ? 32640ULL : 16320ULL;
    tileLength = std::min<uint64_t>(tileLength, maxVecCount);
    tileLength = (tileLength / dataBlockElems) * dataBlockElems;
    if (tileLength < dataBlockElems) {
        tileLength = dataBlockElems;
    }

    tiling->length = length;
    tiling->tileLength = static_cast<uint32_t>(tileLength);
    tiling->reserved = 0;
    tiling->weight = *attrWeight;

    uint32_t blockDim = 1;
    if (length > 0) {
        const uint32_t usefulCores = static_cast<uint32_t>(
            std::min<uint64_t>(static_cast<uint64_t>(numCoresAiv),
                               std::max<uint64_t>(1ULL,
                                   (length + cacheLineElems - 1) / cacheLineElems)));
        const uint64_t rawPerCore = (length + static_cast<uint64_t>(usefulCores) - 1) /
                                     static_cast<uint64_t>(usefulCores);
        tiling->perCoreElements = ((rawPerCore + cacheLineElems - 1) / cacheLineElems) *
                                   cacheLineElems;
        blockDim = static_cast<uint32_t>(
            std::max<uint64_t>(1ULL,
                (length + tiling->perCoreElements - 1) / tiling->perCoreElements));
    } else {
        tiling->perCoreElements = cacheLineElems;
    }

    uint32_t DT_X = static_cast<uint32_t>(tensorStart->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);
    context->SetBlockDim(blockDim);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *startShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    if (startShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *yShape = *startShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

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
}  // namespace ops
