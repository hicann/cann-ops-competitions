// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>

#include "../op_kernel/clip_by_value_tiling.h"
#include "../op_kernel/tiling_key_clip_by_value.h"

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

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    if (tensorX == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const int dtypeSize = ge::GetSizeByDataType(tensorX->GetDataType());
    if (dtypeSize <= 0) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t length = static_cast<uint64_t>(tensorX->GetShapeSize());

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const float *attrMin = attrs->GetFloat(0);
    const float *attrMax = attrs->GetFloat(1);
    if (attrMin == nullptr || attrMax == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ClipByValueTilingData *tiling = context->GetTilingData<ClipByValueTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t dataBlockElems = std::max<uint64_t>(1ULL,
        CLIP_BY_VALUE_DATA_BLOCK_BYTES / static_cast<uint64_t>(dtypeSize));
    const uint64_t cacheLineElems = std::max<uint64_t>(1ULL,
        64ULL / static_cast<uint64_t>(dtypeSize));

    constexpr uint64_t DEFAULT_TILE = 8192;
    constexpr uint64_t UB_RESERVED = 8192;
    uint64_t tileLength = DEFAULT_TILE;
    if (ubSize > UB_RESERVED) {
        const uint64_t usableUb = ubSize - UB_RESERVED;
        const uint64_t ubLimitedTile = usableUb /
            (static_cast<uint64_t>(dtypeSize) * CLIP_BY_VALUE_QUEUE_NUM * 2);
        tileLength = std::min<uint64_t>(tileLength, ubLimitedTile);
    }
    tileLength = std::min<uint64_t>(tileLength,
        (dtypeSize == static_cast<int>(sizeof(uint16_t))) ? 32640ULL : 16320ULL);
    tileLength = (tileLength / dataBlockElems) * dataBlockElems;
    if (tileLength < dataBlockElems) {
        tileLength = dataBlockElems;
    }

    tiling->length = length;
    tiling->tileLength = static_cast<uint32_t>(tileLength);
    tiling->reserved = 0;
    tiling->minValue = *attrMin;
    tiling->maxValue = *attrMax;

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

    uint32_t DT_X = static_cast<uint32_t>(tensorX->GetDataType());
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
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
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
class ClipByValue : public OpDef {
public:
    explicit ClipByValue(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("min").AttrType(REQUIRED).Float();
        this->Attr("max").AttrType(REQUIRED).Float();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(ClipByValue);
}  // namespace ops
