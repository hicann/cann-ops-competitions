// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

#include "../op_kernel/clip_by_value_tiling.h"
#include "../op_kernel/tiling_key_clip_by_value.h"

namespace {
constexpr size_t X_INDEX = 0;
constexpr size_t Y_INDEX = 0;

static bool IsSupportedDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT || dtype == ge::DT_INT32;
}

static uint64_t CeilDivU64(uint64_t a, uint64_t b)
{
    return (b == 0) ? 0 : (a + b - 1) / b;
}

static uint64_t AlignUpU64(uint64_t value, uint64_t align)
{
    if (align == 0) {
        return value;
    }
    return ((value + align - 1) / align) * align;
}

static uint64_t AlignDownU64(uint64_t value, uint64_t align)
{
    if (align == 0) {
        return value;
    }
    return (value / align) * align;
}

static ge::graphStatus CopyInputShapeToOutput(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(X_INDEX);
    gert::Shape *yShape = context->GetOutputShape(Y_INDEX);
    if (xShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 直接复制Shape对象，避免手动SetDimNum/AppendDim在不同CANN版本中的接口兼容问题。
    *yShape = *xShape;
    return ge::GRAPH_SUCCESS;
}
}  // namespace

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

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(X_INDEX);
    if (tensorX == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType dtypeX = tensorX->GetDataType();
    if (!IsSupportedDtype(dtypeX)) {
        return ge::GRAPH_FAILED;
    }

    const int dtypeSize = ge::GetSizeByDataType(dtypeX);
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
        CLIP_BY_VALUE_CACHE_LINE_BYTES / static_cast<uint64_t>(dtypeSize));

    // Maxs/Mins的calCount选用保守上限，避免一次向量指令处理元素数过大。
    const uint64_t maxVectorCalCount = (dtypeSize == static_cast<int>(sizeof(uint16_t)))
        ? CLIP_BY_VALUE_MAX_CAL_COUNT_16BIT
        : CLIP_BY_VALUE_MAX_CAL_COUNT_32BIT;

    // 当前kernel使用一进一出两个Queue，UB按两份tile预留，再留出少量余量。
    uint64_t tileLength = CLIP_BY_VALUE_DEFAULT_TILE_LENGTH;
    if (ubSize > CLIP_BY_VALUE_UB_RESERVED_BYTES) {
        const uint64_t usableUb = ubSize - CLIP_BY_VALUE_UB_RESERVED_BYTES;
        const uint64_t ubLimitedTile = usableUb / (static_cast<uint64_t>(dtypeSize) * CLIP_BY_VALUE_LOCAL_BUFFER_NUM);
        tileLength = std::min<uint64_t>(tileLength, ubLimitedTile);
    }
    tileLength = std::min<uint64_t>(tileLength, maxVectorCalCount);
    tileLength = AlignDownU64(tileLength, dataBlockElems);
    if (tileLength < dataBlockElems) {
        tileLength = dataBlockElems;
    }

    tiling->length = length;
    tiling->perCoreElements = cacheLineElems;
    tiling->tileLength = static_cast<uint32_t>(tileLength);
    tiling->reserved = 0;
    tiling->minValue = *attrMin;
    tiling->maxValue = *attrMax;

    uint32_t blockDim = 1;
    if (length > 0) {
        const uint32_t usefulCores = static_cast<uint32_t>(
            std::min<uint64_t>(static_cast<uint64_t>(numCoresAiv),
                               std::max<uint64_t>(1ULL, CeilDivU64(length, cacheLineElems))));
        const uint64_t rawPerCore = CeilDivU64(length, static_cast<uint64_t>(usefulCores));
        tiling->perCoreElements = AlignUpU64(rawPerCore, cacheLineElems);
        blockDim = static_cast<uint32_t>(
            std::max<uint64_t>(1ULL, CeilDivU64(length, tiling->perCoreElements)));
    }

    uint32_t DT_X = static_cast<uint32_t>(dtypeX);
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
    return CopyInputShapeToOutput(context);
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    ge::DataType inputDtype = context->GetInputDataType(X_INDEX);
    if (!IsSupportedDtype(inputDtype)) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(Y_INDEX, inputDtype);
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
