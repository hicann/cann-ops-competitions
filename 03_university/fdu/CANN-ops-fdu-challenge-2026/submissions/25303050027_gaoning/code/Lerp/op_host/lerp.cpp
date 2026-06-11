// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "../op_kernel/lerp_tiling.h"
#include "../op_kernel/tiling_key_lerp.h"

namespace {
constexpr size_t START_INDEX = 0;
constexpr size_t END_INDEX = 1;
constexpr size_t Y_INDEX = 0;

static bool IsSupportedDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT;
}

static bool IsSameShape(const gert::Shape *lhs, const gert::Shape *rhs)
{
    if (lhs == nullptr || rhs == nullptr) {
        return false;
    }
    if (lhs->GetDimNum() != rhs->GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs->GetDimNum(); ++i) {
        if (lhs->GetDim(i) != rhs->GetDim(i)) {
            return false;
        }
    }
    return true;
}

static uint64_t CeilDiv(uint64_t value, uint64_t factor)
{
    return (factor == 0) ? 0 : (value + factor - 1) / factor;
}

static uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return (align == 0) ? value : CeilDiv(value, align) * align;
}

static uint64_t AlignDown(uint64_t value, uint64_t align)
{
    return (align == 0) ? value : (value / align) * align;
}

static uint32_t SelectMode(float weight)
{
    if (weight == 0.0f) {
        return LERP_MODE_COPY_START;
    }
    if (weight == 1.0f) {
        return LERP_MODE_COPY_END;
    }
    return LERP_MODE_NORMAL;
}

static uint64_t CalcDataBlockElems(int dtypeSize)
{
    return std::max<uint64_t>(
        1ULL, static_cast<uint64_t>(LERP_DATA_BLOCK_BYTES) / static_cast<uint64_t>(dtypeSize));
}

static uint64_t CalcCacheLineElems(int dtypeSize)
{
    return std::max<uint64_t>(
        1ULL, static_cast<uint64_t>(LERP_CACHE_LINE_BYTES) / static_cast<uint64_t>(dtypeSize));
}

static uint64_t SelectTileLength(uint64_t ubSize, int dtypeSize, uint64_t dataBlockElems)
{
    uint64_t tileLength = LERP_DEFAULT_TILE_LENGTH;
    if (ubSize > LERP_UB_RESERVED_BYTES) {
        const uint64_t usableUb = ubSize - LERP_UB_RESERVED_BYTES;
        const uint64_t bytesPerTile =
            static_cast<uint64_t>(dtypeSize) * static_cast<uint64_t>(LERP_LOCAL_BUFFER_NUM) *
            static_cast<uint64_t>(LERP_BUFFER_NUM);
        if (bytesPerTile > 0) {
            tileLength = std::min<uint64_t>(tileLength, usableUb / bytesPerTile);
        }
    }

    tileLength = AlignDown(tileLength, dataBlockElems);
    if (tileLength < dataBlockElems) {
        tileLength = dataBlockElems;
    }
    return tileLength;
}

static void SelectCoreSplit(uint64_t length, int32_t maxCoreNum, int dtypeSize,
                            uint64_t cacheLineElems, uint32_t &blockDim,
                            uint64_t &perCoreElements)
{
    if (length == 0) {
        blockDim = 1;
        perCoreElements = cacheLineElems;
        return;
    }

    const uint64_t maxCores = static_cast<uint64_t>((maxCoreNum > 0) ? maxCoreNum : 1);
    const uint64_t minElemsPerCore = std::max<uint64_t>(
        cacheLineElems,
        static_cast<uint64_t>(LERP_MIN_BYTES_PER_CORE) / static_cast<uint64_t>(dtypeSize));

    // 小shape不强行铺满全部core，避免调度开销超过并行收益；
    // 大shape仍然会使用更多core，提高整体吞吐。
    const uint64_t usefulCores = std::min<uint64_t>(
        maxCores, std::max<uint64_t>(1ULL, CeilDiv(length, minElemsPerCore)));

    perCoreElements = AlignUp(CeilDiv(length, usefulCores), cacheLineElems);
    blockDim = static_cast<uint32_t>(std::max<uint64_t>(1ULL, CeilDiv(length, perCoreElements)));
}

static ge::graphStatus CopyStartShapeToOutput(gert::InferShapeContext *context)
{
    const gert::Shape *startShape = context->GetInputShape(START_INDEX);
    const gert::Shape *endShape = context->GetInputShape(END_INDEX);
    gert::Shape *yShape = context->GetOutputShape(Y_INDEX);
    if (startShape == nullptr || endShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (!IsSameShape(startShape, endShape)) {
        return ge::GRAPH_FAILED;
    }

    *yShape = *startShape;
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

    const gert::Tensor *tensorStart = context->GetRequiredInputTensor(START_INDEX);
    const gert::Tensor *tensorEnd = context->GetRequiredInputTensor(END_INDEX);
    if (tensorStart == nullptr || tensorEnd == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtypeStart = tensorStart->GetDataType();
    if (!IsSupportedDtype(dtypeStart) || tensorEnd->GetDataType() != dtypeStart) {
        return ge::GRAPH_FAILED;
    }

    const int dtypeSize = ge::GetSizeByDataType(dtypeStart);
    if (dtypeSize <= 0) {
        return ge::GRAPH_FAILED;
    }

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

    const uint64_t length = static_cast<uint64_t>(tensorStart->GetShapeSize());
    const uint64_t dataBlockElems = CalcDataBlockElems(dtypeSize);
    const uint64_t cacheLineElems = CalcCacheLineElems(dtypeSize);
    const uint64_t tileLength = SelectTileLength(ubSize, dtypeSize, dataBlockElems);

    uint32_t blockDim = 1;
    uint64_t perCoreElements = cacheLineElems;
    SelectCoreSplit(length, numCoresAiv, dtypeSize, cacheLineElems, blockDim, perCoreElements);

    tiling->length = length;
    tiling->perCoreElements = perCoreElements;
    tiling->tileLength = static_cast<uint32_t>(tileLength);
    tiling->mode = SelectMode(*attrWeight);
    tiling->weight = *attrWeight;

    uint32_t DT_START = static_cast<uint32_t>(dtypeStart);
    ASCENDC_TPL_SEL_PARAM(context, DT_START);
    context->SetBlockDim(blockDim);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace != nullptr) {
        currentWorkspace[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    return CopyStartShapeToOutput(context);
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const ge::DataType startDtype = context->GetInputDataType(START_INDEX);
    if (!IsSupportedDtype(startDtype) || context->GetInputDataType(END_INDEX) != startDtype) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(Y_INDEX, startDtype);
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
