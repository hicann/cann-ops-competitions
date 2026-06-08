#include <algorithm>

#include "../op_kernel/erf_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

namespace {
constexpr uint32_t kFallbackUbSize = 196608;
constexpr uint32_t kSmallSingleCoreThreshold = 16384;
constexpr uint32_t kDoubleBufferMemoryFactor = 7;
constexpr uint32_t kOptimalTileAlign = 2048;

inline uint32_t AlignDown(uint32_t value, uint32_t align)
{
    return (value / align) * align;
}
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t numCores = platform.GetCoreNumAiv();
    if (numCores == 0) {
        numCores = 1;
    }

    platform.ReserveLocalMemory(platform_ascendc::ReservedSize::RESERVED_SIZE_8K);

    uint64_t ubSize = kFallbackUbSize;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) {
        ubSize = kFallbackUbSize;
    }

    const gert::Tensor* tensorX = context->GetRequiredInputTensor(0);
    uint32_t totalLength = tensorX->GetShapeSize();

    ErfTilingData* tiling = context->GetTilingData<ErfTilingData>();
    if (totalLength == 0) {
        context->SetBlockDim(1);
        tiling->totalLength = 0;
        tiling->blocksPerCore = 0;
        tiling->tailBlocks = 0;
        tiling->tileLength = ERF_ALIGN_ELEMENTS;
        size_t* currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    if (totalLength <= ERF_ALIGN_ELEMENTS) {
        context->SetBlockDim(1);
        tiling->totalLength = totalLength;
        tiling->blocksPerCore = 1;
        tiling->tailBlocks = 0;
        tiling->tileLength = ERF_ALIGN_ELEMENTS;
        size_t* currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    const uint32_t totalBlocks = (totalLength + ERF_ALIGN_ELEMENTS - 1) / ERF_ALIGN_ELEMENTS;

    uint32_t usedCoreNum = numCores;
    if (totalLength <= 49152U) {
        usedCoreNum = 1;
    } else if (totalLength <= (256U * 1024U)) {
        usedCoreNum = std::min<uint32_t>(20, numCores);
    } else if (totalLength <= (1024U * 1024U)) {
        usedCoreNum = std::min<uint32_t>(32, numCores);
    } else {
        usedCoreNum = numCores;
    }
    if (totalBlocks < usedCoreNum) {
        usedCoreNum = totalBlocks;
    }
    usedCoreNum = std::max<uint32_t>(1, std::min<uint32_t>(usedCoreNum, totalBlocks));

    const uint32_t blocksPerCore = totalBlocks / usedCoreNum;
    const uint32_t tailBlocks = totalBlocks % usedCoreNum;

    uint32_t maxTileLength =
        static_cast<uint32_t>(ubSize / (kDoubleBufferMemoryFactor * sizeof(float)));
    maxTileLength = AlignDown(maxTileLength, kOptimalTileAlign);
    if (maxTileLength == 0) {
        maxTileLength = kOptimalTileAlign;
    }

    const uint32_t perCoreBlocks = blocksPerCore + (tailBlocks > 0 ? 1U : 0U);
    const uint32_t perCoreLength = perCoreBlocks * ERF_ALIGN_ELEMENTS;

    uint32_t tileLength = maxTileLength;
    if (perCoreLength <= ERF_ALIGN_ELEMENTS) {
        tileLength = ERF_ALIGN_ELEMENTS;
    } else if (perCoreLength <= 512U) {
        tileLength = 512U;
    } else if (perCoreLength <= 2048U) {
        tileLength = 2048U;
    } else if (perCoreLength <= 4096U) {
        tileLength = 4096U;
    }
    tileLength = std::min(tileLength, maxTileLength);

    tiling->totalLength = totalLength;
    tiling->blocksPerCore = blocksPerCore;
    tiling->tailBlocks = tailBlocks;
    tiling->tileLength = tileLength;

    context->SetBlockDim(usedCoreNum);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class Erf : public OpDef {
public:
    explicit Erf(const char* name) : OpDef(name)
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
            .AddConfig("ascend910b")
            .AddConfig("ascend910_93")
            .AddConfig("ascend310p");
    }
};

OP_ADD(Erf);

}  // namespace ops
