#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

#include "../op_kernel/addcmul_tiling.h"
#include "../op_kernel/tiling_key_addcmul.h"

namespace {
constexpr uint32_t BLOCK_BYTES = 32;

uint32_t GetDtypeBytes(ge::DataType dtype) {
    if (dtype == ge::DT_INT8) {
        return 1;
    }
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        return 2;
    }
    return 4;
}

uint32_t AlignDown(uint32_t value, uint32_t unit) {
    return unit == 0 ? value : value / unit * unit;
}

uint32_t ChooseTileElements(uint64_t ubBytes, ge::DataType dtype, uint32_t alignElems) {
    uint32_t bufferSlices = (dtype == ge::DT_INT8) ? 12u : 10u;
    uint32_t blockCount = static_cast<uint32_t>((ubBytes / BLOCK_BYTES / 2u) / bufferSlices);
    blockCount = blockCount <= 8u ? blockCount : AlignDown(blockCount, 8u);
    return std::max(alignElems, blockCount * alignElems);
}

void ReadFlatLengths(gert::TilingContext *context, uint32_t &inputElems, uint32_t &x1Elems,
                     uint32_t &x2Elems, uint32_t &outElems, uint32_t &shortElems) {
    inputElems = static_cast<uint32_t>(context->GetInputShape(0)->GetStorageShape().GetShapeSize());
    x1Elems = static_cast<uint32_t>(context->GetInputShape(1)->GetStorageShape().GetShapeSize());
    x2Elems = static_cast<uint32_t>(context->GetInputShape(2)->GetStorageShape().GetShapeSize());
    outElems = std::max(inputElems, std::max(x1Elems, x2Elems));
    shortElems = std::min(inputElems, std::min(x1Elems, x2Elems));
}

uint32_t ClampBroadcastTile(uint32_t tileElems, uint32_t shortElems, uint32_t alignElems, bool hasBroadcast) {
    if (!hasBroadcast || shortElems == 0) {
        return tileElems;
    }
    uint32_t candidate = std::min(tileElems, shortElems);
    candidate = AlignDown(candidate, alignElems);
    while (candidate > alignElems && (shortElems % candidate) != 0) {
        candidate -= alignElems;
    }
    return std::max(alignElems, candidate);
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);

    ge::DataType dtype = context->GetInputDesc(0)->GetDataType();
    uint32_t dtypeKey = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, dtypeKey);

    uint32_t inputElems = 0;
    uint32_t x1Elems = 0;
    uint32_t x2Elems = 0;
    uint32_t outElems = 0;
    uint32_t shortElems = 0;
    ReadFlatLengths(context, inputElems, x1Elems, x2Elems, outElems, shortElems);

    uint32_t alignElems = BLOCK_BYTES / GetDtypeBytes(dtype);
    uint32_t tileElems = ChooseTileElements(ubBytes, dtype, alignElems);
    tileElems = ClampBroadcastTile(tileElems, shortElems, alignElems, outElems != shortElems);

    uint32_t coreCount = platform.GetCoreNum();
    uint32_t fullTileCount = tileElems == 0 ? 0 : outElems / tileElems;
    coreCount = std::min(coreCount, fullTileCount);
    coreCount = coreCount == 0 ? 1 : coreCount;

    uint32_t splitUnit = alignElems * 8u;
    uint32_t elemsPerCore = AlignDown(outElems / coreCount, splitUnit);
    uint32_t tailElems = outElems - coreCount * elemsPerCore;

    AddcmulTilingData *tiling = context->GetTilingData<AddcmulTilingData>();
    tiling->unusedTileCount = 0;
    tiling->alignElements = alignElems;
    tiling->tileElements = tileElems;
    tiling->activeCores = coreCount;
    tiling->elemsPerCore = elemsPerCore;
    tiling->tailElements = tailElems;
    tiling->outputElements = outElems;
    tiling->inputElements = inputElems;
    tiling->x1Elements = x1Elems;
    tiling->x2Elements = x2Elems;

    context->SetBlockDim(coreCount);
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Addcmul : public OpDef {
public:
    explicit Addcmul(const char *name) : OpDef(name) {
        this->Input("input_data")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};

OP_ADD(Addcmul);
}  // namespace ops
