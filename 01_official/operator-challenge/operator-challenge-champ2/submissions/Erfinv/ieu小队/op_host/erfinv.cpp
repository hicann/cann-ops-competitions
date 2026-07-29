#include "erfinv_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
const uint32_t FALLBACK_BLOCK_DIM = 40;
const uint32_t TILE_ALIGN     = 256;
const uint32_t MAX_TILE_HALF  = 8192;
const uint32_t MAX_TILE_FLOAT = 6912;
const uint32_t MAX_TILE_BF16  = 8192;

static uint32_t RoundUp(uint32_t val, uint32_t align) {
    return ((val + align - 1) / align) * align;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ErfinvTilingData tiling;
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    uint32_t total = static_cast<uint32_t>(x_shape->GetStorageShape().GetShapeSize());

    uint32_t coreNum = FALLBACK_BLOCK_DIM;
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        uint32_t aiv = ascendcPlatform.GetCoreNumAiv();
        if (aiv > 0) coreNum = aiv;
    }

    auto dt = context->GetInputDesc(0)->GetDataType();
    uint64_t tilingKey = 0;
    uint32_t maxTile = MAX_TILE_HALF;
    if (dt == ge::DT_FLOAT) {
        tilingKey = 1;
        maxTile = MAX_TILE_FLOAT;
    } else if (dt == ge::DT_BF16) {
        tilingKey = 2;
        maxTile = MAX_TILE_BF16;
    }

    context->SetTilingKey(tilingKey);

    uint32_t blockLen = (total + coreNum - 1) / coreNum;
    if (blockLen == 0) blockLen = TILE_ALIGN;
    blockLen = RoundUp(blockLen, TILE_ALIGN);

    uint32_t blockDim = (total + blockLen - 1) / blockLen;
    if (blockDim == 0) blockDim = 1;
    if (blockDim > coreNum) blockDim = coreNum;

    uint32_t tileLen = (blockLen < maxTile) ? blockLen : maxTile;
    tileLen = RoundUp(tileLen, TILE_ALIGN);
    if (tileLen == 0) tileLen = TILE_ALIGN;

    uint32_t tileNum = (blockLen + tileLen - 1) / tileLen;
    if (tileNum == 0) tileNum = 1;

    context->SetBlockDim(blockDim);

    tiling.set_totalLength(total);
    tiling.set_blockLength(blockLen);
    tiling.set_tileLength(tileLen);
    tiling.set_tileNum(tileNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Erfinv : public OpDef {
public:
    explicit Erfinv(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Erfinv);
}
