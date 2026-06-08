#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

namespace optiling {

BEGIN_TILING_DATA_DEF(ErfTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalSize);
    TILING_DATA_FIELD_DEF(uint32_t, tileSize);
    TILING_DATA_FIELD_DEF(uint32_t, perCoreSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Erf, ErfTilingData)

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    ErfTilingData tiling;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t numCores = ascendcPlatform.GetCoreNum();

    const gert::StorageShape* xShape = context->GetInputShape(0);
    uint32_t totalSize = 1;
    for (int i = 0; i < xShape->GetStorageShape().GetDimNum(); ++i) {
        totalSize *= xShape->GetStorageShape().GetDim(i);
    }

    uint32_t blockDim;
    uint32_t tileSize;
    uint32_t perCoreSize;

    if (totalSize == 0) {
        blockDim = 1;
        tileSize = 8;
        perCoreSize = 0;
    } else if (totalSize < 8) {
        blockDim = 1;
        tileSize = 8;
        perCoreSize = 8;
    } else if (totalSize < 128) {
        // 小数据量单核处理，避免多核同步 overhead
        blockDim = 1;
        perCoreSize = (totalSize + 7) / 8 * 8;
        tileSize = perCoreSize;
    } else {
        blockDim = std::min(numCores, totalSize / 8);
        if (blockDim == 0) blockDim = 1;

        perCoreSize = (totalSize + blockDim - 1) / blockDim;
        perCoreSize = (perCoreSize + 7) / 8 * 8;

        tileSize = (perCoreSize <= 8192) ? perCoreSize : 8192;
    }

    tiling.set_totalSize(totalSize);
    tiling.set_tileSize(tileSize);
    tiling.set_perCoreSize(perCoreSize);

    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    auto dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, dtype);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class Erf : public OpDef {
public:
    explicit Erf(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Erf);

} // namespace ops
