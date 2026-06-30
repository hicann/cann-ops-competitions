#include "erfinv_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
constexpr int64_t ALIGN_BYTES = 1024;
constexpr uint64_t RESERVE_UB = 8 * 1024;
constexpr int32_t MIN_CORE_NUM = 2;

static inline int64_t BytesPerElemForTiling(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
            return 14;
        case ge::DT_FLOAT:
            return 32;
        case ge::DT_BF16:
        default:
            return 32;
    }
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ErfinvTilingData tiling;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    int64_t totalLength = 1;
    for (int i = 0; i < xShape->GetStorageShape().GetDimNum(); ++i) {
        totalLength *= xShape->GetStorageShape().GetDim(i);
        if (totalLength <= 0) {
            return ge::GRAPH_FAILED;
        }
    }

    const ge::DataType dtype = context->GetInputDesc(0)->GetDataType();
    const int64_t dtypeSize = ge::GetSizeByDataType(dtype);
    if (dtypeSize <= 0 || dtypeSize > ALIGN_BYTES) {
        return ge::GRAPH_FAILED;
    }
    const int64_t elemPerBlock = ALIGN_BYTES / dtypeSize;
    const int64_t totalBlock = (totalLength + elemPerBlock - 1) / elemPerBlock;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize > RESERVE_UB) {
        ubSize -= RESERVE_UB;
    }

    const int64_t bytesPerElem = BytesPerElemForTiling(dtype);
    const int64_t ubPerBlock = elemPerBlock * bytesPerElem;
    int64_t tileBlock = (ubPerBlock > 0) ? (static_cast<int64_t>(ubSize) / ubPerBlock) : 1;
    if (tileBlock < 1) {
        tileBlock = 1;
    }
    const int64_t tileLength = tileBlock * elemPerBlock;

    int64_t coreNum = platform.GetCoreNumAiv();
    if (coreNum < MIN_CORE_NUM) {
        coreNum = MIN_CORE_NUM;
    }
    if (coreNum > totalBlock) {
        coreNum = totalBlock;
    }
    if (coreNum < 1) {
        coreNum = 1;
    }

    const int64_t perCoreBlock = totalBlock / coreNum;
    const int64_t lastCoreBlock = totalBlock - perCoreBlock * coreNum;
    const int64_t alignedLength = totalBlock * elemPerBlock;

    tiling.set_totalLength(alignedLength);
    tiling.set_tileLength(static_cast<int32_t>(tileLength));
    tiling.set_perCoreBlock(static_cast<int32_t>(perCoreBlock));
    tiling.set_lastCoreBlock(static_cast<int32_t>(lastCoreBlock));
    context->SetBlockDim(static_cast<uint32_t>(coreNum));

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
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
        this->AICore().AddConfig("ascend310b");
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Erfinv);
}
