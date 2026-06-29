#include "gelu_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t GM_ALIGN_BYTES = 512;
constexpr uint32_t RESERVED_UB_BYTES = 1024;

static uint32_t CeilDiv(uint32_t value, uint32_t factor)
{
    return (value + factor - 1) / factor;
}

static uint32_t CeilAlign(uint32_t value, uint32_t align)
{
    return CeilDiv(value, align) * align;
}

static uint32_t FloorAlign(uint32_t value, uint32_t align)
{
    return value / align * align;
}
} // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GeluV2TilingData tiling;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    uint32_t dataSize = 1;
    for (int i = 0; i < xShape->GetStorageShape().GetDimNum(); ++i) {
        dataSize *= static_cast<uint32_t>(xShape->GetStorageShape().GetDim(i));
    }
    tiling.set_totalLength(dataSize);

    int64_t approximate = 0;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* approximatePtr = attrs->GetAttrPointer<int64_t>(0);
        if (approximatePtr != nullptr) {
            approximate = *approximatePtr;
        }
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const auto inputDtype = context->GetInputDesc(0)->GetDataType();
    if (inputDtype != ge::DT_FLOAT && inputDtype != ge::DT_FLOAT16 && inputDtype != ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }
    uint32_t dtypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(inputDtype));
    if (dtypeSize == 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize <= RESERVED_UB_BYTES) {
        return ge::GRAPH_FAILED;
    }
    ubSize -= RESERVED_UB_BYTES;

    const bool useTanhApprox = approximate != 0;
    const bool isFloat = inputDtype == ge::DT_FLOAT;
    uint32_t calcBufferCount = useTanhApprox ? (isFloat ? 1 : 2) : (isFloat ? 1 : 3);
    uint32_t alignElems = GM_ALIGN_BYTES / dtypeSize;
    uint32_t ubBytesPerElem = 2 * BUFFER_NUM * dtypeSize + calcBufferCount * sizeof(float);
    uint32_t tileLength = FloorAlign(static_cast<uint32_t>(ubSize / ubBytesPerElem), alignElems);
    if (tileLength == 0) {
        return ge::GRAPH_FAILED;
    }
    tiling.set_tileLength(tileLength);

    uint32_t coreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    if (coreNum == 0) {
        coreNum = 1;
    }

    uint32_t coreDataNum = 0;
    uint32_t usedCoreNum = 1;
    if (dataSize != 0) {
        coreDataNum = CeilAlign(CeilDiv(dataSize, coreNum), tileLength);
        usedCoreNum = CeilDiv(dataSize, coreDataNum);
    }
    tiling.set_coreDataNum(coreDataNum);
    context->SetBlockDim(usedCoreNum);

    context->SetTilingKey(useTanhApprox ? 1 : 0);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

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
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class GeluV2 : public OpDef {
public:
    explicit GeluV2(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("approximate").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(GeluV2);
} // namespace ops
