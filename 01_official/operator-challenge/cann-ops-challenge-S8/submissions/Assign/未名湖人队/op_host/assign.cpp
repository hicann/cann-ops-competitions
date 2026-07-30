#include "../op_kernel/assign_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static uint64_t LargeCopyThresholdBytes(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 16ULL * 1024;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
            return 16ULL * 1024;
        case ge::DT_UINT8:
        case ge::DT_INT8:
        case ge::DT_BOOL:
        default:
            return 256ULL * 1024;
    }
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    constexpr uint32_t kSmallBlockDim = 8;
    constexpr uint32_t kLargeBlockDim = 32;
    constexpr uint64_t kCopyAlignElements = 32;
    const gert::StorageShape* shape = context->GetInputShape(0);
    uint64_t totalSize = 1;
    for (int i = 0; i < shape->GetStorageShape().GetDimNum(); ++i) {
        totalSize *= static_cast<uint64_t>(shape->GetStorageShape().GetDim(i));
    }

    const ge::DataType dtype = context->GetInputDesc(0)->GetDataType();
    const int32_t elementBytes = ge::GetSizeByDataType(dtype);
    const uint64_t totalBytes = totalSize * static_cast<uint64_t>(elementBytes > 0 ? elementBytes : 1);
    const uint64_t largeCopyThresholdBytes = LargeCopyThresholdBytes(dtype);
    const uint32_t blockDim = totalBytes >= largeCopyThresholdBytes ? kLargeBlockDim : kSmallBlockDim;
    const uint64_t rawPerCoreSize = (totalSize + blockDim - 1) / blockDim;
    const uint64_t perCoreSize = (rawPerCoreSize + kCopyAlignElements - 1) / kCopyAlignElements * kCopyAlignElements;
    AssignTilingData tiling;
    tiling.set_totalSize(totalSize);
    tiling.set_perCoreSize(perCoreSize);
    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Assign : public OpDef {
public:
    explicit Assign(const char* name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> dtypes = {
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32,
            ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL};
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("ref").ParamType(REQUIRED).DataType(dtypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("value").ParamType(REQUIRED).DataType(dtypes).Format(formats).UnknownShapeFormat(formats);
        this->Attr("use_locking").AttrType(OPTIONAL).Bool(false);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Assign);
}
