
#include "gelu_v2_tiling.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"

namespace optiling {

// BUFFER_NUM=2 (double buffering): UB layout per tile:
//   inQueue:  BUFFER_NUM * tile * elemSize
//   outQueue: BUFFER_NUM * tile * elemSize
//   4 fp32 compute buffers: 4 * tile * 4
// Total = (4*elemSize + 16) * tile bytes
// Conservative fallback when platform info unavailable
constexpr uint32_t DEFAULT_MAX_TILE_LEN = 4096;
constexpr uint32_t DEFAULT_UB_SIZE      = 196608; // 192KB (ascend910b)

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GeluV2TilingData tiling;

    // --- block dim and UB size from platform ---
    uint32_t blockDim = 20; // default for ascend910b
    uint64_t ubSize   = DEFAULT_UB_SIZE;
    auto* platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        uint32_t coreNum = platformInfo->GetCoreNum();
        if (coreNum > 0) blockDim = coreNum;
        uint64_t ub = 0;
        platformInfo->GetLocalMemSize(fe::LocalMemType::UB, ub);
        if (ub > 0) ubSize = ub;
    }

    // --- total element count ---
    const gert::StorageShape* xShape = context->GetInputShape(0);
    uint64_t totalLength = 1;
    for (int i = 0; i < (int)xShape->GetStorageShape().GetDimNum(); i++) {
        totalLength *= (uint64_t)xShape->GetStorageShape().GetDim(i);
    }

    // --- data type ---
    auto dtype = context->GetInputDesc(0)->GetDataType();
    int32_t dtypeId;
    uint32_t elemSize;
    uint32_t alignNum;   // minimum elements for 32B alignment
    if (dtype == ge::DT_FLOAT16) {
        dtypeId = 0; elemSize = 2; alignNum = 16;
    } else if (dtype == ge::DT_BF16) {
        dtypeId = 1; elemSize = 2; alignNum = 16;
    } else {  // ge::DT_FLOAT
        dtypeId = 2; elemSize = 4; alignNum = 8;
    }

    // --- approximate attribute ---
    const auto* attrs = context->GetAttrs();
    int64_t approximate = 0;
    if (attrs != nullptr) {
        const int64_t* p = attrs->GetAttrPointer<int64_t>(0);
        if (p != nullptr) approximate = *p;
    }

    // Minimal fp32 scratch bufs per mode (must match kernel FP32_BUFS):
    //   fp32-tanh: 1, all others: 2
    uint32_t numFp32Bufs;
    if (dtypeId == 2 && approximate == 1) {
        numFp32Bufs = 1;  // fp32-tanh: only 1 buf (tmp)
    } else {
        numFp32Bufs = 2;  // all erf paths and non-fp32 tanh
    }
    uint32_t bytesPerElem = 4 * elemSize + numFp32Bufs * 4;
    uint32_t maxTileFromUB = (uint32_t)(ubSize / bytesPerElem);
    // Align down to alignNum
    maxTileFromUB = (maxTileFromUB / alignNum) * alignNum;
    if (maxTileFromUB == 0) maxTileFromUB = alignNum;

    // --- tile length (32B aligned, capped by UB and per-block size) ---
    uint32_t tileLen = maxTileFromUB;
    // Cap tile at per-block size (rounded up, aligned)
    uint32_t perBlockLen = ((uint32_t)(totalLength + blockDim - 1) / blockDim + alignNum - 1) / alignNum * alignNum;
    if (tileLen > perBlockLen) tileLen = perBlockLen;
    if (tileLen == 0) tileLen = alignNum;

    tiling.set_totalLength((uint32_t)totalLength);
    tiling.set_tileLength(tileLen);
    tiling.set_approximate((int32_t)approximate);
    tiling.set_dtype(dtypeId);

    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge


namespace ops {
class GeluV2 : public OpDef {
public:
    explicit GeluV2(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("approximate").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(GeluV2);
}  // namespace ops
