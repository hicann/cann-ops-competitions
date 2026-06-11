#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cstdint>

#include "../op_kernel/addcmul_tiling.h"
#include "../op_kernel/tiling_key_addcmul.h"

namespace optiling {
namespace {

constexpr uint32_t kBlockBytes = 32;

static uint32_t DTypeBytes(ge::DataType dtype)
{
    if (dtype == ge::DT_INT8) {
        return 1;
    }
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        return 2;
    }
    return 4;
}

static uint32_t CalcTileElems(uint64_t ubBytes, ge::DataType dtype, uint32_t alignElems)
{
    int32_t bufferParts = (dtype == ge::DT_INT8) ? 12 : 10;
    uint32_t blocks = static_cast<uint32_t>((ubBytes / kBlockBytes / 2) / bufferParts);
    blocks = blocks <= 8 ? blocks : (blocks / 8 * 8);
    return blocks * alignElems;
}

static void ReadLengths(gert::TilingContext *context, uint32_t &inputLen, uint32_t &x1Len, uint32_t &x2Len,
                        uint32_t &totalLen, uint32_t &minLen)
{
    inputLen = static_cast<uint32_t>(context->GetInputShape(0)->GetStorageShape().GetShapeSize());
    x1Len = static_cast<uint32_t>(context->GetInputShape(1)->GetStorageShape().GetShapeSize());
    x2Len = static_cast<uint32_t>(context->GetInputShape(2)->GetStorageShape().GetShapeSize());
    totalLen = std::max(inputLen, std::max(x1Len, x2Len));
    minLen = std::min(inputLen, std::min(x1Len, x2Len));
}

static uint32_t FitBroadcastTile(uint32_t tileElems, uint32_t minLen, uint32_t alignElems, bool hasBroadcast)
{
    if (!hasBroadcast) {
        return tileElems;
    }
    uint32_t fitted = std::min(tileElems, minLen);
    while ((minLen % fitted) != 0 || (minLen % alignElems) != 0) {
        --fitted;
    }
    return fitted;
}

}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);

    ge::DataType dtype = context->GetInputDesc(0)->GetDataType();
    uint32_t dtypeKey = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, dtypeKey);

    uint32_t inputLen = 0;
    uint32_t x1Len = 0;
    uint32_t x2Len = 0;
    uint32_t totalLen = 0;
    uint32_t minLen = 0;
    ReadLengths(context, inputLen, x1Len, x2Len, totalLen, minLen);

    uint32_t alignElems = kBlockBytes / DTypeBytes(dtype);
    uint32_t tileElems = CalcTileElems(ubBytes, dtype, alignElems);
    tileElems = FitBroadcastTile(tileElems, minLen, alignElems, totalLen != minLen);

    uint32_t aivNum = platform.GetCoreNum();
    uint32_t fullTiles = tileElems == 0 ? 0 : (totalLen / tileElems);
    aivNum = (aivNum < fullTiles) ? aivNum : fullTiles;
    aivNum = aivNum == 0 ? 1 : aivNum;

    uint32_t coreSize = (totalLen / aivNum) / (alignElems * 8) * (alignElems * 8);
    uint32_t coreRemain = totalLen - aivNum * coreSize;

    AddcmulTilingData *tiling = context->GetTilingData<AddcmulTilingData>();
    tiling->tileNum = 0;
    tiling->ALIGN_NUM = alignElems;
    tiling->block_size = tileElems;
    tiling->aivNum = aivNum;
    tiling->core_size = coreSize;
    tiling->core_remain = coreRemain;
    tiling->total_length = totalLen;
    tiling->input_data_length = inputLen;
    tiling->x1_length = x1Len;
    tiling->x2_length = x2Len;

    context->SetBlockDim(aivNum);
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class Addcmul : public OpDef {
public:
    explicit Addcmul(const char *name) : OpDef(name)
    {
        this->Input("input_data").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32}).Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x1").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32}).Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32}).Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("value").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32}).Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32}).Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};

OP_ADD(Addcmul);

}  // namespace ops
