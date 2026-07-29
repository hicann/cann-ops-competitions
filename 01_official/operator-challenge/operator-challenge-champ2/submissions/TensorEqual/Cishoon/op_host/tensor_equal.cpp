
#include "tensor_equal_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

namespace optiling {

constexpr uint32_t TE_LINEAR_ALIGN_ELEMS = 512;
constexpr size_t TE_LINEAR_TILING_BYTES = sizeof(uint32_t) * 2;
constexpr uint32_t TE_BROADCAST_TILE_LEN = 4096;

static uint32_t GetComputeTypeSize(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_INT8:
        case ge::DT_UINT8:
            return 2;
        case ge::DT_INT16:
        case ge::DT_INT32:
        case ge::DT_BF16:
            return 4;
        default:
            return GetSizeByDataType(dataType);
    }
}

static bool NeedCastBuffer(ge::DataType dataType)
{
    return dataType == ge::DT_INT8 || dataType == ge::DT_UINT8 ||
           dataType == ge::DT_INT16 || dataType == ge::DT_BF16;
}

static uint32_t CalcLinearTileLen(ge::DataType dataType, uint64_t ubSize)
{
    uint32_t dataTypeSize = GetSizeByDataType(dataType);
    uint32_t computeTypeSize = GetComputeTypeSize(dataType);

    uint32_t bytesPerElement = 4 * dataTypeSize + 9;
    if (dataType == ge::DT_INT32) {
        bytesPerElement += sizeof(int32_t);
    } else if (NeedCastBuffer(dataType)) {
        bytesPerElement += 2 * computeTypeSize;
    }

    uint32_t tileLen = static_cast<uint32_t>(ubSize / bytesPerElement);
    tileLen = (tileLen / TE_LINEAR_ALIGN_ELEMS) * TE_LINEAR_ALIGN_ELEMS;
    return std::max(tileLen, TE_LINEAR_ALIGN_ELEMS);
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TensorEqualTilingData tiling;

    const gert::Shape x1 = context->GetInputShape(0)->GetStorageShape();
    const gert::Shape x2 = context->GetInputShape(1)->GetStorageShape();

    int32_t n1 = static_cast<int32_t>(x1.GetDimNum());
    int32_t n2 = static_cast<int32_t>(x2.GetDimNum());
    int32_t maxN = std::max(n1, n2);
    if (maxN > TE_MAX_SHAPE) {
        return ge::GRAPH_FAILED;
    }

    bool same_shape = true;
    int32_t compact_x1[TE_MAX_SHAPE];
    int32_t compact_x2[TE_MAX_SHAPE];
    int32_t compact_out[TE_MAX_SHAPE];
    bool compact_x1_stride[TE_MAX_SHAPE];
    bool compact_x2_stride[TE_MAX_SHAPE];
    int32_t compact_num = 0;

    for (int32_t i = 0; i < maxN; ++i) {
        int32_t x1Index = i - (maxN - n1);
        int32_t x2Index = i - (maxN - n2);
        int32_t d1 = (x1Index >= 0) ? static_cast<int32_t>(x1.GetDim(x1Index)) : 1;
        int32_t d2 = (x2Index >= 0) ? static_cast<int32_t>(x2.GetDim(x2Index)) : 1;
        if (d1 != d2 && d1 != 1 && d2 != 1) {
            return ge::GRAPH_FAILED;
        }

        int32_t od = std::max(d1, d2);
        if (d1 != d2) {
            same_shape = false;
        }

        bool x1_stride = (d1 == od);
        bool x2_stride = (d2 == od);
        if (compact_num > 0 &&
            compact_x1_stride[compact_num - 1] == x1_stride &&
            compact_x2_stride[compact_num - 1] == x2_stride) {
            compact_x1[compact_num - 1] *= d1;
            compact_x2[compact_num - 1] *= d2;
            compact_out[compact_num - 1] *= od;
        } else {
            compact_x1[compact_num] = d1;
            compact_x2[compact_num] = d2;
            compact_out[compact_num] = od;
            compact_x1_stride[compact_num] = x1_stride;
            compact_x2_stride[compact_num] = x2_stride;
            ++compact_num;
        }
    }
    if (compact_num == 0) {
        compact_x1[0] = 1;
        compact_x2[0] = 1;
        compact_out[0] = 1;
        compact_num = 1;
    }

    int32_t s_x1[TE_MAX_SHAPE];
    int32_t s_x2[TE_MAX_SHAPE];
    int32_t s_out[TE_MAX_SHAPE];
    for (int32_t i = 0; i < TE_MAX_SHAPE; ++i) {
        s_x1[i] = 1;
        s_x2[i] = 1;
        s_out[i] = 1;
    }
    int32_t base = TE_MAX_SHAPE - compact_num;
    for (int32_t i = 0; i < compact_num; ++i) {
        int32_t slot = base + i;
        s_x1[slot] = compact_x1[i];
        s_x2[slot] = compact_x2[i];
        s_out[slot] = compact_out[i];
    }

    uint32_t total_len = 1;
    uint32_t x1_len = 1;
    uint32_t x2_len = 1;
    for (int32_t i = 0; i < TE_MAX_SHAPE; ++i) {
        total_len *= static_cast<uint32_t>(s_out[i]);
        x1_len *= static_cast<uint32_t>(s_x1[i]);
        x2_len *= static_cast<uint32_t>(s_x2[i]);
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t core_num = ascendcPlatform.GetCoreNum();
    if (core_num < 1) {
        core_num = 1;
    }

    uint32_t tiling_key = 3U;
    if (same_shape) {
        tiling_key = 0U;
    } else if (x1_len == 1) {
        tiling_key = 1U;
    } else if (x2_len == 1) {
        tiling_key = 2U;
    }

    uint32_t tile_len = TE_BROADCAST_TILE_LEN;
    if (tiling_key <= 2U) {
        uint64_t ub_size = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        tile_len = CalcLinearTileLen(context->GetInputDesc(0)->GetDataType(), ub_size);
    }

    tiling.set_total_len(total_len);
    tiling.set_tile_len(tile_len);

    uint32_t work_units = 1;
    if (tiling_key <= 2U) {
        work_units = (total_len + TE_LINEAR_ALIGN_ELEMS - 1) / TE_LINEAR_ALIGN_ELEMS;
    } else {
        int32_t inner = s_out[TE_MAX_SHAPE - 1];
        work_units = (inner > 0) ? (total_len / static_cast<uint32_t>(inner)) : 1;
    }
    if (work_units == 0) {
        work_units = 1;
    }
    int32_t used_core_num = static_cast<int32_t>(std::min<uint32_t>(
        static_cast<uint32_t>(core_num), work_units));
    if (used_core_num < 1) {
        used_core_num = 1;
    }
    context->SetBlockDim(used_core_num);
    context->SetTilingKey(tiling_key);

    if (tiling_key <= 2U) {
        auto raw = reinterpret_cast<uint32_t*>(context->GetRawTilingData()->GetData());
        raw[0] = total_len;
        raw[1] = tile_len;
        context->GetRawTilingData()->SetDataSize(TE_LINEAR_TILING_BYTES);
    } else {
        tiling.set_x1_shape(s_x1);
        tiling.set_x2_shape(s_x2);
        tiling.set_out_shape(s_out);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    }

    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling


namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1 = context->GetInputShape(0);
    const gert::Shape* x2 = context->GetInputShape(1);
    gert::Shape*       y  = context->GetOutputShape(0);

    size_t n1 = x1->GetDimNum();
    size_t n2 = x2->GetDimNum();
    size_t n  = std::max(n1, n2);
    if (n > static_cast<size_t>(TE_MAX_SHAPE)) {
        return GRAPH_FAILED;
    }

    y->SetDimNum(n);
    for (size_t i = 0; i < n; ++i) {
        size_t x1Base = n - n1;
        size_t x2Base = n - n2;
        int32_t d1 = (i >= x1Base) ? static_cast<int32_t>(x1->GetDim(i - x1Base)) : 1;
        int32_t d2 = (i >= x2Base) ? static_cast<int32_t>(x2->GetDim(i - x2Base)) : 1;
        if (d1 != d2 && d1 != 1 && d2 != 1) {
            return GRAPH_FAILED;
        }
        y->SetDim(i, std::max(d1, d2));
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_BOOL);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge


namespace ops {

class TensorEqual : public OpDef {
public:
    explicit TensorEqual(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT,
                       ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT,
                       ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL,
                       ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(TensorEqual);

} // namespace ops
