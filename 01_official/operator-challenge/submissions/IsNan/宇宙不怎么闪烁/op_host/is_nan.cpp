#include "is_nan_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

constexpr uint64_t ALIGN_BYTES  = 1024;
constexpr uint64_t RESERVE_UB   = 8 * 1024;
constexpr int32_t  BUF_NUM      = 1;
constexpr int32_t  MIN_CORE_NUM = 2;

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    IsNanTilingData           tiling;
    const gert::StorageShape *x_shape = context->GetInputShape(0);

    int64_t totalLength = 1;
    for (int i = 0; i < x_shape->GetStorageShape().GetDimNum(); i++) {
        totalLength *= x_shape->GetStorageShape().GetDim(i);
    }

    const ge::DataType data_type      = context->GetInputTensor(0)->GetDataType();
    int64_t            data_type_size = ge::GetSizeByDataType(data_type);

    int64_t elem_per_block = ALIGN_BYTES / data_type_size;
    int64_t total_block    = (totalLength + elem_per_block - 1) / elem_per_block;

    auto     ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    ub_size -= RESERVE_UB;

    // x[2] + y[2] + tmp[1] + const[2]
    int64_t bytes_per_elem = 0;
    if (data_type == ge::DT_FLOAT) {
        bytes_per_elem = 4 * BUF_NUM + 1 * BUF_NUM + 2 + 1 + 2;
    } else if (data_type == ge::DT_FLOAT16) {
        bytes_per_elem = 2 * BUF_NUM + 1 * BUF_NUM + 2 + 1 + 2;
    } else if (data_type == ge::DT_BF16) {
        bytes_per_elem = 2 * BUF_NUM + 1 * BUF_NUM + 2 + 1 + 4;
    }

    int64_t ub_per_block = elem_per_block * bytes_per_elem;
    int64_t tile_block   = ub_size / ub_per_block;
    int64_t tile_length  = tile_block * elem_per_block;

    int64_t elem_per_align = ALIGN_BYTES / data_type_size;
    tile_length            = (tile_length / elem_per_align) * elem_per_align;
    if (tile_length < elem_per_align) tile_length = elem_per_align;

    // int64_t coreNum = ascendcPlatform.GetCoreNum();
    int64_t coreNum = 40;
    if (totalLength > 10000000) 
        coreNum = 20;
    
    if (coreNum < MIN_CORE_NUM) coreNum = MIN_CORE_NUM;
    if (coreNum > total_block) coreNum = total_block;

    int64_t perCoreBlock  = total_block / coreNum;
    int64_t lastCoreBlock = total_block - perCoreBlock * coreNum;
    int64_t alignedLength = total_block * elem_per_block;

    if (totalLength > 10000000) 
        coreNum = 40;

    tiling.set_totalLength(alignedLength);
    tiling.set_tileLength(static_cast<int32_t>(tile_length));
    tiling.set_perCoreBlock(static_cast<int32_t>(perCoreBlock));
    tiling.set_lastCoreBlock(static_cast<int32_t>(lastCoreBlock));
    context->SetBlockDim(coreNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape       *y_shape  = context->GetOutputShape(0);
    *y_shape                    = *x1_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class IsNan : public OpDef {
  public:
    explicit IsNan(const char *name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(IsNan);

} // namespace ops
