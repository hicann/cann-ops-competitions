
#include "assign_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cstring>

constexpr int32_t BLOCK_SIZE(64);
constexpr int32_t PRESERVED_UB(0);

#define FLOOR(x, y) ((x) / (y))

namespace optiling {

static void SaveOneShapeTiling(gert::TilingContext *context, int32_t tile_length,
                               int32_t out_shape) {
    AssignOneShapeTilingData tiling;
    tiling.tile_length = tile_length;
    tiling.out_shape   = out_shape;

    std::memcpy(context->GetRawTilingData()->GetData(), &tiling, sizeof(tiling));
    context->GetRawTilingData()->SetDataSize(sizeof(tiling));
}

template <int COMPACT_SHAPE_NUM>
static void SaveCompactTiling(gert::TilingContext *context, int32_t tile_length,
                              const int32_t *shape_other, const int32_t *shape_out) {
    AssignCompactTilingData<COMPACT_SHAPE_NUM> tiling;
    tiling.tile_length = tile_length;

    constexpr int32_t start = MAX_SHAPE - COMPACT_SHAPE_NUM;
    for (int32_t i = 0; i < COMPACT_SHAPE_NUM; ++i) {
        tiling.other_shape[i] = shape_other[start + i];
        tiling.out_shape[i]   = shape_out[start + i];
    }

    std::memcpy(context->GetRawTilingData()->GetData(), &tiling, sizeof(tiling));
    context->GetRawTilingData()->SetDataSize(sizeof(tiling));
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {

    const gert::Shape input_shape = context->GetInputShape(0)->GetStorageShape();
    const gert::Shape other_shape = context->GetInputShape(1)->GetStorageShape();

    auto    ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t core_num        = ascendcPlatform.GetCoreNum();
    if (core_num < 3) core_num = 3;

    const ge::DataType data_type     = context->GetInputTensor(0)->GetDataType();
    int32_t            data_type_size = ge::GetSizeByDataType(data_type);

    int input_num = input_shape.GetDimNum();
    int other_num = other_shape.GetDimNum();

    // output shape = input shape (InferShape uses input)
    // Merge dimensions by broadcast pattern: other vs output(=input)
    uint8_t pre_cast = 0xff;
    int32_t shape_other[MAX_SHAPE], shape_out[MAX_SHAPE];
    int32_t len = 0, cnt_other = 1, cnt_out = 1;
    bool has_broadcast = false;

    for (int i = 0; i < MAX_SHAPE; i++) shape_other[i] = shape_out[i] = 1;

    int max_num = std::max(input_num, other_num);
    for (int i = 0; i < max_num; i++) {
        int32_t out_dim   = (i < input_num) ? input_shape.GetDim(input_num - i - 1) : 1;
        int32_t other_dim = (i < other_num) ? other_shape.GetDim(other_num - i - 1) : 1;

        // cast: 1 = other matches out, 0 = other needs broadcast
        uint8_t cast = (other_dim == out_dim) ? 1 : 0;
        has_broadcast = has_broadcast || (cast == 0);

        if (pre_cast != 0xff && cast != pre_cast) {
            shape_other[len] = cnt_other;
            shape_out[len]   = cnt_out;
            cnt_other = cnt_out = 1;
            len++;
        }
        cnt_other *= other_dim;
        cnt_out *= out_dim;
        pre_cast = cast;
    }
    shape_other[len] = cnt_other;
    shape_out[len]   = cnt_out;
    len++;
    std::reverse(shape_other, shape_other + MAX_SHAPE);
    std::reverse(shape_out, shape_out + MAX_SHAPE);

    int32_t vector_length = shape_out[DIM_LAST];

    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    ub_size -= PRESERVED_UB;

    int32_t buffer_num       = (vector_length > 8192 ? 1 : 2);
    int32_t ub_per_length    = std::max(data_type_size, int32_t(2)) * buffer_num;
    int32_t length_per_block = FLOOR(BLOCK_SIZE, data_type_size);
    int32_t ub_per_block     = length_per_block * ub_per_length;

    int32_t block_per_iter    = FLOOR(ub_size, ub_per_block);
    int32_t length_per_iter   = length_per_block * block_per_iter;

    context->SetBlockDim(core_num);

    int32_t compact_shape_num = std::max(1, std::min(len, MAX_SHAPE));
    // key 0/1 use 8B tiling. Mixed broadcast keeps only the compressed shape suffix.
    if (!has_broadcast) {
        context->SetTilingKey(0);
        SaveOneShapeTiling(context, length_per_iter, shape_out[DIM_LAST]);
        return ge::GRAPH_SUCCESS;
    }
    if (compact_shape_num == 1 && shape_other[DIM_LAST] == 1) {
        context->SetTilingKey(1);
        SaveOneShapeTiling(context, length_per_iter, shape_out[DIM_LAST]);
        return ge::GRAPH_SUCCESS;
    }

    context->SetTilingKey(static_cast<uint64_t>(compact_shape_num + 1));
    switch (compact_shape_num) {
        case 1:
            SaveCompactTiling<1>(context, length_per_iter, shape_other, shape_out);
            break;
        case 2:
            SaveCompactTiling<2>(context, length_per_iter, shape_other, shape_out);
            break;
        case 3:
            SaveCompactTiling<3>(context, length_per_iter, shape_other, shape_out);
            break;
        default:
            SaveCompactTiling<4>(context, length_per_iter, shape_other, shape_out);
            break;
    }

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape       *y_shape  = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Assign : public OpDef {
  public:
    explicit Assign(const char *name) : OpDef(name) {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16,
                       ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("other")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16,
                       ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("use_locking").Bool();
        this->Output("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16,
                       ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Assign);
} // namespace ops
