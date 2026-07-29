#include "tensor_equal_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

constexpr int64_t BLOCK_SIZE(64);
constexpr int64_t PRESERVED_UB(1024);

#define CEIL(x, y)  (((x) + (y) - 1) / (y))
#define FLOOR(x, y) ((x) / (y))
#define RES(x, y)   (((x) - 1) % (y) + 1)

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  TensorEqualTilingData tiling;
  const gert::Shape input_shape = context->GetInputShape(0)->GetStorageShape();
    const gert::Shape other_shape = context->GetInputShape(1)->GetStorageShape();
    const gert::Shape out_shape  = context->GetOutputShape(0)->GetStorageShape();

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    const ge::DataType data_type      = context->GetInputTensor(0)->GetDataType();
    int64_t            data_type_size = ge::GetSizeByDataType(data_type);

    int input_num = input_shape.GetDimNum();
    int other_num = other_shape.GetDimNum();

    uint8_t pre_cast = 0xff;
    int64_t shape[2][MAX_SHAPE], throughput[2][2] = {{1, 1}, {1, 1}};
    int64_t len = 0;

    for (int i = 0; i < MAX_SHAPE; i++) shape[0][i] = shape[1][i] = 1;
    for (int i = 0; i < MAX_SHAPE; i++) {
        int64_t input_dim = 1, other_dim = 1;
        uint8_t cast = 0;
        if (i < input_num) input_dim = input_shape.GetDim(input_num - i - 1);
        if (i < other_num) other_dim = other_shape.GetDim(other_num - i - 1);
        int64_t out_dim = std::max(input_dim, other_dim);

        if (input_dim == out_dim) cast |= 1;
        if (other_dim == out_dim) cast |= 2;

        if (pre_cast != 0xff && cast != pre_cast) len++;
        shape[0][len] *= input_dim;
        shape[1][len] *= other_dim;
        pre_cast = cast;
    }
    for (int i = 0; i < MAX_SHAPE; i++) {
        for (int j = 0; j < 2; j++) {
            throughput[j][j] *= shape[j][i];
            if (!i) throughput[j][j ^ 1] *= shape[j][i];
            else throughput[j][j ^ 1] *= std::max(shape[j][i], shape[j ^ 1][i]);
        }
    }
    std::reverse(shape[0], shape[0] + MAX_SHAPE);
    std::reverse(shape[1], shape[1] + MAX_SHAPE);

    int iter_idx = 0;
    if (throughput[0][0] + throughput[1][0] > throughput[0][1] + throughput[1][1]) iter_idx = 1;

    int64_t vector_length = std::max(shape[0][DIM_LAST], shape[1][DIM_LAST]);
    int64_t vector_count  = 1;
    for (int i = 0; i < DIM_LAST; i++) vector_count *= std::max(shape[0][i], shape[1][i]);

    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    ub_size -= PRESERVED_UB;

    int64_t buffer_num       = 2;
    int64_t ub_per_length    = 16 * 2 * buffer_num;
    int64_t length_per_block = FLOOR(BLOCK_SIZE, data_type_size);
    int64_t ub_per_block     = length_per_block * ub_per_length;

    int64_t block_per_iter    = FLOOR(ub_size, ub_per_block);
    int64_t block_per_vector  = CEIL(vector_length, length_per_block);
    int64_t length_per_iter   = length_per_block * block_per_iter;
    int64_t length_per_vector = length_per_block * block_per_vector;

    int64_t iter_per_vector = CEIL(block_per_vector, block_per_iter);
    int64_t iterations      = vector_count * iter_per_vector;
    int64_t real_residue    = RES(vector_length, length_per_iter);
    int64_t pad_residue     = RES(length_per_vector, length_per_iter);

    tiling.set_iterations(iterations);
    tiling.set_tile_length(length_per_iter);
    tiling.set_iter_per_vector(iter_per_vector);
    tiling.set_real_residue(real_residue);
    tiling.set_pad_residue(pad_residue);
    tiling.set_input_shape(shape[0]);
    tiling.set_other_shape(shape[1]);
    tiling.set_iter_idx(iter_idx);

    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


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
}


namespace ops {
class TensorEqual : public OpDef {
public:
    explicit TensorEqual(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(TensorEqual);
}
