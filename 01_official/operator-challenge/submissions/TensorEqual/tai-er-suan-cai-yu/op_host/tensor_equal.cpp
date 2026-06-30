
#include "tensor_equal_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t MAX_RANK = 4;
constexpr uint32_t BLOCK_DIM = 8;
constexpr uint64_t KEY_SCALAR = 1;
constexpr uint64_t KEY_BYTE_CONTIG = 2;
constexpr uint64_t KEY_BYTE_ROW = 3;
constexpr uint64_t KEY_INT16_CONTIG = 4;
constexpr uint64_t KEY_INT16_ROW = 5;
constexpr uint64_t KEY_INT32_ROW = 6;
constexpr uint64_t KEY_FP_CONTIG = 7;
constexpr uint64_t KEY_FP_ROW = 8;
constexpr uint64_t KEY_INT32_CONTIG = 9;

uint32_t DTypeBytes(ge::DataType dt)
{
    if (dt == ge::DT_FLOAT || dt == ge::DT_INT32) {
        return 4;
    }
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16 || dt == ge::DT_INT16) {
        return 2;
    }
    return 1;
}

bool BuildBroadcast(const gert::Shape &a, const gert::Shape &b, uint32_t outShape[MAX_RANK],
                    uint32_t aStride[MAX_RANK], uint32_t bStride[MAX_RANK], uint32_t &rank,
                    uint32_t &outSize)
{
    const uint32_t aRank = static_cast<uint32_t>(a.GetDimNum());
    const uint32_t bRank = static_cast<uint32_t>(b.GetDimNum());
    rank = aRank > bRank ? aRank : bRank;
    if (rank == 0 || rank > MAX_RANK) {
        return false;
    }

    uint32_t aDims[MAX_RANK] = {1, 1, 1, 1};
    uint32_t bDims[MAX_RANK] = {1, 1, 1, 1};
    for (uint32_t i = 0; i < aRank; ++i) {
        const int64_t dim = a.GetDim(i);
        if (dim <= 0) {
            return false;
        }
        aDims[rank - aRank + i] = static_cast<uint32_t>(dim);
    }
    for (uint32_t i = 0; i < bRank; ++i) {
        const int64_t dim = b.GetDim(i);
        if (dim <= 0) {
            return false;
        }
        bDims[rank - bRank + i] = static_cast<uint32_t>(dim);
    }

    outSize = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        if (aDims[i] != bDims[i] && aDims[i] != 1 && bDims[i] != 1) {
            return false;
        }
        outShape[i] = aDims[i] > bDims[i] ? aDims[i] : bDims[i];
        outSize *= outShape[i];
    }
    for (uint32_t i = rank; i < MAX_RANK; ++i) {
        outShape[i] = 1;
    }

    uint32_t aBaseStride[MAX_RANK] = {0, 0, 0, 0};
    uint32_t bBaseStride[MAX_RANK] = {0, 0, 0, 0};
    uint32_t stride = 1;
    for (int32_t i = static_cast<int32_t>(rank) - 1; i >= 0; --i) {
        aBaseStride[i] = stride;
        stride *= aDims[i];
    }
    stride = 1;
    for (int32_t i = static_cast<int32_t>(rank) - 1; i >= 0; --i) {
        bBaseStride[i] = stride;
        stride *= bDims[i];
    }
    for (uint32_t i = 0; i < rank; ++i) {
        aStride[i] = (aDims[i] == 1 && outShape[i] != 1) ? 0 : aBaseStride[i];
        bStride[i] = (bDims[i] == 1 && outShape[i] != 1) ? 0 : bBaseStride[i];
    }
    for (uint32_t i = rank; i < MAX_RANK; ++i) {
        aStride[i] = 0;
        bStride[i] = 0;
    }
    return true;
}

bool CanMergeStride(uint32_t outerStride, uint32_t innerStride, uint32_t innerShape)
{
    return (outerStride == 0 && innerStride == 0) || (outerStride == innerShape * innerStride);
}

void RemoveDim(uint32_t dim, uint32_t outShape[MAX_RANK], uint32_t aStride[MAX_RANK],
               uint32_t bStride[MAX_RANK], uint32_t &rank)
{
    for (uint32_t i = dim; i + 1 < rank; ++i) {
        outShape[i] = outShape[i + 1];
        aStride[i] = aStride[i + 1];
        bStride[i] = bStride[i + 1];
    }
    --rank;
    outShape[rank] = 1;
    aStride[rank] = 0;
    bStride[rank] = 0;
}

void CompactBroadcast(uint32_t outShape[MAX_RANK], uint32_t aStride[MAX_RANK],
                      uint32_t bStride[MAX_RANK], uint32_t &rank)
{
    if (rank == 0) {
        return;
    }

    uint32_t i = 0;
    while (i + 1 < rank) {
        if (outShape[i] == 1) {
            RemoveDim(i, outShape, aStride, bStride, rank);
            continue;
        }
        if (outShape[i + 1] == 1) {
            RemoveDim(i + 1, outShape, aStride, bStride, rank);
            continue;
        }

        const uint32_t innerShape = outShape[i + 1];
        const bool mergeA = CanMergeStride(aStride[i], aStride[i + 1], innerShape);
        const bool mergeB = CanMergeStride(bStride[i], bStride[i + 1], innerShape);
        if (mergeA && mergeB) {
            outShape[i] *= innerShape;
            aStride[i] = aStride[i + 1];
            bStride[i] = bStride[i + 1];
            RemoveDim(i + 1, outShape, aStride, bStride, rank);
            continue;
        }
        ++i;
    }
}

bool IsContiguousSameStride(const uint32_t aStride[MAX_RANK], const uint32_t bStride[MAX_RANK])
{
    return aStride[0] == bStride[0] && aStride[1] == bStride[1] &&
           aStride[2] == bStride[2] && aStride[3] == bStride[3];
}

bool IsBroadcastRowVector(uint32_t rank, const uint32_t outShape[MAX_RANK],
                          const uint32_t aStride[MAX_RANK], const uint32_t bStride[MAX_RANK])
{
    uint32_t rowLen = outShape[3];
    uint32_t outRows = outShape[0] * outShape[1] * outShape[2];
    uint32_t aInnerStride = aStride[3];
    uint32_t bInnerStride = bStride[3];
    if (rank == 1) {
        rowLen = outShape[0];
        outRows = 1;
        aInnerStride = aStride[0];
        bInnerStride = bStride[0];
    } else if (rank == 2) {
        rowLen = outShape[1];
        outRows = outShape[0];
        aInnerStride = aStride[1];
        bInnerStride = bStride[1];
    } else if (rank == 3) {
        rowLen = outShape[2];
        outRows = outShape[0] * outShape[1];
        aInnerStride = aStride[2];
        bInnerStride = bStride[2];
    }
    const bool hasBroadcast = !IsContiguousSameStride(aStride, bStride);
    return rowLen >= 1 && (aInnerStride == 0 || aInnerStride == 1) &&
           (bInnerStride == 0 || bInnerStride == 1) && hasBroadcast && outRows > 0;
}

uint64_t SelectTilingKey(ge::DataType dtype, uint32_t rank, const uint32_t outShape[MAX_RANK],
                         const uint32_t aStride[MAX_RANK], const uint32_t bStride[MAX_RANK])
{
    const bool contiguous = IsContiguousSameStride(aStride, bStride);
    const bool rowVector = IsBroadcastRowVector(rank, outShape, aStride, bStride);
    if (dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16) {
        if (contiguous) {
            return KEY_FP_CONTIG;
        }
        return rowVector ? KEY_FP_ROW : KEY_SCALAR;
    }
    if (dtype == ge::DT_UINT8 || dtype == ge::DT_INT8 || dtype == ge::DT_BOOL) {
        if (contiguous) {
            return KEY_BYTE_CONTIG;
        }
        return rowVector ? KEY_BYTE_ROW : KEY_SCALAR;
    }
    if (dtype == ge::DT_INT16 || dtype == ge::DT_BF16) {
        if (contiguous) {
            return KEY_INT16_CONTIG;
        }
        return rowVector ? KEY_INT16_ROW : KEY_SCALAR;
    }
    if (dtype == ge::DT_INT32) {
        if (contiguous) {
            return KEY_INT32_CONTIG;
        }
        return rowVector ? KEY_INT32_ROW : KEY_SCALAR;
    }
    return KEY_SCALAR;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* x1Shape = context->GetInputShape(0);
    const gert::StorageShape* x2Shape = context->GetInputShape(1);
    if (x1Shape == nullptr || x2Shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t outShape[MAX_RANK] = {1, 1, 1, 1};
    uint32_t x1Stride[MAX_RANK] = {0, 0, 0, 0};
    uint32_t x2Stride[MAX_RANK] = {0, 0, 0, 0};
    uint32_t rank = 0;
    uint32_t outSize = 0;
    if (!BuildBroadcast(x1Shape->GetStorageShape(), x2Shape->GetStorageShape(), outShape, x1Stride, x2Stride,
                        rank, outSize)) {
        return ge::GRAPH_FAILED;
    }
    CompactBroadcast(outShape, x1Stride, x2Stride, rank);
    const ge::DataType dtype = context->GetInputDesc(0)->GetDataType();
    if (context->SetTilingKey(SelectTilingKey(dtype, rank, outShape, x1Stride, x2Stride)) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    TensorEqualTilingData tiling;
    tiling.set_outSize(outSize);
    tiling.set_elemBytes(DTypeBytes(dtype));
    tiling.set_rank(rank);
    tiling.set_outShape0(outShape[0]);
    tiling.set_outShape1(outShape[1]);
    tiling.set_outShape2(outShape[2]);
    tiling.set_outShape3(outShape[3]);
    tiling.set_x1Stride0(x1Stride[0]);
    tiling.set_x1Stride1(x1Stride[1]);
    tiling.set_x1Stride2(x1Stride[2]);
    tiling.set_x1Stride3(x1Stride[3]);
    tiling.set_x2Stride0(x2Stride[0]);
    tiling.set_x2Stride1(x2Stride[1]);
    tiling.set_x2Stride2(x2Stride[2]);
    tiling.set_x2Stride3(x2Stride[3]);
    context->SetBlockDim(BLOCK_DIM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* x2_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (x1_shape == nullptr || x2_shape == nullptr || y_shape == nullptr) {
        return GRAPH_FAILED;
    }
    uint32_t outShape[MAX_RANK] = {1, 1, 1, 1};
    uint32_t x1Stride[MAX_RANK] = {0, 0, 0, 0};
    uint32_t x2Stride[MAX_RANK] = {0, 0, 0, 0};
    uint32_t rank = 0;
    uint32_t outSize = 0;
    if (!BuildBroadcast(*x1_shape, *x2_shape, outShape, x1Stride, x2Stride, rank, outSize)) {
        return GRAPH_FAILED;
    }
    y_shape->SetDimNum(rank);
    for (uint32_t i = 0; i < rank; ++i) {
        y_shape->SetDim(i, outShape[i]);
    }
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
context->SetOutputDataType(0, ge::DT_BOOL);
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
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(TensorEqual);
}
