#include "tensor_equal_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"

namespace {

constexpr size_t kMaxRank = 8;

constexpr ge::DataType kSupportedTypes[] = {
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32,
    ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL};

inline int64_t GetDataTypeSize(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
            return 2;
        case ge::DT_UINT8:
        case ge::DT_INT8:
        case ge::DT_BOOL:
            return 1;
        default:
            return 0;
    }
}

inline int64_t AlignDown(int64_t value, int64_t align)
{
    return (value / align) * align;
}

inline int64_t AlignUp(int64_t value, int64_t align)
{
    return ((value + align - 1) / align) * align;
}

inline bool NeedsFloatCompareBuffer(ge::DataType dtype)
{
    return dtype == ge::DT_INT16 || dtype == ge::DT_INT32;
}

inline bool NeedsHalfCompareBuffer(ge::DataType dtype)
{
    return dtype == ge::DT_UINT8 || dtype == ge::DT_INT8 || dtype == ge::DT_BOOL;
}

inline int64_t SelectTileElems(ge::DataType dtype, int64_t totalElems,
                               bool x1ScalarBuffer, bool x2ScalarBuffer)
{
    const int64_t elemSize = GetDataTypeSize(dtype);
    if (elemSize <= 0 || totalElems <= 0) {
        return 1;
    }

#ifndef TENSOR_EQUAL_UB_BUDGET_BYTES
#define TENSOR_EQUAL_UB_BUDGET_BYTES (192 * 1024)
#endif
    constexpr int64_t kUbBudgetBytes = TENSOR_EQUAL_UB_BUDGET_BYTES;
    constexpr int64_t kTileAlignElems = 128;
    constexpr int64_t kScalarBlockElems = 32;
    constexpr int64_t kBufferNum = 2;
    constexpr int64_t kAlignBytes = 32;
    const bool needsFloatCompare = NeedsFloatCompareBuffer(dtype);
    const bool needsHalfCompare = NeedsHalfCompareBuffer(dtype);
    const int64_t compareElemSize = needsFloatCompare ? static_cast<int64_t>(sizeof(float)) :
                                    (needsHalfCompare ? static_cast<int64_t>(sizeof(uint16_t)) : 0);

    const int64_t alignElems = std::max<int64_t>(kTileAlignElems, 32 / elemSize);
    auto scalarInputElems = [&]() {
        return (needsFloatCompare || needsHalfCompare) ? kTileAlignElems : kScalarBlockElems;
    };
    auto sideInputBytes = [&](int64_t elems, bool scalarBuffer) {
        const int64_t inputElems = scalarBuffer ? scalarInputElems() : elems;
        return AlignUp(inputElems * elemSize, kAlignBytes);
    };
    auto sideCalcBytes = [&](int64_t elems, bool scalarBuffer) {
        if (compareElemSize == 0) {
            return int64_t{0};
        }
        const int64_t compareElems = scalarBuffer ? kTileAlignElems : elems;
        return AlignUp(compareElems * compareElemSize, kAlignBytes);
    };
    auto ubBytesFor = [&](int64_t elems) {
        const int64_t x1InputBytes = sideInputBytes(elems, x1ScalarBuffer);
        const int64_t x2InputBytes = sideInputBytes(elems, x2ScalarBuffer);
        const int64_t outputBytes = AlignUp(elems, kAlignBytes);
        const int64_t maskBytes = AlignUp(AlignUp(elems, kTileAlignElems) / 8, kAlignBytes);
        const int64_t x1CalcBytes = sideCalcBytes(elems, x1ScalarBuffer);
        const int64_t x2CalcBytes = sideCalcBytes(elems, x2ScalarBuffer);
        return kBufferNum * x1InputBytes + kBufferNum * x2InputBytes +
               outputBytes + maskBytes + x1CalcBytes + x2CalcBytes;
    };

    int64_t upperByBudget = AlignDown(kUbBudgetBytes, alignElems);
    if (upperByBudget < alignElems) {
        upperByBudget = alignElems;
    }
    int64_t low = alignElems;
    int64_t high = std::min(AlignDown(totalElems, alignElems), upperByBudget);
    if (high < alignElems) {
        high = alignElems;
    }
    int64_t tileElems = high;
    while (low <= high) {
        const int64_t mid = AlignDown((low + high) / 2, alignElems);
        if (mid < alignElems) {
            break;
        }
        if (ubBytesFor(mid) <= kUbBudgetBytes) {
            tileElems = mid;
            low = mid + alignElems;
        } else {
            high = mid - alignElems;
        }
    }
    if (tileElems <= 0) {
        tileElems = alignElems;
    }
    return std::min(tileElems, totalElems);
}

inline bool AllDimsOne(size_t rank, const int64_t* dims)
{
    for (size_t i = 0; i < rank; ++i) {
        if (dims[i] != 1) {
            return false;
        }
    }
    return rank > 0;
}

inline bool UseX1ScalarBuffer(bool sameShape, size_t rank, const int64_t* x1Dims, const int64_t* x2Dims)
{
    if (sameShape || rank == 0) {
        return false;
    }
    const bool x1AllBroadcast = AllDimsOne(rank, x1Dims);
    const bool x2AllBroadcast = AllDimsOne(rank, x2Dims);
    if (x1AllBroadcast && !x2AllBroadcast) {
        return true;
    }
    if (x2AllBroadcast && !x1AllBroadcast) {
        return false;
    }
    const bool x1LastBroadcast = (x1Dims[rank - 1] == 1);
    const bool x2LastBroadcast = (x2Dims[rank - 1] == 1);
    return x1LastBroadcast && !x2LastBroadcast;
}

inline bool UseX2ScalarBuffer(bool sameShape, size_t rank, const int64_t* x1Dims, const int64_t* x2Dims)
{
    if (sameShape || rank == 0) {
        return false;
    }
    const bool x1AllBroadcast = AllDimsOne(rank, x1Dims);
    const bool x2AllBroadcast = AllDimsOne(rank, x2Dims);
    if (x2AllBroadcast && !x1AllBroadcast) {
        return true;
    }
    if (x1AllBroadcast && !x2AllBroadcast) {
        return false;
    }
    return x2Dims[rank - 1] == 1;
}

inline int64_t DimFromRight(const gert::Shape& shape, size_t alignedIndex, size_t rank)
{
    const size_t shapeRank = static_cast<size_t>(shape.GetDimNum());
    if (alignedIndex + shapeRank < rank) {
        return 1;
    }
    return shape.GetDim(static_cast<int64_t>(alignedIndex + shapeRank - rank));
}

inline bool IsSupportedType(ge::DataType dtype)
{
    for (ge::DataType supported : kSupportedTypes) {
        if (supported == dtype) {
            return true;
        }
    }
    return false;
}

bool BuildBroadcastMeta(const gert::Shape& x1Shape, const gert::Shape& x2Shape, int64_t* outDims,
                        int64_t* x1Dims, int64_t* x2Dims, int64_t* outStrides,
                        int64_t* x1Strides, int64_t* x2Strides, size_t& rank,
                        int64_t& totalElems, bool& sameShape)
{
    const size_t x1Rank = static_cast<size_t>(x1Shape.GetDimNum());
    const size_t x2Rank = static_cast<size_t>(x2Shape.GetDimNum());
    rank = std::max(x1Rank, x2Rank);
    if (rank == 0 || rank > kMaxRank) {
        return false;
    }

    totalElems = 1;
    sameShape = (x1Rank == x2Rank);
    for (size_t i = 0; i < kMaxRank; ++i) {
        outDims[i] = 1;
        x1Dims[i] = 1;
        x2Dims[i] = 1;
        outStrides[i] = 1;
        x1Strides[i] = 0;
        x2Strides[i] = 0;
    }

    for (size_t i = 0; i < rank; ++i) {
        const int64_t x1Dim = DimFromRight(x1Shape, i, rank);
        const int64_t x2Dim = DimFromRight(x2Shape, i, rank);
        if (x1Dim <= 0 || x2Dim <= 0) {
            return false;
        }
        if (x1Dim != x2Dim && x1Dim != 1 && x2Dim != 1) {
            return false;
        }
        const int64_t outDim = std::max(x1Dim, x2Dim);
        if (totalElems > std::numeric_limits<int64_t>::max() / outDim) {
            return false;
        }
        outDims[i] = outDim;
        x1Dims[i] = x1Dim;
        x2Dims[i] = x2Dim;
        totalElems *= outDim;
        if (x1Dim != x2Dim) {
            sameShape = false;
        }
    }

    int64_t running = 1;
    int64_t x1Running = 1;
    int64_t x2Running = 1;
    for (int64_t i = static_cast<int64_t>(rank) - 1; i >= 0; --i) {
        outStrides[i] = running;
        running *= outDims[i];
        x1Strides[i] = (x1Dims[i] == 1) ? 0 : x1Running;
        x2Strides[i] = (x2Dims[i] == 1) ? 0 : x2Running;
        x1Running *= x1Dims[i];
        x2Running *= x2Dims[i];
    }

    return totalElems > 0;
}

inline void SetTilingArrays(optiling::TensorEqualTilingData& tiling, const int64_t* outDims,
                            const int64_t* x1Dims, const int64_t* x2Dims,
                            const int64_t* outStrides, const int64_t* x1Strides,
                            const int64_t* x2Strides)
{
    tiling.set_out_dim0(outDims[0]);
    tiling.set_out_dim1(outDims[1]);
    tiling.set_out_dim2(outDims[2]);
    tiling.set_out_dim3(outDims[3]);
    tiling.set_out_dim4(outDims[4]);
    tiling.set_out_dim5(outDims[5]);
    tiling.set_out_dim6(outDims[6]);
    tiling.set_out_dim7(outDims[7]);
    tiling.set_x1_dim0(x1Dims[0]);
    tiling.set_x1_dim1(x1Dims[1]);
    tiling.set_x1_dim2(x1Dims[2]);
    tiling.set_x1_dim3(x1Dims[3]);
    tiling.set_x1_dim4(x1Dims[4]);
    tiling.set_x1_dim5(x1Dims[5]);
    tiling.set_x1_dim6(x1Dims[6]);
    tiling.set_x1_dim7(x1Dims[7]);
    tiling.set_x2_dim0(x2Dims[0]);
    tiling.set_x2_dim1(x2Dims[1]);
    tiling.set_x2_dim2(x2Dims[2]);
    tiling.set_x2_dim3(x2Dims[3]);
    tiling.set_x2_dim4(x2Dims[4]);
    tiling.set_x2_dim5(x2Dims[5]);
    tiling.set_x2_dim6(x2Dims[6]);
    tiling.set_x2_dim7(x2Dims[7]);
    tiling.set_out_stride0(outStrides[0]);
    tiling.set_out_stride1(outStrides[1]);
    tiling.set_out_stride2(outStrides[2]);
    tiling.set_out_stride3(outStrides[3]);
    tiling.set_out_stride4(outStrides[4]);
    tiling.set_out_stride5(outStrides[5]);
    tiling.set_out_stride6(outStrides[6]);
    tiling.set_out_stride7(outStrides[7]);
    tiling.set_x1_stride0(x1Strides[0]);
    tiling.set_x1_stride1(x1Strides[1]);
    tiling.set_x1_stride2(x1Strides[2]);
    tiling.set_x1_stride3(x1Strides[3]);
    tiling.set_x1_stride4(x1Strides[4]);
    tiling.set_x1_stride5(x1Strides[5]);
    tiling.set_x1_stride6(x1Strides[6]);
    tiling.set_x1_stride7(x1Strides[7]);
    tiling.set_x2_stride0(x2Strides[0]);
    tiling.set_x2_stride1(x2Strides[1]);
    tiling.set_x2_stride2(x2Strides[2]);
    tiling.set_x2_stride3(x2Strides[3]);
    tiling.set_x2_stride4(x2Strides[4]);
    tiling.set_x2_stride5(x2Strides[5]);
    tiling.set_x2_stride6(x2Strides[6]);
    tiling.set_x2_stride7(x2Strides[7]);
}

}  // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const ge::DataType x1Dtype = context->GetInputTensor(0)->GetDataType();
    const ge::DataType x2Dtype = context->GetInputTensor(1)->GetDataType();
    if (x1Dtype != x2Dtype || !IsSupportedType(x1Dtype)) {
        return ge::GRAPH_FAILED;
    }

    int64_t outDims[kMaxRank];
    int64_t x1Dims[kMaxRank];
    int64_t x2Dims[kMaxRank];
    int64_t outStrides[kMaxRank];
    int64_t x1Strides[kMaxRank];
    int64_t x2Strides[kMaxRank];
    size_t rank = 0;
    int64_t totalElems = 0;
    bool sameShape = false;
    if (!BuildBroadcastMeta(context->GetInputShape(0)->GetStorageShape(),
                            context->GetInputShape(1)->GetStorageShape(),
                            outDims, x1Dims, x2Dims, outStrides, x1Strides, x2Strides,
                            rank, totalElems, sameShape)) {
        return ge::GRAPH_FAILED;
    }

    TensorEqualTilingData tiling;
    tiling.set_total_elems(totalElems);
    tiling.set_elems_per_core(totalElems);
    const bool x1ScalarBuffer = UseX1ScalarBuffer(sameShape, rank, x1Dims, x2Dims);
    const bool x2ScalarBuffer = UseX2ScalarBuffer(sameShape, rank, x1Dims, x2Dims);
    tiling.set_tile_elems(SelectTileElems(x1Dtype, totalElems, x1ScalarBuffer, x2ScalarBuffer));
    tiling.set_rank(static_cast<int64_t>(rank));
    tiling.set_same_shape(sameShape ? 1 : 0);
    SetTilingArrays(tiling, outDims, x1Dims, x2Dims, outStrides, x1Strides, x2Strides);

    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    int64_t outDims[kMaxRank];
    int64_t x1Dims[kMaxRank];
    int64_t x2Dims[kMaxRank];
    int64_t outStrides[kMaxRank];
    int64_t x1Strides[kMaxRank];
    int64_t x2Strides[kMaxRank];
    size_t rank = 0;
    int64_t totalElems = 0;
    bool sameShape = false;
    if (!BuildBroadcastMeta(*context->GetInputShape(0), *context->GetInputShape(1),
                            outDims, x1Dims, x2Dims, outStrides, x1Strides, x2Strides,
                            rank, totalElems, sameShape)) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(static_cast<int64_t>(rank));
    for (size_t i = 0; i < rank; ++i) {
        outputShape->SetDim(static_cast<int64_t>(i), outDims[i]);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context->GetInputDataType(0) != context->GetInputDataType(1)) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_BOOL);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class TensorEqual : public OpDef {
public:
    explicit TensorEqual(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32,
                       ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32,
                       ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL,
                       ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(TensorEqual);

}  // namespace ops
