// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "../op_kernel/addcmul_tiling.h"
#include "../op_kernel/tiling_key_addcmul.h"

namespace {
constexpr size_t INPUT_DATA_INDEX = 0;
constexpr size_t X1_INDEX = 1;
constexpr size_t X2_INDEX = 2;
constexpr size_t VALUE_INDEX = 3;
constexpr size_t Y_INDEX = 0;
constexpr int64_t UNKNOWN_DIM = -1;
constexpr int64_t UNKNOWN_RANK = -2;

struct AccessModeInfo {
    uint32_t mode;
    uint64_t repeat;
    uint64_t dataLength;
};

static bool IsSupportedDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT || dtype == ge::DT_INT8 || dtype == ge::DT_INT32;
}

static bool IsVectorFastDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT || dtype == ge::DT_INT32;
}

static bool IsUnknownShape(const gert::Shape *shape)
{
    if (shape == nullptr) {
        return true;
    }
    if (shape->GetDimNum() == 1 && shape->GetDim(0) == UNKNOWN_RANK) {
        return true;
    }
    for (size_t i = 0; i < shape->GetDimNum(); ++i) {
        if (shape->GetDim(i) == UNKNOWN_DIM) {
            return true;
        }
    }
    return false;
}

static bool SafeMulU64(uint64_t lhs, uint64_t rhs, uint64_t &out)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    out = lhs * rhs;
    return true;
}

static bool CalcShapeSize(const int64_t *dims, size_t rank, uint64_t &size)
{
    size = 1;
    for (size_t i = 0; i < rank; ++i) {
        if (dims[i] < 0) {
            return false;
        }
        if (dims[i] == 0) {
            size = 0;
            return true;
        }
        if (!SafeMulU64(size, static_cast<uint64_t>(dims[i]), size)) {
            return false;
        }
    }
    return true;
}

static bool CalcGertShapeSize(const gert::Shape *shape, uint64_t &size)
{
    if (shape == nullptr || IsUnknownShape(shape)) {
        return false;
    }
    size = 1;
    for (size_t i = 0; i < shape->GetDimNum(); ++i) {
        const int64_t dim = shape->GetDim(i);
        if (dim < 0) {
            return false;
        }
        if (dim == 0) {
            size = 0;
            return true;
        }
        if (!SafeMulU64(size, static_cast<uint64_t>(dim), size)) {
            return false;
        }
    }
    return true;
}

static bool IsOneElementShape(const gert::Shape *shape)
{
    uint64_t shapeSize = 0;
    return CalcGertShapeSize(shape, shapeSize) && shapeSize == 1;
}

static uint64_t CeilDivU64(uint64_t a, uint64_t b)
{
    return (b == 0) ? 0 : (a + b - 1) / b;
}

static uint64_t AlignUpU64(uint64_t value, uint64_t align)
{
    if (align == 0) {
        return value;
    }
    return ((value + align - 1) / align) * align;
}

static uint64_t AlignDownU64(uint64_t value, uint64_t align)
{
    if (align == 0) {
        return value;
    }
    return (value / align) * align;
}

static ge::graphStatus Broadcast3Shapes(const gert::Shape *shape0, const gert::Shape *shape1,
                                        const gert::Shape *shape2, int64_t *outDims, size_t &outRank)
{
    if (shape0 == nullptr || shape1 == nullptr || shape2 == nullptr ||
        IsUnknownShape(shape0) || IsUnknownShape(shape1) || IsUnknownShape(shape2)) {
        return ge::GRAPH_FAILED;
    }

    const size_t rank0 = shape0->GetDimNum();
    const size_t rank1 = shape1->GetDimNum();
    const size_t rank2 = shape2->GetDimNum();
    outRank = std::max(rank0, std::max(rank1, rank2));
    if (outRank > ADDCMUL_MAX_DIMS) {
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < outRank; ++i) {
        const size_t outAxis = outRank - 1 - i;
        const int64_t dims[3] = {
            (i < rank0) ? shape0->GetDim(rank0 - 1 - i) : 1,
            (i < rank1) ? shape1->GetDim(rank1 - 1 - i) : 1,
            (i < rank2) ? shape2->GetDim(rank2 - 1 - i) : 1,
        };

        int64_t outDim = 1;
        for (size_t j = 0; j < 3; ++j) {
            if (dims[j] < 0) {
                return ge::GRAPH_FAILED;
            }
            if (dims[j] == 1) {
                continue;
            }
            if (outDim == 1) {
                outDim = dims[j];
            } else if (outDim != dims[j]) {
                return ge::GRAPH_FAILED;
            }
        }
        outDims[outAxis] = outDim;
    }
    return ge::GRAPH_SUCCESS;
}

static void FillBroadcastStrides(const gert::Shape *shape, const int64_t *outDims, size_t outRank, uint64_t *strides)
{
    for (size_t i = 0; i < ADDCMUL_MAX_DIMS; ++i) {
        strides[i] = 0;
    }

    const size_t rank = shape->GetDimNum();
    if (rank == 0 || outRank == 0) {
        return;
    }

    uint64_t baseStrides[ADDCMUL_MAX_DIMS] = {0};
    baseStrides[rank - 1] = 1;
    for (int32_t axis = static_cast<int32_t>(rank) - 2; axis >= 0; --axis) {
        baseStrides[axis] = baseStrides[axis + 1] * static_cast<uint64_t>(shape->GetDim(axis + 1));
    }

    const size_t leftPad = outRank - rank;
    for (size_t outAxis = 0; outAxis < outRank; ++outAxis) {
        if (outAxis < leftPad) {
            strides[outAxis] = 0;
            continue;
        }
        const size_t inAxis = outAxis - leftPad;
        const int64_t inDim = shape->GetDim(inAxis);
        strides[outAxis] = (inDim == 1 && outDims[outAxis] != 1) ? 0 : baseStrides[inAxis];
    }
}

static bool IsLinearAccess(const uint64_t *strides, const uint64_t *outShape, size_t outRank)
{
    uint64_t contiguousStride = 1;
    for (int32_t axis = static_cast<int32_t>(outRank) - 1; axis >= 0; --axis) {
        const uint64_t extent = outShape[axis];
        if (extent > 1 && strides[axis] != contiguousStride) {
            return false;
        }
        if (!SafeMulU64(contiguousStride, (extent == 0 ? 1ULL : extent), contiguousStride)) {
            return false;
        }
    }
    return true;
}

static AccessModeInfo GetAccessModeInfo(const gert::Shape *shape, const uint64_t *outShape,
                                        size_t outRank, const uint64_t *strides)
{
    AccessModeInfo info = {ADDCMUL_MODE_GENERIC, 1ULL, 1ULL};

    uint64_t srcSize = 0;
    if (CalcGertShapeSize(shape, srcSize) && srcSize == 1) {
        info.mode = ADDCMUL_MODE_SCALAR;
        return info;
    }
    if (IsLinearAccess(strides, outShape, outRank)) {
        info.mode = ADDCMUL_MODE_LINEAR;
        return info;
    }

    // 识别“一个连续源片段在输出中重复”的常见广播模式：
    // [N]->[M,N]、[M,1]->[M,N]、[1,C,1,1]->[B,C,H,W]、[B,C,1,1]->[B,C,H,W] 等。
    int32_t first = -1;
    int32_t last = -1;
    for (size_t axis = 0; axis < outRank; ++axis) {
        if (outShape[axis] > 1 && strides[axis] != 0) {
            if (first < 0) {
                first = static_cast<int32_t>(axis);
            }
            last = static_cast<int32_t>(axis);
        }
    }
    if (first < 0) {
        info.mode = ADDCMUL_MODE_SCALAR;
        return info;
    }

    uint64_t expectedStride = 1;
    for (int32_t axis = last; axis >= first; --axis) {
        if (outShape[axis] <= 1) {
            continue;
        }
        if (strides[axis] != expectedStride) {
            return info;
        }
        if (!SafeMulU64(expectedStride, outShape[axis], expectedStride)) {
            return info;
        }
    }
    for (int32_t axis = 0; axis < first; ++axis) {
        if (outShape[axis] > 1 && strides[axis] != 0) {
            return info;
        }
    }
    for (size_t axis = static_cast<size_t>(last + 1); axis < outRank; ++axis) {
        if (outShape[axis] > 1 && strides[axis] != 0) {
            return info;
        }
    }

    uint64_t repeat = 1;
    for (size_t axis = static_cast<size_t>(last + 1); axis < outRank; ++axis) {
        if (!SafeMulU64(repeat, (outShape[axis] == 0 ? 1ULL : outShape[axis]), repeat)) {
            return info;
        }
    }
    uint64_t dataLength = 1;
    for (int32_t axis = first; axis <= last; ++axis) {
        if (!SafeMulU64(dataLength, (outShape[axis] == 0 ? 1ULL : outShape[axis]), dataLength)) {
            return info;
        }
    }

    info.mode = ADDCMUL_MODE_SEGMENT;
    info.repeat = std::max<uint64_t>(repeat, 1ULL);
    info.dataLength = std::max<uint64_t>(dataLength, 1ULL);
    return info;
}

static ge::graphStatus SetBroadcastOutputShape(gert::InferShapeContext *context)
{
    const gert::Shape *shape0 = context->GetInputShape(INPUT_DATA_INDEX);
    const gert::Shape *shape1 = context->GetInputShape(X1_INDEX);
    const gert::Shape *shape2 = context->GetInputShape(X2_INDEX);
    const gert::Shape *shapeValue = context->GetInputShape(VALUE_INDEX);
    gert::Shape *outShape = context->GetOutputShape(Y_INDEX);
    if (outShape == nullptr || !IsOneElementShape(shapeValue)) {
        return ge::GRAPH_FAILED;
    }

    int64_t outDims[ADDCMUL_MAX_DIMS] = {0};
    size_t outRank = 0;
    if (Broadcast3Shapes(shape0, shape1, shape2, outDims, outRank) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    outShape->SetDimNum(0);
    for (size_t i = 0; i < outRank; ++i) {
        outShape->AppendDim(outDims[i]);
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t numCoresAiv = std::max<int32_t>(1, platform.GetCoreNumAiv());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const gert::Tensor *tensorInputData = context->GetRequiredInputTensor(INPUT_DATA_INDEX);
    const gert::Tensor *tensorX1 = context->GetRequiredInputTensor(X1_INDEX);
    const gert::Tensor *tensorX2 = context->GetRequiredInputTensor(X2_INDEX);
    const gert::Tensor *tensorValue = context->GetRequiredInputTensor(VALUE_INDEX);
    if (tensorInputData == nullptr || tensorX1 == nullptr || tensorX2 == nullptr || tensorValue == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtypeInputData = tensorInputData->GetDataType();
    if (!IsSupportedDtype(dtypeInputData) || tensorX1->GetDataType() != dtypeInputData ||
        tensorX2->GetDataType() != dtypeInputData || tensorValue->GetDataType() != dtypeInputData) {
        return ge::GRAPH_FAILED;
    }
    if (tensorValue->GetShapeSize() != 1) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &shapeInputData = tensorInputData->GetStorageShape();
    const gert::Shape &shapeX1 = tensorX1->GetStorageShape();
    const gert::Shape &shapeX2 = tensorX2->GetStorageShape();

    int64_t outDims[ADDCMUL_MAX_DIMS] = {0};
    size_t outRank = 0;
    if (Broadcast3Shapes(&shapeInputData, &shapeX1, &shapeX2, outDims, outRank) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint64_t outputLength = 0;
    if (!CalcShapeSize(outDims, outRank, outputLength)) {
        return ge::GRAPH_FAILED;
    }

    const int dtypeSize = ge::GetSizeByDataType(dtypeInputData);
    if (dtypeSize <= 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t dataBlockElems = std::max<uint64_t>(1ULL,
        ADDCMUL_DATA_BLOCK_BYTES / static_cast<uint64_t>(dtypeSize));
    const uint64_t cacheLineElems = std::max<uint64_t>(1ULL,
        ADDCMUL_CACHE_LINE_BYTES / static_cast<uint64_t>(dtypeSize));

    AddcmulTilingData *tiling = context->GetTilingData<AddcmulTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tiling->length = outputLength;
    tiling->dimNum = static_cast<uint32_t>(outRank);
    tiling->reserved = 0;
    for (size_t i = 0; i < ADDCMUL_MAX_DIMS; ++i) {
        tiling->outShape[i] = (i < outRank) ? static_cast<uint64_t>(outDims[i]) : 1ULL;
    }
    FillBroadcastStrides(&shapeInputData, outDims, outRank, tiling->inputStrides);
    FillBroadcastStrides(&shapeX1, outDims, outRank, tiling->x1Strides);
    FillBroadcastStrides(&shapeX2, outDims, outRank, tiling->x2Strides);

    const AccessModeInfo inputInfo = GetAccessModeInfo(&shapeInputData, tiling->outShape, outRank, tiling->inputStrides);
    const AccessModeInfo x1Info = GetAccessModeInfo(&shapeX1, tiling->outShape, outRank, tiling->x1Strides);
    const AccessModeInfo x2Info = GetAccessModeInfo(&shapeX2, tiling->outShape, outRank, tiling->x2Strides);
    tiling->inputMode = inputInfo.mode;
    tiling->x1Mode = x1Info.mode;
    tiling->x2Mode = x2Info.mode;
    tiling->inputRepeat = inputInfo.repeat;
    tiling->x1Repeat = x1Info.repeat;
    tiling->x2Repeat = x2Info.repeat;
    tiling->inputDataLength = inputInfo.dataLength;
    tiling->x1DataLength = x1Info.dataLength;
    tiling->x2DataLength = x2Info.dataLength;
    tiling->sameShape = (tiling->inputMode == ADDCMUL_MODE_LINEAR &&
                         tiling->x1Mode == ADDCMUL_MODE_LINEAR &&
                         tiling->x2Mode == ADDCMUL_MODE_LINEAR) ? 1U : 0U;

    uint64_t tileLength = ADDCMUL_DEFAULT_TILE_LENGTH;
    if (ubSize > ADDCMUL_UB_RESERVED_BYTES) {
        const uint64_t usableUb = ubSize - ADDCMUL_UB_RESERVED_BYTES;
        const uint64_t localBufferNum = (IsVectorFastDtype(dtypeInputData) && tiling->sameShape == 1U)
            ? static_cast<uint64_t>(ADDCMUL_LINEAR_DOUBLE_BUFFER_NUM)
            : static_cast<uint64_t>(ADDCMUL_LOCAL_BUFFER_NUM);
        const uint64_t ubLimitedTile = usableUb /
            (static_cast<uint64_t>(dtypeSize) * localBufferNum);
        tileLength = std::min<uint64_t>(tileLength, ubLimitedTile);
    }
    tileLength = AlignDownU64(tileLength, dataBlockElems);
    if (tileLength < dataBlockElems) {
        tileLength = dataBlockElems;
    }
    tiling->tileLength = static_cast<uint32_t>(tileLength);

    uint32_t blockDim = 1;
    if (outputLength > 0) {
        const uint64_t usefulCores = std::min<uint64_t>(static_cast<uint64_t>(numCoresAiv),
            std::max<uint64_t>(1ULL, CeilDivU64(outputLength, cacheLineElems)));
        const uint64_t rawPerCore = CeilDivU64(outputLength, usefulCores);
        tiling->perCoreElements = AlignUpU64(rawPerCore, cacheLineElems);
        blockDim = static_cast<uint32_t>(std::max<uint64_t>(1ULL, CeilDivU64(outputLength, tiling->perCoreElements)));
    } else {
        tiling->perCoreElements = cacheLineElems;
        blockDim = 1;
    }

    const uint32_t DT_INPUT_DATA = static_cast<uint32_t>(dtypeInputData);
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_DATA);
    context->SetBlockDim(blockDim);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    return SetBroadcastOutputShape(context);
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const ge::DataType inputDtype = context->GetInputDataType(INPUT_DATA_INDEX);
    if (!IsSupportedDtype(inputDtype) || context->GetInputDataType(X1_INDEX) != inputDtype ||
        context->GetInputDataType(X2_INDEX) != inputDtype || context->GetInputDataType(VALUE_INDEX) != inputDtype) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(Y_INDEX, inputDtype);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Addcmul : public OpDef {
public:
    explicit Addcmul(const char *name) : OpDef(name)
    {
        this->Input("input_data")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Addcmul);
}  // namespace ops
