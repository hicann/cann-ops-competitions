#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/addcmul_tiling.h"
#include "../op_kernel/tiling_key_addcmul.h"

namespace {
constexpr uint32_t DEFAULT_BLOCK_DIM = 1;
constexpr uint32_t DEFAULT_TILE_BYTES = 16 * 1024;
constexpr uint32_t TARGET_ELEMENTS_PER_CORE = 2048;
constexpr uint32_t TARGET_INT8_ELEMENTS_PER_CORE = 512;

static uint32_t ShapeSize(const gert::Shape &shape)
{
    const size_t dimNum = shape.GetDimNum();
    if (dimNum == 0) {
        return 1;
    }
    uint64_t size = 1;
    for (size_t i = 0; i < dimNum; ++i) {
        size *= static_cast<uint64_t>(shape.GetDim(i));
    }
    return static_cast<uint32_t>(size);
}

static bool BuildBroadcastShape(const gert::Shape &inputShape, const gert::Shape &x1Shape,
                                const gert::Shape &x2Shape, gert::Shape &outShape)
{
    const size_t inputDimNum = inputShape.GetDimNum();
    const size_t x1DimNum = x1Shape.GetDimNum();
    const size_t x2DimNum = x2Shape.GetDimNum();
    size_t dimNum = inputDimNum > x1DimNum ? inputDimNum : x1DimNum;
    dimNum = dimNum > x2DimNum ? dimNum : x2DimNum;
    if (dimNum > ADDCMUL_MAX_DIMS) {
        return false;
    }

    outShape.SetDimNum(dimNum);
    for (size_t rev = 0; rev < dimNum; ++rev) {
        const int64_t inputDim = rev < inputDimNum ? inputShape.GetDim(inputDimNum - 1 - rev) : 1;
        const int64_t x1Dim = rev < x1DimNum ? x1Shape.GetDim(x1DimNum - 1 - rev) : 1;
        const int64_t x2Dim = rev < x2DimNum ? x2Shape.GetDim(x2DimNum - 1 - rev) : 1;
        const bool hasZero = inputDim == 0 || x1Dim == 0 || x2Dim == 0;
        int64_t outDim = hasZero ? 0 : (inputDim > x1Dim ? inputDim : x1Dim);
        if (!hasZero) {
            outDim = outDim > x2Dim ? outDim : x2Dim;
        }
        if ((inputDim != outDim && inputDim != 1) || (x1Dim != outDim && x1Dim != 1) ||
            (x2Dim != outDim && x2Dim != 1)) {
            return false;
        }
        outShape.SetDim(dimNum - 1 - rev, outDim);
    }
    return true;
}

static void FillShapeAndStride(const gert::Shape &srcShape, const gert::Shape &outShape, uint32_t *shape,
                               uint32_t *stride)
{
    const size_t outDimNum = outShape.GetDimNum();
    const size_t srcDimNum = srcShape.GetDimNum();
    uint32_t compactStride[ADDCMUL_MAX_DIMS] = {0};
    uint32_t running = 1;
    for (int32_t i = static_cast<int32_t>(srcDimNum) - 1; i >= 0; --i) {
        compactStride[i] = running;
        running *= static_cast<uint32_t>(srcShape.GetDim(i));
    }

    for (size_t i = 0; i < ADDCMUL_MAX_DIMS; ++i) {
        shape[i] = 1;
        stride[i] = 0;
    }

    const size_t offset = outDimNum - srcDimNum;
    for (size_t i = 0; i < outDimNum; ++i) {
        if (i < offset) {
            shape[i] = 1;
            stride[i] = 0;
            continue;
        }
        const size_t srcIdx = i - offset;
        shape[i] = static_cast<uint32_t>(srcShape.GetDim(srcIdx));
        stride[i] = shape[i] == 1 ? 0 : compactStride[srcIdx];
    }
}

static void FillOutShapeAndStride(const gert::Shape &outShape, uint32_t *shape, uint32_t *stride)
{
    const size_t outDimNum = outShape.GetDimNum();
    uint32_t running = 1;
    for (size_t i = 0; i < ADDCMUL_MAX_DIMS; ++i) {
        shape[i] = 1;
        stride[i] = 0;
    }
    for (int32_t i = static_cast<int32_t>(outDimNum) - 1; i >= 0; --i) {
        shape[i] = static_cast<uint32_t>(outShape.GetDim(i));
        stride[i] = running;
        running *= shape[i];
    }
}

static bool SameShape(const gert::Shape &lhs, const gert::Shape &rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

static bool IsLastDimBroadcast(const gert::Shape &srcShape, const gert::Shape &outShape)
{
    const size_t outDimNum = outShape.GetDimNum();
    if (outDimNum == 0) {
        return false;
    }
    const uint32_t lastDim = static_cast<uint32_t>(outShape.GetDim(outDimNum - 1));
    if (lastDim <= 1 || ShapeSize(srcShape) != lastDim) {
        return false;
    }
    const size_t srcDimNum = srcShape.GetDimNum();
    for (size_t rev = 0; rev < outDimNum; ++rev) {
        const int64_t srcDim = rev < srcDimNum ? srcShape.GetDim(srcDimNum - 1 - rev) : 1;
        const int64_t outDim = outShape.GetDim(outDimNum - 1 - rev);
        if (rev == 0) {
            if (srcDim != outDim) {
                return false;
            }
        } else if (srcDim != 1) {
            return false;
        }
    }
    return true;
}

static bool IsOuterBroadcast(const gert::Shape &srcShape, const gert::Shape &outShape)
{
    const size_t outDimNum = outShape.GetDimNum();
    if (outDimNum == 0) {
        return false;
    }
    const uint32_t lastDim = static_cast<uint32_t>(outShape.GetDim(outDimNum - 1));
    if (lastDim <= 1 || ShapeSize(srcShape) * lastDim != ShapeSize(outShape)) {
        return false;
    }
    const size_t srcDimNum = srcShape.GetDimNum();
    for (size_t rev = 0; rev < outDimNum; ++rev) {
        const int64_t srcDim = rev < srcDimNum ? srcShape.GetDim(srcDimNum - 1 - rev) : 1;
        const int64_t outDim = outShape.GetDim(outDimNum - 1 - rev);
        if (rev == 0) {
            if (srcDim != 1) {
                return false;
            }
        } else if (srcDim != outDim) {
            return false;
        }
    }
    return true;
}

static uint32_t SelectFastMode(ge::DataType dtype, const gert::Shape &inputShape, const gert::Shape &x1Shape,
                               const gert::Shape &x2Shape, const gert::Shape &outShape)
{
    if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT && dtype != ge::DT_INT32 && dtype != ge::DT_INT8) {
        return 99;
    }
    const uint32_t totalLength = ShapeSize(outShape);
    const uint32_t inputLength = ShapeSize(inputShape);
    const uint32_t x1Length = ShapeSize(x1Shape);
    const uint32_t x2Length = ShapeSize(x2Shape);
    if (inputLength == totalLength && x1Length == totalLength && x2Length == totalLength) {
        return 0;
    }
    if (inputLength == totalLength && x1Length == totalLength && x2Length == 1) {
        return 1;
    }
    if (inputLength == totalLength && x1Length == 1 && x2Length == totalLength) {
        return 2;
    }
    if (inputLength == 1 && x1Length == totalLength && x2Length == totalLength) {
        return 3;
    }
    if (inputLength == totalLength && x1Length == totalLength && IsLastDimBroadcast(x2Shape, outShape)) {
        return 4;
    }
    if (inputLength == totalLength && x2Length == totalLength && IsLastDimBroadcast(x1Shape, outShape)) {
        return 5;
    }
    if (x1Length == totalLength && x2Length == totalLength && IsLastDimBroadcast(inputShape, outShape)) {
        return 6;
    }
    if (inputLength == totalLength && x1Length == totalLength && IsOuterBroadcast(x2Shape, outShape)) {
        return 7;
    }
    if (inputLength == totalLength && x2Length == totalLength && IsOuterBroadcast(x1Shape, outShape)) {
        return 8;
    }
    if (x1Length == totalLength && x2Length == totalLength && IsOuterBroadcast(inputShape, outShape)) {
        return 9;
    }
    if (inputLength == totalLength && x1Length == 1 && x2Length == 1) {
        return 10;
    }
    if (inputLength == 1 && x1Length == totalLength && x2Length == 1) {
        return 11;
    }
    if (inputLength == 1 && x1Length == 1 && x2Length == totalLength) {
        return 12;
    }
    if (inputLength == totalLength && IsLastDimBroadcast(x1Shape, outShape) && x2Length == 1) {
        return 13;
    }
    if (inputLength == totalLength && x1Length == 1 && IsLastDimBroadcast(x2Shape, outShape)) {
        return 14;
    }
    if (inputLength == 1 && x1Length == totalLength && IsLastDimBroadcast(x2Shape, outShape)) {
        return 15;
    }
    if (inputLength == 1 && IsLastDimBroadcast(x1Shape, outShape) && x2Length == totalLength) {
        return 16;
    }
    if (IsLastDimBroadcast(inputShape, outShape) && x1Length == totalLength && x2Length == 1) {
        return 17;
    }
    if (IsLastDimBroadcast(inputShape, outShape) && x1Length == 1 && x2Length == totalLength) {
        return 18;
    }
    if (inputLength == totalLength && IsLastDimBroadcast(x1Shape, outShape) &&
        IsLastDimBroadcast(x2Shape, outShape)) {
        return 19;
    }
    if (IsLastDimBroadcast(inputShape, outShape) && x1Length == totalLength &&
        IsLastDimBroadcast(x2Shape, outShape)) {
        return 20;
    }
    if (IsLastDimBroadcast(inputShape, outShape) && IsLastDimBroadcast(x1Shape, outShape) &&
        x2Length == totalLength) {
        return 21;
    }
    return 99;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t coreNum = platform.GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = DEFAULT_BLOCK_DIM;
    }

    const gert::Tensor *tensorInput = context->GetRequiredInputTensor(0);
    ge::DataType dtype = tensorInput->GetDataType();
    const uint32_t dtypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(dtype));
    uint32_t DT_INPUT_DATA = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_DATA);

    const gert::Shape &inputShape = context->GetInputShape(0)->GetOriginShape();
    const gert::Shape &x1Shape = context->GetInputShape(1)->GetOriginShape();
    const gert::Shape &x2Shape = context->GetInputShape(2)->GetOriginShape();
    gert::Shape outShape;
    if (!BuildBroadcastShape(inputShape, x1Shape, x2Shape, outShape)) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t totalLength = ShapeSize(outShape);
    const uint32_t targetElementsPerCore = dtype == ge::DT_INT8 ? TARGET_INT8_ELEMENTS_PER_CORE : TARGET_ELEMENTS_PER_CORE;
    uint32_t blockDim = DEFAULT_BLOCK_DIM;
    if (totalLength > targetElementsPerCore) {
        blockDim = (totalLength + targetElementsPerCore - 1) / targetElementsPerCore;
        if (blockDim > static_cast<uint32_t>(coreNum)) {
            blockDim = static_cast<uint32_t>(coreNum);
        }
    }
    context->SetBlockDim(blockDim);

    AddcmulTilingData *tiling = context->GetTilingData<AddcmulTilingData>();
    tiling->totalLength = totalLength;
    tiling->inputLength = ShapeSize(inputShape);
    tiling->x1Length = ShapeSize(x1Shape);
    tiling->x2Length = ShapeSize(x2Shape);
    const uint32_t alignNum = dtypeSize == 0 ? 1 : (32 / dtypeSize);
    uint32_t blockLength = blockDim == 0 ? 0 : (totalLength + blockDim - 1) / blockDim;
    if (alignNum > 1) {
        blockLength = ((blockLength + alignNum - 1) / alignNum) * alignNum;
    }
    tiling->blockLength = blockLength;
    tiling->alignedLength = alignNum == 0 ? tiling->blockLength : (tiling->blockLength / alignNum) * alignNum;
    uint32_t tileLength = dtypeSize == 0 ? tiling->alignedLength : (DEFAULT_TILE_BYTES / dtypeSize);
    tileLength = alignNum == 0 ? tileLength : (tileLength / alignNum) * alignNum;
    if (tileLength == 0) {
        tileLength = alignNum == 0 ? 1 : alignNum;
    }
    if (tiling->alignedLength > 0 && tileLength > tiling->alignedLength) {
        tileLength = tiling->alignedLength;
    }
    tiling->tileLength = tileLength;
    tiling->tailLength = totalLength;
    tiling->dimNum = static_cast<uint32_t>(outShape.GetDimNum());
    tiling->fastMode = SelectFastMode(dtype, inputShape, x1Shape, x2Shape, outShape);
    tiling->lastDim = outShape.GetDimNum() == 0 ? 1 : static_cast<uint32_t>(outShape.GetDim(outShape.GetDimNum() - 1));
    tiling->reserved = 0;
    if (tiling->fastMode == 99) {
        FillOutShapeAndStride(outShape, tiling->outShape, tiling->outStride);
        FillShapeAndStride(inputShape, outShape, tiling->inputShape, tiling->inputStride);
        FillShapeAndStride(x1Shape, outShape, tiling->x1Shape, tiling->x1Stride);
        FillShapeAndStride(x2Shape, outShape, tiling->x2Shape, tiling->x2Stride);
    }

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    const gert::Shape *x1Shape = context->GetInputShape(1);
    const gert::Shape *x2Shape = context->GetInputShape(2);
    gert::Shape outShape;
    if (!BuildBroadcastShape(*inputShape, *x1Shape, *x2Shape, outShape)) {
        return GRAPH_FAILED;
    }
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = outShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
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
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(Addcmul);
}  // namespace ops
