// Host侧Tiling实现
#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/addcmul_tiling.h"
#include "../op_kernel/tiling_key_addcmul.h"

namespace {

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t UB_BUFFER_NUM = 2;

static bool AlignShape(const gert::Shape *shape, int64_t alignedShape[ADDCMUL_MAX_DIM], uint32_t &dimNum)
{
    dimNum = shape->GetDimNum();
    if (dimNum > ADDCMUL_MAX_DIM) {
        return false;
    }
    for (uint32_t i = 0; i < dimNum; ++i) {
        alignedShape[i] = shape->GetDim(i);
    }
    return true;
}

static bool LeftPadShape(int64_t shape[8], uint32_t &dimNum, uint32_t targetDimNum)
{
    if (targetDimNum > 8 || dimNum > targetDimNum) {
        return false;
    }
    if (dimNum == targetDimNum) {
        return true;
    }
    uint32_t pad = targetDimNum - dimNum;
    for (int32_t i = static_cast<int32_t>(dimNum) - 1; i >= 0; --i) {
        shape[i + pad] = shape[i];
    }
    for (uint32_t i = 0; i < pad; ++i) {
        shape[i] = 1;
    }
    dimNum = targetDimNum;
    return true;
}

static bool BroadcastShapes(const gert::Shape *shapes[], uint32_t shapeCount, int64_t outShape[8], uint32_t &outDimNum)
{
    uint32_t maxDimNum = 0;
    int64_t alignedShapes[3][8] = {};
    uint32_t dimNums[3] = {0, 0, 0};

    for (uint32_t i = 0; i < shapeCount; ++i) {
        if (!AlignShape(shapes[i], alignedShapes[i], dimNums[i])) {
            return false;
        }
        maxDimNum = std::max(maxDimNum, dimNums[i]);
    }

    for (uint32_t i = 0; i < shapeCount; ++i) {
        if (!LeftPadShape(alignedShapes[i], dimNums[i], maxDimNum)) {
            return false;
        }
    }

    outDimNum = maxDimNum;
    for (uint32_t d = 0; d < maxDimNum; ++d) {
        int64_t dimSize = 1;
        for (uint32_t i = 0; i < shapeCount; ++i) {
            int64_t curDim = alignedShapes[i][d];
            if (curDim == 1) {
                continue;
            }
            if (dimSize == 1) {
                dimSize = curDim;
            } else if (dimSize != curDim) {
                return false;
            }
        }
        outShape[d] = dimSize;
    }
    return true;
}

static uint32_t GetTensorShapeSize(const gert::Tensor *tensor)
{
    if (tensor == nullptr) {
        return 0;
    }
    return static_cast<uint32_t>(tensor->GetShapeSize());
}

static uint64_t GetShapeNumel(const gert::Shape *shape)
{
    if (shape == nullptr) {
        return 0;
    }
    uint64_t size = 1;
    for (uint32_t i = 0; i < shape->GetDimNum(); ++i) {
        size *= static_cast<uint64_t>(shape->GetDim(i));
    }
    return size;
}

static void CalcBroadcastStrides(const int64_t inShape[ADDCMUL_MAX_DIM], uint32_t dimNum,
                                 int64_t strides[ADDCMUL_MAX_DIM])
{
    int64_t stride = 1;
    for (int32_t i = static_cast<int32_t>(dimNum) - 1; i >= 0; --i) {
        if (inShape[i] == 1) {
            strides[i] = 0;
        } else {
            strides[i] = stride;
            stride *= inShape[i];
        }
    }
}

static bool FillBroadcastStrides(const gert::Shape *inOrigin, const gert::Shape *x1Origin, const gert::Shape *x2Origin,
                                 AddcmulTilingData *tiling)
{
    const gert::Shape *shapes[] = {inOrigin, x1Origin, x2Origin};
    int64_t outShape[ADDCMUL_MAX_DIM] = {0};
    uint32_t outDimNum = 0;
    if (!BroadcastShapes(shapes, 3, outShape, outDimNum)) {
        return false;
    }

    int64_t inDataShape[ADDCMUL_MAX_DIM] = {0};
    int64_t x1AlignedShape[ADDCMUL_MAX_DIM] = {0};
    int64_t x2AlignedShape[ADDCMUL_MAX_DIM] = {0};
    uint32_t inDimNum = 0;
    uint32_t x1DimNum = 0;
    uint32_t x2DimNum = 0;
    if (!AlignShape(inOrigin, inDataShape, inDimNum) || !AlignShape(x1Origin, x1AlignedShape, x1DimNum) ||
        !AlignShape(x2Origin, x2AlignedShape, x2DimNum)) {
        return false;
    }
    if (!LeftPadShape(inDataShape, inDimNum, outDimNum) || !LeftPadShape(x1AlignedShape, x1DimNum, outDimNum) ||
        !LeftPadShape(x2AlignedShape, x2DimNum, outDimNum)) {
        return false;
    }

    tiling->dimNum = outDimNum;
    for (uint32_t i = 0; i < outDimNum; ++i) {
        tiling->shape[i] = outShape[i];
    }
    CalcBroadcastStrides(inDataShape, outDimNum, tiling->inDataStride);
    CalcBroadcastStrides(x1AlignedShape, outDimNum, tiling->x1Stride);
    CalcBroadcastStrides(x2AlignedShape, outDimNum, tiling->x2Stride);
    return true;
}

}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    constexpr uint32_t UB_NUM_FLOAT = 10;
    constexpr uint32_t UB_NUM_INT8 = 12;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t aivNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const gert::Tensor *tensorInputData = context->GetRequiredInputTensor(0);
    const gert::Tensor *tensorX1 = context->GetRequiredInputTensor(1);
    const gert::Tensor *tensorX2 = context->GetRequiredInputTensor(2);
    if (tensorInputData == nullptr || tensorX1 == nullptr || tensorX2 == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape *storageInputData = context->GetInputShape(0);
    const gert::StorageShape *storageX1 = context->GetInputShape(1);
    const gert::StorageShape *storageX2 = context->GetInputShape(2);
    if (storageInputData == nullptr || storageX1 == nullptr || storageX2 == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType dtype = tensorInputData->GetDataType();
    uint32_t typeSize = static_cast<uint32_t>(ge::GetSizeByDataType(dtype));
    if (typeSize == 0) {
        typeSize = 1;
    }

    uint32_t inputDataLength = GetTensorShapeSize(tensorInputData);
    uint32_t x1Length = GetTensorShapeSize(tensorX1);
    uint32_t x2Length = GetTensorShapeSize(tensorX2);
    uint32_t totalLength = std::max({inputDataLength, x1Length, x2Length});
    uint32_t minLength = std::min({inputDataLength, x1Length, x2Length});
    uint64_t storageInputNumel = GetShapeNumel(&storageInputData->GetStorageShape());
    uint64_t storageX1Numel = GetShapeNumel(&storageX1->GetStorageShape());
    uint64_t storageX2Numel = GetShapeNumel(&storageX2->GetStorageShape());
    bool lengthContiguous =
        (inputDataLength == totalLength && x1Length == totalLength && x2Length == totalLength);
    bool storageContiguous = (storageInputNumel == totalLength && storageX1Numel == totalLength &&
                              storageX2Numel == totalLength);
    uint32_t isContiguous = (lengthContiguous && storageContiguous) ? 1U : 0U;

    uint32_t ubNum = (dtype == ge::DT_INT8) ? UB_NUM_INT8 : UB_NUM_FLOAT;
    uint32_t alignNum = BLOCK_SIZE / typeSize;
    if (alignNum == 0) {
        alignNum = 1;
    }

    uint32_t tilingBlocks = static_cast<uint32_t>(ubSize / BLOCK_SIZE / UB_BUFFER_NUM / ubNum);
    if (tilingBlocks == 0) {
        tilingBlocks = 1;
    }
    tilingBlocks = tilingBlocks <= 8 ? tilingBlocks : (tilingBlocks / 8 * 8);

    uint32_t blockSize = tilingBlocks * alignNum;
    if (totalLength != minLength && minLength > 1) {
        blockSize = std::min(blockSize, minLength);
        while (blockSize > 0 && (minLength % blockSize != 0 || blockSize % alignNum != 0)) {
            --blockSize;
        }
        if (blockSize == 0) {
            blockSize = alignNum;
        }
    }

    if (totalLength == 0) {
        context->SetBlockDim(1);
        AddcmulTilingData *tiling = context->GetTilingData<AddcmulTilingData>();
        tiling->totalLength = 0;
        tiling->inputDataLength = inputDataLength;
        tiling->x1Length = x1Length;
        tiling->x2Length = x2Length;
        tiling->isContiguous = isContiguous;
        tiling->alignNum = alignNum;
        tiling->blockSize = blockSize;
        tiling->coreSize = 0;
        tiling->coreRemain = 0;
        tiling->dimNum = 0;
    } else {
        if (blockSize > 0) {
            aivNum = std::min(aivNum, static_cast<int32_t>(totalLength / blockSize));
        }
        aivNum = std::max(aivNum, 1);

        uint32_t coreSize = totalLength / static_cast<uint32_t>(aivNum);
        uint32_t coreRemain = totalLength % static_cast<uint32_t>(aivNum);

        AddcmulTilingData *tiling = context->GetTilingData<AddcmulTilingData>();
        tiling->totalLength = totalLength;
        tiling->inputDataLength = inputDataLength;
        tiling->x1Length = x1Length;
        tiling->x2Length = x2Length;
        tiling->isContiguous = isContiguous;
        tiling->alignNum = alignNum;
        tiling->blockSize = blockSize;
        tiling->coreSize = coreSize;
        tiling->coreRemain = coreRemain;

        const gert::Shape *originInput = &storageInputData->GetOriginShape();
        const gert::Shape *originX1 = &storageX1->GetOriginShape();
        const gert::Shape *originX2 = &storageX2->GetOriginShape();
        if (!FillBroadcastStrides(originInput, originX1, originX2, tiling)) {
            return ge::GRAPH_FAILED;
        }

        context->SetBlockDim(aivNum);
    }

    uint32_t dtInputData = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, dtInputData);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *shapeInputData = context->GetInputShape(0);
    const gert::Shape *shapeX1 = context->GetInputShape(1);
    const gert::Shape *shapeX2 = context->GetInputShape(2);
    gert::Shape *outputShape = context->GetOutputShape(0);

    if (shapeInputData == nullptr || shapeX1 == nullptr || shapeX2 == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }

    const gert::Shape *shapes[] = {shapeInputData, shapeX1, shapeX2};
    int64_t outShape[8] = {0};
    uint32_t outDimNum = 0;
    if (!BroadcastShapes(shapes, 3, outShape, outDimNum)) {
        return GRAPH_FAILED;
    }

    outputShape->SetDimNum(outDimNum);
    for (uint32_t i = 0; i < outDimNum; ++i) {
        outputShape->SetDim(i, outShape[i]);
    }
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
