#include <algorithm>
#include <cstdint>
#include <vector>

#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/addcmul_tiling.h"
#include "../op_kernel/tiling_key_addcmul.h"

namespace {
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 1;

template <typename SHAPE>
std::vector<int64_t> ShapeToVector(const SHAPE &shape)
{
    std::vector<int64_t> dims;
    const size_t dimNum = shape.GetDimNum();
    dims.reserve(dimNum);
    for (size_t i = 0; i < dimNum; ++i) {
        dims.push_back(shape.GetDim(i));
    }
    return dims;
}

uint32_t ShapeSize(const std::vector<int64_t> &dims)
{
    if (dims.empty()) {
        return 1;
    }
    uint64_t size = 1;
    for (int64_t dim : dims) {
        if (dim == 0) {
            return 0;
        }
        size *= static_cast<uint64_t>(dim);
    }
    return static_cast<uint32_t>(size);
}

bool BroadcastTwoShapes(const std::vector<int64_t> &lhs, const std::vector<int64_t> &rhs,
                        std::vector<int64_t> &out)
{
    const size_t outRank = std::max(lhs.size(), rhs.size());
    out.assign(outRank, 1);

    for (size_t i = 0; i < outRank; ++i) {
        int64_t ldim = 1;
        int64_t rdim = 1;
        if (i >= outRank - lhs.size()) {
            ldim = lhs[i - (outRank - lhs.size())];
        }
        if (i >= outRank - rhs.size()) {
            rdim = rhs[i - (outRank - rhs.size())];
        }

        if (ldim == rdim) {
            out[i] = ldim;
        } else if (ldim == 1) {
            out[i] = rdim;
        } else if (rdim == 1) {
            out[i] = ldim;
        } else {
            return false;
        }
    }
    return true;
}

bool BroadcastAllShapes(const std::vector<int64_t> &inputDataShape, const std::vector<int64_t> &x1Shape,
                        const std::vector<int64_t> &x2Shape, std::vector<int64_t> &outShape)
{
    std::vector<int64_t> tmpShape;
    if (!BroadcastTwoShapes(inputDataShape, x1Shape, tmpShape)) {
        return false;
    }
    return BroadcastTwoShapes(tmpShape, x2Shape, outShape);
}

bool SameShape(const std::vector<int64_t> &lhs, const std::vector<int64_t> &rhs)
{
    return lhs == rhs;
}

bool IsLastDimBroadcast(const std::vector<int64_t> &inputShape, const std::vector<int64_t> &outShape)
{
    if (inputShape.empty() || outShape.empty()) {
        return false;
    }
    if (inputShape.back() != outShape.back()) {
        return false;
    }
    if (inputShape.size() > outShape.size()) {
        return false;
    }
    const size_t rankOffset = outShape.size() - inputShape.size();
    for (size_t i = 0; i + 1 < inputShape.size(); ++i) {
        if (inputShape[i] != 1 && inputShape[i] != outShape[i + rankOffset]) {
            return false;
        }
        if (inputShape[i] != 1) {
            return false;
        }
    }
    return true;
}

bool IsSuffixBroadcast(const std::vector<int64_t> &inputShape, const std::vector<int64_t> &outShape)
{
    if (inputShape.empty()) {
        return false;
    }
    if (inputShape.size() > outShape.size()) {
        return false;
    }

    const size_t rankOffset = outShape.size() - inputShape.size();
    bool inSuffix = true;
    for (int32_t i = static_cast<int32_t>(inputShape.size()) - 1; i >= 0; --i) {
        const int64_t inputDim = inputShape[i];
        const int64_t outDim = outShape[i + rankOffset];
        if (inSuffix && inputDim == outDim) {
            continue;
        }
        inSuffix = false;
        if (inputDim != 1) {
            return false;
        }
    }
    return true;
}

void FillBroadcastStride(const std::vector<int64_t> &inputShape, const std::vector<int64_t> &outShape,
                         uint32_t *stride)
{
    uint32_t inputStride[ADDCMUL_MAX_DIMS] = {0};
    uint32_t runningStride = 1;
    for (int32_t i = static_cast<int32_t>(inputShape.size()) - 1; i >= 0; --i) {
        inputStride[i] = runningStride;
        runningStride *= static_cast<uint32_t>(inputShape[i]);
    }

    const size_t outRank = outShape.size();
    for (size_t i = 0; i < ADDCMUL_MAX_DIMS; ++i) {
        stride[i] = 0;
    }
    for (size_t i = 0; i < outRank; ++i) {
        if (i < outRank - inputShape.size()) {
            stride[i] = 0;
            continue;
        }
        const size_t inputDimIdx = i - (outRank - inputShape.size());
        stride[i] = inputShape[inputDimIdx] == 1 ? 0 : inputStride[inputDimIdx];
    }
}

void InitEmptyTiling(AddcmulTilingData *tiling, uint32_t typeLength, const std::vector<int64_t> &outShape)
{
    tiling->totalLength = 0;
    tiling->inputDataLength = 0;
    tiling->x1Length = 0;
    tiling->x2Length = 0;
    tiling->smallCoreDataNum = 0;
    tiling->bigCoreDataNum = 0;
    tiling->finalBigTileNum = 0;
    tiling->finalSmallTileNum = 0;
    tiling->tileDataNum = std::max(BLOCK_SIZE / typeLength, static_cast<uint32_t>(1));
    tiling->smallTailDataNum = 0;
    tiling->bigTailDataNum = 0;
    tiling->tailBlockNum = 0;
    tiling->dimNum = static_cast<uint32_t>(outShape.size());
    tiling->noBroadcast = 1;
    for (size_t i = 0; i < ADDCMUL_MAX_DIMS; ++i) {
        tiling->outShape[i] = 1;
        tiling->inputDataStride[i] = 0;
        tiling->x1Stride[i] = 0;
        tiling->x2Stride[i] = 0;
    }
}

void FillCoreTiling(AddcmulTilingData *tiling, platform_ascendc::PlatformAscendC &platform, uint32_t &coreNum,
                    uint32_t totalLength, uint32_t typeLength, uint32_t ubTensorNum)
{
    const uint32_t totalBytes = totalLength * typeLength;
    const uint32_t alignedBytes = (totalBytes + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    const uint32_t totalBlockNum = alignedBytes / BLOCK_SIZE;

    if (totalBlockNum == 0) {
        coreNum = 1;
    } else {
        constexpr uint32_t MIN_BLOCK_PER_CORE = 8;
        uint32_t usefulCoreNum = (totalBlockNum + MIN_BLOCK_PER_CORE - 1) / MIN_BLOCK_PER_CORE;
        coreNum = std::min(coreNum, usefulCoreNum);
        coreNum = std::min(coreNum, totalBlockNum);
        coreNum = std::max(coreNum, static_cast<uint32_t>(1));
    }

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint32_t elementsPerBlock = std::max(BLOCK_SIZE / typeLength, static_cast<uint32_t>(1));
    uint32_t maxTileDataNum = static_cast<uint32_t>(ubSize / (BUFFER_NUM * ubTensorNum * typeLength));
    uint32_t tileDataNum = (maxTileDataNum / elementsPerBlock) * elementsPerBlock;
    tileDataNum = std::max(tileDataNum, elementsPerBlock);
    const uint32_t tileBlockNum = std::max(tileDataNum / elementsPerBlock, static_cast<uint32_t>(1));

    const uint32_t smallCoreBlockNum = totalBlockNum / coreNum;
    const uint32_t tailBlockNum = totalBlockNum % coreNum;

    const uint32_t smallCoreDataNum = smallCoreBlockNum * BLOCK_SIZE / typeLength;
    const uint32_t smallFullTileNum = smallCoreBlockNum / tileBlockNum;
    const uint32_t finalSmallTileNum = smallCoreBlockNum == 0 ? 0 :
        (smallCoreBlockNum % tileBlockNum == 0 ? smallFullTileNum : smallFullTileNum + 1);
    uint32_t smallTailDataNum = smallCoreDataNum - tileDataNum * smallFullTileNum;
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    const uint32_t bigCoreBlockNum = smallCoreBlockNum + 1;
    const uint32_t bigCoreDataNum = bigCoreBlockNum * BLOCK_SIZE / typeLength;
    const uint32_t bigFullTileNum = bigCoreBlockNum / tileBlockNum;
    const uint32_t finalBigTileNum = bigCoreBlockNum == 0 ? 0 :
        (bigCoreBlockNum % tileBlockNum == 0 ? bigFullTileNum : bigFullTileNum + 1);
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigFullTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->tailBlockNum = tailBlockNum;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = platform.GetCoreNum();
    }
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));

    ge::DataType dtypeInputData = context->GetInputDesc(0)->GetDataType();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dtypeInputData, typeLength);
    if (typeLength == 0) {
        return ge::GRAPH_FAILED;
    }

    const auto &inputDataStorageShape = context->GetInputShape(0)->GetStorageShape();
    const auto &x1StorageShape = context->GetInputShape(1)->GetStorageShape();
    const auto &x2StorageShape = context->GetInputShape(2)->GetStorageShape();
    const auto &valueStorageShape = context->GetInputShape(3)->GetStorageShape();

    const std::vector<int64_t> inputDataShape = ShapeToVector(inputDataStorageShape);
    const std::vector<int64_t> x1Shape = ShapeToVector(x1StorageShape);
    const std::vector<int64_t> x2Shape = ShapeToVector(x2StorageShape);
    const std::vector<int64_t> valueShape = ShapeToVector(valueStorageShape);
    std::vector<int64_t> outShape;
    if (!BroadcastAllShapes(inputDataShape, x1Shape, x2Shape, outShape) || outShape.size() > ADDCMUL_MAX_DIMS ||
        ShapeSize(valueShape) != 1) {
        return ge::GRAPH_FAILED;
    }

    AddcmulTilingData *tiling = context->GetTilingData<AddcmulTilingData>();
    const uint32_t totalLength = ShapeSize(outShape);
    if (totalLength == 0) {
        InitEmptyTiling(tiling, typeLength, outShape);
        uint32_t DT_INPUT_DATA = static_cast<uint32_t>(dtypeInputData);
        uint32_t ADD_CMUL_MODE = 1;
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_DATA, ADD_CMUL_MODE);
        context->SetBlockDim(1);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    const bool noBroadcast = SameShape(inputDataShape, outShape) && SameShape(x1Shape, outShape) &&
                             SameShape(x2Shape, outShape);
    const bool x2LastDimBroadcast = SameShape(inputDataShape, outShape) && SameShape(x1Shape, outShape) &&
                                    IsLastDimBroadcast(x2Shape, outShape) && !SameShape(x2Shape, outShape);
    const bool x1LastDimBroadcast = SameShape(inputDataShape, outShape) && SameShape(x2Shape, outShape) &&
                                    IsLastDimBroadcast(x1Shape, outShape) && !SameShape(x1Shape, outShape);
    const bool x2ScalarBroadcast = SameShape(inputDataShape, outShape) && SameShape(x1Shape, outShape) &&
                                   ShapeSize(x2Shape) == 1 && !SameShape(x2Shape, outShape);
    const bool x1ScalarBroadcast = SameShape(inputDataShape, outShape) && SameShape(x2Shape, outShape) &&
                                   ShapeSize(x1Shape) == 1 && !SameShape(x1Shape, outShape);
    const bool inputDataScalarBroadcast = ShapeSize(inputDataShape) == 1 && SameShape(x1Shape, outShape) &&
                                          SameShape(x2Shape, outShape) && !SameShape(inputDataShape, outShape);
    const bool bothMulScalarBroadcast = SameShape(inputDataShape, outShape) && ShapeSize(x1Shape) == 1 &&
                                        ShapeSize(x2Shape) == 1 &&
                                        (!SameShape(x1Shape, outShape) || !SameShape(x2Shape, outShape));
    const bool x2SuffixBroadcast = SameShape(inputDataShape, outShape) && SameShape(x1Shape, outShape) &&
                                   IsSuffixBroadcast(x2Shape, outShape) && !SameShape(x2Shape, outShape);
    const bool x1SuffixBroadcast = SameShape(inputDataShape, outShape) && SameShape(x2Shape, outShape) &&
                                   IsSuffixBroadcast(x1Shape, outShape) && !SameShape(x1Shape, outShape);
    const bool inputDataSuffixBroadcast = SameShape(x1Shape, outShape) && SameShape(x2Shape, outShape) &&
                                          IsSuffixBroadcast(inputDataShape, outShape) &&
                                          !SameShape(inputDataShape, outShape);
    const bool vectorFastPath = noBroadcast && (dtypeInputData == ge::DT_FLOAT16 || dtypeInputData == ge::DT_FLOAT);
    const bool rowBroadcastFastPath = x2LastDimBroadcast &&
                                      (dtypeInputData == ge::DT_FLOAT16 || dtypeInputData == ge::DT_FLOAT);
    const bool rowBroadcastX1FastPath = x1LastDimBroadcast &&
                                        (dtypeInputData == ge::DT_FLOAT16 || dtypeInputData == ge::DT_FLOAT);
    const bool scalarX2FastPath = x2ScalarBroadcast;
    const bool scalarX1FastPath = x1ScalarBroadcast;
    const bool inputDataScalarFastPath = inputDataScalarBroadcast;
    const bool bothMulScalarFastPath = bothMulScalarBroadcast;
    const bool integerLocalFastPath = noBroadcast && (dtypeInputData == ge::DT_INT8 || dtypeInputData == ge::DT_INT32);
    uint32_t DT_INPUT_DATA = static_cast<uint32_t>(dtypeInputData);
    uint32_t ADD_CMUL_MODE = 1;
    if (vectorFastPath) {
        ADD_CMUL_MODE = 0;
    } else if (scalarX2FastPath) {
        ADD_CMUL_MODE = 3;
    } else if (scalarX1FastPath) {
        ADD_CMUL_MODE = 4;
    } else if (inputDataScalarFastPath) {
        ADD_CMUL_MODE = 7;
    } else if (bothMulScalarFastPath) {
        ADD_CMUL_MODE = 8;
    } else if (x2SuffixBroadcast) {
        ADD_CMUL_MODE = 9;
    } else if (x1SuffixBroadcast) {
        ADD_CMUL_MODE = 10;
    } else if (inputDataSuffixBroadcast) {
        ADD_CMUL_MODE = 11;
    } else if (rowBroadcastFastPath) {
        ADD_CMUL_MODE = 2;
    } else if (rowBroadcastX1FastPath) {
        ADD_CMUL_MODE = 5;
    } else if (integerLocalFastPath) {
        ADD_CMUL_MODE = 6;
    }

    uint32_t ubTensorNum = 4;
    if (ADD_CMUL_MODE == 3 || ADD_CMUL_MODE == 4 || ADD_CMUL_MODE == 7) {
        ubTensorNum = 3;
    } else if (ADD_CMUL_MODE == 8) {
        ubTensorNum = 2;
    }

    FillCoreTiling(tiling, platform, coreNum, totalLength, typeLength, ubTensorNum);

    tiling->totalLength = totalLength;
    tiling->inputDataLength = ShapeSize(inputDataShape);
    tiling->x1Length = ShapeSize(x1Shape);
    tiling->x2Length = ShapeSize(x2Shape);
    tiling->dimNum = static_cast<uint32_t>(outShape.size());
    tiling->noBroadcast = noBroadcast ? 1 : 0;
    for (size_t i = 0; i < ADDCMUL_MAX_DIMS; ++i) {
        tiling->outShape[i] = 1;
    }
    for (size_t i = 0; i < outShape.size(); ++i) {
        tiling->outShape[i] = static_cast<uint32_t>(outShape[i]);
    }
    if (ADD_CMUL_MODE == 1) {
        FillBroadcastStride(inputDataShape, outShape, tiling->inputDataStride);
        FillBroadcastStride(x1Shape, outShape, tiling->x1Stride);
        FillBroadcastStride(x2Shape, outShape, tiling->x2Stride);
    }

    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_DATA, ADD_CMUL_MODE);

    context->SetBlockDim(coreNum);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const std::vector<int64_t> inputDataShape = ShapeToVector(*context->GetInputShape(0));
    const std::vector<int64_t> x1Shape = ShapeToVector(*context->GetInputShape(1));
    const std::vector<int64_t> x2Shape = ShapeToVector(*context->GetInputShape(2));
    const std::vector<int64_t> valueShape = ShapeToVector(*context->GetInputShape(3));

    std::vector<int64_t> outShape;
    if (!BroadcastAllShapes(inputDataShape, x1Shape, x2Shape, outShape) || outShape.size() > ADDCMUL_MAX_DIMS ||
        ShapeSize(valueShape) != 1) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(outShape.size());
    for (size_t i = 0; i < outShape.size(); ++i) {
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
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Addcmul);
}  // namespace ops
