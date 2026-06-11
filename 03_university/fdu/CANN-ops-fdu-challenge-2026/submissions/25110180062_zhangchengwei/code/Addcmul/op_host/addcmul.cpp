#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/addcmul_tiling.h"
#include "../op_kernel/tiling_key_addcmul.h"

namespace {
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t UB_BUFFER_PARTS = 9;
constexpr uint32_t TILE_BLOCK_ALIGN = 8;

static uint32_t GetTypeLength(ge::DataType dtype) {
    if (dtype == ge::DT_INT8) {
        return 1U;
    }
    if (dtype == ge::DT_FLOAT16) {
        return 2U;
    }
    return 4U;
}

static uint32_t DimNum(const gert::Shape *shape) {
    uint32_t dimNum = static_cast<uint32_t>(shape->GetDimNum());
    return dimNum > ADDCMUL_MAX_DIMS ? ADDCMUL_MAX_DIMS : dimNum;
}

static void LoadShape(const gert::Shape *shape, uint32_t dims[ADDCMUL_MAX_DIMS], uint32_t &rank) {
    uint32_t dimNum = static_cast<uint32_t>(shape->GetDimNum());
    if (dimNum <= ADDCMUL_MAX_DIMS) {
        rank = dimNum;
        for (uint32_t i = 0; i < rank; ++i) {
            dims[i] = static_cast<uint32_t>(shape->GetDim(i));
        }
        return;
    }

    rank = ADDCMUL_MAX_DIMS;
    uint32_t prefix = 1;
    uint32_t prefixEnd = dimNum - (ADDCMUL_MAX_DIMS - 1);
    for (uint32_t i = 0; i < prefixEnd; ++i) {
        prefix *= static_cast<uint32_t>(shape->GetDim(i));
    }
    dims[0] = prefix;
    for (uint32_t i = 1; i < ADDCMUL_MAX_DIMS; ++i) {
        dims[i] = static_cast<uint32_t>(shape->GetDim(prefixEnd + i - 1));
    }
}

static uint32_t ShapeSize(const gert::Shape *shape) {
    int64_t size = shape->GetShapeSize();
    return size <= 0 ? 0U : static_cast<uint32_t>(size);
}

static bool BuildOutputShape(const gert::Shape *a, const gert::Shape *b, const gert::Shape *c,
    uint32_t outShape[ADDCMUL_MAX_DIMS], uint32_t &outRank) {
    uint32_t dimsA[ADDCMUL_MAX_DIMS] = {0};
    uint32_t dimsB[ADDCMUL_MAX_DIMS] = {0};
    uint32_t dimsC[ADDCMUL_MAX_DIMS] = {0};
    uint32_t rankA = 0;
    uint32_t rankB = 0;
    uint32_t rankC = 0;
    LoadShape(a, dimsA, rankA);
    LoadShape(b, dimsB, rankB);
    LoadShape(c, dimsC, rankC);
    outRank = rankA > rankB ? rankA : rankB;
    outRank = outRank > rankC ? outRank : rankC;
    if (outRank > ADDCMUL_MAX_DIMS) {
        return false;
    }

    for (uint32_t i = 0; i < outRank; ++i) {
        uint32_t dim = 1;
        const uint32_t *dims[3] = {dimsA, dimsB, dimsC};
        uint32_t ranks[3] = {rankA, rankB, rankC};
        for (uint32_t j = 0; j < 3; ++j) {
            uint32_t srcDim = 1;
            if (i + ranks[j] >= outRank) {
                uint32_t srcIndex = i + ranks[j] - outRank;
                srcDim = dims[j][srcIndex];
            }
            if (srcDim != 1 && dim != 1 && srcDim != dim) {
                return false;
            }
            if (srcDim != 1) {
                dim = srcDim;
            }
        }
        outShape[i] = dim;
    }
    return true;
}

static void BuildStrides(const gert::Shape *inputShape, const uint32_t outShape[ADDCMUL_MAX_DIMS],
    uint32_t outRank, uint32_t strides[ADDCMUL_MAX_DIMS]) {
    uint32_t inputDims[ADDCMUL_MAX_DIMS] = {0};
    uint32_t inRank = 0;
    LoadShape(inputShape, inputDims, inRank);
    uint32_t srcStrides[ADDCMUL_MAX_DIMS] = {0};
    uint32_t stride = 1;
    for (int32_t i = static_cast<int32_t>(inRank) - 1; i >= 0; --i) {
        srcStrides[i] = stride;
        stride *= inputDims[i];
    }

    for (uint32_t i = 0; i < outRank; ++i) {
        uint32_t inDim = 1;
        uint32_t srcStride = 0;
        if (i + inRank >= outRank) {
            uint32_t srcIndex = i + inRank - outRank;
            inDim = inputDims[srcIndex];
            srcStride = srcStrides[srcIndex];
        }
        strides[i] = (inDim == outShape[i]) ? srcStride : 0U;
    }
    for (uint32_t i = outRank; i < ADDCMUL_MAX_DIMS; ++i) {
        strides[i] = 0;
    }
}

static bool SameShape(const gert::Shape *a, const gert::Shape *b) {
    if (a->GetDimNum() != b->GetDimNum()) {
        return false;
    }
    uint32_t rankA = static_cast<uint32_t>(a->GetDimNum());
    uint32_t rankB = static_cast<uint32_t>(b->GetDimNum());
    if (rankA != rankB) {
        return false;
    }
    for (uint32_t i = 0; i < rankA; ++i) {
        if (a->GetDim(i) != b->GetDim(i)) {
            return false;
        }
    }
    return true;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (coreNum == 0) {
        coreNum = 1;
    }

    const gert::Shape *inputShape = &context->GetInputShape(0)->GetStorageShape();
    const gert::Shape *x1Shape = &context->GetInputShape(1)->GetStorageShape();
    const gert::Shape *x2Shape = &context->GetInputShape(2)->GetStorageShape();
    uint32_t outShape[ADDCMUL_MAX_DIMS] = {0};
    uint32_t outRank = 0;
    if (!BuildOutputShape(inputShape, x1Shape, x2Shape, outShape, outRank)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t inputNum = 1;
    for (uint32_t i = 0; i < outRank; ++i) {
        inputNum *= outShape[i];
    }

    ge::DataType dtype = context->GetRequiredInputDesc(0)->GetDataType();
    uint32_t typeLength = GetTypeLength(dtype);
    bool sameShape = SameShape(inputShape, x1Shape) && SameShape(inputShape, x2Shape) &&
        ShapeSize(inputShape) == inputNum;
    bool vectorCapable = sameShape &&
        (dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT || dtype == ge::DT_INT32);
    uint32_t inputBytes = inputNum * typeLength;
    uint32_t ceilBytes = ((inputBytes + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    uint32_t vectorBytes = vectorCapable ? (inputBytes / BLOCK_SIZE) * BLOCK_SIZE : 0U;
    uint32_t totalBlockNum = vectorCapable ? (vectorBytes / BLOCK_SIZE) : (ceilBytes / BLOCK_SIZE);
    if (vectorCapable && totalBlockNum == 0) {
        vectorCapable = false;
        totalBlockNum = ceilBytes / BLOCK_SIZE;
    }
    uint32_t blockDim = totalBlockNum == 0 ? 1U : (totalBlockNum < coreNum ? totalBlockNum : coreNum);
    context->SetBlockDim(blockDim);

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t tileBlockNum = static_cast<uint32_t>((ubSize / BLOCK_SIZE) / UB_BUFFER_PARTS);
    tileBlockNum = (tileBlockNum / TILE_BLOCK_ALIGN) * TILE_BLOCK_ALIGN;
    if (tileBlockNum == 0) {
        tileBlockNum = TILE_BLOCK_ALIGN;
    }

    uint32_t smallCoreBlockNum = totalBlockNum / blockDim;
    uint32_t tailBlockNum = totalBlockNum % blockDim;
    uint32_t bigCoreBlockNum = smallCoreBlockNum + 1;
    uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / typeLength;
    uint32_t smallCoreDataNum = (smallCoreBlockNum * BLOCK_SIZE) / typeLength;
    uint32_t bigCoreDataNum = (bigCoreBlockNum * BLOCK_SIZE) / typeLength;

    uint32_t smallTileNum = smallCoreBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum = (smallCoreBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint32_t smallTailDataNum = smallCoreDataNum - tileDataNum * smallTileNum;
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    uint32_t bigTileNum = bigCoreBlockNum / tileBlockNum;
    uint32_t finalBigTileNum = (bigCoreBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    AddcmulTilingData *tiling = context->GetTilingData<AddcmulTilingData>();
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->tailBlockNum = tailBlockNum;
    tiling->blockDim = blockDim;
    tiling->length = inputNum;
    tiling->vectorLength = vectorCapable ? (vectorBytes / typeLength) : 0U;
    tiling->sameShape = vectorCapable ? 1U : 0U;
    tiling->rank = outRank;
    for (uint32_t i = 0; i < ADDCMUL_MAX_DIMS; ++i) {
        tiling->shape[i] = i < outRank ? outShape[i] : 1U;
    }
    BuildStrides(inputShape, outShape, outRank, tiling->inputStrides);
    BuildStrides(x1Shape, outShape, outRank, tiling->x1Strides);
    BuildStrides(x2Shape, outShape, outRank, tiling->x2Strides);

    uint32_t dtypeInput = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, dtypeInput);
    context->GetRawTilingData()->SetDataSize(sizeof(AddcmulTilingData));
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *inputShape = context->GetInputShape(0);
    const gert::Shape *x1Shape = context->GetInputShape(1);
    const gert::Shape *x2Shape = context->GetInputShape(2);
    uint32_t outShapeData[ADDCMUL_MAX_DIMS] = {0};
    uint32_t outRank = 0;
    if (!BuildOutputShape(inputShape, x1Shape, x2Shape, outShapeData, outRank)) {
        return GRAPH_FAILED;
    }
    gert::Shape tmpShape;
    for (uint32_t i = 0; i < outRank; ++i) {
        tmpShape.AppendDim(outShapeData[i]);
    }
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = tmpShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Addcmul : public OpDef {
public:
    explicit Addcmul(const char *name) : OpDef(name) {
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
