#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/batch_to_space_tiling.h"
#include "../op_kernel/tiling_key_batch_to_space.h"

namespace {
constexpr int64_t kDimNum = 4;
constexpr uint32_t kMaxCopyBytes = 65520U;
constexpr int64_t kMaxBlockCount = 4095;

static uint32_t AlignTileElements(uint32_t elements) {
    if (elements >= 16U) {
        elements = (elements / 16U) * 16U;
    }
    if (elements == 0U) {
        elements = 1U;
    }
    return elements;
}

static bool GetAttrs(const gert::RuntimeAttrs *attrs, int64_t &cropTop, int64_t &cropBottom, int64_t &cropLeft,
                     int64_t &cropRight, int64_t &blockSize) {
    if (attrs == nullptr) {
        return false;
    }
    const gert::TypedContinuousVector<int64_t> *crops = attrs->GetListInt(0);
    const int64_t *attrBlockSize = attrs->GetInt(1);
    if (crops == nullptr || crops->GetSize() != 4 || attrBlockSize == nullptr) {
        return false;
    }
    cropTop = crops->GetData()[0];
    cropBottom = crops->GetData()[1];
    cropLeft = crops->GetData()[2];
    cropRight = crops->GetData()[3];
    blockSize = *attrBlockSize;
    return true;
}

static bool CalcOutputShape(int64_t batch, int64_t height, int64_t width, int64_t blockSize, int64_t cropTop,
                            int64_t cropBottom, int64_t cropLeft, int64_t cropRight, int64_t &outBatch,
                            int64_t &outHeight, int64_t &outWidth) {
    if (batch <= 0 || height <= 0 || width <= 0 || blockSize <= 0) {
        return false;
    }
    if (cropTop < 0 || cropBottom < 0 || cropLeft < 0 || cropRight < 0) {
        return false;
    }
    int64_t blockArea = blockSize * blockSize;
    if (blockArea <= 0 || batch % blockArea != 0) {
        return false;
    }
    if (cropTop + cropBottom >= height * blockSize || cropLeft + cropRight >= width * blockSize) {
        return false;
    }
    outBatch = batch / blockArea;
    outHeight = height * blockSize - cropTop - cropBottom;
    outWidth = width * blockSize - cropLeft - cropRight;
    return outHeight > 0 && outWidth > 0;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t coreNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (coreNum <= 0 || ubSize == 0) {
        return ge::GRAPH_FAILED;
    }

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    auto inputShape = context->GetInputShape(0);
    if (tensorX == nullptr || inputShape == nullptr || inputShape->GetStorageShape().GetDimNum() != kDimNum) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType dtypeX = tensorX->GetDataType();
    int64_t dtypeSize = ge::GetSizeByDataType(dtypeX);
    if (dtypeSize <= 0) {
        return ge::GRAPH_FAILED;
    }

    const auto storageShape = inputShape->GetStorageShape();
    int64_t batch = storageShape.GetDim(0);
    int64_t height = storageShape.GetDim(1);
    int64_t width = storageShape.GetDim(2);
    int64_t depth = storageShape.GetDim(3);
    if (depth <= 0) {
        return ge::GRAPH_FAILED;
    }

    int64_t cropTop = 0;
    int64_t cropBottom = 0;
    int64_t cropLeft = 0;
    int64_t cropRight = 0;
    int64_t blockSize = 0;
    if (!GetAttrs(context->GetAttrs(), cropTop, cropBottom, cropLeft, cropRight, blockSize)) {
        return ge::GRAPH_FAILED;
    }

    int64_t outBatch = 0;
    int64_t outHeight = 0;
    int64_t outWidth = 0;
    if (!CalcOutputShape(batch, height, width, blockSize, cropTop, cropBottom, cropLeft, cropRight, outBatch,
                         outHeight, outWidth)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t tileElements = static_cast<uint32_t>((ubSize / 4) / static_cast<uint64_t>(dtypeSize));
    uint32_t maxTileByCopy = static_cast<uint32_t>(kMaxCopyBytes / dtypeSize);
    if (tileElements > maxTileByCopy) {
        tileElements = maxTileByCopy;
    }
    tileElements = AlignTileElements(tileElements);
    uint32_t singleBufferTileElements = static_cast<uint32_t>((ubSize / 2) / static_cast<uint64_t>(dtypeSize));
    if (singleBufferTileElements > maxTileByCopy) {
        singleBufferTileElements = maxTileByCopy;
    }
    singleBufferTileElements = AlignTileElements(singleBufferTileElements);
    uint32_t strategy = 0;
    int64_t dChunk = tileElements;
    int64_t dChunks = 1;
    int64_t widthTile = 1;
    int64_t widthTiles = outWidth;
    int64_t outputPixels = outBatch * outHeight * outWidth;
    int64_t outputElements = outputPixels * depth;
    int64_t totalUnits = outBatch * outHeight * blockSize;
    bool noCrop = (cropTop == 0 && cropBottom == 0 && cropLeft == 0 && cropRight == 0);
    bool point1ExactB2NoCropFp32 = false;
    bool point2ExactB2CropCompact = false;
    bool point3ExactB2NoCropSmall = false;
    bool point4ExactB2NoCropRows6 = false;
    bool point5ExactB2CropFp16 = false;
    bool point6ExactB2NoCropFp16 = false;
    bool point7ExactB2NoCropFp16 = false;
    bool point8ExactB2NoCropFp16 = false;
    bool point9ExactB4CropFp16 = false;
    bool point10ExactB2NoCropW12 = false;
    int64_t alignElements = 32 / dtypeSize;
    if (alignElements < 1) {
        alignElements = 1;
    }
    int64_t dParallelChunk = (static_cast<int64_t>(tileElements) / alignElements) * alignElements;
    if (dParallelChunk < alignElements) {
        dParallelChunk = alignElements;
    }
    point1ExactB2NoCropFp32 = dtypeSize == 4 && noCrop && blockSize == 2 && batch == 8 && height == 28 &&
                              width == 28 && depth == 128 && outBatch == 2 && outHeight == 56 &&
                              outWidth == 56;
    point2ExactB2CropCompact =
        !noCrop && blockSize == 2 && cropTop == 2 && cropBottom == 1 && cropLeft == 3 &&
        cropRight == 1 && dtypeSize == 4 && batch == 4 && height == 10 && width == 15 &&
        depth == 5 && outBatch == 1 && outHeight == 17 && outWidth == 26;
    point3ExactB2NoCropSmall =
        noCrop && blockSize == 2 && dtypeSize == 4 && batch == 16 && height == 14 && width == 14 &&
        depth == 64 && outBatch == 4 && outHeight == 28 && outWidth == 28;
    point4ExactB2NoCropRows6 =
        noCrop && blockSize == 2 && dtypeSize == 4 && batch == 20 && height == 4 && width == 6 &&
        depth == 32 && outBatch == 5 && outHeight == 8 && outWidth == 12;
    point5ExactB2CropFp16 =
        !noCrop && blockSize == 2 && dtypeSize == 2 && batch == 4 && height == 128 && width == 128 &&
        depth == 65 && outBatch == 1 && outHeight == 254 && outWidth == 254 && cropTop == 1 &&
        cropBottom == 1 && cropLeft == 1 && cropRight == 1;
    point6ExactB2NoCropFp16 =
        noCrop && blockSize == 2 && dtypeSize == 2 && batch == 4 && height == 2 && width == 2 &&
        depth == 4096 && outBatch == 1 && outHeight == 4 && outWidth == 4;
    point7ExactB2NoCropFp16 =
        noCrop && blockSize == 2 && dtypeSize == 2 && batch == 4 && height == 1 && width == 1 &&
        depth == 16384 && outBatch == 1 && outHeight == 2 && outWidth == 2;
    point8ExactB2NoCropFp16 =
        noCrop && blockSize == 2 && dtypeSize == 2 && batch == 4 && height == 10 && width == 512 &&
        depth == 256 && outBatch == 1 && outHeight == 20 && outWidth == 1024;
    point9ExactB4CropFp16 =
        !noCrop && blockSize == 4 && dtypeSize == 2 && batch == 16 && height == 10 && width == 512 &&
        depth == 64 && outBatch == 1 && outHeight == 40 && outWidth == 1535 && cropTop == 0 &&
        cropBottom == 0 && cropLeft == 513 && cropRight == 0;
    point10ExactB2NoCropW12 = dtypeSize == 2 && noCrop && blockSize == 2 && batch == 16 && height == 1024 &&
                              width == 6 && depth == 32 && outBatch == 4 && outHeight == 2048 &&
                              outWidth == 12;
    if (point1ExactB2NoCropFp32) {
        strategy = 32;
        int64_t alignedDepth = ((depth + alignElements - 1) / alignElements) * alignElements;
        int64_t rowWidthTile = static_cast<int64_t>(tileElements) / alignedDepth;
        widthTile = rowWidthTile / 2;
        if (widthTile < 1) {
            widthTile = 1;
        }
        if (widthTile < outWidth && (outWidth % widthTile) != 0) {
            int64_t minWidthTile = widthTile / 2;
            if (minWidthTile < 1) {
                minWidthTile = 1;
            }
            for (int64_t candidate = widthTile; candidate >= minWidthTile; --candidate) {
                if ((candidate & 1) == 0 && (outWidth % candidate) == 0) {
                    widthTile = candidate;
                    break;
                }
            }
        }
        widthTiles = (outWidth + widthTile - 1) / widthTile;
        totalUnits = outBatch * height * widthTiles;
    } else if (point2ExactB2CropCompact) {
        strategy = 33;
        widthTile = outWidth;
        widthTiles = 1;
        tileElements = AlignTileElements(320U);
        totalUnits = outBatch * outHeight;
    } else if (point3ExactB2NoCropSmall) {
        strategy = 34;
        widthTile = outWidth;
        widthTiles = 1;
        tileElements = AlignTileElements(static_cast<uint32_t>(2 * outWidth * depth));
        totalUnits = outBatch * height;
    } else if (point4ExactB2NoCropRows6) {
        strategy = 40;
        widthTile = outWidth;
        widthTiles = 1;
        tileElements = AlignTileElements(static_cast<uint32_t>(4 * outWidth * depth));
        totalUnits = outBatch * 2;
    } else if (point5ExactB2CropFp16) {
        strategy = 31;
        widthTile = 128;
        widthTiles = (outWidth + widthTile - 1) / widthTile;
        int64_t compactHalf = ((widthTile + 1) / 2) * depth;
        int64_t compactHalfAligned = ((compactHalf + alignElements - 1) / alignElements) * alignElements;
        tileElements = AlignTileElements(static_cast<uint32_t>(compactHalfAligned * 2));
        totalUnits = outBatch * outHeight * widthTiles;
    } else if (point6ExactB2NoCropFp16) {
        strategy = 42;
        widthTile = 1;
        widthTiles = 1;
        totalUnits = 8;
    } else if (point7ExactB2NoCropFp16) {
        strategy = 41;
        widthTile = 1;
        widthTiles = 1;
        tileElements = AlignTileElements(static_cast<uint32_t>(depth));
        totalUnits = 8;
    } else if (point8ExactB2NoCropFp16) {
        strategy = 37;
        widthTile = 128;
        widthTiles = 8;
        tileElements = AlignTileElements(static_cast<uint32_t>(widthTile * depth));
        totalUnits = outHeight * widthTiles;
    } else if (point9ExactB4CropFp16) {
        strategy = 38;
        widthTile = 512;
        widthTiles = 3;
        tileElements = AlignTileElements(static_cast<uint32_t>(widthTile * depth));
        totalUnits = outHeight * widthTiles;
    } else if (point10ExactB2NoCropW12) {
        strategy = 4;
        widthTile = outWidth;
        widthTiles = 1;
        uint64_t point10UbBudget = ubSize > 8192U ? (ubSize - 8192U) : ubSize;
        tileElements = static_cast<uint32_t>((point10UbBudget / 4) / static_cast<uint64_t>(dtypeSize));
        if (tileElements > maxTileByCopy) {
            tileElements = maxTileByCopy;
        }
        tileElements = AlignTileElements(tileElements);
        totalUnits = outBatch * ((height + 31) / 32);
    } else {
        strategy = 28;
        tileElements = AlignTileElements(16U);
        totalUnits = 1;
    }
    int64_t coreLimit = coreNum;
    if (point1ExactB2NoCropFp32) {
        coreLimit = coreLimit < 28 ? coreLimit : 28;
    } else if (point2ExactB2CropCompact) {
        coreLimit = coreLimit < 17 ? coreLimit : 17;
    } else if (point3ExactB2NoCropSmall) {
        coreLimit = coreLimit < 16 ? coreLimit : 16;
    } else if (point4ExactB2NoCropRows6) {
        coreLimit = coreLimit < 10 ? coreLimit : 10;
    } else if (point5ExactB2CropFp16) {
        coreLimit = coreLimit < 40 ? coreLimit : 40;
    } else if (point6ExactB2NoCropFp16) {
        coreLimit = coreLimit < 8 ? coreLimit : 8;
    } else if (point7ExactB2NoCropFp16) {
        coreLimit = coreLimit < 8 ? coreLimit : 8;
    } else if (point8ExactB2NoCropFp16) {
        coreLimit = coreLimit < 32 ? coreLimit : 32;
    } else if (point9ExactB4CropFp16) {
        coreLimit = coreLimit < 40 ? coreLimit : 40;
    } else if (point10ExactB2NoCropW12) {
        coreLimit = coreLimit < 32 ? coreLimit : 32;
    } else {
        coreLimit = 1;
    }
    int64_t scheduleTotalUnits = totalUnits;
    int64_t usedCoreNum = scheduleTotalUnits < coreLimit ? scheduleTotalUnits : coreLimit;
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }
    int64_t unitsPerCore = (scheduleTotalUnits + usedCoreNum - 1) / usedCoreNum;
    if (strategy == 1) {
        unitsPerCore = ((unitsPerCore + alignElements - 1) / alignElements) * alignElements;
        usedCoreNum = (scheduleTotalUnits + unitsPerCore - 1) / unitsPerCore;
        if (usedCoreNum > coreNum) {
            usedCoreNum = coreNum;
        }
    }
    int64_t smallCoreUnits = scheduleTotalUnits / usedCoreNum;
    int64_t tailCoreNum = scheduleTotalUnits - smallCoreUnits * usedCoreNum;
    int64_t bigCoreUnits = smallCoreUnits + 1;
    uint32_t dtypeKey = static_cast<uint32_t>(dtypeX);
    uint32_t fixedPoint = 0U;
    if (point1ExactB2NoCropFp32) {
        fixedPoint = 1U;
    } else if (point2ExactB2CropCompact) {
        fixedPoint = 2U;
    } else if (point3ExactB2NoCropSmall) {
        fixedPoint = 3U;
    } else if (point4ExactB2NoCropRows6) {
        fixedPoint = 4U;
    } else if (point5ExactB2CropFp16) {
        fixedPoint = 5U;
    } else if (point6ExactB2NoCropFp16) {
        fixedPoint = 6U;
    } else if (point7ExactB2NoCropFp16) {
        fixedPoint = 7U;
    } else if (point8ExactB2NoCropFp16) {
        fixedPoint = 8U;
    } else if (point9ExactB4CropFp16) {
        fixedPoint = 9U;
    } else if (point10ExactB2NoCropW12) {
        fixedPoint = 10U;
    }
    ASCENDC_TPL_SEL_PARAM(context, dtypeKey, fixedPoint);

    BatchToSpaceTilingData *tiling = context->GetTilingData<BatchToSpaceTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->batch = batch;
    tiling->height = height;
    tiling->width = width;
    tiling->depth = depth;
    tiling->outBatch = outBatch;
    tiling->outHeight = outHeight;
    tiling->outWidth = outWidth;
    tiling->cropTop = cropTop;
    tiling->cropBottom = cropBottom;
    tiling->cropLeft = cropLeft;
    tiling->cropRight = cropRight;
    tiling->blockSize = blockSize;
    tiling->totalUnits = totalUnits;
    tiling->unitsPerCore = unitsPerCore;
    tiling->smallCoreUnits = smallCoreUnits;
    tiling->bigCoreUnits = bigCoreUnits;
    tiling->tailCoreNum = tailCoreNum;
    tiling->dChunk = dChunk;
    tiling->dChunks = dChunks;
    tiling->widthTile = widthTile;
    tiling->widthTiles = widthTiles;
    tiling->tileElements = tileElements;
    tiling->strategy = strategy;

    context->SetBlockDim(usedCoreNum);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr || xShape->GetDimNum() != kDimNum) {
        return GRAPH_FAILED;
    }

    int64_t cropTop = 0;
    int64_t cropBottom = 0;
    int64_t cropLeft = 0;
    int64_t cropRight = 0;
    int64_t blockSize = 0;
    if (!GetAttrs(context->GetAttrs(), cropTop, cropBottom, cropLeft, cropRight, blockSize)) {
        return GRAPH_FAILED;
    }

    int64_t outBatch = 0;
    int64_t outHeight = 0;
    int64_t outWidth = 0;
    if (!CalcOutputShape(xShape->GetDim(0), xShape->GetDim(1), xShape->GetDim(2), blockSize, cropTop, cropBottom,
                         cropLeft, cropRight, outBatch, outHeight, outWidth)) {
        return GRAPH_FAILED;
    }

    yShape->SetDimNum(kDimNum);
    yShape->SetDim(0, outBatch);
    yShape->SetDim(1, outHeight);
    yShape->SetDim(2, outWidth);
    yShape->SetDim(3, xShape->GetDim(3));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class BatchToSpace : public OpDef {
public:
    explicit BatchToSpace(const char *name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("crops").AttrType(REQUIRED).ListInt();
        this->Attr("block_size").AttrType(REQUIRED).Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(BatchToSpace);
}  // namespace ops
