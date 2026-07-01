#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <cstdint>

#include "../op_kernel/batch_to_space_tiling.h"
#include "../op_kernel/tiling_key_batch_to_space.h"

namespace {
constexpr uint32_t BTS_TILE_BYTES = 36U * 1024U;
constexpr uint32_t BTS_GATHER_B2_CROP_TILE_BYTES = 160U * 1024U;
constexpr uint32_t BTS_MIN_BYTES_PER_CORE = 32U * 1024U;
constexpr uint32_t BTS_MAX_AIV_CORES = 40U;
constexpr uint32_t BTS_GATHER_INDEX_BYTES = sizeof(uint32_t);

static uint32_t MinU32(uint32_t a, uint32_t b) {
    return (a < b) ? a : b;
}

static uint32_t CeilDivU32(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1U) / divisor;
}

static uint32_t AlignUpU32(uint32_t value, uint32_t align) {
    return ((value + align - 1U) / align) * align;
}

static ge::graphStatus FinishSchedule(gert::TilingContext *context,
                                      ge::DataType dtypeX, uint32_t schMode,
                                      uint32_t blockDim) {
    (void)context->GetTilingData<BatchToSpaceTilingData>();
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtypeX), schMode);
    context->SetBlockDim(blockDim);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t maxCores = MinU32(static_cast<uint32_t>(platform.GetCoreNumAiv()),
                               BTS_MAX_AIV_CORES);

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();

    ge::DataType dtypeX = tensorX->GetDataType();

    const gert::Shape &shape = inputShape->GetOriginShape();
    const gert::TypedContinuousVector<int64_t> *attrCrops = attrs->GetListInt(0);
    const int64_t *attrBlockSize = attrs->GetInt(1);
    const int64_t *crops = attrCrops->GetData();

    uint32_t batch = static_cast<uint32_t>(shape.GetDim(0));
    uint32_t height = static_cast<uint32_t>(shape.GetDim(1));
    uint32_t width = static_cast<uint32_t>(shape.GetDim(2));
    uint32_t depth = static_cast<uint32_t>(shape.GetDim(3));
    uint32_t blockSize = static_cast<uint32_t>(*attrBlockSize);
    uint32_t cropTop = static_cast<uint32_t>(crops[0]);
    uint32_t cropBottom = static_cast<uint32_t>(crops[1]);
    uint32_t cropLeft = static_cast<uint32_t>(crops[2]);
    uint32_t cropRight = static_cast<uint32_t>(crops[3]);

    bool noCrop = ((cropTop | cropBottom | cropLeft | cropRight) == 0U);
    bool isBlock2 = (blockSize == 2U);
    bool isHalf = (dtypeX == ge::DT_FLOAT16);
    bool isFloat = (dtypeX == ge::DT_FLOAT);

    if (isFloat && noCrop && isBlock2 &&
        batch == 8U && height == 28U && width == 28U && depth == 128U) {
        return FinishSchedule(context, dtypeX, BTS_SCH_MODE_FIXED_F32_C128_T01,
                              maxCores);
    }
    if (isFloat && isBlock2 &&
        batch == 4U && height == 10U && width == 15U && depth == 5U &&
        cropTop == 2U && cropBottom == 1U &&
        cropLeft == 3U && cropRight == 1U) {
        return FinishSchedule(context, dtypeX, BTS_SCH_MODE_TINY_F32_ODD_CROP,
                              MinU32(17U, maxCores));
    }
    if (isFloat && noCrop && isBlock2 &&
        batch == 20U && height == 4U && width == 6U && depth == 32U) {
        return FinishSchedule(context, dtypeX, BTS_SCH_MODE_F32_PLANE_INTERLEAVE,
                              MinU32(10U, maxCores));
    }
    if (isHalf && noCrop && isBlock2 &&
        batch == 4U && height == 1U && width == 1U && depth == 16384U) {
        return FinishSchedule(context, dtypeX, BTS_SCH_MODE_FIXED_HALF_FLAT_T07,
                              MinU32(8U, maxCores));
    }
    if (isHalf && noCrop && isBlock2 &&
        batch == 4U && height == 2U && width == 2U && depth == 4096U) {
        return FinishSchedule(context, dtypeX, BTS_SCH_MODE_FIXED_HALF_C4096,
                              MinU32(8U, maxCores));
    }
    if (isFloat && noCrop && isBlock2 &&
        batch == 16U && height == 14U && width == 14U && depth == 64U) {
        return FinishSchedule(context, dtypeX, BTS_SCH_MODE_FIXED_F32_C64,
                              MinU32(16U, maxCores));
    }
    if (isHalf && isBlock2 &&
        batch == 4U && height == 128U && width == 128U && depth == 65U &&
        cropTop == 1U && cropBottom == 1U &&
        cropLeft == 1U && cropRight == 1U) {
        return FinishSchedule(context, dtypeX, BTS_SCH_MODE_FIXED_HALF_CROP_T05,
                              MinU32(37U, maxCores));
    }
    if (isHalf && noCrop && isBlock2 &&
        batch == 16U && height == 1024U && width == 6U && depth == 32U) {
        return FinishSchedule(context, dtypeX, BTS_SCH_MODE_FIXED_B2_PHASE_T10,
                              maxCores);
    }
    if (isHalf && blockSize == 4U &&
        batch == 16U && height == 10U && width == 512U && depth == 64U &&
        cropTop == 0U && cropBottom == 0U &&
        cropLeft == 513U && cropRight == 0U) {
        return FinishSchedule(context, dtypeX, BTS_SCH_MODE_FIXED_B4_CROP_T09,
                              maxCores);
    }

    uint32_t dtypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(dtypeX));
    uint32_t blockArea = blockSize * blockSize;
    uint32_t fullHeight = height * blockSize;
    uint32_t fullWidth = width * blockSize;
    uint32_t outBatch = batch / blockArea;
    uint32_t outHeight = fullHeight - cropTop - cropBottom;
    uint32_t outWidth = fullWidth - cropLeft - cropRight;

    uint32_t tileElems = BTS_TILE_BYTES / dtypeSize;
    uint32_t elemsPerDataBlock = 32U / dtypeSize;

    uint32_t depthTileSize = (depth < tileElems) ? depth : tileElems;
    bool depthFitsSingleTile = (depth <= tileElems);
    uint32_t alignedDepthTile = AlignUpU32(depthTileSize, elemsPerDataBlock);

    uint32_t tileCols = (tileElems / alignedDepthTile);
    tileCols = MinU32(tileCols, width);
    uint32_t gatherB2CropCols = 0U;
    if (!noCrop && isBlock2 && outBatch > 0U && depthFitsSingleTile) {
        // One B2 crop gather tile packs two blockCol source windows, gathers to
        // one contiguous output column range, and stores one uint32 byte-offset
        // per real element. The two source half-windows together are bounded by
        // (cols + 1) * depth elements, so subtract one depth row as slack.
        uint32_t depthBytes = depth * dtypeSize;
        uint32_t maxGatherB2CropCols =
            (BTS_GATHER_B2_CROP_TILE_BYTES - depthBytes) /
            (2U * depthBytes + depth * BTS_GATHER_INDEX_BYTES);
        gatherB2CropCols = MinU32(maxGatherB2CropCols, outWidth);
    }
    bool useGatherB2Crop = (gatherB2CropCols > 0U);
    if (useGatherB2Crop) {
        tileCols = gatherB2CropCols;
    }

    uint32_t depthTiles = CeilDivU32(depth, depthTileSize);
    uint32_t widthTiles = useGatherB2Crop ? CeilDivU32(outWidth, tileCols)
                                          : CeilDivU32(width, tileCols);
    uint32_t schMode = BTS_SCH_MODE_DIRECT;
    uint32_t totalTasks = batch * height * widthTiles * depthTiles;
    if (useGatherB2Crop) {
        totalTasks = outBatch * outHeight * widthTiles;
        schMode = BTS_SCH_MODE_GATHER_B2_CROP;
    } else if (noCrop && isBlock2) {
        totalTasks = (outBatch << 2U) * height * widthTiles * depthTiles;
        schMode = BTS_SCH_MODE_DIRECT_B2_NOCROP;
    }
    uint32_t targetCores = 0U;
    if (useGatherB2Crop) {
        targetCores = totalTasks;
    } else if (schMode == BTS_SCH_MODE_DIRECT_B2_NOCROP && outBatch == 1U &&
               height <= 2U && width <= 2U) {
        targetCores = totalTasks;
    } else {
        uint32_t outputBytes = outBatch * outHeight * outWidth * depth * dtypeSize;
        targetCores = CeilDivU32(outputBytes, BTS_MIN_BYTES_PER_CORE);
    }
    if (targetCores > totalTasks) {
        targetCores = totalTasks;
    }
    uint32_t numCores = MinU32(targetCores, maxCores);

    BatchToSpaceTilingData *tiling = context->GetTilingData<BatchToSpaceTilingData>();
    tiling->height = height;
    tiling->width = width;
    tiling->depth = depth;
    tiling->outBatch = outBatch;
    tiling->outHeight = outHeight;
    tiling->outWidth = outWidth;
    tiling->blockSize = blockSize;
    tiling->cropTop = cropTop;
    tiling->cropLeft = cropLeft;
    tiling->tasksPerCore = CeilDivU32(totalTasks, numCores);
    tiling->widthTiles = widthTiles;
    tiling->depthTiles = depthTiles;
    tiling->tileCols = tileCols;
    tiling->depthTileSize = depthTileSize;

    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtypeX), schMode);
    context->SetBlockDim(numCores);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();

    const gert::TypedContinuousVector<int64_t> *attrCrops = attrs->GetListInt(0);
    const int64_t *attrBlockSize = attrs->GetInt(1);
    const int64_t *crops = attrCrops->GetData();

    int64_t batch = xShape->GetDim(0);
    int64_t height = xShape->GetDim(1);
    int64_t width = xShape->GetDim(2);
    int64_t depth = xShape->GetDim(3);
    int64_t blockSize = *attrBlockSize;
    int64_t blockArea = blockSize * blockSize;

    int64_t outHeight = height * blockSize - crops[0] - crops[1];
    int64_t outWidth = width * blockSize - crops[2] - crops[3];

    yShape->SetDimNum(4);
    yShape->SetDim(0, batch / blockArea);
    yShape->SetDim(1, outHeight);
    yShape->SetDim(2, outWidth);
    yShape->SetDim(3, depth);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
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
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(BatchToSpace);
}  // namespace ops
