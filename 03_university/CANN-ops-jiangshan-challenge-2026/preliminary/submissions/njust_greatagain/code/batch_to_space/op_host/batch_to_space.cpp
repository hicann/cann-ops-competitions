// Champion-style host tiling for BatchToSpace.

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/batch_to_space_tiling.h"
#include "../op_kernel/tiling_key_batch_to_space.h"

namespace {
constexpr uint32_t RESERVED_UB_BYTES = 8192U;
constexpr uint32_t MIN_TILE_LENGTH = 8U;

static uint32_t DimToU32(int64_t dim)
{
    return dim > 0 ? static_cast<uint32_t>(dim) : 0U;
}

static uint32_t AttrToU32(int64_t value)
{
    return value > 0 ? static_cast<uint32_t>(value) : 0U;
}

static uint32_t AlignDown(uint64_t value, uint32_t align)
{
    return align == 0 ? static_cast<uint32_t>(value)
                      : static_cast<uint32_t>(value & ~static_cast<uint64_t>(align - 1U));
}

static uint32_t AlignUp(uint64_t value, uint32_t align)
{
    if (align == 0) {
        return static_cast<uint32_t>(value);
    }
    return static_cast<uint32_t>((value + align - 1U) & ~static_cast<uint64_t>(align - 1U));
}

static uint32_t CalcTileLength(uint64_t ubSizeBytes, uint32_t depth, uint32_t dtypeBytes, uint32_t elementsPerBlock)
{
    uint64_t usable = ubSizeBytes > RESERVED_UB_BYTES ? ubSizeBytes - RESERVED_UB_BYTES : 0ULL;
    // Double path uses VECIN x2 + VECOUT x2. Keep the same conservative host-side sizing style as the champion code.
    uint64_t raw = dtypeBytes == 0 ? 0ULL : usable / (4ULL * dtypeBytes);
    uint32_t maxTile = AlignDown(raw, elementsPerBlock);
    if (maxTile < MIN_TILE_LENGTH) {
        maxTile = MIN_TILE_LENGTH;
    }

    if (depth <= maxTile) {
        return depth == 0 ? MIN_TILE_LENGTH : AlignUp(depth, elementsPerBlock);
    }
    return maxTile;
}

static uint32_t CalcBlockDim(uint32_t pixelCount, uint32_t maxCoreNum)
{
    if (pixelCount >= 13376U && pixelCount < 40940U) {
        return maxCoreNum < 24U ? maxCoreNum : 24U;
    }
    // Keep the champion-style presplit architecture, but restore the
    // BatchToSpace-specific bd route proven by 006c. The Erf champion
    // thresholds under-parallelize pixel counts in (16, 32768].
    uint32_t target = pixelCount == 0 ? 1U : (pixelCount <= 16U ? 8U : 40U);
    if (target > maxCoreNum) {
        target = maxCoreNum;
    }
    return target == 0 ? 1U : target;
}

static uint32_t CalcRowTilePixels(
    uint64_t ubSizeBytes,
    uint32_t outWidth,
    uint32_t depth,
    uint32_t dtypeBytes,
    uint32_t elementsPerBlock)
{
    if (outWidth == 0 || depth == 0 || dtypeBytes == 0) {
        return 1U;
    }
    uint64_t usable = ubSizeBytes > RESERVED_UB_BYTES ? ubSizeBytes - RESERVED_UB_BYTES : 0ULL;
    uint64_t maxElems = usable / (2ULL * dtypeBytes);
    uint32_t maxPixels = static_cast<uint32_t>(maxElems / depth);
    if (maxPixels == 0) {
        maxPixels = 1U;
    }
    if (maxPixels > outWidth) {
        maxPixels = outWidth;
    }
    uint64_t rowTileElems = static_cast<uint64_t>(maxPixels) * depth;
    while (maxPixels > 1U && ((rowTileElems * dtypeBytes * 2ULL + RESERVED_UB_BYTES) > ubSizeBytes)) {
        --maxPixels;
        rowTileElems = static_cast<uint64_t>(maxPixels) * depth;
    }
    (void)elementsPerBlock;
    return maxPixels == 0 ? 1U : maxPixels;
}

static uint32_t CalcNonAlignedRowTilePixels(
    uint64_t ubSizeBytes,
    uint32_t outWidth,
    uint32_t depth,
    uint32_t dtypeBytes,
    uint32_t elementsPerBlock)
{
    if (outWidth == 0 || depth == 0 || dtypeBytes == 0 || elementsPerBlock == 0) {
        return 0U;
    }
    uint32_t segmentStride = AlignUp(depth, elementsPerBlock);
    uint32_t maxPixels = outWidth;
    while (maxPixels >= 2U) {
        uint64_t inputPhysicalElems = static_cast<uint64_t>(maxPixels) * segmentStride;
        uint64_t outputTightElems = static_cast<uint64_t>(maxPixels) * depth;
        uint64_t outputPhysicalElems = static_cast<uint64_t>(AlignUp(outputTightElems, elementsPerBlock));
        uint64_t bytes = (inputPhysicalElems + outputPhysicalElems) * dtypeBytes + RESERVED_UB_BYTES;
        if (bytes <= ubSizeBytes) {
            return maxPixels;
        }
        --maxPixels;
    }
    return 0U;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aicNum = static_cast<uint32_t>(platform.GetCoreNumAic());
    uint32_t aivNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
    uint32_t vectorCoreNum = aivNum > 0 ? aivNum : aicNum;
    if (vectorCoreNum == 0) {
        vectorCoreNum = 1U;
    }

    uint64_t ubSizeBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeBytes);

    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x = tensor_x->GetDataType();
    const gert::Shape &shape_x = tensor_x->GetOriginShape();

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *crops = attrs->GetListInt(0)->GetData();
    int64_t block_size_attr = *attrs->GetInt(1);

    uint32_t batch = DimToU32(shape_x.GetDim(0));
    uint32_t height = DimToU32(shape_x.GetDim(1));
    uint32_t width = DimToU32(shape_x.GetDim(2));
    uint32_t depth = DimToU32(shape_x.GetDim(3));
    uint32_t crop_top = AttrToU32(crops[0]);
    uint32_t crop_bottom = AttrToU32(crops[1]);
    uint32_t crop_left = AttrToU32(crops[2]);
    uint32_t crop_right = AttrToU32(crops[3]);
    uint32_t block_size = AttrToU32(block_size_attr);
    uint32_t block_area = block_size * block_size;
    uint32_t out_batch = block_area == 0 ? 0U : batch / block_area;
    uint32_t out_height = height * block_size - crop_top - crop_bottom;
    uint32_t out_width = width * block_size - crop_left - crop_right;
    uint32_t input_length = batch * height * width * depth;
    uint32_t out_length = out_batch * out_height * out_width * depth;
    uint32_t pixel_count = out_batch * out_height * out_width;

    uint32_t dtypeBytes = dtype_x == ge::DT_FLOAT ? 4U : 2U;
    uint32_t elementsPerBlock = 32U / dtypeBytes;
    uint32_t tile_length = CalcTileLength(ubSizeBytes, depth, dtypeBytes, elementsPerBlock);
    uint32_t usedCoreNum = CalcBlockDim(pixel_count, vectorCoreNum);
    if (pixel_count >= 229U && pixel_count < 461U) {
        usedCoreNum = vectorCoreNum < 16U ? vectorCoreNum : 16U;
        if (usedCoreNum == 0U) {
            usedCoreNum = 1U;
        }
    }
    if (pixel_count >= 4704U && pixel_count < 13376U) {
        usedCoreNum = vectorCoreNum < 16U ? vectorCoreNum : 16U;
        if (usedCoreNum == 0U) {
            usedCoreNum = 1U;
        }
    }
    if (pixel_count >= 1808U && pixel_count < 4704U) {
        usedCoreNum = vectorCoreNum < 20U ? vectorCoreNum : 20U;
        if (usedCoreNum == 0U) {
            usedCoreNum = 1U;
        }
    }
    if (pixel_count >= 461U && pixel_count < 1808U) {
        usedCoreNum = vectorCoreNum < 10U ? vectorCoreNum : 10U;
        if (usedCoreNum == 0U) {
            usedCoreNum = 1U;
        }
    }
    if (pixel_count >= 81410U) {
        usedCoreNum = vectorCoreNum < 26U ? vectorCoreNum : 26U;
        if (usedCoreNum == 0U) {
            usedCoreNum = 1U;
        }
    }

    uint32_t isSingleTile = depth <= tile_length ? 1U : 0U;
    uint32_t depthAligned = (elementsPerBlock != 0 && (depth % elementsPerBlock) == 0) ? 1U : 0U;
    uint32_t routeMultiRowStrip = (pixel_count >= 81410U) ? 1U : 0U;
    uint32_t routeRowTileCompose =
        ((pixel_count >= 461U && pixel_count < 62958U))
            ? 1U
            : 0U;
    uint32_t row_count = out_batch * out_height;
    uint32_t row_tile_pixels = CalcRowTilePixels(ubSizeBytes, out_width, depth, dtypeBytes, elementsPerBlock);
    if (pixel_count >= 13376U && pixel_count < 40940U && row_tile_pixels > 172U) {
        row_tile_pixels = 172U;
    }
    uint64_t row_tile_elems = static_cast<uint64_t>(row_tile_pixels) * depth;
    uint64_t row_tile_bytes = row_tile_elems * dtypeBytes * 2ULL + RESERVED_UB_BYTES;
    uint32_t rowTileComposeSafe =
        (depthAligned != 0 && row_tile_pixels > 0U && depth > 0U && row_tile_bytes <= ubSizeBytes) ? 1U : 0U;
    uint32_t enable_row_tile_compose =
        (routeRowTileCompose != 0U && rowTileComposeSafe != 0U) ? 1U : 0U;
    if (enable_row_tile_compose == 0U) {
        row_tile_pixels = 1U;
    }
    uint32_t usedLargeCoreNum = usedCoreNum;
    uint32_t smallCoreRowNum = usedLargeCoreNum == 0 ? 0U : row_count / usedLargeCoreNum;
    uint32_t tailRowBlockNum = usedLargeCoreNum == 0 ? 0U : row_count % usedLargeCoreNum;
    uint32_t bigCoreRowNum = smallCoreRowNum + (tailRowBlockNum > 0 ? 1U : 0U);
    uint64_t row_elems_u64 = static_cast<uint64_t>(out_width) * depth;
    uint64_t usable = ubSizeBytes > RESERVED_UB_BYTES ? ubSizeBytes - RESERVED_UB_BYTES : 0ULL;
    uint64_t maxStripRowsU64 =
        (row_elems_u64 == 0 || dtypeBytes == 0) ? 0ULL : usable / (2ULL * row_elems_u64 * dtypeBytes);
    uint32_t maxStripRows = maxStripRowsU64 > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(maxStripRowsU64);
    uint32_t stripRows = out_height < maxStripRows ? out_height : maxStripRows;
    uint64_t strip_elems_u64 = static_cast<uint64_t>(stripRows) * row_elems_u64;
    uint64_t strip_bytes = strip_elems_u64 * dtypeBytes * 2ULL + RESERVED_UB_BYTES;
    uint32_t multiRowStripSafe =
        (depthAligned != 0 && row_elems_u64 > 0ULL && stripRows >= 2U && strip_bytes <= ubSizeBytes) ? 1U : 0U;
    uint32_t enableMultiRowStrip =
        (routeMultiRowStrip != 0U && multiRowStripSafe != 0U) ? 1U : 0U;
    if (enableMultiRowStrip == 0U) {
        stripRows = 1U;
        strip_elems_u64 = row_elems_u64;
    }
    uint32_t stripElems = strip_elems_u64 > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(strip_elems_u64);

    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X, isSingleTile);

    BatchToSpaceTilingData *tiling = context->GetTilingData<BatchToSpaceTilingData>();
    tiling->height = height;
    tiling->width = width;
    tiling->depth = depth;
    tiling->outBatch = out_batch;
    tiling->outHeight = out_height;
    tiling->outWidth = out_width;
    tiling->cropTop = crop_top;
    tiling->cropBottom = crop_bottom;
    tiling->cropLeft = crop_left;
    tiling->cropRight = crop_right;
    tiling->blockSize = block_size;
    tiling->inputLength = input_length;
    tiling->outLength = out_length;
    tiling->pixelCount = pixel_count;
    tiling->tileLength = tile_length;
    tiling->rowCount = row_count;
    tiling->rowTilePixels = row_tile_pixels;
    tiling->enableRowTileCompose = enable_row_tile_compose;
    tiling->usedLargeCoreNum = usedLargeCoreNum;
    tiling->smallCoreRowNum = smallCoreRowNum;
    tiling->bigCoreRowNum = bigCoreRowNum;
    tiling->tailRowBlockNum = tailRowBlockNum;
    tiling->stripRows = stripRows;
    tiling->stripElems = stripElems;

    context->SetBlockDim(usedCoreNum);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *crops = attrs->GetListInt(0)->GetData();
    int64_t block_size = *attrs->GetInt(1);
    int64_t block_area = block_size * block_size;

    y_shape->SetDimNum(4);
    int64_t batch = x_shape->GetDim(0);
    int64_t height = x_shape->GetDim(1);
    int64_t width = x_shape->GetDim(2);
    int64_t depth = x_shape->GetDim(3);

    y_shape->SetDim(0, (batch > 0 && block_area > 0) ? batch / block_area : -1);
    y_shape->SetDim(1, height > 0 ? height * block_size - crops[0] - crops[1] : -1);
    y_shape->SetDim(2, width > 0 ? width * block_size - crops[2] - crops[3] : -1);
    y_shape->SetDim(3, depth);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class BatchToSpace : public OpDef {
public:
    explicit BatchToSpace(const char *name) : OpDef(name)
    {
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
