#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/batch_to_space_tiling.h"
#include "../op_kernel/tiling_key_batch_to_space.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    constexpr uint64_t kMinParallelOutputElements = 64;
    constexpr uint64_t kMinParallelRowWidth = 8;
    constexpr uint32_t kAlign32 = 32;
    constexpr uint32_t kUbReserveBytes = 1024;
    constexpr uint32_t kCacheLineSize = 32;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t numCoresAiv = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    if (tensorX == nullptr) return ge::GRAPH_FAILED;

    ge::DataType dtypeX = tensorX->GetDataType();
    uint32_t elemSizeBytes = (dtypeX == ge::DT_FLOAT16) ? 2 : 4;
    uint32_t alignElems = kAlign32 / elemSizeBytes;
    auto xShape = context->GetInputShape(0)->GetOriginShape();
    if (xShape.GetDimNum() != 4) return ge::GRAPH_FAILED;

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) return ge::GRAPH_FAILED;

    const gert::TypedContinuousVector<int64_t> *attrCrops = attrs->GetListInt(0);
    const int64_t *attrBlockSize = attrs->GetInt(1);
    if (attrCrops == nullptr || attrBlockSize == nullptr || attrCrops->GetSize() != 4 || *attrBlockSize <= 0)
        return ge::GRAPH_FAILED;

    const int64_t batch = xShape.GetDim(0);
    const int64_t inputHeight = xShape.GetDim(1);
    const int64_t inputWidth = xShape.GetDim(2);
    const int64_t depth = xShape.GetDim(3);
    const int64_t blockSize = *attrBlockSize;
    const int64_t blockSquare = blockSize * blockSize;

    if (batch <= 0 || inputHeight <= 0 || inputWidth <= 0 || depth <= 0 || batch % blockSquare != 0)
        return ge::GRAPH_FAILED;

    const int64_t cropTop = attrCrops->GetData()[0];
    const int64_t cropBottom = attrCrops->GetData()[1];
    const int64_t cropLeft = attrCrops->GetData()[2];
    const int64_t cropRight = attrCrops->GetData()[3];
    const int64_t outBatch = batch / blockSquare;
    const int64_t outHeight = inputHeight * blockSize - cropTop - cropBottom;
    const int64_t outWidth = inputWidth * blockSize - cropLeft - cropRight;

    if (cropTop < 0 || cropBottom < 0 || cropLeft < 0 || cropRight < 0 ||
        outBatch <= 0 || outHeight <= 0 || outWidth <= 0)
        return ge::GRAPH_FAILED;

    uint32_t DT_X = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    BatchToSpaceTilingData *tiling = context->GetTilingData<BatchToSpaceTilingData>();
    if (tiling == nullptr) return ge::GRAPH_FAILED;

    uint64_t totalOutputElements = static_cast<uint64_t>(outBatch) * outHeight * outWidth * depth;
    uint64_t totalOutputRows = static_cast<uint64_t>(outBatch) * outHeight;
    uint64_t outputRowWidth = static_cast<uint64_t>(outWidth) * depth;

    uint32_t rowElems = static_cast<uint32_t>(outputRowWidth);
    uint32_t rowBytes = rowElems * elemSizeBytes;
    uint32_t isDepthAligned = (static_cast<uint32_t>(depth) % alignElems == 0) ? 1 : 0;
    uint32_t alignRows = 1;
    {
        uint32_t a = rowBytes, b = kCacheLineSize;
        while (b) { uint32_t t = b; b = a % b; a = t; }
        if (a < kCacheLineSize) alignRows = kCacheLineSize / a;
    }
    uint32_t ubTileW = 0;
    if (isDepthAligned) {
        uint64_t availUb = ubSize > kUbReserveBytes ? ubSize - kUbReserveBytes : 0;
        auto hostAlignUp = [](uint32_t v, uint32_t a) { return (v + a - 1) / a * a; };
        uint32_t maxSb = static_cast<uint32_t>(availUb / ((static_cast<uint32_t>(blockSize) + 1) * elemSizeBytes));
        uint32_t cand = maxSb > static_cast<uint32_t>(outWidth) ? static_cast<uint32_t>(outWidth) : maxSb;
        while (cand > 0) {
            uint32_t spanW = (cand + static_cast<uint32_t>(blockSize) - 1) / static_cast<uint32_t>(blockSize);
            uint32_t inElems = spanW * static_cast<uint32_t>(depth);
            uint32_t outElems = cand * static_cast<uint32_t>(depth);
            uint32_t aIn = hostAlignUp(inElems, alignElems);
            uint64_t need = (static_cast<uint64_t>(blockSize) * aIn + hostAlignUp(outElems, alignElems)) * elemSizeBytes;
            if (need <= availUb) { ubTileW = cand; break; }
            cand--;
        }
    }

    // ── Multi-core ────────────────────────────────────────────────
    // All paths are now multi-core safe (StridedDma/BulkStridedDma cover
    // non-aligned depth; the old ProcessScalar multi-core bug is no longer
    // reachable at runtime). Use uniform multi-core logic for all shapes.
    uint32_t coreNum = 1;
    if (totalOutputRows > 0 && totalOutputElements >= kMinParallelOutputElements &&
        outputRowWidth >= kMinParallelRowWidth) {
        uint64_t desired = totalOutputRows < static_cast<uint64_t>(numCoresAiv)
            ? totalOutputRows : static_cast<uint64_t>(numCoresAiv);
        coreNum = static_cast<uint32_t>(desired == 0 ? 1 : desired);
    }

    tiling->coreNum = coreNum;
    tiling->ubSizeBytes = static_cast<uint32_t>(ubSize > UINT32_MAX ? UINT32_MAX : ubSize);
    tiling->inputHeight = static_cast<uint32_t>(inputHeight);
    tiling->inputWidth = static_cast<uint32_t>(inputWidth);
    tiling->depth = static_cast<uint32_t>(depth);
    tiling->blockSize = static_cast<uint32_t>(blockSize);
    tiling->outBatch = static_cast<uint32_t>(outBatch);
    tiling->outHeight = static_cast<uint32_t>(outHeight);
    tiling->outWidth = static_cast<uint32_t>(outWidth);
    tiling->cropTop = static_cast<uint32_t>(cropTop);
    tiling->cropBottom = static_cast<uint32_t>(cropBottom);
    tiling->cropLeft = static_cast<uint32_t>(cropLeft);
    tiling->cropRight = static_cast<uint32_t>(cropRight);
    tiling->totalOutputElements = totalOutputElements;
    tiling->rowElems = rowElems;
    tiling->rowBytes = rowBytes;
    tiling->alignElems = alignElems;
    tiling->ubTileW = ubTileW;
    tiling->isDepthAligned = isDepthAligned;
    uint32_t batchWriteRows = 1;
    uint32_t minBwRows = 1;
    if (rowBytes % kAlign32 != 0) {
        uint32_t a = rowBytes, b = kAlign32;
        while (b) { uint32_t t = b; b = a % b; a = t; }
        minBwRows = kAlign32 / a;
        if (rowElems < 64) {
            // Newly enabled by threshold 64→48: small rowElems benefit from
            // larger batches to amortize barrier overhead.
            // Use up to 1/8 UB for batchBuf (conservative to avoid compiler spill).
            uint32_t maxBatchBytes = static_cast<uint32_t>(
                (ubSize > kUbReserveBytes ? ubSize - kUbReserveBytes : 0) / 8);
            uint32_t maxRowsByUb = maxBatchBytes / (static_cast<uint64_t>(rowElems) * elemSizeBytes);
            maxRowsByUb = maxRowsByUb > minBwRows ? maxRowsByUb : minBwRows;
            uint32_t target = static_cast<uint32_t>(totalOutputRows < maxRowsByUb
                ? totalOutputRows : maxRowsByUb);
            target = target < minBwRows ? minBwRows : target;
            batchWriteRows = ((target + minBwRows - 1) / minBwRows) * minBwRows;
        } else {
            batchWriteRows = minBwRows;
        }
    }
    // V5+V1: cap batchWriteRows at rowsPerCore to avoid SetValue tail batch
    if (batchWriteRows > 1 && coreNum > 0) {
        uint64_t rowsPerCore = (totalOutputRows + coreNum - 1) / coreNum;
        while (batchWriteRows > rowsPerCore && batchWriteRows > minBwRows)
            batchWriteRows -= minBwRows;
    }
    tiling->batchWriteRows = batchWriteRows;
    tiling->alignRows = alignRows;
    tiling->workspaceBytes = 0;
    tiling->useSimpleUbDirect = 0;
    tiling->useAlignedBulkDirect = 0;
    if (isDepthAligned && totalOutputElements > 2048 && coreNum > 0) {
        // UbDirect shapes: skip class Init, standalone UbDirect path
        // Only use standalone for few-rows-per-core cases (≤4):
        //   - Standalone is simpler (no class Init) and DataCopyParams merges well.
        //   - Class UbDirectPipelined overlaps load+write for multi-row shapes,
        //     which outperforms standalone's per-row TQueSync overhead.
        uint32_t rowsPerCore = (totalOutputRows + coreNum - 1) / coreNum;
        uint32_t loadBytes = rowsPerCore * static_cast<uint32_t>(inputWidth * depth)
                           * static_cast<uint32_t>(blockSize) * elemSizeBytes;
        uint32_t outBytes = ((rowElems * elemSizeBytes + 31) / 32) * 32;
        uint32_t availUb = static_cast<uint32_t>(ubSize > kUbReserveBytes ? ubSize - kUbReserveBytes : 0);
        if (loadBytes > 0 && outBytes > 0 && rowsPerCore <= 4 &&
            static_cast<uint64_t>(loadBytes) + outBytes + kUbReserveBytes <= availUb) {
            tiling->useSimpleUbDirect = 1;
            tiling->useAlignedBulkDirect = 0;  // set to 0 to fall back to UbDirect
        }
    }
    // Upper bound of per-core input elements (for UbDirect activation check)
    // Used to skip combo enumeration in kernel Init for small shapes.
    if (isDepthAligned && coreNum > 0) {
        uint64_t maxRpC = (totalOutputRows + coreNum - 1) / coreNum;
        tiling->ubTotalInEst = static_cast<uint32_t>(
            (maxRpC * static_cast<uint64_t>(inputWidth) * depth
             * static_cast<uint64_t>(blockSize) * blockSize) > UINT32_MAX
            ? UINT32_MAX : (maxRpC * static_cast<uint64_t>(inputWidth) * depth
                           * static_cast<uint64_t>(blockSize) * blockSize));
    } else {
        tiling->ubTotalInEst = 0;
    }

    context->SetBlockDim(coreNum);
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (inputShape == nullptr || outputShape == nullptr || attrs == nullptr || inputShape->GetDimNum() != 4)
        return GRAPH_FAILED;

    const gert::TypedContinuousVector<int64_t> *attrCrops = attrs->GetListInt(0);
    const int64_t *attrBlockSize = attrs->GetInt(1);
    if (attrCrops == nullptr || attrBlockSize == nullptr || attrCrops->GetSize() != 4 || *attrBlockSize <= 0)
        return GRAPH_FAILED;

    const int64_t blockSize = *attrBlockSize;
    const int64_t batch = inputShape->GetDim(0);
    const int64_t inputHeight = inputShape->GetDim(1);
    const int64_t inputWidth = inputShape->GetDim(2);
    const int64_t depth = inputShape->GetDim(3);
    const int64_t blockSquare = blockSize * blockSize;

    if (batch <= 0 || inputHeight <= 0 || inputWidth <= 0 || depth <= 0 || batch % blockSquare != 0)
        return GRAPH_FAILED;

    const int64_t cropTop = attrCrops->GetData()[0];
    const int64_t cropBottom = attrCrops->GetData()[1];
    const int64_t cropLeft = attrCrops->GetData()[2];
    const int64_t cropRight = attrCrops->GetData()[3];
    const int64_t outBatch = batch / blockSquare;
    const int64_t outHeight = inputHeight * blockSize - cropTop - cropBottom;
    const int64_t outWidth = inputWidth * blockSize - cropLeft - cropRight;

    if (cropTop < 0 || cropBottom < 0 || cropLeft < 0 || cropRight < 0 ||
        outBatch <= 0 || outHeight <= 0 || outWidth <= 0)
        return GRAPH_FAILED;

    outputShape->SetDimNum(4);
    outputShape->SetDim(0, outBatch);
    outputShape->SetDim(1, outHeight);
    outputShape->SetDim(2, outWidth);
    outputShape->SetDim(3, depth);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}

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
            .AddConfig("ascend910b")
            .AddConfig("ascend910_93");
    }
};
OP_ADD(BatchToSpace);
}