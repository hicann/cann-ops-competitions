// Host侧Tiling实现
#include <algorithm>
#include <cstdint>
#include <vector>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/batch_to_space_tiling.h"
#include "../op_kernel/tiling_key_batch_to_space.h"

namespace {
    constexpr uint32_t UB_TILE_ELEMS = 8192; 

    inline bool ParseNHWC(const gert::Shape &shape, uint32_t &n, uint32_t &h, uint32_t &w, uint32_t &c) {
        if (shape.GetDimNum() != 4) return false;
        const int64_t n64 = shape.GetDim(0), h64 = shape.GetDim(1), w64 = shape.GetDim(2), c64 = shape.GetDim(3);
        if (n64 <= 0 || h64 <= 0 || w64 <= 0 || c64 <= 0) return false;
        n = static_cast<uint32_t>(n64); h = static_cast<uint32_t>(h64);
        w = static_cast<uint32_t>(w64); c = static_cast<uint32_t>(c64);
        return true;
    }

    inline bool ParseAttrs(const gert::RuntimeAttrs *attrs, uint32_t &blockSize, uint32_t &cropTop,
                           uint32_t &cropBottom, uint32_t &cropLeft, uint32_t &cropRight) {
        if (attrs == nullptr) return false;
        const auto *attrCrops = attrs->GetListInt(0);
        const int64_t *attrBlockSize = attrs->GetInt(1);
        if (attrCrops == nullptr || attrBlockSize == nullptr || attrCrops->GetSize() != 4 || *attrBlockSize <= 0) return false;
        const int64_t *crops = attrCrops->GetData();
        for (size_t i = 0; i < 4; ++i) if (crops[i] < 0) return false;
        blockSize = static_cast<uint32_t>(*attrBlockSize);
        cropTop = static_cast<uint32_t>(crops[0]); cropBottom = static_cast<uint32_t>(crops[1]);
        cropLeft = static_cast<uint32_t>(crops[2]); cropRight = static_cast<uint32_t>(crops[3]);
        return true;
    }

    inline bool DeriveOutput(uint32_t inN, uint32_t inH, uint32_t inW, uint32_t blockSize, uint32_t cropTop,
                             uint32_t cropBottom, uint32_t cropLeft, uint32_t cropRight, uint32_t &outN,
                             uint32_t &outH, uint32_t &outW) {
        const uint64_t blockArea = static_cast<uint64_t>(blockSize) * blockSize;
        if (blockArea == 0 || (inN % blockArea) != 0) return false;
        const int64_t outH64 = static_cast<int64_t>(inH) * blockSize - cropTop - cropBottom;
        const int64_t outW64 = static_cast<int64_t>(inW) * blockSize - cropLeft - cropRight;
        if (outH64 <= 0 || outW64 <= 0) return false;
        outN = static_cast<uint32_t>(inN / blockArea);
        outH = static_cast<uint32_t>(outH64); outW = static_cast<uint32_t>(outW64);
        return true;
    }
}

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        const gert::StorageShape *xStorage = context->GetInputShape(0);
        const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
        if (xStorage == nullptr || tensorX == nullptr) return ge::GRAPH_FAILED;

        uint32_t inN = 0, inH = 0, inW = 0, channels = 0;
        if (!ParseNHWC(xStorage->GetStorageShape(), inN, inH, inW, channels)) return ge::GRAPH_FAILED;

        uint32_t blockSize = 0, cropTop = 0, cropBottom = 0, cropLeft = 0, cropRight = 0;
        if (!ParseAttrs(context->GetAttrs(), blockSize, cropTop, cropBottom, cropLeft, cropRight)) return ge::GRAPH_FAILED;

        uint32_t outN = 0, outH = 0, outW = 0;
        if (!DeriveOutput(inN, inH, inW, blockSize, cropTop, cropBottom, cropLeft, cropRight, outN, outH, outW)) return ge::GRAPH_FAILED;

        const ge::DataType dtypeX = tensorX->GetDataType();
        const uint32_t dtypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(dtypeX));

        uint32_t alignedMode = 0;
        uint32_t totalUnits = 0;

        if ((dtypeSize > 0) && ((channels * dtypeSize) % 32 == 0) && (channels <= UB_TILE_ELEMS)) {
            const uint32_t colsPerTile = UB_TILE_ELEMS / channels;
            const uint32_t blockLenDb = channels * dtypeSize / 32;
            const uint32_t dstStrideDb = (blockSize - 1) * blockLenDb;
            const uint32_t kMax = (colsPerTile + blockSize - 1) / blockSize;
            
            if (colsPerTile >= 1 && blockLenDb <= 65535 && dstStrideDb <= 65535 && kMax <= 4095) {
                alignedMode = 1;
                const uint32_t numColChunks = (outW + colsPerTile - 1) / colsPerTile;
                totalUnits = outN * outH * numColChunks;
            }
        }
        
        if (alignedMode == 0) {
            totalUnits = outN * outH * outW;
        }

        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
        if (coreNum < 1) coreNum = 1;

        uint32_t unitsPerCore = 0, blockDim = 1;
        if (totalUnits > 0) {
            unitsPerCore = (totalUnits + coreNum - 1) / coreNum;
            blockDim = (totalUnits + unitsPerCore - 1) / unitsPerCore;
        }

        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtypeX));

        BatchToSpaceTilingData *tiling = context->GetTilingData<BatchToSpaceTilingData>();
        tiling->inN = inN; tiling->inH = inH; tiling->inW = inW;
        tiling->outN = outN; tiling->outH = outH; tiling->outW = outW;
        tiling->channels = channels; tiling->blockSize = blockSize;
        tiling->cropTop = cropTop; tiling->cropLeft = cropLeft;
        tiling->alignedMode = alignedMode;
        tiling->totalUnits = totalUnits;
        tiling->unitsPerCore = unitsPerCore;

        context->SetBlockDim(blockDim);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *yShape = context->GetOutputShape(0);
        if (xShape == nullptr || yShape == nullptr) return GRAPH_FAILED;

        uint32_t inN = 0, inH = 0, inW = 0, channels = 0;
        if (!ParseNHWC(*xShape, inN, inH, inW, channels)) return GRAPH_FAILED;

        uint32_t blockSize = 0, cropTop = 0, cropBottom = 0, cropLeft = 0, cropRight = 0;
        if (!ParseAttrs(context->GetAttrs(), blockSize, cropTop, cropBottom, cropLeft, cropRight)) return GRAPH_FAILED;

        uint32_t outN = 0, outH = 0, outW = 0;
        if (!DeriveOutput(inN, inH, inW, blockSize, cropTop, cropBottom, cropLeft, cropRight, outN, outH, outW)) return GRAPH_FAILED;

        yShape->SetDimNum(4);
        yShape->SetDim(0, outN); yShape->SetDim(1, outH);
        yShape->SetDim(2, outW); yShape->SetDim(3, channels);
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const ge::DataType dt = context->GetInputDataType(0);
        if (dt != ge::DT_FLOAT16 && dt != ge::DT_FLOAT) return GRAPH_FAILED;
        context->SetOutputDataType(0, dt);
        return ge::GRAPH_SUCCESS;
    }
}

namespace ops {
    class BatchToSpace : public OpDef {
    public:
        explicit BatchToSpace(const char *name) : OpDef(name) {
            this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("crops").AttrType(REQUIRED).ListInt();
            this->Attr("block_size").AttrType(REQUIRED).Int();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
        }
    };
    OP_ADD(BatchToSpace);
}