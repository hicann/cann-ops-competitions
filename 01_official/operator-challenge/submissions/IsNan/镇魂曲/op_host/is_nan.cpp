#include "is_nan_tiling.h"

#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr uint32_t ALIGN_BYTES = 32U;
constexpr uint32_t UB_RESERVE_BYTES = 1U * 1024U;
constexpr uint32_t UB_TARGET_BYTES = 180U * 1024U;
constexpr uint64_t BASELINE_SPLIT_THRESHOLD_BYTES = 64ULL * 1024ULL;
constexpr uint64_t FP16_MID_SMALL_THRESHOLD_BYTES = 8ULL * 1024ULL * 1024ULL;
constexpr uint64_t FP16_MID_LARGE_THRESHOLD_BYTES = 64ULL * 1024ULL * 1024ULL;

template <typename T>
constexpr T CeilDiv(T x, T y) {
    return (x + y - 1) / y;
}

uint32_t GetElementCount(const gert::StorageShape* shape) {
    uint32_t total = 1U;
    const auto& storage = shape->GetStorageShape();
    for (int32_t i = 0; i < storage.GetDimNum(); ++i) {
        total *= static_cast<uint32_t>(storage.GetDim(i));
    }
    return total;
}

uint32_t AlignUpElements(uint32_t count, uint32_t elementBytes) {
    const uint32_t bytes = CeilDiv(count * elementBytes, ALIGN_BYTES) * ALIGN_BYTES;
    return bytes / elementBytes;
}

uint32_t GetTileCapacity(uint64_t ubSize, uint32_t inputBytes, ge::DataType inputType) {
    uint32_t bytesPerElement =
        inputBytes + sizeof(uint8_t) + sizeof(uint8_t) + 2U * sizeof(uint16_t);
    if (inputType == ge::DT_BF16) {
        bytesPerElement += sizeof(float);
    }
    const uint64_t usableUb = ubSize > UB_RESERVE_BYTES ? ubSize - UB_RESERVE_BYTES : ubSize;
    const uint64_t budget = std::min<uint64_t>(usableUb, UB_TARGET_BYTES);
    uint32_t capacity = static_cast<uint32_t>(budget / std::max<uint32_t>(1U, bytesPerElement));
    const uint32_t minAlign = std::max<uint32_t>(
        CeilDiv(ALIGN_BYTES, inputBytes),
        CeilDiv(ALIGN_BYTES, static_cast<uint32_t>(sizeof(uint8_t))));
    capacity = (capacity / minAlign) * minAlign;
    return std::max<uint32_t>(minAlign, capacity);
}

}  // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    IsNanTilingData tiling;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = platform.GetLibApiWorkSpaceSize();

    const uint32_t totalLength = GetElementCount(context->GetInputShape(0));
    const auto inputType = context->GetInputDesc(0)->GetDataType();
    const uint32_t inputBytes = GetSizeByDataType(inputType);

    uint64_t ubSize = 0U;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    uint32_t tileLength = GetTileCapacity(ubSize, inputBytes, inputType);
    if (totalLength != 0U) {
        tileLength = std::min<uint32_t>(tileLength, totalLength);
    }
    tileLength = std::max<uint32_t>(1U, tileLength);

    const uint64_t totalBytes = static_cast<uint64_t>(totalLength) * inputBytes;
    const bool useBaselineSplit = totalBytes <= BASELINE_SPLIT_THRESHOLD_BYTES;
    const uint32_t dataBlockSize = std::max<uint32_t>(1U, 512U / inputBytes);
    if (!useBaselineSplit) {
        tileLength = std::max<uint32_t>(dataBlockSize, (tileLength / dataBlockSize) * dataBlockSize);
        if (totalLength != 0U) {
            tileLength = std::min<uint32_t>(tileLength, CeilDiv(totalLength, dataBlockSize) * dataBlockSize);
        }
    }

    const uint32_t computeBlocks = totalLength == 0U ? 0U : CeilDiv(totalLength, dataBlockSize);
    const uint32_t coreNum = std::max<uint32_t>(1U, platform.GetCoreNumAiv());
    const uint32_t tileCount = totalLength == 0U ? 0U : CeilDiv(totalLength, tileLength);
    uint32_t blockDim = 1U;
    if (useBaselineSplit) {
        blockDim = tileCount == 0U ? 1U : std::min<uint32_t>(coreNum, tileCount);
    } else {
        blockDim = computeBlocks == 0U ? 1U : std::min<uint32_t>(coreNum, computeBlocks);
    }
    if (totalBytes <= 65536ULL) {
        blockDim = 1U;
    } else if (inputType == ge::DT_FLOAT16 && totalBytes <= FP16_MID_SMALL_THRESHOLD_BYTES) {
        blockDim = std::min<uint32_t>(blockDim, 8U);
    } else if (inputType == ge::DT_FLOAT16 && totalBytes <= FP16_MID_LARGE_THRESHOLD_BYTES) {
        blockDim = std::min<uint32_t>(blockDim, 24U);
    }
    const uint32_t blockLength = blockDim == 0U ? totalLength : CeilDiv(totalLength, blockDim);

    tiling.set_totalLength(totalLength);
    tiling.set_blockLength(blockLength);
    tiling.set_dataBlockSize(dataBlockSize);
    tiling.set_tileLength(tileLength);
    tiling.set_alignedInputLength(std::max<uint32_t>(
        AlignUpElements(tileLength, inputBytes),
        CeilDiv(tileLength, 64U) * 64U));
    tiling.set_alignedOutputLength(AlignUpElements(tileLength, static_cast<uint32_t>(sizeof(uint8_t))));
    tiling.set_useBaselineSplit(useBaselineSplit ? 1U : 0U);

    context->SetBlockDim(blockDim);
    context->SetTilingKey(0U);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    (void)context->GetInputDataType(0);
    context->SetOutputDataType(0, ge::DT_BOOL);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class IsNan : public OpDef {
public:
    explicit IsNan(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(IsNan);

}  // namespace ops
