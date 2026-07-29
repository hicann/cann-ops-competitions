#include "erfinv_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t MAX_TILE_LENGTH = 8192;
constexpr uint32_t MAX_HALF_TILE_LENGTH = 12288;

static uint32_t AlignUp(uint32_t value, uint32_t align) {
  return (value + align - 1) / align * align;
}

static uint32_t GetDataTypeSize(ge::DataType dataType) {
  if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
    return 2;
  }
  return 4;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
  if (aivNum == 0) {
    aivNum = 1;
  }

  const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
  uint32_t totalLength = tensorX->GetShapeSize();
  auto inputDataType = tensorX->GetDataType();
  uint32_t alignNum = BLOCK_SIZE / GetDataTypeSize(inputDataType);
  uint32_t maxBlockDim = aivNum;
  if (inputDataType == ge::DT_FLOAT16 && totalLength <= 512 &&
      maxBlockDim > 8) {
    maxBlockDim = 8;
  } else if (inputDataType == ge::DT_FLOAT16 && totalLength <= 2048 &&
             maxBlockDim > 4) {
    maxBlockDim = 4;
  }

  uint32_t blockDim = (totalLength + alignNum - 1) / alignNum;
  if (blockDim > maxBlockDim) {
    blockDim = maxBlockDim;
  }
  if (blockDim == 0) {
    blockDim = 1;
  }

  uint32_t coreSize = AlignUp((totalLength + blockDim - 1) / blockDim, alignNum);
  while (blockDim > 1 && coreSize * (blockDim - 1) >= totalLength) {
    --blockDim;
    coreSize = AlignUp((totalLength + blockDim - 1) / blockDim, alignNum);
  }
  uint32_t lastCoreSize = totalLength - coreSize * (blockDim - 1);
  uint32_t maxTileLength =
      inputDataType == ge::DT_FLOAT16 ? MAX_HALF_TILE_LENGTH
                                      : MAX_TILE_LENGTH;
  uint32_t tileLength = coreSize > maxTileLength ? maxTileLength : coreSize;
  tileLength = AlignUp(tileLength, alignNum);
  if (tileLength == 0) {
    tileLength = alignNum;
  }

  ErfinvTilingData tiling;
  tiling.set_coreSize(coreSize);
  tiling.set_lastCoreSize(lastCoreSize);
  tiling.set_tileLength(tileLength);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  context->SetBlockDim(blockDim);

  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
  return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
  const gert::Shape *xShape = context->GetInputShape(0);
  gert::Shape *yShape = context->GetOutputShape(0);
  *yShape = *xShape;
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
  const auto inputDataType = context->GetInputDataType(0);
  context->SetOutputDataType(0, inputDataType);
  return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Erfinv : public OpDef {
public:
  explicit Erfinv(const char *name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
  }
};

OP_ADD(Erfinv);
} // namespace ops
