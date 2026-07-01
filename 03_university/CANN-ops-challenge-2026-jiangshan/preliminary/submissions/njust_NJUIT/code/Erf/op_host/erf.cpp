 #include "register/op_def_registry.h"
  #include "register/tilingdata_base.h"
  #include "tiling/platform/platform_ascendc.h"

  #include <cstdint>

  #include "../op_kernel/tiling_key_erf.h"

  namespace optiling {
  BEGIN_TILING_DATA_DEF(ErfTilingData)
      TILING_DATA_FIELD_DEF(uint32_t, totalLength);
      TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
      TILING_DATA_FIELD_DEF(uint32_t, largeCoreLength);
      TILING_DATA_FIELD_DEF(uint32_t, smallCoreLength);
      TILING_DATA_FIELD_DEF(uint32_t, largeCoreCount);
      TILING_DATA_FIELD_DEF(uint32_t, tailLength);
      TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  END_TILING_DATA_DEF;

  REGISTER_TILING_DATA_CLASS(Erf, ErfTilingData)

  constexpr uint32_t ALIGN_NUM = 64;
  constexpr uint32_t MAX_TILE_LENGTH = 4096;
  constexpr uint32_t SMALL_SHAPE_THRESHOLD = 4096;
  constexpr uint32_t MIN_CORE_DATA_LENGTH = 4096;
  constexpr uint32_t TARGET_TILE_COUNT_PER_CORE = 2;

  struct CoreTilingInfo {
      uint32_t usedCoreNum = 1;
      uint32_t largeCoreLength = 0;
      uint32_t smallCoreLength = 0;
      uint32_t largeCoreCount = 0;
      uint32_t tailLength = 0;
      uint32_t maxCoreLength = 0;
  };

  static uint32_t AlignUp(uint32_t value, uint32_t align)
  {
      return ((value + align - 1) / align) * align;
  }

  static uint32_t DivRoundUp(uint32_t value, uint32_t divisor)
  {
      return (value + divisor - 1) / divisor;
  }

  static uint32_t CalcUsedCoreNum(uint32_t lengthX, uint32_t platformCoreNum)
  {
      if (lengthX <= SMALL_SHAPE_THRESHOLD) {
          return 1;
      }

      uint32_t usedCoreNum = DivRoundUp(lengthX, MIN_CORE_DATA_LENGTH);
      if (usedCoreNum == 0) {
          usedCoreNum = 1;
      }
      if (usedCoreNum > platformCoreNum) {
          usedCoreNum = platformCoreNum;
      }
      return usedCoreNum;
  }

  static void CalcCoreTiling(uint32_t lengthX, uint32_t platformCoreNum, CoreTilingInfo &coreTiling)
  {
      coreTiling = {};

      if (lengthX == 0) {
          return;
      }

      uint32_t usedCoreNum = CalcUsedCoreNum(lengthX, platformCoreNum);

      uint32_t fullBlocks = lengthX / ALIGN_NUM;
      uint32_t tailLength = lengthX % ALIGN_NUM;

      if (fullBlocks == 0) {
          coreTiling.usedCoreNum = 1;
          coreTiling.tailLength = tailLength;
          coreTiling.maxCoreLength = tailLength;
          return;
      }

      if (usedCoreNum > fullBlocks) {
          usedCoreNum = fullBlocks;
      }

      uint32_t baseBlocksPerCore = fullBlocks / usedCoreNum;
      uint32_t largeCoreCount = fullBlocks % usedCoreNum;
      uint32_t smallCoreLength = baseBlocksPerCore * ALIGN_NUM;
      uint32_t largeCoreLength = smallCoreLength;
      if (largeCoreCount > 0) {
          largeCoreLength += ALIGN_NUM;
      }

      coreTiling.usedCoreNum = usedCoreNum;
      coreTiling.largeCoreLength = largeCoreLength;
      coreTiling.smallCoreLength = smallCoreLength;
      coreTiling.largeCoreCount = largeCoreCount;
      coreTiling.tailLength = tailLength;
      coreTiling.maxCoreLength = (largeCoreCount > 0) ? largeCoreLength : (smallCoreLength + tailLength);
  }

  static uint32_t CalcTileLength(uint32_t totalLength, uint32_t maxCoreLength)
  {
      if (maxCoreLength == 0) {
          return ALIGN_NUM;
      }

      if (totalLength <= SMALL_SHAPE_THRESHOLD) {
          uint32_t smallShapeTileLength = AlignUp(totalLength, ALIGN_NUM);
          if (smallShapeTileLength < ALIGN_NUM) {
              smallShapeTileLength = ALIGN_NUM;
          }
          if (smallShapeTileLength > MAX_TILE_LENGTH) {
              smallShapeTileLength = MAX_TILE_LENGTH;
          }
          return smallShapeTileLength;
      }

      uint32_t targetLength =
          AlignUp((maxCoreLength + TARGET_TILE_COUNT_PER_CORE - 1) / TARGET_TILE_COUNT_PER_CORE,
                  ALIGN_NUM);
      if (targetLength < ALIGN_NUM) {
          targetLength = ALIGN_NUM;
      }
      if (targetLength > MAX_TILE_LENGTH) {
          targetLength = MAX_TILE_LENGTH;
      }
      return targetLength;
  }

  static ge::graphStatus TilingFunc(gert::TilingContext *context)
  {
      const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
      if (tensorX == nullptr) {
          return ge::GRAPH_FAILED;
      }

      auto *platformInfo = context->GetPlatformInfo();
      if (platformInfo == nullptr) {
          return ge::GRAPH_FAILED;
      }

      auto platform = platform_ascendc::PlatformAscendC(platformInfo);
      uint32_t platformCoreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
      if (platformCoreNum == 0) {
          return ge::GRAPH_FAILED;
      }

      uint32_t lengthX = static_cast<uint32_t>(tensorX->GetShapeSize());
      ge::DataType dtypeX = tensorX->GetDataType();

      uint32_t dtX = static_cast<uint32_t>(dtypeX);
      ASCENDC_TPL_SEL_PARAM(context, dtX);

      CoreTilingInfo coreTiling;
      CalcCoreTiling(lengthX, platformCoreNum, coreTiling);

      uint32_t tileLength = CalcTileLength(lengthX, coreTiling.maxCoreLength);

      ErfTilingData tiling;
      tiling.set_totalLength(lengthX);
      tiling.set_usedCoreNum(coreTiling.usedCoreNum);
      tiling.set_largeCoreLength(coreTiling.largeCoreLength);
      tiling.set_smallCoreLength(coreTiling.smallCoreLength);
      tiling.set_largeCoreCount(coreTiling.largeCoreCount);
      tiling.set_tailLength(coreTiling.tailLength);
      tiling.set_tileLength(tileLength);

      tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                          context->GetRawTilingData()->GetCapacity());
      context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

      context->SetBlockDim(coreTiling.usedCoreNum);

      size_t *currentWorkspace = context->GetWorkspaceSizes(1);
      currentWorkspace[0] = 0;

      return ge::GRAPH_SUCCESS;
  }
  }  // namespace optiling

  namespace ge {
  static graphStatus InferShape(gert::InferShapeContext *context)
  {
      const auto inputShape = context->GetInputShape(0);
      auto outputShape = context->GetOutputShape(0);
      if (inputShape == nullptr || outputShape == nullptr) {
          return GRAPH_FAILED;
      }

      *outputShape = *inputShape;
      return GRAPH_SUCCESS;
  }

  static graphStatus InferDataType(gert::InferDataTypeContext *context)
  {
      context->SetOutputDataType(0, context->GetInputDataType(0));
      return GRAPH_SUCCESS;
  }
  }  // namespace ge

  namespace ops {
  class Erf : public OpDef {
  public:
      explicit Erf(const char *name) : OpDef(name)
      {
          this->Input("x")
              .ParamType(REQUIRED)
              .DataType({ge::DT_FLOAT})
              .Format({ge::FORMAT_ND});

          this->Output("y")
              .ParamType(REQUIRED)
              .DataType({ge::DT_FLOAT})
              .Format({ge::FORMAT_ND});

          this->SetInferShape(ge::InferShape)
              .SetInferDataType(ge::InferDataType);

          this->AICore()
              .SetTiling(optiling::TilingFunc)
              .AddConfig("ascend910b");
      }
  };

  OP_ADD(Erf);
  }  // namespace ops
