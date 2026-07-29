
#include "tensor_equal_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cstdint>


namespace optiling {
constexpr uint32_t kMaxRank = 8;
constexpr uint32_t kBlockBytes = 32;
constexpr uint32_t kReserveUbBytes = 0;

static uint32_t FloorAlign(uint32_t value, uint32_t align)
{
  return value / align * align;
}

static uint32_t GetDataTypeSize(ge::DataType dtype)
{
  if (dtype == ge::DT_INT8 || dtype == ge::DT_UINT8 || dtype == ge::DT_BOOL) {
    return 1;
  }
  if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16 || dtype == ge::DT_INT16) {
    return 2;
  }
  return 4;
}

static uint32_t GetCompareCastUbCost(ge::DataType dtype)
{
  if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16 ||
      dtype == ge::DT_INT32 || dtype == ge::DT_INT16) {
    return sizeof(float);
  }
  if (dtype == ge::DT_INT8 || dtype == ge::DT_UINT8) {
    return sizeof(uint16_t);
  }
  return 0;
}

static uint32_t GetPerElementUbCost(ge::DataType dtype, uint32_t mode)
{
  const uint32_t dataSize = GetDataTypeSize(dtype);
  const uint32_t castSize = GetCompareCastUbCost(dtype);
  const uint32_t materializeWork = sizeof(uint8_t) + 2U * sizeof(uint16_t);
  const uint32_t fastCompareWork =
      castSize == 0 ? sizeof(uint8_t) : sizeof(uint8_t) + 2U * castSize;
  const uint32_t fastWork = std::max(materializeWork, fastCompareWork);
  if (mode == 1U || mode == 4U) {
    return 2U * dataSize + sizeof(uint8_t) + fastWork;
  }
  if (mode == 2U || mode == 3U || mode == 5U || mode == 6U) {
    if (dtype == ge::DT_INT8 || dtype == ge::DT_UINT8) {
      return dataSize + sizeof(uint8_t) + sizeof(uint8_t) + dataSize + 2U * sizeof(uint16_t);
    }
    return dataSize + sizeof(uint8_t) + fastWork;
  }

  const uint32_t genericCompareWork = sizeof(uint8_t) + 2U * dataSize + 2U * castSize;
  return sizeof(uint8_t) + std::max(materializeWork, genericCompareWork);
}

static uint32_t GetTileLength(gert::TilingContext* context, uint32_t mode)
{
  uint64_t ubSize = 0;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  if (ubSize > kReserveUbBytes) {
    ubSize -= kReserveUbBytes;
  }

  const auto dtype = context->GetInputDesc(0)->GetDataType();
  const uint32_t perElementUbCost = GetPerElementUbCost(dtype, mode);
  uint32_t tileLength = static_cast<uint32_t>(ubSize / perElementUbCost);
  tileLength = FloorAlign(tileLength, kBlockBytes);
  return std::max<uint32_t>(tileLength, kBlockBytes);
}

static ge::graphStatus FillBroadcastTiling(const gert::Shape& x1Shape,
                                           const gert::Shape& x2Shape,
                                           uint32_t& mode,
                                           TensorEqualTilingData& tiling)
{
  const uint32_t x1Rank = static_cast<uint32_t>(x1Shape.GetDimNum());
  const uint32_t x2Rank = static_cast<uint32_t>(x2Shape.GetDimNum());
  const uint32_t rank = std::max(x1Rank, x2Rank);
  if (rank > kMaxRank) {
    return ge::GRAPH_FAILED;
  }

  uint32_t x1Dims[kMaxRank] = {1, 1, 1, 1, 1, 1, 1, 1};
  uint32_t x2Dims[kMaxRank] = {1, 1, 1, 1, 1, 1, 1, 1};
  uint32_t x1RawStrides[kMaxRank] = {1, 1, 1, 1, 1, 1, 1, 1};
  uint32_t x2RawStrides[kMaxRank] = {1, 1, 1, 1, 1, 1, 1, 1};
  uint32_t outDims[kMaxRank] = {1, 1, 1, 1, 1, 1, 1, 1};
  uint32_t x1Strides[kMaxRank] = {1, 1, 1, 1, 1, 1, 1, 1};
  uint32_t x2Strides[kMaxRank] = {1, 1, 1, 1, 1, 1, 1, 1};

  for (uint32_t i = 0; i < x1Rank; ++i) {
    x1Dims[kMaxRank - x1Rank + i] = static_cast<uint32_t>(x1Shape.GetDim(i));
  }
  for (uint32_t i = 0; i < x2Rank; ++i) {
    x2Dims[kMaxRank - x2Rank + i] = static_cast<uint32_t>(x2Shape.GetDim(i));
  }

  for (int32_t i = static_cast<int32_t>(kMaxRank) - 2; i >= 0; --i) {
    x1RawStrides[i] = x1RawStrides[i + 1] * x1Dims[i + 1];
    x2RawStrides[i] = x2RawStrides[i + 1] * x2Dims[i + 1];
  }

  tiling.set_rank(rank);
  bool sameShape = (x1Rank == x2Rank);
  uint32_t totalSize = 1;
  uint32_t x1Size = 1;
  uint32_t x2Size = 1;
  for (uint32_t i = 0; i < kMaxRank; ++i) {
    const uint32_t x1Dim = x1Dims[i];
    const uint32_t x2Dim = x2Dims[i];
    if (x1Dim != x2Dim && x1Dim != 1 && x2Dim != 1) {
      return ge::GRAPH_FAILED;
    }
    const uint32_t outDim = std::max(x1Dim, x2Dim);
    if (x1Dim != x2Dim) {
      sameShape = false;
    }
    outDims[i] = outDim;
    x1Strides[i] = (x1Dim == 1 && outDim != 1) ? 0 : x1RawStrides[i];
    x2Strides[i] = (x2Dim == 1 && outDim != 1) ? 0 : x2RawStrides[i];
    totalSize *= outDim;
    x1Size *= x1Dim;
    x2Size *= x2Dim;
  }
  tiling.set_outDims(outDims);
  tiling.set_x1Strides(x1Strides);
  tiling.set_x2Strides(x2Strides);
  tiling.set_totalSize(totalSize);
  mode = (x1Size == totalSize && x2Size == totalSize) ? 1U : 0U;
  if (mode == 0U && totalSize > 1) {
    if (x1Size == 1 && x2Size == totalSize) {
      mode = 2U;
    } else if (x2Size == 1 && x1Size == totalSize) {
      mode = 3U;
    } else if (outDims[kMaxRank - 1] > 1) {
      const bool x1InnerContiguous = x1Strides[kMaxRank - 1] == 1U;
      const bool x2InnerContiguous = x2Strides[kMaxRank - 1] == 1U;
      const bool x1InnerScalar = x1Strides[kMaxRank - 1] == 0U;
      const bool x2InnerScalar = x2Strides[kMaxRank - 1] == 0U;
      if (x1InnerContiguous && x2InnerContiguous) {
        mode = 4U;
      } else if (x1InnerScalar && x2InnerContiguous) {
        mode = 5U;
      } else if (x2InnerScalar && x1InnerContiguous) {
        mode = 6U;
      }
    }
  }
  tiling.set_mode(mode);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  TensorEqualTilingData tiling;
  const gert::Shape& x1Shape = context->GetInputShape(0)->GetStorageShape();
  const gert::Shape& x2Shape = context->GetInputShape(1)->GetStorageShape();
  uint32_t mode = 0;
  auto status = FillBroadcastTiling(x1Shape, x2Shape, mode, tiling);
  if (status != ge::GRAPH_SUCCESS) {
    return status;
  }
  const uint32_t tileLength = GetTileLength(context, mode);
  tiling.set_tileLength(tileLength);
  context->SetBlockDim(1);
  context->SetTilingKey(mode);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* x2_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    const int64_t x1Rank = x1_shape->GetDimNum();
    const int64_t x2Rank = x2_shape->GetDimNum();
    const int64_t rank = std::max(x1Rank, x2Rank);
    y_shape->SetDimNum(rank);
    for (int64_t i = 0; i < rank; ++i) {
        const int64_t x1Index = i - (rank - x1Rank);
        const int64_t x2Index = i - (rank - x2Rank);
        const int64_t x1Dim = x1Index >= 0 ? x1_shape->GetDim(x1Index) : 1;
        const int64_t x2Dim = x2Index >= 0 ? x2_shape->GetDim(x2Index) : 1;
        if (x1Dim != x2Dim && x1Dim != 1 && x2Dim != 1) {
            return GRAPH_FAILED;
        }
        y_shape->SetDim(i, std::max(x1Dim, x2Dim));
    }
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BOOL);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class TensorEqual : public OpDef {
public:
    explicit TensorEqual(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT8, ge::DT_INT16, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT8, ge::DT_INT16, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(TensorEqual);
}
