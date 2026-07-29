
#include "tensor_equal_tiling.h"
#include "register/op_def_registry.h"
#include "host_inc.h"


namespace optiling {
constexpr uint32_t CMP_ALIGN_BYTES = 256;
constexpr uint32_t DATA_COPY_ALIGN_BYTES = 32;
constexpr uint32_t CONTIG_BUFFER_NUM = 2;
constexpr uint32_t UB_RESERVED_BYTES = 8 * 1024;
constexpr uint32_t MAX_TILE_LENGTH = 32 * 1024;
constexpr uint32_t MAX_COMPARE_REPEAT = 255;
constexpr uint32_t MAX_DATACOPYPAD_BLOCK_BYTES = 65535;

static bool BuildBroadcastShapes(const gert::Shape& x1Shape,
                                 const gert::Shape& x2Shape,
                                 uint32_t* x1ShapeRev,
                                 uint32_t* x2ShapeRev,
                                 uint32_t* yShapeRev,
                                 uint32_t& shapeSize,
                                 uint32_t& dataLength)
{
  const uint32_t x1DimNum = static_cast<uint32_t>(x1Shape.GetDimNum());
  const uint32_t x2DimNum = static_cast<uint32_t>(x2Shape.GetDimNum());
  shapeSize = x1DimNum > x2DimNum ? x1DimNum : x2DimNum;
  if (shapeSize > MAX_SHAPE_SIZE) {
    return false;
  }

  dataLength = 1;
  for (uint32_t i = 0; i < shapeSize; ++i) {
    const uint32_t x1Dim = (i < x1DimNum) ? static_cast<uint32_t>(x1Shape.GetDim(x1DimNum - 1 - i)) : 1U;
    const uint32_t x2Dim = (i < x2DimNum) ? static_cast<uint32_t>(x2Shape.GetDim(x2DimNum - 1 - i)) : 1U;
    if (x1Dim != x2Dim && x1Dim != 1U && x2Dim != 1U) {
      return false;
    }

    x1ShapeRev[i] = x1Dim;
    x2ShapeRev[i] = x2Dim;
    yShapeRev[i] = x1Dim > x2Dim ? x1Dim : x2Dim;
    dataLength *= yShapeRev[i];
  }
  return true;
}

static uint32_t GetCompareWorkBytesPerElem(ge::DataType dtype)
{
  uint32_t bytes = sizeof(uint8_t) + 3U * sizeof(uint16_t);
  if (dtype == ge::DT_INT8 || dtype == ge::DT_UINT8) {
    bytes += 2U * sizeof(uint16_t);
  } else if (dtype == ge::DT_BF16 || dtype == ge::DT_INT16) {
    bytes += 2U * sizeof(float);
  }
  return bytes;
}

static uint32_t ComputeTileLength(gert::TilingContext* context, uint32_t preferredLength)
{
  const uint32_t typeSize = static_cast<uint32_t>(getDTSize(context, 0));
  const ge::DataType dtype = context->GetInputTensor(0)->GetDataType();
  const uint32_t alignLength = CMP_ALIGN_BYTES / typeSize;
  const uint64_t ubSize = getUBSize(context);
  const uint64_t usableUb = ubSize > UB_RESERVED_BYTES ? (ubSize - UB_RESERVED_BYTES) : ubSize;

  const uint32_t queueBytesPerElem = CONTIG_BUFFER_NUM * (2U * typeSize + sizeof(uint8_t));
  const uint32_t workBytesPerElem = GetCompareWorkBytesPerElem(dtype);
  const uint32_t bytesPerElem = queueBytesPerElem + workBytesPerElem;
  const uint32_t copyPadLimit = FLOOR(MAX_DATACOPYPAD_BLOCK_BYTES / typeSize, alignLength);
  const uint32_t maxTileLength = MAX_TILE_LENGTH < copyPadLimit ? MAX_TILE_LENGTH : copyPadLimit;

  uint32_t tileLength = static_cast<uint32_t>(usableUb / bytesPerElem);
  tileLength = FLOOR(tileLength, alignLength);
  if (tileLength < alignLength) {
    tileLength = alignLength;
  }
  if (tileLength > maxTileLength) {
    tileLength = FLOOR(maxTileLength, alignLength);
  }
  if (preferredLength > 0U) {
    const uint32_t preferredTileLength = CEIL(preferredLength, alignLength);
    if (tileLength > preferredTileLength) {
      tileLength = preferredTileLength;
    }
  }
  return tileLength;
}

static uint32_t GetCompareTypeSize(ge::DataType dtype)
{
  return (dtype == ge::DT_FLOAT16 || dtype == ge::DT_UINT8 || dtype == ge::DT_INT8) ?
         static_cast<uint32_t>(sizeof(uint16_t)) :
         static_cast<uint32_t>(sizeof(float));
}

static bool IsSameShape(const uint32_t* x1Shape, const uint32_t* x2Shape, const uint32_t* yShape, uint32_t shapeSize)
{
  for (uint32_t i = 0; i < shapeSize; ++i) {
    if (x1Shape[i] != yShape[i] || x2Shape[i] != yShape[i]) {
      return false;
    }
  }
  return true;
}

static bool IsFullShape(const uint32_t* shape, const uint32_t* yShape, uint32_t shapeSize)
{
  for (uint32_t i = 0; i < shapeSize; ++i) {
    if (shape[i] != yShape[i]) {
      return false;
    }
  }
  return true;
}

static bool IsGlobalScalar(const uint32_t* shape, uint32_t shapeSize)
{
  for (uint32_t i = 0; i < shapeSize; ++i) {
    if (shape[i] != 1U) {
      return false;
    }
  }
  return true;
}

static bool IsLastDimScalarBatch(const uint32_t* x1Shape,
                                 const uint32_t* x2Shape,
                                 const uint32_t* yShape,
                                 uint32_t shapeSize)
{
  if (shapeSize == 0U) {
    return false;
  }
  const bool x1Full = IsFullShape(x1Shape, yShape, shapeSize);
  const bool x2Full = IsFullShape(x2Shape, yShape, shapeSize);
  const bool x1GlobalScalar = IsGlobalScalar(x1Shape, shapeSize);
  const bool x2GlobalScalar = IsGlobalScalar(x2Shape, shapeSize);
  return (x1Full && x2Shape[0] == 1U && !x2GlobalScalar) ||
         (x2Full && x1Shape[0] == 1U && !x1GlobalScalar);
}

static uint32_t GetContiguousSpan(const uint32_t* shape,
                                  const uint32_t* yShape,
                                  const uint32_t* yShapeRSum,
                                  uint32_t shapeSize,
                                  uint32_t dataLength)
{
  for (uint32_t i = 0; i < shapeSize; ++i) {
    if (shape[i] == 1U && yShape[i] != 1U) {
      return yShapeRSum[i];
    }
  }
  return dataLength;
}

static bool IsContiguousBatchBroadcast(const uint32_t* x1Shape,
                                       const uint32_t* x2Shape,
                                       const uint32_t* yShape,
                                       const uint32_t* yShapeRSum,
                                       uint32_t shapeSize,
                                       uint32_t dataLength)
{
  if (shapeSize == 0U || x1Shape[0] != yShape[0] || x2Shape[0] != yShape[0]) {
    return false;
  }
  const uint32_t x1Span = GetContiguousSpan(x1Shape, yShape, yShapeRSum, shapeSize, dataLength);
  const uint32_t x2Span = GetContiguousSpan(x2Shape, yShape, yShapeRSum, shapeSize, dataLength);
  const uint32_t minSpan = x1Span < x2Span ? x1Span : x2Span;
  return minSpan > yShape[0];
}

static bool IsSmallInnerBroadcast(const uint32_t* x1Shape,
                                  const uint32_t* x2Shape,
                                  const uint32_t* yShape,
                                  uint32_t shapeSize,
                                  ge::DataType dtype,
                                  uint32_t tileLength,
                                  uint32_t& alignedInnerLength,
                                  uint32_t& rowsPerTile,
                                  uint32_t& cmpRepeatLength,
                                  uint32_t& repeatStrideBlocks)
{
  const uint32_t innerLength = yShape[0];
  const uint32_t compareTypeSize = GetCompareTypeSize(dtype);
  cmpRepeatLength = CMP_ALIGN_BYTES / compareTypeSize;
  alignedInnerLength = CEIL(innerLength, cmpRepeatLength);
  rowsPerTile = alignedInnerLength == 0U ? 0U : (tileLength / alignedInnerLength);
  repeatStrideBlocks = (alignedInnerLength * compareTypeSize) / DATA_COPY_ALIGN_BYTES;
  if (rowsPerTile > MAX_COMPARE_REPEAT) {
    rowsPerTile = MAX_COMPARE_REPEAT;
  }
  return shapeSize > 1U && x1Shape[0] == yShape[0] && x2Shape[0] == yShape[0] &&
         alignedInnerLength <= tileLength && rowsPerTile > 1U &&
         repeatStrideBlocks <= MAX_COMPARE_REPEAT;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  TensorEqualTilingData tiling;
  uint32_t x1Shape[MAX_SHAPE_SIZE]{};
  uint32_t x2Shape[MAX_SHAPE_SIZE]{};
  uint32_t yShape[MAX_SHAPE_SIZE]{};
  uint32_t x1ShapeRSum[MAX_SHAPE_SIZE + 1]{};
  uint32_t x2ShapeRSum[MAX_SHAPE_SIZE + 1]{};
  uint32_t yShapeRSum[MAX_SHAPE_SIZE + 1]{};
  uint32_t shapeSize = 0;
  uint32_t dataLength = 1;

  const auto x1StorageShape = context->GetInputShape(0)->GetStorageShape();
  const auto x2StorageShape = context->GetInputShape(1)->GetStorageShape();
  if (!BuildBroadcastShapes(x1StorageShape, x2StorageShape, x1Shape, x2Shape, yShape, shapeSize, dataLength)) {
    return ge::GRAPH_FAILED;
  }

  x1ShapeRSum[0] = 1;
  x2ShapeRSum[0] = 1;
  yShapeRSum[0] = 1;
  for (uint32_t i = 1; i <= shapeSize; ++i) {
    x1ShapeRSum[i] = x1ShapeRSum[i - 1] * x1Shape[i - 1];
    x2ShapeRSum[i] = x2ShapeRSum[i - 1] * x2Shape[i - 1];
    yShapeRSum[i] = yShapeRSum[i - 1] * yShape[i - 1];
  }

  const bool sameShape = IsSameShape(x1Shape, x2Shape, yShape, shapeSize);
  const bool x1FullShape = IsFullShape(x1Shape, yShape, shapeSize);
  const bool x2FullShape = IsFullShape(x2Shape, yShape, shapeSize);
  const bool globalScalar = (x1FullShape && IsGlobalScalar(x2Shape, shapeSize)) ||
                            (x2FullShape && IsGlobalScalar(x1Shape, shapeSize));
  const bool lastDimScalarBatch = IsLastDimScalarBatch(x1Shape, x2Shape, yShape, shapeSize);
  const bool contiguousBatch = IsContiguousBatchBroadcast(x1Shape, x2Shape, yShape, yShapeRSum, shapeSize, dataLength);
  const ge::DataType dtype = context->GetInputTensor(0)->GetDataType();
  const uint32_t maxTileLength = ComputeTileLength(context, dataLength);
  uint32_t tk6AlignedInnerLength = 0U;
  uint32_t tk6RowsPerTile = 0U;
  uint32_t tk6CmpRepeatLength = 0U;
  uint32_t tk6RepeatStrideBlocks = 0U;
  const bool smallInnerBroadcast = !contiguousBatch &&
                                   IsSmallInnerBroadcast(x1Shape, x2Shape, yShape, shapeSize, dtype, maxTileLength,
                                                         tk6AlignedInnerLength, tk6RowsPerTile,
                                                         tk6CmpRepeatLength, tk6RepeatStrideBlocks);
  const uint32_t tilingKey = sameShape ? 1U :
                             (globalScalar ? 3U :
                             (lastDimScalarBatch ? 4U :
                             (contiguousBatch ? 5U : (smallInnerBroadcast ? 6U : 2U))));
  const uint32_t preferredTileLength = (tilingKey == 1U || tilingKey == 3U || tilingKey == 4U ||
                                      tilingKey == 5U || tilingKey == 6U) ?
                                      dataLength : yShape[0];

  tiling.set_x1Shape(x1Shape);
  tiling.set_x2Shape(x2Shape);
  tiling.set_yShape(yShape);
  tiling.set_x1ShapeRSum(x1ShapeRSum);
  tiling.set_x2ShapeRSum(x2ShapeRSum);
  tiling.set_yShapeRSum(yShapeRSum);
  tiling.set_shapeSize(shapeSize);
  tiling.set_dataLength(dataLength);
  const uint32_t tileLength = ComputeTileLength(context, preferredTileLength);
  const uint32_t innerLength = shapeSize > 0U ? yShape[0] : dataLength;
  tiling.set_tileLength(tileLength);
  tiling.set_innerLength(innerLength);
  tiling.set_outerLength(innerLength == 0U ? 0U : (dataLength / innerLength));
  tiling.set_tk6AlignedInnerLength(tk6AlignedInnerLength);
  tiling.set_tk6RowsPerTile(tk6RowsPerTile);
  tiling.set_tk6CmpRepeatLength(tk6CmpRepeatLength);
  tiling.set_tk6RepeatStrideBlocks(tk6RepeatStrideBlocks);
  context->SetBlockDim(1);
  context->SetTilingKey(tilingKey);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* x2_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    uint32_t x1Shape[MAX_SHAPE_SIZE]{};
    uint32_t x2Shape[MAX_SHAPE_SIZE]{};
    uint32_t yShape[MAX_SHAPE_SIZE]{};
    uint32_t shapeSize = 0;
    uint32_t dataLength = 1;
    if (!optiling::BuildBroadcastShapes(*x1_shape, *x2_shape, x1Shape, x2Shape, yShape, shapeSize, dataLength)) {
        return GRAPH_FAILED;
    }

    y_shape->SetDimNum(0);
    for (uint32_t i = 0; i < shapeSize; ++i) {
        y_shape->AppendDim(yShape[shapeSize - 1U - i]);
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
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
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
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(TensorEqual);
}
