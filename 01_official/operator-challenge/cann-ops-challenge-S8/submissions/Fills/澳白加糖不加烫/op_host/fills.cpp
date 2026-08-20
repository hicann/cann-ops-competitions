
#include "fills_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstring>

namespace {
constexpr uint32_t SMALL_BLOCK_DIM = 8;
constexpr uint32_t MEDIUM_BLOCK_DIM = 20;
constexpr uint32_t LARGE_BLOCK_DIM = 40;
constexpr uint64_t MEDIUM_BLOCK_BYTES = 64 * 1024;
constexpr uint64_t LARGE_BLOCK_BYTES = 1024 * 1024;

enum FillsDType : uint32_t {
  FILL_DTYPE_FLOAT16 = 0,
  FILL_DTYPE_BF16 = 1,
  FILL_DTYPE_FLOAT = 2,
  FILL_DTYPE_INT32 = 3,
  FILL_DTYPE_INT16 = 4,
  FILL_DTYPE_UINT8 = 5,
  FILL_DTYPE_INT8 = 6,
};

static bool GetDTypeCode(ge::DataType dataType, uint32_t& dtype)
{
  switch (dataType) {
    case ge::DT_FLOAT16:
      dtype = FILL_DTYPE_FLOAT16;
      return true;
    case ge::DT_BF16:
      dtype = FILL_DTYPE_BF16;
      return true;
    case ge::DT_FLOAT:
      dtype = FILL_DTYPE_FLOAT;
      return true;
    case ge::DT_INT32:
      dtype = FILL_DTYPE_INT32;
      return true;
    case ge::DT_INT16:
      dtype = FILL_DTYPE_INT16;
      return true;
    case ge::DT_UINT8:
      dtype = FILL_DTYPE_UINT8;
      return true;
    case ge::DT_INT8:
      dtype = FILL_DTYPE_INT8;
      return true;
    default:
      return false;
  }
}

static uint32_t GetDTypeSize(uint32_t dtype)
{
  switch (dtype) {
    case FILL_DTYPE_FLOAT:
    case FILL_DTYPE_INT32:
      return 4;
    case FILL_DTYPE_FLOAT16:
    case FILL_DTYPE_BF16:
    case FILL_DTYPE_INT16:
      return 2;
    case FILL_DTYPE_UINT8:
    case FILL_DTYPE_INT8:
      return 1;
    default:
      return 1;
  }
}

static uint32_t GetBlockDim(uint64_t dataSize, uint32_t dtype, uint32_t hardwareBlockDim)
{
  if (dataSize == 0) {
    return 1;
  }

  const uint64_t totalBytes = dataSize * GetDTypeSize(dtype);
  uint32_t maxBlockDim = SMALL_BLOCK_DIM;
  if (totalBytes >= LARGE_BLOCK_BYTES) {
    maxBlockDim = hardwareBlockDim == 0 ? LARGE_BLOCK_DIM : hardwareBlockDim;
  } else if (totalBytes >= MEDIUM_BLOCK_BYTES) {
    maxBlockDim = MEDIUM_BLOCK_DIM;
  }
  uint32_t blockDim = static_cast<uint32_t>(dataSize < maxBlockDim ? dataSize : maxBlockDim);
  return blockDim;
}

static uint32_t GetByteValue(float value, uint32_t dtype)
{
  if (dtype == FILL_DTYPE_UINT8) {
    return static_cast<uint32_t>(static_cast<uint8_t>(value));
  }
  if (dtype == FILL_DTYPE_INT8) {
    return static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(value)));
  }
  return 0;
}

static int32_t GetIntValue(float value, uint32_t dtype)
{
  if (dtype == FILL_DTYPE_INT16) {
    return static_cast<int32_t>(static_cast<int16_t>(value));
  }
  if (dtype == FILL_DTYPE_INT32) {
    return static_cast<int32_t>(value);
  }
  return 0;
}

static uint32_t GetBf16Value(float value)
{
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t roundBias = 0x7FFF + ((bits >> 16) & 1);
  return (bits + roundBias) >> 16;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  FillsTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  if (x1_shape == nullptr) {
    return ge::GRAPH_FAILED;
  }

  const gert::Shape& storageShape = x1_shape->GetStorageShape();
  uint64_t data_sz = 1;
  for (size_t i = 0; i < storageShape.GetDimNum(); i++) {
    const int64_t dim = storageShape.GetDim(i);
    if (dim < 0) {
      return ge::GRAPH_FAILED;
    }
    data_sz *= static_cast<uint64_t>(dim);
  }

  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  if (attrs == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const float* value = attrs->GetAttrPointer<float>(0);
  if (value == nullptr) {
    return ge::GRAPH_FAILED;
  }

  uint32_t dtype = 0;
  if (!GetDTypeCode(context->GetInputDesc(0)->GetDataType(), dtype)) {
    return ge::GRAPH_FAILED;
  }
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  const uint32_t hardwareBlockDim = ascendcPlatform.GetCoreNum();

  tiling.set_size(data_sz);
  tiling.set_value(*value);
  tiling.set_dtype(dtype);
  tiling.set_int_value(GetIntValue(*value, dtype));
  tiling.set_bf16_value(GetBf16Value(*value));
  tiling.set_byte_value(GetByteValue(*value, dtype));

  context->SetBlockDim(GetBlockDim(data_sz, dtype, hardwareBlockDim));
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class Fills : public OpDef {
public:
    explicit Fills(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("value").Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Fills);
}
