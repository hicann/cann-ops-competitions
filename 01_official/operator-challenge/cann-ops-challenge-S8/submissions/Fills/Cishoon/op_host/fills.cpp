
#include "fills_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

typedef unsigned short ushort;
typedef unsigned int uint;

inline uint as_uint(const float& x) {
    return *(uint*)&x;
}

inline uint float_to_half_bits(const float& x) {
    uint f = as_uint(x);
    uint sign = (f >> 31) & 0x1;
    int exp = ((f >> 23) & 0xff) - 127;
    uint mant = f & 0x7fffff;

    if (exp >= 16) return (sign << 15) | 0x7c00;  // infinity
    if (exp <= -15) return sign << 15;  // 0

    uint half_exp = ((exp + 15) & 0x1f) << 10;
    uint half_mant = (mant >> 13) & 0x3ff;

    // round-to-nearest-even: 检查被丢弃的低13位
    uint remainder = mant & 0x1fff;
    if (remainder > 0x1000 || (remainder == 0x1000 && (half_mant & 1))) {
        half_mant++;
        if (half_mant > 0x3ff) {
            half_mant = 0;
            half_exp += 0x0400;  // 进位到 exponent
            if (half_exp >= 0x7c00) {
                return (sign << 15) | 0x7c00;  // overflow → inf
            }
        }
    }

    return (sign << 15) | half_exp | half_mant;
}

uint float_to_bfloat16_bits(const float x) {
    uint f = as_uint(x);
    // round-to-nearest-even: 加上舍入偏置后截断
    uint rounding_bias = ((f >> 16) & 1) + 0x7FFF;
    f += rounding_bias;
    return (f >> 16) & 0xffff;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  FillsTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);

  // 计算总元素数
  uint32_t totalSize = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    totalSize *= x1_shape->GetStorageShape().GetDim(i);
  // 获取数据类型
  auto dataType = context->GetInputDesc(0)->GetDataType();
  uint32_t origTypeSize = GetSizeByDataType(dataType);

  // 统一按 int32 搬运：half/bf16 Duplicate 时 UB bank 访问不如 int32
  // bench 实测相同字节量下 int32 路径快 ~9%；512B 对齐保证每核字节数是 4 倍数
  uint32_t totalBytes = totalSize * origTypeSize;
  totalSize = (totalBytes + 3) / 4;
  uint32_t dataTypeSize = 4;

  // 计算 blockDataNum（512 字节对齐，保证每核 GM 起始地址 512B 对齐以最大化搬运带宽）
  uint32_t blockDataNum = 512 / dataTypeSize;

  // 获取平台核数
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
  if (coreNum == 0) {
    coreNum = ascendcPlatform.GetCoreNumAic();
  }
  uint32_t totalBlocks = (totalSize + blockDataNum - 1) / blockDataNum;
  // 动态核数：数据量不足时不启动多余的核，避免空核调度开销
  uint32_t usedCoreNum = totalBlocks < coreNum ? totalBlocks : coreNum;
  if (usedCoreNum == 0) usedCoreNum = 1;
  // blockDim=1 走特殊 launch slow path，比 blockDim>=2 慢约 1.7us
  // 小 case 强制至少 2 核，多出的核在 kernel 里 early return
  if (usedCoreNum < 2 && coreNum >= 2) usedCoreNum = 2;
  uint32_t blocksPerCore = totalBlocks / usedCoreNum;
  uint32_t tailBlockNum = totalBlocks % usedCoreNum;

  uint32_t bigCoreDataNum = (blocksPerCore + 1) * blockDataNum;
  uint32_t smallCoreDataNum = blocksPerCore * blockDataNum;

  // smallCoreDataNum 按 128 对齐，低 7 bit 恒为 0；把 tailBlockNum 打包进去少传一个 tiling 字段。
  tiling.set_smallCoreDataNum(smallCoreDataNum | tailBlockNum);

  // 获取 value 属性（第 0 个属性）并转换为目标类型的位表示
  const float* valuePtr = context->GetAttrs()->GetFloat(0);
  float valueFloat = *valuePtr;
  uint32_t valueUint32 = 0;

  // 统一按 int32 搬运，所有 < 4B 类型都要把 fillValue 复制填满 32 位
  switch (dataType) {
    case ge::DT_FLOAT:
      valueUint32 = as_uint(valueFloat);
      break;
    case ge::DT_INT32:
      valueUint32 = (uint32_t)(int32_t)valueFloat;
      break;
    case ge::DT_FLOAT16: {
      uint32_t h = float_to_half_bits(valueFloat);
      valueUint32 = h | (h << 16);
      break;
    }
    case ge::DT_BF16: {
      uint32_t b = float_to_bfloat16_bits(valueFloat);
      valueUint32 = b | (b << 16);
      break;
    }
    case ge::DT_INT16: {
      uint32_t v = (uint16_t)(int16_t)valueFloat;
      valueUint32 = v | (v << 16);
      break;
    }
    case ge::DT_INT8: {
      uint32_t b = (uint8_t)(int8_t)valueFloat;
      valueUint32 = b | (b << 8) | (b << 16) | (b << 24);
      break;
    }
    case ge::DT_UINT8: {
      uint32_t b = (uint8_t)valueFloat;
      valueUint32 = b | (b << 8) | (b << 16) | (b << 24);
      break;
    }
    default:
      break;
  }

  tiling.set_valueUint32(valueUint32);

  // TilingKey 分发：单核数据量 <= UB 容量走 fast path（key=0），否则 slow path（key=1）
  // 910B 物理 UB = 192KB / 4B = 49152 int32（榨满到保留区，必须与 kernel TILE_LENGTH 一致）
  constexpr uint32_t kTileLength = 49152;
  uint64_t tilingKey = (bigCoreDataNum <= kTileLength) ? 0 : 1;
  context->SetTilingKey(tilingKey);

  context->SetBlockDim(usedCoreNum);

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  // 为 Ascend C API 预留 workspace
  size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
  size_t* currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = sysWorkspaceSize;

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
