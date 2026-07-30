
#include "atanh_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  AtanhTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);

  // 计算总元素数
  uint32_t totalSize = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    totalSize *= x1_shape->GetStorageShape().GetDim(i);

  // 获取数据类型
  auto dataType = context->GetInputDesc(0)->GetDataType();
  uint32_t dataTypeSize = GetSizeByDataType(dataType);

  // blockDataNum: 512 字节对齐，保证每核 GM 起始地址 512B 对齐以最大化搬运带宽
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
  uint32_t blocksPerCore = totalBlocks / usedCoreNum;
  uint32_t tailBlockNum = totalBlocks % usedCoreNum;

  uint32_t smallCoreDataNum = blocksPerCore * blockDataNum;

  // smallCoreDataNum 至少 128 对齐（blockDataNum ∈ {128,256,512}），低 7 bit 恒为 0
  // 把 tailBlockNum (< coreNum < 64) 打包进低 7 bit，少传一个 tiling 字段
  // bigCoreDataNum = smallCoreDataNum + blockDataNum 在 kernel 现算（blockDataNum 编译期常量）
  tiling.set_smallCoreDataNum(smallCoreDataNum | tailBlockNum);

  // 根据 UB 大小计算 tileLength
  // float/half 直接计算: inQueue(2) + outQueue(2) = 4 buffers × sizeof(T)
  // 其他类型需要 Cast: inQueue(2) + outQueue(2) 的 SrcT + 2 个 CompT 计算缓冲
  uint64_t ubSize;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  uint32_t bytesPerElement;
  switch (dataType) {
    case ge::DT_FLOAT:    // 直接计算: 4 × 4 = 16
    case ge::DT_FLOAT16:  // 直接计算: 4 × 2 = 8
      bytesPerElement = 4 * dataTypeSize;
      break;
    case ge::DT_BF16:     // SrcT=2, CompT=float(4): 4×2 + 2×4 = 16
    case ge::DT_INT16:    // SrcT=2, CompT=float(4): 4×2 + 2×4 = 16
      bytesPerElement = 4 * dataTypeSize + 2 * 4;
      break;
    case ge::DT_INT32:    // SrcT=4, CompT=float(4): 4×4 + 2×4 = 24
      bytesPerElement = 4 * dataTypeSize + 2 * 4;
      break;
    case ge::DT_INT8:     // SrcT=1, CompT=half(2): 4×1 + 2×2 = 8
    case ge::DT_UINT8:    // SrcT=1, CompT=half(2): 4×1 + 2×2 = 8
      bytesPerElement = 4 * dataTypeSize + 2 * 2;
      break;
    default:
      bytesPerElement = 4 * dataTypeSize;
      break;
  }
  uint32_t tileLength = static_cast<uint32_t>(ubSize / bytesPerElement);
  // 向下对齐到 blockDataNum
  tileLength = (tileLength / blockDataNum) * blockDataNum;
  tiling.set_tileLength(tileLength);

  context->SetBlockDim(usedCoreNum);

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  // 为 Ascend C API 预留 workspace（Div、Ln 等 API 可能需要 GM 缓存）
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
class Atanh : public OpDef {
public:
    explicit Atanh(const char* name) : OpDef(name)
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

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Atanh);
}
