#include "gelu_v2_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

namespace optiling {
    constexpr uint32_t BUFFER_NUM = 2;
    constexpr uint32_t DMA_ALIGNMENT = 512;       // DMA突发对齐(bytes)
    constexpr uint32_t DMA_BLOCK_SIZE = 32;       // DataCopy最小对齐(bytes)

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    GeluV2TilingData tiling;
    uint32_t tilingKey = 0;

    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    auto dataType = context->GetInputDesc(0)->GetDataType();
    
    // BF16支持校验
    if (dataType == ge::DT_BF16 && 
        (socVersion != platform_ascendc::SocVersion::ASCEND910B && socVersion != platform_ascendc::SocVersion::ASCEND310B)) {
        return ge::GRAPH_FAILED;
    }

    // 空输入校验
    uint64_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    if (inputNum == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dataType, typeLength);
    
    // 数据类型 & tilingKey 基础值
    uint32_t typeKey = 0;
    switch (dataType) {
        case ge::DT_FLOAT16: typeKey = 0; break;
        case ge::DT_FLOAT:   typeKey = 1; break;
        case ge::DT_BF16:    typeKey = 2; break;
        default: return ge::GRAPH_FAILED;
    }

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // 内存模型 (双缓冲 + 全量tile计算):
    // Float: inQueue(2*tile*4) + outQueue(2*tile*4) + tempBuf1(tile*4) = 20 bytes/elem
    // Half:  inQueue(2*tile*2) + outQueue(2*tile*2) + (castBuf+temp1+temp2)(3*tile*4) = 20 bytes/elem
    const uint32_t computeTypeSize = sizeof(float);
    const uint32_t numComputeBufs = (dataType == ge::DT_FLOAT) ? 1 : 3;
    const uint32_t memPerElement = BUFFER_NUM * 2 * typeLength + numComputeBufs * computeTypeSize;

    // 计算最大tile大小并DMA对齐
    uint32_t maxElements = static_cast<uint32_t>(ubSize / memPerElement);
    uint32_t tileDataNumBytes = maxElements * typeLength;
    tileDataNumBytes = (tileDataNumBytes / DMA_ALIGNMENT) * DMA_ALIGNMENT;  // 向下对齐
    uint32_t tileDataNum = tileDataNumBytes / typeLength;

    // 验证总 UB 用量
    const uint32_t computeAlignedSize = ((tileDataNum * computeTypeSize + DMA_ALIGNMENT - 1) / DMA_ALIGNMENT * DMA_ALIGNMENT);
    const uint32_t totalUbUsed = BUFFER_NUM * 2 * tileDataNum * typeLength + numComputeBufs * computeAlignedSize;
    if (totalUbUsed > ubSize) {
        tileDataNum = static_cast<uint32_t>(ubSize / memPerElement);
        tileDataNumBytes = tileDataNum * typeLength;
        tileDataNumBytes = (tileDataNumBytes / DMA_ALIGNMENT) * DMA_ALIGNMENT;
        tileDataNum = tileDataNumBytes / typeLength;
    }

    if (tileDataNum == 0) {
        return ge::GRAPH_FAILED;
    }

    // 单核: 计算tile数和尾部tile大小
    uint32_t tileNum = static_cast<uint32_t>((inputNum + tileDataNum - 1) / tileDataNum);
    uint32_t remaining = static_cast<uint32_t>(inputNum) - (tileNum - 1) * tileDataNum;
    // 尾部tile对齐到DataCopy最小粒度(32字节)
    uint32_t dmaBlockElems = DMA_BLOCK_SIZE / typeLength;  // float:8, half:16
    uint32_t lastTileDataNum = ((remaining + dmaBlockElems - 1) / dmaBlockElems) * dmaBlockElems;
    lastTileDataNum = std::min(lastTileDataNum, tileDataNum);  // 不超过完整tile

    // 计算缓冲区对齐大小(float元素数)
    auto alignComputeSize = [](uint32_t dataNum) -> uint32_t {
        constexpr uint32_t COMPUTE_ALIGNMENT = 512;
        uint32_t sizeInBytes = dataNum * sizeof(float);
        return (sizeInBytes + COMPUTE_ALIGNMENT - 1) / COMPUTE_ALIGNMENT * COMPUTE_ALIGNMENT / sizeof(float);
    };
    const uint32_t processDataNumComputes = alignComputeSize(tileDataNum);

    // 赋值tiling数据
    tiling.set_tileDataNum(tileDataNum);
    tiling.set_tileNum(tileNum);
    tiling.set_lastTileDataNum(lastTileDataNum);
    tiling.set_processDataNum_computes(processDataNumComputes);
    tiling.set_typeLength(typeLength);

    // 读取approximate属性，编码进tilingKey
    uint32_t approximate = 0;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        if (const int64_t* approx_ptr = attrs->GetInt(0); approx_ptr != nullptr) {
            approximate = static_cast<uint32_t>(*approx_ptr);
        }
    }
    // tilingKey = typeKey + approximate * 3
    // exact(0): 0=half, 1=float, 2=bf16 | tanh(1): 3=half, 4=float, 5=bf16
    tilingKey = typeKey + approximate * 3;

    // 写入tiling数据
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    
    // workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    
    // 单核: BlockDim=1 & TilingKey
    context->SetBlockDim(1);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class GeluV2 : public OpDef {
public:
    explicit GeluV2(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("approximate").Int(0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend310b");
    }
};

OP_ADD(GeluV2);
} // namespace ops
