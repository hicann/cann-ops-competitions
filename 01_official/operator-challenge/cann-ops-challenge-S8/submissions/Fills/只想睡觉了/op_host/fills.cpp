
#include "fills_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include <cstring>
#include <random>
#include <iostream>
namespace optiling {
static inline uint32_t fp32Tobitwise(float value) {
    return *reinterpret_cast<uint32_t*>(&value);
}

static inline uint16_t fp32Tobf16(float value) {
    union {
        float f;
        uint32_t u;
    } converter;
    converter.f = value;
    uint32_t u = converter.u;

    uint16_t base = (u >> 16);
    uint16_t remainder = u & 0xFFFF;

    // RNE Rounding
    if ((remainder > 0x8000) || ((remainder == 0x8000) && (base & 1))) {
        base += 1;
    }

    return base;
}
static inline int getRandomInt_1_40() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<int> dist(1, 40);
    return dist(gen);
}
static inline uint16_t fp32Tofp16(float value) {
    uint32_t u32 = *reinterpret_cast<uint32_t*>(&value);
    uint32_t sign = (u32 >> 16) & 0x8000;
    int32_t exp = ((u32 >> 23) & 0xFF) - 127;
    uint32_t mant = u32 & 0x007FFFFF;

    if (exp == 128) {
        return sign | 0x7C00 | (mant >> 13);
    } else if (exp >= 16) {
        return sign | 0x7C00;
    } else if (exp >= -14) {
        return sign | ((exp + 15) << 10) | (mant >> 13);
    } else if (exp >= -24) {
        mant |= 0x00800000;
        return sign | (mant >> (-exp - 14));
    }
    return sign;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    FillsTilingData tiling;
    const auto shape = context->GetInputTensor(0)->GetOriginShape();
    const auto runtime_attrs = context->GetAttrs();
    const float value = *(runtime_attrs->GetFloat(0));
    int32_t totalSize = 1;
    for (int i = 0; i < shape.GetDimNum(); i++) {
        totalSize *= shape.GetDim(i);
    }
    auto dt = context->GetInputTensor(0)->GetDataType(); //  ge::DT_FLOAT
    int DataTypeSize = GetSizeByDataType(dt);
    int bitwiseValue = 0;
    if (dt == ge::DT_FLOAT16) {
        uint32_t v = static_cast<uint32_t>(fp32Tofp16(value)) & 0xFFFF;
        bitwiseValue = static_cast<int>((v << 16) | v);
    } else if (dt == ge::DT_BF16) {
        uint32_t v = static_cast<uint32_t>(fp32Tobf16(value)) & 0xFFFF;
        bitwiseValue = static_cast<int>((v << 16) | v);
    } else if (dt == ge::DT_FLOAT) {
        bitwiseValue = fp32Tobitwise(value);
    } else if (dt == ge::DT_INT32) {
        bitwiseValue = static_cast<int>(value);
    } else if (dt == ge::DT_INT16) {
        uint32_t v = static_cast<uint32_t>(static_cast<int>(value)) & 0xFFFF;
        bitwiseValue = static_cast<int>((v << 16) | v);
    } else if (dt == ge::DT_UINT8) {
        uint32_t v = static_cast<uint32_t>(static_cast<int>(value)) & 0xFF;
        bitwiseValue = static_cast<int>((v << 24) | (v << 16) | (v << 8) | v);
    } else if (dt == ge::DT_INT8) {
        uint32_t v = static_cast<uint32_t>(static_cast<int>(value)) & 0xFF;
        bitwiseValue = static_cast<int>((v << 24) | (v << 16) | (v << 8) | v);
    }
    int aivNum = 40;
    int32_t blockSize = 512;
    int32_t blockNum = (totalSize * DataTypeSize + blockSize - 1) / blockSize;
    aivNum = std::max(1, totalSize * DataTypeSize / (118 * 1024));
    aivNum = std::min(aivNum, blockNum);
    aivNum = std::min(40, aivNum);

    int32_t baseSize = (blockNum + aivNum - 1) / aivNum * blockSize / sizeof(int32_t);
    tiling.set_baseSize(baseSize);
    tiling.set_value(bitwiseValue);

    if (totalSize * DataTypeSize <= 48 * 1024 * 1024 && totalSize * DataTypeSize > 40 * 1024) {
        context->SetTilingKey(2);
    } else {
        context->SetTilingKey(1);
    }

    context->SetBlockDim(aivNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t usrSize = 16 * 1024 * 1024; // 设置用户需要使用的workspace大小为256字节。
    // 如需要使用系统workspace需要调用GetLibApiWorkSpaceSize获取系统workspace的大小。
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = usrSize + sysWorkspaceSize; // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。

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
class Fills : public OpDef {
public:
    explicit Fills(const char* name) : OpDef(name) {
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

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Fills);
} // namespace ops
