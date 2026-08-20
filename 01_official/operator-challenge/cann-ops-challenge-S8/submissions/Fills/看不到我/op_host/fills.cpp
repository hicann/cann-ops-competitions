#include "fills_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstring>
#include <cmath>
#include <cstdint>

namespace optiling {

// fp32 -> bf16，round-to-nearest-even（与 PyTorch c10::BFloat16 一致）
static inline uint16_t fp32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    if (std::isnan(f)) {
        // 保 NaN：避免被误舍入成 Inf
        return static_cast<uint16_t>((u >> 16) | 0x0040u);
    }
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;     // RNE bias
    return static_cast<uint16_t>(u >> 16);
}

// fp32 -> fp16，round-to-nearest-even（normal / subnormal / inf / nan 全覆盖）
static inline uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));

    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t  e    = int32_t((x >> 23) & 0xFFu) - 127;
    const uint32_t m    = x & 0x7FFFFFu;

    if (e == 128) {                                              // Inf / NaN
        return uint16_t(sign | 0x7C00u | (m ? 0x200u : 0u));
    }
    if (e > 15)  return uint16_t(sign | 0x7C00u);                // overflow -> Inf
    if (e < -24) return uint16_t(sign);                          // underflow -> 0

    if (e >= -14) {                                              // normal
        uint32_t exp_bits = uint32_t(e + 15) << 10;
        uint32_t mant     = m >> 13;
        uint32_t round    = (m >> 12) & 1u;
        uint32_t sticky   = (m & 0xFFFu) ? 1u : 0u;
        uint32_t lsb      = mant & 1u;
        if (round && (sticky || lsb)) {
            mant += 1;
            if (mant == 0x400u) {
                mant = 0;
                exp_bits += 0x400u;
                if (exp_bits == 0x7C00u) return uint16_t(sign | 0x7C00u);
            }
        }
        return uint16_t(sign | exp_bits | mant);
    }

    // subnormal
    uint32_t shift  = uint32_t(-14 - e) + 13;
    uint32_t imp_m  = m | 0x800000u;
    uint32_t mant   = imp_m >> shift;
    uint32_t round  = (imp_m >> (shift - 1)) & 1u;
    uint32_t sticky = (imp_m & ((1u << (shift - 1)) - 1u)) ? 1u : 0u;
    uint32_t lsb    = mant & 1u;
    if (round && (sticky || lsb)) mant += 1;
    return uint16_t(sign | mant);
}

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    FillsTilingData tiling;

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *scale_attr = attrs->GetFloat(0);
    float scale = *scale_attr;

    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    int32_t data_sz = 1;
    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
        data_sz *= x1_shape->GetStorageShape().GetDim(i);

    auto dt = context->GetInputTensor(0)->GetDataType();

    // ---- 把 scale 打包进 32-bit value_bits，并确定 pack_ratio ----
    int32_t  pack_ratio = 1;     // 1: 32-bit, 2: 16-bit, 4: 8-bit
    uint32_t value_bits = 0;

    if (dt == ge::DT_FLOAT) {
        pack_ratio = 1;
        std::memcpy(&value_bits, &scale, 4);
    } else if (dt == ge::DT_INT32) {
        pack_ratio = 1;
        int32_t i = static_cast<int32_t>(scale);
        std::memcpy(&value_bits, &i, 4);
    } else if (dt == ge::DT_FLOAT16) {
        pack_ratio = 2;
        uint32_t h = fp32_to_fp16(scale);
        value_bits = (h << 16) | h;
    } else if (dt == ge::DT_BF16) {
        pack_ratio = 2;
        uint32_t b = fp32_to_bf16(scale);
        value_bits = (b << 16) | b;
    } else if (dt == ge::DT_INT16) {
        pack_ratio = 2;
        uint16_t u = static_cast<uint16_t>(static_cast<int16_t>(static_cast<int32_t>(scale)));
        value_bits = (uint32_t(u) << 16) | uint32_t(u);
    } else if (dt == ge::DT_UINT8) {
        pack_ratio = 4;
        uint32_t u = static_cast<uint8_t>(static_cast<int32_t>(scale));
        value_bits = (u << 24) | (u << 16) | (u << 8) | u;
    } else if (dt == ge::DT_INT8) {
        pack_ratio = 4;
        uint32_t u = static_cast<uint8_t>(static_cast<int8_t>(static_cast<int32_t>(scale)));
        value_bits = (u << 24) | (u << 16) | (u << 8) | u;
    }

    // ---- 仍按原算法切分 chunk（原元素单位），再转换成 32-bit 单位 ----
    int32_t chunk = ((data_sz / 40 + 511) / 512) * 512;          // 512 元素的倍数
    if (chunk < 512) chunk = 512;
    // 因为 chunk 是 512 的倍数，512 / {1,2,4} 都整除，转换无误差
    uint32_t size_packed  = uint32_t((data_sz + pack_ratio - 1) / pack_ratio);
    size_packed = (size_packed + 7) & ~7u;          // 向上对齐到 8 个 fp32 = 32B
    uint32_t chunk_packed = uint32_t(chunk / pack_ratio);
    
    // cores 也基于对齐后的 size_packed 计算，避免极端情况下越界
    int cores = int((size_packed + chunk_packed - 1) / chunk_packed);
    if (cores % 2) cores += 1;
    context->SetBlockDim(cores);
    
    tiling.set_size(size_packed);
    tiling.set_chunk(chunk_packed);
    tiling.set_value_bits(value_bits);

    context->SetTilingKey(0);    // kernel 只剩单一通路，不再需要 dtype 分支

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Fills : public OpDef {
public:
    explicit Fills(const char* name) : OpDef(name) {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT16, ge::DT_INT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT16, ge::DT_INT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("value").Float();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Fills);
}