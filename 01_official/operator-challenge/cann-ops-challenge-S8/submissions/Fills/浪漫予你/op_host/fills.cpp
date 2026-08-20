#include "fills_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static uint32_t Float32ToUint32Bits(float x)
{
    union {
        float f;
        uint32_t u;
    } cvt;
    cvt.f = x;
    return cvt.u;
}

static uint16_t Float32ToBf16Bits(float x)
{
    uint32_t u = Float32ToUint32Bits(x);

    uint32_t lsb = (u >> 16) & 1;
    uint32_t roundingBias = 0x7fff + lsb;
    u += roundingBias;

    return static_cast<uint16_t>(u >> 16);
}

static uint32_t MakeBf16PackFromFloat(float x)
{
    uint16_t bf16Bits = Float32ToBf16Bits(x);
    return static_cast<uint32_t>(bf16Bits) |
           (static_cast<uint32_t>(bf16Bits) << 16);
}

static uint8_t FloatToUint8Host(float x)
{
    if (x < 0.0f) {
        x = 0.0f;
    }

    if (x > 255.0f) {
        x = 255.0f;
    }

    return static_cast<uint8_t>(int(x));
}

static uint32_t MakeUint8PackHost(float x)
{
    uint8_t v = FloatToUint8Host(x);
    uint32_t v32 = static_cast<uint32_t>(v);

    return v32 |
           (v32 << 8) |
           (v32 << 16) |
           (v32 << 24);
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    const float *value = attrs->GetFloat(0);

    uint16_t formerNum;
    uint32_t smallSize;
    uint16_t incSize;

    FillsTilingData tiling;

    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    int aiv_num = ascendcPlatform.GetCoreNumAiv();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);

    int64_t data_size = 1;
    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
        data_size *= x1_shape->GetStorageShape().GetDim(i);
    }

    auto dt = context->GetInputDesc(0)->GetDataType();

    uint32_t sizeofT = 2;
    if (dt == ge::DT_FLOAT || dt == ge::DT_INT32) {
        sizeofT = 4;
    } else if (dt == ge::DT_INT8 || dt == ge::DT_UINT8) {
        sizeofT = 1;
    } else if (dt == ge::DT_BF16 ||
               dt == ge::DT_FLOAT16 ||
               dt == ge::DT_INT16) {
        sizeofT = 2;
    }

    uint32_t valuePack = 0;
    if (dt == ge::DT_BF16) {
        valuePack = MakeBf16PackFromFloat(*value);
    } else if (dt == ge::DT_INT8 || dt == ge::DT_UINT8) {
        valuePack = MakeUint8PackHost(*value);
    }

    tiling.set_value(*value);
    tiling.set_valuePack(valuePack);

    int block_size = 512;
    if ((int((data_size * sizeofT + 511) / 512)) * 512 < 512 * 40) {
        block_size = 32;
    }

    uint32_t block_num =
        (data_size * sizeofT + block_size - 1) / block_size;

    aiv_num = (block_num < static_cast<uint32_t>(aiv_num))
                  ? block_num
                  : aiv_num;

    if (aiv_num < 1) {
        aiv_num = 1;
    }

    formerNum = block_num % aiv_num;
    smallSize = (block_num / aiv_num) * block_size / sizeofT;
    incSize = block_size / sizeofT;

    tiling.set_incSize(incSize);
    tiling.set_formerNum(formerNum);
    tiling.set_smallSize(smallSize);

    context->SetBlockDim(aiv_num);

    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());

    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

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

}  // namespace ge

namespace ops {

class Fills : public OpDef {
public:
    explicit Fills(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_FLOAT,
                ge::DT_INT32,
                ge::DT_INT16,
                ge::DT_UINT8,
                ge::DT_INT8
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_FLOAT,
                ge::DT_INT32,
                ge::DT_INT16,
                ge::DT_UINT8,
                ge::DT_INT8
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->Attr("value").Float();

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Fills);

}  // namespace ops