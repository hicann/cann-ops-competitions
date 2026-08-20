
#include "../op_kernel/fills_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstring>

namespace optiling {

static uint16_t FloatToBf16Bits(float val)
{
    uint32_t f;
    memcpy(&f, &val, 4);
    f += 0x7FFF + ((f >> 16) & 1);
    return static_cast<uint16_t>(f >> 16);
}

static uint16_t FloatToFp16Bits(float val)
{
    uint32_t f;
    memcpy(&f, &val, 4);
    uint32_t s = (f >> 16) & 0x8000;
    int32_t e = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t m = f & 0x7FFFFF;
    if (e <= 0) {
        if (e < -10)
            return static_cast<uint16_t>(s);
        m = (m | 0x800000) >> (1 - e);
        return static_cast<uint16_t>(s | (m >> 13));
    }
    if (e == 0xFF - 127 + 15)
        return static_cast<uint16_t>(s | 0x7C00 | (m ? (m >> 13) : 0));
    if (e > 30)
        return static_cast<uint16_t>(s | 0x7C00);
    return static_cast<uint16_t>(s | (e << 10) | (m >> 13));
}

static uint32_t MakeFillBits32(ge::DataType dt, float value)
{
    uint32_t bits;
    switch (dt) {
        case ge::DT_FLOAT:
            memcpy(&bits, &value, 4);
            return bits;
        case ge::DT_INT32: {
            int32_t iv = static_cast<int32_t>(value);
            memcpy(&bits, &iv, 4);
            return bits;
        }
        case ge::DT_FLOAT16: {
            uint16_t h = FloatToFp16Bits(value);
            return (uint32_t(h) << 16) | h;
        }
        case ge::DT_BF16: {
            uint16_t h = FloatToBf16Bits(value);
            return (uint32_t(h) << 16) | h;
        }
        case ge::DT_INT16: {
            int16_t iv = static_cast<int16_t>(value);
            uint16_t h;
            memcpy(&h, &iv, 2);
            return (uint32_t(h) << 16) | h;
        }
        case ge::DT_UINT8: {
            uint32_t b = static_cast<uint8_t>(value);
            return b * 0x01010101u;
        }
        case ge::DT_INT8: {
            int8_t sv = static_cast<int8_t>(value);
            uint8_t b;
            memcpy(&b, &sv, 1);
            return uint32_t(b) * 0x01010101u;
        }
        default:
            memcpy(&bits, &value, 4);
            return bits;
    }
}

static uint32_t GetDtypeSize(ge::DataType dt)
{
    if (dt == ge::DT_FLOAT || dt == ge::DT_INT32)
        return 4;
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16 || dt == ge::DT_INT16)
        return 2;
    return 1; // UINT8, INT8
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    FillsTilingData tiling;
    auto plat = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = plat.GetCoreNumAiv();

    // Fixed 8KB buffer — datacopy test shows transfer size doesn't affect throughput
    constexpr uint32_t bufferLen = 8192;
    constexpr uint32_t bufferElems = bufferLen >> 2; // 2048 uint32 elements

    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    float value = *context->GetAttrs()->GetAttrPointer<float>(0);
    auto dt = context->GetInputDesc(0)->GetDataType();
    uint32_t totalBytes = totalLength * GetDtypeSize(dt);
    uint32_t fillBits32 = MakeFillBits32(dt, value);

    // TILING_KEY dispatch:
    // KEY 0: small data (totalBytes <= 512), single core, single DataCopyPad
    // KEY 1: large data, multi-core, loop + tail
    if (totalBytes <= 512) {
        // Small data path — single core
        context->SetBlockDim(1);
        context->SetTilingKey(0);

        tiling.set_fillBits32(fillBits32);
        tiling.set_formerNum(0);
        tiling.set_formerElems(0);
        tiling.set_tailElems(0);
        tiling.set_lastCoreBytes(0);
        tiling.set_bufferLen(512); // minimal buffer
        tiling.set_bufferElems(128);
        tiling.set_usedCoreNum(1);
        tiling.set_totalBytes(totalBytes);
    } else {
        // Large data path — multi-core
        // 512B-aligned core split
        uint32_t usedCoreNum = totalBytes / 512;
        if (usedCoreNum > aivNum)
            usedCoreNum = aivNum;
        if (usedCoreNum == 0)
            usedCoreNum = 1;

        uint32_t formerNum, formerBytes, tailBytes, lastCoreBytes;
        if (usedCoreNum == 1) {
            formerNum = 0;
            formerBytes = 0;
            tailBytes = 0;
            lastCoreBytes = totalBytes;
        } else {
            uint32_t base = (totalBytes / usedCoreNum / 512) * 512;
            uint32_t rem = totalBytes - base * usedCoreNum;
            uint32_t extra = rem / 512;
            formerNum = extra;
            formerBytes = base + 512;
            tailBytes = base;
            if (formerNum >= usedCoreNum)
                formerNum = usedCoreNum - 1;
            uint32_t consumed = formerNum * formerBytes + (usedCoreNum - 1 - formerNum) * tailBytes;
            lastCoreBytes = totalBytes - consumed;
        }

        context->SetBlockDim(usedCoreNum);
        context->SetTilingKey(1);

        tiling.set_fillBits32(fillBits32);
        tiling.set_formerNum(formerNum);
        tiling.set_formerElems(formerBytes >> 2);
        tiling.set_tailElems(tailBytes >> 2);
        tiling.set_lastCoreBytes(lastCoreBytes);
        tiling.set_bufferLen(bufferLen);
        tiling.set_bufferElems(bufferElems);
        tiling.set_usedCoreNum(usedCoreNum);
        tiling.set_totalBytes(totalBytes);
    }

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge


namespace ops {
class Fills : public OpDef {
public:
    explicit Fills(const char *name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("value").AttrType(REQUIRED).Float(0.0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Fills);
} // namespace ops
