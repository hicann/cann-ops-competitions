#include "gelu_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t HALF_FAST_EXACT_LIMIT = 650;
constexpr uint32_t FP32_SAFE_EXACT_LIMIT = 524288;

static inline uint32_t AlignDown(uint32_t x, uint32_t align)
{
    return x / align * align;
}

static inline uint32_t CeilDiv(uint32_t x, uint32_t y)
{
    return (x + y - 1) / y;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GeluV2TilingData tiling;

    uint64_t ubSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    uint32_t inputNum =
        static_cast<uint32_t>(context->GetInputShape(0)->GetStorageShape().GetShapeSize());

    auto dt = context->GetInputDesc(0)->GetDataType();

    uint32_t inputBytes = 4;
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
        inputBytes = 2;
    }

    uint32_t approximate = 0;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* ptr64 = attrs->GetAttrPointer<int64_t>(0);
        if (ptr64 != nullptr) {
            approximate = (*ptr64 == 0) ? 0 : 1;
        } else {
            const int32_t* ptr32 = attrs->GetAttrPointer<int32_t>(0);
            if (ptr32 != nullptr) {
                approximate = (*ptr32 == 0) ? 0 : 1;
            }
        }
    }

    uint32_t alignNum = BLOCK_SIZE / inputBytes;
    if (alignNum == 0) {
        alignNum = 1;
    }

    /*
     * BUFFER_NUM = 2
     *
     * Queue:
     *   xLocal + yLocal = 2 * BUFFER_NUM * inputBytes
     */
    uint32_t bytesPerElem = 2 * 2 * inputBytes;

    if (dt == ge::DT_FLOAT) {
        if (approximate == 0) {
            /*
             * FP32:
             *   Case3: degree-4 logistic path, tmp0 only
             *   Case5: degree-2 aggressive path, tmp0 only
             */
            bytesPerElem += 1 * sizeof(float);
        }
    } else if (dt == ge::DT_BF16) {
        bytesPerElem += (approximate == 0) ? 6 * sizeof(float) : 2 * sizeof(float);
    } else if (dt == ge::DT_FLOAT16) {
        if (approximate == 0) {
            if (inputNum <= HALF_FAST_EXACT_LIMIT) {
                // FP16 Case1 small fast path, no extra buffer
            } else {
                bytesPerElem += 1 * inputBytes;  // FP16 Case4 aggressive, tmp0 half only
            }
        } else {
            bytesPerElem += 2 * sizeof(float);  // FP16 tanh: x32 + y32
        }
    }

    uint32_t usableUb = static_cast<uint32_t>(ubSize);
    constexpr uint32_t UB_RESERVE_BYTES = 4096;
    if (usableUb > UB_RESERVE_BYTES) {
        usableUb -= UB_RESERVE_BYTES;
    }

    uint32_t tileDataNum = usableUb / bytesPerElem;
    tileDataNum = AlignDown(tileDataNum, alignNum);

    uint32_t maxTile = 4096;

    if (inputNum > 4096) {
        if (dt == ge::DT_FLOAT && approximate == 0) {
            /*
             * FP32 exact 两条路径都只用 tmp0。
             * Case3 / Case5 都可放大 tile。
             */
            maxTile = 16384;
        } else if (dt == ge::DT_FLOAT && approximate != 0) {
            maxTile = 32768;
        } else if (dt == ge::DT_FLOAT16 && approximate == 0) {
            maxTile = 32768;
        } else if (dt == ge::DT_BF16 && approximate == 0) {
            maxTile = 8192;
        } else {
            maxTile = 32768;
        }
    }

    if (tileDataNum > maxTile) {
        tileDataNum = maxTile;
    }

    tileDataNum = AlignDown(tileDataNum, alignNum);
    if (tileDataNum < alignNum) {
        tileDataNum = alignNum;
    }

    uint32_t finalTileNum = 0;
    uint32_t tailDataNum = 0;

    if (inputNum > 0) {
        finalTileNum = CeilDiv(inputNum, tileDataNum);
        tailDataNum = inputNum - tileDataNum * (finalTileNum - 1);
        if (tailDataNum == 0) {
            tailDataNum = tileDataNum;
        }
    }

    tiling.set_CoreDataNum(inputNum);
    tiling.set_finalTileNum(finalTileNum);
    tiling.set_tileDataNum(tileDataNum);
    tiling.set_TailDataNum(tailDataNum);
    tiling.set_approximate(approximate);

    context->SetBlockDim(1);

    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity()
    );
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling


namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

}  // namespace ge


namespace ops {

class GeluV2 : public OpDef {
public:
    explicit GeluV2(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->Attr("approximate")
            .AttrType(OPTIONAL)
            .Int(0);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(GeluV2);

}  // namespace ops