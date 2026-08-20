#include "atanh_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#define MIN(x, y) ((x) < (y) ? (x) : (y))

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    uint32_t formerNum;
    uint32_t smallSize;
    uint32_t incSize;

    AtanhTilingData tiling;

    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    int32_t aivNum = ascendcPlatform.GetCoreNumAiv();

    const gert::StorageShape* xShape = context->GetInputShape(0);

    uint64_t dataSize = 1;
    for (int32_t i = 0; i < xShape->GetStorageShape().GetDimNum(); ++i) {
        dataSize *= xShape->GetStorageShape().GetDim(i);
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

    // 按 32B 对齐，保证 DataCopy/Vector 处理更安全
    uint32_t totalLength =
        static_cast<uint32_t>(((dataSize * sizeofT + 31) / 32) * 32 / sizeofT);

    uint32_t blockSize = 512;

    // 小 shape 时减少 block 粒度，避免 block 数太少或尾部浪费太大
    if (((dataSize * sizeofT + 511) / 512) * 512 < 512 * 40) {
        blockSize = 128;
    }

    uint32_t blockNum = (totalLength * sizeofT + blockSize - 1) / blockSize;

    aivNum = MIN(blockNum, static_cast<uint32_t>(aivNum));

    formerNum = blockNum % aivNum;
    smallSize = (blockNum / aivNum) * blockSize / sizeofT;
    incSize = blockSize / sizeofT;

    tiling.set_totalLength(totalLength);
    tiling.set_smallSize(smallSize);
    tiling.set_formerNum(formerNum);
    tiling.set_incSize(incSize);

    context->SetBlockDim(aivNum);

    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity()
    );
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    ge::DataType outputDataType = inputDataType;

    if (inputDataType == ge::DT_INT8 ||
        inputDataType == ge::DT_UINT8 ||
        inputDataType == ge::DT_INT16 ||
        inputDataType == ge::DT_INT32) {
        outputDataType = ge::DT_FLOAT;
    }

    context->SetOutputDataType(0, outputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Atanh : public OpDef {
public:
    explicit Atanh(const char* name) : OpDef(name)
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
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT
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

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Atanh);
}