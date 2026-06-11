#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/addcmul_tiling.h"
#include "../op_kernel/tiling_key_addcmul.h"

namespace {
constexpr uint32_t ADDCMUL_MAX_DIMS_LOCAL = 16;
constexpr uint32_t TILE_DATA_NUM_HOST = 4096;

static uint32_t AlignUp(uint32_t x, uint32_t align)
{
    return (x + align - 1) / align * align;
}

static uint32_t GetDataTypeBytes(ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT || dtype == ge::DT_INT32) {
        return 4;
    }
    if (dtype == ge::DT_FLOAT16) {
        return 2;
    }
    return 1;
}

static bool BroadcastDim3(uint32_t d0, uint32_t d1, uint32_t d2, uint32_t &outDim)
{
    if (d0 == 0 || d1 == 0 || d2 == 0) {
        if ((d0 != 0 && d0 != 1) ||
            (d1 != 0 && d1 != 1) ||
            (d2 != 0 && d2 != 1)) {
            return false;
        }
        outDim = 0;
        return true;
    }

    outDim = d0;
    if (d1 > outDim) {
        outDim = d1;
    }
    if (d2 > outDim) {
        outDim = d2;
    }

    if ((d0 != 1 && d0 != outDim) ||
        (d1 != 1 && d1 != outDim) ||
        (d2 != 1 && d2 != outDim)) {
        return false;
    }

    return true;
}

static bool BroadcastDim3Int64(int64_t d0, int64_t d1, int64_t d2, int64_t &outDim)
{
    if (d0 == 0 || d1 == 0 || d2 == 0) {
        if ((d0 != 0 && d0 != 1) ||
            (d1 != 0 && d1 != 1) ||
            (d2 != 0 && d2 != 1)) {
            return false;
        }
        outDim = 0;
        return true;
    }

    outDim = d0;
    if (d1 > outDim) {
        outDim = d1;
    }
    if (d2 > outDim) {
        outDim = d2;
    }

    if ((d0 != 1 && d0 != outDim) ||
        (d1 != 1 && d1 != outDim) ||
        (d2 != 1 && d2 != outDim)) {
        return false;
    }

    return true;
}

static bool IsSameShape(const gert::Shape &a, const gert::Shape &b)
{
    if (a.GetDimNum() != b.GetDimNum()) {
        return false;
    }

    for (int64_t i = 0; i < a.GetDimNum(); ++i) {
        if (a.GetDim(i) != b.GetDim(i)) {
            return false;
        }
    }

    return true;
}

static void FillShapeRightAligned(
    const gert::Shape &shape,
    uint32_t outRank,
    const uint32_t outShape[ADDCMUL_MAX_DIMS_LOCAL],
    uint32_t inStride[ADDCMUL_MAX_DIMS_LOCAL])
{
    for (uint32_t i = 0; i < ADDCMUL_MAX_DIMS_LOCAL; ++i) {
        inStride[i] = 0;
    }

    uint32_t inRank = static_cast<uint32_t>(shape.GetDimNum());
    uint32_t offset = outRank - inRank;

    uint32_t contiguousStride[ADDCMUL_MAX_DIMS_LOCAL] = {0};
    uint32_t running = 1;

    for (int32_t i = static_cast<int32_t>(inRank) - 1; i >= 0; --i) {
        contiguousStride[i] = running;
        running *= static_cast<uint32_t>(shape.GetDim(i));
    }

    for (uint32_t i = 0; i < inRank; ++i) {
        uint32_t dim = static_cast<uint32_t>(shape.GetDim(i));
        uint32_t outIdx = offset + i;
        inStride[outIdx] = (dim == 1 && outShape[outIdx] != 1) ? 0 : contiguousStride[i];
    }
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t maxCoreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());

    const gert::Tensor *tensor_input_data = context->GetRequiredInputTensor(0);
    const gert::Tensor *tensor_x1 = context->GetRequiredInputTensor(1);
    const gert::Tensor *tensor_x2 = context->GetRequiredInputTensor(2);

    ge::DataType dtype_input_data = tensor_input_data->GetDataType();
    uint32_t DT_INPUT_DATA = static_cast<uint32_t>(dtype_input_data);
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_DATA);

    const gert::Shape &shape_input = tensor_input_data->GetShape().GetStorageShape();
    const gert::Shape &shape_x1 = tensor_x1->GetShape().GetStorageShape();
    const gert::Shape &shape_x2 = tensor_x2->GetShape().GetStorageShape();

    uint32_t rank0 = static_cast<uint32_t>(shape_input.GetDimNum());
    uint32_t rank1 = static_cast<uint32_t>(shape_x1.GetDimNum());
    uint32_t rank2 = static_cast<uint32_t>(shape_x2.GetDimNum());

    uint32_t outRank = rank0;
    if (rank1 > outRank) {
        outRank = rank1;
    }
    if (rank2 > outRank) {
        outRank = rank2;
    }
    if (outRank == 0) {
        outRank = 1;
    }
    if (outRank > ADDCMUL_MAX_DIMS_LOCAL) {
        return ge::GRAPH_FAILED;
    }

    uint32_t outShape[ADDCMUL_MAX_DIMS_LOCAL] = {1};

    for (uint32_t axis = 0; axis < outRank; ++axis) {
        uint32_t d0 = 1;
        uint32_t d1 = 1;
        uint32_t d2 = 1;

        if (axis >= outRank - rank0) {
            d0 = static_cast<uint32_t>(shape_input.GetDim(axis - (outRank - rank0)));
        }
        if (axis >= outRank - rank1) {
            d1 = static_cast<uint32_t>(shape_x1.GetDim(axis - (outRank - rank1)));
        }
        if (axis >= outRank - rank2) {
            d2 = static_cast<uint32_t>(shape_x2.GetDim(axis - (outRank - rank2)));
        }

        uint32_t outDim = 1;
        if (!BroadcastDim3(d0, d1, d2, outDim)) {
            return ge::GRAPH_FAILED;
        }
        outShape[axis] = outDim;
    }

    uint64_t outLen64 = 1;
    for (uint32_t i = 0; i < outRank; ++i) {
        outLen64 *= static_cast<uint64_t>(outShape[i]);
    }

    uint32_t length = static_cast<uint32_t>(outLen64);
    bool isNoBroadcast = IsSameShape(shape_input, shape_x1) && IsSameShape(shape_input, shape_x2);

    uint32_t elemBytes = GetDataTypeBytes(dtype_input_data);
    uint32_t alignNum = 32 / elemBytes;
    if (alignNum == 0) {
        alignNum = 1;
    }

    uint32_t coreNum = 1;
    if (length > 1024 && maxCoreNum > 0) {
        coreNum = maxCoreNum;
        if (length < coreNum) {
            coreNum = length;
        }
    }

    uint32_t blockLength = 0;
    if (length == 0) {
        coreNum = 1;
        blockLength = 0;
    } else {
        uint32_t baseBlockLength = (length + coreNum - 1) / coreNum;
        blockLength = AlignUp(baseBlockLength, alignNum);
    }

    AddcmulTilingData *tiling = context->GetTilingData<AddcmulTilingData>();

    tiling->length = length;
    tiling->blockLength = blockLength;
    tiling->rank = outRank;
    tiling->isNoBroadcast = isNoBroadcast ? 1 : 0;

    for (uint32_t i = 0; i < ADDCMUL_MAX_DIMS_LOCAL; ++i) {
        tiling->outShape[i] = 1;
        tiling->inputStride[i] = 0;
        tiling->x1Stride[i] = 0;
        tiling->x2Stride[i] = 0;
    }

    for (uint32_t i = 0; i < outRank; ++i) {
        tiling->outShape[i] = outShape[i];
    }

    FillShapeRightAligned(shape_input, outRank, outShape, tiling->inputStride);
    FillShapeRightAligned(shape_x1, outRank, outShape, tiling->x1Stride);
    FillShapeRightAligned(shape_x2, outRank, outShape, tiling->x2Stride);

    context->SetBlockDim(coreNum);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *shape_input = context->GetInputShape(0);
    const gert::Shape *shape_x1 = context->GetInputShape(1);
    const gert::Shape *shape_x2 = context->GetInputShape(2);
    gert::Shape *y_shape = context->GetOutputShape(0);

    uint32_t rank0 = static_cast<uint32_t>(shape_input->GetDimNum());
    uint32_t rank1 = static_cast<uint32_t>(shape_x1->GetDimNum());
    uint32_t rank2 = static_cast<uint32_t>(shape_x2->GetDimNum());

    uint32_t outRank = rank0;
    if (rank1 > outRank) {
        outRank = rank1;
    }
    if (rank2 > outRank) {
        outRank = rank2;
    }

    y_shape->SetDimNum(0);

    for (uint32_t axis = 0; axis < outRank; ++axis) {
        int64_t d0 = 1;
        int64_t d1 = 1;
        int64_t d2 = 1;

        if (axis >= outRank - rank0) {
            d0 = shape_input->GetDim(axis - (outRank - rank0));
        }
        if (axis >= outRank - rank1) {
            d1 = shape_x1->GetDim(axis - (outRank - rank1));
        }
        if (axis >= outRank - rank2) {
            d2 = shape_x2->GetDim(axis - (outRank - rank2));
        }

        int64_t outDim = 1;
        if (!BroadcastDim3Int64(d0, d1, d2, outDim)) {
            return GRAPH_FAILED;
        }
        y_shape->AppendDim(outDim);
    }

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Addcmul : public OpDef {
public:
    explicit Addcmul(const char *name) : OpDef(name)
    {
        this->Input("input_data")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Addcmul);
}