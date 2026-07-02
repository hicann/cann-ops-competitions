#include "tensor_equal_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t INPUT_NUM = 2;
constexpr uint32_t MAX_SHAPE_DIM = 4;
constexpr uint32_t SHAPE_STRIDE = MAX_SHAPE_DIM + 1;

static inline uint32_t GetTypeSize(ge::DataType dt)
{
    if (dt == ge::DT_INT8 || dt == ge::DT_UINT8) {
        return 1;
    }

    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16 || dt == ge::DT_INT16) {
        return 2;
    }

    return 4;
}

static inline bool GetPaddedDim(const gert::Shape& shape, uint32_t outputRank, uint32_t axis, int64_t& dim)
{
    uint32_t rank = static_cast<uint32_t>(shape.GetDimNum());
    if (rank + axis < outputRank) {
        dim = 1;
        return true;
    }

    dim = shape.GetDim(rank - outputRank + axis);
    return dim > 0;
}

static ge::graphStatus BuildBroadcastInfo(
    gert::TilingContext* context,
    uint32_t shapeInf[INPUT_NUM * SHAPE_STRIDE],
    uint32_t inputLength[INPUT_NUM],
    uint32_t& outputLength,
    bool& boardCast)
{
    const gert::StorageShape* storageShapes[INPUT_NUM] = {};

    for (uint32_t i = 0; i < INPUT_NUM; ++i) {
        storageShapes[i] = context->GetInputShape(i);
        auto inputTensor = context->GetInputTensor(i);
        if (storageShapes[i] == nullptr || inputTensor == nullptr) {
            return ge::GRAPH_FAILED;
        }

        int64_t shapeSize = inputTensor->GetShapeSize();
        if (shapeSize <= 0 || static_cast<uint64_t>(shapeSize) > std::numeric_limits<uint32_t>::max()) {
            return ge::GRAPH_FAILED;
        }

        inputLength[i] = static_cast<uint32_t>(shapeSize);
    }

    const gert::Shape shape0 = storageShapes[0]->GetStorageShape();
    const gert::Shape shape1 = storageShapes[1]->GetStorageShape();
    uint32_t outputRank = std::max<uint32_t>(
        static_cast<uint32_t>(shape0.GetDimNum()),
        static_cast<uint32_t>(shape1.GetDimNum()));

    uint64_t outSize64 = 1;
    uint32_t paddedDims[INPUT_NUM][MAX_SHAPE_DIM] = {};

    for (uint32_t axis = 0; axis < outputRank; ++axis) {
        int64_t d0 = 1;
        int64_t d1 = 1;
        if (!GetPaddedDim(shape0, outputRank, axis, d0) ||
            !GetPaddedDim(shape1, outputRank, axis, d1)) {
            return ge::GRAPH_FAILED;
        }

        if (d0 != d1 && d0 != 1 && d1 != 1) {
            return ge::GRAPH_FAILED;
        }

        uint32_t outDim = static_cast<uint32_t>(std::max<int64_t>(d0, d1));
        outSize64 *= outDim;
        if (outSize64 > std::numeric_limits<uint32_t>::max()) {
            return ge::GRAPH_FAILED;
        }

        if (outputRank <= MAX_SHAPE_DIM) {
            paddedDims[0][axis] = static_cast<uint32_t>(d0);
            paddedDims[1][axis] = static_cast<uint32_t>(d1);
        }
    }

    outputLength = static_cast<uint32_t>(outSize64);
    boardCast = (inputLength[0] != outputLength || inputLength[1] != outputLength);

    if (outputRank > MAX_SHAPE_DIM) {
        return boardCast ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
    }

    for (uint32_t i = 0; i < INPUT_NUM; ++i) {
        shapeInf[i * SHAPE_STRIDE] = outputRank;
        for (uint32_t axis = 0; axis < outputRank; ++axis) {
            shapeInf[i * SHAPE_STRIDE + axis + 1] = paddedDims[i][axis];
        }
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    TensorEqualTilingData tiling;
    int32_t numBuffers = 24;
    uint32_t shapeInf[INPUT_NUM * SHAPE_STRIDE] = {};
    uint32_t inputLength[INPUT_NUM] = {};
    uint32_t totalLength = 0;
    bool boardCast = false;

    auto ret = BuildBroadcastInfo(context, shapeInf, inputLength, totalLength, boardCast);
    if (ret != ge::GRAPH_SUCCESS || totalLength == 0) {
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t aivNum = ascendcPlatform.GetCoreNum();
    if (aivNum == 0) {
        aivNum = 1;
    }

    auto dt = context->GetInputTensor(0)->GetDataType();
    uint32_t sizeofdatatype = GetTypeSize(dt);
    uint32_t tmpX = 2;

    if (dt == ge::DT_INT8 || dt == ge::DT_UINT8) {
        numBuffers = 15;
    } else if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16 || dt == ge::DT_INT16) {
        numBuffers = 9;
    } else if (dt == ge::DT_INT32) {
        numBuffers = 11;
        tmpX = 1;
    } else {
        numBuffers = 9;
    }

    uint8_t alignNum = static_cast<uint8_t>(BLOCK_SIZE / sizeofdatatype);
    uint32_t tilingSize = static_cast<uint32_t>((ubSize / BLOCK_SIZE / tmpX) / numBuffers);
    tilingSize = tilingSize <= 8 ? tilingSize : tilingSize / 8 * 8;
    if (tilingSize == 0) {
        tilingSize = 1;
    }

    uint32_t blockSize = tilingSize * alignNum;

    uint32_t coreSize = 0;
    uint32_t coreRemain = 0;

    if (boardCast) {
        // Broadcast path is scalar indexing. Do not force single-core here, otherwise
        // large 4D broadcast cases easily timeout. Split the output flat range across cores.
        aivNum = (aivNum < totalLength) ? aivNum : totalLength;
        aivNum = aivNum >= 1 ? aivNum : 1;
        coreSize = totalLength / aivNum;
        coreRemain = totalLength - aivNum * coreSize;
    } else {
        aivNum = (aivNum < totalLength / blockSize) ? aivNum : (totalLength / blockSize);
        aivNum = aivNum >= 1 ? aivNum : 1;
        coreSize = (totalLength / aivNum) / (alignNum * 8) * (alignNum * 8);
        coreRemain = totalLength - aivNum * coreSize;
    }

    tiling.set_ALIGN_NUM(alignNum);
    tiling.set_block_size(blockSize);
    tiling.set_core_size(coreSize);
    tiling.set_core_remain(coreRemain);
    tiling.set_shapeInf(shapeInf);
    tiling.set_boardCast(boardCast);

    context->SetBlockDim(aivNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static inline bool GetPaddedDim(const gert::Shape& shape, uint32_t outputRank, uint32_t axis, int64_t& dim)
{
    uint32_t rank = static_cast<uint32_t>(shape.GetDimNum());
    if (rank + axis < outputRank) {
        dim = 1;
        return true;
    }

    dim = shape.GetDim(rank - outputRank + axis);
    return dim > 0;
}

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    const gert::Shape* x1Shape = context->GetInputShape(0);
    const gert::Shape* x2Shape = context->GetInputShape(1);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (x1Shape == nullptr || x2Shape == nullptr || yShape == nullptr) {
        return GRAPH_FAILED;
    }

    uint32_t outputRank = std::max<uint32_t>(
        static_cast<uint32_t>(x1Shape->GetDimNum()),
        static_cast<uint32_t>(x2Shape->GetDimNum()));

    yShape->SetDimNum(outputRank);
    for (uint32_t axis = 0; axis < outputRank; ++axis) {
        int64_t d0 = 1;
        int64_t d1 = 1;
        if (!GetPaddedDim(*x1Shape, outputRank, axis, d0) ||
            !GetPaddedDim(*x2Shape, outputRank, axis, d1)) {
            return GRAPH_FAILED;
        }

        if (d0 != d1 && d0 != 1 && d1 != 1) {
            return GRAPH_FAILED;
        }

        yShape->SetDim(axis, std::max<int64_t>(d0, d1));
    }

    return GRAPH_SUCCESS;
}
}

namespace ops {
class TensorEqual : public OpDef {
public:
    explicit TensorEqual(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8, ge::DT_BF16, ge::DT_INT16, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8, ge::DT_BF16, ge::DT_INT16, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(TensorEqual);
}
