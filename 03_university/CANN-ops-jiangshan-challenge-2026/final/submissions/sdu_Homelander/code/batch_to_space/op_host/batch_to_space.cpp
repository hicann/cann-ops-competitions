#include "register/op_def_registry.h"
#include "../op_kernel/batch_to_space_tiling.h"
#include "../op_kernel/tiling_key_batch_to_space.h"

namespace {
static inline uint32_t SafeU32(int64_t v) {
    return static_cast<uint32_t>(v);
}
}  // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);

    ge::DataType dtypeX = tensorX->GetDataType();
    int32_t dtypeSizeX = ge::GetSizeByDataType(dtypeX);

    const gert::Shape &xShape = tensorX->GetOriginShape();
    
    uint32_t batch = SafeU32(xShape.GetDim(0));
    uint32_t height = SafeU32(xShape.GetDim(1));
    uint32_t width = SafeU32(xShape.GetDim(2));
    uint32_t depth = SafeU32(xShape.GetDim(3));

    const gert::RuntimeAttrs *attrs = context->GetAttrs();

    const gert::TypedContinuousVector<int64_t> *attrCrops = attrs->GetListInt(0);
    const int64_t *attrBlockSize = attrs->GetInt(1);

    const int64_t *crops = attrCrops->GetData();

    uint32_t cropTop = SafeU32(crops[0]);
    uint32_t cropBottom = SafeU32(crops[1]);
    uint32_t cropLeft = SafeU32(crops[2]);
    uint32_t cropRight = SafeU32(crops[3]);
    uint32_t blockSize = SafeU32(*attrBlockSize);

    uint32_t blockSquare = blockSize * blockSize;
    uint32_t outBatch = batch / blockSquare;
    uint32_t outHeight = height * blockSize - cropTop - cropBottom;
    uint32_t outWidth = width * blockSize - cropLeft - cropRight;

    uint64_t inputLength64 =
        static_cast<uint64_t>(batch) * height * width * depth;

    uint64_t outputLength64 =
        static_cast<uint64_t>(outBatch) * outHeight * outWidth * depth;

    uint32_t depthBytes = depth * static_cast<uint32_t>(dtypeSizeX);

    uint64_t rowTaskNum64 = 0;

    if (depthBytes % 32 == 0) {
        rowTaskNum64 = static_cast<uint64_t>(outBatch) * blockSquare * height;
    } else {
        rowTaskNum64 = static_cast<uint64_t>(outBatch) * height;
    }

    BatchToSpaceTilingData *tiling = context->GetTilingData<BatchToSpaceTilingData>();

    tiling->height = height;
    tiling->width = width;
    tiling->depth = depth;

    tiling->blockSize = blockSize;

    tiling->cropTop = cropTop;
    tiling->cropBottom = cropBottom;
    tiling->cropLeft = cropLeft;
    tiling->cropRight = cropRight;

    tiling->outBatch = outBatch;
    tiling->outHeight = outHeight;
    tiling->outWidth = outWidth;

    tiling->inputLength = static_cast<uint32_t>(inputLength64);
    tiling->outputLength = static_cast<uint32_t>(outputLength64);
    tiling->rowTaskNum = static_cast<uint32_t>(rowTaskNum64);

    uint32_t DT_X = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    uint64_t totalBytes = outputLength64 * static_cast<uint64_t>(dtypeSizeX);

    uint32_t blockDim;

    if (totalBytes <= 12 * 1024) {  //第2个点
        blockDim = 8;  
    }
    else if (totalBytes <= 64 * 1024) { //4
        blockDim = 10;
    }
    else if (totalBytes <= 128 * 1024) { //6 7
        if(width==1) //7
           blockDim = 8;
        else         //6
           blockDim = 8;
    } else if (totalBytes <= 1024 * 1024) { //3
        blockDim = 20;
    } 
    else if (totalBytes <= 4096 * 1024) { //1
        blockDim = 16;
    } 
    else if(totalBytes <= 6144 * 1024) { //10
        blockDim = 40;
    }
    else if (totalBytes <= 7680 * 1024) { //9
        blockDim = 36;
    }
    else if (totalBytes <= 8192 * 1024) { //5
        blockDim = 40;
    }
    else {                               //8
        blockDim = 24;
    }

    context->SetBlockDim(blockDim);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
 

    const gert::TypedContinuousVector<int64_t> *attrCrops = attrs->GetListInt(0);
    const int64_t *attrBlockSize = attrs->GetInt(1);

    const int64_t *crops = attrCrops->GetData();

    int64_t batch = xShape->GetDim(0);
    int64_t height = xShape->GetDim(1);
    int64_t width = xShape->GetDim(2);
    int64_t depth = xShape->GetDim(3);

    int64_t cropTop = crops[0];
    int64_t cropBottom = crops[1];
    int64_t cropLeft = crops[2];
    int64_t cropRight = crops[3];
    int64_t blockSize = *attrBlockSize;


    int64_t blockSquare = blockSize * blockSize;

    int64_t outBatch = batch / blockSquare;
    int64_t outHeight = height * blockSize - cropTop - cropBottom;
    int64_t outWidth = width * blockSize - cropLeft - cropRight;

    yShape->SetDimNum(4);
    yShape->SetDim(0, outBatch);
    yShape->SetDim(1, outHeight);
    yShape->SetDim(2, outWidth);
    yShape->SetDim(3, depth);

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class BatchToSpace : public OpDef {
public:
    explicit BatchToSpace(const char *name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("crops").AttrType(REQUIRED).ListInt();
        this->Attr("block_size").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(BatchToSpace);

}  // namespace ops
