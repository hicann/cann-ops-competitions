// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

#include "../op_kernel/addcmul_tiling.h"
#include "../op_kernel/tiling_key_addcmul.h"

namespace optiling {

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t coreNum = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_input = context->GetRequiredInputTensor(0);
        ge::DataType dtype = tensor_input->GetDataType();
        int32_t typeSize = ge::GetSizeByDataType(dtype);

        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtype));

        AddcmulTilingData *tiling = context->GetTilingData<AddcmulTilingData>();

        uint32_t total_length = 0, min_length = context->GetInputTensor(0)->GetShapeSize();
        for (int i = 0; i < 3; ++i) {
            total_length = std::max<uint32_t>(total_length, context->GetInputTensor(i)->GetShapeSize());
            min_length = std::min<uint32_t>(min_length, context->GetInputTensor(i)->GetShapeSize());
        }
        uint32_t input_data_length = context->GetInputTensor(0)->GetShapeSize();
        uint32_t x1_length = context->GetInputTensor(1)->GetShapeSize();
        uint32_t x2_length = context->GetInputTensor(2)->GetShapeSize();

        tiling->totalLength = total_length;
        tiling->input_data_length = input_data_length;
        tiling->x1_length = x1_length;
        tiling->x2_length = x2_length;
        tiling->needBroadcast = (input_data_length == total_length && x1_length == total_length && x2_length == total_length) ? 0 : 1;

        uint32_t alignNum = 32 / typeSize;
        if (alignNum == 0) alignNum = 1;

        constexpr uint32_t TOTAL_BUF_NUM = 8;
        uint32_t perElemBytes;
        if (dtype == ge::DT_INT8) {
            perElemBytes = TOTAL_BUF_NUM * static_cast<uint32_t>(typeSize)
                         + 2 * static_cast<uint32_t>(2);
        } else {
            perElemBytes = TOTAL_BUF_NUM * static_cast<uint32_t>(typeSize);
        }
        uint32_t availUb = static_cast<uint32_t>(ub_size * 9 / 10);
        uint32_t maxTileDataNum = availUb / perElemBytes;
        maxTileDataNum = (maxTileDataNum / alignNum) * alignNum;
        if (maxTileDataNum < alignNum) maxTileDataNum = alignNum;

        if (tiling->needBroadcast) {
            uint32_t srcMinLen = input_data_length;
            if (x1_length < srcMinLen) srcMinLen = x1_length;
            if (x2_length < srcMinLen) srcMinLen = x2_length;
            uint32_t block_size = maxTileDataNum < srcMinLen ? maxTileDataNum : srcMinLen;
            while (block_size > 0 && (srcMinLen % block_size != 0 || block_size % alignNum != 0)) {
                block_size--;
            }
            if (block_size == 0) block_size = alignNum;
            maxTileDataNum = block_size;
        }

        uint32_t totalLengthAligned = ((total_length + alignNum - 1) / alignNum) * alignNum;

        uint32_t dataPerCore = totalLengthAligned / coreNum;
        dataPerCore = (dataPerCore / alignNum) * alignNum;

        uint32_t tailBlockNum = (totalLengthAligned - dataPerCore * coreNum) / alignNum;
        uint32_t bigCoreDataNum = dataPerCore + alignNum;
        uint32_t smallCoreDataNum = dataPerCore;

        uint32_t finalBigTileNum = 0;
        uint32_t bigTailDataNum = 0;
        if (bigCoreDataNum > 0) {
            finalBigTileNum = (bigCoreDataNum + maxTileDataNum - 1) / maxTileDataNum;
            bigTailDataNum = bigCoreDataNum - (finalBigTileNum - 1) * maxTileDataNum;
        }

        uint32_t finalSmallTileNum = 0;
        uint32_t smallTailDataNum = 0;
        if (smallCoreDataNum > 0) {
            finalSmallTileNum = (smallCoreDataNum + maxTileDataNum - 1) / maxTileDataNum;
            smallTailDataNum = smallCoreDataNum - (finalSmallTileNum - 1) * maxTileDataNum;
        }

        tiling->smallCoreDataNum = smallCoreDataNum;
        tiling->bigCoreDataNum = bigCoreDataNum;
        tiling->finalBigTileNum = finalBigTileNum;
        tiling->finalSmallTileNum = finalSmallTileNum;
        tiling->tileDataNum = maxTileDataNum;
        tiling->smallTailDataNum = smallTailDataNum;
        tiling->bigTailDataNum = bigTailDataNum;
        tiling->tailBlockNum = tailBlockNum;
        tiling->block_size = 0;

        context->SetBlockDim(coreNum);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
} // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape* x1_shape = context->GetInputShape(0);
        gert::Shape* y_shape = context->GetOutputShape(0);
        *y_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const auto inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class Addcmul : public OpDef {
    public:
        explicit Addcmul(const char *name) : OpDef(name) {
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
}  // namespace ops
