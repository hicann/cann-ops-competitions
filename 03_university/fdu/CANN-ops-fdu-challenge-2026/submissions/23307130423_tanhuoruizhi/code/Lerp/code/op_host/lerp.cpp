// Host侧Tiling实现
#include <algorithm>

#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/lerp_tiling.h"
#include "../op_kernel/tiling_key_lerp.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t coreNum = platform.GetCoreNum();
        uint64_t ubSize = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

        uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        uint32_t typeLength = 0;
        ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);

        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const float *attrWeight = attrs->GetFloat(0);

        uint32_t DT_START = static_cast<uint32_t>(context->GetInputDesc(0)->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, DT_START);

        constexpr uint32_t BLOCK_SIZE = 32;
        uint32_t alignNum = BLOCK_SIZE / typeLength;
        uint32_t totalBlockNum = (totalLength * typeLength + BLOCK_SIZE - 1) / BLOCK_SIZE;
        uint32_t maxElementsPerCore = static_cast<uint32_t>(ubSize / (4 * static_cast<uint64_t>(typeLength)));
        maxElementsPerCore = maxElementsPerCore / alignNum * alignNum;
        maxElementsPerCore = std::max(maxElementsPerCore, alignNum);
        uint32_t maxBlocksPerCore = std::max(maxElementsPerCore / alignNum, static_cast<uint32_t>(1));

        uint32_t blockDim = 1;
        if (totalBlockNum > 0) {
            blockDim = 8;
            if (totalBlockNum > maxBlocksPerCore * blockDim) {
                blockDim = (totalBlockNum + maxBlocksPerCore - 1) / maxBlocksPerCore;
                blockDim = std::min(blockDim, coreNum);
            }
            if (blockDim > totalBlockNum) {
                blockDim = totalBlockNum;
                blockDim = std::max(blockDim, static_cast<uint32_t>(1));
            }
        }
        context->SetBlockDim(blockDim);

        LerpTilingData *tiling = context->GetTilingData<LerpTilingData>();
        tiling->totalBlockNum = totalBlockNum;
        tiling->tailBlockNum = blockDim == 0 ? 0 : totalBlockNum % blockDim;
        tiling->weight = *attrWeight;

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *inputShape = context->GetInputShape(0);
        gert::Shape *outputShape = context->GetOutputShape(0);
        *outputShape = *inputShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class Lerp : public OpDef {
    public:
        explicit Lerp(const char *name) : OpDef(name) {
            this->Input("start")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("end")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("weight").AttrType(REQUIRED).Float();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Lerp);
}  // namespace ops
