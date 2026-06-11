// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/lerp_tiling.h"
#include "../op_kernel/tiling_key_lerp.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_start = context->GetRequiredInputTensor(0);
        ge::DataType dtype_start = tensor_start->GetDataType();
        int dtype_size_start = ge::GetSizeByDataType(dtype_start);
        uint32_t totalLength = tensor_start->GetShapeSize();

        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const float *attr_weight = attrs->GetFloat(0);

        uint32_t DT_START = static_cast<uint32_t>(dtype_start);
        ASCENDC_TPL_SEL_PARAM(context, DT_START);

        uint32_t maxElementsPerCore = static_cast<uint32_t>(ub_size / (4 * static_cast<uint64_t>(dtype_size_start)));
        if (maxElementsPerCore < 8) {
            maxElementsPerCore = 8;
        }

        uint32_t blockDim = 8;
        if (totalLength > maxElementsPerCore * blockDim) {
            blockDim = totalLength / maxElementsPerCore + 1;
            if (blockDim > static_cast<uint32_t>(num_cores_aiv)) {
                blockDim = num_cores_aiv;
            }
        }
        const uint32_t minElemsPerBlock = 8;
        if (blockDim * minElemsPerBlock > totalLength && totalLength > 0) {
            blockDim = (totalLength + minElemsPerBlock - 1) / minElemsPerBlock;
            if (blockDim < 1) {
                blockDim = 1;
            }
        }
        if (totalLength == 0) {
            blockDim = 1;
        }

        LerpTilingData *tiling = context->GetTilingData<LerpTilingData>();
        tiling->totalLength = totalLength;
        tiling->tailCoreNum = totalLength % blockDim;
        tiling->weight = *attr_weight;

        context->SetBlockDim(blockDim);

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *s = context->GetInputShape(0);
        gert::Shape *o = context->GetOutputShape(0);
        o->SetDimNum(s->GetDimNum());
        for (size_t i = 0; i < s->GetDimNum(); i++) {
            o->SetDim(i, s->GetDim(i));
        }
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
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("end")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("weight").AttrType(REQUIRED).Float();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Lerp);
}  // namespace ops

