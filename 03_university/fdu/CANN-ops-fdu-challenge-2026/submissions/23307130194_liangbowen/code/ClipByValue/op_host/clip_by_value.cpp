#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/clip_by_value_tiling.h"
#include "../op_kernel/tiling_key_clip_by_value.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        int dtype_size_x = ge::GetSizeByDataType(dtype_x);
        uint32_t totalLength = tensor_x->GetShapeSize();

        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const float *attr_min = attrs->GetFloat(0);
        const float *attr_max = attrs->GetFloat(1);

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 缓冲区数量
        uint32_t bufCount;
        if (dtype_x == ge::DT_FLOAT16) {
            bufCount = 2;   
        } else if (dtype_x == ge::DT_INT32) {
            bufCount = 3;  
        } else {
            bufCount = 2;   
        }

        uint32_t maxElementsPerCore = static_cast<uint32_t>(ub_size / (bufCount * static_cast<uint64_t>(dtype_size_x)));
        if (maxElementsPerCore < 32) {
            maxElementsPerCore = 32;
        }

        uint32_t blockDim = 1;
        if (totalLength > 0) {
            blockDim = (totalLength + maxElementsPerCore - 1) / maxElementsPerCore;
            if (blockDim > static_cast<uint32_t>(num_cores_aiv)) {
                blockDim = static_cast<uint32_t>(num_cores_aiv);
            }
            if (blockDim > totalLength) {
                blockDim = totalLength;
            }
        }

        ClipByValueTilingData *tiling = context->GetTilingData<ClipByValueTilingData>();
        tiling->totalLength = totalLength;
        tiling->tailCoreNum = totalLength % blockDim;
        tiling->min_val = *attr_min;
        tiling->max_val = *attr_max;

        {
            __fp16 min_h = static_cast<__fp16>(*attr_min);
            __fp16 max_h = static_cast<__fp16>(*attr_max);
            tiling->min_half_bits = static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&min_h));
            tiling->max_half_bits = static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&max_h));
        }

        uint32_t coreDataNum = (totalLength + blockDim - 1) / blockDim;
        if (coreDataNum > maxElementsPerCore) {
            tiling->tileDataNum = maxElementsPerCore;
        } else {
            tiling->tileDataNum = 0;
        }

        context->SetBlockDim(blockDim);

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}

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
}

namespace ops {
    class ClipByValue : public OpDef {
    public:
        explicit ClipByValue(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("min").AttrType(REQUIRED).Float();
            this->Attr("max").AttrType(REQUIRED).Float();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(ClipByValue);
}