// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/addcmul_tiling.h"
#include "../op_kernel/tiling_key_addcmul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        const gert::Tensor *tensor_input_data = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_x1 = context->GetRequiredInputTensor(1);
        const gert::Tensor *tensor_x2 = context->GetRequiredInputTensor(2);
        const gert::Tensor *tensor_value = context->GetRequiredInputTensor(3);
        ge::DataType dtype_input_data = tensor_input_data->GetDataType();
        int dtype_size_input_data = ge::GetSizeByDataType(dtype_input_data);

        auto s0 = context->GetInputShape(0)->GetOriginShape();
        auto s1 = context->GetInputShape(1)->GetOriginShape();
        auto s2 = context->GetInputShape(2)->GetOriginShape();
        auto s3 = context->GetInputShape(3)->GetOriginShape();

        size_t d0 = s0.GetDimNum(), d1 = s1.GetDimNum(), d2 = s2.GetDimNum(), d3 = s3.GetDimNum();
        size_t maxD = d0;
        if (d1 > maxD) maxD = d1;
        if (d2 > maxD) maxD = d2;
        if (d3 > maxD) maxD = d3;

        uint32_t totalLength = 1;
        for (size_t i = 0; i < maxD; i++) {
            auto g = [](const decltype(s0)& s, size_t dn, size_t idx) -> int64_t {
                if (idx >= dn) return 1;
                return s.GetDim(dn - 1 - idx);
            };
            int64_t v = g(s0, d0, i);
            if (g(s1, d1, i) > v) v = g(s1, d1, i);
            if (g(s2, d2, i) > v) v = g(s2, d2, i);
            if (g(s3, d3, i) > v) v = g(s3, d3, i);
            totalLength *= static_cast<uint32_t>(v);
        }

        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 调整点1：缓冲区估算系数从6增大至8，使maxElementsPerCore更小，推动使用更多核
        uint32_t bufCount = 8;
        uint32_t maxElementsPerCore = static_cast<uint32_t>(ub_size / (bufCount * static_cast<uint64_t>(dtype_size_input_data)));
        if (maxElementsPerCore < 8) maxElementsPerCore = 8;

        // 调整点2：核数下限设为8，避免小核数导致性能瓶颈
        uint32_t blockDim = 8;
        if (totalLength > maxElementsPerCore * blockDim) {
            blockDim = totalLength / maxElementsPerCore + 1;
            if (blockDim > static_cast<uint32_t>(num_cores_aiv)) {
                blockDim = static_cast<uint32_t>(num_cores_aiv);
            }
        }

        // 调整点3：最小每核元素数，防止任务切分过细导致边界错误
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

        uint32_t DT_INPUT_DATA = static_cast<uint32_t>(dtype_input_data);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_DATA);

        AddcmulTilingData *tiling = context->GetTilingData<AddcmulTilingData>();
        tiling->totalLength = totalLength;
        tiling->tileNum = 1;
        tiling->tailCoreNum = totalLength % blockDim;

        context->SetBlockDim(blockDim);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *s0 = context->GetInputShape(0);
        const gert::Shape *s1 = context->GetInputShape(1);
        const gert::Shape *s2 = context->GetInputShape(2);
        const gert::Shape *s3 = context->GetInputShape(3);

        size_t d0 = s0->GetDimNum(), d1 = s1->GetDimNum(), d2 = s2->GetDimNum(), d3 = s3->GetDimNum();
        size_t maxD = d0;
        if (d1 > maxD) maxD = d1;
        if (d2 > maxD) maxD = d2;
        if (d3 > maxD) maxD = d3;

        gert::Shape *o = context->GetOutputShape(0);
        o->SetDimNum(maxD);
        for (size_t i = 0; i < maxD; i++) {
            auto g = [](const gert::Shape *s, size_t dn, size_t idx) -> int64_t {
                if (idx >= dn) return 1;
                return s->GetDim(dn - 1 - idx);
            };
            int64_t v = g(s0, d0, i);
            if (g(s1, d1, i) > v) v = g(s1, d1, i);
            if (g(s2, d2, i) > v) v = g(s2, d2, i);
            if (g(s3, d3, i) > v) v = g(s3, d3, i);
            o->SetDim(maxD - 1 - i, v);
        }
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
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