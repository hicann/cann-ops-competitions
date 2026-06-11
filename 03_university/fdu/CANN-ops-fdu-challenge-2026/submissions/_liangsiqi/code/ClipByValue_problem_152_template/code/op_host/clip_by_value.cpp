// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <cmath>

#include "../op_kernel/clip_by_value_tiling.h"
#include "../op_kernel/tiling_key_clip_by_value.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t num_cores_aiv = platform.GetCoreNumAiv();

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t length = tensor_x->GetShapeSize();
        uint32_t esz = (uint32_t)ge::GetSizeByDataType(dtype_x); // fp16=2, fp32/int32=4

        // 读取标量属性 min / max
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        float attr_min = *attrs->GetFloat(0);
        float attr_max = *attrs->GetFloat(1);

        // tiling key 按数据类型区分
        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtype_x));

        // 官方 EleWise tiling: 每核>=4KB, blockFormer 对齐512元素, ubFormer 取 UB 最大(2队列x双缓冲=4)
        uint32_t core_num, block_former, block_num, ub_former;
        if (length == 0) {
            core_num = 1; block_former = 0; block_num = 1; ub_former = 256u / esz;
        } else {
            // 最大化并行(每核>=1个512元素块): 延迟评分下吃满核更快(Lerp 已实测有效)
            core_num = (length + 511u) / 512u;
            if (core_num > num_cores_aiv) core_num = num_cores_aiv;
            const uint32_t ELEM_ALIGN = 512u;
            block_former = ((length + core_num - 1) / core_num + ELEM_ALIGN - 1) / ELEM_ALIGN * ELEM_ALIGN;
            block_num = (length + block_former - 1) / block_former;
            uint32_t ub_max = 147456u / (4u * esz); // fp32/int32=9216, fp16=18432
            uint32_t a256 = 256u / esz;
            ub_former = ub_max / a256 * a256;
            if (ub_former < a256) ub_former = a256;
        }

        ClipByValueTilingData *t = context->GetTilingData<ClipByValueTilingData>();
        t->length = length;
        t->tileLength = ub_former;
        t->coreNum = block_num;
        t->blockEle = block_former; // 每核元素数(blockFormer)
        t->remainder = 0;
        t->minVal = attr_min;
        t->maxVal = attr_max;
        // int32 边界: 对齐 torch.clamp -- float 边界向零截断转 int32 (static_cast), 而非 ceil/floor
        t->minI = static_cast<int32_t>(attr_min);
        t->maxI = static_cast<int32_t>(attr_max);

        context->SetBlockDim(block_num); // 只启动实际有数据的核
        // 预留系统 workspace: profiling/msprof 插桩需要, 报 0 会导致执行拿到空 workspace -> 161001
        context->GetWorkspaceSizes(1)[0] = 16 * 1024 * 1024;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *x = context->GetInputShape(0);
        gert::Shape *y = context->GetOutputShape(0);
        *y = *x;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

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
}  // namespace ops
