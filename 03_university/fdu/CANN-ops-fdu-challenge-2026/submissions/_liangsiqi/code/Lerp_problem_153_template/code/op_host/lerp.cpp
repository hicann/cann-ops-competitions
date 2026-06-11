// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <cmath>

#include "../op_kernel/lerp_tiling.h"
#include "../op_kernel/tiling_key_lerp.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t num_cores_aiv = platform.GetCoreNumAiv();

        const gert::Tensor *tensor_start = context->GetRequiredInputTensor(0);
        ge::DataType dtype_start = tensor_start->GetDataType();
        uint32_t length = tensor_start->GetShapeSize();
        uint32_t esz = (uint32_t)ge::GetSizeByDataType(dtype_start); // fp16=2, fp32=4

        // 读取标量属性 weight
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        float attr_weight = *attrs->GetFloat(0);

        // tiling key 按数据类型区分
        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtype_start));

        // 多核切分: 512 元素块为单位, 余数均摊到前 rem 个核(负载均衡, 消除尾核空转/空闲核)
        uint32_t block_num, ub_former, bpc, rem;
        if (length == 0) {
            block_num = 1; bpc = 0; rem = 0; ub_former = 256u / esz;
        } else {
            uint32_t total_blocks = (length + 511u) / 512u;   // 512 元素块数(末块可能不满)
            // 最大化并行: 吃满核, 每核至少 1 块
            block_num = (total_blocks < num_cores_aiv) ? total_blocks : num_cores_aiv;
            bpc = total_blocks / block_num;       // 每核基础块数
            rem = total_blocks % block_num;       // 前 rem 个核各多 1 块(均摊尾块, 核间差<=1块)
            // ubFormer 取 UB 最大(每核单个大 tile, 实测最优; 切多 tile 反而慢)
            uint32_t ub_max = 184320u / (6u * esz); // 无额外占用, ~180KB: fp32=7680, fp16=15360
            uint32_t a256 = 256u / esz;
            ub_former = ub_max / a256 * a256;
            if (ub_former < a256) ub_former = a256;
        }

        LerpTilingData *t = context->GetTilingData<LerpTilingData>();
        t->length = length;
        t->tileLength = ub_former;
        t->coreNum = block_num;
        t->blockEle = bpc;   // 每核基础块数(512元素块)
        t->remainder = rem;  // 前 rem 个核各多 1 块
        t->weight = attr_weight;
        // 对齐 torch.lerp: |w|>=0.5 用数值稳定式
        t->useStable = (std::fabs(attr_weight) < 0.5f) ? 0u : 1u;

        context->SetBlockDim(block_num); // 只启动实际有数据的核
        // 预留系统 workspace, 报 0 在 profiling 下可能触发 161001
        context->GetWorkspaceSizes(1)[0] = 16 * 1024 * 1024;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *s = context->GetInputShape(0);
        gert::Shape *y = context->GetOutputShape(0);
        *y = *s;
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

