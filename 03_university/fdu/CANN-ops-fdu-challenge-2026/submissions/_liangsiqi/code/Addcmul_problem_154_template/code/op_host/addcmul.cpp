// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/addcmul_tiling.h"
#include "../op_kernel/tiling_key_addcmul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t num_cores_aiv = platform.GetCoreNumAiv();
        ge::DataType dtype = context->GetRequiredInputTensor(0)->GetDataType();
        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtype));

        const int N = ADDCMUL_MAX_DIM;
        // 三个输入右对齐补 1 到 N 维, 输出取逐维 max(广播)
        const gert::Shape *sh[3] = {
            &context->GetRequiredInputTensor(0)->GetOriginShape(),
            &context->GetRequiredInputTensor(1)->GetOriginShape(),
            &context->GetRequiredInputTensor(2)->GetOriginShape()};
        uint32_t d[3][N], y[N];
        for (int k = 0; k < 3; k++) {
            int r = sh[k]->GetDimNum();
            for (int i = 0; i < N; i++) {
                int src = i - (N - r);
                d[k][i] = (src >= 0) ? (uint32_t)sh[k]->GetDim(src) : 1;
            }
        }
        for (int i = 0; i < N; i++) {
            y[i] = d[0][i]; if (d[1][i] > y[i]) y[i] = d[1][i]; if (d[2][i] > y[i]) y[i] = d[2][i];
        }
        // 每输入各维元素步长, 广播维(dim==1<y)步长记 0
        uint32_t st[3][N];
        for (int k = 0; k < 3; k++) {
            uint32_t acc = 1;
            for (int i = N - 1; i >= 0; i--) {
                st[k][i] = (d[k][i] == 1 && y[i] != 1) ? 0u : acc;
                acc *= d[k][i];
            }
        }
        uint32_t total = 1; for (int i = 0; i < N; i++) total *= y[i];
        uint32_t lastDim = y[N - 1] == 0 ? 1 : y[N - 1];
        uint32_t rows = (lastDim == 0) ? 0 : total / lastDim;
        uint32_t bcast = (d[0][N-1]==y[N-1]&&d[1][N-1]==y[N-1]&&d[2][N-1]==y[N-1]) ? 0u : 1u;
        for (int i = 0; i < N - 1 && bcast == 0; i++) if (d[0][i]!=y[i]||d[1][i]!=y[i]||d[2][i]!=y[i]) bcast = 1;

        const uint32_t align = 256;
        // 按 dtype 调大 tile, 目标单次搬运尽量逼近 16KB+ 以吃满 DMA 带宽(UB 统一约 180KB):
        //   fp32/int32: 5120 -> 20KB; fp16: 7680 -> 15KB(去 halfBuf 腾空间); int8: 10240 -> 10KB(1字节元素受限)
        uint32_t max_tile;
        if (dtype == ge::DT_FLOAT16) max_tile = 7680;
        else if (dtype == ge::DT_INT8) max_tile = 10240;
        else max_tile = 5120; // fp32 / int32
        uint32_t tile_length = (lastDim + align - 1) / align * align;
        if (tile_length > max_tile) tile_length = max_tile;
        uint32_t core_num = num_cores_aiv;
        if (rows == 0) core_num = 1; else if (rows < core_num) core_num = rows;
        AddcmulTilingData *t = context->GetTilingData<AddcmulTilingData>();
        t->totalLength = total; t->lastDim = static_cast<uint16_t>(lastDim); t->tileLength = tile_length;
        t->blockRows = rows / core_num; t->remainder = static_cast<uint8_t>(rows % core_num); t->bcast = static_cast<uint8_t>(bcast);
        for (int i = 0; i < N; i++) { t->yDim[i]=static_cast<uint16_t>(y[i]); t->s0[i]=st[0][i]; t->s1[i]=st[1][i]; t->s2[i]=st[2][i]; }
        t->last0 = static_cast<uint16_t>(d[0][N-1]); t->last1 = static_cast<uint16_t>(d[1][N-1]); t->last2 = static_cast<uint16_t>(d[2][N-1]);
        uint32_t block_dim = core_num;
        if (bcast == 0) {
            // 同形无广播: 塌成 1D 扁平 elementwise; 512 块余数均摊负载均衡(吃满核 + 核间差<=1块)
            uint32_t bn, bpc, rem;
            if (total == 0) { bn = 1; bpc = 0; rem = 0; }
            else {
                uint32_t total_blocks = (total + 511u) / 512u;   // 512 元素块数
                bn = (total_blocks < num_cores_aiv) ? total_blocks : num_cores_aiv; // 吃满核
                bpc = total_blocks / bn;       // 每核基础块数
                rem = total_blocks % bn;       // 前 rem 核各多 1 块(均摊尾块)
            }
            t->blockRows = bpc;       // 复用: 扁平路径下 = 每核基础块数(512元素块)
            t->remainder = static_cast<uint8_t>(rem);  // 前 rem 核各多 1 块
            t->tileLength = max_tile; // 扁平用满 tile(不再受 lastDim 限制)
            t->last0 = 2; t->last1 = 2; t->last2 = 2; // 哨兵: 非1 -> 关闭行内广播/Duplicate
            block_dim = bn;
        }
        context->SetBlockDim(block_dim);
        // 预留系统 workspace, 报 0 在 profiling 下可能触发 161001
        context->GetWorkspaceSizes(1)[0] = 16 * 1024 * 1024;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // 输出形状 = input_data / x1 / x2 的逐维 max(NumPy 广播)
        const gert::Shape *a = context->GetInputShape(0);
        const gert::Shape *b = context->GetInputShape(1);
        const gert::Shape *c = context->GetInputShape(2);
        gert::Shape *y = context->GetOutputShape(0);
        size_t rank = a->GetDimNum();
        if (b->GetDimNum() > rank) rank = b->GetDimNum();
        if (c->GetDimNum() > rank) rank = c->GetDimNum();
        y->SetDimNum(rank);
        for (size_t i = 0; i < rank; i++) {
            int64_t da = i + a->GetDimNum() >= rank ? a->GetDim(i + a->GetDimNum() - rank) : 1;
            int64_t db = i + b->GetDimNum() >= rank ? b->GetDim(i + b->GetDimNum() - rank) : 1;
            int64_t dc = i + c->GetDimNum() >= rank ? c->GetDim(i + c->GetDimNum() - rank) : 1;
            int64_t m = da; if (db > m) m = db; if (dc > m) m = dc;
            y->SetDim(i, m);
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
