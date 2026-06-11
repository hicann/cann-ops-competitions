// y = input + x1 * x2 * value, host侧的tiling + 注册
//
// 核心优化:
//   1. tile从UB容量反推 — 不写死上限, 利用率0.98, tile越大循环轮次越少
//   2. 自适应核数 — 小张量收缩核数, 省的barrier开销比算的都多
//   3. 相邻同广播签名的维度合并 — 减少外层循环层数, 增大连续搬运段
//   4. 快速路径 — 无广播场景跳过stride计算, 各核直接均分连续地址
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/addcmul_tiling.h"
#include "../op_kernel/tiling_key_addcmul.h"

#include <vector>
#include <array>
#include <cstdint>

namespace optiling {

    // 广播对齐: 左侧补1到输出rank, 再取第i维.
    // 例如 [3] 对齐到 rank=3 → [1,1,3], align(0)=1, align(2)=3
    static inline int64_t align(const std::vector<int64_t> &d, size_t target, size_t i) {
        size_t gap = target - d.size();
        return (i < gap) ? 1 : d[i - gap];
    }

    static void getShape(const gert::Tensor *t, std::vector<int64_t> &d, int64_t &sz) {
        const gert::Shape &s = t->GetStorageShape();
        size_t n = s.GetDimNum();
        d.resize(n);
        sz = 1;
        size_t i = 0;
        while (i < n) {
            d[i] = s.GetDim(i);
            sz *= d[i];
            ++i;
        }
    }

    // 根据dtype和UB容量算tile — 从buffer布局反推
    static inline uint32_t calcTile(ge::DataType dt, uint32_t elemBytes, uint64_t ubBytes) {
        // buffer布局和kernel侧Init()严格一致, 否则UB越界.
        //
        // 非int8 (fp16/fp32/int32): 3输入q+1输出q全depth=2, 直接在队列上算.
        //   perElem = 8 * elemBytes
        //
        // int8: 向量引擎没int8算术, 得切float算.
        //   双缓冲int8 io(qIn+qOut) + 3个float buf(bIn/bX1/bX2) + half中转(bHalf)
        //   perElem = 2*1 + 2*1 + 3*4 + 2 = 18
        //
        // 直接求解不设上限 ×0.98留余量 → 256对齐(向量流水线深度) → clamp最小值
        uint32_t perElem = (dt == ge::DT_INT8) ? (2u * 2u + 3u * 4u + 2u) : (8u * elemBytes);
        uint32_t tile = (ubBytes - 32) / perElem;
        tile = static_cast<uint32_t>(tile * 0.98);
        tile = (tile / 256) * 256;
        return (tile < 256) ? 256 : tile;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {

        const gert::Tensor *inp  = context->GetRequiredInputTensor(0);
        const gert::Tensor *tX1  = context->GetRequiredInputTensor(1);
        const gert::Tensor *tX2  = context->GetRequiredInputTensor(2);

        ge::DataType dt = inp->GetDataType();
        uint32_t elemBytes = static_cast<uint32_t>(ge::GetSizeByDataType(dt));

        uint32_t DT_KEY = static_cast<uint32_t>(dt);
        ASCENDC_TPL_SEL_PARAM(context, DT_KEY);

        // 拿shape — 先取inp的, 算着输出rank, 再把x1 x2的也取了
        std::vector<int64_t> dimIn, dim1, dim2;
        int64_t szIn = 1, sz1 = 1, sz2 = 1;
        getShape(inp, dimIn, szIn);

        size_t oRank = dimIn.size();
        getShape(tX1, dim1, sz1);
        if (dim1.size() > oRank) oRank = dim1.size();
        getShape(tX2, dim2, sz2);
        if (dim2.size() > oRank) oRank = dim2.size();
        if (oRank == 0) oRank = 1;

        // 广播输出shape — numpy规则, 每维取max
        std::vector<int64_t> oShape(oRank, 1);
        int64_t total = 1;
        {
            size_t i = 0;
            while (i < oRank) {
                int64_t a = align(dimIn, oRank, i);
                int64_t b = align(dim1,  oRank, i);
                int64_t c = align(dim2,  oRank, i);
                int64_t mx = a;
                if (b > mx) mx = b;
                if (c > mx) mx = c;
                oShape[i] = mx;
                total *= mx;
                ++i;
            }
        }

        // 快速路径判断 — 只需要sz和total, 在tile计算之前就可以判定
        // (但判完不一定走, 还得等tile写完td)
        bool in_sc  = (szIn == 1),  x1_sc  = (sz1 == 1),  x2_sc  = (sz2 == 1);
        bool in_eq  = (szIn == total), x1_eq  = (sz1 == total), x2_eq  = (sz2 == total);
        bool quick = (in_sc || in_eq) && (x1_sc || x1_eq) && (x2_sc || x2_eq);

        AddcmulTilingData *td = context->GetTilingData<AddcmulTilingData>();
        *td = AddcmulTilingData{};

        if (total == 0) {
            td->totalLength = 0;
            td->mode = 0;
            context->SetBlockDim(1);
            size_t *ws = context->GetWorkspaceSizes(1);
            ws[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        td->totalLength = static_cast<uint32_t>(total);

        // ---- 平台信息 + tile求解 ----
        auto plat = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t nCores = plat.GetCoreNumAiv();
        uint64_t ubBytes = 0;
        plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);

        td->tileLength = calcTile(dt, elemBytes, ubBytes);

        constexpr int64_t GRAIN = 2048;
        auto calcNCores = [&](int64_t work) -> uint32_t {
            int64_t need = (work + GRAIN - 1) / GRAIN;
            if (need < 1) need = 1;
            if (need > static_cast<int64_t>(nCores)) need = nCores;
            return static_cast<uint32_t>(need);
        };

        // 多核启动有同步屏障 (AI Core barrier), 开销是常值约数μs.
        // 小张量时计算时间可能远小于barrier → 收缩核数更快.
        // GRAIN=2048: total≤2048塌缩到1核, ~10w以上仍铺满.
        if (quick) {
            td->mode = 0;
            td->scalarX2  = x2_sc  ? 1 : 0;
            td->scalarX1  = x1_sc  ? 1 : 0;
            td->scalarIn  = in_sc  ? 1 : 0;
            context->SetBlockDim(calcNCores(total));
            size_t *ws = context->GetWorkspaceSizes(1);
            ws[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        // ============================================================
        //  广播路径 (mode=1)
        // ============================================================
        // 把输出看作 outerSize × lastDim 二维矩阵:
        //   最内层(lastDim)连续搬运利用向量带宽
        //   外层stride跳转寻址, 广播维stride=0实现自动复制

        // 逐维标bc标记 + 合并相邻同签名维度 — 两步合在一个循环里做
        // 例: [2,1,3,4] 标记(0,0,1)/(1,0,0)/(0,0,0)/(0,0,0) → 后两维合并 → [2,1,12]
        std::vector<int64_t> mgShape;
        std::vector<std::array<int, 3>> mgBc;
        {
            size_t i = 0;
            while (i < oRank) {
                int fIn  = (align(dimIn, oRank, i) == 1) ? 1 : 0;
                int fX1  = (align(dim1,  oRank, i) == 1) ? 1 : 0;
                int fX2  = (align(dim2,  oRank, i) == 1) ? 1 : 0;
                std::array<int, 3> key = {fIn, fX1, fX2};
                if (!mgShape.empty() && mgBc.back() == key) {
                    mgShape.back() *= oShape[i];
                } else {
                    mgShape.push_back(oShape[i]);
                    mgBc.push_back(key);
                }
                ++i;
            }
        }

        size_t nd = mgShape.size();
        int64_t inner = mgShape[nd - 1];
        size_t nOuter = nd - 1;
        if (nOuter > ADDCMUL_MAX_DIM) return ge::GRAPH_FAILED;

        td->mode = 1;
        td->outerSize  = static_cast<uint32_t>(total / inner);
        td->lastDim    = static_cast<uint32_t>(inner);
        td->outerCount = static_cast<uint32_t>(nOuter);

        // 计算strides — 核心不变式: offset = sum(c_i × stride_i)
        // 从最内向外递推, 广播维stride=0使该项自动归零
        //
        // 三个输入的stride独立计算, 顺序无所谓 — 先x2, 再x1, 最后in
        {
            // x2 stride
            auto &lastF = td->lastBcX2;
            auto *st = td->strideX2;
            int64_t step = 1;
            std::vector<int64_t> buf(nd, 0);
            for (int i = static_cast<int>(nd) - 1; i >= 0; i--) {
                int flg = mgBc[i][2];
                buf[i] = flg ? 0 : step;
                step *= (flg ? 1 : mgShape[i]);
            }
            lastF = (mgBc[nd - 1][2] && mgShape[nd - 1] > 1) ? 1u : 0u;
            for (size_t i = 0; i < nOuter; i++) st[i] = static_cast<uint32_t>(buf[i]);
        }
        {
            // x1 stride
            auto &lastF = td->lastBcX1;
            auto *st = td->strideX1;
            int64_t step = 1;
            std::vector<int64_t> buf(nd, 0);
            for (int i = static_cast<int>(nd) - 1; i >= 0; i--) {
                int flg = mgBc[i][1];
                buf[i] = flg ? 0 : step;
                step *= (flg ? 1 : mgShape[i]);
            }
            lastF = (mgBc[nd - 1][1] && mgShape[nd - 1] > 1) ? 1u : 0u;
            for (size_t i = 0; i < nOuter; i++) st[i] = static_cast<uint32_t>(buf[i]);
        }
        {
            // in stride
            auto &lastF = td->lastBcIn;
            auto *st = td->strideIn;
            int64_t step = 1;
            std::vector<int64_t> buf(nd, 0);
            for (int i = static_cast<int>(nd) - 1; i >= 0; i--) {
                int flg = mgBc[i][0];
                buf[i] = flg ? 0 : step;
                step *= (flg ? 1 : mgShape[i]);
            }
            lastF = (mgBc[nd - 1][0] && mgShape[nd - 1] > 1) ? 1u : 0u;
            for (size_t i = 0; i < nOuter; i++) st[i] = static_cast<uint32_t>(buf[i]);
        }

        {
            size_t i = 0;
            while (i < nOuter) {
                td->outerShape[i] = static_cast<uint32_t>(mgShape[i]);
                ++i;
            }
        }

        // 核数交给kernel, kernel侧自己看 outerSize vs nCores 决定按行分还是行内切
        context->SetBlockDim(calcNCores(total));

        size_t *ws = context->GetWorkspaceSizes(1);
        ws[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferDataType(gert::InferDataTypeContext *ctx) {
        ctx->SetOutputDataType(0, ctx->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }

    static graphStatus InferShape(gert::InferShapeContext *ctx) {
        const gert::Shape *a = ctx->GetInputShape(0);
        const gert::Shape *b = ctx->GetInputShape(1);
        const gert::Shape *c = ctx->GetInputShape(2);
        gert::Shape *o = ctx->GetOutputShape(0);
        if (a == nullptr || b == nullptr || c == nullptr || o == nullptr)
            return GRAPH_FAILED;

        size_t ra = a->GetDimNum(), rb = b->GetDimNum(), rc = c->GetDimNum();
        size_t rk = ra;
        if (rb > rk) rk = rb;
        if (rc > rk) rk = rc;
        if (rk == 0) rk = 1;

        auto pad = [](const gert::Shape *s, size_t rk, size_t i) -> int64_t {
            size_t r = s->GetDimNum();
            size_t g = rk - r;
            return (i < g) ? 1 : s->GetDim(i - g);
        };

        o->SetDimNum(rk);
        size_t i = 0;
        while (i < rk) {
            int64_t va = pad(a, rk, i), vb = pad(b, rk, i), vc = pad(c, rk, i);
            int64_t mx = va;
            if (vb > mx) mx = vb;
            if (vc > mx) mx = vc;
            o->SetDim(i, mx);
            ++i;
        }
        return GRAPH_SUCCESS;
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
