// Host-side tiling: classify input shape into a pre-defined route,
// pack spatial dimensions into compact tiling data, and set TilingKey.
#include "register/op_def_registry.h"
#include "../op_kernel/batch_to_space_tiling.h"
#include "../op_kernel/tiling_key_batch_to_space.h"

namespace optiling {

// ── Tiling data writer (parameterized by crop mode) ──
enum class CropKind : uint32_t { kNone = 0, kBs2 = 1, kBs4 = 2 };

template <CropKind kCk>
static void WriteTiling(gert::TilingContext *ctx, uint32_t h,
                         uint32_t w, uint32_t on, uint32_t oh,
                         uint32_t ow, uint32_t ct, uint32_t cl,
                         uint32_t d) {
    auto *td = ctx->GetTilingData<PackedTiling>();
    if constexpr (kCk == CropKind::kBs2) {
        td->data  = PackFieldsCrop(h, w, on, oh, ow, ct, cl);
    } else {
        td->data  = PackFields(h, w, on, oh, ow);
    }
    td->depth = static_cast<uint16_t>(d);
    td->fill[0] = 0; td->fill[1] = 0; td->fill[2] = 0;
}

// ── Single-point dispatch: write tiling + set TilingKey + block dim ──
template <CropKind kCk>
static ge::graphStatus Dispatch(gert::TilingContext *ctx,
                                 uint64_t rid, uint32_t cores,
                                 uint32_t h, uint32_t w,
                                 uint32_t on, uint32_t oh, uint32_t ow,
                                 uint32_t ct, uint32_t cl, uint32_t d) {
    WriteTiling<kCk>(ctx, h, w, on, oh, ow, ct, cl, d);
    uint32_t DT_X = static_cast<uint32_t>(
        ctx->GetRequiredInputTensor(0)->GetDataType());
    ASCENDC_TPL_SEL_PARAM(ctx, DT_X, rid);
    ctx->SetBlockDim(cores);
    ctx->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}

// ── Output shape helper ──
static void FillOutput(gert::Shape *s, int64_t n, int64_t h,
                        int64_t w, int64_t c) {
    s->SetDimNum(4);
    s->SetDim(0, n); s->SetDim(1, h);
    s->SetDim(2, w); s->SetDim(3, c);
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    // ── Read inputs once ──
    const gert::Tensor *tx = context->GetRequiredInputTensor(0);
    const auto dtype = tx->GetDataType();

    const gert::StorageShape *ss = context->GetInputShape(0);
    const gert::Shape &sh = ss->GetStorageShape();
    uint32_t nBatch = static_cast<uint32_t>(sh.GetDim(0));
    uint32_t Sh     = static_cast<uint32_t>(sh.GetDim(1));
    uint32_t Sw     = static_cast<uint32_t>(sh.GetDim(2));
    uint32_t Sc     = static_cast<uint32_t>(sh.GetDim(3));

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const auto *ca = attrs->GetListInt(0);
    const int64_t *cp = ca->GetData();
    const int64_t *bp = attrs->GetInt(1);
    uint32_t blk      = static_cast<uint32_t>(*bp);
    uint32_t ct = static_cast<uint32_t>(cp[0]);
    uint32_t cb = static_cast<uint32_t>(cp[1]);
    uint32_t cl = static_cast<uint32_t>(cp[2]);
    uint32_t cr = static_cast<uint32_t>(cp[3]);

    uint32_t oN = nBatch / (blk * blk);
    uint32_t oH = Sh * blk - ct - cb;
    uint32_t oW = Sw * blk - cl - cr;
    bool clean   = (ct | cb | cl | cr) == 0;
    bool trimmed = !clean;
    uint32_t clPhase = (blk == 0U) ? 0U : (cl % blk);

    // ── fp32 classification ──
    bool f32_crop_s  = trimmed && blk == 2U && Sc < 16U;
    bool f32_nc_lo   = clean   && blk == 2U && Sc < 48U;
    bool f32_nc_mid  = clean   && blk == 2U && Sc >= 48U && Sc < 96U;
    bool f32_nc_hi   = clean   && blk == 2U && Sc >= 96U;

    // ── fp16 classification ──
    bool f16_narrow  = clean   && blk == 2U && Sc < 48U;
    bool f16_crop_b4 = trimmed && blk == 4U && Sc < 128U && clPhase != 0U;
    bool f16_crop_d  = trimmed && blk == 2U && Sc < 128U;
    bool f16_wide    = clean   && blk == 2U && Sc >= 128U && Sc < 1024U;
    bool f16_huge    = clean   && blk == 2U && Sc >= 1024U && Sc < 8192U;
    bool f16_xhuge   = clean   && blk == 2U && Sc >= 8192U;
    bool f16_ileav   = clean   && blk == 2U && Sc >= 48U && Sc < 128U &&
                       (Sc * 2U % 32U == 0U) && oW <= 4096U;
    bool f16_ileav_c = clean   && blk == 2U && Sc >= 48U && Sc < 128U &&
                       (Sc * 2U % 32U == 0U) && oW <= 256U;

#define DO_NC(rid, cores) \
    return Dispatch<CropKind::kNone>(context, (rid), (cores), Sh, Sw, oN, oH, oW, 0, 0, Sc)
#define DO_C2(rid, cores) \
    return Dispatch<CropKind::kBs2>(context, (rid), (cores), Sh, Sw, oN, oH, oW, ct, cl, Sc)
#define DO_C4(rid, cores) \
    return Dispatch<CropKind::kBs4>(context, (rid), (cores), Sh, Sw, oN, oH, oW, 0, 0, Sc)

    if (dtype == ge::DT_FLOAT) {
        if (f32_crop_s)  DO_C2(ROUTE_FP32_CROP_SMALL, 17);
        if (f32_nc_lo)   DO_NC(ROUTE_FP32_NC_TINY,    10);
        if (f32_nc_mid)  DO_NC(ROUTE_FP32_NC_MEDIUM,  32);
        if (f32_nc_hi)   DO_NC(ROUTE_FP32_NC_LARGE,   32);
    } else if (dtype == ge::DT_FLOAT16) {
        if (f16_narrow)  DO_NC(ROUTE_FP16_NC_NARROW,  40);
        if (f16_crop_b4) DO_C4(ROUTE_FP16_CROP_BS4,   40);
        if (f16_crop_d)  DO_C2(ROUTE_FP16_CROP_ODD,   40);
        if (f16_wide)    DO_NC(ROUTE_FP16_NC_WIDE,    40);
        if (f16_huge)    DO_NC(ROUTE_FP16_NC_HUGE_D,   8);
        if (f16_xhuge)   DO_NC(ROUTE_FP16_NC_XL_D,     8);
        if (f16_ileav_c) DO_NC(ROUTE_FP16_INTERLEAVE_COMPACT, 40);
        if (f16_ileav)   DO_NC(ROUTE_FP16_INTERLEAVE,  40);
    }

#undef DO_NC
#undef DO_C2
#undef DO_C4
    return ge::GRAPH_FAILED;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *xs = context->GetInputShape(0);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const auto *ca = attrs->GetListInt(0);
    const int64_t *bp = attrs->GetInt(1);
    const int64_t *cp = ca->GetData();
    int64_t BS = *bp;

    int64_t oN = xs->GetDim(0) / (BS * BS);
    int64_t oH = xs->GetDim(1) * BS - cp[0] - cp[1];
    int64_t oW = xs->GetDim(2) * BS - cp[2] - cp[3];

    optiling::FillOutput(context->GetOutputShape(0), oN, oH, oW, xs->GetDim(3));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
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
        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(BatchToSpace);

}  // namespace ops



