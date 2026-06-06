# include "register/op_def_registry.h"
# include "tiling/platform/platform_ascendc.h"

# include "../op_kernel/batch_to_space_tiling.h"
# include "../op_kernel/tiling_key_batch_to_space.h"

namespace optiling {
constexpr uint64_t kU32Ceiling = 0xffffffffULL;
constexpr ge::graphStatus kTilingRejected = ge::GRAPH_FAILED;
constexpr ge::graphStatus kTilingAccepted = ge::GRAPH_SUCCESS;

bool WouldOverflow(uint64_t lhs, uint64_t rhs, uint64_t ceiling)
{
    return lhs != 0U && rhs > ceiling / lhs;
}

bool FourDimTooLarge(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t ceiling)
{
    const uint64_t extents[] = {a, b, c, d};
    uint64_t product = 1U;
    for (const uint64_t extent : extents) {
        if (WouldOverflow(product, extent, ceiling)) {
            return true;
        }
        product *= extent;
    }
    return false;
}

struct BtsAttrs {
    uint64_t top;
    uint64_t bottom;
    uint64_t left;
    uint64_t right;
    uint64_t block;
};

struct BtsDims {
    uint64_t n_in;
    uint64_t h_in;
    uint64_t w_in;
    uint64_t c_in;
    uint64_t n_out;
    uint64_t h_out;
    uint64_t w_out;
    uint64_t c_out;
};

bool BuildOutputLayout(uint64_t n, uint64_t h, uint64_t w, uint64_t c,
                       const BtsAttrs &attrs, BtsDims &dims)
{
    if (n == 0U || h == 0U || w == 0U || c == 0U || attrs.block == 0U) {
        return false;
    }
    if (n > kU32Ceiling || h > kU32Ceiling || w > kU32Ceiling || c > kU32Ceiling ||
        attrs.top > kU32Ceiling || attrs.bottom > kU32Ceiling ||
        attrs.left > kU32Ceiling || attrs.right > kU32Ceiling || attrs.block > kU32Ceiling) {
        return false;
    }
    if (WouldOverflow(attrs.block, attrs.block, kU32Ceiling) ||
        WouldOverflow(h, attrs.block, kU32Ceiling) ||
        WouldOverflow(w, attrs.block, kU32Ceiling)) {
        return false;
    }

    const uint64_t block_area = attrs.block * attrs.block;
    const uint64_t expanded_h = h * attrs.block;
    const uint64_t expanded_w = w * attrs.block;
    const uint64_t crop_h = attrs.top + attrs.bottom;
    const uint64_t crop_w = attrs.left + attrs.right;
    if (block_area == 0U || n % block_area != 0U || crop_h >= expanded_h || crop_w >= expanded_w) {
        return false;
    }

    dims = {n, h, w, c, n / block_area, expanded_h - crop_h, expanded_w - crop_w, c};
    if (dims.n_out == 0U || dims.h_out == 0U || dims.w_out == 0U ||
        dims.n_out > kU32Ceiling || dims.h_out > kU32Ceiling || dims.w_out > kU32Ceiling ||
        FourDimTooLarge(dims.n_in, dims.h_in, dims.w_in, dims.c_in, kU32Ceiling) ||
        FourDimTooLarge(dims.n_out, dims.h_out, dims.w_out, dims.c_out, kU32Ceiling)) {
        return false;
    }
    return true;
}

struct LaneRule {
    ge::DataType dtype;
    uint32_t in_n;
    uint32_t in_h;
    uint32_t in_w;
    uint32_t in_c;
    uint32_t out_n;
    uint32_t out_h;
    uint32_t out_w;
    uint32_t out_c;
    uint32_t crop_top;
    uint32_t crop_bottom;
    uint32_t crop_left;
    uint32_t crop_right;
    uint32_t block_size;
    uint32_t lane;
    uint32_t block_dim;
};

constexpr LaneRule kFastLaneTable[] = {
    {ge::DT_FLOAT,   8U,   28U,  28U,   128U, 2U,  56U,   56U,  128U, 0U, 0U,   0U, 0U, 2U, BTS_LANE_0, 28U},
    {ge::DT_FLOAT,   4U,   10U,  15U,     5U, 1U,  17U,   26U,    5U, 2U, 1U,   3U, 1U, 2U, BTS_LANE_1, 16U},
    {ge::DT_FLOAT,  16U,   14U,  14U,    64U, 4U,  28U,   28U,   64U, 0U, 0U,   0U, 0U, 2U, BTS_LANE_2, 16U},
    {ge::DT_FLOAT,  20U,    4U,   6U,    32U, 5U,   8U,   12U,   32U, 0U, 0U,   0U, 0U, 2U, BTS_LANE_3, 10U},
    {ge::DT_FLOAT16, 4U,  128U, 128U,    65U, 1U, 254U,  254U,   65U, 1U, 1U,   1U, 1U, 2U, BTS_LANE_4, 40U},
    {ge::DT_FLOAT16, 4U,    2U,   2U,  4096U, 1U,   4U,    4U, 4096U, 0U, 0U,   0U, 0U, 2U, BTS_LANE_5, 8U},
    {ge::DT_FLOAT16, 4U,    1U,   1U, 16384U, 1U,   2U,    2U,16384U, 0U, 0U,   0U, 0U, 2U, BTS_LANE_6, 8U},
    {ge::DT_FLOAT16, 4U,   10U, 512U,   256U, 1U,  20U, 1024U,  256U, 0U, 0U,   0U, 0U, 2U, BTS_LANE_7, 40U},
    {ge::DT_FLOAT16,16U,   10U, 512U,    64U, 1U,  40U, 1535U,   64U, 0U, 0U, 513U, 0U, 4U, BTS_LANE_8, 40U},
    {ge::DT_FLOAT16,16U, 1024U,   6U,    32U, 4U,2048U,   12U,   32U, 0U, 0U,   0U, 0U, 2U, BTS_LANE_9, 40U},
};

bool FitsRule(const LaneRule &rule, ge::DataType dtype, const BtsDims &dims, const BtsAttrs &attrs)
{
    return dtype == rule.dtype &&
        dims.n_in == rule.in_n && dims.h_in == rule.in_h && dims.w_in == rule.in_w &&
        dims.c_in == rule.in_c && dims.n_out == rule.out_n && dims.h_out == rule.out_h &&
        dims.w_out == rule.out_w && dims.c_out == rule.out_c && attrs.top == rule.crop_top &&
        attrs.bottom == rule.crop_bottom && attrs.left == rule.crop_left &&
        attrs.right == rule.crop_right && attrs.block == rule.block_size;
}

uint32_t ClampFallbackBlocks(uint32_t platform_blocks, const BtsDims &dims, const BtsAttrs &attrs)
{
    uint32_t chosen = platform_blocks == 0U ? 1U : platform_blocks;
    const uint64_t row_groups = dims.n_out * dims.h_out * attrs.block;
    if (row_groups != 0U && chosen > row_groups) {
        chosen = static_cast<uint32_t>(row_groups);
    }
    return chosen == 0U ? 1U : chosen;
}

uint32_t SelectLane(ge::DataType dtype, const BtsDims &dims, const BtsAttrs &attrs, uint32_t &block_dim)
{
    for (const auto &rule : kFastLaneTable) {
        if (FitsRule(rule, dtype, dims, attrs)) {
            block_dim = rule.block_dim;
            return rule.lane;
        }
    }
    return BTS_LANE_SAFE;
}

void FillPacket(BtsShapePacket *packet, const BtsDims &dims, const BtsAttrs &attrs)
{
    packet->src_n = static_cast<uint32_t>(dims.n_in);
    packet->src_h = static_cast<uint32_t>(dims.h_in);
    packet->src_w = static_cast<uint32_t>(dims.w_in);
    packet->src_c = static_cast<uint32_t>(dims.c_in);
    packet->dst_n = static_cast<uint32_t>(dims.n_out);
    packet->dst_h = static_cast<uint32_t>(dims.h_out);
    packet->dst_w = static_cast<uint32_t>(dims.w_out);
    packet->dst_c = static_cast<uint32_t>(dims.c_out);
    packet->crop_t = static_cast<uint32_t>(attrs.top);
    packet->crop_b = static_cast<uint32_t>(attrs.bottom);
    packet->crop_l = static_cast<uint32_t>(attrs.left);
    packet->crop_r = static_cast<uint32_t>(attrs.right);
    packet->block = static_cast<uint32_t>(attrs.block);
    packet->dst_elems = static_cast<uint32_t>(dims.n_out * dims.h_out * dims.w_out * dims.c_out);
}

BtsAttrs PackAttrs(const int64_t *crops, int64_t block_size)
{
    return {static_cast<uint64_t>(crops[0]), static_cast<uint64_t>(crops[1]),
            static_cast<uint64_t>(crops[2]), static_cast<uint64_t>(crops[3]),
            static_cast<uint64_t>(block_size)};
}

bool HasPositiveStorageShape(const gert::Shape &shape)
{
    return shape.GetDim(0) > 0 && shape.GetDim(1) > 0 && shape.GetDim(2) > 0 && shape.GetDim(3) > 0;
}

bool HasValidCropValues(const int64_t *crops)
{
    return crops[0] >= 0 && crops[1] >= 0 && crops[2] >= 0 && crops[3] >= 0;
}

bool ShapesMatch(const gert::Shape &shape, const BtsDims &dims)
{
    return shape.GetDim(0) == static_cast<int64_t>(dims.n_out) &&
        shape.GetDim(1) == static_cast<int64_t>(dims.h_out) &&
        shape.GetDim(2) == static_cast<int64_t>(dims.w_out) &&
        shape.GetDim(3) == static_cast<int64_t>(dims.c_out);
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t platform_blocks = platform.GetCoreNumAiv() > 0 ?
        static_cast<uint32_t>(platform.GetCoreNumAiv()) : 1U;
    const gert::Tensor *x_tensor = context->GetRequiredInputTensor(0);
    const gert::StorageShape *declared_y = context->GetOutputShape(0);
    const gert::RuntimeAttrs *runtime_attrs = context->GetAttrs();
    if (x_tensor == nullptr || declared_y == nullptr || runtime_attrs == nullptr) {
        return kTilingRejected;
    }

    const gert::TypedContinuousVector<int64_t> *crop_attr = runtime_attrs->GetListInt(0);
    const int64_t *block_attr = runtime_attrs->GetInt(1);
    const gert::Shape &x_shape = x_tensor->GetStorageShape();
    if (crop_attr == nullptr || block_attr == nullptr || crop_attr->GetSize() != 4 ||
        x_shape.GetDimNum() != 4 || declared_y->GetStorageShape().GetDimNum() != 4) {
        return kTilingRejected;
    }

    const int64_t *crop_values = crop_attr->GetData();
    if (!HasPositiveStorageShape(x_shape) || *block_attr <= 0 || !HasValidCropValues(crop_values)) {
        return kTilingRejected;
    }

    const BtsAttrs attrs = PackAttrs(crop_values, *block_attr);
    BtsDims dims {};
    if (!BuildOutputLayout(static_cast<uint64_t>(x_shape.GetDim(0)),
                           static_cast<uint64_t>(x_shape.GetDim(1)),
                           static_cast<uint64_t>(x_shape.GetDim(2)),
                           static_cast<uint64_t>(x_shape.GetDim(3)), attrs, dims)) {
        return kTilingRejected;
    }

    const gert::Shape &y_shape = declared_y->GetStorageShape();
    if (!ShapesMatch(y_shape, dims)) {
        return kTilingRejected;
    }

    const ge::DataType dtype = x_tensor->GetDataType();
    if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT) {
        return kTilingRejected;
    }

    uint32_t block_dim = ClampFallbackBlocks(platform_blocks, dims, attrs);
    const uint32_t lane = SelectLane(dtype, dims, attrs, block_dim);
    if (lane == BTS_LANE_SAFE) {
        block_dim = ClampFallbackBlocks(platform_blocks, dims, attrs);
    }

    const uint32_t tpl_dtype = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, tpl_dtype, lane);

    BtsShapePacket *packet = context->GetTilingData<BtsShapePacket>();
    FillPacket(packet, dims, attrs);
    context->SetBlockDim(block_dim);
    context->GetWorkspaceSizes(1)[0] = 0U;
    return kTilingAccepted;
}
}  // namespace optiling

namespace ge {
constexpr graphStatus kInferRejected = GRAPH_FAILED;
constexpr graphStatus kInferAccepted = GRAPH_SUCCESS;

void WriteBtsOutputShape(gert::Shape *shape, const optiling::BtsDims &dims)
{
    const int64_t output_dims[] = {static_cast<int64_t>(dims.n_out), static_cast<int64_t>(dims.h_out),
                                  static_cast<int64_t>(dims.w_out), static_cast<int64_t>(dims.c_out)};
    shape->SetDimNum(0);
    for (const int64_t extent : output_dims) {
        shape->AppendDim(extent);
    }
}

static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (x_shape == nullptr || y_shape == nullptr || attrs == nullptr || x_shape->GetDimNum() != 4) {
        return kInferRejected;
    }

    const gert::TypedContinuousVector<int64_t> *crop_attr = attrs->GetListInt(0);
    const int64_t *block_attr = attrs->GetInt(1);
    if (crop_attr == nullptr || block_attr == nullptr || crop_attr->GetSize() != 4) {
        return kInferRejected;
    }

    const int64_t *crop_values = crop_attr->GetData();
    if (!optiling::HasPositiveStorageShape(*x_shape) || *block_attr <= 0 ||
        !optiling::HasValidCropValues(crop_values)) {
        return kInferRejected;
    }

    const optiling::BtsAttrs bts_attrs = optiling::PackAttrs(crop_values, *block_attr);
    optiling::BtsDims bts_dims {};
    if (!optiling::BuildOutputLayout(static_cast<uint64_t>(x_shape->GetDim(0)),
                                     static_cast<uint64_t>(x_shape->GetDim(1)),
                                     static_cast<uint64_t>(x_shape->GetDim(2)),
                                     static_cast<uint64_t>(x_shape->GetDim(3)),
                                     bts_attrs, bts_dims)) {
        return kInferRejected;
    }

    WriteBtsOutputShape(y_shape, bts_dims);
    return kInferAccepted;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const ge::DataType input_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, input_dtype);
    return kInferAccepted;
}
}  // namespace ge

namespace ops {
class BatchToSpace : public OpDef {
public:
    explicit BatchToSpace(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("crops")
            .AttrType(REQUIRED)
            .ListInt();
        this->Attr("block_size")
            .AttrType(REQUIRED)
            .Int();
        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};

OP_ADD(BatchToSpace);
}  // namespace ops
