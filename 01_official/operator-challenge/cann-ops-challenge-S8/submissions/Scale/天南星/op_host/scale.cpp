#include "scale_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <vector>
#include <algorithm>
template<typename T>
constexpr T ceil_div(T x, T y) { return (x - 1) / y + 1; }

template<typename T>
constexpr T ceil_round(T x, T y) { return ceil_div(x, y) * y; }

static void GetTensorShape(const gert::StorageShape* shape,
                           std::vector<uint64_t>& inshapeVector,
                           uint32_t outDimNum) {
    int n = outDimNum;
    for (int j = shape->GetStorageShape().GetDimNum() - 1; j >= 0; --j) {
        inshapeVector[--n] = shape->GetStorageShape().GetDim(j);
    }
}

static void GetBroadCastParams(std::vector<uint64_t> inshapeVector1,
                               std::vector<uint64_t> inshapeVector2,
                               std::vector<uint64_t> inshapeVector3,
                               gert::Shape outshape,
                               uint32_t* shape_out,
                               uint32_t* reduce1_out,
                               uint32_t* reduce2_out,
                               uint32_t* reduce3_out,
                               uint32_t& dim_out) {
    uint32_t ndim = outshape.GetDimNum();

    std::vector<uint32_t> sh(ndim), r1(ndim), r2(ndim), r3(ndim);
    for (uint32_t i = 0; i < ndim; i++) {
        sh[i] = outshape.GetDim(i);
        r1[i] = (inshapeVector1[i] != (uint64_t)outshape.GetDim(i)) ? 1 : 0;
        r2[i] = (inshapeVector2[i] != (uint64_t)outshape.GetDim(i)) ? 1 : 0;
        r3[i] = (inshapeVector3[i] != (uint64_t)outshape.GetDim(i)) ? 1 : 0;
    }

    std::vector<uint32_t> stk_sh, stk_r1, stk_r2, stk_r3;
    for (int i = (int)ndim - 1; i >= 0; i--) {
        if (!stk_sh.empty() &&
            r1[i] == 0 && stk_r1.back() == 0 &&
            r2[i] == 0 && stk_r2.back() == 0 &&
            r3[i] == 0 && stk_r3.back() == 0) {
            stk_sh.back() *= sh[i];
        } else {
            stk_sh.push_back(sh[i]);
            stk_r1.push_back(r1[i]);
            stk_r2.push_back(r2[i]);
            stk_r3.push_back(r3[i]);
        }
    }

    std::reverse(stk_sh.begin(), stk_sh.end());
    std::reverse(stk_r1.begin(), stk_r1.end());
    std::reverse(stk_r2.begin(), stk_r2.end());
    std::reverse(stk_r3.begin(), stk_r3.end());

    dim_out = (uint32_t)stk_sh.size();
    for (uint32_t i = 0; i < dim_out; i++) {
        shape_out[i]   = stk_sh[i];
        reduce1_out[i] = stk_r1[i];
        reduce2_out[i] = stk_r2[i];
        reduce3_out[i] = stk_r3[i];
    }
}
struct TillingLocalParams {
    uint32_t start_length  = 0;
    uint32_t end_length    = 0;
    uint32_t weight_length = 0;  // bias=None时为0
    uint32_t total_length  = 0;
    uint32_t ALIGN_NUM     = 0;
    uint32_t tiling_size   = 0;
    uint32_t block_size    = 0;
    uint32_t core_size     = 0;
    uint32_t core_remain   = 0;
    //uint32_t mode          = 0;
    int      aivNum        = 1;
    uint32_t has_bias      = 0;
};
static void GetTillingParams(gert::TilingContext* context,
                             TillingLocalParams& p,
                             bool has_bias) {
    uint32_t sizeofdatatype;
    uint64_t ub_size;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    p.aivNum = ascendcPlatform.GetCoreNum();

    p.total_length = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    p.start_length = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    p.end_length   = context->GetInputShape(1)->GetStorageShape().GetShapeSize();

    // ★ 关键修改：bias=None时weight_length强制为0，不读占位tensor的size
    if (has_bias) {
        p.weight_length = context->GetInputShape(2)->GetStorageShape().GetShapeSize();
    } else {
        p.weight_length = 0;
    }

    // if (p.weight_length != 1) {
    //     p.mode = 1;
    // }
    int32_t NUM = 12;
    auto dt = context->GetInputDesc(0)->GetDataType();
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
        sizeofdatatype = 2;

        NUM =10;

        
        if (has_bias){

            NUM  =14;

        }
    } else {
        sizeofdatatype = 4;

         NUM =6;

        
        if (has_bias){

            NUM  =8;

        }
    }

    uint32_t BLOCK_SIZE =32;

    if (dt == ge::DT_BF16){

        BLOCK_SIZE =512;
    }
    

    p.ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;

    const int UB_DIV_COEF = 2;
    p.tiling_size = ((ub_size) / BLOCK_SIZE ) / NUM;
    p.tiling_size = p.tiling_size <= 8 ? p.tiling_size : p.tiling_size / 8 * 8;
    p.block_size  = p.tiling_size * p.ALIGN_NUM;

    p.aivNum = (p.aivNum < (int)(p.total_length / p.block_size)) ?
                p.aivNum : (int)(p.total_length / p.block_size);
    p.aivNum = p.aivNum >= 1 ? p.aivNum : 1;

    p.core_size   = (p.total_length / p.aivNum) / (p.ALIGN_NUM * 8) * (p.ALIGN_NUM * 8);
    p.core_remain = p.total_length - p.aivNum * p.core_size;
}

namespace optiling
{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        // ── 1. 读取输入 shape ──────────────────────────────
        auto *shape_x     = context->GetInputShape(0);
        auto *shape_scale = context->GetInputShape(1);
        auto *shape_bias  = context->GetInputShape(2);



        int64_t axis       = *context->GetAttrs()->GetInt(0);
        int64_t num_axes   = *context->GetAttrs()->GetInt(1);
        bool scaleFromBlob = *context->GetAttrs()->GetBool(2);


        // ═════════ DEBUG START ═════════
        auto* tensor_bias = context->GetInputTensor(2);

        int outDimNum = shape_x->GetStorageShape().GetDimNum();
        bool has_bias = (shape_bias != nullptr &&
                         shape_bias->GetStorageShape().GetShapeSize() > 0);

        // ── 2. 读取属性 ────────────────────────────────────
  

        if (axis < 0) axis += outDimNum;

        int scale_rank = shape_scale->GetStorageShape().GetDimNum();
        int bias_rank  = has_bias ?
                         shape_bias->GetStorageShape().GetDimNum() : 0;

        // ── 3. 计算 start_axis / end_axis ─────────────────
        int start_axis, end_axis;
        if (scaleFromBlob) {
            start_axis = (int)axis;
            end_axis   = (num_axes == -1) ?
                         (outDimNum - 1) :
                         (int)(axis + num_axes - 1);
        } else {
            start_axis = (int)axis;
            end_axis   = (int)(axis + scale_rank - 1);
        }
        if (end_axis >= outDimNum) end_axis = outDimNum - 1;

        auto dtype        = context->GetInputTensor(0)->GetDataType();
        int64_t elem_size = ge::GetSizeByDataType(dtype);

        // ── 4. 三段式 reshape：[high][mid][low] ───────────
        // low  = prod(x.dims[end_axis+1 : outDimNum])
        // mid  = prod(x.dims[start_axis : end_axis+1])
        // high = prod(x.dims[0          : start_axis])
        int64_t low_size  = 1;
        int64_t mid_size  = 1;
        int64_t high_size = 1;

        for (int i = end_axis + 1; i < outDimNum; i++)
            low_size *= shape_x->GetStorageShape().GetDim(i);
        for (int i = start_axis; i <= end_axis; i++)
            mid_size *= shape_x->GetStorageShape().GetDim(i);
        for (int i = 0; i < start_axis; i++)
            high_size *= shape_x->GetStorageShape().GetDim(i);

        int64_t full_size = high_size * mid_size * low_size;

        // ── 5. 计算 scale 实际元素数 ──────────────────────
        int64_t scale_mid = 1;
        for (int i = 0; i < scale_rank; i++)
            scale_mid *= shape_scale->GetStorageShape().GetDim(i);

        // ── 6. 判断 stype ──────────────────────────────────
        // stype=0：完全 flat，无广播
        //   high==1 && scale_mid==mid_size && low==1
        //
        // stype=2：channel-wise 快路径
        //   scale_mid==mid_size（scale 完整覆盖 mid 维）
        //   && low_size >= min_block_elems（最小 DMA 单元）
        //   && low_size * elem_size % 32 == 0（32B 对齐）
        //
        // stype=1：其余情况，workspace 展开后 flat

        int64_t min_block_elems = 32 / elem_size;
        bool low_valid = (low_size >= min_block_elems);

        int stype;
        if (high_size == 1 && scale_mid == mid_size && low_size == 1) {
            stype = 0;
        } else if (scale_mid == mid_size && low_valid) {
            stype = 2;
        } else {
            stype = 1;
        }

        // ── 7. 计算传给 Kernel 的核心 tiling 参数 ─────────
        int tiling_size, tiling_high, tiling_low;
        if (stype == 2) {
            tiling_size = (int)mid_size;
            tiling_high = (int)high_size;
            tiling_low  = (int)low_size;
        } else {
            tiling_size = (int)full_size;
            tiling_high = 1;
            tiling_low  = 1;
        }

        // ── 8. 构造 5D shape 供 stype=1 的 do_broadcast 使用 ──
        // 低维在前的 5D 表示（outDimNum 最大为 5，完整覆盖）
        auto to_5d = [&](const std::vector<int64_t> &shape_vec) {
            int rank = (int)shape_vec.size();
            std::vector<int> dims(5, 1);
            for (int i = 0; i < rank && i < 5; i++)
                dims[i] = (int)shape_vec[rank - 1 - i];
            return dims;
        };

        std::vector<int64_t> x_shape_vec(outDimNum);
        for (int i = 0; i < outDimNum; i++)
            x_shape_vec[i] = shape_x->GetStorageShape().GetDim(i);

        // 把 scale/bias shape 对齐到 x 的维度空间
        std::vector<int64_t> broadcast_scale(outDimNum, 1);
        for (int i = 0; i < scale_rank; i++) {
            int target = start_axis + i;
            if (target < outDimNum)
                broadcast_scale[target] =
                    shape_scale->GetStorageShape().GetDim(i);
        }
        std::vector<int64_t> broadcast_bias(outDimNum, 1);
        if (has_bias) {
            for (int i = 0; i < bias_rank; i++) {
                int target = start_axis + i;
                if (target < outDimNum)
                    broadcast_bias[target] =
                        shape_bias->GetStorageShape().GetDim(i);
            }
        }

        auto input_dims = to_5d(x_shape_vec);
        auto scale_dims = to_5d(broadcast_scale);
        auto bias_dims  = to_5d(broadcast_bias);
        std::vector<int> out_dims(5);
        for (int i = 0; i < 5; i++)
            out_dims[i] = std::max({input_dims[i],
                                    scale_dims[i],
                                    bias_dims[i]});

        // ── 9. workspace offset（仅 stype=1 需要）────────
        long single_size = ceil_round(
            (long)full_size * (long)elem_size, 512L);

        std::vector<long> broadcast_offset(3, -1L);
        long next_offset = 0;

        if (stype == 1) {
            if (scale_dims != out_dims) {
                broadcast_offset[1] = next_offset;
                next_offset += single_size;
            }
            if (has_bias && bias_dims != out_dims) {
                broadcast_offset[2] = next_offset;
                next_offset += single_size;
            }
        }

        int32_t NUM = 12;
        auto dt = context->GetInputDesc(0)->GetDataType();

        uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        uint32_t endLength   = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
        if (dt == ge::DT_FLOAT && totalLength >=4096 && totalLength == endLength ) {

            ScaleTilingDataFloat tiling;

            TillingLocalParams p;

            p.has_bias = has_bias ? 1 : 0;
            GetTillingParams(context, p, has_bias);

            tiling.set_start_length(p.start_length);
            tiling.set_end_length(p.end_length);
            tiling.set_weight_length(p.weight_length);
            tiling.set_total_length(p.total_length);
            tiling.set_ALIGN_NUM(p.ALIGN_NUM);
            tiling.set_tiling_size(p.tiling_size);
            tiling.set_block_size(p.block_size);
            tiling.set_core_size(p.core_size);
            tiling.set_core_remain(p.core_remain);
            //tiling.set_mode(p.mode);
            tiling.set_has_bias(p.has_bias);

            context->SetBlockDim(p.aivNum);


             tiling.SaveToBuffer(
                context->GetRawTilingData()->GetData(),
                context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

            // ── 11. block dim 和 tiling key ───────────────────
            auto platform  = platform_ascendc::PlatformAscendC(
                                context->GetPlatformInfo());
            auto block_dim = platform.GetCoreNumAiv();
            context->SetBlockDim(block_dim);


            context->SetTilingKey(13);



        }

        else if ((dt == ge::DT_FLOAT && totalLength >=4096 && totalLength != endLength ) || (dt == ge::DT_BF16 && totalLength >=256 && totalLength != endLength )  ) {

            ScaleTilingDataFloatBroadCast tiling;

            uint32_t local_shape[20]   = {0};
            uint32_t local_reduce1[20] = {0};
            uint32_t local_reduce2[20] = {0};
            uint32_t local_reduce3[20] = {0};
            uint32_t local_dim = 0;

            auto outshape      = context->GetOutputShape(0)->GetOriginShape();
            uint32_t outDimNum = outshape.GetDimNum();

            const gert::StorageShape* shape1 = context->GetInputShape(0);
            const gert::StorageShape* shape2 = context->GetInputShape(1);
            const gert::StorageShape* shape3 = context->GetInputShape(2);

            std::vector<uint64_t> inshapeVector1(outDimNum, 1);
            std::vector<uint64_t> inshapeVector2(outDimNum, 1);
            std::vector<uint64_t> inshapeVector3(outDimNum, 1);

            GetTensorShape(shape1, inshapeVector1, outDimNum);

            int64_t axis          = *context->GetAttrs()->GetInt(0);
            int64_t num_axes      = *context->GetAttrs()->GetInt(1);
            bool    scaleFromBlob = *context->GetAttrs()->GetBool(2);





            if (axis < 0) axis += (int64_t)outDimNum;

            int scale_rank = shape2->GetStorageShape().GetDimNum();

            // ★ 关键修改：has_bias 判断
            // bias=REQUIRED时框架用scale的tensor填占位，因此需要额外判断：
            // shape3和shape2的指针相同，或shape3的size == shape2的size 且 shape3的dims == shape2的dims
            // 最可靠的方式：判断shape3指针是否与shape2相同（占位时框架复用了同一个tensor）
            bool has_bias = false;
            int  bias_rank = 0;

            if (shape3 == nullptr) {
                // OPTIONAL且bias=None时，框架传nullptr
                has_bias = false;
                std::cout << "[BIAS_DEBUG] shape3 nullptr => has_bias=false" << std::endl;
            } else if (shape3 == shape2) {
                // ★ 框架把scale的tensor复用给bias（REQUIRED占位行为）
                has_bias = false;
                std::cout << "[BIAS_DEBUG] shape3==shape2 (placeholder) => has_bias=false" << std::endl;
            } else {
                bias_rank      = shape3->GetStorageShape().GetDimNum();
                int64_t bias_size = shape3->GetStorageShape().GetShapeSize();
                has_bias = (bias_rank > 0 && bias_size > 0);
                std::cout << "[BIAS_DEBUG] real bias: bias_rank=" << bias_rank
                        << " bias_size=" << bias_size
                        << " has_bias=" << has_bias << std::endl;
            }

            // ========== BIAS DEBUG 2 ==========
            std::cout << "[BIAS_DEBUG] bias_rank=" << bias_rank
                    << " has_bias=" << has_bias << std::endl;
            // ========== BIAS DEBUG 2 END ==========

            int start_axis = (int)axis;
            int effective_axes;

            if (scaleFromBlob) {
                effective_axes = (num_axes == -1) ?
                                (int)outDimNum - start_axis :
                                (int)num_axes;
            } else {
                effective_axes = scale_rank;
            }

            if (start_axis + effective_axes > (int)outDimNum) {
                effective_axes = (int)outDimNum - start_axis;
            }

            // scale 广播shape
            std::vector<uint64_t> broadcast_scale(outDimNum, 1);
            for (int i = 0; i < effective_axes && i < scale_rank; ++i) {
                int target = start_axis + i;
                if (target < (int)outDimNum) {
                    broadcast_scale[target] = shape2->GetStorageShape().GetDim(i);
                }
            }
            inshapeVector2 = broadcast_scale;

            // ★ bias=None时 inshapeVector3 全1（与input相同，不触发广播）
            std::vector<uint64_t> broadcast_bias(outDimNum, 1);
            if (has_bias && shape3 != nullptr) {
                for (int i = 0; i < effective_axes && i < bias_rank; ++i) {
                    int target = start_axis + i;
                    if (target < (int)outDimNum) {
                        broadcast_bias[target] = shape3->GetStorageShape().GetDim(i);
                    }
                }
            }
            inshapeVector3 = broadcast_bias;

            // ★ flag判断：bias=None时inshapeVector3全1，不会错误触发广播
            bool flag = false;
            for (uint32_t i = 0; i < outDimNum; i++) {
                if (inshapeVector1[i] != inshapeVector2[i]) {
                    flag = true;
                    break;
                }
                // 只有真正有bias时才检查bias广播
                if (has_bias && inshapeVector1[i] != inshapeVector3[i]) {
                    flag = true;
                    break;
                }
            }

            if (flag) {
                GetBroadCastParams(inshapeVector1, inshapeVector2, inshapeVector3,
                                outshape,
                                local_shape, local_reduce1, local_reduce2, local_reduce3,
                                local_dim);
                tiling.set_shape(local_shape);
                tiling.set_reduce1(local_reduce1);
                tiling.set_reduce2(local_reduce2);
                tiling.set_reduce3(local_reduce3);
                tiling.set_dim(local_dim);
            }

            // ★ has_bias 先设好，再传给 GetTillingParams
            TillingLocalParams p;
            p.has_bias = has_bias ? 1 : 0;
            GetTillingParams(context, p, has_bias);  // ← 传入has_bias

            tiling.set_start_length(p.start_length);
            tiling.set_end_length(p.end_length);
            tiling.set_weight_length(p.weight_length);
            tiling.set_total_length(p.total_length);
            tiling.set_ALIGN_NUM(p.ALIGN_NUM);
            tiling.set_tiling_size(p.tiling_size);
            tiling.set_block_size(p.block_size);
            tiling.set_core_size(p.core_size);
            tiling.set_core_remain(p.core_remain);
            //tiling.set_mode(p.mode);
            tiling.set_has_bias(p.has_bias);

            context->SetBlockDim(p.aivNum);


             tiling.SaveToBuffer(
                context->GetRawTilingData()->GetData(),
                context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

            // ── 11. block dim 和 tiling key ───────────────────
            auto platform  = platform_ascendc::PlatformAscendC(
                                context->GetPlatformInfo());
            auto block_dim = platform.GetCoreNumAiv();
            context->SetBlockDim(block_dim);


            context->SetTilingKey(14);


        }

        else{

             // ── 10. 填写 tiling data ──────────────────────────
            ScaleTilingData tiling;
            tiling.set_size(tiling_size);
            tiling.set_low_size(tiling_low);
            tiling.set_high_size(tiling_high);
            tiling.set_has_bias(has_bias ? 1 : 0);
            tiling.set_input_n(input_dims.data());
            tiling.set_scale_n(scale_dims.data());
            tiling.set_bias_n(bias_dims.data());
            tiling.set_out_n(out_dims.data());
            tiling.set_broadcast_offset(broadcast_offset.data());

            tiling.SaveToBuffer(
                context->GetRawTilingData()->GetData(),
                context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

            // ── 11. block dim 和 tiling key ───────────────────
            auto platform  = platform_ascendc::PlatformAscendC(
                                context->GetPlatformInfo());
            auto block_dim = platform.GetCoreNumAiv();

            uint32_t totalLength2 = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
            if  (dt == ge::DT_FLOAT && totalLength2 <=256 ){

                block_dim =2;

            }
            context->SetBlockDim(block_dim);

            //context->SetBlockDim(1);
            context->SetTilingKey(dtype + stype * 100);

            if (stype == 1) {
                auto workspace_sizes = context->GetWorkspaceSizes(1);
                workspace_sizes[0] =
                    platform.GetLibApiWorkSpaceSize() + next_offset;
            }



        }



       

        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Scale : public OpDef {
public:
    explicit Scale(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("axis").AttrType(OPTIONAL).Int(1);
        this->Attr("num_axes").AttrType(OPTIONAL).Int(1);
        this->Attr("scale_from_blob").AttrType(OPTIONAL).Bool(true);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Scale);
} // namespace ops
