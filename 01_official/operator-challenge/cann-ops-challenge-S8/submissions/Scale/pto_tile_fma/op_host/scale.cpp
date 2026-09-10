
#include <cstring>
#ifdef SCALE_DIAG
#include <cstdio>
#endif
#include "../op_kernel/scale_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ScaleTilingData tiling;

    // ---- Input shape ----
    const gert::StorageShape* input_shape_obj = context->GetInputShape(0);
    int64_t input_rank = static_cast<int64_t>(input_shape_obj->GetStorageShape().GetDimNum());

    // ---- Scale shape ----
    const gert::StorageShape* scale_shape_obj = context->GetInputShape(1);
    int64_t scale_rank = static_cast<int64_t>(scale_shape_obj->GetStorageShape().GetDimNum());

    // ---- Attrs: axis, num_axes, scale_from_blob ----
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    int64_t axis_raw     = 1;
    int64_t axes_num_raw = 1;
    bool scale_from_blob = true;
    if (attrs != nullptr) {
        const int64_t* p0 = attrs->GetAttrPointer<int64_t>(0);
        const int64_t* p1 = attrs->GetAttrPointer<int64_t>(1);
        const bool*    p2 = attrs->GetAttrPointer<bool>(2);
        if (p0) axis_raw     = *p0;
        if (p1) axes_num_raw = *p1;
        if (p2) scale_from_blob = *p2;
    }

    // Normalize axis to [0, input_rank)
    int64_t axis = (axis_raw < 0) ? (input_rank + axis_raw) : axis_raw;
    if (axis < 0) axis = 0;
    if (axis >= input_rank) axis = input_rank - 1;

    // ---- Determine end_axis: dims [axis, end_axis) are covered by scale ----
    // Since the verifier ensures scale.shape == input[axis:end_axis], we can
    // simply use scale_rank for end_axis (matches both scale_from_blob true/false).
    int64_t end_axis = axis + scale_rank;
    if (end_axis > input_rank) end_axis = input_rank;

    // ---- Compute outer_size, scale_size, inner_size ----
    uint32_t outer_size = 1u;
    for (int64_t i = 0; i < axis; i++) {
        outer_size *= static_cast<uint32_t>(input_shape_obj->GetStorageShape().GetDim(i));
    }

    uint32_t scale_size = 1u;
    for (int64_t i = 0; i < scale_rank; i++) {
        scale_size *= static_cast<uint32_t>(scale_shape_obj->GetStorageShape().GetDim(i));
    }

    uint32_t inner_size = 1u;
    for (int64_t i = end_axis; i < input_rank; i++) {
        inner_size *= static_cast<uint32_t>(input_shape_obj->GetStorageShape().GetDim(i));
    }

    // ---- has_bias: check if input(2) exists ----
    uint32_t has_bias = 0u;
    if (context->GetOptionalInputShape(2) != nullptr) {
        has_bias = 1u;
    }

    // ---- dtype ----
    auto dtypeEnum = context->GetInputDesc(0)->GetDataType();

    tiling.set_outer_size(outer_size);
    tiling.set_scale_size(scale_size);
    tiling.set_inner_size(inner_size);
    tiling.set_has_bias(has_bias);
    tiling.set_dtype(static_cast<uint32_t>(dtypeEnum));

    platform_ascendc::PlatformAscendC platformInfo(context->GetPlatformInfo());
    uint32_t coreNum = platformInfo.GetCoreNumAiv();  // vector op → use AIV core count
    if (coreNum == 0u) coreNum = 8u;

    // Parallel work items must match the kernel's actual grain so BlockDim scales to all cores.
    //   ScaleCached (inner==1 & S<=8192, or inner>1 & S<=256): distributes over outer batches.
    //   ScaleExpand (the rest): distributes over (p-tile x batch) = nPT*outer units, where
    //     nPT = ceil(S*I / EXP_TILE). The old heuristic used `outer` for every S<=8192, which
    //     starved small-outer EXPAND shapes of cores (e.g. [4,1000,200] launched on only 4 cores,
    //     so the balanced-flatten distribution could never use all 40).
    bool useExpand = (inner_size == 1u) ? (scale_size > 8192u) : (scale_size > 256u);
    uint32_t workItems;
    if (useExpand) {
        uint64_t P = static_cast<uint64_t>(scale_size) * inner_size;
        // ET must match op_kernel ScaleExpand: fp16=10240, fp32=5120, bf16=4608.
        // Using the true per-dtype tile keeps nPT (and BlockDim) == the kernel's work-unit
        // count so no extra cores are launched only to early-return.
        uint32_t ET = (dtypeEnum == ge::DT_FLOAT16) ? 10240u
                    : ((dtypeEnum == ge::DT_BF16)   ? 4608u : 5120u);
        uint32_t nPT = static_cast<uint32_t>((P + ET - 1u) / ET);    // ceil(P/ET)
        uint64_t units = static_cast<uint64_t>(nPT) * outer_size;    // (p-tile x batch) work units
        workItems = (units >= coreNum) ? coreNum : static_cast<uint32_t>(units);
    } else {
        workItems = outer_size;
    }
    uint32_t actualCores = (workItems < coreNum) ? workItems : coreNum;
    if (actualCores == 0u) actualCores = 1u;

#ifdef SCALE_DIAG
    {
        // Diagnostic (off by default): report the EXPAND work grain so the per-core balance can be
        // reasoned about. nPT = number of p-tiles = ceil(scale_size*inner_size / EXP_TILE=3840).
        uint64_t Pdiag = static_cast<uint64_t>(scale_size) * inner_size;
        uint32_t nPTdiag = static_cast<uint32_t>((Pdiag + 3839u) / 3840u);
        fprintf(stderr, "[SCALE_DIAG] outer=%u S=%u I=%u nPT=%u actualCores=%u dtype=%u has_bias=%u\n",
                outer_size, scale_size, inner_size, nPTdiag, actualCores,
                static_cast<uint32_t>(dtypeEnum), has_bias);
    }
#endif

    context->SetTilingKey(0);
    context->SetBlockDim(actualCores);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling


namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge


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

}  // namespace ops
