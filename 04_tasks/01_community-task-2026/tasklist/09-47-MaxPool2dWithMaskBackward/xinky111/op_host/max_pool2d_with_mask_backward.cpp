#include "geometry.h"
#include "register/op_def_registry.h"
#include <climits>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* x=context->GetInputShape(1);
    const auto* g=context->GetInputShape(0);
    const auto* m=context->GetInputShape(2);
    const auto* a=context->GetAttrs();
    if (!x || !g || !m || !a) return ge::GRAPH_FAILED;
    const auto& xs=x->GetStorageShape(); const auto& gs=g->GetStorageShape(); const auto& ms=m->GetStorageShape();
    if (xs.GetDimNum()!=4 || gs.GetDimNum()!=4 || ms.GetDimNum()!=4) return ge::GRAPH_FAILED;
    int64_t xd[4],gd[4],md[4];
    for (int i=0;i<4;++i) { xd[i]=xs.GetDim(i);gd[i]=gs.GetDim(i);md[i]=ms.GetDim(i); }
    const auto* k=a->GetListInt(0);const auto* s=a->GetListInt(1);
    const auto* p=a->GetListInt(2);const auto* d=a->GetListInt(3);
    if (!k || !s || !p || !d) return ge::GRAPH_FAILED;
    const bool* ceil=a->GetBool(4);
    PoolTiling t{};
    if (!MakePoolPlan(xd,gd,md,k->GetData(),k->GetSize(),s->GetData(),s->GetSize(),
                      p->GetData(),p->GetSize(),d->GetData(),d->GetSize(),ceil && *ceil,t)) return ge::GRAPH_FAILED;
    auto* data=context->GetTilingData<PoolTiling>();
    if (!data) return ge::GRAPH_FAILED;
    *data=t;
    const int64_t tiles=(t.planes*t.height*t.width-1)/8192+1;
    context->SetBlockDim(static_cast<uint32_t>(tiles<24?tiles:24));
    context->GetWorkspaceSizes(1)[0]=0;
    return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext* c) {
    const auto* x=c->GetInputShape(1);auto* out=c->GetOutputShape(0);
    if (!x || !out) return GRAPH_FAILED;
    *out=*x;return GRAPH_SUCCESS;
}
static graphStatus InferType(gert::InferDataTypeContext* c) {
    c->SetOutputDataType(0,c->GetInputDataType(0));return GRAPH_SUCCESS;
}
}
namespace ops {
class MaxPool2dWithMaskBackward : public OpDef {
public:
    explicit MaxPool2dWithMaskBackward(const char* name):OpDef(name) {
        this->Input("gradOutput").ParamType(REQUIRED).AutoContiguous().DataType({ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
        this->Input("self").ParamType(REQUIRED).AutoContiguous().DataType({ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
        this->Input("indices").ParamType(REQUIRED).AutoContiguous().DataType({ge::DT_INT8,ge::DT_INT8,ge::DT_INT8})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
        this->Output("gradInput").ParamType(REQUIRED).DataType({ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
        this->Attr("kernelSize").ListInt();this->Attr("stride").ListInt();this->Attr("padding").ListInt();
        this->Attr("dilation").ListInt();this->Attr("ceilMode").AttrType(OPTIONAL).Bool(false);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        OpAICoreConfig p310;
        p310.Input("gradOutput").AutoContiguous().DataType({ge::DT_FLOAT16,ge::DT_FLOAT}).Format({ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
        p310.Input("self").AutoContiguous().DataType({ge::DT_FLOAT16,ge::DT_FLOAT}).Format({ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
        p310.Input("indices").AutoContiguous().DataType({ge::DT_INT8,ge::DT_INT8}).Format({ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
        p310.Output("gradInput").DataType({ge::DT_FLOAT16,ge::DT_FLOAT}).Format({ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
        this->AICore().AddConfig("ascend310p",p310);
    }
};
OP_ADD(MaxPool2dWithMaskBackward);
}
