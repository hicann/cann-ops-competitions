/*!
 * \file celu_def.cpp
 * \brief Celu 算子定义
 */
#include "register/op_def_registry.h"

namespace ops {
class Celu : public OpDef {
public:
    explicit Celu(const char* name) : OpDef(name)
    {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Attr("alpha")
        .AttrType(OPTIONAL)
        .Float();
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Celu);
} // namespace ops
