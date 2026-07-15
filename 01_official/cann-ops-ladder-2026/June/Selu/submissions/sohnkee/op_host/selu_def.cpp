/*!
 * \file selu_def.cpp
 * \brief Selu 算子定义
 */
#include "register/op_def_registry.h"

namespace ops {
class Selu : public OpDef {
public:
    explicit Selu(const char* name) : OpDef(name)
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

        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Selu);
} // namespace ops
