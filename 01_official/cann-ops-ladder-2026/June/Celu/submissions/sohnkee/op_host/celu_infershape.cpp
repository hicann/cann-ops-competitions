/*!
 * \file celu_infershape.cpp
 * \brief Celu 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeCelu(gert::InferShapeContext* context)
{
    // TODO: 实现形状推导逻辑
    const gert::Shape* input_shape = context->GetInputShape(0);
    if (input_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* output_shape = context->GetOutputShape(0);
    if (output_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    *output_shape = *input_shape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Celu).InferShape(InferShapeCelu);

} // namespace ops
