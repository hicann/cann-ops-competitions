/*!
 * \file chunk_scaled_dot_kkt_infershape.cpp
 * \brief ChunkScaledDotKkt operator infer shape
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeChunkScaledDotKkt(gert::InferShapeContext* context)
{
    const gert::Shape* kShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (kShape == nullptr || outputShape == nullptr || kShape->GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batch = kShape->GetDim(0);
    const int64_t seqLen = kShape->GetDim(1);
    const gert::Shape* betaShape = context->GetInputShape(1);
    if (betaShape == nullptr || betaShape->GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    const int64_t heads = betaShape->GetDim(1);
    constexpr int64_t kDefaultChunkSize = 64;

    *outputShape = gert::Shape({batch, heads, seqLen, kDefaultChunkSize});
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ChunkScaledDotKkt).InferShape(InferShapeChunkScaledDotKkt);

} // namespace ops
