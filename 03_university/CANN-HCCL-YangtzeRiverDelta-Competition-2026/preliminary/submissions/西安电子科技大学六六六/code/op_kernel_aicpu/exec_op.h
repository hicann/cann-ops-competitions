#ifndef OPS_HCCL_EXEC_OP_H
#define OPS_HCCL_EXEC_OP_H

#include "common.h"

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx);
} // namespace ops_hccl

#endif // OPS_HCCL_EXEC_OP_H