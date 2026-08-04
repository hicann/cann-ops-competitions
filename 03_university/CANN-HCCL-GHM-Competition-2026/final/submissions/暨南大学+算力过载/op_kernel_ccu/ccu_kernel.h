#ifndef OPS_HCCL_REDUCE_SCATTER_CCU_KERNEL_H
#define OPS_HCCL_REDUCE_SCATTER_CCU_KERNEL_H

#include "custom.h"

namespace ops_hccl {
struct ReduceScatterKernelArg : public CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = 0;
};

CcuResult CcuReduceScatterKernel(CcuKernelArg arg);
} // namespace ops_hccl
#endif
