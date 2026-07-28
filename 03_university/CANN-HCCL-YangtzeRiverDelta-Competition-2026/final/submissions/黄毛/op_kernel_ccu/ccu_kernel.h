#ifndef OPS_HCCL_CCU_KERNEL_H
#define OPS_HCCL_CCU_KERNEL_H

#include <ccu/ccu_types.h>

#include "custom.h"

namespace ops_hccl {
CcuResult CcuKernel(CcuKernelArg arg);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
