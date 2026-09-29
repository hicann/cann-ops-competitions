#ifndef OPS_HCCL_CCU_KERNEL_H
#define OPS_HCCL_CCU_KERNEL_H

#include <ccu/ccu_types.h>

namespace ops_hccl {
CcuResult CcuScatterRootKernel(CcuKernelArg arg);
CcuResult CcuScatterPeerKernel(CcuKernelArg arg);
CcuResult CcuScatterPullRootKernel(CcuKernelArg arg);
CcuResult CcuScatterPullPeerKernel(CcuKernelArg arg);
CcuResult CcuScatterSmallRootKernel(CcuKernelArg arg);
CcuResult CcuScatterSmallPeerKernel(CcuKernelArg arg);
CcuResult CcuScatterRelayKernel(CcuKernelArg arg);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
