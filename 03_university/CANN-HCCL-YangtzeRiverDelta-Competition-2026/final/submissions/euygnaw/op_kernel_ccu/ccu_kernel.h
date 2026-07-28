#ifndef OPS_HCCL_CCU_KERNEL_H
#define OPS_HCCL_CCU_KERNEL_H

#include <ccu/ccu_types.h>

namespace ops_hccl {

CcuResult CcuBroadcastDirectRootKernel(CcuKernelArg arg);
CcuResult CcuBroadcastDirectPeerKernel(CcuKernelArg arg);
CcuResult CcuBroadcastScatterKernel(CcuKernelArg arg);
CcuResult CcuBroadcastAllGatherKernel(CcuKernelArg arg);
CcuResult CcuBroadcastScatterAllGatherKernel(CcuKernelArg arg);
CcuResult CcuBroadcastChainNoAckKernel(CcuKernelArg arg);
CcuResult CcuBroadcastChainAckKernel(CcuKernelArg arg);
CcuResult CcuBroadcastDirectRoot4Kernel(CcuKernelArg arg);
CcuResult CcuBroadcast8p4ScatterPrefetchKernel(CcuKernelArg arg);
CcuResult CcuBroadcast8p4AllGatherSkipRootKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H