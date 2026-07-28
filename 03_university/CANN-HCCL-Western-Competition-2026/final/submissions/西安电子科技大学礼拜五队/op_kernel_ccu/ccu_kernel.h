#ifndef OPS_HCCL_CCU_KERNEL_H
#define OPS_HCCL_CCU_KERNEL_H

#include <ccu/ccu_types.h>

#include "custom.h"

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {

enum class PairKernelPhase : uint32_t {
    REDUCE_TO_ROOT = 0,
    BROADCAST_FROM_ROOT = 1,
};

struct AllReduceCcuKernelArg : public CcuKernelArgBase {
    uint32_t rankSize;
    uint32_t rankId;
    uint32_t peerRank;
    bool isRoot;
    PairKernelPhase phase;
    HcclDataType dataType;
    HcclReduceOp reduceType;
};

CcuResult CcuPairKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H