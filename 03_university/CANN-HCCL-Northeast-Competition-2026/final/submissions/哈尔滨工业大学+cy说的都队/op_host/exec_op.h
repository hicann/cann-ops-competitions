#ifndef OPS_HCCL_CCU_EXEC_OP_H
#define OPS_HCCL_CCU_EXEC_OP_H

#include <hccl/hcomm_primitives.h>
#include "common.h"


#if defined(__has_include)
#if __has_include(<ccu/ccu_launch.h>)
#include <ccu/ccu_launch.h>
#elif __has_include(<hcomm/ccu/ccu_launch.h>)
#include <hcomm/ccu/ccu_launch.h>
#elif __has_include("ccu_launch.h")
#include "ccu_launch.h"
#endif

#if __has_include(<hccl/hccl_ccu_res.h>)
#include <hccl/hccl_ccu_res.h>
#elif __has_include(<hcomm/hccl/hccl_ccu_res.h>)
#include <hcomm/hccl/hccl_ccu_res.h>
#elif __has_include(<hcomm/ccu/hccl_ccu_res.h>)
#include <hcomm/ccu/hccl_ccu_res.h>
#elif __has_include("hccl_ccu_res.h")
#include "hccl_ccu_res.h"
#endif
#else
#include <ccu/ccu_launch.h>
#include <hccl/hccl_ccu_res.h>
#endif


#ifdef __cplusplus
extern "C" {
#endif
extern HcclResult HcclCommQueryCcuIns(HcclComm comm, CcuInsHandle *insHandles, uint32_t *insNum);
extern CcuResult HcommCcuKernelRegisterStart(CcuInsHandle insHandle);
extern CcuResult HcommCcuKernelRegister(CcuInsHandle insHandle, uint32_t dieId, const char *kernelFuncName,
    const void *kernelFunc, const void **kernelArgs, uint32_t argNum, CcuKernelHandle *kernelHandle);
extern CcuResult HcommCcuKernelRegisterEnd(CcuInsHandle insHandle);
extern CcuResult HcommCcuKernelLaunch(ThreadHandle threadHandle, CcuKernelHandle kernelHandle,
    const void *taskArgs, uint32_t argSize);
#ifdef __cplusplus
}
#endif

namespace ops_hccl {
// 执行算法任务编排
HcclResult ExecOp(const OpParam &param);
} // namespace ops_hccl
#endif // OPS_HCCL_CCU_EXEC_OP_H
