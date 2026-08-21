/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"
#include "common.h"
#include "launch_aicpu_kernel.h"

namespace {
thread_local aclrtBinHandle g_binKernelHandle = nullptr;

HcclResult LoadAICPUKernel()
{
    if (g_binKernelHandle != nullptr) {
        return HCCL_SUCCESS;
    }

    char *ascendHomePath = std::getenv("ASCEND_HOME_PATH");
    CHK_PTR_NULL(ascendHomePath);
    std::string jsonPath = std::string(ascendHomePath) + "/opp/vendors/cust/aicpu/config/aicpu_kernel.json";

    aclrtBinaryLoadOption option;
    option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
    option.value.cpuKernelMode = 0;
    aclrtBinaryLoadOptions loadOptions = {0};
    loadOptions.numOpt = 1;
    loadOptions.options = &option;
    aclError aclRet = aclrtBinaryLoadFromFile(jsonPath.c_str(), &loadOptions, &g_binKernelHandle);
    CHK_PRT_RET(aclRet != ACL_SUCCESS,
        HCCL_ERROR("Load binary from file error, ret[%d]", aclRet), HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}
} // namespace

namespace ops_hccl {
HcclResult LaunchAICPUKernel(OpParam &param, aclrtStream stream)
{
    CHK_RET(LoadAICPUKernel());
    CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread, param.aicpuThreadOnCpu, 0));

    std::string kernelName = "HcclAICPUKernel";
    aclrtFuncHandle funcHandle;
    aclrtArgsHandle argsHandle;
    ACLCHECK(aclrtBinaryGetFunction(g_binKernelHandle, kernelName.c_str(), &funcHandle));
    ACLCHECK(aclrtKernelArgsInit(funcHandle, &argsHandle));
    aclrtParamHandle paraHandle;
    ACLCHECK(aclrtKernelArgsAppend(argsHandle, &param, sizeof(OpParam), &paraHandle));
    ACLCHECK(aclrtKernelArgsFinalize(argsHandle));

    constexpr uint16_t NOTIFY_DEFAULT_WAIT_TIME = 27 * 68;
    aclrtLaunchKernelAttr attr;
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = NOTIFY_DEFAULT_WAIT_TIME;
    aclrtLaunchKernelCfg cfg;
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    constexpr uint32_t numBlocks = 1;
    ACLCHECK(aclrtLaunchKernelWithConfig(funcHandle, numBlocks, stream, &cfg, argsHandle, nullptr));
    CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread, 0, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
