/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <map>
#include <string>
#include "log.h"
#include "common.h"
#include "launch_aicpu_kernel.h"

namespace {

thread_local std::map<int32_t, aclrtBinHandle> g_binKernelHandles;

HcclResult LoadAICPUKernel()
{
    int32_t device = 0;
    ACLCHECK(aclrtGetDevice(&device));
    if (g_binKernelHandles.count(device) != 0) {
        return HCCL_SUCCESS;
    }

    char *ascendHomePath = std::getenv("ASCEND_HOME_PATH");
    CHK_PTR_NULL(ascendHomePath);
    std::string jsonPath = std::string(ascendHomePath) + "/opp/vendors/cust/aicpu/config/aicpu_kernel.json";

    aclrtBinaryLoadOption option{};
    option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
    option.value.cpuKernelMode = 0;
    aclrtBinaryLoadOptions loadOptions = {0};
    loadOptions.numOpt = 1;
    loadOptions.options = &option;
    aclrtBinHandle handle = nullptr;
    aclError aclRet = aclrtBinaryLoadFromFile(jsonPath.c_str(), &loadOptions, &handle);
    CHK_PRT_RET(aclRet != ACL_SUCCESS, HCCL_ERROR("Load binary from file error, ret[%d]", aclRet), HCCL_E_RUNTIME);
    g_binKernelHandles.emplace(device, handle);
    return HCCL_SUCCESS;
}
} // namespace

namespace ops_hccl {
HcclResult PrepareAICPUKernel()
{
    return LoadAICPUKernel();
}

HcclResult LaunchAICPUKernel(OpParam &param, aclrtStream stream)
{
    CHK_RET(LoadAICPUKernel());

    int32_t device = 0;
    ACLCHECK(aclrtGetDevice(&device));

    std::string kernelName = "HcclAICPUKernel";
    aclrtFuncHandle funcHandle = nullptr;
    aclrtArgsHandle argsHandle = nullptr;
    ACLCHECK(aclrtBinaryGetFunction(g_binKernelHandles.at(device), kernelName.c_str(), &funcHandle));

    ACLCHECK(aclrtKernelArgsInit(funcHandle, &argsHandle));
    aclrtParamHandle paraHandle = nullptr;
    ACLCHECK(aclrtKernelArgsAppend(argsHandle, &param, sizeof(OpParam), &paraHandle));
    ACLCHECK(aclrtKernelArgsFinalize(argsHandle));

    constexpr uint16_t NOTIFY_DEFAULT_WAIT_TIME = 27 * 68; // NotifyWait 超时时间
    aclrtLaunchKernelAttr attr{};
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = NOTIFY_DEFAULT_WAIT_TIME;
    aclrtLaunchKernelCfg cfg{};
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    constexpr uint32_t numBlocks = 1;
    CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread, param.aicpuThreadOnCpu, 0));
    ACLCHECK(aclrtLaunchKernelWithConfig(funcHandle, numBlocks, stream, &cfg, argsHandle, nullptr));

    CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread, 0, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
