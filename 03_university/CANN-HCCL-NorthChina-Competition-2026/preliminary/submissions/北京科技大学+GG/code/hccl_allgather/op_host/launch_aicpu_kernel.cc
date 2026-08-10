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
thread_local aclrtFuncHandle g_funcHandle = nullptr;

// 加载 AICPU Kernel
HcclResult LoadAICPUKernel()
{
    // Avoid both binary reload and repeated name lookup on the hot path.
    if (g_funcHandle != nullptr) {
        return HCCL_SUCCESS;
    }

    if (g_binKernelHandle == nullptr) {
        // AICPU 算子信息库 json 文件路径
        char *ascendHomePath = std::getenv("ASCEND_HOME_PATH");
        CHK_PTR_NULL(ascendHomePath);
        std::string jsonPath =
            std::string(ascendHomePath) + "/opp/vendors/cust/aicpu/config/aicpu_kernel.json";

        // 加载算子二进制
        aclrtBinaryLoadOption option;
        option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
        option.value.cpuKernelMode = 0;
        aclrtBinaryLoadOptions loadOptions = {0};
        loadOptions.numOpt = 1;
        loadOptions.options = &option;
        aclError aclRet =
            aclrtBinaryLoadFromFile(jsonPath.c_str(), &loadOptions, &g_binKernelHandle);
        CHK_PRT_RET(aclRet != ACL_SUCCESS,
            HCCL_ERROR("Load binary from file error, ret[%d]", aclRet), HCCL_E_RUNTIME);
    }

    constexpr char KERNEL_NAME[] = "HcclAICPUKernel";
    ACLCHECK(aclrtBinaryGetFunction(g_binKernelHandle, KERNEL_NAME, &g_funcHandle));
    return HCCL_SUCCESS;
}
} // namespace

namespace ops_hccl {
// 下发 AICPU Kernel
HcclResult LaunchAICPUKernel(
    OpParam &param, aclrtStream stream, aclrtArgsHandle &argsHandle)
{
    // 加载 AICPU Kernel，获取 AICPU 侧链接库的句柄
    CHK_RET(LoadAICPUKernel());

    // Host 通知 Device 当前线程即将下发 AICPU Kernel
    CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread, param.aicpuThreadOnCpu, 0));

    if (argsHandle == nullptr) {
        // aclrtKernelArgsAppend copies the complete OpParam into host-side argument storage.
        // The owning FastPathEntry keeps this finalized handle for all matching hot calls.
        ACLCHECK(aclrtKernelArgsInit(g_funcHandle, &argsHandle));
        aclrtParamHandle paraHandle;
        ACLCHECK(aclrtKernelArgsAppend(argsHandle, &param, sizeof(OpParam), &paraHandle));
        ACLCHECK(aclrtKernelArgsFinalize(argsHandle));
    }

    // 下发 AICPU Kernel
    constexpr uint16_t NOTIFY_DEFAULT_WAIT_TIME = 27 * 68; // NotifyWait 超时时间
    aclrtLaunchKernelAttr attr;
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = NOTIFY_DEFAULT_WAIT_TIME;
    aclrtLaunchKernelCfg cfg;
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    constexpr uint32_t numBlocks = 1;
    ACLCHECK(aclrtLaunchKernelWithConfig(
        g_funcHandle, numBlocks, stream, &cfg, argsHandle, nullptr));

    // Host 等待 Device 通知，确认 AICPU Kernel 任务下发完成
    CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread, 0, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
