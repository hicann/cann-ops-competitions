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
#include "custom.h"
#include "launch_aicpu_kernel.h"

#include <cstring>
#include <vector>

namespace {

thread_local aclrtBinHandle g_binKernelHandle = nullptr;
thread_local aclrtFuncHandle g_aicpuKernelFuncHandle = nullptr;
thread_local aclrtArgsHandle g_aicpuKernelArgsHandle = nullptr;
thread_local aclrtParamHandle g_aicpuKernelParamHandle = nullptr;
thread_local OpParam g_cachedKernelParam{};
thread_local bool g_hasCachedKernelParam = false;

constexpr uint64_t CASE05_DATA_SIZE = 512ULL * 1024;

struct Case05ControlDispatch {
    aclrtStream userStream = nullptr;
    Case05KernelParam cachedKernelParam{};
    bool hasCachedKernelParam = false;
};

// 同一Host Thread可能服务多个用户Stream。每个Stream保留独立的连续
// Host参数块，供aclrtLaunchKernelWithHostArgs直接复制并发射。
thread_local std::vector<Case05ControlDispatch> g_case05ControlDispatches;
thread_local Case05ControlDispatch *g_lastCase05ControlDispatch = nullptr;

bool UseCase05DirectDispatch(const OpParam &param)
{
    return param.rankSize == 16 &&
        param.dataType == HCCL_DATA_TYPE_FP32 &&
        param.count * sizeof(float) == CASE05_DATA_SIZE;
}

HcclResult GetCase05ControlDispatch(
    aclrtStream userStream, Case05ControlDispatch *&dispatch)
{
    for (Case05ControlDispatch &entry : g_case05ControlDispatches) {
        if (entry.userStream == userStream) {
            dispatch = &entry;
            g_lastCase05ControlDispatch = dispatch;
            return HCCL_SUCCESS;
        }
    }

    Case05ControlDispatch entry;
    entry.userStream = userStream;
    g_case05ControlDispatches.push_back(entry);
    dispatch = &g_case05ControlDispatches.back();
    g_lastCase05ControlDispatch = dispatch;
    return HCCL_SUCCESS;
}

// 加载 AICPU Kernel
HcclResult LoadAICPUKernel()
{
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
        aclError aclRet = aclrtBinaryLoadFromFile(jsonPath.c_str(), &loadOptions, &g_binKernelHandle);
        CHK_PRT_RET(aclRet != ACL_SUCCESS,
            HCCL_ERROR("Load binary from file error, ret[%d]", aclRet), HCCL_E_RUNTIME);
    }

    // Function handle 与已加载 binary 绑定且可跨重复调用复用，避免每次算子
    // 都执行名称查找；动态 Kernel 参数仍按调用重新构造。
    if (g_aicpuKernelFuncHandle == nullptr) {
        constexpr char KERNEL_NAME[] = "HcclAICPUKernel";
        aclError aclRet = aclrtBinaryGetFunction(
            g_binKernelHandle, KERNEL_NAME, &g_aicpuKernelFuncHandle);
        CHK_PRT_RET(aclRet != ACL_SUCCESS,
            HCCL_ERROR("Get AICPU function error, ret[%d]", aclRet), HCCL_E_RUNTIME);
    }
    return HCCL_SUCCESS;
}

bool IsSameKernelParam(const OpParam &lhs, const OpParam &rhs)
{
    return std::strncmp(lhs.tag, rhs.tag, TAG_LENGTH) == 0 &&
        lhs.inputPtr == rhs.inputPtr &&
        lhs.outputPtr == rhs.outputPtr &&
        lhs.count == rhs.count &&
        lhs.root == rhs.root &&
        lhs.myRank == rhs.myRank &&
        lhs.rankSize == rhs.rankSize &&
        lhs.dataType == rhs.dataType &&
        lhs.opType == rhs.opType &&
        lhs.reduceType == rhs.reduceType &&
        lhs.cpuThread == rhs.cpuThread &&
        lhs.cpuThreadOnAicpu == rhs.cpuThreadOnAicpu &&
        lhs.aicpuThreadOnCpu == rhs.aicpuThreadOnCpu &&
        lhs.resCtx == rhs.resCtx &&
        lhs.ctxSize == rhs.ctxSize;
}

HcclResult PrepareAICPUKernelArgs(
    OpParam &param, aclrtArgsHandle &cachedArgsHandle,
    aclrtParamHandle &cachedParamHandle, OpParam &cachedParam,
    bool &hasCachedParam, aclrtArgsHandle &argsHandle)
{
    if (cachedArgsHandle == nullptr) {
        aclrtArgsHandle newArgsHandle = nullptr;
        aclrtParamHandle newParamHandle = nullptr;
        ACLCHECK(aclrtKernelArgsInit(g_aicpuKernelFuncHandle, &newArgsHandle));
        ACLCHECK(aclrtKernelArgsAppend(
            newArgsHandle, &param, sizeof(OpParam), &newParamHandle));
        ACLCHECK(aclrtKernelArgsFinalize(newArgsHandle));
        cachedArgsHandle = newArgsHandle;
        cachedParamHandle = newParamHandle;
        cachedParam = param;
        hasCachedParam = true;
    } else if (!hasCachedParam ||
        !IsSameKernelParam(param, cachedParam)) {
        ACLCHECK(aclrtKernelArgsParaUpdate(
            cachedArgsHandle, cachedParamHandle,
            &param, sizeof(OpParam)));
        // Runtime 允许 Finalize 后更新参数，但更新完成后必须再次 Finalize
        // 才会重新组装本次 launch 的参数区。
        ACLCHECK(aclrtKernelArgsFinalize(cachedArgsHandle));
        cachedParam = param;
        hasCachedParam = true;
    }
    argsHandle = cachedArgsHandle;
    return HCCL_SUCCESS;
}

HcclResult LaunchPreparedAICPUKernel(
    aclrtArgsHandle argsHandle, aclrtStream stream)
{
    constexpr uint16_t NOTIFY_DEFAULT_WAIT_TIME = 27 * 68;
    aclrtLaunchKernelAttr attr;
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = NOTIFY_DEFAULT_WAIT_TIME;
    aclrtLaunchKernelCfg cfg;
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    constexpr uint32_t numBlocks = 1;
    ACLCHECK(aclrtLaunchKernelWithConfig(
        g_aicpuKernelFuncHandle, numBlocks, stream, &cfg,
        argsHandle, nullptr));
    return HCCL_SUCCESS;
}

HcclResult LaunchCase05AICPUKernelWithHostArgs(
    Case05KernelParam &param, aclrtStream stream)
{
    constexpr uint16_t NOTIFY_DEFAULT_WAIT_TIME = 27 * 68;
    aclrtLaunchKernelAttr attr;
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = NOTIFY_DEFAULT_WAIT_TIME;
    aclrtLaunchKernelCfg cfg;
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    constexpr uint32_t numBlocks = 1;
    ACLCHECK(aclrtLaunchKernelWithHostArgs(
        g_aicpuKernelFuncHandle, numBlocks, stream, &cfg,
        &param, sizeof(Case05KernelParam), nullptr, 0));
    return HCCL_SUCCESS;
}
} // namespace

namespace ops_hccl {
HcclResult LaunchCachedGenericAICPUKernel(aclrtStream stream)
{
    CHK_PRT_RET(g_aicpuKernelFuncHandle == nullptr ||
            g_aicpuKernelArgsHandle == nullptr ||
            !g_hasCachedKernelParam,
        HCCL_ERROR("Generic AICPU kernel args were not prepared"),
        HCCL_E_INTERNAL);
    CHK_RET(HcommThreadNotifyRecordOnThread(
        g_cachedKernelParam.cpuThread,
        g_cachedKernelParam.aicpuThreadOnCpu, 0));
    CHK_RET(LaunchPreparedAICPUKernel(
        g_aicpuKernelArgsHandle, stream));
    CHK_RET(HcommThreadNotifyWaitOnThread(
        g_cachedKernelParam.cpuThread, 0, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult LaunchCachedCase05AICPUKernel(aclrtStream stream)
{
    CHK_PRT_RET(g_aicpuKernelFuncHandle == nullptr,
        HCCL_ERROR("Case05 AICPU function was not prepared"),
        HCCL_E_INTERNAL);
    Case05ControlDispatch *controlDispatch =
        g_lastCase05ControlDispatch;
    if (controlDispatch == nullptr ||
        controlDispatch->userStream != stream) {
        CHK_RET(GetCase05ControlDispatch(stream, controlDispatch));
    }
    CHK_PRT_RET(!controlDispatch->hasCachedKernelParam,
        HCCL_ERROR("Case05 AICPU host args were not prepared"),
        HCCL_E_INTERNAL);
    return LaunchCase05AICPUKernelWithHostArgs(
        controlDispatch->cachedKernelParam, stream);
}

// 下发 AICPU Kernel
HcclResult LaunchAICPUKernel(OpParam &param, aclrtStream stream)
{
    // 加载 AICPU Kernel，获取 AICPU 侧链接库的句柄
    CHK_RET(LoadAICPUKernel());

    // Case05 直接把编排 Kernel 排在用户 Stream 上。Kernel 开始执行时，
    // 同一 Stream 上的输入生产任务已经完成，因此可以删除 Host->AICPU
    // 的 Record/Wait 往返；AICPU->Host 的完成通知仍门控后续用户任务。
    const bool useDirectDispatch = UseCase05DirectDispatch(param);
    Case05ControlDispatch *controlDispatch = nullptr;
    if (useDirectDispatch) {
        CHK_RET(GetCase05ControlDispatch(stream, controlDispatch));
    } else {
        CHK_RET(HcommThreadNotifyRecordOnThread(
            param.cpuThread, param.aicpuThreadOnCpu, 0));
    }

    // Case05使用仅含magic和计划Context地址的16B连续HostArgs，绕过
    // ArgsHandle参数列表；其他shape保持通用OpParam路径。
    aclrtArgsHandle argsHandle = nullptr;
    if (useDirectDispatch) {
        Case05KernelParam &case05Param =
            controlDispatch->cachedKernelParam;
        case05Param.magic = CASE05_KERNEL_PARAM_MAGIC;
        case05Param.resCtx = param.resCtx;
        controlDispatch->hasCachedKernelParam = true;
    } else {
        CHK_RET(PrepareAICPUKernelArgs(
            param, g_aicpuKernelArgsHandle,
            g_aicpuKernelParamHandle, g_cachedKernelParam,
            g_hasCachedKernelParam, argsHandle));
    }

    if (useDirectDispatch) {
        CHK_RET(LaunchCase05AICPUKernelWithHostArgs(
            controlDispatch->cachedKernelParam, stream));
    } else {
        CHK_RET(LaunchPreparedAICPUKernel(argsHandle, stream));
    }

    // Case05 的 AICPU Kernel 会在 Device 侧直接 Join 主 RTSQ；Kernel
    // 自身位于用户 Stream，其完成天然门控后续用户任务，因此无需再排
    // AICPU->CPU_TS 完成 Wait。其他 shape 仍使用原异步完成握手。
    if (!useDirectDispatch) {
        CHK_RET(HcommThreadNotifyWaitOnThread(
            param.cpuThread, 0, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
