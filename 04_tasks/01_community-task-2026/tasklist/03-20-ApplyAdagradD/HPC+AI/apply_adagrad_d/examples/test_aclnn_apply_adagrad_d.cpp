/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_apply_adagrad_d.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，AscendCL初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor){
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 构造输入 tensor
    aclTensor* var = nullptr;
    void* varDeviceAddr = nullptr;
    aclTensor* accum = nullptr;
    void* accumDeviceAddr = nullptr;
    aclTensor* lr = nullptr;
    void* lrDeviceAddr = nullptr;
    aclTensor* grad = nullptr;
    void* gradDeviceAddr = nullptr;
    std::vector<int64_t> varShape = {59, 61, 67, 71};
    std::vector<float> varHostData(GetShapeSize(varShape), 1);
    ret = CreateAclTensor(varHostData, varShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<int64_t> accumShape = {59, 61, 67, 71};
    std::vector<float> accumHostData(GetShapeSize(accumShape), 1);
    ret = CreateAclTensor(accumHostData, accumShape, &accumDeviceAddr, aclDataType::ACL_FLOAT, &accum);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<int64_t> lrShape = {1};
    std::vector<float> lrHostData(1, 1);
    ret = CreateAclTensor(lrHostData, lrShape, &lrDeviceAddr, aclDataType::ACL_FLOAT, &lr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<int64_t> gradShape = {59, 61, 67, 71};
    std::vector<float> gradHostData(GetShapeSize(gradShape), 1);
    ret = CreateAclTensor(gradHostData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 构造输出 tensor
    aclTensor* var_out = nullptr;
    void* var_outDeviceAddr = nullptr;
    aclTensor* accum_out = nullptr;
    void* accum_outDeviceAddr = nullptr;
    std::vector<int64_t> var_outShape = {59, 61, 67, 71};
    std::vector<float> var_outHostData(GetShapeSize(var_outShape), 0);
    ret = CreateAclTensor(var_outHostData, var_outShape, &var_outDeviceAddr, aclDataType::ACL_FLOAT, &var_out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<int64_t> accum_outShape = {59, 61, 67, 71};
    std::vector<float> accum_outHostData(GetShapeSize(accum_outShape), 0);
    ret = CreateAclTensor(accum_outHostData, accum_outShape, &accum_outDeviceAddr, aclDataType::ACL_FLOAT, &accum_out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用 aclnnApplyAdagradD 第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnApplyAdagradDGetWorkspaceSize(var, accum, lr, grad, false, var_out, accum_out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdagradDGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 申请 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用 aclnnApplyAdagradD 第二段接口
    ret = aclnnApplyAdagradD(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdagradD failed. ERROR: %d\n", ret); return ret);

    // 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 释放资源
    aclDestroyTensor(var);
    aclrtFree(varDeviceAddr);
    aclDestroyTensor(accum);
    aclrtFree(accumDeviceAddr);
    aclDestroyTensor(lr);
    aclrtFree(lrDeviceAddr);
    aclDestroyTensor(grad);
    aclrtFree(gradDeviceAddr);
    aclDestroyTensor(var_out);
    aclrtFree(var_outDeviceAddr);
    aclDestroyTensor(accum_out);
    aclrtFree(accum_outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

  return 0;
}
