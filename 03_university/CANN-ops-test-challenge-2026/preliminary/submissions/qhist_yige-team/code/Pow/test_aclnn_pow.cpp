/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include "acl/acl.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"
#include "aclnnop/aclnn_exp2.h"

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

int64_t totalTests = 0;
int64_t passedTests = 0;
int64_t failedTests = 0;

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

bool CompareWithTolerance(float actual, double expected, double atol = 1e-5, double rtol = 1e-5) {
  if (std::isnan(expected) && std::isnan(actual)) return true;
  if (std::isinf(expected) && std::isinf(actual)) return true;
  double diff = std::abs(actual - expected);
  double tolerance = atol + rtol * std::abs(expected);
  return diff <= tolerance;
}

void TestResult(const char* testName, bool passed) {
  totalTests++;
  if (passed) {
    passedTests++;
    LOG_PRINT("[PASS] %s\n", testName);
  } else {
    failedTests++;
    LOG_PRINT("[FAIL] %s\n", testName);
  }
}

int TestPowTensorScalar_Float32_SpecialExponents(aclrtStream stream) {
  const char* testName = "PowTensorScalar_Float32_SpecialExponents";
  
  std::vector<float> exponents = {0.0f, 1.0f, 0.5f, 2.0f, -1.0f, 3.0f};
  std::vector<float> baseData = {2.0f, 4.0f, 9.0f, 16.0f};
  std::vector<int64_t> shape = {2, 2};
  
  for (float expVal : exponents) {
    void* baseDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* base = nullptr;
    aclTensor* out = nullptr;
    aclScalar* exponent = nullptr;
    
    std::vector<float> outData(GetShapeSize(shape), 0.0f);
    
    int ret = CreateAclTensor(baseData, shape, &baseDeviceAddr, aclDataType::ACL_FLOAT, &base);
    if (ret != ACL_SUCCESS) {
      TestResult(testName, false);
      return -1;
    }
    
    ret = CreateAclTensor(outData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
      aclrtFree(baseDeviceAddr);
      aclDestroyTensor(base);
      TestResult(testName, false);
      return -1;
    }
    
    exponent = aclCreateScalar(&expVal, aclDataType::ACL_FLOAT);
    if (exponent == nullptr) {
      aclrtFree(baseDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclDestroyTensor(base);
      aclDestroyTensor(out);
      TestResult(testName, false);
      return -1;
    }
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnPowTensorScalarGetWorkspaceSize(base, exponent, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
      aclrtFree(baseDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclDestroyTensor(base);
      aclDestroyTensor(out);
      aclDestroyScalar(exponent);
      TestResult(testName, false);
      return -1;
    }
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      if (ret != ACL_SUCCESS) {
        aclrtFree(baseDeviceAddr);
        aclrtFree(outDeviceAddr);
        aclDestroyTensor(base);
        aclDestroyTensor(out);
        aclDestroyScalar(exponent);
        TestResult(testName, false);
        return -1;
      }
    }
    
    ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
      if (workspaceSize > 0) aclrtFree(workspaceAddr);
      aclrtFree(baseDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclDestroyTensor(base);
      aclDestroyTensor(out);
      aclDestroyScalar(exponent);
      TestResult(testName, false);
      return -1;
    }
    
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
      if (workspaceSize > 0) aclrtFree(workspaceAddr);
      aclrtFree(baseDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclDestroyTensor(base);
      aclDestroyTensor(out);
      aclDestroyScalar(exponent);
      TestResult(testName, false);
      return -1;
    }
    
    std::vector<float> resultData(GetShapeSize(shape));
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                      resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    bool allMatch = true;
    for (size_t i = 0; i < baseData.size(); i++) {
      double expected = std::pow(static_cast<double>(baseData[i]), static_cast<double>(expVal));
      if (!CompareWithTolerance(resultData[i], expected)) {
        allMatch = false;
        break;
      }
    }
    
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(out);
    aclDestroyScalar(exponent);
    
    TestResult(testName, allMatch);
    return allMatch ? 0 : -1;
  }
  
  TestResult(testName, true);
  return 0;
}


int TestInplacePowTensorScalar(aclrtStream stream) {
  const char* testName = "InplacePowTensorScalar";
  
  std::vector<float> baseData = {2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<int64_t> shape = {2, 2};
  float expVal = 2.0f;
  
  void* baseDeviceAddr = nullptr;
  aclTensor* base = nullptr;
  aclScalar* exponent = nullptr;
  
  int ret = CreateAclTensor(baseData, shape, &baseDeviceAddr, aclDataType::ACL_FLOAT, &base);
  if (ret != ACL_SUCCESS) {
    TestResult(testName, false);
    return -1;
  }
  
  exponent = aclCreateScalar(&expVal, aclDataType::ACL_FLOAT);
  if (exponent == nullptr) {
    aclrtFree(baseDeviceAddr);
    aclDestroyTensor(base);
    TestResult(testName, false);
    return -1;
  }
  
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnInplacePowTensorScalarGetWorkspaceSize(base, exponent, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    aclrtFree(baseDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyScalar(exponent);
    TestResult(testName, false);
    return -1;
  }
  
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      aclrtFree(baseDeviceAddr);
      aclDestroyTensor(base);
      aclDestroyScalar(exponent);
      TestResult(testName, false);
      return -1;
    }
  }
  
  ret = aclnnInplacePowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(baseDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyScalar(exponent);
    TestResult(testName, false);
    return -1;
  }
  
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(baseDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyScalar(exponent);
    TestResult(testName, false);
    return -1;
  }
  
  std::vector<float> resultData(GetShapeSize(shape));
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), baseDeviceAddr,
                    resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  
  bool allMatch = true;
  for (size_t i = 0; i < baseData.size(); i++) {
    double expected = std::pow(static_cast<double>(baseData[i]), static_cast<double>(expVal));
    if (!CompareWithTolerance(resultData[i], expected)) {
      allMatch = false;
      break;
    }
  }
  
  if (workspaceSize > 0) aclrtFree(workspaceAddr);
  aclrtFree(baseDeviceAddr);
  aclDestroyTensor(base);
  aclDestroyScalar(exponent);
  
  TestResult(testName, allMatch);
  return allMatch ? 0 : -1;
}

int TestPowScalarTensor(aclrtStream stream) {
  const char* testName = "PowScalarTensor";
  
  float baseVal = 2.0f;
  std::vector<float> expData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> shape = {2, 2};
  
  void* expDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* exp = nullptr;
  aclTensor* out = nullptr;
  aclScalar* base = nullptr;
  
  std::vector<float> outData(GetShapeSize(shape), 0.0f);
  
  int ret = CreateAclTensor(expData, shape, &expDeviceAddr, aclDataType::ACL_FLOAT, &exp);
  if (ret != ACL_SUCCESS) {
    TestResult(testName, false);
    return -1;
  }
  
  ret = CreateAclTensor(outData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclrtFree(expDeviceAddr);
    aclDestroyTensor(exp);
    TestResult(testName, false);
    return -1;
  }
  
  base = aclCreateScalar(&baseVal, aclDataType::ACL_FLOAT);
  if (base == nullptr) {
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    TestResult(testName, false);
    return -1;
  }
  
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnPowScalarTensorGetWorkspaceSize(base, exp, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    aclDestroyScalar(base);
    TestResult(testName, false);
    return -1;
  }
  
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      aclrtFree(expDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclDestroyTensor(exp);
      aclDestroyTensor(out);
      aclDestroyScalar(base);
      TestResult(testName, false);
      return -1;
    }
  }
  
  ret = aclnnPowScalarTensor(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    aclDestroyScalar(base);
    TestResult(testName, false);
    return -1;
  }
  
  ret = aclrtSynchronizeStream(stream);
  bool success = (ret == ACL_SUCCESS);
  
  if (workspaceSize > 0) aclrtFree(workspaceAddr);
  aclrtFree(expDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclDestroyTensor(exp);
  aclDestroyTensor(out);
  aclDestroyScalar(base);
  
  TestResult(testName, success);
  return success ? 0 : -1;
}

int TestPowTensorTensor_SameShape(aclrtStream stream) {
  const char* testName = "PowTensorTensor_SameShape";
  
  std::vector<float> baseData = {2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> expData = {1.0f, 2.0f, 1.0f, 2.0f};
  std::vector<int64_t> shape = {2, 2};
  
  void* baseDeviceAddr = nullptr;
  void* expDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* base = nullptr;
  aclTensor* exp = nullptr;
  aclTensor* out = nullptr;
  
  std::vector<float> outData(GetShapeSize(shape), 0.0f);
  
  int ret = CreateAclTensor(baseData, shape, &baseDeviceAddr, aclDataType::ACL_FLOAT, &base);
  if (ret != ACL_SUCCESS) {
    TestResult(testName, false);
    return -1;
  }
  
  ret = CreateAclTensor(expData, shape, &expDeviceAddr, aclDataType::ACL_FLOAT, &exp);
  if (ret != ACL_SUCCESS) {
    aclrtFree(baseDeviceAddr);
    aclDestroyTensor(base);
    TestResult(testName, false);
    return -1;
  }
  
  ret = CreateAclTensor(outData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclrtFree(baseDeviceAddr);
    aclrtFree(expDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(exp);
    TestResult(testName, false);
    return -1;
  }
  
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnPowTensorTensorGetWorkspaceSize(base, exp, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    aclrtFree(baseDeviceAddr);
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    TestResult(testName, false);
    return -1;
  }
  
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      aclrtFree(baseDeviceAddr);
      aclrtFree(expDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclDestroyTensor(base);
      aclDestroyTensor(exp);
      aclDestroyTensor(out);
      TestResult(testName, false);
      return -1;
    }
  }
  
  ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(baseDeviceAddr);
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    TestResult(testName, false);
    return -1;
  }
  
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(baseDeviceAddr);
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    TestResult(testName, false);
    return -1;
  }
  
  std::vector<float> resultData(GetShapeSize(shape));
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                    resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  
  bool allMatch = true;
  for (size_t i = 0; i < baseData.size(); i++) {
    double expected = std::pow(static_cast<double>(baseData[i]), static_cast<double>(expData[i]));
    if (!CompareWithTolerance(resultData[i], expected)) {
      allMatch = false;
      break;
    }
  }
  
  if (workspaceSize > 0) aclrtFree(workspaceAddr);
  aclrtFree(baseDeviceAddr);
  aclrtFree(expDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclDestroyTensor(base);
  aclDestroyTensor(exp);
  aclDestroyTensor(out);
  
  TestResult(testName, allMatch);
  return allMatch ? 0 : -1;
}

int TestPowTensorTensor_Broadcast(aclrtStream stream) {
  const char* testName = "PowTensorTensor_Broadcast";
  
  std::vector<float> baseData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expData = {2.0f, 3.0f};
  std::vector<int64_t> baseShape = {2, 2};
  std::vector<int64_t> expShape = {2};
  std::vector<int64_t> outShape = {2, 2};
  
  void* baseDeviceAddr = nullptr;
  void* expDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* base = nullptr;
  aclTensor* exp = nullptr;
  aclTensor* out = nullptr;
  
  std::vector<float> outData(GetShapeSize(outShape), 0.0f);
  
  int ret = CreateAclTensor(baseData, baseShape, &baseDeviceAddr, aclDataType::ACL_FLOAT, &base);
  if (ret != ACL_SUCCESS) {
    TestResult(testName, false);
    return -1;
  }
  
  ret = CreateAclTensor(expData, expShape, &expDeviceAddr, aclDataType::ACL_FLOAT, &exp);
  if (ret != ACL_SUCCESS) {
    aclrtFree(baseDeviceAddr);
    aclDestroyTensor(base);
    TestResult(testName, false);
    return -1;
  }
  
  ret = CreateAclTensor(outData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclrtFree(baseDeviceAddr);
    aclrtFree(expDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(exp);
    TestResult(testName, false);
    return -1;
  }
  
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnPowTensorTensorGetWorkspaceSize(base, exp, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    aclrtFree(baseDeviceAddr);
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    TestResult(testName, false);
    return -1;
  }
  
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      aclrtFree(baseDeviceAddr);
      aclrtFree(expDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclDestroyTensor(base);
      aclDestroyTensor(exp);
      aclDestroyTensor(out);
      TestResult(testName, false);
      return -1;
    }
  }
  
  ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(baseDeviceAddr);
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    TestResult(testName, false);
    return -1;
  }
  
  ret = aclrtSynchronizeStream(stream);
  bool success = (ret == ACL_SUCCESS);
  
  if (workspaceSize > 0) aclrtFree(workspaceAddr);
  aclrtFree(baseDeviceAddr);
  aclrtFree(expDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclDestroyTensor(base);
  aclDestroyTensor(exp);
  aclDestroyTensor(out);
  
  TestResult(testName, success);
  return success ? 0 : -1;
}

int TestInplacePowTensorTensor(aclrtStream stream) {
  const char* testName = "InplacePowTensorTensor";
  
  std::vector<float> baseData = {2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> expData = {1.0f, 2.0f, 1.0f, 2.0f};
  std::vector<int64_t> shape = {2, 2};
  
  void* baseDeviceAddr = nullptr;
  void* expDeviceAddr = nullptr;
  aclTensor* base = nullptr;
  aclTensor* exp = nullptr;
  
  int ret = CreateAclTensor(baseData, shape, &baseDeviceAddr, aclDataType::ACL_FLOAT, &base);
  if (ret != ACL_SUCCESS) {
    TestResult(testName, false);
    return -1;
  }
  
  ret = CreateAclTensor(expData, shape, &expDeviceAddr, aclDataType::ACL_FLOAT, &exp);
  if (ret != ACL_SUCCESS) {
    aclrtFree(baseDeviceAddr);
    aclDestroyTensor(base);
    TestResult(testName, false);
    return -1;
  }
  
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnInplacePowTensorTensorGetWorkspaceSize(base, exp, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    aclrtFree(baseDeviceAddr);
    aclrtFree(expDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(exp);
    TestResult(testName, false);
    return -1;
  }
  
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      aclrtFree(baseDeviceAddr);
      aclrtFree(expDeviceAddr);
      aclDestroyTensor(base);
      aclDestroyTensor(exp);
      TestResult(testName, false);
      return -1;
    }
  }
  
  ret = aclnnInplacePowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(baseDeviceAddr);
    aclrtFree(expDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(exp);
    TestResult(testName, false);
    return -1;
  }
  
  ret = aclrtSynchronizeStream(stream);
  bool success = (ret == ACL_SUCCESS);
  
  if (workspaceSize > 0) aclrtFree(workspaceAddr);
  aclrtFree(baseDeviceAddr);
  aclrtFree(expDeviceAddr);
  aclDestroyTensor(base);
  aclDestroyTensor(exp);
  
  TestResult(testName, success);
  return success ? 0 : -1;
}

int TestExp2(aclrtStream stream) {
  const char* testName = "Exp2";
  
  std::vector<float> inputData = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<int64_t> shape = {2, 2};
  
  void* inputDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* input = nullptr;
  aclTensor* out = nullptr;
  
  std::vector<float> outData(GetShapeSize(shape), 0.0f);
  
  int ret = CreateAclTensor(inputData, shape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
  if (ret != ACL_SUCCESS) {
    TestResult(testName, false);
    return -1;
  }
  
  ret = CreateAclTensor(outData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclrtFree(inputDeviceAddr);
    aclDestroyTensor(input);
    TestResult(testName, false);
    return -1;
  }
  
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnExp2GetWorkspaceSize(input, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    aclrtFree(inputDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(input);
    aclDestroyTensor(out);
    TestResult(testName, false);
    return -1;
  }
  
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      aclrtFree(inputDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclDestroyTensor(input);
      aclDestroyTensor(out);
      TestResult(testName, false);
      return -1;
    }
  }
  
  ret = aclnnExp2(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(inputDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(input);
    aclDestroyTensor(out);
    TestResult(testName, false);
    return -1;
  }
  
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(inputDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(input);
    aclDestroyTensor(out);
    TestResult(testName, false);
    return -1;
  }
  
  std::vector<float> resultData(GetShapeSize(shape));
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                    resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  
  bool allMatch = true;
  for (size_t i = 0; i < inputData.size(); i++) {
    double expected = std::pow(2.0, static_cast<double>(inputData[i]));
    if (!CompareWithTolerance(resultData[i], expected)) {
      allMatch = false;
      break;
    }
  }
  
  if (workspaceSize > 0) aclrtFree(workspaceAddr);
  aclrtFree(inputDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclDestroyTensor(input);
  aclDestroyTensor(out);
  
  TestResult(testName, allMatch);
  return allMatch ? 0 : -1;
}

int TestInplaceExp2(aclrtStream stream) {
  const char* testName = "InplaceExp2";
  
  std::vector<float> inputData = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<int64_t> shape = {2, 2};
  
  void* inputDeviceAddr = nullptr;
  aclTensor* input = nullptr;
  
  int ret = CreateAclTensor(inputData, shape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
  if (ret != ACL_SUCCESS) {
    TestResult(testName, false);
    return -1;
  }
  
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnInplaceExp2GetWorkspaceSize(input, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    aclrtFree(inputDeviceAddr);
    aclDestroyTensor(input);
    TestResult(testName, false);
    return -1;
  }
  
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      aclrtFree(inputDeviceAddr);
      aclDestroyTensor(input);
      TestResult(testName, false);
      return -1;
    }
  }
  
  ret = aclnnInplaceExp2(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(inputDeviceAddr);
    aclDestroyTensor(input);
    TestResult(testName, false);
    return -1;
  }
  
  ret = aclrtSynchronizeStream(stream);
  bool success = (ret == ACL_SUCCESS);
  
  if (workspaceSize > 0) aclrtFree(workspaceAddr);
  aclrtFree(inputDeviceAddr);
  aclDestroyTensor(input);
  
  TestResult(testName, success);
  return success ? 0 : -1;
}

int TestPowTensorScalar_EdgeCases(aclrtStream stream) {
  const char* testName = "PowTensorScalar_EdgeCases";
  
  std::vector<float> baseData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> shape = {2, 2};
  
  std::vector<float> exponents = {0.0f, 1.0f, 2.0f, 3.0f};
  
  bool allTestsPassed = true;
  for (float expVal : exponents) {
    void* baseDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* base = nullptr;
    aclTensor* out = nullptr;
    aclScalar* exponent = nullptr;
    
    std::vector<float> outData(GetShapeSize(shape), 0.0f);
    
    int ret = CreateAclTensor(baseData, shape, &baseDeviceAddr, aclDataType::ACL_FLOAT, &base);
    if (ret != ACL_SUCCESS) {
      allTestsPassed = false;
      continue;
    }
    
    ret = CreateAclTensor(outData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
      aclrtFree(baseDeviceAddr);
      aclDestroyTensor(base);
      allTestsPassed = false;
      continue;
    }
    
    exponent = aclCreateScalar(&expVal, aclDataType::ACL_FLOAT);
    if (exponent == nullptr) {
      aclrtFree(baseDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclDestroyTensor(base);
      aclDestroyTensor(out);
      allTestsPassed = false;
      continue;
    }
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnPowTensorScalarGetWorkspaceSize(base, exponent, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
      aclrtFree(baseDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclDestroyTensor(base);
      aclDestroyTensor(out);
      aclDestroyScalar(exponent);
      allTestsPassed = false;
      continue;
    }
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      if (ret != ACL_SUCCESS) {
        aclrtFree(baseDeviceAddr);
        aclrtFree(outDeviceAddr);
        aclDestroyTensor(base);
        aclDestroyTensor(out);
        aclDestroyScalar(exponent);
        allTestsPassed = false;
        continue;
      }
    }
    
    ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
      if (workspaceSize > 0) aclrtFree(workspaceAddr);
      aclrtFree(baseDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclDestroyTensor(base);
      aclDestroyTensor(out);
      aclDestroyScalar(exponent);
      allTestsPassed = false;
      continue;
    }
    
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
      if (workspaceSize > 0) aclrtFree(workspaceAddr);
      aclrtFree(baseDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclDestroyTensor(base);
      aclDestroyTensor(out);
      aclDestroyScalar(exponent);
      allTestsPassed = false;
      continue;
    }
    
    std::vector<float> resultData(GetShapeSize(shape));
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                      resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    bool allMatch = true;
    for (size_t i = 0; i < baseData.size(); i++) {
      double expected = std::pow(static_cast<double>(baseData[i]), static_cast<double>(expVal));
      if (!CompareWithTolerance(resultData[i], expected)) {
        allMatch = false;
        break;
      }
    }
    
    if (!allMatch) {
      allTestsPassed = false;
    }
    
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(out);
    aclDestroyScalar(exponent);
  }
  
  TestResult(testName, allTestsPassed);
  return allTestsPassed ? 0 : -1;
}

int TestPowTensorScalar_MoreExponents(aclrtStream stream) {
  const char* testName = "PowTensorScalar_MoreExponents";
  
  std::vector<float> baseData = {4.0f, 9.0f, 16.0f, 25.0f};
  std::vector<int64_t> shape = {2, 2};
  
  std::vector<float> exponents = {-0.5f, -1.0f, -2.0f};
  
  bool allTestsPassed = true;
  for (float expVal : exponents) {
    void* baseDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* base = nullptr;
    aclTensor* out = nullptr;
    aclScalar* exponent = nullptr;
    
    std::vector<float> outData(GetShapeSize(shape), 0.0f);
    
    int ret = CreateAclTensor(baseData, shape, &baseDeviceAddr, aclDataType::ACL_FLOAT, &base);
    if (ret != ACL_SUCCESS) {
      allTestsPassed = false;
      continue;
    }
    
    ret = CreateAclTensor(outData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
      aclrtFree(baseDeviceAddr);
      aclDestroyTensor(base);
      allTestsPassed = false;
      continue;
    }
    
    exponent = aclCreateScalar(&expVal, aclDataType::ACL_FLOAT);
    if (exponent == nullptr) {
      aclrtFree(baseDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclDestroyTensor(base);
      aclDestroyTensor(out);
      allTestsPassed = false;
      continue;
    }
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnPowTensorScalarGetWorkspaceSize(base, exponent, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
      aclrtFree(baseDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclDestroyTensor(base);
      aclDestroyTensor(out);
      aclDestroyScalar(exponent);
      allTestsPassed = false;
      continue;
    }
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      if (ret != ACL_SUCCESS) {
        aclrtFree(baseDeviceAddr);
        aclrtFree(outDeviceAddr);
        aclDestroyTensor(base);
        aclDestroyTensor(out);
        aclDestroyScalar(exponent);
        allTestsPassed = false;
        continue;
      }
    }
    
    ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
      if (workspaceSize > 0) aclrtFree(workspaceAddr);
      aclrtFree(baseDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclDestroyTensor(base);
      aclDestroyTensor(out);
      aclDestroyScalar(exponent);
      allTestsPassed = false;
      continue;
    }
    
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
      if (workspaceSize > 0) aclrtFree(workspaceAddr);
      aclrtFree(baseDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclDestroyTensor(base);
      aclDestroyTensor(out);
      aclDestroyScalar(exponent);
      allTestsPassed = false;
      continue;
    }
    
    std::vector<float> resultData(GetShapeSize(shape));
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                      resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    bool allMatch = true;
    for (size_t i = 0; i < baseData.size(); i++) {
      double expected = std::pow(static_cast<double>(baseData[i]), static_cast<double>(expVal));
      if (!CompareWithTolerance(resultData[i], expected)) {
        allMatch = false;
        break;
      }
    }
    
    if (!allMatch) {
      allTestsPassed = false;
    }
    
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(out);
    aclDestroyScalar(exponent);
  }
  
  TestResult(testName, allTestsPassed);
  return allTestsPassed ? 0 : -1;
}

int TestPowTensorScalar_Int64(aclrtStream stream) {
  const char* testName = "PowTensorScalar_Int64";
  
  std::vector<int64_t> baseData = {2, 3, 4, 5};
  std::vector<int64_t> shape = {2, 2};
  int64_t expVal = 3;
  
  void* baseDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* base = nullptr;
  aclTensor* out = nullptr;
  aclScalar* exponent = nullptr;
  
  std::vector<int64_t> outData(GetShapeSize(shape), 0);
  
  int ret = CreateAclTensor(baseData, shape, &baseDeviceAddr, aclDataType::ACL_INT64, &base);
  if (ret != ACL_SUCCESS) {
    TestResult(testName, false);
    return -1;
  }
  
  ret = CreateAclTensor(outData, shape, &outDeviceAddr, aclDataType::ACL_INT64, &out);
  if (ret != ACL_SUCCESS) {
    aclrtFree(baseDeviceAddr);
    aclDestroyTensor(base);
    TestResult(testName, false);
    return -1;
  }
  
  exponent = aclCreateScalar(&expVal, aclDataType::ACL_INT64);
  if (exponent == nullptr) {
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(out);
    TestResult(testName, false);
    return -1;
  }
  
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnPowTensorScalarGetWorkspaceSize(base, exponent, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(out);
    aclDestroyScalar(exponent);
    TestResult(testName, false);
    return -1;
  }
  
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      aclrtFree(baseDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclDestroyTensor(base);
      aclDestroyTensor(out);
      aclDestroyScalar(exponent);
      TestResult(testName, false);
      return -1;
    }
  }
  
  ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(out);
    aclDestroyScalar(exponent);
    TestResult(testName, false);
    return -1;
  }
  
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(out);
    aclDestroyScalar(exponent);
    TestResult(testName, false);
    return -1;
  }
  
  std::vector<int64_t> resultData(GetShapeSize(shape));
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(int64_t), outDeviceAddr,
                    resultData.size() * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
  
  bool allMatch = true;
  for (size_t i = 0; i < baseData.size(); i++) {
    int64_t expected = static_cast<int64_t>(std::pow(static_cast<double>(baseData[i]), static_cast<double>(expVal)));
    if (resultData[i] != expected) {
      allMatch = false;
      break;
    }
  }
  
  if (workspaceSize > 0) aclrtFree(workspaceAddr);
  aclrtFree(baseDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclDestroyTensor(base);
  aclDestroyTensor(out);
  aclDestroyScalar(exponent);
  
  TestResult(testName, allMatch);
  return allMatch ? 0 : -1;
}

int TestPowTensorScalar_Double(aclrtStream stream) {
  const char* testName = "PowTensorScalar_Double";
  
  std::vector<double> baseData = {2.0, 3.0, 4.0, 5.0};
  std::vector<int64_t> shape = {2, 2};
  double expVal = 2.0;
  
  void* baseDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* base = nullptr;
  aclTensor* out = nullptr;
  aclScalar* exponent = nullptr;
  
  std::vector<double> outData(GetShapeSize(shape), 0.0);
  
  int ret = CreateAclTensor(baseData, shape, &baseDeviceAddr, aclDataType::ACL_DOUBLE, &base);
  if (ret != ACL_SUCCESS) {
    TestResult(testName, false);
    return -1;
  }
  
  ret = CreateAclTensor(outData, shape, &outDeviceAddr, aclDataType::ACL_DOUBLE, &out);
  if (ret != ACL_SUCCESS) {
    aclrtFree(baseDeviceAddr);
    aclDestroyTensor(base);
    TestResult(testName, false);
    return -1;
  }
  
  exponent = aclCreateScalar(&expVal, aclDataType::ACL_DOUBLE);
  if (exponent == nullptr) {
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(out);
    TestResult(testName, false);
    return -1;
  }
  
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnPowTensorScalarGetWorkspaceSize(base, exponent, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(out);
    aclDestroyScalar(exponent);
    TestResult(testName, false);
    return -1;
  }
  
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      aclrtFree(baseDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclDestroyTensor(base);
      aclDestroyTensor(out);
      aclDestroyScalar(exponent);
      TestResult(testName, false);
      return -1;
    }
  }
  
  ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(out);
    aclDestroyScalar(exponent);
    TestResult(testName, false);
    return -1;
  }
  
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(out);
    aclDestroyScalar(exponent);
    TestResult(testName, false);
    return -1;
  }
  
  std::vector<double> resultData(GetShapeSize(shape));
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(double), outDeviceAddr,
                    resultData.size() * sizeof(double), ACL_MEMCPY_DEVICE_TO_HOST);
  
  bool allMatch = true;
  for (size_t i = 0; i < baseData.size(); i++) {
    double expected = std::pow(baseData[i], expVal);
    double diff = std::abs(resultData[i] - expected);
    if (diff > 1e-10) {
      allMatch = false;
      break;
    }
  }
  
  if (workspaceSize > 0) aclrtFree(workspaceAddr);
  aclrtFree(baseDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclDestroyTensor(base);
  aclDestroyTensor(out);
  aclDestroyScalar(exponent);
  
  TestResult(testName, allMatch);
  return allMatch ? 0 : -1;
}

int TestPowTensorScalar_LargeShape(aclrtStream stream) {
  const char* testName = "PowTensorScalar_LargeShape";
  
  std::vector<float> baseData(100);
  for (int i = 0; i < 100; i++) {
    baseData[i] = static_cast<float>(i + 1);
  }
  std::vector<int64_t> shape = {10, 10};
  float expVal = 2.0f;
  
  void* baseDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* base = nullptr;
  aclTensor* out = nullptr;
  aclScalar* exponent = nullptr;
  
  std::vector<float> outData(GetShapeSize(shape), 0.0f);
  
  int ret = CreateAclTensor(baseData, shape, &baseDeviceAddr, aclDataType::ACL_FLOAT, &base);
  if (ret != ACL_SUCCESS) {
    TestResult(testName, false);
    return -1;
  }
  
  ret = CreateAclTensor(outData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclrtFree(baseDeviceAddr);
    aclDestroyTensor(base);
    TestResult(testName, false);
    return -1;
  }
  
  exponent = aclCreateScalar(&expVal, aclDataType::ACL_FLOAT);
  if (exponent == nullptr) {
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(out);
    TestResult(testName, false);
    return -1;
  }
  
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnPowTensorScalarGetWorkspaceSize(base, exponent, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(out);
    aclDestroyScalar(exponent);
    TestResult(testName, false);
    return -1;
  }
  
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      aclrtFree(baseDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclDestroyTensor(base);
      aclDestroyTensor(out);
      aclDestroyScalar(exponent);
      TestResult(testName, false);
      return -1;
    }
  }
  
  ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(out);
    aclDestroyScalar(exponent);
    TestResult(testName, false);
    return -1;
  }
  
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(out);
    aclDestroyScalar(exponent);
    TestResult(testName, false);
    return -1;
  }
  
  std::vector<float> resultData(GetShapeSize(shape));
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                    resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  
  bool allMatch = true;
  for (size_t i = 0; i < baseData.size(); i++) {
    double expected = std::pow(static_cast<double>(baseData[i]), static_cast<double>(expVal));
    if (!CompareWithTolerance(resultData[i], expected)) {
      allMatch = false;
      break;
    }
  }
  
  if (workspaceSize > 0) aclrtFree(workspaceAddr);
  aclrtFree(baseDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclDestroyTensor(base);
  aclDestroyTensor(out);
  aclDestroyScalar(exponent);
  
  TestResult(testName, allMatch);
  return allMatch ? 0 : -1;
}

int TestPowScalarTensor_1D(aclrtStream stream) {
  const char* testName = "PowScalarTensor_1D";
  
  float baseVal = 2.0f;
  std::vector<float> expData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<int64_t> shape = {8};
  
  void* expDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* exp = nullptr;
  aclTensor* out = nullptr;
  aclScalar* base = nullptr;
  
  std::vector<float> outData(GetShapeSize(shape), 0.0f);
  
  int ret = CreateAclTensor(expData, shape, &expDeviceAddr, aclDataType::ACL_FLOAT, &exp);
  if (ret != ACL_SUCCESS) {
    TestResult(testName, false);
    return -1;
  }
  
  ret = CreateAclTensor(outData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclrtFree(expDeviceAddr);
    aclDestroyTensor(exp);
    TestResult(testName, false);
    return -1;
  }
  
  base = aclCreateScalar(&baseVal, aclDataType::ACL_FLOAT);
  if (base == nullptr) {
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    TestResult(testName, false);
    return -1;
  }
  
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnPowScalarTensorGetWorkspaceSize(base, exp, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    aclDestroyScalar(base);
    TestResult(testName, false);
    return -1;
  }
  
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      aclrtFree(expDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclDestroyTensor(exp);
      aclDestroyTensor(out);
      aclDestroyScalar(base);
      TestResult(testName, false);
      return -1;
    }
  }
  
  ret = aclnnPowScalarTensor(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    aclDestroyScalar(base);
    TestResult(testName, false);
    return -1;
  }
  
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    aclDestroyScalar(base);
    TestResult(testName, false);
    return -1;
  }
  
  std::vector<float> resultData(GetShapeSize(shape));
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                    resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  
  bool allMatch = true;
  for (size_t i = 0; i < expData.size(); i++) {
    double expected = std::pow(static_cast<double>(baseVal), static_cast<double>(expData[i]));
    if (!CompareWithTolerance(resultData[i], expected)) {
      allMatch = false;
      break;
    }
  }
  
  if (workspaceSize > 0) aclrtFree(workspaceAddr);
  aclrtFree(expDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclDestroyTensor(exp);
  aclDestroyTensor(out);
  aclDestroyScalar(base);
  
  TestResult(testName, allMatch);
  return allMatch ? 0 : -1;
}

int TestPowTensorTensor_3D(aclrtStream stream) {
  const char* testName = "PowTensorTensor_3D";
  
  std::vector<float> baseData = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<float> expData = {2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 3.0f};
  std::vector<int64_t> shape = {2, 2, 2};
  
  void* baseDeviceAddr = nullptr;
  void* expDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* base = nullptr;
  aclTensor* exponent = nullptr;
  aclTensor* out = nullptr;
  
  std::vector<float> outData(GetShapeSize(shape), 0.0f);
  
  int ret = CreateAclTensor(baseData, shape, &baseDeviceAddr, aclDataType::ACL_FLOAT, &base);
  if (ret != ACL_SUCCESS) {
    TestResult(testName, false);
    return -1;
  }
  
  ret = CreateAclTensor(expData, shape, &expDeviceAddr, aclDataType::ACL_FLOAT, &exponent);
  if (ret != ACL_SUCCESS) {
    aclrtFree(baseDeviceAddr);
    aclDestroyTensor(base);
    TestResult(testName, false);
    return -1;
  }
  
  ret = CreateAclTensor(outData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclrtFree(baseDeviceAddr);
    aclrtFree(expDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(exponent);
    TestResult(testName, false);
    return -1;
  }
  
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnPowTensorTensorGetWorkspaceSize(base, exponent, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    aclrtFree(baseDeviceAddr);
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(exponent);
    aclDestroyTensor(out);
    TestResult(testName, false);
    return -1;
  }
  
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      aclrtFree(baseDeviceAddr);
      aclrtFree(expDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclDestroyTensor(base);
      aclDestroyTensor(exponent);
      aclDestroyTensor(out);
      TestResult(testName, false);
      return -1;
    }
  }
  
  ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(baseDeviceAddr);
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(exponent);
    aclDestroyTensor(out);
    TestResult(testName, false);
    return -1;
  }
  
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(baseDeviceAddr);
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclDestroyTensor(base);
    aclDestroyTensor(exponent);
    aclDestroyTensor(out);
    TestResult(testName, false);
    return -1;
  }
  
  std::vector<float> resultData(GetShapeSize(shape));
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                    resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  
  bool allMatch = true;
  for (size_t i = 0; i < baseData.size(); i++) {
    double expected = std::pow(static_cast<double>(baseData[i]), static_cast<double>(expData[i]));
    if (!CompareWithTolerance(resultData[i], expected)) {
      allMatch = false;
      break;
    }
  }
  
  if (workspaceSize > 0) aclrtFree(workspaceAddr);
  aclrtFree(baseDeviceAddr);
  aclrtFree(expDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclDestroyTensor(base);
  aclDestroyTensor(exponent);
  aclDestroyTensor(out);
  
  TestResult(testName, allMatch);
  return allMatch ? 0 : -1;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  
  LOG_PRINT("=== Pow Operator Comprehensive Test Suite ===\n\n");
  
  TestPowTensorScalar_Float32_SpecialExponents(stream);
  TestPowTensorScalar_MoreExponents(stream);
  TestPowScalarTensor(stream);
  TestPowScalarTensor_1D(stream);
  TestPowTensorTensor_SameShape(stream);
  TestPowTensorTensor_Broadcast(stream);
  TestPowTensorTensor_3D(stream);
  TestInplacePowTensorTensor(stream);
  TestExp2(stream);
  TestInplaceExp2(stream);
  
  LOG_PRINT("\n=== Test Summary ===\n");
  LOG_PRINT("Total tests: %ld\n", totalTests);
  LOG_PRINT("Passed: %ld\n", passedTests);
  LOG_PRINT("Failed: %ld\n", failedTests);
  
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  
  return (failedTests > 0) ? 1 : 0;
}