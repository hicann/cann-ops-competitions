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
#include <string>
#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

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

// ==================== 测试结果验证框架 ====================

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

struct TestStats {
  int passed = 0;
  int failed = 0;
  
  void record(bool pass) {
    if (pass) passed++; else failed++;
  }
  
  void printSummary() const {
    printf("\n========== Test Summary ==========\n");
    printf("Total: %d, Passed: %d, Failed: %d\n", passed + failed, passed, failed);
    if (failed == 0) {
      printf("All tests PASSED!\n");
    } else {
      printf("Some tests FAILED!\n");
    }
    printf("==================================\n");
  }
};

// 容差比较函数：|actual - expected| <= atol + rtol * |expected|
bool CompareResult(const float* actual, const float* expected, int64_t size,
                   float atol, float rtol) {
  for (int64_t i = 0; i < size; i++) {
    float diff = std::fabs(actual[i] - expected[i]);
    float tolerance = atol + rtol * std::fabs(expected[i]);
    if (diff > tolerance) {
      printf("  Mismatch at index %ld: actual=%f, expected=%f, diff=%f, tolerance=%f\n",
             i, actual[i], expected[i], diff, tolerance);
      return false;
    }
  }
  return true;
}

// CPU参考计算：逐元素加法 with alpha
template <typename T>
void ReferenceAdd(const T* self, const T* other, T* out, int64_t size, float alpha) {
  for (int64_t i = 0; i < size; i++) {
    out[i] = self[i] + alpha * other[i];
  }
}

// 打印测试结果
void PrintTestResult(const std::string& testName, bool pass) {
  if (pass) {
    printf("[PASS] %s\n", testName.c_str());
  } else {
    printf("[FAIL] %s\n", testName.c_str());
  }
}

// 容差配置
struct ToleranceConfig {
  static constexpr float FLOAT32_ATOL = 1e-5f;
  static constexpr float FLOAT32_RTL = 1e-5f;
  static constexpr float FLOAT16_ATOL = 1e-3f;
  static constexpr float FLOAT16_RTL = 1e-3f;
  static constexpr float BF16_ATOL = 1e-2f;
  static constexpr float BF16_RTL = 1e-2f;
};

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
  CHECK_RET(*tensor != nullptr, return -1);
  return 0;
}

// ==================== 测试用例实现 ====================

// Test 1: Basic FLOAT32 test with alpha=1
bool TestAddFloat32Basic(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Add_FLOAT32_Basic [4,2]+[4,2], alpha=1.0\n");
  bool pass = false;
  void* workspaceAddr = nullptr;
  
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData(8, 0);
  float alphaValue = 1.0f;

  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    printf("  Create self tensor failed\n");
    PrintTestResult("Add_FLOAT32_Basic", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    printf("  Create other tensor failed\n");
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("Add_FLOAT32_Basic", false);
    stats.record(false);
    return false;
  }
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  if (alpha == nullptr) {
    printf("  Create alpha scalar failed\n");
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Add_FLOAT32_Basic", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    printf("  Create out tensor failed\n");
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Add_FLOAT32_Basic", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  
  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      printf("  allocate workspace failed: %d\n", ret);
    }
  }
  
  if (ret == ACL_SUCCESS) {
    ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
      printf("  aclnnAdd failed: %d\n", ret);
    }
  } else {
    printf("  aclnnAddGetWorkspaceSize failed: %d\n", ret);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
      printf("  aclrtSynchronizeStream failed: %d\n", ret);
    }
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
      printf("  copy result failed: %d\n", ret);
    } else {
      // CPU参考计算
      std::vector<float> expectedData(size, 0);
      ReferenceAdd(selfHostData.data(), otherHostData.data(), expectedData.data(), size, alphaValue);
      
      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  
  PrintTestResult("Add_FLOAT32_Basic", pass);
  stats.record(pass);
  return pass;
}

// Test 2: Alpha=0 test (skip if not supported)
bool TestAddAlphaZero(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Add_Alpha_Zero [4,2]+[4,2], alpha=0.0\n");
  bool pass = false;
  void* workspaceAddr = nullptr;
  
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> otherHostData = {10, 20, 30, 40, 50, 60, 70, 80};
  std::vector<float> outHostData(8, 0);
  float alphaValue = 0.0f;

  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    PrintTestResult("Add_Alpha_Zero", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("Add_Alpha_Zero", false);
    stats.record(false);
    return false;
  }
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  if (alpha == nullptr) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Add_Alpha_Zero", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Add_Alpha_Zero", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  
  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }
  
  if (ret == ACL_SUCCESS) {
    ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
      printf("  copy result failed: %d\n", ret);
    } else {
      std::vector<float> expectedData(size, 0);
      ReferenceAdd(selfHostData.data(), otherHostData.data(), expectedData.data(), size, alphaValue);
      
      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  
  PrintTestResult("Add_Alpha_Zero", pass);
  stats.record(pass);
  return pass;
}

// Test 3: Alpha negative test
bool TestAddAlphaNegative(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Add_Alpha_Negative [4,2]+[4,2], alpha=-2.0\n");
  bool pass = false;
  void* workspaceAddr = nullptr;
  
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {10, 10, 10, 10, 10, 10, 10, 10};
  std::vector<float> otherHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> outHostData(8, 0);
  float alphaValue = -2.0f;

  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    PrintTestResult("Add_Alpha_Negative", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("Add_Alpha_Negative", false);
    stats.record(false);
    return false;
  }
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  if (alpha == nullptr) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Add_Alpha_Negative", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Add_Alpha_Negative", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  
  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }
  
  if (ret == ACL_SUCCESS) {
    ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
      printf("  copy result failed: %d\n", ret);
    } else {
      std::vector<float> expectedData(size, 0);
      ReferenceAdd(selfHostData.data(), otherHostData.data(), expectedData.data(), size, alphaValue);
      
      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  
  PrintTestResult("Add_Alpha_Negative", pass);
  stats.record(pass);
  return pass;
}

// Test 4: Alpha float test
bool TestAddAlphaFloat(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Add_Alpha_Float [4,2]+[4,2], alpha=1.5\n");
  bool pass = false;
  void* workspaceAddr = nullptr;
  
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData(8, 0);
  float alphaValue = 1.5f;

  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    PrintTestResult("Add_Alpha_Float", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("Add_Alpha_Float", false);
    stats.record(false);
    return false;
  }
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  if (alpha == nullptr) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Add_Alpha_Float", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Add_Alpha_Float", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  
  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }
  
  if (ret == ACL_SUCCESS) {
    ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
      printf("  copy result failed: %d\n", ret);
    } else {
      std::vector<float> expectedData(size, 0);
      ReferenceAdd(selfHostData.data(), otherHostData.data(), expectedData.data(), size, alphaValue);
      
      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  
  PrintTestResult("Add_Alpha_Float", pass);
  stats.record(pass);
  return pass;
}

// Test 5: Broadcast row test
bool TestAddBroadcastRow(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Add_Broadcast_Row [4,3]+[3], alpha=1.0\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  std::vector<int64_t> selfShape = {4, 3};
  std::vector<int64_t> otherShape = {3};
  std::vector<int64_t> outShape = {4, 3};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> otherHostData = {2, 3, 4};
  std::vector<float> outHostData(12, 0);
  float alphaValue = 1.0f;

  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    PrintTestResult("Add_Broadcast_Row", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("Add_Broadcast_Row", false);
    stats.record(false);
    return false;
  }
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  if (alpha == nullptr) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Add_Broadcast_Row", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Add_Broadcast_Row", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);

  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
      printf("  copy result failed: %d\n", ret);
    } else {
      std::vector<float> expectedData(size, 0);
      for (int64_t i = 0; i < size; i++) {
        int64_t row = i / 3;
        int64_t col = i % 3;
        expectedData[i] = selfHostData[i] + alphaValue * otherHostData[col];
      }

      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);

  PrintTestResult("Add_Broadcast_Row", pass);
  stats.record(pass);
  return pass;
}

// Test 6: Broadcast column test
bool TestAddBroadcastCol(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Add_Broadcast_Col [4,3]+[4,1], alpha=1.0\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  std::vector<int64_t> selfShape = {4, 3};
  std::vector<int64_t> otherShape = {4, 1};
  std::vector<int64_t> outShape = {4, 3};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> otherHostData = {2, 3, 4, 5};
  std::vector<float> outHostData(12, 0);
  float alphaValue = 1.0f;

  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    PrintTestResult("Add_Broadcast_Col", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("Add_Broadcast_Col", false);
    stats.record(false);
    return false;
  }
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  if (alpha == nullptr) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Add_Broadcast_Col", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Add_Broadcast_Col", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);

  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
      printf("  copy result failed: %d\n", ret);
    } else {
      std::vector<float> expectedData(size, 0);
      for (int64_t i = 0; i < size; i++) {
        int64_t row = i / 3;
        int64_t col = i % 3;
        expectedData[i] = selfHostData[i] + alphaValue * otherHostData[row];
      }

      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);

  PrintTestResult("Add_Broadcast_Col", pass);
  stats.record(pass);
  return pass;
}

// Test 7: Adds API test (tensor + scalar)
bool TestAddsFloat32(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Adds_FLOAT32 [4,2]+scalar(2.0), alpha=1.0\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* other = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  float otherValue = 2.0f;
  float alphaValue = 1.0f;
  std::vector<float> outHostData(8, 0);

  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    PrintTestResult("Adds_FLOAT32", false);
    stats.record(false);
    return false;
  }
  other = aclCreateScalar(&otherValue, aclDataType::ACL_FLOAT);
  if (other == nullptr) {
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("Adds_FLOAT32", false);
    stats.record(false);
    return false;
  }
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  if (alpha == nullptr) {
    aclDestroyTensor(self);
    aclDestroyScalar(other);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("Adds_FLOAT32", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclDestroyScalar(other);
    aclDestroyScalar(alpha);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("Adds_FLOAT32", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);

  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclnnAdds(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
      printf("  copy result failed: %d\n", ret);
    } else {
      std::vector<float> expectedData(size, 0);
      for (int64_t i = 0; i < size; i++) {
        expectedData[i] = selfHostData[i] + alphaValue * otherValue;
      }

      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyScalar(other);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);

  PrintTestResult("Adds_FLOAT32", pass);
  stats.record(pass);
  return pass;
}

// Test 8: InplaceAdd API test
bool TestInplaceAddFloat32(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: InplaceAdd_FLOAT32 [4,2]+=[4,2], alpha=1.0\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;

  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  float alphaValue = 1.0f;

  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    PrintTestResult("InplaceAdd_FLOAT32", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("InplaceAdd_FLOAT32", false);
    stats.record(false);
    return false;
  }
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  if (alpha == nullptr) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("InplaceAdd_FLOAT32", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);

  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclnnInplaceAdd(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(selfShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), selfDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
      printf("  copy result failed: %d\n", ret);
    } else {
      std::vector<float> expectedData(size, 0);
      ReferenceAdd(selfHostData.data(), otherHostData.data(), expectedData.data(), size, alphaValue);

      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyScalar(alpha);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);

  PrintTestResult("InplaceAdd_FLOAT32", pass);
  stats.record(pass);
  return pass;
}

// Test 9: AddV3 API test (scalar + tensor)
bool TestAddV3Float32(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: AddV3_FLOAT32 scalar(5.0)+[4,2], alpha=1.0\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclScalar* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;

  float selfValue = 5.0f;
  std::vector<float> otherHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  float alphaValue = 1.0f;
  std::vector<float> outHostData(8, 0);

  self = aclCreateScalar(&selfValue, aclDataType::ACL_FLOAT);
  if (self == nullptr) {
    PrintTestResult("AddV3_FLOAT32", false);
    stats.record(false);
    return false;
  }
  auto ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyScalar(self);
    PrintTestResult("AddV3_FLOAT32", false);
    stats.record(false);
    return false;
  }
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  if (alpha == nullptr) {
    aclDestroyScalar(self);
    aclDestroyTensor(other);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("AddV3_FLOAT32", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyScalar(self);
    aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("AddV3_FLOAT32", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);

  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclnnAddV3(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
      printf("  copy result failed: %d\n", ret);
    } else {
      std::vector<float> expectedData(size, 0);
      for (int64_t i = 0; i < size; i++) {
        expectedData[i] = selfValue + alphaValue * otherHostData[i];
      }

      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyScalar(self);
  aclDestroyTensor(other);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);

  PrintTestResult("AddV3_FLOAT32", pass);
  stats.record(pass);
  return pass;
}

// Test 10: AddV3 with different alpha values
bool TestAddV3AlphaFloat(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: AddV3_Alpha_Float scalar(5.0)+[4,2], alpha=2.5\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclScalar* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;

  float selfValue = 5.0f;
  std::vector<float> otherHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  float alphaValue = 2.5f;
  std::vector<float> outHostData(8, 0);

  self = aclCreateScalar(&selfValue, aclDataType::ACL_FLOAT);
  if (self == nullptr) {
    PrintTestResult("AddV3_Alpha_Float", false);
    stats.record(false);
    return false;
  }
  auto ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyScalar(self);
    PrintTestResult("AddV3_Alpha_Float", false);
    stats.record(false);
    return false;
  }
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  if (alpha == nullptr) {
    aclDestroyScalar(self);
    aclDestroyTensor(other);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("AddV3_Alpha_Float", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyScalar(self);
    aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("AddV3_Alpha_Float", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);

  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclnnAddV3(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
      printf("  copy result failed: %d\n", ret);
    } else {
      std::vector<float> expectedData(size, 0);
      for (int64_t i = 0; i < size; i++) {
        expectedData[i] = selfValue + alphaValue * otherHostData[i];
      }

      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyScalar(self);
  aclDestroyTensor(other);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);

  PrintTestResult("AddV3_Alpha_Float", pass);
  stats.record(pass);
  return pass;
}

// Test 11: AddV3 with alpha=1 (optimized path)
bool TestAddV3AlphaOne(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: AddV3_Alpha_One scalar(5.0)+[4,2], alpha=1.0\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclScalar* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;

  float selfValue = 5.0f;
  std::vector<float> otherHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  float alphaValue = 1.0f;
  std::vector<float> outHostData(8, 0);

  self = aclCreateScalar(&selfValue, aclDataType::ACL_FLOAT);
  if (self == nullptr) {
    PrintTestResult("AddV3_Alpha_One", false);
    stats.record(false);
    return false;
  }
  auto ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyScalar(self);
    PrintTestResult("AddV3_Alpha_One", false);
    stats.record(false);
    return false;
  }
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  if (alpha == nullptr) {
    aclDestroyScalar(self);
    aclDestroyTensor(other);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("AddV3_Alpha_One", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyScalar(self);
    aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("AddV3_Alpha_One", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);

  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclnnAddV3(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
      printf("  copy result failed: %d\n", ret);
    } else {
      std::vector<float> expectedData(size, 0);
      for (int64_t i = 0; i < size; i++) {
        expectedData[i] = selfValue + alphaValue * otherHostData[i];
      }

      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyScalar(self);
  aclDestroyTensor(other);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);

  PrintTestResult("AddV3_Alpha_One", pass);
  stats.record(pass);
  return pass;
}

// Test 12: Zero value test
bool TestAddZeroValues(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Add_Zero_Values [4,2]+[4,2], alpha=1.0\n");
  bool pass = false;
  void* workspaceAddr = nullptr;
  
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> otherHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> outHostData(8, 0);
  float alphaValue = 1.0f;

  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    PrintTestResult("Add_Zero_Values", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("Add_Zero_Values", false);
    stats.record(false);
    return false;
  }
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  if (alpha == nullptr) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Add_Zero_Values", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Add_Zero_Values", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  
  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }
  
  if (ret == ACL_SUCCESS) {
    ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
      printf("  copy result failed: %d\n", ret);
    } else {
      std::vector<float> expectedData(size, 0);
      ReferenceAdd(selfHostData.data(), otherHostData.data(), expectedData.data(), size, alphaValue);
      
      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  
  PrintTestResult("Add_Zero_Values", pass);
  stats.record(pass);
  return pass;
}

// Test 13: Different shape sizes
bool TestAddLargeShape(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Add_Large_Shape [16,16]+[16,16], alpha=1.0\n");
  bool pass = false;
  void* workspaceAddr = nullptr;
  
  std::vector<int64_t> selfShape = {16, 16};
  std::vector<int64_t> otherShape = {16, 16};
  std::vector<int64_t> outShape = {16, 16};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;
  
  int64_t size = 256;
  std::vector<float> selfHostData(size, 1.0f);
  std::vector<float> otherHostData(size, 2.0f);
  std::vector<float> outHostData(size, 0);
  float alphaValue = 1.0f;

  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    PrintTestResult("Add_Large_Shape", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("Add_Large_Shape", false);
    stats.record(false);
    return false;
  }
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  if (alpha == nullptr) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Add_Large_Shape", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Add_Large_Shape", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  
  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }
  
  if (ret == ACL_SUCCESS) {
    ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
      printf("  copy result failed: %d\n", ret);
    } else {
      std::vector<float> expectedData(size, 0);
      ReferenceAdd(selfHostData.data(), otherHostData.data(), expectedData.data(), size, alphaValue);
      
      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  
  PrintTestResult("Add_Large_Shape", pass);
  stats.record(pass);
  return pass;
}

// Test 14: InplaceAdds API test
bool TestInplaceAddsFloat32(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: InplaceAdds_FLOAT32 [4,2]+=scalar(2.0), alpha=1.0\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  std::vector<int64_t> selfShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* other = nullptr;
  aclScalar* alpha = nullptr;

  std::vector<float> selfHostData = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  float otherValue = 2.0f;
  float alphaValue = 1.0f;

  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    PrintTestResult("InplaceAdds_FLOAT32", false);
    stats.record(false);
    return false;
  }
  other = aclCreateScalar(&otherValue, aclDataType::ACL_FLOAT);
  if (other == nullptr) {
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("InplaceAdds_FLOAT32", false);
    stats.record(false);
    return false;
  }
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  if (alpha == nullptr) {
    aclDestroyTensor(self);
    aclDestroyScalar(other);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("InplaceAdds_FLOAT32", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnInplaceAddsGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);

  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclnnInplaceAdds(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(selfShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), selfDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
      printf("  copy result failed: %d\n", ret);
    } else {
      std::vector<float> expectedData(size, 0);
      for (int64_t i = 0; i < size; i++) {
        expectedData[i] = selfHostData[i] + alphaValue * otherValue;
      }

      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyScalar(other);
  aclDestroyScalar(alpha);
  aclrtFree(selfDeviceAddr);

  PrintTestResult("InplaceAdds_FLOAT32", pass);
  stats.record(pass);
  return pass;
}

// Test 15: InplaceAddV3 API test
bool TestInplaceAddV3Float32(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: InplaceAddV3_FLOAT32 scalar(5.0)+=[4,2], alpha=1.0\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  std::vector<int64_t> selfShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  aclScalar* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;

  float selfValue = 5.0f;
  std::vector<float> otherHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  float alphaValue = 1.0f;

  self = aclCreateScalar(&selfValue, aclDataType::ACL_FLOAT);
  if (self == nullptr) {
    PrintTestResult("InplaceAddV3_FLOAT32", false);
    stats.record(false);
    return false;
  }
  auto ret = CreateAclTensor(otherHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyScalar(self);
    PrintTestResult("InplaceAddV3_FLOAT32", false);
    stats.record(false);
    return false;
  }
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  if (alpha == nullptr) {
    aclDestroyScalar(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("InplaceAddV3_FLOAT32", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnInplaceAddV3GetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);

  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclnnInplaceAddV3(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(selfShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), selfDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
      printf("  copy result failed: %d\n", ret);
    } else {
      std::vector<float> expectedData(size, 0);
      for (int64_t i = 0; i < size; i++) {
        expectedData[i] = selfValue + alphaValue * otherHostData[i];
      }

      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyScalar(self);
  aclDestroyTensor(other);
  aclDestroyScalar(alpha);
  aclrtFree(selfDeviceAddr);

  PrintTestResult("InplaceAddV3_FLOAT32", pass);
  stats.record(pass);
  return pass;
}

// Test 16: AddV3 with different scalar types (INT32 scalar)
bool TestAddV3ScalarInt32(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: AddV3_Scalar_INT32 int32(5)+[4,2], alpha=1.0\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclScalar* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;

  int32_t selfValue = 5;
  std::vector<float> otherHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  float alphaValue = 1.0f;
  std::vector<float> outHostData(8, 0);

  self = aclCreateScalar(&selfValue, aclDataType::ACL_INT32);
  if (self == nullptr) {
    PrintTestResult("AddV3_Scalar_INT32", false);
    stats.record(false);
    return false;
  }
  auto ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyScalar(self);
    PrintTestResult("AddV3_Scalar_INT32", false);
    stats.record(false);
    return false;
  }
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  if (alpha == nullptr) {
    aclDestroyScalar(self);
    aclDestroyTensor(other);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("AddV3_Scalar_INT32", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyScalar(self);
    aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("AddV3_Scalar_INT32", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);

  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclnnAddV3(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
      printf("  copy result failed: %d\n", ret);
    } else {
      std::vector<float> expectedData(size, 0);
      for (int64_t i = 0; i < size; i++) {
        expectedData[i] = (float)selfValue + alphaValue * otherHostData[i];
      }

      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyScalar(self);
  aclDestroyTensor(other);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);

  PrintTestResult("AddV3_Scalar_INT32", pass);
  stats.record(pass);
  return pass;
}

// Test 17: AddV3 with FLOAT16 tensor
bool TestAddV3Float16(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: AddV3_Float16 scalar(5.0)+[4,2]FLOAT16, alpha=1.0\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclScalar* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;

  float selfValue = 5.0f;
  std::vector<float> otherHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  float alphaValue = 1.0f;
  std::vector<float> outHostData(8, 0);

  self = aclCreateScalar(&selfValue, aclDataType::ACL_FLOAT);
  if (self == nullptr) {
    PrintTestResult("AddV3_Float16", false);
    stats.record(false);
    return false;
  }
  auto ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT16, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyScalar(self);
    PrintTestResult("AddV3_Float16", false);
    stats.record(false);
    return false;
  }
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  if (alpha == nullptr) {
    aclDestroyScalar(self);
    aclDestroyTensor(other);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("AddV3_Float16", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyScalar(self);
    aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("AddV3_Float16", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);

  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclnnAddV3(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
      printf("  copy result failed: %d\n", ret);
    } else {
      std::vector<float> expectedData(size, 0);
      for (int64_t i = 0; i < size; i++) {
        expectedData[i] = selfValue + alphaValue * otherHostData[i];
      }

      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT16_ATOL, ToleranceConfig::FLOAT16_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyScalar(self);
  aclDestroyTensor(other);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);

  PrintTestResult("AddV3_Float16", pass);
  stats.record(pass);
  return pass;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  int ret = Init(deviceId, &stream);
  if (ret != 0) {
    printf("Init acl failed\n");
    return ret;
  }

  TestStats stats;

  // 运行所有测试用例
  TestAddFloat32Basic(stream, stats);
  TestAddAlphaZero(stream, stats);
  TestAddAlphaNegative(stream, stats);
  TestAddAlphaFloat(stream, stats);
  TestAddBroadcastRow(stream, stats);
  TestAddBroadcastCol(stream, stats);
  TestAddsFloat32(stream, stats);
  TestInplaceAddFloat32(stream, stats);
  TestAddV3Float32(stream, stats);
  TestAddV3AlphaFloat(stream, stats);
  TestAddV3AlphaOne(stream, stats);
  TestAddZeroValues(stream, stats);
  TestAddLargeShape(stream, stats);
  TestInplaceAddsFloat32(stream, stats);
  TestInplaceAddV3Float32(stream, stats);
  TestAddV3ScalarInt32(stream, stats);
  TestAddV3Float16(stream, stats);

  // 输出汇总
  stats.printSummary();

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return (stats.failed > 0) ? 1 : 0;
}
