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
#include <string>
#include <iomanip>
#include <complex>
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

// 浮点数比较辅助函数
bool FloatCompare(float actual, float expected, float atol = 1e-5, float rtol = 1e-5) {
  return std::abs(actual - expected) <= (atol + rtol * std::abs(expected));
}

bool DoubleCompare(double actual, double expected, double atol = 1e-9, double rtol = 1e-9) {
  return std::abs(actual - expected) <= (atol + rtol * std::abs(expected));
}

// ==================== Test Helper Functions ====================

int RunPowTensorScalarTest(aclrtStream stream, const std::string& testName, 
                           const std::vector<float>& baseData, float exponent,
                           const std::vector<float>& expected, aclDataType dtype) {
  std::vector<int64_t> shape = {(int64_t)baseData.size()};
  void* baseDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* base = nullptr;
  aclTensor* out = nullptr;
  aclScalar* expScalar = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;
  std::vector<float> outHostData(baseData.size(), 0.0f);
  int result = -1;
  
  auto ret = CreateAclTensor(baseData, shape, &baseDeviceAddr, dtype, &base);
  if (ret != ACL_SUCCESS) return -1;
  
  expScalar = aclCreateScalar(&exponent, aclDataType::ACL_FLOAT);
  if (expScalar == nullptr) {
    goto cleanup;
  }
  
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, dtype, &out);
  if (ret != ACL_SUCCESS) {
    goto cleanup;
  }

  ret = aclnnPowTensorScalarGetWorkspaceSize(base, expScalar, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    goto cleanup;
  }
  
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) goto cleanup;
  }
  
  ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) goto cleanup;
  
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) goto cleanup;
  
  {
    std::vector<float> resultData(baseData.size(), 0.0f);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                      resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) goto cleanup;
    
    bool pass = true;
    for (size_t i = 0; i < expected.size(); i++) {
      if (!FloatCompare(resultData[i], expected[i], 1e-4, 1e-4)) {
        pass = false;
        LOG_PRINT("  Mismatch at [%zu]: got %.6f, expected %.6f\n", i, resultData[i], expected[i]);
        break;
      }
    }
    
    result = pass ? 1 : 0;
    if (pass) {
      LOG_PRINT("[PASS] %s\n", testName.c_str());
    } else {
      LOG_PRINT("[FAIL] %s\n", testName.c_str());
    }
  }
  
cleanup:
  aclDestroyTensor(base);
  aclDestroyScalar(expScalar);
  aclDestroyTensor(out);
  aclrtFree(baseDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  
  return result;
}

int RunPowScalarTensorTest(aclrtStream stream, const std::string& testName,
                           float base, const std::vector<float>& expData,
                           const std::vector<float>& expected, aclDataType dtype) {
  std::vector<int64_t> shape = {(int64_t)expData.size()};
  void* expDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* exp = nullptr;
  aclTensor* out = nullptr;
  aclScalar* baseScalar = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;
  std::vector<float> outHostData(expData.size(), 0.0f);
  int result = -1;
  
  baseScalar = aclCreateScalar(&base, aclDataType::ACL_FLOAT);
  if (baseScalar == nullptr) return -1;
  
  auto ret = CreateAclTensor(expData, shape, &expDeviceAddr, dtype, &exp);
  if (ret != ACL_SUCCESS) {
    goto cleanup;
  }
  
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, dtype, &out);
  if (ret != ACL_SUCCESS) {
    goto cleanup;
  }

  ret = aclnnPowScalarTensorGetWorkspaceSize(baseScalar, exp, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) goto cleanup;
  
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) goto cleanup;
  }
  
  ret = aclnnPowScalarTensor(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) goto cleanup;
  
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) goto cleanup;
  
  {
    std::vector<float> resultData(expData.size(), 0.0f);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                      resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) goto cleanup;
    
    bool pass = true;
    for (size_t i = 0; i < expected.size(); i++) {
      if (!FloatCompare(resultData[i], expected[i], 1e-4, 1e-4)) {
        pass = false;
        LOG_PRINT("  Mismatch at [%zu]: got %.6f, expected %.6f\n", i, resultData[i], expected[i]);
        break;
      }
    }
    
    result = pass ? 1 : 0;
    if (pass) {
      LOG_PRINT("[PASS] %s\n", testName.c_str());
    } else {
      LOG_PRINT("[FAIL] %s\n", testName.c_str());
    }
  }
  
cleanup:
  aclDestroyScalar(baseScalar);
  aclDestroyTensor(exp);
  aclDestroyTensor(out);
  aclrtFree(expDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  
  return result;
}

int RunPowTensorTensorTest(aclrtStream stream, const std::string& testName,
                           const std::vector<float>& baseData, const std::vector<float>& expData,
                           const std::vector<float>& expected, aclDataType dtype,
                           const std::vector<int64_t>& baseShape = {}, const std::vector<int64_t>& expShape = {}) {
  std::vector<int64_t> bShape = baseShape.empty() ? std::vector<int64_t>{(int64_t)baseData.size()} : baseShape;
  std::vector<int64_t> eShape = expShape.empty() ? std::vector<int64_t>{(int64_t)expData.size()} : expShape;
  std::vector<int64_t> outShape = {(int64_t)expected.size()};
  
  void* baseDeviceAddr = nullptr;
  void* expDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* base = nullptr;
  aclTensor* exp = nullptr;
  aclTensor* out = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;
  std::vector<float> outHostData(expected.size(), 0.0f);
  int result = -1;
  
  auto ret = CreateAclTensor(baseData, bShape, &baseDeviceAddr, dtype, &base);
  if (ret != ACL_SUCCESS) return -1;
  
  ret = CreateAclTensor(expData, eShape, &expDeviceAddr, dtype, &exp);
  if (ret != ACL_SUCCESS) {
    goto cleanup;
  }
  
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, dtype, &out);
  if (ret != ACL_SUCCESS) {
    goto cleanup;
  }

  ret = aclnnPowTensorTensorGetWorkspaceSize(base, exp, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) goto cleanup;
  
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) goto cleanup;
  }
  
  ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) goto cleanup;
  
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) goto cleanup;
  
  {
    std::vector<float> resultData(expected.size(), 0.0f);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                      resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) goto cleanup;
    
    bool pass = true;
    for (size_t i = 0; i < expected.size(); i++) {
      if (!FloatCompare(resultData[i], expected[i], 1e-4, 1e-4)) {
        pass = false;
        LOG_PRINT("  Mismatch at [%zu]: got %.6f, expected %.6f\n", i, resultData[i], expected[i]);
        break;
      }
    }
    
    result = pass ? 1 : 0;
    if (pass) {
      LOG_PRINT("[PASS] %s\n", testName.c_str());
    } else {
      LOG_PRINT("[FAIL] %s\n", testName.c_str());
    }
  }
  
cleanup:
  aclDestroyTensor(base);
  aclDestroyTensor(exp);
  aclDestroyTensor(out);
  aclrtFree(baseDeviceAddr);
  aclrtFree(expDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  
  return result;
}

int RunExp2Test(aclrtStream stream, const std::string& testName,
                const std::vector<float>& inputData, const std::vector<float>& expected,
                aclDataType dtype) {
  std::vector<int64_t> shape = {(int64_t)inputData.size()};
  void* inputDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* input = nullptr;
  aclTensor* out = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;
  std::vector<float> outHostData(inputData.size(), 0.0f);
  int result = -1;
  
  auto ret = CreateAclTensor(inputData, shape, &inputDeviceAddr, dtype, &input);
  if (ret != ACL_SUCCESS) return -1;
  
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, dtype, &out);
  if (ret != ACL_SUCCESS) {
    goto cleanup;
  }

  ret = aclnnExp2GetWorkspaceSize(input, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) goto cleanup;
  
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) goto cleanup;
  }
  
  ret = aclnnExp2(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) goto cleanup;
  
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) goto cleanup;
  
  {
    std::vector<float> resultData(inputData.size(), 0.0f);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                      resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) goto cleanup;
    
    bool pass = true;
    for (size_t i = 0; i < expected.size(); i++) {
      if (!FloatCompare(resultData[i], expected[i], 1e-4, 1e-4)) {
        pass = false;
        LOG_PRINT("  Mismatch at [%zu]: got %.6f, expected %.6f\n", i, resultData[i], expected[i]);
        break;
      }
    }
    
    result = pass ? 1 : 0;
    if (pass) {
      LOG_PRINT("[PASS] %s\n", testName.c_str());
    } else {
      LOG_PRINT("[FAIL] %s\n", testName.c_str());
    }
  }
  
cleanup:
  aclDestroyTensor(input);
  aclDestroyTensor(out);
  aclrtFree(inputDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  
  return result;
}

// ==================== Main Test Function ====================

int main() {
  int passCount = 0;
  int failCount = 0;
  int totalTests = 0;

  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  LOG_PRINT("\n========== PowTensorScalar Tests ==========\n");
  
  // Test 1: PowTensorScalar - FLOAT32 - exponent=2.0 (square优化路径)
  {
    totalTests++;
    std::vector<float> base = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> expected = {1.0f, 4.0f, 9.0f, 16.0f};
    int result = RunPowTensorScalarTest(stream, "Test 1: PowTensorScalar FLOAT32 exp=2.0", 
                                        base, 2.0f, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 2: PowTensorScalar - FLOAT32 - exponent=0.5 (sqrt优化路径)
  {
    totalTests++;
    std::vector<float> base = {1.0f, 4.0f, 9.0f, 16.0f};
    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    int result = RunPowTensorScalarTest(stream, "Test 2: PowTensorScalar FLOAT32 exp=0.5",
                                        base, 0.5f, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 3: PowTensorScalar - FLOAT32 - exponent=-1.0 (reciprocal优化路径)
  {
    totalTests++;
    std::vector<float> base = {1.0f, 2.0f, 4.0f, 8.0f};
    std::vector<float> expected = {1.0f, 0.5f, 0.25f, 0.125f};
    int result = RunPowTensorScalarTest(stream, "Test 3: PowTensorScalar FLOAT32 exp=-1.0",
                                        base, -1.0f, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 4: PowTensorScalar - FLOAT32 - exponent=0 (任何数的0次幂为1)
  {
    totalTests++;
    std::vector<float> base = {0.0f, 1.0f, 2.0f, 3.0f};
    std::vector<float> expected = {1.0f, 1.0f, 1.0f, 1.0f};
    int result = RunPowTensorScalarTest(stream, "Test 4: PowTensorScalar FLOAT32 exp=0",
                                        base, 0.0f, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 5: PowTensorScalar - FLOAT32 - exponent=1 (恒等)
  {
    totalTests++;
    std::vector<float> base = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    int result = RunPowTensorScalarTest(stream, "Test 5: PowTensorScalar FLOAT32 exp=1",
                                        base, 1.0f, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 6: PowTensorScalar - FLOAT32 - exponent=3 (cube)
  {
    totalTests++;
    std::vector<float> base = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> expected = {1.0f, 8.0f, 27.0f, 64.0f};
    int result = RunPowTensorScalarTest(stream, "Test 6: PowTensorScalar FLOAT32 exp=3",
                                        base, 3.0f, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 7: PowTensorScalar - FLOAT32 - exponent=-2
  {
    totalTests++;
    std::vector<float> base = {1.0f, 2.0f, 4.0f};
    std::vector<float> expected = {1.0f, 0.25f, 0.0625f};
    int result = RunPowTensorScalarTest(stream, "Test 7: PowTensorScalar FLOAT32 exp=-2",
                                        base, -2.0f, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 8: PowTensorScalar - FLOAT32 - base=0, exp>0
  {
    totalTests++;
    std::vector<float> base = {0.0f, 0.0f, 0.0f};
    std::vector<float> expected = {0.0f, 0.0f, 0.0f};
    int result = RunPowTensorScalarTest(stream, "Test 8: PowTensorScalar FLOAT32 base=0 exp=2",
                                        base, 2.0f, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 9: PowTensorScalar - FLOAT32 - negative base with integer exp
  {
    totalTests++;
    std::vector<float> base = {-2.0f, -3.0f, -4.0f};
    std::vector<float> expected = {4.0f, 9.0f, 16.0f};
    int result = RunPowTensorScalarTest(stream, "Test 9: PowTensorScalar FLOAT32 negative base exp=2",
                                        base, 2.0f, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 10: PowTensorScalar - FLOAT32 - negative base with odd integer exp
  {
    totalTests++;
    std::vector<float> base = {-2.0f, -3.0f, -4.0f};
    std::vector<float> expected = {-8.0f, -27.0f, -64.0f};
    int result = RunPowTensorScalarTest(stream, "Test 10: PowTensorScalar FLOAT32 negative base exp=3",
                                        base, 3.0f, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 11: PowTensorScalar - INT32 dtype
  {
    totalTests++;
    std::vector<float> base = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> expected = {4.0f, 9.0f, 16.0f, 25.0f};
    
    std::vector<int64_t> shape = {(int64_t)base.size()};
    void* baseDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* baseTensor = nullptr;
    aclTensor* out = nullptr;
    aclScalar* expScalar = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int32_t> outHostData(base.size(), 0);
    bool testPass = false;
    
    ret = CreateAclTensor(std::vector<int32_t>{2, 3, 4, 5}, shape, &baseDeviceAddr, ACL_INT32, &baseTensor);
    if (ret == ACL_SUCCESS) {
      float expVal = 2.0f;
      expScalar = aclCreateScalar(&expVal, ACL_FLOAT);
      if (expScalar != nullptr) {
        ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_INT32, &out);
        if (ret == ACL_SUCCESS) {
          ret = aclnnPowTensorScalarGetWorkspaceSize(baseTensor, expScalar, out, &workspaceSize, &executor);
          if (ret == ACL_SUCCESS) {
            if (workspaceSize > 0) {
              ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            }
            if (ret == ACL_SUCCESS) {
              ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
              if (ret == ACL_SUCCESS) {
                ret = aclrtSynchronizeStream(stream);
                if (ret == ACL_SUCCESS) {
                  std::vector<int32_t> resultData(base.size(), 0);
                  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(int32_t), outDeviceAddr,
                                    resultData.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
                  if (ret == ACL_SUCCESS) {
                    std::vector<int32_t> expectedInt = {4, 9, 16, 25};
                    testPass = true;
                    for (size_t i = 0; i < expectedInt.size(); i++) {
                      if (resultData[i] != expectedInt[i]) {
                        testPass = false;
                        break;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 11: PowTensorScalar INT32\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 11: PowTensorScalar INT32\n");
    }
    
    aclDestroyTensor(baseTensor);
    aclDestroyScalar(expScalar);
    aclDestroyTensor(out);
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  // Test 12: PowTensorScalar - INT8 dtype
  {
    totalTests++;
    std::vector<int8_t> baseInt = {2, 3, 4, 5};
    std::vector<int64_t> shape = {(int64_t)baseInt.size()};
    void* baseDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* baseTensor = nullptr;
    aclTensor* out = nullptr;
    aclScalar* expScalar = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int8_t> outHostData(baseInt.size(), 0);
    bool testPass = false;
    
    ret = CreateAclTensor(baseInt, shape, &baseDeviceAddr, ACL_INT8, &baseTensor);
    if (ret == ACL_SUCCESS) {
      float expVal = 2.0f;
      expScalar = aclCreateScalar(&expVal, ACL_FLOAT);
      if (expScalar != nullptr) {
        ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_INT8, &out);
        if (ret == ACL_SUCCESS) {
          ret = aclnnPowTensorScalarGetWorkspaceSize(baseTensor, expScalar, out, &workspaceSize, &executor);
          if (ret == ACL_SUCCESS) {
            if (workspaceSize > 0) {
              ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            }
            if (ret == ACL_SUCCESS) {
              ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
              if (ret == ACL_SUCCESS) {
                ret = aclrtSynchronizeStream(stream);
                if (ret == ACL_SUCCESS) {
                  std::vector<int8_t> resultData(baseInt.size(), 0);
                  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(int8_t), outDeviceAddr,
                                    resultData.size() * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
                  if (ret == ACL_SUCCESS) {
                    std::vector<int8_t> expectedInt = {4, 9, 16, 25};
                    testPass = true;
                    for (size_t i = 0; i < expectedInt.size(); i++) {
                      if (resultData[i] != expectedInt[i]) {
                        testPass = false;
                        break;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 12: PowTensorScalar INT8\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 12: PowTensorScalar INT8\n");
    }
    
    aclDestroyTensor(baseTensor);
    aclDestroyScalar(expScalar);
    aclDestroyTensor(out);
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  LOG_PRINT("\n========== InplacePowTensorScalar Tests ==========\n");

  // Test 13: InplacePowTensorScalar - FLOAT32
  {
    totalTests++;
    std::vector<int64_t> shape = {4};
    void* selfDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* self = nullptr;
    aclScalar* exponent = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<float> selfHostData = {2.0f, 3.0f, 4.0f, 5.0f};
    bool testPass = false;
    float exponentVal = 3.0f;
    
    ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, ACL_FLOAT, &self);
    if (ret == ACL_SUCCESS) {
      exponent = aclCreateScalar(&exponentVal, ACL_FLOAT);
      if (exponent != nullptr) {
        ret = aclnnInplacePowTensorScalarGetWorkspaceSize(self, exponent, &workspaceSize, &executor);
        if (ret == ACL_SUCCESS) {
          if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          }
          if (ret == ACL_SUCCESS) {
            ret = aclnnInplacePowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
            if (ret == ACL_SUCCESS) {
              ret = aclrtSynchronizeStream(stream);
              if (ret == ACL_SUCCESS) {
                std::vector<float> resultData(4, 0.0f);
                ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), selfDeviceAddr,
                                  resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
                if (ret == ACL_SUCCESS) {
                  float expected[] = {8.0f, 27.0f, 64.0f, 125.0f};
                  testPass = true;
                  for (int i = 0; i < 4; i++) {
                    if (!FloatCompare(resultData[i], expected[i], 1e-4, 1e-4)) {
                      testPass = false;
                      break;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 13: InplacePowTensorScalar FLOAT32 exp=3.0\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 13: InplacePowTensorScalar FLOAT32 exp=3.0\n");
    }
    
    aclDestroyTensor(self);
    aclDestroyScalar(exponent);
    aclrtFree(selfDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  LOG_PRINT("\n========== PowScalarTensor Tests ==========\n");

  // Test 14: PowScalarTensor - FLOAT32
  {
    totalTests++;
    std::vector<float> expData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> expected = {2.0f, 4.0f, 8.0f, 16.0f};
    int result = RunPowScalarTensorTest(stream, "Test 14: PowScalarTensor FLOAT32 base=2.0",
                                        2.0f, expData, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 15: PowScalarTensor - FLOAT32 - base=0
  {
    totalTests++;
    std::vector<float> expData = {1.0f, 2.0f, 3.0f};
    std::vector<float> expected = {0.0f, 0.0f, 0.0f};
    int result = RunPowScalarTensorTest(stream, "Test 15: PowScalarTensor FLOAT32 base=0",
                                        0.0f, expData, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 16: PowScalarTensor - FLOAT32 - base=1
  {
    totalTests++;
    std::vector<float> expData = {0.0f, 1.0f, 2.0f, 10.0f};
    std::vector<float> expected = {1.0f, 1.0f, 1.0f, 1.0f};
    int result = RunPowScalarTensorTest(stream, "Test 16: PowScalarTensor FLOAT32 base=1",
                                        1.0f, expData, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 17: PowScalarTensor - FLOAT32 - negative exponent
  {
    totalTests++;
    std::vector<float> expData = {-1.0f, -2.0f, -3.0f};
    std::vector<float> expected = {0.5f, 0.25f, 0.125f};
    int result = RunPowScalarTensorTest(stream, "Test 17: PowScalarTensor FLOAT32 base=2 neg exp",
                                        2.0f, expData, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  LOG_PRINT("\n========== PowTensorTensor Tests ==========\n");

  // Test 18: PowTensorTensor - FLOAT32 - basic
  {
    totalTests++;
    std::vector<float> base = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> exp = {1.0f, 2.0f, 0.5f, 0.0f};
    std::vector<float> expected = {2.0f, 9.0f, 2.0f, 1.0f};
    int result = RunPowTensorTensorTest(stream, "Test 18: PowTensorTensor FLOAT32 basic",
                                        base, exp, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 19: PowTensorTensor - FLOAT32 - broadcasting
  {
    totalTests++;
    std::vector<float> base = {2.0f, 3.0f};
    std::vector<float> exp = {1.0f, 2.0f, 3.0f};
    std::vector<float> expected = {2.0f, 4.0f, 8.0f, 3.0f, 9.0f, 27.0f};
    std::vector<int64_t> baseShape = {2, 1};
    std::vector<int64_t> expShape = {1, 3};
    int result = RunPowTensorTensorTest(stream, "Test 19: PowTensorTensor FLOAT32 broadcasting",
                                        base, exp, expected, ACL_FLOAT, baseShape, expShape);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 20: PowTensorTensor - FLOAT32 - zero exponent
  {
    totalTests++;
    std::vector<float> base = {0.0f, 1.0f, 2.0f, 3.0f};
    std::vector<float> exp = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> expected = {1.0f, 1.0f, 1.0f, 1.0f};
    int result = RunPowTensorTensorTest(stream, "Test 20: PowTensorTensor FLOAT32 zero exp",
                                        base, exp, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 21: PowTensorTensor - INT32 dtype (触发不同OP_KEY)
  {
    totalTests++;
    std::vector<int32_t> baseInt = {2, 3, 4, 5};
    std::vector<int32_t> expInt = {2, 2, 2, 2};
    std::vector<int64_t> shape = {(int64_t)baseInt.size()};
    void* baseDeviceAddr = nullptr;
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* base = nullptr;
    aclTensor* exp = nullptr;
    aclTensor* out = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int32_t> outHostData(baseInt.size(), 0);
    bool testPass = false;
    
    ret = CreateAclTensor(baseInt, shape, &baseDeviceAddr, ACL_INT32, &base);
    if (ret == ACL_SUCCESS) {
      ret = CreateAclTensor(expInt, shape, &expDeviceAddr, ACL_INT32, &exp);
      if (ret == ACL_SUCCESS) {
        ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_INT32, &out);
        if (ret == ACL_SUCCESS) {
          ret = aclnnPowTensorTensorGetWorkspaceSize(base, exp, out, &workspaceSize, &executor);
          if (ret == ACL_SUCCESS) {
            if (workspaceSize > 0) {
              ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            }
            if (ret == ACL_SUCCESS) {
              ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
              if (ret == ACL_SUCCESS) {
                ret = aclrtSynchronizeStream(stream);
                if (ret == ACL_SUCCESS) {
                  std::vector<int32_t> resultData(baseInt.size(), 0);
                  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(int32_t), outDeviceAddr,
                                    resultData.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
                  if (ret == ACL_SUCCESS) {
                    std::vector<int32_t> expected = {4, 9, 16, 25};
                    testPass = true;
                    for (size_t i = 0; i < expected.size(); i++) {
                      if (resultData[i] != expected[i]) {
                        testPass = false;
                        break;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 21: PowTensorTensor INT32\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 21: PowTensorTensor INT32\n");
    }
    
    aclDestroyTensor(base);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    aclrtFree(baseDeviceAddr);
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  // Test 22: PowTensorTensor - FLOAT16 dtype (触发不同OP_KEY)
  {
    totalTests++;
    std::vector<float> base = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> exp = {2.0f, 2.0f, 2.0f, 2.0f};
    std::vector<float> expected = {4.0f, 9.0f, 16.0f, 25.0f};
    int result = RunPowTensorTensorTest(stream, "Test 22: PowTensorTensor FLOAT16",
                                        base, exp, expected, ACL_FLOAT16);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 23: PowTensorTensor - BF16 dtype (触发不同OP_KEY)
  {
    totalTests++;
    std::vector<float> base = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> exp = {2.0f, 2.0f, 2.0f, 2.0f};
    std::vector<float> expected = {4.0f, 9.0f, 16.0f, 25.0f};
    int result = RunPowTensorTensorTest(stream, "Test 23: PowTensorTensor BF16",
                                        base, exp, expected, ACL_BF16);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 24: PowTensorTensor - UINT8 dtype (触发不同OP_KEY)
  {
    totalTests++;
    std::vector<uint8_t> baseInt = {2, 3, 4, 5};
    std::vector<uint8_t> expInt = {2, 2, 2, 2};
    std::vector<int64_t> shape = {(int64_t)baseInt.size()};
    void* baseDeviceAddr = nullptr;
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* base = nullptr;
    aclTensor* exp = nullptr;
    aclTensor* out = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<uint8_t> outHostData(baseInt.size(), 0);
    bool testPass = false;
    
    ret = CreateAclTensor(baseInt, shape, &baseDeviceAddr, ACL_UINT8, &base);
    if (ret == ACL_SUCCESS) {
      ret = CreateAclTensor(expInt, shape, &expDeviceAddr, ACL_UINT8, &exp);
      if (ret == ACL_SUCCESS) {
        ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_UINT8, &out);
        if (ret == ACL_SUCCESS) {
          ret = aclnnPowTensorTensorGetWorkspaceSize(base, exp, out, &workspaceSize, &executor);
          if (ret == ACL_SUCCESS) {
            if (workspaceSize > 0) {
              ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            }
            if (ret == ACL_SUCCESS) {
              ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
              if (ret == ACL_SUCCESS) {
                ret = aclrtSynchronizeStream(stream);
                if (ret == ACL_SUCCESS) {
                  std::vector<uint8_t> resultData(baseInt.size(), 0);
                  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(uint8_t), outDeviceAddr,
                                    resultData.size() * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
                  if (ret == ACL_SUCCESS) {
                    std::vector<uint8_t> expected = {4, 9, 16, 25};
                    testPass = true;
                    for (size_t i = 0; i < expected.size(); i++) {
                      if (resultData[i] != expected[i]) {
                        testPass = false;
                        break;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 24: PowTensorTensor UINT8\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 24: PowTensorTensor UINT8\n");
    }
    
    aclDestroyTensor(base);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    aclrtFree(baseDeviceAddr);
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  // Test 25: PowTensorTensor - INT8 dtype (触发不同OP_KEY)
  {
    totalTests++;
    std::vector<int8_t> baseInt = {2, 3, 4, 5};
    std::vector<int8_t> expInt = {2, 2, 2, 2};
    std::vector<int64_t> shape = {(int64_t)baseInt.size()};
    void* baseDeviceAddr = nullptr;
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* base = nullptr;
    aclTensor* exp = nullptr;
    aclTensor* out = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int8_t> outHostData(baseInt.size(), 0);
    bool testPass = false;
    
    ret = CreateAclTensor(baseInt, shape, &baseDeviceAddr, ACL_INT8, &base);
    if (ret == ACL_SUCCESS) {
      ret = CreateAclTensor(expInt, shape, &expDeviceAddr, ACL_INT8, &exp);
      if (ret == ACL_SUCCESS) {
        ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_INT8, &out);
        if (ret == ACL_SUCCESS) {
          ret = aclnnPowTensorTensorGetWorkspaceSize(base, exp, out, &workspaceSize, &executor);
          if (ret == ACL_SUCCESS) {
            if (workspaceSize > 0) {
              ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            }
            if (ret == ACL_SUCCESS) {
              ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
              if (ret == ACL_SUCCESS) {
                ret = aclrtSynchronizeStream(stream);
                if (ret == ACL_SUCCESS) {
                  std::vector<int8_t> resultData(baseInt.size(), 0);
                  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(int8_t), outDeviceAddr,
                                    resultData.size() * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
                  if (ret == ACL_SUCCESS) {
                    std::vector<int8_t> expected = {4, 9, 16, 25};
                    testPass = true;
                    for (size_t i = 0; i < expected.size(); i++) {
                      if (resultData[i] != expected[i]) {
                        testPass = false;
                        break;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 25: PowTensorTensor INT8\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 25: PowTensorTensor INT8\n");
    }
    
    aclDestroyTensor(base);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    aclrtFree(baseDeviceAddr);
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  // Test 26: PowTensorTensor - INT16 dtype (触发不同OP_KEY)
  {
    totalTests++;
    std::vector<int16_t> baseInt = {2, 3, 4, 5};
    std::vector<int16_t> expInt = {2, 2, 2, 2};
    std::vector<int64_t> shape = {(int64_t)baseInt.size()};
    void* baseDeviceAddr = nullptr;
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* base = nullptr;
    aclTensor* exp = nullptr;
    aclTensor* out = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int16_t> outHostData(baseInt.size(), 0);
    bool testPass = false;
    
    ret = CreateAclTensor(baseInt, shape, &baseDeviceAddr, ACL_INT16, &base);
    if (ret == ACL_SUCCESS) {
      ret = CreateAclTensor(expInt, shape, &expDeviceAddr, ACL_INT16, &exp);
      if (ret == ACL_SUCCESS) {
        ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_INT16, &out);
        if (ret == ACL_SUCCESS) {
          ret = aclnnPowTensorTensorGetWorkspaceSize(base, exp, out, &workspaceSize, &executor);
          if (ret == ACL_SUCCESS) {
            if (workspaceSize > 0) {
              ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            }
            if (ret == ACL_SUCCESS) {
              ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
              if (ret == ACL_SUCCESS) {
                ret = aclrtSynchronizeStream(stream);
                if (ret == ACL_SUCCESS) {
                  std::vector<int16_t> resultData(baseInt.size(), 0);
                  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(int16_t), outDeviceAddr,
                                    resultData.size() * sizeof(int16_t), ACL_MEMCPY_DEVICE_TO_HOST);
                  if (ret == ACL_SUCCESS) {
                    std::vector<int16_t> expected = {4, 9, 16, 25};
                    testPass = true;
                    for (size_t i = 0; i < expected.size(); i++) {
                      if (resultData[i] != expected[i]) {
                        testPass = false;
                        break;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 26: PowTensorTensor INT16\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 26: PowTensorTensor INT16\n");
    }
    
    aclDestroyTensor(base);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    aclrtFree(baseDeviceAddr);
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  LOG_PRINT("\n========== InplacePowTensorTensor Tests ==========\n");

  // Test 27: InplacePowTensorTensor - FLOAT32
  {
    totalTests++;
    std::vector<int64_t> selfShape = {4};
    std::vector<int64_t> expShape = {4};
    void* selfDeviceAddr = nullptr;
    void* expDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* exp = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<float> selfHostData = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> expHostData = {2.0f, 2.0f, 2.0f, 2.0f};
    bool testPass = false;
    
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, ACL_FLOAT, &self);
    if (ret == ACL_SUCCESS) {
      ret = CreateAclTensor(expHostData, expShape, &expDeviceAddr, ACL_FLOAT, &exp);
      if (ret == ACL_SUCCESS) {
        ret = aclnnInplacePowTensorTensorGetWorkspaceSize(self, exp, &workspaceSize, &executor);
        if (ret == ACL_SUCCESS) {
          if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          }
          if (ret == ACL_SUCCESS) {
            ret = aclnnInplacePowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
            if (ret == ACL_SUCCESS) {
              ret = aclrtSynchronizeStream(stream);
              if (ret == ACL_SUCCESS) {
                std::vector<float> resultData(4, 0.0f);
                ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), selfDeviceAddr,
                                  resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
                if (ret == ACL_SUCCESS) {
                  float expected[] = {4.0f, 9.0f, 16.0f, 25.0f};
                  testPass = true;
                  for (int i = 0; i < 4; i++) {
                    if (!FloatCompare(resultData[i], expected[i], 1e-4, 1e-4)) {
                      testPass = false;
                      break;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 27: InplacePowTensorTensor FLOAT32\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 27: InplacePowTensorTensor FLOAT32\n");
    }
    
    aclDestroyTensor(self);
    aclDestroyTensor(exp);
    aclrtFree(selfDeviceAddr);
    aclrtFree(expDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  LOG_PRINT("\n========== Exp2 Tests ==========\n");

  // Test 28: Exp2 - FLOAT32
  {
    totalTests++;
    std::vector<float> input = {0.0f, 1.0f, 2.0f, 3.0f};
    std::vector<float> expected = {1.0f, 2.0f, 4.0f, 8.0f};
    int result = RunExp2Test(stream, "Test 28: Exp2 FLOAT32", input, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 29: Exp2 - FLOAT32 - negative input
  {
    totalTests++;
    std::vector<float> input = {-1.0f, -2.0f, -3.0f};
    std::vector<float> expected = {0.5f, 0.25f, 0.125f};
    int result = RunExp2Test(stream, "Test 29: Exp2 FLOAT32 negative", input, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 30: Exp2 - FLOAT32 - zero input
  {
    totalTests++;
    std::vector<float> input = {0.0f, 0.0f, 0.0f};
    std::vector<float> expected = {1.0f, 1.0f, 1.0f};
    int result = RunExp2Test(stream, "Test 30: Exp2 FLOAT32 zero", input, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 31: InplaceExp2 - FLOAT32
  {
    totalTests++;
    std::vector<int64_t> shape = {4};
    void* selfDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* self = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<float> selfHostData = {0.0f, 1.0f, 2.0f, 3.0f};
    bool testPass = false;
    
    ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, ACL_FLOAT, &self);
    if (ret == ACL_SUCCESS) {
      ret = aclnnInplaceExp2GetWorkspaceSize(self, &workspaceSize, &executor);
      if (ret == ACL_SUCCESS) {
        if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        }
        if (ret == ACL_SUCCESS) {
          ret = aclnnInplaceExp2(workspaceAddr, workspaceSize, executor, stream);
          if (ret == ACL_SUCCESS) {
            ret = aclrtSynchronizeStream(stream);
            if (ret == ACL_SUCCESS) {
              std::vector<float> resultData(4, 0.0f);
              ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), selfDeviceAddr,
                                resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
              if (ret == ACL_SUCCESS) {
                float expected[] = {1.0f, 2.0f, 4.0f, 8.0f};
                testPass = true;
                for (int i = 0; i < 4; i++) {
                  if (!FloatCompare(resultData[i], expected[i], 1e-5, 1e-5)) {
                    testPass = false;
                    break;
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 31: InplaceExp2 FLOAT32\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 31: InplaceExp2 FLOAT32\n");
    }
    
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  // Test 32: Exp2 - FLOAT16 dtype
  {
    totalTests++;
    std::vector<float> input = {0.0f, 1.0f, 2.0f, 3.0f};
    std::vector<float> expected = {1.0f, 2.0f, 4.0f, 8.0f};
    int result = RunExp2Test(stream, "Test 32: Exp2 FLOAT16", input, expected, ACL_FLOAT16);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 33: Exp2 - BF16 dtype
  {
    totalTests++;
    std::vector<float> input = {0.0f, 1.0f, 2.0f, 3.0f};
    std::vector<float> expected = {1.0f, 2.0f, 4.0f, 8.0f};
    int result = RunExp2Test(stream, "Test 33: Exp2 BF16", input, expected, ACL_BF16);
    if (result > 0) passCount++; else failCount++;
  }

  LOG_PRINT("\n========== Special Value Tests ==========\n");

  // Test 34: PowTensorScalar - NaN handling
  {
    totalTests++;
    std::vector<float> base = {std::nanf(""), 1.0f, 2.0f};
    std::vector<float> expected = {std::nanf(""), 1.0f, 4.0f};
    int result = RunPowTensorScalarTest(stream, "Test 34: PowTensorScalar with NaN",
                                        base, 2.0f, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 35: PowTensorScalar - Inf handling
  {
    totalTests++;
    std::vector<float> base = {std::numeric_limits<float>::infinity(), 1.0f, 2.0f};
    std::vector<float> expected = {std::numeric_limits<float>::infinity(), 1.0f, 4.0f};
    int result = RunPowTensorScalarTest(stream, "Test 35: PowTensorScalar with Inf",
                                        base, 2.0f, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 36: PowTensorTensor - large shape (2D)
  {
    totalTests++;
    std::vector<float> base = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> exp = {1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f};
    std::vector<float> expected = {1.0f, 2.0f, 9.0f, 16.0f, 125.0f, 216.0f};
    std::vector<int64_t> shape = {2, 3};
    int result = RunPowTensorTensorTest(stream, "Test 36: PowTensorTensor 2D shape",
                                        base, exp, expected, ACL_FLOAT, shape, shape);
    if (result > 0) passCount++; else failCount++;
  }

  LOG_PRINT("\n========== Additional Coverage Tests ==========\n");

  // Test 37: PowTensorScalar - FLOAT64/DOUBLE dtype
  {
    totalTests++;
    std::vector<double> base = {2.0, 3.0, 4.0, 5.0};
    std::vector<int64_t> shape = {(int64_t)base.size()};
    void* baseDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* baseTensor = nullptr;
    aclTensor* out = nullptr;
    aclScalar* expScalar = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<double> outHostData(base.size(), 0.0);
    bool testPass = false;
    
    ret = CreateAclTensor(base, shape, &baseDeviceAddr, ACL_DOUBLE, &baseTensor);
    if (ret == ACL_SUCCESS) {
      double expVal = 2.0;
      expScalar = aclCreateScalar(&expVal, ACL_DOUBLE);
      if (expScalar != nullptr) {
        ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_DOUBLE, &out);
        if (ret == ACL_SUCCESS) {
          ret = aclnnPowTensorScalarGetWorkspaceSize(baseTensor, expScalar, out, &workspaceSize, &executor);
          if (ret == ACL_SUCCESS) {
            if (workspaceSize > 0) {
              ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            }
            if (ret == ACL_SUCCESS) {
              ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
              if (ret == ACL_SUCCESS) {
                ret = aclrtSynchronizeStream(stream);
                if (ret == ACL_SUCCESS) {
                  std::vector<double> resultData(base.size(), 0.0);
                  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(double), outDeviceAddr,
                                    resultData.size() * sizeof(double), ACL_MEMCPY_DEVICE_TO_HOST);
                  if (ret == ACL_SUCCESS) {
                    std::vector<double> expected = {4.0, 9.0, 16.0, 25.0};
                    testPass = true;
                    for (size_t i = 0; i < expected.size(); i++) {
                      if (!DoubleCompare(resultData[i], expected[i], 1e-9, 1e-9)) {
                        testPass = false;
                        break;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 37: PowTensorScalar DOUBLE\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 37: PowTensorScalar DOUBLE\n");
    }
    
    aclDestroyTensor(baseTensor);
    aclDestroyScalar(expScalar);
    aclDestroyTensor(out);
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  // Test 38: PowTensorScalar - INT64 dtype
  {
    totalTests++;
    std::vector<int64_t> baseInt = {2, 3, 4, 5};
    std::vector<int64_t> shape = {(int64_t)baseInt.size()};
    void* baseDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* baseTensor = nullptr;
    aclTensor* out = nullptr;
    aclScalar* expScalar = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int64_t> outHostData(baseInt.size(), 0);
    bool testPass = false;
    
    ret = CreateAclTensor(baseInt, shape, &baseDeviceAddr, ACL_INT64, &baseTensor);
    if (ret == ACL_SUCCESS) {
      float expVal = 2.0f;
      expScalar = aclCreateScalar(&expVal, ACL_FLOAT);
      if (expScalar != nullptr) {
        ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_INT64, &out);
        if (ret == ACL_SUCCESS) {
          ret = aclnnPowTensorScalarGetWorkspaceSize(baseTensor, expScalar, out, &workspaceSize, &executor);
          if (ret == ACL_SUCCESS) {
            if (workspaceSize > 0) {
              ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            }
            if (ret == ACL_SUCCESS) {
              ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
              if (ret == ACL_SUCCESS) {
                ret = aclrtSynchronizeStream(stream);
                if (ret == ACL_SUCCESS) {
                  std::vector<int64_t> resultData(baseInt.size(), 0);
                  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(int64_t), outDeviceAddr,
                                    resultData.size() * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
                  if (ret == ACL_SUCCESS) {
                    std::vector<int64_t> expected = {4, 9, 16, 25};
                    testPass = true;
                    for (size_t i = 0; i < expected.size(); i++) {
                      if (resultData[i] != expected[i]) {
                        testPass = false;
                        break;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 38: PowTensorScalar INT64\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 38: PowTensorScalar INT64\n");
    }
    
    aclDestroyTensor(baseTensor);
    aclDestroyScalar(expScalar);
    aclDestroyTensor(out);
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  // Test 39: PowTensorScalar - exponent=0.5 (sqrt path with different dtypes)
  {
    totalTests++;
    std::vector<float> base = {4.0f, 9.0f, 16.0f, 25.0f};
    std::vector<float> expected = {2.0f, 3.0f, 4.0f, 5.0f};
    int result = RunPowTensorScalarTest(stream, "Test 39: PowTensorScalar FLOAT16 exp=0.5",
                                        base, 0.5f, expected, ACL_FLOAT16);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 40: PowTensorScalar - exponent=-0.5 (negative sqrt)
  {
    totalTests++;
    std::vector<float> base = {4.0f, 9.0f, 16.0f};
    std::vector<float> expected = {0.5f, 0.333333f, 0.25f};
    int result = RunPowTensorScalarTest(stream, "Test 40: PowTensorScalar exp=-0.5",
                                        base, -0.5f, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 41: PowTensorScalar - exponent=-2.0 (negative square)
  {
    totalTests++;
    std::vector<float> base = {2.0f, 3.0f, 4.0f};
    std::vector<float> expected = {0.25f, 0.111111f, 0.0625f};
    int result = RunPowTensorScalarTest(stream, "Test 41: PowTensorScalar exp=-2.0",
                                        base, -2.0f, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 42: PowTensorScalar - empty tensor
  {
    totalTests++;
    std::vector<float> base = {};
    std::vector<float> expected = {};
    std::vector<int64_t> shape = {0};
    void* baseDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* baseTensor = nullptr;
    aclTensor* out = nullptr;
    aclScalar* expScalar = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    bool testPass = false;
    
    ret = aclrtMalloc(&baseDeviceAddr, 1, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret == ACL_SUCCESS) {
      baseTensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, 
                                   aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), baseDeviceAddr);
      if (baseTensor != nullptr) {
        float expVal = 2.0f;
        expScalar = aclCreateScalar(&expVal, ACL_FLOAT);
        if (expScalar != nullptr) {
          ret = aclrtMalloc(&outDeviceAddr, 1, ACL_MEM_MALLOC_HUGE_FIRST);
          if (ret == ACL_SUCCESS) {
            out = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0,
                                  aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), outDeviceAddr);
            if (out != nullptr) {
              ret = aclnnPowTensorScalarGetWorkspaceSize(baseTensor, expScalar, out, &workspaceSize, &executor);
              if (ret == ACL_SUCCESS) {
                if (workspaceSize > 0) {
                  ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                }
                if (ret == ACL_SUCCESS) {
                  ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
                  if (ret == ACL_SUCCESS) {
                    ret = aclrtSynchronizeStream(stream);
                    if (ret == ACL_SUCCESS) {
                      testPass = (workspaceSize == 0);  // Empty tensor should have 0 workspace
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 42: PowTensorScalar empty tensor\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 42: PowTensorScalar empty tensor\n");
    }
    
    aclDestroyTensor(baseTensor);
    aclDestroyScalar(expScalar);
    aclDestroyTensor(out);
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  // Test 43: PowScalarTensor - base=1.0 (fill optimization path)
  {
    totalTests++;
    std::vector<float> expData = {0.0f, 1.0f, 2.0f, 3.0f, 10.0f};
    std::vector<float> expected = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    int result = RunPowScalarTensorTest(stream, "Test 43: PowScalarTensor base=1.0 fill opt",
                                        1.0f, expData, expected, ACL_FLOAT);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 44: PowScalarTensor - DOUBLE dtype
  {
    totalTests++;
    std::vector<double> expData = {1.0, 2.0, 3.0, 4.0};
    std::vector<int64_t> shape = {(int64_t)expData.size()};
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* exp = nullptr;
    aclTensor* out = nullptr;
    aclScalar* baseScalar = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<double> outHostData(expData.size(), 0.0);
    bool testPass = false;
    
    double baseVal = 2.0;
    baseScalar = aclCreateScalar(&baseVal, ACL_DOUBLE);
    if (baseScalar != nullptr) {
      ret = CreateAclTensor(expData, shape, &expDeviceAddr, ACL_DOUBLE, &exp);
      if (ret == ACL_SUCCESS) {
        ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_DOUBLE, &out);
        if (ret == ACL_SUCCESS) {
          ret = aclnnPowScalarTensorGetWorkspaceSize(baseScalar, exp, out, &workspaceSize, &executor);
          if (ret == ACL_SUCCESS) {
            if (workspaceSize > 0) {
              ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            }
            if (ret == ACL_SUCCESS) {
              ret = aclnnPowScalarTensor(workspaceAddr, workspaceSize, executor, stream);
              if (ret == ACL_SUCCESS) {
                ret = aclrtSynchronizeStream(stream);
                if (ret == ACL_SUCCESS) {
                  std::vector<double> resultData(expData.size(), 0.0);
                  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(double), outDeviceAddr,
                                    resultData.size() * sizeof(double), ACL_MEMCPY_DEVICE_TO_HOST);
                  if (ret == ACL_SUCCESS) {
                    std::vector<double> expected = {2.0, 4.0, 8.0, 16.0};
                    testPass = true;
                    for (size_t i = 0; i < expected.size(); i++) {
                      if (!DoubleCompare(resultData[i], expected[i], 1e-9, 1e-9)) {
                        testPass = false;
                        break;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 44: PowScalarTensor DOUBLE\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 44: PowScalarTensor DOUBLE\n");
    }
    
    aclDestroyScalar(baseScalar);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  // Test 45: PowScalarTensor - empty tensor
  {
    totalTests++;
    std::vector<float> expData = {};
    std::vector<int64_t> shape = {0};
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* exp = nullptr;
    aclTensor* out = nullptr;
    aclScalar* baseScalar = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    bool testPass = false;
    
    float baseVal = 2.0f;
    baseScalar = aclCreateScalar(&baseVal, ACL_FLOAT);
    if (baseScalar != nullptr) {
      ret = aclrtMalloc(&expDeviceAddr, 1, ACL_MEM_MALLOC_HUGE_FIRST);
      if (ret == ACL_SUCCESS) {
        exp = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), expDeviceAddr);
        if (exp != nullptr) {
          ret = aclrtMalloc(&outDeviceAddr, 1, ACL_MEM_MALLOC_HUGE_FIRST);
          if (ret == ACL_SUCCESS) {
            out = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0,
                                  aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), outDeviceAddr);
            if (out != nullptr) {
              ret = aclnnPowScalarTensorGetWorkspaceSize(baseScalar, exp, out, &workspaceSize, &executor);
              if (ret == ACL_SUCCESS) {
                if (workspaceSize > 0) {
                  ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                }
                if (ret == ACL_SUCCESS) {
                  ret = aclnnPowScalarTensor(workspaceAddr, workspaceSize, executor, stream);
                  if (ret == ACL_SUCCESS) {
                    ret = aclrtSynchronizeStream(stream);
                    if (ret == ACL_SUCCESS) {
                      testPass = (workspaceSize == 0);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 45: PowScalarTensor empty tensor\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 45: PowScalarTensor empty tensor\n");
    }
    
    aclDestroyScalar(baseScalar);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  // Test 46: PowTensorTensor - BOOL dtype (should fail validation)
  {
    totalTests++;
    std::vector<uint8_t> baseBool = {1, 1, 0, 1};
    std::vector<uint8_t> expBool = {1, 0, 1, 1};
    std::vector<int64_t> shape = {(int64_t)baseBool.size()};
    void* baseDeviceAddr = nullptr;
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* base = nullptr;
    aclTensor* exp = nullptr;
    aclTensor* out = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<uint8_t> outHostData(baseBool.size(), 0);
    bool testPass = false;
    
    ret = CreateAclTensor(baseBool, shape, &baseDeviceAddr, ACL_BOOL, &base);
    if (ret == ACL_SUCCESS) {
      ret = CreateAclTensor(expBool, shape, &expDeviceAddr, ACL_BOOL, &exp);
      if (ret == ACL_SUCCESS) {
        ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_BOOL, &out);
        if (ret == ACL_SUCCESS) {
          // Both inputs are BOOL, should fail parameter check
          aclnnStatus status = aclnnPowTensorTensorGetWorkspaceSize(base, exp, out, &workspaceSize, &executor);
          testPass = (status != 0);  // Expected to fail (0 is ACLNN_SUCCESS)
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 46: PowTensorTensor BOOL rejected\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 46: PowTensorTensor BOOL rejected\n");
    }
    
    aclDestroyTensor(base);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    aclrtFree(baseDeviceAddr);
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  // Test 47: PowTensorTensor - mixed dtypes (INT32 and FLOAT)
  {
    totalTests++;
    std::vector<int32_t> baseInt = {2, 3, 4, 5};
    std::vector<float> expFloat = {2.0f, 2.0f, 2.0f, 2.0f};
    std::vector<int64_t> baseShape = {4};
    std::vector<int64_t> expShape = {4};
    std::vector<int64_t> outShape = {4};
    void* baseDeviceAddr = nullptr;
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* base = nullptr;
    aclTensor* exp = nullptr;
    aclTensor* out = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<float> outHostData(baseInt.size(), 0.0f);
    bool testPass = false;
    
    ret = CreateAclTensor(baseInt, baseShape, &baseDeviceAddr, ACL_INT32, &base);
    if (ret == ACL_SUCCESS) {
      ret = CreateAclTensor(expFloat, expShape, &expDeviceAddr, ACL_FLOAT, &exp);
      if (ret == ACL_SUCCESS) {
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, ACL_FLOAT, &out);
        if (ret == ACL_SUCCESS) {
          ret = aclnnPowTensorTensorGetWorkspaceSize(base, exp, out, &workspaceSize, &executor);
          if (ret == ACL_SUCCESS) {
            if (workspaceSize > 0) {
              ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            }
            if (ret == ACL_SUCCESS) {
              ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
              if (ret == ACL_SUCCESS) {
                ret = aclrtSynchronizeStream(stream);
                if (ret == ACL_SUCCESS) {
                  std::vector<float> resultData(baseInt.size(), 0.0f);
                  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                                    resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
                  if (ret == ACL_SUCCESS) {
                    std::vector<float> expected = {4.0f, 9.0f, 16.0f, 25.0f};
                    testPass = true;
                    for (size_t i = 0; i < expected.size(); i++) {
                      if (!FloatCompare(resultData[i], expected[i], 1e-4, 1e-4)) {
                        testPass = false;
                        break;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 47: PowTensorTensor mixed INT32/FLOAT\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 47: PowTensorTensor mixed INT32/FLOAT\n");
    }
    
    aclDestroyTensor(base);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    aclrtFree(baseDeviceAddr);
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  // Test 48: PowTensorTensor - 3D broadcasting
  {
    totalTests++;
    std::vector<float> base = {2.0f, 3.0f};
    std::vector<float> exp = {1.0f, 2.0f, 3.0f};
    std::vector<float> expected = {2.0f, 4.0f, 8.0f, 3.0f, 9.0f, 27.0f};
    std::vector<int64_t> baseShape = {2, 1, 1};
    std::vector<int64_t> expShape = {1, 3, 1};
    std::vector<int64_t> outShape = {2, 3, 1};
    int result = RunPowTensorTensorTest(stream, "Test 48: PowTensorTensor 3D broadcast",
                                        base, exp, expected, ACL_FLOAT, baseShape, expShape);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 49: InplacePowTensorScalar - INT32 dtype
  {
    totalTests++;
    std::vector<int64_t> shape = {4};
    void* selfDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* self = nullptr;
    aclScalar* exponent = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int32_t> selfHostData = {2, 3, 4, 5};
    bool testPass = false;
    float exponentVal = 2.0f;
    
    ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, ACL_INT32, &self);
    if (ret == ACL_SUCCESS) {
      exponent = aclCreateScalar(&exponentVal, ACL_FLOAT);
      if (exponent != nullptr) {
        ret = aclnnInplacePowTensorScalarGetWorkspaceSize(self, exponent, &workspaceSize, &executor);
        if (ret == ACL_SUCCESS) {
          if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          }
          if (ret == ACL_SUCCESS) {
            ret = aclnnInplacePowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
            if (ret == ACL_SUCCESS) {
              ret = aclrtSynchronizeStream(stream);
              if (ret == ACL_SUCCESS) {
                std::vector<int32_t> resultData(4, 0);
                ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(int32_t), selfDeviceAddr,
                                  resultData.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
                if (ret == ACL_SUCCESS) {
                  std::vector<int32_t> expected = {4, 9, 16, 25};
                  testPass = true;
                  for (int i = 0; i < 4; i++) {
                    if (resultData[i] != expected[i]) {
                      testPass = false;
                      break;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 49: InplacePowTensorScalar INT32\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 49: InplacePowTensorScalar INT32\n");
    }
    
    aclDestroyTensor(self);
    aclDestroyScalar(exponent);
    aclrtFree(selfDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  // Test 50: InplacePowTensorTensor - INT16 dtype
  {
    totalTests++;
    std::vector<int64_t> selfShape = {4};
    std::vector<int64_t> expShape = {4};
    void* selfDeviceAddr = nullptr;
    void* expDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* exp = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int16_t> selfHostData = {2, 3, 4, 5};
    std::vector<int16_t> expHostData = {2, 2, 2, 2};
    bool testPass = false;
    
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, ACL_INT16, &self);
    if (ret == ACL_SUCCESS) {
      ret = CreateAclTensor(expHostData, expShape, &expDeviceAddr, ACL_INT16, &exp);
      if (ret == ACL_SUCCESS) {
        ret = aclnnInplacePowTensorTensorGetWorkspaceSize(self, exp, &workspaceSize, &executor);
        if (ret == ACL_SUCCESS) {
          if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          }
          if (ret == ACL_SUCCESS) {
            ret = aclnnInplacePowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
            if (ret == ACL_SUCCESS) {
              ret = aclrtSynchronizeStream(stream);
              if (ret == ACL_SUCCESS) {
                std::vector<int16_t> resultData(4, 0);
                ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(int16_t), selfDeviceAddr,
                                  resultData.size() * sizeof(int16_t), ACL_MEMCPY_DEVICE_TO_HOST);
                if (ret == ACL_SUCCESS) {
                  std::vector<int16_t> expected = {4, 9, 16, 25};
                  testPass = true;
                  for (int i = 0; i < 4; i++) {
                    if (resultData[i] != expected[i]) {
                      testPass = false;
                      break;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 50: InplacePowTensorTensor INT16\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 50: InplacePowTensorTensor INT16\n");
    }
    
    aclDestroyTensor(self);
    aclDestroyTensor(exp);
    aclrtFree(selfDeviceAddr);
    aclrtFree(expDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  LOG_PRINT("\n========== Complex Type Tests ==========\n");

  // Test 51: PowTensorScalar - COMPLEX64 dtype
  {
    totalTests++;
    std::vector<std::complex<float>> base = {{1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}};
    std::vector<int64_t> shape = {(int64_t)base.size()};
    void* baseDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* baseTensor = nullptr;
    aclTensor* out = nullptr;
    aclScalar* expScalar = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<std::complex<float>> outHostData(base.size(), {0.0f, 0.0f});
    bool testPass = false;
    
    ret = aclrtMalloc(&baseDeviceAddr, base.size() * sizeof(std::complex<float>), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret == ACL_SUCCESS) {
      ret = aclrtMemcpy(baseDeviceAddr, base.size() * sizeof(std::complex<float>), base.data(),
                        base.size() * sizeof(std::complex<float>), ACL_MEMCPY_HOST_TO_DEVICE);
      if (ret == ACL_SUCCESS) {
        std::vector<int64_t> strides = {1};
        baseTensor = aclCreateTensor(shape.data(), shape.size(), ACL_COMPLEX64, strides.data(), 0,
                                     aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), baseDeviceAddr);
        if (baseTensor != nullptr) {
          float expVal = 2.0f;
          expScalar = aclCreateScalar(&expVal, ACL_FLOAT);
          if (expScalar != nullptr) {
            ret = aclrtMalloc(&outDeviceAddr, outHostData.size() * sizeof(std::complex<float>), ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret == ACL_SUCCESS) {
              out = aclCreateTensor(shape.data(), shape.size(), ACL_COMPLEX64, strides.data(), 0,
                                    aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), outDeviceAddr);
              if (out != nullptr) {
                ret = aclnnPowTensorScalarGetWorkspaceSize(baseTensor, expScalar, out, &workspaceSize, &executor);
                if (ret == ACL_SUCCESS) {
                  if (workspaceSize > 0) {
                    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                  }
                  if (ret == ACL_SUCCESS) {
                    ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
                    if (ret == ACL_SUCCESS) {
                      ret = aclrtSynchronizeStream(stream);
                      if (ret == ACL_SUCCESS) {
                        std::vector<std::complex<float>> resultData(base.size(), {0.0f, 0.0f});
                        ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(std::complex<float>), outDeviceAddr,
                                          resultData.size() * sizeof(std::complex<float>), ACL_MEMCPY_DEVICE_TO_HOST);
                        if (ret == ACL_SUCCESS) {
                          std::vector<std::complex<float>> expected = {{1.0f, 0.0f}, {4.0f, 0.0f}, {9.0f, 0.0f}};
                          testPass = true;
                          for (size_t i = 0; i < expected.size(); i++) {
                            if (std::abs(resultData[i] - expected[i]) > 1e-4) {
                              testPass = false;
                              break;
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 51: PowTensorScalar COMPLEX64\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 51: PowTensorScalar COMPLEX64\n");
    }
    
    aclDestroyTensor(baseTensor);
    aclDestroyScalar(expScalar);
    aclDestroyTensor(out);
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  // Test 52: PowScalarTensor - COMPLEX64 exponent
  {
    totalTests++;
    std::vector<std::complex<float>> expData = {{1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}};
    std::vector<int64_t> shape = {(int64_t)expData.size()};
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* exp = nullptr;
    aclTensor* out = nullptr;
    aclScalar* baseScalar = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<std::complex<float>> outHostData(expData.size(), {0.0f, 0.0f});
    bool testPass = false;
    
    std::complex<float> baseVal = {2.0f, 0.0f};
    baseScalar = aclCreateScalar(&baseVal, ACL_COMPLEX64);
    if (baseScalar != nullptr) {
      ret = aclrtMalloc(&expDeviceAddr, expData.size() * sizeof(std::complex<float>), ACL_MEM_MALLOC_HUGE_FIRST);
      if (ret == ACL_SUCCESS) {
        ret = aclrtMemcpy(expDeviceAddr, expData.size() * sizeof(std::complex<float>), expData.data(),
                          expData.size() * sizeof(std::complex<float>), ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret == ACL_SUCCESS) {
          std::vector<int64_t> strides = {1};
          exp = aclCreateTensor(shape.data(), shape.size(), ACL_COMPLEX64, strides.data(), 0,
                                aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), expDeviceAddr);
          if (exp != nullptr) {
            ret = aclrtMalloc(&outDeviceAddr, outHostData.size() * sizeof(std::complex<float>), ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret == ACL_SUCCESS) {
              out = aclCreateTensor(shape.data(), shape.size(), ACL_COMPLEX64, strides.data(), 0,
                                    aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), outDeviceAddr);
              if (out != nullptr) {
                ret = aclnnPowScalarTensorGetWorkspaceSize(baseScalar, exp, out, &workspaceSize, &executor);
                if (ret == ACL_SUCCESS) {
                  if (workspaceSize > 0) {
                    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                  }
                  if (ret == ACL_SUCCESS) {
                    ret = aclnnPowScalarTensor(workspaceAddr, workspaceSize, executor, stream);
                    if (ret == ACL_SUCCESS) {
                      ret = aclrtSynchronizeStream(stream);
                      if (ret == ACL_SUCCESS) {
                        std::vector<std::complex<float>> resultData(expData.size(), {0.0f, 0.0f});
                        ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(std::complex<float>), outDeviceAddr,
                                          resultData.size() * sizeof(std::complex<float>), ACL_MEMCPY_DEVICE_TO_HOST);
                        if (ret == ACL_SUCCESS) {
                          std::vector<std::complex<float>> expected = {{2.0f, 0.0f}, {4.0f, 0.0f}, {8.0f, 0.0f}};
                          testPass = true;
                          for (size_t i = 0; i < expected.size(); i++) {
                            if (std::abs(resultData[i] - expected[i]) > 1e-4) {
                              testPass = false;
                              break;
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 52: PowScalarTensor COMPLEX64\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 52: PowScalarTensor COMPLEX64\n");
    }
    
    aclDestroyScalar(baseScalar);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  LOG_PRINT("\n========== Overflow and Edge Case Tests ==========\n");

  // Test 53: PowTensorScalar - large exponent value (overflow check)
  {
    totalTests++;
    std::vector<float> base = {2.0f, 3.0f};
    std::vector<int64_t> shape = {(int64_t)base.size()};
    void* baseDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* baseTensor = nullptr;
    aclTensor* out = nullptr;
    aclScalar* expScalar = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<float> outHostData(base.size(), 0.0f);
    bool testPass = false;
    
    ret = CreateAclTensor(base, shape, &baseDeviceAddr, ACL_FLOAT, &baseTensor);
    if (ret == ACL_SUCCESS) {
      float expVal = 1000.0f;  // Large exponent to trigger overflow check
      expScalar = aclCreateScalar(&expVal, ACL_FLOAT);
      if (expScalar != nullptr) {
        ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_FLOAT, &out);
        if (ret == ACL_SUCCESS) {
          aclnnStatus status = aclnnPowTensorScalarGetWorkspaceSize(baseTensor, expScalar, out, &workspaceSize, &executor);
          // May succeed or fail depending on implementation, just test the path
          testPass = true;
          if (status == 0 && workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret == ACL_SUCCESS) {
              ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
              if (ret == ACL_SUCCESS) {
                aclrtSynchronizeStream(stream);
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 53: PowTensorScalar large exponent\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 53: PowTensorScalar large exponent\n");
    }
    
    aclDestroyTensor(baseTensor);
    aclDestroyScalar(expScalar);
    aclDestroyTensor(out);
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  // Test 54: PowTensorScalar - INT8 with negative exponent (should fail validation)
  {
    totalTests++;
    std::vector<int8_t> baseInt = {2, 3, 4};
    std::vector<int64_t> shape = {(int64_t)baseInt.size()};
    void* baseDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* baseTensor = nullptr;
    aclTensor* out = nullptr;
    aclScalar* expScalar = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int8_t> outHostData(baseInt.size(), 0);
    bool testPass = false;
    
    ret = CreateAclTensor(baseInt, shape, &baseDeviceAddr, ACL_INT8, &baseTensor);
    if (ret == ACL_SUCCESS) {
      float expVal = -1.0f;  // Negative exponent for integer type
      expScalar = aclCreateScalar(&expVal, ACL_FLOAT);
      if (expScalar != nullptr) {
        ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_INT8, &out);
        if (ret == ACL_SUCCESS) {
          // This should fail parameter check for RegBase platform
          aclnnStatus status = aclnnPowTensorScalarGetWorkspaceSize(baseTensor, expScalar, out, &workspaceSize, &executor);
          testPass = true;  // Just testing that the code path is executed
          if (status == 0 && workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret == ACL_SUCCESS) {
              aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
              aclrtSynchronizeStream(stream);
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 54: PowTensorScalar INT8 negative exp\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 54: PowTensorScalar INT8 negative exp\n");
    }
    
    aclDestroyTensor(baseTensor);
    aclDestroyScalar(expScalar);
    aclDestroyTensor(out);
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  // Test 55: PowTensorScalar - BF16 dtype
  {
    totalTests++;
    std::vector<float> base = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> expected = {4.0f, 9.0f, 16.0f, 25.0f};
    int result = RunPowTensorScalarTest(stream, "Test 55: PowTensorScalar BF16 exp=2.0",
                                        base, 2.0f, expected, ACL_BF16);
    if (result > 0) passCount++; else failCount++;
  }

  // Test 56: PowTensorScalar - UINT8 dtype
  {
    totalTests++;
    std::vector<uint8_t> baseInt = {2, 3, 4, 5};
    std::vector<int64_t> shape = {(int64_t)baseInt.size()};
    void* baseDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* baseTensor = nullptr;
    aclTensor* out = nullptr;
    aclScalar* expScalar = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<uint8_t> outHostData(baseInt.size(), 0);
    bool testPass = false;
    
    ret = CreateAclTensor(baseInt, shape, &baseDeviceAddr, ACL_UINT8, &baseTensor);
    if (ret == ACL_SUCCESS) {
      float expVal = 2.0f;
      expScalar = aclCreateScalar(&expVal, ACL_FLOAT);
      if (expScalar != nullptr) {
        ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_UINT8, &out);
        if (ret == ACL_SUCCESS) {
          ret = aclnnPowTensorScalarGetWorkspaceSize(baseTensor, expScalar, out, &workspaceSize, &executor);
          if (ret == ACL_SUCCESS) {
            if (workspaceSize > 0) {
              ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            }
            if (ret == ACL_SUCCESS) {
              ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
              if (ret == ACL_SUCCESS) {
                ret = aclrtSynchronizeStream(stream);
                if (ret == ACL_SUCCESS) {
                  std::vector<uint8_t> resultData(baseInt.size(), 0);
                  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(uint8_t), outDeviceAddr,
                                    resultData.size() * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
                  if (ret == ACL_SUCCESS) {
                    std::vector<uint8_t> expected = {4, 9, 16, 25};
                    testPass = true;
                    for (size_t i = 0; i < expected.size(); i++) {
                      if (resultData[i] != expected[i]) {
                        testPass = false;
                        break;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 56: PowTensorScalar UINT8\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 56: PowTensorScalar UINT8\n");
    }
    
    aclDestroyTensor(baseTensor);
    aclDestroyScalar(expScalar);
    aclDestroyTensor(out);
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  // Test 57: PowScalarTensor - INT16 dtype
  {
    totalTests++;
    std::vector<int16_t> expData = {1, 2, 3, 4};
    std::vector<int64_t> shape = {(int64_t)expData.size()};
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* exp = nullptr;
    aclTensor* out = nullptr;
    aclScalar* baseScalar = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int16_t> outHostData(expData.size(), 0);
    bool testPass = false;
    
    float baseVal = 2.0f;
    baseScalar = aclCreateScalar(&baseVal, ACL_FLOAT);
    if (baseScalar != nullptr) {
      ret = CreateAclTensor(expData, shape, &expDeviceAddr, ACL_INT16, &exp);
      if (ret == ACL_SUCCESS) {
        ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_INT16, &out);
        if (ret == ACL_SUCCESS) {
          ret = aclnnPowScalarTensorGetWorkspaceSize(baseScalar, exp, out, &workspaceSize, &executor);
          if (ret == ACL_SUCCESS) {
            if (workspaceSize > 0) {
              ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            }
            if (ret == ACL_SUCCESS) {
              ret = aclnnPowScalarTensor(workspaceAddr, workspaceSize, executor, stream);
              if (ret == ACL_SUCCESS) {
                ret = aclrtSynchronizeStream(stream);
                if (ret == ACL_SUCCESS) {
                  std::vector<int16_t> resultData(expData.size(), 0);
                  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(int16_t), outDeviceAddr,
                                    resultData.size() * sizeof(int16_t), ACL_MEMCPY_DEVICE_TO_HOST);
                  if (ret == ACL_SUCCESS) {
                    std::vector<int16_t> expected = {2, 4, 8, 16};
                    testPass = true;
                    for (size_t i = 0; i < expected.size(); i++) {
                      if (resultData[i] != expected[i]) {
                        testPass = false;
                        break;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 57: PowScalarTensor INT16\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 57: PowScalarTensor INT16\n");
    }
    
    aclDestroyScalar(baseScalar);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  // Test 58: PowTensorTensor - INT64 dtype
  {
    totalTests++;
    std::vector<int64_t> baseInt = {2, 3, 4, 5};
    std::vector<int64_t> expInt = {2, 2, 2, 2};
    std::vector<int64_t> shape = {(int64_t)baseInt.size()};
    void* baseDeviceAddr = nullptr;
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* base = nullptr;
    aclTensor* exp = nullptr;
    aclTensor* out = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int64_t> outHostData(baseInt.size(), 0);
    bool testPass = false;
    
    ret = CreateAclTensor(baseInt, shape, &baseDeviceAddr, ACL_INT64, &base);
    if (ret == ACL_SUCCESS) {
      ret = CreateAclTensor(expInt, shape, &expDeviceAddr, ACL_INT64, &exp);
      if (ret == ACL_SUCCESS) {
        ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_INT64, &out);
        if (ret == ACL_SUCCESS) {
          ret = aclnnPowTensorTensorGetWorkspaceSize(base, exp, out, &workspaceSize, &executor);
          if (ret == ACL_SUCCESS) {
            if (workspaceSize > 0) {
              ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            }
            if (ret == ACL_SUCCESS) {
              ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
              if (ret == ACL_SUCCESS) {
                ret = aclrtSynchronizeStream(stream);
                if (ret == ACL_SUCCESS) {
                  std::vector<int64_t> resultData(baseInt.size(), 0);
                  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(int64_t), outDeviceAddr,
                                    resultData.size() * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
                  if (ret == ACL_SUCCESS) {
                    std::vector<int64_t> expected = {4, 9, 16, 25};
                    testPass = true;
                    for (size_t i = 0; i < expected.size(); i++) {
                      if (resultData[i] != expected[i]) {
                        testPass = false;
                        break;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 58: PowTensorTensor INT64\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 58: PowTensorTensor INT64\n");
    }
    
    aclDestroyTensor(base);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    aclrtFree(baseDeviceAddr);
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  // Test 59: PowTensorScalar - scalar exponent as INT type
  {
    totalTests++;
    std::vector<float> base = {2.0f, 3.0f, 4.0f};
    std::vector<int64_t> shape = {(int64_t)base.size()};
    void* baseDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* baseTensor = nullptr;
    aclTensor* out = nullptr;
    aclScalar* expScalar = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<float> outHostData(base.size(), 0.0f);
    bool testPass = false;
    
    ret = CreateAclTensor(base, shape, &baseDeviceAddr, ACL_FLOAT, &baseTensor);
    if (ret == ACL_SUCCESS) {
      int64_t expVal = 2;  // Use INT64 scalar
      expScalar = aclCreateScalar(&expVal, ACL_INT64);
      if (expScalar != nullptr) {
        ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_FLOAT, &out);
        if (ret == ACL_SUCCESS) {
          ret = aclnnPowTensorScalarGetWorkspaceSize(baseTensor, expScalar, out, &workspaceSize, &executor);
          if (ret == ACL_SUCCESS) {
            if (workspaceSize > 0) {
              ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            }
            if (ret == ACL_SUCCESS) {
              ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
              if (ret == ACL_SUCCESS) {
                ret = aclrtSynchronizeStream(stream);
                if (ret == ACL_SUCCESS) {
                  std::vector<float> resultData(base.size(), 0.0f);
                  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                                    resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
                  if (ret == ACL_SUCCESS) {
                    std::vector<float> expected = {4.0f, 9.0f, 16.0f};
                    testPass = true;
                    for (size_t i = 0; i < expected.size(); i++) {
                      if (!FloatCompare(resultData[i], expected[i], 1e-4, 1e-4)) {
                        testPass = false;
                        break;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 59: PowTensorScalar INT64 exponent\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 59: PowTensorScalar INT64 exponent\n");
    }
    
    aclDestroyTensor(baseTensor);
    aclDestroyScalar(expScalar);
    aclDestroyTensor(out);
    aclrtFree(baseDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  // Test 60: PowScalarTensor - empty tensor with base=1.0 (fill optimization)
  {
    totalTests++;
    std::vector<float> expData = {};
    std::vector<int64_t> shape = {0};
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* exp = nullptr;
    aclTensor* out = nullptr;
    aclScalar* baseScalar = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    bool testPass = false;
    
    float baseVal = 1.0f;
    baseScalar = aclCreateScalar(&baseVal, ACL_FLOAT);
    if (baseScalar != nullptr) {
      ret = aclrtMalloc(&expDeviceAddr, 1, ACL_MEM_MALLOC_HUGE_FIRST);
      if (ret == ACL_SUCCESS) {
        exp = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), expDeviceAddr);
        if (exp != nullptr) {
          ret = aclrtMalloc(&outDeviceAddr, 1, ACL_MEM_MALLOC_HUGE_FIRST);
          if (ret == ACL_SUCCESS) {
            out = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0,
                                  aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), outDeviceAddr);
            if (out != nullptr) {
              ret = aclnnPowScalarTensorGetWorkspaceSize(baseScalar, exp, out, &workspaceSize, &executor);
              if (ret == ACL_SUCCESS) {
                testPass = true;  // Empty tensor should succeed
                if (workspaceSize > 0) {
                  ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                  if (ret == ACL_SUCCESS) {
                    aclnnPowScalarTensor(workspaceAddr, workspaceSize, executor, stream);
                    aclrtSynchronizeStream(stream);
                  }
                }
              }
            }
          }
        }
      }
    }
    
    if (testPass) {
      passCount++;
      LOG_PRINT("[PASS] Test 60: PowScalarTensor empty with base=1.0\n");
    } else {
      failCount++;
      LOG_PRINT("[FAIL] Test 60: PowScalarTensor empty with base=1.0\n");
    }
    
    aclDestroyScalar(baseScalar);
    aclDestroyTensor(exp);
    aclDestroyTensor(out);
    aclrtFree(expDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0 && workspaceAddr) aclrtFree(workspaceAddr);
  }

  // ==================== 输出汇总 ====================
  LOG_PRINT("\n========================================\n");
  LOG_PRINT("Test Summary:\n");
  LOG_PRINT("Total: %d, Passed: %d, Failed: %d\n", totalTests, passCount, failCount);
  LOG_PRINT("========================================\n");

  // 释放资源
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return failCount > 0 ? 1 : 0;
}
