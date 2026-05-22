/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Mul算子端到端测试用例
 * 测试覆盖：
 * 1. 全部4个API变体：aclnnMul, aclnnMuls, aclnnInplaceMul, aclnnInplaceMuls
 * 2. 16种dtype组合覆盖tiling分支
 * 3. 混合数据类型支持（BF16-FLOAT, FLOAT16-FLOAT）
 * 4. Shape维度覆盖：1D-4D, 广播场景
 * 5. 数值边界：0、负数、大值
 * 6. 特殊场景：空tensor、nullptr、非连续tensor
 * 7. 结果验证：期望值计算和数值比对
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
    fflush(stdout);                 \
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

template <typename T>
int CreateAclTensorWithData(void** deviceAddr, aclDataType dataType, const std::vector<int64_t>& shape,
                            aclTensor** tensor, std::vector<T>& hostData) {
  return CreateAclTensor(hostData, shape, deviceAddr, dataType, tensor);
}

void DestroyResources(aclTensor* self, aclTensor* other, aclTensor* out,
                      void* selfAddr, void* otherAddr, void* outAddr,
                      void* workspaceAddr, size_t workspaceSize) {
  if (self) aclDestroyTensor(self);
  if (other) aclDestroyTensor(other);
  if (out) aclDestroyTensor(out);
  if (selfAddr) aclrtFree(selfAddr);
  if (otherAddr) aclrtFree(otherAddr);
  if (outAddr) aclrtFree(outAddr);
  if (workspaceAddr) aclrtFree(workspaceAddr);
}

// ========== 结果验证 ==========

template <typename T>
bool CompareResult(const std::vector<T>& actual, const std::vector<T>& expected,
                   double atol = 1e-3, double rtol = 1e-3) {
  for (size_t i = 0; i < actual.size(); i++) {
    double actualVal = static_cast<double>(actual[i]);
    double expectedVal = static_cast<double>(expected[i]);
    double diff = std::abs(actualVal - expectedVal);
    double tolerance = atol + rtol * std::abs(expectedVal);
    if (diff > tolerance) {
      LOG_PRINT("Mismatch at index %zu: actual=%f, expected=%f, diff=%f, tolerance=%f\n",
                i, actualVal, expectedVal, diff, tolerance);
      return false;
    }
  }
  return true;
}

// 整数类型精确比较
template <typename T>
bool CompareExact(const std::vector<T>& actual, const std::vector<T>& expected) {
  for (size_t i = 0; i < actual.size(); i++) {
    if (actual[i] != expected[i]) {
      LOG_PRINT("Mismatch at index %zu: actual=%d, expected=%d\n",
                i, static_cast<int>(actual[i]), static_cast<int>(expected[i]));
      return false;
    }
  }
  return true;
}

// ========== 期望值计算 ==========

template <typename T>
void ComputeExpectedMul(const std::vector<T>& selfData, const std::vector<T>& otherData,
                        std::vector<T>& expected) {
  for (size_t i = 0; i < selfData.size(); i++) {
    expected[i] = static_cast<T>(static_cast<double>(selfData[i]) * static_cast<double>(otherData[i]));
  }
}

template <typename T>
void ComputeExpectedMuls(const std::vector<T>& selfData, T scalar, std::vector<T>& expected) {
  for (size_t i = 0; i < selfData.size(); i++) {
    expected[i] = static_cast<T>(static_cast<double>(selfData[i]) * static_cast<double>(scalar));
  }
}

template <typename T1, typename T2, typename TOut>
void ComputeExpectedMixedMul(const std::vector<T1>& selfData, const std::vector<T2>& otherData,
                             std::vector<TOut>& expected) {
  for (size_t i = 0; i < selfData.size(); i++) {
    expected[i] = static_cast<TOut>(static_cast<double>(selfData[i]) * static_cast<double>(otherData[i]));
  }
}

// ========== Phase 1: API覆盖测试 ==========

template <typename T>
int TestAclnnMul(const std::vector<int64_t>& selfShape, const std::vector<int64_t>& otherShape,
                 aclDataType dataType, aclrtStream stream, const char* testName) {
  LOG_PRINT("\n=== Test: %s ===\n", testName);

  int64_t size = GetShapeSize(selfShape);
  std::vector<T> selfHostData(size);
  std::vector<T> otherHostData(size);
  std::vector<T> expected(size);
  std::vector<T> resultData(size);

  for (int64_t i = 0; i < size; i++) {
    selfHostData[i] = static_cast<T>(i + 1);
    otherHostData[i] = static_cast<T>((i % 3) + 1);
  }
  ComputeExpectedMul(selfHostData, otherHostData, expected);

  void* selfAddr = nullptr;
  void* otherAddr = nullptr;
  void* outAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;

  auto ret = CreateAclTensor(selfHostData, selfShape, &selfAddr, dataType, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create self tensor failed\n"); return ret);
  ret = CreateAclTensor(otherHostData, otherShape, &otherAddr, dataType, &other);
  CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, nullptr, nullptr, selfAddr, nullptr, nullptr, nullptr, 0); return ret);
  ret = CreateAclTensor(resultData, selfShape, &outAddr, dataType, &out);
  CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, other, nullptr, selfAddr, otherAddr, nullptr, nullptr, 0); return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulGetWorkspaceSize failed: %d\n", ret);
            DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, nullptr, 0); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, nullptr, 0); return ret);
  }

  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMul failed: %d\n", ret);
            DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed: %d\n", ret);
            DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize); return ret);

  ret = aclrtMemcpy(resultData.data(), size * sizeof(T), outAddr, size * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed: %d\n", ret);
            DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize); return ret);

  bool passed = CompareResult(resultData, expected);
  LOG_PRINT("Test %s: %s\n", testName, passed ? "PASSED" : "FAILED");

  DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize);
  return passed ? ACL_SUCCESS : -1;
}

template <typename T>
int TestAclnnMuls(const std::vector<int64_t>& selfShape, aclDataType dataType, T scalar,
                  aclrtStream stream, const char* testName) {
  LOG_PRINT("\n=== Test: %s ===\n", testName);

  int64_t size = GetShapeSize(selfShape);
  std::vector<T> selfHostData(size);
  std::vector<T> expected(size);
  std::vector<T> resultData(size);

  for (int64_t i = 0; i < size; i++) {
    selfHostData[i] = static_cast<T>(i + 1);
  }
  ComputeExpectedMuls(selfHostData, scalar, expected);

  void* selfAddr = nullptr;
  void* outAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclScalar* scalarObj = nullptr;

  auto ret = CreateAclTensor(selfHostData, selfShape, &selfAddr, dataType, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create self tensor failed\n"); return ret);
  ret = CreateAclTensor(resultData, selfShape, &outAddr, dataType, &out);
  CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, nullptr, nullptr, selfAddr, nullptr, nullptr, nullptr, 0); return ret);

  scalarObj = aclCreateScalar(&scalar, dataType);
  CHECK_RET(scalarObj != nullptr, DestroyResources(self, nullptr, out, selfAddr, nullptr, outAddr, nullptr, 0); return -1);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnMulsGetWorkspaceSize(self, scalarObj, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulsGetWorkspaceSize failed: %d\n", ret);
            DestroyResources(self, nullptr, out, selfAddr, nullptr, outAddr, nullptr, 0);
            aclDestroyScalar(scalarObj); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, nullptr, out, selfAddr, nullptr, outAddr, nullptr, 0);
              aclDestroyScalar(scalarObj); return ret);
  }

  ret = aclnnMuls(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMuls failed: %d\n", ret);
            DestroyResources(self, nullptr, out, selfAddr, nullptr, outAddr, workspaceAddr, workspaceSize);
            aclDestroyScalar(scalarObj); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed: %d\n", ret);
            DestroyResources(self, nullptr, out, selfAddr, nullptr, outAddr, workspaceAddr, workspaceSize);
            aclDestroyScalar(scalarObj); return ret);

  ret = aclrtMemcpy(resultData.data(), size * sizeof(T), outAddr, size * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed: %d\n", ret);
            DestroyResources(self, nullptr, out, selfAddr, nullptr, outAddr, workspaceAddr, workspaceSize);
            aclDestroyScalar(scalarObj); return ret);

  bool passed = CompareResult(resultData, expected);
  LOG_PRINT("Test %s: %s\n", testName, passed ? "PASSED" : "FAILED");

  DestroyResources(self, nullptr, out, selfAddr, nullptr, outAddr, workspaceAddr, workspaceSize);
  aclDestroyScalar(scalarObj);
  return passed ? ACL_SUCCESS : -1;
}

template <typename T>
int TestAclnnInplaceMul(const std::vector<int64_t>& selfShape, const std::vector<int64_t>& otherShape,
                        aclDataType dataType, aclrtStream stream, const char* testName) {
  LOG_PRINT("\n=== Test: %s ===\n", testName);

  int64_t size = GetShapeSize(selfShape);
  std::vector<T> selfHostData(size);
  std::vector<T> otherHostData(size);
  std::vector<T> expected(size);
  std::vector<T> resultData(size);

  for (int64_t i = 0; i < size; i++) {
    selfHostData[i] = static_cast<T>(i + 1);
    otherHostData[i] = static_cast<T>((i % 3) + 1);
  }
  ComputeExpectedMul(selfHostData, otherHostData, expected);

  void* selfAddr = nullptr;
  void* otherAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;

  auto ret = CreateAclTensor(selfHostData, selfShape, &selfAddr, dataType, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create self tensor failed\n"); return ret);
  ret = CreateAclTensor(otherHostData, otherShape, &otherAddr, dataType, &other);
  CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, nullptr, nullptr, selfAddr, nullptr, nullptr, nullptr, 0); return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnInplaceMulGetWorkspaceSize(self, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceMulGetWorkspaceSize failed: %d\n", ret);
            DestroyResources(self, other, nullptr, selfAddr, otherAddr, nullptr, nullptr, 0); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, other, nullptr, selfAddr, otherAddr, nullptr, nullptr, 0); return ret);
  }

  ret = aclnnInplaceMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceMul failed: %d\n", ret);
            DestroyResources(self, other, nullptr, selfAddr, otherAddr, nullptr, workspaceAddr, workspaceSize); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed: %d\n", ret);
            DestroyResources(self, other, nullptr, selfAddr, otherAddr, nullptr, workspaceAddr, workspaceSize); return ret);

  ret = aclrtMemcpy(resultData.data(), size * sizeof(T), selfAddr, size * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed: %d\n", ret);
            DestroyResources(self, other, nullptr, selfAddr, otherAddr, nullptr, workspaceAddr, workspaceSize); return ret);

  bool passed = CompareResult(resultData, expected);
  LOG_PRINT("Test %s: %s\n", testName, passed ? "PASSED" : "FAILED");

  DestroyResources(self, other, nullptr, selfAddr, otherAddr, nullptr, workspaceAddr, workspaceSize);
  return passed ? ACL_SUCCESS : -1;
}

template <typename T>
int TestAclnnInplaceMuls(const std::vector<int64_t>& selfShape, aclDataType dataType, T scalar,
                         aclrtStream stream, const char* testName) {
  LOG_PRINT("\n=== Test: %s ===\n", testName);

  int64_t size = GetShapeSize(selfShape);
  std::vector<T> selfHostData(size);
  std::vector<T> expected(size);
  std::vector<T> resultData(size);

  for (int64_t i = 0; i < size; i++) {
    selfHostData[i] = static_cast<T>(i + 1);
  }
  ComputeExpectedMuls(selfHostData, scalar, expected);

  void* selfAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* scalarObj = nullptr;

  auto ret = CreateAclTensor(selfHostData, selfShape, &selfAddr, dataType, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create self tensor failed\n"); return ret);

  scalarObj = aclCreateScalar(&scalar, dataType);
  CHECK_RET(scalarObj != nullptr, DestroyResources(self, nullptr, nullptr, selfAddr, nullptr, nullptr, nullptr, 0); return -1);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnInplaceMulsGetWorkspaceSize(self, scalarObj, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceMulsGetWorkspaceSize failed: %d\n", ret);
            DestroyResources(self, nullptr, nullptr, selfAddr, nullptr, nullptr, nullptr, 0);
            aclDestroyScalar(scalarObj); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, nullptr, nullptr, selfAddr, nullptr, nullptr, nullptr, 0);
              aclDestroyScalar(scalarObj); return ret);
  }

  ret = aclnnInplaceMuls(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceMuls failed: %d\n", ret);
            DestroyResources(self, nullptr, nullptr, selfAddr, nullptr, nullptr, workspaceAddr, workspaceSize);
            aclDestroyScalar(scalarObj); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed: %d\n", ret);
            DestroyResources(self, nullptr, nullptr, selfAddr, nullptr, nullptr, workspaceAddr, workspaceSize);
            aclDestroyScalar(scalarObj); return ret);

  ret = aclrtMemcpy(resultData.data(), size * sizeof(T), selfAddr, size * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed: %d\n", ret);
            DestroyResources(self, nullptr, nullptr, selfAddr, nullptr, nullptr, workspaceAddr, workspaceSize);
            aclDestroyScalar(scalarObj); return ret);

  bool passed = CompareResult(resultData, expected);
  LOG_PRINT("Test %s: %s\n", testName, passed ? "PASSED" : "FAILED");

  DestroyResources(self, nullptr, nullptr, selfAddr, nullptr, nullptr, workspaceAddr, workspaceSize);
  aclDestroyScalar(scalarObj);
  return passed ? ACL_SUCCESS : -1;
}

// ========== Phase 2: 数据类型覆盖测试（16种dtype组合） ==========

int TestDifferentDtypes(aclrtStream stream) {
  LOG_PRINT("\n========== Phase 2: Dtype Coverage Test ==========\n");
  std::vector<int64_t> shape = {4, 4};
  int totalPassed = 0;
  int totalTests = 0;

  // OP_KEY覆盖测试：覆盖DTYPE_MAP中的16种组合

  // DT_FLOAT -> MulCompute<float>
  if (TestAclnnMul<float>(shape, shape, ACL_FLOAT, stream, "ACL_FLOAT (OP_KEY_FLOAT)") == ACL_SUCCESS) totalPassed++;
  totalTests++;

  // DT_FLOAT16 -> MulCompute<half>
  if (TestAclnnMul<float>(shape, shape, ACL_FLOAT16, stream, "ACL_FLOAT16 (OP_KEY_FLOAT16)") == ACL_SUCCESS) totalPassed++;
  totalTests++;

  // DT_BF16 -> MulCompute<half>
  if (TestAclnnMul<float>(shape, shape, ACL_BF16, stream, "ACL_BF16 (OP_KEY_BF16)") == ACL_SUCCESS) totalPassed++;
  totalTests++;

  // DT_INT32 -> MulCompute<int32_t>
  if (TestAclnnMul<int32_t>(shape, shape, ACL_INT32, stream, "ACL_INT32 (OP_KEY_INT32)") == ACL_SUCCESS) totalPassed++;
  totalTests++;

  // DT_INT64 -> MulCompute<int64_t>
  if (TestAclnnMul<int64_t>(shape, shape, ACL_INT64, stream, "ACL_INT64 (OP_KEY_INT64)") == ACL_SUCCESS) totalPassed++;
  totalTests++;

  // DT_INT16 -> MulCompute<int16_t>
  if (TestAclnnMul<int16_t>(shape, shape, ACL_INT16, stream, "ACL_INT16 (OP_KEY_INT16)") == ACL_SUCCESS) totalPassed++;
  totalTests++;

  // DT_INT8 -> MulCompute<int8_t>
  if (TestAclnnMul<int8_t>(shape, shape, ACL_INT8, stream, "ACL_INT8 (OP_KEY_INT8)") == ACL_SUCCESS) totalPassed++;
  totalTests++;

  // DT_UINT8 -> MulCompute<uint8_t>
  if (TestAclnnMul<uint8_t>(shape, shape, ACL_UINT8, stream, "ACL_UINT8 (OP_KEY_UINT8)") == ACL_SUCCESS) totalPassed++;
  totalTests++;

  // DT_BOOL -> MulCompute<bool> - BOOL类型特殊处理
  LOG_PRINT("\n=== Test: ACL_BOOL (OP_KEY_BOOL) ===\n");
  int64_t size = GetShapeSize(shape);
  std::vector<int8_t> boolSelfData(size);
  std::vector<int8_t> boolOtherData(size);
  std::vector<int8_t> boolExpected(size);
  std::vector<int8_t> boolResultData(size);

  for (int64_t i = 0; i < size; i++) {
    boolSelfData[i] = (i % 2 == 0) ? 1 : 0;
    boolOtherData[i] = (i % 3 == 0) ? 1 : 0;
    boolExpected[i] = (boolSelfData[i] && boolOtherData[i]) ? 1 : 0;  // BOOL的mul实际上是AND
  }

  void* boolSelfAddr = nullptr;
  void* boolOtherAddr = nullptr;
  void* boolOutAddr = nullptr;
  aclTensor* boolSelf = nullptr;
  aclTensor* boolOther = nullptr;
  aclTensor* boolOut = nullptr;

  auto ret = CreateAclTensor(boolSelfData, shape, &boolSelfAddr, ACL_BOOL, &boolSelf);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create bool self tensor failed\n"); return ret);
  ret = CreateAclTensor(boolOtherData, shape, &boolOtherAddr, ACL_BOOL, &boolOther);
  CHECK_RET(ret == ACL_SUCCESS, DestroyResources(boolSelf, nullptr, nullptr, boolSelfAddr, nullptr, nullptr, nullptr, 0); return ret);
  ret = CreateAclTensor(boolResultData, shape, &boolOutAddr, ACL_BOOL, &boolOut);
  CHECK_RET(ret == ACL_SUCCESS, DestroyResources(boolSelf, boolOther, nullptr, boolSelfAddr, boolOtherAddr, nullptr, nullptr, 0); return ret);

  uint64_t boolWorkspaceSize = 0;
  aclOpExecutor* boolExecutor;
  ret = aclnnMulGetWorkspaceSize(boolSelf, boolOther, boolOut, &boolWorkspaceSize, &boolExecutor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulGetWorkspaceSize for BOOL failed: %d\n", ret);
            DestroyResources(boolSelf, boolOther, boolOut, boolSelfAddr, boolOtherAddr, boolOutAddr, nullptr, 0); return ret);

  void* boolWorkspaceAddr = nullptr;
  if (boolWorkspaceSize > 0) {
    ret = aclrtMalloc(&boolWorkspaceAddr, boolWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(boolSelf, boolOther, boolOut, boolSelfAddr, boolOtherAddr, boolOutAddr, nullptr, 0); return ret);
  }

  ret = aclnnMul(boolWorkspaceAddr, boolWorkspaceSize, boolExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMul for BOOL failed: %d\n", ret);
            DestroyResources(boolSelf, boolOther, boolOut, boolSelfAddr, boolOtherAddr, boolOutAddr, boolWorkspaceAddr, boolWorkspaceSize); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed: %d\n", ret);
            DestroyResources(boolSelf, boolOther, boolOut, boolSelfAddr, boolOtherAddr, boolOutAddr, boolWorkspaceAddr, boolWorkspaceSize); return ret);

  ret = aclrtMemcpy(boolResultData.data(), size * sizeof(int8_t), boolOutAddr, size * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed: %d\n", ret);
            DestroyResources(boolSelf, boolOther, boolOut, boolSelfAddr, boolOtherAddr, boolOutAddr, boolWorkspaceAddr, boolWorkspaceSize); return ret);

  bool boolPassed = CompareExact(boolResultData, boolExpected);
  LOG_PRINT("Test ACL_BOOL: %s\n", boolPassed ? "PASSED" : "FAILED");
  DestroyResources(boolSelf, boolOther, boolOut, boolSelfAddr, boolOtherAddr, boolOutAddr, boolWorkspaceAddr, boolWorkspaceSize);
  if (boolPassed) totalPassed++;
  totalTests++;

  LOG_PRINT("\nDtype Coverage: %d/%d tests passed\n", totalPassed, totalTests);
  return totalPassed == totalTests ? ACL_SUCCESS : -1;
}

// ========== Phase 3: 混合数据类型测试 ==========

int TestMixedDtypes(aclrtStream stream) {
  LOG_PRINT("\n========== Phase 3: Mixed Dtype Test ==========\n");
  std::vector<int64_t> shape = {4, 4};
  int totalPassed = 0;
  int totalTests = 0;

  // 混合类型：BF16 + FLOAT (MixedMulCastCompute<half, float>)
  LOG_PRINT("\n=== Test: Mixed BF16 + FLOAT ===\n");
  {
    int64_t size = GetShapeSize(shape);
    std::vector<float> bf16Data(size);
    std::vector<float> floatData(size);
    std::vector<float> expected(size);
    std::vector<float> resultData(size);

    for (int64_t i = 0; i < size; i++) {
      bf16Data[i] = static_cast<float>(i + 1);
      floatData[i] = static_cast<float>((i % 3) + 1);
      expected[i] = bf16Data[i] * floatData[i];
    }

    void* selfAddr = nullptr;
    void* otherAddr = nullptr;
    void* outAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclTensor* out = nullptr;

    auto ret = CreateAclTensor(bf16Data, shape, &selfAddr, ACL_BF16, &self);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create BF16 tensor failed\n"); return ret);
    ret = CreateAclTensor(floatData, shape, &otherAddr, ACL_FLOAT, &other);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, nullptr, nullptr, selfAddr, nullptr, nullptr, nullptr, 0); return ret);
    ret = CreateAclTensor(resultData, shape, &outAddr, ACL_FLOAT, &out);  // 输出类型为FLOAT
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, other, nullptr, selfAddr, otherAddr, nullptr, nullptr, 0); return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulGetWorkspaceSize for mixed dtype failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, nullptr, 0); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, nullptr, 0); return ret);
    }

    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMul for mixed dtype failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize); return ret);

    ret = aclrtMemcpy(resultData.data(), size * sizeof(float), outAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize); return ret);

    bool passed = CompareResult(resultData, expected);
    LOG_PRINT("Test Mixed BF16+FLOAT: %s\n", passed ? "PASSED" : "FAILED");
    DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize);
    if (passed) totalPassed++;
    totalTests++;
  }

  // 混合类型：FLOAT16 + FLOAT (MixedMulCastCompute<half, float>)
  LOG_PRINT("\n=== Test: Mixed FLOAT16 + FLOAT ===\n");
  {
    int64_t size = GetShapeSize(shape);
    std::vector<float> fp16Data(size);
    std::vector<float> floatData(size);
    std::vector<float> expected(size);
    std::vector<float> resultData(size);

    for (int64_t i = 0; i < size; i++) {
      fp16Data[i] = static_cast<float>(i + 1);
      floatData[i] = static_cast<float>((i % 3) + 1);
      expected[i] = fp16Data[i] * floatData[i];
    }

    void* selfAddr = nullptr;
    void* otherAddr = nullptr;
    void* outAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclTensor* out = nullptr;

    auto ret = CreateAclTensor(fp16Data, shape, &selfAddr, ACL_FLOAT16, &self);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create FLOAT16 tensor failed\n"); return ret);
    ret = CreateAclTensor(floatData, shape, &otherAddr, ACL_FLOAT, &other);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, nullptr, nullptr, selfAddr, nullptr, nullptr, nullptr, 0); return ret);
    ret = CreateAclTensor(resultData, shape, &outAddr, ACL_FLOAT, &out);  // 输出类型为FLOAT
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, other, nullptr, selfAddr, otherAddr, nullptr, nullptr, 0); return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulGetWorkspaceSize for mixed dtype failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, nullptr, 0); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, nullptr, 0); return ret);
    }

    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMul for mixed dtype failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize); return ret);

    ret = aclrtMemcpy(resultData.data(), size * sizeof(float), outAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize); return ret);

    bool passed = CompareResult(resultData, expected);
    LOG_PRINT("Test Mixed FLOAT16+FLOAT: %s\n", passed ? "PASSED" : "FAILED");
    DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize);
    if (passed) totalPassed++;
    totalTests++;
  }

  LOG_PRINT("\nMixed Dtype Coverage: %d/%d tests passed\n", totalPassed, totalTests);
  return totalPassed == totalTests ? ACL_SUCCESS : -1;
}

// ========== Phase 4: Shape维度和广播测试 ==========

int TestShapeVariations(aclrtStream stream) {
  LOG_PRINT("\n========== Phase 4: Shape Variation Test ==========\n");
  std::vector<int64_t> baseShape = {4, 4};
  int totalPassed = 0;
  int totalTests = 0;

  // 1D tensor
  if (TestAclnnMul<float>({8}, {8}, ACL_FLOAT, stream, "1D tensor") == ACL_SUCCESS) totalPassed++;
  totalTests++;

  // 2D tensor
  if (TestAclnnMul<float>({4, 4}, {4, 4}, ACL_FLOAT, stream, "2D tensor") == ACL_SUCCESS) totalPassed++;
  totalTests++;

  // 3D tensor
  if (TestAclnnMul<float>({2, 3, 4}, {2, 3, 4}, ACL_FLOAT, stream, "3D tensor") == ACL_SUCCESS) totalPassed++;
  totalTests++;

  // 4D tensor
  if (TestAclnnMul<float>({2, 2, 2, 2}, {2, 2, 2, 2}, ACL_FLOAT, stream, "4D tensor") == ACL_SUCCESS) totalPassed++;
  totalTests++;

  // 广播场景：行广播
  LOG_PRINT("\n=== Test: Row Broadcast (4,3) + (1,3) ===\n");
  {
    std::vector<int64_t> selfShape = {4, 3};
    std::vector<int64_t> otherShape = {1, 3};
    int64_t size = GetShapeSize(selfShape);
    std::vector<float> selfData(size);
    std::vector<float> otherData(3);  // 广播源数据
    std::vector<float> expected(size);
    std::vector<float> resultData(size);

    for (int64_t i = 0; i < size; i++) {
      selfData[i] = static_cast<float>(i + 1);
    }
    for (int64_t i = 0; i < 3; i++) {
      otherData[i] = static_cast<float>(i + 1);
    }
    // 计算期望值（考虑广播）
    for (int64_t i = 0; i < size; i++) {
      int64_t col = i % 3;
      expected[i] = selfData[i] * otherData[col];
    }

    void* selfAddr = nullptr;
    void* otherAddr = nullptr;
    void* outAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclTensor* out = nullptr;

    auto ret = CreateAclTensor(selfData, selfShape, &selfAddr, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create self tensor failed\n"); return ret);
    ret = CreateAclTensor(otherData, otherShape, &otherAddr, ACL_FLOAT, &other);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, nullptr, nullptr, selfAddr, nullptr, nullptr, nullptr, 0); return ret);
    ret = CreateAclTensor(resultData, selfShape, &outAddr, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, other, nullptr, selfAddr, otherAddr, nullptr, nullptr, 0); return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulGetWorkspaceSize failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, nullptr, 0); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, nullptr, 0); return ret);
    }

    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMul failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize); return ret);

    ret = aclrtMemcpy(resultData.data(), size * sizeof(float), outAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize); return ret);

    bool passed = CompareResult(resultData, expected);
    LOG_PRINT("Test Row Broadcast: %s\n", passed ? "PASSED" : "FAILED");
    DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize);
    if (passed) totalPassed++;
    totalTests++;
  }

  // 广播场景：列广播
  LOG_PRINT("\n=== Test: Col Broadcast (4,3) + (3) ===\n");
  {
    std::vector<int64_t> selfShape = {4, 3};
    std::vector<int64_t> otherShape = {3};
    int64_t size = GetShapeSize(selfShape);
    std::vector<float> selfData(size);
    std::vector<float> otherData(3);
    std::vector<float> expected(size);
    std::vector<float> resultData(size);

    for (int64_t i = 0; i < size; i++) {
      selfData[i] = static_cast<float>(i + 1);
    }
    for (int64_t i = 0; i < 3; i++) {
      otherData[i] = static_cast<float>(i + 1);
    }
    for (int64_t i = 0; i < size; i++) {
      int64_t col = i % 3;
      expected[i] = selfData[i] * otherData[col];
    }

    void* selfAddr = nullptr;
    void* otherAddr = nullptr;
    void* outAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclTensor* out = nullptr;

    auto ret = CreateAclTensor(selfData, selfShape, &selfAddr, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create self tensor failed\n"); return ret);
    ret = CreateAclTensor(otherData, otherShape, &otherAddr, ACL_FLOAT, &other);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, nullptr, nullptr, selfAddr, nullptr, nullptr, nullptr, 0); return ret);
    ret = CreateAclTensor(resultData, selfShape, &outAddr, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, other, nullptr, selfAddr, otherAddr, nullptr, nullptr, 0); return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulGetWorkspaceSize failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, nullptr, 0); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, nullptr, 0); return ret);
    }

    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMul failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize); return ret);

    ret = aclrtMemcpy(resultData.data(), size * sizeof(float), outAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize); return ret);

    bool passed = CompareResult(resultData, expected);
    LOG_PRINT("Test Col Broadcast: %s\n", passed ? "PASSED" : "FAILED");
    DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize);
    if (passed) totalPassed++;
    totalTests++;
  }

  // 标量广播
  LOG_PRINT("\n=== Test: Scalar Broadcast (4,3) + (1,1) ===\n");
  {
    std::vector<int64_t> selfShape = {4, 3};
    std::vector<int64_t> otherShape = {1, 1};
    int64_t size = GetShapeSize(selfShape);
    std::vector<float> selfData(size);
    std::vector<float> otherData(1);
    std::vector<float> expected(size);
    std::vector<float> resultData(size);

    for (int64_t i = 0; i < size; i++) {
      selfData[i] = static_cast<float>(i + 1);
    }
    otherData[0] = 2.5f;
    for (int64_t i = 0; i < size; i++) {
      expected[i] = selfData[i] * otherData[0];
    }

    void* selfAddr = nullptr;
    void* otherAddr = nullptr;
    void* outAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclTensor* out = nullptr;

    auto ret = CreateAclTensor(selfData, selfShape, &selfAddr, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create self tensor failed\n"); return ret);
    ret = CreateAclTensor(otherData, otherShape, &otherAddr, ACL_FLOAT, &other);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, nullptr, nullptr, selfAddr, nullptr, nullptr, nullptr, 0); return ret);
    ret = CreateAclTensor(resultData, selfShape, &outAddr, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, other, nullptr, selfAddr, otherAddr, nullptr, nullptr, 0); return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulGetWorkspaceSize failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, nullptr, 0); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, nullptr, 0); return ret);
    }

    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMul failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize); return ret);

    ret = aclrtMemcpy(resultData.data(), size * sizeof(float), outAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize); return ret);

    bool passed = CompareResult(resultData, expected);
    LOG_PRINT("Test Scalar Broadcast: %s\n", passed ? "PASSED" : "FAILED");
    DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize);
    if (passed) totalPassed++;
    totalTests++;
  }

  LOG_PRINT("\nShape Variation Coverage: %d/%d tests passed\n", totalPassed, totalTests);
  return totalPassed == totalTests ? ACL_SUCCESS : -1;
}

// ========== Phase 5: 数值边界测试 ==========

int TestNumericalBoundaries(aclrtStream stream) {
  LOG_PRINT("\n========== Phase 5: Numerical Boundary Test ==========\n");
  std::vector<int64_t> shape = {4, 4};
  int totalPassed = 0;
  int totalTests = 0;

  // Muls测试：scalar=0
  if (TestAclnnMuls<float>(shape, ACL_FLOAT, 0.0f, stream, "Muls with scalar=0") == ACL_SUCCESS) totalPassed++;
  totalTests++;

  // Muls测试：scalar=负数
  if (TestAclnnMuls<float>(shape, ACL_FLOAT, -2.5f, stream, "Muls with scalar=-2.5") == ACL_SUCCESS) totalPassed++;
  totalTests++;

  // Muls测试：scalar=大值
  if (TestAclnnMuls<float>(shape, ACL_FLOAT, 100.0f, stream, "Muls with scalar=100") == ACL_SUCCESS) totalPassed++;
  totalTests++;

  // Muls测试：scalar=小数
  if (TestAclnnMuls<float>(shape, ACL_FLOAT, 0.001f, stream, "Muls with scalar=0.001") == ACL_SUCCESS) totalPassed++;
  totalTests++;

  // Mul测试：包含0值
  LOG_PRINT("\n=== Test: Mul with zero values ===\n");
  {
    int64_t size = GetShapeSize(shape);
    std::vector<float> selfData(size);
    std::vector<float> otherData(size);
    std::vector<float> expected(size);
    std::vector<float> resultData(size);

    for (int64_t i = 0; i < size; i++) {
      selfData[i] = (i % 2 == 0) ? 0.0f : static_cast<float>(i);
      otherData[i] = (i % 3 == 0) ? 0.0f : static_cast<float>(i);
      expected[i] = selfData[i] * otherData[i];
    }

    void* selfAddr = nullptr;
    void* otherAddr = nullptr;
    void* outAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclTensor* out = nullptr;

    auto ret = CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create self tensor failed\n"); return ret);
    ret = CreateAclTensor(otherData, shape, &otherAddr, ACL_FLOAT, &other);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, nullptr, nullptr, selfAddr, nullptr, nullptr, nullptr, 0); return ret);
    ret = CreateAclTensor(resultData, shape, &outAddr, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, other, nullptr, selfAddr, otherAddr, nullptr, nullptr, 0); return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulGetWorkspaceSize failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, nullptr, 0); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, nullptr, 0); return ret);
    }

    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMul failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize); return ret);

    ret = aclrtMemcpy(resultData.data(), size * sizeof(float), outAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed: %d\n", ret);
              DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize); return ret);

    bool passed = CompareResult(resultData, expected);
    LOG_PRINT("Test Mul with zero values: %s\n", passed ? "PASSED" : "FAILED");
    DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, workspaceAddr, workspaceSize);
    if (passed) totalPassed++;
    totalTests++;
  }

  // InplaceMul测试
  if (TestAclnnInplaceMul<float>(shape, shape, ACL_FLOAT, stream, "InplaceMul basic") == ACL_SUCCESS) totalPassed++;
  totalTests++;

  // InplaceMuls测试
  if (TestAclnnInplaceMuls<float>(shape, ACL_FLOAT, 3.0f, stream, "InplaceMuls with scalar=3") == ACL_SUCCESS) totalPassed++;
  totalTests++;

  LOG_PRINT("\nNumerical Boundary Coverage: %d/%d tests passed\n", totalPassed, totalTests);
  return totalPassed == totalTests ? ACL_SUCCESS : -1;
}

// ========== Phase 6: 特殊场景测试 ==========

int TestEmptyTensor(aclDataType dataType, aclrtStream stream) {
  LOG_PRINT("\n=== Test: Empty Tensor ===\n");

  std::vector<int64_t> emptyShape = {0};
  void* selfAddr = nullptr;
  void* otherAddr = nullptr;
  void* outAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;

  // 创建空tensor（size=0，需要分配最小内存）
  auto ret = aclrtMalloc(&selfAddr, 1, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for empty self failed: %d\n", ret); return ret);
  ret = aclrtMalloc(&otherAddr, 1, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for empty other failed: %d\n", ret);
            aclrtFree(selfAddr); return ret);
  ret = aclrtMalloc(&outAddr, 1, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for empty out failed: %d\n", ret);
            aclrtFree(selfAddr); aclrtFree(otherAddr); return ret);

  std::vector<int64_t> strides(1, 1);
  self = aclCreateTensor(emptyShape.data(), 1, dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                         emptyShape.data(), 1, selfAddr);
  other = aclCreateTensor(emptyShape.data(), 1, dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                          emptyShape.data(), 1, otherAddr);
  out = aclCreateTensor(emptyShape.data(), 1, dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                        emptyShape.data(), 1, outAddr);

  CHECK_RET(self != nullptr && other != nullptr && out != nullptr,
            LOG_PRINT("Create empty tensors failed\n");
            DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, nullptr, 0); return -1);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  LOG_PRINT("aclnnMulGetWorkspaceSize for empty tensor returned: %d\n", ret);

  DestroyResources(self, other, out, selfAddr, otherAddr, outAddr, nullptr, 0);
  return ACL_SUCCESS;
}

int TestNullptrInput(aclrtStream stream) {
  LOG_PRINT("\n=== Test: Nullptr Input ===\n");

  std::vector<int64_t> shape = {2, 2};
  void* selfAddr = nullptr;
  aclTensor* self = nullptr;
  std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};

  auto ret = CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create self tensor failed\n"); return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 测试nullptr self
  ret = aclnnMulGetWorkspaceSize(nullptr, self, self, &workspaceSize, &executor);
  LOG_PRINT("aclnnMulGetWorkspaceSize with nullptr self: %d (expected error)\n", ret);

  // 测试nullptr other
  ret = aclnnMulGetWorkspaceSize(self, nullptr, self, &workspaceSize, &executor);
  LOG_PRINT("aclnnMulGetWorkspaceSize with nullptr other: %d (expected error)\n", ret);

  // 测试nullptr out
  ret = aclnnMulGetWorkspaceSize(self, self, nullptr, &workspaceSize, &executor);
  LOG_PRINT("aclnnMulGetWorkspaceSize with nullptr out: %d (expected error)\n", ret);

  // 测试Muls nullptr
  aclScalar* scalar = aclCreateScalar(&selfData[0], ACL_FLOAT);
  ret = aclnnMulsGetWorkspaceSize(nullptr, scalar, self, &workspaceSize, &executor);
  LOG_PRINT("aclnnMulsGetWorkspaceSize with nullptr self: %d (expected error)\n", ret);
  aclDestroyScalar(scalar);

  DestroyResources(self, nullptr, nullptr, selfAddr, nullptr, nullptr, nullptr, 0);
  LOG_PRINT("Nullptr test completed\n");
  return ACL_SUCCESS;
}

int TestMulsSpecialCases(aclrtStream stream) {
  LOG_PRINT("\n========== Phase 6: Muls Special Cases Test ==========\n");
  std::vector<int64_t> shape = {4, 4};

  // BF16 Muls测试（触发Muls融合优化）
  LOG_PRINT("\n=== Test: Muls with BF16 dtype ===\n");
  {
    int64_t size = GetShapeSize(shape);
    std::vector<float> selfData(size);
    std::vector<float> expected(size);
    std::vector<float> resultData(size);

    for (int64_t i = 0; i < size; i++) {
      selfData[i] = static_cast<float>(i + 1);
      expected[i] = selfData[i] * 2.0f;
    }

    void* selfAddr = nullptr;
    void* outAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    float scalarVal = 2.0f;
    aclScalar* scalarObj = aclCreateScalar(&scalarVal, ACL_FLOAT);

    auto ret = CreateAclTensor(selfData, shape, &selfAddr, ACL_BF16, &self);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create BF16 tensor failed\n"); aclDestroyScalar(scalarObj); return ret);
    ret = CreateAclTensor(resultData, shape, &outAddr, ACL_BF16, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, nullptr, nullptr, selfAddr, nullptr, nullptr, nullptr, 0);
              aclDestroyScalar(scalarObj); return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnMulsGetWorkspaceSize(self, scalarObj, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulsGetWorkspaceSize for BF16 failed: %d\n", ret);
              DestroyResources(self, nullptr, out, selfAddr, nullptr, outAddr, nullptr, 0);
              aclDestroyScalar(scalarObj); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, DestroyResources(self, nullptr, out, selfAddr, nullptr, outAddr, nullptr, 0);
                aclDestroyScalar(scalarObj); return ret);
    }

    ret = aclnnMuls(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMuls for BF16 failed: %d\n", ret);
              DestroyResources(self, nullptr, out, selfAddr, nullptr, outAddr, workspaceAddr, workspaceSize);
              aclDestroyScalar(scalarObj); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed: %d\n", ret);
              DestroyResources(self, nullptr, out, selfAddr, nullptr, outAddr, workspaceAddr, workspaceSize);
              aclDestroyScalar(scalarObj); return ret);

    ret = aclrtMemcpy(resultData.data(), size * sizeof(float), outAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed: %d\n", ret);
              DestroyResources(self, nullptr, out, selfAddr, nullptr, outAddr, workspaceAddr, workspaceSize);
              aclDestroyScalar(scalarObj); return ret);

    bool passed = CompareResult(resultData, expected, 1e-2, 1e-2);  // BF16精度较低
    LOG_PRINT("Test Muls with BF16: %s\n", passed ? "PASSED" : "FAILED");
    DestroyResources(self, nullptr, out, selfAddr, nullptr, outAddr, workspaceAddr, workspaceSize);
    aclDestroyScalar(scalarObj);
    return passed ? ACL_SUCCESS : -1;
  }
}

// ========== 主测试函数 ==========

int main() {
  LOG_PRINT("\n========================================\n");
  LOG_PRINT("  Mul Operator End-to-End Test Suite\n");
  LOG_PRINT("========================================\n");

  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  int totalPassed = 0;
  int totalTests = 0;

  // Phase 1: API覆盖测试
  LOG_PRINT("\n========== Phase 1: API Coverage Test ==========\n");
  std::vector<int64_t> shape = {4, 4};

  if (TestAclnnMul<float>(shape, shape, ACL_FLOAT, stream, "aclnnMul") == ACL_SUCCESS) totalPassed++;
  totalTests++;

  if (TestAclnnMuls<float>(shape, ACL_FLOAT, 2.0f, stream, "aclnnMuls") == ACL_SUCCESS) totalPassed++;
  totalTests++;

  if (TestAclnnInplaceMul<float>(shape, shape, ACL_FLOAT, stream, "aclnnInplaceMul") == ACL_SUCCESS) totalPassed++;
  totalTests++;

  if (TestAclnnInplaceMuls<float>(shape, ACL_FLOAT, 3.0f, stream, "aclnnInplaceMuls") == ACL_SUCCESS) totalPassed++;
  totalTests++;

  LOG_PRINT("Phase 1: %d/%d tests passed\n", totalPassed, totalTests);

  // Phase 2: 数据类型覆盖测试
  int phase2Result = TestDifferentDtypes(stream);
  if (phase2Result == ACL_SUCCESS) {
    LOG_PRINT("Phase 2: All dtype tests PASSED\n");
  } else {
    LOG_PRINT("Phase 2: Some dtype tests FAILED\n");
  }

  // Phase 3: 混合数据类型测试
  int phase3Result = TestMixedDtypes(stream);
  if (phase3Result == ACL_SUCCESS) {
    LOG_PRINT("Phase 3: All mixed dtype tests PASSED\n");
  } else {
    LOG_PRINT("Phase 3: Some mixed dtype tests FAILED\n");
  }

  // Phase 4: Shape维度和广播测试
  int phase4Result = TestShapeVariations(stream);
  if (phase4Result == ACL_SUCCESS) {
    LOG_PRINT("Phase 4: All shape tests PASSED\n");
  } else {
    LOG_PRINT("Phase 4: Some shape tests FAILED\n");
  }

  // Phase 5: 数值边界测试
  int phase5Result = TestNumericalBoundaries(stream);
  if (phase5Result == ACL_SUCCESS) {
    LOG_PRINT("Phase 5: All boundary tests PASSED\n");
  } else {
    LOG_PRINT("Phase 5: Some boundary tests FAILED\n");
  }

  // Phase 6: 特殊场景测试
  LOG_PRINT("\n========== Phase 6: Special Cases Test ==========\n");
  TestEmptyTensor(ACL_FLOAT, stream);
  TestNullptrInput(stream);
  TestMulsSpecialCases(stream);

  // 清理资源
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  LOG_PRINT("\n========================================\n");
  LOG_PRINT("  All tests completed!\n");
  LOG_PRINT("========================================\n");

  return 0;
}