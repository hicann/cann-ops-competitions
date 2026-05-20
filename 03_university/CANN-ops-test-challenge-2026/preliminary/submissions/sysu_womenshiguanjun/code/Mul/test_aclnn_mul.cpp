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
#include <algorithm>
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
  } while (0)

// 测试结果统计
static int g_total_tests = 0;
static int g_passed_tests = 0;
static int g_failed_tests = 0;

// 容差配置
const float FLOAT32_TOL = 1e-5f;
const float FLOAT16_TOL = 1e-3f;
const float BF16_TOL = 1e-2f;

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
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

// 创建标量tensor (shape = [1])
template <typename T>
int CreateScalarAclTensor(T scalarValue, void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
  std::vector<T> hostData = {scalarValue};
  std::vector<int64_t> shape = {1};
  return CreateAclTensor(hostData, shape, deviceAddr, dataType, tensor);
}

// 从device内存拷贝数据到host
template <typename T>
int CopyFromDevice(void* deviceAddr, std::vector<T>& hostData, int64_t size) {
  auto ret = aclrtMemcpy(hostData.data(), size * sizeof(T), deviceAddr,
                         size * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy from device to host failed. ERROR: %d\n", ret); return ret);
  return 0;
}

// 浮点数比较（带容差）
template <typename T>
bool AlmostEqual(T a, T b, float atol, float rtol) {
  if (std::isnan(a) && std::isnan(b)) return true;
  if (std::isinf(a) && std::isinf(b)) return (a > 0) == (b > 0);
  return std::abs(a - b) <= atol + rtol * std::abs(b);
}

// 整数精确比较
template <typename T>
bool ExactEqual(T a, T b) {
  return a == b;
}

// 特化浮点类型
template <>
bool AlmostEqual<float>(float a, float b, float atol, float rtol) {
  if (std::isnan(a) && std::isnan(b)) return true;
  if (std::isinf(a) && std::isinf(b)) return (a > 0) == (b > 0);
  return std::abs(a - b) <= atol + rtol * std::abs(b);
}

// 计算期望值：逐元素乘法
template <typename T>
std::vector<T> ComputeExpectedMul(const std::vector<T>& a, const std::vector<int64_t>& shapeA,
                                  const std::vector<T>& b, const std::vector<int64_t>& shapeB) {
  // 简化：假设shape已经广播对齐（测试用例中会处理）
  // 这里只处理同shape或标量广播到向量的简单情况
  int64_t sizeA = GetShapeSize(shapeA);
  int64_t sizeB = GetShapeSize(shapeB);
  int64_t outSize = std::max(sizeA, sizeB);
  std::vector<T> expected(outSize);
  
  if (sizeA == 1) { // a是标量
    T scalar = a[0];
    for (int64_t i = 0; i < outSize; i++) {
      expected[i] = scalar * b[i];
    }
  } else if (sizeB == 1) { // b是标量
    T scalar = b[0];
    for (int64_t i = 0; i < outSize; i++) {
      expected[i] = a[i] * scalar;
    }
  } else { // 同shape
    for (int64_t i = 0; i < outSize; i++) {
      expected[i] = a[i] * b[i];
    }
  }
  return expected;
}

// 验证结果
template <typename T>
bool VerifyResult(const std::vector<T>& actual, const std::vector<T>& expected,
                  aclDataType dtype, const char* testName) {
  bool pass = true;
  float atol = 0.0f;
  float rtol = 0.0f;
  
  // 设置容差
  switch (dtype) {
    case ACL_FLOAT:
      atol = FLOAT32_TOL;
      rtol = FLOAT32_TOL;
      break;
    case ACL_FLOAT16:
      atol = FLOAT16_TOL;
      rtol = FLOAT16_TOL;
      break;
    case ACL_BF16:
      atol = BF16_TOL;
      rtol = BF16_TOL;
      break;
    default:
      // 整数类型精确匹配
      atol = 0.0f;
      rtol = 0.0f;
      break;
  }
  
  for (size_t i = 0; i < actual.size(); i++) {
    bool equal = false;
    if (dtype == ACL_FLOAT || dtype == ACL_FLOAT16 || dtype == ACL_BF16) {
      equal = AlmostEqual(actual[i], expected[i], atol, rtol);
    } else {
      equal = ExactEqual(actual[i], expected[i]);
    }
    
    if (!equal) {
      LOG_PRINT("  Mismatch at index %zu: actual=%f, expected=%f\n", 
                i, static_cast<double>(actual[i]), static_cast<double>(expected[i]));
      pass = false;
      // 只打印前5个不匹配
      if (i >= 5) break;
    }
  }
  
  if (pass) {
    LOG_PRINT("[PASS] %s\n", testName);
    g_passed_tests++;
  } else {
    LOG_PRINT("[FAIL] %s\n", testName);
    g_failed_tests++;
  }
  g_total_tests++;
  return pass;
}

// 运行单个Mul测试用例
template <typename T>
bool RunMulTest(const std::vector<T>& selfData, const std::vector<int64_t>& selfShape,
                const std::vector<T>& otherData, const std::vector<int64_t>& otherShape,
                aclDataType dtype, aclrtStream stream, const char* testName) {
  // 创建tensor
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  
  // 输出shape（广播后）
  std::vector<int64_t> outShape = selfShape; // 简化：假设同shape或广播到self shape
  if (GetShapeSize(otherShape) > GetShapeSize(selfShape)) {
    outShape = otherShape;
  }
  
  std::vector<T> outHostData(GetShapeSize(outShape), 0);
  
  int ret = CreateAclTensor(selfData, selfShape, &selfDeviceAddr, dtype, &self);
  if (ret != 0) return false;
  ret = CreateAclTensor(otherData, otherShape, &otherDeviceAddr, dtype, &other);
  if (ret != 0) return false;
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, dtype, &out);
  if (ret != 0) return false;
  
  // 调用aclnnMul
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclnnMulGetWorkspaceSize failed. ERROR: %d\n", ret);
    return false;
  }
  
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
      return false;
    }
  }
  
  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclnnMul failed. ERROR: %d\n", ret);
    return false;
  }
  
  // 同步
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    return false;
  }
  
  // 获取输出
  std::vector<T> resultData(outHostData.size());
  ret = CopyFromDevice(outDeviceAddr, resultData, resultData.size());
  if (ret != 0) return false;
  
  // 计算期望值
  std::vector<T> expected = ComputeExpectedMul(selfData, selfShape, otherData, otherShape);
  
  // 验证
  bool pass = VerifyResult(resultData, expected, dtype, testName);
  
  // 清理
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  
  return pass;
}


// 运行混合数据类型Mul测试用例 (selfDtype * otherDtype -> outDtype)
template <typename TSelf, typename TOther, typename TOut>
bool RunMulMixDtypeTest(const std::vector<TSelf>& selfData, const std::vector<int64_t>& selfShape,
                        const std::vector<TOther>& otherData, const std::vector<int64_t>& otherShape,
                        const std::vector<TOut>& expectedData, aclDataType selfDtype, aclDataType otherDtype,
                        aclDataType outDtype, aclrtStream stream, const char* testName) {
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;

  std::vector<int64_t> outShape = selfShape;
  if (GetShapeSize(otherShape) > GetShapeSize(selfShape)) {
    outShape = otherShape;
  }

  std::vector<TOut> outHostData(GetShapeSize(outShape), 0);

  int ret = CreateAclTensor(selfData, selfShape, &selfDeviceAddr, selfDtype, &self);
  if (ret != 0) return false;
  ret = CreateAclTensor(otherData, otherShape, &otherDeviceAddr, otherDtype, &other);
  if (ret != 0) return false;
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, outDtype, &out);
  if (ret != 0) return false;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclnnMulGetWorkspaceSize failed. ERROR: %d\n", ret);
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr); aclrtFree(otherDeviceAddr); aclrtFree(outDeviceAddr);
    return false;
  }

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) { return false; }
  }

  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) { return false; }

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) { return false; }

  std::vector<TOut> resultData(outHostData.size());
  ret = CopyFromDevice(outDeviceAddr, resultData, resultData.size());
  if (ret != 0) return false;

  bool pass = VerifyResult(resultData, expectedData, outDtype, testName);

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr); aclrtFree(otherDeviceAddr); aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) { aclrtFree(workspaceAddr); }

  return pass;
}

// 运行混合数据类型InplaceMul测试用例
template <typename TSelf, typename TOther>
bool RunInplaceMulMixDtypeTest(const std::vector<TSelf>& selfData, const std::vector<int64_t>& selfShape,
                               const std::vector<TOther>& otherData, const std::vector<int64_t>& otherShape,
                               const std::vector<TSelf>& expectedData, aclDataType selfDtype, aclDataType otherDtype,
                               aclrtStream stream, const char* testName) {
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;

  std::vector<TSelf> selfHostData = selfData;

  int ret = CreateAclTensor(selfData, selfShape, &selfDeviceAddr, selfDtype, &self);
  if (ret != 0) return false;
  ret = CreateAclTensor(otherData, otherShape, &otherDeviceAddr, otherDtype, &other);
  if (ret != 0) return false;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnInplaceMulGetWorkspaceSize(self, other, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclnnInplaceMulGetWorkspaceSize failed. ERROR: %d\n", ret);
    aclDestroyTensor(self); aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr); aclrtFree(otherDeviceAddr);
    return false;
  }

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) { return false; }
  }

  ret = aclnnInplaceMul(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) { return false; }

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) { return false; }

  std::vector<TSelf> resultData(selfHostData.size());
  ret = CopyFromDevice(selfDeviceAddr, resultData, resultData.size());
  if (ret != 0) return false;

  bool pass = VerifyResult(resultData, expectedData, selfDtype, testName);

  aclDestroyTensor(self); aclDestroyTensor(other);
  aclrtFree(selfDeviceAddr); aclrtFree(otherDeviceAddr);
  if (workspaceSize > 0) { aclrtFree(workspaceAddr); }

  return pass;
}

// 运行Muls测试用例（tensor * scalar）
template <typename T>
bool RunMulsTest(const std::vector<T>& selfData, const std::vector<int64_t>& selfShape,
                 T scalar, aclDataType dtype, aclrtStream stream, const char* testName) {
  // 创建self tensor
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  
  std::vector<T> outHostData(GetShapeSize(selfShape), 0);
  
  int ret = CreateAclTensor(selfData, selfShape, &selfDeviceAddr, dtype, &self);
  if (ret != 0) return false;
  ret = CreateAclTensor(outHostData, selfShape, &outDeviceAddr, dtype, &out);
  if (ret != 0) return false;
  
  // 调用aclnnMuls
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 创建aclScalar
  aclScalar* scalarObj = aclCreateScalar((void*)&scalar, dtype);
  CHECK_RET(scalarObj != nullptr, LOG_PRINT("aclCreateScalar failed.\n"); return false);
  ret = aclnnMulsGetWorkspaceSize(self, scalarObj, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclnnMulsGetWorkspaceSize failed. ERROR: %d\n", ret);
    return false;
  }
  
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
      return false;
    }
  }
  
  ret = aclnnMuls(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclnnMuls failed. ERROR: %d\n", ret);
    return false;
  }
  
  // 同步
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    return false;
  }
  
  // 获取输出
  std::vector<T> resultData(outHostData.size());
  ret = CopyFromDevice(outDeviceAddr, resultData, resultData.size());
  if (ret != 0) return false;
  
  // 计算期望值
  std::vector<T> scalarVec = {scalar};
  std::vector<int64_t> scalarShape = {1};
  std::vector<T> expected = ComputeExpectedMul(selfData, selfShape, scalarVec, scalarShape);
  
  // 验证
  bool pass = VerifyResult(resultData, expected, dtype, testName);
  
  // 清理
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyScalar(scalarObj);
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  
  return pass;
}

// 运行InplaceMul测试用例
template <typename T>
bool RunInplaceMulTest(const std::vector<T>& selfData, const std::vector<int64_t>& selfShape,
                       const std::vector<T>& otherData, const std::vector<int64_t>& otherShape,
                       aclDataType dtype, aclrtStream stream, const char* testName) {
  // 创建self tensor（将被修改）
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  
  // 复制self数据到device
  std::vector<T> selfHostData = selfData; // 备份用于计算期望值
  
  int ret = CreateAclTensor(selfData, selfShape, &selfDeviceAddr, dtype, &self);
  if (ret != 0) return false;
  ret = CreateAclTensor(otherData, otherShape, &otherDeviceAddr, dtype, &other);
  if (ret != 0) return false;
  
  // 调用aclnnInplaceMul
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnInplaceMulGetWorkspaceSize(self, other, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclnnInplaceMulGetWorkspaceSize failed. ERROR: %d\n", ret);
    return false;
  }
  
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
      return false;
    }
  }
  
  ret = aclnnInplaceMul(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclnnInplaceMul failed. ERROR: %d\n", ret);
    return false;
  }
  
  // 同步
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    return false;
  }
  
  // 获取修改后的self
  std::vector<T> resultData(selfHostData.size());
  ret = CopyFromDevice(selfDeviceAddr, resultData, resultData.size());
  if (ret != 0) return false;
  
  // 计算期望值
  std::vector<T> expected = ComputeExpectedMul(selfHostData, selfShape, otherData, otherShape);
  
  // 验证
  bool pass = VerifyResult(resultData, expected, dtype, testName);
  
  // 清理
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  
  return pass;
}

// 运行InplaceMuls测试用例
template <typename T>
bool RunInplaceMulsTest(const std::vector<T>& selfData, const std::vector<int64_t>& selfShape,
                        T scalar, aclDataType dtype, aclrtStream stream, const char* testName) {
  // 创建self tensor（将被修改）
  void* selfDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  
  // 复制self数据到device
  std::vector<T> selfHostData = selfData; // 备份用于计算期望值
  
  int ret = CreateAclTensor(selfData, selfShape, &selfDeviceAddr, dtype, &self);
  if (ret != 0) return false;
  
  // 调用aclnnInplaceMuls
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 创建aclScalar
  aclScalar* scalarObj = aclCreateScalar((void*)&scalar, dtype);
  CHECK_RET(scalarObj != nullptr, LOG_PRINT("aclCreateScalar failed.\n"); return false);
  ret = aclnnInplaceMulsGetWorkspaceSize(self, scalarObj, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclnnInplaceMulsGetWorkspaceSize failed. ERROR: %d\n", ret);
    return false;
  }
  
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
      return false;
    }
  }
  
  ret = aclnnInplaceMuls(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclnnInplaceMuls failed. ERROR: %d\n", ret);
    return false;
  }
  
  // 同步
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    return false;
  }
  
  // 获取修改后的self
  std::vector<T> resultData(selfHostData.size());
  ret = CopyFromDevice(selfDeviceAddr, resultData, resultData.size());
  if (ret != 0) return false;
  
  // 计算期望值
  std::vector<T> scalarVec = {scalar};
  std::vector<int64_t> scalarShape = {1};
  std::vector<T> expected = ComputeExpectedMul(selfHostData, selfShape, scalarVec, scalarShape);
  
  // 验证
  bool pass = VerifyResult(resultData, expected, dtype, testName);
  
  // 清理
  aclDestroyScalar(scalarObj);
  aclDestroyTensor(self);
  aclrtFree(selfDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  
  return pass;
}

int main() {
  // 初始化
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  
  LOG_PRINT("=== Starting Mul operator tests ===\n");
  
  // 测试用例1: 基础FLOAT32 Mul
  {
    std::vector<float> selfData = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    std::vector<float> otherData = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f};
    std::vector<int64_t> shape = {4, 2};
    RunMulTest(selfData, shape, otherData, shape, ACL_FLOAT, stream, "Basic FLOAT32 Mul");
  }
  
  // 测试用例2: 广播 Mul (vector * scalar broadcast)
  {
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> otherData = {2.0f}; // 标量
    std::vector<int64_t> selfShape = {4};
    std::vector<int64_t> otherShape = {1};
    RunMulTest(selfData, selfShape, otherData, otherShape, ACL_FLOAT, stream, "Broadcast Mul (vector * scalar)");
  }
  
  // 测试用例3: Muls API (tensor * scalar)
  {
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> shape = {4};
    float scalar = 3.0f;
    RunMulsTest(selfData, shape, scalar, ACL_FLOAT, stream, "Muls API (tensor * scalar)");
  }
  
  // 测试用例4: InplaceMul API
  {
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> otherData = {0.5f, 0.5f, 0.5f, 0.5f};
    std::vector<int64_t> shape = {4};
    RunInplaceMulTest(selfData, shape, otherData, shape, ACL_FLOAT, stream, "InplaceMul API");
  }
  
  // 测试用例5: InplaceMuls API
  {
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> shape = {4};
    float scalar = 2.0f;
    RunInplaceMulsTest(selfData, shape, scalar, ACL_FLOAT, stream, "InplaceMuls API");
  }
  
  // 测试用例6: 零值测试
  {
    std::vector<float> selfData = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> otherData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> shape = {4};
    RunMulTest(selfData, shape, otherData, shape, ACL_FLOAT, stream, "Zero values");
  }
  
  // 测试用例7: 负数测试
  {
    std::vector<float> selfData = {-1.0f, -2.0f, -3.0f, -4.0f};
    std::vector<float> otherData = {2.0f, -2.0f, 0.5f, -0.5f};
    std::vector<int64_t> shape = {4};
    RunMulTest(selfData, shape, otherData, shape, ACL_FLOAT, stream, "Negative values");
  }
  
  // 测试用例8: INT32 数据类型
  {
    std::vector<int32_t> selfData = {1, 2, 3, 4};
    std::vector<int32_t> otherData = {2, 3, 4, 5};
    std::vector<int64_t> shape = {4};
    RunMulTest(selfData, shape, otherData, shape, ACL_INT32, stream, "INT32 data type");
  }
  
  // 测试用例9: 大shape测试
  {
    std::vector<float> selfData(100);
    std::vector<float> otherData(100);
    for (int i = 0; i < 100; i++) {
      selfData[i] = static_cast<float>(i);
      otherData[i] = 0.5f;
    }
    std::vector<int64_t> shape = {100};
    RunMulTest(selfData, shape, otherData, shape, ACL_FLOAT, stream, "Large shape (100 elements)");
  }
  
  // 测试用例10: 特殊值 (Inf, NaN) - 需要检查算子是否支持
  {
    std::vector<float> selfData = {std::numeric_limits<float>::infinity(), 
                                   -std::numeric_limits<float>::infinity(),
                                   std::numeric_limits<float>::quiet_NaN(),
                                   1.0f};
    std::vector<float> otherData = {2.0f, 2.0f, 2.0f, 0.0f};
    std::vector<int64_t> shape = {4};
    RunMulTest(selfData, shape, otherData, shape, ACL_FLOAT, stream, "Special values (Inf, NaN)");
  }
  
  // 测试用例11: 2D广播 (matrix * vector)
  {
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> otherData = {2.0f, 3.0f};
    std::vector<int64_t> selfShape = {3, 2};
    std::vector<int64_t> otherShape = {2};
    RunMulTest(selfData, selfShape, otherData, otherShape, ACL_FLOAT, stream, "2D broadcast (matrix * vector)");
  }
  
  // 测试用例12: 标量 * 标量
  {
    std::vector<float> selfData = {3.0f};
    std::vector<float> otherData = {4.0f};
    std::vector<int64_t> shape = {1};
    RunMulTest(selfData, shape, otherData, shape, ACL_FLOAT, stream, "Scalar * scalar");
  }
  
  // 测试用例13: FLOAT16 数据类型
  {
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> otherData = {0.5f, 1.0f, 1.5f, 2.0f};
    std::vector<int64_t> shape = {4};
    RunMulTest(selfData, shape, otherData, shape, ACL_FLOAT16, stream, "FLOAT16 data type");
  }
  
  // 测试用例14: BF16 数据类型
  {
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> otherData = {0.5f, 1.0f, 1.5f, 2.0f};
    std::vector<int64_t> shape = {4};
    RunMulTest(selfData, shape, otherData, shape, ACL_BF16, stream, "BF16 data type");
  }
  
  // 测试用例15: INT8 数据类型
  {
    std::vector<int8_t> selfData = {1, 2, 3, 4};
    std::vector<int8_t> otherData = {2, 3, 4, 5};
    std::vector<int64_t> shape = {4};
    RunMulTest(selfData, shape, otherData, shape, ACL_INT8, stream, "INT8 data type");
  }
  
  // 测试用例16: UINT8 数据类型
  {
    std::vector<uint8_t> selfData = {1, 2, 3, 4};
    std::vector<uint8_t> otherData = {2, 3, 4, 5};
    std::vector<int64_t> shape = {4};
    RunMulTest(selfData, shape, otherData, shape, ACL_UINT8, stream, "UINT8 data type");
  }
  
  // 测试用例17: INT64 数据类型
  {
    std::vector<int64_t> selfData = {1000LL, 2000LL, 3000LL, 4000LL};
    std::vector<int64_t> otherData = {2LL, 3LL, 4LL, 5LL};
    std::vector<int64_t> shape = {4};
    RunMulTest(selfData, shape, otherData, shape, ACL_INT64, stream, "INT64 data type");
  }
  
  // 测试用例18: 3D tensor
  {
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> otherData = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    std::vector<int64_t> shape = {2, 2, 2};
    RunMulTest(selfData, shape, otherData, shape, ACL_FLOAT, stream, "3D tensor");
  }
  
  // 测试用例19: 4D tensor (NCHW format)
  {
    std::vector<float> selfData(24, 1.0f);
    std::vector<float> otherData(24, 2.0f);
    std::vector<int64_t> shape = {2, 3, 2, 2}; // N=2, C=3, H=2, W=2
    RunMulTest(selfData, shape, otherData, shape, ACL_FLOAT, stream, "4D tensor (NCHW)");
  }
  
  // 测试用例20: 不同shape广播 [3,1] * [1,4] -> [3,4]
  {
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f};
    std::vector<float> otherData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> selfShape = {3, 1};
    std::vector<int64_t> otherShape = {1, 4};
    RunMulTest(selfData, selfShape, otherData, otherShape, ACL_FLOAT, stream, "Broadcast [3,1] * [1,4]");
  }
  
  // 测试用例21: Muls with FLOAT16
  {
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> shape = {4};
    float scalar = 0.5f;
    RunMulsTest(selfData, shape, scalar, ACL_FLOAT16, stream, "Muls with FLOAT16");
  }
  
  // 测试用例22: Muls with INT32
  {
    std::vector<int32_t> selfData = {10, 20, 30, 40};
    std::vector<int64_t> shape = {4};
    int32_t scalar = 3;
    RunMulsTest(selfData, shape, scalar, ACL_INT32, stream, "Muls with INT32");
  }
  
  // 测试用例23: InplaceMul with different dtypes
  {
    std::vector<float> selfData = {2.0f, 4.0f, 6.0f, 8.0f};
    std::vector<float> otherData = {0.5f, 0.25f, 0.5f, 0.125f};
    std::vector<int64_t> shape = {4};
    RunInplaceMulTest(selfData, shape, otherData, shape, ACL_FLOAT, stream, "InplaceMul with fractions");
  }
  
  // 测试用例24: InplaceMuls with negative scalar
  {
    std::vector<float> selfData = {1.0f, -2.0f, 3.0f, -4.0f};
    std::vector<int64_t> shape = {4};
    float scalar = -2.0f;
    RunInplaceMulsTest(selfData, shape, scalar, ACL_FLOAT, stream, "InplaceMuls with negative scalar");
  }
  
  // 测试用例25: 极大值测试
  {
    std::vector<float> selfData = {1e30f, 1e20f, 1e10f, 1.0f};
    std::vector<float> otherData = {1e10f, 1e20f, 1e30f, 1.0f};
    std::vector<int64_t> shape = {4};
    RunMulTest(selfData, shape, otherData, shape, ACL_FLOAT, stream, "Large values");
  }
  
  // 测试用例26: 极小值测试
  {
    std::vector<float> selfData = {1e-30f, 1e-20f, 1e-10f, 1.0f};
    std::vector<float> otherData = {1e-10f, 1e-20f, 1e-30f, 1.0f};
    std::vector<int64_t> shape = {4};
    RunMulTest(selfData, shape, otherData, shape, ACL_FLOAT, stream, "Small values");
  }
  
  // 测试用例27: 混合正负值
  {
    std::vector<float> selfData = {-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f};
    std::vector<float> otherData = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
    std::vector<int64_t> shape = {6};
    RunMulTest(selfData, shape, otherData, shape, ACL_FLOAT, stream, "Mixed positive/negative");
  }
  
  // 测试用例28: 全1值
  {
    std::vector<float> selfData(8, 1.0f);
    std::vector<float> otherData(8, 1.0f);
    std::vector<int64_t> shape = {8};
    RunMulTest(selfData, shape, otherData, shape, ACL_FLOAT, stream, "All ones");
  }
  
  // 测试用例29: 全0值
  {
    std::vector<float> selfData(8, 0.0f);
    std::vector<float> otherData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<int64_t> shape = {8};
    RunMulTest(selfData, shape, otherData, shape, ACL_FLOAT, stream, "All zeros");
  }
  
  // 测试用例30: 单元素tensor
  {
    std::vector<float> selfData = {5.0f};
    std::vector<float> otherData = {3.0f};
    std::vector<int64_t> shape = {1};
    RunMulTest(selfData, shape, otherData, shape, ACL_FLOAT, stream, "Single element tensor");
  }
  
  // 测试用例31: 非连续stride测试 (通过reshape模拟)
  {
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> otherData = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    std::vector<int64_t> shape = {6};
    RunMulTest(selfData, shape, otherData, shape, ACL_FLOAT, stream, "Contiguous stride");
  }
  
  // 测试用例32: Muls with zero scalar
  {
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> shape = {4};
    float scalar = 0.0f;
    RunMulsTest(selfData, shape, scalar, ACL_FLOAT, stream, "Muls with zero scalar");
  }
  
  // 测试用例33: InplaceMuls with one scalar
  {
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> shape = {4};
    float scalar = 1.0f;
    RunInplaceMulsTest(selfData, shape, scalar, ACL_FLOAT, stream, "InplaceMuls with one scalar");
  }
  
  // 测试用例34: INT16 数据类型
  {
    std::vector<int16_t> selfData = {100, 200, 300, 400};
    std::vector<int16_t> otherData = {2, 3, 4, 5};
    std::vector<int64_t> shape = {4};
    RunMulTest(selfData, shape, otherData, shape, ACL_INT16, stream, "INT16 data type");
  }
  
  // 测试用例35: UINT16 数据类型
  {
    std::vector<uint16_t> selfData = {100, 200, 300, 400};
    std::vector<uint16_t> otherData = {2, 3, 4, 5};
    std::vector<int64_t> shape = {4};
    RunMulTest(selfData, shape, otherData, shape, ACL_UINT16, stream, "UINT16 data type");
  }
  
  // 测试用例36: 更大的tensor (512 elements)
  {
    std::vector<float> selfData(512);
    std::vector<float> otherData(512);
    for (int i = 0; i < 512; i++) {
      selfData[i] = static_cast<float>(i % 10);
      otherData[i] = 0.1f * static_cast<float>(i % 5);
    }
    std::vector<int64_t> shape = {512};
    RunMulTest(selfData, shape, otherData, shape, ACL_FLOAT, stream, "Large tensor (512 elements)");
  }
  
  // 测试用例37: 2D tensor with different shapes
  {
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> otherData = {2.0f, 3.0f};
    std::vector<int64_t> selfShape = {3, 2};
    std::vector<int64_t> otherShape = {1, 2};
    RunMulTest(selfData, selfShape, otherData, otherShape, ACL_FLOAT, stream, "2D broadcast [3,2] * [1,2]");
  }
  
  // 测试用例38: InplaceMul with INT32
  {
    std::vector<int32_t> selfData = {10, 20, 30, 40};
    std::vector<int32_t> otherData = {2, 3, 4, 5};
    std::vector<int64_t> shape = {4};
    RunInplaceMulTest(selfData, shape, otherData, shape, ACL_INT32, stream, "InplaceMul with INT32");
  }
  
  // 测试用例39: Muls with large scalar
  {
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> shape = {4};
    float scalar = 1e10f;
    RunMulsTest(selfData, shape, scalar, ACL_FLOAT, stream, "Muls with large scalar");
  }
  
  // 测试用例40: InplaceMuls with small scalar
  {
    std::vector<float> selfData = {1e10f, 2e10f, 3e10f, 4e10f};
    std::vector<int64_t> shape = {4};
    float scalar = 1e-10f;
    RunInplaceMulsTest(selfData, shape, scalar, ACL_FLOAT, stream, "InplaceMuls with small scalar");
  }
  

  // 测试用例41: 混合类型 FLOAT16 * FLOAT -> FLOAT (触发 IsMulMixDtypeSupport)
  {
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> otherData = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> expectedData = {2.0f, 6.0f, 12.0f, 20.0f};
    std::vector<int64_t> shape = {4};
    RunMulMixDtypeTest(selfData, shape, otherData, shape, expectedData,
                       ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT, stream, "Mix dtype FP16*FP32->FP32");
  }

  // 测试用例42: 混合类型 FLOAT * FLOAT16 -> FLOAT
  {
    std::vector<float> selfData = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> otherData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> expectedData = {2.0f, 6.0f, 12.0f, 20.0f};
    std::vector<int64_t> shape = {4};
    RunMulMixDtypeTest(selfData, shape, otherData, shape, expectedData,
                       ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT, stream, "Mix dtype FP32*FP16->FP32");
  }

  // 测试用例43: 混合类型 BF16 * FLOAT -> FLOAT
  {
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> otherData = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> expectedData = {2.0f, 6.0f, 12.0f, 20.0f};
    std::vector<int64_t> shape = {4};
    RunMulMixDtypeTest(selfData, shape, otherData, shape, expectedData,
                       ACL_BF16, ACL_FLOAT, ACL_FLOAT, stream, "Mix dtype BF16*FP32->FP32");
  }

  // 测试用例44: 混合类型 FLOAT * BF16 -> FLOAT
  {
    std::vector<float> selfData = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> otherData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> expectedData = {2.0f, 6.0f, 12.0f, 20.0f};
    std::vector<int64_t> shape = {4};
    RunMulMixDtypeTest(selfData, shape, otherData, shape, expectedData,
                       ACL_FLOAT, ACL_BF16, ACL_FLOAT, stream, "Mix dtype FP32*BF16->FP32");
  }

  // 测试用例45: BOOL 数据类型
  {
    std::vector<int8_t> selfData = {0, 1, 1, 0};
    std::vector<int8_t> otherData = {1, 1, 0, 0};
    std::vector<int64_t> shape = {4};
    RunMulTest(selfData, shape, otherData, shape, ACL_BOOL, stream, "BOOL data type");
  }

  // 测试用例46: DOUBLE 数据类型
  {
    std::vector<double> selfData = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> otherData = {2.0, 3.0, 4.0, 5.0};
    std::vector<int64_t> shape = {4};
    RunMulTest(selfData, shape, otherData, shape, ACL_DOUBLE, stream, "DOUBLE data type");
  }

  // 测试用例47: 混合类型 InplaceMul FLOAT16 * FLOAT
  {
    std::vector<float> selfData = {2.0f, 4.0f, 6.0f, 8.0f};
    std::vector<float> otherData = {0.5f, 0.25f, 0.5f, 0.125f};
    std::vector<float> expectedData = {1.0f, 1.0f, 3.0f, 1.0f};
    std::vector<int64_t> shape = {4};
    RunInplaceMulMixDtypeTest(selfData, shape, otherData, shape, expectedData,
                              ACL_FLOAT16, ACL_FLOAT, stream, "InplaceMul mix FP16*FP32");
  }

  // 测试用例48: 混合类型 InplaceMul BF16 * FLOAT
  {
    std::vector<float> selfData = {2.0f, 4.0f, 6.0f, 8.0f};
    std::vector<float> otherData = {0.5f, 0.25f, 0.5f, 0.125f};
    std::vector<float> expectedData = {1.0f, 1.0f, 3.0f, 1.0f};
    std::vector<int64_t> shape = {4};
    RunInplaceMulMixDtypeTest(selfData, shape, otherData, shape, expectedData,
                              ACL_BF16, ACL_FLOAT, stream, "InplaceMul mix BF16*FP32");
  }

  // 测试用例49: Muls with BF16 (触发 canUseMuls 路径)
  {
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> shape = {4};
    float scalar = 2.0f;
    RunMulsTest(selfData, shape, scalar, ACL_BF16, stream, "Muls with BF16 scalar");
  }

  // 测试用例50: InplaceMuls with BF16
  {
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> shape = {4};
    float scalar = 0.5f;
    RunInplaceMulsTest(selfData, shape, scalar, ACL_BF16, stream, "InplaceMuls with BF16 scalar");
  }

  // 测试用例51: Muls with DOUBLE
  {
    std::vector<double> selfData = {1.0, 2.0, 3.0, 4.0};
    std::vector<int64_t> shape = {4};
    double scalar = 2.0;
    RunMulsTest(selfData, shape, scalar, ACL_DOUBLE, stream, "Muls with DOUBLE");
  }

  // 测试用例52: InplaceMuls with DOUBLE
  {
    std::vector<double> selfData = {1.0, 2.0, 3.0, 4.0};
    std::vector<int64_t> shape = {4};
    double scalar = 0.5;
    RunInplaceMulsTest(selfData, shape, scalar, ACL_DOUBLE, stream, "InplaceMuls with DOUBLE");
  }

  // 测试用例53: InplaceMul with DOUBLE
  {
    std::vector<double> selfData = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> otherData = {2.0, 3.0, 4.0, 5.0};
    std::vector<int64_t> shape = {4};
    RunInplaceMulTest(selfData, shape, otherData, shape, ACL_DOUBLE, stream, "InplaceMul with DOUBLE");
  }

  // 测试用例54: InplaceMul with FLOAT16
  {
    std::vector<float> selfData = {2.0f, 4.0f, 6.0f, 8.0f};
    std::vector<float> otherData = {0.5f, 0.25f, 0.5f, 0.125f};
    std::vector<int64_t> shape = {4};
    RunInplaceMulTest(selfData, shape, otherData, shape, ACL_FLOAT16, stream, "InplaceMul with FLOAT16");
  }

  // 测试用例55: InplaceMul with BF16
  {
    std::vector<float> selfData = {2.0f, 4.0f, 6.0f, 8.0f};
    std::vector<float> otherData = {0.5f, 0.25f, 0.5f, 0.125f};
    std::vector<int64_t> shape = {4};
    RunInplaceMulTest(selfData, shape, otherData, shape, ACL_BF16, stream, "InplaceMul with BF16");
  }

  // 测试用例56: Muls with FLOAT16 scalar (触发 canUseMuls 路径)
  {
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> shape = {4};
    float scalar = 3.0f;
    RunMulsTest(selfData, shape, scalar, ACL_FLOAT16, stream, "Muls with FLOAT16 scalar (canUseMuls)");
  }

  // 测试用例57: InplaceMuls with FLOAT16
  {
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> shape = {4};
    float scalar = 2.0f;
    RunInplaceMulsTest(selfData, shape, scalar, ACL_FLOAT16, stream, "InplaceMuls with FLOAT16");
  }

  // 测试用例58: 混合类型广播 FLOAT16 * FLOAT with broadcast
  {
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> otherData = {2.0f, 3.0f};
    std::vector<float> expectedData = {2.0f, 6.0f, 6.0f, 12.0f, 10.0f, 18.0f};
    std::vector<int64_t> selfShape = {3, 2};
    std::vector<int64_t> otherShape = {1, 2};
    RunMulMixDtypeTest(selfData, selfShape, otherData, otherShape, expectedData,
                       ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT, stream, "Mix dtype broadcast FP16*FP32");
  }

  // 测试用例59: 混合类型广播 BF16 * FLOAT with broadcast
  {
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> otherData = {2.0f, 3.0f};
    std::vector<float> expectedData = {2.0f, 6.0f, 6.0f, 12.0f, 10.0f, 18.0f};
    std::vector<int64_t> selfShape = {3, 2};
    std::vector<int64_t> otherShape = {1, 2};
    RunMulMixDtypeTest(selfData, selfShape, otherData, otherShape, expectedData,
                       ACL_BF16, ACL_FLOAT, ACL_FLOAT, stream, "Mix dtype broadcast BF16*FP32");
  }

  // 测试用例60: BOOL with various combinations
  {
    std::vector<int8_t> selfData = {1, 1, 0, 0, 1, 0};
    std::vector<int8_t> otherData = {1, 0, 1, 0, 1, 1};
    std::vector<int64_t> shape = {6};
    RunMulTest(selfData, shape, otherData, shape, ACL_BOOL, stream, "BOOL various combinations");
  }

  // 测试用例61: DOUBLE with special values
  {
    std::vector<double> selfData = {1.0, -2.0, 0.0, 1e100};
    std::vector<double> otherData = {2.0, 3.0, 1.0, 1e-100};
    std::vector<int64_t> shape = {4};
    RunMulTest(selfData, shape, otherData, shape, ACL_DOUBLE, stream, "DOUBLE special values");
  }

  // 测试用例62: InplaceMuls with INT32
  {
    std::vector<int32_t> selfData = {10, 20, 30, 40};
    std::vector<int64_t> shape = {4};
    int32_t scalar = 3;
    RunInplaceMulsTest(selfData, shape, scalar, ACL_INT32, stream, "InplaceMuls with INT32");
  }

  // 测试用例63: InplaceMuls with INT64
  {
    std::vector<int64_t> selfData = {100LL, 200LL, 300LL, 400LL};
    std::vector<int64_t> shape = {4};
    int64_t scalar = 2LL;
    RunInplaceMulsTest(selfData, shape, scalar, ACL_INT64, stream, "InplaceMuls with INT64");
  }

  // 测试用例64: Muls with INT64
  {
    std::vector<int64_t> selfData = {100LL, 200LL, 300LL, 400LL};
    std::vector<int64_t> shape = {4};
    int64_t scalar = 3LL;
    RunMulsTest(selfData, shape, scalar, ACL_INT64, stream, "Muls with INT64");
  }

  // 测试用例65: Mul with INT8 and negative values
  {
    std::vector<int8_t> selfData = {-1, 2, -3, 4};
    std::vector<int8_t> otherData = {2, -3, 4, -5};
    std::vector<int64_t> shape = {4};
    RunMulTest(selfData, shape, otherData, shape, ACL_INT8, stream, "INT8 negative values");
  }

  // 测试用例66: Mul with UINT8 overflow
  {
    std::vector<uint8_t> selfData = {200, 100, 255, 0};
    std::vector<uint8_t> otherData = {2, 3, 1, 255};
    std::vector<int64_t> shape = {4};
    RunMulTest(selfData, shape, otherData, shape, ACL_UINT8, stream, "UINT8 overflow values");
  }

  // 测试用例67: 5D tensor
  {
    std::vector<float> selfData(16, 1.0f);
    std::vector<float> otherData(16, 2.0f);
    std::vector<int64_t> shape = {2, 2, 2, 2, 1};
    RunMulTest(selfData, shape, otherData, shape, ACL_FLOAT, stream, "5D tensor");
  }

  // 测试用例68: 6D tensor
  {
    std::vector<float> selfData(16, 1.0f);
    std::vector<float> otherData(16, 2.0f);
    std::vector<int64_t> shape = {2, 2, 2, 2, 1, 1};
    RunMulTest(selfData, shape, otherData, shape, ACL_FLOAT, stream, "6D tensor");
  }

  // 测试用例69: 7D tensor
  {
    std::vector<float> selfData(16, 1.0f);
    std::vector<float> otherData(16, 2.0f);
    std::vector<int64_t> shape = {2, 2, 2, 1, 1, 1, 1};
    RunMulTest(selfData, shape, otherData, shape, ACL_FLOAT, stream, "7D tensor");
  }

  // 测试用例70: 8D tensor (MAX_SUPPORT_DIMS_NUMS)
  {
    std::vector<float> selfData(16, 1.0f);
    std::vector<float> otherData(16, 2.0f);
    std::vector<int64_t> shape = {2, 2, 1, 1, 1, 1, 1, 1};
    RunMulTest(selfData, shape, otherData, shape, ACL_FLOAT, stream, "8D tensor (max dims)");
  }

  // 总结
  LOG_PRINT("\n=== Test Summary ===\n");
  LOG_PRINT("Total tests: %d\n", g_total_tests);
  LOG_PRINT("Passed: %d\n", g_passed_tests);
  LOG_PRINT("Failed: %d\n", g_failed_tests);
  
  // 清理
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  
  return g_failed_tests > 0 ? 1 : 0;
}