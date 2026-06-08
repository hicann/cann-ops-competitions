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
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>
#include <complex>
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

// 计算 tensor 的 shape 大小
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

bool GetBroadcastShape(const std::vector<int64_t>& selfShape,
                       const std::vector<int64_t>& otherShape,
                       std::vector<int64_t>& outShape) {
  const size_t outRank = std::max(selfShape.size(), otherShape.size());
  outShape.assign(outRank, 1);
  for (size_t i = 0; i < outRank; ++i) {
    const int64_t selfDim =
        (i < outRank - selfShape.size()) ? 1 : selfShape[i - (outRank - selfShape.size())];
    const int64_t otherDim =
        (i < outRank - otherShape.size()) ? 1 : otherShape[i - (outRank - otherShape.size())];
    if (selfDim != otherDim && selfDim != 1 && otherDim != 1) {
      return false;
    }
    outShape[i] = std::max(selfDim, otherDim);
  }
  return true;
}

std::vector<int64_t> GetLinearToCoords(size_t linearIndex, const std::vector<int64_t>& shape) {
  std::vector<int64_t> coords(shape.size(), 0);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    if (shape[i] == 0) {
      return coords;
    }
    coords[i] = linearIndex % shape[i];
    linearIndex /= shape[i];
  }
  return coords;
}

size_t GetBroadcastedOffset(const std::vector<int64_t>& inShape, const std::vector<int64_t>& outCoords) {
  if (inShape.empty()) {
    return 0;
  }
  size_t offset = 0;
  size_t stride = 1;
  const size_t rankGap = outCoords.size() - inShape.size();
  for (int64_t i = static_cast<int64_t>(inShape.size()) - 1; i >= 0; --i) {
    const int64_t coord = (inShape[i] == 1) ? 0 : outCoords[rankGap + i];
    offset += static_cast<size_t>(coord) * stride;
    stride *= static_cast<size_t>(inShape[i]);
  }
  return offset;
}

// 初始化运行环境：ACL、Device、Stream
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

// 通用近似比较
template <typename T>
struct IsStdComplex : std::false_type {};

template <typename T>
struct IsStdComplex<std::complex<T>> : std::true_type {};

template <typename T>
bool AlmostEqual(T actual, T expected, double atol = 1e-5f, double rtol = 1e-5f) {
  if constexpr (std::is_same_v<T, bool> || std::is_integral_v<T>) {
    return actual == expected;
  } else if constexpr (IsStdComplex<T>::value) {
    using ValueType = typename T::value_type;
    const bool realBothNaN = std::isnan(expected.real()) && std::isnan(actual.real());
    const bool imagBothNaN = std::isnan(expected.imag()) && std::isnan(actual.imag());
    const bool realBothInf = std::isinf(expected.real()) && std::isinf(actual.real()) &&
                             ((expected.real() > static_cast<ValueType>(0)) ==
                              (actual.real() > static_cast<ValueType>(0)));
    const bool imagBothInf = std::isinf(expected.imag()) && std::isinf(actual.imag()) &&
                             ((expected.imag() > static_cast<ValueType>(0)) ==
                              (actual.imag() > static_cast<ValueType>(0)));
    const bool realClose = realBothNaN || realBothInf ||
                           std::abs(actual.real() - expected.real()) <=
                               (atol + rtol * std::abs(expected.real()));
    const bool imagClose = imagBothNaN || imagBothInf ||
                           std::abs(actual.imag() - expected.imag()) <=
                               (atol + rtol * std::abs(expected.imag()));
    return realClose && imagClose;
  } else {
    if (std::isnan(expected) && std::isnan(actual))
      return true;
    if (std::isinf(expected) && std::isinf(actual))
      return (expected > 0) == (actual > 0);
    return std::abs(actual - expected) <= (atol + rtol * std::abs(expected));
  }
}

// 1. 定义处理标量的辅助函数
template <typename T>
void LogMismatch(size_t i, const T& res, const T& exp) {
    LOG_PRINT("mismatch at index %zu, result=%f expected=%f\n", 
              i, (double)res, (double)exp);
}

// 2. 定义处理复数的辅助函数
template <typename T>
void LogMismatch(size_t i, const std::complex<T>& res, const std::complex<T>& exp) {
    LOG_PRINT("mismatch at index %zu, result=(%f, %f) expected=(%f, %f)\n", 
              i, (double)res.real(), (double)res.imag(),
              (double)exp.real(), (double)exp.imag());
}

// --- 核心校验函数修改 ---
template <typename TOut, typename TExp>
bool CheckResultTyped(const std::vector<TOut>& resultData, const std::vector<TExp>& expectedData) {
  if (resultData.size() != expectedData.size()) {
    LOG_PRINT("result size mismatch, result=%zu expected=%zu\n", resultData.size(), expectedData.size());
    return false;
  }
  for (size_t i = 0; i < resultData.size(); ++i) {
    if (!AlmostEqual(resultData[i], expectedData[i])) {
        LogMismatch(i, resultData[i], expectedData[i]); // 编译器根据类型自动选择匹配的函数
        return false;
    }
  }
  return true;
}

// float 结果校验
bool CheckResult(const std::vector<float>& resultData, const std::vector<float>& expectedData) {
  if (resultData.size() != expectedData.size()) {
    LOG_PRINT("result size mismatch, result=%zu expected=%zu\n", resultData.size(), expectedData.size());
    return false;
  }
  for (size_t i = 0; i < resultData.size(); ++i) {
    if (!AlmostEqual(resultData[i], expectedData[i])) {
      LOG_PRINT("mismatch at index %zu, result=%f expected=%f\n", i, resultData[i], expectedData[i]);
      return false;
    }
  }
  return true;
}

template <typename T>
// 在 device 上申请内存并创建 aclTensor
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

// Mul 基础测试流程：执行 -> 回拷 -> 计算期望 -> 校验 -> 输出 PASS/FAIL
bool RunMulTest(const char* testName,
                aclrtStream stream,
                const std::vector<float>& selfHostData,
                const std::vector<float>& otherHostData,
                const std::vector<int64_t>& shape) {
  if (selfHostData.size() != otherHostData.size()) {
    LOG_PRINT("[%s] FAIL: input size mismatch\n", testName);
    return false;
  }

  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> outHostData(selfHostData.size(), 0.0f);
  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out tensor failed. ERROR: %d\n", testName, ret); return false);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclnnMulGetWorkspaceSize failed. ERROR: %d\n", testName, ret); return false);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[%s] FAIL: allocate workspace failed. ERROR: %d\n", testName, ret); return false);
  }

  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclnnMul failed. ERROR: %d\n", testName, ret); return false);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret); return false);

  std::vector<float> resultData(selfHostData.size(), 0.0f);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: copy result from device to host failed. ERROR: %d\n", testName, ret); return false);

  std::vector<float> expectedData(selfHostData.size(), 0.0f);
  for (size_t i = 0; i < selfHostData.size(); ++i) {
    expectedData[i] = selfHostData[i] * otherHostData[i];
  }

  bool pass = CheckResult(resultData, expectedData);
  LOG_PRINT("[%s] %s\n", testName, pass ? "PASS" : "FAIL");

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

template <typename TSelf, typename TOther, typename TOut>
// Mul 多数据类型测试流程，支持 expectSuccess=false 的异常路径验证
bool RunMulTypedTest(const char* testName,
                     aclrtStream stream,
                     const std::vector<TSelf>& selfHostData,
                     const std::vector<TOther>& otherHostData,
                     const std::vector<int64_t>& shape,
                     aclDataType selfType,
                     aclDataType otherType,
                     aclDataType outType,
                     bool expectSuccess) {
  // if (selfHostData.size() != otherHostData.size()) {
  //   LOG_PRINT("[%s] FAIL: input size mismatch\n", testName);
  //   return false;
  // }

  int ret = 0;
  bool pass = true;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<TOut> outHostData(selfHostData.size(), static_cast<TOut>(0));
  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, selfType, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, otherType, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, outType, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out tensor failed. ERROR: %d\n", testName, ret); return false);

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  if (!expectSuccess) {
    pass = (ret != ACL_SUCCESS);
    LOG_PRINT("[%s] %s\n", testName, pass ? "PASS" : "FAIL");
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    aclrtFree(outDeviceAddr);
    return pass;
  }

  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclnnMulGetWorkspaceSize failed. ERROR: %d\n", testName, ret); return false);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[%s] FAIL: allocate workspace failed. ERROR: %d\n", testName, ret); return false);
  }

  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclnnMul failed. ERROR: %d\n", testName, ret); return false);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret); return false);

  std::vector<TOut> resultData(selfHostData.size(), static_cast<TOut>(0));
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST); 
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: copy result from device to host failed. ERROR: %d\n", testName, ret); return false);

  std::vector<TOut> expectedData(selfHostData.size(), static_cast<TOut>(0));
  for (size_t i = 0; i < selfHostData.size(); ++i) {
    expectedData[i] = static_cast<TOut>(selfHostData[i]) * static_cast<TOut>(otherHostData[i]);
  }

  pass = CheckResultTyped(resultData, expectedData);
  LOG_PRINT("[%s] %s\n", testName, pass ? "PASS" : "FAIL");

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

template <typename TSelf, typename TOther, typename TOut>
bool RunMulBroadcastTypedTest(const char* testName,
                              aclrtStream stream,
                              const std::vector<TSelf>& selfHostData,
                              const std::vector<TOther>& otherHostData,
                              const std::vector<int64_t>& selfShape,
                              const std::vector<int64_t>& otherShape,
                              const std::vector<int64_t>& outShape,
                              aclDataType selfType,
                              aclDataType otherType,
                              aclDataType outType,
                              bool expectSuccess) {
  std::vector<int64_t> inferredOutShape;
  const bool canBroadcast = GetBroadcastShape(selfShape, otherShape, inferredOutShape);
  if (expectSuccess) {
    if (!canBroadcast) {
      LOG_PRINT("[%s] FAIL: shapes cannot be broadcast\n", testName);
      return false;
    }
    if (inferredOutShape != outShape) {
      LOG_PRINT("[%s] FAIL: output shape mismatch\n", testName);
      return false;
    }
  }

  if (selfHostData.size() != static_cast<size_t>(GetShapeSize(selfShape))) {
    LOG_PRINT("[%s] FAIL: self data size mismatch with self shape\n", testName);
    return false;
  }
  if (otherHostData.size() != static_cast<size_t>(GetShapeSize(otherShape))) {
    LOG_PRINT("[%s] FAIL: other data size mismatch with other shape\n", testName);
    return false;
  }

  int ret = 0;
  bool pass = true;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<TOut> outHostData(static_cast<size_t>(GetShapeSize(outShape)), static_cast<TOut>(0));
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, selfType, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, otherType, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, outType, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out tensor failed. ERROR: %d\n", testName, ret); return false);

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  if (!expectSuccess) {
    pass = (ret != ACL_SUCCESS);
    LOG_PRINT("[%s] %s\n", testName, pass ? "PASS" : "FAIL");
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    aclrtFree(outDeviceAddr);
    return pass;
  }

  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclnnMulGetWorkspaceSize failed. ERROR: %d\n", testName, ret); return false);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[%s] FAIL: allocate workspace failed. ERROR: %d\n", testName, ret); return false);
  }

  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclnnMul failed. ERROR: %d\n", testName, ret); return false);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret); return false);

  std::vector<TOut> resultData(static_cast<size_t>(GetShapeSize(outShape)), static_cast<TOut>(0));
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: copy result from device to host failed. ERROR: %d\n", testName, ret); return false);

  std::vector<TOut> expectedData(static_cast<size_t>(GetShapeSize(outShape)), static_cast<TOut>(0));
  for (size_t i = 0; i < expectedData.size(); ++i) {
    const std::vector<int64_t> outCoords = GetLinearToCoords(i, outShape);
    const size_t selfOffset = GetBroadcastedOffset(selfShape, outCoords);
    const size_t otherOffset = GetBroadcastedOffset(otherShape, outCoords);
    expectedData[i] = static_cast<TOut>(selfHostData[selfOffset]) * static_cast<TOut>(otherHostData[otherOffset]);
  }

  pass = CheckResultTyped(resultData, expectedData);
  LOG_PRINT("[%s] %s\n", testName, pass ? "PASS" : "FAIL");

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

// Muls（tensor * scalar）测试流程
bool RunMulsTest(const char* testName,
                 aclrtStream stream,
                 const std::vector<float>& selfHostData,
                 float scalarValue,
                 const std::vector<int64_t>& shape) {
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> outHostData(selfHostData.size(), 0.0f);
  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out tensor failed. ERROR: %d\n", testName, ret); return false);

  aclScalar* scalar = aclCreateScalar(&scalarValue, aclDataType::ACL_FLOAT);
  CHECK_RET(scalar != nullptr, LOG_PRINT("[%s] FAIL: aclCreateScalar failed.\n", testName); return false);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnMulsGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclnnMulsGetWorkspaceSize failed. ERROR: %d\n", testName, ret); return false);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[%s] FAIL: allocate workspace failed. ERROR: %d\n", testName, ret); return false);
  }

  ret = aclnnMuls(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclnnMuls failed. ERROR: %d\n", testName, ret); return false);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret); return false);

  std::vector<float> resultData(selfHostData.size(), 0.0f);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: copy result from device to host failed. ERROR: %d\n", testName, ret); return false);

  std::vector<float> expectedData(selfHostData.size(), 0.0f);
  for (size_t i = 0; i < selfHostData.size(); ++i) {
    expectedData[i] = selfHostData[i] * scalarValue;
  }

  bool pass = CheckResult(resultData, expectedData);
  LOG_PRINT("[%s] %s\n", testName, pass ? "PASS" : "FAIL");

  aclDestroyScalar(scalar);
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return pass;
}

// 混合数据类型场景：FLOAT16(self) * FLOAT32(other) -> FLOAT32(out)
// 说明：FLOAT16 输入使用 half 的位模式（uint16_t）构造
bool RunFloat16FloatMixTest(const char* testName,
                            aclrtStream stream,
                            const std::vector<uint16_t>& selfHalfBits,
                            const std::vector<float>& selfFloatValue,
                            const std::vector<float>& otherHostData,
                            const std::vector<int64_t>& shape) {
  if (selfHalfBits.size() != otherHostData.size() || selfFloatValue.size() != otherHostData.size()) {
    LOG_PRINT("[%s] FAIL: input size mismatch\n", testName);
    return false;
  }

  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<float> outHostData(selfHalfBits.size(), 0.0f);
  ret = CreateAclTensor(selfHalfBits, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self(float16) tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other(float32) tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out(float32) tensor failed. ERROR: %d\n", testName, ret); return false);

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclnnMulGetWorkspaceSize failed. ERROR: %d\n", testName, ret); return false);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[%s] FAIL: allocate workspace failed. ERROR: %d\n", testName, ret); return false);
  }

  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclnnMul failed. ERROR: %d\n", testName, ret); return false);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret); return false);

  std::vector<float> resultData(selfHalfBits.size(), 0.0f);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: copy result from device to host failed. ERROR: %d\n", testName, ret); return false);

  std::vector<float> expectedData(selfHalfBits.size(), 0.0f);
  for (size_t i = 0; i < selfHalfBits.size(); ++i) {
    expectedData[i] = selfFloatValue[i] * otherHostData[i];
  }
  bool pass = CheckResult(resultData, expectedData);
  LOG_PRINT("[%s] %s\n", testName, pass ? "PASS" : "FAIL");

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

// InplaceMul 测试流程：结果直接覆盖 self
bool RunInplaceMulTest(const char* testName,
                       aclrtStream stream,
                       const std::vector<float>& selfHostData,
                       const std::vector<float>& otherHostData,
                       const std::vector<int64_t>& shape) {
  if (selfHostData.size() != otherHostData.size()) {
    LOG_PRINT("[%s] FAIL: input size mismatch\n", testName);
    return false;
  }
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);

  ret = aclnnInplaceMulGetWorkspaceSize(self, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclnnInplaceMulGetWorkspaceSize failed. ERROR: %d\n", testName, ret); return false);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[%s] FAIL: allocate workspace failed. ERROR: %d\n", testName, ret); return false);
  }

  ret = aclnnInplaceMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclnnInplaceMul failed. ERROR: %d\n", testName, ret); return false);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret); return false);

  std::vector<float> resultData(selfHostData.size(), 0.0f);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: copy result from device to host failed. ERROR: %d\n", testName, ret); return false);

  std::vector<float> expectedData(selfHostData.size(), 0.0f);
  for (size_t i = 0; i < selfHostData.size(); ++i) {
    expectedData[i] = selfHostData[i] * otherHostData[i];
  }
  bool pass = CheckResult(resultData, expectedData);
  LOG_PRINT("[%s] %s\n", testName, pass ? "PASS" : "FAIL");

  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return pass;
}

// InplaceMuls 测试流程：结果直接覆盖 self
bool RunInplaceMulsTest(const char* testName,
                        aclrtStream stream,
                        const std::vector<float>& selfHostData,
                        float scalarValue,
                        const std::vector<int64_t>& shape) {
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* scalar = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  scalar = aclCreateScalar(&scalarValue, aclDataType::ACL_FLOAT);
  CHECK_RET(scalar != nullptr, LOG_PRINT("[%s] FAIL: aclCreateScalar failed.\n", testName); return false);

  ret = aclnnInplaceMulsGetWorkspaceSize(self, scalar, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclnnInplaceMulsGetWorkspaceSize failed. ERROR: %d\n", testName, ret); return false);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[%s] FAIL: allocate workspace failed. ERROR: %d\n", testName, ret); return false);
  }

  ret = aclnnInplaceMuls(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclnnInplaceMuls failed. ERROR: %d\n", testName, ret); return false);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret); return false);

  std::vector<float> resultData(selfHostData.size(), 0.0f);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: copy result from device to host failed. ERROR: %d\n", testName, ret); return false);

  std::vector<float> expectedData(selfHostData.size(), 0.0f);
  for (size_t i = 0; i < selfHostData.size(); ++i) {
    expectedData[i] = selfHostData[i] * scalarValue;
  }
  bool pass = CheckResult(resultData, expectedData);
  LOG_PRINT("[%s] %s\n", testName, pass ? "PASS" : "FAIL");

  aclDestroyScalar(scalar);
  aclDestroyTensor(self);
  aclrtFree(selfDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return pass;
}

// 异常路径：输入 nullptr，期望返回非成功状态
bool RunNullptrTest(const char* testName) {
  int ret = 0;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> outData = {0.0f, 0.0f, 0.0f, 0.0f};

  ret = CreateAclTensor(data, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(outData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out tensor failed. ERROR: %d\n", testName, ret); return false);

  ret = aclnnMulGetWorkspaceSize(nullptr, other, out, &workspaceSize, &executor);
  bool pass = (ret != ACL_SUCCESS);
  LOG_PRINT("[%s] %s (ret=%d)\n", testName, pass ? "PASS" : "FAIL", ret);

  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  return pass;
}

template <typename TSelf, typename TOther, typename TOut>
bool RunMulPromoteProbeTyped(const char* testName,
                             const std::vector<TSelf>& selfHostData,
                             const std::vector<TOther>& otherHostData,
                             const std::vector<int64_t>& shape,
                             aclDataType selfType,
                             aclDataType otherType,
                             aclDataType outType) {
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<TOut> outHostData(selfHostData.size(), static_cast<TOut>(0));
  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, selfType, &self);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] PROBE setup fail: create self tensor failed. ERROR: %d\n", testName, ret); return true);
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, otherType, &other);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] PROBE setup fail: create other tensor failed. ERROR: %d\n", testName, ret); return true);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, outType, &out);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] PROBE setup fail: create out tensor failed. ERROR: %d\n", testName, ret); return true);

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  LOG_PRINT("[%s] PROBE ret=%d\n", testName, ret);

  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  return true;
}

template <typename TSelf, typename TOther>
bool RunInplaceMulPromoteProbeTyped(const char* testName,
                                    const std::vector<TSelf>& selfHostData,
                                    const std::vector<TOther>& otherHostData,
                                    const std::vector<int64_t>& shape,
                                    aclDataType selfType,
                                    aclDataType otherType) {
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, selfType, &self);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] PROBE setup fail: create self tensor failed. ERROR: %d\n", testName, ret); return true);
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, otherType, &other);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] PROBE setup fail: create other tensor failed. ERROR: %d\n", testName, ret); return true);

  ret = aclnnInplaceMulGetWorkspaceSize(self, other, &workspaceSize, &executor);
  LOG_PRINT("[%s] PROBE ret=%d\n", testName, ret);

  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  return true;
}

template <typename TSelf, typename TOut, typename TScalar>
bool RunMulsPromoteProbeTyped(const char* testName,
                              const std::vector<TSelf>& selfHostData,
                              TScalar scalarValue,
                              const std::vector<int64_t>& shape,
                              aclDataType selfType,
                              aclDataType scalarType,
                              aclDataType outType) {
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclScalar* scalar = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<TOut> outHostData(selfHostData.size(), static_cast<TOut>(0));
  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, selfType, &self);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] PROBE setup fail: create self tensor failed. ERROR: %d\n", testName, ret); return true);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, outType, &out);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] PROBE setup fail: create out tensor failed. ERROR: %d\n", testName, ret); return true);

  scalar = aclCreateScalar(&scalarValue, scalarType);
  CHECK_RET(scalar != nullptr,
            LOG_PRINT("[%s] PROBE setup fail: aclCreateScalar failed.\n", testName); return true);

  ret = aclnnMulsGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor);
  LOG_PRINT("[%s] PROBE ret=%d\n", testName, ret);

  aclDestroyScalar(scalar);
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  return true;
}

// 异常路径：shape 不可 broadcast，期望返回非成功状态
bool RunInvalidShapeTest(const char* testName) {
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> otherShape = {4};
  std::vector<int64_t> outShape = {2, 3};
  std::vector<float> selfData = {1, 2, 3, 4, 5, 6};
  std::vector<float> otherData = {1, 2, 3, 4};
  std::vector<float> outData = {0, 0, 0, 0, 0, 0};

  ret = CreateAclTensor(selfData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(otherData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(outData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out tensor failed. ERROR: %d\n", testName, ret); return false);

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  bool pass = (ret != ACL_SUCCESS);
  LOG_PRINT("[%s] %s (ret=%d)\n", testName, pass ? "PASS" : "FAIL", ret);

  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  return pass;
}

template <typename TSelf, typename TScalar, typename TOut>
bool RunMulsTypedTest(const char* testName,
                      aclrtStream stream,
                      const std::vector<TSelf>& selfHostData,
                      TScalar scalarValue,
                      const std::vector<int64_t>& shape,
                      aclDataType selfType,
                      aclDataType scalarType,
                      aclDataType outType,
                      bool expectSuccess) {
  if (selfHostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
    LOG_PRINT("[%s] FAIL: self data size mismatch with shape\n", testName);
    return false;
  }

  int ret = 0;
  bool pass = true;
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclScalar* scalar = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<TOut> outHostData(selfHostData.size(), static_cast<TOut>(0));
  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, selfType, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, outType, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out tensor failed. ERROR: %d\n", testName, ret); return false);

  scalar = aclCreateScalar(&scalarValue, scalarType);
  CHECK_RET(scalar != nullptr, LOG_PRINT("[%s] FAIL: aclCreateScalar failed.\n", testName); return false);

  ret = aclnnMulsGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor);
  if (!expectSuccess) {
    pass = (ret != ACL_SUCCESS);
    LOG_PRINT("[%s] %s\n", testName, pass ? "PASS" : "FAIL");
    aclDestroyScalar(scalar);
    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr);
    aclrtFree(outDeviceAddr);
    return pass;
  }

  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclnnMulsGetWorkspaceSize failed. ERROR: %d\n", testName, ret); return false);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[%s] FAIL: allocate workspace failed. ERROR: %d\n", testName, ret); return false);
  }

  ret = aclnnMuls(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclnnMuls failed. ERROR: %d\n", testName, ret); return false);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret); return false);

  std::vector<TOut> resultData(selfHostData.size(), static_cast<TOut>(0));
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: copy result from device to host failed. ERROR: %d\n", testName, ret); return false);

  std::vector<TOut> expectedData(selfHostData.size(), static_cast<TOut>(0));
  for (size_t i = 0; i < selfHostData.size(); ++i) {
    expectedData[i] = static_cast<TOut>(selfHostData[i]) * static_cast<TOut>(scalarValue);
  }

  pass = CheckResultTyped(resultData, expectedData);
  LOG_PRINT("[%s] %s\n", testName, pass ? "PASS" : "FAIL");

  aclDestroyScalar(scalar);
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return pass;
}

template <typename TSelf, typename TOther>
bool RunInplaceMulTypedTest(const char* testName,
                            aclrtStream stream,
                            const std::vector<TSelf>& selfHostData,
                            const std::vector<TOther>& otherHostData,
                            const std::vector<int64_t>& shape,
                            aclDataType selfType,
                            aclDataType otherType,
                            bool expectSuccess) {
  if (selfHostData.size() != static_cast<size_t>(GetShapeSize(shape)) ||
      otherHostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
    LOG_PRINT("[%s] FAIL: input data size mismatch with shape\n", testName);
    return false;
  }

  int ret = 0;
  bool pass = true;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, selfType, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, otherType, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);

  ret = aclnnInplaceMulGetWorkspaceSize(self, other, &workspaceSize, &executor);
  if (!expectSuccess) {
    pass = (ret != ACL_SUCCESS);
    LOG_PRINT("[%s] %s\n", testName, pass ? "PASS" : "FAIL");
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    return pass;
  }

  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclnnInplaceMulGetWorkspaceSize failed. ERROR: %d\n", testName, ret); return false);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[%s] FAIL: allocate workspace failed. ERROR: %d\n", testName, ret); return false);
  }

  ret = aclnnInplaceMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclnnInplaceMul failed. ERROR: %d\n", testName, ret); return false);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret); return false);

  std::vector<TSelf> resultData(selfHostData.size(), static_cast<TSelf>(0));
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: copy result from device to host failed. ERROR: %d\n", testName, ret); return false);

  std::vector<TSelf> expectedData(selfHostData.size(), static_cast<TSelf>(0));
  for (size_t i = 0; i < selfHostData.size(); ++i) {
    expectedData[i] = static_cast<TSelf>(selfHostData[i]) * static_cast<TSelf>(otherHostData[i]);
  }

  pass = CheckResultTyped(resultData, expectedData);
  LOG_PRINT("[%s] %s\n", testName, pass ? "PASS" : "FAIL");

  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return pass;
}

template <typename TSelf, typename TOther>
bool RunInplaceMulBroadcastTypedTest(const char* testName,
                                     aclrtStream stream,
                                     const std::vector<TSelf>& selfHostData,
                                     const std::vector<TOther>& otherHostData,
                                     const std::vector<int64_t>& selfShape,
                                     const std::vector<int64_t>& otherShape,
                                     aclDataType selfType,
                                     aclDataType otherType,
                                     bool expectSuccess) {
  std::vector<int64_t> inferredOutShape;
  const bool canBroadcast = GetBroadcastShape(selfShape, otherShape, inferredOutShape);
  if (expectSuccess) {
    if (!canBroadcast || inferredOutShape != selfShape) {
      LOG_PRINT("[%s] FAIL: inplace output shape must equal self shape\n", testName);
      return false;
    }
  }

  if (selfHostData.size() != static_cast<size_t>(GetShapeSize(selfShape)) ||
      otherHostData.size() != static_cast<size_t>(GetShapeSize(otherShape))) {
    LOG_PRINT("[%s] FAIL: input data size mismatch with shape\n", testName);
    return false;
  }

  int ret = 0;
  bool pass = true;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, selfType, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, otherType, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);

  ret = aclnnInplaceMulGetWorkspaceSize(self, other, &workspaceSize, &executor);
  if (!expectSuccess) {
    pass = (ret != ACL_SUCCESS);
    LOG_PRINT("[%s] %s\n", testName, pass ? "PASS" : "FAIL");
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    return pass;
  }

  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclnnInplaceMulGetWorkspaceSize failed. ERROR: %d\n", testName, ret); return false);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[%s] FAIL: allocate workspace failed. ERROR: %d\n", testName, ret); return false);
  }

  ret = aclnnInplaceMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclnnInplaceMul failed. ERROR: %d\n", testName, ret); return false);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret); return false);

  std::vector<TSelf> resultData(selfHostData.size(), static_cast<TSelf>(0));
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: copy result from device to host failed. ERROR: %d\n", testName, ret); return false);

  std::vector<TSelf> expectedData(selfHostData.size(), static_cast<TSelf>(0));
  for (size_t i = 0; i < expectedData.size(); ++i) {
    const std::vector<int64_t> outCoords = GetLinearToCoords(i, selfShape);
    const size_t otherOffset = GetBroadcastedOffset(otherShape, outCoords);
    expectedData[i] = static_cast<TSelf>(selfHostData[i]) * static_cast<TSelf>(otherHostData[otherOffset]);
  }

  pass = CheckResultTyped(resultData, expectedData);
  LOG_PRINT("[%s] %s\n", testName, pass ? "PASS" : "FAIL");

  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return pass;
}

template <typename TSelf, typename TScalar>
bool RunInplaceMulsTypedTest(const char* testName,
                             aclrtStream stream,
                             const std::vector<TSelf>& selfHostData,
                             TScalar scalarValue,
                             const std::vector<int64_t>& shape,
                             aclDataType selfType,
                             aclDataType scalarType,
                             bool expectSuccess) {
  if (selfHostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
    LOG_PRINT("[%s] FAIL: self data size mismatch with shape\n", testName);
    return false;
  }

  int ret = 0;
  bool pass = true;
  void* selfDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* scalar = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, selfType, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  scalar = aclCreateScalar(&scalarValue, scalarType);
  CHECK_RET(scalar != nullptr, LOG_PRINT("[%s] FAIL: aclCreateScalar failed.\n", testName); return false);

  ret = aclnnInplaceMulsGetWorkspaceSize(self, scalar, &workspaceSize, &executor);
  if (!expectSuccess) {
    pass = (ret != ACL_SUCCESS);
    LOG_PRINT("[%s] %s\n", testName, pass ? "PASS" : "FAIL");
    aclDestroyScalar(scalar);
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    return pass;
  }

  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclnnInplaceMulsGetWorkspaceSize failed. ERROR: %d\n", testName, ret); return false);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[%s] FAIL: allocate workspace failed. ERROR: %d\n", testName, ret); return false);
  }

  ret = aclnnInplaceMuls(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: aclnnInplaceMuls failed. ERROR: %d\n", testName, ret); return false);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret); return false);

  std::vector<TSelf> resultData(selfHostData.size(), static_cast<TSelf>(0));
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[%s] FAIL: copy result from device to host failed. ERROR: %d\n", testName, ret); return false);

  std::vector<TSelf> expectedData(selfHostData.size(), static_cast<TSelf>(0));
  for (size_t i = 0; i < selfHostData.size(); ++i) {
    expectedData[i] = static_cast<TSelf>(selfHostData[i]) * static_cast<TSelf>(scalarValue);
  }

  pass = CheckResultTyped(resultData, expectedData);
  LOG_PRINT("[%s] %s\n", testName, pass ? "PASS" : "FAIL");

  aclDestroyScalar(scalar);
  aclDestroyTensor(self);
  aclrtFree(selfDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return pass;
}

bool RunMulNullptrVariantTest(const char* testName, bool nullSelf, bool nullOther, bool nullOut) {
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> outData = {0.0f, 0.0f, 0.0f, 0.0f};

  if (!nullSelf) {
    ret = CreateAclTensor(data, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  }
  if (!nullOther) {
    ret = CreateAclTensor(data, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  }
  if (!nullOut) {
    ret = CreateAclTensor(outData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out tensor failed. ERROR: %d\n", testName, ret); return false);
  }

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  const bool pass = (ret != ACL_SUCCESS);
  LOG_PRINT("[%s] %s (ret=%d)\n", testName, pass ? "PASS" : "FAIL", ret);

  if (self != nullptr) aclDestroyTensor(self);
  if (other != nullptr) aclDestroyTensor(other);
  if (out != nullptr) aclDestroyTensor(out);
  if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
  if (otherDeviceAddr != nullptr) aclrtFree(otherDeviceAddr);
  if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);
  return pass;
}

bool RunMulsNullptrScalarTest(const char* testName) {
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> outData = {0.0f, 0.0f, 0.0f, 0.0f};

  ret = CreateAclTensor(data, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(outData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out tensor failed. ERROR: %d\n", testName, ret); return false);

  ret = aclnnMulsGetWorkspaceSize(self, nullptr, out, &workspaceSize, &executor);
  const bool pass = (ret != ACL_SUCCESS);
  LOG_PRINT("[%s] %s (ret=%d)\n", testName, pass ? "PASS" : "FAIL", ret);

  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  return pass;
}

bool RunInplaceMulNullptrOtherTest(const char* testName) {
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

  ret = CreateAclTensor(data, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);

  ret = aclnnInplaceMulGetWorkspaceSize(self, nullptr, &workspaceSize, &executor);
  const bool pass = (ret != ACL_SUCCESS);
  LOG_PRINT("[%s] %s (ret=%d)\n", testName, pass ? "PASS" : "FAIL", ret);

  aclDestroyTensor(self);
  aclrtFree(selfDeviceAddr);
  return pass;
}

bool RunInplaceMulsNullptrScalarTest(const char* testName) {
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

  ret = CreateAclTensor(data, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);

  ret = aclnnInplaceMulsGetWorkspaceSize(self, nullptr, &workspaceSize, &executor);
  const bool pass = (ret != ACL_SUCCESS);
  LOG_PRINT("[%s] %s (ret=%d)\n", testName, pass ? "PASS" : "FAIL", ret);

  aclDestroyTensor(self);
  aclrtFree(selfDeviceAddr);
  return pass;
}

bool RunWrongOutShapeTest(const char* testName) {
  int ret = 0;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> otherShape = {3};
  std::vector<int64_t> wrongOutShape = {3, 2};
  std::vector<float> selfData = {1, 2, 3, 4, 5, 6};
  std::vector<float> otherData = {1, 2, 3};
  std::vector<float> outData = {0, 0, 0, 0, 0, 0};

  ret = CreateAclTensor(selfData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create self tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(otherData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create other tensor failed. ERROR: %d\n", testName, ret); return false);
  ret = CreateAclTensor(outData, wrongOutShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] FAIL: create out tensor failed. ERROR: %d\n", testName, ret); return false);

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  const bool pass = (ret != ACL_SUCCESS);
  LOG_PRINT("[%s] %s (ret=%d)\n", testName, pass ? "PASS" : "FAIL", ret);

  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  return pass;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  bool allPass = true;

  // 1) 基础 float32 与边界值场景
  allPass = RunMulTest("float32_basic_mul", stream,
                       {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f},
                       {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f},
                       {4, 2}) && allPass;

  allPass = RunMulTest("negative_zero_boundary_mul", stream,
                       {-1.0f, 0.0f, 2.5f, -3.0f, 0.0f, 4.0f, -7.5f, 8.0f},
                       {2.0f, -3.0f, 0.0f, -1.0f, 5.0f, 0.0f, -2.0f, 0.0f},
                       {4, 2}) && allPass;

  allPass = RunMulsTest("aclnnMuls_variant", stream,
                        {-2.0f, -1.5f, 0.0f, 1.5f, 2.0f, 4.0f, -6.0f, 8.0f},
                        2.5f,
                        {4, 2}) && allPass;

  // 2) 多 dtype 与混合 dtype 场景
  allPass = RunMulTypedTest<int32_t, int32_t, int32_t>("int32_mul", stream,
                                                        {1, -2, 3, -4, 5, 0, -7, 8},
                                                        {-1, 2, -3, 4, 0, 6, -2, 1},
                                                        {4, 2},
                                                        aclDataType::ACL_INT32,
                                                        aclDataType::ACL_INT32,
                                                        aclDataType::ACL_INT32,
                                                        true) && allPass;

  allPass = RunMulTypedTest<double, double, double>("double_mul", stream,
                                                     {1.0, -2.0, 3.5, -4.0, 5.0, 0.0, -7.5, 8.0},
                                                     {-1.0, 2.0, -3.0, 4.0, 0.0, 6.0, -2.0, 1.0},
                                                     {4, 2},
                                                     aclDataType::ACL_DOUBLE,
                                                     aclDataType::ACL_DOUBLE,
                                                     aclDataType::ACL_DOUBLE,
                                                     true) && allPass;

  allPass = RunMulTypedTest<int32_t, float, float>("mixed_int32_float_mul", stream,
                                                    {1, -2, 3, -4, 5, 0, -7, 8},
                                                    {-1.5f, 2.0f, -3.0f, 4.0f, 0.5f, 6.0f, -2.0f, 1.0f},
                                                    {4, 2},
                                                    aclDataType::ACL_INT32,
                                                    aclDataType::ACL_FLOAT,
                                                    aclDataType::ACL_FLOAT,
                                                    false) && allPass;

  allPass = RunMulTypedTest<int8_t, int8_t, int8_t>("int8_mul", stream,
                                                     {1, -2, 3, -4, 5, 0, -7, 8},
                                                     {-1, 2, -3, 4, 0, 6, -2, 1},
                                                     {4, 2},
                                                     aclDataType::ACL_INT8,
                                                     aclDataType::ACL_INT8,
                                                     aclDataType::ACL_INT8,
                                                     true) && allPass;

  allPass = RunFloat16FloatMixTest("mixed_float16_float32_mul", stream,
                                   {0x3C00, 0xC000, 0x4200, 0x4400, 0x0000, 0xBC00, 0x4000, 0xC200},
                                   {1.0f, -2.0f, 3.0f, 4.0f, 0.0f, -1.0f, 2.0f, -3.0f},
                                   {2.0f, -0.5f, 1.5f, -2.0f, 3.0f, 4.0f, -1.0f, 0.5f},
                                   {4, 2}) && allPass;
  
  std::vector<int64_t> shape = {2, 2};

  // 1. BF16, BF16, BF16 (同类型) - 使用 uint16_t 位模式模拟
  allPass = RunMulTypedTest<uint16_t, uint16_t, uint16_t>("1_BF16_Same", stream, 
            {0x3F80, 0x4000, 0x4040, 0x4080}, {0x3F80, 0x3F80, 0x3F80, 0x3F80}, shape, 
            ACL_BF16, ACL_BF16, ACL_BF16, true) && allPass;

  // 2. BF16, FLOAT, FLOAT (混合类型)
  allPass = RunMulTypedTest<uint16_t, float, float>("2_BF16_Mixed", stream, 
            {0x4000, 0x4000, 0x4000, 0x4000}, {1.0f, 2.0f, 3.0f, 4.0f}, shape, 
            ACL_BF16, ACL_FLOAT, ACL_FLOAT, true) && allPass;

  // 3. FLOAT, BF16, FLOAT (混合类型)
  allPass = RunMulTypedTest<float, uint16_t, float>("3_BF16_Mixed_Rev", stream, 
            {1.0f, 2.0f, 3.0f, 4.0f}, {0x4000, 0x4000, 0x4000, 0x4000}, shape, 
            ACL_FLOAT, ACL_BF16, ACL_FLOAT, true) && allPass;

  // 4. FLOAT16, FLOAT16, FLOAT16 (同类型)
  allPass = RunMulTypedTest<uint16_t, uint16_t, uint16_t>("4_FP16_Same", stream, 
            {0x3C00, 0x3C00, 0x3C00, 0x3C00}, {0x3C00, 0x3C00, 0x3C00, 0x3C00}, shape, 
            ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, true) && allPass;

  // 5. FLOAT16, FLOAT, FLOAT (混合类型)
  allPass = RunMulTypedTest<uint16_t, float, float>("5_FP16_Mixed", stream, 
            {0x3C00, 0x3C00, 0x3C00, 0x3C00}, {2.0f, 3.0f, 4.0f, 5.0f}, shape, 
            ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT, true) && allPass;

  // 6. FLOAT, FLOAT16, FLOAT (混合类型)
  allPass = RunMulTypedTest<float, uint16_t, float>("6_FP16_Mixed_Rev", stream, 
            {2.0f, 3.0f, 4.0f, 5.0f}, {0x3C00, 0x3C00, 0x3C00, 0x3C00}, shape, 
            ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT, true) && allPass;

  // 7. FLOAT, FLOAT, FLOAT (同类型)
  allPass = RunMulTypedTest<float, float, float>("7_FP32_Same", stream, 
            {1.1f, 2.2f, 3.3f, 4.4f}, {2.0f, 2.0f, 2.0f, 2.0f}, shape, 
            ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true) && allPass;

  // 8. INT32, INT32, INT32 (同类型)
  allPass = RunMulTypedTest<int32_t, int32_t, int32_t>("8_INT32_Same", stream, 
            {10, -20, 30, 0}, {2, 3, -1, 5}, shape, 
            ACL_INT32, ACL_INT32, ACL_INT32, true) && allPass;

  // 9. UINT8, UINT8, UINT8 (同类型)
  allPass = RunMulTypedTest<uint8_t, uint8_t, uint8_t>("9_UINT8_Same", stream, 
            {1, 2, 3, 4}, {10, 20, 30, 40}, shape, 
            ACL_UINT8, ACL_UINT8, ACL_UINT8, true) && allPass;

  // 10. INT8, INT8, INT8 (同类型)
  allPass = RunMulTypedTest<int8_t, int8_t, int8_t>("10_INT8_Same", stream, 
            {-1, 2, -3, 4}, {5, 5, 5, 5}, shape, 
            ACL_INT8, ACL_INT8, ACL_INT8, true) && allPass;

  // 11. INT64, INT64, INT64 (同类型)
  allPass = RunMulTypedTest<int64_t, int64_t, int64_t>("11_INT64_Same", stream, 
            {100, 200, 300, 400}, {2, 2, 2, 2}, shape, 
            ACL_INT64, ACL_INT64, ACL_INT64, true) && allPass;

  // 12. INT16, INT16, INT16 (同类型)
  allPass = RunMulTypedTest<int16_t, int16_t, int16_t>("12_INT16_Same", stream, 
            {10, 20, 30, 40}, {2, 2, 2, 2}, shape, 
            ACL_INT16, ACL_INT16, ACL_INT16, true) && allPass;

  // 13. COMPLEX32, COMPLEX32, COMPLEX32 (同类型)
  typedef std::complex<float> c64;
  allPass = RunMulTypedTest<c64, c64, c64>("13_Complex32_Same", stream, 
            {0x3C003C00}, {0x3C003C00}, {1, 1}, 
            ACL_COMPLEX32, ACL_COMPLEX32, ACL_COMPLEX32, true) && allPass;

  // 14. COMPLEX64, COMPLEX64, COMPLEX64 (同类型)
  allPass = RunMulTypedTest<c64, c64, c64>("14_Complex64_Same", stream, 
            {{1.0f, 2.0f}}, {{2.0f, 0.0f}}, {1, 1}, 
            ACL_COMPLEX64, ACL_COMPLEX64, ACL_COMPLEX64, true) && allPass;
  // FLOAT, COMPLEX64, COMPLEX64 (混合类型)  
  allPass = RunMulTypedTest<float, c64, c64>("14_FP32_Complex64_Mixed", stream, 
            {{1.0f, 2.0f}}, {{3.0f, 4.0f}}, {1, 1},
            ACL_FLOAT, ACL_COMPLEX64, ACL_COMPLEX64, true) && allPass;
  
  allPass = RunMulTypedTest<c64, float, c64>("14_Complex64_FP32_Mixed", stream, 
            {{1.0f, 2.0f}}, {{3.0f}}, {1, 1}, 
            ACL_COMPLEX64, ACL_FLOAT, ACL_COMPLEX64, true) && allPass;
  
  allPass = RunMulTypedTest<float, float, c64>("14_FP32_Complex64_Mixed_Rev2", stream, 
            {{1.0f}}, {{3.0f, 4.0f}}, {1, 1}, 
            ACL_FLOAT, ACL_FLOAT, ACL_COMPLEX64, true) && allPass;

  // 15. COMPLEX128, COMPLEX128, COMPLEX128 (同类型)
  typedef std::complex<double> c128;
  allPass = RunMulTypedTest<c128, c128, c128>("15_Complex128_Same", stream, 
            {{1.0, 2.0}}, {{2.0, 0.0}}, {1, 1}, 
            ACL_COMPLEX128, ACL_COMPLEX128, ACL_COMPLEX128, true) && allPass;
  
  allPass = RunMulTypedTest<double, c128, c128>("15_FP64_Complex128_Mixed", stream, 
            {1.0}, {{3.0, 4.0}}, {1, 1}, 
            ACL_DOUBLE, ACL_COMPLEX128, ACL_COMPLEX128, true) && allPass;

  // 16. BOOL, BOOL, BOOL (同类型)
  allPass = RunMulTypedTest<uint8_t, uint8_t, uint8_t>("16_BOOL_Same", stream, 
            {1, 0, 1, 0}, {1, 1, 0, 0}, shape, 
            ACL_BOOL, ACL_BOOL, ACL_BOOL, true) && allPass;

  // 17. DOUBLE, DOUBLE, DOUBLE (同类型)
  allPass = RunMulTypedTest<double, double, double>("17_Double_Same", stream, 
            {1.2345, 6.789}, {2.0, 1.0}, {2, 1}, 
            ACL_DOUBLE, ACL_DOUBLE, ACL_DOUBLE, true) && allPass;

  // 18. 负向测试 (UINT32 不在注册表中)
  allPass = RunMulTypedTest<uint32_t, uint32_t, uint32_t>("18_Illegal_UINT32", stream, 
            {1}, {1}, {1, 1}, 
            ACL_UINT32, ACL_UINT32, ACL_UINT32, false) && allPass;

  // 19. 更大的非 2 次幂 shape，覆盖不同 tiling 路径
  std::vector<int64_t> shapeLong = {33, 65};
  std::vector<float> longSelf(static_cast<size_t>(GetShapeSize(shapeLong)), 0.0f);
  std::vector<float> longOther(static_cast<size_t>(GetShapeSize(shapeLong)), 0.0f);
  for (size_t i = 0; i < longSelf.size(); ++i) {
    longSelf[i] = static_cast<float>((static_cast<int>(i) % 17) - 8);
    longOther[i] = static_cast<float>((static_cast<int>(i) % 7) - 3);
  }
  allPass = RunMulTypedTest<float, float, float>("19_FP32_NonPowerOfTwo_SameShape", stream,
                                                 longSelf, longOther, shapeLong,
                                                 ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true) && allPass;

  // 20. 标量样式广播: [2, 3] x [1]
  allPass = RunMulBroadcastTypedTest<float, float, float>("20_ScalarLike_Broadcast", stream,
              {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f},
              {2.5f},
              {2, 3}, {1}, {2, 3},
              ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true) && allPass;

  // 21. 中间维广播: [2, 1, 4] x [2, 3, 4]
  allPass = RunMulBroadcastTypedTest<float, float, float>("21_Broadcast_MiddleDim", stream,
              {1, 2, 3, 4, 5, 6, 7, 8},
              {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
               4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6},
              {2, 1, 4}, {2, 3, 4}, {2, 3, 4},
              ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true) && allPass;

  // 22. 含空维广播
  allPass = RunMulBroadcastTypedTest<float, float, float>("22_EmptyDim_Broadcast", stream,
              {},
              {},
              {2, 0, 3}, {1, 3}, {2, 0, 3},
              ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true) && allPass;

  // 23. 下溢边界
  allPass = RunMulTypedTest<float, float, float>("23_FP32_Underflow", stream,
              {std::numeric_limits<float>::min(), -std::numeric_limits<float>::min()},
              {1e-20f, 1e-20f},
              {2},
              ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true) && allPass;

  // 24. Inf * 0 -> NaN
  allPass = RunMulBroadcastTypedTest<float, float, float>("24_Inf_Times_Zero", stream,
              {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()},
              {0.0f, -0.0f},
              {2}, {2}, {2},
              ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true) && allPass;

  // 25. DOUBLE Muls
  allPass = RunMulsTypedTest<double, double, double>("25_Double_Muls", stream,
              {1.25, -2.5, 0.0, 8.0},
              -4.0,
              {4},
              ACL_DOUBLE, ACL_DOUBLE, ACL_DOUBLE, true) && allPass;

  // 26. FLOAT Muls with zero scalar
  allPass = RunMulsTypedTest<float, float, float>("26_Float_Muls_ZeroScalar", stream,
              {1.0f, -2.0f, 3.5f, -4.5f},
              -0.0f,
              {4},
              ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true) && allPass;
  
  allPass = RunMulsTypedTest<float, c64, c64>("27_Float_COMPLEX64_Muls", stream,
              {1.0f, -2.0f, 3.5f, -4.5f},
              -4.0,
              {4},
              ACL_FLOAT, ACL_COMPLEX64, ACL_COMPLEX64, true) && allPass;
  
  allPass = RunMulsTypedTest<c64, float, c64>("28_Complex64_Float_Muls", stream,
              {{1.0f, 2.0f}, {3.0f, 4.0f}},
              -0.0f,
              {2, 2},
              ACL_COMPLEX64, ACL_FLOAT, ACL_COMPLEX64, true) && allPass;
  
  allPass = RunMulsTypedTest<c64, c64, c64>("29_Complex64_Muls", stream,
              {{1.0f, 2.0f}, {3.0f, 4.0f}},
              -1.0f,
              {2, 2},
              ACL_COMPLEX64, ACL_COMPLEX64, ACL_COMPLEX64, true) && allPass;
  
  allPass = RunMulsTypedTest<float, float, c64>("30_FLoat_Complex_Muls_Mixed", stream,
              {1.0f, 2.0f, 3.0f, 4.0f},
              -1.0f,
              {2, 2},
              ACL_FLOAT, ACL_FLOAT, ACL_COMPLEX64, true) && allPass;
  
  allPass = RunMulsTypedTest<int64_t, c64, c64>("31_Int64_Complex_Muls_Mixed", stream,
              {10, -20, 30, -40},
              -2.0f,
              {2, 2},
              ACL_INT64, ACL_COMPLEX64, ACL_COMPLEX64, true) && allPass;

  allPass = RunMulsTypedTest<c64, int64_t, c64>("32_Complex_Int_Muls_Mixed", stream,
              {{1.0f, 2.0f}, {3.0f, 4.0f}},
              -3,
              {2, 2},
              ACL_COMPLEX64, ACL_INT64, ACL_COMPLEX64, true) && allPass;
  
  allPass = RunMulsTypedTest<c128, float, c128>("33_Complex128_Float_Muls", stream,
              {{1.0f, 2.0f}, {3.0f, 4.0f}},
              -1.5f,
              {2, 2},
              ACL_COMPLEX128, ACL_FLOAT, ACL_COMPLEX128, true) && allPass;
  
  allPass = RunMulsTypedTest<c128, double, c128>("34_Complex128_Double_Muls", stream,
              {{1.0f, 2.0f}, {3.0f, 4.0f}},
              -2.5,
              {2, 2},
              ACL_COMPLEX128, ACL_DOUBLE, ACL_COMPLEX128, true) && allPass;
  
  allPass = RunMulsTypedTest<c128, uint8_t, c128>("35_Complex128_BOOL_Muls", stream,
              {{1.0f, 2.0f}, {3.0f, 4.0f}},
              0,
              {2, 2},
              ACL_COMPLEX128, ACL_BOOL, ACL_COMPLEX128, true) && allPass;
  
  allPass = RunMulsTypedTest<c128, c128, c128>("36_Complex128_Muls", stream,
              {{1.0f, 2.0f}, {3.0f, 4.0f}},
              -1.0f,
              {2, 2},
              ACL_COMPLEX128, ACL_COMPLEX128, ACL_COMPLEX128, true) && allPass;

  // 37. INT16 Muls
  allPass = RunMulsTypedTest<int16_t, int16_t, int16_t>("37_Int16_Muls", stream,
              {10, -20, 30, -40},
              static_cast<int16_t>(-3),
              {2, 2},
              ACL_INT16, ACL_INT16, ACL_INT16, true) && allPass;
  
  // 38. 使用不支持的 UINT32 数据类型测试 Muls
  allPass = RunMulsTypedTest<uint32_t, uint32_t, uint32_t>("38_Muls_Illegal_UINT32", stream,
              {1}, 
              static_cast<uint32_t>(1),
              {1, 1},
              ACL_UINT32, ACL_UINT32, ACL_UINT32, false) && allPass;
  
  // 39. FP32 乘以 FP16 标量
  allPass = RunMulsTypedTest<float, uint16_t, float>("39_Muls_FP32_FP16Scalar", stream,
              {1.0f, 2.0f, 3.0f}, 
              static_cast<uint16_t>(2),  // FP16 标量
              {3},
              ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT, true) && allPass;

  // 40. FP32 乘以 BF16 标量
  allPass = RunMulsTypedTest<float, uint16_t, float>("40_Muls_FP32_BF16Scalar", stream,
              {1.0f, 2.0f, 3.0f}, 
              static_cast<uint16_t>(2),  // BF16 标量
              {3},
              ACL_FLOAT, ACL_BF16, ACL_FLOAT, true) && allPass;

  // 40b-40e. 扩展 Muls dtype 组合，提升 CheckMulsPromoteDtype 的分支覆盖
  allPass = RunMulsTypedTest<int32_t, int32_t, int32_t>("40b_Muls_INT32_INT32", stream,
              {1, -2, 3, -4},
              5,
              {2, 2},
              ACL_INT32, ACL_INT32, ACL_INT32, true) && allPass;

  allPass = RunMulsTypedTest<int8_t, int8_t, int8_t>("40c_Muls_INT8_INT8", stream,
              {1, -2, 3, -4},
              static_cast<int8_t>(3),
              {2, 2},
              ACL_INT8, ACL_INT8, ACL_INT8, true) && allPass;

  allPass = RunMulsTypedTest<uint8_t, uint8_t, uint8_t>("40d_Muls_UINT8_UINT8", stream,
              {1, 2, 3, 4},
              static_cast<uint8_t>(2),
              {2, 2},
              ACL_UINT8, ACL_UINT8, ACL_UINT8, true) && allPass;

  allPass = RunMulsTypedTest<uint8_t, uint8_t, uint8_t>("40e_Muls_BOOL_BOOL", stream,
              {1, 0, 1, 0},
              static_cast<uint8_t>(1),
              {2, 2},
              ACL_BOOL, ACL_BOOL, ACL_BOOL, true) && allPass;

  // 41. INT32 InplaceMul
  allPass = RunInplaceMulTypedTest<int32_t, int32_t>("41_Int32_InplaceMul", stream,
              {1, -2, 3, -4},
              {5, 6, -7, 8},
              {2, 2},
              ACL_INT32, ACL_INT32, true) && allPass;

  // 42. 广播 InplaceMul
  allPass = RunInplaceMulBroadcastTypedTest<float, float>("42_Broadcast_InplaceMul", stream,
              {1, 2, 3, 4, 5, 6},
              {10, 20, 30},
              {2, 3}, {3},
              ACL_FLOAT, ACL_FLOAT, true) && allPass;

  // 43. DOUBLE InplaceMuls
  allPass = RunInplaceMulsTypedTest<double, double>("43_Double_InplaceMuls", stream,
              {1.5, -2.0, 0.0, 3.0},
              0.5,
              {4},
              ACL_DOUBLE, ACL_DOUBLE, true) && allPass;

  // 44. INT32 InplaceMuls
  allPass = RunInplaceMulsTypedTest<int32_t, int32_t>("44_Int32_InplaceMuls", stream,
              {1, -2, 3, -4},
              6,
              {2, 2},
              ACL_INT32, ACL_INT32, true) && allPass;

  // 45. 复数特殊值
  float local_nan = std::numeric_limits<float>::quiet_NaN();
  float local_inf = std::numeric_limits<float>::infinity();
  allPass = RunMulTypedTest<c64, c64, c64>("45_Complex64_NaN_Inf", stream,
              {{local_nan, 1.0f}, {local_inf, -local_inf}},
              {{1.0f, 2.0f}, {0.0f, 1.0f}},
              {2},
              ACL_COMPLEX64, ACL_COMPLEX64, ACL_COMPLEX64, true) && allPass;

  // 3) Inplace API 路径
  allPass = RunInplaceMulTest("aclnnInplaceMul_variant", stream,
                              {1.0f, -2.0f, 0.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f},
                              {-1.0f, 2.0f, 3.0f, -4.0f, 0.0f, 1.5f, -2.0f, 0.5f},
                              {4, 2}) && allPass;

  allPass = RunInplaceMulsTest("aclnnInplaceMuls_variant", stream,
                               {1.0f, -2.0f, 0.5f, 3.0f, -4.5f, 5.0f, 0.0f, 7.0f},
                               -2.0f,
                               {4, 2}) && allPass;

  // 4) 异常输入路径
  allPass = RunNullptrTest("nullptr_input_check") && allPass;
  allPass = RunInvalidShapeTest("invalid_shape_check") && allPass;
  allPass = RunMulNullptrVariantTest("nullptr_other_check", false, true, false) && allPass;
  allPass = RunMulNullptrVariantTest("nullptr_out_check", false, false, true) && allPass;
  allPass = RunWrongOutShapeTest("wrong_out_shape_check") && allPass;
  allPass = RunMulsNullptrScalarTest("muls_null_scalar_check") && allPass;
  allPass = RunInplaceMulNullptrOtherTest("inplace_mul_null_other_check") && allPass;
  allPass = RunInplaceMulsNullptrScalarTest("inplace_muls_null_scalar_check") && allPass;

  // 4b) 定点覆盖 dtype 参数检查分支（self/other/out 分离验证）
  allPass = RunMulsTypedTest<float, float, uint32_t>("dtype_check_muls_out_uint32", stream,
              {1.0f, 2.0f, 3.0f, 4.0f},
              2.0f,
              {2, 2},
              ACL_FLOAT, ACL_FLOAT, ACL_UINT32, false) && allPass;

  allPass = RunMulTypedTest<float, uint32_t, float>("dtype_check_mul_other_uint32", stream,
              {1.0f, 2.0f, 3.0f, 4.0f},
              {1u, 2u, 3u, 4u},
              {2, 2},
              ACL_FLOAT, ACL_UINT32, ACL_FLOAT,
              false) && allPass;

  allPass = RunMulTypedTest<float, float, uint32_t>("dtype_check_mul_out_uint32", stream,
              {1.0f, 2.0f, 3.0f, 4.0f},
              {1.0f, 2.0f, 3.0f, 4.0f},
              {2, 2},
              ACL_FLOAT, ACL_FLOAT, ACL_UINT32,
              false) && allPass;

  allPass = RunInplaceMulTypedTest<uint32_t, float>("dtype_check_inplace_mul_self_uint32", stream,
              {1u, 2u, 3u, 4u},
              {1.0f, 2.0f, 3.0f, 4.0f},
              {2, 2},
              ACL_UINT32, ACL_FLOAT, false) && allPass;

  allPass = RunInplaceMulTypedTest<float, uint32_t>("dtype_check_inplace_mul_other_uint32", stream,
              {1.0f, 2.0f, 3.0f, 4.0f},
              {1u, 2u, 3u, 4u},
              {2, 2},
              ACL_FLOAT, ACL_UINT32, false) && allPass;

  // 4c) Promote 失败路径探测（不以 ret 判定失败，尽量触发 0/n 分支）
  allPass = RunMulsPromoteProbeTyped<uint8_t, uint8_t, int16_t>("probe_muls_bool_int16", 
              {1, 0, 1, 0},
              static_cast<int16_t>(2),
              {2, 2},
              ACL_BOOL, ACL_INT16, ACL_BOOL) && allPass;

  allPass = RunMulsPromoteProbeTyped<uint8_t, uint8_t, uint32_t>("probe_muls_bool_uint32", 
              {1, 0, 1, 0},
              static_cast<uint32_t>(2),
              {2, 2},
              ACL_BOOL, ACL_UINT32, ACL_BOOL) && allPass;

  allPass = RunMulPromoteProbeTyped<uint8_t, int16_t, uint8_t>("probe_mul_bool_int16_to_bool",
              {1, 0, 1, 0},
              {1, 2, 3, 4},
              {2, 2},
              ACL_BOOL, ACL_INT16, ACL_BOOL) && allPass;

  typedef std::complex<float> c64_probe;
  typedef std::complex<double> c128_probe;
  allPass = RunMulPromoteProbeTyped<uint8_t, c64_probe, c64_probe>("probe_mul_bool_complex64",
              {1, 0, 1, 0},
              {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, -1.0f}, {0.0f, 2.0f}},
              {2, 2},
              ACL_BOOL, ACL_COMPLEX64, ACL_COMPLEX64) && allPass;

  allPass = RunMulPromoteProbeTyped<int8_t, c128_probe, c128_probe>("probe_mul_int8_complex128",
              {1, -2, 3, -4},
              {{1.0, 2.0}, {3.0, 4.0}, {5.0, -1.0}, {0.0, 2.0}},
              {2, 2},
              ACL_INT8, ACL_COMPLEX128, ACL_COMPLEX128) && allPass;

  allPass = RunInplaceMulPromoteProbeTyped<uint8_t, int16_t>("probe_inplace_mul_bool_int16",
              {1, 0, 1, 0},
              {1, 2, 3, 4},
              {2, 2},
              ACL_BOOL, ACL_INT16) && allPass;

  allPass = RunInplaceMulPromoteProbeTyped<uint8_t, c64_probe>("probe_inplace_mul_bool_complex64",
              {1, 0, 1, 0},
              {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, -1.0f}, {0.0f, 2.0f}},
              {2, 2},
              ACL_BOOL, ACL_COMPLEX64) && allPass;

  // 测试 1D 广播到 2D
  std::vector<int64_t> shapeA_1 = {3};
  std::vector<int64_t> shapeB_1 = {2, 3};
  allPass = RunMulBroadcastTypedTest<float, float, float>("Broadcast_1D_2D", stream, 
              {1.0, 2.0, 3.0},                       // A: [3]
              {1.0, 1.0, 1.0, 2.0, 2.0, 2.0},       // B: [2, 3]
              shapeA_1, shapeB_1, {2, 3},
              ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true) && allPass;

  // 测试双向多维广播
  std::vector<int64_t> shapeA_2 = {2, 1, 3};
  std::vector<int64_t> shapeB_2 = {1, 4, 3};
  // 构造数据时需注意元素个数：A 为 6 个，B 为 12 个
  allPass = RunMulBroadcastTypedTest<float, float, float>("Broadcast_Complex", stream, 
              {1, 2, 3, 4, 5, 6},                   // A
              {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4}, // B
              shapeA_2, shapeB_2, {2, 4, 3},
              ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true) && allPass;

  // 测试不可广播场景 (EXPECT_FAIL)
  std::vector<int64_t> shapeA_fail = {2, 3};
  std::vector<int64_t> shapeB_fail = {2, 4};
  // 这里最后传 false，表示我们预期这个用例会失败
  allPass = RunMulBroadcastTypedTest<float, float, float>("Incompatible_Shapes", stream, 
              {1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1}, 
              shapeA_fail, shapeB_fail, {2, 3},
              ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false) && allPass;

  allPass = RunInplaceMulBroadcastTypedTest<float, float>("Inplace_Incompatible_Shapes", stream,
              {1, 2, 3, 4, 5, 6},
              {1, 1, 1, 1, 1, 1, 1, 1},
              {2, 3}, {2, 4},
              ACL_FLOAT, ACL_FLOAT, false) && allPass;

  // 覆盖 Size 为 0 的场景
  std::vector<int64_t> shapeEmpty = {2, 0, 3};
  allPass = RunMulBroadcastTypedTest<float, float, float>("Edge_EmptyTensor", stream, 
              {}, {}, // 输入数据为空
              shapeEmpty, shapeEmpty, shapeEmpty,
              ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true) && allPass;
  
  // 覆盖维度大于 4 的高维逻辑 (5维)
  std::vector<int64_t> shape5D_A = {2, 2, 1, 2, 1};
  std::vector<int64_t> shape5D_B = {2, 2, 1, 2, 1};
  std::vector<float> data5D(8, 1.0f);
  allPass = RunMulBroadcastTypedTest<float, float, float>("Edge_HighRank_5D", stream, 
              data5D, data5D, 
              shape5D_A, shape5D_B, shape5D_A,
              ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true) && allPass;

  // 覆盖 NaN 和 Inf 的计算逻辑
  float my_nan = std::numeric_limits<float>::quiet_NaN();
  float my_inf = std::numeric_limits<float>::infinity();

  allPass = RunMulBroadcastTypedTest<float, float, float>("Edge_NaN_Inf", stream, 
              {1.0f, 0.0f, my_nan, my_inf},   // Input A
              {my_inf, my_inf, 1.0f, my_inf}, // Input B
              {4}, {4}, {4},
              ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true) && allPass;

  float max_float = std::numeric_limits<float>::max();
  // max * 2.0 应该产生 Inf
  allPass = RunMulBroadcastTypedTest<float, float, float>("Edge_Overflow", stream, 
              {max_float}, {2.0f}, 
              {1}, {1}, {1},
              ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true) && allPass;

  // 提升 l0op::Mul 内 BroadcastInfer 覆盖：补充更多不同 rank / singleton / 零维组合
  allPass = RunMulBroadcastTypedTest<float, float, float>("Edge_Broadcast_Scalar_To3D", stream,
              {2.0f},
              {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
              {}, {1, 2, 3}, {1, 2, 3},
              ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true) && allPass;

  allPass = RunMulBroadcastTypedTest<float, float, float>("Edge_Broadcast_2D_4D", stream,
              {1.0f, -2.0f, 3.0f},
              {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
              {1, 3}, {2, 1, 1, 3}, {2, 1, 1, 3},
              ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true) && allPass;

  allPass = RunMulBroadcastTypedTest<float, float, float>("Edge_Broadcast_4D_CrossSingleton", stream,
              {1, 2, 3, 4, 5, 6},
              {1, 2, 3, 4, 5, 6, 7, 8},
              {1, 3, 1, 2}, {2, 1, 4, 1}, {2, 3, 4, 2},
              ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true) && allPass;

  allPass = RunMulBroadcastTypedTest<float, float, float>("Edge_Broadcast_6D_HighRank", stream,
              {1, 2, 3, 4},
              {1, 2, 3, 4, 5, 6},
              {2, 1, 1, 2, 1, 1}, {1, 3, 1, 1, 2, 1}, {2, 3, 1, 2, 2, 1},
              ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true) && allPass;

  allPass = RunMulBroadcastTypedTest<float, float, float>("Edge_Broadcast_ZeroDim_4D", stream,
              {},
              {},
              {1, 0, 3, 1}, {2, 0, 1, 5}, {2, 0, 3, 5},
              ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true) && allPass;

  allPass = RunMulBroadcastTypedTest<float, float, float>("Edge_Broadcast_ZeroDim_5D", stream,
              {},
              {},
              {2, 1, 0, 4, 1}, {1, 3, 0, 1, 5}, {2, 3, 0, 4, 5},
              ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true) && allPass;
  
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  LOG_PRINT("Overall result: %s\n", allPass ? "PASS" : "FAIL");
  return allPass ? 0 : 1;
}
