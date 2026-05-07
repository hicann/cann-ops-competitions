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
#include <type_traits>
#include <cstdint>
#include <cstring>
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

static inline float Float16ToFloat(uint16_t value) {
  unsigned int sign = (value >> 15) & 0x1;
  unsigned int exponent = (value >> 10) & 0x1f;
  unsigned int mantissa = value & 0x3ff;

  float result;
  if (exponent == 0) {
    result = mantissa * 0.0000019073486328125f;
  } else if (exponent == 31) {
    result = (mantissa == 0) ? 1.0f / 0.0f : 0.0f / 0.0f;
  } else {
    result = (1.0f + mantissa * 0.0009765625f) * std::pow(2.0f, static_cast<int>(exponent) - 15);
  }
  return sign ? -result : result;
}

static inline uint16_t FloatToFloat16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  uint16_t sign = (bits >> 16) & 0x8000;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffff;

  if (exponent <= 0) {
    return sign;
  }
  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00);
  }
  return static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
}

static inline float BFloat16ToFloat(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

static inline uint16_t FloatToBFloat16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

bool AlmostEqual(double expected, double actual, double atol, double rtol) {
  if (std::isnan(expected) && std::isnan(actual)) {
    return true;
  }
  if (std::isinf(expected) && std::isinf(actual)) {
    return (expected > 0) == (actual > 0);
  }
  return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

template <typename T>
bool ValueEqual(T expected, T actual, double atol, double rtol) {
  if constexpr (std::is_floating_point<T>::value) {
    return AlmostEqual(static_cast<double>(expected), static_cast<double>(actual), atol, rtol);
  }
  return expected == actual;
}

template <typename T>
int CreateAclTensorWithFormat(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                              void** deviceAddr, aclDataType dataType, aclFormat format,
                              aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  *deviceAddr = nullptr;
  auto ret = ACL_SUCCESS;
  if (size > 0) {
    // 调用aclrtMalloc申请device侧内存
    ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  }

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
                            shape.data(), shape.size(), *deviceAddr);
  CHECK_RET(*tensor != nullptr,
            LOG_PRINT("aclCreateTensor failed. dtype=%d\n", static_cast<int>(dataType));
            if (*deviceAddr != nullptr) {
              aclrtFree(*deviceAddr);
              *deviceAddr = nullptr;
            }
            return 1);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  return CreateAclTensorWithFormat(hostData, shape, deviceAddr, dataType, aclFormat::ACL_FORMAT_ND, tensor);
}

template <typename T>
bool ComplexAlmostEqual(const std::complex<T>& expected, const std::complex<T>& actual, double atol,
                        double rtol) {
  return AlmostEqual(static_cast<double>(expected.real()), static_cast<double>(actual.real()), atol, rtol) &&
         AlmostEqual(static_cast<double>(expected.imag()), static_cast<double>(actual.imag()), atol, rtol);
}

int RunMulTest(const char* caseName, const std::vector<float>& selfHostData, const std::vector<float>& otherHostData,
               const std::vector<int64_t>& shape, aclrtStream stream) {
  int ret;
  const int64_t size = GetShapeSize(shape);
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<float> outHostData(size, 0);
  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMulGetWorkspaceSize ret=%d\n", caseName, ret); return 1);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: workspace alloc ret=%d\n", caseName, ret); return 1);
  }

  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMul ret=%d\n", caseName, ret); return 1);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: sync ret=%d\n", caseName, ret); return 1);

  ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(float), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: memcpy ret=%d\n", caseName, ret); return 1);

  int mismatch = 0;
  for (int64_t i = 0; i < size; i++) {
    const double expected = static_cast<double>(selfHostData[i]) * static_cast<double>(otherHostData[i]);
    if (!AlmostEqual(expected, outHostData[i], 1e-5, 1e-5)) {
      mismatch++;
      LOG_PRINT("  mismatch[%ld]: expected=%f, actual=%f\n", i, expected, outHostData[i]);
    }
  }
  LOG_PRINT(mismatch == 0 ? "[PASS] %s\n" : "[FAIL] %s: mismatch=%d\n", caseName, mismatch);

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  return mismatch > 0 ? 1 : 0;
}

template <typename T>
int RunMulTypedTest(const char* caseName, const std::vector<T>& selfHostData, const std::vector<T>& otherHostData,
                   const std::vector<int64_t>& shape, aclDataType dataType, aclrtStream stream, double atol,
                   double rtol) {
  int ret;
  const int64_t size = GetShapeSize(shape);
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<T> outHostData(size, static_cast<T>(0));
  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, dataType, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, dataType, &other);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, dataType, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMulGetWorkspaceSize ret=%d\n", caseName, ret); return 1);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: workspace alloc ret=%d\n", caseName, ret); return 1);
  }

  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMul ret=%d\n", caseName, ret); return 1);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: sync ret=%d\n", caseName, ret); return 1);

  ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(T), outDeviceAddr, size * sizeof(T),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: memcpy ret=%d\n", caseName, ret); return 1);

  int mismatch = 0;
  for (int64_t i = 0; i < size; i++) {
    const T expected = static_cast<T>(selfHostData[i] * otherHostData[i]);
    if (!ValueEqual(expected, outHostData[i], atol, rtol)) {
      mismatch++;
      LOG_PRINT("  mismatch[%ld]: expected=%f, actual=%f\n", i, static_cast<double>(expected),
                static_cast<double>(outHostData[i]));
    }
  }
  LOG_PRINT(mismatch == 0 ? "[PASS] %s\n" : "[FAIL] %s: mismatch=%d\n", caseName, mismatch);

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  return mismatch > 0 ? 1 : 0;
}

int RunMulBoolTest(const char* caseName, const std::vector<uint8_t>& selfHostData,
                   const std::vector<uint8_t>& otherHostData, const std::vector<int64_t>& shape,
                   aclrtStream stream) {
  int ret;
  const int64_t size = GetShapeSize(shape);
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<uint8_t> outHostData(size, static_cast<uint8_t>(0));
  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_BOOL, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, aclDataType::ACL_BOOL, &other);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_BOOL, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMulGetWorkspaceSize ret=%d\n", caseName, ret); return 1);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: workspace alloc ret=%d\n", caseName, ret); return 1);
  }

  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMul ret=%d\n", caseName, ret); return 1);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: sync ret=%d\n", caseName, ret); return 1);

  ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(uint8_t), outDeviceAddr, size * sizeof(uint8_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: memcpy ret=%d\n", caseName, ret); return 1);

  int mismatch = 0;
  for (int64_t i = 0; i < size; i++) {
    const uint8_t expected = static_cast<uint8_t>((selfHostData[i] != 0) && (otherHostData[i] != 0));
    if (outHostData[i] != expected) {
      mismatch++;
      LOG_PRINT("  mismatch[%ld]: expected=%d, actual=%d\n", i, static_cast<int>(expected),
                static_cast<int>(outHostData[i]));
    }
  }
  LOG_PRINT(mismatch == 0 ? "[PASS] %s\n" : "[FAIL] %s: mismatch=%d\n", caseName, mismatch);

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  return mismatch > 0 ? 1 : 0;
}

int RunMulFp16Test(const char* caseName, const std::vector<float>& selfFp32, const std::vector<float>& otherFp32,
                   const std::vector<int64_t>& shape, aclrtStream stream) {
  int ret;
  const int64_t size = GetShapeSize(shape);
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<uint16_t> selfHostData(size, 0);
  std::vector<uint16_t> otherHostData(size, 0);
  std::vector<uint16_t> outHostData(size, 0);
  for (int64_t i = 0; i < size; i++) {
    selfHostData[i] = FloatToFloat16(selfFp32[i]);
    otherHostData[i] = FloatToFloat16(otherFp32[i]);
  }

  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT16, &other);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMulGetWorkspaceSize ret=%d\n", caseName, ret); return 1);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: workspace alloc ret=%d\n", caseName, ret); return 1);
  }

  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMul ret=%d\n", caseName, ret); return 1);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: sync ret=%d\n", caseName, ret); return 1);

  ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(uint16_t), outDeviceAddr, size * sizeof(uint16_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: memcpy ret=%d\n", caseName, ret); return 1);

  int mismatch = 0;
  for (int64_t i = 0; i < size; i++) {
    const float expected = Float16ToFloat(selfHostData[i]) * Float16ToFloat(otherHostData[i]);
    const float actual = Float16ToFloat(outHostData[i]);
    if (!AlmostEqual(expected, actual, 1e-3, 1e-3)) {
      mismatch++;
      LOG_PRINT("  mismatch[%ld]: expected=%f, actual=%f\n", i, expected, actual);
    }
  }
  LOG_PRINT(mismatch == 0 ? "[PASS] %s\n" : "[FAIL] %s: mismatch=%d\n", caseName, mismatch);

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  return mismatch > 0 ? 1 : 0;
}

int RunMulBf16Test(const char* caseName, const std::vector<float>& selfFp32, const std::vector<float>& otherFp32,
                   const std::vector<int64_t>& shape, aclrtStream stream) {
  int ret;
  const int64_t size = GetShapeSize(shape);
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<uint16_t> selfHostData(size, 0);
  std::vector<uint16_t> otherHostData(size, 0);
  std::vector<uint16_t> outHostData(size, 0);
  for (int64_t i = 0; i < size; i++) {
    selfHostData[i] = FloatToBFloat16(selfFp32[i]);
    otherHostData[i] = FloatToBFloat16(otherFp32[i]);
  }

  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_BF16, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, aclDataType::ACL_BF16, &other);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_BF16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMulGetWorkspaceSize ret=%d\n", caseName, ret); return 1);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: workspace alloc ret=%d\n", caseName, ret); return 1);
  }

  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMul ret=%d\n", caseName, ret); return 1);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: sync ret=%d\n", caseName, ret); return 1);

  ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(uint16_t), outDeviceAddr, size * sizeof(uint16_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: memcpy ret=%d\n", caseName, ret); return 1);

  int mismatch = 0;
  for (int64_t i = 0; i < size; i++) {
    const float expected = BFloat16ToFloat(selfHostData[i]) * BFloat16ToFloat(otherHostData[i]);
    const float actual = BFloat16ToFloat(outHostData[i]);
    if (!AlmostEqual(expected, actual, 1e-2, 1e-2)) {
      mismatch++;
      LOG_PRINT("  mismatch[%ld]: expected=%f, actual=%f\n", i, expected, actual);
    }
  }
  LOG_PRINT(mismatch == 0 ? "[PASS] %s\n" : "[FAIL] %s: mismatch=%d\n", caseName, mismatch);

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  return mismatch > 0 ? 1 : 0;
}

int RunMulFp16Fp32MixTest(const char* caseName, const std::vector<float>& selfFp32,
                          const std::vector<float>& otherFp32, const std::vector<int64_t>& shape,
                          bool selfIsFp16, aclrtStream stream) {
  int ret;
  const int64_t size = GetShapeSize(shape);
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<uint16_t> fp16HostData(size, 0);
  for (int64_t i = 0; i < size; i++) {
    fp16HostData[i] = FloatToFloat16(selfIsFp16 ? selfFp32[i] : otherFp32[i]);
  }
  std::vector<float> fp32HostData = selfIsFp16 ? otherFp32 : selfFp32;
  std::vector<float> outHostData(size, 0.0f);

  if (selfIsFp16) {
    ret = CreateAclTensor(fp16HostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
    CHECK_RET(ret == ACL_SUCCESS, return 1);
    ret = CreateAclTensor(fp32HostData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
    CHECK_RET(ret == ACL_SUCCESS, return 1);
  } else {
    ret = CreateAclTensor(fp32HostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return 1);
    ret = CreateAclTensor(fp16HostData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT16, &other);
    CHECK_RET(ret == ACL_SUCCESS, return 1);
  }
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMulGetWorkspaceSize ret=%d\n", caseName, ret); return 1);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: workspace alloc ret=%d\n", caseName, ret); return 1);
  }

  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMul ret=%d\n", caseName, ret); return 1);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: sync ret=%d\n", caseName, ret); return 1);

  ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(float), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: memcpy ret=%d\n", caseName, ret); return 1);

  int mismatch = 0;
  for (int64_t i = 0; i < size; i++) {
    const float left = selfIsFp16 ? Float16ToFloat(fp16HostData[i]) : fp32HostData[i];
    const float right = selfIsFp16 ? fp32HostData[i] : Float16ToFloat(fp16HostData[i]);
    const float expected = left * right;
    if (!AlmostEqual(expected, outHostData[i], 1e-4, 1e-4)) {
      mismatch++;
      LOG_PRINT("  mismatch[%ld]: expected=%f, actual=%f\n", i, expected, outHostData[i]);
    }
  }
  LOG_PRINT(mismatch == 0 ? "[PASS] %s\n" : "[FAIL] %s: mismatch=%d\n", caseName, mismatch);

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  return mismatch > 0 ? 1 : 0;
}

int RunMulBf16Fp32MixTest(const char* caseName, const std::vector<float>& selfFp32,
                          const std::vector<float>& otherFp32, const std::vector<int64_t>& shape,
                          bool selfIsBf16, aclrtStream stream) {
  int ret;
  const int64_t size = GetShapeSize(shape);
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<uint16_t> bf16HostData(size, 0);
  for (int64_t i = 0; i < size; i++) {
    bf16HostData[i] = FloatToBFloat16(selfIsBf16 ? selfFp32[i] : otherFp32[i]);
  }
  std::vector<float> fp32HostData = selfIsBf16 ? otherFp32 : selfFp32;
  std::vector<float> outHostData(size, 0.0f);

  if (selfIsBf16) {
    ret = CreateAclTensor(bf16HostData, shape, &selfDeviceAddr, aclDataType::ACL_BF16, &self);
    CHECK_RET(ret == ACL_SUCCESS, return 1);
    ret = CreateAclTensor(fp32HostData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
    CHECK_RET(ret == ACL_SUCCESS, return 1);
  } else {
    ret = CreateAclTensor(fp32HostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return 1);
    ret = CreateAclTensor(bf16HostData, shape, &otherDeviceAddr, aclDataType::ACL_BF16, &other);
    CHECK_RET(ret == ACL_SUCCESS, return 1);
  }
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMulGetWorkspaceSize ret=%d\n", caseName, ret); return 1);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: workspace alloc ret=%d\n", caseName, ret); return 1);
  }

  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMul ret=%d\n", caseName, ret); return 1);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: sync ret=%d\n", caseName, ret); return 1);

  ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(float), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: memcpy ret=%d\n", caseName, ret); return 1);

  int mismatch = 0;
  for (int64_t i = 0; i < size; i++) {
    const float left = selfIsBf16 ? BFloat16ToFloat(bf16HostData[i]) : fp32HostData[i];
    const float right = selfIsBf16 ? fp32HostData[i] : BFloat16ToFloat(bf16HostData[i]);
    const float expected = left * right;
    if (!AlmostEqual(expected, outHostData[i], 1e-4, 1e-4)) {
      mismatch++;
      LOG_PRINT("  mismatch[%ld]: expected=%f, actual=%f\n", i, expected, outHostData[i]);
    }
  }
  LOG_PRINT(mismatch == 0 ? "[PASS] %s\n" : "[FAIL] %s: mismatch=%d\n", caseName, mismatch);

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  return mismatch > 0 ? 1 : 0;
}

int RunMulsTest(const char* caseName, const std::vector<float>& selfHostData, const std::vector<int64_t>& shape,
                float scalarValue, aclrtStream stream) {
  int ret;
  const int64_t size = GetShapeSize(shape);
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclScalar* scalar = nullptr;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<float> outHostData(size, 0);
  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  scalar = aclCreateScalar(&scalarValue, aclDataType::ACL_FLOAT);
  ret = aclnnMulsGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMulsGetWorkspaceSize ret=%d\n", caseName, ret); return 1);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: workspace alloc ret=%d\n", caseName, ret); return 1);
  }

  ret = aclnnMuls(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMuls ret=%d\n", caseName, ret); return 1);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: sync ret=%d\n", caseName, ret); return 1);

  ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(float), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: memcpy ret=%d\n", caseName, ret); return 1);

  int mismatch = 0;
  for (int64_t i = 0; i < size; i++) {
    const double expected = static_cast<double>(selfHostData[i]) * static_cast<double>(scalarValue);
    if (!AlmostEqual(expected, outHostData[i], 1e-5, 1e-5)) {
      mismatch++;
      LOG_PRINT("  mismatch[%ld]: expected=%f, actual=%f\n", i, expected, outHostData[i]);
    }
  }
  LOG_PRINT(mismatch == 0 ? "[PASS] %s\n" : "[FAIL] %s: mismatch=%d\n", caseName, mismatch);

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyScalar(scalar);
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  return mismatch > 0 ? 1 : 0;
}

int RunInplaceMulsTest(const char* caseName, const std::vector<float>& selfHostData,
                       const std::vector<int64_t>& shape, float scalarValue, aclrtStream stream) {
  int ret;
  const int64_t size = GetShapeSize(shape);
  void* selfDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* scalar = nullptr;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<float> inplaceOutHost(size, 0.0f);
  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  scalar = aclCreateScalar(&scalarValue, aclDataType::ACL_FLOAT);
  CHECK_RET(scalar != nullptr, LOG_PRINT("[FAIL] %s: create scalar failed\n", caseName); return 1);
  ret = aclnnInplaceMulsGetWorkspaceSize(self, scalar, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[FAIL] %s: aclnnInplaceMulsGetWorkspaceSize ret=%d\n", caseName, ret); return 1);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: workspace alloc ret=%d\n", caseName, ret); return 1);
  }

  ret = aclnnInplaceMuls(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnInplaceMuls ret=%d\n", caseName, ret); return 1);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: sync ret=%d\n", caseName, ret); return 1);

  ret = aclrtMemcpy(inplaceOutHost.data(), inplaceOutHost.size() * sizeof(float), selfDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: memcpy ret=%d\n", caseName, ret); return 1);

  int mismatch = 0;
  for (int64_t i = 0; i < size; i++) {
    const double expected = static_cast<double>(selfHostData[i]) * static_cast<double>(scalarValue);
    if (!AlmostEqual(expected, inplaceOutHost[i], 1e-5, 1e-5)) {
      mismatch++;
      LOG_PRINT("  mismatch[%ld]: expected=%f, actual=%f\n", i, expected, inplaceOutHost[i]);
    }
  }
  LOG_PRINT(mismatch == 0 ? "[PASS] %s\n" : "[FAIL] %s: mismatch=%d\n", caseName, mismatch);

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyScalar(scalar);
  aclrtFree(selfDeviceAddr);
  return mismatch > 0 ? 1 : 0;
}

int RunMulsBf16DoubleScalarTest(const char* caseName, const std::vector<float>& selfFp32,
                                const std::vector<int64_t>& shape, double scalarValue, aclrtStream stream) {
  int ret;
  const int64_t size = GetShapeSize(shape);
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclScalar* scalar = nullptr;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<uint16_t> selfHostData(size, 0);
  for (int64_t i = 0; i < size; i++) {
    selfHostData[i] = FloatToBFloat16(selfFp32[i]);
  }
  std::vector<float> outHostData(size, 0.0f);

  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_BF16, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  scalar = aclCreateScalar(&scalarValue, aclDataType::ACL_DOUBLE);
  CHECK_RET(scalar != nullptr, LOG_PRINT("[FAIL] %s: create scalar failed\n", caseName); return 1);
  ret = aclnnMulsGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMulsGetWorkspaceSize ret=%d\n", caseName, ret); return 1);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: workspace alloc ret=%d\n", caseName, ret); return 1);
  }

  ret = aclnnMuls(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMuls ret=%d\n", caseName, ret); return 1);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: sync ret=%d\n", caseName, ret); return 1);

  ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(float), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: memcpy ret=%d\n", caseName, ret); return 1);

  int mismatch = 0;
  for (int64_t i = 0; i < size; i++) {
    const double expected = static_cast<double>(BFloat16ToFloat(selfHostData[i])) * scalarValue;
    if (!AlmostEqual(expected, outHostData[i], 1e-3, 1e-3)) {
      mismatch++;
      LOG_PRINT("  mismatch[%ld]: expected=%f, actual=%f\n", i, expected, outHostData[i]);
    }
  }
  LOG_PRINT(mismatch == 0 ? "[PASS] %s\n" : "[FAIL] %s: mismatch=%d\n", caseName, mismatch);

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyScalar(scalar);
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  return mismatch > 0 ? 1 : 0;
}

int RunMulBroadcastTest(const char* caseName, const std::vector<float>& selfHostData,
                        const std::vector<int64_t>& selfShape, const std::vector<float>& otherHostData,
                        const std::vector<int64_t>& otherShape, const std::vector<int64_t>& outShape,
                        const std::vector<float>& expectedHostData, aclrtStream stream) {
  int ret;
  const int64_t outSize = GetShapeSize(outShape);
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<float> outHostData(outSize, 0.0f);
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMulGetWorkspaceSize ret=%d\n", caseName, ret); return 1);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: workspace alloc ret=%d\n", caseName, ret); return 1);
  }

  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMul ret=%d\n", caseName, ret); return 1);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: sync ret=%d\n", caseName, ret); return 1);

  ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(float), outDeviceAddr, outSize * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: memcpy ret=%d\n", caseName, ret); return 1);

  int mismatch = 0;
  for (int64_t i = 0; i < outSize; i++) {
    if (!AlmostEqual(expectedHostData[i], outHostData[i], 1e-5, 1e-5)) {
      mismatch++;
      LOG_PRINT("  mismatch[%ld]: expected=%f, actual=%f\n", i, expectedHostData[i], outHostData[i]);
    }
  }
  LOG_PRINT(mismatch == 0 ? "[PASS] %s\n" : "[FAIL] %s: mismatch=%d\n", caseName, mismatch);

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  return mismatch > 0 ? 1 : 0;
}

int RunInplaceMulTest(const char* caseName, const std::vector<float>& selfHostData,
                     const std::vector<float>& otherHostData, const std::vector<int64_t>& shape,
                     aclrtStream stream) {
  int ret;
  const int64_t size = GetShapeSize(shape);
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<float> inplaceOutHost(size, 0.0f);
  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnInplaceMulGetWorkspaceSize(self, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[FAIL] %s: aclnnInplaceMulGetWorkspaceSize ret=%d\n", caseName, ret); return 1);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: workspace alloc ret=%d\n", caseName, ret); return 1);
  }

  ret = aclnnInplaceMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnInplaceMul ret=%d\n", caseName, ret); return 1);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: sync ret=%d\n", caseName, ret); return 1);

  ret = aclrtMemcpy(inplaceOutHost.data(), inplaceOutHost.size() * sizeof(float), selfDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: memcpy ret=%d\n", caseName, ret); return 1);

  int mismatch = 0;
  for (int64_t i = 0; i < size; i++) {
    const double expected = static_cast<double>(selfHostData[i]) * static_cast<double>(otherHostData[i]);
    if (!AlmostEqual(expected, inplaceOutHost[i], 1e-5, 1e-5)) {
      mismatch++;
      LOG_PRINT("  mismatch[%ld]: expected=%f, actual=%f\n", i, expected, inplaceOutHost[i]);
    }
  }
  LOG_PRINT(mismatch == 0 ? "[PASS] %s\n" : "[FAIL] %s: mismatch=%d\n", caseName, mismatch);

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  return mismatch > 0 ? 1 : 0;
}

int RunMulNullptrGuardTest(const char* caseName, aclrtStream stream) {
  int ret;
  (void)stream;
  std::vector<float> otherHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> outHostData(4, 0.0f);
  std::vector<int64_t> shape = {2, 2};
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnMulGetWorkspaceSize(nullptr, other, out, &workspaceSize, &executor);
  if (ret == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s: expected failure but got ACL_SUCCESS\n", caseName);
    aclDestroyTensor(other);
    aclDestroyTensor(out);
    aclrtFree(otherDeviceAddr);
    aclrtFree(outDeviceAddr);
    return 1;
  }

  LOG_PRINT("[PASS] %s (ret=%d)\n", caseName, ret);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  return 0;
}

int RunMulInvalidBroadcastGuardTest(const char* caseName, aclrtStream stream) {
  int ret;
  (void)stream;
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> otherHostData = {10, 20, 30, 40};
  std::vector<float> outHostData(6, 0.0f);
  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> otherShape = {2, 2};
  std::vector<int64_t> outShape = {2, 3};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  if (ret == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s: expected invalid-broadcast failure but got ACL_SUCCESS\n", caseName);
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    aclrtFree(outDeviceAddr);
    return 1;
  }

  LOG_PRINT("[PASS] %s (ret=%d)\n", caseName, ret);
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  return 0;
}

int RunMulOutShapeMismatchGuardTest(const char* caseName, aclrtStream stream) {
  int ret;
  (void)stream;
  std::vector<float> selfHostData = {1, 2, 3, 4};
  std::vector<float> otherHostData = {10, 20, 30, 40};
  std::vector<float> outHostData(3, 0.0f);
  std::vector<int64_t> shape = {2, 2};
  std::vector<int64_t> outShape = {3};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  if (ret == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s: expected out-shape mismatch failure but got ACL_SUCCESS\n", caseName);
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    aclrtFree(outDeviceAddr);
    return 1;
  }

  LOG_PRINT("[PASS] %s (ret=%d)\n", caseName, ret);
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  return 0;
}

int RunMulsNullptrGuardTest(const char* caseName, aclrtStream stream) {
  int ret;
  (void)stream;
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> outHostData(4, 0.0f);
  std::vector<int64_t> shape = {2, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  float scalarValue = 2.0f;
  aclScalar* scalar = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  scalar = aclCreateScalar(&scalarValue, aclDataType::ACL_FLOAT);

  ret = aclnnMulsGetWorkspaceSize(self, scalar, nullptr, &workspaceSize, &executor);
  if (ret == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s: expected nullptr failure but got ACL_SUCCESS\n", caseName);
    aclDestroyScalar(scalar);
    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr);
    aclrtFree(outDeviceAddr);
    return 1;
  }

  LOG_PRINT("[PASS] %s (ret=%d)\n", caseName, ret);
  aclDestroyScalar(scalar);
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  return 0;
}

int RunInplaceMulsNullptrGuardTest(const char* caseName, aclrtStream stream) {
  int ret;
  (void)stream;
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> shape = {2, 2};
  void* selfDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnInplaceMulsGetWorkspaceSize(self, nullptr, &workspaceSize, &executor);
  if (ret == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s: expected nullptr failure but got ACL_SUCCESS\n", caseName);
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    return 1;
  }

  LOG_PRINT("[PASS] %s (ret=%d)\n", caseName, ret);
  aclDestroyTensor(self);
  aclrtFree(selfDeviceAddr);
  return 0;
}

int RunInplaceMulInvalidBroadcastGuardTest(const char* caseName, aclrtStream stream) {
  int ret;
  (void)stream;
  std::vector<float> selfHostData = {1, 2, 3, 4};
  std::vector<float> otherHostData = {10, 20, 30, 40, 50, 60};
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> otherShape = {2, 3};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnInplaceMulGetWorkspaceSize(self, other, &workspaceSize, &executor);
  if (ret == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s: expected invalid-broadcast failure but got ACL_SUCCESS\n", caseName);
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    return 1;
  }

  LOG_PRINT("[PASS] %s (ret=%d)\n", caseName, ret);
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  return 0;
}

int RunInplaceMulShapeExpandGuardTest(const char* caseName, aclrtStream stream) {
  int ret;
  (void)stream;
  std::vector<float> selfHostData = {1, 2};
  std::vector<float> otherHostData = {10, 20};
  std::vector<int64_t> selfShape = {2, 1};
  std::vector<int64_t> otherShape = {1, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnInplaceMulGetWorkspaceSize(self, other, &workspaceSize, &executor);
  if (ret == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s: expected shape-expand failure but got ACL_SUCCESS\n", caseName);
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    return 1;
  }

  LOG_PRINT("[PASS] %s (ret=%d)\n", caseName, ret);
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  return 0;
}

int RunMulsFp16ScalarTest(const char* caseName, const std::vector<float>& selfFp32,
                          const std::vector<int64_t>& shape, float scalarValue, aclrtStream stream) {
  int ret;
  const int64_t size = GetShapeSize(shape);
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclScalar* scalar = nullptr;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<uint16_t> selfHostData(size, 0);
  std::vector<uint16_t> outHostData(size, 0);
  for (int64_t i = 0; i < size; i++) {
    selfHostData[i] = FloatToFloat16(selfFp32[i]);
  }

  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  scalar = aclCreateScalar(&scalarValue, aclDataType::ACL_FLOAT);
  CHECK_RET(scalar != nullptr, LOG_PRINT("[FAIL] %s: create scalar failed\n", caseName); return 1);
  ret = aclnnMulsGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMulsGetWorkspaceSize ret=%d\n", caseName, ret); return 1);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: workspace alloc ret=%d\n", caseName, ret); return 1);
  }

  ret = aclnnMuls(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMuls ret=%d\n", caseName, ret); return 1);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: sync ret=%d\n", caseName, ret); return 1);

  ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(uint16_t), outDeviceAddr, size * sizeof(uint16_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: memcpy ret=%d\n", caseName, ret); return 1);

  int mismatch = 0;
  for (int64_t i = 0; i < size; i++) {
    const float expected = Float16ToFloat(selfHostData[i]) * scalarValue;
    const float actual = Float16ToFloat(outHostData[i]);
    if (!AlmostEqual(expected, actual, 1e-3, 1e-3)) {
      mismatch++;
      LOG_PRINT("  mismatch[%ld]: expected=%f, actual=%f\n", i, expected, actual);
    }
  }
  LOG_PRINT(mismatch == 0 ? "[PASS] %s\n" : "[FAIL] %s: mismatch=%d\n", caseName, mismatch);

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyScalar(scalar);
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  return mismatch > 0 ? 1 : 0;
}

int RunInplaceMulsFp16ScalarTest(const char* caseName, const std::vector<float>& selfFp32,
                                 const std::vector<int64_t>& shape, float scalarValue, aclrtStream stream) {
  int ret;
  const int64_t size = GetShapeSize(shape);
  void* selfDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* scalar = nullptr;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<uint16_t> selfHostData(size, 0);
  std::vector<uint16_t> inplaceOutHost(size, 0);
  for (int64_t i = 0; i < size; i++) {
    selfHostData[i] = FloatToFloat16(selfFp32[i]);
  }

  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  scalar = aclCreateScalar(&scalarValue, aclDataType::ACL_FLOAT);
  CHECK_RET(scalar != nullptr, LOG_PRINT("[FAIL] %s: create scalar failed\n", caseName); return 1);
  ret = aclnnInplaceMulsGetWorkspaceSize(self, scalar, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[FAIL] %s: aclnnInplaceMulsGetWorkspaceSize ret=%d\n", caseName, ret); return 1);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: workspace alloc ret=%d\n", caseName, ret); return 1);
  }

  ret = aclnnInplaceMuls(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnInplaceMuls ret=%d\n", caseName, ret); return 1);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: sync ret=%d\n", caseName, ret); return 1);

  ret = aclrtMemcpy(inplaceOutHost.data(), inplaceOutHost.size() * sizeof(uint16_t), selfDeviceAddr,
                    size * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: memcpy ret=%d\n", caseName, ret); return 1);

  int mismatch = 0;
  for (int64_t i = 0; i < size; i++) {
    const float expected = Float16ToFloat(selfHostData[i]) * scalarValue;
    const float actual = Float16ToFloat(inplaceOutHost[i]);
    if (!AlmostEqual(expected, actual, 1e-3, 1e-3)) {
      mismatch++;
      LOG_PRINT("  mismatch[%ld]: expected=%f, actual=%f\n", i, expected, actual);
    }
  }
  LOG_PRINT(mismatch == 0 ? "[PASS] %s\n" : "[FAIL] %s: mismatch=%d\n", caseName, mismatch);

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyScalar(scalar);
  aclrtFree(selfDeviceAddr);
  return mismatch > 0 ? 1 : 0;
}

int RunMulInt32Fp32PromoteTest(const char* caseName, const std::vector<int32_t>& selfHostData,
                               const std::vector<float>& otherHostData, const std::vector<int64_t>& shape,
                               aclrtStream stream) {
  int ret;
  const int64_t size = GetShapeSize(shape);
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<float> outHostData(size, 0.0f);
  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_INT32, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMulGetWorkspaceSize ret=%d\n", caseName, ret); return 1);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: workspace alloc ret=%d\n", caseName, ret); return 1);
  }

  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMul ret=%d\n", caseName, ret); return 1);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: sync ret=%d\n", caseName, ret); return 1);

  ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(float), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: memcpy ret=%d\n", caseName, ret); return 1);

  int mismatch = 0;
  for (int64_t i = 0; i < size; i++) {
    const double expected = static_cast<double>(selfHostData[i]) * static_cast<double>(otherHostData[i]);
    if (!AlmostEqual(expected, outHostData[i], 1e-5, 1e-5)) {
      mismatch++;
      LOG_PRINT("  mismatch[%ld]: expected=%f, actual=%f\n", i, expected, outHostData[i]);
    }
  }
  LOG_PRINT(mismatch == 0 ? "[PASS] %s\n" : "[FAIL] %s: mismatch=%d\n", caseName, mismatch);

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  return mismatch > 0 ? 1 : 0;
}

template <typename T>
int RunMulComplexTest(const char* caseName, const std::vector<std::complex<T>>& selfHostData,
                      const std::vector<std::complex<T>>& otherHostData, const std::vector<int64_t>& shape,
                      aclDataType dataType, aclrtStream stream, double atol, double rtol) {
  int ret;
  const int64_t size = GetShapeSize(shape);
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  std::vector<std::complex<T>> outHostData(size, std::complex<T>(0, 0));
  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, dataType, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, dataType, &other);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, dataType, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMulGetWorkspaceSize ret=%d\n", caseName, ret); return 1);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: workspace alloc ret=%d\n", caseName, ret); return 1);
  }

  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMul ret=%d\n", caseName, ret); return 1);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: sync ret=%d\n", caseName, ret); return 1);

  ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(std::complex<T>), outDeviceAddr,
                    size * sizeof(std::complex<T>), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: memcpy ret=%d\n", caseName, ret); return 1);

  int mismatch = 0;
  for (int64_t i = 0; i < size; i++) {
    const std::complex<T> expected = selfHostData[i] * otherHostData[i];
    if (!ComplexAlmostEqual(expected, outHostData[i], atol, rtol)) {
      mismatch++;
      LOG_PRINT("  mismatch[%ld]: expected=(%f,%f), actual=(%f,%f)\n", i, static_cast<double>(expected.real()),
                static_cast<double>(expected.imag()), static_cast<double>(outHostData[i].real()),
                static_cast<double>(outHostData[i].imag()));
    }
  }
  LOG_PRINT(mismatch == 0 ? "[PASS] %s\n" : "[FAIL] %s: mismatch=%d\n", caseName, mismatch);

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  return mismatch > 0 ? 1 : 0;
}

int RunMulComplex128RouteProbeTest(const char* caseName, aclrtStream stream) {
  int ret;
  const std::vector<int64_t> shape = {2, 2};
  std::vector<std::complex<double>> selfHostData = {
      {1.0, -2.0}, {-3.0, 4.0}, {0.5, 0.25}, {-1.25, -0.75}};
  std::vector<std::complex<double>> otherHostData = {
      {-2.0, 3.0}, {0.5, -1.0}, {2.0, 2.5}, {-0.5, 0.5}};
  const int64_t size = GetShapeSize(shape);
  std::vector<std::complex<double>> outHostData(size, std::complex<double>(0.0, 0.0));

  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_COMPLEX128, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, aclDataType::ACL_COMPLEX128, &other);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_COMPLEX128, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[PASS] %s: getws ret=%d (route covered)\n", caseName, ret);
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    aclrtFree(outDeviceAddr);
    return 0;
  }

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: workspace alloc ret=%d\n", caseName, ret); return 1);
  }

  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMul ret=%d\n", caseName, ret); return 1);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: sync ret=%d\n", caseName, ret); return 1);
  ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(std::complex<double>), outDeviceAddr,
                    size * sizeof(std::complex<double>), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: memcpy ret=%d\n", caseName, ret); return 1);

  int mismatch = 0;
  for (int64_t i = 0; i < size; i++) {
    const std::complex<double> expected = selfHostData[i] * otherHostData[i];
    if (!ComplexAlmostEqual(expected, outHostData[i], 1e-10, 1e-10)) {
      mismatch++;
    }
  }
  LOG_PRINT(mismatch == 0 ? "[PASS] %s\n" : "[FAIL] %s: mismatch=%d\n", caseName, mismatch);

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  return mismatch > 0 ? 1 : 0;
}

int RunMulFormatWarningTest(const char* caseName, aclrtStream stream) {
  int ret;
  const std::vector<int64_t> shape = {1, 1, 2, 2};
  const int64_t size = GetShapeSize(shape);
  std::vector<float> selfHostData = {1.0f, -2.0f, 3.0f, -4.0f};
  std::vector<float> otherHostData = {2.0f, 0.5f, -1.0f, 3.0f};
  std::vector<float> outHostData(size, 0.0f);

  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensorWithFormat(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT,
                                  aclFormat::ACL_FORMAT_NCHW, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMulGetWorkspaceSize ret=%d\n", caseName, ret); return 1);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: workspace alloc ret=%d\n", caseName, ret); return 1);
  }
  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMul ret=%d\n", caseName, ret); return 1);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: sync ret=%d\n", caseName, ret); return 1);
  ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(float), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: memcpy ret=%d\n", caseName, ret); return 1);

  int mismatch = 0;
  for (int64_t i = 0; i < size; i++) {
    const float expected = selfHostData[i] * otherHostData[i];
    if (!AlmostEqual(expected, outHostData[i], 1e-5, 1e-5)) {
      mismatch++;
    }
  }
  LOG_PRINT(mismatch == 0 ? "[PASS] %s\n" : "[FAIL] %s: mismatch=%d\n", caseName, mismatch);

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  return mismatch > 0 ? 1 : 0;
}

int RunMulsFormatWarningTest(const char* caseName, aclrtStream stream) {
  int ret;
  const std::vector<int64_t> shape = {1, 1, 2, 2};
  const int64_t size = GetShapeSize(shape);
  std::vector<float> selfHostData = {1.0f, -2.0f, 3.0f, -4.0f};
  std::vector<float> outHostData(size, 0.0f);
  float scalarValue = 1.75f;

  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclScalar* scalar = nullptr;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensorWithFormat(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT,
                                  aclFormat::ACL_FORMAT_NCHW, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  scalar = aclCreateScalar(&scalarValue, aclDataType::ACL_FLOAT);
  CHECK_RET(scalar != nullptr, LOG_PRINT("[FAIL] %s: create scalar failed\n", caseName); return 1);

  ret = aclnnMulsGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMulsGetWorkspaceSize ret=%d\n", caseName, ret); return 1);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: workspace alloc ret=%d\n", caseName, ret); return 1);
  }

  ret = aclnnMuls(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: aclnnMuls ret=%d\n", caseName, ret); return 1);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: sync ret=%d\n", caseName, ret); return 1);
  ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(float), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s: memcpy ret=%d\n", caseName, ret); return 1);

  int mismatch = 0;
  for (int64_t i = 0; i < size; i++) {
    const float expected = selfHostData[i] * scalarValue;
    if (!AlmostEqual(expected, outHostData[i], 1e-5, 1e-5)) {
      mismatch++;
    }
  }
  LOG_PRINT(mismatch == 0 ? "[PASS] %s\n" : "[FAIL] %s: mismatch=%d\n", caseName, mismatch);

  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyScalar(scalar);
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  return mismatch > 0 ? 1 : 0;
}

int RunMulEmptyTensorFastPathTest(const char* caseName, aclrtStream stream) {
  int ret;
  (void)stream;
  std::vector<int64_t> shape = {0, 2};
  std::vector<float> emptyData;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  uint64_t workspaceSize = 1;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(emptyData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(emptyData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(emptyData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS || workspaceSize != 0) {
    LOG_PRINT("[FAIL] %s: ret=%d workspace=%llu\n", caseName, ret,
              static_cast<unsigned long long>(workspaceSize));
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyTensor(out);
    return 1;
  }

  LOG_PRINT("[PASS] %s\n", caseName);
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  return 0;
}

int RunMulsEmptyTensorFastPathTest(const char* caseName, aclrtStream stream) {
  int ret;
  (void)stream;
  std::vector<int64_t> shape = {0, 2};
  std::vector<float> emptyData;
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  float scalarValue = 2.0f;
  aclScalar* scalar = nullptr;
  uint64_t workspaceSize = 1;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(emptyData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(emptyData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  scalar = aclCreateScalar(&scalarValue, aclDataType::ACL_FLOAT);

  ret = aclnnMulsGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS || workspaceSize != 0) {
    LOG_PRINT("[FAIL] %s: ret=%d workspace=%llu\n", caseName, ret,
              static_cast<unsigned long long>(workspaceSize));
    aclDestroyScalar(scalar);
    aclDestroyTensor(self);
    aclDestroyTensor(out);
    return 1;
  }

  LOG_PRINT("[PASS] %s\n", caseName);
  aclDestroyScalar(scalar);
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  return 0;
}

int RunInplaceMulsEmptyTensorFastPathTest(const char* caseName, aclrtStream stream) {
  int ret;
  (void)stream;
  std::vector<int64_t> shape = {0, 2};
  std::vector<float> emptyData;
  void* selfDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  float scalarValue = 2.0f;
  aclScalar* scalar = nullptr;
  uint64_t workspaceSize = 1;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(emptyData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  scalar = aclCreateScalar(&scalarValue, aclDataType::ACL_FLOAT);

  ret = aclnnInplaceMulsGetWorkspaceSize(self, scalar, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS || workspaceSize != 0) {
    LOG_PRINT("[FAIL] %s: ret=%d workspace=%llu\n", caseName, ret,
              static_cast<unsigned long long>(workspaceSize));
    aclDestroyScalar(scalar);
    aclDestroyTensor(self);
    return 1;
  }

  LOG_PRINT("[PASS] %s\n", caseName);
  aclDestroyScalar(scalar);
  aclDestroyTensor(self);
  return 0;
}

int RunInplaceMulEmptyTensorFastPathTest(const char* caseName, aclrtStream stream) {
  int ret;
  (void)stream;
  std::vector<int64_t> shape = {0, 2};
  std::vector<float> emptyData;
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  uint64_t workspaceSize = 1;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(emptyData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(emptyData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnInplaceMulGetWorkspaceSize(self, other, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS || workspaceSize != 0) {
    LOG_PRINT("[FAIL] %s: ret=%d workspace=%llu\n", caseName, ret,
              static_cast<unsigned long long>(workspaceSize));
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    return 1;
  }

  LOG_PRINT("[PASS] %s\n", caseName);
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  return 0;
}

int RunInplaceMulNullptrGuardTest(const char* caseName, aclrtStream stream) {
  int ret;
  (void)stream;
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> shape = {2, 2};
  void* selfDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnInplaceMulGetWorkspaceSize(nullptr, self, &workspaceSize, &executor);
  if (ret == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s: expected nullptr failure but got ACL_SUCCESS\n", caseName);
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    return 1;
  }

  LOG_PRINT("[PASS] %s (ret=%d)\n", caseName, ret);
  aclDestroyTensor(self);
  aclrtFree(selfDeviceAddr);
  return 0;
}

int RunMulsShapeMismatchGuardTest(const char* caseName, aclrtStream stream) {
  int ret;
  (void)stream;
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> outHostData(3, 0.0f);
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> outShape = {3};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  float scalarValue = 2.0f;
  aclScalar* scalar = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  scalar = aclCreateScalar(&scalarValue, aclDataType::ACL_FLOAT);

  ret = aclnnMulsGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor);
  if (ret == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s: expected shape mismatch failure but got ACL_SUCCESS\n", caseName);
    aclDestroyScalar(scalar);
    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr);
    aclrtFree(outDeviceAddr);
    return 1;
  }

  LOG_PRINT("[PASS] %s (ret=%d)\n", caseName, ret);
  aclDestroyScalar(scalar);
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  return 0;
}

int RunMulNullptrOtherGuardTest(const char* caseName, aclrtStream stream) {
  int ret;
  (void)stream;
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> outHostData(4, 0.0f);
  std::vector<int64_t> shape = {2, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnMulGetWorkspaceSize(self, nullptr, out, &workspaceSize, &executor);
  if (ret == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s: expected nullptr failure but got ACL_SUCCESS\n", caseName);
    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr);
    aclrtFree(outDeviceAddr);
    return 1;
  }

  LOG_PRINT("[PASS] %s (ret=%d)\n", caseName, ret);
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  return 0;
}

int RunMulsNullptrSelfGuardTest(const char* caseName, aclrtStream stream) {
  int ret;
  (void)stream;
  std::vector<float> outHostData(4, 0.0f);
  std::vector<int64_t> shape = {2, 2};
  void* outDeviceAddr = nullptr;
  aclTensor* out = nullptr;
  float scalarValue = 2.0f;
  aclScalar* scalar = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  scalar = aclCreateScalar(&scalarValue, aclDataType::ACL_FLOAT);

  ret = aclnnMulsGetWorkspaceSize(nullptr, scalar, out, &workspaceSize, &executor);
  if (ret == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s: expected nullptr failure but got ACL_SUCCESS\n", caseName);
    aclDestroyScalar(scalar);
    aclDestroyTensor(out);
    aclrtFree(outDeviceAddr);
    return 1;
  }

  LOG_PRINT("[PASS] %s (ret=%d)\n", caseName, ret);
  aclDestroyScalar(scalar);
  aclDestroyTensor(out);
  aclrtFree(outDeviceAddr);
  return 0;
}

int RunInplaceMulsNullptrSelfGuardTest(const char* caseName, aclrtStream stream) {
  int ret;
  (void)stream;
  float scalarValue = 2.0f;
  aclScalar* scalar = aclCreateScalar(&scalarValue, aclDataType::ACL_FLOAT);
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = aclnnInplaceMulsGetWorkspaceSize(nullptr, scalar, &workspaceSize, &executor);
  if (ret == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s: expected nullptr failure but got ACL_SUCCESS\n", caseName);
    aclDestroyScalar(scalar);
    return 1;
  }

  LOG_PRINT("[PASS] %s (ret=%d)\n", caseName, ret);
  aclDestroyScalar(scalar);
  return 0;
}

int RunInplaceMulNullptrOtherGuardTest(const char* caseName, aclrtStream stream) {
  int ret;
  (void)stream;
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> shape = {2, 2};
  void* selfDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnInplaceMulGetWorkspaceSize(self, nullptr, &workspaceSize, &executor);
  if (ret == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s: expected nullptr failure but got ACL_SUCCESS\n", caseName);
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    return 1;
  }

  LOG_PRINT("[PASS] %s (ret=%d)\n", caseName, ret);
  aclDestroyTensor(self);
  aclrtFree(selfDeviceAddr);
  return 0;
}

int RunMulMaxDimSelfGuardTest(const char* caseName, aclrtStream stream) {
  int ret;
  (void)stream;
  std::vector<int64_t> bigShape = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int64_t> smallShape = {1};
  std::vector<float> selfHostData(1, 2.0f);
  std::vector<float> otherHostData(1, 3.0f);
  std::vector<float> outHostData(1, 0.0f);
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(selfHostData, bigShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(otherHostData, smallShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(outHostData, bigShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  if (ret == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s: expected max-dim failure but got ACL_SUCCESS\n", caseName);
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    aclrtFree(outDeviceAddr);
    return 1;
  }

  LOG_PRINT("[PASS] %s (ret=%d)\n", caseName, ret);
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  return 0;
}

int RunMulMaxDimOtherGuardTest(const char* caseName, aclrtStream stream) {
  int ret;
  (void)stream;
  std::vector<int64_t> bigShape = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int64_t> smallShape = {1};
  std::vector<float> selfHostData(1, 2.0f);
  std::vector<float> otherHostData(1, 3.0f);
  std::vector<float> outHostData(1, 0.0f);
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = CreateAclTensor(selfHostData, smallShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(otherHostData, bigShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return 1);
  ret = CreateAclTensor(outHostData, bigShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return 1);

  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  if (ret == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s: expected max-dim failure but got ACL_SUCCESS\n", caseName);
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    aclrtFree(outDeviceAddr);
    return 1;
  }

  LOG_PRINT("[PASS] %s (ret=%d)\n", caseName, ret);
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  return 0;
}

int main() {
  // 1.（固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  int totalFailed = 0;
  totalFailed += RunMulTest("float32_basic", {0, 1, 2, 3, 4, 5, 6, 7}, {1, 1, 1, 2, 2, 2, 3, 3}, {4, 2}, stream);
  totalFailed += RunMulsTest("float32_muls_scalar", {1.0f, -2.0f, 0.5f, 8.0f}, {2, 2}, 2.5f, stream);
  totalFailed += RunMulTypedTest<int32_t>("int32_basic", {1, -2, 3, -4}, {2, 3, -1, 0}, {2, 2},
                                          aclDataType::ACL_INT32, stream, 0.0, 0.0);
  totalFailed += RunMulTypedTest<int16_t>("int16_basic", {1, -2, 30, -4}, {2, 3, -1, 0}, {2, 2},
                                          aclDataType::ACL_INT16, stream, 0.0, 0.0);
  totalFailed += RunMulTypedTest<int8_t>("int8_basic", {1, -2, 3, -4}, {2, 3, -1, 0}, {2, 2},
                                         aclDataType::ACL_INT8, stream, 0.0, 0.0);
  totalFailed += RunMulTypedTest<uint8_t>("uint8_basic", {1, 2, 3, 4}, {2, 3, 1, 0}, {2, 2},
                                          aclDataType::ACL_UINT8, stream, 0.0, 0.0);
  totalFailed += RunMulTypedTest<int64_t>("int64_basic", {1000000, -2000000, 30, -4}, {2, 3, -1, 0}, {2, 2},
                                          aclDataType::ACL_INT64, stream, 0.0, 0.0);
  totalFailed += RunMulTypedTest<double>("double_basic", {1.25, -2.5, 3.0, -4.0}, {2.0, 3.0, -0.5, 0.0}, {2, 2},
                                         aclDataType::ACL_DOUBLE, stream, 1e-10, 1e-10);
  totalFailed += RunMulComplexTest<float>("complex64_basic", {{1.0f, -2.0f}, {-3.0f, 4.0f}, {0.5f, 0.25f},
                                                                {-1.25f, -0.75f}},
                                          {{-2.0f, 3.0f}, {0.5f, -1.0f}, {2.0f, 2.5f}, {-0.5f, 0.5f}}, {2, 2},
                                          aclDataType::ACL_COMPLEX64, stream, 1e-5, 1e-5);
  totalFailed += RunMulComplex128RouteProbeTest("complex128_route_probe", stream);
  totalFailed += RunMulBoolTest("bool_basic", {1, 0, 1, 0}, {1, 1, 0, 0}, {2, 2}, stream);
  totalFailed += RunMulFp16Test("fp16_basic", {1.0f, -2.0f, 0.5f, 4.0f}, {2.0f, 3.0f, -1.5f, 0.25f}, {2, 2},
                                stream);
  totalFailed += RunMulBf16Test("bf16_basic", {1.0f, -2.0f, 0.5f, 4.0f}, {2.0f, 3.0f, -1.5f, 0.25f}, {2, 2},
                                stream);
  totalFailed += RunMulFp16Fp32MixTest("fp16_fp32_mix_self_fp16", {1.0f, -2.0f, 3.0f, -4.0f},
                                       {1.5f, 0.5f, -2.0f, -1.0f}, {2, 2}, true, stream);
  totalFailed += RunMulFp16Fp32MixTest("fp16_fp32_mix_other_fp16", {1.0f, -2.0f, 3.0f, -4.0f},
                                       {1.5f, 0.5f, -2.0f, -1.0f}, {2, 2}, false, stream);
  totalFailed += RunMulBf16Fp32MixTest("bf16_fp32_mix_self_bf16", {1.0f, -2.0f, 3.0f, -4.0f},
                                       {1.5f, 0.5f, -2.0f, -1.0f}, {2, 2}, true, stream);
  totalFailed += RunMulBf16Fp32MixTest("bf16_fp32_mix_other_bf16", {1.0f, -2.0f, 3.0f, -4.0f},
                                       {1.5f, 0.5f, -2.0f, -1.0f}, {2, 2}, false, stream);
  totalFailed += RunMulInt32Fp32PromoteTest("int32_fp32_promote", {1, -2, 3, -4},
                                            {0.5f, 2.0f, -1.5f, 0.0f}, {2, 2}, stream);
  totalFailed += RunMulTest("float32_5d_noncontiguous_gate", {1.0f, 2.0f, 3.0f, 4.0f},
                            {0.5f, -1.0f, 2.0f, 3.0f}, {1, 1, 1, 1, 4}, stream);
  totalFailed += RunMulsBf16DoubleScalarTest("bf16_muls_double_scalar", {1.0f, -2.0f, 3.0f, -4.0f}, {2, 2},
                                             1.25, stream);
  totalFailed += RunMulsBf16DoubleScalarTest("bf16_muls_double_scalar_rounding", {1.0f, -2.0f, 3.0f, -4.0f},
                                             {2, 2}, 1.3, stream);
  totalFailed += RunMulsFp16ScalarTest("fp16_muls_scalar", {1.0f, -2.0f, 3.0f, -4.0f}, {2, 2}, 1.25f, stream);
  totalFailed += RunMulsFp16ScalarTest("fp16_muls_scalar_rounding", {1.0f, -2.0f, 3.0f, -4.0f}, {2, 2}, 1.3f,
                                        stream);
  totalFailed += RunMulBroadcastTest("float32_broadcast_2x3_1x3", {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3},
                                     {10.0f, 100.0f, 1000.0f}, {1, 3}, {2, 3},
                                     {10.0f, 200.0f, 3000.0f, 40.0f, 500.0f, 6000.0f}, stream);
  totalFailed += RunMulFormatWarningTest("mul_format_nchw_warning", stream);
  totalFailed += RunMulsFormatWarningTest("muls_format_nchw_warning", stream);
  totalFailed += RunMulEmptyTensorFastPathTest("mul_empty_tensor_fastpath", stream);
  totalFailed += RunMulsEmptyTensorFastPathTest("muls_empty_tensor_fastpath", stream);
  totalFailed += RunInplaceMulEmptyTensorFastPathTest("inplace_mul_empty_tensor_fastpath", stream);
  totalFailed += RunInplaceMulsEmptyTensorFastPathTest("inplace_muls_empty_tensor_fastpath", stream);
  totalFailed += RunInplaceMulTest("float32_inplace_mul", {1.0f, -2.0f, 3.0f, -4.0f}, {3.0f, 4.0f, 0.5f, -1.0f},
                                   {2, 2}, stream);
  totalFailed += RunInplaceMulsTest("float32_inplace_muls_scalar", {1.0f, -2.0f, 3.0f, -4.0f}, {2, 2}, 1.5f,
                                    stream);
  totalFailed += RunInplaceMulsFp16ScalarTest("fp16_inplace_muls_scalar", {1.0f, -2.0f, 3.0f, -4.0f},
                                               {2, 2}, 1.5f, stream);
  totalFailed += RunInplaceMulsFp16ScalarTest("fp16_inplace_muls_scalar_rounding", {1.0f, -2.0f, 3.0f, -4.0f},
                                               {2, 2}, 1.3f, stream);
  totalFailed += RunMulNullptrGuardTest("mul_getws_nullptr_self", stream);
  totalFailed += RunMulNullptrOtherGuardTest("mul_getws_nullptr_other", stream);
  totalFailed += RunMulInvalidBroadcastGuardTest("mul_getws_invalid_broadcast", stream);
  totalFailed += RunMulOutShapeMismatchGuardTest("mul_getws_out_shape_mismatch", stream);
  totalFailed += RunMulMaxDimSelfGuardTest("mul_getws_max_dim_self", stream);
  totalFailed += RunMulMaxDimOtherGuardTest("mul_getws_max_dim_other", stream);
  totalFailed += RunMulsNullptrSelfGuardTest("muls_getws_nullptr_self", stream);
  totalFailed += RunMulsNullptrGuardTest("muls_getws_nullptr_out", stream);
  totalFailed += RunMulsShapeMismatchGuardTest("muls_getws_out_shape_mismatch", stream);
  totalFailed += RunInplaceMulsNullptrSelfGuardTest("inplace_muls_getws_nullptr_self", stream);
  totalFailed += RunInplaceMulsNullptrGuardTest("inplace_muls_getws_nullptr_scalar", stream);
  totalFailed += RunInplaceMulNullptrGuardTest("inplace_mul_getws_nullptr_self", stream);
  totalFailed += RunInplaceMulNullptrOtherGuardTest("inplace_mul_getws_nullptr_other", stream);
  totalFailed += RunInplaceMulInvalidBroadcastGuardTest("inplace_mul_getws_invalid_broadcast", stream);
  totalFailed += RunInplaceMulShapeExpandGuardTest("inplace_mul_getws_shape_expand", stream);

  LOG_PRINT("\n=== Summary: %d failed ===\n", totalFailed);
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return totalFailed;
}