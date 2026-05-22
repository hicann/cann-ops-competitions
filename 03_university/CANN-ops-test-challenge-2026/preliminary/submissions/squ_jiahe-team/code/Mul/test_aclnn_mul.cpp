/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "../op_api/aclnn_mul.h"

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

namespace {

struct Complex64Value {
  float real;
  float imag;
};

struct TensorResource {
  void* deviceAddr = nullptr;
  aclTensor* tensor = nullptr;
};

struct ScalarResource {
  aclScalar* scalar = nullptr;
};

struct TensorSpec {
  std::vector<int64_t> shape;
  aclDataType dataType;
  aclFormat format = aclFormat::ACL_FORMAT_ND;
  uint8_t seed = 1;
  std::vector<int64_t> strides;
  std::vector<int64_t> storageShape;
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  int64_t shapeSize = 1;
  for (auto dim : shape) {
    shapeSize *= dim;
  }
  return shapeSize;
}

size_t GetDataTypeSize(aclDataType dataType)
{
  switch (dataType) {
    case aclDataType::ACL_BOOL:
    case aclDataType::ACL_INT8:
    case aclDataType::ACL_UINT8:
      return 1;
    case aclDataType::ACL_FLOAT16:
    case aclDataType::ACL_BF16:
    case aclDataType::ACL_INT16:
      return 2;
    case aclDataType::ACL_FLOAT:
    case aclDataType::ACL_INT32:
      return 4;
    case aclDataType::ACL_DOUBLE:
    case aclDataType::ACL_INT64:
    case aclDataType::ACL_COMPLEX64:
      return 8;
    case aclDataType::ACL_COMPLEX128:
      return 16;
    default:
      return 0;
  }
}

std::vector<int64_t> MakeContiguousStrides(const std::vector<int64_t>& shape)
{
  std::vector<int64_t> strides(shape.size(), 1);
  if (shape.empty()) {
    return strides;
  }
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
  }
  return strides;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor, aclFormat format = aclFormat::ACL_FORMAT_ND)
{
  const auto size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
  if (size > 0) {
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  }

  auto strides = MakeContiguousStrides(shape);
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(), shape.size(),
                            *deviceAddr);
  CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
  return ACL_SUCCESS;
}

int CreateAclTensorRaw(const TensorSpec& spec, TensorResource* resource)
{
  const auto& storageShape = spec.storageShape.empty() ? spec.shape : spec.storageShape;
  const auto& strides = spec.strides.empty() ? MakeContiguousStrides(spec.shape) : spec.strides;
  const auto elementCount = static_cast<size_t>(GetShapeSize(storageShape));
  const auto typeSize = GetDataTypeSize(spec.dataType);
  const bool allowUndefinedDtype = spec.dataType == aclDataType::ACL_DT_UNDEFINED;
  CHECK_RET(typeSize > 0 || allowUndefinedDtype,
            LOG_PRINT("unsupported aclDataType in CreateAclTensorRaw: %d\n", static_cast<int>(spec.dataType));
            return ACL_ERROR_FAILURE);

  std::vector<uint8_t> hostBytes;
  if (!allowUndefinedDtype) {
    hostBytes.assign(elementCount * typeSize, 0);
    if (spec.dataType == aclDataType::ACL_BOOL) {
      for (size_t i = 0; i < hostBytes.size(); ++i) {
        hostBytes[i] = static_cast<uint8_t>((i + spec.seed) % 2);
      }
    } else {
      for (size_t i = 0; i < hostBytes.size(); ++i) {
        hostBytes[i] = static_cast<uint8_t>(spec.seed + (i % 17));
      }
    }
  }

  if (allowUndefinedDtype) {
    resource->tensor = aclCreateTensor(spec.shape.data(), spec.shape.size(), spec.dataType, strides.data(), 0, spec.format,
                                       storageShape.data(), storageShape.size(), nullptr);
    CHECK_RET(resource->tensor != nullptr, LOG_PRINT("aclCreateTensor failed for undefined dtype.\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
  }

  const auto size = hostBytes.size();
  if (size > 0) {
    auto ret = aclrtMalloc(&resource->deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(resource->deviceAddr, size, hostBytes.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  }

  resource->tensor = aclCreateTensor(spec.shape.data(), spec.shape.size(), spec.dataType, strides.data(), 0, spec.format,
                                     storageShape.data(), storageShape.size(), resource->deviceAddr);
  CHECK_RET(resource->tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
  return ACL_SUCCESS;
}

void DestroyTensor(TensorResource* resource)
{
  if (resource->tensor != nullptr) {
    aclDestroyTensor(resource->tensor);
    resource->tensor = nullptr;
  }
  if (resource->deviceAddr != nullptr) {
    aclrtFree(resource->deviceAddr);
    resource->deviceAddr = nullptr;
  }
}

void DestroyScalar(ScalarResource* resource)
{
  if (resource->scalar != nullptr) {
    aclDestroyScalar(resource->scalar);
    resource->scalar = nullptr;
  }
}

template <typename T>
int CreateScalarResource(const T& value, aclDataType dataType, ScalarResource* resource)
{
  T valueCopy = value;
  resource->scalar = aclCreateScalar(&valueCopy, dataType);
  CHECK_RET(resource->scalar != nullptr, LOG_PRINT("aclCreateScalar failed.\n"); return ACL_ERROR_FAILURE);
  return ACL_SUCCESS;
}

bool CheckFloatResult(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& caseName,
                      float atol = 1e-5F, float rtol = 1e-5F)
{
  CHECK_RET(actual.size() == expected.size(),
            LOG_PRINT("[%s] result size mismatch, actual=%zu expected=%zu\n", caseName.c_str(), actual.size(),
                      expected.size());
            return false);

  for (size_t i = 0; i < actual.size(); ++i) {
    const float diff = std::fabs(actual[i] - expected[i]);
    const float limit = atol + rtol * std::fabs(expected[i]);
    if (diff > limit) {
      LOG_PRINT("[%s] result[%zu] mismatch, actual=%f expected=%f diff=%f limit=%f\n", caseName.c_str(), i, actual[i],
                expected[i], diff, limit);
      return false;
    }
  }
  LOG_PRINT("[%s] PASS\n", caseName.c_str());
  return true;
}

template <typename T>
std::string FormatScalarValue(const T& value)
{
  std::ostringstream oss;
  if constexpr (std::is_same_v<T, bool>) {
    oss << (value ? "true" : "false");
  } else if constexpr (std::is_same_v<T, int8_t>) {
    oss << static_cast<int>(value);
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    oss << static_cast<unsigned int>(value);
  } else {
    oss << value;
  }
  return oss.str();
}

std::string FormatShape(const std::vector<int64_t>& shape)
{
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

template <typename T>
void AppendTensorRecursive(std::ostringstream& oss, const std::vector<T>& values, const std::vector<int64_t>& shape,
                           const std::vector<int64_t>& strides, size_t dim, size_t offset)
{
  if (dim == shape.size()) {
    oss << FormatScalarValue(values[offset]);
    return;
  }

  oss << "[";
  for (int64_t i = 0; i < shape[dim]; ++i) {
    if (i > 0) {
      oss << ", ";
    }
    AppendTensorRecursive(oss, values, shape, strides, dim + 1, offset + static_cast<size_t>(i * strides[dim]));
  }
  oss << "]";
}

template <typename T>
std::string FormatTensorByShape(const std::vector<T>& values, const std::vector<int64_t>& shape)
{
  if (shape.empty()) {
    return values.empty() ? "[]" : FormatScalarValue(values[0]);
  }

  std::ostringstream oss;
  AppendTensorRecursive(oss, values, shape, MakeContiguousStrides(shape), 0, 0);
  return oss.str();
}

template <typename T>
bool CheckVectorResult(const std::vector<T>& actual, const std::vector<T>& expected, const std::string& caseName,
                       double atol = 1e-5, double rtol = 1e-5)
{
  CHECK_RET(actual.size() == expected.size(),
            LOG_PRINT("[%s] result size mismatch, actual=%zu expected=%zu\n", caseName.c_str(), actual.size(),
                      expected.size());
            return false);

  for (size_t i = 0; i < actual.size(); ++i) {
    if constexpr (std::is_floating_point_v<T>) {
      const double diff = std::fabs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
      const double limit = atol + rtol * std::fabs(static_cast<double>(expected[i]));
      if (diff > limit) {
        LOG_PRINT("[%s] result[%zu] mismatch, actual=%g expected=%g diff=%g limit=%g\n", caseName.c_str(), i,
                  static_cast<double>(actual[i]), static_cast<double>(expected[i]), diff, limit);
        return false;
      }
    } else {
      if (actual[i] != expected[i]) {
        LOG_PRINT("[%s] result[%zu] mismatch, actual=%s expected=%s\n", caseName.c_str(), i,
                  FormatScalarValue(actual[i]).c_str(), FormatScalarValue(expected[i]).c_str());
        return false;
      }
    }
  }

  LOG_PRINT("[%s] PASS\n", caseName.c_str());
  return true;
}

template <typename T>
int ReportPrecisionCase(const std::string& caseName, const std::string& testContent, const std::string& inputSummary,
                        const std::vector<int64_t>& outShape, const std::vector<T>& actual,
                        const std::vector<T>& expected, double atol = 1e-5, double rtol = 1e-5)
{
  LOG_PRINT("[Precision][%s]\n", caseName.c_str());
  LOG_PRINT("  Tested: %s\n", testContent.c_str());
  LOG_PRINT("  Inputs: %s\n", inputSummary.c_str());
  LOG_PRINT("  Output shape: %s\n", FormatShape(outShape).c_str());
  LOG_PRINT("  Expected: %s\n", FormatTensorByShape(expected, outShape).c_str());
  LOG_PRINT("  Actual:   %s\n", FormatTensorByShape(actual, outShape).c_str());
  LOG_PRINT("  Criterion: atol=%g, rtol=%g\n", atol, rtol);

  if (!CheckVectorResult(actual, expected, caseName, atol, rtol)) {
    LOG_PRINT("  Result: FAIL\n");
    return 1;
  }

  LOG_PRINT("  Result: PASS\n");
  return 0;
}

int CheckStatus(const std::string& caseName, aclnnStatus actual, aclnnStatus expected)
{
  if (actual != expected) {
    LOG_PRINT("[%s] failed, actual status=%d expected status=%d\n", caseName.c_str(), actual, expected);
    return 1;
  }
  LOG_PRINT("[%s] PASS, status=%d\n", caseName.c_str(), actual);
  return 0;
}

template <typename T>
int RunMulPrecisionCaseTyped(const std::string& caseName, const std::string& testContent, const std::string& inputSummary,
                             const std::vector<int64_t>& selfShape, const std::vector<int64_t>& otherShape,
                             const std::vector<int64_t>& outShape, const std::vector<T>& selfHostData,
                             const std::vector<T>& otherHostData, const std::vector<T>& expected,
                             aclDataType dataType, aclrtStream stream)
{
  std::vector<T> actual(expected.size());
  std::vector<T> outHostData(expected.size(), static_cast<T>(0));
  TensorResource self;
  TensorResource other;
  TensorResource out;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  int ret = ACL_SUCCESS;

  ret = CreateAclTensor(selfHostData, selfShape, &self.deviceAddr, dataType, &self.tensor);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateAclTensor(otherHostData, otherShape, &other.deviceAddr, dataType, &other.tensor);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateAclTensor(outHostData, outShape, &out.deviceAddr, dataType, &out.tensor);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  ret = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s aclnnMulGetWorkspaceSize failed. ERROR: %d\n", caseName.c_str(), ret);
            goto cleanup);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s allocate workspace failed. ERROR: %d\n", caseName.c_str(), ret);
              goto cleanup);
  }
  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s aclnnMul failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s aclrtSynchronizeStream failed. ERROR: %d\n", caseName.c_str(), ret);
            goto cleanup);
  ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(T), out.deviceAddr, actual.size() * sizeof(T),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s copy result failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  ret = ReportPrecisionCase(caseName, testContent, inputSummary, outShape, actual, expected);
  CHECK_RET(ret == 0, goto cleanup);

cleanup:
  DestroyTensor(&self);
  DestroyTensor(&other);
  DestroyTensor(&out);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ret;
}

template <typename T>
int RunInplaceMulPrecisionCaseTyped(const std::string& caseName, const std::string& testContent, const std::string& inputSummary,
                                    const std::vector<int64_t>& selfShape, const std::vector<int64_t>& otherShape,
                                    const std::vector<T>& selfHostData, const std::vector<T>& otherHostData,
                                    const std::vector<T>& expected, aclDataType dataType, aclrtStream stream)
{
  std::vector<T> actual(expected.size());
  TensorResource self;
  TensorResource other;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  int ret = ACL_SUCCESS;

  ret = CreateAclTensor(selfHostData, selfShape, &self.deviceAddr, dataType, &self.tensor);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateAclTensor(otherHostData, otherShape, &other.deviceAddr, dataType, &other.tensor);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  ret = aclnnInplaceMulGetWorkspaceSize(self.tensor, other.tensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("%s aclnnInplaceMulGetWorkspaceSize failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s allocate workspace failed. ERROR: %d\n", caseName.c_str(), ret);
              goto cleanup);
  }
  ret = aclnnInplaceMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s aclnnInplaceMul failed. ERROR: %d\n", caseName.c_str(), ret);
            goto cleanup);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s aclrtSynchronizeStream failed. ERROR: %d\n", caseName.c_str(), ret);
            goto cleanup);
  ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(T), self.deviceAddr, actual.size() * sizeof(T),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s copy result failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  ret = ReportPrecisionCase(caseName, testContent, inputSummary, selfShape, actual, expected);
  CHECK_RET(ret == 0, goto cleanup);

cleanup:
  DestroyTensor(&self);
  DestroyTensor(&other);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ret;
}

template <typename TensorT, typename ScalarT, typename OutT>
int RunMulsPrecisionCaseTyped(const std::string& caseName, const std::string& testContent, const std::string& inputSummary,
                              const std::vector<int64_t>& selfShape, const std::vector<int64_t>& outShape,
                              const std::vector<TensorT>& selfHostData, const ScalarT& scalarValue,
                              const std::vector<OutT>& expected, aclDataType selfType, aclDataType scalarType,
                              aclDataType outType, aclrtStream stream)
{
  std::vector<OutT> actual(expected.size());
  std::vector<OutT> outHostData(expected.size(), static_cast<OutT>(0));
  TensorResource self;
  TensorResource out;
  ScalarResource scalar;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  int ret = ACL_SUCCESS;

  ret = CreateAclTensor(selfHostData, selfShape, &self.deviceAddr, selfType, &self.tensor);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateScalarResource(scalarValue, scalarType, &scalar);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateAclTensor(outHostData, outShape, &out.deviceAddr, outType, &out.tensor);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  ret = aclnnMulsGetWorkspaceSize(self.tensor, scalar.scalar, out.tensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s aclnnMulsGetWorkspaceSize failed. ERROR: %d\n", caseName.c_str(), ret);
            goto cleanup);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s allocate workspace failed. ERROR: %d\n", caseName.c_str(), ret);
              goto cleanup);
  }
  ret = aclnnMuls(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s aclnnMuls failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s aclrtSynchronizeStream failed. ERROR: %d\n", caseName.c_str(), ret);
            goto cleanup);
  ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(OutT), out.deviceAddr, actual.size() * sizeof(OutT),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s copy result failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  ret = ReportPrecisionCase(caseName, testContent, inputSummary, outShape, actual, expected);
  CHECK_RET(ret == 0, goto cleanup);

cleanup:
  DestroyTensor(&self);
  DestroyTensor(&out);
  DestroyScalar(&scalar);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ret;
}

template <typename TensorT, typename ScalarT>
int RunInplaceMulsPrecisionCaseTyped(const std::string& caseName, const std::string& testContent,
                                     const std::string& inputSummary, const std::vector<int64_t>& selfShape,
                                     const std::vector<TensorT>& selfHostData, const ScalarT& scalarValue,
                                     const std::vector<TensorT>& expected, aclDataType selfType,
                                     aclDataType scalarType, aclrtStream stream)
{
  std::vector<TensorT> actual(expected.size());
  TensorResource self;
  ScalarResource scalar;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  int ret = ACL_SUCCESS;

  ret = CreateAclTensor(selfHostData, selfShape, &self.deviceAddr, selfType, &self.tensor);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateScalarResource(scalarValue, scalarType, &scalar);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  ret = aclnnInplaceMulsGetWorkspaceSize(self.tensor, scalar.scalar, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("%s aclnnInplaceMulsGetWorkspaceSize failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s allocate workspace failed. ERROR: %d\n", caseName.c_str(), ret);
              goto cleanup);
  }
  ret = aclnnInplaceMuls(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s aclnnInplaceMuls failed. ERROR: %d\n", caseName.c_str(), ret);
            goto cleanup);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s aclrtSynchronizeStream failed. ERROR: %d\n", caseName.c_str(), ret);
            goto cleanup);
  ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(TensorT), self.deviceAddr, actual.size() * sizeof(TensorT),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s copy result failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  ret = ReportPrecisionCase(caseName, testContent, inputSummary, selfShape, actual, expected);
  CHECK_RET(ret == 0, goto cleanup);

cleanup:
  DestroyTensor(&self);
  DestroyScalar(&scalar);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ret;
}

int RunMulExecuteCase(aclrtStream stream)
{
  const std::vector<int64_t> selfShape = {4, 2};
  const std::vector<int64_t> otherShape = {4, 2};
  const std::vector<int64_t> outShape = {4, 2};
  const std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  const std::vector<float> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  const std::vector<float> outHostData(8, 0);
  const std::vector<float> expected = {0, 1, 2, 6, 8, 10, 18, 21};
  std::vector<float> actual(expected.size(), 0);
  TensorResource self;
  TensorResource other;
  TensorResource out;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  int ret = ACL_SUCCESS;

  ret = CreateAclTensor(selfHostData, selfShape, &self.deviceAddr, aclDataType::ACL_FLOAT, &self.tensor);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateAclTensor(otherHostData, otherShape, &other.deviceAddr, aclDataType::ACL_FLOAT, &other.tensor);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateAclTensor(outHostData, outShape, &out.deviceAddr, aclDataType::ACL_FLOAT, &out.tensor);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  ret = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulGetWorkspaceSize failed. ERROR: %d\n", ret); goto cleanup);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace for aclnnMul failed. ERROR: %d\n", ret); goto cleanup);
  }
  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMul failed. ERROR: %d\n", ret); goto cleanup);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed after aclnnMul. ERROR: %d\n", ret); goto cleanup);
  ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(float), out.deviceAddr, actual.size() * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy aclnnMul result failed. ERROR: %d\n", ret); goto cleanup);
  ret = ReportPrecisionCase("aclnnMul_basic_fp32",
                            "Tensor x Tensor precision, same-shape fp32 multiply",
                            "self shape=[4,2], other shape=[4,2], out dtype=float",
                            outShape, actual, expected);
  CHECK_RET(ret == 0, goto cleanup);

cleanup:
  DestroyTensor(&self);
  DestroyTensor(&other);
  DestroyTensor(&out);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ret;
}

int RunMulBroadcastExecuteCase(aclrtStream stream)
{
  const std::vector<int64_t> selfShape = {2, 3};
  const std::vector<int64_t> otherShape = {1, 3};
  const std::vector<int64_t> outShape = {2, 3};
  const std::vector<float> selfHostData = {1.0F, -2.0F, 3.0F, 4.0F, -5.0F, 6.0F};
  const std::vector<float> otherHostData = {0.5F, 2.0F, -1.0F};
  const std::vector<float> outHostData(6, 0);
  const std::vector<float> expected = {0.5F, -4.0F, -3.0F, 2.0F, -10.0F, -6.0F};
  std::vector<float> actual(expected.size(), 0);
  TensorResource self;
  TensorResource other;
  TensorResource out;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  int ret = ACL_SUCCESS;

  ret = CreateAclTensor(selfHostData, selfShape, &self.deviceAddr, aclDataType::ACL_FLOAT, &self.tensor);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateAclTensor(otherHostData, otherShape, &other.deviceAddr, aclDataType::ACL_FLOAT, &other.tensor);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateAclTensor(outHostData, outShape, &out.deviceAddr, aclDataType::ACL_FLOAT, &out.tensor);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  ret = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulGetWorkspaceSize failed. ERROR: %d\n", ret); goto cleanup);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace for broadcast aclnnMul failed. ERROR: %d\n", ret);
              goto cleanup);
  }
  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("broadcast aclnnMul failed. ERROR: %d\n", ret); goto cleanup);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed after broadcast aclnnMul. ERROR: %d\n", ret); goto cleanup);
  ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(float), out.deviceAddr, actual.size() * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy broadcast aclnnMul result failed. ERROR: %d\n", ret); goto cleanup);
  ret = ReportPrecisionCase("aclnnMul_broadcast_fp32",
                            "Tensor x Tensor precision, fp32 broadcast multiply",
                            "self shape=[2,3], other shape=[1,3], out shape=[2,3]",
                            outShape, actual, expected);
  CHECK_RET(ret == 0, goto cleanup);

cleanup:
  DestroyTensor(&self);
  DestroyTensor(&other);
  DestroyTensor(&out);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ret;
}

int RunInplaceMulExecuteCase(aclrtStream stream)
{
  const std::vector<int64_t> selfShape = {4, 2};
  const std::vector<int64_t> otherShape = {4, 2};
  const std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  const std::vector<float> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  const std::vector<float> expected = {0, 1, 2, 6, 8, 10, 18, 21};
  std::vector<float> actual(expected.size(), 0);
  TensorResource self;
  TensorResource other;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  int ret = ACL_SUCCESS;

  ret = CreateAclTensor(selfHostData, selfShape, &self.deviceAddr, aclDataType::ACL_FLOAT, &self.tensor);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateAclTensor(otherHostData, otherShape, &other.deviceAddr, aclDataType::ACL_FLOAT, &other.tensor);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  ret = aclnnInplaceMulGetWorkspaceSize(self.tensor, other.tensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceMulGetWorkspaceSize failed. ERROR: %d\n", ret); goto cleanup);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace for aclnnInplaceMul failed. ERROR: %d\n", ret); goto cleanup);
  }
  ret = aclnnInplaceMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceMul failed. ERROR: %d\n", ret); goto cleanup);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed after aclnnInplaceMul. ERROR: %d\n", ret); goto cleanup);
  ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(float), self.deviceAddr, actual.size() * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy aclnnInplaceMul result failed. ERROR: %d\n", ret); goto cleanup);
  ret = ReportPrecisionCase("aclnnInplaceMul_basic_fp32",
                            "Inplace Tensor x Tensor precision, same-shape fp32 multiply",
                            "self shape=[4,2], other shape=[4,2], self updated in-place",
                            selfShape, actual, expected);
  CHECK_RET(ret == 0, goto cleanup);

cleanup:
  DestroyTensor(&self);
  DestroyTensor(&other);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ret;
}

int RunMulsExecuteCase(aclrtStream stream)
{
  const std::vector<int64_t> selfShape = {4, 2};
  const std::vector<int64_t> outShape = {4, 2};
  const std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  const std::vector<float> outHostData(8, 0);
  const std::vector<float> expected = {0, 1.25F, 2.5F, 3.75F, 5.0F, 6.25F, 7.5F, 8.75F};
  const float scalarValue = 1.25F;
  std::vector<float> actual(expected.size(), 0);
  TensorResource self;
  TensorResource out;
  ScalarResource scalar;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  int ret = ACL_SUCCESS;

  ret = CreateAclTensor(selfHostData, selfShape, &self.deviceAddr, aclDataType::ACL_FLOAT, &self.tensor);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateScalarResource(scalarValue, aclDataType::ACL_FLOAT, &scalar);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateAclTensor(outHostData, outShape, &out.deviceAddr, aclDataType::ACL_FLOAT, &out.tensor);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  ret = aclnnMulsGetWorkspaceSize(self.tensor, scalar.scalar, out.tensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulsGetWorkspaceSize failed. ERROR: %d\n", ret); goto cleanup);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace for aclnnMuls failed. ERROR: %d\n", ret); goto cleanup);
  }
  ret = aclnnMuls(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMuls failed. ERROR: %d\n", ret); goto cleanup);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed after aclnnMuls. ERROR: %d\n", ret); goto cleanup);
  ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(float), out.deviceAddr, actual.size() * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy aclnnMuls result failed. ERROR: %d\n", ret); goto cleanup);
  ret = ReportPrecisionCase("aclnnMuls_basic_fp32",
                            "Tensor x Scalar precision, fp32 scalar multiply",
                            "self shape=[4,2], scalar=1.25, out dtype=float",
                            outShape, actual, expected);
  CHECK_RET(ret == 0, goto cleanup);

cleanup:
  DestroyTensor(&self);
  DestroyTensor(&out);
  DestroyScalar(&scalar);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ret;
}

int RunInplaceMulsExecuteCase(aclrtStream stream)
{
  const std::vector<int64_t> selfShape = {4, 2};
  const std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  const std::vector<float> expected = {0, 1.25F, 2.5F, 3.75F, 5.0F, 6.25F, 7.5F, 8.75F};
  const float scalarValue = 1.25F;
  std::vector<float> actual(expected.size(), 0);
  TensorResource self;
  ScalarResource scalar;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  int ret = ACL_SUCCESS;

  ret = CreateAclTensor(selfHostData, selfShape, &self.deviceAddr, aclDataType::ACL_FLOAT, &self.tensor);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateScalarResource(scalarValue, aclDataType::ACL_FLOAT, &scalar);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  ret = aclnnInplaceMulsGetWorkspaceSize(self.tensor, scalar.scalar, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceMulsGetWorkspaceSize failed. ERROR: %d\n", ret); goto cleanup);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace for aclnnInplaceMuls failed. ERROR: %d\n", ret); goto cleanup);
  }
  ret = aclnnInplaceMuls(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceMuls failed. ERROR: %d\n", ret); goto cleanup);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed after aclnnInplaceMuls. ERROR: %d\n", ret); goto cleanup);
  ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(float), self.deviceAddr, actual.size() * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy aclnnInplaceMuls result failed. ERROR: %d\n", ret); goto cleanup);
  ret = ReportPrecisionCase("aclnnInplaceMuls_basic_fp32",
                            "Inplace Tensor x Scalar precision, fp32 scalar multiply",
                            "self shape=[4,2], scalar=1.25, self updated in-place",
                            selfShape, actual, expected);
  CHECK_RET(ret == 0, goto cleanup);

cleanup:
  DestroyTensor(&self);
  DestroyScalar(&scalar);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ret;
}

int RunMulWorkspaceCase(const std::string& caseName, const TensorSpec& selfSpec, const TensorSpec& otherSpec,
                        const TensorSpec& outSpec, aclnnStatus expectedStatus)
{
  TensorResource self;
  TensorResource other;
  TensorResource out;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  int ret = ACL_SUCCESS;

  ret = CreateAclTensorRaw(selfSpec, &self);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateAclTensorRaw(otherSpec, &other);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateAclTensorRaw(outSpec, &out);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  ret = CheckStatus(caseName, aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor),
                    expectedStatus);

cleanup:
  DestroyTensor(&self);
  DestroyTensor(&other);
  DestroyTensor(&out);
  return ret;
}

int RunMulNullptrCase(const std::string& caseName, aclnnStatus expectedStatus)
{
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  return CheckStatus(caseName, aclnnMulGetWorkspaceSize(nullptr, nullptr, nullptr, &workspaceSize, &executor),
                     expectedStatus);
}

int RunMulPartialNullptrCase(const std::string& caseName, bool nullSelf, bool nullOther, bool nullOut, aclnnStatus expectedStatus)
{
  TensorResource self;
  TensorResource other;
  TensorResource out;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  int ret = ACL_SUCCESS;

  if (!nullSelf) {
    ret = CreateAclTensorRaw({{2, 3}, aclDataType::ACL_FLOAT}, &self);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  }
  if (!nullOther) {
    ret = CreateAclTensorRaw({{2, 3}, aclDataType::ACL_FLOAT}, &other);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  }
  if (!nullOut) {
    ret = CreateAclTensorRaw({{2, 3}, aclDataType::ACL_FLOAT}, &out);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  }

  ret = CheckStatus(caseName,
                    aclnnMulGetWorkspaceSize(nullSelf ? nullptr : self.tensor, nullOther ? nullptr : other.tensor,
                                             nullOut ? nullptr : out.tensor, &workspaceSize, &executor),
                    expectedStatus);

cleanup:
  DestroyTensor(&self);
  DestroyTensor(&other);
  DestroyTensor(&out);
  return ret;
}

int RunInplaceMulWorkspaceCase(const std::string& caseName, const TensorSpec& selfSpec, const TensorSpec& otherSpec,
                               aclnnStatus expectedStatus)
{
  TensorResource self;
  TensorResource other;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  int ret = ACL_SUCCESS;

  ret = CreateAclTensorRaw(selfSpec, &self);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateAclTensorRaw(otherSpec, &other);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  ret = CheckStatus(caseName, aclnnInplaceMulGetWorkspaceSize(self.tensor, other.tensor, &workspaceSize, &executor),
                    expectedStatus);

cleanup:
  DestroyTensor(&self);
  DestroyTensor(&other);
  return ret;
}

int RunInplaceMulNullptrCase(const std::string& caseName, aclnnStatus expectedStatus)
{
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  return CheckStatus(caseName, aclnnInplaceMulGetWorkspaceSize(nullptr, nullptr, &workspaceSize, &executor),
                     expectedStatus);
}

int RunInplaceMulPartialNullptrCase(const std::string& caseName, bool nullSelf, bool nullOther, aclnnStatus expectedStatus)
{
  TensorResource self;
  TensorResource other;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  int ret = ACL_SUCCESS;

  if (!nullSelf) {
    ret = CreateAclTensorRaw({{2, 3}, aclDataType::ACL_FLOAT}, &self);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  }
  if (!nullOther) {
    ret = CreateAclTensorRaw({{2, 3}, aclDataType::ACL_FLOAT}, &other);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  }

  ret = CheckStatus(caseName,
                    aclnnInplaceMulGetWorkspaceSize(nullSelf ? nullptr : self.tensor, nullOther ? nullptr : other.tensor,
                                                    &workspaceSize, &executor),
                    expectedStatus);

cleanup:
  DestroyTensor(&self);
  DestroyTensor(&other);
  return ret;
}

int RunMulsWorkspaceCase(const std::string& caseName, const TensorSpec& selfSpec, aclScalar* scalar,
                         const TensorSpec& outSpec, aclnnStatus expectedStatus)
{
  TensorResource self;
  TensorResource out;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  int ret = ACL_SUCCESS;

  ret = CreateAclTensorRaw(selfSpec, &self);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateAclTensorRaw(outSpec, &out);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  ret = CheckStatus(caseName, aclnnMulsGetWorkspaceSize(self.tensor, scalar, out.tensor, &workspaceSize, &executor),
                    expectedStatus);

cleanup:
  DestroyTensor(&self);
  DestroyTensor(&out);
  return ret;
}

int RunMulsNullptrCase(const std::string& caseName, aclnnStatus expectedStatus)
{
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  return CheckStatus(caseName, aclnnMulsGetWorkspaceSize(nullptr, nullptr, nullptr, &workspaceSize, &executor),
                     expectedStatus);
}

int RunMulsPartialNullptrCase(const std::string& caseName, bool nullSelf, bool nullScalar, bool nullOut, aclnnStatus expectedStatus)
{
  TensorResource self;
  TensorResource out;
  ScalarResource scalar;
  float scalarValue = 1.0F;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  int ret = ACL_SUCCESS;

  if (!nullSelf) {
    ret = CreateAclTensorRaw({{2, 3}, aclDataType::ACL_FLOAT}, &self);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  }
  if (!nullScalar) {
    ret = CreateScalarResource(scalarValue, aclDataType::ACL_FLOAT, &scalar);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  }
  if (!nullOut) {
    ret = CreateAclTensorRaw({{2, 3}, aclDataType::ACL_FLOAT}, &out);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  }

  ret = CheckStatus(caseName,
                    aclnnMulsGetWorkspaceSize(nullSelf ? nullptr : self.tensor, nullScalar ? nullptr : scalar.scalar,
                                              nullOut ? nullptr : out.tensor, &workspaceSize, &executor),
                    expectedStatus);

cleanup:
  DestroyTensor(&self);
  DestroyTensor(&out);
  DestroyScalar(&scalar);
  return ret;
}

int RunInplaceMulsWorkspaceCase(const std::string& caseName, const TensorSpec& selfSpec, aclScalar* scalar,
                                aclnnStatus expectedStatus)
{
  TensorResource self;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  int ret = ACL_SUCCESS;

  ret = CreateAclTensorRaw(selfSpec, &self);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  ret = CheckStatus(caseName, aclnnInplaceMulsGetWorkspaceSize(self.tensor, scalar, &workspaceSize, &executor),
                    expectedStatus);

cleanup:
  DestroyTensor(&self);
  return ret;
}

int RunInplaceMulsNullptrCase(const std::string& caseName, aclnnStatus expectedStatus)
{
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  return CheckStatus(caseName, aclnnInplaceMulsGetWorkspaceSize(nullptr, nullptr, &workspaceSize, &executor),
                     expectedStatus);
}

int RunInplaceMulsPartialNullptrCase(const std::string& caseName, bool nullSelf, bool nullScalar, aclnnStatus expectedStatus)
{
  TensorResource self;
  ScalarResource scalar;
  float scalarValue = 1.0F;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  int ret = ACL_SUCCESS;

  if (!nullSelf) {
    ret = CreateAclTensorRaw({{2, 3}, aclDataType::ACL_FLOAT}, &self);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  }
  if (!nullScalar) {
    ret = CreateScalarResource(scalarValue, aclDataType::ACL_FLOAT, &scalar);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  }

  ret = CheckStatus(caseName,
                    aclnnInplaceMulsGetWorkspaceSize(nullSelf ? nullptr : self.tensor,
                                                     nullScalar ? nullptr : scalar.scalar,
                                                     &workspaceSize, &executor),
                    expectedStatus);

cleanup:
  DestroyTensor(&self);
  DestroyScalar(&scalar);
  return ret;
}

}  // namespace

int main()
{
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  int ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  int failed = 0;

  LOG_PRINT("========== Mul Precision Tests ==========\n");
  failed += RunMulExecuteCase(stream);
  failed += RunMulBroadcastExecuteCase(stream);
  failed += RunMulPrecisionCaseTyped<float>("aclnnMul_3d_fp32",
                                            "Tensor x Tensor precision, 3D fp32 multiply",
                                            "self shape=[2,2,3], other shape=[2,2,3], out dtype=float",
                                            {2, 2, 3}, {2, 2, 3}, {2, 2, 3},
                                            {1.0F, -2.0F, 3.0F, 4.0F, -5.0F, 6.0F, 0.5F, -1.0F, 2.0F, -3.0F, 1.5F, 4.0F},
                                            {2.0F, 0.5F, -1.0F, 1.0F, 2.0F, -2.0F, 4.0F, -3.0F, 0.5F, 2.0F, -1.0F, 0.25F},
                                            {2.0F, -1.0F, -3.0F, 4.0F, -10.0F, -12.0F, 2.0F, 3.0F, 1.0F, -6.0F, -1.5F, 1.0F},
                                            aclDataType::ACL_FLOAT, stream);
  failed += RunMulPrecisionCaseTyped<float>("aclnnMul_broadcast_3d_fp32",
                                            "Tensor x Tensor precision, 3D fp32 broadcast multiply",
                                            "self shape=[2,2,3], other shape=[1,2,3], out dtype=float",
                                            {2, 2, 3}, {1, 2, 3}, {2, 2, 3},
                                            {1.0F, 2.0F, 3.0F, -1.0F, 0.5F, 4.0F, 2.0F, -2.0F, 1.5F, 3.0F, -4.0F, 0.25F},
                                            {2.0F, -1.0F, 0.5F, -3.0F, 4.0F, 2.0F},
                                            {2.0F, -2.0F, 1.5F, 3.0F, 2.0F, 8.0F, 4.0F, 2.0F, 0.75F, -9.0F, -16.0F, 0.5F},
                                            aclDataType::ACL_FLOAT, stream);
  failed += RunMulPrecisionCaseTyped<double>("aclnnMul_basic_f64",
                                             "Tensor x Tensor precision, same-shape double multiply",
                                             "self shape=[2,2], other shape=[2,2], out dtype=double",
                                             {2, 2}, {2, 2}, {2, 2},
                                             {1.25, -2.0, 3.5, -4.0},
                                             {2.0, 0.5, -1.5, -2.0},
                                             {2.5, -1.0, -5.25, 8.0},
                                             aclDataType::ACL_DOUBLE, stream);
  failed += RunMulPrecisionCaseTyped<int32_t>("aclnnMul_basic_i32",
                                              "Tensor x Tensor precision, same-shape int32 multiply",
                                              "self shape=[2,3], other shape=[2,3], out dtype=int32",
                                              {2, 3}, {2, 3}, {2, 3},
                                              {2, -3, 4, 5, -6, 7},
                                              {-1, 2, 3, -4, 5, 6},
                                              {-2, -6, 12, -20, -30, 42},
                                              aclDataType::ACL_INT32, stream);
  failed += RunMulPrecisionCaseTyped<int16_t>("aclnnMul_basic_i16",
                                              "Tensor x Tensor precision, same-shape int16 multiply",
                                              "self shape=[2,2], other shape=[2,2], out dtype=int16",
                                              {2, 2}, {2, 2}, {2, 2},
                                              {2, -3, 4, 5},
                                              {7, -8, -2, 3},
                                              {14, 24, -8, 15},
                                              aclDataType::ACL_INT16, stream);
  failed += RunInplaceMulExecuteCase(stream);
  failed += RunInplaceMulPrecisionCaseTyped<double>("aclnnInplaceMul_basic_f64",
                                                    "Inplace Tensor x Tensor precision, same-shape double multiply",
                                                    "self shape=[2,2], other shape=[2,2], self updated in-place",
                                                    {2, 2}, {2, 2},
                                                    {1.25, -2.0, 3.5, -4.0},
                                                    {2.0, 0.5, -1.5, -2.0},
                                                    {2.5, -1.0, -5.25, 8.0},
                                                    aclDataType::ACL_DOUBLE, stream);
  failed += RunMulsExecuteCase(stream);
  failed += RunMulsPrecisionCaseTyped<double, double, double>("aclnnMuls_basic_f64",
                                                              "Tensor x Scalar precision, double scalar multiply",
                                                              "self shape=[2,2], scalar=-0.5, out dtype=double",
                                                              {2, 2}, {2, 2},
                                                              {1.5, -2.0, 3.0, -4.5},
                                                              -0.5,
                                                              {-0.75, 1.0, -1.5, 2.25},
                                                              aclDataType::ACL_DOUBLE, aclDataType::ACL_DOUBLE,
                                                              aclDataType::ACL_DOUBLE, stream);
  failed += RunMulsPrecisionCaseTyped<int32_t, int32_t, int32_t>("aclnnMuls_basic_i32",
                                                                 "Tensor x Scalar precision, int32 scalar multiply",
                                                                 "self shape=[2,2], scalar=3, out dtype=int32",
                                                                 {2, 2}, {2, 2},
                                                                 {2, -3, 4, -5},
                                                                 3,
                                                                 {6, -9, 12, -15},
                                                                 aclDataType::ACL_INT32, aclDataType::ACL_INT32,
                                                                 aclDataType::ACL_INT32, stream);
  failed += RunInplaceMulsExecuteCase(stream);
  failed += RunInplaceMulsPrecisionCaseTyped<double, double>("aclnnInplaceMuls_basic_f64",
                                                             "Inplace Tensor x Scalar precision, double scalar multiply",
                                                             "self shape=[2,2], scalar=-0.5, self updated in-place",
                                                             {2, 2},
                                                             {1.5, -2.0, 3.0, -4.5},
                                                             -0.5,
                                                             {-0.75, 1.0, -1.5, 2.25},
                                                             aclDataType::ACL_DOUBLE, aclDataType::ACL_DOUBLE, stream);
  LOG_PRINT("======= Mul Coverage/Validation Tests =======\n");

  failed += RunMulWorkspaceCase("mul_broadcast_fp32",
                                {{2, 3}, aclDataType::ACL_FLOAT},
                                {{1, 3}, aclDataType::ACL_FLOAT},
                                {{2, 3}, aclDataType::ACL_FLOAT},
                                ACL_SUCCESS);
  failed += RunMulWorkspaceCase("mul_mix_fp16_fp32",
                                {{2, 3}, aclDataType::ACL_FLOAT16},
                                {{2, 3}, aclDataType::ACL_FLOAT},
                                {{2, 3}, aclDataType::ACL_FLOAT},
                                ACL_SUCCESS);
  failed += RunMulWorkspaceCase("mul_mix_fp32_fp16",
                                {{2, 3}, aclDataType::ACL_FLOAT},
                                {{2, 3}, aclDataType::ACL_FLOAT16},
                                {{2, 3}, aclDataType::ACL_FLOAT},
                                ACL_SUCCESS);
  failed += RunMulWorkspaceCase("mul_mix_bf16_fp32",
                                {{2, 3}, aclDataType::ACL_BF16},
                                {{2, 3}, aclDataType::ACL_FLOAT},
                                {{2, 3}, aclDataType::ACL_FLOAT},
                                ACL_SUCCESS);
  failed += RunMulWorkspaceCase("mul_mix_fp32_bf16",
                                {{2, 3}, aclDataType::ACL_FLOAT},
                                {{2, 3}, aclDataType::ACL_BF16},
                                {{2, 3}, aclDataType::ACL_FLOAT},
                                ACL_SUCCESS);
  failed += RunMulWorkspaceCase("mul_fp16",
                                {{2, 3}, aclDataType::ACL_FLOAT16},
                                {{2, 3}, aclDataType::ACL_FLOAT16},
                                {{2, 3}, aclDataType::ACL_FLOAT16},
                                ACL_SUCCESS);
  failed += RunMulWorkspaceCase("mul_bf16",
                                {{2, 3}, aclDataType::ACL_BF16},
                                {{2, 3}, aclDataType::ACL_BF16},
                                {{2, 3}, aclDataType::ACL_BF16},
                                ACL_SUCCESS);
  failed += RunMulWorkspaceCase("mul_shape_dim_gt4",
                                {{1, 1, 2, 2, 2}, aclDataType::ACL_FLOAT},
                                {{1, 1, 2, 2, 2}, aclDataType::ACL_FLOAT},
                                {{1, 1, 2, 2, 2}, aclDataType::ACL_FLOAT},
                                ACL_SUCCESS);
  failed += RunMulWorkspaceCase("mul_shape_dim_gt8_invalid",
                                {{1, 1, 1, 1, 1, 1, 1, 1, 1}, aclDataType::ACL_FLOAT},
                                {{1, 1, 1, 1, 1, 1, 1, 1, 1}, aclDataType::ACL_FLOAT},
                                {{1, 1, 1, 1, 1, 1, 1, 1, 1}, aclDataType::ACL_FLOAT},
                                ACLNN_ERR_PARAM_INVALID);
  failed += RunMulWorkspaceCase("mul_int8",
                                {{2, 3}, aclDataType::ACL_INT8},
                                {{2, 3}, aclDataType::ACL_INT8},
                                {{2, 3}, aclDataType::ACL_INT8},
                                ACL_SUCCESS);
  failed += RunMulWorkspaceCase("mul_uint8",
                                {{2, 3}, aclDataType::ACL_UINT8},
                                {{2, 3}, aclDataType::ACL_UINT8},
                                {{2, 3}, aclDataType::ACL_UINT8},
                                ACL_SUCCESS);
  failed += RunMulWorkspaceCase("mul_bool",
                                {{2, 3}, aclDataType::ACL_BOOL},
                                {{2, 3}, aclDataType::ACL_BOOL},
                                {{2, 3}, aclDataType::ACL_BOOL},
                                ACL_SUCCESS);
  failed += RunMulWorkspaceCase("mul_int16",
                                {{2, 3}, aclDataType::ACL_INT16},
                                {{2, 3}, aclDataType::ACL_INT16},
                                {{2, 3}, aclDataType::ACL_INT16},
                                ACL_SUCCESS);
  failed += RunMulWorkspaceCase("mul_int32",
                                {{2, 3}, aclDataType::ACL_INT32},
                                {{2, 3}, aclDataType::ACL_INT32},
                                {{2, 3}, aclDataType::ACL_INT32},
                                ACL_SUCCESS);
  failed += RunMulWorkspaceCase("mul_int64",
                                {{2, 3}, aclDataType::ACL_INT64},
                                {{2, 3}, aclDataType::ACL_INT64},
                                {{2, 3}, aclDataType::ACL_INT64},
                                ACL_SUCCESS);
  failed += RunMulWorkspaceCase("mul_double",
                                {{2, 3}, aclDataType::ACL_DOUBLE},
                                {{2, 3}, aclDataType::ACL_DOUBLE},
                                {{2, 3}, aclDataType::ACL_DOUBLE},
                                ACL_SUCCESS);
  failed += RunMulWorkspaceCase("mul_complex64",
                                {{2, 3}, aclDataType::ACL_COMPLEX64},
                                {{2, 3}, aclDataType::ACL_COMPLEX64},
                                {{2, 3}, aclDataType::ACL_COMPLEX64},
                                ACL_SUCCESS);
  failed += RunMulWorkspaceCase("mul_empty_tensor",
                                {{0, 3}, aclDataType::ACL_FLOAT16},
                                {{0, 3}, aclDataType::ACL_FLOAT16},
                                {{0, 3}, aclDataType::ACL_FLOAT16},
                                ACL_SUCCESS);
  failed += RunMulWorkspaceCase("mul_shape_mismatch",
                                {{2, 3}, aclDataType::ACL_FLOAT},
                                {{2, 3}, aclDataType::ACL_FLOAT},
                                {{3, 2}, aclDataType::ACL_FLOAT},
                                ACLNN_ERR_PARAM_INVALID);
  failed += RunMulWorkspaceCase("mul_non_nd_format",
                                {{1, 1, 2, 3}, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_NCHW},
                                {{1, 1, 2, 3}, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_NCHW},
                                {{1, 1, 2, 3}, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_NCHW},
                                ACL_SUCCESS);
  failed += RunMulWorkspaceCase("mul_non_contiguous_fp32",
                                {{2, 3}, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, 1, {4, 1}, {2, 4}},
                                {{2, 3}, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, 3, {4, 1}, {2, 4}},
                                {{2, 3}, aclDataType::ACL_FLOAT},
                                ACL_SUCCESS);
  failed += RunMulWorkspaceCase("mul_non_contiguous_double",
                                {{2, 3}, aclDataType::ACL_DOUBLE, aclFormat::ACL_FORMAT_ND, 5, {4, 1}, {2, 4}},
                                {{2, 3}, aclDataType::ACL_DOUBLE, aclFormat::ACL_FORMAT_ND, 7, {4, 1}, {2, 4}},
                                {{2, 3}, aclDataType::ACL_DOUBLE},
                                ACL_SUCCESS);
  failed += RunMulWorkspaceCase("mul_non_contiguous_complex64",
                                {{2, 3}, aclDataType::ACL_COMPLEX64, aclFormat::ACL_FORMAT_ND, 9, {4, 1}, {2, 4}},
                                {{2, 3}, aclDataType::ACL_COMPLEX64, aclFormat::ACL_FORMAT_ND, 11, {4, 1}, {2, 4}},
                                {{2, 3}, aclDataType::ACL_COMPLEX64},
                                ACL_SUCCESS);
  failed += RunMulWorkspaceCase("mul_undefined_self_dtype",
                                {{2, 3}, aclDataType::ACL_DT_UNDEFINED},
                                {{2, 3}, aclDataType::ACL_FLOAT},
                                {{2, 3}, aclDataType::ACL_FLOAT},
                                ACLNN_ERR_PARAM_INVALID);
  failed += RunMulWorkspaceCase("mul_undefined_other_dtype",
                                {{2, 3}, aclDataType::ACL_FLOAT},
                                {{2, 3}, aclDataType::ACL_DT_UNDEFINED},
                                {{2, 3}, aclDataType::ACL_FLOAT},
                                ACLNN_ERR_PARAM_INVALID);
  failed += RunMulWorkspaceCase("mul_undefined_out_dtype",
                                {{2, 3}, aclDataType::ACL_FLOAT},
                                {{2, 3}, aclDataType::ACL_FLOAT},
                                {{2, 3}, aclDataType::ACL_DT_UNDEFINED},
                                ACLNN_ERR_PARAM_INVALID);
  failed += RunMulNullptrCase("mul_nullptr", ACLNN_ERR_PARAM_NULLPTR);
  failed += RunMulPartialNullptrCase("mul_null_self", true, false, false, ACLNN_ERR_PARAM_NULLPTR);
  failed += RunMulPartialNullptrCase("mul_null_other", false, true, false, ACLNN_ERR_PARAM_NULLPTR);
  failed += RunMulPartialNullptrCase("mul_null_out", false, false, true, ACLNN_ERR_PARAM_NULLPTR);

  failed += RunInplaceMulWorkspaceCase("inplace_mul_empty_tensor",
                                       {{0, 3}, aclDataType::ACL_FLOAT16},
                                       {{0, 3}, aclDataType::ACL_FLOAT16},
                                       ACL_SUCCESS);
  failed += RunInplaceMulWorkspaceCase("inplace_mul_shape_mismatch",
                                       {{2, 3}, aclDataType::ACL_FLOAT},
                                       {{2, 1, 3}, aclDataType::ACL_FLOAT},
                                       ACLNN_ERR_PARAM_INVALID);
  failed += RunInplaceMulWorkspaceCase("inplace_mul_non_nd_format",
                                       {{1, 1, 2, 3}, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_NCHW},
                                       {{1, 1, 2, 3}, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_NCHW},
                                       ACL_SUCCESS);
  failed += RunInplaceMulWorkspaceCase("inplace_mul_undefined_self_dtype",
                                       {{2, 3}, aclDataType::ACL_DT_UNDEFINED},
                                       {{2, 3}, aclDataType::ACL_FLOAT},
                                       ACLNN_ERR_PARAM_INVALID);
  failed += RunInplaceMulWorkspaceCase("inplace_mul_undefined_other_dtype",
                                       {{2, 3}, aclDataType::ACL_FLOAT},
                                       {{2, 3}, aclDataType::ACL_DT_UNDEFINED},
                                       ACLNN_ERR_PARAM_INVALID);
  failed += RunInplaceMulNullptrCase("inplace_mul_nullptr", ACLNN_ERR_PARAM_NULLPTR);
  failed += RunInplaceMulPartialNullptrCase("inplace_mul_null_self", true, false, ACLNN_ERR_PARAM_NULLPTR);
  failed += RunInplaceMulPartialNullptrCase("inplace_mul_null_other", false, true, ACLNN_ERR_PARAM_NULLPTR);

  ScalarResource scalarFloat;
  ScalarResource scalarDouble;
  ScalarResource scalarBool;
  ScalarResource scalarComplex64;
  float scalarFloatValue = 1.75F;
  double scalarDoubleValue = 2.5;
  bool scalarBoolValue = true;
  Complex64Value scalarComplex64Value = {2.0F, -0.5F};
  failed += CreateScalarResource(scalarFloatValue, aclDataType::ACL_FLOAT, &scalarFloat);
  failed += CreateScalarResource(scalarDoubleValue, aclDataType::ACL_DOUBLE, &scalarDouble);
  failed += CreateScalarResource(scalarBoolValue, aclDataType::ACL_BOOL, &scalarBool);
  failed += CreateScalarResource(scalarComplex64Value, aclDataType::ACL_COMPLEX64, &scalarComplex64);

  failed += RunMulsWorkspaceCase("muls_complex64_scalar_float",
                                 {{2, 3}, aclDataType::ACL_COMPLEX64},
                                 scalarFloat.scalar,
                                 {{2, 3}, aclDataType::ACL_COMPLEX64},
                                 ACL_SUCCESS);
  failed += RunMulsWorkspaceCase("muls_fp16_scalar_complex64_invalid",
                                 {{2, 3}, aclDataType::ACL_FLOAT16},
                                 scalarComplex64.scalar,
                                 {{2, 3}, aclDataType::ACL_BOOL},
                                 ACLNN_ERR_PARAM_INVALID);
  failed += RunMulsWorkspaceCase("muls_bf16_scalar_complex64_invalid",
                                 {{2, 3}, aclDataType::ACL_BF16},
                                 scalarComplex64.scalar,
                                 {{2, 3}, aclDataType::ACL_BOOL},
                                 ACLNN_ERR_PARAM_INVALID);
  failed += RunMulsWorkspaceCase("muls_fp32_scalar_complex64_invalid",
                                 {{2, 3}, aclDataType::ACL_FLOAT},
                                 scalarComplex64.scalar,
                                 {{2, 3}, aclDataType::ACL_BOOL},
                                 ACLNN_ERR_PARAM_INVALID);
  failed += RunMulsWorkspaceCase("muls_double_scalar_complex64_invalid",
                                 {{2, 3}, aclDataType::ACL_DOUBLE},
                                 scalarComplex64.scalar,
                                 {{2, 3}, aclDataType::ACL_BOOL},
                                 ACLNN_ERR_PARAM_INVALID);
  failed += RunMulsWorkspaceCase("muls_double_scalar_bool",
                                 {{2, 3}, aclDataType::ACL_DOUBLE},
                                 scalarBool.scalar,
                                 {{2, 3}, aclDataType::ACL_DOUBLE},
                                 ACL_SUCCESS);
  failed += RunMulsWorkspaceCase("muls_int32_scalar_bool",
                                 {{2, 3}, aclDataType::ACL_INT32},
                                 scalarBool.scalar,
                                 {{2, 3}, aclDataType::ACL_INT32},
                                 ACL_SUCCESS);
  failed += RunMulsWorkspaceCase("muls_bool_scalar_bool",
                                 {{2, 3}, aclDataType::ACL_BOOL},
                                 scalarBool.scalar,
                                 {{2, 3}, aclDataType::ACL_BOOL},
                                 ACL_SUCCESS);
  failed += RunMulsWorkspaceCase("muls_empty_tensor",
                                 {{0, 3}, aclDataType::ACL_FLOAT16},
                                 scalarFloat.scalar,
                                 {{0, 3}, aclDataType::ACL_FLOAT16},
                                 ACL_SUCCESS);
  failed += RunMulsWorkspaceCase("muls_shape_mismatch",
                                 {{2, 3}, aclDataType::ACL_FLOAT},
                                 scalarFloat.scalar,
                                 {{3, 2}, aclDataType::ACL_FLOAT},
                                 ACLNN_ERR_PARAM_INVALID);
  failed += RunMulsWorkspaceCase("muls_invalid_out_dtype",
                                 {{2, 3}, aclDataType::ACL_BOOL},
                                 scalarComplex64.scalar,
                                 {{2, 3}, aclDataType::ACL_BOOL},
                                 ACLNN_ERR_PARAM_INVALID);
  failed += RunMulsWorkspaceCase("muls_non_nd_format",
                                 {{1, 1, 2, 3}, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_NCHW},
                                 scalarFloat.scalar,
                                 {{1, 1, 2, 3}, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_NCHW},
                                 ACL_SUCCESS);
  failed += RunMulsWorkspaceCase("muls_undefined_self_dtype",
                                 {{2, 3}, aclDataType::ACL_DT_UNDEFINED},
                                 scalarFloat.scalar,
                                 {{2, 3}, aclDataType::ACL_FLOAT},
                                 ACLNN_ERR_PARAM_INVALID);
  failed += RunMulsWorkspaceCase("muls_undefined_out_dtype",
                                 {{2, 3}, aclDataType::ACL_FLOAT},
                                 scalarFloat.scalar,
                                 {{2, 3}, aclDataType::ACL_DT_UNDEFINED},
                                 ACLNN_ERR_PARAM_INVALID);
  failed += RunMulsNullptrCase("muls_nullptr", ACLNN_ERR_PARAM_NULLPTR);
  failed += RunMulsPartialNullptrCase("muls_null_self", true, false, false, ACLNN_ERR_PARAM_NULLPTR);
  failed += RunMulsPartialNullptrCase("muls_null_scalar", false, true, false, ACLNN_ERR_PARAM_NULLPTR);
  failed += RunMulsPartialNullptrCase("muls_null_out", false, false, true, ACLNN_ERR_PARAM_NULLPTR);

  failed += RunInplaceMulsWorkspaceCase("inplace_muls_complex64_scalar_float",
                                        {{2, 3}, aclDataType::ACL_COMPLEX64},
                                        scalarFloat.scalar,
                                        ACL_SUCCESS);
  failed += RunInplaceMulsWorkspaceCase("inplace_muls_fp16_scalar_complex64_invalid",
                                        {{2, 3}, aclDataType::ACL_FLOAT16},
                                        scalarComplex64.scalar,
                                        ACLNN_ERR_PARAM_INVALID);
  failed += RunInplaceMulsWorkspaceCase("inplace_muls_bf16_scalar_complex64_invalid",
                                        {{2, 3}, aclDataType::ACL_BF16},
                                        scalarComplex64.scalar,
                                        ACLNN_ERR_PARAM_INVALID);
  failed += RunInplaceMulsWorkspaceCase("inplace_muls_fp32_scalar_complex64_invalid",
                                        {{2, 3}, aclDataType::ACL_FLOAT},
                                        scalarComplex64.scalar,
                                        ACLNN_ERR_PARAM_INVALID);
  failed += RunInplaceMulsWorkspaceCase("inplace_muls_double_scalar_complex64_invalid",
                                        {{2, 3}, aclDataType::ACL_DOUBLE},
                                        scalarComplex64.scalar,
                                        ACLNN_ERR_PARAM_INVALID);
  failed += RunInplaceMulsWorkspaceCase("inplace_muls_double_scalar_bool",
                                        {{2, 3}, aclDataType::ACL_DOUBLE},
                                        scalarBool.scalar,
                                        ACL_SUCCESS);
  failed += RunInplaceMulsWorkspaceCase("inplace_muls_int32_scalar_bool",
                                        {{2, 3}, aclDataType::ACL_INT32},
                                        scalarBool.scalar,
                                        ACL_SUCCESS);
  failed += RunInplaceMulsWorkspaceCase("inplace_muls_empty_tensor",
                                        {{0, 3}, aclDataType::ACL_FLOAT16},
                                        scalarFloat.scalar,
                                        ACL_SUCCESS);
  failed += RunInplaceMulsWorkspaceCase("inplace_muls_invalid_promote",
                                        {{2, 3}, aclDataType::ACL_BOOL},
                                        scalarComplex64.scalar,
                                        ACLNN_ERR_PARAM_INVALID);
  failed += RunInplaceMulsWorkspaceCase("inplace_muls_undefined_self_dtype",
                                        {{2, 3}, aclDataType::ACL_DT_UNDEFINED},
                                        scalarFloat.scalar,
                                        ACLNN_ERR_PARAM_INVALID);
  failed += RunInplaceMulsNullptrCase("inplace_muls_nullptr", ACLNN_ERR_PARAM_NULLPTR);
  failed += RunInplaceMulsPartialNullptrCase("inplace_muls_null_self", true, false, ACLNN_ERR_PARAM_NULLPTR);
  failed += RunInplaceMulsPartialNullptrCase("inplace_muls_null_scalar", false, true, ACLNN_ERR_PARAM_NULLPTR);

  DestroyScalar(&scalarFloat);
  DestroyScalar(&scalarDouble);
  DestroyScalar(&scalarBool);
  DestroyScalar(&scalarComplex64);

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  CHECK_RET(failed == 0, LOG_PRINT("mul integrated example failed, failed cases=%d\n", failed); return 1);
  LOG_PRINT("All mul integrated coverage cases PASS\n");
  return 0;
}
