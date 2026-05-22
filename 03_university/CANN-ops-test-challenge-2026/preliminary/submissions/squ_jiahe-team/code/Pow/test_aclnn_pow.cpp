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
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "aclnn/opdev/op_errno.h"
#include "../op_api/aclnn_exp2.h"
#include "../op_api/aclnn_pow.h"
#include "../op_api/aclnn_pow_tensor_tensor.h"

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
  aclFormat format = ACL_FORMAT_ND;
  std::vector<int64_t> strides;
  std::vector<int64_t> storageShape;
  uint8_t seed = 1;
};

struct TestStats {
  int passed = 0;
  int failed = 0;
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  if (shape.empty()) {
    return 1;
  }
  int64_t size = 1;
  for (int64_t dim : shape) {
    size *= dim;
  }
  return size;
}

size_t GetDataTypeSize(aclDataType dataType)
{
  switch (dataType) {
    case ACL_BOOL:
    case ACL_INT8:
    case ACL_UINT8:
      return 1;
    case ACL_FLOAT16:
    case ACL_BF16:
    case ACL_INT16:
      return 2;
    case ACL_FLOAT:
    case ACL_INT32:
      return 4;
    case ACL_DOUBLE:
    case ACL_INT64:
      return 8;
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
std::string FormatScalarValue(T value)
{
  std::ostringstream oss;
  if constexpr (std::is_same_v<T, int8_t>) {
    oss << static_cast<int>(value);
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    oss << static_cast<unsigned int>(value);
  } else if constexpr (std::is_same_v<T, bool>) {
    oss << (value ? "true" : "false");
  } else {
    oss << value;
  }
  return oss.str();
}

template <typename T>
std::string FormatSample(const std::vector<T>& values, size_t maxCount = 6)
{
  std::ostringstream oss;
  oss << "[";
  const size_t count = values.size() < maxCount ? values.size() : maxCount;
  for (size_t i = 0; i < count; ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << FormatScalarValue(values[i]);
  }
  if (values.size() > maxCount) {
    oss << ", ...";
  }
  oss << "]";
  return oss.str();
}

void LogCasePass(const std::string& caseName)
{
  LOG_PRINT("[PASS] %s\n", caseName.c_str());
}

void LogCaseFail(const std::string& caseName, const std::string& detail)
{
  LOG_PRINT("[FAIL] %s: %s\n", caseName.c_str(), detail.c_str());
}

void RecordCase(TestStats* stats, bool pass)
{
  if (pass) {
    ++stats->passed;
  } else {
    ++stats->failed;
  }
}

bool EnableExtendedRuntimeCases()
{
  const char* value = std::getenv("POW_ENABLE_EXTENDED_RUNTIME");
  return value != nullptr && std::string(value) == "1";
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType,
                    TensorResource* resource, aclFormat format = ACL_FORMAT_ND)
{
  const size_t tensorSize = hostData.size() * sizeof(T);
  if (tensorSize > 0) {
    auto ret = aclrtMalloc(&resource->deviceAddr, tensorSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(resource->deviceAddr, tensorSize, hostData.data(), tensorSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  }

  auto strides = MakeContiguousStrides(shape);
  resource->tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(),
                                     shape.size(), resource->deviceAddr);
  CHECK_RET(resource->tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
  return ACL_SUCCESS;
}

int CreateAclTensorRaw(const TensorSpec& spec, TensorResource* resource)
{
  const auto& storageShape = spec.storageShape.empty() ? spec.shape : spec.storageShape;
  const auto& strides = spec.strides.empty() ? MakeContiguousStrides(spec.shape) : spec.strides;
  const size_t typeSize = GetDataTypeSize(spec.dataType);
  CHECK_RET(typeSize > 0, LOG_PRINT("unsupported dtype for raw tensor: %d\n", static_cast<int>(spec.dataType));
            return ACL_ERROR_FAILURE);

  const size_t elementCount = static_cast<size_t>(GetShapeSize(storageShape));
  std::vector<uint8_t> hostBytes(elementCount * typeSize, 0);
  for (size_t i = 0; i < hostBytes.size(); ++i) {
    hostBytes[i] = static_cast<uint8_t>(spec.seed + (i % 31));
  }

  if (!hostBytes.empty()) {
    auto ret = aclrtMalloc(&resource->deviceAddr, hostBytes.size(), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(resource->deviceAddr, hostBytes.size(), hostBytes.data(), hostBytes.size(),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  }

  resource->tensor = aclCreateTensor(spec.shape.data(), spec.shape.size(), spec.dataType, strides.data(), 0,
                                     spec.format, storageShape.data(), storageShape.size(), resource->deviceAddr);
  CHECK_RET(resource->tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
  return ACL_SUCCESS;
}

template <typename T>
int CopyDeviceToHost(const TensorResource& resource, std::vector<T>* hostData)
{
  if (hostData->empty()) {
    return ACL_SUCCESS;
  }
  auto ret = aclrtMemcpy(hostData->data(), hostData->size() * sizeof(T), resource.deviceAddr,
                         hostData->size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

template <typename T>
int CreateScalarResource(const T& value, aclDataType dataType, ScalarResource* resource)
{
  T valueCopy = value;
  resource->scalar = aclCreateScalar(&valueCopy, dataType);
  CHECK_RET(resource->scalar != nullptr, LOG_PRINT("aclCreateScalar failed.\n"); return ACL_ERROR_FAILURE);
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

bool ExpectStatus(const std::string& caseName, aclnnStatus actual, aclnnStatus expected)
{
  if (actual != expected) {
    std::ostringstream oss;
    oss << "status mismatch, actual=" << actual << ", expected=" << expected;
    LogCaseFail(caseName, oss.str());
    return false;
  }
  LogCasePass(caseName);
  return true;
}

bool IsFiniteMatch(double actual, double expected)
{
  if (std::isnan(actual) && std::isnan(expected)) {
    return true;
  }
  if (std::isinf(actual) && std::isinf(expected)) {
    return std::signbit(actual) == std::signbit(expected);
  }
  return false;
}

template <typename T>
bool ValidateResult(const std::string& caseName, const std::string& apiName, const std::string& inputSummary,
                    const std::vector<int64_t>& outShape, const std::vector<T>& actual,
                    const std::vector<T>& expected, double atol, double rtol)
{
  if (actual.size() != expected.size()) {
    LogCaseFail(caseName, "result size mismatch");
    return false;
  }

  double maxAbsDiff = 0.0;
  double maxRelDiff = 0.0;
  size_t worstIndex = 0;

  for (size_t i = 0; i < actual.size(); ++i) {
    if constexpr (std::is_floating_point_v<T>) {
      const double actualValue = static_cast<double>(actual[i]);
      const double expectedValue = static_cast<double>(expected[i]);
      if (IsFiniteMatch(actualValue, expectedValue)) {
        continue;
      }
      const double absDiff = std::fabs(actualValue - expectedValue);
      const double relDiff = std::fabs(expectedValue) > 0.0 ? absDiff / std::fabs(expectedValue) : absDiff;
      const double limit = atol + rtol * std::fabs(expectedValue);
      if (absDiff > maxAbsDiff) {
        maxAbsDiff = absDiff;
        maxRelDiff = relDiff;
        worstIndex = i;
      }
      if (!(absDiff <= limit)) {
        LOG_PRINT("[Precision][%s]\n", caseName.c_str());
        LOG_PRINT("  API: %s\n", apiName.c_str());
        LOG_PRINT("  Inputs: %s\n", inputSummary.c_str());
        LOG_PRINT("  Shape: %s\n", FormatShape(outShape).c_str());
        LOG_PRINT("  Expected sample: %s\n", FormatSample(expected).c_str());
        LOG_PRINT("  Actual sample:   %s\n", FormatSample(actual).c_str());
        LOG_PRINT("  Worst index: %zu, actual=%g, expected=%g, abs_diff=%g, rel_diff=%g, limit=%g\n", i,
                  actualValue, expectedValue, absDiff, relDiff, limit);
        LogCaseFail(caseName, "precision mismatch");
        return false;
      }
    } else {
      if (actual[i] != expected[i]) {
        LOG_PRINT("[Precision][%s]\n", caseName.c_str());
        LOG_PRINT("  API: %s\n", apiName.c_str());
        LOG_PRINT("  Inputs: %s\n", inputSummary.c_str());
        LOG_PRINT("  Shape: %s\n", FormatShape(outShape).c_str());
        LOG_PRINT("  Expected sample: %s\n", FormatSample(expected).c_str());
        LOG_PRINT("  Actual sample:   %s\n", FormatSample(actual).c_str());
        LOG_PRINT("  Worst index: %zu, actual=%s, expected=%s\n", i, FormatScalarValue(actual[i]).c_str(),
                  FormatScalarValue(expected[i]).c_str());
        LogCaseFail(caseName, "value mismatch");
        return false;
      }
    }
  }

  LOG_PRINT("[Precision][%s]\n", caseName.c_str());
  LOG_PRINT("  API: %s\n", apiName.c_str());
  LOG_PRINT("  Inputs: %s\n", inputSummary.c_str());
  LOG_PRINT("  Shape: %s\n", FormatShape(outShape).c_str());
  LOG_PRINT("  Expected sample: %s\n", FormatSample(expected).c_str());
  LOG_PRINT("  Actual sample:   %s\n", FormatSample(actual).c_str());
  if constexpr (std::is_floating_point_v<T>) {
    LOG_PRINT("  Max abs diff: %g\n", maxAbsDiff);
    LOG_PRINT("  Max rel diff: %g\n", maxRelDiff);
    LOG_PRINT("  Worst index: %zu\n", worstIndex);
    LOG_PRINT("  Threshold: atol=%g, rtol=%g\n", atol, rtol);
  } else {
    LOG_PRINT("  Exact compare: true\n");
  }
  LogCasePass(caseName);
  return true;
}

std::vector<size_t> ComputeStrides(const std::vector<int64_t>& shape)
{
  std::vector<size_t> strides(shape.size(), 1);
  if (shape.empty()) {
    return strides;
  }
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = static_cast<size_t>(shape[static_cast<size_t>(i + 1)]) *
                                      strides[static_cast<size_t>(i + 1)];
  }
  return strides;
}

std::vector<int64_t> PadShapeLeft(const std::vector<int64_t>& shape, size_t rank)
{
  std::vector<int64_t> padded(rank, 1);
  const size_t offset = rank - shape.size();
  for (size_t i = 0; i < shape.size(); ++i) {
    padded[offset + i] = shape[i];
  }
  return padded;
}

size_t ComputeBroadcastOffset(size_t linearIndex, const std::vector<int64_t>& outShape,
                              const std::vector<int64_t>& inShape)
{
  if (inShape.empty()) {
    return 0;
  }

  const auto paddedShape = PadShapeLeft(inShape, outShape.size());
  const auto inStrides = ComputeStrides(paddedShape);
  const auto outStrides = ComputeStrides(outShape);
  size_t offset = 0;
  size_t remainder = linearIndex;
  for (size_t dim = 0; dim < outShape.size(); ++dim) {
    const size_t coord = outStrides[dim] == 0 ? 0 : remainder / outStrides[dim];
    remainder = outStrides[dim] == 0 ? 0 : remainder % outStrides[dim];
    const size_t inCoord = paddedShape[dim] == 1 ? 0 : coord;
    offset += inCoord * inStrides[dim];
  }
  return offset;
}

template <typename TensorT, typename OutT>
std::vector<OutT> ComputeTensorScalarExpected(const std::vector<TensorT>& input, const TensorT& exponent)
{
  std::vector<OutT> expected(input.size(), static_cast<OutT>(0));
  for (size_t i = 0; i < input.size(); ++i) {
    expected[i] = static_cast<OutT>(std::pow(static_cast<double>(input[i]), static_cast<double>(exponent)));
  }
  return expected;
}

template <typename TensorT, typename OutT>
std::vector<OutT> ComputeScalarTensorExpected(const TensorT& self, const std::vector<TensorT>& exponent)
{
  std::vector<OutT> expected(exponent.size(), static_cast<OutT>(0));
  for (size_t i = 0; i < exponent.size(); ++i) {
    expected[i] = static_cast<OutT>(std::pow(static_cast<double>(self), static_cast<double>(exponent[i])));
  }
  return expected;
}

template <typename BaseT, typename ExpT, typename OutT>
std::vector<OutT> ComputeTensorTensorExpected(const std::vector<BaseT>& base, const std::vector<int64_t>& baseShape,
                                              const std::vector<ExpT>& exponent,
                                              const std::vector<int64_t>& exponentShape,
                                              const std::vector<int64_t>& outShape)
{
  std::vector<OutT> expected(static_cast<size_t>(GetShapeSize(outShape)), static_cast<OutT>(0));
  for (size_t i = 0; i < expected.size(); ++i) {
    const size_t baseOffset = ComputeBroadcastOffset(i, outShape, baseShape);
    const size_t expOffset = ComputeBroadcastOffset(i, outShape, exponentShape);
    expected[i] = static_cast<OutT>(std::pow(static_cast<double>(base[baseOffset]),
                                             static_cast<double>(exponent[expOffset])));
  }
  return expected;
}

template <typename InputT, typename OutT = InputT>
std::vector<OutT> ComputeExp2Expected(const std::vector<InputT>& input)
{
  std::vector<OutT> expected(input.size(), static_cast<OutT>(0));
  for (size_t i = 0; i < input.size(); ++i) {
    expected[i] = static_cast<OutT>(std::pow(2.0, static_cast<double>(input[i])));
  }
  return expected;
}

template <typename TensorT, typename ScalarT, typename OutT>
bool RunTensorScalarPrecisionCase(TestStats* stats, const std::string& caseName, const std::vector<int64_t>& shape,
                                  const std::vector<TensorT>& input, const ScalarT& exponent,
                                  const std::vector<OutT>& expected, aclDataType tensorType,
                                  aclDataType scalarType, aclDataType outType, aclrtStream stream,
                                  double atol = 1e-5, double rtol = 1e-5)
{
  TensorResource self;
  TensorResource out;
  ScalarResource expScalar;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  std::vector<OutT> actual(expected.size(), static_cast<OutT>(0));
  std::vector<OutT> outInit(expected.size(), static_cast<OutT>(0));
  int ret = ACL_SUCCESS;
  bool pass = false;

  ret = CreateAclTensor(input, shape, tensorType, &self);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateAclTensor(outInit, shape, outType, &out);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateScalarResource(exponent, scalarType, &expScalar);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  ret = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, expScalar.scalar, out.tensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s workspace failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s workspace alloc failed. ERROR: %d\n", caseName.c_str(), ret);
              goto cleanup);
  }
  ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s run failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s sync failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  ret = CopyDeviceToHost(out, &actual);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  {
    std::ostringstream summary;
    summary << "self_shape=" << FormatShape(shape) << ", exponent=" << FormatScalarValue(exponent);
    pass = ValidateResult(caseName, "aclnnPowTensorScalar", summary.str(), shape, actual, expected, atol, rtol);
  }

cleanup:
  if (ret != ACL_SUCCESS && !pass) {
    LogCaseFail(caseName, "runtime failure");
  }
  DestroyTensor(&self);
  DestroyTensor(&out);
  DestroyScalar(&expScalar);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  RecordCase(stats, pass);
  return pass;
}

template <typename TensorT, typename ScalarT>
bool RunInplaceTensorScalarPrecisionCase(TestStats* stats, const std::string& caseName,
                                         const std::vector<int64_t>& shape, const std::vector<TensorT>& input,
                                         const ScalarT& exponent, const std::vector<TensorT>& expected,
                                         aclDataType tensorType, aclDataType scalarType, aclrtStream stream,
                                         double atol = 1e-5, double rtol = 1e-5)
{
  TensorResource self;
  ScalarResource expScalar;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  std::vector<TensorT> actual(expected.size(), static_cast<TensorT>(0));
  int ret = ACL_SUCCESS;
  bool pass = false;

  ret = CreateAclTensor(input, shape, tensorType, &self);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateScalarResource(exponent, scalarType, &expScalar);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  ret = aclnnInplacePowTensorScalarGetWorkspaceSize(self.tensor, expScalar.scalar, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s workspace failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s workspace alloc failed. ERROR: %d\n", caseName.c_str(), ret);
              goto cleanup);
  }
  ret = aclnnInplacePowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s run failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s sync failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  ret = CopyDeviceToHost(self, &actual);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  {
    std::ostringstream summary;
    summary << "self_shape=" << FormatShape(shape) << ", exponent=" << FormatScalarValue(exponent)
            << ", inplace=true";
    pass = ValidateResult(caseName, "aclnnInplacePowTensorScalar", summary.str(), shape, actual, expected, atol,
                          rtol);
  }

cleanup:
  if (ret != ACL_SUCCESS && !pass) {
    LogCaseFail(caseName, "runtime failure");
  }
  DestroyTensor(&self);
  DestroyScalar(&expScalar);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  RecordCase(stats, pass);
  return pass;
}

template <typename TensorT, typename ScalarT, typename OutT>
bool RunScalarTensorPrecisionCase(TestStats* stats, const std::string& caseName,
                                  const std::vector<int64_t>& shape, const ScalarT& selfValue,
                                  const std::vector<TensorT>& exponent, const std::vector<OutT>& expected,
                                  aclDataType scalarType, aclDataType exponentType, aclDataType outType,
                                  aclrtStream stream, double atol = 1e-5, double rtol = 1e-5)
{
  TensorResource expTensor;
  TensorResource out;
  ScalarResource selfScalar;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  std::vector<OutT> actual(expected.size(), static_cast<OutT>(0));
  std::vector<OutT> outInit(expected.size(), static_cast<OutT>(0));
  int ret = ACL_SUCCESS;
  bool pass = false;

  ret = CreateAclTensor(exponent, shape, exponentType, &expTensor);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateAclTensor(outInit, shape, outType, &out);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateScalarResource(selfValue, scalarType, &selfScalar);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  ret = aclnnPowScalarTensorGetWorkspaceSize(selfScalar.scalar, expTensor.tensor, out.tensor, &workspaceSize,
                                             &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s workspace failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s workspace alloc failed. ERROR: %d\n", caseName.c_str(), ret);
              goto cleanup);
  }
  ret = aclnnPowScalarTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s run failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s sync failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  ret = CopyDeviceToHost(out, &actual);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  {
    std::ostringstream summary;
    summary << "scalar=" << FormatScalarValue(selfValue) << ", exponent_shape=" << FormatShape(shape);
    pass = ValidateResult(caseName, "aclnnPowScalarTensor", summary.str(), shape, actual, expected, atol, rtol);
  }

cleanup:
  if (ret != ACL_SUCCESS && !pass) {
    LogCaseFail(caseName, "runtime failure");
  }
  DestroyTensor(&expTensor);
  DestroyTensor(&out);
  DestroyScalar(&selfScalar);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  RecordCase(stats, pass);
  return pass;
}

template <typename BaseT, typename ExpT, typename OutT>
bool RunTensorTensorPrecisionCase(TestStats* stats, const std::string& caseName,
                                  const std::vector<int64_t>& baseShape, const std::vector<int64_t>& exponentShape,
                                  const std::vector<int64_t>& outShape, const std::vector<BaseT>& base,
                                  const std::vector<ExpT>& exponent, const std::vector<OutT>& expected,
                                  aclDataType baseType, aclDataType exponentType, aclDataType outType,
                                  aclrtStream stream, double atol = 1e-5, double rtol = 1e-5)
{
  TensorResource self;
  TensorResource expTensor;
  TensorResource out;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  std::vector<OutT> actual(expected.size(), static_cast<OutT>(0));
  std::vector<OutT> outInit(expected.size(), static_cast<OutT>(0));
  int ret = ACL_SUCCESS;
  bool pass = false;

  ret = CreateAclTensor(base, baseShape, baseType, &self);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateAclTensor(exponent, exponentShape, exponentType, &expTensor);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateAclTensor(outInit, outShape, outType, &out);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  ret = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, expTensor.tensor, out.tensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s workspace failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s workspace alloc failed. ERROR: %d\n", caseName.c_str(), ret);
              goto cleanup);
  }
  ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s run failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s sync failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  ret = CopyDeviceToHost(out, &actual);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  {
    std::ostringstream summary;
    summary << "self_shape=" << FormatShape(baseShape) << ", exponent_shape=" << FormatShape(exponentShape)
            << ", out_shape=" << FormatShape(outShape);
    pass = ValidateResult(caseName, "aclnnPowTensorTensor", summary.str(), outShape, actual, expected, atol, rtol);
  }

cleanup:
  if (ret != ACL_SUCCESS && !pass) {
    LogCaseFail(caseName, "runtime failure");
  }
  DestroyTensor(&self);
  DestroyTensor(&expTensor);
  DestroyTensor(&out);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  RecordCase(stats, pass);
  return pass;
}

template <typename T>
bool RunInplaceTensorTensorPrecisionCase(TestStats* stats, const std::string& caseName,
                                         const std::vector<int64_t>& selfShape,
                                         const std::vector<int64_t>& exponentShape,
                                         const std::vector<T>& selfData, const std::vector<T>& exponentData,
                                         const std::vector<T>& expected, aclDataType dataType, aclrtStream stream)
{
  TensorResource self;
  TensorResource expTensor;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  std::vector<T> actual(expected.size(), static_cast<T>(0));
  int ret = ACL_SUCCESS;
  bool pass = false;

  ret = CreateAclTensor(selfData, selfShape, dataType, &self);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateAclTensor(exponentData, exponentShape, dataType, &expTensor);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  ret = aclnnInplacePowTensorTensorGetWorkspaceSize(self.tensor, expTensor.tensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s workspace failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s workspace alloc failed. ERROR: %d\n", caseName.c_str(), ret);
              goto cleanup);
  }
  ret = aclnnInplacePowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s run failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s sync failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  ret = CopyDeviceToHost(self, &actual);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  {
    std::ostringstream summary;
    summary << "self_shape=" << FormatShape(selfShape) << ", exponent_shape=" << FormatShape(exponentShape)
            << ", inplace=true";
    pass = ValidateResult(caseName, "aclnnInplacePowTensorTensor", summary.str(), selfShape, actual, expected, 0.0,
                          0.0);
  }

cleanup:
  if (ret != ACL_SUCCESS && !pass) {
    LogCaseFail(caseName, "runtime failure");
  }
  DestroyTensor(&self);
  DestroyTensor(&expTensor);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  RecordCase(stats, pass);
  return pass;
}

template <typename InputT, typename OutT>
bool RunExp2PrecisionCase(TestStats* stats, const std::string& caseName, const std::vector<int64_t>& shape,
                          const std::vector<InputT>& input, const std::vector<OutT>& expected, aclDataType selfType,
                          aclDataType outType, aclrtStream stream, double atol = 1e-5, double rtol = 1e-5)
{
  TensorResource self;
  TensorResource out;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  std::vector<OutT> actual(expected.size(), static_cast<OutT>(0));
  std::vector<OutT> outInit(expected.size(), static_cast<OutT>(0));
  int ret = ACL_SUCCESS;
  bool pass = false;

  ret = CreateAclTensor(input, shape, selfType, &self);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
  ret = CreateAclTensor(outInit, shape, outType, &out);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  ret = aclnnExp2GetWorkspaceSize(self.tensor, out.tensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s workspace failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s workspace alloc failed. ERROR: %d\n", caseName.c_str(), ret);
              goto cleanup);
  }
  ret = aclnnExp2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s run failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s sync failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  ret = CopyDeviceToHost(out, &actual);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  {
    std::ostringstream summary;
    summary << "self_shape=" << FormatShape(shape);
    pass = ValidateResult(caseName, "aclnnExp2", summary.str(), shape, actual, expected, atol, rtol);
  }

cleanup:
  if (ret != ACL_SUCCESS && !pass) {
    LogCaseFail(caseName, "runtime failure");
  }
  DestroyTensor(&self);
  DestroyTensor(&out);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  RecordCase(stats, pass);
  return pass;
}

template <typename T>
bool RunInplaceExp2PrecisionCase(TestStats* stats, const std::string& caseName, const std::vector<int64_t>& shape,
                                 const std::vector<T>& input, const std::vector<T>& expected, aclDataType dataType,
                                 aclrtStream stream, double atol = 1e-5, double rtol = 1e-5)
{
  TensorResource self;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  std::vector<T> actual(expected.size(), static_cast<T>(0));
  int ret = ACL_SUCCESS;
  bool pass = false;

  ret = CreateAclTensor(input, shape, dataType, &self);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  ret = aclnnInplaceExp2GetWorkspaceSize(self.tensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s workspace failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s workspace alloc failed. ERROR: %d\n", caseName.c_str(), ret);
              goto cleanup);
  }
  ret = aclnnInplaceExp2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s run failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s sync failed. ERROR: %d\n", caseName.c_str(), ret); goto cleanup);
  ret = CopyDeviceToHost(self, &actual);
  CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

  {
    std::ostringstream summary;
    summary << "self_shape=" << FormatShape(shape) << ", inplace=true";
    pass = ValidateResult(caseName, "aclnnInplaceExp2", summary.str(), shape, actual, expected, atol, rtol);
  }

cleanup:
  if (ret != ACL_SUCCESS && !pass) {
    LogCaseFail(caseName, "runtime failure");
  }
  DestroyTensor(&self);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  RecordCase(stats, pass);
  return pass;
}

bool RunWorkspaceStatusCase(TestStats* stats, const std::string& caseName, aclnnStatus actual, aclnnStatus expected)
{
  const bool pass = ExpectStatus(caseName, actual, expected);
  RecordCase(stats, pass);
  return pass;
}

bool RunTensorScalarStatusCases(TestStats* stats)
{
  bool allPass = true;

  {
    TensorResource self;
    TensorResource out;
    ScalarResource exponent;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = CreateAclTensor(std::vector<float>{1.0f, 2.0f}, {2}, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = CreateAclTensor(std::vector<float>{0.0f, 0.0f}, {2}, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&self); return false);
    ret = CreateScalarResource(2.0f, ACL_FLOAT, &exponent);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&self); DestroyTensor(&out); return false);

    allPass &= RunWorkspaceStatusCase(stats, "TensorScalar_NullSelf",
                                      aclnnPowTensorScalarGetWorkspaceSize(nullptr, exponent.scalar, out.tensor,
                                                                           &workspaceSize, &executor),
                                      ACLNN_ERR_PARAM_NULLPTR);
    allPass &= RunWorkspaceStatusCase(stats, "TensorScalar_NullExponent",
                                      aclnnPowTensorScalarGetWorkspaceSize(self.tensor, nullptr, out.tensor,
                                                                           &workspaceSize, &executor),
                                      ACLNN_ERR_PARAM_NULLPTR);
    allPass &= RunWorkspaceStatusCase(stats, "TensorScalar_NullOut",
                                      aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent.scalar, nullptr,
                                                                           &workspaceSize, &executor),
                                      ACLNN_ERR_PARAM_NULLPTR);

    DestroyTensor(&self);
    DestroyTensor(&out);
    DestroyScalar(&exponent);
  }

  {
    TensorResource self;
    TensorResource out;
    ScalarResource exponent;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = CreateAclTensor(std::vector<float>{1.0f, 2.0f}, {2}, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = CreateAclTensor(std::vector<float>{0.0f, 0.0f, 0.0f}, {3}, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&self); return false);
    ret = CreateScalarResource(2.0f, ACL_FLOAT, &exponent);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&self); DestroyTensor(&out); return false);

    allPass &= RunWorkspaceStatusCase(stats, "TensorScalar_OutShapeMismatch",
                                      aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent.scalar, out.tensor,
                                                                           &workspaceSize, &executor),
                                      ACLNN_ERR_PARAM_INVALID);

    DestroyTensor(&self);
    DestroyTensor(&out);
    DestroyScalar(&exponent);
  }

  {
    TensorResource self;
    TensorResource out;
    ScalarResource exponent;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = CreateAclTensor(std::vector<int32_t>{1, 2, 3}, {3}, ACL_INT32, &self);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = CreateAclTensor(std::vector<int32_t>{0, 0, 0}, {3}, ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&self); return false);
    ret = CreateScalarResource(-1, ACL_INT32, &exponent);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&self); DestroyTensor(&out); return false);

    allPass &= RunWorkspaceStatusCase(stats, "TensorScalar_IntegerNegativeExponentInvalid",
                                      aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent.scalar, out.tensor,
                                                                           &workspaceSize, &executor),
                                      ACLNN_ERR_PARAM_INVALID);

    DestroyTensor(&self);
    DestroyTensor(&out);
    DestroyScalar(&exponent);
  }

  {
    TensorResource self;
    TensorResource out;
    ScalarResource exponent;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = CreateAclTensor(std::vector<float>{}, {0}, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = CreateAclTensor(std::vector<float>{}, {0}, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&self); return false);
    ret = CreateScalarResource(2.0f, ACL_FLOAT, &exponent);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&self); DestroyTensor(&out); return false);

    allPass &= RunWorkspaceStatusCase(stats, "TensorScalar_EmptyTensor",
                                      aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent.scalar, out.tensor,
                                                                           &workspaceSize, &executor),
                                      ACL_SUCCESS);

    DestroyTensor(&self);
    DestroyTensor(&out);
    DestroyScalar(&exponent);
  }

  return allPass;
}

bool RunScalarTensorStatusCases(TestStats* stats)
{
  bool allPass = true;

  {
    TensorResource exponent;
    TensorResource out;
    ScalarResource self;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = CreateAclTensor(std::vector<float>{1.0f, 2.0f}, {2}, ACL_FLOAT, &exponent);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = CreateAclTensor(std::vector<float>{0.0f, 0.0f}, {2}, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&exponent); return false);
    ret = CreateScalarResource(2.0f, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&exponent); DestroyTensor(&out); return false);

    allPass &= RunWorkspaceStatusCase(stats, "ScalarTensor_NullExponent",
                                      aclnnPowScalarTensorGetWorkspaceSize(self.scalar, nullptr, out.tensor,
                                                                           &workspaceSize, &executor),
                                      ACLNN_ERR_PARAM_NULLPTR);
    allPass &= RunWorkspaceStatusCase(stats, "ScalarTensor_NullOut",
                                      aclnnPowScalarTensorGetWorkspaceSize(self.scalar, exponent.tensor, nullptr,
                                                                           &workspaceSize, &executor),
                                      ACLNN_ERR_PARAM_NULLPTR);

    DestroyTensor(&exponent);
    DestroyTensor(&out);
    DestroyScalar(&self);
  }

  {
    TensorResource exponent;
    TensorResource out;
    ScalarResource self;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = CreateAclTensor(std::vector<float>{1.0f, 2.0f}, {2}, ACL_FLOAT, &exponent);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = CreateAclTensor(std::vector<float>{0.0f, 0.0f, 0.0f}, {3}, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&exponent); return false);
    ret = CreateScalarResource(2.0f, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&exponent); DestroyTensor(&out); return false);

    allPass &= RunWorkspaceStatusCase(stats, "ScalarTensor_OutShapeMismatch",
                                      aclnnPowScalarTensorGetWorkspaceSize(self.scalar, exponent.tensor, out.tensor,
                                                                           &workspaceSize, &executor),
                                      ACLNN_ERR_PARAM_INVALID);

    DestroyTensor(&exponent);
    DestroyTensor(&out);
    DestroyScalar(&self);
  }

  {
    TensorResource exponent;
    TensorResource out;
    ScalarResource self;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = CreateAclTensor(std::vector<float>{}, {0}, ACL_FLOAT, &exponent);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = CreateAclTensor(std::vector<float>{}, {0}, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&exponent); return false);
    ret = CreateScalarResource(2.0f, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&exponent); DestroyTensor(&out); return false);

    allPass &= RunWorkspaceStatusCase(stats, "ScalarTensor_EmptyTensor",
                                      aclnnPowScalarTensorGetWorkspaceSize(self.scalar, exponent.tensor, out.tensor,
                                                                           &workspaceSize, &executor),
                                      ACL_SUCCESS);

    DestroyTensor(&exponent);
    DestroyTensor(&out);
    DestroyScalar(&self);
  }

  return allPass;
}

bool RunTensorTensorStatusCases(TestStats* stats)
{
  bool allPass = true;

  {
    TensorResource self;
    TensorResource exponent;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = CreateAclTensor(std::vector<float>{1.0f, 2.0f}, {2}, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = CreateAclTensor(std::vector<float>{1.0f, 2.0f}, {2}, ACL_FLOAT, &exponent);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&self); return false);
    ret = CreateAclTensor(std::vector<float>{0.0f, 0.0f}, {2}, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&self); DestroyTensor(&exponent); return false);

    allPass &= RunWorkspaceStatusCase(stats, "TensorTensor_NullSelf",
                                      aclnnPowTensorTensorGetWorkspaceSize(nullptr, exponent.tensor, out.tensor,
                                                                           &workspaceSize, &executor),
                                      ACLNN_ERR_PARAM_NULLPTR);
    allPass &= RunWorkspaceStatusCase(stats, "TensorTensor_NullExponent",
                                      aclnnPowTensorTensorGetWorkspaceSize(self.tensor, nullptr, out.tensor,
                                                                           &workspaceSize, &executor),
                                      ACLNN_ERR_PARAM_NULLPTR);
    allPass &= RunWorkspaceStatusCase(stats, "TensorTensor_NullOut",
                                      aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, nullptr,
                                                                           &workspaceSize, &executor),
                                      ACLNN_ERR_PARAM_NULLPTR);

    DestroyTensor(&self);
    DestroyTensor(&exponent);
    DestroyTensor(&out);
  }

  {
    TensorResource self;
    TensorResource exponent;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = CreateAclTensor(std::vector<float>{1.0f, 2.0f}, {2}, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = CreateAclTensor(std::vector<float>{1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT, &exponent);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&self); return false);
    ret = CreateAclTensor(std::vector<float>{0.0f, 0.0f, 0.0f}, {3}, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&self); DestroyTensor(&exponent); return false);

    allPass &= RunWorkspaceStatusCase(stats, "TensorTensor_BroadcastInvalid",
                                      aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor,
                                                                           &workspaceSize, &executor),
                                      ACLNN_ERR_PARAM_INVALID);

    DestroyTensor(&self);
    DestroyTensor(&exponent);
    DestroyTensor(&out);
  }

  {
    TensorResource self;
    TensorResource exponent;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = CreateAclTensor(std::vector<float>{1.0f, 2.0f}, {1, 2}, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = CreateAclTensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, ACL_FLOAT, &exponent);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&self); return false);
    ret = CreateAclTensor(std::vector<float>{0.0f, 0.0f}, {1, 2}, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&self); DestroyTensor(&exponent); return false);

    allPass &= RunWorkspaceStatusCase(stats, "TensorTensor_OutShapeMismatch",
                                      aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor,
                                                                           &workspaceSize, &executor),
                                      ACLNN_ERR_PARAM_INVALID);

    DestroyTensor(&self);
    DestroyTensor(&exponent);
    DestroyTensor(&out);
  }

  {
    TensorResource self;
    TensorResource exponent;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = CreateAclTensorRaw({{2}, ACL_BOOL}, &self);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = CreateAclTensorRaw({{2}, ACL_BOOL}, &exponent);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&self); return false);
    ret = CreateAclTensorRaw({{2}, ACL_BOOL}, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&self); DestroyTensor(&exponent); return false);

    allPass &= RunWorkspaceStatusCase(stats, "TensorTensor_BoolBoolInvalid",
                                      aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor,
                                                                           &workspaceSize, &executor),
                                      ACLNN_ERR_PARAM_INVALID);

    DestroyTensor(&self);
    DestroyTensor(&exponent);
    DestroyTensor(&out);
  }

  {
    TensorResource self;
    TensorResource exponent;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = CreateAclTensor(std::vector<float>{}, {0, 2}, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = CreateAclTensor(std::vector<float>{1.0f, 2.0f}, {1, 2}, ACL_FLOAT, &exponent);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&self); return false);
    ret = CreateAclTensor(std::vector<float>{}, {0, 2}, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&self); DestroyTensor(&exponent); return false);

    allPass &= RunWorkspaceStatusCase(stats, "TensorTensor_EmptyTensor",
                                      aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor,
                                                                           &workspaceSize, &executor),
                                      ACL_SUCCESS);

    DestroyTensor(&self);
    DestroyTensor(&exponent);
    DestroyTensor(&out);
  }

  return allPass;
}

bool RunExp2StatusCases(TestStats* stats)
{
  bool allPass = true;

  {
    TensorResource self;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = CreateAclTensor(std::vector<float>{1.0f, 2.0f}, {2}, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = CreateAclTensor(std::vector<float>{0.0f, 0.0f}, {2}, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&self); return false);

    allPass &= RunWorkspaceStatusCase(stats, "Exp2_NullSelf",
                                      aclnnExp2GetWorkspaceSize(nullptr, out.tensor, &workspaceSize, &executor),
                                      ACLNN_ERR_PARAM_NULLPTR);
    allPass &= RunWorkspaceStatusCase(stats, "Exp2_NullOut",
                                      aclnnExp2GetWorkspaceSize(self.tensor, nullptr, &workspaceSize, &executor),
                                      ACLNN_ERR_PARAM_NULLPTR);

    DestroyTensor(&self);
    DestroyTensor(&out);
  }

  {
    TensorResource self;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = CreateAclTensor(std::vector<float>{1.0f, 2.0f}, {2}, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = CreateAclTensor(std::vector<float>{0.0f, 0.0f, 0.0f}, {3}, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&self); return false);

    allPass &= RunWorkspaceStatusCase(stats, "Exp2_ShapeMismatch",
                                      aclnnExp2GetWorkspaceSize(self.tensor, out.tensor, &workspaceSize, &executor),
                                      ACLNN_ERR_PARAM_INVALID);

    DestroyTensor(&self);
    DestroyTensor(&out);
  }

  {
    TensorResource self;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = CreateAclTensor(std::vector<int32_t>{1, 2}, {2}, ACL_INT32, &self);
    CHECK_RET(ret == ACL_SUCCESS, return false);

    allPass &= RunWorkspaceStatusCase(stats, "InplaceExp2_Int32Invalid",
                                      aclnnInplaceExp2GetWorkspaceSize(self.tensor, &workspaceSize, &executor),
                                      ACLNN_ERR_PARAM_INVALID);

    DestroyTensor(&self);
  }

  {
    TensorResource self;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = CreateAclTensor(std::vector<float>{}, {0}, ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = CreateAclTensor(std::vector<float>{}, {0}, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&self); return false);

    allPass &= RunWorkspaceStatusCase(stats, "Exp2_EmptyTensor",
                                      aclnnExp2GetWorkspaceSize(self.tensor, out.tensor, &workspaceSize, &executor),
                                      ACL_SUCCESS);

    DestroyTensor(&self);
    DestroyTensor(&out);
  }

  return allPass;
}

bool RunTensorTensorDtypeCoverageCases(TestStats* stats)
{
  struct DtypeCase {
    const char* name;
    aclDataType dtype;
    uint8_t seed;
  };

  const std::vector<DtypeCase> cases = {
    {"TensorTensor_Float16OpKey", ACL_FLOAT16, 3},
    {"TensorTensor_BFloat16OpKey", ACL_BF16, 5},
    {"TensorTensor_Float32OpKey", ACL_FLOAT, 7},
    {"TensorTensor_Uint8OpKey", ACL_UINT8, 11},
    {"TensorTensor_Int8OpKey", ACL_INT8, 13},
    {"TensorTensor_Int16OpKey", ACL_INT16, 17},
    {"TensorTensor_Int32OpKey", ACL_INT32, 19},
  };

  bool allPass = true;
  for (const auto& item : cases) {
    TensorResource self;
    TensorResource exponent;
    TensorResource out;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto ret = CreateAclTensorRaw({{2, 3}, item.dtype, ACL_FORMAT_ND, {}, {}, item.seed}, &self);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ret = CreateAclTensorRaw({{2, 3}, item.dtype, ACL_FORMAT_ND, {}, {}, static_cast<uint8_t>(item.seed + 1)},
                             &exponent);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&self); return false);
    ret = CreateAclTensorRaw({{2, 3}, item.dtype, ACL_FORMAT_ND, {}, {}, static_cast<uint8_t>(item.seed + 2)}, &out);
    CHECK_RET(ret == ACL_SUCCESS, DestroyTensor(&self); DestroyTensor(&exponent); return false);

    const bool pass = RunWorkspaceStatusCase(stats, item.name,
      aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor, &workspaceSize, &executor),
      ACL_SUCCESS);
    allPass &= pass;

    DestroyTensor(&self);
    DestroyTensor(&exponent);
    DestroyTensor(&out);
  }
  return allPass;
}

}  // namespace

int main()
{
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  TestStats stats;

  LOG_PRINT("========== Pow Coverage And Precision Test ==========\n");

  RunTensorScalarPrecisionCase(&stats, "TensorScalar_Float_Generic_4p1",
                               {2, 3},
                               std::vector<float>{0.25f, 1.0f, 2.0f, 3.0f, 4.0f, 1.5f},
                               4.1f,
                               ComputeTensorScalarExpected<float, float>(
                                 std::vector<float>{0.25f, 1.0f, 2.0f, 3.0f, 4.0f, 1.5f}, 4.1f),
                               ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, stream, 1e-5, 1e-5);

  RunTensorScalarPrecisionCase(&stats, "TensorScalar_Float_Sqrt",
                               {4},
                               std::vector<float>{0.25f, 1.0f, 4.0f, 9.0f},
                               0.5f,
                               ComputeTensorScalarExpected<float, float>(
                                 std::vector<float>{0.25f, 1.0f, 4.0f, 9.0f}, 0.5f),
                               ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, stream, 1e-5, 1e-5);

  if (EnableExtendedRuntimeCases()) {
    LOG_PRINT("[INFO] POW_ENABLE_EXTENDED_RUNTIME=1, run extended runtime precision cases.\n");

    RunTensorScalarPrecisionCase(&stats, "TensorScalar_Int32_Cube",
                                 {5},
                                 std::vector<int32_t>{-2, -1, 0, 3, 4},
                                 3,
                                 ComputeTensorScalarExpected<int32_t, int32_t>(
                                   std::vector<int32_t>{-2, -1, 0, 3, 4}, 3),
                                 ACL_INT32, ACL_INT32, ACL_INT32, stream, 0.0, 0.0);

    RunInplaceTensorScalarPrecisionCase(&stats, "InplaceTensorScalar_Float_Reciprocal",
                                        {4},
                                        std::vector<float>{0.5f, 2.0f, 4.0f, 8.0f},
                                        -1.0f,
                                        ComputeTensorScalarExpected<float, float>(
                                          std::vector<float>{0.5f, 2.0f, 4.0f, 8.0f}, -1.0f),
                                        ACL_FLOAT, ACL_FLOAT, stream, 1e-5, 1e-5);

    RunScalarTensorPrecisionCase(&stats, "ScalarTensor_Float_Generic",
                                 {6},
                                 2.0f,
                                 std::vector<float>{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f},
                                 ComputeScalarTensorExpected<float, float>(
                                   2.0f, std::vector<float>{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f}),
                                 ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, stream, 1e-5, 1e-5);

    RunTensorTensorPrecisionCase(&stats, "TensorTensor_Float_Broadcast",
                                 {2, 1, 3}, {1, 4, 1}, {2, 4, 3},
                                 std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 0.5f, 1.5f},
                                 std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f},
                                 ComputeTensorTensorExpected<float, float, float>(
                                   std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 0.5f, 1.5f}, {2, 1, 3},
                                   std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f}, {1, 4, 1},
                                   {2, 4, 3}),
                                 ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, stream, 1e-5, 1e-5);

    RunInplaceTensorTensorPrecisionCase(&stats, "InplaceTensorTensor_Int32_Exact",
                                        {2, 2}, {2, 2},
                                        std::vector<int32_t>{2, 3, 4, 5},
                                        std::vector<int32_t>{1, 2, 3, 0},
                                        ComputeTensorTensorExpected<int32_t, int32_t, int32_t>(
                                          std::vector<int32_t>{2, 3, 4, 5}, {2, 2},
                                          std::vector<int32_t>{1, 2, 3, 0}, {2, 2}, {2, 2}),
                                        ACL_INT32, stream);

    RunExp2PrecisionCase(&stats, "Exp2_Float32_MainPath",
                         {6},
                         std::vector<float>{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f},
                         ComputeExp2Expected(std::vector<float>{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f}),
                         ACL_FLOAT, ACL_FLOAT, stream, 1e-5, 1e-5);

    RunExp2PrecisionCase(&stats, "Exp2_Int32_CastToFloat",
                         {5},
                         std::vector<int32_t>{-3, -1, 0, 2, 4},
                         ComputeExp2Expected<int32_t, float>(std::vector<int32_t>{-3, -1, 0, 2, 4}),
                         ACL_INT32, ACL_FLOAT, stream, 1e-5, 1e-5);

    RunInplaceExp2PrecisionCase(&stats, "InplaceExp2_Float32",
                                {4},
                                std::vector<float>{-3.0f, -0.5f, 0.5f, 4.0f},
                                ComputeExp2Expected(std::vector<float>{-3.0f, -0.5f, 0.5f, 4.0f}),
                                ACL_FLOAT, stream, 1e-5, 1e-5);
  } else {
    LOG_PRINT("[INFO] Skip extended runtime cases by default. Set POW_ENABLE_EXTENDED_RUNTIME=1 to enable them.\n");
  }

  RunTensorScalarStatusCases(&stats);
  RunScalarTensorStatusCases(&stats);
  RunTensorTensorStatusCases(&stats);
  RunExp2StatusCases(&stats);
  RunTensorTensorDtypeCoverageCases(&stats);

  LOG_PRINT("=====================================================\n");
  LOG_PRINT("Total cases: %d\n", stats.passed + stats.failed);
  LOG_PRINT("Passed: %d\n", stats.passed);
  LOG_PRINT("Failed: %d\n", stats.failed);

  if (stream != nullptr) {
    aclrtDestroyStream(stream);
  }
  aclrtResetDevice(deviceId);
  aclFinalize();
  return stats.failed == 0 ? 0 : 1;
}
