/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto dim : shape) {
    shapeSize *= dim;
  }
  return shapeSize;
}

std::vector<int64_t> MakeContiguousStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
  }
  return strides;
}

float Float16ToFloat(uint16_t value) {
  const unsigned int sign = (value >> 15) & 0x1;
  const unsigned int exponent = (value >> 10) & 0x1f;
  const unsigned int mantissa = value & 0x3ff;

  float result = 0.0f;
  if (exponent == 0) {
    result = static_cast<float>(mantissa) * 0.0000019073486328125f;
  } else if (exponent == 31) {
    result = (mantissa == 0U) ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
  } else {
    result =
        (1.0f + static_cast<float>(mantissa) * 0.0009765625f) * std::pow(2.0f, static_cast<int>(exponent) - 15);
  }
  return sign ? -result : result;
}

uint16_t FloatToFloat16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000);
  const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
  const uint32_t mantissa = bits & 0x7fffff;

  if (exponent <= 0) {
    return sign;
  }
  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00);
  }
  return static_cast<uint16_t>(sign | (static_cast<uint16_t>(exponent) << 10) | static_cast<uint16_t>(mantissa >> 13));
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

struct DeviceTensor {
  void* addr = nullptr;
  aclTensor* tensor = nullptr;
  size_t bytes = 0;
};

struct DeviceWorkspace {
  void* addr = nullptr;
  uint64_t size = 0;
};

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType,
                    DeviceTensor* out) {
  out->bytes = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
  auto ret = aclrtMalloc(&out->addr, out->bytes, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  ret = aclrtMemcpy(out->addr, out->bytes, hostData.data(), out->bytes, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides = MakeContiguousStrides(shape);
  out->tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                                shape.data(), shape.size(), out->addr);
  CHECK_RET(out->tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
  return ACL_SUCCESS;
}

template <typename T>
int CopyDeviceToHost(const DeviceTensor& deviceTensor, std::vector<T>* outHost) {
  auto ret = aclrtMemcpy(outHost->data(), outHost->size() * sizeof(T), deviceTensor.addr,
                         outHost->size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

void DestroyDeviceTensor(DeviceTensor* tensor) {
  if (tensor->tensor != nullptr) {
    aclDestroyTensor(tensor->tensor);
    tensor->tensor = nullptr;
  }
  if (tensor->addr != nullptr) {
    aclrtFree(tensor->addr);
    tensor->addr = nullptr;
  }
  tensor->bytes = 0;
}

int AllocWorkspace(uint64_t size, DeviceWorkspace* workspace) {
  workspace->size = size;
  workspace->addr = nullptr;
  if (size == 0) {
    return ACL_SUCCESS;
  }
  auto ret = aclrtMalloc(&workspace->addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

void FreeWorkspace(DeviceWorkspace* workspace) {
  if (workspace->addr != nullptr) {
    aclrtFree(workspace->addr);
    workspace->addr = nullptr;
  }
  workspace->size = 0;
}

bool NearlyEqual(double actual, double expected, double atol = 1e-5, double rtol = 1e-4) {
  if (std::isnan(expected)) {
    return std::isnan(actual);
  }
  if (std::isinf(expected)) {
    return std::isinf(actual) && (std::signbit(actual) == std::signbit(expected));
  }
  return std::fabs(actual - expected) <= (atol + rtol * std::fabs(expected));
}

template <typename T>
bool CheckVectorClose(const std::vector<T>& actual, const std::vector<double>& expected, double atol, double rtol,
                      std::string* message) {
  if (actual.size() != expected.size()) {
    *message = "size mismatch";
    return false;
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!NearlyEqual(static_cast<double>(actual[i]), expected[i], atol, rtol)) {
      *message = "first mismatch at index " + std::to_string(i) + ", actual=" + std::to_string(actual[i]) +
                 ", expected=" + std::to_string(expected[i]);
      return false;
    }
  }
  return true;
}

std::vector<double> ExpectedPowTensorScalar(const std::vector<float>& base, double exponent) {
  std::vector<double> expected(base.size(), 0.0);
  for (size_t i = 0; i < base.size(); ++i) {
    expected[i] = std::pow(static_cast<double>(base[i]), exponent);
  }
  return expected;
}

std::vector<double> ExpectedPowScalarTensor(double base, const std::vector<float>& exponent) {
  std::vector<double> expected(exponent.size(), 0.0);
  for (size_t i = 0; i < exponent.size(); ++i) {
    expected[i] = std::pow(base, static_cast<double>(exponent[i]));
  }
  return expected;
}

std::vector<int64_t> MakeCoords(int64_t linearIndex, const std::vector<int64_t>& shape) {
  std::vector<int64_t> coords(shape.size(), 0);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    const int64_t dim = shape[static_cast<size_t>(i)];
    coords[static_cast<size_t>(i)] = linearIndex % dim;
    linearIndex /= dim;
  }
  return coords;
}

int64_t CoordsToLinear(const std::vector<int64_t>& coords, const std::vector<int64_t>& shape) {
  const std::vector<int64_t> strides = MakeContiguousStrides(shape);
  int64_t index = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    index += coords[i] * strides[i];
  }
  return index;
}

int64_t BroadcastLinearIndex(const std::vector<int64_t>& outCoords, const std::vector<int64_t>& inShape) {
  std::vector<int64_t> inCoords(inShape.size(), 0);
  const int64_t outRank = static_cast<int64_t>(outCoords.size());
  const int64_t inRank = static_cast<int64_t>(inShape.size());
  for (int64_t i = inRank - 1; i >= 0; --i) {
    const int64_t outAxis = outRank - (inRank - i);
    const int64_t coord = outCoords[static_cast<size_t>(outAxis)];
    inCoords[static_cast<size_t>(i)] = (inShape[static_cast<size_t>(i)] == 1) ? 0 : coord;
  }
  return CoordsToLinear(inCoords, inShape);
}

std::vector<double> ExpectedPowTensorTensor(const std::vector<float>& base, const std::vector<int64_t>& baseShape,
                                            const std::vector<float>& exponent,
                                            const std::vector<int64_t>& exponentShape,
                                            const std::vector<int64_t>& outShape) {
  const int64_t outSize = GetShapeSize(outShape);
  std::vector<double> expected(static_cast<size_t>(outSize), 0.0);
  for (int64_t i = 0; i < outSize; ++i) {
    const std::vector<int64_t> outCoords = MakeCoords(i, outShape);
    const int64_t baseIdx = BroadcastLinearIndex(outCoords, baseShape);
    const int64_t expIdx = BroadcastLinearIndex(outCoords, exponentShape);
    expected[static_cast<size_t>(i)] = std::pow(static_cast<double>(base[static_cast<size_t>(baseIdx)]),
                                                static_cast<double>(exponent[static_cast<size_t>(expIdx)]));
  }
  return expected;
}

void ReportCase(const std::string& name, bool pass, const std::string& detail) {
  if (pass) {
    LOG_PRINT("[PASS] %s\n", name.c_str());
  } else {
    LOG_PRINT("[FAIL] %s: %s\n", name.c_str(), detail.c_str());
  }
}

bool RunPowTensorScalarCase(aclrtStream stream, const std::string& caseName, const std::vector<float>& selfHost,
                            const std::vector<int64_t>& shape, float exponentVal, double atol = 1e-5,
                            double rtol = 1e-4) {
  DeviceTensor self;
  DeviceTensor out;
  DeviceWorkspace workspace;
  aclScalar* exponent = nullptr;
  aclOpExecutor* executor = nullptr;

  const std::vector<float> outInit(selfHost.size(), 0.0f);
  int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    ReportCase(caseName, false, "create self tensor failed");
    return false;
  }
  ret = CreateAclTensor(outInit, shape, ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    DestroyDeviceTensor(&self);
    ReportCase(caseName, false, "create out tensor failed");
    return false;
  }
  exponent = aclCreateScalar(&exponentVal, ACL_FLOAT);
  if (exponent == nullptr) {
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "create scalar failed");
    return false;
  }

  uint64_t workspaceSize = 0;
  ret = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent, out.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    aclDestroyScalar(exponent);
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "GetWorkspaceSize failed");
    return false;
  }

  ret = AllocWorkspace(workspaceSize, &workspace);
  if (ret != ACL_SUCCESS) {
    aclDestroyScalar(exponent);
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "allocate workspace failed");
    return false;
  }

  ret = aclnnPowTensorScalar(workspace.addr, workspace.size, executor, stream);
  if (ret != ACL_SUCCESS) {
    FreeWorkspace(&workspace);
    aclDestroyScalar(exponent);
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "run op failed");
    return false;
  }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    FreeWorkspace(&workspace);
    aclDestroyScalar(exponent);
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "stream sync failed");
    return false;
  }

  std::vector<float> actual(selfHost.size(), 0.0f);
  ret = CopyDeviceToHost(out, &actual);
  std::string detail;
  bool pass = (ret == ACL_SUCCESS) &&
              CheckVectorClose(actual, ExpectedPowTensorScalar(selfHost, static_cast<double>(exponentVal)), atol, rtol,
                               &detail);
  ReportCase(caseName, pass, detail.empty() ? "compare failed" : detail);

  FreeWorkspace(&workspace);
  aclDestroyScalar(exponent);
  DestroyDeviceTensor(&self);
  DestroyDeviceTensor(&out);
  return pass;
}

bool RunInplacePowTensorScalarCase(aclrtStream stream, const std::string& caseName, const std::vector<float>& selfHost,
                                   const std::vector<int64_t>& shape, float exponentVal, double atol = 1e-5,
                                   double rtol = 1e-4) {
  DeviceTensor self;
  DeviceWorkspace workspace;
  aclScalar* exponent = nullptr;
  aclOpExecutor* executor = nullptr;

  int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    ReportCase(caseName, false, "create self tensor failed");
    return false;
  }

  exponent = aclCreateScalar(&exponentVal, ACL_FLOAT);
  if (exponent == nullptr) {
    DestroyDeviceTensor(&self);
    ReportCase(caseName, false, "create scalar failed");
    return false;
  }

  uint64_t workspaceSize = 0;
  ret = aclnnInplacePowTensorScalarGetWorkspaceSize(self.tensor, exponent, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    aclDestroyScalar(exponent);
    DestroyDeviceTensor(&self);
    ReportCase(caseName, false, "GetWorkspaceSize failed");
    return false;
  }

  ret = AllocWorkspace(workspaceSize, &workspace);
  if (ret != ACL_SUCCESS) {
    aclDestroyScalar(exponent);
    DestroyDeviceTensor(&self);
    ReportCase(caseName, false, "allocate workspace failed");
    return false;
  }

  ret = aclnnInplacePowTensorScalar(workspace.addr, workspace.size, executor, stream);
  if (ret != ACL_SUCCESS) {
    FreeWorkspace(&workspace);
    aclDestroyScalar(exponent);
    DestroyDeviceTensor(&self);
    ReportCase(caseName, false, "run op failed");
    return false;
  }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    FreeWorkspace(&workspace);
    aclDestroyScalar(exponent);
    DestroyDeviceTensor(&self);
    ReportCase(caseName, false, "stream sync failed");
    return false;
  }

  std::vector<float> actual(selfHost.size(), 0.0f);
  ret = CopyDeviceToHost(self, &actual);
  std::string detail;
  bool pass = (ret == ACL_SUCCESS) &&
              CheckVectorClose(actual, ExpectedPowTensorScalar(selfHost, static_cast<double>(exponentVal)), atol, rtol,
                               &detail);
  ReportCase(caseName, pass, detail.empty() ? "compare failed" : detail);

  FreeWorkspace(&workspace);
  aclDestroyScalar(exponent);
  DestroyDeviceTensor(&self);
  return pass;
}

bool RunPowScalarTensorCase(aclrtStream stream, const std::string& caseName, float baseScalar,
                            const std::vector<float>& exponentHost, const std::vector<int64_t>& shape,
                            double atol = 1e-5, double rtol = 1e-4) {
  DeviceTensor exponent;
  DeviceTensor out;
  DeviceWorkspace workspace;
  aclScalar* base = nullptr;
  aclOpExecutor* executor = nullptr;

  const std::vector<float> outInit(exponentHost.size(), 0.0f);
  int ret = CreateAclTensor(exponentHost, shape, ACL_FLOAT, &exponent);
  if (ret != ACL_SUCCESS) {
    ReportCase(caseName, false, "create exponent tensor failed");
    return false;
  }
  ret = CreateAclTensor(outInit, shape, ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    DestroyDeviceTensor(&exponent);
    ReportCase(caseName, false, "create out tensor failed");
    return false;
  }
  base = aclCreateScalar(&baseScalar, ACL_FLOAT);
  if (base == nullptr) {
    DestroyDeviceTensor(&exponent);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "create scalar failed");
    return false;
  }

  uint64_t workspaceSize = 0;
  ret = aclnnPowScalarTensorGetWorkspaceSize(base, exponent.tensor, out.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    aclDestroyScalar(base);
    DestroyDeviceTensor(&exponent);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "GetWorkspaceSize failed");
    return false;
  }

  ret = AllocWorkspace(workspaceSize, &workspace);
  if (ret != ACL_SUCCESS) {
    aclDestroyScalar(base);
    DestroyDeviceTensor(&exponent);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "allocate workspace failed");
    return false;
  }

  ret = aclnnPowScalarTensor(workspace.addr, workspace.size, executor, stream);
  if (ret != ACL_SUCCESS) {
    FreeWorkspace(&workspace);
    aclDestroyScalar(base);
    DestroyDeviceTensor(&exponent);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "run op failed");
    return false;
  }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    FreeWorkspace(&workspace);
    aclDestroyScalar(base);
    DestroyDeviceTensor(&exponent);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "stream sync failed");
    return false;
  }

  std::vector<float> actual(exponentHost.size(), 0.0f);
  ret = CopyDeviceToHost(out, &actual);
  std::string detail;
  bool pass = (ret == ACL_SUCCESS) &&
              CheckVectorClose(actual, ExpectedPowScalarTensor(static_cast<double>(baseScalar), exponentHost), atol,
                               rtol, &detail);
  ReportCase(caseName, pass, detail.empty() ? "compare failed" : detail);

  FreeWorkspace(&workspace);
  aclDestroyScalar(base);
  DestroyDeviceTensor(&exponent);
  DestroyDeviceTensor(&out);
  return pass;
}

bool RunPowTensorTensorFloatCase(aclrtStream stream, const std::string& caseName, const std::vector<float>& selfHost,
                                 const std::vector<int64_t>& selfShape, const std::vector<float>& expHost,
                                 const std::vector<int64_t>& expShape, const std::vector<int64_t>& outShape,
                                 double atol = 1e-5, double rtol = 1e-4) {
  DeviceTensor self;
  DeviceTensor exponent;
  DeviceTensor out;
  DeviceWorkspace workspace;
  aclOpExecutor* executor = nullptr;

  const std::vector<float> outInit(static_cast<size_t>(GetShapeSize(outShape)), 0.0f);
  int ret = CreateAclTensor(selfHost, selfShape, ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    ReportCase(caseName, false, "create self tensor failed");
    return false;
  }
  ret = CreateAclTensor(expHost, expShape, ACL_FLOAT, &exponent);
  if (ret != ACL_SUCCESS) {
    DestroyDeviceTensor(&self);
    ReportCase(caseName, false, "create exponent tensor failed");
    return false;
  }
  ret = CreateAclTensor(outInit, outShape, ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    ReportCase(caseName, false, "create out tensor failed");
    return false;
  }

  uint64_t workspaceSize = 0;
  ret = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "GetWorkspaceSize failed");
    return false;
  }

  ret = AllocWorkspace(workspaceSize, &workspace);
  if (ret != ACL_SUCCESS) {
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "allocate workspace failed");
    return false;
  }

  ret = aclnnPowTensorTensor(workspace.addr, workspace.size, executor, stream);
  if (ret != ACL_SUCCESS) {
    FreeWorkspace(&workspace);
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "run op failed");
    return false;
  }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    FreeWorkspace(&workspace);
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "stream sync failed");
    return false;
  }

  std::vector<float> actual(static_cast<size_t>(GetShapeSize(outShape)), 0.0f);
  ret = CopyDeviceToHost(out, &actual);
  std::string detail;
  const std::vector<double> expected = ExpectedPowTensorTensor(selfHost, selfShape, expHost, expShape, outShape);
  bool pass = (ret == ACL_SUCCESS) && CheckVectorClose(actual, expected, atol, rtol, &detail);
  ReportCase(caseName, pass, detail.empty() ? "compare failed" : detail);

  FreeWorkspace(&workspace);
  DestroyDeviceTensor(&self);
  DestroyDeviceTensor(&exponent);
  DestroyDeviceTensor(&out);
  return pass;
}

template <typename T>
bool RunPowTensorTensorIntegerCase(aclrtStream stream, const std::string& caseName, const std::vector<T>& selfHost,
                                   const std::vector<int64_t>& selfShape, const std::vector<T>& expHost,
                                   const std::vector<int64_t>& expShape, const std::vector<int64_t>& outShape,
                                   aclDataType dtype) {
  DeviceTensor self;
  DeviceTensor exponent;
  DeviceTensor out;
  DeviceWorkspace workspace;
  aclOpExecutor* executor = nullptr;

  const size_t outSize = static_cast<size_t>(GetShapeSize(outShape));
  const std::vector<T> outInit(outSize, static_cast<T>(0));
  int ret = CreateAclTensor(selfHost, selfShape, dtype, &self);
  if (ret != ACL_SUCCESS) {
    ReportCase(caseName, false, "create self tensor failed");
    return false;
  }
  ret = CreateAclTensor(expHost, expShape, dtype, &exponent);
  if (ret != ACL_SUCCESS) {
    DestroyDeviceTensor(&self);
    ReportCase(caseName, false, "create exponent tensor failed");
    return false;
  }
  ret = CreateAclTensor(outInit, outShape, dtype, &out);
  if (ret != ACL_SUCCESS) {
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    ReportCase(caseName, false, "create out tensor failed");
    return false;
  }

  uint64_t workspaceSize = 0;
  ret = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "GetWorkspaceSize failed");
    return false;
  }

  ret = AllocWorkspace(workspaceSize, &workspace);
  if (ret != ACL_SUCCESS) {
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "allocate workspace failed");
    return false;
  }

  ret = aclnnPowTensorTensor(workspace.addr, workspace.size, executor, stream);
  if (ret != ACL_SUCCESS) {
    FreeWorkspace(&workspace);
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "run op failed");
    return false;
  }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    FreeWorkspace(&workspace);
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "stream sync failed");
    return false;
  }

  std::vector<T> actual(outSize, static_cast<T>(0));
  ret = CopyDeviceToHost(out, &actual);
  std::vector<float> selfAsFloat(selfHost.begin(), selfHost.end());
  std::vector<float> expAsFloat(expHost.begin(), expHost.end());
  const std::vector<double> expected =
      ExpectedPowTensorTensor(selfAsFloat, selfShape, expAsFloat, expShape, outShape);

  std::string detail;
  bool pass = (ret == ACL_SUCCESS) && CheckVectorClose(actual, expected, 0.0, 0.0, &detail);
  ReportCase(caseName, pass, detail.empty() ? "compare failed" : detail);

  FreeWorkspace(&workspace);
  DestroyDeviceTensor(&self);
  DestroyDeviceTensor(&exponent);
  DestroyDeviceTensor(&out);
  return pass;
}

bool RunPowTensorTensorInt32Case(aclrtStream stream, const std::string& caseName, const std::vector<int32_t>& selfHost,
                                 const std::vector<int64_t>& shape, const std::vector<int32_t>& expHost,
                                 const std::vector<int64_t>& expShape, const std::vector<int64_t>& outShape) {
  return RunPowTensorTensorIntegerCase<int32_t>(stream, caseName, selfHost, shape, expHost, expShape, outShape,
                                                ACL_INT32);
}

bool RunPowTensorTensorInt8Case(aclrtStream stream, const std::string& caseName, const std::vector<int8_t>& selfHost,
                                const std::vector<int8_t>& expHost, const std::vector<int64_t>& shape) {
  return RunPowTensorTensorIntegerCase<int8_t>(stream, caseName, selfHost, shape, expHost, shape, shape, ACL_INT8);
}

bool RunPowTensorTensorUint8Case(aclrtStream stream, const std::string& caseName,
                                 const std::vector<uint8_t>& selfHost, const std::vector<uint8_t>& expHost,
                                 const std::vector<int64_t>& shape) {
  return RunPowTensorTensorIntegerCase<uint8_t>(stream, caseName, selfHost, shape, expHost, shape, shape,
                                                 ACL_UINT8);
}

bool RunPowTensorTensorInt16Case(aclrtStream stream, const std::string& caseName,
                                 const std::vector<int16_t>& selfHost, const std::vector<int16_t>& expHost,
                                 const std::vector<int64_t>& shape) {
  return RunPowTensorTensorIntegerCase<int16_t>(stream, caseName, selfHost, shape, expHost, shape, shape,
                                                 ACL_INT16);
}

bool RunPowTensorTensorFloat16Case(aclrtStream stream, const std::string& caseName,
                                   const std::vector<uint16_t>& selfHost, const std::vector<uint16_t>& expHost,
                                   const std::vector<int64_t>& shape, double atol = 1e-2, double rtol = 1e-2) {
  DeviceTensor self;
  DeviceTensor exponent;
  DeviceTensor out;
  DeviceWorkspace workspace;
  aclOpExecutor* executor = nullptr;

  const std::vector<uint16_t> outInit(selfHost.size(), 0);
  int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT16, &self);
  if (ret != ACL_SUCCESS) {
    ReportCase(caseName, false, "create self tensor failed");
    return false;
  }
  ret = CreateAclTensor(expHost, shape, ACL_FLOAT16, &exponent);
  if (ret != ACL_SUCCESS) {
    DestroyDeviceTensor(&self);
    ReportCase(caseName, false, "create exponent tensor failed");
    return false;
  }
  ret = CreateAclTensor(outInit, shape, ACL_FLOAT16, &out);
  if (ret != ACL_SUCCESS) {
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    ReportCase(caseName, false, "create out tensor failed");
    return false;
  }

  uint64_t workspaceSize = 0;
  ret = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "GetWorkspaceSize failed");
    return false;
  }

  ret = AllocWorkspace(workspaceSize, &workspace);
  if (ret != ACL_SUCCESS) {
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "allocate workspace failed");
    return false;
  }

  ret = aclnnPowTensorTensor(workspace.addr, workspace.size, executor, stream);
  if (ret != ACL_SUCCESS) {
    FreeWorkspace(&workspace);
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "run op failed");
    return false;
  }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    FreeWorkspace(&workspace);
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "stream sync failed");
    return false;
  }

  std::vector<uint16_t> actualRaw(selfHost.size(), 0);
  ret = CopyDeviceToHost(out, &actualRaw);

  std::vector<float> selfFloat(selfHost.size(), 0.0f);
  std::vector<float> expFloat(expHost.size(), 0.0f);
  std::vector<float> actual(selfHost.size(), 0.0f);
  for (size_t i = 0; i < selfHost.size(); ++i) {
    selfFloat[i] = Float16ToFloat(selfHost[i]);
    expFloat[i] = Float16ToFloat(expHost[i]);
    actual[i] = Float16ToFloat(actualRaw[i]);
  }
  const std::vector<double> expected = ExpectedPowTensorTensor(selfFloat, shape, expFloat, shape, shape);

  std::string detail;
  bool pass = (ret == ACL_SUCCESS) && CheckVectorClose(actual, expected, atol, rtol, &detail);
  ReportCase(caseName, pass, detail.empty() ? "compare failed" : detail);

  FreeWorkspace(&workspace);
  DestroyDeviceTensor(&self);
  DestroyDeviceTensor(&exponent);
  DestroyDeviceTensor(&out);
  return pass;
}

bool RunPowTensorTensorDoubleCase(aclrtStream stream, const std::string& caseName,
                                  const std::vector<double>& selfHost, const std::vector<double>& expHost,
                                  const std::vector<int64_t>& shape, double atol = 1e-10,
                                  double rtol = 1e-10) {
  DeviceTensor self;
  DeviceTensor exponent;
  DeviceTensor out;
  DeviceWorkspace workspace;
  aclOpExecutor* executor = nullptr;

  const std::vector<double> outInit(selfHost.size(), 0.0);
  int ret = CreateAclTensor(selfHost, shape, ACL_DOUBLE, &self);
  if (ret != ACL_SUCCESS) {
    ReportCase(caseName, false, "create self tensor failed");
    return false;
  }
  ret = CreateAclTensor(expHost, shape, ACL_DOUBLE, &exponent);
  if (ret != ACL_SUCCESS) {
    DestroyDeviceTensor(&self);
    ReportCase(caseName, false, "create exponent tensor failed");
    return false;
  }
  ret = CreateAclTensor(outInit, shape, ACL_DOUBLE, &out);
  if (ret != ACL_SUCCESS) {
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    ReportCase(caseName, false, "create out tensor failed");
    return false;
  }

  uint64_t workspaceSize = 0;
  ret = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "GetWorkspaceSize failed");
    return false;
  }

  ret = AllocWorkspace(workspaceSize, &workspace);
  if (ret != ACL_SUCCESS) {
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "allocate workspace failed");
    return false;
  }

  ret = aclnnPowTensorTensor(workspace.addr, workspace.size, executor, stream);
  if (ret != ACL_SUCCESS) {
    FreeWorkspace(&workspace);
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "run op failed");
    return false;
  }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    FreeWorkspace(&workspace);
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    DestroyDeviceTensor(&out);
    ReportCase(caseName, false, "stream sync failed");
    return false;
  }

  std::vector<double> actual(selfHost.size(), 0.0);
  ret = CopyDeviceToHost(out, &actual);
  std::vector<double> expected(selfHost.size(), 0.0);
  for (size_t i = 0; i < selfHost.size(); ++i) {
    expected[i] = std::pow(selfHost[i], expHost[i]);
  }

  std::string detail;
  bool pass = (ret == ACL_SUCCESS) && CheckVectorClose(actual, expected, atol, rtol, &detail);
  ReportCase(caseName, pass, detail.empty() ? "compare failed" : detail);

  FreeWorkspace(&workspace);
  DestroyDeviceTensor(&self);
  DestroyDeviceTensor(&exponent);
  DestroyDeviceTensor(&out);
  return pass;
}

bool RunInplacePowTensorTensorCase(aclrtStream stream, const std::string& caseName,
                                   const std::vector<float>& selfHost, const std::vector<float>& expHost,
                                   const std::vector<int64_t>& shape, double atol = 1e-5,
                                   double rtol = 1e-4) {
  DeviceTensor self;
  DeviceTensor exponent;
  DeviceWorkspace workspace;
  aclOpExecutor* executor = nullptr;

  int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    ReportCase(caseName, false, "create self tensor failed");
    return false;
  }
  ret = CreateAclTensor(expHost, shape, ACL_FLOAT, &exponent);
  if (ret != ACL_SUCCESS) {
    DestroyDeviceTensor(&self);
    ReportCase(caseName, false, "create exponent tensor failed");
    return false;
  }

  uint64_t workspaceSize = 0;
  ret = aclnnInplacePowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    ReportCase(caseName, false, "GetWorkspaceSize failed");
    return false;
  }

  ret = AllocWorkspace(workspaceSize, &workspace);
  if (ret != ACL_SUCCESS) {
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    ReportCase(caseName, false, "allocate workspace failed");
    return false;
  }

  ret = aclnnInplacePowTensorTensor(workspace.addr, workspace.size, executor, stream);
  if (ret != ACL_SUCCESS) {
    FreeWorkspace(&workspace);
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    ReportCase(caseName, false, "run op failed");
    return false;
  }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    FreeWorkspace(&workspace);
    DestroyDeviceTensor(&self);
    DestroyDeviceTensor(&exponent);
    ReportCase(caseName, false, "stream sync failed");
    return false;
  }

  std::vector<float> actual(selfHost.size(), 0.0f);
  ret = CopyDeviceToHost(self, &actual);
  std::vector<double> expected(selfHost.size(), 0.0);
  for (size_t i = 0; i < selfHost.size(); ++i) {
    expected[i] = std::pow(static_cast<double>(selfHost[i]), static_cast<double>(expHost[i]));
  }

  std::string detail;
  bool pass = (ret == ACL_SUCCESS) && CheckVectorClose(actual, expected, atol, rtol, &detail);
  ReportCase(caseName, pass, detail.empty() ? "compare failed" : detail);

  FreeWorkspace(&workspace);
  DestroyDeviceTensor(&self);
  DestroyDeviceTensor(&exponent);
  return pass;
}

bool RunPowTensorScalarErrorCases() {
  const std::vector<int64_t> shape = {2, 2};
  bool allPass = true;

  DeviceTensor selfFloat;
  DeviceTensor outFloat;
  DeviceTensor outBadShape;
  DeviceTensor selfBool;
  DeviceTensor outBool;
  DeviceTensor selfInt32;
  DeviceTensor outInt32;

  aclScalar* exponentFloat = nullptr;
  aclScalar* exponentBool = nullptr;
  aclScalar* exponentNegI64 = nullptr;
  aclScalar* exponentHugeI64 = nullptr;
  aclScalar* exponentHugeDouble = nullptr;

  const std::vector<float> floatData = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<int32_t> intData = {1, 2, 3, 4};
  const std::vector<uint8_t> boolData = {0, 1, 1, 0};
  const std::vector<float> outBadData = {0.0f, 0.0f, 0.0f, 0.0f};

  int ret = CreateAclTensor(floatData, shape, ACL_FLOAT, &selfFloat);
  CHECK_RET(ret == ACL_SUCCESS, ReportCase("PowTensorScalar_errors_setup", false, "create selfFloat failed"); return false);
  ret = CreateAclTensor(floatData, shape, ACL_FLOAT, &outFloat);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyDeviceTensor(&selfFloat);
            ReportCase("PowTensorScalar_errors_setup", false, "create outFloat failed");
            return false);
  ret = CreateAclTensor(outBadData, {4}, ACL_FLOAT, &outBadShape);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyDeviceTensor(&selfFloat);
            DestroyDeviceTensor(&outFloat);
            ReportCase("PowTensorScalar_errors_setup", false, "create outBadShape failed");
            return false);
  ret = CreateAclTensor(boolData, shape, ACL_BOOL, &selfBool);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyDeviceTensor(&selfFloat);
            DestroyDeviceTensor(&outFloat);
            DestroyDeviceTensor(&outBadShape);
            ReportCase("PowTensorScalar_errors_setup", false, "create selfBool failed");
            return false);
  ret = CreateAclTensor(boolData, shape, ACL_BOOL, &outBool);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyDeviceTensor(&selfFloat);
            DestroyDeviceTensor(&outFloat);
            DestroyDeviceTensor(&outBadShape);
            DestroyDeviceTensor(&selfBool);
            ReportCase("PowTensorScalar_errors_setup", false, "create outBool failed");
            return false);
  ret = CreateAclTensor(intData, shape, ACL_INT32, &selfInt32);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyDeviceTensor(&selfFloat);
            DestroyDeviceTensor(&outFloat);
            DestroyDeviceTensor(&outBadShape);
            DestroyDeviceTensor(&selfBool);
            DestroyDeviceTensor(&outBool);
            ReportCase("PowTensorScalar_errors_setup", false, "create selfInt32 failed");
            return false);
  ret = CreateAclTensor(intData, shape, ACL_INT32, &outInt32);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyDeviceTensor(&selfFloat);
            DestroyDeviceTensor(&outFloat);
            DestroyDeviceTensor(&outBadShape);
            DestroyDeviceTensor(&selfBool);
            DestroyDeviceTensor(&outBool);
            DestroyDeviceTensor(&selfInt32);
            ReportCase("PowTensorScalar_errors_setup", false, "create outInt32 failed");
            return false);

  float exp2 = 2.0f;
  bool expBool = true;
  int64_t expNeg = -1;
  int64_t expHuge = std::numeric_limits<int64_t>::max();
  double expHugeDouble = std::numeric_limits<double>::max();
  exponentFloat = aclCreateScalar(&exp2, ACL_FLOAT);
  exponentBool = aclCreateScalar(&expBool, ACL_BOOL);
  exponentNegI64 = aclCreateScalar(&expNeg, ACL_INT64);
  exponentHugeI64 = aclCreateScalar(&expHuge, ACL_INT64);
  exponentHugeDouble = aclCreateScalar(&expHugeDouble, ACL_DOUBLE);
  CHECK_RET(exponentFloat != nullptr && exponentBool != nullptr && exponentNegI64 != nullptr &&
                exponentHugeI64 != nullptr && exponentHugeDouble != nullptr,
            DestroyDeviceTensor(&selfFloat);
            DestroyDeviceTensor(&outFloat);
            DestroyDeviceTensor(&outBadShape);
            DestroyDeviceTensor(&selfBool);
            DestroyDeviceTensor(&outBool);
            DestroyDeviceTensor(&selfInt32);
            DestroyDeviceTensor(&outInt32);
            ReportCase("PowTensorScalar_errors_setup", false, "create scalar failed");
            return false);

  auto checkFail = [&](const std::string& caseName, const aclTensor* self, const aclScalar* exponent,
                       const aclTensor* out) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const bool pass = (aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor) !=
                       ACL_SUCCESS);
    ReportCase(caseName, pass, "expected failure");
    allPass = allPass && pass;
  };

  checkFail("PowTensorScalar_err_null_self", nullptr, exponentFloat, outFloat.tensor);
  checkFail("PowTensorScalar_err_null_exponent", selfFloat.tensor, nullptr, outFloat.tensor);
  checkFail("PowTensorScalar_err_null_out", selfFloat.tensor, exponentFloat, nullptr);
  checkFail("PowTensorScalar_err_shape_mismatch", selfFloat.tensor, exponentFloat, outBadShape.tensor);
  checkFail("PowTensorScalar_err_bool_bool", selfBool.tensor, exponentBool, outBool.tensor);
  checkFail("PowTensorScalar_err_integral_negative_exp", selfInt32.tensor, exponentNegI64, outInt32.tensor);
  checkFail("PowTensorScalar_err_integral_overflow", selfInt32.tensor, exponentHugeI64, outInt32.tensor);
  checkFail("PowTensorScalar_err_float_overflow", selfFloat.tensor, exponentHugeDouble, outFloat.tensor);

  aclDestroyScalar(exponentFloat);
  aclDestroyScalar(exponentBool);
  aclDestroyScalar(exponentNegI64);
  aclDestroyScalar(exponentHugeI64);
  aclDestroyScalar(exponentHugeDouble);
  DestroyDeviceTensor(&selfFloat);
  DestroyDeviceTensor(&outFloat);
  DestroyDeviceTensor(&outBadShape);
  DestroyDeviceTensor(&selfBool);
  DestroyDeviceTensor(&outBool);
  DestroyDeviceTensor(&selfInt32);
  DestroyDeviceTensor(&outInt32);
  return allPass;
}

bool RunPowScalarTensorErrorCases() {
  const std::vector<int64_t> shape = {2, 2};
  bool allPass = true;

  DeviceTensor exponentFloat;
  DeviceTensor outFloat;
  DeviceTensor outBadShape;
  DeviceTensor exponentBool;
  DeviceTensor outBool;

  aclScalar* scalarFloat = nullptr;
  aclScalar* scalarBool = nullptr;

  const std::vector<float> floatData = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<uint8_t> boolData = {0, 1, 1, 0};

  int ret = CreateAclTensor(floatData, shape, ACL_FLOAT, &exponentFloat);
  CHECK_RET(ret == ACL_SUCCESS,
            ReportCase("PowScalarTensor_errors_setup", false, "create exponentFloat failed");
            return false);
  ret = CreateAclTensor(floatData, shape, ACL_FLOAT, &outFloat);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyDeviceTensor(&exponentFloat);
            ReportCase("PowScalarTensor_errors_setup", false, "create outFloat failed");
            return false);
  ret = CreateAclTensor(floatData, {4}, ACL_FLOAT, &outBadShape);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyDeviceTensor(&exponentFloat);
            DestroyDeviceTensor(&outFloat);
            ReportCase("PowScalarTensor_errors_setup", false, "create outBadShape failed");
            return false);
  ret = CreateAclTensor(boolData, shape, ACL_BOOL, &exponentBool);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyDeviceTensor(&exponentFloat);
            DestroyDeviceTensor(&outFloat);
            DestroyDeviceTensor(&outBadShape);
            ReportCase("PowScalarTensor_errors_setup", false, "create exponentBool failed");
            return false);
  ret = CreateAclTensor(boolData, shape, ACL_BOOL, &outBool);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyDeviceTensor(&exponentFloat);
            DestroyDeviceTensor(&outFloat);
            DestroyDeviceTensor(&outBadShape);
            DestroyDeviceTensor(&exponentBool);
            ReportCase("PowScalarTensor_errors_setup", false, "create outBool failed");
            return false);

  float scalar2 = 2.0f;
  bool scalarTrue = true;
  scalarFloat = aclCreateScalar(&scalar2, ACL_FLOAT);
  scalarBool = aclCreateScalar(&scalarTrue, ACL_BOOL);
  CHECK_RET(scalarFloat != nullptr && scalarBool != nullptr,
            DestroyDeviceTensor(&exponentFloat);
            DestroyDeviceTensor(&outFloat);
            DestroyDeviceTensor(&outBadShape);
            DestroyDeviceTensor(&exponentBool);
            DestroyDeviceTensor(&outBool);
            ReportCase("PowScalarTensor_errors_setup", false, "create scalar failed");
            return false);

  auto checkFail = [&](const std::string& caseName, const aclScalar* self, const aclTensor* exponent,
                       const aclTensor* out) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const bool pass = (aclnnPowScalarTensorGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor) !=
                       ACL_SUCCESS);
    ReportCase(caseName, pass, "expected failure");
    allPass = allPass && pass;
  };

  checkFail("PowScalarTensor_err_null_self", nullptr, exponentFloat.tensor, outFloat.tensor);
  checkFail("PowScalarTensor_err_null_exponent", scalarFloat, nullptr, outFloat.tensor);
  checkFail("PowScalarTensor_err_null_out", scalarFloat, exponentFloat.tensor, nullptr);
  checkFail("PowScalarTensor_err_shape_mismatch", scalarFloat, exponentFloat.tensor, outBadShape.tensor);
  checkFail("PowScalarTensor_err_bool_bool", scalarBool, exponentBool.tensor, outBool.tensor);

  aclDestroyScalar(scalarFloat);
  aclDestroyScalar(scalarBool);
  DestroyDeviceTensor(&exponentFloat);
  DestroyDeviceTensor(&outFloat);
  DestroyDeviceTensor(&outBadShape);
  DestroyDeviceTensor(&exponentBool);
  DestroyDeviceTensor(&outBool);
  return allPass;
}

bool RunPowTensorTensorErrorCases() {
  const std::vector<int64_t> shape = {2, 2};
  bool allPass = true;

  DeviceTensor selfFloat;
  DeviceTensor expFloat;
  DeviceTensor outFloat;
  DeviceTensor outBadShape;
  DeviceTensor selfBool;
  DeviceTensor expBool;
  DeviceTensor outBool;
  DeviceTensor expBroadcastFail;
  DeviceTensor outBroadcastFail;

  const std::vector<float> floatData = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<uint8_t> boolData = {0, 1, 1, 0};
  const std::vector<float> expBroadcastData = {1.0f, 2.0f, 3.0f};
  const std::vector<float> outBroadcastData = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  int ret = CreateAclTensor(floatData, shape, ACL_FLOAT, &selfFloat);
  CHECK_RET(ret == ACL_SUCCESS,
            ReportCase("PowTensorTensor_errors_setup", false, "create selfFloat failed");
            return false);
  ret = CreateAclTensor(floatData, shape, ACL_FLOAT, &expFloat);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyDeviceTensor(&selfFloat);
            ReportCase("PowTensorTensor_errors_setup", false, "create expFloat failed");
            return false);
  ret = CreateAclTensor(floatData, shape, ACL_FLOAT, &outFloat);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyDeviceTensor(&selfFloat);
            DestroyDeviceTensor(&expFloat);
            ReportCase("PowTensorTensor_errors_setup", false, "create outFloat failed");
            return false);
  ret = CreateAclTensor(floatData, {4}, ACL_FLOAT, &outBadShape);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyDeviceTensor(&selfFloat);
            DestroyDeviceTensor(&expFloat);
            DestroyDeviceTensor(&outFloat);
            ReportCase("PowTensorTensor_errors_setup", false, "create outBadShape failed");
            return false);
  ret = CreateAclTensor(boolData, shape, ACL_BOOL, &selfBool);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyDeviceTensor(&selfFloat);
            DestroyDeviceTensor(&expFloat);
            DestroyDeviceTensor(&outFloat);
            DestroyDeviceTensor(&outBadShape);
            ReportCase("PowTensorTensor_errors_setup", false, "create selfBool failed");
            return false);
  ret = CreateAclTensor(boolData, shape, ACL_BOOL, &expBool);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyDeviceTensor(&selfFloat);
            DestroyDeviceTensor(&expFloat);
            DestroyDeviceTensor(&outFloat);
            DestroyDeviceTensor(&outBadShape);
            DestroyDeviceTensor(&selfBool);
            ReportCase("PowTensorTensor_errors_setup", false, "create expBool failed");
            return false);
  ret = CreateAclTensor(boolData, shape, ACL_BOOL, &outBool);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyDeviceTensor(&selfFloat);
            DestroyDeviceTensor(&expFloat);
            DestroyDeviceTensor(&outFloat);
            DestroyDeviceTensor(&outBadShape);
            DestroyDeviceTensor(&selfBool);
            DestroyDeviceTensor(&expBool);
            ReportCase("PowTensorTensor_errors_setup", false, "create outBool failed");
            return false);
  ret = CreateAclTensor(expBroadcastData, {3, 1}, ACL_FLOAT, &expBroadcastFail);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyDeviceTensor(&selfFloat);
            DestroyDeviceTensor(&expFloat);
            DestroyDeviceTensor(&outFloat);
            DestroyDeviceTensor(&outBadShape);
            DestroyDeviceTensor(&selfBool);
            DestroyDeviceTensor(&expBool);
            DestroyDeviceTensor(&outBool);
            ReportCase("PowTensorTensor_errors_setup", false, "create expBroadcastFail failed");
            return false);
  ret = CreateAclTensor(outBroadcastData, {3, 2}, ACL_FLOAT, &outBroadcastFail);
  CHECK_RET(ret == ACL_SUCCESS,
            DestroyDeviceTensor(&selfFloat);
            DestroyDeviceTensor(&expFloat);
            DestroyDeviceTensor(&outFloat);
            DestroyDeviceTensor(&outBadShape);
            DestroyDeviceTensor(&selfBool);
            DestroyDeviceTensor(&expBool);
            DestroyDeviceTensor(&outBool);
            DestroyDeviceTensor(&expBroadcastFail);
            ReportCase("PowTensorTensor_errors_setup", false, "create outBroadcastFail failed");
            return false);

  auto checkFail = [&](const std::string& caseName, const aclTensor* self, const aclTensor* exponent,
                       aclTensor* out) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const bool pass = (aclnnPowTensorTensorGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor) !=
                       ACL_SUCCESS);
    ReportCase(caseName, pass, "expected failure");
    allPass = allPass && pass;
  };

  checkFail("PowTensorTensor_err_null_self", nullptr, expFloat.tensor, outFloat.tensor);
  checkFail("PowTensorTensor_err_null_exponent", selfFloat.tensor, nullptr, outFloat.tensor);
  checkFail("PowTensorTensor_err_null_out", selfFloat.tensor, expFloat.tensor, nullptr);
  checkFail("PowTensorTensor_err_bool_bool", selfBool.tensor, expBool.tensor, outBool.tensor);
  checkFail("PowTensorTensor_err_out_shape_mismatch", selfFloat.tensor, expFloat.tensor, outBadShape.tensor);
  checkFail("PowTensorTensor_err_broadcast_fail", selfFloat.tensor, expBroadcastFail.tensor, outBroadcastFail.tensor);

  DestroyDeviceTensor(&selfFloat);
  DestroyDeviceTensor(&expFloat);
  DestroyDeviceTensor(&outFloat);
  DestroyDeviceTensor(&outBadShape);
  DestroyDeviceTensor(&selfBool);
  DestroyDeviceTensor(&expBool);
  DestroyDeviceTensor(&outBool);
  DestroyDeviceTensor(&expBroadcastFail);
  DestroyDeviceTensor(&outBroadcastFail);
  return allPass;
}

}  // namespace

int main() {
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  int ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

  int passCount = 0;
  int failCount = 0;
  auto runCase = [&](const std::function<bool()>& fn) {
    if (fn()) {
      ++passCount;
    } else {
      ++failCount;
    }
  };

  runCase([&]() {
    return RunPowTensorScalarCase(stream, "PowTensorScalar_exp0", {0.0f, -2.0f, 3.0f, 10.0f}, {2, 2}, 0.0f);
  });
  runCase([&]() {
    return RunPowTensorScalarCase(stream, "PowTensorScalar_exp2", {-2.0f, -1.5f, 0.5f, 4.0f}, {2, 2}, 2.0f);
  });
  runCase([&]() {
    return RunPowTensorScalarCase(stream, "PowTensorScalar_exp_half", {0.0f, 1.0f, 4.0f, 9.0f}, {2, 2}, 0.5f);
  });
  runCase([&]() {
    return RunPowTensorScalarCase(stream, "PowTensorScalar_exp_minus1", {1.0f, 2.0f, 4.0f, 8.0f}, {2, 2}, -1.0f);
  });
  runCase([&]() {
    return RunInplacePowTensorScalarCase(stream, "InplacePowTensorScalar_exp3", {-2.0f, -1.0f, 2.0f, 3.0f}, {2, 2},
                                          3.0f);
  });

  runCase([&]() {
    return RunPowScalarTensorCase(stream, "PowScalarTensor_base2", 2.0f, {0.0f, 1.0f, 2.0f, 3.0f}, {2, 2});
  });
  runCase([&]() {
    return RunPowScalarTensorCase(stream, "PowScalarTensor_base1_fill_one", 1.0f, {-3.0f, 0.0f, 2.0f, 5.0f}, {2, 2});
  });

  runCase([&]() {
    return RunPowTensorTensorFloatCase(stream, "PowTensorTensor_same_shape_float", {1.0f, 2.0f, 3.0f, 4.0f}, {2, 2},
                                       {3.0f, 2.0f, 1.0f, 0.0f}, {2, 2}, {2, 2});
  });
  runCase([&]() {
    return RunPowTensorTensorFloatCase(stream, "PowTensorTensor_broadcast_float", {2.0f, 3.0f}, {2, 1},
                                       {1.0f, 2.0f, 3.0f}, {1, 3}, {2, 3});
  });
  runCase([&]() {
    return RunInplacePowTensorTensorCase(stream, "InplacePowTensorTensor_float", {2.0f, 3.0f, 4.0f, 5.0f},
                                         {3.0f, 2.0f, 1.0f, 0.0f}, {2, 2});
  });

  runCase([&]() {
    return RunPowTensorTensorInt32Case(stream, "PowTensorTensor_int32", {1, 2, 3, 4}, {2, 2}, {3, 2, 1, 0}, {2, 2},
                                       {2, 2});
  });
  runCase([&]() {
    return RunPowTensorTensorInt8Case(stream, "PowTensorTensor_int8", {1, 2, 3, 4}, {3, 2, 1, 0}, {2, 2});
  });
  runCase([&]() {
    return RunPowTensorTensorUint8Case(stream, "PowTensorTensor_uint8", {1, 2, 3, 4}, {3, 2, 1, 0}, {2, 2});
  });
  runCase([&]() {
    return RunPowTensorTensorInt16Case(stream, "PowTensorTensor_int16", {1, 2, 3, 4}, {3, 2, 1, 0}, {2, 2});
  });
  runCase([&]() {
    const std::vector<uint16_t> base = {
        FloatToFloat16(1.0f), FloatToFloat16(2.0f), FloatToFloat16(4.0f), FloatToFloat16(8.0f)};
    const std::vector<uint16_t> exponent = {
        FloatToFloat16(0.0f), FloatToFloat16(1.0f), FloatToFloat16(2.0f), FloatToFloat16(3.0f)};
    return RunPowTensorTensorFloat16Case(stream, "PowTensorTensor_float16", base, exponent, {2, 2});
  });
  runCase([&]() {
    return RunPowTensorTensorDoubleCase(stream, "PowTensorTensor_double_aicpu", {1.0, 2.0, 4.0, 8.0},
                                        {3.0, 2.0, 1.0, 0.0}, {2, 2});
  });

  runCase([&]() { return RunPowTensorScalarErrorCases(); });
  runCase([&]() { return RunPowScalarTensorErrorCases(); });
  runCase([&]() { return RunPowTensorTensorErrorCases(); });

  LOG_PRINT("\n========== Pow Test Summary ==========" "\n");
  LOG_PRINT("Total: %d, PASS: %d, FAIL: %d\n", passCount + failCount, passCount, failCount);

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return (failCount == 0) ? 0 : 1;
}