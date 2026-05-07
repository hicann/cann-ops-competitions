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
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_exp2.h"
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
  return ACL_SUCCESS;
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
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return -1);
  return ACL_SUCCESS;
}

bool AlmostEqual(double expected, double actual, double atol = 1e-5, double rtol = 1e-5) {
  if (std::isnan(expected) && std::isnan(actual)) {
    return true;
  }
  if (std::isinf(expected) && std::isinf(actual)) {
    return (expected > 0) == (actual > 0);
  }
  return std::fabs(actual - expected) <= (atol + rtol * std::fabs(expected));
}

template <typename T>
struct TensorResource {
  aclTensor* tensor = nullptr;
  void* deviceAddr = nullptr;
  std::vector<int64_t> shape;

  int Create(const std::vector<T>& hostData, const std::vector<int64_t>& tensorShape, aclDataType dataType) {
    shape = tensorShape;
    return CreateAclTensor(hostData, shape, &deviceAddr, dataType, &tensor);
  }

  void Destroy() {
    if (tensor != nullptr) {
      aclDestroyTensor(tensor);
      tensor = nullptr;
    }
    if (deviceAddr != nullptr) {
      aclrtFree(deviceAddr);
      deviceAddr = nullptr;
    }
  }
};

void PrintCaseResult(const std::string& name, bool passed) {
  LOG_PRINT("[%s] %s\n", passed ? "PASS" : "FAIL", name.c_str());
}

bool CheckAclRet(const std::string& testName, const char* step, int ret) {
  if (ret == ACL_SUCCESS) {
    return true;
  }
  LOG_PRINT("  %s failed in %s. ERROR: %d\n", testName.c_str(), step, ret);
  return false;
}

template <typename T>
bool CopyDeviceToHost(const std::string& testName, const TensorResource<T>& resource, std::vector<T>* result) {
  auto elementCount = GetShapeSize(resource.shape);
  result->assign(elementCount, static_cast<T>(0));
  auto ret = aclrtMemcpy(result->data(), elementCount * sizeof(T), resource.deviceAddr, elementCount * sizeof(T),
                         ACL_MEMCPY_DEVICE_TO_HOST);
  return CheckAclRet(testName, "aclrtMemcpy device to host", ret);
}

bool FloatVectorAlmostEqual(const std::vector<float>& actual, const std::vector<float>& expected,
                            const std::string& testName, double atol = 1e-5, double rtol = 1e-5) {
  if (actual.size() != expected.size()) {
    LOG_PRINT("  %s size mismatch. actual=%zu expected=%zu\n", testName.c_str(), actual.size(), expected.size());
    return false;
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!AlmostEqual(expected[i], actual[i], atol, rtol)) {
      LOG_PRINT("  %s mismatch at %zu. actual=%f expected=%f\n", testName.c_str(), i, actual[i], expected[i]);
      return false;
    }
  }
  return true;
}

bool DoubleVectorAlmostEqual(const std::vector<double>& actual, const std::vector<double>& expected,
                             const std::string& testName, double atol = 1e-8, double rtol = 1e-8) {
  if (actual.size() != expected.size()) {
    LOG_PRINT("  %s size mismatch. actual=%zu expected=%zu\n", testName.c_str(), actual.size(), expected.size());
    return false;
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!AlmostEqual(expected[i], actual[i], atol, rtol)) {
      LOG_PRINT("  %s mismatch at %zu. actual=%lf expected=%lf\n", testName.c_str(), i, actual[i], expected[i]);
      return false;
    }
  }
  return true;
}

template <typename T>
bool ExactVectorEqual(const std::vector<T>& actual, const std::vector<T>& expected, const std::string& testName) {
  if (actual.size() != expected.size()) {
    LOG_PRINT("  %s size mismatch. actual=%zu expected=%zu\n", testName.c_str(), actual.size(), expected.size());
    return false;
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      LOG_PRINT("  %s mismatch at %zu\n", testName.c_str(), i);
      return false;
    }
  }
  return true;
}

std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  return strides;
}

std::vector<float> PowTensorScalarExpected(const std::vector<float>& selfData, float exponent) {
  std::vector<float> expected(selfData.size(), 0.0f);
  for (size_t i = 0; i < selfData.size(); ++i) {
    expected[i] = static_cast<float>(std::pow(selfData[i], exponent));
  }
  return expected;
}

std::vector<double> PowTensorScalarExpected(const std::vector<double>& selfData, double exponent) {
  std::vector<double> expected(selfData.size(), 0.0);
  for (size_t i = 0; i < selfData.size(); ++i) {
    expected[i] = std::pow(selfData[i], exponent);
  }
  return expected;
}

std::vector<float> PowTensorTensorExpected(const std::vector<float>& selfData, const std::vector<float>& otherData) {
  std::vector<float> expected(selfData.size(), 0.0f);
  for (size_t i = 0; i < selfData.size(); ++i) {
    expected[i] = static_cast<float>(std::pow(selfData[i], otherData[i]));
  }
  return expected;
}

std::vector<double> PowTensorTensorExpected(const std::vector<double>& selfData, const std::vector<double>& otherData) {
  std::vector<double> expected(selfData.size(), 0.0);
  for (size_t i = 0; i < selfData.size(); ++i) {
    expected[i] = std::pow(selfData[i], otherData[i]);
  }
  return expected;
}

std::vector<float> BroadcastPowTensorTensorExpected(const std::vector<float>& selfData, const std::vector<int64_t>& selfShape,
                                                    const std::vector<float>& otherData, const std::vector<int64_t>& otherShape,
                                                    const std::vector<int64_t>& outShape) {
  std::vector<float> expected(GetShapeSize(outShape), 0.0f);
  std::vector<int64_t> selfStrides = MakeStrides(selfShape);
  std::vector<int64_t> otherStrides = MakeStrides(otherShape);
  std::vector<int64_t> outStrides = MakeStrides(outShape);
  int64_t rankDiffSelf = static_cast<int64_t>(outShape.size()) - static_cast<int64_t>(selfShape.size());
  int64_t rankDiffOther = static_cast<int64_t>(outShape.size()) - static_cast<int64_t>(otherShape.size());

  for (int64_t outIndex = 0; outIndex < GetShapeSize(outShape); ++outIndex) {
    int64_t remain = outIndex;
    int64_t selfIndex = 0;
    int64_t otherIndex = 0;
    for (size_t dim = 0; dim < outShape.size(); ++dim) {
      int64_t coord = remain / outStrides[dim];
      remain %= outStrides[dim];

      int64_t selfDim = static_cast<int64_t>(dim) - rankDiffSelf;
      if (selfDim >= 0) {
        int64_t selfCoord = (selfShape[selfDim] == 1) ? 0 : coord;
        selfIndex += selfCoord * selfStrides[selfDim];
      }

      int64_t otherDim = static_cast<int64_t>(dim) - rankDiffOther;
      if (otherDim >= 0) {
        int64_t otherCoord = (otherShape[otherDim] == 1) ? 0 : coord;
        otherIndex += otherCoord * otherStrides[otherDim];
      }
    }
    expected[outIndex] = static_cast<float>(std::pow(selfData[selfIndex], otherData[otherIndex]));
  }
  return expected;
}

bool RunPowTensorScalar(aclTensor* self, aclScalar* exponent, aclTensor* out, aclrtStream stream, const std::string& testName) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
  CHECK_RET(CheckAclRet(testName, "aclnnPowTensorScalarGetWorkspaceSize", ret), return false);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(CheckAclRet(testName, "aclrtMalloc workspace", ret), return false);
  }

  ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
  bool ok = CheckAclRet(testName, "aclnnPowTensorScalar", ret);
  if (ok) {
    ret = aclrtSynchronizeStream(stream);
    ok = CheckAclRet(testName, "aclrtSynchronizeStream", ret);
  }

  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ok;
}

bool RunInplacePowTensorScalar(aclTensor* self, aclScalar* exponent, aclrtStream stream, const std::string& testName) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnInplacePowTensorScalarGetWorkspaceSize(self, exponent, &workspaceSize, &executor);
  CHECK_RET(CheckAclRet(testName, "aclnnInplacePowTensorScalarGetWorkspaceSize", ret), return false);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(CheckAclRet(testName, "aclrtMalloc workspace", ret), return false);
  }

  ret = aclnnInplacePowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
  bool ok = CheckAclRet(testName, "aclnnInplacePowTensorScalar", ret);
  if (ok) {
    ret = aclrtSynchronizeStream(stream);
    ok = CheckAclRet(testName, "aclrtSynchronizeStream", ret);
  }

  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ok;
}

bool RunPowTensorTensor(aclTensor* self, aclTensor* exponent, aclTensor* out, aclrtStream stream, const std::string& testName) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnPowTensorTensorGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
  CHECK_RET(CheckAclRet(testName, "aclnnPowTensorTensorGetWorkspaceSize", ret), return false);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(CheckAclRet(testName, "aclrtMalloc workspace", ret), return false);
  }

  ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
  bool ok = CheckAclRet(testName, "aclnnPowTensorTensor", ret);
  if (ok) {
    ret = aclrtSynchronizeStream(stream);
    ok = CheckAclRet(testName, "aclrtSynchronizeStream", ret);
  }

  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ok;
}

bool RunAclnnExp2(aclTensor* self, aclTensor* out, aclrtStream stream, const std::string& testName) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnExp2GetWorkspaceSize(self, out, &workspaceSize, &executor);
  CHECK_RET(CheckAclRet(testName, "aclnnExp2GetWorkspaceSize", ret), return false);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(CheckAclRet(testName, "aclrtMalloc workspace", ret), return false);
  }

  ret = aclnnExp2(workspaceAddr, workspaceSize, executor, stream);
  bool ok = CheckAclRet(testName, "aclnnExp2", ret);
  if (ok) {
    ret = aclrtSynchronizeStream(stream);
    ok = CheckAclRet(testName, "aclrtSynchronizeStream", ret);
  }

  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ok;
}

bool TestPowTensorScalarBasic(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_basic";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = 4.0f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarNegZero(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_neg_zero";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHostData = {-2.0f, -1.0f, -0.0f, 0.0f, 1.0f, 2.0f, -3.0f, 4.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = 3.0f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarSpecialValues(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_special_values";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHostData = {NAN, INFINITY, -INFINITY, -1.0f, 0.0f, -0.0f, 4.0f, 16.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = 0.5f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestInplacePowTensorScalarBasic(aclrtStream stream) {
  const std::string testName = "inplace_pow_tensor_scalar_basic";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  float exponentValue = 2.0f;

  TensorResource<float> self;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       RunInplacePowTensorScalar(self.tensor, exponent, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, self, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestInplacePowTensorScalarBoundary(aclrtStream stream) {
  const std::string testName = "inplace_pow_tensor_scalar_boundary";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHostData = {0.0f, -0.0f, 1.0f, -1.0f, 2.0f, 4.0f, 16.0f, INFINITY};
  float exponentValue = -1.0f;

  TensorResource<float> self;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       RunInplacePowTensorScalar(self.tensor, exponent, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, self, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorTensorBasic(aclrtStream stream) {
  const std::string testName = "pow_tensor_tensor_basic";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> exponentHostData = {2.0f, 3.0f, 2.0f, 0.5f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> exponent;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create exponent", exponent.Create(exponentHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
            RunPowTensorTensor(self.tensor, exponent.tensor, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorTensorExpected(selfHostData, exponentHostData);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  self.Destroy();
  exponent.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorTensorBroadcastBasic(aclrtStream stream) {
  const std::string testName = "pow_tensor_tensor_broadcast_basic";
  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> exponentShape = {3};
  std::vector<int64_t> outShape = {2, 3};
  std::vector<float> selfHostData = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
  std::vector<float> exponentHostData = {1.0f, 0.5f, 2.0f};
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> exponent;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create exponent", exponent.Create(exponentHostData, exponentShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT)) &&
            RunPowTensorTensor(self.tensor, exponent.tensor, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected =
        BroadcastPowTensorTensorExpected(selfHostData, selfShape, exponentHostData, exponentShape, outShape);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  self.Destroy();
  exponent.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorTensorInvalidShape(aclrtStream stream) {
  (void)stream;
  const std::string testName = "pow_tensor_tensor_invalid_shape";
  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> exponentShape = {2, 2};
  std::vector<int64_t> outShape = {2, 3};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> exponentHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> exponent;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create exponent", exponent.Create(exponentHostData, exponentShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  self.Destroy();
  exponent.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestNullptrSelfInvalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "nullptr_self_invalid";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = 2.0f;

  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr) && CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorScalarGetWorkspaceSize(nullptr, exponent, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestNullptrOutInvalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "nullptr_out_invalid";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  float exponentValue = 2.0f;

  TensorResource<float> self;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr) && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent, nullptr, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestNullptrExponentInvalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "nullptr_exponent_invalid";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, nullptr, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestOutDtypeMismatchInvalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "out_dtype_mismatch_invalid";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int32_t> outHostData(GetShapeSize(shape), 0);
  float exponentValue = 2.0f;

  TensorResource<float> self;
  TensorResource<int32_t> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_INT32));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarZeroExp(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_zero_exp";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHostData = {0.0f, -0.0f, -2.0f, -1.0f, 1.0f, 2.0f, INFINITY, NAN};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = 0.0f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarOneExp(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_one_exp";
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHostData = {-8.0f, -1.5f, -0.0f, 0.0f, 2.5f, 9.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = 1.0f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarNegativeIntExp(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_negative_int_exp";
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHostData = {-4.0f, -2.0f, -1.0f, 1.0f, 2.0f, 4.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = -2.0f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarNegativeFractionExp(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_negative_fraction_exp";
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHostData = {0.25f, 1.0f, 4.0f, 9.0f, 16.0f, 100.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = -0.5f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarSpecialBranchValues(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_special_branch_values";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHostData = {-1.0f, 0.0f, -0.0f, 1.0f, -INFINITY, INFINITY, NAN, 64.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = 2.0f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarBaseZeroNegativeExp(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_base_zero_negative_exp";
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHostData = {0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = -1.0f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarBaseOneAnyExp(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_base_one_any_exp";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHostData = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = -37.5f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarBaseMinusOneExp(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_base_minus_one_exp";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHostData = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = 101.0f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarNegativeBaseNonintegerExp(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_negative_base_noninteger_exp";
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHostData = {-1.0f, -2.0f, -4.0f, -9.0f, -16.0f, -25.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = 0.5f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestNullptrSelfInvalidExtra(aclrtStream stream) {
  (void)stream;
  const std::string testName = "nullptr_self_invalid_extra";
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> exponentHostData = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);

  TensorResource<float> exponent;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create exponent", exponent.Create(exponentHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorTensorGetWorkspaceSize(nullptr, exponent.tensor, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  exponent.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestNullptrOutInvalidExtra(aclrtStream stream) {
  (void)stream;
  const std::string testName = "nullptr_out_invalid_extra";
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> exponentHostData = {1.0f, 0.5f, 2.0f, 1.0f, 0.5f, 2.0f};

  TensorResource<float> self;
  TensorResource<float> exponent;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create exponent", exponent.Create(exponentHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, nullptr, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  self.Destroy();
  exponent.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestNullptrExponentInvalidExtra(aclrtStream stream) {
  (void)stream;
  const std::string testName = "nullptr_exponent_invalid_extra";
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, nullptr, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestOutDtypeMismatchInvalidExtra(aclrtStream stream) {
  (void)stream;
  const std::string testName = "out_dtype_mismatch_invalid_extra";
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHostData = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
  std::vector<float> exponentHostData = {1.0f, 0.5f, 2.0f, 1.0f, 0.5f, 2.0f};
  std::vector<int32_t> outHostData(GetShapeSize(shape), 0);

  TensorResource<float> self;
  TensorResource<float> exponent;
  TensorResource<int32_t> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create exponent", exponent.Create(exponentHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_INT32));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  self.Destroy();
  exponent.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestUnsupportedDtypeInvalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "unsupported_dtype_invalid";
  std::vector<int64_t> shape = {2};
  std::vector<int32_t> data = {1, 2};

  TensorResource<int32_t> self;
  TensorResource<int32_t> out;
  aclScalar* exponent = aclCreateScalar(&data[0], static_cast<aclDataType>(999));
  int ret1 = self.Create(data, shape, static_cast<aclDataType>(999));
  int ret2 = out.Create(data, shape, static_cast<aclDataType>(999));
  bool ok = true;

  if (ret1 == ACL_SUCCESS && ret2 == ACL_SUCCESS && exponent != nullptr) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorTensorInvalidOutshape(aclrtStream stream) {
  (void)stream;
  const std::string testName = "pow_tensor_tensor_invalid_outshape";
  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> exponentShape = {3};
  std::vector<int64_t> outShape = {2, 2};
  std::vector<float> selfHostData = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
  std::vector<float> exponentHostData = {1.0f, 0.5f, 2.0f};
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> exponent;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create exponent", exponent.Create(exponentHostData, exponentShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  self.Destroy();
  exponent.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorTensorInvalidRankMismatch(aclrtStream stream) {
  (void)stream;
  const std::string testName = "pow_tensor_tensor_invalid_rank_mismatch";
  std::vector<int64_t> selfShape = {2, 1, 3};
  std::vector<int64_t> exponentShape = {4, 2};
  std::vector<int64_t> outShape = {2, 4, 3};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> exponentHostData = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> exponent;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create exponent", exponent.Create(exponentHostData, exponentShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  self.Destroy();
  exponent.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorTensorInvalidDtypeCombo(aclrtStream stream) {
  (void)stream;
  const std::string testName = "pow_tensor_tensor_invalid_dtype_combo";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int32_t> exponentHostData = {1, 2, 3, 4};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);

  TensorResource<float> self;
  TensorResource<int32_t> exponent;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create exponent", exponent.Create(exponentHostData, shape, ACL_INT32)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  self.Destroy();
  exponent.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestBaseZeroNegativeExp(aclrtStream stream) {
  const std::string testName = "base_zero_negative_exp";
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHostData = {0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = -1.0f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestBaseNegativeNonintegerExp(aclrtStream stream) {
  const std::string testName = "base_negative_noninteger_exp";
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHostData = {-1.0f, -2.0f, -4.0f, -9.0f, -16.0f, -25.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = 0.5f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestBaseOneAnyExp(aclrtStream stream) {
  const std::string testName = "base_one_any_exp";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHostData = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> exponentHostData = {-100.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 10.0f, 100.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> exponent;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create exponent", exponent.Create(exponentHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
            RunPowTensorTensor(self.tensor, exponent.tensor, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorTensorExpected(selfHostData, exponentHostData);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  self.Destroy();
  exponent.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestBaseMinusOneLargeExp(aclrtStream stream) {
  const std::string testName = "base_minus_one_large_exp";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHostData = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
  std::vector<float> exponentHostData = {101.0f, 100.0f, 999.0f, 1000.0f, -101.0f, -100.0f, -999.0f, -1000.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> exponent;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create exponent", exponent.Create(exponentHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
            RunPowTensorTensor(self.tensor, exponent.tensor, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorTensorExpected(selfHostData, exponentHostData);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  self.Destroy();
  exponent.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarBaseZeroFractionExp(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_base_zero_fraction_exp";
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHostData = {0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = 0.5f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarBaseNegativeLargeEvenExp(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_base_negative_large_even_exp";
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHostData = {-2.0f, -3.0f, -4.0f, -1.0f, -0.5f, -8.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = 20.0f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarBaseNegativeLargeOddExp(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_base_negative_large_odd_exp";
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHostData = {-2.0f, -3.0f, -4.0f, -1.0f, -0.5f, -8.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = 21.0f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarBaseMinusOneLargeExp(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_base_minus_one_large_exp";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHostData = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = 999.0f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarExponentNan(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_exponent_nan";
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHostData = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, INFINITY};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = NAN;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarExponentInf(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_exponent_inf";
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHostData = {0.5f, 1.0f, 2.0f, -1.0f, -2.0f, INFINITY};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = INFINITY;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarExponentNegInf(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_exponent_neg_inf";
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHostData = {0.5f, 1.0f, 2.0f, -1.0f, -2.0f, 0.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = -INFINITY;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarInvalidOutDtypeExtra(aclrtStream stream) {
  (void)stream;
  const std::string testName = "pow_tensor_scalar_invalid_out_dtype_extra";
  std::vector<int64_t> shape = {2};
  std::vector<double> selfHostData = {2.0, 3.0};
  std::vector<float> outHostData = {0.0f, 0.0f};
  double exponentValue = 2.0;

  TensorResource<double> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_DOUBLE);
  bool ok = (exponent != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_DOUBLE)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarInvalidSelfDtypeCombo(aclrtStream stream) {
  (void)stream;
  const std::string testName = "pow_tensor_scalar_invalid_self_dtype_combo";
  std::vector<int64_t> shape = {2};
  std::vector<int32_t> selfHostData = {2, 3};
  std::vector<float> outHostData = {0.0f, 0.0f};
  float exponentValue = 2.0f;

  TensorResource<int32_t> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_INT32)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestNullptrExecutorInvalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "nullptr_executor_invalid";
  auto ret = aclnnPowTensorScalar(nullptr, 0, nullptr, nullptr);
  bool ok = (ret != ACL_SUCCESS);
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorTensorInvalidBroadcastExtra(aclrtStream stream) {
  (void)stream;
  const std::string testName = "pow_tensor_tensor_invalid_broadcast_extra";
  std::vector<int64_t> selfShape = {2, 1, 3};
  std::vector<int64_t> exponentShape = {2, 2, 1};
  std::vector<int64_t> outShape = {2, 2, 3};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> exponentHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> exponent;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create exponent", exponent.Create(exponentHostData, exponentShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  self.Destroy();
  exponent.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarDouble(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_double";
  std::vector<int64_t> shape = {2};
  std::vector<double> selfHostData = {-2.0, NAN};
  std::vector<double> outHostData = {0.0, 0.0};
  double exponentValue = 3.0;

  TensorResource<double> self;
  TensorResource<double> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_DOUBLE);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_DOUBLE)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_DOUBLE)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<double> result;
  if (ok) {
    std::vector<double> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && DoubleVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarDoubleBoundary(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_double_boundary";
  std::vector<int64_t> shape = {4};
  std::vector<double> selfHostData = {-1.0, 0.0, 1.0, INFINITY};
  std::vector<double> outHostData = {0.0, 0.0, 0.0, 0.0};
  double exponentValue = -2.0;

  TensorResource<double> self;
  TensorResource<double> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_DOUBLE);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_DOUBLE)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_DOUBLE)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<double> result;
  if (ok) {
    std::vector<double> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && DoubleVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarInt32(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_int32";
  std::vector<int64_t> shape = {2};
  std::vector<int32_t> selfHostData = {2, 3};
  std::vector<int32_t> outHostData = {0, 0};
  int32_t exponentValue = 2;

  TensorResource<int32_t> self;
  TensorResource<int32_t> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_INT32);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_INT32)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_INT32)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<int32_t> result;
  if (ok) {
    std::vector<int32_t> expected = {4, 9};
    ok = CopyDeviceToHost(testName, out, &result) && ExactVectorEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarInt32Basic(aclrtStream stream) {
  return TestPowTensorScalarInt32(stream);
}

bool TestPowTensorTensorDoubleBasic(aclrtStream stream) {
  const std::string testName = "pow_tensor_tensor_double_basic";
  std::vector<int64_t> shape = {2};
  std::vector<double> selfHostData = {4.0, 9.0};
  std::vector<double> exponentHostData = {0.5, 2.0};
  std::vector<double> outHostData = {0.0, 0.0};

  TensorResource<double> self;
  TensorResource<double> exponent;
  TensorResource<double> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_DOUBLE)) &&
            CheckAclRet(testName, "create exponent", exponent.Create(exponentHostData, shape, ACL_DOUBLE)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_DOUBLE)) &&
            RunPowTensorTensor(self.tensor, exponent.tensor, out.tensor, stream, testName);

  std::vector<double> result;
  if (ok) {
    std::vector<double> expected = PowTensorTensorExpected(selfHostData, exponentHostData);
    ok = CopyDeviceToHost(testName, out, &result) && DoubleVectorAlmostEqual(result, expected, testName);
  }

  self.Destroy();
  exponent.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestExp2Double(aclrtStream stream) {
  const std::string testName = "exp2_double";
  std::vector<int64_t> shape = {2};
  std::vector<double> selfHostData = {2.0, -3.0};
  std::vector<double> outHostData = {0.0, 0.0};

  TensorResource<double> self;
  TensorResource<double> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_DOUBLE)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_DOUBLE)) &&
            RunAclnnExp2(self.tensor, out.tensor, stream, testName);

  std::vector<double> result;
  if (ok) {
    std::vector<double> expected = {std::pow(2.0, selfHostData[0]), std::pow(2.0, selfHostData[1])};
    ok = CopyDeviceToHost(testName, out, &result) && DoubleVectorAlmostEqual(result, expected, testName);
  }

  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarBaseZeroNegativeExpExtra(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_base_zero_negative_exp_extra";
  std::vector<int64_t> shape = {4};
  std::vector<float> selfHostData = {0.0f, -0.0f, 0.0f, -0.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = -3.0f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarBaseZeroFractionExpExtra(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_base_zero_fraction_exp_extra";
  std::vector<int64_t> shape = {4};
  std::vector<float> selfHostData = {0.0f, -0.0f, 0.0f, -0.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = 0.25f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarBaseNegativeFractionExpExtra(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_base_negative_fraction_exp_extra";
  std::vector<int64_t> shape = {4};
  std::vector<float> selfHostData = {-0.25f, -1.0f, -4.0f, -16.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = -0.5f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarBaseNegativeNonintegerExpExtra(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_base_negative_noninteger_exp_extra";
  std::vector<int64_t> shape = {4};
  std::vector<float> selfHostData = {-2.0f, -3.0f, -5.0f, -7.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = 1.5f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarBaseNegativeEvenIntegerExp(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_base_negative_even_integer_exp";
  std::vector<int64_t> shape = {4};
  std::vector<float> selfHostData = {-2.0f, -3.0f, -4.0f, -5.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = 6.0f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarBaseNegativeOddIntegerExp(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_base_negative_odd_integer_exp";
  std::vector<int64_t> shape = {4};
  std::vector<float> selfHostData = {-2.0f, -3.0f, -4.0f, -5.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = 7.0f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarBaseOneLargeExp(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_base_one_large_exp";
  std::vector<int64_t> shape = {4};
  std::vector<float> selfHostData = {1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = 1024.0f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarBaseMinusOneEvenExp(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_base_minus_one_even_exp";
  std::vector<int64_t> shape = {4};
  std::vector<float> selfHostData = {-1.0f, -1.0f, -1.0f, -1.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = 1000.0f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarBaseMinusOneOddExp(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_base_minus_one_odd_exp";
  std::vector<int64_t> shape = {4};
  std::vector<float> selfHostData = {-1.0f, -1.0f, -1.0f, -1.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float exponentValue = 1001.0f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarExponentZeroDimScalarLike(aclrtStream stream) {
  const std::string testName = "pow_tensor_scalar_exponent_zero_dim_scalar_like";
  std::vector<int64_t> shape = {};
  std::vector<float> selfHostData = {2.0f};
  std::vector<float> outHostData = {0.0f};
  float exponentValue = 3.0f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunPowTensorScalar(self.tensor, exponent, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = PowTensorScalarExpected(selfHostData, exponentValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarInvalidOutDtypeExtra2(aclrtStream stream) {
  (void)stream;
  const std::string testName = "pow_tensor_scalar_invalid_out_dtype_extra2";
  std::vector<int64_t> shape = {2};
  std::vector<double> selfHostData = {4.0, 9.0};
  std::vector<int32_t> outHostData = {0, 0};
  double exponentValue = 0.5;

  TensorResource<double> self;
  TensorResource<int32_t> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_DOUBLE);
  bool ok = (exponent != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_DOUBLE)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_INT32));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarInvalidSelfDtypeExtra(aclrtStream stream) {
  (void)stream;
  const std::string testName = "pow_tensor_scalar_invalid_self_dtype_extra";
  std::vector<int64_t> shape = {2};
  std::vector<int32_t> selfHostData = {2, 3};
  std::vector<double> outHostData = {0.0, 0.0};
  double exponentValue = 2.0;

  TensorResource<int32_t> self;
  TensorResource<double> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_DOUBLE);
  bool ok = (exponent != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_INT32)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_DOUBLE));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestPowTensorScalarInvalidScalarDtypeExtra(aclrtStream stream) {
  (void)stream;
  const std::string testName = "pow_tensor_scalar_invalid_scalar_dtype_extra";
  std::vector<int64_t> shape = {2};
  std::vector<float> selfHostData = {2.0f, 3.0f};
  std::vector<float> outHostData = {0.0f, 0.0f};
  int32_t exponentValue = 2;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_INT32);
  bool ok = (exponent != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestNullptrSelfInvalidExtra2(aclrtStream stream) {
  (void)stream;
  const std::string testName = "nullptr_self_invalid_extra2";
  std::vector<int64_t> shape = {2};
  std::vector<float> outHostData = {0.0f, 0.0f};
  float exponentValue = -2.0f;

  TensorResource<float> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr) && CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorScalarGetWorkspaceSize(nullptr, exponent, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestNullptrOutInvalidExtra2(aclrtStream stream) {
  (void)stream;
  const std::string testName = "nullptr_out_invalid_extra2";
  std::vector<int64_t> shape = {2};
  std::vector<float> selfHostData = {2.0f, 3.0f};
  float exponentValue = -2.0f;

  TensorResource<float> self;
  aclScalar* exponent = aclCreateScalar(&exponentValue, ACL_FLOAT);
  bool ok = (exponent != nullptr) && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent, nullptr, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestNullptrExponentInvalidExtra2(aclrtStream stream) {
  (void)stream;
  const std::string testName = "nullptr_exponent_invalid_extra2";
  std::vector<int64_t> shape = {2};
  std::vector<float> selfHostData = {2.0f, 3.0f};
  std::vector<float> outHostData = {0.0f, 0.0f};

  TensorResource<float> self;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, nullptr, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestNullptrExecutorInvalidExtra2(aclrtStream stream) {
  (void)stream;
  const std::string testName = "nullptr_executor_invalid_extra2";
  auto ret = aclnnPowTensorScalar(nullptr, 0, nullptr, nullptr);
  bool ok = (ret != ACL_SUCCESS);
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestUnsupportedDtypeInvalidExtra(aclrtStream stream) {
  (void)stream;
  const std::string testName = "unsupported_dtype_invalid_extra";
  std::vector<int64_t> shape = {2};
  std::vector<uint16_t> selfHostData = {0, 0};
  std::vector<uint16_t> outHostData = {0, 0};
  int32_t exponentValue = 2;

  TensorResource<uint16_t> self;
  TensorResource<uint16_t> out;
  aclScalar* exponent = aclCreateScalar(&exponentValue, static_cast<aclDataType>(999));
  int ret1 = self.Create(selfHostData, shape, static_cast<aclDataType>(999));
  int ret2 = out.Create(outHostData, shape, static_cast<aclDataType>(999));
  bool ok = true;

  if (ret1 == ACL_SUCCESS && ret2 == ACL_SUCCESS && exponent != nullptr) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  int failedCount = 0;
  failedCount += TestPowTensorScalarBasic(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarNegZero(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarSpecialValues(stream) ? 0 : 1;
  failedCount += TestInplacePowTensorScalarBasic(stream) ? 0 : 1;
  failedCount += TestInplacePowTensorScalarBoundary(stream) ? 0 : 1;
  failedCount += TestPowTensorTensorBasic(stream) ? 0 : 1;
  failedCount += TestPowTensorTensorBroadcastBasic(stream) ? 0 : 1;
  failedCount += TestPowTensorTensorInvalidShape(stream) ? 0 : 1;
  failedCount += TestNullptrSelfInvalid(stream) ? 0 : 1;
  failedCount += TestNullptrOutInvalid(stream) ? 0 : 1;
  failedCount += TestNullptrExponentInvalid(stream) ? 0 : 1;
  failedCount += TestOutDtypeMismatchInvalid(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarZeroExp(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarOneExp(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarNegativeIntExp(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarNegativeFractionExp(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarBaseZeroNegativeExp(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarBaseZeroNegativeExpExtra(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarBaseZeroFractionExp(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarBaseZeroFractionExpExtra(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarBaseOneAnyExp(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarBaseOneLargeExp(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarBaseMinusOneExp(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarBaseMinusOneEvenExp(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarBaseMinusOneOddExp(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarBaseMinusOneLargeExp(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarNegativeBaseNonintegerExp(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarBaseNegativeFractionExpExtra(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarBaseNegativeNonintegerExpExtra(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarBaseNegativeEvenIntegerExp(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarBaseNegativeOddIntegerExp(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarBaseNegativeLargeEvenExp(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarBaseNegativeLargeOddExp(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarExponentZeroDimScalarLike(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarExponentNan(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarExponentInf(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarExponentNegInf(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarSpecialBranchValues(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarInvalidOutDtypeExtra(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarInvalidOutDtypeExtra2(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarInvalidSelfDtypeExtra(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarInvalidScalarDtypeExtra(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarInvalidSelfDtypeCombo(stream) ? 0 : 1;
  failedCount += TestNullptrExecutorInvalid(stream) ? 0 : 1;
  failedCount += TestNullptrSelfInvalidExtra2(stream) ? 0 : 1;
  failedCount += TestNullptrOutInvalidExtra2(stream) ? 0 : 1;
  failedCount += TestNullptrExponentInvalidExtra2(stream) ? 0 : 1;
  failedCount += TestNullptrExecutorInvalidExtra2(stream) ? 0 : 1;
  failedCount += TestNullptrSelfInvalidExtra(stream) ? 0 : 1;
  failedCount += TestNullptrOutInvalidExtra(stream) ? 0 : 1;
  failedCount += TestNullptrExponentInvalidExtra(stream) ? 0 : 1;
  failedCount += TestOutDtypeMismatchInvalidExtra(stream) ? 0 : 1;
  failedCount += TestUnsupportedDtypeInvalid(stream) ? 0 : 1;
  failedCount += TestUnsupportedDtypeInvalidExtra(stream) ? 0 : 1;
  failedCount += TestPowTensorTensorInvalidOutshape(stream) ? 0 : 1;
  failedCount += TestPowTensorTensorInvalidRankMismatch(stream) ? 0 : 1;
  failedCount += TestPowTensorTensorInvalidDtypeCombo(stream) ? 0 : 1;
  failedCount += TestPowTensorTensorInvalidBroadcastExtra(stream) ? 0 : 1;
  failedCount += TestBaseZeroNegativeExp(stream) ? 0 : 1;
  failedCount += TestBaseNegativeNonintegerExp(stream) ? 0 : 1;
  failedCount += TestBaseOneAnyExp(stream) ? 0 : 1;
  failedCount += TestBaseMinusOneLargeExp(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarDouble(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarDoubleBoundary(stream) ? 0 : 1;
  failedCount += TestPowTensorScalarInt32(stream) ? 0 : 1;
  failedCount += TestPowTensorTensorDoubleBasic(stream) ? 0 : 1;
  failedCount += TestExp2Double(stream) ? 0 : 1;

  const int totalCount = 67;
  LOG_PRINT("==================================================\n");
  LOG_PRINT("Pow test summary: total=%d, passed=%d, failed=%d\n", totalCount, totalCount - failedCount, failedCount);

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return failedCount == 0 ? 0 : 1;
}
