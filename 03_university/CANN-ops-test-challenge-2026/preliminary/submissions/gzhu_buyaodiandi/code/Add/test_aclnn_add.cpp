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
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"

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
  for (auto dim : shape) {
    shapeSize *= dim;
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

std::vector<float> AddExpected(const std::vector<float>& selfData, const std::vector<float>& otherData, float alpha) {
  std::vector<float> expected(selfData.size(), 0.0f);
  for (size_t i = 0; i < selfData.size(); ++i) {
    expected[i] = selfData[i] + alpha * otherData[i];
  }
  return expected;
}

std::vector<double> AddExpected(const std::vector<double>& selfData, const std::vector<double>& otherData,
                                double alpha) {
  std::vector<double> expected(selfData.size(), 0.0);
  for (size_t i = 0; i < selfData.size(); ++i) {
    expected[i] = selfData[i] + alpha * otherData[i];
  }
  return expected;
}

template <typename T>
std::vector<T> AddExpectedIntegral(const std::vector<T>& selfData, const std::vector<T>& otherData, T alpha) {
  std::vector<T> expected(selfData.size(), static_cast<T>(0));
  for (size_t i = 0; i < selfData.size(); ++i) {
    expected[i] = static_cast<T>(selfData[i] + alpha * otherData[i]);
  }
  return expected;
}

std::vector<float> AddScalarExpected(const std::vector<float>& selfData, float other, float alpha) {
  std::vector<float> expected(selfData.size(), 0.0f);
  for (size_t i = 0; i < selfData.size(); ++i) {
    expected[i] = selfData[i] + other * alpha;
  }
  return expected;
}

std::vector<float> BroadcastAddExpected(const std::vector<float>& selfData, const std::vector<int64_t>& selfShape,
                                        const std::vector<float>& otherData, const std::vector<int64_t>& otherShape,
                                        const std::vector<int64_t>& outShape, float alpha) {
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
    expected[outIndex] = selfData[selfIndex] + alpha * otherData[otherIndex];
  }
  return expected;
}

uint16_t FloatToHalfBits(float value) {
  union {
    float f;
    uint32_t u;
  } src;
  src.f = value;
  uint32_t sign = (src.u >> 16) & 0x8000;
  uint32_t mantissa = src.u & 0x007fffff;
  int32_t exponent = static_cast<int32_t>((src.u >> 23) & 0xff) - 127 + 15;
  if (((src.u >> 23) & 0xff) == 0xff) {
    if (mantissa != 0) {
      return static_cast<uint16_t>(sign | 0x7fff);
    }
    return static_cast<uint16_t>(sign | 0x7c00);
  }
  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa |= 0x00800000;
    uint32_t shift = static_cast<uint32_t>(14 - exponent);
    uint32_t halfMantissa = mantissa >> shift;
    if ((mantissa >> (shift - 1)) & 1U) {
      halfMantissa += 1;
    }
    return static_cast<uint16_t>(sign | halfMantissa);
  }
  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00);
  }
  uint16_t half = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
  if (mantissa & 0x00001000) {
    half = static_cast<uint16_t>(half + 1);
  }
  return half;
}

std::vector<uint16_t> FloatsToHalfBits(const std::vector<float>& values) {
  std::vector<uint16_t> halfValues(values.size(), 0);
  for (size_t i = 0; i < values.size(); ++i) {
    halfValues[i] = FloatToHalfBits(values[i]);
  }
  return halfValues;
}

std::vector<uint16_t> AddExpectedHalfBits(const std::vector<float>& selfData, const std::vector<float>& otherData,
                                          float alpha) {
  return FloatsToHalfBits(AddExpected(selfData, otherData, alpha));
}

bool RunAdd(aclTensor* self, aclTensor* other, aclScalar* alpha, aclTensor* out, aclrtStream stream,
            const std::string& testName) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  CHECK_RET(CheckAclRet(testName, "aclnnAddGetWorkspaceSize", ret), return false);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(CheckAclRet(testName, "aclrtMalloc workspace", ret), return false);
  }

  ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
  bool ok = CheckAclRet(testName, "aclnnAdd", ret);
  if (ok) {
    ret = aclrtSynchronizeStream(stream);
    ok = CheckAclRet(testName, "aclrtSynchronizeStream", ret);
  }

  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ok;
}

bool RunAclnnInplaceAdd(aclTensor* selfRef, aclTensor* other, aclScalar* alpha, aclrtStream stream,
                        const std::string& testName) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnInplaceAddGetWorkspaceSize(selfRef, other, alpha, &workspaceSize, &executor);
  CHECK_RET(CheckAclRet(testName, "aclnnInplaceAddGetWorkspaceSize", ret), return false);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(CheckAclRet(testName, "aclrtMalloc workspace", ret), return false);
  }

  ret = aclnnInplaceAdd(workspaceAddr, workspaceSize, executor, stream);
  bool ok = CheckAclRet(testName, "aclnnInplaceAdd", ret);
  if (ok) {
    ret = aclrtSynchronizeStream(stream);
    ok = CheckAclRet(testName, "aclrtSynchronizeStream", ret);
  }

  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ok;
}

bool RunAclnnAdds(aclTensor* self, aclScalar* other, aclScalar* alpha, aclTensor* out, aclrtStream stream,
                  const std::string& testName) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  CHECK_RET(CheckAclRet(testName, "aclnnAddsGetWorkspaceSize", ret), return false);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(CheckAclRet(testName, "aclrtMalloc workspace", ret), return false);
  }

  ret = aclnnAdds(workspaceAddr, workspaceSize, executor, stream);
  bool ok = CheckAclRet(testName, "aclnnAdds", ret);
  if (ok) {
    ret = aclrtSynchronizeStream(stream);
    ok = CheckAclRet(testName, "aclrtSynchronizeStream", ret);
  }

  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ok;
}

bool RunAclnnInplaceAdds(aclTensor* selfRef, aclScalar* other, aclScalar* alpha, aclrtStream stream,
                         const std::string& testName) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnInplaceAddsGetWorkspaceSize(selfRef, other, alpha, &workspaceSize, &executor);
  CHECK_RET(CheckAclRet(testName, "aclnnInplaceAddsGetWorkspaceSize", ret), return false);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(CheckAclRet(testName, "aclrtMalloc workspace", ret), return false);
  }

  ret = aclnnInplaceAdds(workspaceAddr, workspaceSize, executor, stream);
  bool ok = CheckAclRet(testName, "aclnnInplaceAdds", ret);
  if (ok) {
    ret = aclrtSynchronizeStream(stream);
    ok = CheckAclRet(testName, "aclrtSynchronizeStream", ret);
  }

  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ok;
}

bool TestAddBasic(aclrtStream stream) {
  const std::string testName = "add_basic";
  std::vector<int64_t> shape = {4, 2};
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float alphaValue = 1.2f;

  TensorResource<float> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         FloatVectorAlmostEqual(result, AddExpected(selfHostData, otherHostData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddNegZero(aclrtStream stream) {
  const std::string testName = "add_neg_zero";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHostData = {-0.0f, 0.0f, -1.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f};
  std::vector<float> otherHostData = {0.0f, -0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float alphaValue = 1.0f;

  TensorResource<float> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         FloatVectorAlmostEqual(result, AddExpected(selfHostData, otherHostData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddAlphaZero(aclrtStream stream) {
  const std::string testName = "add_alpha_zero";
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHostData = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f, 5.0f};
  std::vector<float> otherHostData = {100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float alphaValue = 0.0f;

  TensorResource<float> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         FloatVectorAlmostEqual(result, AddExpected(selfHostData, otherHostData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddAlphaOne(aclrtStream stream) {
  const std::string testName = "add_alpha_one";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> otherHostData = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float alphaValue = 1.0f;

  TensorResource<float> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         FloatVectorAlmostEqual(result, AddExpected(selfHostData, otherHostData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddAlphaNegative(aclrtStream stream) {
  const std::string testName = "add_alpha_negative";
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHostData = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};
  std::vector<float> otherHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float alphaValue = -2.0f;

  TensorResource<float> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         FloatVectorAlmostEqual(result, AddExpected(selfHostData, otherHostData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddAlphaFraction(aclrtStream stream) {
  const std::string testName = "add_alpha_fraction";
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHostData = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
  std::vector<float> otherHostData = {2.0f, 8.0f, 18.0f, 32.0f, 50.0f, 72.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float alphaValue = 0.25f;

  TensorResource<float> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         FloatVectorAlmostEqual(result, AddExpected(selfHostData, otherHostData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddBroadcastBasic(aclrtStream stream) {
  const std::string testName = "add_broadcast_basic";
  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> otherShape = {3};
  std::vector<int64_t> outShape = {2, 3};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f};
  std::vector<float> otherHostData = {0.5f, 1.5f, 2.5f};
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);
  float alphaValue = 2.0f;

  TensorResource<float> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create other", other.Create(otherHostData, otherShape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT)) &&
       RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         FloatVectorAlmostEqual(result,
                                BroadcastAddExpected(selfHostData, selfShape, otherHostData, otherShape, outShape,
                                                     alphaValue),
                                testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddBroadcastHighdim(aclrtStream stream) {
  const std::string testName = "add_broadcast_highdim";
  std::vector<int64_t> selfShape = {2, 1, 3};
  std::vector<int64_t> otherShape = {1, 4, 1};
  std::vector<int64_t> outShape = {2, 4, 3};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f};
  std::vector<float> otherHostData = {0.5f, 1.5f, -2.0f, 4.0f};
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);
  float alphaValue = -0.5f;

  TensorResource<float> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create other", other.Create(otherHostData, otherShape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT)) &&
       RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         FloatVectorAlmostEqual(result,
                                BroadcastAddExpected(selfHostData, selfShape, otherHostData, otherShape, outShape,
                                                     alphaValue),
                                testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddInvalidBroadcast(aclrtStream stream) {
  (void)stream;
  const std::string testName = "add_invalid_broadcast";
  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> otherShape = {2, 2};
  std::vector<int64_t> outShape = {2, 3};
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> otherHostData = {10, 20, 30, 40};
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);
  float alphaValue = 1.0f;

  TensorResource<float> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, otherShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestNullptrSelfInvalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "nullptr_self_invalid";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> otherHostData = {1, 2, 3, 4};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float alphaValue = 1.0f;

  TensorResource<float> other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(nullptr, other.tensor, alpha, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestNullptrOtherInvalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "nullptr_other_invalid";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {1, 2, 3, 4};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float alphaValue = 1.0f;

  TensorResource<float> self, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self.tensor, nullptr, alpha, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestNullptrOutInvalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "nullptr_out_invalid";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {1, 2, 3, 4};
  std::vector<float> otherHostData = {5, 6, 7, 8};
  float alphaValue = 1.0f;

  TensorResource<float> self, other;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, nullptr, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestNullptrAlphaInvalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "nullptr_alpha_invalid";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {1, 2, 3, 4};
  std::vector<float> otherHostData = {5, 6, 7, 8};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);

  TensorResource<float> self, other, out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, nullptr, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestOutDtypeMismatchInvalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "out_dtype_mismatch_invalid";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {1, 2, 3, 4};
  std::vector<float> otherHostData = {5, 6, 7, 8};
  std::vector<int32_t> outHostData(GetShapeSize(shape), 0);
  float alphaValue = 1.0f;

  TensorResource<float> self;
  TensorResource<float> other;
  TensorResource<int32_t> out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_INT32));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddSpecialValues(aclrtStream stream) {
  const std::string testName = "add_special_values";
  std::vector<int64_t> shape = {2, 4};
  float nanValue = std::numeric_limits<float>::quiet_NaN();
  float infValue = std::numeric_limits<float>::infinity();
  std::vector<float> selfHostData = {nanValue, infValue, -infValue, -0.0f, 0.0f, 1.0f, -1.0f, 100.0f};
  std::vector<float> otherHostData = {1.0f, -1.0f, 2.0f, 0.0f, -0.0f, infValue, nanValue, -50.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float alphaValue = 0.5f;

  TensorResource<float> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         FloatVectorAlmostEqual(result, AddExpected(selfHostData, otherHostData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddAlphaLargePositive(aclrtStream stream) {
  const std::string testName = "add_alpha_large_positive";
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHostData = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
  std::vector<float> otherHostData = {0.25f, 0.5f, 0.75f, 1.0f, -1.25f, -1.5f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float alphaValue = 1024.0f;

  TensorResource<float> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         FloatVectorAlmostEqual(result, AddExpected(selfHostData, otherHostData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddAlphaLargeNegative(aclrtStream stream) {
  const std::string testName = "add_alpha_large_negative";
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHostData = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
  std::vector<float> otherHostData = {0.25f, 0.5f, 0.75f, 1.0f, -1.25f, -1.5f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float alphaValue = -1024.0f;

  TensorResource<float> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         FloatVectorAlmostEqual(result, AddExpected(selfHostData, otherHostData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddAlphaNan(aclrtStream stream) {
  const std::string testName = "add_alpha_nan";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> otherHostData = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float alphaValue = std::numeric_limits<float>::quiet_NaN();

  TensorResource<float> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         FloatVectorAlmostEqual(result, AddExpected(selfHostData, otherHostData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddAlphaInf(aclrtStream stream) {
  const std::string testName = "add_alpha_inf";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {1.0f, 2.0f, -3.0f, -4.0f};
  std::vector<float> otherHostData = {5.0f, -6.0f, 7.0f, -8.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float alphaValue = std::numeric_limits<float>::infinity();

  TensorResource<float> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         FloatVectorAlmostEqual(result, AddExpected(selfHostData, otherHostData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddAlphaNegInf(aclrtStream stream) {
  const std::string testName = "add_alpha_neg_inf";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {1.0f, 2.0f, -3.0f, -4.0f};
  std::vector<float> otherHostData = {5.0f, -6.0f, 7.0f, -8.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float alphaValue = -std::numeric_limits<float>::infinity();

  TensorResource<float> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         FloatVectorAlmostEqual(result, AddExpected(selfHostData, otherHostData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddInt32Basic(aclrtStream stream) {
  const std::string testName = "add_int32_basic";
  std::vector<int64_t> shape = {2, 2};
  std::vector<int32_t> selfHostData = {1, 2, 3, 4};
  std::vector<int32_t> otherHostData = {5, 6, 7, 8};
  std::vector<int32_t> outHostData(4, 0);
  int32_t alphaValue = 2;

  TensorResource<int32_t> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_INT32);
  bool ok = (alpha != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_INT32)) &&
       CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_INT32)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_INT32)) &&
       RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<int32_t> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         ExactVectorEqual(result, AddExpectedIntegral(selfHostData, otherHostData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddInt16Basic(aclrtStream stream) {
  const std::string testName = "add_int16_basic";
  std::vector<int64_t> shape = {2, 2};
  std::vector<int16_t> selfHostData = {1, 2, 3, 4};
  std::vector<int16_t> otherHostData = {5, 6, 7, 8};
  std::vector<int16_t> outHostData(4, 0);
  int16_t alphaValue = 2;

  TensorResource<int16_t> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_INT16);
  bool ok = (alpha != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_INT16)) &&
       CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_INT16)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_INT16)) &&
       RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<int16_t> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         ExactVectorEqual(result, AddExpectedIntegral(selfHostData, otherHostData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddInt8Basic(aclrtStream stream) {
  const std::string testName = "add_int8_basic";
  std::vector<int64_t> shape = {2, 2};
  std::vector<int8_t> selfHostData = {1, 2, 3, 4};
  std::vector<int8_t> otherHostData = {5, 6, 7, 8};
  std::vector<int8_t> outHostData(4, 0);
  int8_t alphaValue = 1;

  TensorResource<int8_t> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_INT8);
  bool ok = (alpha != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_INT8)) &&
       CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_INT8)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_INT8)) &&
       RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<int8_t> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         ExactVectorEqual(result, AddExpectedIntegral(selfHostData, otherHostData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddUint8Basic(aclrtStream stream) {
  const std::string testName = "add_uint8_basic";
  std::vector<int64_t> shape = {2, 2};
  std::vector<uint8_t> selfHostData = {1, 2, 3, 4};
  std::vector<uint8_t> otherHostData = {5, 6, 7, 8};
  std::vector<uint8_t> outHostData(4, 0);
  uint8_t alphaValue = 1;

  TensorResource<uint8_t> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_UINT8);
  bool ok = (alpha != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_UINT8)) &&
       CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_UINT8)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_UINT8)) &&
       RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<uint8_t> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         ExactVectorEqual(result, AddExpectedIntegral(selfHostData, otherHostData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddDoubleBasic(aclrtStream stream) {
  const std::string testName = "add_double_basic";
  std::vector<int64_t> shape = {2, 2};
  std::vector<double> selfHostData = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> otherHostData = {5.0, 6.0, 7.0, 8.0};
  std::vector<double> outHostData(4, 0.0);
  double alphaValue = 0.5;

  TensorResource<double> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_DOUBLE);
  bool ok = (alpha != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_DOUBLE)) &&
       CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_DOUBLE)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_DOUBLE)) &&
       RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<double> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         DoubleVectorAlmostEqual(result, AddExpected(selfHostData, otherHostData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddFloat16(aclrtStream stream) {
  const std::string testName = "add_float16";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfFloatData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> otherFloatData = {0.5f, 1.5f, 2.5f, 3.5f};
  std::vector<uint16_t> selfHostData = FloatsToHalfBits(selfFloatData);
  std::vector<uint16_t> otherHostData = FloatsToHalfBits(otherFloatData);
  std::vector<uint16_t> outHostData(4, 0);
  float alphaValue = 2.0f;

  TensorResource<uint16_t> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT16)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT16)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT16)) &&
            RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<uint16_t> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         ExactVectorEqual(result, AddExpectedHalfBits(selfFloatData, otherFloatData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddFloat16Basic(aclrtStream stream) {
  const std::string testName = "add_float16_basic";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfFloatData = {-1.0f, 0.0f, 1.0f, 2.0f};
  std::vector<float> otherFloatData = {0.5f, 1.0f, 1.5f, 2.0f};
  std::vector<uint16_t> selfHostData = FloatsToHalfBits(selfFloatData);
  std::vector<uint16_t> otherHostData = FloatsToHalfBits(otherFloatData);
  std::vector<uint16_t> outHostData(4, 0);
  float alphaValue = 1.0f;

  TensorResource<uint16_t> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT16)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT16)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT16)) &&
            RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<uint16_t> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         ExactVectorEqual(result, AddExpectedHalfBits(selfFloatData, otherFloatData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddInt32(aclrtStream stream) {
  const std::string testName = "add_int32";
  std::vector<int64_t> shape = {2, 2};
  std::vector<int32_t> selfHostData = {1, 2, 3, 4};
  std::vector<int32_t> otherHostData = {5, 6, 7, 8};
  std::vector<int32_t> outHostData(4, 0);
  int32_t alphaValue = 2;

  TensorResource<int32_t> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_INT32);
  bool ok = (alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_INT32)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_INT32)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_INT32)) &&
            RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<int32_t> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         ExactVectorEqual(result, AddExpectedIntegral(selfHostData, otherHostData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddMixedType(aclrtStream stream) {
  const std::string testName = "add_mixed_type";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfFloatData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<uint16_t> selfHostData = FloatsToHalfBits(selfFloatData);
  std::vector<float> otherHostData = {0.5f, 1.5f, 2.5f, 3.5f};
  std::vector<float> outHostData(4, 0.0f);
  float alphaValue = 1.0f;

  TensorResource<uint16_t> self;
  TensorResource<float> other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT16)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
            RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         FloatVectorAlmostEqual(result, AddExpected(selfFloatData, otherHostData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddBroadcastRankdiffA(aclrtStream stream) {
  const std::string testName = "add_broadcast_rankdiff_a";
  std::vector<int64_t> selfShape = {2, 3, 4};
  std::vector<int64_t> otherShape = {4};
  std::vector<int64_t> outShape = {2, 3, 4};
  std::vector<float> selfHostData(GetShapeSize(selfShape), 0.0f);
  std::vector<float> otherHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  for (size_t i = 0; i < selfHostData.size(); ++i) {
    selfHostData[i] = static_cast<float>(i);
  }
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);
  float alphaValue = 1.5f;

  TensorResource<float> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create other", other.Create(otherHostData, otherShape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT)) &&
       RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         FloatVectorAlmostEqual(result,
                                BroadcastAddExpected(selfHostData, selfShape, otherHostData, otherShape, outShape,
                                                     alphaValue),
                                testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddBroadcastRankdiffB(aclrtStream stream) {
  const std::string testName = "add_broadcast_rankdiff_b";
  std::vector<int64_t> selfShape = {4};
  std::vector<int64_t> otherShape = {2, 1, 4};
  std::vector<int64_t> outShape = {2, 1, 4};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> otherHostData = {10.0f, 20.0f, 30.0f, 40.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);
  float alphaValue = -1.0f;

  TensorResource<float> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create other", other.Create(otherHostData, otherShape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT)) &&
       RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         FloatVectorAlmostEqual(result,
                                BroadcastAddExpected(selfHostData, selfShape, otherHostData, otherShape, outShape,
                                                     alphaValue),
                                testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddBroadcastScalarLike(aclrtStream stream) {
  const std::string testName = "add_broadcast_scalar_like";
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> otherShape = {1};
  std::vector<int64_t> outShape = {2, 2};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> otherHostData = {5.0f};
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);
  float alphaValue = 0.5f;

  TensorResource<float> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create other", other.Create(otherHostData, otherShape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT)) &&
       RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         FloatVectorAlmostEqual(result,
                                BroadcastAddExpected(selfHostData, selfShape, otherHostData, otherShape, outShape,
                                                     alphaValue),
                                testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddInvalidBroadcastExtra(aclrtStream stream) {
  (void)stream;
  const std::string testName = "add_invalid_broadcast_extra";
  std::vector<int64_t> selfShape = {2, 1, 3};
  std::vector<int64_t> otherShape = {2, 2, 2};
  std::vector<int64_t> outShape = {2, 2, 3};
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> otherHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);
  float alphaValue = 1.0f;

  TensorResource<float> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, otherShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddInvalidOutshape(aclrtStream stream) {
  (void)stream;
  const std::string testName = "add_invalid_outshape";
  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> otherShape = {3};
  std::vector<int64_t> outShape = {3, 2};
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> otherHostData = {1, 2, 3};
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);
  float alphaValue = 1.0f;

  TensorResource<float> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, otherShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddInvalidHighdimBroadcast(aclrtStream stream) {
  (void)stream;
  const std::string testName = "add_invalid_highdim_broadcast";
  std::vector<int64_t> selfShape = {2, 1, 3, 1};
  std::vector<int64_t> otherShape = {1, 4, 2};
  std::vector<int64_t> outShape = {2, 4, 3, 2};
  std::vector<float> selfHostData(GetShapeSize(selfShape), 1.0f);
  std::vector<float> otherHostData(GetShapeSize(otherShape), 2.0f);
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);
  float alphaValue = 1.0f;

  TensorResource<float> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, otherShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestNullptrExecutorInvalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "nullptr_executor_invalid";
  auto ret = aclnnAdd(nullptr, 0, nullptr, nullptr);
  bool ok = (ret != ACL_SUCCESS);
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestUnsupportedDtypeInvalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "unsupported_dtype_invalid";
  std::vector<int64_t> shape = {2};
  std::vector<int32_t> selfHostData = {1, 2};
  std::vector<int32_t> otherHostData = {3, 4};
  std::vector<int32_t> outHostData = {0, 0};
  int32_t alphaValue = 1;

  TensorResource<int32_t> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, static_cast<aclDataType>(999));
  int ret1 = self.Create(selfHostData, shape, static_cast<aclDataType>(999));
  int ret2 = other.Create(otherHostData, shape, static_cast<aclDataType>(999));
  int ret3 = out.Create(outHostData, shape, static_cast<aclDataType>(999));
  bool ok = true;

  if (ret1 == ACL_SUCCESS && ret2 == ACL_SUCCESS && ret3 == ACL_SUCCESS && alpha != nullptr) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestSelfOtherDtypeMismatchInvalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "self_other_dtype_mismatch_invalid";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {1, 2, 3, 4};
  std::vector<int32_t> otherHostData = {5, 6, 7, 8};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float alphaValue = 1.0f;

  TensorResource<float> self;
  TensorResource<int32_t> other;
  TensorResource<float> out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_INT32)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAlphaDtypeMismatchInvalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "alpha_dtype_mismatch_invalid";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {1, 2, 3, 4};
  std::vector<float> otherHostData = {5, 6, 7, 8};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  int32_t alphaValue = 2;

  TensorResource<float> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_INT32);
  bool ok = (alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestOutDtypeMismatchInvalidExtra(aclrtStream stream) {
  (void)stream;
  const std::string testName = "out_dtype_mismatch_invalid_extra";
  std::vector<int64_t> shape = {2, 2};
  std::vector<double> selfHostData = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> otherHostData = {5.0, 6.0, 7.0, 8.0};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  double alphaValue = 0.5;

  TensorResource<double> self, other;
  TensorResource<float> out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_DOUBLE);
  bool ok = (alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_DOUBLE)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_DOUBLE)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestInplaceAddBasic(aclrtStream stream) {
  const std::string testName = "inplace_add_basic";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> otherHostData = {5.0f, 6.0f, 7.0f, 8.0f};
  float alphaValue = 1.0f;

  TensorResource<float> self, other;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
            RunAclnnInplaceAdd(self.tensor, other.tensor, alpha, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, self, &result) &&
         FloatVectorAlmostEqual(result, AddExpected(selfHostData, otherHostData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestInplaceAddAlphaZero(aclrtStream stream) {
  const std::string testName = "inplace_add_alpha_zero";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> otherHostData = {5.0f, 6.0f, 7.0f, 8.0f};
  float alphaValue = 0.0f;

  TensorResource<float> self, other;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
            RunAclnnInplaceAdd(self.tensor, other.tensor, alpha, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, self, &result) &&
         FloatVectorAlmostEqual(result, AddExpected(selfHostData, otherHostData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestInplaceAddAlphaNegative(aclrtStream stream) {
  const std::string testName = "inplace_add_alpha_negative";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {1.0f, -2.0f, 3.0f, -4.0f};
  std::vector<float> otherHostData = {5.0f, 6.0f, 7.0f, 8.0f};
  float alphaValue = -0.5f;

  TensorResource<float> self, other;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
            RunAclnnInplaceAdd(self.tensor, other.tensor, alpha, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, self, &result) &&
         FloatVectorAlmostEqual(result, AddExpected(selfHostData, otherHostData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestInplaceAddInvalidBroadcast(aclrtStream stream) {
  (void)stream;
  const std::string testName = "inplace_add_invalid_broadcast";
  std::vector<int64_t> selfShape = {3};
  std::vector<int64_t> otherShape = {2, 3};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f};
  std::vector<float> otherHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float alphaValue = 1.0f;

  TensorResource<float> self, other;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, otherShape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestInplaceAddInvalidShape(aclrtStream stream) {
  (void)stream;
  const std::string testName = "inplace_add_invalid_shape";
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> otherShape = {3, 2};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> otherHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float alphaValue = 1.0f;

  TensorResource<float> self, other;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, otherShape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestInplaceAddNullptrInvalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "inplace_add_nullptr_invalid";
  auto ret = aclnnInplaceAdd(nullptr, 0, nullptr, nullptr);
  bool ok = (ret != ACL_SUCCESS);
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddsBasic(aclrtStream stream) {
  const std::string testName = "adds_basic";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> outHostData(4, 0.0f);
  float otherValue = 5.0f;
  float alphaValue = 2.0f;

  TensorResource<float> self, out;
  aclScalar* other = aclCreateScalar(&otherValue, ACL_FLOAT);
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (other != nullptr && alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
            RunAclnnAdds(self.tensor, other, alpha, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         FloatVectorAlmostEqual(result, AddScalarExpected(selfHostData, otherValue, alphaValue), testName);
  }

  if (other != nullptr) {
    aclDestroyScalar(other);
  }
  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestInplaceAddsBasic(aclrtStream stream) {
  const std::string testName = "inplace_adds_basic";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {1.0f, -2.0f, 3.0f, -4.0f};
  float otherValue = 3.0f;
  float alphaValue = -1.0f;

  TensorResource<float> self;
  aclScalar* other = aclCreateScalar(&otherValue, ACL_FLOAT);
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (other != nullptr && alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            RunAclnnInplaceAdds(self.tensor, other, alpha, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, self, &result) &&
         FloatVectorAlmostEqual(result, AddScalarExpected(selfHostData, otherValue, alphaValue), testName);
  }

  if (other != nullptr) {
    aclDestroyScalar(other);
  }
  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddsNullptrInvalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "adds_nullptr_invalid";
  std::vector<int64_t> shape = {2};
  std::vector<float> selfHostData = {1.0f, 2.0f};
  std::vector<float> outHostData = {0.0f, 0.0f};
  float alphaValue = 1.0f;

  TensorResource<float> self, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self.tensor, nullptr, alpha, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddsFloat16(aclrtStream stream) {
  const std::string testName = "adds_float16";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfFloatData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<uint16_t> selfHostData = FloatsToHalfBits(selfFloatData);
  std::vector<uint16_t> outHostData(4, 0);
  float otherValue = 5.0f;
  float alphaValue = 1.0f;

  TensorResource<uint16_t> self, out;
  aclScalar* other = aclCreateScalar(&otherValue, ACL_FLOAT);
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (other != nullptr && alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT16)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT16)) &&
            RunAclnnAdds(self.tensor, other, alpha, out.tensor, stream, testName);

  if (other != nullptr) {
    aclDestroyScalar(other);
  }
  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddsInt32(aclrtStream stream) {
  const std::string testName = "adds_int32";
  std::vector<int64_t> shape = {2, 2};
  std::vector<int32_t> selfHostData = {1, 2, 3, 4};
  std::vector<int32_t> outHostData(4, 0);
  int32_t otherValue = 5;
  int32_t alphaValue = 2;

  TensorResource<int32_t> self, out;
  aclScalar* other = aclCreateScalar(&otherValue, ACL_INT32);
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_INT32);
  bool ok = (other != nullptr && alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_INT32)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_INT32)) &&
            RunAclnnAdds(self.tensor, other, alpha, out.tensor, stream, testName);

  std::vector<int32_t> result;
  if (ok) {
    std::vector<int32_t> expected(selfHostData.size(), 0);
    for (size_t i = 0; i < selfHostData.size(); ++i) {
      expected[i] = static_cast<int32_t>(selfHostData[i] + otherValue * alphaValue);
    }
    ok = CopyDeviceToHost(testName, out, &result) && ExactVectorEqual(result, expected, testName);
  }

  if (other != nullptr) {
    aclDestroyScalar(other);
  }
  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestInplaceAddsFloat16(aclrtStream stream) {
  const std::string testName = "inplace_adds_float16";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfFloatData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<uint16_t> selfHostData = FloatsToHalfBits(selfFloatData);
  float otherValue = 3.0f;
  float alphaValue = 1.0f;

  TensorResource<uint16_t> self;
  aclScalar* other = aclCreateScalar(&otherValue, ACL_FLOAT);
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (other != nullptr && alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT16)) &&
            RunAclnnInplaceAdds(self.tensor, other, alpha, stream, testName);

  if (other != nullptr) {
    aclDestroyScalar(other);
  }
  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddsAlphaZeroFastPath(aclrtStream stream) {
  const std::string testName = "adds_alpha_zero_fastpath";
  std::vector<int64_t> shape = {2};
  std::vector<float> selfHostData = {1.0f, 2.0f};
  std::vector<float> outHostData = {0.0f, 0.0f};
  float otherValue = 999.0f;
  float alphaValue = 0.0f;

  TensorResource<float> self, out;
  aclScalar* other = aclCreateScalar(&otherValue, ACL_FLOAT);
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
  bool ok = (other != nullptr && alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
            RunAclnnAdds(self.tensor, other, alpha, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         FloatVectorAlmostEqual(result, AddScalarExpected(selfHostData, otherValue, alphaValue), testName);
  }

  if (other != nullptr) {
    aclDestroyScalar(other);
  }
  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestAddInt64Basic(aclrtStream stream) {
  const std::string testName = "add_int64_basic";
  std::vector<int64_t> shape = {2};
  std::vector<int64_t> selfHostData = {10000000000LL, 20000000000LL};
  std::vector<int64_t> otherHostData = {5000000000LL, 6000000000LL};
  std::vector<int64_t> outHostData(2, 0);
  int64_t alphaValue = 1;

  TensorResource<int64_t> self, other, out;
  aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_INT64);
  bool ok = (alpha != nullptr) &&
            CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_INT64)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_INT64)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_INT64)) &&
            RunAdd(self.tensor, other.tensor, alpha, out.tensor, stream, testName);

  std::vector<int64_t> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) &&
         ExactVectorEqual(result, AddExpectedIntegral(selfHostData, otherHostData, alphaValue), testName);
  }

  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  self.Destroy();
  other.Destroy();
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
  failedCount += TestAddBasic(stream) ? 0 : 1;
  failedCount += TestAddNegZero(stream) ? 0 : 1;
  failedCount += TestAddAlphaZero(stream) ? 0 : 1;
  failedCount += TestAddAlphaOne(stream) ? 0 : 1;
  failedCount += TestAddAlphaNegative(stream) ? 0 : 1;
  failedCount += TestAddAlphaFraction(stream) ? 0 : 1;
  failedCount += TestAddBroadcastBasic(stream) ? 0 : 1;
  failedCount += TestAddBroadcastHighdim(stream) ? 0 : 1;
  failedCount += TestAddInvalidBroadcast(stream) ? 0 : 1;
  failedCount += TestNullptrSelfInvalid(stream) ? 0 : 1;
  failedCount += TestNullptrOtherInvalid(stream) ? 0 : 1;
  failedCount += TestNullptrOutInvalid(stream) ? 0 : 1;
  failedCount += TestNullptrAlphaInvalid(stream) ? 0 : 1;
  failedCount += TestOutDtypeMismatchInvalid(stream) ? 0 : 1;
  failedCount += TestAddSpecialValues(stream) ? 0 : 1;
  failedCount += TestAddAlphaLargePositive(stream) ? 0 : 1;
  failedCount += TestAddAlphaLargeNegative(stream) ? 0 : 1;
  failedCount += TestAddAlphaNan(stream) ? 0 : 1;
  failedCount += TestAddAlphaInf(stream) ? 0 : 1;
  failedCount += TestAddAlphaNegInf(stream) ? 0 : 1;
  failedCount += TestAddInt32Basic(stream) ? 0 : 1;
  failedCount += TestAddInt16Basic(stream) ? 0 : 1;
  failedCount += TestAddInt8Basic(stream) ? 0 : 1;
  failedCount += TestAddUint8Basic(stream) ? 0 : 1;
  failedCount += TestAddDoubleBasic(stream) ? 0 : 1;
  failedCount += TestAddFloat16(stream) ? 0 : 1;
  failedCount += TestAddFloat16Basic(stream) ? 0 : 1;
  failedCount += TestAddInt32(stream) ? 0 : 1;
  failedCount += TestAddMixedType(stream) ? 0 : 1;
  failedCount += TestAddBroadcastRankdiffA(stream) ? 0 : 1;
  failedCount += TestAddBroadcastRankdiffB(stream) ? 0 : 1;
  failedCount += TestAddBroadcastScalarLike(stream) ? 0 : 1;
  failedCount += TestAddInvalidBroadcastExtra(stream) ? 0 : 1;
  failedCount += TestAddInvalidOutshape(stream) ? 0 : 1;
  failedCount += TestAddInvalidHighdimBroadcast(stream) ? 0 : 1;
  failedCount += TestNullptrExecutorInvalid(stream) ? 0 : 1;
  failedCount += TestUnsupportedDtypeInvalid(stream) ? 0 : 1;
  failedCount += TestSelfOtherDtypeMismatchInvalid(stream) ? 0 : 1;
  failedCount += TestAlphaDtypeMismatchInvalid(stream) ? 0 : 1;
  failedCount += TestOutDtypeMismatchInvalidExtra(stream) ? 0 : 1;
  failedCount += TestInplaceAddBasic(stream) ? 0 : 1;
  failedCount += TestInplaceAddAlphaZero(stream) ? 0 : 1;
  failedCount += TestInplaceAddAlphaNegative(stream) ? 0 : 1;
  failedCount += TestInplaceAddInvalidBroadcast(stream) ? 0 : 1;
  failedCount += TestInplaceAddInvalidShape(stream) ? 0 : 1;
  failedCount += TestInplaceAddNullptrInvalid(stream) ? 0 : 1;
  failedCount += TestAddsBasic(stream) ? 0 : 1;
  failedCount += TestInplaceAddsBasic(stream) ? 0 : 1;
  failedCount += TestAddsNullptrInvalid(stream) ? 0 : 1;
  failedCount += TestAddsFloat16(stream) ? 0 : 1;
  failedCount += TestAddsInt32(stream) ? 0 : 1;
  failedCount += TestInplaceAddsFloat16(stream) ? 0 : 1;
  failedCount += TestAddsAlphaZeroFastPath(stream) ? 0 : 1;
  failedCount += TestAddInt64Basic(stream) ? 0 : 1;

  const int totalCount = 54;
  LOG_PRINT("==================================================\n");
  LOG_PRINT("Add test summary: total=%d, passed=%d, failed=%d\n", totalCount, totalCount - failedCount, failedCount);

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return failedCount == 0 ? 0 : 1;
}
