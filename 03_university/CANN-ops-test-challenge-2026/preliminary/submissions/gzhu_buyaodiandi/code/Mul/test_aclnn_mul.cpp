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

template <typename T>
bool FloatingVectorAlmostEqual(const std::vector<T>& actual, const std::vector<T>& expected, const std::string& testName,
                               double atol = 1e-5, double rtol = 1e-5) {
  if (actual.size() != expected.size()) {
    LOG_PRINT("  %s size mismatch. actual=%zu expected=%zu\n", testName.c_str(), actual.size(), expected.size());
    return false;
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!AlmostEqual(static_cast<double>(expected[i]), static_cast<double>(actual[i]), atol, rtol)) {
      LOG_PRINT("  %s mismatch at %zu. actual=%lf expected=%lf\n", testName.c_str(), i, static_cast<double>(actual[i]),
                static_cast<double>(expected[i]));
      return false;
    }
  }
  return true;
}

bool FloatVectorAlmostEqual(const std::vector<float>& actual, const std::vector<float>& expected,
                            const std::string& testName) {
  return FloatingVectorAlmostEqual(actual, expected, testName);
}

template <typename T>
std::vector<T> ElementwiseMulExpected(const std::vector<T>& selfData, const std::vector<T>& otherData) {
  std::vector<T> expected(selfData.size(), static_cast<T>(0));
  for (size_t i = 0; i < selfData.size(); ++i) {
    expected[i] = selfData[i] * otherData[i];
  }
  return expected;
}

template <typename T>
std::vector<T> MulsExpected(const std::vector<T>& selfData, T scalar) {
  std::vector<T> expected(selfData.size(), static_cast<T>(0));
  for (size_t i = 0; i < selfData.size(); ++i) {
    expected[i] = selfData[i] * scalar;
  }
  return expected;
}

std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  return strides;
}

template <typename T>
std::vector<T> BroadcastMulExpected(const std::vector<T>& selfData, const std::vector<int64_t>& selfShape,
                                    const std::vector<T>& otherData, const std::vector<int64_t>& otherShape,
                                    const std::vector<int64_t>& outShape) {
  std::vector<T> expected(GetShapeSize(outShape), static_cast<T>(0));
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
    expected[outIndex] = selfData[selfIndex] * otherData[otherIndex];
  }
  return expected;
}

bool RunAclnnMul(aclTensor* self, aclTensor* other, aclTensor* out, aclrtStream stream, const std::string& testName) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(CheckAclRet(testName, "aclnnMulGetWorkspaceSize", ret), return false);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(CheckAclRet(testName, "aclrtMalloc workspace", ret), return false);
  }

  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  bool ok = CheckAclRet(testName, "aclnnMul", ret);
  if (ok) {
    ret = aclrtSynchronizeStream(stream);
    ok = CheckAclRet(testName, "aclrtSynchronizeStream", ret);
  }

  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ok;
}

bool RunAclnnMuls(aclTensor* self, aclScalar* scalar, aclTensor* out, aclrtStream stream, const std::string& testName) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulsGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor);
  CHECK_RET(CheckAclRet(testName, "aclnnMulsGetWorkspaceSize", ret), return false);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(CheckAclRet(testName, "aclrtMalloc workspace", ret), return false);
  }

  ret = aclnnMuls(workspaceAddr, workspaceSize, executor, stream);
  bool ok = CheckAclRet(testName, "aclnnMuls", ret);
  if (ok) {
    ret = aclrtSynchronizeStream(stream);
    ok = CheckAclRet(testName, "aclrtSynchronizeStream", ret);
  }

  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ok;
}

bool RunAclnnInplaceMul(aclTensor* self, aclTensor* other, aclrtStream stream, const std::string& testName) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnInplaceMulGetWorkspaceSize(self, other, &workspaceSize, &executor);
  CHECK_RET(CheckAclRet(testName, "aclnnInplaceMulGetWorkspaceSize", ret), return false);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(CheckAclRet(testName, "aclrtMalloc workspace", ret), return false);
  }

  ret = aclnnInplaceMul(workspaceAddr, workspaceSize, executor, stream);
  bool ok = CheckAclRet(testName, "aclnnInplaceMul", ret);
  if (ok) {
    ret = aclrtSynchronizeStream(stream);
    ok = CheckAclRet(testName, "aclrtSynchronizeStream", ret);
  }

  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ok;
}

bool RunAclnnInplaceMuls(aclTensor* self, aclScalar* scalar, aclrtStream stream, const std::string& testName) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnInplaceMulsGetWorkspaceSize(self, scalar, &workspaceSize, &executor);
  CHECK_RET(CheckAclRet(testName, "aclnnInplaceMulsGetWorkspaceSize", ret), return false);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(CheckAclRet(testName, "aclrtMalloc workspace", ret), return false);
  }

  ret = aclnnInplaceMuls(workspaceAddr, workspaceSize, executor, stream);
  bool ok = CheckAclRet(testName, "aclnnInplaceMuls", ret);
  if (ok) {
    ret = aclrtSynchronizeStream(stream);
    ok = CheckAclRet(testName, "aclrtSynchronizeStream", ret);
  }

  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ok;
}

bool TestFloat32Basic(aclrtStream stream) {
  const std::string testName = "float32_basic";
  std::vector<int64_t> shape = {4, 2};
  std::vector<float> selfHostData = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<float> otherHostData = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> other;
  TensorResource<float> out;

  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
            RunAclnnMul(self.tensor, other.tensor, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = ElementwiseMulExpected(selfHostData, otherHostData);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestFloat32NegZero(aclrtStream stream) {
  const std::string testName = "float32_neg_zero";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHostData = {-1.0f, 0.0f, -2.5f, 3.0f, 0.0f, -4.0f, 5.5f, -0.0f};
  std::vector<float> otherHostData = {0.0f, -3.0f, 2.0f, -0.0f, -7.0f, -1.5f, 0.0f, 6.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> other;
  TensorResource<float> out;

  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
            RunAclnnMul(self.tensor, other.tensor, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = ElementwiseMulExpected(selfHostData, otherHostData);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestFloat32Muls(aclrtStream stream) {
  const std::string testName = "float32_muls";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHostData = {-2.0f, -1.0f, 0.0f, 1.5f, 2.0f, 3.5f, 4.0f, 8.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float scalarValue = -2.5f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);

  bool ok = (scalar != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunAclnnMuls(self.tensor, scalar, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = MulsExpected(selfHostData, scalarValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (scalar != nullptr) {
    aclDestroyScalar(scalar);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestInt32Basic(aclrtStream stream) {
  const std::string testName = "int32_basic";
  std::vector<int64_t> shape = {2, 4};
  std::vector<int32_t> selfHostData = {-3, -2, -1, 0, 1, 2, 3, 4};
  std::vector<int32_t> otherHostData = {4, -3, 2, 100, -1, 0, 7, -8};
  std::vector<int32_t> outHostData(GetShapeSize(shape), 0);

  TensorResource<int32_t> self;
  TensorResource<int32_t> other;
  TensorResource<int32_t> out;

  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_INT32)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_INT32)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_INT32)) &&
            RunAclnnMul(self.tensor, other.tensor, out.tensor, stream, testName);

  std::vector<int32_t> result;
  if (ok) {
    std::vector<int32_t> expected = ElementwiseMulExpected(selfHostData, otherHostData);
    ok = CopyDeviceToHost(testName, out, &result) && ExactVectorEqual(result, expected, testName);
  }

  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestBroadcastBasic(aclrtStream stream) {
  const std::string testName = "broadcast_basic";
  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> otherShape = {3};
  std::vector<int64_t> outShape = {2, 3};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f};
  std::vector<float> otherHostData = {10.0f, 0.5f, -2.0f};
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> other;
  TensorResource<float> out;

  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, otherShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT)) &&
            RunAclnnMul(self.tensor, other.tensor, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = BroadcastMulExpected(selfHostData, selfShape, otherHostData, otherShape, outShape);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestBroadcastInvalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "broadcast_invalid";
  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> otherShape = {2, 2};
  std::vector<int64_t> outShape = {2, 3};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> otherHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> other;
  TensorResource<float> out;

  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, otherShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
    if (!ok) {
      LOG_PRINT("  %s expected aclnnMulGetWorkspaceSize to fail, but got ACL_SUCCESS.\n", testName.c_str());
    }
  }

  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestInplaceMulBasic(aclrtStream stream) {
  const std::string testName = "inplace_mul_basic";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHostData = {1.0f, -2.0f, 3.0f, -4.0f, 0.0f, 5.0f, -6.0f, 7.0f};
  std::vector<float> otherHostData = {2.0f, 3.0f, -1.0f, -0.5f, 9.0f, 0.0f, -2.0f, 4.0f};

  TensorResource<float> self;
  TensorResource<float> other;

  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
            RunAclnnInplaceMul(self.tensor, other.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = ElementwiseMulExpected(selfHostData, otherHostData);
    ok = CopyDeviceToHost(testName, self, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  self.Destroy();
  other.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestInplaceMulsBasic(aclrtStream stream) {
  const std::string testName = "inplace_muls_basic";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHostData = {-4.0f, -2.0f, 0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f};
  float scalarValue = 0.25f;

  TensorResource<float> self;
  aclScalar* scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);

  bool ok = (scalar != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }

  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       RunAclnnInplaceMuls(self.tensor, scalar, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = MulsExpected(selfHostData, scalarValue);
    ok = CopyDeviceToHost(testName, self, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  if (scalar != nullptr) {
    aclDestroyScalar(scalar);
  }
  self.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestDoubleExtreme(aclrtStream stream) {
  const std::string testName = "double_extreme_boundary";
  std::vector<int64_t> shape = {4, 2};
  std::vector<double> selfHostData = {INFINITY, -INFINITY, NAN, 1e-300, 1e300, 0.0, -0.0, -5.5};
  std::vector<double> otherHostData = {0.0, INFINITY, -INFINITY, 1e-300, 1e300, NAN, -0.0, 2.0};
  std::vector<double> outHostData(GetShapeSize(shape), 0.0);

  TensorResource<double> self;
  TensorResource<double> other;
  TensorResource<double> out;

  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_DOUBLE)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_DOUBLE)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_DOUBLE)) &&
            RunAclnnMul(self.tensor, other.tensor, out.tensor, stream, testName);

  std::vector<double> result;
  if (ok) {
    std::vector<double> expected = ElementwiseMulExpected(selfHostData, otherHostData);
    ok = CopyDeviceToHost(testName, out, &result);
    if (ok) {
      for (size_t i = 0; i < result.size(); ++i) {
        if (!AlmostEqual(expected[i], result[i], 1e-10, 1e-10)) {
          LOG_PRINT("  %s mismatch at %zu. actual=%lf expected=%lf\n", testName.c_str(), i, result[i], expected[i]);
          ok = false;
          break;
        }
      }
    }
  }

  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestFloat16Float32Mixed(aclrtStream stream) {
  const std::string testName = "float16_float32_mixed";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfFloatData = {0.5f, -1.5f, 2.0f, -4.0f, 8.0f, 0.25f, -0.75f, 16.0f};
  std::vector<float> otherHostData = {2.0f, 3.0f, -1.5f, 0.5f, -2.0f, 4.0f, 8.0f, -0.25f};
  std::vector<uint16_t> selfHostData = FloatsToHalfBits(selfFloatData);
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);

  TensorResource<uint16_t> self;
  TensorResource<float> other;
  TensorResource<float> out;

  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT16)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
            RunAclnnMul(self.tensor, other.tensor, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = ElementwiseMulExpected(selfFloatData, otherHostData);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestFloat32Float16Mixed(aclrtStream stream) {
  const std::string testName = "float32_float16_mixed";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHostData = {1.25f, -2.0f, 3.5f, -4.5f, 0.125f, 6.0f, -7.0f, 8.0f};
  std::vector<float> otherFloatData = {-2.0f, 0.5f, 4.0f, -1.0f, 8.0f, -0.25f, 2.0f, 0.0f};
  std::vector<uint16_t> otherHostData = FloatsToHalfBits(otherFloatData);
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);

  TensorResource<float> self;
  TensorResource<uint16_t> other;
  TensorResource<float> out;

  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT16)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
            RunAclnnMul(self.tensor, other.tensor, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = ElementwiseMulExpected(selfHostData, otherFloatData);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestDoubleBasic(aclrtStream stream) {
  const std::string testName = "double_basic";
  std::vector<int64_t> shape = {2, 4};
  std::vector<double> selfHostData = {1.0, -2.0, 3.5, -4.5, 5.25, 6.0, -7.75, 8.0};
  std::vector<double> otherHostData = {2.0, 3.0, -1.0, 0.5, 4.0, -2.0, 0.25, -8.0};
  std::vector<double> outHostData(GetShapeSize(shape), 0.0);

  TensorResource<double> self;
  TensorResource<double> other;
  TensorResource<double> out;

  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_DOUBLE)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_DOUBLE)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_DOUBLE)) &&
            RunAclnnMul(self.tensor, other.tensor, out.tensor, stream, testName);

  std::vector<double> result;
  if (ok) {
    std::vector<double> expected = ElementwiseMulExpected(selfHostData, otherHostData);
    ok = CopyDeviceToHost(testName, out, &result) &&
         FloatingVectorAlmostEqual(result, expected, testName, 1e-10, 1e-10);
  }

  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestBoolBasic(aclrtStream stream) {
  const std::string testName = "bool_basic";
  std::vector<int64_t> shape = {2, 4};
  std::vector<uint8_t> selfHostData = {1, 0, 1, 0, 1, 1, 0, 0};
  std::vector<uint8_t> otherHostData = {1, 1, 0, 0, 1, 0, 1, 0};
  std::vector<uint8_t> outHostData(GetShapeSize(shape), 0);

  TensorResource<uint8_t> self;
  TensorResource<uint8_t> other;
  TensorResource<uint8_t> out;

  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_BOOL)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_BOOL)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_BOOL)) &&
            RunAclnnMul(self.tensor, other.tensor, out.tensor, stream, testName);

  std::vector<uint8_t> result;
  if (ok) {
    std::vector<uint8_t> expected = ElementwiseMulExpected(selfHostData, otherHostData);
    ok = CopyDeviceToHost(testName, out, &result) && ExactVectorEqual(result, expected, testName);
  }

  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestInt8Basic(aclrtStream stream) {
  const std::string testName = "int8_basic";
  std::vector<int64_t> shape = {2, 4};
  std::vector<int8_t> selfHostData = {-8, -4, -1, 0, 1, 2, 3, 4};
  std::vector<int8_t> otherHostData = {4, -3, 2, 7, -1, 0, 5, -2};
  std::vector<int8_t> outHostData(GetShapeSize(shape), 0);

  TensorResource<int8_t> self;
  TensorResource<int8_t> other;
  TensorResource<int8_t> out;

  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_INT8)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_INT8)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_INT8)) &&
            RunAclnnMul(self.tensor, other.tensor, out.tensor, stream, testName);

  std::vector<int8_t> result;
  if (ok) {
    std::vector<int8_t> expected = ElementwiseMulExpected(selfHostData, otherHostData);
    ok = CopyDeviceToHost(testName, out, &result) && ExactVectorEqual(result, expected, testName);
  }

  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestUint8Basic(aclrtStream stream) {
  const std::string testName = "uint8_basic";
  std::vector<int64_t> shape = {2, 4};
  std::vector<uint8_t> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint8_t> otherHostData = {8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<uint8_t> outHostData(GetShapeSize(shape), 0);

  TensorResource<uint8_t> self;
  TensorResource<uint8_t> other;
  TensorResource<uint8_t> out;

  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_UINT8)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_UINT8)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_UINT8)) &&
            RunAclnnMul(self.tensor, other.tensor, out.tensor, stream, testName);

  std::vector<uint8_t> result;
  if (ok) {
    std::vector<uint8_t> expected = ElementwiseMulExpected(selfHostData, otherHostData);
    ok = CopyDeviceToHost(testName, out, &result) && ExactVectorEqual(result, expected, testName);
  }

  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestNullptrInvalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "nullptr_invalid";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> otherHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);

  TensorResource<float> other;
  TensorResource<float> out;

  bool ok = CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(nullptr, other.tensor, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
    if (!ok) {
      LOG_PRINT("  %s expected aclnnMulGetWorkspaceSize to fail with nullptr input.\n", testName.c_str());
    }
  }

  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestBroadcastHighdim(aclrtStream stream) {
  const std::string testName = "broadcast_highdim";
  std::vector<int64_t> selfShape = {2, 1, 3, 1};
  std::vector<int64_t> otherShape = {1, 4, 1, 5};
  std::vector<int64_t> outShape = {2, 4, 3, 5};
  std::vector<float> selfHostData = {
      1.0f, 2.0f, 3.0f,
      -1.0f, -2.0f, -3.0f};
  std::vector<float> otherHostData = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
      -1.0f, -2.0f, -3.0f, -4.0f, -5.0f,
      0.5f, 1.5f, 2.5f, 3.5f, 4.5f,
      -0.5f, -1.5f, -2.5f, -3.5f, -4.5f};
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> other;
  TensorResource<float> out;

  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, otherShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT)) &&
            RunAclnnMul(self.tensor, other.tensor, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = BroadcastMulExpected(selfHostData, selfShape, otherHostData, otherShape, outShape);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }

  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestInt64Basic(aclrtStream stream) {
  const std::string testName = "int64_basic";
  std::vector<int64_t> shape = {2};
  std::vector<int64_t> selfHostData = {-10000000000LL, 20000000000LL};
  std::vector<int64_t> otherHostData = {2LL, -3LL};
  std::vector<int64_t> outHostData(GetShapeSize(shape), 0);

  TensorResource<int64_t> self;
  TensorResource<int64_t> other;
  TensorResource<int64_t> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_INT64)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_INT64)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_INT64)) &&
            RunAclnnMul(self.tensor, other.tensor, out.tensor, stream, testName);

  std::vector<int64_t> result;
  if (ok) {
    std::vector<int64_t> expected = ElementwiseMulExpected(selfHostData, otherHostData);
    ok = CopyDeviceToHost(testName, out, &result) && ExactVectorEqual(result, expected, testName);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestInt16Basic(aclrtStream stream) {
  const std::string testName = "int16_basic";
  std::vector<int64_t> shape = {2};
  std::vector<int16_t> selfHostData = {-1000, 2000};
  std::vector<int16_t> otherHostData = {2, -3};
  std::vector<int16_t> outHostData(GetShapeSize(shape), 0);

  TensorResource<int16_t> self;
  TensorResource<int16_t> other;
  TensorResource<int16_t> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_INT16)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_INT16)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_INT16)) &&
            RunAclnnMul(self.tensor, other.tensor, out.tensor, stream, testName);

  std::vector<int16_t> result;
  if (ok) {
    std::vector<int16_t> expected = ElementwiseMulExpected(selfHostData, otherHostData);
    ok = CopyDeviceToHost(testName, out, &result) && ExactVectorEqual(result, expected, testName);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestFloat32SpecialValues(aclrtStream stream) {
  const std::string testName = "float32_special_values";
  std::vector<int64_t> shape = {4, 2};
  std::vector<float> selfHostData = {NAN, INFINITY, -INFINITY, 1e-30f, 1e30f, -0.0f, 0.0f, -3.25f};
  std::vector<float> otherHostData = {1.0f, 2.0f, -1.0f, 1e-10f, 1e10f, INFINITY, -INFINITY, 4.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> other;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
            RunAclnnMul(self.tensor, other.tensor, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = ElementwiseMulExpected(selfHostData, otherHostData);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestDoubleSpecialValues(aclrtStream stream) {
  const std::string testName = "double_special_values";
  std::vector<int64_t> shape = {2, 4};
  std::vector<double> selfHostData = {NAN, INFINITY, -INFINITY, 1e-200, -1e200, 0.0, -0.0, 7.25};
  std::vector<double> otherHostData = {-1.0, -2.0, 3.0, 1e-100, 1e100, NAN, INFINITY, -8.0};
  std::vector<double> outHostData(GetShapeSize(shape), 0.0);

  TensorResource<double> self;
  TensorResource<double> other;
  TensorResource<double> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_DOUBLE)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_DOUBLE)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_DOUBLE)) &&
            RunAclnnMul(self.tensor, other.tensor, out.tensor, stream, testName);

  std::vector<double> result;
  if (ok) {
    std::vector<double> expected = ElementwiseMulExpected(selfHostData, otherHostData);
    ok = CopyDeviceToHost(testName, out, &result) &&
         FloatingVectorAlmostEqual(result, expected, testName, 1e-10, 1e-10);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestBroadcastRankDiffA(aclrtStream stream) {
  const std::string testName = "broadcast_rankdiff_a";
  std::vector<int64_t> selfShape = {2, 1, 3};
  std::vector<int64_t> otherShape = {1, 4, 3};
  std::vector<int64_t> outShape = {2, 4, 3};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f};
  std::vector<float> otherHostData = {
      1.0f, 0.5f, -1.0f,
      2.0f, 1.5f, -2.0f,
      3.0f, 2.5f, -3.0f,
      4.0f, 3.5f, -4.0f};
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> other;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, otherShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT)) &&
            RunAclnnMul(self.tensor, other.tensor, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = BroadcastMulExpected(selfHostData, selfShape, otherHostData, otherShape, outShape);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestBroadcastRankDiffB(aclrtStream stream) {
  const std::string testName = "broadcast_rankdiff_b";
  std::vector<int64_t> selfShape = {1, 3, 1};
  std::vector<int64_t> otherShape = {2, 3, 5};
  std::vector<int64_t> outShape = {2, 3, 5};
  std::vector<float> selfHostData = {1.0f, -2.0f, 3.0f};
  std::vector<float> otherHostData = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
      -1.0f, -2.0f, -3.0f, -4.0f, -5.0f,
      0.5f, 1.5f, 2.5f, 3.5f, 4.5f,
      6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
      -6.0f, -7.0f, -8.0f, -9.0f, -10.0f,
      -0.5f, -1.5f, -2.5f, -3.5f, -4.5f};
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> other;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, otherShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT)) &&
            RunAclnnMul(self.tensor, other.tensor, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = BroadcastMulExpected(selfHostData, selfShape, otherHostData, otherShape, outShape);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestBroadcastScalarLike(aclrtStream stream) {
  const std::string testName = "broadcast_scalar_like";
  std::vector<int64_t> selfShape = {2, 3, 4};
  std::vector<int64_t> otherShape = {1};
  std::vector<int64_t> outShape = {2, 3, 4};
  std::vector<float> selfHostData = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
      -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f,
      0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 1.75f, 2.0f};
  std::vector<float> otherHostData = {-2.0f};
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> other;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, otherShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT)) &&
            RunAclnnMul(self.tensor, other.tensor, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = BroadcastMulExpected(selfHostData, selfShape, otherHostData, otherShape, outShape);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestBroadcastInvalidExtra(aclrtStream stream) {
  (void)stream;
  const std::string testName = "broadcast_invalid_extra";
  std::vector<int64_t> selfShape = {2, 1, 3};
  std::vector<int64_t> otherShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2, 3};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> otherHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> other;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, otherShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
    if (!ok) {
      LOG_PRINT("  %s expected failure, but got SUCCESS.\n", testName.c_str());
    }
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
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> otherHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);

  TensorResource<float> other;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT));
  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(nullptr, other.tensor, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestNullptrOtherInvalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "nullptr_other_invalid";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT));
  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(self.tensor, nullptr, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestNullptrOutInvalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "nullptr_out_invalid";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> otherHostData = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

  TensorResource<float> self;
  TensorResource<float> other;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT));
  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, nullptr, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }
  self.Destroy();
  other.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestNullptrExecutorInvalid(aclrtStream stream) {
  const std::string testName = "nullptr_executor_invalid";
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> otherHostData = {4.0f, 3.0f, 2.0f, 1.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> other;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor);
    ok = CheckAclRet(testName, "aclnnMulGetWorkspaceSize", ret);
    if (ok) {
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ok = CheckAclRet(testName, "aclrtMalloc workspace", ret);
      }
      if (ok) {
        ret = aclnnMul(workspaceAddr, workspaceSize, nullptr, stream);
        ok = (ret != ACL_SUCCESS);
        if (!ok) {
          LOG_PRINT("  %s expected failure, but got SUCCESS.\n", testName.c_str());
        }
      }
      if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
      }
    }
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
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> otherHostData = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  std::vector<int32_t> outHostData(GetShapeSize(shape), 0);

  TensorResource<float> self;
  TensorResource<float> other;
  TensorResource<int32_t> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_INT32));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }
  self.Destroy();
  other.Destroy();
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
  TensorResource<int32_t> other;
  TensorResource<int32_t> out;
  int ret1 = self.Create(data, shape, static_cast<aclDataType>(999));
  int ret2 = other.Create(data, shape, static_cast<aclDataType>(999));
  int ret3 = out.Create(data, shape, static_cast<aclDataType>(999));
  bool ok = true;

  if (ret1 == ACL_SUCCESS && ret2 == ACL_SUCCESS && ret3 == ACL_SUCCESS) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }

  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestInplaceMulBoundary(aclrtStream stream) {
  const std::string testName = "inplace_mul_boundary";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHostData = {0.0f, -0.0f, 1e-20f, -1e20f, INFINITY, -INFINITY, 3.0f, -4.0f};
  std::vector<float> otherHostData = {2.0f, -3.0f, 1e20f, 1e-20f, 0.0f, -0.0f, -1.5f, 2.0f};

  TensorResource<float> self;
  TensorResource<float> other;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
            RunAclnnInplaceMul(self.tensor, other.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = ElementwiseMulExpected(selfHostData, otherHostData);
    ok = CopyDeviceToHost(testName, self, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }
  self.Destroy();
  other.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestInplaceMulsZeroScalar(aclrtStream stream) {
  const std::string testName = "inplace_muls_zero_scalar";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHostData = {-7.0f, -1.0f, 0.0f, 1.0f, 3.5f, 9.0f, -2.5f, 100.0f};
  float scalarValue = 0.0f;

  TensorResource<float> self;
  aclScalar* scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);
  bool ok = (scalar != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       RunAclnnInplaceMuls(self.tensor, scalar, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = MulsExpected(selfHostData, scalarValue);
    ok = CopyDeviceToHost(testName, self, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }
  if (scalar != nullptr) {
    aclDestroyScalar(scalar);
  }
  self.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestUnsupportedDtypeUint32Invalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "unsupported_dtype_uint32_invalid";
  std::vector<int64_t> shape = {2, 2};
  std::vector<uint32_t> selfHostData = {1U, 2U, 3U, 4U};
  std::vector<uint32_t> otherHostData = {4U, 3U, 2U, 1U};
  std::vector<uint32_t> outHostData(GetShapeSize(shape), 0U);

  TensorResource<uint32_t> self;
  TensorResource<uint32_t> other;
  TensorResource<uint32_t> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_UINT32)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_UINT32)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_UINT32));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
    if (!ok) {
      LOG_PRINT("  %s expected failure, but got SUCCESS.\n", testName.c_str());
    }
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestUnsupportedMixedDtypeInvalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "unsupported_mixed_dtype_invalid";
  std::vector<int64_t> shape = {2, 2};
  std::vector<int32_t> selfHostData = {1, 2, 3, 4};
  std::vector<float> otherHostData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);

  TensorResource<int32_t> self;
  TensorResource<float> other;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_INT32)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
    if (!ok) {
      LOG_PRINT("  %s expected failure, but got SUCCESS.\n", testName.c_str());
    }
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestBroadcastInvalidRankMismatch(aclrtStream stream) {
  (void)stream;
  const std::string testName = "broadcast_invalid_rank_mismatch";
  std::vector<int64_t> selfShape = {2, 1, 3};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {2, 4, 3};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> otherHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> other;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, otherShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestBroadcastInvalidHighdim(aclrtStream stream) {
  (void)stream;
  const std::string testName = "broadcast_invalid_highdim";
  std::vector<int64_t> selfShape = {2, 1, 3, 1};
  std::vector<int64_t> otherShape = {1, 4, 2, 5};
  std::vector<int64_t> outShape = {2, 4, 3, 5};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f};
  std::vector<float> otherHostData(40, 1.0f);
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> other;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, otherShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestBroadcastInvalidOutshape(aclrtStream stream) {
  (void)stream;
  const std::string testName = "broadcast_invalid_outshape";
  std::vector<int64_t> selfShape = {2, 1, 3};
  std::vector<int64_t> otherShape = {1, 4, 3};
  std::vector<int64_t> outShape = {2, 4, 1};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f};
  std::vector<float> otherHostData = {
      1.0f, 0.5f, -1.0f,
      2.0f, 1.5f, -2.0f,
      3.0f, 2.5f, -3.0f,
      4.0f, 3.5f, -4.0f};
  std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);

  TensorResource<float> self;
  TensorResource<float> other;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, selfShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, otherShape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, outShape, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestMixedDtypeAltShape1(aclrtStream stream) {
  const std::string testName = "mixed_dtype_alt_shape_1";
  std::vector<int64_t> shape = {1, 3, 2};
  std::vector<float> selfFloatData = {0.125f, -0.25f, 0.5f, -1.0f, 2.0f, -4.0f};
  std::vector<uint16_t> selfHostData = FloatsToHalfBits(selfFloatData);
  std::vector<float> otherHostData = {8.0f, -4.0f, 2.0f, -1.0f, 0.5f, -0.25f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);

  TensorResource<uint16_t> self;
  TensorResource<float> other;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT16)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
            RunAclnnMul(self.tensor, other.tensor, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = ElementwiseMulExpected(selfFloatData, otherHostData);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestMixedDtypeAltShape2(aclrtStream stream) {
  const std::string testName = "mixed_dtype_alt_shape_2";
  std::vector<int64_t> shape = {2, 1, 2, 2};
  std::vector<float> selfHostData = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f};
  std::vector<float> otherFloatData = {0.5f, 1.5f, -2.5f, -3.5f, 4.5f, 5.5f, -6.5f, -7.5f};
  std::vector<uint16_t> otherHostData = FloatsToHalfBits(otherFloatData);
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);

  TensorResource<float> self;
  TensorResource<uint16_t> other;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT16)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
            RunAclnnMul(self.tensor, other.tensor, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = ElementwiseMulExpected(selfHostData, otherFloatData);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestInplaceInvalidNullScalar(aclrtStream stream) {
  (void)stream;
  const std::string testName = "inplace_invalid_null_scalar";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};

  TensorResource<float> self;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT));
  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceMulsGetWorkspaceSize(self.tensor, nullptr, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }
  self.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestMulsAltShapeOrBoundary(aclrtStream stream) {
  const std::string testName = "muls_alt_shape_or_boundary";
  std::vector<int64_t> shape = {1, 3, 1, 2};
  std::vector<float> selfHostData = {1e-10f, -1e-10f, 1e10f, -1e10f, 3.0f, -3.0f};
  std::vector<float> outHostData(GetShapeSize(shape), 0.0f);
  float scalarValue = -0.125f;

  TensorResource<float> self;
  TensorResource<float> out;
  aclScalar* scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);
  bool ok = (scalar != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
       RunAclnnMuls(self.tensor, scalar, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = MulsExpected(selfHostData, scalarValue);
    ok = CopyDeviceToHost(testName, out, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }
  if (scalar != nullptr) {
    aclDestroyScalar(scalar);
  }
  self.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestInplaceMulsBoundaryCase(aclrtStream stream) {
  const std::string testName = "inplace_muls_boundary_case";
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHostData = {NAN, INFINITY, -INFINITY, 0.0f, -0.0f, 1e-20f, -1e20f, 8.0f};
  float scalarValue = 1.0f;

  TensorResource<float> self;
  aclScalar* scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);
  bool ok = (scalar != nullptr);
  if (!ok) {
    LOG_PRINT("  %s failed in aclCreateScalar.\n", testName.c_str());
  }
  ok = ok && CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
       RunAclnnInplaceMuls(self.tensor, scalar, stream, testName);

  std::vector<float> result;
  if (ok) {
    std::vector<float> expected = MulsExpected(selfHostData, scalarValue);
    ok = CopyDeviceToHost(testName, self, &result) && FloatVectorAlmostEqual(result, expected, testName);
  }
  if (scalar != nullptr) {
    aclDestroyScalar(scalar);
  }
  self.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestZeroDimShape(aclrtStream stream) {
  const std::string testName = "shape_zero_dim";
  std::vector<int64_t> shape = {};
  std::vector<float> selfHostData = {3.14f};
  std::vector<float> otherHostData = {2.0f};
  std::vector<float> outHostData(1, 0.0f);

  TensorResource<float> self;
  TensorResource<float> other;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(otherHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
            RunAclnnMul(self.tensor, other.tensor, out.tensor, stream, testName);

  std::vector<float> result;
  if (ok) {
    ok = CopyDeviceToHost(testName, out, &result) && AlmostEqual(6.28, result[0]);
  }
  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestExtremeHighRank(aclrtStream stream) {
  const std::string testName = "shape_extreme_high_rank";
  std::vector<int64_t> shape = {1, 1, 1, 1, 2, 1, 1, 2};
  std::vector<float> selfHostData = {1.5f, -2.5f, 3.5f, -4.5f};
  std::vector<float> outHostData(4, 0.0f);

  TensorResource<float> self;
  TensorResource<float> other;
  TensorResource<float> out;
  bool ok = CheckAclRet(testName, "create self", self.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(selfHostData, shape, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", out.Create(outHostData, shape, ACL_FLOAT)) &&
            RunAclnnMul(self.tensor, other.tensor, out.tensor, stream, testName);

  self.Destroy();
  other.Destroy();
  out.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

bool TestShapeMismatchInvalid(aclrtStream stream) {
  (void)stream;
  const std::string testName = "shape_mismatch_creation_invalid";
  std::vector<int64_t> shape1 = {2, 3};
  std::vector<int64_t> shape2 = {2, 4};
  std::vector<float> data1(6, 1.0f);
  std::vector<float> data2(8, 1.0f);
  std::vector<float> out(6, 0.0f);

  TensorResource<float> self;
  TensorResource<float> other;
  TensorResource<float> outRes;
  bool ok = CheckAclRet(testName, "create self", self.Create(data1, shape1, ACL_FLOAT)) &&
            CheckAclRet(testName, "create other", other.Create(data2, shape2, ACL_FLOAT)) &&
            CheckAclRet(testName, "create out", outRes.Create(out, shape1, ACL_FLOAT));

  if (ok) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, outRes.tensor, &workspaceSize, &executor);
    ok = (ret != ACL_SUCCESS);
  }
  self.Destroy();
  other.Destroy();
  outRes.Destroy();
  PrintCaseResult(testName, ok);
  return ok;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  int failedCount = 0;
  failedCount += TestFloat32Basic(stream) ? 0 : 1;
  failedCount += TestFloat32NegZero(stream) ? 0 : 1;
  failedCount += TestFloat32Muls(stream) ? 0 : 1;
  failedCount += TestInt32Basic(stream) ? 0 : 1;
  failedCount += TestBroadcastBasic(stream) ? 0 : 1;
  failedCount += TestBroadcastInvalid(stream) ? 0 : 1;
  failedCount += TestInplaceMulBasic(stream) ? 0 : 1;
  failedCount += TestInplaceMulsBasic(stream) ? 0 : 1;
  failedCount += TestDoubleExtreme(stream) ? 0 : 1;
  failedCount += TestFloat16Float32Mixed(stream) ? 0 : 1;
  failedCount += TestFloat32Float16Mixed(stream) ? 0 : 1;
  failedCount += TestDoubleBasic(stream) ? 0 : 1;
  failedCount += TestBoolBasic(stream) ? 0 : 1;
  failedCount += TestInt8Basic(stream) ? 0 : 1;
  failedCount += TestUint8Basic(stream) ? 0 : 1;
  failedCount += TestNullptrInvalid(stream) ? 0 : 1;
  failedCount += TestBroadcastHighdim(stream) ? 0 : 1;
  failedCount += TestInt64Basic(stream) ? 0 : 1;
  failedCount += TestInt16Basic(stream) ? 0 : 1;
  failedCount += TestFloat32SpecialValues(stream) ? 0 : 1;
  failedCount += TestDoubleSpecialValues(stream) ? 0 : 1;
  failedCount += TestBroadcastRankDiffA(stream) ? 0 : 1;
  failedCount += TestBroadcastRankDiffB(stream) ? 0 : 1;
  failedCount += TestBroadcastScalarLike(stream) ? 0 : 1;
  failedCount += TestBroadcastInvalidExtra(stream) ? 0 : 1;
  failedCount += TestNullptrSelfInvalid(stream) ? 0 : 1;
  failedCount += TestNullptrOtherInvalid(stream) ? 0 : 1;
  failedCount += TestNullptrOutInvalid(stream) ? 0 : 1;
  failedCount += TestNullptrExecutorInvalid(stream) ? 0 : 1;
  failedCount += TestOutDtypeMismatchInvalid(stream) ? 0 : 1;
  failedCount += TestUnsupportedDtypeInvalid(stream) ? 0 : 1;
  failedCount += TestInplaceMulBoundary(stream) ? 0 : 1;
  failedCount += TestInplaceMulsZeroScalar(stream) ? 0 : 1;
  failedCount += TestUnsupportedDtypeUint32Invalid(stream) ? 0 : 1;
  failedCount += TestUnsupportedMixedDtypeInvalid(stream) ? 0 : 1;
  failedCount += TestBroadcastInvalidRankMismatch(stream) ? 0 : 1;
  failedCount += TestBroadcastInvalidHighdim(stream) ? 0 : 1;
  failedCount += TestBroadcastInvalidOutshape(stream) ? 0 : 1;
  failedCount += TestMixedDtypeAltShape1(stream) ? 0 : 1;
  failedCount += TestMixedDtypeAltShape2(stream) ? 0 : 1;
  failedCount += TestInplaceInvalidNullScalar(stream) ? 0 : 1;
  failedCount += TestMulsAltShapeOrBoundary(stream) ? 0 : 1;
  failedCount += TestInplaceMulsBoundaryCase(stream) ? 0 : 1;
  failedCount += TestZeroDimShape(stream) ? 0 : 1;
  failedCount += TestExtremeHighRank(stream) ? 0 : 1;
  failedCount += TestShapeMismatchInvalid(stream) ? 0 : 1;

  const int totalCount = 46;
  LOG_PRINT("==================================================\n");
  LOG_PRINT("Mul test summary: total=%d, passed=%d, failed=%d\n", totalCount, totalCount - failedCount, failedCount);

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return failedCount == 0 ? 0 : 1;
}
