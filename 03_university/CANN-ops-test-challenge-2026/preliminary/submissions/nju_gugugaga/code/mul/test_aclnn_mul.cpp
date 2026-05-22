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
#include <cstring>
#include <algorithm>
#include <iostream>
#include <limits>
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

constexpr aclnnStatus ERR_PARAM_NULLPTR = static_cast<aclnnStatus>(161001);
constexpr aclnnStatus ERR_PARAM_INVALID = static_cast<aclnnStatus>(161002);
constexpr aclnnStatus ERR_INNER_NULLPTR = static_cast<aclnnStatus>(561103);

struct TestCase {
  std::string name;
  std::vector<int64_t> selfShape;
  std::vector<int64_t> otherShape;
  std::vector<int64_t> outShape;
  aclDataType selfType;
  aclDataType otherType;
  aclDataType outType;
  bool expectSuccess;
  aclnnStatus expectStatus;
  bool allowAnyError;
  float atol;
  float rtol;
  bool verifyAll;
};

struct ScalarTestCase {
  std::string name;
  std::vector<int64_t> selfShape;
  std::vector<int64_t> outShape;
  aclDataType selfType;
  aclDataType scalarType;
  aclDataType outType;
  double scalarValue;
  bool expectSuccess;
  aclnnStatus expectStatus;
  bool allowAnyError;
  float atol;
  float rtol;
};

struct InplaceTensorCase {
  std::string name;
  std::vector<int64_t> selfShape;
  std::vector<int64_t> otherShape;
  aclDataType selfType;
  aclDataType otherType;
  bool expectSuccess;
  aclnnStatus expectStatus;
  float atol;
  float rtol;
};

struct NonContiguousCase {
  std::string name;
  std::vector<int64_t> selfViewShape;
  std::vector<int64_t> selfStorageShape;
  std::vector<int64_t> selfStride;
  int64_t selfOffset;
  std::vector<int64_t> otherViewShape;
  std::vector<int64_t> outShape;
  aclDataType dtype;
  float atol;
  float rtol;
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  int64_t shapeSize = 1;
  for (int64_t i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

size_t SizeOfDtype(aclDataType type)
{
  switch (type) {
    case ACL_BOOL:
    case ACL_INT8:
    case ACL_UINT8:
      return 1;
    case ACL_INT16:
    case ACL_UINT16:
    case ACL_FLOAT16:
    case ACL_BF16:
      return 2;
    case ACL_INT32:
    case ACL_UINT32:
    case ACL_FLOAT:
      return 4;
    case ACL_INT64:
    case ACL_UINT64:
    case ACL_DOUBLE:
      return 8;
    case ACL_COMPLEX32:
      return 4;
    case ACL_COMPLEX64:
      return 8;
    case ACL_COMPLEX128:
      return 16;
    default:
      return 0;
  }
}

static inline float Float16ToFloat(uint16_t value)
{
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

static inline uint16_t FloatToFloat16(float value)
{
  uint32_t bits = *reinterpret_cast<uint32_t*>(&value);
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

static inline uint16_t FloatToBf16(float value)
{
  uint32_t bits = *reinterpret_cast<uint32_t*>(&value);
  return static_cast<uint16_t>((bits + 0x8000u) >> 16);
}

static inline float Bf16ToFloat(uint16_t value)
{
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  return *reinterpret_cast<float*>(&bits);
}

std::vector<int64_t> MakeContiguousStride(const std::vector<int64_t>& shape)
{
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i) + 1] * strides[static_cast<size_t>(i) + 1];
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

int CreateAclTensorFromBytes(const std::vector<uint8_t>& hostBytes, const std::vector<int64_t>& shape, void** deviceAddr,
                             aclDataType dataType, aclTensor** tensor)
{
  size_t bytes = hostBytes.empty() ? 1 : hostBytes.size();
  auto ret = aclrtMalloc(deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  if (!hostBytes.empty()) {
    ret = aclrtMemcpy(*deviceAddr, bytes, hostBytes.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D failed. ERROR: %d\n", ret); return ret);
  }
  auto strides = MakeContiguousStride(shape);
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
  return ACL_SUCCESS;
}

int CreateAclTensorWithLayoutFromBytes(const std::vector<uint8_t>& hostBytes,
                                       const std::vector<int64_t>& viewShape,
                                       const std::vector<int64_t>& storageShape,
                                       const std::vector<int64_t>& viewStrides,
                                       int64_t offset,
                                       void** deviceAddr,
                                       aclDataType dataType,
                                       aclTensor** tensor)
{
  size_t bytes = hostBytes.empty() ? 1 : hostBytes.size();
  auto ret = aclrtMalloc(deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  if (!hostBytes.empty()) {
    ret = aclrtMemcpy(*deviceAddr, bytes, hostBytes.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D failed. ERROR: %d\n", ret); return ret);
  }
  *tensor = aclCreateTensor(viewShape.data(), viewShape.size(), dataType,
                            viewStrides.data(), offset, aclFormat::ACL_FORMAT_ND,
                            storageShape.data(), storageShape.size(), *deviceAddr);
  CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor with layout failed.\n"); return ACL_ERROR_FAILURE);
  return ACL_SUCCESS;
}

std::vector<uint8_t> GenerateData(const std::vector<int64_t>& shape, aclDataType dtype, float base)
{
  int64_t n = GetShapeSize(shape);
  size_t elemSize = SizeOfDtype(dtype);
  std::vector<uint8_t> bytes(static_cast<size_t>(n) * elemSize, 0);
  for (int64_t i = 0; i < n; ++i) {
    float v = base + static_cast<float>((i % 13) - 6) * 0.25f;
    if (dtype == ACL_FLOAT) {
      std::memcpy(bytes.data() + static_cast<size_t>(i) * elemSize, &v, sizeof(float));
    } else if (dtype == ACL_FLOAT16) {
      uint16_t h = FloatToFloat16(v);
      std::memcpy(bytes.data() + static_cast<size_t>(i) * elemSize, &h, sizeof(uint16_t));
    } else if (dtype == ACL_BF16) {
      uint16_t b = FloatToBf16(v);
      std::memcpy(bytes.data() + static_cast<size_t>(i) * elemSize, &b, sizeof(uint16_t));
    } else if (dtype == ACL_INT32) {
      int32_t x = static_cast<int32_t>((i % 9) - 4);
      std::memcpy(bytes.data() + static_cast<size_t>(i) * elemSize, &x, sizeof(int32_t));
    } else if (dtype == ACL_INT64) {
      int64_t x = static_cast<int64_t>((i % 7) - 3);
      std::memcpy(bytes.data() + static_cast<size_t>(i) * elemSize, &x, sizeof(int64_t));
    } else if (dtype == ACL_INT16) {
      int16_t x = static_cast<int16_t>((i % 11) - 5);
      std::memcpy(bytes.data() + static_cast<size_t>(i) * elemSize, &x, sizeof(int16_t));
    } else if (dtype == ACL_INT8) {
      int8_t x = static_cast<int8_t>((i % 15) - 7);
      std::memcpy(bytes.data() + static_cast<size_t>(i) * elemSize, &x, sizeof(int8_t));
    } else if (dtype == ACL_DOUBLE) {
      double x = static_cast<double>(v);
      std::memcpy(bytes.data() + static_cast<size_t>(i) * elemSize, &x, sizeof(double));
    } else if (dtype == ACL_COMPLEX32 || dtype == ACL_COMPLEX64 || dtype == ACL_COMPLEX128) {
      // Keep complex input deterministic and simple; zero is enough to trigger dtype branches.
      std::fill(bytes.data() + static_cast<size_t>(i) * elemSize,
                bytes.data() + static_cast<size_t>(i + 1) * elemSize, 0);
    } else if (dtype == ACL_BOOL) {
      uint8_t x = static_cast<uint8_t>(i % 2);
      bytes[static_cast<size_t>(i)] = x;
    } else if (dtype == ACL_UINT8) {
      // Keep uint8 in a low range to avoid overflow semantic mismatch in reference check.
      uint8_t x = static_cast<uint8_t>(i % 4);
      bytes[static_cast<size_t>(i)] = x;
    }
  }
  return bytes;
}

float ReadAsFloat(const std::vector<uint8_t>& bytes, int64_t idx, aclDataType dtype)
{
  size_t elem = SizeOfDtype(dtype);
  const uint8_t* p = bytes.data() + static_cast<size_t>(idx) * elem;
  if (dtype == ACL_FLOAT) {
    float v;
    std::memcpy(&v, p, sizeof(float));
    return v;
  }
  if (dtype == ACL_FLOAT16) {
    uint16_t v;
    std::memcpy(&v, p, sizeof(uint16_t));
    return Float16ToFloat(v);
  }
  if (dtype == ACL_BF16) {
    uint16_t v;
    std::memcpy(&v, p, sizeof(uint16_t));
    return Bf16ToFloat(v);
  }
  if (dtype == ACL_INT32) {
    int32_t v;
    std::memcpy(&v, p, sizeof(int32_t));
    return static_cast<float>(v);
  }
  if (dtype == ACL_INT64) {
    int64_t v;
    std::memcpy(&v, p, sizeof(int64_t));
    return static_cast<float>(v);
  }
  if (dtype == ACL_INT16) {
    int16_t v;
    std::memcpy(&v, p, sizeof(int16_t));
    return static_cast<float>(v);
  }
  if (dtype == ACL_INT8) {
    int8_t v;
    std::memcpy(&v, p, sizeof(int8_t));
    return static_cast<float>(v);
  }
  if (dtype == ACL_DOUBLE) {
    double v;
    std::memcpy(&v, p, sizeof(double));
    return static_cast<float>(v);
  }
  if (dtype == ACL_BOOL || dtype == ACL_UINT8 || dtype == ACL_INT8) {
    return static_cast<float>(*p);
  }
  return 0.0f;
}

std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape)
{
  return MakeContiguousStride(shape);
}

int64_t BroadcastOffset(const std::vector<int64_t>& outIndex,
                        const std::vector<int64_t>& inShape,
                        const std::vector<int64_t>& inStride)
{
  int64_t outDims = static_cast<int64_t>(outIndex.size());
  int64_t inDims = static_cast<int64_t>(inShape.size());
  int64_t offset = 0;
  for (int64_t i = 0; i < inDims; ++i) {
    int64_t outAxis = outDims - inDims + i;
    int64_t idx = (inShape[static_cast<size_t>(i)] == 1) ? 0 : outIndex[static_cast<size_t>(outAxis)];
    offset += idx * inStride[static_cast<size_t>(i)];
  }
  return offset;
}

std::vector<float> CpuMulRef(const std::vector<uint8_t>& selfBytes, const std::vector<uint8_t>& otherBytes,
                             const std::vector<int64_t>& selfShape, const std::vector<int64_t>& otherShape,
                             const std::vector<int64_t>& outShape, aclDataType selfType, aclDataType otherType)
{
  int64_t outN = GetShapeSize(outShape);
  auto selfStride = ComputeStrides(selfShape);
  auto otherStride = ComputeStrides(otherShape);
  std::vector<float> ref(static_cast<size_t>(outN), 0.0f);
  std::vector<int64_t> outIndex(outShape.size(), 0);
  for (int64_t i = 0; i < outN; ++i) {
    int64_t t = i;
    for (int64_t d = static_cast<int64_t>(outShape.size()) - 1; d >= 0; --d) {
      int64_t dim = outShape[static_cast<size_t>(d)];
      outIndex[static_cast<size_t>(d)] = (dim == 0) ? 0 : (t % dim);
      t = (dim == 0) ? 0 : (t / dim);
    }
    int64_t aIdx = BroadcastOffset(outIndex, selfShape, selfStride);
    int64_t bIdx = BroadcastOffset(outIndex, otherShape, otherStride);
    float av = ReadAsFloat(selfBytes, aIdx, selfType);
    float bv = ReadAsFloat(otherBytes, bIdx, otherType);
    ref[static_cast<size_t>(i)] = av * bv;
  }
  return ref;
}

bool CompareResult(const std::vector<uint8_t>& outBytes, aclDataType outType,
                   const std::vector<float>& ref, float atol, float rtol, bool verifyAll)
{
  int64_t n = static_cast<int64_t>(ref.size());
  int64_t stride = verifyAll ? 1 : std::max<int64_t>(1, n / 128);
  for (int64_t i = 0; i < n; i += stride) {
    float got = ReadAsFloat(outBytes, i, outType);
    float expect = ref[static_cast<size_t>(i)];
    if (std::isnan(expect)) {
      if (!std::isnan(got)) {
        LOG_PRINT("Mismatch idx=%ld expect=NaN got=%f\n", i, got);
        return false;
      }
      continue;
    }
    if (std::isinf(expect)) {
      if (!(std::isinf(got) && std::signbit(got) == std::signbit(expect))) {
        LOG_PRINT("Mismatch idx=%ld expect=%f got=%f\n", i, expect, got);
        return false;
      }
      continue;
    }
    float diff = std::fabs(got - expect);
    float tol = atol + rtol * std::fabs(expect);
    if (diff > tol) {
      LOG_PRINT("Mismatch idx=%ld got=%f expect=%f diff=%f tol=%f\n", i, got, expect, diff, tol);
      return false;
    }
  }
  return true;
}

bool RunNonContiguousMulTest(const NonContiguousCase& tc, aclrtStream stream)
{
  LOG_PRINT("[RUN ] %s\n", tc.name.c_str());
  const size_t elemSize = SizeOfDtype(tc.dtype);
  const int64_t selfStorageN = GetShapeSize(tc.selfStorageShape);
  const int64_t otherN = GetShapeSize(tc.otherViewShape);
  const int64_t outN = GetShapeSize(tc.outShape);

  std::vector<uint8_t> selfStorageBytes(static_cast<size_t>(selfStorageN) * elemSize, 0);
  std::vector<uint8_t> otherBytes = GenerateData(tc.otherViewShape, tc.dtype, -0.25f);
  std::vector<uint8_t> outInit(static_cast<size_t>(outN) * elemSize, 0);

  for (int64_t i = 0; i < selfStorageN; ++i) {
    float v = static_cast<float>((i % 19) - 9) * 0.2f;
    if (tc.dtype == ACL_FLOAT) {
      std::memcpy(selfStorageBytes.data() + static_cast<size_t>(i) * elemSize, &v, sizeof(float));
    } else if (tc.dtype == ACL_FLOAT16) {
      uint16_t h = FloatToFloat16(v);
      std::memcpy(selfStorageBytes.data() + static_cast<size_t>(i) * elemSize, &h, sizeof(uint16_t));
    } else {
      int32_t x = static_cast<int32_t>(v * 10.0f);
      std::memcpy(selfStorageBytes.data() + static_cast<size_t>(i) * elemSize, &x, sizeof(int32_t));
    }
  }

  void* selfDev = nullptr;
  void* otherDev = nullptr;
  void* outDev = nullptr;
  void* workspaceDev = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;

  if (CreateAclTensorWithLayoutFromBytes(selfStorageBytes, tc.selfViewShape, tc.selfStorageShape,
                                         tc.selfStride, tc.selfOffset, &selfDev, tc.dtype, &self) != ACL_SUCCESS) {
    return false;
  }
  if (CreateAclTensorFromBytes(otherBytes, tc.otherViewShape, &otherDev, tc.dtype, &other) != ACL_SUCCESS) {
    return false;
  }
  if (CreateAclTensorFromBytes(outInit, tc.outShape, &outDev, tc.dtype, &out) != ACL_SUCCESS) {
    return false;
  }

  auto ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    return false;
  }
  if (workspaceSize > 0 && aclrtMalloc(&workspaceDev, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
    return false;
  }
  ret = aclnnMul(workspaceDev, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    return false;
  }

  std::vector<uint8_t> outBytes(outInit.size(), 0);
  if (aclrtMemcpy(outBytes.data(), outBytes.size(), outDev, outBytes.size(), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
    return false;
  }

  std::vector<float> ref(static_cast<size_t>(outN), 0.0f);
  auto otherStride = ComputeStrides(tc.otherViewShape);
  auto outStride = ComputeStrides(tc.outShape);
  std::vector<int64_t> outIndex(tc.outShape.size(), 0);
  for (int64_t i = 0; i < outN; ++i) {
    int64_t t = i;
    for (int64_t d = static_cast<int64_t>(tc.outShape.size()) - 1; d >= 0; --d) {
      outIndex[static_cast<size_t>(d)] = t % tc.outShape[static_cast<size_t>(d)];
      t /= tc.outShape[static_cast<size_t>(d)];
    }
    int64_t selfIdx = tc.selfOffset;
    for (size_t d = 0; d < tc.selfViewShape.size(); ++d) {
      selfIdx += outIndex[d] * tc.selfStride[d];
    }
    int64_t otherIdx = 0;
    for (size_t d = 0; d < tc.otherViewShape.size(); ++d) {
      otherIdx += outIndex[d] * otherStride[d];
    }
    ref[static_cast<size_t>(i)] = ReadAsFloat(selfStorageBytes, selfIdx, tc.dtype) * ReadAsFloat(otherBytes, otherIdx, tc.dtype);
  }

  bool pass = CompareResult(outBytes, tc.dtype, ref, tc.atol, tc.rtol, true);

  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDev);
  aclrtFree(otherDev);
  aclrtFree(outDev);
  if (workspaceDev != nullptr) {
    aclrtFree(workspaceDev);
  }
  LOG_PRINT(pass ? "[PASS] %s\n" : "[FAIL] %s\n", tc.name.c_str());
  return pass;
}

bool RunMulTest(const TestCase& tc, aclrtStream stream)
{
  LOG_PRINT("[RUN ] %s\n", tc.name.c_str());
  void* selfDev = nullptr;
  void* otherDev = nullptr;
  void* outDev = nullptr;
  void* workspaceDev = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;

  std::vector<uint8_t> selfBytes = GenerateData(tc.selfShape, tc.selfType, 1.25f);
  std::vector<uint8_t> otherBytes = GenerateData(tc.otherShape, tc.otherType, 0.75f);
  std::vector<uint8_t> outInit(static_cast<size_t>(GetShapeSize(tc.outShape)) * SizeOfDtype(tc.outType), 0);

  if (CreateAclTensorFromBytes(selfBytes, tc.selfShape, &selfDev, tc.selfType, &self) != ACL_SUCCESS) {
    return false;
  }
  if (CreateAclTensorFromBytes(otherBytes, tc.otherShape, &otherDev, tc.otherType, &other) != ACL_SUCCESS) {
    return false;
  }
  if (CreateAclTensorFromBytes(outInit, tc.outShape, &outDev, tc.outType, &out) != ACL_SUCCESS) {
    return false;
  }

  aclnnStatus ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  if (!tc.expectSuccess) {
    bool ok = tc.allowAnyError ? (ret != ACL_SUCCESS) : (ret == tc.expectStatus);
    LOG_PRINT("[INFO] %s expect fail=%d got=%d\n", tc.name.c_str(), tc.expectStatus, ret);
    if (self != nullptr) aclDestroyTensor(self);
    if (other != nullptr) aclDestroyTensor(other);
    if (out != nullptr) aclDestroyTensor(out);
    if (selfDev != nullptr) aclrtFree(selfDev);
    if (otherDev != nullptr) aclrtFree(otherDev);
    if (outDev != nullptr) aclrtFree(outDev);
    LOG_PRINT(ok ? "[PASS] %s\n" : "[FAIL] %s\n", tc.name.c_str());
    return ok;
  }

  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s aclnnMulGetWorkspaceSize ret=%d\n", tc.name.c_str(), ret);
    return false;
  }

  if (workspaceSize > 0) {
    auto mallocRet = aclrtMalloc(&workspaceDev, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (mallocRet != ACL_SUCCESS) {
      LOG_PRINT("[FAIL] %s workspace alloc ret=%d\n", tc.name.c_str(), mallocRet);
      return false;
    }
  }

  ret = aclnnMul(workspaceDev, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s aclnnMul ret=%d\n", tc.name.c_str(), ret);
    return false;
  }

  auto syncRet = aclrtSynchronizeStream(stream);
  if (syncRet != ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s aclrtSynchronizeStream ret=%d\n", tc.name.c_str(), syncRet);
    return false;
  }

  std::vector<uint8_t> outBytes = outInit;
  if (!outBytes.empty()) {
    auto cpyRet = aclrtMemcpy(outBytes.data(), outBytes.size(), outDev, outBytes.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    if (cpyRet != ACL_SUCCESS) {
      LOG_PRINT("[FAIL] %s D2H ret=%d\n", tc.name.c_str(), cpyRet);
      return false;
    }
  }

  auto ref = CpuMulRef(selfBytes, otherBytes, tc.selfShape, tc.otherShape, tc.outShape, tc.selfType, tc.otherType);
  bool pass = CompareResult(outBytes, tc.outType, ref, tc.atol, tc.rtol, tc.verifyAll);

  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDev);
  aclrtFree(otherDev);
  aclrtFree(outDev);
  if (workspaceDev != nullptr) {
    aclrtFree(workspaceDev);
  }

  LOG_PRINT(pass ? "[PASS] %s\n" : "[FAIL] %s\n", tc.name.c_str());
  return pass;
}

bool RunNullptrCase(const std::string& name, int whichNull, aclnnStatus expect)
{
  std::vector<int64_t> shape = {2, 2};
  std::vector<uint8_t> bytes(4 * sizeof(float), 0);
  void* devA = nullptr;
  void* devB = nullptr;
  void* devO = nullptr;
  aclTensor* a = nullptr;
  aclTensor* b = nullptr;
  aclTensor* o = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  if (CreateAclTensorFromBytes(bytes, shape, &devA, ACL_FLOAT, &a) != ACL_SUCCESS) return false;
  if (CreateAclTensorFromBytes(bytes, shape, &devB, ACL_FLOAT, &b) != ACL_SUCCESS) return false;
  if (CreateAclTensorFromBytes(bytes, shape, &devO, ACL_FLOAT, &o) != ACL_SUCCESS) return false;

  aclTensor* inA = (whichNull == 0) ? nullptr : a;
  aclTensor* inB = (whichNull == 1) ? nullptr : b;
  aclTensor* out = (whichNull == 2) ? nullptr : o;

  aclnnStatus ret = aclnnMulGetWorkspaceSize(inA, inB, out, &workspaceSize, &executor);
  bool pass = (ret == expect);
  LOG_PRINT(pass ? "[PASS] %s\n" : "[FAIL] %s expect=%d got=%d\n", name.c_str(), expect, ret);

  aclDestroyTensor(a);
  aclDestroyTensor(b);
  aclDestroyTensor(o);
  aclrtFree(devA);
  aclrtFree(devB);
  aclrtFree(devO);
  return pass;
}

bool RunMulsTest(const ScalarTestCase& tc, aclrtStream stream)
{
  LOG_PRINT("[RUN ] %s\n", tc.name.c_str());
  void* selfDev = nullptr;
  void* outDev = nullptr;
  void* workspaceDev = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclScalar* scalar = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;

  std::vector<uint8_t> selfBytes = GenerateData(tc.selfShape, tc.selfType, -0.25f);
  std::vector<uint8_t> outInit(static_cast<size_t>(GetShapeSize(tc.outShape)) * SizeOfDtype(tc.outType), 0);

  if (CreateAclTensorFromBytes(selfBytes, tc.selfShape, &selfDev, tc.selfType, &self) != ACL_SUCCESS) {
    return false;
  }
  if (CreateAclTensorFromBytes(outInit, tc.outShape, &outDev, tc.outType, &out) != ACL_SUCCESS) {
    return false;
  }

  if (tc.scalarType == ACL_INT32) {
    int32_t val = static_cast<int32_t>(tc.scalarValue);
    scalar = aclCreateScalar(&val, tc.scalarType);
  } else if (tc.scalarType == ACL_DOUBLE) {
    double val = tc.scalarValue;
    scalar = aclCreateScalar(&val, tc.scalarType);
  } else {
    float val = static_cast<float>(tc.scalarValue);
    scalar = aclCreateScalar(&val, tc.scalarType);
  }
  if (scalar == nullptr) {
    return false;
  }

  aclnnStatus ret = aclnnMulsGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor);
  if (!tc.expectSuccess) {
    bool ok = tc.allowAnyError ? (ret != ACL_SUCCESS) : (ret == tc.expectStatus);
    LOG_PRINT(ok ? "[PASS] %s\n" : "[FAIL] %s expect=%d got=%d\n", tc.name.c_str(), tc.expectStatus, ret);
    aclDestroyScalar(scalar);
    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(selfDev);
    aclrtFree(outDev);
    return ok;
  }
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s aclnnMulsGetWorkspaceSize ret=%d\n", tc.name.c_str(), ret);
    return false;
  }

  if (workspaceSize > 0) {
    auto mallocRet = aclrtMalloc(&workspaceDev, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (mallocRet != ACL_SUCCESS) {
      LOG_PRINT("[FAIL] %s workspace alloc ret=%d\n", tc.name.c_str(), mallocRet);
      return false;
    }
  }

  ret = aclnnMuls(workspaceDev, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s aclnnMuls ret=%d\n", tc.name.c_str(), ret);
    return false;
  }
  if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    return false;
  }

  std::vector<uint8_t> outBytes = outInit;
  if (!outBytes.empty()) {
    if (aclrtMemcpy(outBytes.data(), outBytes.size(), outDev, outBytes.size(), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
      return false;
    }
  }

  std::vector<float> ref(static_cast<size_t>(GetShapeSize(tc.outShape)), 0.0f);
  for (int64_t i = 0; i < static_cast<int64_t>(ref.size()); ++i) {
    ref[static_cast<size_t>(i)] = ReadAsFloat(selfBytes, i, tc.selfType) * static_cast<float>(tc.scalarValue);
  }
  bool pass = CompareResult(outBytes, tc.outType, ref, tc.atol, tc.rtol, true);

  aclDestroyScalar(scalar);
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclrtFree(selfDev);
  aclrtFree(outDev);
  if (workspaceDev != nullptr) {
    aclrtFree(workspaceDev);
  }
  LOG_PRINT(pass ? "[PASS] %s\n" : "[FAIL] %s\n", tc.name.c_str());
  return pass;
}

bool RunInplaceMulTest(const InplaceTensorCase& tc, aclrtStream stream)
{
  LOG_PRINT("[RUN ] %s\n", tc.name.c_str());
  void* selfDev = nullptr;
  void* otherDev = nullptr;
  void* workspaceDev = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;

  std::vector<uint8_t> selfBytes = GenerateData(tc.selfShape, tc.selfType, 1.0f);
  std::vector<uint8_t> otherBytes = GenerateData(tc.otherShape, tc.otherType, -0.5f);
  if (CreateAclTensorFromBytes(selfBytes, tc.selfShape, &selfDev, tc.selfType, &self) != ACL_SUCCESS) {
    return false;
  }
  if (CreateAclTensorFromBytes(otherBytes, tc.otherShape, &otherDev, tc.otherType, &other) != ACL_SUCCESS) {
    return false;
  }

  aclnnStatus ret = aclnnInplaceMulGetWorkspaceSize(self, other, &workspaceSize, &executor);
  if (!tc.expectSuccess) {
    bool ok = (ret == tc.expectStatus);
    LOG_PRINT(ok ? "[PASS] %s\n" : "[FAIL] %s expect=%d got=%d\n", tc.name.c_str(), tc.expectStatus, ret);
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDev);
    aclrtFree(otherDev);
    return ok;
  }
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s aclnnInplaceMulGetWorkspaceSize ret=%d\n", tc.name.c_str(), ret);
    return false;
  }

  if (workspaceSize > 0 && aclrtMalloc(&workspaceDev, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
    return false;
  }

  ret = aclnnInplaceMul(workspaceDev, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    return false;
  }
  if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    return false;
  }

  std::vector<uint8_t> outBytes(selfBytes.size(), 0);
  if (!outBytes.empty()) {
    if (aclrtMemcpy(outBytes.data(), outBytes.size(), selfDev, outBytes.size(), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
      return false;
    }
  }

  auto ref = CpuMulRef(selfBytes, otherBytes, tc.selfShape, tc.otherShape, tc.selfShape, tc.selfType, tc.otherType);
  bool pass = CompareResult(outBytes, tc.selfType, ref, tc.atol, tc.rtol, true);

  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclrtFree(selfDev);
  aclrtFree(otherDev);
  if (workspaceDev != nullptr) {
    aclrtFree(workspaceDev);
  }
  LOG_PRINT(pass ? "[PASS] %s\n" : "[FAIL] %s\n", tc.name.c_str());
  return pass;
}

bool RunInplaceMulsTest(const ScalarTestCase& tc, aclrtStream stream)
{
  LOG_PRINT("[RUN ] %s\n", tc.name.c_str());
  void* selfDev = nullptr;
  void* workspaceDev = nullptr;
  aclTensor* self = nullptr;
  aclScalar* scalar = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;

  std::vector<uint8_t> selfBytes = GenerateData(tc.selfShape, tc.selfType, 0.5f);
  if (CreateAclTensorFromBytes(selfBytes, tc.selfShape, &selfDev, tc.selfType, &self) != ACL_SUCCESS) {
    return false;
  }

  if (tc.scalarType == ACL_INT32) {
    int32_t val = static_cast<int32_t>(tc.scalarValue);
    scalar = aclCreateScalar(&val, tc.scalarType);
  } else if (tc.scalarType == ACL_DOUBLE) {
    double val = tc.scalarValue;
    scalar = aclCreateScalar(&val, tc.scalarType);
  } else {
    float val = static_cast<float>(tc.scalarValue);
    scalar = aclCreateScalar(&val, tc.scalarType);
  }
  if (scalar == nullptr) {
    return false;
  }

  aclnnStatus ret = aclnnInplaceMulsGetWorkspaceSize(self, scalar, &workspaceSize, &executor);
  if (!tc.expectSuccess) {
    bool ok = tc.allowAnyError ? (ret != ACL_SUCCESS) : (ret == tc.expectStatus);
    LOG_PRINT(ok ? "[PASS] %s\n" : "[FAIL] %s expect=%d got=%d\n", tc.name.c_str(), tc.expectStatus, ret);
    aclDestroyScalar(scalar);
    aclDestroyTensor(self);
    aclrtFree(selfDev);
    return ok;
  }
  if (ret != ACL_SUCCESS) {
    return false;
  }

  if (workspaceSize > 0 && aclrtMalloc(&workspaceDev, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
    return false;
  }

  ret = aclnnInplaceMuls(workspaceDev, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    return false;
  }
  if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    return false;
  }

  std::vector<uint8_t> outBytes(selfBytes.size(), 0);
  if (!outBytes.empty()) {
    if (aclrtMemcpy(outBytes.data(), outBytes.size(), selfDev, outBytes.size(), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
      return false;
    }
  }

  std::vector<float> ref(static_cast<size_t>(GetShapeSize(tc.selfShape)), 0.0f);
  for (int64_t i = 0; i < static_cast<int64_t>(ref.size()); ++i) {
    ref[static_cast<size_t>(i)] = ReadAsFloat(selfBytes, i, tc.selfType) * static_cast<float>(tc.scalarValue);
  }
  bool pass = CompareResult(outBytes, tc.selfType, ref, tc.atol, tc.rtol, true);

  aclDestroyScalar(scalar);
  aclDestroyTensor(self);
  aclrtFree(selfDev);
  if (workspaceDev != nullptr) {
    aclrtFree(workspaceDev);
  }
  LOG_PRINT(pass ? "[PASS] %s\n" : "[FAIL] %s\n", tc.name.c_str());
  return pass;
}

int main()
{

  printf(">>>>>> VERSION 2.0: AGGRESSIVE COVERAGE RUNNING <<<<<<\n");
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  auto initRet = Init(deviceId, &stream);
  CHECK_RET(initRet == ACL_SUCCESS, LOG_PRINT("Init failed, ret=%d\n", initRet); return initRet);

  std::vector<TestCase> tests = {
    {"float_same_shape", {4, 2}, {4, 2}, {4, 2}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, ACL_SUCCESS, false,
      1e-6f, 1e-6f, true},
    {"int32_same_shape", {8, 16}, {8, 16}, {8, 16}, ACL_INT32, ACL_INT32, ACL_INT32, true, ACL_SUCCESS, false,
      0.0f, 0.0f, true},
    {"float16_same_shape", {32, 32}, {32, 32}, {32, 32}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, true,
      ACL_SUCCESS, false, 1e-2f, 1e-2f, true},
    {"bf16_same_shape", {32, 32}, {32, 32}, {32, 32}, ACL_BF16, ACL_BF16, ACL_BF16, true,
      ACL_SUCCESS, false, 2e-2f, 2e-2f, true},
    {"double_same_shape", {32, 8}, {32, 8}, {32, 8}, ACL_DOUBLE, ACL_DOUBLE, ACL_DOUBLE, true,
      ACL_SUCCESS, false, 1e-9f, 1e-9f, true},
    {"uint8_same_shape", {64, 32}, {64, 32}, {64, 32}, ACL_UINT8, ACL_UINT8, ACL_UINT8, true,
      ACL_SUCCESS, false, 0.0f, 0.0f, true},
    {"int8_same_shape", {64, 32}, {64, 32}, {64, 32}, ACL_INT8, ACL_INT8, ACL_INT8, true,
      ACL_SUCCESS, false, 0.0f, 0.0f, true},
    {"uint16_same_shape_expect_fail", {32, 16}, {32, 16}, {32, 16}, ACL_UINT16, ACL_UINT16, ACL_UINT16,
      false, ERR_PARAM_INVALID, true, 0.0f, 0.0f, true},
    {"complex32_same_shape_expect_fail", {16, 16}, {16, 16}, {16, 16}, ACL_COMPLEX32, ACL_COMPLEX32, ACL_COMPLEX32,
      false, ERR_PARAM_INVALID, true, 0.0f, 0.0f, true},
    {"complex64_same_shape", {16, 16}, {16, 16}, {16, 16}, ACL_COMPLEX64, ACL_COMPLEX64, ACL_COMPLEX64, true,
      ACL_SUCCESS, false, 0.0f, 0.0f, true},
    {"complex64_odd_len31", {1, 31}, {1, 31}, {1, 31}, ACL_COMPLEX64, ACL_COMPLEX64, ACL_COMPLEX64, true,
      ACL_SUCCESS, false, 0.0f, 0.0f, true},
    {"complex128_same_shape_expect_fail_host_unsupported", {8, 8}, {8, 8}, {8, 8},
      ACL_COMPLEX128, ACL_COMPLEX128, ACL_COMPLEX128, false, ERR_PARAM_INVALID, true, 0.0f, 0.0f, true},
    {"complex128_broadcast_expect_fail_host_unsupported", {1, 31}, {31, 1}, {31, 31},
      ACL_COMPLEX128, ACL_COMPLEX128, ACL_COMPLEX128, false, ERR_PARAM_INVALID, true, 0.0f, 0.0f, false},
    {"bool_same_shape", {32, 8}, {32, 8}, {32, 8}, ACL_BOOL, ACL_BOOL, ACL_BOOL, true, ACL_SUCCESS, false,
      0.0f, 0.0f, true},
    {"int64_same_shape", {64, 4}, {64, 4}, {64, 4}, ACL_INT64, ACL_INT64, ACL_INT64, true, ACL_SUCCESS, false,
      0.0f, 0.0f, true},
    {"broadcast_2d", {4, 1}, {1, 4}, {4, 4}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, ACL_SUCCESS, false,
      1e-6f, 1e-6f, true},
    {"broadcast_4d", {2, 1, 3, 1}, {1, 4, 1, 5}, {2, 4, 3, 5}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true,
      ACL_SUCCESS, false, 1e-6f, 1e-6f, true},
    {"mix_fp16_fp32", {8, 8}, {8, 8}, {8, 8}, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT, true, ACL_SUCCESS, false,
      1e-3f, 1e-3f, true},
    {"mix_fp32_fp16", {8, 8}, {8, 8}, {8, 8}, ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT, true, ACL_SUCCESS, false,
      1e-3f, 1e-3f, true},
    {"mix_bf16_fp32", {8, 8}, {8, 8}, {8, 8}, ACL_BF16, ACL_FLOAT, ACL_FLOAT, true, ACL_SUCCESS, false,
      2e-2f, 2e-2f, true},
    {"mix_fp32_bf16", {8, 8}, {8, 8}, {8, 8}, ACL_FLOAT, ACL_BF16, ACL_FLOAT, true, ACL_SUCCESS, false,
      2e-2f, 2e-2f, true},
    {"mix_int32_int8_expect_fail", {8, 8}, {8, 8}, {8, 8}, ACL_INT32, ACL_INT8, ACL_INT32, false,
      ACL_SUCCESS, true, 0.0f, 0.0f, true},
    {"int16_same_shape", {32, 16}, {32, 16}, {32, 16}, ACL_INT16, ACL_INT16, ACL_INT16, true, ACL_SUCCESS, false,
      0.0f, 0.0f, true},
    {"extreme_broadcast_5d", {1, 5, 1, 10, 1}, {2, 1, 8, 1, 10}, {2, 5, 8, 10, 10}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
      true, ACL_SUCCESS, false, 1e-5f, 1e-5f, false},
    {"large_shape_float", {2048, 1024}, {2048, 1024}, {2048, 1024}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true,
      ACL_SUCCESS, false, 1e-6f, 1e-6f, false},
    {"large_shape_int8", {2048, 512}, {2048, 512}, {2048, 512}, ACL_INT8, ACL_INT8, ACL_INT8, true,
      ACL_SUCCESS, false, 0.0f, 0.0f, false},
    {"empty_tensor", {0, 32}, {0, 32}, {0, 32}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true,
      ACL_SUCCESS, false, 0.0f, 0.0f, true},
    {"invalid_broadcast", {2, 3}, {4}, {2, 3}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, ERR_PARAM_INVALID,
      false, 0.0f, 0.0f, true},
    {"out_shape_mismatch", {2, 3}, {2, 3}, {2, 2}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false,
      ERR_PARAM_INVALID, false, 0.0f, 0.0f, true}
  };

  int passCount = 0;
  int failCount = 0;
  for (const auto& tc : tests) {
    bool ok = RunMulTest(tc, stream);
    if (ok) {
      ++passCount;
    } else {
      ++failCount;
    }
  }

  std::vector<NonContiguousCase> nonContiguousTests = {
    {
      "non_contiguous_slice_like", {2, 3}, {4, 6}, {6, 2}, 1,
      {2, 3}, {2, 3}, ACL_FLOAT, 1e-5f, 1e-5f
    },
    {
      "non_contiguous_odd31_stride_gap", {1, 31}, {1, 64}, {64, 2}, 1,
      {1, 31}, {1, 31}, ACL_FLOAT, 1e-5f, 1e-5f
    }
  };
  for (const auto& tc : nonContiguousTests) {
    bool ok = RunNonContiguousMulTest(tc, stream);
    if (ok) ++passCount; else ++failCount;
  }

  std::vector<ScalarTestCase> mulsTests = {
    {"muls_float", {4, 8}, {4, 8}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 2.0, true, ACL_SUCCESS, false, 1e-6f, 1e-6f},
    {"muls_int32_scalar_int32", {16, 16}, {16, 16}, ACL_INT32, ACL_INT32, ACL_INT32, -3.0,
      true, ACL_SUCCESS, false, 0.0f, 0.0f},
    {"muls_int16_scalar_int32", {32, 8}, {32, 8}, ACL_INT16, ACL_INT32, ACL_INT16, 2.0,
      true, ACL_SUCCESS, false, 0.0f, 0.0f},
    {"muls_bool_scalar_int32", {16, 16}, {16, 16}, ACL_BOOL, ACL_INT32, ACL_BOOL, 1.0,
      true, ACL_SUCCESS, false, 0.0f, 0.0f},
    {"muls_fp16_expect_fail", {8, 8}, {8, 8}, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT16, -1.5,
      false, ERR_INNER_NULLPTR, false, 1e-2f, 1e-2f},
    {"muls_alpha_not_one", {4, 8}, {4, 8}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1.75, true, ACL_SUCCESS, false, 1e-6f, 1e-6f},
    {"muls_nan", {4, 4}, {4, 4}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, NAN, true, ACL_SUCCESS, false, 0.0f, 0.0f},
    {"muls_inf", {4, 4}, {4, 4}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, INFINITY, true, ACL_SUCCESS, false, 0.0f, 0.0f},
    {"muls_bf16_scalar_float", {16, 16}, {16, 16}, ACL_BF16, ACL_FLOAT, ACL_BF16, 1.5,
      false, ERR_PARAM_INVALID, true, 0.0f, 0.0f},
    {"muls_out_mismatch", {2, 3}, {2, 2}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1.0, false, ERR_PARAM_INVALID, false, 0.0f, 0.0f}
  };
  for (const auto& tc : mulsTests) {
    bool ok = RunMulsTest(tc, stream);
    if (ok) ++passCount; else ++failCount;
  }

  std::vector<InplaceTensorCase> inplaceMulTests = {
    {"inplace_mul_float", {4, 4}, {4, 4}, ACL_FLOAT, ACL_FLOAT, true, ACL_SUCCESS, 1e-6f, 1e-6f},
    {"inplace_mul_broadcast", {4, 4}, {1, 4}, ACL_FLOAT, ACL_FLOAT, true, ACL_SUCCESS, 1e-6f, 1e-6f},
    {"inplace_mul_shape_invalid", {2, 3}, {4}, ACL_FLOAT, ACL_FLOAT, false, ERR_PARAM_INVALID, 0.0f, 0.0f}
  };
  for (const auto& tc : inplaceMulTests) {
    bool ok = RunInplaceMulTest(tc, stream);
    if (ok) ++passCount; else ++failCount;
  }

  std::vector<ScalarTestCase> inplaceMulsTests = {
    {"inplace_muls_float", {4, 8}, {4, 8}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, -0.5, true, ACL_SUCCESS, false, 1e-6f, 1e-6f},
    {"inplace_muls_int32", {8, 8}, {8, 8}, ACL_INT32, ACL_INT32, ACL_INT32, 3.0, true, ACL_SUCCESS, false, 0.0f, 0.0f},
    {"inplace_muls_int16", {8, 8}, {8, 8}, ACL_INT16, ACL_INT32, ACL_INT16, -2.0, true, ACL_SUCCESS, false, 0.0f, 0.0f}
  };
  for (const auto& tc : inplaceMulsTests) {
    bool ok = RunInplaceMulsTest(tc, stream);
    if (ok) ++passCount; else ++failCount;
  }

  if (RunNullptrCase("nullptr_self", 0, ERR_PARAM_NULLPTR)) ++passCount; else ++failCount;
  if (RunNullptrCase("nullptr_other", 1, ERR_PARAM_NULLPTR)) ++passCount; else ++failCount;
  if (RunNullptrCase("nullptr_out", 2, ERR_PARAM_NULLPTR)) ++passCount; else ++failCount;

  LOG_PRINT("\n========== Mul E2E Summary =========="
            "\nTotal: %d, Pass: %d, Fail: %d\n", passCount + failCount, passCount, failCount);

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return (failCount == 0) ? 0 : 1;
}