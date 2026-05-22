/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_aclnn_pow.cpp
 * @brief Pow 算子端到端集成测试：覆盖 TensorScalar / ScalarTensor / TensorTensor / Exp2 及 Inplace 变体，
 *        含 CPU 期望值比对、广播、多 dtype 与异常入参。
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

namespace {

static int g_passCount = 0;
static int g_failCount = 0;

constexpr double kRtol = 1e-3;
constexpr double kAtol = 1e-4;

int64_t MultiToFlat(const std::vector<int64_t>& idx, const std::vector<int64_t>& shape) {
  int64_t flat = 0;
  int64_t stride = 1;
  for (int64_t d = static_cast<int64_t>(shape.size()) - 1; d >= 0; --d) {
    flat += idx[static_cast<size_t>(d)] * stride;
    stride *= shape[static_cast<size_t>(d)];
  }
  return flat;
}

void FlatToMulti(int64_t flat, const std::vector<int64_t>& shape, std::vector<int64_t>* idx) {
  idx->resize(shape.size());
  for (int64_t d = static_cast<int64_t>(shape.size()) - 1; d >= 0; --d) {
    (*idx)[static_cast<size_t>(d)] = flat % shape[static_cast<size_t>(d)];
    flat /= shape[static_cast<size_t>(d)];
  }
}

/**
 * @brief 将输出下标映射到参与 broadcast 的输入张量线性下标。
 */
int64_t BroadcastInputIndex(const std::vector<int64_t>& outIdx, const std::vector<int64_t>& inShape) {
  std::vector<int64_t> inIdx(inShape.size());
  const size_t off = outIdx.size() - inShape.size();
  for (size_t k = 0; k < inShape.size(); ++k) {
    const int64_t oi = outIdx[off + k];
    inIdx[k] = (inShape[k] == 1) ? 0 : oi;
  }
  return MultiToFlat(inIdx, inShape);
}

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

bool FloatClose(double actual, double expected) {
  const double diff = std::fabs(actual - expected);
  return diff <= kAtol + kRtol * std::fabs(expected);
}

uint16_t FloatToFloat16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000);
  int32_t exponent = static_cast<int32_t>(((bits >> 23) & 0xff) - 127 + 15);
  const uint32_t mantissa = bits & 0x7fffff;
  if (exponent <= 0) {
    return sign;
  }
  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00);
  }
  const uint16_t fp16 = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
  return fp16;
}

float Float16ToFloat(uint16_t value) {
  const unsigned int sign = (value >> 15) & 0x1u;
  const unsigned int exponent = (value >> 10) & 0x1fu;
  const unsigned int mantissa = value & 0x3ffu;
  float result = 0.0f;
  if (exponent == 0) {
    result = static_cast<float>(mantissa) * 0.0000019073486328125f;
  } else if (exponent == 31) {
    result = (mantissa == 0) ? (1.0f / 0.0f) : (0.0f / 0.0f);
  } else {
    result = (1.0f + mantissa * 0.0009765625f) * std::pow(2.0f, static_cast<float>(static_cast<int>(exponent) - 15));
  }
  return sign ? -result : result;
}

void ReportCase(const char* name, bool ok) {
  if (ok) {
    LOG_PRINT("[PASS] %s\n", name);
    ++g_passCount;
  } else {
    LOG_PRINT("[FAIL] %s\n", name);
    ++g_failCount;
  }
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
  const auto size = static_cast<size_t>(GetShapeSize(shape) * sizeof(T));
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
    strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return ACL_SUCCESS;
}

/**
 * @brief 两段式 API：同步完成后释放 workspace（顺序与官方示例一致）。
 */
aclnnStatus RunPowTensorScalar(aclTensor* self, aclScalar* exponent, aclTensor* out, aclrtStream stream) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      return ret;
    }
  }
  ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceAddr != nullptr) {
      aclrtFree(workspaceAddr);
    }
    return ret;
  }
  ret = aclrtSynchronizeStream(stream);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ret;
}

aclnnStatus RunInplacePowTensorScalar(aclTensor* self, aclScalar* exponent, aclrtStream stream) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnInplacePowTensorScalarGetWorkspaceSize(self, exponent, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      return ret;
    }
  }
  ret = aclnnInplacePowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceAddr != nullptr) {
      aclrtFree(workspaceAddr);
    }
    return ret;
  }
  ret = aclrtSynchronizeStream(stream);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ret;
}

aclnnStatus RunPowScalarTensor(aclScalar* base, aclTensor* exp, aclTensor* out, aclrtStream stream) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnPowScalarTensorGetWorkspaceSize(base, exp, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      return ret;
    }
  }
  ret = aclnnPowScalarTensor(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceAddr != nullptr) {
      aclrtFree(workspaceAddr);
    }
    return ret;
  }
  ret = aclrtSynchronizeStream(stream);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ret;
}

aclnnStatus RunPowTensorTensor(aclTensor* self, aclTensor* exp, aclTensor* out, aclrtStream stream) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnPowTensorTensorGetWorkspaceSize(self, exp, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      return ret;
    }
  }
  ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceAddr != nullptr) {
      aclrtFree(workspaceAddr);
    }
    return ret;
  }
  ret = aclrtSynchronizeStream(stream);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ret;
}

aclnnStatus RunInplacePowTensorTensor(aclTensor* self, aclTensor* exp, aclrtStream stream) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnInplacePowTensorTensorGetWorkspaceSize(self, exp, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      return ret;
    }
  }
  ret = aclnnInplacePowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceAddr != nullptr) {
      aclrtFree(workspaceAddr);
    }
    return ret;
  }
  ret = aclrtSynchronizeStream(stream);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ret;
}

aclnnStatus RunExp2(aclTensor* self, aclTensor* out, aclrtStream stream) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnExp2GetWorkspaceSize(self, out, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      return ret;
    }
  }
  ret = aclnnExp2(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceAddr != nullptr) {
      aclrtFree(workspaceAddr);
    }
    return ret;
  }
  ret = aclrtSynchronizeStream(stream);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ret;
}

aclnnStatus RunInplaceExp2(aclTensor* selfRef, aclrtStream stream) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnInplaceExp2GetWorkspaceSize(selfRef, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      return ret;
    }
  }
  ret = aclnnInplaceExp2(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceAddr != nullptr) {
      aclrtFree(workspaceAddr);
    }
    return ret;
  }
  ret = aclrtSynchronizeStream(stream);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  return ret;
}

bool TestPowTensorScalarCase(aclrtStream stream, const char* caseName, const std::vector<float>& selfHost,
                             const std::vector<int64_t>& shape, float expVal) {
  void* selfDev = nullptr;
  void* outDev = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclScalar* exponent = aclCreateScalar(&expVal, aclDataType::ACL_FLOAT);
  if (exponent == nullptr) {
    ReportCase(caseName, false);
    return false;
  }
  std::vector<float> outHost(static_cast<size_t>(GetShapeSize(shape)), 0.0f);
  if (CreateAclTensor(selfHost, shape, &selfDev, aclDataType::ACL_FLOAT, &self) != ACL_SUCCESS ||
      CreateAclTensor(outHost, shape, &outDev, aclDataType::ACL_FLOAT, &out) != ACL_SUCCESS) {
    aclDestroyScalar(exponent);
    if (self != nullptr) {
      aclDestroyTensor(self);
    }
    if (selfDev != nullptr) {
      aclrtFree(selfDev);
    }
    ReportCase(caseName, false);
    return false;
  }
  aclnnStatus st = RunPowTensorScalar(self, exponent, out, stream);
  const int64_t n = GetShapeSize(shape);
  std::vector<float> result(static_cast<size_t>(n));
  if (st == ACL_SUCCESS) {
    st = aclrtMemcpy(result.data(), static_cast<size_t>(n) * sizeof(float), outDev,
                     static_cast<size_t>(n) * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  }
  bool ok = (st == ACL_SUCCESS);
  for (int64_t i = 0; ok && i < n; ++i) {
    const double expected = std::pow(static_cast<double>(selfHost[static_cast<size_t>(i)]), static_cast<double>(expVal));
    if (!FloatClose(static_cast<double>(result[static_cast<size_t>(i)]), expected)) {
      ok = false;
    }
  }
  aclDestroyTensor(out);
  aclDestroyTensor(self);
  aclrtFree(outDev);
  aclrtFree(selfDev);
  aclDestroyScalar(exponent);
  ReportCase(caseName, ok);
  return ok;
}

bool TestInplacePowTensorScalarCase(aclrtStream stream, const char* caseName, std::vector<float> selfHost,
                                    const std::vector<int64_t>& shape, float expVal) {
  void* selfDev = nullptr;
  aclTensor* self = nullptr;
  aclScalar* exponent = aclCreateScalar(&expVal, aclDataType::ACL_FLOAT);
  if (exponent == nullptr) {
    ReportCase(caseName, false);
    return false;
  }
  if (CreateAclTensor(selfHost, shape, &selfDev, aclDataType::ACL_FLOAT, &self) != ACL_SUCCESS) {
    aclDestroyScalar(exponent);
    ReportCase(caseName, false);
    return false;
  }
  aclnnStatus st = RunInplacePowTensorScalar(self, exponent, stream);
  const int64_t n = GetShapeSize(shape);
  std::vector<float> result(static_cast<size_t>(n));
  if (st == ACL_SUCCESS) {
    st = aclrtMemcpy(result.data(), static_cast<size_t>(n) * sizeof(float), selfDev,
                     static_cast<size_t>(n) * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  }
  bool ok = (st == ACL_SUCCESS);
  for (int64_t i = 0; ok && i < n; ++i) {
    const double expected =
        std::pow(static_cast<double>(selfHost[static_cast<size_t>(i)]), static_cast<double>(expVal));
    if (!FloatClose(static_cast<double>(result[static_cast<size_t>(i)]), expected)) {
      ok = false;
    }
  }
  aclDestroyTensor(self);
  aclrtFree(selfDev);
  aclDestroyScalar(exponent);
  ReportCase(caseName, ok);
  return ok;
}

bool TestPowScalarTensorCase(aclrtStream stream, const char* caseName, float baseVal,
                             const std::vector<float>& expHost, const std::vector<int64_t>& expShape) {
  void* expDev = nullptr;
  void* outDev = nullptr;
  aclTensor* expTensor = nullptr;
  aclTensor* out = nullptr;
  aclScalar* baseScalar = aclCreateScalar(&baseVal, aclDataType::ACL_FLOAT);
  if (baseScalar == nullptr) {
    ReportCase(caseName, false);
    return false;
  }
  std::vector<float> outHost(static_cast<size_t>(GetShapeSize(expShape)), 0.0f);
  if (CreateAclTensor(expHost, expShape, &expDev, aclDataType::ACL_FLOAT, &expTensor) != ACL_SUCCESS ||
      CreateAclTensor(outHost, expShape, &outDev, aclDataType::ACL_FLOAT, &out) != ACL_SUCCESS) {
    aclDestroyScalar(baseScalar);
    if (expTensor != nullptr) {
      aclDestroyTensor(expTensor);
    }
    if (expDev != nullptr) {
      aclrtFree(expDev);
    }
    ReportCase(caseName, false);
    return false;
  }
  aclnnStatus st = RunPowScalarTensor(baseScalar, expTensor, out, stream);
  const int64_t n = GetShapeSize(expShape);
  std::vector<float> result(static_cast<size_t>(n));
  if (st == ACL_SUCCESS) {
    st = aclrtMemcpy(result.data(), static_cast<size_t>(n) * sizeof(float), outDev,
                     static_cast<size_t>(n) * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  }
  bool ok = (st == ACL_SUCCESS);
  for (int64_t i = 0; ok && i < n; ++i) {
    const double expected =
        std::pow(static_cast<double>(baseVal), static_cast<double>(expHost[static_cast<size_t>(i)]));
    if (!FloatClose(static_cast<double>(result[static_cast<size_t>(i)]), expected)) {
      ok = false;
    }
  }
  aclDestroyTensor(out);
  aclDestroyTensor(expTensor);
  aclrtFree(outDev);
  aclrtFree(expDev);
  aclDestroyScalar(baseScalar);
  ReportCase(caseName, ok);
  return ok;
}

bool TestPowTensorTensorCase(aclrtStream stream, const char* caseName, const std::vector<float>& selfHost,
                             const std::vector<int64_t>& selfShape, const std::vector<float>& expHost,
                             const std::vector<int64_t>& expShape, const std::vector<int64_t>& outShape) {
  void* sDev = nullptr;
  void* eDev = nullptr;
  void* oDev = nullptr;
  aclTensor* self = nullptr;
  aclTensor* exp = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> outHost(static_cast<size_t>(GetShapeSize(outShape)), 0.0f);
  if (CreateAclTensor(selfHost, selfShape, &sDev, aclDataType::ACL_FLOAT, &self) != ACL_SUCCESS ||
      CreateAclTensor(expHost, expShape, &eDev, aclDataType::ACL_FLOAT, &exp) != ACL_SUCCESS ||
      CreateAclTensor(outHost, outShape, &oDev, aclDataType::ACL_FLOAT, &out) != ACL_SUCCESS) {
    if (self != nullptr) {
      aclDestroyTensor(self);
    }
    if (sDev != nullptr) {
      aclrtFree(sDev);
    }
    if (exp != nullptr) {
      aclDestroyTensor(exp);
    }
    if (eDev != nullptr) {
      aclrtFree(eDev);
    }
    ReportCase(caseName, false);
    return false;
  }
  aclnnStatus st = RunPowTensorTensor(self, exp, out, stream);
  const int64_t n = GetShapeSize(outShape);
  std::vector<float> result(static_cast<size_t>(n));
  if (st == ACL_SUCCESS) {
    st = aclrtMemcpy(result.data(), static_cast<size_t>(n) * sizeof(float), oDev,
                     static_cast<size_t>(n) * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  }
  bool ok = (st == ACL_SUCCESS);
  std::vector<int64_t> outIdx;
  for (int64_t i = 0; ok && i < n; ++i) {
    FlatToMulti(i, outShape, &outIdx);
    const int64_t is = BroadcastInputIndex(outIdx, selfShape);
    const int64_t ie = BroadcastInputIndex(outIdx, expShape);
    const double expected = std::pow(static_cast<double>(selfHost[static_cast<size_t>(is)]),
                                     static_cast<double>(expHost[static_cast<size_t>(ie)]));
    if (!FloatClose(static_cast<double>(result[static_cast<size_t>(i)]), expected)) {
      ok = false;
    }
  }
  aclDestroyTensor(out);
  aclDestroyTensor(exp);
  aclDestroyTensor(self);
  aclrtFree(oDev);
  aclrtFree(eDev);
  aclrtFree(sDev);
  ReportCase(caseName, ok);
  return ok;
}

bool TestInplacePowTensorTensorCase(aclrtStream stream, const char* caseName, std::vector<float> selfHost,
                                    const std::vector<int64_t>& selfShape, const std::vector<float>& expHost,
                                    const std::vector<int64_t>& expShape) {
  void* sDev = nullptr;
  void* eDev = nullptr;
  aclTensor* self = nullptr;
  aclTensor* exp = nullptr;
  if (CreateAclTensor(selfHost, selfShape, &sDev, aclDataType::ACL_FLOAT, &self) != ACL_SUCCESS ||
      CreateAclTensor(expHost, expShape, &eDev, aclDataType::ACL_FLOAT, &exp) != ACL_SUCCESS) {
    if (self != nullptr) {
      aclDestroyTensor(self);
    }
    if (sDev != nullptr) {
      aclrtFree(sDev);
    }
    ReportCase(caseName, false);
    return false;
  }
  aclnnStatus st = RunInplacePowTensorTensor(self, exp, stream);
  const int64_t n = GetShapeSize(selfShape);
  std::vector<float> result(static_cast<size_t>(n));
  if (st == ACL_SUCCESS) {
    st = aclrtMemcpy(result.data(), static_cast<size_t>(n) * sizeof(float), sDev,
                     static_cast<size_t>(n) * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  }
  bool ok = (st == ACL_SUCCESS);
  std::vector<int64_t> outIdx;
  for (int64_t i = 0; ok && i < n; ++i) {
    FlatToMulti(i, selfShape, &outIdx);
    const int64_t ie = BroadcastInputIndex(outIdx, expShape);
    const double expected = std::pow(static_cast<double>(selfHost[static_cast<size_t>(i)]),
                                     static_cast<double>(expHost[static_cast<size_t>(ie)]));
    if (!FloatClose(static_cast<double>(result[static_cast<size_t>(i)]), expected)) {
      ok = false;
    }
  }
  aclDestroyTensor(exp);
  aclDestroyTensor(self);
  aclrtFree(eDev);
  aclrtFree(sDev);
  ReportCase(caseName, ok);
  return ok;
}

bool TestExp2Case(aclrtStream stream, const char* caseName, const std::vector<float>& selfHost,
                  const std::vector<int64_t>& shape) {
  void* sDev = nullptr;
  void* oDev = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> outHost(static_cast<size_t>(GetShapeSize(shape)), 0.0f);
  if (CreateAclTensor(selfHost, shape, &sDev, aclDataType::ACL_FLOAT, &self) != ACL_SUCCESS ||
      CreateAclTensor(outHost, shape, &oDev, aclDataType::ACL_FLOAT, &out) != ACL_SUCCESS) {
    if (self != nullptr) {
      aclDestroyTensor(self);
    }
    if (sDev != nullptr) {
      aclrtFree(sDev);
    }
    ReportCase(caseName, false);
    return false;
  }
  aclnnStatus st = RunExp2(self, out, stream);
  const int64_t n = GetShapeSize(shape);
  std::vector<float> result(static_cast<size_t>(n));
  if (st == ACL_SUCCESS) {
    st = aclrtMemcpy(result.data(), static_cast<size_t>(n) * sizeof(float), oDev,
                     static_cast<size_t>(n) * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  }
  bool ok = (st == ACL_SUCCESS);
  for (int64_t i = 0; ok && i < n; ++i) {
    const double expected = std::pow(2.0, static_cast<double>(selfHost[static_cast<size_t>(i)]));
    if (!FloatClose(static_cast<double>(result[static_cast<size_t>(i)]), expected)) {
      ok = false;
    }
  }
  aclDestroyTensor(out);
  aclDestroyTensor(self);
  aclrtFree(oDev);
  aclrtFree(sDev);
  ReportCase(caseName, ok);
  return ok;
}

bool TestInplaceExp2Case(aclrtStream stream, const char* caseName, std::vector<float> selfHost,
                         const std::vector<int64_t>& shape) {
  void* sDev = nullptr;
  aclTensor* self = nullptr;
  if (CreateAclTensor(selfHost, shape, &sDev, aclDataType::ACL_FLOAT, &self) != ACL_SUCCESS) {
    ReportCase(caseName, false);
    return false;
  }
  aclnnStatus st = RunInplaceExp2(self, stream);
  const int64_t n = GetShapeSize(shape);
  std::vector<float> result(static_cast<size_t>(n));
  if (st == ACL_SUCCESS) {
    st = aclrtMemcpy(result.data(), static_cast<size_t>(n) * sizeof(float), sDev,
                     static_cast<size_t>(n) * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  }
  bool ok = (st == ACL_SUCCESS);
  for (int64_t i = 0; ok && i < n; ++i) {
    const double expected = std::pow(2.0, static_cast<double>(selfHost[static_cast<size_t>(i)]));
    if (!FloatClose(static_cast<double>(result[static_cast<size_t>(i)]), expected)) {
      ok = false;
    }
  }
  aclDestroyTensor(self);
  aclrtFree(sDev);
  ReportCase(caseName, ok);
  return ok;
}

bool TestPowTensorScalarFp16(aclrtStream stream, const char* caseName) {
  const std::vector<int64_t> shape = {2, 2};
  const std::vector<uint16_t> selfBits = {FloatToFloat16(2.0f), FloatToFloat16(3.0f), FloatToFloat16(0.5f),
                                          FloatToFloat16(-1.0f)};
  float expVal = 2.0f;
  void* selfDev = nullptr;
  void* outDev = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclScalar* exponent = aclCreateScalar(&expVal, aclDataType::ACL_FLOAT);
  if (exponent == nullptr) {
    ReportCase(caseName, false);
    return false;
  }
  std::vector<uint16_t> outBits(static_cast<size_t>(GetShapeSize(shape)), 0);
  if (CreateAclTensor(selfBits, shape, &selfDev, aclDataType::ACL_FLOAT16, &self) != ACL_SUCCESS ||
      CreateAclTensor(outBits, shape, &outDev, aclDataType::ACL_FLOAT16, &out) != ACL_SUCCESS) {
    aclDestroyScalar(exponent);
    if (self != nullptr) {
      aclDestroyTensor(self);
    }
    if (selfDev != nullptr) {
      aclrtFree(selfDev);
    }
    ReportCase(caseName, false);
    return false;
  }
  aclnnStatus st = RunPowTensorScalar(self, exponent, out, stream);
  const int64_t n = GetShapeSize(shape);
  std::vector<uint16_t> result(static_cast<size_t>(n));
  if (st == ACL_SUCCESS) {
    st = aclrtMemcpy(result.data(), static_cast<size_t>(n) * sizeof(uint16_t), outDev,
                     static_cast<size_t>(n) * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
  }
  bool ok = (st == ACL_SUCCESS);
  for (int64_t i = 0; ok && i < n; ++i) {
    const float baseF = Float16ToFloat(selfBits[static_cast<size_t>(i)]);
    const double expected = std::pow(static_cast<double>(baseF), static_cast<double>(expVal));
    const float actualF = Float16ToFloat(result[static_cast<size_t>(i)]);
    if (!FloatClose(static_cast<double>(actualF), expected)) {
      ok = false;
    }
  }
  aclDestroyTensor(out);
  aclDestroyTensor(self);
  aclrtFree(outDev);
  aclrtFree(selfDev);
  aclDestroyScalar(exponent);
  ReportCase(caseName, ok);
  return ok;
}

bool TestPowTensorScalarInt32(aclrtStream stream, const char* caseName) {
  const std::vector<int64_t> shape = {4};
  const std::vector<int32_t> selfHost = {2, -2, 0, 5};
  int32_t expInt = 3;
  void* selfDev = nullptr;
  void* outDev = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclScalar* exponent = aclCreateScalar(&expInt, aclDataType::ACL_INT32);
  if (exponent == nullptr) {
    ReportCase(caseName, false);
    return false;
  }
  std::vector<int32_t> outHost(4, 0);
  if (CreateAclTensor(selfHost, shape, &selfDev, aclDataType::ACL_INT32, &self) != ACL_SUCCESS ||
      CreateAclTensor(outHost, shape, &outDev, aclDataType::ACL_INT32, &out) != ACL_SUCCESS) {
    aclDestroyScalar(exponent);
    if (self != nullptr) {
      aclDestroyTensor(self);
    }
    if (selfDev != nullptr) {
      aclrtFree(selfDev);
    }
    ReportCase(caseName, false);
    return false;
  }
  aclnnStatus st = RunPowTensorScalar(self, exponent, out, stream);
  std::vector<int32_t> result(4);
  if (st == ACL_SUCCESS) {
    st = aclrtMemcpy(result.data(), sizeof(int32_t) * 4, outDev, sizeof(int32_t) * 4, ACL_MEMCPY_DEVICE_TO_HOST);
  }
  bool ok = (st == ACL_SUCCESS);
  for (int i = 0; ok && i < 4; ++i) {
    const int32_t expected =
        static_cast<int32_t>(std::llround(std::pow(static_cast<double>(selfHost[static_cast<size_t>(i)]),
                                                    static_cast<double>(expInt))));
    if (result[static_cast<size_t>(i)] != expected) {
      ok = false;
    }
  }
  aclDestroyTensor(out);
  aclDestroyTensor(self);
  aclrtFree(outDev);
  aclrtFree(selfDev);
  aclDestroyScalar(exponent);
  ReportCase(caseName, ok);
  return ok;
}

/**
 * @brief self 为空指针时第一段接口应返回参数错误（覆盖校验分支）。
 */
void TestNullptrGetWorkspaceSize() {
  std::vector<float> outData = {0.0f};
  std::vector<int64_t> shape = {1};
  void* outDev = nullptr;
  aclTensor* out = nullptr;
  float f = 1.0f;
  aclScalar* s = aclCreateScalar(&f, aclDataType::ACL_FLOAT);
  if (s == nullptr) {
    ReportCase("aclnnPowTensorScalarGetWorkspaceSize_null_self", false);
    return;
  }
  if (CreateAclTensor(outData, shape, &outDev, aclDataType::ACL_FLOAT, &out) != ACL_SUCCESS) {
    aclDestroyScalar(s);
    ReportCase("aclnnPowTensorScalarGetWorkspaceSize_null_self", false);
    return;
  }
  uint64_t ws = 0;
  aclOpExecutor* ex = nullptr;
  const aclnnStatus st = aclnnPowTensorScalarGetWorkspaceSize(nullptr, s, out, &ws, &ex);
  const bool ok = (st != ACL_SUCCESS);
  aclDestroyTensor(out);
  aclrtFree(outDev);
  aclDestroyScalar(s);
  ReportCase("aclnnPowTensorScalarGetWorkspaceSize_null_self", ok);
}

}  // namespace

int main() {
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  auto ret = Init(deviceId, &stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
    return 1;
  }

  TestPowTensorScalarCase(stream, "pow_ts_float_exp0", {0.0f, 1.0f, 2.0f, -1.0f}, {2, 2}, 0.0f);
  TestPowTensorScalarCase(stream, "pow_ts_float_exp1", {0.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, 1.0f);
  TestPowTensorScalarCase(stream, "pow_ts_float_exp2", {0.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, 2.0f);
  TestPowTensorScalarCase(stream, "pow_ts_float_exp_half", {4.0f, 9.0f, 2.0f, 0.0f}, {2, 2}, 0.5f);
  TestPowTensorScalarCase(stream, "pow_ts_float_exp_neg1", {2.0f, 4.0f, 1.0f, 8.0f}, {2, 2}, -1.0f);
  TestPowTensorScalarCase(stream, "pow_ts_float_exp3", {2.0f, 3.0f, -1.0f, 0.0f}, {2, 2}, 3.0f);
  TestPowTensorScalarCase(stream, "pow_ts_float_exp_frac", {0.0f, 1.0f, 2.0f, 3.0f}, {2, 2}, 4.1f);

  TestInplacePowTensorScalarCase(stream, "inplace_pow_ts_exp2", {1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, 2.0f);

  TestPowScalarTensorCase(stream, "pow_st_base2", 2.0f, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f}, {5});

  TestPowTensorTensorCase(stream, "pow_tt_same_shape", {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}, {4, 2},
                          {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f}, {4, 2}, {4, 2});

  TestPowTensorTensorCase(stream, "pow_tt_broadcast", {2.0f, 3.0f, 4.0f, 5.0f}, {4, 1}, {0.0f, 2.0f}, {1, 2},
                          {4, 2});

  TestInplacePowTensorTensorCase(stream, "inplace_pow_tt", {2.0f, 3.0f, 4.0f, 5.0f}, {2, 2},
                                 {1.0f, 2.0f, 0.5f, 2.0f}, {2, 2});

  TestExp2Case(stream, "exp2_basic", {0.0f, 1.0f, 2.0f, 3.0f}, {2, 2});
  TestInplaceExp2Case(stream, "inplace_exp2", {0.0f, 2.0f, -1.0f, 4.0f}, {2, 2});

  TestPowTensorScalarFp16(stream, "pow_ts_fp16_exp2");
  TestPowTensorScalarInt32(stream, "pow_ts_int32_exp3");

  TestNullptrGetWorkspaceSize();

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  LOG_PRINT("\n========== 汇总 ==========\n");
  LOG_PRINT("通过: %d, 失败: %d\n", g_passCount, g_failCount);
  return g_failCount > 0 ? 2 : 0;
}
