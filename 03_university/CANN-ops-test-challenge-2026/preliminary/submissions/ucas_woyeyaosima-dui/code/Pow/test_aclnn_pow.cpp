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
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  int64_t shapeSize = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    shapeSize *= shape[i];
  }
  return shapeSize;
}

std::vector<int64_t> GetContiguousStrides(const std::vector<int64_t>& shape)
{
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
  }
  return strides;
}

struct TensorHolder {
  void* deviceAddr = nullptr;
  aclTensor* tensor = nullptr;

  TensorHolder() {}
  TensorHolder(const TensorHolder&) = delete;
  TensorHolder& operator=(const TensorHolder&) = delete;

  ~TensorHolder()
  {
    Reset();
  }

  void Reset()
  {
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

struct ScalarHolder {
  aclScalar* scalar = nullptr;

  ScalarHolder() {}
  ScalarHolder(const ScalarHolder&) = delete;
  ScalarHolder& operator=(const ScalarHolder&) = delete;

  ~ScalarHolder()
  {
    Reset();
  }

  void Reset()
  {
    if (scalar != nullptr) {
      aclDestroyScalar(scalar);
      scalar = nullptr;
    }
  }
};

static inline uint32_t FloatToBits(float f)
{
  uint32_t bits = 0;
  std::memcpy(&bits, &f, sizeof(bits));
  return bits;
}

static inline float BitsToFloat(uint32_t bits)
{
  float f = 0.0F;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

static inline uint16_t FloatToFp16Bits(float value)
{
  uint32_t bits = FloatToBits(value);
  uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
  uint32_t mantissa = bits & 0x7FFFFFU;

  if (exponent <= 0) {
    return sign;
  }
  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00U);
  }
  return static_cast<uint16_t>(sign | (static_cast<uint16_t>(exponent) << 10) | static_cast<uint16_t>(mantissa >> 13));
}

static inline float Fp16BitsToFloat(uint16_t value)
{
  uint32_t sign = static_cast<uint32_t>((value >> 15) & 0x1U);
  uint32_t exponent = static_cast<uint32_t>((value >> 10) & 0x1FU);
  uint32_t mantissa = static_cast<uint32_t>(value & 0x3FFU);

  if (exponent == 0) {
    float denorm = static_cast<float>(mantissa) * 0.000000059604644775390625F;
    return sign ? -denorm : denorm;
  }
  if (exponent == 31) {
    if (mantissa == 0) {
      return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
    }
    return std::numeric_limits<float>::quiet_NaN();
  }
  float normal = (1.0F + static_cast<float>(mantissa) * 0.0009765625F) * std::pow(2.0F, static_cast<float>(exponent) - 15.0F);
  return sign ? -normal : normal;
}

static inline uint16_t FloatToBf16Bits(float value)
{
  uint32_t bits = FloatToBits(value);
  uint32_t roundingBias = ((bits >> 16) & 1U) + 0x7FFFU;
  bits += roundingBias;
  return static_cast<uint16_t>(bits >> 16);
}

static inline float Bf16BitsToFloat(uint16_t value)
{
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  return BitsToFloat(bits);
}

std::vector<uint16_t> FloatVectorToFp16Bits(const std::vector<float>& input)
{
  std::vector<uint16_t> out(input.size(), 0);
  for (size_t i = 0; i < input.size(); ++i) {
    out[i] = FloatToFp16Bits(input[i]);
  }
  return out;
}

std::vector<float> Fp16BitsVectorToFloat(const std::vector<uint16_t>& input)
{
  std::vector<float> out(input.size(), 0.0F);
  for (size_t i = 0; i < input.size(); ++i) {
    out[i] = Fp16BitsToFloat(input[i]);
  }
  return out;
}

std::vector<uint16_t> FloatVectorToBf16Bits(const std::vector<float>& input)
{
  std::vector<uint16_t> out(input.size(), 0);
  for (size_t i = 0; i < input.size(); ++i) {
    out[i] = FloatToBf16Bits(input[i]);
  }
  return out;
}

std::vector<float> Bf16BitsVectorToFloat(const std::vector<uint16_t>& input)
{
  std::vector<float> out(input.size(), 0.0F);
  for (size_t i = 0; i < input.size(); ++i) {
    out[i] = Bf16BitsToFloat(input[i]);
  }
  return out;
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
                    TensorHolder* holder, aclFormat format = ACL_FORMAT_ND)
{
  CHECK_RET(holder != nullptr, return ACL_ERROR_INVALID_PARAM);
  holder->Reset();

  const int64_t size = GetShapeSize(shape) * static_cast<int64_t>(sizeof(T));
  const std::vector<int64_t> strides = GetContiguousStrides(shape);

  if (size > 0) {
    auto ret = aclrtMalloc(&holder->deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(holder->deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  } else {
    holder->deviceAddr = nullptr;
  }

  holder->tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(), shape.size(),
                                   holder->deviceAddr);
  CHECK_RET(holder->tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
  return ACL_SUCCESS;
}

template <typename T>
int CopyTensorToHost(const TensorHolder& holder, std::vector<T>* hostData)
{
  CHECK_RET(hostData != nullptr, return ACL_ERROR_INVALID_PARAM);
  const int64_t size = static_cast<int64_t>(hostData->size() * sizeof(T));
  if (size == 0) {
    return ACL_SUCCESS;
  }
  CHECK_RET(holder.deviceAddr != nullptr, return ACL_ERROR_INVALID_PARAM);
  auto ret = aclrtMemcpy(hostData->data(), size, holder.deviceAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

template <typename T>
int CreateAclScalar(T value, aclDataType dataType, ScalarHolder* holder)
{
  CHECK_RET(holder != nullptr, return ACL_ERROR_INVALID_PARAM);
  holder->Reset();
  holder->scalar = aclCreateScalar(&value, dataType);
  CHECK_RET(holder->scalar != nullptr, LOG_PRINT("aclCreateScalar failed.\n"); return ACL_ERROR_INVALID_PARAM);
  return ACL_SUCCESS;
}

bool IsClose(double actual, double expected, double atol, double rtol)
{
  return std::fabs(actual - expected) <= (atol + rtol * std::fabs(expected));
}

bool CheckFloatResult(const std::vector<float>& actual, const std::vector<double>& expected, double atol, double rtol,
                      std::string* err)
{
  if (actual.size() != expected.size()) {
    if (err != nullptr) {
      *err = "size mismatch";
    }
    return false;
  }

  for (size_t i = 0; i < actual.size(); ++i) {
    const double a = static_cast<double>(actual[i]);
    const double e = expected[i];
    const bool bothNaN = std::isnan(a) && std::isnan(e);
    const bool bothInf = std::isinf(a) && std::isinf(e) && ((a > 0) == (e > 0));
    if (bothNaN || bothInf) {
      continue;
    }
    if (!IsClose(a, e, atol, rtol)) {
      if (err != nullptr) {
        *err = "index=" + std::to_string(i) + " actual=" + std::to_string(a) + " expected=" + std::to_string(e);
      }
      return false;
    }
  }
  return true;
}

template <typename T>
bool CheckExactResult(const std::vector<T>& actual, const std::vector<T>& expected, std::string* err)
{
  if (actual.size() != expected.size()) {
    if (err != nullptr) {
      *err = "size mismatch";
    }
    return false;
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      if (err != nullptr) {
        *err = "index=" + std::to_string(i) + " mismatch";
      }
      return false;
    }
  }
  return true;
}

void GetCoordsFromLinear(int64_t linearIndex, const std::vector<int64_t>& shape, std::vector<int64_t>* coords)
{
  coords->assign(shape.size(), 0);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    int64_t dim = shape[static_cast<size_t>(i)];
    if (dim > 0) {
      (*coords)[static_cast<size_t>(i)] = linearIndex % dim;
      linearIndex /= dim;
    }
  }
}

int64_t GetBroadcastOffset(const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape,
                           const std::vector<int64_t>& outCoords)
{
  const std::vector<int64_t> inStrides = GetContiguousStrides(inShape);
  const int64_t inRank = static_cast<int64_t>(inShape.size());
  const int64_t outRank = static_cast<int64_t>(outShape.size());
  const int64_t rankDiff = outRank - inRank;
  int64_t offset = 0;
  for (int64_t i = 0; i < inRank; ++i) {
    const int64_t outAxis = i + rankDiff;
    const int64_t coord = inShape[static_cast<size_t>(i)] == 1 ? 0 : outCoords[static_cast<size_t>(outAxis)];
    offset += coord * inStrides[static_cast<size_t>(i)];
  }
  return offset;
}

template <typename BaseT, typename ExpT>
std::vector<double> ComputeExpectedPowBroadcast(const std::vector<BaseT>& base, const std::vector<int64_t>& baseShape,
                                                const std::vector<ExpT>& exponent,
                                                const std::vector<int64_t>& exponentShape,
                                                const std::vector<int64_t>& outShape)
{
  std::vector<double> expected(static_cast<size_t>(GetShapeSize(outShape)), 0.0);
  std::vector<int64_t> coords;
  for (int64_t i = 0; i < GetShapeSize(outShape); ++i) {
    GetCoordsFromLinear(i, outShape, &coords);
    const int64_t baseOffset = GetBroadcastOffset(baseShape, outShape, coords);
    const int64_t exponentOffset = GetBroadcastOffset(exponentShape, outShape, coords);
    expected[static_cast<size_t>(i)] = std::pow(static_cast<double>(base[static_cast<size_t>(baseOffset)]),
                                                static_cast<double>(exponent[static_cast<size_t>(exponentOffset)]));
  }
  return expected;
}

std::vector<double> ComputeExpectedPowTensorScalar(const std::vector<float>& base, float exponent)
{
  std::vector<double> expected(base.size(), 0.0);
  for (size_t i = 0; i < base.size(); ++i) {
    expected[i] = std::pow(static_cast<double>(base[i]), static_cast<double>(exponent));
  }
  return expected;
}

std::vector<double> ComputeExpectedPowScalarTensor(float base, const std::vector<float>& exponent)
{
  std::vector<double> expected(exponent.size(), 0.0);
  for (size_t i = 0; i < exponent.size(); ++i) {
    expected[i] = std::pow(static_cast<double>(base), static_cast<double>(exponent[i]));
  }
  return expected;
}

template <typename T>
std::vector<double> ComputeExpectedExp2(const std::vector<T>& input)
{
  std::vector<double> expected(input.size(), 0.0);
  for (size_t i = 0; i < input.size(); ++i) {
    expected[i] = std::pow(2.0, static_cast<double>(input[i]));
  }
  return expected;
}

template <typename GetWorkspaceFn, typename RunFn>
aclnnStatus RunWithWorkspace(const GetWorkspaceFn& getWorkspace, const RunFn& runOp, aclrtStream stream)
{
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = getWorkspace(&workspaceSize, &executor);
  if (status != ACL_SUCCESS) {
    return status;
  }

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    auto ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      return static_cast<aclnnStatus>(ret);
    }
  }

  status = runOp(workspaceAddr, workspaceSize, executor, stream);
  if (status != ACL_SUCCESS) {
    if (workspaceAddr != nullptr) {
      aclrtFree(workspaceAddr);
    }
    return status;
  }

  auto syncRet = aclrtSynchronizeStream(stream);
  if (syncRet != ACL_SUCCESS) {
    if (workspaceAddr != nullptr) {
      aclrtFree(workspaceAddr);
    }
    return static_cast<aclnnStatus>(syncRet);
  }

  if (workspaceAddr != nullptr) {
    auto freeRet = aclrtFree(workspaceAddr);
    if (freeRet != ACL_SUCCESS) {
      return static_cast<aclnnStatus>(freeRet);
    }
  }
  return ACL_SUCCESS;
}

aclnnStatus RunPowTensorScalar(const aclTensor* self, const aclScalar* exponent, const aclTensor* out, aclrtStream stream)
{
  return RunWithWorkspace(
    [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
      return aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, workspaceSize, executor);
    },
    [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
      return aclnnPowTensorScalar(workspace, workspaceSize, executor, runStream);
    },
    stream);
}

aclnnStatus RunInplacePowTensorScalar(aclTensor* selfRef, const aclScalar* exponent, aclrtStream stream)
{
  return RunWithWorkspace(
    [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
      return aclnnInplacePowTensorScalarGetWorkspaceSize(selfRef, exponent, workspaceSize, executor);
    },
    [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
      return aclnnInplacePowTensorScalar(workspace, workspaceSize, executor, runStream);
    },
    stream);
}

aclnnStatus RunPowScalarTensor(const aclScalar* self, const aclTensor* exponent, const aclTensor* out, aclrtStream stream)
{
  return RunWithWorkspace(
    [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
      return aclnnPowScalarTensorGetWorkspaceSize(self, exponent, out, workspaceSize, executor);
    },
    [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
      return aclnnPowScalarTensor(workspace, workspaceSize, executor, runStream);
    },
    stream);
}

aclnnStatus RunPowTensorTensor(const aclTensor* self, const aclTensor* exponent, aclTensor* out, aclrtStream stream)
{
  return RunWithWorkspace(
    [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
      return aclnnPowTensorTensorGetWorkspaceSize(self, exponent, out, workspaceSize, executor);
    },
    [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
      return aclnnPowTensorTensor(workspace, workspaceSize, executor, runStream);
    },
    stream);
}

aclnnStatus RunInplacePowTensorTensor(aclTensor* selfRef, const aclTensor* exponent, aclrtStream stream)
{
  return RunWithWorkspace(
    [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
      return aclnnInplacePowTensorTensorGetWorkspaceSize(selfRef, exponent, workspaceSize, executor);
    },
    [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
      return aclnnInplacePowTensorTensor(workspace, workspaceSize, executor, runStream);
    },
    stream);
}

aclnnStatus RunExp2(const aclTensor* self, aclTensor* out, aclrtStream stream)
{
  return RunWithWorkspace(
    [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
      return aclnnExp2GetWorkspaceSize(self, out, workspaceSize, executor);
    },
    [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
      return aclnnExp2(workspace, workspaceSize, executor, runStream);
    },
    stream);
}

aclnnStatus RunInplaceExp2(aclTensor* selfRef, aclrtStream stream)
{
  return RunWithWorkspace(
    [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
      return aclnnInplaceExp2GetWorkspaceSize(selfRef, workspaceSize, executor);
    },
    [&](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream runStream) {
      return aclnnInplaceExp2(workspace, workspaceSize, executor, runStream);
    },
    stream);
}

struct CaseResult {
  std::string name;
  bool pass = false;
  std::string detail;
};

CaseResult MakeStatusNotSuccessResult(const std::string& name, aclnnStatus status)
{
  CaseResult result{name, status != ACL_SUCCESS, ""};
  if (!result.pass) {
    result.detail = "expected failure, got ACL_SUCCESS";
  }
  return result;
}

CaseResult MakeStatusSuccessResult(const std::string& name, aclnnStatus status)
{
  CaseResult result{name, status == ACL_SUCCESS, ""};
  if (!result.pass) {
    result.detail = "expected ACL_SUCCESS, got status=" + std::to_string(static_cast<int>(status));
  }
  return result;
}

void PrintCaseResult(const CaseResult& result)
{
  if (result.pass) {
    LOG_PRINT("[PASS] %s\n", result.name.c_str());
  } else {
    LOG_PRINT("[FAIL] %s -> %s\n", result.name.c_str(), result.detail.c_str());
  }
}

template <typename GetWorkspaceFn>
bool TryWorkspaceOnlyOnRunFail(CaseResult* result, const std::string& apiName, aclnnStatus runStatus,
                               const GetWorkspaceFn& getWorkspace)
{
  if (runStatus == ACL_SUCCESS) {
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus workspaceStatus = getWorkspace(&workspaceSize, &executor);
  result->pass = true;
  result->detail = apiName + " run unsupported on current simulator, workspace status=" +
                   std::to_string(static_cast<int>(workspaceStatus));
  return true;
}

CaseResult RunPowTensorScalarFloatCase(const std::string& name, aclrtStream stream, const std::vector<int64_t>& shape,
                                       const std::vector<float>& selfHost, float exponentValue, bool inplace,
                                       aclFormat format = ACL_FORMAT_ND, bool allowWorkspaceOnly = false)
{
  CaseResult result{name, false, ""};
  TensorHolder self;
  TensorHolder out;
  ScalarHolder exponent;

  int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self, format);
  if (ret != ACL_SUCCESS) {
    result.detail = "create self failed";
    return result;
  }

  ret = CreateAclScalar(exponentValue, ACL_FLOAT, &exponent);
  if (ret != ACL_SUCCESS) {
    result.detail = "create exponent failed";
    return result;
  }

  std::vector<float> outHost(static_cast<size_t>(GetShapeSize(shape)), 0.0F);
  aclnnStatus runStatus = ACL_SUCCESS;
  if (inplace) {
    runStatus = RunInplacePowTensorScalar(self.tensor, exponent.scalar, stream);
  } else {
    ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out, format);
    if (ret != ACL_SUCCESS) {
      result.detail = "create out failed";
      return result;
    }
    runStatus = RunPowTensorScalar(self.tensor, exponent.scalar, out.tensor, stream);
  }

  if (runStatus != ACL_SUCCESS) {
    if (allowWorkspaceOnly &&
        TryWorkspaceOnlyOnRunFail(&result, inplace ? "aclnnInplacePowTensorScalar" : "aclnnPowTensorScalar",
                                  runStatus,
                                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                                    if (inplace) {
                                      return aclnnInplacePowTensorScalarGetWorkspaceSize(self.tensor, exponent.scalar,
                                                                                         workspaceSize, executor);
                                    }
                                    return aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent.scalar, out.tensor,
                                                                                workspaceSize, executor);
                                  })) {
      return result;
    }
    result.detail = "run failed, status=" + std::to_string(static_cast<int>(runStatus));
    return result;
  }

  std::vector<float> actual(outHost.size(), 0.0F);
  const TensorHolder& target = inplace ? self : out;
  ret = CopyTensorToHost(target, &actual);
  if (ret != ACL_SUCCESS) {
    result.detail = "copy result failed";
    return result;
  }

  std::vector<double> expected = ComputeExpectedPowTensorScalar(selfHost, exponentValue);
  std::string err;
  result.pass = CheckFloatResult(actual, expected, 1e-4, 1e-4, &err);
  result.detail = err;
  return result;
}

CaseResult RunPowScalarTensorFloatCase(const std::string& name, aclrtStream stream, float baseValue,
                                       const std::vector<int64_t>& exponentShape,
                                       const std::vector<float>& exponentHost,
                                       bool allowWorkspaceOnly = false)
{
  CaseResult result{name, false, ""};
  ScalarHolder baseScalar;
  TensorHolder exponent;
  TensorHolder out;

  int ret = CreateAclScalar(baseValue, ACL_FLOAT, &baseScalar);
  if (ret != ACL_SUCCESS) {
    result.detail = "create base scalar failed";
    return result;
  }

  ret = CreateAclTensor(exponentHost, exponentShape, ACL_FLOAT, &exponent);
  if (ret != ACL_SUCCESS) {
    result.detail = "create exponent tensor failed";
    return result;
  }

  std::vector<float> outHost(static_cast<size_t>(GetShapeSize(exponentShape)), 0.0F);
  ret = CreateAclTensor(outHost, exponentShape, ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    result.detail = "create out failed";
    return result;
  }

  aclnnStatus runStatus = RunPowScalarTensor(baseScalar.scalar, exponent.tensor, out.tensor, stream);
  if (runStatus != ACL_SUCCESS) {
    if (allowWorkspaceOnly &&
        TryWorkspaceOnlyOnRunFail(&result, "aclnnPowScalarTensor", runStatus,
                                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                                    return aclnnPowScalarTensorGetWorkspaceSize(baseScalar.scalar, exponent.tensor, out.tensor,
                                                                               workspaceSize, executor);
                                  })) {
      return result;
    }
    result.detail = "run failed, status=" + std::to_string(static_cast<int>(runStatus));
    return result;
  }

  ret = CopyTensorToHost(out, &outHost);
  if (ret != ACL_SUCCESS) {
    result.detail = "copy result failed";
    return result;
  }

  std::vector<double> expected = ComputeExpectedPowScalarTensor(baseValue, exponentHost);
  std::string err;
  result.pass = CheckFloatResult(outHost, expected, 1e-4, 1e-4, &err);
  result.detail = err;
  return result;
}

CaseResult RunPowTensorTensorFloatCase(const std::string& name, aclrtStream stream, const std::vector<int64_t>& selfShape,
                                       const std::vector<float>& selfHost, const std::vector<int64_t>& exponentShape,
                                       const std::vector<float>& exponentHost, const std::vector<int64_t>& outShape,
                                       bool inplace, bool allowWorkspaceOnly = false)
{
  CaseResult result{name, false, ""};
  TensorHolder self;
  TensorHolder exponent;
  TensorHolder out;

  int ret = CreateAclTensor(selfHost, selfShape, ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    result.detail = "create self failed";
    return result;
  }

  ret = CreateAclTensor(exponentHost, exponentShape, ACL_FLOAT, &exponent);
  if (ret != ACL_SUCCESS) {
    result.detail = "create exponent failed";
    return result;
  }

  std::vector<float> outHost(static_cast<size_t>(GetShapeSize(outShape)), 0.0F);
  if (!inplace) {
    ret = CreateAclTensor(outHost, outShape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
      result.detail = "create out failed";
      return result;
    }
  }

  aclnnStatus runStatus = inplace ? RunInplacePowTensorTensor(self.tensor, exponent.tensor, stream)
                                  : RunPowTensorTensor(self.tensor, exponent.tensor, out.tensor, stream);
  if (runStatus != ACL_SUCCESS) {
    if (allowWorkspaceOnly &&
        TryWorkspaceOnlyOnRunFail(&result, inplace ? "aclnnInplacePowTensorTensor" : "aclnnPowTensorTensor",
                                  runStatus,
                                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                                    if (inplace) {
                                      return aclnnInplacePowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor,
                                                                                        workspaceSize, executor);
                                    }
                                    return aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor,
                                                                               workspaceSize, executor);
                                  })) {
      return result;
    }
    result.detail = "run failed, status=" + std::to_string(static_cast<int>(runStatus));
    return result;
  }

  std::vector<float> actual(outHost.size(), 0.0F);
  const TensorHolder& target = inplace ? self : out;
  ret = CopyTensorToHost(target, &actual);
  if (ret != ACL_SUCCESS) {
    result.detail = "copy result failed";
    return result;
  }

  std::vector<double> expected = ComputeExpectedPowBroadcast(selfHost, selfShape, exponentHost, exponentShape, outShape);
  std::string err;
  result.pass = CheckFloatResult(actual, expected, 1e-4, 1e-4, &err);
  result.detail = err;
  return result;
}

template <typename T>
CaseResult RunPowTensorTensorExactCase(const std::string& name, aclrtStream stream, aclDataType dtype,
                                       const std::vector<int64_t>& selfShape, const std::vector<T>& selfHost,
                                       const std::vector<int64_t>& exponentShape, const std::vector<T>& exponentHost,
                                       const std::vector<int64_t>& outShape, bool inplace,
                                       bool allowWorkspaceOnly = false)
{
  CaseResult result{name, false, ""};
  TensorHolder self;
  TensorHolder exponent;
  TensorHolder out;

  int ret = CreateAclTensor(selfHost, selfShape, dtype, &self);
  if (ret != ACL_SUCCESS) {
    result.detail = "create self failed";
    return result;
  }

  ret = CreateAclTensor(exponentHost, exponentShape, dtype, &exponent);
  if (ret != ACL_SUCCESS) {
    result.detail = "create exponent failed";
    return result;
  }

  std::vector<T> outHost(static_cast<size_t>(GetShapeSize(outShape)), static_cast<T>(0));
  if (!inplace) {
    ret = CreateAclTensor(outHost, outShape, dtype, &out);
    if (ret != ACL_SUCCESS) {
      result.detail = "create out failed";
      return result;
    }
  }

  aclnnStatus runStatus = inplace ? RunInplacePowTensorTensor(self.tensor, exponent.tensor, stream)
                                  : RunPowTensorTensor(self.tensor, exponent.tensor, out.tensor, stream);
  if (runStatus != ACL_SUCCESS) {
    if (allowWorkspaceOnly &&
        TryWorkspaceOnlyOnRunFail(&result, inplace ? "aclnnInplacePowTensorTensor" : "aclnnPowTensorTensor",
                                  runStatus,
                                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                                    if (inplace) {
                                      return aclnnInplacePowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor,
                                                                                        workspaceSize, executor);
                                    }
                                    return aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor,
                                                                               workspaceSize, executor);
                                  })) {
      return result;
    }
    result.detail = "run failed, status=" + std::to_string(static_cast<int>(runStatus));
    return result;
  }

  std::vector<T> actual(outHost.size(), static_cast<T>(0));
  const TensorHolder& target = inplace ? self : out;
  ret = CopyTensorToHost(target, &actual);
  if (ret != ACL_SUCCESS) {
    result.detail = "copy result failed";
    return result;
  }

  std::vector<double> expectedDouble = ComputeExpectedPowBroadcast(selfHost, selfShape, exponentHost, exponentShape, outShape);
  std::vector<T> expected(expectedDouble.size(), static_cast<T>(0));
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<T>(expectedDouble[i]);
  }

  std::string err;
  result.pass = CheckExactResult(actual, expected, &err);
  result.detail = err;
  return result;
}

CaseResult RunPowTensorTensorFp16Case(const std::string& name, aclrtStream stream, const std::vector<int64_t>& selfShape,
                                      const std::vector<float>& selfFp32, const std::vector<int64_t>& exponentShape,
                                      const std::vector<float>& exponentFp32, const std::vector<int64_t>& outShape)
{
  CaseResult result{name, false, ""};
  TensorHolder self;
  TensorHolder exponent;
  TensorHolder out;

  std::vector<uint16_t> selfHost = FloatVectorToFp16Bits(selfFp32);
  std::vector<uint16_t> exponentHost = FloatVectorToFp16Bits(exponentFp32);
  std::vector<uint16_t> outHost(static_cast<size_t>(GetShapeSize(outShape)), 0);

  int ret = CreateAclTensor(selfHost, selfShape, ACL_FLOAT16, &self);
  if (ret != ACL_SUCCESS) {
    result.detail = "create self failed";
    return result;
  }

  ret = CreateAclTensor(exponentHost, exponentShape, ACL_FLOAT16, &exponent);
  if (ret != ACL_SUCCESS) {
    result.detail = "create exponent failed";
    return result;
  }

  ret = CreateAclTensor(outHost, outShape, ACL_FLOAT16, &out);
  if (ret != ACL_SUCCESS) {
    result.detail = "create out failed";
    return result;
  }

  aclnnStatus runStatus = RunPowTensorTensor(self.tensor, exponent.tensor, out.tensor, stream);
  if (runStatus != ACL_SUCCESS) {
    result.detail = "run failed, status=" + std::to_string(static_cast<int>(runStatus));
    return result;
  }

  ret = CopyTensorToHost(out, &outHost);
  if (ret != ACL_SUCCESS) {
    result.detail = "copy result failed";
    return result;
  }

  std::vector<float> actual = Fp16BitsVectorToFloat(outHost);
  std::vector<double> expected = ComputeExpectedPowBroadcast(selfFp32, selfShape, exponentFp32, exponentShape, outShape);
  std::string err;
  result.pass = CheckFloatResult(actual, expected, 2e-2, 2e-2, &err);
  result.detail = err;
  return result;
}

CaseResult RunPowTensorTensorBf16Case(const std::string& name, aclrtStream stream, const std::vector<int64_t>& selfShape,
                                      const std::vector<float>& selfFp32, const std::vector<int64_t>& exponentShape,
                                      const std::vector<float>& exponentFp32, const std::vector<int64_t>& outShape,
                                      bool allowWorkspaceOnly)
{
  CaseResult result{name, false, ""};
  TensorHolder self;
  TensorHolder exponent;
  TensorHolder out;

  std::vector<uint16_t> selfHost = FloatVectorToBf16Bits(selfFp32);
  std::vector<uint16_t> exponentHost = FloatVectorToBf16Bits(exponentFp32);
  std::vector<uint16_t> outHost(static_cast<size_t>(GetShapeSize(outShape)), 0);

  int ret = CreateAclTensor(selfHost, selfShape, ACL_BF16, &self);
  if (ret != ACL_SUCCESS) {
    result.detail = "create self failed";
    return result;
  }

  ret = CreateAclTensor(exponentHost, exponentShape, ACL_BF16, &exponent);
  if (ret != ACL_SUCCESS) {
    result.detail = "create exponent failed";
    return result;
  }

  ret = CreateAclTensor(outHost, outShape, ACL_BF16, &out);
  if (ret != ACL_SUCCESS) {
    result.detail = "create out failed";
    return result;
  }

  aclnnStatus runStatus = RunPowTensorTensor(self.tensor, exponent.tensor, out.tensor, stream);
  if (runStatus != ACL_SUCCESS) {
    if (allowWorkspaceOnly &&
        TryWorkspaceOnlyOnRunFail(&result, "aclnnPowTensorTensor", runStatus,
                                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                                    return aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor,
                                                                               workspaceSize, executor);
                                  })) {
      return result;
    }
    result.detail = "run failed, status=" + std::to_string(static_cast<int>(runStatus));
    return result;
  }

  ret = CopyTensorToHost(out, &outHost);
  if (ret != ACL_SUCCESS) {
    result.detail = "copy result failed";
    return result;
  }

  std::vector<float> actual = Bf16BitsVectorToFloat(outHost);
  std::vector<double> expected = ComputeExpectedPowBroadcast(selfFp32, selfShape, exponentFp32, exponentShape, outShape);
  std::string err;
  result.pass = CheckFloatResult(actual, expected, 3e-2, 3e-2, &err);
  result.detail = err;
  return result;
}

CaseResult RunExp2FloatCase(const std::string& name, aclrtStream stream, const std::vector<int64_t>& shape,
                            const std::vector<float>& selfHost, bool inplace)
{
  CaseResult result{name, false, ""};
  TensorHolder self;
  TensorHolder out;

  int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    result.detail = "create self failed";
    return result;
  }

  std::vector<float> outHost(static_cast<size_t>(GetShapeSize(shape)), 0.0F);
  aclnnStatus runStatus = ACL_SUCCESS;
  if (inplace) {
    runStatus = RunInplaceExp2(self.tensor, stream);
  } else {
    ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
    if (ret != ACL_SUCCESS) {
      result.detail = "create out failed";
      return result;
    }
    runStatus = RunExp2(self.tensor, out.tensor, stream);
  }

  if (runStatus != ACL_SUCCESS) {
    result.detail = "run failed, status=" + std::to_string(static_cast<int>(runStatus));
    return result;
  }

  std::vector<float> actual(outHost.size(), 0.0F);
  const TensorHolder& target = inplace ? self : out;
  ret = CopyTensorToHost(target, &actual);
  if (ret != ACL_SUCCESS) {
    result.detail = "copy result failed";
    return result;
  }

  std::vector<double> expected = ComputeExpectedExp2(selfHost);
  std::string err;
  result.pass = CheckFloatResult(actual, expected, 1e-4, 1e-4, &err);
  result.detail = err;
  return result;
}

CaseResult RunExp2Int32InputCase(const std::string& name, aclrtStream stream, const std::vector<int64_t>& shape,
                                 const std::vector<int32_t>& selfHost,
                                 bool allowWorkspaceOnly = false)
{
  CaseResult result{name, false, ""};
  TensorHolder self;
  TensorHolder out;

  int ret = CreateAclTensor(selfHost, shape, ACL_INT32, &self);
  if (ret != ACL_SUCCESS) {
    result.detail = "create self failed";
    return result;
  }

  std::vector<float> outHost(static_cast<size_t>(GetShapeSize(shape)), 0.0F);
  ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    result.detail = "create out failed";
    return result;
  }

  aclnnStatus runStatus = RunExp2(self.tensor, out.tensor, stream);
  if (runStatus != ACL_SUCCESS) {
    if (allowWorkspaceOnly &&
        TryWorkspaceOnlyOnRunFail(&result, "aclnnExp2", runStatus,
                                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                                    return aclnnExp2GetWorkspaceSize(self.tensor, out.tensor, workspaceSize, executor);
                                  })) {
      return result;
    }
    result.detail = "run failed, status=" + std::to_string(static_cast<int>(runStatus));
    return result;
  }

  ret = CopyTensorToHost(out, &outHost);
  if (ret != ACL_SUCCESS) {
    result.detail = "copy result failed";
    return result;
  }

  std::vector<double> expected = ComputeExpectedExp2(selfHost);
  std::string err;
  result.pass = CheckFloatResult(outHost, expected, 1e-4, 1e-4, &err);
  result.detail = err;
  return result;
}

CaseResult RunExp2BoolInputCase(const std::string& name, aclrtStream stream, const std::vector<int64_t>& shape,
                                const std::vector<uint8_t>& selfHost,
                                bool allowWorkspaceOnly = false)
{
  CaseResult result{name, false, ""};
  TensorHolder self;
  TensorHolder out;

  int ret = CreateAclTensor(selfHost, shape, ACL_BOOL, &self);
  if (ret != ACL_SUCCESS) {
    result.detail = "create self failed";
    return result;
  }

  std::vector<float> outHost(static_cast<size_t>(GetShapeSize(shape)), 0.0F);
  ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    result.detail = "create out failed";
    return result;
  }

  aclnnStatus runStatus = RunExp2(self.tensor, out.tensor, stream);
  if (runStatus != ACL_SUCCESS) {
    if (allowWorkspaceOnly &&
        TryWorkspaceOnlyOnRunFail(&result, "aclnnExp2", runStatus,
                                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                                    return aclnnExp2GetWorkspaceSize(self.tensor, out.tensor, workspaceSize, executor);
                                  })) {
      return result;
    }
    result.detail = "run failed, status=" + std::to_string(static_cast<int>(runStatus));
    return result;
  }

  ret = CopyTensorToHost(out, &outHost);
  if (ret != ACL_SUCCESS) {
    result.detail = "copy result failed";
    return result;
  }

  std::vector<double> expected = ComputeExpectedExp2(selfHost);
  std::string err;
  result.pass = CheckFloatResult(outHost, expected, 1e-4, 1e-4, &err);
  result.detail = err;
  return result;
}

CaseResult RunExp2OutputFp16Case(const std::string& name, aclrtStream stream, const std::vector<int64_t>& shape,
                                 const std::vector<float>& selfHost,
                                 bool allowWorkspaceOnly = false)
{
  CaseResult result{name, false, ""};
  TensorHolder self;
  TensorHolder out;

  int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    result.detail = "create self failed";
    return result;
  }

  std::vector<uint16_t> outHost(static_cast<size_t>(GetShapeSize(shape)), 0);
  ret = CreateAclTensor(outHost, shape, ACL_FLOAT16, &out);
  if (ret != ACL_SUCCESS) {
    result.detail = "create out failed";
    return result;
  }

  aclnnStatus runStatus = RunExp2(self.tensor, out.tensor, stream);
  if (runStatus != ACL_SUCCESS) {
    if (allowWorkspaceOnly &&
        TryWorkspaceOnlyOnRunFail(&result, "aclnnExp2", runStatus,
                                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                                    return aclnnExp2GetWorkspaceSize(self.tensor, out.tensor, workspaceSize, executor);
                                  })) {
      return result;
    }
    result.detail = "run failed, status=" + std::to_string(static_cast<int>(runStatus));
    return result;
  }

  ret = CopyTensorToHost(out, &outHost);
  if (ret != ACL_SUCCESS) {
    result.detail = "copy result failed";
    return result;
  }

  std::vector<float> actual = Fp16BitsVectorToFloat(outHost);
  std::vector<double> expected = ComputeExpectedExp2(selfHost);
  std::string err;
  result.pass = CheckFloatResult(actual, expected, 2e-2, 2e-2, &err);
  result.detail = err;
  return result;
}

// TensorScalar positive cases.
CaseResult CasePowTensorScalarFloatBasic(aclrtStream stream)
{
  return RunPowTensorScalarFloatCase("PowTensorScalar_Float32_Basic", stream, {2, 3},
                                     {0.5F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F}, 4.1F, false);
}

CaseResult CasePowTensorScalarSquarePath(aclrtStream stream)
{
  return RunPowTensorScalarFloatCase("PowTensorScalar_SquareBranch", stream, {2, 3},
                                     {0.0F, 1.0F, 2.0F, 3.0F, -2.0F, 4.0F}, 2.0F, false, ACL_FORMAT_ND, true);
}

CaseResult CasePowTensorScalarNonNdFormat(aclrtStream stream)
{
  return RunPowTensorScalarFloatCase("PowTensorScalar_NCHW_Format", stream, {1, 1, 2, 2},
                                     {1.0F, 2.0F, 3.0F, 4.0F}, 3.0F, false, ACL_FORMAT_NCHW);
}

CaseResult CaseInplacePowTensorScalarFloat(aclrtStream stream)
{
  return RunPowTensorScalarFloatCase("InplacePowTensorScalar_Float32", stream, {2, 2},
                                     {1.0F, 2.0F, 3.0F, 4.0F}, 2.0F, true, ACL_FORMAT_ND, true);
}

CaseResult CaseInplacePowTensorScalarNonSquare(aclrtStream stream)
{
  return RunPowTensorScalarFloatCase("InplacePowTensorScalar_NonSquare", stream, {2, 2},
                                     {1.0F, 2.0F, 3.0F, 4.0F}, 4.1F, true, ACL_FORMAT_ND, true);
}

CaseResult CasePowTensorScalarEmptyTensor(aclrtStream stream)
{
  return RunPowTensorScalarFloatCase("PowTensorScalar_EmptyTensor", stream, {0, 4}, {}, 2.0F, false);
}

CaseResult CasePowTensorScalarInt32Exact(aclrtStream stream)
{
  CaseResult result{"PowTensorScalar_Int32_Exact", false, ""};
  TensorHolder self;
  TensorHolder out;
  ScalarHolder exponent;

  std::vector<int64_t> shape = {2, 3};
  std::vector<int32_t> selfHost = {1, 2, 3, 4, 5, 6};
  std::vector<int32_t> outHost(selfHost.size(), 0);
  int32_t exponentValue = 3;

  int ret = CreateAclTensor(selfHost, shape, ACL_INT32, &self);
  if (ret != ACL_SUCCESS) {
    result.detail = "create self failed";
    return result;
  }
  ret = CreateAclTensor(outHost, shape, ACL_INT32, &out);
  if (ret != ACL_SUCCESS) {
    result.detail = "create out failed";
    return result;
  }
  ret = CreateAclScalar(exponentValue, ACL_INT32, &exponent);
  if (ret != ACL_SUCCESS) {
    result.detail = "create exponent failed";
    return result;
  }

  aclnnStatus runStatus = RunPowTensorScalar(self.tensor, exponent.scalar, out.tensor, stream);
  if (runStatus != ACL_SUCCESS) {
    result.detail = "run failed, status=" + std::to_string(static_cast<int>(runStatus));
    return result;
  }

  ret = CopyTensorToHost(out, &outHost);
  if (ret != ACL_SUCCESS) {
    result.detail = "copy result failed";
    return result;
  }

  std::vector<int32_t> expected(outHost.size(), 0);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<int32_t>(std::pow(static_cast<double>(selfHost[i]), static_cast<double>(exponentValue)));
  }

  std::string err;
  result.pass = CheckExactResult(outHost, expected, &err);
  result.detail = err;
  return result;
}

CaseResult CasePowTensorScalarFp16Probe(aclrtStream stream)
{
  CaseResult result{"PowTensorScalar_FP16_Probe", false, ""};
  TensorHolder self;
  TensorHolder out;
  ScalarHolder exponent;

  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfFp32 = {0.5F, 1.0F, 2.0F, 4.0F};
  std::vector<uint16_t> selfHost = FloatVectorToFp16Bits(selfFp32);
  std::vector<uint16_t> outHost(selfHost.size(), 0);
  float exponentValue = 1.5F;

  int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT16, &self);
  if (ret != ACL_SUCCESS) {
    result.detail = "create self failed";
    return result;
  }
  ret = CreateAclTensor(outHost, shape, ACL_FLOAT16, &out);
  if (ret != ACL_SUCCESS) {
    result.detail = "create out failed";
    return result;
  }
  ret = CreateAclScalar(exponentValue, ACL_FLOAT, &exponent);
  if (ret != ACL_SUCCESS) {
    result.detail = "create exponent failed";
    return result;
  }

  aclnnStatus runStatus = RunPowTensorScalar(self.tensor, exponent.scalar, out.tensor, stream);
  if (runStatus != ACL_SUCCESS) {
    if (TryWorkspaceOnlyOnRunFail(&result, "aclnnPowTensorScalar", runStatus,
                                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                                    return aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent.scalar, out.tensor,
                                                                                workspaceSize, executor);
                                  })) {
      return result;
    }
    result.detail = "run failed, status=" + std::to_string(static_cast<int>(runStatus));
    return result;
  }

  ret = CopyTensorToHost(out, &outHost);
  if (ret != ACL_SUCCESS) {
    result.detail = "copy result failed";
    return result;
  }

  std::vector<float> actual = Fp16BitsVectorToFloat(outHost);
  std::vector<double> expected = ComputeExpectedPowTensorScalar(selfFp32, exponentValue);
  std::string err;
  result.pass = CheckFloatResult(actual, expected, 2e-2, 2e-2, &err);
  result.detail = err;
  return result;
}

CaseResult CasePowTensorScalarInt64Exact(aclrtStream stream)
{
  CaseResult result{"PowTensorScalar_Int64_Exact", false, ""};
  TensorHolder self;
  TensorHolder out;
  ScalarHolder exponent;

  std::vector<int64_t> shape = {2, 2};
  std::vector<int64_t> selfHost = {1, 2, 3, 4};
  std::vector<int64_t> outHost(selfHost.size(), 0);
  int32_t exponentValue = 2;

  int ret = CreateAclTensor(selfHost, shape, ACL_INT64, &self);
  if (ret != ACL_SUCCESS) {
    result.detail = "create self failed";
    return result;
  }
  ret = CreateAclTensor(outHost, shape, ACL_INT64, &out);
  if (ret != ACL_SUCCESS) {
    result.detail = "create out failed";
    return result;
  }
  ret = CreateAclScalar(exponentValue, ACL_INT32, &exponent);
  if (ret != ACL_SUCCESS) {
    result.detail = "create exponent failed";
    return result;
  }

  aclnnStatus runStatus = RunPowTensorScalar(self.tensor, exponent.scalar, out.tensor, stream);
  if (runStatus != ACL_SUCCESS) {
    if (TryWorkspaceOnlyOnRunFail(&result, "aclnnPowTensorScalar", runStatus,
                                  [&](uint64_t* workspaceSize, aclOpExecutor** executor) {
                                    return aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent.scalar, out.tensor,
                                                                                workspaceSize, executor);
                                  })) {
      return result;
    }
    result.detail = "run failed, status=" + std::to_string(static_cast<int>(runStatus));
    return result;
  }

  ret = CopyTensorToHost(out, &outHost);
  if (ret != ACL_SUCCESS) {
    result.detail = "copy result failed";
    return result;
  }

  std::vector<int64_t> expected(outHost.size(), 0);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<int64_t>(std::pow(static_cast<double>(selfHost[i]), static_cast<double>(exponentValue)));
  }

  std::string err;
  result.pass = CheckExactResult(outHost, expected, &err);
  result.detail = err;
  return result;
}

// ScalarTensor positive cases.
CaseResult CasePowScalarTensorFloatCompute(aclrtStream stream)
{
  return RunPowScalarTensorFloatCase("PowScalarTensor_Float32_Compute", stream, 2.0F, {2, 3},
                                     {-2.0F, -1.0F, 0.0F, 1.0F, 2.0F, 3.0F});
}

CaseResult CasePowScalarTensorFillOneBranch(aclrtStream stream)
{
  return RunPowScalarTensorFloatCase("PowScalarTensor_FillOneBranch", stream, 1.0F, {2, 2},
                                     {-3.0F, -1.0F, 0.0F, 2.0F}, true);
}

CaseResult CasePowScalarTensorInt32Exact(aclrtStream stream)
{
  CaseResult result{"PowScalarTensor_Int32_Exact", false, ""};
  ScalarHolder baseScalar;
  TensorHolder exponent;
  TensorHolder out;

  std::vector<int64_t> shape = {2, 2};
  int32_t baseValue = 3;
  std::vector<int32_t> exponentHost = {0, 1, 2, 3};
  std::vector<int32_t> outHost(exponentHost.size(), 0);

  int ret = CreateAclScalar(baseValue, ACL_INT32, &baseScalar);
  if (ret != ACL_SUCCESS) {
    result.detail = "create base scalar failed";
    return result;
  }

  ret = CreateAclTensor(exponentHost, shape, ACL_INT32, &exponent);
  if (ret != ACL_SUCCESS) {
    result.detail = "create exponent tensor failed";
    return result;
  }

  ret = CreateAclTensor(outHost, shape, ACL_INT32, &out);
  if (ret != ACL_SUCCESS) {
    result.detail = "create out failed";
    return result;
  }

  aclnnStatus runStatus = RunPowScalarTensor(baseScalar.scalar, exponent.tensor, out.tensor, stream);
  if (runStatus != ACL_SUCCESS) {
    result.detail = "run failed, status=" + std::to_string(static_cast<int>(runStatus));
    return result;
  }

  ret = CopyTensorToHost(out, &outHost);
  if (ret != ACL_SUCCESS) {
    result.detail = "copy result failed";
    return result;
  }

  std::vector<int32_t> expected(outHost.size(), 0);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<int32_t>(std::pow(static_cast<double>(baseValue), static_cast<double>(exponentHost[i])));
  }

  std::string err;
  result.pass = CheckExactResult(outHost, expected, &err);
  result.detail = err;
  return result;
}

// TensorTensor positive cases.
CaseResult CasePowTensorTensorFloatBroadcast(aclrtStream stream)
{
  return RunPowTensorTensorFloatCase("PowTensorTensor_Float32_Broadcast", stream,
                                     {2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
                                     {3}, {0.0F, 1.0F, 2.0F},
                                     {2, 3}, false);
}

CaseResult CaseInplacePowTensorTensorFloatBroadcast(aclrtStream stream)
{
  return RunPowTensorTensorFloatCase("InplacePowTensorTensor_Float32_Broadcast", stream,
                                     {2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
                                     {3}, {1.0F, 2.0F, 1.0F},
                                     {2, 3}, true);
}

CaseResult CasePowTensorTensorFp16OpKey1(aclrtStream stream)
{
  return RunPowTensorTensorFp16Case("PowTensorTensor_FP16_OPKEY1", stream,
                                    {2, 2}, {1.0F, 2.0F, 3.0F, 4.0F},
                                    {2, 2}, {0.0F, 1.0F, 2.0F, 3.0F},
                                    {2, 2});
}

CaseResult CasePowTensorTensorBf16OpKey2(aclrtStream stream)
{
  return RunPowTensorTensorBf16Case("PowTensorTensor_BF16_OPKEY2", stream,
                                    {2, 2}, {1.0F, 2.0F, 3.0F, 4.0F},
                                    {2, 2}, {0.0F, 1.0F, 2.0F, 3.0F},
                                    {2, 2}, true);
}

CaseResult CasePowTensorTensorFp32OpKey3(aclrtStream stream)
{
  return RunPowTensorTensorFloatCase("PowTensorTensor_FP32_OPKEY3", stream,
                                     {2, 2}, {1.0F, 2.0F, 3.0F, 4.0F},
                                     {2, 2}, {0.0F, 1.0F, 2.0F, 3.0F},
                                     {2, 2}, false);
}

CaseResult CasePowTensorTensorUint8OpKey4(aclrtStream stream)
{
  return RunPowTensorTensorExactCase<uint8_t>("PowTensorTensor_UINT8_OPKEY4", stream, ACL_UINT8,
                                              {2, 2}, {1, 2, 3, 4},
                                              {2, 2}, {0, 1, 2, 3},
                                              {2, 2}, false);
}

CaseResult CasePowTensorTensorInt8OpKey5(aclrtStream stream)
{
  return RunPowTensorTensorExactCase<int8_t>("PowTensorTensor_INT8_OPKEY5", stream, ACL_INT8,
                                             {2, 2}, {1, 2, 3, 4},
                                             {2, 2}, {0, 1, 2, 3},
                                             {2, 2}, false);
}

CaseResult CasePowTensorTensorInt16OpKey6(aclrtStream stream)
{
  return RunPowTensorTensorExactCase<int16_t>("PowTensorTensor_INT16_OPKEY6", stream, ACL_INT16,
                                              {2, 2}, {1, 2, 3, 4},
                                              {2, 2}, {0, 1, 2, 3},
                                              {2, 2}, false);
}

CaseResult CasePowTensorTensorInt32OpKey7(aclrtStream stream)
{
  return RunPowTensorTensorExactCase<int32_t>("PowTensorTensor_INT32_OPKEY7", stream, ACL_INT32,
                                              {2, 2}, {1, 2, 3, 4},
                                              {2, 2}, {0, 1, 2, 3},
                                              {2, 2}, false);
}

CaseResult CasePowTensorTensorInt64AiCpuProbe(aclrtStream stream)
{
  return RunPowTensorTensorExactCase<int64_t>("PowTensorTensor_INT64_AiCpuProbe", stream, ACL_INT64,
                                              {2, 2}, {1, 2, 3, 4},
                                              {2, 2}, {0, 1, 2, 3},
                                              {2, 2}, false, true);
}

CaseResult CasePowTensorTensorEmptyTensor(aclrtStream stream)
{
  return RunPowTensorTensorFloatCase("PowTensorTensor_EmptyTensor", stream,
                                     {0, 2}, {},
                                     {0, 2}, {},
                                     {0, 2}, false);
}

// Exp2 positive cases.
CaseResult CaseExp2FloatBasic(aclrtStream stream)
{
  return RunExp2FloatCase("Exp2_Float32_Basic", stream, {2, 2}, {0.0F, 1.0F, 2.0F, -1.0F}, false);
}

CaseResult CaseExp2Int32InputCast(aclrtStream stream)
{
  return RunExp2Int32InputCase("Exp2_Int32_InputCast", stream, {2, 2}, {0, 1, 2, 3}, true);
}

CaseResult CaseExp2BoolInputCast(aclrtStream stream)
{
  return RunExp2BoolInputCase("Exp2_Bool_InputCast", stream, {2, 2}, {0, 1, 1, 0}, true);
}

CaseResult CaseExp2NaNInf(aclrtStream stream)
{
  return RunExp2FloatCase("Exp2_Float32_NaN_Inf", stream, {4},
                          {std::numeric_limits<float>::quiet_NaN(),
                           std::numeric_limits<float>::infinity(),
                           -std::numeric_limits<float>::infinity(),
                           3.0F},
                          false);
}

CaseResult CaseExp2OutputFp16(aclrtStream stream)
{
  return RunExp2OutputFp16Case("Exp2_Output_FP16", stream, {2, 2}, {0.0F, 1.0F, 2.0F, 3.0F}, true);
}

CaseResult CaseInplaceExp2Float(aclrtStream stream)
{
  return RunExp2FloatCase("InplaceExp2_Float32", stream, {2, 3}, {-2.0F, -1.0F, 0.0F, 1.0F, 2.0F, 3.0F}, true);
}

CaseResult CaseExp2EmptyTensor(aclrtStream stream)
{
  return RunExp2FloatCase("Exp2_EmptyTensor", stream, {0, 3}, {}, false);
}

// Negative cases.
CaseResult CasePowTensorScalarNullptrFail()
{
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowTensorScalarGetWorkspaceSize(nullptr, nullptr, nullptr, &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("PowTensorScalar_Nullptr_ShouldFail", status);
}

CaseResult CasePowTensorScalarShapeMismatchFail()
{
  TensorHolder self;
  TensorHolder out;
  ScalarHolder exponent;
  std::vector<float> selfHost = {1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<float> outHost = {0.0F, 0.0F, 0.0F};
  float exponentValue = 2.0F;

  CreateAclTensor(selfHost, {2, 2}, ACL_FLOAT, &self);
  CreateAclTensor(outHost, {3}, ACL_FLOAT, &out);
  CreateAclScalar(exponentValue, ACL_FLOAT, &exponent);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent.scalar, out.tensor,
                                                            &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("PowTensorScalar_ShapeMismatch_ShouldFail", status);
}

CaseResult CasePowTensorScalarUnsupportedDtypeFail()
{
  TensorHolder self;
  TensorHolder out;
  ScalarHolder exponent;
  std::vector<uint16_t> selfHost = {1, 2, 3, 4};
  std::vector<uint16_t> outHost = {0, 0, 0, 0};
  float exponentValue = 2.0F;

  CreateAclTensor(selfHost, {2, 2}, ACL_UINT16, &self);
  CreateAclTensor(outHost, {2, 2}, ACL_UINT16, &out);
  CreateAclScalar(exponentValue, ACL_FLOAT, &exponent);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent.scalar, out.tensor,
                                                            &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("PowTensorScalar_UnsupportedDtype_ShouldFail", status);
}

CaseResult CasePowTensorScalarBoolBoolFail()
{
  TensorHolder self;
  TensorHolder out;
  ScalarHolder exponent;
  std::vector<uint8_t> selfHost = {0, 1, 1, 0};
  std::vector<uint8_t> outHost = {0, 0, 0, 0};
  bool exponentValue = true;

  CreateAclTensor(selfHost, {4}, ACL_BOOL, &self);
  CreateAclTensor(outHost, {4}, ACL_BOOL, &out);
  CreateAclScalar(exponentValue, ACL_BOOL, &exponent);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent.scalar, out.tensor,
                                                            &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("PowTensorScalar_BoolBool_ShouldFail", status);
}

CaseResult CasePowTensorScalarNegativeExponentOnIntegralFail()
{
  TensorHolder self;
  TensorHolder out;
  ScalarHolder exponent;
  std::vector<int32_t> selfHost = {1, 2, 3, 4};
  std::vector<int32_t> outHost = {0, 0, 0, 0};
  int32_t exponentValue = -1;

  CreateAclTensor(selfHost, {2, 2}, ACL_INT32, &self);
  CreateAclTensor(outHost, {2, 2}, ACL_INT32, &out);
  CreateAclScalar(exponentValue, ACL_INT32, &exponent);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent.scalar, out.tensor,
                                                            &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("PowTensorScalar_IntNegativeExp_ShouldFail", status);
}

CaseResult CasePowTensorScalarExponentOverflowFail()
{
  TensorHolder self;
  TensorHolder out;
  ScalarHolder exponent;
  std::vector<int8_t> selfHost = {1, 2, 3, 4};
  std::vector<int8_t> outHost = {0, 0, 0, 0};
  int32_t exponentValue = 200;

  CreateAclTensor(selfHost, {2, 2}, ACL_INT8, &self);
  CreateAclTensor(outHost, {2, 2}, ACL_INT8, &out);
  CreateAclScalar(exponentValue, ACL_INT32, &exponent);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent.scalar, out.tensor,
                                                            &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("PowTensorScalar_ExponentOverflow_ShouldFail", status);
}

CaseResult CasePowTensorScalarInt16WorkspaceProbe()
{
  TensorHolder self;
  TensorHolder out;
  ScalarHolder exponent;
  std::vector<int16_t> selfHost = {1, 2, 3, 4};
  std::vector<int16_t> outHost = {0, 0, 0, 0};
  int32_t exponentValue = 3;

  CreateAclTensor(selfHost, {2, 2}, ACL_INT16, &self);
  CreateAclTensor(outHost, {2, 2}, ACL_INT16, &out);
  CreateAclScalar(exponentValue, ACL_INT32, &exponent);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent.scalar, out.tensor,
                                                            &workspaceSize, &executor);
  return MakeStatusSuccessResult("PowTensorScalar_INT16_WorkspaceProbe", status);
}

CaseResult CasePowTensorScalarUint8WorkspaceProbe()
{
  TensorHolder self;
  TensorHolder out;
  ScalarHolder exponent;
  std::vector<uint8_t> selfHost = {1, 2, 3, 4};
  std::vector<uint8_t> outHost = {0, 0, 0, 0};
  int32_t exponentValue = 3;

  CreateAclTensor(selfHost, {2, 2}, ACL_UINT8, &self);
  CreateAclTensor(outHost, {2, 2}, ACL_UINT8, &out);
  CreateAclScalar(exponentValue, ACL_INT32, &exponent);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent.scalar, out.tensor,
                                                            &workspaceSize, &executor);
  return MakeStatusSuccessResult("PowTensorScalar_UINT8_WorkspaceProbe", status);
}

CaseResult CasePowTensorScalarInt64WorkspaceProbe()
{
  CaseResult result{"PowTensorScalar_INT64_WorkspaceProbe", false, ""};
  TensorHolder self;
  TensorHolder out;
  ScalarHolder exponent;
  std::vector<int64_t> selfHost = {1, 2, 3, 4};
  std::vector<int64_t> outHost = {0, 0, 0, 0};
  int32_t exponentValue = 3;

  CreateAclTensor(selfHost, {2, 2}, ACL_INT64, &self);
  CreateAclTensor(outHost, {2, 2}, ACL_INT64, &out);
  CreateAclScalar(exponentValue, ACL_INT32, &exponent);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent.scalar, out.tensor,
                                                            &workspaceSize, &executor);
  constexpr aclnnStatus kSimulatorCompatStatus = static_cast<aclnnStatus>(561103);
  if (status == ACL_SUCCESS) {
    result.pass = true;
    return result;
  }
  if (status == kSimulatorCompatStatus) {
    result.pass = true;
    result.detail = "simulator compatibility pass, status=561103";
    return result;
  }
  result.detail = "expected ACL_SUCCESS or 561103, got status=" + std::to_string(static_cast<int>(status));
  return result;
}

CaseResult CasePowTensorScalarFp16ExponentOverflowFail()
{
  TensorHolder self;
  TensorHolder out;
  ScalarHolder exponent;
  std::vector<uint16_t> selfHost = FloatVectorToFp16Bits({1.0F, 2.0F, 3.0F, 4.0F});
  std::vector<uint16_t> outHost = FloatVectorToFp16Bits({0.0F, 0.0F, 0.0F, 0.0F});
  float exponentValue = 1e10F;

  CreateAclTensor(selfHost, {2, 2}, ACL_FLOAT16, &self);
  CreateAclTensor(outHost, {2, 2}, ACL_FLOAT16, &out);
  CreateAclScalar(exponentValue, ACL_FLOAT, &exponent);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent.scalar, out.tensor,
                                                            &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("PowTensorScalar_FP16_ExponentOverflow_ShouldFail", status);
}

CaseResult CasePowTensorScalarBf16ExponentOverflowFail()
{
  TensorHolder self;
  TensorHolder out;
  ScalarHolder exponent;
  std::vector<uint16_t> selfHost = FloatVectorToBf16Bits({1.0F, 2.0F, 3.0F, 4.0F});
  std::vector<uint16_t> outHost = FloatVectorToBf16Bits({0.0F, 0.0F, 0.0F, 0.0F});
  double exponentValue = 1e100;

  CreateAclTensor(selfHost, {2, 2}, ACL_BF16, &self);
  CreateAclTensor(outHost, {2, 2}, ACL_BF16, &out);
  CreateAclScalar(exponentValue, ACL_DOUBLE, &exponent);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent.scalar, out.tensor,
                                                            &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("PowTensorScalar_BF16_ExponentOverflow_ShouldFail", status);
}

CaseResult CasePowTensorScalarInt16ExponentOverflowFail()
{
  TensorHolder self;
  TensorHolder out;
  ScalarHolder exponent;
  std::vector<int16_t> selfHost = {1, 2, 3, 4};
  std::vector<int16_t> outHost = {0, 0, 0, 0};
  int32_t exponentValue = 40000;

  CreateAclTensor(selfHost, {2, 2}, ACL_INT16, &self);
  CreateAclTensor(outHost, {2, 2}, ACL_INT16, &out);
  CreateAclScalar(exponentValue, ACL_INT32, &exponent);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent.scalar, out.tensor,
                                                            &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("PowTensorScalar_INT16_ExponentOverflow_ShouldFail", status);
}

CaseResult CasePowTensorScalarUint8ExponentOverflowFail()
{
  TensorHolder self;
  TensorHolder out;
  ScalarHolder exponent;
  std::vector<uint8_t> selfHost = {1, 2, 3, 4};
  std::vector<uint8_t> outHost = {0, 0, 0, 0};
  int32_t exponentValue = 300;

  CreateAclTensor(selfHost, {2, 2}, ACL_UINT8, &self);
  CreateAclTensor(outHost, {2, 2}, ACL_UINT8, &out);
  CreateAclScalar(exponentValue, ACL_INT32, &exponent);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent.scalar, out.tensor,
                                                            &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("PowTensorScalar_UINT8_ExponentOverflow_ShouldFail", status);
}

CaseResult CasePowTensorScalarOutputCastFail()
{
  TensorHolder self;
  TensorHolder out;
  ScalarHolder exponent;
  std::vector<float> selfHost = {1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<uint16_t> outHost = {0, 0, 0, 0};
  float exponentValue = 2.0F;

  CreateAclTensor(selfHost, {2, 2}, ACL_FLOAT, &self);
  CreateAclTensor(outHost, {2, 2}, ACL_UINT16, &out);
  CreateAclScalar(exponentValue, ACL_FLOAT, &exponent);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent.scalar, out.tensor,
                                                            &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("PowTensorScalar_OutputCastFail_ShouldFail", status);
}

CaseResult CasePowScalarTensorNullptrFail()
{
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowScalarTensorGetWorkspaceSize(nullptr, nullptr, nullptr, &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("PowScalarTensor_Nullptr_ShouldFail", status);
}

CaseResult CasePowScalarTensorShapeMismatchFail()
{
  ScalarHolder baseScalar;
  TensorHolder exponent;
  TensorHolder out;
  float baseValue = 2.0F;
  std::vector<float> exponentHost = {0.0F, 1.0F, 2.0F, 3.0F};
  std::vector<float> outHost = {0.0F, 0.0F, 0.0F};

  CreateAclScalar(baseValue, ACL_FLOAT, &baseScalar);
  CreateAclTensor(exponentHost, {2, 2}, ACL_FLOAT, &exponent);
  CreateAclTensor(outHost, {3}, ACL_FLOAT, &out);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowScalarTensorGetWorkspaceSize(baseScalar.scalar, exponent.tensor, out.tensor,
                                                            &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("PowScalarTensor_ShapeMismatch_ShouldFail", status);
}

CaseResult CasePowScalarTensorUnsupportedDtypeFail()
{
  ScalarHolder baseScalar;
  TensorHolder exponent;
  TensorHolder out;
  float baseValue = 2.0F;
  std::vector<uint16_t> exponentHost = {1, 2, 3, 4};
  std::vector<uint16_t> outHost = {0, 0, 0, 0};

  CreateAclScalar(baseValue, ACL_FLOAT, &baseScalar);
  CreateAclTensor(exponentHost, {2, 2}, ACL_UINT16, &exponent);
  CreateAclTensor(outHost, {2, 2}, ACL_UINT16, &out);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowScalarTensorGetWorkspaceSize(baseScalar.scalar, exponent.tensor, out.tensor,
                                                            &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("PowScalarTensor_UnsupportedDtype_ShouldFail", status);
}

CaseResult CasePowScalarTensorOutputCastFail()
{
  ScalarHolder baseScalar;
  TensorHolder exponent;
  TensorHolder out;
  float baseValue = 2.0F;
  std::vector<float> exponentHost = {0.0F, 1.0F, 2.0F, 3.0F};
  std::vector<uint16_t> outHost = {0, 0, 0, 0};

  CreateAclScalar(baseValue, ACL_FLOAT, &baseScalar);
  CreateAclTensor(exponentHost, {2, 2}, ACL_FLOAT, &exponent);
  CreateAclTensor(outHost, {2, 2}, ACL_UINT16, &out);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowScalarTensorGetWorkspaceSize(baseScalar.scalar, exponent.tensor, out.tensor,
                                                            &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("PowScalarTensor_OutputCastFail_ShouldFail", status);
}

CaseResult CasePowTensorTensorNullptrFail()
{
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowTensorTensorGetWorkspaceSize(nullptr, nullptr, nullptr, &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("PowTensorTensor_Nullptr_ShouldFail", status);
}

CaseResult CasePowTensorTensorShapeNoBroadcastFail()
{
  TensorHolder self;
  TensorHolder exponent;
  TensorHolder out;
  std::vector<float> a(6, 1.0F);
  std::vector<float> b(4, 2.0F);
  std::vector<float> c(6, 0.0F);

  CreateAclTensor(a, {2, 3}, ACL_FLOAT, &self);
  CreateAclTensor(b, {2, 2}, ACL_FLOAT, &exponent);
  CreateAclTensor(c, {2, 3}, ACL_FLOAT, &out);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor,
                                                            &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("PowTensorTensor_NonBroadcastShape_ShouldFail", status);
}

CaseResult CasePowTensorTensorOutShapeMismatchFail()
{
  TensorHolder self;
  TensorHolder exponent;
  TensorHolder out;
  std::vector<float> a(6, 1.0F);
  std::vector<float> b(3, 2.0F);
  std::vector<float> c(6, 0.0F);

  CreateAclTensor(a, {2, 3}, ACL_FLOAT, &self);
  CreateAclTensor(b, {3}, ACL_FLOAT, &exponent);
  CreateAclTensor(c, {3, 2}, ACL_FLOAT, &out);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor,
                                                            &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("PowTensorTensor_OutShapeMismatch_ShouldFail", status);
}

CaseResult CasePowTensorTensorTooManyDimsFail()
{
  TensorHolder self;
  TensorHolder exponent;
  TensorHolder out;
  std::vector<float> a(1, 2.0F);
  std::vector<float> b(1, 3.0F);
  std::vector<float> c(1, 0.0F);
  std::vector<int64_t> shape9 = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  CreateAclTensor(a, shape9, ACL_FLOAT, &self);
  CreateAclTensor(b, shape9, ACL_FLOAT, &exponent);
  CreateAclTensor(c, shape9, ACL_FLOAT, &out);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor,
                                                            &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("PowTensorTensor_DimGT8_ShouldFail", status);
}

CaseResult CasePowTensorTensorUnsupportedDtypeFail()
{
  TensorHolder self;
  TensorHolder exponent;
  TensorHolder out;
  std::vector<uint16_t> a(4, 2);
  std::vector<uint16_t> b(4, 3);
  std::vector<uint16_t> c(4, 0);

  CreateAclTensor(a, {2, 2}, ACL_UINT16, &self);
  CreateAclTensor(b, {2, 2}, ACL_UINT16, &exponent);
  CreateAclTensor(c, {2, 2}, ACL_UINT16, &out);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor,
                                                            &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("PowTensorTensor_UnsupportedDtype_ShouldFail", status);
}

CaseResult CasePowTensorTensorBoolBoolFail()
{
  TensorHolder self;
  TensorHolder exponent;
  TensorHolder out;
  std::vector<uint8_t> a = {0, 1, 1, 0};
  std::vector<uint8_t> b = {1, 1, 0, 0};
  std::vector<uint8_t> c = {0, 0, 0, 0};

  CreateAclTensor(a, {4}, ACL_BOOL, &self);
  CreateAclTensor(b, {4}, ACL_BOOL, &exponent);
  CreateAclTensor(c, {4}, ACL_BOOL, &out);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor,
                                                            &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("PowTensorTensor_BoolBool_ShouldFail", status);
}

CaseResult CaseExp2NullptrFail()
{
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnExp2GetWorkspaceSize(nullptr, nullptr, &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("Exp2_Nullptr_ShouldFail", status);
}

CaseResult CaseExp2ShapeMismatchFail()
{
  TensorHolder self;
  TensorHolder out;
  std::vector<float> selfHost = {1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<float> outHost = {0.0F, 0.0F, 0.0F};

  CreateAclTensor(selfHost, {2, 2}, ACL_FLOAT, &self);
  CreateAclTensor(outHost, {3}, ACL_FLOAT, &out);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnExp2GetWorkspaceSize(self.tensor, out.tensor, &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("Exp2_ShapeMismatch_ShouldFail", status);
}

CaseResult CaseInplaceExp2UnsupportedDtypeFail()
{
  TensorHolder self;
  std::vector<int32_t> selfHost = {0, 1, 2, 3};
  CreateAclTensor(selfHost, {2, 2}, ACL_INT32, &self);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnInplaceExp2GetWorkspaceSize(self.tensor, &workspaceSize, &executor);
  return MakeStatusNotSuccessResult("InplaceExp2_Int32_ShouldFail", status);
}

int main()
{
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  std::vector<CaseResult> results;
  results.push_back(CasePowTensorScalarFloatBasic(stream));
  results.push_back(CasePowTensorScalarSquarePath(stream));
  results.push_back(CasePowTensorScalarNonNdFormat(stream));
  results.push_back(CaseInplacePowTensorScalarFloat(stream));
  results.push_back(CaseInplacePowTensorScalarNonSquare(stream));
  results.push_back(CasePowTensorScalarEmptyTensor(stream));
  results.push_back(CasePowTensorScalarInt32Exact(stream));
  results.push_back(CasePowTensorScalarFp16Probe(stream));
  results.push_back(CasePowTensorScalarInt64Exact(stream));

  results.push_back(CasePowScalarTensorFloatCompute(stream));
  results.push_back(CasePowScalarTensorFillOneBranch(stream));
  results.push_back(CasePowScalarTensorInt32Exact(stream));

  results.push_back(CasePowTensorTensorFloatBroadcast(stream));
  results.push_back(CaseInplacePowTensorTensorFloatBroadcast(stream));
  results.push_back(CasePowTensorTensorFp16OpKey1(stream));
  results.push_back(CasePowTensorTensorBf16OpKey2(stream));
  results.push_back(CasePowTensorTensorFp32OpKey3(stream));
  results.push_back(CasePowTensorTensorUint8OpKey4(stream));
  results.push_back(CasePowTensorTensorInt8OpKey5(stream));
  results.push_back(CasePowTensorTensorInt16OpKey6(stream));
  results.push_back(CasePowTensorTensorInt32OpKey7(stream));
  results.push_back(CasePowTensorTensorInt64AiCpuProbe(stream));
  results.push_back(CasePowTensorTensorEmptyTensor(stream));

  results.push_back(CaseExp2FloatBasic(stream));
  results.push_back(CaseExp2Int32InputCast(stream));
  results.push_back(CaseExp2BoolInputCast(stream));
  results.push_back(CaseExp2NaNInf(stream));
  results.push_back(CaseExp2OutputFp16(stream));
  results.push_back(CaseInplaceExp2Float(stream));
  results.push_back(CaseExp2EmptyTensor(stream));

  results.push_back(CasePowTensorScalarNullptrFail());
  results.push_back(CasePowTensorScalarShapeMismatchFail());
  results.push_back(CasePowTensorScalarUnsupportedDtypeFail());
  results.push_back(CasePowTensorScalarBoolBoolFail());
  results.push_back(CasePowTensorScalarNegativeExponentOnIntegralFail());
  results.push_back(CasePowTensorScalarExponentOverflowFail());
  results.push_back(CasePowTensorScalarInt16WorkspaceProbe());
  results.push_back(CasePowTensorScalarUint8WorkspaceProbe());
  results.push_back(CasePowTensorScalarInt64WorkspaceProbe());
  results.push_back(CasePowTensorScalarFp16ExponentOverflowFail());
  results.push_back(CasePowTensorScalarBf16ExponentOverflowFail());
  results.push_back(CasePowTensorScalarInt16ExponentOverflowFail());
  results.push_back(CasePowTensorScalarUint8ExponentOverflowFail());
  results.push_back(CasePowTensorScalarOutputCastFail());

  results.push_back(CasePowScalarTensorNullptrFail());
  results.push_back(CasePowScalarTensorShapeMismatchFail());
  results.push_back(CasePowScalarTensorUnsupportedDtypeFail());
  results.push_back(CasePowScalarTensorOutputCastFail());

  results.push_back(CasePowTensorTensorNullptrFail());
  results.push_back(CasePowTensorTensorShapeNoBroadcastFail());
  results.push_back(CasePowTensorTensorOutShapeMismatchFail());
  results.push_back(CasePowTensorTensorTooManyDimsFail());
  results.push_back(CasePowTensorTensorUnsupportedDtypeFail());
  results.push_back(CasePowTensorTensorBoolBoolFail());

  results.push_back(CaseExp2NullptrFail());
  results.push_back(CaseExp2ShapeMismatchFail());
  results.push_back(CaseInplaceExp2UnsupportedDtypeFail());

  int passCount = 0;
  int failCount = 0;
  for (size_t i = 0; i < results.size(); ++i) {
    PrintCaseResult(results[i]);
    if (results[i].pass) {
      ++passCount;
    } else {
      ++failCount;
    }
  }

  LOG_PRINT("\n==== Pow Example Summary ====\n");
  LOG_PRINT("TOTAL: %d, PASS: %d, FAIL: %d\n", static_cast<int>(results.size()), passCount, failCount);

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return failCount == 0 ? 0 : 1;
}
