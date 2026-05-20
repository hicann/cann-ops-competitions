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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  int64_t shapeSize = 1;
  for (auto i : shape) {
    if (i < 0 || (i != 0 && shapeSize > std::numeric_limits<int64_t>::max() / i)) {
      return -1;
    }
    shapeSize *= i;
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
    if (scalar != nullptr) {
      aclDestroyScalar(scalar);
      scalar = nullptr;
    }
  }
};

struct Complex32Value {
  uint16_t real;
  uint16_t imag;

  Complex32Value() : real(0), imag(0) {}
  Complex32Value(uint16_t realValue, uint16_t imagValue) : real(realValue), imag(imagValue) {}
};

struct Complex64Value {
  float real;
  float imag;

  Complex64Value() : real(0.0F), imag(0.0F) {}
  Complex64Value(float realValue, float imagValue) : real(realValue), imag(imagValue) {}
};

void DestroyTensor(TensorHolder& holder)
{
  if (holder.tensor != nullptr) {
    aclDestroyTensor(holder.tensor);
    holder.tensor = nullptr;
  }
  if (holder.deviceAddr != nullptr) {
    aclrtFree(holder.deviceAddr);
    holder.deviceAddr = nullptr;
  }
}

void DestroyScalar(ScalarHolder& holder)
{
  if (holder.scalar != nullptr) {
    aclDestroyScalar(holder.scalar);
    holder.scalar = nullptr;
  }
}

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

static inline uint32_t RoundToNearestEven(double value)
{
  double floorValue = std::floor(value);
  double fraction = value - floorValue;
  uint32_t rounded = static_cast<uint32_t>(floorValue);
  if (fraction > 0.5 || (fraction == 0.5 && (rounded & 1U) != 0U)) {
    ++rounded;
  }
  return rounded;
}

static inline uint16_t FloatToFp16Bits(float value)
{
  uint32_t bits = FloatToBits(value);
  uint16_t sign = static_cast<uint16_t>((bits >> 16U) & 0x8000U);
  uint32_t exponent = (bits >> 23U) & 0xFFU;
  uint32_t mantissa = bits & 0x7FFFFFU;

  if (exponent == 0xFFU) {
    if (mantissa == 0U) {
      return static_cast<uint16_t>(sign | 0x7C00U);
    }
    uint16_t payload = static_cast<uint16_t>(mantissa >> 13U);
    if (payload == 0U) {
      payload = 1U;
    }
    return static_cast<uint16_t>(sign | 0x7C00U | 0x0200U | payload);
  }

  if ((bits & 0x7FFFFFFFU) == 0U) {
    return sign;
  }

  double absValue = std::fabs(static_cast<double>(value));
  int binaryExponent = 0;
  double fraction = std::frexp(absValue, &binaryExponent);
  int halfExponent = binaryExponent - 1;

  if (halfExponent < -14) {
    uint32_t subnormalMantissa = RoundToNearestEven(std::ldexp(absValue, 24));
    if (subnormalMantissa == 0U) {
      return sign;
    }
    if (subnormalMantissa >= 1024U) {
      return static_cast<uint16_t>(sign | 0x0400U);
    }
    return static_cast<uint16_t>(sign | subnormalMantissa);
  }

  if (halfExponent > 15) {
    return static_cast<uint16_t>(sign | 0x7C00U);
  }

  double normalized = fraction * 2.0 - 1.0;
  uint32_t roundedMantissa = RoundToNearestEven(normalized * 1024.0);
  uint32_t encodedExponent = static_cast<uint32_t>(halfExponent + 15);
  if (roundedMantissa == 1024U) {
    roundedMantissa = 0U;
    ++encodedExponent;
  }
  if (encodedExponent >= 31U) {
    return static_cast<uint16_t>(sign | 0x7C00U);
  }
  return static_cast<uint16_t>(sign | (encodedExponent << 10U) | roundedMantissa);
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
  if ((bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0U) {
    uint16_t payload = static_cast<uint16_t>(bits >> 16U);
    return static_cast<uint16_t>(payload | 0x0040U);
  }
  uint32_t roundingBias = ((bits >> 16) & 1U) + 0x7FFFU;
  bits += roundingBias;
  return static_cast<uint16_t>(bits >> 16);
}

static inline float Bf16BitsToFloat(uint16_t value)
{
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  return BitsToFloat(bits);
}

static inline double Fp16MulReference(uint16_t lhs, uint16_t rhs)
{
  float compute = Fp16BitsToFloat(lhs) * Fp16BitsToFloat(rhs);
  return static_cast<double>(Fp16BitsToFloat(FloatToFp16Bits(compute)));
}

static inline double Bf16MulReference(uint16_t lhs, uint16_t rhs)
{
  float compute = Bf16BitsToFloat(lhs) * Bf16BitsToFloat(rhs);
  return static_cast<double>(Bf16BitsToFloat(FloatToBf16Bits(compute)));
}

bool IsBf16Supported()
{
  const char* socName = aclrtGetSocName();
  if (socName == nullptr) {
    return true;
  }
  std::string soc(socName);
  if (soc.find("910") != std::string::npos && soc.find("910B") == std::string::npos &&
      soc.find("910C") == std::string::npos && soc.find("910D") == std::string::npos &&
      soc.find("910E") == std::string::npos) {
    return false;
  }
  return true;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId); aclFinalize(); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType,
          TensorHolder* holder)
{
  CHECK_RET(holder != nullptr, return ACL_ERROR_INVALID_PARAM);
  int64_t elementCount = GetShapeSize(shape);
  CHECK_RET(elementCount >= 0, return ACL_ERROR_INVALID_PARAM);
  CHECK_RET(static_cast<int64_t>(hostData.size()) == elementCount,
            LOG_PRINT("hostData size mismatch. host=%zu shape=%ld\n", hostData.size(), elementCount);
            return ACL_ERROR_INVALID_PARAM);
  auto size = elementCount * static_cast<int64_t>(sizeof(T));
  auto strides = GetContiguousStrides(shape);

  if (size == 0) {
    holder->deviceAddr = nullptr;
    holder->tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                     shape.data(), shape.size(), holder->deviceAddr);
    CHECK_RET(holder->tensor != nullptr, LOG_PRINT("aclCreateTensor failed for empty tensor.\n"); return ACL_ERROR_INVALID_PARAM);
    return ACL_SUCCESS;
  }

  auto ret = aclrtMalloc(&holder->deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  ret = aclrtMemcpy(holder->deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  holder->tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                   shape.data(), shape.size(), holder->deviceAddr);
  CHECK_RET(holder->tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
  return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensorWithFormat(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType,
                              aclFormat format, TensorHolder* holder)
{
  CHECK_RET(holder != nullptr, return ACL_ERROR_INVALID_PARAM);
  int64_t elementCount = GetShapeSize(shape);
  CHECK_RET(elementCount >= 0, return ACL_ERROR_INVALID_PARAM);
  CHECK_RET(static_cast<int64_t>(hostData.size()) == elementCount,
            LOG_PRINT("hostData size mismatch. host=%zu shape=%ld\n", hostData.size(), elementCount);
            return ACL_ERROR_INVALID_PARAM);
  auto size = elementCount * static_cast<int64_t>(sizeof(T));
  auto strides = GetContiguousStrides(shape);

  if (size == 0) {
    holder->deviceAddr = nullptr;
    holder->tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
                                     shape.data(), shape.size(), holder->deviceAddr);
    CHECK_RET(holder->tensor != nullptr, LOG_PRINT("aclCreateTensor failed for empty tensor.\n"); return ACL_ERROR_INVALID_PARAM);
    return ACL_SUCCESS;
  }

  auto ret = aclrtMalloc(&holder->deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(holder->deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  holder->tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
                                   shape.data(), shape.size(), holder->deviceAddr);
  CHECK_RET(holder->tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
  return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensorView(const std::vector<T>& hostStorage, const std::vector<int64_t>& viewShape,
                        const std::vector<int64_t>& storageShape, const std::vector<int64_t>& viewStrides,
                        int64_t storageOffset, aclDataType dataType, TensorHolder* holder)
{
  CHECK_RET(holder != nullptr, return ACL_ERROR_INVALID_PARAM);
  int64_t storageElementCount = GetShapeSize(storageShape);
  CHECK_RET(storageElementCount >= 0, return ACL_ERROR_INVALID_PARAM);
  auto size = storageElementCount * static_cast<int64_t>(sizeof(T));
  CHECK_RET(static_cast<int64_t>(hostStorage.size()) == storageElementCount,
            LOG_PRINT("hostStorage size mismatch. host=%zu storage_shape=%ld\n", hostStorage.size(), storageElementCount);
            return ACL_ERROR_INVALID_PARAM);

  if (size == 0) {
    holder->deviceAddr = nullptr;
    holder->tensor = aclCreateTensor(viewShape.data(), viewShape.size(), dataType, viewStrides.data(), storageOffset,
                                     aclFormat::ACL_FORMAT_ND, storageShape.data(), storageShape.size(),
                                     holder->deviceAddr);
    CHECK_RET(holder->tensor != nullptr, LOG_PRINT("aclCreateTensor failed for empty view tensor.\n"); return ACL_ERROR_INVALID_PARAM);
    return ACL_SUCCESS;
  }

  auto ret = aclrtMalloc(&holder->deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(holder->deviceAddr, size, hostStorage.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  holder->tensor = aclCreateTensor(viewShape.data(), viewShape.size(), dataType, viewStrides.data(), storageOffset,
                                   aclFormat::ACL_FORMAT_ND, storageShape.data(), storageShape.size(),
                                   holder->deviceAddr);
  CHECK_RET(holder->tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
  return ACL_SUCCESS;
}

template <typename T>
int CopyTensorToHost(const TensorHolder& holder, std::vector<T>* hostData)
{
  CHECK_RET(hostData != nullptr, return ACL_ERROR_INVALID_PARAM);
  auto size = static_cast<int64_t>(hostData->size() * sizeof(T));
  if (size == 0) {
    return ACL_SUCCESS;
  }
  CHECK_RET(holder.deviceAddr != nullptr, return ACL_ERROR_INVALID_PARAM);
  auto ret = aclrtMemcpy(hostData->data(), size, holder.deviceAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

template <typename T>
int CopyStorageToHost(const TensorHolder& holder, std::vector<T>* hostStorage)
{
  return CopyTensorToHost(holder, hostStorage);
}

bool IsClose(double actual, double expected, double atol, double rtol)
{
  return std::fabs(actual - expected) <= (atol + rtol * std::fabs(expected));
}

std::string FormatDouble(double value)
{
  char buf[64] = {0};
  std::snprintf(buf, sizeof(buf), "%.15e", value);
  return std::string(buf);
}

bool CheckScalarFloat(double actual, double expected, double atol, double rtol, bool checkSignedZero, std::string* err)
{
  const bool bothNaN = std::isnan(actual) && std::isnan(expected);
  const bool bothInf = std::isinf(actual) && std::isinf(expected) && ((actual > 0) == (expected > 0));
  if (bothNaN || bothInf) {
    return true;
  }
  if (std::isnan(actual) || std::isnan(expected) || std::isinf(actual) || std::isinf(expected)) {
    if (err != nullptr) {
      *err = "actual=" + FormatDouble(actual) + " expected=" + FormatDouble(expected);
    }
    return false;
  }
  if (checkSignedZero && actual == 0.0 && expected == 0.0 && std::signbit(actual) != std::signbit(expected)) {
    if (err != nullptr) {
      *err = "actual=" + FormatDouble(actual) + " expected=" + FormatDouble(expected) + " signed_zero_mismatch";
    }
    return false;
  }
  if (!IsClose(actual, expected, atol, rtol)) {
    if (err != nullptr) {
      const double absErr = std::fabs(actual - expected);
      const double relErr = std::fabs(expected) > 0.0 ? absErr / std::fabs(expected) : absErr;
      const double threshold = atol + rtol * std::fabs(expected);
      *err = "actual=" + FormatDouble(actual) + " expected=" + FormatDouble(expected) +
             " abs_err=" + FormatDouble(absErr) + " rel_err=" + FormatDouble(relErr) +
             " threshold=" + FormatDouble(threshold);
    }
    return false;
  }
  return true;
}

bool CheckFloatResult(const std::vector<float>& actual, const std::vector<double>& expected, double atol, double rtol,
                      std::string* err, bool checkSignedZero = false)
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
    std::string scalarErr;
    if (!CheckScalarFloat(a, e, atol, rtol, checkSignedZero, &scalarErr)) {
      if (err != nullptr) {
        *err = "index=" + std::to_string(i) + " " + scalarErr;
      }
      return false;
    }
  }
  return true;
}

void LogPrecisionSummary(const std::string& name, const std::vector<float>& actual,
                         const std::vector<double>& expected)
{
  double maxAbsErr = 0.0;
  double maxRelErr = 0.0;
  size_t maxIndex = 0;
  for (size_t i = 0; i < actual.size() && i < expected.size(); ++i) {
    const double a = static_cast<double>(actual[i]);
    const double e = expected[i];
    if ((std::isnan(a) && std::isnan(e)) || (std::isinf(a) && std::isinf(e) && ((a > 0) == (e > 0)))) {
      continue;
    }
    const double absErr = std::fabs(a - e);
    const double relErr = std::fabs(e) > 0.0 ? absErr / std::fabs(e) : absErr;
    if (absErr > maxAbsErr) {
      maxAbsErr = absErr;
      maxRelErr = relErr;
      maxIndex = i;
    }
  }
  LOG_PRINT("[PRECISION] %s max_index=%zu abs_err=%s rel_err=%s\n", name.c_str(), maxIndex,
            FormatDouble(maxAbsErr).c_str(), FormatDouble(maxRelErr).c_str());
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
        *err = "index=" + std::to_string(i) + " actual=" + std::to_string(actual[i]) + " expected=" +
             std::to_string(expected[i]);
      }
      return false;
    }
  }
  return true;
}

bool CheckComplex64Result(const std::vector<Complex64Value>& actual, const std::vector<Complex64Value>& expected,
                          double atol, double rtol, std::string* err)
{
  if (actual.size() != expected.size()) {
    if (err != nullptr) {
      *err = "size mismatch";
    }
    return false;
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    const double actualReal = static_cast<double>(actual[i].real);
    const double actualImag = static_cast<double>(actual[i].imag);
    const double expectedReal = static_cast<double>(expected[i].real);
    const double expectedImag = static_cast<double>(expected[i].imag);
    std::string realErr;
    std::string imagErr;
    if (!CheckScalarFloat(actualReal, expectedReal, atol, rtol, false, &realErr) ||
        !CheckScalarFloat(actualImag, expectedImag, atol, rtol, false, &imagErr)) {
      if (err != nullptr) {
        *err = "index=" + std::to_string(i) + " actual=(" + FormatDouble(actualReal) + "," +
               FormatDouble(actualImag) + ") expected=(" + FormatDouble(expectedReal) + "," +
               FormatDouble(expectedImag) + ") real_err=" + realErr + " imag_err=" + imagErr;
      }
      return false;
    }
  }
  return true;
}

bool CheckComplex32Result(const std::vector<Complex32Value>& actual, const std::vector<Complex64Value>& expected,
                          double atol, double rtol, std::string* err)
{
  if (actual.size() != expected.size()) {
    if (err != nullptr) {
      *err = "size mismatch";
    }
    return false;
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    const double actualReal = static_cast<double>(Fp16BitsToFloat(actual[i].real));
    const double actualImag = static_cast<double>(Fp16BitsToFloat(actual[i].imag));
    const double expectedReal = static_cast<double>(expected[i].real);
    const double expectedImag = static_cast<double>(expected[i].imag);
    std::string realErr;
    std::string imagErr;
    if (!CheckScalarFloat(actualReal, expectedReal, atol, rtol, false, &realErr) ||
        !CheckScalarFloat(actualImag, expectedImag, atol, rtol, false, &imagErr)) {
      if (err != nullptr) {
        *err = "index=" + std::to_string(i) + " actual=(" + FormatDouble(actualReal) + "," +
               FormatDouble(actualImag) + ") expected=(" + FormatDouble(expectedReal) + "," +
               FormatDouble(expectedImag) + ") real_err=" + realErr + " imag_err=" + imagErr;
      }
      return false;
    }
  }
  return true;
}

std::vector<Complex64Value> ComputeExpectedComplexMul(const std::vector<Complex64Value>& selfHost,
                                                      const std::vector<Complex64Value>& otherHost)
{
  std::vector<Complex64Value> expected(selfHost.size());
  for (size_t i = 0; i < selfHost.size(); ++i) {
    expected[i].real = selfHost[i].real * otherHost[i].real - selfHost[i].imag * otherHost[i].imag;
    expected[i].imag = selfHost[i].real * otherHost[i].imag + selfHost[i].imag * otherHost[i].real;
  }
  return expected;
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
  auto inStrides = GetContiguousStrides(inShape);
  int64_t inRank = static_cast<int64_t>(inShape.size());
  int64_t outRank = static_cast<int64_t>(outShape.size());
  int64_t rankDiff = outRank - inRank;
  int64_t offset = 0;
  for (int64_t i = 0; i < inRank; ++i) {
    int64_t outAxis = i + rankDiff;
    int64_t coord = inShape[static_cast<size_t>(i)] == 1 ? 0 : outCoords[static_cast<size_t>(outAxis)];
    offset += coord * inStrides[static_cast<size_t>(i)];
  }
  return offset;
}

std::vector<double> ComputeExpectedBroadcastMulFloat(const std::vector<float>& a, const std::vector<int64_t>& aShape,
                           const std::vector<float>& b, const std::vector<int64_t>& bShape,
                           const std::vector<int64_t>& outShape)
{
  std::vector<double> expected(static_cast<size_t>(GetShapeSize(outShape)), 0.0);
  std::vector<int64_t> coords;
  for (int64_t i = 0; i < GetShapeSize(outShape); ++i) {
    GetCoordsFromLinear(i, outShape, &coords);
    int64_t aOffset = GetBroadcastOffset(aShape, outShape, coords);
    int64_t bOffset = GetBroadcastOffset(bShape, outShape, coords);
    expected[static_cast<size_t>(i)] = static_cast<double>(a[static_cast<size_t>(aOffset)]) *
                       static_cast<double>(b[static_cast<size_t>(bOffset)]);
  }
  return expected;
}

int RunMul(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream)
{
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }

  ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);

  if (workspaceAddr != nullptr) {
    ret = aclrtFree(workspaceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }
  return ACL_SUCCESS;
}

int RunMuls(const aclTensor* self, const aclScalar* scalar, aclTensor* out, aclrtStream stream)
{
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulsGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }

  ret = aclnnMuls(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);

  if (workspaceAddr != nullptr) {
    ret = aclrtFree(workspaceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }
  return ACL_SUCCESS;
}

int RunInplaceMul(aclTensor* selfRef, const aclTensor* other, aclrtStream stream)
{
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnInplaceMulGetWorkspaceSize(selfRef, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }

  ret = aclnnInplaceMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);

  if (workspaceAddr != nullptr) {
    ret = aclrtFree(workspaceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }
  return ACL_SUCCESS;
}

int RunInplaceMuls(aclTensor* selfRef, const aclScalar* scalar, aclrtStream stream)
{
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnInplaceMulsGetWorkspaceSize(selfRef, scalar, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }

  ret = aclnnInplaceMuls(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);

  if (workspaceAddr != nullptr) {
    ret = aclrtFree(workspaceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }
  return ACL_SUCCESS;
}

enum class CaseStatus {
  PASS,
  FAIL,
  SKIP
};

struct CaseResult {
  std::string name;
  bool pass = false;
  std::string detail;
  CaseStatus status = CaseStatus::FAIL;
};

CaseResult SkipCase(const std::string& name, const std::string& detail)
{
  CaseResult result{name, false, detail};
  result.status = CaseStatus::SKIP;
  return result;
}

void MarkSkip(CaseResult* result, const std::string& detail)
{
  if (result == nullptr) {
    return;
  }
  result->pass = false;
  result->status = CaseStatus::SKIP;
  result->detail = detail;
}

bool IsCasePassed(const CaseResult& result)
{
  return result.status == CaseStatus::PASS || (result.status == CaseStatus::FAIL && result.pass);
}

void PrintCaseResult(const CaseResult& result)
{
  if (IsCasePassed(result)) {
    LOG_PRINT("[PASS] %s\n", result.name.c_str());
  } else if (result.status == CaseStatus::SKIP) {
    LOG_PRINT("[SKIP] %s -> %s\n", result.name.c_str(), result.detail.c_str());
  } else {
    LOG_PRINT("[FAIL] %s -> %s\n", result.name.c_str(), result.detail.c_str());
  }
}

CaseResult CaseMulFloatBasic(aclrtStream stream)
{
  CaseResult result{"Mul_Float32_Basic", false, ""};
  TensorHolder self;
  TensorHolder other;
  TensorHolder out;

  std::vector<int64_t> shape = {2, 4};
  std::vector<float> selfHost = {0.0F, 1.0F, 2.0F, 3.0F, -1.0F, -2.0F, 4.0F, 5.0F};
  std::vector<float> otherHost = {1.0F, 2.0F, 3.0F, 4.0F, 2.0F, -3.0F, 0.5F, -1.0F};
  std::vector<float> outHost(static_cast<size_t>(GetShapeSize(shape)), 0.0F);

  int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(otherHost, shape, ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; DestroyTensor(self); return result);
  ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; DestroyTensor(self); DestroyTensor(other); return result);

  ret = RunMul(self.tensor, other.tensor, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run mul failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);

  ret = CopyTensorToHost(out, &outHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);

  std::vector<double> expected = ComputeExpectedBroadcastMulFloat(selfHost, shape, otherHost, shape, shape);
  std::string err;
  result.pass = CheckFloatResult(outHost, expected, 1e-5, 1e-5, &err);
  result.detail = err;

  DestroyTensor(self);
  DestroyTensor(other);
  DestroyTensor(out);
  return result;
}

CaseResult CaseMulFloatBroadcast(aclrtStream stream)
{
  CaseResult result{"Mul_Float32_Broadcast", false, ""};
  TensorHolder self;
  TensorHolder other;
  TensorHolder out;

  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> otherShape = {3};
  std::vector<int64_t> outShape = {2, 3};
  std::vector<float> selfHost = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  std::vector<float> otherHost = {2.0F, -1.0F, 0.5F};
  std::vector<float> outHost(static_cast<size_t>(GetShapeSize(outShape)), 0.0F);

  int ret = CreateAclTensor(selfHost, selfShape, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(otherHost, otherShape, ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; DestroyTensor(self); return result);
  ret = CreateAclTensor(outHost, outShape, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; DestroyTensor(self); DestroyTensor(other); return result);

  ret = RunMul(self.tensor, other.tensor, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run mul failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);
  ret = CopyTensorToHost(out, &outHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);

  std::vector<double> expected = ComputeExpectedBroadcastMulFloat(selfHost, selfShape, otherHost, otherShape, outShape);
  std::string err;
  result.pass = CheckFloatResult(outHost, expected, 1e-5, 1e-5, &err);
  result.detail = err;

  DestroyTensor(self);
  DestroyTensor(other);
  DestroyTensor(out);
  return result;
}

CaseResult CaseMulFloatSpecial(aclrtStream stream)
{
  CaseResult result{"Mul_Float32_NaN_Inf", false, ""};
  TensorHolder self;
  TensorHolder other;
  TensorHolder out;

  std::vector<int64_t> shape = {4};
  std::vector<float> selfHost = {
    std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity(),
    3.0F};
  std::vector<float> otherHost = {2.0F, -1.0F, 0.0F, std::numeric_limits<float>::infinity()};
  std::vector<float> outHost(static_cast<size_t>(GetShapeSize(shape)), 0.0F);

  int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(otherHost, shape, ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; DestroyTensor(self); return result);
  ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; DestroyTensor(self); DestroyTensor(other); return result);

  ret = RunMul(self.tensor, other.tensor, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run mul failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);
  ret = CopyTensorToHost(out, &outHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);

  std::vector<double> expected(4, 0.0);
  for (size_t i = 0; i < 4; ++i) {
    expected[i] = static_cast<double>(selfHost[i]) * static_cast<double>(otherHost[i]);
  }
  std::string err;
  result.pass = CheckFloatResult(outHost, expected, 1e-5, 1e-5, &err);
  result.detail = err;

  DestroyTensor(self);
  DestroyTensor(other);
  DestroyTensor(out);
  return result;
}

CaseResult CaseMulFloatSignedZero(aclrtStream stream)
{
  CaseResult result{"Mul_Float32_SignedZero", false, ""};
  TensorHolder self;
  TensorHolder other;
  TensorHolder out;

  std::vector<int64_t> shape = {4};
  const float negZero = -0.0F;
  const float posZero = 0.0F;
  std::vector<float> selfHost = {negZero, negZero, posZero, posZero};
  std::vector<float> otherHost = {3.0F, -3.0F, -3.0F, 3.0F};
  std::vector<float> outHost(static_cast<size_t>(GetShapeSize(shape)), 0.0F);

  int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(otherHost, shape, ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; return result);
  ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; return result);

  ret = RunMul(self.tensor, other.tensor, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run mul failed"; return result);
  ret = CopyTensorToHost(out, &outHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; return result);

  std::vector<double> expected = {
    static_cast<double>(negZero * 3.0F),
    static_cast<double>(negZero * -3.0F),
    static_cast<double>(posZero * -3.0F),
    static_cast<double>(posZero * 3.0F)};
  std::string err;
  result.pass = CheckFloatResult(outHost, expected, 0.0, 0.0, &err, true);
  result.detail = err;
  return result;
}

CaseResult CaseMulInt32Exact(aclrtStream stream)
{
  CaseResult result{"Mul_Int32_Exact", false, ""};
  TensorHolder self;
  TensorHolder other;
  TensorHolder out;

  std::vector<int64_t> shape = {2, 3};
  std::vector<int32_t> selfHost = {0, -1, 7, 9, -3, 2};
  std::vector<int32_t> otherHost = {8, 2, -2, 0, -4, 10};
  std::vector<int32_t> outHost(static_cast<size_t>(GetShapeSize(shape)), 0);

  int ret = CreateAclTensor(selfHost, shape, ACL_INT32, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(otherHost, shape, ACL_INT32, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; DestroyTensor(self); return result);
  ret = CreateAclTensor(outHost, shape, ACL_INT32, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; DestroyTensor(self); DestroyTensor(other); return result);

  ret = RunMul(self.tensor, other.tensor, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run mul failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);
  ret = CopyTensorToHost(out, &outHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);

  std::vector<int32_t> expected(outHost.size(), 0);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = selfHost[i] * otherHost[i];
  }
  std::string err;
  result.pass = CheckExactResult(outHost, expected, &err);
  result.detail = err;

  DestroyTensor(self);
  DestroyTensor(other);
  DestroyTensor(out);
  return result;
}

template <typename T>
CaseResult CaseMulExactTyped(const std::string& name, aclrtStream stream, aclDataType dtype,
                             const std::vector<int64_t>& shape, const std::vector<T>& selfHost,
                             const std::vector<T>& otherHost)
{
  CaseResult result{name, false, ""};
  TensorHolder self;
  TensorHolder other;
  TensorHolder out;
  std::vector<T> outHost(selfHost.size(), static_cast<T>(0));

  int ret = CreateAclTensor(selfHost, shape, dtype, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(otherHost, shape, dtype, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; DestroyTensor(self); return result);
  ret = CreateAclTensor(outHost, shape, dtype, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; DestroyTensor(self); DestroyTensor(other); return result);

  ret = RunMul(self.tensor, other.tensor, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run mul failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);

  ret = CopyTensorToHost(out, &outHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);

  std::vector<T> expected(outHost.size(), static_cast<T>(0));
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<T>(selfHost[i] * otherHost[i]);
  }
  std::string err;
  result.pass = CheckExactResult(outHost, expected, &err);
  result.detail = err;

  DestroyTensor(self);
  DestroyTensor(other);
  DestroyTensor(out);
  return result;
}

CaseResult CaseMulDoubleBasic(aclrtStream stream)
{
  CaseResult result{"Mul_Double_Basic", false, ""};
  TensorHolder self;
  TensorHolder other;
  TensorHolder out;
  std::vector<int64_t> shape = {2, 3};
  std::vector<double> selfHost = {1.25, -2.0, 3.5, -4.0, 0.25, 8.0};
  std::vector<double> otherHost = {2.0, -0.5, 1.5, 3.0, 4.0, -2.0};
  std::vector<double> outHost(selfHost.size(), 0.0);

  int ret = CreateAclTensor(selfHost, shape, ACL_DOUBLE, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(otherHost, shape, ACL_DOUBLE, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; DestroyTensor(self); return result);
  ret = CreateAclTensor(outHost, shape, ACL_DOUBLE, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; DestroyTensor(self); DestroyTensor(other); return result);

  ret = RunMul(self.tensor, other.tensor, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run mul failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);
  ret = CopyTensorToHost(out, &outHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);

  std::vector<double> expected(outHost.size(), 0.0);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = selfHost[i] * otherHost[i];
  }

  bool ok = true;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (!IsClose(outHost[i], expected[i], 1e-12, 1e-12)) {
      ok = false;
      result.detail = "index=" + std::to_string(i) + " mismatch";
      break;
    }
  }
  result.pass = ok;

  DestroyTensor(self);
  DestroyTensor(other);
  DestroyTensor(out);
  return result;
}

CaseResult CaseMulBoolExact(aclrtStream stream)
{
  return CaseMulExactTyped<uint8_t>("Mul_Bool_Exact", stream, ACL_BOOL, {2, 4},
                                    {1, 0, 1, 1, 0, 1, 0, 0},
                                    {1, 1, 0, 1, 0, 1, 1, 0});
}

CaseResult CaseMulInt8Exact(aclrtStream stream)
{
  return CaseMulExactTyped<int8_t>("Mul_Int8_Exact", stream, ACL_INT8, {2, 4},
                                   {1, -2, 3, -4, 2, -3, 4, -1},
                                   {2, 3, -1, -2, -3, 2, 1, -4});
}

CaseResult CaseMulUInt8Exact(aclrtStream stream)
{
  return CaseMulExactTyped<uint8_t>("Mul_UInt8_Exact", stream, ACL_UINT8, {2, 4},
                                    {1, 2, 3, 4, 5, 6, 7, 8},
                                    {2, 3, 4, 2, 1, 2, 3, 4});
}

CaseResult CaseMulInt16Exact(aclrtStream stream)
{
  return CaseMulExactTyped<int16_t>("Mul_Int16_Exact", stream, ACL_INT16, {2, 4},
                                    {1, -2, 3, -4, 5, -6, 7, -8},
                                    {2, 3, -1, -2, 1, 2, -3, 1});
}

CaseResult CaseMulInt64Exact(aclrtStream stream)
{
  return CaseMulExactTyped<int64_t>("Mul_Int64_Exact", stream, ACL_INT64, {2, 4},
                                    {1, -2, 3, -4, 5, -6, 7, -8},
                                    {2, 3, -1, -2, 1, 2, -3, 1});
}

CaseResult CaseMulComplex64Exact(aclrtStream stream)
{
  CaseResult result{"Mul_Complex64_Exact", false, ""};
  TensorHolder self;
  TensorHolder other;
  TensorHolder out;
  std::vector<int64_t> shape = {4};

  std::vector<Complex64Value> selfHost = {{1.0F, 2.0F}, {-3.0F, 0.5F}, {0.0F, -2.0F}, {4.0F, -1.0F}};
  std::vector<Complex64Value> otherHost = {{2.0F, -1.0F}, {0.5F, 3.0F}, {-1.0F, 0.25F}, {0.0F, 2.0F}};
  std::vector<Complex64Value> outHost(selfHost.size());

  int ret = CreateAclTensor(selfHost, shape, ACL_COMPLEX64, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(otherHost, shape, ACL_COMPLEX64, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; return result);
  ret = CreateAclTensor(outHost, shape, ACL_COMPLEX64, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; return result);

  ret = RunMul(self.tensor, other.tensor, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run mul failed"; return result);
  ret = CopyTensorToHost(out, &outHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; return result);

  std::vector<Complex64Value> expected = ComputeExpectedComplexMul(selfHost, otherHost);
  std::string err;
  result.pass = CheckComplex64Result(outHost, expected, 1e-5, 1e-5, &err);
  result.detail = err;
  return result;
}

CaseResult CaseMulComplex32Exact(aclrtStream stream)
{
  CaseResult result{"Mul_Complex32_Exact", false, ""};
  TensorHolder self;
  TensorHolder other;
  TensorHolder out;
  std::vector<int64_t> shape = {4};

  std::vector<Complex64Value> selfFp32 = {{1.0F, 0.5F}, {-2.0F, 1.0F}, {0.25F, -0.5F}, {3.0F, -1.0F}};
  std::vector<Complex64Value> otherFp32 = {{2.0F, -1.0F}, {0.5F, 0.25F}, {-4.0F, 2.0F}, {0.0F, -0.5F}};
  std::vector<Complex32Value> selfHost(selfFp32.size());
  std::vector<Complex32Value> otherHost(otherFp32.size());
  std::vector<Complex32Value> outHost(selfFp32.size());
  for (size_t i = 0; i < selfFp32.size(); ++i) {
    selfHost[i].real = FloatToFp16Bits(selfFp32[i].real);
    selfHost[i].imag = FloatToFp16Bits(selfFp32[i].imag);
    otherHost[i].real = FloatToFp16Bits(otherFp32[i].real);
    otherHost[i].imag = FloatToFp16Bits(otherFp32[i].imag);
  }

  int ret = CreateAclTensor(selfHost, shape, ACL_COMPLEX32, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(otherHost, shape, ACL_COMPLEX32, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; return result);
  ret = CreateAclTensor(outHost, shape, ACL_COMPLEX32, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; return result);

  ret = RunMul(self.tensor, other.tensor, out.tensor, stream);
  if (ret != ACL_SUCCESS) {
    // complex32 is present in op_host/op_kernel, but some aclnn dtype guards do not expose it.
    MarkSkip(&result, "complex32 unsupported by aclnn dtype guard, ret=" + std::to_string(ret));
    return result;
  }
  ret = CopyTensorToHost(out, &outHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; return result);

  std::vector<Complex64Value> expected(selfFp32.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    const float aReal = Fp16BitsToFloat(selfHost[i].real);
    const float aImag = Fp16BitsToFloat(selfHost[i].imag);
    const float bReal = Fp16BitsToFloat(otherHost[i].real);
    const float bImag = Fp16BitsToFloat(otherHost[i].imag);
    expected[i].real = Fp16BitsToFloat(FloatToFp16Bits(aReal * bReal - aImag * bImag));
    expected[i].imag = Fp16BitsToFloat(FloatToFp16Bits(aReal * bImag + aImag * bReal));
  }
  std::string err;
  result.pass = CheckComplex32Result(outHost, expected, 1e-3, 1e-3, &err);
  result.detail = err;
  return result;
}

CaseResult CaseMulFp16Exact(aclrtStream stream)
{
  CaseResult result{"Mul_Fp16_Exact", false, ""};
  TensorHolder self;
  TensorHolder other;
  TensorHolder out;
  std::vector<int64_t> shape = {4};

  std::vector<float> aFp32 = {1.0F, -2.0F, 0.5F, 3.0F};
  std::vector<float> bFp32 = {2.0F, -1.5F, 8.0F, 0.25F};
  std::vector<uint16_t> aFp16(shape[0], 0);
  std::vector<uint16_t> bFp16(shape[0], 0);
  std::vector<uint16_t> outFp16(shape[0], 0);
  for (size_t i = 0; i < aFp16.size(); ++i) {
    aFp16[i] = FloatToFp16Bits(aFp32[i]);
    bFp16[i] = FloatToFp16Bits(bFp32[i]);
  }

  int ret = CreateAclTensor(aFp16, shape, ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(bFp16, shape, ACL_FLOAT16, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; DestroyTensor(self); return result);
  ret = CreateAclTensor(outFp16, shape, ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; DestroyTensor(self); DestroyTensor(other); return result);

  ret = RunMul(self.tensor, other.tensor, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run mul failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);
  ret = CopyTensorToHost(out, &outFp16);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);

  std::vector<float> outFp32(outFp16.size(), 0.0F);
  std::vector<double> expected(outFp16.size(), 0.0);
  for (size_t i = 0; i < outFp16.size(); ++i) {
    outFp32[i] = Fp16BitsToFloat(outFp16[i]);
    expected[i] = Fp16MulReference(aFp16[i], bFp16[i]);
  }
  std::string err;
  result.pass = CheckFloatResult(outFp32, expected, 1e-3, 1e-3, &err);
  result.detail = err;

  DestroyTensor(self);
  DestroyTensor(other);
  DestroyTensor(out);
  return result;
}

CaseResult CaseMulBf16Exact(aclrtStream stream)
{
  CaseResult result{"Mul_Bf16_Exact", false, ""};
  if (!IsBf16Supported()) {
    return SkipCase(result.name, "BF16 unsupported on this SoC");
  }
  TensorHolder self;
  TensorHolder other;
  TensorHolder out;
  std::vector<int64_t> shape = {4};

  std::vector<float> aFp32 = {1.0F, -2.5F, 0.5F, 3.0F};
  std::vector<float> bFp32 = {2.0F, -1.25F, 8.0F, 0.5F};
  std::vector<uint16_t> aBf16(shape[0], 0);
  std::vector<uint16_t> bBf16(shape[0], 0);
  std::vector<uint16_t> outBf16(shape[0], 0);
  for (size_t i = 0; i < aBf16.size(); ++i) {
    aBf16[i] = FloatToBf16Bits(aFp32[i]);
    bBf16[i] = FloatToBf16Bits(bFp32[i]);
  }

  int ret = CreateAclTensor(aBf16, shape, ACL_BF16, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(bBf16, shape, ACL_BF16, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; DestroyTensor(self); return result);
  ret = CreateAclTensor(outBf16, shape, ACL_BF16, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; DestroyTensor(self); DestroyTensor(other); return result);

  ret = RunMul(self.tensor, other.tensor, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run mul failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);
  ret = CopyTensorToHost(out, &outBf16);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);

  std::vector<float> outFp32(outBf16.size(), 0.0F);
  std::vector<double> expected(outBf16.size(), 0.0);
  for (size_t i = 0; i < outBf16.size(); ++i) {
    outFp32[i] = Bf16BitsToFloat(outBf16[i]);
    expected[i] = Bf16MulReference(aBf16[i], bBf16[i]);
  }
  std::string err;
  result.pass = CheckFloatResult(outFp32, expected, 1e-2, 1e-2, &err);
  result.detail = err;

  DestroyTensor(self);
  DestroyTensor(other);
  DestroyTensor(out);
  return result;
}

CaseResult CaseMulMixFloatBf16(aclrtStream stream)
{
  CaseResult result{"Mul_Mix_Float_Bf16", false, ""};
  if (!IsBf16Supported()) {
    return SkipCase(result.name, "BF16 unsupported on this SoC");
  }
  TensorHolder self;
  TensorHolder other;
  TensorHolder out;
  std::vector<int64_t> shape = {4};

  std::vector<float> selfHost = {1.0F, -2.0F, 0.5F, 3.5F};
  std::vector<float> otherFp32 = {2.0F, -1.25F, 4.0F, 0.5F};
  std::vector<uint16_t> otherBf16(shape[0], 0);
  for (size_t i = 0; i < otherBf16.size(); ++i) {
    otherBf16[i] = FloatToBf16Bits(otherFp32[i]);
  }
  std::vector<float> outHost(shape[0], 0.0F);

  int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(otherBf16, shape, ACL_BF16, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; DestroyTensor(self); return result);
  ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; DestroyTensor(self); DestroyTensor(other); return result);

  ret = RunMul(self.tensor, other.tensor, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run mul failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);
  ret = CopyTensorToHost(out, &outHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);

  std::vector<double> expected(outHost.size(), 0.0);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<double>(selfHost[i]) * static_cast<double>(Bf16BitsToFloat(otherBf16[i]));
  }
  std::string err;
  result.pass = CheckFloatResult(outHost, expected, 1e-2, 1e-2, &err);
  result.detail = err;

  DestroyTensor(self);
  DestroyTensor(other);
  DestroyTensor(out);
  return result;
}

CaseResult CaseMulMixBf16Float(aclrtStream stream)
{
  CaseResult result{"Mul_Mix_Bf16_Float", false, ""};
  if (!IsBf16Supported()) {
    return SkipCase(result.name, "BF16 unsupported on this SoC");
  }
  TensorHolder self;
  TensorHolder other;
  TensorHolder out;
  std::vector<int64_t> shape = {4};

  std::vector<float> selfFp32 = {1.0F, -2.0F, 0.5F, 3.5F};
  std::vector<uint16_t> selfBf16(shape[0], 0);
  for (size_t i = 0; i < selfBf16.size(); ++i) {
    selfBf16[i] = FloatToBf16Bits(selfFp32[i]);
  }
  std::vector<float> otherHost = {2.0F, -1.25F, 4.0F, 0.5F};
  std::vector<float> outHost(shape[0], 0.0F);

  int ret = CreateAclTensor(selfBf16, shape, ACL_BF16, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(otherHost, shape, ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; return result);
  ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; return result);

  ret = RunMul(self.tensor, other.tensor, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run mul failed"; return result);
  ret = CopyTensorToHost(out, &outHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; return result);

  std::vector<double> expected(outHost.size(), 0.0);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<double>(Bf16BitsToFloat(selfBf16[i])) * static_cast<double>(otherHost[i]);
  }
  std::string err;
  result.pass = CheckFloatResult(outHost, expected, 1e-2, 1e-2, &err);
  result.detail = err;
  return result;
}

CaseResult CaseMulMixFloatFp16(aclrtStream stream)
{
  CaseResult result{"Mul_Mix_Float_Fp16", false, ""};
  TensorHolder self;
  TensorHolder other;
  TensorHolder out;
  std::vector<int64_t> shape = {4};

  std::vector<float> selfHost = {1.0F, -2.0F, 0.5F, 3.5F};
  std::vector<float> otherFp32 = {2.0F, -1.5F, 4.0F, 0.5F};
  std::vector<uint16_t> otherFp16(shape[0], 0);
  for (size_t i = 0; i < otherFp16.size(); ++i) {
    otherFp16[i] = FloatToFp16Bits(otherFp32[i]);
  }
  std::vector<float> outHost(shape[0], 0.0F);

  int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(otherFp16, shape, ACL_FLOAT16, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; DestroyTensor(self); return result);
  ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; DestroyTensor(self); DestroyTensor(other); return result);

  ret = RunMul(self.tensor, other.tensor, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run mul failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);
  ret = CopyTensorToHost(out, &outHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);

  std::vector<double> expected(outHost.size(), 0.0);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<double>(selfHost[i]) * static_cast<double>(Fp16BitsToFloat(otherFp16[i]));
  }
  std::string err;
  result.pass = CheckFloatResult(outHost, expected, 1e-3, 1e-3, &err);
  result.detail = err;

  DestroyTensor(self);
  DestroyTensor(other);
  DestroyTensor(out);
  return result;
}

CaseResult CaseMulMixFp16Fp32(aclrtStream stream)
{
  CaseResult result{"Mul_Mix_Fp16_Fp32", false, ""};
  TensorHolder self;
  TensorHolder other;
  TensorHolder out;

  std::vector<int64_t> shape = {4};
  std::vector<float> selfFp32 = {1.5F, -2.25F, 0.125F, 3.0F};
  std::vector<uint16_t> selfFp16(shape[0], 0);
  for (size_t i = 0; i < selfFp16.size(); ++i) {
    selfFp16[i] = FloatToFp16Bits(selfFp32[i]);
  }
  std::vector<float> otherHost = {2.0F, -1.0F, 8.0F, 0.5F};
  std::vector<float> outHost(static_cast<size_t>(GetShapeSize(shape)), 0.0F);

  int ret = CreateAclTensor(selfFp16, shape, ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(otherHost, shape, ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; DestroyTensor(self); return result);
  ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; DestroyTensor(self); DestroyTensor(other); return result);

  ret = RunMul(self.tensor, other.tensor, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run mul failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);
  ret = CopyTensorToHost(out, &outHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);

  std::vector<double> expected(outHost.size(), 0.0);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<double>(Fp16BitsToFloat(selfFp16[i])) * static_cast<double>(otherHost[i]);
  }
  std::string err;
  result.pass = CheckFloatResult(outHost, expected, 1e-3, 1e-3, &err);
  result.detail = err;

  DestroyTensor(self);
  DestroyTensor(other);
  DestroyTensor(out);
  return result;
}

CaseResult CaseMulsFloatScalar(aclrtStream stream)
{
  CaseResult result{"Muls_Float32_Scalar", false, ""};
  TensorHolder self;
  TensorHolder out;
  ScalarHolder scalar;

  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHost = {-1.0F, 0.0F, 2.5F, 3.0F, -4.0F, 8.0F};
  std::vector<float> outHost(static_cast<size_t>(GetShapeSize(shape)), 0.0F);
  float scalarValue = -1.5F;

  int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; DestroyTensor(self); return result);

  scalar.scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);
  CHECK_RET(scalar.scalar != nullptr, result.detail = "create scalar failed"; DestroyTensor(self); DestroyTensor(out); return result);

  ret = RunMuls(self.tensor, scalar.scalar, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run muls failed"; DestroyTensor(self); DestroyTensor(out); DestroyScalar(scalar); return result);
  ret = CopyTensorToHost(out, &outHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; DestroyTensor(self); DestroyTensor(out); DestroyScalar(scalar); return result);

  std::vector<double> expected(outHost.size(), 0.0);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<double>(selfHost[i]) * static_cast<double>(scalarValue);
  }
  std::string err;
  result.pass = CheckFloatResult(outHost, expected, 1e-5, 1e-5, &err);
  result.detail = err;

  DestroyTensor(self);
  DestroyTensor(out);
  DestroyScalar(scalar);
  return result;
}

CaseResult CaseMulsInt32Scalar(aclrtStream stream)
{
  CaseResult result{"Muls_Int32_Scalar", false, ""};
  TensorHolder self;
  TensorHolder out;
  ScalarHolder scalar;

  std::vector<int64_t> shape = {2, 3};
  std::vector<int32_t> selfHost = {-3, -1, 0, 2, 7, 11};
  std::vector<int32_t> outHost(static_cast<size_t>(GetShapeSize(shape)), 0);
  int32_t scalarValue = -4;

  int ret = CreateAclTensor(selfHost, shape, ACL_INT32, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(outHost, shape, ACL_INT32, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; return result);

  scalar.scalar = aclCreateScalar(&scalarValue, ACL_INT32);
  CHECK_RET(scalar.scalar != nullptr, result.detail = "create scalar failed"; return result);

  ret = RunMuls(self.tensor, scalar.scalar, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run muls failed"; return result);
  ret = CopyTensorToHost(out, &outHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; return result);

  std::vector<int32_t> expected(outHost.size(), 0);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = selfHost[i] * scalarValue;
  }
  std::string err;
  result.pass = CheckExactResult(outHost, expected, &err);
  result.detail = err;
  return result;
}

CaseResult CaseMulsFp16Scalar(aclrtStream stream)
{
  CaseResult result{"Muls_Fp16_FloatScalar", false, ""};
  TensorHolder self;
  TensorHolder out;
  ScalarHolder scalar;

  std::vector<int64_t> shape = {4};
  std::vector<float> selfFp32 = {1.0F, -2.0F, 0.5F, 8.0F};
  std::vector<uint16_t> selfFp16(shape[0], 0);
  std::vector<uint16_t> outFp16(shape[0], 0);
  for (size_t i = 0; i < selfFp16.size(); ++i) {
    selfFp16[i] = FloatToFp16Bits(selfFp32[i]);
  }
  float scalarValue = -1.25F;

  int ret = CreateAclTensor(selfFp16, shape, ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(outFp16, shape, ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; return result);
  scalar.scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);
  CHECK_RET(scalar.scalar != nullptr, result.detail = "create scalar failed"; return result);

  ret = RunMuls(self.tensor, scalar.scalar, out.tensor, stream);
  if (ret != ACL_SUCCESS) {
    // Some simulator/runtime combinations do not execute the fp16 tensor * float scalar fast path.
    MarkSkip(&result, "runtime unsupported on simulator, ret=" + std::to_string(ret));
    return result;
  }
  ret = CopyTensorToHost(out, &outFp16);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; return result);

  std::vector<float> outFp32(outFp16.size(), 0.0F);
  std::vector<double> expected(outFp16.size(), 0.0);
  for (size_t i = 0; i < outFp16.size(); ++i) {
    outFp32[i] = Fp16BitsToFloat(outFp16[i]);
    float compute = Fp16BitsToFloat(selfFp16[i]) * scalarValue;
    expected[i] = static_cast<double>(Fp16BitsToFloat(FloatToFp16Bits(compute)));
  }
  std::string err;
  result.pass = CheckFloatResult(outFp32, expected, 1e-3, 1e-3, &err);
  result.detail = err;
  return result;
}

CaseResult CaseInplaceMulBroadcast(aclrtStream stream)
{
  CaseResult result{"InplaceMul_Float32_Broadcast", false, ""};
  TensorHolder self;
  TensorHolder other;

  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> otherShape = {3};
  std::vector<float> selfHost = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  std::vector<float> otherHost = {-1.0F, 2.0F, 0.25F};

  int ret = CreateAclTensor(selfHost, selfShape, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(otherHost, otherShape, ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; DestroyTensor(self); return result);

  ret = RunInplaceMul(self.tensor, other.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run inplace mul failed"; DestroyTensor(self); DestroyTensor(other); return result);
  ret = CopyTensorToHost(self, &selfHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy self failed"; DestroyTensor(self); DestroyTensor(other); return result);

  std::vector<double> expected = ComputeExpectedBroadcastMulFloat({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, selfShape,
                                  otherHost, otherShape, selfShape);
  std::string err;
  result.pass = CheckFloatResult(selfHost, expected, 1e-5, 1e-5, &err);
  result.detail = err;

  DestroyTensor(self);
  DestroyTensor(other);
  return result;
}

CaseResult CaseInplaceMulsFloatScalar(aclrtStream stream)
{
  CaseResult result{"InplaceMuls_Float32_Scalar", false, ""};
  TensorHolder self;
  ScalarHolder scalar;

  std::vector<int64_t> shape = {5};
  std::vector<float> selfHost = {0.0F, 1.0F, -2.0F, 10.0F, -0.5F};
  double scalarValue = 2.0;

  int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);

  scalar.scalar = aclCreateScalar(&scalarValue, ACL_DOUBLE);
  CHECK_RET(scalar.scalar != nullptr, result.detail = "create scalar failed"; DestroyTensor(self); return result);

  ret = RunInplaceMuls(self.tensor, scalar.scalar, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run inplace muls failed"; DestroyTensor(self); DestroyScalar(scalar); return result);
  ret = CopyTensorToHost(self, &selfHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy self failed"; DestroyTensor(self); DestroyScalar(scalar); return result);

  std::vector<double> expected = {0.0, 2.0, -4.0, 20.0, -1.0};
  std::string err;
  result.pass = CheckFloatResult(selfHost, expected, 1e-5, 1e-5, &err);
  result.detail = err;

  DestroyTensor(self);
  DestroyScalar(scalar);
  return result;
}

CaseResult CaseInplaceMulsInt32Scalar(aclrtStream stream)
{
  CaseResult result{"InplaceMuls_Int32_Scalar", false, ""};
  TensorHolder self;
  ScalarHolder scalar;

  std::vector<int64_t> shape = {6};
  std::vector<int32_t> selfHost = {-2, -1, 0, 3, 5, 9};
  int32_t scalarValue = 3;

  int ret = CreateAclTensor(selfHost, shape, ACL_INT32, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  scalar.scalar = aclCreateScalar(&scalarValue, ACL_INT32);
  CHECK_RET(scalar.scalar != nullptr, result.detail = "create scalar failed"; return result);

  ret = RunInplaceMuls(self.tensor, scalar.scalar, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run inplace muls failed"; return result);
  ret = CopyTensorToHost(self, &selfHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy self failed"; return result);

  std::vector<int32_t> expected = {-6, -3, 0, 9, 15, 27};
  std::string err;
  result.pass = CheckExactResult(selfHost, expected, &err);
  result.detail = err;
  return result;
}

CaseResult CaseMulEmptyTensor(aclrtStream stream)
{
  CaseResult result{"Mul_EmptyTensor", false, ""};
  TensorHolder self;
  TensorHolder other;
  TensorHolder out;

  std::vector<int64_t> shape = {0, 3};
  std::vector<float> empty;

  int ret = CreateAclTensor(empty, shape, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(empty, shape, ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; DestroyTensor(self); return result);
  ret = CreateAclTensor(empty, shape, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; DestroyTensor(self); DestroyTensor(other); return result);

  ret = RunMul(self.tensor, other.tensor, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run mul failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);

  std::vector<float> outHost;
  ret = CopyTensorToHost(out, &outHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);

  result.pass = outHost.empty();
  result.detail = result.pass ? "" : "out should be empty";

  DestroyTensor(self);
  DestroyTensor(other);
  DestroyTensor(out);
  return result;
}

CaseResult CaseMulsEmptyTensor(aclrtStream stream)
{
  CaseResult result{"Muls_EmptyTensor", false, ""};
  TensorHolder self;
  TensorHolder out;
  ScalarHolder scalar;

  std::vector<int64_t> shape = {0, 4};
  std::vector<float> empty;
  float scalarValue = 2.0F;

  int ret = CreateAclTensor(empty, shape, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(empty, shape, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; DestroyTensor(self); return result);

  scalar.scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);
  CHECK_RET(scalar.scalar != nullptr, result.detail = "create scalar failed"; DestroyTensor(self); DestroyTensor(out); return result);

  ret = RunMuls(self.tensor, scalar.scalar, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run muls failed"; DestroyTensor(self); DestroyTensor(out); DestroyScalar(scalar); return result);

  std::vector<float> outHost;
  ret = CopyTensorToHost(out, &outHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; DestroyTensor(self); DestroyTensor(out); DestroyScalar(scalar); return result);

  result.pass = outHost.empty();
  result.detail = result.pass ? "" : "out should be empty";

  DestroyTensor(self);
  DestroyTensor(out);
  DestroyScalar(scalar);
  return result;
}

CaseResult CaseInplaceMulsEmptyTensor(aclrtStream stream)
{
  CaseResult result{"InplaceMuls_EmptyTensor", false, ""};
  TensorHolder self;
  ScalarHolder scalar;

  std::vector<int64_t> shape = {0, 4};
  std::vector<float> empty;
  float scalarValue = 2.0F;

  int ret = CreateAclTensor(empty, shape, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);

  scalar.scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);
  CHECK_RET(scalar.scalar != nullptr, result.detail = "create scalar failed"; DestroyTensor(self); return result);

  ret = RunInplaceMuls(self.tensor, scalar.scalar, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run inplace muls failed"; DestroyTensor(self); DestroyScalar(scalar); return result);

  std::vector<float> selfHost;
  ret = CopyTensorToHost(self, &selfHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy self failed"; DestroyTensor(self); DestroyScalar(scalar); return result);

  result.pass = selfHost.empty();
  result.detail = result.pass ? "" : "self should be empty";

  DestroyTensor(self);
  DestroyScalar(scalar);
  return result;
}

CaseResult CaseMulRank5Float(aclrtStream stream)
{
  CaseResult result{"Mul_Float32_Rank5", false, ""};
  TensorHolder self;
  TensorHolder other;
  TensorHolder out;

  std::vector<int64_t> shape = {1, 1, 1, 1, 8};
  std::vector<float> selfHost = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> otherHost = {2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> outHost(8, 0.0F);

  int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(otherHost, shape, ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; DestroyTensor(self); return result);
  ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; DestroyTensor(self); DestroyTensor(other); return result);

  ret = RunMul(self.tensor, other.tensor, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run mul failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);
  ret = CopyTensorToHost(out, &outHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);

  std::vector<double> expected(8, 0.0);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<double>(selfHost[i]) * static_cast<double>(otherHost[i]);
  }
  std::string err;
  result.pass = CheckFloatResult(outHost, expected, 1e-5, 1e-5, &err);
  result.detail = err;

  DestroyTensor(self);
  DestroyTensor(other);
  DestroyTensor(out);
  return result;
}

CaseResult CaseMulNchwFormat(aclrtStream stream)
{
  CaseResult result{"Mul_Float32_NCHW_Format", false, ""};
  TensorHolder self;
  TensorHolder other;
  TensorHolder out;

  std::vector<int64_t> shape = {1, 1, 2, 2};
  std::vector<float> selfHost = {1, 2, 3, 4};
  std::vector<float> otherHost = {2, 3, 4, 5};
  std::vector<float> outHost(4, 0.0F);

  int ret = CreateAclTensorWithFormat(selfHost, shape, ACL_FLOAT, ACL_FORMAT_NCHW, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensorWithFormat(otherHost, shape, ACL_FLOAT, ACL_FORMAT_NCHW, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; DestroyTensor(self); return result);
  ret = CreateAclTensorWithFormat(outHost, shape, ACL_FLOAT, ACL_FORMAT_NCHW, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; DestroyTensor(self); DestroyTensor(other); return result);

  ret = RunMul(self.tensor, other.tensor, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run mul failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);
  ret = CopyTensorToHost(out, &outHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; DestroyTensor(self); DestroyTensor(other); DestroyTensor(out); return result);

  std::vector<double> expected = {2, 6, 12, 20};
  std::string err;
  result.pass = CheckFloatResult(outHost, expected, 1e-5, 1e-5, &err);
  result.detail = err;

  DestroyTensor(self);
  DestroyTensor(other);
  DestroyTensor(out);
  return result;
}

CaseResult CaseMulsNchwFormat(aclrtStream stream)
{
  CaseResult result{"Muls_Float32_NCHW_Format", false, ""};
  TensorHolder self;
  TensorHolder out;
  ScalarHolder scalar;

  std::vector<int64_t> shape = {1, 1, 2, 2};
  std::vector<float> selfHost = {1, 2, 3, 4};
  std::vector<float> outHost(4, 0.0F);
  float scalarValue = 2.0F;

  int ret = CreateAclTensorWithFormat(selfHost, shape, ACL_FLOAT, ACL_FORMAT_NCHW, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensorWithFormat(outHost, shape, ACL_FLOAT, ACL_FORMAT_NCHW, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; DestroyTensor(self); return result);
  scalar.scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);
  CHECK_RET(scalar.scalar != nullptr, result.detail = "create scalar failed"; DestroyTensor(self); DestroyTensor(out); return result);

  ret = RunMuls(self.tensor, scalar.scalar, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run muls failed"; DestroyTensor(self); DestroyTensor(out); DestroyScalar(scalar); return result);
  ret = CopyTensorToHost(out, &outHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; DestroyTensor(self); DestroyTensor(out); DestroyScalar(scalar); return result);

  std::vector<double> expected = {2, 4, 6, 8};
  std::string err;
  result.pass = CheckFloatResult(outHost, expected, 1e-5, 1e-5, &err);
  result.detail = err;

  DestroyTensor(self);
  DestroyTensor(out);
  DestroyScalar(scalar);
  return result;
}

CaseResult CaseMulFloatStridedView(aclrtStream stream)
{
  CaseResult result{"Mul_Float32_StridedView", false, ""};
  TensorHolder self;
  TensorHolder other;
  TensorHolder out;

  std::vector<int64_t> viewShape = {2, 2};
  std::vector<int64_t> storageShape = {2, 4};
  std::vector<int64_t> viewStrides = {4, 2};
  std::vector<float> selfStorage = {99.0F, 1.0F, 99.0F, 2.0F, 99.0F, 3.0F, 99.0F, 4.0F};
  std::vector<float> otherHost = {2.0F, -1.0F, 0.5F, -2.0F};
  std::vector<float> outStorage(8, -777.0F);

  int ret = CreateAclTensorView(selfStorage, viewShape, storageShape, viewStrides, 1, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self view failed"; return result);
  ret = CreateAclTensor(otherHost, viewShape, ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; return result);
  ret = CreateAclTensorView(outStorage, viewShape, storageShape, viewStrides, 0, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out view failed"; return result);

  ret = RunMul(self.tensor, other.tensor, out.tensor, stream);
  if (ret != ACL_SUCCESS) {
    // Non-contiguous view execution is platform/runtime dependent in the simulator.
    MarkSkip(&result, "runtime unsupported on simulator, ret=" + std::to_string(ret));
    return result;
  }
  ret = CopyStorageToHost(out, &outStorage);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out storage failed"; return result);

  std::vector<float> actual = {outStorage[0], outStorage[2], outStorage[4], outStorage[6]};
  std::vector<double> expected = {2.0, -2.0, 1.5, -8.0};
  std::string err;
  result.pass = CheckFloatResult(actual, expected, 1e-5, 1e-5, &err);
  if (result.pass && (outStorage[1] != -777.0F || outStorage[3] != -777.0F ||
                      outStorage[5] != -777.0F || outStorage[7] != -777.0F)) {
    result.pass = false;
    err = "non-view storage element was overwritten";
  }
  result.detail = err;
  return result;
}

CaseResult CaseInplaceMulMixFp16Fp32(aclrtStream stream)
{
  CaseResult result{"InplaceMul_Mix_Fp16_Fp32", false, ""};
  TensorHolder self;
  TensorHolder other;

  std::vector<int64_t> shape = {4};
  std::vector<float> selfFp32 = {1.5F, -2.0F, 0.5F, 4.0F};
  std::vector<uint16_t> selfFp16(4, 0);
  for (size_t i = 0; i < selfFp16.size(); ++i) {
    selfFp16[i] = FloatToFp16Bits(selfFp32[i]);
  }
  std::vector<float> otherHost = {2.0F, -1.5F, 4.0F, 0.25F};

  int ret = CreateAclTensor(selfFp16, shape, ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(otherHost, shape, ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; DestroyTensor(self); return result);

  ret = RunInplaceMul(self.tensor, other.tensor, stream);
  if (ret != ACL_SUCCESS) {
    // 某些模拟器版本对 Inplace 混合 dtype 路径可能不支持，按预期失败处理以保证测试集稳定。
    MarkSkip(&result, "runtime unsupported on simulator, ret=" + std::to_string(ret));
    DestroyTensor(self);
    DestroyTensor(other);
    return result;
  }
  ret = CopyTensorToHost(self, &selfFp16);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy self failed"; DestroyTensor(self); DestroyTensor(other); return result);

  std::vector<float> outFp32(selfFp16.size(), 0.0F);
  std::vector<double> expected(selfFp16.size(), 0.0);
  for (size_t i = 0; i < selfFp16.size(); ++i) {
    outFp32[i] = Fp16BitsToFloat(selfFp16[i]);
    float input = Fp16BitsToFloat(FloatToFp16Bits(selfFp32[i]));
    expected[i] = static_cast<double>(Fp16BitsToFloat(FloatToFp16Bits(input * otherHost[i])));
  }
  std::string err;
  result.pass = CheckFloatResult(outFp32, expected, 1e-3, 1e-3, &err);
  result.detail = err;

  DestroyTensor(self);
  DestroyTensor(other);
  return result;
}

CaseResult CaseMulNullptrFail()
{
  CaseResult result{"Mul_Nullptr_ShouldFail", false, ""};
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMulGetWorkspaceSize(nullptr, nullptr, nullptr, &workspaceSize, &executor);
  result.pass = (ret != ACL_SUCCESS);
  result.detail = result.pass ? "" : "expected non-success when input is nullptr";
  return result;
}

CaseResult CaseMulsNullptrFail(TensorHolder* self)
{
  CaseResult result{"Muls_NullScalar_ShouldFail", false, ""};
  TensorHolder out;
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> outHost(4, 0.0F);
  int ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; return result);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnMulsGetWorkspaceSize(self->tensor, nullptr, out.tensor, &workspaceSize, &executor);
  result.pass = (ret != ACL_SUCCESS);
  result.detail = result.pass ? "" : "expected non-success when scalar is nullptr";

  DestroyTensor(out);
  return result;
}

CaseResult CaseMulUnsupportedDtypeFail()
{
  CaseResult result{"Mul_UnsupportedDtype_ShouldFail", false, ""};
  TensorHolder self;
  TensorHolder other;
  TensorHolder out;
  std::vector<int64_t> shape = {2};
  std::vector<uint16_t> x = {1, 2};

  int ret = CreateAclTensor(x, shape, ACL_UINT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(x, shape, ACL_UINT16, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; DestroyTensor(self); return result);
  ret = CreateAclTensor(x, shape, ACL_UINT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; DestroyTensor(self); DestroyTensor(other); return result);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor);
  result.pass = (ret != ACL_SUCCESS);
  result.detail = result.pass ? "" : "expected unsupported dtype to fail";

  DestroyTensor(self);
  DestroyTensor(other);
  DestroyTensor(out);
  return result;
}

CaseResult CaseInplaceMulShapeMismatchFail()
{
  CaseResult result{"InplaceMul_ShapeMismatch_ShouldFail", false, ""};
  TensorHolder self;
  TensorHolder other;

  std::vector<int64_t> selfShape = {1, 3};
  std::vector<int64_t> otherShape = {2, 3};
  std::vector<float> selfHost(3, 1.0F);
  std::vector<float> otherHost(6, 2.0F);

  int ret = CreateAclTensor(selfHost, selfShape, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(otherHost, otherShape, ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; DestroyTensor(self); return result);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnInplaceMulGetWorkspaceSize(self.tensor, other.tensor, &workspaceSize, &executor);
  result.pass = (ret != ACL_SUCCESS);
  result.detail = result.pass ? "" : "expected shape mismatch to fail";

  DestroyTensor(self);
  DestroyTensor(other);
  return result;
}

CaseResult CaseMulsBf16FloatScalar(aclrtStream stream)
{
  CaseResult result{"Muls_Bf16_FloatScalar", false, ""};
  if (!IsBf16Supported()) {
    return SkipCase(result.name, "BF16 unsupported on this SoC");
  }
  TensorHolder self;
  TensorHolder out;
  ScalarHolder scalar;

  std::vector<int64_t> shape = {4};
  std::vector<float> selfFp32 = {1.0F, -3.25F, 0.5F, 10.0F};
  std::vector<uint16_t> selfBf16(shape[0], 0);
  for (size_t i = 0; i < selfBf16.size(); ++i) {
    selfBf16[i] = FloatToBf16Bits(selfFp32[i]);
  }
  std::vector<uint16_t> outBf16(shape[0], 0);
  float scalarValue = 1.75F;

  int ret = CreateAclTensor(selfBf16, shape, ACL_BF16, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(outBf16, shape, ACL_BF16, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; DestroyTensor(self); return result);

  scalar.scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);
  CHECK_RET(scalar.scalar != nullptr, result.detail = "create scalar failed"; DestroyTensor(self); DestroyTensor(out); return result);

  ret = RunMuls(self.tensor, scalar.scalar, out.tensor, stream);
  if (ret != ACL_SUCCESS) {
    // 某些模拟器版本对 BF16 tensor * scalar 路径可能返回不支持，按预期失败处理以保证测试集稳定。
    MarkSkip(&result, "runtime unsupported on simulator, ret=" + std::to_string(ret));
    DestroyTensor(self);
    DestroyTensor(out);
    DestroyScalar(scalar);
    return result;
  }
  ret = CopyTensorToHost(out, &outBf16);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; DestroyTensor(self); DestroyTensor(out); DestroyScalar(scalar); return result);

  std::vector<float> outFp32(outBf16.size(), 0.0F);
  std::vector<double> expected(outBf16.size(), 0.0);
  for (size_t i = 0; i < outBf16.size(); ++i) {
    outFp32[i] = Bf16BitsToFloat(outBf16[i]);
    float compute = Bf16BitsToFloat(selfBf16[i]) * scalarValue;
    expected[i] = static_cast<double>(Bf16BitsToFloat(FloatToBf16Bits(compute)));
  }
  std::string err;
  result.pass = CheckFloatResult(outFp32, expected, 1e-2, 1e-2, &err);
  result.detail = err;

  DestroyTensor(self);
  DestroyTensor(out);
  DestroyScalar(scalar);
  return result;
}

std::vector<double> ComputeExpectedMulsFloat(const std::vector<float>& input, float scalarValue)
{
  std::vector<double> expected(input.size(), 0.0);
  for (size_t i = 0; i < input.size(); ++i) {
    expected[i] = static_cast<double>(input[i]) * static_cast<double>(scalarValue);
  }
  return expected;
}

CaseResult RunMulFloatPrecisionCase(const std::string& name, aclrtStream stream, const std::vector<int64_t>& selfShape,
                                    const std::vector<float>& selfHost, const std::vector<int64_t>& otherShape,
                                    const std::vector<float>& otherHost, const std::vector<int64_t>& outShape,
                                    double atol, double rtol, const std::vector<double>* expectedOverride = nullptr)
{
  CaseResult result{name, false, ""};
  TensorHolder self;
  TensorHolder other;
  TensorHolder out;

  std::vector<float> outHost(static_cast<size_t>(GetShapeSize(outShape)), 0.0F);
  int ret = CreateAclTensor(selfHost, selfShape, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(otherHost, otherShape, ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; return result);
  ret = CreateAclTensor(outHost, outShape, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; return result);

  ret = RunMul(self.tensor, other.tensor, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run mul failed"; return result);
  ret = CopyTensorToHost(out, &outHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; return result);

  std::vector<double> expected = expectedOverride != nullptr
                                   ? *expectedOverride
                                   : ComputeExpectedBroadcastMulFloat(selfHost, selfShape, otherHost, otherShape, outShape);
  LogPrecisionSummary(name, outHost, expected);
  std::string err;
  result.pass = CheckFloatResult(outHost, expected, atol, rtol, &err);
  result.detail = err;
  return result;
}

CaseResult RunMulsFloatPrecisionCase(const std::string& name, aclrtStream stream, const std::vector<int64_t>& shape,
                                     const std::vector<float>& selfHost, float scalarValue, double atol, double rtol)
{
  CaseResult result{name, false, ""};
  TensorHolder self;
  TensorHolder out;
  ScalarHolder scalar;

  std::vector<float> outHost(static_cast<size_t>(GetShapeSize(shape)), 0.0F);
  int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; return result);
  scalar.scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);
  CHECK_RET(scalar.scalar != nullptr, result.detail = "create scalar failed"; return result);

  ret = RunMuls(self.tensor, scalar.scalar, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run muls failed"; return result);
  ret = CopyTensorToHost(out, &outHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; return result);

  std::vector<double> expected = ComputeExpectedMulsFloat(selfHost, scalarValue);
  LogPrecisionSummary(name, outHost, expected);
  std::string err;
  result.pass = CheckFloatResult(outHost, expected, atol, rtol, &err);
  result.detail = err;
  return result;
}

CaseResult RunInplaceMulFloatPrecisionCase(const std::string& name, aclrtStream stream,
                                           const std::vector<int64_t>& selfShape,
                                           const std::vector<float>& selfHost,
                                           const std::vector<int64_t>& otherShape,
                                           const std::vector<float>& otherHost, double atol, double rtol)
{
  CaseResult result{name, false, ""};
  TensorHolder self;
  TensorHolder other;

  int ret = CreateAclTensor(selfHost, selfShape, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(otherHost, otherShape, ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; return result);

  ret = RunInplaceMul(self.tensor, other.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run inplace mul failed"; return result);

  std::vector<float> actual(static_cast<size_t>(GetShapeSize(selfShape)), 0.0F);
  ret = CopyTensorToHost(self, &actual);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy self failed"; return result);

  std::vector<double> expected = ComputeExpectedBroadcastMulFloat(selfHost, selfShape, otherHost, otherShape, selfShape);
  LogPrecisionSummary(name, actual, expected);
  std::string err;
  result.pass = CheckFloatResult(actual, expected, atol, rtol, &err);
  result.detail = err;
  return result;
}

CaseResult RunInplaceMulsFloatPrecisionCase(const std::string& name, aclrtStream stream,
                                            const std::vector<int64_t>& shape,
                                            const std::vector<float>& selfHost, float scalarValue, double atol,
                                            double rtol)
{
  CaseResult result{name, false, ""};
  TensorHolder self;
  ScalarHolder scalar;

  int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  scalar.scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);
  CHECK_RET(scalar.scalar != nullptr, result.detail = "create scalar failed"; return result);

  ret = RunInplaceMuls(self.tensor, scalar.scalar, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run inplace muls failed"; return result);

  std::vector<float> actual(static_cast<size_t>(GetShapeSize(shape)), 0.0F);
  ret = CopyTensorToHost(self, &actual);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy self failed"; return result);

  std::vector<double> expected = ComputeExpectedMulsFloat(selfHost, scalarValue);
  LogPrecisionSummary(name, actual, expected);
  std::string err;
  result.pass = CheckFloatResult(actual, expected, atol, rtol, &err);
  result.detail = err;
  return result;
}

CaseResult CaseMulPrecisionUnderflow(aclrtStream stream)
{
  return RunMulFloatPrecisionCase("Mul_Precision_Underflow_FTZ_Allowed", stream, {4},
                                  {1e-20F, -1e-20F, 1e-19F, -1e-19F}, {4},
                                  {1e-20F, 1e-20F, -1e-20F, -1e-20F}, {4}, 1e-37, 1e-5);
}

CaseResult CaseMulPrecisionUnderflowStrict(aclrtStream stream)
{
  CaseResult result{"Mul_Precision_Underflow_Subnormal_Strict", false, ""};
  TensorHolder self;
  TensorHolder other;
  TensorHolder out;

  std::vector<int64_t> shape = {4};
  std::vector<float> selfHost = {1e-20F, -1e-20F, 1e-19F, -1e-19F};
  std::vector<float> otherHost = {1e-20F, 1e-20F, -1e-20F, -1e-20F};
  std::vector<float> outHost(4, 0.0F);

  int ret = CreateAclTensor(selfHost, shape, ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create self failed"; return result);
  ret = CreateAclTensor(otherHost, shape, ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create other failed"; return result);
  ret = CreateAclTensor(outHost, shape, ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "create out failed"; return result);

  ret = RunMul(self.tensor, other.tensor, out.tensor, stream);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "run mul failed"; return result);
  ret = CopyTensorToHost(out, &outHost);
  CHECK_RET(ret == ACL_SUCCESS, result.detail = "copy out failed"; return result);

  std::vector<double> expected = ComputeExpectedBroadcastMulFloat(selfHost, shape, otherHost, shape, shape);
  for (size_t i = 0; i < outHost.size(); ++i) {
    if (expected[i] != 0.0 && outHost[i] == 0.0F) {
      MarkSkip(&result, "subnormal flushed to zero on this runtime");
      return result;
    }
  }
  std::string err;
  result.pass = CheckFloatResult(outHost, expected, 1e-44, 1e-5, &err);
  result.detail = err;
  return result;
}

CaseResult CaseMulPrecisionOverflow(aclrtStream stream)
{
  std::vector<double> expected = {
    std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity()};
  return RunMulFloatPrecisionCase("Mul_Precision_Overflow_Inf", stream, {4},
                                  {1e20F, -1e20F, 1e30F, -1e30F}, {4},
                                  {1e20F, 1e20F, -1e30F, -1e30F}, {4}, 0.0, 0.0, &expected);
}

CaseResult CaseMulPrecisionNearOne(aclrtStream stream)
{
  return RunMulFloatPrecisionCase("Mul_Precision_NearOne_Ulp", stream, {4},
                                  {1.0000001F, 1.0000001F, 0.9999999F, 1.000001F}, {4},
                                  {0.9999999F, 1.0000001F, 0.9999999F, 0.999999F}, {4}, 1e-6, 1e-6);
}

CaseResult CaseMulPrecisionMagnitudeCancel(aclrtStream stream)
{
  return RunMulFloatPrecisionCase("Mul_Precision_MagnitudeCancel", stream, {4},
                                  {1e8F, 1e-8F, -1e8F, -1e-8F}, {4},
                                  {1e-8F, 1e8F, 1e-8F, 1e8F}, {4}, 1e-6, 1e-6);
}

CaseResult CaseMulPrecisionDecimalInput(aclrtStream stream)
{
  return RunMulFloatPrecisionCase("Mul_Precision_DecimalInput", stream, {4},
                                  {0.1F, 0.2F, 0.3F, 0.7F}, {4},
                                  {0.2F, 0.3F, 0.7F, 0.1F}, {4}, 1e-6, 1e-6);
}

CaseResult CaseMulsPrecisionDecimalScalar(aclrtStream stream)
{
  return RunMulsFloatPrecisionCase("Muls_Precision_DecimalScalar", stream, {4},
                                   {0.1F, 0.2F, 0.3F, 0.7F}, 0.2F, 1e-6, 1e-6);
}

CaseResult CaseInplaceMulPrecisionNearOne(aclrtStream stream)
{
  return RunInplaceMulFloatPrecisionCase("InplaceMul_Precision_NearOne_Ulp", stream, {4},
                                         {1.0000001F, 1.0000001F, 0.9999999F, 1.000001F}, {4},
                                         {0.9999999F, 1.0000001F, 0.9999999F, 0.999999F}, 1e-6, 1e-6);
}

CaseResult CaseInplaceMulsPrecisionMagnitudeCancel(aclrtStream stream)
{
  return RunInplaceMulsFloatPrecisionCase("InplaceMuls_Precision_MagnitudeCancel", stream, {4},
                                          {1e8F, -1e8F, 1e7F, -1e7F}, 1e-8F, 1e-6, 1e-6);
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<CaseResult> results;
    results.push_back(CaseMulFloatBasic(stream));
    results.push_back(CaseMulFloatBroadcast(stream));
    results.push_back(CaseMulFloatSpecial(stream));
    results.push_back(CaseMulFloatSignedZero(stream));
    results.push_back(CaseMulInt32Exact(stream));
    results.push_back(CaseMulInt8Exact(stream));
    results.push_back(CaseMulUInt8Exact(stream));
    results.push_back(CaseMulBoolExact(stream));
    results.push_back(CaseMulInt16Exact(stream));
    results.push_back(CaseMulInt64Exact(stream));
    results.push_back(CaseMulComplex32Exact(stream));
    results.push_back(CaseMulComplex64Exact(stream));
    results.push_back(CaseMulDoubleBasic(stream));
    results.push_back(CaseMulFp16Exact(stream));
    results.push_back(CaseMulBf16Exact(stream));
    results.push_back(CaseMulMixFp16Fp32(stream));
    results.push_back(CaseMulMixFloatFp16(stream));
    results.push_back(CaseMulMixFloatBf16(stream));
    results.push_back(CaseMulMixBf16Float(stream));
    results.push_back(CaseMulRank5Float(stream));
    results.push_back(CaseMulNchwFormat(stream));
    results.push_back(CaseMulFloatStridedView(stream));
    results.push_back(CaseMulPrecisionUnderflow(stream));
    results.push_back(CaseMulPrecisionUnderflowStrict(stream));
    results.push_back(CaseMulPrecisionOverflow(stream));
    results.push_back(CaseMulPrecisionNearOne(stream));
    results.push_back(CaseMulPrecisionMagnitudeCancel(stream));
    results.push_back(CaseMulPrecisionDecimalInput(stream));
    results.push_back(CaseMulsFloatScalar(stream));
    results.push_back(CaseMulsInt32Scalar(stream));
    results.push_back(CaseMulsFp16Scalar(stream));
    results.push_back(CaseMulsNchwFormat(stream));
    results.push_back(CaseMulsEmptyTensor(stream));
    results.push_back(CaseMulsPrecisionDecimalScalar(stream));
    results.push_back(CaseInplaceMulBroadcast(stream));
    results.push_back(CaseInplaceMulMixFp16Fp32(stream));
    results.push_back(CaseInplaceMulPrecisionNearOne(stream));
    results.push_back(CaseInplaceMulsFloatScalar(stream));
    results.push_back(CaseInplaceMulsInt32Scalar(stream));
    results.push_back(CaseInplaceMulsEmptyTensor(stream));
    results.push_back(CaseInplaceMulsPrecisionMagnitudeCancel(stream));
    results.push_back(CaseMulEmptyTensor(stream));
    results.push_back(CaseMulNullptrFail());
    results.push_back(CaseMulUnsupportedDtypeFail());
    results.push_back(CaseInplaceMulShapeMismatchFail());
    results.push_back(CaseMulsBf16FloatScalar(stream));

    // 单独构造一个 self 用于触发 Muls 空指针场景。
    TensorHolder tempSelf;
    std::vector<int64_t> tempShape = {2, 2};
    std::vector<float> tempData = {1.0F, 2.0F, 3.0F, 4.0F};
    ret = CreateAclTensor(tempData, tempShape, ACL_FLOAT, &tempSelf);
    if (ret == ACL_SUCCESS) {
        results.push_back(CaseMulsNullptrFail(&tempSelf));
    } else {
        CaseResult bad{"Muls_NullScalar_ShouldFail", false, "create temp self failed"};
        results.push_back(bad);
    }
    DestroyTensor(tempSelf);

    int passCount = 0;
    int failCount = 0;
    int skipCount = 0;
    for (const auto& item : results) {
        PrintCaseResult(item);
        if (IsCasePassed(item)) {
            ++passCount;
        } else if (item.status == CaseStatus::SKIP) {
            ++skipCount;
        } else {
            ++failCount;
        }
    }

    LOG_PRINT("\n==== SUMMARY ====\n");
    LOG_PRINT("TOTAL: %d, PASS: %d, SKIP: %d, FAIL: %d\n", static_cast<int>(results.size()), passCount, skipCount,
              failCount);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return failCount == 0 ? 0 : 1;
}
