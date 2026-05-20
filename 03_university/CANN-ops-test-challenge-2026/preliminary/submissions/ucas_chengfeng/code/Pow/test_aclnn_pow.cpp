/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl.h>

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if __has_include("aclnnop/aclnn_pow.h")
#include "aclnnop/aclnn_pow.h"
#else
#include "../op_api/aclnn_pow.h"
#endif

#if __has_include("aclnnop/aclnn_pow_tensor_tensor.h")
#include "aclnnop/aclnn_pow_tensor_tensor.h"
#else
#include "../op_api/aclnn_pow_tensor_tensor.h"
#endif

#if __has_include("aclnnop/aclnn_exp2.h")
#include "aclnnop/aclnn_exp2.h"
#else
#include "../op_api/aclnn_exp2.h"
#endif

#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS 0
#endif

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

enum class DType {
  kFloat32,
  kFloat16,
  kBFloat16,
  kDouble,
  kUInt8,
  kInt8,
  kInt16,
  kInt32,
  kInt64,
  kBool,
};

enum class ValueKind {
  kFinite,
  kNaN,
  kPosInf,
  kNegInf,
};

struct Tolerance {
  double atol = 0.0;
  double rtol = 0.0;
};

struct Value {
  ValueKind kind = ValueKind::kFinite;
  long double value = 0.0;
};

struct TensorSpec {
  std::vector<int64_t> shape;
  DType dtype = DType::kFloat32;
  std::vector<std::string> data;
};

struct ScalarSpec {
  DType dtype = DType::kFloat32;
  std::string value;
};

struct TensorScalarCase {
  const char *name;
  TensorSpec self;
  ScalarSpec exponent;
  DType outDtype;
  int expectStatus;
  Tolerance tol;
};

struct ScalarTensorCase {
  const char *name;
  ScalarSpec self;
  TensorSpec exponent;
  DType outDtype;
  int expectStatus;
  Tolerance tol;
};

struct TensorTensorCase {
  const char *name;
  TensorSpec self;
  TensorSpec exponent;
  std::vector<int64_t> outShape;
  DType outDtype;
  int expectStatus;
  Tolerance tol;
};

struct Exp2Case {
  const char *name;
  TensorSpec self;
  DType outDtype;
  int expectStatus;
  Tolerance tol;
};

struct DeviceTensor {
  void *deviceAddr = nullptr;
  aclTensor *tensor = nullptr;
};

struct HostScalar {
  std::vector<uint8_t> storage;
  aclScalar *scalar = nullptr;
};

struct Summary {
  int total = 0;
  int failed = 0;
  int skipped = 0;
};

std::string FormatVectorPreview(const std::vector<std::string> &values)
{
  std::string text = "[";
  const size_t limit = values.size() < 8 ? values.size() : 8;
  for (size_t i = 0; i < limit; ++i) {
    if (i != 0) {
      text += ", ";
    }
    text += values[i];
  }
  if (values.size() > limit) {
    text += ", ...";
  }
  text += "]";
  return text;
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
  if (shape.empty()) {
    return 1;
  }
  int64_t shapeSize = 1;
  for (int64_t dim : shape) {
    shapeSize *= dim;
  }
  return shapeSize;
}

std::vector<int64_t> ComputeStrides(const std::vector<int64_t> &shape)
{
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
  }
  return strides;
}

std::vector<int64_t> UnravelIndex(int64_t flatIndex, const std::vector<int64_t> &shape)
{
  if (shape.empty()) {
    return {};
  }
  std::vector<int64_t> index(shape.size(), 0);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    const int64_t dim = shape[static_cast<size_t>(i)];
    index[static_cast<size_t>(i)] = flatIndex % dim;
    flatIndex /= dim;
  }
  return index;
}

int64_t BroadcastOffset(const std::vector<int64_t> &outIndex, const std::vector<int64_t> &shape)
{
  if (shape.empty()) {
    return 0;
  }
  const std::vector<int64_t> strides = ComputeStrides(shape);
  const size_t outRank = outIndex.size();
  const size_t rank = shape.size();
  int64_t offset = 0;
  for (size_t i = 0; i < rank; ++i) {
    const size_t outAxis = outRank - rank + i;
    const int64_t idx = shape[i] == 1 ? 0 : outIndex[outAxis];
    offset += idx * strides[i];
  }
  return offset;
}

std::vector<int64_t> InferBroadcastShape(const std::vector<int64_t> &lhs, const std::vector<int64_t> &rhs)
{
  const size_t maxRank = lhs.size() > rhs.size() ? lhs.size() : rhs.size();
  std::vector<int64_t> shape(maxRank, 1);
  for (size_t i = 0; i < maxRank; ++i) {
    const int64_t left = i < maxRank - lhs.size() ? 1 : lhs[i - (maxRank - lhs.size())];
    const int64_t right = i < maxRank - rhs.size() ? 1 : rhs[i - (maxRank - rhs.size())];
    shape[i] = left > right ? left : right;
  }
  return shape;
}

std::string ToLower(std::string value)
{
  for (char &ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

Value ParseValue(const std::string &token)
{
  const std::string lower = ToLower(token);
  if (lower == "nan") {
    return {ValueKind::kNaN, 0.0};
  }
  if (lower == "inf" || lower == "+inf" || lower == "infinity" || lower == "+infinity") {
    return {ValueKind::kPosInf, 0.0};
  }
  if (lower == "-inf" || lower == "-infinity") {
    return {ValueKind::kNegInf, 0.0};
  }
  if (lower == "true") {
    return {ValueKind::kFinite, 1.0};
  }
  if (lower == "false") {
    return {ValueKind::kFinite, 0.0};
  }
  return {ValueKind::kFinite, std::strtold(token.c_str(), nullptr)};
}

std::string FormatValue(const Value &value)
{
  if (value.kind == ValueKind::kNaN) {
    return "nan";
  }
  if (value.kind == ValueKind::kPosInf) {
    return "inf";
  }
  if (value.kind == ValueKind::kNegInf) {
    return "-inf";
  }
  char buffer[64] = {0};
  std::snprintf(buffer, sizeof(buffer), "%.17g", static_cast<double>(value.value));
  return buffer;
}

aclDataType ToAclDataType(DType dtype)
{
  switch (dtype) {
    case DType::kFloat32:
      return ACL_FLOAT;
    case DType::kFloat16:
      return ACL_FLOAT16;
    case DType::kBFloat16:
      return ACL_BF16;
    case DType::kDouble:
      return ACL_DOUBLE;
    case DType::kUInt8:
      return ACL_UINT8;
    case DType::kInt8:
      return ACL_INT8;
    case DType::kInt16:
      return ACL_INT16;
    case DType::kInt32:
      return ACL_INT32;
    case DType::kInt64:
      return ACL_INT64;
    case DType::kBool:
      return ACL_BOOL;
  }
  return ACL_DT_UNDEFINED;
}

const char *DTypeName(DType dtype)
{
  switch (dtype) {
    case DType::kFloat32:
      return "float32";
    case DType::kFloat16:
      return "float16";
    case DType::kBFloat16:
      return "bfloat16";
    case DType::kDouble:
      return "double";
    case DType::kUInt8:
      return "uint8";
    case DType::kInt8:
      return "int8";
    case DType::kInt16:
      return "int16";
    case DType::kInt32:
      return "int32";
    case DType::kInt64:
      return "int64";
    case DType::kBool:
      return "bool";
  }
  return "unknown";
}

size_t DTypeSize(DType dtype)
{
  switch (dtype) {
    case DType::kFloat32:
      return sizeof(float);
    case DType::kFloat16:
      return sizeof(aclFloat16);
    case DType::kBFloat16:
      return sizeof(uint16_t);
    case DType::kDouble:
      return sizeof(double);
    case DType::kUInt8:
      return sizeof(uint8_t);
    case DType::kInt8:
      return sizeof(int8_t);
    case DType::kInt16:
      return sizeof(int16_t);
    case DType::kInt32:
      return sizeof(int32_t);
    case DType::kInt64:
      return sizeof(int64_t);
    case DType::kBool:
      return sizeof(uint8_t);
  }
  return 0;
}

Tolerance DefaultTolerance(DType dtype)
{
  switch (dtype) {
    case DType::kFloat32:
      return {1e-5, 1e-5};
    case DType::kFloat16:
      return {1e-3, 1e-3};
    case DType::kBFloat16:
      return {2e-2, 2e-2};
    case DType::kDouble:
      return {1e-12, 1e-12};
    default:
      return {0.0, 0.0};
  }
}

bool IsFloatingDType(DType dtype)
{
  return dtype == DType::kFloat32 || dtype == DType::kFloat16 || dtype == DType::kBFloat16 || dtype == DType::kDouble;
}

uint16_t FloatToBFloat16(float value)
{
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1U;
  bits += 0x7FFFU + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

float BFloat16ToFloat(uint16_t value)
{
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

template <typename T>
void AppendBytes(std::vector<uint8_t> *buffer, T value)
{
  const uint8_t *src = reinterpret_cast<const uint8_t *>(&value);
  buffer->insert(buffer->end(), src, src + sizeof(T));
}

std::vector<uint8_t> EncodeValues(const std::vector<std::string> &tokens, DType dtype)
{
  std::vector<uint8_t> bytes;
  bytes.reserve(tokens.size() * DTypeSize(dtype));
  for (const std::string &token : tokens) {
    const Value parsed = ParseValue(token);
    switch (dtype) {
      case DType::kFloat32: {
        float value = 0.0f;
        if (parsed.kind == ValueKind::kNaN) {
          value = std::numeric_limits<float>::quiet_NaN();
        } else if (parsed.kind == ValueKind::kPosInf) {
          value = std::numeric_limits<float>::infinity();
        } else if (parsed.kind == ValueKind::kNegInf) {
          value = -std::numeric_limits<float>::infinity();
        } else {
          value = static_cast<float>(parsed.value);
        }
        AppendBytes(&bytes, value);
        break;
      }
      case DType::kFloat16: {
        float value = 0.0f;
        if (parsed.kind == ValueKind::kNaN) {
          value = std::numeric_limits<float>::quiet_NaN();
        } else if (parsed.kind == ValueKind::kPosInf) {
          value = std::numeric_limits<float>::infinity();
        } else if (parsed.kind == ValueKind::kNegInf) {
          value = -std::numeric_limits<float>::infinity();
        } else {
          value = static_cast<float>(parsed.value);
        }
        AppendBytes(&bytes, aclFloatToFloat16(value));
        break;
      }
      case DType::kBFloat16: {
        float value = 0.0f;
        if (parsed.kind == ValueKind::kNaN) {
          value = std::numeric_limits<float>::quiet_NaN();
        } else if (parsed.kind == ValueKind::kPosInf) {
          value = std::numeric_limits<float>::infinity();
        } else if (parsed.kind == ValueKind::kNegInf) {
          value = -std::numeric_limits<float>::infinity();
        } else {
          value = static_cast<float>(parsed.value);
        }
        AppendBytes(&bytes, FloatToBFloat16(value));
        break;
      }
      case DType::kDouble: {
        double value = 0.0;
        if (parsed.kind == ValueKind::kNaN) {
          value = std::numeric_limits<double>::quiet_NaN();
        } else if (parsed.kind == ValueKind::kPosInf) {
          value = std::numeric_limits<double>::infinity();
        } else if (parsed.kind == ValueKind::kNegInf) {
          value = -std::numeric_limits<double>::infinity();
        } else {
          value = static_cast<double>(parsed.value);
        }
        AppendBytes(&bytes, value);
        break;
      }
      case DType::kUInt8:
        AppendBytes(&bytes, static_cast<uint8_t>(std::llround(parsed.value)));
        break;
      case DType::kInt8:
        AppendBytes(&bytes, static_cast<int8_t>(std::llround(parsed.value)));
        break;
      case DType::kInt16:
        AppendBytes(&bytes, static_cast<int16_t>(std::llround(parsed.value)));
        break;
      case DType::kInt32:
        AppendBytes(&bytes, static_cast<int32_t>(std::llround(parsed.value)));
        break;
      case DType::kInt64:
        AppendBytes(&bytes, static_cast<int64_t>(std::llround(parsed.value)));
        break;
      case DType::kBool:
        AppendBytes(&bytes, static_cast<uint8_t>(parsed.value != 0.0 ? 1 : 0));
        break;
    }
  }
  return bytes;
}

std::vector<std::string> DecodeValues(const std::vector<uint8_t> &bytes, DType dtype)
{
  std::vector<std::string> values;
  const size_t elemSize = DTypeSize(dtype);
  values.reserve(bytes.size() / elemSize);
  for (size_t offset = 0; offset + elemSize <= bytes.size(); offset += elemSize) {
    switch (dtype) {
      case DType::kFloat32: {
        float value = 0.0f;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        values.push_back(FormatValue(std::isnan(value)   ? Value {ValueKind::kNaN, 0.0}
                                     : std::isinf(value) ? Value {value > 0 ? ValueKind::kPosInf : ValueKind::kNegInf, 0.0}
                                                         : Value {ValueKind::kFinite, value}));
        break;
      }
      case DType::kFloat16: {
        aclFloat16 value = 0;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        const float decoded = aclFloat16ToFloat(value);
        values.push_back(FormatValue(std::isnan(decoded)   ? Value {ValueKind::kNaN, 0.0}
                                     : std::isinf(decoded) ? Value {decoded > 0 ? ValueKind::kPosInf : ValueKind::kNegInf, 0.0}
                                                           : Value {ValueKind::kFinite, decoded}));
        break;
      }
      case DType::kBFloat16: {
        uint16_t value = 0;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        const float decoded = BFloat16ToFloat(value);
        values.push_back(FormatValue(std::isnan(decoded)   ? Value {ValueKind::kNaN, 0.0}
                                     : std::isinf(decoded) ? Value {decoded > 0 ? ValueKind::kPosInf : ValueKind::kNegInf, 0.0}
                                                           : Value {ValueKind::kFinite, decoded}));
        break;
      }
      case DType::kDouble: {
        double value = 0.0;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        values.push_back(FormatValue(std::isnan(value)   ? Value {ValueKind::kNaN, 0.0}
                                     : std::isinf(value) ? Value {value > 0 ? ValueKind::kPosInf : ValueKind::kNegInf, 0.0}
                                                         : Value {ValueKind::kFinite, value}));
        break;
      }
      case DType::kUInt8: {
        uint8_t value = 0;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        values.push_back(std::to_string(static_cast<unsigned int>(value)));
        break;
      }
      case DType::kInt8: {
        int8_t value = 0;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        values.push_back(std::to_string(static_cast<int>(value)));
        break;
      }
      case DType::kInt16: {
        int16_t value = 0;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        values.push_back(std::to_string(value));
        break;
      }
      case DType::kInt32: {
        int32_t value = 0;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        values.push_back(std::to_string(value));
        break;
      }
      case DType::kInt64: {
        int64_t value = 0;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        values.push_back(std::to_string(value));
        break;
      }
      case DType::kBool: {
        uint8_t value = 0;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        values.push_back(value == 0 ? "false" : "true");
        break;
      }
    }
  }
  return values;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

int CreateAclTensor(const std::vector<uint8_t> &hostData, const std::vector<int64_t> &shape, DType dtype,
                    DeviceTensor *deviceTensor)
{
  const size_t bytes = static_cast<size_t>(GetShapeSize(shape)) * DTypeSize(dtype);
  if (bytes > 0) {
    auto ret = aclrtMalloc(&deviceTensor->deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(deviceTensor->deviceAddr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  }
  const std::vector<int64_t> strides = ComputeStrides(shape);
  deviceTensor->tensor = aclCreateTensor(shape.data(), shape.size(), ToAclDataType(dtype), strides.data(), 0,
                                         ACL_FORMAT_ND, shape.data(), shape.size(), deviceTensor->deviceAddr);
  CHECK_RET(deviceTensor->tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
  return ACL_SUCCESS;
}

int CreateAclScalar(const ScalarSpec &scalarSpec, HostScalar *scalar)
{
  scalar->storage = EncodeValues({scalarSpec.value}, scalarSpec.dtype);
  scalar->scalar = aclCreateScalar(scalar->storage.data(), ToAclDataType(scalarSpec.dtype));
  CHECK_RET(scalar->scalar != nullptr, LOG_PRINT("aclCreateScalar failed.\n"); return ACL_ERROR_FAILURE);
  return ACL_SUCCESS;
}

void DestroyTensor(DeviceTensor *deviceTensor)
{
  if (deviceTensor->tensor != nullptr) {
    aclDestroyTensor(deviceTensor->tensor);
    deviceTensor->tensor = nullptr;
  }
  if (deviceTensor->deviceAddr != nullptr) {
    aclrtFree(deviceTensor->deviceAddr);
    deviceTensor->deviceAddr = nullptr;
  }
}

void DestroyScalar(HostScalar *scalar)
{
  if (scalar->scalar != nullptr) {
    aclDestroyScalar(scalar->scalar);
    scalar->scalar = nullptr;
  }
  scalar->storage.clear();
}

bool CurrentPlatformSupportsBFloat16()
{
  const char *socName = aclrtGetSocName();
  if (socName == nullptr) {
    return false;
  }
  const std::string lower = ToLower(socName);
  return lower.find("950") != std::string::npos || lower.find("910b") != std::string::npos ||
         lower.find("910_93") != std::string::npos || lower.find("a3") != std::string::npos;
}

bool NeedSkipBFloat16(DType dtype0, DType dtype1, DType outDtype)
{
  if (dtype0 != DType::kBFloat16 && dtype1 != DType::kBFloat16 && outDtype != DType::kBFloat16) {
    return false;
  }
  return !CurrentPlatformSupportsBFloat16();
}

bool NeedSkipBFloat16(DType dtype0, DType outDtype)
{
  if (dtype0 != DType::kBFloat16 && outDtype != DType::kBFloat16) {
    return false;
  }
  return !CurrentPlatformSupportsBFloat16();
}

Value PowReference(const Value &base, const Value &exp)
{
  if (base.kind == ValueKind::kNaN || exp.kind == ValueKind::kNaN) {
    return {ValueKind::kNaN, 0.0};
  }
  if (base.kind == ValueKind::kPosInf || base.kind == ValueKind::kNegInf || exp.kind == ValueKind::kPosInf ||
      exp.kind == ValueKind::kNegInf) {
    const long double result = std::pow(base.kind == ValueKind::kPosInf   ? std::numeric_limits<long double>::infinity()
                                       : base.kind == ValueKind::kNegInf ? -std::numeric_limits<long double>::infinity()
                                                                         : base.value,
                                        exp.kind == ValueKind::kPosInf   ? std::numeric_limits<long double>::infinity()
                                        : exp.kind == ValueKind::kNegInf ? -std::numeric_limits<long double>::infinity()
                                                                          : exp.value);
    if (std::isnan(static_cast<double>(result))) {
      return {ValueKind::kNaN, 0.0};
    }
    if (std::isinf(static_cast<double>(result))) {
      return {result > 0 ? ValueKind::kPosInf : ValueKind::kNegInf, 0.0};
    }
    return {ValueKind::kFinite, result};
  }
  const long double result = std::pow(base.value, exp.value);
  if (std::isnan(static_cast<double>(result))) {
    return {ValueKind::kNaN, 0.0};
  }
  if (std::isinf(static_cast<double>(result))) {
    return {result > 0 ? ValueKind::kPosInf : ValueKind::kNegInf, 0.0};
  }
  return {ValueKind::kFinite, result};
}

std::vector<std::string> CastExpected(const std::vector<Value> &values, DType outDtype)
{
  std::vector<std::string> tokens;
  tokens.reserve(values.size());
  for (const Value &value : values) {
    tokens.push_back(FormatValue(value));
  }
  return DecodeValues(EncodeValues(tokens, outDtype), outDtype);
}

std::vector<std::string> ComputeExpectedTensorScalar(const TensorSpec &self, const ScalarSpec &exponent, DType outDtype)
{
  std::vector<Value> values;
  values.reserve(self.data.size());
  const Value expValue = ParseValue(exponent.value);
  for (const std::string &token : self.data) {
    values.push_back(PowReference(ParseValue(token), expValue));
  }
  return CastExpected(values, outDtype);
}

std::vector<std::string> ComputeExpectedScalarTensor(const ScalarSpec &self, const TensorSpec &exponent, DType outDtype)
{
  std::vector<Value> values;
  values.reserve(exponent.data.size());
  const Value baseValue = ParseValue(self.value);
  for (const std::string &token : exponent.data) {
    values.push_back(PowReference(baseValue, ParseValue(token)));
  }
  return CastExpected(values, outDtype);
}

std::vector<std::string> ComputeExpectedTensorTensor(const TensorSpec &self, const TensorSpec &exponent,
                                                     const std::vector<int64_t> &outShape, DType outDtype)
{
  const int64_t total = GetShapeSize(outShape);
  std::vector<Value> values;
  values.reserve(static_cast<size_t>(total));
  for (int64_t i = 0; i < total; ++i) {
    const std::vector<int64_t> outIndex = UnravelIndex(i, outShape);
    const int64_t selfOffset = BroadcastOffset(outIndex, self.shape);
    const int64_t exponentOffset = BroadcastOffset(outIndex, exponent.shape);
    values.push_back(PowReference(ParseValue(self.data[static_cast<size_t>(selfOffset)]),
                                  ParseValue(exponent.data[static_cast<size_t>(exponentOffset)])));
  }
  return CastExpected(values, outDtype);
}

std::vector<std::string> ComputeExpectedExp2(const TensorSpec &self, DType outDtype)
{
  std::vector<Value> values;
  values.reserve(self.data.size());
  const Value two {ValueKind::kFinite, 2.0};
  for (const std::string &token : self.data) {
    values.push_back(PowReference(two, ParseValue(token)));
  }
  return CastExpected(values, outDtype);
}

bool CompareVectors(const std::vector<std::string> &expected, const std::vector<std::string> &actual, DType dtype,
                    const Tolerance &tol, std::string *message)
{
  if (expected.size() != actual.size()) {
    *message = "size mismatch, expected=" + std::to_string(expected.size()) + ", actual=" +
               std::to_string(actual.size());
    return false;
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    const Value expectedValue = ParseValue(expected[i]);
    const Value actualValue = ParseValue(actual[i]);
    if (expectedValue.kind == ValueKind::kNaN) {
      if (actualValue.kind != ValueKind::kNaN) {
        *message = "expected NaN at index " + std::to_string(i);
        return false;
      }
      continue;
    }
    if (expectedValue.kind == ValueKind::kPosInf || expectedValue.kind == ValueKind::kNegInf) {
      if (expectedValue.kind != actualValue.kind) {
        *message = "expected Inf with same sign at index " + std::to_string(i);
        return false;
      }
      continue;
    }
    if (IsFloatingDType(dtype)) {
      const long double diff = std::fabs(expectedValue.value - actualValue.value);
      const long double allowed = static_cast<long double>(tol.atol) +
                                  static_cast<long double>(tol.rtol) * std::fabs(expectedValue.value);
      if (diff > allowed) {
        *message = "float mismatch at index " + std::to_string(i) + ", expected=" + expected[i] + ", actual=" +
                   actual[i];
        return false;
      }
    } else if (expected[i] != actual[i]) {
      *message = "integer mismatch at index " + std::to_string(i) + ", expected=" + expected[i] + ", actual=" +
                 actual[i];
      return false;
    }
  }
  *message = "ok";
  return true;
}

void RecordResult(Summary *summary, const char *name, bool pass, bool skipped, const std::string &message)
{
  ++summary->total;
  if (skipped) {
    ++summary->skipped;
    LOG_PRINT("[SKIP] %s: %s\n", name, message.c_str());
    return;
  }
  if (!pass) {
    ++summary->failed;
  }
  LOG_PRINT("[%s] %s: %s\n", pass ? "PASS" : "FAIL", name, message.c_str());
}

void RecordExecResult(Summary *summary, const char *name, bool pass, const std::string &message,
                      const std::vector<std::string> &expected, const std::vector<std::string> &actual)
{
  ++summary->total;
  if (!pass) {
    ++summary->failed;
  }
  LOG_PRINT("[%s] %s: %s\n", pass ? "PASS" : "FAIL", name, message.c_str());
  LOG_PRINT("        expect=%s\n", FormatVectorPreview(expected).c_str());
  LOG_PRINT("        actual=%s\n", FormatVectorPreview(actual).c_str());
}

std::string UnexpectedStatusMessage(int expectStatus, int actualStatus)
{
  return "unexpected GetWorkspaceSize status, expect=" + std::to_string(expectStatus) +
         ", actual=" + std::to_string(actualStatus);
}

int CopyTensorToHost(const DeviceTensor &deviceTensor, const std::vector<int64_t> &shape, DType dtype,
                     std::vector<std::string> *decoded)
{
  const size_t bytes = static_cast<size_t>(GetShapeSize(shape)) * DTypeSize(dtype);
  std::vector<uint8_t> host(bytes, 0);
  if (bytes > 0) {
    auto ret = aclrtMemcpy(host.data(), bytes, deviceTensor.deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }
  *decoded = DecodeValues(host, dtype);
  return ACL_SUCCESS;
}

bool RunPowTensorScalarCase(const TensorScalarCase &testCase, aclrtStream stream, Summary *summary)
{
  if (NeedSkipBFloat16(testCase.self.dtype, testCase.exponent.dtype, testCase.outDtype)) {
    RecordResult(summary, testCase.name, true, true, "bfloat16 requires supported SoC");
    return true;
  }
  DeviceTensor selfTensor;
  DeviceTensor outTensor;
  HostScalar exponentScalar;
  void *workspace = nullptr;
  aclOpExecutor *executor = nullptr;
  uint64_t workspaceSize = 0;

  int ret = CreateAclTensor(EncodeValues(testCase.self.data, testCase.self.dtype), testCase.self.shape,
                            testCase.self.dtype, &selfTensor);
  if (ret != ACL_SUCCESS) {
    RecordResult(summary, testCase.name, false, false, "failed to create self tensor");
    return false;
  }
  ret = CreateAclScalar(testCase.exponent, &exponentScalar);
  if (ret != ACL_SUCCESS) {
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, false, false, "failed to create exponent scalar");
    return false;
  }
  ret = CreateAclTensor(std::vector<uint8_t>(static_cast<size_t>(GetShapeSize(testCase.self.shape)) *
                                                 DTypeSize(testCase.outDtype),
                                             0),
                        testCase.self.shape, testCase.outDtype, &outTensor);
  if (ret != ACL_SUCCESS) {
    DestroyScalar(&exponentScalar);
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, false, false, "failed to create out tensor");
    return false;
  }

  const aclnnStatus status = aclnnPowTensorScalarGetWorkspaceSize(selfTensor.tensor, exponentScalar.scalar,
                                                                  outTensor.tensor, &workspaceSize, &executor);
  if (status != testCase.expectStatus) {
    DestroyTensor(&outTensor);
    DestroyScalar(&exponentScalar);
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, false, false, UnexpectedStatusMessage(testCase.expectStatus, status));
    return false;
  }
  if (status != ACLNN_SUCCESS) {
    DestroyTensor(&outTensor);
    DestroyScalar(&exponentScalar);
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, true, false, "expected api failure observed");
    return true;
  }

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      DestroyTensor(&outTensor);
      DestroyScalar(&exponentScalar);
      DestroyTensor(&selfTensor);
      RecordResult(summary, testCase.name, false, false, "allocate workspace failed");
      return false;
    }
  }

  ret = aclnnPowTensorScalar(workspace, workspaceSize, executor, stream);
  if (ret == ACLNN_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }
  if (ret != ACLNN_SUCCESS) {
    if (workspace != nullptr) {
      aclrtFree(workspace);
    }
    DestroyTensor(&outTensor);
    DestroyScalar(&exponentScalar);
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, false, false, "execution failed");
    return false;
  }

  std::vector<std::string> actual;
  ret = CopyTensorToHost(outTensor, testCase.self.shape, testCase.outDtype, &actual);
  const std::vector<std::string> expected =
      ComputeExpectedTensorScalar(testCase.self, testCase.exponent, testCase.outDtype);
  std::string message;
  const Tolerance tol = (testCase.tol.atol == 0.0 && testCase.tol.rtol == 0.0) ? DefaultTolerance(testCase.outDtype)
                                                                                : testCase.tol;
  const bool pass = ret == ACL_SUCCESS && CompareVectors(expected, actual, testCase.outDtype, tol, &message);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  DestroyTensor(&outTensor);
  DestroyScalar(&exponentScalar);
  DestroyTensor(&selfTensor);
  RecordExecResult(summary, testCase.name, pass, pass ? "ok" : message, expected, actual);
  return pass;
}

bool RunInplacePowTensorScalarCase(const TensorScalarCase &testCase, aclrtStream stream, Summary *summary)
{
  if (NeedSkipBFloat16(testCase.self.dtype, testCase.exponent.dtype, testCase.outDtype)) {
    RecordResult(summary, testCase.name, true, true, "bfloat16 requires supported SoC");
    return true;
  }
  DeviceTensor selfTensor;
  HostScalar exponentScalar;
  void *workspace = nullptr;
  aclOpExecutor *executor = nullptr;
  uint64_t workspaceSize = 0;

  int ret = CreateAclTensor(EncodeValues(testCase.self.data, testCase.self.dtype), testCase.self.shape,
                            testCase.self.dtype, &selfTensor);
  if (ret != ACL_SUCCESS) {
    RecordResult(summary, testCase.name, false, false, "failed to create self tensor");
    return false;
  }
  ret = CreateAclScalar(testCase.exponent, &exponentScalar);
  if (ret != ACL_SUCCESS) {
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, false, false, "failed to create exponent scalar");
    return false;
  }

  const aclnnStatus status =
      aclnnInplacePowTensorScalarGetWorkspaceSize(selfTensor.tensor, exponentScalar.scalar, &workspaceSize, &executor);
  if (status != testCase.expectStatus) {
    DestroyScalar(&exponentScalar);
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, false, false, UnexpectedStatusMessage(testCase.expectStatus, status));
    return false;
  }
  if (status != ACLNN_SUCCESS) {
    DestroyScalar(&exponentScalar);
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, true, false, "expected api failure observed");
    return true;
  }

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      DestroyScalar(&exponentScalar);
      DestroyTensor(&selfTensor);
      RecordResult(summary, testCase.name, false, false, "allocate workspace failed");
      return false;
    }
  }

  ret = aclnnInplacePowTensorScalar(workspace, workspaceSize, executor, stream);
  if (ret == ACLNN_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }
  if (ret != ACLNN_SUCCESS) {
    if (workspace != nullptr) {
      aclrtFree(workspace);
    }
    DestroyScalar(&exponentScalar);
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, false, false, "execution failed");
    return false;
  }

  std::vector<std::string> actual;
  ret = CopyTensorToHost(selfTensor, testCase.self.shape, testCase.outDtype, &actual);
  const std::vector<std::string> expected =
      ComputeExpectedTensorScalar(testCase.self, testCase.exponent, testCase.outDtype);
  std::string message;
  const Tolerance tol = (testCase.tol.atol == 0.0 && testCase.tol.rtol == 0.0) ? DefaultTolerance(testCase.outDtype)
                                                                                : testCase.tol;
  const bool pass = ret == ACL_SUCCESS && CompareVectors(expected, actual, testCase.outDtype, tol, &message);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  DestroyScalar(&exponentScalar);
  DestroyTensor(&selfTensor);
  RecordExecResult(summary, testCase.name, pass, pass ? "ok" : message, expected, actual);
  return pass;
}

bool RunPowScalarTensorCase(const ScalarTensorCase &testCase, aclrtStream stream, Summary *summary)
{
  if (NeedSkipBFloat16(testCase.self.dtype, testCase.exponent.dtype, testCase.outDtype)) {
    RecordResult(summary, testCase.name, true, true, "bfloat16 requires supported SoC");
    return true;
  }
  DeviceTensor exponentTensor;
  DeviceTensor outTensor;
  HostScalar selfScalar;
  void *workspace = nullptr;
  aclOpExecutor *executor = nullptr;
  uint64_t workspaceSize = 0;

  int ret = CreateAclScalar(testCase.self, &selfScalar);
  if (ret != ACL_SUCCESS) {
    RecordResult(summary, testCase.name, false, false, "failed to create self scalar");
    return false;
  }
  ret = CreateAclTensor(EncodeValues(testCase.exponent.data, testCase.exponent.dtype), testCase.exponent.shape,
                        testCase.exponent.dtype, &exponentTensor);
  if (ret != ACL_SUCCESS) {
    DestroyScalar(&selfScalar);
    RecordResult(summary, testCase.name, false, false, "failed to create exponent tensor");
    return false;
  }
  ret = CreateAclTensor(std::vector<uint8_t>(static_cast<size_t>(GetShapeSize(testCase.exponent.shape)) *
                                                 DTypeSize(testCase.outDtype),
                                             0),
                        testCase.exponent.shape, testCase.outDtype, &outTensor);
  if (ret != ACL_SUCCESS) {
    DestroyTensor(&exponentTensor);
    DestroyScalar(&selfScalar);
    RecordResult(summary, testCase.name, false, false, "failed to create out tensor");
    return false;
  }

  const aclnnStatus status = aclnnPowScalarTensorGetWorkspaceSize(selfScalar.scalar, exponentTensor.tensor,
                                                                  outTensor.tensor, &workspaceSize, &executor);
  if (status != testCase.expectStatus) {
    DestroyTensor(&outTensor);
    DestroyTensor(&exponentTensor);
    DestroyScalar(&selfScalar);
    RecordResult(summary, testCase.name, false, false, UnexpectedStatusMessage(testCase.expectStatus, status));
    return false;
  }
  if (status != ACLNN_SUCCESS) {
    DestroyTensor(&outTensor);
    DestroyTensor(&exponentTensor);
    DestroyScalar(&selfScalar);
    RecordResult(summary, testCase.name, true, false, "expected api failure observed");
    return true;
  }

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      DestroyTensor(&outTensor);
      DestroyTensor(&exponentTensor);
      DestroyScalar(&selfScalar);
      RecordResult(summary, testCase.name, false, false, "allocate workspace failed");
      return false;
    }
  }

  ret = aclnnPowScalarTensor(workspace, workspaceSize, executor, stream);
  if (ret == ACLNN_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }
  if (ret != ACLNN_SUCCESS) {
    if (workspace != nullptr) {
      aclrtFree(workspace);
    }
    DestroyTensor(&outTensor);
    DestroyTensor(&exponentTensor);
    DestroyScalar(&selfScalar);
    RecordResult(summary, testCase.name, false, false, "execution failed");
    return false;
  }

  std::vector<std::string> actual;
  ret = CopyTensorToHost(outTensor, testCase.exponent.shape, testCase.outDtype, &actual);
  const std::vector<std::string> expected =
      ComputeExpectedScalarTensor(testCase.self, testCase.exponent, testCase.outDtype);
  std::string message;
  const Tolerance tol = (testCase.tol.atol == 0.0 && testCase.tol.rtol == 0.0) ? DefaultTolerance(testCase.outDtype)
                                                                                : testCase.tol;
  const bool pass = ret == ACL_SUCCESS && CompareVectors(expected, actual, testCase.outDtype, tol, &message);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  DestroyTensor(&outTensor);
  DestroyTensor(&exponentTensor);
  DestroyScalar(&selfScalar);
  RecordExecResult(summary, testCase.name, pass, pass ? "ok" : message, expected, actual);
  return pass;
}

bool RunPowTensorTensorCase(const TensorTensorCase &testCase, aclrtStream stream, Summary *summary)
{
  if (NeedSkipBFloat16(testCase.self.dtype, testCase.exponent.dtype, testCase.outDtype)) {
    RecordResult(summary, testCase.name, true, true, "bfloat16 requires supported SoC");
    return true;
  }
  DeviceTensor selfTensor;
  DeviceTensor exponentTensor;
  DeviceTensor outTensor;
  void *workspace = nullptr;
  aclOpExecutor *executor = nullptr;
  uint64_t workspaceSize = 0;

  int ret = CreateAclTensor(EncodeValues(testCase.self.data, testCase.self.dtype), testCase.self.shape,
                            testCase.self.dtype, &selfTensor);
  if (ret != ACL_SUCCESS) {
    RecordResult(summary, testCase.name, false, false, "failed to create self tensor");
    return false;
  }
  ret = CreateAclTensor(EncodeValues(testCase.exponent.data, testCase.exponent.dtype), testCase.exponent.shape,
                        testCase.exponent.dtype, &exponentTensor);
  if (ret != ACL_SUCCESS) {
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, false, false, "failed to create exponent tensor");
    return false;
  }
  ret = CreateAclTensor(std::vector<uint8_t>(static_cast<size_t>(GetShapeSize(testCase.outShape)) *
                                                 DTypeSize(testCase.outDtype),
                                             0),
                        testCase.outShape, testCase.outDtype, &outTensor);
  if (ret != ACL_SUCCESS) {
    DestroyTensor(&exponentTensor);
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, false, false, "failed to create out tensor");
    return false;
  }

  const aclnnStatus status = aclnnPowTensorTensorGetWorkspaceSize(selfTensor.tensor, exponentTensor.tensor,
                                                                  outTensor.tensor, &workspaceSize, &executor);
  if (status != testCase.expectStatus) {
    DestroyTensor(&outTensor);
    DestroyTensor(&exponentTensor);
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, false, false, UnexpectedStatusMessage(testCase.expectStatus, status));
    return false;
  }
  if (status != ACLNN_SUCCESS) {
    DestroyTensor(&outTensor);
    DestroyTensor(&exponentTensor);
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, true, false, "expected api failure observed");
    return true;
  }

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      DestroyTensor(&outTensor);
      DestroyTensor(&exponentTensor);
      DestroyTensor(&selfTensor);
      RecordResult(summary, testCase.name, false, false, "allocate workspace failed");
      return false;
    }
  }

  ret = aclnnPowTensorTensor(workspace, workspaceSize, executor, stream);
  if (ret == ACLNN_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }
  if (ret != ACLNN_SUCCESS) {
    if (workspace != nullptr) {
      aclrtFree(workspace);
    }
    DestroyTensor(&outTensor);
    DestroyTensor(&exponentTensor);
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, false, false, "execution failed");
    return false;
  }

  std::vector<std::string> actual;
  ret = CopyTensorToHost(outTensor, testCase.outShape, testCase.outDtype, &actual);
  const std::vector<std::string> expected =
      ComputeExpectedTensorTensor(testCase.self, testCase.exponent, testCase.outShape, testCase.outDtype);
  std::string message;
  const Tolerance tol = (testCase.tol.atol == 0.0 && testCase.tol.rtol == 0.0) ? DefaultTolerance(testCase.outDtype)
                                                                                : testCase.tol;
  const bool pass = ret == ACL_SUCCESS && CompareVectors(expected, actual, testCase.outDtype, tol, &message);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  DestroyTensor(&outTensor);
  DestroyTensor(&exponentTensor);
  DestroyTensor(&selfTensor);
  RecordExecResult(summary, testCase.name, pass, pass ? "ok" : message, expected, actual);
  return pass;
}

bool RunInplacePowTensorTensorCase(const TensorTensorCase &testCase, aclrtStream stream, Summary *summary)
{
  if (NeedSkipBFloat16(testCase.self.dtype, testCase.exponent.dtype, testCase.outDtype)) {
    RecordResult(summary, testCase.name, true, true, "bfloat16 requires supported SoC");
    return true;
  }
  DeviceTensor selfTensor;
  DeviceTensor exponentTensor;
  void *workspace = nullptr;
  aclOpExecutor *executor = nullptr;
  uint64_t workspaceSize = 0;

  int ret = CreateAclTensor(EncodeValues(testCase.self.data, testCase.self.dtype), testCase.self.shape,
                            testCase.self.dtype, &selfTensor);
  if (ret != ACL_SUCCESS) {
    RecordResult(summary, testCase.name, false, false, "failed to create self tensor");
    return false;
  }
  ret = CreateAclTensor(EncodeValues(testCase.exponent.data, testCase.exponent.dtype), testCase.exponent.shape,
                        testCase.exponent.dtype, &exponentTensor);
  if (ret != ACL_SUCCESS) {
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, false, false, "failed to create exponent tensor");
    return false;
  }

  const aclnnStatus status =
      aclnnInplacePowTensorTensorGetWorkspaceSize(selfTensor.tensor, exponentTensor.tensor, &workspaceSize, &executor);
  if (status != testCase.expectStatus) {
    DestroyTensor(&exponentTensor);
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, false, false, UnexpectedStatusMessage(testCase.expectStatus, status));
    return false;
  }
  if (status != ACLNN_SUCCESS) {
    DestroyTensor(&exponentTensor);
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, true, false, "expected api failure observed");
    return true;
  }

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      DestroyTensor(&exponentTensor);
      DestroyTensor(&selfTensor);
      RecordResult(summary, testCase.name, false, false, "allocate workspace failed");
      return false;
    }
  }

  ret = aclnnInplacePowTensorTensor(workspace, workspaceSize, executor, stream);
  if (ret == ACLNN_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }
  if (ret != ACLNN_SUCCESS) {
    if (workspace != nullptr) {
      aclrtFree(workspace);
    }
    DestroyTensor(&exponentTensor);
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, false, false, "execution failed");
    return false;
  }

  std::vector<std::string> actual;
  ret = CopyTensorToHost(selfTensor, testCase.outShape, testCase.outDtype, &actual);
  const std::vector<std::string> expected =
      ComputeExpectedTensorTensor(testCase.self, testCase.exponent, testCase.outShape, testCase.outDtype);
  std::string message;
  const Tolerance tol = (testCase.tol.atol == 0.0 && testCase.tol.rtol == 0.0) ? DefaultTolerance(testCase.outDtype)
                                                                                : testCase.tol;
  const bool pass = ret == ACL_SUCCESS && CompareVectors(expected, actual, testCase.outDtype, tol, &message);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  DestroyTensor(&exponentTensor);
  DestroyTensor(&selfTensor);
  RecordExecResult(summary, testCase.name, pass, pass ? "ok" : message, expected, actual);
  return pass;
}

bool RunExp2Case(const Exp2Case &testCase, aclrtStream stream, Summary *summary)
{
  if (NeedSkipBFloat16(testCase.self.dtype, testCase.outDtype)) {
    RecordResult(summary, testCase.name, true, true, "bfloat16 requires supported SoC");
    return true;
  }
  DeviceTensor selfTensor;
  DeviceTensor outTensor;
  void *workspace = nullptr;
  aclOpExecutor *executor = nullptr;
  uint64_t workspaceSize = 0;

  int ret = CreateAclTensor(EncodeValues(testCase.self.data, testCase.self.dtype), testCase.self.shape,
                            testCase.self.dtype, &selfTensor);
  if (ret != ACL_SUCCESS) {
    RecordResult(summary, testCase.name, false, false, "failed to create self tensor");
    return false;
  }
  ret = CreateAclTensor(std::vector<uint8_t>(static_cast<size_t>(GetShapeSize(testCase.self.shape)) *
                                                 DTypeSize(testCase.outDtype),
                                             0),
                        testCase.self.shape, testCase.outDtype, &outTensor);
  if (ret != ACL_SUCCESS) {
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, false, false, "failed to create out tensor");
    return false;
  }

  const aclnnStatus status = aclnnExp2GetWorkspaceSize(selfTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
  if (status != testCase.expectStatus) {
    DestroyTensor(&outTensor);
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, false, false, UnexpectedStatusMessage(testCase.expectStatus, status));
    return false;
  }
  if (status != ACLNN_SUCCESS) {
    DestroyTensor(&outTensor);
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, true, false, "expected api failure observed");
    return true;
  }

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      DestroyTensor(&outTensor);
      DestroyTensor(&selfTensor);
      RecordResult(summary, testCase.name, false, false, "allocate workspace failed");
      return false;
    }
  }

  ret = aclnnExp2(workspace, workspaceSize, executor, stream);
  if (ret == ACLNN_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }
  if (ret != ACLNN_SUCCESS) {
    if (workspace != nullptr) {
      aclrtFree(workspace);
    }
    DestroyTensor(&outTensor);
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, false, false, "execution failed");
    return false;
  }

  std::vector<std::string> actual;
  ret = CopyTensorToHost(outTensor, testCase.self.shape, testCase.outDtype, &actual);
  const std::vector<std::string> expected = ComputeExpectedExp2(testCase.self, testCase.outDtype);
  std::string message;
  const Tolerance tol = (testCase.tol.atol == 0.0 && testCase.tol.rtol == 0.0) ? DefaultTolerance(testCase.outDtype)
                                                                                : testCase.tol;
  const bool pass = ret == ACL_SUCCESS && CompareVectors(expected, actual, testCase.outDtype, tol, &message);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  DestroyTensor(&outTensor);
  DestroyTensor(&selfTensor);
  RecordExecResult(summary, testCase.name, pass, pass ? "ok" : message, expected, actual);
  return pass;
}

bool RunInplaceExp2Case(const Exp2Case &testCase, aclrtStream stream, Summary *summary)
{
  if (NeedSkipBFloat16(testCase.self.dtype, testCase.outDtype)) {
    RecordResult(summary, testCase.name, true, true, "bfloat16 requires supported SoC");
    return true;
  }
  DeviceTensor selfTensor;
  void *workspace = nullptr;
  aclOpExecutor *executor = nullptr;
  uint64_t workspaceSize = 0;

  int ret = CreateAclTensor(EncodeValues(testCase.self.data, testCase.self.dtype), testCase.self.shape,
                            testCase.self.dtype, &selfTensor);
  if (ret != ACL_SUCCESS) {
    RecordResult(summary, testCase.name, false, false, "failed to create self tensor");
    return false;
  }

  const aclnnStatus status = aclnnInplaceExp2GetWorkspaceSize(selfTensor.tensor, &workspaceSize, &executor);
  if (status != testCase.expectStatus) {
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, false, false, "unexpected GetWorkspaceSize status");
    return false;
  }
  if (status != ACLNN_SUCCESS) {
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, true, false, "expected api failure observed");
    return true;
  }

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      DestroyTensor(&selfTensor);
      RecordResult(summary, testCase.name, false, false, "allocate workspace failed");
      return false;
    }
  }

  ret = aclnnInplaceExp2(workspace, workspaceSize, executor, stream);
  if (ret == ACLNN_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }
  if (ret != ACLNN_SUCCESS) {
    if (workspace != nullptr) {
      aclrtFree(workspace);
    }
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, false, false, "execution failed");
    return false;
  }

  std::vector<std::string> actual;
  ret = CopyTensorToHost(selfTensor, testCase.self.shape, testCase.outDtype, &actual);
  const std::vector<std::string> expected = ComputeExpectedExp2(testCase.self, testCase.outDtype);
  std::string message;
  const Tolerance tol = (testCase.tol.atol == 0.0 && testCase.tol.rtol == 0.0) ? DefaultTolerance(testCase.outDtype)
                                                                                : testCase.tol;
  const bool pass = ret == ACL_SUCCESS && CompareVectors(expected, actual, testCase.outDtype, tol, &message);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  DestroyTensor(&selfTensor);
  RecordExecResult(summary, testCase.name, pass, pass ? "ok" : message, expected, actual);
  return pass;
}

bool RunPowTensorScalarNullptrApiCase(aclrtStream stream, Summary *summary)
{
  (void)stream;
  TensorSpec self {{2, 2}, DType::kFloat32, {"1", "2", "3", "4"}};
  ScalarSpec exponent {DType::kFloat32, "2"};
  DeviceTensor outTensor;
  HostScalar exponentScalar;
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  int ret = CreateAclScalar(exponent, &exponentScalar);
  if (ret != ACL_SUCCESS) {
    RecordResult(summary, "api_pow_tensor_scalar_null_self", false, false, "failed to create exponent scalar");
    return false;
  }
  ret = CreateAclTensor(std::vector<uint8_t>(4 * sizeof(float), 0), self.shape, DType::kFloat32, &outTensor);
  if (ret != ACL_SUCCESS) {
    DestroyScalar(&exponentScalar);
    RecordResult(summary, "api_pow_tensor_scalar_null_self", false, false, "failed to create out tensor");
    return false;
  }
  const aclnnStatus status =
      aclnnPowTensorScalarGetWorkspaceSize(nullptr, exponentScalar.scalar, outTensor.tensor, &workspaceSize, &executor);
  DestroyTensor(&outTensor);
  DestroyScalar(&exponentScalar);
  const bool pass = status != ACLNN_SUCCESS;
  RecordResult(summary, "api_pow_tensor_scalar_null_self", pass, false, pass ? "expected api failure observed"
                                                                              : "unexpected success");
  return pass;
}

bool RunPowTensorTensorBroadcastInvalidApiCase(aclrtStream stream, Summary *summary)
{
  (void)stream;
  TensorTensorCase testCase {"api_pow_tensor_tensor_broadcast_invalid",
                             {{2, 3}, DType::kFloat32, {"1", "2", "3", "4", "5", "6"}},
                             {{4}, DType::kFloat32, {"1", "2", "3", "4"}},
                             {2, 3},
                             DType::kFloat32,
                             161002,
                             {}};
  DeviceTensor selfTensor;
  DeviceTensor exponentTensor;
  DeviceTensor outTensor;
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  int ret = CreateAclTensor(EncodeValues(testCase.self.data, testCase.self.dtype), testCase.self.shape,
                            testCase.self.dtype, &selfTensor);
  if (ret != ACL_SUCCESS) {
    RecordResult(summary, testCase.name, false, false, "failed to create self tensor");
    return false;
  }
  ret = CreateAclTensor(EncodeValues(testCase.exponent.data, testCase.exponent.dtype), testCase.exponent.shape,
                        testCase.exponent.dtype, &exponentTensor);
  if (ret != ACL_SUCCESS) {
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, false, false, "failed to create exponent tensor");
    return false;
  }
  ret = CreateAclTensor(std::vector<uint8_t>(6 * sizeof(float), 0), testCase.outShape, testCase.outDtype, &outTensor);
  if (ret != ACL_SUCCESS) {
    DestroyTensor(&exponentTensor);
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, false, false, "failed to create out tensor");
    return false;
  }
  const aclnnStatus status = aclnnPowTensorTensorGetWorkspaceSize(selfTensor.tensor, exponentTensor.tensor,
                                                                  outTensor.tensor, &workspaceSize, &executor);
  DestroyTensor(&outTensor);
  DestroyTensor(&exponentTensor);
  DestroyTensor(&selfTensor);
  const bool pass = status == testCase.expectStatus;
  RecordResult(summary, testCase.name, pass, false, pass ? "expected api failure observed" : "unexpected status");
  return pass;
}

bool RunExp2InvalidDtypeApiCase(aclrtStream stream, Summary *summary)
{
  (void)stream;
  Exp2Case testCase {"api_inplace_exp2_invalid_int32", {{4}, DType::kInt32, {"0", "1", "2", "3"}}, DType::kInt32,
                     161002, {}};
  DeviceTensor selfTensor;
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  int ret = CreateAclTensor(EncodeValues(testCase.self.data, testCase.self.dtype), testCase.self.shape,
                            testCase.self.dtype, &selfTensor);
  if (ret != ACL_SUCCESS) {
    RecordResult(summary, testCase.name, false, false, "failed to create self tensor");
    return false;
  }
  const aclnnStatus status = aclnnInplaceExp2GetWorkspaceSize(selfTensor.tensor, &workspaceSize, &executor);
  DestroyTensor(&selfTensor);
  const bool pass = status == testCase.expectStatus;
  RecordResult(summary, testCase.name, pass, false, pass ? "expected api failure observed" : "unexpected status");
  return pass;
}

bool RunPowScalarTensorNullExponentApiCase(aclrtStream stream, Summary *summary)
{
  (void)stream;
  HostScalar selfScalar;
  DeviceTensor outTensor;
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  const char *name = "api_pow_scalar_tensor_null_exponent";
  const ScalarSpec self {DType::kFloat32, "2"};
  int ret = CreateAclScalar(self, &selfScalar);
  if (ret != ACL_SUCCESS) {
    RecordResult(summary, name, false, false, "failed to create self scalar");
    return false;
  }
  ret = CreateAclTensor(std::vector<uint8_t>(4 * sizeof(float), 0), {4}, DType::kFloat32, &outTensor);
  if (ret != ACL_SUCCESS) {
    DestroyScalar(&selfScalar);
    RecordResult(summary, name, false, false, "failed to create out tensor");
    return false;
  }
  const aclnnStatus status =
      aclnnPowScalarTensorGetWorkspaceSize(selfScalar.scalar, nullptr, outTensor.tensor, &workspaceSize, &executor);
  DestroyTensor(&outTensor);
  DestroyScalar(&selfScalar);
  const bool pass = status != ACLNN_SUCCESS;
  RecordResult(summary, name, pass, false, pass ? "expected api failure observed" : "unexpected success");
  return pass;
}

bool RunPowTensorScalarNegativeExponentForIntApiCase(aclrtStream stream, Summary *summary)
{
  TensorScalarCase testCase {"api_pow_tensor_scalar_int_negative_exponent",
                             {{4}, DType::kInt32, {"1", "2", "4", "8"}},
                             {DType::kInt32, "-1"},
                             DType::kInt32,
                             161002,
                             {}};
  return RunPowTensorScalarCase(testCase, stream, summary);
}

bool RunInplacePowTensorTensorBroadcastInvalidApiCase(aclrtStream stream, Summary *summary)
{
  (void)stream;
  const char *name = "api_inplace_pow_tensor_tensor_broadcast_invalid";
  DeviceTensor selfTensor;
  DeviceTensor exponentTensor;
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  int ret = CreateAclTensor(EncodeValues({"1", "2", "3", "4", "5", "6"}, DType::kFloat32), {2, 3}, DType::kFloat32,
                            &selfTensor);
  if (ret != ACL_SUCCESS) {
    RecordResult(summary, name, false, false, "failed to create self tensor");
    return false;
  }
  ret = CreateAclTensor(EncodeValues({"1", "2", "3", "4"}, DType::kFloat32), {4}, DType::kFloat32, &exponentTensor);
  if (ret != ACL_SUCCESS) {
    DestroyTensor(&selfTensor);
    RecordResult(summary, name, false, false, "failed to create exponent tensor");
    return false;
  }
  const aclnnStatus status =
      aclnnInplacePowTensorTensorGetWorkspaceSize(selfTensor.tensor, exponentTensor.tensor, &workspaceSize, &executor);
  DestroyTensor(&exponentTensor);
  DestroyTensor(&selfTensor);
  const bool pass = status != ACLNN_SUCCESS;
  RecordResult(summary, name, pass, false, pass ? "expected api failure observed" : "unexpected success");
  return pass;
}

bool RunExp2RankTooLargeApiCase(aclrtStream stream, Summary *summary)
{
  (void)stream;
  const char *name = "api_exp2_rank_too_large";
  DeviceTensor selfTensor;
  DeviceTensor outTensor;
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  const std::vector<int64_t> shape {1, 1, 1, 1, 1, 1, 1, 1, 1};
  int ret = CreateAclTensor(EncodeValues({"1"}, DType::kFloat32), shape, DType::kFloat32, &selfTensor);
  if (ret != ACL_SUCCESS) {
    RecordResult(summary, name, false, false, "failed to create self tensor");
    return false;
  }
  ret = CreateAclTensor(std::vector<uint8_t>(sizeof(float), 0), shape, DType::kFloat32, &outTensor);
  if (ret != ACL_SUCCESS) {
    DestroyTensor(&selfTensor);
    RecordResult(summary, name, false, false, "failed to create out tensor");
    return false;
  }
  const aclnnStatus status = aclnnExp2GetWorkspaceSize(selfTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
  DestroyTensor(&outTensor);
  DestroyTensor(&selfTensor);
  const bool pass = status != ACLNN_SUCCESS;
  RecordResult(summary, name, pass, false, pass ? "expected api failure observed" : "unexpected success");
  return pass;
}

bool RunPowTensorScalarRankTooLargeApiCase(aclrtStream stream, Summary *summary)
{
  (void)stream;
  TensorScalarCase testCase {"api_pow_tensor_scalar_rank_too_large",
                             {{1, 1, 1, 1, 1, 1, 1, 1, 1}, DType::kFloat32, {"1"}},
                             {DType::kFloat32, "2"},
                             DType::kFloat32,
                             0,
                             {}};
  DeviceTensor selfTensor;
  DeviceTensor outTensor;
  HostScalar exponentScalar;
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;

  int ret = CreateAclTensor(EncodeValues(testCase.self.data, testCase.self.dtype), testCase.self.shape,
                            testCase.self.dtype, &selfTensor);
  if (ret != ACL_SUCCESS) {
    RecordResult(summary, testCase.name, false, false, "failed to create self tensor");
    return false;
  }
  ret = CreateAclScalar(testCase.exponent, &exponentScalar);
  if (ret != ACL_SUCCESS) {
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, false, false, "failed to create exponent scalar");
    return false;
  }
  ret = CreateAclTensor(std::vector<uint8_t>(sizeof(float), 0), testCase.self.shape, testCase.outDtype, &outTensor);
  if (ret != ACL_SUCCESS) {
    DestroyScalar(&exponentScalar);
    DestroyTensor(&selfTensor);
    RecordResult(summary, testCase.name, false, false, "failed to create out tensor");
    return false;
  }
  const aclnnStatus status = aclnnPowTensorScalarGetWorkspaceSize(selfTensor.tensor, exponentScalar.scalar,
                                                                  outTensor.tensor, &workspaceSize, &executor);
  DestroyTensor(&outTensor);
  DestroyScalar(&exponentScalar);
  DestroyTensor(&selfTensor);
  const bool pass = status != ACLNN_SUCCESS;
  RecordResult(summary, testCase.name, pass, false, pass ? "expected api failure observed" : "unexpected success");
  return pass;
}

bool RunPowTensorTensorOutShapeMismatchApiCase(aclrtStream stream, Summary *summary)
{
  (void)stream;
  TensorTensorCase testCase {"api_pow_tensor_tensor_out_shape_mismatch",
                             {{2, 2}, DType::kFloat32, {"1", "2", "3", "4"}},
                             {{2, 2}, DType::kFloat32, {"2", "3", "2", "1"}},
                             {4},
                             DType::kFloat32,
                             161002,
                             {}};
  return RunPowTensorTensorCase(testCase, stream, summary);
}

bool RunPowTensorScalarBoolBoolApiCase(aclrtStream stream, Summary *summary)
{
  (void)stream;
  TensorScalarCase testCase {"api_pow_tensor_scalar_bool_bool_invalid",
                             {{4}, DType::kBool, {"true", "false", "true", "false"}},
                             {DType::kBool, "true"},
                             DType::kBool,
                             161002,
                             {}};
  return RunPowTensorScalarCase(testCase, stream, summary);
}

bool RunPowScalarTensorBoolBoolApiCase(aclrtStream stream, Summary *summary)
{
  (void)stream;
  ScalarTensorCase testCase {"api_pow_scalar_tensor_bool_bool_invalid",
                             {DType::kBool, "true"},
                             {{4}, DType::kBool, {"true", "false", "true", "false"}},
                             DType::kBool,
                             161002,
                             {}};
  return RunPowScalarTensorCase(testCase, stream, summary);
}

bool RunPowTensorTensorBoolBoolApiCase(aclrtStream stream, Summary *summary)
{
  (void)stream;
  TensorTensorCase testCase {"api_pow_tensor_tensor_bool_bool_invalid",
                             {{2, 2}, DType::kBool, {"true", "false", "true", "false"}},
                             {{2, 2}, DType::kBool, {"true", "true", "false", "false"}},
                             {2, 2},
                             DType::kBool,
                             161002,
                             {}};
  return RunPowTensorTensorCase(testCase, stream, summary);
}

bool RunPowTensorScalarExponentOverflowApiCase(aclrtStream stream, Summary *summary)
{
  (void)stream;
  TensorScalarCase testCase {"api_pow_tensor_scalar_exponent_overflow_int8",
                             {{4}, DType::kInt8, {"1", "2", "3", "4"}},
                             {DType::kInt32, "300"},
                             DType::kInt8,
                             161002,
                             {}};
  return RunPowTensorScalarCase(testCase, stream, summary);
}

bool RunPowScalarTensorOutShapeMismatchApiCase(aclrtStream stream, Summary *summary)
{
  (void)stream;
  ScalarTensorCase testCase {"api_pow_scalar_tensor_out_shape_mismatch",
                             {DType::kFloat32, "2"},
                             {{2, 2}, DType::kFloat32, {"1", "2", "3", "4"}},
                             DType::kFloat32,
                             161002,
                             {}};

  DeviceTensor exponentTensor;
  DeviceTensor outTensor;
  HostScalar selfScalar;
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;

  int ret = CreateAclScalar(testCase.self, &selfScalar);
  if (ret != ACL_SUCCESS) {
    RecordResult(summary, testCase.name, false, false, "failed to create self scalar");
    return false;
  }
  ret = CreateAclTensor(EncodeValues(testCase.exponent.data, testCase.exponent.dtype), testCase.exponent.shape,
                        testCase.exponent.dtype, &exponentTensor);
  if (ret != ACL_SUCCESS) {
    DestroyScalar(&selfScalar);
    RecordResult(summary, testCase.name, false, false, "failed to create exponent tensor");
    return false;
  }
  ret = CreateAclTensor(std::vector<uint8_t>(2 * sizeof(float), 0), {2}, testCase.outDtype, &outTensor);
  if (ret != ACL_SUCCESS) {
    DestroyTensor(&exponentTensor);
    DestroyScalar(&selfScalar);
    RecordResult(summary, testCase.name, false, false, "failed to create out tensor");
    return false;
  }
  const aclnnStatus status = aclnnPowScalarTensorGetWorkspaceSize(selfScalar.scalar, exponentTensor.tensor,
                                                                  outTensor.tensor, &workspaceSize, &executor);
  DestroyTensor(&outTensor);
  DestroyTensor(&exponentTensor);
  DestroyScalar(&selfScalar);
  const bool pass = status == testCase.expectStatus;
  RecordResult(summary, testCase.name, pass, false, pass ? "expected api failure observed" : "unexpected status");
  return pass;
}

bool RunExp2OutShapeMismatchApiCase(aclrtStream stream, Summary *summary)
{
  (void)stream;
  const char *name = "api_exp2_out_shape_mismatch";
  DeviceTensor selfTensor;
  DeviceTensor outTensor;
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  int ret = CreateAclTensor(EncodeValues({"0", "1", "2", "3"}, DType::kFloat32), {4}, DType::kFloat32, &selfTensor);
  if (ret != ACL_SUCCESS) {
    RecordResult(summary, name, false, false, "failed to create self tensor");
    return false;
  }
  ret = CreateAclTensor(std::vector<uint8_t>(2 * sizeof(float), 0), {2}, DType::kFloat32, &outTensor);
  if (ret != ACL_SUCCESS) {
    DestroyTensor(&selfTensor);
    RecordResult(summary, name, false, false, "failed to create out tensor");
    return false;
  }
  const aclnnStatus status = aclnnExp2GetWorkspaceSize(selfTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
  DestroyTensor(&outTensor);
  DestroyTensor(&selfTensor);
  const bool pass = status != ACLNN_SUCCESS;
  RecordResult(summary, name, pass, false, pass ? "expected api failure observed" : "unexpected success");
  return pass;
}

}  // namespace

int main()
{
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  int ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  Summary summary;

  RunPowTensorScalarNullptrApiCase(stream, &summary);
  RunPowTensorScalarRankTooLargeApiCase(stream, &summary);
  RunPowTensorScalarBoolBoolApiCase(stream, &summary);
  RunPowTensorTensorBroadcastInvalidApiCase(stream, &summary);
  RunPowTensorTensorOutShapeMismatchApiCase(stream, &summary);
  RunPowScalarTensorNullExponentApiCase(stream, &summary);
  RunPowScalarTensorBoolBoolApiCase(stream, &summary);
  RunPowScalarTensorOutShapeMismatchApiCase(stream, &summary);
  RunPowTensorScalarNegativeExponentForIntApiCase(stream, &summary);
  RunPowTensorScalarExponentOverflowApiCase(stream, &summary);
  RunPowTensorTensorBoolBoolApiCase(stream, &summary);
  RunInplacePowTensorTensorBroadcastInvalidApiCase(stream, &summary);
  RunExp2InvalidDtypeApiCase(stream, &summary);
  RunExp2RankTooLargeApiCase(stream, &summary);
  RunExp2OutShapeMismatchApiCase(stream, &summary);

  RunPowTensorScalarCase({"pow_tensor_scalar_f32_exp0", {{4}, DType::kFloat32, {"2", "3", "4", "5"}},
                          {DType::kFloat32, "0"}, DType::kFloat32, ACLNN_SUCCESS, {}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_f32_exp1", {{4}, DType::kFloat32, {"2", "3", "4", "5"}},
                          {DType::kFloat32, "1"}, DType::kFloat32, ACLNN_SUCCESS, {}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_f32_sqrt", {{4}, DType::kFloat32, {"1", "4", "9", "16"}},
                          {DType::kFloat32, "0.5"}, DType::kFloat32, ACLNN_SUCCESS, {}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_f32_rsqrt", {{4}, DType::kFloat32, {"1", "4", "16", "64"}},
                          {DType::kFloat32, "-0.5"}, DType::kFloat32, ACLNN_SUCCESS, {}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_f32_square", {{4}, DType::kFloat32, {"1", "2", "3", "4"}},
                          {DType::kFloat32, "2"}, DType::kFloat32, ACLNN_SUCCESS, {}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_f32_cube", {{4}, DType::kFloat32, {"1", "2", "3", "4"}},
                          {DType::kFloat32, "3"}, DType::kFloat32, ACLNN_SUCCESS, {}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_f32_reciprocal", {{4}, DType::kFloat32, {"1", "2", "4", "8"}},
                          {DType::kFloat32, "-1"}, DType::kFloat32, ACLNN_SUCCESS, {}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_f32_reciprocal_square", {{4}, DType::kFloat32, {"1", "2", "4", "8"}},
                          {DType::kFloat32, "-2"}, DType::kFloat32, ACLNN_SUCCESS, {}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_f32_generic", {{4}, DType::kFloat32, {"0", "1", "4", "9"}},
                          {DType::kFloat32, "1.5"}, DType::kFloat32, ACLNN_SUCCESS, {1e-4, 1e-4}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_f32_overflow_inf",
                          {{4}, DType::kFloat32, {"100", "1000", "1e10", "2"}},
                          {DType::kFloat32, "20"},
                          DType::kFloat32,
                          ACLNN_SUCCESS,
                          {}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_f32_underflow_zero",
                          {{4}, DType::kFloat32, {"0.5", "0.25", "0.125", "0.0625"}},
                          {DType::kFloat32, "20"},
                          DType::kFloat32,
                          ACLNN_SUCCESS,
                          {}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_f32_negative_fractional_nan",
                          {{4}, DType::kFloat32, {"-1", "-4", "9", "16"}},
                          {DType::kFloat32, "0.5"},
                          DType::kFloat32,
                          ACLNN_SUCCESS,
                          {}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_f32_precision_boundary",
                          {{4}, DType::kFloat32, {"1.0001", "0.9999", "1.125", "0.875"}},
                          {DType::kFloat32, "1024"},
                          DType::kFloat32,
                          ACLNN_SUCCESS,
                          {1e-4, 1e-4}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_f32_fractional_precision",
                          {{4}, DType::kFloat32, {"0.125", "0.2", "1.75", "3.25"}},
                          {DType::kFloat32, "2.5"},
                          DType::kFloat32,
                          ACLNN_SUCCESS,
                          {1e-4, 1e-4}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_f32_large_non_integer",
                          {{4}, DType::kFloat32, {"8", "16", "32", "64"}},
                          {DType::kFloat32, "1.25"},
                          DType::kFloat32,
                          ACLNN_SUCCESS,
                          {1e-4, 1e-4}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_f16_square", {{4}, DType::kFloat16, {"1", "2", "3", "4"}},
                          {DType::kFloat16, "2"}, DType::kFloat16, ACLNN_SUCCESS, {}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_f16_generic", {{4}, DType::kFloat16, {"0.5", "1.5", "2.5", "3.5"}},
                          {DType::kFloat16, "1.5"}, DType::kFloat16, ACLNN_SUCCESS, {1e-3, 1e-3}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_f16_precision_boundary",
                          {{4}, DType::kFloat16, {"1.03125", "0.96875", "1.125", "0.875"}},
                          {DType::kFloat16, "5"},
                          DType::kFloat16,
                          ACLNN_SUCCESS,
                          {2e-3, 2e-3}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_f16_underflow_boundary",
                          {{4}, DType::kFloat16, {"0.5", "0.75", "0.875", "0.9375"}},
                          {DType::kFloat16, "11"},
                          DType::kFloat16,
                          ACLNN_SUCCESS,
                          {2e-3, 2e-3}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_f16_fractional_precision",
                          {{4}, DType::kFloat16, {"0.25", "0.5", "1.5", "2.25"}},
                          {DType::kFloat16, "2.5"},
                          DType::kFloat16,
                          ACLNN_SUCCESS,
                          {2e-3, 2e-3}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_bf16_square", {{4}, DType::kBFloat16, {"1", "2", "3", "4"}},
                          {DType::kBFloat16, "2"}, DType::kBFloat16, ACLNN_SUCCESS, {2e-2, 2e-2}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_bf16_precision_boundary",
                          {{4}, DType::kBFloat16, {"1.0078125", "0.9921875", "1.125", "0.875"}},
                          {DType::kBFloat16, "16"},
                          DType::kBFloat16,
                          ACLNN_SUCCESS,
                          {3e-2, 3e-2}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_bf16_fractional_precision",
                          {{4}, DType::kBFloat16, {"0.5", "0.75", "1.5", "2.5"}},
                          {DType::kBFloat16, "1.5"},
                          DType::kBFloat16,
                          ACLNN_SUCCESS,
                          {3e-2, 3e-2}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_u8_cube", {{4}, DType::kUInt8, {"1", "2", "3", "4"}},
                          {DType::kUInt8, "3"}, DType::kUInt8, ACLNN_SUCCESS, {}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_s8_square", {{4}, DType::kInt8, {"1", "2", "3", "4"}},
                          {DType::kInt8, "2"}, DType::kInt8, ACLNN_SUCCESS, {}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_s16_generic", {{4}, DType::kInt16, {"1", "2", "3", "4"}},
                          {DType::kInt16, "4"}, DType::kInt16, ACLNN_SUCCESS, {}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_s32_exp0", {{4}, DType::kInt32, {"2", "3", "4", "5"}},
                          {DType::kInt32, "0"}, DType::kInt32, ACLNN_SUCCESS, {}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_f64_generic",
                          {{4}, DType::kDouble, {"0.5", "1.5", "2.5", "3.5"}},
                          {DType::kDouble, "1.5"},
                          DType::kDouble,
                          ACLNN_SUCCESS,
                          {1e-12, 1e-12}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_f64_precision_boundary",
                          {{4}, DType::kDouble, {"1.0000001", "0.9999999", "1.125", "0.875"}},
                          {DType::kDouble, "4096"},
                          DType::kDouble,
                          ACLNN_SUCCESS,
                          {1e-11, 1e-11}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_s64_positive_integer_exp",
                          {{4}, DType::kInt64, {"2", "3", "4", "5"}},
                          {DType::kInt64, "3"},
                          DType::kInt64,
                          ACLNN_SUCCESS,
                          {}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_s64_exp0",
                          {{4}, DType::kInt64, {"2", "3", "4", "5"}},
                          {DType::kInt64, "0"},
                          DType::kInt64,
                          ACLNN_SUCCESS,
                          {}},
                         stream, &summary);
  RunPowTensorScalarCase({"pow_tensor_scalar_empty_f32", {{0}, DType::kFloat32, {}},
                          {DType::kFloat32, "2"}, DType::kFloat32, ACLNN_SUCCESS, {}},
                         stream, &summary);

  RunInplacePowTensorScalarCase({"inplace_pow_tensor_scalar_f32_square", {{4}, DType::kFloat32, {"1", "2", "3", "4"}},
                                 {DType::kFloat32, "2"}, DType::kFloat32, ACLNN_SUCCESS, {}},
                                stream, &summary);
  RunInplacePowTensorScalarCase({"inplace_pow_tensor_scalar_s32_cube", {{4}, DType::kInt32, {"1", "2", "3", "4"}},
                                 {DType::kInt32, "3"}, DType::kInt32, ACLNN_SUCCESS, {}},
                                stream, &summary);
  RunInplacePowTensorScalarCase({"inplace_pow_tensor_scalar_f16_generic",
                                 {{4}, DType::kFloat16, {"0.5", "1.5", "2.5", "3.5"}},
                                 {DType::kFloat16, "1.5"},
                                 DType::kFloat16,
                                 ACLNN_SUCCESS,
                                 {1e-3, 1e-3}},
                                stream, &summary);
  RunInplacePowTensorScalarCase({"inplace_pow_tensor_scalar_f32_fractional_precision",
                                 {{4}, DType::kFloat32, {"0.125", "0.5", "1.5", "3.5"}},
                                 {DType::kFloat32, "2.5"},
                                 DType::kFloat32,
                                 ACLNN_SUCCESS,
                                 {1e-4, 1e-4}},
                                stream, &summary);
  RunInplacePowTensorScalarCase({"inplace_pow_tensor_scalar_f64_generic",
                                 {{4}, DType::kDouble, {"0.5", "1.5", "2.5", "3.5"}},
                                 {DType::kDouble, "1.5"},
                                 DType::kDouble,
                                 ACLNN_SUCCESS,
                                 {1e-12, 1e-12}},
                                stream, &summary);
  RunInplacePowTensorScalarCase({"inplace_pow_tensor_scalar_empty_s32", {{0}, DType::kInt32, {}},
                                 {DType::kInt32, "3"}, DType::kInt32, ACLNN_SUCCESS, {}},
                                stream, &summary);

  RunPowScalarTensorCase({"pow_scalar_tensor_fill_ones", {DType::kFloat32, "1"},
                          {{4}, DType::kFloat32, {"0", "1", "2", "3"}}, DType::kFloat32, ACLNN_SUCCESS, {}},
                         stream, &summary);
  RunPowScalarTensorCase({"pow_scalar_tensor_f32_generic", {DType::kFloat32, "4"},
                          {{4}, DType::kFloat32, {"0.5", "1", "1.5", "2"}}, DType::kFloat32, ACLNN_SUCCESS,
                          {1e-4, 1e-4}},
                         stream, &summary);
  RunPowScalarTensorCase({"pow_scalar_tensor_f32_overflow_inf", {DType::kFloat32, "1e10"},
                          {{4}, DType::kFloat32, {"2", "4", "8", "16"}}, DType::kFloat32, ACLNN_SUCCESS, {}},
                         stream, &summary);
  RunPowScalarTensorCase({"pow_scalar_tensor_f32_precision_boundary", {DType::kFloat32, "1.0002"},
                          {{4}, DType::kFloat32, {"128", "256", "512", "1024"}}, DType::kFloat32, ACLNN_SUCCESS,
                          {1e-4, 1e-4}},
                         stream, &summary);
  RunPowScalarTensorCase({"pow_scalar_tensor_f32_negative_exponents",
                          {DType::kFloat32, "4"},
                          {{4}, DType::kFloat32, {"-1", "-2", "-3", "-4"}},
                          DType::kFloat32,
                          ACLNN_SUCCESS,
                          {1e-4, 1e-4}},
                         stream, &summary);
  RunPowScalarTensorCase({"pow_scalar_tensor_f32_fractional_precision",
                          {DType::kFloat32, "1.03125"},
                          {{4}, DType::kFloat32, {"8", "16", "24", "32"}},
                          DType::kFloat32,
                          ACLNN_SUCCESS,
                          {1e-4, 1e-4}},
                         stream, &summary);
  RunPowScalarTensorCase({"pow_scalar_tensor_f16", {DType::kFloat16, "2"},
                          {{4}, DType::kFloat16, {"0.5", "1", "2", "3"}}, DType::kFloat16, ACLNN_SUCCESS, {1e-3, 1e-3}},
                         stream, &summary);
  RunPowScalarTensorCase({"pow_scalar_tensor_bf16", {DType::kBFloat16, "2"},
                          {{4}, DType::kBFloat16, {"0.5", "1", "2", "3"}}, DType::kBFloat16, ACLNN_SUCCESS, {2e-2, 2e-2}},
                         stream, &summary);
  RunPowScalarTensorCase({"pow_scalar_tensor_s32", {DType::kInt32, "2"},
                          {{4}, DType::kInt32, {"0", "1", "2", "3"}}, DType::kInt32, ACLNN_SUCCESS, {}},
                         stream, &summary);
  RunPowScalarTensorCase({"pow_scalar_tensor_f64_generic",
                          {DType::kDouble, "2.5"},
                          {{4}, DType::kDouble, {"0.5", "1", "2", "3"}},
                          DType::kDouble,
                          ACLNN_SUCCESS,
                          {1e-12, 1e-12}},
                         stream, &summary);
  RunPowScalarTensorCase({"pow_scalar_tensor_f64_precision_boundary",
                          {DType::kDouble, "1.0000001"},
                          {{4}, DType::kDouble, {"1024", "2048", "4096", "8192"}},
                          DType::kDouble,
                          ACLNN_SUCCESS,
                          {1e-11, 1e-11}},
                         stream, &summary);
  RunPowScalarTensorCase({"pow_scalar_tensor_s64",
                          {DType::kInt64, "2"},
                          {{4}, DType::kInt64, {"0", "1", "2", "3"}},
                          DType::kInt64,
                          ACLNN_SUCCESS,
                          {}},
                         stream, &summary);
  RunPowScalarTensorCase({"pow_scalar_tensor_s64_negative_exponents",
                          {DType::kInt64, "2"},
                          {{4}, DType::kInt64, {"-1", "-2", "-3", "-4"}},
                          DType::kInt64,
                          ACLNN_SUCCESS,
                          {}},
                         stream, &summary);
  RunPowScalarTensorCase({"pow_scalar_tensor_f16_precision_boundary",
                          {DType::kFloat16, "1.03125"},
                          {{4}, DType::kFloat16, {"4", "8", "12", "16"}},
                          DType::kFloat16,
                          ACLNN_SUCCESS,
                          {2e-3, 2e-3}},
                         stream, &summary);
  RunPowScalarTensorCase({"pow_scalar_tensor_bf16_precision_boundary",
                          {DType::kBFloat16, "1.03125"},
                          {{4}, DType::kBFloat16, {"4", "8", "12", "16"}},
                          DType::kBFloat16,
                          ACLNN_SUCCESS,
                          {3e-2, 3e-2}},
                         stream, &summary);
  RunPowScalarTensorCase({"pow_scalar_tensor_empty_f32", {DType::kFloat32, "2"},
                          {{0}, DType::kFloat32, {}}, DType::kFloat32, ACLNN_SUCCESS, {}},
                         stream, &summary);

  RunPowTensorTensorCase({"pow_tensor_tensor_f32_same_shape",
                          {{2, 2}, DType::kFloat32, {"1", "2", "3", "4"}},
                          {{2, 2}, DType::kFloat32, {"2", "3", "2", "1"}},
                          {2, 2},
                          DType::kFloat32,
                          ACLNN_SUCCESS,
                          {}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_f16_same_shape",
                          {{2, 2}, DType::kFloat16, {"1", "2", "3", "4"}},
                          {{2, 2}, DType::kFloat16, {"2", "3", "2", "1"}},
                          {2, 2},
                          DType::kFloat16,
                          ACLNN_SUCCESS,
                          {1e-3, 1e-3}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_bf16_same_shape",
                          {{2, 2}, DType::kBFloat16, {"1", "2", "3", "4"}},
                          {{2, 2}, DType::kBFloat16, {"2", "3", "2", "1"}},
                          {2, 2},
                          DType::kBFloat16,
                          ACLNN_SUCCESS,
                          {2e-2, 2e-2}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_u8_same_shape",
                          {{2, 2}, DType::kUInt8, {"1", "2", "3", "4"}},
                          {{2, 2}, DType::kUInt8, {"2", "3", "2", "1"}},
                          {2, 2},
                          DType::kUInt8,
                          ACLNN_SUCCESS,
                          {}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_s8_same_shape",
                          {{2, 2}, DType::kInt8, {"1", "2", "3", "4"}},
                          {{2, 2}, DType::kInt8, {"2", "3", "2", "1"}},
                          {2, 2},
                          DType::kInt8,
                          ACLNN_SUCCESS,
                          {}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_s16_same_shape",
                          {{2, 2}, DType::kInt16, {"1", "2", "3", "4"}},
                          {{2, 2}, DType::kInt16, {"2", "3", "2", "1"}},
                          {2, 2},
                          DType::kInt16,
                          ACLNN_SUCCESS,
                          {}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_s32_same_shape",
                          {{2, 2}, DType::kInt32, {"1", "2", "3", "4"}},
                          {{2, 2}, DType::kInt32, {"2", "3", "2", "1"}},
                          {2, 2},
                          DType::kInt32,
                          ACLNN_SUCCESS,
                          {}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_f64_same_shape",
                          {{2, 2}, DType::kDouble, {"1", "2", "3", "4"}},
                          {{2, 2}, DType::kDouble, {"2", "3", "2", "1"}},
                          {2, 2},
                          DType::kDouble,
                          ACLNN_SUCCESS,
                          {1e-12, 1e-12}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_f64_broadcast",
                          {{2, 1}, DType::kDouble, {"2", "3"}},
                          {{1, 2}, DType::kDouble, {"2", "3"}},
                          {2, 2},
                          DType::kDouble,
                          ACLNN_SUCCESS,
                          {1e-12, 1e-12}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_f64_precision_boundary",
                          {{4}, DType::kDouble, {"1.0000001", "0.9999999", "1.0000002", "0.9999998"}},
                          {{4}, DType::kDouble, {"8192", "8192", "16384", "16384"}},
                          {4},
                          DType::kDouble,
                          ACLNN_SUCCESS,
                          {1e-11, 1e-11}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_s64_same_shape",
                          {{2, 2}, DType::kInt64, {"1", "2", "3", "4"}},
                          {{2, 2}, DType::kInt64, {"2", "3", "2", "1"}},
                          {2, 2},
                          DType::kInt64,
                          ACLNN_SUCCESS,
                          {}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_s64_broadcast",
                          {{2, 1}, DType::kInt64, {"2", "3"}},
                          {{1, 2}, DType::kInt64, {"2", "3"}},
                          {2, 2},
                          DType::kInt64,
                          ACLNN_SUCCESS,
                          {}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_f32_broadcast_without_loops",
                          {{2, 1}, DType::kFloat32, {"2", "3"}},
                          {{1, 2}, DType::kFloat32, {"2", "3"}},
                          {2, 2},
                          DType::kFloat32,
                          ACLNN_SUCCESS,
                          {}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_f32_broadcast_loops",
                          {{1, 1, 1, 1, 2, 2}, DType::kFloat32, {"1", "4", "9", "16"}},
                          {{1, 1, 1, 1, 2, 1}, DType::kFloat32, {"0.5", "2"}},
                          {1, 1, 1, 1, 2, 2},
                          DType::kFloat32,
                          ACLNN_SUCCESS,
                          {1e-4, 1e-4}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_f32_broadcast_precision",
                          {{2, 1, 2}, DType::kFloat32, {"1.0001", "0.9999", "1.0002", "0.9998"}},
                          {{1, 2, 1}, DType::kFloat32, {"256", "512"}},
                          {2, 2, 2},
                          DType::kFloat32,
                          ACLNN_SUCCESS,
                          {1e-4, 1e-4}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_s32_broadcast",
                          {{2, 1}, DType::kInt32, {"2", "3"}},
                          {{1, 2}, DType::kInt32, {"2", "3"}},
                          {2, 2},
                          DType::kInt32,
                          ACLNN_SUCCESS,
                          {}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_f16_broadcast",
                          {{2, 1}, DType::kFloat16, {"2", "3"}},
                          {{1, 2}, DType::kFloat16, {"2", "3"}},
                          {2, 2},
                          DType::kFloat16,
                          ACLNN_SUCCESS,
                          {1e-3, 1e-3}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_bf16_broadcast",
                          {{2, 1}, DType::kBFloat16, {"2", "3"}},
                          {{1, 2}, DType::kBFloat16, {"2", "3"}},
                          {2, 2},
                          DType::kBFloat16,
                          ACLNN_SUCCESS,
                          {2e-2, 2e-2}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_u8_broadcast",
                          {{2, 1}, DType::kUInt8, {"2", "3"}},
                          {{1, 2}, DType::kUInt8, {"2", "3"}},
                          {2, 2},
                          DType::kUInt8,
                          ACLNN_SUCCESS,
                          {}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_s8_broadcast",
                          {{2, 1}, DType::kInt8, {"2", "3"}},
                          {{1, 2}, DType::kInt8, {"2", "3"}},
                          {2, 2},
                          DType::kInt8,
                          ACLNN_SUCCESS,
                          {}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_s16_broadcast",
                          {{2, 1}, DType::kInt16, {"2", "3"}},
                          {{1, 2}, DType::kInt16, {"2", "3"}},
                          {2, 2},
                          DType::kInt16,
                          ACLNN_SUCCESS,
                          {}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_f32_negative_fractional_nan",
                          {{4}, DType::kFloat32, {"-1", "-4", "9", "16"}},
                          {{4}, DType::kFloat32, {"0.5", "0.5", "0.5", "0.5"}},
                          {4},
                          DType::kFloat32,
                          ACLNN_SUCCESS,
                          {}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_f32_overflow_inf",
                          {{4}, DType::kFloat32, {"1e10", "1e8", "1000", "100"}},
                          {{4}, DType::kFloat32, {"4", "8", "16", "32"}},
                          {4},
                          DType::kFloat32,
                          ACLNN_SUCCESS,
                          {}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_f32_precision_boundary",
                          {{4}, DType::kFloat32, {"1.0001", "0.9999", "1.0002", "0.9998"}},
                          {{4}, DType::kFloat32, {"1024", "1024", "2048", "2048"}},
                          {4},
                          DType::kFloat32,
                          ACLNN_SUCCESS,
                          {1e-4, 1e-4}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_f16_precision_boundary",
                          {{4}, DType::kFloat16, {"1.03125", "0.96875", "1.0625", "0.9375"}},
                          {{4}, DType::kFloat16, {"5", "5", "9", "9"}},
                          {4},
                          DType::kFloat16,
                          ACLNN_SUCCESS,
                          {2e-3, 2e-3}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_bf16_precision_boundary",
                          {{4}, DType::kBFloat16, {"1.0078125", "0.9921875", "1.03125", "0.96875"}},
                          {{4}, DType::kBFloat16, {"16", "16", "32", "32"}},
                          {4},
                          DType::kBFloat16,
                          ACLNN_SUCCESS,
                          {3e-2, 3e-2}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_f32_fractional_mix",
                          {{4}, DType::kFloat32, {"0.125", "0.5", "1.5", "3.5"}},
                          {{4}, DType::kFloat32, {"2.5", "1.5", "0.5", "-0.5"}},
                          {4},
                          DType::kFloat32,
                          ACLNN_SUCCESS,
                          {1e-4, 1e-4}},
                         stream, &summary);
  RunPowTensorTensorCase({"pow_tensor_tensor_empty_f32",
                          {{0}, DType::kFloat32, {}},
                          {{0}, DType::kFloat32, {}},
                          {0},
                          DType::kFloat32,
                          ACLNN_SUCCESS,
                          {}},
                         stream, &summary);

  RunInplacePowTensorTensorCase({"inplace_pow_tensor_tensor_f32",
                                 {{2, 2}, DType::kFloat32, {"1", "2", "3", "4"}},
                                 {{2, 2}, DType::kFloat32, {"2", "3", "2", "1"}},
                                 {2, 2},
                                 DType::kFloat32,
                                 ACLNN_SUCCESS,
                                 {}},
                                stream, &summary);
  RunInplacePowTensorTensorCase({"inplace_pow_tensor_tensor_s32",
                                 {{2, 2}, DType::kInt32, {"1", "2", "3", "4"}},
                                 {{2, 2}, DType::kInt32, {"2", "3", "2", "1"}},
                                 {2, 2},
                                 DType::kInt32,
                                 ACLNN_SUCCESS,
                                 {}},
                                stream, &summary);
  RunInplacePowTensorTensorCase({"inplace_pow_tensor_tensor_f16",
                                 {{2, 2}, DType::kFloat16, {"1", "2", "3", "4"}},
                                 {{2, 2}, DType::kFloat16, {"2", "3", "2", "1"}},
                                 {2, 2},
                                 DType::kFloat16,
                                 ACLNN_SUCCESS,
                                 {1e-3, 1e-3}},
                                stream, &summary);
  RunInplacePowTensorTensorCase({"inplace_pow_tensor_tensor_f64",
                                 {{2, 2}, DType::kDouble, {"1", "2", "3", "4"}},
                                 {{2, 2}, DType::kDouble, {"2", "3", "2", "1"}},
                                 {2, 2},
                                 DType::kDouble,
                                 ACLNN_SUCCESS,
                                 {1e-12, 1e-12}},
                                stream, &summary);
  RunInplacePowTensorTensorCase({"inplace_pow_tensor_tensor_f32_broadcast_like",
                                 {{2, 2}, DType::kFloat32, {"1.0001", "0.9999", "1.0002", "0.9998"}},
                                 {{2, 2}, DType::kFloat32, {"256", "512", "1024", "2048"}},
                                 {2, 2},
                                 DType::kFloat32,
                                 ACLNN_SUCCESS,
                                 {1e-4, 1e-4}},
                                stream, &summary);
  RunInplacePowTensorTensorCase({"inplace_pow_tensor_tensor_empty_f32",
                                 {{0}, DType::kFloat32, {}},
                                 {{0}, DType::kFloat32, {}},
                                 {0},
                                 DType::kFloat32,
                                 ACLNN_SUCCESS,
                                 {}},
                                stream, &summary);

  RunExp2Case({"exp2_f32_basic", {{4}, DType::kFloat32, {"0", "1", "2", "3"}}, DType::kFloat32, ACLNN_SUCCESS, {}},
              stream, &summary);
  RunExp2Case({"exp2_f16_basic", {{4}, DType::kFloat16, {"0", "1", "2", "3"}}, DType::kFloat16, ACLNN_SUCCESS,
               {1e-3, 1e-3}},
              stream, &summary);
  RunExp2Case({"exp2_f64_basic", {{4}, DType::kDouble, {"0", "1", "2", "3"}}, DType::kDouble, ACLNN_SUCCESS,
               {1e-12, 1e-12}},
              stream, &summary);
  RunExp2Case({"exp2_f64_midrange_precision", {{4}, DType::kDouble, {"-3.5", "-1.25", "1.25", "3.5"}},
               DType::kDouble, ACLNN_SUCCESS, {1e-12, 1e-12}},
              stream, &summary);
  RunExp2Case({"exp2_s64_to_f64", {{4}, DType::kInt64, {"0", "1", "2", "3"}}, DType::kDouble, ACLNN_SUCCESS,
               {1e-12, 1e-12}},
              stream, &summary);
  RunExp2Case({"exp2_bf16_basic", {{4}, DType::kBFloat16, {"0", "1", "2", "3"}}, DType::kBFloat16, ACLNN_SUCCESS,
               {2e-2, 2e-2}},
              stream, &summary);
  RunExp2Case({"exp2_s32_to_f32", {{4}, DType::kInt32, {"0", "1", "2", "3"}}, DType::kFloat32, ACLNN_SUCCESS, {}},
              stream, &summary);
  RunExp2Case({"exp2_f32_overflow_underflow", {{4}, DType::kFloat32, {"-150", "-20", "20", "150"}}, DType::kFloat32,
               ACLNN_SUCCESS, {}},
              stream, &summary);
  RunExp2Case({"exp2_f32_precision_boundary", {{4}, DType::kFloat32, {"-0.0001", "0.0001", "10.1", "-10.1"}},
               DType::kFloat32, ACLNN_SUCCESS, {1e-5, 1e-5}},
              stream, &summary);
  RunExp2Case({"exp2_f16_precision_boundary", {{4}, DType::kFloat16, {"-0.5", "0.5", "5.25", "-5.25"}},
               DType::kFloat16, ACLNN_SUCCESS, {1e-3, 1e-3}},
              stream, &summary);
  RunExp2Case({"exp2_f32_midrange_precision", {{4}, DType::kFloat32, {"-3.5", "-1.25", "1.25", "3.5"}},
               DType::kFloat32, ACLNN_SUCCESS, {1e-5, 1e-5}},
              stream, &summary);
  RunExp2Case({"exp2_f16_midrange_precision", {{4}, DType::kFloat16, {"-3.5", "-1.25", "1.25", "3.5"}},
               DType::kFloat16, ACLNN_SUCCESS, {1e-3, 1e-3}},
              stream, &summary);
  RunExp2Case({"exp2_bf16_midrange_precision", {{4}, DType::kBFloat16, {"-3.5", "-1.25", "1.25", "3.5"}},
               DType::kBFloat16, ACLNN_SUCCESS, {2e-2, 2e-2}},
              stream, &summary);
  RunExp2Case({"exp2_empty_f32", {{0}, DType::kFloat32, {}}, DType::kFloat32, ACLNN_SUCCESS, {}},
              stream, &summary);

  RunInplaceExp2Case({"inplace_exp2_f32_basic", {{4}, DType::kFloat32, {"0", "1", "2", "3"}}, DType::kFloat32,
                      ACLNN_SUCCESS, {}},
                     stream, &summary);
  RunInplaceExp2Case({"inplace_exp2_f16_basic", {{4}, DType::kFloat16, {"0", "1", "2", "3"}}, DType::kFloat16,
                      ACLNN_SUCCESS, {1e-3, 1e-3}},
                     stream, &summary);
  RunInplaceExp2Case({"inplace_exp2_bf16_basic", {{4}, DType::kBFloat16, {"0", "1", "2", "3"}}, DType::kBFloat16,
                      ACLNN_SUCCESS, {2e-2, 2e-2}},
                     stream, &summary);
  RunInplaceExp2Case({"inplace_exp2_f32_precision_boundary",
                      {{4}, DType::kFloat32, {"-0.0001", "0.0001", "10.1", "-10.1"}},
                      DType::kFloat32,
                      ACLNN_SUCCESS,
                      {1e-5, 1e-5}},
                     stream, &summary);
  RunInplaceExp2Case({"inplace_exp2_f16_precision_boundary",
                      {{4}, DType::kFloat16, {"-0.5", "0.5", "5.25", "-5.25"}},
                      DType::kFloat16,
                      ACLNN_SUCCESS,
                      {1e-3, 1e-3}},
                     stream, &summary);
  RunInplaceExp2Case({"inplace_exp2_f64_basic", {{4}, DType::kDouble, {"0", "1", "2", "3"}}, DType::kDouble,
                      ACLNN_SUCCESS, {1e-12, 1e-12}},
                     stream, &summary);
  RunInplaceExp2Case({"inplace_exp2_empty_f32", {{0}, DType::kFloat32, {}}, DType::kFloat32, ACLNN_SUCCESS, {}},
                     stream, &summary);

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  LOG_PRINT("pow test finished. total=%d failed=%d skipped=%d\n", summary.total, summary.failed, summary.skipped);
  return summary.failed == 0 ? 0 : 1;
}
