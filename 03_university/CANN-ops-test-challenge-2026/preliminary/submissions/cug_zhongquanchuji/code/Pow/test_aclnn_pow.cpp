/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

struct CaseStats {
  int passed = 0;
  int failed = 0;
};

struct TensorHolder {
  void *deviceAddr = nullptr;
  aclTensor *tensor = nullptr;
  std::vector<int64_t> shape;
  aclDataType dtype = ACL_FLOAT;
};

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto dim : shape) {
    shapeSize *= dim;
  }
  return shapeSize;
}

size_t GetDataTypeSize(aclDataType dtype) {
  switch (dtype) {
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

double GetTolerance(aclDataType dtype) {
  switch (dtype) {
    case ACL_FLOAT16:
      return 1e-3;
    case ACL_BF16:
      return 1e-2;
    case ACL_FLOAT:
      return 1e-5;
    case ACL_DOUBLE:
      return 1e-8;
    default:
      return 0.0;
  }
}

bool IsFloatingOutput(aclDataType dtype) {
  return dtype == ACL_FLOAT16 || dtype == ACL_BF16 || dtype == ACL_FLOAT || dtype == ACL_DOUBLE;
}

uint32_t FloatBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float BitsToFloat(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

static inline float Float16ToFloat(uint16_t value) {
  unsigned int sign = (value >> 15) & 0x1;
  unsigned int exponent = (value >> 10) & 0x1f;
  unsigned int mantissa = value & 0x3ff;

  float result;
  if (exponent == 0) {
    result = mantissa * 0.0000019073486328125f;
  } else if (exponent == 31) {
    result = (mantissa == 0) ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
  } else {
    result = (1.0f + mantissa * 0.0009765625f) * std::pow(2.0f, static_cast<int>(exponent) - 15);
  }
  return sign ? -result : result;
}

static inline uint16_t FloatToFloat16(float value) {
  uint32_t bits = FloatBits(value);
  uint16_t sign = (bits >> 16) & 0x8000;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffff;

  if (std::isnan(value)) {
    return sign | 0x7e00;
  }
  if (std::isinf(value)) {
    return sign | 0x7c00;
  }
  if (exponent <= 0) {
    return sign;
  }
  if (exponent >= 31) {
    return sign | 0x7c00;
  }
  return static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
}

uint16_t FloatToBFloat16(float value) {
  uint32_t bits = FloatBits(value);
  return static_cast<uint16_t>(bits >> 16);
}

float BFloat16ToFloat(uint16_t value) {
  return BitsToFloat(static_cast<uint32_t>(value) << 16);
}

void AppendEncodedValue(std::vector<uint8_t> *buffer, aclDataType dtype, double value) {
  switch (dtype) {
    case ACL_FLOAT: {
      float converted = static_cast<float>(value);
      const auto *ptr = reinterpret_cast<const uint8_t *>(&converted);
      buffer->insert(buffer->end(), ptr, ptr + sizeof(converted));
      return;
    }
    case ACL_DOUBLE: {
      double converted = value;
      const auto *ptr = reinterpret_cast<const uint8_t *>(&converted);
      buffer->insert(buffer->end(), ptr, ptr + sizeof(converted));
      return;
    }
    case ACL_FLOAT16: {
      uint16_t converted = FloatToFloat16(static_cast<float>(value));
      const auto *ptr = reinterpret_cast<const uint8_t *>(&converted);
      buffer->insert(buffer->end(), ptr, ptr + sizeof(converted));
      return;
    }
    case ACL_BF16: {
      uint16_t converted = FloatToBFloat16(static_cast<float>(value));
      const auto *ptr = reinterpret_cast<const uint8_t *>(&converted);
      buffer->insert(buffer->end(), ptr, ptr + sizeof(converted));
      return;
    }
    case ACL_INT32: {
      int32_t converted = static_cast<int32_t>(value);
      const auto *ptr = reinterpret_cast<const uint8_t *>(&converted);
      buffer->insert(buffer->end(), ptr, ptr + sizeof(converted));
      return;
    }
    case ACL_INT16: {
      int16_t converted = static_cast<int16_t>(value);
      const auto *ptr = reinterpret_cast<const uint8_t *>(&converted);
      buffer->insert(buffer->end(), ptr, ptr + sizeof(converted));
      return;
    }
    case ACL_INT8: {
      int8_t converted = static_cast<int8_t>(value);
      const auto *ptr = reinterpret_cast<const uint8_t *>(&converted);
      buffer->insert(buffer->end(), ptr, ptr + sizeof(converted));
      return;
    }
    case ACL_UINT8: {
      uint8_t converted = static_cast<uint8_t>(value);
      const auto *ptr = reinterpret_cast<const uint8_t *>(&converted);
      buffer->insert(buffer->end(), ptr, ptr + sizeof(converted));
      return;
    }
    case ACL_BOOL: {
      uint8_t converted = (value != 0.0) ? 1 : 0;
      buffer->push_back(converted);
      return;
    }
    default:
      return;
  }
}

std::vector<uint8_t> EncodeValues(const std::vector<double> &values, aclDataType dtype) {
  std::vector<uint8_t> buffer;
  buffer.reserve(values.size() * GetDataTypeSize(dtype));
  for (double value : values) {
    AppendEncodedValue(&buffer, dtype, value);
  }
  return buffer;
}

double DecodeValue(const std::vector<uint8_t> &buffer, aclDataType dtype, int64_t index) {
  size_t elemSize = GetDataTypeSize(dtype);
  const uint8_t *ptr = buffer.data() + static_cast<size_t>(index) * elemSize;
  switch (dtype) {
    case ACL_FLOAT: {
      float value = 0.0f;
      std::memcpy(&value, ptr, sizeof(value));
      return static_cast<double>(value);
    }
    case ACL_DOUBLE: {
      double value = 0.0;
      std::memcpy(&value, ptr, sizeof(value));
      return value;
    }
    case ACL_FLOAT16: {
      uint16_t value = 0;
      std::memcpy(&value, ptr, sizeof(value));
      return static_cast<double>(Float16ToFloat(value));
    }
    case ACL_BF16: {
      uint16_t value = 0;
      std::memcpy(&value, ptr, sizeof(value));
      return static_cast<double>(BFloat16ToFloat(value));
    }
    case ACL_INT32: {
      int32_t value = 0;
      std::memcpy(&value, ptr, sizeof(value));
      return static_cast<double>(value);
    }
    case ACL_INT16: {
      int16_t value = 0;
      std::memcpy(&value, ptr, sizeof(value));
      return static_cast<double>(value);
    }
    case ACL_INT8: {
      int8_t value = 0;
      std::memcpy(&value, ptr, sizeof(value));
      return static_cast<double>(value);
    }
    case ACL_UINT8: {
      uint8_t value = 0;
      std::memcpy(&value, ptr, sizeof(value));
      return static_cast<double>(value);
    }
    case ACL_BOOL: {
      uint8_t value = 0;
      std::memcpy(&value, ptr, sizeof(value));
      return value == 0 ? 0.0 : 1.0;
    }
    default:
      return 0.0;
  }
}

std::vector<double> DecodeValues(const std::vector<uint8_t> &buffer, aclDataType dtype, int64_t elemCount) {
  std::vector<double> values(elemCount, 0.0);
  for (int64_t i = 0; i < elemCount; ++i) {
    values[static_cast<size_t>(i)] = DecodeValue(buffer, dtype, i);
  }
  return values;
}

double QuantizeForOutput(double value, aclDataType dtype) {
  std::vector<double> values = {value};
  auto encoded = EncodeValues(values, dtype);
  return DecodeValue(encoded, dtype, 0);
}

bool AlmostEqual(double actual, double expected, aclDataType dtype) {
  if (std::isnan(actual) && std::isnan(expected)) {
    return true;
  }
  if (std::isinf(actual) || std::isinf(expected)) {
    return actual == expected;
  }
  double tol = GetTolerance(dtype);
  if (!IsFloatingOutput(dtype)) {
    return actual == expected;
  }
  return std::fabs(actual - expected) <= tol + tol * std::fabs(expected);
}

std::vector<int64_t> ComputeStrides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
  }
  return strides;
}

int Init(int32_t deviceId, aclrtStream *stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

int CreateAclTensorFromBytes(const std::vector<uint8_t> &hostData, const std::vector<int64_t> &shape,
                             aclDataType dataType, TensorHolder *holder) {
  holder->shape = shape;
  holder->dtype = dataType;
  auto ret = ACL_SUCCESS;
  if (!hostData.empty()) {
    ret = aclrtMalloc(&holder->deviceAddr, hostData.size(), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(holder->deviceAddr, hostData.size(), hostData.data(), hostData.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  }

  std::vector<int64_t> strides = ComputeStrides(shape);
  holder->tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                   shape.data(), shape.size(), holder->deviceAddr);
  CHECK_RET(holder->tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
  return ACL_SUCCESS;
}

int CreateAclTensorFromDoubles(const std::vector<double> &values, const std::vector<int64_t> &shape,
                               aclDataType dataType, TensorHolder *holder) {
  return CreateAclTensorFromBytes(EncodeValues(values, dataType), shape, dataType, holder);
}

void DestroyTensorHolder(TensorHolder *holder) {
  if (holder->tensor != nullptr) {
    aclDestroyTensor(holder->tensor);
    holder->tensor = nullptr;
  }
  if (holder->deviceAddr != nullptr) {
    aclrtFree(holder->deviceAddr);
    holder->deviceAddr = nullptr;
  }
}

bool CopyTensorToHost(const TensorHolder &holder, std::vector<uint8_t> *hostData) {
  hostData->assign(static_cast<size_t>(GetShapeSize(holder.shape)) * GetDataTypeSize(holder.dtype), 0);
  auto ret = aclrtMemcpy(hostData->data(), hostData->size(), holder.deviceAddr, hostData->size(),
                         ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
    return false;
  }
  return true;
}

std::vector<int64_t> BroadcastShape(const std::vector<int64_t> &a, const std::vector<int64_t> &b) {
  size_t rank = a.size() > b.size() ? a.size() : b.size();
  std::vector<int64_t> result(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    int64_t aDim = 1;
    int64_t bDim = 1;
    if (i >= rank - a.size()) {
      aDim = a[i - (rank - a.size())];
    }
    if (i >= rank - b.size()) {
      bDim = b[i - (rank - b.size())];
    }
    if (aDim != bDim && aDim != 1 && bDim != 1) {
      return {};
    }
    result[i] = aDim > bDim ? aDim : bDim;
  }
  return result;
}

int64_t GetBroadcastOffset(const std::vector<int64_t> &shape, const std::vector<int64_t> &outShape, int64_t flatIndex) {
  std::vector<int64_t> outStrides = ComputeStrides(outShape);
  std::vector<int64_t> inStrides = ComputeStrides(shape);
  int64_t offset = 0;
  size_t outRank = outShape.size();
  size_t inRank = shape.size();
  size_t shift = outRank - inRank;
  for (size_t i = 0; i < outRank; ++i) {
    int64_t coord = outStrides.empty() ? 0 : (flatIndex / outStrides[i]) % outShape[i];
    if (i < shift) {
      continue;
    }
    size_t inIndex = i - shift;
    int64_t inCoord = shape[inIndex] == 1 ? 0 : coord;
    offset += inCoord * inStrides[inIndex];
  }
  return offset;
}

bool ReportCase(const std::string &caseName, bool success, CaseStats *stats) {
  if (success) {
    ++stats->passed;
    LOG_PRINT("[PASS] %s\n", caseName.c_str());
    return true;
  }
  ++stats->failed;
  LOG_PRINT("[FAIL] %s\n", caseName.c_str());
  return false;
}

bool VerifyVector(const std::string &caseName, const std::vector<double> &actual, const std::vector<double> &expected,
                  aclDataType outDtype, CaseStats *stats) {
  if (actual.size() != expected.size()) {
    LOG_PRINT("size mismatch in %s, actual=%zu expected=%zu\n", caseName.c_str(), actual.size(), expected.size());
    return ReportCase(caseName, false, stats);
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!AlmostEqual(actual[i], expected[i], outDtype)) {
      LOG_PRINT("case %s mismatch at %zu, actual=%f expected=%f\n", caseName.c_str(), i, actual[i], expected[i]);
      return ReportCase(caseName, false, stats);
    }
  }
  return ReportCase(caseName, true, stats);
}

std::vector<double> QuantizedExpected(const std::vector<double> &rawExpected, aclDataType outDtype) {
  std::vector<double> expected(rawExpected.size(), 0.0);
  for (size_t i = 0; i < rawExpected.size(); ++i) {
    expected[i] = QuantizeForOutput(rawExpected[i], outDtype);
  }
  return expected;
}

bool AllocateWorkspace(uint64_t workspaceSize, void **workspaceAddr) {
  *workspaceAddr = nullptr;
  if (workspaceSize == 0) {
    return true;
  }
  auto ret = aclrtMalloc(workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
    return false;
  }
  return true;
}

bool RunPowTensorScalarCase(const std::string &caseName, const std::vector<double> &selfValues,
                            const std::vector<int64_t> &shape, aclDataType dtype, double exponentValue,
                            aclDataType exponentType, aclrtStream stream, CaseStats *stats) {
  TensorHolder self;
  TensorHolder out;
  aclScalar *exponent = nullptr;
  void *workspaceAddr = nullptr;
  aclOpExecutor *executor = nullptr;
  uint64_t workspaceSize = 0;
  bool success = false;
  bool reported = false;

  if (CreateAclTensorFromDoubles(selfValues, shape, dtype, &self) != ACL_SUCCESS) {
    goto cleanup;
  }
  if (CreateAclTensorFromDoubles(std::vector<double>(selfValues.size(), 0.0), shape, dtype, &out) != ACL_SUCCESS) {
    goto cleanup;
  }

  if (exponentType == ACL_INT32) {
    int32_t exponentData = static_cast<int32_t>(exponentValue);
    exponent = aclCreateScalar(&exponentData, ACL_INT32);
  } else {
    float exponentData = static_cast<float>(exponentValue);
    exponent = aclCreateScalar(&exponentData, ACL_FLOAT);
  }
  if (exponent == nullptr) {
    goto cleanup;
  }

  if (aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent, out.tensor, &workspaceSize, &executor) != ACL_SUCCESS) {
    goto cleanup;
  }
  if (!AllocateWorkspace(workspaceSize, &workspaceAddr)) {
    goto cleanup;
  }
  if (aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream) != ACL_SUCCESS) {
    goto cleanup;
  }
  if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    goto cleanup;
  }

  {
    std::vector<uint8_t> hostBytes;
    if (!CopyTensorToHost(out, &hostBytes)) {
      goto cleanup;
    }
    auto actual = DecodeValues(hostBytes, dtype, GetShapeSize(shape));
    std::vector<double> rawExpected(selfValues.size(), 0.0);
    for (size_t i = 0; i < selfValues.size(); ++i) {
      rawExpected[i] = std::pow(selfValues[i], exponentValue);
    }
    success = VerifyVector(caseName, actual, QuantizedExpected(rawExpected, dtype), dtype, stats);
    reported = true;
  }

cleanup:
  if (!success && !reported) {
    ReportCase(caseName, false, stats);
  }
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  DestroyTensorHolder(&self);
  DestroyTensorHolder(&out);
  return success;
}

bool RunInplacePowTensorScalarCase(const std::string &caseName, const std::vector<double> &selfValues,
                                   const std::vector<int64_t> &shape, aclDataType dtype, double exponentValue,
                                   aclrtStream stream, CaseStats *stats) {
  TensorHolder self;
  aclScalar *exponent = nullptr;
  void *workspaceAddr = nullptr;
  aclOpExecutor *executor = nullptr;
  uint64_t workspaceSize = 0;
  bool success = false;
  bool reported = false;

  if (CreateAclTensorFromDoubles(selfValues, shape, dtype, &self) != ACL_SUCCESS) {
    goto cleanup;
  }

  {
    float exponentData = static_cast<float>(exponentValue);
    exponent = aclCreateScalar(&exponentData, ACL_FLOAT);
  }
  if (exponent == nullptr) {
    goto cleanup;
  }

  if (aclnnInplacePowTensorScalarGetWorkspaceSize(self.tensor, exponent, &workspaceSize, &executor) != ACL_SUCCESS) {
    goto cleanup;
  }
  if (!AllocateWorkspace(workspaceSize, &workspaceAddr)) {
    goto cleanup;
  }
  if (aclnnInplacePowTensorScalar(workspaceAddr, workspaceSize, executor, stream) != ACL_SUCCESS) {
    goto cleanup;
  }
  if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    goto cleanup;
  }

  {
    std::vector<uint8_t> hostBytes;
    if (!CopyTensorToHost(self, &hostBytes)) {
      goto cleanup;
    }
    auto actual = DecodeValues(hostBytes, dtype, GetShapeSize(shape));
    std::vector<double> rawExpected(selfValues.size(), 0.0);
    for (size_t i = 0; i < selfValues.size(); ++i) {
      rawExpected[i] = std::pow(selfValues[i], exponentValue);
    }
    success = VerifyVector(caseName, actual, QuantizedExpected(rawExpected, dtype), dtype, stats);
    reported = true;
  }

cleanup:
  if (!success && !reported) {
    ReportCase(caseName, false, stats);
  }
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  DestroyTensorHolder(&self);
  return success;
}

bool RunPowScalarTensorCase(const std::string &caseName, double baseValue, aclDataType baseType,
                            const std::vector<double> &expValues, const std::vector<int64_t> &shape,
                            aclDataType dtype, aclrtStream stream, CaseStats *stats) {
  TensorHolder exponent;
  TensorHolder out;
  aclScalar *base = nullptr;
  void *workspaceAddr = nullptr;
  aclOpExecutor *executor = nullptr;
  uint64_t workspaceSize = 0;
  bool success = false;
  bool reported = false;

  if (CreateAclTensorFromDoubles(expValues, shape, dtype, &exponent) != ACL_SUCCESS) {
    goto cleanup;
  }
  if (CreateAclTensorFromDoubles(std::vector<double>(expValues.size(), 0.0), shape, dtype, &out) != ACL_SUCCESS) {
    goto cleanup;
  }

  if (baseType == ACL_INT32) {
    int32_t baseData = static_cast<int32_t>(baseValue);
    base = aclCreateScalar(&baseData, ACL_INT32);
  } else if (baseType == ACL_DOUBLE) {
    double baseData = baseValue;
    base = aclCreateScalar(&baseData, ACL_DOUBLE);
  } else {
    float baseData = static_cast<float>(baseValue);
    base = aclCreateScalar(&baseData, ACL_FLOAT);
  }
  if (base == nullptr) {
    goto cleanup;
  }

  if (aclnnPowScalarTensorGetWorkspaceSize(base, exponent.tensor, out.tensor, &workspaceSize, &executor) != ACL_SUCCESS) {
    goto cleanup;
  }
  if (!AllocateWorkspace(workspaceSize, &workspaceAddr)) {
    goto cleanup;
  }
  if (aclnnPowScalarTensor(workspaceAddr, workspaceSize, executor, stream) != ACL_SUCCESS) {
    goto cleanup;
  }
  if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    goto cleanup;
  }

  {
    std::vector<uint8_t> hostBytes;
    if (!CopyTensorToHost(out, &hostBytes)) {
      goto cleanup;
    }
    auto actual = DecodeValues(hostBytes, dtype, GetShapeSize(shape));
    std::vector<double> rawExpected(expValues.size(), 0.0);
    for (size_t i = 0; i < expValues.size(); ++i) {
      rawExpected[i] = std::pow(baseValue, expValues[i]);
    }
    success = VerifyVector(caseName, actual, QuantizedExpected(rawExpected, dtype), dtype, stats);
    reported = true;
  }

cleanup:
  if (!success && !reported) {
    ReportCase(caseName, false, stats);
  }
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  if (base != nullptr) {
    aclDestroyScalar(base);
  }
  DestroyTensorHolder(&exponent);
  DestroyTensorHolder(&out);
  return success;
}

bool RunPowTensorTensorCase(const std::string &caseName, const std::vector<double> &selfValues,
                            const std::vector<int64_t> &selfShape, const std::vector<double> &expValues,
                            const std::vector<int64_t> &expShape, aclDataType dtype, aclrtStream stream,
                            CaseStats *stats) {
  TensorHolder self;
  TensorHolder exponent;
  TensorHolder out;
  void *workspaceAddr = nullptr;
  aclOpExecutor *executor = nullptr;
  uint64_t workspaceSize = 0;
  bool success = false;
  bool reported = false;

  auto outShape = BroadcastShape(selfShape, expShape);
  if (outShape.empty()) {
    return ReportCase(caseName, false, stats);
  }
  if (CreateAclTensorFromDoubles(selfValues, selfShape, dtype, &self) != ACL_SUCCESS) {
    goto cleanup;
  }
  if (CreateAclTensorFromDoubles(expValues, expShape, dtype, &exponent) != ACL_SUCCESS) {
    goto cleanup;
  }
  if (CreateAclTensorFromDoubles(std::vector<double>(static_cast<size_t>(GetShapeSize(outShape)), 0.0), outShape, dtype,
                                 &out) != ACL_SUCCESS) {
    goto cleanup;
  }

  if (aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor, &workspaceSize, &executor) != ACL_SUCCESS) {
    goto cleanup;
  }
  if (!AllocateWorkspace(workspaceSize, &workspaceAddr)) {
    goto cleanup;
  }
  if (aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream) != ACL_SUCCESS) {
    goto cleanup;
  }
  if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    goto cleanup;
  }

  {
    std::vector<uint8_t> hostBytes;
    if (!CopyTensorToHost(out, &hostBytes)) {
      goto cleanup;
    }
    auto actual = DecodeValues(hostBytes, dtype, GetShapeSize(outShape));
    std::vector<double> rawExpected(static_cast<size_t>(GetShapeSize(outShape)), 0.0);
    for (int64_t i = 0; i < GetShapeSize(outShape); ++i) {
      int64_t selfIndex = GetBroadcastOffset(selfShape, outShape, i);
      int64_t expIndex = GetBroadcastOffset(expShape, outShape, i);
      rawExpected[static_cast<size_t>(i)] = std::pow(selfValues[static_cast<size_t>(selfIndex)],
                                                     expValues[static_cast<size_t>(expIndex)]);
    }
    success = VerifyVector(caseName, actual, QuantizedExpected(rawExpected, dtype), dtype, stats);
    reported = true;
  }

cleanup:
  if (!success && !reported) {
    ReportCase(caseName, false, stats);
  }
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  DestroyTensorHolder(&self);
  DestroyTensorHolder(&exponent);
  DestroyTensorHolder(&out);
  return success;
}

bool RunInplacePowTensorTensorCase(const std::string &caseName, const std::vector<double> &selfValues,
                                   const std::vector<int64_t> &shape, const std::vector<double> &expValues,
                                   aclDataType dtype, aclrtStream stream, CaseStats *stats) {
  TensorHolder self;
  TensorHolder exponent;
  void *workspaceAddr = nullptr;
  aclOpExecutor *executor = nullptr;
  uint64_t workspaceSize = 0;
  bool success = false;
  bool reported = false;

  if (CreateAclTensorFromDoubles(selfValues, shape, dtype, &self) != ACL_SUCCESS) {
    goto cleanup;
  }
  if (CreateAclTensorFromDoubles(expValues, shape, dtype, &exponent) != ACL_SUCCESS) {
    goto cleanup;
  }

  if (aclnnInplacePowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, &workspaceSize, &executor) != ACL_SUCCESS) {
    goto cleanup;
  }
  if (!AllocateWorkspace(workspaceSize, &workspaceAddr)) {
    goto cleanup;
  }
  if (aclnnInplacePowTensorTensor(workspaceAddr, workspaceSize, executor, stream) != ACL_SUCCESS) {
    goto cleanup;
  }
  if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    goto cleanup;
  }

  {
    std::vector<uint8_t> hostBytes;
    if (!CopyTensorToHost(self, &hostBytes)) {
      goto cleanup;
    }
    auto actual = DecodeValues(hostBytes, dtype, GetShapeSize(shape));
    std::vector<double> rawExpected(selfValues.size(), 0.0);
    for (size_t i = 0; i < selfValues.size(); ++i) {
      rawExpected[i] = std::pow(selfValues[i], expValues[i]);
    }
    success = VerifyVector(caseName, actual, QuantizedExpected(rawExpected, dtype), dtype, stats);
    reported = true;
  }

cleanup:
  if (!success && !reported) {
    ReportCase(caseName, false, stats);
  }
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  DestroyTensorHolder(&self);
  DestroyTensorHolder(&exponent);
  return success;
}

bool RunExp2Case(const std::string &caseName, const std::vector<double> &selfValues, const std::vector<int64_t> &shape,
                 aclDataType selfDtype, aclDataType outDtype, aclrtStream stream, CaseStats *stats) {
  TensorHolder self;
  TensorHolder out;
  void *workspaceAddr = nullptr;
  aclOpExecutor *executor = nullptr;
  uint64_t workspaceSize = 0;
  bool success = false;
  bool reported = false;

  if (CreateAclTensorFromDoubles(selfValues, shape, selfDtype, &self) != ACL_SUCCESS) {
    goto cleanup;
  }
  if (CreateAclTensorFromDoubles(std::vector<double>(selfValues.size(), 0.0), shape, outDtype, &out) != ACL_SUCCESS) {
    goto cleanup;
  }

  if (aclnnExp2GetWorkspaceSize(self.tensor, out.tensor, &workspaceSize, &executor) != ACL_SUCCESS) {
    goto cleanup;
  }
  if (!AllocateWorkspace(workspaceSize, &workspaceAddr)) {
    goto cleanup;
  }
  if (aclnnExp2(workspaceAddr, workspaceSize, executor, stream) != ACL_SUCCESS) {
    goto cleanup;
  }
  if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    goto cleanup;
  }

  {
    std::vector<uint8_t> hostBytes;
    if (!CopyTensorToHost(out, &hostBytes)) {
      goto cleanup;
    }
    auto actual = DecodeValues(hostBytes, outDtype, GetShapeSize(shape));
    std::vector<double> rawExpected(selfValues.size(), 0.0);
    for (size_t i = 0; i < selfValues.size(); ++i) {
      rawExpected[i] = std::pow(2.0, selfValues[i]);
    }
    success = VerifyVector(caseName, actual, QuantizedExpected(rawExpected, outDtype), outDtype, stats);
    reported = true;
  }

cleanup:
  if (!success && !reported) {
    ReportCase(caseName, false, stats);
  }
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  DestroyTensorHolder(&self);
  DestroyTensorHolder(&out);
  return success;
}

bool RunInplaceExp2Case(const std::string &caseName, const std::vector<double> &selfValues,
                        const std::vector<int64_t> &shape, aclDataType dtype, aclrtStream stream,
                        CaseStats *stats) {
  TensorHolder self;
  void *workspaceAddr = nullptr;
  aclOpExecutor *executor = nullptr;
  uint64_t workspaceSize = 0;
  bool success = false;
  bool reported = false;

  if (CreateAclTensorFromDoubles(selfValues, shape, dtype, &self) != ACL_SUCCESS) {
    goto cleanup;
  }
  if (aclnnInplaceExp2GetWorkspaceSize(self.tensor, &workspaceSize, &executor) != ACL_SUCCESS) {
    goto cleanup;
  }
  if (!AllocateWorkspace(workspaceSize, &workspaceAddr)) {
    goto cleanup;
  }
  if (aclnnInplaceExp2(workspaceAddr, workspaceSize, executor, stream) != ACL_SUCCESS) {
    goto cleanup;
  }
  if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    goto cleanup;
  }

  {
    std::vector<uint8_t> hostBytes;
    if (!CopyTensorToHost(self, &hostBytes)) {
      goto cleanup;
    }
    auto actual = DecodeValues(hostBytes, dtype, GetShapeSize(shape));
    std::vector<double> rawExpected(selfValues.size(), 0.0);
    for (size_t i = 0; i < selfValues.size(); ++i) {
      rawExpected[i] = std::pow(2.0, selfValues[i]);
    }
    success = VerifyVector(caseName, actual, QuantizedExpected(rawExpected, dtype), dtype, stats);
    reported = true;
  }

cleanup:
  if (!success && !reported) {
    ReportCase(caseName, false, stats);
  }
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  DestroyTensorHolder(&self);
  return success;
}

bool ExpectPowTensorScalarGetWorkspaceFailure(const std::string &caseName, const std::vector<double> &selfValues,
                                              const std::vector<int64_t> &shape, aclDataType dtype,
                                              int32_t exponentValue, aclrtStream, CaseStats *stats) {
  TensorHolder self;
  TensorHolder out;
  aclScalar *exponent = nullptr;
  aclOpExecutor *executor = nullptr;
  uint64_t workspaceSize = 0;

  if (CreateAclTensorFromDoubles(selfValues, shape, dtype, &self) != ACL_SUCCESS) {
    DestroyTensorHolder(&self);
    return ReportCase(caseName, false, stats);
  }
  if (CreateAclTensorFromDoubles(std::vector<double>(selfValues.size(), 0.0), shape, dtype, &out) != ACL_SUCCESS) {
    DestroyTensorHolder(&self);
    DestroyTensorHolder(&out);
    return ReportCase(caseName, false, stats);
  }
  exponent = aclCreateScalar(&exponentValue, ACL_INT32);
  bool failedAsExpected = exponent != nullptr &&
      aclnnPowTensorScalarGetWorkspaceSize(self.tensor, exponent, out.tensor, &workspaceSize, &executor) != ACL_SUCCESS;

  if (exponent != nullptr) {
    aclDestroyScalar(exponent);
  }
  DestroyTensorHolder(&self);
  DestroyTensorHolder(&out);
  return ReportCase(caseName, failedAsExpected, stats);
}

bool ExpectPowTensorTensorGetWorkspaceFailure(const std::string &caseName, const std::vector<double> &selfValues,
                                              const std::vector<int64_t> &selfShape, aclDataType selfDtype,
                                              const std::vector<double> &expValues,
                                              const std::vector<int64_t> &expShape, aclDataType expDtype,
                                              const std::vector<int64_t> &outShape, aclDataType outDtype,
                                              CaseStats *stats) {
  TensorHolder self;
  TensorHolder exponent;
  TensorHolder out;
  aclOpExecutor *executor = nullptr;
  uint64_t workspaceSize = 0;

  if (CreateAclTensorFromDoubles(selfValues, selfShape, selfDtype, &self) != ACL_SUCCESS) {
    return ReportCase(caseName, false, stats);
  }
  if (CreateAclTensorFromDoubles(expValues, expShape, expDtype, &exponent) != ACL_SUCCESS) {
    DestroyTensorHolder(&self);
    return ReportCase(caseName, false, stats);
  }
  if (CreateAclTensorFromDoubles(std::vector<double>(static_cast<size_t>(GetShapeSize(outShape)), 0.0), outShape, outDtype,
                                 &out) != ACL_SUCCESS) {
    DestroyTensorHolder(&self);
    DestroyTensorHolder(&exponent);
    return ReportCase(caseName, false, stats);
  }

  bool failedAsExpected =
      aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exponent.tensor, out.tensor, &workspaceSize, &executor) !=
      ACL_SUCCESS;
  DestroyTensorHolder(&self);
  DestroyTensorHolder(&exponent);
  DestroyTensorHolder(&out);
  return ReportCase(caseName, failedAsExpected, stats);
}

int main() {
  CaseStats stats;
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> squareShape = {2, 2};
  std::vector<double> squareBase = {0.25, 1.0, 2.0, 4.0};
  std::vector<double> doubleBase = {1.25, 2.5, 3.5, 4.5};
  std::vector<double> doubleExp = {1.0, 2.0, 0.5, 3.0};
  std::vector<int64_t> emptyShape = {0, 2};

  RunPowTensorScalarCase("PowTensorScalar_float32_generic", squareBase, squareShape, ACL_FLOAT, 1.5, ACL_FLOAT, stream,
                         &stats);
  RunPowTensorScalarCase("PowTensorScalar_float32_cube", squareBase, squareShape, ACL_FLOAT, 3.0, ACL_FLOAT, stream,
                         &stats);
  RunPowTensorScalarCase("PowTensorScalar_double_aicpu_generic", doubleBase, squareShape, ACL_DOUBLE, 1.5, ACL_FLOAT,
                         stream, &stats);
  RunInplacePowTensorScalarCase("InplacePowTensorScalar_float32_generic", {1.0, 2.0, 3.0, 4.0}, squareShape,
                                ACL_FLOAT, 1.5, stream, &stats);
  RunPowTensorScalarCase("PowTensorScalar_empty_tensor", {}, emptyShape, ACL_FLOAT, 1.5, ACL_FLOAT, stream, &stats);
  ExpectPowTensorScalarGetWorkspaceFailure("PowTensorScalar_int32_negative_exponent_invalid", {1.0, 2.0, 3.0, 4.0},
                                           squareShape, ACL_INT32, -1, stream, &stats);

  RunPowScalarTensorCase("PowScalarTensor_normal_branch", 2.0, ACL_FLOAT, {0.0, 1.0, 2.0, 3.0}, squareShape,
                         ACL_FLOAT, stream, &stats);
  RunPowScalarTensorCase("PowScalarTensor_double_aicpu_generic", 2.0, ACL_FLOAT, doubleExp, squareShape,
                         ACL_DOUBLE, stream, &stats);
  RunPowScalarTensorCase("PowScalarTensor_empty_tensor", 2.0, ACL_FLOAT, {}, emptyShape, ACL_FLOAT, stream, &stats);

  RunPowTensorTensorCase("PowTensorTensor_float32_same_shape", {1.0, 2.0, 3.0, 4.0}, squareShape,
                         {2.0, 3.0, 1.0, 0.0}, squareShape, ACL_FLOAT, stream, &stats);
  RunPowTensorTensorCase("PowTensorTensor_float32_broadcast", {1.0, 2.0, 4.0, 8.0, 3.0, 9.0}, {2, 3}, {1.0, 2.0, 3.0},
                         {1, 3}, ACL_FLOAT, stream, &stats);
  RunPowTensorTensorCase("PowTensorTensor_double_aicpu_same_shape", doubleBase, squareShape,
                         doubleExp, squareShape, ACL_DOUBLE, stream, &stats);
  RunPowTensorTensorCase("PowTensorTensor_float16_tiling_key", {1.0, 2.0, 3.0, 4.0}, squareShape,
                         {2.0, 1.0, 2.0, 1.0}, squareShape, ACL_FLOAT16, stream, &stats);
  RunPowTensorTensorCase("PowTensorTensor_bf16_tiling_key", {1.0, 2.0, 3.0, 4.0}, squareShape,
                         {2.0, 1.0, 2.0, 1.0}, squareShape, ACL_BF16, stream, &stats);
  RunPowTensorTensorCase("PowTensorTensor_uint8_tiling_key", {1.0, 2.0, 3.0, 4.0}, squareShape,
                         {2.0, 1.0, 2.0, 1.0}, squareShape, ACL_UINT8, stream, &stats);
  RunPowTensorTensorCase("PowTensorTensor_int8_tiling_key", {1.0, 2.0, 3.0, 4.0}, squareShape,
                         {2.0, 1.0, 2.0, 1.0}, squareShape, ACL_INT8, stream, &stats);
  RunPowTensorTensorCase("PowTensorTensor_int16_tiling_key", {1.0, 2.0, 3.0, 4.0}, squareShape,
                         {2.0, 1.0, 2.0, 1.0}, squareShape, ACL_INT16, stream, &stats);
  RunPowTensorTensorCase("PowTensorTensor_int32_tiling_key", {1.0, 2.0, 3.0, 4.0}, squareShape,
                         {2.0, 1.0, 2.0, 1.0}, squareShape, ACL_INT32, stream, &stats);
  RunInplacePowTensorTensorCase("InplacePowTensorTensor_int16", {1.0, 2.0, 3.0, 4.0}, squareShape,
                                {2.0, 1.0, 2.0, 1.0}, ACL_INT16, stream, &stats);
  ExpectPowTensorTensorGetWorkspaceFailure("PowTensorTensor_bool_bool_invalid", {1.0, 0.0, 1.0, 0.0}, squareShape,
                                           ACL_BOOL, {1.0, 1.0, 0.0, 0.0}, squareShape, ACL_BOOL, squareShape, ACL_BOOL,
                                           &stats);
  ExpectPowTensorTensorGetWorkspaceFailure("PowTensorTensor_broadcast_invalid", {1.0, 2.0, 3.0, 4.0}, {2, 2}, ACL_FLOAT,
                                           {1.0, 2.0, 3.0}, {3}, ACL_FLOAT, {2, 2}, ACL_FLOAT, &stats);

  RunExp2Case("Exp2_float32", {0.0, 1.0, 2.0, 3.0}, squareShape, ACL_FLOAT, ACL_FLOAT, stream, &stats);
  RunInplaceExp2Case("InplaceExp2_float16", {0.0, 1.0, 2.0, 3.0}, squareShape, ACL_FLOAT16, stream, &stats);

  LOG_PRINT("Summary: passed=%d failed=%d total=%d\n", stats.passed, stats.failed, stats.passed + stats.failed);

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return stats.failed == 0 ? 0 : 1;
}
