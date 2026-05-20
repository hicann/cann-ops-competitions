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
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"
#include "aclnnop/aclnn_exp2.h"

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

namespace {

struct Tolerance {
  double atol;
  double rtol;
};

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    n *= shape[i];
  }
  return n;
}

std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  return strides;
}

std::string ShapeToString(const std::vector<int64_t>& shape) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

const char* DataTypeName(aclDataType dtype) {
  switch (dtype) {
    case ACL_FLOAT:
      return "FLOAT32";
    case ACL_FLOAT16:
      return "FLOAT16";
    case ACL_BF16:
      return "BF16";
    case ACL_DOUBLE:
      return "DOUBLE";
    case ACL_INT8:
      return "INT8";
    case ACL_UINT8:
      return "UINT8";
    case ACL_INT16:
      return "INT16";
    case ACL_INT32:
      return "INT32";
    case ACL_INT64:
      return "INT64";
    default:
      return "UNKNOWN";
  }
}

size_t GetDataTypeSize(aclDataType dtype) {
  switch (dtype) {
    case ACL_FLOAT:
      return sizeof(float);
    case ACL_FLOAT16:
      return sizeof(uint16_t);
    case ACL_BF16:
      return sizeof(uint16_t);
    case ACL_DOUBLE:
      return sizeof(double);
    case ACL_INT8:
      return sizeof(int8_t);
    case ACL_UINT8:
      return sizeof(uint8_t);
    case ACL_INT16:
      return sizeof(int16_t);
    case ACL_INT32:
      return sizeof(int32_t);
    case ACL_INT64:
      return sizeof(int64_t);
    default:
      return 0;
  }
}

bool IsFloatingType(aclDataType dtype) {
  return dtype == ACL_FLOAT || dtype == ACL_FLOAT16 || dtype == ACL_BF16 || dtype == ACL_DOUBLE;
}

Tolerance GetTolerance(aclDataType dtype) {
  switch (dtype) {
    case ACL_FLOAT:
      return {1e-4, 1e-4};
    case ACL_FLOAT16:
      return {5e-2, 5e-2};
    case ACL_BF16:
      return {8e-2, 8e-2};
    case ACL_DOUBLE:
      return {1e-12, 1e-12};
    default:
      return {0.0, 0.0};
  }
}

uint32_t FloatToBits(float x) {
  uint32_t bits = 0;
  std::memcpy(&bits, &x, sizeof(bits));
  return bits;
}

float BitsToFloat(uint32_t bits) {
  float x = 0.0f;
  std::memcpy(&x, &bits, sizeof(x));
  return x;
}

uint16_t FloatToHalfBits(float value) {
  uint32_t bits = FloatToBits(value);
  uint32_t sign = (bits >> 16) & 0x8000u;
  uint32_t exp = (bits >> 23) & 0xFFu;
  uint32_t mant = bits & 0x7FFFFFu;

  if (exp == 0xFFu) {
    if (mant == 0) {
      return static_cast<uint16_t>(sign | 0x7C00u);
    }
    return static_cast<uint16_t>(sign | 0x7E00u);
  }

  int32_t newExp = static_cast<int32_t>(exp) - 127 + 15;
  if (newExp >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00u);
  }

  if (newExp <= 0) {
    if (newExp < -10) {
      return static_cast<uint16_t>(sign);
    }
    mant |= 0x00800000u;
    uint32_t shift = static_cast<uint32_t>(14 - newExp);
    uint32_t halfMant = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1u);
    uint32_t halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (halfMant & 1u))) {
      ++halfMant;
    }
    return static_cast<uint16_t>(sign | halfMant);
  }

  uint32_t halfExp = static_cast<uint32_t>(newExp) << 10;
  uint32_t halfMant = mant >> 13;
  uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (halfMant & 1u))) {
    ++halfMant;
    if (halfMant == 0x400u) {
      halfMant = 0;
      halfExp += 0x400u;
      if (halfExp >= 0x7C00u) {
        return static_cast<uint16_t>(sign | 0x7C00u);
      }
    }
  }
  return static_cast<uint16_t>(sign | halfExp | halfMant);
}

float HalfBitsToFloat(uint16_t h) {
  uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x03FFu;
  uint32_t bits = 0;

  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      int e = -14;
      while ((mant & 0x0400u) == 0) {
        mant <<= 1;
        --e;
      }
      mant &= 0x03FFu;
      uint32_t exp32 = static_cast<uint32_t>(e + 127);
      bits = sign | (exp32 << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    uint32_t exp32 = exp + (127 - 15);
    bits = sign | (exp32 << 23) | (mant << 13);
  }

  return BitsToFloat(bits);
}

uint16_t FloatToBf16Bits(float value) {
  uint32_t bits = FloatToBits(value);
  uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7FFFu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

float Bf16BitsToFloat(uint16_t bf16) {
  uint32_t bits = static_cast<uint32_t>(bf16) << 16;
  return BitsToFloat(bits);
}

std::vector<uint8_t> EncodeValues(const std::vector<double>& values, aclDataType dtype) {
  const size_t elemSize = GetDataTypeSize(dtype);
  std::vector<uint8_t> bytes(values.size() * elemSize, 0);

  for (size_t i = 0; i < values.size(); ++i) {
    uint8_t* dst = bytes.data() + i * elemSize;
    switch (dtype) {
      case ACL_FLOAT: {
        float v = static_cast<float>(values[i]);
        std::memcpy(dst, &v, sizeof(v));
        break;
      }
      case ACL_FLOAT16: {
        uint16_t v = FloatToHalfBits(static_cast<float>(values[i]));
        std::memcpy(dst, &v, sizeof(v));
        break;
      }
      case ACL_BF16: {
        uint16_t v = FloatToBf16Bits(static_cast<float>(values[i]));
        std::memcpy(dst, &v, sizeof(v));
        break;
      }
      case ACL_DOUBLE: {
        double v = values[i];
        std::memcpy(dst, &v, sizeof(v));
        break;
      }
      case ACL_INT8: {
        int8_t v = static_cast<int8_t>(values[i]);
        std::memcpy(dst, &v, sizeof(v));
        break;
      }
      case ACL_UINT8: {
        uint8_t v = static_cast<uint8_t>(values[i]);
        std::memcpy(dst, &v, sizeof(v));
        break;
      }
      case ACL_INT16: {
        int16_t v = static_cast<int16_t>(values[i]);
        std::memcpy(dst, &v, sizeof(v));
        break;
      }
      case ACL_INT32: {
        int32_t v = static_cast<int32_t>(values[i]);
        std::memcpy(dst, &v, sizeof(v));
        break;
      }
      case ACL_INT64: {
        int64_t v = static_cast<int64_t>(values[i]);
        std::memcpy(dst, &v, sizeof(v));
        break;
      }
      default:
        break;
    }
  }
  return bytes;
}

std::vector<double> DecodeValues(const uint8_t* data, size_t count, aclDataType dtype) {
  const size_t elemSize = GetDataTypeSize(dtype);
  std::vector<double> values(count, 0.0);

  for (size_t i = 0; i < count; ++i) {
    const uint8_t* src = data + i * elemSize;
    switch (dtype) {
      case ACL_FLOAT: {
        float v = 0.0f;
        std::memcpy(&v, src, sizeof(v));
        values[i] = static_cast<double>(v);
        break;
      }
      case ACL_FLOAT16: {
        uint16_t v = 0;
        std::memcpy(&v, src, sizeof(v));
        values[i] = static_cast<double>(HalfBitsToFloat(v));
        break;
      }
      case ACL_BF16: {
        uint16_t v = 0;
        std::memcpy(&v, src, sizeof(v));
        values[i] = static_cast<double>(Bf16BitsToFloat(v));
        break;
      }
      case ACL_DOUBLE: {
        double v = 0.0;
        std::memcpy(&v, src, sizeof(v));
        values[i] = v;
        break;
      }
      case ACL_INT8: {
        int8_t v = 0;
        std::memcpy(&v, src, sizeof(v));
        values[i] = static_cast<double>(v);
        break;
      }
      case ACL_UINT8: {
        uint8_t v = 0;
        std::memcpy(&v, src, sizeof(v));
        values[i] = static_cast<double>(v);
        break;
      }
      case ACL_INT16: {
        int16_t v = 0;
        std::memcpy(&v, src, sizeof(v));
        values[i] = static_cast<double>(v);
        break;
      }
      case ACL_INT32: {
        int32_t v = 0;
        std::memcpy(&v, src, sizeof(v));
        values[i] = static_cast<double>(v);
        break;
      }
      case ACL_INT64: {
        int64_t v = 0;
        std::memcpy(&v, src, sizeof(v));
        values[i] = static_cast<double>(v);
        break;
      }
      default:
        break;
    }
  }
  return values;
}

double QuantizeValueToDType(double value, aclDataType dtype) {
  std::vector<double> in(1, value);
  std::vector<uint8_t> bytes = EncodeValues(in, dtype);
  std::vector<double> out = DecodeValues(bytes.data(), 1, dtype);
  return out[0];
}

std::vector<double> QuantizeValuesToDType(const std::vector<double>& values, aclDataType dtype) {
  std::vector<uint8_t> bytes = EncodeValues(values, dtype);
  return DecodeValues(bytes.data(), values.size(), dtype);
}

double CastResultToOutputType(double value, aclDataType outDtype) {
  return QuantizeValueToDType(value, outDtype);
}

struct DeviceTensor {
  void* deviceAddr = nullptr;
  aclTensor* tensor = nullptr;
  std::vector<int64_t> shape;
  aclDataType dtype = ACL_FLOAT;
  size_t bytes = 0;

  ~DeviceTensor() {
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

struct DeviceScalar {
  aclScalar* scalar = nullptr;
  aclDataType dtype = ACL_FLOAT;
  union {
    uint8_t u8;
    int8_t i8;
    int16_t i16;
    int32_t i32;
    int64_t i64;
    float f32;
    double f64;
    uint16_t u16;
  } storage;

  DeviceScalar() { std::memset(&storage, 0, sizeof(storage)); }

  ~DeviceScalar() {
    if (scalar != nullptr) {
      aclDestroyScalar(scalar);
      scalar = nullptr;
    }
  }
};

struct WorkspaceHolder {
  void* addr = nullptr;
  uint64_t size = 0;

  ~WorkspaceHolder() {
    if (addr != nullptr) {
      aclrtFree(addr);
      addr = nullptr;
    }
  }
};

bool PrintCaseResult(const std::string& name, bool pass, const std::string& msg = "") {
  if (pass) {
    LOG_PRINT("[PASS] %s\n", name.c_str());
  } else {
    LOG_PRINT("[FAIL] %s", name.c_str());
    if (!msg.empty()) {
      LOG_PRINT(" : %s", msg.c_str());
    }
    LOG_PRINT("\n");
  }
  return pass;
}

bool InitAcl(int32_t deviceId, aclrtStream* stream, std::string* err) {
  auto ret = aclInit(nullptr);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "aclInit failed, ret=" << static_cast<int>(ret);
    *err = oss.str();
    return false;
  }

  ret = aclrtSetDevice(deviceId);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "aclrtSetDevice failed, ret=" << static_cast<int>(ret);
    *err = oss.str();
    return false;
  }

  ret = aclrtCreateStream(stream);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "aclrtCreateStream failed, ret=" << static_cast<int>(ret);
    *err = oss.str();
    return false;
  }
  return true;
}

bool CreateTensorWithFormat(const std::vector<double>& hostValues,
                            const std::vector<int64_t>& shape,
                            aclDataType dtype,
                            aclFormat format,
                            DeviceTensor* out,
                            std::string* err) {
  const int64_t numel = GetShapeSize(shape);
  if (numel < 0 || static_cast<int64_t>(hostValues.size()) != numel) {
    std::ostringstream oss;
    oss << "hostValues.size()=" << hostValues.size()
        << " mismatch shape=" << ShapeToString(shape)
        << ", numel=" << numel;
    *err = oss.str();
    return false;
  }

  const size_t elemSize = GetDataTypeSize(dtype);
  if (elemSize == 0) {
    std::ostringstream oss;
    oss << "unsupported dtype in helper: " << DataTypeName(dtype);
    *err = oss.str();
    return false;
  }

  out->shape = shape;
  out->dtype = dtype;
  out->bytes = static_cast<size_t>(std::max<int64_t>(numel, 0)) * elemSize;

  std::vector<uint8_t> hostBytes = EncodeValues(hostValues, dtype);
  size_t allocBytes = std::max<size_t>(out->bytes, 1);

  auto ret = aclrtMalloc(&out->deviceAddr, allocBytes, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "aclrtMalloc failed, ret=" << static_cast<int>(ret);
    *err = oss.str();
    return false;
  }

  if (out->bytes > 0) {
    ret = aclrtMemcpy(out->deviceAddr, out->bytes, hostBytes.data(), out->bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
      std::ostringstream oss;
      oss << "aclrtMemcpy(H2D) failed, ret=" << static_cast<int>(ret);
      *err = oss.str();
      return false;
    }
  }

  std::vector<int64_t> strides = MakeStrides(shape);
  out->tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, format,
                                shape.data(), shape.size(), out->deviceAddr);
  if (out->tensor == nullptr) {
    *err = "aclCreateTensor failed";
    return false;
  }
  return true;
}

bool CreateTensorFromHostValues(const std::vector<double>& hostValues,
                                const std::vector<int64_t>& shape,
                                aclDataType dtype,
                                DeviceTensor* out,
                                std::string* err) {
  return CreateTensorWithFormat(hostValues, shape, dtype, ACL_FORMAT_ND, out, err);
}

bool CreateZeroTensorWithFormat(const std::vector<int64_t>& shape,
                                aclDataType dtype,
                                aclFormat format,
                                DeviceTensor* out,
                                std::string* err) {
  std::vector<double> zeros(static_cast<size_t>(std::max<int64_t>(GetShapeSize(shape), 0)), 0.0);
  return CreateTensorWithFormat(zeros, shape, dtype, format, out, err);
}

bool CreateZeroTensor(const std::vector<int64_t>& shape,
                      aclDataType dtype,
                      DeviceTensor* out,
                      std::string* err) {
  return CreateZeroTensorWithFormat(shape, dtype, ACL_FORMAT_ND, out, err);
}

bool CreateScalarFromDouble(double value, aclDataType dtype, DeviceScalar* out, std::string* err) {
  out->dtype = dtype;
  void* ptr = nullptr;

  switch (dtype) {
    case ACL_FLOAT:
      out->storage.f32 = static_cast<float>(value);
      ptr = &out->storage.f32;
      break;
    case ACL_FLOAT16:
      out->storage.u16 = FloatToHalfBits(static_cast<float>(value));
      ptr = &out->storage.u16;
      break;
    case ACL_BF16:
      out->storage.u16 = FloatToBf16Bits(static_cast<float>(value));
      ptr = &out->storage.u16;
      break;
    case ACL_DOUBLE:
      out->storage.f64 = value;
      ptr = &out->storage.f64;
      break;
    case ACL_INT8:
      out->storage.i8 = static_cast<int8_t>(value);
      ptr = &out->storage.i8;
      break;
    case ACL_UINT8:
      out->storage.u8 = static_cast<uint8_t>(value);
      ptr = &out->storage.u8;
      break;
    case ACL_INT16:
      out->storage.i16 = static_cast<int16_t>(value);
      ptr = &out->storage.i16;
      break;
    case ACL_INT32:
      out->storage.i32 = static_cast<int32_t>(value);
      ptr = &out->storage.i32;
      break;
    case ACL_INT64:
      out->storage.i64 = static_cast<int64_t>(value);
      ptr = &out->storage.i64;
      break;
    default:
      *err = "unsupported scalar dtype in helper";
      return false;
  }

  out->scalar = aclCreateScalar(ptr, dtype);
  if (out->scalar == nullptr) {
    *err = "aclCreateScalar failed";
    return false;
  }
  return true;
}

bool ReadTensorToHostValues(const DeviceTensor& tensor, std::vector<double>* hostValues, std::string* err) {
  const int64_t numel = GetShapeSize(tensor.shape);
  if (numel < 0) {
    *err = "invalid negative numel";
    return false;
  }
  if (numel == 0) {
    hostValues->clear();
    return true;
  }

  std::vector<uint8_t> hostBytes(tensor.bytes, 0);
  auto ret = aclrtMemcpy(hostBytes.data(), tensor.bytes, tensor.deviceAddr, tensor.bytes, ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "aclrtMemcpy(D2H) failed, ret=" << static_cast<int>(ret);
    *err = oss.str();
    return false;
  }

  *hostValues = DecodeValues(hostBytes.data(), static_cast<size_t>(numel), tensor.dtype);
  return true;
}

bool AllocWorkspace(uint64_t size, WorkspaceHolder* ws, std::string* err) {
  ws->size = size;
  if (size == 0) {
    return true;
  }
  auto ret = aclrtMalloc(&ws->addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "workspace aclrtMalloc failed, ret=" << static_cast<int>(ret)
        << ", size=" << static_cast<unsigned long long>(size);
    *err = oss.str();
    return false;
  }
  return true;
}

bool ComputeBroadcastShape(const std::vector<int64_t>& a,
                           const std::vector<int64_t>& b,
                           std::vector<int64_t>* out) {
  const size_t rank = std::max(a.size(), b.size());
  out->assign(rank, 1);

  for (size_t i = 0; i < rank; ++i) {
    int64_t da = 1;
    int64_t db = 1;
    if (i >= rank - a.size()) {
      da = a[i - (rank - a.size())];
    }
    if (i >= rank - b.size()) {
      db = b[i - (rank - b.size())];
    }
    if (da != db && da != 1 && db != 1) {
      return false;
    }
    (*out)[i] = std::max(da, db);
  }
  return true;
}

size_t GetBroadcastOffset(size_t linearIndex,
                          const std::vector<int64_t>& outShape,
                          const std::vector<int64_t>& inShape) {
  std::vector<int64_t> coords(outShape.size(), 0);
  size_t tmp = linearIndex;
  for (int64_t i = static_cast<int64_t>(outShape.size()) - 1; i >= 0; --i) {
    coords[static_cast<size_t>(i)] = static_cast<int64_t>(tmp % static_cast<size_t>(outShape[static_cast<size_t>(i)]));
    tmp /= static_cast<size_t>(outShape[static_cast<size_t>(i)]);
  }

  std::vector<int64_t> strides = MakeStrides(inShape);
  const size_t shift = outShape.size() - inShape.size();
  size_t offset = 0;
  for (size_t i = 0; i < inShape.size(); ++i) {
    int64_t c = coords[i + shift];
    if (inShape[i] == 1) {
      c = 0;
    }
    offset += static_cast<size_t>(c * strides[i]);
  }
  return offset;
}

std::vector<double> BuildExpectedPowTensorScalar(const std::vector<double>& selfValues,
                                                 double exponent,
                                                 aclDataType outDtype) {
  std::vector<double> expected(selfValues.size(), 0.0);
  for (size_t i = 0; i < selfValues.size(); ++i) {
    expected[i] = CastResultToOutputType(std::pow(selfValues[i], exponent), outDtype);
  }
  return expected;
}

std::vector<double> BuildExpectedPowScalarTensor(double base,
                                                 const std::vector<double>& exponentValues,
                                                 aclDataType outDtype) {
  std::vector<double> expected(exponentValues.size(), 0.0);
  for (size_t i = 0; i < exponentValues.size(); ++i) {
    expected[i] = CastResultToOutputType(std::pow(base, exponentValues[i]), outDtype);
  }
  return expected;
}

std::vector<double> BuildExpectedPowTensorTensor(const std::vector<double>& selfValues,
                                                 const std::vector<int64_t>& selfShape,
                                                 const std::vector<double>& exponentValues,
                                                 const std::vector<int64_t>& exponentShape,
                                                 const std::vector<int64_t>& outShape,
                                                 aclDataType outDtype) {
  const size_t outNum = static_cast<size_t>(GetShapeSize(outShape));
  std::vector<double> expected(outNum, 0.0);
  for (size_t i = 0; i < outNum; ++i) {
    const size_t selfOff = GetBroadcastOffset(i, outShape, selfShape);
    const size_t expOff = GetBroadcastOffset(i, outShape, exponentShape);
    expected[i] = CastResultToOutputType(std::pow(selfValues[selfOff], exponentValues[expOff]), outDtype);
  }
  return expected;
}

std::vector<double> BuildExpectedExp2(const std::vector<double>& selfValues, aclDataType outDtype) {
  std::vector<double> expected(selfValues.size(), 0.0);
  for (size_t i = 0; i < selfValues.size(); ++i) {
    expected[i] = CastResultToOutputType(std::pow(2.0, selfValues[i]), outDtype);
  }
  return expected;
}

bool AlmostEqual(double actual, double expected, aclDataType dtype) {
  if (!IsFloatingType(dtype)) {
    return actual == expected;
  }

  if (std::isnan(expected)) {
    return std::isnan(actual);
  }
  if (std::isinf(expected)) {
    return std::isinf(actual) && (std::signbit(expected) == std::signbit(actual));
  }
  if (std::isnan(actual) || std::isinf(actual)) {
    return false;
  }

  Tolerance tol = GetTolerance(dtype);
  double diff = std::fabs(actual - expected);
  double limit = tol.atol + tol.rtol * std::fabs(expected);
  return diff <= limit;
}

bool CompareResults(const std::vector<double>& actual,
                    const std::vector<double>& expected,
                    aclDataType dtype,
                    std::string* err) {
  if (actual.size() != expected.size()) {
    std::ostringstream oss;
    oss << "size mismatch, actual=" << actual.size() << ", expected=" << expected.size();
    *err = oss.str();
    return false;
  }

  std::ostringstream oss;
  oss << std::setprecision(10);
  int mismatchCount = 0;

  for (size_t i = 0; i < actual.size(); ++i) {
    if (!AlmostEqual(actual[i], expected[i], dtype)) {
      if (mismatchCount == 0) {
        oss << "first mismatch at index " << i << ", dtype=" << DataTypeName(dtype)
            << ", actual=" << actual[i] << ", expected=" << expected[i];
      }
      ++mismatchCount;
      if (mismatchCount >= 8) {
        break;
      }
    }
  }

  if (mismatchCount > 0) {
    *err = oss.str();
    return false;
  }
  return true;
}

bool ExpectNonSuccess(const std::string& name, int ret) {
  if (ret == ACL_SUCCESS) {
    return PrintCaseResult(name, false, "expected failure, but ret == ACL_SUCCESS");
  }
  return PrintCaseResult(name, true);
}

std::vector<double> RepeatPattern(const std::vector<double>& pattern, size_t total) {
  std::vector<double> out(total, 0.0);
  if (pattern.empty()) {
    return out;
  }
  for (size_t i = 0; i < total; ++i) {
    out[i] = pattern[i % pattern.size()];
  }
  return out;
}

std::vector<double> MakeRangeDouble(int start, int endInclusive) {
  std::vector<double> out;
  for (int v = start; v <= endInclusive; ++v) {
    out.push_back(static_cast<double>(v));
  }
  return out;
}

bool RunPowTensorScalarCase(aclrtStream stream,
                            const std::string& name,
                            aclDataType tensorDtype,
                            const std::vector<int64_t>& selfShape,
                            const std::vector<double>& selfHostValues,
                            double exponentValue,
                            aclDataType scalarDtype,
                            bool inplace) {
  DeviceTensor selfTensor;
  DeviceTensor outTensor;
  DeviceScalar exponent;
  WorkspaceHolder workspace;
  std::string err;

  if (!CreateTensorFromHostValues(selfHostValues, selfShape, tensorDtype, &selfTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateScalarFromDouble(exponentValue, scalarDtype, &exponent, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!inplace && !CreateZeroTensor(selfShape, tensorDtype, &outTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = inplace
                 ? aclnnInplacePowTensorScalarGetWorkspaceSize(selfTensor.tensor, exponent.scalar, &workspaceSize,
                                                               &executor)
                 : aclnnPowTensorScalarGetWorkspaceSize(selfTensor.tensor, exponent.scalar, outTensor.tensor,
                                                        &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "GetWorkspaceSize failed, ret=" << static_cast<int>(ret);
    return PrintCaseResult(name, false, oss.str());
  }

  if (!AllocWorkspace(workspaceSize, &workspace, &err)) {
    return PrintCaseResult(name, false, err);
  }

  ret = inplace ? aclnnInplacePowTensorScalar(workspace.addr, workspaceSize, executor, stream)
                : aclnnPowTensorScalar(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "execute failed, ret=" << static_cast<int>(ret);
    return PrintCaseResult(name, false, oss.str());
  }

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "aclrtSynchronizeStream failed, ret=" << static_cast<int>(ret);
    return PrintCaseResult(name, false, oss.str());
  }

  std::vector<double> actual;
  if (!ReadTensorToHostValues(inplace ? selfTensor : outTensor, &actual, &err)) {
    return PrintCaseResult(name, false, err);
  }

  const std::vector<double> qSelf = QuantizeValuesToDType(selfHostValues, tensorDtype);
  const double qExp = QuantizeValueToDType(exponentValue, scalarDtype);
  const std::vector<double> expected = BuildExpectedPowTensorScalar(qSelf, qExp, tensorDtype);

  if (!CompareResults(actual, expected, tensorDtype, &err)) {
    return PrintCaseResult(name, false, err);
  }
  return PrintCaseResult(name, true);
}

bool RunPowScalarTensorCase(aclrtStream stream,
                            const std::string& name,
                            aclDataType tensorDtype,
                            const std::vector<int64_t>& exponentShape,
                            const std::vector<double>& exponentHostValues,
                            double baseScalarValue,
                            aclDataType scalarDtype) {
  DeviceTensor exponentTensor;
  DeviceTensor outTensor;
  DeviceScalar baseScalar;
  WorkspaceHolder workspace;
  std::string err;

  if (!CreateTensorFromHostValues(exponentHostValues, exponentShape, tensorDtype, &exponentTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateScalarFromDouble(baseScalarValue, scalarDtype, &baseScalar, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateZeroTensor(exponentShape, tensorDtype, &outTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnPowScalarTensorGetWorkspaceSize(baseScalar.scalar, exponentTensor.tensor, outTensor.tensor,
                                                  &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "GetWorkspaceSize failed, ret=" << static_cast<int>(ret);
    return PrintCaseResult(name, false, oss.str());
  }

  if (!AllocWorkspace(workspaceSize, &workspace, &err)) {
    return PrintCaseResult(name, false, err);
  }

  ret = aclnnPowScalarTensor(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "execute failed, ret=" << static_cast<int>(ret);
    return PrintCaseResult(name, false, oss.str());
  }

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "aclrtSynchronizeStream failed, ret=" << static_cast<int>(ret);
    return PrintCaseResult(name, false, oss.str());
  }

  std::vector<double> actual;
  if (!ReadTensorToHostValues(outTensor, &actual, &err)) {
    return PrintCaseResult(name, false, err);
  }

  const std::vector<double> qExp = QuantizeValuesToDType(exponentHostValues, tensorDtype);
  const double qBase = QuantizeValueToDType(baseScalarValue, scalarDtype);
  const std::vector<double> expected = BuildExpectedPowScalarTensor(qBase, qExp, tensorDtype);

  if (!CompareResults(actual, expected, tensorDtype, &err)) {
    return PrintCaseResult(name, false, err);
  }
  return PrintCaseResult(name, true);
}

bool RunPowTensorScalarCaseWithFormat(aclrtStream stream,
                                      const std::string& name,
                                      aclDataType tensorDtype,
                                      const std::vector<int64_t>& selfShape,
                                      const std::vector<double>& selfHostValues,
                                      double exponentValue,
                                      aclDataType scalarDtype,
                                      aclFormat selfFormat,
                                      aclFormat outFormat) {
  DeviceTensor selfTensor;
  DeviceTensor outTensor;
  DeviceScalar exponent;
  WorkspaceHolder workspace;
  std::string err;

  if (!CreateTensorWithFormat(selfHostValues, selfShape, tensorDtype, selfFormat, &selfTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateScalarFromDouble(exponentValue, scalarDtype, &exponent, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateZeroTensorWithFormat(selfShape, tensorDtype, outFormat, &outTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnPowTensorScalarGetWorkspaceSize(selfTensor.tensor, exponent.scalar, outTensor.tensor,
                                                  &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "GetWorkspaceSize failed, ret=" << static_cast<int>(ret);
    return PrintCaseResult(name, false, oss.str());
  }

  if (!AllocWorkspace(workspaceSize, &workspace, &err)) {
    return PrintCaseResult(name, false, err);
  }

  ret = aclnnPowTensorScalar(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "execute failed, ret=" << static_cast<int>(ret);
    return PrintCaseResult(name, false, oss.str());
  }

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "aclrtSynchronizeStream failed, ret=" << static_cast<int>(ret);
    return PrintCaseResult(name, false, oss.str());
  }

  std::vector<double> actual;
  if (!ReadTensorToHostValues(outTensor, &actual, &err)) {
    return PrintCaseResult(name, false, err);
  }

  const std::vector<double> qSelf = QuantizeValuesToDType(selfHostValues, tensorDtype);
  const double qExp = QuantizeValueToDType(exponentValue, scalarDtype);
  const std::vector<double> expected = BuildExpectedPowTensorScalar(qSelf, qExp, tensorDtype);

  if (!CompareResults(actual, expected, tensorDtype, &err)) {
    return PrintCaseResult(name, false, err);
  }
  return PrintCaseResult(name, true);
}

bool RunPowScalarTensorCaseWithFormat(aclrtStream stream,
                                      const std::string& name,
                                      aclDataType tensorDtype,
                                      const std::vector<int64_t>& exponentShape,
                                      const std::vector<double>& exponentHostValues,
                                      double baseScalarValue,
                                      aclDataType scalarDtype,
                                      aclFormat exponentFormat,
                                      aclFormat outFormat) {
  DeviceTensor exponentTensor;
  DeviceTensor outTensor;
  DeviceScalar baseScalar;
  WorkspaceHolder workspace;
  std::string err;

  if (!CreateTensorWithFormat(exponentHostValues, exponentShape, tensorDtype, exponentFormat, &exponentTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateScalarFromDouble(baseScalarValue, scalarDtype, &baseScalar, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateZeroTensorWithFormat(exponentShape, tensorDtype, outFormat, &outTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnPowScalarTensorGetWorkspaceSize(baseScalar.scalar, exponentTensor.tensor, outTensor.tensor,
                                                  &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "GetWorkspaceSize failed, ret=" << static_cast<int>(ret);
    return PrintCaseResult(name, false, oss.str());
  }

  if (!AllocWorkspace(workspaceSize, &workspace, &err)) {
    return PrintCaseResult(name, false, err);
  }

  ret = aclnnPowScalarTensor(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "execute failed, ret=" << static_cast<int>(ret);
    return PrintCaseResult(name, false, oss.str());
  }

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "aclrtSynchronizeStream failed, ret=" << static_cast<int>(ret);
    return PrintCaseResult(name, false, oss.str());
  }

  std::vector<double> actual;
  if (!ReadTensorToHostValues(outTensor, &actual, &err)) {
    return PrintCaseResult(name, false, err);
  }

  const std::vector<double> qExp = QuantizeValuesToDType(exponentHostValues, tensorDtype);
  const double qBase = QuantizeValueToDType(baseScalarValue, scalarDtype);
  const std::vector<double> expected = BuildExpectedPowScalarTensor(qBase, qExp, tensorDtype);

  if (!CompareResults(actual, expected, tensorDtype, &err)) {
    return PrintCaseResult(name, false, err);
  }
  return PrintCaseResult(name, true);
}

bool RunPowTensorTensorCase(aclrtStream stream,
                            const std::string& name,
                            aclDataType tensorDtype,
                            const std::vector<int64_t>& selfShape,
                            const std::vector<double>& selfHostValues,
                            const std::vector<int64_t>& exponentShape,
                            const std::vector<double>& exponentHostValues,
                            bool inplace) {
  std::vector<int64_t> outShape;
  if (!ComputeBroadcastShape(selfShape, exponentShape, &outShape)) {
    return PrintCaseResult(name, false, "host-side broadcast shape inference failed");
  }
  if (inplace && outShape != selfShape) {
    return PrintCaseResult(name, false, "inplace case requires broadcast result shape == self shape");
  }

  DeviceTensor selfTensor;
  DeviceTensor exponentTensor;
  DeviceTensor outTensor;
  WorkspaceHolder workspace;
  std::string err;

  if (!CreateTensorFromHostValues(selfHostValues, selfShape, tensorDtype, &selfTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateTensorFromHostValues(exponentHostValues, exponentShape, tensorDtype, &exponentTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!inplace && !CreateZeroTensor(outShape, tensorDtype, &outTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = inplace ? aclnnInplacePowTensorTensorGetWorkspaceSize(selfTensor.tensor, exponentTensor.tensor,
                                                                   &workspaceSize, &executor)
                     : aclnnPowTensorTensorGetWorkspaceSize(selfTensor.tensor, exponentTensor.tensor, outTensor.tensor,
                                                           &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "GetWorkspaceSize failed, ret=" << static_cast<int>(ret);
    return PrintCaseResult(name, false, oss.str());
  }

  if (!AllocWorkspace(workspaceSize, &workspace, &err)) {
    return PrintCaseResult(name, false, err);
  }

  ret = inplace ? aclnnInplacePowTensorTensor(workspace.addr, workspaceSize, executor, stream)
                : aclnnPowTensorTensor(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "execute failed, ret=" << static_cast<int>(ret);
    return PrintCaseResult(name, false, oss.str());
  }

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "aclrtSynchronizeStream failed, ret=" << static_cast<int>(ret);
    return PrintCaseResult(name, false, oss.str());
  }

  std::vector<double> actual;
  if (!ReadTensorToHostValues(inplace ? selfTensor : outTensor, &actual, &err)) {
    return PrintCaseResult(name, false, err);
  }

  const std::vector<double> qSelf = QuantizeValuesToDType(selfHostValues, tensorDtype);
  const std::vector<double> qExp = QuantizeValuesToDType(exponentHostValues, tensorDtype);
  const std::vector<double> expected =
      BuildExpectedPowTensorTensor(qSelf, selfShape, qExp, exponentShape, outShape, tensorDtype);

  if (!CompareResults(actual, expected, tensorDtype, &err)) {
    return PrintCaseResult(name, false, err);
  }
  return PrintCaseResult(name, true);
}

bool RunExp2Case(aclrtStream stream,
                 const std::string& name,
                 aclDataType tensorDtype,
                 const std::vector<int64_t>& selfShape,
                 const std::vector<double>& selfHostValues,
                 bool inplace) {
  DeviceTensor selfTensor;
  DeviceTensor outTensor;
  WorkspaceHolder workspace;
  std::string err;

  if (!CreateTensorFromHostValues(selfHostValues, selfShape, tensorDtype, &selfTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!inplace && !CreateZeroTensor(selfShape, tensorDtype, &outTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = inplace ? aclnnInplaceExp2GetWorkspaceSize(selfTensor.tensor, &workspaceSize, &executor)
                     : aclnnExp2GetWorkspaceSize(selfTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "GetWorkspaceSize failed, ret=" << static_cast<int>(ret);
    return PrintCaseResult(name, false, oss.str());
  }

  if (!AllocWorkspace(workspaceSize, &workspace, &err)) {
    return PrintCaseResult(name, false, err);
  }

  ret = inplace ? aclnnInplaceExp2(workspace.addr, workspaceSize, executor, stream)
                : aclnnExp2(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "execute failed, ret=" << static_cast<int>(ret);
    return PrintCaseResult(name, false, oss.str());
  }

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    std::ostringstream oss;
    oss << "aclrtSynchronizeStream failed, ret=" << static_cast<int>(ret);
    return PrintCaseResult(name, false, oss.str());
  }

  std::vector<double> actual;
  if (!ReadTensorToHostValues(inplace ? selfTensor : outTensor, &actual, &err)) {
    return PrintCaseResult(name, false, err);
  }

  const std::vector<double> qSelf = QuantizeValuesToDType(selfHostValues, tensorDtype);
  const std::vector<double> expected = BuildExpectedExp2(qSelf, tensorDtype);

  if (!CompareResults(actual, expected, tensorDtype, &err)) {
    return PrintCaseResult(name, false, err);
  }
  return PrintCaseResult(name, true);
}

// -------------------- negative cases --------------------

bool NegativePowTensorScalarNullSelf(const std::string& name) {
  DeviceTensor outTensor;
  DeviceScalar exponent;
  std::string err;

  if (!CreateZeroTensor({2, 2}, ACL_FLOAT, &outTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateScalarFromDouble(2.0, ACL_FLOAT, &exponent, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnPowTensorScalarGetWorkspaceSize(nullptr, exponent.scalar, outTensor.tensor, &workspaceSize, &executor);
  return ExpectNonSuccess(name, static_cast<int>(ret));
}

bool NegativePowTensorScalarNullExponent(const std::string& name) {
  DeviceTensor selfTensor;
  DeviceTensor outTensor;
  std::string err;

  if (!CreateTensorFromHostValues({1, 2, 3, 4}, {2, 2}, ACL_FLOAT, &selfTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateZeroTensor({2, 2}, ACL_FLOAT, &outTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnPowTensorScalarGetWorkspaceSize(selfTensor.tensor, nullptr, outTensor.tensor,
                                                  &workspaceSize, &executor);
  return ExpectNonSuccess(name, static_cast<int>(ret));
}

bool NegativePowTensorScalarNullOut(const std::string& name) {
  DeviceTensor selfTensor;
  DeviceScalar exponent;
  std::string err;

  if (!CreateTensorFromHostValues({1, 2, 3, 4}, {2, 2}, ACL_FLOAT, &selfTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateScalarFromDouble(2.0, ACL_FLOAT, &exponent, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnPowTensorScalarGetWorkspaceSize(selfTensor.tensor, exponent.scalar, nullptr,
                                                  &workspaceSize, &executor);
  return ExpectNonSuccess(name, static_cast<int>(ret));
}

bool NegativeInplacePowTensorScalarNullSelf(const std::string& name) {
  DeviceScalar exponent;
  std::string err;

  if (!CreateScalarFromDouble(2.0, ACL_FLOAT, &exponent, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnInplacePowTensorScalarGetWorkspaceSize(nullptr, exponent.scalar, &workspaceSize, &executor);
  return ExpectNonSuccess(name, static_cast<int>(ret));
}

bool NegativeInplacePowTensorScalarNullExponent(const std::string& name) {
  DeviceTensor selfTensor;
  std::string err;

  if (!CreateTensorFromHostValues({1, 2, 3, 4}, {2, 2}, ACL_FLOAT, &selfTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnInplacePowTensorScalarGetWorkspaceSize(selfTensor.tensor, nullptr, &workspaceSize, &executor);
  return ExpectNonSuccess(name, static_cast<int>(ret));
}

bool NegativePowTensorScalarOutputShapeMismatch(const std::string& name) {
  DeviceTensor selfTensor;
  DeviceTensor outTensor;
  DeviceScalar exponent;
  std::string err;

  if (!CreateTensorFromHostValues({1, 2, 3, 4}, {2, 2}, ACL_FLOAT, &selfTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateZeroTensor({2, 3}, ACL_FLOAT, &outTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateScalarFromDouble(2.0, ACL_FLOAT, &exponent, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret =
      aclnnPowTensorScalarGetWorkspaceSize(selfTensor.tensor, exponent.scalar, outTensor.tensor, &workspaceSize, &executor);
  return ExpectNonSuccess(name, static_cast<int>(ret));
}

bool NegativePowTensorScalarOutputDtypeMismatch(const std::string& name) {
  DeviceTensor selfTensor;
  DeviceTensor outTensor;
  DeviceScalar exponent;
  std::string err;

  if (!CreateTensorFromHostValues({1, 2, 3, 4}, {2, 2}, ACL_FLOAT, &selfTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateZeroTensor({2, 2}, ACL_INT32, &outTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateScalarFromDouble(2.0, ACL_FLOAT, &exponent, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret =
      aclnnPowTensorScalarGetWorkspaceSize(selfTensor.tensor, exponent.scalar, outTensor.tensor,
                                           &workspaceSize, &executor);
  return ExpectNonSuccess(name, static_cast<int>(ret));
}

bool NegativePowTensorScalarUnsupportedDouble(const std::string& name) {
  DeviceTensor selfTensor;
  DeviceTensor outTensor;
  DeviceScalar exponent;
  std::string err;

  if (!CreateTensorFromHostValues({1, 2, 3, 4}, {2, 2}, ACL_DOUBLE, &selfTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateZeroTensor({2, 2}, ACL_DOUBLE, &outTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateScalarFromDouble(2.0, ACL_FLOAT, &exponent, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret =
      aclnnPowTensorScalarGetWorkspaceSize(selfTensor.tensor, exponent.scalar, outTensor.tensor, &workspaceSize, &executor);
  return ExpectNonSuccess(name, static_cast<int>(ret));
}

bool NegativePowTensorScalarFormatMismatch(const std::string& name) {
  DeviceTensor selfTensor;
  DeviceTensor outTensor;
  DeviceScalar exponent;
  std::string err;

  std::vector<double> data(12, 1.0);
  if (!CreateTensorWithFormat(data, {1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCHW, &selfTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateZeroTensorWithFormat({1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND, &outTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateScalarFromDouble(2.0, ACL_FLOAT, &exponent, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnPowTensorScalarGetWorkspaceSize(selfTensor.tensor, exponent.scalar, outTensor.tensor,
                                                  &workspaceSize, &executor);
  return ExpectNonSuccess(name, static_cast<int>(ret));
}

bool NegativePowTensorScalarIntegralBaseNegativeExponent(const std::string& name) {
  DeviceTensor selfTensor;
  DeviceTensor outTensor;
  DeviceScalar exponent;
  std::string err;

  if (!CreateTensorFromHostValues({1, 2, 3, 4}, {2, 2}, ACL_INT32, &selfTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateZeroTensor({2, 2}, ACL_INT32, &outTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateScalarFromDouble(-1.0, ACL_INT32, &exponent, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret =
      aclnnPowTensorScalarGetWorkspaceSize(selfTensor.tensor, exponent.scalar, outTensor.tensor,
                                           &workspaceSize, &executor);
  return ExpectNonSuccess(name, static_cast<int>(ret));
}

bool NegativePowScalarTensorNullScalar(const std::string& name) {
  DeviceTensor exponentTensor;
  DeviceTensor outTensor;
  std::string err;

  if (!CreateTensorFromHostValues({0, 1, 2, 3}, {2, 2}, ACL_FLOAT, &exponentTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateZeroTensor({2, 2}, ACL_FLOAT, &outTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret =
      aclnnPowScalarTensorGetWorkspaceSize(nullptr, exponentTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
  return ExpectNonSuccess(name, static_cast<int>(ret));
}

bool NegativePowScalarTensorNullExponentTensor(const std::string& name) {
  DeviceTensor outTensor;
  DeviceScalar baseScalar;
  std::string err;

  if (!CreateZeroTensor({2, 2}, ACL_FLOAT, &outTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateScalarFromDouble(2.0, ACL_FLOAT, &baseScalar, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret =
      aclnnPowScalarTensorGetWorkspaceSize(baseScalar.scalar, nullptr, outTensor.tensor, &workspaceSize, &executor);
  return ExpectNonSuccess(name, static_cast<int>(ret));
}

bool NegativePowScalarTensorNullOut(const std::string& name) {
  DeviceTensor exponentTensor;
  DeviceScalar baseScalar;
  std::string err;

  if (!CreateTensorFromHostValues({0, 1, 2, 3}, {2, 2}, ACL_FLOAT, &exponentTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateScalarFromDouble(2.0, ACL_FLOAT, &baseScalar, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret =
      aclnnPowScalarTensorGetWorkspaceSize(baseScalar.scalar, exponentTensor.tensor, nullptr, &workspaceSize, &executor);
  return ExpectNonSuccess(name, static_cast<int>(ret));
}

bool NegativePowScalarTensorOutputDtypeMismatch(const std::string& name) {
  DeviceTensor exponentTensor;
  DeviceTensor outTensor;
  DeviceScalar baseScalar;
  std::string err;

  if (!CreateTensorFromHostValues({0, 1, 2, 3}, {2, 2}, ACL_FLOAT, &exponentTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateZeroTensor({2, 2}, ACL_INT32, &outTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateScalarFromDouble(2.0, ACL_FLOAT, &baseScalar, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret =
      aclnnPowScalarTensorGetWorkspaceSize(baseScalar.scalar, exponentTensor.tensor, outTensor.tensor,
                                           &workspaceSize, &executor);
  return ExpectNonSuccess(name, static_cast<int>(ret));
}

bool NegativePowScalarTensorOutputShapeMismatch(const std::string& name) {
  DeviceTensor exponentTensor;
  DeviceTensor outTensor;
  DeviceScalar baseScalar;
  std::string err;

  if (!CreateTensorFromHostValues({0, 1, 2, 3}, {2, 2}, ACL_FLOAT, &exponentTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateZeroTensor({2, 3}, ACL_FLOAT, &outTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateScalarFromDouble(2.0, ACL_FLOAT, &baseScalar, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret =
      aclnnPowScalarTensorGetWorkspaceSize(baseScalar.scalar, exponentTensor.tensor, outTensor.tensor,
                                           &workspaceSize, &executor);
  return ExpectNonSuccess(name, static_cast<int>(ret));
}

bool NegativePowScalarTensorUnsupportedDouble(const std::string& name) {
  DeviceTensor exponentTensor;
  DeviceTensor outTensor;
  DeviceScalar baseScalar;
  std::string err;

  if (!CreateTensorFromHostValues({0, 1, 2, 3}, {2, 2}, ACL_DOUBLE, &exponentTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateZeroTensor({2, 2}, ACL_DOUBLE, &outTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateScalarFromDouble(2.0, ACL_FLOAT, &baseScalar, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret =
      aclnnPowScalarTensorGetWorkspaceSize(baseScalar.scalar, exponentTensor.tensor, outTensor.tensor,
                                           &workspaceSize, &executor);
  return ExpectNonSuccess(name, static_cast<int>(ret));
}

bool NegativePowTensorTensorBroadcastMismatch(const std::string& name) {
  DeviceTensor selfTensor;
  DeviceTensor exponentTensor;
  DeviceTensor outTensor;
  std::string err;

  if (!CreateTensorFromHostValues({1, 2, 3, 4, 5, 6}, {2, 3}, ACL_FLOAT, &selfTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateTensorFromHostValues({1, 2, 3, 4}, {2, 2}, ACL_FLOAT, &exponentTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateZeroTensor({2, 3}, ACL_FLOAT, &outTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnPowTensorTensorGetWorkspaceSize(selfTensor.tensor, exponentTensor.tensor, outTensor.tensor,
                                                  &workspaceSize, &executor);
  return ExpectNonSuccess(name, static_cast<int>(ret));
}

bool NegativePowTensorTensorUnsupportedDouble(const std::string& name) {
  DeviceTensor selfTensor;
  DeviceTensor exponentTensor;
  DeviceTensor outTensor;
  std::string err;

  if (!CreateTensorFromHostValues({1, 2, 3, 4}, {2, 2}, ACL_DOUBLE, &selfTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateTensorFromHostValues({0, 1, 2, 3}, {2, 2}, ACL_DOUBLE, &exponentTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateZeroTensor({2, 2}, ACL_DOUBLE, &outTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnPowTensorTensorGetWorkspaceSize(selfTensor.tensor, exponentTensor.tensor, outTensor.tensor,
                                                  &workspaceSize, &executor);
  return ExpectNonSuccess(name, static_cast<int>(ret));
}

bool NegativePowTensorTensorOutputDtypeMismatch(const std::string& name) {
  DeviceTensor selfTensor;
  DeviceTensor exponentTensor;
  DeviceTensor outTensor;
  std::string err;

  if (!CreateTensorFromHostValues({1, 2, 3, 4}, {2, 2}, ACL_FLOAT, &selfTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateTensorFromHostValues({0, 1, 2, 3}, {2, 2}, ACL_FLOAT, &exponentTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }
  if (!CreateZeroTensor({2, 2}, ACL_INT32, &outTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnPowTensorTensorGetWorkspaceSize(selfTensor.tensor, exponentTensor.tensor, outTensor.tensor,
                                                  &workspaceSize, &executor);
  return ExpectNonSuccess(name, static_cast<int>(ret));
}

bool NegativeExp2NullOut(const std::string& name) {
  DeviceTensor selfTensor;
  std::string err;

  if (!CreateTensorFromHostValues({-1, 0, 1, 2}, {2, 2}, ACL_FLOAT, &selfTensor, &err)) {
    return PrintCaseResult(name, false, err);
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnExp2GetWorkspaceSize(selfTensor.tensor, nullptr, &workspaceSize, &executor);
  return ExpectNonSuccess(name, static_cast<int>(ret));
}

}  // namespace

int main() {
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  std::string err;

  if (!InitAcl(deviceId, &stream, &err)) {
    LOG_PRINT("[FAIL] Init ACL runtime failed: %s\n", err.c_str());
    return 1;
  }

  LOG_PRINT("============================================================\n");
  LOG_PRINT("Pow operator end-to-end test start\n");
  LOG_PRINT("Coverage focus: TensorScalar / ScalarTensor op_api paths\n");
  LOG_PRINT("============================================================\n");

  int passed = 0;
  int failed = 0;
  std::vector<std::string> failedCases;

  auto Record = [&](const std::string& name, bool ok) {
    if (ok) {
      ++passed;
    } else {
      ++failed;
      failedCases.push_back(name);
    }
  };

  const std::vector<double> ts_exp0_input = MakeRangeDouble(-8, 8);
  const std::vector<double> ts_exp1_input = RepeatPattern({-3.5, -1.0, 0.0, 0.5, 2.0, 7.25}, 18);
  const std::vector<double> ts_exphalf_input = {0.0, 1.0, 4.0, 9.0, 16.0, 0.25, 2.25, -1.0, -4.0};
  const std::vector<double> ts_exp2_input = RepeatPattern({-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0}, 35);
  const std::vector<double> ts_exp3_input = RepeatPattern({-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0}, 33);
  const std::vector<double> ts_neg1_inplace_input = {1.0, 2.0, -4.0, 0.0, 0.5, -0.25, 8.0, -16.0};
  const std::vector<double> ts_i32_input = MakeRangeDouble(0, 12);

  const std::vector<double> st_f32_exp_base2 = {0.0, 1.0, 2.0, 3.0, -1.0, 0.5, -0.5};
  const std::vector<double> st_f32_exp_base_neg2 = {2.0, 3.0, 0.5, -1.0, 4.0, 1.0, 0.0, 2.0};
  const std::vector<double> st_i16_exp_base2 = {0, 1, 2, 3, 4, 1, 0, 2, 3};
  const std::vector<double> st_i8_exp_base2 = {0, 1, 2, 3, 4, 5};

  const std::vector<double> tt_f32_base = {0.0, 1.0, 4.0};
  const std::vector<double> tt_f32_exp = {0.5, 2.0, -1.0, 0.0, 3.0, 1.0};
  const std::vector<double> tt_f16_base = {0.5, 1.0, 2.0, 4.0, 8.0, 3.0, 1.5, 0.25};
  const std::vector<double> tt_f16_exp = {2.0, 3.0, 0.5, -1.0, 1.0, 2.0, 3.0, 0.0};
  const std::vector<double> tt_bf16_base = {1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, 2.5};
  const std::vector<double> tt_bf16_exp = {0.0, 1.0, 2.0, 3.0, -1.0, 0.5, 2.0, 1.0};
  const std::vector<double> tt_i32_base = {1, 2, 3, 4, 5, 2, 3, 1};
  const std::vector<double> tt_i32_exp = {0, 1, 2, 3, 1, 4, 0, 5};
  const std::vector<double> tt_i16_base = {1, 2, 3, 4, 2, 1, 5, 3};
  const std::vector<double> tt_i16_exp = {1, 2, 1, 0, 3, 4, 1, 2};
  const std::vector<double> tt_i8_base = {1, 2, 3, 4, 2, 1, 5, 3};
  const std::vector<double> tt_i8_exp = {1, 2, 1, 0, 3, 4, 1, 2};
  const std::vector<double> tt_u8_base = {1, 2, 3, 4, 5, 6, 2, 3};
  const std::vector<double> tt_u8_exp = {0, 1, 2, 1, 2, 1, 3, 2};
  const std::vector<double> tt_inplace_f32_base = {1.0, 2.0, 4.0, 9.0, 16.0, 0.5, -2.0, -4.0, 0.0};
  const std::vector<double> tt_inplace_f32_exp = {0.0, 1.0, 0.5, -1.0, 2.0, 3.0, 2.0, 3.0, 0.0};

  const std::vector<double> exp2_f32_input = {-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5, -0.5};
  const std::vector<double> exp2_f16_inplace_input = {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

  const std::vector<double> ts_f16_large_odd = RepeatPattern({-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0}, 4097);
  const std::vector<double> ts_bf16_half_input = {0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 0.25, 2.25};
  const std::vector<double> st_bf16_exp_base2 = {0.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0};
  const std::vector<double> ts_u8_exp2_input = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

  const std::vector<double> empty_f32_input = {};
  const std::vector<double> nchw_f32_input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  const std::vector<double> aligned16_f16_input = RepeatPattern({0.0, 1.0, 2.0, 3.0}, 16);
  const std::vector<double> tail17_f16_input = RepeatPattern({0.0, 1.0, 2.0, 3.0}, 17);

  // -------------------- positive cases --------------------
  Record("TensorScalar_F32_Exp0",
         RunPowTensorScalarCase(stream, "TensorScalar_F32_Exp0",
                                ACL_FLOAT, {17}, ts_exp0_input,
                                0.0, ACL_FLOAT, false));

  Record("TensorScalar_F32_Exp1",
         RunPowTensorScalarCase(stream, "TensorScalar_F32_Exp1",
                                ACL_FLOAT, {2, 3, 3}, ts_exp1_input,
                                1.0, ACL_FLOAT, false));

  Record("TensorScalar_F32_ExpHalf",
         RunPowTensorScalarCase(stream, "TensorScalar_F32_ExpHalf",
                                ACL_FLOAT, {9}, ts_exphalf_input,
                                0.5, ACL_FLOAT, false));

  Record("TensorScalar_F32_Exp4",
         RunPowTensorScalarCase(stream, "TensorScalar_F32_Exp4",
                                ACL_FLOAT, {5, 7}, ts_exp2_input,
                                4.0, ACL_FLOAT, false));

  Record("TensorScalar_F32_Exp3",
         RunPowTensorScalarCase(stream, "TensorScalar_F32_Exp3",
                                ACL_FLOAT, {33}, ts_exp3_input,
                                3.0, ACL_FLOAT, false));

  Record("InplaceTensorScalar_F32_ExpNeg1",
         RunPowTensorScalarCase(stream, "InplaceTensorScalar_F32_ExpNeg1",
                                ACL_FLOAT, {2, 4}, ts_neg1_inplace_input,
                                -1.0, ACL_FLOAT, true));

  Record("TensorScalar_I32_Exp3",
         RunPowTensorScalarCase(stream, "TensorScalar_I32_Exp3",
                                ACL_INT32, {13}, ts_i32_input,
                                3.0, ACL_INT32, false));

  Record("TensorScalar_F16_LargeOdd_Exp3",
         RunPowTensorScalarCase(stream, "TensorScalar_F16_LargeOdd_Exp3",
                                ACL_FLOAT16, {4097}, ts_f16_large_odd,
                                3.0, ACL_FLOAT, false));

  Record("TensorScalar_BF16_ExpHalf",
         RunPowTensorScalarCase(stream, "TensorScalar_BF16_ExpHalf",
                                ACL_BF16, {8}, ts_bf16_half_input,
                                0.5, ACL_FLOAT, false));

  Record("TensorScalar_U8_Exp2",
         RunPowTensorScalarCase(stream, "TensorScalar_U8_Exp2",
                                ACL_UINT8, {9}, ts_u8_exp2_input,
                                2.0, ACL_UINT8, false));

  Record("TensorScalar_F32_Empty_Exp2",
         RunPowTensorScalarCase(stream, "TensorScalar_F32_Empty_Exp2",
                                ACL_FLOAT, {0, 4}, empty_f32_input,
                                2.0, ACL_FLOAT, false));

  Record("TensorScalar_F32_NCHW_Exp3",
         RunPowTensorScalarCaseWithFormat(stream, "TensorScalar_F32_NCHW_Exp3",
                                          ACL_FLOAT, {1, 2, 2, 3}, nchw_f32_input,
                                          3.0, ACL_FLOAT,
                                          ACL_FORMAT_NCHW, ACL_FORMAT_NCHW));

  Record("TensorScalar_F32_Int32ScalarExp3",
         RunPowTensorScalarCase(stream, "TensorScalar_F32_Int32ScalarExp3",
                                ACL_FLOAT, {9}, ts_u8_exp2_input,
                                3.0, ACL_INT32, false));

  Record("TensorScalar_F32_Int8ScalarExp3",
         RunPowTensorScalarCase(stream, "TensorScalar_F32_Int8ScalarExp3",
                                ACL_FLOAT, {9}, ts_u8_exp2_input,
                                3.0, ACL_INT8, false));

  Record("TensorScalar_F32_Int16ScalarExp3",
         RunPowTensorScalarCase(stream, "TensorScalar_F32_Int16ScalarExp3",
                                ACL_FLOAT, {9}, ts_u8_exp2_input,
                                3.0, ACL_INT16, false));

  Record("TensorScalar_F32_Int64ScalarExp3",
         RunPowTensorScalarCase(stream, "TensorScalar_F32_Int64ScalarExp3",
                                ACL_FLOAT, {9}, ts_u8_exp2_input,
                                3.0, ACL_INT64, false));

  Record("TensorScalar_F16_Aligned16_Exp3",
         RunPowTensorScalarCase(stream, "TensorScalar_F16_Aligned16_Exp3",
                                ACL_FLOAT16, {16}, aligned16_f16_input,
                                3.0, ACL_FLOAT, false));

  Record("TensorScalar_F16_Tail17_Exp3",
         RunPowTensorScalarCase(stream, "TensorScalar_F16_Tail17_Exp3",
                                ACL_FLOAT16, {17}, tail17_f16_input,
                                3.0, ACL_FLOAT, false));

  Record("ScalarTensor_F32_Base2",
         RunPowScalarTensorCase(stream, "ScalarTensor_F32_Base2",
                                ACL_FLOAT, {7}, st_f32_exp_base2,
                                2.0, ACL_FLOAT));

  Record("ScalarTensor_F32_Base1_FillOne",
         RunPowScalarTensorCase(stream, "ScalarTensor_F32_Base1_FillOne",
                                ACL_FLOAT, {7}, st_f32_exp_base2,
                                1.0, ACL_FLOAT));

  Record("ScalarTensor_F32_BaseNeg2",
         RunPowScalarTensorCase(stream, "ScalarTensor_F32_BaseNeg2",
                                ACL_FLOAT, {2, 4}, st_f32_exp_base_neg2,
                                -2.0, ACL_FLOAT));

  Record("ScalarTensor_I16_Base2",
         RunPowScalarTensorCase(stream, "ScalarTensor_I16_Base2",
                                ACL_INT16, {9}, st_i16_exp_base2,
                                2.0, ACL_INT16));

  Record("ScalarTensor_I8_Base2",
         RunPowScalarTensorCase(stream, "ScalarTensor_I8_Base2",
                                ACL_INT8, {6}, st_i8_exp_base2,
                                2.0, ACL_INT8));

  Record("ScalarTensor_BF16_Base2",
         RunPowScalarTensorCase(stream, "ScalarTensor_BF16_Base2",
                                ACL_BF16, {8}, st_bf16_exp_base2,
                                2.0, ACL_FLOAT));

  Record("ScalarTensor_F32_Empty_Base2",
         RunPowScalarTensorCase(stream, "ScalarTensor_F32_Empty_Base2",
                                ACL_FLOAT, {0, 4}, empty_f32_input,
                                2.0, ACL_FLOAT));

  Record("TensorTensor_F32_BroadcastBase",
         RunPowTensorTensorCase(stream, "TensorTensor_F32_BroadcastBase",
                                ACL_FLOAT, {1, 3}, tt_f32_base,
                                {2, 3}, tt_f32_exp, false));

  Record("TensorTensor_F16",
         RunPowTensorTensorCase(stream, "TensorTensor_F16",
                                ACL_FLOAT16, {2, 4}, tt_f16_base,
                                {2, 4}, tt_f16_exp, false));

  Record("TensorTensor_BF16",
         RunPowTensorTensorCase(stream, "TensorTensor_BF16",
                                ACL_BF16, {2, 4}, tt_bf16_base,
                                {2, 4}, tt_bf16_exp, false));

  Record("TensorTensor_I32",
         RunPowTensorTensorCase(stream, "TensorTensor_I32",
                                ACL_INT32, {8}, tt_i32_base,
                                {8}, tt_i32_exp, false));

  Record("TensorTensor_I16",
         RunPowTensorTensorCase(stream, "TensorTensor_I16",
                                ACL_INT16, {8}, tt_i16_base,
                                {8}, tt_i16_exp, false));

  Record("TensorTensor_I8",
         RunPowTensorTensorCase(stream, "TensorTensor_I8",
                                ACL_INT8, {8}, tt_i8_base,
                                {8}, tt_i8_exp, false));

  Record("TensorTensor_U8",
         RunPowTensorTensorCase(stream, "TensorTensor_U8",
                                ACL_UINT8, {8}, tt_u8_base,
                                {8}, tt_u8_exp, false));

  Record("InplaceTensorTensor_F32",
         RunPowTensorTensorCase(stream, "InplaceTensorTensor_F32",
                                ACL_FLOAT, {9}, tt_inplace_f32_base,
                                {9}, tt_inplace_f32_exp, true));

  Record("Exp2_F32",
         RunExp2Case(stream, "Exp2_F32",
                     ACL_FLOAT, {11}, exp2_f32_input, false));

  Record("InplaceExp2_F16",
         RunExp2Case(stream, "InplaceExp2_F16",
                     ACL_FLOAT16, {9}, exp2_f16_inplace_input, true));

  // -------------------- negative cases --------------------
  Record("Negative_TensorScalar_NullSelf",
         NegativePowTensorScalarNullSelf("Negative_TensorScalar_NullSelf"));

  Record("Negative_TensorScalar_NullExponent",
         NegativePowTensorScalarNullExponent("Negative_TensorScalar_NullExponent"));

  Record("Negative_TensorScalar_NullOut",
         NegativePowTensorScalarNullOut("Negative_TensorScalar_NullOut"));

  Record("Negative_InplaceTensorScalar_NullSelf",
         NegativeInplacePowTensorScalarNullSelf("Negative_InplaceTensorScalar_NullSelf"));

    Record("Negative_InplaceTensorScalar_NullExponent",
         NegativeInplacePowTensorScalarNullExponent("Negative_InplaceTensorScalar_NullExponent"));

  Record("Negative_TensorScalar_OutputShapeMismatch",
         NegativePowTensorScalarOutputShapeMismatch("Negative_TensorScalar_OutputShapeMismatch"));

  Record("Negative_TensorScalar_OutputDtypeMismatch",
         NegativePowTensorScalarOutputDtypeMismatch("Negative_TensorScalar_OutputDtypeMismatch"));

  Record("Negative_TensorScalar_UnsupportedDouble",
         NegativePowTensorScalarUnsupportedDouble("Negative_TensorScalar_UnsupportedDouble"));

  Record("Negative_TensorScalar_FormatMismatch",
         NegativePowTensorScalarFormatMismatch("Negative_TensorScalar_FormatMismatch"));

  Record("Negative_TensorScalar_IntegralBaseNegativeExponent",
         NegativePowTensorScalarIntegralBaseNegativeExponent("Negative_TensorScalar_IntegralBaseNegativeExponent"));

  Record("Negative_ScalarTensor_NullScalar",
         NegativePowScalarTensorNullScalar("Negative_ScalarTensor_NullScalar"));

  Record("Negative_ScalarTensor_NullExponentTensor",
         NegativePowScalarTensorNullExponentTensor("Negative_ScalarTensor_NullExponentTensor"));

  Record("Negative_ScalarTensor_NullOut",
         NegativePowScalarTensorNullOut("Negative_ScalarTensor_NullOut"));

  Record("Negative_ScalarTensor_OutputDtypeMismatch",
         NegativePowScalarTensorOutputDtypeMismatch("Negative_ScalarTensor_OutputDtypeMismatch"));

  Record("Negative_ScalarTensor_OutputShapeMismatch",
         NegativePowScalarTensorOutputShapeMismatch("Negative_ScalarTensor_OutputShapeMismatch"));

  Record("Negative_ScalarTensor_UnsupportedDouble",
         NegativePowScalarTensorUnsupportedDouble("Negative_ScalarTensor_UnsupportedDouble"));

  Record("Negative_TensorTensor_BroadcastMismatch",
         NegativePowTensorTensorBroadcastMismatch("Negative_TensorTensor_BroadcastMismatch"));

  Record("Negative_TensorTensor_UnsupportedDouble",
         NegativePowTensorTensorUnsupportedDouble("Negative_TensorTensor_UnsupportedDouble"));

  Record("Negative_TensorTensor_OutputDtypeMismatch",
         NegativePowTensorTensorOutputDtypeMismatch("Negative_TensorTensor_OutputDtypeMismatch"));

  Record("Negative_Exp2_NullOut",
         NegativeExp2NullOut("Negative_Exp2_NullOut"));

  // -------------------- summary --------------------
  LOG_PRINT("============================================================\n");
  LOG_PRINT("Pow operator test summary\n");
  LOG_PRINT("Total: %d, Passed: %d, Failed: %d\n", passed + failed, passed, failed);
  if (!failedCases.empty()) {
    LOG_PRINT("Failed cases:\n");
    for (size_t i = 0; i < failedCases.size(); ++i) {
      LOG_PRINT("  - %s\n", failedCases[i].c_str());
    }
  }
  LOG_PRINT("============================================================\n");

  int exitCode = (failed == 0) ? 0 : 1;

  auto ret = aclrtDestroyStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[FAIL] aclrtDestroyStream failed, ret=%d\n", ret);
    exitCode = 1;
  }
  ret = aclrtResetDevice(deviceId);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[FAIL] aclrtResetDevice failed, ret=%d\n", ret);
    exitCode = 1;
  }
  ret = aclFinalize();
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[FAIL] aclFinalize failed, ret=%d\n", ret);
    exitCode = 1;
  }

  return exitCode;
}